"""Policy engine for Level 5 policy definition and enforcement.

This module defines declarative policy rules. It should feel like configuration,
not logic. All rules are explicit, documented, and deterministic.

Policies are facts, not opinions.
"""

from dataclasses import dataclass
from typing import Any

from intent.schema import IntentSchema, InterpretabilityPriority, ModelFamily


@dataclass
class PolicyRule:
    """A single policy rule.

    This is a declarative statement of what is allowed or forbidden.
    """

    name: str
    description: str
    check: callable  # Function that checks if rule is violated
    severity: str = "critical"  # "critical" or "warning"


class PolicyEngine:
    """Declarative policy engine.

    This class defines allowed behaviors as explicit rules.
    It does NOT make decisions - it only defines what to check.
    """

    def __init__(self, intent: IntentSchema):
        """Initialize policy engine with user intent.

        Args:
            intent: User intent (read-only)
        """
        self.intent = intent
        self.rules: list[PolicyRule] = []

        # Define all policy rules declaratively
        self._define_rules()

    def _define_rules(self) -> None:
        """Define all policy rules.

        This method should feel like configuration, not logic.
        Each rule is a declarative statement of what is allowed.
        """
        # Rule 1: Feature count limit
        self.rules.append(
            PolicyRule(
                name="max_features_limit",
                description=f"Total feature count must not exceed {self.intent.constraints.max_features}",
                check=self._check_max_features,
            )
        )

        # Rule 2: Interaction count limit
        self.rules.append(
            PolicyRule(
                name="max_interactions_limit",
                description=f"Total interaction features must not exceed {self.intent.constraints.max_interactions}",
                check=self._check_max_interactions,
            )
        )

        # Rule 3: Target column must be present
        self.rules.append(
            PolicyRule(
                name="target_column_present",
                description=f"Target column '{self.intent.task.target_column}' must be present",
                check=self._check_target_present,
            )
        )

        # Rule 4: Target column must not be used as input
        self.rules.append(
            PolicyRule(
                name="target_column_not_used_as_input",
                description="Target column must never be used as input to any feature",
                check=self._check_target_not_used,
            )
        )

        # Rule 5: Interpretability requirements (if high priority)
        if self.intent.preferences.interpretability_priority == InterpretabilityPriority.HIGH:
            self.rules.append(
                PolicyRule(
                    name="interpretability_high",
                    description="Features must maintain high interpretability",
                    check=self._check_interpretability,
                )
            )

        # Rule 6: Model family constraints
        self.rules.append(
            PolicyRule(
                name="model_family_compatibility",
                description=f"Features must be compatible with {self.intent.model.family} model family",
                check=self._check_model_family,
                severity="warning",  # Warning, not critical
            )
        )

        # Rule 7: Column dropping (if not allowed)
        if not self.intent.preferences.allow_column_dropping:
            self.rules.append(
                PolicyRule(
                    name="no_column_dropping",
                    description="Columns may not be dropped",
                    check=self._check_no_column_dropping,
                )
            )

    def _check_max_features(self, pipeline_output: dict[str, Any]) -> tuple[bool, str]:
        """Check feature count limit.

        Returns:
            Tuple of (is_violated, violation_message)
        """
        feature_dataframe = pipeline_output.get("feature_dataframe")
        target_column = pipeline_output.get("target_column")

        if feature_dataframe is None or target_column is None:
            return False, ""  # Cannot check, assume pass

        feature_count = len([col for col in feature_dataframe.columns if col != target_column])
        max_features = self.intent.constraints.max_features

        if feature_count > max_features:
            return True, f"Feature count ({feature_count}) exceeds maximum ({max_features})"

        return False, ""

    def _check_max_interactions(self, pipeline_output: dict[str, Any]) -> tuple[bool, str]:
        """Check interaction count limit.

        Returns:
            Tuple of (is_violated, violation_message)
        """
        feature_provenance = pipeline_output.get("feature_provenance", [])

        interaction_count = sum(1 for prov in feature_provenance if len(prov.source_columns) > 1)
        max_interactions = self.intent.constraints.max_interactions

        if interaction_count > max_interactions:
            return True, f"Interaction count ({interaction_count}) exceeds maximum ({max_interactions})"

        return False, ""

    def _check_target_present(self, pipeline_output: dict[str, Any]) -> tuple[bool, str]:
        """Check that target column is present.

        Returns:
            Tuple of (is_violated, violation_message)
        """
        feature_dataframe = pipeline_output.get("feature_dataframe")
        target_column = pipeline_output.get("target_column")

        if feature_dataframe is None or target_column is None:
            return False, ""  # Cannot check, assume pass

        if target_column not in feature_dataframe.columns:
            return True, f"Target column '{target_column}' not found in final dataset"

        return False, ""

    def _check_target_not_used(self, pipeline_output: dict[str, Any]) -> tuple[bool, str]:
        """Check that target column is not used as input.

        Returns:
            Tuple of (is_violated, violation_message)
        """
        feature_provenance = pipeline_output.get("feature_provenance", [])
        target_column = pipeline_output.get("target_column")

        if target_column is None:
            return False, ""  # Cannot check, assume pass

        for prov in feature_provenance:
            if target_column in prov.source_columns:
                return True, f"Target column '{target_column}' used as input to feature '{prov.feature_name}'"

        return False, ""

    def _check_interpretability(self, pipeline_output: dict[str, Any]) -> tuple[bool, str]:
        """Check interpretability requirements.

        Returns:
            Tuple of (is_violated, violation_message)
        """
        if self.intent.preferences.interpretability_priority != InterpretabilityPriority.HIGH:
            return False, ""  # Only check if high priority

        feature_provenance = pipeline_output.get("feature_provenance", [])

        from level4_feature.feature_catalog import get_catalog

        catalog = get_catalog()
        non_interpretable = []

        for prov in feature_provenance:
            trans = catalog.get(prov.transformation)
            if trans and trans.interpretability_impact == "reduces":
                non_interpretable.append(prov.feature_name)

        if non_interpretable:
            return True, f"Features with reduced interpretability: {', '.join(non_interpretable)}"

        return False, ""

    def _check_model_family(self, pipeline_output: dict[str, Any]) -> tuple[bool, str]:
        """Check model family compatibility.

        Returns:
            Tuple of (is_violated, violation_message)
        """
        # Model family compatibility is a warning, not a hard constraint
        # For now, we always pass (can be extended with specific rules)
        return False, ""

    def _check_no_column_dropping(self, pipeline_output: dict[str, Any]) -> tuple[bool, str]:
        """Check that columns were not dropped.

        Excludes columns dropped for leakage remediation (required for data integrity).

        Returns:
            Tuple of (is_violated, violation_message)
        """
        original_column_count = pipeline_output.get("original_column_count")
        feature_dataframe = pipeline_output.get("feature_dataframe")
        leakage_dropped_columns = pipeline_output.get("leakage_dropped_columns", [])

        if original_column_count is None or feature_dataframe is None:
            return False, ""  # Cannot check, assume pass

        final_column_count = len(feature_dataframe.columns)
        leakage_drop_count = len(leakage_dropped_columns)

        # Calculate net column drop (excluding leakage drops)
        net_dropped = original_column_count - final_column_count - leakage_drop_count

        if net_dropped > 0:
            return True, f"Columns were dropped ({original_column_count} → {final_column_count})"

        return False, ""

    def check_violations(self, pipeline_output: dict[str, Any]) -> list[str]:
        """Check all policy rules and return violations.

        Args:
            pipeline_output: Dictionary containing pipeline outputs

        Returns:
            List of violation messages (empty if no violations)
        """
        violations: list[str] = []

        for rule in self.rules:
            is_violated, message = rule.check(pipeline_output)
            if is_violated:
                violations.append(f"{rule.name}: {message}")

        return violations
