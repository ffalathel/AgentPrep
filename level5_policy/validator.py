"""Validator for Level 5 policy enforcement.

This module implements enforcement logic that compares pipeline metadata
against policy rules. It returns explicit violations.

Validators check, they do not fix.
"""

from dataclasses import dataclass
from typing import Any

from intent.schema import IntentSchema


@dataclass
class ValidationResult:
    """Result of policy validation.

    This is an explicit result object - no logging as substitute.
    """

    passed: bool
    violations: list[str]

    def __str__(self) -> str:
        """Human-readable representation."""
        if self.passed:
            return "Validation PASSED: No violations"
        return f"Validation FAILED: {len(self.violations)} violation(s)"


class ConstraintValidator:
    """Validates pipeline outputs against policy constraints.

    This class compares pipeline metadata against policy rules and
    returns explicit violations. It is deterministic and auditable.
    """

    def __init__(self, intent: IntentSchema):
        """Initialize constraint validator.

        Args:
            intent: User intent (read-only)
        """
        self.intent = intent

    def validate(self, pipeline_output: dict[str, Any]) -> list[str]:
        """Validate pipeline output against constraints.

        This method checks all constraints and returns violations.
        It does NOT modify data or suggest fixes.

        Args:
            pipeline_output: Dictionary containing pipeline outputs:
                - feature_dataframe: Final feature DataFrame
                - target_column: Normalized target column name
                - feature_provenance: List of feature provenance records
                - schema_metadata: Schema metadata from Level 1
                - metadata_path: Path to metadata file (if exists)
                - original_column_count: Number of columns in original dataset

        Returns:
            List of violation messages (empty if no violations)
        """
        violations: list[str] = []

        # Validate feature count
        feature_violation = self._validate_feature_count(pipeline_output)
        if feature_violation:
            violations.append(feature_violation)

        # Validate interaction count
        interaction_violation = self._validate_interaction_count(pipeline_output)
        if interaction_violation:
            violations.append(interaction_violation)

        # Validate target column presence
        target_presence_violation = self._validate_target_presence(pipeline_output)
        if target_presence_violation:
            violations.append(target_presence_violation)

        # Validate metadata completeness
        metadata_violation = self._validate_metadata_completeness(pipeline_output)
        if metadata_violation:
            violations.append(metadata_violation)

        return violations

    def _validate_feature_count(self, pipeline_output: dict[str, Any]) -> str:
        """Validate feature count constraint.

        Returns:
            Violation message if violated, empty string otherwise
        """
        feature_dataframe = pipeline_output.get("feature_dataframe")
        target_column = pipeline_output.get("target_column")

        if feature_dataframe is None or target_column is None:
            return ""  # Cannot validate

        feature_count = len([col for col in feature_dataframe.columns if col != target_column])
        max_features = self.intent.constraints.max_features

        if feature_count > max_features:
            return f"Feature count ({feature_count}) exceeds maximum ({max_features})"

        return ""

    def _validate_interaction_count(self, pipeline_output: dict[str, Any]) -> str:
        """Validate interaction count constraint.

        Returns:
            Violation message if violated, empty string otherwise
        """
        feature_provenance = pipeline_output.get("feature_provenance")
        if feature_provenance is None:
            return ""  # Cannot validate

        interaction_count = sum(1 for prov in feature_provenance if len(prov.source_columns) > 1)
        max_interactions = self.intent.constraints.max_interactions

        if interaction_count > max_interactions:
            return f"Interaction count ({interaction_count}) exceeds maximum ({max_interactions})"

        return ""

    def _validate_target_presence(self, pipeline_output: dict[str, Any]) -> str:
        """Validate that target column is present.

        Returns:
            Violation message if violated, empty string otherwise
        """
        feature_dataframe = pipeline_output.get("feature_dataframe")
        target_column = pipeline_output.get("target_column")

        if feature_dataframe is None or target_column is None:
            return ""  # Cannot validate

        if target_column not in feature_dataframe.columns:
            return f"Target column '{target_column}' not found in final dataset"

        return ""

    def _validate_metadata_completeness(self, pipeline_output: dict[str, Any]) -> str:
        """Validate that metadata is complete.

        Returns:
            Violation message if violated, empty string otherwise
        """
        missing = []

        if pipeline_output.get("metadata_path") is None:
            missing.append("metadata file")

        if pipeline_output.get("schema_metadata") is None:
            missing.append("schema metadata")

        if pipeline_output.get("feature_provenance") is None:
            missing.append("feature provenance")

        if missing:
            return f"Missing metadata: {', '.join(missing)}"

        return ""
