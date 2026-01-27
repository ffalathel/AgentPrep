"""Gatekeeper for Level 5 governance orchestration.

This module provides the final authority that decides whether pipeline output
is allowed to exist. It orchestrates policy enforcement, leakage detection,
and validation, then produces an explicit PASS/FAIL decision.

Governance does NOT:
- Define policy rules
- Inspect datasets
- Modify metadata
- Contain heuristics
- Auto-fix violations

Governance DOES:
- Invoke policy enforcement
- Invoke leakage detection
- Aggregate violations
- Produce final decision
"""

from dataclasses import dataclass
from typing import Any, Optional

from intent.schema import IntentSchema

from .policies import PolicyEnforcer
from .validators import ValidatorOrchestrator
import time

@dataclass
class GovernanceDecision:
    """Final governance decision.

    This is the explicit output of governance - a clear PASS/FAIL decision
    with explainable reasons.
    """

    approved: bool
    violations: list[str]  # Blocking violations that cause rejection
    warnings: list[str] = None  # Informational warnings that don't block
    leakage_detected: bool = False
    reason: str = ""  # One-sentence explanation
    remediation_info: dict[str, Any] = None  # Information for auto-remediation

    def __post_init__(self):
        """Initialize remediation_info and warnings if not provided."""
        if self.remediation_info is None:
            self.remediation_info = {}
        if self.warnings is None:
            self.warnings = []

    def __str__(self) -> str:
        """Human-readable representation."""
        status = "APPROVED" if self.approved else "REJECTED"
        return f"Governance {status}: {self.reason}"


class GovernanceGatekeeper:
    """Final authority for pipeline governance.

    This class orchestrates all governance checks and produces a final decision.
    It does not implement rules or inspect data - it only orchestrates.
    """

    def __init__(self, intent: IntentSchema):
        """Initialize the gatekeeper.

        Args:
            intent: User intent (read-only)
        """
        self.intent = intent
        self.policy_enforcer = PolicyEnforcer(intent)
        self.validator_orchestrator = ValidatorOrchestrator(intent)

    def _find_leaking_features(self, pipeline_output: dict[str, Any]) -> list[str]:
        """Identify features that likely cause leakage.

        Uses simple deterministic rules aligned with the leakage detector:
        - Target column used as input to a feature
        - Feature names containing suspicious target-like patterns
        """
        feature_provenance = pipeline_output.get("feature_provenance", [])
        target_column = pipeline_output.get("target_column")
        feature_dataframe = pipeline_output.get("feature_dataframe")

        if not feature_provenance or not target_column:
            return []

        leaking: set[str] = set()
        target = target_column

        # 1) Target used as input (direct leakage)
        for prov in feature_provenance:
            source_cols = getattr(prov, "source_columns", [])
            feature_name = getattr(prov, "feature_name", None)
            if feature_name is None:
                continue
            if target in source_cols:
                leaking.add(feature_name)

        # 2) Suspicious feature names (proxy leakage)
        suspicious_patterns = [
            target.lower(),
            "target",
            "label",
            "y_",
            "_target",
            "_label",
            f"{target}_",
            f"_{target}",
        ]

        for prov in feature_provenance:
            feature_name = getattr(prov, "feature_name", None)
            if not feature_name:
                continue
            col_lower = feature_name.lower()
            if any(pattern in col_lower for pattern in suspicious_patterns):
                leaking.add(feature_name)

        # Only keep leaking features that still exist in the current dataframe
        if feature_dataframe is not None:
            df_cols_lower = {c.lower(): c for c in feature_dataframe.columns}
            filtered = []
            for feat in leaking:
                if feat is None:
                    continue
                if feat in feature_dataframe.columns:
                    filtered.append(feat)
                elif feat.lower() in df_cols_lower:
                    filtered.append(df_cols_lower[feat.lower()])
            result = sorted(set(filtered))
        else:
            result = sorted(leaking)
        return result

    def decide(
        self,
        pipeline_output: dict[str, Any],
    ) -> GovernanceDecision:
        """Make final governance decision.

        This is the single entry point for governance. It:
        1. Invokes policy enforcement
        2. Invokes leakage detection
        3. Invokes validators
        4. Aggregates all violations
        5. Produces final decision

        Args:
            pipeline_output: Dictionary containing all pipeline outputs:
                - feature_dataframe: Final feature DataFrame
                - target_column: Normalized target column name
                - feature_provenance: List of feature provenance records
                - schema_metadata: Schema metadata from Level 1
                - metadata_path: Path to metadata file (if exists)
                - original_column_count: Number of columns in original dataset

        Returns:
            GovernanceDecision with explicit PASS/FAIL and reasons
        """
        violations: list[str] = []
        leakage_detected = False
        remediation_info: dict[str, Any] = {}

        # Step 1: Invoke policy enforcement
        policy_violations = self.policy_enforcer.enforce(pipeline_output)
        violations.extend(policy_violations)

        # Step 2: Invoke leakage detection
        leakage_detected = self.policy_enforcer.detect_leakage(pipeline_output)
        leaking_features = []
        if leakage_detected:
            violations.append("Feature leakage detected")
            # Find specific leaking features for remediation
            leaking_features = self._find_leaking_features(pipeline_output)
            remediation_info["leakage"] = {
                "features_to_remove": leaking_features,
            }

        # Step 3: Invoke validators
        validator_result = self.validator_orchestrator.validate(pipeline_output)
        # Handle both old format (violations, remediation_info) and new format (violations, warnings, remediation_info)
        if len(validator_result) == 2:
            validator_violations, validator_remediation_info = validator_result
            validator_warnings = []
        elif len(validator_result) == 3:
            validator_violations, validator_warnings, validator_remediation_info = validator_result
        else:
            validator_violations = []
            validator_warnings = []
            validator_remediation_info = {}
        
        violations.extend(validator_violations)
        warnings = list(validator_warnings)  # Collect warnings separately
        # Merge validator remediation info with leakage remediation info
        remediation_info.update(validator_remediation_info)

        # Step 4: Aggregate and decide
        # Only reject on blocking violations, not warnings
        approved = len(violations) == 0 and not leakage_detected

        # Step 5: Produce explainable reason
        if approved:
            reason = "Pipeline output complies with all governance policies"
        else:
            if leakage_detected:
                reason = f"The pipeline violated policy due to feature leakage"
            elif violations:
                # Use first violation for one-sentence explanation
                reason = f"The pipeline violated policy: {violations[0]}"
            else:
                reason = "The pipeline violated policy"

        return GovernanceDecision(
            approved=approved,
            violations=violations,
            warnings=warnings,
            leakage_detected=leakage_detected,
            reason=reason,
            remediation_info=remediation_info,
        )
