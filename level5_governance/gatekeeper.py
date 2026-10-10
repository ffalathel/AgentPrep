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
from typing import Any

from intent.schema import IntentSchema
from level5_policy.leakage_detector import (
    LeakageDetector,
    is_classification,
    single_column_score,
)

from .policies import PolicyEnforcer
from .validators import ValidatorOrchestrator


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
        self.leakage_detector = LeakageDetector(intent)

    def _suspected_leak_warnings(
        self, pipeline_output: dict[str, Any], confirmed: dict[str, str]
    ) -> list[str]:
        """Surface LLM-nominated semantic leaks with their measured single-column score.

        The LLM's claim alone never drops a column; data-confirmed leaks are already
        violations. These are for the user to decide.
        """
        df = pipeline_output.get("feature_dataframe")
        target = pipeline_output.get("target_column")
        warnings = []
        for suspect in pipeline_output.get("suspected_leaks", []):
            col = suspect["column"]
            if df is None or target not in df.columns or col not in df.columns or col == target or col in confirmed:
                continue
            y = df.loc[df[target].notna(), target]
            score = single_column_score(df.loc[y.index, col], y, is_classification(self.intent, y))
            warnings.append(
                f"Suspected leak '{col}' (LLM): {suspect['reason']} "
                f"[single-column score {score:.2f}]. Not removed; confirm it is known at prediction time."
            )
        return warnings

    def _joint_leak_warnings(self, pipeline_output: dict[str, Any]) -> list[str]:
        """Pairs that leak together but that no evidence pins on one column; the user decides."""
        df, source = pipeline_output.get("feature_dataframe"), pipeline_output.get("source_dataframe")
        warnings = []
        for a, b, score in self.leakage_detector.joint_leaks:
            if df is not None and source is not None and any(c in source.columns and c not in df.columns for c in (a, b)):
                continue  # one of the pair was removed, so the pair no longer leaks
            warnings.append(
                f"Possible joint leak: '{a}' and '{b}' together predict the target (held-out score {score:.2f}); "
                "neither does alone. Check whether either is recorded after the outcome."
            )
        return warnings

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

        # Step 2: Invoke leakage detection (deterministic, data-driven)
        leaks = self.leakage_detector.find_leaks(pipeline_output)
        leakage_detected = bool(leaks)
        if leakage_detected:
            violations.append(
                "Feature leakage detected: "
                + "; ".join(f"'{col}' {reason}" for col, reason in leaks.items())
            )
            remediation_info["leakage"] = {"features_to_remove": sorted(leaks)}

        # Step 3: Invoke validators
        validator_violations, validator_warnings, validator_remediation_info = (
            self.validator_orchestrator.validate(pipeline_output)
        )
        violations.extend(validator_violations)
        warnings = list(validator_warnings)  # Collect warnings separately
        warnings.extend(self._suspected_leak_warnings(pipeline_output, leaks))
        warnings.extend(self._joint_leak_warnings(pipeline_output))
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
                reason = "The pipeline violated policy due to feature leakage"
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
