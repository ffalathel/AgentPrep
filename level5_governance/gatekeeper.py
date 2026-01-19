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


@dataclass
class GovernanceDecision:
    """Final governance decision.

    This is the explicit output of governance - a clear PASS/FAIL decision
    with explainable reasons.
    """

    approved: bool
    violations: list[str]
    leakage_detected: bool
    reason: str  # One-sentence explanation

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

        # Step 1: Invoke policy enforcement
        policy_violations = self.policy_enforcer.enforce(pipeline_output)
        violations.extend(policy_violations)

        # Step 2: Invoke leakage detection
        leakage_detected = self.policy_enforcer.detect_leakage(pipeline_output)
        if leakage_detected:
            violations.append("Feature leakage detected")

        # Step 3: Invoke validators
        validator_violations = self.validator_orchestrator.validate(pipeline_output)
        violations.extend(validator_violations)

        # Step 4: Aggregate and decide
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
            leakage_detected=leakage_detected,
            reason=reason,
        )
