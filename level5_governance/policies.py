"""Policy enforcement adapter for Level 5 governance.

Thin wrapper that delegates to level5_policy/policy_engine. Leakage detection
is invoked directly by the gatekeeper (level5_policy/leakage_detector).
"""

from typing import Any

from intent.schema import IntentSchema
from level5_policy.policy_engine import PolicyEngine


class PolicyEnforcer:
    """Thin adapter to the policy engine."""

    def __init__(self, intent: IntentSchema):
        self.intent = intent
        self._policy_engine = PolicyEngine(intent)

    def enforce(self, pipeline_output: dict[str, Any]) -> list[str]:
        """Enforce policies and return violation messages (empty if none)."""
        return self._policy_engine.check_violations(pipeline_output)
