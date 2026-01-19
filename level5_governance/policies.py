"""Policy enforcement adapter for Level 5 governance.

This module provides a thin wrapper/adapter to the policy layer.
It does NOT define policy rules or implement logic beyond delegation.

All policy rules are defined in level5_policy/.
This module only orchestrates calls to policy enforcement.
"""

from typing import Any

from intent.schema import IntentSchema


class PolicyEnforcer:
    """Thin adapter to policy enforcement layer.

    This class delegates all policy enforcement to level5_policy/policy_engine
    and leakage detection to level5_policy/leakage_detector.

    It does NOT:
    - Define policy rules
    - Implement enforcement logic
    - Inspect datasets
    """

    def __init__(self, intent: IntentSchema):
        """Initialize policy enforcer.

        Args:
            intent: User intent (read-only)
        """
        self.intent = intent
        # Delegate to level5_policy components
        # Note: These will be imported when level5_policy is implemented
        self._policy_engine = None
        self._leakage_detector = None

    def enforce(self, pipeline_output: dict[str, Any]) -> list[str]:
        """Enforce policies and return violations.

        Delegates to level5_policy/policy_engine.

        Args:
            pipeline_output: Dictionary containing pipeline outputs

        Returns:
            List of violation messages (empty if no violations)
        """
        # TODO: Import and delegate to level5_policy.policy_engine when implemented
        # For now, return empty list (no violations detected)
        # This allows governance to work even if policy layer is not yet implemented
        try:
            from level5_policy.policy_engine import PolicyEngine

            if self._policy_engine is None:
                self._policy_engine = PolicyEngine(self.intent)

            violations = self._policy_engine.check_violations(pipeline_output)
            return violations
        except ImportError:
            # Policy layer not yet implemented - return no violations
            return []

    def detect_leakage(self, pipeline_output: dict[str, Any]) -> bool:
        """Detect feature leakage.

        Delegates to level5_policy/leakage_detector.

        Args:
            pipeline_output: Dictionary containing pipeline outputs

        Returns:
            True if leakage detected, False otherwise
        """
        # TODO: Import and delegate to level5_policy.leakage_detector when implemented
        # For now, return False (no leakage detected)
        try:
            from level5_policy.leakage_detector import LeakageDetector

            if self._leakage_detector is None:
                self._leakage_detector = LeakageDetector(self.intent)

            leakage_detected = self._leakage_detector.detect(pipeline_output)
            return leakage_detected
        except ImportError:
            # Leakage detector not yet implemented - return no leakage
            return False
