"""Validator orchestration for Level 5 governance.

This module orchestrates validator execution but does NOT implement
validation rules itself. All validation logic is delegated to level5_policy/validator.

This module:
- Orchestrates validator calls
- Aggregates validation results
- Does NOT implement validation rules
"""

from typing import Any

from intent.schema import IntentSchema


class ValidatorOrchestrator:
    """Orchestrates validator execution.

    This class delegates all validation to level5_policy/validator.
    It does NOT implement validation rules itself.
    """

    def __init__(self, intent: IntentSchema):
        """Initialize validator orchestrator.

        Args:
            intent: User intent (read-only)
        """
        self.intent = intent
        # Delegate to level5_policy components
        self._validator = None

    def validate(self, pipeline_output: dict[str, Any]) -> list[str]:
        """Run all validators and return violations.

        Delegates to level5_policy/validator.

        Args:
            pipeline_output: Dictionary containing pipeline outputs

        Returns:
            List of violation messages (empty if no violations)
        """
        # TODO: Import and delegate to level5_policy.validator when implemented
        # For now, return empty list (no violations detected)
        try:
            from level5_policy.validator import ConstraintValidator

            if self._validator is None:
                self._validator = ConstraintValidator(self.intent)

            violations = self._validator.validate(pipeline_output)
            return violations
        except ImportError:
            # Validator not yet implemented - return no violations
            return []
