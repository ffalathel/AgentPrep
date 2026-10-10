"""Validator orchestration for Level 5 governance.

Delegates all validation to level5_policy/validator.
"""

from typing import Any

from intent.schema import IntentSchema
from level5_policy.validator import ConstraintValidator


class ValidatorOrchestrator:
    """Thin adapter to the constraint validator."""

    def __init__(self, intent: IntentSchema):
        self.intent = intent
        self._validator = ConstraintValidator(intent)

    def validate(self, pipeline_output: dict[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
        """Return (blocking violations, warnings, remediation_info)."""
        return self._validator.validate(pipeline_output)
