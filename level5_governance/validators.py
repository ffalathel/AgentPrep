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

    def validate(self, pipeline_output: dict[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
        """Run all validators and return violations, warnings, and remediation info.

        Delegates to level5_policy/validator.

        Args:
            pipeline_output: Dictionary containing pipeline outputs

        Returns:
            Tuple of (blocking violation messages, warning messages, remediation_info dict)
        """
        try:
            from level5_policy.validator import ConstraintValidator

            if self._validator is None:
                self._validator = ConstraintValidator(self.intent)

            result = self._validator.validate(pipeline_output)
            # Handle both old format (violations, remediation_info) and new format (violations, warnings, remediation_info)
            if len(result) == 2:
                violations, remediation_info = result
                return violations, [], remediation_info
            elif len(result) == 3:
                violations, warnings, remediation_info = result
                return violations, warnings, remediation_info
            else:
                return [], [], {}
        except ImportError:
            # Validator not yet implemented - return no violations
            return [], [], {}
        except (TypeError, ValueError):
            # Backward compatibility: if validator returns only violations
            try:
                result = self._validator.validate(pipeline_output)
                if isinstance(result, tuple) and len(result) == 2:
                    violations, remediation_info = result
                    return violations, [], remediation_info
                elif isinstance(result, list):
                    return result, [], {}
                else:
                    return [], [], {}
            except Exception:
                return [], [], {}
