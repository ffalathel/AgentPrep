"""Safety gate executor for Level 2 quality actions.

This module validates and executes LLM-proposed data cleaning actions.
All validation is deterministic and enforces strict safety rules.

Safety boundaries:
- Validates all actions against schema and intent
- Rejects unsafe actions (target column modification, constraint violations)
- Applies only approved transformations
- Logs all rejected actions with reasons
"""

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from core.recipe import apply_step, fit_step, placeholder_mask
from intent.schema import IntentSchema, OutlierPolicy
from level1_ingestion.schema_inferencer import SchemaMetadata
from utils import get_logger

logger = get_logger(__name__)

# Above this, one fill value replaces most of the column and erases the missingness signal
# (seen on Titanic: `cabin`, 77% missing, imputed with its mode)
MAX_IMPUTE_MISSING_RATE = 0.5


class ExecutionError(Exception):
    """Raised when action execution fails."""

    pass


@dataclass
class AppliedAction:
    """Record of an applied action."""

    column: str
    action: str
    method: Optional[str]
    justification: Optional[str]
    status: str  # "applied" or "rejected"
    reason: Optional[str] = None  # Reason for rejection if applicable


class ActionExecutor:
    """Validates and executes data quality improvement actions.

    This executor:
    - Validates all proposed actions
    - Rejects unsafe actions
    - Applies approved transformations deterministically
    - Logs all actions (applied and rejected)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        schema_metadata: SchemaMetadata,
        intent: IntentSchema,
        normalized_target: str,
    ):
        """Initialize the action executor.

        Args:
            df: DataFrame to transform
            schema_metadata: Schema metadata from Level 1
            intent: User intent from Level 0
            normalized_target: Normalized name of target column
        """
        self.df = df.copy()  # Work on a copy
        self.schema_metadata = schema_metadata
        self.intent = intent
        self.normalized_target = normalized_target
        self.applied_actions: list[AppliedAction] = []
        self.steps: list[dict[str, Any]] = []  # fitted, replayable on new data (core.recipe)

        logger.info("ActionExecutor initialized")
        logger.debug(f"Target column: {normalized_target}")

    def validate_action(self, action: dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate a single action against safety rules.

        Args:
            action: Action dictionary with keys: column, action, method, justification

        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        # Check required fields
        required_fields = ["column", "action"]
        for field in required_fields:
            if field not in action:
                return False, f"Missing required field: {field}"

        column_name = action["column"]
        action_type = action["action"]

        # Validate column exists
        if column_name not in self.df.columns:
            return False, f"Column '{column_name}' does not exist in dataset"

        # CRITICAL: Never allow actions on target column
        if column_name == self.normalized_target:
            return False, f"Cannot modify target column '{column_name}'"

        # Validate action type
        valid_actions = ["impute", "clip", "flag", "drop", "replace_with_missing"]
        if action_type not in valid_actions:
            return False, f"Invalid action type: {action_type}. Must be one of {valid_actions}"

        # Validate drop action constraints
        if action_type == "drop":
            if not self.intent.preferences.allow_column_dropping:
                return False, "Column dropping not allowed by intent preferences"

            # Check max_features constraint
            current_feature_count = len(
                [c for c in self.df.columns if c != self.normalized_target]
            )
            if current_feature_count - 1 < 1:
                return False, "Cannot drop column: would violate minimum feature requirement"

        # Validate outlier actions against policy
        if action_type in ["clip", "flag"]:
            if self.intent.preferences.outlier_policy == OutlierPolicy.PRESERVE:
                return False, f"Outlier action '{action_type}' violates preserve policy"

            if action_type == "clip" and self.intent.preferences.outlier_policy != OutlierPolicy.CLIP:
                return False, "Clip action only allowed when outlier_policy is 'clip'"

            if action_type == "flag" and self.intent.preferences.outlier_policy != OutlierPolicy.FLAG:
                return False, "Flag action only allowed when outlier_policy is 'flag'"

        if action_type == "replace_with_missing":
            if "value" not in action:
                return False, "replace_with_missing needs the placeholder 'value'"
            s, value = self.df[column_name], action["value"]
            if not placeholder_mask(s, value).any():
                return False, f"Value {value!r} does not occur in '{column_name}'"
            numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
            if numeric and value not in (s.min(), s.max()):
                return False, (
                    f"Value {value!r} is inside the range of '{column_name}'; "
                    "a numeric placeholder must be its lowest or highest value"
                )

        # Validate method for impute action
        if action_type == "impute":
            method = action.get("method", "")
            valid_methods = ["mean", "median", "mode"]
            if method not in valid_methods:
                return False, f"Invalid imputation method: {method}. Must be one of {valid_methods}"

            missing_rate = self.df[column_name].isna().mean()
            if missing_rate > MAX_IMPUTE_MISSING_RATE:
                return False, (
                    f"Imputing {missing_rate:.0%} missing values would overwrite most of "
                    f"'{column_name}' with one value; use a missing indicator or drop it"
                )

            # Check if method is appropriate for column type
            col_metadata = self.schema_metadata.columns.get(column_name)
            if col_metadata:
                if method in ["mean", "median"] and col_metadata.semantic_type not in [
                    "numeric"
                ]:
                    return False, f"Method '{method}' only valid for numeric columns"

                if method == "mode" and col_metadata.semantic_type not in [
                    "categorical",
                    "text",
                ]:
                    return False, "Method 'mode' only valid for categorical/text columns"

        return True, None

    def apply_action(self, action: dict[str, Any]) -> AppliedAction:
        """Apply a single validated action to the DataFrame.

        Args:
            action: Validated action dictionary

        Returns:
            AppliedAction record
        """
        column_name = action["column"]
        action_type = action["action"]
        method = action.get("method")
        justification = action.get("justification", "")

        try:
            step = fit_step(action_type, self.df, [column_name], method=method, value=action.get("value"))
            self.df = apply_step(self.df, step)
            self.steps.append(step)

            return AppliedAction(
                column=column_name,
                action=action_type,
                method=method,
                justification=justification,
                status="applied",
            )

        except Exception as e:
            logger.error(f"Failed to apply action {action_type} on {column_name}: {e}")
            return AppliedAction(
                column=column_name,
                action=action_type,
                method=method,
                justification=justification,
                status="rejected",
                reason=str(e),
            )

    def execute_actions(self, actions: list[dict[str, Any]]) -> pd.DataFrame:
        """Validate and execute a list of proposed actions.

        This is the main entry point. It validates all actions,
        applies approved ones, and logs rejections.

        Args:
            actions: List of action dictionaries from LLM

        Returns:
            Transformed DataFrame with approved actions applied
        """
        logger.info(f"Validating and executing {len(actions)} proposed actions")
        # Placeholders become missing first, so imputation and the missing-rate check see them
        actions = sorted(actions, key=lambda a: not (isinstance(a, dict) and a.get("action") == "replace_with_missing"))

        for action in actions:
            is_valid, rejection_reason = self.validate_action(action)

            if not is_valid:
                # Log rejected action
                applied_action = AppliedAction(
                    column=action.get("column", "unknown"),
                    action=action.get("action", "unknown"),
                    method=action.get("method"),
                    justification=action.get("justification"),
                    status="rejected",
                    reason=rejection_reason,
                )
                self.applied_actions.append(applied_action)
                logger.warning(
                    f"Rejected action on '{applied_action.column}': {rejection_reason}"
                )
                continue

            # Apply validated action
            applied_action = self.apply_action(action)
            self.applied_actions.append(applied_action)
            if applied_action.status == "applied":
                logger.info(
                    f"Applied {applied_action.action} on '{applied_action.column}'"
                )

        applied_count = sum(1 for a in self.applied_actions if a.status == "applied")
        rejected_count = sum(1 for a in self.applied_actions if a.status == "rejected")

        logger.info(
            f"Action execution complete: {applied_count} applied, {rejected_count} rejected"
        )

        return self.df
