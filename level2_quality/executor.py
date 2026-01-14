"""Safety gate executor for Level 2 quality actions.

This module validates and executes LLM-proposed data cleaning actions.
All validation is deterministic and enforces strict safety rules.

Safety boundaries:
- Validates all actions against schema and intent
- Rejects unsafe actions (target column modification, constraint violations)
- Applies only approved transformations
- Logs all rejected actions with reasons
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from intent.schema import IntentSchema, OutlierPolicy
from level1_ingestion.schema_inferencer import SchemaMetadata
from level2_quality.agent import AgentError

logger = logging.getLogger(__name__)


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
        valid_actions = ["impute", "clip", "flag", "drop"]
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

        # Validate method for impute action
        if action_type == "impute":
            method = action.get("method", "")
            valid_methods = ["mean", "median", "mode"]
            if method not in valid_methods:
                return False, f"Invalid imputation method: {method}. Must be one of {valid_methods}"

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
                    return False, f"Method 'mode' only valid for categorical/text columns"

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
            if action_type == "impute":
                self._apply_impute(column_name, method)
            elif action_type == "clip":
                self._apply_clip(column_name)
            elif action_type == "flag":
                self._apply_flag(column_name)
            elif action_type == "drop":
                self._apply_drop(column_name)
            else:
                raise ExecutionError(f"Unknown action type: {action_type}")

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

    def _apply_impute(self, column_name: str, method: str) -> None:
        """Apply imputation to a column.

        Args:
            column_name: Name of column to impute
            method: Imputation method (mean, median, mode)
        """
        series = self.df[column_name]
        missing_mask = series.isna()

        if method == "mean":
            fill_value = series.mean()
        elif method == "median":
            fill_value = series.median()
        elif method == "mode":
            fill_value = series.mode()[0] if len(series.mode()) > 0 else None
        else:
            raise ExecutionError(f"Unknown imputation method: {method}")

        if fill_value is not None:
            self.df.loc[missing_mask, column_name] = fill_value
            logger.debug(f"Imputed {missing_mask.sum()} missing values in '{column_name}' using {method}")

    def _apply_clip(self, column_name: str) -> None:
        """Apply outlier clipping to a numeric column.

        Args:
            column_name: Name of column to clip
        """
        series = self.df[column_name]
        if not pd.api.types.is_numeric_dtype(series):
            raise ExecutionError(f"Cannot clip non-numeric column: {column_name}")

        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25

        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        clipped_count = ((series < lower_bound) | (series > upper_bound)).sum()
        self.df[column_name] = series.clip(lower=lower_bound, upper=upper_bound)

        logger.debug(f"Clipped {clipped_count} outliers in '{column_name}'")

    def _apply_flag(self, column_name: str) -> None:
        """Flag outliers by creating a new boolean column.

        Args:
            column_name: Name of column to flag outliers in
        """
        series = self.df[column_name]
        if not pd.api.types.is_numeric_dtype(series):
            raise ExecutionError(f"Cannot flag outliers in non-numeric column: {column_name}")

        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25

        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        outlier_flag_name = f"{column_name}_outlier"
        self.df[outlier_flag_name] = (series < lower_bound) | (series > upper_bound)

        flagged_count = self.df[outlier_flag_name].sum()
        logger.debug(f"Flagged {flagged_count} outliers in '{column_name}' as '{outlier_flag_name}'")

    def _apply_drop(self, column_name: str) -> None:
        """Drop a column from the DataFrame.

        Args:
            column_name: Name of column to drop
        """
        if column_name not in self.df.columns:
            raise ExecutionError(f"Column '{column_name}' does not exist")

        dropped_count = len(self.df.columns)
        self.df = self.df.drop(columns=[column_name])
        logger.debug(f"Dropped column '{column_name}' ({dropped_count} -> {len(self.df.columns)} columns)")

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
