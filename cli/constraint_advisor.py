"""Heuristic-based constraint advisor for interactive CLI.

This module provides deterministic, rule-based suggestions for constraint values
(max_features, max_interactions, max_cardinality) based on dataset characteristics.
No LLM calls - purely heuristic-based.
"""

from typing import Optional

import pandas as pd

from utils import get_logger

logger = get_logger(__name__)


class ConstraintAdvisor:
    """Deterministic constraint advisor using heuristic rules.

    Analyzes dataset characteristics and suggests appropriate constraint values
    based on dataset size, task type, model family, and column characteristics.
    """

    def __init__(self):
        """Initialize the constraint advisor."""
        logger.debug("ConstraintAdvisor initialized")

    def suggest_constraints(
        self,
        dataframe: pd.DataFrame,
        task_type: str,
        model_family: str,
        target_column: Optional[str] = None,
    ) -> dict:
        """Suggest constraint values based on dataset characteristics.

        Args:
            dataframe: Dataset DataFrame
            task_type: Task type (classification, regression, etc.)
            model_family: Model family (tree, linear, neural, unknown)
            target_column: Optional target column name (to exclude from feature count)

        Returns:
            Dictionary with max_features, max_interactions, max_cardinality, reasoning
        """
        rows, cols = dataframe.shape

        # Calculate current number of feature columns (excluding target)
        if target_column and target_column in dataframe.columns:
            current_feature_count = cols - 1
        else:
            # Conservative estimate: assume one target column
            current_feature_count = max(1, cols - 1)

        # Analyze categorical columns
        categorical_cols = self._get_categorical_columns(dataframe)
        max_categorical_cardinality = 0
        num_categorical = len(categorical_cols)

        if categorical_cols:
            max_categorical_cardinality = int(
                dataframe[categorical_cols].nunique().max()
            )

        # Max features based on model family
        max_features = self._suggest_max_features(rows, cols, model_family, task_type)
        
        # Ensure max_features is at least as large as current feature count
        # (since we can't reduce features during normalization)
        max_features = max(max_features, current_feature_count)

        # Max interactions based on dataset size and max_features
        max_interactions = self._suggest_max_interactions(
            rows, max_features, task_type
        )

        # Max cardinality based on actual categorical data
        max_cardinality = self._suggest_max_cardinality(
            rows, max_categorical_cardinality, num_categorical
        )

        # Build reasoning
        reasoning = self._build_reasoning(
            rows,
            cols,
            model_family,
            task_type,
            max_features,
            max_interactions,
            max_cardinality,
            num_categorical,
            max_categorical_cardinality,
        )

        return {
            "max_features": max_features,
            "max_interactions": max_interactions,
            "max_cardinality": max_cardinality,
            "reasoning": reasoning,
        }

    def _get_categorical_columns(self, dataframe: pd.DataFrame) -> list[str]:
        """Get list of categorical column names.

        Args:
            dataframe: Dataset DataFrame

        Returns:
            List of categorical column names
        """
        categorical_cols = []
        for col in dataframe.columns:
            dtype = dataframe[col].dtype
            # Check for object, category, or string dtypes
            if (
                dtype == "object"
                or dtype.name == "category"
                or str(dtype).startswith("string")
                or isinstance(dtype, pd.CategoricalDtype)
            ):
                categorical_cols.append(col)
        return categorical_cols

    def _suggest_max_features(
        self, rows: int, cols: int, model_family: str, task_type: str
    ) -> int:
        """Suggest max_features based on model family and task type.

        Args:
            rows: Number of rows in dataset
            cols: Number of columns in dataset
            model_family: Model family (tree, linear, neural, unknown)
            task_type: Task type (classification, regression, etc.)

        Returns:
            Suggested max_features value
        """
        # Base calculation by model family
        if model_family == "tree":
            # Tree models can handle many features
            base = min(500, cols * 10, max(10, rows // 10))
        elif model_family == "linear":
            # Linear models need fewer features
            base = min(100, cols * 5, max(10, rows // 20))
        elif model_family == "neural":
            # Neural networks moderate capacity
            base = min(200, cols * 8, max(10, rows // 15))
        else:  # unknown
            # Conservative default
            base = min(100, cols * 5)

        # Task type adjustments
        if task_type == "classification":
            base = int(base * 1.1)  # Classification can handle more features
        elif task_type == "regression":
            base = int(base * 0.9)  # Regression more conservative

        # Ensure within bounds
        return max(10, min(10000, base))

    def _suggest_max_interactions(
        self, rows: int, max_features: int, task_type: str
    ) -> int:
        """Suggest max_interactions based on dataset size and max_features.

        Args:
            rows: Number of rows in dataset
            max_features: Suggested max_features value
            task_type: Task type (classification, regression, etc.)

        Returns:
            Suggested max_interactions value
        """
        # Base on dataset size
        if rows < 1000:
            # Small datasets
            base = max(5, int(max_features * 0.2))
        elif rows < 10000:
            # Medium datasets
            base = max(10, int(max_features * 0.3))
        else:
            # Large datasets
            base = max(20, int(max_features * 0.4))

        # Task type adjustments
        if task_type == "time_series":
            base = int(base * 1.2)  # Time series benefits from more interactions

        # Ensure within bounds
        return max(0, min(1000, base))

    def _suggest_max_cardinality(
        self, rows: int, max_categorical_cardinality: int, num_categorical: int
    ) -> int:
        """Suggest max_cardinality based on actual categorical data.

        Args:
            rows: Number of rows in dataset
            max_categorical_cardinality: Maximum unique values in any categorical column
            num_categorical: Number of categorical columns

        Returns:
            Suggested max_cardinality value
        """
        # If no categorical columns, return default
        if num_categorical == 0 or max_categorical_cardinality == 0:
            return 10000

        # Base on dataset size
        if rows < 1000:
            # Small datasets
            base = max(100, int(max_categorical_cardinality * 1.5), 1000)
        elif rows < 10000:
            # Medium datasets
            base = max(1000, int(max_categorical_cardinality * 1.2), 5000)
        else:
            # Large datasets
            base = max(5000, int(max_categorical_cardinality * 1.1), 10000)

        # Ensure within bounds
        return max(2, min(1000000, base))

    def _build_reasoning(
        self,
        rows: int,
        cols: int,
        model_family: str,
        task_type: str,
        max_features: int,
        max_interactions: int,
        max_cardinality: int,
        num_categorical: int,
        max_categorical_cardinality: int,
    ) -> str:
        """Build human-readable reasoning for suggestions.

        Args:
            rows: Number of rows
            cols: Number of columns
            model_family: Model family
            task_type: Task type
            max_features: Suggested max_features
            max_interactions: Suggested max_interactions
            max_cardinality: Suggested max_cardinality
            num_categorical: Number of categorical columns
            max_categorical_cardinality: Max cardinality found

        Returns:
            Reasoning string
        """
        parts = [
            f"Based on {rows:,} rows, {cols} columns, {model_family} model family, "
            f"and {task_type} task."
        ]

        # Model family explanation
        if model_family == "tree":
            parts.append("Tree-based models can handle more features effectively.")
        elif model_family == "linear":
            parts.append("Linear models work best with fewer, well-selected features.")
        elif model_family == "neural":
            parts.append("Neural networks have moderate feature capacity.")

        # Categorical information
        if num_categorical > 0:
            parts.append(
                f"Found {num_categorical} categorical column(s) with max cardinality of {max_categorical_cardinality}."
            )
        else:
            parts.append("No categorical columns detected.")

        return " ".join(parts)
