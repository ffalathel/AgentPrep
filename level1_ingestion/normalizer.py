"""Dataset normalizer for Level 1 ingestion.

This module normalizes column names and types, ensuring the target column
is protected and intent constraints are satisfied.
"""

import logging
import re
from typing import Optional

import pandas as pd

from intent.schema import IntentSchema

logger = logging.getLogger(__name__)


class NormalizationError(Exception):
    """Raised when normalization fails."""

    pass


def to_snake_case(name: str) -> str:
    """Convert a string to snake_case.

    Handles various naming conventions:
    - CamelCase -> camel_case
    - kebab-case -> kebab_case
    - PascalCase -> pascal_case
    - spaces -> underscores
    - multiple underscores -> single underscore

    Args:
        name: String to convert

    Returns:
        snake_case version of the string
    """
    # Replace spaces and hyphens with underscores
    name = re.sub(r"[\s-]+", "_", name)

    # Insert underscore before uppercase letters (for CamelCase)
    name = re.sub(r"(?<!^)(?<!_)([A-Z])", r"_\1", name)

    # Convert to lowercase
    name = name.lower()

    # Remove leading/trailing underscores
    name = name.strip("_")

    # Replace multiple underscores with single underscore
    name = re.sub(r"_+", "_", name)

    return name


def normalize_column_names(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """Normalize column names to lowercase snake_case.

    Creates a mapping from original to normalized names and ensures
    the target column is preserved correctly.

    Args:
        df: DataFrame to normalize
        target_column: Name of the target column (must exist in original)

    Returns:
        Tuple of (normalized DataFrame, mapping dict from original to normalized names)

    Raises:
        NormalizationError: If target column doesn't exist or normalization fails
    """
    if target_column not in df.columns:
        raise NormalizationError(
            f"Target column '{target_column}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # Create mapping from original to normalized names
    name_mapping = {}
    for original_name in df.columns:
        normalized = to_snake_case(original_name)
        name_mapping[original_name] = normalized

    # Check for duplicate normalized names
    normalized_names = list(name_mapping.values())
    if len(normalized_names) != len(set(normalized_names)):
        duplicates = [
            orig for orig, norm in name_mapping.items()
            if normalized_names.count(norm) > 1
        ]
        raise NormalizationError(
            f"Column name normalization created duplicates: {duplicates}. "
            "Original column names are too similar."
        )

    # Rename columns
    df_normalized = df.rename(columns=name_mapping).copy()

    # Verify target column mapping
    normalized_target = name_mapping[target_column]
    if normalized_target not in df_normalized.columns:
        raise NormalizationError(
            f"Target column normalization failed: '{target_column}' -> '{normalized_target}'"
        )

    logger.info(f"Normalized {len(name_mapping)} column names")
    logger.debug(f"Target column mapping: '{target_column}' -> '{normalized_target}'")

    return df_normalized, name_mapping


def safe_cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Safely cast column types to appropriate pandas dtypes.

    Attempts to optimize types where safe (e.g., int64 -> int32 for memory).
    Does not change semantic meaning of data.

    Args:
        df: DataFrame to cast

    Returns:
        DataFrame with optimized types
    """
    df_casted = df.copy()

    for col in df_casted.columns:
        original_dtype = df_casted[col].dtype

        # Try to downcast numeric types if safe
        if pd.api.types.is_integer_dtype(df_casted[col]):
            df_casted[col] = pd.to_numeric(df_casted[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df_casted[col]):
            df_casted[col] = pd.to_numeric(df_casted[col], downcast="float")

        new_dtype = df_casted[col].dtype
        if original_dtype != new_dtype:
            logger.debug(f"Column '{col}': {original_dtype} -> {new_dtype}")

    return df_casted


def validate_intent_constraints(
    df: pd.DataFrame, intent: IntentSchema, normalized_target: str
) -> None:
    """Validate dataset against intent constraints.

    Checks that the dataset satisfies constraints defined in intent.
    Fails fast if constraints are violated.

    Args:
        df: Normalized DataFrame
        intent: Validated IntentSchema
        normalized_target: Normalized name of target column

    Raises:
        NormalizationError: If constraints are violated
    """
    # Check max_features constraint
    feature_columns = [col for col in df.columns if col != normalized_target]
    if len(feature_columns) > intent.constraints.max_features:
        raise NormalizationError(
            f"Dataset has {len(feature_columns)} feature columns, "
            f"exceeds max_features constraint of {intent.constraints.max_features}"
        )

    # Check max_cardinality for categorical columns
    for col in feature_columns:
        is_categorical = (
            df[col].dtype == "object"
            or isinstance(df[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(df[col])
        )
        if is_categorical:
            cardinality = df[col].nunique()
            if cardinality > intent.constraints.max_cardinality:
                raise NormalizationError(
                    f"Column '{col}' has cardinality {cardinality}, "
                    f"exceeds max_cardinality constraint of {intent.constraints.max_cardinality}"
                )

    logger.debug("Intent constraints validated successfully")


def normalize_dataset(
    df: pd.DataFrame, intent: IntentSchema
) -> tuple[pd.DataFrame, dict[str, str], str]:
    """Normalize dataset: column names and types.

    This is the main entry point for normalization. It:
    1. Normalizes column names to snake_case
    2. Safely casts types
    3. Validates intent constraints
    4. Ensures target column is protected

    Args:
        df: Raw DataFrame to normalize
        intent: Validated IntentSchema

    Returns:
        Tuple of (normalized DataFrame, column name mapping, normalized target column name)

    Raises:
        NormalizationError: If normalization or validation fails
    """
    logger.info("Starting dataset normalization")

    # Step 1: Normalize column names
    df_normalized, name_mapping = normalize_column_names(df, intent.task.target_column)
    normalized_target = name_mapping[intent.task.target_column]

    # Step 2: Safe type casting
    df_normalized = safe_cast_types(df_normalized)

    # Step 3: Validate intent constraints
    validate_intent_constraints(df_normalized, intent, normalized_target)

    logger.info("Dataset normalization completed successfully")
    logger.info(f"Normalized dataset shape: {df_normalized.shape}")
    logger.info(f"Normalized target column: '{normalized_target}'")

    return df_normalized, name_mapping, normalized_target
