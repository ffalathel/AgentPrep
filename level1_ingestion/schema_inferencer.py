"""Schema inference for Level 1 ingestion.

This module infers schema metadata for each column in the dataset,
including semantic types, missing value statistics, and data types.
All outputs are machine-readable structured objects.
"""

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class SemanticType:
    """Semantic type constants for columns."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"


@dataclass
class ColumnMetadata:
    """Metadata for a single column.

    All fields are machine-readable and deterministic.
    """

    name: str
    semantic_type: str
    missing_percentage: float
    unique_count: int
    pandas_dtype: str
    total_count: int


@dataclass
class SchemaMetadata:
    """Complete schema metadata for a dataset.

    Contains metadata for all columns in a structured format.
    """

    columns: dict[str, ColumnMetadata]
    total_rows: int
    total_columns: int


def infer_semantic_type(series: pd.Series) -> str:
    """Infer semantic type for a pandas Series.

    Args:
        series: pandas Series to analyze

    Returns:
        Semantic type string (numeric, categorical, boolean, datetime, text)
    """
    # Check for boolean type
    if series.dtype == "bool" or series.dtype.name == "bool":
        return SemanticType.BOOLEAN

    # Check for datetime type
    if pd.api.types.is_datetime64_any_dtype(series):
        return SemanticType.DATETIME

    # Check for numeric type
    if pd.api.types.is_numeric_dtype(series):
        # Distinguish between numeric and categorical
        # If numeric but has low cardinality relative to size, might be categorical
        # But for now, we'll classify based on dtype
        return SemanticType.NUMERIC

    # Check for categorical (object type with limited unique values)
    if series.dtype == "object" or series.dtype.name == "object":
        # Check if it's actually categorical (low cardinality) or text (high cardinality)
        unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
        # If unique ratio is low (< 0.5) and cardinality is reasonable, treat as categorical
        if unique_ratio < 0.5 and series.nunique() < 1000:
            return SemanticType.CATEGORICAL
        else:
            return SemanticType.TEXT

    # Check for string/category dtype
    if pd.api.types.is_string_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
        if unique_ratio < 0.5 and series.nunique() < 1000:
            return SemanticType.CATEGORICAL
        else:
            return SemanticType.TEXT

    # Default to text for unknown types
    return SemanticType.TEXT


def infer_column_metadata(df: pd.DataFrame, column_name: str) -> ColumnMetadata:
    """Infer metadata for a single column.

    Args:
        df: DataFrame containing the column
        column_name: Name of the column to analyze

    Returns:
        ColumnMetadata instance with inferred information
    """
    series = df[column_name]
    total_count = len(series)
    missing_count = series.isna().sum()
    missing_percentage = (missing_count / total_count * 100) if total_count > 0 else 0.0
    unique_count = series.nunique()
    semantic_type = infer_semantic_type(series)
    pandas_dtype = str(series.dtype)

    return ColumnMetadata(
        name=column_name,
        semantic_type=semantic_type,
        missing_percentage=round(missing_percentage, 4),
        unique_count=unique_count,
        pandas_dtype=pandas_dtype,
        total_count=total_count,
    )


def infer_schema(df: pd.DataFrame) -> SchemaMetadata:
    """Infer complete schema metadata for a DataFrame.

    Args:
        df: DataFrame to analyze

    Returns:
        SchemaMetadata instance with metadata for all columns
    """
    logger.info(f"Inferring schema for {df.shape[1]} columns")

    columns_metadata = {}
    for column_name in df.columns:
        metadata = infer_column_metadata(df, column_name)
        columns_metadata[column_name] = metadata
        logger.debug(
            f"Column '{column_name}': type={metadata.semantic_type}, "
            f"missing={metadata.missing_percentage}%, unique={metadata.unique_count}"
        )

    schema = SchemaMetadata(
        columns=columns_metadata,
        total_rows=len(df),
        total_columns=len(df.columns),
    )

    logger.info(f"Schema inference complete: {schema.total_rows} rows, {schema.total_columns} columns")
    return schema
