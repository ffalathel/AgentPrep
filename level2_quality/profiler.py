"""Deterministic dataset profiler for Level 2 quality checks.

This module computes quality metrics for each column in a dataset.
All profiling is deterministic - no LLM involvement.
Output is machine-readable structured data only.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from utils import get_logger

logger = get_logger(__name__)


@dataclass
class NumericStats:
    """Statistics for numeric columns."""

    min: float
    max: float
    mean: float
    median: float
    std: float
    q25: float  # First quartile
    q75: float  # Third quartile
    iqr: float  # Interquartile range
    outlier_count_lower: int  # Count below Q1 - 1.5*IQR
    outlier_count_upper: int  # Count above Q3 + 1.5*IQR
    outlier_count_total: int


@dataclass
class ColumnQualityProfile:
    """Quality profile for a single column.

    All fields are machine-readable and deterministic.
    """

    column_name: str
    missing_count: int
    missing_percentage: float
    unique_count: int
    is_constant: bool  # True if all values are the same
    is_near_constant: bool  # True if >95% of values are the same
    numeric_stats: Optional[NumericStats] = None
    # Values that may stand in for "missing"; the LLM decides which really do
    placeholder_candidates: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DatasetQualityProfile:
    """Complete quality profile for a dataset.

    Contains quality metrics for all columns.
    """

    columns: dict[str, ColumnQualityProfile]
    total_rows: int
    total_columns: int
    duplicate_row_count: int = 0


def compute_numeric_stats(series: pd.Series) -> NumericStats:
    """Compute numeric statistics including outlier detection.

    Uses IQR-based method for outlier detection.

    Args:
        series: Numeric pandas Series

    Returns:
        NumericStats instance with computed statistics
    """
    # Remove NaN values for statistics
    clean_series = series.dropna()

    if len(clean_series) == 0:
        # Return zeros if no valid data
        return NumericStats(
            min=0.0,
            max=0.0,
            mean=0.0,
            median=0.0,
            std=0.0,
            q25=0.0,
            q75=0.0,
            iqr=0.0,
            outlier_count_lower=0,
            outlier_count_upper=0,
            outlier_count_total=0,
        )

    q25 = clean_series.quantile(0.25)
    q75 = clean_series.quantile(0.75)
    iqr = q75 - q25

    # Outlier bounds using IQR method
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr

    outlier_count_lower = (clean_series < lower_bound).sum()
    outlier_count_upper = (clean_series > upper_bound).sum()
    outlier_count_total = outlier_count_lower + outlier_count_upper

    return NumericStats(
        min=float(clean_series.min()),
        max=float(clean_series.max()),
        mean=float(clean_series.mean()),
        median=float(clean_series.median()),
        std=float(clean_series.std()),
        q25=float(q25),
        q75=float(q75),
        iqr=float(iqr),
        outlier_count_lower=int(outlier_count_lower),
        outlier_count_upper=int(outlier_count_upper),
        outlier_count_total=int(outlier_count_total),
    )


PLACEHOLDER_TEXT = {"?", "-", "", "unknown", "n/a", "na", "null", "none", "missing"}
PLACEHOLDER_NUMBERS = {0, -1, -9, -99, -999, 99, 999, 9999, 99999}


def placeholder_candidates(series: pd.Series) -> list[dict[str, Any]]:
    """Possible placeholders: a numeric min/max that is set apart from the rest or a common
    placeholder number (0, -1, 999), or text like "?".

    Many are real values (0 purchases, -1 = never contacted), so these are only shown to the LLM.
    """
    s = series.dropna()
    if s.empty or pd.api.types.is_bool_dtype(s):
        return []
    if pd.api.types.is_numeric_dtype(s):
        values = sorted(s.unique())
        if len(values) < 3:
            return []
        iqr = s.quantile(0.75) - s.quantile(0.25)
        found = []
        for edge, neighbour in ((values[0], values[1]), (values[-1], values[-2])):
            share = float((s == edge).mean())
            set_apart = abs(neighbour - edge) >= 0.5 * iqr
            if share >= 0.003 and (set_apart or edge in PLACEHOLDER_NUMBERS):
                found.append({"value": edge.item() if hasattr(edge, "item") else edge, "share": round(share, 3)})
        return found
    text = s.astype(str).str.strip()
    shares = text[text.str.lower().isin(PLACEHOLDER_TEXT)].value_counts() / len(s)
    return [{"value": v, "share": round(float(share), 3)} for v, share in shares.head(5).items()]


def profile_column(df: pd.DataFrame, column_name: str) -> ColumnQualityProfile:
    """Profile a single column for quality issues.

    Args:
        df: DataFrame containing the column
        column_name: Name of the column to profile

    Returns:
        ColumnQualityProfile instance with quality metrics
    """
    series = df[column_name]
    total_count = len(series)
    missing_count = int(series.isna().sum())
    missing_percentage = (missing_count / total_count * 100) if total_count > 0 else 0.0
    unique_count = int(series.nunique())

    # Check for constant or near-constant columns
    is_constant = unique_count <= 1
    # Near-constant: more than 95% of values are the same
    if total_count > 0:
        value_counts = series.value_counts()
        most_common_count = value_counts.iloc[0] if len(value_counts) > 0 else 0
        most_common_percentage = (most_common_count / total_count) * 100
        is_near_constant = most_common_percentage > 95.0
    else:
        is_near_constant = False

    # Compute numeric stats if column is numeric (but not boolean)
    numeric_stats = None
    if pd.api.types.is_numeric_dtype(series) and series.dtype != "bool" and series.dtype.name != "bool":
        numeric_stats = compute_numeric_stats(series)

    return ColumnQualityProfile(
        column_name=column_name,
        missing_count=missing_count,
        missing_percentage=round(missing_percentage, 4),
        unique_count=unique_count,
        is_constant=is_constant,
        is_near_constant=is_near_constant,
        numeric_stats=numeric_stats,
        placeholder_candidates=placeholder_candidates(series),
    )


def profile_dataset(df: pd.DataFrame) -> DatasetQualityProfile:
    """Profile entire dataset for quality issues.

    This is the main entry point for dataset profiling.
    All computation is deterministic - no LLM involvement.

    Args:
        df: DataFrame to profile

    Returns:
        DatasetQualityProfile instance with quality metrics for all columns
    """
    logger.info(f"Profiling dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    columns_profile = {}
    for column_name in df.columns:
        profile = profile_column(df, column_name)
        columns_profile[column_name] = profile
        logger.debug(
            f"Column '{column_name}': missing={profile.missing_percentage}%, "
            f"unique={profile.unique_count}, constant={profile.is_constant}"
        )

    quality_profile = DatasetQualityProfile(
        columns=columns_profile,
        total_rows=len(df),
        total_columns=len(df.columns),
        duplicate_row_count=int(df.duplicated().sum()),
    )

    logger.info(f"Quality profiling complete: {quality_profile.total_columns} columns profiled")
    return quality_profile
