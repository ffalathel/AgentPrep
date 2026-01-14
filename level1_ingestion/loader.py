"""Dataset loader for Level 1 ingestion.

This module handles loading datasets from disk in supported formats
(CSV, Parquet) using pandas. It validates file existence and format
without modifying the data.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DatasetLoadError(Exception):
    """Raised when dataset loading fails."""

    pass


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load dataset from disk.

    Supports CSV and Parquet formats. Validates file existence and format
    before loading. Returns DataFrame without any modifications.

    Args:
        file_path: Path to dataset file

    Returns:
        pandas DataFrame containing the raw dataset

    Raises:
        DatasetLoadError: If file doesn't exist, format is unsupported, or loading fails
    """
    file_path = Path(file_path)

    # Validate file exists
    if not file_path.exists():
        raise DatasetLoadError(f"Dataset file not found: {file_path.absolute()}")

    if not file_path.is_file():
        raise DatasetLoadError(f"Dataset path is not a file: {file_path.absolute()}")

    # Determine format from extension
    suffix = file_path.suffix.lower()
    if suffix not in (".csv", ".parquet"):
        raise DatasetLoadError(
            f"Unsupported file format: {suffix}. Supported formats: .csv, .parquet"
        )

    logger.info(f"Loading dataset from: {file_path}")
    logger.debug(f"File format: {suffix}")

    try:
        if suffix == ".csv":
            df = pd.read_csv(file_path)
        elif suffix == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            # This should never happen due to validation above
            raise DatasetLoadError(f"Unsupported format: {suffix}")

        logger.info(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.debug(f"Column names: {list(df.columns)}")

        return df

    except pd.errors.EmptyDataError as e:
        raise DatasetLoadError(f"Dataset file is empty: {file_path}") from e
    except pd.errors.ParserError as e:
        raise DatasetLoadError(f"Failed to parse dataset file: {e}") from e
    except Exception as e:
        raise DatasetLoadError(f"Failed to load dataset: {e}") from e
