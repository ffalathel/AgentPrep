"""Dataset loader for Level 1 ingestion.

This module handles loading datasets from disk in supported formats
(CSV, Parquet) using pandas. It validates file existence and format
without modifying the data.
"""

from pathlib import Path

import pandas as pd

from utils import PathValidationError, get_logger, validate_path_safe

logger = get_logger(__name__)


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
        PathValidationError: If path contains directory traversal or security issues
    """
    # Validate path for security (prevents directory traversal)
    try:
        file_path = validate_path_safe(
            file_path, must_exist=True, must_be_file=True
        )
    except PathValidationError as e:
        raise DatasetLoadError(f"Invalid dataset path: {e}") from e
    except FileNotFoundError as e:
        raise DatasetLoadError(f"Dataset file not found: {file_path}") from e

    # Determine format from extension
    suffix = file_path.suffix.lower()
    if suffix not in (".csv", ".parquet"):
        raise DatasetLoadError(
            f"Unsupported file format: {suffix}. Supported formats: .csv, .parquet"
        )

    logger.info(f"Loading dataset from: {file_path}")
    logger.debug(f"File format: {suffix}")

    try:
        # Use resolved path for actual file operations
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

    except (OSError, IOError) as e:
        raise DatasetLoadError(f"Failed to read dataset file {file_path}: I/O error: {e}") from e
    except pd.errors.EmptyDataError as e:
        raise DatasetLoadError(f"Dataset file is empty: {file_path}") from e
    except pd.errors.ParserError as e:
        raise DatasetLoadError(f"Failed to parse dataset file {file_path}: {e}") from e
    except ImportError as e:
        raise DatasetLoadError(
            f"Failed to load dataset {file_path}: Missing required library for {suffix} format: {e}"
        ) from e
    except MemoryError as e:
        raise DatasetLoadError(f"Failed to load dataset {file_path}: Insufficient memory: {e}") from e
    except Exception as e:
        raise DatasetLoadError(f"Unexpected error loading dataset {file_path}: {e}") from e
