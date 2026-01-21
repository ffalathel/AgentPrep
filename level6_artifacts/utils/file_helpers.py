"""File helper utilities for Level 6 artifacts.

This module provides safe file operations for artifact storage.
All operations are deterministic and ensure immutability.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class FileHelperError(Exception):
    """Raised when file operations fail."""

    pass


def ensure_directory(path: Path) -> None:
    """Ensure directory exists, creating it if necessary.

    Resolves path before operations to prevent symlink attacks.

    Args:
        path: Directory path to ensure

    Raises:
        FileHelperError: If directory creation fails
    """
    try:
        # Resolve path to prevent symlink attacks
        resolved_path = path.resolve()
        resolved_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {resolved_path}")
    except (OSError, RuntimeError) as e:
        raise FileHelperError(f"Failed to resolve or create directory {path}: {e}") from e
    except Exception as e:
        raise FileHelperError(f"Failed to create directory {path}: {e}") from e


def safe_write_json(data: Any, file_path: Path, overwrite: bool = False) -> None:
    """Safely write JSON data to file.

    Resolves path before operations to prevent symlink attacks.

    Args:
        data: Data to serialize to JSON
        file_path: Path to write file
        overwrite: If True, overwrite existing file; if False, raise error if exists

    Raises:
        FileHelperError: If write fails or file exists and overwrite=False
    """
    try:
        # Resolve path to prevent symlink attacks
        resolved_path = file_path.resolve()
    except (OSError, RuntimeError) as e:
        raise FileHelperError(f"Failed to resolve path {file_path}: {e}") from e

    if resolved_path.exists() and not overwrite:
        raise FileHelperError(f"File already exists: {resolved_path} (use overwrite=True to replace)")

    try:
        ensure_directory(resolved_path.parent)
        with open(resolved_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.debug(f"JSON written to: {resolved_path}")
    except (OSError, IOError) as e:
        raise FileHelperError(f"Failed to write JSON to {resolved_path}: I/O error: {e}") from e
    except (TypeError, ValueError) as e:
        raise FileHelperError(f"Failed to serialize data to JSON for {resolved_path}: {e}") from e
    except Exception as e:
        raise FileHelperError(f"Unexpected error writing JSON to {resolved_path}: {e}") from e


def safe_write_dataframe(
    df: pd.DataFrame, file_path: Path, format: str = "parquet", overwrite: bool = False
) -> None:
    """Safely write DataFrame to file.

    Resolves path before operations to prevent symlink attacks.

    Args:
        df: DataFrame to write
        file_path: Path to write file
        format: File format ("parquet" or "csv")
        overwrite: If True, overwrite existing file; if False, raise error if exists

    Raises:
        FileHelperError: If write fails or file exists and overwrite=False
    """
    try:
        # Resolve path to prevent symlink attacks
        resolved_path = file_path.resolve()
    except (OSError, RuntimeError) as e:
        raise FileHelperError(f"Failed to resolve path {file_path}: {e}") from e

    if resolved_path.exists() and not overwrite:
        raise FileHelperError(f"File already exists: {resolved_path} (use overwrite=True to replace)")

    try:
        ensure_directory(resolved_path.parent)
        if format == "parquet":
            df.to_parquet(resolved_path, index=False)
        elif format == "csv":
            df.to_csv(resolved_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        logger.debug(f"DataFrame written to: {resolved_path} ({format})")
    except (OSError, IOError) as e:
        raise FileHelperError(f"Failed to write DataFrame to {resolved_path}: I/O error: {e}") from e
    except ImportError as e:
        raise FileHelperError(
            f"Failed to write DataFrame to {resolved_path}: Missing required library for {format} format: {e}"
        ) from e
    except ValueError as e:
        raise FileHelperError(f"Failed to write DataFrame to {resolved_path}: Invalid format or data: {e}") from e
    except Exception as e:
        raise FileHelperError(f"Unexpected error writing DataFrame to {resolved_path}: {e}") from e


def safe_write_text(text: str, file_path: Path, overwrite: bool = False) -> None:
    """Safely write text to file.

    Resolves path before operations to prevent symlink attacks.

    Args:
        text: Text content to write
        file_path: Path to write file
        overwrite: If True, overwrite existing file; if False, raise error if exists

    Raises:
        FileHelperError: If write fails or file exists and overwrite=False
    """
    try:
        # Resolve path to prevent symlink attacks
        resolved_path = file_path.resolve()
    except (OSError, RuntimeError) as e:
        raise FileHelperError(f"Failed to resolve path {file_path}: {e}") from e

    if resolved_path.exists() and not overwrite:
        raise FileHelperError(f"File already exists: {resolved_path} (use overwrite=True to replace)")

    try:
        ensure_directory(resolved_path.parent)
        with open(resolved_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.debug(f"Text written to: {resolved_path}")
    except (OSError, IOError) as e:
        raise FileHelperError(f"Failed to write text to {resolved_path}: I/O error: {e}") from e
    except UnicodeEncodeError as e:
        raise FileHelperError(f"Failed to write text to {resolved_path}: Encoding error: {e}") from e
    except Exception as e:
        raise FileHelperError(f"Unexpected error writing text to {resolved_path}: {e}") from e


def safe_read_json(file_path: Path) -> dict[str, Any]:
    """Safely read JSON from file.

    Resolves path before operations to prevent symlink attacks.

    Args:
        file_path: Path to read file

    Returns:
        Parsed JSON data

    Raises:
        FileHelperError: If read fails or file doesn't exist
    """
    try:
        # Resolve path to prevent symlink attacks
        resolved_path = file_path.resolve()
    except (OSError, RuntimeError) as e:
        raise FileHelperError(f"Failed to resolve path {file_path}: {e}") from e

    if not resolved_path.exists():
        raise FileHelperError(f"File does not exist: {resolved_path}")

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"JSON read from: {resolved_path}")
        return data
    except (OSError, IOError) as e:
        raise FileHelperError(f"Failed to read JSON from {resolved_path}: I/O error: {e}") from e
    except json.JSONDecodeError as e:
        raise FileHelperError(f"Failed to parse JSON from {resolved_path}: Invalid JSON syntax: {e}") from e
    except UnicodeDecodeError as e:
        raise FileHelperError(f"Failed to read JSON from {resolved_path}: Encoding error: {e}") from e
    except Exception as e:
        raise FileHelperError(f"Unexpected error reading JSON from {resolved_path}: {e}") from e


def safe_read_dataframe(file_path: Path, format: str = "parquet") -> pd.DataFrame:
    """Safely read DataFrame from file.

    Resolves path before operations to prevent symlink attacks.

    Args:
        file_path: Path to read file
        format: File format ("parquet" or "csv")

    Returns:
        DataFrame

    Raises:
        FileHelperError: If read fails or file doesn't exist
    """
    try:
        # Resolve path to prevent symlink attacks
        resolved_path = file_path.resolve()
    except (OSError, RuntimeError) as e:
        raise FileHelperError(f"Failed to resolve path {file_path}: {e}") from e

    if not resolved_path.exists():
        raise FileHelperError(f"File does not exist: {resolved_path}")

    try:
        if format == "parquet":
            df = pd.read_parquet(resolved_path)
        elif format == "csv":
            df = pd.read_csv(resolved_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        logger.debug(f"DataFrame read from: {resolved_path} ({format})")
        return df
    except (OSError, IOError) as e:
        raise FileHelperError(f"Failed to read DataFrame from {resolved_path}: I/O error: {e}") from e
    except ImportError as e:
        raise FileHelperError(
            f"Failed to read DataFrame from {resolved_path}: Missing required library for {format} format: {e}"
        ) from e
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        raise FileHelperError(f"Failed to parse DataFrame from {resolved_path}: {e}") from e
    except ValueError as e:
        raise FileHelperError(f"Failed to read DataFrame from {resolved_path}: Invalid format or data: {e}") from e
    except Exception as e:
        raise FileHelperError(f"Unexpected error reading DataFrame from {resolved_path}: {e}") from e
