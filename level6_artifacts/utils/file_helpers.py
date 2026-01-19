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

    Args:
        path: Directory path to ensure

    Raises:
        FileHelperError: If directory creation fails
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {path}")
    except Exception as e:
        raise FileHelperError(f"Failed to create directory {path}: {e}") from e


def safe_write_json(data: Any, file_path: Path, overwrite: bool = False) -> None:
    """Safely write JSON data to file.

    Args:
        data: Data to serialize to JSON
        file_path: Path to write file
        overwrite: If True, overwrite existing file; if False, raise error if exists

    Raises:
        FileHelperError: If write fails or file exists and overwrite=False
    """
    if file_path.exists() and not overwrite:
        raise FileHelperError(f"File already exists: {file_path} (use overwrite=True to replace)")

    try:
        ensure_directory(file_path.parent)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.debug(f"JSON written to: {file_path}")
    except Exception as e:
        raise FileHelperError(f"Failed to write JSON to {file_path}: {e}") from e


def safe_write_dataframe(
    df: pd.DataFrame, file_path: Path, format: str = "parquet", overwrite: bool = False
) -> None:
    """Safely write DataFrame to file.

    Args:
        df: DataFrame to write
        file_path: Path to write file
        format: File format ("parquet" or "csv")
        overwrite: If True, overwrite existing file; if False, raise error if exists

    Raises:
        FileHelperError: If write fails or file exists and overwrite=False
    """
    if file_path.exists() and not overwrite:
        raise FileHelperError(f"File already exists: {file_path} (use overwrite=True to replace)")

    try:
        ensure_directory(file_path.parent)
        if format == "parquet":
            df.to_parquet(file_path, index=False)
        elif format == "csv":
            df.to_csv(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        logger.debug(f"DataFrame written to: {file_path} ({format})")
    except Exception as e:
        raise FileHelperError(f"Failed to write DataFrame to {file_path}: {e}") from e


def safe_write_text(text: str, file_path: Path, overwrite: bool = False) -> None:
    """Safely write text to file.

    Args:
        text: Text content to write
        file_path: Path to write file
        overwrite: If True, overwrite existing file; if False, raise error if exists

    Raises:
        FileHelperError: If write fails or file exists and overwrite=False
    """
    if file_path.exists() and not overwrite:
        raise FileHelperError(f"File already exists: {file_path} (use overwrite=True to replace)")

    try:
        ensure_directory(file_path.parent)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.debug(f"Text written to: {file_path}")
    except Exception as e:
        raise FileHelperError(f"Failed to write text to {file_path}: {e}") from e


def safe_read_json(file_path: Path) -> dict[str, Any]:
    """Safely read JSON from file.

    Args:
        file_path: Path to read file

    Returns:
        Parsed JSON data

    Raises:
        FileHelperError: If read fails or file doesn't exist
    """
    if not file_path.exists():
        raise FileHelperError(f"File does not exist: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"JSON read from: {file_path}")
        return data
    except Exception as e:
        raise FileHelperError(f"Failed to read JSON from {file_path}: {e}") from e


def safe_read_dataframe(file_path: Path, format: str = "parquet") -> pd.DataFrame:
    """Safely read DataFrame from file.

    Args:
        file_path: Path to read file
        format: File format ("parquet" or "csv")

    Returns:
        DataFrame

    Raises:
        FileHelperError: If read fails or file doesn't exist
    """
    if not file_path.exists():
        raise FileHelperError(f"File does not exist: {file_path}")

    try:
        if format == "parquet":
            df = pd.read_parquet(file_path)
        elif format == "csv":
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        logger.debug(f"DataFrame read from: {file_path} ({format})")
        return df
    except Exception as e:
        raise FileHelperError(f"Failed to read DataFrame from {file_path}: {e}") from e
