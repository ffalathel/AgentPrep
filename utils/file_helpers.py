"""File helper utilities for AgentPrep.

This module provides common file operations used across the application.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure

    Returns:
        Path object (for chaining)

    Raises:
        OSError: If directory creation fails
    """
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Directory ensured: {path}")
    return path


def get_file_extension(file_path: str | Path) -> str:
    """Get file extension from path.

    Args:
        file_path: File path

    Returns:
        File extension (without dot), empty string if no extension
    """
    path = Path(file_path)
    return path.suffix.lstrip(".").lower()


def is_supported_dataset_format(file_path: str | Path) -> bool:
    """Check if file is a supported dataset format.

    Args:
        file_path: File path to check

    Returns:
        True if format is supported
    """
    extension = get_file_extension(file_path)
    return extension in ["csv", "parquet"]


def is_supported_config_format(file_path: str | Path) -> bool:
    """Check if file is a supported config format.

    Args:
        file_path: File path to check

    Returns:
        True if format is supported
    """
    extension = get_file_extension(file_path)
    return extension in ["yaml", "yml", "json"]


def generate_run_id() -> str:
    """Generate a unique run ID.

    Returns:
        Run ID string (timestamp-based)
    """
    from datetime import datetime

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    return f"run_{timestamp}"
