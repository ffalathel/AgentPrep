"""Logging utilities for AgentPrep.

This module provides shared logging configuration and utilities
used across the entire application.
"""

import logging
import sys
from typing import Optional


def setup_logging(verbose: bool = False, level: Optional[int] = None) -> None:
    """Configure logging for AgentPrep.

    This function sets up the root logger with a consistent format.
    It can be called multiple times safely.

    Args:
        verbose: If True, set log level to DEBUG, otherwise INFO
        level: Optional explicit log level (overrides verbose)
    """
    # Clear existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Determine log level
    if level is not None:
        log_level = level
    else:
        log_level = logging.DEBUG if verbose else logging.INFO

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,  # Override any existing configuration
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
