"""Logging utilities for Level 6 artifacts.

This module provides structured logging for artifact operations.
"""

import logging

logger = logging.getLogger(__name__)


def log_artifact_created(artifact_name: str, artifact_path: str) -> None:
    """Log artifact creation.

    Args:
        artifact_name: Name of the artifact
        artifact_path: Path where artifact was created
    """
    logger.info(f"Artifact created: {artifact_name} -> {artifact_path}")


def log_artifact_exported(artifact_name: str, export_path: str, format: str) -> None:
    """Log artifact export.

    Args:
        artifact_name: Name of the artifact
        export_path: Path where artifact was exported
        format: Export format (json, csv, markdown)
    """
    logger.info(f"Artifact exported: {artifact_name} -> {export_path} ({format})")


def log_report_generated(report_type: str, report_path: str) -> None:
    """Log report generation.

    Args:
        report_type: Type of report (run_summary, governance, etc.)
        report_path: Path where report was generated
    """
    logger.info(f"Report generated: {report_type} -> {report_path}")
