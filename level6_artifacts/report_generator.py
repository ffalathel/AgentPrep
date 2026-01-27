"""Report generator for Level 6 artifacts.

This module converts artifacts into human-readable reports.
It does not recompute logic - it reads from artifact_registry and schemas.
"""

from pathlib import Path
from typing import Any, Optional

from utils import get_logger

from .artifact_registry import ArtifactRegistry
from .schemas import ArtifactManifest, RunSummary
from .utils.file_helpers import FileHelperError, safe_write_text
from .utils.logging import log_report_generated

logger = get_logger(__name__)


class ReportGenerationError(Exception):
    """Raised when report generation fails."""

    pass


class ReportGenerator:
    """Generates human-readable reports from artifacts.

    This class:
    - Converts artifacts into reports
    - Does not recompute logic
    - Reads from artifact_registry and schemas
    - Produces Markdown reports
    """

    def __init__(self, output_dir: Path, run_id: str):
        """Initialize report generator.

        Args:
            output_dir: Base directory for reports (should be pre-validated)
            run_id: Unique run identifier
        """
        # Resolve output directory to prevent symlink attacks
        try:
            self.output_dir = Path(output_dir).resolve()
        except (OSError, RuntimeError) as e:
            logger.warning(f"Failed to resolve output directory, using as-is: {e}")
            self.output_dir = Path(output_dir)
        
        self.run_id = run_id
        # Sanitize run_id to prevent path injection
        sanitized_run_id = "".join(c for c in run_id if c.isalnum() or c in ['-', '_', '.'])[:100]
        self.reports_dir = self.output_dir / "reports" / sanitized_run_id
        # Resolve reports directory before creating
        try:
            resolved_reports_dir = self.reports_dir.resolve()
            resolved_reports_dir.mkdir(parents=True, exist_ok=True)
            self.reports_dir = resolved_reports_dir
        except (OSError, RuntimeError) as e:
            logger.error(f"Failed to resolve or create reports directory: {e}")
            raise
        logger.debug(f"ReportGenerator initialized: {self.reports_dir}")

    def generate_run_report(self, manifest: ArtifactManifest) -> Path:
        """Generate pipeline run summary report.

        Args:
            manifest: ArtifactManifest with run information

        Returns:
            Path to generated report file

        Raises:
            ReportGenerationError: If report generation fails
        """
        filename = f"report_{self.run_id}.md"
        report_path = self.reports_dir / filename

        try:
            summary = manifest.run_summary
            report_content = self._format_run_report(summary, manifest)

            safe_write_text(report_content, report_path, overwrite=True)
            log_report_generated("run_summary", str(report_path))
            return report_path

        except FileHelperError as e:
            raise ReportGenerationError(f"Failed to generate run report: {e}") from e

    def generate_governance_report(self, manifest: ArtifactManifest) -> Optional[Path]:
        """Generate governance compliance report.

        Args:
            manifest: ArtifactManifest with governance information

        Returns:
            Path to generated report file, or None if no governance artifact

        Raises:
            ReportGenerationError: If report generation fails
        """
        if not manifest.governance_artifact:
            return None

        filename = f"governance_{self.run_id}.md"
        report_path = self.reports_dir / filename

        try:
            snapshot = manifest.governance_artifact.snapshot
            decision = snapshot.decision

            report_content = self._format_governance_report(decision, manifest)

            safe_write_text(report_content, report_path, overwrite=True)
            log_report_generated("governance", str(report_path))
            return report_path

        except FileHelperError as e:
            raise ReportGenerationError(f"Failed to generate governance report: {e}") from e

    def generate_dataset_overview(self, manifest: ArtifactManifest) -> Optional[Path]:
        """Generate dataset and feature overview report.

        Args:
            manifest: ArtifactManifest with dataset information

        Returns:
            Path to generated report file, or None if no dataset artifact

        Raises:
            ReportGenerationError: If report generation fails
        """
        if not manifest.dataset_artifact:
            return None

        filename = f"dataset_overview_{self.run_id}.md"
        report_path = self.reports_dir / filename

        try:
            report_content = self._format_dataset_overview(manifest)

            safe_write_text(report_content, report_path, overwrite=True)
            log_report_generated("dataset_overview", str(report_path))
            return report_path

        except FileHelperError as e:
            raise ReportGenerationError(f"Failed to generate dataset overview: {e}") from e

    def generate_all_reports(self, manifest: ArtifactManifest) -> dict[str, Path]:
        """Generate all reports.

        Args:
            manifest: ArtifactManifest with all information

        Returns:
            Dictionary mapping report types to paths

        Raises:
            ReportGenerationError: If report generation fails
        """
        reports = {}

        try:
            # Generate run report
            run_report = self.generate_run_report(manifest)
            reports["run_summary"] = run_report

            # Generate governance report
            governance_report = self.generate_governance_report(manifest)
            if governance_report:
                reports["governance"] = governance_report

            # Generate dataset overview
            dataset_overview = self.generate_dataset_overview(manifest)
            if dataset_overview:
                reports["dataset_overview"] = dataset_overview

            logger.info(f"Generated {len(reports)} reports in {self.reports_dir}")
            return reports

        except Exception as e:
            raise ReportGenerationError(f"Failed to generate all reports: {e}") from e

    def _format_run_report(self, summary: RunSummary, manifest: ArtifactManifest) -> str:
        """Format run summary report.

        Args:
            summary: RunSummary object
            manifest: ArtifactManifest

        Returns:
            Markdown-formatted report
        """
        status_icon = "✓" if summary.success else "✗"
        status_text = "SUCCESS" if summary.success else "FAILED"

        report = f"""# Pipeline Run Report

**Run ID:** `{summary.run_id}`  
**Status:** {status_icon} {status_text}  
**Exit Code:** {summary.exit_code}

## Execution Summary

- **Started:** {summary.started_at.isoformat()}
- **Completed:** {summary.completed_at.isoformat()}
- **Duration:** {self._format_duration(summary.started_at, summary.completed_at)}

## Dataset Information

- **Dataset Path:** `{summary.dataset_path}`
- **Target Column:** `{summary.target_column}`
- **Feature Count:** {summary.feature_count}

## Governance

- **Approved:** {status_icon if summary.governance_approved else "✗"} {summary.governance_approved}
- **Reason:** {summary.governance_reason}

## Artifacts

"""

        artifacts = manifest.to_dict()["artifacts"]
        for artifact_name, artifact_path in artifacts.items():
            if artifact_path:
                report += f"- **{artifact_name.replace('_', ' ').title()}:** `{artifact_path}`\n"

        return report

    def _format_governance_report(self, decision: Any, manifest: ArtifactManifest) -> str:
        """Format governance compliance report.

        Args:
            decision: GovernanceDecision object
            manifest: ArtifactManifest

        Returns:
            Markdown-formatted report
        """
        status_icon = "✓" if decision.approved else "✗"
        status_text = "APPROVED" if decision.approved else "REJECTED"

        report = f"""# Governance Compliance Report

**Run ID:** `{manifest.run_id}`  
**Status:** {status_icon} {status_text}

## Decision

{decision.reason}

## Violations

"""

        if decision.violations:
            for i, violation in enumerate(decision.violations, 1):
                report += f"{i}. {violation}\n"
        else:
            report += "No violations detected.\n"

        # Add warnings section if any warnings exist
        warnings = getattr(decision, "warnings", [])
        if warnings:
            report += f"\n## Warnings (Informational)\n\n"
            for i, warning in enumerate(warnings, 1):
                report += f"{i}. ⚠️ {warning}\n"
            report += "\n*Note: Warnings do not block pipeline execution but should be reviewed.*\n"

        report += f"\n## Leakage Detection\n\n"
        if decision.leakage_detected:
            report += "✗ **Leakage detected**\n"
        else:
            report += "✓ **No leakage detected**\n"

        return report

    def _format_dataset_overview(self, manifest: ArtifactManifest) -> str:
        """Format dataset overview report.

        Args:
            manifest: ArtifactManifest

        Returns:
            Markdown-formatted report
        """
        dataset = manifest.dataset_artifact
        if not dataset:
            return "# Dataset Overview\n\nNo dataset artifact available.\n"

        report = f"""# Dataset Overview

**Run ID:** `{manifest.run_id}`

## Dataset Statistics

- **Rows:** {dataset.row_count:,}
- **Columns:** {dataset.column_count}
- **Target Column:** `{dataset.target_column}`
- **Format:** {dataset.format}
- **Path:** `{dataset.path}`

## Feature Information

"""

        if manifest.feature_provenance_artifact:
            report += f"- **Generated Features:** {manifest.feature_provenance_artifact.feature_count}\n"
            report += f"- **Provenance Path:** `{manifest.feature_provenance_artifact.path}`\n"
        else:
            report += "- **Generated Features:** 0\n"

        return report

    def _format_duration(self, start: Any, end: Any) -> str:
        """Format duration between two timestamps.

        Args:
            start: Start timestamp
            end: End timestamp

        Returns:
            Formatted duration string
        """
        delta = end - start
        total_seconds = int(delta.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
