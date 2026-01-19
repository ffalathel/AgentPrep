"""Exporter for Level 6 artifacts.

This module exports artifacts in human-readable and machine-readable formats.
It does not generate content, only formats existing artifacts.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from utils import get_logger

from .artifact_registry import ArtifactRegistry
from .schemas import ArtifactManifest
from .utils.file_helpers import FileHelperError, safe_write_dataframe, safe_write_json, safe_write_text
from .utils.logging import log_artifact_exported

logger = get_logger(__name__)


class ExportError(Exception):
    """Raised when export operations fail."""

    pass


class ArtifactExporter:
    """Exports artifacts in various formats.

    This class:
    - Exports artifacts in JSON, CSV, Markdown formats
    - Ensures file paths are deterministic
    - Supports batch export
    - Does not generate content, only formats existing artifacts
    """

    def __init__(self, output_dir: Path, run_id: str):
        """Initialize artifact exporter.

        Args:
            output_dir: Base directory for exports
            run_id: Unique run identifier
        """
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.exports_dir = self.output_dir / "exports" / run_id
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"ArtifactExporter initialized: {self.exports_dir}")

    def export_dataset(
        self, artifact: Any, formats: list[str] = ["csv", "parquet"]
    ) -> dict[str, Path]:
        """Export dataset artifact in multiple formats.

        Args:
            artifact: DatasetArtifact with path
            formats: List of formats to export ("csv", "parquet")

        Returns:
            Dictionary mapping format to export path

        Raises:
            ExportError: If export fails
        """
        exported_paths = {}

        try:
            # Load dataset
            df = pd.read_parquet(artifact.path) if artifact.format == "parquet" else pd.read_csv(artifact.path)

            for format_type in formats:
                extension = ".csv" if format_type == "csv" else ".parquet"
                filename = f"dataset_{self.run_id}{extension}"
                export_path = self.exports_dir / filename

                safe_write_dataframe(df, export_path, format=format_type, overwrite=True)
                exported_paths[format_type] = export_path
                log_artifact_exported("dataset", str(export_path), format_type)

            return exported_paths

        except Exception as e:
            raise ExportError(f"Failed to export dataset: {e}") from e

    def export_manifest(self, manifest: ArtifactManifest) -> Path:
        """Export artifact manifest as JSON.

        Args:
            manifest: ArtifactManifest to export

        Returns:
            Path to exported manifest file

        Raises:
            ExportError: If export fails
        """
        filename = f"manifest_{self.run_id}.json"
        export_path = self.exports_dir / filename

        try:
            data = manifest.to_dict()
            safe_write_json(data, export_path, overwrite=True)
            log_artifact_exported("manifest", str(export_path), "json")
            return export_path

        except FileHelperError as e:
            raise ExportError(f"Failed to export manifest: {e}") from e

    def export_all(self, registry: ArtifactRegistry, manifest: ArtifactManifest) -> dict[str, Path]:
        """Export all artifacts in batch.

        Args:
            registry: ArtifactRegistry with all artifacts
            manifest: ArtifactManifest to export

        Returns:
            Dictionary mapping artifact names to export paths

        Raises:
            ExportError: If export fails
        """
        exported = {}

        try:
            # Export manifest
            manifest_path = self.export_manifest(manifest)
            exported["manifest"] = manifest_path

            # Export dataset if available
            if registry.dataset_artifact:
                dataset_paths = self.export_dataset(registry.dataset_artifact, formats=["csv", "parquet"])
                exported["dataset_csv"] = dataset_paths.get("csv")
                exported["dataset_parquet"] = dataset_paths.get("parquet")

            # Export other artifacts as JSON (they're already JSON)
            if registry.schema_artifact:
                exported["schema"] = self._copy_json_artifact(registry.schema_artifact, "schema")

            if registry.quality_profile_artifact:
                exported["quality_profile"] = self._copy_json_artifact(registry.quality_profile_artifact, "quality_profile")

            if registry.feature_provenance_artifact:
                exported["feature_provenance"] = self._copy_json_artifact(
                    registry.feature_provenance_artifact, "feature_provenance"
                )

            if registry.metadata_artifact:
                exported["metadata"] = self._copy_json_artifact(registry.metadata_artifact, "metadata")

            if registry.governance_artifact:
                exported["governance"] = self._copy_json_artifact(registry.governance_artifact, "governance")

            logger.info(f"Exported {len(exported)} artifacts to {self.exports_dir}")
            return exported

        except Exception as e:
            raise ExportError(f"Failed to export all artifacts: {e}") from e

    def _copy_json_artifact(self, artifact: Any, artifact_name: str) -> Path:
        """Copy JSON artifact to exports directory.

        Args:
            artifact: Artifact with path attribute
            artifact_name: Name of artifact

        Returns:
            Path to exported file
        """
        filename = f"{artifact_name}_{self.run_id}.json"
        export_path = self.exports_dir / filename

        try:
            import shutil

            shutil.copy2(artifact.path, export_path)
            log_artifact_exported(artifact_name, str(export_path), "json")
            return export_path

        except Exception as e:
            raise ExportError(f"Failed to copy {artifact_name} artifact: {e}") from e
