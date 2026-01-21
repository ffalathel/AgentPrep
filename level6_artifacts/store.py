"""Persistence layer for Level 6 artifacts.

This module handles saving and loading artifacts to disk.
It ensures immutability and deterministic file paths.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from level4_feature.generator import FeatureProvenance
from level5_governance.gatekeeper import GovernanceDecision
from utils import get_logger

from .schemas import (
    DatasetArtifact,
    FeatureProvenanceArtifact,
    GovernanceArtifact,
    GovernanceSnapshot,
    MetadataArtifact,
    QualityProfileArtifact,
    SchemaArtifact,
)
from .utils.file_helpers import (
    FileHelperError,
    ensure_directory,
    safe_write_dataframe,
    safe_write_json,
)

logger = get_logger(__name__)


class ArtifactStoreError(Exception):
    """Raised when artifact storage operations fail."""

    pass


class ArtifactStore:
    """Persistence layer for artifacts.

    This class:
    - Saves artifacts to disk
    - Handles directories, filenames, versioning
    - Ensures immutability (no overwrite unless explicitly allowed)
    - Does not modify or compute artifacts
    """

    def __init__(self, output_dir: Path, run_id: str):
        """Initialize artifact store.

        Args:
            output_dir: Base directory for artifacts (should be pre-validated)
            run_id: Unique run identifier

        Raises:
            ArtifactStoreError: If output directory is invalid or cannot be created
        """
        # Output directory should already be validated by orchestrator, but ensure it's safe
        try:
            from utils import validate_output_path
            self.output_dir = validate_output_path(output_dir, allow_existing=True)
        except Exception:
            # If validation fails, use original but log warning
            logger.warning(f"Output directory validation failed, using as-is: {output_dir}")
            self.output_dir = Path(output_dir)
        
        self.run_id = run_id
        # Sanitize run_id to prevent path injection
        sanitized_run_id = "".join(c for c in run_id if c.isalnum() or c in ['-', '_', '.'])[:100]
        self.artifacts_dir = self.output_dir / "artifacts" / sanitized_run_id
        ensure_directory(self.artifacts_dir)
        logger.debug(f"ArtifactStore initialized: {self.artifacts_dir}")

    def save_dataset(
        self, dataframe: pd.DataFrame, format: str = "csv", overwrite: bool = False
    ) -> DatasetArtifact:
        """Save dataset artifact.

        Args:
            dataframe: DataFrame to save
            format: File format ("parquet" or "csv")
            overwrite: If True, overwrite existing file

        Returns:
            DatasetArtifact with path and metadata

        Raises:
            ArtifactStoreError: If save fails
        """
        extension = ".parquet" if format == "parquet" else ".csv"
        filename = f"dataset_{self.run_id}{extension}"
        file_path = self.artifacts_dir / filename

        try:
            safe_write_dataframe(dataframe, file_path, format=format, overwrite=overwrite)

            # Determine target column (assume last column or find it)
            target_column = dataframe.columns[-1] if len(dataframe.columns) > 0 else ""

            artifact = DatasetArtifact(
                run_id=self.run_id,
                path=file_path,
                row_count=len(dataframe),
                column_count=len(dataframe.columns),
                target_column=target_column,
                format=format,
            )

            logger.info(f"Dataset artifact saved: {file_path}")
            return artifact

        except FileHelperError as e:
            raise ArtifactStoreError(f"Failed to save dataset artifact: {e}") from e

    def save_schema_metadata(
        self, schema_metadata: Any, overwrite: bool = False
    ) -> SchemaArtifact:
        """Save schema metadata artifact.

        Args:
            schema_metadata: Schema metadata object (from Level 1)
            overwrite: If True, overwrite existing file

        Returns:
            SchemaArtifact with path and metadata

        Raises:
            ArtifactStoreError: If save fails
        """
        filename = f"schema_{self.run_id}.json"
        file_path = self.artifacts_dir / filename

        try:
            # Convert schema metadata to dict for JSON serialization
            if hasattr(schema_metadata, "__dict__"):
                data = self._dataclass_to_dict(schema_metadata)
            else:
                data = schema_metadata

            safe_write_json(data, file_path, overwrite=overwrite)

            artifact = SchemaArtifact(
                run_id=self.run_id,
                path=file_path,
                total_rows=schema_metadata.total_rows if hasattr(schema_metadata, "total_rows") else 0,
                total_columns=schema_metadata.total_columns if hasattr(schema_metadata, "total_columns") else 0,
            )

            logger.info(f"Schema artifact saved: {file_path}")
            return artifact

        except FileHelperError as e:
            raise ArtifactStoreError(f"Failed to save schema artifact: {e}") from e

    def save_quality_profile(
        self, quality_profile: Any, overwrite: bool = False
    ) -> QualityProfileArtifact:
        """Save quality profile artifact.

        Args:
            quality_profile: Quality profile object (from Level 2)
            overwrite: If True, overwrite existing file

        Returns:
            QualityProfileArtifact with path and metadata

        Raises:
            ArtifactStoreError: If save fails
        """
        filename = f"quality_{self.run_id}.json"
        file_path = self.artifacts_dir / filename

        try:
            # Convert quality profile to dict for JSON serialization
            if hasattr(quality_profile, "__dict__"):
                data = self._dataclass_to_dict(quality_profile)
            else:
                data = quality_profile

            safe_write_json(data, file_path, overwrite=overwrite)

            artifact = QualityProfileArtifact(
                run_id=self.run_id,
                path=file_path,
                total_columns=quality_profile.total_columns if hasattr(quality_profile, "total_columns") else 0,
            )

            logger.info(f"Quality profile artifact saved: {file_path}")
            return artifact

        except FileHelperError as e:
            raise ArtifactStoreError(f"Failed to save quality profile artifact: {e}") from e

    def save_feature_provenance(
        self, provenance: list[FeatureProvenance], overwrite: bool = False
    ) -> FeatureProvenanceArtifact:
        """Save feature provenance artifact.

        Args:
            provenance: List of FeatureProvenance objects
            overwrite: If True, overwrite existing file

        Returns:
            FeatureProvenanceArtifact with path and metadata

        Raises:
            ArtifactStoreError: If save fails
        """
        filename = f"provenance_{self.run_id}.json"
        file_path = self.artifacts_dir / filename

        try:
            # Convert provenance to list of dicts
            data = [self._dataclass_to_dict(prov) for prov in provenance]

            safe_write_json(data, file_path, overwrite=overwrite)

            artifact = FeatureProvenanceArtifact(
                run_id=self.run_id,
                path=file_path,
                feature_count=len(provenance),
            )

            logger.info(f"Feature provenance artifact saved: {file_path}")
            return artifact

        except FileHelperError as e:
            raise ArtifactStoreError(f"Failed to save feature provenance artifact: {e}") from e

    def save_metadata(self, metadata_path: Path, overwrite: bool = False) -> MetadataArtifact:
        """Save metadata artifact (from Level 3).

        Args:
            metadata_path: Path to existing metadata file
            overwrite: If True, overwrite existing file

        Returns:
            MetadataArtifact with path

        Raises:
            ArtifactStoreError: If save fails
        """
        if not metadata_path.exists():
            raise ArtifactStoreError(f"Metadata file does not exist: {metadata_path}")

        filename = f"metadata_{self.run_id}.json"
        file_path = self.artifacts_dir / filename

        try:
            # Copy metadata file to artifacts directory
            ensure_directory(file_path.parent)
            shutil.copy2(metadata_path, file_path)

            artifact = MetadataArtifact(
                run_id=self.run_id,
                path=file_path,
            )

            logger.info(f"Metadata artifact saved: {file_path}")
            return artifact

        except Exception as e:
            raise ArtifactStoreError(f"Failed to save metadata artifact: {e}") from e

    def save_governance_snapshot(
        self, decision: GovernanceDecision, overwrite: bool = False
    ) -> GovernanceArtifact:
        """Save governance snapshot artifact.

        Args:
            decision: GovernanceDecision object
            overwrite: If True, overwrite existing file

        Returns:
            GovernanceArtifact with path and snapshot

        Raises:
            ArtifactStoreError: If save fails
        """
        filename = f"governance_{self.run_id}.json"
        file_path = self.artifacts_dir / filename

        try:
            snapshot = GovernanceSnapshot(
                run_id=self.run_id,
                decision=decision,
            )

            # Convert snapshot to dict for JSON serialization
            data = {
                "run_id": snapshot.run_id,
                "decision": {
                    "approved": snapshot.decision.approved,
                    "violations": snapshot.decision.violations,
                    "leakage_detected": snapshot.decision.leakage_detected,
                    "reason": snapshot.decision.reason,
                },
                "created_at": snapshot.created_at.isoformat(),
            }

            safe_write_json(data, file_path, overwrite=overwrite)

            artifact = GovernanceArtifact(
                run_id=self.run_id,
                path=file_path,
                snapshot=snapshot,
            )

            logger.info(f"Governance artifact saved: {file_path}")
            return artifact

        except FileHelperError as e:
            raise ArtifactStoreError(f"Failed to save governance artifact: {e}") from e

    def _dataclass_to_dict(self, obj: Any) -> dict[str, Any]:
        """Convert dataclass to dictionary for JSON serialization.

        Args:
            obj: Dataclass instance

        Returns:
            Dictionary representation
        """
        if hasattr(obj, "__dict__"):
            result = {}
            for key, value in obj.__dict__.items():
                if hasattr(value, "__dict__"):
                    result[key] = self._dataclass_to_dict(value)
                elif isinstance(value, list):
                    result[key] = [self._dataclass_to_dict(item) if hasattr(item, "__dict__") else item for item in value]
                elif isinstance(value, dict):
                    result[key] = {k: self._dataclass_to_dict(v) if hasattr(v, "__dict__") else v for k, v in value.items()}
                else:
                    result[key] = value
            return result
        return obj
