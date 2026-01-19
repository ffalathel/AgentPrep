"""Artifact registry for Level 6 artifacts.

This module tracks all artifacts produced in a pipeline run.
It maps artifact names to paths and metadata, but does not store files itself.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils import get_logger

from .schemas import (
    ArtifactManifest,
    DatasetArtifact,
    FeatureProvenanceArtifact,
    GovernanceArtifact,
    MetadataArtifact,
    QualityProfileArtifact,
    RunSummary,
    SchemaArtifact,
)

logger = get_logger(__name__)


@dataclass
class ArtifactRegistry:
    """Registry tracking all artifacts for a pipeline run.

    This registry:
    - Maps artifact names to artifact objects
    - Tracks artifact paths and metadata
    - Does not store files itself (delegates to store.py)
    - Is versionable and queryable
    """

    run_id: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    dataset_artifact: Optional[DatasetArtifact] = None
    schema_artifact: Optional[SchemaArtifact] = None
    quality_profile_artifact: Optional[QualityProfileArtifact] = None
    feature_provenance_artifact: Optional[FeatureProvenanceArtifact] = None
    metadata_artifact: Optional[MetadataArtifact] = None
    governance_artifact: Optional[GovernanceArtifact] = None

    def register_dataset(self, artifact: DatasetArtifact) -> None:
        """Register dataset artifact.

        Args:
            artifact: DatasetArtifact to register
        """
        self.dataset_artifact = artifact
        logger.debug(f"Registered dataset artifact: {artifact.path}")

    def register_schema(self, artifact: SchemaArtifact) -> None:
        """Register schema artifact.

        Args:
            artifact: SchemaArtifact to register
        """
        self.schema_artifact = artifact
        logger.debug(f"Registered schema artifact: {artifact.path}")

    def register_quality_profile(self, artifact: QualityProfileArtifact) -> None:
        """Register quality profile artifact.

        Args:
            artifact: QualityProfileArtifact to register
        """
        self.quality_profile_artifact = artifact
        logger.debug(f"Registered quality profile artifact: {artifact.path}")

    def register_feature_provenance(self, artifact: FeatureProvenanceArtifact) -> None:
        """Register feature provenance artifact.

        Args:
            artifact: FeatureProvenanceArtifact to register
        """
        self.feature_provenance_artifact = artifact
        logger.debug(f"Registered feature provenance artifact: {artifact.path}")

    def register_metadata(self, artifact: MetadataArtifact) -> None:
        """Register metadata artifact.

        Args:
            artifact: MetadataArtifact to register
        """
        self.metadata_artifact = artifact
        logger.debug(f"Registered metadata artifact: {artifact.path}")

    def register_governance(self, artifact: GovernanceArtifact) -> None:
        """Register governance artifact.

        Args:
            artifact: GovernanceArtifact to register
        """
        self.governance_artifact = artifact
        logger.debug(f"Registered governance artifact: {artifact.path}")

    def get_artifact_path(self, artifact_name: str) -> Optional[Path]:
        """Get path for a registered artifact.

        Args:
            artifact_name: Name of artifact (e.g., "dataset", "schema")

        Returns:
            Path to artifact if registered, None otherwise
        """
        mapping = {
            "dataset": self.dataset_artifact,
            "schema": self.schema_artifact,
            "quality_profile": self.quality_profile_artifact,
            "feature_provenance": self.feature_provenance_artifact,
            "metadata": self.metadata_artifact,
            "governance": self.governance_artifact,
        }

        artifact = mapping.get(artifact_name)
        return artifact.path if artifact else None

    def list_artifacts(self) -> list[str]:
        """List all registered artifact names.

        Returns:
            List of artifact names
        """
        artifacts = []
        if self.dataset_artifact:
            artifacts.append("dataset")
        if self.schema_artifact:
            artifacts.append("schema")
        if self.quality_profile_artifact:
            artifacts.append("quality_profile")
        if self.feature_provenance_artifact:
            artifacts.append("feature_provenance")
        if self.metadata_artifact:
            artifacts.append("metadata")
        if self.governance_artifact:
            artifacts.append("governance")
        return artifacts

    def build_manifest(
        self, completed_at: datetime, success: bool, exit_code: int = 0
    ) -> ArtifactManifest:
        """Build artifact manifest from registry.

        Args:
            completed_at: Completion timestamp
            success: Whether run was successful
            exit_code: Exit code from pipeline

        Returns:
            ArtifactManifest with all registered artifacts
        """
        # Build run summary
        run_summary = RunSummary(
            run_id=self.run_id,
            started_at=self.started_at,
            completed_at=completed_at,
            success=success,
            dataset_path=str(self.dataset_artifact.path) if self.dataset_artifact else "",
            target_column=self.dataset_artifact.target_column if self.dataset_artifact else "",
            feature_count=self.feature_provenance_artifact.feature_count if self.feature_provenance_artifact else 0,
            governance_approved=self.governance_artifact.snapshot.decision.approved if self.governance_artifact else False,
            governance_reason=self.governance_artifact.snapshot.decision.reason if self.governance_artifact else "",
            exit_code=exit_code,
        )

        manifest = ArtifactManifest(
            run_id=self.run_id,
            run_summary=run_summary,
            dataset_artifact=self.dataset_artifact,
            schema_artifact=self.schema_artifact,
            quality_profile_artifact=self.quality_profile_artifact,
            feature_provenance_artifact=self.feature_provenance_artifact,
            metadata_artifact=self.metadata_artifact,
            governance_artifact=self.governance_artifact,
        )

        logger.debug(f"Built artifact manifest for run {self.run_id}")
        return manifest
