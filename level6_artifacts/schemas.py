"""Artifact schemas for Level 6 artifacts.

This module defines immutable, typed data structures for artifacts.
All schemas are serializable, versionable, and deterministic.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from level5_governance.gatekeeper import GovernanceDecision


@dataclass(frozen=True)
class DatasetArtifact:
    """Artifact representing the final processed dataset.

    This is immutable - once created, it cannot be modified.
    """

    run_id: str
    path: Path
    row_count: int
    column_count: int
    target_column: str
    format: str = "parquet"  # "parquet" or "csv"
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class SchemaArtifact:
    """Artifact representing dataset schema metadata."""

    run_id: str
    path: Path
    total_rows: int
    total_columns: int
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class QualityProfileArtifact:
    """Artifact representing data quality profile."""

    run_id: str
    path: Path
    total_columns: int
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class FeatureProvenanceArtifact:
    """Artifact representing feature engineering provenance."""

    run_id: str
    path: Path
    feature_count: int
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class MetadataArtifact:
    """Artifact representing pipeline metadata (from Level 3)."""

    run_id: str
    path: Path
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class GovernanceSnapshot:
    """Snapshot of governance decision and violations.

    This is immutable and represents the final governance state.
    """

    run_id: str
    decision: GovernanceDecision
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class GovernanceArtifact:
    """Artifact representing governance snapshot."""

    run_id: str
    path: Path
    snapshot: GovernanceSnapshot
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class RunSummary:
    """Summary of a pipeline run.

    This captures the essential information about a run:
    - Success/failure status
    - Timestamps
    - Key metrics
    """

    run_id: str
    started_at: datetime
    completed_at: datetime
    success: bool
    dataset_path: str
    target_column: str
    feature_count: int
    governance_approved: bool
    governance_reason: str
    exit_code: int = 0


@dataclass(frozen=True)
class ArtifactManifest:
    """Manifest of all artifacts produced in a run.

    This is the complete record of what was produced.
    """

    run_id: str
    run_summary: RunSummary
    dataset_artifact: Optional[DatasetArtifact] = None
    schema_artifact: Optional[SchemaArtifact] = None
    quality_profile_artifact: Optional[QualityProfileArtifact] = None
    feature_provenance_artifact: Optional[FeatureProvenanceArtifact] = None
    metadata_artifact: Optional[MetadataArtifact] = None
    governance_artifact: Optional[GovernanceArtifact] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "run_id": self.run_id,
            "run_summary": {
                "run_id": self.run_summary.run_id,
                "started_at": self.run_summary.started_at.isoformat(),
                "completed_at": self.run_summary.completed_at.isoformat(),
                "success": self.run_summary.success,
                "dataset_path": self.run_summary.dataset_path,
                "target_column": self.run_summary.target_column,
                "feature_count": self.run_summary.feature_count,
                "governance_approved": self.run_summary.governance_approved,
                "governance_reason": self.run_summary.governance_reason,
                "exit_code": self.run_summary.exit_code,
            },
            "artifacts": {
                "dataset": str(self.dataset_artifact.path) if self.dataset_artifact else None,
                "schema": str(self.schema_artifact.path) if self.schema_artifact else None,
                "quality_profile": str(self.quality_profile_artifact.path) if self.quality_profile_artifact else None,
                "feature_provenance": str(self.feature_provenance_artifact.path) if self.feature_provenance_artifact else None,
                "metadata": str(self.metadata_artifact.path) if self.metadata_artifact else None,
                "governance": str(self.governance_artifact.path) if self.governance_artifact else None,
            },
            "created_at": self.created_at.isoformat(),
        }
