"""Level 6: Artifacts, Storage, and Reporting.

This module records and outputs pipeline artifacts, including datasets,
feature transformations, metadata, and governance snapshots.

It does not modify data, does not enforce policy, and does not use LLMs or agents.
Its purpose is to capture, store, and summarize pipeline results in a
reproducible and auditable way.
"""

from .artifact_registry import ArtifactRegistry
from .exporter import ArtifactExporter, ExportError
from .report_generator import ReportGenerationError, ReportGenerator
from .schemas import (
    ArtifactManifest,
    DatasetArtifact,
    FeatureProvenanceArtifact,
    GovernanceArtifact,
    GovernanceSnapshot,
    MetadataArtifact,
    QualityProfileArtifact,
    RunSummary,
    SchemaArtifact,
)
from .store import ArtifactStore, ArtifactStoreError

__all__ = [
    "ArtifactRegistry",
    "ArtifactExporter",
    "ExportError",
    "ReportGenerator",
    "ReportGenerationError",
    "ArtifactManifest",
    "DatasetArtifact",
    "FeatureProvenanceArtifact",
    "GovernanceArtifact",
    "GovernanceSnapshot",
    "MetadataArtifact",
    "QualityProfileArtifact",
    "RunSummary",
    "SchemaArtifact",
    "ArtifactStore",
    "ArtifactStoreError",
]
