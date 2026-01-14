"""Level 3: Metadata & Profiling Persistence.

This module handles persistence and traceability of pipeline execution.
It captures what happened and why decisions were made, enabling audit,
debugging, and reproducibility.

Key principles:
- Deterministic only
- No LLMs or agents
- No data mutation
- Machine-readable artifacts
- Reproducibility first
"""

from .builder import MetadataBuilder
from .metadata_schema import (
    ActionRecord,
    DataQualityActions,
    DatasetSummary,
    IntentSnapshot,
    PipelineMetadata,
)
from .writer import MetadataWriter

__all__ = [
    "MetadataBuilder",
    "MetadataWriter",
    "PipelineMetadata",
    "DatasetSummary",
    "IntentSnapshot",
    "DataQualityActions",
    "ActionRecord",
]
