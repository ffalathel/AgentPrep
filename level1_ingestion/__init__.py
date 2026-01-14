"""Level 1: Data Ingestion & Schema Normalization.

This module handles loading datasets, inferring schema metadata,
and normalizing column names and types.
"""

from .loader import DatasetLoadError, load_dataset
from .normalizer import NormalizationError, normalize_dataset
from .schema_inferencer import (
    ColumnMetadata,
    SchemaMetadata,
    SemanticType,
    infer_schema,
)

__all__ = [
    "load_dataset",
    "DatasetLoadError",
    "infer_schema",
    "SchemaMetadata",
    "ColumnMetadata",
    "SemanticType",
    "normalize_dataset",
    "NormalizationError",
]
