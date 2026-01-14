"""Metadata schemas for Level 3 persistence.

This module defines structured schemas for pipeline metadata that enable
traceability, auditability, and reproducibility.

All schemas are:
- Explicit and machine-readable
- Serializable to JSON
- Versionable
- Deterministic
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

from intent.schema import IntentSchema
from level1_ingestion.schema_inferencer import SchemaMetadata
from level2_quality.executor import AppliedAction
from level2_quality.profiler import DatasetQualityProfile


@dataclass
class DatasetSummary:
    """Summary statistics for the processed dataset."""

    row_count: int
    column_count: int
    target_column: str
    normalized_target_column: str


@dataclass
class ActionRecord:
    """Record of a single data quality action."""

    column: str
    action: str
    method: Optional[str]
    justification: Optional[str]
    status: str  # "applied" or "rejected"
    reason: Optional[str] = None


@dataclass
class DataQualityActions:
    """Records of all data quality actions."""

    proposed_count: int
    applied_count: int
    rejected_count: int
    proposed_actions: list[dict[str, Any]] = field(default_factory=list)
    applied_actions: list[ActionRecord] = field(default_factory=list)
    rejected_actions: list[ActionRecord] = field(default_factory=list)


@dataclass
class IntentSnapshot:
    """Read-only snapshot of user intent.

    This captures the exact intent configuration used for this pipeline run.
    """

    dataset_path: str
    task_type: str
    target_column: str
    model_family: str
    outlier_policy: str
    allow_column_dropping: bool
    interpretability_priority: str
    max_features: int
    max_interactions: int
    max_cardinality: int

    @classmethod
    def from_intent(cls, intent: IntentSchema) -> "IntentSnapshot":
        """Create an intent snapshot from an IntentSchema."""
        return cls(
            dataset_path=intent.dataset_path,
            task_type=intent.task.type.value,
            target_column=intent.task.target_column,
            model_family=intent.model.family.value,
            outlier_policy=intent.preferences.outlier_policy.value,
            allow_column_dropping=intent.preferences.allow_column_dropping,
            interpretability_priority=intent.preferences.interpretability_priority.value,
            max_features=intent.constraints.max_features,
            max_interactions=intent.constraints.max_interactions,
            max_cardinality=intent.constraints.max_cardinality,
        )


@dataclass
class PipelineMetadata:
    """Complete metadata for a pipeline run.

    This is the root schema that aggregates all metadata from Levels 0-2.
    """

    # Dataset summary
    dataset_summary: DatasetSummary

    # Intent snapshot
    intent: IntentSnapshot

    # Schema metadata (from Level 1)
    schema_metadata: dict[str, Any]  # Serialized SchemaMetadata

    # Quality profile (from Level 2)
    quality_profile: dict[str, Any]  # Serialized DatasetQualityProfile

    # Data quality actions (from Level 2)
    quality_actions: DataQualityActions

    # Pipeline identification (with defaults, must come last)
    version: str = "1.0.0"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize metadata to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


def serialize_schema_metadata(schema: SchemaMetadata) -> dict[str, Any]:
    """Serialize SchemaMetadata to dictionary."""
    return {
        "total_rows": schema.total_rows,
        "total_columns": schema.total_columns,
        "columns": {
            col_name: {
                "name": col_meta.name,
                "semantic_type": col_meta.semantic_type,
                "missing_count": col_meta.missing_count,
                "missing_percentage": col_meta.missing_percentage,
                "unique_count": col_meta.unique_count,
                "pandas_dtype": col_meta.pandas_dtype,
                "total_count": col_meta.total_count,
            }
            for col_name, col_meta in schema.columns.items()
        },
    }


def serialize_quality_profile(profile: DatasetQualityProfile) -> dict[str, Any]:
    """Serialize DatasetQualityProfile to dictionary."""
    columns_dict = {}
    for col_name, col_profile in profile.columns.items():
        col_dict = {
            "column_name": col_profile.column_name,
            "missing_count": col_profile.missing_count,
            "missing_percentage": col_profile.missing_percentage,
            "unique_count": col_profile.unique_count,
            "is_constant": bool(col_profile.is_constant),
            "is_near_constant": bool(col_profile.is_near_constant),
        }

        if col_profile.numeric_stats:
            col_dict["numeric_stats"] = {
                "min": col_profile.numeric_stats.min,
                "max": col_profile.numeric_stats.max,
                "mean": col_profile.numeric_stats.mean,
                "median": col_profile.numeric_stats.median,
                "std": col_profile.numeric_stats.std,
                "q25": col_profile.numeric_stats.q25,
                "q75": col_profile.numeric_stats.q75,
                "iqr": col_profile.numeric_stats.iqr,
                "outlier_count_lower": col_profile.numeric_stats.outlier_count_lower,
                "outlier_count_upper": col_profile.numeric_stats.outlier_count_upper,
                "outlier_count_total": col_profile.numeric_stats.outlier_count_total,
            }

        columns_dict[col_name] = col_dict

    return {
        "total_rows": profile.total_rows,
        "total_columns": profile.total_columns,
        "columns": columns_dict,
    }
