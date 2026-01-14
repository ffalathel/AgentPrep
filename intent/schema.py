"""Intent schema definitions using Pydantic.

This module defines the strict, immutable policy contract for user intent.
All user inputs must conform to these schemas with no free-form text fields.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskType(str, Enum):
    """Supported ML task types."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    CLUSTERING = "clustering"


class ModelFamily(str, Enum):
    """Supported model families."""

    TREE = "tree"
    LINEAR = "linear"
    NEURAL = "neural"
    UNKNOWN = "unknown"


class OutlierPolicy(str, Enum):
    """Policy for handling outliers."""

    PRESERVE = "preserve"
    CLIP = "clip"
    FLAG = "flag"


class InterpretabilityPriority(str, Enum):
    """Priority level for interpretability."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskConfig(BaseModel):
    """Task configuration."""

    type: TaskType = Field(..., description="Type of ML task")
    target_column: str = Field(..., min_length=1, description="Name of target column")

    @field_validator("target_column")
    @classmethod
    def validate_target_column(cls, v: str) -> str:
        """Validate target column name."""
        if not v or not v.strip():
            raise ValueError("target_column cannot be empty")
        return v.strip()


class ModelConfig(BaseModel):
    """Model configuration."""

    family: ModelFamily = Field(..., description="Model family to optimize for")


class PreferencesConfig(BaseModel):
    """User preferences for preprocessing."""

    outlier_policy: OutlierPolicy = Field(..., description="Policy for handling outliers")
    allow_column_dropping: bool = Field(
        default=False, description="Whether columns can be dropped"
    )
    interpretability_priority: InterpretabilityPriority = Field(
        ..., description="Priority level for interpretability"
    )


class ConstraintsConfig(BaseModel):
    """Constraints for preprocessing."""

    max_features: int = Field(
        ..., ge=1, le=10000, description="Maximum number of features"
    )
    max_interactions: int = Field(
        ..., ge=0, le=1000, description="Maximum number of feature interactions"
    )
    max_cardinality: int = Field(
        ..., ge=2, le=1000000, description="Maximum cardinality for categorical features"
    )


class IntentSchema(BaseModel):
    """Complete intent schema - immutable policy contract.

    This schema represents the user's intent and must be validated before
    any pipeline execution. Once validated, it is read-only.
    """

    dataset: dict[str, str] = Field(..., description="Dataset configuration")
    task: TaskConfig = Field(..., description="Task configuration")
    model: ModelConfig = Field(..., description="Model configuration")
    preferences: PreferencesConfig = Field(..., description="User preferences")
    constraints: ConstraintsConfig = Field(..., description="Processing constraints")

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, v: dict) -> dict:
        """Validate dataset configuration."""
        if "path" not in v:
            raise ValueError("dataset.path is required")
        if not v["path"] or not str(v["path"]).strip():
            raise ValueError("dataset.path cannot be empty")
        return v

    @property
    def dataset_path(self) -> str:
        """Get dataset path."""
        return str(self.dataset["path"]).strip()

    model_config = ConfigDict(
        frozen=True,  # Make intent immutable after validation
        extra="forbid",  # Reject extra fields
    )
