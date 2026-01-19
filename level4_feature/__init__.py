"""Level 4: Feature Engineering Agent.

This module handles agent-guided feature engineering with strict safety controls.
The LLM proposes features, but all execution is deterministic and validated.
"""

from .agent import FeatureAgentError, FeatureEngineeringAgent
from .feature_catalog import (
    FeatureCatalog,
    FeatureTransformation,
    InputType,
    LeakageRisk,
    OutputType,
    get_catalog,
)
from .generator import FeatureGenerationError, FeatureGenerator, FeatureProvenance
from .validator import (
    FeatureValidationError,
    FeatureValidator,
    RejectedFeature,
    ValidatedFeature,
)

__all__ = [
    "FeatureAgentError",
    "FeatureEngineeringAgent",
    "FeatureCatalog",
    "FeatureTransformation",
    "InputType",
    "LeakageRisk",
    "OutputType",
    "get_catalog",
    "FeatureGenerationError",
    "FeatureGenerator",
    "FeatureProvenance",
    "FeatureValidationError",
    "FeatureValidator",
    "RejectedFeature",
    "ValidatedFeature",
]
