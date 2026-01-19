"""Feature transformation catalog for Level 4 feature engineering.

This module defines the whitelist of allowed feature transformations.
All transformations must be explicitly declared here with their metadata.
This ensures safety, explainability, and prevents arbitrary code execution.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class LeakageRisk(str, Enum):
    """Leakage risk levels for feature transformations."""

    NONE = "none"  # No leakage risk
    LOW = "low"  # Minimal leakage risk with proper validation
    MEDIUM = "medium"  # Requires careful validation
    HIGH = "high"  # High risk, requires strict safeguards


class InputType(str, Enum):
    """Required input types for transformations."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT = "text"
    ANY = "any"  # Accepts any type


class OutputType(str, Enum):
    """Output types for transformations."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


@dataclass
class FeatureTransformation:
    """Metadata for a feature transformation.

    This defines what transformations are allowed, their requirements,
    and their safety characteristics.
    """

    name: str
    description: str
    required_input_types: list[InputType]
    output_type: OutputType
    leakage_risk: LeakageRisk
    requires_target: bool = False  # True if transformation needs target column (dangerous!)
    max_interaction_order: int = 1  # Maximum number of columns for interactions
    interpretability_impact: str = "neutral"  # "improves", "neutral", "reduces"


class FeatureCatalog:
    """Catalog of allowed feature transformations.

    This is the whitelist - only transformations listed here can be proposed
    by the LLM agent. This ensures safety and explainability.
    """

    def __init__(self):
        """Initialize the feature catalog with allowed transformations."""
        self._transformations: dict[str, FeatureTransformation] = {}

        # Numeric scaling transformations
        self._register(
            FeatureTransformation(
                name="standard_scaler",
                description="Standardize numeric features (mean=0, std=1)",
                required_input_types=[InputType.NUMERIC],
                output_type=OutputType.NUMERIC,
                leakage_risk=LeakageRisk.NONE,
                interpretability_impact="neutral",
            )
        )

        self._register(
            FeatureTransformation(
                name="min_max_scaler",
                description="Scale numeric features to [0, 1] range",
                required_input_types=[InputType.NUMERIC],
                output_type=OutputType.NUMERIC,
                leakage_risk=LeakageRisk.NONE,
                interpretability_impact="neutral",
            )
        )

        # Log transforms
        self._register(
            FeatureTransformation(
                name="log",
                description="Natural logarithm transform (log(x + 1) for safety)",
                required_input_types=[InputType.NUMERIC],
                output_type=OutputType.NUMERIC,
                leakage_risk=LeakageRisk.NONE,
                interpretability_impact="reduces",
            )
        )

        self._register(
            FeatureTransformation(
                name="log10",
                description="Base-10 logarithm transform (log10(x + 1) for safety)",
                required_input_types=[InputType.NUMERIC],
                output_type=OutputType.NUMERIC,
                leakage_risk=LeakageRisk.NONE,
                interpretability_impact="reduces",
            )
        )

        # Binning transformations
        self._register(
            FeatureTransformation(
                name="bin_uniform",
                description="Bin numeric values into uniform-width bins",
                required_input_types=[InputType.NUMERIC],
                output_type=OutputType.CATEGORICAL,
                leakage_risk=LeakageRisk.NONE,
                interpretability_impact="improves",
            )
        )

        self._register(
            FeatureTransformation(
                name="bin_quantile",
                description="Bin numeric values into quantile-based bins",
                required_input_types=[InputType.NUMERIC],
                output_type=OutputType.CATEGORICAL,
                leakage_risk=LeakageRisk.LOW,  # Quantiles depend on distribution
                interpretability_impact="improves",
            )
        )

        # Categorical encoding
        self._register(
            FeatureTransformation(
                name="one_hot_encode",
                description="One-hot encoding for categorical features",
                required_input_types=[InputType.CATEGORICAL],
                output_type=OutputType.NUMERIC,
                leakage_risk=LeakageRisk.NONE,
                interpretability_impact="neutral",
            )
        )

        self._register(
            FeatureTransformation(
                name="frequency_encode",
                description="Encode categories by their frequency (target-safe)",
                required_input_types=[InputType.CATEGORICAL],
                output_type=OutputType.NUMERIC,
                leakage_risk=LeakageRisk.NONE,  # Uses only feature distribution
                interpretability_impact="neutral",
            )
        )

        # Datetime decomposition
        self._register(
            FeatureTransformation(
                name="extract_year",
                description="Extract year from datetime column",
                required_input_types=[InputType.DATETIME],
                output_type=OutputType.NUMERIC,
                leakage_risk=LeakageRisk.NONE,
                interpretability_impact="improves",
            )
        )

        self._register(
            FeatureTransformation(
                name="extract_month",
                description="Extract month from datetime column",
                required_input_types=[InputType.DATETIME],
                output_type=OutputType.NUMERIC,
                leakage_risk=LeakageRisk.NONE,
                interpretability_impact="improves",
            )
        )

        self._register(
            FeatureTransformation(
                name="extract_day_of_week",
                description="Extract day of week from datetime column",
                required_input_types=[InputType.DATETIME],
                output_type=OutputType.NUMERIC,
                leakage_risk=LeakageRisk.NONE,
                interpretability_impact="improves",
            )
        )

        self._register(
            FeatureTransformation(
                name="extract_hour",
                description="Extract hour from datetime column",
                required_input_types=[InputType.DATETIME],
                output_type=OutputType.NUMERIC,
                leakage_risk=LeakageRisk.NONE,
                interpretability_impact="improves",
            )
        )

        # Interaction terms (bounded)
        self._register(
            FeatureTransformation(
                name="multiply",
                description="Multiply two numeric features",
                required_input_types=[InputType.NUMERIC, InputType.NUMERIC],
                output_type=OutputType.NUMERIC,
                leakage_risk=LeakageRisk.NONE,
                max_interaction_order=2,
                interpretability_impact="reduces",
            )
        )

        self._register(
            FeatureTransformation(
                name="divide",
                description="Divide two numeric features (with zero protection)",
                required_input_types=[InputType.NUMERIC, InputType.NUMERIC],
                output_type=OutputType.NUMERIC,
                leakage_risk=LeakageRisk.NONE,
                max_interaction_order=2,
                interpretability_impact="reduces",
            )
        )

        self._register(
            FeatureTransformation(
                name="add",
                description="Add two numeric features",
                required_input_types=[InputType.NUMERIC, InputType.NUMERIC],
                output_type=OutputType.NUMERIC,
                leakage_risk=LeakageRisk.NONE,
                max_interaction_order=2,
                interpretability_impact="neutral",
            )
        )

        self._register(
            FeatureTransformation(
                name="subtract",
                description="Subtract two numeric features",
                required_input_types=[InputType.NUMERIC, InputType.NUMERIC],
                output_type=OutputType.NUMERIC,
                leakage_risk=LeakageRisk.NONE,
                max_interaction_order=2,
                interpretability_impact="neutral",
            )
        )

        # Polynomial features (limited to quadratic)
        self._register(
            FeatureTransformation(
                name="square",
                description="Square a numeric feature (x^2)",
                required_input_types=[InputType.NUMERIC],
                output_type=OutputType.NUMERIC,
                leakage_risk=LeakageRisk.NONE,
                interpretability_impact="reduces",
            )
        )

        # Missing value indicators
        self._register(
            FeatureTransformation(
                name="missing_indicator",
                description="Create binary indicator for missing values",
                required_input_types=[InputType.ANY],
                output_type=OutputType.BOOLEAN,
                leakage_risk=LeakageRisk.NONE,
                interpretability_impact="improves",
            )
        )

    def _register(self, transformation: FeatureTransformation) -> None:
        """Register a transformation in the catalog.

        Args:
            transformation: FeatureTransformation to register
        """
        self._transformations[transformation.name] = transformation

    def get(self, name: str) -> Optional[FeatureTransformation]:
        """Get a transformation by name.

        Args:
            name: Name of the transformation

        Returns:
            FeatureTransformation if found, None otherwise
        """
        return self._transformations.get(name)

    def list_all(self) -> list[FeatureTransformation]:
        """List all registered transformations.

        Returns:
            List of all FeatureTransformation instances
        """
        return list(self._transformations.values())

    def get_summary(self) -> dict:
        """Get a summary of the catalog for LLM consumption.

        Returns:
            Dictionary with transformation summaries
        """
        summary = {}
        for name, trans in self._transformations.items():
            summary[name] = {
                "description": trans.description,
                "input_types": [t.value for t in trans.required_input_types],
                "output_type": trans.output_type.value,
                "leakage_risk": trans.leakage_risk.value,
                "max_interaction_order": trans.max_interaction_order,
            }
        return summary

    def is_valid_transformation(self, name: str) -> bool:
        """Check if a transformation name is valid.

        Args:
            name: Transformation name to check

        Returns:
            True if transformation exists in catalog
        """
        return name in self._transformations


# Global catalog instance
_catalog = None


def get_catalog() -> FeatureCatalog:
    """Get the global feature catalog instance.

    Returns:
        FeatureCatalog singleton instance
    """
    global _catalog
    if _catalog is None:
        _catalog = FeatureCatalog()
    return _catalog
