"""Deterministic feature generator for Level 4 feature engineering.

This module applies validated feature transformations to create new features.
All transformations are deterministic and safe.
"""

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd

from level1_ingestion.normalizer import to_snake_case
from utils import get_logger

from .feature_catalog import FeatureCatalog, get_catalog
from .validator import ValidatedFeature

logger = get_logger(__name__)


class FeatureGenerationError(Exception):
    """Raised when feature generation fails."""

    pass


@dataclass
class FeatureProvenance:
    """Provenance information for a generated feature.

    Tracks how a feature was created for auditability.
    """

    feature_name: str
    source_columns: list[str]
    transformation: str
    reason: Optional[str] = None


class FeatureGenerator:
    """Deterministic feature generator.

    Applies validated transformations to create new features.
    All operations are deterministic and safe.
    """

    def __init__(self, df: pd.DataFrame):
        """Initialize the feature generator.

        Args:
            df: Input DataFrame (will not be modified)
        """
        self.df = df.copy()  # Work on copy to avoid modifying original
        self.catalog = get_catalog()
        self.provenance: list[FeatureProvenance] = []
        logger.debug("FeatureGenerator initialized")

    def generate_features(
        self, validated_features: list[ValidatedFeature]
    ) -> pd.DataFrame:
        """Generate features from validated proposals.

        Args:
            validated_features: List of validated feature proposals

        Returns:
            DataFrame with original columns plus new features

        Raises:
            FeatureGenerationError: If feature generation fails
        """
        logger.info(f"Generating {len(validated_features)} features")

        for feature in validated_features:
            try:
                # Resolve source columns against the current dataframe (handle normalization and case)
                resolved_feature = self._resolve_source_columns(feature)
                if resolved_feature is None:
                    logger.warning(
                        "Skipping feature %s because one or more source columns "
                        "are missing after normalization/cleaning: %s",
                        feature.name,
                        feature.source_columns,
                    )
                    continue

                self._apply_transformation(resolved_feature)
                self.provenance.append(
                    FeatureProvenance(
                        feature_name=resolved_feature.name,
                        source_columns=resolved_feature.source_columns,
                        transformation=resolved_feature.transformation,
                        reason=resolved_feature.reason,
                    )
                )
                logger.debug(f"Generated feature: {resolved_feature.name}")
            except FeatureGenerationError as e:
                # Critical generation issue – abort pipeline
                logger.error(f"Failed to generate feature {feature.name}: {e}")
                raise
            except (KeyError, ValueError) as e:
                logger.error(f"Failed to generate feature {feature.name}: Invalid column or transformation: {e}")
                raise FeatureGenerationError(
                    f"Failed to generate feature {feature.name}: Invalid column or transformation: {e}"
                ) from e
            except (TypeError, AttributeError) as e:
                logger.error(f"Failed to generate feature {feature.name}: Invalid data type or attribute: {e}")
                raise FeatureGenerationError(
                    f"Failed to generate feature {feature.name}: Invalid data type or attribute: {e}"
                ) from e
            except Exception as e:
                logger.error(f"Failed to generate feature {feature.name}: Unexpected error: {e}")
                raise FeatureGenerationError(
                    f"Failed to generate feature {feature.name}: Unexpected error: {e}"
                ) from e

        logger.info(f"Feature generation complete: {len(self.provenance)} features created")
        return self.df

    def _apply_transformation(self, feature: ValidatedFeature) -> None:
        """Apply a single transformation to create a feature.

        Args:
            feature: Validated feature proposal

        Raises:
            FeatureGenerationError: If transformation fails
        """
        trans_meta = self.catalog.get(feature.transformation)
        if trans_meta is None:
            raise FeatureGenerationError(
                f"Transformation '{feature.transformation}' not found in catalog"
            )

        # Route to appropriate transformation function
        if feature.transformation == "standard_scaler":
            self._apply_standard_scaler(feature)
        elif feature.transformation == "min_max_scaler":
            self._apply_min_max_scaler(feature)
        elif feature.transformation == "log":
            self._apply_log(feature)
        elif feature.transformation == "log10":
            self._apply_log10(feature)
        elif feature.transformation == "bin_uniform":
            self._apply_bin_uniform(feature)
        elif feature.transformation == "bin_quantile":
            self._apply_bin_quantile(feature)
        elif feature.transformation == "one_hot_encode":
            self._apply_one_hot_encode(feature)
        elif feature.transformation == "frequency_encode":
            self._apply_frequency_encode(feature)
        elif feature.transformation == "extract_year":
            self._apply_extract_year(feature)
        elif feature.transformation == "extract_month":
            self._apply_extract_month(feature)
        elif feature.transformation == "extract_day_of_week":
            self._apply_extract_day_of_week(feature)
        elif feature.transformation == "extract_hour":
            self._apply_extract_hour(feature)
        elif feature.transformation == "multiply":
            self._apply_multiply(feature)
        elif feature.transformation == "divide":
            self._apply_divide(feature)
        elif feature.transformation == "add":
            self._apply_add(feature)
        elif feature.transformation == "subtract":
            self._apply_subtract(feature)
        elif feature.transformation == "square":
            self._apply_square(feature)
        elif feature.transformation == "missing_indicator":
            self._apply_missing_indicator(feature)
        else:
            raise FeatureGenerationError(
                f"Transformation '{feature.transformation}' not implemented"
            )

    def _resolve_source_columns(
        self, feature: ValidatedFeature
    ) -> Optional[ValidatedFeature]:
        """Resolve feature.source_columns against current dataframe columns.

        Handles:
        - Normalized column names (snake_case)
        - Case differences (e.g., 'Close' vs 'close')
        - Dropped columns (returns None to signal skip)
        """
        resolved: list[str] = []
        df_cols = list(self.df.columns)
        lower_map = {c.lower(): c for c in df_cols}

        for col in feature.source_columns:
            # Exact match
            if col in df_cols:
                resolved.append(col)
                continue

            # Case-insensitive match
            normalized = col.lower()
            if normalized in lower_map:
                resolved.append(lower_map[normalized])
                continue

            # Normalization-aware match (snake_case, as in Level 1)
            snake = to_snake_case(col)
            if snake in df_cols:
                resolved.append(snake)
                continue
            if snake.lower() in lower_map:
                resolved.append(lower_map[snake.lower()])
                continue

            # Column truly missing – likely dropped earlier
            logger.debug(
                "Source column '%s' for feature '%s' not found after normalization/cleaning",
                col,
                feature.name,
            )
            return None

        # Return a copy of feature with resolved column names
        return replace(feature, source_columns=resolved)

    # Numeric scaling transformations

    def _apply_standard_scaler(self, feature: ValidatedFeature) -> None:
        """Apply standard scaling (z-score normalization)."""
        col = feature.source_columns[0]
        series = self.df[col]
        mean = series.mean()
        std = series.std()
        if std == 0:
            # Constant column - set to 0
            self.df[feature.name] = 0.0
        else:
            self.df[feature.name] = (series - mean) / std

    def _apply_min_max_scaler(self, feature: ValidatedFeature) -> None:
        """Apply min-max scaling to [0, 1] range."""
        col = feature.source_columns[0]
        series = self.df[col]
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            # Constant column - set to 0
            self.df[feature.name] = 0.0
        else:
            self.df[feature.name] = (series - min_val) / (max_val - min_val)

    # Log transforms

    def _apply_log(self, feature: ValidatedFeature) -> None:
        """Apply natural logarithm (log(x + 1) for safety)."""
        col = feature.source_columns[0]
        series = self.df[col]
        # Use log1p to handle zeros and negatives safely
        self.df[feature.name] = np.log1p(series - series.min())

    def _apply_log10(self, feature: ValidatedFeature) -> None:
        """Apply base-10 logarithm (log10(x + 1) for safety)."""
        col = feature.source_columns[0]
        series = self.df[col]
        # Use log1p equivalent for base 10
        shifted = series - series.min()
        self.df[feature.name] = np.log10(shifted + 1)

    # Binning transformations

    def _apply_bin_uniform(self, feature: ValidatedFeature) -> None:
        """Apply uniform-width binning."""
        col = feature.source_columns[0]
        series = self.df[col]
        # Use 10 bins by default
        self.df[feature.name] = pd.cut(series, bins=10, labels=False, duplicates="drop")

    def _apply_bin_quantile(self, feature: ValidatedFeature) -> None:
        """Apply quantile-based binning."""
        col = feature.source_columns[0]
        series = self.df[col]
        # Use 10 quantile bins
        self.df[feature.name] = pd.qcut(
            series, q=10, labels=False, duplicates="drop"
        )

    # Categorical encoding

    def _apply_one_hot_encode(self, feature: ValidatedFeature) -> None:
        """Apply one-hot encoding (creates multiple columns)."""
        col = feature.source_columns[0]
        series = self.df[col]
        # Get dummies
        dummies = pd.get_dummies(series, prefix=feature.name)
        # Add to dataframe
        for dummy_col in dummies.columns:
            self.df[dummy_col] = dummies[dummy_col]

    def _apply_frequency_encode(self, feature: ValidatedFeature) -> None:
        """Apply frequency encoding (target-safe)."""
        col = feature.source_columns[0]
        series = self.df[col]
        # Count frequencies
        freq_map = series.value_counts().to_dict()
        # Map to frequencies
        self.df[feature.name] = series.map(freq_map)

    # Datetime decomposition

    def _apply_extract_year(self, feature: ValidatedFeature) -> None:
        """Extract year from datetime column."""
        col = feature.source_columns[0]
        series = pd.to_datetime(self.df[col])
        self.df[feature.name] = series.dt.year

    def _apply_extract_month(self, feature: ValidatedFeature) -> None:
        """Extract month from datetime column."""
        col = feature.source_columns[0]
        series = pd.to_datetime(self.df[col])
        self.df[feature.name] = series.dt.month

    def _apply_extract_day_of_week(self, feature: ValidatedFeature) -> None:
        """Extract day of week from datetime column."""
        col = feature.source_columns[0]
        series = pd.to_datetime(self.df[col])
        self.df[feature.name] = series.dt.dayofweek

    def _apply_extract_hour(self, feature: ValidatedFeature) -> None:
        """Extract hour from datetime column."""
        col = feature.source_columns[0]
        series = pd.to_datetime(self.df[col])
        self.df[feature.name] = series.dt.hour

    # Interaction terms

    def _apply_multiply(self, feature: ValidatedFeature) -> None:
        """Multiply two numeric features."""
        col1, col2 = feature.source_columns[0], feature.source_columns[1]
        self.df[feature.name] = self.df[col1] * self.df[col2]

    def _apply_divide(self, feature: ValidatedFeature) -> None:
        """Divide two numeric features (with zero protection)."""
        col1, col2 = feature.source_columns[0], feature.source_columns[1]
        divisor = self.df[col2]
        # Protect against division by zero
        divisor = divisor.replace(0, np.nan)
        self.df[feature.name] = self.df[col1] / divisor
        # Fill NaN with 0 (or could use a large number)
        self.df[feature.name] = self.df[feature.name].fillna(0)

    def _apply_add(self, feature: ValidatedFeature) -> None:
        """Add two numeric features."""
        col1, col2 = feature.source_columns[0], feature.source_columns[1]
        self.df[feature.name] = self.df[col1] + self.df[col2]

    def _apply_subtract(self, feature: ValidatedFeature) -> None:
        """Subtract two numeric features."""
        col1, col2 = feature.source_columns[0], feature.source_columns[1]
        self.df[feature.name] = self.df[col1] - self.df[col2]

    def _apply_square(self, feature: ValidatedFeature) -> None:
        """Square a numeric feature."""
        col = feature.source_columns[0]
        self.df[feature.name] = self.df[col] ** 2

    def _apply_missing_indicator(self, feature: ValidatedFeature) -> None:
        """Create binary indicator for missing values."""
        col = feature.source_columns[0]
        self.df[feature.name] = self.df[col].isna().astype(int)
