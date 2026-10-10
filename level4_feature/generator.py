"""Deterministic feature generator for Level 4 feature engineering.

This module applies validated feature transformations to create new features.
All transformations are deterministic and safe.
"""

from dataclasses import dataclass, replace
from typing import Optional

import pandas as pd

from core.recipe import apply_step, fit_step
from level1_ingestion.normalizer import to_snake_case
from utils import get_logger

from .feature_catalog import get_catalog
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

    def __init__(self, df: pd.DataFrame, raw: Optional[pd.DataFrame] = None):
        """Initialize the feature generator.

        Args:
            df: Input DataFrame (will not be modified)
            raw: Data before Level 2 cleaning; missing indicators read from it
        """
        self.df = df.copy()  # Work on copy to avoid modifying original
        self.raw = raw
        self.steps: list[dict] = []  # fitted, replayable on new data (core.recipe)
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

        step = fit_step(feature.transformation, self.df, feature.source_columns, name=feature.name)
        self.df = apply_step(self.df, step, self.raw)
        self.steps.append(step)

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
