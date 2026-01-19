"""Feature proposal validator for Level 4 feature engineering.

This module provides critical safety validation for LLM-proposed features.
All proposals must pass validation before being applied.
"""

from dataclasses import dataclass
from typing import Any, Optional

from intent.schema import IntentSchema, InterpretabilityPriority, ModelFamily
from level1_ingestion.schema_inferencer import SchemaMetadata
from utils import get_logger

from .feature_catalog import FeatureCatalog, get_catalog, InputType

logger = get_logger(__name__)


class FeatureValidationError(Exception):
    """Raised when feature validation fails."""

    pass


@dataclass
class ValidatedFeature:
    """A validated feature proposal.

    This represents a feature that has passed all safety checks
    and is ready to be generated.
    """

    name: str
    source_columns: list[str]
    transformation: str
    reason: Optional[str] = None


@dataclass
class RejectedFeature:
    """A rejected feature proposal with reason."""

    name: str
    source_columns: list[str]
    transformation: str
    reason: str


class FeatureValidator:
    """Validates feature engineering proposals for safety and compliance.

    This validator ensures:
    - All transformations exist in catalog
    - Source columns exist
    - Target column is never used as input
    - No feature leakage
    - Feature count limits are enforced
    - Transformations respect intent constraints
    """

    def __init__(
        self,
        schema_metadata: SchemaMetadata,
        intent: IntentSchema,
        target_column: str,
    ):
        """Initialize the feature validator.

        Args:
            schema_metadata: Schema metadata from Level 1
            intent: User intent (read-only)
            target_column: Normalized target column name
        """
        self.schema_metadata = schema_metadata
        self.intent = intent
        self.target_column = target_column
        self.catalog = get_catalog()
        logger.debug("FeatureValidator initialized")

    def validate_proposals(
        self, proposals: list[dict[str, Any]]
    ) -> tuple[list[ValidatedFeature], list[RejectedFeature]]:
        """Validate feature proposals.

        Args:
            proposals: List of feature proposal dictionaries

        Returns:
            Tuple of (validated_features, rejected_features)
        """
        validated = []
        rejected = []

        # Track feature counts
        total_features = len(self.schema_metadata.columns) - 1  # Exclude target
        interaction_count = 0

        for proposal in proposals:
            try:
                # Extract proposal fields
                name = proposal.get("name", "")
                source_columns = proposal.get("source_columns", [])
                transformation = proposal.get("transformation", "")
                reason = proposal.get("reason", "")

                # Validate transformation exists in catalog
                if not self.catalog.is_valid_transformation(transformation):
                    rejected.append(
                        RejectedFeature(
                            name=name,
                            source_columns=source_columns,
                            transformation=transformation,
                            reason=f"Transformation '{transformation}' not in catalog",
                        )
                    )
                    continue

                # Validate source columns exist
                missing_columns = [
                    col
                    for col in source_columns
                    if col not in self.schema_metadata.columns
                ]
                if missing_columns:
                    rejected.append(
                        RejectedFeature(
                            name=name,
                            source_columns=source_columns,
                            transformation=transformation,
                            reason=f"Source columns not found: {missing_columns}",
                        )
                    )
                    continue

                # CRITICAL: Target column must never be used as input
                if self.target_column in source_columns:
                    rejected.append(
                        RejectedFeature(
                            name=name,
                            source_columns=source_columns,
                            transformation=transformation,
                            reason=f"Target column '{self.target_column}' cannot be used as input",
                        )
                    )
                    continue

                # Validate input types match transformation requirements
                trans_meta = self.catalog.get(transformation)
                if trans_meta is None:
                    rejected.append(
                        RejectedFeature(
                            name=name,
                            source_columns=source_columns,
                            transformation=transformation,
                            reason=f"Transformation metadata not found",
                        )
                    )
                    continue

                # Check input type compatibility
                if not self._validate_input_types(source_columns, trans_meta):
                    rejected.append(
                        RejectedFeature(
                            name=name,
                            source_columns=source_columns,
                            transformation=transformation,
                            reason=f"Input types incompatible with transformation requirements",
                        )
                    )
                    continue

                # Validate interaction order
                if len(source_columns) > trans_meta.max_interaction_order:
                    rejected.append(
                        RejectedFeature(
                            name=name,
                            source_columns=source_columns,
                            transformation=transformation,
                            reason=f"Too many source columns ({len(source_columns)} > {trans_meta.max_interaction_order})",
                        )
                    )
                    continue

                # Check interaction count limit
                if len(source_columns) > 1:
                    interaction_count += 1
                    if interaction_count > self.intent.constraints.max_interactions:
                        rejected.append(
                            RejectedFeature(
                                name=name,
                                source_columns=source_columns,
                                transformation=transformation,
                                reason=f"Exceeds max_interactions limit ({self.intent.constraints.max_interactions})",
                            )
                        )
                        interaction_count -= 1  # Don't count this one
                        continue

                # Validate interpretability constraints
                if not self._validate_interpretability(trans_meta):
                    rejected.append(
                        RejectedFeature(
                            name=name,
                            source_columns=source_columns,
                            transformation=transformation,
                            reason=f"Transformation reduces interpretability but priority is high",
                        )
                    )
                    continue

                # Validate model family compatibility
                if not self._validate_model_family(trans_meta):
                    rejected.append(
                        RejectedFeature(
                            name=name,
                            source_columns=source_columns,
                            transformation=transformation,
                            reason=f"Transformation not suitable for model family {self.intent.model.family}",
                        )
                    )
                    continue

                # Check total feature count limit
                total_features += 1
                if total_features > self.intent.constraints.max_features:
                    rejected.append(
                        RejectedFeature(
                            name=name,
                            source_columns=source_columns,
                            transformation=transformation,
                            reason=f"Would exceed max_features limit ({self.intent.constraints.max_features})",
                        )
                    )
                    total_features -= 1  # Don't count this one
                    continue

                # All checks passed - validate
                validated.append(
                    ValidatedFeature(
                        name=name,
                        source_columns=source_columns,
                        transformation=transformation,
                        reason=reason,
                    )
                )

            except Exception as e:
                logger.warning(f"Error validating proposal {proposal}: {e}")
                rejected.append(
                    RejectedFeature(
                        name=proposal.get("name", "unknown"),
                        source_columns=proposal.get("source_columns", []),
                        transformation=proposal.get("transformation", ""),
                        reason=f"Validation error: {e}",
                    )
                )

        logger.info(
            f"Feature validation: {len(validated)} validated, {len(rejected)} rejected"
        )
        return validated, rejected

    def _validate_input_types(
        self, source_columns: list[str], trans_meta: Any
    ) -> bool:
        """Validate that source column types match transformation requirements.

        Args:
            source_columns: List of source column names
            trans_meta: FeatureTransformation metadata

        Returns:
            True if input types are compatible
        """
        if InputType.ANY in trans_meta.required_input_types:
            return True

        if len(source_columns) != len(trans_meta.required_input_types):
            return False

        for col_name, required_type in zip(
            source_columns, trans_meta.required_input_types
        ):
            col_meta = self.schema_metadata.columns.get(col_name)
            if col_meta is None:
                return False

            # Map semantic type to InputType
            semantic_type = col_meta.semantic_type
            if required_type == InputType.NUMERIC:
                if semantic_type not in ["numeric"]:
                    return False
            elif required_type == InputType.CATEGORICAL:
                if semantic_type not in ["categorical"]:
                    return False
            elif required_type == InputType.DATETIME:
                if semantic_type not in ["datetime"]:
                    return False
            elif required_type == InputType.BOOLEAN:
                if semantic_type not in ["boolean"]:
                    return False
            elif required_type == InputType.TEXT:
                if semantic_type not in ["text"]:
                    return False

        return True

    def _validate_interpretability(self, trans_meta: Any) -> bool:
        """Validate that transformation respects interpretability priority.

        Args:
            trans_meta: FeatureTransformation metadata

        Returns:
            True if transformation is allowed given interpretability priority
        """
        if self.intent.preferences.interpretability_priority == InterpretabilityPriority.HIGH:
            # High interpretability: reject transformations that reduce interpretability
            return trans_meta.interpretability_impact != "reduces"

        # Medium or low: allow all transformations
        return True

    def _validate_model_family(self, trans_meta: Any) -> bool:
        """Validate that transformation is suitable for model family.

        Args:
            trans_meta: FeatureTransformation metadata

        Returns:
            True if transformation is suitable
        """
        # Tree models benefit from binning and don't need scaling
        if self.intent.model.family == ModelFamily.TREE:
            # Prefer binning over scaling for tree models
            if trans_meta.name in ["standard_scaler", "min_max_scaler"]:
                # Scaling is less useful for trees but not harmful
                return True
            return True  # All other transformations are fine

        # Linear models benefit from scaling
        if self.intent.model.family == ModelFamily.LINEAR:
            return True  # All transformations are fine

        # Neural models benefit from scaling
        if self.intent.model.family == ModelFamily.NEURAL:
            return True  # All transformations are fine

        # Unknown: allow all
        return True
