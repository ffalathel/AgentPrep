"""Metadata builder for Level 3.

This module aggregates metadata from Levels 0-2 into a single
structured metadata object for persistence.

This is a pure aggregation layer with no I/O operations.
"""

import logging
from typing import Optional

from intent.schema import IntentSchema
from level1_ingestion.schema_inferencer import SchemaMetadata
from level2_quality.executor import AppliedAction
from level2_quality.profiler import DatasetQualityProfile

from .metadata_schema import (
    ActionRecord,
    DataQualityActions,
    DatasetSummary,
    IntentSnapshot,
    PipelineMetadata,
    serialize_quality_profile,
    serialize_schema_metadata,
)

logger = logging.getLogger(__name__)


class MetadataBuilder:
    """Builds pipeline metadata from Levels 0-2 outputs.

    This builder:
    - Collects metadata from all pipeline levels
    - Aggregates into a single structured object
    - Performs no I/O operations
    - Is deterministic and pure
    """

    def __init__(self):
        """Initialize the metadata builder."""
        logger.debug("MetadataBuilder initialized")

    def build(
        self,
        intent: IntentSchema,
        schema_metadata: SchemaMetadata,
        quality_profile: DatasetQualityProfile,
        applied_actions: list[AppliedAction],
        proposed_actions: list[dict],
        normalized_dataframe_row_count: int,
        normalized_dataframe_column_count: int,
        normalized_target_column: str,
    ) -> PipelineMetadata:
        """Build complete pipeline metadata.

        Args:
            intent: User intent from Level 0
            schema_metadata: Schema metadata from Level 1
            quality_profile: Quality profile from Level 2
            applied_actions: List of applied/rejected actions from Level 2
            proposed_actions: List of proposed actions from Level 2 agent
            normalized_dataframe_row_count: Row count of normalized dataset
            normalized_dataframe_column_count: Column count of normalized dataset
            normalized_target_column: Normalized name of target column

        Returns:
            Complete PipelineMetadata object
        """
        logger.debug("Building pipeline metadata")

        # Build dataset summary
        dataset_summary = DatasetSummary(
            row_count=normalized_dataframe_row_count,
            column_count=normalized_dataframe_column_count,
            target_column=intent.task.target_column,
            normalized_target_column=normalized_target_column,
        )

        # Build intent snapshot
        intent_snapshot = IntentSnapshot.from_intent(intent)

        # Serialize schema metadata
        schema_dict = serialize_schema_metadata(schema_metadata)

        # Serialize quality profile
        quality_dict = serialize_quality_profile(quality_profile)

        # Build quality actions
        quality_actions = self._build_quality_actions(
            applied_actions, proposed_actions
        )

        # Build complete metadata
        metadata = PipelineMetadata(
            dataset_summary=dataset_summary,
            intent=intent_snapshot,
            schema_metadata=schema_dict,
            quality_profile=quality_dict,
            quality_actions=quality_actions,
        )

        logger.debug("Pipeline metadata built successfully")
        return metadata

    def _build_quality_actions(
        self,
        applied_actions: list[AppliedAction],
        proposed_actions: list[dict],
    ) -> DataQualityActions:
        """Build quality actions metadata.

        Args:
            applied_actions: List of AppliedAction records
            proposed_actions: List of proposed action dictionaries

        Returns:
            DataQualityActions object
        """
        # Separate applied and rejected actions
        applied_records = []
        rejected_records = []

        for action in applied_actions:
            record = ActionRecord(
                column=action.column,
                action=action.action,
                method=action.method,
                justification=action.justification,
                status=action.status,
                reason=action.reason,
            )

            if action.status == "applied":
                applied_records.append(record)
            else:
                rejected_records.append(record)

        return DataQualityActions(
            proposed_count=len(proposed_actions),
            applied_count=len(applied_records),
            rejected_count=len(rejected_records),
            proposed_actions=proposed_actions,
            applied_actions=applied_records,
            rejected_actions=rejected_records,
        )
