"""Pipeline orchestrator for coordinating preprocessing pipeline execution.

This module defines the PipelineOrchestrator class which accepts validated
intent and orchestrates the preprocessing pipeline across all levels.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from intent.schema import IntentSchema
from level1_ingestion.loader import DatasetLoadError, load_dataset
from level1_ingestion.normalizer import NormalizationError, normalize_dataset
from level1_ingestion.schema_inferencer import SchemaMetadata, infer_schema
from level2_quality.agent import AgentError, DataQualityAgent
from level2_quality.executor import ActionExecutor, ExecutionError
from level2_quality.profiler import DatasetQualityProfile, profile_dataset
from level3_metadata.builder import MetadataBuilder
from level3_metadata.writer import MetadataWriter
from level4_feature.agent import FeatureAgentError, FeatureEngineeringAgent
from level4_feature.generator import FeatureGenerationError, FeatureGenerator
from level4_feature.validator import FeatureValidator, FeatureValidationError
from utils import get_logger

logger = get_logger(__name__)


class PipelineOrchestrator:
    """Orchestrates the preprocessing pipeline execution.

    This class accepts a validated intent and coordinates the execution
    of the preprocessing pipeline across multiple levels.

    Args:
        intent: Validated IntentSchema instance
        output_path: Optional path for pipeline outputs
    """

    def __init__(
        self,
        intent: IntentSchema,
        output_path: Optional[Path] = None,
        llm_client: Optional[object] = None,
    ):
        """Initialize the orchestrator with validated intent.

        Args:
            intent: Validated IntentSchema instance (read-only)
            output_path: Optional path for pipeline outputs
            llm_client: Optional LLM client for agents (if None, agents return empty proposals)
        """
        self.intent = intent
        self.output_path = Path(output_path) if output_path else None
        self.llm_client = llm_client

        # Level 1 results (populated after Level 1 execution)
        self.raw_dataframe: Optional[pd.DataFrame] = None
        self.normalized_dataframe: Optional[pd.DataFrame] = None
        self.schema_metadata: Optional[SchemaMetadata] = None
        self.column_name_mapping: Optional[dict[str, str]] = None
        self.normalized_target_column: Optional[str] = None

        # Level 2 results (populated after Level 2 execution)
        self.quality_profile: Optional[DatasetQualityProfile] = None
        self.cleaned_dataframe: Optional[pd.DataFrame] = None
        self.applied_actions: Optional[list] = None
        self.proposed_actions: Optional[list] = None

        # Level 3 results (populated after Level 3 execution)
        self.metadata_path: Optional[Path] = None

        # Level 4 results (populated after Level 4 execution)
        self.feature_dataframe: Optional[pd.DataFrame] = None
        self.validated_features: Optional[list] = None
        self.rejected_features: Optional[list] = None
        self.feature_provenance: Optional[list] = None

        logger.info("PipelineOrchestrator initialized")
        logger.debug(f"Intent: task={intent.task.type}, model={intent.model.family}")
        if self.output_path:
            logger.debug(f"Output path: {self.output_path}")

    def _run_level1_ingestion(self) -> None:
        """Execute Level 1: Data Ingestion & Schema Normalization.

        This level:
        - Loads the dataset from disk
        - Infers schema metadata
        - Normalizes column names and types
        - Validates intent constraints

        Raises:
            DatasetLoadError: If dataset loading fails
            NormalizationError: If normalization or validation fails
        """
        logger.info("=" * 60)
        logger.info("Level 1: Data Ingestion & Schema Normalization")
        logger.info("=" * 60)

        # Step 1: Load dataset
        logger.info("Step 1: Loading dataset...")
        try:
            self.raw_dataframe = load_dataset(self.intent.dataset_path)
            logger.info(f"✓ Dataset loaded: {self.raw_dataframe.shape[0]} rows, {self.raw_dataframe.shape[1]} columns")
        except DatasetLoadError as e:
            logger.error(f"✗ Dataset loading failed: {e}")
            raise

        # Step 2: Infer schema metadata
        logger.info("Step 2: Inferring schema metadata...")
        self.schema_metadata = infer_schema(self.raw_dataframe)
        logger.info(f"✓ Schema inferred: {self.schema_metadata.total_columns} columns analyzed")

        # Step 3: Normalize dataset
        logger.info("Step 3: Normalizing dataset...")
        try:
            (
                self.normalized_dataframe,
                self.column_name_mapping,
                self.normalized_target_column,
            ) = normalize_dataset(self.raw_dataframe, self.intent)
            logger.info(f"✓ Dataset normalized: {self.normalized_dataframe.shape[0]} rows, {self.normalized_dataframe.shape[1]} columns")
            logger.info(f"✓ Target column normalized: '{self.normalized_target_column}'")
        except NormalizationError as e:
            logger.error(f"✗ Dataset normalization failed: {e}")
            raise

        logger.info("Level 1 completed successfully")

    def _run_level2_quality(self) -> None:
        """Execute Level 2: Data Quality Agent.

        This level:
        - Profiles dataset quality (deterministic)
        - Requests action proposals from LLM
        - Validates and applies safe actions (deterministic)

        Raises:
            AgentError: If LLM agent fails
            ExecutionError: If action execution fails
        """
        logger.info("=" * 60)
        logger.info("Level 2: Data Quality Agent")
        logger.info("=" * 60)

        if self.normalized_dataframe is None or self.schema_metadata is None:
            raise RuntimeError("Level 1 must complete before Level 2")

        # Step 1: Profile dataset quality (deterministic)
        logger.info("Step 1: Profiling dataset quality...")
        self.quality_profile = profile_dataset(self.normalized_dataframe)
        logger.info(f"✓ Quality profiling complete: {self.quality_profile.total_columns} columns analyzed")

        # Step 2: Request action proposals from LLM
        logger.info("Step 2: Requesting action proposals from LLM...")
        agent = DataQualityAgent(llm_client=self.llm_client)
        try:
            proposed_actions = agent.propose_actions(
                self.quality_profile,
                self.schema_metadata,
                self.intent,
            )
            logger.info(f"✓ LLM proposed {len(proposed_actions.get('actions', []))} actions")
        except AgentError as e:
            logger.warning(f"LLM agent unavailable or returned invalid output: {e}")
            logger.info("Continuing without LLM proposals (empty actions)")
            proposed_actions = {"actions": []}

        # Step 3: Validate and execute actions (deterministic)
        logger.info("Step 3: Validating and executing actions...")
        executor = ActionExecutor(
            self.normalized_dataframe,
            self.schema_metadata,
            self.intent,
            self.normalized_target_column,
        )
        try:
            self.cleaned_dataframe = executor.execute_actions(
                proposed_actions.get("actions", [])
            )
            self.applied_actions = executor.applied_actions
            self.proposed_actions = proposed_actions.get("actions", [])

            applied_count = sum(1 for a in self.applied_actions if a.status == "applied")
            rejected_count = sum(1 for a in self.applied_actions if a.status == "rejected")
            logger.info(f"✓ Actions executed: {applied_count} applied, {rejected_count} rejected")
        except ExecutionError as e:
            logger.error(f"✗ Action execution failed: {e}")
            raise

        logger.info("Level 2 completed successfully")

    def _run_level3_metadata(self) -> None:
        """Execute Level 3: Metadata & Profiling Persistence.

        This level:
        - Aggregates metadata from Levels 0-2
        - Persists metadata to disk as JSON
        - Enables traceability and reproducibility

        Note: Metadata writing failures are logged but do not fail the pipeline.
        """
        logger.info("=" * 60)
        logger.info("Level 3: Metadata & Profiling Persistence")
        logger.info("=" * 60)

        if (
            self.normalized_dataframe is None
            or self.schema_metadata is None
            or self.quality_profile is None
            or self.applied_actions is None
        ):
            logger.warning(
                "Level 3 requires Level 1 and Level 2 outputs. Skipping metadata generation."
            )
            return

        try:
            # Step 1: Build metadata
            logger.info("Step 1: Building pipeline metadata...")
            builder = MetadataBuilder()
            metadata = builder.build(
                intent=self.intent,
                schema_metadata=self.schema_metadata,
                quality_profile=self.quality_profile,
                applied_actions=self.applied_actions,
                proposed_actions=self.proposed_actions or [],
                normalized_dataframe_row_count=len(self.normalized_dataframe),
                normalized_dataframe_column_count=len(self.normalized_dataframe.columns),
                normalized_target_column=self.normalized_target_column,
            )
            logger.info("✓ Pipeline metadata built")

            # Step 2: Write metadata to disk
            logger.info("Step 2: Persisting metadata to disk...")
            output_dir = self.output_path if self.output_path else Path.cwd()
            writer = MetadataWriter(output_dir=output_dir)
            self.metadata_path = writer.write_to_default_location(metadata, overwrite=False)
            logger.info(f"✓ Metadata persisted to: {self.metadata_path}")

        except Exception as e:
            # Metadata failures should not break the pipeline
            logger.error(f"✗ Metadata generation/persistence failed: {e}")
            logger.warning("Pipeline will continue despite metadata failure")
            self.metadata_path = None

        logger.info("Level 3 completed")

    def _run_level4_features(self) -> None:
        """Execute Level 4: Feature Engineering Agent.

        This level:
        - Requests feature proposals from LLM
        - Validates proposals for safety and compliance
        - Generates features deterministically
        - Tracks feature provenance

        Raises:
            FeatureAgentError: If LLM agent fails
            FeatureValidationError: If validation fails critically
            FeatureGenerationError: If feature generation fails
        """
        logger.info("=" * 60)
        logger.info("Level 4: Feature Engineering Agent")
        logger.info("=" * 60)

        if (
            self.cleaned_dataframe is None
            or self.schema_metadata is None
            or self.quality_profile is None
            or self.normalized_target_column is None
        ):
            raise RuntimeError("Level 1, 2, and 3 must complete before Level 4")

        # Step 1: Request feature proposals from LLM
        logger.info("Step 1: Requesting feature proposals from LLM...")
        agent = FeatureEngineeringAgent(llm_client=self.llm_client)
        try:
            proposed_features = agent.propose_features(
                self.schema_metadata,
                self.quality_profile,
                self.intent,
            )
            logger.info(
                f"✓ LLM proposed {len(proposed_features.get('features', []))} features"
            )
        except FeatureAgentError as e:
            logger.warning(f"LLM agent unavailable or returned invalid output: {e}")
            logger.info("Continuing without LLM proposals (empty features)")
            proposed_features = {"features": []}

        # Step 2: Validate proposals (critical safety layer)
        logger.info("Step 2: Validating feature proposals...")
        validator = FeatureValidator(
            self.schema_metadata,
            self.intent,
            self.normalized_target_column,
        )
        try:
            validated, rejected = validator.validate_proposals(
                proposed_features.get("features", [])
            )
            self.validated_features = validated
            self.rejected_features = rejected

            validated_count = len(validated)
            rejected_count = len(rejected)
            logger.info(
                f"✓ Feature validation: {validated_count} validated, {rejected_count} rejected"
            )

            if rejected_count > 0:
                logger.debug("Rejected features:")
                for r in rejected:
                    logger.debug(f"  - {r.name}: {r.reason}")

        except FeatureValidationError as e:
            logger.error(f"✗ Feature validation failed: {e}")
            raise

        # Step 3: Generate features deterministically
        logger.info("Step 3: Generating features...")
        generator = FeatureGenerator(self.cleaned_dataframe)
        try:
            self.feature_dataframe = generator.generate_features(validated)
            self.feature_provenance = generator.provenance

            logger.info(
                f"✓ Feature generation complete: {len(self.feature_dataframe.columns)} total columns "
                f"({len(self.feature_provenance)} new features)"
            )
        except FeatureGenerationError as e:
            logger.error(f"✗ Feature generation failed: {e}")
            raise

        logger.info("Level 4 completed successfully")

    def run(self) -> int:
        """Run the preprocessing pipeline.

        Orchestrates execution across all levels:
        - Level 1: Data ingestion & schema normalization
        - Level 2: Data quality agent
        - Level 3: Metadata generation (future)
        - Level 4: Feature engineering (future)
        - Level 5: Policy enforcement (future)
        - Level 6: Artifact generation (future)

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        logger.info("Starting preprocessing pipeline")
        logger.info(f"Dataset: {self.intent.dataset_path}")
        logger.info(f"Task: {self.intent.task.type}")
        logger.info(f"Target column: {self.intent.task.target_column}")
        logger.info(f"Model family: {self.intent.model.family}")

        try:
            # Level 1: Data Ingestion & Schema Normalization
            self._run_level1_ingestion()

            # Level 2: Data Quality Agent
            self._run_level2_quality()

            # Level 3: Metadata & Profiling Persistence
            self._run_level3_metadata()

            # Level 4: Feature Engineering Agent
            self._run_level4_features()

            # Prepare for Level 5 (future)
            logger.info("Level 4 outputs prepared for Level 5")

            if self.output_path:
                logger.info(f"Output will be written to: {self.output_path}")

            if self.metadata_path:
                logger.info(f"Metadata artifact: {self.metadata_path}")

            logger.info("Pipeline completed successfully")
            return 0

        except (DatasetLoadError, NormalizationError) as e:
            logger.error(f"Pipeline failed at Level 1: {e}")
            return 1
        except (AgentError, ExecutionError) as e:
            logger.error(f"Pipeline failed at Level 2: {e}")
            return 1
        except (FeatureAgentError, FeatureValidationError, FeatureGenerationError) as e:
            logger.error(f"Pipeline failed at Level 4: {e}")
            return 1
        except Exception as e:
            logger.exception(f"Unexpected error during pipeline execution: {e}")
            return 3
