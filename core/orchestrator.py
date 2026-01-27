"""Pipeline orchestrator for coordinating preprocessing pipeline execution.

This module defines the PipelineOrchestrator class which accepts validated
intent and orchestrates the preprocessing pipeline across all levels.
"""

from datetime import datetime
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
from level5_governance.gatekeeper import GovernanceDecision, GovernanceGatekeeper
from level6_artifacts.artifact_registry import ArtifactRegistry
from level6_artifacts.store import ArtifactStore, ArtifactStoreError
from level6_artifacts.report_generator import ReportGenerationError, ReportGenerator
from utils import PathValidationError, get_logger, validate_output_path
from utils.constants import EXIT_POLICY_VIOLATION, EXIT_RUNTIME_ERROR, EXIT_SUCCESS
from utils.file_helpers import generate_run_id

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

        Raises:
            PathValidationError: If output_path contains directory traversal
        """
        self.intent = intent
        if output_path:
            try:
                # Validate output path for security
                self.output_path = validate_output_path(output_path)
            except PathValidationError as e:
                logger.error(f"Invalid output path: {e}")
                raise
        else:
            self.output_path = None
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

        # Level 5 results (governance decision)
        self.governance_decision: Optional[GovernanceDecision] = None

        # Level 6 results (artifacts & reports)
        self.run_id: Optional[str] = None
        self.artifact_registry: Optional[ArtifactRegistry] = None
        # Cached pipeline output for governance (used for optional remediation)
        self._governance_pipeline_output: Optional[dict[str, object]] = None
        # Track columns dropped for leakage (excluded from no_column_dropping policy)
        self.leakage_dropped_columns: set[str] = set()

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

    def _run_level5_governance(self) -> None:
        """Execute Level 5: Governance & Policy Enforcement.

        This level:
        - Invokes policy enforcement and leakage detection
        - Aggregates violations
        - Produces a final GovernanceDecision

        Policy violations do NOT raise exceptions; they are reflected in the
        GovernanceDecision and mapped to an exit code in run().
        """
        logger.info("=" * 60)
        logger.info("Level 5: Governance & Policy Enforcement")
        logger.info("=" * 60)

        if self.feature_dataframe is None or self.normalized_target_column is None:
            logger.warning(
                "Level 5 requires Level 4 outputs. Skipping governance checks."
            )
            return

        pipeline_output = {
            "feature_dataframe": self.feature_dataframe,
            "target_column": self.normalized_target_column,
            "feature_provenance": self.feature_provenance or [],
            "schema_metadata": self.schema_metadata,
            "metadata_path": self.metadata_path,
            "original_column_count": (
                len(self.normalized_dataframe.columns)
                if self.normalized_dataframe is not None
                else None
            ),
            "leakage_dropped_columns": list(self.leakage_dropped_columns),
        }
        # Cache for potential remediation/re-evaluation
        self._governance_pipeline_output = pipeline_output

        gatekeeper = GovernanceGatekeeper(self.intent)
        decision = gatekeeper.decide(pipeline_output)
        self.governance_decision = decision

        logger.info(str(decision))
        if decision.violations:
            logger.info("Governance violations:")
            for v in decision.violations:
                logger.info(f"  - {v}")
        if decision.warnings:
            logger.info("Governance warnings (informational):")
            for w in decision.warnings:
                logger.warning(f"  - {w}")

    def _run_level6_artifacts(self, success: bool, exit_code: int) -> None:
        """Execute Level 6: Artifact Storage & Reporting.

        This level:
        - Saves key artifacts to disk (dataset, schema, quality, provenance, metadata, governance)
        - Builds an artifact manifest
        - Generates human-readable reports

        Failures in this level are logged but do not change the pipeline exit code.
        """
        logger.info("=" * 60)
        logger.info("Level 6: Artifacts, Storage & Reporting")
        logger.info("=" * 60)

        # Determine base output directory
        output_dir = self.output_path if self.output_path else Path.cwd()

        # Ensure we have a run_id
        if not self.run_id:
            self.run_id = generate_run_id()

        store = ArtifactStore(output_dir=output_dir, run_id=self.run_id)
        registry = ArtifactRegistry(run_id=self.run_id)

        try:
            # Dataset artifact (use feature dataframe if available, else cleaned dataframe)
            dataset_df = (
                self.feature_dataframe
                if self.feature_dataframe is not None
                else self.cleaned_dataframe
            )
            if dataset_df is not None:
                dataset_artifact = store.save_dataset(dataset_df)
                registry.register_dataset(dataset_artifact)

            # Schema metadata artifact
            if self.schema_metadata is not None:
                schema_artifact = store.save_schema_metadata(self.schema_metadata)
                registry.register_schema(schema_artifact)

            # Quality profile artifact
            if self.quality_profile is not None:
                quality_artifact = store.save_quality_profile(self.quality_profile)
                registry.register_quality_profile(quality_artifact)

            # Feature provenance artifact
            if self.feature_provenance:
                provenance_artifact = store.save_feature_provenance(
                    self.feature_provenance
                )
                registry.register_feature_provenance(provenance_artifact)

            # Metadata artifact (Level 3)
            if self.metadata_path is not None:
                metadata_artifact = store.save_metadata(self.metadata_path)
                registry.register_metadata(metadata_artifact)

            # Governance artifact (Level 5)
            if self.governance_decision is not None:
                governance_artifact = store.save_governance_snapshot(
                    self.governance_decision
                )
                registry.register_governance(governance_artifact)

            # Build manifest & generate reports
            completed_at = datetime.utcnow()
            manifest = registry.build_manifest(
                completed_at=completed_at, success=success, exit_code=exit_code
            )
            self.artifact_registry = registry

            report_generator = ReportGenerator(output_dir=output_dir, run_id=self.run_id)
            report_paths = report_generator.generate_all_reports(manifest)
            if report_paths:
                logger.info(
                    f"Generated {len(report_paths)} reports in {report_generator.reports_dir}"
                )

        except (ArtifactStoreError, ReportGenerationError) as e:
            logger.error(f"✗ Artifact persistence/report generation failed: {e}")
            logger.warning("Pipeline will continue despite artifact/report failures")
        except Exception as e:
            logger.error(f"✗ Unexpected error in Level 6: {e}")
            logger.warning("Pipeline will continue despite artifact/report failures")

    def _find_leaking_features(self) -> list[str]:
        """Identify features that likely cause leakage.

        Uses simple deterministic rules aligned with the leakage detector:
        - Target column used as input to a feature
        - Feature names containing suspicious target-like patterns
        """
        if not self.feature_provenance or not self.normalized_target_column:
            return []

        leaking: set[str] = set()
        target = self.normalized_target_column

        # 1) Target used as input (direct leakage)
        for prov in self.feature_provenance:
            source_cols = getattr(prov, "source_columns", [])
            feature_name = getattr(prov, "feature_name", None)
            if feature_name is None:
                continue
            if target in source_cols:
                leaking.add(feature_name)

        # 2) Suspicious feature names (proxy leakage)
        suspicious_patterns = [
            target.lower(),
            "target",
            "label",
            "y_",
            "_target",
            "_label",
            f"{target}_",
            f"_{target}",
        ]

        for prov in self.feature_provenance:
            feature_name = getattr(prov, "feature_name", None)
            if not feature_name:
                continue
            col_lower = feature_name.lower()
            if any(pattern in col_lower for pattern in suspicious_patterns):
                leaking.add(feature_name)

        return sorted(leaking)

    def _remove_features(self, feature_names: list[str]) -> None:
        """Remove specified features from dataframe and provenance."""
        if not feature_names:
            return

        if self.feature_dataframe is not None:
            cols_to_drop = [name for name in feature_names if name in self.feature_dataframe.columns]
            if cols_to_drop:
                logger.info(
                    "Dropping %d feature(s) due to governance violations: %s",
                    len(cols_to_drop),
                    ", ".join(cols_to_drop),
                )
                self.feature_dataframe = self.feature_dataframe.drop(columns=cols_to_drop)

        if self.feature_provenance:
            self.feature_provenance = [
                prov
                for prov in self.feature_provenance
                if getattr(prov, "feature_name", None) not in feature_names
            ]

    def _attempt_remediation(self) -> bool:
        """Attempt to remediate data quality issues automatically.

        Returns:
            True if any remediation was applied, False otherwise
        """
        if self.governance_decision is None:
            return False

        remediation_info = getattr(self.governance_decision, "remediation_info", {})
        if not remediation_info:
            return False

        auto_remediate = getattr(
            self.intent.data_quality, "auto_remediate", True
        )  # Default to True

        if not auto_remediate:
            logger.info("Auto-remediation is disabled. Skipping remediation.")
            return False

        leakage_features_to_remove = set()
        other_features_to_remove = set()

        # 1. Remediate leakage (always allowed, even if column dropping is disabled)
        if "leakage" in remediation_info:
            leakage_features = remediation_info["leakage"].get("features_to_remove", [])
            if leakage_features:
                leakage_features_to_remove.update(leakage_features)
                if not self.intent.preferences.allow_column_dropping:
                    logger.warning(
                        "Feature leakage detected: dropping %d feature(s) (%s) to prevent leakage. "
                        "This is required for data integrity, even though column dropping is disabled.",
                        len(leakage_features),
                        ", ".join(leakage_features[:5])
                        + ("..." if len(leakage_features) > 5 else ""),
                    )
                else:
                    logger.info(
                        "Remediating leakage: removing %d feature(s): %s",
                        len(leakage_features),
                        ", ".join(leakage_features[:5])
                        + ("..." if len(leakage_features) > 5 else ""),
                    )

        # 2. Remediate multicollinearity (only if column dropping is allowed)
        if "multicollinearity" in remediation_info:
            multicoll_features = remediation_info["multicollinearity"].get(
                "features_to_remove", []
            )
            if multicoll_features:
                if self.intent.preferences.allow_column_dropping:
                    other_features_to_remove.update(multicoll_features)
                    logger.info(
                        "Remediating multicollinearity: removing %d feature(s): %s",
                        len(multicoll_features),
                        ", ".join(multicoll_features[:5])
                        + ("..." if len(multicoll_features) > 5 else ""),
                    )
                else:
                    logger.warning(
                        "Multicollinearity detected: %d feature(s) should be removed (%s), "
                        "but column dropping is disabled. Skipping removal.",
                        len(multicoll_features),
                        ", ".join(multicoll_features[:5])
                        + ("..." if len(multicoll_features) > 5 else ""),
                    )

        # 3. Remediate information loss (only if column dropping is allowed)
        if "information_loss" in remediation_info:
            loss_features = remediation_info["information_loss"].get(
                "features_to_remove", []
            )
            if loss_features:
                if self.intent.preferences.allow_column_dropping:
                    other_features_to_remove.update(loss_features)
                    logger.info(
                        "Remediating information loss: removing %d feature(s): %s",
                        len(loss_features),
                        ", ".join(loss_features[:5]) + ("..." if len(loss_features) > 5 else ""),
                    )
                else:
                    logger.warning(
                        "Information loss detected: %d feature(s) should be removed (%s), "
                        "but column dropping is disabled. Skipping removal.",
                        len(loss_features),
                        ", ".join(loss_features[:5]) + ("..." if len(loss_features) > 5 else ""),
                    )

        # 4. Remediate class imbalance (requires resampling, handled separately)
        if "class_imbalance" in remediation_info:
            try:
                from level5_policy.remediation import DataQualityRemediator

                remediator = DataQualityRemediator(self.intent)
                pipeline_output = {
                    "feature_dataframe": self.feature_dataframe,
                    "target_column": self.normalized_target_column,
                    "feature_provenance": self.feature_provenance or [],
                }
                imbalance_result = remediator.remediate_class_imbalance(pipeline_output)

                if imbalance_result.success and imbalance_result.features_modified:
                    # Resampling was applied - update dataframe
                    logger.info(f"Remediating class imbalance: {imbalance_result.reason}")
                    # Note: The remediator would need to return the resampled dataframe
                    # For now, we log the warning
                    if imbalance_result.warnings:
                        for warning in imbalance_result.warnings:
                            logger.warning(f"  - {warning}")
                elif imbalance_result.warnings:
                    for warning in imbalance_result.warnings:
                        logger.warning(f"  - {warning}")
            except ImportError:
                logger.warning(
                    "Class imbalance detected but imbalanced-learn not available. "
                    "Install with: pip install imbalanced-learn"
                )

        # Apply feature removals
        # Leakage features are always removed (required for data integrity)
        # Other features are only removed if column dropping is allowed
        all_features_to_remove = leakage_features_to_remove | other_features_to_remove
        
        if all_features_to_remove:
            # Track leakage-dropped columns for policy exclusion
            if leakage_features_to_remove:
                self.leakage_dropped_columns.update(leakage_features_to_remove)
            
            self._remove_features(sorted(all_features_to_remove))
            return True

        return False

    def run(self) -> int:
        """Run the preprocessing pipeline.

        Orchestrates execution across all levels:
        - Level 1: Data ingestion & schema normalization
        - Level 2: Data quality agent
        - Level 3: Metadata generation 
        - Level 4: Feature engineering 
        - Level 5: Policy enforcement 
        - Level 6: Artifact generation 

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

            # Level 5: Governance & Policy Enforcement
            self._run_level5_governance()

            # Optional remediation: try to auto-repair data quality issues (multiple attempts)
            attempts = 0
            max_attempts = 3
            while (
                self.governance_decision is not None
                and not self.governance_decision.approved
                and attempts < max_attempts
            ):
                remediation_applied = self._attempt_remediation()
                attempts += 1
                if remediation_applied:
                    # Re-run governance with updated outputs
                    logger.info("Re-running governance after remediation...")
                    self._run_level5_governance()
                else:
                    break

            # Determine exit code based on (possibly remediated) governance decision
            if self.governance_decision is not None and not self.governance_decision.approved:
                exit_code = EXIT_POLICY_VIOLATION
                logger.error(
                    f"Pipeline rejected by governance: {self.governance_decision.reason}"
                )
            else:
                exit_code = EXIT_SUCCESS

            # Level 6: Artifact Storage & Reporting (best-effort)
            self._run_level6_artifacts(success=(exit_code == EXIT_SUCCESS), exit_code=exit_code)

            if self.output_path:
                logger.info(f"Output will be written to: {self.output_path}")

            if self.metadata_path:
                logger.info(f"Metadata artifact: {self.metadata_path}")

            if exit_code == EXIT_SUCCESS:
                logger.info("Pipeline completed successfully")
            else:
                logger.info("Pipeline completed with policy violations")

            return exit_code

        except (DatasetLoadError, NormalizationError) as e:
            logger.error(f"Pipeline failed at Level 1 (ingestion): {e}")
            return 1  # Level 1 errors return 1 (maintained for backward compatibility)
        except (AgentError, ExecutionError) as e:
            logger.error(f"Pipeline failed at Level 2 (quality): {e}")
            return 1  # Level 2 errors return 1
        except (FeatureAgentError, FeatureValidationError, FeatureGenerationError) as e:
            logger.error(f"Pipeline failed at Level 4 (feature engineering): {e}")
            return 1  # Level 4 errors return 1
        except (OSError, IOError) as e:
            logger.error(f"Pipeline failed due to I/O error: {e}")
            return EXIT_RUNTIME_ERROR
        except MemoryError as e:
            logger.error(f"Pipeline failed due to insufficient memory: {e}")
            return EXIT_RUNTIME_ERROR
        except KeyboardInterrupt:
            logger.warning("Pipeline interrupted by user")
            return EXIT_RUNTIME_ERROR
        except Exception as e:
            logger.exception(f"Unexpected error during pipeline execution: {type(e).__name__}: {e}")
            return EXIT_RUNTIME_ERROR
