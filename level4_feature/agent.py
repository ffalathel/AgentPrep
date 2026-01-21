"""LLM agent interface for feature engineering proposals.

This module provides the interface for LLM-based feature engineering proposals.
The agent proposes features but NEVER executes them - that's done deterministically.
"""

import json
import os
from typing import Any, Optional

from intent.schema import IntentSchema
from level1_ingestion.schema_inferencer import SchemaMetadata
from level2_quality.profiler import DatasetQualityProfile
from utils import (
    get_logger,
    sanitize_column_name,
    sanitize_json_for_prompt,
    sanitize_prompt_variable,
    sanitize_string_for_prompt,
)

from .feature_catalog import FeatureCatalog, get_catalog

logger = get_logger(__name__)


class FeatureAgentError(Exception):
    """Raised when feature agent fails."""

    pass


class FeatureEngineeringAgent:
    """LLM agent for proposing feature engineering transformations.

    This agent:
    - Takes schema metadata, quality profile, and intent
    - Proposes feature transformations as JSON
    - NEVER executes code or touches raw data
    - Only proposes transformations from the feature catalog
    """

    def __init__(self, llm_client: Optional[Any] = None):
        """Initialize the feature engineering agent.

        Args:
            llm_client: Optional LLM client (if None, returns empty proposals)
        """
        self.llm_client = llm_client
        self.catalog = get_catalog()
        logger.debug("FeatureEngineeringAgent initialized")

    def propose_features(
        self,
        schema_metadata: SchemaMetadata,
        quality_profile: DatasetQualityProfile,
        intent: IntentSchema,
    ) -> dict[str, Any]:
        """Propose feature engineering transformations.

        This method constructs a prompt for the LLM and requests feature proposals.
        The LLM must return JSON only, with no executable code.

        Args:
            schema_metadata: Schema metadata from Level 1
            quality_profile: Quality profile from Level 2
            intent: User intent (read-only)

        Returns:
            Dictionary with "features" key containing list of proposed features

        Raises:
            FeatureAgentError: If LLM fails or returns invalid output
        """
        if self.llm_client is None:
            logger.warning("No LLM client provided, returning empty feature proposals")
            return {"features": []}

        try:
            # Construct prompt
            prompt = self._construct_prompt(
                schema_metadata, quality_profile, intent
            )

            # Call LLM
            logger.debug("Requesting feature proposals from LLM")
            response = self._call_llm(prompt)

            # Parse and validate response
            proposals = self._parse_response(response)

            logger.info(f"LLM proposed {len(proposals.get('features', []))} features")
            return proposals

        except json.JSONDecodeError as e:
            # For testing/debugging: optionally print what the LLM returned.
            if os.getenv("AGENTPREP_PRINT_LLM_RESPONSE", "").strip() == "1":
                snippet = response if isinstance(response, str) else str(response)
                snippet = snippet[:2000]
                print("\n--- AgentPrep: raw LLM response (truncated) ---")
                print(snippet)
                print("--- end raw LLM response ---\n")

            logger.warning(
                "LLM returned invalid JSON. Response snippet (first 500 chars): %s",
                (response if isinstance(response, str) else str(response))[:500],
            )
            raise FeatureAgentError(f"LLM returned invalid JSON: {e}") from e
        except Exception as e:
            logger.error(f"Feature agent failed: {e}")
            raise FeatureAgentError(f"Feature agent failed: {e}") from e

    def _construct_prompt(
        self,
        schema_metadata: SchemaMetadata,
        quality_profile: DatasetQualityProfile,
        intent: IntentSchema,
    ) -> str:
        """Construct prompt for LLM.

        The prompt includes:
        - Available columns and their types
        - Quality statistics (no raw data)
        - Intent constraints
        - Feature catalog summary
        - Output format specification

        Args:
            schema_metadata: Schema metadata
            quality_profile: Quality profile
            intent: User intent

        Returns:
            Prompt string for LLM
        """
        # Build column summary (no raw data)
        column_info = []
        for col_name, col_meta in schema_metadata.columns.items():
            if col_name == intent.task.target_column:
                continue  # Skip target column

            # Sanitize column name
            sanitized_col_name = sanitize_column_name(col_name)
            col_quality = quality_profile.columns.get(col_name)
            col_info = {
                "name": sanitized_col_name,
                "semantic_type": sanitize_string_for_prompt(col_meta.semantic_type),
                "missing_percentage": round(col_meta.missing_percentage, 2),
                "unique_count": col_meta.unique_count,
            }

            # Add quality stats if available
            if col_quality:
                col_info["is_constant"] = bool(col_quality.is_constant)
                col_info["is_near_constant"] = bool(col_quality.is_near_constant)
                if col_quality.numeric_stats:
                    col_info["has_outliers"] = col_quality.numeric_stats.outlier_count_total > 0

            column_info.append(col_info)

        # Get feature catalog summary
        catalog_summary = self.catalog.get_summary()

        # Sanitize all user-provided values
        sanitized_target_column = sanitize_column_name(intent.task.target_column)
        sanitized_task_type = sanitize_string_for_prompt(str(intent.task.type))
        sanitized_model_family = sanitize_string_for_prompt(str(intent.model.family))
        sanitized_interpretability = sanitize_string_for_prompt(
            str(intent.preferences.interpretability_priority)
        )
        
        # Sanitize JSON data
        sanitized_column_info_json = sanitize_json_for_prompt(column_info)
        sanitized_catalog_summary_json = sanitize_json_for_prompt(catalog_summary)

        prompt = f"""You are a feature engineering assistant for ML preprocessing.

TASK: Propose feature engineering transformations to improve model readiness.

CONSTRAINTS:
- Task type: {sanitized_task_type}
- Model family: {sanitized_model_family}
- Max features: {intent.constraints.max_features}
- Max interactions: {intent.constraints.max_interactions}
- Interpretability priority: {sanitized_interpretability}
- Target column: {sanitized_target_column} (NEVER use as input)

AVAILABLE COLUMNS (excluding target):
{sanitized_column_info_json}

FEATURE CATALOG (allowed transformations only):
{sanitized_catalog_summary_json}

RULES:
1. Only propose transformations from the catalog above
2. Never use the target column as input
3. Respect max_features and max_interactions limits
4. Prefer interpretable features if interpretability_priority is high
5. Consider model family (e.g., tree models benefit from binning)
6. No executable code - only transformation names from catalog

OUTPUT FORMAT (JSON only):
{{
  "features": [
    {{
      "name": "feature_name",
      "source_columns": ["col1", "col2"],
      "transformation": "transformation_name_from_catalog",
      "reason": "brief explanation"
    }}
  ]
}}

Propose features that will improve model performance while respecting constraints.
Return ONLY valid JSON, no other text."""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt via the unified client."""
        if self.llm_client is None:
            raise FeatureAgentError("No LLM client available")

        try:
            # Choose a sensible default model based on provider, if available
            provider = getattr(self.llm_client, "provider", None)
            if provider == "openai":
                model_name = "gpt-4"
            elif provider == "anthropic":
                model_name = "claude-3-opus-20240229"
            elif provider == "gemini":
                model_name = "gemini-2.5-flash-lite"
            else:
                model_name = None

            return self.llm_client.complete(
                prompt,
                model=model_name,
                temperature=0.2,
            )
        except Exception as e:
            raise FeatureAgentError(f"Failed to call LLM: {e}") from e

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse LLM response and validate structure.

        Args:
            response: Raw LLM response string

        Returns:
            Parsed proposals dictionary

        Raises:
            FeatureAgentError: If response is invalid
        """
        try:
            # Try to parse JSON (strip optional ```json ... ``` fences)
            text = response.strip()
            if text.startswith("```"):
                first_nl = text.find("\n")
                last_ticks = text.rfind("```")
                if first_nl != -1 and last_ticks != -1 and last_ticks > first_nl:
                    text = text[first_nl + 1:last_ticks].strip()
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise FeatureAgentError(
                f"LLM returned invalid JSON: {e}"
            ) from e

        # Validate structure
        if not isinstance(data, dict):
            raise FeatureAgentError("LLM response must be a JSON object")

        if "features" not in data:
            raise FeatureAgentError("LLM response must contain 'features' key")

        if not isinstance(data["features"], list):
            raise FeatureAgentError("'features' must be a list")

        # Validate each feature proposal
        for i, feature in enumerate(data["features"]):
            if not isinstance(feature, dict):
                raise FeatureAgentError(
                    f"Feature proposal {i} must be a dictionary"
                )

            required_fields = ["name", "source_columns", "transformation"]
            for field in required_fields:
                if field not in feature:
                    raise FeatureAgentError(
                        f"Feature proposal {i} missing required field: {field}"
                    )

            if not isinstance(feature["source_columns"], list):
                raise FeatureAgentError(
                    f"Feature proposal {i}: 'source_columns' must be a list"
                )

            if not isinstance(feature["transformation"], str):
                raise FeatureAgentError(
                    f"Feature proposal {i}: 'transformation' must be a string"
                )

        return data
