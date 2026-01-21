"""LLM-based data quality agent for Level 2.

This module provides the interface to an LLM that proposes data cleaning actions.
The LLM never sees raw data rows and only outputs structured JSON.

Safety boundaries:
- LLM only receives aggregated statistics
- LLM output must be validated JSON
- LLM never executes code
- All actions must be validated by executor
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
    sanitize_string_for_prompt,
)

logger = get_logger(__name__)


class AgentError(Exception):
    """Raised when agent operations fail."""

    pass


class DataQualityAgent:
    """LLM-based agent that proposes data quality improvement actions.

    This agent:
    - Receives aggregated profiling data (never raw rows)
    - Proposes structured cleaning actions in JSON format
    - Actions must be validated by executor before execution
    """

    def __init__(self, llm_client: Optional[Any] = None):
        """Initialize the data quality agent.

        Args:
            llm_client: Optional LLM client. If None, uses a mock/stub for testing.
        """
        self.llm_client = llm_client
        logger.info("DataQualityAgent initialized")

    def _construct_prompt(
        self,
        quality_profile: DatasetQualityProfile,
        schema_metadata: SchemaMetadata,
        intent: IntentSchema,
    ) -> str:
        """Construct a strict prompt for the LLM.

        Args:
            quality_profile: Dataset quality profile from profiler
            schema_metadata: Schema metadata from Level 1
            intent: User intent from Level 0

        Returns:
            Formatted prompt string
        """
        # Build column information (aggregated stats only, no raw data)
        column_info = []
        for col_name, col_profile in quality_profile.columns.items():
            col_schema = schema_metadata.columns.get(col_name)
            # Sanitize column name
            sanitized_col_name = sanitize_column_name(col_name)
            col_info = {
                "column": sanitized_col_name,
                "semantic_type": sanitize_string_for_prompt(
                    col_schema.semantic_type if col_schema else "unknown"
                ),
                "missing_percentage": col_profile.missing_percentage,
                "missing_count": col_profile.missing_count,
                "unique_count": col_profile.unique_count,
                "is_constant": bool(col_profile.is_constant),  # Convert numpy bool to Python bool
                "is_near_constant": bool(col_profile.is_near_constant),  # Convert numpy bool to Python bool
            }

            # Add numeric stats if available
            if col_profile.numeric_stats:
                col_info["numeric_stats"] = {
                    "mean": col_profile.numeric_stats.mean,
                    "median": col_profile.numeric_stats.median,
                    "std": col_profile.numeric_stats.std,
                    "outlier_count": col_profile.numeric_stats.outlier_count_total,
                    "iqr": col_profile.numeric_stats.iqr,
                }

            column_info.append(col_info)

        # Sanitize all user-provided values
        sanitized_target_column = sanitize_column_name(intent.task.target_column)
        sanitized_task_type = sanitize_string_for_prompt(intent.task.type.value)
        sanitized_model_family = sanitize_string_for_prompt(intent.model.family.value)
        sanitized_outlier_policy = sanitize_string_for_prompt(intent.preferences.outlier_policy.value)
        
        # Sanitize JSON data
        sanitized_column_info_json = sanitize_json_for_prompt(column_info)

        prompt = f"""You are a data quality expert analyzing a dataset for ML preprocessing.

Dataset Summary:
- Total rows: {quality_profile.total_rows}
- Total columns: {quality_profile.total_columns}
- Task type: {sanitized_task_type}
- Target column: {sanitized_target_column}
- Model family: {sanitized_model_family}
- Outlier policy: {sanitized_outlier_policy}
- Max features: {intent.constraints.max_features}

Column Quality Metrics:
{sanitized_column_info_json}

Constraints:
- Target column '{sanitized_target_column}' must NEVER be dropped or modified
- Maximum {intent.constraints.max_features} feature columns allowed
- Outlier policy: {sanitized_outlier_policy}
- Column dropping allowed: {intent.preferences.allow_column_dropping}

Analyze the quality metrics and propose data cleaning actions. Return ONLY valid JSON in this exact format:

{{
  "actions": [
    {{
      "column": "column_name",
      "action": "impute | clip | flag | drop",
      "method": "mean | median | mode | iqr",
      "justification": "brief reason"
    }}
  ]
}}

Rules:
- Use "impute" for missing values (method: mean/median for numeric, mode for categorical)
- Use "clip" for outliers (only if outlier_policy is "clip")
- Use "flag" for outliers (only if outlier_policy is "flag")
- Use "drop" only for constant/near-constant columns (if allow_column_dropping is true)
- Column names must match exactly
- Never propose actions on the target column
- Return empty actions array if no actions needed

Return ONLY the JSON, no other text."""

        return prompt

    def propose_actions(
        self,
        quality_profile: DatasetQualityProfile,
        schema_metadata: SchemaMetadata,
        intent: IntentSchema,
    ) -> dict[str, Any]:
        """Propose data quality improvement actions.

        This method calls the LLM with aggregated statistics and returns
        structured action proposals. The LLM never sees raw data rows.

        Args:
            quality_profile: Dataset quality profile from profiler
            schema_metadata: Schema metadata from Level 1
            intent: User intent from Level 0

        Returns:
            Dictionary with "actions" key containing list of proposed actions

        Raises:
            AgentError: If LLM call fails or output is invalid
        """
        logger.info("Requesting action proposals from LLM")

        prompt = self._construct_prompt(quality_profile, schema_metadata, intent)

        try:
            if self.llm_client is None:
                logger.warning("No LLM client provided, returning empty actions")
                return {"actions": []}

            response = self._call_llm(prompt)

            # Parse and validate JSON response
            if isinstance(response, str):
                text = response.strip()
                # Handle common LLM pattern of wrapping JSON in ```json ... ``` fences
                if text.startswith("```"):
                    first_nl = text.find("\n")
                    last_ticks = text.rfind("```")
                    if first_nl != -1 and last_ticks != -1 and last_ticks > first_nl:
                        text = text[first_nl + 1:last_ticks].strip()
                actions_data = json.loads(text)
            else:
                actions_data = response

            # Validate structure
            if not isinstance(actions_data, dict) or "actions" not in actions_data:
                raise AgentError("LLM response missing 'actions' key")

            if not isinstance(actions_data["actions"], list):
                raise AgentError("LLM response 'actions' must be a list")

            logger.info(f"LLM proposed {len(actions_data['actions'])} actions")
            return actions_data

        except json.JSONDecodeError as e:
            # For testing/debugging: optionally print what the LLM returned.
            if os.getenv("AGENTPREP_PRINT_LLM_RESPONSE", "").strip() == "1":
                snippet = (response if isinstance(response, str) else str(response))
                snippet = snippet[:2000]
                print("\n--- AgentPrep: raw LLM response (truncated) ---")
                print(snippet)
                print("--- end raw LLM response ---\n")

            # Always log a short snippet to help debugging without flooding logs.
            try:
                snippet = (response if isinstance(response, str) else str(response))
                logger.warning(
                    "LLM returned invalid JSON. Response snippet (first 500 chars): %s",
                    snippet[:500],
                )
            except Exception:
                pass

            raise AgentError(f"LLM returned invalid JSON: {e}") from e
        except Exception as e:
            raise AgentError(f"Failed to get LLM proposals: {e}") from e

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the prompt using the unified client."""
        if self.llm_client is None:
            raise AgentError("No LLM client available")

        # Uses utils.llm_client.LLMClientWrapper.complete()
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
            raise AgentError(f"Failed to call LLM: {e}") from e
