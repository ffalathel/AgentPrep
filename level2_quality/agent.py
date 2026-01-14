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
import logging
from typing import Any, Optional

from intent.schema import IntentSchema
from level1_ingestion.schema_inferencer import SchemaMetadata
from level2_quality.profiler import DatasetQualityProfile

logger = logging.getLogger(__name__)


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
            col_info = {
                "column": col_name,
                "semantic_type": col_schema.semantic_type if col_schema else "unknown",
                "missing_percentage": col_profile.missing_percentage,
                "missing_count": col_profile.missing_count,
                "unique_count": col_profile.unique_count,
                "is_constant": col_profile.is_constant,
                "is_near_constant": col_profile.is_near_constant,
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

        prompt = f"""You are a data quality expert analyzing a dataset for ML preprocessing.

Dataset Summary:
- Total rows: {quality_profile.total_rows}
- Total columns: {quality_profile.total_columns}
- Task type: {intent.task.type.value}
- Target column: {intent.task.target_column}
- Model family: {intent.model.family.value}
- Outlier policy: {intent.preferences.outlier_policy.value}
- Max features: {intent.constraints.max_features}

Column Quality Metrics:
{json.dumps(column_info, indent=2)}

Constraints:
- Target column '{intent.task.target_column}' must NEVER be dropped or modified
- Maximum {intent.constraints.max_features} feature columns allowed
- Outlier policy: {intent.preferences.outlier_policy.value}
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
                # Stub implementation for testing/development
                logger.warning("No LLM client provided, returning empty actions")
                return {"actions": []}

            # Call LLM (this would be implemented with actual LLM client)
            # For now, this is a placeholder
            response = self._call_llm(prompt)

            # Parse and validate JSON response
            if isinstance(response, str):
                actions_data = json.loads(response)
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
            raise AgentError(f"LLM returned invalid JSON: {e}") from e
        except Exception as e:
            raise AgentError(f"Failed to get LLM proposals: {e}") from e

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the prompt.

        This is a placeholder method. In production, this would call
        the actual LLM client (OpenAI, Anthropic, etc.).

        Args:
            prompt: Formatted prompt string

        Returns:
            LLM response string (should be JSON)

        Raises:
            AgentError: If LLM call fails
        """
        # Placeholder implementation
        # In production, this would be:
        # response = self.llm_client.complete(prompt, ...)
        # return response

        raise AgentError(
            "LLM client not implemented. Provide an LLM client in constructor."
        )
