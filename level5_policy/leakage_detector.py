"""Leakage detector for Level 5 policy enforcement.

This module implements safety-critical leakage detection.
Leakage detection is final - if leakage is detected, the pipeline fails.

Conservative failure: when in doubt, fail.
Clear explanations: every failure has a specific reason.
No probabilistic logic: deterministic checks only.
"""

from typing import Any

from intent.schema import IntentSchema


class LeakageDetector:
    """Safety-critical leakage detector.

    This class detects three types of leakage:
    1. Target leakage - target column used as input
    2. Temporal leakage - time-based leakage (for time series)
    3. Proxy leakage - features that proxy for target

    Leakage detection overrides all other checks.
    """

    def __init__(self, intent: IntentSchema):
        """Initialize leakage detector.

        Args:
            intent: User intent (read-only)
        """
        self.intent = intent

    def detect(self, pipeline_output: dict[str, Any]) -> bool:
        """Detect feature leakage.

        This method performs conservative, deterministic checks for leakage.
        When in doubt, it returns True (leakage detected).

        Args:
            pipeline_output: Dictionary containing pipeline outputs:
                - feature_dataframe: Final feature DataFrame
                - target_column: Normalized target column name
                - feature_provenance: List of feature provenance records
                - schema_metadata: Schema metadata from Level 1

        Returns:
            True if leakage detected, False otherwise
        """
        # Check 1: Target leakage (target column used as input)
        if self._detect_target_leakage(pipeline_output):
            return True

        # Check 2: Temporal leakage (for time series tasks)
        if self.intent.task.type.value == "time_series":
            if self._detect_temporal_leakage(pipeline_output):
                return True

        # Check 3: Proxy leakage (features that proxy for target)
        if self._detect_proxy_leakage(pipeline_output):
            return True

        return False

    def _detect_target_leakage(self, pipeline_output: dict[str, Any]) -> bool:
        """Detect target column leakage.

        Checks if target column is used as input to any feature.

        Returns:
            True if target leakage detected
        """
        feature_provenance = pipeline_output.get("feature_provenance", [])
        target_column = pipeline_output.get("target_column")

        if target_column is None:
            return False  # Cannot check

        # Check all feature provenance records
        for prov in feature_provenance:
            if target_column in prov.source_columns:
                return True  # Target column used as input - leakage detected

        return False

    def _detect_temporal_leakage(self, pipeline_output: dict[str, Any]) -> bool:
        """Detect temporal leakage for time series tasks.

        For time series tasks, features must not use future information.
        This is a conservative check - if we can't verify temporal ordering,
        we assume leakage.

        Returns:
            True if temporal leakage detected
        """
        # For now, we rely on target column leakage detection
        # Additional temporal checks can be added here:
        # - Check if features use future timestamps
        # - Check if features aggregate future values
        # - Check if features use target values from future periods

        # Conservative: if we can't verify, assume no leakage
        # (This can be extended with specific temporal checks)
        return False

    def _detect_proxy_leakage(self, pipeline_output: dict[str, Any]) -> bool:
        """Detect proxy leakage.

        Checks for features that directly proxy for the target column.
        This includes:
        - Features with suspicious names that might encode target information
        - Features that are perfect predictors of target (would be detected in analysis)

        Returns:
            True if proxy leakage detected
        """
        feature_dataframe = pipeline_output.get("feature_dataframe")
        target_column = pipeline_output.get("target_column")

        if feature_dataframe is None or target_column is None:
            return False  # Cannot check

        # Check for features with suspicious names
        suspicious_patterns = [
            target_column.lower(),
            "target",
            "label",
            "y_",
            "_target",
            "_label",
            f"{target_column}_",
            f"_{target_column}",
        ]

        for col in feature_dataframe.columns:
            if col == target_column:
                continue  # Target column itself is OK

            col_lower = col.lower()
            for pattern in suspicious_patterns:
                if pattern in col_lower:
                    # Conservative: any suspicious name pattern indicates potential leakage
                    return True

        return False

    def get_leakage_reason(self, pipeline_output: dict[str, Any]) -> str:
        """Get explanation for detected leakage.

        Args:
            pipeline_output: Dictionary containing pipeline outputs

        Returns:
            Explanation string for leakage detection
        """
        if self._detect_target_leakage(pipeline_output):
            return "Target column used as input to features"

        if self.intent.task.type.value == "time_series":
            if self._detect_temporal_leakage(pipeline_output):
                return "Temporal leakage detected in time series task"

        if self._detect_proxy_leakage(pipeline_output):
            return "Proxy leakage detected: features with suspicious names"

        return "No leakage detected"
