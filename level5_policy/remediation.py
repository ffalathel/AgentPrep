"""Remediation strategies for data quality issues.

This module implements automatic remediation strategies for various data quality
issues detected during governance. Remediation is optional and configurable.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from intent.schema import IntentSchema, TaskType
from level5_policy.data_quality_detectors import (
    ClassImbalanceDetector,
    InformationLossDetector,
    MulticollinearityDetector,
)


@dataclass
class RemediationResult:
    """Result of remediation attempt."""

    success: bool
    features_removed: list[str]
    features_modified: list[str]
    reason: str
    warnings: list[str]


class DataQualityRemediator:
    """Remediates data quality issues automatically."""

    def __init__(self, intent: IntentSchema):
        """Initialize remediator.

        Args:
            intent: User intent (read-only)
        """
        self.intent = intent

    def remediate_multicollinearity(
        self, pipeline_output: dict[str, Any]
    ) -> RemediationResult:
        """Remediate multicollinearity by removing one feature from each highly correlated pair.

        Strategy: For each pair, remove the feature that appears later in the column list
        (preserves earlier features which may be more important).

        Args:
            pipeline_output: Dictionary containing pipeline outputs

        Returns:
            RemediationResult with removed features
        """
        detector = MulticollinearityDetector(
            self.intent, threshold=self.intent.data_quality.multicollinearity_threshold
        )
        result = detector.detect(pipeline_output)

        if not result.multicollinearity_detected:
            return RemediationResult(
                success=True,
                features_removed=[],
                features_modified=[],
                reason="No multicollinearity detected",
                warnings=[],
            )

        # Collect features to remove (second feature in each pair)
        features_to_remove = set()
        for feat1, feat2, _ in result.highly_correlated_pairs:
            # Remove the second feature (arbitrary choice, but consistent)
            features_to_remove.add(feat2)

        return RemediationResult(
            success=True,
            features_removed=sorted(features_to_remove),
            features_modified=[],
            reason=f"Removed {len(features_to_remove)} feature(s) to resolve multicollinearity",
            warnings=[],
        )

    def remediate_information_loss(
        self, pipeline_output: dict[str, Any]
    ) -> RemediationResult:
        """Remediate information loss by removing constant/near-constant features.

        Args:
            pipeline_output: Dictionary containing pipeline outputs

        Returns:
            RemediationResult with removed features
        """
        detector = InformationLossDetector(self.intent)
        result = detector.detect(pipeline_output)

        if not result.loss_detected:
            return RemediationResult(
                success=True,
                features_removed=[],
                features_modified=[],
                reason="No information loss detected",
                warnings=[],
            )

        return RemediationResult(
            success=True,
            features_removed=result.affected_features,
            features_modified=[],
            reason=f"Removed {len(result.affected_features)} feature(s) with information loss",
            warnings=[],
        )

    def remediate_class_imbalance(
        self, pipeline_output: dict[str, Any]
    ) -> RemediationResult:
        """Remediate class imbalance using resampling.

        Note: This requires imbalanced-learn library. If not available, returns
        a warning suggesting manual resampling.

        Args:
            pipeline_output: Dictionary containing pipeline outputs

        Returns:
            RemediationResult (may include warnings if resampling not possible)
        """
        # Only applicable to classification tasks
        if self.intent.task.type != TaskType.CLASSIFICATION:
            return RemediationResult(
                success=True,
                features_removed=[],
                features_modified=[],
                reason="Class imbalance remediation only applies to classification tasks",
                warnings=[],
            )

        detector = ClassImbalanceDetector(
            self.intent,
            severe_threshold=self.intent.data_quality.class_imbalance_severe_threshold,
            moderate_threshold=self.intent.data_quality.class_imbalance_moderate_threshold,
        )
        result = detector.detect(pipeline_output)

        if not result.imbalance_detected:
            return RemediationResult(
                success=True,
                features_removed=[],
                features_modified=[],
                reason="No class imbalance detected",
                warnings=[],
            )

        # Try to apply resampling
        feature_dataframe = pipeline_output.get("feature_dataframe")
        target_column = pipeline_output.get("target_column")

        if feature_dataframe is None or target_column is None:
            return RemediationResult(
                success=False,
                features_removed=[],
                features_modified=[],
                reason="Cannot remediate: missing data",
                warnings=["Class imbalance detected but cannot remediate without data"],
            )

        # Check if imbalanced-learn is available
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.under_sampling import RandomUnderSampler

            # Separate features and target
            X = feature_dataframe.drop(columns=[target_column])
            y = feature_dataframe[target_column]

            # Apply SMOTE for severe imbalance, undersampling for moderate
            if result.severity == "severe":
                # Oversample minority class
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                method = "SMOTE (oversampling)"
            else:
                # Undersample majority class
                undersampler = RandomUnderSampler(random_state=42)
                X_resampled, y_resampled = undersampler.fit_resample(X, y)
                method = "Random undersampling"

            # Reconstruct dataframe
            resampled_df = X_resampled.copy()
            resampled_df[target_column] = y_resampled

            # Update pipeline output (this will be done by orchestrator)
            return RemediationResult(
                success=True,
                features_removed=[],
                features_modified=[target_column],  # Target was modified
                reason=f"Applied {method} to address {result.severity} class imbalance",
                warnings=[],
            )

        except ImportError:
            return RemediationResult(
                success=False,
                features_removed=[],
                features_modified=[],
                reason="Class imbalance detected but imbalanced-learn not available",
                warnings=[
                    f"{result.severity.capitalize()} class imbalance detected (ratio: {result.imbalance_ratio:.3f})",
                    "Install imbalanced-learn to enable automatic resampling: pip install imbalanced-learn",
                    "Alternatively, manually apply resampling techniques",
                ],
            )

    def remediate_all(
        self, pipeline_output: dict[str, Any], auto_remediate: bool = True
    ) -> dict[str, RemediationResult]:
        """Remediate all detected data quality issues.

        Args:
            pipeline_output: Dictionary containing pipeline outputs
            auto_remediate: If True, automatically apply remediations. If False, only report.

        Returns:
            Dictionary mapping issue type to RemediationResult
        """
        results = {}

        # 1. Multicollinearity (always safe to auto-remediate)
        if auto_remediate:
            results["multicollinearity"] = self.remediate_multicollinearity(pipeline_output)
        else:
            detector = MulticollinearityDetector(
                self.intent,
                threshold=self.intent.data_quality.multicollinearity_threshold,
            )
            multicoll_result = detector.detect(pipeline_output)
            if multicoll_result.multicollinearity_detected:
                features_to_remove = {feat2 for _, feat2, _ in multicoll_result.highly_correlated_pairs}
                results["multicollinearity"] = RemediationResult(
                    success=False,
                    features_removed=sorted(features_to_remove),
                    features_modified=[],
                    reason="Multicollinearity detected (auto-remediation disabled)",
                    warnings=["Enable auto_remediate to automatically remove correlated features"],
                )

        # 2. Information loss (always safe to auto-remediate)
        if auto_remediate:
            results["information_loss"] = self.remediate_information_loss(pipeline_output)
        else:
            detector = InformationLossDetector(self.intent)
            loss_result = detector.detect(pipeline_output)
            if loss_result.loss_detected:
                results["information_loss"] = RemediationResult(
                    success=False,
                    features_removed=loss_result.affected_features,
                    features_modified=[],
                    reason="Information loss detected (auto-remediation disabled)",
                    warnings=["Enable auto_remediate to automatically remove low-information features"],
                )

        # 3. Class imbalance (requires optional dependency)
        results["class_imbalance"] = self.remediate_class_imbalance(pipeline_output)

        return results
