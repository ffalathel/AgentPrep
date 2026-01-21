"""Data quality detectors for Level 5 policy enforcement.

This module implements detectors for various data quality issues:
- Distribution shift detection
- Class imbalance detection
- Multicollinearity detection
- Protected attribute detection
- Information loss detection

These detectors return warnings/violations but do not modify data.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

from intent.schema import IntentSchema, TaskType


@dataclass
class DistributionShiftResult:
    """Result of distribution shift detection."""

    shift_detected: bool
    shifted_features: list[str]
    shift_scores: dict[str, float]  # Feature -> KS statistic
    reason: str


@dataclass
class ClassImbalanceResult:
    """Result of class imbalance detection."""

    imbalance_detected: bool
    imbalance_ratio: float  # Ratio of minority to majority class
    class_distribution: dict[Any, int]
    severity: str  # "low", "moderate", "severe"
    reason: str


@dataclass
class MulticollinearityResult:
    """Result of multicollinearity detection."""

    multicollinearity_detected: bool
    highly_correlated_pairs: list[tuple[str, str, float]]  # (feat1, feat2, correlation)
    reason: str


@dataclass
class ProtectedAttributeResult:
    """Result of protected attribute detection."""

    protected_attributes_detected: bool
    detected_attributes: list[str]
    reason: str


@dataclass
class InformationLossResult:
    """Result of information loss detection."""

    loss_detected: bool
    affected_features: list[str]
    loss_reasons: dict[str, str]  # Feature -> reason
    reason: str


class DistributionShiftDetector:
    """Detects distribution shift between train/test or temporal drift."""

    def __init__(self, intent: IntentSchema, threshold: float = 0.1):
        """Initialize distribution shift detector.

        Args:
            intent: User intent (read-only)
            threshold: KS statistic threshold for shift detection (default: 0.1)
        """
        self.intent = intent
        self.threshold = threshold

    def detect(
        self,
        pipeline_output: dict[str, Any],
        reference_dataframe: Optional[pd.DataFrame] = None,
    ) -> DistributionShiftResult:
        """Detect distribution shift.

        If reference_dataframe is provided, compares current data against reference.
        Otherwise, uses temporal split (first 80% vs last 20%) for time series.

        Args:
            pipeline_output: Dictionary containing pipeline outputs
            reference_dataframe: Optional reference DataFrame for comparison

        Returns:
            DistributionShiftResult with detected shifts
        """
        feature_dataframe = pipeline_output.get("feature_dataframe")
        if feature_dataframe is None or len(feature_dataframe) < 20:
            return DistributionShiftResult(
                shift_detected=False,
                shifted_features=[],
                shift_scores={},
                reason="Insufficient data for distribution shift detection",
            )

        shifted_features = []
        shift_scores = {}

        # Get numeric columns only
        numeric_cols = feature_dataframe.select_dtypes(include=[np.number]).columns.tolist()
        target_column = pipeline_output.get("target_column")
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)

        if not numeric_cols:
            return DistributionShiftResult(
                shift_detected=False,
                shifted_features=[],
                shift_scores={},
                reason="No numeric features to check for distribution shift",
            )

        # Determine comparison strategy
        if reference_dataframe is not None:
            # Compare against reference dataset
            current_df = feature_dataframe[numeric_cols]
            ref_df = reference_dataframe[numeric_cols]
            current_data = current_df
            reference_data = ref_df
        elif self.intent.task.type == TaskType.TIME_SERIES:
            # Temporal split for time series
            split_idx = int(len(feature_dataframe) * 0.8)
            current_data = feature_dataframe.iloc[:split_idx][numeric_cols]
            reference_data = feature_dataframe.iloc[split_idx:][numeric_cols]
        else:
            # For non-time-series, use random split (80/20)
            np.random.seed(42)  # For reproducibility
            mask = np.random.rand(len(feature_dataframe)) < 0.8
            current_data = feature_dataframe[mask][numeric_cols]
            reference_data = feature_dataframe[~mask][numeric_cols]

        # Perform Kolmogorov-Smirnov test for each numeric feature
        for col in numeric_cols:
            try:
                current_vals = current_data[col].dropna()
                ref_vals = reference_data[col].dropna()

                if len(current_vals) < 10 or len(ref_vals) < 10:
                    continue

                # KS test (if scipy available)
                if SCIPY_AVAILABLE:
                    ks_statistic, _ = stats.ks_2samp(current_vals, ref_vals)
                    shift_scores[col] = ks_statistic
                else:
                    # Fallback: use simple mean/std comparison
                    mean_diff = abs(current_vals.mean() - ref_vals.mean())
                    std_combined = (current_vals.std() + ref_vals.std()) / 2
                    if std_combined > 0:
                        ks_statistic = mean_diff / std_combined
                    else:
                        ks_statistic = 0.0
                    shift_scores[col] = ks_statistic

                if ks_statistic >= self.threshold:
                    shifted_features.append(col)
            except Exception:
                # Skip if test fails
                continue

        shift_detected = len(shifted_features) > 0
        reason = (
            f"Distribution shift detected in {len(shifted_features)} feature(s)"
            if shift_detected
            else "No significant distribution shift detected"
        )

        return DistributionShiftResult(
            shift_detected=shift_detected,
            shifted_features=shifted_features,
            shift_scores=shift_scores,
            reason=reason,
        )


class ClassImbalanceDetector:
    """Detects class imbalance in classification tasks."""

    def __init__(
        self,
        intent: IntentSchema,
        severe_threshold: float = 0.1,
        moderate_threshold: float = 0.3,
    ):
        """Initialize class imbalance detector.

        Args:
            intent: User intent (read-only)
            severe_threshold: Ratio threshold for severe imbalance (default: 0.1)
            moderate_threshold: Ratio threshold for moderate imbalance (default: 0.3)
        """
        self.intent = intent
        self.severe_threshold = severe_threshold
        self.moderate_threshold = moderate_threshold

    def detect(self, pipeline_output: dict[str, Any]) -> ClassImbalanceResult:
        """Detect class imbalance.

        Only runs for classification tasks.

        Args:
            pipeline_output: Dictionary containing pipeline outputs

        Returns:
            ClassImbalanceResult with imbalance information
        """
        # Only check for classification tasks
        if self.intent.task.type != TaskType.CLASSIFICATION:
            return ClassImbalanceResult(
                imbalance_detected=False,
                imbalance_ratio=1.0,
                class_distribution={},
                severity="none",
                reason="Class imbalance check only applies to classification tasks",
            )

        feature_dataframe = pipeline_output.get("feature_dataframe")
        target_column = pipeline_output.get("target_column")

        if feature_dataframe is None or target_column is None:
            return ClassImbalanceResult(
                imbalance_detected=False,
                imbalance_ratio=1.0,
                class_distribution={},
                severity="none",
                reason="Cannot detect class imbalance: missing data",
            )

        if target_column not in feature_dataframe.columns:
            return ClassImbalanceResult(
                imbalance_detected=False,
                imbalance_ratio=1.0,
                class_distribution={},
                severity="none",
                reason=f"Target column '{target_column}' not found",
            )

        # Get class distribution
        target_series = feature_dataframe[target_column].dropna()
        class_counts = target_series.value_counts().to_dict()

        if len(class_counts) < 2:
            return ClassImbalanceResult(
                imbalance_detected=False,
                imbalance_ratio=1.0,
                class_distribution=class_counts,
                severity="none",
                reason="Less than 2 classes found",
            )

        # Calculate imbalance ratio (minority / majority)
        counts = list(class_counts.values())
        majority_count = max(counts)
        minority_count = min(counts)
        imbalance_ratio = minority_count / majority_count if majority_count > 0 else 0.0

        # Determine severity
        if imbalance_ratio < self.severe_threshold:
            severity = "severe"
        elif imbalance_ratio < self.moderate_threshold:
            severity = "moderate"
        else:
            severity = "low"

        imbalance_detected = severity in ["moderate", "severe"]

        reason = (
            f"{severity.capitalize()} class imbalance detected (ratio: {imbalance_ratio:.3f})"
            if imbalance_detected
            else f"Balanced classes (ratio: {imbalance_ratio:.3f})"
        )

        return ClassImbalanceResult(
            imbalance_detected=imbalance_detected,
            imbalance_ratio=imbalance_ratio,
            class_distribution=class_counts,
            severity=severity,
            reason=reason,
        )


class MulticollinearityDetector:
    """Detects multicollinearity between features."""

    def __init__(self, intent: IntentSchema, threshold: float = 0.9):
        """Initialize multicollinearity detector.

        Args:
            intent: User intent (read-only)
            threshold: Correlation threshold for multicollinearity (default: 0.9)
        """
        self.intent = intent
        self.threshold = threshold

    def detect(self, pipeline_output: dict[str, Any]) -> MulticollinearityResult:
        """Detect multicollinearity between features.

        Args:
            pipeline_output: Dictionary containing pipeline outputs

        Returns:
            MulticollinearityResult with highly correlated pairs
        """
        feature_dataframe = pipeline_output.get("feature_dataframe")
        target_column = pipeline_output.get("target_column")

        if feature_dataframe is None:
            return MulticollinearityResult(
                multicollinearity_detected=False,
                highly_correlated_pairs=[],
                reason="Cannot detect multicollinearity: missing data",
            )

        # Get numeric columns only
        numeric_cols = feature_dataframe.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)

        if len(numeric_cols) < 2:
            return MulticollinearityResult(
                multicollinearity_detected=False,
                highly_correlated_pairs=[],
                reason="Insufficient numeric features for multicollinearity detection",
            )

        # Compute correlation matrix
        try:
            corr_matrix = feature_dataframe[numeric_cols].corr()
        except Exception:
            return MulticollinearityResult(
                multicollinearity_detected=False,
                highly_correlated_pairs=[],
                reason="Failed to compute correlation matrix",
            )

        # Find highly correlated pairs
        highly_correlated_pairs = []
        seen_pairs = set()

        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1 :]:
                pair_key = tuple(sorted([col1, col2]))
                if pair_key in seen_pairs:
                    continue

                try:
                    corr_value = corr_matrix.loc[col1, col2]
                    if not np.isnan(corr_value) and abs(corr_value) >= self.threshold:
                        highly_correlated_pairs.append((col1, col2, corr_value))
                        seen_pairs.add(pair_key)
                except Exception:
                    continue

        multicollinearity_detected = len(highly_correlated_pairs) > 0
        reason = (
            f"Multicollinearity detected: {len(highly_correlated_pairs)} highly correlated feature pair(s)"
            if multicollinearity_detected
            else "No significant multicollinearity detected"
        )

        return MulticollinearityResult(
            multicollinearity_detected=multicollinearity_detected,
            highly_correlated_pairs=highly_correlated_pairs,
            reason=reason,
        )


class ProtectedAttributeDetector:
    """Detects protected/sensitive attributes."""

    # Common protected attribute patterns
    PROTECTED_PATTERNS = [
        "race",
        "ethnicity",
        "gender",
        "sex",
        "age",
        "religion",
        "disability",
        "marital_status",
        "sexual_orientation",
        "national_origin",
        "zip_code",
        "postal_code",
        "ssn",
        "social_security",
        "credit_score",  # Can be proxy for protected attributes
    ]

    def __init__(self, intent: IntentSchema, custom_patterns: Optional[list[str]] = None):
        """Initialize protected attribute detector.

        Args:
            intent: User intent (read-only)
            custom_patterns: Optional list of custom patterns to check
        """
        self.intent = intent
        self.patterns = list(self.PROTECTED_PATTERNS)
        if custom_patterns:
            self.patterns.extend(custom_patterns)

    def detect(self, pipeline_output: dict[str, Any]) -> ProtectedAttributeResult:
        """Detect protected attributes.

        Args:
            pipeline_output: Dictionary containing pipeline outputs

        Returns:
            ProtectedAttributeResult with detected attributes
        """
        feature_dataframe = pipeline_output.get("feature_dataframe")
        if feature_dataframe is None:
            return ProtectedAttributeResult(
                protected_attributes_detected=False,
                detected_attributes=[],
                reason="Cannot detect protected attributes: missing data",
            )

        detected_attributes = []
        columns_lower = {col.lower(): col for col in feature_dataframe.columns}

        for pattern in self.patterns:
            pattern_lower = pattern.lower()
            for col_lower, col_original in columns_lower.items():
                if pattern_lower in col_lower and col_original not in detected_attributes:
                    detected_attributes.append(col_original)

        protected_attributes_detected = len(detected_attributes) > 0
        reason = (
            f"Protected attributes detected: {', '.join(detected_attributes)}"
            if protected_attributes_detected
            else "No protected attributes detected"
        )

        return ProtectedAttributeResult(
            protected_attributes_detected=protected_attributes_detected,
            detected_attributes=detected_attributes,
            reason=reason,
        )


class InformationLossDetector:
    """Detects information loss from transformations."""

    def __init__(self, intent: IntentSchema):
        """Initialize information loss detector.

        Args:
            intent: User intent (read-only)
        """
        self.intent = intent

    def detect(self, pipeline_output: dict[str, Any]) -> InformationLossResult:
        """Detect information loss.

        Checks for:
        - Constant/near-constant features (after transformations)
        - Excessive binning/aggregation
        - Features with very low variance

        Args:
            pipeline_output: Dictionary containing pipeline outputs

        Returns:
            InformationLossResult with affected features
        """
        feature_dataframe = pipeline_output.get("feature_dataframe")
        target_column = pipeline_output.get("target_column")
        feature_provenance = pipeline_output.get("feature_provenance", [])

        if feature_dataframe is None:
            return InformationLossResult(
                loss_detected=False,
                affected_features=[],
                loss_reasons={},
                reason="Cannot detect information loss: missing data",
            )

        affected_features = []
        loss_reasons = {}

        # Get engineered feature names
        engineered_features = {
            getattr(prov, "feature_name", None) for prov in feature_provenance
        }
        engineered_features.discard(None)

        # Check each engineered feature
        for col in feature_dataframe.columns:
            if col == target_column:
                continue

            # Only check engineered features
            if col not in engineered_features:
                continue

            try:
                series = feature_dataframe[col].dropna()

                if len(series) == 0:
                    continue

                # Check 1: Constant or near-constant
                unique_ratio = series.nunique() / len(series)
                if unique_ratio < 0.01:  # Less than 1% unique values
                    affected_features.append(col)
                    loss_reasons[col] = f"Near-constant feature (uniqueness: {unique_ratio:.3f})"
                    continue

                # Check 2: Very low variance (for numeric)
                if pd.api.types.is_numeric_dtype(series):
                    variance = series.var()
                    if variance < 1e-10:  # Very low variance
                        affected_features.append(col)
                        loss_reasons[col] = f"Very low variance (var: {variance:.2e})"
                        continue

                # Check 3: Excessive cardinality reduction (for categorical)
                if not pd.api.types.is_numeric_dtype(series):
                    cardinality = series.nunique()
                    if cardinality == 1:
                        affected_features.append(col)
                        loss_reasons[col] = "Constant feature (single value)"

            except Exception:
                # Skip if check fails
                continue

        loss_detected = len(affected_features) > 0
        reason = (
            f"Information loss detected in {len(affected_features)} feature(s)"
            if loss_detected
            else "No significant information loss detected"
        )

        return InformationLossResult(
            loss_detected=loss_detected,
            affected_features=affected_features,
            loss_reasons=loss_reasons,
            reason=reason,
        )
