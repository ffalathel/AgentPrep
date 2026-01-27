"""Validator for Level 5 policy enforcement.

This module implements enforcement logic that compares pipeline metadata
against policy rules. It returns explicit violations.

Validators check, they do not fix.
"""

from dataclasses import dataclass
from typing import Any

from intent.schema import IntentSchema

try:
    from level5_policy.data_quality_detectors import (
        ClassImbalanceDetector,
        DistributionShiftDetector,
        InformationLossDetector,
        MulticollinearityDetector,
        ProtectedAttributeDetector,
    )

    DATA_QUALITY_DETECTORS_AVAILABLE = True
except ImportError:
    DATA_QUALITY_DETECTORS_AVAILABLE = False

try:
    from level5_policy.remediation import DataQualityRemediator

    REMEDIATION_AVAILABLE = True
except ImportError:
    REMEDIATION_AVAILABLE = False


@dataclass
class ValidationResult:
    """Result of policy validation.

    This is an explicit result object - no logging as substitute.
    """

    passed: bool
    violations: list[str]

    def __str__(self) -> str:
        """Human-readable representation."""
        if self.passed:
            return "Validation PASSED: No violations"
        return f"Validation FAILED: {len(self.violations)} violation(s)"


class ConstraintValidator:
    """Validates pipeline outputs against policy constraints.

    This class compares pipeline metadata against policy rules and
    returns explicit violations. It is deterministic and auditable.
    """

    def __init__(self, intent: IntentSchema):
        """Initialize constraint validator.

        Args:
            intent: User intent (read-only)
        """
        self.intent = intent

        # Initialize data quality detectors if available
        if DATA_QUALITY_DETECTORS_AVAILABLE:
            self.distribution_shift_detector = DistributionShiftDetector(
                intent, threshold=intent.data_quality.distribution_shift_threshold
            )
            self.class_imbalance_detector = ClassImbalanceDetector(
                intent,
                severe_threshold=intent.data_quality.class_imbalance_severe_threshold,
                moderate_threshold=intent.data_quality.class_imbalance_moderate_threshold,
            )
            self.multicollinearity_detector = MulticollinearityDetector(
                intent, threshold=intent.data_quality.multicollinearity_threshold
            )
            self.protected_attribute_detector = ProtectedAttributeDetector(
                intent, custom_patterns=intent.data_quality.protected_attribute_patterns
            )
            self.information_loss_detector = InformationLossDetector(intent)
        else:
            self.distribution_shift_detector = None
            self.class_imbalance_detector = None
            self.multicollinearity_detector = None
            self.protected_attribute_detector = None
            self.information_loss_detector = None

    def validate(self, pipeline_output: dict[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
        """Validate pipeline output against constraints.

        This method checks all constraints and returns violations and warnings.
        It does NOT modify data or suggest fixes.

        Args:
            pipeline_output: Dictionary containing pipeline outputs:
                - feature_dataframe: Final feature DataFrame
                - target_column: Normalized target column name
                - feature_provenance: List of feature provenance records
                - schema_metadata: Schema metadata from Level 1
                - metadata_path: Path to metadata file (if exists)
                - original_column_count: Number of columns in original dataset

        Returns:
            Tuple of (blocking violation messages, warning messages, remediation_info dict)
        """
        violations: list[str] = []
        warnings: list[str] = []
        remediation_info: dict[str, Any] = {}

        # Validate feature count
        feature_violation = self._validate_feature_count(pipeline_output)
        if feature_violation:
            violations.append(feature_violation)

        # Validate interaction count
        interaction_violation = self._validate_interaction_count(pipeline_output)
        if interaction_violation:
            violations.append(interaction_violation)

        # Validate target column presence
        target_presence_violation = self._validate_target_presence(pipeline_output)
        if target_presence_violation:
            violations.append(target_presence_violation)

        # Validate metadata completeness
        metadata_violation = self._validate_metadata_completeness(pipeline_output)
        if metadata_violation:
            violations.append(metadata_violation)

        # Data quality checks (warnings or violations based on config)
        if DATA_QUALITY_DETECTORS_AVAILABLE:
            quality_violations, quality_warnings, quality_remediation = self._validate_data_quality(pipeline_output)
            violations.extend(quality_violations)
            warnings.extend(quality_warnings)
            remediation_info.update(quality_remediation)

        return violations, warnings, remediation_info

    def _validate_feature_count(self, pipeline_output: dict[str, Any]) -> str:
        """Validate feature count constraint.

        Returns:
            Violation message if violated, empty string otherwise
        """
        feature_dataframe = pipeline_output.get("feature_dataframe")
        target_column = pipeline_output.get("target_column")

        if feature_dataframe is None or target_column is None:
            return ""  # Cannot validate

        feature_count = len([col for col in feature_dataframe.columns if col != target_column])
        max_features = self.intent.constraints.max_features

        if feature_count > max_features:
            return f"Feature count ({feature_count}) exceeds maximum ({max_features})"

        return ""

    def _validate_interaction_count(self, pipeline_output: dict[str, Any]) -> str:
        """Validate interaction count constraint.

        Returns:
            Violation message if violated, empty string otherwise
        """
        feature_provenance = pipeline_output.get("feature_provenance")
        if feature_provenance is None:
            return ""  # Cannot validate

        interaction_count = sum(1 for prov in feature_provenance if len(prov.source_columns) > 1)
        max_interactions = self.intent.constraints.max_interactions

        if interaction_count > max_interactions:
            return f"Interaction count ({interaction_count}) exceeds maximum ({max_interactions})"

        return ""

    def _validate_target_presence(self, pipeline_output: dict[str, Any]) -> str:
        """Validate that target column is present.

        Returns:
            Violation message if violated, empty string otherwise
        """
        feature_dataframe = pipeline_output.get("feature_dataframe")
        target_column = pipeline_output.get("target_column")

        if feature_dataframe is None or target_column is None:
            return ""  # Cannot validate

        if target_column not in feature_dataframe.columns:
            return f"Target column '{target_column}' not found in final dataset"

        return ""

    def _validate_metadata_completeness(self, pipeline_output: dict[str, Any]) -> str:
        """Validate that metadata is complete.

        Returns:
            Violation message if violated, empty string otherwise
        """
        missing = []

        if pipeline_output.get("metadata_path") is None:
            missing.append("metadata file")

        if pipeline_output.get("schema_metadata") is None:
            missing.append("schema metadata")

        if pipeline_output.get("feature_provenance") is None:
            missing.append("feature provenance")

        if missing:
            return f"Missing metadata: {', '.join(missing)}"

        return ""

    def _validate_data_quality(self, pipeline_output: dict[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
        """Validate data quality issues.

        Returns:
            Tuple of (blocking violation messages, warning messages, remediation_info dict)
        """
        violations = []
        warnings = []
        remediation_info: dict[str, Any] = {}
        warn_only = self.intent.data_quality.warn_on_quality_issues

        # Distribution shift detection
        if self.distribution_shift_detector:
            try:
                shift_result = self.distribution_shift_detector.detect(pipeline_output)
                if shift_result.shift_detected:
                    msg = f"Distribution shift: {shift_result.reason}"
                    if shift_result.shifted_features:
                        msg += f" (features: {', '.join(shift_result.shifted_features[:5])}"
                        if len(shift_result.shifted_features) > 5:
                            msg += f" and {len(shift_result.shifted_features) - 5} more"
                        msg += ")"
                    if not warn_only:
                        violations.append(msg)
            except Exception:
                pass  # Skip if detection fails

        # Class imbalance detection (classification only)
        if self.class_imbalance_detector:
            try:
                imbalance_result = self.class_imbalance_detector.detect(pipeline_output)
                if imbalance_result.imbalance_detected:
                    msg = f"Class imbalance: {imbalance_result.reason}"
                    if not warn_only and imbalance_result.severity == "severe":
                        violations.append(msg)
            except Exception:
                pass  # Skip if detection fails

        # Multicollinearity detection
        if self.multicollinearity_detector:
            try:
                multicoll_result = self.multicollinearity_detector.detect(pipeline_output)
                if multicoll_result.multicollinearity_detected:
                    msg = f"Multicollinearity: {multicoll_result.reason}"
                    if multicoll_result.highly_correlated_pairs:
                        pairs_str = ", ".join(
                            [f"{f1}-{f2}(r={r:.2f})" for f1, f2, r in multicoll_result.highly_correlated_pairs[:3]]
                        )
                        msg += f" (pairs: {pairs_str}"
                        if len(multicoll_result.highly_correlated_pairs) > 3:
                            msg += f" and {len(multicoll_result.highly_correlated_pairs) - 3} more"
                        msg += ")"
                    if not warn_only:
                        violations.append(msg)
                    # Store remediation info: features to remove (second feature in each pair)
                    features_to_remove = {feat2 for _, feat2, _ in multicoll_result.highly_correlated_pairs}
                    remediation_info["multicollinearity"] = {
                        "features_to_remove": sorted(features_to_remove),
                        "correlated_pairs": multicoll_result.highly_correlated_pairs,
                    }
            except Exception:
                pass  # Skip if detection fails

        # Protected attribute detection (always warn, never block)
        if self.protected_attribute_detector:
            try:
                protected_result = self.protected_attribute_detector.detect(pipeline_output)
                if protected_result.protected_attributes_detected:
                    msg = f"Protected attributes detected: {', '.join(protected_result.detected_attributes)}"
                    # Always warn, never block (user decision)
                    warnings.append(msg)
            except Exception:
                pass  # Skip if detection fails

        # Information loss detection
        if self.information_loss_detector:
            try:
                loss_result = self.information_loss_detector.detect(pipeline_output)
                if loss_result.loss_detected:
                    msg = f"Information loss: {loss_result.reason}"
                    if loss_result.affected_features:
                        msg += f" (features: {', '.join(loss_result.affected_features[:5])}"
                        if len(loss_result.affected_features) > 5:
                            msg += f" and {len(loss_result.affected_features) - 5} more"
                        msg += ")"
                    if not warn_only:
                        violations.append(msg)
                    # Store remediation info: features to remove
                    remediation_info["information_loss"] = {
                        "features_to_remove": loss_result.affected_features,
                        "loss_reasons": loss_result.loss_reasons,
                    }
            except Exception:
                pass  # Skip if detection fails

        # Class imbalance detection (store for potential remediation)
        if self.class_imbalance_detector:
            try:
                imbalance_result = self.class_imbalance_detector.detect(pipeline_output)
                if imbalance_result.imbalance_detected:
                    remediation_info["class_imbalance"] = {
                        "imbalance_ratio": imbalance_result.imbalance_ratio,
                        "severity": imbalance_result.severity,
                        "class_distribution": imbalance_result.class_distribution,
                    }
            except Exception:
                pass  # Skip if detection fails

        return violations, warnings, remediation_info
