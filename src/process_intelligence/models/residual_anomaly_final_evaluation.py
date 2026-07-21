"""Final residual anomaly scoring on the held-out test partition (Step 7H).

Reuses the train-fitted regression model and validation-calibrated
``ResidualAnomalyDetector`` from a ``ResidualAnomalyPipelineOutcome``, scores
the independent test partition exactly once, and never refits, recalibrates,
or changes thresholds from test results.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Self

import numpy as np
import polars as pl
from pydantic import BaseModel, Field, field_validator, model_validator

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.exceptions import (
    DataLeakageError,
    DataValidationError,
    InsufficientDataError,
    ProcessIntelligenceError,
)
from process_intelligence.core.protocols import BaseAnalysisModel
from process_intelligence.core.schemas import ModelMetadata, ModelSpec
from process_intelligence.data.loader import ORIGINAL_ROW_ID_COLUMN
from process_intelligence.evaluation.leakage import LeakageReport, LeakageSeverity
from process_intelligence.evaluation.splitting import DatasetSplit
from process_intelligence.models.residual_anomaly import (
    ResidualAnomalyDetector,
    ResidualAnomalyResult,
    ResidualCalibration,
    ResidualThresholdMethod,
)
from process_intelligence.models.residual_anomaly_pipeline import (
    ResidualAnomalyPipelineOutcome,
    ResidualAnomalyPipelineReport,
    ResidualPartitionSummary,
)

_REGRESSION_PREDICTION_COLUMN = "_regression_prediction"
_REGRESSION_RESIDUAL_COLUMN = "_regression_residual"
_ABSOLUTE_CENTERED_RESIDUAL_COLUMN = "_absolute_centered_residual"
_RESIDUAL_ANOMALY_SCORE_COLUMN = "_residual_anomaly_score"
_IS_RESIDUAL_ANOMALY_COLUMN = "_is_residual_anomaly"
_RESIDUAL_ANOMALY_RAW_PREDICTION_COLUMN = "_residual_anomaly_raw_prediction"
_DATA_PARTITION_COLUMN = "_data_partition"
_RESERVED_RESULT_COLUMNS: tuple[str, ...] = (
    _REGRESSION_PREDICTION_COLUMN,
    _REGRESSION_RESIDUAL_COLUMN,
    _ABSOLUTE_CENTERED_RESIDUAL_COLUMN,
    _RESIDUAL_ANOMALY_SCORE_COLUMN,
    _IS_RESIDUAL_ANOMALY_COLUMN,
    _RESIDUAL_ANOMALY_RAW_PREDICTION_COLUMN,
    _DATA_PARTITION_COLUMN,
)

_VALIDATION_METRIC_KEYS: tuple[str, ...] = (
    "validation_anomaly_fraction",
    "validation_residual_mean",
    "validation_residual_std",
    "validation_score_min",
    "validation_score_max",
    "validation_score_mean",
    "validation_score_std",
    "validation_score_range",
)

_TEST_METRIC_KEYS: tuple[str, ...] = (
    "test_anomaly_fraction",
    "test_residual_mean",
    "test_residual_std",
    "test_score_min",
    "test_score_max",
    "test_score_mean",
    "test_score_std",
    "test_score_range",
)

_FLAG_FRACTION_BELOW = (
    "test residual anomaly fraction below expected minimum"
)
_FLAG_FRACTION_ABOVE = (
    "test residual anomaly fraction above expected maximum"
)
_FLAG_FRACTION_SHIFT = (
    "validation-test residual anomaly fraction shift is large"
)
_FLAG_SCORE_NEARLY_CONSTANT = (
    "test residual anomaly scores are nearly constant"
)
_FLAG_DEGENERATE = "test residual anomaly detection is degenerate"
_FLAG_NEGATIVE_SEPARATION = "residual anomaly score direction is inconsistent"
_FLAG_SCORE_MEAN_SHIFT = (
    "test residual anomaly score mean shifted substantially"
)
_FLAG_SCORE_STD_CONTRACTED = (
    "test residual anomaly score variation contracted substantially"
)
_FLAG_SCORE_STD_EXPANDED = (
    "test residual anomaly score variation expanded substantially"
)

_WARNING_LEAKAGE = (
    "LeakageReport contains WARNING issues; proceed with caution"
)
_WARNING_QUALITY_FLAGS = (
    "Final residual anomaly evaluation reported one or more quality flags"
)
_WARNING_DISTRIBUTION_SHIFT = (
    "Validation-test residual anomaly score distribution shifted "
    "beyond policy limits"
)
_WARNING_NO_ANOMALIES = "Test partition contains no residual anomalies"
_WARNING_ALL_ANOMALIES = (
    "Test partition marks every row as a residual anomaly"
)


def _is_finite_number(value: object) -> bool:
    """Return True when ``value`` is a finite real number (bool excluded)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _require_real_bool(value: object, *, field_name: str) -> bool:
    """Validate a real bool (reject int 0/1 and bool-like strings)."""
    if type(value) is not bool:
        raise ValueError(
            f"{field_name} must be a bool, got {type(value).__name__}"
        )
    return value


def _require_positive_int(value: object, *, field_name: str) -> int:
    """Validate a non-bool integer >= 1."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{field_name} must be an int >= 1 (bool not allowed), "
            f"got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1, got {value}")
    return value


def _require_non_negative_finite_float(value: object, *, field_name: str) -> float:
    """Validate a finite float >= 0 (bool excluded)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a finite float >= 0 (bool not allowed), "
            f"got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    if number < 0.0:
        raise ValueError(f"{field_name} must be >= 0, got {number}")
    return number


def _require_positive_finite_float(value: object, *, field_name: str) -> float:
    """Validate a finite float > 0 (bool excluded)."""
    number = _require_non_negative_finite_float(value, field_name=field_name)
    if number <= 0.0:
        raise ValueError(f"{field_name} must be > 0, got {number}")
    return number


def _require_fraction(value: object, *, field_name: str) -> float:
    """Validate a finite float in [0.0, 1.0]."""
    number = _require_non_negative_finite_float(value, field_name=field_name)
    if number > 1.0:
        raise ValueError(f"{field_name} must be <= 1.0, got {number}")
    return number


def _require_finite_float(value: object, *, field_name: str) -> float:
    """Validate a finite float allowing any sign (bool excluded)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a finite float (bool not allowed), "
            f"got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return number


def _reject_duplicate_strings(values: list[str], *, field_name: str) -> list[str]:
    """Reject duplicate strings while preserving order."""
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicate values")
    return list(values)


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    """Return unique strings preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _fractions_match(actual: float, expected: float) -> bool:
    """Compare fractions with a tight absolute tolerance."""
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)


def _timings_match(actual: float, expected: float) -> bool:
    """Compare timing sums with a tight absolute tolerance."""
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)


def _values_close(actual: float, expected: float) -> bool:
    """Compare floats with a tight absolute tolerance."""
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)


class ResidualAnomalyFinalEvaluationPolicy(BaseModel):
    """Policy knobs for residual anomaly final test scoring.

    Controls leakage gating, pipeline and split-summary consistency,
    minimum test size, and label-free quality / distribution-shift thresholds.
    """

    require_safe_leakage_report: bool = True
    require_pipeline_consistency: bool = True
    require_split_summary_match: bool = True
    minimum_test_rows: int = 1
    minimum_anomaly_fraction: float = 0.001
    maximum_anomaly_fraction: float = 0.25
    maximum_anomaly_fraction_shift: float = 0.10
    minimum_score_std: float = 1e-12
    maximum_standardized_score_mean_shift: float = 1.0
    minimum_score_std_ratio: float = 0.5
    maximum_score_std_ratio: float = 2.0
    warn_on_distribution_shift: bool = True

    @field_validator(
        "require_safe_leakage_report",
        "require_pipeline_consistency",
        "require_split_summary_match",
        "warn_on_distribution_shift",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_real_bool(value, field_name="policy flag")

    @field_validator("minimum_test_rows", mode="before")
    @classmethod
    def _validate_minimum_test_rows(cls, value: object) -> int:
        return _require_positive_int(value, field_name="minimum_test_rows")

    @field_validator(
        "minimum_anomaly_fraction",
        "maximum_anomaly_fraction",
        "maximum_anomaly_fraction_shift",
        mode="before",
    )
    @classmethod
    def _validate_fraction_fields(cls, value: object) -> float:
        return _require_fraction(value, field_name="fraction field")

    @field_validator("minimum_score_std", mode="before")
    @classmethod
    def _validate_minimum_score_std(cls, value: object) -> float:
        return _require_positive_finite_float(
            value, field_name="minimum_score_std"
        )

    @field_validator("maximum_standardized_score_mean_shift", mode="before")
    @classmethod
    def _validate_maximum_standardized_score_mean_shift(
        cls, value: object
    ) -> float:
        return _require_positive_finite_float(
            value, field_name="maximum_standardized_score_mean_shift"
        )

    @field_validator("minimum_score_std_ratio", mode="before")
    @classmethod
    def _validate_minimum_score_std_ratio(cls, value: object) -> float:
        return _require_non_negative_finite_float(
            value, field_name="minimum_score_std_ratio"
        )

    @field_validator("maximum_score_std_ratio", mode="before")
    @classmethod
    def _validate_maximum_score_std_ratio(cls, value: object) -> float:
        return _require_positive_finite_float(
            value, field_name="maximum_score_std_ratio"
        )

    @model_validator(mode="after")
    def _validate_relationships(self) -> Self:
        if self.minimum_anomaly_fraction >= self.maximum_anomaly_fraction:
            raise ValueError(
                "minimum_anomaly_fraction must be < maximum_anomaly_fraction, "
                f"got {self.minimum_anomaly_fraction} >= "
                f"{self.maximum_anomaly_fraction}"
            )
        if self.minimum_score_std_ratio >= self.maximum_score_std_ratio:
            raise ValueError(
                "minimum_score_std_ratio must be < maximum_score_std_ratio, "
                f"got {self.minimum_score_std_ratio} >= "
                f"{self.maximum_score_std_ratio}"
            )
        return self


class ResidualAnomalyFinalEvaluationReport(BaseModel):
    """Structured report of one residual anomaly final test scoring.

    Preserves validation residual summaries from the pipeline report, records
    test residual metrics and validation-test distribution shifts, and never
    stores estimators, full prediction arrays, or input frames.
    """

    task: AnalysisTask
    model_name: str
    estimator_key: str
    target_column: str
    feature_columns: list[str]
    detector_method: ResidualThresholdMethod
    calibration_partition: Literal["validation"] = "validation"
    test_partition: Literal["test"] = "test"
    train_row_count: int
    validation_row_count: int
    test_row_count: int
    regression_fit_row_count: int
    calibration_row_count: int
    residual_center: float
    residual_scale: float | None
    threshold: float
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    test_has_normal_and_anomaly: bool
    test_score_separation: float | None = None
    anomaly_fraction_shift: float
    score_mean_shift: float
    standardized_score_mean_shift: float
    score_std_ratio: float
    quality_flags: list[str] = Field(default_factory=list)
    prediction_seconds: float
    scoring_seconds: float
    total_seconds: float
    evaluated_at: datetime
    warnings: list[str] = Field(default_factory=list)

    @field_validator("task", mode="after")
    @classmethod
    def _validate_task(cls, value: AnalysisTask) -> AnalysisTask:
        if value is not AnalysisTask.REGRESSION:
            raise ValueError(
                f"task must be AnalysisTask.REGRESSION, got {value!r}"
            )
        return value

    @field_validator(
        "model_name",
        "estimator_key",
        "target_column",
        mode="before",
    )
    @classmethod
    def _validate_non_blank_text(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(f"must be str, got {type(value).__name__}")
        if value == "" or value.strip() == "":
            raise ValueError("must be a non-empty, non-whitespace string")
        return value

    @field_validator("feature_columns", mode="after")
    @classmethod
    def _validate_feature_columns(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("feature_columns must contain at least one feature")
        if len(value) != len(set(value)):
            raise ValueError("feature_columns must not contain duplicates")
        if ORIGINAL_ROW_ID_COLUMN in value:
            raise ValueError(
                f"feature_columns must not include {ORIGINAL_ROW_ID_COLUMN!r}"
            )
        return list(value)

    @field_validator("calibration_partition", mode="after")
    @classmethod
    def _validate_calibration_partition(
        cls, value: Literal["validation"]
    ) -> Literal["validation"]:
        if value != "validation":
            raise ValueError(
                f"calibration_partition must be 'validation', got {value!r}"
            )
        return value

    @field_validator("test_partition", mode="after")
    @classmethod
    def _validate_test_partition(
        cls, value: Literal["test"]
    ) -> Literal["test"]:
        if value != "test":
            raise ValueError(f"test_partition must be 'test', got {value!r}")
        return value

    @field_validator(
        "train_row_count",
        "validation_row_count",
        "test_row_count",
        "regression_fit_row_count",
        "calibration_row_count",
        mode="before",
    )
    @classmethod
    def _validate_positive_row_counts(cls, value: object) -> int:
        return _require_positive_int(value, field_name="row count")

    @field_validator("residual_center", mode="before")
    @classmethod
    def _validate_residual_center(cls, value: object) -> float:
        return _require_finite_float(value, field_name="residual_center")

    @field_validator("residual_scale", mode="before")
    @classmethod
    def _validate_residual_scale(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_positive_finite_float(
            value, field_name="residual_scale"
        )

    @field_validator("threshold", mode="before")
    @classmethod
    def _validate_threshold(cls, value: object) -> float:
        return _require_non_negative_finite_float(
            value, field_name="threshold"
        )

    @field_validator("validation_metrics", "test_metrics", mode="before")
    @classmethod
    def _validate_metrics_dict(cls, value: object) -> dict[str, float]:
        if not isinstance(value, dict):
            raise ValueError(
                f"metrics must be a dict[str, float], got {type(value).__name__}"
            )
        if not value:
            raise ValueError("metrics must not be empty")
        cleaned: dict[str, float] = {}
        for key, raw in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("metrics keys must be non-empty strings")
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(
                    f"metrics[{key!r}] must be a finite number "
                    f"(bool not allowed), got {type(raw).__name__}"
                )
            number = float(raw)
            if not math.isfinite(number):
                raise ValueError(f"metrics[{key!r}] must be finite, got {raw!r}")
            cleaned[key] = number
        return cleaned

    @field_validator("test_has_normal_and_anomaly", mode="before")
    @classmethod
    def _validate_has_groups(cls, value: object) -> bool:
        return _require_real_bool(value, field_name="test_has_normal_and_anomaly")

    @field_validator("test_score_separation", mode="before")
    @classmethod
    def _validate_optional_separation(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_finite_float(value, field_name="test_score_separation")

    @field_validator("anomaly_fraction_shift", mode="before")
    @classmethod
    def _validate_anomaly_fraction_shift(cls, value: object) -> float:
        return _require_fraction(value, field_name="anomaly_fraction_shift")

    @field_validator("score_mean_shift", mode="before")
    @classmethod
    def _validate_score_mean_shift(cls, value: object) -> float:
        return _require_finite_float(value, field_name="score_mean_shift")

    @field_validator(
        "standardized_score_mean_shift",
        "score_std_ratio",
        mode="before",
    )
    @classmethod
    def _validate_non_negative_shift(cls, value: object) -> float:
        return _require_non_negative_finite_float(
            value, field_name="shift/ratio field"
        )

    @field_validator(
        "prediction_seconds",
        "scoring_seconds",
        "total_seconds",
        mode="before",
    )
    @classmethod
    def _validate_timing(cls, value: object) -> float:
        return _require_non_negative_finite_float(value, field_name="timing")

    @field_validator("evaluated_at", mode="after")
    @classmethod
    def _validate_evaluated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("evaluated_at must be timezone-aware")
        return value

    @field_validator("quality_flags", "warnings", mode="after")
    @classmethod
    def _validate_string_lists(cls, value: list[str]) -> list[str]:
        return _reject_duplicate_strings(value, field_name="list field")

    @model_validator(mode="after")
    def _validate_report_consistency(self) -> Self:
        if self.target_column in self.feature_columns:
            raise ValueError(
                "target_column must not be included in feature_columns"
            )
        if self.regression_fit_row_count != self.train_row_count:
            raise ValueError(
                "regression_fit_row_count must equal train_row_count "
                f"({self.train_row_count}), got {self.regression_fit_row_count}"
            )
        if self.calibration_row_count != self.validation_row_count:
            raise ValueError(
                "calibration_row_count must equal validation_row_count "
                f"({self.validation_row_count}), got {self.calibration_row_count}"
            )

        for key in _VALIDATION_METRIC_KEYS:
            if key not in self.validation_metrics:
                raise ValueError(f"validation_metrics requires key {key!r}")
        for key in _TEST_METRIC_KEYS:
            if key not in self.test_metrics:
                raise ValueError(f"test_metrics requires key {key!r}")

        self._validate_metric_bundle(
            self.validation_metrics,
            prefix="validation",
        )
        self._validate_metric_bundle(self.test_metrics, prefix="test")

        if self.test_has_normal_and_anomaly:
            if self.test_score_separation is None:
                raise ValueError(
                    "test_score_separation is required when "
                    "test_has_normal_and_anomaly=True"
                )
            if not _is_finite_number(self.test_score_separation):
                raise ValueError(
                    "test_score_separation must be finite when "
                    "test_has_normal_and_anomaly=True"
                )

        expected_total = self.prediction_seconds + self.scoring_seconds
        if not _timings_match(self.total_seconds, expected_total):
            raise ValueError(
                "total_seconds must equal prediction_seconds + scoring_seconds "
                f"({expected_total}), got {self.total_seconds}"
            )
        return self

    @staticmethod
    def _validate_metric_bundle(
        metrics: Mapping[str, float],
        *,
        prefix: str,
    ) -> None:
        """Validate shared residual metric invariants for one partition."""
        fraction = metrics[f"{prefix}_anomaly_fraction"]
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError(
                f"{prefix}_anomaly_fraction must be in [0.0, 1.0], got {fraction}"
            )
        residual_std = metrics[f"{prefix}_residual_std"]
        score_std = metrics[f"{prefix}_score_std"]
        if residual_std < 0.0:
            raise ValueError(
                f"{prefix}_residual_std must be >= 0, got {residual_std}"
            )
        if score_std < 0.0:
            raise ValueError(f"{prefix}_score_std must be >= 0, got {score_std}")
        score_min = metrics[f"{prefix}_score_min"]
        score_mean = metrics[f"{prefix}_score_mean"]
        score_max = metrics[f"{prefix}_score_max"]
        if not (score_min <= score_mean <= score_max):
            raise ValueError(
                f"{prefix} score summaries must satisfy "
                "score_min <= score_mean <= score_max"
            )
        expected_range = score_max - score_min
        actual_range = metrics[f"{prefix}_score_range"]
        if not _values_close(actual_range, expected_range):
            raise ValueError(
                f"{prefix}_score_range must equal score_max - score_min "
                f"({expected_range}), got {actual_range}"
            )


@dataclass(frozen=True, slots=True)
class ResidualAnomalyFinalEvaluationOutcome:
    """Immutable result of one residual anomaly final test scoring."""

    regression_model: BaseAnalysisModel
    residual_detector: ResidualAnomalyDetector
    test_scored: pl.DataFrame
    report: ResidualAnomalyFinalEvaluationReport


class ResidualAnomalyFinalEvaluator:
    """Score residual anomalies on test using a completed pipeline outcome.

    Reuses the pipeline's train-fitted regression model and
    validation-calibrated detector without refitting or recalibration.
    Evaluator instances do not cache models, detectors, or outcomes.
    """

    def __init__(
        self,
        *,
        policy: ResidualAnomalyFinalEvaluationPolicy | None = None,
    ) -> None:
        """Create an evaluator with an isolated policy copy.

        Args:
            policy: Final-evaluation policy. ``None`` uses defaults.
                Deep-copied so later mutations do not affect this evaluator.

        Raises:
            TypeError: If ``policy`` has an invalid type.
        """
        if policy is None:
            stored_policy = ResidualAnomalyFinalEvaluationPolicy()
        elif isinstance(policy, ResidualAnomalyFinalEvaluationPolicy):
            stored_policy = policy.model_copy(deep=True)
        else:
            raise TypeError(
                "policy must be ResidualAnomalyFinalEvaluationPolicy or None, "
                f"got {type(policy).__name__}"
            )
        self._policy = stored_policy

    def evaluate(
        self,
        split: DatasetSplit,
        pipeline_outcome: ResidualAnomalyPipelineOutcome,
        *,
        target_column: str,
        feature_columns: Sequence[str],
        leakage_report: LeakageReport,
    ) -> ResidualAnomalyFinalEvaluationOutcome:
        """Score test residuals once without mutating inputs or fitted state.

        Args:
            split: Train/validation/test partitions with summary metadata.
            pipeline_outcome: Pipeline outcome providing the fitted regression
                model and validation-calibrated residual detector.
            target_column: Regression target column name.
            feature_columns: Feature names used for prediction, in order.
            leakage_report: Structural leakage report aligned to features.

        Returns:
            Outcome with the reused regression model, residual detector,
            scored test frame, and final-evaluation report.

        Raises:
            TypeError: If inputs have invalid types.
            DataValidationError: If structural or consistency validation fails.
            DataLeakageError: If blockers are present under a safe-leakage policy.
            InsufficientDataError: If partition sizes are insufficient.
            ProcessIntelligenceError: If predictions, detection results, or
                state snapshots are inconsistent.
        """
        validated_target = _validate_target_column(target_column)
        validated_features = _validate_feature_columns(
            feature_columns,
            target_column=validated_target,
        )
        validated_split = _validate_split(
            split,
            target_column=validated_target,
            feature_columns=validated_features,
        )
        _validate_minimum_data(validated_split, policy=self._policy)
        _validate_leakage_report(
            leakage_report,
            feature_columns=validated_features,
            require_safe=self._policy.require_safe_leakage_report,
        )
        _validate_reserved_result_columns(validated_split)
        _validate_original_row_ids(validated_split)
        if self._policy.require_split_summary_match:
            _validate_split_summary_match(validated_split)

        validated_outcome = _validate_pipeline_outcome(pipeline_outcome)
        regression_model = validated_outcome.regression_model
        residual_detector = validated_outcome.residual_detector
        pipeline_report = validated_outcome.report

        if self._policy.require_pipeline_consistency:
            estimator_key = _validate_pipeline_consistency(
                outcome=validated_outcome,
                split=validated_split,
                target_column=validated_target,
                feature_columns=validated_features,
            )
        else:
            estimator_key = _validate_pipeline_basics(
                outcome=validated_outcome,
                feature_columns=validated_features,
                train_row_count=validated_split.train.height,
            )

        model_snapshot = _snapshot_regression_model(regression_model)
        detector_snapshot = _snapshot_detector(residual_detector)
        calibration = residual_detector.calibration
        if not isinstance(calibration, ResidualCalibration):
            raise ProcessIntelligenceError(
                "residual_detector.calibration must be ResidualCalibration "
                "before test scoring"
            )

        test = validated_split.test
        x_test = test.select(list(validated_features))
        y_test = test.get_column(validated_target)

        prediction_started = time.perf_counter()
        test_prediction = regression_model.predict(x_test)
        prediction_seconds = time.perf_counter() - prediction_started
        test_prediction = _validate_prediction_array(
            test_prediction,
            expected_length=test.height,
        )

        scoring_started = time.perf_counter()
        detection_result = residual_detector.detect(y_test, test_prediction)
        scoring_seconds = time.perf_counter() - scoring_started
        detection_result = _validate_residual_result(
            detection_result,
            expected_row_count=test.height,
            expected_prediction=test_prediction,
            expected_y_true=y_test,
            expected_threshold=float(calibration.threshold),
            residual_center=float(calibration.residual_center),
        )

        _assert_state_unchanged(
            regression_model=regression_model,
            residual_detector=residual_detector,
            model_snapshot=model_snapshot,
            detector_snapshot=detector_snapshot,
        )

        test_scored = _build_test_scored_frame(test, result=detection_result)
        _assert_scored_alignment(partition=test, scored=test_scored)

        validation_metrics = _copy_validation_metrics(pipeline_report)
        test_metrics = _build_test_metrics(detection_result)
        has_groups, score_separation = _compute_score_separation(
            detection_result,
            minimum_score_std=self._policy.minimum_score_std,
        )

        validation_fraction = validation_metrics["validation_anomaly_fraction"]
        validation_score_mean = validation_metrics["validation_score_mean"]
        validation_score_std = validation_metrics["validation_score_std"]
        anomaly_fraction_shift = abs(
            test_metrics["test_anomaly_fraction"] - validation_fraction
        )
        score_mean_shift = (
            test_metrics["test_score_mean"] - validation_score_mean
        )
        denom = max(validation_score_std, self._policy.minimum_score_std)
        standardized_score_mean_shift = abs(score_mean_shift) / denom
        score_std_ratio = test_metrics["test_score_std"] / denom
        for name, value in (
            ("anomaly_fraction_shift", anomaly_fraction_shift),
            ("score_mean_shift", score_mean_shift),
            ("standardized_score_mean_shift", standardized_score_mean_shift),
            ("score_std_ratio", score_std_ratio),
        ):
            if not math.isfinite(value):
                raise ProcessIntelligenceError(
                    f"{name} must be finite, got {value!r}"
                )

        quality_flags = _compute_quality_flags(
            test_metrics=test_metrics,
            anomaly_fraction_shift=anomaly_fraction_shift,
            standardized_score_mean_shift=standardized_score_mean_shift,
            score_std_ratio=score_std_ratio,
            has_normal_and_anomaly=has_groups,
            score_separation=score_separation,
            policy=self._policy,
        )
        warnings = _build_warnings(
            leakage_report=leakage_report,
            pipeline_report=pipeline_report,
            quality_flags=quality_flags,
            anomaly_fraction_shift=anomaly_fraction_shift,
            standardized_score_mean_shift=standardized_score_mean_shift,
            score_std_ratio=score_std_ratio,
            detection_result=detection_result,
            policy=self._policy,
        )

        metadata = regression_model.get_metadata()
        report = ResidualAnomalyFinalEvaluationReport(
            task=AnalysisTask.REGRESSION,
            model_name=str(metadata.model_name),
            estimator_key=str(estimator_key),
            target_column=validated_target,
            feature_columns=list(validated_features),
            detector_method=calibration.method,
            calibration_partition="validation",
            test_partition="test",
            train_row_count=int(validated_split.train.height),
            validation_row_count=int(validated_split.validation.height),
            test_row_count=int(test.height),
            regression_fit_row_count=int(validated_split.train.height),
            calibration_row_count=int(calibration.row_count),
            residual_center=float(calibration.residual_center),
            residual_scale=(
                None
                if calibration.residual_scale is None
                else float(calibration.residual_scale)
            ),
            threshold=float(calibration.threshold),
            validation_metrics=dict(validation_metrics),
            test_metrics=dict(test_metrics),
            test_has_normal_and_anomaly=has_groups,
            test_score_separation=score_separation,
            anomaly_fraction_shift=float(anomaly_fraction_shift),
            score_mean_shift=float(score_mean_shift),
            standardized_score_mean_shift=float(standardized_score_mean_shift),
            score_std_ratio=float(score_std_ratio),
            quality_flags=list(quality_flags),
            prediction_seconds=float(prediction_seconds),
            scoring_seconds=float(scoring_seconds),
            total_seconds=float(prediction_seconds + scoring_seconds),
            evaluated_at=datetime.now(UTC),
            warnings=warnings,
        )

        outcome = ResidualAnomalyFinalEvaluationOutcome(
            regression_model=regression_model,
            residual_detector=residual_detector,
            test_scored=test_scored,
            report=report,
        )
        _validate_outcome(
            outcome,
            model_snapshot=model_snapshot,
            detector_snapshot=detector_snapshot,
        )
        return outcome


def _validate_target_column(target_column: object) -> str:
    """Validate the regression target column name."""
    if not isinstance(target_column, str):
        raise TypeError(
            f"target_column must be str, got {type(target_column).__name__}"
        )
    if target_column == "" or target_column.strip() == "":
        raise DataValidationError(
            "target_column must be a non-empty, non-whitespace string"
        )
    if target_column == ORIGINAL_ROW_ID_COLUMN:
        raise DataValidationError(
            f"target_column must not be reserved column {ORIGINAL_ROW_ID_COLUMN!r}"
        )
    return target_column


def _validate_feature_columns(
    feature_columns: object,
    *,
    target_column: str,
) -> list[str]:
    """Validate feature names while preserving input order."""
    if isinstance(feature_columns, (str, bytes)):
        raise TypeError(
            "feature_columns must be a Sequence[str], not "
            f"{type(feature_columns).__name__}"
        )
    if not isinstance(feature_columns, Sequence):
        raise TypeError(
            "feature_columns must be a Sequence[str], "
            f"got {type(feature_columns).__name__}"
        )

    validated: list[str] = []
    seen: set[str] = set()
    for index, name in enumerate(feature_columns):
        if not isinstance(name, str):
            raise TypeError(
                f"feature_columns[{index}] must be str, got {type(name).__name__}"
            )
        if name == "" or name.strip() == "":
            raise DataValidationError(
                f"feature_columns[{index}] must be a non-empty, "
                "non-whitespace string"
            )
        if name == ORIGINAL_ROW_ID_COLUMN:
            raise DataValidationError(
                "feature_columns must not include reserved column "
                f"{ORIGINAL_ROW_ID_COLUMN!r}"
            )
        if name == target_column:
            raise DataValidationError(
                "feature_columns must not include target_column "
                f"{target_column!r}"
            )
        if name in _RESERVED_RESULT_COLUMNS:
            raise DataValidationError(
                "feature_columns must not include reserved residual result "
                f"column {name!r}"
            )
        if name in seen:
            raise DataValidationError(
                f"feature_columns must not contain duplicates, found {name!r}"
            )
        seen.add(name)
        validated.append(name)

    if not validated:
        raise DataValidationError(
            "feature_columns must contain at least one feature"
        )
    return validated


def _validate_split(
    split: object,
    *,
    target_column: str,
    feature_columns: Sequence[str],
) -> DatasetSplit:
    """Validate split type, partition frames, schema, and required columns."""
    if not isinstance(split, DatasetSplit):
        raise TypeError(f"split must be DatasetSplit, got {type(split).__name__}")

    partitions = {
        "train": split.train,
        "validation": split.validation,
        "test": split.test,
    }
    for name, frame in partitions.items():
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(
                f"split.{name} must be a polars.DataFrame, "
                f"got {type(frame).__name__}"
            )

    train_columns = list(split.train.columns)
    if list(split.validation.columns) != train_columns:
        raise DataValidationError(
            "train and validation partitions must share identical column "
            "names and order"
        )
    if list(split.test.columns) != train_columns:
        raise DataValidationError(
            "train and test partitions must share identical column names "
            "and order"
        )

    train_dtypes = list(split.train.dtypes)
    if list(split.validation.dtypes) != train_dtypes:
        raise DataValidationError(
            "train and validation partitions must share identical dtypes"
        )
    if list(split.test.dtypes) != train_dtypes:
        raise DataValidationError(
            "train and test partitions must share identical dtypes"
        )

    if ORIGINAL_ROW_ID_COLUMN not in train_columns:
        raise DataValidationError(
            f"all partitions require reserved column {ORIGINAL_ROW_ID_COLUMN!r}"
        )
    if target_column not in train_columns:
        raise DataValidationError(
            f"target_column {target_column!r} is missing from split partitions"
        )

    missing_features = [
        name for name in feature_columns if name not in train_columns
    ]
    if missing_features:
        raise DataValidationError(
            "feature_columns missing from split partitions: "
            f"{', '.join(missing_features)}"
        )
    return split


def _validate_minimum_data(
    split: DatasetSplit,
    *,
    policy: ResidualAnomalyFinalEvaluationPolicy,
) -> None:
    """Enforce non-empty train/validation and sufficient test rows."""
    if split.train.height < 1:
        raise InsufficientDataError(
            "train partition must contain at least one row for residual "
            "anomaly final evaluation"
        )
    if split.validation.height < 1:
        raise InsufficientDataError(
            "validation partition must contain at least one row for residual "
            "anomaly final evaluation"
        )
    if split.test.height < policy.minimum_test_rows:
        raise InsufficientDataError(
            "test partition must contain at least "
            f"{policy.minimum_test_rows} row(s) for residual anomaly final "
            f"evaluation, got {split.test.height}"
        )


def _validate_leakage_report(
    leakage_report: object,
    *,
    feature_columns: Sequence[str],
    require_safe: bool,
) -> None:
    """Validate leakage report type, feature alignment, and optional safety gate."""
    if not isinstance(leakage_report, LeakageReport):
        raise TypeError(
            "leakage_report must be LeakageReport, "
            f"got {type(leakage_report).__name__}"
        )
    checked = list(leakage_report.checked_feature_columns)
    expected = list(feature_columns)
    if checked != expected:
        raise DataValidationError(
            "leakage_report.checked_feature_columns must exactly match "
            f"feature_columns in order; expected {expected!r}, got {checked!r}"
        )
    if require_safe and leakage_report.blocker_count >= 1:
        blocker_types: list[str] = []
        seen: set[str] = set()
        for issue in leakage_report.issues:
            if issue.severity is not LeakageSeverity.BLOCKER:
                continue
            type_name = str(issue.issue_type)
            if type_name not in seen:
                seen.add(type_name)
                blocker_types.append(type_name)
        raise DataLeakageError(
            "Data leakage blockers detected during residual anomaly final "
            f"evaluation: count={leakage_report.blocker_count}, "
            f"types={', '.join(blocker_types)}"
        )


def _partition_row_ids(frame: pl.DataFrame) -> list[int]:
    """Return ``_original_row_id`` values as Python ints in frame order."""
    return [
        int(value)
        for value in frame.get_column(ORIGINAL_ROW_ID_COLUMN).to_list()
    ]


def _validate_original_row_id_column(
    frame: pl.DataFrame,
    *,
    partition_name: str,
) -> list[int]:
    """Validate one partition's ``_original_row_id`` column and return IDs."""
    if ORIGINAL_ROW_ID_COLUMN not in frame.columns:
        raise DataValidationError(
            f"{partition_name} partition is missing required column "
            f"{ORIGINAL_ROW_ID_COLUMN!r}"
        )

    dtype = frame.schema[ORIGINAL_ROW_ID_COLUMN]
    if dtype == pl.Boolean:
        raise DataValidationError(
            f"{partition_name} column {ORIGINAL_ROW_ID_COLUMN!r} must be an "
            "integer type, got Boolean"
        )
    if not dtype.is_integer():
        raise DataValidationError(
            f"{partition_name} column {ORIGINAL_ROW_ID_COLUMN!r} must be an "
            f"integer type, got {dtype}"
        )

    series = frame.get_column(ORIGINAL_ROW_ID_COLUMN)
    if series.null_count() > 0:
        raise DataValidationError(
            f"{partition_name} column {ORIGINAL_ROW_ID_COLUMN!r} must not "
            "contain null values"
        )
    if series.n_unique() != series.len():
        raise DataValidationError(
            f"{partition_name} column {ORIGINAL_ROW_ID_COLUMN!r} must "
            "contain unique values"
        )
    if series.len() > 0 and bool((series < 0).any()):
        raise DataValidationError(
            f"{partition_name} column {ORIGINAL_ROW_ID_COLUMN!r} must not "
            "contain negative values"
        )
    return _partition_row_ids(frame)


def _validate_original_row_ids(split: DatasetSplit) -> None:
    """Validate train/validation/test IDs and cross-partition uniqueness."""
    train_ids = _validate_original_row_id_column(
        split.train, partition_name="train"
    )
    validation_ids = _validate_original_row_id_column(
        split.validation, partition_name="validation"
    )
    test_ids = _validate_original_row_id_column(
        split.test, partition_name="test"
    )

    if set(train_ids) & set(validation_ids):
        raise DataValidationError(
            "Overlapping `_original_row_id` values detected between "
            "train and validation partitions"
        )
    if set(train_ids) & set(test_ids):
        raise DataValidationError(
            "Overlapping `_original_row_id` values detected between "
            "train and test partitions"
        )
    if set(validation_ids) & set(test_ids):
        raise DataValidationError(
            "Overlapping `_original_row_id` values detected between "
            "validation and test partitions"
        )


def _validate_split_summary_match(split: DatasetSplit) -> None:
    """Require partition ID order to match ``SplitSummary`` lists."""
    comparisons = (
        ("train", _partition_row_ids(split.train), split.summary.train_original_row_ids),
        (
            "validation",
            _partition_row_ids(split.validation),
            split.summary.validation_original_row_ids,
        ),
        ("test", _partition_row_ids(split.test), split.summary.test_original_row_ids),
    )
    mismatched: list[str] = []
    for name, frame_ids, summary_ids in comparisons:
        if list(frame_ids) != list(summary_ids):
            mismatched.append(name)
    if mismatched:
        raise DataValidationError(
            "SplitSummary `_original_row_id` sequences do not match partition "
            f"frames for: {', '.join(mismatched)}"
        )


def _validate_reserved_result_columns(split: DatasetSplit) -> None:
    """Reject partitions that already contain residual result columns."""
    present: list[str] = []
    seen: set[str] = set()
    for frame in (split.train, split.validation, split.test):
        for name in _RESERVED_RESULT_COLUMNS:
            if name in frame.columns and name not in seen:
                seen.add(name)
                present.append(name)
    if present:
        raise DataValidationError(
            "input partitions already contain reserved residual result "
            f"columns: {', '.join(present)}"
        )


def _validate_pipeline_outcome(
    pipeline_outcome: object,
) -> ResidualAnomalyPipelineOutcome:
    """Validate pipeline outcome type and nested object types."""
    if not isinstance(pipeline_outcome, ResidualAnomalyPipelineOutcome):
        raise TypeError(
            "pipeline_outcome must be ResidualAnomalyPipelineOutcome, "
            f"got {type(pipeline_outcome).__name__}"
        )
    if not isinstance(pipeline_outcome.regression_model, BaseAnalysisModel):
        raise TypeError(
            "pipeline_outcome.regression_model must be BaseAnalysisModel, "
            f"got {type(pipeline_outcome.regression_model).__name__}"
        )
    if not isinstance(pipeline_outcome.residual_detector, ResidualAnomalyDetector):
        raise TypeError(
            "pipeline_outcome.residual_detector must be ResidualAnomalyDetector, "
            f"got {type(pipeline_outcome.residual_detector).__name__}"
        )
    if not isinstance(pipeline_outcome.report, ResidualAnomalyPipelineReport):
        raise TypeError(
            "pipeline_outcome.report must be ResidualAnomalyPipelineReport, "
            f"got {type(pipeline_outcome.report).__name__}"
        )
    for name, frame in (
        ("train_scored", pipeline_outcome.train_scored),
        ("validation_scored", pipeline_outcome.validation_scored),
        ("combined_scored", pipeline_outcome.combined_scored),
    ):
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(
                f"pipeline_outcome.{name} must be a polars.DataFrame, "
                f"got {type(frame).__name__}"
            )
    return pipeline_outcome


def _resolve_estimator_key(
    model: BaseAnalysisModel,
    metadata: ModelMetadata,
) -> str | None:
    """Resolve estimator key from metadata or the model's bound ModelSpec."""
    estimator_key = getattr(metadata, "estimator_key", None)
    if isinstance(estimator_key, str) and estimator_key.strip() != "":
        return estimator_key
    model_spec = getattr(model, "_spec", None)
    if isinstance(model_spec, ModelSpec):
        return model_spec.estimator_key
    return estimator_key if isinstance(estimator_key, str) else None


def _resolve_fit_row_count(
    model: BaseAnalysisModel,
    metadata: ModelMetadata,
) -> int | None:
    """Resolve fit row count from the model property or metadata."""
    fit_row_count = getattr(model, "fit_row_count", None)
    if isinstance(fit_row_count, int) and not isinstance(fit_row_count, bool):
        return fit_row_count
    metadata_fit = getattr(metadata, "fit_row_count", None)
    if isinstance(metadata_fit, int) and not isinstance(metadata_fit, bool):
        return metadata_fit
    return None


def _validate_regression_model_basics(
    model: BaseAnalysisModel,
    *,
    feature_columns: Sequence[str],
    train_row_count: int,
) -> str:
    """Validate fitted regression model basics and return estimator key."""
    if hasattr(model, "is_fitted") and not bool(model.is_fitted):
        raise ProcessIntelligenceError(
            "regression model must be in a fitted state"
        )
    predict_method = getattr(model, "predict", None)
    if not callable(predict_method):
        raise ProcessIntelligenceError(
            "regression model must provide a callable predict method"
        )

    metadata = model.get_metadata()
    if not isinstance(metadata, ModelMetadata):
        raise ProcessIntelligenceError(
            "regression_model.get_metadata must return ModelMetadata, "
            f"got {type(metadata).__name__}"
        )
    if metadata.task is not AnalysisTask.REGRESSION:
        raise ProcessIntelligenceError(
            "regression model metadata task must be AnalysisTask.REGRESSION, "
            f"got {metadata.task!r}"
        )
    if list(metadata.features) != list(feature_columns):
        raise ProcessIntelligenceError(
            "regression model metadata features must match feature_columns: "
            f"expected {list(feature_columns)}, got {list(metadata.features)}"
        )

    estimator_key = _resolve_estimator_key(model, metadata)
    if not isinstance(estimator_key, str) or estimator_key.strip() == "":
        raise ProcessIntelligenceError(
            "regression model estimator_key must be a non-empty string"
        )

    fit_row_count = _resolve_fit_row_count(model, metadata)
    if fit_row_count != train_row_count:
        raise ProcessIntelligenceError(
            "regression model fit_row_count must equal train height "
            f"({train_row_count}), got {fit_row_count}"
        )

    feature_names = getattr(model, "feature_names", None)
    if feature_names is not None and list(feature_names) != list(feature_columns):
        raise ProcessIntelligenceError(
            "regression model feature_names must match feature_columns: "
            f"expected {list(feature_columns)}, got {list(feature_names)}"
        )
    return estimator_key


def _validate_detector_basics(detector: ResidualAnomalyDetector) -> ResidualCalibration:
    """Validate that the residual detector is fitted with a calibration."""
    if not detector.is_fitted:
        raise ProcessIntelligenceError(
            "residual detector must be in a fitted state"
        )
    calibration = detector.calibration
    if not isinstance(calibration, ResidualCalibration):
        raise ProcessIntelligenceError(
            "residual detector.calibration must be ResidualCalibration, "
            f"got {type(calibration).__name__}"
        )
    if (
        calibration.fitted_at.tzinfo is None
        or calibration.fitted_at.utcoffset() is None
    ):
        raise ProcessIntelligenceError(
            "detector calibration.fitted_at must be timezone-aware"
        )
    return calibration


def _validate_pipeline_basics(
    outcome: ResidualAnomalyPipelineOutcome,
    *,
    feature_columns: Sequence[str],
    train_row_count: int,
) -> str:
    """Validate minimal pipeline readiness when consistency is disabled."""
    report = outcome.report
    if report.task is not AnalysisTask.REGRESSION:
        raise DataValidationError(
            "pipeline report task must be AnalysisTask.REGRESSION, "
            f"got {report.task!r}"
        )
    estimator_key = _validate_regression_model_basics(
        outcome.regression_model,
        feature_columns=feature_columns,
        train_row_count=train_row_count,
    )
    _validate_detector_basics(outcome.residual_detector)
    return estimator_key


def _find_validation_summary(
    report: ResidualAnomalyPipelineReport,
) -> ResidualPartitionSummary:
    """Return the unique scored validation partition summary."""
    matches = [
        summary
        for summary in report.partition_summaries
        if summary.partition == "validation"
    ]
    if len(matches) == 0:
        raise DataValidationError(
            "pipeline report is missing a validation ResidualPartitionSummary"
        )
    if len(matches) > 1:
        raise DataValidationError(
            "pipeline report contains multiple validation "
            "ResidualPartitionSummary entries"
        )
    return matches[0]


def _validate_pipeline_consistency(
    *,
    outcome: ResidualAnomalyPipelineOutcome,
    split: DatasetSplit,
    target_column: str,
    feature_columns: Sequence[str],
) -> str:
    """Validate pipeline report, model, detector, and validation summary."""
    report = outcome.report
    mismatches: list[str] = []

    if report.task is not AnalysisTask.REGRESSION:
        mismatches.append(f"task (report={report.task!r})")
    if report.target_column != target_column:
        mismatches.append(
            "target_column "
            f"(report={report.target_column!r}, expected={target_column!r})"
        )
    if list(report.feature_columns) != list(feature_columns):
        mismatches.append(
            "feature_columns "
            f"(report={list(report.feature_columns)!r}, "
            f"expected={list(feature_columns)!r})"
        )
    if report.train_row_count != split.train.height:
        mismatches.append(
            "train_row_count "
            f"(report={report.train_row_count}, expected={split.train.height})"
        )
    if report.validation_row_count != split.validation.height:
        mismatches.append(
            "validation_row_count "
            f"(report={report.validation_row_count}, "
            f"expected={split.validation.height})"
        )
    if report.test_row_count != split.test.height:
        mismatches.append(
            "test_row_count "
            f"(report={report.test_row_count}, expected={split.test.height})"
        )
    if report.calibration_partition != "validation":
        mismatches.append(
            "calibration_partition "
            f"(report={report.calibration_partition!r})"
        )
    if report.calibration_row_count != split.validation.height:
        mismatches.append(
            "calibration_row_count "
            f"(report={report.calibration_row_count}, "
            f"expected={split.validation.height})"
        )
    if report.model_name == "" or report.model_name.strip() == "":
        mismatches.append("model_name is missing or blank")
    if report.estimator_key == "" or report.estimator_key.strip() == "":
        mismatches.append("estimator_key is missing or blank")
    if mismatches:
        raise DataValidationError(
            "pipeline_outcome is inconsistent with residual anomaly final "
            "evaluation inputs: " + "; ".join(mismatches)
        )

    estimator_key = _validate_regression_model_basics(
        outcome.regression_model,
        feature_columns=feature_columns,
        train_row_count=split.train.height,
    )
    metadata = outcome.regression_model.get_metadata()
    if metadata.model_name != report.model_name:
        raise ProcessIntelligenceError(
            "regression model metadata name does not match pipeline report: "
            f"metadata={metadata.model_name!r}, report={report.model_name!r}"
        )
    if estimator_key != report.estimator_key:
        raise ProcessIntelligenceError(
            "regression model estimator_key does not match pipeline report: "
            f"model={estimator_key!r}, report={report.estimator_key!r}"
        )

    calibration = _validate_detector_basics(outcome.residual_detector)
    detector = outcome.residual_detector
    if calibration.method is not report.detector_method:
        raise ProcessIntelligenceError(
            "detector calibration.method must match pipeline report: "
            f"calibration={calibration.method!r}, "
            f"report={report.detector_method!r}"
        )
    if calibration.row_count != split.validation.height:
        raise ProcessIntelligenceError(
            "detector calibration.row_count must equal validation height "
            f"({split.validation.height}), got {calibration.row_count}"
        )
    if not _values_close(
        float(calibration.residual_center),
        float(report.residual_center),
    ):
        raise ProcessIntelligenceError(
            "detector residual_center must match pipeline report: "
            f"calibration={calibration.residual_center}, "
            f"report={report.residual_center}"
        )
    if (calibration.residual_scale is None) != (report.residual_scale is None):
        raise ProcessIntelligenceError(
            "detector residual_scale nullability must match pipeline report"
        )
    if (
        calibration.residual_scale is not None
        and report.residual_scale is not None
        and not _values_close(
            float(calibration.residual_scale),
            float(report.residual_scale),
        )
    ):
        raise ProcessIntelligenceError(
            "detector residual_scale must match pipeline report: "
            f"calibration={calibration.residual_scale}, "
            f"report={report.residual_scale}"
        )
    if not _values_close(float(calibration.threshold), float(report.threshold)):
        raise ProcessIntelligenceError(
            "detector calibration.threshold must match pipeline report: "
            f"calibration={calibration.threshold}, report={report.threshold}"
        )
    if detector.threshold is None or not _values_close(
        float(detector.threshold),
        float(report.threshold),
    ):
        raise ProcessIntelligenceError(
            "detector.threshold must match pipeline report threshold: "
            f"detector={detector.threshold}, report={report.threshold}"
        )

    validation_summary = _find_validation_summary(report)
    if not validation_summary.scored:
        raise DataValidationError(
            "validation ResidualPartitionSummary must have scored=True"
        )
    if validation_summary.prediction_count != split.validation.height:
        raise DataValidationError(
            "validation summary prediction_count must equal validation "
            f"height ({split.validation.height}), "
            f"got {validation_summary.prediction_count}"
        )
    for name, summary in (
        ("anomaly_fraction", validation_summary.anomaly_fraction),
        ("residual_mean", validation_summary.residual_mean),
        ("residual_std", validation_summary.residual_std),
        ("score_min", validation_summary.score_min),
        ("score_max", validation_summary.score_max),
        ("score_mean", validation_summary.score_mean),
        ("score_std", validation_summary.score_std),
    ):
        if summary is None or not _is_finite_number(summary):
            raise DataValidationError(
                "validation summary requires finite "
                f"{name}, got {summary!r}"
            )
    return estimator_key


def _snapshot_regression_model(model: BaseAnalysisModel) -> dict[str, Any]:
    """Capture comparable regression model state before test prediction."""
    metadata = model.get_metadata()
    feature_names = getattr(model, "feature_names", None)
    fit_row_count = _resolve_fit_row_count(model, metadata)
    return {
        "is_fitted": bool(getattr(model, "is_fitted", True)),
        "model_name": str(metadata.model_name),
        "task": metadata.task,
        "features": list(metadata.features),
        "feature_names": (
            None if feature_names is None else list(feature_names)
        ),
        "fit_row_count": fit_row_count,
        "estimator_key": _resolve_estimator_key(model, metadata),
        "metadata_dump": metadata.model_dump(),
    }


def _snapshot_detector(detector: ResidualAnomalyDetector) -> dict[str, Any]:
    """Capture comparable detector state before test scoring."""
    calibration = detector.calibration
    return {
        "is_fitted": bool(detector.is_fitted),
        "threshold": detector.threshold,
        "fitted_at": detector.fitted_at,
        "calibration": (
            None if calibration is None else calibration.model_dump()
        ),
        "metadata": detector.get_metadata(),
    }


def _assert_state_unchanged(
    *,
    regression_model: BaseAnalysisModel,
    residual_detector: ResidualAnomalyDetector,
    model_snapshot: Mapping[str, Any],
    detector_snapshot: Mapping[str, Any],
) -> None:
    """Require model and detector state to match pre-scoring snapshots."""
    current_model = _snapshot_regression_model(regression_model)
    if current_model != dict(model_snapshot):
        raise ProcessIntelligenceError(
            "regression model state changed during test prediction/scoring"
        )
    current_detector = _snapshot_detector(residual_detector)
    if current_detector != dict(detector_snapshot):
        raise ProcessIntelligenceError(
            "residual detector state changed during test prediction/scoring"
        )


def _validate_prediction_array(
    prediction: object,
    *,
    expected_length: int,
) -> np.ndarray:
    """Validate a regression prediction array without mutating the input."""
    if not isinstance(prediction, np.ndarray):
        raise ProcessIntelligenceError(
            "test prediction must be a numpy.ndarray, "
            f"got {type(prediction).__name__}"
        )
    if prediction.ndim != 1:
        raise ProcessIntelligenceError(
            f"test prediction must be 1-dimensional, got shape {prediction.shape}"
        )
    if prediction.shape[0] != expected_length:
        raise ProcessIntelligenceError(
            f"test prediction length ({prediction.shape[0]}) must equal "
            f"test height ({expected_length})"
        )
    if prediction.dtype == np.bool_ or prediction.dtype.kind == "b":
        raise ProcessIntelligenceError(
            "test prediction must be numeric; boolean values are not allowed"
        )
    if prediction.dtype.kind not in {"f", "i", "u"}:
        raise ProcessIntelligenceError(
            f"test prediction must be numeric, got dtype {prediction.dtype}"
        )
    if not np.isfinite(prediction.astype(np.float64, copy=False)).all():
        raise ProcessIntelligenceError(
            "test prediction must contain only finite values"
        )
    return prediction


def _series_to_float_array(values: pl.Series) -> np.ndarray:
    """Convert a Polars Series to an independent float64 NumPy array."""
    return np.asarray(values.to_numpy(), dtype=np.float64)


def _validate_residual_result(
    result: object,
    *,
    expected_row_count: int,
    expected_prediction: np.ndarray,
    expected_y_true: pl.Series,
    expected_threshold: float,
    residual_center: float,
) -> ResidualAnomalyResult:
    """Validate ``ResidualAnomalyResult`` consistency against test inputs."""
    if not isinstance(result, ResidualAnomalyResult):
        raise ProcessIntelligenceError(
            "detect must return ResidualAnomalyResult, "
            f"got {type(result).__name__}"
        )
    if result.row_count != expected_row_count:
        raise ProcessIntelligenceError(
            "ResidualAnomalyResult.row_count must equal test height "
            f"({expected_row_count}), got {result.row_count}"
        )
    for name, collection in (
        ("predictions", result.predictions),
        ("residuals", result.residuals),
        ("absolute_centered_residuals", result.absolute_centered_residuals),
        ("scores", result.scores),
        ("is_anomaly", result.is_anomaly),
        ("raw_predictions", result.raw_predictions),
    ):
        if len(collection) != expected_row_count:
            raise ProcessIntelligenceError(
                f"ResidualAnomalyResult.{name} length must equal test height "
                f"({expected_row_count}), got {len(collection)}"
            )

    y_true = _series_to_float_array(expected_y_true)
    expected_residuals = y_true - expected_prediction.astype(np.float64, copy=False)
    expected_centered = np.abs(expected_residuals - float(residual_center))

    for index, value in enumerate(result.predictions):
        if not _is_finite_number(value):
            raise ProcessIntelligenceError(
                f"predictions[{index}] must be finite, got {value!r}"
            )
        if not _values_close(float(value), float(expected_prediction[index])):
            raise ProcessIntelligenceError(
                "result.predictions must match regression test predictions"
            )
    for index, value in enumerate(result.residuals):
        if not _is_finite_number(value):
            raise ProcessIntelligenceError(
                f"residuals[{index}] must be finite, got {value!r}"
            )
        if not _values_close(float(value), float(expected_residuals[index])):
            raise ProcessIntelligenceError(
                "result.residuals must equal y_true - prediction"
            )
    for index, value in enumerate(result.absolute_centered_residuals):
        if not _is_finite_number(value) or float(value) < 0.0:
            raise ProcessIntelligenceError(
                f"absolute_centered_residuals[{index}] must be a finite "
                f"float >= 0, got {value!r}"
            )
        if not _values_close(float(value), float(expected_centered[index])):
            raise ProcessIntelligenceError(
                "absolute_centered_residuals must use validation residual center"
            )
    for index, value in enumerate(result.scores):
        if not _is_finite_number(value) or float(value) < 0.0:
            raise ProcessIntelligenceError(
                f"scores[{index}] must be a finite float >= 0, got {value!r}"
            )
    for index, raw in enumerate(result.raw_predictions):
        if raw not in (-1, 1):
            raise ProcessIntelligenceError(
                f"raw_predictions[{index}] must be -1 or 1, got {raw!r}"
            )
        expected_flag = raw == -1
        if bool(result.is_anomaly[index]) is not expected_flag:
            raise ProcessIntelligenceError(
                "is_anomaly must match raw_predictions == -1 "
                f"at index {index}"
            )

    true_count = sum(1 for flag in result.is_anomaly if flag)
    if true_count != result.anomaly_count:
        raise ProcessIntelligenceError(
            "anomaly_count must equal True count in is_anomaly "
            f"({true_count}), got {result.anomaly_count}"
        )
    expected_fraction = result.anomaly_count / expected_row_count
    if not _fractions_match(result.anomaly_fraction, expected_fraction):
        raise ProcessIntelligenceError(
            "anomaly_fraction must equal anomaly_count / row_count "
            f"({expected_fraction}), got {result.anomaly_fraction}"
        )

    for name, summary in (
        ("residual_mean", result.residual_mean),
        ("residual_std", result.residual_std),
        ("score_min", result.score_min),
        ("score_max", result.score_max),
        ("score_mean", result.score_mean),
        ("score_std", result.score_std),
    ):
        if summary is None or not _is_finite_number(summary):
            raise ProcessIntelligenceError(
                f"{name} must be a finite float, got {summary!r}"
            )
    assert result.residual_std is not None
    assert result.score_std is not None
    assert result.score_min is not None
    assert result.score_max is not None
    assert result.score_mean is not None
    if float(result.residual_std) < 0.0:
        raise ProcessIntelligenceError(
            f"residual_std must be >= 0, got {result.residual_std}"
        )
    if float(result.score_std) < 0.0:
        raise ProcessIntelligenceError(
            f"score_std must be >= 0, got {result.score_std}"
        )
    if not (result.score_min <= result.score_mean <= result.score_max):
        raise ProcessIntelligenceError(
            "score summaries must satisfy score_min <= score_mean <= score_max"
        )
    if not _values_close(float(result.threshold), float(expected_threshold)):
        raise ProcessIntelligenceError(
            "result.threshold must equal detector threshold "
            f"({expected_threshold}), got {result.threshold}"
        )
    return result


def _build_test_scored_frame(
    partition: pl.DataFrame,
    *,
    result: ResidualAnomalyResult,
) -> pl.DataFrame:
    """Append residual anomaly result columns to a test partition copy."""
    height = partition.height
    predictions = list(result.predictions)
    residuals = list(result.residuals)
    centered = list(result.absolute_centered_residuals)
    scores = list(result.scores)
    flags = list(result.is_anomaly)
    raws = list(result.raw_predictions)
    if (
        len(predictions) != height
        or len(residuals) != height
        or len(centered) != height
        or len(scores) != height
        or len(flags) != height
        or len(raws) != height
    ):
        raise ProcessIntelligenceError(
            "residual anomaly result lengths must match scored frame "
            f"height ({height})"
        )

    return partition.with_columns(
        [
            pl.Series(
                _REGRESSION_PREDICTION_COLUMN,
                predictions,
                dtype=pl.Float64,
            ),
            pl.Series(
                _REGRESSION_RESIDUAL_COLUMN,
                residuals,
                dtype=pl.Float64,
            ),
            pl.Series(
                _ABSOLUTE_CENTERED_RESIDUAL_COLUMN,
                centered,
                dtype=pl.Float64,
            ),
            pl.Series(
                _RESIDUAL_ANOMALY_SCORE_COLUMN,
                scores,
                dtype=pl.Float64,
            ),
            pl.Series(
                _IS_RESIDUAL_ANOMALY_COLUMN,
                flags,
                dtype=pl.Boolean,
            ),
            pl.Series(
                _RESIDUAL_ANOMALY_RAW_PREDICTION_COLUMN,
                raws,
                dtype=pl.Int64,
            ),
            pl.Series(
                _DATA_PARTITION_COLUMN,
                ["test"] * height,
                dtype=pl.String,
            ),
        ]
    )


def _assert_scored_alignment(
    *,
    partition: pl.DataFrame,
    scored: pl.DataFrame,
) -> None:
    """Require 1:1 row alignment between input test partition and scored frame."""
    if scored.height != partition.height:
        raise ProcessIntelligenceError(
            f"test scored height ({scored.height}) must equal "
            f"input height ({partition.height})"
        )
    input_ids = _partition_row_ids(partition)
    scored_ids = _partition_row_ids(scored)
    if scored_ids != input_ids:
        raise ProcessIntelligenceError(
            "test scored `_original_row_id` order must match the input partition"
        )
    if len(scored_ids) != len(set(scored_ids)):
        raise ProcessIntelligenceError(
            "test scored `_original_row_id` values must be unique"
        )
    for column in partition.columns:
        if scored.get_column(column).dtype != partition.get_column(column).dtype:
            raise ProcessIntelligenceError(
                f"test scored column {column!r} dtype changed"
            )
        if scored.get_column(column).to_list() != partition.get_column(column).to_list():
            raise ProcessIntelligenceError(
                f"test scored column {column!r} values changed"
            )
    expected_columns = list(partition.columns) + list(_RESERVED_RESULT_COLUMNS)
    if list(scored.columns) != expected_columns:
        raise ProcessIntelligenceError(
            "test scored columns must keep original columns then residual "
            f"result columns; expected {expected_columns}, "
            f"got {list(scored.columns)}"
        )


def _copy_validation_metrics(
    report: ResidualAnomalyPipelineReport,
) -> dict[str, float]:
    """Copy validation residual metrics from the pipeline report."""
    validation_summary = _find_validation_summary(report)
    if not validation_summary.scored:
        raise ProcessIntelligenceError(
            "validation summary must be scored to copy validation metrics"
        )
    for name, value in (
        ("anomaly_fraction", validation_summary.anomaly_fraction),
        ("residual_mean", validation_summary.residual_mean),
        ("residual_std", validation_summary.residual_std),
        ("score_min", validation_summary.score_min),
        ("score_max", validation_summary.score_max),
        ("score_mean", validation_summary.score_mean),
        ("score_std", validation_summary.score_std),
    ):
        if value is None or not _is_finite_number(value):
            raise ProcessIntelligenceError(
                f"validation summary {name} must be finite, got {value!r}"
            )
    assert validation_summary.residual_mean is not None
    assert validation_summary.residual_std is not None
    assert validation_summary.score_min is not None
    assert validation_summary.score_max is not None
    assert validation_summary.score_mean is not None
    assert validation_summary.score_std is not None
    score_range = float(validation_summary.score_max) - float(
        validation_summary.score_min
    )
    metrics = {
        "validation_anomaly_fraction": float(validation_summary.anomaly_fraction),
        "validation_residual_mean": float(validation_summary.residual_mean),
        "validation_residual_std": float(validation_summary.residual_std),
        "validation_score_min": float(validation_summary.score_min),
        "validation_score_max": float(validation_summary.score_max),
        "validation_score_mean": float(validation_summary.score_mean),
        "validation_score_std": float(validation_summary.score_std),
        "validation_score_range": float(score_range),
    }
    if list(metrics.keys()) != list(_VALIDATION_METRIC_KEYS):
        raise ProcessIntelligenceError(
            "validation_metrics key order must match the required contract"
        )
    return metrics


def _build_test_metrics(result: ResidualAnomalyResult) -> dict[str, float]:
    """Build ordered test residual metrics from a detection result."""
    assert result.residual_mean is not None
    assert result.residual_std is not None
    assert result.score_min is not None
    assert result.score_max is not None
    assert result.score_mean is not None
    assert result.score_std is not None
    score_range = float(result.score_max) - float(result.score_min)
    metrics = {
        "test_anomaly_fraction": float(result.anomaly_fraction),
        "test_residual_mean": float(result.residual_mean),
        "test_residual_std": float(result.residual_std),
        "test_score_min": float(result.score_min),
        "test_score_max": float(result.score_max),
        "test_score_mean": float(result.score_mean),
        "test_score_std": float(result.score_std),
        "test_score_range": float(score_range),
    }
    for key, value in metrics.items():
        if not math.isfinite(value):
            raise ProcessIntelligenceError(
                f"{key} must be finite, got {value!r}"
            )
    if list(metrics.keys()) != list(_TEST_METRIC_KEYS):
        raise ProcessIntelligenceError(
            "test_metrics key order must match the required contract"
        )
    return metrics


def _compute_score_separation(
    result: ResidualAnomalyResult,
    *,
    minimum_score_std: float,
) -> tuple[bool, float | None]:
    """Compute standardized anomaly/normal score separation when both exist."""
    anomaly_scores = [
        float(score)
        for score, raw in zip(result.scores, result.raw_predictions, strict=True)
        if raw == -1
    ]
    normal_scores = [
        float(score)
        for score, raw in zip(result.scores, result.raw_predictions, strict=True)
        if raw == 1
    ]
    if not anomaly_scores or not normal_scores:
        return False, None
    assert result.score_std is not None
    denom = max(float(result.score_std), float(minimum_score_std))
    separation = (
        float(np.mean(np.asarray(anomaly_scores, dtype=np.float64)))
        - float(np.mean(np.asarray(normal_scores, dtype=np.float64)))
    ) / denom
    if not math.isfinite(separation):
        raise ProcessIntelligenceError(
            f"test_score_separation must be finite, got {separation!r}"
        )
    return True, float(separation)


def _compute_quality_flags(
    *,
    test_metrics: Mapping[str, float],
    anomaly_fraction_shift: float,
    standardized_score_mean_shift: float,
    score_std_ratio: float,
    has_normal_and_anomaly: bool,
    score_separation: float | None,
    policy: ResidualAnomalyFinalEvaluationPolicy,
) -> list[str]:
    """Build deterministic quality flags without mutating model state."""
    flags: list[str] = []
    fraction = test_metrics["test_anomaly_fraction"]
    score_std = test_metrics["test_score_std"]

    if fraction < policy.minimum_anomaly_fraction:
        flags.append(_FLAG_FRACTION_BELOW)
    if fraction > policy.maximum_anomaly_fraction:
        flags.append(_FLAG_FRACTION_ABOVE)
    if anomaly_fraction_shift > policy.maximum_anomaly_fraction_shift:
        flags.append(_FLAG_FRACTION_SHIFT)
    if score_std < policy.minimum_score_std:
        flags.append(_FLAG_SCORE_NEARLY_CONSTANT)
    if not has_normal_and_anomaly:
        flags.append(_FLAG_DEGENERATE)
    if score_separation is not None and score_separation < 0.0:
        flags.append(_FLAG_NEGATIVE_SEPARATION)
    if (
        standardized_score_mean_shift
        > policy.maximum_standardized_score_mean_shift
    ):
        flags.append(_FLAG_SCORE_MEAN_SHIFT)
    if score_std_ratio < policy.minimum_score_std_ratio:
        flags.append(_FLAG_SCORE_STD_CONTRACTED)
    if score_std_ratio > policy.maximum_score_std_ratio:
        flags.append(_FLAG_SCORE_STD_EXPANDED)
    return _dedupe_preserve_order(flags)


def _has_distribution_shift(
    *,
    anomaly_fraction_shift: float,
    standardized_score_mean_shift: float,
    score_std_ratio: float,
    policy: ResidualAnomalyFinalEvaluationPolicy,
) -> bool:
    """Return True when any configured distribution-shift limit is exceeded."""
    if anomaly_fraction_shift > policy.maximum_anomaly_fraction_shift:
        return True
    if (
        standardized_score_mean_shift
        > policy.maximum_standardized_score_mean_shift
    ):
        return True
    if score_std_ratio < policy.minimum_score_std_ratio:
        return True
    if score_std_ratio > policy.maximum_score_std_ratio:
        return True
    return False


def _build_warnings(
    *,
    leakage_report: LeakageReport,
    pipeline_report: ResidualAnomalyPipelineReport,
    quality_flags: Sequence[str],
    anomaly_fraction_shift: float,
    standardized_score_mean_shift: float,
    score_std_ratio: float,
    detection_result: ResidualAnomalyResult,
    policy: ResidualAnomalyFinalEvaluationPolicy,
) -> list[str]:
    """Build deterministic, de-duplicated final-evaluation warnings."""
    warnings: list[str] = []

    if any(
        issue.severity is LeakageSeverity.WARNING
        for issue in leakage_report.issues
    ):
        warnings.append(_WARNING_LEAKAGE)

    for warning in pipeline_report.warnings:
        if warning:
            warnings.append(warning)

    if quality_flags:
        warnings.append(_WARNING_QUALITY_FLAGS)

    if policy.warn_on_distribution_shift and _has_distribution_shift(
        anomaly_fraction_shift=anomaly_fraction_shift,
        standardized_score_mean_shift=standardized_score_mean_shift,
        score_std_ratio=score_std_ratio,
        policy=policy,
    ):
        warnings.append(_WARNING_DISTRIBUTION_SHIFT)

    for warning in detection_result.warnings:
        if warning:
            warnings.append(warning)

    if detection_result.anomaly_count == 0:
        warnings.append(_WARNING_NO_ANOMALIES)
    if (
        detection_result.row_count > 0
        and detection_result.anomaly_count == detection_result.row_count
    ):
        warnings.append(_WARNING_ALL_ANOMALIES)

    return _dedupe_preserve_order(warnings)


def _validate_outcome(
    outcome: ResidualAnomalyFinalEvaluationOutcome,
    *,
    model_snapshot: Mapping[str, Any],
    detector_snapshot: Mapping[str, Any],
) -> None:
    """Validate final outcome consistency before returning to the caller."""
    if not isinstance(outcome.regression_model, BaseAnalysisModel):
        raise ProcessIntelligenceError(
            "outcome.regression_model must be BaseAnalysisModel, "
            f"got {type(outcome.regression_model).__name__}"
        )
    if hasattr(outcome.regression_model, "is_fitted") and not bool(
        outcome.regression_model.is_fitted
    ):
        raise ProcessIntelligenceError(
            "outcome.regression_model must be in a fitted state"
        )
    if not isinstance(outcome.residual_detector, ResidualAnomalyDetector):
        raise ProcessIntelligenceError(
            "outcome.residual_detector must be ResidualAnomalyDetector, "
            f"got {type(outcome.residual_detector).__name__}"
        )
    if not outcome.residual_detector.is_fitted:
        raise ProcessIntelligenceError(
            "outcome.residual_detector must be in a fitted state"
        )
    if not isinstance(outcome.test_scored, pl.DataFrame):
        raise ProcessIntelligenceError(
            "outcome.test_scored must be a polars.DataFrame, "
            f"got {type(outcome.test_scored).__name__}"
        )
    report = outcome.report
    if not isinstance(report, ResidualAnomalyFinalEvaluationReport):
        raise ProcessIntelligenceError(
            "outcome.report must be ResidualAnomalyFinalEvaluationReport, "
            f"got {type(report).__name__}"
        )

    if outcome.test_scored.height != report.test_row_count:
        raise ProcessIntelligenceError(
            "test_scored.height must equal report.test_row_count "
            f"({report.test_row_count}), got {outcome.test_scored.height}"
        )

    anomaly_true_count = int(
        outcome.test_scored.get_column(_IS_RESIDUAL_ANOMALY_COLUMN).sum()
    )
    expected_fraction = (
        float(anomaly_true_count) / float(outcome.test_scored.height)
        if outcome.test_scored.height > 0
        else 0.0
    )
    if not _fractions_match(
        report.test_metrics["test_anomaly_fraction"],
        expected_fraction,
    ):
        raise ProcessIntelligenceError(
            "test_scored anomaly True count must match "
            "report.test_metrics['test_anomaly_fraction']"
        )

    ids = outcome.test_scored.get_column(ORIGINAL_ROW_ID_COLUMN)
    if ids.n_unique() != ids.len():
        raise ProcessIntelligenceError(
            "test_scored `_original_row_id` values must be unique"
        )

    metadata = outcome.regression_model.get_metadata()
    if metadata.model_name != report.model_name:
        raise ProcessIntelligenceError(
            "report.model_name must match regression model metadata"
        )
    resolved_key = _resolve_estimator_key(outcome.regression_model, metadata)
    if resolved_key != report.estimator_key:
        raise ProcessIntelligenceError(
            "report.estimator_key must match regression model estimator key"
        )

    calibration = outcome.residual_detector.calibration
    if calibration is None:
        raise ProcessIntelligenceError(
            "outcome.residual_detector.calibration must be available"
        )
    if calibration.method is not report.detector_method:
        raise ProcessIntelligenceError(
            "report.detector_method must match detector calibration method"
        )
    if not _values_close(
        float(outcome.residual_detector.threshold or -1.0),
        float(report.threshold),
    ):
        raise ProcessIntelligenceError(
            "report.threshold must match detector threshold"
        )

    _assert_state_unchanged(
        regression_model=outcome.regression_model,
        residual_detector=outcome.residual_detector,
        model_snapshot=model_snapshot,
        detector_snapshot=detector_snapshot,
    )
