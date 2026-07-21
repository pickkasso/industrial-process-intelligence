"""Residual anomaly pipeline using a train-fitted regression model (Step 7G).

Reuses a screening-selected fitted regression model, calibrates a
``ResidualAnomalyDetector`` on validation residuals only, scores train and
validation partitions according to policy, and never reads test feature or
target values.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Self

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
    ResidualAnomalyConfig,
    ResidualAnomalyDetector,
    ResidualAnomalyResult,
    ResidualCalibration,
    ResidualThresholdMethod,
)
from process_intelligence.models.screening import (
    CandidateRunStatus,
    CandidateScreeningResult,
    ModelScreeningOutcome,
    ModelScreeningSummary,
)

_PARTITION_NAMES: tuple[Literal["train", "validation"], ...] = (
    "train",
    "validation",
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

_WARNING_LEAKAGE = (
    "LeakageReport contains WARNING issues; proceed with caution"
)
_WARNING_TRAIN_SCORING_DISABLED = (
    "train residual anomaly scoring was disabled by policy"
)
_WARNING_FRACTION_SHIFT = (
    "train/validation residual anomaly fraction shift exceeds the configured maximum"
)
_WARNING_VALIDATION_ZERO_ANOMALIES = (
    "No validation rows were flagged as residual anomalies"
)
_WARNING_VALIDATION_ALL_ANOMALIES = (
    "All validation rows were flagged as residual anomalies"
)
_PARTITION_DISABLED_WARNING = "train scoring disabled by policy"


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


def _require_non_negative_int(value: object, *, field_name: str) -> int:
    """Validate a non-bool integer >= 0."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{field_name} must be an int >= 0 (bool not allowed), "
            f"got {type(value).__name__}"
        )
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0, got {value}")
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


def _require_fraction(value: object, *, field_name: str) -> float:
    """Validate a finite float in [0.0, 1.0]."""
    number = _require_non_negative_finite_float(value, field_name=field_name)
    if number > 1.0:
        raise ValueError(f"{field_name} must be <= 1.0, got {number}")
    return number


def _reject_duplicate_warnings(warnings: list[str]) -> list[str]:
    """Reject duplicate warning strings while preserving order."""
    if len(warnings) != len(set(warnings)):
        raise ValueError("warnings must not contain duplicate values")
    return list(warnings)


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


class ResidualAnomalyPipelinePolicy(BaseModel):
    """Policy knobs for residual anomaly pipeline execution.

    Controls leakage gating, screening and split-summary consistency,
    minimum validation size, whether train is scored, combined-result sorting,
    and train/validation anomaly-fraction shift warnings.
    """

    require_safe_leakage_report: bool = True
    require_screening_consistency: bool = True
    require_split_summary_match: bool = True
    minimum_validation_rows: int = 1
    score_train_partition: bool = True
    sort_combined_by_original_row_id: bool = True
    maximum_anomaly_fraction_shift: float = 0.10
    warn_on_fraction_shift: bool = True

    @field_validator(
        "require_safe_leakage_report",
        "require_screening_consistency",
        "require_split_summary_match",
        "score_train_partition",
        "sort_combined_by_original_row_id",
        "warn_on_fraction_shift",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_real_bool(value, field_name="policy flag")

    @field_validator("minimum_validation_rows", mode="before")
    @classmethod
    def _validate_minimum_validation_rows(cls, value: object) -> int:
        return _require_positive_int(value, field_name="minimum_validation_rows")

    @field_validator("maximum_anomaly_fraction_shift", mode="before")
    @classmethod
    def _validate_maximum_anomaly_fraction_shift(cls, value: object) -> float:
        return _require_fraction(value, field_name="maximum_anomaly_fraction_shift")


class ResidualPartitionSummary(BaseModel):
    """Per-partition residual anomaly scoring summary for one pipeline run."""

    partition: Literal["train", "validation"]
    input_row_count: int
    scored: bool
    prediction_count: int
    anomaly_count: int
    anomaly_fraction: float
    residual_mean: float | None = None
    residual_std: float | None = None
    score_min: float | None = None
    score_max: float | None = None
    score_mean: float | None = None
    score_std: float | None = None
    prediction_seconds: float
    scoring_seconds: float
    total_seconds: float
    warnings: list[str] = Field(default_factory=list)

    @field_validator(
        "input_row_count",
        "prediction_count",
        "anomaly_count",
        mode="before",
    )
    @classmethod
    def _validate_non_negative_counts(cls, value: object) -> int:
        return _require_non_negative_int(value, field_name="row/anomaly count")

    @field_validator("anomaly_fraction", mode="before")
    @classmethod
    def _validate_anomaly_fraction(cls, value: object) -> float:
        return _require_fraction(value, field_name="anomaly_fraction")

    @field_validator(
        "prediction_seconds",
        "scoring_seconds",
        "total_seconds",
        mode="before",
    )
    @classmethod
    def _validate_timing(cls, value: object) -> float:
        return _require_non_negative_finite_float(value, field_name="timing")

    @field_validator("scored", mode="before")
    @classmethod
    def _validate_scored_flag(cls, value: object) -> bool:
        return _require_real_bool(value, field_name="scored")

    @field_validator("warnings", mode="after")
    @classmethod
    def _validate_warnings(cls, value: list[str]) -> list[str]:
        return _reject_duplicate_warnings(value)

    @model_validator(mode="after")
    def _validate_summary_consistency(self) -> Self:
        if self.prediction_count > self.input_row_count:
            raise ValueError(
                "prediction_count cannot exceed input_row_count: "
                f"{self.prediction_count} > {self.input_row_count}"
            )
        if self.anomaly_count > self.prediction_count:
            raise ValueError(
                "anomaly_count cannot exceed prediction_count: "
                f"{self.anomaly_count} > {self.prediction_count}"
            )
        if not _timings_match(
            self.total_seconds,
            self.prediction_seconds + self.scoring_seconds,
        ):
            raise ValueError(
                "total_seconds must equal prediction_seconds + scoring_seconds "
                f"({self.prediction_seconds + self.scoring_seconds}), "
                f"got {self.total_seconds}"
            )

        if self.partition == "validation" and not self.scored:
            raise ValueError("validation ResidualPartitionSummary must have scored=True")

        if not self.scored:
            if self.prediction_count != 0:
                raise ValueError(
                    "scored=False requires prediction_count=0, "
                    f"got {self.prediction_count}"
                )
            if self.anomaly_count != 0:
                raise ValueError(
                    "scored=False requires anomaly_count=0, "
                    f"got {self.anomaly_count}"
                )
            if self.anomaly_fraction != 0.0:
                raise ValueError(
                    "scored=False requires anomaly_fraction=0.0, "
                    f"got {self.anomaly_fraction}"
                )
            if (
                self.residual_mean is not None
                or self.residual_std is not None
                or self.score_min is not None
                or self.score_max is not None
                or self.score_mean is not None
                or self.score_std is not None
            ):
                raise ValueError(
                    "scored=False requires residual and score summaries to be None"
                )
            if (
                self.prediction_seconds != 0.0
                or self.scoring_seconds != 0.0
                or self.total_seconds != 0.0
            ):
                raise ValueError(
                    "scored=False requires prediction_seconds, scoring_seconds, "
                    "and total_seconds to be 0.0"
                )
            return self

        if self.input_row_count >= 1:
            if self.prediction_count != self.input_row_count:
                raise ValueError(
                    "scored=True with input_row_count >= 1 requires "
                    f"prediction_count == input_row_count ({self.input_row_count}), "
                    f"got {self.prediction_count}"
                )
            expected_fraction = self.anomaly_count / self.prediction_count
            if not _fractions_match(self.anomaly_fraction, expected_fraction):
                raise ValueError(
                    "anomaly_fraction must equal anomaly_count / prediction_count "
                    f"({expected_fraction}), got {self.anomaly_fraction}"
                )
            for name, summary in (
                ("residual_mean", self.residual_mean),
                ("residual_std", self.residual_std),
                ("score_min", self.score_min),
                ("score_max", self.score_max),
                ("score_mean", self.score_mean),
                ("score_std", self.score_std),
            ):
                if summary is None or not _is_finite_number(summary):
                    raise ValueError(
                        f"{name} must be a finite float when scored=True and "
                        f"input_row_count >= 1, got {summary!r}"
                    )
            assert self.residual_std is not None
            assert self.score_std is not None
            assert self.score_min is not None
            assert self.score_max is not None
            assert self.score_mean is not None
            if float(self.residual_std) < 0.0:
                raise ValueError(
                    f"residual_std must be >= 0, got {self.residual_std}"
                )
            if float(self.score_std) < 0.0:
                raise ValueError(f"score_std must be >= 0, got {self.score_std}")
            if not (self.score_min <= self.score_mean <= self.score_max):
                raise ValueError(
                    "score summaries must satisfy score_min <= score_mean <= score_max"
                )
        return self


class ResidualAnomalyPipelineReport(BaseModel):
    """Aggregate report for one residual anomaly pipeline run."""

    task: AnalysisTask
    model_name: str
    estimator_key: str
    target_column: str
    feature_columns: list[str]
    detector_method: ResidualThresholdMethod
    calibration_partition: Literal["validation"] = "validation"
    calibration_row_count: int
    residual_center: float
    residual_scale: float | None
    threshold: float
    train_row_count: int
    validation_row_count: int
    test_row_count: int
    partition_summaries: list[ResidualPartitionSummary]
    total_scored_row_count: int
    total_anomaly_count: int
    total_anomaly_fraction: float
    train_validation_anomaly_fraction_shift: float | None
    calibration_seconds: float
    total_prediction_seconds: float
    total_scoring_seconds: float
    total_seconds: float
    combined_sorted_by_original_row_id: bool
    created_at: datetime
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

    @field_validator("calibration_row_count", mode="before")
    @classmethod
    def _validate_calibration_row_count(cls, value: object) -> int:
        return _require_positive_int(value, field_name="calibration_row_count")

    @field_validator("threshold", mode="before")
    @classmethod
    def _validate_threshold(cls, value: object) -> float:
        return _require_non_negative_finite_float(value, field_name="threshold")

    @field_validator("residual_center", mode="before")
    @classmethod
    def _validate_residual_center(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "residual_center must be a finite float (bool not allowed), "
                f"got {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"residual_center must be finite, got {value!r}")
        return number

    @field_validator("residual_scale", mode="before")
    @classmethod
    def _validate_residual_scale(cls, value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "residual_scale must be None or a finite float > 0 "
                f"(bool not allowed), got {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(
                "residual_scale must be None or a finite float > 0, "
                f"got {value!r}"
            )
        return number

    @field_validator("train_row_count", "validation_row_count", mode="before")
    @classmethod
    def _validate_positive_partition_rows(cls, value: object) -> int:
        return _require_positive_int(value, field_name="partition row count")

    @field_validator("test_row_count", mode="before")
    @classmethod
    def _validate_test_row_count(cls, value: object) -> int:
        return _require_non_negative_int(value, field_name="test_row_count")

    @field_validator(
        "total_scored_row_count",
        "total_anomaly_count",
        mode="before",
    )
    @classmethod
    def _validate_total_counts(cls, value: object) -> int:
        return _require_non_negative_int(value, field_name="total count")

    @field_validator("total_anomaly_fraction", mode="before")
    @classmethod
    def _validate_total_fraction(cls, value: object) -> float:
        return _require_fraction(value, field_name="total_anomaly_fraction")

    @field_validator(
        "train_validation_anomaly_fraction_shift",
        mode="before",
    )
    @classmethod
    def _validate_fraction_shift(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_fraction(
            value, field_name="train_validation_anomaly_fraction_shift"
        )

    @field_validator(
        "calibration_seconds",
        "total_prediction_seconds",
        "total_scoring_seconds",
        "total_seconds",
        mode="before",
    )
    @classmethod
    def _validate_timing(cls, value: object) -> float:
        return _require_non_negative_finite_float(value, field_name="timing")

    @field_validator("combined_sorted_by_original_row_id", mode="before")
    @classmethod
    def _validate_sort_flag(cls, value: object) -> bool:
        return _require_real_bool(
            value, field_name="combined_sorted_by_original_row_id"
        )

    @field_validator("created_at", mode="after")
    @classmethod
    def _validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @field_validator("warnings", mode="after")
    @classmethod
    def _validate_warnings(cls, value: list[str]) -> list[str]:
        return _reject_duplicate_warnings(value)

    @model_validator(mode="after")
    def _validate_report_consistency(self) -> Self:
        if self.target_column in self.feature_columns:
            raise ValueError(
                "target_column must not be included in feature_columns"
            )

        if len(self.partition_summaries) != 2:
            raise ValueError(
                "partition_summaries must contain exactly two entries, "
                f"got {len(self.partition_summaries)}"
            )
        expected_order = list(_PARTITION_NAMES)
        actual_order = [item.partition for item in self.partition_summaries]
        if actual_order != expected_order:
            raise ValueError(
                "partition_summaries must be ordered as "
                f"{expected_order}, got {actual_order}"
            )
        if len(set(actual_order)) != 2:
            raise ValueError(
                "partition_summaries must contain each partition exactly once"
            )

        train_summary = self.partition_summaries[0]
        validation_summary = self.partition_summaries[1]
        if not validation_summary.scored:
            raise ValueError("validation partition summary must have scored=True")

        prediction_sum = sum(
            item.prediction_count for item in self.partition_summaries
        )
        anomaly_sum = sum(item.anomaly_count for item in self.partition_summaries)
        if self.total_scored_row_count != prediction_sum:
            raise ValueError(
                "total_scored_row_count must equal the sum of partition "
                f"prediction_count values ({prediction_sum}), "
                f"got {self.total_scored_row_count}"
            )
        if self.total_anomaly_count != anomaly_sum:
            raise ValueError(
                "total_anomaly_count must equal the sum of partition "
                f"anomaly_count values ({anomaly_sum}), "
                f"got {self.total_anomaly_count}"
            )
        if self.total_anomaly_count > self.total_scored_row_count:
            raise ValueError(
                "total_anomaly_count cannot exceed total_scored_row_count: "
                f"{self.total_anomaly_count} > {self.total_scored_row_count}"
            )
        if self.total_scored_row_count == 0:
            if self.total_anomaly_fraction != 0.0:
                raise ValueError(
                    "total_anomaly_fraction must be 0.0 when "
                    "total_scored_row_count is 0"
                )
        else:
            expected = self.total_anomaly_count / self.total_scored_row_count
            if not _fractions_match(self.total_anomaly_fraction, expected):
                raise ValueError(
                    "total_anomaly_fraction must equal "
                    "total_anomaly_count / total_scored_row_count "
                    f"({expected}), got {self.total_anomaly_fraction}"
                )

        if train_summary.scored:
            if self.train_validation_anomaly_fraction_shift is None:
                raise ValueError(
                    "train_validation_anomaly_fraction_shift must be set when "
                    "train scoring is enabled"
                )
        elif self.train_validation_anomaly_fraction_shift is not None:
            raise ValueError(
                "train_validation_anomaly_fraction_shift must be None when "
                "train scoring is disabled"
            )

        expected_total = (
            self.calibration_seconds
            + self.total_prediction_seconds
            + self.total_scoring_seconds
        )
        if not _timings_match(self.total_seconds, expected_total):
            raise ValueError(
                "total_seconds must equal calibration_seconds + "
                "total_prediction_seconds + total_scoring_seconds "
                f"({expected_total}), got {self.total_seconds}"
            )
        return self


@dataclass(frozen=True, slots=True)
class ResidualAnomalyPipelineOutcome:
    """Immutable result of one residual anomaly pipeline run."""

    regression_model: BaseAnalysisModel
    residual_detector: ResidualAnomalyDetector
    train_scored: pl.DataFrame
    validation_scored: pl.DataFrame
    combined_scored: pl.DataFrame
    report: ResidualAnomalyPipelineReport


class ResidualAnomalyPipeline:
    """Calibrate residual anomalies from a screening-selected regression model.

    Each ``run`` creates a fresh ``ResidualAnomalyDetector``, calibrates it on
    validation residuals only, scores enabled partitions, and returns scored
    frames plus a report. Pipeline instances do not cache models or outcomes.
    """

    def __init__(
        self,
        *,
        detector_config: ResidualAnomalyConfig | None = None,
        policy: ResidualAnomalyPipelinePolicy | None = None,
    ) -> None:
        """Create a pipeline with isolated config and policy copies.

        Args:
            detector_config: Residual anomaly configuration. ``None`` uses
                defaults. Deep-copied so later mutations do not affect this
                pipeline.
            policy: Scoring and gating policy. ``None`` uses defaults.
                Deep-copied so later mutations do not affect this pipeline.

        Raises:
            TypeError: If ``detector_config`` or ``policy`` has an invalid type.
        """
        if detector_config is None:
            stored_config = ResidualAnomalyConfig()
        elif isinstance(detector_config, ResidualAnomalyConfig):
            stored_config = detector_config.model_copy(deep=True)
        else:
            raise TypeError(
                "detector_config must be ResidualAnomalyConfig or None, "
                f"got {type(detector_config).__name__}"
            )

        if policy is None:
            stored_policy = ResidualAnomalyPipelinePolicy()
        elif isinstance(policy, ResidualAnomalyPipelinePolicy):
            stored_policy = policy.model_copy(deep=True)
        else:
            raise TypeError(
                "policy must be ResidualAnomalyPipelinePolicy or None, "
                f"got {type(policy).__name__}"
            )

        self._detector_config = stored_config
        self._policy = stored_policy

    def run(
        self,
        split: DatasetSplit,
        screening_outcome: ModelScreeningOutcome,
        *,
        target_column: str,
        feature_columns: Sequence[str],
        leakage_report: LeakageReport,
    ) -> ResidualAnomalyPipelineOutcome:
        """Score residual anomalies without mutating inputs or the selected model.

        Args:
            split: Train/validation/test partitions with summary metadata.
            screening_outcome: Screening outcome whose selected model is reused.
            target_column: Regression target column name.
            feature_columns: Feature names used for prediction, in order.
            leakage_report: Structural leakage report aligned to features.

        Returns:
            Outcome with the selected regression model, calibrated detector,
            per-partition scored frames, combined scored frame, and report.

        Raises:
            TypeError: If inputs have invalid types.
            DataValidationError: If structural or consistency validation fails.
            DataLeakageError: If blockers are present under a safe-leakage policy.
            InsufficientDataError: If train or validation sizes are insufficient.
            ProcessIntelligenceError: If predictions, calibration, or scoring
                results are inconsistent.
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
        _validate_original_row_ids(validated_split)
        if self._policy.require_split_summary_match:
            _validate_split_summary_match(validated_split)
        _validate_reserved_result_columns(validated_split)

        validated_outcome = _validate_screening_outcome(
            screening_outcome,
            target_column=validated_target,
            feature_columns=validated_features,
            split=validated_split,
            require_consistency=self._policy.require_screening_consistency,
        )
        regression_model = validated_outcome.selected_model
        summary = validated_outcome.summary
        _ = _find_selected_candidate(summary)
        estimator_key = _validate_selected_regression_model(
            regression_model,
            summary=summary,
            feature_columns=validated_features,
            train_row_count=validated_split.train.height,
        )

        train = validated_split.train
        validation = validated_split.validation

        x_validation = validation.select(list(validated_features))
        y_validation = validation.get_column(validated_target)

        prediction_started = time.perf_counter()
        validation_prediction = regression_model.predict(x_validation)
        validation_prediction_seconds = time.perf_counter() - prediction_started
        validation_prediction = _validate_prediction_array(
            validation_prediction,
            expected_length=validation.height,
            partition_name="validation",
        )

        train_prediction: np.ndarray | None = None
        train_prediction_seconds = 0.0
        y_train: pl.Series | None = None
        if self._policy.score_train_partition:
            x_train = train.select(list(validated_features))
            y_train = train.get_column(validated_target)
            train_pred_started = time.perf_counter()
            train_prediction = regression_model.predict(x_train)
            train_prediction_seconds = time.perf_counter() - train_pred_started
            train_prediction = _validate_prediction_array(
                train_prediction,
                expected_length=train.height,
                partition_name="train",
            )

        detector = ResidualAnomalyDetector(
            config=self._detector_config.model_copy(deep=True)
        )
        calibration_started = time.perf_counter()
        detector.fit(y_validation, validation_prediction)
        calibration_seconds = time.perf_counter() - calibration_started
        calibration = _validate_fitted_detector(
            detector,
            expected_row_count=validation.height,
            expected_method=self._detector_config.method,
        )

        validation_score_started = time.perf_counter()
        validation_result = detector.detect(y_validation, validation_prediction)
        validation_scoring_seconds = time.perf_counter() - validation_score_started
        validation_result = _validate_residual_result(
            validation_result,
            expected_row_count=validation.height,
            expected_threshold=float(calibration.threshold),
            partition_name="validation",
        )

        train_result: ResidualAnomalyResult | None = None
        train_scoring_seconds = 0.0
        if (
            self._policy.score_train_partition
            and train_prediction is not None
            and y_train is not None
        ):
            train_score_started = time.perf_counter()
            train_result = detector.detect(y_train, train_prediction)
            train_scoring_seconds = time.perf_counter() - train_score_started
            train_result = _validate_residual_result(
                train_result,
                expected_row_count=train.height,
                expected_threshold=float(calibration.threshold),
                partition_name="train",
            )

        validation_scored = _build_scored_frame(
            validation,
            partition_name="validation",
            result=validation_result,
            empty_rows=False,
        )
        _assert_scored_alignment(
            partition=validation,
            scored=validation_scored,
            partition_name="validation",
        )

        if self._policy.score_train_partition and train_result is not None:
            train_scored = _build_scored_frame(
                train,
                partition_name="train",
                result=train_result,
                empty_rows=False,
            )
            _assert_scored_alignment(
                partition=train,
                scored=train_scored,
                partition_name="train",
            )
            train_summary = _build_partition_summary(
                partition_name="train",
                input_row_count=train.height,
                scored=True,
                result=train_result,
                prediction_seconds=float(train_prediction_seconds),
                scoring_seconds=float(train_scoring_seconds),
                extra_warnings=[],
            )
        else:
            train_scored = _build_scored_frame(
                train,
                partition_name="train",
                result=None,
                empty_rows=True,
            )
            train_summary = _build_partition_summary(
                partition_name="train",
                input_row_count=train.height,
                scored=False,
                result=None,
                prediction_seconds=0.0,
                scoring_seconds=0.0,
                extra_warnings=[_PARTITION_DISABLED_WARNING],
            )

        validation_summary = _build_partition_summary(
            partition_name="validation",
            input_row_count=validation.height,
            scored=True,
            result=validation_result,
            prediction_seconds=float(validation_prediction_seconds),
            scoring_seconds=float(validation_scoring_seconds),
            extra_warnings=[],
        )

        combined_scored = pl.concat(
            [train_scored, validation_scored],
            how="vertical",
        )
        if self._policy.sort_combined_by_original_row_id:
            combined_scored = combined_scored.sort(
                ORIGINAL_ROW_ID_COLUMN,
                descending=False,
                maintain_order=True,
            )
        _validate_combined_scored(
            combined_scored,
            train_scored=train_scored,
            validation_scored=validation_scored,
            expected_row_count=(
                train_summary.prediction_count + validation_summary.prediction_count
            ),
        )

        fraction_shift: float | None
        if train_summary.scored:
            fraction_shift = abs(
                float(validation_summary.anomaly_fraction)
                - float(train_summary.anomaly_fraction)
            )
        else:
            fraction_shift = None

        total_scored_row_count = (
            train_summary.prediction_count + validation_summary.prediction_count
        )
        total_anomaly_count = (
            train_summary.anomaly_count + validation_summary.anomaly_count
        )
        if total_scored_row_count == 0:
            total_anomaly_fraction = 0.0
        else:
            total_anomaly_fraction = float(total_anomaly_count) / float(
                total_scored_row_count
            )

        total_prediction_seconds = float(
            train_prediction_seconds + validation_prediction_seconds
        )
        total_scoring_seconds = float(
            train_scoring_seconds + validation_scoring_seconds
        )
        total_seconds = float(
            calibration_seconds + total_prediction_seconds + total_scoring_seconds
        )

        report_warnings = _build_report_warnings(
            leakage_report=leakage_report,
            score_train_partition=self._policy.score_train_partition,
            calibration=calibration,
            validation_result=validation_result,
            train_result=train_result,
            fraction_shift=fraction_shift,
            maximum_anomaly_fraction_shift=(
                self._policy.maximum_anomaly_fraction_shift
            ),
            warn_on_fraction_shift=self._policy.warn_on_fraction_shift,
        )

        metadata = regression_model.get_metadata()
        report = ResidualAnomalyPipelineReport(
            task=AnalysisTask.REGRESSION,
            model_name=str(metadata.model_name),
            estimator_key=str(estimator_key),
            target_column=validated_target,
            feature_columns=list(validated_features),
            detector_method=calibration.method,
            calibration_partition="validation",
            calibration_row_count=int(calibration.row_count),
            residual_center=float(calibration.residual_center),
            residual_scale=(
                None
                if calibration.residual_scale is None
                else float(calibration.residual_scale)
            ),
            threshold=float(calibration.threshold),
            train_row_count=int(train.height),
            validation_row_count=int(validation.height),
            test_row_count=int(validated_split.test.height),
            partition_summaries=[train_summary, validation_summary],
            total_scored_row_count=int(total_scored_row_count),
            total_anomaly_count=int(total_anomaly_count),
            total_anomaly_fraction=float(total_anomaly_fraction),
            train_validation_anomaly_fraction_shift=fraction_shift,
            calibration_seconds=float(calibration_seconds),
            total_prediction_seconds=float(total_prediction_seconds),
            total_scoring_seconds=float(total_scoring_seconds),
            total_seconds=float(total_seconds),
            combined_sorted_by_original_row_id=(
                self._policy.sort_combined_by_original_row_id
            ),
            created_at=datetime.now(UTC),
            warnings=report_warnings,
        )

        outcome = ResidualAnomalyPipelineOutcome(
            regression_model=regression_model,
            residual_detector=detector,
            train_scored=train_scored,
            validation_scored=validation_scored,
            combined_scored=combined_scored,
            report=report,
        )
        _validate_outcome(outcome)
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
    policy: ResidualAnomalyPipelinePolicy,
) -> None:
    """Enforce non-empty train and sufficient validation rows."""
    if split.train.height < 1:
        raise InsufficientDataError(
            "train partition must contain at least one row for residual "
            "anomaly pipeline"
        )
    if split.validation.height < policy.minimum_validation_rows:
        raise InsufficientDataError(
            "validation partition must contain at least "
            f"{policy.minimum_validation_rows} row(s) for residual threshold "
            f"calibration, got {split.validation.height}"
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
            "Data leakage blockers detected during residual anomaly pipeline: "
            f"count={leakage_report.blocker_count}, "
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
    """Validate train/validation IDs without reading test ID values."""
    train_ids = _validate_original_row_id_column(split.train, partition_name="train")
    validation_ids = _validate_original_row_id_column(
        split.validation, partition_name="validation"
    )

    if set(train_ids) & set(validation_ids):
        raise DataValidationError(
            "Overlapping `_original_row_id` values detected between "
            "train and validation partitions"
        )


def _validate_split_summary_match(split: DatasetSplit) -> None:
    """Require train/validation ID order to match ``SplitSummary`` lists."""
    comparisons = (
        ("train", _partition_row_ids(split.train), split.summary.train_original_row_ids),
        (
            "validation",
            _partition_row_ids(split.validation),
            split.summary.validation_original_row_ids,
        ),
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
    present = [
        name
        for name in _RESERVED_RESULT_COLUMNS
        if name in split.train.columns
    ]
    if present:
        raise DataValidationError(
            "input partitions already contain reserved residual result "
            f"columns: {', '.join(present)}"
        )


def _validate_screening_outcome(
    screening_outcome: object,
    *,
    target_column: str,
    feature_columns: Sequence[str],
    split: DatasetSplit,
    require_consistency: bool,
) -> ModelScreeningOutcome:
    """Validate screening outcome type and optional consistency checks."""
    if not isinstance(screening_outcome, ModelScreeningOutcome):
        raise TypeError(
            "screening_outcome must be ModelScreeningOutcome, "
            f"got {type(screening_outcome).__name__}"
        )
    if not isinstance(screening_outcome.selected_model, BaseAnalysisModel):
        raise TypeError(
            "screening_outcome.selected_model must be BaseAnalysisModel, "
            f"got {type(screening_outcome.selected_model).__name__}"
        )
    if not isinstance(screening_outcome.summary, ModelScreeningSummary):
        raise TypeError(
            "screening_outcome.summary must be ModelScreeningSummary, "
            f"got {type(screening_outcome.summary).__name__}"
        )

    summary = screening_outcome.summary
    if summary.task is not AnalysisTask.REGRESSION:
        raise DataValidationError(
            "screening_outcome.summary.task must be AnalysisTask.REGRESSION, "
            f"got {summary.task!r}"
        )

    if require_consistency:
        mismatches: list[str] = []
        if summary.target_column != target_column:
            mismatches.append(
                "target_column "
                f"(summary={summary.target_column!r}, expected={target_column!r})"
            )
        if list(summary.feature_columns) != list(feature_columns):
            mismatches.append(
                "feature_columns "
                f"(summary={list(summary.feature_columns)!r}, "
                f"expected={list(feature_columns)!r})"
            )
        if summary.train_row_count != split.train.height:
            mismatches.append(
                "train_row_count "
                f"(summary={summary.train_row_count}, "
                f"expected={split.train.height})"
            )
        if summary.validation_row_count != split.validation.height:
            mismatches.append(
                "validation_row_count "
                f"(summary={summary.validation_row_count}, "
                f"expected={split.validation.height})"
            )
        if summary.test_row_count != split.test.height:
            mismatches.append(
                "test_row_count "
                f"(summary={summary.test_row_count}, "
                f"expected={split.test.height})"
            )
        if (
            summary.selected_model_name == ""
            or summary.selected_model_name.strip() == ""
        ):
            mismatches.append("selected_model_name is missing or blank")
        if (
            summary.selected_estimator_key == ""
            or summary.selected_estimator_key.strip() == ""
        ):
            mismatches.append("selected_estimator_key is missing or blank")
        if not summary.selected_metrics:
            mismatches.append("selected_metrics is missing or empty")
        if summary.selected_registry_rank < 0:
            mismatches.append(
                "selected_registry_rank must be >= 0, "
                f"got {summary.selected_registry_rank}"
            )
        if mismatches:
            raise DataValidationError(
                "screening_outcome is inconsistent with residual anomaly "
                "pipeline inputs: " + "; ".join(mismatches)
            )
    return screening_outcome


def _find_selected_candidate(
    summary: ModelScreeningSummary,
) -> CandidateScreeningResult:
    """Locate the unique SUCCESS candidate matching the screening selection."""
    matches = [
        result
        for result in summary.candidate_results
        if (
            result.status is CandidateRunStatus.SUCCESS
            and result.registry_rank == summary.selected_registry_rank
            and result.spec.name == summary.selected_model_name
            and result.spec.estimator_key == summary.selected_estimator_key
        )
    ]
    if len(matches) == 0:
        raise DataValidationError(
            "No SUCCESS candidate matches the screening selection: "
            f"name={summary.selected_model_name!r}, "
            f"estimator_key={summary.selected_estimator_key!r}, "
            f"registry_rank={summary.selected_registry_rank}"
        )
    if len(matches) > 1:
        raise DataValidationError(
            "Multiple SUCCESS candidates match the screening selection: "
            f"name={summary.selected_model_name!r}, "
            f"estimator_key={summary.selected_estimator_key!r}, "
            f"registry_rank={summary.selected_registry_rank}, "
            f"match_count={len(matches)}"
        )
    return matches[0]


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


def _validate_selected_regression_model(
    model: BaseAnalysisModel,
    *,
    summary: ModelScreeningSummary,
    feature_columns: Sequence[str],
    train_row_count: int,
) -> str:
    """Validate the screening-selected fitted regression model and return key."""
    if not isinstance(model, BaseAnalysisModel):
        raise ProcessIntelligenceError(
            "selected regression model must be BaseAnalysisModel, "
            f"got {type(model).__name__}"
        )
    if hasattr(model, "is_fitted") and not bool(model.is_fitted):
        raise ProcessIntelligenceError(
            "selected regression model must be in a fitted state"
        )
    predict_method = getattr(model, "predict", None)
    if not callable(predict_method):
        raise ProcessIntelligenceError(
            "selected regression model must provide a callable predict method"
        )

    metadata = model.get_metadata()
    if not isinstance(metadata, ModelMetadata):
        raise ProcessIntelligenceError(
            "selected_model.get_metadata must return ModelMetadata, "
            f"got {type(metadata).__name__}"
        )
    if metadata.model_name != summary.selected_model_name:
        raise ProcessIntelligenceError(
            "selected model metadata name does not match screening summary: "
            f"metadata={metadata.model_name!r}, "
            f"summary={summary.selected_model_name!r}"
        )
    if metadata.task is not AnalysisTask.REGRESSION:
        raise ProcessIntelligenceError(
            "selected model metadata task must be AnalysisTask.REGRESSION, "
            f"got {metadata.task!r}"
        )
    if list(metadata.features) != list(feature_columns):
        raise ProcessIntelligenceError(
            "selected model metadata features must match feature_columns: "
            f"expected {list(feature_columns)}, got {list(metadata.features)}"
        )

    estimator_key = _resolve_estimator_key(model, metadata)
    if estimator_key != summary.selected_estimator_key:
        raise ProcessIntelligenceError(
            "selected model estimator_key does not match screening summary: "
            f"model={estimator_key!r}, "
            f"summary={summary.selected_estimator_key!r}"
        )

    fit_row_count = _resolve_fit_row_count(model, metadata)
    if fit_row_count != train_row_count:
        raise ProcessIntelligenceError(
            "selected model fit_row_count must equal train height "
            f"({train_row_count}), got {fit_row_count}"
        )

    feature_names = getattr(model, "feature_names", None)
    if feature_names is not None and list(feature_names) != list(feature_columns):
        raise ProcessIntelligenceError(
            "selected model feature_names must match feature_columns: "
            f"expected {list(feature_columns)}, got {list(feature_names)}"
        )

    if not isinstance(estimator_key, str) or estimator_key.strip() == "":
        raise ProcessIntelligenceError(
            "selected model estimator_key must be a non-empty string"
        )
    return estimator_key


def _validate_prediction_array(
    prediction: object,
    *,
    expected_length: int,
    partition_name: str,
) -> np.ndarray:
    """Validate a regression prediction array without mutating the input."""
    if not isinstance(prediction, np.ndarray):
        raise ProcessIntelligenceError(
            f"{partition_name} prediction must be a numpy.ndarray, "
            f"got {type(prediction).__name__}"
        )
    if prediction.ndim != 1:
        raise ProcessIntelligenceError(
            f"{partition_name} prediction must be 1-dimensional, "
            f"got shape {prediction.shape}"
        )
    if prediction.shape[0] != expected_length:
        raise ProcessIntelligenceError(
            f"{partition_name} prediction length ({prediction.shape[0]}) must "
            f"equal partition height ({expected_length})"
        )
    if prediction.dtype == np.bool_ or prediction.dtype.kind == "b":
        raise ProcessIntelligenceError(
            f"{partition_name} prediction must be numeric; boolean values "
            "are not allowed"
        )
    if prediction.dtype.kind not in {"f", "i", "u"}:
        raise ProcessIntelligenceError(
            f"{partition_name} prediction must be numeric, "
            f"got dtype {prediction.dtype}"
        )
    if not np.isfinite(prediction.astype(np.float64, copy=False)).all():
        raise ProcessIntelligenceError(
            f"{partition_name} prediction must contain only finite values"
        )
    return prediction


def _validate_fitted_detector(
    detector: ResidualAnomalyDetector,
    *,
    expected_row_count: int,
    expected_method: ResidualThresholdMethod,
) -> ResidualCalibration:
    """Validate detector calibration state after fit."""
    if not detector.is_fitted:
        raise ProcessIntelligenceError(
            "ResidualAnomalyDetector must be fitted after calibration"
        )
    calibration = detector.calibration
    if not isinstance(calibration, ResidualCalibration):
        raise ProcessIntelligenceError(
            "ResidualAnomalyDetector.calibration must be ResidualCalibration "
            f"after fit, got {type(calibration).__name__}"
        )
    if calibration.row_count != expected_row_count:
        raise ProcessIntelligenceError(
            "detector calibration.row_count must equal validation height "
            f"({expected_row_count}), got {calibration.row_count}"
        )
    if calibration.method is not expected_method:
        raise ProcessIntelligenceError(
            "detector calibration.method must match detector config method: "
            f"expected={expected_method!r}, got={calibration.method!r}"
        )
    if not _is_finite_number(calibration.threshold) or float(calibration.threshold) < 0.0:
        raise ProcessIntelligenceError(
            "detector calibration.threshold must be a finite float >= 0, "
            f"got {calibration.threshold!r}"
        )
    if (
        calibration.fitted_at.tzinfo is None
        or calibration.fitted_at.utcoffset() is None
    ):
        raise ProcessIntelligenceError(
            "detector calibration.fitted_at must be timezone-aware UTC"
        )
    return calibration


def _validate_residual_result(
    result: object,
    *,
    expected_row_count: int,
    expected_threshold: float,
    partition_name: str,
) -> ResidualAnomalyResult:
    """Validate ``ResidualAnomalyResult`` consistency against partition size."""
    if not isinstance(result, ResidualAnomalyResult):
        raise ProcessIntelligenceError(
            f"{partition_name} detect must return ResidualAnomalyResult, "
            f"got {type(result).__name__}"
        )
    if result.row_count != expected_row_count:
        raise ProcessIntelligenceError(
            f"{partition_name} ResidualAnomalyResult.row_count must equal "
            f"partition height ({expected_row_count}), got {result.row_count}"
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
                f"{partition_name} ResidualAnomalyResult.{name} length must "
                f"equal partition height ({expected_row_count}), "
                f"got {len(collection)}"
            )

    for index, value in enumerate(result.predictions):
        if not _is_finite_number(value):
            raise ProcessIntelligenceError(
                f"{partition_name} predictions[{index}] must be finite, "
                f"got {value!r}"
            )
    for index, value in enumerate(result.residuals):
        if not _is_finite_number(value):
            raise ProcessIntelligenceError(
                f"{partition_name} residuals[{index}] must be finite, "
                f"got {value!r}"
            )
    for index, value in enumerate(result.absolute_centered_residuals):
        if not _is_finite_number(value) or float(value) < 0.0:
            raise ProcessIntelligenceError(
                f"{partition_name} absolute_centered_residuals[{index}] must "
                f"be a finite float >= 0, got {value!r}"
            )
    for index, value in enumerate(result.scores):
        if not _is_finite_number(value) or float(value) < 0.0:
            raise ProcessIntelligenceError(
                f"{partition_name} scores[{index}] must be a finite float "
                f">= 0, got {value!r}"
            )
    for index, raw in enumerate(result.raw_predictions):
        if raw not in (-1, 1):
            raise ProcessIntelligenceError(
                f"{partition_name} raw_predictions[{index}] must be -1 or 1, "
                f"got {raw!r}"
            )
        expected_flag = raw == -1
        if bool(result.is_anomaly[index]) is not expected_flag:
            raise ProcessIntelligenceError(
                f"{partition_name} is_anomaly must match raw_predictions == -1 "
                f"at index {index}"
            )

    true_count = sum(1 for flag in result.is_anomaly if flag)
    if true_count != result.anomaly_count:
        raise ProcessIntelligenceError(
            f"{partition_name} anomaly_count must equal True count in "
            f"is_anomaly ({true_count}), got {result.anomaly_count}"
        )
    if expected_row_count == 0:
        if result.anomaly_fraction != 0.0:
            raise ProcessIntelligenceError(
                f"{partition_name} anomaly_fraction must be 0.0 for empty "
                "partitions"
            )
    else:
        expected_fraction = result.anomaly_count / expected_row_count
        if not _fractions_match(result.anomaly_fraction, expected_fraction):
            raise ProcessIntelligenceError(
                f"{partition_name} anomaly_fraction must equal "
                f"anomaly_count / row_count ({expected_fraction}), "
                f"got {result.anomaly_fraction}"
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
                    f"{partition_name} {name} must be a finite float when "
                    f"row_count >= 1, got {summary!r}"
                )
        assert result.residual_std is not None
        assert result.score_std is not None
        assert result.score_min is not None
        assert result.score_max is not None
        assert result.score_mean is not None
        if float(result.residual_std) < 0.0:
            raise ProcessIntelligenceError(
                f"{partition_name} residual_std must be >= 0, "
                f"got {result.residual_std}"
            )
        if float(result.score_std) < 0.0:
            raise ProcessIntelligenceError(
                f"{partition_name} score_std must be >= 0, got {result.score_std}"
            )
        if not (result.score_min <= result.score_mean <= result.score_max):
            raise ProcessIntelligenceError(
                f"{partition_name} score summaries must satisfy "
                "score_min <= score_mean <= score_max"
            )

    if not math.isclose(
        float(result.threshold),
        float(expected_threshold),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ProcessIntelligenceError(
            f"{partition_name} result.threshold must equal detector threshold "
            f"({expected_threshold}), got {result.threshold}"
        )
    return result


def _build_scored_frame(
    partition: pl.DataFrame,
    *,
    partition_name: Literal["train", "validation"],
    result: ResidualAnomalyResult | None,
    empty_rows: bool,
) -> pl.DataFrame:
    """Append residual anomaly result columns to a partition frame copy."""
    base = partition.clear() if empty_rows else partition
    height = base.height
    if result is None:
        predictions: list[float] = []
        residuals: list[float] = []
        centered: list[float] = []
        scores: list[float] = []
        flags: list[bool] = []
        raws: list[int] = []
    else:
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

    return base.with_columns(
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
                [partition_name] * height,
                dtype=pl.String,
            ),
        ]
    )


def _assert_scored_alignment(
    *,
    partition: pl.DataFrame,
    scored: pl.DataFrame,
    partition_name: str,
) -> None:
    """Require 1:1 row alignment between input partition and scored frame."""
    if scored.height != partition.height:
        raise ProcessIntelligenceError(
            f"{partition_name} scored height ({scored.height}) must equal "
            f"input height ({partition.height})"
        )
    input_ids = _partition_row_ids(partition)
    scored_ids = _partition_row_ids(scored)
    if scored_ids != input_ids:
        raise ProcessIntelligenceError(
            f"{partition_name} scored `_original_row_id` order must match "
            "the input partition"
        )
    for column in partition.columns:
        if scored.get_column(column).dtype != partition.get_column(column).dtype:
            raise ProcessIntelligenceError(
                f"{partition_name} scored column {column!r} dtype changed"
            )
        if scored.get_column(column).to_list() != partition.get_column(column).to_list():
            raise ProcessIntelligenceError(
                f"{partition_name} scored column {column!r} values changed"
            )
    expected_columns = list(partition.columns) + list(_RESERVED_RESULT_COLUMNS)
    if list(scored.columns) != expected_columns:
        raise ProcessIntelligenceError(
            f"{partition_name} scored columns must keep original columns "
            f"then residual result columns; expected {expected_columns}, "
            f"got {list(scored.columns)}"
        )


def _build_partition_summary(
    *,
    partition_name: Literal["train", "validation"],
    input_row_count: int,
    scored: bool,
    result: ResidualAnomalyResult | None,
    prediction_seconds: float,
    scoring_seconds: float,
    extra_warnings: Sequence[str],
) -> ResidualPartitionSummary:
    """Build one partition summary from scoring results or a disabled state."""
    warnings = _dedupe_preserve_order(list(extra_warnings))
    if not scored or result is None:
        return ResidualPartitionSummary(
            partition=partition_name,
            input_row_count=input_row_count,
            scored=False,
            prediction_count=0,
            anomaly_count=0,
            anomaly_fraction=0.0,
            residual_mean=None,
            residual_std=None,
            score_min=None,
            score_max=None,
            score_mean=None,
            score_std=None,
            prediction_seconds=0.0,
            scoring_seconds=0.0,
            total_seconds=0.0,
            warnings=warnings,
        )

    return ResidualPartitionSummary(
        partition=partition_name,
        input_row_count=input_row_count,
        scored=True,
        prediction_count=int(result.row_count),
        anomaly_count=int(result.anomaly_count),
        anomaly_fraction=float(result.anomaly_fraction),
        residual_mean=result.residual_mean,
        residual_std=result.residual_std,
        score_min=result.score_min,
        score_max=result.score_max,
        score_mean=result.score_mean,
        score_std=result.score_std,
        prediction_seconds=float(prediction_seconds),
        scoring_seconds=float(scoring_seconds),
        total_seconds=float(prediction_seconds + scoring_seconds),
        warnings=warnings,
    )


def _validate_combined_scored(
    combined: pl.DataFrame,
    *,
    train_scored: pl.DataFrame,
    validation_scored: pl.DataFrame,
    expected_row_count: int,
) -> None:
    """Validate combined scored frame schema, size, and unique IDs."""
    if combined.height != expected_row_count:
        raise ProcessIntelligenceError(
            "combined_scored height must equal the sum of scored prediction "
            f"counts ({expected_row_count}), got {combined.height}"
        )
    if list(combined.columns) != list(train_scored.columns):
        raise ProcessIntelligenceError(
            "combined_scored schema must match train_scored schema"
        )
    if list(combined.columns) != list(validation_scored.columns):
        raise ProcessIntelligenceError(
            "combined_scored schema must match validation_scored schema"
        )
    if list(combined.dtypes) != list(train_scored.dtypes):
        raise ProcessIntelligenceError(
            "combined_scored dtypes must match train_scored dtypes"
        )
    if ORIGINAL_ROW_ID_COLUMN in combined.columns and combined.height > 0:
        ids = combined.get_column(ORIGINAL_ROW_ID_COLUMN)
        if ids.n_unique() != ids.len():
            raise ProcessIntelligenceError(
                "combined_scored `_original_row_id` values must be unique"
            )


def _build_report_warnings(
    *,
    leakage_report: LeakageReport,
    score_train_partition: bool,
    calibration: ResidualCalibration,
    validation_result: ResidualAnomalyResult,
    train_result: ResidualAnomalyResult | None,
    fraction_shift: float | None,
    maximum_anomaly_fraction_shift: float,
    warn_on_fraction_shift: bool,
) -> list[str]:
    """Build deterministic, de-duplicated pipeline report warnings."""
    warnings: list[str] = []

    if any(
        issue.severity is LeakageSeverity.WARNING for issue in leakage_report.issues
    ):
        warnings.append(_WARNING_LEAKAGE)

    if not score_train_partition:
        warnings.append(_WARNING_TRAIN_SCORING_DISABLED)

    for warning in calibration.warnings:
        if warning:
            warnings.append(warning)

    for warning in validation_result.warnings:
        if warning:
            warnings.append(warning)

    if train_result is not None:
        for warning in train_result.warnings:
            if warning:
                warnings.append(warning)

    if (
        warn_on_fraction_shift
        and fraction_shift is not None
        and fraction_shift > maximum_anomaly_fraction_shift
    ):
        warnings.append(_WARNING_FRACTION_SHIFT)

    if validation_result.anomaly_count == 0:
        warnings.append(_WARNING_VALIDATION_ZERO_ANOMALIES)
    if (
        validation_result.row_count > 0
        and validation_result.anomaly_count == validation_result.row_count
    ):
        warnings.append(_WARNING_VALIDATION_ALL_ANOMALIES)

    return _dedupe_preserve_order(warnings)


def _validate_outcome(outcome: ResidualAnomalyPipelineOutcome) -> None:
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

    for name, frame in (
        ("train_scored", outcome.train_scored),
        ("validation_scored", outcome.validation_scored),
        ("combined_scored", outcome.combined_scored),
    ):
        if not isinstance(frame, pl.DataFrame):
            raise ProcessIntelligenceError(
                f"outcome.{name} must be a polars.DataFrame, "
                f"got {type(frame).__name__}"
            )

    report = outcome.report
    if not isinstance(report, ResidualAnomalyPipelineReport):
        raise ProcessIntelligenceError(
            "outcome.report must be ResidualAnomalyPipelineReport, "
            f"got {type(report).__name__}"
        )

    if outcome.validation_scored.height != report.validation_row_count:
        raise ProcessIntelligenceError(
            "validation_scored.height must equal report.validation_row_count "
            f"({report.validation_row_count}), "
            f"got {outcome.validation_scored.height}"
        )
    if outcome.combined_scored.height != report.total_scored_row_count:
        raise ProcessIntelligenceError(
            "combined_scored.height must equal report.total_scored_row_count "
            f"({report.total_scored_row_count}), "
            f"got {outcome.combined_scored.height}"
        )

    if outcome.combined_scored.height > 0:
        anomaly_true_count = int(
            outcome.combined_scored.get_column(_IS_RESIDUAL_ANOMALY_COLUMN)
            .sum()
        )
        if anomaly_true_count != report.total_anomaly_count:
            raise ProcessIntelligenceError(
                "combined residual anomaly True count must equal "
                f"report.total_anomaly_count ({report.total_anomaly_count}), "
                f"got {anomaly_true_count}"
            )
        ids = outcome.combined_scored.get_column(ORIGINAL_ROW_ID_COLUMN)
        if ids.n_unique() != ids.len():
            raise ProcessIntelligenceError(
                "combined_scored `_original_row_id` values must be unique"
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
    if not math.isclose(
        float(calibration.threshold),
        float(report.threshold),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ProcessIntelligenceError(
            "report.threshold must match detector threshold"
        )
