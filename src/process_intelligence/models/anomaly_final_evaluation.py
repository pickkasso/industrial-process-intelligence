"""Final unsupervised anomaly evaluation on the held-out test partition (Step 7E).

Instantiates the screening-selected ``ModelSpec``, optionally refits on
train+validation, then scores the independent test partition exactly once.
Test metrics are recorded only; they never drive model reselection, threshold
changes, or additional fitting.
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
from process_intelligence.core.protocols import BaseAnomalyModel
from process_intelligence.core.schemas import ModelMetadata, ModelSpec
from process_intelligence.data.loader import ORIGINAL_ROW_ID_COLUMN
from process_intelligence.evaluation.leakage import LeakageReport, LeakageSeverity
from process_intelligence.evaluation.splitting import DatasetSplit
from process_intelligence.models.anomaly import AnomalyDetectionResult
from process_intelligence.models.anomaly_screening import (
    AnomalyCandidateRunStatus,
    AnomalyCandidateScreeningResult,
    AnomalyScreeningOutcome,
    AnomalyScreeningSummary,
)
from process_intelligence.models.registry import ModelRegistry

_SCORE_COLUMN = "_anomaly_score"
_IS_ANOMALY_COLUMN = "_is_anomaly"
_RAW_PREDICTION_COLUMN = "_anomaly_raw_prediction"
_DATA_PARTITION_COLUMN = "_data_partition"
_RESERVED_RESULT_COLUMNS: tuple[str, ...] = (
    _SCORE_COLUMN,
    _IS_ANOMALY_COLUMN,
    _RAW_PREDICTION_COLUMN,
    _DATA_PARTITION_COLUMN,
)

_TEST_METRIC_KEYS: tuple[str, ...] = (
    "test_anomaly_fraction",
    "test_score_min",
    "test_score_max",
    "test_score_mean",
    "test_score_std",
    "test_score_range",
)

_VALIDATION_REQUIRED_METRICS: tuple[str, ...] = (
    "validation_anomaly_fraction",
    "validation_score_mean",
    "validation_score_std",
)

_FLAG_FRACTION_BELOW = "test anomaly fraction below expected minimum"
_FLAG_FRACTION_ABOVE = "test anomaly fraction above expected maximum"
_FLAG_FRACTION_SHIFT = "validation-test anomaly fraction shift is large"
_FLAG_SCORE_NEARLY_CONSTANT = "test anomaly scores are nearly constant"
_FLAG_DEGENERATE = "test detection is degenerate"
_FLAG_NEGATIVE_SEPARATION = (
    "anomaly and normal score direction is inconsistent"
)

_WARNING_LEAKAGE = (
    "LeakageReport contains WARNING issues; proceed with caution"
)
_WARNING_NO_REFIT = (
    "Final anomaly model was not refit on train+validation; "
    "using the screening-selected fitted model"
)
_WARNING_QUALITY_FLAGS = (
    "Final anomaly evaluation reported one or more quality flags"
)
_WARNING_DISTRIBUTION_SHIFT = (
    "Validation-test anomaly fraction shift exceeds policy maximum"
)
_WARNING_CANDIDATE_QUALITY = (
    "Selected anomaly screening candidate has one or more quality penalties"
)
_WARNING_NO_ANOMALIES = "Test partition contains no detected anomalies"
_WARNING_ALL_ANOMALIES = "Test partition marks every row as an anomaly"


class AnomalyFinalEvaluationPolicy(BaseModel):
    """Configurable rules for final unsupervised anomaly evaluation.

    Controls train+validation refit, leakage and screening-consistency gates,
    minimum test size, and label-free quality / distribution-shift thresholds.
    """

    refit_on_train_validation: bool = True
    require_safe_leakage_report: bool = True
    require_screening_consistency: bool = True
    minimum_test_rows: int = 1
    minimum_anomaly_fraction: float = 0.001
    maximum_anomaly_fraction: float = 0.25
    maximum_fraction_shift: float = 0.10
    minimum_score_std: float = 1e-12
    warn_on_distribution_shift: bool = True

    @field_validator(
        "refit_on_train_validation",
        "require_safe_leakage_report",
        "require_screening_consistency",
        "warn_on_distribution_shift",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError(f"must be a bool, got {type(value).__name__}")
        return value

    @field_validator("minimum_test_rows", mode="before")
    @classmethod
    def _validate_minimum_test_rows(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "minimum_test_rows must be an int >= 1 "
                f"(bool not allowed), got {type(value).__name__}"
            )
        if value < 1:
            raise ValueError(f"minimum_test_rows must be >= 1, got {value}")
        return value

    @field_validator(
        "minimum_anomaly_fraction",
        "maximum_anomaly_fraction",
        "maximum_fraction_shift",
        mode="before",
    )
    @classmethod
    def _validate_fraction_fields(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "fraction fields must be finite floats in [0.0, 1.0] "
                f"(bool not allowed), got {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"fraction fields must be finite, got {value!r}")
        if number < 0.0 or number > 1.0:
            raise ValueError(
                f"fraction fields must be in [0.0, 1.0], got {number}"
            )
        return number

    @field_validator("minimum_score_std", mode="before")
    @classmethod
    def _validate_minimum_score_std(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "minimum_score_std must be a finite float > 0 "
                f"(bool not allowed), got {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"minimum_score_std must be finite, got {value!r}")
        if number <= 0.0:
            raise ValueError(f"minimum_score_std must be > 0, got {number}")
        return number

    @model_validator(mode="after")
    def _validate_fraction_relationship(self) -> Self:
        if self.minimum_anomaly_fraction >= self.maximum_anomaly_fraction:
            raise ValueError(
                "minimum_anomaly_fraction must be < maximum_anomaly_fraction, "
                f"got {self.minimum_anomaly_fraction} >= "
                f"{self.maximum_anomaly_fraction}"
            )
        return self


class AnomalyFinalEvaluationReport(BaseModel):
    """Structured report of a single final anomaly refit and test scoring.

    Records selected model identity, partition sizes, preserved validation
    metrics, label-free test metrics, distribution-shift summaries, quality
    flags, timings, and deterministic warnings. Does not store estimators,
    training frames, or full prediction arrays.
    """

    task: AnalysisTask
    model_name: str
    estimator_key: str
    feature_columns: list[str]
    refit_on_train_validation: bool
    fit_partitions: list[Literal["train", "validation"]]
    train_row_count: int
    validation_row_count: int
    test_row_count: int
    final_fit_row_count: int
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    validation_score_separation: float | None = None
    test_score_separation: float | None = None
    test_has_normal_and_anomaly: bool
    anomaly_fraction_shift: float
    score_mean_shift: float
    score_std_ratio: float
    quality_flags: list[str] = Field(default_factory=list)
    fit_seconds: float
    test_scoring_seconds: float
    total_seconds: float
    evaluated_at: datetime
    warnings: list[str] = Field(default_factory=list)

    @field_validator("task", mode="before")
    @classmethod
    def _validate_task(cls, value: object) -> AnalysisTask:
        if not isinstance(value, AnalysisTask):
            raise ValueError(
                f"task must be AnalysisTask, got {type(value).__name__}"
            )
        if value is not AnalysisTask.UNSUPERVISED_ANOMALY:
            raise ValueError(
                "task must be AnalysisTask.UNSUPERVISED_ANOMALY, "
                f"got {value!r}"
            )
        return value

    @field_validator("model_name", "estimator_key", mode="before")
    @classmethod
    def _validate_non_empty_str(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(f"must be str, got {type(value).__name__}")
        if value == "" or value.strip() == "":
            raise ValueError("must be a non-empty, non-whitespace string")
        return value

    @field_validator("feature_columns", mode="after")
    @classmethod
    def _validate_feature_columns(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("feature_columns must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("feature_columns must not contain duplicates")
        if ORIGINAL_ROW_ID_COLUMN in value:
            raise ValueError(
                f"feature_columns must not include {ORIGINAL_ROW_ID_COLUMN!r}"
            )
        return list(value)

    @field_validator("refit_on_train_validation", "test_has_normal_and_anomaly", mode="before")
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError(f"must be a bool, got {type(value).__name__}")
        return value

    @field_validator("fit_partitions", mode="after")
    @classmethod
    def _validate_fit_partitions(
        cls,
        value: list[Literal["train", "validation"]],
    ) -> list[Literal["train", "validation"]]:
        if len(value) != len(set(value)):
            raise ValueError("fit_partitions must not contain duplicates")
        return list(value)

    @field_validator(
        "train_row_count",
        "validation_row_count",
        "test_row_count",
        "final_fit_row_count",
        mode="before",
    )
    @classmethod
    def _validate_row_counts(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "row counts must be int >= 0 (bool not allowed), "
                f"got {type(value).__name__}"
            )
        if value < 0:
            raise ValueError(f"row counts must be >= 0, got {value}")
        return value

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

    @field_validator(
        "validation_score_separation",
        "test_score_separation",
        mode="before",
    )
    @classmethod
    def _validate_optional_separation(cls, value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "score separation must be None or a finite number "
                f"(bool not allowed), got {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"score separation must be finite, got {value!r}")
        return number

    @field_validator("anomaly_fraction_shift", "score_std_ratio", mode="before")
    @classmethod
    def _validate_non_negative_finite(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "must be a finite number >= 0 (bool not allowed), "
                f"got {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"must be finite, got {value!r}")
        if number < 0.0:
            raise ValueError(f"must be >= 0, got {number}")
        return number

    @field_validator("score_mean_shift", mode="before")
    @classmethod
    def _validate_score_mean_shift(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "score_mean_shift must be a finite number "
                f"(bool not allowed), got {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"score_mean_shift must be finite, got {value!r}")
        return number

    @field_validator(
        "fit_seconds",
        "test_scoring_seconds",
        "total_seconds",
        mode="before",
    )
    @classmethod
    def _validate_timing(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "timing fields must be finite numbers >= 0 "
                f"(bool not allowed), got {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"timing fields must be finite, got {value!r}")
        if number < 0.0:
            raise ValueError(f"timing fields must be >= 0, got {number}")
        return number

    @field_validator("evaluated_at", mode="after")
    @classmethod
    def _validate_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("evaluated_at must be timezone-aware")
        return value

    @field_validator("quality_flags", "warnings", mode="after")
    @classmethod
    def _reject_duplicate_strings(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("list must not contain duplicates")
        return list(value)

    @model_validator(mode="after")
    def _validate_report_consistency(self) -> Self:
        if self.train_row_count < 1:
            raise ValueError("train_row_count must be >= 1")
        if self.validation_row_count < 1:
            raise ValueError("validation_row_count must be >= 1")
        if self.test_row_count < 1:
            raise ValueError("test_row_count must be >= 1")
        if self.final_fit_row_count < 1:
            raise ValueError("final_fit_row_count must be >= 1")

        if self.refit_on_train_validation:
            if list(self.fit_partitions) != ["train", "validation"]:
                raise ValueError(
                    "when refit_on_train_validation=True, fit_partitions must "
                    "be exactly ['train', 'validation']"
                )
            expected_fit = self.train_row_count + self.validation_row_count
            if self.final_fit_row_count != expected_fit:
                raise ValueError(
                    "when refit_on_train_validation=True, final_fit_row_count "
                    "must equal train_row_count + validation_row_count"
                )
        else:
            if list(self.fit_partitions) != ["train"]:
                raise ValueError(
                    "when refit_on_train_validation=False, fit_partitions must "
                    "be exactly ['train']"
                )
            if self.final_fit_row_count != self.train_row_count:
                raise ValueError(
                    "when refit_on_train_validation=False, final_fit_row_count "
                    "must equal train_row_count"
                )

        for key in _TEST_METRIC_KEYS:
            if key not in self.test_metrics:
                raise ValueError(f"test_metrics requires key {key!r}")
        for key in _VALIDATION_REQUIRED_METRICS:
            if key not in self.validation_metrics:
                raise ValueError(f"validation_metrics requires key {key!r}")

        if self.test_has_normal_and_anomaly and self.test_score_separation is None:
            raise ValueError(
                "test_score_separation is required when "
                "test_has_normal_and_anomaly=True"
            )

        expected_total = self.fit_seconds + self.test_scoring_seconds
        if not math.isclose(
            self.total_seconds,
            expected_total,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "total_seconds must equal fit_seconds + test_scoring_seconds"
            )
        return self


@dataclass(frozen=True, slots=True)
class AnomalyFinalEvaluationOutcome:
    """Fitted final anomaly model paired with scored test data and report."""

    final_model: BaseAnomalyModel
    test_scored: pl.DataFrame
    report: AnomalyFinalEvaluationReport


class AnomalyFinalEvaluator:
    """Refit a screening-selected anomaly model and score test once.

    Uses the registry only to instantiate a fresh model for optional refit.
    Never mutates the registry, screening outcome, split partitions, feature
    list, or leakage report. Test data is used solely for the single final
    ``detect`` call after the final model is confirmed.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        policy: AnomalyFinalEvaluationPolicy | None = None,
    ) -> None:
        """Create an evaluator with a read-only registry and isolated policy.

        Args:
            registry: Model registry used to instantiate the selected specification.
            policy: Optional evaluation policy. When ``None``, defaults are used.

        Raises:
            TypeError: If ``registry`` or ``policy`` has an invalid type.
        """
        if not isinstance(registry, ModelRegistry):
            raise TypeError(
                f"registry must be ModelRegistry, got {type(registry).__name__}"
            )
        if policy is None:
            stored_policy = AnomalyFinalEvaluationPolicy()
        elif isinstance(policy, AnomalyFinalEvaluationPolicy):
            stored_policy = policy.model_copy(deep=True)
        else:
            raise TypeError(
                "policy must be AnomalyFinalEvaluationPolicy or None, "
                f"got {type(policy).__name__}"
            )
        self._registry = registry
        self._policy = stored_policy

    def evaluate(
        self,
        split: DatasetSplit,
        screening_outcome: AnomalyScreeningOutcome,
        *,
        feature_columns: Sequence[str],
        leakage_report: LeakageReport,
    ) -> AnomalyFinalEvaluationOutcome:
        """Refit the selected anomaly model and score the test partition once.

        Args:
            split: Train/validation/test partitions. Test is used only for the
                final ``detect`` call after the model is confirmed.
            screening_outcome: Screening result providing the selected
                specification and validation metrics.
            feature_columns: Feature column names in evaluation order.
            leakage_report: Structural leakage report for the feature set.

        Returns:
            Outcome containing the fitted final model, scored test frame, and
            evaluation report.

        Raises:
            TypeError: If inputs have invalid types.
            DataValidationError: If inputs fail structural or consistency checks.
            DataLeakageError: If blockers are present and the policy requires a
                safe leakage report.
            InsufficientDataError: If any partition is below the required size.
            ProcessIntelligenceError: If detection results or model metadata are
                inconsistent with the final-evaluation contract.
        """
        validated_features = _validate_feature_columns(feature_columns)
        validated_split = _validate_split(
            split,
            feature_columns=validated_features,
        )
        validated_outcome = _validate_screening_outcome(screening_outcome)
        _validate_leakage_report(
            leakage_report,
            feature_columns=validated_features,
            require_safe=self._policy.require_safe_leakage_report,
        )

        train = validated_split.train
        validation = validated_split.validation
        test = validated_split.test

        if train.height < 1:
            raise InsufficientDataError(
                "train partition must contain at least one row for "
                "anomaly final evaluation"
            )
        if validation.height < 1:
            raise InsufficientDataError(
                "validation partition must contain at least one row for "
                "anomaly final evaluation"
            )
        if test.height < self._policy.minimum_test_rows:
            raise InsufficientDataError(
                "test partition must contain at least "
                f"{self._policy.minimum_test_rows} rows for anomaly final "
                f"evaluation, got {test.height}"
            )

        _reject_reserved_result_columns(validated_split)

        if self._policy.require_screening_consistency:
            _validate_screening_consistency(
                outcome=validated_outcome,
                feature_columns=validated_features,
                split=validated_split,
            )
        else:
            _validate_screening_basics(validated_outcome)

        selected_candidate = _find_selected_candidate(validated_outcome.summary)
        selected_spec = selected_candidate.spec
        validation_metrics = _copy_validation_metrics(
            validated_outcome.summary.selected_metrics
        )
        validation_score_separation = (
            validated_outcome.summary.selected_score_separation
        )

        fit_seconds = 0.0
        fit_partitions: list[Literal["train", "validation"]]
        if self._policy.refit_on_train_validation:
            final_model = self._registry.instantiate(selected_spec)
            if not isinstance(final_model, BaseAnomalyModel):
                raise ProcessIntelligenceError(
                    "registry.instantiate must return BaseAnomalyModel for "
                    "anomaly final evaluation, "
                    f"got {type(final_model).__name__}"
                )
            if final_model is validated_outcome.selected_model:
                raise ProcessIntelligenceError(
                    "final model must be a new instance distinct from the "
                    "screening selected_model when refitting"
                )
            fit_frame = pl.concat([train, validation], how="vertical")
            x_fit = fit_frame.select(validated_features)
            fit_started = time.perf_counter()
            final_model.fit(x_fit)
            fit_seconds = time.perf_counter() - fit_started
            final_fit_row_count = int(train.height + validation.height)
            fit_partitions = ["train", "validation"]
        else:
            final_model = validated_outcome.selected_model
            if not isinstance(final_model, BaseAnomalyModel):
                raise ProcessIntelligenceError(
                    "screening selected_model must be BaseAnomalyModel, "
                    f"got {type(final_model).__name__}"
                )
            if hasattr(final_model, "is_fitted") and not bool(final_model.is_fitted):
                raise ProcessIntelligenceError(
                    "screening selected_model must be fitted when "
                    "refit_on_train_validation is False"
                )
            feature_names = getattr(final_model, "feature_names", None)
            if feature_names is not None and list(feature_names) != list(
                validated_features
            ):
                raise ProcessIntelligenceError(
                    "screening selected_model feature_names must match "
                    f"feature_columns: expected {list(validated_features)}, "
                    f"got {list(feature_names)}"
                )
            fit_row_count = getattr(final_model, "fit_row_count", None)
            if fit_row_count is not None and fit_row_count != int(train.height):
                raise ProcessIntelligenceError(
                    "screening selected_model fit_row_count must equal train "
                    f"height ({train.height}), got {fit_row_count}"
                )
            final_fit_row_count = int(train.height)
            fit_partitions = ["train"]

        x_test = test.select(validated_features)
        detect = getattr(final_model, "detect", None)
        if not callable(detect):
            raise ProcessIntelligenceError(
                "final anomaly model must expose a callable detect method, "
                f"got {type(final_model).__name__}"
            )
        scoring_started = time.perf_counter()
        detection = detect(x_test)
        test_scoring_seconds = time.perf_counter() - scoring_started

        detection_result = _validate_detection_result(
            detection,
            expected_row_count=int(test.height),
        )
        test_scored = _build_test_scored_frame(
            test,
            result=detection_result,
        )
        _assert_scored_alignment(partition=test, scored=test_scored)

        test_metrics, test_score_separation, has_groups = _compute_test_metrics(
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
        score_std_ratio = test_metrics["test_score_std"] / max(
            validation_score_std,
            self._policy.minimum_score_std,
        )
        for name, value in (
            ("anomaly_fraction_shift", anomaly_fraction_shift),
            ("score_mean_shift", score_mean_shift),
            ("score_std_ratio", score_std_ratio),
        ):
            if not math.isfinite(value):
                raise ProcessIntelligenceError(
                    f"{name} must be finite, got {value!r}"
                )

        quality_flags = _compute_quality_flags(
            test_metrics=test_metrics,
            anomaly_fraction_shift=anomaly_fraction_shift,
            has_normal_and_anomaly=has_groups,
            score_separation=test_score_separation,
            policy=self._policy,
        )
        warnings = _build_warnings(
            leakage_report=leakage_report,
            refit_on_train_validation=self._policy.refit_on_train_validation,
            quality_flags=quality_flags,
            anomaly_fraction_shift=anomaly_fraction_shift,
            warn_on_distribution_shift=self._policy.warn_on_distribution_shift,
            maximum_fraction_shift=self._policy.maximum_fraction_shift,
            selected_candidate=selected_candidate,
            detection_result=detection_result,
        )

        _assert_final_model_metadata(
            final_model=final_model,
            selected_spec=selected_spec,
            feature_columns=validated_features,
            final_fit_row_count=final_fit_row_count,
        )

        report = AnomalyFinalEvaluationReport(
            task=AnalysisTask.UNSUPERVISED_ANOMALY,
            model_name=selected_spec.name,
            estimator_key=selected_spec.estimator_key,
            feature_columns=list(validated_features),
            refit_on_train_validation=self._policy.refit_on_train_validation,
            fit_partitions=list(fit_partitions),
            train_row_count=int(train.height),
            validation_row_count=int(validation.height),
            test_row_count=int(test.height),
            final_fit_row_count=final_fit_row_count,
            validation_metrics=dict(validation_metrics),
            test_metrics=dict(test_metrics),
            validation_score_separation=validation_score_separation,
            test_score_separation=test_score_separation,
            test_has_normal_and_anomaly=has_groups,
            anomaly_fraction_shift=float(anomaly_fraction_shift),
            score_mean_shift=float(score_mean_shift),
            score_std_ratio=float(score_std_ratio),
            quality_flags=list(quality_flags),
            fit_seconds=float(fit_seconds),
            test_scoring_seconds=float(test_scoring_seconds),
            total_seconds=float(fit_seconds + test_scoring_seconds),
            evaluated_at=datetime.now(UTC),
            warnings=warnings,
        )
        _assert_outcome_consistency(
            final_model=final_model,
            test_scored=test_scored,
            report=report,
        )
        return AnomalyFinalEvaluationOutcome(
            final_model=final_model,
            test_scored=test_scored,
            report=report,
        )


def _validate_feature_columns(feature_columns: object) -> list[str]:
    """Validate feature column names while preserving input order."""
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
                f"feature_columns[{index}] must be a non-empty, non-whitespace string"
            )
        if name == ORIGINAL_ROW_ID_COLUMN:
            raise DataValidationError(
                "feature_columns must not include reserved column "
                f"{ORIGINAL_ROW_ID_COLUMN!r}"
            )
        if name in _RESERVED_RESULT_COLUMNS:
            raise DataValidationError(
                "feature_columns must not include reserved anomaly result "
                f"column {name!r}"
            )
        if name in seen:
            raise DataValidationError(
                f"feature_columns must not contain duplicates, found {name!r}"
            )
        seen.add(name)
        validated.append(name)

    if not validated:
        raise DataValidationError("feature_columns must contain at least one feature")
    return validated


def _validate_split(
    split: object,
    *,
    feature_columns: Sequence[str],
) -> DatasetSplit:
    """Validate split type, partition frames, schema, dtypes, and columns."""
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

    missing_features = [
        name for name in feature_columns if name not in train_columns
    ]
    if missing_features:
        raise DataValidationError(
            "feature_columns missing from split partitions: "
            f"{', '.join(missing_features)}"
        )
    return split


def _validate_screening_outcome(
    screening_outcome: object,
) -> AnomalyScreeningOutcome:
    """Validate screening outcome type and nested selected model / summary."""
    if not isinstance(screening_outcome, AnomalyScreeningOutcome):
        raise TypeError(
            "screening_outcome must be AnomalyScreeningOutcome, "
            f"got {type(screening_outcome).__name__}"
        )
    if not isinstance(screening_outcome.selected_model, BaseAnomalyModel):
        raise TypeError(
            "screening_outcome.selected_model must be BaseAnomalyModel, "
            f"got {type(screening_outcome.selected_model).__name__}"
        )
    if not isinstance(screening_outcome.summary, AnomalyScreeningSummary):
        raise TypeError(
            "screening_outcome.summary must be AnomalyScreeningSummary, "
            f"got {type(screening_outcome.summary).__name__}"
        )
    return screening_outcome


def _validate_leakage_report(
    leakage_report: object,
    *,
    feature_columns: Sequence[str],
    require_safe: bool,
) -> None:
    """Validate leakage report type, feature alignment, and optional safety gate."""
    if not isinstance(leakage_report, LeakageReport):
        raise TypeError(
            f"leakage_report must be LeakageReport, got {type(leakage_report).__name__}"
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
            "Data leakage blockers detected during anomaly final evaluation: "
            f"count={leakage_report.blocker_count}, "
            f"types={', '.join(blocker_types)}"
        )


def _reject_reserved_result_columns(split: DatasetSplit) -> None:
    """Reject reserved anomaly result columns present in any partition schema."""
    conflicts = [
        name
        for name in _RESERVED_RESULT_COLUMNS
        if name in split.train.columns
    ]
    if conflicts:
        raise DataValidationError(
            "split partitions must not already contain reserved anomaly "
            f"result columns: {', '.join(conflicts)}"
        )


def _validate_screening_basics(outcome: AnomalyScreeningOutcome) -> None:
    """Require basic screening outcome type and task contracts."""
    if outcome.summary.task is not AnalysisTask.UNSUPERVISED_ANOMALY:
        raise DataValidationError(
            "screening summary task must be UNSUPERVISED_ANOMALY, "
            f"got {outcome.summary.task!r}"
        )


def _validate_screening_consistency(
    *,
    outcome: AnomalyScreeningOutcome,
    feature_columns: Sequence[str],
    split: DatasetSplit,
) -> None:
    """Require screening summary and selected model to match current inputs."""
    summary = outcome.summary
    mismatches: list[str] = []

    if summary.task is not AnalysisTask.UNSUPERVISED_ANOMALY:
        mismatches.append(
            f"task (summary={summary.task!r}, "
            f"expected={AnalysisTask.UNSUPERVISED_ANOMALY!r})"
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
            f"(summary={summary.train_row_count}, expected={split.train.height})"
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
            f"(summary={summary.test_row_count}, expected={split.test.height})"
        )
    if summary.selected_model_name == "" or summary.selected_model_name.strip() == "":
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
            f"selected_registry_rank must be >= 0, got {summary.selected_registry_rank}"
        )

    if mismatches:
        raise DataValidationError(
            "screening_outcome is inconsistent with anomaly final evaluation "
            "inputs: " + "; ".join(mismatches)
        )

    selected_model = outcome.selected_model
    metadata = selected_model.get_metadata()
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
    estimator_key = getattr(metadata, "estimator_key", None)
    if estimator_key != summary.selected_estimator_key:
        raise ProcessIntelligenceError(
            "selected model metadata estimator_key does not match summary: "
            f"metadata={estimator_key!r}, "
            f"summary={summary.selected_estimator_key!r}"
        )
    if metadata.task is not AnalysisTask.UNSUPERVISED_ANOMALY:
        raise ProcessIntelligenceError(
            "selected model metadata task must be UNSUPERVISED_ANOMALY, "
            f"got {metadata.task!r}"
        )
    if list(metadata.features) != list(feature_columns):
        raise ProcessIntelligenceError(
            "selected model metadata features must match feature_columns: "
            f"expected {list(feature_columns)}, got {list(metadata.features)}"
        )
    fit_row_count = getattr(metadata, "fit_row_count", None)
    if fit_row_count != split.train.height:
        raise ProcessIntelligenceError(
            "selected model metadata fit_row_count must equal train height "
            f"({split.train.height}), got {fit_row_count}"
        )


def _find_selected_candidate(
    summary: AnomalyScreeningSummary,
) -> AnomalyCandidateScreeningResult:
    """Locate the unique SUCCESS candidate matching the screening selection."""
    matches = [
        result
        for result in summary.candidate_results
        if (
            result.status is AnomalyCandidateRunStatus.SUCCESS
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


def _copy_validation_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Copy screening validation metrics without mutating the input."""
    if not isinstance(metrics, dict) or not metrics:
        raise ProcessIntelligenceError(
            "validation metrics from screening must be a non-empty dict"
        )
    cleaned: dict[str, float] = {}
    for key, raw in metrics.items():
        if not isinstance(key, str) or not key:
            raise ProcessIntelligenceError(
                "validation metric keys must be non-empty strings"
            )
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ProcessIntelligenceError(
                f"validation metric {key!r} must be a finite number "
                f"(bool not allowed), got {type(raw).__name__}"
            )
        number = float(raw)
        if not math.isfinite(number):
            raise ProcessIntelligenceError(
                f"validation metric {key!r} must be finite, got {raw!r}"
            )
        cleaned[key] = number
    for key in _VALIDATION_REQUIRED_METRICS:
        if key not in cleaned:
            raise ProcessIntelligenceError(
                f"validation metrics missing required key {key!r}"
            )
    return cleaned


def _validate_detection_result(
    result: object,
    *,
    expected_row_count: int,
) -> AnomalyDetectionResult:
    """Validate an ``AnomalyDetectionResult`` against test expectations."""
    if not isinstance(result, AnomalyDetectionResult):
        raise ProcessIntelligenceError(
            "test detect must return AnomalyDetectionResult, "
            f"got {type(result).__name__}"
        )
    if result.row_count != expected_row_count:
        raise ProcessIntelligenceError(
            f"test detect row_count ({result.row_count}) must equal "
            f"test height ({expected_row_count})"
        )
    if len(result.scores) != expected_row_count:
        raise ProcessIntelligenceError(
            "test detect scores length must equal test height"
        )
    if len(result.is_anomaly) != expected_row_count:
        raise ProcessIntelligenceError(
            "test detect is_anomaly length must equal test height"
        )
    if len(result.raw_predictions) != expected_row_count:
        raise ProcessIntelligenceError(
            "test detect raw_predictions length must equal test height"
        )
    for index, score in enumerate(result.scores):
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ProcessIntelligenceError(
                f"test detect scores[{index}] must be a finite number"
            )
        if not math.isfinite(float(score)):
            raise ProcessIntelligenceError(
                f"test detect scores[{index}] must be finite, got {score!r}"
            )
    for index, raw in enumerate(result.raw_predictions):
        if raw not in (-1, 1):
            raise ProcessIntelligenceError(
                f"test detect raw_predictions[{index}] must be -1 or 1, "
                f"got {raw!r}"
            )
        expected_anomaly = raw == -1
        if bool(result.is_anomaly[index]) is not expected_anomaly:
            raise ProcessIntelligenceError(
                "test detect is_anomaly must match raw_predictions == -1 "
                f"at index {index}"
            )
    true_count = sum(1 for flag in result.is_anomaly if flag)
    if true_count != result.anomaly_count:
        raise ProcessIntelligenceError(
            "test detect anomaly_count must equal True flags "
            f"({true_count}), got {result.anomaly_count}"
        )
    expected_fraction = result.anomaly_count / expected_row_count
    if not math.isclose(
        float(result.anomaly_fraction),
        expected_fraction,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ProcessIntelligenceError(
            "test detect anomaly_fraction must equal anomaly_count / row_count"
        )
    if (
        result.score_min is None
        or result.score_max is None
        or result.score_mean is None
    ):
        raise ProcessIntelligenceError(
            "test detect requires score_min/max/mean when row_count >= 1"
        )
    for name, summary in (
        ("score_min", result.score_min),
        ("score_max", result.score_max),
        ("score_mean", result.score_mean),
    ):
        if isinstance(summary, bool) or not isinstance(summary, (int, float)):
            raise ProcessIntelligenceError(
                f"test detect {name} must be a finite number"
            )
        if not math.isfinite(float(summary)):
            raise ProcessIntelligenceError(f"test detect {name} must be finite")
    if not (result.score_min <= result.score_mean <= result.score_max):
        raise ProcessIntelligenceError(
            "test detect scores must satisfy score_min <= score_mean <= score_max"
        )
    if isinstance(result.threshold, bool) or not isinstance(
        result.threshold, (int, float)
    ):
        raise ProcessIntelligenceError(
            "test detect threshold must be a finite number"
        )
    if not math.isfinite(float(result.threshold)):
        raise ProcessIntelligenceError("test detect threshold must be finite")
    return result


def _build_test_scored_frame(
    partition: pl.DataFrame,
    *,
    result: AnomalyDetectionResult,
) -> pl.DataFrame:
    """Append anomaly result columns to the test partition without mutation."""
    height = partition.height
    scores = list(result.scores)
    flags = list(result.is_anomaly)
    raws = list(result.raw_predictions)
    if len(scores) != height or len(flags) != height or len(raws) != height:
        raise ProcessIntelligenceError(
            "anomaly result lengths must match test frame height "
            f"({height})"
        )
    return partition.with_columns(
        [
            pl.Series(_SCORE_COLUMN, scores, dtype=pl.Float64),
            pl.Series(_IS_ANOMALY_COLUMN, flags, dtype=pl.Boolean),
            pl.Series(_RAW_PREDICTION_COLUMN, raws, dtype=pl.Int64),
            pl.Series(
                _DATA_PARTITION_COLUMN,
                ["test"] * height,
                dtype=pl.String,
            ),
        ]
    )


def _partition_row_ids(frame: pl.DataFrame) -> list[int]:
    """Return integer original row IDs from a partition frame."""
    return [int(value) for value in frame.get_column(ORIGINAL_ROW_ID_COLUMN).to_list()]


def _assert_scored_alignment(
    *,
    partition: pl.DataFrame,
    scored: pl.DataFrame,
) -> None:
    """Require 1:1 row alignment between input test and scored frame."""
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
            "test scored columns must keep original columns then anomaly "
            f"result columns; expected {expected_columns}, "
            f"got {list(scored.columns)}"
        )


def _compute_test_metrics(
    result: AnomalyDetectionResult,
    *,
    minimum_score_std: float,
) -> tuple[dict[str, float], float | None, bool]:
    """Compute ordered label-free test metrics and optional score separation."""
    scores = np.asarray(result.scores, dtype=np.float64)
    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    score_mean = float(np.mean(scores))
    score_std = float(np.std(scores, ddof=0))
    score_range = score_max - score_min
    metrics: dict[str, float] = {
        "test_anomaly_fraction": float(result.anomaly_fraction),
        "test_score_min": score_min,
        "test_score_max": score_max,
        "test_score_mean": score_mean,
        "test_score_std": score_std,
        "test_score_range": score_range,
    }
    for key, value in metrics.items():
        if not math.isfinite(value):
            raise ProcessIntelligenceError(
                f"computed test metric {key!r} must be finite, got {value!r}"
            )

    raw = list(result.raw_predictions)
    anomaly_scores = [
        float(score)
        for score, prediction in zip(result.scores, raw, strict=True)
        if prediction == -1
    ]
    normal_scores = [
        float(score)
        for score, prediction in zip(result.scores, raw, strict=True)
        if prediction == 1
    ]
    has_groups = bool(anomaly_scores) and bool(normal_scores)
    if not has_groups:
        return metrics, None, False

    mean_anomaly = float(sum(anomaly_scores) / len(anomaly_scores))
    mean_normal = float(sum(normal_scores) / len(normal_scores))
    denominator = max(score_std, minimum_score_std)
    separation = (mean_anomaly - mean_normal) / denominator
    if not math.isfinite(separation):
        raise ProcessIntelligenceError(
            f"test_score_separation must be finite, got {separation!r}"
        )
    return metrics, float(separation), True


def _compute_quality_flags(
    *,
    test_metrics: dict[str, float],
    anomaly_fraction_shift: float,
    has_normal_and_anomaly: bool,
    score_separation: float | None,
    policy: AnomalyFinalEvaluationPolicy,
) -> list[str]:
    """Build ordered, deduplicated quality flags for the test scoring result."""
    flags: list[str] = []
    fraction = test_metrics["test_anomaly_fraction"]
    if fraction < policy.minimum_anomaly_fraction:
        flags.append(_FLAG_FRACTION_BELOW)
    if fraction > policy.maximum_anomaly_fraction:
        flags.append(_FLAG_FRACTION_ABOVE)
    if anomaly_fraction_shift > policy.maximum_fraction_shift:
        flags.append(_FLAG_FRACTION_SHIFT)
    if test_metrics["test_score_std"] < policy.minimum_score_std:
        flags.append(_FLAG_SCORE_NEARLY_CONSTANT)
    if not has_normal_and_anomaly:
        flags.append(_FLAG_DEGENERATE)
    if score_separation is not None and score_separation < 0.0:
        flags.append(_FLAG_NEGATIVE_SEPARATION)
    return flags


def _build_warnings(
    *,
    leakage_report: LeakageReport,
    refit_on_train_validation: bool,
    quality_flags: Sequence[str],
    anomaly_fraction_shift: float,
    warn_on_distribution_shift: bool,
    maximum_fraction_shift: float,
    selected_candidate: AnomalyCandidateScreeningResult,
    detection_result: AnomalyDetectionResult,
) -> list[str]:
    """Build deterministic, de-duplicated final-evaluation warnings."""
    warnings: list[str] = []

    if any(
        issue.severity is LeakageSeverity.WARNING for issue in leakage_report.issues
    ):
        warnings.append(_WARNING_LEAKAGE)

    if not refit_on_train_validation:
        warnings.append(_WARNING_NO_REFIT)

    if quality_flags:
        warnings.append(_WARNING_QUALITY_FLAGS)

    if (
        warn_on_distribution_shift
        and anomaly_fraction_shift > maximum_fraction_shift
    ):
        warnings.append(_WARNING_DISTRIBUTION_SHIFT)

    if selected_candidate.quality_penalty_count >= 1:
        warnings.append(_WARNING_CANDIDATE_QUALITY)
    for warning in selected_candidate.warnings:
        if warning and warning not in warnings:
            warnings.append(warning)

    for warning in detection_result.warnings:
        if warning and warning not in warnings:
            warnings.append(warning)

    if detection_result.anomaly_count == 0:
        warnings.append(_WARNING_NO_ANOMALIES)
    if (
        detection_result.row_count > 0
        and detection_result.anomaly_count == detection_result.row_count
    ):
        warnings.append(_WARNING_ALL_ANOMALIES)

    return warnings


def _assert_final_model_metadata(
    *,
    final_model: BaseAnomalyModel,
    selected_spec: ModelSpec,
    feature_columns: Sequence[str],
    final_fit_row_count: int,
) -> None:
    """Verify the fitted final model matches the selected specification."""
    if not isinstance(final_model, BaseAnomalyModel):
        raise ProcessIntelligenceError(
            "final_model must be BaseAnomalyModel, "
            f"got {type(final_model).__name__}"
        )
    if hasattr(final_model, "is_fitted") and not bool(final_model.is_fitted):
        raise ProcessIntelligenceError("final_model must be in a fitted state")

    metadata = final_model.get_metadata()
    if not isinstance(metadata, ModelMetadata):
        raise ProcessIntelligenceError(
            "final_model.get_metadata must return ModelMetadata, "
            f"got {type(metadata).__name__}"
        )
    if metadata.model_name != selected_spec.name:
        raise ProcessIntelligenceError(
            "final model metadata model_name does not match selected spec: "
            f"metadata={metadata.model_name!r}, selected={selected_spec.name!r}"
        )
    estimator_key = getattr(metadata, "estimator_key", None)
    if estimator_key != selected_spec.estimator_key:
        raise ProcessIntelligenceError(
            "final model metadata estimator_key does not match selected spec: "
            f"metadata={estimator_key!r}, "
            f"selected={selected_spec.estimator_key!r}"
        )
    if metadata.task is not AnalysisTask.UNSUPERVISED_ANOMALY:
        raise ProcessIntelligenceError(
            "final model metadata task must be UNSUPERVISED_ANOMALY, "
            f"got {metadata.task!r}"
        )
    if list(metadata.features) != list(feature_columns):
        raise ProcessIntelligenceError(
            "final model metadata features must match feature_columns: "
            f"expected {list(feature_columns)}, got {list(metadata.features)}"
        )
    fit_row_count = getattr(metadata, "fit_row_count", None)
    if fit_row_count != final_fit_row_count:
        raise ProcessIntelligenceError(
            "final model metadata fit_row_count must equal final_fit_row_count "
            f"({final_fit_row_count}), got {fit_row_count}"
        )


def _assert_outcome_consistency(
    *,
    final_model: BaseAnomalyModel,
    test_scored: pl.DataFrame,
    report: AnomalyFinalEvaluationReport,
) -> None:
    """Validate outcome components before returning to the caller."""
    if not isinstance(final_model, BaseAnomalyModel):
        raise ProcessIntelligenceError(
            "outcome final_model must be BaseAnomalyModel, "
            f"got {type(final_model).__name__}"
        )
    if hasattr(final_model, "is_fitted") and not bool(final_model.is_fitted):
        raise ProcessIntelligenceError(
            "outcome final_model must be in a fitted state"
        )
    if not isinstance(test_scored, pl.DataFrame):
        raise ProcessIntelligenceError(
            "outcome test_scored must be a polars.DataFrame, "
            f"got {type(test_scored).__name__}"
        )
    if not isinstance(report, AnomalyFinalEvaluationReport):
        raise ProcessIntelligenceError(
            "outcome report must be AnomalyFinalEvaluationReport, "
            f"got {type(report).__name__}"
        )
    if test_scored.height != report.test_row_count:
        raise ProcessIntelligenceError(
            "test_scored height must equal report.test_row_count: "
            f"{test_scored.height} != {report.test_row_count}"
        )
    anomaly_true_count = int(test_scored.get_column(_IS_ANOMALY_COLUMN).sum())
    expected_fraction = anomaly_true_count / report.test_row_count
    if not math.isclose(
        report.test_metrics["test_anomaly_fraction"],
        expected_fraction,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ProcessIntelligenceError(
            "test_scored anomaly count is inconsistent with "
            "test_metrics['test_anomaly_fraction']"
        )
    scored_ids = _partition_row_ids(test_scored)
    if len(scored_ids) != len(set(scored_ids)):
        raise ProcessIntelligenceError(
            "test_scored `_original_row_id` values must be unique"
        )

    metadata = final_model.get_metadata()
    if metadata.model_name != report.model_name:
        raise ProcessIntelligenceError(
            "final model metadata model_name does not match report: "
            f"metadata={metadata.model_name!r}, report={report.model_name!r}"
        )
    estimator_key = getattr(metadata, "estimator_key", None)
    if estimator_key != report.estimator_key:
        raise ProcessIntelligenceError(
            "final model metadata estimator_key does not match report: "
            f"metadata={estimator_key!r}, report={report.estimator_key!r}"
        )
    if metadata.task is not report.task:
        raise ProcessIntelligenceError(
            "final model metadata task does not match report: "
            f"metadata={metadata.task!r}, report={report.task!r}"
        )
