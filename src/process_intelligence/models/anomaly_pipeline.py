"""Unsupervised Isolation Forest anomaly detection pipeline (Step 7B).

Fits on the train partition only, scores enabled partitions according to
policy, and joins anomaly results back to original rows via
``_original_row_id`` without mutating inputs.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Self

import polars as pl
from pydantic import BaseModel, Field, field_validator, model_validator

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.exceptions import (
    DataLeakageError,
    DataValidationError,
    InsufficientDataError,
    ProcessIntelligenceError,
)
from process_intelligence.data.loader import ORIGINAL_ROW_ID_COLUMN
from process_intelligence.evaluation.leakage import LeakageReport, LeakageSeverity
from process_intelligence.evaluation.splitting import DatasetSplit
from process_intelligence.models.anomaly import (
    AnomalyDetectionResult,
    IsolationForestAnomalyModel,
    IsolationForestConfig,
    create_isolation_forest_anomaly_model,
)

_PARTITION_NAMES: tuple[Literal["train", "validation", "test"], ...] = (
    "train",
    "validation",
    "test",
)
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

_WARNING_LEAKAGE = (
    "LeakageReport contains WARNING issues; proceed with caution"
)
_WARNING_SCORING_DISABLED = "Scoring was disabled by policy for one or more partitions"
_WARNING_EMPTY_SCORED = (
    "One or more enabled validation/test partitions were empty and scored "
    "as empty results"
)
_WARNING_ZERO_ANOMALIES = "No scored rows were flagged as anomalies"
_WARNING_ALL_ANOMALIES = "All scored rows were flagged as anomalies"
_PARTITION_DISABLED_WARNING = "scoring was disabled by policy"
_PARTITION_EMPTY_SCORED_WARNING = (
    "empty partition was scored as an empty result"
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


class AnomalyPipelinePolicy(BaseModel):
    """Policy knobs for unsupervised anomaly pipeline execution.

    Controls leakage gating, split-summary matching, which partitions are
    scored, and whether combined results are restored to original row order.
    """

    require_safe_leakage_report: bool = True
    require_split_summary_match: bool = True
    score_train_partition: bool = True
    score_validation_partition: bool = True
    score_test_partition: bool = True
    sort_combined_by_original_row_id: bool = True

    @field_validator(
        "require_safe_leakage_report",
        "require_split_summary_match",
        "score_train_partition",
        "score_validation_partition",
        "score_test_partition",
        "sort_combined_by_original_row_id",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_real_bool(value, field_name="policy flag")

    @model_validator(mode="after")
    def _require_at_least_one_score_partition(self) -> Self:
        if not (
            self.score_train_partition
            or self.score_validation_partition
            or self.score_test_partition
        ):
            raise ValueError(
                "at least one of score_train_partition, "
                "score_validation_partition, or score_test_partition "
                "must be True"
            )
        return self


class AnomalyPartitionSummary(BaseModel):
    """Per-partition anomaly scoring summary for one pipeline run."""

    partition: Literal["train", "validation", "test"]
    input_row_count: int
    scored: bool
    scored_row_count: int
    anomaly_count: int
    anomaly_fraction: float
    score_min: float | None = None
    score_max: float | None = None
    score_mean: float | None = None
    scoring_seconds: float
    warnings: list[str] = Field(default_factory=list)

    @field_validator("input_row_count", "scored_row_count", "anomaly_count", mode="before")
    @classmethod
    def _validate_non_negative_counts(cls, value: object) -> int:
        return _require_non_negative_int(value, field_name="row/anomaly count")

    @field_validator("anomaly_fraction", mode="before")
    @classmethod
    def _validate_anomaly_fraction(cls, value: object) -> float:
        return _require_fraction(value, field_name="anomaly_fraction")

    @field_validator("scoring_seconds", mode="before")
    @classmethod
    def _validate_scoring_seconds(cls, value: object) -> float:
        return _require_non_negative_finite_float(
            value, field_name="scoring_seconds"
        )

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
        if self.scored_row_count > self.input_row_count:
            raise ValueError(
                "scored_row_count cannot exceed input_row_count: "
                f"{self.scored_row_count} > {self.input_row_count}"
            )
        if self.anomaly_count > self.scored_row_count:
            raise ValueError(
                "anomaly_count cannot exceed scored_row_count: "
                f"{self.anomaly_count} > {self.scored_row_count}"
            )

        if not self.scored:
            if self.scored_row_count != 0:
                raise ValueError(
                    "scored=False requires scored_row_count=0, "
                    f"got {self.scored_row_count}"
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
                self.score_min is not None
                or self.score_max is not None
                or self.score_mean is not None
            ):
                raise ValueError(
                    "scored=False requires score_min, score_max, and "
                    "score_mean to be None"
                )
            return self

        if self.scored_row_count == 0:
            if self.anomaly_count != 0:
                raise ValueError(
                    "empty scored partition requires anomaly_count=0, "
                    f"got {self.anomaly_count}"
                )
            if self.anomaly_fraction != 0.0:
                raise ValueError(
                    "empty scored partition requires anomaly_fraction=0.0, "
                    f"got {self.anomaly_fraction}"
                )
            if (
                self.score_min is not None
                or self.score_max is not None
                or self.score_mean is not None
            ):
                raise ValueError(
                    "empty scored partition requires score_min, score_max, "
                    "and score_mean to be None"
                )
            return self

        if self.scored_row_count != self.input_row_count:
            raise ValueError(
                "non-empty scored partition requires scored_row_count == "
                f"input_row_count ({self.input_row_count}), "
                f"got {self.scored_row_count}"
            )
        expected_fraction = self.anomaly_count / self.scored_row_count
        if not _fractions_match(self.anomaly_fraction, expected_fraction):
            raise ValueError(
                "anomaly_fraction must equal anomaly_count / scored_row_count "
                f"({expected_fraction}), got {self.anomaly_fraction}"
            )
        for name, summary in (
            ("score_min", self.score_min),
            ("score_max", self.score_max),
            ("score_mean", self.score_mean),
        ):
            if summary is None or not _is_finite_number(summary):
                raise ValueError(
                    f"{name} must be a finite float when scored_row_count >= 1, "
                    f"got {summary!r}"
                )
        assert self.score_min is not None
        assert self.score_max is not None
        assert self.score_mean is not None
        if not (self.score_min <= self.score_mean <= self.score_max):
            raise ValueError(
                "score summaries must satisfy score_min <= score_mean <= score_max"
            )
        return self


class AnomalyPipelineReport(BaseModel):
    """Aggregate report for one unsupervised anomaly pipeline run."""

    task: AnalysisTask
    model_name: str
    estimator_key: str
    feature_columns: list[str]
    fit_partition: Literal["train"] = "train"
    fit_row_count: int
    partition_summaries: list[AnomalyPartitionSummary]
    total_scored_row_count: int
    total_anomaly_count: int
    total_anomaly_fraction: float
    fit_seconds: float
    total_scoring_seconds: float
    total_seconds: float
    combined_sorted_by_original_row_id: bool
    created_at: datetime
    warnings: list[str] = Field(default_factory=list)

    @field_validator("task", mode="after")
    @classmethod
    def _validate_task(cls, value: AnalysisTask) -> AnalysisTask:
        if value is not AnalysisTask.UNSUPERVISED_ANOMALY:
            raise ValueError(
                "task must be AnalysisTask.UNSUPERVISED_ANOMALY, "
                f"got {value!r}"
            )
        return value

    @field_validator("model_name", "estimator_key", mode="before")
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

    @field_validator("fit_partition", mode="after")
    @classmethod
    def _validate_fit_partition(
        cls, value: Literal["train"]
    ) -> Literal["train"]:
        if value != "train":
            raise ValueError(f"fit_partition must be 'train', got {value!r}")
        return value

    @field_validator("fit_row_count", mode="before")
    @classmethod
    def _validate_fit_row_count(cls, value: object) -> int:
        number = _require_non_negative_int(value, field_name="fit_row_count")
        if number < 1:
            raise ValueError(f"fit_row_count must be >= 1, got {number}")
        return number

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
        "fit_seconds",
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
        if len(self.partition_summaries) != 3:
            raise ValueError(
                "partition_summaries must contain exactly three entries, "
                f"got {len(self.partition_summaries)}"
            )
        expected_order = list(_PARTITION_NAMES)
        actual_order = [item.partition for item in self.partition_summaries]
        if actual_order != expected_order:
            raise ValueError(
                "partition_summaries must be ordered as "
                f"{expected_order}, got {actual_order}"
            )
        if len(set(actual_order)) != 3:
            raise ValueError(
                "partition_summaries must contain each partition exactly once"
            )

        scored_sum = sum(item.scored_row_count for item in self.partition_summaries)
        anomaly_sum = sum(item.anomaly_count for item in self.partition_summaries)
        if self.total_scored_row_count != scored_sum:
            raise ValueError(
                "total_scored_row_count must equal the sum of partition "
                f"scored_row_count values ({scored_sum}), "
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

        expected_total = self.fit_seconds + self.total_scoring_seconds
        if not math.isclose(
            self.total_seconds,
            expected_total,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "total_seconds must equal fit_seconds + total_scoring_seconds "
                f"({expected_total}), got {self.total_seconds}"
            )
        return self


@dataclass(frozen=True, slots=True)
class AnomalyPipelineOutcome:
    """Immutable result of one unsupervised anomaly pipeline run."""

    fitted_model: IsolationForestAnomalyModel
    train_scored: pl.DataFrame
    validation_scored: pl.DataFrame
    test_scored: pl.DataFrame
    combined_scored: pl.DataFrame
    report: AnomalyPipelineReport


class UnsupervisedAnomalyPipeline:
    """Train-only Isolation Forest anomaly scoring over a ``DatasetSplit``.

    Each ``run`` call creates a fresh model via
    ``create_isolation_forest_anomaly_model``, fits on train features only,
    scores enabled partitions, and returns scored frames plus a report.
    Pipeline instances do not cache fitted models or outcomes.
    """

    def __init__(
        self,
        *,
        model_config: IsolationForestConfig | None = None,
        policy: AnomalyPipelinePolicy | None = None,
    ) -> None:
        """Create a pipeline with isolated config and policy copies.

        Args:
            model_config: Isolation Forest hyperparameters. ``None`` uses
                defaults. Deep-copied so later mutations do not affect this
                pipeline.
            policy: Scoring and gating policy. ``None`` uses defaults.
                Deep-copied so later mutations do not affect this pipeline.

        Raises:
            TypeError: If ``model_config`` or ``policy`` has an invalid type.
        """
        if model_config is None:
            stored_config = IsolationForestConfig()
        elif isinstance(model_config, IsolationForestConfig):
            stored_config = model_config.model_copy(deep=True)
        else:
            raise TypeError(
                "model_config must be IsolationForestConfig or None, "
                f"got {type(model_config).__name__}"
            )

        if policy is None:
            stored_policy = AnomalyPipelinePolicy()
        elif isinstance(policy, AnomalyPipelinePolicy):
            stored_policy = policy.model_copy(deep=True)
        else:
            raise TypeError(
                "policy must be AnomalyPipelinePolicy or None, "
                f"got {type(policy).__name__}"
            )

        self._model_config = stored_config
        self._policy = stored_policy

    def run(
        self,
        split: DatasetSplit,
        *,
        feature_columns: Sequence[str],
        leakage_report: LeakageReport,
    ) -> AnomalyPipelineOutcome:
        """Fit on train and score enabled partitions without mutating inputs.

        Args:
            split: Train/validation/test partitions with summary metadata.
            feature_columns: Feature names used for fit and scoring, in order.
            leakage_report: Structural leakage report aligned to features.

        Returns:
            Outcome with the fitted model, per-partition scored frames,
            combined scored frame, and aggregate report.

        Raises:
            TypeError: If inputs have invalid types.
            DataValidationError: If structural validation fails.
            DataLeakageError: If blockers are present under a safe-leakage
                policy, or partition ``_original_row_id`` sets overlap.
            InsufficientDataError: If train is empty or no enabled partition
                has rows to score.
            ProcessIntelligenceError: If fit/detect results are inconsistent.
        """
        validated_features = _validate_feature_columns(feature_columns)
        validated_split = _validate_split(split, feature_columns=validated_features)
        _validate_leakage_report(
            leakage_report,
            feature_columns=validated_features,
            require_safe=self._policy.require_safe_leakage_report,
        )
        _validate_original_row_ids(validated_split)
        if self._policy.require_split_summary_match:
            _validate_split_summary_match(validated_split)
        _validate_reserved_result_columns(validated_split)
        _validate_minimum_data(validated_split, policy=self._policy)

        model = create_isolation_forest_anomaly_model(
            config=self._model_config.model_copy(deep=True)
        )
        if model.is_fitted:
            raise ProcessIntelligenceError(
                "factory model must be unfitted before pipeline fit"
            )

        x_train = validated_split.train.select(validated_features)
        fit_started = time.perf_counter()
        model.fit(x_train)
        fit_seconds = time.perf_counter() - fit_started

        if not model.is_fitted:
            raise ProcessIntelligenceError(
                "IsolationForestAnomalyModel must be fitted after fit()"
            )
        if list(model.feature_names) != list(validated_features):
            raise ProcessIntelligenceError(
                "fitted model feature_names must match feature_columns order: "
                f"expected {list(validated_features)}, "
                f"got {list(model.feature_names)}"
            )
        if model.fit_row_count != validated_split.train.height:
            raise ProcessIntelligenceError(
                "fitted model fit_row_count must equal train height "
                f"({validated_split.train.height}), got {model.fit_row_count}"
            )

        score_flags = {
            "train": self._policy.score_train_partition,
            "validation": self._policy.score_validation_partition,
            "test": self._policy.score_test_partition,
        }
        partitions = {
            "train": validated_split.train,
            "validation": validated_split.validation,
            "test": validated_split.test,
        }

        scored_frames: dict[str, pl.DataFrame] = {}
        summaries: list[AnomalyPartitionSummary] = []
        detect_warnings: list[str] = []
        total_scoring_seconds = 0.0

        for name in _PARTITION_NAMES:
            frame = partitions[name]
            enabled = score_flags[name]
            scored, summary, partition_detect_warnings = _score_partition(
                fitted_model=model,
                partition_name=name,
                partition=frame,
                feature_columns=validated_features,
                enabled=enabled,
            )
            scored_frames[name] = scored
            summaries.append(summary)
            total_scoring_seconds += summary.scoring_seconds
            detect_warnings.extend(partition_detect_warnings)

        train_scored = scored_frames["train"]
        validation_scored = scored_frames["validation"]
        test_scored = scored_frames["test"]
        combined_scored = _build_combined_scored(
            train_scored=train_scored,
            validation_scored=validation_scored,
            test_scored=test_scored,
            sort_by_original_row_id=self._policy.sort_combined_by_original_row_id,
        )

        metadata = model.get_metadata()
        _validate_model_metadata(
            metadata=metadata,
            feature_columns=validated_features,
            fit_row_count=validated_split.train.height,
        )

        total_scored_row_count = sum(item.scored_row_count for item in summaries)
        total_anomaly_count = sum(item.anomaly_count for item in summaries)
        if total_scored_row_count == 0:
            total_anomaly_fraction = 0.0
        else:
            total_anomaly_fraction = (
                float(total_anomaly_count) / float(total_scored_row_count)
            )

        report_warnings = _build_report_warnings(
            leakage_report=leakage_report,
            summaries=summaries,
            detect_warnings=detect_warnings,
            metadata=metadata,
            total_scored_row_count=total_scored_row_count,
            total_anomaly_count=total_anomaly_count,
        )

        report = AnomalyPipelineReport(
            task=AnalysisTask.UNSUPERVISED_ANOMALY,
            model_name=str(metadata.model_name),
            estimator_key=str(getattr(metadata, "estimator_key", "")),
            feature_columns=list(validated_features),
            fit_partition="train",
            fit_row_count=int(validated_split.train.height),
            partition_summaries=list(summaries),
            total_scored_row_count=total_scored_row_count,
            total_anomaly_count=total_anomaly_count,
            total_anomaly_fraction=total_anomaly_fraction,
            fit_seconds=float(fit_seconds),
            total_scoring_seconds=float(total_scoring_seconds),
            total_seconds=float(fit_seconds + total_scoring_seconds),
            combined_sorted_by_original_row_id=(
                self._policy.sort_combined_by_original_row_id
            ),
            created_at=datetime.now(UTC),
            warnings=report_warnings,
        )

        outcome = AnomalyPipelineOutcome(
            fitted_model=model,
            train_scored=train_scored,
            validation_scored=validation_scored,
            test_scored=test_scored,
            combined_scored=combined_scored,
            report=report,
        )
        _validate_outcome(outcome)
        return outcome


def _validate_feature_columns(feature_columns: object) -> list[str]:
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
        raise DataValidationError(
            "feature_columns must contain at least one feature"
        )
    return validated


def _validate_split(
    split: object,
    *,
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

    missing_features = [
        name for name in feature_columns if name not in train_columns
    ]
    if missing_features:
        raise DataValidationError(
            "feature_columns missing from split partitions: "
            f"{', '.join(missing_features)}"
        )
    return split


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
            "Data leakage blockers detected during anomaly pipeline: "
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
    """Validate per-partition IDs and cross-partition disjointness."""
    train_ids = _validate_original_row_id_column(split.train, partition_name="train")
    validation_ids = _validate_original_row_id_column(
        split.validation, partition_name="validation"
    )
    test_ids = _validate_original_row_id_column(split.test, partition_name="test")

    train_set = set(train_ids)
    validation_set = set(validation_ids)
    test_set = set(test_ids)

    overlaps: list[str] = []
    if train_set & validation_set:
        overlaps.append("train/validation")
    if train_set & test_set:
        overlaps.append("train/test")
    if validation_set & test_set:
        overlaps.append("validation/test")
    if overlaps:
        raise DataLeakageError(
            "Overlapping `_original_row_id` values detected between "
            f"partitions: {', '.join(overlaps)}"
        )

    total_rows = split.train.height + split.validation.height + split.test.height
    union_size = len(train_set | validation_set | test_set)
    if union_size != total_rows:
        raise DataValidationError(
            "union of partition `_original_row_id` values must equal the "
            f"total row count ({total_rows}), got union size {union_size}"
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
    """Reject partitions that already contain anomaly result columns."""
    present = [
        name
        for name in _RESERVED_RESULT_COLUMNS
        if name in split.train.columns
    ]
    if present:
        raise DataValidationError(
            "input partitions already contain reserved anomaly result "
            f"columns: {', '.join(present)}"
        )


def _validate_minimum_data(
    split: DatasetSplit,
    *,
    policy: AnomalyPipelinePolicy,
) -> None:
    """Enforce non-empty train and at least one scorables row when enabled."""
    if split.train.height < 1:
        raise InsufficientDataError(
            "train partition must contain at least one row for anomaly fitting"
        )

    enabled_heights = []
    if policy.score_train_partition:
        enabled_heights.append(split.train.height)
    if policy.score_validation_partition:
        enabled_heights.append(split.validation.height)
    if policy.score_test_partition:
        enabled_heights.append(split.test.height)
    if enabled_heights and sum(enabled_heights) == 0:
        raise InsufficientDataError(
            "all scoring-enabled partitions are empty; no rows available to score"
        )


def _empty_anomaly_result() -> AnomalyDetectionResult:
    """Build an empty detection result without calling sklearn methods."""
    return AnomalyDetectionResult(
        scores=[],
        is_anomaly=[],
        raw_predictions=[],
        threshold=0.0,
        row_count=0,
        anomaly_count=0,
        anomaly_fraction=0.0,
        score_min=None,
        score_max=None,
        score_mean=None,
        warnings=[],
    )


def _validate_detection_result(
    result: object,
    *,
    expected_row_count: int,
) -> AnomalyDetectionResult:
    """Validate ``AnomalyDetectionResult`` consistency against partition size."""
    if not isinstance(result, AnomalyDetectionResult):
        raise ProcessIntelligenceError(
            "detect must return AnomalyDetectionResult, "
            f"got {type(result).__name__}"
        )
    if result.row_count != expected_row_count:
        raise ProcessIntelligenceError(
            "AnomalyDetectionResult.row_count must equal partition height "
            f"({expected_row_count}), got {result.row_count}"
        )
    if len(result.scores) != expected_row_count:
        raise ProcessIntelligenceError(
            "AnomalyDetectionResult.scores length must equal partition height "
            f"({expected_row_count}), got {len(result.scores)}"
        )
    if len(result.is_anomaly) != expected_row_count:
        raise ProcessIntelligenceError(
            "AnomalyDetectionResult.is_anomaly length must equal partition "
            f"height ({expected_row_count}), got {len(result.is_anomaly)}"
        )
    if len(result.raw_predictions) != expected_row_count:
        raise ProcessIntelligenceError(
            "AnomalyDetectionResult.raw_predictions length must equal "
            f"partition height ({expected_row_count}), "
            f"got {len(result.raw_predictions)}"
        )
    if not _is_finite_number(result.threshold):
        raise ProcessIntelligenceError(
            f"AnomalyDetectionResult.threshold must be finite, "
            f"got {result.threshold!r}"
        )
    for index, score in enumerate(result.scores):
        if not _is_finite_number(score):
            raise ProcessIntelligenceError(
                f"AnomalyDetectionResult.scores[{index}] must be finite, "
                f"got {score!r}"
            )
    for index, raw in enumerate(result.raw_predictions):
        if raw not in (-1, 1):
            raise ProcessIntelligenceError(
                f"AnomalyDetectionResult.raw_predictions[{index}] must be "
                f"-1 or 1, got {raw!r}"
            )
        expected_flag = raw == -1
        if bool(result.is_anomaly[index]) is not expected_flag:
            raise ProcessIntelligenceError(
                "AnomalyDetectionResult.is_anomaly must match "
                f"raw_predictions == -1 at index {index}"
            )
    true_count = sum(1 for flag in result.is_anomaly if flag)
    if true_count != result.anomaly_count:
        raise ProcessIntelligenceError(
            "AnomalyDetectionResult.anomaly_count must equal True count in "
            f"is_anomaly ({true_count}), got {result.anomaly_count}"
        )
    if expected_row_count == 0:
        if result.anomaly_fraction != 0.0:
            raise ProcessIntelligenceError(
                "AnomalyDetectionResult.anomaly_fraction must be 0.0 for "
                "empty partitions"
            )
    else:
        expected_fraction = result.anomaly_count / expected_row_count
        if not _fractions_match(result.anomaly_fraction, expected_fraction):
            raise ProcessIntelligenceError(
                "AnomalyDetectionResult.anomaly_fraction must equal "
                f"anomaly_count / row_count ({expected_fraction}), "
                f"got {result.anomaly_fraction}"
            )
    return result


def _build_scored_frame(
    partition: pl.DataFrame,
    *,
    partition_name: Literal["train", "validation", "test"],
    result: AnomalyDetectionResult | None,
    empty_rows: bool,
) -> pl.DataFrame:
    """Append anomaly result columns to a partition frame copy."""
    base = partition.clear() if empty_rows else partition
    height = base.height
    if result is None:
        scores: list[float] = []
        flags: list[bool] = []
        raws: list[int] = []
    else:
        scores = list(result.scores)
        flags = list(result.is_anomaly)
        raws = list(result.raw_predictions)
        if len(scores) != height or len(flags) != height or len(raws) != height:
            raise ProcessIntelligenceError(
                "anomaly result lengths must match scored frame height "
                f"({height})"
            )

    scored = base.with_columns(
        [
            pl.Series(_SCORE_COLUMN, scores, dtype=pl.Float64),
            pl.Series(_IS_ANOMALY_COLUMN, flags, dtype=pl.Boolean),
            pl.Series(_RAW_PREDICTION_COLUMN, raws, dtype=pl.Int64),
            pl.Series(
                _DATA_PARTITION_COLUMN,
                [partition_name] * height,
                dtype=pl.String,
            ),
        ]
    )
    return scored


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
            f"then anomaly result columns; expected {expected_columns}, "
            f"got {list(scored.columns)}"
        )


def _score_partition(
    *,
    fitted_model: IsolationForestAnomalyModel,
    partition_name: Literal["train", "validation", "test"],
    partition: pl.DataFrame,
    feature_columns: Sequence[str],
    enabled: bool,
) -> tuple[pl.DataFrame, AnomalyPartitionSummary, list[str]]:
    """Score one partition or return an empty disabled/empty scored frame."""
    input_row_count = int(partition.height)
    detect_warnings: list[str] = []

    if not enabled:
        scored = _build_scored_frame(
            partition,
            partition_name=partition_name,
            result=None,
            empty_rows=True,
        )
        summary = AnomalyPartitionSummary(
            partition=partition_name,
            input_row_count=input_row_count,
            scored=False,
            scored_row_count=0,
            anomaly_count=0,
            anomaly_fraction=0.0,
            score_min=None,
            score_max=None,
            score_mean=None,
            scoring_seconds=0.0,
            warnings=[_PARTITION_DISABLED_WARNING],
        )
        return scored, summary, detect_warnings

    started = time.perf_counter()
    if input_row_count == 0:
        result = _empty_anomaly_result()
        result = _validate_detection_result(result, expected_row_count=0)
        scored = _build_scored_frame(
            partition,
            partition_name=partition_name,
            result=result,
            empty_rows=True,
        )
        scoring_seconds = time.perf_counter() - started
        summary = AnomalyPartitionSummary(
            partition=partition_name,
            input_row_count=0,
            scored=True,
            scored_row_count=0,
            anomaly_count=0,
            anomaly_fraction=0.0,
            score_min=None,
            score_max=None,
            score_mean=None,
            scoring_seconds=float(scoring_seconds),
            warnings=[_PARTITION_EMPTY_SCORED_WARNING],
        )
        return scored, summary, detect_warnings

    x_partition = partition.select(list(feature_columns))
    raw_result = fitted_model.detect(x_partition)
    result = _validate_detection_result(
        raw_result, expected_row_count=input_row_count
    )
    detect_warnings.extend(list(result.warnings))
    scored = _build_scored_frame(
        partition,
        partition_name=partition_name,
        result=result,
        empty_rows=False,
    )
    _assert_scored_alignment(
        partition=partition,
        scored=scored,
        partition_name=partition_name,
    )
    scoring_seconds = time.perf_counter() - started
    summary = AnomalyPartitionSummary(
        partition=partition_name,
        input_row_count=input_row_count,
        scored=True,
        scored_row_count=input_row_count,
        anomaly_count=int(result.anomaly_count),
        anomaly_fraction=float(result.anomaly_fraction),
        score_min=result.score_min,
        score_max=result.score_max,
        score_mean=result.score_mean,
        scoring_seconds=float(scoring_seconds),
        warnings=[],
    )
    return scored, summary, detect_warnings


def _build_combined_scored(
    *,
    train_scored: pl.DataFrame,
    validation_scored: pl.DataFrame,
    test_scored: pl.DataFrame,
    sort_by_original_row_id: bool,
) -> pl.DataFrame:
    """Concatenate partition scored frames and optionally restore row order."""
    frames = [train_scored, validation_scored, test_scored]
    schemas = [list(frame.schema.items()) for frame in frames]
    if not all(schema == schemas[0] for schema in schemas):
        raise ProcessIntelligenceError(
            "all scored partition frames must share an identical schema"
        )
    combined = pl.concat(frames, how="vertical")
    if sort_by_original_row_id:
        combined = combined.sort(
            ORIGINAL_ROW_ID_COLUMN,
            maintain_order=True,
        )
    ids = _partition_row_ids(combined)
    if len(ids) != len(set(ids)):
        raise ProcessIntelligenceError(
            "combined_scored `_original_row_id` values must be unique"
        )
    return combined


def _validate_model_metadata(
    *,
    metadata: object,
    feature_columns: Sequence[str],
    fit_row_count: int,
) -> None:
    """Validate fitted-model metadata against the pipeline contract."""
    model_name = getattr(metadata, "model_name", None)
    estimator_key = getattr(metadata, "estimator_key", None)
    task = getattr(metadata, "task", None)
    features = getattr(metadata, "features", None)
    metadata_fit_rows = getattr(metadata, "fit_row_count", None)

    if not isinstance(model_name, str) or model_name.strip() == "":
        raise ProcessIntelligenceError(
            "model metadata model_name must be a non-empty string"
        )
    if not isinstance(estimator_key, str) or estimator_key.strip() == "":
        raise ProcessIntelligenceError(
            "model metadata estimator_key must be a non-empty string"
        )
    if task is not AnalysisTask.UNSUPERVISED_ANOMALY:
        raise ProcessIntelligenceError(
            "model metadata task must be UNSUPERVISED_ANOMALY, "
            f"got {task!r}"
        )
    if not isinstance(features, list):
        raise ProcessIntelligenceError(
            "model metadata features must be a list of feature names"
        )
    if list(features) != list(feature_columns):
        raise ProcessIntelligenceError(
            "model metadata features must match feature_columns order: "
            f"expected {list(feature_columns)}, got {list(features)}"
        )
    if metadata_fit_rows != fit_row_count:
        raise ProcessIntelligenceError(
            "model metadata fit_row_count must equal train height "
            f"({fit_row_count}), got {metadata_fit_rows}"
        )


def _build_report_warnings(
    *,
    leakage_report: LeakageReport,
    summaries: Sequence[AnomalyPartitionSummary],
    detect_warnings: Sequence[str],
    metadata: object,
    total_scored_row_count: int,
    total_anomaly_count: int,
) -> list[str]:
    """Build deterministic, de-duplicated report warnings."""
    warnings: list[str] = []

    if any(
        issue.severity is LeakageSeverity.WARNING
        for issue in leakage_report.issues
    ):
        warnings.append(_WARNING_LEAKAGE)

    if any(not item.scored for item in summaries):
        warnings.append(_WARNING_SCORING_DISABLED)

    empty_enabled = any(
        item.scored
        and item.input_row_count == 0
        and item.partition in ("validation", "test")
        for item in summaries
    )
    if empty_enabled:
        warnings.append(_WARNING_EMPTY_SCORED)

    metadata_warnings = getattr(metadata, "warnings", None)
    if isinstance(metadata_warnings, list):
        for warning in metadata_warnings:
            if isinstance(warning, str) and warning:
                warnings.append(warning)
    for warning in detect_warnings:
        if warning:
            warnings.append(warning)

    if total_scored_row_count > 0 and total_anomaly_count == 0:
        warnings.append(_WARNING_ZERO_ANOMALIES)
    if (
        total_scored_row_count > 0
        and total_anomaly_count == total_scored_row_count
    ):
        warnings.append(_WARNING_ALL_ANOMALIES)

    return _dedupe_preserve_order(warnings)


def _validate_outcome(outcome: AnomalyPipelineOutcome) -> None:
    """Final consistency checks before returning a pipeline outcome."""
    if not isinstance(outcome.fitted_model, IsolationForestAnomalyModel):
        raise ProcessIntelligenceError(
            "outcome.fitted_model must be IsolationForestAnomalyModel"
        )
    if not outcome.fitted_model.is_fitted:
        raise ProcessIntelligenceError(
            "outcome.fitted_model must be fitted"
        )

    frames = (
        outcome.train_scored,
        outcome.validation_scored,
        outcome.test_scored,
        outcome.combined_scored,
    )
    for frame in frames:
        if not isinstance(frame, pl.DataFrame):
            raise ProcessIntelligenceError(
                "all scored results must be polars.DataFrame instances"
            )

    for name, frame in (
        ("train", outcome.train_scored),
        ("validation", outcome.validation_scored),
        ("test", outcome.test_scored),
    ):
        if len(frame.columns) < 4:
            raise ProcessIntelligenceError(
                f"{name} scored frame is missing anomaly result columns"
            )
        if list(frame.columns[-4:]) != list(_RESERVED_RESULT_COLUMNS):
            raise ProcessIntelligenceError(
                f"{name} scored frame must end with anomaly result columns "
                f"{list(_RESERVED_RESULT_COLUMNS)}, "
                f"got {list(frame.columns[-4:])}"
            )

    if outcome.combined_scored.height != outcome.report.total_scored_row_count:
        raise ProcessIntelligenceError(
            "combined_scored height must equal report.total_scored_row_count "
            f"({outcome.report.total_scored_row_count}), "
            f"got {outcome.combined_scored.height}"
        )
    if outcome.combined_scored.height > 0:
        anomaly_true = int(
            outcome.combined_scored.get_column(_IS_ANOMALY_COLUMN).sum()
        )
    else:
        anomaly_true = 0
    if anomaly_true != outcome.report.total_anomaly_count:
        raise ProcessIntelligenceError(
            "combined anomaly True count must equal "
            f"report.total_anomaly_count ({outcome.report.total_anomaly_count}), "
            f"got {anomaly_true}"
        )

    combined_ids = _partition_row_ids(outcome.combined_scored)
    if len(combined_ids) != len(set(combined_ids)):
        raise ProcessIntelligenceError(
            "combined_scored `_original_row_id` values must be unique"
        )

    metadata = outcome.fitted_model.get_metadata()
    if outcome.report.model_name != metadata.model_name:
        raise ProcessIntelligenceError(
            "report.model_name must match fitted model metadata"
        )
    if outcome.report.estimator_key != getattr(metadata, "estimator_key", None):
        raise ProcessIntelligenceError(
            "report.estimator_key must match fitted model metadata"
        )
    if outcome.report.task is not metadata.task:
        raise ProcessIntelligenceError(
            "report.task must match fitted model metadata"
        )
