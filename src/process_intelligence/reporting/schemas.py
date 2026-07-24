"""UI/API-safe presentation DTOs for analysis workflow reports (Step 11A).

Transforms backend ``AnalysisWorkflowReport`` contracts into JSON-serializable
view models. Does not embed DataFrames, estimators, model objects, file paths,
or tracebacks, and does not recompute metrics or recommendations.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.evaluation.performance_acceptance import (
    MetricAcceptanceDirection,
    ModelPerformanceAcceptanceStatus,
)
from process_intelligence.recommendation.enums import (
    RecommendationObjective,
    RecommendationSafetyStatus,
    RecommendationStatus,
    WhatIfPerturbationDirection,
    WhatIfStabilityClassification,
    WhatIfVerificationScenarioType,
    WhatIfVerificationStatus,
)
from process_intelligence.recommendation.target_domain import (
    RecommendationTargetPlausibility,
)
from process_intelligence.workflow.dataset_fingerprint import (
    normalize_optional_dataset_fingerprint,
)
from process_intelligence.workflow.enums import (
    AnalysisWorkflowStage,
    AnalysisWorkflowStatus,
    AnomalyContextOrderBasis,
)

ScalarMetadataValue = str | int | float | bool | None
"""Allowed scalar types for presentation metadata dictionaries."""

_FORBIDDEN_PRESENTATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"guaranteed", re.IGNORECASE),
    re.compile(r"proven\s+optimal", re.IGNORECASE),
    re.compile(r"will\s+improve", re.IGNORECASE),
    re.compile(r"will\s+fix", re.IGNORECASE),
    re.compile(r"confirmed\s+root\s+cause", re.IGNORECASE),
    re.compile(r"반드시\s*개선"),
    re.compile(r"최적\s*조건\s*확정"),
)

_EXPECTED_HEADLINES: dict[AnalysisWorkflowStatus, str] = {
    AnalysisWorkflowStatus.COMPLETED: (
        "Analysis completed with a generated recommendation"
    ),
    AnalysisWorkflowStatus.PARTIAL: (
        "Analysis completed without an executable recommendation"
    ),
    AnalysisWorkflowStatus.REFUSED: (
        "Analysis was stopped by a safety or validation gate"
    ),
}

_ANOMALY_SCORE_DIRECTION = "higher_is_more_anomalous"


def _require_non_empty_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be str, got {type(value).__name__}")
    if value == "" or value.strip() == "":
        raise ValueError(f"{field_name} must be a non-empty, non-whitespace string")
    return value


def _require_optional_non_empty_str(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_non_empty_str(value, field_name=field_name)


def _require_strict_bool(value: object, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(
            f"{field_name} must be a bool (0/1 and strings rejected), "
            f"got {type(value).__name__}"
        )
    return value


def _require_strict_int_ge(
    value: object,
    *,
    field_name: str,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{field_name} must be an int >= {minimum} "
            f"(bool not allowed), got {type(value).__name__}"
        )
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {value}")
    return value


def _require_strict_int_ge0(value: object, *, field_name: str) -> int:
    return _require_strict_int_ge(value, field_name=field_name, minimum=0)


def _require_finite_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a finite float "
            f"(bool not allowed), got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return number


def _require_optional_finite_float(
    value: object,
    *,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    return _require_finite_float(value, field_name=field_name)


def _require_non_negative_finite_float(value: object, *, field_name: str) -> float:
    number = _require_finite_float(value, field_name=field_name)
    if number < 0.0:
        raise ValueError(f"{field_name} must be >= 0, got {number}")
    return number


def _require_confidence(value: object, *, field_name: str = "confidence") -> float:
    number = _require_finite_float(value, field_name=field_name)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{field_name} must be in [0.0, 1.0], got {number}")
    return number


def _require_timezone_aware(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _validate_scalar_metadata(
    value: object,
    *,
    field_name: str = "metadata",
) -> dict[str, ScalarMetadataValue]:
    if not isinstance(value, dict):
        raise ValueError(
            f"{field_name} must be a dict[str, scalar], got {type(value).__name__}"
        )
    cleaned: dict[str, ScalarMetadataValue] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or key == "" or key.strip() == "":
            raise ValueError(f"{field_name} keys must be non-empty strings")
        if raw is None or isinstance(raw, (str, bool)):
            cleaned[key] = raw
            continue
        if isinstance(raw, int) and not isinstance(raw, bool):
            cleaned[key] = raw
            continue
        if isinstance(raw, float):
            if not math.isfinite(raw):
                raise ValueError(
                    f"{field_name}[{key!r}] float must be finite, got {raw!r}"
                )
            cleaned[key] = raw
            continue
        raise ValueError(
            f"{field_name}[{key!r}] must be str, int, float, bool, or None "
            f"(no DataFrame, ndarray, estimator, or nested objects); "
            f"got {type(raw).__name__}"
        )
    return cleaned


def _validate_unique_non_empty_strings(
    values: list[str],
    *,
    field_name: str,
) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _require_non_empty_str(item, field_name=field_name)
        if text in seen:
            raise ValueError(f"{field_name} must not contain duplicates: {text!r}")
        seen.add(text)
        cleaned.append(text)
    return cleaned


def _reject_forbidden_presentation_text(value: str, *, field_name: str) -> str:
    for pattern in _FORBIDDEN_PRESENTATION_PATTERNS:
        if pattern.search(value) is not None:
            raise ValueError(
                f"{field_name} must not use causal or guaranteed-outcome language"
            )
    return value


def stage_status_label(
    *,
    executed: bool,
    succeeded: bool,
    structured_refusal: bool,
) -> str:
    """Derive the presentation status label for one workflow stage record."""
    if not executed:
        return "SKIPPED"
    if structured_refusal:
        return "REFUSED"
    if succeeded:
        return "SUCCEEDED"
    return "INCOMPLETE"


class WorkflowOverviewView(BaseModel):
    """High-level presentation summary for one analysis workflow run."""

    model_config = ConfigDict(extra="forbid")

    status: AnalysisWorkflowStatus
    terminal_stage: AnalysisWorkflowStage
    headline: str
    summary: str
    started_at: datetime
    completed_at: datetime
    total_seconds: float
    recommendation_status: RecommendationStatus | None = None

    @field_validator("status", mode="before")
    @classmethod
    def _validate_status(cls, value: object) -> AnalysisWorkflowStatus:
        if isinstance(value, AnalysisWorkflowStatus):
            return value
        if isinstance(value, str):
            try:
                return AnalysisWorkflowStatus(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid AnalysisWorkflowStatus: {value!r}"
                ) from exc
        raise ValueError(
            f"status must be AnalysisWorkflowStatus, got {type(value).__name__}"
        )

    @field_validator("terminal_stage", mode="before")
    @classmethod
    def _validate_terminal_stage(cls, value: object) -> AnalysisWorkflowStage:
        if isinstance(value, AnalysisWorkflowStage):
            return value
        if isinstance(value, str):
            try:
                return AnalysisWorkflowStage(value)
            except ValueError as exc:
                raise ValueError(f"invalid AnalysisWorkflowStage: {value!r}") from exc
        raise ValueError(
            f"terminal_stage must be AnalysisWorkflowStage, got {type(value).__name__}"
        )

    @field_validator("headline", "summary", mode="before")
    @classmethod
    def _validate_text_fields(cls, value: object) -> str:
        text = _require_non_empty_str(value, field_name="overview text field")
        return _reject_forbidden_presentation_text(text, field_name="overview text field")

    @field_validator("started_at", "completed_at", mode="after")
    @classmethod
    def _validate_datetimes(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value, field_name="datetime field")

    @field_validator("total_seconds", mode="before")
    @classmethod
    def _validate_total_seconds(cls, value: object) -> float:
        return _require_non_negative_finite_float(value, field_name="total_seconds")

    @field_validator("recommendation_status", mode="before")
    @classmethod
    def _validate_recommendation_status(
        cls,
        value: object,
    ) -> RecommendationStatus | None:
        if value is None:
            return None
        if isinstance(value, RecommendationStatus):
            return value
        if isinstance(value, str):
            try:
                return RecommendationStatus(value)
            except ValueError as exc:
                raise ValueError(f"invalid RecommendationStatus: {value!r}") from exc
        raise ValueError(
            "recommendation_status must be RecommendationStatus or None, "
            f"got {type(value).__name__}"
        )

    @model_validator(mode="after")
    def _validate_overview_consistency(self) -> Self:
        expected = _EXPECTED_HEADLINES.get(self.status)
        if expected is not None and self.headline != expected:
            raise ValueError(
                f"headline must match status {self.status.value!r}: expected "
                f"{expected!r}, got {self.headline!r}"
            )
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must be >= started_at")
        return self


class WorkflowDataSummaryView(BaseModel):
    """Row-count and identity summary for one workflow presentation report."""

    model_config = ConfigDict(extra="forbid")

    raw_row_count: int
    processed_row_count: int
    cohort_row_count: int
    train_row_count: int
    validation_row_count: int
    test_row_count: int
    anomaly_event_count: int
    diagnosis_factor_count: int
    selected_operating_row_id: int | str | None = None
    row_identity_preserved: bool

    @field_validator(
        "raw_row_count",
        "processed_row_count",
        "cohort_row_count",
        "train_row_count",
        "validation_row_count",
        "test_row_count",
        "anomaly_event_count",
        "diagnosis_factor_count",
        mode="before",
    )
    @classmethod
    def _validate_counts(cls, value: object) -> int:
        return _require_strict_int_ge0(value, field_name="count field")

    @field_validator("selected_operating_row_id", mode="before")
    @classmethod
    def _validate_operating_row_id(cls, value: object) -> int | str | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("selected_operating_row_id must not be a bool")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return _require_non_empty_str(value, field_name="selected_operating_row_id")
        raise ValueError(
            "selected_operating_row_id must be int, str, or None, "
            f"got {type(value).__name__}"
        )

    @field_validator("row_identity_preserved", mode="before")
    @classmethod
    def _validate_row_identity_preserved(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="row_identity_preserved")

    @model_validator(mode="after")
    def _validate_count_relationships(self) -> Self:
        if self.processed_row_count > self.raw_row_count:
            raise ValueError(
                "processed_row_count must be <= raw_row_count "
                f"(got {self.processed_row_count} > {self.raw_row_count})"
            )
        if self.cohort_row_count > self.processed_row_count:
            raise ValueError(
                "cohort_row_count must be <= processed_row_count "
                f"(got {self.cohort_row_count} > {self.processed_row_count})"
            )
        split_total = (
            self.train_row_count + self.validation_row_count + self.test_row_count
        )
        if split_total > 0 and split_total != self.cohort_row_count:
            raise ValueError(
                "train_row_count + validation_row_count + test_row_count "
                "must equal cohort_row_count when split counts are present "
                f"(got {split_total} != {self.cohort_row_count})"
            )
        return self


class WorkflowCohortFilterSummaryView(BaseModel):
    """Presentation summary for the explicit operating cohort filter."""

    model_config = ConfigDict(extra="forbid")

    configured: bool
    column_name: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    include_lower: bool | None = None
    include_upper: bool | None = None
    exclude_filter_column_from_features: bool | None = None
    source_row_count: int
    retained_row_count: int
    excluded_row_count: int
    null_excluded_count: int
    range_display: str
    filter_column_used_as_feature_display: str

    @field_validator("configured", mode="before")
    @classmethod
    def _validate_configured(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="configured")

    @field_validator("column_name", mode="before")
    @classmethod
    def _validate_column_name(cls, value: object) -> str | None:
        return _require_optional_non_empty_str(value, field_name="column_name")

    @field_validator("lower_bound", "upper_bound", mode="before")
    @classmethod
    def _validate_optional_bounds(cls, value: object) -> float | None:
        return _require_optional_finite_float(value, field_name="cohort filter bound")

    @field_validator(
        "include_lower",
        "include_upper",
        "exclude_filter_column_from_features",
        mode="before",
    )
    @classmethod
    def _validate_optional_bools(cls, value: object) -> bool | None:
        if value is None:
            return None
        return _require_strict_bool(value, field_name="cohort filter bool field")

    @field_validator(
        "source_row_count",
        "retained_row_count",
        "excluded_row_count",
        "null_excluded_count",
        mode="before",
    )
    @classmethod
    def _validate_counts(cls, value: object) -> int:
        return _require_strict_int_ge0(value, field_name="cohort filter count")

    @field_validator(
        "range_display",
        "filter_column_used_as_feature_display",
        mode="before",
    )
    @classmethod
    def _validate_display_text(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="cohort filter display text")

    @model_validator(mode="after")
    def _validate_summary_consistency(self) -> Self:
        if self.retained_row_count + self.excluded_row_count != self.source_row_count:
            raise ValueError(
                "retained_row_count + excluded_row_count must equal source_row_count"
            )
        if self.null_excluded_count > self.excluded_row_count:
            raise ValueError(
                "null_excluded_count must be <= excluded_row_count"
            )
        return self


class WorkflowRoutingSummaryView(BaseModel):
    """Industry, task, and feature routing summary for presentation."""

    model_config = ConfigDict(extra="forbid")

    selected_industry: str | None = None
    selected_task: AnalysisTask | None = None
    inferred_task: AnalysisTask | None = None
    task_selection_source: str | None = None
    task_override_applied: bool = False
    target_column: str | None = None
    feature_count: int | None = None
    target_suitable: bool | None = None
    target_unique_non_null_count: int | None = None
    target_refusal_code: str | None = None
    target_suitability_message: str | None = None

    @field_validator("selected_industry", "target_column", mode="before")
    @classmethod
    def _validate_optional_strings(cls, value: object) -> str | None:
        return _require_optional_non_empty_str(value, field_name="optional string")

    @field_validator("selected_task", "inferred_task", mode="before")
    @classmethod
    def _validate_task_fields(cls, value: object) -> AnalysisTask | None:
        if value is None:
            return None
        if isinstance(value, AnalysisTask):
            return value
        if isinstance(value, str):
            try:
                return AnalysisTask(value)
            except ValueError as exc:
                raise ValueError(f"invalid AnalysisTask: {value!r}") from exc
        raise ValueError(
            f"task field must be AnalysisTask or None, got {type(value).__name__}"
        )

    @field_validator("task_selection_source", mode="before")
    @classmethod
    def _validate_task_selection_source(cls, value: object) -> str | None:
        return _require_optional_non_empty_str(
            value,
            field_name="task_selection_source",
        )

    @field_validator("task_override_applied", mode="before")
    @classmethod
    def _validate_task_override_applied(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="task_override_applied")

    @field_validator("feature_count", "target_unique_non_null_count", mode="before")
    @classmethod
    def _validate_feature_count(cls, value: object) -> int | None:
        if value is None:
            return None
        return _require_strict_int_ge0(value, field_name="count")

    @field_validator("target_suitable", mode="before")
    @classmethod
    def _validate_target_suitable(cls, value: object) -> bool | None:
        if value is None:
            return None
        return _require_strict_bool(value, field_name="target_suitable")

    @field_validator(
        "target_refusal_code",
        "target_suitability_message",
        mode="before",
    )
    @classmethod
    def _validate_target_suitability_text(cls, value: object) -> str | None:
        return _require_optional_non_empty_str(
            value,
            field_name="target suitability text",
        )


class WorkflowModelSummaryView(BaseModel):
    """Selected model keys and independent-test safety flags for presentation."""

    model_config = ConfigDict(extra="forbid")

    supervised_model_key: str | None = None
    anomaly_model_key: str | None = None
    independent_test_evaluation_performed: bool
    residual_calibration_performed: bool
    test_used_for_model_selection: bool
    test_used_for_threshold_calibration: bool
    anomaly_score_direction: str | None = None

    @field_validator("supervised_model_key", "anomaly_model_key", mode="before")
    @classmethod
    def _validate_optional_model_keys(cls, value: object) -> str | None:
        return _require_optional_non_empty_str(value, field_name="model key")

    @field_validator(
        "independent_test_evaluation_performed",
        "residual_calibration_performed",
        "test_used_for_model_selection",
        "test_used_for_threshold_calibration",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="model summary bool field")

    @field_validator("anomaly_score_direction", mode="before")
    @classmethod
    def _validate_anomaly_score_direction(cls, value: object) -> str | None:
        return _require_optional_non_empty_str(
            value,
            field_name="anomaly_score_direction",
        )

    @model_validator(mode="after")
    def _validate_safety_flags(self) -> Self:
        if self.test_used_for_model_selection:
            raise ValueError(
                "test_used_for_model_selection must be False for presentation reports"
            )
        if self.test_used_for_threshold_calibration:
            raise ValueError(
                "test_used_for_threshold_calibration must be False for "
                "presentation reports"
            )
        if (
            self.anomaly_score_direction is not None
            and self.anomaly_score_direction != _ANOMALY_SCORE_DIRECTION
        ):
            raise ValueError(
                "anomaly_score_direction must be "
                f"{_ANOMALY_SCORE_DIRECTION!r} when present, "
                f"got {self.anomaly_score_direction!r}"
            )
        return self


class PerformanceMetricView(BaseModel):
    """Presentation view of one independent-test metric acceptance result."""

    model_config = ConfigDict(extra="forbid")

    metric_name: str
    observed_value: float | None
    threshold: float
    direction: MetricAcceptanceDirection
    required: bool
    available: bool
    passed: bool | None
    message: str

    @field_validator("metric_name", "message", mode="before")
    @classmethod
    def _validate_non_empty_str(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="string field")

    @field_validator("threshold", mode="before")
    @classmethod
    def _validate_threshold(cls, value: object) -> float:
        return _require_finite_float(value, field_name="threshold")

    @field_validator("observed_value", mode="before")
    @classmethod
    def _validate_observed_value(cls, value: object) -> float | None:
        return _require_optional_finite_float(value, field_name="observed_value")

    @field_validator("direction", mode="before")
    @classmethod
    def _validate_direction(cls, value: object) -> MetricAcceptanceDirection:
        if isinstance(value, MetricAcceptanceDirection):
            return value
        if isinstance(value, str):
            try:
                return MetricAcceptanceDirection(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid MetricAcceptanceDirection: {value!r}"
                ) from exc
        raise ValueError(
            f"direction must be MetricAcceptanceDirection, got {type(value).__name__}"
        )

    @field_validator("required", "available", mode="before")
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="metric bool field")

    @field_validator("passed", mode="before")
    @classmethod
    def _validate_passed(cls, value: object) -> bool | None:
        if value is None:
            return None
        return _require_strict_bool(value, field_name="passed")

    @model_validator(mode="after")
    def _validate_availability_relationship(self) -> Self:
        if not self.available:
            if self.observed_value is not None:
                raise ValueError(
                    "observed_value must be None when available=False"
                )
            if self.passed is not None:
                raise ValueError("passed must be None when available=False")
        else:
            if self.observed_value is None:
                raise ValueError(
                    "observed_value must be a finite float when available=True"
                )
            if self.passed is None:
                raise ValueError("passed must be a bool when available=True")
        return self


class ModelPerformanceView(BaseModel):
    """Presentation view of model-performance acceptance assessment."""

    model_config = ConfigDict(extra="forbid")

    status: ModelPerformanceAcceptanceStatus
    evaluation_available: bool
    independent_test_evaluation: bool
    test_row_count: int
    metrics: list[PerformanceMetricView]
    required_rule_count: int
    passed_required_rule_count: int
    failed_required_rule_count: int
    unavailable_required_rule_count: int
    assessed_at: datetime
    warnings: list[str] = Field(default_factory=list)

    @field_validator("status", mode="before")
    @classmethod
    def _validate_status(cls, value: object) -> ModelPerformanceAcceptanceStatus:
        if isinstance(value, ModelPerformanceAcceptanceStatus):
            return value
        if isinstance(value, str):
            try:
                return ModelPerformanceAcceptanceStatus(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid ModelPerformanceAcceptanceStatus: {value!r}"
                ) from exc
        raise ValueError(
            "status must be ModelPerformanceAcceptanceStatus, "
            f"got {type(value).__name__}"
        )

    @field_validator(
        "evaluation_available",
        "independent_test_evaluation",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="performance bool field")

    @field_validator(
        "test_row_count",
        "required_rule_count",
        "passed_required_rule_count",
        "failed_required_rule_count",
        "unavailable_required_rule_count",
        mode="before",
    )
    @classmethod
    def _validate_counts(cls, value: object) -> int:
        return _require_strict_int_ge0(value, field_name="count field")

    @field_validator("metrics", mode="before")
    @classmethod
    def _validate_metrics_before(cls, value: object) -> list[PerformanceMetricView]:
        if not isinstance(value, list):
            raise ValueError(
                f"metrics must be a list[PerformanceMetricView], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("metrics", mode="after")
    @classmethod
    def _validate_metrics(
        cls,
        value: list[PerformanceMetricView],
    ) -> list[PerformanceMetricView]:
        seen: set[str] = set()
        copied: list[PerformanceMetricView] = []
        for item in value:
            if not isinstance(item, PerformanceMetricView):
                raise ValueError(
                    "metrics entries must be PerformanceMetricView, "
                    f"got {type(item).__name__}"
                )
            if item.metric_name in seen:
                raise ValueError(
                    "metrics must not contain duplicate metric_name: "
                    f"{item.metric_name!r}"
                )
            seen.add(item.metric_name)
            copied.append(item.model_copy(deep=True))
        return copied

    @field_validator("assessed_at", mode="after")
    @classmethod
    def _validate_assessed_at(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value, field_name="assessed_at")

    @field_validator("warnings", mode="before")
    @classmethod
    def _validate_warnings_before(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(f"warnings must be a list[str], got {type(value).__name__}")
        return list(value)

    @field_validator("warnings", mode="after")
    @classmethod
    def _validate_warnings(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(value, field_name="warnings")

    @model_validator(mode="after")
    def _validate_count_consistency(self) -> Self:
        required_results = [item for item in self.metrics if item.required]
        if self.required_rule_count != len(required_results):
            raise ValueError(
                "required_rule_count must equal the number of required "
                f"metrics (got {self.required_rule_count} != {len(required_results)})"
            )
        passed = sum(
            1 for item in required_results if item.available and item.passed is True
        )
        failed = sum(
            1 for item in required_results if item.available and item.passed is False
        )
        unavailable = sum(1 for item in required_results if not item.available)
        if self.passed_required_rule_count != passed:
            raise ValueError(
                "passed_required_rule_count must match required metrics "
                f"(got {self.passed_required_rule_count} != {passed})"
            )
        if self.failed_required_rule_count != failed:
            raise ValueError(
                "failed_required_rule_count must match required metrics "
                f"(got {self.failed_required_rule_count} != {failed})"
            )
        if self.unavailable_required_rule_count != unavailable:
            raise ValueError(
                "unavailable_required_rule_count must match required metrics "
                f"(got {self.unavailable_required_rule_count} != {unavailable})"
            )
        if (
            self.passed_required_rule_count
            + self.failed_required_rule_count
            + self.unavailable_required_rule_count
            != self.required_rule_count
        ):
            raise ValueError(
                "passed + failed + unavailable required counts must equal "
                "required_rule_count"
            )
        return self


class WorkflowStageView(BaseModel):
    """Presentation view of one analysis workflow stage record."""

    model_config = ConfigDict(extra="forbid")

    sequence: int
    stage: AnalysisWorkflowStage
    executed: bool
    succeeded: bool
    structured_refusal: bool
    status_label: str
    row_count: int | None = None
    message: str
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)

    @field_validator("sequence", mode="before")
    @classmethod
    def _validate_sequence(cls, value: object) -> int:
        return _require_strict_int_ge(value, field_name="sequence", minimum=1)

    @field_validator("stage", mode="before")
    @classmethod
    def _validate_stage(cls, value: object) -> AnalysisWorkflowStage:
        if isinstance(value, AnalysisWorkflowStage):
            return value
        if isinstance(value, str):
            try:
                return AnalysisWorkflowStage(value)
            except ValueError as exc:
                raise ValueError(f"invalid AnalysisWorkflowStage: {value!r}") from exc
        raise ValueError(
            f"stage must be AnalysisWorkflowStage, got {type(value).__name__}"
        )

    @field_validator("executed", "succeeded", "structured_refusal", mode="before")
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="stage bool field")

    @field_validator("status_label", "message", mode="before")
    @classmethod
    def _validate_non_empty_str(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="stage string field")

    @field_validator("row_count", mode="before")
    @classmethod
    def _validate_row_count(cls, value: object) -> int | None:
        if value is None:
            return None
        return _require_strict_int_ge0(value, field_name="row_count")

    @field_validator("warnings", mode="before")
    @classmethod
    def _validate_warnings_before(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(f"warnings must be a list[str], got {type(value).__name__}")
        return list(value)

    @field_validator("warnings", mode="after")
    @classmethod
    def _validate_warnings(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(value, field_name="warnings")

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> dict[str, ScalarMetadataValue]:
        if value is None:
            return {}
        return _validate_scalar_metadata(value)

    @model_validator(mode="after")
    def _validate_status_relationships(self) -> Self:
        if not self.executed:
            if self.succeeded:
                raise ValueError("succeeded must be False when executed=False")
            if self.structured_refusal:
                raise ValueError(
                    "structured_refusal must be False when executed=False"
                )
        if self.succeeded and self.structured_refusal:
            raise ValueError(
                "succeeded and structured_refusal cannot both be True"
            )
        expected = stage_status_label(
            executed=self.executed,
            succeeded=self.succeeded,
            structured_refusal=self.structured_refusal,
        )
        if self.status_label != expected:
            raise ValueError(
                f"status_label must be {expected!r} for the given stage flags, "
                f"got {self.status_label!r}"
            )
        return self


class RecommendationChangeView(BaseModel):
    """Presentation view of one proposed controllable-variable change."""

    model_config = ConfigDict(extra="forbid")

    variable: str
    current_value: float
    proposed_value: float
    delta: float
    relative_delta: float | None = None
    rationale: str
    confidence: float
    requires_verification: bool

    @field_validator("variable", "rationale", mode="before")
    @classmethod
    def _validate_non_empty_str(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="change string field")

    @field_validator("current_value", "proposed_value", "delta", mode="before")
    @classmethod
    def _validate_numeric_fields(cls, value: object) -> float:
        return _require_finite_float(value, field_name="numeric change field")

    @field_validator("relative_delta", mode="before")
    @classmethod
    def _validate_relative_delta(cls, value: object) -> float | None:
        return _require_optional_finite_float(value, field_name="relative_delta")

    @field_validator("confidence", mode="before")
    @classmethod
    def _validate_confidence(cls, value: object) -> float:
        return _require_confidence(value, field_name="confidence")

    @field_validator("requires_verification", mode="before")
    @classmethod
    def _validate_requires_verification(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="requires_verification")

    @model_validator(mode="after")
    def _validate_delta_consistency(self) -> Self:
        expected = self.proposed_value - self.current_value
        if not math.isclose(self.delta, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "delta must equal proposed_value - current_value "
                f"(got delta={self.delta}, expected={expected})"
            )
        return self


class RecommendationView(BaseModel):
    """Presentation view of the workflow final recommendation result."""

    model_config = ConfigDict(extra="forbid")

    status: RecommendationStatus
    objective: RecommendationObjective
    changes: list[RecommendationChangeView]
    confidence: float
    baseline_prediction: float | None = None
    proposed_prediction: float | None = None
    baseline_anomaly_score: float | None = None
    proposed_anomaly_score: float | None = None
    extrapolation_flag: bool
    uncertainty_available: bool
    generated_at: datetime
    disclaimer: str
    warnings: list[str] = Field(default_factory=list)
    safety_status: RecommendationSafetyStatus
    safety_messages: list[str] = Field(default_factory=list)
    target_prediction_plausibility: RecommendationTargetPlausibility | None = None

    @field_validator("status", mode="before")
    @classmethod
    def _validate_status(cls, value: object) -> RecommendationStatus:
        if isinstance(value, RecommendationStatus):
            return value
        if isinstance(value, str):
            try:
                return RecommendationStatus(value)
            except ValueError as exc:
                raise ValueError(f"invalid RecommendationStatus: {value!r}") from exc
        raise ValueError(
            f"status must be RecommendationStatus, got {type(value).__name__}"
        )

    @field_validator("objective", mode="before")
    @classmethod
    def _validate_objective(cls, value: object) -> RecommendationObjective:
        if isinstance(value, RecommendationObjective):
            return value
        if isinstance(value, str):
            try:
                return RecommendationObjective(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid RecommendationObjective: {value!r}"
                ) from exc
        raise ValueError(
            f"objective must be RecommendationObjective, got {type(value).__name__}"
        )

    @field_validator("changes", mode="before")
    @classmethod
    def _validate_changes_before(
        cls,
        value: object,
    ) -> list[RecommendationChangeView]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"changes must be a list[RecommendationChangeView], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("changes", mode="after")
    @classmethod
    def _validate_changes(
        cls,
        value: list[RecommendationChangeView],
    ) -> list[RecommendationChangeView]:
        seen: set[str] = set()
        copied: list[RecommendationChangeView] = []
        for item in value:
            if not isinstance(item, RecommendationChangeView):
                raise ValueError(
                    "changes entries must be RecommendationChangeView, "
                    f"got {type(item).__name__}"
                )
            if item.variable in seen:
                raise ValueError(
                    f"changes must not contain duplicate variables: {item.variable!r}"
                )
            seen.add(item.variable)
            copied.append(item.model_copy(deep=True))
        return copied

    @field_validator(
        "baseline_prediction",
        "proposed_prediction",
        "baseline_anomaly_score",
        "proposed_anomaly_score",
        mode="before",
    )
    @classmethod
    def _validate_optional_scores(cls, value: object) -> float | None:
        return _require_optional_finite_float(value, field_name="score/prediction")

    @field_validator("confidence", mode="before")
    @classmethod
    def _validate_confidence(cls, value: object) -> float:
        return _require_confidence(value, field_name="confidence")

    @field_validator("extrapolation_flag", "uncertainty_available", mode="before")
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="recommendation bool field")

    @field_validator("disclaimer", mode="before")
    @classmethod
    def _validate_disclaimer(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="disclaimer")

    @field_validator("generated_at", mode="after")
    @classmethod
    def _validate_generated_at(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value, field_name="generated_at")

    @field_validator("warnings", "safety_messages", mode="before")
    @classmethod
    def _validate_string_lists_before(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(f"string list must be a list[str], got {type(value).__name__}")
        return list(value)

    @field_validator("warnings", "safety_messages", mode="after")
    @classmethod
    def _validate_string_lists(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(value, field_name="string list")

    @field_validator("safety_status", mode="before")
    @classmethod
    def _validate_safety_status(cls, value: object) -> RecommendationSafetyStatus:
        if isinstance(value, RecommendationSafetyStatus):
            return value
        if isinstance(value, str):
            try:
                return RecommendationSafetyStatus(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid RecommendationSafetyStatus: {value!r}"
                ) from exc
        raise ValueError(
            "safety_status must be RecommendationSafetyStatus, "
            f"got {type(value).__name__}"
        )

    @field_validator("target_prediction_plausibility", mode="before")
    @classmethod
    def _validate_target_prediction_plausibility(
        cls,
        value: object,
    ) -> RecommendationTargetPlausibility | None:
        if value is None:
            return None
        if isinstance(value, RecommendationTargetPlausibility):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return RecommendationTargetPlausibility.model_validate(value)
        raise ValueError(
            "target_prediction_plausibility must be "
            "RecommendationTargetPlausibility or None, "
            f"got {type(value).__name__}"
        )

    @model_validator(mode="after")
    def _validate_status_relationships(self) -> Self:
        if self.status is RecommendationStatus.GENERATED:
            if not self.changes:
                raise ValueError("GENERATED requires at least one change")
        elif self.status is RecommendationStatus.READY_FOR_OPTIMIZATION:
            if self.changes:
                raise ValueError("READY_FOR_OPTIMIZATION must have empty changes")
            if self.proposed_prediction is not None:
                raise ValueError(
                    "READY_FOR_OPTIMIZATION must have proposed_prediction=None"
                )
            if self.proposed_anomaly_score is not None:
                raise ValueError(
                    "READY_FOR_OPTIMIZATION must have proposed_anomaly_score=None"
                )
            if self.safety_status is RecommendationSafetyStatus.REFUSED:
                raise ValueError(
                    "READY_FOR_OPTIMIZATION requires APPROVED or CAUTION safety status"
                )
        elif self.status is RecommendationStatus.REFUSED:
            if self.changes:
                raise ValueError("REFUSED must have empty changes")
            if self.proposed_prediction is not None:
                raise ValueError("REFUSED must have proposed_prediction=None")
            if self.proposed_anomaly_score is not None:
                raise ValueError("REFUSED must have proposed_anomaly_score=None")
            if self.safety_status is not RecommendationSafetyStatus.REFUSED:
                raise ValueError("REFUSED requires safety_status REFUSED")
        return self


_STABILITY_MESSAGES: dict[WhatIfStabilityClassification, str] = {
    WhatIfStabilityClassification.STABLE: (
        "All feasible adjacent grid scenarios remained improved relative to "
        "the baseline under the same fitted model."
    ),
    WhatIfStabilityClassification.MIXED: (
        "Improvement was retained for some, but not all, adjacent grid scenarios."
    ),
    WhatIfStabilityClassification.ISOLATED: (
        "Improvement was observed at the proposed grid point, but not at its "
        "feasible immediate neighbors."
    ),
    WhatIfStabilityClassification.NO_NEIGHBORS: (
        "The proposed values were located at grid boundaries, so no adjacent "
        "verification scenarios were available."
    ),
    WhatIfStabilityClassification.NOT_IMPROVING: (
        "The proposed scenario did not improve the configured objective "
        "relative to the baseline during verification."
    ),
    WhatIfStabilityClassification.UNAVAILABLE: (
        "Local what-if verification could not be completed."
    ),
}


def stability_classification_message(
    classification: WhatIfStabilityClassification,
) -> str:
    """Return the deterministic UI message for a stability classification."""
    return _STABILITY_MESSAGES[classification]


class WhatIfVerificationScenarioView(BaseModel):
    """Presentation row for one local what-if verification scenario."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    scenario_type: WhatIfVerificationScenarioType
    perturbed_variable: str | None = None
    perturbation_direction: WhatIfPerturbationDirection | None = None
    perturbed_value: float | None = None
    variable_values: dict[str, float] = Field(default_factory=dict)
    predicted_quality: float | None = None
    anomaly_score: float | None = None
    objective_value: float
    improves_over_baseline: bool
    improves_or_matches_proposed: bool
    extrapolated: bool | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("scenario_id", mode="before")
    @classmethod
    def _validate_scenario_id(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="scenario_id")

    @field_validator("scenario_type", mode="before")
    @classmethod
    def _validate_scenario_type(
        cls,
        value: object,
    ) -> WhatIfVerificationScenarioType:
        if isinstance(value, WhatIfVerificationScenarioType):
            return value
        if isinstance(value, str):
            return WhatIfVerificationScenarioType(value)
        raise ValueError(
            "scenario_type must be WhatIfVerificationScenarioType, "
            f"got {type(value).__name__}"
        )

    @field_validator("perturbed_variable", mode="before")
    @classmethod
    def _validate_perturbed_variable(cls, value: object) -> str | None:
        return _require_optional_non_empty_str(
            value,
            field_name="perturbed_variable",
        )

    @field_validator("perturbation_direction", mode="before")
    @classmethod
    def _validate_direction(
        cls,
        value: object,
    ) -> WhatIfPerturbationDirection | None:
        if value is None:
            return None
        if isinstance(value, WhatIfPerturbationDirection):
            return value
        if isinstance(value, str):
            return WhatIfPerturbationDirection(value)
        raise ValueError(
            "perturbation_direction must be WhatIfPerturbationDirection or None"
        )

    @field_validator(
        "perturbed_value",
        "predicted_quality",
        "anomaly_score",
        mode="before",
    )
    @classmethod
    def _validate_optional_floats(cls, value: object) -> float | None:
        return _require_optional_finite_float(value, field_name="optional float")

    @field_validator("variable_values", mode="before")
    @classmethod
    def _validate_variable_values(cls, value: object) -> dict[str, float]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("variable_values must be a dict[str, float]")
        cleaned: dict[str, float] = {}
        for key, raw in value.items():
            name = _require_non_empty_str(key, field_name="variable_values key")
            cleaned[name] = _require_finite_float(
                raw,
                field_name=f"variable_values[{name!r}]",
            )
        return cleaned

    @field_validator("objective_value", mode="before")
    @classmethod
    def _validate_objective_value(cls, value: object) -> float:
        return _require_finite_float(value, field_name="objective_value")

    @field_validator(
        "improves_over_baseline",
        "improves_or_matches_proposed",
        mode="before",
    )
    @classmethod
    def _validate_bools(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="scenario bool")

    @field_validator("extrapolated", mode="before")
    @classmethod
    def _validate_extrapolated(cls, value: object) -> bool | None:
        if value is None:
            return None
        return _require_strict_bool(value, field_name="extrapolated")

    @field_validator("warnings", mode="before")
    @classmethod
    def _validate_warnings_before(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("warnings must be a list[str]")
        return list(value)

    @field_validator("warnings", mode="after")
    @classmethod
    def _validate_warnings(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(value, field_name="warnings")


class RecommendationWhatIfVerificationView(BaseModel):
    """Presentation summary for recommendation local what-if verification."""

    model_config = ConfigDict(extra="forbid")

    status: WhatIfVerificationStatus
    objective: RecommendationObjective
    baseline_objective_value: float | None = None
    proposed_objective_value: float | None = None
    scenario_count: int
    neighbor_scenario_count: int
    improving_neighbor_count: int
    non_improving_neighbor_count: int
    extrapolated_scenario_count: int
    stability_classification: WhatIfStabilityClassification
    stability_message: str
    scenarios: list[WhatIfVerificationScenarioView] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rationale: str
    disclaimer: str = (
        "Adjacent constraint-grid scenarios are scored with the same fitted "
        "model to show whether the proposed improvement persists locally. "
        "This is model-local stability evidence, not proof of physical safety "
        "or causation."
    )

    @field_validator("status", mode="before")
    @classmethod
    def _validate_status(cls, value: object) -> WhatIfVerificationStatus:
        if isinstance(value, WhatIfVerificationStatus):
            return value
        if isinstance(value, str):
            return WhatIfVerificationStatus(value)
        raise ValueError("status must be WhatIfVerificationStatus")

    @field_validator("objective", mode="before")
    @classmethod
    def _validate_objective(cls, value: object) -> RecommendationObjective:
        if isinstance(value, RecommendationObjective):
            return value
        if isinstance(value, str):
            return RecommendationObjective(value)
        raise ValueError("objective must be RecommendationObjective")

    @field_validator(
        "baseline_objective_value",
        "proposed_objective_value",
        mode="before",
    )
    @classmethod
    def _validate_optional_objectives(cls, value: object) -> float | None:
        return _require_optional_finite_float(value, field_name="objective value")

    @field_validator(
        "scenario_count",
        "neighbor_scenario_count",
        "improving_neighbor_count",
        "non_improving_neighbor_count",
        "extrapolated_scenario_count",
        mode="before",
    )
    @classmethod
    def _validate_counts(cls, value: object) -> int:
        return _require_strict_int_ge(value, field_name="count", minimum=0)

    @field_validator("stability_classification", mode="before")
    @classmethod
    def _validate_stability(
        cls,
        value: object,
    ) -> WhatIfStabilityClassification:
        if isinstance(value, WhatIfStabilityClassification):
            return value
        if isinstance(value, str):
            return WhatIfStabilityClassification(value)
        raise ValueError("stability_classification must be WhatIfStabilityClassification")

    @field_validator("stability_message", "rationale", "disclaimer", mode="before")
    @classmethod
    def _validate_text(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="text field")

    @field_validator("scenarios", mode="before")
    @classmethod
    def _validate_scenarios_before(
        cls,
        value: object,
    ) -> list[WhatIfVerificationScenarioView]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("scenarios must be a list")
        return list(value)

    @field_validator("scenarios", mode="after")
    @classmethod
    def _validate_scenarios_after(
        cls,
        value: list[WhatIfVerificationScenarioView],
    ) -> list[WhatIfVerificationScenarioView]:
        return [item.model_copy(deep=True) for item in value]

    @field_validator("warnings", mode="before")
    @classmethod
    def _validate_warnings_before(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("warnings must be a list[str]")
        return list(value)

    @field_validator("warnings", mode="after")
    @classmethod
    def _validate_warnings(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(value, field_name="warnings")


class AnomalyEventView(BaseModel):
    """Presentation view of one selected anomaly event."""

    model_config = ConfigDict(extra="forbid")

    rank: int
    original_row_id: int | str
    anomaly_score: float
    is_operating_row: bool
    selection_source: str
    score_direction: str
    is_anomaly_flagged: bool | None = None

    @field_validator("rank", mode="before")
    @classmethod
    def _validate_rank(cls, value: object) -> int:
        return _require_strict_int_ge(value, field_name="rank", minimum=1)

    @field_validator("original_row_id", mode="before")
    @classmethod
    def _validate_original_row_id(cls, value: object) -> int | str:
        if isinstance(value, bool):
            raise ValueError("original_row_id must not be a bool")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return _require_non_empty_str(value, field_name="original_row_id")
        raise ValueError(
            f"original_row_id must be int or str, got {type(value).__name__}"
        )

    @field_validator("anomaly_score", mode="before")
    @classmethod
    def _validate_anomaly_score(cls, value: object) -> float:
        return _require_finite_float(value, field_name="anomaly_score")

    @field_validator("is_operating_row", mode="before")
    @classmethod
    def _validate_is_operating_row(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="is_operating_row")

    @field_validator("selection_source", "score_direction", mode="before")
    @classmethod
    def _validate_non_empty_str(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="anomaly event string field")

    @field_validator("is_anomaly_flagged", mode="before")
    @classmethod
    def _validate_is_anomaly_flagged(cls, value: object) -> bool | None:
        if value is None:
            return None
        return _require_strict_bool(value, field_name="is_anomaly_flagged")

    @model_validator(mode="after")
    def _validate_score_direction(self) -> Self:
        if self.score_direction != _ANOMALY_SCORE_DIRECTION:
            raise ValueError(
                "score_direction must be "
                f"{_ANOMALY_SCORE_DIRECTION!r}, got {self.score_direction!r}"
            )
        return self


class DiagnosisFactorView(BaseModel):
    """Presentation view of one diagnosis association factor."""

    model_config = ConfigDict(extra="forbid")

    rank: int
    feature_name: str
    diagnostic_score: float | None = None
    direction: str | None = None
    anomaly_group_value: float | None = None
    normal_group_value: float | None = None
    raw_group_difference: float | None = None
    robust_scale: float | None = None
    robust_z_score: float | None = None
    robust_scale_status: str | None = None
    effect_size: float | None = None
    confidence: float | None = None
    source: str | None = None

    @field_validator("rank", mode="before")
    @classmethod
    def _validate_rank(cls, value: object) -> int:
        return _require_strict_int_ge(value, field_name="rank", minimum=1)

    @field_validator("feature_name", mode="before")
    @classmethod
    def _validate_feature_name(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="feature_name")

    @field_validator("diagnostic_score", mode="before")
    @classmethod
    def _validate_diagnostic_score(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_confidence(value, field_name="diagnostic_score")

    @field_validator(
        "anomaly_group_value",
        "normal_group_value",
        "raw_group_difference",
        "robust_z_score",
        "effect_size",
        mode="before",
    )
    @classmethod
    def _validate_optional_finite(cls, value: object) -> float | None:
        return _require_optional_finite_float(value, field_name="optional finite field")

    @field_validator("robust_scale", mode="before")
    @classmethod
    def _validate_robust_scale(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_non_negative_finite_float(value, field_name="robust_scale")

    @field_validator("direction", "source", "robust_scale_status", mode="before")
    @classmethod
    def _validate_optional_strings(cls, value: object) -> str | None:
        return _require_optional_non_empty_str(value, field_name="optional string")

    @field_validator("confidence", mode="before")
    @classmethod
    def _validate_confidence(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_confidence(value, field_name="confidence")


class AnomalyContextValueView(BaseModel):
    """Presentation view of one original feature value in a context row."""

    model_config = ConfigDict(extra="forbid")

    feature_name: str
    value: ScalarMetadataValue = None

    @field_validator("feature_name", mode="before")
    @classmethod
    def _validate_feature_name(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="feature_name")

    @field_validator("value", mode="before")
    @classmethod
    def _validate_value(cls, value: object) -> ScalarMetadataValue:
        if value is None or isinstance(value, (str, bool)):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float):
            return _require_finite_float(value, field_name="value")
        raise ValueError(
            f"value must be str, int, float, bool, or None, got {type(value).__name__}"
        )


class AnomalyContextIdentifierValueView(BaseModel):
    """Presentation view of one original identifier value in a context row."""

    model_config = ConfigDict(extra="forbid")

    column_name: str
    value: ScalarMetadataValue = None

    @field_validator("column_name", mode="before")
    @classmethod
    def _validate_column_name(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="column_name")

    @field_validator("value", mode="before")
    @classmethod
    def _validate_value(cls, value: object) -> ScalarMetadataValue:
        if value is None or isinstance(value, (str, bool)):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float):
            return _require_finite_float(value, field_name="value")
        raise ValueError(
            f"value must be str, int, float, bool, or None, got {type(value).__name__}"
        )


class AnomalyContextRowView(BaseModel):
    """Presentation view of one analysis-order context row."""

    model_config = ConfigDict(extra="forbid")

    analysis_position: int
    original_row_id: int | str
    relative_offset: int
    is_center_event: bool
    is_selected_anomaly_event: bool
    identifier_values: list[AnomalyContextIdentifierValueView] = Field(
        default_factory=list
    )
    timestamp_value: ScalarMetadataValue = None
    feature_values: list[AnomalyContextValueView] = Field(default_factory=list)

    @field_validator("analysis_position", mode="before")
    @classmethod
    def _validate_analysis_position(cls, value: object) -> int:
        return _require_strict_int_ge0(value, field_name="analysis_position")

    @field_validator("original_row_id", mode="before")
    @classmethod
    def _validate_original_row_id(cls, value: object) -> int | str:
        if isinstance(value, bool):
            raise ValueError("original_row_id must not be a bool")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return _require_non_empty_str(value, field_name="original_row_id")
        raise ValueError(
            f"original_row_id must be int or str, got {type(value).__name__}"
        )

    @field_validator("relative_offset", mode="before")
    @classmethod
    def _validate_relative_offset(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "relative_offset must be an int (bool not allowed), "
                f"got {type(value).__name__}"
            )
        return value

    @field_validator("is_center_event", "is_selected_anomaly_event", mode="before")
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="context row bool field")

    @field_validator("identifier_values", mode="before")
    @classmethod
    def _validate_identifier_values_before(
        cls,
        value: object,
    ) -> list[AnomalyContextIdentifierValueView]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                "identifier_values must be a list[AnomalyContextIdentifierValueView], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("identifier_values", mode="after")
    @classmethod
    def _validate_identifier_values(
        cls,
        value: list[AnomalyContextIdentifierValueView],
    ) -> list[AnomalyContextIdentifierValueView]:
        copied: list[AnomalyContextIdentifierValueView] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, AnomalyContextIdentifierValueView):
                raise ValueError(
                    "identifier_values entries must be "
                    "AnomalyContextIdentifierValueView, "
                    f"got {type(item).__name__}"
                )
            if item.column_name in seen:
                raise ValueError(
                    "identifier_values must not contain duplicate column_name: "
                    f"{item.column_name!r}"
                )
            seen.add(item.column_name)
            copied.append(item.model_copy(deep=True))
        return copied

    @field_validator("timestamp_value", mode="before")
    @classmethod
    def _validate_timestamp_value(cls, value: object) -> ScalarMetadataValue:
        if value is None or isinstance(value, (str, bool)):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float):
            return _require_finite_float(value, field_name="timestamp_value")
        raise ValueError(
            "timestamp_value must be str, int, float, bool, or None, "
            f"got {type(value).__name__}"
        )

    @field_validator("feature_values", mode="before")
    @classmethod
    def _validate_feature_values_before(
        cls,
        value: object,
    ) -> list[AnomalyContextValueView]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                "feature_values must be a list[AnomalyContextValueView], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("feature_values", mode="after")
    @classmethod
    def _validate_feature_values(
        cls,
        value: list[AnomalyContextValueView],
    ) -> list[AnomalyContextValueView]:
        copied: list[AnomalyContextValueView] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, AnomalyContextValueView):
                raise ValueError(
                    "feature_values entries must be AnomalyContextValueView, "
                    f"got {type(item).__name__}"
                )
            if item.feature_name in seen:
                raise ValueError(
                    "feature_values must not contain duplicate feature_name: "
                    f"{item.feature_name!r}"
                )
            seen.add(item.feature_name)
            copied.append(item.model_copy(deep=True))
        return copied

    @model_validator(mode="after")
    def _validate_center_offset_consistency(self) -> Self:
        if self.is_center_event and self.relative_offset != 0:
            raise ValueError(
                "is_center_event=True requires relative_offset=0, "
                f"got {self.relative_offset}"
            )
        if self.relative_offset == 0 and not self.is_center_event:
            raise ValueError("relative_offset=0 requires is_center_event=True")
        return self


class AnomalyContextWindowView(BaseModel):
    """Presentation view of one anomaly event context window."""

    model_config = ConfigDict(extra="forbid")

    event_rank: int
    center_original_row_id: int | str
    center_anomaly_score: float
    radius: int
    order_basis: AnomalyContextOrderBasis
    feature_names: list[str] = Field(default_factory=list)
    rows: list[AnomalyContextRowView] = Field(default_factory=list)

    @field_validator("event_rank", mode="before")
    @classmethod
    def _validate_event_rank(cls, value: object) -> int:
        return _require_strict_int_ge(value, field_name="event_rank", minimum=1)

    @field_validator("center_original_row_id", mode="before")
    @classmethod
    def _validate_center_original_row_id(cls, value: object) -> int | str:
        if isinstance(value, bool):
            raise ValueError("center_original_row_id must not be a bool")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return _require_non_empty_str(value, field_name="center_original_row_id")
        raise ValueError(
            "center_original_row_id must be int or str, "
            f"got {type(value).__name__}"
        )

    @field_validator("center_anomaly_score", mode="before")
    @classmethod
    def _validate_center_anomaly_score(cls, value: object) -> float:
        return _require_finite_float(value, field_name="center_anomaly_score")

    @field_validator("radius", mode="before")
    @classmethod
    def _validate_radius(cls, value: object) -> int:
        return _require_strict_int_ge0(value, field_name="radius")

    @field_validator("order_basis", mode="before")
    @classmethod
    def _validate_order_basis(cls, value: object) -> AnomalyContextOrderBasis:
        if isinstance(value, AnomalyContextOrderBasis):
            return value
        if isinstance(value, str):
            try:
                return AnomalyContextOrderBasis(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid AnomalyContextOrderBasis: {value!r}"
                ) from exc
        raise ValueError(
            "order_basis must be AnomalyContextOrderBasis, "
            f"got {type(value).__name__}"
        )

    @field_validator("feature_names", mode="before")
    @classmethod
    def _validate_feature_names_before(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"feature_names must be a list[str], got {type(value).__name__}"
            )
        return list(value)

    @field_validator("feature_names", mode="after")
    @classmethod
    def _validate_feature_names(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(value, field_name="feature_names")

    @field_validator("rows", mode="before")
    @classmethod
    def _validate_rows_before(cls, value: object) -> list[AnomalyContextRowView]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"rows must be a list[AnomalyContextRowView], got {type(value).__name__}"
            )
        return list(value)

    @field_validator("rows", mode="after")
    @classmethod
    def _validate_rows(
        cls,
        value: list[AnomalyContextRowView],
    ) -> list[AnomalyContextRowView]:
        copied: list[AnomalyContextRowView] = []
        for item in value:
            if not isinstance(item, AnomalyContextRowView):
                raise ValueError(
                    "rows entries must be AnomalyContextRowView, "
                    f"got {type(item).__name__}"
                )
            copied.append(item.model_copy(deep=True))
        return copied


class WorkflowPresentationReport(BaseModel):
    """JSON-safe presentation DTO for one analysis workflow report."""

    model_config = ConfigDict(extra="forbid")

    overview: WorkflowOverviewView
    data_summary: WorkflowDataSummaryView
    cohort_filter_summary: WorkflowCohortFilterSummaryView
    routing_summary: WorkflowRoutingSummaryView
    model_summary: WorkflowModelSummaryView
    model_performance: ModelPerformanceView | None = None
    stages: list[WorkflowStageView]
    anomaly_events: list[AnomalyEventView] = Field(default_factory=list)
    diagnosis_factors: list[DiagnosisFactorView] = Field(default_factory=list)
    anomaly_context_windows: list[AnomalyContextWindowView] = Field(
        default_factory=list
    )
    recommendation: RecommendationView | None = None
    recommendation_verification: RecommendationWhatIfVerificationView | None = None
    warnings: list[str] = Field(default_factory=list)
    disclaimers: list[str] = Field(default_factory=list)
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)
    dataset_fingerprint: str | None = None

    @field_validator("overview", mode="before")
    @classmethod
    def _validate_overview(cls, value: object) -> WorkflowOverviewView:
        if isinstance(value, WorkflowOverviewView):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return WorkflowOverviewView.model_validate(value)
        raise ValueError(
            f"overview must be WorkflowOverviewView, got {type(value).__name__}"
        )

    @field_validator("data_summary", mode="before")
    @classmethod
    def _validate_data_summary(cls, value: object) -> WorkflowDataSummaryView:
        if isinstance(value, WorkflowDataSummaryView):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return WorkflowDataSummaryView.model_validate(value)
        raise ValueError(
            "data_summary must be WorkflowDataSummaryView, "
            f"got {type(value).__name__}"
        )

    @field_validator("cohort_filter_summary", mode="before")
    @classmethod
    def _validate_cohort_filter_summary(
        cls,
        value: object,
    ) -> WorkflowCohortFilterSummaryView:
        if isinstance(value, WorkflowCohortFilterSummaryView):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return WorkflowCohortFilterSummaryView.model_validate(value)
        raise ValueError(
            "cohort_filter_summary must be WorkflowCohortFilterSummaryView, "
            f"got {type(value).__name__}"
        )

    @field_validator("routing_summary", mode="before")
    @classmethod
    def _validate_routing_summary(cls, value: object) -> WorkflowRoutingSummaryView:
        if isinstance(value, WorkflowRoutingSummaryView):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return WorkflowRoutingSummaryView.model_validate(value)
        raise ValueError(
            "routing_summary must be WorkflowRoutingSummaryView, "
            f"got {type(value).__name__}"
        )

    @field_validator("model_summary", mode="before")
    @classmethod
    def _validate_model_summary(cls, value: object) -> WorkflowModelSummaryView:
        if isinstance(value, WorkflowModelSummaryView):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return WorkflowModelSummaryView.model_validate(value)
        raise ValueError(
            "model_summary must be WorkflowModelSummaryView, "
            f"got {type(value).__name__}"
        )

    @field_validator("model_performance", mode="before")
    @classmethod
    def _validate_model_performance(cls, value: object) -> ModelPerformanceView | None:
        if value is None:
            return None
        if isinstance(value, ModelPerformanceView):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return ModelPerformanceView.model_validate(value)
        raise ValueError(
            "model_performance must be ModelPerformanceView or None, "
            f"got {type(value).__name__}"
        )

    @field_validator("stages", mode="before")
    @classmethod
    def _validate_stages_before(cls, value: object) -> list[WorkflowStageView]:
        if not isinstance(value, list):
            raise ValueError(
                f"stages must be a list[WorkflowStageView], got {type(value).__name__}"
            )
        return list(value)

    @field_validator("stages", mode="after")
    @classmethod
    def _validate_stages(cls, value: list[WorkflowStageView]) -> list[WorkflowStageView]:
        if not value:
            raise ValueError("stages must contain at least one WorkflowStageView")
        seen: set[AnalysisWorkflowStage] = set()
        copied: list[WorkflowStageView] = []
        for index, item in enumerate(value, start=1):
            if not isinstance(item, WorkflowStageView):
                raise ValueError(
                    "stages entries must be WorkflowStageView, "
                    f"got {type(item).__name__}"
                )
            if item.sequence != index:
                raise ValueError(
                    "stages sequence must be contiguous starting at 1 "
                    f"(expected {index}, got {item.sequence})"
                )
            if item.stage in seen:
                raise ValueError(
                    f"stages must not contain duplicate stages: {item.stage!r}"
                )
            seen.add(item.stage)
            copied.append(item.model_copy(deep=True))
        return copied

    @field_validator("anomaly_events", mode="before")
    @classmethod
    def _validate_anomaly_events_before(cls, value: object) -> list[AnomalyEventView]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"anomaly_events must be a list[AnomalyEventView], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("anomaly_events", mode="after")
    @classmethod
    def _validate_anomaly_events(
        cls,
        value: list[AnomalyEventView],
    ) -> list[AnomalyEventView]:
        copied: list[AnomalyEventView] = []
        seen_ids: set[int | str] = set()
        for index, item in enumerate(value, start=1):
            if not isinstance(item, AnomalyEventView):
                raise ValueError(
                    "anomaly_events entries must be AnomalyEventView, "
                    f"got {type(item).__name__}"
                )
            if item.rank != index:
                raise ValueError(
                    "anomaly_events rank must be contiguous starting at 1 "
                    f"(expected {index}, got {item.rank})"
                )
            if item.original_row_id in seen_ids:
                raise ValueError(
                    "anomaly_events must not contain duplicate original_row_id: "
                    f"{item.original_row_id!r}"
                )
            seen_ids.add(item.original_row_id)
            copied.append(item.model_copy(deep=True))
        return copied

    @field_validator("diagnosis_factors", mode="before")
    @classmethod
    def _validate_diagnosis_factors_before(
        cls,
        value: object,
    ) -> list[DiagnosisFactorView]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"diagnosis_factors must be a list[DiagnosisFactorView], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("diagnosis_factors", mode="after")
    @classmethod
    def _validate_diagnosis_factors(
        cls,
        value: list[DiagnosisFactorView],
    ) -> list[DiagnosisFactorView]:
        copied: list[DiagnosisFactorView] = []
        seen_names: set[str] = set()
        for index, item in enumerate(value, start=1):
            if not isinstance(item, DiagnosisFactorView):
                raise ValueError(
                    "diagnosis_factors entries must be DiagnosisFactorView, "
                    f"got {type(item).__name__}"
                )
            if item.rank != index:
                raise ValueError(
                    "diagnosis_factors rank must be contiguous starting at 1 "
                    f"(expected {index}, got {item.rank})"
                )
            if item.feature_name in seen_names:
                raise ValueError(
                    "diagnosis_factors must not contain duplicate feature_name: "
                    f"{item.feature_name!r}"
                )
            seen_names.add(item.feature_name)
            copied.append(item.model_copy(deep=True))
        return copied

    @field_validator("anomaly_context_windows", mode="before")
    @classmethod
    def _validate_anomaly_context_windows_before(
        cls,
        value: object,
    ) -> list[AnomalyContextWindowView]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                "anomaly_context_windows must be a list[AnomalyContextWindowView], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("anomaly_context_windows", mode="after")
    @classmethod
    def _validate_anomaly_context_windows(
        cls,
        value: list[AnomalyContextWindowView],
    ) -> list[AnomalyContextWindowView]:
        copied: list[AnomalyContextWindowView] = []
        seen_ranks: set[int] = set()
        for item in value:
            if not isinstance(item, AnomalyContextWindowView):
                raise ValueError(
                    "anomaly_context_windows entries must be AnomalyContextWindowView, "
                    f"got {type(item).__name__}"
                )
            if item.event_rank in seen_ranks:
                raise ValueError(
                    "anomaly_context_windows must not contain duplicate event_rank: "
                    f"{item.event_rank!r}"
                )
            seen_ranks.add(item.event_rank)
            copied.append(item.model_copy(deep=True))
        return copied

    @field_validator("recommendation", mode="before")
    @classmethod
    def _validate_recommendation(cls, value: object) -> RecommendationView | None:
        if value is None:
            return None
        if isinstance(value, RecommendationView):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return RecommendationView.model_validate(value)
        raise ValueError(
            "recommendation must be RecommendationView or None, "
            f"got {type(value).__name__}"
        )

    @field_validator("recommendation_verification", mode="before")
    @classmethod
    def _validate_recommendation_verification(
        cls,
        value: object,
    ) -> RecommendationWhatIfVerificationView | None:
        if value is None:
            return None
        if isinstance(value, RecommendationWhatIfVerificationView):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return RecommendationWhatIfVerificationView.model_validate(value)
        raise ValueError(
            "recommendation_verification must be "
            "RecommendationWhatIfVerificationView or None, "
            f"got {type(value).__name__}"
        )

    @field_validator("warnings", "disclaimers", mode="before")
    @classmethod
    def _validate_string_lists_before(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(f"string list must be a list[str], got {type(value).__name__}")
        return list(value)

    @field_validator("warnings", "disclaimers", mode="after")
    @classmethod
    def _validate_string_lists(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(value, field_name="string list")

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> dict[str, ScalarMetadataValue]:
        if value is None:
            return {}
        return _validate_scalar_metadata(value)

    @field_validator("dataset_fingerprint", mode="before")
    @classmethod
    def _validate_dataset_fingerprint(cls, value: object) -> str | None:
        return normalize_optional_dataset_fingerprint(value)

    @model_validator(mode="after")
    def _validate_presentation_consistency(self) -> Self:
        status = self.overview.status
        recommendation = self.recommendation
        overview_rec_status = self.overview.recommendation_status

        if recommendation is None:
            if overview_rec_status is not None:
                raise ValueError(
                    "overview.recommendation_status must be None when "
                    "recommendation is None"
                )
        elif overview_rec_status is not recommendation.status:
            raise ValueError(
                "overview.recommendation_status must equal recommendation.status"
            )

        if status is AnalysisWorkflowStatus.COMPLETED:
            if recommendation is None:
                raise ValueError("COMPLETED requires recommendation")
            if recommendation.status is not RecommendationStatus.GENERATED:
                raise ValueError("COMPLETED requires recommendation.status GENERATED")
        elif status is AnalysisWorkflowStatus.PARTIAL:
            if recommendation is not None:
                if (
                    recommendation.status
                    is not RecommendationStatus.READY_FOR_OPTIMIZATION
                ):
                    raise ValueError(
                        "PARTIAL with recommendation requires "
                        "READY_FOR_OPTIMIZATION status"
                    )
        elif status is AnalysisWorkflowStatus.REFUSED:
            if recommendation is not None:
                if recommendation.status is not RecommendationStatus.REFUSED:
                    raise ValueError(
                        "REFUSED with recommendation requires "
                        "RecommendationStatus.REFUSED"
                    )
        return self


class DiagnosisFactorPresenceStatus(StrEnum):
    """Whether a diagnosis factor appears in baseline, current, or both runs."""

    SHARED = "SHARED"
    BASELINE_ONLY = "BASELINE_ONLY"
    CURRENT_ONLY = "CURRENT_ONLY"


class EventOverlapPresenceStatus(StrEnum):
    """Whether a selected anomaly event appears in baseline, current, or both."""

    SHARED = "SHARED"
    BASELINE_ONLY = "BASELINE_ONLY"
    CURRENT_ONLY = "CURRENT_ONLY"


class RunConfigurationComparisonView(BaseModel):
    """Presentation comparison of baseline vs current run configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_analysis_mode: str | None = None
    current_analysis_mode: str | None = None
    baseline_industry: str | None = None
    current_industry: str | None = None
    baseline_cohort_configured: bool
    current_cohort_configured: bool
    baseline_cohort_description: str
    current_cohort_description: str
    baseline_analysis_rows: int
    current_analysis_rows: int
    baseline_feature_count: int | None = None
    current_feature_count: int | None = None
    baseline_train_row_count: int
    current_train_row_count: int
    baseline_validation_row_count: int
    current_validation_row_count: int
    baseline_test_row_count: int
    current_test_row_count: int
    baseline_anomaly_model: str | None = None
    current_anomaly_model: str | None = None
    baseline_anomaly_event_count: int
    current_anomaly_event_count: int
    baseline_diagnosis_factor_count: int
    current_diagnosis_factor_count: int
    baseline_operating_row_id: int | str | None = None
    current_operating_row_id: int | str | None = None

    @field_validator(
        "baseline_analysis_mode",
        "current_analysis_mode",
        "baseline_industry",
        "current_industry",
        "baseline_anomaly_model",
        "current_anomaly_model",
        mode="before",
    )
    @classmethod
    def _validate_optional_strings(cls, value: object) -> str | None:
        return _require_optional_non_empty_str(value, field_name="optional string")

    @field_validator(
        "baseline_cohort_configured",
        "current_cohort_configured",
        mode="before",
    )
    @classmethod
    def _validate_bools(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="cohort configured")

    @field_validator(
        "baseline_cohort_description",
        "current_cohort_description",
        mode="before",
    )
    @classmethod
    def _validate_descriptions(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="cohort description")

    @field_validator(
        "baseline_analysis_rows",
        "current_analysis_rows",
        "baseline_train_row_count",
        "current_train_row_count",
        "baseline_validation_row_count",
        "current_validation_row_count",
        "baseline_test_row_count",
        "current_test_row_count",
        "baseline_anomaly_event_count",
        "current_anomaly_event_count",
        "baseline_diagnosis_factor_count",
        "current_diagnosis_factor_count",
        mode="before",
    )
    @classmethod
    def _validate_counts(cls, value: object) -> int:
        return _require_strict_int_ge0(value, field_name="count field")

    @field_validator(
        "baseline_feature_count",
        "current_feature_count",
        mode="before",
    )
    @classmethod
    def _validate_optional_feature_count(cls, value: object) -> int | None:
        if value is None:
            return None
        return _require_strict_int_ge0(value, field_name="feature_count")

    @field_validator(
        "baseline_operating_row_id",
        "current_operating_row_id",
        mode="before",
    )
    @classmethod
    def _validate_operating_row_id(cls, value: object) -> int | str | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("operating row ID must not be a bool")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return _require_non_empty_str(value, field_name="operating row ID")
        raise ValueError(
            "operating row ID must be int, str, or None, "
            f"got {type(value).__name__}"
        )


class EventOverlapEntryView(BaseModel):
    """One original-row identity compared across baseline and current events."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    presence_status: EventOverlapPresenceStatus
    original_row_id: int | str
    baseline_rank: int | None = None
    current_rank: int | None = None

    @field_validator("presence_status", mode="before")
    @classmethod
    def _validate_presence_status(cls, value: object) -> EventOverlapPresenceStatus:
        if isinstance(value, EventOverlapPresenceStatus):
            return value
        if isinstance(value, str):
            try:
                return EventOverlapPresenceStatus(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid EventOverlapPresenceStatus: {value!r}"
                ) from exc
        raise ValueError(
            "presence_status must be EventOverlapPresenceStatus, "
            f"got {type(value).__name__}"
        )

    @field_validator("original_row_id", mode="before")
    @classmethod
    def _validate_original_row_id(cls, value: object) -> int | str:
        if isinstance(value, bool):
            raise ValueError("original_row_id must not be a bool")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return _require_non_empty_str(value, field_name="original_row_id")
        raise ValueError(
            f"original_row_id must be int or str, got {type(value).__name__}"
        )

    @field_validator("baseline_rank", "current_rank", mode="before")
    @classmethod
    def _validate_optional_rank(cls, value: object) -> int | None:
        if value is None:
            return None
        return _require_strict_int_ge(value, field_name="rank", minimum=1)

    @model_validator(mode="after")
    def _validate_rank_presence(self) -> Self:
        if self.presence_status is EventOverlapPresenceStatus.SHARED:
            if self.baseline_rank is None or self.current_rank is None:
                raise ValueError("SHARED events require both baseline and current ranks")
        elif self.presence_status is EventOverlapPresenceStatus.BASELINE_ONLY:
            if self.baseline_rank is None or self.current_rank is not None:
                raise ValueError(
                    "BASELINE_ONLY events require baseline_rank only"
                )
        elif self.presence_status is EventOverlapPresenceStatus.CURRENT_ONLY:
            if self.current_rank is None or self.baseline_rank is not None:
                raise ValueError(
                    "CURRENT_ONLY events require current_rank only"
                )
        return self


class EventOverlapView(BaseModel):
    """Selected-event overlap between baseline and current anomaly runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_event_count: int
    current_event_count: int
    shared_event_count: int
    shared_original_row_ids: list[int | str] = Field(default_factory=list)
    baseline_only_original_row_ids: list[int | str] = Field(default_factory=list)
    current_only_original_row_ids: list[int | str] = Field(default_factory=list)
    entries: list[EventOverlapEntryView] = Field(default_factory=list)

    @field_validator(
        "baseline_event_count",
        "current_event_count",
        "shared_event_count",
        mode="before",
    )
    @classmethod
    def _validate_counts(cls, value: object) -> int:
        return _require_strict_int_ge0(value, field_name="event count")

    @field_validator(
        "shared_original_row_ids",
        "baseline_only_original_row_ids",
        "current_only_original_row_ids",
        mode="before",
    )
    @classmethod
    def _validate_id_lists_before(cls, value: object) -> list[int | str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"row ID list must be a list[int | str], got {type(value).__name__}"
            )
        return list(value)

    @field_validator(
        "shared_original_row_ids",
        "baseline_only_original_row_ids",
        "current_only_original_row_ids",
        mode="after",
    )
    @classmethod
    def _validate_id_lists(cls, value: list[int | str]) -> list[int | str]:
        cleaned: list[int | str] = []
        seen: set[int | str] = set()
        for item in value:
            if isinstance(item, bool):
                raise ValueError("row ID list must not contain bool values")
            if isinstance(item, int):
                row_id: int | str = item
            elif isinstance(item, str):
                row_id = _require_non_empty_str(item, field_name="row ID")
            else:
                raise ValueError(
                    "row ID list entries must be int or str, "
                    f"got {type(item).__name__}"
                )
            if row_id in seen:
                raise ValueError(f"row ID list must not contain duplicates: {row_id!r}")
            seen.add(row_id)
            cleaned.append(row_id)
        return cleaned

    @field_validator("entries", mode="before")
    @classmethod
    def _validate_entries_before(cls, value: object) -> list[EventOverlapEntryView]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"entries must be a list[EventOverlapEntryView], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("entries", mode="after")
    @classmethod
    def _validate_entries(
        cls,
        value: list[EventOverlapEntryView],
    ) -> list[EventOverlapEntryView]:
        copied: list[EventOverlapEntryView] = []
        seen: set[int | str] = set()
        for item in value:
            if not isinstance(item, EventOverlapEntryView):
                raise ValueError(
                    "entries entries must be EventOverlapEntryView, "
                    f"got {type(item).__name__}"
                )
            if item.original_row_id in seen:
                raise ValueError(
                    "entries must not contain duplicate original_row_id: "
                    f"{item.original_row_id!r}"
                )
            seen.add(item.original_row_id)
            copied.append(item.model_copy(deep=True))
        return copied

    @model_validator(mode="after")
    def _validate_overlap_consistency(self) -> Self:
        if self.shared_event_count != len(self.shared_original_row_ids):
            raise ValueError(
                "shared_event_count must equal len(shared_original_row_ids)"
            )
        if self.shared_event_count > self.baseline_event_count:
            raise ValueError("shared_event_count must be <= baseline_event_count")
        if self.shared_event_count > self.current_event_count:
            raise ValueError("shared_event_count must be <= current_event_count")
        if len(self.baseline_only_original_row_ids) != (
            self.baseline_event_count - self.shared_event_count
        ):
            raise ValueError(
                "baseline_only_original_row_ids length must equal "
                "baseline_event_count - shared_event_count"
            )
        if len(self.current_only_original_row_ids) != (
            self.current_event_count - self.shared_event_count
        ):
            raise ValueError(
                "current_only_original_row_ids length must equal "
                "current_event_count - shared_event_count"
            )
        return self


class DiagnosisFactorComparisonView(BaseModel):
    """One diagnosis factor compared across baseline and current ranks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_name: str
    baseline_rank: int | None = None
    current_rank: int | None = None
    baseline_direction: str | None = None
    current_direction: str | None = None
    presence_status: DiagnosisFactorPresenceStatus

    @field_validator("feature_name", mode="before")
    @classmethod
    def _validate_feature_name(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="feature_name")

    @field_validator("baseline_rank", "current_rank", mode="before")
    @classmethod
    def _validate_optional_rank(cls, value: object) -> int | None:
        if value is None:
            return None
        return _require_strict_int_ge(value, field_name="rank", minimum=1)

    @field_validator("baseline_direction", "current_direction", mode="before")
    @classmethod
    def _validate_optional_direction(cls, value: object) -> str | None:
        return _require_optional_non_empty_str(value, field_name="direction")

    @field_validator("presence_status", mode="before")
    @classmethod
    def _validate_presence_status(
        cls,
        value: object,
    ) -> DiagnosisFactorPresenceStatus:
        if isinstance(value, DiagnosisFactorPresenceStatus):
            return value
        if isinstance(value, str):
            try:
                return DiagnosisFactorPresenceStatus(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid DiagnosisFactorPresenceStatus: {value!r}"
                ) from exc
        raise ValueError(
            "presence_status must be DiagnosisFactorPresenceStatus, "
            f"got {type(value).__name__}"
        )

    @model_validator(mode="after")
    def _validate_presence_ranks(self) -> Self:
        if self.presence_status is DiagnosisFactorPresenceStatus.SHARED:
            if self.baseline_rank is None or self.current_rank is None:
                raise ValueError("SHARED factors require both baseline and current ranks")
        elif self.presence_status is DiagnosisFactorPresenceStatus.BASELINE_ONLY:
            if self.baseline_rank is None or self.current_rank is not None:
                raise ValueError(
                    "BASELINE_ONLY factors require baseline_rank only"
                )
        elif self.presence_status is DiagnosisFactorPresenceStatus.CURRENT_ONLY:
            if self.current_rank is None or self.baseline_rank is not None:
                raise ValueError(
                    "CURRENT_ONLY factors require current_rank only"
                )
        return self


class AnomalyRunComparisonView(BaseModel):
    """Immutable presentation DTO for baseline vs current anomaly-run comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    compatible: bool
    compatibility_message: str
    same_dataset: bool
    configuration: RunConfigurationComparisonView
    event_overlap: EventOverlapView
    factor_comparison: list[DiagnosisFactorComparisonView] = Field(
        default_factory=list
    )
    warnings: list[str] = Field(default_factory=list)
    disclaimers: list[str] = Field(default_factory=list)

    @field_validator("compatible", "same_dataset", mode="before")
    @classmethod
    def _validate_bools(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="comparison bool field")

    @field_validator("compatibility_message", mode="before")
    @classmethod
    def _validate_compatibility_message(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="compatibility_message")

    @field_validator("configuration", mode="before")
    @classmethod
    def _validate_configuration(
        cls,
        value: object,
    ) -> RunConfigurationComparisonView:
        if isinstance(value, RunConfigurationComparisonView):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return RunConfigurationComparisonView.model_validate(value)
        raise ValueError(
            "configuration must be RunConfigurationComparisonView, "
            f"got {type(value).__name__}"
        )

    @field_validator("event_overlap", mode="before")
    @classmethod
    def _validate_event_overlap(cls, value: object) -> EventOverlapView:
        if isinstance(value, EventOverlapView):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return EventOverlapView.model_validate(value)
        raise ValueError(
            f"event_overlap must be EventOverlapView, got {type(value).__name__}"
        )

    @field_validator("factor_comparison", mode="before")
    @classmethod
    def _validate_factor_comparison_before(
        cls,
        value: object,
    ) -> list[DiagnosisFactorComparisonView]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                "factor_comparison must be a list[DiagnosisFactorComparisonView], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("factor_comparison", mode="after")
    @classmethod
    def _validate_factor_comparison(
        cls,
        value: list[DiagnosisFactorComparisonView],
    ) -> list[DiagnosisFactorComparisonView]:
        copied: list[DiagnosisFactorComparisonView] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, DiagnosisFactorComparisonView):
                raise ValueError(
                    "factor_comparison entries must be DiagnosisFactorComparisonView, "
                    f"got {type(item).__name__}"
                )
            if item.feature_name in seen:
                raise ValueError(
                    "factor_comparison must not contain duplicate feature_name: "
                    f"{item.feature_name!r}"
                )
            seen.add(item.feature_name)
            copied.append(item.model_copy(deep=True))
        return copied

    @field_validator("warnings", "disclaimers", mode="before")
    @classmethod
    def _validate_string_lists_before(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(f"string list must be a list[str], got {type(value).__name__}")
        return list(value)

    @field_validator("warnings", "disclaimers", mode="after")
    @classmethod
    def _validate_string_lists(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(value, field_name="string list")


@dataclass(frozen=True, slots=True)
class WorkflowPresentationOutcome:
    """Immutable container for a completed workflow presentation report."""

    report: WorkflowPresentationReport
