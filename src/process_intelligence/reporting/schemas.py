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
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.evaluation.performance_acceptance import (
    MetricAcceptanceDirection,
    ModelPerformanceAcceptanceStatus,
)
from process_intelligence.recommendation.enums import (
    RecommendationSafetyStatus,
    RecommendationStatus,
)
from process_intelligence.workflow.enums import (
    AnalysisWorkflowStage,
    AnalysisWorkflowStatus,
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
        split_total = (
            self.train_row_count + self.validation_row_count + self.test_row_count
        )
        if split_total > 0 and split_total != self.processed_row_count:
            raise ValueError(
                "train_row_count + validation_row_count + test_row_count "
                "must equal processed_row_count when split counts are present "
                f"(got {split_total} != {self.processed_row_count})"
            )
        return self


class WorkflowRoutingSummaryView(BaseModel):
    """Industry, task, and feature routing summary for presentation."""

    model_config = ConfigDict(extra="forbid")

    selected_industry: str | None = None
    selected_task: AnalysisTask | None = None
    target_column: str | None = None
    feature_count: int | None = None

    @field_validator("selected_industry", "target_column", mode="before")
    @classmethod
    def _validate_optional_strings(cls, value: object) -> str | None:
        return _require_optional_non_empty_str(value, field_name="optional string")

    @field_validator("selected_task", mode="before")
    @classmethod
    def _validate_selected_task(cls, value: object) -> AnalysisTask | None:
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
            f"selected_task must be AnalysisTask or None, got {type(value).__name__}"
        )

    @field_validator("feature_count", mode="before")
    @classmethod
    def _validate_feature_count(cls, value: object) -> int | None:
        if value is None:
            return None
        return _require_strict_int_ge0(value, field_name="feature_count")


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


class WorkflowPresentationReport(BaseModel):
    """JSON-safe presentation DTO for one analysis workflow report."""

    model_config = ConfigDict(extra="forbid")

    overview: WorkflowOverviewView
    data_summary: WorkflowDataSummaryView
    routing_summary: WorkflowRoutingSummaryView
    model_summary: WorkflowModelSummaryView
    model_performance: ModelPerformanceView | None = None
    stages: list[WorkflowStageView]
    recommendation: RecommendationView | None = None
    warnings: list[str] = Field(default_factory=list)
    disclaimers: list[str] = Field(default_factory=list)
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)

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


@dataclass(frozen=True, slots=True)
class WorkflowPresentationOutcome:
    """Immutable container for a completed workflow presentation report."""

    report: WorkflowPresentationReport
