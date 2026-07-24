"""Pydantic contracts for the industrial analysis workflow (Step 10C).

Reuses core, diagnosis, and recommendation schemas. Does not embed DataFrames,
estimators, or model objects in reports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator

from process_intelligence.core.enums import AnalysisTask, ColumnRole
from process_intelligence.core.schemas import (
    AnomalyEvent,
    RootCauseFactor,
    VariableConstraint,
)
from process_intelligence.evaluation.performance_acceptance import (
    ModelPerformanceAcceptancePolicy,
    ModelPerformanceAcceptanceReport,
)
from process_intelligence.recommendation.enums import (
    RecommendationObjective,
    RecommendationStatus,
)
from process_intelligence.recommendation.scenario_ranking import (
    QualityOptimizationDirection,
)
from process_intelligence.recommendation.schemas import RecommendationResult
from process_intelligence.recommendation.what_if_verification import (
    RecommendationWhatIfVerificationResult,
)
from process_intelligence.workflow.dataset_fingerprint import (
    normalize_optional_dataset_fingerprint,
)
from process_intelligence.workflow.enums import (
    AnalysisExecutionMode,
    AnalysisWorkflowStage,
    AnalysisWorkflowStatus,
    AnomalyContextOrderBasis,
    OperatingPointSelectionMode,
    TaskSelectionSource,
)

ScalarMetadataValue = str | int | float | bool | None
"""Allowed scalar types for workflow metadata dictionaries."""

_ORIGINAL_ROW_ID = "_original_row_id"

_CANONICAL_STAGES: tuple[AnalysisWorkflowStage, ...] = tuple(AnalysisWorkflowStage)

ANOMALY_CONTEXT_RADIUS = 3
"""Fixed analysis-order radius for anomaly context windows (center ± radius)."""

ANOMALY_CONTEXT_MAX_FEATURES = 5
"""Maximum diagnosis features included in each anomaly context window."""


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


def _require_strict_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{field_name} must be an int (bool not allowed), "
            f"got {type(value).__name__}"
        )
    return value


def _require_json_safe_scalar(
    value: object,
    *,
    field_name: str,
) -> ScalarMetadataValue:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} float must be finite, got {value!r}")
        return value
    raise ValueError(
        f"{field_name} must be str, int, float, bool, or None "
        f"(no DataFrame, ndarray, estimator, or nested objects); "
        f"got {type(value).__name__}"
    )


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


def _validate_column_name(value: object, *, field_name: str) -> str:
    text = _require_non_empty_str(value, field_name=field_name)
    if text == _ORIGINAL_ROW_ID:
        raise ValueError(
            f"{field_name} cannot be reserved column '{_ORIGINAL_ROW_ID}'"
        )
    return text


def _validate_constraint_list(
    value: object,
    *,
    field_name: str,
) -> list[VariableConstraint]:
    if not isinstance(value, list):
        raise ValueError(
            f"{field_name} must be a list[VariableConstraint], "
            f"got {type(value).__name__}"
        )
    seen: set[str] = set()
    copied: list[VariableConstraint] = []
    for item in value:
        if not isinstance(item, VariableConstraint):
            raise ValueError(
                f"{field_name} entries must be VariableConstraint, "
                f"got {type(item).__name__}"
            )
        name = item.variable
        if name in seen:
            raise ValueError(
                f"{field_name} must not contain duplicate variables: {name!r}"
            )
        seen.add(name)
        copied.append(item.model_copy(deep=True))
    return copied


class NumericCohortFilter(BaseModel):
    """User-confirmed numeric operating-range filter for anomaly-only analysis.

    Restricts the analysis cohort to rows whose ``column_name`` values fall
    inside the configured inclusive/exclusive bounds. Bounds are never inferred
    from data quantiles or column names. At most one filter is supported per
    request.
    """

    column_name: str
    lower_bound: float
    upper_bound: float
    include_lower: bool = True
    include_upper: bool = True
    exclude_filter_column_from_features: bool = True

    @field_validator("column_name", mode="before")
    @classmethod
    def _validate_column_name(cls, value: object) -> str:
        return _validate_column_name(value, field_name="column_name")

    @field_validator("lower_bound", "upper_bound", mode="before")
    @classmethod
    def _validate_bounds(cls, value: object) -> float:
        return _require_finite_float(value, field_name="cohort filter bound")

    @field_validator(
        "include_lower",
        "include_upper",
        "exclude_filter_column_from_features",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="cohort filter bool field")

    @model_validator(mode="after")
    def _validate_bound_order(self) -> Self:
        if self.lower_bound > self.upper_bound:
            raise ValueError(
                "lower_bound must be <= upper_bound "
                f"(got {self.lower_bound} > {self.upper_bound})"
            )
        return self


class CohortFilterSummary(BaseModel):
    """Immutable summary of the explicit operating cohort filter for one run.

    Always present on workflow reports. When no filter was configured,
    ``configured`` is False and retained rows equal the analysis source rows.
    """

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

    @model_validator(mode="after")
    def _validate_summary_consistency(self) -> Self:
        if self.retained_row_count + self.excluded_row_count != self.source_row_count:
            raise ValueError(
                "retained_row_count + excluded_row_count must equal source_row_count "
                f"(got {self.retained_row_count} + {self.excluded_row_count} != "
                f"{self.source_row_count})"
            )
        if self.null_excluded_count > self.excluded_row_count:
            raise ValueError(
                "null_excluded_count must be <= excluded_row_count "
                f"(got {self.null_excluded_count} > {self.excluded_row_count})"
            )
        if self.configured:
            if self.column_name is None:
                raise ValueError(
                    "column_name is required when cohort filter is configured"
                )
            if self.lower_bound is None or self.upper_bound is None:
                raise ValueError(
                    "lower_bound and upper_bound are required when cohort filter "
                    "is configured"
                )
            if self.include_lower is None or self.include_upper is None:
                raise ValueError(
                    "include_lower and include_upper are required when cohort "
                    "filter is configured"
                )
            if self.exclude_filter_column_from_features is None:
                raise ValueError(
                    "exclude_filter_column_from_features is required when cohort "
                    "filter is configured"
                )
            if self.lower_bound > self.upper_bound:
                raise ValueError(
                    "lower_bound must be <= upper_bound "
                    f"(got {self.lower_bound} > {self.upper_bound})"
                )
        else:
            if self.column_name is not None:
                raise ValueError(
                    "column_name must be None when cohort filter is not configured"
                )
            if (
                self.lower_bound is not None
                or self.upper_bound is not None
                or self.include_lower is not None
                or self.include_upper is not None
                or self.exclude_filter_column_from_features is not None
            ):
                raise ValueError(
                    "filter bounds and flags must be None when cohort filter "
                    "is not configured"
                )
            if self.excluded_row_count != 0 or self.null_excluded_count != 0:
                raise ValueError(
                    "excluded_row_count and null_excluded_count must be 0 when "
                    "cohort filter is not configured"
                )
            if self.retained_row_count != self.source_row_count:
                raise ValueError(
                    "retained_row_count must equal source_row_count when cohort "
                    "filter is not configured"
                )
        return self


class AnalysisWorkflowPolicy(BaseModel):
    """Configurable safety and aggregation rules for analysis workflow runs.

    Controls early termination on validation or leakage blockers, industry and
    task requirements, anomaly-event bounds, and warning aggregation limits.
    Does not embed models or DataFrames.
    """

    stop_on_validation_blocker: bool = True
    stop_on_leakage_blocker: bool = True
    require_semiconductor_industry: bool = False
    require_regression_task: bool = True
    require_residual_diagnosis: bool = True
    allow_partial_diagnosis_ensemble: bool = False
    maximum_anomaly_events: int = 5
    minimum_anomaly_events: int = 1
    preserve_stage_outputs: bool = True
    include_stage_warnings: bool = True
    maximum_aggregated_warnings: int = 200

    @field_validator(
        "stop_on_validation_blocker",
        "stop_on_leakage_blocker",
        "require_semiconductor_industry",
        "require_regression_task",
        "require_residual_diagnosis",
        "allow_partial_diagnosis_ensemble",
        "preserve_stage_outputs",
        "include_stage_warnings",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="policy bool field")

    @field_validator(
        "maximum_anomaly_events",
        "minimum_anomaly_events",
        "maximum_aggregated_warnings",
        mode="before",
    )
    @classmethod
    def _validate_positive_ints(cls, value: object) -> int:
        return _require_strict_int_ge(value, field_name="policy int field", minimum=1)

    @model_validator(mode="after")
    def _validate_event_bounds(self) -> Self:
        if self.minimum_anomaly_events > self.maximum_anomaly_events:
            raise ValueError(
                "minimum_anomaly_events must be <= maximum_anomaly_events "
                f"(got {self.minimum_anomaly_events} > {self.maximum_anomaly_events})"
            )
        return self


class AnomalyRecommendationConfig(BaseModel):
    """Explicit opt-in for anomaly-only recommendation generation.

    When ``enabled`` is False, ANOMALY_ONLY keeps the recommendation stage
    skipped. When True, the workflow may run the existing recommendation
    pipeline with ``REDUCE_ANOMALY_SCORE`` only. Simultaneous-change limits and
    stage-output retention reuse the request/policy fields already present on
    ``AnalysisWorkflowRequest`` / ``AnalysisWorkflowPolicy``.
    """

    enabled: bool = False

    @field_validator("enabled", mode="before")
    @classmethod
    def _validate_enabled(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="enabled")


class AnalysisWorkflowRequest(BaseModel):
    """Caller inputs for a single raw-CSV industrial analysis workflow run.

    Declares the CSV path, column roles, recommendation objective, constraints,
    operating-point selection mode, and explicit model-performance acceptance
    policy. Does not load data or fit models.

    SUPERVISED mode requires a target, objective, and performance policy.
    ANOMALY_ONLY mode requires those supervised fields to be absent unless
    anomaly-only recommendation is explicitly enabled.
    """

    csv_path: Path
    feature_columns: list[str]
    analysis_mode: AnalysisExecutionMode = AnalysisExecutionMode.SUPERVISED
    target_column: str | None = None
    model_performance_policy: ModelPerformanceAcceptancePolicy | None = None
    timestamp_column: str | None = None
    identifier_columns: list[str] = Field(default_factory=list)
    excluded_columns: list[str] = Field(default_factory=list)
    column_role_overrides: dict[str, ColumnRole] = Field(default_factory=dict)
    requested_task: AnalysisTask | None = None
    objective: RecommendationObjective | None = None
    quality_direction: QualityOptimizationDirection | None = None
    quality_target: float | None = None
    declared_target_minimum: float | None = None
    declared_target_maximum: float | None = None
    request_constraints: list[VariableConstraint] = Field(default_factory=list)
    industry_constraints: list[VariableConstraint] = Field(default_factory=list)
    user_overrides: list[VariableConstraint] = Field(default_factory=list)
    user_confirmed_controllable_variables: list[str] = Field(default_factory=list)
    user_verified_variables: list[str] = Field(default_factory=list)
    max_simultaneous_changes: int = 3
    operating_point_selection: OperatingPointSelectionMode = (
        OperatingPointSelectionMode.TOP_RESIDUAL_ANOMALY
    )
    explicit_operating_row_id: int | str | None = None
    cohort_filter: NumericCohortFilter | None = None
    anomaly_recommendation: AnomalyRecommendationConfig = Field(
        default_factory=AnomalyRecommendationConfig
    )
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)

    @field_validator("csv_path", mode="before")
    @classmethod
    def _validate_csv_path(cls, value: object) -> Path:
        if isinstance(value, Path):
            path = value
        elif isinstance(value, str):
            if value == "" or value.strip() == "":
                raise ValueError("csv_path must be a non-empty path")
            path = Path(value)
        else:
            raise ValueError(
                f"csv_path must be Path or str, got {type(value).__name__}"
            )
        if str(path) == "" or str(path).strip() == "":
            raise ValueError("csv_path must be a non-empty path")
        if path.suffix.lower() != ".csv":
            raise ValueError(
                f"csv_path must have a .csv extension, got {path.suffix!r}"
            )
        return path

    @field_validator("analysis_mode", mode="before")
    @classmethod
    def _validate_analysis_mode(cls, value: object) -> AnalysisExecutionMode:
        if isinstance(value, AnalysisExecutionMode):
            return value
        if isinstance(value, str):
            try:
                return AnalysisExecutionMode(value)
            except ValueError as exc:
                raise ValueError(f"invalid AnalysisExecutionMode: {value!r}") from exc
        raise ValueError(
            "analysis_mode must be AnalysisExecutionMode, "
            f"got {type(value).__name__}"
        )

    @field_validator("target_column", mode="before")
    @classmethod
    def _validate_target_column(cls, value: object) -> str | None:
        if value is None:
            return None
        return _validate_column_name(value, field_name="target_column")

    @field_validator("feature_columns", mode="before")
    @classmethod
    def _validate_feature_columns_before(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            raise ValueError(
                f"feature_columns must be a list[str], got {type(value).__name__}"
            )
        return list(value)

    @field_validator("feature_columns", mode="after")
    @classmethod
    def _validate_feature_columns(cls, value: list[str]) -> list[str]:
        cleaned = _validate_unique_non_empty_strings(
            value,
            field_name="feature_columns",
        )
        if not cleaned:
            raise ValueError("feature_columns must contain at least one feature")
        for name in cleaned:
            if name == _ORIGINAL_ROW_ID:
                raise ValueError(
                    f"feature_columns cannot include reserved column "
                    f"'{_ORIGINAL_ROW_ID}'"
                )
        return cleaned

    @field_validator("timestamp_column", mode="before")
    @classmethod
    def _validate_timestamp_column(cls, value: object) -> str | None:
        return _require_optional_non_empty_str(value, field_name="timestamp_column")

    @field_validator("identifier_columns", "excluded_columns", mode="before")
    @classmethod
    def _validate_column_lists_before(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"column list must be a list[str], got {type(value).__name__}"
            )
        return list(value)

    @field_validator("identifier_columns", "excluded_columns", mode="after")
    @classmethod
    def _validate_column_lists(cls, value: list[str]) -> list[str]:
        cleaned = _validate_unique_non_empty_strings(value, field_name="column list")
        for name in cleaned:
            if name == _ORIGINAL_ROW_ID:
                raise ValueError(
                    f"column list cannot include reserved column '{_ORIGINAL_ROW_ID}'"
                )
        return cleaned

    @field_validator("column_role_overrides", mode="before")
    @classmethod
    def _validate_overrides_before(cls, value: object) -> dict[str, ColumnRole]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(
                f"column_role_overrides must be a dict[str, ColumnRole], "
                f"got {type(value).__name__}"
            )
        cleaned: dict[str, ColumnRole] = {}
        for key, raw in value.items():
            name = _validate_column_name(key, field_name="column_role_overrides key")
            if name in cleaned:
                raise ValueError(
                    f"column_role_overrides must not contain duplicate keys: {name!r}"
                )
            if isinstance(raw, ColumnRole):
                role = raw
            elif isinstance(raw, str):
                try:
                    role = ColumnRole(raw)
                except ValueError as exc:
                    raise ValueError(f"invalid ColumnRole: {raw!r}") from exc
            else:
                raise ValueError(
                    "column_role_overrides values must be ColumnRole, "
                    f"got {type(raw).__name__}"
                )
            cleaned[name] = role
        return cleaned

    @field_validator("requested_task", mode="before")
    @classmethod
    def _validate_requested_task(cls, value: object) -> AnalysisTask | None:
        if value is None:
            return None
        if isinstance(value, AnalysisTask):
            task = value
        elif isinstance(value, str):
            try:
                task = AnalysisTask(value)
            except ValueError as exc:
                raise ValueError(f"invalid AnalysisTask: {value!r}") from exc
        else:
            raise ValueError(
                "requested_task must be AnalysisTask or None, "
                f"got {type(value).__name__}"
            )
        if task not in {
            AnalysisTask.REGRESSION,
            AnalysisTask.CLASSIFICATION,
        }:
            raise ValueError(
                "requested_task must be REGRESSION, CLASSIFICATION, or None, "
                f"got {task!r}"
            )
        return task

    @field_validator("objective", mode="before")
    @classmethod
    def _validate_objective(cls, value: object) -> RecommendationObjective | None:
        if value is None:
            return None
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
            "objective must be RecommendationObjective or None, "
            f"got {type(value).__name__}"
        )

    @field_validator("quality_direction", mode="before")
    @classmethod
    def _validate_quality_direction(
        cls,
        value: object,
    ) -> QualityOptimizationDirection | None:
        if value is None:
            return None
        if isinstance(value, QualityOptimizationDirection):
            return value
        if isinstance(value, str):
            try:
                return QualityOptimizationDirection(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid QualityOptimizationDirection: {value!r}"
                ) from exc
        raise ValueError(
            "quality_direction must be QualityOptimizationDirection or None, "
            f"got {type(value).__name__}"
        )

    @field_validator("quality_target", mode="before")
    @classmethod
    def _validate_quality_target(cls, value: object) -> float | None:
        return _require_optional_finite_float(value, field_name="quality_target")

    @field_validator("declared_target_minimum", "declared_target_maximum", mode="before")
    @classmethod
    def _validate_declared_target_bounds(cls, value: object) -> float | None:
        return _require_optional_finite_float(value, field_name="declared target bound")

    @field_validator(
        "request_constraints",
        "industry_constraints",
        "user_overrides",
        mode="before",
    )
    @classmethod
    def _validate_constraints_before(cls, value: object) -> list[VariableConstraint]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"constraint list must be a list[VariableConstraint], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator(
        "request_constraints",
        "industry_constraints",
        "user_overrides",
        mode="after",
    )
    @classmethod
    def _validate_constraints(
        cls,
        value: list[VariableConstraint],
    ) -> list[VariableConstraint]:
        return _validate_constraint_list(value, field_name="constraint list")

    @field_validator(
        "user_confirmed_controllable_variables",
        "user_verified_variables",
        mode="before",
    )
    @classmethod
    def _validate_confirmation_before(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"confirmation list must be a list[str], got {type(value).__name__}"
            )
        return list(value)

    @field_validator(
        "user_confirmed_controllable_variables",
        "user_verified_variables",
        mode="after",
    )
    @classmethod
    def _validate_confirmation(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(value, field_name="confirmation list")

    @field_validator("max_simultaneous_changes", mode="before")
    @classmethod
    def _validate_max_simultaneous_changes(cls, value: object) -> int:
        return _require_strict_int_ge(
            value,
            field_name="max_simultaneous_changes",
            minimum=1,
        )

    @field_validator("operating_point_selection", mode="before")
    @classmethod
    def _validate_operating_mode(
        cls,
        value: object,
    ) -> OperatingPointSelectionMode:
        if isinstance(value, OperatingPointSelectionMode):
            return value
        if isinstance(value, str):
            try:
                return OperatingPointSelectionMode(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid OperatingPointSelectionMode: {value!r}"
                ) from exc
        raise ValueError(
            "operating_point_selection must be OperatingPointSelectionMode, "
            f"got {type(value).__name__}"
        )

    @field_validator("explicit_operating_row_id", mode="before")
    @classmethod
    def _validate_explicit_row_id(cls, value: object) -> int | str | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("explicit_operating_row_id must not be a bool")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return _require_non_empty_str(value, field_name="explicit_operating_row_id")
        raise ValueError(
            "explicit_operating_row_id must be int, str, or None, "
            f"got {type(value).__name__}"
        )

    @field_validator("model_performance_policy", mode="before")
    @classmethod
    def _validate_model_performance_policy(
        cls,
        value: object,
    ) -> ModelPerformanceAcceptancePolicy | None:
        if value is None:
            return None
        if isinstance(value, ModelPerformanceAcceptancePolicy):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return ModelPerformanceAcceptancePolicy.model_validate(value)
        raise ValueError(
            "model_performance_policy must be ModelPerformanceAcceptancePolicy "
            f"or None, got {type(value).__name__}"
        )

    @field_validator("cohort_filter", mode="before")
    @classmethod
    def _validate_cohort_filter(cls, value: object) -> NumericCohortFilter | None:
        if value is None:
            return None
        if isinstance(value, NumericCohortFilter):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return NumericCohortFilter.model_validate(value)
        raise ValueError(
            "cohort_filter must be NumericCohortFilter or None, "
            f"got {type(value).__name__}"
        )

    @field_validator("anomaly_recommendation", mode="before")
    @classmethod
    def _validate_anomaly_recommendation(
        cls,
        value: object,
    ) -> AnomalyRecommendationConfig:
        if value is None:
            return AnomalyRecommendationConfig()
        if isinstance(value, AnomalyRecommendationConfig):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return AnomalyRecommendationConfig.model_validate(value)
        raise ValueError(
            "anomaly_recommendation must be AnomalyRecommendationConfig, "
            f"got {type(value).__name__}"
        )

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> dict[str, ScalarMetadataValue]:
        if value is None:
            return {}
        return _validate_scalar_metadata(value)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> Self:
        feature_set = set(self.feature_columns)
        identifier_set = set(self.identifier_columns)
        excluded_set = set(self.excluded_columns)
        anomaly_recommendation_enabled = self.anomaly_recommendation.enabled

        if self.analysis_mode is AnalysisExecutionMode.ANOMALY_ONLY:
            if self.target_column is not None:
                raise ValueError(
                    "target_column must be None when analysis_mode is ANOMALY_ONLY"
                )
            if self.requested_task is not None:
                raise ValueError(
                    "requested_task must be None when analysis_mode is ANOMALY_ONLY"
                )
            if self.model_performance_policy is not None:
                raise ValueError(
                    "model_performance_policy must be None when analysis_mode "
                    "is ANOMALY_ONLY"
                )
            if self.quality_direction is not None:
                raise ValueError(
                    "quality_direction must be None when analysis_mode "
                    "is ANOMALY_ONLY"
                )
            if self.quality_target is not None:
                raise ValueError(
                    "quality_target must be None when analysis_mode is ANOMALY_ONLY"
                )
            if (
                self.declared_target_minimum is not None
                or self.declared_target_maximum is not None
            ):
                raise ValueError(
                    "declared_target_minimum and declared_target_maximum must be "
                    "None when analysis_mode is ANOMALY_ONLY"
                )
            if anomaly_recommendation_enabled:
                if self.objective is not RecommendationObjective.REDUCE_ANOMALY_SCORE:
                    raise ValueError(
                        "objective must be REDUCE_ANOMALY_SCORE when "
                        "anomaly_recommendation.enabled is True"
                    )
            elif self.objective is not None:
                raise ValueError(
                    "objective must be None when analysis_mode is ANOMALY_ONLY "
                    "and anomaly_recommendation.enabled is False"
                )
        else:
            if anomaly_recommendation_enabled:
                raise ValueError(
                    "anomaly_recommendation.enabled must be False when "
                    "analysis_mode is SUPERVISED"
                )
            if self.cohort_filter is not None:
                raise ValueError(
                    "cohort_filter is only supported when analysis_mode is "
                    "ANOMALY_ONLY"
                )
            if self.target_column is None:
                raise ValueError(
                    "target_column is required when analysis_mode is SUPERVISED"
                )
            if self.model_performance_policy is None:
                raise ValueError(
                    "model_performance_policy is required when analysis_mode "
                    "is SUPERVISED"
                )
            if self.objective is None:
                raise ValueError(
                    "objective is required when analysis_mode is SUPERVISED"
                )

        if self.target_column is not None and self.target_column in feature_set:
            raise ValueError(
                "feature_columns must not include target_column "
                f"({self.target_column!r})"
            )

        if identifier_set & excluded_set:
            raise ValueError(
                "identifier_columns and excluded_columns must be disjoint: "
                f"{sorted(identifier_set & excluded_set)}"
            )
        if self.target_column is not None and (
            self.target_column in identifier_set or self.target_column in excluded_set
        ):
            raise ValueError(
                "target_column must not appear in identifier_columns or "
                "excluded_columns"
            )
        overlap_features = feature_set & (identifier_set | excluded_set)
        if overlap_features:
            raise ValueError(
                "feature_columns must not overlap identifier_columns or "
                f"excluded_columns: {sorted(overlap_features)}"
            )

        if self.timestamp_column is not None:
            if (
                self.target_column is not None
                and self.timestamp_column == self.target_column
            ):
                raise ValueError("timestamp_column must not equal target_column")
            if self.timestamp_column in feature_set:
                raise ValueError("timestamp_column must not appear in feature_columns")
            if self.timestamp_column in identifier_set:
                raise ValueError(
                    "timestamp_column must not appear in identifier_columns"
                )
            if self.timestamp_column in excluded_set:
                raise ValueError("timestamp_column must not appear in excluded_columns")

        for name, role in self.column_role_overrides.items():
            if (
                self.target_column is not None
                and name == self.target_column
                and role is not ColumnRole.TARGET_QUALITY
            ):
                raise ValueError(
                    "target_column override must use ColumnRole.TARGET_QUALITY"
                )

        if self.objective is not None:
            needs_quality = self.objective in {
                RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
                RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
            }
            if needs_quality and self.quality_direction is None:
                raise ValueError(
                    "quality_direction is required for IMPROVE_PREDICTED_QUALITY "
                    "and BALANCE_QUALITY_AND_ANOMALY objectives"
                )

        if self.quality_direction is QualityOptimizationDirection.TARGET:
            if self.quality_target is None:
                raise ValueError(
                    "quality_target is required when quality_direction is TARGET"
                )
        elif self.quality_target is not None:
            raise ValueError(
                "quality_target must be None unless quality_direction is TARGET"
            )

        if (self.declared_target_minimum is None) ^ (
            self.declared_target_maximum is None
        ):
            raise ValueError(
                "declared_target_minimum and declared_target_maximum must both "
                "be set or both None"
            )
        if (
            self.declared_target_minimum is not None
            and self.declared_target_maximum is not None
            and self.declared_target_minimum > self.declared_target_maximum
        ):
            raise ValueError(
                "declared_target_minimum must be <= declared_target_maximum "
                f"(got {self.declared_target_minimum} > "
                f"{self.declared_target_maximum})"
            )

        for field_name, constraints in (
            ("request_constraints", self.request_constraints),
            ("industry_constraints", self.industry_constraints),
            ("user_overrides", self.user_overrides),
        ):
            for item in constraints:
                if item.variable not in feature_set:
                    raise ValueError(
                        f"{field_name} variable {item.variable!r} must exist in "
                        "feature_columns"
                    )

        for field_name, names in (
            (
                "user_confirmed_controllable_variables",
                self.user_confirmed_controllable_variables,
            ),
            ("user_verified_variables", self.user_verified_variables),
        ):
            missing = [name for name in names if name not in feature_set]
            if missing:
                raise ValueError(
                    f"{field_name} contains variables absent from feature_columns: "
                    f"{missing}"
                )

        if self.max_simultaneous_changes > len(self.feature_columns):
            raise ValueError(
                "max_simultaneous_changes must be <= len(feature_columns) "
                f"(got {self.max_simultaneous_changes} > {len(self.feature_columns)})"
            )

        if (
            self.operating_point_selection
            is OperatingPointSelectionMode.EXPLICIT_ROW_ID
        ):
            if self.explicit_operating_row_id is None:
                raise ValueError(
                    "explicit_operating_row_id is required when "
                    "operating_point_selection is EXPLICIT_ROW_ID"
                )
        elif self.explicit_operating_row_id is not None:
            raise ValueError(
                "explicit_operating_row_id must be None unless "
                "operating_point_selection is EXPLICIT_ROW_ID"
            )

        return self


class AnomalyContextValue(BaseModel):
    """One original feature value inside an anomaly context row."""

    feature_name: str
    value: ScalarMetadataValue = None

    @field_validator("feature_name", mode="before")
    @classmethod
    def _validate_feature_name(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="feature_name")

    @field_validator("value", mode="before")
    @classmethod
    def _validate_value(cls, value: object) -> ScalarMetadataValue:
        return _require_json_safe_scalar(value, field_name="value")


class AnomalyContextIdentifierValue(BaseModel):
    """One original identifier value inside an anomaly context row."""

    column_name: str
    value: ScalarMetadataValue = None

    @field_validator("column_name", mode="before")
    @classmethod
    def _validate_column_name(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="column_name")

    @field_validator("value", mode="before")
    @classmethod
    def _validate_value(cls, value: object) -> ScalarMetadataValue:
        return _require_json_safe_scalar(value, field_name="value")


class AnomalyContextRow(BaseModel):
    """One analysis-order row inside an anomaly context window."""

    analysis_position: int
    original_row_id: int | str
    relative_offset: int
    is_center_event: bool
    is_selected_anomaly_event: bool
    identifier_values: list[AnomalyContextIdentifierValue] = Field(
        default_factory=list
    )
    timestamp_value: ScalarMetadataValue = None
    feature_values: list[AnomalyContextValue] = Field(default_factory=list)

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
        return _require_strict_int(value, field_name="relative_offset")

    @field_validator("is_center_event", "is_selected_anomaly_event", mode="before")
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="context row bool field")

    @field_validator("identifier_values", mode="before")
    @classmethod
    def _validate_identifier_values_before(
        cls,
        value: object,
    ) -> list[AnomalyContextIdentifierValue]:
        if value is None:
            return []
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list):
            raise ValueError(
                "identifier_values must be a list[AnomalyContextIdentifierValue], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("identifier_values", mode="after")
    @classmethod
    def _validate_identifier_values(
        cls,
        value: list[AnomalyContextIdentifierValue],
    ) -> list[AnomalyContextIdentifierValue]:
        seen: set[str] = set()
        copied: list[AnomalyContextIdentifierValue] = []
        for item in value:
            if not isinstance(item, AnomalyContextIdentifierValue):
                raise ValueError(
                    "identifier_values entries must be AnomalyContextIdentifierValue, "
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
        return _require_json_safe_scalar(value, field_name="timestamp_value")

    @field_validator("feature_values", mode="before")
    @classmethod
    def _validate_feature_values_before(
        cls,
        value: object,
    ) -> list[AnomalyContextValue]:
        if value is None:
            return []
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list):
            raise ValueError(
                "feature_values must be a list[AnomalyContextValue], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("feature_values", mode="after")
    @classmethod
    def _validate_feature_values(
        cls,
        value: list[AnomalyContextValue],
    ) -> list[AnomalyContextValue]:
        seen: set[str] = set()
        copied: list[AnomalyContextValue] = []
        for item in value:
            if not isinstance(item, AnomalyContextValue):
                raise ValueError(
                    "feature_values entries must be AnomalyContextValue, "
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
            raise ValueError(
                "relative_offset=0 requires is_center_event=True"
            )
        return self


class AnomalyContextWindow(BaseModel):
    """Compact analysis-order context window for one selected anomaly event."""

    event_rank: int
    center_original_row_id: int | str
    center_anomaly_score: float
    radius: int = ANOMALY_CONTEXT_RADIUS
    order_basis: AnomalyContextOrderBasis
    feature_names: list[str] = Field(default_factory=list)
    rows: list[AnomalyContextRow] = Field(default_factory=list)

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
        number = _require_strict_int_ge(value, field_name="radius", minimum=0)
        if number != ANOMALY_CONTEXT_RADIUS:
            raise ValueError(
                f"radius must equal ANOMALY_CONTEXT_RADIUS "
                f"({ANOMALY_CONTEXT_RADIUS}), got {number}"
            )
        return number

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
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list):
            raise ValueError(
                f"feature_names must be a list[str], got {type(value).__name__}"
            )
        return list(value)

    @field_validator("feature_names", mode="after")
    @classmethod
    def _validate_feature_names(cls, value: list[str]) -> list[str]:
        cleaned = _validate_unique_non_empty_strings(value, field_name="feature_names")
        if len(cleaned) > ANOMALY_CONTEXT_MAX_FEATURES:
            raise ValueError(
                "feature_names must contain at most "
                f"{ANOMALY_CONTEXT_MAX_FEATURES} features, got {len(cleaned)}"
            )
        return cleaned

    @field_validator("rows", mode="before")
    @classmethod
    def _validate_rows_before(cls, value: object) -> list[AnomalyContextRow]:
        if value is None:
            return []
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list):
            raise ValueError(
                f"rows must be a list[AnomalyContextRow], got {type(value).__name__}"
            )
        return list(value)

    @field_validator("rows", mode="after")
    @classmethod
    def _validate_rows(cls, value: list[AnomalyContextRow]) -> list[AnomalyContextRow]:
        max_rows = (2 * ANOMALY_CONTEXT_RADIUS) + 1
        if len(value) > max_rows:
            raise ValueError(
                f"rows must contain at most {max_rows} entries, got {len(value)}"
            )
        copied: list[AnomalyContextRow] = []
        for item in value:
            if not isinstance(item, AnomalyContextRow):
                raise ValueError(
                    "rows entries must be AnomalyContextRow, "
                    f"got {type(item).__name__}"
                )
            copied.append(item.model_copy(deep=True))
        return copied

    @model_validator(mode="after")
    def _validate_window_consistency(self) -> Self:
        center_rows = [row for row in self.rows if row.is_center_event]
        if self.rows and len(center_rows) != 1:
            raise ValueError(
                "rows must contain exactly one center event when non-empty, "
                f"got {len(center_rows)}"
            )
        if center_rows:
            center = center_rows[0]
            if center.original_row_id != self.center_original_row_id:
                raise ValueError(
                    "center row original_row_id must equal center_original_row_id"
                )
            if center.relative_offset != 0:
                raise ValueError("center row relative_offset must be 0")
        for row in self.rows:
            if abs(row.relative_offset) > self.radius:
                raise ValueError(
                    "row relative_offset must be within "
                    f"[{-self.radius}, {self.radius}], got {row.relative_offset}"
                )
            row_feature_names = [item.feature_name for item in row.feature_values]
            if row_feature_names != self.feature_names:
                raise ValueError(
                    "each row feature_values order must match window feature_names"
                )
        return self


class AnalysisWorkflowStageRecord(BaseModel):
    """Per-stage execution record for one analysis workflow run.

    Captures whether the stage executed, succeeded, or ended in a structured
    refusal, plus optional row counts and scalar metadata. Does not store
    DataFrames or estimators.
    """

    stage: AnalysisWorkflowStage
    executed: bool
    succeeded: bool
    structured_refusal: bool
    row_count: int | None = None
    message: str
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)

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
        return _require_strict_bool(value, field_name="stage record bool field")

    @field_validator("row_count", mode="before")
    @classmethod
    def _validate_row_count(cls, value: object) -> int | None:
        if value is None:
            return None
        return _require_strict_int_ge0(value, field_name="row_count")

    @field_validator("message", mode="before")
    @classmethod
    def _validate_message(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="message")

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
        return self


class AnalysisWorkflowReport(BaseModel):
    """Typed end-to-end report for one industrial analysis workflow run.

    Aggregates stage records, selected routing and model keys, partition counts,
    optional model-performance assessment, optional final recommendation,
    timings, and scalar metadata. Does not embed DataFrames or estimators.
    """

    status: AnalysisWorkflowStatus
    terminal_stage: AnalysisWorkflowStage
    stage_records: list[AnalysisWorkflowStageRecord]
    analysis_mode: AnalysisExecutionMode = AnalysisExecutionMode.SUPERVISED
    model_performance_assessment: ModelPerformanceAcceptanceReport | None = None
    final_recommendation: RecommendationResult | None = None
    recommendation_verification: RecommendationWhatIfVerificationResult | None = None
    selected_industry: str | None = None
    selected_task: AnalysisTask | None = None
    inferred_task: AnalysisTask | None = None
    task_selection_source: TaskSelectionSource | None = None
    task_override_applied: bool = False
    selected_supervised_model_key: str | None = None
    selected_anomaly_model_key: str | None = None
    selected_operating_row_id: int | str | None = None
    anomaly_event_count: int
    diagnosis_factor_count: int
    anomaly_events: list[AnomalyEvent] = Field(default_factory=list)
    diagnosis_factors: list[RootCauseFactor] = Field(default_factory=list)
    anomaly_context_windows: list[AnomalyContextWindow] = Field(default_factory=list)
    raw_row_count: int
    processed_row_count: int
    cohort_row_count: int
    train_row_count: int
    validation_row_count: int
    test_row_count: int
    cohort_filter_summary: CohortFilterSummary
    dataset_fingerprint: str | None = None
    started_at: datetime
    completed_at: datetime
    total_seconds: float
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)

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

    @field_validator("analysis_mode", mode="before")
    @classmethod
    def _validate_analysis_mode(cls, value: object) -> AnalysisExecutionMode:
        if isinstance(value, AnalysisExecutionMode):
            return value
        if isinstance(value, str):
            try:
                return AnalysisExecutionMode(value)
            except ValueError as exc:
                raise ValueError(f"invalid AnalysisExecutionMode: {value!r}") from exc
        raise ValueError(
            "analysis_mode must be AnalysisExecutionMode, "
            f"got {type(value).__name__}"
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

    @field_validator("stage_records", mode="before")
    @classmethod
    def _validate_stage_records_before(
        cls,
        value: object,
    ) -> list[AnalysisWorkflowStageRecord]:
        if not isinstance(value, list):
            raise ValueError(
                f"stage_records must be a list[AnalysisWorkflowStageRecord], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("stage_records", mode="after")
    @classmethod
    def _validate_stage_records(
        cls,
        value: list[AnalysisWorkflowStageRecord],
    ) -> list[AnalysisWorkflowStageRecord]:
        if not value:
            raise ValueError("stage_records must contain at least one record")
        seen: set[AnalysisWorkflowStage] = set()
        copied: list[AnalysisWorkflowStageRecord] = []
        for item in value:
            if not isinstance(item, AnalysisWorkflowStageRecord):
                raise ValueError(
                    "stage_records entries must be AnalysisWorkflowStageRecord, "
                    f"got {type(item).__name__}"
                )
            if item.stage in seen:
                raise ValueError(
                    f"stage_records must not contain duplicate stages: {item.stage!r}"
                )
            seen.add(item.stage)
            copied.append(item.model_copy(deep=True))

        stage_order = {stage: index for index, stage in enumerate(_CANONICAL_STAGES)}
        for earlier, later in zip(copied, copied[1:], strict=False):
            if stage_order[earlier.stage] >= stage_order[later.stage]:
                raise ValueError(
                    "stage_records must follow AnalysisWorkflowStage canonical order"
                )

        if copied[0].stage is not AnalysisWorkflowStage.LOAD:
            raise ValueError("first stage_records entry must be LOAD")

        executed = [record for record in copied if record.executed]
        if not executed:
            raise ValueError("at least one stage_records entry must be executed")

        # Executed stages may have explicit skipped (executed=False) records
        # between them. Every canonical stage from LOAD through the last
        # executed stage must appear in the records.
        last_executed_index = stage_order[executed[-1].stage]
        present = {record.stage for record in copied}
        for stage in _CANONICAL_STAGES[: last_executed_index + 1]:
            if stage not in present:
                raise ValueError(
                    "stage_records must include every canonical stage from LOAD "
                    f"through the last executed stage (missing {stage!r})"
                )
        return copied

    @field_validator("model_performance_assessment", mode="before")
    @classmethod
    def _validate_model_performance_assessment(
        cls,
        value: object,
    ) -> ModelPerformanceAcceptanceReport | None:
        if value is None:
            return None
        if isinstance(value, ModelPerformanceAcceptanceReport):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return ModelPerformanceAcceptanceReport.model_validate(value)
        raise ValueError(
            "model_performance_assessment must be "
            "ModelPerformanceAcceptanceReport or None, "
            f"got {type(value).__name__}"
        )

    @field_validator("final_recommendation", mode="before")
    @classmethod
    def _validate_final_recommendation(
        cls,
        value: object,
    ) -> RecommendationResult | None:
        if value is None:
            return None
        if isinstance(value, RecommendationResult):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return RecommendationResult.model_validate(value)
        raise ValueError(
            f"final_recommendation must be RecommendationResult or None, "
            f"got {type(value).__name__}"
        )

    @field_validator("recommendation_verification", mode="before")
    @classmethod
    def _validate_recommendation_verification(
        cls,
        value: object,
    ) -> RecommendationWhatIfVerificationResult | None:
        if value is None:
            return None
        if isinstance(value, RecommendationWhatIfVerificationResult):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return RecommendationWhatIfVerificationResult.model_validate(value)
        raise ValueError(
            "recommendation_verification must be "
            "RecommendationWhatIfVerificationResult or None, "
            f"got {type(value).__name__}"
        )

    @field_validator("selected_industry", "selected_supervised_model_key",
                     "selected_anomaly_model_key", mode="before")
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

    @field_validator("inferred_task", mode="before")
    @classmethod
    def _validate_inferred_task(cls, value: object) -> AnalysisTask | None:
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
            f"inferred_task must be AnalysisTask or None, got {type(value).__name__}"
        )

    @field_validator("task_selection_source", mode="before")
    @classmethod
    def _validate_task_selection_source(
        cls,
        value: object,
    ) -> TaskSelectionSource | None:
        if value is None:
            return None
        if isinstance(value, TaskSelectionSource):
            return value
        if isinstance(value, str):
            try:
                return TaskSelectionSource(value)
            except ValueError as exc:
                raise ValueError(f"invalid TaskSelectionSource: {value!r}") from exc
        raise ValueError(
            "task_selection_source must be TaskSelectionSource or None, "
            f"got {type(value).__name__}"
        )

    @field_validator("task_override_applied", mode="before")
    @classmethod
    def _validate_task_override_applied(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="task_override_applied")

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

    @field_validator(
        "anomaly_event_count",
        "diagnosis_factor_count",
        "raw_row_count",
        "processed_row_count",
        "cohort_row_count",
        "train_row_count",
        "validation_row_count",
        "test_row_count",
        mode="before",
    )
    @classmethod
    def _validate_counts(cls, value: object) -> int:
        return _require_strict_int_ge0(value, field_name="count field")

    @field_validator("cohort_filter_summary", mode="before")
    @classmethod
    def _validate_cohort_filter_summary(cls, value: object) -> CohortFilterSummary:
        if isinstance(value, CohortFilterSummary):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return CohortFilterSummary.model_validate(value)
        raise ValueError(
            "cohort_filter_summary must be CohortFilterSummary, "
            f"got {type(value).__name__}"
        )

    @field_validator("dataset_fingerprint", mode="before")
    @classmethod
    def _validate_dataset_fingerprint(cls, value: object) -> str | None:
        return normalize_optional_dataset_fingerprint(value)

    @field_validator("anomaly_events", mode="before")
    @classmethod
    def _validate_anomaly_events_before(cls, value: object) -> list[AnomalyEvent]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"anomaly_events must be a list[AnomalyEvent], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("anomaly_events", mode="after")
    @classmethod
    def _validate_anomaly_events(cls, value: list[AnomalyEvent]) -> list[AnomalyEvent]:
        seen_ids: set[str] = set()
        copied: list[AnomalyEvent] = []
        for event in value:
            if not isinstance(event, AnomalyEvent):
                raise ValueError(
                    "anomaly_events entries must be AnomalyEvent, "
                    f"got {type(event).__name__}"
                )
            if event.anomaly_id in seen_ids:
                raise ValueError(
                    "anomaly_events must not contain duplicate anomaly_id values: "
                    f"{event.anomaly_id!r}"
                )
            seen_ids.add(event.anomaly_id)
            copied.append(event.model_copy(deep=True))
        return copied

    @field_validator("diagnosis_factors", mode="before")
    @classmethod
    def _validate_diagnosis_factors_before(
        cls,
        value: object,
    ) -> list[RootCauseFactor]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"diagnosis_factors must be a list[RootCauseFactor], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("diagnosis_factors", mode="after")
    @classmethod
    def _validate_diagnosis_factors(
        cls,
        value: list[RootCauseFactor],
    ) -> list[RootCauseFactor]:
        seen_variables: set[str] = set()
        copied: list[RootCauseFactor] = []
        for factor in value:
            if not isinstance(factor, RootCauseFactor):
                raise ValueError(
                    "diagnosis_factors entries must be RootCauseFactor, "
                    f"got {type(factor).__name__}"
                )
            if factor.variable in seen_variables:
                raise ValueError(
                    "diagnosis_factors must not contain duplicate variables: "
                    f"{factor.variable!r}"
                )
            seen_variables.add(factor.variable)
            copied.append(factor.model_copy(deep=True))
        return copied

    @field_validator("anomaly_context_windows", mode="before")
    @classmethod
    def _validate_anomaly_context_windows_before(
        cls,
        value: object,
    ) -> list[AnomalyContextWindow]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                "anomaly_context_windows must be a list[AnomalyContextWindow], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("anomaly_context_windows", mode="after")
    @classmethod
    def _validate_anomaly_context_windows(
        cls,
        value: list[AnomalyContextWindow],
    ) -> list[AnomalyContextWindow]:
        seen_ranks: set[int] = set()
        copied: list[AnomalyContextWindow] = []
        for window in value:
            if not isinstance(window, AnomalyContextWindow):
                raise ValueError(
                    "anomaly_context_windows entries must be AnomalyContextWindow, "
                    f"got {type(window).__name__}"
                )
            if window.event_rank in seen_ranks:
                raise ValueError(
                    "anomaly_context_windows must not contain duplicate event_rank: "
                    f"{window.event_rank!r}"
                )
            seen_ranks.add(window.event_rank)
            copied.append(window.model_copy(deep=True))
        return copied

    @field_validator("started_at", "completed_at", mode="before")
    @classmethod
    def _validate_datetimes(cls, value: object) -> datetime:
        if not isinstance(value, datetime):
            raise ValueError(
                f"datetime fields must be datetime, got {type(value).__name__}"
            )
        return _require_timezone_aware(value, field_name="datetime field")

    @field_validator("total_seconds", mode="before")
    @classmethod
    def _validate_total_seconds(cls, value: object) -> float:
        return _require_non_negative_finite_float(value, field_name="total_seconds")

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
    def _validate_report_consistency(self) -> Self:
        executed = [record for record in self.stage_records if record.executed]
        last_executed = executed[-1]
        if self.terminal_stage is not last_executed.stage:
            raise ValueError(
                "terminal_stage must equal the last executed stage "
                f"(expected {last_executed.stage!r}, got {self.terminal_stage!r})"
            )

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
        if self.cohort_filter_summary.retained_row_count != self.cohort_row_count:
            raise ValueError(
                "cohort_filter_summary.retained_row_count must equal "
                "cohort_row_count "
                f"(got {self.cohort_filter_summary.retained_row_count} != "
                f"{self.cohort_row_count})"
            )

        split_executed = any(
            record.stage is AnalysisWorkflowStage.SPLIT and record.executed
            for record in self.stage_records
        )
        if split_executed:
            split_total = (
                self.train_row_count
                + self.validation_row_count
                + self.test_row_count
            )
            if split_total != self.cohort_row_count:
                raise ValueError(
                    "train_row_count + validation_row_count + test_row_count "
                    "must equal cohort_row_count when SPLIT executed "
                    f"(got {split_total} != {self.cohort_row_count})"
                )

        if self.completed_at < self.started_at:
            raise ValueError("completed_at must be >= started_at")

        supervised_final_executed = any(
            record.stage is AnalysisWorkflowStage.SUPERVISED_FINAL_EVALUATION
            and record.executed
            for record in self.stage_records
        )
        if supervised_final_executed:
            if self.model_performance_assessment is None:
                raise ValueError(
                    "model_performance_assessment is required when "
                    "SUPERVISED_FINAL_EVALUATION executed"
                )
        elif self.model_performance_assessment is not None:
            raise ValueError(
                "model_performance_assessment must be None when "
                "SUPERVISED_FINAL_EVALUATION did not execute"
            )

        if self.status is AnalysisWorkflowStatus.COMPLETED:
            if self.terminal_stage not in {
                AnalysisWorkflowStage.RECOMMENDATION,
                AnalysisWorkflowStage.WHAT_IF_VERIFICATION,
            }:
                raise ValueError(
                    "COMPLETED status requires terminal_stage RECOMMENDATION "
                    "or WHAT_IF_VERIFICATION"
                )
            if self.final_recommendation is None:
                raise ValueError(
                    "COMPLETED status requires final_recommendation"
                )
            if self.final_recommendation.status is not RecommendationStatus.GENERATED:
                raise ValueError(
                    "COMPLETED status requires final_recommendation.status GENERATED"
                )
        elif self.status is AnalysisWorkflowStatus.PARTIAL:
            if self.final_recommendation is not None:
                if (
                    self.final_recommendation.status
                    is not RecommendationStatus.READY_FOR_OPTIMIZATION
                ):
                    raise ValueError(
                        "PARTIAL with final_recommendation requires "
                        "READY_FOR_OPTIMIZATION status"
                    )
            elif self.terminal_stage is AnalysisWorkflowStage.RECOMMENDATION:
                raise ValueError(
                    "PARTIAL without final_recommendation cannot terminate at "
                    "RECOMMENDATION"
                )
            elif self.terminal_stage is AnalysisWorkflowStage.WHAT_IF_VERIFICATION:
                raise ValueError(
                    "PARTIAL without final_recommendation cannot terminate at "
                    "WHAT_IF_VERIFICATION"
                )
        elif self.status is AnalysisWorkflowStatus.REFUSED:
            if self.final_recommendation is not None:
                if self.final_recommendation.status is not RecommendationStatus.REFUSED:
                    raise ValueError(
                        "REFUSED with final_recommendation requires "
                        "RecommendationStatus.REFUSED"
                    )
            refusal_present = any(
                record.structured_refusal for record in self.stage_records
            )
            if (
                not refusal_present
                and self.final_recommendation is None
                and last_executed.succeeded
            ):
                raise ValueError(
                    "REFUSED status requires a structured refusal stage or a "
                    "REFUSED final_recommendation"
                )

        return self


@dataclass(frozen=True, slots=True)
class AnalysisWorkflowOutcome:
    """Immutable container for a completed analysis workflow report."""

    report: AnalysisWorkflowReport
