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
from process_intelligence.core.schemas import VariableConstraint
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
from process_intelligence.workflow.enums import (
    AnalysisWorkflowStage,
    AnalysisWorkflowStatus,
    OperatingPointSelectionMode,
)

ScalarMetadataValue = str | int | float | bool | None
"""Allowed scalar types for workflow metadata dictionaries."""

_ORIGINAL_ROW_ID = "_original_row_id"

_CANONICAL_STAGES: tuple[AnalysisWorkflowStage, ...] = tuple(AnalysisWorkflowStage)


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


class AnalysisWorkflowRequest(BaseModel):
    """Caller inputs for a single raw-CSV industrial analysis workflow run.

    Declares the CSV path, column roles, recommendation objective, constraints,
    operating-point selection mode, and explicit model-performance acceptance
    policy. Does not load data or fit models.
    """

    csv_path: Path
    target_column: str
    feature_columns: list[str]
    model_performance_policy: ModelPerformanceAcceptancePolicy
    timestamp_column: str | None = None
    identifier_columns: list[str] = Field(default_factory=list)
    excluded_columns: list[str] = Field(default_factory=list)
    column_role_overrides: dict[str, ColumnRole] = Field(default_factory=dict)
    objective: RecommendationObjective
    quality_direction: QualityOptimizationDirection | None = None
    quality_target: float | None = None
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

    @field_validator("target_column", mode="before")
    @classmethod
    def _validate_target_column(cls, value: object) -> str:
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
    ) -> ModelPerformanceAcceptancePolicy:
        if isinstance(value, ModelPerformanceAcceptancePolicy):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return ModelPerformanceAcceptancePolicy.model_validate(value)
        raise ValueError(
            "model_performance_policy must be ModelPerformanceAcceptancePolicy, "
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
        if self.target_column in feature_set:
            raise ValueError(
                "feature_columns must not include target_column "
                f"({self.target_column!r})"
            )

        identifier_set = set(self.identifier_columns)
        excluded_set = set(self.excluded_columns)
        if identifier_set & excluded_set:
            raise ValueError(
                "identifier_columns and excluded_columns must be disjoint: "
                f"{sorted(identifier_set & excluded_set)}"
            )
        if self.target_column in identifier_set or self.target_column in excluded_set:
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
            if self.timestamp_column == self.target_column:
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
            if name == self.target_column and role is not ColumnRole.TARGET_QUALITY:
                raise ValueError(
                    "target_column override must use ColumnRole.TARGET_QUALITY"
                )

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
    model_performance_assessment: ModelPerformanceAcceptanceReport | None = None
    final_recommendation: RecommendationResult | None = None
    selected_industry: str | None = None
    selected_task: AnalysisTask | None = None
    selected_supervised_model_key: str | None = None
    selected_anomaly_model_key: str | None = None
    selected_operating_row_id: int | str | None = None
    anomaly_event_count: int
    diagnosis_factor_count: int
    raw_row_count: int
    processed_row_count: int
    train_row_count: int
    validation_row_count: int
    test_row_count: int
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
        for index, record in enumerate(executed):
            expected = _CANONICAL_STAGES[index]
            if record.stage is not expected:
                raise ValueError(
                    "executed stages must form a contiguous prefix of the "
                    f"canonical order starting at LOAD (expected {expected!r}, "
                    f"got {record.stage!r})"
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
        "train_row_count",
        "validation_row_count",
        "test_row_count",
        mode="before",
    )
    @classmethod
    def _validate_counts(cls, value: object) -> int:
        return _require_strict_int_ge0(value, field_name="count field")

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
            if split_total != self.processed_row_count:
                raise ValueError(
                    "train_row_count + validation_row_count + test_row_count "
                    "must equal processed_row_count when SPLIT executed "
                    f"(got {split_total} != {self.processed_row_count})"
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
            if self.terminal_stage is not AnalysisWorkflowStage.RECOMMENDATION:
                raise ValueError(
                    "COMPLETED status requires terminal_stage RECOMMENDATION"
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
