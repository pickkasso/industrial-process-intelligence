"""Pydantic input DTOs for the Streamlit MVP UI (Step 11B).

These models capture user-facing workflow configuration before conversion into
``AnalysisWorkflowRequest``. They do not store Path, UploadedFile, DataFrame,
model, or estimator objects.
"""

from __future__ import annotations

import math
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from process_intelligence.core.enums import AnalysisTask, ColumnRole
from process_intelligence.evaluation.performance_acceptance import (
    MetricAcceptanceDirection,
)
from process_intelligence.recommendation import (
    QualityOptimizationDirection,
    RecommendationObjective,
)
from process_intelligence.workflow.enums import (
    AnalysisExecutionMode,
    OperatingPointSelectionMode,
)
from process_intelligence.workflow.schemas import NumericCohortFilter

ScalarMetadataValue = str | int | float | bool | None
"""Allowed scalar types for UI submission metadata dictionaries."""

_ORIGINAL_ROW_ID = "_original_row_id"


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


class UiMetricRuleInput(BaseModel):
    """UI input DTO for one independent-test metric acceptance rule.

    Direction is never inferred from ``metric_name``. Convert to
    ``MetricAcceptanceRule`` only inside the request builder.
    """

    model_config = ConfigDict(extra="forbid")

    metric_name: str
    direction: MetricAcceptanceDirection
    threshold: float
    required: bool = True

    @field_validator("metric_name", mode="before")
    @classmethod
    def _validate_metric_name(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="metric_name")

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

    @field_validator("threshold", mode="before")
    @classmethod
    def _validate_threshold(cls, value: object) -> float:
        return _require_finite_float(value, field_name="threshold")

    @field_validator("required", mode="before")
    @classmethod
    def _validate_required(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="required")


class UiVariableConstraintInput(BaseModel):
    """UI input DTO for one process-variable numeric constraint range.

    Does not infer bounds from the dataset. Convert to ``VariableConstraint``
    only inside the request builder.
    """

    model_config = ConfigDict(extra="forbid")

    variable: str
    minimum: float
    maximum: float

    @field_validator("variable", mode="before")
    @classmethod
    def _validate_variable(cls, value: object) -> str:
        return _validate_column_name(value, field_name="variable")

    @field_validator("minimum", "maximum", mode="before")
    @classmethod
    def _validate_bounds(cls, value: object) -> float:
        return _require_finite_float(value, field_name="constraint bound")

    @model_validator(mode="after")
    def _validate_range(self) -> Self:
        if self.minimum >= self.maximum:
            raise ValueError(
                "minimum must be < maximum "
                f"(got minimum={self.minimum}, maximum={self.maximum})"
            )
        return self


class WorkflowUiSubmission(BaseModel):
    """Validated UI submission for one analysis workflow run.

    Mirrors ``AnalysisWorkflowRequest`` relationship rules without embedding a
    CSV path, uploaded file, DataFrame, or model object.
    """

    model_config = ConfigDict(extra="forbid")

    analysis_mode: AnalysisExecutionMode = AnalysisExecutionMode.SUPERVISED
    target_column: str | None = None
    feature_columns: list[str]
    timestamp_column: str | None = None
    identifier_columns: list[str] = Field(default_factory=list)
    excluded_columns: list[str] = Field(default_factory=list)
    column_role_overrides: dict[str, ColumnRole] = Field(default_factory=dict)
    requested_task: AnalysisTask | None = None
    objective: RecommendationObjective | None = None
    quality_direction: QualityOptimizationDirection | None = None
    quality_target: float | None = None
    performance_rules: list[UiMetricRuleInput] = Field(default_factory=list)
    constraints: list[UiVariableConstraintInput] = Field(default_factory=list)
    user_confirmed_controllable_variables: list[str] = Field(default_factory=list)
    user_verified_variables: list[str] = Field(default_factory=list)
    max_simultaneous_changes: int = 3
    operating_point_selection: OperatingPointSelectionMode = (
        OperatingPointSelectionMode.TOP_RESIDUAL_ANOMALY
    )
    explicit_operating_row_id: int | str | None = None
    cohort_filter: NumericCohortFilter | None = None
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)

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

    @field_validator("performance_rules", mode="before")
    @classmethod
    def _validate_performance_rules_before(
        cls,
        value: object,
    ) -> list[UiMetricRuleInput]:
        if not isinstance(value, list):
            raise ValueError(
                f"performance_rules must be a list[UiMetricRuleInput], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("performance_rules", mode="after")
    @classmethod
    def _validate_performance_rules(
        cls,
        value: list[UiMetricRuleInput],
    ) -> list[UiMetricRuleInput]:
        seen: set[str] = set()
        copied: list[UiMetricRuleInput] = []
        for item in value:
            if not isinstance(item, UiMetricRuleInput):
                raise ValueError(
                    "performance_rules entries must be UiMetricRuleInput, "
                    f"got {type(item).__name__}"
                )
            if item.metric_name in seen:
                raise ValueError(
                    "performance_rules must not contain duplicate metric_name: "
                    f"{item.metric_name!r}"
                )
            seen.add(item.metric_name)
            copied.append(item.model_copy(deep=True))
        return copied

    @field_validator("constraints", mode="before")
    @classmethod
    def _validate_constraints_before(
        cls,
        value: object,
    ) -> list[UiVariableConstraintInput]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"constraints must be a list[UiVariableConstraintInput], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("constraints", mode="after")
    @classmethod
    def _validate_constraints(
        cls,
        value: list[UiVariableConstraintInput],
    ) -> list[UiVariableConstraintInput]:
        seen: set[str] = set()
        copied: list[UiVariableConstraintInput] = []
        for item in value:
            if not isinstance(item, UiVariableConstraintInput):
                raise ValueError(
                    "constraints entries must be UiVariableConstraintInput, "
                    f"got {type(item).__name__}"
                )
            if item.variable in seen:
                raise ValueError(
                    "constraints must not contain duplicate variables: "
                    f"{item.variable!r}"
                )
            seen.add(item.variable)
            copied.append(item.model_copy(deep=True))
        return copied

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

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> dict[str, ScalarMetadataValue]:
        if value is None:
            return {}
        return _validate_scalar_metadata(value)

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

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> Self:
        feature_set = set(self.feature_columns)
        identifier_set = set(self.identifier_columns)
        excluded_set = set(self.excluded_columns)

        if self.analysis_mode is AnalysisExecutionMode.ANOMALY_ONLY:
            if self.target_column is not None:
                raise ValueError(
                    "target_column must be None when analysis_mode is ANOMALY_ONLY"
                )
            if self.requested_task is not None:
                raise ValueError(
                    "requested_task must be None when analysis_mode is ANOMALY_ONLY"
                )
            if self.performance_rules:
                raise ValueError(
                    "performance_rules must be empty when analysis_mode "
                    "is ANOMALY_ONLY"
                )
            if self.objective is not None:
                raise ValueError(
                    "objective must be None when analysis_mode is ANOMALY_ONLY"
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
        else:
            if self.cohort_filter is not None:
                raise ValueError(
                    "cohort_filter is only supported when analysis_mode is "
                    "ANOMALY_ONLY"
                )
            if self.target_column is None:
                raise ValueError(
                    "target_column is required when analysis_mode is SUPERVISED"
                )
            if self.objective is None:
                raise ValueError(
                    "objective is required when analysis_mode is SUPERVISED"
                )
            if not self.performance_rules:
                raise ValueError(
                    "performance_rules must contain at least one rule when "
                    "analysis_mode is SUPERVISED"
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

        for item in self.constraints:
            if item.variable not in feature_set:
                raise ValueError(
                    f"constraints variable {item.variable!r} must exist in "
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


class StreamlitUiConfig(BaseModel):
    """Strict configuration for Streamlit page limits and presentation."""

    model_config = ConfigDict(extra="forbid")

    page_title: str = "Industrial Process Intelligence"
    page_icon: str = "🏭"
    maximum_upload_bytes: int = 50_000_000
    preview_row_count: int = 20
    maximum_warning_display: int = 100
    maximum_stage_metadata_items: int = 20

    @field_validator("page_title", "page_icon", mode="before")
    @classmethod
    def _validate_text_fields(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="config text field")

    @field_validator(
        "maximum_upload_bytes",
        "preview_row_count",
        "maximum_warning_display",
        "maximum_stage_metadata_items",
        mode="before",
    )
    @classmethod
    def _validate_positive_ints(cls, value: object) -> int:
        return _require_strict_int_ge(value, field_name="config int field", minimum=1)
