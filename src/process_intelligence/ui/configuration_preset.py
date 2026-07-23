"""Analysis configuration preset import/export for the Streamlit UI.

Captures user-facing analysis settings as a JSON-safe immutable DTO. Does not
store CSV bytes, DataFrames, models, paths, workflow reports, or session dumps.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Self, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from process_intelligence.core.enums import AnalysisTask, ColumnRole
from process_intelligence.evaluation import MetricAcceptanceDirection
from process_intelligence.recommendation import (
    QualityOptimizationDirection,
    RecommendationObjective,
)
from process_intelligence.ui.schemas import (
    UiMetricRuleInput,
    UiVariableConstraintInput,
)
from process_intelligence.workflow.enums import (
    AnalysisExecutionMode,
    OperatingPointSelectionMode,
)
from process_intelligence.workflow.schemas import NumericCohortFilter

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


def _validate_column_name(value: object, *, field_name: str) -> str:
    text = _require_non_empty_str(value, field_name=field_name)
    if text == _ORIGINAL_ROW_ID:
        raise ValueError(
            f"{field_name} cannot be reserved column '{_ORIGINAL_ROW_ID}'"
        )
    return text


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

CONFIGURATION_PRESET_FILENAME = "process_intelligence_configuration.json"
CONFIGURATION_PRESET_SCHEMA_VERSION: Literal[1] = 1

# Session keys used by Streamlit widgets / preset staging.
SESSION_PRESET_PARSED_KEY = "ui_configuration_preset_parsed"
SESSION_PRESET_VALIDATION_MESSAGE_KEY = "ui_configuration_preset_validation_message"
SESSION_PRESET_VALIDATION_OK_KEY = "ui_configuration_preset_validation_ok"
SESSION_PRESET_MISSING_COLUMNS_KEY = "ui_configuration_preset_missing_columns"
SESSION_PRESET_APPLY_SUMMARY_KEY = "ui_configuration_preset_apply_summary"
SESSION_PRESET_PENDING_APPLY_KEY = "ui_configuration_preset_pending_apply"

WIDGET_STATE_KEY_PREFIXES: tuple[str, ...] = (
    "role_override_",
    "metric_name_",
    "metric_direction_",
    "metric_threshold_",
    "metric_required_",
    "constraint_min_",
    "constraint_max_",
    "anomaly_rec_role_",
    "anomaly_rec_constraint_min_",
    "anomaly_rec_constraint_max_",
)

SESSION_ANALYSIS_MODE_KEY = "ui_analysis_mode"
SESSION_TARGET_KEY = "ui_target_column"
SESSION_TIMESTAMP_KEY = "ui_timestamp_column"
SESSION_IDENTIFIERS_KEY = "ui_identifier_columns"
SESSION_EXCLUDED_KEY = "ui_excluded_columns"
SESSION_USE_RECOMMENDED_FEATURES_KEY = "ui_use_recommended_numeric_feature_set"
SESSION_EXPLICIT_FEATURES_KEY = "ui_explicit_feature_columns"
SESSION_ROLE_OVERRIDE_COLUMNS_KEY = "ui_role_override_columns"
SESSION_REQUESTED_TASK_KEY = "ui_requested_task"
SESSION_OBJECTIVE_KEY = "ui_recommendation_objective"
SESSION_QUALITY_DIRECTION_KEY = "ui_quality_direction"
SESSION_QUALITY_TARGET_KEY = "ui_quality_target"
SESSION_RULE_COUNT_KEY = "ui_performance_rule_count"
SESSION_CONSTRAINT_VARIABLES_KEY = "ui_recommendation_constraint_columns"
SESSION_CONFIRMED_CONTROLLABLE_KEY = "ui_confirmed_controllable_variables"
SESSION_VERIFIED_VARIABLES_KEY = "ui_verified_variables"
SESSION_MAX_CHANGES_KEY = "ui_maximum_simultaneous_changes"
SESSION_OPERATING_MODE_KEY = "ui_operating_point_selection_mode"
SESSION_EXPLICIT_ROW_ID_KEY = "ui_explicit_operating_row_id"
SESSION_RESTRICT_COHORT_KEY = "ui_restrict_operating_cohort"
SESSION_COHORT_COLUMN_KEY = "ui_cohort_filter_column"
SESSION_COHORT_LOWER_KEY = "ui_cohort_filter_lower"
SESSION_COHORT_UPPER_KEY = "ui_cohort_filter_upper"
SESSION_COHORT_INCLUDE_LOWER_KEY = "ui_cohort_filter_include_lower"
SESSION_COHORT_INCLUDE_UPPER_KEY = "ui_cohort_filter_include_upper"
SESSION_COHORT_EXCLUDE_FEATURE_KEY = "ui_cohort_filter_exclude_feature"
SESSION_ANOMALY_RECOMMENDATION_ENABLED_KEY = "anomaly_recommendation_enabled"
SESSION_ANOMALY_REVIEW_VARIABLES_KEY = "anomaly_recommendation_review_variables"
SESSION_ANOMALY_CONFIRMED_KEY = "anomaly_rec_confirmed_controllable"
SESSION_ANOMALY_VERIFIED_KEY = "anomaly_rec_verified_variables"
SESSION_ANOMALY_CONSTRAINT_VARIABLES_KEY = "anomaly_rec_constrained_variables"

_TARGET_PLACEHOLDER = "(select target)"
_TIMESTAMP_NONE = "(none)"
_ROLE_NONE = "(no override)"
_TASK_AUTO = "AUTO"
_OBJECTIVE_PLACEHOLDER = "(select objective)"
_QUALITY_DIRECTION_PLACEHOLDER = "(select quality direction)"
_DIRECTION_PLACEHOLDER = "(select direction)"
_COHORT_COLUMN_PLACEHOLDER = "(select cohort column)"

_FORBIDDEN_TYPE_NAMES = frozenset(
    {
        "DataFrame",
        "Series",
        "ndarray",
        "Path",
        "PosixPath",
        "WindowsPath",
        "UploadedFile",
        "bytes",
        "bytearray",
        "memoryview",
    }
)


def _reject_forbidden_runtime_object(value: object, *, field_name: str) -> None:
    type_name = type(value).__name__
    if type_name in _FORBIDDEN_TYPE_NAMES:
        raise ValueError(
            f"{field_name} must not contain runtime object type {type_name}"
        )
    module = getattr(type(value), "__module__", "")
    if module.startswith("pandas") or module.startswith("polars"):
        raise ValueError(
            f"{field_name} must not contain dataframe/series runtime objects"
        )


_EnumT = TypeVar("_EnumT", bound=Enum)


def _require_optional_enum(
    value: object,
    *,
    field_name: str,
    enum_type: type[_EnumT],
) -> _EnumT | None:
    if value is None:
        return None
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError as exc:
            raise ValueError(f"invalid {enum_type.__name__}: {value!r}") from exc
    raise ValueError(
        f"{field_name} must be {enum_type.__name__} or None, "
        f"got {type(value).__name__}"
    )


class WorkflowUiConfigurationPreset(BaseModel):
    """Immutable JSON-safe analysis configuration preset (schema version 1).

    Stores user analysis settings only. Optional unset UI selections are
    represented with ``None`` or empty collections. Incomplete draft rows are
    rejected at export time by ``build_exportable_configuration_preset`` and are
    never silently omitted from a downloaded preset.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = CONFIGURATION_PRESET_SCHEMA_VERSION
    analysis_mode: AnalysisExecutionMode = AnalysisExecutionMode.SUPERVISED
    requested_task: AnalysisTask | None = None
    target_column: str | None = None
    use_recommended_numeric_feature_set: bool = True
    explicit_feature_columns: list[str] = Field(default_factory=list)
    timestamp_column: str | None = None
    identifier_columns: list[str] = Field(default_factory=list)
    excluded_columns: list[str] = Field(default_factory=list)
    role_overrides: dict[str, ColumnRole] = Field(default_factory=dict)
    recommendation_objective: RecommendationObjective | None = None
    quality_direction: QualityOptimizationDirection | None = None
    quality_target: float | None = None
    performance_rules: list[UiMetricRuleInput] = Field(default_factory=list)
    cohort_filter: NumericCohortFilter | None = None
    anomaly_recommendation_enabled: bool = False
    recommendation_constraints: list[UiVariableConstraintInput] = Field(
        default_factory=list
    )
    confirmed_controllable_variables: list[str] = Field(default_factory=list)
    verified_variables: list[str] = Field(default_factory=list)
    maximum_simultaneous_changes: int = 3
    operating_point_selection_mode: OperatingPointSelectionMode = (
        OperatingPointSelectionMode.TOP_RESIDUAL_ANOMALY
    )
    explicit_operating_row_id: int | str | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def _validate_schema_version(cls, value: object) -> Literal[1]:
        if value is None:
            raise ValueError("schema_version is required")
        if value != 1:
            raise ValueError(f"unsupported schema_version: {value!r} (expected 1)")
        return 1

    @field_validator("analysis_mode", mode="before")
    @classmethod
    def _validate_analysis_mode(cls, value: object) -> AnalysisExecutionMode:
        _reject_forbidden_runtime_object(value, field_name="analysis_mode")
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

    @field_validator("requested_task", mode="before")
    @classmethod
    def _validate_requested_task(cls, value: object) -> AnalysisTask | None:
        _reject_forbidden_runtime_object(value, field_name="requested_task")
        task = _require_optional_enum(
            value,
            field_name="requested_task",
            enum_type=AnalysisTask,
        )
        if task is None:
            return None
        if task not in {AnalysisTask.REGRESSION, AnalysisTask.CLASSIFICATION}:
            raise ValueError(
                "requested_task must be REGRESSION, CLASSIFICATION, or None, "
                f"got {task!r}"
            )
        return task

    @field_validator("target_column", "timestamp_column", mode="before")
    @classmethod
    def _validate_optional_columns(cls, value: object) -> str | None:
        _reject_forbidden_runtime_object(value, field_name="optional column")
        if value is None:
            return None
        return _validate_column_name(value, field_name="optional column")

    @field_validator("use_recommended_numeric_feature_set", mode="before")
    @classmethod
    def _validate_use_recommended(cls, value: object) -> bool:
        return _require_strict_bool(
            value,
            field_name="use_recommended_numeric_feature_set",
        )

    @field_validator(
        "explicit_feature_columns",
        "identifier_columns",
        "excluded_columns",
        "confirmed_controllable_variables",
        "verified_variables",
        mode="before",
    )
    @classmethod
    def _validate_string_lists_before(cls, value: object) -> list[str]:
        _reject_forbidden_runtime_object(value, field_name="column list")
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"column list must be a list[str], got {type(value).__name__}"
            )
        return list(value)

    @field_validator(
        "explicit_feature_columns",
        "identifier_columns",
        "excluded_columns",
        "confirmed_controllable_variables",
        "verified_variables",
        mode="after",
    )
    @classmethod
    def _validate_string_lists(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(value, field_name="column list")

    @field_validator("role_overrides", mode="before")
    @classmethod
    def _validate_role_overrides_before(cls, value: object) -> dict[str, ColumnRole]:
        _reject_forbidden_runtime_object(value, field_name="role_overrides")
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(
                f"role_overrides must be a dict[str, ColumnRole], "
                f"got {type(value).__name__}"
            )
        cleaned: dict[str, ColumnRole] = {}
        for key, raw in value.items():
            name = _validate_column_name(key, field_name="role_overrides key")
            if name in cleaned:
                raise ValueError(
                    f"role_overrides must not contain duplicate keys: {name!r}"
                )
            role = _require_optional_enum(
                raw,
                field_name="role_overrides value",
                enum_type=ColumnRole,
            )
            if role is None:
                raise ValueError("role_overrides values must be ColumnRole")
            cleaned[name] = role
        return cleaned

    @field_validator("recommendation_objective", mode="before")
    @classmethod
    def _validate_objective(cls, value: object) -> RecommendationObjective | None:
        _reject_forbidden_runtime_object(value, field_name="recommendation_objective")
        return _require_optional_enum(
            value,
            field_name="recommendation_objective",
            enum_type=RecommendationObjective,
        )

    @field_validator("quality_direction", mode="before")
    @classmethod
    def _validate_quality_direction(
        cls,
        value: object,
    ) -> QualityOptimizationDirection | None:
        _reject_forbidden_runtime_object(value, field_name="quality_direction")
        return _require_optional_enum(
            value,
            field_name="quality_direction",
            enum_type=QualityOptimizationDirection,
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
        _reject_forbidden_runtime_object(value, field_name="performance_rules")
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"performance_rules must be a list, got {type(value).__name__}"
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
            if isinstance(item, dict):
                item = UiMetricRuleInput.model_validate(item)
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

    @field_validator("cohort_filter", mode="before")
    @classmethod
    def _validate_cohort_filter(cls, value: object) -> NumericCohortFilter | None:
        _reject_forbidden_runtime_object(value, field_name="cohort_filter")
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

    @field_validator("anomaly_recommendation_enabled", mode="before")
    @classmethod
    def _validate_anomaly_recommendation_enabled(cls, value: object) -> bool:
        return _require_strict_bool(
            value,
            field_name="anomaly_recommendation_enabled",
        )

    @field_validator("recommendation_constraints", mode="before")
    @classmethod
    def _validate_constraints_before(
        cls,
        value: object,
    ) -> list[UiVariableConstraintInput]:
        _reject_forbidden_runtime_object(
            value,
            field_name="recommendation_constraints",
        )
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                "recommendation_constraints must be a list, "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("recommendation_constraints", mode="after")
    @classmethod
    def _validate_constraints(
        cls,
        value: list[UiVariableConstraintInput],
    ) -> list[UiVariableConstraintInput]:
        seen: set[str] = set()
        copied: list[UiVariableConstraintInput] = []
        for item in value:
            if isinstance(item, dict):
                item = UiVariableConstraintInput.model_validate(item)
            if not isinstance(item, UiVariableConstraintInput):
                raise ValueError(
                    "recommendation_constraints entries must be "
                    f"UiVariableConstraintInput, got {type(item).__name__}"
                )
            if item.variable in seen:
                raise ValueError(
                    "recommendation_constraints must not contain duplicate "
                    f"variables: {item.variable!r}"
                )
            seen.add(item.variable)
            copied.append(item.model_copy(deep=True))
        return copied

    @field_validator("maximum_simultaneous_changes", mode="before")
    @classmethod
    def _validate_max_changes(cls, value: object) -> int:
        return _require_strict_int_ge(
            value,
            field_name="maximum_simultaneous_changes",
            minimum=1,
        )

    @field_validator("operating_point_selection_mode", mode="before")
    @classmethod
    def _validate_operating_mode(
        cls,
        value: object,
    ) -> OperatingPointSelectionMode:
        _reject_forbidden_runtime_object(
            value,
            field_name="operating_point_selection_mode",
        )
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
            "operating_point_selection_mode must be OperatingPointSelectionMode, "
            f"got {type(value).__name__}"
        )

    @field_validator("explicit_operating_row_id", mode="before")
    @classmethod
    def _validate_explicit_row_id(cls, value: object) -> int | str | None:
        _reject_forbidden_runtime_object(
            value,
            field_name="explicit_operating_row_id",
        )
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("explicit_operating_row_id must not be a bool")
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            number = _require_finite_float(
                value,
                field_name="explicit_operating_row_id",
            )
            if not number.is_integer():
                raise ValueError(
                    "explicit_operating_row_id float must be an integer value"
                )
            return int(number)
        if isinstance(value, str):
            return _require_optional_non_empty_str(
                value,
                field_name="explicit_operating_row_id",
            )
        raise ValueError(
            "explicit_operating_row_id must be int, str, or None, "
            f"got {type(value).__name__}"
        )

    @model_validator(mode="after")
    def _reject_non_json_safe_payload(self) -> Self:
        # Defensive: model_dump(mode="json") must succeed for export.
        self.model_dump(mode="json")
        return self


@dataclass(frozen=True, slots=True)
class ConfigurationPresetCompatibilityResult:
    """Result of comparing a preset against current CSV column names."""

    compatible: bool
    missing_columns: tuple[str, ...]
    message: str


@dataclass(frozen=True, slots=True)
class ConfigurationPresetParseResult:
    """Result of parsing and validating a configuration preset JSON payload."""

    ok: bool
    preset: WorkflowUiConfigurationPreset | None
    parsed_dict: dict[str, Any] | None
    error_message: str | None


@dataclass(frozen=True, slots=True)
class ConfigurationPresetApplyResult:
    """Result of an atomic preset apply attempt against session state."""

    applied: bool
    error_message: str | None
    summary: dict[str, Any] | None
    missing_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConfigurationPresetBuildResult:
    """Result of building a preset for export with completeness validation.

    Distinguishes fully empty optional rows (omit), fully valid rows (include),
    and partial/invalid rows (block export). Never mutates caller inputs.
    """

    preset: WorkflowUiConfigurationPreset | None
    is_exportable: bool
    issues: tuple[str, ...]


def _is_blank_optional_text(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _join_required_fields(fields: Sequence[str]) -> str:
    items = list(fields)
    if not items:
        return "required fields are missing"
    if len(items) == 1:
        return f"{items[0]} is required"
    if len(items) == 2:
        return f"{items[0]} and {items[1]} are required"
    return f"{', '.join(items[:-1])}, and {items[-1]} are required"


def _parse_optional_finite_number(value: object) -> float | None:
    """Return a finite float, or None when the value is blank."""
    if _is_blank_optional_text(value):
        return None
    if isinstance(value, bool):
        raise ValueError("bool is not a finite number")
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        number = float(str(value).strip())
    if not math.isfinite(number):
        raise ValueError("non-finite number")
    return number


def _performance_rule_draft_is_empty(draft: Mapping[str, Any]) -> bool:
    metric_blank = _is_blank_optional_text(draft.get("metric_name"))
    direction_raw = draft.get("direction")
    direction_blank = direction_raw is None or (
        isinstance(direction_raw, str) and direction_raw.strip() == ""
    )
    threshold_blank = _is_blank_optional_text(draft.get("threshold"))
    return metric_blank and direction_blank and threshold_blank


def _validate_performance_rule_drafts(
    drafts: Sequence[Mapping[str, Any]],
) -> tuple[list[UiMetricRuleInput], list[str]]:
    """Validate UI performance-rule drafts in row order.

    Fully empty optional rows are omitted. Partial or invalid rows produce
    deterministic issue messages and block export.
    """
    rules: list[UiMetricRuleInput] = []
    issues: list[str] = []
    seen_metrics: set[str] = set()

    for index, draft in enumerate(drafts):
        rule_number = index + 1
        if _performance_rule_draft_is_empty(draft):
            continue

        metric_raw = draft.get("metric_name")
        direction_raw = draft.get("direction")
        threshold_raw = draft.get("threshold")
        required_raw = draft.get("required", True)

        missing: list[str] = []
        metric_name: str | None = None
        if _is_blank_optional_text(metric_raw):
            missing.append("metric name")
        else:
            metric_name = str(metric_raw).strip()

        direction: MetricAcceptanceDirection | None = None
        if direction_raw is None or (
            isinstance(direction_raw, str) and direction_raw.strip() == ""
        ):
            missing.append("direction")
        elif isinstance(direction_raw, MetricAcceptanceDirection):
            direction = direction_raw
        elif isinstance(direction_raw, str):
            try:
                direction = MetricAcceptanceDirection(direction_raw.strip())
            except ValueError:
                missing.append("direction")
        else:
            missing.append("direction")

        threshold_blank = _is_blank_optional_text(threshold_raw)
        threshold_value: float | None = None
        threshold_nonfinite = False
        if threshold_blank:
            missing.append("threshold")
        else:
            try:
                threshold_value = _parse_optional_finite_number(threshold_raw)
            except (TypeError, ValueError):
                threshold_nonfinite = True

        if missing:
            issues.append(
                f"Performance rule {rule_number} is incomplete: "
                f"{_join_required_fields(missing)}."
            )
            continue
        if threshold_nonfinite or threshold_value is None:
            issues.append(
                f"Threshold for performance rule {rule_number} must be finite."
            )
            continue
        if metric_name is None or direction is None:
            issues.append(
                f"Performance rule {rule_number} is incomplete: "
                f"{_join_required_fields(['metric name', 'direction'])}."
            )
            continue
        if metric_name in seen_metrics:
            issues.append(
                f"Performance rule {rule_number} is invalid: "
                f"metric name '{metric_name}' is duplicated."
            )
            continue
        if type(required_raw) is not bool:
            issues.append(
                f"Performance rule {rule_number} is invalid: "
                "required must be a boolean."
            )
            continue

        seen_metrics.add(metric_name)
        rules.append(
            UiMetricRuleInput(
                metric_name=metric_name,
                direction=direction,
                threshold=threshold_value,
                required=required_raw,
            )
        )

    return rules, issues


def _validate_constraint_drafts(
    drafts: Sequence[Mapping[str, Any]],
) -> tuple[list[UiVariableConstraintInput], list[str]]:
    """Validate UI constraint drafts in selection order.

    A draft present in the sequence means the variable was selected. Selected
    rows with missing or invalid bounds block export; they are never omitted.
    """
    constraints: list[UiVariableConstraintInput] = []
    issues: list[str] = []
    seen_variables: set[str] = set()

    for draft in drafts:
        variable_raw = draft.get("variable")
        if _is_blank_optional_text(variable_raw):
            issues.append(
                "Constraint row is incomplete: variable name is required."
            )
            continue
        variable = str(variable_raw).strip()
        if variable in seen_variables:
            issues.append(
                f"Constraint for '{variable}' is invalid: "
                "variable is duplicated."
            )
            continue

        minimum_raw = draft.get("minimum")
        maximum_raw = draft.get("maximum")
        min_missing = _is_blank_optional_text(minimum_raw)
        max_missing = _is_blank_optional_text(maximum_raw)

        if min_missing or max_missing:
            missing_bounds: list[str] = []
            if min_missing:
                missing_bounds.append("lower")
            if max_missing:
                missing_bounds.append("upper")
            if len(missing_bounds) == 2:
                bound_text = "lower and upper bounds are required"
            else:
                bound_text = f"{missing_bounds[0]} bound is required"
            issues.append(
                f"Constraint for '{variable}' is incomplete: {bound_text}."
            )
            continue

        try:
            minimum_value = _parse_optional_finite_number(minimum_raw)
            maximum_value = _parse_optional_finite_number(maximum_raw)
        except (TypeError, ValueError):
            issues.append(
                f"Constraint for '{variable}' is invalid: "
                "bounds must be finite."
            )
            continue

        if minimum_value is None or maximum_value is None:
            issues.append(
                f"Constraint for '{variable}' is incomplete: "
                "lower and upper bounds are required."
            )
            continue

        if minimum_value >= maximum_value:
            issues.append(
                f"Constraint for '{variable}' is invalid: "
                "lower bound must not exceed upper bound."
            )
            continue

        seen_variables.add(variable)
        constraints.append(
            UiVariableConstraintInput(
                variable=variable,
                minimum=minimum_value,
                maximum=maximum_value,
            )
        )

    return constraints, issues


def build_exportable_configuration_preset(
    *,
    analysis_mode: AnalysisExecutionMode,
    requested_task: AnalysisTask | None = None,
    target_column: str | None = None,
    use_recommended_numeric_feature_set: bool = True,
    explicit_feature_columns: Sequence[str] | None = None,
    timestamp_column: str | None = None,
    identifier_columns: Sequence[str] | None = None,
    excluded_columns: Sequence[str] | None = None,
    role_overrides: Mapping[str, ColumnRole] | None = None,
    recommendation_objective: RecommendationObjective | None = None,
    quality_direction: QualityOptimizationDirection | None = None,
    quality_target: float | None = None,
    performance_rule_drafts: Sequence[Mapping[str, Any]] | None = None,
    cohort_filter: NumericCohortFilter | None = None,
    operating_cohort_restricted: bool = False,
    anomaly_recommendation_enabled: bool = False,
    recommendation_constraint_drafts: Sequence[Mapping[str, Any]] | None = None,
    confirmed_controllable_variables: Sequence[str] | None = None,
    verified_variables: Sequence[str] | None = None,
    maximum_simultaneous_changes: int = 3,
    operating_point_selection_mode: OperatingPointSelectionMode = (
        OperatingPointSelectionMode.TOP_RESIDUAL_ANOMALY
    ),
    explicit_operating_row_id: int | str | None = None,
) -> ConfigurationPresetBuildResult:
    """Inspect current UI drafts and build a preset only when exportable.

    Completely empty optional performance-rule rows may be omitted. Selected
    constraint rows and partially filled performance-rule rows never silently
    drop: they produce issues and ``is_exportable=False``. An explicitly
    enabled but incomplete operating cohort filter also blocks export rather
    than being omitted. Caller inputs are not mutated or auto-corrected.
    """
    rule_drafts = performance_rule_drafts or ()
    constraint_drafts = recommendation_constraint_drafts or ()

    performance_rules, rule_issues = _validate_performance_rule_drafts(rule_drafts)
    recommendation_constraints, constraint_issues = _validate_constraint_drafts(
        constraint_drafts
    )
    cohort_issues: list[str] = []
    if operating_cohort_restricted and cohort_filter is None:
        cohort_issues.append(
            "Operating cohort filter is incomplete: select a cohort column and "
            "finite lower/upper bounds, or disable the filter before export."
        )
    issues = tuple([*rule_issues, *constraint_issues, *cohort_issues])
    if issues:
        return ConfigurationPresetBuildResult(
            preset=None,
            is_exportable=False,
            issues=issues,
        )

    try:
        preset = build_configuration_preset(
            analysis_mode=analysis_mode,
            requested_task=requested_task,
            target_column=target_column,
            use_recommended_numeric_feature_set=use_recommended_numeric_feature_set,
            explicit_feature_columns=explicit_feature_columns,
            timestamp_column=timestamp_column,
            identifier_columns=identifier_columns,
            excluded_columns=excluded_columns,
            role_overrides=role_overrides,
            recommendation_objective=recommendation_objective,
            quality_direction=quality_direction,
            quality_target=quality_target,
            performance_rules=performance_rules,
            cohort_filter=cohort_filter,
            anomaly_recommendation_enabled=anomaly_recommendation_enabled,
            recommendation_constraints=recommendation_constraints,
            confirmed_controllable_variables=confirmed_controllable_variables,
            verified_variables=verified_variables,
            maximum_simultaneous_changes=maximum_simultaneous_changes,
            operating_point_selection_mode=operating_point_selection_mode,
            explicit_operating_row_id=explicit_operating_row_id,
        )
    except (TypeError, ValueError, ValidationError) as exc:
        return ConfigurationPresetBuildResult(
            preset=None,
            is_exportable=False,
            issues=(f"Configuration preset export is unavailable: {exc}",),
        )

    return ConfigurationPresetBuildResult(
        preset=preset,
        is_exportable=True,
        issues=(),
    )


def build_configuration_preset(
    *,
    analysis_mode: AnalysisExecutionMode,
    requested_task: AnalysisTask | None = None,
    target_column: str | None = None,
    use_recommended_numeric_feature_set: bool = True,
    explicit_feature_columns: Sequence[str] | None = None,
    timestamp_column: str | None = None,
    identifier_columns: Sequence[str] | None = None,
    excluded_columns: Sequence[str] | None = None,
    role_overrides: Mapping[str, ColumnRole] | None = None,
    recommendation_objective: RecommendationObjective | None = None,
    quality_direction: QualityOptimizationDirection | None = None,
    quality_target: float | None = None,
    performance_rules: Sequence[UiMetricRuleInput] | None = None,
    cohort_filter: NumericCohortFilter | None = None,
    anomaly_recommendation_enabled: bool = False,
    recommendation_constraints: Sequence[UiVariableConstraintInput] | None = None,
    confirmed_controllable_variables: Sequence[str] | None = None,
    verified_variables: Sequence[str] | None = None,
    maximum_simultaneous_changes: int = 3,
    operating_point_selection_mode: OperatingPointSelectionMode = (
        OperatingPointSelectionMode.TOP_RESIDUAL_ANOMALY
    ),
    explicit_operating_row_id: int | str | None = None,
) -> WorkflowUiConfigurationPreset:
    """Build a preset DTO from current UI configuration values."""
    return WorkflowUiConfigurationPreset(
        schema_version=1,
        analysis_mode=analysis_mode,
        requested_task=requested_task,
        target_column=target_column,
        use_recommended_numeric_feature_set=use_recommended_numeric_feature_set,
        explicit_feature_columns=list(explicit_feature_columns or ()),
        timestamp_column=timestamp_column,
        identifier_columns=list(identifier_columns or ()),
        excluded_columns=list(excluded_columns or ()),
        role_overrides=dict(role_overrides or {}),
        recommendation_objective=recommendation_objective,
        quality_direction=quality_direction,
        quality_target=quality_target,
        performance_rules=[
            item.model_copy(deep=True) for item in (performance_rules or ())
        ],
        cohort_filter=(
            None if cohort_filter is None else cohort_filter.model_copy(deep=True)
        ),
        anomaly_recommendation_enabled=anomaly_recommendation_enabled,
        recommendation_constraints=[
            item.model_copy(deep=True) for item in (recommendation_constraints or ())
        ],
        confirmed_controllable_variables=list(confirmed_controllable_variables or ()),
        verified_variables=list(verified_variables or ()),
        maximum_simultaneous_changes=maximum_simultaneous_changes,
        operating_point_selection_mode=operating_point_selection_mode,
        explicit_operating_row_id=explicit_operating_row_id,
    )


def configuration_preset_to_json(preset: WorkflowUiConfigurationPreset) -> str:
    """Serialize a preset to deterministic, JSON-safe text."""
    if not isinstance(preset, WorkflowUiConfigurationPreset):
        raise TypeError(
            "preset must be WorkflowUiConfigurationPreset, "
            f"got {type(preset).__name__}"
        )
    payload = preset.model_dump(mode="json")
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _reject_nonfinite_constants(value: str) -> object:
    raise ValueError(f"non-finite JSON constant rejected: {value}")


def _walk_reject_nonfinite(value: object, *, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite, got {value!r}")
    if isinstance(value, dict):
        for key, nested in value.items():
            _walk_reject_nonfinite(nested, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _walk_reject_nonfinite(nested, path=f"{path}[{index}]")


def parse_configuration_preset_json(text: str) -> ConfigurationPresetParseResult:
    """Parse and validate configuration preset JSON without applying it."""
    if not isinstance(text, str):
        return ConfigurationPresetParseResult(
            ok=False,
            preset=None,
            parsed_dict=None,
            error_message=(
                "Configuration JSON must be text, "
                f"got {type(text).__name__}"
            ),
        )
    try:
        loaded = json.loads(text, parse_constant=_reject_nonfinite_constants)
    except json.JSONDecodeError as exc:
        return ConfigurationPresetParseResult(
            ok=False,
            preset=None,
            parsed_dict=None,
            error_message=f"JSON parsing failed: {exc.msg}",
        )
    except ValueError as exc:
        return ConfigurationPresetParseResult(
            ok=False,
            preset=None,
            parsed_dict=None,
            error_message=str(exc),
        )

    if not isinstance(loaded, dict):
        return ConfigurationPresetParseResult(
            ok=False,
            preset=None,
            parsed_dict=None,
            error_message=(
                "Configuration JSON root must be an object, "
                f"got {type(loaded).__name__}"
            ),
        )

    if "schema_version" not in loaded:
        return ConfigurationPresetParseResult(
            ok=False,
            preset=None,
            parsed_dict=None,
            error_message="schema_version is required",
        )
    if loaded["schema_version"] != 1:
        return ConfigurationPresetParseResult(
            ok=False,
            preset=None,
            parsed_dict=None,
            error_message=(
                "unsupported schema_version: "
                f"{loaded['schema_version']!r} (expected 1)"
            ),
        )

    try:
        _walk_reject_nonfinite(loaded, path="preset")
        preset = WorkflowUiConfigurationPreset.model_validate(loaded)
    except (ValueError, TypeError, ValidationError) as exc:
        return ConfigurationPresetParseResult(
            ok=False,
            preset=None,
            parsed_dict=None,
            error_message=f"Configuration validation failed: {exc}",
        )

    return ConfigurationPresetParseResult(
        ok=True,
        preset=preset,
        parsed_dict=preset.model_dump(mode="json"),
        error_message=None,
    )


def collect_referenced_columns(
    preset: WorkflowUiConfigurationPreset,
) -> tuple[str, ...]:
    """Return sorted unique column names referenced by a preset."""
    names: set[str] = set()
    if preset.target_column is not None:
        names.add(preset.target_column)
    names.update(preset.explicit_feature_columns)
    if preset.timestamp_column is not None:
        names.add(preset.timestamp_column)
    names.update(preset.identifier_columns)
    names.update(preset.excluded_columns)
    names.update(preset.role_overrides.keys())
    if preset.cohort_filter is not None:
        names.add(preset.cohort_filter.column_name)
    for item in preset.recommendation_constraints:
        names.add(item.variable)
    names.update(preset.confirmed_controllable_variables)
    names.update(preset.verified_variables)
    return tuple(sorted(names))


def check_configuration_preset_column_compatibility(
    preset: WorkflowUiConfigurationPreset,
    *,
    available_columns: Sequence[str],
) -> ConfigurationPresetCompatibilityResult:
    """Verify that all preset-referenced columns exist in the current CSV."""
    if not isinstance(preset, WorkflowUiConfigurationPreset):
        raise TypeError(
            "preset must be WorkflowUiConfigurationPreset, "
            f"got {type(preset).__name__}"
        )
    if not isinstance(available_columns, Sequence) or isinstance(
        available_columns,
        (str, bytes),
    ):
        raise TypeError(
            "available_columns must be a sequence of column name strings, "
            f"got {type(available_columns).__name__}"
        )
    available = set(available_columns)
    missing = [name for name in collect_referenced_columns(preset) if name not in available]
    missing_sorted = tuple(sorted(missing))
    if missing_sorted:
        return ConfigurationPresetCompatibilityResult(
            compatible=False,
            missing_columns=missing_sorted,
            message=(
                "Configuration references columns absent from the current CSV: "
                + ", ".join(missing_sorted)
            ),
        )
    return ConfigurationPresetCompatibilityResult(
        compatible=True,
        missing_columns=(),
        message="Configuration is compatible with the current CSV columns.",
    )


def build_configuration_preset_summary(
    preset: WorkflowUiConfigurationPreset,
) -> dict[str, Any]:
    """Build a compact JSON-safe summary for successful apply display."""
    feature_mode = (
        "recommended numeric feature set"
        if preset.use_recommended_numeric_feature_set
        else "explicit feature columns"
    )
    target_display = (
        "Not required"
        if preset.analysis_mode is AnalysisExecutionMode.ANOMALY_ONLY
        else (preset.target_column or "Not selected")
    )
    return {
        "schema_version": preset.schema_version,
        "analysis_mode": preset.analysis_mode.value,
        "target": target_display,
        "feature_configuration": feature_mode,
        "cohort_filter_configured": preset.cohort_filter is not None,
        "anomaly_recommendation_enabled": preset.anomaly_recommendation_enabled,
        "constraint_variable_count": len(preset.recommendation_constraints),
    }


def _clear_key_prefix(session_state: MutableMapping[str, Any], prefix: str) -> None:
    for key in list(session_state.keys()):
        if isinstance(key, str) and key.startswith(prefix):
            session_state.pop(key, None)


def _clear_supervised_widget_state(session_state: MutableMapping[str, Any]) -> None:
    session_state[SESSION_TARGET_KEY] = _TARGET_PLACEHOLDER
    session_state[SESSION_REQUESTED_TASK_KEY] = _TASK_AUTO
    session_state[SESSION_OBJECTIVE_KEY] = _OBJECTIVE_PLACEHOLDER
    session_state[SESSION_QUALITY_DIRECTION_KEY] = _QUALITY_DIRECTION_PLACEHOLDER
    session_state.pop(SESSION_QUALITY_TARGET_KEY, None)
    session_state[SESSION_RULE_COUNT_KEY] = 1
    _clear_key_prefix(session_state, "metric_name_")
    _clear_key_prefix(session_state, "metric_direction_")
    _clear_key_prefix(session_state, "metric_threshold_")
    _clear_key_prefix(session_state, "metric_required_")
    session_state[SESSION_CONSTRAINT_VARIABLES_KEY] = []
    _clear_key_prefix(session_state, "constraint_min_")
    _clear_key_prefix(session_state, "constraint_max_")
    session_state[SESSION_CONFIRMED_CONTROLLABLE_KEY] = []
    session_state[SESSION_VERIFIED_VARIABLES_KEY] = []


def _clear_anomaly_only_widget_state(session_state: MutableMapping[str, Any]) -> None:
    session_state[SESSION_RESTRICT_COHORT_KEY] = False
    session_state[SESSION_COHORT_COLUMN_KEY] = _COHORT_COLUMN_PLACEHOLDER
    session_state[SESSION_COHORT_LOWER_KEY] = ""
    session_state[SESSION_COHORT_UPPER_KEY] = ""
    session_state[SESSION_COHORT_INCLUDE_LOWER_KEY] = True
    session_state[SESSION_COHORT_INCLUDE_UPPER_KEY] = True
    session_state[SESSION_COHORT_EXCLUDE_FEATURE_KEY] = True
    session_state[SESSION_ANOMALY_RECOMMENDATION_ENABLED_KEY] = False
    session_state[SESSION_ANOMALY_REVIEW_VARIABLES_KEY] = []
    session_state[SESSION_ANOMALY_CONFIRMED_KEY] = []
    session_state[SESSION_ANOMALY_VERIFIED_KEY] = []
    session_state[SESSION_ANOMALY_CONSTRAINT_VARIABLES_KEY] = []
    _clear_key_prefix(session_state, "anomaly_rec_role_")
    _clear_key_prefix(session_state, "anomaly_rec_constraint_min_")
    _clear_key_prefix(session_state, "anomaly_rec_constraint_max_")


def _apply_shared_column_state(
    session_state: MutableMapping[str, Any],
    *,
    preset: WorkflowUiConfigurationPreset,
) -> None:
    session_state[SESSION_ANALYSIS_MODE_KEY] = preset.analysis_mode.value
    session_state[SESSION_TIMESTAMP_KEY] = (
        _TIMESTAMP_NONE
        if preset.timestamp_column is None
        else preset.timestamp_column
    )
    session_state[SESSION_IDENTIFIERS_KEY] = list(preset.identifier_columns)
    session_state[SESSION_EXCLUDED_KEY] = list(preset.excluded_columns)
    session_state[SESSION_USE_RECOMMENDED_FEATURES_KEY] = (
        preset.use_recommended_numeric_feature_set
    )
    session_state[SESSION_EXPLICIT_FEATURES_KEY] = list(preset.explicit_feature_columns)
    override_columns = list(preset.role_overrides.keys())
    session_state[SESSION_ROLE_OVERRIDE_COLUMNS_KEY] = override_columns
    _clear_key_prefix(session_state, "role_override_")
    for column_name, role in preset.role_overrides.items():
        session_state[f"role_override_{column_name}"] = role.value
    session_state[SESSION_MAX_CHANGES_KEY] = int(preset.maximum_simultaneous_changes)
    session_state[SESSION_OPERATING_MODE_KEY] = (
        preset.operating_point_selection_mode.value
    )
    session_state[SESSION_EXPLICIT_ROW_ID_KEY] = (
        ""
        if preset.explicit_operating_row_id is None
        else str(preset.explicit_operating_row_id)
    )


def _apply_supervised_state(
    session_state: MutableMapping[str, Any],
    *,
    preset: WorkflowUiConfigurationPreset,
) -> None:
    session_state[SESSION_TARGET_KEY] = (
        _TARGET_PLACEHOLDER if preset.target_column is None else preset.target_column
    )
    session_state[SESSION_REQUESTED_TASK_KEY] = (
        _TASK_AUTO if preset.requested_task is None else preset.requested_task.value
    )
    session_state[SESSION_OBJECTIVE_KEY] = (
        _OBJECTIVE_PLACEHOLDER
        if preset.recommendation_objective is None
        else preset.recommendation_objective.value
    )
    session_state[SESSION_QUALITY_DIRECTION_KEY] = (
        _QUALITY_DIRECTION_PLACEHOLDER
        if preset.quality_direction is None
        else preset.quality_direction.value
    )
    if preset.quality_target is None:
        session_state.pop(SESSION_QUALITY_TARGET_KEY, None)
    else:
        session_state[SESSION_QUALITY_TARGET_KEY] = float(preset.quality_target)

    rule_count = max(1, len(preset.performance_rules))
    session_state[SESSION_RULE_COUNT_KEY] = rule_count
    _clear_key_prefix(session_state, "metric_name_")
    _clear_key_prefix(session_state, "metric_direction_")
    _clear_key_prefix(session_state, "metric_threshold_")
    _clear_key_prefix(session_state, "metric_required_")
    for index, rule in enumerate(preset.performance_rules):
        session_state[f"metric_name_{index}"] = rule.metric_name
        session_state[f"metric_direction_{index}"] = rule.direction.value
        session_state[f"metric_threshold_{index}"] = str(rule.threshold)
        session_state[f"metric_required_{index}"] = bool(rule.required)

    constraint_names = [item.variable for item in preset.recommendation_constraints]
    session_state[SESSION_CONSTRAINT_VARIABLES_KEY] = list(constraint_names)
    _clear_key_prefix(session_state, "constraint_min_")
    _clear_key_prefix(session_state, "constraint_max_")
    for item in preset.recommendation_constraints:
        session_state[f"constraint_min_{item.variable}"] = float(item.minimum)
        session_state[f"constraint_max_{item.variable}"] = float(item.maximum)

    session_state[SESSION_CONFIRMED_CONTROLLABLE_KEY] = list(
        preset.confirmed_controllable_variables
    )
    session_state[SESSION_VERIFIED_VARIABLES_KEY] = list(preset.verified_variables)


def _apply_anomaly_only_state(
    session_state: MutableMapping[str, Any],
    *,
    preset: WorkflowUiConfigurationPreset,
) -> None:
    if preset.cohort_filter is None:
        session_state[SESSION_RESTRICT_COHORT_KEY] = False
        session_state[SESSION_COHORT_COLUMN_KEY] = _COHORT_COLUMN_PLACEHOLDER
        session_state[SESSION_COHORT_LOWER_KEY] = ""
        session_state[SESSION_COHORT_UPPER_KEY] = ""
        session_state[SESSION_COHORT_INCLUDE_LOWER_KEY] = True
        session_state[SESSION_COHORT_INCLUDE_UPPER_KEY] = True
        session_state[SESSION_COHORT_EXCLUDE_FEATURE_KEY] = True
    else:
        cohort = preset.cohort_filter
        session_state[SESSION_RESTRICT_COHORT_KEY] = True
        session_state[SESSION_COHORT_COLUMN_KEY] = cohort.column_name
        session_state[SESSION_COHORT_LOWER_KEY] = str(cohort.lower_bound)
        session_state[SESSION_COHORT_UPPER_KEY] = str(cohort.upper_bound)
        session_state[SESSION_COHORT_INCLUDE_LOWER_KEY] = bool(cohort.include_lower)
        session_state[SESSION_COHORT_INCLUDE_UPPER_KEY] = bool(cohort.include_upper)
        session_state[SESSION_COHORT_EXCLUDE_FEATURE_KEY] = bool(
            cohort.exclude_filter_column_from_features
        )

    enabled = bool(preset.anomaly_recommendation_enabled)
    session_state[SESSION_ANOMALY_RECOMMENDATION_ENABLED_KEY] = enabled
    if not enabled:
        session_state[SESSION_ANOMALY_REVIEW_VARIABLES_KEY] = []
        session_state[SESSION_ANOMALY_CONFIRMED_KEY] = []
        session_state[SESSION_ANOMALY_VERIFIED_KEY] = []
        session_state[SESSION_ANOMALY_CONSTRAINT_VARIABLES_KEY] = []
        _clear_key_prefix(session_state, "anomaly_rec_role_")
        _clear_key_prefix(session_state, "anomaly_rec_constraint_min_")
        _clear_key_prefix(session_state, "anomaly_rec_constraint_max_")
        return

    review_variables = sorted(
        {
            *preset.confirmed_controllable_variables,
            *preset.verified_variables,
            *[item.variable for item in preset.recommendation_constraints],
            *preset.role_overrides.keys(),
        }
    )
    session_state[SESSION_ANOMALY_REVIEW_VARIABLES_KEY] = list(review_variables)
    session_state[SESSION_ANOMALY_CONFIRMED_KEY] = list(
        preset.confirmed_controllable_variables
    )
    session_state[SESSION_ANOMALY_VERIFIED_KEY] = list(preset.verified_variables)
    constraint_names = [item.variable for item in preset.recommendation_constraints]
    session_state[SESSION_ANOMALY_CONSTRAINT_VARIABLES_KEY] = list(constraint_names)
    _clear_key_prefix(session_state, "anomaly_rec_role_")
    _clear_key_prefix(session_state, "anomaly_rec_constraint_min_")
    _clear_key_prefix(session_state, "anomaly_rec_constraint_max_")
    for column_name, role in preset.role_overrides.items():
        session_state[f"anomaly_rec_role_{column_name}"] = role.value
    for item in preset.recommendation_constraints:
        session_state[f"anomaly_rec_constraint_min_{item.variable}"] = str(
            item.minimum
        )
        session_state[f"anomaly_rec_constraint_max_{item.variable}"] = str(
            item.maximum
        )


def build_configuration_preset_session_updates(
    preset: WorkflowUiConfigurationPreset,
) -> dict[str, Any]:
    """Build JSON-safe widget session-state updates for a validated preset."""
    if not isinstance(preset, WorkflowUiConfigurationPreset):
        raise TypeError(
            "preset must be WorkflowUiConfigurationPreset, "
            f"got {type(preset).__name__}"
        )
    staged: dict[str, Any] = {}
    _apply_shared_column_state(staged, preset=preset)
    if preset.analysis_mode is AnalysisExecutionMode.ANOMALY_ONLY:
        _clear_supervised_widget_state(staged)
        _apply_anomaly_only_state(staged, preset=preset)
    else:
        _clear_anomaly_only_widget_state(staged)
        _apply_supervised_state(staged, preset=preset)
    summary = build_configuration_preset_summary(preset)
    staged[SESSION_PRESET_APPLY_SUMMARY_KEY] = summary
    return staged


def clear_configuration_preset_widget_prefixes(
    session_state: MutableMapping[str, Any],
) -> None:
    """Remove dynamic per-column/per-rule widget keys before an atomic apply."""
    for key in list(session_state.keys()):
        if isinstance(key, str) and key.startswith(WIDGET_STATE_KEY_PREFIXES):
            session_state.pop(key, None)


def apply_configuration_preset_to_session_state(
    session_state: MutableMapping[str, Any],
    *,
    preset: WorkflowUiConfigurationPreset,
    available_columns: Sequence[str],
) -> ConfigurationPresetApplyResult:
    """Atomically apply a validated preset to widget session-state keys.

    Validation and update construction happen before any mutation. On
    incompatibility the provided ``session_state`` mapping is left unchanged.

    For Streamlit widget keys, callers must invoke this only before those
    widgets are instantiated in the current script run (or apply a previously
    staged update dict via ``build_configuration_preset_session_updates``).
    """
    if not isinstance(preset, WorkflowUiConfigurationPreset):
        raise TypeError(
            "preset must be WorkflowUiConfigurationPreset, "
            f"got {type(preset).__name__}"
        )

    compatibility = check_configuration_preset_column_compatibility(
        preset,
        available_columns=available_columns,
    )
    if not compatibility.compatible:
        return ConfigurationPresetApplyResult(
            applied=False,
            error_message=compatibility.message,
            summary=None,
            missing_columns=compatibility.missing_columns,
        )

    staged = build_configuration_preset_session_updates(preset)
    clear_configuration_preset_widget_prefixes(session_state)
    for key, value in staged.items():
        session_state[key] = value

    return ConfigurationPresetApplyResult(
        applied=True,
        error_message=None,
        summary=staged.get(SESSION_PRESET_APPLY_SUMMARY_KEY),
        missing_columns=(),
    )
