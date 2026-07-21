"""Deterministic candidate value grids and scenarios (Step 9C).

Builds bounded continuous grids and unranked scenarios from
``CandidateVariableSet``. Does not call models, score scenarios, or emit
recommendations.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Self

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from process_intelligence.core.exceptions import DataValidationError
from process_intelligence.recommendation.candidate_selection import (
    CandidateVariable,
    CandidateVariableSet,
)
from process_intelligence.recommendation.constraint_resolution import (
    ConstraintResolutionStatus,
)
from process_intelligence.recommendation.enums import (
    RecommendationObjective,
    RecommendationSafetyStatus,
)
from process_intelligence.recommendation.schemas import ScalarMetadataValue

_ORIGINAL_ROW_ID = "_original_row_id"
_SCENARIO_ID_RE = re.compile(r"^SCN-(\d{6})$")

_WARNING_SAFETY_CAUTION = (
    "Safety status is CAUTION; treat generated scenarios conservatively."
)
_WARNING_RESOLUTION_PARTIAL = (
    "Constraint resolution is PARTIAL; only resolved candidates were used."
)
_WARNING_DEDUP = (
    "One or more variable grids merged nearby points during deduplication."
)
_WARNING_NONZERO_DELTA = (
    "One or more change points were removed due to minimum_nonzero_delta."
)
_WARNING_NO_CHANGE_POINTS = "One or more variables have no usable change points."
_WARNING_COMBINATION_CAP = (
    "Effective combination limit is smaller than the candidate change budget."
)
_WARNING_TRUNCATED = (
    "Scenario generation was truncated to the maximum_scenarios limit."
)
_WARNING_BASELINE_ONLY = "Only the baseline scenario was generated."
_WARNING_NO_SCENARIOS = "No candidate scenarios were generated."
_WARNING_MODEL_SCORING = "Model scoring was not performed."
_WARNING_VERIFICATION = (
    "Generated candidate scenarios require model, extrapolation, uncertainty, "
    "domain, and operational verification."
)


def _require_non_empty_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be str, got {type(value).__name__}")
    if value == "" or value.strip() == "":
        raise ValueError(f"{field_name} must be a non-empty, non-whitespace string")
    return value


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


def _require_non_negative_finite_float(
    value: object,
    *,
    field_name: str,
) -> float:
    number = _require_finite_float(value, field_name=field_name)
    if number < 0.0:
        raise ValueError(f"{field_name} must be >= 0, got {number}")
    return number


def _require_timezone_aware(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _validate_variable_name(value: object, *, field_name: str) -> str:
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


def _append_unique(target: list[str], message: str) -> None:
    if message not in target:
        target.append(message)


def _scenario_id_for_index(index: int) -> str:
    return f"SCN-{index:06d}"


class CandidateGridStatus(StrEnum):
    """Outcome status for deterministic candidate grid and scenario generation.

    ``READY`` means all allowed scenarios fit within the scenario limit.
    ``TRUNCATED`` means generation stopped at ``maximum_scenarios``.
    ``EMPTY`` means no non-baseline change scenarios were produced.
    ``REFUSED`` means safety or constraint resolution refused generation.
    """

    READY = "READY"
    TRUNCATED = "TRUNCATED"
    EMPTY = "EMPTY"
    REFUSED = "REFUSED"


class CandidateScenarioType(StrEnum):
    """Scenario shape produced by the candidate grid generator.

    ``BASELINE`` keeps every candidate at its current value.
    ``SINGLE_VARIABLE`` changes exactly one candidate variable.
    ``MULTI_VARIABLE`` changes two or more candidate variables.
    """

    BASELINE = "BASELINE"
    SINGLE_VARIABLE = "SINGLE_VARIABLE"
    MULTI_VARIABLE = "MULTI_VARIABLE"


class CandidateGridPolicy(BaseModel):
    """Tunable policy for bounded grid and scenario materialization.

    Controls linspace density, scenario caps, combination width, and
    deterministic ordering. Does not score or rank scenarios.
    """

    linear_points_per_variable: int = 5
    maximum_scenarios: int = 500
    maximum_combination_variables: int = 3
    include_single_variable_scenarios: bool = True
    include_multi_variable_scenarios: bool = True
    prefer_smaller_changes: bool = True
    deduplication_tolerance: float = 1e-12
    minimum_nonzero_delta: float = 1e-12

    @field_validator("linear_points_per_variable", mode="before")
    @classmethod
    def _validate_linear_points(cls, value: object) -> int:
        return _require_strict_int_ge(
            value,
            field_name="linear_points_per_variable",
            minimum=2,
        )

    @field_validator("maximum_scenarios", mode="before")
    @classmethod
    def _validate_maximum_scenarios(cls, value: object) -> int:
        return _require_strict_int_ge(
            value,
            field_name="maximum_scenarios",
            minimum=1,
        )

    @field_validator("maximum_combination_variables", mode="before")
    @classmethod
    def _validate_maximum_combination_variables(cls, value: object) -> int:
        return _require_strict_int_ge(
            value,
            field_name="maximum_combination_variables",
            minimum=1,
        )

    @field_validator(
        "include_single_variable_scenarios",
        "include_multi_variable_scenarios",
        "prefer_smaller_changes",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="policy bool field")

    @field_validator(
        "deduplication_tolerance",
        "minimum_nonzero_delta",
        mode="before",
    )
    @classmethod
    def _validate_tolerance_fields(cls, value: object) -> float:
        return _require_non_negative_finite_float(
            value,
            field_name="tolerance field",
        )


class CandidateValuePoint(BaseModel):
    """Single bounded candidate value relative to a variable current value.

    Records delta, optional relative delta, and normalized position on the
    effective span. Does not imply predicted process improvement.
    """

    value: float
    delta: float
    relative_delta: float | None
    normalized_position: float
    is_current: bool
    is_lower_bound: bool
    is_upper_bound: bool

    @field_validator("value", "delta", "normalized_position", mode="before")
    @classmethod
    def _validate_numeric_fields(cls, value: object) -> float:
        return _require_finite_float(value, field_name="numeric point field")

    @field_validator("relative_delta", mode="before")
    @classmethod
    def _validate_relative_delta(cls, value: object) -> float | None:
        return _require_optional_finite_float(value, field_name="relative_delta")

    @field_validator(
        "is_current",
        "is_lower_bound",
        "is_upper_bound",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="point bool field")

    @model_validator(mode="after")
    def _validate_point_consistency(self) -> Self:
        if self.normalized_position < 0.0 or self.normalized_position > 1.0:
            raise ValueError(
                "normalized_position must be in [0.0, 1.0], "
                f"got {self.normalized_position}"
            )
        if self.is_current and not math.isclose(
            self.delta,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("is_current=True requires delta to be math.isclose to 0")
        return self


class VariableCandidateGrid(BaseModel):
    """Ascending bounded value grid for one continuous candidate variable.

    Preserves minimum, maximum, and current anchors. Does not propose an
    optimal setpoint.
    """

    variable: str
    diagnosis_rank: int
    current_value: float
    minimum: float
    maximum: float
    effective_span: float
    points: list[CandidateValuePoint]
    point_count: int
    change_point_count: int
    warnings: list[str] = Field(default_factory=list)

    @field_validator("variable", mode="before")
    @classmethod
    def _validate_variable(cls, value: object) -> str:
        return _validate_variable_name(value, field_name="variable")

    @field_validator("diagnosis_rank", mode="before")
    @classmethod
    def _validate_diagnosis_rank(cls, value: object) -> int:
        return _require_strict_int_ge(value, field_name="diagnosis_rank", minimum=1)

    @field_validator(
        "current_value",
        "minimum",
        "maximum",
        "effective_span",
        mode="before",
    )
    @classmethod
    def _validate_numeric_fields(cls, value: object) -> float:
        return _require_finite_float(value, field_name="numeric grid field")

    @field_validator("points", mode="before")
    @classmethod
    def _validate_points_before(cls, value: object) -> list[CandidateValuePoint]:
        if not isinstance(value, list):
            raise ValueError(
                f"points must be a list[CandidateValuePoint], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("points", mode="after")
    @classmethod
    def _validate_points_after(
        cls,
        value: list[CandidateValuePoint],
    ) -> list[CandidateValuePoint]:
        if not value:
            raise ValueError("points must contain at least one point")
        copied: list[CandidateValuePoint] = []
        seen_values: set[float] = set()
        previous: float | None = None
        for item in value:
            if not isinstance(item, CandidateValuePoint):
                raise ValueError(
                    "points entries must be CandidateValuePoint, "
                    f"got {type(item).__name__}"
                )
            if item.value in seen_values:
                raise ValueError(
                    f"points must not contain duplicate values: {item.value!r}"
                )
            seen_values.add(item.value)
            if previous is not None and item.value < previous:
                raise ValueError("points must be sorted by ascending value")
            previous = item.value
            copied.append(item.model_copy(deep=True))
        return copied

    @field_validator("point_count", "change_point_count", mode="before")
    @classmethod
    def _validate_counts(cls, value: object) -> int:
        return _require_strict_int_ge(value, field_name="grid count field", minimum=0)

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
    def _validate_grid_consistency(self) -> Self:
        if self.minimum > self.current_value or self.current_value > self.maximum:
            raise ValueError(
                "current_value must satisfy minimum <= current_value <= maximum"
            )
        if self.effective_span <= 0.0:
            raise ValueError("effective_span must be > 0")
        expected_span = self.maximum - self.minimum
        if not math.isclose(
            self.effective_span,
            expected_span,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "effective_span must equal maximum - minimum "
                f"(got {self.effective_span}, expected {expected_span})"
            )
        if self.point_count != len(self.points):
            raise ValueError("point_count must equal len(points)")
        expected_change = sum(1 for point in self.points if not point.is_current)
        if self.change_point_count != expected_change:
            raise ValueError(
                "change_point_count must equal the number of non-current points"
            )

        current_points = [point for point in self.points if point.is_current]
        if len(current_points) != 1:
            raise ValueError("points must contain exactly one current point")
        current_point = current_points[0]
        if not math.isclose(
            current_point.value,
            self.current_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("current point value must match current_value")

        if not any(point.is_lower_bound for point in self.points):
            raise ValueError("points must include a lower-bound point")
        if not any(point.is_upper_bound for point in self.points):
            raise ValueError("points must include an upper-bound point")

        for point in self.points:
            if point.value < self.minimum or point.value > self.maximum:
                if not (
                    math.isclose(
                        point.value,
                        self.minimum,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or math.isclose(
                        point.value,
                        self.maximum,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ):
                    raise ValueError(
                        f"point value {point.value} is outside "
                        f"[{self.minimum}, {self.maximum}]"
                    )
            expected_norm = (point.value - self.minimum) / self.effective_span
            if not math.isclose(
                point.normalized_position,
                expected_norm,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "point normalized_position must equal "
                    "(value - minimum) / effective_span"
                )
            expected_lower = math.isclose(
                point.value,
                self.minimum,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            expected_upper = math.isclose(
                point.value,
                self.maximum,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            if point.is_lower_bound != expected_lower:
                raise ValueError(
                    "is_lower_bound must match math.isclose(value, minimum)"
                )
            if point.is_upper_bound != expected_upper:
                raise ValueError(
                    "is_upper_bound must match math.isclose(value, maximum)"
                )
        return self


class CandidateScenario(BaseModel):
    """Immutable unranked candidate scenario for later model scoring.

    Stores full variable assignments and change metadata. Does not encode
    predicted performance or recommendation status.
    """

    scenario_id: str
    scenario_index: int
    scenario_type: CandidateScenarioType
    variable_values: dict[str, float]
    changed_variables: list[str]
    deltas: dict[str, float]
    relative_deltas: dict[str, float | None]
    change_count: int
    normalized_change_magnitude: float

    @field_validator("scenario_id", mode="before")
    @classmethod
    def _validate_scenario_id(cls, value: object) -> str:
        text = _require_non_empty_str(value, field_name="scenario_id")
        if _SCENARIO_ID_RE.fullmatch(text) is None:
            raise ValueError(
                f"scenario_id must match SCN-000000 format, got {text!r}"
            )
        return text

    @field_validator("scenario_index", mode="before")
    @classmethod
    def _validate_scenario_index(cls, value: object) -> int:
        return _require_strict_int_ge(value, field_name="scenario_index", minimum=0)

    @field_validator("scenario_type", mode="before")
    @classmethod
    def _validate_scenario_type(cls, value: object) -> CandidateScenarioType:
        if isinstance(value, CandidateScenarioType):
            return value
        if isinstance(value, str):
            try:
                return CandidateScenarioType(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid CandidateScenarioType: {value!r}"
                ) from exc
        raise ValueError(
            f"scenario_type must be CandidateScenarioType, got {type(value).__name__}"
        )

    @field_validator("variable_values", mode="before")
    @classmethod
    def _validate_variable_values_before(cls, value: object) -> dict[str, float]:
        if not isinstance(value, dict):
            raise ValueError(
                f"variable_values must be a dict[str, float], got {type(value).__name__}"
            )
        cleaned: dict[str, float] = {}
        for key, raw in value.items():
            name = _require_non_empty_str(key, field_name="variable_values key")
            cleaned[name] = _require_finite_float(
                raw,
                field_name=f"variable_values[{name!r}]",
            )
        return cleaned

    @field_validator("changed_variables", mode="before")
    @classmethod
    def _validate_changed_variables_before(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"changed_variables must be a list[str], got {type(value).__name__}"
            )
        return list(value)

    @field_validator("changed_variables", mode="after")
    @classmethod
    def _validate_changed_variables(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(
            value,
            field_name="changed_variables",
        )

    @field_validator("deltas", mode="before")
    @classmethod
    def _validate_deltas_before(cls, value: object) -> dict[str, float]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(
                f"deltas must be a dict[str, float], got {type(value).__name__}"
            )
        cleaned: dict[str, float] = {}
        for key, raw in value.items():
            name = _require_non_empty_str(key, field_name="deltas key")
            cleaned[name] = _require_finite_float(raw, field_name=f"deltas[{name!r}]")
        return cleaned

    @field_validator("relative_deltas", mode="before")
    @classmethod
    def _validate_relative_deltas_before(
        cls,
        value: object,
    ) -> dict[str, float | None]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(
                f"relative_deltas must be a dict[str, float | None], "
                f"got {type(value).__name__}"
            )
        cleaned: dict[str, float | None] = {}
        for key, raw in value.items():
            name = _require_non_empty_str(key, field_name="relative_deltas key")
            cleaned[name] = _require_optional_finite_float(
                raw,
                field_name=f"relative_deltas[{name!r}]",
            )
        return cleaned

    @field_validator("change_count", mode="before")
    @classmethod
    def _validate_change_count(cls, value: object) -> int:
        return _require_strict_int_ge(value, field_name="change_count", minimum=0)

    @field_validator("normalized_change_magnitude", mode="before")
    @classmethod
    def _validate_normalized_change_magnitude(cls, value: object) -> float:
        return _require_non_negative_finite_float(
            value,
            field_name="normalized_change_magnitude",
        )

    @model_validator(mode="after")
    def _validate_scenario_consistency(self) -> Self:
        match = _SCENARIO_ID_RE.fullmatch(self.scenario_id)
        assert match is not None
        if int(match.group(1)) != self.scenario_index:
            raise ValueError("scenario_id numeric suffix must equal scenario_index")

        for name in self.changed_variables:
            if name not in self.variable_values:
                raise ValueError(
                    f"changed_variables entry {name!r} missing from variable_values"
                )

        changed_set = set(self.changed_variables)
        if set(self.deltas.keys()) != changed_set:
            raise ValueError("deltas keys must exactly match changed_variables")
        if set(self.relative_deltas.keys()) != changed_set:
            raise ValueError(
                "relative_deltas keys must exactly match changed_variables"
            )

        for name, delta in self.deltas.items():
            if math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"changed variable delta for {name!r} must be non-zero"
                )

        if self.change_count != len(self.changed_variables):
            raise ValueError("change_count must equal len(changed_variables)")

        if self.scenario_type is CandidateScenarioType.BASELINE:
            if self.scenario_index != 0:
                raise ValueError("BASELINE requires scenario_index=0")
            if self.changed_variables:
                raise ValueError("BASELINE requires empty changed_variables")
            if self.deltas or self.relative_deltas:
                raise ValueError("BASELINE requires empty deltas and relative_deltas")
            if self.change_count != 0:
                raise ValueError("BASELINE requires change_count=0")
            if not math.isclose(
                self.normalized_change_magnitude,
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("BASELINE requires normalized_change_magnitude=0")
        elif self.scenario_type is CandidateScenarioType.SINGLE_VARIABLE:
            if self.change_count != 1:
                raise ValueError("SINGLE_VARIABLE requires change_count=1")
        elif self.scenario_type is CandidateScenarioType.MULTI_VARIABLE:
            if self.change_count < 2:
                raise ValueError("MULTI_VARIABLE requires change_count>=2")
        else:
            raise ValueError(f"unsupported scenario_type: {self.scenario_type!r}")
        return self


class CandidateGridReport(BaseModel):
    """Structured report of generated variable grids and candidate scenarios.

    Scenarios are deterministic and unranked. Model scoring and optimization
    are intentionally out of scope for this report.
    """

    status: CandidateGridStatus
    objective: RecommendationObjective
    safety_status: RecommendationSafetyStatus
    resolution_status: ConstraintResolutionStatus
    candidate_variables: list[str]
    variable_grids: list[VariableCandidateGrid]
    scenarios: list[CandidateScenario]
    baseline_scenario_id: str | None
    potential_scenario_count: int
    generated_scenario_count: int
    change_scenario_count: int
    truncated_scenario_count: int
    maximum_scenarios: int
    effective_combination_limit: int
    generated_at: datetime
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)

    @field_validator("status", mode="before")
    @classmethod
    def _validate_status(cls, value: object) -> CandidateGridStatus:
        if isinstance(value, CandidateGridStatus):
            return value
        if isinstance(value, str):
            try:
                return CandidateGridStatus(value)
            except ValueError as exc:
                raise ValueError(f"invalid CandidateGridStatus: {value!r}") from exc
        raise ValueError(
            f"status must be CandidateGridStatus, got {type(value).__name__}"
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

    @field_validator("resolution_status", mode="before")
    @classmethod
    def _validate_resolution_status(cls, value: object) -> ConstraintResolutionStatus:
        if isinstance(value, ConstraintResolutionStatus):
            return value
        if isinstance(value, str):
            try:
                return ConstraintResolutionStatus(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid ConstraintResolutionStatus: {value!r}"
                ) from exc
        raise ValueError(
            "resolution_status must be ConstraintResolutionStatus, "
            f"got {type(value).__name__}"
        )

    @field_validator("candidate_variables", mode="before")
    @classmethod
    def _validate_candidate_variables_before(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"candidate_variables must be a list[str], got {type(value).__name__}"
            )
        return list(value)

    @field_validator("candidate_variables", mode="after")
    @classmethod
    def _validate_candidate_variables(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(
            value,
            field_name="candidate_variables",
        )

    @field_validator("variable_grids", mode="before")
    @classmethod
    def _validate_variable_grids_before(
        cls,
        value: object,
    ) -> list[VariableCandidateGrid]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"variable_grids must be a list[VariableCandidateGrid], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("variable_grids", mode="after")
    @classmethod
    def _validate_variable_grids(
        cls,
        value: list[VariableCandidateGrid],
    ) -> list[VariableCandidateGrid]:
        copied: list[VariableCandidateGrid] = []
        for item in value:
            if not isinstance(item, VariableCandidateGrid):
                raise ValueError(
                    "variable_grids entries must be VariableCandidateGrid, "
                    f"got {type(item).__name__}"
                )
            copied.append(item.model_copy(deep=True))
        return copied

    @field_validator("scenarios", mode="before")
    @classmethod
    def _validate_scenarios_before(cls, value: object) -> list[CandidateScenario]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"scenarios must be a list[CandidateScenario], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("scenarios", mode="after")
    @classmethod
    def _validate_scenarios(
        cls,
        value: list[CandidateScenario],
    ) -> list[CandidateScenario]:
        return [item.model_copy(deep=True) for item in value]

    @field_validator("baseline_scenario_id", mode="before")
    @classmethod
    def _validate_baseline_scenario_id(cls, value: object) -> str | None:
        if value is None:
            return None
        return _require_non_empty_str(value, field_name="baseline_scenario_id")

    @field_validator(
        "potential_scenario_count",
        "generated_scenario_count",
        "change_scenario_count",
        "truncated_scenario_count",
        "effective_combination_limit",
        mode="before",
    )
    @classmethod
    def _validate_non_negative_counts(cls, value: object) -> int:
        return _require_strict_int_ge(value, field_name="count field", minimum=0)

    @field_validator("maximum_scenarios", mode="before")
    @classmethod
    def _validate_maximum_scenarios(cls, value: object) -> int:
        return _require_strict_int_ge(value, field_name="maximum_scenarios", minimum=1)

    @field_validator("generated_at", mode="after")
    @classmethod
    def _validate_generated_at(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value, field_name="generated_at")

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
        grid_names = [grid.variable for grid in self.variable_grids]
        if len(grid_names) != len(set(grid_names)):
            raise ValueError("variable_grids must not contain duplicate variables")
        if grid_names != self.candidate_variables:
            raise ValueError(
                "variable_grids order must exactly match candidate_variables"
            )

        ranks = [grid.diagnosis_rank for grid in self.variable_grids]
        if len(ranks) != len(set(ranks)):
            raise ValueError("diagnosis_rank must be unique across variable_grids")
        if ranks != sorted(ranks):
            raise ValueError("variable_grids must be ordered by ascending diagnosis_rank")

        grid_by_variable = {grid.variable: grid for grid in self.variable_grids}

        scenario_ids = [item.scenario_id for item in self.scenarios]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("scenario IDs must be unique")
        scenario_indexes = [item.scenario_index for item in self.scenarios]
        if len(scenario_indexes) != len(set(scenario_indexes)):
            raise ValueError("scenario indexes must be unique")
        if scenario_indexes != sorted(scenario_indexes):
            raise ValueError("scenarios must be ordered by ascending scenario_index")
        for expected_index, scenario in enumerate(self.scenarios):
            if scenario.scenario_index != expected_index:
                raise ValueError("scenario IDs/indexes must be contiguous from 0")
            if scenario.scenario_id != _scenario_id_for_index(expected_index):
                raise ValueError("scenario IDs must be contiguous SCN-000000 style")

        if self.generated_scenario_count != len(self.scenarios):
            raise ValueError("generated_scenario_count must equal len(scenarios)")
        expected_change = sum(
            1
            for item in self.scenarios
            if item.scenario_type is not CandidateScenarioType.BASELINE
        )
        if self.change_scenario_count != expected_change:
            raise ValueError(
                "change_scenario_count must equal the number of non-baseline scenarios"
            )
        if self.potential_scenario_count < self.generated_scenario_count:
            raise ValueError(
                "potential_scenario_count must be >= generated_scenario_count"
            )
        if self.truncated_scenario_count != (
            self.potential_scenario_count - self.generated_scenario_count
        ):
            raise ValueError(
                "truncated_scenario_count must equal "
                "potential_scenario_count - generated_scenario_count"
            )
        if self.generated_scenario_count > self.maximum_scenarios:
            raise ValueError("generated_scenario_count must be <= maximum_scenarios")

        baselines = [
            item
            for item in self.scenarios
            if item.scenario_type is CandidateScenarioType.BASELINE
        ]
        if len(baselines) > 1:
            raise ValueError("at most one BASELINE scenario is allowed")

        if self.scenarios:
            first = self.scenarios[0]
            if first.scenario_type is not CandidateScenarioType.BASELINE:
                raise ValueError("first scenario must be BASELINE when scenarios exist")
            if first.scenario_index != 0:
                raise ValueError("baseline scenario_index must be 0")
            if self.baseline_scenario_id != first.scenario_id:
                raise ValueError(
                    "baseline_scenario_id must match the first scenario ID"
                )
            if set(first.variable_values.keys()) != set(self.candidate_variables):
                raise ValueError(
                    "baseline variable_values keys must match candidate_variables"
                )
            for name in self.candidate_variables:
                grid = grid_by_variable[name]
                if not math.isclose(
                    first.variable_values[name],
                    grid.current_value,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"baseline value for {name!r} must match grid current_value"
                    )
        elif self.baseline_scenario_id is not None:
            raise ValueError("baseline_scenario_id must be None when scenarios is empty")

        for scenario in self.scenarios:
            if set(scenario.variable_values.keys()) != set(self.candidate_variables):
                raise ValueError(
                    "scenario variable_values keys must match candidate_variables"
                )
            for name, value in scenario.variable_values.items():
                grid = grid_by_variable[name]
                point_values = {point.value for point in grid.points}
                if value not in point_values:
                    raise ValueError(
                        f"scenario value for {name!r} is not a grid point"
                    )

            if scenario.scenario_type is not CandidateScenarioType.BASELINE:
                expected_changed_order = [
                    name
                    for name in self.candidate_variables
                    if name in set(scenario.changed_variables)
                ]
                if scenario.changed_variables != expected_changed_order:
                    raise ValueError(
                        "changed_variables must follow candidate_variables order"
                    )

            magnitude = 0.0
            for name in self.candidate_variables:
                grid = grid_by_variable[name]
                value = scenario.variable_values[name]
                expected_delta = value - grid.current_value
                if name in scenario.changed_variables:
                    if math.isclose(
                        value,
                        grid.current_value,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise ValueError(
                            f"changed variable {name!r} must differ from current_value"
                        )
                    actual_delta = scenario.deltas[name]
                    if not math.isclose(
                        actual_delta,
                        expected_delta,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise ValueError(
                            f"delta for {name!r} must equal value - current_value"
                        )
                    relative = scenario.relative_deltas[name]
                    if grid.current_value == 0.0:
                        if relative is not None and not math.isfinite(relative):
                            raise ValueError(
                                f"relative_delta for {name!r} must be finite or None"
                            )
                    else:
                        expected_relative = expected_delta / abs(grid.current_value)
                        if relative is None or not math.isclose(
                            relative,
                            expected_relative,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        ):
                            raise ValueError(
                                f"relative_delta for {name!r} must equal "
                                "delta / abs(current_value)"
                            )
                    magnitude += abs(actual_delta) / grid.effective_span
                else:
                    if not math.isclose(
                        value,
                        grid.current_value,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise ValueError(
                            f"unchanged variable {name!r} must keep current_value"
                        )

            if not math.isclose(
                scenario.normalized_change_magnitude,
                magnitude,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "normalized_change_magnitude must equal "
                    "sum(abs(delta) / effective_span)"
                )
            if scenario.change_count > self.effective_combination_limit:
                raise ValueError(
                    "change_count must be <= effective_combination_limit"
                )

        safety_refused = self.safety_status is RecommendationSafetyStatus.REFUSED
        resolution_refused = (
            self.resolution_status is ConstraintResolutionStatus.REFUSED
        )

        if self.status is CandidateGridStatus.REFUSED:
            if not (safety_refused or resolution_refused):
                raise ValueError(
                    "REFUSED requires safety_status or resolution_status REFUSED"
                )
            if self.variable_grids or self.scenarios:
                raise ValueError("REFUSED requires empty variable_grids and scenarios")
            if self.generated_scenario_count != 0 or self.change_scenario_count != 0:
                raise ValueError("REFUSED requires zero generated and change counts")
        elif self.status is CandidateGridStatus.EMPTY:
            if self.change_scenario_count != 0:
                raise ValueError("EMPTY requires change_scenario_count=0")
            if len(self.scenarios) > 1:
                raise ValueError(
                    "EMPTY allows at most a single baseline scenario"
                )
            if self.scenarios and (
                self.scenarios[0].scenario_type is not CandidateScenarioType.BASELINE
            ):
                raise ValueError("EMPTY non-empty scenarios must be baseline-only")
        elif self.status is CandidateGridStatus.READY:
            if self.change_scenario_count < 1:
                raise ValueError("READY requires change_scenario_count>=1")
            if self.truncated_scenario_count != 0:
                raise ValueError("READY requires truncated_scenario_count=0")
        elif self.status is CandidateGridStatus.TRUNCATED:
            if self.change_scenario_count < 1:
                raise ValueError("TRUNCATED requires change_scenario_count>=1")
            if self.truncated_scenario_count <= 0:
                raise ValueError("TRUNCATED requires truncated_scenario_count>0")
            if self.generated_scenario_count != self.maximum_scenarios:
                raise ValueError(
                    "TRUNCATED requires generated_scenario_count == maximum_scenarios"
                )
        else:
            raise ValueError(f"unsupported CandidateGridStatus: {self.status!r}")

        return self


@dataclass(frozen=True, slots=True)
class CandidateGridOutcome:
    """Immutable wrapper around a candidate grid report.

    Holds only ``CandidateGridReport``. Contains no business logic and does
    not store models, estimators, or DataFrames.
    """

    report: CandidateGridReport


class CandidateGridGenerator:
    """Deterministically generate bounded grids and unranked scenarios.

    Stateless across calls aside from an immutable policy copy. Does not
    score scenarios, optimize, or mutate the input candidate set.
    """

    def __init__(self, *, policy: CandidateGridPolicy | None = None) -> None:
        if policy is None:
            self._policy = CandidateGridPolicy()
        elif isinstance(policy, CandidateGridPolicy):
            self._policy = policy.model_copy(deep=True)
        else:
            raise TypeError(
                "policy must be CandidateGridPolicy or None, "
                f"got {type(policy).__name__}"
            )

    def generate(self, candidate_set: CandidateVariableSet) -> CandidateGridOutcome:
        """Generate grids and scenarios from a candidate variable set.

        Args:
            candidate_set: Bounded candidates from Step 9B.

        Returns:
            A ``CandidateGridOutcome`` with the immutable grid report.
        """
        if not isinstance(candidate_set, CandidateVariableSet):
            raise TypeError(
                "candidate_set must be CandidateVariableSet, "
                f"got {type(candidate_set).__name__}"
            )

        self._validate_candidate_set(candidate_set)
        policy = self._policy.model_copy(deep=True)
        generated_at = datetime.now(tz=UTC)

        if (
            candidate_set.safety_status is RecommendationSafetyStatus.REFUSED
            or candidate_set.resolution_status is ConstraintResolutionStatus.REFUSED
        ):
            report = self._build_refused_report(
                candidate_set=candidate_set,
                policy=policy,
                generated_at=generated_at,
            )
            return CandidateGridOutcome(report=report)

        if not candidate_set.candidates:
            report = self._build_empty_no_candidate_report(
                candidate_set=candidate_set,
                policy=policy,
                generated_at=generated_at,
            )
            return CandidateGridOutcome(report=report)

        warnings: list[str] = []
        if candidate_set.safety_status is RecommendationSafetyStatus.CAUTION:
            _append_unique(warnings, _WARNING_SAFETY_CAUTION)
        if candidate_set.resolution_status is ConstraintResolutionStatus.PARTIAL:
            _append_unique(warnings, _WARNING_RESOLUTION_PARTIAL)

        variable_grids: list[VariableCandidateGrid] = []
        change_points_by_variable: dict[str, list[CandidateValuePoint]] = {}
        for candidate in candidate_set.candidates:
            grid, ordered_changes, grid_flags = self._build_variable_grid(
                candidate=candidate,
                policy=policy,
            )
            variable_grids.append(grid)
            change_points_by_variable[grid.variable] = ordered_changes
            if grid_flags["dedup"]:
                _append_unique(warnings, _WARNING_DEDUP)
            if grid_flags["removed_nonzero"]:
                _append_unique(warnings, _WARNING_NONZERO_DELTA)
            if grid.change_point_count == 0:
                _append_unique(warnings, _WARNING_NO_CHANGE_POINTS)

        effective_combination_limit = min(
            candidate_set.effective_change_budget,
            policy.maximum_combination_variables,
            len(candidate_set.candidates),
        )
        if effective_combination_limit < candidate_set.effective_change_budget:
            _append_unique(warnings, _WARNING_COMBINATION_CAP)

        if effective_combination_limit == 0:
            report = self._build_empty_with_grids_report(
                candidate_set=candidate_set,
                policy=policy,
                variable_grids=variable_grids,
                effective_combination_limit=0,
                warnings=warnings,
                generated_at=generated_at,
                include_baseline=False,
            )
            return CandidateGridOutcome(report=report)

        potential = self._compute_potential_scenario_count(
            candidate_variables=list(candidate_set.candidate_variables),
            change_points_by_variable=change_points_by_variable,
            policy=policy,
            effective_combination_limit=effective_combination_limit,
        )

        scenarios = self._generate_scenarios(
            candidate_set=candidate_set,
            variable_grids=variable_grids,
            change_points_by_variable=change_points_by_variable,
            policy=policy,
            effective_combination_limit=effective_combination_limit,
        )

        change_count = sum(
            1
            for item in scenarios
            if item.scenario_type is not CandidateScenarioType.BASELINE
        )
        generated_count = len(scenarios)
        truncated_count = potential - generated_count

        if change_count == 0:
            status = CandidateGridStatus.EMPTY
            if scenarios:
                _append_unique(warnings, _WARNING_BASELINE_ONLY)
            else:
                _append_unique(warnings, _WARNING_NO_SCENARIOS)
        elif truncated_count > 0:
            status = CandidateGridStatus.TRUNCATED
            _append_unique(warnings, _WARNING_TRUNCATED)
        else:
            status = CandidateGridStatus.READY

        _append_unique(warnings, _WARNING_MODEL_SCORING)
        _append_unique(warnings, _WARNING_VERIFICATION)

        baseline_id = scenarios[0].scenario_id if scenarios else None
        report = CandidateGridReport(
            status=status,
            objective=candidate_set.objective,
            safety_status=candidate_set.safety_status,
            resolution_status=candidate_set.resolution_status,
            candidate_variables=list(candidate_set.candidate_variables),
            variable_grids=variable_grids,
            scenarios=scenarios,
            baseline_scenario_id=baseline_id,
            potential_scenario_count=potential,
            generated_scenario_count=generated_count,
            change_scenario_count=change_count,
            truncated_scenario_count=truncated_count,
            maximum_scenarios=policy.maximum_scenarios,
            effective_combination_limit=effective_combination_limit,
            generated_at=generated_at,
            warnings=warnings,
            metadata=self._build_report_metadata(
                candidate_set=candidate_set,
                variable_grids=variable_grids,
                potential_scenario_count=potential,
                generated_scenario_count=generated_count,
                change_scenario_count=change_count,
                truncated_scenario_count=truncated_count,
                maximum_scenarios=policy.maximum_scenarios,
                effective_combination_limit=effective_combination_limit,
            ),
        )
        return CandidateGridOutcome(report=report)

    def get_metadata(self) -> dict[str, ScalarMetadataValue]:
        """Return scalar-only generator metadata.

        Returns an independent dict on every call.
        """
        policy = self._policy
        return {
            "linear_points_per_variable": policy.linear_points_per_variable,
            "maximum_scenarios": policy.maximum_scenarios,
            "maximum_combination_variables": policy.maximum_combination_variables,
            "include_single_variable_scenarios": (
                policy.include_single_variable_scenarios
            ),
            "include_multi_variable_scenarios": (
                policy.include_multi_variable_scenarios
            ),
            "prefer_smaller_changes": policy.prefer_smaller_changes,
            "deduplication_tolerance": policy.deduplication_tolerance,
            "minimum_nonzero_delta": policy.minimum_nonzero_delta,
            "candidate_values_generated": True,
            "model_scoring_performed": False,
            "optimization_performed": False,
            "deterministic_generation": True,
        }

    def _validate_candidate_set(self, candidate_set: CandidateVariableSet) -> None:
        names = [item.variable for item in candidate_set.candidates]
        if names != list(candidate_set.candidate_variables):
            raise DataValidationError(
                "candidate_variables must match candidates variable order exactly"
            )

        seen_ranks: set[int] = set()
        previous_rank = 0
        for item in candidate_set.candidates:
            if item.diagnosis_rank in seen_ranks:
                raise DataValidationError(
                    f"duplicate diagnosis_rank: {item.diagnosis_rank}"
                )
            if item.diagnosis_rank < previous_rank:
                raise DataValidationError(
                    "candidates must be ordered by ascending diagnosis_rank"
                )
            seen_ranks.add(item.diagnosis_rank)
            previous_rank = item.diagnosis_rank

            if item.minimum > item.current_value or item.current_value > item.maximum:
                raise DataValidationError(
                    f"candidate {item.variable!r} violates "
                    "minimum <= current_value <= maximum"
                )
            span = item.maximum - item.minimum
            if span <= 0.0:
                raise DataValidationError(
                    f"candidate {item.variable!r} must have positive effective span"
                )

        n = len(candidate_set.candidates)
        if n == 0:
            if candidate_set.effective_change_budget != 0:
                raise DataValidationError(
                    "effective_change_budget must be 0 when candidates is empty"
                )
        else:
            expected_budget = min(candidate_set.max_simultaneous_changes, n)
            if candidate_set.effective_change_budget != expected_budget:
                raise DataValidationError(
                    "effective_change_budget must equal "
                    "min(max_simultaneous_changes, len(candidates))"
                )
            if candidate_set.effective_change_budget > n:
                raise DataValidationError(
                    "effective_change_budget must be <= len(candidates)"
                )

        if candidate_set.safety_status is RecommendationSafetyStatus.REFUSED and n:
            raise DataValidationError(
                "safety_status REFUSED requires empty candidates"
            )
        if (
            candidate_set.resolution_status is ConstraintResolutionStatus.REFUSED
            and n
        ):
            raise DataValidationError(
                "resolution_status REFUSED requires empty candidates"
            )

    def _build_variable_grid(
        self,
        *,
        candidate: CandidateVariable,
        policy: CandidateGridPolicy,
    ) -> tuple[VariableCandidateGrid, list[CandidateValuePoint], dict[str, bool]]:
        minimum = float(candidate.minimum)
        maximum = float(candidate.maximum)
        current_value = float(candidate.current_value)
        effective_span = maximum - minimum
        if effective_span <= 0.0:
            raise DataValidationError(
                f"candidate {candidate.variable!r} must have positive effective span"
            )

        linspace_values = np.linspace(
            minimum,
            maximum,
            policy.linear_points_per_variable,
            dtype=float,
        )
        raw_values = [float(value) for value in linspace_values]
        raw_values.append(current_value)

        for value in raw_values:
            if value < minimum or value > maximum:
                if not (
                    math.isclose(value, minimum, rel_tol=0.0, abs_tol=1e-12)
                    or math.isclose(value, maximum, rel_tol=0.0, abs_tol=1e-12)
                ):
                    raise DataValidationError(
                        f"grid value {value} for {candidate.variable!r} is outside "
                        f"[{minimum}, {maximum}]"
                    )

        deduped, dedup_occurred = self._deduplicate_values(
            values=raw_values,
            current_value=current_value,
            minimum=minimum,
            maximum=maximum,
            tolerance=policy.deduplication_tolerance,
        )

        points: list[CandidateValuePoint] = []
        removed_nonzero = False
        for value in deduped:
            delta = value - current_value
            is_current = math.isclose(
                value,
                current_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            is_lower = math.isclose(value, minimum, rel_tol=0.0, abs_tol=1e-12)
            is_upper = math.isclose(value, maximum, rel_tol=0.0, abs_tol=1e-12)

            if (
                not is_current
                and not is_lower
                and not is_upper
                and abs(delta) <= policy.minimum_nonzero_delta
            ):
                removed_nonzero = True
                continue

            if is_current:
                exact_value = current_value
                delta = 0.0
            elif is_lower:
                exact_value = minimum
                delta = exact_value - current_value
            elif is_upper:
                exact_value = maximum
                delta = exact_value - current_value
            else:
                exact_value = value

            if (
                not is_current
                and abs(delta) <= policy.minimum_nonzero_delta
                and not (is_lower or is_upper)
            ):
                removed_nonzero = True
                continue

            # Boundary points with tiny delta are retained but treated as
            # non-change for scenario generation via is_current when equal.
            if is_current:
                is_current_flag = True
            else:
                is_current_flag = False

            # Recompute flags after canonicalization.
            is_lower = math.isclose(exact_value, minimum, rel_tol=0.0, abs_tol=1e-12)
            is_upper = math.isclose(exact_value, maximum, rel_tol=0.0, abs_tol=1e-12)
            is_current_flag = math.isclose(
                exact_value,
                current_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            delta = exact_value - current_value
            if is_current_flag:
                delta = 0.0

            if current_value == 0.0:
                relative_delta: float | None = None
            else:
                relative_delta = delta / abs(current_value)

            points.append(
                CandidateValuePoint(
                    value=exact_value,
                    delta=delta,
                    relative_delta=relative_delta,
                    normalized_position=(exact_value - minimum) / effective_span,
                    is_current=is_current_flag,
                    is_lower_bound=is_lower,
                    is_upper_bound=is_upper,
                )
            )

        # Ensure anchors survived nonzero filtering.
        points = self._ensure_anchor_points(
            points=points,
            current_value=current_value,
            minimum=minimum,
            maximum=maximum,
            effective_span=effective_span,
        )
        points.sort(key=lambda item: item.value)

        # Exact-value dedupe after anchor enforcement.
        unique_points: list[CandidateValuePoint] = []
        for point in points:
            if unique_points and point.value == unique_points[-1].value:
                # Prefer current/bound flags already computed.
                continue
            unique_points.append(point)
        points = unique_points

        grid = VariableCandidateGrid(
            variable=candidate.variable,
            diagnosis_rank=candidate.diagnosis_rank,
            current_value=current_value,
            minimum=minimum,
            maximum=maximum,
            effective_span=effective_span,
            points=points,
            point_count=len(points),
            change_point_count=sum(1 for point in points if not point.is_current),
            warnings=[],
        )
        ordered_changes = self._order_change_points(
            points=points,
            effective_span=effective_span,
            prefer_smaller_changes=policy.prefer_smaller_changes,
            minimum_nonzero_delta=policy.minimum_nonzero_delta,
        )
        return (
            grid,
            ordered_changes,
            {"dedup": dedup_occurred, "removed_nonzero": removed_nonzero},
        )

    def _ensure_anchor_points(
        self,
        *,
        points: list[CandidateValuePoint],
        current_value: float,
        minimum: float,
        maximum: float,
        effective_span: float,
    ) -> list[CandidateValuePoint]:
        def _make_point(value: float) -> CandidateValuePoint:
            is_current = math.isclose(
                value,
                current_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            delta = 0.0 if is_current else value - current_value
            if current_value == 0.0:
                relative: float | None = None
            else:
                relative = delta / abs(current_value)
            return CandidateValuePoint(
                value=value,
                delta=delta,
                relative_delta=relative,
                normalized_position=(value - minimum) / effective_span,
                is_current=is_current,
                is_lower_bound=math.isclose(
                    value,
                    minimum,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                is_upper_bound=math.isclose(
                    value,
                    maximum,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
            )

        present = {point.value for point in points}
        result = list(points)
        for anchor in (minimum, maximum, current_value):
            if not any(
                math.isclose(value, anchor, rel_tol=0.0, abs_tol=1e-12)
                for value in present
            ):
                result.append(_make_point(anchor))
                present.add(anchor)
        return result

    def _deduplicate_values(
        self,
        *,
        values: list[float],
        current_value: float,
        minimum: float,
        maximum: float,
        tolerance: float,
    ) -> tuple[list[float], bool]:
        sorted_values = sorted(float(value) for value in values)
        merged: list[float] = []
        dedup_occurred = False
        i = 0
        while i < len(sorted_values):
            cluster = [sorted_values[i]]
            j = i + 1
            while j < len(sorted_values):
                # Never merge minimum with maximum even when within tolerance.
                if (
                    math.isclose(cluster[0], minimum, rel_tol=0.0, abs_tol=1e-12)
                    and math.isclose(
                        sorted_values[j],
                        maximum,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ):
                    break
                if any(
                    math.isclose(item, minimum, rel_tol=0.0, abs_tol=1e-12)
                    for item in cluster
                ) and math.isclose(
                    sorted_values[j],
                    maximum,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    break
                if abs(sorted_values[j] - cluster[-1]) <= tolerance:
                    cluster.append(sorted_values[j])
                    j += 1
                    continue
                break

            if len(cluster) > 1:
                dedup_occurred = True
            canonical = self._canonical_cluster_value(
                cluster=cluster,
                current_value=current_value,
                minimum=minimum,
                maximum=maximum,
            )
            if not merged or canonical != merged[-1]:
                # Preserve separate min/max even if somehow adjacent.
                if (
                    merged
                    and math.isclose(merged[-1], minimum, rel_tol=0.0, abs_tol=1e-12)
                    and math.isclose(canonical, maximum, rel_tol=0.0, abs_tol=1e-12)
                ):
                    merged.append(canonical)
                elif not merged or abs(canonical - merged[-1]) > tolerance:
                    merged.append(canonical)
                else:
                    preferred = self._canonical_cluster_value(
                        cluster=[merged[-1], canonical],
                        current_value=current_value,
                        minimum=minimum,
                        maximum=maximum,
                    )
                    if (
                        math.isclose(merged[-1], minimum, rel_tol=0.0, abs_tol=1e-12)
                        and math.isclose(
                            preferred,
                            maximum,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                    ):
                        merged.append(preferred)
                    else:
                        merged[-1] = preferred
                        dedup_occurred = True
            i = j if j > i else i + 1

        # Force exact anchors.
        def _force_anchor(target: list[float], anchor: float) -> None:
            for idx, value in enumerate(target):
                if abs(value - anchor) <= tolerance or math.isclose(
                    value,
                    anchor,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    # Do not replace maximum with minimum or vice versa.
                    if math.isclose(anchor, minimum, rel_tol=0.0, abs_tol=1e-12) and (
                        math.isclose(value, maximum, rel_tol=0.0, abs_tol=1e-12)
                        and not math.isclose(
                            minimum,
                            maximum,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                    ):
                        continue
                    if math.isclose(anchor, maximum, rel_tol=0.0, abs_tol=1e-12) and (
                        math.isclose(value, minimum, rel_tol=0.0, abs_tol=1e-12)
                        and not math.isclose(
                            minimum,
                            maximum,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                    ):
                        continue
                    target[idx] = anchor
                    return
            target.append(anchor)

        _force_anchor(merged, minimum)
        _force_anchor(merged, maximum)
        _force_anchor(merged, current_value)
        merged = sorted(set(merged))
        # Ensure min and max both present as exact values.
        if minimum not in merged:
            merged.append(minimum)
        if maximum not in merged:
            merged.append(maximum)
        if current_value not in merged and not any(
            math.isclose(value, current_value, rel_tol=0.0, abs_tol=1e-12)
            for value in merged
        ):
            merged.append(current_value)
        # Replace near-current with exact current when isclose.
        replaced: list[float] = []
        for value in sorted(merged):
            if math.isclose(value, current_value, rel_tol=0.0, abs_tol=1e-12):
                if current_value not in replaced:
                    replaced.append(current_value)
                continue
            if math.isclose(value, minimum, rel_tol=0.0, abs_tol=1e-12):
                if minimum not in replaced:
                    replaced.append(minimum)
                continue
            if math.isclose(value, maximum, rel_tol=0.0, abs_tol=1e-12):
                if maximum not in replaced:
                    replaced.append(maximum)
                continue
            replaced.append(value)
        return sorted(replaced), dedup_occurred

    def _canonical_cluster_value(
        self,
        *,
        cluster: list[float],
        current_value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        for anchor in (current_value, minimum, maximum):
            if any(
                math.isclose(value, anchor, rel_tol=0.0, abs_tol=1e-12)
                for value in cluster
            ):
                return float(anchor)
        return float(min(cluster))

    def _order_change_points(
        self,
        *,
        points: list[CandidateValuePoint],
        effective_span: float,
        prefer_smaller_changes: bool,
        minimum_nonzero_delta: float,
    ) -> list[CandidateValuePoint]:
        changes = [
            point
            for point in points
            if not point.is_current and abs(point.delta) > minimum_nonzero_delta
        ]
        if prefer_smaller_changes:
            changes.sort(
                key=lambda point: (
                    abs(point.delta) / effective_span,
                    0 if point.delta < 0.0 else 1,
                    point.value,
                )
            )
        else:
            changes.sort(key=lambda point: point.value)
        return changes

    def _compute_potential_scenario_count(
        self,
        *,
        candidate_variables: list[str],
        change_points_by_variable: dict[str, list[CandidateValuePoint]],
        policy: CandidateGridPolicy,
        effective_combination_limit: int,
    ) -> int:
        total = 1  # baseline
        if policy.include_single_variable_scenarios and effective_combination_limit >= 1:
            total += sum(
                len(change_points_by_variable[name]) for name in candidate_variables
            )
        if (
            policy.include_multi_variable_scenarios
            and effective_combination_limit >= 2
        ):
            for change_count in range(2, effective_combination_limit + 1):
                for combo in itertools.combinations(candidate_variables, change_count):
                    product = 1
                    for name in combo:
                        product *= len(change_points_by_variable[name])
                    total += product
        return total

    def _generate_scenarios(
        self,
        *,
        candidate_set: CandidateVariableSet,
        variable_grids: list[VariableCandidateGrid],
        change_points_by_variable: dict[str, list[CandidateValuePoint]],
        policy: CandidateGridPolicy,
        effective_combination_limit: int,
    ) -> list[CandidateScenario]:
        grid_by_variable = {grid.variable: grid for grid in variable_grids}
        candidate_variables = list(candidate_set.candidate_variables)
        current_values = {
            grid.variable: grid.current_value for grid in variable_grids
        }

        scenarios: list[CandidateScenario] = []
        seen_signatures: set[tuple[float, ...]] = set()
        max_scenarios = policy.maximum_scenarios

        def _try_add(
            *,
            scenario_type: CandidateScenarioType,
            values: dict[str, float],
            changed: list[str],
        ) -> bool:
            if len(scenarios) >= max_scenarios:
                return False
            signature = tuple(values[name] for name in candidate_variables)
            if signature in seen_signatures:
                return True
            seen_signatures.add(signature)
            index = len(scenarios)
            deltas: dict[str, float] = {}
            relative_deltas: dict[str, float | None] = {}
            magnitude = 0.0
            for name in changed:
                grid = grid_by_variable[name]
                delta = values[name] - grid.current_value
                deltas[name] = delta
                if grid.current_value == 0.0:
                    relative_deltas[name] = None
                else:
                    relative_deltas[name] = delta / abs(grid.current_value)
                magnitude += abs(delta) / grid.effective_span
            scenarios.append(
                CandidateScenario(
                    scenario_id=_scenario_id_for_index(index),
                    scenario_index=index,
                    scenario_type=scenario_type,
                    variable_values=dict(values),
                    changed_variables=list(changed),
                    deltas=deltas,
                    relative_deltas=relative_deltas,
                    change_count=len(changed),
                    normalized_change_magnitude=float(magnitude),
                )
            )
            return len(scenarios) < max_scenarios

        baseline_values = {
            name: current_values[name] for name in candidate_variables
        }
        _try_add(
            scenario_type=CandidateScenarioType.BASELINE,
            values=baseline_values,
            changed=[],
        )

        if (
            policy.include_single_variable_scenarios
            and effective_combination_limit >= 1
            and len(scenarios) < max_scenarios
        ):
            for name in candidate_variables:
                for point in change_points_by_variable[name]:
                    values = {
                        key: current_values[key] for key in candidate_variables
                    }
                    values[name] = point.value
                    if not _try_add(
                        scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
                        values=values,
                        changed=[name],
                    ):
                        return scenarios

        if (
            policy.include_multi_variable_scenarios
            and effective_combination_limit >= 2
            and len(scenarios) < max_scenarios
        ):
            for change_count in range(2, effective_combination_limit + 1):
                for combo in itertools.combinations(
                    candidate_variables,
                    change_count,
                ):
                    point_lists = [
                        change_points_by_variable[name] for name in combo
                    ]
                    for point_combo in itertools.product(*point_lists):
                        values = {
                            key: current_values[key] for key in candidate_variables
                        }
                        changed = list(combo)
                        for name, point in zip(combo, point_combo, strict=True):
                            values[name] = point.value
                        if not _try_add(
                            scenario_type=CandidateScenarioType.MULTI_VARIABLE,
                            values=values,
                            changed=changed,
                        ):
                            return scenarios
        return scenarios

    def _build_refused_report(
        self,
        *,
        candidate_set: CandidateVariableSet,
        policy: CandidateGridPolicy,
        generated_at: datetime,
    ) -> CandidateGridReport:
        warnings: list[str] = []
        _append_unique(warnings, _WARNING_NO_SCENARIOS)
        _append_unique(warnings, _WARNING_MODEL_SCORING)
        _append_unique(warnings, _WARNING_VERIFICATION)
        return CandidateGridReport(
            status=CandidateGridStatus.REFUSED,
            objective=candidate_set.objective,
            safety_status=candidate_set.safety_status,
            resolution_status=candidate_set.resolution_status,
            candidate_variables=[],
            variable_grids=[],
            scenarios=[],
            baseline_scenario_id=None,
            potential_scenario_count=0,
            generated_scenario_count=0,
            change_scenario_count=0,
            truncated_scenario_count=0,
            maximum_scenarios=policy.maximum_scenarios,
            effective_combination_limit=0,
            generated_at=generated_at,
            warnings=warnings,
            metadata=self._build_report_metadata(
                candidate_set=candidate_set,
                variable_grids=[],
                potential_scenario_count=0,
                generated_scenario_count=0,
                change_scenario_count=0,
                truncated_scenario_count=0,
                maximum_scenarios=policy.maximum_scenarios,
                effective_combination_limit=0,
            ),
        )

    def _build_empty_no_candidate_report(
        self,
        *,
        candidate_set: CandidateVariableSet,
        policy: CandidateGridPolicy,
        generated_at: datetime,
    ) -> CandidateGridReport:
        warnings: list[str] = []
        if candidate_set.safety_status is RecommendationSafetyStatus.CAUTION:
            _append_unique(warnings, _WARNING_SAFETY_CAUTION)
        if candidate_set.resolution_status is ConstraintResolutionStatus.PARTIAL:
            _append_unique(warnings, _WARNING_RESOLUTION_PARTIAL)
        _append_unique(warnings, _WARNING_NO_SCENARIOS)
        _append_unique(warnings, _WARNING_MODEL_SCORING)
        _append_unique(warnings, _WARNING_VERIFICATION)
        return CandidateGridReport(
            status=CandidateGridStatus.EMPTY,
            objective=candidate_set.objective,
            safety_status=candidate_set.safety_status,
            resolution_status=candidate_set.resolution_status,
            candidate_variables=[],
            variable_grids=[],
            scenarios=[],
            baseline_scenario_id=None,
            potential_scenario_count=0,
            generated_scenario_count=0,
            change_scenario_count=0,
            truncated_scenario_count=0,
            maximum_scenarios=policy.maximum_scenarios,
            effective_combination_limit=0,
            generated_at=generated_at,
            warnings=warnings,
            metadata=self._build_report_metadata(
                candidate_set=candidate_set,
                variable_grids=[],
                potential_scenario_count=0,
                generated_scenario_count=0,
                change_scenario_count=0,
                truncated_scenario_count=0,
                maximum_scenarios=policy.maximum_scenarios,
                effective_combination_limit=0,
            ),
        )

    def _build_empty_with_grids_report(
        self,
        *,
        candidate_set: CandidateVariableSet,
        policy: CandidateGridPolicy,
        variable_grids: list[VariableCandidateGrid],
        effective_combination_limit: int,
        warnings: list[str],
        generated_at: datetime,
        include_baseline: bool,
    ) -> CandidateGridReport:
        scenarios: list[CandidateScenario] = []
        if include_baseline and variable_grids:
            values = {
                grid.variable: grid.current_value for grid in variable_grids
            }
            scenarios.append(
                CandidateScenario(
                    scenario_id=_scenario_id_for_index(0),
                    scenario_index=0,
                    scenario_type=CandidateScenarioType.BASELINE,
                    variable_values=values,
                    changed_variables=[],
                    deltas={},
                    relative_deltas={},
                    change_count=0,
                    normalized_change_magnitude=0.0,
                )
            )
            _append_unique(warnings, _WARNING_BASELINE_ONLY)
            potential = 1
        else:
            _append_unique(warnings, _WARNING_NO_SCENARIOS)
            potential = 0

        _append_unique(warnings, _WARNING_MODEL_SCORING)
        _append_unique(warnings, _WARNING_VERIFICATION)
        return CandidateGridReport(
            status=CandidateGridStatus.EMPTY,
            objective=candidate_set.objective,
            safety_status=candidate_set.safety_status,
            resolution_status=candidate_set.resolution_status,
            candidate_variables=list(candidate_set.candidate_variables),
            variable_grids=variable_grids,
            scenarios=scenarios,
            baseline_scenario_id=(
                scenarios[0].scenario_id if scenarios else None
            ),
            potential_scenario_count=potential,
            generated_scenario_count=len(scenarios),
            change_scenario_count=0,
            truncated_scenario_count=potential - len(scenarios),
            maximum_scenarios=policy.maximum_scenarios,
            effective_combination_limit=effective_combination_limit,
            generated_at=generated_at,
            warnings=warnings,
            metadata=self._build_report_metadata(
                candidate_set=candidate_set,
                variable_grids=variable_grids,
                potential_scenario_count=potential,
                generated_scenario_count=len(scenarios),
                change_scenario_count=0,
                truncated_scenario_count=potential - len(scenarios),
                maximum_scenarios=policy.maximum_scenarios,
                effective_combination_limit=effective_combination_limit,
            ),
        )

    def _build_report_metadata(
        self,
        *,
        candidate_set: CandidateVariableSet,
        variable_grids: list[VariableCandidateGrid],
        potential_scenario_count: int,
        generated_scenario_count: int,
        change_scenario_count: int,
        truncated_scenario_count: int,
        maximum_scenarios: int,
        effective_combination_limit: int,
    ) -> dict[str, ScalarMetadataValue]:
        return {
            "candidate_variable_count": len(candidate_set.candidate_variables),
            "variable_grid_count": len(variable_grids),
            "total_grid_point_count": sum(grid.point_count for grid in variable_grids),
            "total_change_point_count": sum(
                grid.change_point_count for grid in variable_grids
            ),
            "potential_scenario_count": potential_scenario_count,
            "generated_scenario_count": generated_scenario_count,
            "change_scenario_count": change_scenario_count,
            "truncated_scenario_count": truncated_scenario_count,
            "maximum_scenarios": maximum_scenarios,
            "effective_combination_limit": effective_combination_limit,
            "candidate_values_generated": True,
            "model_scoring_performed": False,
            "optimization_performed": False,
            "recommendation_generated": False,
            "scenarios_are_unranked": True,
            "deterministic_generation": True,
            "continuous_bounds": True,
        }
