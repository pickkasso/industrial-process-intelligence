"""Constraint resolution for recommendation optimization candidates (Step 9B).

Merges industry, request, and user-override ``VariableConstraint`` sources into
per-variable effective numeric bounds. Does not generate proposed values,
grid points, or run optimization.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator

from process_intelligence.core.exceptions import DataValidationError
from process_intelligence.core.schemas import VariableConstraint
from process_intelligence.recommendation.enums import RecommendationSafetyStatus
from process_intelligence.recommendation.schemas import (
    RecommendationRequest,
    RecommendationSafetyDecision,
    ScalarMetadataValue,
)

_ORIGINAL_ROW_ID = "_original_row_id"

_SAFETY_REFUSED_ISSUE_VARIABLE = "*"

_ALLOWED_ISSUE_CODES = frozenset(
    {
        "SAFETY_DECISION_REFUSED",
        "SAFETY_VARIABLE_NOT_ELIGIBLE",
        "CONSTRAINT_MISSING",
        "CONFLICTING_BOUNDS",
        "USER_OVERRIDE_WIDENS_BOUNDS",
        "CURRENT_VALUE_MISSING",
        "CURRENT_VALUE_OUTSIDE_BOUNDS",
        "EFFECTIVE_SPAN_TOO_SMALL",
        "DUPLICATE_SOURCE_CONSTRAINT",
        "UNSUPPORTED_CONSTRAINT_SHAPE",
    }
)

_ISSUE_CODE_SORT_ORDER: dict[str, int] = {
    "SAFETY_DECISION_REFUSED": 0,
    "SAFETY_VARIABLE_NOT_ELIGIBLE": 1,
    "DUPLICATE_SOURCE_CONSTRAINT": 2,
    "UNSUPPORTED_CONSTRAINT_SHAPE": 2,
    "CONSTRAINT_MISSING": 3,
    "CONFLICTING_BOUNDS": 4,
    "USER_OVERRIDE_WIDENS_BOUNDS": 5,
    "CURRENT_VALUE_MISSING": 6,
    "CURRENT_VALUE_OUTSIDE_BOUNDS": 6,
    "EFFECTIVE_SPAN_TOO_SMALL": 7,
}


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


def _require_non_negative_finite_float(value: object, *, field_name: str) -> float:
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


class ConstraintSource(StrEnum):
    """Origin of a numeric variable constraint used during resolution.

    Distinguishes industry defaults, request-scoped constraints, and explicit
    operator overrides. Sources are merged into effective bounds; they do not
    invent proposed recommendation values.
    """

    INDUSTRY_DEFAULT = "INDUSTRY_DEFAULT"
    REQUEST = "REQUEST"
    USER_OVERRIDE = "USER_OVERRIDE"


_SOURCE_CANONICAL_ORDER = (
    ConstraintSource.INDUSTRY_DEFAULT,
    ConstraintSource.REQUEST,
    ConstraintSource.USER_OVERRIDE,
)


class ConstraintResolutionStatus(StrEnum):
    """Outcome of merging constraints for safety-eligible variables.

    ``READY`` means every eligible variable has usable bounded constraints.
    ``PARTIAL`` means some variables resolved while others did not.
    ``REFUSED`` means no usable candidate constraints exist or safety refused.
    """

    READY = "READY"
    PARTIAL = "PARTIAL"
    REFUSED = "REFUSED"


class ConstraintResolutionPolicy(BaseModel):
    """Tunable policy for merging industry, request, and user constraints.

    Controls intersection rules, user-bound widening, current-value checks,
    and partial resolution. Does not generate candidate values or optimize.
    """

    require_non_refused_safety_decision: bool = True
    intersect_industry_and_request_constraints: bool = True
    allow_user_bound_widening: bool = False
    require_current_value_within_bounds: bool = True
    minimum_effective_span: float = 1e-12
    allow_partial_resolution: bool = True
    preserve_safety_ranking: bool = True

    @field_validator(
        "require_non_refused_safety_decision",
        "intersect_industry_and_request_constraints",
        "allow_user_bound_widening",
        "require_current_value_within_bounds",
        "allow_partial_resolution",
        "preserve_safety_ranking",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="policy bool field")

    @field_validator("minimum_effective_span", mode="before")
    @classmethod
    def _validate_minimum_effective_span(cls, value: object) -> float:
        return _require_non_negative_finite_float(
            value,
            field_name="minimum_effective_span",
        )


class ConstraintResolutionIssue(BaseModel):
    """Structured finding for a variable during constraint resolution.

    Captures blocking and non-blocking issues such as missing bounds,
    conflicting intersections, or current values outside effective ranges.
    """

    variable: str
    source: ConstraintSource | None = None
    code: str
    message: str
    blocking: bool

    @field_validator("variable", mode="before")
    @classmethod
    def _validate_variable(cls, value: object) -> str:
        return _validate_variable_name(value, field_name="variable")

    @field_validator("source", mode="before")
    @classmethod
    def _validate_source(cls, value: object) -> ConstraintSource | None:
        if value is None:
            return None
        if isinstance(value, ConstraintSource):
            return value
        if isinstance(value, str):
            try:
                return ConstraintSource(value)
            except ValueError as exc:
                raise ValueError(f"invalid ConstraintSource: {value!r}") from exc
        raise ValueError(
            f"source must be ConstraintSource or None, got {type(value).__name__}"
        )

    @field_validator("code", mode="before")
    @classmethod
    def _validate_code(cls, value: object) -> str:
        text = _require_non_empty_str(value, field_name="code")
        if text not in _ALLOWED_ISSUE_CODES:
            raise ValueError(
                f"code must be one of {sorted(_ALLOWED_ISSUE_CODES)}, got {text!r}"
            )
        return text

    @field_validator("message", mode="before")
    @classmethod
    def _validate_message(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="message")

    @field_validator("blocking", mode="before")
    @classmethod
    def _validate_blocking(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="blocking")


class ResolvedVariableConstraint(BaseModel):
    """Effective bounded numeric constraint for one optimization candidate.

    Requires finite minimum and maximum with the current value inside the
    range. Does not invent missing bounds or proposed values.
    """

    variable: str
    current_value: float
    minimum: float
    maximum: float
    effective_span: float
    lower_room: float
    upper_room: float
    relative_position: float
    source_chain: list[ConstraintSource]
    industry_constraint_present: bool
    request_constraint_present: bool
    user_override_present: bool
    user_override_widened_bounds: bool
    warnings: list[str] = Field(default_factory=list)

    @field_validator("variable", mode="before")
    @classmethod
    def _validate_variable(cls, value: object) -> str:
        return _validate_variable_name(value, field_name="variable")

    @field_validator(
        "current_value",
        "minimum",
        "maximum",
        "effective_span",
        "lower_room",
        "upper_room",
        "relative_position",
        mode="before",
    )
    @classmethod
    def _validate_numeric_fields(cls, value: object) -> float:
        return _require_finite_float(value, field_name="numeric constraint field")

    @field_validator("source_chain", mode="before")
    @classmethod
    def _validate_source_chain_before(cls, value: object) -> list[object]:
        if not isinstance(value, list):
            raise ValueError(
                f"source_chain must be a list[ConstraintSource], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("source_chain", mode="after")
    @classmethod
    def _validate_source_chain(
        cls,
        value: list[ConstraintSource],
    ) -> list[ConstraintSource]:
        if not value:
            raise ValueError("source_chain must contain at least one source")
        cleaned: list[ConstraintSource] = []
        seen: set[ConstraintSource] = set()
        for item in value:
            if isinstance(item, ConstraintSource):
                source = item
            elif isinstance(item, str):
                try:
                    source = ConstraintSource(item)
                except ValueError as exc:
                    raise ValueError(f"invalid ConstraintSource: {item!r}") from exc
            else:
                raise ValueError(
                    "source_chain entries must be ConstraintSource, "
                    f"got {type(item).__name__}"
                )
            if source in seen:
                raise ValueError(
                    f"source_chain must not contain duplicates: {source!r}"
                )
            seen.add(source)
            cleaned.append(source)

        order_index = {source: idx for idx, source in enumerate(_SOURCE_CANONICAL_ORDER)}
        previous = -1
        for source in cleaned:
            idx = order_index[source]
            if idx < previous:
                raise ValueError(
                    "source_chain must follow canonical order "
                    "INDUSTRY_DEFAULT, REQUEST, USER_OVERRIDE"
                )
            previous = idx
        return cleaned

    @field_validator(
        "industry_constraint_present",
        "request_constraint_present",
        "user_override_present",
        "user_override_widened_bounds",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="resolved bool field")

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
    def _validate_resolved_consistency(self) -> Self:
        if self.minimum > self.current_value:
            raise ValueError(
                "minimum must be <= current_value "
                f"(got minimum={self.minimum}, current_value={self.current_value})"
            )
        if self.current_value > self.maximum:
            raise ValueError(
                "current_value must be <= maximum "
                f"(got current_value={self.current_value}, maximum={self.maximum})"
            )
        expected_span = self.maximum - self.minimum
        if self.effective_span <= 0.0:
            raise ValueError(
                f"effective_span must be > 0, got {self.effective_span}"
            )
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
        expected_lower = self.current_value - self.minimum
        expected_upper = self.maximum - self.current_value
        if not math.isclose(
            self.lower_room,
            expected_lower,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "lower_room must equal current_value - minimum "
                f"(got {self.lower_room}, expected {expected_lower})"
            )
        if not math.isclose(
            self.upper_room,
            expected_upper,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "upper_room must equal maximum - current_value "
                f"(got {self.upper_room}, expected {expected_upper})"
            )
        if self.lower_room < 0.0 or self.upper_room < 0.0:
            raise ValueError("lower_room and upper_room must be >= 0")
        if self.relative_position < 0.0 or self.relative_position > 1.0:
            raise ValueError(
                f"relative_position must be in [0.0, 1.0], got {self.relative_position}"
            )
        expected_relative = self.lower_room / self.effective_span
        if not math.isclose(
            self.relative_position,
            expected_relative,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "relative_position must equal lower_room / effective_span "
                f"(got {self.relative_position}, expected {expected_relative})"
            )

        present = {
            ConstraintSource.INDUSTRY_DEFAULT: self.industry_constraint_present,
            ConstraintSource.REQUEST: self.request_constraint_present,
            ConstraintSource.USER_OVERRIDE: self.user_override_present,
        }
        for source in _SOURCE_CANONICAL_ORDER:
            in_chain = source in self.source_chain
            if present[source] != in_chain:
                raise ValueError(
                    f"source presence flag for {source.value} must match source_chain"
                )

        if self.user_override_widened_bounds and not self.user_override_present:
            raise ValueError(
                "user_override_widened_bounds=True requires user_override_present=True"
            )
        return self


class ConstraintResolutionReport(BaseModel):
    """Structured report of constraint resolution across eligible variables.

    Separates resolved and unresolved variables, records issues, and never
    embeds models, estimators, or DataFrames.
    """

    status: ConstraintResolutionStatus
    safety_status: RecommendationSafetyStatus
    requested_eligible_variables: list[str]
    resolved_variables: list[str]
    unresolved_variables: list[str]
    resolved_constraints: list[ResolvedVariableConstraint]
    issues: list[ConstraintResolutionIssue]
    evaluated_at: datetime
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)

    @field_validator("status", mode="before")
    @classmethod
    def _validate_status(cls, value: object) -> ConstraintResolutionStatus:
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
            f"status must be ConstraintResolutionStatus, got {type(value).__name__}"
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

    @field_validator(
        "requested_eligible_variables",
        "resolved_variables",
        "unresolved_variables",
        mode="before",
    )
    @classmethod
    def _validate_variable_lists_before(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"variable list must be a list[str], got {type(value).__name__}"
            )
        return list(value)

    @field_validator(
        "requested_eligible_variables",
        "resolved_variables",
        "unresolved_variables",
        mode="after",
    )
    @classmethod
    def _validate_variable_lists(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(value, field_name="variable list")

    @field_validator("resolved_constraints", mode="before")
    @classmethod
    def _validate_constraints_before(
        cls,
        value: object,
    ) -> list[ResolvedVariableConstraint]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                "resolved_constraints must be a list[ResolvedVariableConstraint], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("resolved_constraints", mode="after")
    @classmethod
    def _validate_constraints(
        cls,
        value: list[ResolvedVariableConstraint],
    ) -> list[ResolvedVariableConstraint]:
        seen: set[str] = set()
        copied: list[ResolvedVariableConstraint] = []
        for item in value:
            if not isinstance(item, ResolvedVariableConstraint):
                raise ValueError(
                    "resolved_constraints entries must be ResolvedVariableConstraint, "
                    f"got {type(item).__name__}"
                )
            if item.variable in seen:
                raise ValueError(
                    "resolved_constraints must not contain duplicate variables: "
                    f"{item.variable!r}"
                )
            seen.add(item.variable)
            copied.append(item.model_copy(deep=True))
        return copied

    @field_validator("issues", mode="before")
    @classmethod
    def _validate_issues_before(cls, value: object) -> list[ConstraintResolutionIssue]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                "issues must be a list[ConstraintResolutionIssue], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("issues", mode="after")
    @classmethod
    def _validate_issues(
        cls,
        value: list[ConstraintResolutionIssue],
    ) -> list[ConstraintResolutionIssue]:
        return [item.model_copy(deep=True) for item in value]

    @field_validator("evaluated_at", mode="after")
    @classmethod
    def _validate_evaluated_at(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value, field_name="evaluated_at")

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
        requested = self.requested_eligible_variables
        resolved = self.resolved_variables
        unresolved = self.unresolved_variables
        requested_set = set(requested)
        resolved_set = set(resolved)
        unresolved_set = set(unresolved)

        overlap = resolved_set & unresolved_set
        if overlap:
            raise ValueError(
                "resolved_variables and unresolved_variables must be disjoint; "
                f"overlap={sorted(overlap)}"
            )
        if resolved_set | unresolved_set != requested_set:
            raise ValueError(
                "resolved_variables and unresolved_variables must cover exactly "
                "requested_eligible_variables"
            )

        constraint_names = [item.variable for item in self.resolved_constraints]
        if constraint_names != resolved:
            raise ValueError(
                "resolved_constraints variable order must match resolved_variables"
            )

        for issue in self.issues:
            if issue.code == "SAFETY_DECISION_REFUSED":
                continue
            if issue.variable not in requested_set:
                raise ValueError(
                    f"issue variable {issue.variable!r} must be in "
                    "requested_eligible_variables"
                )

        safety_refused = self.safety_status is RecommendationSafetyStatus.REFUSED
        if self.status is ConstraintResolutionStatus.READY:
            if safety_refused:
                raise ValueError("READY requires safety_status that is not REFUSED")
            if not resolved:
                raise ValueError("READY requires at least one resolved variable")
            if unresolved:
                raise ValueError("READY requires zero unresolved variables")
        elif self.status is ConstraintResolutionStatus.PARTIAL:
            if safety_refused:
                raise ValueError("PARTIAL requires safety_status that is not REFUSED")
            if not resolved:
                raise ValueError("PARTIAL requires at least one resolved variable")
            if not unresolved:
                raise ValueError("PARTIAL requires at least one unresolved variable")
        elif self.status is ConstraintResolutionStatus.REFUSED:
            # REFUSED may clear resolved lists; unresolved must still partition
            # requested eligible variables.
            pass
        else:
            raise ValueError(f"unsupported resolution status: {self.status!r}")

        return self


@dataclass(frozen=True, slots=True)
class ConstraintResolutionOutcome:
    """Immutable wrapper around a constraint resolution report.

    Holds only ``ConstraintResolutionReport``. Contains no business logic and
    does not store models, estimators, or DataFrames.
    """

    report: ConstraintResolutionReport


def _constraint_map(
    constraints: Sequence[VariableConstraint],
    *,
    source_label: str,
) -> dict[str, VariableConstraint]:
    mapping: dict[str, VariableConstraint] = {}
    for item in constraints:
        if not isinstance(item, VariableConstraint):
            raise TypeError(
                f"{source_label} entries must be VariableConstraint, "
                f"got {type(item).__name__}"
            )
        name = item.variable
        if name in mapping:
            raise DataValidationError(
                f"{source_label} contains duplicate constraint for variable {name!r}"
            )
        mapping[name] = item
    return mapping


def _require_constraint_sequence(
    value: object,
    *,
    field_name: str,
) -> Sequence[VariableConstraint]:
    if isinstance(value, (str, bytes)):
        raise TypeError(
            f"{field_name} must be a Sequence[VariableConstraint], "
            f"got {type(value).__name__}"
        )
    if not isinstance(value, Sequence):
        raise TypeError(
            f"{field_name} must be a Sequence[VariableConstraint], "
            f"got {type(value).__name__}"
        )
    return value


def _extract_bounds(
    constraint: VariableConstraint | None,
) -> tuple[float | None, float | None]:
    if constraint is None:
        return None, None
    minimum = constraint.minimum
    maximum = constraint.maximum
    if minimum is not None:
        if isinstance(minimum, bool) or not isinstance(minimum, (int, float)):
            return None, None  # handled as unsupported by caller via shape check
        minimum = float(minimum)
        if not math.isfinite(minimum):
            return None, None
    if maximum is not None:
        if isinstance(maximum, bool) or not isinstance(maximum, (int, float)):
            return None, None
        maximum = float(maximum)
        if not math.isfinite(maximum):
            return None, None
    return minimum, maximum


def _bounds_are_finite_supported(
    constraint: VariableConstraint,
) -> bool:
    for bound in (constraint.minimum, constraint.maximum):
        if bound is None:
            continue
        if isinstance(bound, bool) or not isinstance(bound, (int, float)):
            return False
        if not math.isfinite(float(bound)):
            return False
    return True


def _merge_industry_request(
    *,
    industry: VariableConstraint | None,
    request: VariableConstraint | None,
    intersect: bool,
) -> tuple[float | None, float | None, list[ConstraintSource], str | None]:
    """Return (minimum, maximum, sources, conflict_code_or_None)."""
    if industry is not None and not _bounds_are_finite_supported(industry):
        return None, None, [], "UNSUPPORTED_CONSTRAINT_SHAPE"
    if request is not None and not _bounds_are_finite_supported(request):
        return None, None, [], "UNSUPPORTED_CONSTRAINT_SHAPE"

    ind_min, ind_max = _extract_bounds(industry)
    req_min, req_max = _extract_bounds(request)

    sources: list[ConstraintSource] = []
    if industry is not None:
        sources.append(ConstraintSource.INDUSTRY_DEFAULT)
    if request is not None:
        sources.append(ConstraintSource.REQUEST)

    if industry is None and request is None:
        return None, None, [], "CONSTRAINT_MISSING"

    if not intersect:
        if request is not None:
            return req_min, req_max, [ConstraintSource.REQUEST], None
        return ind_min, ind_max, [ConstraintSource.INDUSTRY_DEFAULT], None

    # intersect=True
    if industry is not None and request is not None:
        eff_min: float | None
        eff_max: float | None
        if ind_min is None and req_min is None:
            eff_min = None
        elif ind_min is None:
            eff_min = req_min
        elif req_min is None:
            eff_min = ind_min
        else:
            eff_min = max(ind_min, req_min)

        if ind_max is None and req_max is None:
            eff_max = None
        elif ind_max is None:
            eff_max = req_max
        elif req_max is None:
            eff_max = ind_max
        else:
            eff_max = min(ind_max, req_max)

        if (
            eff_min is not None
            and eff_max is not None
            and eff_min > eff_max
        ):
            return None, None, sources, "CONFLICTING_BOUNDS"
        return eff_min, eff_max, sources, None

    if request is not None:
        return req_min, req_max, [ConstraintSource.REQUEST], None
    return ind_min, ind_max, [ConstraintSource.INDUSTRY_DEFAULT], None


def _apply_user_override(
    *,
    effective_min: float | None,
    effective_max: float | None,
    base_sources: list[ConstraintSource],
    user: VariableConstraint | None,
    allow_widening: bool,
) -> tuple[
    float | None,
    float | None,
    list[ConstraintSource],
    bool,
    bool,
    list[str],
    str | None,
    str | None,
]:
    """Apply user override.

    Returns:
        minimum, maximum, sources, widened, widened_attempt,
        warnings, blocking_code, non_blocking_code
    """
    if user is None:
        return (
            effective_min,
            effective_max,
            list(base_sources),
            False,
            False,
            [],
            None,
            None,
        )

    if not _bounds_are_finite_supported(user):
        return (
            None,
            None,
            list(base_sources),
            False,
            False,
            [],
            "UNSUPPORTED_CONSTRAINT_SHAPE",
            None,
        )

    user_min, user_max = _extract_bounds(user)
    sources = list(base_sources)
    if ConstraintSource.USER_OVERRIDE not in sources:
        sources.append(ConstraintSource.USER_OVERRIDE)
    # Keep canonical order.
    sources = [s for s in _SOURCE_CANONICAL_ORDER if s in sources]

    warnings: list[str] = []
    widened = False
    widened_attempt = False

    if effective_min is None and effective_max is None:
        # No base bounds; user alone must provide both finite ends eventually.
        return (
            user_min,
            user_max,
            sources,
            False,
            False,
            warnings,
            None,
            None,
        )

    if allow_widening:
        new_min = effective_min
        new_max = effective_max
        if user_min is not None:
            if effective_min is not None and user_min < effective_min:
                widened = True
                widened_attempt = True
            new_min = user_min
        if user_max is not None:
            if effective_max is not None and user_max > effective_max:
                widened = True
                widened_attempt = True
            new_max = user_max
        if widened:
            warnings.append(
                "User override widened effective bounds; treat with caution."
            )
        if (
            new_min is not None
            and new_max is not None
            and new_min > new_max
        ):
            return (
                None,
                None,
                sources,
                widened,
                widened_attempt,
                warnings,
                "CONFLICTING_BOUNDS",
                None,
            )
        return (
            new_min,
            new_max,
            sources,
            widened,
            widened_attempt,
            warnings,
            None,
            "USER_OVERRIDE_WIDENS_BOUNDS" if widened else None,
        )

    # allow_widening=False: intersect only; record widening attempt as warning.
    if effective_min is not None and user_min is not None and user_min < effective_min:
        widened_attempt = True
    if effective_max is not None and user_max is not None and user_max > effective_max:
        widened_attempt = True

    new_min = effective_min
    new_max = effective_max
    if user_min is not None:
        if new_min is None:
            new_min = user_min
        else:
            new_min = max(new_min, user_min)
    if user_max is not None:
        if new_max is None:
            new_max = user_max
        else:
            new_max = min(new_max, user_max)

    non_blocking: str | None = None
    if widened_attempt:
        warnings.append(
            "User override attempted to widen bounds; only the safe intersection "
            "was retained."
        )
        non_blocking = "USER_OVERRIDE_WIDENS_BOUNDS"

    if (
        new_min is not None
        and new_max is not None
        and new_min > new_max
    ):
        return (
            None,
            None,
            sources,
            False,
            widened_attempt,
            warnings,
            "CONFLICTING_BOUNDS",
            non_blocking,
        )

    # Complete non-overlap can also appear when one side is missing after intersect.
    if widened_attempt and (
        (effective_min is not None and user_max is not None and user_max < effective_min)
        or (
            effective_max is not None
            and user_min is not None
            and user_min > effective_max
        )
    ):
        return (
            None,
            None,
            sources,
            False,
            widened_attempt,
            warnings,
            "CONFLICTING_BOUNDS",
            non_blocking,
        )

    return (
        new_min,
        new_max,
        sources,
        False,
        widened_attempt,
        warnings,
        None,
        non_blocking,
    )


class ConstraintResolver:
    """Merge multi-source constraints into effective bounded candidates.

    Holds an isolated deep copy of ``ConstraintResolutionPolicy``. Does not
    cache results, mutate caller-owned inputs, or generate proposed values.
    """

    def __init__(
        self,
        *,
        policy: ConstraintResolutionPolicy | None = None,
    ) -> None:
        """Create a resolver with an isolated policy copy.

        Args:
            policy: Optional resolution policy. When ``None``, defaults are used.
                The provided policy is deep-copied so later mutations do not
                affect this resolver.

        Raises:
            TypeError: If ``policy`` is not ``None`` or a
                ``ConstraintResolutionPolicy``.
        """
        if policy is None:
            self._policy = ConstraintResolutionPolicy()
        elif not isinstance(policy, ConstraintResolutionPolicy):
            raise TypeError(
                "policy must be ConstraintResolutionPolicy, "
                f"got {type(policy).__name__}"
            )
        else:
            self._policy = policy.model_copy(deep=True)

    def resolve(
        self,
        request: RecommendationRequest,
        *,
        safety_decision: RecommendationSafetyDecision,
        industry_constraints: Sequence[VariableConstraint] = (),
        user_overrides: Sequence[VariableConstraint] = (),
    ) -> ConstraintResolutionOutcome:
        """Resolve effective numeric bounds for safety-eligible variables.

        Expected safety refusals return a structured ``REFUSED`` report rather
        than raising. Invalid types raise ``TypeError``. Inputs are not mutated.

        Args:
            request: Validated recommendation request.
            safety_decision: Prior safety-gate decision for the same request.
            industry_constraints: Optional industry-default constraints.
            user_overrides: Optional explicit operator overrides.

        Returns:
            A ``ConstraintResolutionOutcome`` with the resolution report.
        """
        if not isinstance(request, RecommendationRequest):
            raise TypeError(
                f"request must be RecommendationRequest, got {type(request).__name__}"
            )
        if not isinstance(safety_decision, RecommendationSafetyDecision):
            raise TypeError(
                "safety_decision must be RecommendationSafetyDecision, "
                f"got {type(safety_decision).__name__}"
            )
        industry_seq = _require_constraint_sequence(
            industry_constraints,
            field_name="industry_constraints",
        )
        user_seq = _require_constraint_sequence(
            user_overrides,
            field_name="user_overrides",
        )

        self._validate_safety_consistency(request, safety_decision)

        policy = self._policy
        eligible = list(safety_decision.eligible_variables)
        # Snapshot current values without mutating request.
        current_values = dict(request.current_values)

        if (
            policy.require_non_refused_safety_decision
            and safety_decision.status is RecommendationSafetyStatus.REFUSED
        ):
            return ConstraintResolutionOutcome(
                report=self._build_safety_refused_report(
                    safety_decision=safety_decision,
                    eligible=eligible,
                    industry_count=len(industry_seq),
                    request_count=len(request.constraints),
                    user_count=len(user_seq),
                )
            )

        industry_map = _constraint_map(
            industry_seq,
            source_label="industry_constraints",
        )
        request_map = {
            item.variable: item for item in request.constraints
        }
        user_map = _constraint_map(
            user_seq,
            source_label="user_overrides",
        )

        resolved_constraints: list[ResolvedVariableConstraint] = []
        resolved_variables: list[str] = []
        unresolved_variables: list[str] = []
        issues: list[ConstraintResolutionIssue] = []
        warnings: list[str] = []
        widening_count = 0
        industry_missing_count = 0
        request_missing_count = 0

        if safety_decision.status is RecommendationSafetyStatus.CAUTION:
            _append_unique(
                warnings,
                "Safety decision status is CAUTION; treat resolved candidates "
                "conservatively.",
            )

        for variable in eligible:
            industry = industry_map.get(variable)
            request_constraint = request_map.get(variable)
            user = user_map.get(variable)

            if industry is None:
                industry_missing_count += 1
            if request_constraint is None:
                request_missing_count += 1

            current_value = current_values.get(variable)
            if current_value is None:
                unresolved_variables.append(variable)
                issues.append(
                    ConstraintResolutionIssue(
                        variable=variable,
                        source=None,
                        code="CURRENT_VALUE_MISSING",
                        message=(
                            f"Current value for variable {variable!r} is missing "
                            "from request.current_values."
                        ),
                        blocking=True,
                    )
                )
                continue

            eff_min, eff_max, sources, merge_code = _merge_industry_request(
                industry=industry,
                request=request_constraint,
                intersect=policy.intersect_industry_and_request_constraints,
            )
            if merge_code is not None:
                unresolved_variables.append(variable)
                issues.append(
                    ConstraintResolutionIssue(
                        variable=variable,
                        source=None,
                        code=merge_code,
                        message=self._issue_message(variable, merge_code),
                        blocking=True,
                    )
                )
                continue

            (
                eff_min,
                eff_max,
                sources,
                widened,
                _widened_attempt,
                override_warnings,
                blocking_code,
                non_blocking_code,
            ) = _apply_user_override(
                effective_min=eff_min,
                effective_max=eff_max,
                base_sources=sources,
                user=user,
                allow_widening=policy.allow_user_bound_widening,
            )

            variable_warnings = list(override_warnings)
            if non_blocking_code == "USER_OVERRIDE_WIDENS_BOUNDS":
                issues.append(
                    ConstraintResolutionIssue(
                        variable=variable,
                        source=ConstraintSource.USER_OVERRIDE,
                        code="USER_OVERRIDE_WIDENS_BOUNDS",
                        message=(
                            f"User override for {variable!r} widens or attempted "
                            "to widen effective bounds."
                        ),
                        blocking=False,
                    )
                )
                if widened or _widened_attempt:
                    widening_count += 1
                    _append_unique(
                        warnings,
                        "One or more user overrides widened or attempted to "
                        "widen effective bounds.",
                    )

            if blocking_code is not None:
                unresolved_variables.append(variable)
                issues.append(
                    ConstraintResolutionIssue(
                        variable=variable,
                        source=ConstraintSource.USER_OVERRIDE
                        if user is not None
                        else None,
                        code=blocking_code,
                        message=self._issue_message(variable, blocking_code),
                        blocking=True,
                    )
                )
                continue

            if eff_min is None or eff_max is None:
                unresolved_variables.append(variable)
                if eff_min is None and eff_max is None:
                    code = "CONSTRAINT_MISSING"
                else:
                    code = "UNSUPPORTED_CONSTRAINT_SHAPE"
                issues.append(
                    ConstraintResolutionIssue(
                        variable=variable,
                        source=None,
                        code=code,
                        message=self._issue_message(variable, code),
                        blocking=True,
                    )
                )
                continue

            if eff_min > eff_max:
                unresolved_variables.append(variable)
                issues.append(
                    ConstraintResolutionIssue(
                        variable=variable,
                        source=None,
                        code="CONFLICTING_BOUNDS",
                        message=self._issue_message(variable, "CONFLICTING_BOUNDS"),
                        blocking=True,
                    )
                )
                continue

            # Current value vs bounds — never clamp.
            outside = current_value < eff_min or current_value > eff_max
            if outside:
                unresolved_variables.append(variable)
                issues.append(
                    ConstraintResolutionIssue(
                        variable=variable,
                        source=None,
                        code="CURRENT_VALUE_OUTSIDE_BOUNDS",
                        message=(
                            f"Current value for {variable!r} is outside effective "
                            f"bounds [{eff_min}, {eff_max}]."
                        ),
                        blocking=True,
                    )
                )
                continue

            span = float(eff_max - eff_min)
            if span < policy.minimum_effective_span or span <= 0.0:
                unresolved_variables.append(variable)
                issues.append(
                    ConstraintResolutionIssue(
                        variable=variable,
                        source=None,
                        code="EFFECTIVE_SPAN_TOO_SMALL",
                        message=(
                            f"Effective span for {variable!r} is too small "
                            f"({span}) for optimization candidates."
                        ),
                        blocking=True,
                    )
                )
                continue

            lower_room = float(current_value - eff_min)
            upper_room = float(eff_max - current_value)
            relative_position = float(lower_room / span)

            resolved = ResolvedVariableConstraint(
                variable=variable,
                current_value=float(current_value),
                minimum=float(eff_min),
                maximum=float(eff_max),
                effective_span=span,
                lower_room=lower_room,
                upper_room=upper_room,
                relative_position=relative_position,
                source_chain=list(sources),
                industry_constraint_present=(
                    ConstraintSource.INDUSTRY_DEFAULT in sources
                ),
                request_constraint_present=ConstraintSource.REQUEST in sources,
                user_override_present=ConstraintSource.USER_OVERRIDE in sources,
                user_override_widened_bounds=widened,
                warnings=variable_warnings,
            )
            resolved_constraints.append(resolved)
            resolved_variables.append(variable)

        # Report-level missing constraint warnings.
        if industry_missing_count > 0:
            _append_unique(
                warnings,
                "One or more eligible variables lack industry-default constraints.",
            )
        if request_missing_count > 0:
            _append_unique(
                warnings,
                "One or more eligible variables lack request constraints.",
            )

        status = self._determine_status(
            resolved_count=len(resolved_variables),
            unresolved_count=len(unresolved_variables),
            safety_status=safety_decision.status,
            allow_partial=policy.allow_partial_resolution,
        )

        if status is ConstraintResolutionStatus.PARTIAL:
            _append_unique(
                warnings,
                "Constraint resolution is partial; some eligible variables remain "
                "unresolved.",
            )
        if status is ConstraintResolutionStatus.REFUSED and not resolved_variables:
            _append_unique(
                warnings,
                "No variables were resolved to effective bounded constraints.",
            )

        # When partial disabled, clear resolved lists for REFUSED outcome.
        final_resolved = list(resolved_variables)
        final_constraints = list(resolved_constraints)
        final_unresolved = list(unresolved_variables)
        if (
            status is ConstraintResolutionStatus.REFUSED
            and unresolved_variables
            and resolved_variables
            and not policy.allow_partial_resolution
        ):
            final_unresolved = list(eligible)
            final_resolved = []
            final_constraints = []

        issues = self._sort_issues(issues, eligible_order=eligible)

        metadata: dict[str, ScalarMetadataValue] = {
            "requested_eligible_count": len(eligible),
            "resolved_count": len(final_resolved),
            "unresolved_count": len(final_unresolved),
            "industry_constraint_count": len(industry_map),
            "request_constraint_count": len(request_map),
            "user_override_count": len(user_map),
            "user_override_widening_count": widening_count,
            "partial_resolution": status is ConstraintResolutionStatus.PARTIAL,
            "optimization_performed": False,
            "candidate_values_generated": False,
            "bounds_are_continuous": True,
        }

        report = ConstraintResolutionReport(
            status=status,
            safety_status=safety_decision.status,
            requested_eligible_variables=list(eligible),
            resolved_variables=final_resolved,
            unresolved_variables=final_unresolved,
            resolved_constraints=final_constraints,
            issues=issues,
            evaluated_at=datetime.now(tz=UTC),
            warnings=warnings,
            metadata=metadata,
        )
        return ConstraintResolutionOutcome(report=report)

    def get_metadata(self) -> dict[str, ScalarMetadataValue]:
        """Return scalar-only resolver metadata for the current policy.

        Returns an independent dict on every call. Does not include estimators,
        models, DataFrames, or ndarray values.
        """
        policy = self._policy
        return {
            "require_non_refused_safety_decision": (
                policy.require_non_refused_safety_decision
            ),
            "intersect_industry_and_request_constraints": (
                policy.intersect_industry_and_request_constraints
            ),
            "allow_user_bound_widening": policy.allow_user_bound_widening,
            "require_current_value_within_bounds": (
                policy.require_current_value_within_bounds
            ),
            "minimum_effective_span": policy.minimum_effective_span,
            "allow_partial_resolution": policy.allow_partial_resolution,
            "preserve_safety_ranking": policy.preserve_safety_ranking,
            "generates_candidate_values": False,
            "performs_optimization": False,
        }

    def _validate_safety_consistency(
        self,
        request: RecommendationRequest,
        safety_decision: RecommendationSafetyDecision,
    ) -> None:
        if safety_decision.objective != request.objective:
            raise DataValidationError(
                "safety_decision.objective must equal request.objective "
                f"(got safety={safety_decision.objective!r}, "
                f"request={request.objective!r})"
            )

        factor_variables = [factor.variable for factor in request.diagnosis.factors]
        factor_set = set(factor_variables)
        assessment_variables = {
            item.variable for item in safety_decision.variable_assessments
        }
        if assessment_variables != factor_set:
            raise DataValidationError(
                "safety_decision.variable_assessments must match diagnosis factors "
                f"(assessments={sorted(assessment_variables)}, "
                f"factors={sorted(factor_set)})"
            )

        for name in safety_decision.eligible_variables:
            if name not in request.current_values:
                raise DataValidationError(
                    f"safety_decision.eligible_variables entry {name!r} is absent "
                    "from request.current_values"
                )
            if name not in factor_set:
                raise DataValidationError(
                    f"safety_decision.eligible_variables entry {name!r} is absent "
                    "from diagnosis factors"
                )

    def _build_safety_refused_report(
        self,
        *,
        safety_decision: RecommendationSafetyDecision,
        eligible: list[str],
        industry_count: int,
        request_count: int,
        user_count: int,
    ) -> ConstraintResolutionReport:
        issues = [
            ConstraintResolutionIssue(
                variable=_SAFETY_REFUSED_ISSUE_VARIABLE,
                source=None,
                code="SAFETY_DECISION_REFUSED",
                message=(
                    "Constraint resolution refused because the safety decision "
                    "status is REFUSED."
                ),
                blocking=True,
            )
        ]
        warnings = [
            "No variables were resolved to effective bounded constraints.",
        ]
        return ConstraintResolutionReport(
            status=ConstraintResolutionStatus.REFUSED,
            safety_status=safety_decision.status,
            requested_eligible_variables=list(eligible),
            resolved_variables=[],
            unresolved_variables=list(eligible),
            resolved_constraints=[],
            issues=issues,
            evaluated_at=datetime.now(tz=UTC),
            warnings=warnings,
            metadata={
                "requested_eligible_count": len(eligible),
                "resolved_count": 0,
                "unresolved_count": len(eligible),
                "industry_constraint_count": industry_count,
                "request_constraint_count": request_count,
                "user_override_count": user_count,
                "user_override_widening_count": 0,
                "partial_resolution": False,
                "optimization_performed": False,
                "candidate_values_generated": False,
                "bounds_are_continuous": True,
            },
        )

    @staticmethod
    def _issue_message(variable: str, code: str) -> str:
        messages = {
            "CONSTRAINT_MISSING": (
                f"No usable numeric constraint bounds found for {variable!r}."
            ),
            "CONFLICTING_BOUNDS": (
                f"Merged constraints for {variable!r} produce conflicting bounds."
            ),
            "UNSUPPORTED_CONSTRAINT_SHAPE": (
                f"Constraint shape for {variable!r} is unsupported for bounded "
                "optimization candidates."
            ),
            "EFFECTIVE_SPAN_TOO_SMALL": (
                f"Effective span for {variable!r} is too small."
            ),
            "CURRENT_VALUE_OUTSIDE_BOUNDS": (
                f"Current value for {variable!r} is outside effective bounds."
            ),
            "CURRENT_VALUE_MISSING": (
                f"Current value for {variable!r} is missing."
            ),
        }
        return messages.get(code, f"Constraint resolution issue {code} for {variable!r}.")

    @staticmethod
    def _determine_status(
        *,
        resolved_count: int,
        unresolved_count: int,
        safety_status: RecommendationSafetyStatus,
        allow_partial: bool,
    ) -> ConstraintResolutionStatus:
        if safety_status is RecommendationSafetyStatus.REFUSED:
            return ConstraintResolutionStatus.REFUSED
        if resolved_count == 0:
            return ConstraintResolutionStatus.REFUSED
        if unresolved_count == 0:
            return ConstraintResolutionStatus.READY
        if allow_partial:
            return ConstraintResolutionStatus.PARTIAL
        return ConstraintResolutionStatus.REFUSED

    @staticmethod
    def _sort_issues(
        issues: list[ConstraintResolutionIssue],
        *,
        eligible_order: list[str],
    ) -> list[ConstraintResolutionIssue]:
        order_index = {name: idx for idx, name in enumerate(eligible_order)}

        def sort_key(issue: ConstraintResolutionIssue) -> tuple[int, int, int]:
            code_rank = _ISSUE_CODE_SORT_ORDER.get(issue.code, 99)
            if issue.code == "SAFETY_DECISION_REFUSED":
                var_rank = -1
            else:
                var_rank = order_index.get(issue.variable, 10_000)
            blocking_rank = 0 if issue.blocking else 1
            return (code_rank, var_rank, blocking_rank)

        return sorted(issues, key=sort_key)
