"""Candidate variable selection for bounded recommendation search (Step 9B).

Converts safety-eligible, constraint-resolved diagnosis factors into ordered
optimization candidates. Does not generate proposed values, grids, or run
optimization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator

from process_intelligence.core.exceptions import DataValidationError
from process_intelligence.core.schemas import RootCauseFactor
from process_intelligence.recommendation.constraint_resolution import (
    ConstraintResolutionOutcome,
    ConstraintResolutionStatus,
    ConstraintSource,
    ResolvedVariableConstraint,
)
from process_intelligence.recommendation.enums import (
    RecommendationObjective,
    RecommendationSafetyStatus,
)
from process_intelligence.recommendation.schemas import (
    RecommendationRequest,
    RecommendationSafetyDecision,
    ScalarMetadataValue,
)

_ORIGINAL_ROW_ID = "_original_row_id"

_SOURCE_CANONICAL_ORDER = (
    ConstraintSource.INDUSTRY_DEFAULT,
    ConstraintSource.REQUEST,
    ConstraintSource.USER_OVERRIDE,
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


def _require_confidence(value: object, *, field_name: str = "confidence") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a finite float in [0.0, 1.0] "
            f"(bool not allowed), got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{field_name} must be in [0.0, 1.0], got {number}")
    return number


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


def _require_factor_direction(value: object) -> str:
    """Accept any non-empty direction string allowed by ``RootCauseFactor``."""
    return _require_non_empty_str(value, field_name="factor_direction")


class CandidateVariable(BaseModel):
    """Bounded optimization candidate derived from diagnosis and constraints.

    Preserves diagnosis ranking evidence and effective numeric bounds. Does not
    store proposed values or optimizer outputs.
    """

    variable: str
    diagnosis_rank: int
    factor_confidence: float
    factor_direction: str
    factor_controllable: bool
    factor_needs_verification: bool
    current_value: float
    minimum: float
    maximum: float
    lower_room: float
    upper_room: float
    relative_position: float
    at_lower_bound: bool
    at_upper_bound: bool
    source_chain: list[ConstraintSource]
    user_override_applied: bool
    user_override_widened_bounds: bool
    warnings: list[str] = Field(default_factory=list)

    @field_validator("variable", mode="before")
    @classmethod
    def _validate_variable(cls, value: object) -> str:
        return _validate_variable_name(value, field_name="variable")

    @field_validator("diagnosis_rank", mode="before")
    @classmethod
    def _validate_diagnosis_rank(cls, value: object) -> int:
        return _require_strict_int_ge(value, field_name="diagnosis_rank", minimum=1)

    @field_validator("factor_confidence", mode="before")
    @classmethod
    def _validate_factor_confidence(cls, value: object) -> float:
        return _require_confidence(value, field_name="factor_confidence")

    @field_validator("factor_direction", mode="before")
    @classmethod
    def _validate_factor_direction(cls, value: object) -> str:
        return _require_factor_direction(value)

    @field_validator(
        "current_value",
        "minimum",
        "maximum",
        "lower_room",
        "upper_room",
        "relative_position",
        mode="before",
    )
    @classmethod
    def _validate_numeric_fields(cls, value: object) -> float:
        return _require_finite_float(value, field_name="numeric candidate field")

    @field_validator(
        "factor_controllable",
        "factor_needs_verification",
        "at_lower_bound",
        "at_upper_bound",
        "user_override_applied",
        "user_override_widened_bounds",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="candidate bool field")

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
    def _validate_candidate_consistency(self) -> Self:
        if self.minimum > self.current_value or self.current_value > self.maximum:
            raise ValueError(
                "current_value must satisfy minimum <= current_value <= maximum"
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
        span = self.maximum - self.minimum
        if span <= 0.0:
            raise ValueError("maximum - minimum must be > 0")
        expected_relative = self.lower_room / span
        if self.relative_position < 0.0 or self.relative_position > 1.0:
            raise ValueError(
                f"relative_position must be in [0.0, 1.0], got {self.relative_position}"
            )
        if not math.isclose(
            self.relative_position,
            expected_relative,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "relative_position must equal lower_room / (maximum - minimum)"
            )

        expected_lower_flag = math.isclose(
            self.current_value,
            self.minimum,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        expected_upper_flag = math.isclose(
            self.current_value,
            self.maximum,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        if self.at_lower_bound != expected_lower_flag:
            raise ValueError(
                "at_lower_bound must equal math.isclose(current_value, minimum)"
            )
        if self.at_upper_bound != expected_upper_flag:
            raise ValueError(
                "at_upper_bound must equal math.isclose(current_value, maximum)"
            )

        override_present = ConstraintSource.USER_OVERRIDE in self.source_chain
        if self.user_override_applied != override_present:
            raise ValueError(
                "user_override_applied must match USER_OVERRIDE presence in "
                "source_chain"
            )
        if self.user_override_widened_bounds and not self.user_override_applied:
            raise ValueError(
                "user_override_widened_bounds=True requires user_override_applied=True"
            )
        return self


class CandidateVariableSet(BaseModel):
    """Ordered set of bounded candidates ready for later optimization.

    Ranking follows diagnosis factor order. Does not generate proposed values
    or candidate combinations.
    """

    objective: RecommendationObjective
    safety_status: RecommendationSafetyStatus
    resolution_status: ConstraintResolutionStatus
    candidates: list[CandidateVariable]
    candidate_variables: list[str]
    max_simultaneous_changes: int
    effective_change_budget: int
    generated_at: datetime
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)

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

    @field_validator("candidates", mode="before")
    @classmethod
    def _validate_candidates_before(cls, value: object) -> list[CandidateVariable]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"candidates must be a list[CandidateVariable], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("candidates", mode="after")
    @classmethod
    def _validate_candidates(
        cls,
        value: list[CandidateVariable],
    ) -> list[CandidateVariable]:
        seen_vars: set[str] = set()
        seen_ranks: set[int] = set()
        previous_rank = 0
        copied: list[CandidateVariable] = []
        for item in value:
            if not isinstance(item, CandidateVariable):
                raise ValueError(
                    "candidates entries must be CandidateVariable, "
                    f"got {type(item).__name__}"
                )
            if item.variable in seen_vars:
                raise ValueError(
                    f"candidates must not contain duplicate variables: "
                    f"{item.variable!r}"
                )
            if item.diagnosis_rank in seen_ranks:
                raise ValueError(
                    f"candidates must not contain duplicate diagnosis_rank: "
                    f"{item.diagnosis_rank}"
                )
            if item.diagnosis_rank < previous_rank:
                raise ValueError(
                    "candidates must be ordered by ascending diagnosis_rank"
                )
            seen_vars.add(item.variable)
            seen_ranks.add(item.diagnosis_rank)
            previous_rank = item.diagnosis_rank
            copied.append(item.model_copy(deep=True))
        return copied

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

    @field_validator("max_simultaneous_changes", mode="before")
    @classmethod
    def _validate_max_simultaneous_changes(cls, value: object) -> int:
        return _require_strict_int_ge(
            value,
            field_name="max_simultaneous_changes",
            minimum=1,
        )

    @field_validator("effective_change_budget", mode="before")
    @classmethod
    def _validate_effective_change_budget(cls, value: object) -> int:
        return _require_strict_int_ge(
            value,
            field_name="effective_change_budget",
            minimum=0,
        )

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
    def _validate_set_consistency(self) -> Self:
        names = [item.variable for item in self.candidates]
        if names != self.candidate_variables:
            raise ValueError(
                "candidate_variables must match candidates variable order exactly"
            )

        if self.effective_change_budget > len(self.candidates):
            raise ValueError(
                "effective_change_budget must be <= len(candidates)"
            )

        if self.candidates:
            expected_budget = min(
                self.max_simultaneous_changes,
                len(self.candidates),
            )
            if self.effective_change_budget != expected_budget:
                raise ValueError(
                    "effective_change_budget must equal "
                    "min(max_simultaneous_changes, len(candidates)) when candidates "
                    f"exist (got {self.effective_change_budget}, expected "
                    f"{expected_budget})"
                )
        elif self.effective_change_budget != 0:
            raise ValueError(
                "effective_change_budget must be 0 when candidates is empty"
            )

        if self.resolution_status is ConstraintResolutionStatus.REFUSED:
            if self.candidates:
                raise ValueError(
                    "resolution_status REFUSED requires empty candidates"
                )
        if self.safety_status is RecommendationSafetyStatus.REFUSED:
            if self.candidates:
                raise ValueError("safety_status REFUSED requires empty candidates")

        return self


@dataclass(frozen=True, slots=True)
class CandidateSelectionOutcome:
    """Immutable wrapper around a candidate variable set.

    Holds only ``CandidateVariableSet``. Contains no business logic and does
    not store models, estimators, or DataFrames.
    """

    candidate_set: CandidateVariableSet


class CandidateVariableSelector:
    """Select bounded optimization candidates from safety and resolution outputs.

    Stateless selector. Does not cache results, mutate inputs, generate
    proposed values, or run optimization.
    """

    def select(
        self,
        request: RecommendationRequest,
        *,
        safety_decision: RecommendationSafetyDecision,
        resolution: ConstraintResolutionOutcome,
    ) -> CandidateSelectionOutcome:
        """Build ordered candidates from eligible resolved variables.

        Args:
            request: Validated recommendation request.
            safety_decision: Prior safety-gate decision for the same request.
            resolution: Constraint resolution outcome for the same request.

        Returns:
            A ``CandidateSelectionOutcome`` with the candidate set.
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
        if not isinstance(resolution, ConstraintResolutionOutcome):
            raise TypeError(
                "resolution must be ConstraintResolutionOutcome, "
                f"got {type(resolution).__name__}"
            )

        report = resolution.report
        self._validate_consistency(
            request=request,
            safety_decision=safety_decision,
            resolution=resolution,
        )

        warnings: list[str] = []

        if (
            safety_decision.status is RecommendationSafetyStatus.REFUSED
            or report.status is ConstraintResolutionStatus.REFUSED
        ):
            candidate_set = CandidateVariableSet(
                objective=request.objective,
                safety_status=safety_decision.status,
                resolution_status=report.status,
                candidates=[],
                candidate_variables=[],
                max_simultaneous_changes=request.max_simultaneous_changes,
                effective_change_budget=0,
                generated_at=datetime.now(tz=UTC),
                warnings=self._build_empty_warnings(
                    safety_decision=safety_decision,
                    resolution_status=report.status,
                    base_warnings=warnings,
                ),
                metadata=self._build_metadata(
                    request=request,
                    safety_decision=safety_decision,
                    resolution=resolution,
                    candidates=[],
                ),
            )
            return CandidateSelectionOutcome(candidate_set=candidate_set)

        if safety_decision.status is RecommendationSafetyStatus.CAUTION:
            _append_unique(
                warnings,
                "Safety decision status is CAUTION; treat candidates conservatively.",
            )
        if report.status is ConstraintResolutionStatus.PARTIAL:
            _append_unique(
                warnings,
                "Constraint resolution is PARTIAL; only resolved variables are "
                "candidates.",
            )

        factor_map, factor_order = self._build_factor_map(request)
        resolved_map = {
            item.variable: item for item in report.resolved_constraints
        }
        eligible_set = set(safety_decision.eligible_variables)
        resolved_set = set(report.resolved_variables)

        # Detect safety vs diagnosis order mismatch among shared eligible+resolved.
        safety_order = [
            name
            for name in safety_decision.eligible_variables
            if name in resolved_set
        ]
        diagnosis_order = [
            name
            for name in factor_order
            if name in eligible_set and name in resolved_set
        ]
        if safety_order != diagnosis_order and safety_order and diagnosis_order:
            _append_unique(
                warnings,
                "Safety eligible order differs from diagnosis factor ranking; "
                "diagnosis factor order is used for candidates.",
            )

        candidates: list[CandidateVariable] = []
        override_count = 0
        widened_count = 0
        lower_bound_count = 0
        upper_bound_count = 0
        verification_count = 0

        for rank, variable in enumerate(factor_order, start=1):
            if variable not in eligible_set:
                continue
            if variable not in resolved_set:
                continue
            factor = factor_map[variable]
            constraint = resolved_map[variable]
            candidate = self._build_candidate(
                factor=factor,
                diagnosis_rank=rank,
                constraint=constraint,
                request=request,
            )
            candidates.append(candidate)

            if candidate.user_override_applied:
                override_count += 1
                _append_unique(
                    warnings,
                    "One or more candidates include user override constraints.",
                )
            if candidate.user_override_widened_bounds:
                widened_count += 1
                _append_unique(
                    warnings,
                    "One or more candidates used widened user override bounds.",
                )
            if candidate.at_lower_bound or candidate.at_upper_bound:
                if candidate.at_lower_bound:
                    lower_bound_count += 1
                if candidate.at_upper_bound:
                    upper_bound_count += 1
                _append_unique(
                    warnings,
                    "One or more candidates sit at a bound boundary.",
                )
            if candidate.factor_needs_verification:
                verification_count += 1
                _append_unique(
                    warnings,
                    "One or more candidates still need process verification.",
                )

        if not candidates:
            _append_unique(warnings, "No optimization candidates were selected.")

        budget = (
            0
            if not candidates
            else min(request.max_simultaneous_changes, len(candidates))
        )

        candidate_set = CandidateVariableSet(
            objective=request.objective,
            safety_status=safety_decision.status,
            resolution_status=report.status,
            candidates=candidates,
            candidate_variables=[item.variable for item in candidates],
            max_simultaneous_changes=request.max_simultaneous_changes,
            effective_change_budget=budget,
            generated_at=datetime.now(tz=UTC),
            warnings=warnings,
            metadata=self._build_metadata(
                request=request,
                safety_decision=safety_decision,
                resolution=resolution,
                candidates=candidates,
                override_count=override_count,
                widened_count=widened_count,
                lower_bound_count=lower_bound_count,
                upper_bound_count=upper_bound_count,
            ),
        )
        return CandidateSelectionOutcome(candidate_set=candidate_set)

    def get_metadata(self) -> dict[str, ScalarMetadataValue]:
        """Return scalar-only selector metadata.

        Returns an independent dict on every call.
        """
        return {
            "ranking_source": "diagnosis_factor_order",
            "requires_safety_eligibility": True,
            "requires_resolved_constraints": True,
            "generates_candidate_values": False,
            "performs_optimization": False,
        }

    def _validate_consistency(
        self,
        *,
        request: RecommendationRequest,
        safety_decision: RecommendationSafetyDecision,
        resolution: ConstraintResolutionOutcome,
    ) -> None:
        report = resolution.report
        if request.objective != safety_decision.objective:
            raise DataValidationError(
                "request.objective must equal safety_decision.objective"
            )
        if report.safety_status != safety_decision.status:
            raise DataValidationError(
                "resolution.report.safety_status must equal safety_decision.status"
            )
        if report.requested_eligible_variables != list(
            safety_decision.eligible_variables
        ):
            raise DataValidationError(
                "resolution requested_eligible_variables must match "
                "safety_decision.eligible_variables"
            )

        eligible_set = set(safety_decision.eligible_variables)
        if not set(report.resolved_variables).issubset(eligible_set):
            raise DataValidationError(
                "resolved_variables must be a subset of safety eligible variables"
            )

        factor_variables = {factor.variable for factor in request.diagnosis.factors}
        for name in safety_decision.eligible_variables:
            if name not in factor_variables:
                raise DataValidationError(
                    f"eligible variable {name!r} is absent from diagnosis factors"
                )

        seen_constraints: set[str] = set()
        for constraint in report.resolved_constraints:
            if constraint.variable in seen_constraints:
                raise DataValidationError(
                    "resolved_constraints contain duplicate variable "
                    f"{constraint.variable!r}"
                )
            seen_constraints.add(constraint.variable)
            current = request.current_values.get(constraint.variable)
            if current is None:
                raise DataValidationError(
                    f"resolved constraint variable {constraint.variable!r} missing "
                    "from request.current_values"
                )
            if not math.isclose(
                constraint.current_value,
                float(current),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise DataValidationError(
                    f"resolved constraint current_value for {constraint.variable!r} "
                    "does not match request.current_values"
                )

    def _build_factor_map(
        self,
        request: RecommendationRequest,
    ) -> tuple[dict[str, RootCauseFactor], list[str]]:
        factor_map: dict[str, RootCauseFactor] = {}
        order: list[str] = []
        for factor in request.diagnosis.factors:
            if factor.variable in factor_map:
                raise DataValidationError(
                    f"diagnosis factors contain duplicate variable {factor.variable!r}"
                )
            factor_map[factor.variable] = factor
            order.append(factor.variable)
        return factor_map, order

    def _build_candidate(
        self,
        *,
        factor: RootCauseFactor,
        diagnosis_rank: int,
        constraint: ResolvedVariableConstraint,
        request: RecommendationRequest,
    ) -> CandidateVariable:
        current_value = float(request.current_values[factor.variable])
        warnings = list(constraint.warnings)
        if constraint.user_override_present:
            _append_unique(warnings, "User override constraint applied.")
        if constraint.user_override_widened_bounds:
            _append_unique(
                warnings,
                "User override widened effective bounds; treat with caution.",
            )
        if factor.needs_verification:
            _append_unique(
                warnings,
                "Factor needs verification; candidate retained due to safety "
                "eligibility.",
            )

        at_lower = math.isclose(
            current_value,
            constraint.minimum,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        at_upper = math.isclose(
            current_value,
            constraint.maximum,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        if at_lower or at_upper:
            _append_unique(warnings, "Current value is at a bound boundary.")

        return CandidateVariable(
            variable=factor.variable,
            diagnosis_rank=diagnosis_rank,
            factor_confidence=float(factor.confidence),
            factor_direction=str(factor.direction),
            factor_controllable=bool(factor.controllable),
            factor_needs_verification=bool(factor.needs_verification),
            current_value=current_value,
            minimum=float(constraint.minimum),
            maximum=float(constraint.maximum),
            lower_room=float(constraint.lower_room),
            upper_room=float(constraint.upper_room),
            relative_position=float(constraint.relative_position),
            at_lower_bound=at_lower,
            at_upper_bound=at_upper,
            source_chain=list(constraint.source_chain),
            user_override_applied=constraint.user_override_present,
            user_override_widened_bounds=constraint.user_override_widened_bounds,
            warnings=warnings,
        )

    def _build_empty_warnings(
        self,
        *,
        safety_decision: RecommendationSafetyDecision,
        resolution_status: ConstraintResolutionStatus,
        base_warnings: list[str],
    ) -> list[str]:
        warnings = list(base_warnings)
        if safety_decision.status is RecommendationSafetyStatus.CAUTION:
            _append_unique(
                warnings,
                "Safety decision status is CAUTION; treat candidates conservatively.",
            )
        if resolution_status is ConstraintResolutionStatus.PARTIAL:
            _append_unique(
                warnings,
                "Constraint resolution is PARTIAL; only resolved variables are "
                "candidates.",
            )
        _append_unique(warnings, "No optimization candidates were selected.")
        return warnings

    def _build_metadata(
        self,
        *,
        request: RecommendationRequest,
        safety_decision: RecommendationSafetyDecision,
        resolution: ConstraintResolutionOutcome,
        candidates: list[CandidateVariable],
        override_count: int = 0,
        widened_count: int = 0,
        lower_bound_count: int = 0,
        upper_bound_count: int = 0,
    ) -> dict[str, ScalarMetadataValue]:
        budget = (
            0
            if not candidates
            else min(request.max_simultaneous_changes, len(candidates))
        )
        return {
            "diagnosis_factor_count": len(request.diagnosis.factors),
            "safety_eligible_count": len(safety_decision.eligible_variables),
            "resolved_constraint_count": len(
                resolution.report.resolved_constraints
            ),
            "candidate_count": len(candidates),
            "effective_change_budget": budget,
            "user_override_candidate_count": override_count,
            "widened_override_candidate_count": widened_count,
            "lower_boundary_candidate_count": lower_bound_count,
            "upper_boundary_candidate_count": upper_bound_count,
            "candidate_values_generated": False,
            "optimization_performed": False,
            "ranking_source": "diagnosis_factor_order",
        }
