"""Pydantic contracts for recommendation requests, safety, and results (Step 9A).

Reuses core ``VariableConstraint``, ``RootCauseFactor`` (via diagnosis), and
evaluation ``LeakageReport``. Does not redefine those schemas and does not
compute proposed recommendation values.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.schemas import VariableConstraint
from process_intelligence.diagnosis.schemas import DiagnosisBatchResult, DiagnosisResult
from process_intelligence.evaluation.leakage import LeakageReport
from process_intelligence.recommendation.enums import (
    RecommendationObjective,
    RecommendationReasonCode,
    RecommendationSafetyStatus,
    RecommendationStatus,
)

ScalarMetadataValue = str | int | float | bool | None
"""Allowed scalar types for recommendation metadata dictionaries."""

_ORIGINAL_ROW_ID = "_original_row_id"

_VARIABLE_BLOCKING_REASONS = frozenset(
    {
        RecommendationReasonCode.CURRENT_VALUE_MISSING,
        RecommendationReasonCode.CONSTRAINT_MISSING,
        RecommendationReasonCode.NON_CONTROLLABLE_VARIABLE,
        RecommendationReasonCode.USER_CONFIRMATION_REQUIRED,
        RecommendationReasonCode.VERIFICATION_REQUIRED,
        RecommendationReasonCode.CURRENT_VALUE_OUTSIDE_CONSTRAINT,
        RecommendationReasonCode.CANDIDATE_LIMIT_EXCEEDED,
    }
)

_HARD_GLOBAL_BLOCKERS = frozenset(
    {
        RecommendationReasonCode.LEAKAGE_BLOCKER,
        RecommendationReasonCode.FINAL_EVALUATION_MISSING,
        RecommendationReasonCode.MODEL_PERFORMANCE_UNACCEPTABLE,
        RecommendationReasonCode.DIAGNOSIS_CONFIDENCE_TOO_LOW,
        RecommendationReasonCode.REFERENCE_SAMPLE_TOO_SMALL,
        RecommendationReasonCode.EXTRAPOLATION_RISK,
        RecommendationReasonCode.UNCERTAINTY_UNAVAILABLE,
        RecommendationReasonCode.UNCERTAINTY_UNACCEPTABLE,
        RecommendationReasonCode.NO_ELIGIBLE_VARIABLES,
    }
)

_FORBIDDEN_RATIONALE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"proven\s+root\s+cause", re.IGNORECASE),
    re.compile(r"this\s+will\s+fix", re.IGNORECASE),
    re.compile(r"will\s+fix\s+the\s+process", re.IGNORECASE),
    re.compile(r"guaranteed\s+improvement", re.IGNORECASE),
    re.compile(r"반드시"),
    re.compile(r"무조건"),
    re.compile(r"확실한\s*원인"),
)

_DISCLAIMER_MODEL_BASED = re.compile(r"model[- ]based", re.IGNORECASE)
_DISCLAIMER_VERIFICATION = re.compile(
    r"(process|domain|safety|operational).{0,40}verif",
    re.IGNORECASE,
)
_DISCLAIMER_ASSOCIATION = re.compile(
    r"association.{0,40}(not|does not).{0,20}caus",
    re.IGNORECASE,
)

DEFAULT_RECOMMENDATION_DISCLAIMER = (
    "These recommendations are model-based decision support. "
    "Diagnosis reflects association and not established causation. "
    "Proposed process changes require domain, safety, and operational verification."
)
"""Deterministic English disclaimer required on safety decisions and results."""


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


def _require_optional_finite_float(
    value: object,
    *,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    return _require_finite_float(value, field_name=field_name)


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


def _validate_variable_name(value: object, *, field_name: str) -> str:
    text = _require_non_empty_str(value, field_name=field_name)
    if text == _ORIGINAL_ROW_ID:
        raise ValueError(
            f"{field_name} cannot be reserved column '{_ORIGINAL_ROW_ID}'"
        )
    return text


def _validate_disclaimer_meaning(value: str, *, field_name: str) -> str:
    text = _require_non_empty_str(value, field_name=field_name)
    if _DISCLAIMER_MODEL_BASED.search(text) is None:
        raise ValueError(
            f"{field_name} must state that recommendations are model-based"
        )
    if _DISCLAIMER_VERIFICATION.search(text) is None:
        raise ValueError(
            f"{field_name} must state that process or domain verification is required"
        )
    if _DISCLAIMER_ASSOCIATION.search(text) is None:
        raise ValueError(
            f"{field_name} must state that association does not establish causation"
        )
    return text


def _reject_causal_rationale(value: str, *, field_name: str) -> str:
    text = _require_non_empty_str(value, field_name=field_name)
    for pattern in _FORBIDDEN_RATIONALE_PATTERNS:
        if pattern.search(text) is not None:
            raise ValueError(
                f"{field_name} must not use causal or guaranteed-outcome language"
            )
    return text


class RecommendationSafetyPolicy(BaseModel):
    """Tunable safety thresholds for recommendation eligibility evaluation.

    Controls leakage, model trust, diagnosis confidence, controllability,
    constraint, verification, and extrapolation gates. Does not generate
    proposed process values.
    """

    require_safe_leakage_report: bool = True
    require_final_evaluation: bool = True
    require_acceptable_model_performance: bool = True
    minimum_diagnosis_confidence: float = 0.20
    minimum_reference_rows: int = 5
    require_constraints: bool = True
    require_controllable_process_role: bool = True
    require_user_controllability_confirmation: bool = True
    block_unverified_factors: bool = True
    block_on_extrapolation: bool = True
    require_uncertainty: bool = False
    require_acceptable_uncertainty: bool = False
    maximum_candidate_variables: int = 10
    allow_partial_eligibility: bool = True

    @field_validator(
        "require_safe_leakage_report",
        "require_final_evaluation",
        "require_acceptable_model_performance",
        "require_constraints",
        "require_controllable_process_role",
        "require_user_controllability_confirmation",
        "block_unverified_factors",
        "block_on_extrapolation",
        "require_uncertainty",
        "require_acceptable_uncertainty",
        "allow_partial_eligibility",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="policy bool field")

    @field_validator("minimum_diagnosis_confidence", mode="before")
    @classmethod
    def _validate_minimum_diagnosis_confidence(cls, value: object) -> float:
        return _require_confidence(value, field_name="minimum_diagnosis_confidence")

    @field_validator("minimum_reference_rows", mode="before")
    @classmethod
    def _validate_minimum_reference_rows(cls, value: object) -> int:
        return _require_strict_int_ge(
            value,
            field_name="minimum_reference_rows",
            minimum=1,
        )

    @field_validator("maximum_candidate_variables", mode="before")
    @classmethod
    def _validate_maximum_candidate_variables(cls, value: object) -> int:
        return _require_strict_int_ge(
            value,
            field_name="maximum_candidate_variables",
            minimum=1,
        )

    @model_validator(mode="after")
    def _validate_uncertainty_relationship(self) -> Self:
        if self.require_acceptable_uncertainty and not self.require_uncertainty:
            raise ValueError(
                "require_acceptable_uncertainty=True requires require_uncertainty=True"
            )
        return self


class RecommendationRequest(BaseModel):
    """Request contract for recommendation safety evaluation and later optimization.

    Holds a single ``DiagnosisResult`` (not a batch), current variable values,
    constraints, and user confirmation lists. Does not embed DataFrames,
    estimators, or proposed recommendation values.
    """

    task: AnalysisTask
    diagnosis: DiagnosisResult
    objective: RecommendationObjective
    current_values: dict[str, float]
    constraints: list[VariableConstraint]
    user_confirmed_controllable_variables: list[str]
    user_verified_variables: list[str]
    max_simultaneous_changes: int = 3
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)

    @field_validator("task", mode="before")
    @classmethod
    def _validate_task(cls, value: object) -> AnalysisTask:
        if isinstance(value, AnalysisTask):
            return value
        if isinstance(value, str):
            try:
                return AnalysisTask(value)
            except ValueError as exc:
                raise ValueError(f"invalid AnalysisTask: {value!r}") from exc
        raise ValueError(f"task must be AnalysisTask, got {type(value).__name__}")

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

    @field_validator("diagnosis", mode="before")
    @classmethod
    def _validate_diagnosis_type(cls, value: object) -> DiagnosisResult:
        if isinstance(value, DiagnosisBatchResult):
            raise ValueError(
                "diagnosis must be DiagnosisResult; convert DiagnosisBatchResult "
                "to a per-event or aggregate DiagnosisResult first"
            )
        if isinstance(value, DiagnosisResult):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return DiagnosisResult.model_validate(value)
        raise ValueError(
            f"diagnosis must be DiagnosisResult, got {type(value).__name__}"
        )

    @field_validator("current_values", mode="before")
    @classmethod
    def _validate_current_values_before(cls, value: object) -> dict[str, float]:
        if not isinstance(value, dict):
            raise ValueError(
                f"current_values must be a dict[str, float], got {type(value).__name__}"
            )
        if not value:
            raise ValueError("current_values must contain at least one entry")
        cleaned: dict[str, float] = {}
        for key, raw in value.items():
            name = _validate_variable_name(key, field_name="current_values key")
            number = _require_finite_float(raw, field_name=f"current_values[{name!r}]")
            cleaned[name] = number
        return cleaned

    @field_validator("constraints", mode="before")
    @classmethod
    def _validate_constraints_before(cls, value: object) -> list[VariableConstraint]:
        if not isinstance(value, list):
            raise ValueError(
                f"constraints must be a list[VariableConstraint], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("constraints", mode="after")
    @classmethod
    def _validate_constraints(
        cls,
        value: list[VariableConstraint],
    ) -> list[VariableConstraint]:
        seen: set[str] = set()
        copied: list[VariableConstraint] = []
        for item in value:
            if not isinstance(item, VariableConstraint):
                raise ValueError(
                    "constraints entries must be VariableConstraint, "
                    f"got {type(item).__name__}"
                )
            name = item.variable
            if name in seen:
                raise ValueError(
                    f"constraints must not contain duplicate variables: {name!r}"
                )
            seen.add(name)
            copied.append(item.model_copy(deep=True))
        return copied

    @field_validator(
        "user_confirmed_controllable_variables",
        "user_verified_variables",
        mode="before",
    )
    @classmethod
    def _validate_confirmation_lists_before(cls, value: object) -> list[str]:
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
    def _validate_confirmation_lists(cls, value: list[str]) -> list[str]:
        cleaned = _validate_unique_non_empty_strings(value, field_name="confirmation list")
        for name in cleaned:
            if name == _ORIGINAL_ROW_ID:
                raise ValueError(
                    f"confirmation list cannot include reserved column "
                    f"'{_ORIGINAL_ROW_ID}'"
                )
        return cleaned

    @field_validator("max_simultaneous_changes", mode="before")
    @classmethod
    def _validate_max_simultaneous_changes(cls, value: object) -> int:
        return _require_strict_int_ge(
            value,
            field_name="max_simultaneous_changes",
            minimum=1,
        )

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> dict[str, ScalarMetadataValue]:
        if value is None:
            return {}
        return _validate_scalar_metadata(value)

    @model_validator(mode="after")
    def _validate_request_consistency(self) -> Self:
        if self.diagnosis.task != self.task:
            raise ValueError(
                "diagnosis.task must equal request.task "
                f"(got diagnosis.task={self.diagnosis.task!r}, task={self.task!r})"
            )

        current_keys = set(self.current_values)
        for field_name, names in (
            (
                "user_confirmed_controllable_variables",
                self.user_confirmed_controllable_variables,
            ),
            ("user_verified_variables", self.user_verified_variables),
        ):
            missing = [name for name in names if name not in current_keys]
            if missing:
                raise ValueError(
                    f"{field_name} contains variables absent from current_values: "
                    f"{missing}"
                )

        if self.max_simultaneous_changes > len(self.current_values):
            raise ValueError(
                "max_simultaneous_changes must be <= len(current_values) "
                f"(got {self.max_simultaneous_changes} > {len(self.current_values)})"
            )

        return self


class RecommendationSafetyContext(BaseModel):
    """External safety signals consumed by the recommendation safety gate.

    Carries leakage, final-evaluation, model-performance, extrapolation, and
    uncertainty flags. Does not compute those signals; callers supply them.
    """

    leakage_report: LeakageReport
    final_evaluation_available: bool
    model_performance_acceptable: bool
    model_performance_reason: str | None = None
    extrapolation_detected: bool = False
    uncertainty_available: bool = False
    uncertainty_acceptable: bool | None = None
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)

    @field_validator("leakage_report", mode="before")
    @classmethod
    def _validate_leakage_report(cls, value: object) -> LeakageReport:
        if isinstance(value, LeakageReport):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return LeakageReport.model_validate(value)
        raise ValueError(
            f"leakage_report must be LeakageReport, got {type(value).__name__}"
        )

    @field_validator(
        "final_evaluation_available",
        "model_performance_acceptable",
        "extrapolation_detected",
        "uncertainty_available",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="context bool field")

    @field_validator("model_performance_reason", mode="before")
    @classmethod
    def _validate_model_performance_reason(cls, value: object) -> str | None:
        return _require_optional_non_empty_str(
            value,
            field_name="model_performance_reason",
        )

    @field_validator("uncertainty_acceptable", mode="before")
    @classmethod
    def _validate_uncertainty_acceptable(cls, value: object) -> bool | None:
        if value is None:
            return None
        return _require_strict_bool(value, field_name="uncertainty_acceptable")

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> dict[str, ScalarMetadataValue]:
        if value is None:
            return {}
        return _validate_scalar_metadata(value)

    @model_validator(mode="after")
    def _validate_uncertainty_relationship(self) -> Self:
        if not self.uncertainty_available and self.uncertainty_acceptable is not None:
            raise ValueError(
                "uncertainty_acceptable must be None when uncertainty_available=False"
            )
        return self


class VariableEligibilityAssessment(BaseModel):
    """Per-variable eligibility outcome from recommendation safety evaluation.

    Records factor ranking evidence, constraint and confirmation state, and
    structured reason codes. Blocking reason codes prevent ``eligible=True``.
    """

    variable: str
    factor_rank: int
    factor_confidence: float
    factor_role: str
    factor_controllable: bool
    factor_needs_verification: bool
    current_value: float | None
    constraint_present: bool
    user_confirmed_controllable: bool
    user_verified: bool
    eligible: bool
    reason_codes: list[RecommendationReasonCode]
    warnings: list[str]

    @field_validator("variable", mode="before")
    @classmethod
    def _validate_variable(cls, value: object) -> str:
        return _validate_variable_name(value, field_name="variable")

    @field_validator("factor_rank", mode="before")
    @classmethod
    def _validate_factor_rank(cls, value: object) -> int:
        return _require_strict_int_ge(value, field_name="factor_rank", minimum=1)

    @field_validator("factor_confidence", mode="before")
    @classmethod
    def _validate_factor_confidence(cls, value: object) -> float:
        return _require_confidence(value, field_name="factor_confidence")

    @field_validator("factor_role", mode="before")
    @classmethod
    def _validate_factor_role(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="factor_role")

    @field_validator(
        "factor_controllable",
        "factor_needs_verification",
        "constraint_present",
        "user_confirmed_controllable",
        "user_verified",
        "eligible",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="assessment bool field")

    @field_validator("current_value", mode="before")
    @classmethod
    def _validate_current_value(cls, value: object) -> float | None:
        return _require_optional_finite_float(value, field_name="current_value")

    @field_validator("reason_codes", mode="before")
    @classmethod
    def _validate_reason_codes_before(cls, value: object) -> list[object]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"reason_codes must be a list[RecommendationReasonCode], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("reason_codes", mode="after")
    @classmethod
    def _validate_reason_codes(
        cls,
        value: list[RecommendationReasonCode],
    ) -> list[RecommendationReasonCode]:
        cleaned: list[RecommendationReasonCode] = []
        seen: set[RecommendationReasonCode] = set()
        for item in value:
            if isinstance(item, RecommendationReasonCode):
                code = item
            elif isinstance(item, str):
                try:
                    code = RecommendationReasonCode(item)
                except ValueError as exc:
                    raise ValueError(
                        f"invalid RecommendationReasonCode: {item!r}"
                    ) from exc
            else:
                raise ValueError(
                    "reason_codes entries must be RecommendationReasonCode, "
                    f"got {type(item).__name__}"
                )
            if code in seen:
                raise ValueError(f"reason_codes must not contain duplicates: {code!r}")
            seen.add(code)
            cleaned.append(code)
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
    def _validate_eligibility_consistency(self) -> Self:
        if self.eligible:
            blocking = [
                code for code in self.reason_codes if code in _VARIABLE_BLOCKING_REASONS
            ]
            if blocking:
                raise ValueError(
                    "eligible=True cannot include blocking reason codes: "
                    f"{blocking}"
                )
        return self


class RecommendationSafetyDecision(BaseModel):
    """Structured safety-gate decision for recommendation readiness.

    Lists eligible and blocked variables with per-variable assessments.
    ``APPROVED`` / ``CAUTION`` allow later optimization; ``REFUSED`` blocks
    strong recommendations. Association is not treated as causation.
    """

    status: RecommendationSafetyStatus
    objective: RecommendationObjective
    eligible_variables: list[str]
    blocked_variables: list[str]
    variable_assessments: list[VariableEligibilityAssessment]
    global_reason_codes: list[RecommendationReasonCode]
    messages: list[str]
    disclaimer: str
    evaluated_at: datetime
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)

    @field_validator("status", mode="before")
    @classmethod
    def _validate_status(cls, value: object) -> RecommendationSafetyStatus:
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
            f"status must be RecommendationSafetyStatus, got {type(value).__name__}"
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

    @field_validator("eligible_variables", "blocked_variables", mode="before")
    @classmethod
    def _validate_variable_lists_before(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"variable list must be a list[str], got {type(value).__name__}"
            )
        return list(value)

    @field_validator("eligible_variables", "blocked_variables", mode="after")
    @classmethod
    def _validate_variable_lists(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(value, field_name="variable list")

    @field_validator("variable_assessments", mode="before")
    @classmethod
    def _validate_assessments_before(
        cls,
        value: object,
    ) -> list[VariableEligibilityAssessment]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                "variable_assessments must be a list[VariableEligibilityAssessment], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("variable_assessments", mode="after")
    @classmethod
    def _validate_assessments(
        cls,
        value: list[VariableEligibilityAssessment],
    ) -> list[VariableEligibilityAssessment]:
        seen: set[str] = set()
        copied: list[VariableEligibilityAssessment] = []
        previous_rank = 0
        for item in value:
            if not isinstance(item, VariableEligibilityAssessment):
                raise ValueError(
                    "variable_assessments entries must be "
                    "VariableEligibilityAssessment, "
                    f"got {type(item).__name__}"
                )
            if item.variable in seen:
                raise ValueError(
                    "variable_assessments must not contain duplicate variables: "
                    f"{item.variable!r}"
                )
            seen.add(item.variable)
            if item.factor_rank < previous_rank:
                raise ValueError(
                    "variable_assessments must follow diagnosis factor ranking order"
                )
            previous_rank = item.factor_rank
            copied.append(item.model_copy(deep=True))
        return copied

    @field_validator("global_reason_codes", mode="before")
    @classmethod
    def _validate_global_reason_codes_before(cls, value: object) -> list[object]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                "global_reason_codes must be a list[RecommendationReasonCode], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("global_reason_codes", mode="after")
    @classmethod
    def _validate_global_reason_codes(
        cls,
        value: list[RecommendationReasonCode],
    ) -> list[RecommendationReasonCode]:
        cleaned: list[RecommendationReasonCode] = []
        seen: set[RecommendationReasonCode] = set()
        for item in value:
            if isinstance(item, RecommendationReasonCode):
                code = item
            elif isinstance(item, str):
                try:
                    code = RecommendationReasonCode(item)
                except ValueError as exc:
                    raise ValueError(
                        f"invalid RecommendationReasonCode: {item!r}"
                    ) from exc
            else:
                raise ValueError(
                    "global_reason_codes entries must be RecommendationReasonCode, "
                    f"got {type(item).__name__}"
                )
            if code in seen:
                raise ValueError(
                    f"global_reason_codes must not contain duplicates: {code!r}"
                )
            seen.add(code)
            cleaned.append(code)
        return cleaned

    @field_validator("messages", mode="before")
    @classmethod
    def _validate_messages_before(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(f"messages must be a list[str], got {type(value).__name__}")
        return list(value)

    @field_validator("messages", mode="after")
    @classmethod
    def _validate_messages(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(value, field_name="messages")

    @field_validator("disclaimer", mode="before")
    @classmethod
    def _validate_disclaimer(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(f"disclaimer must be str, got {type(value).__name__}")
        return _validate_disclaimer_meaning(value, field_name="disclaimer")

    @field_validator("evaluated_at", mode="after")
    @classmethod
    def _validate_evaluated_at(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value, field_name="evaluated_at")

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> dict[str, ScalarMetadataValue]:
        if value is None:
            return {}
        return _validate_scalar_metadata(value)

    @model_validator(mode="after")
    def _validate_decision_consistency(self) -> Self:
        eligible_set = set(self.eligible_variables)
        blocked_set = set(self.blocked_variables)
        overlap = eligible_set & blocked_set
        if overlap:
            raise ValueError(
                "eligible_variables and blocked_variables must be disjoint; "
                f"overlap={sorted(overlap)}"
            )

        assessment_by_variable = {
            item.variable: item for item in self.variable_assessments
        }
        eligible_from_assessments = {
            item.variable for item in self.variable_assessments if item.eligible
        }
        blocked_from_assessments = {
            item.variable for item in self.variable_assessments if not item.eligible
        }
        if eligible_from_assessments != eligible_set:
            raise ValueError(
                "eligible_variables must match eligible assessments "
                f"(decision={sorted(eligible_set)}, "
                f"assessments={sorted(eligible_from_assessments)})"
            )
        if blocked_from_assessments != blocked_set:
            raise ValueError(
                "blocked_variables must match ineligible assessments "
                f"(decision={sorted(blocked_set)}, "
                f"assessments={sorted(blocked_from_assessments)})"
            )

        # Ensure listed variables appear in assessments.
        for name in self.eligible_variables + self.blocked_variables:
            if name not in assessment_by_variable:
                raise ValueError(
                    f"variable {name!r} listed without a matching assessment"
                )

        hard_blockers = [
            code for code in self.global_reason_codes if code in _HARD_GLOBAL_BLOCKERS
        ]
        has_hard_blocker = bool(hard_blockers)
        has_eligible = len(self.eligible_variables) >= 1
        has_blocked = len(self.blocked_variables) >= 1

        if self.status is RecommendationSafetyStatus.REFUSED:
            if not has_hard_blocker and has_eligible:
                raise ValueError(
                    "REFUSED requires a hard global blocker or zero eligible variables"
                )
        elif self.status is RecommendationSafetyStatus.APPROVED:
            if has_hard_blocker:
                raise ValueError("APPROVED cannot include hard global blockers")
            if not has_eligible:
                raise ValueError("APPROVED requires at least one eligible variable")
            if has_blocked:
                raise ValueError("APPROVED cannot include blocked variables")
            if RecommendationReasonCode.PARTIAL_ELIGIBILITY in self.global_reason_codes:
                raise ValueError("APPROVED cannot include PARTIAL_ELIGIBILITY")
        elif self.status is RecommendationSafetyStatus.CAUTION:
            if has_hard_blocker:
                raise ValueError("CAUTION cannot include hard global blockers")
            if not has_eligible:
                raise ValueError("CAUTION requires at least one eligible variable")
        else:
            raise ValueError(f"unsupported safety status: {self.status!r}")

        return self


class RecommendationChange(BaseModel):
    """Proposed change for a single controllable process variable.

    Intended for later optimizer steps. Step 9A defines the schema only and
    does not compute proposed values.
    """

    variable: str
    current_value: float
    proposed_value: float
    delta: float
    relative_delta: float | None = None
    rationale: str
    confidence: float
    requires_verification: bool = True

    @field_validator("variable", mode="before")
    @classmethod
    def _validate_variable(cls, value: object) -> str:
        return _validate_variable_name(value, field_name="variable")

    @field_validator("current_value", "proposed_value", "delta", mode="before")
    @classmethod
    def _validate_numeric_fields(cls, value: object) -> float:
        return _require_finite_float(value, field_name="numeric change field")

    @field_validator("relative_delta", mode="before")
    @classmethod
    def _validate_relative_delta(cls, value: object) -> float | None:
        return _require_optional_finite_float(value, field_name="relative_delta")

    @field_validator("rationale", mode="before")
    @classmethod
    def _validate_rationale(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(f"rationale must be str, got {type(value).__name__}")
        return _reject_causal_rationale(value, field_name="rationale")

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
        if self.current_value == 0.0:
            if self.relative_delta is not None:
                # relative_delta may still be provided as a finite float when
                # current is zero only if callers choose; contract allows None.
                pass
        return self


class RecommendationResult(BaseModel):
    """Recommendation outcome after safety evaluation and optional optimization.

    Step 9A concrete logic does not emit ``GENERATED`` results. ``REFUSED``
    and ``READY_FOR_OPTIMIZATION`` are the safety-gate facing statuses.
    """

    status: RecommendationStatus
    objective: RecommendationObjective
    safety_decision: RecommendationSafetyDecision
    changes: list[RecommendationChange]
    baseline_prediction: float | None = None
    proposed_prediction: float | None = None
    baseline_anomaly_score: float | None = None
    proposed_anomaly_score: float | None = None
    confidence: float
    extrapolation_flag: bool
    uncertainty_available: bool
    disclaimer: str
    generated_at: datetime
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)

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

    @field_validator("safety_decision", mode="before")
    @classmethod
    def _validate_safety_decision(
        cls,
        value: object,
    ) -> RecommendationSafetyDecision:
        if isinstance(value, RecommendationSafetyDecision):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return RecommendationSafetyDecision.model_validate(value)
        raise ValueError(
            "safety_decision must be RecommendationSafetyDecision, "
            f"got {type(value).__name__}"
        )

    @field_validator("changes", mode="before")
    @classmethod
    def _validate_changes_before(cls, value: object) -> list[RecommendationChange]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"changes must be a list[RecommendationChange], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("changes", mode="after")
    @classmethod
    def _validate_changes(
        cls,
        value: list[RecommendationChange],
    ) -> list[RecommendationChange]:
        seen: set[str] = set()
        copied: list[RecommendationChange] = []
        for item in value:
            if not isinstance(item, RecommendationChange):
                raise ValueError(
                    "changes entries must be RecommendationChange, "
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
        return _require_strict_bool(value, field_name="result bool field")

    @field_validator("disclaimer", mode="before")
    @classmethod
    def _validate_disclaimer(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="disclaimer")

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
    def _validate_result_consistency(self) -> Self:
        # baseline_anomaly_score / proposed_anomaly_score follow the anomaly
        # adapter contract: None or finite float (negative, zero, or positive).
        # Higher values mean more anomalous. Sign is not constrained here.

        safety_status = self.safety_decision.status
        eligible = set(self.safety_decision.eligible_variables)

        if self.status is RecommendationStatus.REFUSED:
            if safety_status is not RecommendationSafetyStatus.REFUSED:
                raise ValueError(
                    "REFUSED result requires safety_decision.status == REFUSED"
                )
            if self.changes:
                raise ValueError("REFUSED result must have empty changes")
            if self.proposed_prediction is not None:
                raise ValueError("REFUSED result must have proposed_prediction=None")
            if self.proposed_anomaly_score is not None:
                raise ValueError(
                    "REFUSED result must have proposed_anomaly_score=None"
                )
        elif self.status is RecommendationStatus.READY_FOR_OPTIMIZATION:
            if safety_status is RecommendationSafetyStatus.REFUSED:
                raise ValueError(
                    "READY_FOR_OPTIMIZATION requires APPROVED or CAUTION safety status"
                )
            if not self.safety_decision.eligible_variables:
                raise ValueError(
                    "READY_FOR_OPTIMIZATION requires at least one eligible variable"
                )
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
        elif self.status is RecommendationStatus.GENERATED:
            if safety_status is RecommendationSafetyStatus.REFUSED:
                raise ValueError(
                    "GENERATED requires APPROVED or CAUTION safety status"
                )
            if not self.changes:
                raise ValueError("GENERATED requires at least one change")
            for change in self.changes:
                if change.variable not in eligible:
                    raise ValueError(
                        "GENERATED change variables must be eligible in "
                        f"safety_decision; got {change.variable!r}"
                    )
        else:
            raise ValueError(f"unsupported recommendation status: {self.status!r}")

        return self


@dataclass(frozen=True, slots=True)
class RecommendationOutcome:
    """Immutable wrapper around a recommendation result.

    Holds only ``RecommendationResult``. Does not store models, estimators,
    or DataFrames, and contains no business logic.
    """

    result: RecommendationResult
