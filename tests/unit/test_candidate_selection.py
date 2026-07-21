"""Unit tests for candidate selection (Step 9B)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from process_intelligence.core.enums import AnalysisTask, ColumnRole
from process_intelligence.core.exceptions import DataValidationError
from process_intelligence.core.schemas import RootCauseFactor, VariableConstraint
from process_intelligence.diagnosis.enums import DiagnosisMethod, DiagnosisScope
from process_intelligence.diagnosis.schemas import DiagnosisResult
from process_intelligence.recommendation import (
    CandidateSelectionOutcome,
    CandidateVariable,
    CandidateVariableSelector,
    CandidateVariableSet,
    ConstraintResolutionIssue,
    ConstraintResolutionOutcome,
    ConstraintResolutionReport,
    ConstraintResolutionStatus,
    ConstraintResolver,
    ConstraintSource,
    RecommendationObjective,
    RecommendationReasonCode,
    RecommendationRequest,
    RecommendationSafetyDecision,
    RecommendationSafetyStatus,
    ResolvedVariableConstraint,
    VariableEligibilityAssessment,
)
from process_intelligence.recommendation.schemas import DEFAULT_RECOMMENDATION_DISCLAIMER


def _factor(
    variable: str,
    *,
    confidence: float = 0.8,
    direction: str = "POSITIVE",
    controllable: bool = True,
    needs_verification: bool = False,
) -> RootCauseFactor:
    return RootCauseFactor(
        variable=variable,
        direction=direction,
        deviation=1.0,
        role=ColumnRole.CONTROLLABLE_PROCESS,
        controllable=controllable,
        evidence="Associated driver; association only.",
        confidence=confidence,
        needs_verification=needs_verification,
    )


def _constraint(
    variable: str,
    *,
    minimum: float | None = 0.0,
    maximum: float | None = 100.0,
) -> VariableConstraint:
    return VariableConstraint(
        variable=variable,
        adjustable=True,
        minimum=minimum,
        maximum=maximum,
        fixed=False,
    )


def _diagnosis(factors: list[RootCauseFactor] | None = None) -> DiagnosisResult:
    return DiagnosisResult(
        anomaly_id="a-1",
        task=AnalysisTask.UNSUPERVISED_ANOMALY,
        method_used=[DiagnosisMethod.GROUP_COMPARISON],
        scope=DiagnosisScope.SINGLE_EVENT,
        factors=factors or [_factor("pressure"), _factor("temperature")],
        confidence=0.8,
        analyzed_row_count=2,
        reference_row_count=10,
        caveats=["Association only."],
        generated_at=datetime(2026, 7, 21, 9, 0, tzinfo=UTC),
    )


def _assessment(
    variable: str,
    *,
    eligible: bool = True,
    factor_rank: int = 1,
    current_value: float | None = 50.0,
    reason_codes: list[RecommendationReasonCode] | None = None,
) -> VariableEligibilityAssessment:
    return VariableEligibilityAssessment(
        variable=variable,
        factor_rank=factor_rank,
        factor_confidence=0.8,
        factor_role=str(ColumnRole.CONTROLLABLE_PROCESS),
        factor_controllable=True,
        factor_needs_verification=False,
        current_value=current_value,
        constraint_present=True,
        user_confirmed_controllable=True,
        user_verified=False,
        eligible=eligible,
        reason_codes=list(reason_codes or []),
        warnings=[],
    )


def _decision(
    *,
    status: RecommendationSafetyStatus = RecommendationSafetyStatus.APPROVED,
    assessments: list[VariableEligibilityAssessment] | None = None,
    objective: RecommendationObjective = RecommendationObjective.REDUCE_ANOMALY_SCORE,
    global_reason_codes: list[RecommendationReasonCode] | None = None,
) -> RecommendationSafetyDecision:
    if assessments is None:
        assessments = [
            _assessment("pressure", factor_rank=1),
            _assessment("temperature", factor_rank=2, current_value=80.0),
        ]
    eligible = [item.variable for item in assessments if item.eligible]
    blocked = [item.variable for item in assessments if not item.eligible]
    reasons = list(global_reason_codes or [])
    if status is RecommendationSafetyStatus.REFUSED and not reasons:
        reasons = [RecommendationReasonCode.NO_ELIGIBLE_VARIABLES]
    if status is RecommendationSafetyStatus.CAUTION and blocked:
        if RecommendationReasonCode.PARTIAL_ELIGIBILITY not in reasons:
            reasons.append(RecommendationReasonCode.PARTIAL_ELIGIBILITY)
    return RecommendationSafetyDecision(
        status=status,
        objective=objective,
        eligible_variables=eligible,
        blocked_variables=blocked,
        variable_assessments=assessments,
        global_reason_codes=reasons,
        messages=["ok"],
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        evaluated_at=datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
        metadata={},
    )


def _request(**overrides: Any) -> RecommendationRequest:
    factors = overrides.pop("factors", None)
    diagnosis = overrides.pop("diagnosis", _diagnosis(factors))
    current_values = overrides.pop(
        "current_values",
        {"pressure": 50.0, "temperature": 80.0},
    )
    constraints = overrides.pop(
        "constraints",
        [_constraint("pressure"), _constraint("temperature")],
    )
    payload: dict[str, Any] = {
        "task": AnalysisTask.UNSUPERVISED_ANOMALY,
        "diagnosis": diagnosis,
        "objective": RecommendationObjective.REDUCE_ANOMALY_SCORE,
        "current_values": current_values,
        "constraints": constraints,
        "user_confirmed_controllable_variables": list(current_values.keys()),
        "user_verified_variables": [],
        "max_simultaneous_changes": min(3, len(current_values)),
        "metadata": {},
    }
    payload.update(overrides)
    return RecommendationRequest(**payload)


def _resolved(
    variable: str = "pressure",
    *,
    current_value: float = 50.0,
    minimum: float = 0.0,
    maximum: float = 100.0,
    source_chain: list[ConstraintSource] | None = None,
    **overrides: Any,
) -> ResolvedVariableConstraint:
    span = maximum - minimum
    lower = current_value - minimum
    upper = maximum - current_value
    chain = source_chain or [ConstraintSource.REQUEST]
    payload: dict[str, Any] = {
        "variable": variable,
        "current_value": current_value,
        "minimum": minimum,
        "maximum": maximum,
        "effective_span": span,
        "lower_room": lower,
        "upper_room": upper,
        "relative_position": lower / span,
        "source_chain": chain,
        "industry_constraint_present": ConstraintSource.INDUSTRY_DEFAULT in chain,
        "request_constraint_present": ConstraintSource.REQUEST in chain,
        "user_override_present": ConstraintSource.USER_OVERRIDE in chain,
        "user_override_widened_bounds": False,
        "warnings": [],
    }
    payload.update(overrides)
    return ResolvedVariableConstraint(**payload)


def _resolution(
    *,
    status: ConstraintResolutionStatus = ConstraintResolutionStatus.READY,
    safety_status: RecommendationSafetyStatus = RecommendationSafetyStatus.APPROVED,
    eligible: list[str] | None = None,
    resolved: list[ResolvedVariableConstraint] | None = None,
    unresolved: list[str] | None = None,
) -> ConstraintResolutionOutcome:
    eligible = eligible or ["pressure", "temperature"]
    resolved = resolved or [
        _resolved("pressure", current_value=50.0),
        _resolved("temperature", current_value=80.0),
    ]
    unresolved = unresolved or []
    return ConstraintResolutionOutcome(
        report=ConstraintResolutionReport(
            status=status,
            safety_status=safety_status,
            requested_eligible_variables=list(eligible),
            resolved_variables=[item.variable for item in resolved],
            unresolved_variables=list(unresolved),
            resolved_constraints=list(resolved),
            issues=[],
            evaluated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
            warnings=[],
            metadata={"optimization_performed": False},
        )
    )


def _candidate(**overrides: Any) -> CandidateVariable:
    payload: dict[str, Any] = {
        "variable": "pressure",
        "diagnosis_rank": 1,
        "factor_confidence": 0.8,
        "factor_direction": "POSITIVE",
        "factor_controllable": True,
        "factor_needs_verification": False,
        "current_value": 50.0,
        "minimum": 0.0,
        "maximum": 100.0,
        "lower_room": 50.0,
        "upper_room": 50.0,
        "relative_position": 0.5,
        "at_lower_bound": False,
        "at_upper_bound": False,
        "source_chain": [ConstraintSource.REQUEST],
        "user_override_applied": False,
        "user_override_widened_bounds": False,
        "warnings": [],
    }
    payload.update(overrides)
    return CandidateVariable(**payload)


def _candidate_set(**overrides: Any) -> CandidateVariableSet:
    candidates = overrides.pop("candidates", [_candidate()])
    payload: dict[str, Any] = {
        "objective": RecommendationObjective.REDUCE_ANOMALY_SCORE,
        "safety_status": RecommendationSafetyStatus.APPROVED,
        "resolution_status": ConstraintResolutionStatus.READY,
        "candidates": candidates,
        "candidate_variables": [item.variable for item in candidates],
        "max_simultaneous_changes": 3,
        "effective_change_budget": min(3, len(candidates)),
        "generated_at": datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
        "warnings": [],
        "metadata": {"optimization_performed": False},
    }
    payload.update(overrides)
    if "candidate_variables" not in overrides and "candidates" in payload:
        payload["candidate_variables"] = [item.variable for item in payload["candidates"]]
    if "effective_change_budget" not in overrides:
        n = len(payload["candidates"])
        payload["effective_change_budget"] = 0 if n == 0 else min(
            payload["max_simultaneous_changes"], n
        )
    return CandidateVariableSet(**payload)


# --- CandidateVariable ---


def test_candidate_valid() -> None:
    item = _candidate()
    assert item.diagnosis_rank == 1


def test_candidate_empty_variable_rejected() -> None:
    with pytest.raises(ValidationError):
        _candidate(variable=" ")


def test_candidate_rank_zero_and_bool_rejected() -> None:
    with pytest.raises(ValidationError):
        _candidate(diagnosis_rank=0)
    with pytest.raises(ValidationError):
        _candidate(diagnosis_rank=True)  # type: ignore[arg-type]


def test_candidate_confidence_range() -> None:
    with pytest.raises(ValidationError):
        _candidate(factor_confidence=1.5)
    with pytest.raises(ValidationError):
        _candidate(factor_confidence=-0.1)


def test_candidate_direction_empty_rejected() -> None:
    with pytest.raises(ValidationError):
        _candidate(factor_direction="")


def test_candidate_bound_relation() -> None:
    with pytest.raises(ValidationError):
        _candidate(current_value=120.0, lower_room=120.0, upper_room=-20.0, relative_position=1.2)


def test_candidate_room_relation() -> None:
    with pytest.raises(ValidationError):
        _candidate(lower_room=10.0)


def test_candidate_relative_position() -> None:
    with pytest.raises(ValidationError):
        _candidate(relative_position=2.0)


def test_candidate_lower_bound_flag() -> None:
    item = _candidate(
        current_value=0.0,
        lower_room=0.0,
        upper_room=100.0,
        relative_position=0.0,
        at_lower_bound=True,
        at_upper_bound=False,
    )
    assert item.at_lower_bound is True
    with pytest.raises(ValidationError):
        _candidate(
            current_value=0.0,
            lower_room=0.0,
            upper_room=100.0,
            relative_position=0.0,
            at_lower_bound=False,
        )


def test_candidate_upper_bound_flag() -> None:
    item = _candidate(
        current_value=100.0,
        lower_room=100.0,
        upper_room=0.0,
        relative_position=1.0,
        at_lower_bound=False,
        at_upper_bound=True,
    )
    assert item.at_upper_bound is True


def test_candidate_bool_strict() -> None:
    with pytest.raises(ValidationError):
        _candidate(factor_controllable=1)  # type: ignore[arg-type]


def test_candidate_source_chain_duplicate() -> None:
    with pytest.raises(ValidationError):
        _candidate(
            source_chain=[ConstraintSource.REQUEST, ConstraintSource.REQUEST],
        )


def test_candidate_override_relation() -> None:
    with pytest.raises(ValidationError):
        _candidate(user_override_widened_bounds=True)


def test_candidate_warning_duplicate() -> None:
    with pytest.raises(ValidationError):
        _candidate(warnings=["a", "a"])


def test_candidate_round_trip() -> None:
    item = _candidate()
    assert CandidateVariable.model_validate(item.model_dump()) == item


# --- CandidateVariableSet ---


def test_candidate_set_valid() -> None:
    result = _candidate_set()
    assert result.effective_change_budget == 1


def test_candidate_set_empty() -> None:
    result = _candidate_set(
        candidates=[],
        candidate_variables=[],
        effective_change_budget=0,
        max_simultaneous_changes=1,
    )
    assert result.effective_change_budget == 0


def test_candidate_set_duplicate_variables() -> None:
    with pytest.raises(ValidationError):
        _candidate_set(
            candidates=[
                _candidate(variable="pressure", diagnosis_rank=1),
                _candidate(variable="pressure", diagnosis_rank=2),
            ]
        )


def test_candidate_set_variable_order_mismatch() -> None:
    with pytest.raises(ValidationError):
        CandidateVariableSet(
            objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
            safety_status=RecommendationSafetyStatus.APPROVED,
            resolution_status=ConstraintResolutionStatus.READY,
            candidates=[_candidate()],
            candidate_variables=["temperature"],
            max_simultaneous_changes=1,
            effective_change_budget=1,
            generated_at=datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
            warnings=[],
            metadata={},
        )


def test_candidate_set_rank_duplicate() -> None:
    with pytest.raises(ValidationError):
        _candidate_set(
            candidates=[
                _candidate(variable="pressure", diagnosis_rank=1),
                _candidate(variable="temperature", diagnosis_rank=1),
            ]
        )


def test_candidate_set_rank_order_error() -> None:
    with pytest.raises(ValidationError):
        _candidate_set(
            candidates=[
                _candidate(variable="temperature", diagnosis_rank=2),
                _candidate(variable="pressure", diagnosis_rank=1),
            ]
        )


def test_candidate_set_max_changes_invalid() -> None:
    with pytest.raises(ValidationError):
        _candidate_set(max_simultaneous_changes=0)
    with pytest.raises(ValidationError):
        _candidate_set(max_simultaneous_changes=True)  # type: ignore[arg-type]


def test_candidate_set_budget_negative() -> None:
    with pytest.raises(ValidationError):
        _candidate_set(effective_change_budget=-1)


def test_candidate_set_budget_relation_error() -> None:
    with pytest.raises(ValidationError):
        _candidate_set(effective_change_budget=2)


def test_candidate_set_refused_with_candidates_rejected() -> None:
    with pytest.raises(ValidationError):
        _candidate_set(resolution_status=ConstraintResolutionStatus.REFUSED)
    with pytest.raises(ValidationError):
        _candidate_set(safety_status=RecommendationSafetyStatus.REFUSED)


def test_candidate_set_naive_datetime() -> None:
    with pytest.raises(ValidationError):
        _candidate_set(generated_at=datetime(2026, 7, 21, 14, 0))


def test_candidate_set_metadata_limit() -> None:
    with pytest.raises(ValidationError):
        _candidate_set(metadata={"x": [1, 2]})  # type: ignore[dict-item]


def test_candidate_set_round_trip() -> None:
    result = _candidate_set()
    assert CandidateVariableSet.model_validate(result.model_dump()) == result


# --- Outcome ---


def test_outcome_frozen_and_slots() -> None:
    outcome = CandidateSelectionOutcome(candidate_set=_candidate_set())
    assert hasattr(outcome, "__slots__")
    with pytest.raises(FrozenInstanceError):
        outcome.candidate_set = _candidate_set(candidates=[])  # type: ignore[misc]


# --- Selector ---


def test_default_selector_and_metadata() -> None:
    selector = CandidateVariableSelector()
    meta = selector.get_metadata()
    assert meta["ranking_source"] == "diagnosis_factor_order"
    assert meta["generates_candidate_values"] is False
    assert meta["performs_optimization"] is False
    other = selector.get_metadata()
    assert meta is not other


def test_selector_type_errors() -> None:
    selector = CandidateVariableSelector()
    request = _request()
    decision = _decision()
    resolution = _resolution()
    with pytest.raises(TypeError):
        selector.select("bad", safety_decision=decision, resolution=resolution)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        selector.select(request, safety_decision="bad", resolution=resolution)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        selector.select(request, safety_decision=decision, resolution="bad")  # type: ignore[arg-type]


def test_selector_input_immutability() -> None:
    request = _request()
    decision = _decision()
    resolution = _resolution()
    before_req = request.model_dump()
    before_dec = decision.model_dump()
    before_res = resolution.report.model_dump()
    CandidateVariableSelector().select(
        request,
        safety_decision=decision,
        resolution=resolution,
    )
    assert request.model_dump() == before_req
    assert decision.model_dump() == before_dec
    assert resolution.report.model_dump() == before_res


# --- Consistency ---


def test_objective_mismatch() -> None:
    with pytest.raises(DataValidationError):
        CandidateVariableSelector().select(
            _request(objective=RecommendationObjective.REDUCE_ANOMALY_SCORE),
            safety_decision=_decision(
                objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY
            ),
            resolution=_resolution(),
        )


def test_safety_status_mismatch() -> None:
    with pytest.raises(DataValidationError):
        CandidateVariableSelector().select(
            _request(),
            safety_decision=_decision(status=RecommendationSafetyStatus.APPROVED),
            resolution=_resolution(safety_status=RecommendationSafetyStatus.CAUTION),
        )


def test_requested_eligible_mismatch() -> None:
    with pytest.raises(DataValidationError):
        CandidateVariableSelector().select(
            _request(),
            safety_decision=_decision(),
            resolution=_resolution(
                eligible=["pressure"],
                resolved=[_resolved("pressure")],
                unresolved=[],
            ),
        )


def test_resolved_not_subset_of_eligible() -> None:
    decision = _decision(
        assessments=[_assessment("pressure", factor_rank=1)],
    )
    request = _request(
        factors=[_factor("pressure")],
        current_values={"pressure": 50.0},
        constraints=[_constraint("pressure")],
    )
    with pytest.raises((DataValidationError, ValidationError)):
        bad = ConstraintResolutionReport(
            status=ConstraintResolutionStatus.READY,
            safety_status=RecommendationSafetyStatus.APPROVED,
            requested_eligible_variables=["pressure"],
            resolved_variables=["temperature"],
            unresolved_variables=[],
            resolved_constraints=[_resolved("temperature", current_value=80.0)],
            issues=[],
            evaluated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
            warnings=[],
            metadata={},
        )
        CandidateVariableSelector().select(
            request,
            safety_decision=decision,
            resolution=ConstraintResolutionOutcome(report=bad),
        )


def test_current_value_mismatch() -> None:
    request = _request(
        factors=[_factor("pressure")],
        current_values={"pressure": 50.0},
        constraints=[_constraint("pressure")],
    )
    decision = _decision(assessments=[_assessment("pressure")])
    resolution = _resolution(
        eligible=["pressure"],
        resolved=[_resolved("pressure", current_value=40.0)],
        unresolved=[],
    )
    with pytest.raises(DataValidationError):
        CandidateVariableSelector().select(
            request,
            safety_decision=decision,
            resolution=resolution,
        )


def test_diagnosis_factor_missing() -> None:
    request = _request(
        factors=[_factor("pressure")],
        current_values={"pressure": 50.0, "temperature": 80.0},
        constraints=[_constraint("pressure"), _constraint("temperature")],
    )
    decision = _decision()
    with pytest.raises(DataValidationError):
        CandidateVariableSelector().select(
            request,
            safety_decision=decision,
            resolution=_resolution(),
        )


# --- Empty path ---


def test_safety_refused_empty_candidates() -> None:
    request = _request(
        factors=[_factor("pressure")],
        current_values={"pressure": 50.0},
        constraints=[_constraint("pressure")],
    )
    decision = _decision(
        status=RecommendationSafetyStatus.REFUSED,
        assessments=[
            _assessment(
                "pressure",
                eligible=False,
                reason_codes=[RecommendationReasonCode.LEAKAGE_BLOCKER],
            )
        ],
        global_reason_codes=[RecommendationReasonCode.LEAKAGE_BLOCKER],
    )
    resolution = ConstraintResolutionOutcome(
        report=ConstraintResolutionReport(
            status=ConstraintResolutionStatus.REFUSED,
            safety_status=RecommendationSafetyStatus.REFUSED,
            requested_eligible_variables=[],
            resolved_variables=[],
            unresolved_variables=[],
            resolved_constraints=[],
            issues=[
                ConstraintResolutionIssue(
                    variable="*",
                    code="SAFETY_DECISION_REFUSED",
                    message="refused",
                    blocking=True,
                )
            ],
            evaluated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
            warnings=[],
            metadata={},
        )
    )
    outcome = CandidateVariableSelector().select(
        request,
        safety_decision=decision,
        resolution=resolution,
    )
    assert outcome.candidate_set.candidates == []
    assert outcome.candidate_set.effective_change_budget == 0
    assert any("No optimization candidates" in w for w in outcome.candidate_set.warnings)


def test_resolution_refused_empty_candidates() -> None:
    request = _request(
        factors=[_factor("pressure")],
        current_values={"pressure": 50.0},
        constraints=[],
    )
    decision = _decision(assessments=[_assessment("pressure")])
    resolution = ConstraintResolutionOutcome(
        report=ConstraintResolutionReport(
            status=ConstraintResolutionStatus.REFUSED,
            safety_status=RecommendationSafetyStatus.APPROVED,
            requested_eligible_variables=["pressure"],
            resolved_variables=[],
            unresolved_variables=["pressure"],
            resolved_constraints=[],
            issues=[],
            evaluated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
            warnings=[],
            metadata={},
        )
    )
    outcome = CandidateVariableSelector().select(
        request,
        safety_decision=decision,
        resolution=resolution,
    )
    assert outcome.candidate_set.candidates == []


# --- Selection ---


def test_selects_eligible_resolved_only() -> None:
    request = _request()
    decision = _decision(
        status=RecommendationSafetyStatus.CAUTION,
        assessments=[
            _assessment("pressure", eligible=True, factor_rank=1),
            _assessment(
                "temperature",
                eligible=False,
                factor_rank=2,
                current_value=80.0,
                reason_codes=[RecommendationReasonCode.NON_CONTROLLABLE_VARIABLE],
            ),
        ],
    )
    resolution = _resolution(
        status=ConstraintResolutionStatus.READY,
        safety_status=RecommendationSafetyStatus.CAUTION,
        eligible=["pressure"],
        resolved=[_resolved("pressure")],
        unresolved=[],
    )
    outcome = CandidateVariableSelector().select(
        request,
        safety_decision=decision,
        resolution=resolution,
    )
    assert outcome.candidate_set.candidate_variables == ["pressure"]


def test_excludes_unresolved() -> None:
    request = _request()
    decision = _decision()
    resolution = _resolution(
        status=ConstraintResolutionStatus.PARTIAL,
        eligible=["pressure", "temperature"],
        resolved=[_resolved("pressure")],
        unresolved=["temperature"],
    )
    outcome = CandidateVariableSelector().select(
        request,
        safety_decision=decision,
        resolution=resolution,
    )
    assert outcome.candidate_set.candidate_variables == ["pressure"]


def test_preserves_factor_fields_and_bounds() -> None:
    request = _request(
        factors=[
            _factor(
                "pressure",
                confidence=0.91,
                direction="NEGATIVE",
                controllable=True,
                needs_verification=True,
            )
        ],
        current_values={"pressure": 50.0},
        constraints=[_constraint("pressure", minimum=10.0, maximum=90.0)],
        user_verified_variables=["pressure"],
    )
    decision = _decision(assessments=[_assessment("pressure")])
    resolution = _resolution(
        eligible=["pressure"],
        resolved=[
            _resolved(
                "pressure",
                minimum=10.0,
                maximum=90.0,
                source_chain=[
                    ConstraintSource.REQUEST,
                    ConstraintSource.USER_OVERRIDE,
                ],
                user_override_present=True,
                user_override_widened_bounds=True,
                warnings=["widened"],
            )
        ],
        unresolved=[],
    )
    outcome = CandidateVariableSelector().select(
        request,
        safety_decision=decision,
        resolution=resolution,
    )
    item = outcome.candidate_set.candidates[0]
    assert item.factor_confidence == pytest.approx(0.91)
    assert item.factor_direction == "NEGATIVE"
    assert item.factor_controllable is True
    assert item.factor_needs_verification is True
    assert item.minimum == pytest.approx(10.0)
    assert item.maximum == pytest.approx(90.0)
    assert item.source_chain[-1] is ConstraintSource.USER_OVERRIDE
    assert item.user_override_applied is True
    assert item.user_override_widened_bounds is True
    assert any("widen" in w.lower() for w in item.warnings)


def test_boundary_warning() -> None:
    request = _request(
        factors=[_factor("pressure")],
        current_values={"pressure": 0.0},
        constraints=[_constraint("pressure")],
    )
    decision = _decision(
        assessments=[_assessment("pressure", current_value=0.0)]
    )
    resolution = _resolution(
        eligible=["pressure"],
        resolved=[_resolved("pressure", current_value=0.0)],
        unresolved=[],
    )
    outcome = CandidateVariableSelector().select(
        request,
        safety_decision=decision,
        resolution=resolution,
    )
    assert outcome.candidate_set.candidates[0].at_lower_bound is True
    assert any("bound" in w.lower() for w in outcome.candidate_set.warnings)


# --- Ordering ---


def test_diagnosis_order_authority() -> None:
    request = _request(
        factors=[_factor("temperature"), _factor("pressure")],
        current_values={"temperature": 80.0, "pressure": 50.0},
        constraints=[_constraint("temperature"), _constraint("pressure")],
    )
    decision = RecommendationSafetyDecision(
        status=RecommendationSafetyStatus.APPROVED,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        eligible_variables=["pressure", "temperature"],
        blocked_variables=[],
        variable_assessments=[
            _assessment("pressure", factor_rank=1),
            _assessment("temperature", factor_rank=2, current_value=80.0),
        ],
        global_reason_codes=[],
        messages=["ok"],
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        evaluated_at=datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
        metadata={},
    )
    resolution = _resolution(
        eligible=["pressure", "temperature"],
        resolved=[
            _resolved("pressure"),
            _resolved("temperature", current_value=80.0),
        ],
    )
    outcome = CandidateVariableSelector().select(
        request,
        safety_decision=decision,
        resolution=resolution,
    )
    assert outcome.candidate_set.candidate_variables == ["temperature", "pressure"]
    assert [c.diagnosis_rank for c in outcome.candidate_set.candidates] == [1, 2]
    assert any("diagnosis factor" in w.lower() for w in outcome.candidate_set.warnings)


def test_no_confidence_resort() -> None:
    request = _request(
        factors=[
            _factor("pressure", confidence=0.2),
            _factor("temperature", confidence=0.9),
        ]
    )
    decision = _decision()
    resolution = _resolution()
    outcome = CandidateVariableSelector().select(
        request,
        safety_decision=decision,
        resolution=resolution,
    )
    assert outcome.candidate_set.candidate_variables == ["pressure", "temperature"]


# --- Budget ---


def test_budget_capped_by_max_changes() -> None:
    request = _request(max_simultaneous_changes=1)
    outcome = CandidateVariableSelector().select(
        request,
        safety_decision=_decision(),
        resolution=_resolution(),
    )
    assert len(outcome.candidate_set.candidates) == 2
    assert outcome.candidate_set.effective_change_budget == 1
    assert outcome.candidate_set.max_simultaneous_changes == 1


def test_budget_equals_candidate_count_when_smaller() -> None:
    request = _request(
        factors=[_factor("pressure")],
        current_values={"pressure": 50.0},
        constraints=[_constraint("pressure")],
        max_simultaneous_changes=1,
    )
    decision = _decision(assessments=[_assessment("pressure")])
    resolution = _resolution(
        eligible=["pressure"],
        resolved=[_resolved("pressure")],
        unresolved=[],
    )
    outcome = CandidateVariableSelector().select(
        request,
        safety_decision=decision,
        resolution=resolution,
    )
    assert outcome.candidate_set.effective_change_budget == 1


def test_no_combinations_or_proposed_values() -> None:
    outcome = CandidateVariableSelector().select(
        _request(),
        safety_decision=_decision(),
        resolution=_resolution(),
    )
    dumped = outcome.candidate_set.model_dump()
    assert "proposed_value" not in dumped
    assert outcome.candidate_set.metadata["candidate_values_generated"] is False
    assert outcome.candidate_set.metadata["optimization_performed"] is False


# --- Output / state ---


def test_warning_order_and_metadata() -> None:
    request = _request(
        factors=[_factor("pressure"), _factor("temperature")],
        current_values={"pressure": 0.0, "temperature": 80.0},
        constraints=[_constraint("pressure"), _constraint("temperature")],
    )
    decision = _decision(
        status=RecommendationSafetyStatus.CAUTION,
        assessments=[
            _assessment("pressure", eligible=True, factor_rank=1, current_value=0.0),
            _assessment(
                "temperature",
                eligible=False,
                factor_rank=2,
                current_value=80.0,
                reason_codes=[RecommendationReasonCode.NON_CONTROLLABLE_VARIABLE],
            ),
        ],
    )
    resolution = _resolution(
        status=ConstraintResolutionStatus.READY,
        safety_status=RecommendationSafetyStatus.CAUTION,
        eligible=["pressure"],
        resolved=[_resolved("pressure", current_value=0.0)],
        unresolved=[],
    )
    outcome = CandidateVariableSelector().select(
        request,
        safety_decision=decision,
        resolution=resolution,
    )
    warnings = outcome.candidate_set.warnings
    assert warnings[0].startswith("Safety decision status is CAUTION")
    assert len(warnings) == len(set(warnings))
    assert outcome.candidate_set.metadata["ranking_source"] == "diagnosis_factor_order"
    assert outcome.candidate_set.metadata["candidate_count"] == 1


def test_no_model_in_metadata() -> None:
    outcome = CandidateVariableSelector().select(
        _request(),
        safety_decision=_decision(),
        resolution=_resolution(),
    )
    for value in outcome.candidate_set.metadata.values():
        assert value is None or isinstance(value, (str, int, float, bool))


def test_selector_stateless_and_isolated() -> None:
    selector = CandidateVariableSelector()
    first = selector.select(
        _request(),
        safety_decision=_decision(),
        resolution=_resolution(),
    )
    second = selector.select(
        _request(
            factors=[_factor("pressure")],
            current_values={"pressure": 50.0},
            constraints=[_constraint("pressure")],
        ),
        safety_decision=_decision(assessments=[_assessment("pressure")]),
        resolution=_resolution(
            eligible=["pressure"],
            resolved=[_resolved("pressure")],
            unresolved=[],
        ),
    )
    assert len(first.candidate_set.candidates) == 2
    assert len(second.candidate_set.candidates) == 1
    first.candidate_set.candidate_variables.append("hacked")
    again = selector.select(
        _request(),
        safety_decision=_decision(),
        resolution=_resolution(),
    )
    assert "hacked" not in again.candidate_set.candidate_variables


def test_end_to_end_with_resolver() -> None:
    request = _request()
    decision = _decision()
    resolution = ConstraintResolver().resolve(request, safety_decision=decision)
    outcome = CandidateVariableSelector().select(
        request,
        safety_decision=decision,
        resolution=resolution,
    )
    assert outcome.candidate_set.candidate_variables == ["pressure", "temperature"]
    assert outcome.candidate_set.metadata["optimization_performed"] is False
