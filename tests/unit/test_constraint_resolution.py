"""Unit tests for constraint resolution (Step 9B)."""

from __future__ import annotations

import math
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
    ConstraintResolutionIssue,
    ConstraintResolutionOutcome,
    ConstraintResolutionPolicy,
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
    role: ColumnRole = ColumnRole.CONTROLLABLE_PROCESS,
    controllable: bool = True,
    confidence: float = 0.8,
    needs_verification: bool = False,
    direction: str = "POSITIVE",
) -> RootCauseFactor:
    return RootCauseFactor(
        variable=variable,
        direction=direction,
        deviation=1.0,
        role=role,
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


def _diagnosis(
    factors: list[RootCauseFactor] | None = None,
) -> DiagnosisResult:
    return DiagnosisResult(
        anomaly_id="a-1",
        task=AnalysisTask.UNSUPERVISED_ANOMALY,
        method_used=[DiagnosisMethod.GROUP_COMPARISON],
        scope=DiagnosisScope.SINGLE_EVENT,
        factors=factors
        or [
            _factor("pressure"),
            _factor("temperature"),
        ],
        confidence=0.8,
        analyzed_row_count=2,
        reference_row_count=10,
        caveats=["Association only; causation is not established."],
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
    eligible: list[str] | None = None,
    blocked: list[str] | None = None,
    assessments: list[VariableEligibilityAssessment] | None = None,
    objective: RecommendationObjective = RecommendationObjective.REDUCE_ANOMALY_SCORE,
    global_reason_codes: list[RecommendationReasonCode] | None = None,
) -> RecommendationSafetyDecision:
    if assessments is None:
        eligible = eligible or ["pressure", "temperature"]
        blocked = blocked or []
        assessments = [
            _assessment(name, eligible=True, factor_rank=idx + 1)
            for idx, name in enumerate(eligible)
        ]
        assessments.extend(
            _assessment(
                name,
                eligible=False,
                factor_rank=len(eligible) + idx + 1,
                reason_codes=[RecommendationReasonCode.NON_CONTROLLABLE_VARIABLE],
            )
            for idx, name in enumerate(blocked)
        )
        # For APPROVED, blocked must be empty; rebuild when status needs factors.
        if status is RecommendationSafetyStatus.APPROVED:
            assessments = [
                item for item in assessments if item.eligible
            ]
            blocked = []
    eligible_names = [item.variable for item in assessments if item.eligible]
    blocked_names = [item.variable for item in assessments if not item.eligible]
    reasons = list(global_reason_codes or [])
    if status is RecommendationSafetyStatus.REFUSED and not reasons:
        reasons = [RecommendationReasonCode.NO_ELIGIBLE_VARIABLES]
    if status is RecommendationSafetyStatus.CAUTION and blocked_names:
        if RecommendationReasonCode.PARTIAL_ELIGIBILITY not in reasons:
            reasons.append(RecommendationReasonCode.PARTIAL_ELIGIBILITY)
    return RecommendationSafetyDecision(
        status=status,
        objective=objective,
        eligible_variables=eligible_names,
        blocked_variables=blocked_names,
        variable_assessments=assessments,
        global_reason_codes=reasons,
        messages=["Safety evaluation complete."],
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


def _report(**overrides: Any) -> ConstraintResolutionReport:
    resolved = [_resolved("pressure")]
    payload: dict[str, Any] = {
        "status": ConstraintResolutionStatus.READY,
        "safety_status": RecommendationSafetyStatus.APPROVED,
        "requested_eligible_variables": ["pressure"],
        "resolved_variables": ["pressure"],
        "unresolved_variables": [],
        "resolved_constraints": resolved,
        "issues": [],
        "evaluated_at": datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
        "warnings": [],
        "metadata": {"optimization_performed": False},
    }
    payload.update(overrides)
    return ConstraintResolutionReport(**payload)


# --- Enum and Policy ---


def test_constraint_source_values() -> None:
    assert ConstraintSource.INDUSTRY_DEFAULT == "INDUSTRY_DEFAULT"
    assert ConstraintSource.REQUEST == "REQUEST"
    assert ConstraintSource.USER_OVERRIDE == "USER_OVERRIDE"
    assert set(ConstraintSource) == {
        ConstraintSource.INDUSTRY_DEFAULT,
        ConstraintSource.REQUEST,
        ConstraintSource.USER_OVERRIDE,
    }


def test_resolution_status_values() -> None:
    assert set(ConstraintResolutionStatus) == {
        ConstraintResolutionStatus.READY,
        ConstraintResolutionStatus.PARTIAL,
        ConstraintResolutionStatus.REFUSED,
    }


def test_default_policy() -> None:
    policy = ConstraintResolutionPolicy()
    assert policy.require_non_refused_safety_decision is True
    assert policy.intersect_industry_and_request_constraints is True
    assert policy.allow_user_bound_widening is False
    assert policy.require_current_value_within_bounds is True
    assert policy.minimum_effective_span == pytest.approx(1e-12)
    assert policy.allow_partial_resolution is True
    assert policy.preserve_safety_ranking is True


@pytest.mark.parametrize("field", [
    "require_non_refused_safety_decision",
    "intersect_industry_and_request_constraints",
    "allow_user_bound_widening",
    "require_current_value_within_bounds",
    "allow_partial_resolution",
    "preserve_safety_ranking",
])
def test_policy_bool_strict(field: str) -> None:
    with pytest.raises(ValidationError):
        ConstraintResolutionPolicy(**{field: 1})
    with pytest.raises(ValidationError):
        ConstraintResolutionPolicy(**{field: "true"})


def test_policy_minimum_span_negative_rejected() -> None:
    with pytest.raises(ValidationError):
        ConstraintResolutionPolicy(minimum_effective_span=-1.0)


def test_policy_minimum_span_bool_rejected() -> None:
    with pytest.raises(ValidationError):
        ConstraintResolutionPolicy(minimum_effective_span=True)


def test_policy_minimum_span_nan_inf_rejected() -> None:
    with pytest.raises(ValidationError):
        ConstraintResolutionPolicy(minimum_effective_span=math.nan)
    with pytest.raises(ValidationError):
        ConstraintResolutionPolicy(minimum_effective_span=math.inf)


def test_policy_round_trip() -> None:
    policy = ConstraintResolutionPolicy(allow_partial_resolution=False)
    restored = ConstraintResolutionPolicy.model_validate(policy.model_dump())
    assert restored == policy


# --- Issue ---


def test_issue_valid() -> None:
    issue = ConstraintResolutionIssue(
        variable="pressure",
        source=ConstraintSource.REQUEST,
        code="CONSTRAINT_MISSING",
        message="Missing constraint",
        blocking=True,
    )
    assert issue.blocking is True


def test_issue_empty_variable_rejected() -> None:
    with pytest.raises(ValidationError):
        ConstraintResolutionIssue(
            variable=" ",
            code="CONSTRAINT_MISSING",
            message="x",
            blocking=True,
        )


def test_issue_empty_code_rejected() -> None:
    with pytest.raises(ValidationError):
        ConstraintResolutionIssue(
            variable="pressure",
            code=" ",
            message="x",
            blocking=True,
        )


def test_issue_unknown_code_rejected() -> None:
    with pytest.raises(ValidationError):
        ConstraintResolutionIssue(
            variable="pressure",
            code="NOT_A_REAL_CODE",
            message="x",
            blocking=True,
        )


def test_issue_empty_message_rejected() -> None:
    with pytest.raises(ValidationError):
        ConstraintResolutionIssue(
            variable="pressure",
            code="CONSTRAINT_MISSING",
            message="",
            blocking=True,
        )


def test_issue_blocking_strict_bool() -> None:
    with pytest.raises(ValidationError):
        ConstraintResolutionIssue(
            variable="pressure",
            code="CONSTRAINT_MISSING",
            message="x",
            blocking=1,  # type: ignore[arg-type]
        )


def test_issue_round_trip() -> None:
    issue = ConstraintResolutionIssue(
        variable="pressure",
        code="CONFLICTING_BOUNDS",
        message="Conflict",
        blocking=True,
    )
    assert ConstraintResolutionIssue.model_validate(issue.model_dump()) == issue


# --- Resolved schema ---


def test_resolved_valid() -> None:
    item = _resolved()
    assert item.effective_span == pytest.approx(100.0)
    assert item.relative_position == pytest.approx(0.5)


def test_resolved_minimum_gt_current_rejected() -> None:
    with pytest.raises(ValidationError):
        _resolved(current_value=10.0, minimum=20.0, maximum=100.0)


def test_resolved_current_gt_maximum_rejected() -> None:
    with pytest.raises(ValidationError):
        _resolved(current_value=120.0, minimum=0.0, maximum=100.0)


def test_resolved_span_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        _resolved(effective_span=50.0)


def test_resolved_room_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        _resolved(lower_room=10.0)


def test_resolved_relative_position_out_of_range() -> None:
    with pytest.raises(ValidationError):
        _resolved(relative_position=1.5)


def test_resolved_source_chain_duplicate_rejected() -> None:
    with pytest.raises(ValidationError):
        _resolved(
            source_chain=[ConstraintSource.REQUEST, ConstraintSource.REQUEST],
        )


def test_resolved_source_chain_order_rejected() -> None:
    with pytest.raises(ValidationError):
        _resolved(
            source_chain=[
                ConstraintSource.USER_OVERRIDE,
                ConstraintSource.REQUEST,
            ],
            user_override_present=True,
            request_constraint_present=True,
        )


def test_resolved_presence_flag_mismatch() -> None:
    with pytest.raises(ValidationError):
        _resolved(
            source_chain=[ConstraintSource.REQUEST],
            industry_constraint_present=True,
        )


def test_resolved_widening_requires_override() -> None:
    with pytest.raises(ValidationError):
        _resolved(user_override_widened_bounds=True)


def test_resolved_warning_duplicate_rejected() -> None:
    with pytest.raises(ValidationError):
        _resolved(warnings=["a", "a"])


def test_resolved_round_trip() -> None:
    item = _resolved()
    assert ResolvedVariableConstraint.model_validate(item.model_dump()) == item


# --- Report and Outcome ---


def test_ready_report() -> None:
    report = _report()
    assert report.status is ConstraintResolutionStatus.READY


def test_partial_report() -> None:
    report = _report(
        status=ConstraintResolutionStatus.PARTIAL,
        requested_eligible_variables=["pressure", "temperature"],
        resolved_variables=["pressure"],
        unresolved_variables=["temperature"],
        resolved_constraints=[_resolved("pressure")],
    )
    assert report.status is ConstraintResolutionStatus.PARTIAL


def test_refused_report() -> None:
    report = _report(
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
    )
    assert report.status is ConstraintResolutionStatus.REFUSED


def test_report_duplicate_variables_rejected() -> None:
    with pytest.raises(ValidationError):
        _report(resolved_variables=["pressure", "pressure"])


def test_report_resolved_unresolved_overlap_rejected() -> None:
    with pytest.raises(ValidationError):
        _report(
            status=ConstraintResolutionStatus.PARTIAL,
            requested_eligible_variables=["pressure"],
            resolved_variables=["pressure"],
            unresolved_variables=["pressure"],
        )


def test_report_requested_set_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        _report(
            requested_eligible_variables=["pressure", "temperature"],
            resolved_variables=["pressure"],
            unresolved_variables=[],
        )


def test_report_constraint_order_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        _report(
            status=ConstraintResolutionStatus.READY,
            requested_eligible_variables=["pressure", "temperature"],
            resolved_variables=["pressure", "temperature"],
            unresolved_variables=[],
            resolved_constraints=[
                _resolved("temperature", current_value=80.0),
                _resolved("pressure"),
            ],
        )


def test_report_status_relation_error() -> None:
    with pytest.raises(ValidationError):
        _report(
            status=ConstraintResolutionStatus.READY,
            unresolved_variables=["x"],
            requested_eligible_variables=["pressure", "x"],
            resolved_variables=["pressure"],
        )


def test_report_naive_datetime_rejected() -> None:
    with pytest.raises(ValidationError):
        _report(evaluated_at=datetime(2026, 7, 21, 13, 0))


def test_report_metadata_rejects_nested() -> None:
    with pytest.raises(ValidationError):
        _report(metadata={"bad": {"nested": 1}})  # type: ignore[dict-item]


def test_outcome_frozen() -> None:
    outcome = ConstraintResolutionOutcome(report=_report())
    with pytest.raises(FrozenInstanceError):
        outcome.report = _report(  # type: ignore[misc]
            status=ConstraintResolutionStatus.REFUSED,
            resolved_variables=[],
            unresolved_variables=["pressure"],
            resolved_constraints=[],
            requested_eligible_variables=["pressure"],
        )


# --- Resolver construction ---


def test_default_resolver() -> None:
    resolver = ConstraintResolver()
    meta = resolver.get_metadata()
    assert meta["generates_candidate_values"] is False
    assert meta["performs_optimization"] is False


def test_custom_policy_resolver() -> None:
    policy = ConstraintResolutionPolicy(allow_partial_resolution=False)
    resolver = ConstraintResolver(policy=policy)
    assert resolver.get_metadata()["allow_partial_resolution"] is False


def test_resolver_policy_type_error() -> None:
    with pytest.raises(TypeError):
        ConstraintResolver(policy="bad")  # type: ignore[arg-type]


def test_resolver_policy_immutability() -> None:
    policy = ConstraintResolutionPolicy(allow_partial_resolution=True)
    resolver = ConstraintResolver(policy=policy)
    policy.allow_partial_resolution = False
    assert resolver.get_metadata()["allow_partial_resolution"] is True


def test_resolver_state_isolation() -> None:
    a = ConstraintResolver(
        policy=ConstraintResolutionPolicy(allow_partial_resolution=True)
    )
    b = ConstraintResolver(
        policy=ConstraintResolutionPolicy(allow_partial_resolution=False)
    )
    assert a.get_metadata()["allow_partial_resolution"] is True
    assert b.get_metadata()["allow_partial_resolution"] is False


def test_resolver_metadata_independence() -> None:
    resolver = ConstraintResolver()
    first = resolver.get_metadata()
    second = resolver.get_metadata()
    assert first == second
    assert first is not second
    first["performs_optimization"] = True
    assert resolver.get_metadata()["performs_optimization"] is False


# --- Inputs ---


def test_resolve_request_type_error() -> None:
    resolver = ConstraintResolver()
    with pytest.raises(TypeError):
        resolver.resolve(
            "bad",  # type: ignore[arg-type]
            safety_decision=_decision(),
        )


def test_resolve_safety_type_error() -> None:
    resolver = ConstraintResolver()
    with pytest.raises(TypeError):
        resolver.resolve(_request(), safety_decision="bad")  # type: ignore[arg-type]


def test_resolve_industry_sequence_type_error() -> None:
    resolver = ConstraintResolver()
    with pytest.raises(TypeError):
        resolver.resolve(
            _request(),
            safety_decision=_decision(),
            industry_constraints="bad",  # type: ignore[arg-type]
        )


def test_resolve_user_override_sequence_type_error() -> None:
    resolver = ConstraintResolver()
    with pytest.raises(TypeError):
        resolver.resolve(
            _request(),
            safety_decision=_decision(),
            user_overrides="bad",  # type: ignore[arg-type]
        )


def test_resolve_constraint_element_type_error() -> None:
    resolver = ConstraintResolver()
    with pytest.raises(TypeError):
        resolver.resolve(
            _request(),
            safety_decision=_decision(),
            industry_constraints=[object()],  # type: ignore[list-item]
        )


def test_resolve_input_immutability() -> None:
    request = _request()
    decision = _decision()
    industry = [_constraint("pressure", minimum=10.0, maximum=90.0)]
    overrides = [_constraint("pressure", minimum=20.0, maximum=80.0)]
    industry_before = industry[0].model_dump()
    override_before = overrides[0].model_dump()
    request_constraints_before = [c.model_dump() for c in request.constraints]
    decision_before = decision.model_dump()

    resolver = ConstraintResolver()
    resolver.resolve(
        request,
        safety_decision=decision,
        industry_constraints=industry,
        user_overrides=overrides,
    )

    assert industry[0].model_dump() == industry_before
    assert overrides[0].model_dump() == override_before
    assert [c.model_dump() for c in request.constraints] == request_constraints_before
    assert decision.model_dump() == decision_before


# --- Safety consistency ---


def test_objective_mismatch_rejected() -> None:
    resolver = ConstraintResolver()
    request = _request(objective=RecommendationObjective.REDUCE_ANOMALY_SCORE)
    decision = _decision(
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
    )
    with pytest.raises(DataValidationError):
        resolver.resolve(request, safety_decision=decision)


def test_refused_safety_structured_refusal() -> None:
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
    outcome = ConstraintResolver().resolve(request, safety_decision=decision)
    assert outcome.report.status is ConstraintResolutionStatus.REFUSED
    assert outcome.report.resolved_variables == []
    assert any(
        issue.code == "SAFETY_DECISION_REFUSED" for issue in outcome.report.issues
    )
    assert outcome.report.metadata["optimization_performed"] is False


def test_require_non_refused_false_still_resolves_when_possible() -> None:
    # REFUSED with hard blocker can still list eligible assessments in schema
    # only when hard blocker present; build eligible empty REFUSED and disable
    # early refuse to exercise the path with empty eligible → REFUSED.
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
    resolver = ConstraintResolver(
        policy=ConstraintResolutionPolicy(require_non_refused_safety_decision=False)
    )
    outcome = resolver.resolve(request, safety_decision=decision)
    assert outcome.report.status is ConstraintResolutionStatus.REFUSED
    assert outcome.report.resolved_variables == []


def test_eligible_order_preserved() -> None:
    request = _request()
    decision = _decision(eligible=["temperature", "pressure"])
    # Rebuild decision with custom order matching assessments ranking.
    decision = RecommendationSafetyDecision(
        status=RecommendationSafetyStatus.APPROVED,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        eligible_variables=["temperature", "pressure"],
        blocked_variables=[],
        variable_assessments=[
            _assessment("temperature", factor_rank=1, current_value=80.0),
            _assessment("pressure", factor_rank=2, current_value=50.0),
        ],
        global_reason_codes=[],
        messages=["ok"],
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        evaluated_at=datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
        metadata={},
    )
    # diagnosis factors order differs; safety consistency requires assessment
    # variables == factor variables (set), not order.
    request = _request(
        factors=[_factor("temperature"), _factor("pressure")],
        current_values={"temperature": 80.0, "pressure": 50.0},
        constraints=[_constraint("temperature"), _constraint("pressure")],
    )
    outcome = ConstraintResolver().resolve(request, safety_decision=decision)
    assert outcome.report.resolved_variables == ["temperature", "pressure"]


# --- Source mapping ---


def test_industry_duplicate_constraint_rejected() -> None:
    resolver = ConstraintResolver()
    with pytest.raises(DataValidationError):
        resolver.resolve(
            _request(),
            safety_decision=_decision(),
            industry_constraints=[
                _constraint("pressure"),
                _constraint("pressure", minimum=1.0),
            ],
        )


def test_request_duplicate_rejected_by_schema() -> None:
    with pytest.raises(ValidationError):
        _request(
            constraints=[
                _constraint("pressure"),
                _constraint("pressure", minimum=1.0),
            ]
        )


def test_user_override_duplicate_rejected() -> None:
    resolver = ConstraintResolver()
    with pytest.raises(DataValidationError):
        resolver.resolve(
            _request(),
            safety_decision=_decision(),
            user_overrides=[
                _constraint("pressure", minimum=10.0, maximum=90.0),
                _constraint("pressure", minimum=20.0, maximum=80.0),
            ],
        )


def test_non_eligible_constraint_ignored() -> None:
    request = _request(
        factors=[_factor("pressure")],
        current_values={"pressure": 50.0},
        constraints=[_constraint("pressure")],
    )
    decision = _decision(
        assessments=[_assessment("pressure", eligible=True, factor_rank=1)]
    )
    outcome = ConstraintResolver().resolve(
        request,
        safety_decision=decision,
        industry_constraints=[
            _constraint("pressure", minimum=0.0, maximum=100.0),
            _constraint("humidity", minimum=0.0, maximum=50.0),
        ],
    )
    assert outcome.report.resolved_variables == ["pressure"]
    assert "humidity" not in outcome.report.resolved_variables


def test_variable_identity_exact() -> None:
    outcome = ConstraintResolver().resolve(
        _request(
            factors=[_factor("pressure")],
            current_values={"pressure": 50.0},
            constraints=[_constraint("pressure", minimum=10.0, maximum=90.0)],
        ),
        safety_decision=_decision(
            assessments=[_assessment("pressure", factor_rank=1)]
        ),
        industry_constraints=[_constraint("pressure", minimum=0.0, maximum=100.0)],
    )
    item = outcome.report.resolved_constraints[0]
    assert item.variable == "pressure"
    assert item.minimum == pytest.approx(10.0)
    assert item.maximum == pytest.approx(90.0)


# --- Merge ---


def test_industry_only() -> None:
    request = _request(
        factors=[_factor("pressure")],
        current_values={"pressure": 50.0},
        constraints=[],
    )
    decision = _decision(assessments=[_assessment("pressure")])
    outcome = ConstraintResolver().resolve(
        request,
        safety_decision=decision,
        industry_constraints=[_constraint("pressure", minimum=5.0, maximum=95.0)],
    )
    item = outcome.report.resolved_constraints[0]
    assert item.minimum == pytest.approx(5.0)
    assert item.maximum == pytest.approx(95.0)
    assert item.source_chain == [ConstraintSource.INDUSTRY_DEFAULT]


def test_request_only() -> None:
    request = _request(
        factors=[_factor("pressure")],
        current_values={"pressure": 50.0},
        constraints=[_constraint("pressure", minimum=20.0, maximum=80.0)],
    )
    decision = _decision(assessments=[_assessment("pressure")])
    outcome = ConstraintResolver().resolve(request, safety_decision=decision)
    item = outcome.report.resolved_constraints[0]
    assert item.minimum == pytest.approx(20.0)
    assert item.source_chain == [ConstraintSource.REQUEST]


def test_industry_request_intersection() -> None:
    request = _request(
        factors=[_factor("pressure")],
        current_values={"pressure": 50.0},
        constraints=[_constraint("pressure", minimum=20.0, maximum=80.0)],
    )
    decision = _decision(assessments=[_assessment("pressure")])
    outcome = ConstraintResolver().resolve(
        request,
        safety_decision=decision,
        industry_constraints=[_constraint("pressure", minimum=10.0, maximum=90.0)],
    )
    item = outcome.report.resolved_constraints[0]
    assert item.minimum == pytest.approx(20.0)
    assert item.maximum == pytest.approx(80.0)


def test_intersection_conflict() -> None:
    request = _request(
        factors=[_factor("pressure")],
        current_values={"pressure": 50.0},
        constraints=[_constraint("pressure", minimum=60.0, maximum=80.0)],
    )
    decision = _decision(assessments=[_assessment("pressure", current_value=50.0)])
    # current 50 is outside request bounds already for safety, but for resolution:
    request = _request(
        factors=[_factor("pressure")],
        current_values={"pressure": 70.0},
        constraints=[_constraint("pressure", minimum=60.0, maximum=80.0)],
    )
    decision = _decision(
        assessments=[_assessment("pressure", current_value=70.0)]
    )
    outcome = ConstraintResolver().resolve(
        request,
        safety_decision=decision,
        industry_constraints=[_constraint("pressure", minimum=0.0, maximum=50.0)],
    )
    assert outcome.report.status is ConstraintResolutionStatus.REFUSED
    assert any(i.code == "CONFLICTING_BOUNDS" for i in outcome.report.issues)


def test_intersect_false_request_priority() -> None:
    request = _request(
        factors=[_factor("pressure")],
        current_values={"pressure": 50.0},
        constraints=[_constraint("pressure", minimum=30.0, maximum=70.0)],
    )
    decision = _decision(assessments=[_assessment("pressure")])
    resolver = ConstraintResolver(
        policy=ConstraintResolutionPolicy(
            intersect_industry_and_request_constraints=False
        )
    )
    outcome = resolver.resolve(
        request,
        safety_decision=decision,
        industry_constraints=[_constraint("pressure", minimum=0.0, maximum=100.0)],
    )
    item = outcome.report.resolved_constraints[0]
    assert item.minimum == pytest.approx(30.0)
    assert item.maximum == pytest.approx(70.0)
    assert item.source_chain == [ConstraintSource.REQUEST]


def test_all_constraints_missing() -> None:
    request = _request(
        factors=[_factor("pressure")],
        current_values={"pressure": 50.0},
        constraints=[],
    )
    decision = _decision(assessments=[_assessment("pressure")])
    outcome = ConstraintResolver().resolve(request, safety_decision=decision)
    assert outcome.report.resolved_variables == []
    assert any(i.code == "CONSTRAINT_MISSING" for i in outcome.report.issues)


def test_one_sided_bound_merge_to_full() -> None:
    request = _request(
        factors=[_factor("pressure")],
        current_values={"pressure": 50.0},
        constraints=[_constraint("pressure", minimum=None, maximum=90.0)],
    )
    decision = _decision(assessments=[_assessment("pressure")])
    outcome = ConstraintResolver().resolve(
        request,
        safety_decision=decision,
        industry_constraints=[_constraint("pressure", minimum=10.0, maximum=None)],
    )
    item = outcome.report.resolved_constraints[0]
    assert item.minimum == pytest.approx(10.0)
    assert item.maximum == pytest.approx(90.0)


def test_one_sided_final_unresolved() -> None:
    request = _request(
        factors=[_factor("pressure")],
        current_values={"pressure": 50.0},
        constraints=[_constraint("pressure", minimum=10.0, maximum=None)],
    )
    decision = _decision(assessments=[_assessment("pressure")])
    outcome = ConstraintResolver().resolve(request, safety_decision=decision)
    assert "pressure" in outcome.report.unresolved_variables
    assert any(
        i.code in {"CONSTRAINT_MISSING", "UNSUPPORTED_CONSTRAINT_SHAPE"}
        for i in outcome.report.issues
    )


def test_finite_bound_result() -> None:
    outcome = ConstraintResolver().resolve(
        _request(
            factors=[_factor("pressure")],
            current_values={"pressure": 50.0},
            constraints=[_constraint("pressure")],
        ),
        safety_decision=_decision(assessments=[_assessment("pressure")]),
    )
    item = outcome.report.resolved_constraints[0]
    assert math.isfinite(item.minimum)
    assert math.isfinite(item.maximum)


# --- User override ---


def test_user_override_safe_narrowing() -> None:
    outcome = ConstraintResolver().resolve(
        _request(
            factors=[_factor("pressure")],
            current_values={"pressure": 50.0},
            constraints=[_constraint("pressure", minimum=0.0, maximum=100.0)],
        ),
        safety_decision=_decision(assessments=[_assessment("pressure")]),
        user_overrides=[_constraint("pressure", minimum=20.0, maximum=80.0)],
    )
    item = outcome.report.resolved_constraints[0]
    assert item.minimum == pytest.approx(20.0)
    assert item.maximum == pytest.approx(80.0)
    assert item.user_override_present is True


def test_partial_widening_safe_intersection() -> None:
    outcome = ConstraintResolver().resolve(
        _request(
            factors=[_factor("pressure")],
            current_values={"pressure": 50.0},
            constraints=[_constraint("pressure", minimum=20.0, maximum=80.0)],
        ),
        safety_decision=_decision(assessments=[_assessment("pressure")]),
        user_overrides=[_constraint("pressure", minimum=10.0, maximum=70.0)],
    )
    item = outcome.report.resolved_constraints[0]
    assert item.minimum == pytest.approx(20.0)
    assert item.maximum == pytest.approx(70.0)
    assert any(i.code == "USER_OVERRIDE_WIDENS_BOUNDS" for i in outcome.report.issues)


def test_complete_non_overlap_unresolved() -> None:
    outcome = ConstraintResolver().resolve(
        _request(
            factors=[_factor("pressure")],
            current_values={"pressure": 50.0},
            constraints=[_constraint("pressure", minimum=0.0, maximum=40.0)],
        ),
        safety_decision=_decision(
            assessments=[_assessment("pressure", current_value=30.0)]
        ),
        user_overrides=[_constraint("pressure", minimum=50.0, maximum=100.0)],
    )
    # Adjust current into base range.
    outcome = ConstraintResolver().resolve(
        _request(
            factors=[_factor("pressure")],
            current_values={"pressure": 30.0},
            constraints=[_constraint("pressure", minimum=0.0, maximum=40.0)],
        ),
        safety_decision=_decision(
            assessments=[_assessment("pressure", current_value=30.0)]
        ),
        user_overrides=[_constraint("pressure", minimum=50.0, maximum=100.0)],
    )
    assert "pressure" in outcome.report.unresolved_variables


def test_widening_allowed_false_does_not_expand() -> None:
    outcome = ConstraintResolver(
        policy=ConstraintResolutionPolicy(allow_user_bound_widening=False)
    ).resolve(
        _request(
            factors=[_factor("pressure")],
            current_values={"pressure": 50.0},
            constraints=[_constraint("pressure", minimum=20.0, maximum=80.0)],
        ),
        safety_decision=_decision(assessments=[_assessment("pressure")]),
        user_overrides=[_constraint("pressure", minimum=0.0, maximum=100.0)],
    )
    item = outcome.report.resolved_constraints[0]
    assert item.minimum == pytest.approx(20.0)
    assert item.maximum == pytest.approx(80.0)
    assert item.user_override_widened_bounds is False


def test_widening_allowed_true() -> None:
    outcome = ConstraintResolver(
        policy=ConstraintResolutionPolicy(allow_user_bound_widening=True)
    ).resolve(
        _request(
            factors=[_factor("pressure")],
            current_values={"pressure": 50.0},
            constraints=[_constraint("pressure", minimum=20.0, maximum=80.0)],
        ),
        safety_decision=_decision(assessments=[_assessment("pressure")]),
        user_overrides=[_constraint("pressure", minimum=0.0, maximum=100.0)],
    )
    item = outcome.report.resolved_constraints[0]
    assert item.minimum == pytest.approx(0.0)
    assert item.maximum == pytest.approx(100.0)
    assert item.user_override_widened_bounds is True


def test_widening_flag_and_source_chain() -> None:
    outcome = ConstraintResolver(
        policy=ConstraintResolutionPolicy(allow_user_bound_widening=True)
    ).resolve(
        _request(
            factors=[_factor("pressure")],
            current_values={"pressure": 50.0},
            constraints=[_constraint("pressure")],
        ),
        safety_decision=_decision(assessments=[_assessment("pressure")]),
        industry_constraints=[_constraint("pressure", minimum=10.0, maximum=90.0)],
        user_overrides=[_constraint("pressure", minimum=0.0, maximum=100.0)],
    )
    item = outcome.report.resolved_constraints[0]
    assert item.source_chain == [
        ConstraintSource.INDUSTRY_DEFAULT,
        ConstraintSource.REQUEST,
        ConstraintSource.USER_OVERRIDE,
    ]
    assert item.user_override_widened_bounds is True


def test_warning_order_contains_widening() -> None:
    outcome = ConstraintResolver(
        policy=ConstraintResolutionPolicy(allow_user_bound_widening=True)
    ).resolve(
        _request(
            factors=[_factor("pressure"), _factor("temperature")],
            current_values={"pressure": 50.0, "temperature": 80.0},
            constraints=[
                _constraint("pressure", minimum=20.0, maximum=80.0),
                _constraint("temperature"),
            ],
        ),
        safety_decision=_decision(
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
            global_reason_codes=[RecommendationReasonCode.PARTIAL_ELIGIBILITY],
        ),
        user_overrides=[_constraint("pressure", minimum=0.0, maximum=100.0)],
    )
    warnings = outcome.report.warnings
    assert warnings[0].startswith("Safety decision status is CAUTION")
    assert any("widen" in w.lower() for w in warnings)


# --- Current and span ---


def test_current_inside_bounds() -> None:
    outcome = ConstraintResolver().resolve(
        _request(
            factors=[_factor("pressure")],
            current_values={"pressure": 50.0},
            constraints=[_constraint("pressure")],
        ),
        safety_decision=_decision(assessments=[_assessment("pressure")]),
    )
    assert outcome.report.status is ConstraintResolutionStatus.READY


def test_lower_boundary_allowed() -> None:
    outcome = ConstraintResolver().resolve(
        _request(
            factors=[_factor("pressure")],
            current_values={"pressure": 0.0},
            constraints=[_constraint("pressure")],
        ),
        safety_decision=_decision(
            assessments=[_assessment("pressure", current_value=0.0)]
        ),
    )
    item = outcome.report.resolved_constraints[0]
    assert item.lower_room == pytest.approx(0.0)


def test_upper_boundary_allowed() -> None:
    outcome = ConstraintResolver().resolve(
        _request(
            factors=[_factor("pressure")],
            current_values={"pressure": 100.0},
            constraints=[_constraint("pressure")],
        ),
        safety_decision=_decision(
            assessments=[_assessment("pressure", current_value=100.0)]
        ),
    )
    item = outcome.report.resolved_constraints[0]
    assert item.upper_room == pytest.approx(0.0)


def test_current_outside_rejected_no_clamp() -> None:
    outcome = ConstraintResolver().resolve(
        _request(
            factors=[_factor("pressure")],
            current_values={"pressure": 150.0},
            constraints=[_constraint("pressure")],
        ),
        safety_decision=_decision(
            assessments=[_assessment("pressure", current_value=150.0)]
        ),
    )
    assert outcome.report.resolved_variables == []
    assert any(i.code == "CURRENT_VALUE_OUTSIDE_BOUNDS" for i in outcome.report.issues)


def test_span_below_minimum_rejected() -> None:
    resolver = ConstraintResolver(
        policy=ConstraintResolutionPolicy(minimum_effective_span=10.0)
    )
    outcome = resolver.resolve(
        _request(
            factors=[_factor("pressure")],
            current_values={"pressure": 50.0},
            constraints=[_constraint("pressure", minimum=49.0, maximum=51.0)],
        ),
        safety_decision=_decision(assessments=[_assessment("pressure")]),
    )
    assert any(i.code == "EFFECTIVE_SPAN_TOO_SMALL" for i in outcome.report.issues)


def test_zero_width_rejected() -> None:
    outcome = ConstraintResolver(
        policy=ConstraintResolutionPolicy(minimum_effective_span=0.0)
    ).resolve(
        _request(
            factors=[_factor("pressure")],
            current_values={"pressure": 50.0},
            constraints=[_constraint("pressure", minimum=50.0, maximum=50.0)],
        ),
        safety_decision=_decision(assessments=[_assessment("pressure")]),
    )
    assert any(i.code == "EFFECTIVE_SPAN_TOO_SMALL" for i in outcome.report.issues)


def test_room_and_relative_position() -> None:
    outcome = ConstraintResolver().resolve(
        _request(
            factors=[_factor("pressure")],
            current_values={"pressure": 25.0},
            constraints=[_constraint("pressure", minimum=0.0, maximum=100.0)],
        ),
        safety_decision=_decision(
            assessments=[_assessment("pressure", current_value=25.0)]
        ),
    )
    item = outcome.report.resolved_constraints[0]
    assert item.lower_room == pytest.approx(25.0)
    assert item.upper_room == pytest.approx(75.0)
    assert item.relative_position == pytest.approx(0.25)


# --- Result status ---


def test_all_resolved_ready() -> None:
    outcome = ConstraintResolver().resolve(_request(), safety_decision=_decision())
    assert outcome.report.status is ConstraintResolutionStatus.READY


def test_partial_resolution() -> None:
    request = _request(
        factors=[_factor("pressure"), _factor("temperature")],
        current_values={"pressure": 50.0, "temperature": 80.0},
        constraints=[_constraint("pressure")],
    )
    decision = _decision()
    outcome = ConstraintResolver().resolve(request, safety_decision=decision)
    assert outcome.report.status is ConstraintResolutionStatus.PARTIAL
    assert outcome.report.resolved_variables == ["pressure"]
    assert outcome.report.unresolved_variables == ["temperature"]


def test_partial_disabled_refused() -> None:
    request = _request(
        factors=[_factor("pressure"), _factor("temperature")],
        current_values={"pressure": 50.0, "temperature": 80.0},
        constraints=[_constraint("pressure")],
    )
    outcome = ConstraintResolver(
        policy=ConstraintResolutionPolicy(allow_partial_resolution=False)
    ).resolve(request, safety_decision=_decision())
    assert outcome.report.status is ConstraintResolutionStatus.REFUSED
    assert outcome.report.resolved_variables == []


def test_zero_resolved_refused() -> None:
    request = _request(
        factors=[_factor("pressure")],
        current_values={"pressure": 50.0},
        constraints=[],
    )
    outcome = ConstraintResolver().resolve(
        request,
        safety_decision=_decision(assessments=[_assessment("pressure")]),
    )
    assert outcome.report.status is ConstraintResolutionStatus.REFUSED


def test_resolved_and_unresolved_order() -> None:
    request = _request(
        factors=[_factor("pressure"), _factor("temperature")],
        current_values={"pressure": 50.0, "temperature": 80.0},
        constraints=[_constraint("temperature")],
    )
    outcome = ConstraintResolver().resolve(request, safety_decision=_decision())
    assert outcome.report.resolved_variables == ["temperature"]
    assert outcome.report.unresolved_variables == ["pressure"]


def test_issue_order_deterministic() -> None:
    request = _request(
        factors=[_factor("pressure"), _factor("temperature")],
        current_values={"pressure": 50.0, "temperature": 200.0},
        constraints=[_constraint("temperature")],
    )
    outcome = ConstraintResolver().resolve(request, safety_decision=_decision())
    codes = [i.code for i in outcome.report.issues]
    assert "CONSTRAINT_MISSING" in codes
    assert "CURRENT_VALUE_OUTSIDE_BOUNDS" in codes
    assert codes.index("CONSTRAINT_MISSING") < codes.index(
        "CURRENT_VALUE_OUTSIDE_BOUNDS"
    )


def test_warnings_deduplicated_and_metadata() -> None:
    outcome = ConstraintResolver().resolve(_request(), safety_decision=_decision())
    assert len(outcome.report.warnings) == len(set(outcome.report.warnings))
    assert outcome.report.metadata["optimization_performed"] is False
    assert outcome.report.metadata["candidate_values_generated"] is False
    assert outcome.report.metadata["resolved_count"] == 2


# --- State ---


def test_no_cache_across_calls() -> None:
    resolver = ConstraintResolver()
    first = resolver.resolve(_request(), safety_decision=_decision())
    second = resolver.resolve(
        _request(
            factors=[_factor("pressure")],
            current_values={"pressure": 50.0},
            constraints=[_constraint("pressure")],
        ),
        safety_decision=_decision(assessments=[_assessment("pressure")]),
    )
    assert first.report.resolved_variables == ["pressure", "temperature"]
    assert second.report.resolved_variables == ["pressure"]


def test_deterministic_same_input() -> None:
    resolver = ConstraintResolver()
    request = _request()
    decision = _decision()
    a = resolver.resolve(request, safety_decision=decision)
    b = resolver.resolve(request, safety_decision=decision)
    assert a.report.resolved_variables == b.report.resolved_variables
    assert a.report.status == b.report.status
    assert [
        c.model_dump(exclude={"warnings"}) for c in a.report.resolved_constraints
    ] == [
        c.model_dump(exclude={"warnings"}) for c in b.report.resolved_constraints
    ]


def test_returned_report_mutation_isolated() -> None:
    outcome = ConstraintResolver().resolve(_request(), safety_decision=_decision())
    outcome.report.resolved_variables.append("hacked")
    again = ConstraintResolver().resolve(_request(), safety_decision=_decision())
    assert "hacked" not in again.report.resolved_variables
