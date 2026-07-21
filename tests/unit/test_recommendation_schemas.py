"""Unit tests for recommendation enums and Pydantic schemas (Step 9A)."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from process_intelligence.core.enums import AnalysisTask, ColumnRole
from process_intelligence.core.schemas import RootCauseFactor, VariableConstraint
from process_intelligence.diagnosis.enums import DiagnosisMethod, DiagnosisScope
from process_intelligence.diagnosis.schemas import DiagnosisBatchResult, DiagnosisResult
from process_intelligence.evaluation.leakage import LeakageReport
from process_intelligence.recommendation import (
    BaseRecommendationEngine,
    RecommendationChange,
    RecommendationObjective,
    RecommendationOutcome,
    RecommendationReasonCode,
    RecommendationRequest,
    RecommendationResult,
    RecommendationSafetyContext,
    RecommendationSafetyDecision,
    RecommendationSafetyGate,
    RecommendationSafetyPolicy,
    RecommendationSafetyStatus,
    RecommendationStatus,
    VariableEligibilityAssessment,
)
from process_intelligence.recommendation.schemas import DEFAULT_RECOMMENDATION_DISCLAIMER


def _factor(
    variable: str = "pressure",
    *,
    role: ColumnRole = ColumnRole.CONTROLLABLE_PROCESS,
    controllable: bool = True,
    confidence: float = 0.7,
    needs_verification: bool = False,
) -> RootCauseFactor:
    return RootCauseFactor(
        variable=variable,
        direction="increase",
        deviation=1.5,
        role=role,
        controllable=controllable,
        evidence="Associated with higher anomaly score; association only.",
        confidence=confidence,
        needs_verification=needs_verification,
    )


def _constraint(
    variable: str = "pressure",
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


def _diagnosis(**overrides: Any) -> DiagnosisResult:
    payload: dict[str, Any] = {
        "anomaly_id": "a-1",
        "task": AnalysisTask.UNSUPERVISED_ANOMALY,
        "method_used": [DiagnosisMethod.GROUP_COMPARISON],
        "scope": DiagnosisScope.SINGLE_EVENT,
        "factors": [_factor("pressure"), _factor("temperature")],
        "confidence": 0.75,
        "analyzed_row_count": 3,
        "reference_row_count": 10,
        "caveats": ["Association only; causation is not established."],
        "generated_at": datetime(2026, 7, 21, 9, 0, tzinfo=UTC),
        "metadata": {"version": 1},
    }
    payload.update(overrides)
    return DiagnosisResult(**payload)


def _safe_leakage() -> LeakageReport:
    return LeakageReport(
        is_safe=True,
        issues=[],
        blocker_count=0,
        warning_count=0,
        checked_feature_columns=["pressure"],
        checked_preprocessing_event_count=0,
    )


def _assessment(
    variable: str = "pressure",
    *,
    eligible: bool = True,
    reason_codes: list[RecommendationReasonCode] | None = None,
    factor_rank: int = 1,
    **overrides: Any,
) -> VariableEligibilityAssessment:
    payload: dict[str, Any] = {
        "variable": variable,
        "factor_rank": factor_rank,
        "factor_confidence": 0.7,
        "factor_role": str(ColumnRole.CONTROLLABLE_PROCESS),
        "factor_controllable": True,
        "factor_needs_verification": False,
        "current_value": 50.0,
        "constraint_present": True,
        "user_confirmed_controllable": True,
        "user_verified": False,
        "eligible": eligible,
        "reason_codes": list(reason_codes or []),
        "warnings": [],
    }
    payload.update(overrides)
    return VariableEligibilityAssessment(**payload)


def _decision(**overrides: Any) -> RecommendationSafetyDecision:
    assessments = [
        _assessment("pressure", eligible=True, factor_rank=1),
    ]
    payload: dict[str, Any] = {
        "status": RecommendationSafetyStatus.APPROVED,
        "objective": RecommendationObjective.REDUCE_ANOMALY_SCORE,
        "eligible_variables": ["pressure"],
        "blocked_variables": [],
        "variable_assessments": assessments,
        "global_reason_codes": [],
        "messages": ["Safety checks passed."],
        "disclaimer": DEFAULT_RECOMMENDATION_DISCLAIMER,
        "evaluated_at": datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
        "metadata": {"eligible_variable_count": 1},
    }
    payload.update(overrides)
    return RecommendationSafetyDecision(**payload)


def _request(**overrides: Any) -> RecommendationRequest:
    payload: dict[str, Any] = {
        "task": AnalysisTask.UNSUPERVISED_ANOMALY,
        "diagnosis": _diagnosis(),
        "objective": RecommendationObjective.REDUCE_ANOMALY_SCORE,
        "current_values": {"pressure": 50.0, "temperature": 80.0},
        "constraints": [_constraint("pressure"), _constraint("temperature")],
        "user_confirmed_controllable_variables": ["pressure"],
        "user_verified_variables": [],
        "max_simultaneous_changes": 2,
        "metadata": {"seed": 1},
    }
    payload.update(overrides)
    return RecommendationRequest(**payload)


def _change(**overrides: Any) -> RecommendationChange:
    payload: dict[str, Any] = {
        "variable": "pressure",
        "current_value": 50.0,
        "proposed_value": 48.0,
        "delta": -2.0,
        "relative_delta": -0.04,
        "rationale": "Candidate change within observed support; association only.",
        "confidence": 0.6,
        "requires_verification": True,
    }
    payload.update(overrides)
    return RecommendationChange(**payload)


# --- enums ---


def test_recommendation_objective_members() -> None:
    expected = {
        "REDUCE_ANOMALY_SCORE": "REDUCE_ANOMALY_SCORE",
        "IMPROVE_PREDICTED_QUALITY": "IMPROVE_PREDICTED_QUALITY",
        "BALANCE_QUALITY_AND_ANOMALY": "BALANCE_QUALITY_AND_ANOMALY",
    }
    assert {m.name: m.value for m in RecommendationObjective} == expected
    assert len(RecommendationObjective) == 3
    assert RecommendationObjective.__doc__
    assert RecommendationObjective("REDUCE_ANOMALY_SCORE") is (
        RecommendationObjective.REDUCE_ANOMALY_SCORE
    )
    with pytest.raises(ValueError):
        RecommendationObjective("INVALID")


def test_recommendation_safety_status_members() -> None:
    expected = {
        "APPROVED": "APPROVED",
        "CAUTION": "CAUTION",
        "REFUSED": "REFUSED",
    }
    assert {m.name: m.value for m in RecommendationSafetyStatus} == expected
    assert len(RecommendationSafetyStatus) == 3
    assert RecommendationSafetyStatus.__doc__


def test_recommendation_status_members() -> None:
    expected = {
        "REFUSED": "REFUSED",
        "READY_FOR_OPTIMIZATION": "READY_FOR_OPTIMIZATION",
        "GENERATED": "GENERATED",
    }
    assert {m.name: m.value for m in RecommendationStatus} == expected
    assert len(RecommendationStatus) == 3
    assert RecommendationStatus.__doc__


def test_recommendation_reason_code_members() -> None:
    expected = {
        "LEAKAGE_BLOCKER",
        "FINAL_EVALUATION_MISSING",
        "MODEL_PERFORMANCE_UNACCEPTABLE",
        "DIAGNOSIS_CONFIDENCE_TOO_LOW",
        "REFERENCE_SAMPLE_TOO_SMALL",
        "EXTRAPOLATION_RISK",
        "UNCERTAINTY_UNAVAILABLE",
        "UNCERTAINTY_UNACCEPTABLE",
        "CURRENT_VALUE_MISSING",
        "CONSTRAINT_MISSING",
        "NON_CONTROLLABLE_VARIABLE",
        "USER_CONFIRMATION_REQUIRED",
        "VERIFICATION_REQUIRED",
        "CURRENT_VALUE_OUTSIDE_CONSTRAINT",
        "CANDIDATE_LIMIT_EXCEEDED",
        "NO_ELIGIBLE_VARIABLES",
        "PARTIAL_ELIGIBILITY",
    }
    assert {m.name for m in RecommendationReasonCode} == expected
    assert len(RecommendationReasonCode) == 17
    assert RecommendationReasonCode.__doc__
    assert RecommendationReasonCode("LEAKAGE_BLOCKER") is (
        RecommendationReasonCode.LEAKAGE_BLOCKER
    )
    with pytest.raises(ValueError):
        RecommendationReasonCode("NOT_A_CODE")


# --- policy ---


def test_policy_defaults() -> None:
    policy = RecommendationSafetyPolicy()
    assert policy.require_safe_leakage_report is True
    assert policy.minimum_diagnosis_confidence == 0.20
    assert policy.minimum_reference_rows == 5
    assert policy.maximum_candidate_variables == 10
    assert policy.require_uncertainty is False
    assert policy.allow_partial_eligibility is True


def test_policy_bool_strict() -> None:
    with pytest.raises(ValidationError):
        RecommendationSafetyPolicy(require_safe_leakage_report=1)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        RecommendationSafetyPolicy(require_constraints="true")  # type: ignore[arg-type]


def test_policy_confidence_range_and_bool() -> None:
    with pytest.raises(ValidationError):
        RecommendationSafetyPolicy(minimum_diagnosis_confidence=1.5)
    with pytest.raises(ValidationError):
        RecommendationSafetyPolicy(minimum_diagnosis_confidence=True)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        RecommendationSafetyPolicy(minimum_diagnosis_confidence=math.nan)


def test_policy_row_and_candidate_counts() -> None:
    with pytest.raises(ValidationError):
        RecommendationSafetyPolicy(minimum_reference_rows=0)
    with pytest.raises(ValidationError):
        RecommendationSafetyPolicy(maximum_candidate_variables=0)
    with pytest.raises(ValidationError):
        RecommendationSafetyPolicy(minimum_reference_rows=True)  # type: ignore[arg-type]


def test_policy_uncertainty_relationship() -> None:
    with pytest.raises(ValidationError):
        RecommendationSafetyPolicy(
            require_uncertainty=False,
            require_acceptable_uncertainty=True,
        )
    ok = RecommendationSafetyPolicy(
        require_uncertainty=True,
        require_acceptable_uncertainty=True,
    )
    assert ok.require_acceptable_uncertainty is True


def test_policy_round_trip() -> None:
    policy = RecommendationSafetyPolicy(maximum_candidate_variables=3)
    restored = RecommendationSafetyPolicy.model_validate(policy.model_dump())
    assert restored == policy


# --- request ---


def test_request_valid() -> None:
    request = _request()
    assert request.task is AnalysisTask.UNSUPERVISED_ANOMALY
    assert request.diagnosis.confidence == 0.75
    assert request.current_values["pressure"] == 50.0


def test_request_task_diagnosis_mismatch() -> None:
    with pytest.raises(ValidationError):
        _request(task=AnalysisTask.REGRESSION)


def test_request_rejects_batch_result() -> None:
    batch = DiagnosisBatchResult(
        results=[_diagnosis()],
        aggregate_factors=[],
        requested_event_count=1,
        diagnosed_event_count=1,
        failed_event_count=0,
        generated_at=datetime(2026, 7, 21, 9, 0, tzinfo=UTC),
    )
    with pytest.raises(ValidationError):
        _request(diagnosis=batch)  # type: ignore[arg-type]


def test_request_empty_current_values() -> None:
    with pytest.raises(ValidationError):
        _request(current_values={})


def test_request_current_key_blank() -> None:
    with pytest.raises(ValidationError):
        _request(current_values={"  ": 1.0})


def test_request_current_value_bool_nan_inf() -> None:
    with pytest.raises(ValidationError):
        _request(current_values={"pressure": True})  # type: ignore[dict-item]
    with pytest.raises(ValidationError):
        _request(current_values={"pressure": math.nan})
    with pytest.raises(ValidationError):
        _request(current_values={"pressure": math.inf})


def test_request_duplicate_constraint_variable() -> None:
    with pytest.raises(ValidationError):
        _request(constraints=[_constraint("pressure"), _constraint("pressure")])


def test_request_confirmation_duplicate_and_missing() -> None:
    with pytest.raises(ValidationError):
        _request(user_confirmed_controllable_variables=["pressure", "pressure"])
    with pytest.raises(ValidationError):
        _request(user_confirmed_controllable_variables=["missing_var"])


def test_request_verified_duplicate() -> None:
    with pytest.raises(ValidationError):
        _request(user_verified_variables=["pressure", "pressure"])


def test_request_max_simultaneous_rules() -> None:
    with pytest.raises(ValidationError):
        _request(max_simultaneous_changes=0)
    with pytest.raises(ValidationError):
        _request(max_simultaneous_changes=True)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        _request(max_simultaneous_changes=3)  # only 2 current values


def test_request_metadata_scalar_and_independence() -> None:
    with pytest.raises(ValidationError):
        _request(metadata={"frame": {"a": 1}})  # type: ignore[dict-item]
    values = {"pressure": 10.0, "temperature": 20.0}
    confirmed = ["pressure"]
    request = _request(
        current_values=values,
        user_confirmed_controllable_variables=confirmed,
    )
    values["pressure"] = 999.0
    confirmed.append("temperature")
    assert request.current_values["pressure"] == 10.0
    assert request.user_confirmed_controllable_variables == ["pressure"]


def test_request_round_trip() -> None:
    request = _request()
    restored = RecommendationRequest.model_validate(request.model_dump())
    assert restored.task == request.task
    assert restored.current_values == request.current_values


def test_request_rejects_original_row_id() -> None:
    with pytest.raises(ValidationError):
        _request(current_values={"_original_row_id": 1.0, "pressure": 1.0})


# --- context ---


def test_context_valid() -> None:
    context = RecommendationSafetyContext(
        leakage_report=_safe_leakage(),
        final_evaluation_available=True,
        model_performance_acceptable=True,
    )
    assert context.extrapolation_detected is False
    assert context.uncertainty_acceptable is None


def test_context_leakage_type_and_bool_strict() -> None:
    with pytest.raises(ValidationError):
        RecommendationSafetyContext(
            leakage_report={"is_safe": True},  # type: ignore[arg-type]
            final_evaluation_available=True,
            model_performance_acceptable=True,
        )
    with pytest.raises(ValidationError):
        RecommendationSafetyContext(
            leakage_report=_safe_leakage(),
            final_evaluation_available=1,  # type: ignore[arg-type]
            model_performance_acceptable=True,
        )


def test_context_performance_reason_and_uncertainty() -> None:
    with pytest.raises(ValidationError):
        RecommendationSafetyContext(
            leakage_report=_safe_leakage(),
            final_evaluation_available=True,
            model_performance_acceptable=False,
            model_performance_reason="   ",
        )
    with pytest.raises(ValidationError):
        RecommendationSafetyContext(
            leakage_report=_safe_leakage(),
            final_evaluation_available=True,
            model_performance_acceptable=True,
            uncertainty_available=False,
            uncertainty_acceptable=False,
        )
    ok = RecommendationSafetyContext(
        leakage_report=_safe_leakage(),
        final_evaluation_available=True,
        model_performance_acceptable=True,
        uncertainty_available=True,
        uncertainty_acceptable=False,
    )
    assert ok.uncertainty_acceptable is False


def test_context_metadata_and_round_trip() -> None:
    with pytest.raises(ValidationError):
        RecommendationSafetyContext(
            leakage_report=_safe_leakage(),
            final_evaluation_available=True,
            model_performance_acceptable=True,
            metadata={"arr": [1, 2]},  # type: ignore[dict-item]
        )
    context = RecommendationSafetyContext(
        leakage_report=_safe_leakage(),
        final_evaluation_available=True,
        model_performance_acceptable=True,
        metadata={"ok": 1},
    )
    restored = RecommendationSafetyContext.model_validate(context.model_dump())
    assert restored.final_evaluation_available is True


# --- assessment ---


def test_assessment_eligible_and_blocked() -> None:
    ok = _assessment(eligible=True)
    assert ok.eligible is True
    blocked = _assessment(
        eligible=False,
        reason_codes=[RecommendationReasonCode.CURRENT_VALUE_MISSING],
        current_value=None,
    )
    assert blocked.eligible is False


def test_assessment_validation_rules() -> None:
    with pytest.raises(ValidationError):
        _assessment(variable="  ")
    with pytest.raises(ValidationError):
        _assessment(factor_rank=0)
    with pytest.raises(ValidationError):
        _assessment(factor_rank=True)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        _assessment(factor_confidence=1.2)
    with pytest.raises(ValidationError):
        _assessment(current_value=math.nan)
    with pytest.raises(ValidationError):
        _assessment(eligible=1)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        _assessment(
            eligible=False,
            reason_codes=[
                RecommendationReasonCode.CURRENT_VALUE_MISSING,
                RecommendationReasonCode.CURRENT_VALUE_MISSING,
            ],
        )
    with pytest.raises(ValidationError):
        _assessment(eligible=True, warnings=["a", "a"])
    with pytest.raises(ValidationError):
        _assessment(
            eligible=True,
            reason_codes=[RecommendationReasonCode.CONSTRAINT_MISSING],
        )


def test_assessment_round_trip() -> None:
    assessment = _assessment()
    restored = VariableEligibilityAssessment.model_validate(assessment.model_dump())
    assert restored == assessment


# --- decision ---


def test_decision_approved_caution_refused() -> None:
    approved = _decision()
    assert approved.status is RecommendationSafetyStatus.APPROVED

    caution = _decision(
        status=RecommendationSafetyStatus.CAUTION,
        eligible_variables=["pressure"],
        blocked_variables=["temperature"],
        variable_assessments=[
            _assessment("pressure", eligible=True, factor_rank=1),
            _assessment(
                "temperature",
                eligible=False,
                factor_rank=2,
                reason_codes=[RecommendationReasonCode.USER_CONFIRMATION_REQUIRED],
            ),
        ],
        global_reason_codes=[RecommendationReasonCode.PARTIAL_ELIGIBILITY],
        messages=["Partial eligibility observed."],
    )
    assert caution.status is RecommendationSafetyStatus.CAUTION

    refused = _decision(
        status=RecommendationSafetyStatus.REFUSED,
        eligible_variables=[],
        blocked_variables=["pressure"],
        variable_assessments=[
            _assessment(
                "pressure",
                eligible=False,
                reason_codes=[RecommendationReasonCode.CURRENT_VALUE_MISSING],
                current_value=None,
            )
        ],
        global_reason_codes=[RecommendationReasonCode.NO_ELIGIBLE_VARIABLES],
        messages=["No eligible variables."],
    )
    assert refused.status is RecommendationSafetyStatus.REFUSED


def test_decision_set_consistency() -> None:
    with pytest.raises(ValidationError):
        _decision(eligible_variables=["pressure", "pressure"])
    with pytest.raises(ValidationError):
        _decision(
            status=RecommendationSafetyStatus.CAUTION,
            eligible_variables=["pressure"],
            blocked_variables=["pressure"],
            variable_assessments=[
                _assessment("pressure", eligible=True),
            ],
        )
    with pytest.raises(ValidationError):
        _decision(
            variable_assessments=[
                _assessment("pressure", eligible=True, factor_rank=1),
                _assessment("pressure", eligible=False, factor_rank=2),
            ]
        )
    with pytest.raises(ValidationError):
        _decision(
            eligible_variables=["pressure"],
            blocked_variables=[],
            variable_assessments=[
                _assessment(
                    "pressure",
                    eligible=False,
                    reason_codes=[RecommendationReasonCode.CONSTRAINT_MISSING],
                )
            ],
        )


def test_decision_status_and_disclaimer_datetime() -> None:
    with pytest.raises(ValidationError):
        _decision(
            status=RecommendationSafetyStatus.APPROVED,
            eligible_variables=[],
            blocked_variables=[],
            variable_assessments=[],
            global_reason_codes=[RecommendationReasonCode.NO_ELIGIBLE_VARIABLES],
        )
    with pytest.raises(ValidationError):
        _decision(disclaimer="Missing required meanings.")
    with pytest.raises(ValidationError):
        _decision(evaluated_at=datetime(2026, 7, 21, 12, 0))
    with pytest.raises(ValidationError):
        _decision(metadata={"model": object()})  # type: ignore[dict-item]


def test_decision_round_trip() -> None:
    decision = _decision()
    restored = RecommendationSafetyDecision.model_validate(decision.model_dump())
    assert restored.status == decision.status


# --- change ---


def test_change_valid_and_delta() -> None:
    change = _change()
    assert change.delta == -2.0
    with pytest.raises(ValidationError):
        _change(delta=-1.0)
    with pytest.raises(ValidationError):
        _change(current_value=math.nan)
    with pytest.raises(ValidationError):
        _change(rationale="This is a proven root cause fix.")
    with pytest.raises(ValidationError):
        _change(confidence=1.5)
    with pytest.raises(ValidationError):
        _change(requires_verification=1)  # type: ignore[arg-type]


def test_change_round_trip() -> None:
    change = _change()
    restored = RecommendationChange.model_validate(change.model_dump())
    assert restored == change


# --- result / outcome ---


def test_result_refused_ready_generated() -> None:
    refused_decision = _decision(
        status=RecommendationSafetyStatus.REFUSED,
        eligible_variables=[],
        blocked_variables=["pressure"],
        variable_assessments=[
            _assessment(
                "pressure",
                eligible=False,
                reason_codes=[RecommendationReasonCode.NO_ELIGIBLE_VARIABLES],
            )
        ],
        global_reason_codes=[RecommendationReasonCode.NO_ELIGIBLE_VARIABLES],
        messages=["Refused."],
    )
    # NO_ELIGIBLE_VARIABLES is global, not variable blocking - fix assessment
    refused_decision = _decision(
        status=RecommendationSafetyStatus.REFUSED,
        eligible_variables=[],
        blocked_variables=["pressure"],
        variable_assessments=[
            _assessment(
                "pressure",
                eligible=False,
                reason_codes=[RecommendationReasonCode.CURRENT_VALUE_MISSING],
                current_value=None,
            )
        ],
        global_reason_codes=[RecommendationReasonCode.NO_ELIGIBLE_VARIABLES],
        messages=["Refused."],
    )
    refused = RecommendationResult(
        status=RecommendationStatus.REFUSED,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        safety_decision=refused_decision,
        changes=[],
        confidence=0.0,
        extrapolation_flag=False,
        uncertainty_available=False,
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
    )
    assert refused.changes == []

    ready = RecommendationResult(
        status=RecommendationStatus.READY_FOR_OPTIMIZATION,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        safety_decision=_decision(),
        changes=[],
        confidence=0.5,
        extrapolation_flag=False,
        uncertainty_available=False,
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
    )
    assert ready.status is RecommendationStatus.READY_FOR_OPTIMIZATION

    generated = RecommendationResult(
        status=RecommendationStatus.GENERATED,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        safety_decision=_decision(),
        changes=[_change()],
        confidence=0.5,
        extrapolation_flag=False,
        uncertainty_available=False,
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
        warnings=["Verify operationally before applying."],
    )
    assert len(generated.changes) == 1


def test_result_status_rules() -> None:
    with pytest.raises(ValidationError):
        RecommendationResult(
            status=RecommendationStatus.REFUSED,
            objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
            safety_decision=_decision(),
            changes=[],
            confidence=0.0,
            extrapolation_flag=False,
            uncertainty_available=False,
            disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
            generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
        )
    with pytest.raises(ValidationError):
        RecommendationResult(
            status=RecommendationStatus.GENERATED,
            objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
            safety_decision=_decision(),
            changes=[_change(variable="temperature")],
            confidence=0.5,
            extrapolation_flag=False,
            uncertainty_available=False,
            disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
            generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
        )
    with pytest.raises(ValidationError):
        RecommendationResult(
            status=RecommendationStatus.GENERATED,
            objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
            safety_decision=_decision(),
            changes=[_change(), _change()],
            confidence=0.5,
            extrapolation_flag=False,
            uncertainty_available=False,
            disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
            generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
        )


def test_result_time_warning_metadata_and_outcome() -> None:
    with pytest.raises(ValidationError):
        RecommendationResult(
            status=RecommendationStatus.READY_FOR_OPTIMIZATION,
            objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
            safety_decision=_decision(),
            changes=[],
            confidence=0.5,
            extrapolation_flag=False,
            uncertainty_available=False,
            disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
            generated_at=datetime(2026, 7, 21, 13, 0),
        )
    with pytest.raises(ValidationError):
        RecommendationResult(
            status=RecommendationStatus.READY_FOR_OPTIMIZATION,
            objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
            safety_decision=_decision(),
            changes=[],
            confidence=0.5,
            extrapolation_flag=False,
            uncertainty_available=False,
            disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
            generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
            warnings=["a", "a"],
        )
    result = RecommendationResult(
        status=RecommendationStatus.READY_FOR_OPTIMIZATION,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        safety_decision=_decision(),
        changes=[],
        confidence=0.5,
        extrapolation_flag=False,
        uncertainty_available=False,
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
        metadata={"ready": True},
    )
    outcome = RecommendationOutcome(result=result)
    assert outcome.result.status is RecommendationStatus.READY_FOR_OPTIMIZATION
    with pytest.raises(AttributeError):
        outcome.result = result  # type: ignore[misc]


def _refused_decision() -> RecommendationSafetyDecision:
    return _decision(
        status=RecommendationSafetyStatus.REFUSED,
        eligible_variables=[],
        blocked_variables=["pressure"],
        variable_assessments=[
            _assessment(
                "pressure",
                eligible=False,
                reason_codes=[RecommendationReasonCode.CURRENT_VALUE_MISSING],
                current_value=None,
            )
        ],
        global_reason_codes=[RecommendationReasonCode.NO_ELIGIBLE_VARIABLES],
        messages=["Refused."],
    )


def test_refused_allows_negative_baseline_anomaly_score() -> None:
    result = RecommendationResult(
        status=RecommendationStatus.REFUSED,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        safety_decision=_refused_decision(),
        changes=[],
        baseline_anomaly_score=-0.2,
        proposed_anomaly_score=None,
        confidence=0.0,
        extrapolation_flag=False,
        uncertainty_available=False,
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
    )
    assert result.baseline_anomaly_score == pytest.approx(-0.2)
    assert result.proposed_anomaly_score is None


def test_ready_allows_negative_baseline_anomaly_score() -> None:
    result = RecommendationResult(
        status=RecommendationStatus.READY_FOR_OPTIMIZATION,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        safety_decision=_decision(),
        changes=[],
        baseline_anomaly_score=-0.35,
        proposed_anomaly_score=None,
        confidence=0.5,
        extrapolation_flag=False,
        uncertainty_available=False,
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
    )
    assert result.baseline_anomaly_score == pytest.approx(-0.35)
    assert result.proposed_anomaly_score is None


def test_generated_allows_negative_baseline_anomaly_score() -> None:
    result = RecommendationResult(
        status=RecommendationStatus.GENERATED,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        safety_decision=_decision(),
        changes=[_change()],
        baseline_anomaly_score=-0.2,
        proposed_anomaly_score=0.1,
        confidence=0.5,
        extrapolation_flag=False,
        uncertainty_available=False,
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
    )
    assert result.baseline_anomaly_score == pytest.approx(-0.2)


def test_generated_allows_negative_proposed_anomaly_score() -> None:
    result = RecommendationResult(
        status=RecommendationStatus.GENERATED,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        safety_decision=_decision(),
        changes=[_change()],
        baseline_anomaly_score=-0.2,
        proposed_anomaly_score=-0.8,
        confidence=0.5,
        extrapolation_flag=False,
        uncertainty_available=False,
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
    )
    assert result.proposed_anomaly_score == pytest.approx(-0.8)
    assert result.baseline_anomaly_score - result.proposed_anomaly_score == pytest.approx(
        0.6
    )


def test_generated_allows_negative_baseline_positive_proposed() -> None:
    result = RecommendationResult(
        status=RecommendationStatus.GENERATED,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        safety_decision=_decision(),
        changes=[_change()],
        baseline_anomaly_score=-0.1,
        proposed_anomaly_score=0.4,
        confidence=0.5,
        extrapolation_flag=False,
        uncertainty_available=False,
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
    )
    assert result.baseline_anomaly_score == pytest.approx(-0.1)
    assert result.proposed_anomaly_score == pytest.approx(0.4)


def test_generated_allows_positive_baseline_negative_proposed() -> None:
    result = RecommendationResult(
        status=RecommendationStatus.GENERATED,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        safety_decision=_decision(),
        changes=[_change()],
        baseline_anomaly_score=0.3,
        proposed_anomaly_score=-0.5,
        confidence=0.5,
        extrapolation_flag=False,
        uncertainty_available=False,
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
    )
    assert result.baseline_anomaly_score == pytest.approx(0.3)
    assert result.proposed_anomaly_score == pytest.approx(-0.5)


def test_result_allows_zero_anomaly_scores() -> None:
    result = RecommendationResult(
        status=RecommendationStatus.GENERATED,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        safety_decision=_decision(),
        changes=[_change()],
        baseline_anomaly_score=0.0,
        proposed_anomaly_score=0.0,
        confidence=0.5,
        extrapolation_flag=False,
        uncertainty_available=False,
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
    )
    assert result.baseline_anomaly_score == pytest.approx(0.0)
    assert result.proposed_anomaly_score == pytest.approx(0.0)


@pytest.mark.parametrize("field", ["baseline_anomaly_score", "proposed_anomaly_score"])
def test_result_anomaly_score_bool_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        RecommendationResult(
            status=RecommendationStatus.GENERATED,
            objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
            safety_decision=_decision(),
            changes=[_change()],
            confidence=0.5,
            extrapolation_flag=False,
            uncertainty_available=False,
            disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
            generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
            **{field: True},
        )


def test_result_baseline_anomaly_nan_rejected() -> None:
    with pytest.raises(ValidationError):
        RecommendationResult(
            status=RecommendationStatus.GENERATED,
            objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
            safety_decision=_decision(),
            changes=[_change()],
            baseline_anomaly_score=float("nan"),
            proposed_anomaly_score=-0.1,
            confidence=0.5,
            extrapolation_flag=False,
            uncertainty_available=False,
            disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
            generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
        )


def test_result_proposed_anomaly_nan_rejected() -> None:
    with pytest.raises(ValidationError):
        RecommendationResult(
            status=RecommendationStatus.GENERATED,
            objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
            safety_decision=_decision(),
            changes=[_change()],
            baseline_anomaly_score=-0.1,
            proposed_anomaly_score=float("nan"),
            confidence=0.5,
            extrapolation_flag=False,
            uncertainty_available=False,
            disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
            generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
        )


def test_result_baseline_anomaly_pos_inf_rejected() -> None:
    with pytest.raises(ValidationError):
        RecommendationResult(
            status=RecommendationStatus.GENERATED,
            objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
            safety_decision=_decision(),
            changes=[_change()],
            baseline_anomaly_score=float("inf"),
            proposed_anomaly_score=-0.1,
            confidence=0.5,
            extrapolation_flag=False,
            uncertainty_available=False,
            disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
            generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
        )


def test_result_proposed_anomaly_neg_inf_rejected() -> None:
    with pytest.raises(ValidationError):
        RecommendationResult(
            status=RecommendationStatus.GENERATED,
            objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
            safety_decision=_decision(),
            changes=[_change()],
            baseline_anomaly_score=-0.1,
            proposed_anomaly_score=float("-inf"),
            confidence=0.5,
            extrapolation_flag=False,
            uncertainty_available=False,
            disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
            generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
        )


@pytest.mark.parametrize("field", ["baseline_anomaly_score", "proposed_anomaly_score"])
def test_result_anomaly_score_string_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        RecommendationResult(
            status=RecommendationStatus.GENERATED,
            objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
            safety_decision=_decision(),
            changes=[_change()],
            confidence=0.5,
            extrapolation_flag=False,
            uncertainty_available=False,
            disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
            generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
            **{field: "-0.2"},
        )


def test_result_negative_anomaly_score_round_trip() -> None:
    result = RecommendationResult(
        status=RecommendationStatus.GENERATED,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        safety_decision=_decision(),
        changes=[_change()],
        baseline_anomaly_score=-0.2,
        proposed_anomaly_score=-0.8,
        confidence=0.5,
        extrapolation_flag=False,
        uncertainty_available=False,
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
        metadata={"seed": 1},
    )
    restored = RecommendationResult.model_validate(result.model_dump())
    assert restored.baseline_anomaly_score == pytest.approx(-0.2)
    assert restored.proposed_anomaly_score == pytest.approx(-0.8)
    assert restored.status is RecommendationStatus.GENERATED
    assert restored.confidence == pytest.approx(0.5)
    assert math.isclose(restored.baseline_anomaly_score, -0.2)
    assert math.isclose(restored.proposed_anomaly_score, -0.8)


def test_result_status_rules_unchanged_with_negative_scores() -> None:
    # REFUSED still forbids proposed_anomaly_score even when baseline is negative.
    with pytest.raises(ValidationError):
        RecommendationResult(
            status=RecommendationStatus.REFUSED,
            objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
            safety_decision=_refused_decision(),
            changes=[],
            baseline_anomaly_score=-0.2,
            proposed_anomaly_score=-0.1,
            confidence=0.0,
            extrapolation_flag=False,
            uncertainty_available=False,
            disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
            generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
        )
    # READY still forbids proposed_anomaly_score.
    with pytest.raises(ValidationError):
        RecommendationResult(
            status=RecommendationStatus.READY_FOR_OPTIMIZATION,
            objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
            safety_decision=_decision(),
            changes=[],
            baseline_anomaly_score=-0.2,
            proposed_anomaly_score=-0.1,
            confidence=0.5,
            extrapolation_flag=False,
            uncertainty_available=False,
            disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
            generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
        )
    # GENERATED still requires eligible change variables.
    with pytest.raises(ValidationError):
        RecommendationResult(
            status=RecommendationStatus.GENERATED,
            objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
            safety_decision=_decision(),
            changes=[_change(variable="temperature")],
            baseline_anomaly_score=-0.2,
            proposed_anomaly_score=-0.8,
            confidence=0.5,
            extrapolation_flag=False,
            uncertainty_available=False,
            disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
            generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
        )


def test_public_imports() -> None:
    assert BaseRecommendationEngine is not None
    assert RecommendationSafetyGate is not None
    assert RecommendationSafetyPolicy is not None
