"""Unit tests for RecommendationSafetyGate (Step 9A)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from process_intelligence.core.enums import AnalysisTask, ColumnRole
from process_intelligence.core.schemas import RootCauseFactor, VariableConstraint
from process_intelligence.diagnosis.enums import DiagnosisMethod, DiagnosisScope
from process_intelligence.diagnosis.schemas import DiagnosisResult
from process_intelligence.evaluation.leakage import (
    LeakageIssue,
    LeakageIssueType,
    LeakageReport,
    LeakageSeverity,
)
from process_intelligence.recommendation import (
    RecommendationObjective,
    RecommendationReasonCode,
    RecommendationRequest,
    RecommendationSafetyContext,
    RecommendationSafetyGate,
    RecommendationSafetyPolicy,
    RecommendationSafetyStatus,
)
from process_intelligence.recommendation.schemas import DEFAULT_RECOMMENDATION_DISCLAIMER


def _factor(
    variable: str,
    *,
    role: ColumnRole = ColumnRole.CONTROLLABLE_PROCESS,
    controllable: bool = True,
    confidence: float = 0.8,
    needs_verification: bool = False,
) -> RootCauseFactor:
    return RootCauseFactor(
        variable=variable,
        direction="increase",
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
    *,
    confidence: float = 0.8,
    reference_row_count: int = 10,
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
        confidence=confidence,
        analyzed_row_count=2,
        reference_row_count=reference_row_count,
        caveats=["Association only; causation is not established."],
        generated_at=datetime(2026, 7, 21, 9, 0, tzinfo=UTC),
    )


def _safe_leakage(*, warnings: list[LeakageIssue] | None = None) -> LeakageReport:
    issues = list(warnings or [])
    warning_count = sum(
        1 for issue in issues if issue.severity is LeakageSeverity.WARNING
    )
    blocker_count = sum(
        1 for issue in issues if issue.severity is LeakageSeverity.BLOCKER
    )
    return LeakageReport(
        is_safe=blocker_count == 0,
        issues=issues,
        blocker_count=blocker_count,
        warning_count=warning_count,
        checked_feature_columns=["pressure", "temperature"],
        checked_preprocessing_event_count=0,
    )


def _blocker_leakage() -> LeakageReport:
    return LeakageReport(
        is_safe=False,
        issues=[
            LeakageIssue(
                issue_type=LeakageIssueType.TARGET_INCLUDED_AS_FEATURE,
                severity=LeakageSeverity.BLOCKER,
                columns=["quality"],
                partitions=[],
                message="Target included as feature",
                suggested_action="Remove target from features",
            )
        ],
        blocker_count=1,
        warning_count=0,
        checked_feature_columns=["pressure"],
        checked_preprocessing_event_count=0,
    )


def _warning_leakage() -> LeakageReport:
    return _safe_leakage(
        warnings=[
            LeakageIssue(
                issue_type=LeakageIssueType.PREPROCESSING_FIT_SCOPE_UNKNOWN,
                severity=LeakageSeverity.WARNING,
                columns=["pressure"],
                partitions=["train", "validation", "test"],
                message="Fit scope unknown for impute_missing_values",
                suggested_action="Record fitted_on_training_data",
            )
        ]
    )


def _request(**overrides: Any) -> RecommendationRequest:
    factors = overrides.pop("factors", None)
    diagnosis_kwargs = overrides.pop("diagnosis_kwargs", {})
    diagnosis = overrides.pop(
        "diagnosis",
        _diagnosis(factors, **diagnosis_kwargs),
    )
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
        "user_confirmed_controllable_variables": list(
            overrides.pop(
                "user_confirmed_controllable_variables",
                list(current_values.keys()),
            )
        ),
        "user_verified_variables": list(overrides.pop("user_verified_variables", [])),
        "max_simultaneous_changes": overrides.pop(
            "max_simultaneous_changes",
            min(3, len(current_values)),
        ),
        "metadata": {},
    }
    payload.update(overrides)
    return RecommendationRequest(**payload)


def _context(**overrides: Any) -> RecommendationSafetyContext:
    payload: dict[str, Any] = {
        "leakage_report": _safe_leakage(),
        "final_evaluation_available": True,
        "model_performance_acceptable": True,
        "model_performance_reason": None,
        "extrapolation_detected": False,
        "uncertainty_available": False,
        "uncertainty_acceptable": None,
        "metadata": {},
    }
    payload.update(overrides)
    return RecommendationSafetyContext(**payload)


def _approved_policy(**overrides: Any) -> RecommendationSafetyPolicy:
    """Policy that can reach APPROVED when factors need no verification."""
    payload: dict[str, Any] = {
        "block_unverified_factors": True,
        "require_user_controllability_confirmation": True,
    }
    payload.update(overrides)
    return RecommendationSafetyPolicy(**payload)


# --- construction ---


def test_default_gate() -> None:
    gate = RecommendationSafetyGate()
    meta = gate.get_metadata()
    assert meta["require_safe_leakage_report"] is True
    assert meta["generates_recommendations"] is False
    assert meta["performs_optimization"] is False
    assert meta["association_not_causation"] is True


def test_custom_policy_and_type_error() -> None:
    policy = RecommendationSafetyPolicy(maximum_candidate_variables=2)
    gate = RecommendationSafetyGate(policy=policy)
    assert gate.get_metadata()["maximum_candidate_variables"] == 2
    with pytest.raises(TypeError):
        RecommendationSafetyGate(policy="bad")  # type: ignore[arg-type]


def test_external_policy_immutability_and_gate_isolation() -> None:
    policy = RecommendationSafetyPolicy(maximum_candidate_variables=2)
    gate_a = RecommendationSafetyGate(policy=policy)
    gate_b = RecommendationSafetyGate(policy=policy)
    mutated = policy.model_copy(update={"maximum_candidate_variables": 9})
    assert mutated.maximum_candidate_variables == 9
    assert gate_a.get_metadata()["maximum_candidate_variables"] == 2
    assert gate_b.get_metadata()["maximum_candidate_variables"] == 2
    meta_a = gate_a.get_metadata()
    meta_b = gate_b.get_metadata()
    assert meta_a is not meta_b
    meta_a["maximum_candidate_variables"] = 99
    assert gate_a.get_metadata()["maximum_candidate_variables"] == 2


def test_metadata_scalar_only_and_independence() -> None:
    gate = RecommendationSafetyGate()
    first = gate.get_metadata()
    second = gate.get_metadata()
    assert first == second
    assert first is not second
    for value in first.values():
        assert value is None or isinstance(value, (str, int, float, bool))


# --- input validation / immutability ---


def test_request_and_context_type_errors() -> None:
    gate = RecommendationSafetyGate()
    with pytest.raises(TypeError):
        gate.evaluate("bad", context=_context())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        gate.evaluate(_request(), context="bad")  # type: ignore[arg-type]


def test_input_immutability() -> None:
    gate = RecommendationSafetyGate()
    request = _request()
    context = _context()
    factor_evidence = request.diagnosis.factors[0].evidence
    constraint_min = request.constraints[0].minimum
    leakage_safe = context.leakage_report.is_safe
    decision = gate.evaluate(request, context=context)
    assert decision.status in {
        RecommendationSafetyStatus.APPROVED,
        RecommendationSafetyStatus.CAUTION,
        RecommendationSafetyStatus.REFUSED,
    }
    assert request.diagnosis.factors[0].evidence == factor_evidence
    assert request.constraints[0].minimum == constraint_min
    assert context.leakage_report.is_safe is leakage_safe


# --- global gates ---


def test_safe_leakage_passes() -> None:
    gate = RecommendationSafetyGate(policy=_approved_policy())
    decision = gate.evaluate(_request(), context=_context())
    assert RecommendationReasonCode.LEAKAGE_BLOCKER not in decision.global_reason_codes
    assert decision.status is RecommendationSafetyStatus.APPROVED


def test_leakage_warning_causes_caution() -> None:
    gate = RecommendationSafetyGate(policy=_approved_policy())
    decision = gate.evaluate(_request(), context=_context(leakage_report=_warning_leakage()))
    assert decision.status is RecommendationSafetyStatus.CAUTION
    assert RecommendationReasonCode.LEAKAGE_BLOCKER not in decision.global_reason_codes
    assert any("Leakage warning" in message for message in decision.messages)


def test_leakage_blocker_refusal() -> None:
    gate = RecommendationSafetyGate()
    decision = gate.evaluate(
        _request(),
        context=_context(leakage_report=_blocker_leakage()),
    )
    assert decision.status is RecommendationSafetyStatus.REFUSED
    assert RecommendationReasonCode.LEAKAGE_BLOCKER in decision.global_reason_codes
    assert decision.eligible_variables == []


def test_require_safe_false_allows_blocker() -> None:
    gate = RecommendationSafetyGate(
        policy=_approved_policy(require_safe_leakage_report=False)
    )
    decision = gate.evaluate(
        _request(),
        context=_context(leakage_report=_blocker_leakage()),
    )
    assert RecommendationReasonCode.LEAKAGE_BLOCKER not in decision.global_reason_codes


def test_final_evaluation_missing_and_disabled() -> None:
    gate = RecommendationSafetyGate()
    refused = gate.evaluate(
        _request(),
        context=_context(final_evaluation_available=False),
    )
    assert RecommendationReasonCode.FINAL_EVALUATION_MISSING in refused.global_reason_codes
    gate2 = RecommendationSafetyGate(
        policy=_approved_policy(require_final_evaluation=False)
    )
    allowed = gate2.evaluate(
        _request(),
        context=_context(final_evaluation_available=False),
    )
    assert (
        RecommendationReasonCode.FINAL_EVALUATION_MISSING
        not in allowed.global_reason_codes
    )


def test_model_performance_gates() -> None:
    gate = RecommendationSafetyGate()
    refused = gate.evaluate(
        _request(),
        context=_context(
            model_performance_acceptable=False,
            model_performance_reason="RMSE above threshold",
        ),
    )
    assert (
        RecommendationReasonCode.MODEL_PERFORMANCE_UNACCEPTABLE
        in refused.global_reason_codes
    )
    assert any("RMSE above threshold" in message for message in refused.messages)

    gate2 = RecommendationSafetyGate(
        policy=_approved_policy(require_acceptable_model_performance=False)
    )
    allowed = gate2.evaluate(
        _request(),
        context=_context(model_performance_acceptable=False),
    )
    assert (
        RecommendationReasonCode.MODEL_PERFORMANCE_UNACCEPTABLE
        not in allowed.global_reason_codes
    )


def test_diagnosis_confidence_and_boundary() -> None:
    gate = RecommendationSafetyGate(
        policy=_approved_policy(minimum_diagnosis_confidence=0.20)
    )
    refused = gate.evaluate(
        _request(diagnosis_kwargs={"confidence": 0.19}),
        context=_context(),
    )
    assert (
        RecommendationReasonCode.DIAGNOSIS_CONFIDENCE_TOO_LOW
        in refused.global_reason_codes
    )
    allowed = gate.evaluate(
        _request(diagnosis_kwargs={"confidence": 0.20}),
        context=_context(),
    )
    assert (
        RecommendationReasonCode.DIAGNOSIS_CONFIDENCE_TOO_LOW
        not in allowed.global_reason_codes
    )


def test_reference_rows_and_extrapolation() -> None:
    gate = RecommendationSafetyGate(
        policy=_approved_policy(minimum_reference_rows=5)
    )
    refused = gate.evaluate(
        _request(diagnosis_kwargs={"reference_row_count": 4}),
        context=_context(),
    )
    assert (
        RecommendationReasonCode.REFERENCE_SAMPLE_TOO_SMALL
        in refused.global_reason_codes
    )

    blocked = gate.evaluate(
        _request(),
        context=_context(extrapolation_detected=True),
    )
    assert RecommendationReasonCode.EXTRAPOLATION_RISK in blocked.global_reason_codes

    gate2 = RecommendationSafetyGate(
        policy=_approved_policy(block_on_extrapolation=False)
    )
    allowed = gate2.evaluate(
        _request(),
        context=_context(extrapolation_detected=True),
    )
    assert RecommendationReasonCode.EXTRAPOLATION_RISK not in allowed.global_reason_codes


def test_uncertainty_gates() -> None:
    gate = RecommendationSafetyGate(
        policy=RecommendationSafetyPolicy(
            require_uncertainty=True,
            require_acceptable_uncertainty=False,
        )
    )
    unavailable = gate.evaluate(
        _request(),
        context=_context(uncertainty_available=False),
    )
    assert (
        RecommendationReasonCode.UNCERTAINTY_UNAVAILABLE
        in unavailable.global_reason_codes
    )

    gate2 = RecommendationSafetyGate(
        policy=RecommendationSafetyPolicy(
            require_uncertainty=True,
            require_acceptable_uncertainty=True,
        )
    )
    unacceptable = gate2.evaluate(
        _request(),
        context=_context(
            uncertainty_available=True,
            uncertainty_acceptable=False,
        ),
    )
    assert (
        RecommendationReasonCode.UNCERTAINTY_UNACCEPTABLE
        in unacceptable.global_reason_codes
    )

    gate3 = RecommendationSafetyGate(policy=_approved_policy())
    allowed = gate3.evaluate(
        _request(),
        context=_context(uncertainty_available=False),
    )
    assert (
        RecommendationReasonCode.UNCERTAINTY_UNAVAILABLE
        not in allowed.global_reason_codes
    )


def test_global_reason_order_deterministic() -> None:
    gate = RecommendationSafetyGate(
        policy=RecommendationSafetyPolicy(
            require_safe_leakage_report=True,
            require_final_evaluation=True,
            require_acceptable_model_performance=True,
            minimum_diagnosis_confidence=0.9,
            minimum_reference_rows=50,
            block_on_extrapolation=True,
            require_uncertainty=True,
            require_acceptable_uncertainty=True,
        )
    )
    decision = gate.evaluate(
        _request(diagnosis_kwargs={"confidence": 0.1, "reference_row_count": 1}),
        context=_context(
            leakage_report=_blocker_leakage(),
            final_evaluation_available=False,
            model_performance_acceptable=False,
            model_performance_reason="poor",
            extrapolation_detected=True,
            uncertainty_available=False,
        ),
    )
    expected_prefix = [
        RecommendationReasonCode.LEAKAGE_BLOCKER,
        RecommendationReasonCode.FINAL_EVALUATION_MISSING,
        RecommendationReasonCode.MODEL_PERFORMANCE_UNACCEPTABLE,
        RecommendationReasonCode.DIAGNOSIS_CONFIDENCE_TOO_LOW,
        RecommendationReasonCode.REFERENCE_SAMPLE_TOO_SMALL,
        RecommendationReasonCode.EXTRAPOLATION_RISK,
        RecommendationReasonCode.UNCERTAINTY_UNAVAILABLE,
        RecommendationReasonCode.UNCERTAINTY_UNACCEPTABLE,
    ]
    assert decision.global_reason_codes[: len(expected_prefix)] == expected_prefix


# --- variable eligibility ---


def test_controllable_process_passes() -> None:
    gate = RecommendationSafetyGate(policy=_approved_policy())
    decision = gate.evaluate(_request(), context=_context())
    assert "pressure" in decision.eligible_variables
    assert decision.status is RecommendationSafetyStatus.APPROVED


def test_current_value_missing() -> None:
    gate = RecommendationSafetyGate(policy=_approved_policy())
    decision = gate.evaluate(
        _request(
            factors=[_factor("pressure"), _factor("temperature")],
            current_values={"temperature": 80.0},
            constraints=[_constraint("pressure"), _constraint("temperature")],
            user_confirmed_controllable_variables=["temperature"],
        ),
        context=_context(),
    )
    pressure = next(
        item for item in decision.variable_assessments if item.variable == "pressure"
    )
    assert RecommendationReasonCode.CURRENT_VALUE_MISSING in pressure.reason_codes
    assert pressure.eligible is False


def test_constraint_missing_and_optional_caution() -> None:
    gate = RecommendationSafetyGate(policy=_approved_policy())
    refused_or_blocked = gate.evaluate(
        _request(
            factors=[_factor("pressure")],
            current_values={"pressure": 50.0},
            constraints=[],
            user_confirmed_controllable_variables=["pressure"],
        ),
        context=_context(),
    )
    assessment = refused_or_blocked.variable_assessments[0]
    assert RecommendationReasonCode.CONSTRAINT_MISSING in assessment.reason_codes

    gate2 = RecommendationSafetyGate(
        policy=_approved_policy(require_constraints=False)
    )
    caution = gate2.evaluate(
        _request(
            factors=[_factor("pressure")],
            current_values={"pressure": 50.0},
            constraints=[],
            user_confirmed_controllable_variables=["pressure"],
        ),
        context=_context(),
    )
    assert caution.status is RecommendationSafetyStatus.CAUTION
    assert caution.variable_assessments[0].eligible is True
    assert any(
        "Constraint missing" in warning
        for warning in caution.variable_assessments[0].warnings
    )


def test_non_controllable_and_unknown_role() -> None:
    gate = RecommendationSafetyGate(policy=_approved_policy())
    decision = gate.evaluate(
        _request(
            factors=[
                _factor(
                    "sensor",
                    role=ColumnRole.STATE_SENSOR,
                    controllable=False,
                )
            ],
            current_values={"sensor": 1.0},
            constraints=[_constraint("sensor")],
            user_confirmed_controllable_variables=[],
        ),
        context=_context(),
    )
    assert (
        RecommendationReasonCode.NON_CONTROLLABLE_VARIABLE
        in decision.variable_assessments[0].reason_codes
    )

    unknown = gate.evaluate(
        _request(
            factors=[_factor("x", role=ColumnRole.UNKNOWN, controllable=False)],
            current_values={"x": 1.0},
            constraints=[_constraint("x")],
            user_confirmed_controllable_variables=[],
        ),
        context=_context(),
    )
    assert (
        RecommendationReasonCode.NON_CONTROLLABLE_VARIABLE
        in unknown.variable_assessments[0].reason_codes
    )


def test_user_controllability_override_and_confirmation_required() -> None:
    gate = RecommendationSafetyGate(policy=_approved_policy())
    override = gate.evaluate(
        _request(
            factors=[
                _factor(
                    "sensor",
                    role=ColumnRole.STATE_SENSOR,
                    controllable=False,
                )
            ],
            current_values={"sensor": 1.0},
            constraints=[_constraint("sensor")],
            user_confirmed_controllable_variables=["sensor"],
        ),
        context=_context(),
    )
    assert override.status is RecommendationSafetyStatus.CAUTION
    assert override.variable_assessments[0].eligible is True
    assert override.metadata["user_controllability_override_count"] == 1

    missing_confirmation = gate.evaluate(
        _request(
            factors=[_factor("pressure")],
            current_values={"pressure": 50.0},
            constraints=[_constraint("pressure")],
            user_confirmed_controllable_variables=[],
        ),
        context=_context(),
    )
    assert (
        RecommendationReasonCode.USER_CONFIRMATION_REQUIRED
        in missing_confirmation.variable_assessments[0].reason_codes
    )


def test_verification_required_and_override() -> None:
    gate = RecommendationSafetyGate(policy=_approved_policy())
    blocked = gate.evaluate(
        _request(
            factors=[_factor("pressure", needs_verification=True)],
            current_values={"pressure": 50.0},
            constraints=[_constraint("pressure")],
            user_confirmed_controllable_variables=["pressure"],
            user_verified_variables=[],
        ),
        context=_context(),
    )
    assert (
        RecommendationReasonCode.VERIFICATION_REQUIRED
        in blocked.variable_assessments[0].reason_codes
    )

    caution = gate.evaluate(
        _request(
            factors=[_factor("pressure", needs_verification=True)],
            current_values={"pressure": 50.0},
            constraints=[_constraint("pressure")],
            user_confirmed_controllable_variables=["pressure"],
            user_verified_variables=["pressure"],
        ),
        context=_context(),
    )
    assert caution.status is RecommendationSafetyStatus.CAUTION
    assert caution.variable_assessments[0].eligible is True
    assert caution.metadata["user_verification_override_count"] == 1


def test_constraint_bounds_and_boundaries() -> None:
    gate = RecommendationSafetyGate(policy=_approved_policy())
    low = gate.evaluate(
        _request(
            factors=[_factor("pressure")],
            current_values={"pressure": -1.0},
            constraints=[_constraint("pressure", minimum=0.0, maximum=100.0)],
            user_confirmed_controllable_variables=["pressure"],
        ),
        context=_context(),
    )
    assert (
        RecommendationReasonCode.CURRENT_VALUE_OUTSIDE_CONSTRAINT
        in low.variable_assessments[0].reason_codes
    )

    high = gate.evaluate(
        _request(
            factors=[_factor("pressure")],
            current_values={"pressure": 101.0},
            constraints=[_constraint("pressure", minimum=0.0, maximum=100.0)],
            user_confirmed_controllable_variables=["pressure"],
        ),
        context=_context(),
    )
    assert (
        RecommendationReasonCode.CURRENT_VALUE_OUTSIDE_CONSTRAINT
        in high.variable_assessments[0].reason_codes
    )

    boundary = gate.evaluate(
        _request(
            factors=[_factor("pressure")],
            current_values={"pressure": 0.0},
            constraints=[_constraint("pressure", minimum=0.0, maximum=100.0)],
            user_confirmed_controllable_variables=["pressure"],
        ),
        context=_context(),
    )
    assert boundary.variable_assessments[0].eligible is True


def test_no_allowed_values_field_on_variable_constraint() -> None:
    """VariableConstraint has no allowed-values field; do not invent membership."""
    fields = set(VariableConstraint.model_fields)
    assert "allowed_values" not in fields
    assert "step" not in fields
    assert "variable" in fields
    assert "minimum" in fields
    assert "maximum" in fields


def test_ranking_order_candidate_limit_and_assessments() -> None:
    gate = RecommendationSafetyGate(
        policy=_approved_policy(maximum_candidate_variables=1)
    )
    decision = gate.evaluate(
        _request(
            factors=[_factor("pressure"), _factor("temperature"), _factor("flow")],
            current_values={"pressure": 1.0, "temperature": 2.0, "flow": 3.0},
            constraints=[
                _constraint("pressure"),
                _constraint("temperature"),
                _constraint("flow"),
            ],
            user_confirmed_controllable_variables=["pressure", "temperature", "flow"],
        ),
        context=_context(),
    )
    assert [item.variable for item in decision.variable_assessments] == [
        "pressure",
        "temperature",
        "flow",
    ]
    assert [item.factor_rank for item in decision.variable_assessments] == [1, 2, 3]
    assert decision.eligible_variables == ["pressure"]
    assert decision.blocked_variables == ["temperature", "flow"]
    for variable in ("temperature", "flow"):
        assessment = next(
            item for item in decision.variable_assessments if item.variable == variable
        )
        assert (
            RecommendationReasonCode.CANDIDATE_LIMIT_EXCEEDED in assessment.reason_codes
        )
    assert {item.variable for item in decision.variable_assessments if item.eligible} == (
        set(decision.eligible_variables)
    )
    assert {
        item.variable for item in decision.variable_assessments if not item.eligible
    } == set(decision.blocked_variables)


def test_factor_duplicate_defense() -> None:
    # DiagnosisResult rejects duplicate factor variables; request construction fails.
    with pytest.raises(ValidationError):
        _request(factors=[_factor("pressure"), _factor("pressure")])


# --- status ---


def test_status_approved_caution_refused_priority() -> None:
    gate = RecommendationSafetyGate(policy=_approved_policy())
    approved = gate.evaluate(_request(), context=_context())
    assert approved.status is RecommendationSafetyStatus.APPROVED

    caution_override = gate.evaluate(
        _request(
            factors=[
                _factor(
                    "sensor",
                    role=ColumnRole.STATE_SENSOR,
                    controllable=False,
                )
            ],
            current_values={"sensor": 1.0},
            constraints=[_constraint("sensor")],
            user_confirmed_controllable_variables=["sensor"],
        ),
        context=_context(),
    )
    assert caution_override.status is RecommendationSafetyStatus.CAUTION

    caution_partial = gate.evaluate(
        _request(
            factors=[_factor("pressure"), _factor("temperature")],
            user_confirmed_controllable_variables=["pressure"],
        ),
        context=_context(),
    )
    assert caution_partial.status is RecommendationSafetyStatus.CAUTION
    assert RecommendationReasonCode.PARTIAL_ELIGIBILITY in (
        caution_partial.global_reason_codes
    )

    gate_no_partial = RecommendationSafetyGate(
        policy=_approved_policy(allow_partial_eligibility=False)
    )
    refused_partial = gate_no_partial.evaluate(
        _request(
            factors=[_factor("pressure"), _factor("temperature")],
            user_confirmed_controllable_variables=["pressure"],
        ),
        context=_context(),
    )
    assert refused_partial.status is RecommendationSafetyStatus.REFUSED
    assert refused_partial.eligible_variables == []

    no_eligible = gate.evaluate(
        _request(
            factors=[_factor("pressure")],
            current_values={"pressure": 50.0},
            constraints=[_constraint("pressure")],
            user_confirmed_controllable_variables=[],
        ),
        context=_context(),
    )
    assert no_eligible.status is RecommendationSafetyStatus.REFUSED
    assert RecommendationReasonCode.NO_ELIGIBLE_VARIABLES in (
        no_eligible.global_reason_codes
    )

    global_block = gate.evaluate(
        _request(),
        context=_context(leakage_report=_blocker_leakage()),
    )
    assert global_block.status is RecommendationSafetyStatus.REFUSED
    assert global_block.eligible_variables == []
    assert all(not item.eligible for item in global_block.variable_assessments)


# --- output ---


def test_output_order_disclaimer_metadata_and_cleanliness() -> None:
    gate = RecommendationSafetyGate(
        policy=_approved_policy(maximum_candidate_variables=1)
    )
    decision = gate.evaluate(
        _request(
            factors=[_factor("pressure"), _factor("temperature")],
            user_confirmed_controllable_variables=["pressure", "temperature"],
        ),
        context=_context(),
    )
    assert decision.eligible_variables == ["pressure"]
    assert decision.blocked_variables == ["temperature"]
    assert decision.variable_assessments[0].factor_rank == 1
    assert decision.disclaimer == DEFAULT_RECOMMENDATION_DISCLAIMER
    assert "model-based" in decision.disclaimer.lower()
    assert "association" in decision.disclaimer.lower()
    assert "causation" in decision.disclaimer.lower()
    assert "verif" in decision.disclaimer.lower()
    assert decision.metadata["recommendation_generated"] is False
    assert decision.metadata["optimization_performed"] is False
    assert decision.metadata["association_not_causation"] is True
    assert decision.metadata["eligible_variable_count"] == 1
    assert decision.metadata["blocked_variable_count"] == 1
    assert decision.evaluated_at.tzinfo is not None
    assert decision.evaluated_at.utcoffset() is not None
    for value in decision.metadata.values():
        assert value is None or isinstance(value, (str, int, float, bool))
    assert len(decision.messages) == len(set(decision.messages))


def test_language_soft() -> None:
    gate = RecommendationSafetyGate()
    decision = gate.evaluate(
        _request(),
        context=_context(leakage_report=_blocker_leakage()),
    )
    joined = " ".join(decision.messages).lower()
    assert "proven root cause" not in joined
    assert "will fix" not in joined
    assert "guaranteed improvement" not in joined
    assert "반드시" not in joined
    assert "무조건" not in joined


# --- determinism / no cache ---


def test_no_cache_determinism_and_isolation() -> None:
    gate = RecommendationSafetyGate(policy=_approved_policy())
    request = _request()
    context = _context()
    first = gate.evaluate(request, context=context)
    second = gate.evaluate(request, context=context)
    assert first.model_dump(exclude={"evaluated_at"}) == second.model_dump(
        exclude={"evaluated_at"}
    )
    first.eligible_variables.append("mutated")
    third = gate.evaluate(request, context=context)
    assert "mutated" not in third.eligible_variables

    gate_a = RecommendationSafetyGate(policy=_approved_policy())
    gate_b = RecommendationSafetyGate(policy=_approved_policy())
    assert gate_a.evaluate(request, context=context).status == gate_b.evaluate(
        request, context=context
    ).status
