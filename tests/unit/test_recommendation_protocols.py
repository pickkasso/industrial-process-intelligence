"""Unit tests for BaseRecommendationEngine abstract contract (Step 9A)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from process_intelligence.core.enums import AnalysisTask, ColumnRole
from process_intelligence.core.schemas import RootCauseFactor, VariableConstraint
from process_intelligence.diagnosis.enums import DiagnosisMethod, DiagnosisScope
from process_intelligence.diagnosis.schemas import DiagnosisResult
from process_intelligence.recommendation import (
    BaseRecommendationEngine,
    RecommendationObjective,
    RecommendationReasonCode,
    RecommendationRequest,
    RecommendationResult,
    RecommendationSafetyDecision,
    RecommendationSafetyStatus,
    RecommendationStatus,
    VariableEligibilityAssessment,
)
from process_intelligence.recommendation.schemas import DEFAULT_RECOMMENDATION_DISCLAIMER


def _diagnosis() -> DiagnosisResult:
    return DiagnosisResult(
        anomaly_id="a-1",
        task=AnalysisTask.UNSUPERVISED_ANOMALY,
        method_used=[DiagnosisMethod.GROUP_COMPARISON],
        scope=DiagnosisScope.SINGLE_EVENT,
        factors=[
            RootCauseFactor(
                variable="pressure",
                direction="increase",
                deviation=1.0,
                role=ColumnRole.CONTROLLABLE_PROCESS,
                controllable=True,
                evidence="Association only.",
                confidence=0.8,
                needs_verification=False,
            )
        ],
        confidence=0.8,
        analyzed_row_count=1,
        reference_row_count=10,
        caveats=["Association only; causation is not established."],
        generated_at=datetime(2026, 7, 21, 10, 0, tzinfo=UTC),
    )


def _request() -> RecommendationRequest:
    return RecommendationRequest(
        task=AnalysisTask.UNSUPERVISED_ANOMALY,
        diagnosis=_diagnosis(),
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        current_values={"pressure": 50.0},
        constraints=[
            VariableConstraint(
                variable="pressure",
                adjustable=True,
                minimum=0.0,
                maximum=100.0,
                fixed=False,
            )
        ],
        user_confirmed_controllable_variables=["pressure"],
        user_verified_variables=[],
        max_simultaneous_changes=1,
    )


def _approved_decision() -> RecommendationSafetyDecision:
    return RecommendationSafetyDecision(
        status=RecommendationSafetyStatus.APPROVED,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        eligible_variables=["pressure"],
        blocked_variables=[],
        variable_assessments=[
            VariableEligibilityAssessment(
                variable="pressure",
                factor_rank=1,
                factor_confidence=0.8,
                factor_role=str(ColumnRole.CONTROLLABLE_PROCESS),
                factor_controllable=True,
                factor_needs_verification=False,
                current_value=50.0,
                constraint_present=True,
                user_confirmed_controllable=True,
                user_verified=False,
                eligible=True,
                reason_codes=[],
                warnings=[],
            )
        ],
        global_reason_codes=[],
        messages=["Safety checks passed."],
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        evaluated_at=datetime(2026, 7, 21, 10, 0, tzinfo=UTC),
    )


def _refused_decision() -> RecommendationSafetyDecision:
    return RecommendationSafetyDecision(
        status=RecommendationSafetyStatus.REFUSED,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        eligible_variables=[],
        blocked_variables=["pressure"],
        variable_assessments=[
            VariableEligibilityAssessment(
                variable="pressure",
                factor_rank=1,
                factor_confidence=0.8,
                factor_role=str(ColumnRole.CONTROLLABLE_PROCESS),
                factor_controllable=True,
                factor_needs_verification=False,
                current_value=None,
                constraint_present=True,
                user_confirmed_controllable=True,
                user_verified=False,
                eligible=False,
                reason_codes=[RecommendationReasonCode.CURRENT_VALUE_MISSING],
                warnings=[],
            )
        ],
        global_reason_codes=[RecommendationReasonCode.NO_ELIGIBLE_VARIABLES],
        messages=["No eligible variables."],
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        evaluated_at=datetime(2026, 7, 21, 10, 0, tzinfo=UTC),
    )


class _ConcreteEngine(BaseRecommendationEngine):
    """Minimal concrete engine used only for contract tests."""

    def __init__(self) -> None:
        self._meta: dict[str, str | int | float | bool | None] = {
            "name": "concrete",
            "version": 1,
            "generates_recommendations": False,
        }

    def recommend(
        self,
        request: RecommendationRequest,
        *,
        safety_decision: RecommendationSafetyDecision,
    ) -> RecommendationResult:
        _ = request
        if safety_decision.status is RecommendationSafetyStatus.REFUSED:
            return RecommendationResult(
                status=RecommendationStatus.REFUSED,
                objective=safety_decision.objective,
                safety_decision=safety_decision,
                changes=[],
                confidence=0.0,
                extrapolation_flag=False,
                uncertainty_available=False,
                disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
                generated_at=datetime(2026, 7, 21, 10, 0, tzinfo=UTC),
                metadata={"deterministic": True},
            )
        return RecommendationResult(
            status=RecommendationStatus.READY_FOR_OPTIMIZATION,
            objective=safety_decision.objective,
            safety_decision=safety_decision,
            changes=[],
            confidence=0.5,
            extrapolation_flag=False,
            uncertainty_available=False,
            disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
            generated_at=datetime(2026, 7, 21, 10, 0, tzinfo=UTC),
            metadata={"deterministic": True},
        )

    def get_metadata(self) -> dict[str, str | int | float | bool | None]:
        return dict(self._meta)


def test_base_engine_is_abc() -> None:
    from abc import ABC

    assert issubclass(BaseRecommendationEngine, ABC)


def test_cannot_instantiate_base() -> None:
    with pytest.raises(TypeError):
        BaseRecommendationEngine()  # type: ignore[abstract]


def test_missing_recommend_rejected() -> None:
    class Incomplete(BaseRecommendationEngine):
        def get_metadata(self) -> dict[str, str | int | float | bool | None]:
            return {}

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_missing_get_metadata_rejected() -> None:
    class Incomplete(BaseRecommendationEngine):
        def recommend(
            self,
            request: RecommendationRequest,
            *,
            safety_decision: RecommendationSafetyDecision,
        ) -> RecommendationResult:
            raise AssertionError("unused")

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_concrete_subclass_and_signature() -> None:
    engine = _ConcreteEngine()
    result = engine.recommend(_request(), safety_decision=_approved_decision())
    assert isinstance(result, RecommendationResult)
    assert result.status is RecommendationStatus.READY_FOR_OPTIMIZATION
    assert result.changes == []


def test_refused_decision_does_not_generate_changes() -> None:
    engine = _ConcreteEngine()
    result = engine.recommend(_request(), safety_decision=_refused_decision())
    assert result.status is RecommendationStatus.REFUSED
    assert result.changes == []
    assert result.proposed_prediction is None
    assert result.proposed_anomaly_score is None


def test_metadata_independence_and_scalars() -> None:
    engine = _ConcreteEngine()
    first = engine.get_metadata()
    second = engine.get_metadata()
    assert first == second
    assert first is not second
    first["name"] = "mutated"
    assert engine.get_metadata()["name"] == "concrete"
    for value in engine.get_metadata().values():
        assert value is None or isinstance(value, (str, int, float, bool))


def test_docstrings_and_determinism() -> None:
    assert BaseRecommendationEngine.__doc__
    assert BaseRecommendationEngine.recommend.__doc__
    assert BaseRecommendationEngine.get_metadata.__doc__
    engine = _ConcreteEngine()
    decision = _approved_decision()
    request = _request()
    first = engine.recommend(request, safety_decision=decision)
    second = engine.recommend(request, safety_decision=decision)
    assert first.model_dump() == second.model_dump()
