"""Tests for recommendation diagnosis-confidence orchestration policy."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from process_intelligence.core.enums import AnalysisTask, ColumnRole
from process_intelligence.core.schemas import RootCauseFactor
from process_intelligence.diagnosis.enums import DiagnosisMethod, DiagnosisScope
from process_intelligence.diagnosis.schemas import DiagnosisResult
from process_intelligence.evaluation.leakage import LeakageReport
from process_intelligence.recommendation.enums import (
    RecommendationObjective,
    RecommendationReasonCode,
)
from process_intelligence.recommendation.safety_gate import RecommendationSafetyGate
from process_intelligence.recommendation.schemas import (
    RecommendationRequest,
    RecommendationSafetyContext,
    RecommendationSafetyPolicy,
)


def _factor(name: str, confidence: float) -> RootCauseFactor:
    return RootCauseFactor(
        variable=name,
        direction="increase",
        deviation=1.0,
        role=ColumnRole.CONTROLLABLE_PROCESS,
        controllable=True,
        evidence="synthetic association evidence for tests",
        confidence=confidence,
        needs_verification=False,
    )


def _diagnosis(factors: list[RootCauseFactor]) -> DiagnosisResult:
    if factors:
        confidence = float(sum(item.confidence for item in factors) / len(factors))
    else:
        confidence = 0.0
    return DiagnosisResult(
        anomaly_id="a1",
        task=AnalysisTask.REGRESSION,
        method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
        scope=DiagnosisScope.SINGLE_EVENT,
        factors=factors,
        confidence=confidence,
        analyzed_row_count=10,
        reference_row_count=50,
        caveats=["association does not establish causation"],
        generated_at=datetime.now(tz=UTC),
        metadata={},
    )


def _leading_mean(factors: list[RootCauseFactor], *, count: int = 5) -> float:
    ranked = sorted((float(item.confidence) for item in factors), reverse=True)
    leading = ranked[: min(count, len(ranked))]
    return float(sum(leading) / len(leading)) if leading else 0.0


def _gate_decision(diagnosis: DiagnosisResult):
    request = RecommendationRequest(
        task=AnalysisTask.REGRESSION,
        diagnosis=diagnosis,
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        current_values={factor.variable: 1.0 for factor in diagnosis.factors}
        or {"placeholder": 1.0},
        constraints=[],
        user_confirmed_controllable_variables=[
            factor.variable for factor in diagnosis.factors
        ],
        user_verified_variables=[factor.variable for factor in diagnosis.factors],
        max_simultaneous_changes=2,
    )
    context = RecommendationSafetyContext(
        leakage_report=LeakageReport(
            is_safe=True,
            issues=[],
            blocker_count=0,
            warning_count=0,
            checked_feature_columns=[],
            checked_preprocessing_event_count=0,
        ),
        final_evaluation_available=True,
        model_performance_acceptable=True,
        model_performance_reason=None,
        extrapolation_detected=False,
        uncertainty_available=False,
        uncertainty_acceptable=None,
        metadata={},
    )
    gate = RecommendationSafetyGate(
        policy=RecommendationSafetyPolicy(
            minimum_diagnosis_confidence=0.20,
            require_constraints=False,
        )
    )
    return gate.evaluate(request, context=context)


def test_full_factor_confidence_is_mean_of_all_factors() -> None:
    factors = [
        _factor("a", 0.40),
        _factor("b", 0.30),
        _factor("c", 0.10),
        _factor("d", 0.05),
        _factor("e", 0.02),
        _factor("f", 0.01),
    ]
    diagnosis = _diagnosis(factors)
    assert diagnosis.confidence == pytest.approx(0.1466666667)
    assert _leading_mean(factors) == pytest.approx(0.174)


def test_leading_factor_confidence_can_exceed_full_mean_and_flip_gate() -> None:
    factors = [
        _factor("strong_a", 0.25),
        _factor("strong_b", 0.24),
        _factor("strong_c", 0.22),
        _factor("mid_d", 0.18),
        _factor("mid_e", 0.14),
        _factor("weak_f", 0.05),
        _factor("weak_g", 0.04),
        _factor("weak_h", 0.03),
        _factor("weak_i", 0.02),
        _factor("weak_j", 0.01),
    ]
    full = _diagnosis(factors)
    leading_confidence = _leading_mean(factors, count=5)
    assert full.confidence < 0.20
    assert leading_confidence >= 0.20

    full_decision = _gate_decision(full)
    assert (
        RecommendationReasonCode.DIAGNOSIS_CONFIDENCE_TOO_LOW
        in full_decision.global_reason_codes
    )

    leading_diagnosis = full.model_copy(update={"confidence": leading_confidence})
    leading_decision = _gate_decision(leading_diagnosis)
    assert (
        RecommendationReasonCode.DIAGNOSIS_CONFIDENCE_TOO_LOW
        not in leading_decision.global_reason_codes
    )


def test_mixed_strong_and_weak_factors_keep_full_mean_policy() -> None:
    factors = [_factor("hot", 0.90)] + [
        _factor(f"tail_{index}", 0.01) for index in range(9)
    ]
    diagnosis = _diagnosis(factors)
    assert diagnosis.confidence == pytest.approx(0.099)
    assert _leading_mean(factors) == pytest.approx(0.188)
    decision = _gate_decision(diagnosis)
    assert (
        RecommendationReasonCode.DIAGNOSIS_CONFIDENCE_TOO_LOW
        in decision.global_reason_codes
    )


def test_contradictory_high_ranked_factors_remain_in_full_mean() -> None:
    factors = [
        _factor("up", 0.50),
        RootCauseFactor(
            variable="down",
            direction="decrease",
            deviation=-1.0,
            role=ColumnRole.CONTROLLABLE_PROCESS,
            controllable=True,
            evidence="opposing association evidence for tests",
            confidence=0.45,
            needs_verification=True,
        ),
        _factor("weak", 0.02),
        _factor("weaker", 0.01),
    ]
    diagnosis = _diagnosis(factors)
    assert diagnosis.confidence == pytest.approx(0.245)
    assert _leading_mean(factors, count=2) == pytest.approx(0.475)
    decision = _gate_decision(diagnosis)
    assert (
        RecommendationReasonCode.DIAGNOSIS_CONFIDENCE_TOO_LOW
        not in decision.global_reason_codes
    )


def test_fewer_than_five_factors_uses_all_available() -> None:
    factors = [_factor("a", 0.30), _factor("b", 0.10), _factor("c", 0.05)]
    diagnosis = _diagnosis(factors)
    assert diagnosis.confidence == pytest.approx(0.15)
    assert _leading_mean(factors, count=5) == pytest.approx(0.15)


def test_deterministic_ordering_under_equal_scores() -> None:
    factors = [
        _factor("b", 0.20),
        _factor("a", 0.20),
        _factor("c", 0.20),
        _factor("d", 0.20),
        _factor("e", 0.20),
        _factor("f", 0.20),
    ]
    diagnosis = _diagnosis(factors)
    assert diagnosis.confidence == pytest.approx(0.20)
    ranked_names = [item.variable for item in diagnosis.factors]
    assert ranked_names == ["b", "a", "c", "d", "e", "f"]
    assert _leading_mean(factors) == pytest.approx(0.20)


def test_pipeline_no_longer_exports_leading_confidence_helper() -> None:
    import process_intelligence.workflow.pipeline as pipeline

    assert not hasattr(pipeline, "_leading_diagnosis_confidence")
    assert not hasattr(pipeline, "_RECOMMENDATION_LEADING_FACTOR_COUNT")
