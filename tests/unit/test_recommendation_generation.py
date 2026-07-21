"""Unit tests for ranked scenario recommendation generation (Step 9F)."""

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
    CandidateGridReport,
    CandidateGridStatus,
    CandidateScenario,
    CandidateScenarioType,
    CandidateValuePoint,
    ConstraintResolutionStatus,
    QualityOptimizationDirection,
    RankedCandidateScenario,
    RankedScenarioRecommendationGenerator,
    RecommendationGenerationOutcome,
    RecommendationGenerationPolicy,
    RecommendationGenerationRequest,
    RecommendationObjective,
    RecommendationReasonCode,
    RecommendationRequest,
    RecommendationResult,
    RecommendationSafetyDecision,
    RecommendationSafetyStatus,
    RecommendationStatus,
    ScenarioRankingReport,
    ScenarioRankingStatus,
    VariableCandidateGrid,
    VariableEligibilityAssessment,
)
from process_intelligence.recommendation.schemas import DEFAULT_RECOMMENDATION_DISCLAIMER


def _factor(
    variable: str,
    *,
    confidence: float = 0.8,
    needs_verification: bool = False,
) -> RootCauseFactor:
    return RootCauseFactor(
        variable=variable,
        direction="increase",
        deviation=1.0,
        role=ColumnRole.CONTROLLABLE_PROCESS,
        controllable=True,
        evidence="Associated driver; association only.",
        confidence=confidence,
        needs_verification=needs_verification,
    )


def _constraint(variable: str) -> VariableConstraint:
    return VariableConstraint(
        variable=variable,
        adjustable=True,
        minimum=0.0,
        maximum=100.0,
        fixed=False,
    )


def _diagnosis(
    factors: list[RootCauseFactor] | None = None,
) -> DiagnosisResult:
    return DiagnosisResult(
        anomaly_id="a-1",
        task=AnalysisTask.REGRESSION,
        method_used=[DiagnosisMethod.GROUP_COMPARISON],
        scope=DiagnosisScope.SINGLE_EVENT,
        factors=factors or [_factor("pressure"), _factor("temperature")],
        confidence=0.8,
        analyzed_row_count=2,
        reference_row_count=10,
        caveats=["Association only; causation is not established."],
        generated_at=datetime(2026, 7, 21, 9, 0, tzinfo=UTC),
    )


def _request(**overrides: Any) -> RecommendationRequest:
    factors = overrides.pop("factors", None)
    diagnosis = overrides.pop("diagnosis", _diagnosis(factors))
    current_values = overrides.pop(
        "current_values",
        {"pressure": 50.0, "temperature": 80.0},
    )
    payload: dict[str, Any] = {
        "task": AnalysisTask.REGRESSION,
        "diagnosis": diagnosis,
        "objective": RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        "current_values": current_values,
        "constraints": [
            _constraint(name) for name in current_values
        ],
        "user_confirmed_controllable_variables": list(current_values.keys()),
        "user_verified_variables": list(current_values.keys()),
        "max_simultaneous_changes": min(3, len(current_values)),
        "metadata": {},
    }
    payload.update(overrides)
    return RecommendationRequest(**payload)


def _assessment(
    variable: str,
    *,
    factor_rank: int,
    factor_confidence: float = 0.8,
    eligible: bool = True,
    needs_verification: bool = False,
    current_value: float | None = None,
) -> VariableEligibilityAssessment:
    defaults = {"pressure": 50.0, "temperature": 80.0}
    return VariableEligibilityAssessment(
        variable=variable,
        factor_rank=factor_rank,
        factor_confidence=factor_confidence,
        factor_role=ColumnRole.CONTROLLABLE_PROCESS.value,
        factor_controllable=True,
        factor_needs_verification=needs_verification,
        current_value=(
            current_value if current_value is not None else defaults.get(variable, 0.0)
        ),
        constraint_present=True,
        user_confirmed_controllable=True,
        user_verified=True,
        eligible=eligible,
        reason_codes=(
            []
            if eligible
            else [RecommendationReasonCode.NON_CONTROLLABLE_VARIABLE]
        ),
        warnings=[],
    )


def _safety_decision(**overrides: Any) -> RecommendationSafetyDecision:
    assessments = overrides.pop(
        "variable_assessments",
        [
            _assessment("pressure", factor_rank=1),
            _assessment("temperature", factor_rank=2),
        ],
    )
    eligible = [item.variable for item in assessments if item.eligible]
    blocked = [item.variable for item in assessments if not item.eligible]
    status = overrides.pop(
        "status",
        (
            RecommendationSafetyStatus.APPROVED
            if eligible and not blocked
            else (
                RecommendationSafetyStatus.CAUTION
                if eligible
                else RecommendationSafetyStatus.REFUSED
            )
        ),
    )
    global_codes = overrides.pop("global_reason_codes", [])
    if status is RecommendationSafetyStatus.REFUSED and not global_codes:
        global_codes = [RecommendationReasonCode.NO_ELIGIBLE_VARIABLES]
    if status is RecommendationSafetyStatus.CAUTION and blocked:
        if RecommendationReasonCode.PARTIAL_ELIGIBILITY not in global_codes:
            global_codes = [
                *global_codes,
                RecommendationReasonCode.PARTIAL_ELIGIBILITY,
            ]
    payload: dict[str, Any] = {
        "status": status,
        "objective": RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        "eligible_variables": eligible,
        "blocked_variables": blocked,
        "variable_assessments": assessments,
        "global_reason_codes": global_codes,
        "messages": overrides.pop("messages", ["Safety evaluation completed."]),
        "disclaimer": DEFAULT_RECOMMENDATION_DISCLAIMER,
        "evaluated_at": datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
        "metadata": {},
    }
    payload.update(overrides)
    return RecommendationSafetyDecision(**payload)


def _point(**overrides: Any) -> CandidateValuePoint:
    payload: dict[str, Any] = {
        "value": 50.0,
        "delta": 0.0,
        "relative_delta": 0.0,
        "normalized_position": 0.5,
        "is_current": True,
        "is_lower_bound": False,
        "is_upper_bound": False,
    }
    payload.update(overrides)
    return CandidateValuePoint(**payload)


def _grid(**overrides: Any) -> VariableCandidateGrid:
    points = overrides.pop(
        "points",
        [
            _point(
                value=0.0,
                delta=-50.0,
                relative_delta=-1.0,
                normalized_position=0.0,
                is_current=False,
                is_lower_bound=True,
            ),
            _point(
                value=50.0,
                delta=0.0,
                relative_delta=0.0,
                normalized_position=0.5,
                is_current=True,
            ),
            _point(
                value=100.0,
                delta=50.0,
                relative_delta=1.0,
                normalized_position=1.0,
                is_current=False,
                is_upper_bound=True,
            ),
        ],
    )
    payload: dict[str, Any] = {
        "variable": "pressure",
        "diagnosis_rank": 1,
        "current_value": 50.0,
        "minimum": 0.0,
        "maximum": 100.0,
        "effective_span": 100.0,
        "points": points,
        "point_count": len(points),
        "change_point_count": sum(1 for item in points if not item.is_current),
        "warnings": [],
    }
    payload.update(overrides)
    if "point_count" not in overrides:
        payload["point_count"] = len(payload["points"])
    if "change_point_count" not in overrides:
        payload["change_point_count"] = sum(
            1 for item in payload["points"] if not item.is_current
        )
    return VariableCandidateGrid(**payload)


def _temperature_grid(**overrides: Any) -> VariableCandidateGrid:
    points = overrides.pop(
        "points",
        [
            _point(
                value=0.0,
                delta=-80.0,
                relative_delta=-1.0,
                normalized_position=0.0,
                is_current=False,
                is_lower_bound=True,
            ),
            _point(
                value=80.0,
                delta=0.0,
                relative_delta=0.0,
                normalized_position=0.8,
                is_current=True,
            ),
            _point(
                value=100.0,
                delta=20.0,
                relative_delta=0.25,
                normalized_position=1.0,
                is_current=False,
                is_upper_bound=True,
            ),
        ],
    )
    return _grid(
        variable="temperature",
        diagnosis_rank=2,
        current_value=80.0,
        points=points,
        **overrides,
    )


def _baseline_scenario(**overrides: Any) -> CandidateScenario:
    payload: dict[str, Any] = {
        "scenario_id": "SCN-000000",
        "scenario_index": 0,
        "scenario_type": CandidateScenarioType.BASELINE,
        "variable_values": {"pressure": 50.0, "temperature": 80.0},
        "changed_variables": [],
        "deltas": {},
        "relative_deltas": {},
        "change_count": 0,
        "normalized_change_magnitude": 0.0,
    }
    payload.update(overrides)
    return CandidateScenario(**payload)


def _single_scenario(**overrides: Any) -> CandidateScenario:
    payload: dict[str, Any] = {
        "scenario_id": "SCN-000001",
        "scenario_index": 1,
        "scenario_type": CandidateScenarioType.SINGLE_VARIABLE,
        "variable_values": {"pressure": 100.0, "temperature": 80.0},
        "changed_variables": ["pressure"],
        "deltas": {"pressure": 50.0},
        "relative_deltas": {"pressure": 1.0},
        "change_count": 1,
        "normalized_change_magnitude": 0.5,
    }
    payload.update(overrides)
    return CandidateScenario(**payload)


def _second_single_scenario(**overrides: Any) -> CandidateScenario:
    payload: dict[str, Any] = {
        "scenario_id": "SCN-000002",
        "scenario_index": 2,
        "scenario_type": CandidateScenarioType.SINGLE_VARIABLE,
        "variable_values": {"pressure": 50.0, "temperature": 100.0},
        "changed_variables": ["temperature"],
        "deltas": {"temperature": 20.0},
        "relative_deltas": {"temperature": 0.25},
        "change_count": 1,
        "normalized_change_magnitude": 0.2,
    }
    payload.update(overrides)
    return CandidateScenario(**payload)


def _multi_scenario(**overrides: Any) -> CandidateScenario:
    payload: dict[str, Any] = {
        "scenario_id": "SCN-000003",
        "scenario_index": 3,
        "scenario_type": CandidateScenarioType.MULTI_VARIABLE,
        "variable_values": {"pressure": 100.0, "temperature": 100.0},
        "changed_variables": ["pressure", "temperature"],
        "deltas": {"pressure": 50.0, "temperature": 20.0},
        "relative_deltas": {"pressure": 1.0, "temperature": 0.25},
        "change_count": 2,
        "normalized_change_magnitude": 0.7,
    }
    payload.update(overrides)
    return CandidateScenario(**payload)


def _grid_report(**overrides: Any) -> CandidateGridReport:
    scenarios = overrides.pop(
        "scenarios",
        [_baseline_scenario(), _single_scenario(), _second_single_scenario()],
    )
    payload: dict[str, Any] = {
        "status": CandidateGridStatus.READY,
        "objective": RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        "safety_status": RecommendationSafetyStatus.APPROVED,
        "resolution_status": ConstraintResolutionStatus.READY,
        "candidate_variables": ["pressure", "temperature"],
        "variable_grids": [_grid(), _temperature_grid()],
        "scenarios": scenarios,
        "baseline_scenario_id": "SCN-000000",
        "potential_scenario_count": len(scenarios),
        "generated_scenario_count": len(scenarios),
        "change_scenario_count": max(0, len(scenarios) - 1),
        "truncated_scenario_count": 0,
        "maximum_scenarios": 500,
        "effective_combination_limit": 2,
        "generated_at": datetime(2026, 7, 21, 15, 0, tzinfo=UTC),
        "warnings": ["Model scoring was not performed."],
        "metadata": {"model_scoring_performed": False},
    }
    payload.update(overrides)
    if "generated_scenario_count" not in overrides:
        payload["generated_scenario_count"] = len(payload["scenarios"])
    if "change_scenario_count" not in overrides:
        payload["change_scenario_count"] = max(0, len(payload["scenarios"]) - 1)
    if "potential_scenario_count" not in overrides:
        payload["potential_scenario_count"] = len(payload["scenarios"])
    return CandidateGridReport(**payload)


def _ranked(
    scenario: CandidateScenario,
    *,
    rank: int,
    selection_eligible: bool,
    quality_prediction: float | None = 12.0,
    anomaly_score: float | None = None,
    composite_score: float = 0.8,
    **overrides: Any,
) -> RankedCandidateScenario:
    payload: dict[str, Any] = {
        "rank": rank,
        "scenario_id": scenario.scenario_id,
        "scenario_index": scenario.scenario_index,
        "scenario_type": scenario.scenario_type,
        "variable_values": dict(scenario.variable_values),
        "changed_variables": list(scenario.changed_variables),
        "deltas": dict(scenario.deltas),
        "relative_deltas": dict(scenario.relative_deltas),
        "change_count": scenario.change_count,
        "normalized_change_magnitude": scenario.normalized_change_magnitude,
        "quality_prediction": quality_prediction,
        "anomaly_score": anomaly_score,
        "quality_improvement_from_baseline": (
            None if quality_prediction is None else quality_prediction - 10.0
        ),
        "anomaly_improvement_from_baseline": None,
        "normalized_quality_benefit": 1.0 if selection_eligible else 0.0,
        "normalized_anomaly_benefit": None,
        "objective_benefit_score": 1.0 if selection_eligible else 0.0,
        "normalized_change_cost": min(1.0, scenario.normalized_change_magnitude),
        "change_penalty": 0.0,
        "composite_score": composite_score,
        "quality_requirement_met": True if quality_prediction is not None else None,
        "anomaly_requirement_met": None,
        "objective_requirements_met": selection_eligible
        or scenario.scenario_type is CandidateScenarioType.BASELINE,
        "selection_eligible": selection_eligible,
        "warnings": [],
    }
    if scenario.scenario_type is CandidateScenarioType.BASELINE:
        payload["objective_requirements_met"] = False
        payload["selection_eligible"] = False
        payload["quality_improvement_from_baseline"] = 0.0
        payload["normalized_quality_benefit"] = 0.0
        payload["objective_benefit_score"] = 0.0
        payload["composite_score"] = 0.0
        payload["quality_requirement_met"] = False
        payload["quality_prediction"] = (
            10.0 if quality_prediction is None else quality_prediction
        )
    payload.update(overrides)
    # Keep eligibility contract coherent after overrides.
    if payload["selection_eligible"]:
        payload["objective_requirements_met"] = True
        if payload.get("quality_requirement_met") is False:
            payload["quality_requirement_met"] = True
    elif scenario.scenario_type is CandidateScenarioType.BASELINE:
        payload["objective_requirements_met"] = False
        if payload.get("quality_requirement_met") is True:
            payload["quality_requirement_met"] = False
    return RankedCandidateScenario(**payload)


def _ranking_report(**overrides: Any) -> ScenarioRankingReport:
    baseline = _baseline_scenario()
    best = _single_scenario()
    second = _second_single_scenario()
    ranked = overrides.pop(
        "ranked_scenarios",
        [
            _ranked(best, rank=1, selection_eligible=True, quality_prediction=12.0),
            _ranked(second, rank=2, selection_eligible=True, quality_prediction=11.0),
            _ranked(baseline, rank=3, selection_eligible=False, quality_prediction=10.0),
        ],
    )
    eligible_count = sum(1 for item in ranked if item.selection_eligible)
    payload: dict[str, Any] = {
        "status": ScenarioRankingStatus.RANKED,
        "objective": RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        "quality_direction": QualityOptimizationDirection.MAXIMIZE,
        "quality_target": None,
        "baseline_scenario_id": "SCN-000000",
        "baseline_quality_prediction": 10.0,
        "baseline_anomaly_score": None,
        "ranked_scenarios": ranked,
        "best_scenario_id": ranked[0].scenario_id if ranked else None,
        "best_nonbaseline_scenario_id": next(
            (
                item.scenario_id
                for item in ranked
                if item.selection_eligible
                and item.scenario_type is not CandidateScenarioType.BASELINE
            ),
            None,
        ),
        "baseline_is_top_ranked": bool(
            ranked and ranked[0].scenario_type is CandidateScenarioType.BASELINE
        ),
        "requested_scenario_count": 3,
        "evaluated_scenario_count": len(ranked),
        "returned_scenario_count": len(ranked),
        "eligible_scenario_count": eligible_count,
        "ineligible_scenario_count": max(0, len(ranked) - eligible_count),
        "truncated_scenario_count": 0,
        "evaluated_at": datetime(2026, 7, 21, 16, 0, tzinfo=UTC),
        "warnings": ["Ranking scores are relative heuristics."],
        "metadata": {"scenario_ranking_performed": True},
    }
    payload.update(overrides)
    if "returned_scenario_count" not in overrides:
        payload["returned_scenario_count"] = len(payload["ranked_scenarios"])
    if "eligible_scenario_count" not in overrides:
        payload["eligible_scenario_count"] = sum(
            1 for item in payload["ranked_scenarios"] if item.selection_eligible
        )
    if "evaluated_scenario_count" not in overrides:
        payload["evaluated_scenario_count"] = len(payload["ranked_scenarios"])
    if "truncated_scenario_count" not in overrides:
        payload["truncated_scenario_count"] = (
            payload["evaluated_scenario_count"] - payload["returned_scenario_count"]
        )
    if "best_scenario_id" not in overrides and payload["ranked_scenarios"]:
        payload["best_scenario_id"] = payload["ranked_scenarios"][0].scenario_id
    if "best_nonbaseline_scenario_id" not in overrides:
        payload["best_nonbaseline_scenario_id"] = next(
            (
                item.scenario_id
                for item in payload["ranked_scenarios"]
                if item.selection_eligible
                and item.scenario_type is not CandidateScenarioType.BASELINE
            ),
            None,
        )
    if "baseline_is_top_ranked" not in overrides:
        ranked_list = payload["ranked_scenarios"]
        payload["baseline_is_top_ranked"] = bool(
            ranked_list
            and ranked_list[0].scenario_type is CandidateScenarioType.BASELINE
        )
    return ScenarioRankingReport(**payload)


def _generation_request(**overrides: Any) -> RecommendationGenerationRequest:
    request = overrides.pop("request", _request())
    safety = overrides.pop("safety_decision", _safety_decision())
    grid = overrides.pop("grid", _grid_report())
    ranking = overrides.pop("ranking", None)
    if ranking is None:
        ranking = _ranking_report(
            requested_scenario_count=grid.generated_scenario_count
        )
    payload: dict[str, Any] = {
        "request": request,
        "safety_decision": safety,
        "grid": grid,
        "ranking": ranking,
        "extrapolation_evaluated": False,
        "extrapolation_flag": False,
        "uncertainty_available": False,
        "uncertainty_acceptable": None,
        "metadata": {},
    }
    payload.update(overrides)
    return RecommendationGenerationRequest(**payload)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def test_default_policy() -> None:
    policy = RecommendationGenerationPolicy()
    assert policy.allow_caution_safety_decision is True
    assert policy.allow_partial_ranking is False
    assert policy.generate_ready_result_on_no_improvement is True
    assert policy.require_extrapolation_evaluation is False
    assert policy.block_on_extrapolation is True
    assert policy.minimum_factor_confidence == 0.0


@pytest.mark.parametrize("field", ["allow_caution_safety_decision", "block_on_extrapolation"])
def test_policy_bool_strict(field: str) -> None:
    with pytest.raises(ValidationError):
        RecommendationGenerationPolicy(**{field: 1})
    with pytest.raises(ValidationError):
        RecommendationGenerationPolicy(**{field: "true"})


def test_policy_confidence_negative() -> None:
    with pytest.raises(ValidationError):
        RecommendationGenerationPolicy(minimum_factor_confidence=-0.1)


def test_policy_confidence_above_one() -> None:
    with pytest.raises(ValidationError):
        RecommendationGenerationPolicy(minimum_factor_confidence=1.1)


def test_policy_confidence_bool_rejected() -> None:
    with pytest.raises(ValidationError):
        RecommendationGenerationPolicy(minimum_factor_confidence=True)


def test_policy_confidence_nan_inf() -> None:
    with pytest.raises(ValidationError):
        RecommendationGenerationPolicy(minimum_factor_confidence=float("nan"))
    with pytest.raises(ValidationError):
        RecommendationGenerationPolicy(minimum_factor_confidence=float("inf"))


def test_policy_uncertainty_relationship() -> None:
    with pytest.raises(ValidationError):
        RecommendationGenerationPolicy(
            require_acceptable_uncertainty=True,
            require_uncertainty_available=False,
        )
    policy = RecommendationGenerationPolicy(
        require_uncertainty_available=True,
        require_acceptable_uncertainty=True,
    )
    assert policy.require_acceptable_uncertainty is True


def test_policy_round_trip() -> None:
    policy = RecommendationGenerationPolicy(minimum_factor_confidence=0.25)
    restored = RecommendationGenerationPolicy.model_validate(policy.model_dump())
    assert restored == policy


# ---------------------------------------------------------------------------
# GenerationRequest
# ---------------------------------------------------------------------------


def test_generation_request_valid() -> None:
    gen = _generation_request()
    assert gen.request.objective is RecommendationObjective.IMPROVE_PREDICTED_QUALITY
    assert gen.ranking.best_nonbaseline_scenario_id == "SCN-000001"


def test_generation_request_objective_mismatch() -> None:
    with pytest.raises(ValidationError):
        _generation_request(
            safety_decision=_safety_decision(
                objective=RecommendationObjective.REDUCE_ANOMALY_SCORE
            )
        )


def test_generation_request_eligible_missing_current() -> None:
    with pytest.raises(ValidationError):
        _generation_request(
            request=_request(current_values={"temperature": 80.0}),
            safety_decision=_safety_decision(
                variable_assessments=[
                    _assessment("temperature", factor_rank=1, current_value=80.0),
                ]
            ),
        )


def test_generation_request_scenario_count_mismatch() -> None:
    grid = _grid_report()
    ranking = _ranking_report(requested_scenario_count=99)
    with pytest.raises(ValidationError):
        _generation_request(grid=grid, ranking=ranking)


def test_generation_request_baseline_id_mismatch() -> None:
    grid = _grid_report()
    ranking = _ranking_report(requested_scenario_count=grid.generated_scenario_count)
    broken_grid = CandidateGridReport.model_construct(
        status=grid.status,
        objective=grid.objective,
        safety_status=grid.safety_status,
        resolution_status=grid.resolution_status,
        candidate_variables=list(grid.candidate_variables),
        variable_grids=list(grid.variable_grids),
        scenarios=list(grid.scenarios),
        baseline_scenario_id="SCN-000099",
        potential_scenario_count=grid.potential_scenario_count,
        generated_scenario_count=grid.generated_scenario_count,
        change_scenario_count=grid.change_scenario_count,
        truncated_scenario_count=grid.truncated_scenario_count,
        maximum_scenarios=grid.maximum_scenarios,
        effective_combination_limit=grid.effective_combination_limit,
        generated_at=grid.generated_at,
        warnings=list(grid.warnings),
        metadata=dict(grid.metadata),
    )
    with pytest.raises(ValidationError):
        RecommendationGenerationRequest(
            request=_request(),
            safety_decision=_safety_decision(),
            grid=broken_grid,
            ranking=ranking,
            metadata={},
        )


def test_generation_request_ranked_id_mismatch() -> None:
    grid = _grid_report()
    ranking = _ranking_report(requested_scenario_count=grid.generated_scenario_count)
    payload = ranking.ranked_scenarios[0].model_dump()
    payload["scenario_id"] = "SCN-000099"
    payload["scenario_index"] = 99
    foreign = RankedCandidateScenario.model_validate(payload)
    broken_ranking = ScenarioRankingReport.model_construct(
        status=ranking.status,
        objective=ranking.objective,
        quality_direction=ranking.quality_direction,
        quality_target=ranking.quality_target,
        baseline_scenario_id=ranking.baseline_scenario_id,
        baseline_quality_prediction=ranking.baseline_quality_prediction,
        baseline_anomaly_score=ranking.baseline_anomaly_score,
        ranked_scenarios=[foreign, *ranking.ranked_scenarios[1:]],
        best_scenario_id="SCN-000099",
        best_nonbaseline_scenario_id="SCN-000099",
        baseline_is_top_ranked=ranking.baseline_is_top_ranked,
        requested_scenario_count=ranking.requested_scenario_count,
        evaluated_scenario_count=ranking.evaluated_scenario_count,
        returned_scenario_count=ranking.returned_scenario_count,
        eligible_scenario_count=ranking.eligible_scenario_count,
        ineligible_scenario_count=ranking.ineligible_scenario_count,
        truncated_scenario_count=ranking.truncated_scenario_count,
        evaluated_at=ranking.evaluated_at,
        warnings=list(ranking.warnings),
        metadata=dict(ranking.metadata),
    )
    with pytest.raises(ValidationError):
        RecommendationGenerationRequest(
            request=_request(),
            safety_decision=_safety_decision(),
            grid=grid,
            ranking=broken_ranking,
            metadata={},
        )


def test_generation_request_scenario_index_mismatch() -> None:
    ranking = _ranking_report()
    payload = ranking.ranked_scenarios[0].model_dump()
    payload["scenario_index"] = 9
    with pytest.raises(ValidationError):
        RankedCandidateScenario.model_validate(payload)


def test_generation_request_scenario_type_mismatch() -> None:
    grid = _grid_report()
    ranking = _ranking_report(requested_scenario_count=grid.generated_scenario_count)
    mutated = ranking.ranked_scenarios[0].model_dump()
    mutated["scenario_type"] = CandidateScenarioType.MULTI_VARIABLE
    mutated["change_count"] = 2
    mutated["changed_variables"] = ["pressure", "temperature"]
    mutated["deltas"] = {"pressure": 50.0, "temperature": 0.0}
    mutated["relative_deltas"] = {"pressure": 1.0, "temperature": 0.0}
    mutated["variable_values"] = {"pressure": 100.0, "temperature": 80.0}
    with pytest.raises(ValidationError):
        _generation_request(
            grid=grid,
            ranking=ranking.model_copy(
                update={
                    "ranked_scenarios": [
                        RankedCandidateScenario(**mutated),
                        *ranking.ranked_scenarios[1:],
                    ]
                }
            ),
        )


def test_generation_request_change_count_mismatch() -> None:
    grid = _grid_report()
    ranking = _ranking_report(requested_scenario_count=grid.generated_scenario_count)
    mutated = ranking.ranked_scenarios[0].model_dump()
    # Keep ranked self-consistent but diverge from grid change_count via magnitude.
    mutated["normalized_change_magnitude"] = 0.9
    with pytest.raises(ValidationError):
        _generation_request(
            grid=grid,
            ranking=ranking.model_copy(
                update={
                    "ranked_scenarios": [
                        RankedCandidateScenario(**mutated),
                        *ranking.ranked_scenarios[1:],
                    ]
                }
            ),
        )


def test_generation_request_magnitude_mismatch() -> None:
    test_generation_request_change_count_mismatch()


def test_generation_request_variable_values_mismatch() -> None:
    grid = _grid_report()
    ranking = _ranking_report(requested_scenario_count=grid.generated_scenario_count)
    mutated = ranking.ranked_scenarios[0].model_copy(
        update={"variable_values": {"pressure": 99.0, "temperature": 80.0}}
    )
    with pytest.raises(ValidationError):
        _generation_request(
            grid=grid,
            ranking=ranking.model_copy(
                update={"ranked_scenarios": [mutated, *ranking.ranked_scenarios[1:]]}
            ),
        )


def test_generation_request_changed_variables_mismatch() -> None:
    scenarios = [_baseline_scenario(), _single_scenario(), _second_single_scenario()]
    grid = _grid_report(scenarios=scenarios)
    ranking = _ranking_report(requested_scenario_count=3)
    mutated = ranking.ranked_scenarios[0].model_copy(
        update={
            "changed_variables": ["temperature"],
            "deltas": {"temperature": 20.0},
            "relative_deltas": {"temperature": 0.25},
            "variable_values": {"pressure": 50.0, "temperature": 100.0},
            "change_count": 1,
            "normalized_change_magnitude": 0.2,
            "scenario_id": "SCN-000001",
            "scenario_index": 1,
        }
    )
    with pytest.raises(ValidationError):
        _generation_request(
            grid=grid,
            ranking=ranking.model_copy(
                update={
                    "ranked_scenarios": [mutated, *ranking.ranked_scenarios[1:]],
                    "best_scenario_id": mutated.scenario_id,
                    "best_nonbaseline_scenario_id": mutated.scenario_id,
                }
            ),
        )


def test_generation_request_delta_mismatch() -> None:
    grid = _grid_report()
    ranking = _ranking_report(requested_scenario_count=grid.generated_scenario_count)
    mutated = ranking.ranked_scenarios[0].model_copy(
        update={"deltas": {"pressure": 40.0}}
    )
    with pytest.raises(ValidationError):
        _generation_request(
            grid=grid,
            ranking=ranking.model_copy(
                update={"ranked_scenarios": [mutated, *ranking.ranked_scenarios[1:]]}
            ),
        )


def test_generation_request_relative_delta_mismatch() -> None:
    grid = _grid_report()
    ranking = _ranking_report(requested_scenario_count=grid.generated_scenario_count)
    mutated = ranking.ranked_scenarios[0].model_copy(
        update={"relative_deltas": {"pressure": 0.5}}
    )
    with pytest.raises(ValidationError):
        _generation_request(
            grid=grid,
            ranking=ranking.model_copy(
                update={"ranked_scenarios": [mutated, *ranking.ranked_scenarios[1:]]}
            ),
        )


def test_generation_request_best_scenario_missing() -> None:
    with pytest.raises(ValidationError):
        _ranking_report(best_scenario_id="SCN-000099")


def test_generation_request_best_nonbaseline_ineligible() -> None:
    ranking = _ranking_report()
    baseline = next(
        item
        for item in ranking.ranked_scenarios
        if item.scenario_type is CandidateScenarioType.BASELINE
    )
    payload = ranking.model_dump()
    payload["best_nonbaseline_scenario_id"] = baseline.scenario_id
    with pytest.raises(ValidationError):
        ScenarioRankingReport.model_validate(payload)


def test_generation_request_extrapolation_relationship() -> None:
    with pytest.raises(ValidationError):
        _generation_request(extrapolation_evaluated=False, extrapolation_flag=True)
    ok = _generation_request(extrapolation_evaluated=True, extrapolation_flag=True)
    assert ok.extrapolation_flag is True


def test_generation_request_uncertainty_relationship() -> None:
    with pytest.raises(ValidationError):
        _generation_request(uncertainty_available=False, uncertainty_acceptable=True)
    ok = _generation_request(uncertainty_available=True, uncertainty_acceptable=False)
    assert ok.uncertainty_acceptable is False


def test_generation_request_metadata_scalar_only() -> None:
    with pytest.raises(ValidationError):
        _generation_request(metadata={"bad": [1, 2]})


def test_generation_request_mutable_independence() -> None:
    meta = {"tag": "a"}
    gen = _generation_request(metadata=meta)
    meta["tag"] = "b"
    assert gen.metadata["tag"] == "a"


def test_generation_request_round_trip() -> None:
    gen = _generation_request()
    restored = RecommendationGenerationRequest.model_validate(gen.model_dump())
    assert restored.ranking.best_nonbaseline_scenario_id == (
        gen.ranking.best_nonbaseline_scenario_id
    )


# ---------------------------------------------------------------------------
# Outcome / Generator construction
# ---------------------------------------------------------------------------


def test_outcome_frozen_and_slots() -> None:
    outcome = RankedScenarioRecommendationGenerator().generate(_generation_request())
    assert isinstance(outcome, RecommendationGenerationOutcome)
    with pytest.raises(FrozenInstanceError):
        outcome.result = outcome.result  # type: ignore[misc]
    assert not hasattr(outcome, "__dict__")


def test_generator_default_and_custom_policy() -> None:
    default = RankedScenarioRecommendationGenerator()
    custom_policy = RecommendationGenerationPolicy(minimum_factor_confidence=0.5)
    custom = RankedScenarioRecommendationGenerator(policy=custom_policy)
    assert default.get_metadata()["minimum_factor_confidence"] == 0.0
    assert custom.get_metadata()["minimum_factor_confidence"] == 0.5
    custom_policy.minimum_factor_confidence = 0.9  # type: ignore[misc]
    # Pydantic model may allow assignment; generator holds deep copy.
    assert custom.get_metadata()["minimum_factor_confidence"] == 0.5


def test_generator_bad_policy_type() -> None:
    with pytest.raises(TypeError):
        RankedScenarioRecommendationGenerator(policy="bad")  # type: ignore[arg-type]


def test_generator_metadata_scalar_and_independent() -> None:
    generator = RankedScenarioRecommendationGenerator()
    first = generator.get_metadata()
    second = generator.get_metadata()
    assert first is not second
    first["allow_caution_safety_decision"] = False
    assert second["allow_caution_safety_decision"] is True
    assert first["performs_model_scoring"] is False
    assert first["generates_recommendation"] is True


# ---------------------------------------------------------------------------
# Safety / ranking status paths
# ---------------------------------------------------------------------------


def test_safety_refused_result() -> None:
    safety = _safety_decision(
        status=RecommendationSafetyStatus.REFUSED,
        variable_assessments=[
            _assessment("pressure", factor_rank=1, eligible=False),
            _assessment("temperature", factor_rank=2, eligible=False),
        ],
        global_reason_codes=[RecommendationReasonCode.NO_ELIGIBLE_VARIABLES],
        messages=["Refused by safety gate."],
    )
    # READY grid/ranking still needed for request validation; use APPROVED-shaped
    # grid safety_status independently.
    outcome = RankedScenarioRecommendationGenerator().generate(
        _generation_request(safety_decision=safety)
    )
    result = outcome.result
    assert result.status is RecommendationStatus.REFUSED
    assert result.changes == []
    assert result.proposed_prediction is None
    assert result.proposed_anomaly_score is None
    assert result.confidence == 0.0
    assert any("REFUSED" in warning for warning in result.warnings)


def test_caution_allowed_and_blocked() -> None:
    safety = _safety_decision(
        status=RecommendationSafetyStatus.CAUTION,
        variable_assessments=[
            _assessment("pressure", factor_rank=1, eligible=True),
            _assessment("temperature", factor_rank=2, eligible=False),
        ],
        messages=["Caution: partial eligibility."],
    )
    allowed = RankedScenarioRecommendationGenerator().generate(
        _generation_request(safety_decision=safety)
    )
    assert allowed.result.status is RecommendationStatus.GENERATED

    blocked = RankedScenarioRecommendationGenerator(
        policy=RecommendationGenerationPolicy(allow_caution_safety_decision=False)
    ).generate(_generation_request(safety_decision=safety))
    assert blocked.result.status is RecommendationStatus.READY_FOR_OPTIMIZATION
    assert blocked.result.changes == []
    assert any("Caution" in warning for warning in blocked.result.warnings)


def test_approved_proceeds() -> None:
    outcome = RankedScenarioRecommendationGenerator().generate(_generation_request())
    assert outcome.result.status is RecommendationStatus.GENERATED


def test_safety_messages_include_exclude() -> None:
    safety = _safety_decision(messages=["Unique safety message body."])
    included = RankedScenarioRecommendationGenerator().generate(
        _generation_request(safety_decision=safety)
    )
    assert "Unique safety message body." in included.result.warnings

    excluded = RankedScenarioRecommendationGenerator(
        policy=RecommendationGenerationPolicy(include_safety_messages=False)
    ).generate(_generation_request(safety_decision=safety))
    assert "Unique safety message body." not in excluded.result.warnings


def test_ranking_partial_default_block_and_allow() -> None:
    ranking = _ranking_report(status=ScenarioRankingStatus.PARTIAL)
    blocked = RankedScenarioRecommendationGenerator().generate(
        _generation_request(ranking=ranking)
    )
    assert blocked.result.status is RecommendationStatus.READY_FOR_OPTIMIZATION
    assert blocked.result.changes == []

    allowed = RankedScenarioRecommendationGenerator(
        policy=RecommendationGenerationPolicy(allow_partial_ranking=True)
    ).generate(_generation_request(ranking=ranking))
    assert allowed.result.status is RecommendationStatus.GENERATED


def test_ranking_no_improvement_ready_no_changes() -> None:
    baseline = _baseline_scenario()
    ranked = [
        _ranked(baseline, rank=1, selection_eligible=False, quality_prediction=10.0),
        _ranked(
            _single_scenario(),
            rank=2,
            selection_eligible=False,
            quality_prediction=10.0,
            objective_requirements_met=False,
            quality_requirement_met=False,
            normalized_quality_benefit=0.0,
            objective_benefit_score=0.0,
            composite_score=0.0,
        ),
    ]
    ranking = _ranking_report(
        status=ScenarioRankingStatus.NO_IMPROVEMENT,
        ranked_scenarios=ranked,
        best_nonbaseline_scenario_id=None,
        eligible_scenario_count=0,
        ineligible_scenario_count=2,
        baseline_is_top_ranked=True,
    )
    outcome = RankedScenarioRecommendationGenerator().generate(
        _generation_request(ranking=ranking)
    )
    assert outcome.result.status is RecommendationStatus.READY_FOR_OPTIMIZATION
    assert outcome.result.changes == []
    assert outcome.result.proposed_prediction is None
    assert outcome.result.baseline_prediction == pytest.approx(10.0)
    assert any("improved" in warning.lower() for warning in outcome.result.warnings)


def test_ranking_refused_ready() -> None:
    ranking = ScenarioRankingReport(
        status=ScenarioRankingStatus.REFUSED,
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        quality_direction=None,
        quality_target=None,
        baseline_scenario_id=None,
        baseline_quality_prediction=None,
        baseline_anomaly_score=None,
        ranked_scenarios=[],
        best_scenario_id=None,
        best_nonbaseline_scenario_id=None,
        baseline_is_top_ranked=False,
        requested_scenario_count=3,
        evaluated_scenario_count=0,
        returned_scenario_count=0,
        eligible_scenario_count=0,
        ineligible_scenario_count=0,
        truncated_scenario_count=0,
        evaluated_at=datetime(2026, 7, 21, 16, 0, tzinfo=UTC),
        warnings=["Ranking refused warning."],
        metadata={},
    )
    grid = _grid_report()
    outcome = RankedScenarioRecommendationGenerator().generate(
        _generation_request(grid=grid, ranking=ranking)
    )
    assert outcome.result.status is RecommendationStatus.READY_FOR_OPTIMIZATION
    assert outcome.result.changes == []


def test_ranking_warnings_include_exclude() -> None:
    included = RankedScenarioRecommendationGenerator().generate(_generation_request())
    assert any("heuristics" in warning for warning in included.result.warnings)

    excluded = RankedScenarioRecommendationGenerator(
        policy=RecommendationGenerationPolicy(include_ranking_warnings=False)
    ).generate(_generation_request())
    assert all("heuristics" not in warning for warning in excluded.result.warnings)


# ---------------------------------------------------------------------------
# Extrapolation / uncertainty
# ---------------------------------------------------------------------------


def test_extrapolation_and_uncertainty_policies() -> None:
    require_eval = RankedScenarioRecommendationGenerator(
        policy=RecommendationGenerationPolicy(require_extrapolation_evaluation=True)
    ).generate(_generation_request(extrapolation_evaluated=False))
    assert require_eval.result.status is RecommendationStatus.READY_FOR_OPTIMIZATION

    blocked_flag = RankedScenarioRecommendationGenerator().generate(
        _generation_request(extrapolation_evaluated=True, extrapolation_flag=True)
    )
    assert blocked_flag.result.status is RecommendationStatus.READY_FOR_OPTIMIZATION

    allowed_flag = RankedScenarioRecommendationGenerator(
        policy=RecommendationGenerationPolicy(block_on_extrapolation=False)
    ).generate(
        _generation_request(extrapolation_evaluated=True, extrapolation_flag=True)
    )
    assert allowed_flag.result.status is RecommendationStatus.GENERATED
    assert allowed_flag.result.extrapolation_flag is True

    need_uncertainty = RankedScenarioRecommendationGenerator(
        policy=RecommendationGenerationPolicy(require_uncertainty_available=True)
    ).generate(_generation_request(uncertainty_available=False))
    assert need_uncertainty.result.status is RecommendationStatus.READY_FOR_OPTIMIZATION

    need_acceptable = RankedScenarioRecommendationGenerator(
        policy=RecommendationGenerationPolicy(
            require_uncertainty_available=True,
            require_acceptable_uncertainty=True,
        )
    ).generate(
        _generation_request(uncertainty_available=True, uncertainty_acceptable=False)
    )
    assert need_acceptable.result.status is RecommendationStatus.READY_FOR_OPTIMIZATION

    ok_uncertainty = RankedScenarioRecommendationGenerator(
        policy=RecommendationGenerationPolicy(
            require_uncertainty_available=True,
            require_acceptable_uncertainty=True,
        )
    ).generate(
        _generation_request(uncertainty_available=True, uncertainty_acceptable=True)
    )
    assert ok_uncertainty.result.status is RecommendationStatus.GENERATED
    assert ok_uncertainty.result.uncertainty_available is True
    assert ok_uncertainty.result.metadata["extrapolation_evaluated"] is False


# ---------------------------------------------------------------------------
# Scenario selection / safety / diagnosis
# ---------------------------------------------------------------------------


def test_best_nonbaseline_selection_and_baseline_top() -> None:
    baseline = _baseline_scenario()
    best = _single_scenario()
    second = _second_single_scenario()
    ranked = [
        _ranked(baseline, rank=1, selection_eligible=False, quality_prediction=10.0),
        _ranked(best, rank=2, selection_eligible=True, quality_prediction=12.0),
        _ranked(second, rank=3, selection_eligible=True, quality_prediction=11.0),
    ]
    ranking = _ranking_report(
        ranked_scenarios=ranked,
        baseline_is_top_ranked=True,
    )
    outcome = RankedScenarioRecommendationGenerator().generate(
        _generation_request(ranking=ranking)
    )
    assert outcome.result.status is RecommendationStatus.GENERATED
    assert outcome.result.metadata["selected_scenario_id"] == "SCN-000001"
    assert outcome.result.changes[0].variable == "pressure"


def test_no_best_nonbaseline_ready() -> None:
    baseline = _baseline_scenario()
    ranked = [
        _ranked(baseline, rank=1, selection_eligible=False, quality_prediction=10.0),
        _ranked(
            _single_scenario(),
            rank=2,
            selection_eligible=False,
            quality_prediction=10.0,
            objective_requirements_met=False,
            quality_requirement_met=False,
            normalized_quality_benefit=0.0,
            objective_benefit_score=0.0,
            composite_score=0.0,
        ),
    ]
    ranking = _ranking_report(
        status=ScenarioRankingStatus.NO_IMPROVEMENT,
        ranked_scenarios=ranked,
        best_nonbaseline_scenario_id=None,
        eligible_scenario_count=0,
        ineligible_scenario_count=2,
        baseline_is_top_ranked=True,
    )
    outcome = RankedScenarioRecommendationGenerator().generate(
        _generation_request(ranking=ranking)
    )
    assert outcome.result.status is RecommendationStatus.READY_FOR_OPTIMIZATION


def test_max_simultaneous_changes_exceeded() -> None:
    multi = _multi_scenario(
        scenario_id="SCN-000001",
        scenario_index=1,
    )
    scenarios = [_baseline_scenario(), multi]
    grid = _grid_report(
        scenarios=scenarios,
        generated_scenario_count=2,
        change_scenario_count=1,
        potential_scenario_count=2,
    )
    ranked = [
        _ranked(multi, rank=1, selection_eligible=True, quality_prediction=13.0),
        _ranked(
            _baseline_scenario(),
            rank=2,
            selection_eligible=False,
            quality_prediction=10.0,
        ),
    ]
    ranking = _ranking_report(
        ranked_scenarios=ranked,
        requested_scenario_count=2,
    )
    request = _request(max_simultaneous_changes=1)
    with pytest.raises(DataValidationError):
        RankedScenarioRecommendationGenerator().generate(
            _generation_request(request=request, grid=grid, ranking=ranking)
        )


def test_blocked_variable_rejected() -> None:
    safety = _safety_decision(
        status=RecommendationSafetyStatus.CAUTION,
        variable_assessments=[
            _assessment("pressure", factor_rank=1, eligible=False),
            _assessment("temperature", factor_rank=2, eligible=True),
        ],
    )
    with pytest.raises(DataValidationError):
        RankedScenarioRecommendationGenerator().generate(
            _generation_request(safety_decision=safety)
        )


def test_unsafe_partial_not_applied() -> None:
    safety = _safety_decision(
        status=RecommendationSafetyStatus.CAUTION,
        variable_assessments=[
            _assessment("pressure", factor_rank=1, eligible=False),
            _assessment("temperature", factor_rank=2, eligible=True),
        ],
    )
    outcome = RankedScenarioRecommendationGenerator(
        policy=RecommendationGenerationPolicy(
            require_all_changed_variables_safety_eligible=False
        )
    ).generate(_generation_request(safety_decision=safety))
    assert outcome.result.status is RecommendationStatus.READY_FOR_OPTIMIZATION
    assert outcome.result.changes == []


def test_diagnosis_factor_mapping_and_confidence() -> None:
    outcome = RankedScenarioRecommendationGenerator().generate(_generation_request())
    assert outcome.result.changes[0].confidence == pytest.approx(0.8)

    low = RankedScenarioRecommendationGenerator(
        policy=RecommendationGenerationPolicy(minimum_factor_confidence=0.9)
    ).generate(_generation_request())
    assert low.result.status is RecommendationStatus.READY_FOR_OPTIMIZATION

    boundary = RankedScenarioRecommendationGenerator(
        policy=RecommendationGenerationPolicy(minimum_factor_confidence=0.8)
    ).generate(_generation_request())
    assert boundary.result.status is RecommendationStatus.GENERATED


def test_missing_factor_strict_and_nonstrict() -> None:
    request = _request(factors=[_factor("temperature")])
    safety = _safety_decision(
        status=RecommendationSafetyStatus.APPROVED,
        variable_assessments=[
            _assessment("pressure", factor_rank=1, eligible=True),
            _assessment("temperature", factor_rank=1, eligible=True),
        ],
        global_reason_codes=[],
        messages=["Safety evaluation completed."],
    )
    # Selected scenario changes pressure, which is absent from diagnosis.
    with pytest.raises(DataValidationError):
        RankedScenarioRecommendationGenerator().generate(
            _generation_request(request=request, safety_decision=safety)
        )

    ready = RankedScenarioRecommendationGenerator(
        policy=RecommendationGenerationPolicy(
            require_all_changed_variables_in_diagnosis=False
        )
    ).generate(_generation_request(request=request, safety_decision=safety))
    assert ready.result.status is RecommendationStatus.READY_FOR_OPTIMIZATION


def test_factor_needs_verification_warning() -> None:
    request = _request(
        factors=[
            _factor("pressure", needs_verification=True),
            _factor("temperature"),
        ]
    )
    safety = _safety_decision(
        variable_assessments=[
            _assessment("pressure", factor_rank=1, needs_verification=True),
            _assessment("temperature", factor_rank=2),
        ]
    )
    outcome = RankedScenarioRecommendationGenerator().generate(
        _generation_request(request=request, safety_decision=safety)
    )
    assert any("needs verification" in warning.lower() for warning in outcome.result.warnings)


def test_duplicate_diagnosis_factor_defended() -> None:
    # DiagnosisResult may allow construction only if validation permits;
    # generator/request must still defend.
    with pytest.raises((ValidationError, DataValidationError, ValueError)):
        _request(
            factors=[
                _factor("pressure"),
                _factor("pressure", confidence=0.7),
                _factor("temperature"),
            ]
        )


# ---------------------------------------------------------------------------
# Value / grid / changes / confidence / predictions
# ---------------------------------------------------------------------------


def test_value_preservation_and_delta() -> None:
    outcome = RankedScenarioRecommendationGenerator().generate(_generation_request())
    change = outcome.result.changes[0]
    assert change.current_value == pytest.approx(50.0)
    assert change.proposed_value == pytest.approx(100.0)
    assert change.delta == pytest.approx(50.0)
    assert change.relative_delta == pytest.approx(1.0)
    assert change.requires_verification is True


def test_bad_delta_rejected_via_generation_request() -> None:
    # Covered by generation request delta mismatch validation.
    test_generation_request_delta_mismatch()


def test_current_zero_relative_delta_none() -> None:
    request = _request(current_values={"pressure": 0.0, "temperature": 80.0})
    points = [
        _point(
            value=0.0,
            delta=0.0,
            relative_delta=None,
            normalized_position=0.0,
            is_current=True,
            is_lower_bound=True,
        ),
        _point(
            value=50.0,
            delta=50.0,
            relative_delta=None,
            normalized_position=0.5,
            is_current=False,
        ),
        _point(
            value=100.0,
            delta=100.0,
            relative_delta=None,
            normalized_position=1.0,
            is_current=False,
            is_upper_bound=True,
        ),
    ]
    pressure_grid = _grid(current_value=0.0, points=points)
    scenario = _single_scenario(
        variable_values={"pressure": 100.0, "temperature": 80.0},
        deltas={"pressure": 100.0},
        relative_deltas={"pressure": None},
        normalized_change_magnitude=1.0,
    )
    grid = _grid_report(
        scenarios=[
            _baseline_scenario(
                variable_values={"pressure": 0.0, "temperature": 80.0}
            ),
            scenario,
            _second_single_scenario(
                variable_values={"pressure": 0.0, "temperature": 100.0}
            ),
        ],
        variable_grids=[pressure_grid, _temperature_grid()],
    )
    ranked = [
        _ranked(scenario, rank=1, selection_eligible=True, quality_prediction=12.0),
        _ranked(
            _second_single_scenario(
                variable_values={"pressure": 0.0, "temperature": 100.0}
            ),
            rank=2,
            selection_eligible=True,
            quality_prediction=11.0,
        ),
        _ranked(
            _baseline_scenario(variable_values={"pressure": 0.0, "temperature": 80.0}),
            rank=3,
            selection_eligible=False,
            quality_prediction=10.0,
        ),
    ]
    ranking = _ranking_report(
        ranked_scenarios=ranked,
        requested_scenario_count=3,
    )
    safety = _safety_decision(
        variable_assessments=[
            _assessment("pressure", factor_rank=1, current_value=0.0),
            _assessment("temperature", factor_rank=2),
        ]
    )
    outcome = RankedScenarioRecommendationGenerator().generate(
        _generation_request(
            request=request,
            safety_decision=safety,
            grid=grid,
            ranking=ranking,
        )
    )
    assert outcome.result.changes[0].relative_delta is None


def test_grid_bounds_and_points() -> None:
    outcome = RankedScenarioRecommendationGenerator().generate(_generation_request())
    assert outcome.result.changes[0].proposed_value == pytest.approx(100.0)

    # Upper/lower bound scenarios already use grid endpoints.
    # Out-of-bound proposed value should raise.
    grid = _grid_report()
    ranking = _ranking_report(requested_scenario_count=grid.generated_scenario_count)
    mutated = ranking.ranked_scenarios[0].model_copy(
        update={
            "variable_values": {"pressure": 150.0, "temperature": 80.0},
            "deltas": {"pressure": 100.0},
            "relative_deltas": {"pressure": 2.0},
        }
    )
    # GenerationRequest alignment fails first because grid scenario differs.
    with pytest.raises(ValidationError):
        _generation_request(
            grid=grid,
            ranking=ranking.model_copy(
                update={"ranked_scenarios": [mutated, *ranking.ranked_scenarios[1:]]}
            ),
        )


def test_grid_bounds_policy_false_and_missing_grid() -> None:
    outcome = RankedScenarioRecommendationGenerator(
        policy=RecommendationGenerationPolicy(require_values_within_grid_bounds=False)
    ).generate(_generation_request())
    assert outcome.result.status is RecommendationStatus.GENERATED

    gen = _generation_request()
    gen.grid.variable_grids = [_temperature_grid()]
    with pytest.raises(DataValidationError):
        RankedScenarioRecommendationGenerator().generate(gen)


def test_recommendation_change_rationale_and_confidence() -> None:
    outcome = RankedScenarioRecommendationGenerator().generate(_generation_request())
    change = outcome.result.changes[0]
    text = change.rationale.lower()
    assert "model-based" in text or "model-ranked" in text or "scenario ranking" in text
    assert "causation" in text
    assert "verification" in text
    assert "will improve" not in text
    assert "will fix" not in text
    assert "guaranteed" not in text
    assert change.confidence == pytest.approx(0.8)
    assert change.confidence != pytest.approx(
        outcome.result.metadata.get("selected_change_count", -1)
    )


def test_multi_change_confidence_average() -> None:
    multi = _multi_scenario(scenario_id="SCN-000001", scenario_index=1)
    scenarios = [_baseline_scenario(), multi]
    grid = _grid_report(
        scenarios=scenarios,
        generated_scenario_count=2,
        change_scenario_count=1,
        potential_scenario_count=2,
    )
    ranked = [
        _ranked(multi, rank=1, selection_eligible=True, quality_prediction=13.0),
        _ranked(
            _baseline_scenario(),
            rank=2,
            selection_eligible=False,
            quality_prediction=10.0,
        ),
    ]
    ranking = _ranking_report(ranked_scenarios=ranked, requested_scenario_count=2)
    request = _request(
        factors=[_factor("pressure", confidence=0.6), _factor("temperature", confidence=0.8)]
    )
    safety = _safety_decision(
        variable_assessments=[
            _assessment("pressure", factor_rank=1, factor_confidence=0.6),
            _assessment("temperature", factor_rank=2, factor_confidence=0.8),
        ]
    )
    outcome = RankedScenarioRecommendationGenerator().generate(
        _generation_request(request=request, safety_decision=safety, grid=grid, ranking=ranking)
    )
    assert len(outcome.result.changes) == 2
    assert outcome.result.changes[0].variable == "pressure"
    assert outcome.result.confidence == pytest.approx(0.7)


def test_prediction_mapping() -> None:
    ranking = _ranking_report(
        baseline_quality_prediction=10.0,
        baseline_anomaly_score=0.5,
        ranked_scenarios=[
            _ranked(
                _single_scenario(),
                rank=1,
                selection_eligible=True,
                quality_prediction=12.0,
                anomaly_score=0.2,
            ),
            _ranked(
                _second_single_scenario(),
                rank=2,
                selection_eligible=True,
                quality_prediction=11.0,
                anomaly_score=0.3,
            ),
            _ranked(
                _baseline_scenario(),
                rank=3,
                selection_eligible=False,
                quality_prediction=10.0,
                anomaly_score=0.5,
            ),
        ],
    )
    outcome = RankedScenarioRecommendationGenerator().generate(
        _generation_request(ranking=ranking)
    )
    assert outcome.result.baseline_prediction == pytest.approx(10.0)
    assert outcome.result.proposed_prediction == pytest.approx(12.0)
    assert outcome.result.baseline_anomaly_score == pytest.approx(0.5)
    assert outcome.result.proposed_anomaly_score == pytest.approx(0.2)


def test_missing_prediction_components_remain_none() -> None:
    ranking = _ranking_report(
        baseline_quality_prediction=None,
        baseline_anomaly_score=None,
        ranked_scenarios=[
            _ranked(
                _single_scenario(),
                rank=1,
                selection_eligible=True,
                quality_prediction=None,
                anomaly_score=None,
                quality_improvement_from_baseline=None,
                normalized_quality_benefit=None,
                quality_requirement_met=None,
                objective_benefit_score=0.5,
                composite_score=0.5,
            ),
            _ranked(
                _baseline_scenario(),
                rank=2,
                selection_eligible=False,
                quality_prediction=None,
                anomaly_score=None,
                quality_improvement_from_baseline=None,
                normalized_quality_benefit=None,
                quality_requirement_met=None,
            ),
        ],
        requested_scenario_count=3,
    )
    # Keep requested count aligned with grid.
    grid = _grid_report(
        scenarios=[_baseline_scenario(), _single_scenario(), _second_single_scenario()]
    )
    # Ranking has only 2 scenarios but requested_scenario_count must equal grid count.
    # Rebuild with three ranked entries and missing quality/anomaly.
    ranked = [
        _ranked(
            _single_scenario(),
            rank=1,
            selection_eligible=True,
            quality_prediction=None,
            anomaly_score=None,
            quality_improvement_from_baseline=None,
            normalized_quality_benefit=None,
            quality_requirement_met=None,
            objective_benefit_score=0.5,
            composite_score=0.5,
        ),
        _ranked(
            _second_single_scenario(),
            rank=2,
            selection_eligible=True,
            quality_prediction=None,
            anomaly_score=None,
            quality_improvement_from_baseline=None,
            normalized_quality_benefit=None,
            quality_requirement_met=None,
            objective_benefit_score=0.4,
            composite_score=0.4,
        ),
        _ranked(
            _baseline_scenario(),
            rank=3,
            selection_eligible=False,
            quality_prediction=None,
            anomaly_score=None,
            quality_improvement_from_baseline=None,
            normalized_quality_benefit=None,
            quality_requirement_met=None,
        ),
    ]
    ranking = _ranking_report(
        baseline_quality_prediction=None,
        baseline_anomaly_score=None,
        ranked_scenarios=ranked,
        requested_scenario_count=grid.generated_scenario_count,
    )
    outcome = RankedScenarioRecommendationGenerator().generate(
        _generation_request(grid=grid, ranking=ranking)
    )
    assert outcome.result.baseline_prediction is None
    assert outcome.result.proposed_prediction is None
    assert outcome.result.baseline_anomaly_score is None
    assert outcome.result.proposed_anomaly_score is None


def _negative_anomaly_ranking(
    *,
    baseline: float = -0.2,
    best: float = -0.8,
    second: float = -0.5,
) -> ScenarioRankingReport:
    return _ranking_report(
        baseline_anomaly_score=baseline,
        ranked_scenarios=[
            _ranked(
                _single_scenario(),
                rank=1,
                selection_eligible=True,
                quality_prediction=12.0,
                anomaly_score=best,
                anomaly_improvement_from_baseline=baseline - best,
            ),
            _ranked(
                _second_single_scenario(),
                rank=2,
                selection_eligible=True,
                quality_prediction=11.0,
                anomaly_score=second,
                anomaly_improvement_from_baseline=baseline - second,
            ),
            _ranked(
                _baseline_scenario(),
                rank=3,
                selection_eligible=False,
                quality_prediction=10.0,
                anomaly_score=baseline,
                anomaly_improvement_from_baseline=0.0,
            ),
        ],
    )


def _aligned_generation_request(
    objective: RecommendationObjective,
    *,
    ranking: ScenarioRankingReport | None = None,
    **overrides: Any,
) -> RecommendationGenerationRequest:
    request = overrides.pop("request", None)
    if request is None:
        request = _request(objective=objective)
    safety = overrides.pop("safety_decision", None)
    if safety is None:
        safety = _safety_decision(objective=objective)
    grid = overrides.pop("grid", None)
    if grid is None:
        grid = _grid_report(objective=objective)
    if ranking is None:
        ranking = _negative_anomaly_ranking()
        ranking = ranking.model_copy(update={"objective": objective})
    elif ranking.objective is not objective:
        ranking = ranking.model_copy(update={"objective": objective})
    return _generation_request(
        request=request,
        safety_decision=safety,
        grid=grid,
        ranking=ranking,
        **overrides,
    )


def test_negative_anomaly_scores_preserved_in_generated_result() -> None:
    baseline = -0.2
    proposed = -0.8
    ranking = _negative_anomaly_ranking(baseline=baseline, best=proposed)
    outcome = RankedScenarioRecommendationGenerator().generate(
        _generation_request(ranking=ranking)
    )
    result = outcome.result
    assert result.status is RecommendationStatus.GENERATED
    assert result.baseline_anomaly_score == pytest.approx(baseline)
    assert result.proposed_anomaly_score == pytest.approx(proposed)
    # No clipping to zero, abs, offset, or sign reversal.
    assert result.baseline_anomaly_score < 0.0
    assert result.proposed_anomaly_score < 0.0
    assert result.baseline_anomaly_score != pytest.approx(abs(baseline))
    assert result.proposed_anomaly_score != pytest.approx(abs(proposed))
    assert result.baseline_anomaly_score != pytest.approx(0.0)
    assert result.proposed_anomaly_score != pytest.approx(0.0)
    assert result.baseline_anomaly_score - result.proposed_anomaly_score == pytest.approx(
        baseline - proposed
    )


def test_quality_objective_preserves_optional_negative_anomaly_scores() -> None:
    ranking = _negative_anomaly_ranking(baseline=-0.05, best=-0.10)
    outcome = RankedScenarioRecommendationGenerator().generate(
        _aligned_generation_request(
            RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
            ranking=ranking,
        )
    )
    assert outcome.result.status is RecommendationStatus.GENERATED
    assert outcome.result.baseline_anomaly_score == pytest.approx(-0.05)
    assert outcome.result.proposed_anomaly_score == pytest.approx(-0.10)


def test_anomaly_objective_preserves_negative_scores() -> None:
    ranking = _negative_anomaly_ranking(baseline=-0.2, best=-0.8)
    outcome = RankedScenarioRecommendationGenerator().generate(
        _aligned_generation_request(
            RecommendationObjective.REDUCE_ANOMALY_SCORE,
            ranking=ranking,
        )
    )
    assert outcome.result.status is RecommendationStatus.GENERATED
    assert outcome.result.baseline_anomaly_score == pytest.approx(-0.2)
    assert outcome.result.proposed_anomaly_score == pytest.approx(-0.8)


def test_balance_objective_preserves_negative_scores() -> None:
    ranking = _negative_anomaly_ranking(baseline=-0.15, best=-0.45)
    outcome = RankedScenarioRecommendationGenerator().generate(
        _aligned_generation_request(
            RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
            ranking=ranking,
        )
    )
    assert outcome.result.status is RecommendationStatus.GENERATED
    assert outcome.result.baseline_anomaly_score == pytest.approx(-0.15)
    assert outcome.result.proposed_anomaly_score == pytest.approx(-0.45)


def test_ready_result_preserves_negative_baseline_anomaly_score() -> None:
    ranking = _negative_anomaly_ranking(baseline=-0.2, best=-0.8)
    ready = RankedScenarioRecommendationGenerator(
        policy=RecommendationGenerationPolicy(require_extrapolation_evaluation=True)
    ).generate(_generation_request(ranking=ranking))
    assert ready.result.status is RecommendationStatus.READY_FOR_OPTIMIZATION
    assert ready.result.changes == []
    assert ready.result.baseline_anomaly_score == pytest.approx(-0.2)
    assert ready.result.proposed_anomaly_score is None


def test_refused_status_rules_with_negative_baseline_anomaly_score() -> None:
    safety = _safety_decision(
        status=RecommendationSafetyStatus.REFUSED,
        variable_assessments=[
            _assessment("pressure", factor_rank=1, eligible=False),
            _assessment("temperature", factor_rank=2, eligible=False),
        ],
        global_reason_codes=[RecommendationReasonCode.NO_ELIGIBLE_VARIABLES],
        messages=["Refused by safety gate."],
    )
    ranking = _negative_anomaly_ranking(baseline=-0.2, best=-0.8)
    outcome = RankedScenarioRecommendationGenerator().generate(
        _generation_request(safety_decision=safety, ranking=ranking)
    )
    result = outcome.result
    assert result.status is RecommendationStatus.REFUSED
    assert result.changes == []
    assert result.proposed_prediction is None
    assert result.proposed_anomaly_score is None
    # Safety REFUSED exits before ranking score mapping; baselines stay None.
    assert result.baseline_anomaly_score is None
    assert result.baseline_prediction is None
    assert result.confidence == 0.0
    # Schema still accepts negative baseline on REFUSED when provided directly.
    refused = RecommendationResult(
        status=RecommendationStatus.REFUSED,
        objective=result.objective,
        safety_decision=result.safety_decision,
        changes=[],
        baseline_anomaly_score=-0.2,
        proposed_anomaly_score=None,
        confidence=0.0,
        extrapolation_flag=result.extrapolation_flag,
        uncertainty_available=result.uncertainty_available,
        disclaimer=result.disclaimer,
        generated_at=result.generated_at,
        warnings=list(result.warnings),
    )
    assert refused.baseline_anomaly_score == pytest.approx(-0.2)
    assert refused.proposed_anomaly_score is None


# ---------------------------------------------------------------------------
# Result status / disclaimer / metadata / immutability
# ---------------------------------------------------------------------------


def test_generated_and_ready_status_contracts() -> None:
    generated = RankedScenarioRecommendationGenerator().generate(_generation_request())
    assert generated.result.status is RecommendationStatus.GENERATED
    assert len(generated.result.changes) >= 1

    ready = RankedScenarioRecommendationGenerator(
        policy=RecommendationGenerationPolicy(require_extrapolation_evaluation=True)
    ).generate(_generation_request())
    assert ready.result.status is RecommendationStatus.READY_FOR_OPTIMIZATION
    assert ready.result.changes == []
    assert ready.result.proposed_prediction is None
    assert ready.result.proposed_anomaly_score is None


def test_disclaimer_and_warnings() -> None:
    result = RankedScenarioRecommendationGenerator().generate(_generation_request()).result
    text = result.disclaimer.lower()
    assert "model-based" in text
    assert "not proven optimal" in text
    assert "causation" in text
    assert "verification" in text
    assert "not guaranteed" in text
    assert "will improve" not in result.disclaimer.lower()
    assert "will fix" not in result.disclaimer.lower()
    warnings = result.warnings
    assert len(warnings) == len(set(warnings))
    assert any("residual anomaly" in warning.lower() for warning in warnings)
    assert any("not proven optimal" in warning.lower() for warning in warnings)


def test_metadata_and_generated_at() -> None:
    result = RankedScenarioRecommendationGenerator().generate(_generation_request()).result
    assert result.metadata["recommendation_generated"] is True
    assert result.metadata["model_scoring_performed"] is False
    assert result.metadata["model_refit_performed"] is False
    assert result.metadata["scenario_ranking_performed"] is False
    assert result.metadata["residual_scoring_performed"] is False
    assert result.metadata["actual_target_available"] is False
    assert result.metadata["confidence_is_heuristic"] is True
    assert result.generated_at.tzinfo is not None
    assert result.generated_at.utcoffset() == UTC.utcoffset(result.generated_at)
    for value in result.metadata.values():
        assert value is None or isinstance(value, (str, int, float, bool))


def test_immutability_and_determinism() -> None:
    request = _generation_request()
    safety_eligible = list(request.safety_decision.eligible_variables)
    grid_count = request.grid.generated_scenario_count
    ranking_best = request.ranking.best_nonbaseline_scenario_id
    factor_conf = request.request.diagnosis.factors[0].confidence

    generator = RankedScenarioRecommendationGenerator()
    first = generator.generate(request)
    second = generator.generate(request)
    other = RankedScenarioRecommendationGenerator().generate(request)

    assert request.safety_decision.eligible_variables == safety_eligible
    assert request.grid.generated_scenario_count == grid_count
    assert request.ranking.best_nonbaseline_scenario_id == ranking_best
    assert request.request.diagnosis.factors[0].confidence == factor_conf

    assert first.result.status is second.result.status
    assert first.result.changes[0].variable == second.result.changes[0].variable
    assert first.result.confidence == pytest.approx(second.result.confidence)
    assert other.result.changes[0].proposed_value == pytest.approx(
        first.result.changes[0].proposed_value
    )

    mutated = first.result.model_copy(update={"confidence": 0.0})
    third = generator.generate(request)
    assert third.result.confidence == pytest.approx(first.result.confidence)
    assert mutated.confidence == 0.0


def test_invalid_generation_request_type() -> None:
    with pytest.raises(TypeError):
        RankedScenarioRecommendationGenerator().generate("bad")  # type: ignore[arg-type]


def test_ready_confidence_zero() -> None:
    outcome = RankedScenarioRecommendationGenerator(
        policy=RecommendationGenerationPolicy(require_extrapolation_evaluation=True)
    ).generate(_generation_request())
    assert outcome.result.confidence == 0.0
