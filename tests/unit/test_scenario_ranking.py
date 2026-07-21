"""Unit tests for candidate scenario ranking (Step 9E)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.exceptions import DataValidationError
from process_intelligence.recommendation import (
    CandidateGridReport,
    CandidateGridStatus,
    CandidateScenario,
    CandidateScenarioRanker,
    CandidateScenarioScore,
    CandidateScenarioType,
    CandidateValuePoint,
    ConstraintResolutionStatus,
    QualityOptimizationDirection,
    RankedCandidateScenario,
    RecommendationObjective,
    RecommendationSafetyStatus,
    ScenarioRankingOutcome,
    ScenarioRankingPolicy,
    ScenarioRankingReport,
    ScenarioRankingRequest,
    ScenarioRankingStatus,
    ScenarioScoringReport,
    ScenarioScoringStatus,
    VariableCandidateGrid,
)
from process_intelligence.recommendation.scenario_ranking import (
    _WARNING_NO_GUARANTEE,
    _WARNING_NO_IMPROVEMENT,
    _WARNING_RELATIVE_HEURISTIC,
    _WARNING_VERIFICATION,
)

FEATURE_COLUMNS = ["pressure", "temperature", "humidity"]


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


def _ready_grid_report(**overrides: Any) -> CandidateGridReport:
    scenarios = overrides.pop(
        "scenarios",
        [_baseline_scenario(), _single_scenario(), _second_single_scenario()],
    )
    objective = overrides.pop(
        "objective",
        RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
    )
    payload: dict[str, Any] = {
        "status": CandidateGridStatus.READY,
        "objective": objective,
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


def _refused_grid_report(**overrides: Any) -> CandidateGridReport:
    payload: dict[str, Any] = {
        "status": CandidateGridStatus.REFUSED,
        "objective": RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        "safety_status": RecommendationSafetyStatus.REFUSED,
        "resolution_status": ConstraintResolutionStatus.REFUSED,
        "candidate_variables": [],
        "variable_grids": [],
        "scenarios": [],
        "baseline_scenario_id": None,
        "potential_scenario_count": 0,
        "generated_scenario_count": 0,
        "change_scenario_count": 0,
        "truncated_scenario_count": 0,
        "maximum_scenarios": 500,
        "effective_combination_limit": 0,
        "generated_at": datetime(2026, 7, 21, 15, 0, tzinfo=UTC),
        "warnings": ["Model scoring was not performed."],
        "metadata": {"model_scoring_performed": False},
    }
    payload.update(overrides)
    return CandidateGridReport(**payload)


def _scenario_score(**overrides: Any) -> CandidateScenarioScore:
    payload: dict[str, Any] = {
        "scenario_id": "SCN-000000",
        "scenario_index": 0,
        "scenario_type": CandidateScenarioType.BASELINE,
        "change_count": 0,
        "normalized_change_magnitude": 0.0,
        "quality_prediction": 10.0,
        "anomaly_score": -0.05,
        "quality_delta_from_baseline": 0.0,
        "anomaly_score_delta_from_baseline": 0.0,
        "quality_scored": True,
        "anomaly_scored": True,
        "warnings": [],
    }
    payload.update(overrides)
    return CandidateScenarioScore(**payload)


def _scoring_report(
    *,
    objective: RecommendationObjective = RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
    scores: list[CandidateScenarioScore] | None = None,
    **overrides: Any,
) -> ScenarioScoringReport:
    if scores is None:
        scores = [
            _scenario_score(),
            _scenario_score(
                scenario_id="SCN-000001",
                scenario_index=1,
                scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
                change_count=1,
                normalized_change_magnitude=0.5,
                quality_prediction=12.0,
                anomaly_score=-0.10,
                quality_delta_from_baseline=2.0,
                anomaly_score_delta_from_baseline=-0.05,
            ),
            _scenario_score(
                scenario_id="SCN-000002",
                scenario_index=2,
                scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
                change_count=1,
                normalized_change_magnitude=0.2,
                quality_prediction=11.0,
                anomaly_score=-0.08,
                quality_delta_from_baseline=1.0,
                anomaly_score_delta_from_baseline=-0.03,
            ),
        ]
    needs_quality = objective in {
        RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
    }
    needs_anomaly = objective in {
        RecommendationObjective.REDUCE_ANOMALY_SCORE,
        RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
    }
    payload: dict[str, Any] = {
        "status": ScenarioScoringStatus.SCORED,
        "objective": objective,
        "task": AnalysisTask.REGRESSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": "quality" if needs_quality else None,
        "quality_model_name": "Linear Regression" if needs_quality else None,
        "quality_estimator_key": "linear_regression" if needs_quality else None,
        "anomaly_model_name": "Isolation Forest" if needs_anomaly else None,
        "anomaly_estimator_key": "isolation_forest" if needs_anomaly else None,
        "baseline_scenario_id": "SCN-000000",
        "scores": scores,
        "requested_scenario_count": len(scores),
        "scored_scenario_count": len(scores),
        "quality_scored_count": sum(1 for item in scores if item.quality_scored),
        "anomaly_scored_count": sum(1 for item in scores if item.anomaly_scored),
        "prediction_seconds": 0.01,
        "anomaly_scoring_seconds": 0.02,
        "total_seconds": 0.04,
        "evaluated_at": datetime(2026, 7, 21, 16, 0, tzinfo=UTC),
        "warnings": ["Scenarios remain unranked"],
        "metadata": {"batch": 1},
    }
    payload.update(overrides)
    if "requested_scenario_count" not in overrides:
        payload["requested_scenario_count"] = len(payload["scores"])
    if "scored_scenario_count" not in overrides:
        payload["scored_scenario_count"] = len(payload["scores"])
    if "quality_scored_count" not in overrides:
        payload["quality_scored_count"] = sum(
            1 for item in payload["scores"] if item.quality_scored
        )
    if "anomaly_scored_count" not in overrides:
        payload["anomaly_scored_count"] = sum(
            1 for item in payload["scores"] if item.anomaly_scored
        )
    return ScenarioScoringReport(**payload)


def _ranking_request(**overrides: Any) -> ScenarioRankingRequest:
    objective = overrides.pop(
        "objective",
        RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
    )
    grid = overrides.pop("grid", None)
    scoring = overrides.pop("scoring", None)
    if grid is None:
        grid = _ready_grid_report(objective=objective)
    if scoring is None:
        scoring = _scoring_report(objective=objective)
    payload: dict[str, Any] = {
        "grid": grid,
        "scoring": scoring,
        "quality_direction": QualityOptimizationDirection.MAXIMIZE,
        "quality_target": None,
        "metadata": {"source": "unit_test"},
    }
    if objective is RecommendationObjective.REDUCE_ANOMALY_SCORE:
        payload["quality_direction"] = None
    payload.update(overrides)
    return ScenarioRankingRequest(**payload)


def _ranked_scenario(**overrides: Any) -> RankedCandidateScenario:
    payload: dict[str, Any] = {
        "rank": 1,
        "scenario_id": "SCN-000000",
        "scenario_index": 0,
        "scenario_type": CandidateScenarioType.BASELINE,
        "variable_values": {"pressure": 50.0, "temperature": 80.0},
        "changed_variables": [],
        "deltas": {},
        "relative_deltas": {},
        "change_count": 0,
        "normalized_change_magnitude": 0.0,
        "quality_prediction": 10.0,
        "anomaly_score": -0.05,
        "quality_improvement_from_baseline": 0.0,
        "anomaly_improvement_from_baseline": None,
        "normalized_quality_benefit": 0.0,
        "normalized_anomaly_benefit": None,
        "objective_benefit_score": 0.0,
        "normalized_change_cost": 0.0,
        "change_penalty": 0.0,
        "composite_score": 0.0,
        "quality_requirement_met": True,
        "anomaly_requirement_met": None,
        "objective_requirements_met": True,
        "selection_eligible": False,
        "warnings": ["scenario is the unchanged baseline"],
    }
    payload.update(overrides)
    return RankedCandidateScenario(**payload)


def _ranking_report(**overrides: Any) -> ScenarioRankingReport:
    ranked = overrides.pop(
        "ranked_scenarios",
        [
            _ranked_scenario(),
            _ranked_scenario(
                rank=2,
                scenario_id="SCN-000001",
                scenario_index=1,
                scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
                variable_values={"pressure": 100.0, "temperature": 80.0},
                changed_variables=["pressure"],
                deltas={"pressure": 50.0},
                relative_deltas={"pressure": 1.0},
                change_count=1,
                normalized_change_magnitude=0.5,
                quality_prediction=12.0,
                anomaly_score=-0.10,
                quality_improvement_from_baseline=2.0,
                normalized_quality_benefit=1.0,
                objective_benefit_score=1.0,
                normalized_change_cost=1.0,
                change_penalty=0.1,
                composite_score=0.9,
                quality_requirement_met=True,
                objective_requirements_met=True,
                selection_eligible=True,
                warnings=[],
            ),
        ],
    )
    eligible = sum(1 for item in ranked if item.selection_eligible)
    payload: dict[str, Any] = {
        "status": ScenarioRankingStatus.RANKED,
        "objective": RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        "quality_direction": QualityOptimizationDirection.MAXIMIZE,
        "quality_target": None,
        "baseline_scenario_id": "SCN-000000",
        "baseline_quality_prediction": 10.0,
        "baseline_anomaly_score": -0.05,
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
        "requested_scenario_count": len(ranked),
        "evaluated_scenario_count": len(ranked),
        "returned_scenario_count": len(ranked),
        "eligible_scenario_count": eligible,
        "ineligible_scenario_count": max(0, len(ranked) - 1 - eligible),
        "truncated_scenario_count": 0,
        "evaluated_at": datetime(2026, 7, 21, 17, 0, tzinfo=UTC),
        "warnings": [_WARNING_RELATIVE_HEURISTIC],
        "metadata": {"scenario_ranking_performed": True},
    }
    payload.update(overrides)
    return ScenarioRankingReport(**payload)


# --- Enum / Policy ---


def test_quality_optimization_direction_values() -> None:
    assert list(QualityOptimizationDirection) == [
        QualityOptimizationDirection.MAXIMIZE,
        QualityOptimizationDirection.MINIMIZE,
        QualityOptimizationDirection.TARGET,
    ]


def test_scenario_ranking_status_values() -> None:
    assert list(ScenarioRankingStatus) == [
        ScenarioRankingStatus.RANKED,
        ScenarioRankingStatus.PARTIAL,
        ScenarioRankingStatus.NO_IMPROVEMENT,
        ScenarioRankingStatus.REFUSED,
    ]


def test_no_extra_enum_members() -> None:
    assert len(QualityOptimizationDirection) == 3
    assert len(ScenarioRankingStatus) == 4


def test_default_policy() -> None:
    policy = ScenarioRankingPolicy()
    assert policy.quality_weight == 0.5
    assert policy.anomaly_weight == 0.5
    assert policy.change_penalty_weight == pytest.approx(0.10)
    assert policy.minimum_quality_improvement == 0.0
    assert policy.minimum_anomaly_improvement == 0.0
    assert policy.minimum_composite_advantage == pytest.approx(1e-12)
    assert policy.allow_required_component_worsening is False
    assert policy.allow_partial_scoring_report is False
    assert policy.prefer_fewer_changes is True
    assert policy.prefer_smaller_changes is True
    assert policy.require_baseline_scenario is True
    assert policy.include_ineligible_scenarios is True
    assert policy.maximum_ranked_scenarios == 5000


@pytest.mark.parametrize(
    "field,value",
    [
        ("quality_weight", -0.1),
        ("anomaly_weight", -0.1),
        ("quality_weight", 1.1),
        ("anomaly_weight", 1.1),
        ("quality_weight", True),
        ("anomaly_weight", False),
        ("quality_weight", float("nan")),
        ("anomaly_weight", float("inf")),
        ("change_penalty_weight", -0.01),
        ("change_penalty_weight", 1.5),
        ("change_penalty_weight", True),
        ("minimum_quality_improvement", -1.0),
        ("minimum_anomaly_improvement", -0.1),
        ("minimum_composite_advantage", -1e-12),
        ("minimum_quality_improvement", True),
        ("minimum_composite_advantage", float("nan")),
        ("maximum_ranked_scenarios", 0),
        ("maximum_ranked_scenarios", True),
        ("prefer_fewer_changes", 1),
        ("include_ineligible_scenarios", "true"),
    ],
)
def test_policy_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        ScenarioRankingPolicy(**{field: value})


def test_policy_rejects_both_weights_zero() -> None:
    with pytest.raises(ValidationError):
        ScenarioRankingPolicy(quality_weight=0.0, anomaly_weight=0.0)


def test_policy_round_trip() -> None:
    policy = ScenarioRankingPolicy(quality_weight=0.7, anomaly_weight=0.3)
    restored = ScenarioRankingPolicy.model_validate(policy.model_dump())
    assert restored == policy


# --- Request ---


def test_request_quality_maximize() -> None:
    request = _ranking_request(
        quality_direction=QualityOptimizationDirection.MAXIMIZE,
    )
    assert request.quality_direction is QualityOptimizationDirection.MAXIMIZE
    assert request.quality_target is None


def test_request_quality_minimize() -> None:
    request = _ranking_request(
        quality_direction=QualityOptimizationDirection.MINIMIZE,
    )
    assert request.quality_direction is QualityOptimizationDirection.MINIMIZE


def test_request_quality_target() -> None:
    request = _ranking_request(
        quality_direction=QualityOptimizationDirection.TARGET,
        quality_target=10.5,
    )
    assert request.quality_target == pytest.approx(10.5)


def test_request_anomaly_only() -> None:
    request = _ranking_request(
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        quality_direction=None,
    )
    assert request.quality_direction is None
    assert request.grid.objective is RecommendationObjective.REDUCE_ANOMALY_SCORE


def test_request_balance() -> None:
    request = _ranking_request(
        objective=RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
        quality_direction=QualityOptimizationDirection.MAXIMIZE,
    )
    assert request.scoring.objective is RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY


def test_request_objective_mismatch() -> None:
    grid = _ready_grid_report(
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
    )
    scoring = _scoring_report(
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
    )
    with pytest.raises(ValidationError):
        ScenarioRankingRequest(
            grid=grid,
            scoring=scoring,
            quality_direction=None,
        )


def test_request_scenario_count_mismatch() -> None:
    grid = _ready_grid_report(
        scenarios=[_baseline_scenario(), _single_scenario()],
    )
    scoring = _scoring_report()
    with pytest.raises(ValidationError):
        ScenarioRankingRequest(
            grid=grid,
            scoring=scoring,
            quality_direction=QualityOptimizationDirection.MAXIMIZE,
        )


def test_request_scenario_id_mismatch() -> None:
    grid = _ready_grid_report()
    scoring = _scoring_report()
    # Bypass score model construction constraints to force an ID mismatch at
    # the request boundary while keeping indexes contiguous.
    mismatched = scoring.model_copy(deep=True)
    score = mismatched.scores[1]
    object.__setattr__(score, "scenario_id", "SCN-000009")
    with pytest.raises((ValidationError, ValueError)):
        # Reconstruct through request validation path.
        ScenarioRankingRequest.model_validate(
            {
                "grid": grid.model_dump(),
                "scoring": {
                    **mismatched.model_dump(),
                    "scores": [
                        mismatched.scores[0].model_dump(),
                        {
                            **mismatched.scores[1].model_dump(),
                            "scenario_id": "SCN-000009",
                            "scenario_index": 1,
                        },
                        mismatched.scores[2].model_dump(),
                    ],
                },
                "quality_direction": "MAXIMIZE",
            }
        )


def test_request_scenario_index_mismatch() -> None:
    scoring = _scoring_report(
        scores=[
            _scenario_score(),
            _scenario_score(
                scenario_id="SCN-000001",
                scenario_index=1,
                scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
                change_count=1,
                normalized_change_magnitude=0.5,
                quality_prediction=12.0,
                quality_delta_from_baseline=2.0,
            ),
            _scenario_score(
                scenario_id="SCN-000002",
                scenario_index=2,
                scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
                change_count=1,
                normalized_change_magnitude=0.2,
                quality_prediction=11.0,
                quality_delta_from_baseline=1.0,
            ),
        ],
    )
    # Force type mismatch via model_construct bypass is not available; mutate
    # through a mismatched change_count instead covered elsewhere.
    bad_scores = [
        _scenario_score(),
        _scenario_score(
            scenario_id="SCN-000001",
            scenario_index=1,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.5,
            quality_prediction=12.0,
            quality_delta_from_baseline=2.0,
        ),
        _scenario_score(
            scenario_id="SCN-000002",
            scenario_index=2,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.2,
            quality_prediction=11.0,
            quality_delta_from_baseline=1.0,
        ),
    ]
    scoring = _scoring_report(scores=bad_scores)
    grid = _ready_grid_report()
    # Swap index identity by using mismatched scenario_type
    mismatched = scoring.model_copy(deep=True)
    object.__setattr__(
        mismatched.scores[1],
        "__dict__",
        {
            **mismatched.scores[1].__dict__,
            "scenario_type": CandidateScenarioType.MULTI_VARIABLE,
            "change_count": 2,
        },
    )
    # Pydantic models may be frozen-ish; rebuild request with constructed mismatch
    with pytest.raises(ValidationError):
        ScenarioRankingRequest(
            grid=grid,
            scoring=_scoring_report(
                scores=[
                    _scenario_score(),
                    _scenario_score(
                        scenario_id="SCN-000001",
                        scenario_index=1,
                        scenario_type=CandidateScenarioType.MULTI_VARIABLE,
                        change_count=2,
                        normalized_change_magnitude=0.5,
                        quality_prediction=12.0,
                        quality_delta_from_baseline=2.0,
                    ),
                    _scenario_score(
                        scenario_id="SCN-000002",
                        scenario_index=2,
                        scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
                        change_count=1,
                        normalized_change_magnitude=0.2,
                        quality_prediction=11.0,
                        quality_delta_from_baseline=1.0,
                    ),
                ],
            ),
            quality_direction=QualityOptimizationDirection.MAXIMIZE,
        )


def test_request_change_count_mismatch() -> None:
    with pytest.raises(ValidationError):
        _ranking_request(
            scoring=_scoring_report(
                scores=[
                    _scenario_score(),
                    _scenario_score(
                        scenario_id="SCN-000001",
                        scenario_index=1,
                        scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
                        change_count=1,
                        normalized_change_magnitude=0.5,
                        quality_prediction=12.0,
                        quality_delta_from_baseline=2.0,
                    ),
                    _scenario_score(
                        scenario_id="SCN-000002",
                        scenario_index=2,
                        scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
                        change_count=0,
                        normalized_change_magnitude=0.2,
                        quality_prediction=11.0,
                        quality_delta_from_baseline=1.0,
                    ),
                ],
            ),
        )


def test_request_magnitude_mismatch() -> None:
    with pytest.raises(ValidationError):
        _ranking_request(
            scoring=_scoring_report(
                scores=[
                    _scenario_score(),
                    _scenario_score(
                        scenario_id="SCN-000001",
                        scenario_index=1,
                        scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
                        change_count=1,
                        normalized_change_magnitude=0.9,
                        quality_prediction=12.0,
                        quality_delta_from_baseline=2.0,
                    ),
                    _scenario_score(
                        scenario_id="SCN-000002",
                        scenario_index=2,
                        scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
                        change_count=1,
                        normalized_change_magnitude=0.2,
                        quality_prediction=11.0,
                        quality_delta_from_baseline=1.0,
                    ),
                ],
            ),
        )


def test_request_score_order_mismatch() -> None:
    with pytest.raises(ValidationError):
        _scoring_report(
            scores=[
                _scenario_score(
                    scenario_id="SCN-000001",
                    scenario_index=1,
                    scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
                    change_count=1,
                    normalized_change_magnitude=0.5,
                    quality_prediction=12.0,
                    quality_delta_from_baseline=2.0,
                ),
                _scenario_score(),
            ],
        )


def test_request_missing_quality_direction() -> None:
    with pytest.raises(ValidationError):
        ScenarioRankingRequest(
            grid=_ready_grid_report(),
            scoring=_scoring_report(),
            quality_direction=None,
        )


def test_request_target_missing_value() -> None:
    with pytest.raises(ValidationError):
        _ranking_request(
            quality_direction=QualityOptimizationDirection.TARGET,
            quality_target=None,
        )


def test_request_non_target_with_target_rejected() -> None:
    with pytest.raises(ValidationError):
        _ranking_request(
            quality_direction=QualityOptimizationDirection.MAXIMIZE,
            quality_target=10.0,
        )


@pytest.mark.parametrize("value", [True, float("nan"), float("inf")])
def test_request_target_invalid_numeric(value: object) -> None:
    with pytest.raises(ValidationError):
        _ranking_request(
            quality_direction=QualityOptimizationDirection.TARGET,
            quality_target=value,
        )


def test_request_metadata_scalar_only() -> None:
    with pytest.raises(ValidationError):
        _ranking_request(metadata={"frame": {"a": 1}})


def test_request_mutable_state_independence() -> None:
    metadata = {"k": 1}
    request = _ranking_request(metadata=metadata)
    metadata["k"] = 2
    assert request.metadata["k"] == 1


def test_request_round_trip() -> None:
    request = _ranking_request()
    restored = ScenarioRankingRequest.model_validate(request.model_dump())
    assert restored.grid.objective == request.grid.objective
    assert restored.scoring.requested_scenario_count == (
        request.scoring.requested_scenario_count
    )


# --- RankedCandidateScenario ---


def test_ranked_baseline_ok() -> None:
    item = _ranked_scenario()
    assert item.selection_eligible is False
    assert item.composite_score == 0.0


def test_ranked_eligible_ok() -> None:
    item = _ranked_scenario(
        rank=2,
        scenario_id="SCN-000001",
        scenario_index=1,
        scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
        variable_values={"pressure": 100.0, "temperature": 80.0},
        changed_variables=["pressure"],
        deltas={"pressure": 50.0},
        relative_deltas={"pressure": 1.0},
        change_count=1,
        normalized_change_magnitude=0.5,
        quality_improvement_from_baseline=2.0,
        normalized_quality_benefit=1.0,
        objective_benefit_score=1.0,
        normalized_change_cost=1.0,
        change_penalty=0.1,
        composite_score=0.9,
        selection_eligible=True,
        warnings=[],
    )
    assert item.selection_eligible is True


def test_ranked_ineligible_ok() -> None:
    item = _ranked_scenario(
        rank=2,
        scenario_id="SCN-000001",
        scenario_index=1,
        scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
        variable_values={"pressure": 100.0, "temperature": 80.0},
        changed_variables=["pressure"],
        deltas={"pressure": 50.0},
        relative_deltas={"pressure": 1.0},
        change_count=1,
        normalized_change_magnitude=0.5,
        quality_improvement_from_baseline=-1.0,
        normalized_quality_benefit=0.0,
        objective_benefit_score=0.0,
        quality_requirement_met=False,
        objective_requirements_met=False,
        selection_eligible=False,
        warnings=["quality requirement not met"],
    )
    assert item.objective_requirements_met is False


@pytest.mark.parametrize("rank", [0, True])
def test_ranked_rank_invalid(rank: object) -> None:
    with pytest.raises(ValidationError):
        _ranked_scenario(rank=rank)


def test_ranked_nan_inf_rejected() -> None:
    with pytest.raises(ValidationError):
        _ranked_scenario(composite_score=float("nan"))
    with pytest.raises(ValidationError):
        _ranked_scenario(objective_benefit_score=float("inf"))


def test_ranked_normalized_benefit_range() -> None:
    with pytest.raises(ValidationError):
        _ranked_scenario(normalized_quality_benefit=1.1)


def test_ranked_normalized_cost_range() -> None:
    with pytest.raises(ValidationError):
        _ranked_scenario(normalized_change_cost=-0.1)


def test_ranked_baseline_selection_eligible_rejected() -> None:
    with pytest.raises(ValidationError):
        _ranked_scenario(selection_eligible=True)


def test_ranked_flag_relationship() -> None:
    with pytest.raises(ValidationError):
        _ranked_scenario(
            rank=2,
            scenario_id="SCN-000001",
            scenario_index=1,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            variable_values={"pressure": 100.0, "temperature": 80.0},
            changed_variables=["pressure"],
            deltas={"pressure": 50.0},
            relative_deltas={"pressure": 1.0},
            change_count=1,
            normalized_change_magnitude=0.5,
            quality_requirement_met=False,
            objective_requirements_met=True,
            selection_eligible=True,
            warnings=[],
        )


def test_ranked_warning_duplicates_rejected() -> None:
    with pytest.raises(ValidationError):
        _ranked_scenario(warnings=["a", "a"])


def test_ranked_round_trip() -> None:
    item = _ranked_scenario()
    restored = RankedCandidateScenario.model_validate(item.model_dump())
    assert restored == item


# --- Report / Outcome ---


def test_report_ranked_ok() -> None:
    report = _ranking_report()
    assert report.status is ScenarioRankingStatus.RANKED
    assert report.best_nonbaseline_scenario_id == "SCN-000001"


def test_report_partial_ok() -> None:
    report = _ranking_report(status=ScenarioRankingStatus.PARTIAL)
    assert report.status is ScenarioRankingStatus.PARTIAL


def test_report_no_improvement_ok() -> None:
    ranked = [_ranked_scenario()]
    report = _ranking_report(
        status=ScenarioRankingStatus.NO_IMPROVEMENT,
        ranked_scenarios=ranked,
        best_scenario_id="SCN-000000",
        best_nonbaseline_scenario_id=None,
        baseline_is_top_ranked=True,
        eligible_scenario_count=0,
        ineligible_scenario_count=0,
        requested_scenario_count=1,
        evaluated_scenario_count=1,
        returned_scenario_count=1,
    )
    assert report.status is ScenarioRankingStatus.NO_IMPROVEMENT


def test_report_refused_ok() -> None:
    report = ScenarioRankingReport(
        status=ScenarioRankingStatus.REFUSED,
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        quality_direction=QualityOptimizationDirection.MAXIMIZE,
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
        evaluated_at=datetime(2026, 7, 21, 17, 0, tzinfo=UTC),
        warnings=[],
        metadata={},
    )
    assert report.status is ScenarioRankingStatus.REFUSED


def test_report_duplicate_rank_rejected() -> None:
    with pytest.raises(ValidationError):
        _ranking_report(
            ranked_scenarios=[
                _ranked_scenario(rank=1),
                _ranked_scenario(
                    rank=1,
                    scenario_id="SCN-000001",
                    scenario_index=1,
                    scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
                    variable_values={"pressure": 100.0, "temperature": 80.0},
                    changed_variables=["pressure"],
                    deltas={"pressure": 50.0},
                    relative_deltas={"pressure": 1.0},
                    change_count=1,
                    normalized_change_magnitude=0.5,
                    selection_eligible=True,
                    warnings=[],
                ),
            ],
        )


def test_report_noncontiguous_rank_rejected() -> None:
    with pytest.raises(ValidationError):
        _ranking_report(
            ranked_scenarios=[
                _ranked_scenario(rank=1),
                _ranked_scenario(
                    rank=3,
                    scenario_id="SCN-000001",
                    scenario_index=1,
                    scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
                    variable_values={"pressure": 100.0, "temperature": 80.0},
                    changed_variables=["pressure"],
                    deltas={"pressure": 50.0},
                    relative_deltas={"pressure": 1.0},
                    change_count=1,
                    normalized_change_magnitude=0.5,
                    selection_eligible=True,
                    warnings=[],
                ),
            ],
        )


def test_report_rank_order_rejected() -> None:
    first = _ranked_scenario(
        rank=2,
        scenario_id="SCN-000001",
        scenario_index=1,
        scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
        variable_values={"pressure": 100.0, "temperature": 80.0},
        changed_variables=["pressure"],
        deltas={"pressure": 50.0},
        relative_deltas={"pressure": 1.0},
        change_count=1,
        normalized_change_magnitude=0.5,
        selection_eligible=True,
        warnings=[],
    )
    second = _ranked_scenario(rank=1)
    with pytest.raises(ValidationError):
        _ranking_report(ranked_scenarios=[first, second])


def test_report_duplicate_scenario_id_rejected() -> None:
    with pytest.raises(ValidationError):
        _ranking_report(
            ranked_scenarios=[
                _ranked_scenario(),
                _ranked_scenario(
                    rank=2,
                    scenario_id="SCN-000000",
                    scenario_index=0,
                    selection_eligible=False,
                ),
            ],
        )


def test_report_naive_datetime_rejected() -> None:
    with pytest.raises(ValidationError):
        _ranking_report(evaluated_at=datetime(2026, 7, 21, 17, 0))


def test_report_metadata_rejects_nested() -> None:
    with pytest.raises(ValidationError):
        _ranking_report(metadata={"scores": [1, 2]})


def test_report_round_trip() -> None:
    report = _ranking_report()
    restored = ScenarioRankingReport.model_validate(report.model_dump())
    assert restored.status == report.status


def test_outcome_frozen_and_slots() -> None:
    outcome = ScenarioRankingOutcome(report=_ranking_report())
    assert outcome.__slots__ == ("report",)
    with pytest.raises(FrozenInstanceError):
        outcome.report = _ranking_report(status=ScenarioRankingStatus.PARTIAL)  # type: ignore[misc]


# --- Ranker ---


def test_ranker_default() -> None:
    ranker = CandidateScenarioRanker()
    outcome = ranker.rank(_ranking_request())
    assert outcome.report.status is ScenarioRankingStatus.RANKED


def test_ranker_custom_policy() -> None:
    policy = ScenarioRankingPolicy(change_penalty_weight=0.0)
    ranker = CandidateScenarioRanker(policy=policy)
    assert ranker.get_metadata()["change_penalty_weight"] == 0.0


def test_ranker_policy_type_error() -> None:
    with pytest.raises(TypeError):
        CandidateScenarioRanker(policy={"quality_weight": 0.5})  # type: ignore[arg-type]


def test_ranker_external_policy_immutability() -> None:
    policy = ScenarioRankingPolicy(quality_weight=0.6, anomaly_weight=0.4)
    ranker = CandidateScenarioRanker(policy=policy)
    mutated = policy.model_copy(update={"quality_weight": 0.1})
    assert ranker.get_metadata()["quality_weight"] == pytest.approx(0.6)
    assert mutated.quality_weight == pytest.approx(0.1)


def test_ranker_state_isolation() -> None:
    a = CandidateScenarioRanker(
        policy=ScenarioRankingPolicy(quality_weight=0.8, anomaly_weight=0.2),
    )
    b = CandidateScenarioRanker(
        policy=ScenarioRankingPolicy(quality_weight=0.2, anomaly_weight=0.8),
    )
    assert a.get_metadata()["quality_weight"] != b.get_metadata()["quality_weight"]


def test_ranker_metadata_scalar_and_independent() -> None:
    ranker = CandidateScenarioRanker()
    meta = ranker.get_metadata()
    assert meta["performs_model_scoring"] is False
    assert meta["performs_model_refit"] is False
    assert meta["performs_scenario_ranking"] is True
    assert meta["generates_recommendation"] is False
    assert meta["ranking_is_relative_heuristic"] is True
    meta["quality_weight"] = -1
    assert ranker.get_metadata()["quality_weight"] == pytest.approx(0.5)


# --- Structured refusal ---


def test_refuse_grid_refused() -> None:
    scoring = ScenarioScoringReport(
        status=ScenarioScoringStatus.REFUSED,
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        task=AnalysisTask.REGRESSION,
        feature_columns=list(FEATURE_COLUMNS),
        target_column="quality",
        quality_model_name=None,
        quality_estimator_key=None,
        anomaly_model_name=None,
        anomaly_estimator_key=None,
        baseline_scenario_id=None,
        scores=[],
        requested_scenario_count=0,
        scored_scenario_count=0,
        quality_scored_count=0,
        anomaly_scored_count=0,
        prediction_seconds=0.0,
        anomaly_scoring_seconds=0.0,
        total_seconds=0.0,
        evaluated_at=datetime(2026, 7, 21, 16, 0, tzinfo=UTC),
        warnings=[],
        metadata={},
    )
    request = ScenarioRankingRequest(
        grid=_refused_grid_report(),
        scoring=scoring,
        quality_direction=QualityOptimizationDirection.MAXIMIZE,
    )
    outcome = CandidateScenarioRanker().rank(request)
    assert outcome.report.status is ScenarioRankingStatus.REFUSED
    assert outcome.report.ranked_scenarios == []


def test_refuse_scoring_refused() -> None:
    grid = _ready_grid_report()
    scoring = ScenarioScoringReport(
        status=ScenarioScoringStatus.REFUSED,
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        task=AnalysisTask.REGRESSION,
        feature_columns=list(FEATURE_COLUMNS),
        target_column="quality",
        quality_model_name=None,
        quality_estimator_key=None,
        anomaly_model_name=None,
        anomaly_estimator_key=None,
        baseline_scenario_id=None,
        scores=[],
        requested_scenario_count=len(grid.scenarios),
        scored_scenario_count=0,
        quality_scored_count=0,
        anomaly_scored_count=0,
        prediction_seconds=0.0,
        anomaly_scoring_seconds=0.0,
        total_seconds=0.0,
        evaluated_at=datetime(2026, 7, 21, 16, 0, tzinfo=UTC),
        warnings=[],
        metadata={},
    )
    request = ScenarioRankingRequest(
        grid=grid,
        scoring=scoring,
        quality_direction=QualityOptimizationDirection.MAXIMIZE,
    )
    outcome = CandidateScenarioRanker().rank(request)
    assert outcome.report.status is ScenarioRankingStatus.REFUSED
    assert not isinstance(outcome, Exception)


def test_refuse_partial_by_default() -> None:
    scores = [
        _scenario_score(
            anomaly_score=None,
            anomaly_score_delta_from_baseline=None,
            anomaly_scored=False,
        ),
        _scenario_score(
            scenario_id="SCN-000001",
            scenario_index=1,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.5,
            quality_prediction=12.0,
            quality_delta_from_baseline=2.0,
            anomaly_score=None,
            anomaly_score_delta_from_baseline=None,
            anomaly_scored=False,
        ),
        _scenario_score(
            scenario_id="SCN-000002",
            scenario_index=2,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.2,
            quality_prediction=None,
            quality_delta_from_baseline=None,
            quality_scored=False,
            anomaly_score=None,
            anomaly_score_delta_from_baseline=None,
            anomaly_scored=False,
        ),
    ]
    scoring = _scoring_report(
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        status=ScenarioScoringStatus.PARTIAL,
        scores=scores,
        anomaly_model_name=None,
        anomaly_estimator_key=None,
    )
    request = _ranking_request(scoring=scoring)
    outcome = CandidateScenarioRanker().rank(request)
    assert outcome.report.status is ScenarioRankingStatus.REFUSED
    assert outcome.report.metadata["model_scoring_performed"] is False


def test_refuse_baseline_required_output_missing() -> None:
    scores = [
        _scenario_score(
            quality_prediction=None,
            quality_delta_from_baseline=None,
            quality_scored=False,
            anomaly_score=None,
            anomaly_score_delta_from_baseline=None,
            anomaly_scored=False,
        ),
        _scenario_score(
            scenario_id="SCN-000001",
            scenario_index=1,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.5,
            quality_prediction=12.0,
            quality_delta_from_baseline=None,
            anomaly_score=None,
            anomaly_score_delta_from_baseline=None,
            anomaly_scored=False,
        ),
        _scenario_score(
            scenario_id="SCN-000002",
            scenario_index=2,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.2,
            quality_prediction=11.0,
            quality_delta_from_baseline=None,
            anomaly_score=None,
            anomaly_score_delta_from_baseline=None,
            anomaly_scored=False,
        ),
    ]
    scoring = _scoring_report(
        status=ScenarioScoringStatus.PARTIAL,
        scores=scores,
        anomaly_model_name=None,
        anomaly_estimator_key=None,
    )
    request = _ranking_request(scoring=scoring)
    outcome = CandidateScenarioRanker(
        policy=ScenarioRankingPolicy(allow_partial_scoring_report=True),
    ).rank(request)
    assert outcome.report.status is ScenarioRankingStatus.REFUSED


# --- Core ranking behavior ---


def test_maximize_and_minimize_and_target_improvements() -> None:
    ranker = CandidateScenarioRanker(
        policy=ScenarioRankingPolicy(change_penalty_weight=0.0),
    )
    maximize = ranker.rank(
        _ranking_request(quality_direction=QualityOptimizationDirection.MAXIMIZE),
    ).report
    by_id = {item.scenario_id: item for item in maximize.ranked_scenarios}
    assert by_id["SCN-000001"].quality_improvement_from_baseline == pytest.approx(2.0)
    assert by_id["SCN-000002"].quality_improvement_from_baseline == pytest.approx(1.0)

    # Minimize: lower quality is better; reverse predictions relative to baseline.
    scores = [
        _scenario_score(quality_prediction=10.0),
        _scenario_score(
            scenario_id="SCN-000001",
            scenario_index=1,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.5,
            quality_prediction=8.0,
            quality_delta_from_baseline=-2.0,
        ),
        _scenario_score(
            scenario_id="SCN-000002",
            scenario_index=2,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.2,
            quality_prediction=12.0,
            quality_delta_from_baseline=2.0,
        ),
    ]
    minimize = ranker.rank(
        _ranking_request(
            scoring=_scoring_report(scores=scores),
            quality_direction=QualityOptimizationDirection.MINIMIZE,
        ),
    ).report
    by_id = {item.scenario_id: item for item in minimize.ranked_scenarios}
    assert by_id["SCN-000001"].quality_improvement_from_baseline == pytest.approx(2.0)
    assert by_id["SCN-000002"].quality_improvement_from_baseline == pytest.approx(-2.0)

    target = ranker.rank(
        _ranking_request(
            quality_direction=QualityOptimizationDirection.TARGET,
            quality_target=10.0,
            scoring=_scoring_report(
                scores=[
                    _scenario_score(quality_prediction=8.0),
                    _scenario_score(
                        scenario_id="SCN-000001",
                        scenario_index=1,
                        scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
                        change_count=1,
                        normalized_change_magnitude=0.5,
                        quality_prediction=10.0,
                        quality_delta_from_baseline=2.0,
                    ),
                    _scenario_score(
                        scenario_id="SCN-000002",
                        scenario_index=2,
                        scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
                        change_count=1,
                        normalized_change_magnitude=0.2,
                        quality_prediction=6.0,
                        quality_delta_from_baseline=-2.0,
                    ),
                ],
            ),
        ),
    ).report
    by_id = {item.scenario_id: item for item in target.ranked_scenarios}
    assert by_id["SCN-000001"].quality_improvement_from_baseline == pytest.approx(2.0)
    assert by_id["SCN-000002"].quality_improvement_from_baseline == pytest.approx(-2.0)
    assert by_id["SCN-000000"].quality_improvement_from_baseline == pytest.approx(0.0)


def test_anomaly_improvement_and_negative_scores() -> None:
    scores = [
        _scenario_score(
            anomaly_score=-0.05,
            quality_prediction=None,
            quality_delta_from_baseline=None,
            quality_scored=False,
        ),
        _scenario_score(
            scenario_id="SCN-000001",
            scenario_index=1,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.5,
            quality_prediction=None,
            quality_delta_from_baseline=None,
            quality_scored=False,
            anomaly_score=-0.20,
            anomaly_score_delta_from_baseline=-0.15,
        ),
        _scenario_score(
            scenario_id="SCN-000002",
            scenario_index=2,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.2,
            quality_prediction=None,
            quality_delta_from_baseline=None,
            quality_scored=False,
            anomaly_score=0.10,
            anomaly_score_delta_from_baseline=0.15,
        ),
    ]
    request = _ranking_request(
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        grid=_ready_grid_report(
            objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        ),
        scoring=_scoring_report(
            objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
            scores=scores,
        ),
        quality_direction=None,
    )
    report = CandidateScenarioRanker(
        policy=ScenarioRankingPolicy(change_penalty_weight=0.0),
    ).rank(request).report
    by_id = {item.scenario_id: item for item in report.ranked_scenarios}
    assert by_id["SCN-000001"].anomaly_improvement_from_baseline == pytest.approx(0.15)
    assert by_id["SCN-000002"].anomaly_improvement_from_baseline == pytest.approx(-0.15)


def test_benefit_normalization_and_baseline_zero() -> None:
    report = CandidateScenarioRanker(
        policy=ScenarioRankingPolicy(change_penalty_weight=0.0),
    ).rank(_ranking_request()).report
    by_id = {item.scenario_id: item for item in report.ranked_scenarios}
    assert by_id["SCN-000000"].normalized_quality_benefit == pytest.approx(0.0)
    assert by_id["SCN-000001"].normalized_quality_benefit == pytest.approx(1.0)
    assert by_id["SCN-000002"].normalized_quality_benefit == pytest.approx(0.5)
    assert by_id["SCN-000000"].composite_score == pytest.approx(0.0)
    assert by_id["SCN-000000"].selection_eligible is False


def test_change_penalty_and_composite() -> None:
    report = CandidateScenarioRanker(
        policy=ScenarioRankingPolicy(change_penalty_weight=0.2),
    ).rank(_ranking_request()).report
    by_id = {item.scenario_id: item for item in report.ranked_scenarios}
    assert by_id["SCN-000000"].normalized_change_cost == pytest.approx(0.0)
    assert by_id["SCN-000001"].normalized_change_cost == pytest.approx(1.0)
    assert by_id["SCN-000001"].change_penalty == pytest.approx(0.2)
    assert by_id["SCN-000001"].composite_score == pytest.approx(0.8)
    assert by_id["SCN-000002"].normalized_change_cost == pytest.approx(0.4)


def test_balance_objective_weights() -> None:
    request = _ranking_request(
        objective=RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
        grid=_ready_grid_report(
            objective=RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
        ),
        scoring=_scoring_report(
            objective=RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
        ),
    )
    report = CandidateScenarioRanker(
        policy=ScenarioRankingPolicy(
            quality_weight=0.25,
            anomaly_weight=0.75,
            change_penalty_weight=0.0,
        ),
    ).rank(request).report
    top = next(item for item in report.ranked_scenarios if item.selection_eligible)
    assert 0.0 <= top.objective_benefit_score <= 1.0


def test_worsening_rejected_and_allowed() -> None:
    scores = [
        _scenario_score(),
        _scenario_score(
            scenario_id="SCN-000001",
            scenario_index=1,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.5,
            quality_prediction=8.0,
            quality_delta_from_baseline=-2.0,
        ),
        _scenario_score(
            scenario_id="SCN-000002",
            scenario_index=2,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.2,
            quality_prediction=8.0,
            quality_delta_from_baseline=-2.0,
        ),
    ]
    request = _ranking_request(scoring=_scoring_report(scores=scores))
    default = CandidateScenarioRanker().rank(request).report
    assert default.status is ScenarioRankingStatus.NO_IMPROVEMENT
    assert default.eligible_scenario_count == 0

    allowed = CandidateScenarioRanker(
        policy=ScenarioRankingPolicy(allow_required_component_worsening=True),
    ).rank(request).report
    # Still no positive raw improvement, so still ineligible.
    assert allowed.eligible_scenario_count == 0
    worsened = [
        item
        for item in default.ranked_scenarios
        if "required component worsened" in item.warnings
    ]
    assert worsened


def test_zero_improvement_not_selected() -> None:
    scores = [
        _scenario_score(),
        _scenario_score(
            scenario_id="SCN-000001",
            scenario_index=1,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.5,
            quality_prediction=10.0,
            quality_delta_from_baseline=0.0,
        ),
        _scenario_score(
            scenario_id="SCN-000002",
            scenario_index=2,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.2,
            quality_prediction=10.0,
            quality_delta_from_baseline=0.0,
        ),
    ]
    report = CandidateScenarioRanker().rank(
        _ranking_request(scoring=_scoring_report(scores=scores)),
    ).report
    assert report.status is ScenarioRankingStatus.NO_IMPROVEMENT
    assert report.best_nonbaseline_scenario_id is None


def test_deterministic_ranking_order() -> None:
    report = CandidateScenarioRanker(
        policy=ScenarioRankingPolicy(change_penalty_weight=0.0),
    ).rank(_ranking_request()).report
    ids = [item.scenario_id for item in report.ranked_scenarios]
    # Higher quality improvement first among eligible; baseline may not be top.
    eligible = [item for item in report.ranked_scenarios if item.selection_eligible]
    assert eligible[0].scenario_id == "SCN-000001"
    assert eligible[1].scenario_id == "SCN-000002"
    again = CandidateScenarioRanker(
        policy=ScenarioRankingPolicy(change_penalty_weight=0.0),
    ).rank(_ranking_request()).report
    assert [item.scenario_id for item in again.ranked_scenarios] == ids


def test_baseline_beats_worse_candidates() -> None:
    scores = [
        _scenario_score(),
        _scenario_score(
            scenario_id="SCN-000001",
            scenario_index=1,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.5,
            quality_prediction=5.0,
            quality_delta_from_baseline=-5.0,
        ),
        _scenario_score(
            scenario_id="SCN-000002",
            scenario_index=2,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.2,
            quality_prediction=4.0,
            quality_delta_from_baseline=-6.0,
        ),
    ]
    report = CandidateScenarioRanker(
        policy=ScenarioRankingPolicy(change_penalty_weight=0.1),
    ).rank(_ranking_request(scoring=_scoring_report(scores=scores))).report
    assert report.ranked_scenarios[0].scenario_type is CandidateScenarioType.BASELINE
    assert report.baseline_is_top_ranked is True


def test_include_and_exclude_ineligible() -> None:
    scores = [
        _scenario_score(),
        _scenario_score(
            scenario_id="SCN-000001",
            scenario_index=1,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.5,
            quality_prediction=12.0,
            quality_delta_from_baseline=2.0,
        ),
        _scenario_score(
            scenario_id="SCN-000002",
            scenario_index=2,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.2,
            quality_prediction=5.0,
            quality_delta_from_baseline=-5.0,
        ),
    ]
    request = _ranking_request(scoring=_scoring_report(scores=scores))
    included = CandidateScenarioRanker().rank(request).report
    assert len(included.ranked_scenarios) == 3
    excluded = CandidateScenarioRanker(
        policy=ScenarioRankingPolicy(include_ineligible_scenarios=False),
    ).rank(request).report
    assert all(
        item.selection_eligible or item.scenario_type is CandidateScenarioType.BASELINE
        for item in excluded.ranked_scenarios
    )
    assert any(
        item.scenario_type is CandidateScenarioType.BASELINE
        for item in excluded.ranked_scenarios
    )


def test_ranking_cap_preserves_baseline() -> None:
    scenarios = [
        _baseline_scenario(),
        _single_scenario(),
        _second_single_scenario(),
        _multi_scenario(),
    ]
    scores = [
        _scenario_score(),
        _scenario_score(
            scenario_id="SCN-000001",
            scenario_index=1,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.5,
            quality_prediction=12.0,
            quality_delta_from_baseline=2.0,
        ),
        _scenario_score(
            scenario_id="SCN-000002",
            scenario_index=2,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.2,
            quality_prediction=11.5,
            quality_delta_from_baseline=1.5,
        ),
        _scenario_score(
            scenario_id="SCN-000003",
            scenario_index=3,
            scenario_type=CandidateScenarioType.MULTI_VARIABLE,
            change_count=2,
            normalized_change_magnitude=0.7,
            quality_prediction=11.0,
            quality_delta_from_baseline=1.0,
        ),
    ]
    request = _ranking_request(
        grid=_ready_grid_report(scenarios=scenarios),
        scoring=_scoring_report(scores=scores),
    )
    report = CandidateScenarioRanker(
        policy=ScenarioRankingPolicy(maximum_ranked_scenarios=2),
    ).rank(request).report
    assert report.returned_scenario_count == 2
    assert report.truncated_scenario_count == 2
    assert any(
        item.scenario_type is CandidateScenarioType.BASELINE
        for item in report.ranked_scenarios
    )
    assert [item.rank for item in report.ranked_scenarios] == [1, 2]
    assert report.best_nonbaseline_scenario_id is not None
    assert report.status is ScenarioRankingStatus.RANKED

    only_baseline = CandidateScenarioRanker(
        policy=ScenarioRankingPolicy(maximum_ranked_scenarios=1),
    ).rank(request).report
    assert only_baseline.returned_scenario_count == 1
    assert (
        only_baseline.ranked_scenarios[0].scenario_type
        is CandidateScenarioType.BASELINE
    )
    assert only_baseline.best_nonbaseline_scenario_id is None
    assert only_baseline.status is ScenarioRankingStatus.NO_IMPROVEMENT


def test_partial_scoring_allowed() -> None:
    scores = [
        _scenario_score(
            anomaly_score=None,
            anomaly_score_delta_from_baseline=None,
            anomaly_scored=False,
        ),
        _scenario_score(
            scenario_id="SCN-000001",
            scenario_index=1,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.5,
            quality_prediction=12.0,
            quality_delta_from_baseline=2.0,
            anomaly_score=None,
            anomaly_score_delta_from_baseline=None,
            anomaly_scored=False,
        ),
        _scenario_score(
            scenario_id="SCN-000002",
            scenario_index=2,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.2,
            quality_prediction=None,
            quality_delta_from_baseline=None,
            quality_scored=False,
            anomaly_score=None,
            anomaly_score_delta_from_baseline=None,
            anomaly_scored=False,
        ),
    ]
    scoring = _scoring_report(
        status=ScenarioScoringStatus.PARTIAL,
        scores=scores,
        anomaly_model_name=None,
        anomaly_estimator_key=None,
    )
    report = CandidateScenarioRanker(
        policy=ScenarioRankingPolicy(
            allow_partial_scoring_report=True,
            change_penalty_weight=0.0,
        ),
    ).rank(_ranking_request(scoring=scoring)).report
    assert report.status is ScenarioRankingStatus.PARTIAL
    assert report.evaluated_scenario_count == 2
    assert report.best_nonbaseline_scenario_id == "SCN-000001"
    assert any("required model outputs" in warning for warning in report.warnings)


def test_value_preservation() -> None:
    report = CandidateScenarioRanker().rank(_ranking_request()).report
    grid_by_id = {item.scenario_id: item for item in _ready_grid_report().scenarios}
    score_by_id = {
        item.scenario_id: item for item in _scoring_report().scores
    }
    for item in report.ranked_scenarios:
        scenario = grid_by_id[item.scenario_id]
        score = score_by_id[item.scenario_id]
        assert item.variable_values == scenario.variable_values
        assert item.changed_variables == scenario.changed_variables
        assert item.deltas == scenario.deltas
        assert item.relative_deltas == scenario.relative_deltas
        assert item.scenario_type == scenario.scenario_type
        assert item.change_count == scenario.change_count
        assert item.normalized_change_magnitude == scenario.normalized_change_magnitude
        assert item.quality_prediction == score.quality_prediction
        assert item.anomaly_score == score.anomaly_score


def test_warnings_and_metadata_contracts() -> None:
    report = CandidateScenarioRanker().rank(_ranking_request()).report
    text = " | ".join(report.warnings)
    assert _WARNING_RELATIVE_HEURISTIC in report.warnings
    assert _WARNING_VERIFICATION in report.warnings
    assert _WARNING_NO_GUARANTEE in report.warnings
    assert "optimal process condition" not in text.lower()
    assert "proven best" not in text.lower()
    assert "guaranteed improvement" not in text.lower()
    assert "will fix" not in text.lower()
    assert report.metadata["model_scoring_performed"] is False
    assert report.metadata["model_refit_performed"] is False
    assert report.metadata["recommendation_generated"] is False
    assert report.metadata["baseline_preserved"] is True
    assert report.metadata["ranking_is_relative_heuristic"] is True
    assert report.metadata["association_or_prediction_not_causation"] is True
    for value in report.metadata.values():
        assert value is None or isinstance(value, (str, int, float, bool))


def test_no_improvement_warning_present() -> None:
    scores = [
        _scenario_score(),
        _scenario_score(
            scenario_id="SCN-000001",
            scenario_index=1,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.5,
            quality_prediction=9.0,
            quality_delta_from_baseline=-1.0,
        ),
        _scenario_score(
            scenario_id="SCN-000002",
            scenario_index=2,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.2,
            quality_prediction=8.0,
            quality_delta_from_baseline=-2.0,
        ),
    ]
    report = CandidateScenarioRanker().rank(
        _ranking_request(scoring=_scoring_report(scores=scores)),
    ).report
    assert _WARNING_NO_IMPROVEMENT in report.warnings


def test_immutability_and_determinism() -> None:
    request = _ranking_request()
    grid_before = request.grid.model_dump()
    scoring_before = request.scoring.model_dump()
    policy = ScenarioRankingPolicy(change_penalty_weight=0.05)
    policy_before = policy.model_dump()
    ranker = CandidateScenarioRanker(policy=policy)
    first = ranker.rank(request)
    second = ranker.rank(request)
    assert first.report.ranked_scenarios[0].scenario_id == (
        second.report.ranked_scenarios[0].scenario_id
    )
    assert [
        (item.scenario_id, item.composite_score)
        for item in first.report.ranked_scenarios
    ] == [
        (item.scenario_id, item.composite_score)
        for item in second.report.ranked_scenarios
    ]
    assert request.grid.model_dump() == grid_before
    assert request.scoring.model_dump() == scoring_before
    assert policy.model_dump() == policy_before
    mutated = first.report.model_copy(deep=True)
    mutated.ranked_scenarios[0].warnings.append("mutated")  # type: ignore[attr-defined]
    third = ranker.rank(request)
    assert "mutated" not in third.report.warnings
    assert third.report.eligible_scenario_count == first.report.eligible_scenario_count


def test_request_type_error() -> None:
    with pytest.raises(TypeError):
        CandidateScenarioRanker().rank({"grid": 1})  # type: ignore[arg-type]


def test_tie_break_fewer_and_smaller_changes() -> None:
    # Existing grid points: pressure=100 (mag 0.5) vs temperature=100 (mag 0.2).
    scenarios = [
        _baseline_scenario(),
        _single_scenario(),
        _second_single_scenario(),
    ]
    scores = [
        _scenario_score(),
        _scenario_score(
            scenario_id="SCN-000001",
            scenario_index=1,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.5,
            quality_prediction=12.0,
            quality_delta_from_baseline=2.0,
        ),
        _scenario_score(
            scenario_id="SCN-000002",
            scenario_index=2,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.2,
            quality_prediction=12.0,
            quality_delta_from_baseline=2.0,
        ),
    ]
    report = CandidateScenarioRanker(
        policy=ScenarioRankingPolicy(change_penalty_weight=0.0),
    ).rank(
        _ranking_request(
            grid=_ready_grid_report(scenarios=scenarios),
            scoring=_scoring_report(scores=scores),
        ),
    ).report
    eligible = [item for item in report.ranked_scenarios if item.selection_eligible]
    assert eligible[0].scenario_id == "SCN-000002"


def test_objective_weight_validation_error() -> None:
    with pytest.raises(DataValidationError):
        CandidateScenarioRanker(
            policy=ScenarioRankingPolicy(quality_weight=0.0, anomaly_weight=1.0),
        ).rank(_ranking_request())
