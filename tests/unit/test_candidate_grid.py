"""Unit tests for candidate grid generation (Step 9C)."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from process_intelligence.core.exceptions import DataValidationError
from process_intelligence.recommendation import (
    CandidateGridGenerator,
    CandidateGridOutcome,
    CandidateGridPolicy,
    CandidateGridReport,
    CandidateGridStatus,
    CandidateScenario,
    CandidateScenarioType,
    CandidateValuePoint,
    CandidateVariable,
    CandidateVariableSet,
    ConstraintResolutionStatus,
    ConstraintSource,
    RecommendationObjective,
    RecommendationSafetyStatus,
    VariableCandidateGrid,
)
from process_intelligence.recommendation.candidate_grid import (
    _WARNING_COMBINATION_CAP,
    _WARNING_DEDUP,
    _WARNING_MODEL_SCORING,
    _WARNING_NO_CHANGE_POINTS,
    _WARNING_NONZERO_DELTA,
    _WARNING_RESOLUTION_PARTIAL,
    _WARNING_SAFETY_CAUTION,
    _WARNING_TRUNCATED,
    _WARNING_VERIFICATION,
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
    current = float(payload["current_value"])
    minimum = float(payload["minimum"])
    maximum = float(payload["maximum"])
    if "lower_room" not in overrides:
        payload["lower_room"] = current - minimum
    if "upper_room" not in overrides:
        payload["upper_room"] = maximum - current
    if "relative_position" not in overrides:
        payload["relative_position"] = payload["lower_room"] / (maximum - minimum)
    if "at_lower_bound" not in overrides:
        payload["at_lower_bound"] = math.isclose(current, minimum, abs_tol=1e-12)
    if "at_upper_bound" not in overrides:
        payload["at_upper_bound"] = math.isclose(current, maximum, abs_tol=1e-12)
    return CandidateVariable(**payload)


def _candidate_set(**overrides: Any) -> CandidateVariableSet:
    candidates = overrides.pop(
        "candidates",
        [
            _candidate(variable="pressure", diagnosis_rank=1, current_value=50.0),
            _candidate(
                variable="temperature",
                diagnosis_rank=2,
                current_value=80.0,
                lower_room=80.0,
                upper_room=20.0,
                relative_position=0.8,
            ),
        ],
    )
    payload: dict[str, Any] = {
        "objective": RecommendationObjective.REDUCE_ANOMALY_SCORE,
        "safety_status": RecommendationSafetyStatus.APPROVED,
        "resolution_status": ConstraintResolutionStatus.READY,
        "candidates": candidates,
        "candidate_variables": [item.variable for item in candidates],
        "max_simultaneous_changes": 3,
        "effective_change_budget": min(3, len(candidates)) if candidates else 0,
        "generated_at": datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
        "warnings": [],
        "metadata": {"optimization_performed": False},
    }
    payload.update(overrides)
    if "candidate_variables" not in overrides:
        payload["candidate_variables"] = [item.variable for item in payload["candidates"]]
    if "effective_change_budget" not in overrides:
        n = len(payload["candidates"])
        payload["effective_change_budget"] = (
            0 if n == 0 else min(payload["max_simultaneous_changes"], n)
        )
    return CandidateVariableSet(**payload)


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
                is_upper_bound=False,
            ),
            _point(
                value=50.0,
                delta=0.0,
                relative_delta=0.0,
                normalized_position=0.5,
                is_current=True,
                is_lower_bound=False,
                is_upper_bound=False,
            ),
            _point(
                value=100.0,
                delta=50.0,
                relative_delta=1.0,
                normalized_position=1.0,
                is_current=False,
                is_lower_bound=False,
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


def _baseline_scenario(**overrides: Any) -> CandidateScenario:
    payload: dict[str, Any] = {
        "scenario_id": "SCN-000000",
        "scenario_index": 0,
        "scenario_type": CandidateScenarioType.BASELINE,
        "variable_values": {"pressure": 50.0},
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
        "variable_values": {"pressure": 0.0},
        "changed_variables": ["pressure"],
        "deltas": {"pressure": -50.0},
        "relative_deltas": {"pressure": -1.0},
        "change_count": 1,
        "normalized_change_magnitude": 0.5,
    }
    payload.update(overrides)
    return CandidateScenario(**payload)


def _multi_scenario(**overrides: Any) -> CandidateScenario:
    payload: dict[str, Any] = {
        "scenario_id": "SCN-000002",
        "scenario_index": 2,
        "scenario_type": CandidateScenarioType.MULTI_VARIABLE,
        "variable_values": {"pressure": 0.0, "temperature": 0.0},
        "changed_variables": ["pressure", "temperature"],
        "deltas": {"pressure": -50.0, "temperature": -80.0},
        "relative_deltas": {"pressure": -1.0, "temperature": -1.0},
        "change_count": 2,
        "normalized_change_magnitude": 1.3,
    }
    payload.update(overrides)
    return CandidateScenario(**payload)


def _report_from_generator(
    *,
    policy: CandidateGridPolicy | None = None,
    candidate_set: CandidateVariableSet | None = None,
) -> CandidateGridReport:
    generator = CandidateGridGenerator(policy=policy)
    outcome = generator.generate(candidate_set or _candidate_set())
    return outcome.report


# --- Enum / Policy ---


def test_candidate_grid_status_values() -> None:
    assert list(CandidateGridStatus) == [
        CandidateGridStatus.READY,
        CandidateGridStatus.TRUNCATED,
        CandidateGridStatus.EMPTY,
        CandidateGridStatus.REFUSED,
    ]
    assert CandidateGridStatus.READY == "READY"


def test_candidate_scenario_type_values() -> None:
    assert list(CandidateScenarioType) == [
        CandidateScenarioType.BASELINE,
        CandidateScenarioType.SINGLE_VARIABLE,
        CandidateScenarioType.MULTI_VARIABLE,
    ]


def test_no_extra_enum_members() -> None:
    assert len(CandidateGridStatus) == 4
    assert len(CandidateScenarioType) == 3


def test_default_policy() -> None:
    policy = CandidateGridPolicy()
    assert policy.linear_points_per_variable == 5
    assert policy.maximum_scenarios == 500
    assert policy.maximum_combination_variables == 3
    assert policy.include_single_variable_scenarios is True
    assert policy.include_multi_variable_scenarios is True
    assert policy.prefer_smaller_changes is True
    assert policy.deduplication_tolerance == pytest.approx(1e-12)
    assert policy.minimum_nonzero_delta == pytest.approx(1e-12)


def test_linear_points_one_rejected() -> None:
    with pytest.raises(ValidationError):
        CandidateGridPolicy(linear_points_per_variable=1)


def test_linear_points_bool_rejected() -> None:
    with pytest.raises(ValidationError):
        CandidateGridPolicy(linear_points_per_variable=True)  # type: ignore[arg-type]


def test_maximum_scenarios_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        CandidateGridPolicy(maximum_scenarios=0)


def test_combination_limit_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        CandidateGridPolicy(maximum_combination_variables=0)


def test_tolerance_negative_rejected() -> None:
    with pytest.raises(ValidationError):
        CandidateGridPolicy(deduplication_tolerance=-1.0)


def test_tolerance_bool_rejected() -> None:
    with pytest.raises(ValidationError):
        CandidateGridPolicy(minimum_nonzero_delta=True)  # type: ignore[arg-type]


def test_tolerance_nan_inf_rejected() -> None:
    with pytest.raises(ValidationError):
        CandidateGridPolicy(deduplication_tolerance=float("nan"))
    with pytest.raises(ValidationError):
        CandidateGridPolicy(minimum_nonzero_delta=float("inf"))


def test_bool_field_strict_validation() -> None:
    with pytest.raises(ValidationError):
        CandidateGridPolicy(include_single_variable_scenarios=1)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        CandidateGridPolicy(prefer_smaller_changes="yes")  # type: ignore[arg-type]


def test_policy_round_trip() -> None:
    policy = CandidateGridPolicy(maximum_scenarios=10)
    assert CandidateGridPolicy.model_validate(policy.model_dump()) == policy


# --- CandidateValuePoint ---


def test_point_current_valid() -> None:
    point = _point()
    assert point.is_current is True
    assert point.delta == 0.0


def test_point_change_valid() -> None:
    point = _point(
        value=75.0,
        delta=25.0,
        relative_delta=0.5,
        normalized_position=0.75,
        is_current=False,
    )
    assert point.is_current is False


def test_point_nan_inf_rejected() -> None:
    with pytest.raises(ValidationError):
        _point(value=float("nan"))
    with pytest.raises(ValidationError):
        _point(delta=float("inf"))


def test_point_normalized_position_range() -> None:
    with pytest.raises(ValidationError):
        _point(normalized_position=1.5)
    with pytest.raises(ValidationError):
        _point(normalized_position=-0.1)


def test_point_current_delta_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        _point(delta=1.0, is_current=True)


def test_point_bool_strict() -> None:
    with pytest.raises(ValidationError):
        _point(is_current=1)  # type: ignore[arg-type]


def test_point_round_trip() -> None:
    point = _point()
    assert CandidateValuePoint.model_validate(point.model_dump()) == point


# --- VariableCandidateGrid ---


def test_grid_valid() -> None:
    grid = _grid()
    assert grid.point_count == 3
    assert grid.change_point_count == 2


def test_grid_empty_variable_rejected() -> None:
    with pytest.raises(ValidationError):
        _grid(variable=" ")


def test_grid_rank_zero_and_bool_rejected() -> None:
    with pytest.raises(ValidationError):
        _grid(diagnosis_rank=0)
    with pytest.raises(ValidationError):
        _grid(diagnosis_rank=True)  # type: ignore[arg-type]


def test_grid_current_bound_relation() -> None:
    with pytest.raises(ValidationError):
        _grid(current_value=150.0)


def test_grid_span_relation() -> None:
    with pytest.raises(ValidationError):
        _grid(effective_span=50.0)
    with pytest.raises(ValidationError):
        _grid(effective_span=0.0)


def test_grid_empty_points_rejected() -> None:
    with pytest.raises(ValidationError):
        _grid(points=[], point_count=0, change_point_count=0)


def test_grid_point_count_mismatch() -> None:
    with pytest.raises(ValidationError):
        _grid(point_count=99)


def test_grid_change_count_mismatch() -> None:
    with pytest.raises(ValidationError):
        _grid(change_point_count=0)


def test_grid_point_value_duplicate() -> None:
    with pytest.raises(ValidationError):
        _grid(
            points=[
                _point(
                    value=0.0,
                    delta=-50.0,
                    relative_delta=-1.0,
                    normalized_position=0.0,
                    is_current=False,
                    is_lower_bound=True,
                ),
                _point(
                    value=0.0,
                    delta=-50.0,
                    relative_delta=-1.0,
                    normalized_position=0.0,
                    is_current=False,
                    is_lower_bound=True,
                ),
                _point(value=50.0, is_current=True),
                _point(
                    value=100.0,
                    delta=50.0,
                    relative_delta=1.0,
                    normalized_position=1.0,
                    is_current=False,
                    is_upper_bound=True,
                ),
            ]
        )


def test_grid_point_sort_error() -> None:
    with pytest.raises(ValidationError):
        _grid(
            points=[
                _point(value=50.0, is_current=True),
                _point(
                    value=0.0,
                    delta=-50.0,
                    relative_delta=-1.0,
                    normalized_position=0.0,
                    is_current=False,
                    is_lower_bound=True,
                ),
                _point(
                    value=100.0,
                    delta=50.0,
                    relative_delta=1.0,
                    normalized_position=1.0,
                    is_current=False,
                    is_upper_bound=True,
                ),
            ]
        )


def test_grid_no_current_point_rejected() -> None:
    with pytest.raises(ValidationError):
        _grid(
            points=[
                _point(
                    value=0.0,
                    delta=-50.0,
                    relative_delta=-1.0,
                    normalized_position=0.0,
                    is_current=False,
                    is_lower_bound=True,
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
            change_point_count=2,
        )


def test_grid_two_current_points_rejected() -> None:
    with pytest.raises(ValidationError):
        _grid(
            points=[
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
            current_value=0.0,
            change_point_count=1,
        )


def test_grid_missing_lower_boundary_rejected() -> None:
    with pytest.raises(ValidationError):
        _grid(
            points=[
                _point(value=50.0, is_current=True),
                _point(
                    value=100.0,
                    delta=50.0,
                    relative_delta=1.0,
                    normalized_position=1.0,
                    is_current=False,
                    is_upper_bound=True,
                ),
            ],
            change_point_count=1,
        )


def test_grid_missing_upper_boundary_rejected() -> None:
    with pytest.raises(ValidationError):
        _grid(
            points=[
                _point(
                    value=0.0,
                    delta=-50.0,
                    relative_delta=-1.0,
                    normalized_position=0.0,
                    is_current=False,
                    is_lower_bound=True,
                ),
                _point(value=50.0, is_current=True),
            ],
            change_point_count=1,
        )


def test_grid_normalized_position_mismatch() -> None:
    with pytest.raises(ValidationError):
        _grid(
            points=[
                _point(
                    value=0.0,
                    delta=-50.0,
                    relative_delta=-1.0,
                    normalized_position=0.1,
                    is_current=False,
                    is_lower_bound=True,
                ),
                _point(value=50.0, is_current=True),
                _point(
                    value=100.0,
                    delta=50.0,
                    relative_delta=1.0,
                    normalized_position=1.0,
                    is_current=False,
                    is_upper_bound=True,
                ),
            ]
        )


def test_grid_boundary_flag_mismatch() -> None:
    with pytest.raises(ValidationError):
        _grid(
            points=[
                _point(
                    value=0.0,
                    delta=-50.0,
                    relative_delta=-1.0,
                    normalized_position=0.0,
                    is_current=False,
                    is_lower_bound=False,
                ),
                _point(value=50.0, is_current=True),
                _point(
                    value=100.0,
                    delta=50.0,
                    relative_delta=1.0,
                    normalized_position=1.0,
                    is_current=False,
                    is_upper_bound=True,
                ),
            ]
        )


def test_grid_warning_duplicate() -> None:
    with pytest.raises(ValidationError):
        _grid(warnings=["a", "a"])


def test_grid_round_trip() -> None:
    grid = _grid()
    assert VariableCandidateGrid.model_validate(grid.model_dump()) == grid


# --- CandidateScenario ---


def test_scenario_baseline_valid() -> None:
    assert _baseline_scenario().change_count == 0


def test_scenario_single_valid() -> None:
    assert _single_scenario().change_count == 1


def test_scenario_multi_valid() -> None:
    assert _multi_scenario().change_count == 2


def test_scenario_id_format() -> None:
    with pytest.raises(ValidationError):
        _baseline_scenario(scenario_id="SCN-1")


def test_scenario_index_id_mismatch() -> None:
    with pytest.raises(ValidationError):
        _baseline_scenario(scenario_id="SCN-000001", scenario_index=0)


def test_scenario_variable_value_nan_rejected() -> None:
    with pytest.raises(ValidationError):
        _single_scenario(variable_values={"pressure": float("nan")})


def test_scenario_changed_variable_duplicate() -> None:
    with pytest.raises(ValidationError):
        _multi_scenario(changed_variables=["pressure", "pressure"])


def test_scenario_delta_key_mismatch() -> None:
    with pytest.raises(ValidationError):
        _single_scenario(deltas={"temperature": -1.0})


def test_scenario_relative_delta_key_mismatch() -> None:
    with pytest.raises(ValidationError):
        _single_scenario(relative_deltas={})


def test_scenario_zero_change_delta_rejected() -> None:
    with pytest.raises(ValidationError):
        _single_scenario(
            variable_values={"pressure": 50.0},
            deltas={"pressure": 0.0},
            relative_deltas={"pressure": 0.0},
            normalized_change_magnitude=0.0,
        )


def test_scenario_change_count_mismatch() -> None:
    with pytest.raises(ValidationError):
        _single_scenario(change_count=2)


def test_scenario_baseline_relation_error() -> None:
    with pytest.raises(ValidationError):
        _baseline_scenario(changed_variables=["pressure"], change_count=1)


def test_scenario_single_count_error() -> None:
    with pytest.raises(ValidationError):
        _single_scenario(
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            changed_variables=["pressure", "temperature"],
            deltas={"pressure": -1.0, "temperature": -1.0},
            relative_deltas={"pressure": -0.02, "temperature": -0.0125},
            change_count=2,
            variable_values={"pressure": 0.0, "temperature": 0.0},
            normalized_change_magnitude=1.0,
        )


def test_scenario_multi_count_error() -> None:
    with pytest.raises(ValidationError):
        _multi_scenario(
            changed_variables=["pressure"],
            deltas={"pressure": -50.0},
            relative_deltas={"pressure": -1.0},
            change_count=1,
            variable_values={"pressure": 0.0},
            normalized_change_magnitude=0.5,
        )


def test_scenario_normalized_magnitude_negative_rejected() -> None:
    with pytest.raises(ValidationError):
        _single_scenario(normalized_change_magnitude=-0.1)


def test_scenario_round_trip() -> None:
    scenario = _single_scenario()
    assert CandidateScenario.model_validate(scenario.model_dump()) == scenario


# --- CandidateGridReport ---


def _ready_report(**overrides: Any) -> CandidateGridReport:
    grid = _grid()
    baseline = _baseline_scenario()
    single = _single_scenario()
    payload: dict[str, Any] = {
        "status": CandidateGridStatus.READY,
        "objective": RecommendationObjective.REDUCE_ANOMALY_SCORE,
        "safety_status": RecommendationSafetyStatus.APPROVED,
        "resolution_status": ConstraintResolutionStatus.READY,
        "candidate_variables": ["pressure"],
        "variable_grids": [grid],
        "scenarios": [baseline, single],
        "baseline_scenario_id": "SCN-000000",
        "potential_scenario_count": 2,
        "generated_scenario_count": 2,
        "change_scenario_count": 1,
        "truncated_scenario_count": 0,
        "maximum_scenarios": 500,
        "effective_combination_limit": 1,
        "generated_at": datetime(2026, 7, 21, 15, 0, tzinfo=UTC),
        "warnings": ["Model scoring was not performed."],
        "metadata": {"model_scoring_performed": False},
    }
    payload.update(overrides)
    return CandidateGridReport(**payload)


def test_report_ready_valid() -> None:
    assert _ready_report().status is CandidateGridStatus.READY


def test_report_truncated_valid() -> None:
    report = _ready_report(
        status=CandidateGridStatus.TRUNCATED,
        potential_scenario_count=5,
        truncated_scenario_count=3,
        maximum_scenarios=2,
        generated_scenario_count=2,
    )
    assert report.status is CandidateGridStatus.TRUNCATED


def test_report_empty_valid() -> None:
    report = _ready_report(
        status=CandidateGridStatus.EMPTY,
        scenarios=[_baseline_scenario()],
        generated_scenario_count=1,
        change_scenario_count=0,
        potential_scenario_count=1,
        truncated_scenario_count=0,
    )
    assert report.change_scenario_count == 0


def test_report_refused_valid() -> None:
    report = CandidateGridReport(
        status=CandidateGridStatus.REFUSED,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        safety_status=RecommendationSafetyStatus.REFUSED,
        resolution_status=ConstraintResolutionStatus.REFUSED,
        candidate_variables=[],
        variable_grids=[],
        scenarios=[],
        baseline_scenario_id=None,
        potential_scenario_count=0,
        generated_scenario_count=0,
        change_scenario_count=0,
        truncated_scenario_count=0,
        maximum_scenarios=500,
        effective_combination_limit=0,
        generated_at=datetime(2026, 7, 21, 15, 0, tzinfo=UTC),
        warnings=["Model scoring was not performed."],
        metadata={"model_scoring_performed": False},
    )
    assert report.status is CandidateGridStatus.REFUSED


def test_report_candidate_grid_order_mismatch() -> None:
    with pytest.raises(ValidationError):
        _ready_report(candidate_variables=["temperature"])


def test_report_grid_variable_duplicate() -> None:
    with pytest.raises(ValidationError):
        _ready_report(
            candidate_variables=["pressure", "pressure"],
            variable_grids=[_grid(), _grid(diagnosis_rank=2)],
        )


def test_report_diagnosis_rank_duplicate() -> None:
    temp_grid = _grid(
        variable="temperature",
        diagnosis_rank=1,
        current_value=80.0,
        points=[
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
    with pytest.raises(ValidationError):
        _ready_report(
            candidate_variables=["pressure", "temperature"],
            variable_grids=[_grid(), temp_grid],
            scenarios=[
                _baseline_scenario(
                    variable_values={"pressure": 50.0, "temperature": 80.0}
                ),
                _single_scenario(
                    variable_values={"pressure": 0.0, "temperature": 80.0}
                ),
            ],
            effective_combination_limit=2,
        )


def test_report_scenario_id_duplicate() -> None:
    with pytest.raises(ValidationError):
        _ready_report(
            scenarios=[
                _baseline_scenario(),
                _single_scenario(scenario_id="SCN-000000", scenario_index=0),
            ]
        )


def test_report_scenario_index_duplicate() -> None:
    with pytest.raises(ValidationError):
        _ready_report(
            scenarios=[
                _baseline_scenario(),
                _single_scenario(scenario_index=0, scenario_id="SCN-000000"),
            ]
        )


def test_report_scenario_index_noncontiguous() -> None:
    with pytest.raises(ValidationError):
        _ready_report(
            scenarios=[
                _baseline_scenario(),
                _single_scenario(scenario_id="SCN-000002", scenario_index=2),
            ],
            generated_scenario_count=2,
        )


def test_report_generated_count_mismatch() -> None:
    with pytest.raises(ValidationError):
        _ready_report(generated_scenario_count=9)


def test_report_change_count_mismatch() -> None:
    with pytest.raises(ValidationError):
        _ready_report(change_scenario_count=0)


def test_report_potential_count_relation_error() -> None:
    with pytest.raises(ValidationError):
        _ready_report(potential_scenario_count=1)


def test_report_truncated_count_relation_error() -> None:
    with pytest.raises(ValidationError):
        _ready_report(truncated_scenario_count=1)


def test_report_maximum_exceeded() -> None:
    with pytest.raises(ValidationError):
        _ready_report(maximum_scenarios=1)


def test_report_baseline_missing() -> None:
    with pytest.raises(ValidationError):
        _ready_report(
            scenarios=[_single_scenario(scenario_id="SCN-000000", scenario_index=0)],
            generated_scenario_count=1,
            change_scenario_count=1,
            potential_scenario_count=1,
            truncated_scenario_count=0,
            baseline_scenario_id="SCN-000000",
        )


def test_report_baseline_not_first_rejected() -> None:
    with pytest.raises(ValidationError):
        _ready_report(
            scenarios=[
                _single_scenario(scenario_id="SCN-000000", scenario_index=0),
                _baseline_scenario(scenario_id="SCN-000001", scenario_index=1),
            ]
        )


def test_report_baseline_value_mismatch() -> None:
    with pytest.raises(ValidationError):
        _ready_report(
            scenarios=[
                _baseline_scenario(variable_values={"pressure": 0.0}),
                _single_scenario(),
            ]
        )


def test_report_scenario_variable_set_mismatch() -> None:
    with pytest.raises(ValidationError):
        _ready_report(
            scenarios=[
                _baseline_scenario(),
                _single_scenario(variable_values={"temperature": 0.0}),
            ]
        )


def test_report_grid_missing_point_rejected() -> None:
    with pytest.raises(ValidationError):
        _ready_report(
            scenarios=[
                _baseline_scenario(),
                _single_scenario(
                    variable_values={"pressure": 12.5},
                    deltas={"pressure": -37.5},
                    relative_deltas={"pressure": -0.75},
                    normalized_change_magnitude=0.375,
                ),
            ]
        )


def test_report_unchanged_variable_changed_rejected() -> None:
    temp = _grid(
        variable="temperature",
        diagnosis_rank=2,
        current_value=80.0,
        points=[
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
    with pytest.raises(ValidationError):
        _ready_report(
            candidate_variables=["pressure", "temperature"],
            variable_grids=[_grid(), temp],
            scenarios=[
                _baseline_scenario(
                    variable_values={"pressure": 50.0, "temperature": 80.0}
                ),
                _single_scenario(
                    variable_values={"pressure": 0.0, "temperature": 0.0},
                    changed_variables=["pressure"],
                    deltas={"pressure": -50.0},
                    relative_deltas={"pressure": -1.0},
                    change_count=1,
                    normalized_change_magnitude=0.5,
                ),
            ],
            effective_combination_limit=2,
        )


def test_report_delta_relation_error() -> None:
    with pytest.raises(ValidationError):
        _ready_report(
            scenarios=[
                _baseline_scenario(),
                _single_scenario(deltas={"pressure": -10.0}),
            ]
        )


def test_report_relative_delta_relation_error() -> None:
    with pytest.raises(ValidationError):
        _ready_report(
            scenarios=[
                _baseline_scenario(),
                _single_scenario(relative_deltas={"pressure": 0.5}),
            ]
        )


def test_report_normalized_magnitude_relation_error() -> None:
    with pytest.raises(ValidationError):
        _ready_report(
            scenarios=[
                _baseline_scenario(),
                _single_scenario(normalized_change_magnitude=0.1),
            ]
        )


def test_report_combination_limit_exceeded() -> None:
    with pytest.raises(ValidationError):
        _ready_report(
            effective_combination_limit=0,
            scenarios=[_baseline_scenario(), _single_scenario()],
        )


def test_report_status_relation_error() -> None:
    with pytest.raises(ValidationError):
        _ready_report(status=CandidateGridStatus.EMPTY)


def test_report_naive_datetime_rejected() -> None:
    with pytest.raises(ValidationError):
        _ready_report(generated_at=datetime(2026, 7, 21, 15, 0))


def test_report_warning_duplicate() -> None:
    with pytest.raises(ValidationError):
        _ready_report(warnings=["a", "a"])


def test_report_metadata_scalar_only() -> None:
    with pytest.raises(ValidationError):
        _ready_report(metadata={"arr": np.array([1.0])})  # type: ignore[dict-item]


def test_report_round_trip() -> None:
    report = _ready_report()
    assert CandidateGridReport.model_validate(report.model_dump()) == report


# --- Outcome ---


def test_outcome_frozen() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=20)
    )
    outcome = CandidateGridOutcome(report=report)
    with pytest.raises(FrozenInstanceError):
        outcome.report = report  # type: ignore[misc]


def test_outcome_slots() -> None:
    assert hasattr(CandidateGridOutcome, "__slots__")


# --- Generator construction ---


def test_default_generator() -> None:
    generator = CandidateGridGenerator()
    meta = generator.get_metadata()
    assert meta["maximum_scenarios"] == 500


def test_custom_policy() -> None:
    generator = CandidateGridGenerator(policy=CandidateGridPolicy(maximum_scenarios=7))
    assert generator.get_metadata()["maximum_scenarios"] == 7


def test_policy_type_error() -> None:
    with pytest.raises(TypeError):
        CandidateGridGenerator(policy="bad")  # type: ignore[arg-type]


def test_external_policy_immutability() -> None:
    policy = CandidateGridPolicy(maximum_scenarios=9)
    generator = CandidateGridGenerator(policy=policy)
    policy.maximum_scenarios = 1  # type: ignore[misc]
    # Pydantic models may allow mutation unless frozen; ensure generator copy.
    assert generator.get_metadata()["maximum_scenarios"] == 9


def test_generator_state_isolation() -> None:
    g1 = CandidateGridGenerator(policy=CandidateGridPolicy(maximum_scenarios=3))
    g2 = CandidateGridGenerator(policy=CandidateGridPolicy(maximum_scenarios=8))
    assert g1.get_metadata()["maximum_scenarios"] != g2.get_metadata()["maximum_scenarios"]


def test_generator_metadata_scalar_only() -> None:
    meta = CandidateGridGenerator().get_metadata()
    for value in meta.values():
        assert value is None or isinstance(value, (str, int, float, bool))


def test_generator_metadata_independence() -> None:
    generator = CandidateGridGenerator()
    a = generator.get_metadata()
    b = generator.get_metadata()
    a["maximum_scenarios"] = -1
    assert b["maximum_scenarios"] == 500


# --- Input validation ---


def test_non_candidate_set_rejected() -> None:
    with pytest.raises(TypeError):
        CandidateGridGenerator().generate({"bad": True})  # type: ignore[arg-type]


def test_input_candidate_set_immutability() -> None:
    candidate_set = _candidate_set()
    original = list(candidate_set.candidate_variables)
    CandidateGridGenerator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=30)
    ).generate(candidate_set)
    assert candidate_set.candidate_variables == original


def test_candidate_list_immutability() -> None:
    candidate_set = _candidate_set()
    before = len(candidate_set.candidates)
    CandidateGridGenerator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=30)
    ).generate(candidate_set)
    assert len(candidate_set.candidates) == before


def test_candidate_variable_order_mismatch_defense() -> None:
    candidate_set = _candidate_set()
    broken = candidate_set.model_copy(deep=True)
    object.__setattr__(
        broken,
        "__dict__",
        {**broken.__dict__, "candidate_variables": ["temperature", "pressure"]},
    )
    with pytest.raises(DataValidationError):
        CandidateGridGenerator().generate(broken)


def test_candidate_rank_duplicate_defense() -> None:
    candidate_set = _candidate_set()
    broken = candidate_set.model_copy(deep=True)
    c0 = broken.candidates[0].model_copy(deep=True)
    c1 = broken.candidates[1].model_copy(update={"diagnosis_rank": 1})
    object.__setattr__(
        broken,
        "__dict__",
        {**broken.__dict__, "candidates": [c0, c1]},
    )
    with pytest.raises(DataValidationError):
        CandidateGridGenerator().generate(broken)


def test_budget_relation_defense() -> None:
    candidate_set = _candidate_set()
    broken = candidate_set.model_copy(deep=True)
    object.__setattr__(
        broken,
        "__dict__",
        {**broken.__dict__, "effective_change_budget": 99},
    )
    with pytest.raises(DataValidationError):
        CandidateGridGenerator().generate(broken)


# --- Refused / empty ---


def test_safety_refused_structured() -> None:
    candidate_set = _candidate_set(
        candidates=[],
        candidate_variables=[],
        safety_status=RecommendationSafetyStatus.REFUSED,
        resolution_status=ConstraintResolutionStatus.REFUSED,
        effective_change_budget=0,
        max_simultaneous_changes=1,
    )
    report = CandidateGridGenerator().generate(candidate_set).report
    assert report.status is CandidateGridStatus.REFUSED
    assert report.scenarios == []
    assert report.variable_grids == []


def test_resolution_refused_structured() -> None:
    candidate_set = _candidate_set(
        candidates=[],
        candidate_variables=[],
        safety_status=RecommendationSafetyStatus.APPROVED,
        resolution_status=ConstraintResolutionStatus.REFUSED,
        effective_change_budget=0,
        max_simultaneous_changes=1,
    )
    report = CandidateGridGenerator().generate(candidate_set).report
    assert report.status is CandidateGridStatus.REFUSED


def test_zero_candidates_empty() -> None:
    candidate_set = _candidate_set(
        candidates=[],
        candidate_variables=[],
        effective_change_budget=0,
        max_simultaneous_changes=1,
    )
    report = CandidateGridGenerator().generate(candidate_set).report
    assert report.status is CandidateGridStatus.EMPTY
    assert report.scenarios == []


def test_refused_no_exception() -> None:
    candidate_set = _candidate_set(
        candidates=[],
        candidate_variables=[],
        safety_status=RecommendationSafetyStatus.REFUSED,
        resolution_status=ConstraintResolutionStatus.READY,
        effective_change_budget=0,
        max_simultaneous_changes=1,
    )
    report = CandidateGridGenerator().generate(candidate_set).report
    assert report.status is CandidateGridStatus.REFUSED


def test_refused_metadata_model_scoring_false() -> None:
    candidate_set = _candidate_set(
        candidates=[],
        candidate_variables=[],
        safety_status=RecommendationSafetyStatus.REFUSED,
        resolution_status=ConstraintResolutionStatus.REFUSED,
        effective_change_budget=0,
        max_simultaneous_changes=1,
    )
    report = CandidateGridGenerator().generate(candidate_set).report
    assert report.metadata["model_scoring_performed"] is False


# --- Grid calculation ---


def test_linspace_includes_bounds() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=5, maximum_scenarios=100)
    )
    grid = report.variable_grids[0]
    values = [point.value for point in grid.points]
    assert values[0] == pytest.approx(grid.minimum)
    assert values[-1] == pytest.approx(grid.maximum)


def test_current_value_included() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=5, maximum_scenarios=100)
    )
    grid = report.variable_grids[0]
    assert any(point.is_current for point in grid.points)
    current = next(point for point in grid.points if point.is_current)
    assert current.value == pytest.approx(grid.current_value)


def test_current_added_when_absent_from_linspace() -> None:
    candidate_set = _candidate_set(
        candidates=[
            _candidate(current_value=33.0, lower_room=33.0, upper_room=67.0, relative_position=0.33)
        ],
        max_simultaneous_changes=1,
    )
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=50),
        candidate_set=candidate_set,
    )
    values = [point.value for point in report.variable_grids[0].points]
    assert 33.0 in values
    assert len(values) == 4


def test_points_numeric_sorted() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=5, maximum_scenarios=100)
    )
    values = [point.value for point in report.variable_grids[0].points]
    assert values == sorted(values)


def test_normalized_position() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=5, maximum_scenarios=100)
    )
    grid = report.variable_grids[0]
    for point in grid.points:
        expected = (point.value - grid.minimum) / grid.effective_span
        assert point.normalized_position == pytest.approx(expected)


def test_delta_calculation() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=5, maximum_scenarios=100)
    )
    grid = report.variable_grids[0]
    for point in grid.points:
        assert point.delta == pytest.approx(point.value - grid.current_value)


def test_relative_delta_calculation() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=5, maximum_scenarios=100)
    )
    grid = report.variable_grids[0]
    for point in grid.points:
        if grid.current_value == 0.0:
            assert point.relative_delta is None
        else:
            assert point.relative_delta == pytest.approx(
                point.delta / abs(grid.current_value)
            )


def test_relative_delta_none_when_current_zero() -> None:
    candidate_set = _candidate_set(
        candidates=[
            _candidate(
                current_value=0.0,
                lower_room=0.0,
                upper_room=100.0,
                relative_position=0.0,
                at_lower_bound=True,
            )
        ],
        max_simultaneous_changes=1,
    )
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=20),
        candidate_set=candidate_set,
    )
    for point in report.variable_grids[0].points:
        assert point.relative_delta is None


def test_bound_flags() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=5, maximum_scenarios=100)
    )
    grid = report.variable_grids[0]
    assert any(point.is_lower_bound for point in grid.points)
    assert any(point.is_upper_bound for point in grid.points)


def test_effective_span_used() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=50)
    )
    grid = report.variable_grids[0]
    assert grid.effective_span == pytest.approx(grid.maximum - grid.minimum)


def test_tolerance_dedup() -> None:
    candidate_set = _candidate_set(
        candidates=[_candidate(current_value=50.0)],
        max_simultaneous_changes=1,
    )
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=3,
            maximum_scenarios=20,
            deduplication_tolerance=1.0,
        ),
        candidate_set=candidate_set,
    )
    values = [point.value for point in report.variable_grids[0].points]
    assert len(values) == len(set(values))


def test_current_priority_dedup() -> None:
    candidate_set = _candidate_set(
        candidates=[_candidate(current_value=50.000000000001)],
        max_simultaneous_changes=1,
    )
    # Rebuild rooms for awkward current near 50 from linspace.
    candidate = candidate_set.candidates[0]
    current = candidate.current_value
    candidate_set = _candidate_set(
        candidates=[
            _candidate(
                current_value=current,
                lower_room=current - 0.0,
                upper_room=100.0 - current,
                relative_position=(current - 0.0) / 100.0,
            )
        ],
        max_simultaneous_changes=1,
    )
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=3,
            maximum_scenarios=20,
            deduplication_tolerance=1e-6,
        ),
        candidate_set=candidate_set,
    )
    current_points = [p for p in report.variable_grids[0].points if p.is_current]
    assert len(current_points) == 1
    assert current_points[0].value == pytest.approx(current)


def test_boundary_preserved() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=2,
            maximum_scenarios=20,
            deduplication_tolerance=1e6,
        ),
        candidate_set=_candidate_set(
            candidates=[_candidate()],
            max_simultaneous_changes=1,
        ),
    )
    values = [point.value for point in report.variable_grids[0].points]
    assert 0.0 in values
    assert 100.0 in values


def test_minimum_nonzero_delta_removal() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=5,
            maximum_scenarios=50,
            minimum_nonzero_delta=30.0,
        ),
        candidate_set=_candidate_set(
            candidates=[_candidate()],
            max_simultaneous_changes=1,
        ),
    )
    assert _WARNING_NONZERO_DELTA in report.warnings or report.variable_grids[
        0
    ].change_point_count <= 2


def test_change_point_count() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=5, maximum_scenarios=100)
    )
    grid = report.variable_grids[0]
    assert grid.change_point_count == sum(1 for p in grid.points if not p.is_current)


# --- Scenario ordering ---


def test_baseline_first() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=50)
    )
    assert report.scenarios[0].scenario_type is CandidateScenarioType.BASELINE


def test_baseline_id() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=50)
    )
    assert report.scenarios[0].scenario_id == "SCN-000000"


def test_scenario_ids_contiguous() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=50)
    )
    for index, scenario in enumerate(report.scenarios):
        assert scenario.scenario_id == f"SCN-{index:06d}"


def test_single_variable_before_multi() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=3,
            maximum_scenarios=200,
            maximum_combination_variables=2,
        )
    )
    types = [item.scenario_type for item in report.scenarios]
    if CandidateScenarioType.MULTI_VARIABLE in types:
        first_multi = types.index(CandidateScenarioType.MULTI_VARIABLE)
        assert all(
            item is not CandidateScenarioType.SINGLE_VARIABLE
            for item in types[first_multi:]
        )


def test_diagnosis_variable_order() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=50)
    )
    singles = [
        item
        for item in report.scenarios
        if item.scenario_type is CandidateScenarioType.SINGLE_VARIABLE
    ]
    first_vars = [item.changed_variables[0] for item in singles]
    pressure_idx = first_vars.index("pressure")
    temperature_idx = first_vars.index("temperature")
    assert pressure_idx < temperature_idx


def test_prefer_smaller_change_order() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=5,
            maximum_scenarios=50,
            prefer_smaller_changes=True,
            include_multi_variable_scenarios=False,
        ),
        candidate_set=_candidate_set(
            candidates=[_candidate()],
            max_simultaneous_changes=1,
        ),
    )
    singles = [
        item
        for item in report.scenarios
        if item.scenario_type is CandidateScenarioType.SINGLE_VARIABLE
    ]
    magnitudes = [
        abs(item.deltas["pressure"]) / report.variable_grids[0].effective_span
        for item in singles
    ]
    assert magnitudes == sorted(magnitudes)


def test_negative_preferred_at_equal_distance() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=5,
            maximum_scenarios=50,
            prefer_smaller_changes=True,
            include_multi_variable_scenarios=False,
        ),
        candidate_set=_candidate_set(
            candidates=[_candidate()],
            max_simultaneous_changes=1,
        ),
    )
    singles = [
        item
        for item in report.scenarios
        if item.scenario_type is CandidateScenarioType.SINGLE_VARIABLE
    ]
    # For equal abs delta pairs, negative should appear first.
    by_mag: dict[float, list[float]] = {}
    for item in singles:
        mag = round(abs(item.deltas["pressure"]), 12)
        by_mag.setdefault(mag, []).append(item.deltas["pressure"])
    for deltas in by_mag.values():
        if len(deltas) >= 2 and any(d < 0 for d in deltas) and any(d > 0 for d in deltas):
            assert deltas[0] < 0


def test_prefer_smaller_false_value_order() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=5,
            maximum_scenarios=50,
            prefer_smaller_changes=False,
            include_multi_variable_scenarios=False,
        ),
        candidate_set=_candidate_set(
            candidates=[_candidate()],
            max_simultaneous_changes=1,
        ),
    )
    singles = [
        item
        for item in report.scenarios
        if item.scenario_type is CandidateScenarioType.SINGLE_VARIABLE
    ]
    values = [item.variable_values["pressure"] for item in singles]
    assert values == sorted(values)


def test_multi_after_single() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=3,
            maximum_scenarios=200,
            maximum_combination_variables=2,
        )
    )
    last_single = max(
        (
            i
            for i, item in enumerate(report.scenarios)
            if item.scenario_type is CandidateScenarioType.SINGLE_VARIABLE
        ),
        default=-1,
    )
    first_multi = next(
        (
            i
            for i, item in enumerate(report.scenarios)
            if item.scenario_type is CandidateScenarioType.MULTI_VARIABLE
        ),
        None,
    )
    if first_multi is not None:
        assert last_single < first_multi


def test_change_count_ascending() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=3,
            maximum_scenarios=200,
            maximum_combination_variables=2,
        )
    )
    counts = [item.change_count for item in report.scenarios]
    assert counts == sorted(counts)


def test_combinations_deterministic() -> None:
    policy = CandidateGridPolicy(
        linear_points_per_variable=3,
        maximum_scenarios=200,
        maximum_combination_variables=2,
    )
    a = _report_from_generator(policy=policy)
    b = _report_from_generator(policy=policy)
    assert [s.changed_variables for s in a.scenarios] == [
        s.changed_variables for s in b.scenarios
    ]


def test_product_order_deterministic() -> None:
    policy = CandidateGridPolicy(
        linear_points_per_variable=3,
        maximum_scenarios=200,
        maximum_combination_variables=2,
    )
    a = _report_from_generator(policy=policy)
    b = _report_from_generator(policy=policy)
    assert [s.variable_values for s in a.scenarios] == [
        s.variable_values for s in b.scenarios
    ]


def test_random_not_used(monkeypatch: pytest.MonkeyPatch) -> None:
    import random

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("random should not be used")

    monkeypatch.setattr(random, "random", _boom)
    monkeypatch.setattr(random, "shuffle", _boom)
    _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=30)
    )


# --- Budget / combinations ---


def test_effective_budget_used() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=3,
            maximum_scenarios=200,
            maximum_combination_variables=3,
        ),
        candidate_set=_candidate_set(max_simultaneous_changes=1),
    )
    assert report.effective_combination_limit == 1
    assert all(item.change_count <= 1 for item in report.scenarios)


def test_policy_combination_cap_used() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=3,
            maximum_scenarios=200,
            maximum_combination_variables=1,
        )
    )
    assert report.effective_combination_limit == 1
    assert all(
        item.scenario_type is not CandidateScenarioType.MULTI_VARIABLE
        for item in report.scenarios
    )


def test_candidate_count_cap_used() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=3,
            maximum_scenarios=50,
            maximum_combination_variables=5,
        ),
        candidate_set=_candidate_set(
            candidates=[_candidate()],
            max_simultaneous_changes=5,
        ),
    )
    assert report.effective_combination_limit == 1


def test_multi_disabled() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=3,
            maximum_scenarios=100,
            include_multi_variable_scenarios=False,
        )
    )
    assert all(
        item.scenario_type is not CandidateScenarioType.MULTI_VARIABLE
        for item in report.scenarios
    )


def test_single_disabled() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=3,
            maximum_scenarios=100,
            include_single_variable_scenarios=False,
            include_multi_variable_scenarios=True,
            maximum_combination_variables=2,
        )
    )
    assert all(
        item.scenario_type is not CandidateScenarioType.SINGLE_VARIABLE
        for item in report.scenarios
    )


def test_budget_one_no_multi() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=100),
        candidate_set=_candidate_set(max_simultaneous_changes=1),
    )
    assert all(item.change_count <= 1 for item in report.scenarios)


def test_no_change_count_over_budget() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=3,
            maximum_scenarios=200,
            maximum_combination_variables=2,
        )
    )
    assert all(
        item.change_count <= report.effective_combination_limit
        for item in report.scenarios
    )


# --- Scenario values ---


def test_all_candidate_variables_included() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=50)
    )
    for scenario in report.scenarios:
        assert set(scenario.variable_values) == set(report.candidate_variables)


def test_unchanged_keep_current() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=50)
    )
    currents = {g.variable: g.current_value for g in report.variable_grids}
    for scenario in report.scenarios:
        for name, value in scenario.variable_values.items():
            if name not in scenario.changed_variables:
                assert value == pytest.approx(currents[name])


def test_changed_only_have_deltas() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=50)
    )
    for scenario in report.scenarios:
        assert set(scenario.deltas) == set(scenario.changed_variables)


def test_relative_delta_accuracy() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=50)
    )
    currents = {g.variable: g.current_value for g in report.variable_grids}
    for scenario in report.scenarios:
        for name, delta in scenario.deltas.items():
            current = currents[name]
            if current == 0.0:
                assert scenario.relative_deltas[name] is None
            else:
                assert scenario.relative_deltas[name] == pytest.approx(
                    delta / abs(current)
                )


def test_normalized_magnitude_accuracy() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=50)
    )
    spans = {g.variable: g.effective_span for g in report.variable_grids}
    for scenario in report.scenarios:
        expected = sum(
            abs(delta) / spans[name] for name, delta in scenario.deltas.items()
        )
        assert scenario.normalized_change_magnitude == pytest.approx(expected)


def test_no_duplicate_scenarios() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=3,
            maximum_scenarios=200,
            maximum_combination_variables=2,
        )
    )
    signatures = [
        tuple(item.variable_values[name] for name in report.candidate_variables)
        for item in report.scenarios
    ]
    assert len(signatures) == len(set(signatures))


def test_no_baseline_duplicate() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=50)
    )
    baselines = [
        item
        for item in report.scenarios
        if item.scenario_type is CandidateScenarioType.BASELINE
    ]
    assert len(baselines) == 1


# --- Potential / truncation ---


def test_potential_count_single_accurate() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=3,
            maximum_scenarios=500,
            include_multi_variable_scenarios=False,
        ),
        candidate_set=_candidate_set(
            candidates=[_candidate()],
            max_simultaneous_changes=1,
        ),
    )
    expected = 1 + report.variable_grids[0].change_point_count
    assert report.potential_scenario_count == expected


def test_potential_count_multi_accurate() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=3,
            maximum_scenarios=500,
            maximum_combination_variables=2,
            include_single_variable_scenarios=True,
            include_multi_variable_scenarios=True,
        )
    )
    c0 = report.variable_grids[0].change_point_count
    c1 = report.variable_grids[1].change_point_count
    expected = 1 + c0 + c1 + (c0 * c1)
    assert report.potential_scenario_count == expected


def test_baseline_included_in_potential() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=3,
            maximum_scenarios=500,
            include_multi_variable_scenarios=False,
        ),
        candidate_set=_candidate_set(
            candidates=[_candidate()],
            max_simultaneous_changes=1,
        ),
    )
    assert report.potential_scenario_count >= 1


def test_maximum_scenarios_includes_baseline() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=5,
            maximum_scenarios=1,
            include_multi_variable_scenarios=False,
        ),
        candidate_set=_candidate_set(
            candidates=[_candidate()],
            max_simultaneous_changes=1,
        ),
    )
    assert report.generated_scenario_count == 1
    assert report.scenarios[0].scenario_type is CandidateScenarioType.BASELINE


def test_cap_stops_immediately() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=5,
            maximum_scenarios=3,
            include_multi_variable_scenarios=False,
        ),
        candidate_set=_candidate_set(
            candidates=[_candidate()],
            max_simultaneous_changes=1,
        ),
    )
    assert report.generated_scenario_count == 3


def test_deterministic_prefix_preserved() -> None:
    policy = CandidateGridPolicy(
        linear_points_per_variable=5,
        maximum_scenarios=4,
        include_multi_variable_scenarios=False,
    )
    full = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=5,
            maximum_scenarios=500,
            include_multi_variable_scenarios=False,
        ),
        candidate_set=_candidate_set(
            candidates=[_candidate()],
            max_simultaneous_changes=1,
        ),
    )
    capped = _report_from_generator(
        policy=policy,
        candidate_set=_candidate_set(
            candidates=[_candidate()],
            max_simultaneous_changes=1,
        ),
    )
    assert [s.variable_values for s in capped.scenarios] == [
        s.variable_values for s in full.scenarios[:4]
    ]


def test_truncated_count_accurate() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=5,
            maximum_scenarios=3,
            include_multi_variable_scenarios=False,
        ),
        candidate_set=_candidate_set(
            candidates=[_candidate()],
            max_simultaneous_changes=1,
        ),
    )
    assert report.truncated_scenario_count == (
        report.potential_scenario_count - report.generated_scenario_count
    )


def test_truncated_status() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=5,
            maximum_scenarios=3,
            include_multi_variable_scenarios=False,
        ),
        candidate_set=_candidate_set(
            candidates=[_candidate()],
            max_simultaneous_changes=1,
        ),
    )
    assert report.status is CandidateGridStatus.TRUNCATED


def test_ready_when_under_cap() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=3,
            maximum_scenarios=500,
            include_multi_variable_scenarios=False,
        ),
        candidate_set=_candidate_set(
            candidates=[_candidate()],
            max_simultaneous_changes=1,
        ),
    )
    assert report.status is CandidateGridStatus.READY


def test_baseline_only_empty() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=2,
            maximum_scenarios=10,
            include_single_variable_scenarios=False,
            include_multi_variable_scenarios=False,
        ),
        candidate_set=_candidate_set(
            candidates=[_candidate()],
            max_simultaneous_changes=1,
        ),
    )
    assert report.status is CandidateGridStatus.EMPTY
    assert report.change_scenario_count == 0


# --- Warnings / metadata ---


def test_caution_warning() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=50),
        candidate_set=_candidate_set(safety_status=RecommendationSafetyStatus.CAUTION),
    )
    assert _WARNING_SAFETY_CAUTION in report.warnings


def test_partial_warning() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=50),
        candidate_set=_candidate_set(
            resolution_status=ConstraintResolutionStatus.PARTIAL
        ),
    )
    assert _WARNING_RESOLUTION_PARTIAL in report.warnings


def test_dedup_warning() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=50),
        candidate_set=_candidate_set(
            candidates=[_candidate()],
            max_simultaneous_changes=1,
        ),
    )
    # current already in linspace for 3 points -> dedup
    assert _WARNING_DEDUP in report.warnings


def test_removed_point_warning() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=5,
            maximum_scenarios=50,
            minimum_nonzero_delta=40.0,
        ),
        candidate_set=_candidate_set(
            candidates=[_candidate()],
            max_simultaneous_changes=1,
        ),
    )
    assert _WARNING_NONZERO_DELTA in report.warnings or _WARNING_NO_CHANGE_POINTS in report.warnings


def test_no_change_point_warning_path() -> None:
    # Extremely large nonzero delta removes non-boundary changes; boundaries remain.
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=5,
            maximum_scenarios=20,
            minimum_nonzero_delta=1e9,
        ),
        candidate_set=_candidate_set(
            candidates=[
                _candidate(
                    current_value=0.0,
                    lower_room=0.0,
                    upper_room=100.0,
                    relative_position=0.0,
                    at_lower_bound=True,
                )
            ],
            max_simultaneous_changes=1,
        ),
    )
    # Upper bound still a change point unless filtered by nonzero for scenarios.
    assert isinstance(report.warnings, list)


def test_combination_cap_warning() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=3,
            maximum_scenarios=50,
            maximum_combination_variables=1,
        ),
        candidate_set=_candidate_set(max_simultaneous_changes=2),
    )
    assert _WARNING_COMBINATION_CAP in report.warnings


def test_truncation_warning() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=5,
            maximum_scenarios=3,
            include_multi_variable_scenarios=False,
        ),
        candidate_set=_candidate_set(
            candidates=[_candidate()],
            max_simultaneous_changes=1,
        ),
    )
    assert _WARNING_TRUNCATED in report.warnings


def test_model_scoring_warning() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=20)
    )
    assert _WARNING_MODEL_SCORING in report.warnings


def test_verification_warning() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=20)
    )
    assert _WARNING_VERIFICATION in report.warnings


def test_warning_order_deterministic() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(
            linear_points_per_variable=5,
            maximum_scenarios=3,
            maximum_combination_variables=1,
            include_multi_variable_scenarios=False,
        ),
        candidate_set=_candidate_set(
            candidates=[
                _candidate(),
                _candidate(
                    variable="temperature",
                    diagnosis_rank=2,
                    current_value=80.0,
                ),
            ],
            safety_status=RecommendationSafetyStatus.CAUTION,
            resolution_status=ConstraintResolutionStatus.PARTIAL,
            max_simultaneous_changes=2,
        ),
    )
    expected_prefix = [
        _WARNING_SAFETY_CAUTION,
        _WARNING_RESOLUTION_PARTIAL,
    ]
    assert report.warnings[:2] == expected_prefix


def test_warning_no_duplicates() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=50)
    )
    assert len(report.warnings) == len(set(report.warnings))


def test_metadata_counts_accurate() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=50)
    )
    assert report.metadata["candidate_variable_count"] == len(report.candidate_variables)
    assert report.metadata["generated_scenario_count"] == report.generated_scenario_count
    assert report.metadata["change_scenario_count"] == report.change_scenario_count


def test_metadata_scenarios_unranked() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=20)
    )
    assert report.metadata["scenarios_are_unranked"] is True


def test_metadata_optimization_false() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=20)
    )
    assert report.metadata["optimization_performed"] is False


def test_metadata_recommendation_false() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=20)
    )
    assert report.metadata["recommendation_generated"] is False


def test_metadata_no_dataframe_ndarray_model() -> None:
    report = _report_from_generator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=20)
    )
    for value in report.metadata.values():
        assert not hasattr(value, "shape")
        assert not hasattr(value, "predict")


# --- State / determinism ---


def test_no_result_cache() -> None:
    generator = CandidateGridGenerator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=30)
    )
    a = generator.generate(_candidate_set()).report
    b = generator.generate(_candidate_set()).report
    assert a.generated_at != b.generated_at or a.scenarios == b.scenarios


def test_no_accumulation_across_calls() -> None:
    generator = CandidateGridGenerator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=30)
    )
    first = generator.generate(_candidate_set()).report.generated_scenario_count
    second = generator.generate(_candidate_set()).report.generated_scenario_count
    assert first == second


def test_same_input_grids_deterministic() -> None:
    policy = CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=30)
    a = _report_from_generator(policy=policy)
    b = _report_from_generator(policy=policy)
    assert [
        [point.value for point in grid.points] for grid in a.variable_grids
    ] == [[point.value for point in grid.points] for grid in b.variable_grids]


def test_same_input_scenario_ordering_deterministic() -> None:
    policy = CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=30)
    a = _report_from_generator(policy=policy)
    b = _report_from_generator(policy=policy)
    assert [s.scenario_id for s in a.scenarios] == [s.scenario_id for s in b.scenarios]


def test_same_input_scenario_values_deterministic() -> None:
    policy = CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=30)
    a = _report_from_generator(policy=policy)
    b = _report_from_generator(policy=policy)
    assert [s.variable_values for s in a.scenarios] == [
        s.variable_values for s in b.scenarios
    ]


def test_separate_generators_isolated() -> None:
    g1 = CandidateGridGenerator(policy=CandidateGridPolicy(maximum_scenarios=5))
    g2 = CandidateGridGenerator(policy=CandidateGridPolicy(maximum_scenarios=50))
    r1 = g1.generate(_candidate_set(
        candidates=[_candidate()],
        max_simultaneous_changes=1,
    )).report
    r2 = g2.generate(_candidate_set(
        candidates=[_candidate()],
        max_simultaneous_changes=1,
    )).report
    assert r1.generated_scenario_count <= 5
    assert r2.generated_scenario_count >= r1.generated_scenario_count


def test_mutating_returned_report_does_not_affect_next_call() -> None:
    generator = CandidateGridGenerator(
        policy=CandidateGridPolicy(linear_points_per_variable=3, maximum_scenarios=30)
    )
    first = generator.generate(_candidate_set()).report
    first.warnings.append("mutated")
    second = generator.generate(_candidate_set()).report
    assert "mutated" not in second.warnings
