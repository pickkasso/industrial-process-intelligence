"""Unit tests for recommendation what-if verification (Step 11B.13)."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from typing import Any

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from process_intelligence.core.protocols import BaseAnalysisModel, BaseAnomalyModel
from process_intelligence.models import (
    create_isolation_forest_anomaly_model,
    create_linear_regression,
)
from process_intelligence.models.anomaly import IsolationForestConfig
from process_intelligence.recommendation import (
    CandidateGridReport,
    CandidateGridStatus,
    CandidateScenario,
    CandidateScenarioScorer,
    CandidateScenarioType,
    CandidateValuePoint,
    ConstraintResolutionStatus,
    QualityOptimizationDirection,
    RecommendationChange,
    RecommendationObjective,
    RecommendationReasonCode,
    RecommendationResult,
    RecommendationSafetyDecision,
    RecommendationSafetyStatus,
    RecommendationStatus,
    RecommendationWhatIfVerificationOutcome,
    RecommendationWhatIfVerificationResult,
    RecommendationWhatIfVerifier,
    VariableCandidateGrid,
    VariableEligibilityAssessment,
    WhatIfPerturbationDirection,
    WhatIfStabilityClassification,
    WhatIfVerificationScenario,
    WhatIfVerificationScenarioType,
    WhatIfVerificationStatus,
    classify_what_if_stability,
    recommendation_warnings_for_stability,
)
from process_intelligence.recommendation.schemas import DEFAULT_RECOMMENDATION_DISCLAIMER

# ---------------------------------------------------------------------------
# Builders: grids / grid reports
# ---------------------------------------------------------------------------


def _grid_point(
    value: float,
    *,
    current: float,
    minimum: float,
    maximum: float,
) -> CandidateValuePoint:
    is_current = math.isclose(value, current, rel_tol=0.0, abs_tol=1e-12)
    delta = 0.0 if is_current else value - current
    relative_delta = None if current == 0.0 else delta / abs(current)
    return CandidateValuePoint(
        value=value,
        delta=delta,
        relative_delta=relative_delta,
        normalized_position=(value - minimum) / (maximum - minimum),
        is_current=is_current,
        is_lower_bound=math.isclose(value, minimum, rel_tol=0.0, abs_tol=1e-12),
        is_upper_bound=math.isclose(value, maximum, rel_tol=0.0, abs_tol=1e-12),
    )


def _variable_grid(
    variable: str,
    *,
    diagnosis_rank: int = 1,
    current: float = 50.0,
    minimum: float = 0.0,
    maximum: float = 100.0,
    extra_values: tuple[float, ...] = (55.0, 65.0, 75.0),
) -> VariableCandidateGrid:
    values = sorted({minimum, current, maximum, *extra_values})
    points = [
        _grid_point(value, current=current, minimum=minimum, maximum=maximum)
        for value in values
    ]
    return VariableCandidateGrid(
        variable=variable,
        diagnosis_rank=diagnosis_rank,
        current_value=current,
        minimum=minimum,
        maximum=maximum,
        effective_span=maximum - minimum,
        points=points,
        point_count=len(points),
        change_point_count=sum(1 for point in points if not point.is_current),
        warnings=[],
    )


def _baseline_candidate_scenario(
    candidate_variables: list[str],
    grids: dict[str, VariableCandidateGrid],
) -> CandidateScenario:
    values = {name: grids[name].current_value for name in candidate_variables}
    return CandidateScenario(
        scenario_id="SCN-000000",
        scenario_index=0,
        scenario_type=CandidateScenarioType.BASELINE,
        variable_values=values,
        changed_variables=[],
        deltas={},
        relative_deltas={},
        change_count=0,
        normalized_change_magnitude=0.0,
    )


def _grid_report(
    *,
    candidate_variables: list[str],
    grids: dict[str, VariableCandidateGrid],
    objective: RecommendationObjective,
    generated_at: datetime | None = None,
) -> CandidateGridReport:
    baseline = _baseline_candidate_scenario(candidate_variables, grids)
    return CandidateGridReport(
        status=CandidateGridStatus.EMPTY,
        objective=objective,
        safety_status=RecommendationSafetyStatus.APPROVED,
        resolution_status=ConstraintResolutionStatus.READY,
        candidate_variables=list(candidate_variables),
        variable_grids=[grids[name] for name in candidate_variables],
        scenarios=[baseline],
        baseline_scenario_id="SCN-000000",
        potential_scenario_count=1,
        generated_scenario_count=1,
        change_scenario_count=0,
        truncated_scenario_count=0,
        maximum_scenarios=500,
        effective_combination_limit=max(1, len(candidate_variables)),
        generated_at=generated_at or datetime(2026, 7, 21, 15, 0, tzinfo=UTC),
        warnings=[],
        metadata={},
    )


# ---------------------------------------------------------------------------
# Builders: recommendation
# ---------------------------------------------------------------------------


def _eligibility_assessment(
    variable: str,
    *,
    current_value: float,
    factor_rank: int = 1,
) -> VariableEligibilityAssessment:
    return VariableEligibilityAssessment(
        variable=variable,
        factor_rank=factor_rank,
        factor_confidence=0.7,
        factor_role="CONTROLLABLE_PROCESS",
        factor_controllable=True,
        factor_needs_verification=False,
        current_value=current_value,
        constraint_present=True,
        user_confirmed_controllable=True,
        user_verified=False,
        eligible=True,
        reason_codes=[],
        warnings=[],
    )


def _safety_decision(
    *,
    eligible_variables: list[str],
    current_values: dict[str, float],
) -> RecommendationSafetyDecision:
    return RecommendationSafetyDecision(
        status=RecommendationSafetyStatus.APPROVED,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        eligible_variables=list(eligible_variables),
        blocked_variables=[],
        variable_assessments=[
            _eligibility_assessment(
                name,
                current_value=current_values[name],
                factor_rank=index + 1,
            )
            for index, name in enumerate(eligible_variables)
        ],
        global_reason_codes=[],
        messages=["Safety checks passed."],
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        evaluated_at=datetime(2026, 7, 21, 15, 0, tzinfo=UTC),
        metadata={},
    )


def _change(
    variable: str,
    *,
    current_value: float,
    proposed_value: float,
    confidence: float = 0.6,
) -> RecommendationChange:
    delta = proposed_value - current_value
    relative_delta = None if current_value == 0.0 else delta / abs(current_value)
    return RecommendationChange(
        variable=variable,
        current_value=current_value,
        proposed_value=proposed_value,
        delta=delta,
        relative_delta=relative_delta,
        rationale="Candidate change within observed support; association only.",
        confidence=confidence,
        requires_verification=True,
    )


def _generated_result(
    *,
    objective: RecommendationObjective,
    changes: list[RecommendationChange],
    eligible_variables: list[str],
    current_values: dict[str, float],
) -> RecommendationResult:
    return RecommendationResult(
        status=RecommendationStatus.GENERATED,
        objective=objective,
        safety_decision=_safety_decision(
            eligible_variables=eligible_variables,
            current_values=current_values,
        ),
        changes=changes,
        confidence=0.5,
        extrapolation_flag=False,
        uncertainty_available=False,
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        generated_at=datetime(2026, 7, 21, 15, 0, tzinfo=UTC),
        warnings=[],
    )


def _ready_result(
    *,
    eligible_variables: list[str],
    current_values: dict[str, float],
) -> RecommendationResult:
    return RecommendationResult(
        status=RecommendationStatus.READY_FOR_OPTIMIZATION,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        safety_decision=_safety_decision(
            eligible_variables=eligible_variables,
            current_values=current_values,
        ),
        changes=[],
        confidence=0.5,
        extrapolation_flag=False,
        uncertainty_available=False,
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        generated_at=datetime(2026, 7, 21, 15, 0, tzinfo=UTC),
    )


def _refused_result() -> RecommendationResult:
    decision = RecommendationSafetyDecision(
        status=RecommendationSafetyStatus.REFUSED,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        eligible_variables=[],
        blocked_variables=["temperature"],
        variable_assessments=[
            VariableEligibilityAssessment(
                variable="temperature",
                factor_rank=1,
                factor_confidence=0.7,
                factor_role="CONTROLLABLE_PROCESS",
                factor_controllable=True,
                factor_needs_verification=False,
                current_value=None,
                constraint_present=False,
                user_confirmed_controllable=True,
                user_verified=False,
                eligible=False,
                reason_codes=[RecommendationReasonCode.NO_ELIGIBLE_VARIABLES],
                warnings=[],
            )
        ],
        global_reason_codes=[RecommendationReasonCode.NO_ELIGIBLE_VARIABLES],
        messages=["Recommendation refused by safety gate."],
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        evaluated_at=datetime(2026, 7, 21, 15, 0, tzinfo=UTC),
        metadata={},
    )
    return RecommendationResult(
        status=RecommendationStatus.REFUSED,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        safety_decision=decision,
        changes=[],
        confidence=0.0,
        extrapolation_flag=False,
        uncertainty_available=False,
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        generated_at=datetime(2026, 7, 21, 15, 0, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# Builders: WhatIfVerificationScenario / RecommendationWhatIfVerificationResult
# ---------------------------------------------------------------------------


def _baseline_view(**overrides: Any) -> WhatIfVerificationScenario:
    payload: dict[str, Any] = {
        "scenario_id": "WIF-000000",
        "scenario_type": WhatIfVerificationScenarioType.BASELINE,
        "perturbed_variable": None,
        "perturbation_direction": None,
        "variable_values": {"temperature": 50.0},
        "predicted_quality": None,
        "anomaly_score": 0.0,
        "objective_value": 0.0,
        "improves_over_baseline": False,
        "improves_or_matches_proposed": False,
        "extrapolated": None,
        "warnings": [],
    }
    payload.update(overrides)
    return WhatIfVerificationScenario(**payload)


def _proposed_view(**overrides: Any) -> WhatIfVerificationScenario:
    payload: dict[str, Any] = {
        "scenario_id": "WIF-000001",
        "scenario_type": WhatIfVerificationScenarioType.PROPOSED_CENTER,
        "perturbed_variable": None,
        "perturbation_direction": None,
        "variable_values": {"temperature": 65.0},
        "predicted_quality": None,
        "anomaly_score": -0.5,
        "objective_value": -0.5,
        "improves_over_baseline": True,
        "improves_or_matches_proposed": True,
        "extrapolated": None,
        "warnings": [],
    }
    payload.update(overrides)
    return WhatIfVerificationScenario(**payload)


def _lower_neighbor_view(**overrides: Any) -> WhatIfVerificationScenario:
    payload: dict[str, Any] = {
        "scenario_id": "WIF-000002",
        "scenario_type": WhatIfVerificationScenarioType.LOWER_NEIGHBOR,
        "perturbed_variable": "temperature",
        "perturbation_direction": WhatIfPerturbationDirection.LOWER,
        "variable_values": {"temperature": 55.0},
        "predicted_quality": None,
        "anomaly_score": -0.6,
        "objective_value": -0.6,
        "improves_over_baseline": True,
        "improves_or_matches_proposed": True,
        "extrapolated": None,
        "warnings": [],
    }
    payload.update(overrides)
    return WhatIfVerificationScenario(**payload)


def _upper_neighbor_view(**overrides: Any) -> WhatIfVerificationScenario:
    payload: dict[str, Any] = {
        "scenario_id": "WIF-000003",
        "scenario_type": WhatIfVerificationScenarioType.UPPER_NEIGHBOR,
        "perturbed_variable": "temperature",
        "perturbation_direction": WhatIfPerturbationDirection.UPPER,
        "variable_values": {"temperature": 75.0},
        "predicted_quality": None,
        "anomaly_score": 0.1,
        "objective_value": 0.1,
        "improves_over_baseline": False,
        "improves_or_matches_proposed": False,
        "extrapolated": None,
        "warnings": [],
    }
    payload.update(overrides)
    return WhatIfVerificationScenario(**payload)


def _completed_result(**overrides: Any) -> RecommendationWhatIfVerificationResult:
    scenarios = overrides.pop(
        "scenarios",
        [_baseline_view(), _proposed_view(), _lower_neighbor_view(), _upper_neighbor_view()],
    )
    neighbor_types = {
        WhatIfVerificationScenarioType.LOWER_NEIGHBOR,
        WhatIfVerificationScenarioType.UPPER_NEIGHBOR,
    }
    neighbors = [item for item in scenarios if item.scenario_type in neighbor_types]
    payload: dict[str, Any] = {
        "status": WhatIfVerificationStatus.COMPLETED,
        "objective": RecommendationObjective.REDUCE_ANOMALY_SCORE,
        "baseline_objective_value": 0.0,
        "proposed_objective_value": -0.5,
        "scenario_count": len(scenarios),
        "neighbor_scenario_count": len(neighbors),
        "improving_neighbor_count": sum(
            1 for item in neighbors if item.improves_over_baseline
        ),
        "non_improving_neighbor_count": sum(
            1 for item in neighbors if not item.improves_over_baseline
        ),
        "extrapolated_scenario_count": sum(
            1 for item in scenarios if item.extrapolated is True
        ),
        "stability_classification": WhatIfStabilityClassification.MIXED,
        "scenarios": scenarios,
        "warnings": [
            "What-if verification describes model-local adjacent-grid stability "
            "only and does not establish physical safety or causation."
        ],
        "rationale": (
            "Adjacent constraint-grid scenarios were scored with the same "
            "fitted model used for recommendation generation."
        ),
        "evaluated_at": datetime(2026, 7, 21, 15, 30, tzinfo=UTC),
        "metadata": {
            "verification_executed": True,
            "model_refit_performed": False,
            "recommendation_mutated": False,
        },
    }
    payload.update(overrides)
    return RecommendationWhatIfVerificationResult(**payload)


# ---------------------------------------------------------------------------
# Builders: fitted models + spies (mocking pattern from test_scenario_scoring.py)
# ---------------------------------------------------------------------------


def _training_frame(feature_columns: list[str], *, rows: int = 40) -> pl.DataFrame:
    rng = np.random.default_rng(11)
    return pl.DataFrame(
        {name: rng.uniform(0.0, 100.0, rows).tolist() for name in feature_columns}
    )


def _fitted_quality_model(feature_columns: list[str]) -> BaseAnalysisModel:
    frame = _training_frame(feature_columns)
    target = pl.Series("quality", (frame[feature_columns[0]] * 0.1).to_list())
    model = create_linear_regression(random_state=3)
    model.fit(frame.select(feature_columns), target)
    return model


def _fitted_anomaly_model(feature_columns: list[str]) -> BaseAnomalyModel:
    frame = _training_frame(feature_columns)
    model = create_isolation_forest_anomaly_model(
        config=IsolationForestConfig(random_state=3),
    )
    model.fit(frame.select(feature_columns))
    return model


class _QualityModelSpy(BaseAnalysisModel):
    """Thin wrapper controlling predict() output and counting calls."""

    def __init__(self, inner: BaseAnalysisModel) -> None:
        self._inner = inner
        self.predict_call_count = 0
        self.last_predict_frame: pl.DataFrame | None = None
        self.custom_predict_return: object | None = None

    @property
    def is_fitted(self) -> bool:
        return self._inner.is_fitted

    @property
    def feature_names(self) -> tuple[str, ...] | list[str]:
        return self._inner.feature_names

    def get_metadata(self) -> object:
        return self._inner.get_metadata()

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> _QualityModelSpy:
        self._inner.fit(X, y)
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        self.predict_call_count += 1
        self.last_predict_frame = frame
        if self.custom_predict_return is not None:
            return np.asarray(self.custom_predict_return)
        return self._inner.predict(frame)

    def evaluate(self, X: pl.DataFrame, y: pl.Series | None = None) -> object:
        return self._inner.evaluate(X, y)

    def explain(self, X: pl.DataFrame) -> object:
        return self._inner.explain(X)


class _AnomalyModelSpy(BaseAnomalyModel):
    """Thin wrapper controlling score_samples() output and counting calls."""

    def __init__(self, inner: BaseAnomalyModel) -> None:
        self._inner = inner
        self.score_call_count = 0
        self.last_score_frame: pl.DataFrame | None = None
        self.custom_score_return: object | None = None

    @property
    def is_fitted(self) -> bool:
        return self._inner.is_fitted

    @property
    def feature_names(self) -> tuple[str, ...] | list[str]:
        return self._inner.feature_names

    def get_metadata(self) -> object:
        return self._inner.get_metadata()

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> _AnomalyModelSpy:
        self._inner.fit(X, y)
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return self._inner.predict(frame)

    def score_samples(self, frame: pl.DataFrame) -> np.ndarray:
        self.score_call_count += 1
        self.last_score_frame = frame
        if self.custom_score_return is not None:
            return np.asarray(self.custom_score_return)
        return self._inner.score_samples(frame)

    def evaluate(self, X: pl.DataFrame, y: pl.Series | None = None) -> object:
        return self._inner.evaluate(X, y)

    def explain(self, X: pl.DataFrame) -> object:
        return self._inner.explain(X)

    def classify_anomalies(self, X: pl.DataFrame) -> object:
        return self._inner.classify_anomalies(X)


# ---------------------------------------------------------------------------
# Shared fixture-scale scenario: single controllable "temperature" variable
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = ["temperature", "pressure", "sensor_a"]
BASELINE_FEATURES = {"temperature": 50.0, "pressure": 50.0, "sensor_a": 10.0}


def _single_var_setup(
    *,
    objective: RecommendationObjective = RecommendationObjective.REDUCE_ANOMALY_SCORE,
    proposed_value: float = 65.0,
    extra_values: tuple[float, ...] = (55.0, 65.0, 75.0),
) -> tuple[RecommendationResult, CandidateGridReport]:
    grid = _variable_grid("temperature", extra_values=extra_values)
    grid_report = _grid_report(
        candidate_variables=["temperature"],
        grids={"temperature": grid},
        objective=objective,
    )
    change = _change("temperature", current_value=50.0, proposed_value=proposed_value)
    recommendation = _generated_result(
        objective=objective,
        changes=[change],
        eligible_variables=["temperature"],
        current_values={"temperature": 50.0},
    )
    return recommendation, grid_report


def _verify(
    *,
    recommendation: RecommendationResult,
    grid_report: CandidateGridReport,
    quality_model: BaseAnalysisModel | None = None,
    anomaly_model: BaseAnomalyModel | None = None,
    feature_columns: list[str] | None = None,
    baseline_features: dict[str, float] | None = None,
    quality_direction: QualityOptimizationDirection | None = None,
    quality_target: float | None = None,
    target_column: str | None = None,
    extrapolation_flag: bool | None = None,
) -> RecommendationWhatIfVerificationOutcome:
    scorer = CandidateScenarioScorer(quality_model=quality_model, anomaly_model=anomaly_model)
    verifier = RecommendationWhatIfVerifier(scenario_scorer=scorer)
    return verifier.verify(
        recommendation=recommendation,
        grid_report=grid_report,
        feature_columns=feature_columns or list(FEATURE_COLUMNS),
        baseline_features=baseline_features or dict(BASELINE_FEATURES),
        quality_direction=quality_direction,
        quality_target=quality_target,
        target_column=target_column,
        extrapolation_flag=extrapolation_flag,
    )


# ===========================================================================
# Schema tests
# ===========================================================================


def test_not_applicable_outcome_is_valid() -> None:
    verifier = RecommendationWhatIfVerifier(scenario_scorer=CandidateScenarioScorer())
    outcome = verifier.not_applicable(objective=RecommendationObjective.REDUCE_ANOMALY_SCORE)
    result = outcome.result
    assert result.status is WhatIfVerificationStatus.NOT_APPLICABLE
    assert result.scenarios == []
    assert result.scenario_count == 0
    assert result.stability_classification is WhatIfStabilityClassification.UNAVAILABLE
    assert result.baseline_objective_value is None
    assert result.proposed_objective_value is None


def test_unavailable_outcome_is_valid() -> None:
    verifier = RecommendationWhatIfVerifier(scenario_scorer=CandidateScenarioScorer())
    outcome = verifier.unavailable(
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        reason="Candidate constraint grid is unavailable.",
    )
    result = outcome.result
    assert result.status is WhatIfVerificationStatus.UNAVAILABLE
    assert result.scenarios == []
    assert result.stability_classification is WhatIfStabilityClassification.UNAVAILABLE
    assert "Candidate constraint grid is unavailable." in result.warnings


def test_completed_result_is_valid() -> None:
    result = _completed_result()
    assert result.status is WhatIfVerificationStatus.COMPLETED
    assert result.scenario_count == 4
    assert result.neighbor_scenario_count == 2
    assert result.improving_neighbor_count == 1
    assert result.non_improving_neighbor_count == 1
    assert result.stability_classification is WhatIfStabilityClassification.MIXED


@pytest.mark.parametrize(
    ("proposed_improves", "neighbor_improves", "expected"),
    [
        (True, [True, True], WhatIfStabilityClassification.STABLE),
        (True, [True, False], WhatIfStabilityClassification.MIXED),
        (True, [False, True], WhatIfStabilityClassification.MIXED),
        (True, [False, False], WhatIfStabilityClassification.ISOLATED),
        (True, [], WhatIfStabilityClassification.NO_NEIGHBORS),
        (False, [True, True], WhatIfStabilityClassification.NOT_IMPROVING),
        (False, [], WhatIfStabilityClassification.NOT_IMPROVING),
    ],
)
def test_classify_what_if_stability_all_classifications(
    proposed_improves: bool,
    neighbor_improves: list[bool],
    expected: WhatIfStabilityClassification,
) -> None:
    assert (
        classify_what_if_stability(
            proposed_improves_over_baseline=proposed_improves,
            neighbor_improves=neighbor_improves,
        )
        is expected
    )


def test_not_applicable_and_unavailable_use_unavailable_stability_only() -> None:
    # UNAVAILABLE stability classification is never produced by
    # classify_what_if_stability(); it is only used directly by
    # not_applicable()/unavailable() and forbidden for COMPLETED.
    with pytest.raises(ValidationError):
        _completed_result(stability_classification=WhatIfStabilityClassification.UNAVAILABLE)


def test_scenario_rejects_nan_variable_value() -> None:
    with pytest.raises(ValidationError):
        _baseline_view(variable_values={"temperature": float("nan")})


def test_scenario_rejects_inf_variable_value() -> None:
    with pytest.raises(ValidationError):
        _baseline_view(variable_values={"temperature": float("inf")})


def test_scenario_rejects_nan_objective_value() -> None:
    with pytest.raises(ValidationError):
        _baseline_view(objective_value=float("nan"))


def test_scenario_rejects_inf_anomaly_score() -> None:
    with pytest.raises(ValidationError):
        _baseline_view(anomaly_score=float("-inf"))


def test_result_rejects_nan_baseline_objective_value() -> None:
    with pytest.raises(ValidationError):
        _completed_result(baseline_objective_value=float("nan"))


def test_scenario_rejects_non_strict_bool_improves_over_baseline() -> None:
    with pytest.raises(ValidationError):
        _proposed_view(improves_over_baseline=1)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        _proposed_view(improves_over_baseline="true")  # type: ignore[arg-type]


def test_scenario_rejects_non_strict_bool_extrapolated() -> None:
    with pytest.raises(ValidationError):
        _proposed_view(extrapolated=0)  # type: ignore[arg-type]


def test_outcome_wrapper_is_frozen() -> None:
    # ``WhatIfVerificationScenario``/``RecommendationWhatIfVerificationResult``
    # are plain (mutable) pydantic models, matching this codebase's existing
    # pattern (see e.g. ``CandidateGridReport``). The immutability guarantee
    # that ``RecommendationWhatIfVerifier`` actually provides is the frozen
    # ``RecommendationWhatIfVerificationOutcome`` dataclass wrapper returned
    # from ``verify()``/``not_applicable()``/``unavailable()``.
    outcome = RecommendationWhatIfVerificationOutcome(result=_completed_result())
    with pytest.raises(FrozenInstanceError):
        outcome.result = _completed_result()  # type: ignore[misc]


def test_outcome_wrapper_has_slots() -> None:
    assert hasattr(RecommendationWhatIfVerificationOutcome, "__slots__")


def test_result_scenarios_are_deep_copied() -> None:
    scenario = _baseline_view()
    scenarios = [scenario, _proposed_view()]
    result = _completed_result(
        scenarios=scenarios,
        scenario_count=2,
        neighbor_scenario_count=0,
        improving_neighbor_count=0,
        non_improving_neighbor_count=0,
        stability_classification=WhatIfStabilityClassification.NO_NEIGHBORS,
    )
    original_copy = scenario.model_copy(deep=True)
    # Mutating the caller's list must not affect the stored result.
    scenarios.append(_lower_neighbor_view())
    assert len(result.scenarios) == 2
    assert result.scenarios[0] == original_copy


def test_scenario_json_round_trip() -> None:
    scenario = _lower_neighbor_view()
    dumped = json.dumps(scenario.model_dump(mode="json"))
    restored = WhatIfVerificationScenario.model_validate(json.loads(dumped))
    assert restored == scenario


def test_result_json_round_trip() -> None:
    result = _completed_result()
    dumped = json.dumps(result.model_dump(mode="json"))
    restored = RecommendationWhatIfVerificationResult.model_validate(json.loads(dumped))
    assert restored == result


def test_scenario_warnings_default_is_independent_per_instance() -> None:
    first = _baseline_view()
    second = _baseline_view()
    assert first.warnings == []
    assert second.warnings == []
    # Pydantic model instances are frozen; confirm the underlying default_factory
    # produced independent list objects (not a shared mutable default).
    assert first.warnings is not second.warnings


def test_result_warnings_default_is_independent_per_instance() -> None:
    first = _completed_result()
    second = _completed_result()
    assert first.warnings is not second.warnings


def test_scenario_id_must_match_wif_format() -> None:
    with pytest.raises(ValidationError):
        _baseline_view(scenario_id="SCN-000000")
    with pytest.raises(ValidationError):
        _baseline_view(scenario_id="WIF-1")


def test_result_requires_baseline_and_proposed_scenario_types() -> None:
    with pytest.raises(ValidationError):
        _completed_result(
            scenarios=[_proposed_view(scenario_id="WIF-000000")],
            scenario_count=1,
            neighbor_scenario_count=0,
            improving_neighbor_count=0,
            non_improving_neighbor_count=0,
            stability_classification=WhatIfStabilityClassification.NO_NEIGHBORS,
        )


def test_result_rejects_duplicate_scenario_ids() -> None:
    with pytest.raises(ValidationError):
        _completed_result(
            scenarios=[
                _baseline_view(),
                _proposed_view(scenario_id="WIF-000000"),
            ],
            scenario_count=2,
            neighbor_scenario_count=0,
            improving_neighbor_count=0,
            non_improving_neighbor_count=0,
            stability_classification=WhatIfStabilityClassification.NO_NEIGHBORS,
        )


def test_not_applicable_requires_empty_scenarios() -> None:
    with pytest.raises(ValidationError):
        _completed_result(status=WhatIfVerificationStatus.NOT_APPLICABLE)


# ===========================================================================
# Neighbor generation tests (through the public verify() API)
# ===========================================================================


def test_verify_center_scenario_matches_proposed_value() -> None:
    recommendation, grid_report = _single_var_setup(proposed_value=65.0)
    anomaly_model = _fitted_anomaly_model(FEATURE_COLUMNS)
    outcome = _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_model,
    )
    result = outcome.result
    assert result.status is WhatIfVerificationStatus.COMPLETED
    center = next(
        item
        for item in result.scenarios
        if item.scenario_type is WhatIfVerificationScenarioType.PROPOSED_CENTER
    )
    assert center.variable_values == {"temperature": 65.0}


def test_verify_interior_point_has_lower_and_upper_neighbors() -> None:
    recommendation, grid_report = _single_var_setup(proposed_value=65.0)
    anomaly_model = _fitted_anomaly_model(FEATURE_COLUMNS)
    outcome = _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_model,
    )
    result = outcome.result
    types = [item.scenario_type for item in result.scenarios]
    assert types == [
        WhatIfVerificationScenarioType.BASELINE,
        WhatIfVerificationScenarioType.PROPOSED_CENTER,
        WhatIfVerificationScenarioType.LOWER_NEIGHBOR,
        WhatIfVerificationScenarioType.UPPER_NEIGHBOR,
    ]
    lower = result.scenarios[2]
    upper = result.scenarios[3]
    assert lower.variable_values == {"temperature": 55.0}
    assert upper.variable_values == {"temperature": 75.0}
    assert lower.perturbation_direction is WhatIfPerturbationDirection.LOWER
    assert upper.perturbation_direction is WhatIfPerturbationDirection.UPPER
    assert result.neighbor_scenario_count == 2


def test_verify_boundary_minimum_has_no_lower_neighbor() -> None:
    recommendation, grid_report = _single_var_setup(
        proposed_value=0.0,
        extra_values=(15.0, 65.0),
    )
    anomaly_model = _fitted_anomaly_model(FEATURE_COLUMNS)
    outcome = _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_model,
        baseline_features={"temperature": 50.0, "pressure": 50.0, "sensor_a": 10.0},
    )
    result = outcome.result
    assert result.status is WhatIfVerificationStatus.COMPLETED
    neighbor_types = {item.scenario_type for item in result.scenarios}
    assert WhatIfVerificationScenarioType.LOWER_NEIGHBOR not in neighbor_types
    assert WhatIfVerificationScenarioType.UPPER_NEIGHBOR in neighbor_types
    assert result.neighbor_scenario_count == 1


def test_verify_boundary_maximum_has_no_upper_neighbor() -> None:
    recommendation, grid_report = _single_var_setup(
        proposed_value=100.0,
        extra_values=(15.0, 65.0),
    )
    anomaly_model = _fitted_anomaly_model(FEATURE_COLUMNS)
    outcome = _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_model,
    )
    result = outcome.result
    assert result.status is WhatIfVerificationStatus.COMPLETED
    neighbor_types = {item.scenario_type for item in result.scenarios}
    assert WhatIfVerificationScenarioType.UPPER_NEIGHBOR not in neighbor_types
    assert WhatIfVerificationScenarioType.LOWER_NEIGHBOR in neighbor_types
    assert result.neighbor_scenario_count == 1


def test_verify_single_point_grid_reuses_baseline_score_for_coincident_neighbor() -> None:
    # Regression test: a 2-point grid where the only neighbor of the proposed
    # value is numerically identical to the current/baseline value. This must
    # not raise (previously crashed with an uncaught pydantic ValidationError
    # inside _build_scenarios); the duplicate neighbor row must reuse the
    # baseline's exact score instead of being rescored.
    grid = _variable_grid("temperature", current=50.0, minimum=0.0, maximum=60.0, extra_values=())
    grid_report = _grid_report(
        candidate_variables=["temperature"],
        grids={"temperature": grid},
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
    )
    change = _change("temperature", current_value=50.0, proposed_value=60.0)
    recommendation = _generated_result(
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        changes=[change],
        eligible_variables=["temperature"],
        current_values={"temperature": 50.0},
    )
    anomaly_model = _fitted_anomaly_model(["temperature"])
    outcome = _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_model,
        feature_columns=["temperature"],
        baseline_features={"temperature": 50.0},
    )
    result = outcome.result
    assert result.status is WhatIfVerificationStatus.COMPLETED
    assert result.scenario_count == 3
    assert result.neighbor_scenario_count == 1
    baseline = result.scenarios[0]
    lower_neighbor = next(
        item
        for item in result.scenarios
        if item.scenario_type is WhatIfVerificationScenarioType.LOWER_NEIGHBOR
    )
    assert lower_neighbor.variable_values == {"temperature": 50.0}
    # The duplicate neighbor's score must exactly match the baseline's score
    # (same feature row scored once, reused rather than rescored).
    assert lower_neighbor.anomaly_score == baseline.anomaly_score
    assert lower_neighbor.improves_over_baseline is False


def test_verify_two_variable_ofat_has_no_cartesian_combinations() -> None:
    temperature_grid = _variable_grid("temperature")
    pressure_grid = _variable_grid("pressure", diagnosis_rank=2)
    grid_report = _grid_report(
        candidate_variables=["temperature", "pressure"],
        grids={"temperature": temperature_grid, "pressure": pressure_grid},
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
    )
    changes = [
        _change("temperature", current_value=50.0, proposed_value=65.0),
        _change("pressure", current_value=50.0, proposed_value=65.0),
    ]
    recommendation = _generated_result(
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        changes=changes,
        eligible_variables=["temperature", "pressure"],
        current_values={"temperature": 50.0, "pressure": 50.0},
    )
    anomaly_model = _fitted_anomaly_model(FEATURE_COLUMNS)
    outcome = _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_model,
    )
    result = outcome.result
    assert result.status is WhatIfVerificationStatus.COMPLETED
    # baseline + proposed + (2 neighbors x 2 variables) = 6, never a Cartesian
    # product of both variables perturbed simultaneously in one scenario.
    assert result.scenario_count == 6
    assert result.neighbor_scenario_count == 4
    for scenario in result.scenarios:
        if scenario.scenario_type in {
            WhatIfVerificationScenarioType.LOWER_NEIGHBOR,
            WhatIfVerificationScenarioType.UPPER_NEIGHBOR,
        }:
            # Each neighbor perturbs exactly one variable; the compact view
            # still reports both changed candidate variables (temperature and
            # pressure), but only one differs from its own proposed value.
            assert set(scenario.variable_values) == {"temperature", "pressure"}
            perturbed = scenario.perturbed_variable
            assert perturbed is not None
            other = "pressure" if perturbed == "temperature" else "temperature"
            proposed_other_value = 65.0
            assert scenario.variable_values[other] == proposed_other_value


def test_verify_neighbor_count_capped_at_2_per_changed_variable() -> None:
    temperature_grid = _variable_grid("temperature")
    pressure_grid = _variable_grid("pressure", diagnosis_rank=2)
    grid_report = _grid_report(
        candidate_variables=["temperature", "pressure"],
        grids={"temperature": temperature_grid, "pressure": pressure_grid},
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
    )
    changes = [
        _change("temperature", current_value=50.0, proposed_value=65.0),
        _change("pressure", current_value=50.0, proposed_value=65.0),
    ]
    recommendation = _generated_result(
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        changes=changes,
        eligible_variables=["temperature", "pressure"],
        current_values={"temperature": 50.0, "pressure": 50.0},
    )
    anomaly_model = _fitted_anomaly_model(FEATURE_COLUMNS)
    outcome = _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_model,
    )
    result = outcome.result
    change_count = len(changes)
    assert result.neighbor_scenario_count <= 2 * change_count
    assert result.scenario_count <= 2 + 2 * change_count


def test_verify_tolerance_based_grid_point_lookup() -> None:
    recommendation, grid_report = _single_var_setup(proposed_value=65.0 + 4e-13)
    anomaly_model = _fitted_anomaly_model(FEATURE_COLUMNS)
    outcome = _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_model,
    )
    result = outcome.result
    assert result.status is WhatIfVerificationStatus.COMPLETED
    assert result.neighbor_scenario_count == 2


def test_verify_proposed_value_not_on_grid_returns_unavailable() -> None:
    recommendation, grid_report = _single_var_setup(proposed_value=65.0)
    # Mutate the recommendation's change to a value absent from the grid.
    bad_change = _change("temperature", current_value=50.0, proposed_value=62.5)
    bad_recommendation = recommendation.model_copy(
        update={"changes": [bad_change]},
    )
    anomaly_model = _fitted_anomaly_model(FEATURE_COLUMNS)
    outcome = _verify(
        recommendation=bad_recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_model,
    )
    result = outcome.result
    assert result.status is WhatIfVerificationStatus.UNAVAILABLE
    assert result.stability_classification is WhatIfStabilityClassification.UNAVAILABLE


def test_verify_scenario_order_is_deterministic_across_runs() -> None:
    recommendation, grid_report = _single_var_setup(proposed_value=65.0)
    anomaly_model_a = _fitted_anomaly_model(FEATURE_COLUMNS)
    anomaly_model_b = _fitted_anomaly_model(FEATURE_COLUMNS)
    outcome_a = _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_model_a,
    )
    outcome_b = _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_model_b,
    )
    order_a = [item.scenario_type for item in outcome_a.result.scenarios]
    order_b = [item.scenario_type for item in outcome_b.result.scenarios]
    assert order_a == order_b
    ids_a = [item.scenario_id for item in outcome_a.result.scenarios]
    ids_b = [item.scenario_id for item in outcome_b.result.scenarios]
    assert ids_a == ids_b == ["WIF-000000", "WIF-000001", "WIF-000002", "WIF-000003"]


def test_verify_does_not_mutate_input_grid_report() -> None:
    recommendation, grid_report = _single_var_setup(proposed_value=65.0)
    before = grid_report.model_copy(deep=True)
    anomaly_model = _fitted_anomaly_model(FEATURE_COLUMNS)
    _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_model,
    )
    assert grid_report == before


def test_verify_does_not_mutate_input_recommendation() -> None:
    recommendation, grid_report = _single_var_setup(proposed_value=65.0)
    before = recommendation.model_copy(deep=True)
    anomaly_model = _fitted_anomaly_model(FEATURE_COLUMNS)
    _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_model,
    )
    assert recommendation == before


def test_verify_compact_variable_mapping_excludes_unchanged_candidates() -> None:
    # temperature and pressure are both candidate variables (both have grids),
    # but only temperature was actually changed by the recommendation. The
    # compact per-scenario variable_values view must show only "temperature".
    temperature_grid = _variable_grid("temperature")
    pressure_grid = _variable_grid("pressure", diagnosis_rank=2)
    grid_report = _grid_report(
        candidate_variables=["temperature", "pressure"],
        grids={"temperature": temperature_grid, "pressure": pressure_grid},
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
    )
    change = _change("temperature", current_value=50.0, proposed_value=65.0)
    recommendation = _generated_result(
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        changes=[change],
        eligible_variables=["temperature", "pressure"],
        current_values={"temperature": 50.0, "pressure": 50.0},
    )
    anomaly_model = _fitted_anomaly_model(FEATURE_COLUMNS)
    outcome = _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_model,
    )
    result = outcome.result
    assert result.status is WhatIfVerificationStatus.COMPLETED
    for scenario in result.scenarios:
        assert set(scenario.variable_values) == {"temperature"}


# ===========================================================================
# Scoring contract tests (mocks)
# ===========================================================================


def test_verify_scores_all_scenarios_in_a_single_batch_call() -> None:
    recommendation, grid_report = _single_var_setup(proposed_value=65.0)
    anomaly_spy = _AnomalyModelSpy(_fitted_anomaly_model(FEATURE_COLUMNS))
    _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_spy,
    )
    assert anomaly_spy.score_call_count == 1
    assert anomaly_spy.last_score_frame is not None
    assert anomaly_spy.last_score_frame.height == 4


def test_verify_does_not_refit_models() -> None:
    recommendation, grid_report = _single_var_setup(proposed_value=65.0)
    anomaly_model = _fitted_anomaly_model(FEATURE_COLUMNS)
    fit_calls_before = getattr(anomaly_model, "_fit_call_count", None)
    is_fitted_before = anomaly_model.is_fitted
    outcome = _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_model,
    )
    assert anomaly_model.is_fitted == is_fitted_before
    assert getattr(anomaly_model, "_fit_call_count", None) == fit_calls_before
    assert outcome.result.metadata["model_refit_performed"] is False


def test_verify_anomaly_objective_lower_is_better() -> None:
    recommendation, grid_report = _single_var_setup(
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        proposed_value=65.0,
    )
    anomaly_spy = _AnomalyModelSpy(_fitted_anomaly_model(FEATURE_COLUMNS))
    # baseline, proposed, lower, upper
    anomaly_spy.custom_score_return = [0.0, -0.5, -0.6, 0.1]
    outcome = _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_spy,
    )
    result = outcome.result
    proposed = result.scenarios[1]
    lower = result.scenarios[2]
    upper = result.scenarios[3]
    assert proposed.improves_over_baseline is True  # -0.5 < 0.0
    assert lower.improves_over_baseline is True  # -0.6 < 0.0
    assert upper.improves_over_baseline is False  # 0.1 > 0.0
    assert result.baseline_objective_value == 0.0
    assert result.proposed_objective_value == -0.5


def test_verify_quality_objective_maximize() -> None:
    recommendation, grid_report = _single_var_setup(
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        proposed_value=65.0,
    )
    quality_spy = _QualityModelSpy(_fitted_quality_model(FEATURE_COLUMNS))
    quality_spy.custom_predict_return = [80.0, 85.0, 86.0, 84.0]
    outcome = _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        quality_model=quality_spy,
        quality_direction=QualityOptimizationDirection.MAXIMIZE,
    )
    result = outcome.result
    proposed, lower, upper = result.scenarios[1], result.scenarios[2], result.scenarios[3]
    assert proposed.improves_over_baseline is True
    assert lower.improves_over_baseline is True
    assert upper.improves_over_baseline is True
    assert result.stability_classification is WhatIfStabilityClassification.STABLE


def test_verify_quality_objective_minimize() -> None:
    recommendation, grid_report = _single_var_setup(
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        proposed_value=65.0,
    )
    quality_spy = _QualityModelSpy(_fitted_quality_model(FEATURE_COLUMNS))
    # Lower is better: baseline=20, proposed=15 (improves), lower=14 (improves), upper=25 (worse)
    quality_spy.custom_predict_return = [20.0, 15.0, 14.0, 25.0]
    outcome = _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        quality_model=quality_spy,
        quality_direction=QualityOptimizationDirection.MINIMIZE,
    )
    result = outcome.result
    proposed, lower, upper = result.scenarios[1], result.scenarios[2], result.scenarios[3]
    assert proposed.improves_over_baseline is True
    assert lower.improves_over_baseline is True
    assert upper.improves_over_baseline is False


def test_verify_quality_objective_target() -> None:
    recommendation, grid_report = _single_var_setup(
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        proposed_value=65.0,
    )
    quality_spy = _QualityModelSpy(_fitted_quality_model(FEATURE_COLUMNS))
    # target=50: baseline=40 (|40-50|=10), proposed=48 (|48-50|=2, improves),
    # lower=55 (|55-50|=5, improves over baseline's 10 but worse than proposed),
    # upper=70 (|70-50|=20, does not improve)
    quality_spy.custom_predict_return = [40.0, 48.0, 55.0, 70.0]
    outcome = _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        quality_model=quality_spy,
        quality_direction=QualityOptimizationDirection.TARGET,
        quality_target=50.0,
    )
    result = outcome.result
    proposed, lower, upper = result.scenarios[1], result.scenarios[2], result.scenarios[3]
    assert proposed.improves_over_baseline is True
    assert lower.improves_over_baseline is True
    assert upper.improves_over_baseline is False


def test_verify_negative_anomaly_scores_are_preserved() -> None:
    recommendation, grid_report = _single_var_setup(proposed_value=65.0)
    anomaly_spy = _AnomalyModelSpy(_fitted_anomaly_model(FEATURE_COLUMNS))
    anomaly_spy.custom_score_return = [-0.9, -0.95, -0.99, -0.2]
    outcome = _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_spy,
    )
    result = outcome.result
    for scenario, expected in zip(result.scenarios, [-0.9, -0.95, -0.99, -0.2], strict=True):
        assert scenario.anomaly_score == expected
        assert scenario.objective_value == expected


def test_verify_proposed_scenario_never_marked_improved_over_itself() -> None:
    recommendation, grid_report = _single_var_setup(proposed_value=65.0)
    anomaly_spy = _AnomalyModelSpy(_fitted_anomaly_model(FEATURE_COLUMNS))
    anomaly_spy.custom_score_return = [0.0, -0.5, -0.6, 0.1]
    outcome = _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_spy,
    )
    baseline, proposed = outcome.result.scenarios[0], outcome.result.scenarios[1]
    assert baseline.improves_over_baseline is False
    # PROPOSED_CENTER's improves_over_baseline compares against the true
    # baseline, and here it genuinely improves (-0.5 < 0.0).
    assert proposed.improves_over_baseline is True
    assert proposed.improves_or_matches_proposed is True


# ===========================================================================
# Stability classification rules (direct unit tests)
# ===========================================================================


def test_classify_not_improving_when_proposed_regresses() -> None:
    assert (
        classify_what_if_stability(
            proposed_improves_over_baseline=False,
            neighbor_improves=[True, True],
        )
        is WhatIfStabilityClassification.NOT_IMPROVING
    )


def test_classify_not_improving_takes_precedence_over_neighbor_flags() -> None:
    # Even when every neighbor "improves", a non-improving proposed center
    # is NOT_IMPROVING: neighbor flags are irrelevant once the primary
    # recommendation itself fails to improve on the baseline.
    assert (
        classify_what_if_stability(
            proposed_improves_over_baseline=False,
            neighbor_improves=[],
        )
        is WhatIfStabilityClassification.NOT_IMPROVING
    )


def test_classify_no_neighbors_when_proposed_improves_but_list_empty() -> None:
    assert (
        classify_what_if_stability(
            proposed_improves_over_baseline=True,
            neighbor_improves=[],
        )
        is WhatIfStabilityClassification.NO_NEIGHBORS
    )


def test_classify_stable_when_every_neighbor_improves() -> None:
    assert (
        classify_what_if_stability(
            proposed_improves_over_baseline=True,
            neighbor_improves=[True, True],
        )
        is WhatIfStabilityClassification.STABLE
    )


def test_classify_stable_with_single_improving_neighbor() -> None:
    assert (
        classify_what_if_stability(
            proposed_improves_over_baseline=True,
            neighbor_improves=[True],
        )
        is WhatIfStabilityClassification.STABLE
    )


def test_classify_isolated_when_no_neighbor_improves() -> None:
    assert (
        classify_what_if_stability(
            proposed_improves_over_baseline=True,
            neighbor_improves=[False, False],
        )
        is WhatIfStabilityClassification.ISOLATED
    )


def test_classify_mixed_when_some_neighbors_improve() -> None:
    assert (
        classify_what_if_stability(
            proposed_improves_over_baseline=True,
            neighbor_improves=[True, False],
        )
        is WhatIfStabilityClassification.MIXED
    )
    assert (
        classify_what_if_stability(
            proposed_improves_over_baseline=True,
            neighbor_improves=[False, True],
        )
        is WhatIfStabilityClassification.MIXED
    )


# ===========================================================================
# Stability classification via end-to-end verify()
# ===========================================================================


def test_stability_not_improving_when_proposed_regresses() -> None:
    recommendation, grid_report = _single_var_setup(proposed_value=65.0)
    anomaly_spy = _AnomalyModelSpy(_fitted_anomaly_model(FEATURE_COLUMNS))
    anomaly_spy.custom_score_return = [0.0, 0.5, -0.6, -0.7]
    outcome = _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_spy,
    )
    assert outcome.result.stability_classification is WhatIfStabilityClassification.NOT_IMPROVING


def test_stability_at_a_boundary_with_a_single_baseline_coincident_neighbor() -> None:
    # A ``VariableCandidateGrid`` always has at least a minimum and a maximum
    # point distinct from each other, so proposing the boundary value
    # opposite the current value ("0.0" here, with current="50.0" and no
    # extra grid points) yields exactly one OFAT neighbor -- and that
    # neighbor is numerically identical to the true baseline. Genuine
    # "zero neighbors" is therefore unreachable through verify() itself;
    # see the direct classify_what_if_stability() tests above for that case.
    recommendation, grid_report = _single_var_setup(
        proposed_value=0.0,
        extra_values=(),
    )
    anomaly_model = _fitted_anomaly_model(FEATURE_COLUMNS)
    outcome = _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_model,
        baseline_features={"temperature": 50.0, "pressure": 50.0, "sensor_a": 10.0},
    )
    result = outcome.result
    assert result.status is WhatIfVerificationStatus.COMPLETED
    assert result.neighbor_scenario_count == 1
    neighbor = result.scenarios[-1]
    assert neighbor.perturbation_direction is WhatIfPerturbationDirection.UPPER
    # The lone neighbor's score is reused from baseline (deduplication), so
    # it can never register as strictly improving over baseline.
    assert neighbor.improves_over_baseline is False
    if result.scenarios[1].improves_over_baseline:
        assert result.stability_classification is WhatIfStabilityClassification.ISOLATED
    else:
        assert result.stability_classification is WhatIfStabilityClassification.NOT_IMPROVING


def test_stability_mixed_via_anomaly_only_fixture() -> None:
    outcome = _run_anomaly_only_mixed_fixture()
    assert outcome.result.stability_classification is WhatIfStabilityClassification.MIXED
    assert "Recommendation is locally mixed under adjacent grid perturbations." in (
        outcome.result.warnings
    )


def test_stability_isolated_when_no_neighbor_improves() -> None:
    recommendation, grid_report = _single_var_setup(proposed_value=65.0)
    anomaly_spy = _AnomalyModelSpy(_fitted_anomaly_model(FEATURE_COLUMNS))
    anomaly_spy.custom_score_return = [0.0, -0.5, 0.2, 0.3]
    outcome = _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_spy,
    )
    result = outcome.result
    assert result.stability_classification is WhatIfStabilityClassification.ISOLATED
    assert "Recommendation improvement was isolated to the proposed grid point." in (
        result.warnings
    )


def test_stability_stable_via_supervised_fixture() -> None:
    outcome = _run_supervised_stable_fixture()
    assert outcome.result.stability_classification is WhatIfStabilityClassification.STABLE


# ===========================================================================
# ANOMALY_ONLY MIXED and SUPERVISED STABLE end-to-end fixtures
# ===========================================================================


def _run_anomaly_only_mixed_fixture() -> RecommendationWhatIfVerificationOutcome:
    """Controllable temperature + pressure, sensor_a passthrough, GENERATED
    recommendation at an interior grid point where the lower neighbor
    improves and the upper neighbor does not, under the same fitted
    anomaly model (no quality model, ANOMALY_ONLY objective)."""
    temperature_grid = _variable_grid("temperature")
    pressure_grid = _variable_grid("pressure", diagnosis_rank=2)
    grid_report = _grid_report(
        candidate_variables=["temperature", "pressure"],
        grids={"temperature": temperature_grid, "pressure": pressure_grid},
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
    )
    change = _change("temperature", current_value=50.0, proposed_value=65.0)
    recommendation = _generated_result(
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        changes=[change],
        eligible_variables=["temperature", "pressure"],
        current_values={"temperature": 50.0, "pressure": 50.0},
    )
    anomaly_spy = _AnomalyModelSpy(_fitted_anomaly_model(FEATURE_COLUMNS))
    # order: baseline, proposed, lower(55), upper(75)
    anomaly_spy.custom_score_return = [0.0, -0.4, -0.55, 0.2]
    return _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_spy,
    )


def _run_supervised_stable_fixture() -> RecommendationWhatIfVerificationOutcome:
    """Quality target objective, one controllable variable ("pressure"),
    both neighbors improve under the same fitted quality model."""
    grid = _variable_grid("pressure")
    grid_report = _grid_report(
        candidate_variables=["pressure"],
        grids={"pressure": grid},
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
    )
    change = _change("pressure", current_value=50.0, proposed_value=65.0)
    recommendation = _generated_result(
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        changes=[change],
        eligible_variables=["pressure"],
        current_values={"pressure": 50.0},
    )
    quality_spy = _QualityModelSpy(_fitted_quality_model(FEATURE_COLUMNS))
    # order: baseline=80, proposed=85, lower(55)=86, upper(75)=87 (both improve)
    quality_spy.custom_predict_return = [80.0, 85.0, 86.0, 87.0]
    return _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        quality_model=quality_spy,
        quality_direction=QualityOptimizationDirection.MAXIMIZE,
    )


def test_anomaly_only_mixed_fixture_end_to_end() -> None:
    outcome = _run_anomaly_only_mixed_fixture()
    result = outcome.result
    assert result.status is WhatIfVerificationStatus.COMPLETED
    lower = result.scenarios[2]
    upper = result.scenarios[3]
    assert lower.improves_over_baseline is True
    assert upper.improves_over_baseline is False
    assert result.stability_classification is WhatIfStabilityClassification.MIXED
    assert result.metadata["model_refit_performed"] is False


def test_supervised_stable_fixture_end_to_end() -> None:
    outcome = _run_supervised_stable_fixture()
    result = outcome.result
    assert result.status is WhatIfVerificationStatus.COMPLETED
    lower = result.scenarios[2]
    upper = result.scenarios[3]
    assert lower.improves_over_baseline is True
    assert upper.improves_over_baseline is True
    assert result.stability_classification is WhatIfStabilityClassification.STABLE
    assert result.improving_neighbor_count == 2
    assert result.non_improving_neighbor_count == 0


def test_recommendation_warnings_for_stability_mapping() -> None:
    assert recommendation_warnings_for_stability(WhatIfStabilityClassification.STABLE) == []
    assert recommendation_warnings_for_stability(WhatIfStabilityClassification.NO_NEIGHBORS) == []
    mixed = recommendation_warnings_for_stability(WhatIfStabilityClassification.MIXED)
    assert len(mixed) == 1
    isolated = recommendation_warnings_for_stability(WhatIfStabilityClassification.ISOLATED)
    assert len(isolated) == 1
    assert mixed != isolated


# ===========================================================================
# Non-GENERATED recommendation status / structural unavailability
# ===========================================================================


def test_verify_ready_for_optimization_returns_not_applicable() -> None:
    recommendation = _ready_result(
        eligible_variables=["temperature"],
        current_values={"temperature": 50.0},
    )
    grid = _variable_grid("temperature")
    grid_report = _grid_report(
        candidate_variables=["temperature"],
        grids={"temperature": grid},
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
    )
    anomaly_model = _fitted_anomaly_model(FEATURE_COLUMNS)
    outcome = _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_model,
    )
    assert outcome.result.status is WhatIfVerificationStatus.NOT_APPLICABLE


def test_verify_refused_recommendation_returns_not_applicable() -> None:
    recommendation = _refused_result()
    grid = _variable_grid("temperature")
    grid_report = _grid_report(
        candidate_variables=["temperature"],
        grids={"temperature": grid},
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
    )
    anomaly_model = _fitted_anomaly_model(FEATURE_COLUMNS)
    outcome = _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_model,
    )
    assert outcome.result.status is WhatIfVerificationStatus.NOT_APPLICABLE


def test_verify_missing_grid_returns_unavailable() -> None:
    recommendation, grid_report = _single_var_setup(proposed_value=65.0)
    empty_grid_report = grid_report.model_copy(
        update={"candidate_variables": [], "variable_grids": []},
    )
    anomaly_model = _fitted_anomaly_model(FEATURE_COLUMNS)
    outcome = _verify(
        recommendation=recommendation,
        grid_report=empty_grid_report,
        anomaly_model=anomaly_model,
    )
    assert outcome.result.status is WhatIfVerificationStatus.UNAVAILABLE


def test_verify_feature_schema_mismatch_returns_unavailable() -> None:
    recommendation, grid_report = _single_var_setup(proposed_value=65.0)
    anomaly_model = _fitted_anomaly_model(FEATURE_COLUMNS)
    outcome = _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        anomaly_model=anomaly_model,
        baseline_features={"temperature": 50.0, "pressure": 50.0},  # missing sensor_a
        feature_columns=FEATURE_COLUMNS,
    )
    assert outcome.result.status is WhatIfVerificationStatus.UNAVAILABLE


def test_verify_missing_quality_direction_returns_unavailable() -> None:
    recommendation, grid_report = _single_var_setup(
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        proposed_value=65.0,
    )
    quality_model = _fitted_quality_model(FEATURE_COLUMNS)
    outcome = _verify(
        recommendation=recommendation,
        grid_report=grid_report,
        quality_model=quality_model,
        quality_direction=None,
    )
    assert outcome.result.status is WhatIfVerificationStatus.UNAVAILABLE


def test_verifier_rejects_non_scorer_scenario_scorer() -> None:
    with pytest.raises(TypeError):
        RecommendationWhatIfVerifier(scenario_scorer="not-a-scorer")  # type: ignore[arg-type]


def test_verifier_get_metadata_contract() -> None:
    verifier = RecommendationWhatIfVerifier(scenario_scorer=CandidateScenarioScorer())
    metadata = verifier.get_metadata()
    assert metadata["performs_model_refit"] is False
    assert metadata["performs_model_reselection"] is False
    assert metadata["performs_recommendation_reoptimization"] is False
    assert metadata["asserts_physical_safety"] is False
    assert metadata["asserts_causation"] is False
    assert metadata["model_local_stability_only"] is True
