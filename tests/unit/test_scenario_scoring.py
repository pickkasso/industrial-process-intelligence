"""Unit tests for candidate scenario scoring (Step 9D)."""

from __future__ import annotations

import copy
import math
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from typing import Any

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.exceptions import DataValidationError, ProcessIntelligenceError
from process_intelligence.core.protocols import BaseAnalysisModel, BaseAnomalyModel
from process_intelligence.models import (
    create_isolation_forest_anomaly_model,
    create_linear_regression,
    create_logistic_regression,
)
from process_intelligence.models.anomaly import IsolationForestConfig
from process_intelligence.recommendation import (
    CandidateGridReport,
    CandidateGridStatus,
    CandidateScenario,
    CandidateScenarioScore,
    CandidateScenarioScorer,
    CandidateScenarioType,
    CandidateValuePoint,
    ConstraintResolutionStatus,
    RecommendationObjective,
    RecommendationSafetyStatus,
    ScenarioScoringOutcome,
    ScenarioScoringPolicy,
    ScenarioScoringReport,
    ScenarioScoringRequest,
    ScenarioScoringStatus,
    VariableCandidateGrid,
)
from process_intelligence.recommendation.scenario_scoring import (
    _WARNING_ANOMALY_MODEL_MISSING,
    _WARNING_BATCH_LIMIT,
    _WARNING_FEATURE_METADATA,
    _WARNING_GRID_REFUSED,
    _WARNING_NO_CAUSATION,
    _WARNING_NO_SCENARIOS,
    _WARNING_PARTIAL,
    _WARNING_QUALITY_MODEL_MISSING,
    _WARNING_RESIDUAL,
    _WARNING_UNRANKED,
    _WARNING_VERIFICATION,
)

FEATURE_COLUMNS = ["pressure", "temperature", "humidity"]
BASELINE_FEATURES = {"pressure": 50.0, "temperature": 80.0, "humidity": 45.0}


# --- Grid / request builders (pattern from test_candidate_grid.py) ---


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
        "variable_values": {"pressure": 0.0, "temperature": 80.0},
        "changed_variables": ["pressure"],
        "deltas": {"pressure": -50.0},
        "relative_deltas": {"pressure": -1.0},
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
        "variable_values": {"pressure": 50.0, "temperature": 0.0},
        "changed_variables": ["temperature"],
        "deltas": {"temperature": -80.0},
        "relative_deltas": {"temperature": -1.0},
        "change_count": 1,
        "normalized_change_magnitude": 0.8,
    }
    payload.update(overrides)
    return CandidateScenario(**payload)


def _ready_grid_report(**overrides: Any) -> CandidateGridReport:
    baseline = _baseline_scenario()
    single = _single_scenario()
    scenarios = overrides.pop("scenarios", [baseline, single])
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


def _scoring_request(**overrides: Any) -> ScenarioScoringRequest:
    payload: dict[str, Any] = {
        "grid": _ready_grid_report(),
        "feature_columns": list(FEATURE_COLUMNS),
        "baseline_features": dict(BASELINE_FEATURES),
        "target_column": "quality",
        "metadata": {"source": "unit_test"},
    }
    payload.update(overrides)
    return ScenarioScoringRequest(**payload)


def _training_frame(*, rows: int = 40) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    pressure = rng.uniform(0.0, 100.0, rows)
    temperature = rng.uniform(0.0, 100.0, rows)
    humidity = rng.uniform(20.0, 60.0, rows)
    return pl.DataFrame(
        {
            "pressure": pressure.tolist(),
            "temperature": temperature.tolist(),
            "humidity": humidity.tolist(),
        }
    )


def _quality_target(frame: pl.DataFrame) -> pl.Series:
    values = (
        frame["pressure"] * 0.1
        + frame["temperature"] * 0.05
        + frame["humidity"] * 0.02
    )
    return pl.Series("quality", values.to_list())


def _fitted_quality_model() -> BaseAnalysisModel:
    frame = _training_frame()
    model = create_linear_regression(random_state=7)
    model.fit(frame.select(FEATURE_COLUMNS), _quality_target(frame))
    return model


def _fitted_anomaly_model() -> BaseAnomalyModel:
    frame = _training_frame()
    model = create_isolation_forest_anomaly_model(
        config=IsolationForestConfig(random_state=7),
    )
    model.fit(frame.select(FEATURE_COLUMNS))
    return model


class _QualityModelSpy(BaseAnalysisModel):
    """Thin wrapper counting predict calls and optionally raising errors."""

    def __init__(self, inner: BaseAnalysisModel) -> None:
        self._inner = inner
        self.predict_call_count = 0
        self.last_predict_frame: pl.DataFrame | None = None
        self.predict_error: BaseException | None = None
        self.mutate_fitted_on_predict = False
        self.custom_predict_return: object | None = None

    @property
    def is_fitted(self) -> bool:
        return self._inner.is_fitted

    @property
    def feature_names(self) -> tuple[str, ...] | list[str]:
        return self._inner.feature_names

    def get_metadata(self) -> object:
        return self._inner.get_metadata()

    def fit(
        self,
        X: pl.DataFrame,
        y: pl.Series | None = None,
    ) -> _QualityModelSpy:
        self._inner.fit(X, y)
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        self.predict_call_count += 1
        self.last_predict_frame = frame
        if self.mutate_fitted_on_predict:
            self._inner._is_fitted = False  # type: ignore[attr-defined]
        if self.predict_error is not None:
            raise self.predict_error
        if self.custom_predict_return is not None:
            return np.asarray(self.custom_predict_return)
        return self._inner.predict(frame)

    def evaluate(self, X: pl.DataFrame, y: pl.Series | None = None) -> object:
        return self._inner.evaluate(X, y)

    def explain(self, X: pl.DataFrame) -> object:
        return self._inner.explain(X)


class _AnomalyModelSpy(BaseAnomalyModel):
    """Thin wrapper counting score_samples calls and optionally raising errors."""

    def __init__(self, inner: BaseAnomalyModel) -> None:
        self._inner = inner
        self.score_call_count = 0
        self.last_score_frame: pl.DataFrame | None = None
        self.score_error: BaseException | None = None
        self.mutate_fitted_on_score = False
        self.custom_score_return: object | None = None

    @property
    def is_fitted(self) -> bool:
        return self._inner.is_fitted

    @property
    def feature_names(self) -> tuple[str, ...] | list[str]:
        return self._inner.feature_names

    def get_metadata(self) -> object:
        return self._inner.get_metadata()

    def fit(
        self,
        X: pl.DataFrame,
        y: pl.Series | None = None,
    ) -> _AnomalyModelSpy:
        self._inner.fit(X, y)
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return self._inner.predict(frame)

    def score_samples(self, frame: pl.DataFrame) -> np.ndarray:
        self.score_call_count += 1
        self.last_score_frame = frame
        if self.mutate_fitted_on_score:
            self._inner._is_fitted = False  # type: ignore[attr-defined]
        if self.score_error is not None:
            raise self.score_error
        if self.custom_score_return is not None:
            return np.asarray(self.custom_score_return)
        return self._inner.score_samples(frame)

    def evaluate(self, X: pl.DataFrame, y: pl.Series | None = None) -> object:
        return self._inner.evaluate(X, y)

    def explain(self, X: pl.DataFrame) -> object:
        return self._inner.explain(X)

    def classify_anomalies(self, X: pl.DataFrame) -> object:
        return self._inner.classify_anomalies(X)


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


def _scoring_report(**overrides: Any) -> ScenarioScoringReport:
    baseline_score = _scenario_score()
    change_score = _scenario_score(
        scenario_id="SCN-000001",
        scenario_index=1,
        scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
        change_count=1,
        normalized_change_magnitude=0.5,
        quality_prediction=11.0,
        anomaly_score=-0.02,
        quality_delta_from_baseline=1.0,
        anomaly_score_delta_from_baseline=0.03,
    )
    scores = overrides.pop("scores", [baseline_score, change_score])
    payload: dict[str, Any] = {
        "status": ScenarioScoringStatus.SCORED,
        "objective": RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
        "task": AnalysisTask.REGRESSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": "quality",
        "quality_model_name": "Linear Regression",
        "quality_estimator_key": "linear_regression",
        "anomaly_model_name": "Isolation Forest",
        "anomaly_estimator_key": "isolation_forest",
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
        "warnings": [_WARNING_RESIDUAL],
        "metadata": {"batch": 1},
    }
    payload.update(overrides)
    if "requested_scenario_count" not in overrides:
        payload["requested_scenario_count"] = len(scores)
    if "scored_scenario_count" not in overrides:
        payload["scored_scenario_count"] = len(scores)
    if "quality_scored_count" not in overrides:
        payload["quality_scored_count"] = sum(
            1 for item in scores if item.quality_scored
        )
    if "anomaly_scored_count" not in overrides:
        payload["anomaly_scored_count"] = sum(
            1 for item in scores if item.anomaly_scored
        )
    return ScenarioScoringReport(**payload)


def _score_with_both(
    *,
    request: ScenarioScoringRequest | None = None,
    policy: ScenarioScoringPolicy | None = None,
    quality: BaseAnalysisModel | _QualityModelSpy | None = None,
    anomaly: BaseAnomalyModel | _AnomalyModelSpy | None = None,
) -> ScenarioScoringOutcome:
    scorer = CandidateScenarioScorer(
        quality_model=quality or _fitted_quality_model(),
        anomaly_model=anomaly or _fitted_anomaly_model(),
        policy=policy,
    )
    return scorer.score(request or _scoring_request())


# --- Enum / Policy (1-7) ---


def test_scenario_scoring_status_values() -> None:
    assert list(ScenarioScoringStatus) == [
        ScenarioScoringStatus.SCORED,
        ScenarioScoringStatus.PARTIAL,
        ScenarioScoringStatus.REFUSED,
    ]
    assert ScenarioScoringStatus.SCORED == "SCORED"


def test_no_extra_scenario_scoring_status_members() -> None:
    assert len(ScenarioScoringStatus) == 3


def test_default_scenario_scoring_policy() -> None:
    policy = ScenarioScoringPolicy()
    assert policy.maximum_batch_rows == 5000
    assert policy.require_regression_task is True
    assert policy.require_fitted_models is True
    assert policy.require_exact_feature_set is True
    assert policy.allow_partial_scoring is False
    assert policy.preserve_scenario_order is True
    assert policy.require_baseline_scenario is True
    assert policy.require_finite_predictions is True
    assert policy.require_finite_anomaly_scores is True


def test_maximum_batch_rows_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        ScenarioScoringPolicy(maximum_batch_rows=0)


def test_policy_bool_field_rejected() -> None:
    with pytest.raises(ValidationError):
        ScenarioScoringPolicy(require_fitted_models=1)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ScenarioScoringPolicy(allow_partial_scoring="yes")  # type: ignore[arg-type]


def test_policy_bool_field_strict_validation() -> None:
    with pytest.raises(ValidationError):
        ScenarioScoringPolicy(preserve_scenario_order=0)  # type: ignore[arg-type]


def test_scenario_scoring_policy_round_trip() -> None:
    policy = ScenarioScoringPolicy(maximum_batch_rows=100)
    assert ScenarioScoringPolicy.model_validate(policy.model_dump()) == policy


# --- Request (8-18) ---


def test_scenario_scoring_request_valid() -> None:
    request = _scoring_request()
    assert request.target_column == "quality"
    assert request.feature_columns == FEATURE_COLUMNS


def test_request_empty_feature_columns_rejected() -> None:
    with pytest.raises(ValidationError):
        _scoring_request(feature_columns=[])


def test_request_duplicate_feature_columns_rejected() -> None:
    with pytest.raises(ValidationError):
        _scoring_request(feature_columns=["pressure", "pressure", "humidity"])


def test_request_target_column_conflict_rejected() -> None:
    with pytest.raises(ValidationError):
        _scoring_request(target_column="pressure")


def test_request_baseline_key_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        _scoring_request(
            baseline_features={"pressure": 50.0, "temperature": 80.0},
        )


def test_request_baseline_bool_nan_inf_rejected() -> None:
    with pytest.raises(ValidationError):
        _scoring_request(
            baseline_features={
                **BASELINE_FEATURES,
                "humidity": True,  # type: ignore[dict-item]
            },
        )
    with pytest.raises(ValidationError):
        _scoring_request(
            baseline_features={**BASELINE_FEATURES, "humidity": float("nan")},
        )
    with pytest.raises(ValidationError):
        _scoring_request(
            baseline_features={**BASELINE_FEATURES, "humidity": float("inf")},
        )


def test_request_candidate_not_in_features_rejected() -> None:
    grid = _ready_grid_report(candidate_variables=["pressure", "temperature"])
    with pytest.raises(ValidationError):
        _scoring_request(
            grid=grid,
            feature_columns=["pressure", "humidity"],
            baseline_features={"pressure": 50.0, "humidity": 45.0},
        )


def test_request_grid_current_vs_baseline_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        _scoring_request(
            baseline_features={
                **BASELINE_FEATURES,
                "pressure": 49.0,
            },
        )


def test_request_metadata_scalar_validation() -> None:
    with pytest.raises(ValidationError):
        _scoring_request(metadata={"bad": {"nested": 1}})


def test_request_mutable_independence() -> None:
    request = _scoring_request()
    before = request.model_dump()
    request.feature_columns.append("extra")
    request.baseline_features["pressure"] = 999.0
    request.grid.scenarios.append(
        _single_scenario(scenario_id="SCN-000002", scenario_index=2)
    )
    rebuilt = ScenarioScoringRequest.model_validate(before)
    assert rebuilt.feature_columns == FEATURE_COLUMNS
    assert rebuilt.baseline_features["pressure"] == pytest.approx(50.0)
    assert len(rebuilt.grid.scenarios) == 2


def test_scenario_scoring_request_round_trip() -> None:
    request = _scoring_request()
    assert ScenarioScoringRequest.model_validate(request.model_dump()) == request


# --- CandidateScenarioScore (19-29) ---


def test_candidate_scenario_score_quality_and_anomaly_valid() -> None:
    score = _scenario_score()
    assert score.quality_scored is True
    assert score.anomaly_scored is True
    assert score.quality_prediction == pytest.approx(10.0)
    assert score.anomaly_score == pytest.approx(-0.05)


def test_candidate_scenario_score_id_index_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        _scenario_score(scenario_id="SCN-000005", scenario_index=0)


def test_candidate_scenario_score_change_count_bool_rejected() -> None:
    with pytest.raises(ValidationError):
        _scenario_score(change_count=True)  # type: ignore[arg-type]


def test_candidate_scenario_score_nan_inf_outputs_rejected() -> None:
    with pytest.raises(ValidationError):
        _scenario_score(quality_prediction=float("nan"))
    with pytest.raises(ValidationError):
        _scenario_score(anomaly_score=float("inf"))


def test_candidate_scenario_score_scored_flag_relations() -> None:
    with pytest.raises(ValidationError):
        _scenario_score(quality_scored=True, quality_prediction=None)
    with pytest.raises(ValidationError):
        _scenario_score(
            quality_scored=False,
            quality_prediction=1.0,
            quality_delta_from_baseline=None,
            anomaly_scored=False,
            anomaly_score=None,
            anomaly_score_delta_from_baseline=None,
        )


def test_candidate_scenario_score_baseline_delta_zero() -> None:
    with pytest.raises(ValidationError):
        _scenario_score(quality_delta_from_baseline=0.5)


def test_candidate_scenario_score_warning_duplicate_rejected() -> None:
    with pytest.raises(ValidationError):
        _scenario_score(warnings=["dup", "dup"])


def test_candidate_scenario_score_round_trip() -> None:
    score = _scenario_score()
    assert CandidateScenarioScore.model_validate(score.model_dump()) == score


def test_candidate_scenario_score_quality_only_unscored_deltas_none() -> None:
    score = _scenario_score(
        anomaly_scored=False,
        anomaly_score=None,
        anomaly_score_delta_from_baseline=None,
    )
    assert score.anomaly_score_delta_from_baseline is None


# --- Report (30-45) ---


def test_scenario_scoring_report_scored_balance_valid() -> None:
    report = _scoring_report()
    assert report.status is ScenarioScoringStatus.SCORED


def test_scenario_scoring_report_scored_quality_objective() -> None:
    scores = [
        _scenario_score(
            quality_scored=True,
            anomaly_scored=False,
            anomaly_score=None,
            anomaly_score_delta_from_baseline=None,
        ),
    ]
    report = _scoring_report(
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        scores=scores,
        quality_scored_count=1,
        anomaly_scored_count=0,
        anomaly_model_name=None,
        anomaly_estimator_key=None,
    )
    assert report.status is ScenarioScoringStatus.SCORED


def test_scenario_scoring_report_scored_anomaly_objective() -> None:
    scores = [
        _scenario_score(
            quality_scored=False,
            quality_prediction=None,
            quality_delta_from_baseline=None,
            anomaly_scored=True,
        ),
    ]
    report = _scoring_report(
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        task=AnalysisTask.UNSUPERVISED_ANOMALY,
        scores=scores,
        quality_scored_count=0,
        anomaly_scored_count=1,
        quality_model_name=None,
        quality_estimator_key=None,
    )
    assert report.status is ScenarioScoringStatus.SCORED


def test_scenario_scoring_report_partial_balance_valid() -> None:
    scores = [
        _scenario_score(
            quality_scored=True,
            anomaly_scored=False,
            anomaly_score=None,
            anomaly_score_delta_from_baseline=None,
        ),
        _scenario_score(
            scenario_id="SCN-000001",
            scenario_index=1,
            scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
            change_count=1,
            normalized_change_magnitude=0.5,
            quality_prediction=11.0,
            quality_delta_from_baseline=1.0,
            quality_scored=True,
            anomaly_scored=False,
            anomaly_score=None,
            anomaly_score_delta_from_baseline=None,
        ),
    ]
    report = _scoring_report(
        status=ScenarioScoringStatus.PARTIAL,
        scores=scores,
        quality_scored_count=2,
        anomaly_scored_count=0,
    )
    assert report.status is ScenarioScoringStatus.PARTIAL


def test_scenario_scoring_report_refused_valid() -> None:
    report = _scoring_report(
        status=ScenarioScoringStatus.REFUSED,
        scores=[],
        baseline_scenario_id=None,
        scored_scenario_count=0,
        quality_scored_count=0,
        anomaly_scored_count=0,
        quality_model_name=None,
        quality_estimator_key=None,
        anomaly_model_name=None,
        anomaly_estimator_key=None,
        prediction_seconds=0.0,
        anomaly_scoring_seconds=0.0,
        total_seconds=0.0,
    )
    assert report.scores == []


def test_scenario_scoring_report_name_key_pair_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        _scoring_report(quality_model_name="Linear Regression", quality_estimator_key=None)


def test_scenario_scoring_report_id_duplicate_rejected() -> None:
    duplicate = CandidateScenarioScore.model_construct(
        scenario_id="SCN-000000",
        scenario_index=1,
        scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
        change_count=1,
        normalized_change_magnitude=0.5,
        quality_prediction=11.0,
        anomaly_score=-0.02,
        quality_delta_from_baseline=1.0,
        anomaly_score_delta_from_baseline=0.03,
        quality_scored=True,
        anomaly_scored=True,
        warnings=[],
    )
    with pytest.raises(ValidationError):
        _scoring_report(scores=[_scenario_score(), duplicate])


def test_scenario_scoring_report_index_duplicate_rejected() -> None:
    duplicate = CandidateScenarioScore.model_construct(
        scenario_id="SCN-000001",
        scenario_index=0,
        scenario_type=CandidateScenarioType.SINGLE_VARIABLE,
        change_count=1,
        normalized_change_magnitude=0.5,
        quality_prediction=11.0,
        anomaly_score=-0.02,
        quality_delta_from_baseline=1.0,
        anomaly_score_delta_from_baseline=0.03,
        quality_scored=True,
        anomaly_scored=True,
        warnings=[],
    )
    with pytest.raises(ValidationError):
        _scoring_report(scores=[_scenario_score(), duplicate])


def test_scenario_scoring_report_index_order_rejected() -> None:
    out_of_order = [
        _scenario_score(scenario_id="SCN-000001", scenario_index=1),
        _scenario_score(),
    ]
    with pytest.raises(ValidationError):
        _scoring_report(scores=out_of_order)


def test_scenario_scoring_report_count_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        _scoring_report(scored_scenario_count=99)


def test_scenario_scoring_report_objective_required_outputs_scored_invalid() -> None:
    scores = [
        _scenario_score(
            quality_scored=False,
            quality_prediction=None,
            quality_delta_from_baseline=None,
            anomaly_scored=True,
        ),
    ]
    with pytest.raises(ValidationError):
        _scoring_report(
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
            scores=scores,
            quality_scored_count=0,
            anomaly_scored_count=1,
            anomaly_model_name=None,
            anomaly_estimator_key=None,
        )


def test_scenario_scoring_report_refused_with_scores_rejected() -> None:
    with pytest.raises(ValidationError):
        _scoring_report(status=ScenarioScoringStatus.REFUSED)


def test_scenario_scoring_report_naive_datetime_rejected() -> None:
    with pytest.raises(ValidationError):
        _scoring_report(evaluated_at=datetime(2026, 7, 21, 16, 0))


def test_scenario_scoring_report_warning_duplicate_rejected() -> None:
    with pytest.raises(ValidationError):
        _scoring_report(warnings=["a", "a"])


def test_scenario_scoring_report_metadata_scalar_validation() -> None:
    with pytest.raises(ValidationError):
        _scoring_report(metadata={"frame": pl.DataFrame({"x": [1]})})


def test_scenario_scoring_report_round_trip() -> None:
    report = _scoring_report()
    assert ScenarioScoringReport.model_validate(report.model_dump()) == report


# --- Outcome (46-47) ---


def test_scenario_scoring_outcome_frozen_and_slots() -> None:
    outcome = ScenarioScoringOutcome(report=_scoring_report())
    with pytest.raises(FrozenInstanceError):
        outcome.report = outcome.report  # type: ignore[misc]
    assert ScenarioScoringOutcome.__slots__ == ("report",)


# --- Scorer ctor (48-58) ---


def test_scorer_quality_only_ctor() -> None:
    scorer = CandidateScenarioScorer(quality_model=_fitted_quality_model())
    meta = scorer.get_metadata()
    assert meta["quality_model_available"] is True
    assert meta["anomaly_model_available"] is False


def test_scorer_anomaly_only_ctor() -> None:
    scorer = CandidateScenarioScorer(anomaly_model=_fitted_anomaly_model())
    meta = scorer.get_metadata()
    assert meta["quality_model_available"] is False
    assert meta["anomaly_model_available"] is True


def test_scorer_both_models_ctor() -> None:
    scorer = CandidateScenarioScorer(
        quality_model=_fitted_quality_model(),
        anomaly_model=_fitted_anomaly_model(),
    )
    meta = scorer.get_metadata()
    assert meta["quality_model_available"] is True
    assert meta["anomaly_model_available"] is True


def test_scorer_neither_model_ctor() -> None:
    scorer = CandidateScenarioScorer()
    meta = scorer.get_metadata()
    assert meta["quality_model_available"] is False
    assert meta["anomaly_model_available"] is False


def test_scorer_bad_policy_type_rejected() -> None:
    with pytest.raises(TypeError):
        CandidateScenarioScorer(policy={"maximum_batch_rows": 10})  # type: ignore[arg-type]


def test_scorer_raw_sklearn_quality_rejected() -> None:
    with pytest.raises(TypeError, match="raw sklearn"):
        CandidateScenarioScorer(quality_model=LinearRegression())


def test_scorer_raw_sklearn_anomaly_rejected() -> None:
    with pytest.raises(TypeError, match="raw sklearn"):
        CandidateScenarioScorer(anomaly_model=IsolationForest())


def test_scorer_external_policy_immutability() -> None:
    policy = ScenarioScoringPolicy(maximum_batch_rows=10)
    scorer = CandidateScenarioScorer(policy=policy)
    policy.maximum_batch_rows = 999
    assert scorer.get_metadata()["maximum_batch_rows"] == 10


def test_scorer_state_isolation_between_instances() -> None:
    policy_a = ScenarioScoringPolicy(maximum_batch_rows=11)
    policy_b = ScenarioScoringPolicy(maximum_batch_rows=22)
    scorer_a = CandidateScenarioScorer(
        quality_model=_fitted_quality_model(),
        policy=policy_a,
    )
    scorer_b = CandidateScenarioScorer(
        quality_model=_fitted_quality_model(),
        policy=policy_b,
    )
    assert scorer_a.get_metadata()["maximum_batch_rows"] == 11
    assert scorer_b.get_metadata()["maximum_batch_rows"] == 22


def test_scorer_metadata_scalar_and_independence() -> None:
    scorer = CandidateScenarioScorer(quality_model=_fitted_quality_model())
    meta = scorer.get_metadata()
    meta["maximum_batch_rows"] = 1
    assert scorer.get_metadata()["maximum_batch_rows"] == 5000


# --- Grid / objective (59-68) ---


def test_score_grid_refused_structured() -> None:
    request = _scoring_request(grid=_refused_grid_report())
    outcome = _score_with_both(request=request)
    report = outcome.report
    assert report.status is ScenarioScoringStatus.REFUSED
    assert report.scores == []
    assert report.baseline_scenario_id is None
    assert _WARNING_GRID_REFUSED in report.warnings


def test_score_empty_scenarios_refused() -> None:
    grid = CandidateGridReport(
        status=CandidateGridStatus.EMPTY,
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        safety_status=RecommendationSafetyStatus.APPROVED,
        resolution_status=ConstraintResolutionStatus.READY,
        candidate_variables=["pressure", "temperature"],
        variable_grids=[_grid(), _temperature_grid()],
        scenarios=[],
        baseline_scenario_id=None,
        potential_scenario_count=0,
        generated_scenario_count=0,
        change_scenario_count=0,
        truncated_scenario_count=0,
        maximum_scenarios=500,
        effective_combination_limit=2,
        generated_at=datetime(2026, 7, 21, 15, 0, tzinfo=UTC),
        warnings=["Model scoring was not performed."],
        metadata={"model_scoring_performed": False},
    )
    request = _scoring_request(grid=grid)
    outcome = _score_with_both(request=request)
    assert outcome.report.status is ScenarioScoringStatus.REFUSED
    assert _WARNING_NO_SCENARIOS in outcome.report.warnings


def test_score_baseline_missing_raises() -> None:
    request = _scoring_request()
    request.grid.baseline_scenario_id = "SCN-000999"
    with pytest.raises(DataValidationError, match="baseline_scenario_id"):
        _score_with_both(request=request)


def test_score_batch_limit_refusal() -> None:
    grid = _ready_grid_report(
        scenarios=[
            _baseline_scenario(),
            _single_scenario(),
            _second_single_scenario(),
        ],
        generated_scenario_count=3,
        change_scenario_count=2,
        potential_scenario_count=3,
    )
    request = _scoring_request(grid=grid)
    quality = _QualityModelSpy(_fitted_quality_model())
    anomaly = _AnomalyModelSpy(_fitted_anomaly_model())
    outcome = _score_with_both(
        request=request,
        policy=ScenarioScoringPolicy(maximum_batch_rows=2),
        quality=quality,
        anomaly=anomaly,
    )
    assert outcome.report.status is ScenarioScoringStatus.REFUSED
    assert _WARNING_BATCH_LIMIT in outcome.report.warnings
    assert quality.predict_call_count == 0
    assert anomaly.score_call_count == 0


def test_score_quality_required_model_missing_refused() -> None:
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        ),
    )
    outcome = CandidateScenarioScorer(anomaly_model=_fitted_anomaly_model()).score(
        request
    )
    assert outcome.report.status is ScenarioScoringStatus.REFUSED
    assert _WARNING_QUALITY_MODEL_MISSING in outcome.report.warnings


def test_score_anomaly_required_model_missing_refused() -> None:
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        ),
    )
    outcome = CandidateScenarioScorer(quality_model=_fitted_quality_model()).score(
        request
    )
    assert outcome.report.status is ScenarioScoringStatus.REFUSED
    assert _WARNING_ANOMALY_MODEL_MISSING in outcome.report.warnings


def test_score_balance_requires_both_models_for_scored() -> None:
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
        ),
    )
    outcome = CandidateScenarioScorer(quality_model=_fitted_quality_model()).score(
        request
    )
    assert outcome.report.status is ScenarioScoringStatus.REFUSED


def test_score_optional_model_still_outputs_when_provided() -> None:
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        ),
    )
    outcome = _score_with_both(request=request)
    report = outcome.report
    assert report.status is ScenarioScoringStatus.SCORED
    assert all(item.quality_scored for item in report.scores)
    assert all(item.anomaly_scored for item in report.scores)


def test_score_regression_validation_passes() -> None:
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        ),
    )
    outcome = CandidateScenarioScorer(quality_model=_fitted_quality_model()).score(
        request
    )
    assert outcome.report.status is ScenarioScoringStatus.SCORED
    assert outcome.report.task is AnalysisTask.REGRESSION


def test_score_classification_quality_model_rejected() -> None:
    frame = _training_frame()
    labels = pl.Series("label", [0, 1] * (frame.height // 2))
    classifier = create_logistic_regression(random_state=3)
    classifier.fit(frame.select(FEATURE_COLUMNS), labels)
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        ),
    )
    with pytest.raises(DataValidationError, match="REGRESSION"):
        CandidateScenarioScorer(quality_model=classifier).score(request)


# --- Feature construction (69-78) ---


def test_feature_matrix_baseline_full_row() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    request = _scoring_request()
    _score_with_both(request=request, quality=quality)
    assert quality.last_predict_frame is not None
    baseline_row = quality.last_predict_frame.row(0, named=True)
    assert baseline_row == BASELINE_FEATURES


def test_feature_matrix_overlay_applies_candidate_changes() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    request = _scoring_request()
    _score_with_both(request=request, quality=quality)
    assert quality.last_predict_frame is not None
    changed_row = quality.last_predict_frame.row(1, named=True)
    assert changed_row["pressure"] == pytest.approx(0.0)
    assert changed_row["temperature"] == pytest.approx(80.0)


def test_feature_matrix_non_candidate_preserved() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    request = _scoring_request()
    _score_with_both(request=request, quality=quality)
    assert quality.last_predict_frame is not None
    humidity = quality.last_predict_frame["humidity"].to_list()
    assert humidity == [BASELINE_FEATURES["humidity"]] * request.grid.generated_scenario_count


def test_feature_matrix_column_order_matches_request() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    request = _scoring_request()
    _score_with_both(request=request, quality=quality)
    assert quality.last_predict_frame is not None
    assert quality.last_predict_frame.columns == FEATURE_COLUMNS


def test_feature_matrix_no_autofill_missing_feature() -> None:
    grid = _ready_grid_report(
        candidate_variables=["pressure"],
        variable_grids=[_grid()],
        effective_combination_limit=1,
        scenarios=[
            _baseline_scenario(variable_values={"pressure": 50.0}),
            _single_scenario(variable_values={"pressure": 0.0}),
        ],
    )
    request = _scoring_request(
        grid=grid,
        feature_columns=["pressure", "humidity"],
        baseline_features={"pressure": 50.0, "humidity": 45.0},
    )
    full = _training_frame()
    frame = full.select(["pressure", "humidity"])
    target = full["pressure"] * 0.1 + full["humidity"] * 0.02
    model = create_linear_regression(random_state=7)
    model.fit(frame, pl.Series("quality", target.to_list()))
    quality = _QualityModelSpy(model)
    scorer = CandidateScenarioScorer(quality_model=quality)
    outcome = scorer.score(request)
    assert outcome.report.status is ScenarioScoringStatus.SCORED
    assert quality.last_predict_frame is not None
    assert quality.last_predict_frame["humidity"].to_list()[0] == pytest.approx(45.0)


def test_feature_matrix_excludes_target_column() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    request = _scoring_request(target_column="quality")
    _score_with_both(request=request, quality=quality)
    assert quality.last_predict_frame is not None
    assert "quality" not in quality.last_predict_frame.columns


def test_feature_matrix_no_scaling_or_encoding_applied() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    request = _scoring_request()
    _score_with_both(request=request, quality=quality)
    assert quality.last_predict_frame is not None
    first = quality.last_predict_frame.row(0, named=True)
    assert first["pressure"] == pytest.approx(50.0)
    assert first["temperature"] == pytest.approx(80.0)


def test_feature_schema_mismatch_raises_before_predict() -> None:
    model = _fitted_quality_model()
    quality = _QualityModelSpy(model)
    reordered = _scoring_request(
        feature_columns=["temperature", "pressure", "humidity"],
        baseline_features={
            "temperature": 80.0,
            "pressure": 50.0,
            "humidity": 45.0,
        },
    )
    with pytest.raises(DataValidationError, match="exactly match"):
        CandidateScenarioScorer(quality_model=quality).score(reordered)
    assert quality.predict_call_count == 0


def test_missing_feature_metadata_warning_and_refusal() -> None:
    inner = _fitted_quality_model()
    inner._feature_names = ()  # type: ignore[attr-defined]
    metadata = inner.get_metadata()
    object.__setattr__(metadata, "features", [])
    quality = _QualityModelSpy(inner)
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        ),
    )
    with pytest.raises(DataValidationError):
        CandidateScenarioScorer(quality_model=quality).score(request)
    outcome = CandidateScenarioScorer(
        quality_model=quality,
        policy=ScenarioScoringPolicy(require_exact_feature_set=False),
    ).score(request)
    assert _WARNING_FEATURE_METADATA in outcome.report.warnings


# --- Fitted (79-87) ---


def test_fitted_models_allowed() -> None:
    outcome = _score_with_both()
    assert outcome.report.status is ScenarioScoringStatus.SCORED


def test_unfitted_quality_model_rejected() -> None:
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        ),
    )
    with pytest.raises(DataValidationError, match="fitted"):
        CandidateScenarioScorer(quality_model=create_linear_regression()).score(request)


def test_unfitted_anomaly_model_rejected() -> None:
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        ),
    )
    with pytest.raises(DataValidationError, match="fitted"):
        CandidateScenarioScorer(
            anomaly_model=create_isolation_forest_anomaly_model(),
        ).score(request)


def test_quality_state_change_detected() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    quality.mutate_fitted_on_predict = True
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        ),
    )
    with pytest.raises(ProcessIntelligenceError, match="state changed"):
        CandidateScenarioScorer(quality_model=quality).score(request)


def test_anomaly_state_change_detected() -> None:
    anomaly = _AnomalyModelSpy(_fitted_anomaly_model())
    anomaly.mutate_fitted_on_score = True
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        ),
    )
    with pytest.raises(ProcessIntelligenceError, match="state changed"):
        CandidateScenarioScorer(anomaly_model=anomaly).score(request)


def test_scorer_does_not_fit_or_refit_models() -> None:
    quality = create_linear_regression(random_state=1)
    anomaly = create_isolation_forest_anomaly_model(
        config=IsolationForestConfig(random_state=1),
    )
    frame = _training_frame()
    quality.fit(frame.select(FEATURE_COLUMNS), _quality_target(frame))
    anomaly.fit(frame.select(FEATURE_COLUMNS))
    quality_before = copy.deepcopy(quality.get_metadata().model_dump())
    anomaly_before = copy.deepcopy(anomaly.get_metadata().model_dump())
    _score_with_both(quality=quality, anomaly=anomaly)
    assert quality.get_metadata().model_dump() == quality_before
    assert anomaly.get_metadata().model_dump() == anomaly_before


def test_scorer_does_not_recalibrate_threshold() -> None:
    anomaly = _fitted_anomaly_model()
    threshold_before = anomaly.get_metadata().threshold
    _score_with_both(anomaly=anomaly)
    assert anomaly.get_metadata().threshold == threshold_before


# --- Batch (88-92) ---


def test_batch_predict_called_exactly_once() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    _score_with_both(quality=quality)
    assert quality.predict_call_count == 1


def test_batch_score_samples_called_exactly_once() -> None:
    anomaly = _AnomalyModelSpy(_fitted_anomaly_model())
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        ),
    )
    CandidateScenarioScorer(anomaly_model=anomaly).score(request)
    assert anomaly.score_call_count == 1


def test_no_per_scenario_model_calls() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    anomaly = _AnomalyModelSpy(_fitted_anomaly_model())
    grid = _ready_grid_report(
        scenarios=[
            _baseline_scenario(),
            _single_scenario(),
            _second_single_scenario(),
        ],
        generated_scenario_count=3,
        change_scenario_count=2,
        potential_scenario_count=3,
    )
    _score_with_both(request=_scoring_request(grid=grid), quality=quality, anomaly=anomaly)
    assert quality.predict_call_count == 1
    assert anomaly.score_call_count == 1


def test_validation_before_model_call_on_unfitted() -> None:
    quality = _QualityModelSpy(create_linear_regression())
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        ),
    )
    with pytest.raises(DataValidationError):
        CandidateScenarioScorer(quality_model=quality).score(request)
    assert quality.predict_call_count == 0


def test_no_unnecessary_call_on_early_grid_refusal() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    anomaly = _AnomalyModelSpy(_fitted_anomaly_model())
    _score_with_both(
        request=_scoring_request(grid=_refused_grid_report()),
        quality=quality,
        anomaly=anomaly,
    )
    assert quality.predict_call_count == 0
    assert anomaly.score_call_count == 0


# --- Quality output (93-99) ---


def test_quality_prediction_length_matches_scenarios() -> None:
    outcome = _score_with_both()
    assert len(outcome.report.scores) == outcome.report.requested_scenario_count
    assert all(item.quality_prediction is not None for item in outcome.report.scores)


def test_quality_prediction_length_mismatch_refused() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    quality.custom_predict_return = [1.0]
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        ),
    )
    outcome = CandidateScenarioScorer(quality_model=quality).score(request)
    assert outcome.report.status is ScenarioScoringStatus.REFUSED


def test_quality_prediction_finite_required() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    quality.custom_predict_return = [1.0, float("nan")]
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        ),
    )
    outcome = CandidateScenarioScorer(quality_model=quality).score(request)
    assert outcome.report.status is ScenarioScoringStatus.REFUSED


def test_quality_prediction_numpy_scalar_coerced() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    quality.custom_predict_return = np.array([np.float64(1.5), np.float64(2.5)])
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        ),
    )
    outcome = CandidateScenarioScorer(quality_model=quality).score(request)
    assert outcome.report.status is ScenarioScoringStatus.SCORED
    assert outcome.report.scores[0].quality_prediction == pytest.approx(1.5)


def test_quality_prediction_n_by_one_shape_accepted() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    quality.custom_predict_return = np.array([[3.0], [4.0]])
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        ),
    )
    outcome = CandidateScenarioScorer(quality_model=quality).score(request)
    assert outcome.report.status is ScenarioScoringStatus.SCORED


def test_quality_multi_output_rejected() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    quality.custom_predict_return = np.array([[1.0, 2.0], [3.0, 4.0]])
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        ),
    )
    outcome = CandidateScenarioScorer(quality_model=quality).score(request)
    assert outcome.report.status is ScenarioScoringStatus.REFUSED


def test_quality_string_prediction_rejected() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    quality.custom_predict_return = np.array(["1.0", "2.0"], dtype=object)
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        ),
    )
    outcome = CandidateScenarioScorer(quality_model=quality).score(request)
    assert outcome.report.status is ScenarioScoringStatus.REFUSED


# --- Anomaly (100-106) ---


def test_anomaly_score_length_matches_scenarios() -> None:
    request = _scoring_request(
        grid=_ready_grid_report(objective=RecommendationObjective.REDUCE_ANOMALY_SCORE),
    )
    outcome = CandidateScenarioScorer(anomaly_model=_fitted_anomaly_model()).score(
        request
    )
    assert all(item.anomaly_score is not None for item in outcome.report.scores)


def test_anomaly_score_length_mismatch_refused() -> None:
    anomaly = _AnomalyModelSpy(_fitted_anomaly_model())
    anomaly.custom_score_return = [-0.1]
    request = _scoring_request(
        grid=_ready_grid_report(objective=RecommendationObjective.REDUCE_ANOMALY_SCORE),
    )
    outcome = CandidateScenarioScorer(anomaly_model=anomaly).score(request)
    assert outcome.report.status is ScenarioScoringStatus.REFUSED


def test_anomaly_score_finite_required() -> None:
    anomaly = _AnomalyModelSpy(_fitted_anomaly_model())
    anomaly.custom_score_return = [-0.1, float("inf")]
    request = _scoring_request(
        grid=_ready_grid_report(objective=RecommendationObjective.REDUCE_ANOMALY_SCORE),
    )
    outcome = CandidateScenarioScorer(anomaly_model=anomaly).score(request)
    assert outcome.report.status is ScenarioScoringStatus.REFUSED


def test_anomaly_adapter_direction_higher_is_more_anomalous() -> None:
    grid = _ready_grid_report(
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        scenarios=[
            _baseline_scenario(),
            _single_scenario(
                variable_values={"pressure": 100.0, "temperature": 80.0},
                deltas={"pressure": 50.0},
                relative_deltas={"pressure": 1.0},
                normalized_change_magnitude=0.5,
            ),
        ],
    )
    request = _scoring_request(grid=grid)
    outcome = CandidateScenarioScorer(anomaly_model=_fitted_anomaly_model()).score(
        request
    )
    baseline = outcome.report.scores[0].anomaly_score
    extreme = outcome.report.scores[1].anomaly_score
    assert baseline is not None and extreme is not None
    assert extreme > baseline


def test_anomaly_negative_scores_allowed() -> None:
    anomaly = _AnomalyModelSpy(_fitted_anomaly_model())
    anomaly.custom_score_return = [-0.5, -0.2]
    request = _scoring_request(
        grid=_ready_grid_report(objective=RecommendationObjective.REDUCE_ANOMALY_SCORE),
    )
    outcome = CandidateScenarioScorer(anomaly_model=anomaly).score(request)
    assert outcome.report.status is ScenarioScoringStatus.SCORED
    assert outcome.report.scores[0].anomaly_score == pytest.approx(-0.5)


def test_anomaly_uses_score_samples_not_raw_estimator() -> None:
    anomaly = _AnomalyModelSpy(_fitted_anomaly_model())
    request = _scoring_request(
        grid=_ready_grid_report(objective=RecommendationObjective.REDUCE_ANOMALY_SCORE),
    )
    CandidateScenarioScorer(anomaly_model=anomaly).score(request)
    assert anomaly.score_call_count == 1


def test_anomaly_no_residual_scoring_metadata() -> None:
    request = _scoring_request(
        grid=_ready_grid_report(objective=RecommendationObjective.REDUCE_ANOMALY_SCORE),
    )
    outcome = CandidateScenarioScorer(anomaly_model=_fitted_anomaly_model()).score(
        request
    )
    assert outcome.report.metadata["residual_scoring_performed"] is False


# --- Baseline delta (107-112) ---


def test_baseline_scenario_is_first_score() -> None:
    outcome = _score_with_both()
    assert outcome.report.scores[0].scenario_id == "SCN-000000"
    assert outcome.report.scores[0].scenario_type is CandidateScenarioType.BASELINE


def test_baseline_deltas_are_zero() -> None:
    outcome = _score_with_both()
    baseline = outcome.report.scores[0]
    assert baseline.quality_delta_from_baseline == pytest.approx(0.0)
    assert baseline.anomaly_score_delta_from_baseline == pytest.approx(0.0)


def test_change_scenario_deltas_relative_to_baseline() -> None:
    outcome = _score_with_both()
    baseline_q = outcome.report.scores[0].quality_prediction
    changed = outcome.report.scores[1]
    assert changed.quality_prediction is not None
    assert baseline_q is not None
    expected = changed.quality_prediction - baseline_q
    assert changed.quality_delta_from_baseline == pytest.approx(expected)


def test_unscored_outputs_have_none_deltas() -> None:
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        ),
    )
    outcome = CandidateScenarioScorer(quality_model=_fitted_quality_model()).score(
        request
    )
    for item in outcome.report.scores:
        assert item.anomaly_score is None
        assert item.anomaly_score_delta_from_baseline is None


# --- Ordering (113-117) ---


def test_grid_scenario_order_preserved() -> None:
    grid = _ready_grid_report(
        scenarios=[
            _baseline_scenario(),
            _second_single_scenario(scenario_id="SCN-000001", scenario_index=1),
            _single_scenario(
                scenario_id="SCN-000002",
                scenario_index=2,
                variable_values={"pressure": 100.0, "temperature": 80.0},
                changed_variables=["pressure"],
                deltas={"pressure": 50.0},
                relative_deltas={"pressure": 1.0},
            ),
        ],
        generated_scenario_count=3,
        change_scenario_count=2,
        potential_scenario_count=3,
    )
    outcome = _score_with_both(request=_scoring_request(grid=grid))
    ids = [item.scenario_id for item in outcome.report.scores]
    assert ids == ["SCN-000000", "SCN-000001", "SCN-000002"]


def test_scores_not_sorted_by_prediction() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    quality.custom_predict_return = [5.0, 1.0]
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        ),
    )
    outcome = CandidateScenarioScorer(quality_model=quality).score(request)
    preds = [item.quality_prediction for item in outcome.report.scores]
    assert preds == [pytest.approx(5.0), pytest.approx(1.0)]
    assert outcome.report.scores[0].scenario_index == 0


def test_contiguous_scenario_ids_in_report() -> None:
    outcome = _score_with_both()
    for expected, score in enumerate(outcome.report.scores):
        assert score.scenario_index == expected
        assert score.scenario_id == f"SCN-{expected:06d}"


def test_metadata_scores_preserve_grid_order() -> None:
    outcome = _score_with_both()
    assert outcome.report.metadata["scores_preserve_grid_order"] is True


# --- Partial / failure (118-126) ---


def test_partial_false_both_required_one_fails_refused() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    quality.predict_error = ValueError("quality boom")
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
        ),
    )
    outcome = _score_with_both(request=request, quality=quality)
    assert outcome.report.status is ScenarioScoringStatus.REFUSED
    assert any("quality scoring failed" in w for w in outcome.report.warnings)


def test_partial_true_balance_quality_only() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    anomaly = _AnomalyModelSpy(_fitted_anomaly_model())
    anomaly.score_error = ValueError("anomaly boom")
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
        ),
    )
    policy = ScenarioScoringPolicy(allow_partial_scoring=True)
    outcome = _score_with_both(
        request=request,
        policy=policy,
        quality=quality,
        anomaly=anomaly,
    )
    assert outcome.report.status is ScenarioScoringStatus.PARTIAL
    assert all(item.quality_scored for item in outcome.report.scores)
    assert not any(item.anomaly_scored for item in outcome.report.scores)
    assert _WARNING_PARTIAL in outcome.report.warnings


def test_partial_true_balance_anomaly_only() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    quality.predict_error = ValueError("quality boom")
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
        ),
    )
    policy = ScenarioScoringPolicy(allow_partial_scoring=True)
    outcome = _score_with_both(request=request, policy=policy, quality=quality)
    assert outcome.report.status is ScenarioScoringStatus.PARTIAL
    assert all(item.anomaly_scored for item in outcome.report.scores)
    assert not any(item.quality_scored for item in outcome.report.scores)


def test_partial_false_both_fail_refused() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    anomaly = _AnomalyModelSpy(_fitted_anomaly_model())
    quality.predict_error = ValueError("quality boom")
    anomaly.score_error = ValueError("anomaly boom")
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
        ),
    )
    policy = ScenarioScoringPolicy(allow_partial_scoring=True)
    outcome = _score_with_both(
        request=request,
        policy=policy,
        quality=quality,
        anomaly=anomaly,
    )
    assert outcome.report.status is ScenarioScoringStatus.REFUSED


def test_failure_warning_includes_role_and_class() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    quality.predict_error = ValueError("predict broke")
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        ),
    )
    outcome = CandidateScenarioScorer(quality_model=quality).score(request)
    warning = next(w for w in outcome.report.warnings if "quality scoring failed" in w)
    assert "ValueError" in warning
    assert "predict broke" in warning


def test_failure_warning_no_traceback() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    quality.predict_error = ValueError("short")
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        ),
    )
    outcome = CandidateScenarioScorer(quality_model=quality).score(request)
    joined = "\n".join(outcome.report.warnings)
    assert "Traceback" not in joined


def test_system_exceptions_not_isolated() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    quality.predict_error = KeyboardInterrupt()
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        ),
    )
    with pytest.raises(KeyboardInterrupt):
        CandidateScenarioScorer(quality_model=quality).score(request)


# --- Status / metadata (127-139) ---


def test_scored_status_for_complete_quality_objective() -> None:
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        ),
    )
    outcome = CandidateScenarioScorer(quality_model=_fitted_quality_model()).score(
        request
    )
    assert outcome.report.status is ScenarioScoringStatus.SCORED


def test_partial_status_metadata_counts() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    anomaly = _AnomalyModelSpy(_fitted_anomaly_model())
    anomaly.score_error = ValueError("fail")
    policy = ScenarioScoringPolicy(allow_partial_scoring=True)
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
        ),
    )
    outcome = _score_with_both(
        request=request,
        policy=policy,
        quality=quality,
        anomaly=anomaly,
    )
    report = outcome.report
    assert report.status is ScenarioScoringStatus.PARTIAL
    assert report.quality_scored_count == len(report.scores)
    assert report.anomaly_scored_count == 0


def test_refused_status_zero_counts() -> None:
    outcome = _score_with_both(request=_scoring_request(grid=_refused_grid_report()))
    report = outcome.report
    assert report.status is ScenarioScoringStatus.REFUSED
    assert report.scored_scenario_count == 0
    assert report.quality_scored_count == 0
    assert report.anomaly_scored_count == 0


def test_metadata_batch_call_counts() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    anomaly = _AnomalyModelSpy(_fitted_anomaly_model())
    outcome = _score_with_both(quality=quality, anomaly=anomaly)
    meta = outcome.report.metadata
    assert meta["batch_prediction_call_count"] == 1
    assert meta["batch_anomaly_call_count"] == 1


def test_metadata_capability_flags_false() -> None:
    outcome = _score_with_both()
    meta = outcome.report.metadata
    assert meta["residual_scoring_performed"] is False
    assert meta["model_refit_performed"] is False
    assert meta["scenario_ranking_performed"] is False
    assert meta["recommendation_generated"] is False
    assert meta["actual_target_available"] is False


def test_metadata_no_model_dataframe_or_ndarray() -> None:
    outcome = _score_with_both()
    for value in outcome.report.metadata.values():
        assert not isinstance(value, (pl.DataFrame, np.ndarray))


def test_scorer_get_metadata_matches_report_flags() -> None:
    scorer = CandidateScenarioScorer(
        quality_model=_fitted_quality_model(),
        anomaly_model=_fitted_anomaly_model(),
    )
    scorer_meta = scorer.get_metadata()
    outcome = scorer.score(_scoring_request())
    report_meta = outcome.report.metadata
    assert scorer_meta["scores_residual_anomaly"] is False
    assert scorer_meta["performs_model_refit"] is False
    assert report_meta["model_refit_performed"] is False


# --- Timing (140-142) ---


def test_timing_fields_non_negative() -> None:
    outcome = _score_with_both()
    report = outcome.report
    assert report.prediction_seconds >= 0.0
    assert report.anomaly_scoring_seconds >= 0.0
    assert report.total_seconds >= 0.0
    assert report.total_seconds + 1e-12 >= (
        report.prediction_seconds + report.anomaly_scoring_seconds
    )


def test_evaluated_at_timezone_aware_utc() -> None:
    outcome = _score_with_both()
    evaluated = outcome.report.evaluated_at
    assert evaluated.tzinfo is not None
    assert evaluated.utcoffset() is not None


def test_timing_not_exact_equality_required() -> None:
    first = _score_with_both()
    second = _score_with_both()
    assert first.report.total_seconds >= 0.0
    assert second.report.total_seconds >= 0.0


# --- Language (143-148) ---


def test_language_residual_caveat_present() -> None:
    outcome = _score_with_both()
    assert any(_WARNING_RESIDUAL in w for w in outcome.report.warnings)


def test_language_unranked_caveat_present() -> None:
    outcome = _score_with_both()
    assert any(_WARNING_UNRANKED in w for w in outcome.report.warnings)


def test_language_no_guaranteed_improvement_claim() -> None:
    outcome = _score_with_both()
    joined = " ".join(outcome.report.warnings).lower()
    assert "guarantee improvement" in joined


def test_language_no_proven_optimal_claim() -> None:
    outcome = _score_with_both()
    joined = " ".join(outcome.report.warnings).lower()
    assert "optimal" not in joined


def test_language_no_causal_proof_claim() -> None:
    outcome = _score_with_both()
    assert any(_WARNING_NO_CAUSATION in w for w in outcome.report.warnings)


def test_language_no_korean_certainty_terms() -> None:
    outcome = _score_with_both()
    joined = "\n".join(outcome.report.warnings)
    assert "반드시" not in joined
    assert "확정" not in joined
    assert _WARNING_VERIFICATION in joined


# --- Immutability (149-160) ---


def test_request_grid_deep_copy_immutable() -> None:
    request = _scoring_request()
    original_len = len(request.grid.scenarios)
    request.grid.scenarios.clear()
    request2 = _scoring_request()
    assert len(request2.grid.scenarios) == original_len


def test_baseline_features_not_mutated_by_scoring() -> None:
    baseline = dict(BASELINE_FEATURES)
    request = _scoring_request(baseline_features=baseline)
    _score_with_both(request=request)
    assert baseline == BASELINE_FEATURES


def test_model_metadata_stable_after_scoring() -> None:
    quality = _fitted_quality_model()
    before = quality.get_metadata().model_dump()
    _score_with_both(quality=quality)
    after = quality.get_metadata().model_dump()
    assert before == after


def test_scorer_does_not_cache_prior_results() -> None:
    scorer = CandidateScenarioScorer(
        quality_model=_fitted_quality_model(),
        anomaly_model=_fitted_anomaly_model(),
    )
    request = _scoring_request()
    first = scorer.score(request)
    second = scorer.score(request)
    assert first.report.scores[0].quality_prediction == pytest.approx(
        second.report.scores[0].quality_prediction
    )
    assert first.report is not second.report


def test_repeated_calls_no_call_count_accumulation() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    scorer = CandidateScenarioScorer(
        quality_model=quality,
        anomaly_model=_fitted_anomaly_model(),
    )
    request = _scoring_request(
        grid=_ready_grid_report(
            objective=RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
        ),
    )
    scorer.score(request)
    scorer.score(request)
    assert quality.predict_call_count == 2


def test_deterministic_predictions_and_scores() -> None:
    request = _scoring_request()
    first = _score_with_both(request=request)
    second = _score_with_both(request=request)
    q1 = [item.quality_prediction for item in first.report.scores]
    q2 = [item.quality_prediction for item in second.report.scores]
    a1 = [item.anomaly_score for item in first.report.scores]
    a2 = [item.anomaly_score for item in second.report.scores]
    assert q1 == q2
    assert a1 == a2


def test_scorer_instance_isolation() -> None:
    scorer_a = CandidateScenarioScorer(quality_model=_fitted_quality_model())
    scorer_b = CandidateScenarioScorer(quality_model=_fitted_quality_model())
    out_a = scorer_a.score(_scoring_request())
    out_b = scorer_b.score(_scoring_request())
    assert out_a.report.scores[0].quality_prediction == pytest.approx(
        out_b.report.scores[0].quality_prediction
    )
    assert out_a is not out_b


def test_mutating_returned_report_does_not_affect_next_call() -> None:
    scorer = CandidateScenarioScorer(
        quality_model=_fitted_quality_model(),
        anomaly_model=_fitted_anomaly_model(),
    )
    request = _scoring_request()
    first = scorer.score(request)
    first.report.scores[0].quality_prediction = 99999.0
    second = scorer.score(request)
    assert second.report.scores[0].quality_prediction != pytest.approx(99999.0)


def test_score_rejects_non_request_type() -> None:
    scorer = CandidateScenarioScorer(quality_model=_fitted_quality_model())
    with pytest.raises(TypeError):
        scorer.score(_scoring_request().model_dump())  # type: ignore[arg-type]


def test_scored_anomaly_only_objective_end_to_end() -> None:
    request = _scoring_request(
        grid=_ready_grid_report(objective=RecommendationObjective.REDUCE_ANOMALY_SCORE),
    )
    outcome = CandidateScenarioScorer(anomaly_model=_fitted_anomaly_model()).score(
        request
    )
    report = outcome.report
    assert report.status is ScenarioScoringStatus.SCORED
    assert report.task is AnalysisTask.UNSUPERVISED_ANOMALY
    assert report.quality_model_name is None
    assert all(item.anomaly_scored for item in report.scores)
    assert math.isclose(
        report.scores[0].anomaly_score_delta_from_baseline or 0.0,
        0.0,
        abs_tol=1e-12,
    )
