"""Recommendation MVP integration smoke tests (Step 10A).

Runs Steps 9A–9G end-to-end with real fitted regression and anomaly adapters.
Does not modify production source, use raw sklearn estimators in the pipeline,
or monkeypatch production logic.
"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from typing import Any

import numpy as np
import polars as pl
import pytest

from process_intelligence.core.enums import AnalysisTask, ColumnRole
from process_intelligence.core.protocols import BaseAnalysisModel, BaseAnomalyModel
from process_intelligence.core.schemas import (
    ModelMetadata,
    ModelSpec,
    RootCauseFactor,
    VariableConstraint,
)
from process_intelligence.diagnosis.enums import DiagnosisMethod, DiagnosisScope
from process_intelligence.diagnosis.schemas import DiagnosisResult
from process_intelligence.evaluation.leakage import (
    LeakageIssue,
    LeakageIssueType,
    LeakageReport,
    LeakageSeverity,
)
from process_intelligence.models import (
    create_isolation_forest_anomaly_model,
    create_linear_regression,
)
from process_intelligence.models.anomaly import IsolationForestConfig
from process_intelligence.recommendation import (
    CandidateScenarioScorer,
    CandidateScenarioType,
    QualityOptimizationDirection,
    RecommendationObjective,
    RecommendationPipeline,
    RecommendationPipelineRequest,
    RecommendationPipelineStage,
    RecommendationReasonCode,
    RecommendationRequest,
    RecommendationSafetyContext,
    RecommendationSafetyStatus,
    RecommendationStatus,
    ScenarioRankingStatus,
)

FEATURE_COLUMNS = ["temperature", "pressure", "flow_rate"]
TARGET_COLUMN = "quality_score"
_CANONICAL_STAGES = list(RecommendationPipelineStage)

_FORBIDDEN_OUTCOME_PATTERNS = (
    re.compile(r"guaranteed\s+improvement", re.IGNORECASE),
    re.compile(r"proven\s+root\s+cause", re.IGNORECASE),
    re.compile(r"this\s+will\s+fix", re.IGNORECASE),
    re.compile(r"(?<!not\s)proven\s+optimal", re.IGNORECASE),
)


# ---------------------------------------------------------------------------
# Synthetic data and fitted adapters (function-scoped via fixtures)
# ---------------------------------------------------------------------------


def _build_synthetic_frame(*, rows: int = 48) -> tuple[pl.DataFrame, pl.Series]:
    """Build deterministic process-like features and quality_score target."""
    rng = np.random.default_rng(42)
    temperature = rng.uniform(60.0, 100.0, rows)
    pressure = rng.uniform(40.0, 80.0, rows)
    flow_rate = rng.uniform(10.0, 30.0, rows)
    # Representative off-center operating point (row 0) for recommendation current.
    temperature[0] = 62.0
    pressure[0] = 42.0
    flow_rate[0] = 12.0
    noise = np.asarray([((i % 7) - 3) * 0.05 for i in range(rows)], dtype=np.float64)
    quality = 0.4 * temperature + 0.25 * pressure + 0.15 * flow_rate + noise
    frame = pl.DataFrame(
        {
            "temperature": temperature.tolist(),
            "pressure": pressure.tolist(),
            "flow_rate": flow_rate.tolist(),
        }
    )
    target = pl.Series(TARGET_COLUMN, quality.tolist())
    return frame, target


def _baseline_from_frame(frame: pl.DataFrame, *, row_index: int = 0) -> dict[str, float]:
    return {name: float(frame[name][row_index]) for name in FEATURE_COLUMNS}


def _fit_quality_model(frame: pl.DataFrame, target: pl.Series) -> BaseAnalysisModel:
    model = create_linear_regression(random_state=7)
    model.fit(frame.select(FEATURE_COLUMNS), target)
    assert model.is_fitted
    assert list(model.feature_names) == FEATURE_COLUMNS
    return model


def _fit_anomaly_model(frame: pl.DataFrame) -> BaseAnomalyModel:
    model = create_isolation_forest_anomaly_model(
        config=IsolationForestConfig(random_state=7, contamination=0.05),
    )
    model.fit(frame.select(FEATURE_COLUMNS))
    assert model.is_fitted
    assert list(model.feature_names) == FEATURE_COLUMNS
    return model


@pytest.fixture
def synthetic_frame() -> tuple[pl.DataFrame, pl.Series]:
    return _build_synthetic_frame()


@pytest.fixture
def baseline_features(synthetic_frame: tuple[pl.DataFrame, pl.Series]) -> dict[str, float]:
    frame, _ = synthetic_frame
    return _baseline_from_frame(frame, row_index=0)


@pytest.fixture
def fitted_quality_model(
    synthetic_frame: tuple[pl.DataFrame, pl.Series],
) -> BaseAnalysisModel:
    frame, target = synthetic_frame
    return _fit_quality_model(frame, target)


@pytest.fixture
def fitted_anomaly_model(
    synthetic_frame: tuple[pl.DataFrame, pl.Series],
) -> BaseAnomalyModel:
    frame, _ = synthetic_frame
    return _fit_anomaly_model(frame)


# ---------------------------------------------------------------------------
# Request / context builders
# ---------------------------------------------------------------------------


def _factor(
    variable: str,
    *,
    confidence: float,
    needs_verification: bool = True,
) -> RootCauseFactor:
    return RootCauseFactor(
        variable=variable,
        direction="increase",
        deviation=1.0,
        role=ColumnRole.CONTROLLABLE_PROCESS,
        controllable=True,
        evidence="Associated process driver; association only, not causation.",
        confidence=confidence,
        needs_verification=needs_verification,
    )


def _diagnosis(*, task: AnalysisTask) -> DiagnosisResult:
    return DiagnosisResult(
        anomaly_id="smoke-event-1",
        task=task,
        method_used=[DiagnosisMethod.GROUP_COMPARISON],
        scope=DiagnosisScope.SINGLE_EVENT,
        factors=[
            _factor("temperature", confidence=0.85),
            _factor("pressure", confidence=0.75),
        ],
        confidence=0.80,
        analyzed_row_count=2,
        reference_row_count=40,
        caveats=["Diagnosis reflects association only; causation is not established."],
        generated_at=datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
        metadata={"fixture": "recommendation_smoke"},
    )


def _constraints() -> list[VariableConstraint]:
    return [
        VariableConstraint(
            variable="temperature",
            adjustable=True,
            minimum=60.0,
            maximum=100.0,
            fixed=False,
        ),
        VariableConstraint(
            variable="pressure",
            adjustable=True,
            minimum=40.0,
            maximum=80.0,
            fixed=False,
        ),
    ]


def _safe_leakage() -> LeakageReport:
    return LeakageReport(
        is_safe=True,
        issues=[],
        blocker_count=0,
        warning_count=0,
        checked_feature_columns=list(FEATURE_COLUMNS),
        checked_preprocessing_event_count=0,
    )


def _blocker_leakage() -> LeakageReport:
    return LeakageReport(
        is_safe=False,
        issues=[
            LeakageIssue(
                issue_type=LeakageIssueType.TARGET_INCLUDED_AS_FEATURE,
                severity=LeakageSeverity.BLOCKER,
                columns=[TARGET_COLUMN],
                partitions=[],
                message="Target included as feature",
                suggested_action="Remove target from features",
            )
        ],
        blocker_count=1,
        warning_count=0,
        checked_feature_columns=list(FEATURE_COLUMNS),
        checked_preprocessing_event_count=0,
    )


def _recommendation_request(
    *,
    objective: RecommendationObjective,
    baseline_features: dict[str, float],
    task: AnalysisTask | None = None,
    user_confirmed: list[str] | None = None,
    user_verified: list[str] | None = None,
    constraints: list[VariableConstraint] | None = None,
) -> RecommendationRequest:
    if task is None:
        if objective is RecommendationObjective.REDUCE_ANOMALY_SCORE:
            task = AnalysisTask.UNSUPERVISED_ANOMALY
        else:
            task = AnalysisTask.REGRESSION
    current_values = {
        "temperature": baseline_features["temperature"],
        "pressure": baseline_features["pressure"],
    }
    confirmed = (
        list(user_confirmed)
        if user_confirmed is not None
        else ["temperature", "pressure"]
    )
    verified = (
        list(user_verified) if user_verified is not None else ["temperature", "pressure"]
    )
    return RecommendationRequest(
        task=task,
        diagnosis=_diagnosis(task=task),
        objective=objective,
        current_values=current_values,
        constraints=constraints if constraints is not None else _constraints(),
        user_confirmed_controllable_variables=confirmed,
        user_verified_variables=verified,
        max_simultaneous_changes=2,
        metadata={"fixture": "recommendation_smoke"},
    )


def _safety_context(**overrides: Any) -> RecommendationSafetyContext:
    payload: dict[str, Any] = {
        "leakage_report": _safe_leakage(),
        "final_evaluation_available": True,
        "model_performance_acceptable": True,
        "model_performance_reason": None,
        "extrapolation_detected": False,
        "uncertainty_available": False,
        "uncertainty_acceptable": None,
        "metadata": {"fixture": "recommendation_smoke"},
    }
    payload.update(overrides)
    return RecommendationSafetyContext(**payload)


def _pipeline_request(
    *,
    objective: RecommendationObjective,
    baseline_features: dict[str, float],
    safety_context: RecommendationSafetyContext | None = None,
    recommendation_request: RecommendationRequest | None = None,
) -> RecommendationPipelineRequest:
    if recommendation_request is None:
        recommendation_request = _recommendation_request(
            objective=objective,
            baseline_features=baseline_features,
        )
    if objective is RecommendationObjective.REDUCE_ANOMALY_SCORE:
        quality_direction = None
        target_column = None
    else:
        quality_direction = QualityOptimizationDirection.MAXIMIZE
        target_column = TARGET_COLUMN
    return RecommendationPipelineRequest(
        recommendation_request=recommendation_request,
        safety_context=safety_context or _safety_context(),
        industry_constraints=[],
        user_overrides=[],
        feature_columns=list(FEATURE_COLUMNS),
        baseline_features=dict(baseline_features),
        target_column=target_column,
        quality_direction=quality_direction,
        quality_target=None,
        extrapolation_evaluated=False,
        extrapolation_flag=False,
        uncertainty_available=False,
        uncertainty_acceptable=None,
        metadata={"fixture": "recommendation_smoke"},
    )


def _build_pipeline(
    *,
    objective: RecommendationObjective,
    quality_model: BaseAnalysisModel | None,
    anomaly_model: BaseAnomalyModel | None,
) -> RecommendationPipeline:
    if objective is RecommendationObjective.IMPROVE_PREDICTED_QUALITY:
        scorer = CandidateScenarioScorer(
            quality_model=quality_model,
            anomaly_model=anomaly_model,
        )
    elif objective is RecommendationObjective.REDUCE_ANOMALY_SCORE:
        scorer = CandidateScenarioScorer(anomaly_model=anomaly_model)
    else:
        scorer = CandidateScenarioScorer(
            quality_model=quality_model,
            anomaly_model=anomaly_model,
        )
    return RecommendationPipeline(scenario_scorer=scorer)


# ---------------------------------------------------------------------------
# Spy adapters (call-count checks only)
# ---------------------------------------------------------------------------


class _QualityModelSpy(BaseAnalysisModel):
    def __init__(self, inner: BaseAnalysisModel) -> None:
        self._inner = inner
        self.predict_call_count = 0
        self.fit_call_count = 0

    @property
    def is_fitted(self) -> bool:
        return self._inner.is_fitted

    @property
    def feature_names(self) -> tuple[str, ...] | list[str]:
        return self._inner.feature_names

    def get_metadata(self) -> object:
        return self._inner.get_metadata()

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> _QualityModelSpy:
        self.fit_call_count += 1
        self._inner.fit(X, y)
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        self.predict_call_count += 1
        return self._inner.predict(frame)

    def evaluate(self, X: pl.DataFrame, y: pl.Series | None = None) -> object:
        return self._inner.evaluate(X, y)

    def explain(self, X: pl.DataFrame) -> object:
        return self._inner.explain(X)


class _AnomalyModelSpy(BaseAnomalyModel):
    def __init__(self, inner: BaseAnomalyModel) -> None:
        self._inner = inner
        self.score_call_count = 0
        self.fit_call_count = 0

    @property
    def is_fitted(self) -> bool:
        return self._inner.is_fitted

    @property
    def feature_names(self) -> tuple[str, ...] | list[str]:
        return self._inner.feature_names

    def get_metadata(self) -> object:
        return self._inner.get_metadata()

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> _AnomalyModelSpy:
        self.fit_call_count += 1
        self._inner.fit(X, y)
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return self._inner.predict(frame)

    def score_samples(self, frame: pl.DataFrame) -> np.ndarray:
        self.score_call_count += 1
        return self._inner.score_samples(frame)

    def evaluate(self, X: pl.DataFrame, y: pl.Series | None = None) -> object:
        return self._inner.evaluate(X, y)

    def explain(self, X: pl.DataFrame) -> object:
        return self._inner.explain(X)

    def classify_anomalies(self, X: pl.DataFrame) -> object:
        return self._inner.classify_anomalies(X)


class _NegativeAnomalyAdapter(BaseAnomalyModel):
    """Deterministic anomaly adapter for negative-score preservation only."""

    def __init__(self, feature_names: list[str]) -> None:
        self._feature_names = tuple(feature_names)
        self._is_fitted = False
        self._spec = ModelSpec(
            name="NegativeAnomalyIntegrationAdapter",
            task=AnalysisTask.UNSUPERVISED_ANOMALY,
            estimator_key="integration_negative_anomaly",
            optional_dependencies=[],
            priority=99,
            time_budget_seconds=5.0,
        )

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_name=self._spec.name,
            version="0.1.0",
            task=AnalysisTask.UNSUPERVISED_ANOMALY,
            features=list(self._feature_names),
            training_timestamp=datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
            seed=7,
            optional_dependencies_used=[],
        )

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> _NegativeAnomalyAdapter:
        del y
        assert list(X.columns) == list(self._feature_names)
        self._is_fitted = True
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        scores = self.score_samples(frame)
        return np.where(scores > 0.0, -1, 1).astype(np.int64)

    def score_samples(self, frame: pl.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("adapter must be fitted before score_samples")
        assert list(frame.columns) == list(self._feature_names)
        # Higher = more anomalous; includes finite negatives.
        temperature = frame["temperature"].to_numpy()
        pressure = frame["pressure"].to_numpy()
        flow = frame["flow_rate"].to_numpy()
        centered = (
            (temperature - 80.0) / 20.0
            + (pressure - 60.0) / 20.0
            + (flow - 20.0) / 10.0
        )
        return np.asarray(centered - 1.5, dtype=np.float64)

    def evaluate(self, X: pl.DataFrame, y: pl.Series | None = None) -> object:
        del X, y
        return {"status": "not_evaluated"}

    def explain(self, X: pl.DataFrame) -> object:
        del X
        return {"status": "not_explained"}

    def classify_anomalies(self, X: pl.DataFrame) -> object:
        labels = self.predict(X)
        return {"labels": labels.tolist()}


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _assert_scalar_metadata(metadata: dict[str, Any], *, field_name: str) -> None:
    for key, value in metadata.items():
        assert isinstance(key, str) and key.strip() != ""
        assert value is None or isinstance(value, (str, bool, int, float)), (
            f"{field_name}[{key!r}] must be scalar, got {type(value).__name__}"
        )
        if isinstance(value, float):
            assert math.isfinite(value)


def _assert_timezone_aware(value: datetime, *, field_name: str) -> None:
    assert value.tzinfo is not None and value.tzinfo.utcoffset(value) is not None, (
        f"{field_name} must be timezone-aware"
    )


def _assert_successful_full_run(report: Any) -> None:
    assert report.executed_stages == _CANONICAL_STAGES
    assert report.skipped_stages == []
    assert report.terminal_stage is RecommendationPipelineStage.RECOMMENDATION_GENERATION
    assert report.safety_decision is not None
    assert report.constraint_report is not None
    assert report.candidate_set is not None
    assert report.grid_report is not None
    assert report.scoring_report is not None
    assert report.ranking_report is not None
    assert report.final_result is not None
    assert report.status is report.final_result.status
    assert report.status in {
        RecommendationStatus.GENERATED,
        RecommendationStatus.READY_FOR_OPTIMIZATION,
    }
    if report.status is RecommendationStatus.READY_FOR_OPTIMIZATION:
        assert report.ranking_report.status is ScenarioRankingStatus.NO_IMPROVEMENT
    assert len(report.warnings) == len(set(report.warnings))
    _assert_scalar_metadata(report.metadata, field_name="report.metadata")
    _assert_scalar_metadata(
        report.final_result.metadata,
        field_name="final_result.metadata",
    )
    _assert_timezone_aware(report.started_at, field_name="started_at")
    _assert_timezone_aware(report.completed_at, field_name="completed_at")
    _assert_timezone_aware(
        report.final_result.generated_at,
        field_name="final_result.generated_at",
    )
    _assert_timezone_aware(
        report.safety_decision.evaluated_at,
        field_name="safety_decision.evaluated_at",
    )
    _assert_timezone_aware(
        report.scoring_report.evaluated_at,
        field_name="scoring_report.evaluated_at",
    )
    _assert_timezone_aware(
        report.ranking_report.evaluated_at,
        field_name="ranking_report.evaluated_at",
    )
    assert report.metadata.get("model_fit_performed") is False
    assert report.metadata.get("model_refit_performed") is False


def _assert_change_contracts(report: Any) -> None:
    result = report.final_result
    eligible = set(report.safety_decision.eligible_variables)
    grid_by_variable = {
        item.variable: item for item in report.grid_report.variable_grids
    }
    for change in result.changes:
        assert change.variable in eligible
        assert math.isclose(
            change.delta,
            change.proposed_value - change.current_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        grid = grid_by_variable[change.variable]
        assert change.proposed_value >= grid.minimum or math.isclose(
            change.proposed_value,
            grid.minimum,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        assert change.proposed_value <= grid.maximum or math.isclose(
            change.proposed_value,
            grid.maximum,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        rationale = change.rationale.lower()
        assert "model-based" in rationale or "model based" in rationale
        assert "association" in rationale
        assert "verification" in rationale
        for pattern in _FORBIDDEN_OUTCOME_PATTERNS:
            assert pattern.search(change.rationale) is None
        assert "guaranteed improvement" not in change.rationale.lower()
        assert "optimal" not in change.rationale.lower()

    for pattern in _FORBIDDEN_OUTCOME_PATTERNS:
        assert pattern.search(result.disclaimer) is None
    # Disclaimer may say improvement is "not guaranteed" / "not proven optimal".
    assert "not guaranteed" in result.disclaimer.lower()
    assert "not proven optimal" in result.disclaimer.lower()


def _business_snapshot(report: Any) -> dict[str, Any]:
    scoring = report.scoring_report
    ranking = report.ranking_report
    grid = report.grid_report
    candidate_set = report.candidate_set
    result = report.final_result
    return {
        "status": report.status,
        "terminal_stage": report.terminal_stage,
        "executed_stages": list(report.executed_stages),
        "skipped_stages": list(report.skipped_stages),
        "candidate_variables": (
            list(candidate_set.candidate_variables) if candidate_set is not None else None
        ),
        "grid_points": (
            [
                {
                    "variable": item.variable,
                    "points": [point.value for point in item.points],
                }
                for item in grid.variable_grids
            ]
            if grid is not None
            else None
        ),
        "scenario_ids": (
            [scenario.scenario_id for scenario in grid.scenarios]
            if grid is not None
            else None
        ),
        "quality_predictions": (
            [score.quality_prediction for score in scoring.scores]
            if scoring is not None
            else None
        ),
        "anomaly_scores": (
            [score.anomaly_score for score in scoring.scores]
            if scoring is not None
            else None
        ),
        "ranking_order": (
            [item.scenario_id for item in ranking.ranked_scenarios]
            if ranking is not None
            else None
        ),
        "best_scenario_id": ranking.best_scenario_id if ranking is not None else None,
        "best_nonbaseline_scenario_id": (
            ranking.best_nonbaseline_scenario_id if ranking is not None else None
        ),
        "final_changes": [
            {
                "variable": change.variable,
                "current_value": change.current_value,
                "proposed_value": change.proposed_value,
                "delta": change.delta,
                "confidence": change.confidence,
            }
            for change in result.changes
        ],
        "baseline_prediction": result.baseline_prediction,
        "proposed_prediction": result.proposed_prediction,
        "baseline_anomaly_score": result.baseline_anomaly_score,
        "proposed_anomaly_score": result.proposed_anomaly_score,
        "final_confidence": result.confidence,
    }


def _stable_model_metadata(model: BaseAnalysisModel | BaseAnomalyModel) -> dict[str, Any]:
    metadata = model.get_metadata()
    return {
        "model_name": metadata.model_name,
        "version": metadata.version,
        "task": metadata.task,
        "features": list(metadata.features),
        "seed": metadata.seed,
        "optional_dependencies_used": list(metadata.optional_dependencies_used),
        "is_fitted": bool(model.is_fitted),
        "feature_names": list(model.feature_names),
    }


# ---------------------------------------------------------------------------
# Test 1: Quality objective smoke
# ---------------------------------------------------------------------------


def test_quality_objective_smoke(
    fitted_quality_model: BaseAnalysisModel,
    fitted_anomaly_model: BaseAnomalyModel,
    baseline_features: dict[str, float],
) -> None:
    quality_spy = _QualityModelSpy(fitted_quality_model)
    anomaly_spy = _AnomalyModelSpy(fitted_anomaly_model)
    pipeline = _build_pipeline(
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        quality_model=quality_spy,
        anomaly_model=anomaly_spy,
    )
    request = _pipeline_request(
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        baseline_features=baseline_features,
    )
    outcome = pipeline.run(request)
    report = outcome.report
    _assert_successful_full_run(report)

    scoring = report.scoring_report
    ranking = report.ranking_report
    assert scoring is not None
    assert ranking is not None
    assert scoring.quality_scored_count == len(scoring.scores)
    assert scoring.quality_scored_count == len(report.grid_report.scenarios)
    baseline_score = next(
        item for item in scoring.scores if item.scenario_type is CandidateScenarioType.BASELINE
    )
    assert baseline_score.quality_prediction is not None
    assert math.isfinite(baseline_score.quality_prediction)
    assert ranking.quality_direction is QualityOptimizationDirection.MAXIMIZE
    assert report.final_result.baseline_prediction is not None
    assert math.isfinite(report.final_result.baseline_prediction)

    if report.status is RecommendationStatus.GENERATED:
        assert report.final_result.proposed_prediction is not None
        assert math.isfinite(report.final_result.proposed_prediction)
        assert report.final_result.changes
        _assert_change_contracts(report)

    if scoring.anomaly_scored_count > 0:
        for item in scoring.scores:
            if item.anomaly_scored:
                assert item.anomaly_score is not None
                assert math.isfinite(item.anomaly_score)

    assert quality_spy.fit_call_count == 0
    assert anomaly_spy.fit_call_count == 0
    assert quality_spy.predict_call_count == 1


# ---------------------------------------------------------------------------
# Test 2: Anomaly objective smoke
# ---------------------------------------------------------------------------


def test_anomaly_objective_smoke(
    fitted_anomaly_model: BaseAnomalyModel,
    baseline_features: dict[str, float],
) -> None:
    anomaly_spy = _AnomalyModelSpy(fitted_anomaly_model)
    pipeline = _build_pipeline(
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        quality_model=None,
        anomaly_model=anomaly_spy,
    )
    request = _pipeline_request(
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        baseline_features=baseline_features,
    )
    outcome = pipeline.run(request)
    report = outcome.report
    _assert_successful_full_run(report)

    scoring = report.scoring_report
    ranking = report.ranking_report
    assert scoring is not None
    assert ranking is not None
    assert scoring.anomaly_scored_count == len(scoring.scores)
    assert scoring.anomaly_scored_count == len(report.grid_report.scenarios)
    assert ranking.quality_direction is None

    baseline_score = next(
        item for item in scoring.scores if item.scenario_type is CandidateScenarioType.BASELINE
    )
    assert baseline_score.anomaly_score is not None
    assert math.isfinite(baseline_score.anomaly_score)
    assert report.final_result.baseline_anomaly_score == pytest.approx(
        baseline_score.anomaly_score
    )
    assert ranking.baseline_anomaly_score == pytest.approx(baseline_score.anomaly_score)

    for ranked in ranking.ranked_scenarios:
        scored = next(
            item for item in scoring.scores if item.scenario_id == ranked.scenario_id
        )
        assert ranked.anomaly_score == pytest.approx(scored.anomaly_score)
        if ranked.scenario_type is not CandidateScenarioType.BASELINE:
            assert ranked.anomaly_improvement_from_baseline == pytest.approx(
                baseline_score.anomaly_score - scored.anomaly_score
            )

    if report.status is RecommendationStatus.GENERATED:
        assert report.final_result.proposed_anomaly_score is not None
        selected_id = ranking.best_nonbaseline_scenario_id
        assert selected_id is not None
        selected = next(
            item for item in ranking.ranked_scenarios if item.scenario_id == selected_id
        )
        assert report.final_result.proposed_anomaly_score == pytest.approx(
            selected.anomaly_score
        )
        _assert_change_contracts(report)
        for change in report.final_result.changes:
            assert change.variable in report.safety_decision.eligible_variables

    assert anomaly_spy.fit_call_count == 0
    assert anomaly_spy.score_call_count == 1


# ---------------------------------------------------------------------------
# Test 3: Balance objective smoke
# ---------------------------------------------------------------------------


def test_balance_objective_smoke(
    fitted_quality_model: BaseAnalysisModel,
    fitted_anomaly_model: BaseAnomalyModel,
    baseline_features: dict[str, float],
) -> None:
    quality_spy = _QualityModelSpy(fitted_quality_model)
    anomaly_spy = _AnomalyModelSpy(fitted_anomaly_model)
    pipeline = _build_pipeline(
        objective=RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
        quality_model=quality_spy,
        anomaly_model=anomaly_spy,
    )
    request = _pipeline_request(
        objective=RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
        baseline_features=baseline_features,
    )
    outcome = pipeline.run(request)
    report = outcome.report
    _assert_successful_full_run(report)

    scoring = report.scoring_report
    ranking = report.ranking_report
    assert scoring is not None
    assert ranking is not None
    assert scoring.quality_scored_count == len(scoring.scores)
    assert scoring.anomaly_scored_count == len(scoring.scores)
    for item in scoring.scores:
        assert item.quality_prediction is not None
        assert item.anomaly_score is not None
        assert math.isfinite(item.quality_prediction)
        assert math.isfinite(item.anomaly_score)

    for ranked in ranking.ranked_scenarios:
        assert ranked.normalized_quality_benefit is not None
        assert ranked.normalized_anomaly_benefit is not None
        assert math.isfinite(ranked.objective_benefit_score)

    if ranking.best_nonbaseline_scenario_id is not None:
        selected = next(
            item
            for item in ranking.ranked_scenarios
            if item.scenario_id == ranking.best_nonbaseline_scenario_id
        )
        assert selected.scenario_type is not CandidateScenarioType.BASELINE
        if report.status is RecommendationStatus.GENERATED:
            assert report.final_result.proposed_prediction == pytest.approx(
                selected.quality_prediction
            )
            assert report.final_result.proposed_anomaly_score == pytest.approx(
                selected.anomaly_score
            )
            _assert_change_contracts(report)

    negative_scores = [
        item.anomaly_score
        for item in scoring.scores
        if item.anomaly_score is not None and item.anomaly_score < 0.0
    ]
    if negative_scores:
        for value in negative_scores:
            matching = [
                item.anomaly_score
                for item in ranking.ranked_scenarios
                if item.anomaly_score is not None
                and math.isclose(item.anomaly_score, value, rel_tol=0.0, abs_tol=1e-12)
            ]
            assert matching

    assert quality_spy.predict_call_count == 1
    assert anomaly_spy.score_call_count == 1
    assert quality_spy.fit_call_count == 0
    assert anomaly_spy.fit_call_count == 0


# ---------------------------------------------------------------------------
# Test 4: Safety refusal smoke
# ---------------------------------------------------------------------------


def test_safety_refusal_smoke(
    fitted_quality_model: BaseAnalysisModel,
    fitted_anomaly_model: BaseAnomalyModel,
    baseline_features: dict[str, float],
) -> None:
    quality_spy = _QualityModelSpy(fitted_quality_model)
    anomaly_spy = _AnomalyModelSpy(fitted_anomaly_model)
    pipeline = RecommendationPipeline(
        scenario_scorer=CandidateScenarioScorer(
            quality_model=quality_spy,
            anomaly_model=anomaly_spy,
        ),
    )
    request = _pipeline_request(
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        baseline_features=baseline_features,
        safety_context=_safety_context(
            leakage_report=_blocker_leakage(),
            final_evaluation_available=False,
        ),
    )
    outcome = pipeline.run(request)
    report = outcome.report

    assert report.status is RecommendationStatus.REFUSED
    assert report.terminal_stage is RecommendationPipelineStage.SAFETY
    assert report.executed_stages == [RecommendationPipelineStage.SAFETY]
    assert report.skipped_stages == _CANONICAL_STAGES[1:]
    assert report.constraint_report is None
    assert report.candidate_set is None
    assert report.grid_report is None
    assert report.scoring_report is None
    assert report.ranking_report is None
    assert report.final_result.changes == []
    assert report.final_result.proposed_prediction is None
    assert report.final_result.proposed_anomaly_score is None
    assert report.final_result.confidence == pytest.approx(0.0)
    assert report.safety_decision.status is RecommendationSafetyStatus.REFUSED
    assert RecommendationReasonCode.LEAKAGE_BLOCKER in (
        report.safety_decision.global_reason_codes
    ) or RecommendationReasonCode.FINAL_EVALUATION_MISSING in (
        report.safety_decision.global_reason_codes
    )
    assert any("leakage" in message.lower() or "evaluation" in message.lower()
               for message in report.warnings + report.safety_decision.messages)
    assert "pipeline_terminal_stage" in report.final_result.metadata
    assert quality_spy.predict_call_count == 0
    assert anomaly_spy.score_call_count == 0
    assert quality_spy.fit_call_count == 0
    assert anomaly_spy.fit_call_count == 0


# ---------------------------------------------------------------------------
# Test 5: No eligible variable smoke
# ---------------------------------------------------------------------------


def test_no_eligible_variable_smoke(
    fitted_quality_model: BaseAnalysisModel,
    fitted_anomaly_model: BaseAnomalyModel,
    baseline_features: dict[str, float],
) -> None:
    quality_spy = _QualityModelSpy(fitted_quality_model)
    anomaly_spy = _AnomalyModelSpy(fitted_anomaly_model)
    pipeline = RecommendationPipeline(
        scenario_scorer=CandidateScenarioScorer(
            quality_model=quality_spy,
            anomaly_model=anomaly_spy,
        ),
    )
    recommendation_request = _recommendation_request(
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        baseline_features=baseline_features,
        user_confirmed=[],
        user_verified=[],
    )
    request = _pipeline_request(
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        baseline_features=baseline_features,
        recommendation_request=recommendation_request,
    )
    outcome = pipeline.run(request)
    report = outcome.report

    assert report.status in {
        RecommendationStatus.REFUSED,
        RecommendationStatus.READY_FOR_OPTIMIZATION,
    }
    assert report.terminal_stage in {
        RecommendationPipelineStage.SAFETY,
        RecommendationPipelineStage.CANDIDATE_SELECTION,
    }
    assert report.final_result.changes == []
    assert report.final_result.proposed_prediction is None
    assert report.final_result.proposed_anomaly_score is None
    if report.candidate_set is not None:
        assert report.candidate_set.candidates == []
    assert report.grid_report is None or report.grid_report.scenarios == []
    assert report.scoring_report is None
    assert report.ranking_report is None
    assert quality_spy.predict_call_count == 0
    assert anomaly_spy.score_call_count == 0


# ---------------------------------------------------------------------------
# Test 6: Negative anomaly score preservation
# ---------------------------------------------------------------------------


def test_negative_anomaly_score_preservation(
    fitted_anomaly_model: BaseAnomalyModel,
    baseline_features: dict[str, float],
    synthetic_frame: tuple[pl.DataFrame, pl.Series],
) -> None:
    frame, _ = synthetic_frame
    real_scores = fitted_anomaly_model.score_samples(frame.select(FEATURE_COLUMNS))
    use_real_adapter = bool(np.any(real_scores < 0.0))

    if use_real_adapter:
        anomaly_model: BaseAnomalyModel = fitted_anomaly_model
    else:
        anomaly_model = _NegativeAnomalyAdapter(list(FEATURE_COLUMNS))
        anomaly_model.fit(frame.select(FEATURE_COLUMNS))

    pipeline = RecommendationPipeline(
        scenario_scorer=CandidateScenarioScorer(anomaly_model=anomaly_model),
    )
    request = _pipeline_request(
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        baseline_features=baseline_features,
    )
    outcome = pipeline.run(request)
    report = outcome.report
    _assert_successful_full_run(report)

    scoring = report.scoring_report
    ranking = report.ranking_report
    assert scoring is not None
    assert ranking is not None

    scored_values = [
        item.anomaly_score
        for item in scoring.scores
        if item.anomaly_score is not None
    ]
    assert scored_values
    assert any(value < 0.0 for value in scored_values) or not use_real_adapter
    if not use_real_adapter:
        assert any(value < 0.0 for value in scored_values)

    for item in scoring.scores:
        if item.anomaly_score is None:
            continue
        ranked = next(
            row for row in ranking.ranked_scenarios if row.scenario_id == item.scenario_id
        )
        assert ranked.anomaly_score == pytest.approx(item.anomaly_score)
        assert math.isfinite(item.anomaly_score)

    if report.final_result.baseline_anomaly_score is not None:
        baseline = next(
            item
            for item in scoring.scores
            if item.scenario_type is CandidateScenarioType.BASELINE
        )
        assert report.final_result.baseline_anomaly_score == pytest.approx(
            baseline.anomaly_score
        )
        assert ranking.baseline_anomaly_score == pytest.approx(baseline.anomaly_score)

    if report.final_result.proposed_anomaly_score is not None:
        selected_id = ranking.best_nonbaseline_scenario_id
        assert selected_id is not None
        selected = next(
            item for item in ranking.ranked_scenarios if item.scenario_id == selected_id
        )
        assert report.final_result.proposed_anomaly_score == pytest.approx(
            selected.anomaly_score
        )


# ---------------------------------------------------------------------------
# Test 7: Determinism smoke
# ---------------------------------------------------------------------------


def test_determinism_smoke(
    fitted_quality_model: BaseAnalysisModel,
    fitted_anomaly_model: BaseAnomalyModel,
    baseline_features: dict[str, float],
) -> None:
    pipeline = _build_pipeline(
        objective=RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
        quality_model=fitted_quality_model,
        anomaly_model=fitted_anomaly_model,
    )
    request = _pipeline_request(
        objective=RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
        baseline_features=baseline_features,
    )
    before_quality = _stable_model_metadata(fitted_quality_model)
    before_anomaly = _stable_model_metadata(fitted_anomaly_model)
    before_pipeline = dict(pipeline.get_metadata())

    first = pipeline.run(request).report
    second = pipeline.run(request).report

    assert _business_snapshot(first) == _business_snapshot(second)
    assert _stable_model_metadata(fitted_quality_model) == before_quality
    assert _stable_model_metadata(fitted_anomaly_model) == before_anomaly
    assert dict(pipeline.get_metadata()) == before_pipeline
    assert first.metadata.get("model_fit_performed") is False
    assert second.metadata.get("model_refit_performed") is False


# ---------------------------------------------------------------------------
# Test 8: Input immutability smoke
# ---------------------------------------------------------------------------


def test_input_immutability_smoke(
    fitted_quality_model: BaseAnalysisModel,
    fitted_anomaly_model: BaseAnomalyModel,
    baseline_features: dict[str, float],
) -> None:
    recommendation_request = _recommendation_request(
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        baseline_features=baseline_features,
    )
    safety_context = _safety_context()
    request = _pipeline_request(
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        baseline_features=baseline_features,
        recommendation_request=recommendation_request,
        safety_context=safety_context,
    )

    request_before = request.model_dump(mode="python")
    recommendation_before = recommendation_request.model_dump(mode="python")
    diagnosis_before = recommendation_request.diagnosis.model_dump(mode="python")
    safety_before = safety_context.model_dump(mode="python")
    constraints_before = [
        item.model_dump(mode="python") for item in recommendation_request.constraints
    ]
    baseline_before = dict(baseline_features)
    quality_before = _stable_model_metadata(fitted_quality_model)
    anomaly_before = _stable_model_metadata(fitted_anomaly_model)

    pipeline = _build_pipeline(
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        quality_model=fitted_quality_model,
        anomaly_model=fitted_anomaly_model,
    )
    outcome = pipeline.run(request)
    _assert_successful_full_run(outcome.report)

    assert request.model_dump(mode="python") == request_before
    assert recommendation_request.model_dump(mode="python") == recommendation_before
    assert recommendation_request.diagnosis.model_dump(mode="python") == diagnosis_before
    assert safety_context.model_dump(mode="python") == safety_before
    assert [
        item.model_dump(mode="python") for item in recommendation_request.constraints
    ] == constraints_before
    assert baseline_features == baseline_before
    assert _stable_model_metadata(fitted_quality_model) == quality_before
    assert _stable_model_metadata(fitted_anomaly_model) == anomaly_before
