"""Unit tests for Isolation Forest anomaly detection (Step 7A)."""

from __future__ import annotations

import copy
import math
from datetime import UTC, date
from typing import Any, Self

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import pytest
import sklearn
from pydantic import ValidationError
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.exceptions import (
    DataValidationError,
    InsufficientDataError,
    ProcessIntelligenceError,
)
from process_intelligence.core.protocols import BaseAnalysisModel, BaseAnomalyModel
from process_intelligence.core.schemas import ExplanationResult, ModelMetadata, ModelSpec
from process_intelligence.models import (
    AnomalyDetectionResult,
    IsolationForestAnomalyModel,
    IsolationForestConfig,
    create_isolation_forest_anomaly_model,
)
from process_intelligence.models.anomaly import IsolationForestModelMetadata


def _anomaly_spec(**overrides: Any) -> ModelSpec:
    payload: dict[str, Any] = {
        "name": "Isolation Forest",
        "task": AnalysisTask.UNSUPERVISED_ANOMALY,
        "estimator_key": "isolation_forest",
        "optional_dependencies": [],
        "priority": 10,
        "time_budget_seconds": 15.0,
    }
    payload.update(overrides)
    return ModelSpec(**payload)


def _cluster_frame(
    *,
    rows: int = 40,
    as_pandas: bool = False,
    include_outlier: bool = True,
) -> Any:
    """Build a small deterministic numeric frame with an optional extreme row."""
    rng = np.random.default_rng(0)
    normal = rng.normal(loc=0.0, scale=1.0, size=(rows, 2))
    if include_outlier:
        normal[-1] = np.array([25.0, -25.0])
    data = {
        "f1": normal[:, 0].tolist(),
        "f2": normal[:, 1].tolist(),
    }
    if as_pandas:
        return pd.DataFrame(data)
    return pl.DataFrame(data)


def _empty_like(frame: Any) -> Any:
    if isinstance(frame, pl.DataFrame):
        return frame.clear()
    return frame.iloc[0:0].copy()


class _MissingFit:
    def predict(self, X: Any) -> np.ndarray:
        return np.ones(len(X), dtype=int)

    def decision_function(self, X: Any) -> np.ndarray:
        return np.zeros(len(X), dtype=float)

    def score_samples(self, X: Any) -> np.ndarray:
        return np.zeros(len(X), dtype=float)


class _MissingPredict:
    def fit(self, X: Any, y: Any = None) -> Self:
        _ = (X, y)
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        return np.zeros(len(X), dtype=float)

    def score_samples(self, X: Any) -> np.ndarray:
        return np.zeros(len(X), dtype=float)


class _MissingDecisionFunction:
    def fit(self, X: Any, y: Any = None) -> Self:
        _ = (X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.ones(len(X), dtype=int)

    def score_samples(self, X: Any) -> np.ndarray:
        return np.zeros(len(X), dtype=float)


class _MissingScoreSamples:
    def fit(self, X: Any, y: Any = None) -> Self:
        _ = (X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.ones(len(X), dtype=int)

    def decision_function(self, X: Any) -> np.ndarray:
        return np.zeros(len(X), dtype=float)


class _NotCloneable:
    def fit(self, X: Any, y: Any = None) -> Self:
        _ = (X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.ones(len(X), dtype=int)

    def decision_function(self, X: Any) -> np.ndarray:
        return np.zeros(len(X), dtype=float)

    def score_samples(self, X: Any) -> np.ndarray:
        return np.zeros(len(X), dtype=float)


class _FailingFitEstimator:
    def __init__(self, random_state: int | None = None) -> None:
        self.random_state = random_state

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        _ = deep
        return {"random_state": self.random_state}

    def set_params(self, **params: Any) -> Self:
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X: Any, y: Any = None) -> Self:
        _ = (X, y)
        raise ValueError("intentional fit failure")

    def predict(self, X: Any) -> np.ndarray:
        return np.ones(len(X), dtype=int)

    def decision_function(self, X: Any) -> np.ndarray:
        return np.zeros(len(X), dtype=float)

    def score_samples(self, X: Any) -> np.ndarray:
        return np.zeros(len(X), dtype=float)


def test_config_defaults() -> None:
    config = IsolationForestConfig()
    assert config.n_estimators == 100
    assert config.contamination == 0.05
    assert config.max_samples == "auto"
    assert config.max_features == 1.0
    assert config.bootstrap is False
    assert config.random_state == 42
    assert config.n_jobs == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_estimators", 0),
        ("n_estimators", -1),
        ("n_estimators", True),
        ("contamination", 0.0),
        ("contamination", 0.51),
        ("contamination", True),
        ("contamination", float("nan")),
        ("contamination", float("inf")),
        ("contamination", "AUTO"),
        ("max_samples", 0),
        ("max_samples", 0.0),
        ("max_samples", 1.1),
        ("max_samples", True),
        ("max_features", 0),
        ("max_features", 1.1),
        ("max_features", True),
        ("bootstrap", 1),
        ("random_state", -1),
        ("random_state", True),
        ("n_jobs", 0),
        ("n_jobs", -2),
        ("n_jobs", True),
    ],
)
def test_config_invalid_values_rejected(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        IsolationForestConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contamination", "auto"),
        ("contamination", 0.1),
        ("max_samples", "auto"),
        ("max_samples", 10),
        ("max_samples", 0.5),
        ("max_features", 2),
        ("max_features", 0.5),
        ("n_jobs", -1),
    ],
)
def test_config_valid_values_accepted(field: str, value: Any) -> None:
    config = IsolationForestConfig(**{field: value})
    assert getattr(config, field) == value


def test_config_round_trip() -> None:
    config = IsolationForestConfig(n_estimators=50, contamination="auto", n_jobs=-1)
    restored = IsolationForestConfig.model_validate(config.model_dump())
    assert restored == config


def test_result_valid_and_empty() -> None:
    result = AnomalyDetectionResult(
        scores=[0.2, 1.5],
        is_anomaly=[False, True],
        raw_predictions=[1, -1],
        threshold=0.0,
        row_count=2,
        anomaly_count=1,
        anomaly_fraction=0.5,
        score_min=0.2,
        score_max=1.5,
        score_mean=0.85,
    )
    assert result.anomaly_count == 1

    empty = AnomalyDetectionResult(
        scores=[],
        is_anomaly=[],
        raw_predictions=[],
        threshold=0.0,
        row_count=0,
        anomaly_count=0,
        anomaly_fraction=0.0,
        score_min=None,
        score_max=None,
        score_mean=None,
    )
    assert empty.row_count == 0


@pytest.mark.parametrize(
    "payload",
    [
        {
            "scores": [0.1],
            "is_anomaly": [False],
            "raw_predictions": [1],
            "threshold": 0.0,
            "row_count": -1,
            "anomaly_count": 0,
            "anomaly_fraction": 0.0,
            "score_min": 0.1,
            "score_max": 0.1,
            "score_mean": 0.1,
        },
        {
            "scores": [0.1],
            "is_anomaly": [False],
            "raw_predictions": [1],
            "threshold": 0.0,
            "row_count": 1,
            "anomaly_count": -1,
            "anomaly_fraction": 0.0,
            "score_min": 0.1,
            "score_max": 0.1,
            "score_mean": 0.1,
        },
        {
            "scores": [0.1],
            "is_anomaly": [False],
            "raw_predictions": [1],
            "threshold": 0.0,
            "row_count": 1,
            "anomaly_count": 2,
            "anomaly_fraction": 1.0,
            "score_min": 0.1,
            "score_max": 0.1,
            "score_mean": 0.1,
        },
        {
            "scores": [0.1, 0.2],
            "is_anomaly": [False],
            "raw_predictions": [1],
            "threshold": 0.0,
            "row_count": 1,
            "anomaly_count": 0,
            "anomaly_fraction": 0.0,
            "score_min": 0.1,
            "score_max": 0.1,
            "score_mean": 0.1,
        },
        {
            "scores": [0.1],
            "is_anomaly": [False, True],
            "raw_predictions": [1],
            "threshold": 0.0,
            "row_count": 1,
            "anomaly_count": 0,
            "anomaly_fraction": 0.0,
            "score_min": 0.1,
            "score_max": 0.1,
            "score_mean": 0.1,
        },
        {
            "scores": [0.1],
            "is_anomaly": [False],
            "raw_predictions": [1, -1],
            "threshold": 0.0,
            "row_count": 1,
            "anomaly_count": 0,
            "anomaly_fraction": 0.0,
            "score_min": 0.1,
            "score_max": 0.1,
            "score_mean": 0.1,
        },
        {
            "scores": [0.1],
            "is_anomaly": [False],
            "raw_predictions": [0],
            "threshold": 0.0,
            "row_count": 1,
            "anomaly_count": 0,
            "anomaly_fraction": 0.0,
            "score_min": 0.1,
            "score_max": 0.1,
            "score_mean": 0.1,
        },
        {
            "scores": [0.1],
            "is_anomaly": [True],
            "raw_predictions": [1],
            "threshold": 0.0,
            "row_count": 1,
            "anomaly_count": 1,
            "anomaly_fraction": 1.0,
            "score_min": 0.1,
            "score_max": 0.1,
            "score_mean": 0.1,
        },
        {
            "scores": [0.1],
            "is_anomaly": [True],
            "raw_predictions": [-1],
            "threshold": 0.0,
            "row_count": 1,
            "anomaly_count": 0,
            "anomaly_fraction": 0.0,
            "score_min": 0.1,
            "score_max": 0.1,
            "score_mean": 0.1,
        },
        {
            "scores": [0.1],
            "is_anomaly": [False],
            "raw_predictions": [1],
            "threshold": 0.0,
            "row_count": 1,
            "anomaly_count": 0,
            "anomaly_fraction": 1.5,
            "score_min": 0.1,
            "score_max": 0.1,
            "score_mean": 0.1,
        },
        {
            "scores": [0.1],
            "is_anomaly": [False],
            "raw_predictions": [1],
            "threshold": 0.0,
            "row_count": 1,
            "anomaly_count": 0,
            "anomaly_fraction": 0.5,
            "score_min": 0.1,
            "score_max": 0.1,
            "score_mean": 0.1,
        },
        {
            "scores": [float("nan")],
            "is_anomaly": [False],
            "raw_predictions": [1],
            "threshold": 0.0,
            "row_count": 1,
            "anomaly_count": 0,
            "anomaly_fraction": 0.0,
            "score_min": 0.0,
            "score_max": 0.0,
            "score_mean": 0.0,
        },
        {
            "scores": [float("inf")],
            "is_anomaly": [False],
            "raw_predictions": [1],
            "threshold": 0.0,
            "row_count": 1,
            "anomaly_count": 0,
            "anomaly_fraction": 0.0,
            "score_min": 0.0,
            "score_max": 0.0,
            "score_mean": 0.0,
        },
        {
            "scores": [0.1],
            "is_anomaly": [False],
            "raw_predictions": [1],
            "threshold": float("nan"),
            "row_count": 1,
            "anomaly_count": 0,
            "anomaly_fraction": 0.0,
            "score_min": 0.1,
            "score_max": 0.1,
            "score_mean": 0.1,
        },
        {
            "scores": [],
            "is_anomaly": [],
            "raw_predictions": [],
            "threshold": 0.0,
            "row_count": 0,
            "anomaly_count": 0,
            "anomaly_fraction": 0.0,
            "score_min": 0.0,
            "score_max": 0.0,
            "score_mean": 0.0,
        },
        {
            "scores": [0.1],
            "is_anomaly": [False],
            "raw_predictions": [1],
            "threshold": 0.0,
            "row_count": 1,
            "anomaly_count": 0,
            "anomaly_fraction": 0.0,
            "score_min": None,
            "score_max": 0.1,
            "score_mean": 0.1,
        },
        {
            "scores": [0.1, 0.3],
            "is_anomaly": [False, False],
            "raw_predictions": [1, 1],
            "threshold": 0.0,
            "row_count": 2,
            "anomaly_count": 0,
            "anomaly_fraction": 0.0,
            "score_min": 0.3,
            "score_max": 0.1,
            "score_mean": 0.2,
        },
        {
            "scores": [0.1],
            "is_anomaly": [False],
            "raw_predictions": [1],
            "threshold": 0.0,
            "row_count": 1,
            "anomaly_count": 0,
            "anomaly_fraction": 0.0,
            "score_min": 0.1,
            "score_max": 0.1,
            "score_mean": 0.1,
            "warnings": ["a", "a"],
        },
    ],
)
def test_result_invalid_payloads_rejected(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        AnomalyDetectionResult(**payload)


def test_result_mutable_defaults_independent_and_round_trip() -> None:
    first = AnomalyDetectionResult(
        threshold=0.0,
        row_count=0,
        anomaly_count=0,
        anomaly_fraction=0.0,
    )
    second = AnomalyDetectionResult(
        threshold=0.0,
        row_count=0,
        anomaly_count=0,
        anomaly_fraction=0.0,
    )
    first.warnings.append("x")
    assert second.warnings == []
    restored = AnomalyDetectionResult.model_validate(first.model_dump())
    assert restored.warnings == ["x"]


def test_model_construction_and_base_types() -> None:
    model = IsolationForestAnomalyModel(spec=_anomaly_spec())
    assert isinstance(model, BaseAnomalyModel)
    assert isinstance(model, BaseAnalysisModel)
    assert model.is_fitted is False
    assert model.feature_names == ()
    assert model.fit_row_count == 0
    assert model.threshold is None
    assert model.fitted_at is None


def test_constructor_rejects_invalid_spec_config_estimator() -> None:
    with pytest.raises(TypeError):
        IsolationForestAnomalyModel(spec="bad")  # type: ignore[arg-type]
    with pytest.raises(DataValidationError):
        IsolationForestAnomalyModel(
            spec=_anomaly_spec(task=AnalysisTask.REGRESSION),
        )
    with pytest.raises(TypeError):
        IsolationForestAnomalyModel(spec=_anomaly_spec(), config="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        IsolationForestAnomalyModel(spec=_anomaly_spec(), estimator=_MissingFit())
    with pytest.raises(TypeError):
        IsolationForestAnomalyModel(spec=_anomaly_spec(), estimator=_MissingPredict())
    with pytest.raises(TypeError):
        IsolationForestAnomalyModel(
            spec=_anomaly_spec(),
            estimator=_MissingDecisionFunction(),
        )
    with pytest.raises(TypeError):
        IsolationForestAnomalyModel(
            spec=_anomaly_spec(),
            estimator=_MissingScoreSamples(),
        )
    with pytest.raises(TypeError):
        IsolationForestAnomalyModel(spec=_anomaly_spec(), estimator=_NotCloneable())
    with pytest.raises(TypeError):
        IsolationForestAnomalyModel(
            spec=_anomaly_spec(),
            estimator=LinearRegression(),
        )


def test_estimator_clone_and_config_independence() -> None:
    external = IsolationForest(n_estimators=20, random_state=7, contamination=0.1)
    before = copy.deepcopy(external.get_params())
    config = IsolationForestConfig(random_state=99, n_estimators=20)
    model = IsolationForestAnomalyModel(
        spec=_anomaly_spec(),
        config=config,
        estimator=external,
    )
    frame = _cluster_frame()
    model.fit(frame)

    assert external.get_params() == before
    with pytest.raises((ValueError, TypeError, AttributeError)):
        check_is_fitted(external)
    assert model._estimator_template.get_params()["random_state"] == 99
    assert external.get_params()["random_state"] == 7

    config.random_state = 1
    assert model.get_metadata().seed == 99

    other = IsolationForestAnomalyModel(
        spec=_anomaly_spec(),
        config=IsolationForestConfig(),
    )
    other.fit(frame)
    assert model._fitted_estimator is not other._fitted_estimator


@pytest.mark.parametrize("as_pandas", [False, True])
def test_fit_accepts_polars_and_pandas(as_pandas: bool) -> None:
    model = IsolationForestAnomalyModel(spec=_anomaly_spec())
    frame = _cluster_frame(as_pandas=as_pandas)
    snapshot = frame.copy() if as_pandas else frame.clone()
    fitted = model.fit(frame)
    assert fitted is model
    assert model.is_fitted is True
    if as_pandas:
        pd.testing.assert_frame_equal(frame, snapshot)
    else:
        assert frame.equals(snapshot)


def test_x_validation_errors_include_column_names() -> None:
    model = IsolationForestAnomalyModel(spec=_anomaly_spec())
    with pytest.raises(TypeError):
        model.fit([[1.0, 2.0]])  # type: ignore[arg-type]
    with pytest.raises(InsufficientDataError):
        model.fit(pl.DataFrame({"f1": pl.Series([], dtype=pl.Float64)}))
    with pytest.raises(InsufficientDataError):
        model.fit(pl.DataFrame())
    with pytest.raises(DataValidationError):
        model.fit(pd.DataFrame({1: [1.0, 2.0]}))
    with pytest.raises(DataValidationError):
        model.fit(pl.DataFrame({"": [1.0, 2.0]}))
    with pytest.raises(DataValidationError):
        model.fit(pl.DataFrame({"   ": [1.0, 2.0]}))
    with pytest.raises(DataValidationError):
        bad = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["f1", "f1"])
        model.fit(bad)
    with pytest.raises(DataValidationError, match="_original_row_id"):
        model.fit(pl.DataFrame({"_original_row_id": [1.0, 2.0], "f1": [1.0, 2.0]}))

    with pytest.raises(DataValidationError, match="flag"):
        model.fit(pl.DataFrame({"flag": [True, False], "f1": [1.0, 2.0]}))
    with pytest.raises(DataValidationError, match="label"):
        model.fit(pl.DataFrame({"label": ["a", "b"], "f1": [1.0, 2.0]}))
    with pytest.raises(DataValidationError, match="day"):
        model.fit(
            pl.DataFrame(
                {
                    "day": [date(2020, 1, 1), date(2020, 1, 2)],
                    "f1": [1.0, 2.0],
                }
            )
        )
    with pytest.raises(DataValidationError, match="f1"):
        model.fit(pl.DataFrame({"f1": [1.0, None], "f2": [1.0, 2.0]}))
    with pytest.raises(DataValidationError, match="f1"):
        model.fit(pl.DataFrame({"f1": [1.0, float("nan")], "f2": [1.0, 2.0]}))
    with pytest.raises(DataValidationError, match="f1"):
        model.fit(pl.DataFrame({"f1": [1.0, float("inf")], "f2": [1.0, 2.0]}))
    with pytest.raises(DataValidationError, match="f1"):
        model.fit(pl.DataFrame({"f1": [1.0, float("-inf")], "f2": [1.0, 2.0]}))

    ok_int = pl.DataFrame({"f1": [1, 2, 3, 4], "f2": [5, 6, 7, 8]})
    ok_float = pl.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [0.5, 0.6, 0.7, 0.8]})
    IsolationForestAnomalyModel(spec=_anomaly_spec()).fit(ok_int)
    IsolationForestAnomalyModel(spec=_anomaly_spec()).fit(ok_float)


def test_fit_updates_state_and_rejects_y() -> None:
    model = IsolationForestAnomalyModel(spec=_anomaly_spec())
    frame = _cluster_frame()
    returned = model.fit(frame)
    assert returned is model
    assert isinstance(returned, IsolationForestAnomalyModel)
    assert model.is_fitted is True
    assert model.feature_names == ("f1", "f2")
    assert isinstance(model.feature_names, tuple)
    assert model.fit_row_count == frame.height
    assert model.threshold == 0.0
    assert model.fitted_at is not None
    assert model.fitted_at.tzinfo == UTC

    with pytest.raises(DataValidationError):
        model.fit(frame, y=pl.Series("y", [0.0] * frame.height))


def test_fit_atomicity_and_refit() -> None:
    model = IsolationForestAnomalyModel(
        spec=_anomaly_spec(),
        estimator=_FailingFitEstimator(random_state=0),
    )
    with pytest.raises(ValueError, match="intentional fit failure"):
        model.fit(_cluster_frame())
    assert model.is_fitted is False
    assert model.feature_names == ()
    assert model.fit_row_count == 0

    model = IsolationForestAnomalyModel(spec=_anomaly_spec())
    first = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.1, 0.2, 0.3, 0.4]})
    model.fit(first)
    assert model.feature_names == ("a", "b")
    assert model.fit_row_count == 4
    previous_estimator = model._fitted_estimator

    second = pl.DataFrame(
        {
            "x": [float(i) for i in range(10)],
            "y": [float(i) * 0.5 for i in range(10)],
        }
    )
    model.fit(second)
    assert model.feature_names == ("x", "y")
    assert model.fit_row_count == 10
    assert model._fitted_estimator is not previous_estimator

    preserved_names = model.feature_names
    preserved_count = model.fit_row_count
    preserved_fitted_at = model.fitted_at
    preserved_estimator = model._fitted_estimator
    model._estimator_template = _FailingFitEstimator(random_state=1)
    with pytest.raises(ValueError, match="intentional fit failure"):
        model.fit(first)
    assert model.is_fitted is True
    assert model.feature_names == preserved_names
    assert model.fit_row_count == preserved_count
    assert model.fitted_at == preserved_fitted_at
    assert model._fitted_estimator is preserved_estimator

    external = IsolationForest(random_state=3)
    before = copy.deepcopy(external.get_params())
    model2 = IsolationForestAnomalyModel(spec=_anomaly_spec(), estimator=external)
    model2.fit(_cluster_frame())
    assert external.get_params() == before
    with pytest.raises((ValueError, TypeError, AttributeError)):
        check_is_fitted(external)


def test_prediction_methods_require_fit_and_shapes() -> None:
    model = IsolationForestAnomalyModel(spec=_anomaly_spec())
    frame = _cluster_frame()
    with pytest.raises(ProcessIntelligenceError):
        model.decision_function(frame)
    with pytest.raises(ProcessIntelligenceError):
        model.score_samples(frame)
    with pytest.raises(ProcessIntelligenceError):
        model.predict(frame)
    with pytest.raises(ProcessIntelligenceError):
        model.detect(frame)

    model.fit(frame)
    decisions = model.decision_function(frame)
    scores = model.score_samples(frame)
    preds = model.predict(frame)
    assert isinstance(decisions, np.ndarray) and decisions.ndim == 1
    assert isinstance(scores, np.ndarray) and scores.ndim == 1
    assert isinstance(preds, np.ndarray) and preds.ndim == 1
    assert len(decisions) == frame.height
    assert len(scores) == frame.height
    assert len(preds) == frame.height
    assert np.isfinite(decisions).all()
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(scores, -decisions)
    assert set(np.unique(preds).tolist()).issubset({-1, 1})


@pytest.mark.parametrize("as_pandas", [False, True])
def test_prediction_schema_and_immutability(as_pandas: bool) -> None:
    model = IsolationForestAnomalyModel(spec=_anomaly_spec())
    frame = _cluster_frame(as_pandas=as_pandas)
    snapshot = frame.copy() if as_pandas else frame.clone()
    model.fit(frame)

    model.predict(frame)
    model.score_samples(frame)
    if as_pandas:
        pd.testing.assert_frame_equal(frame, snapshot)
    else:
        assert frame.equals(snapshot)

    reordered = (
        frame.select(["f2", "f1"]) if not as_pandas else frame.loc[:, ["f2", "f1"]]
    )
    with pytest.raises(DataValidationError):
        model.predict(reordered)
    missing = frame.select(["f1"]) if not as_pandas else frame.loc[:, ["f1"]]
    with pytest.raises(DataValidationError):
        model.predict(missing)
    if as_pandas:
        extra = frame.copy()
        extra["f3"] = 0.0
    else:
        extra = frame.with_columns(pl.lit(0.0).alias("f3"))
    with pytest.raises(DataValidationError):
        model.predict(extra)

    if as_pandas:
        dirty = frame.copy()
        dirty.iloc[0, 0] = float("nan")
    else:
        dirty = frame.with_columns(
            pl.when(pl.int_range(0, pl.len()) == 0)
            .then(None)
            .otherwise(pl.col("f1"))
            .alias("f1")
        )
    with pytest.raises(DataValidationError):
        model.predict(dirty)

    first_scores = model.score_samples(frame).copy()
    second_scores = model.score_samples(frame)
    np.testing.assert_array_equal(first_scores, second_scores)
    first_preds = model.predict(frame).copy()
    second_preds = model.predict(frame)
    np.testing.assert_array_equal(first_preds, second_preds)


def test_empty_prediction_inputs() -> None:
    model = IsolationForestAnomalyModel(spec=_anomaly_spec())
    frame = _cluster_frame()
    model.fit(frame)
    empty = _empty_like(frame)
    decisions = model.decision_function(empty)
    scores = model.score_samples(empty)
    preds = model.predict(empty)
    result = model.detect(empty)
    assert decisions.shape == (0,)
    assert scores.shape == (0,)
    assert preds.shape == (0,)
    assert result.row_count == 0
    assert result.scores == []
    assert result.anomaly_fraction == 0.0
    assert result.score_min is None

    wrong_empty = pl.DataFrame(schema={"a": pl.Float64, "b": pl.Float64})
    with pytest.raises(DataValidationError):
        model.predict(wrong_empty)


def test_detect_result_contents_and_outlier_direction() -> None:
    model = IsolationForestAnomalyModel(
        spec=_anomaly_spec(),
        config=IsolationForestConfig(contamination=0.05, random_state=42),
    )
    frame = _cluster_frame(include_outlier=True)
    snapshot = frame.clone()
    model.fit(frame)
    result = model.detect(frame)
    assert isinstance(result, AnomalyDetectionResult)
    assert result.row_count == frame.height
    assert result.anomaly_count == sum(result.is_anomaly)
    assert math.isclose(
        result.anomaly_fraction,
        result.anomaly_count / result.row_count,
    )
    assert result.threshold == 0.0
    assert result.scores == list(model.score_samples(frame))
    assert result.raw_predictions == [int(v) for v in model.predict(frame).tolist()]
    assert result.is_anomaly == [raw == -1 for raw in result.raw_predictions]
    assert result.score_min == min(result.scores)
    assert result.score_max == max(result.scores)
    assert math.isclose(result.score_mean, sum(result.scores) / result.row_count)
    assert frame.equals(snapshot)

    result.scores.append(999.0)
    again = model.detect(frame)
    assert len(again.scores) == frame.height

    outlier_score = result.scores[-1]
    normal_mean = float(np.mean(result.scores[:-1]))
    assert outlier_score > normal_mean

    high_cont = IsolationForestAnomalyModel(
        spec=_anomaly_spec(),
        config=IsolationForestConfig(contamination=0.2, random_state=42),
    )
    high_cont.fit(frame)
    assert high_cont._fitted_estimator.get_params()["contamination"] == 0.2


def test_metadata_evaluate_explain_and_factory() -> None:
    model = create_isolation_forest_anomaly_model(
        config=IsolationForestConfig(random_state=123, contamination=0.08),
    )
    assert isinstance(model, IsolationForestAnomalyModel)
    assert model.is_fitted is False

    meta = model.get_metadata()
    assert isinstance(meta, ModelMetadata)
    assert isinstance(meta, IsolationForestModelMetadata)
    assert meta.fitted is False
    assert meta.fitted_at is None
    assert meta.training_timestamp is None
    assert meta.model_name == "Isolation Forest"
    assert meta.estimator_key == "isolation_forest"
    assert meta.task is AnalysisTask.UNSUPERVISED_ANOMALY
    assert meta.estimator_class == "IsolationForest"
    assert meta.seed == 123
    assert meta.version == sklearn.__version__
    assert meta.contamination == 0.08
    assert meta.anomaly_score_direction == "higher_is_more_anomalous"
    assert meta.optional_dependencies_used == []

    frame = _cluster_frame()
    model.fit(frame)
    meta2 = model.get_metadata()
    assert meta2.fitted is True
    assert meta2.features == ["f1", "f2"]
    assert meta2.fit_row_count == frame.height
    assert meta2.threshold == 0.0
    assert meta2.fitted_at is not None
    dumped = meta2.model_dump()
    assert "estimator" not in dumped or not isinstance(dumped.get("estimator"), object)
    assert all(key != "training_data" for key in dumped)
    meta2.features.append("hack")
    assert model.get_metadata().features == ["f1", "f2"]

    with pytest.raises(ProcessIntelligenceError, match="Step 7A"):
        model.evaluate(frame, y=pl.Series("y", [0] * frame.height))

    unfitted = create_isolation_forest_anomaly_model()
    with pytest.raises(ProcessIntelligenceError):
        unfitted.explain(frame)
    explanation = model.explain(frame)
    assert isinstance(explanation, ExplanationResult)
    assert explanation.feature_importances == {}
    assert "native" in explanation.method
    assert explanation.notes

    with pytest.raises(TypeError):
        create_isolation_forest_anomaly_model(config="bad")  # type: ignore[arg-type]

    first = create_isolation_forest_anomaly_model(
        config=IsolationForestConfig(random_state=5),
    )
    second = create_isolation_forest_anomaly_model(
        config=IsolationForestConfig(random_state=5),
    )
    assert first is not second
    assert first._estimator_template is not second._estimator_template
    assert first._spec is not second._spec
    assert first._config is not second._config
    assert first._spec.name == "Isolation Forest"
    assert first._spec.estimator_key == "isolation_forest"
    assert first._spec.task is AnalysisTask.UNSUPERVISED_ANOMALY
    assert first._spec.priority == 10
    assert first._spec.optional_dependencies == []
    assert first.is_fitted is False
    assert first.get_metadata().seed == 5


def test_determinism_and_instance_isolation() -> None:
    frame = _cluster_frame()
    snapshot = frame.clone()
    a = create_isolation_forest_anomaly_model(
        config=IsolationForestConfig(random_state=42),
    )
    b = create_isolation_forest_anomaly_model(
        config=IsolationForestConfig(random_state=42),
    )
    a.fit(frame)
    b.fit(frame)
    np.testing.assert_allclose(a.score_samples(frame), b.score_samples(frame))
    np.testing.assert_array_equal(a.predict(frame), b.predict(frame))
    a.detect(frame)
    b.detect(frame)
    assert frame.equals(snapshot)
    assert a._fitted_estimator is not b._fitted_estimator
