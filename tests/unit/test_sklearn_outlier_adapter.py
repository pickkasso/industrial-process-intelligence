"""Unit tests for SklearnOutlierDetectorAdapter (Step 7C)."""

from __future__ import annotations

import copy
from datetime import UTC, date
from typing import Any, Self

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import pytest
import sklearn
from sklearn.utils.validation import check_is_fitted

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.exceptions import (
    DataValidationError,
    InsufficientDataError,
    ProcessIntelligenceError,
)
from process_intelligence.core.protocols import BaseAnomalyModel
from process_intelligence.core.schemas import ExplanationResult, ModelMetadata, ModelSpec
from process_intelligence.models.anomaly import AnomalyDetectionResult
from process_intelligence.models.sklearn_outlier_adapter import (
    SklearnOutlierDetectorAdapter,
    SklearnOutlierModelMetadata,
)


def _anomaly_spec(**overrides: Any) -> ModelSpec:
    payload: dict[str, Any] = {
        "name": "Test Outlier",
        "task": AnalysisTask.UNSUPERVISED_ANOMALY,
        "estimator_key": "test_outlier",
        "optional_dependencies": [],
        "priority": 50,
        "time_budget_seconds": 10.0,
    }
    payload.update(overrides)
    return ModelSpec(**payload)


def _cluster_frame(
    *,
    rows: int = 40,
    as_pandas: bool = False,
    include_outlier: bool = True,
) -> Any:
    rng = np.random.default_rng(0)
    normal = rng.normal(loc=0.0, scale=1.0, size=(rows, 2))
    if include_outlier:
        normal[-1] = np.array([25.0, -25.0])
    data = {"f1": normal[:, 0].tolist(), "f2": normal[:, 1].tolist()}
    if as_pandas:
        return pd.DataFrame(data)
    return pl.DataFrame(data)


def _empty_like(frame: Any) -> Any:
    if isinstance(frame, pl.DataFrame):
        return frame.clear()
    return frame.iloc[0:0].copy()


class _CloneableOutlier:
    """Minimal cloneable outlier detector used by adapter tests."""

    def __init__(
        self,
        *,
        random_state: int | None = None,
        fail_fit: bool = False,
    ) -> None:
        self.random_state = random_state
        self.fail_fit = fail_fit
        self._fitted = False

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        _ = deep
        return {"random_state": self.random_state, "fail_fit": self.fail_fit}

    def set_params(self, **params: Any) -> Self:
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X: Any, y: Any = None) -> Self:
        _ = y
        if self.fail_fit:
            raise ValueError("intentional fit failure")
        self._center = np.asarray(X, dtype=np.float64).mean(axis=0)
        self._fitted = True
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=np.float64)
        distances = np.linalg.norm(matrix - self._center, axis=1)
        return 1.0 - distances

    def predict(self, X: Any) -> np.ndarray:
        decisions = self.decision_function(X)
        return np.where(decisions >= 0.0, 1, -1).astype(np.int64)

    def score_samples(self, X: Any) -> np.ndarray:
        return -self.decision_function(X)


class _NoRandomStateOutlier(_CloneableOutlier):
    def __init__(self, *, fail_fit: bool = False) -> None:
        super().__init__(random_state=None, fail_fit=fail_fit)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        _ = deep
        return {"fail_fit": self.fail_fit}


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


class ConcreteOutlierAdapter(SklearnOutlierDetectorAdapter):
    """Concrete test adapter used to exercise the abstract base."""

    def _validate_estimator(self, estimator: Any) -> Any:
        return estimator


def test_adapter_is_base_anomaly_and_abstract() -> None:
    assert issubclass(SklearnOutlierDetectorAdapter, BaseAnomalyModel)
    with pytest.raises(TypeError):
        SklearnOutlierDetectorAdapter(  # type: ignore[abstract]
            spec=_anomaly_spec(),
            estimator=_CloneableOutlier(),
        )


def test_concrete_adapter_construction_and_state() -> None:
    model = ConcreteOutlierAdapter(
        spec=_anomaly_spec(),
        estimator=_CloneableOutlier(random_state=1),
        random_state=7,
    )
    assert model.is_fitted is False
    assert model.feature_names == ()
    assert model.fit_row_count == 0
    assert model.threshold is None
    assert model.fitted_at is None


def test_constructor_rejects_invalid_spec_and_estimator() -> None:
    with pytest.raises(TypeError):
        ConcreteOutlierAdapter(spec="bad", estimator=_CloneableOutlier())  # type: ignore[arg-type]
    with pytest.raises(DataValidationError):
        ConcreteOutlierAdapter(
            spec=_anomaly_spec(task=AnalysisTask.REGRESSION),
            estimator=_CloneableOutlier(),
        )
    with pytest.raises(TypeError):
        ConcreteOutlierAdapter(spec=_anomaly_spec(), estimator=_MissingFit())
    with pytest.raises(TypeError):
        ConcreteOutlierAdapter(spec=_anomaly_spec(), estimator=_MissingPredict())
    with pytest.raises(TypeError):
        ConcreteOutlierAdapter(
            spec=_anomaly_spec(),
            estimator=_MissingDecisionFunction(),
        )
    with pytest.raises(TypeError):
        ConcreteOutlierAdapter(
            spec=_anomaly_spec(),
            estimator=_MissingScoreSamples(),
        )
    with pytest.raises(TypeError):
        ConcreteOutlierAdapter(spec=_anomaly_spec(), estimator=_NotCloneable())
    with pytest.raises(ValueError):
        ConcreteOutlierAdapter(
            spec=_anomaly_spec(),
            estimator=_CloneableOutlier(),
            random_state=-1,
        )
    with pytest.raises(TypeError):
        ConcreteOutlierAdapter(
            spec=_anomaly_spec(),
            estimator=_CloneableOutlier(),
            random_state=True,  # type: ignore[arg-type]
        )


def test_clone_random_state_and_metadata_independence() -> None:
    external = _CloneableOutlier(random_state=3)
    before = copy.deepcopy(external.get_params())
    meta_params = {"nu": 0.1, "nested": {"a": 1}}
    model = ConcreteOutlierAdapter(
        spec=_anomaly_spec(),
        estimator=external,
        random_state=99,
        metadata_parameters=meta_params,
    )
    assert external.get_params() == before
    assert model._estimator_template.get_params()["random_state"] == 99
    meta_params["nu"] = 0.9
    meta_params["nested"]["a"] = 2
    assert model.get_metadata().parameters["nu"] == 0.1
    assert model.get_metadata().parameters["nested"]["a"] == 1

    no_rs = ConcreteOutlierAdapter(
        spec=_anomaly_spec(),
        estimator=_NoRandomStateOutlier(),
        random_state=5,
    )
    assert "random_state" not in no_rs._estimator_template.get_params()

    other = ConcreteOutlierAdapter(
        spec=_anomaly_spec(),
        estimator=_CloneableOutlier(random_state=1),
        random_state=1,
    )
    frame = _cluster_frame()
    model.fit(frame)
    other.fit(frame)
    assert model._fitted_estimator is not other._fitted_estimator


@pytest.mark.parametrize("as_pandas", [False, True])
def test_fit_accepts_polars_and_pandas(as_pandas: bool) -> None:
    model = ConcreteOutlierAdapter(
        spec=_anomaly_spec(),
        estimator=_CloneableOutlier(random_state=0),
    )
    frame = _cluster_frame(as_pandas=as_pandas)
    snapshot = frame.copy() if as_pandas else frame.clone()
    fitted = model.fit(frame)
    assert fitted is model
    assert model.is_fitted is True
    if as_pandas:
        pd.testing.assert_frame_equal(frame, snapshot)
    else:
        assert frame.equals(snapshot)


def test_x_validation_errors() -> None:
    model = ConcreteOutlierAdapter(
        spec=_anomaly_spec(),
        estimator=_CloneableOutlier(random_state=0),
    )
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
        model.fit(pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["f1", "f1"]))
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

    ConcreteOutlierAdapter(
        spec=_anomaly_spec(),
        estimator=_CloneableOutlier(random_state=0),
    ).fit(pl.DataFrame({"f1": [1, 2, 3, 4], "f2": [5, 6, 7, 8]}))
    ConcreteOutlierAdapter(
        spec=_anomaly_spec(),
        estimator=_CloneableOutlier(random_state=0),
    ).fit(pl.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [0.5, 0.6, 0.7, 0.8]}))


def test_fit_state_y_rejection_and_atomicity() -> None:
    model = ConcreteOutlierAdapter(
        spec=_anomaly_spec(),
        estimator=_CloneableOutlier(random_state=0),
    )
    frame = _cluster_frame()
    model.fit(frame)
    assert model.is_fitted is True
    assert model.feature_names == ("f1", "f2")
    assert model.fit_row_count == frame.height
    assert model.threshold == 0.0
    assert model.fitted_at is not None
    assert model.fitted_at.tzinfo == UTC
    with pytest.raises(DataValidationError):
        model.fit(frame, y=pl.Series("y", [0.0] * frame.height))

    failing = ConcreteOutlierAdapter(
        spec=_anomaly_spec(),
        estimator=_CloneableOutlier(random_state=0, fail_fit=True),
    )
    with pytest.raises(ValueError, match="intentional fit failure"):
        failing.fit(frame)
    assert failing.is_fitted is False
    assert failing.feature_names == ()
    assert failing.fit_row_count == 0

    model.fit(pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.1, 0.2, 0.3]}))
    preserved_names = model.feature_names
    preserved_count = model.fit_row_count
    preserved_fitted_at = model.fitted_at
    preserved_estimator = model._fitted_estimator
    model._estimator_template = _CloneableOutlier(random_state=1, fail_fit=True)
    with pytest.raises(ValueError, match="intentional fit failure"):
        model.fit(frame)
    assert model.is_fitted is True
    assert model.feature_names == preserved_names
    assert model.fit_row_count == preserved_count
    assert model.fitted_at == preserved_fitted_at
    assert model._fitted_estimator is preserved_estimator

    external = _CloneableOutlier(random_state=3)
    before = copy.deepcopy(external.get_params())
    model2 = ConcreteOutlierAdapter(spec=_anomaly_spec(), estimator=external)
    model2.fit(frame)
    assert external.get_params() == before
    with pytest.raises((ValueError, TypeError, AttributeError)):
        check_is_fitted(external)


def test_prediction_detect_schema_and_immutability() -> None:
    model = ConcreteOutlierAdapter(
        spec=_anomaly_spec(),
        estimator=_CloneableOutlier(random_state=0),
    )
    frame = _cluster_frame()
    with pytest.raises(ProcessIntelligenceError):
        model.decision_function(frame)
    with pytest.raises(ProcessIntelligenceError):
        model.score_samples(frame)
    with pytest.raises(ProcessIntelligenceError):
        model.predict(frame)
    with pytest.raises(ProcessIntelligenceError):
        model.detect(frame)

    snapshot = frame.clone()
    model.fit(frame)
    decisions = model.decision_function(frame)
    scores = model.score_samples(frame)
    preds = model.predict(frame)
    assert decisions.ndim == 1
    assert scores.ndim == 1
    assert preds.ndim == 1
    assert len(decisions) == frame.height
    np.testing.assert_allclose(scores, -decisions)
    assert set(np.unique(preds).tolist()).issubset({-1, 1})

    with pytest.raises(DataValidationError):
        model.predict(frame.select(["f2", "f1"]))
    with pytest.raises(DataValidationError):
        model.predict(frame.select(["f1"]))
    with pytest.raises(DataValidationError):
        model.predict(frame.with_columns(pl.lit(0.0).alias("f3")))

    empty = _empty_like(frame)
    assert model.decision_function(empty).shape == (0,)
    assert model.score_samples(empty).shape == (0,)
    assert model.predict(empty).shape == (0,)
    empty_result = model.detect(empty)
    assert empty_result.row_count == 0
    assert empty_result.score_min is None

    result = model.detect(frame)
    assert isinstance(result, AnomalyDetectionResult)
    assert result.threshold == 0.0
    assert result.is_anomaly == [raw == -1 for raw in result.raw_predictions]
    assert result.anomaly_count == sum(result.is_anomaly)
    assert frame.equals(snapshot)
    result.scores.append(999.0)
    again = model.detect(frame)
    assert len(again.scores) == frame.height


def test_metadata_evaluate_explain() -> None:
    model = ConcreteOutlierAdapter(
        spec=_anomaly_spec(),
        estimator=_CloneableOutlier(random_state=0),
        random_state=11,
        metadata_parameters={"alpha": 0.2},
    )
    meta = model.get_metadata()
    assert isinstance(meta, ModelMetadata)
    assert isinstance(meta, SklearnOutlierModelMetadata)
    assert meta.fitted is False
    assert meta.seed == 11
    assert meta.parameters["alpha"] == 0.2
    assert meta.version == sklearn.__version__
    assert meta.anomaly_score_direction == "higher_is_more_anomalous"

    frame = _cluster_frame()
    model.fit(frame)
    meta2 = model.get_metadata()
    assert meta2.fitted is True
    assert meta2.features == ["f1", "f2"]
    assert meta2.threshold == 0.0
    dumped = meta2.model_dump()
    assert "training_data" not in dumped
    assert not any(
        isinstance(value, _CloneableOutlier) for value in dumped.values()
    )
    meta2.features.append("hack")
    assert model.get_metadata().features == ["f1", "f2"]

    with pytest.raises(ProcessIntelligenceError, match="Step 7C"):
        model.evaluate(frame)
    try:
        model.evaluate(frame)
    except NotImplementedError:
        pytest.fail("evaluate must not raise NotImplementedError")
    except ProcessIntelligenceError:
        pass

    unfitted = ConcreteOutlierAdapter(
        spec=_anomaly_spec(),
        estimator=_CloneableOutlier(random_state=0),
    )
    with pytest.raises(ProcessIntelligenceError):
        unfitted.explain(frame)
    explanation = model.explain(frame)
    assert isinstance(explanation, ExplanationResult)
    assert explanation.feature_importances == {}
    assert "native" in explanation.method
    assert explanation.notes
