"""Unit tests for SklearnModelAdapter (Step 6A)."""

from __future__ import annotations

from datetime import UTC, date
from typing import Any, Self

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import pytest
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.exceptions import (
    DataValidationError,
    InsufficientDataError,
    ProcessIntelligenceError,
)
from process_intelligence.core.protocols import BaseAnalysisModel, DataFrameLike, SeriesLike
from process_intelligence.core.schemas import (
    ExplanationResult,
    ModelEvaluation,
    ModelSpec,
)
from process_intelligence.models import SklearnModelAdapter


class ConcreteSklearnAdapter(SklearnModelAdapter):
    """Minimal concrete adapter for testing shared fit/predict behavior."""

    def evaluate(
        self,
        X: DataFrameLike,
        y: SeriesLike | None = None,
    ) -> ModelEvaluation:
        _ = (X, y)
        return ModelEvaluation(metrics={}, notes=["test"])

    def explain(self, X: DataFrameLike) -> ExplanationResult:
        _ = X
        return ExplanationResult(method="test", feature_importances={})


class _FitPredictOnly:
    """Estimator-like object that is not sklearn-cloneable."""

    def fit(self, X: Any, y: Any = None) -> Self:
        _ = (X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.zeros(len(X))


class _MissingPredict:
    def fit(self, X: Any, y: Any = None) -> Self:
        _ = (X, y)
        return self


class _MissingFit:
    def predict(self, X: Any) -> np.ndarray:
        return np.zeros(len(X))


class _BadShapeEstimator:
    """Cloneable estimator that returns an invalid prediction shape."""

    def __init__(self, random_state: int | None = None) -> None:
        self.random_state = random_state
        self._y_mean = 0.0

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        _ = deep
        return {"random_state": self.random_state}

    def set_params(self, **params: Any) -> Self:
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X: Any, y: Any = None) -> Self:
        _ = X
        self._y_mean = float(np.mean(np.asarray(y)))
        self.is_fitted_ = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        # Flattening yields 2 * n_rows, which must be rejected by the adapter.
        return np.full((len(X), 2), self._y_mean)


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
        raise RuntimeError("fit failed")

    def predict(self, X: Any) -> np.ndarray:
        return np.zeros(len(X))


def _spec(**overrides: Any) -> ModelSpec:
    payload = {
        "name": "dummy",
        "task": AnalysisTask.REGRESSION,
        "estimator_key": "dummy_regressor",
        "optional_dependencies": [],
        "priority": 1,
        "time_budget_seconds": None,
    }
    payload.update(overrides)
    return ModelSpec(**payload)


def _adapter(
    estimator: Any | None = None,
    *,
    random_state: int = 42,
    spec: ModelSpec | None = None,
) -> ConcreteSklearnAdapter:
    return ConcreteSklearnAdapter(
        spec=_spec() if spec is None else spec,
        estimator=DummyRegressor(strategy="mean") if estimator is None else estimator,
        random_state=random_state,
    )


def _polars_xy(
    *,
    rows: int = 5,
    features: tuple[str, ...] = ("f1", "f2"),
) -> tuple[pl.DataFrame, pl.Series]:
    data = {name: [float(i + index) for i in range(rows)] for index, name in enumerate(features)}
    frame = pl.DataFrame(data)
    target = pl.Series("y", [float(i) for i in range(rows)])
    return frame, target


# --- Construction and abstractness ----------------------------------------


def test_adapter_is_base_analysis_model_subclass() -> None:
    assert issubclass(SklearnModelAdapter, BaseAnalysisModel)


def test_abstract_adapter_cannot_instantiate() -> None:
    with pytest.raises(TypeError):
        SklearnModelAdapter(  # type: ignore[abstract]
            spec=_spec(),
            estimator=DummyRegressor(),
        )


def test_concrete_adapter_creation_defaults() -> None:
    adapter = _adapter()
    assert adapter.is_fitted is False
    assert adapter.feature_names == ()
    assert adapter.fit_row_count == 0


def test_rejects_non_model_spec() -> None:
    with pytest.raises(TypeError):
        ConcreteSklearnAdapter(
            spec="bad",  # type: ignore[arg-type]
            estimator=DummyRegressor(),
        )


def test_rejects_estimator_without_fit() -> None:
    with pytest.raises(TypeError):
        ConcreteSklearnAdapter(spec=_spec(), estimator=_MissingFit())


def test_rejects_estimator_without_predict() -> None:
    with pytest.raises(TypeError):
        ConcreteSklearnAdapter(spec=_spec(), estimator=_MissingPredict())


def test_rejects_non_cloneable_estimator() -> None:
    with pytest.raises(TypeError):
        ConcreteSklearnAdapter(spec=_spec(), estimator=_FitPredictOnly())


def test_rejects_negative_random_state() -> None:
    with pytest.raises(ValueError):
        _adapter(random_state=-1)


def test_rejects_bool_random_state() -> None:
    with pytest.raises(TypeError):
        ConcreteSklearnAdapter(
            spec=_spec(),
            estimator=DummyRegressor(),
            random_state=True,  # type: ignore[arg-type]
        )


# --- Estimator clone and random_state -------------------------------------


def test_external_estimator_remains_unfitted_and_independent() -> None:
    external = DummyRegressor(strategy="mean")
    adapter = _adapter(external)
    frame, target = _polars_xy()
    adapter.fit(frame, target)
    with pytest.raises(NotFittedError):
        check_is_fitted(external)
    assert adapter.is_fitted is True


def test_random_state_applied_when_supported() -> None:
    class Seeded:
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
            self.is_fitted_ = True
            return self

        def predict(self, X: Any) -> np.ndarray:
            return np.zeros(len(X))

    external = Seeded(random_state=0)
    adapter = ConcreteSklearnAdapter(
        spec=_spec(),
        estimator=external,
        random_state=99,
    )
    frame, target = _polars_xy()
    adapter.fit(frame, target)
    assert external.random_state == 0
    metadata = adapter.get_metadata()
    assert metadata.seed == 99


def test_estimator_without_random_state_allowed() -> None:
    adapter = _adapter(DummyRegressor(strategy="mean"), random_state=7)
    frame, target = _polars_xy()
    adapter.fit(frame, target)
    assert adapter.is_fitted is True


def test_adapters_do_not_share_estimator_state() -> None:
    shared = DummyRegressor(strategy="mean")
    left = _adapter(shared)
    right = _adapter(shared)
    frame, target = _polars_xy()
    left.fit(frame, target)
    assert left.is_fitted is True
    assert right.is_fitted is False


# --- X validation ---------------------------------------------------------


def test_fit_accepts_polars_and_pandas() -> None:
    adapter = _adapter()
    frame, target = _polars_xy()
    adapter.fit(frame, target)
    assert adapter.is_fitted is True

    pandas_adapter = _adapter()
    pdf = frame.to_pandas()
    pandas_adapter.fit(pdf, target.to_pandas())
    assert pandas_adapter.is_fitted is True


def test_x_rejects_non_dataframe() -> None:
    adapter = _adapter()
    with pytest.raises(TypeError):
        adapter.fit([[1.0, 2.0]], pl.Series("y", [1.0]))  # type: ignore[arg-type]


def test_x_rejects_empty_rows_and_columns() -> None:
    adapter = _adapter()
    with pytest.raises(InsufficientDataError):
        adapter.fit(pl.DataFrame({"f1": []}), pl.Series("y", []))
    with pytest.raises(InsufficientDataError):
        adapter.fit(pl.DataFrame({"f1": [1.0]}).select([]), pl.Series("y", [1.0]))


def test_x_rejects_non_string_column_names() -> None:
    adapter = _adapter()
    frame = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])
    frame.columns = pd.Index([1, 2])
    with pytest.raises(DataValidationError):
        adapter.fit(frame, pd.Series([0.0, 1.0]))


@pytest.mark.parametrize(
    "columns",
    [
        {"": [1.0, 2.0]},
        {"  ": [1.0, 2.0]},
    ],
)
def test_x_rejects_invalid_column_names(columns: dict[Any, list[float]]) -> None:
    adapter = _adapter()
    frame = pl.DataFrame(columns)
    with pytest.raises(DataValidationError):
        adapter.fit(frame, pl.Series("y", [0.0, 1.0]))


def test_x_rejects_duplicate_and_reserved_columns() -> None:
    adapter = _adapter()
    duplicate = pd.DataFrame([[1.0, 3.0], [2.0, 4.0]], columns=["f1", "f1"])
    with pytest.raises(DataValidationError):
        adapter.fit(duplicate, pd.Series([0.0, 1.0]))

    reserved = pl.DataFrame({"_original_row_id": [1, 2], "f1": [1.0, 2.0]})
    with pytest.raises(DataValidationError):
        adapter.fit(reserved, pl.Series("y", [0.0, 1.0]))


@pytest.mark.parametrize(
    "frame",
    [
        pl.DataFrame({"f1": ["a", "b"], "f2": [1.0, 2.0]}),
        pl.DataFrame({"f1": [True, False], "f2": [1.0, 2.0]}),
        pl.DataFrame(
            {
                "f1": [date(2024, 1, 1), date(2024, 1, 2)],
                "f2": [1.0, 2.0],
            }
        ),
    ],
)
def test_x_rejects_forbidden_dtypes(frame: pl.DataFrame) -> None:
    adapter = _adapter()
    with pytest.raises(DataValidationError) as exc_info:
        adapter.fit(frame, pl.Series("y", [0.0, 1.0]))
    assert "f1" in str(exc_info.value)


@pytest.mark.parametrize(
    "values",
    [
        [1.0, None],
        [1.0, float("nan")],
        [1.0, float("inf")],
        [1.0, float("-inf")],
    ],
)
def test_x_rejects_null_nan_infinity(values: list[float | None]) -> None:
    adapter = _adapter()
    frame = pl.DataFrame({"f1": values, "f2": [1.0, 2.0]})
    with pytest.raises(DataValidationError) as exc_info:
        adapter.fit(frame, pl.Series("y", [0.0, 1.0]))
    assert "f1" in str(exc_info.value)


def test_x_allows_int_and_float_features_without_mutating_input() -> None:
    adapter = _adapter()
    frame = pl.DataFrame({"f1": [1, 2, 3], "f2": [1.5, 2.5, 3.5]})
    original = frame.clone()
    target = pl.Series("y", [0.0, 1.0, 2.0])
    adapter.fit(frame, target)
    assert frame.equals(original)
    assert adapter.feature_names == ("f1", "f2")


# --- y validation ---------------------------------------------------------


def test_y_accepts_polars_pandas_numpy_and_sequence() -> None:
    frame, _ = _polars_xy(rows=3)
    for target in (
        pl.Series("y", [0.0, 1.0, 2.0]),
        pd.Series([0.0, 1.0, 2.0]),
        np.asarray([0.0, 1.0, 2.0]),
        [0.0, 1.0, 2.0],
    ):
        adapter = _adapter()
        adapter.fit(frame, target)  # type: ignore[arg-type]
        assert adapter.is_fitted is True


def test_y_rejects_none_str_bytes_and_2d() -> None:
    adapter = _adapter()
    frame, _ = _polars_xy(rows=2)
    with pytest.raises(DataValidationError):
        adapter.fit(frame, None)
    with pytest.raises(TypeError):
        adapter.fit(frame, "ab")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        adapter.fit(frame, b"ab")  # type: ignore[arg-type]
    with pytest.raises(DataValidationError):
        adapter.fit(frame, np.asarray([[0.0], [1.0]]))


def test_y_rejects_length_mismatch_empty_null_nan_inf() -> None:
    adapter = _adapter()
    frame, _ = _polars_xy(rows=2)
    with pytest.raises(DataValidationError):
        adapter.fit(frame, pl.Series("y", [0.0]))
    with pytest.raises(InsufficientDataError):
        adapter.fit(
            pl.DataFrame({"f1": [1.0], "f2": [2.0]}),
            pl.Series("y", []),
        )
    with pytest.raises(DataValidationError):
        adapter.fit(frame, pl.Series("y", [0.0, None]))
    with pytest.raises(DataValidationError):
        adapter.fit(frame, np.asarray([0.0, float("nan")]))
    with pytest.raises(DataValidationError):
        adapter.fit(frame, np.asarray([0.0, float("inf")]))


def test_y_allows_string_classification_targets_without_mutation() -> None:
    adapter = _adapter()
    frame = pl.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.5, 1.5, 2.5]})
    target = pl.Series("y", ["a", "b", "a"])
    original = target.clone()
    # DummyRegressor cannot fit string targets; use a tolerant cloneable estimator.
    class PassThrough:
        def __init__(self, random_state: int | None = None) -> None:
            self.random_state = random_state
            self.classes_: list[str] = []

        def get_params(self, deep: bool = True) -> dict[str, Any]:
            _ = deep
            return {"random_state": self.random_state}

        def set_params(self, **params: Any) -> Self:
            for key, value in params.items():
                setattr(self, key, value)
            return self

        def fit(self, X: Any, y: Any = None) -> Self:
            _ = X
            self.classes_ = list(y)
            self.is_fitted_ = True
            return self

        def predict(self, X: Any) -> np.ndarray:
            return np.asarray([self.classes_[0]] * len(X))

    adapter = ConcreteSklearnAdapter(
        spec=_spec(task=AnalysisTask.CLASSIFICATION),
        estimator=PassThrough(),
        random_state=0,
    )
    adapter.fit(frame, target)
    assert target.equals(original)
    assert adapter.is_fitted is True


# --- fit ------------------------------------------------------------------


def test_fit_success_updates_state_and_metadata() -> None:
    adapter = _adapter()
    frame, target = _polars_xy(rows=4, features=("a", "b"))
    result = adapter.fit(frame, target)
    assert result is adapter
    assert adapter.is_fitted is True
    assert adapter.feature_names == ("a", "b")
    assert isinstance(adapter.feature_names, tuple)
    assert adapter.fit_row_count == 4
    metadata = adapter.get_metadata()
    assert metadata.training_timestamp is not None
    assert metadata.training_timestamp.tzinfo is UTC
    assert metadata.features == ["a", "b"]


def test_refit_replaces_feature_state() -> None:
    adapter = _adapter()
    first_x, first_y = _polars_xy(rows=3, features=("a", "b"))
    adapter.fit(first_x, first_y)
    second_x, second_y = _polars_xy(rows=5, features=("x", "y", "z"))
    adapter.fit(second_x, second_y)
    assert adapter.feature_names == ("x", "y", "z")
    assert adapter.fit_row_count == 5


def test_failed_initial_fit_does_not_mark_fitted() -> None:
    adapter = _adapter(_FailingFitEstimator())
    frame, target = _polars_xy()
    with pytest.raises(RuntimeError):
        adapter.fit(frame, target)
    assert adapter.is_fitted is False
    assert adapter.feature_names == ()
    assert adapter.fit_row_count == 0


def test_failed_refit_preserves_previous_fitted_state() -> None:
    adapter = _adapter()
    frame, target = _polars_xy(rows=3, features=("a", "b"))
    adapter.fit(frame, target)
    with pytest.raises(DataValidationError):
        adapter.fit(
            pl.DataFrame({"a": [1.0, float("nan")], "b": [2.0, 3.0]}),
            pl.Series("y", [0.0, 1.0]),
        )
    assert adapter.is_fitted is True
    assert adapter.feature_names == ("a", "b")
    assert adapter.fit_row_count == 3


def test_fit_does_not_mutate_external_estimator_template() -> None:
    external = DummyRegressor(strategy="mean")
    adapter = _adapter(external)
    frame, target = _polars_xy()
    adapter.fit(frame, target)
    with pytest.raises(NotFittedError):
        check_is_fitted(external)


# --- predict --------------------------------------------------------------


def test_predict_before_fit_raises() -> None:
    adapter = _adapter()
    frame, _ = _polars_xy()
    with pytest.raises(ProcessIntelligenceError):
        adapter.predict(frame)


def test_predict_polars_and_pandas_success() -> None:
    adapter = _adapter()
    frame, target = _polars_xy(rows=4)
    adapter.fit(frame, target)
    pred = adapter.predict(frame)
    assert isinstance(pred, np.ndarray)
    assert pred.ndim == 1
    assert pred.shape == (4,)
    pandas_pred = adapter.predict(frame.to_pandas())
    assert pandas_pred.shape == (4,)


def test_predict_requires_exact_feature_order() -> None:
    adapter = _adapter()
    frame, target = _polars_xy(rows=3, features=("a", "b"))
    adapter.fit(frame, target)
    with pytest.raises(DataValidationError):
        adapter.predict(frame.select(["b", "a"]))
    with pytest.raises(DataValidationError):
        adapter.predict(frame.select(["a"]))
    with pytest.raises(DataValidationError):
        adapter.predict(frame.with_columns(pl.lit(1.0).alias("c")))


def test_predict_rejects_null_nan_and_does_not_mutate_or_accumulate() -> None:
    adapter = _adapter()
    frame, target = _polars_xy(rows=3)
    adapter.fit(frame, target)
    with pytest.raises(DataValidationError):
        adapter.predict(pl.DataFrame({"f1": [1.0, None, 3.0], "f2": [1.0, 2.0, 3.0]}))
    with pytest.raises(DataValidationError):
        adapter.predict(
            pl.DataFrame({"f1": [1.0, float("nan"), 3.0], "f2": [1.0, 2.0, 3.0]})
        )
    original = frame.clone()
    first = adapter.predict(frame).copy()
    second = adapter.predict(frame).copy()
    assert frame.equals(original)
    np.testing.assert_array_equal(first, second)


def test_predict_rejects_bad_estimator_shape() -> None:
    adapter = ConcreteSklearnAdapter(
        spec=_spec(),
        estimator=_BadShapeEstimator(),
        random_state=0,
    )
    frame, target = _polars_xy(rows=3)
    adapter.fit(frame, target)
    with pytest.raises(ProcessIntelligenceError):
        adapter.predict(frame)


# --- metadata -------------------------------------------------------------


def test_metadata_before_and_after_fit() -> None:
    spec = _spec(name="demo", estimator_key="demo_key")
    adapter = _adapter(spec=spec, random_state=11)
    before = adapter.get_metadata()
    assert before.model_name == "demo"
    assert before.task is AnalysisTask.REGRESSION
    assert before.seed == 11
    assert before.version == sklearn.__version__
    assert before.features == []
    assert before.training_timestamp is None

    frame, target = _polars_xy(rows=2, features=("a", "b"))
    adapter.fit(frame, target)
    after = adapter.get_metadata()
    assert after.features == ["a", "b"]
    assert after.training_timestamp is not None
    after.features.append("mutated")
    assert adapter.get_metadata().features == ["a", "b"]
    dumped = after.model_dump()
    assert "estimator" not in dumped
    assert str(frame.to_numpy().tolist()) not in str(dumped)


# --- Immutability and determinism -----------------------------------------


def test_spec_and_instance_isolation_and_determinism() -> None:
    spec = _spec(optional_dependencies=["numpy"])
    left = _adapter(spec=spec, random_state=42)
    right = _adapter(spec=spec, random_state=42)
    spec.optional_dependencies.append("mutated")
    assert left.get_metadata().optional_dependencies_used == ["numpy"]

    frame, target = _polars_xy(rows=6)
    original_x = frame.clone()
    original_y = target.clone()
    left.fit(frame, target)
    right.fit(frame, target)
    assert frame.equals(original_x)
    assert target.equals(original_y)
    np.testing.assert_array_equal(left.predict(frame), right.predict(frame))
    assert left is not right
    assert left.is_fitted is True
    assert right.is_fitted is True
