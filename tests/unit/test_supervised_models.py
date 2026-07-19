"""Unit tests for supervised sklearn regression and classification models (Step 6B)."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.exceptions import (
    DataValidationError,
    InsufficientDataError,
    ProcessIntelligenceError,
)
from process_intelligence.core.protocols import BaseAnalysisModel
from process_intelligence.core.schemas import ExplanationResult, ModelEvaluation, ModelSpec
from process_intelligence.models import (
    SklearnClassifierModel,
    SklearnModelAdapter,
    SklearnRegressorModel,
    create_dummy_classifier,
    create_dummy_regressor,
    create_linear_regression,
    create_logistic_regression,
    create_random_forest_classifier,
    create_random_forest_regressor,
    create_ridge_regression,
)


def _regression_spec(**overrides: Any) -> ModelSpec:
    payload: dict[str, Any] = {
        "name": "test regressor",
        "task": AnalysisTask.REGRESSION,
        "estimator_key": "test_regressor",
        "optional_dependencies": [],
        "priority": 1,
        "time_budget_seconds": 1.0,
    }
    payload.update(overrides)
    return ModelSpec(**payload)


def _classification_spec(**overrides: Any) -> ModelSpec:
    payload: dict[str, Any] = {
        "name": "test classifier",
        "task": AnalysisTask.CLASSIFICATION,
        "estimator_key": "test_classifier",
        "optional_dependencies": [],
        "priority": 1,
        "time_budget_seconds": 1.0,
    }
    payload.update(overrides)
    return ModelSpec(**payload)


def _numeric_xy(
    *,
    rows: int = 8,
    as_pandas: bool = False,
) -> tuple[Any, Any]:
    x_data = {
        "f1": [float(i) for i in range(rows)],
        "f2": [float(i) * 0.5 for i in range(rows)],
    }
    y_data = [float(i) * 2.0 + 1.0 for i in range(rows)]
    if as_pandas:
        return pd.DataFrame(x_data), pd.Series(y_data)
    return pl.DataFrame(x_data), pl.Series("y", y_data)


def _classification_xy(
    *,
    rows: int = 8,
    labels: list[Any] | None = None,
    as_pandas: bool = False,
) -> tuple[Any, Any]:
    x_data = {
        "f1": [float(i) for i in range(rows)],
        "f2": [float(i % 2) for i in range(rows)],
    }
    if labels is None:
        labels = [0 if i < rows // 2 else 1 for i in range(rows)]
    if as_pandas:
        return pd.DataFrame(x_data), pd.Series(labels)
    return pl.DataFrame(x_data), pl.Series("y", labels)


def test_supervised_models_are_adapter_subclasses() -> None:
    assert issubclass(SklearnRegressorModel, SklearnModelAdapter)
    assert issubclass(SklearnClassifierModel, SklearnModelAdapter)


def test_supervised_models_are_base_analysis_model_instances() -> None:
    regressor = SklearnRegressorModel(
        spec=_regression_spec(),
        estimator=DummyRegressor(strategy="mean"),
    )
    classifier = SklearnClassifierModel(
        spec=_classification_spec(),
        estimator=DummyClassifier(strategy="most_frequent"),
    )
    assert isinstance(regressor, BaseAnalysisModel)
    assert isinstance(classifier, BaseAnalysisModel)


def test_wrong_spec_task_rejected() -> None:
    with pytest.raises(DataValidationError):
        SklearnRegressorModel(
            spec=_regression_spec(task=AnalysisTask.CLASSIFICATION),
            estimator=DummyRegressor(strategy="mean"),
        )
    with pytest.raises(DataValidationError):
        SklearnClassifierModel(
            spec=_classification_spec(task=AnalysisTask.REGRESSION),
            estimator=DummyClassifier(strategy="most_frequent"),
        )


def test_estimator_type_mismatch_rejected() -> None:
    with pytest.raises(TypeError):
        SklearnRegressorModel(
            spec=_regression_spec(),
            estimator=DummyClassifier(strategy="most_frequent"),
        )
    with pytest.raises(TypeError):
        SklearnClassifierModel(
            spec=_classification_spec(),
            estimator=DummyRegressor(strategy="mean"),
        )


def test_created_models_unfitted_and_inputs_immutable() -> None:
    estimator = DummyRegressor(strategy="mean")
    before = copy.deepcopy(estimator.get_params())
    spec = _regression_spec()
    spec_dump = spec.model_dump()
    model = SklearnRegressorModel(spec=spec, estimator=estimator)
    assert model.is_fitted is False
    assert estimator.get_params() == before
    assert spec.model_dump() == spec_dump

    other = SklearnRegressorModel(
        spec=_regression_spec(),
        estimator=DummyRegressor(strategy="mean"),
    )
    x, y = _numeric_xy()
    model.fit(x, y)
    assert other.is_fitted is False


def test_regression_fit_predict_numeric_success() -> None:
    model = create_linear_regression(random_state=7)
    x, y = _numeric_xy()
    model.fit(x, y)
    predictions = model.predict(x)
    assert predictions.shape == (x.height,)
    assert np.isfinite(predictions).all()


@pytest.mark.parametrize(
    "bad_y",
    [
        pl.Series([True, False, True, False, True, False, True, False]),
        pl.Series(["a", "b", "a", "b", "a", "b", "a", "b"]),
        pl.Series([1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0]),
        pl.Series([1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, 7.0, 8.0]),
        pl.Series([1.0, 2.0, float("inf"), 4.0, 5.0, 6.0, 7.0, 8.0]),
    ],
)
def test_regression_fit_rejects_invalid_targets(bad_y: pl.Series) -> None:
    model = create_linear_regression()
    x, _ = _numeric_xy()
    with pytest.raises((DataValidationError, InsufficientDataError, TypeError)):
        model.fit(x, bad_y)


def test_regression_polars_pandas_immutability_and_determinism() -> None:
    x_pl, y_pl = _numeric_xy()
    x_pd, y_pd = _numeric_xy(as_pandas=True)
    before_x = x_pl.to_dicts()
    before_y = y_pl.to_list()

    model_a = create_random_forest_regressor(random_state=11)
    model_b = create_random_forest_regressor(random_state=11)
    model_a.fit(x_pl, y_pl)
    model_b.fit(x_pd, y_pd)
    pred_a = model_a.predict(x_pl)
    pred_b = model_b.predict(x_pd)
    assert np.allclose(pred_a, pred_b)
    assert x_pl.to_dicts() == before_x
    assert y_pl.to_list() == before_y

    model_c = create_random_forest_regressor(random_state=11)
    model_c.fit(x_pl, y_pl)
    assert model_a.is_fitted and model_c.is_fitted
    assert model_a is not model_c


def test_regression_evaluate_contract() -> None:
    model = create_linear_regression()
    x, y = _numeric_xy()
    with pytest.raises(ProcessIntelligenceError):
        model.evaluate(x, y)

    model.fit(x, y)
    before_x = x.to_dicts()
    before_y = y.to_list()
    first = model.evaluate(x, y)
    second = model.evaluate(x, y)
    assert isinstance(first, ModelEvaluation)
    assert "mae" in first.metrics
    assert "rmse" in first.metrics
    assert "r2" in first.metrics
    assert "sample_count" in first.metrics
    assert first.metrics["mae"] == pytest.approx(0.0, abs=1e-8)
    assert first.metrics is not second.metrics
    assert x.to_dicts() == before_x
    assert y.to_list() == before_y


def test_regression_explain_coef_importance_and_fallback() -> None:
    x, y = _numeric_xy()
    linear = create_linear_regression()
    with pytest.raises(ProcessIntelligenceError):
        linear.explain(x)
    linear.fit(x, y)
    linear_exp = linear.explain(x)
    assert isinstance(linear_exp, ExplanationResult)
    assert linear_exp.method == "coef_"
    assert set(linear_exp.feature_importances) == {"f1", "f2"}
    assert list(linear_exp.feature_importances) == sorted(
        linear_exp.feature_importances,
        key=lambda name: (-abs(linear_exp.feature_importances[name]), ("f1", "f2").index(name)),
    )
    assert "estimator" not in linear_exp.model_dump()
    assert not any(isinstance(value, (pl.DataFrame, pd.DataFrame))
                   for value in linear_exp.model_dump().values())

    ridge = create_ridge_regression()
    ridge.fit(x, y)
    assert ridge.explain(x).method == "coef_"

    forest = create_random_forest_regressor(random_state=3)
    forest.fit(x, y)
    forest_exp = forest.explain(x)
    assert forest_exp.method == "feature_importances_"
    values = list(forest_exp.feature_importances.values())
    assert values == sorted(values, reverse=True)

    dummy = create_dummy_regressor()
    dummy.fit(x, y)
    dummy_exp = dummy.explain(x)
    assert dummy_exp.feature_importances == {}
    assert dummy_exp.notes


def test_regression_explain_stable_ties() -> None:
    model = SklearnRegressorModel(
        spec=_regression_spec(),
        estimator=LinearRegression(),
    )
    # Symmetric design so absolute coefficients are equal; original feature order wins.
    x = pl.DataFrame(
        {
            "f1": [1.0, -1.0, 1.0, -1.0],
            "f2": [1.0, -1.0, 1.0, -1.0],
        }
    )
    y = pl.Series([2.0, -2.0, 2.0, -2.0])
    model.fit(x, y)
    explanation = model.explain(x)
    values = list(explanation.feature_importances.values())
    assert abs(abs(values[0]) - abs(values[1])) < 1e-8
    assert list(explanation.feature_importances.keys()) == ["f1", "f2"]


def test_classification_fit_label_types() -> None:
    x, _ = _classification_xy()
    for labels in (
        [0, 0, 0, 0, 1, 1, 1, 1],
        ["a", "a", "a", "a", "b", "b", "b", "b"],
        [True, True, True, True, False, False, False, False],
    ):
        model = create_logistic_regression(random_state=5)
        model.fit(x, pl.Series(labels))
        predictions = model.predict(x)
        assert predictions.shape == (x.height,)
        assert set(map(str, predictions.tolist())) <= set(map(str, labels))


def test_classification_fit_rejects_invalid_targets() -> None:
    x, _ = _classification_xy()
    single_class = create_logistic_regression()
    with pytest.raises(InsufficientDataError):
        single_class.fit(x, pl.Series([0] * x.height))

    with pytest.raises(DataValidationError):
        create_logistic_regression().fit(
            x,
            pl.Series([0, 1, None, 1, 0, 1, 0, 1]),
        )
    with pytest.raises(DataValidationError):
        create_logistic_regression().fit(
            x,
            pl.Series([0.0, 1.0, float("nan"), 1.0, 0.0, 1.0, 0.0, 1.0]),
        )
    with pytest.raises(DataValidationError):
        create_logistic_regression().fit(
            x,
            pl.Series([0.0, 1.0, float("inf"), 1.0, 0.0, 1.0, 0.0, 1.0]),
        )


def test_classification_determinism_and_immutability() -> None:
    x, y = _classification_xy()
    before_x = x.to_dicts()
    before_y = y.to_list()
    a = create_random_forest_classifier(random_state=9)
    b = create_random_forest_classifier(random_state=9)
    a.fit(x, y)
    b.fit(x, y)
    assert np.array_equal(a.predict(x), b.predict(x))
    assert x.to_dicts() == before_x
    assert y.to_list() == before_y


def test_classification_evaluate_contract() -> None:
    model = create_logistic_regression(random_state=2)
    x, y = _classification_xy()
    with pytest.raises(ProcessIntelligenceError):
        model.evaluate(x, y)
    model.fit(x, y)
    first = model.evaluate(x, y)
    second = model.evaluate(x, y)
    assert isinstance(first, ModelEvaluation)
    for key in (
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "class_count",
        "sample_count",
    ):
        assert key in first.metrics
        assert 0.0 <= first.metrics[key] <= 1.0 or key in {"class_count", "sample_count"}
    assert first.metrics is not second.metrics


def test_classification_explain_paths() -> None:
    x, y = _classification_xy()
    logistic = create_logistic_regression(random_state=1)
    with pytest.raises(ProcessIntelligenceError):
        logistic.explain(x)
    logistic.fit(x, y)
    binary = logistic.explain(x)
    assert binary.method == "coef_"
    assert set(binary.feature_importances) == {"f1", "f2"}

    multi_x = pl.DataFrame(
        {
            "f1": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "f2": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        }
    )
    multi_y = pl.Series([0, 1, 2, 0, 1, 2])
    multi = create_logistic_regression(random_state=1)
    multi.fit(multi_x, multi_y)
    multi_exp = multi.explain(multi_x)
    assert multi_exp.method == "multiclass_mean_absolute_coef_"
    assert list(multi_exp.feature_importances.values()) == sorted(
        multi_exp.feature_importances.values(),
        reverse=True,
    )

    forest = create_random_forest_classifier(random_state=4)
    forest.fit(x, y)
    assert forest.explain(x).method == "feature_importances_"

    dummy = create_dummy_classifier()
    dummy.fit(x, y)
    dummy_exp = dummy.explain(x)
    assert dummy_exp.feature_importances == {}
    assert dummy_exp.notes


def test_predict_proba_contract() -> None:
    x, y = _classification_xy()
    logistic = create_logistic_regression(random_state=6)
    with pytest.raises(ProcessIntelligenceError):
        logistic.predict_proba(x)
    logistic.fit(x, y)
    before = x.to_dicts()
    proba = logistic.predict_proba(x)
    assert isinstance(proba, np.ndarray)
    assert proba.ndim == 2
    assert proba.shape[0] == x.height
    assert np.isfinite(proba).all()
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    assert x.to_dicts() == before

    forest = create_random_forest_classifier(random_state=6)
    forest.fit(x, y)
    assert forest.predict_proba(x).ndim == 2

    blocked = create_logistic_regression(random_state=6)
    blocked.fit(x, y)
    # Disable probability support on the fitted estimator for this contract check.
    blocked._fitted_estimator.predict_proba = None  # type: ignore[method-assign]
    with pytest.raises(ProcessIntelligenceError):
        blocked.predict_proba(x)


def test_factory_return_types_and_specs() -> None:
    cases = [
        (create_dummy_regressor, SklearnRegressorModel, "Dummy Regressor",
         AnalysisTask.REGRESSION, "dummy_regressor", 10),
        (create_linear_regression, SklearnRegressorModel, "Linear Regression",
         AnalysisTask.REGRESSION, "linear_regression", 20),
        (create_ridge_regression, SklearnRegressorModel, "Ridge Regression",
         AnalysisTask.REGRESSION, "ridge_regression", 30),
        (create_random_forest_regressor, SklearnRegressorModel, "Random Forest Regressor",
         AnalysisTask.REGRESSION, "random_forest_regressor", 40),
        (create_dummy_classifier, SklearnClassifierModel, "Dummy Classifier",
         AnalysisTask.CLASSIFICATION, "dummy_classifier", 10),
        (create_logistic_regression, SklearnClassifierModel, "Logistic Regression",
         AnalysisTask.CLASSIFICATION, "logistic_regression", 20),
        (create_random_forest_classifier, SklearnClassifierModel,
         "Random Forest Classifier", AnalysisTask.CLASSIFICATION,
         "random_forest_classifier", 40),
    ]
    for factory, model_type, name, task, key, priority in cases:
        model = factory(random_state=42)
        assert isinstance(model, model_type)
        assert model.is_fitted is False
        meta = model.get_metadata()
        assert meta.model_name == name
        assert meta.task is task
        assert model._spec.estimator_key == key
        assert model._spec.priority == priority


def test_factory_random_state_validation_and_independence() -> None:
    with pytest.raises(ValueError):
        create_linear_regression(random_state=-1)
    with pytest.raises(TypeError):
        create_linear_regression(random_state=True)  # type: ignore[arg-type]

    first = create_logistic_regression(random_state=3)
    second = create_logistic_regression(random_state=3)
    assert first is not second
    assert first._spec is not second._spec
    assert first._spec.model_dump() == second._spec.model_dump()
    first._spec.name = "mutated"
    assert second._spec.name == "Logistic Regression"
    assert first.is_fitted is False
    assert second.is_fitted is False
