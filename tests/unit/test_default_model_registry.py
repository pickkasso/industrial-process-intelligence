"""Unit tests for the default supervised model registry (Step 6B)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.protocols import BaseAnalysisModel
from process_intelligence.models import (
    ModelRegistry,
    create_default_supervised_model_registry,
)


def _xy_regression() -> tuple[pl.DataFrame, pl.Series]:
    frame = pl.DataFrame(
        {
            "f1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "f2": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        }
    )
    target = pl.Series("y", [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    return frame, target


def _xy_classification() -> tuple[pl.DataFrame, pl.Series]:
    frame = pl.DataFrame(
        {
            "f1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "f2": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        }
    )
    target = pl.Series("y", [0, 0, 0, 1, 1, 1])
    return frame, target


def test_create_default_registry_type_and_random_state_validation() -> None:
    registry = create_default_supervised_model_registry()
    assert isinstance(registry, ModelRegistry)
    with pytest.raises(ValueError):
        create_default_supervised_model_registry(random_state=-1)
    with pytest.raises(TypeError):
        create_default_supervised_model_registry(random_state=True)  # type: ignore[arg-type]


def test_default_registry_spec_counts_and_order() -> None:
    registry = create_default_supervised_model_registry(random_state=42)
    assert len(registry) == 7

    regression = registry.list_specs(AnalysisTask.REGRESSION)
    classification = registry.list_specs(AnalysisTask.CLASSIFICATION)
    assert len(regression) == 4
    assert len(classification) == 3

    assert [spec.name for spec in regression] == [
        "Dummy Regressor",
        "Linear Regression",
        "Ridge Regression",
        "Random Forest Regressor",
    ]
    assert [spec.name for spec in classification] == [
        "Dummy Classifier",
        "Logistic Regression",
        "Random Forest Classifier",
    ]
    assert [spec.priority for spec in regression] == [10, 20, 30, 40]
    assert [spec.priority for spec in classification] == [10, 20, 40]


def test_default_registry_estimator_keys_and_availability() -> None:
    registry = create_default_supervised_model_registry()
    expected_keys = {
        "dummy_regressor",
        "linear_regression",
        "ridge_regression",
        "random_forest_regressor",
        "dummy_classifier",
        "logistic_regression",
        "random_forest_classifier",
    }
    specs = registry.list_specs()
    assert {spec.estimator_key for spec in specs} == expected_keys
    assert all(spec.optional_dependencies == [] for spec in specs)

    for task in (AnalysisTask.REGRESSION, AnalysisTask.CLASSIFICATION):
        statuses = registry.inspect_candidates(task)
        assert all(status.available for status in statuses)
        assert all(status.factory_registered for status in statuses)


def test_default_registry_get_candidates_counts() -> None:
    registry = create_default_supervised_model_registry()
    assert len(registry.get_candidates(AnalysisTask.REGRESSION)) == 4
    assert len(registry.get_candidates(AnalysisTask.CLASSIFICATION)) == 3


def test_default_registry_instantiate_all_models() -> None:
    registry = create_default_supervised_model_registry(random_state=7)
    names = [
        (AnalysisTask.REGRESSION, "Dummy Regressor"),
        (AnalysisTask.REGRESSION, "Linear Regression"),
        (AnalysisTask.REGRESSION, "Ridge Regression"),
        (AnalysisTask.REGRESSION, "Random Forest Regressor"),
        (AnalysisTask.CLASSIFICATION, "Dummy Classifier"),
        (AnalysisTask.CLASSIFICATION, "Logistic Regression"),
        (AnalysisTask.CLASSIFICATION, "Random Forest Classifier"),
    ]
    models = []
    for task, name in names:
        model = registry.instantiate(registry.get(task, name))
        assert isinstance(model, BaseAnalysisModel)
        models.append(model)

    first = registry.instantiate(registry.get(AnalysisTask.REGRESSION, "Linear Regression"))
    second = registry.instantiate(registry.get(AnalysisTask.REGRESSION, "Linear Regression"))
    assert first is not second
    assert first is not models[1]


def test_default_registry_factories_not_called_during_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from process_intelligence.models import default_registry as module

    calls = {"count": 0}

    def _wrap(create):  # type: ignore[no-untyped-def]
        def _wrapped(*, random_state: int = 42):  # type: ignore[no-untyped-def]
            calls["count"] += 1
            return create(random_state=random_state)

        return _wrapped

    for name in (
        "create_dummy_regressor",
        "create_linear_regression",
        "create_ridge_regression",
        "create_random_forest_regressor",
        "create_dummy_classifier",
        "create_logistic_regression",
        "create_random_forest_classifier",
    ):
        monkeypatch.setattr(module, name, _wrap(getattr(module, name)))

    registry = create_default_supervised_model_registry(random_state=1)
    assert calls["count"] == 0
    registry.instantiate(registry.get(AnalysisTask.REGRESSION, "Dummy Regressor"))
    assert calls["count"] == 1


def test_default_registry_factory_specs_match_registered_specs() -> None:
    registry = create_default_supervised_model_registry(random_state=12)
    for task in (AnalysisTask.REGRESSION, AnalysisTask.CLASSIFICATION):
        for spec in registry.list_specs(task):
            model = registry.instantiate(spec)
            model_spec = model.get_metadata()
            assert model_spec.model_name == spec.name
            assert model_spec.task is spec.task
            assert model._spec.estimator_key == spec.estimator_key
            assert model._spec.priority == spec.priority
            assert model._spec.time_budget_seconds == spec.time_budget_seconds
            assert model._spec.optional_dependencies == spec.optional_dependencies


def test_default_registry_time_budget_filters_random_forest() -> None:
    registry = create_default_supervised_model_registry()
    regression = registry.get_candidates(
        AnalysisTask.REGRESSION,
        time_budget_seconds=2.0,
    )
    names = [spec.name for spec in regression]
    assert "Random Forest Regressor" not in names
    assert "Dummy Regressor" in names
    assert "Linear Regression" in names
    assert "Ridge Regression" in names

    classification = registry.get_candidates(
        AnalysisTask.CLASSIFICATION,
        time_budget_seconds=5.0,
    )
    class_names = [spec.name for spec in classification]
    assert "Random Forest Classifier" not in class_names
    assert "Dummy Classifier" in class_names
    assert "Logistic Regression" in class_names


def test_default_registry_random_state_determinism() -> None:
    registry = create_default_supervised_model_registry(random_state=21)
    x, y = _xy_regression()
    model_a = registry.instantiate(
        registry.get(AnalysisTask.REGRESSION, "Random Forest Regressor")
    )
    model_b = registry.instantiate(
        registry.get(AnalysisTask.REGRESSION, "Random Forest Regressor")
    )
    model_a.fit(x, y)
    model_b.fit(x, y)
    assert np.allclose(model_a.predict(x), model_b.predict(x))
    assert model_a.get_metadata().seed == 21
    assert model_b.get_metadata().seed == 21


def test_default_registry_instance_isolation() -> None:
    first = create_default_supervised_model_registry(random_state=1)
    second = create_default_supervised_model_registry(random_state=1)
    assert first is not second

    removed = first.unregister(AnalysisTask.REGRESSION, "Dummy Regressor")
    assert removed.name == "Dummy Regressor"
    assert first.contains(AnalysisTask.REGRESSION, "Dummy Regressor") is False
    assert second.contains(AnalysisTask.REGRESSION, "Dummy Regressor") is True
    assert len(first) == 6
    assert len(second) == 7


def test_default_registry_classification_fit_smoke() -> None:
    registry = create_default_supervised_model_registry(random_state=3)
    x, y = _xy_classification()
    model = registry.instantiate(
        registry.get(AnalysisTask.CLASSIFICATION, "Logistic Regression")
    )
    model.fit(x, y)
    evaluation = model.evaluate(x, y)
    assert "accuracy" in evaluation.metrics
