"""Unit tests for One-Class SVM and Elliptic Envelope models (Step 7C)."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import pytest
from pydantic import ValidationError
from sklearn.covariance import EllipticEnvelope
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.exceptions import (
    DataValidationError,
    InsufficientDataError,
    ProcessIntelligenceError,
)
from process_intelligence.core.protocols import BaseAnomalyModel
from process_intelligence.core.schemas import ExplanationResult, ModelSpec
from process_intelligence.models import (
    AnomalyDetectionResult,
    EllipticEnvelopeAnomalyModel,
    EllipticEnvelopeConfig,
    OneClassSVMAnomalyModel,
    OneClassSVMConfig,
    create_elliptic_envelope_anomaly_model,
    create_one_class_svm_anomaly_model,
)
from process_intelligence.models.sklearn_outlier_adapter import (
    SklearnOutlierModelMetadata,
)


def _ocsvm_spec(**overrides: Any) -> ModelSpec:
    payload: dict[str, Any] = {
        "name": "One-Class SVM",
        "task": AnalysisTask.UNSUPERVISED_ANOMALY,
        "estimator_key": "one_class_svm",
        "optional_dependencies": [],
        "priority": 20,
        "time_budget_seconds": 20.0,
    }
    payload.update(overrides)
    return ModelSpec(**payload)


def _ee_spec(**overrides: Any) -> ModelSpec:
    payload: dict[str, Any] = {
        "name": "Elliptic Envelope",
        "task": AnalysisTask.UNSUPERVISED_ANOMALY,
        "estimator_key": "elliptic_envelope",
        "optional_dependencies": [],
        "priority": 30,
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


def test_one_class_svm_config_defaults_and_validation() -> None:
    config = OneClassSVMConfig()
    assert config.kernel == "rbf"
    assert config.degree == 3
    assert config.gamma == "scale"
    assert config.coef0 == 0.0
    assert config.tol == 0.001
    assert config.nu == 0.05
    assert config.shrinking is True
    assert config.cache_size == 200.0
    assert config.max_iter == -1

    with pytest.raises(ValidationError):
        OneClassSVMConfig(kernel="precomputed")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        OneClassSVMConfig(kernel="bad")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        OneClassSVMConfig(degree=0)
    with pytest.raises(ValidationError):
        OneClassSVMConfig(degree=True)  # type: ignore[arg-type]
    assert OneClassSVMConfig(gamma="auto").gamma == "auto"
    assert OneClassSVMConfig(gamma=0.5).gamma == 0.5
    for bad_gamma in (0.0, float("nan"), float("inf"), True):
        with pytest.raises(ValidationError):
            OneClassSVMConfig(gamma=bad_gamma)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        OneClassSVMConfig(coef0=float("nan"))
    with pytest.raises(ValidationError):
        OneClassSVMConfig(tol=0.0)
    with pytest.raises(ValidationError):
        OneClassSVMConfig(nu=0.0)
    with pytest.raises(ValidationError):
        OneClassSVMConfig(nu=1.1)
    with pytest.raises(ValidationError):
        OneClassSVMConfig(shrinking=1)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        OneClassSVMConfig(cache_size=0.0)
    assert OneClassSVMConfig(max_iter=-1).max_iter == -1
    assert OneClassSVMConfig(max_iter=100).max_iter == 100
    with pytest.raises(ValidationError):
        OneClassSVMConfig(max_iter=0)
    with pytest.raises(ValidationError):
        OneClassSVMConfig(max_iter=-2)

    restored = OneClassSVMConfig.model_validate(config.model_dump())
    assert restored == config


def test_one_class_svm_model_and_factory() -> None:
    config = OneClassSVMConfig(nu=0.1, kernel="rbf")
    snapshot = config.model_copy(deep=True)
    model = OneClassSVMAnomalyModel(spec=_ocsvm_spec(), config=config)
    assert isinstance(model, BaseAnomalyModel)
    assert model.is_fitted is False
    config.nu = 0.9
    assert model._config.nu == snapshot.nu

    with pytest.raises(DataValidationError):
        OneClassSVMAnomalyModel(
            spec=_ocsvm_spec(task=AnalysisTask.REGRESSION),
        )
    with pytest.raises(TypeError):
        OneClassSVMAnomalyModel(
            spec=_ocsvm_spec(),
            estimator=LinearRegression(),
        )

    frame = _cluster_frame()
    snapshot_frame = frame.clone()
    model.fit(frame)
    decisions = model.decision_function(frame)
    scores = model.score_samples(frame)
    preds = model.predict(frame)
    np.testing.assert_allclose(scores, -decisions)
    assert set(np.unique(preds).tolist()).issubset({-1, 1})
    result = model.detect(frame)
    assert isinstance(result, AnomalyDetectionResult)
    assert result.is_anomaly == [raw == -1 for raw in result.raw_predictions]
    assert frame.equals(snapshot_frame)

    meta = model.get_metadata()
    assert isinstance(meta, SklearnOutlierModelMetadata)
    assert meta.parameters["kernel"] == "rbf"
    assert meta.parameters["nu"] == 0.1
    assert meta.anomaly_score_direction == "higher_is_more_anomalous"
    assert meta.seed is None

    a = create_one_class_svm_anomaly_model(config=OneClassSVMConfig(nu=0.08))
    b = create_one_class_svm_anomaly_model(config=OneClassSVMConfig(nu=0.08))
    assert isinstance(a, OneClassSVMAnomalyModel)
    assert a is not b
    assert a._estimator_template is not b._estimator_template
    assert a._spec.name == "One-Class SVM"
    assert a._spec.estimator_key == "one_class_svm"
    assert a._spec.priority == 20
    assert a._spec.optional_dependencies == []
    assert a._spec.time_budget_seconds == 20.0
    a.fit(frame)
    b.fit(frame)
    np.testing.assert_allclose(a.score_samples(frame), b.score_samples(frame))
    np.testing.assert_array_equal(a.predict(frame), b.predict(frame))


def test_elliptic_envelope_config_defaults_and_validation() -> None:
    config = EllipticEnvelopeConfig()
    assert config.store_precision is True
    assert config.assume_centered is False
    assert config.support_fraction is None
    assert config.contamination == 0.05
    assert config.random_state == 42

    with pytest.raises(ValidationError):
        EllipticEnvelopeConfig(store_precision=1)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        EllipticEnvelopeConfig(assume_centered=0)  # type: ignore[arg-type]
    assert EllipticEnvelopeConfig(support_fraction=None).support_fraction is None
    assert EllipticEnvelopeConfig(support_fraction=0.8).support_fraction == 0.8
    for bad in (0.0, 1.1, float("nan"), float("inf"), True):
        with pytest.raises(ValidationError):
            EllipticEnvelopeConfig(support_fraction=bad)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        EllipticEnvelopeConfig(contamination=0.0)
    with pytest.raises(ValidationError):
        EllipticEnvelopeConfig(contamination=0.51)
    with pytest.raises(ValidationError):
        EllipticEnvelopeConfig(random_state=-1)
    with pytest.raises(ValidationError):
        EllipticEnvelopeConfig(random_state=True)  # type: ignore[arg-type]

    restored = EllipticEnvelopeConfig.model_validate(config.model_dump())
    assert restored == config


def test_elliptic_envelope_model_and_factory() -> None:
    external = EllipticEnvelope(random_state=7, contamination=0.1)
    before = copy.deepcopy(external.get_params())
    config = EllipticEnvelopeConfig(random_state=99, contamination=0.1)
    model = EllipticEnvelopeAnomalyModel(
        spec=_ee_spec(),
        config=config,
        estimator=external,
    )
    assert isinstance(model, BaseAnomalyModel)
    assert model._estimator_template.get_params()["random_state"] == 99
    assert external.get_params() == before

    with pytest.raises(DataValidationError):
        EllipticEnvelopeAnomalyModel(
            spec=_ee_spec(task=AnalysisTask.CLASSIFICATION),
        )
    with pytest.raises(TypeError):
        EllipticEnvelopeAnomalyModel(
            spec=_ee_spec(),
            estimator=LinearRegression(),
        )

    with pytest.raises(InsufficientDataError):
        EllipticEnvelopeAnomalyModel(spec=_ee_spec()).fit(
            pl.DataFrame({"f1": [1.0], "f2": [2.0]})
        )

    frame = _cluster_frame()
    snapshot = frame.clone()
    model.fit(frame)
    with pytest.raises((ValueError, TypeError, AttributeError)):
        check_is_fitted(external)

    decisions = model.decision_function(frame)
    scores = model.score_samples(frame)
    preds = model.predict(frame)
    np.testing.assert_allclose(scores, -decisions)
    assert set(np.unique(preds).tolist()).issubset({-1, 1})
    result = model.detect(frame)
    assert isinstance(result, AnomalyDetectionResult)
    assert result.is_anomaly == [raw == -1 for raw in result.raw_predictions]
    assert frame.equals(snapshot)

    meta = model.get_metadata()
    assert meta.parameters["contamination"] == 0.1
    assert meta.parameters["random_state"] == 99
    assert meta.seed == 99

    a = create_elliptic_envelope_anomaly_model(
        config=EllipticEnvelopeConfig(random_state=42),
    )
    b = create_elliptic_envelope_anomaly_model(
        config=EllipticEnvelopeConfig(random_state=42),
    )
    assert isinstance(a, EllipticEnvelopeAnomalyModel)
    assert a is not b
    assert a._spec.name == "Elliptic Envelope"
    assert a._spec.estimator_key == "elliptic_envelope"
    assert a._spec.priority == 30
    assert a._spec.time_budget_seconds == 15.0
    a.fit(frame)
    b.fit(frame)
    np.testing.assert_allclose(a.score_samples(frame), b.score_samples(frame))
    np.testing.assert_array_equal(a.predict(frame), b.predict(frame))
    assert a._fitted_estimator is not b._fitted_estimator


@pytest.mark.parametrize(
    "factory",
    [create_one_class_svm_anomaly_model, create_elliptic_envelope_anomaly_model],
)
def test_shared_contracts_for_additional_models(factory: Any) -> None:
    model = factory()
    frame = _cluster_frame()
    snapshot = frame.clone()
    model.fit(frame)

    with pytest.raises(DataValidationError):
        model.predict(frame.select(["f2", "f1"]))
    with pytest.raises(DataValidationError, match="_original_row_id"):
        model.fit(pl.DataFrame({"_original_row_id": [1.0, 2.0], "f1": [1.0, 2.0]}))
    with pytest.raises(DataValidationError, match="f1"):
        model.predict(
            frame.with_columns(
                pl.when(pl.int_range(0, pl.len()) == 0)
                .then(None)
                .otherwise(pl.col("f1"))
                .alias("f1")
            )
        )

    empty = _empty_like(frame)
    empty_result = model.detect(empty)
    assert empty_result.row_count == 0
    assert empty_result.score_min is None
    assert frame.equals(snapshot)

    other = factory()
    other.fit(frame)
    assert model._fitted_estimator is not other._fitted_estimator

    with pytest.raises(ProcessIntelligenceError):
        model.evaluate(frame)
    explanation = model.explain(frame)
    assert isinstance(explanation, ExplanationResult)
    assert explanation.feature_importances == {}
    dumped = model.get_metadata().model_dump()
    assert "training_data" not in dumped
    assert all(
        "estimator" not in key or key in {"estimator_key", "estimator_class"}
        for key in dumped
    )
