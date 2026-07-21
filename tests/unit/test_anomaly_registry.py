"""Unit tests for default anomaly model registry (Step 7C)."""

from __future__ import annotations

import copy

import pytest

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.protocols import BaseAnomalyModel
from process_intelligence.models import (
    EllipticEnvelopeAnomalyModel,
    EllipticEnvelopeConfig,
    IsolationForestAnomalyModel,
    IsolationForestConfig,
    ModelRegistry,
    OneClassSVMAnomalyModel,
    OneClassSVMConfig,
    create_default_anomaly_model_registry,
)


def test_create_default_anomaly_model_registry_type_and_validation() -> None:
    registry = create_default_anomaly_model_registry()
    assert isinstance(registry, ModelRegistry)

    with pytest.raises(ValueError):
        create_default_anomaly_model_registry(random_state=-1)
    with pytest.raises(TypeError):
        create_default_anomaly_model_registry(random_state=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        create_default_anomaly_model_registry(
            isolation_forest_config="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        create_default_anomaly_model_registry(
            one_class_svm_config="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        create_default_anomaly_model_registry(
            elliptic_envelope_config="bad",  # type: ignore[arg-type]
        )


def test_candidate_inventory_and_order() -> None:
    registry = create_default_anomaly_model_registry()
    specs = registry.list_specs()
    assert len(specs) == 3
    assert len(registry.list_specs(AnalysisTask.REGRESSION)) == 0

    candidates = registry.get_candidates(AnalysisTask.UNSUPERVISED_ANOMALY)
    assert len(candidates) == 3
    assert [spec.name for spec in candidates] == [
        "Isolation Forest",
        "One-Class SVM",
        "Elliptic Envelope",
    ]
    assert [spec.estimator_key for spec in candidates] == [
        "isolation_forest",
        "one_class_svm",
        "elliptic_envelope",
    ]
    assert [spec.priority for spec in candidates] == [10, 20, 30]
    assert all(spec.optional_dependencies == [] for spec in candidates)

    statuses = registry.inspect_candidates(AnalysisTask.UNSUPERVISED_ANOMALY)
    assert len(statuses) == 3
    assert all(status.factory_registered for status in statuses)
    assert all(status.available for status in statuses)
    assert [status.spec.name for status in statuses] == [
        "Isolation Forest",
        "One-Class SVM",
        "Elliptic Envelope",
    ]


def test_instantiate_independence_and_lazy_factories() -> None:
    call_count = {"n": 0}
    original = create_default_anomaly_model_registry
    registry = original()

    # Registry construction must not instantiate models.
    assert call_count["n"] == 0

    if_spec = registry.get(
        AnalysisTask.UNSUPERVISED_ANOMALY,
        "Isolation Forest",
    )
    ocsvm_spec = registry.get(
        AnalysisTask.UNSUPERVISED_ANOMALY,
        "One-Class SVM",
    )
    ee_spec = registry.get(
        AnalysisTask.UNSUPERVISED_ANOMALY,
        "Elliptic Envelope",
    )

    if_model = registry.instantiate(if_spec)
    ocsvm_model = registry.instantiate(ocsvm_spec)
    ee_model = registry.instantiate(ee_spec)

    assert isinstance(if_model, IsolationForestAnomalyModel)
    assert isinstance(ocsvm_model, OneClassSVMAnomalyModel)
    assert isinstance(ee_model, EllipticEnvelopeAnomalyModel)
    assert isinstance(if_model, BaseAnomalyModel)
    assert isinstance(ocsvm_model, BaseAnomalyModel)
    assert isinstance(ee_model, BaseAnomalyModel)

    assert if_model._spec.model_dump() == if_spec.model_dump()
    assert ocsvm_model._spec.model_dump() == ocsvm_spec.model_dump()
    assert ee_model._spec.model_dump() == ee_spec.model_dump()

    first = registry.instantiate(if_spec)
    second = registry.instantiate(if_spec)
    assert first is not second
    assert first._estimator_template is not second._estimator_template


def test_time_budget_and_config_random_state_behavior() -> None:
    registry = create_default_anomaly_model_registry()
    filtered = registry.get_candidates(
        AnalysisTask.UNSUPERVISED_ANOMALY,
        time_budget_seconds=16.0,
    )
    assert [spec.estimator_key for spec in filtered] == [
        "isolation_forest",
        "elliptic_envelope",
    ]

    if_config = IsolationForestConfig(random_state=7, n_estimators=25)
    ee_config = EllipticEnvelopeConfig(random_state=11, contamination=0.08)
    ocsvm_config = OneClassSVMConfig(nu=0.07)
    if_before = copy.deepcopy(if_config.model_dump())
    ee_before = copy.deepcopy(ee_config.model_dump())
    ocsvm_before = copy.deepcopy(ocsvm_config.model_dump())

    custom = create_default_anomaly_model_registry(
        random_state=99,
        isolation_forest_config=if_config,
        one_class_svm_config=ocsvm_config,
        elliptic_envelope_config=ee_config,
    )
    assert if_config.model_dump() == if_before
    assert ee_config.model_dump() == ee_before
    assert ocsvm_config.model_dump() == ocsvm_before

    if_model = custom.instantiate(
        custom.get(AnalysisTask.UNSUPERVISED_ANOMALY, "Isolation Forest")
    )
    ee_model = custom.instantiate(
        custom.get(AnalysisTask.UNSUPERVISED_ANOMALY, "Elliptic Envelope")
    )
    ocsvm_model = custom.instantiate(
        custom.get(AnalysisTask.UNSUPERVISED_ANOMALY, "One-Class SVM")
    )
    assert isinstance(if_model, IsolationForestAnomalyModel)
    assert isinstance(ee_model, EllipticEnvelopeAnomalyModel)
    assert isinstance(ocsvm_model, OneClassSVMAnomalyModel)
    assert if_model._config.random_state == 7
    assert if_model._config.n_estimators == 25
    assert ee_model._config.random_state == 11
    assert ee_model._config.contamination == 0.08
    assert ocsvm_model._config.nu == 0.07
    assert ocsvm_model.get_metadata().seed is None

    defaulted = create_default_anomaly_model_registry(random_state=55)
    default_if = defaulted.instantiate(
        defaulted.get(AnalysisTask.UNSUPERVISED_ANOMALY, "Isolation Forest")
    )
    default_ee = defaulted.instantiate(
        defaulted.get(AnalysisTask.UNSUPERVISED_ANOMALY, "Elliptic Envelope")
    )
    assert isinstance(default_if, IsolationForestAnomalyModel)
    assert isinstance(default_ee, EllipticEnvelopeAnomalyModel)
    assert default_if._config.random_state == 55
    assert default_ee._config.random_state == 55


def test_registry_isolation_no_singleton() -> None:
    first = create_default_anomaly_model_registry()
    second = create_default_anomaly_model_registry()
    assert first is not second

    removed = first.unregister(
        AnalysisTask.UNSUPERVISED_ANOMALY,
        "Isolation Forest",
    )
    assert removed.estimator_key == "isolation_forest"
    assert len(first.list_specs()) == 2
    assert len(second.list_specs()) == 3
    assert second.contains(
        AnalysisTask.UNSUPERVISED_ANOMALY,
        "Isolation Forest",
    )
