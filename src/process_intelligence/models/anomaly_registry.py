"""Default unsupervised anomaly model registry factory (Step 7C)."""

from __future__ import annotations

from collections.abc import Callable

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.protocols import BaseAnalysisModel
from process_intelligence.core.schemas import ModelSpec
from process_intelligence.models.anomaly import (
    IsolationForestConfig,
    create_isolation_forest_anomaly_model,
)
from process_intelligence.models.elliptic_envelope import (
    EllipticEnvelopeConfig,
    create_elliptic_envelope_anomaly_model,
)
from process_intelligence.models.one_class_svm import (
    OneClassSVMConfig,
    create_one_class_svm_anomaly_model,
)
from process_intelligence.models.registry import ModelRegistry

_ISOLATION_FOREST_TIME_BUDGET_SECONDS = 15.0
_ONE_CLASS_SVM_TIME_BUDGET_SECONDS = 20.0
_ELLIPTIC_ENVELOPE_TIME_BUDGET_SECONDS = 15.0


def _validate_random_state(value: object) -> int:
    """Validate a non-negative integer random seed (bool not allowed)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"random_state must be a non-negative int (bool not allowed), "
            f"got {type(value).__name__}"
        )
    if value < 0:
        raise ValueError(f"random_state must be >= 0, got {value}")
    return value


def _resolve_isolation_forest_config(
    config: IsolationForestConfig | None,
    *,
    random_state: int,
) -> IsolationForestConfig:
    """Resolve an Isolation Forest config snapshot without mutating inputs."""
    if config is None:
        return IsolationForestConfig(random_state=random_state)
    if not isinstance(config, IsolationForestConfig):
        raise TypeError(
            "isolation_forest_config must be IsolationForestConfig or None, "
            f"got {type(config).__name__}"
        )
    return config.model_copy(deep=True)


def _resolve_one_class_svm_config(
    config: OneClassSVMConfig | None,
) -> OneClassSVMConfig:
    """Resolve a One-Class SVM config snapshot without mutating inputs."""
    if config is None:
        return OneClassSVMConfig()
    if not isinstance(config, OneClassSVMConfig):
        raise TypeError(
            "one_class_svm_config must be OneClassSVMConfig or None, "
            f"got {type(config).__name__}"
        )
    return config.model_copy(deep=True)


def _resolve_elliptic_envelope_config(
    config: EllipticEnvelopeConfig | None,
    *,
    random_state: int,
) -> EllipticEnvelopeConfig:
    """Resolve an Elliptic Envelope config snapshot without mutating inputs."""
    if config is None:
        return EllipticEnvelopeConfig(random_state=random_state)
    if not isinstance(config, EllipticEnvelopeConfig):
        raise TypeError(
            "elliptic_envelope_config must be EllipticEnvelopeConfig or None, "
            f"got {type(config).__name__}"
        )
    return config.model_copy(deep=True)


def _isolation_forest_factory(
    config: IsolationForestConfig,
) -> Callable[[], BaseAnalysisModel]:
    """Build a zero-argument Isolation Forest factory with an early-bound config."""

    def _create() -> BaseAnalysisModel:
        return create_isolation_forest_anomaly_model(
            config=config.model_copy(deep=True),
        )

    return _create


def _one_class_svm_factory(
    config: OneClassSVMConfig,
) -> Callable[[], BaseAnalysisModel]:
    """Build a zero-argument One-Class SVM factory with an early-bound config."""

    def _create() -> BaseAnalysisModel:
        return create_one_class_svm_anomaly_model(
            config=config.model_copy(deep=True),
        )

    return _create


def _elliptic_envelope_factory(
    config: EllipticEnvelopeConfig,
) -> Callable[[], BaseAnalysisModel]:
    """Build a zero-argument Elliptic Envelope factory with an early-bound config."""

    def _create() -> BaseAnalysisModel:
        return create_elliptic_envelope_anomaly_model(
            config=config.model_copy(deep=True),
        )

    return _create


def create_default_anomaly_model_registry(
    *,
    random_state: int = 42,
    isolation_forest_config: IsolationForestConfig | None = None,
    one_class_svm_config: OneClassSVMConfig | None = None,
    elliptic_envelope_config: EllipticEnvelopeConfig | None = None,
) -> ModelRegistry:
    """Create an isolated registry with default unsupervised anomaly models.

    Registers Isolation Forest, One-Class SVM, and Elliptic Envelope. Local
    Outlier Factor is intentionally excluded because its novelty scoring
    semantics differ from the shared detect contract.

    Args:
        random_state: Non-negative seed applied to default Isolation Forest and
            Elliptic Envelope configs when those configs are omitted.
        isolation_forest_config: Optional Isolation Forest hyperparameters.
            When provided, its ``random_state`` is preserved.
        one_class_svm_config: Optional One-Class SVM hyperparameters. One-Class
            SVM has no ``random_state`` parameter.
        elliptic_envelope_config: Optional Elliptic Envelope hyperparameters.
            When provided, its ``random_state`` is preserved.

    Returns:
        A new ``ModelRegistry`` instance. Factories are not invoked during
        registration, and no global singleton is created.
    """
    seed = _validate_random_state(random_state)
    if_config = _resolve_isolation_forest_config(
        isolation_forest_config,
        random_state=seed,
    )
    ocsvm_config = _resolve_one_class_svm_config(one_class_svm_config)
    ee_config = _resolve_elliptic_envelope_config(
        elliptic_envelope_config,
        random_state=seed,
    )

    registry = ModelRegistry()

    isolation_spec = ModelSpec(
        name="Isolation Forest",
        task=AnalysisTask.UNSUPERVISED_ANOMALY,
        estimator_key="isolation_forest",
        optional_dependencies=[],
        priority=10,
        time_budget_seconds=_ISOLATION_FOREST_TIME_BUDGET_SECONDS,
    )
    ocsvm_spec = ModelSpec(
        name="One-Class SVM",
        task=AnalysisTask.UNSUPERVISED_ANOMALY,
        estimator_key="one_class_svm",
        optional_dependencies=[],
        priority=20,
        time_budget_seconds=_ONE_CLASS_SVM_TIME_BUDGET_SECONDS,
    )
    elliptic_spec = ModelSpec(
        name="Elliptic Envelope",
        task=AnalysisTask.UNSUPERVISED_ANOMALY,
        estimator_key="elliptic_envelope",
        optional_dependencies=[],
        priority=30,
        time_budget_seconds=_ELLIPTIC_ENVELOPE_TIME_BUDGET_SECONDS,
    )

    registry.register_factory(
        "isolation_forest",
        _isolation_forest_factory(if_config),
    )
    registry.register_factory(
        "one_class_svm",
        _one_class_svm_factory(ocsvm_config),
    )
    registry.register_factory(
        "elliptic_envelope",
        _elliptic_envelope_factory(ee_config),
    )

    registry.register(isolation_spec)
    registry.register(ocsvm_spec)
    registry.register(elliptic_spec)

    return registry
