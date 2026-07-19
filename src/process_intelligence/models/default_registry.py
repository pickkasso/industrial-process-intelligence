"""Default supervised model registry factory (Step 6B)."""

from __future__ import annotations

from collections.abc import Callable

from process_intelligence.core.protocols import BaseAnalysisModel
from process_intelligence.models.classification import (
    build_dummy_classifier_spec,
    build_logistic_regression_spec,
    build_random_forest_classifier_spec,
    create_dummy_classifier,
    create_logistic_regression,
    create_random_forest_classifier,
)
from process_intelligence.models.registry import ModelRegistry
from process_intelligence.models.regression import (
    build_dummy_regressor_spec,
    build_linear_regression_spec,
    build_random_forest_regressor_spec,
    build_ridge_regression_spec,
    create_dummy_regressor,
    create_linear_regression,
    create_random_forest_regressor,
    create_ridge_regression,
)


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


def _factory(
    create: Callable[..., BaseAnalysisModel],
    random_state: int,
) -> Callable[[], BaseAnalysisModel]:
    """Build a zero-argument factory closure with early-bound seed."""

    def _create() -> BaseAnalysisModel:
        return create(random_state=random_state)

    return _create


def create_default_supervised_model_registry(
    *,
    random_state: int = 42,
) -> ModelRegistry:
    """Create an isolated registry populated with default supervised models.

    Args:
        random_state: Non-negative seed forwarded to model factories.

    Returns:
        A new ``ModelRegistry`` instance. Factories are not invoked during
        registration, and no global singleton is created.
    """
    seed = _validate_random_state(random_state)
    registry = ModelRegistry()

    factory_entries: list[tuple[str, Callable[..., BaseAnalysisModel]]] = [
        ("dummy_regressor", create_dummy_regressor),
        ("linear_regression", create_linear_regression),
        ("ridge_regression", create_ridge_regression),
        ("random_forest_regressor", create_random_forest_regressor),
        ("dummy_classifier", create_dummy_classifier),
        ("logistic_regression", create_logistic_regression),
        ("random_forest_classifier", create_random_forest_classifier),
    ]
    for estimator_key, create in factory_entries:
        registry.register_factory(estimator_key, _factory(create, seed))

    for spec_builder in (
        build_dummy_regressor_spec,
        build_linear_regression_spec,
        build_ridge_regression_spec,
        build_random_forest_regressor_spec,
        build_dummy_classifier_spec,
        build_logistic_regression_spec,
        build_random_forest_classifier_spec,
    ):
        registry.register(spec_builder())

    return registry
