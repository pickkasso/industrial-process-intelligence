"""Public exports for the models package (Step 6B)."""

from process_intelligence.models.classification import (
    SklearnClassifierModel,
    create_dummy_classifier,
    create_logistic_regression,
    create_random_forest_classifier,
)
from process_intelligence.models.default_registry import (
    create_default_supervised_model_registry,
)
from process_intelligence.models.registry import (
    ModelCandidateStatus,
    ModelFactory,
    ModelRegistry,
)
from process_intelligence.models.regression import (
    SklearnRegressorModel,
    create_dummy_regressor,
    create_linear_regression,
    create_random_forest_regressor,
    create_ridge_regression,
)
from process_intelligence.models.sklearn_adapter import SklearnModelAdapter

__all__ = [
    "ModelCandidateStatus",
    "ModelFactory",
    "ModelRegistry",
    "SklearnClassifierModel",
    "SklearnModelAdapter",
    "SklearnRegressorModel",
    "create_default_supervised_model_registry",
    "create_dummy_classifier",
    "create_dummy_regressor",
    "create_linear_regression",
    "create_logistic_regression",
    "create_random_forest_classifier",
    "create_random_forest_regressor",
    "create_ridge_regression",
]
