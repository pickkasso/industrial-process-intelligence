"""One-Class SVM unsupervised anomaly detection (Step 7C)."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, field_validator
from sklearn.base import is_classifier, is_regressor  # type: ignore[import-untyped]
from sklearn.svm import OneClassSVM  # type: ignore[import-untyped]

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.exceptions import DataValidationError
from process_intelligence.core.schemas import ModelSpec
from process_intelligence.models.sklearn_outlier_adapter import (
    SklearnOutlierDetectorAdapter,
)

_FACTORY_TIME_BUDGET_SECONDS = 20.0


def _require_real_int(value: object, *, field_name: str) -> int:
    """Validate a non-bool integer value."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{field_name} must be an int (bool not allowed), "
            f"got {type(value).__name__}"
        )
    return value


def _require_finite_float(value: object, *, field_name: str) -> float:
    """Validate a non-bool finite float value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a finite float (bool not allowed), "
            f"got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be a finite float, got {value!r}")
    return number


class OneClassSVMConfig(BaseModel):
    """Configuration for sklearn One-Class SVM anomaly detection."""

    kernel: Literal["linear", "poly", "rbf", "sigmoid"] = "rbf"
    degree: int = 3
    gamma: Literal["scale", "auto"] | float = "scale"
    coef0: float = 0.0
    tol: float = 0.001
    nu: float = 0.05
    shrinking: bool = True
    cache_size: float = 200.0
    max_iter: int = -1

    @field_validator("kernel", mode="before")
    @classmethod
    def _validate_kernel(cls, value: object) -> str:
        allowed = {"linear", "poly", "rbf", "sigmoid"}
        if not isinstance(value, str) or value not in allowed:
            raise ValueError(
                "kernel must be one of "
                f"{sorted(allowed)}, got {value!r}"
            )
        return value

    @field_validator("degree", mode="before")
    @classmethod
    def _validate_degree(cls, value: object) -> int:
        number = _require_real_int(value, field_name="degree")
        if number < 1:
            raise ValueError(f"degree must be >= 1, got {number}")
        return number

    @field_validator("gamma", mode="before")
    @classmethod
    def _validate_gamma(cls, value: object) -> str | float:
        if isinstance(value, str):
            if value not in {"scale", "auto"}:
                raise ValueError(
                    "gamma string must be 'scale' or 'auto', "
                    f"got {value!r}"
                )
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "gamma must be 'scale', 'auto', or a float > 0, "
                f"got {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"gamma must be a finite float, got {value!r}")
        if number <= 0.0:
            raise ValueError(f"gamma float must be > 0, got {number}")
        return number

    @field_validator("coef0", mode="before")
    @classmethod
    def _validate_coef0(cls, value: object) -> float:
        return _require_finite_float(value, field_name="coef0")

    @field_validator("tol", mode="before")
    @classmethod
    def _validate_tol(cls, value: object) -> float:
        number = _require_finite_float(value, field_name="tol")
        if number <= 0.0:
            raise ValueError(f"tol must be > 0, got {number}")
        return number

    @field_validator("nu", mode="before")
    @classmethod
    def _validate_nu(cls, value: object) -> float:
        number = _require_finite_float(value, field_name="nu")
        if number <= 0.0 or number > 1.0:
            raise ValueError(f"nu must be in (0.0, 1.0], got {number}")
        return number

    @field_validator("shrinking", mode="before")
    @classmethod
    def _validate_shrinking(cls, value: object) -> bool:
        if not isinstance(value, bool):
            raise ValueError(
                f"shrinking must be a bool, got {type(value).__name__}"
            )
        return value

    @field_validator("cache_size", mode="before")
    @classmethod
    def _validate_cache_size(cls, value: object) -> float:
        number = _require_finite_float(value, field_name="cache_size")
        if number <= 0.0:
            raise ValueError(f"cache_size must be > 0, got {number}")
        return number

    @field_validator("max_iter", mode="before")
    @classmethod
    def _validate_max_iter(cls, value: object) -> int:
        number = _require_real_int(value, field_name="max_iter")
        if number != -1 and number < 1:
            raise ValueError(
                f"max_iter must be -1 or >= 1, got {number}"
            )
        return number


def _build_one_class_svm(config: OneClassSVMConfig) -> OneClassSVM:
    """Create a fresh OneClassSVM from ``config`` values."""
    return OneClassSVM(
        kernel=config.kernel,
        degree=config.degree,
        gamma=config.gamma,
        coef0=config.coef0,
        tol=config.tol,
        nu=config.nu,
        shrinking=config.shrinking,
        cache_size=config.cache_size,
        max_iter=config.max_iter,
    )


def _require_one_class_svm_estimator(estimator: object) -> Any:
    """Validate an unsupervised outlier estimator duck-type and role."""
    for method_name in ("fit", "predict", "decision_function", "score_samples"):
        method = getattr(estimator, method_name, None)
        if not callable(method):
            raise TypeError(
                f"estimator must have a callable {method_name} method, "
                f"got {type(estimator).__name__}"
            )
    try:
        supervised = bool(is_regressor(estimator) or is_classifier(estimator))
    except AttributeError:
        supervised = False
    if supervised:
        raise TypeError(
            "estimator must be an outlier detector, not a "
            f"regressor/classifier ({type(estimator).__name__})"
        )
    return estimator


class OneClassSVMAnomalyModel(SklearnOutlierDetectorAdapter):
    """One-Class SVM concrete model implementing ``BaseAnomalyModel``."""

    def __init__(
        self,
        *,
        spec: ModelSpec,
        config: OneClassSVMConfig | None = None,
        estimator: Any | None = None,
    ) -> None:
        """Create a One-Class SVM anomaly model around a cloned estimator.

        Args:
            spec: Model specification with ``task=UNSUPERVISED_ANOMALY``.
            config: One-Class SVM hyperparameters. ``None`` uses defaults.
            estimator: Optional sklearn outlier detector template. When
                provided it is cloned and never fitted or mutated. When
                omitted, a ``OneClassSVM`` is built from ``config``.

        Raises:
            TypeError: If ``spec``, ``config``, or ``estimator`` is invalid,
                or if the estimator cannot be cloned.
            DataValidationError: If ``spec.task`` is not unsupervised anomaly.
        """
        if not isinstance(spec, ModelSpec):
            raise TypeError(f"spec must be ModelSpec, got {type(spec).__name__}")
        if spec.task is not AnalysisTask.UNSUPERVISED_ANOMALY:
            raise DataValidationError(
                "OneClassSVMAnomalyModel requires "
                "task=UNSUPERVISED_ANOMALY, "
                f"got {spec.task!r}"
            )

        if config is None:
            resolved_config = OneClassSVMConfig()
        elif isinstance(config, OneClassSVMConfig):
            resolved_config = config.model_copy(deep=True)
        else:
            raise TypeError(
                "config must be OneClassSVMConfig or None, "
                f"got {type(config).__name__}"
            )

        if estimator is None:
            template: Any = _build_one_class_svm(resolved_config)
        else:
            template = _require_one_class_svm_estimator(estimator)

        super().__init__(
            spec=spec,
            estimator=template,
            random_state=None,
            metadata_parameters={
                "kernel": resolved_config.kernel,
                "degree": resolved_config.degree,
                "gamma": resolved_config.gamma,
                "coef0": resolved_config.coef0,
                "tol": resolved_config.tol,
                "nu": resolved_config.nu,
                "shrinking": resolved_config.shrinking,
                "cache_size": resolved_config.cache_size,
                "max_iter": resolved_config.max_iter,
            },
        )
        self._config: OneClassSVMConfig = resolved_config

    def _validate_estimator(self, estimator: Any) -> Any:
        """Reject supervised estimators while allowing compatible test doubles."""
        return _require_one_class_svm_estimator(estimator)


def create_one_class_svm_anomaly_model(
    *,
    config: OneClassSVMConfig | None = None,
) -> OneClassSVMAnomalyModel:
    """Create an independent unfitted One-Class SVM anomaly model.

    Args:
        config: Optional hyperparameters. ``None`` uses defaults. The input
            config is deep-copied and never mutated.

    Returns:
        A new ``OneClassSVMAnomalyModel`` with a dedicated estimator,
        config, and ``ModelSpec``.

    Raises:
        TypeError: If ``config`` is not ``None`` or ``OneClassSVMConfig``.
    """
    if config is None:
        resolved = OneClassSVMConfig()
    elif isinstance(config, OneClassSVMConfig):
        resolved = config.model_copy(deep=True)
    else:
        raise TypeError(
            "config must be OneClassSVMConfig or None, "
            f"got {type(config).__name__}"
        )

    spec = ModelSpec(
        name="One-Class SVM",
        task=AnalysisTask.UNSUPERVISED_ANOMALY,
        estimator_key="one_class_svm",
        optional_dependencies=[],
        priority=20,
        time_budget_seconds=_FACTORY_TIME_BUDGET_SECONDS,
    )
    return OneClassSVMAnomalyModel(spec=spec, config=resolved)
