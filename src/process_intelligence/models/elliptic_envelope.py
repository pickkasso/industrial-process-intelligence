"""Elliptic Envelope unsupervised anomaly detection (Step 7C)."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, field_validator
from sklearn.base import is_classifier, is_regressor  # type: ignore[import-untyped]
from sklearn.covariance import EllipticEnvelope  # type: ignore[import-untyped]

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.exceptions import DataValidationError, InsufficientDataError
from process_intelligence.core.schemas import ModelSpec
from process_intelligence.models.sklearn_outlier_adapter import (
    SklearnOutlierDetectorAdapter,
)

_FACTORY_TIME_BUDGET_SECONDS = 15.0


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


class EllipticEnvelopeConfig(BaseModel):
    """Configuration for sklearn Elliptic Envelope anomaly detection."""

    store_precision: bool = True
    assume_centered: bool = False
    support_fraction: float | None = None
    contamination: float = 0.05
    random_state: int = 42

    @field_validator("store_precision", mode="before")
    @classmethod
    def _validate_store_precision(cls, value: object) -> bool:
        if not isinstance(value, bool):
            raise ValueError(
                f"store_precision must be a bool, got {type(value).__name__}"
            )
        return value

    @field_validator("assume_centered", mode="before")
    @classmethod
    def _validate_assume_centered(cls, value: object) -> bool:
        if not isinstance(value, bool):
            raise ValueError(
                f"assume_centered must be a bool, got {type(value).__name__}"
            )
        return value

    @field_validator("support_fraction", mode="before")
    @classmethod
    def _validate_support_fraction(cls, value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "support_fraction must be None or a float in (0.0, 1.0], "
                f"got {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(
                f"support_fraction must be a finite float, got {value!r}"
            )
        if number <= 0.0 or number > 1.0:
            raise ValueError(
                f"support_fraction must be in (0.0, 1.0], got {number}"
            )
        return number

    @field_validator("contamination", mode="before")
    @classmethod
    def _validate_contamination(cls, value: object) -> float:
        number = _require_finite_float(value, field_name="contamination")
        if number <= 0.0 or number > 0.5:
            raise ValueError(
                f"contamination must be in (0.0, 0.5], got {number}"
            )
        return number

    @field_validator("random_state", mode="before")
    @classmethod
    def _validate_random_state(cls, value: object) -> int:
        number = _require_real_int(value, field_name="random_state")
        if number < 0:
            raise ValueError(f"random_state must be >= 0, got {number}")
        return number


def _build_elliptic_envelope(config: EllipticEnvelopeConfig) -> EllipticEnvelope:
    """Create a fresh EllipticEnvelope from ``config`` values."""
    return EllipticEnvelope(
        store_precision=config.store_precision,
        assume_centered=config.assume_centered,
        support_fraction=config.support_fraction,
        contamination=config.contamination,
        random_state=config.random_state,
    )


def _require_elliptic_envelope_estimator(estimator: object) -> Any:
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


class EllipticEnvelopeAnomalyModel(SklearnOutlierDetectorAdapter):
    """Elliptic Envelope concrete model implementing ``BaseAnomalyModel``."""

    def __init__(
        self,
        *,
        spec: ModelSpec,
        config: EllipticEnvelopeConfig | None = None,
        estimator: Any | None = None,
    ) -> None:
        """Create an Elliptic Envelope anomaly model around a cloned estimator.

        Args:
            spec: Model specification with ``task=UNSUPERVISED_ANOMALY``.
            config: Elliptic Envelope hyperparameters. ``None`` uses defaults.
            estimator: Optional sklearn outlier detector template. When
                provided it is cloned and never fitted or mutated. When
                omitted, an ``EllipticEnvelope`` is built from ``config``.

        Raises:
            TypeError: If ``spec``, ``config``, or ``estimator`` is invalid,
                or if the estimator cannot be cloned.
            DataValidationError: If ``spec.task`` is not unsupervised anomaly.
        """
        if not isinstance(spec, ModelSpec):
            raise TypeError(f"spec must be ModelSpec, got {type(spec).__name__}")
        if spec.task is not AnalysisTask.UNSUPERVISED_ANOMALY:
            raise DataValidationError(
                "EllipticEnvelopeAnomalyModel requires "
                "task=UNSUPERVISED_ANOMALY, "
                f"got {spec.task!r}"
            )

        if config is None:
            resolved_config = EllipticEnvelopeConfig()
        elif isinstance(config, EllipticEnvelopeConfig):
            resolved_config = config.model_copy(deep=True)
        else:
            raise TypeError(
                "config must be EllipticEnvelopeConfig or None, "
                f"got {type(config).__name__}"
            )

        if estimator is None:
            template: Any = _build_elliptic_envelope(resolved_config)
        else:
            template = _require_elliptic_envelope_estimator(estimator)

        super().__init__(
            spec=spec,
            estimator=template,
            random_state=resolved_config.random_state,
            metadata_parameters={
                "store_precision": resolved_config.store_precision,
                "assume_centered": resolved_config.assume_centered,
                "support_fraction": resolved_config.support_fraction,
                "contamination": resolved_config.contamination,
                "random_state": resolved_config.random_state,
            },
        )
        self._config: EllipticEnvelopeConfig = resolved_config

    def _validate_estimator(self, estimator: Any) -> Any:
        """Reject supervised estimators while allowing compatible test doubles."""
        return _require_elliptic_envelope_estimator(estimator)

    def _guard_fit(self, *, row_count: int, feature_count: int) -> None:
        """Require enough rows for covariance-based envelope estimation."""
        _ = feature_count
        if row_count < 2:
            raise InsufficientDataError(
                "EllipticEnvelopeAnomalyModel.fit requires at least 2 rows "
                f"for covariance estimation, got {row_count}"
            )


def create_elliptic_envelope_anomaly_model(
    *,
    config: EllipticEnvelopeConfig | None = None,
) -> EllipticEnvelopeAnomalyModel:
    """Create an independent unfitted Elliptic Envelope anomaly model.

    Args:
        config: Optional hyperparameters. ``None`` uses defaults. The input
            config is deep-copied and never mutated.

    Returns:
        A new ``EllipticEnvelopeAnomalyModel`` with a dedicated estimator,
        config, and ``ModelSpec``.

    Raises:
        TypeError: If ``config`` is not ``None`` or ``EllipticEnvelopeConfig``.
    """
    if config is None:
        resolved = EllipticEnvelopeConfig()
    elif isinstance(config, EllipticEnvelopeConfig):
        resolved = config.model_copy(deep=True)
    else:
        raise TypeError(
            "config must be EllipticEnvelopeConfig or None, "
            f"got {type(config).__name__}"
        )

    spec = ModelSpec(
        name="Elliptic Envelope",
        task=AnalysisTask.UNSUPERVISED_ANOMALY,
        estimator_key="elliptic_envelope",
        optional_dependencies=[],
        priority=30,
        time_budget_seconds=_FACTORY_TIME_BUDGET_SECONDS,
    )
    return EllipticEnvelopeAnomalyModel(spec=spec, config=resolved)
