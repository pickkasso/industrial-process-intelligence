"""Concrete sklearn regression models and default factories (Step 6B)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
from sklearn.base import is_regressor  # type: ignore[import-untyped]
from sklearn.dummy import DummyRegressor  # type: ignore[import-untyped]
from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]
from sklearn.linear_model import LinearRegression, Ridge  # type: ignore[import-untyped]

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.exceptions import (
    DataValidationError,
    InsufficientDataError,
    ProcessIntelligenceError,
)
from process_intelligence.core.protocols import DataFrameLike, SeriesLike
from process_intelligence.core.schemas import ExplanationResult, ModelEvaluation, ModelSpec
from process_intelligence.evaluation.metrics import evaluate_regression
from process_intelligence.models.sklearn_adapter import SklearnModelAdapter


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


def _is_string_or_bytes(value: object) -> bool:
    """Return True for str or bytes targets."""
    return isinstance(value, (str, bytes))


def _target_to_array(y: object) -> np.ndarray:
    """Convert a supervised target to a 1D NumPy array without mutating input."""
    if y is None:
        raise DataValidationError("y is required for supervised regression models")
    if _is_string_or_bytes(y):
        raise TypeError(f"y must not be str or bytes; got {type(y).__name__}")

    if isinstance(y, pl.Series):
        values = np.array(y.to_numpy(), copy=True)
    elif isinstance(y, pd.Series):
        values = y.to_numpy(copy=True)
    elif isinstance(y, np.ndarray):
        values = np.array(y, copy=True)
    elif isinstance(y, Sequence):
        values = np.asarray(list(y))
    else:
        raise TypeError(
            "y must be a polars.Series, pandas.Series, 1D numpy.ndarray, "
            f"or Sequence, got {type(y).__name__}"
        )

    if values.ndim != 1:
        raise DataValidationError(f"y must be 1-dimensional, got shape {values.shape}")
    return values


def _is_boolean_array(values: np.ndarray) -> bool:
    """Return True when values are boolean."""
    if values.dtype == np.bool_ or values.dtype.kind == "b":
        return True
    if values.dtype.kind == "O" and values.size > 0:
        return all(isinstance(value, (bool, np.bool_)) for value in values.tolist())
    return False


def _validate_regression_target(y: object) -> None:
    """Validate regression-specific target constraints without mutating ``y``."""
    values = _target_to_array(y)
    if values.size < 1:
        raise InsufficientDataError("y must contain at least one value")

    if isinstance(y, pl.Series) and y.null_count() > 0:
        raise DataValidationError("y must not contain null values")
    if isinstance(y, pd.Series) and y.isna().any():
        raise DataValidationError("y must not contain null values")

    if _is_boolean_array(values):
        raise DataValidationError(
            "regression target must be numeric; boolean values are not allowed"
        )
    if values.dtype.kind in {"U", "S"}:
        raise DataValidationError(
            "regression target must be numeric; string values are not allowed"
        )
    if values.dtype.kind == "O":
        if any(value is None for value in values.tolist()):
            raise DataValidationError("y must not contain null values")
        if any(isinstance(value, str) for value in values.tolist()):
            raise DataValidationError(
                "regression target must be numeric; string values are not allowed"
            )
        if all(isinstance(value, (bool, np.bool_)) for value in values.tolist()):
            raise DataValidationError(
                "regression target must be numeric; boolean values are not allowed"
            )
        try:
            numeric = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise DataValidationError("regression target must be numeric") from exc
    elif values.dtype.kind in {"f", "i", "u"}:
        numeric = values.astype(np.float64, copy=False)
    else:
        raise DataValidationError("regression target must be numeric")

    if not np.isfinite(numeric).all():
        raise DataValidationError("y must not contain NaN or infinite values")


def _normalize_zero(value: float) -> float:
    """Normalize signed zero to ``0.0``."""
    return 0.0 if value == 0.0 else value


def _sorted_importances(
    feature_names: tuple[str, ...],
    values: np.ndarray,
    *,
    by_absolute: bool,
) -> dict[str, float]:
    """Build ordered feature importances with stable tie breaking."""
    if values.ndim != 1:
        raise ProcessIntelligenceError(
            f"importance values must be 1-dimensional, got shape {values.shape}"
        )
    if values.shape[0] != len(feature_names):
        raise ProcessIntelligenceError(
            f"importance length ({values.shape[0]}) must match "
            f"feature count ({len(feature_names)})"
        )
    if not np.isfinite(values.astype(np.float64, copy=False)).all():
        raise ProcessIntelligenceError("importance values must be finite")

    indexed = list(enumerate(values.astype(np.float64, copy=False).tolist()))
    if by_absolute:
        indexed.sort(key=lambda item: (-abs(item[1]), item[0]))
    else:
        indexed.sort(key=lambda item: (-item[1], item[0]))

    result: dict[str, float] = {}
    for original_index, raw_value in indexed:
        result[feature_names[original_index]] = _normalize_zero(float(raw_value))
    return result


def _regression_global_explanation(
    estimator: Any,
    feature_names: tuple[str, ...],
) -> ExplanationResult:
    """Build a global explanation from coef_ or feature_importances_."""
    coef = getattr(estimator, "coef_", None)
    if coef is not None:
        array = np.asarray(coef, dtype=np.float64)
        if array.ndim == 2:
            if array.shape[0] != 1:
                raise ProcessIntelligenceError(
                    "regression coef_ with multiple rows is not supported for "
                    f"global explanation, got shape {array.shape}"
                )
            array = array.reshape(-1)
        elif array.ndim != 1:
            raise ProcessIntelligenceError(
                f"regression coef_ must be 1D or (1, n_features), got shape {array.shape}"
            )
        return ExplanationResult(
            method="coef_",
            feature_importances=_sorted_importances(
                feature_names,
                array,
                by_absolute=True,
            ),
            notes=[],
        )

    importances = getattr(estimator, "feature_importances_", None)
    if importances is not None:
        array = np.asarray(importances, dtype=np.float64)
        return ExplanationResult(
            method="feature_importances_",
            feature_importances=_sorted_importances(
                feature_names,
                array,
                by_absolute=False,
            ),
            notes=[],
        )

    return ExplanationResult(
        method="unsupported",
        feature_importances={},
        notes=[
            "estimator does not provide coef_ or feature_importances_ "
            "for global explanation"
        ],
    )


class SklearnRegressorModel(SklearnModelAdapter):
    """Concrete sklearn adapter for supervised regression estimators."""

    def __init__(
        self,
        *,
        spec: ModelSpec,
        estimator: Any,
        random_state: int = 42,
    ) -> None:
        """Create a regression model around a cloned sklearn regressor.

        Args:
            spec: Model specification with ``task=REGRESSION``.
            estimator: Sklearn regressor with ``fit`` and ``predict``.
            random_state: Non-negative seed applied when supported.

        Raises:
            TypeError: If ``spec``, ``estimator``, or ``random_state`` is invalid.
            DataValidationError: If ``spec.task`` is not regression.
            ValueError: If ``random_state`` is negative.
        """
        if not isinstance(spec, ModelSpec):
            raise TypeError(f"spec must be ModelSpec, got {type(spec).__name__}")
        if spec.task is not AnalysisTask.REGRESSION:
            raise DataValidationError(
                f"SklearnRegressorModel requires task=REGRESSION, got {spec.task!r}"
            )
        if not is_regressor(estimator):
            raise TypeError(
                "estimator must be a sklearn regressor, "
                f"got {type(estimator).__name__}"
            )
        super().__init__(spec=spec, estimator=estimator, random_state=random_state)

    def fit(self, X: DataFrameLike, y: SeriesLike | None = None) -> SklearnRegressorModel:
        """Fit the regressor after validating numeric regression targets.

        Args:
            X: Feature frame.
            y: Numeric regression target.

        Returns:
            The fitted model instance.
        """
        _validate_regression_target(y)
        return super().fit(X, y)

    def evaluate(
        self,
        X: DataFrameLike,
        y: SeriesLike | None = None,
    ) -> ModelEvaluation:
        """Evaluate regression performance on ``X`` against ``y``.

        Args:
            X: Feature frame.
            y: Ground-truth regression targets.

        Returns:
            Structured metrics in a ``ModelEvaluation`` payload.
        """
        if not self.is_fitted:
            raise ProcessIntelligenceError(
                "evaluate requires a successful fit before it can be called"
            )
        predictions = self.predict(X)
        metrics = evaluate_regression(y, predictions)
        # ModelEvaluation.metrics is dict[str, float], so omit r2 when unavailable.
        payload: dict[str, float] = {
            "mae": metrics.mae,
            "rmse": metrics.rmse,
            "sample_count": float(metrics.sample_count),
        }
        if metrics.r2 is not None:
            payload["r2"] = metrics.r2
        return ModelEvaluation(metrics=payload, notes=list(metrics.warnings))

    def explain(self, X: DataFrameLike) -> ExplanationResult:
        """Return a global feature explanation for the fitted regressor.

        Args:
            X: Feature frame used only for interface compatibility. Global
                explanations do not depend on row values.

        Returns:
            Structured global explanation result.
        """
        _ = X
        if not self.is_fitted or self._fitted_estimator is None:
            raise ProcessIntelligenceError(
                "explain requires a successful fit before it can be called"
            )
        return _regression_global_explanation(
            self._fitted_estimator,
            self.feature_names,
        )


def build_dummy_regressor_spec() -> ModelSpec:
    """Build the default Dummy Regressor specification."""
    return ModelSpec(
        name="Dummy Regressor",
        task=AnalysisTask.REGRESSION,
        estimator_key="dummy_regressor",
        optional_dependencies=[],
        priority=10,
        time_budget_seconds=1.0,
    )


def build_linear_regression_spec() -> ModelSpec:
    """Build the default Linear Regression specification."""
    return ModelSpec(
        name="Linear Regression",
        task=AnalysisTask.REGRESSION,
        estimator_key="linear_regression",
        optional_dependencies=[],
        priority=20,
        time_budget_seconds=2.0,
    )


def build_ridge_regression_spec() -> ModelSpec:
    """Build the default Ridge Regression specification."""
    return ModelSpec(
        name="Ridge Regression",
        task=AnalysisTask.REGRESSION,
        estimator_key="ridge_regression",
        optional_dependencies=[],
        priority=30,
        time_budget_seconds=2.0,
    )


def build_random_forest_regressor_spec() -> ModelSpec:
    """Build the default Random Forest Regressor specification."""
    return ModelSpec(
        name="Random Forest Regressor",
        task=AnalysisTask.REGRESSION,
        estimator_key="random_forest_regressor",
        optional_dependencies=[],
        priority=40,
        time_budget_seconds=15.0,
    )


def create_dummy_regressor(*, random_state: int = 42) -> SklearnRegressorModel:
    """Create an unfitted DummyRegressor model with mean strategy."""
    seed = _validate_random_state(random_state)
    return SklearnRegressorModel(
        spec=build_dummy_regressor_spec(),
        estimator=DummyRegressor(strategy="mean"),
        random_state=seed,
    )


def create_linear_regression(*, random_state: int = 42) -> SklearnRegressorModel:
    """Create an unfitted LinearRegression model."""
    seed = _validate_random_state(random_state)
    return SklearnRegressorModel(
        spec=build_linear_regression_spec(),
        estimator=LinearRegression(),
        random_state=seed,
    )


def create_ridge_regression(*, random_state: int = 42) -> SklearnRegressorModel:
    """Create an unfitted Ridge regression model."""
    seed = _validate_random_state(random_state)
    return SklearnRegressorModel(
        spec=build_ridge_regression_spec(),
        estimator=Ridge(alpha=1.0),
        random_state=seed,
    )


def create_random_forest_regressor(*, random_state: int = 42) -> SklearnRegressorModel:
    """Create an unfitted RandomForestRegressor model."""
    seed = _validate_random_state(random_state)
    return SklearnRegressorModel(
        spec=build_random_forest_regressor_spec(),
        estimator=RandomForestRegressor(
            n_estimators=100,
            random_state=seed,
            n_jobs=1,
        ),
        random_state=seed,
    )
