"""Concrete sklearn classification models and default factories (Step 6B)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
from sklearn.base import is_classifier  # type: ignore[import-untyped]
from sklearn.dummy import DummyClassifier  # type: ignore[import-untyped]
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.exceptions import (
    DataValidationError,
    InsufficientDataError,
    ProcessIntelligenceError,
)
from process_intelligence.core.protocols import DataFrameLike, SeriesLike
from process_intelligence.core.schemas import ExplanationResult, ModelEvaluation, ModelSpec
from process_intelligence.evaluation.metrics import evaluate_classification
from process_intelligence.models.sklearn_adapter import (
    SklearnModelAdapter,
    _frame_to_numeric_array,
    _validate_x,
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


def _is_string_or_bytes(value: object) -> bool:
    """Return True for str or bytes targets."""
    return isinstance(value, (str, bytes))


def _target_to_array(y: object) -> np.ndarray:
    """Convert a supervised target to a 1D NumPy array without mutating input."""
    if y is None:
        raise DataValidationError("y is required for supervised classification models")
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


def _validate_classification_target(y: object) -> None:
    """Validate classification-specific target constraints without mutating ``y``."""
    values = _target_to_array(y)
    if values.size < 1:
        raise InsufficientDataError("y must contain at least one value")

    if isinstance(y, pl.Series) and y.null_count() > 0:
        raise DataValidationError("y must not contain null values")
    if isinstance(y, pd.Series) and y.isna().any():
        raise DataValidationError("y must not contain null values")

    if values.dtype.kind == "O" and any(value is None for value in values.tolist()):
        raise DataValidationError("y must not contain null values")

    if values.dtype.kind in {"f", "c"}:
        if not np.isfinite(values.astype(np.float64, copy=False)).all():
            raise DataValidationError("y must not contain NaN or infinite values")
    elif values.dtype.kind == "O":
        for value in values.tolist():
            if isinstance(value, (bool, np.bool_, str)):
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                if not np.isfinite(float(value)):
                    raise DataValidationError(
                        "y must not contain NaN or infinite values"
                    )

    unique_count = int(np.unique(values).shape[0])
    if unique_count < 2:
        raise InsufficientDataError(
            "classification target must contain at least 2 unique classes"
        )


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


def _classification_global_explanation(
    estimator: Any,
    feature_names: tuple[str, ...],
) -> ExplanationResult:
    """Build a global explanation from coef_ or feature_importances_."""
    coef = getattr(estimator, "coef_", None)
    if coef is not None:
        array = np.asarray(coef, dtype=np.float64)
        if array.ndim == 1:
            return ExplanationResult(
                method="coef_",
                feature_importances=_sorted_importances(
                    feature_names,
                    array,
                    by_absolute=True,
                ),
                notes=[],
            )
        if array.ndim == 2:
            if array.shape[1] != len(feature_names):
                raise ProcessIntelligenceError(
                    f"coef_ feature dimension ({array.shape[1]}) must match "
                    f"feature count ({len(feature_names)})"
                )
            if array.shape[0] == 1:
                return ExplanationResult(
                    method="coef_",
                    feature_importances=_sorted_importances(
                        feature_names,
                        array.reshape(-1),
                        by_absolute=True,
                    ),
                    notes=[],
                )
            mean_abs = np.mean(np.abs(array), axis=0)
            return ExplanationResult(
                method="multiclass_mean_absolute_coef_",
                feature_importances=_sorted_importances(
                    feature_names,
                    mean_abs,
                    by_absolute=False,
                ),
                notes=[],
            )
        raise ProcessIntelligenceError(
            f"classification coef_ must be 1D or 2D, got shape {array.shape}"
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


class SklearnClassifierModel(SklearnModelAdapter):
    """Concrete sklearn adapter for supervised classification estimators."""

    def __init__(
        self,
        *,
        spec: ModelSpec,
        estimator: Any,
        random_state: int = 42,
    ) -> None:
        """Create a classification model around a cloned sklearn classifier.

        Args:
            spec: Model specification with ``task=CLASSIFICATION``.
            estimator: Sklearn classifier with ``fit`` and ``predict``.
            random_state: Non-negative seed applied when supported.

        Raises:
            TypeError: If ``spec``, ``estimator``, or ``random_state`` is invalid.
            DataValidationError: If ``spec.task`` is not classification.
            ValueError: If ``random_state`` is negative.
        """
        if not isinstance(spec, ModelSpec):
            raise TypeError(f"spec must be ModelSpec, got {type(spec).__name__}")
        if spec.task is not AnalysisTask.CLASSIFICATION:
            raise DataValidationError(
                "SklearnClassifierModel requires task=CLASSIFICATION, "
                f"got {spec.task!r}"
            )
        if not is_classifier(estimator):
            raise TypeError(
                "estimator must be a sklearn classifier, "
                f"got {type(estimator).__name__}"
            )
        super().__init__(spec=spec, estimator=estimator, random_state=random_state)

    def fit(self, X: DataFrameLike, y: SeriesLike | None = None) -> SklearnClassifierModel:
        """Fit the classifier after validating class-label constraints.

        Args:
            X: Feature frame.
            y: Class labels (numeric, string, or boolean).

        Returns:
            The fitted model instance.
        """
        _validate_classification_target(y)
        return super().fit(X, y)

    def evaluate(
        self,
        X: DataFrameLike,
        y: SeriesLike | None = None,
    ) -> ModelEvaluation:
        """Evaluate classification performance on ``X`` against ``y``.

        Args:
            X: Feature frame.
            y: Ground-truth class labels.

        Returns:
            Structured metrics in a ``ModelEvaluation`` payload.
        """
        if not self.is_fitted:
            raise ProcessIntelligenceError(
                "evaluate requires a successful fit before it can be called"
            )
        predictions = self.predict(X)
        metrics = evaluate_classification(y, predictions)
        payload = {
            "accuracy": metrics.accuracy,
            "balanced_accuracy": metrics.balanced_accuracy,
            "precision_macro": metrics.precision_macro,
            "recall_macro": metrics.recall_macro,
            "f1_macro": metrics.f1_macro,
            "sample_count": float(metrics.sample_count),
            "class_count": float(metrics.class_count),
        }
        return ModelEvaluation(metrics=payload, notes=list(metrics.warnings))

    def explain(self, X: DataFrameLike) -> ExplanationResult:
        """Return a global feature explanation for the fitted classifier.

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
        return _classification_global_explanation(
            self._fitted_estimator,
            self.feature_names,
        )

    def predict_proba(self, X: DataFrameLike) -> np.ndarray:
        """Return class probability estimates for ``X``.

        Args:
            X: Feature frame with the same columns and order as training.

        Returns:
            Two-dimensional NumPy array of finite probabilities in ``[0, 1]``.
        """
        if not self.is_fitted or self._fitted_estimator is None:
            raise ProcessIntelligenceError(
                "predict_proba requires a successful fit before it can be called"
            )
        predict_proba = getattr(self._fitted_estimator, "predict_proba", None)
        if not callable(predict_proba):
            raise ProcessIntelligenceError(
                "estimator does not provide a callable predict_proba method"
            )

        frame, feature_names = _validate_x(X)
        if feature_names != self._feature_names:
            raise DataValidationError(
                "X feature names and order must exactly match training features: "
                f"expected {list(self._feature_names)}, got {list(feature_names)}"
            )

        matrix = _frame_to_numeric_array(frame, feature_names)
        probabilities = np.asarray(predict_proba(matrix), dtype=np.float64)
        if probabilities.ndim != 2:
            raise ProcessIntelligenceError(
                "predict_proba must return a 2D array, "
                f"got shape {probabilities.shape}"
            )
        if probabilities.shape[0] != matrix.shape[0]:
            raise ProcessIntelligenceError(
                "predict_proba row count "
                f"({probabilities.shape[0]}) must match X row count ({matrix.shape[0]})"
            )
        if not np.isfinite(probabilities).all():
            raise ProcessIntelligenceError("predict_proba values must be finite")
        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise ProcessIntelligenceError(
                "predict_proba values must be between 0 and 1 inclusive"
            )
        return probabilities


def build_dummy_classifier_spec() -> ModelSpec:
    """Build the default Dummy Classifier specification."""
    return ModelSpec(
        name="Dummy Classifier",
        task=AnalysisTask.CLASSIFICATION,
        estimator_key="dummy_classifier",
        optional_dependencies=[],
        priority=10,
        time_budget_seconds=1.0,
    )


def build_logistic_regression_spec() -> ModelSpec:
    """Build the default Logistic Regression specification."""
    return ModelSpec(
        name="Logistic Regression",
        task=AnalysisTask.CLASSIFICATION,
        estimator_key="logistic_regression",
        optional_dependencies=[],
        priority=20,
        time_budget_seconds=5.0,
    )


def build_random_forest_classifier_spec() -> ModelSpec:
    """Build the default Random Forest Classifier specification."""
    return ModelSpec(
        name="Random Forest Classifier",
        task=AnalysisTask.CLASSIFICATION,
        estimator_key="random_forest_classifier",
        optional_dependencies=[],
        priority=40,
        time_budget_seconds=15.0,
    )


def create_dummy_classifier(*, random_state: int = 42) -> SklearnClassifierModel:
    """Create an unfitted DummyClassifier model with most_frequent strategy."""
    seed = _validate_random_state(random_state)
    return SklearnClassifierModel(
        spec=build_dummy_classifier_spec(),
        estimator=DummyClassifier(strategy="most_frequent"),
        random_state=seed,
    )


def create_logistic_regression(*, random_state: int = 42) -> SklearnClassifierModel:
    """Create an unfitted LogisticRegression model."""
    seed = _validate_random_state(random_state)
    return SklearnClassifierModel(
        spec=build_logistic_regression_spec(),
        estimator=LogisticRegression(max_iter=1000, random_state=seed),
        random_state=seed,
    )


def create_random_forest_classifier(*, random_state: int = 42) -> SklearnClassifierModel:
    """Create an unfitted RandomForestClassifier model."""
    seed = _validate_random_state(random_state)
    return SklearnClassifierModel(
        spec=build_random_forest_classifier_spec(),
        estimator=RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=1,
        ),
        random_state=seed,
    )
