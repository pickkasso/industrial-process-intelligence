"""Regression and classification evaluation metrics (Step 6B)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
from pydantic import BaseModel, Field, field_validator
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)

from process_intelligence.core.exceptions import DataValidationError, InsufficientDataError

_R2_MIN_SAMPLES_WARNING = (
    "R2 requires at least 2 samples; r2 is unavailable for a single sample"
)
_MISSING_PRED_CLASS_WARNING = (
    "y_pred is missing one or more classes present in y_true"
)
_NEW_PRED_CLASS_WARNING = (
    "y_pred contains one or more classes not present in y_true"
)


def _normalize_zero(value: float) -> float:
    """Normalize signed zero to ``0.0``."""
    return 0.0 if value == 0.0 else value


def _require_finite_metric(name: str, value: float) -> float:
    """Return a finite Python float metric or raise ``DataValidationError``."""
    number = float(value)
    if not math.isfinite(number):
        raise DataValidationError(f"{name} metric must be finite, got {number!r}")
    return _normalize_zero(number)


def _is_string_or_bytes(value: object) -> bool:
    """Return True for str or bytes inputs."""
    return isinstance(value, (str, bytes))


def _coerce_array(values: Any, *, field_name: str) -> np.ndarray:
    """Convert supported 1D inputs to an independent NumPy array."""
    if _is_string_or_bytes(values):
        raise TypeError(
            f"{field_name} must not be str or bytes; got {type(values).__name__}"
        )

    if isinstance(values, pl.Series):
        # Polars Series.to_numpy() does not accept copy=; copy via NumPy.
        array = np.array(values.to_numpy(), copy=True)
    elif isinstance(values, pd.Series):
        array = values.to_numpy(copy=True)
    elif isinstance(values, np.ndarray):
        array = np.array(values, copy=True)
    elif isinstance(values, Sequence):
        array = np.asarray(list(values))
    else:
        raise TypeError(
            f"{field_name} must be a polars.Series, pandas.Series, "
            f"1D numpy.ndarray, or Sequence (excluding str/bytes), "
            f"got {type(values).__name__}"
        )

    if array.ndim != 1:
        raise DataValidationError(
            f"{field_name} must be 1-dimensional, got shape {array.shape}"
        )
    return array


def _reject_polars_nulls(values: Any, *, field_name: str) -> None:
    """Reject Polars Series inputs that contain null values."""
    if isinstance(values, pl.Series) and int(values.null_count()) > 0:
        raise DataValidationError(f"{field_name} must not contain null values")


def _validate_aligned_arrays(
    y_true: Any,
    y_pred: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce aligned 1D true/pred arrays."""
    _reject_polars_nulls(y_true, field_name="y_true")
    _reject_polars_nulls(y_pred, field_name="y_pred")
    true_values = _coerce_array(y_true, field_name="y_true")
    pred_values = _coerce_array(y_pred, field_name="y_pred")

    if true_values.size == 0 or pred_values.size == 0:
        raise InsufficientDataError("y_true and y_pred must contain at least one sample")
    if true_values.shape[0] != pred_values.shape[0]:
        raise DataValidationError(
            f"y_true length ({true_values.shape[0]}) must match "
            f"y_pred length ({pred_values.shape[0]})"
        )
    return true_values, pred_values


def _has_null_entries(values: np.ndarray) -> bool:
    """Return True when the array contains null-like entries."""
    if values.dtype.kind == "O":
        return any(value is None or (isinstance(value, float) and math.isnan(value))
                   for value in values.tolist())
    if values.dtype.kind == "f":
        return bool(np.isnan(values).any())
    return False


def _has_non_finite_numeric(values: np.ndarray) -> bool:
    """Return True when numeric values contain NaN or infinity."""
    if values.dtype.kind in {"f", "c"}:
        return not bool(np.isfinite(values.astype(np.float64, copy=False)).all())
    if values.dtype.kind == "O":
        for value in values.tolist():
            if isinstance(value, (bool, np.bool_)):
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                if not math.isfinite(float(value)):
                    return True
    return False


def _is_boolean_array(values: np.ndarray) -> bool:
    """Return True when values are boolean labels."""
    if values.dtype == np.bool_ or values.dtype.kind == "b":
        return True
    if values.dtype.kind == "O" and values.size > 0:
        return all(isinstance(value, (bool, np.bool_)) for value in values.tolist())
    return False


def _is_string_array(values: np.ndarray) -> bool:
    """Return True when values are string-like labels."""
    if values.dtype.kind in {"U", "S"}:
        return True
    if values.dtype.kind == "O" and values.size > 0:
        return all(isinstance(value, str) for value in values.tolist())
    return False


def _validate_regression_values(values: np.ndarray, *, field_name: str) -> np.ndarray:
    """Validate regression targets/predictions as finite non-boolean numerics."""
    if _has_null_entries(values):
        raise DataValidationError(f"{field_name} must not contain null values")
    if _is_boolean_array(values):
        raise DataValidationError(
            f"{field_name} must be numeric for regression; boolean values are not allowed"
        )
    if _is_string_array(values):
        raise DataValidationError(
            f"{field_name} must be numeric for regression; string values are not allowed"
        )
    if values.dtype.kind not in {"f", "i", "u"}:
        if values.dtype.kind == "O":
            try:
                numeric = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise DataValidationError(
                    f"{field_name} must be numeric for regression"
                ) from exc
        else:
            raise DataValidationError(f"{field_name} must be numeric for regression")
    else:
        numeric = values.astype(np.float64, copy=True)

    if not np.isfinite(numeric).all():
        raise DataValidationError(
            f"{field_name} must not contain NaN or infinite values"
        )
    return numeric


def _validate_classification_values(values: np.ndarray, *, field_name: str) -> np.ndarray:
    """Validate classification labels (numeric, string, or boolean)."""
    if _has_null_entries(values):
        raise DataValidationError(f"{field_name} must not contain null values")
    if _has_non_finite_numeric(values):
        raise DataValidationError(
            f"{field_name} must not contain NaN or infinite numeric values"
        )
    return values


def _dedupe_warnings(warnings: list[str]) -> list[str]:
    """Return unique warning strings preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for warning in warnings:
        if warning in seen:
            continue
        seen.add(warning)
        result.append(warning)
    return result


class RegressionMetrics(BaseModel):
    """Structured regression evaluation metrics for a single prediction set."""

    sample_count: int
    mae: float
    rmse: float
    r2: float | None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("sample_count")
    @classmethod
    def _validate_sample_count(cls, value: int) -> int:
        if value < 1:
            raise ValueError("sample_count must be >= 1")
        return value

    @field_validator("mae", "rmse")
    @classmethod
    def _validate_non_negative_finite(cls, value: float) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("metric must be a finite float")
        if number < 0.0:
            raise ValueError("metric must be >= 0")
        return _normalize_zero(number)

    @field_validator("r2")
    @classmethod
    def _validate_r2(cls, value: float | None) -> float | None:
        if value is None:
            return None
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("r2 must be None or a finite float")
        return _normalize_zero(number)

    @field_validator("warnings")
    @classmethod
    def _validate_warnings(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("warnings must not contain duplicates")
        return value


class ClassificationMetrics(BaseModel):
    """Structured classification evaluation metrics for a single prediction set."""

    sample_count: int
    class_count: int
    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    warnings: list[str] = Field(default_factory=list)

    @field_validator("sample_count", "class_count")
    @classmethod
    def _validate_counts(cls, value: int) -> int:
        if value < 1:
            raise ValueError("count must be >= 1")
        return value

    @field_validator(
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    )
    @classmethod
    def _validate_unit_interval(cls, value: float) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("metric must be a finite float")
        if number < 0.0 or number > 1.0:
            raise ValueError("metric must be between 0 and 1 inclusive")
        return _normalize_zero(number)

    @field_validator("warnings")
    @classmethod
    def _validate_warnings(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("warnings must not contain duplicates")
        return value


def evaluate_regression(y_true: Any, y_pred: Any) -> RegressionMetrics:
    """Compute regression metrics from true and predicted values.

    Args:
        y_true: Ground-truth regression targets.
        y_pred: Model predictions aligned with ``y_true``.

    Returns:
        Structured regression metrics. Inputs are never modified.

    Raises:
        TypeError: If an input type is unsupported.
        InsufficientDataError: If either input is empty.
        DataValidationError: If lengths differ or values are invalid.
    """
    true_raw, pred_raw = _validate_aligned_arrays(y_true, y_pred)
    true_values = _validate_regression_values(true_raw, field_name="y_true")
    pred_values = _validate_regression_values(pred_raw, field_name="y_pred")

    sample_count = int(true_values.shape[0])
    mae = _require_finite_metric("mae", float(mean_absolute_error(true_values, pred_values)))
    rmse = _require_finite_metric(
        "rmse",
        float(root_mean_squared_error(true_values, pred_values)),
    )

    warnings: list[str] = []
    r2: float | None
    if sample_count < 2:
        r2 = None
        warnings.append(_R2_MIN_SAMPLES_WARNING)
    else:
        r2 = _require_finite_metric("r2", float(r2_score(true_values, pred_values)))

    return RegressionMetrics(
        sample_count=sample_count,
        mae=mae,
        rmse=rmse,
        r2=r2,
        warnings=_dedupe_warnings(warnings),
    )


def evaluate_classification(y_true: Any, y_pred: Any) -> ClassificationMetrics:
    """Compute classification metrics from true and predicted labels.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels aligned with ``y_true``.

    Returns:
        Structured classification metrics. Inputs are never modified.

    Raises:
        TypeError: If an input type is unsupported.
        InsufficientDataError: If either input is empty.
        DataValidationError: If lengths differ or values are invalid.
    """
    true_values, pred_values = _validate_aligned_arrays(y_true, y_pred)
    true_values = _validate_classification_values(true_values, field_name="y_true")
    pred_values = _validate_classification_values(pred_values, field_name="y_pred")

    sample_count = int(true_values.shape[0])
    combined = np.concatenate([true_values, pred_values])
    # np.unique on object arrays of mixed comparable labels is sufficient here.
    class_count = int(np.unique(combined).shape[0])
    if class_count < 1:
        raise DataValidationError("class_count must be >= 1")

    warnings: list[str] = []
    true_classes = {str(value) for value in np.unique(true_values).tolist()}
    pred_classes = {str(value) for value in np.unique(pred_values).tolist()}
    if true_classes - pred_classes:
        warnings.append(_MISSING_PRED_CLASS_WARNING)
    if pred_classes - true_classes:
        warnings.append(_NEW_PRED_CLASS_WARNING)

    accuracy = _require_finite_metric(
        "accuracy",
        float(accuracy_score(true_values, pred_values)),
    )
    balanced_accuracy = _require_finite_metric(
        "balanced_accuracy",
        float(balanced_accuracy_score(true_values, pred_values)),
    )
    precision_macro = _require_finite_metric(
        "precision_macro",
        float(precision_score(true_values, pred_values, average="macro", zero_division=0)),
    )
    recall_macro = _require_finite_metric(
        "recall_macro",
        float(recall_score(true_values, pred_values, average="macro", zero_division=0)),
    )
    f1_macro = _require_finite_metric(
        "f1_macro",
        float(f1_score(true_values, pred_values, average="macro", zero_division=0)),
    )

    return ClassificationMetrics(
        sample_count=sample_count,
        class_count=class_count,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        f1_macro=f1_macro,
        warnings=_dedupe_warnings(warnings),
    )
