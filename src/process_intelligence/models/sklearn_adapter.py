"""Common sklearn estimator adapter implementing BaseAnalysisModel (Step 6A)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Self

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import sklearn  # type: ignore[import-untyped]
from sklearn.base import clone  # type: ignore[import-untyped]

from process_intelligence.core.exceptions import (
    DataValidationError,
    InsufficientDataError,
    ProcessIntelligenceError,
)
from process_intelligence.core.protocols import (
    BaseAnalysisModel,
    DataFrameLike,
    SeriesLike,
)
from process_intelligence.core.schemas import (
    ExplanationResult,
    ModelEvaluation,
    ModelMetadata,
    ModelSpec,
)
from process_intelligence.data.loader import ORIGINAL_ROW_ID_COLUMN

_FORBIDDEN_DTYPE_NAMES = frozenset(
    {
        "boolean",
        "bool",
        "string",
        "utf8",
        "categorical",
        "enum",
        "date",
        "datetime",
        "time",
        "object",
        "struct",
        "list",
        "array",
        "null",
    }
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


def _require_sklearn_estimator(estimator: object) -> Any:
    """Validate that ``estimator`` exposes callable fit and predict methods."""
    fit_method = getattr(estimator, "fit", None)
    predict_method = getattr(estimator, "predict", None)
    if not callable(fit_method):
        raise TypeError(
            f"estimator must have a callable fit method, got {type(estimator).__name__}"
        )
    if not callable(predict_method):
        raise TypeError(
            "estimator must have a callable predict method, "
            f"got {type(estimator).__name__}"
        )
    return estimator


def _clone_estimator(estimator: Any) -> Any:
    """Clone an estimator using sklearn's public clone API."""
    try:
        return clone(estimator)
    except TypeError as exc:
        raise TypeError(
            f"estimator is not cloneable by sklearn.base.clone: "
            f"{type(estimator).__name__}"
        ) from exc


def _apply_random_state(estimator: Any, random_state: int) -> Any:
    """Set ``random_state`` on a cloned estimator when the parameter exists."""
    params = estimator.get_params(deep=False)
    if "random_state" in params:
        return estimator.set_params(random_state=random_state)
    return estimator


def _column_names(frame: DataFrameLike) -> list[Any]:
    """Return raw column labels from a Polars or Pandas frame."""
    if isinstance(frame, pl.DataFrame):
        return list(frame.columns)
    return list(frame.columns)


def _row_count(frame: DataFrameLike) -> int:
    """Return the number of rows in a Polars or Pandas frame."""
    if isinstance(frame, pl.DataFrame):
        return frame.height
    return int(len(frame))


def _validate_feature_names(columns: list[Any]) -> tuple[str, ...]:
    """Validate feature column names and return them as a tuple of strings."""
    if not columns:
        raise InsufficientDataError("X must contain at least one feature column")

    names: list[str] = []
    seen: set[str] = set()
    for column in columns:
        if not isinstance(column, str):
            raise DataValidationError(
                f"X column names must be str, got {type(column).__name__}"
            )
        if not column.strip():
            raise DataValidationError(
                "X column names must not be empty or whitespace-only"
            )
        if column == ORIGINAL_ROW_ID_COLUMN:
            raise DataValidationError(
                f"X cannot include reserved feature column '{ORIGINAL_ROW_ID_COLUMN}'"
            )
        if column in seen:
            raise DataValidationError(f"X contains duplicate column name {column!r}")
        seen.add(column)
        names.append(column)
    return tuple(names)


def _polars_dtype_forbidden(dtype: pl.DataType) -> bool:
    """Return True when a Polars dtype is not allowed as a model feature."""
    if dtype == pl.Boolean:
        return True
    if dtype.is_numeric():
        return False
    base_name = dtype.base_type().__name__.casefold()
    type_name = str(dtype).casefold()
    if base_name in _FORBIDDEN_DTYPE_NAMES:
        return True
    return any(token in type_name for token in _FORBIDDEN_DTYPE_NAMES)


def _pandas_dtype_forbidden(dtype: Any) -> bool:
    """Return True when a Pandas dtype is not allowed as a model feature."""
    if pd.api.types.is_bool_dtype(dtype):
        return True
    if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
        return True if pd.api.types.is_complex_dtype(dtype) else False
    if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        return True
    if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(
        dtype
    ):
        return True
    if str(dtype).casefold() in _FORBIDDEN_DTYPE_NAMES:
        return True
    # Categorical / extension types that are not numeric.
    return not pd.api.types.is_numeric_dtype(dtype)


def _validate_feature_dtypes(frame: DataFrameLike, feature_names: tuple[str, ...]) -> None:
    """Reject unsupported feature dtypes, naming the offending columns."""
    bad_columns: list[str] = []
    if isinstance(frame, pl.DataFrame):
        for name in feature_names:
            if _polars_dtype_forbidden(frame.schema[name]):
                bad_columns.append(name)
    else:
        for name in feature_names:
            if _pandas_dtype_forbidden(frame[name].dtype):
                bad_columns.append(name)

    if bad_columns:
        raise DataValidationError(
            "X contains unsupported feature dtypes in columns: "
            + ", ".join(bad_columns)
        )


def _validate_numeric_finite_frame(
    frame: DataFrameLike,
    feature_names: tuple[str, ...],
) -> None:
    """Reject null, NaN, and infinite values in numeric feature columns."""
    bad_columns: list[str] = []
    if isinstance(frame, pl.DataFrame):
        for name in feature_names:
            series = frame.get_column(name)
            if int(series.null_count()) > 0:
                bad_columns.append(name)
                continue
            values = series.to_numpy()
            numeric = np.asarray(values, dtype=np.float64)
            if not np.isfinite(numeric).all():
                bad_columns.append(name)
    else:
        for name in feature_names:
            series = frame[name]
            if series.isna().any():
                bad_columns.append(name)
                continue
            numeric = np.asarray(series.to_numpy(dtype=np.float64, copy=True))
            if not np.isfinite(numeric).all():
                bad_columns.append(name)

    if bad_columns:
        raise DataValidationError(
            "X contains null, NaN, or infinite values in columns: "
            + ", ".join(bad_columns)
        )


def _frame_to_numeric_array(
    frame: DataFrameLike,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    """Convert validated frame columns to a 2D float64 NumPy array."""
    if isinstance(frame, pl.DataFrame):
        selected = frame.select(list(feature_names))
        # Cast Decimal and integer columns to Float64 for sklearn stability.
        cast_exprs = [
            pl.col(name).cast(pl.Float64, strict=False).alias(name)
            for name in feature_names
        ]
        array = selected.select(cast_exprs).to_numpy()
    else:
        selected = frame.loc[:, list(feature_names)]
        array = selected.to_numpy(dtype=np.float64, copy=True)

    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2:
        raise ProcessIntelligenceError(
            f"Expected 2D feature matrix, got array with shape {array.shape}"
        )
    return array


def _validate_x(frame: object) -> tuple[DataFrameLike, tuple[str, ...]]:
    """Validate an input feature frame and return it with feature names."""
    if not isinstance(frame, (pl.DataFrame, pd.DataFrame)):
        raise TypeError(
            "X must be a polars.DataFrame or pandas.DataFrame, "
            f"got {type(frame).__name__}"
        )
    if _row_count(frame) == 0:
        raise InsufficientDataError("X must contain at least one row")

    feature_names = _validate_feature_names(_column_names(frame))
    _validate_feature_dtypes(frame, feature_names)
    _validate_numeric_finite_frame(frame, feature_names)
    return frame, feature_names


def _is_string_or_bytes(value: object) -> bool:
    """Return True for str or bytes targets, which are not Sequence targets."""
    return isinstance(value, (str, bytes))


def _coerce_y(y: object, *, expected_length: int) -> np.ndarray:
    """Validate and coerce a supervised target to a 1D NumPy array."""
    if y is None:
        raise DataValidationError("y is required for supervised sklearn adapters")
    if _is_string_or_bytes(y):
        raise TypeError(
            f"y must not be str or bytes; got {type(y).__name__}"
        )

    values: np.ndarray
    if isinstance(y, pl.Series):
        if y.null_count() > 0:
            raise DataValidationError("y must not contain null values")
        values = y.to_numpy()
    elif isinstance(y, pd.Series):
        if y.isna().any():
            raise DataValidationError("y must not contain null values")
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
        raise DataValidationError(
            f"y must be 1-dimensional, got shape {values.shape}"
        )
    if values.size == 0:
        raise InsufficientDataError("y must contain at least one value")
    if values.shape[0] != expected_length:
        raise DataValidationError(
            f"y length ({values.shape[0]}) must match X row count ({expected_length})"
        )

    if values.dtype.kind in {"f", "i", "u"}:
        numeric = values.astype(np.float64, copy=False)
        if not np.isfinite(numeric).all():
            raise DataValidationError("y must not contain NaN or infinite values")
    elif values.dtype == object:
        if any(value is None for value in values.tolist()):
            raise DataValidationError("y must not contain null values")

    return values


class SklearnModelAdapter(BaseAnalysisModel, ABC):
    """Shared supervised adapter that wraps a cloneable sklearn-style estimator."""

    def __init__(
        self,
        *,
        spec: ModelSpec,
        estimator: Any,
        random_state: int = 42,
    ) -> None:
        """Create an adapter around a cloned estimator template.

        Args:
            spec: Model specification stored as a deep copy.
            estimator: Sklearn-style estimator with callable ``fit`` and
                ``predict``. The input estimator is cloned and never fitted.
            random_state: Non-negative seed applied when the estimator exposes a
                ``random_state`` parameter.

        Raises:
            TypeError: If ``spec``, ``estimator``, or ``random_state`` is invalid,
                or if the estimator cannot be cloned.
            ValueError: If ``random_state`` is negative.
        """
        if not isinstance(spec, ModelSpec):
            raise TypeError(f"spec must be ModelSpec, got {type(spec).__name__}")

        validated_estimator = _require_sklearn_estimator(estimator)
        validated_random_state = _validate_random_state(random_state)

        template = _clone_estimator(validated_estimator)
        template = _apply_random_state(template, validated_random_state)

        self._spec: ModelSpec = spec.model_copy(deep=True)
        self._estimator_template: Any = template
        self._random_state: int = validated_random_state
        self._fitted_estimator: Any | None = None
        self._is_fitted: bool = False
        self._feature_names: tuple[str, ...] = ()
        self._fit_row_count: int = 0
        self._fitted_at: datetime | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether the adapter has a successfully fitted estimator."""
        return self._is_fitted

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Feature names learned during the last successful fit."""
        return self._feature_names

    @property
    def fit_row_count(self) -> int:
        """Number of training rows from the last successful fit."""
        return self._fit_row_count

    def fit(self, X: DataFrameLike, y: SeriesLike | None = None) -> Self:
        """Fit a fresh estimator clone on validated feature and target inputs.

        Args:
            X: Polars or Pandas feature frame.
            y: Supervised target values. ``None`` is rejected.

        Returns:
            The fitted adapter instance.

        Raises:
            TypeError: If ``X`` or ``y`` has an unsupported type.
            InsufficientDataError: If ``X`` or ``y`` is empty.
            DataValidationError: If feature/target validation fails.
        """
        frame, feature_names = _validate_x(X)
        target = _coerce_y(y, expected_length=_row_count(frame))
        matrix = _frame_to_numeric_array(frame, feature_names)

        candidate = _clone_estimator(self._estimator_template)
        candidate.fit(matrix, target)

        self._fitted_estimator = candidate
        self._is_fitted = True
        self._feature_names = feature_names
        self._fit_row_count = int(matrix.shape[0])
        self._fitted_at = datetime.now(tz=UTC)
        return self

    def predict(self, X: DataFrameLike) -> np.ndarray:
        """Generate predictions for a validated feature frame.

        Args:
            X: Polars or Pandas feature frame with the same columns and order
                as the training features.

        Returns:
            One-dimensional NumPy prediction array.

        Raises:
            ProcessIntelligenceError: If the adapter is not fitted or the
                estimator returns an invalid prediction shape.
            TypeError: If ``X`` has an unsupported type.
            InsufficientDataError: If ``X`` is empty.
            DataValidationError: If feature validation or column alignment fails.
        """
        if not self._is_fitted or self._fitted_estimator is None:
            raise ProcessIntelligenceError(
                "predict requires a successful fit before it can be called"
            )

        frame, feature_names = _validate_x(X)
        if feature_names != self._feature_names:
            raise DataValidationError(
                "X feature names and order must exactly match training features: "
                f"expected {list(self._feature_names)}, got {list(feature_names)}"
            )

        matrix = _frame_to_numeric_array(frame, feature_names)
        predictions = np.asarray(self._fitted_estimator.predict(matrix))
        predictions = np.reshape(predictions, -1)
        if predictions.ndim != 1:
            raise ProcessIntelligenceError(
                f"estimator.predict must return a 1D array, got shape {predictions.shape}"
            )
        if predictions.shape[0] != matrix.shape[0]:
            raise ProcessIntelligenceError(
                "estimator.predict row count "
                f"({predictions.shape[0]}) must match X row count ({matrix.shape[0]})"
            )
        return predictions

    def get_metadata(self) -> ModelMetadata:
        """Return traceable metadata for the wrapped model.

        Returns:
            A new ``ModelMetadata`` instance. Mutating the returned object does
            not affect adapter state.
        """
        return ModelMetadata(
            model_name=self._spec.name,
            version=sklearn.__version__,
            task=self._spec.task,
            features=list(self._feature_names),
            training_timestamp=self._fitted_at,
            seed=self._random_state,
            optional_dependencies_used=list(self._spec.optional_dependencies),
        )

    @abstractmethod
    def evaluate(
        self,
        X: DataFrameLike,
        y: SeriesLike | None = None,
    ) -> ModelEvaluation:
        """Evaluate model performance and return structured metrics."""

    @abstractmethod
    def explain(self, X: DataFrameLike) -> ExplanationResult:
        """Explain model predictions for the provided features."""
