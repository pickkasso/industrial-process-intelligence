"""Fit/transform dataset preprocessing with train-only statistics (Step 2G)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Integral, Real
from typing import Any, Literal, Self

import polars as pl
from pydantic import BaseModel, Field, field_validator, model_validator

from process_intelligence.core.exceptions import (
    DataValidationError,
    InsufficientDataError,
    ProcessIntelligenceError,
)
from process_intelligence.core.schemas import PreprocessingEvent
from process_intelligence.data.loader import ORIGINAL_ROW_ID_COLUMN

MISSING_CATEGORY_TOKEN = "__MISSING__"


class PreprocessorConfig(BaseModel):
    """Explicit column lists and strategies for dataset preprocessing.

    Statistics are always fitted on training data only. Column roles are not
    inferred; callers must name numeric and categorical columns explicitly.
    """

    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    numeric_imputation: Literal["none", "median", "mean"] = "median"
    categorical_imputation: Literal["none", "constant", "most_frequent"] = "constant"
    categorical_fill_value: str = MISSING_CATEGORY_TOKEN
    scaling: Literal["none", "standard", "robust"] = "none"

    @field_validator("numeric_columns", "categorical_columns", mode="after")
    @classmethod
    def _copy_and_validate_column_names(cls, value: list[str]) -> list[str]:
        copied = list(value)
        for index, name in enumerate(copied):
            if not isinstance(name, str):
                raise ValueError(
                    f"column names must be str, got {type(name).__name__} at index {index}"
                )
            if name == "" or name.strip() == "":
                raise ValueError(
                    "column names must be non-empty and must not be whitespace-only"
                )
        if len(set(copied)) != len(copied):
            raise ValueError(f"duplicate column names are not allowed: {copied}")
        if ORIGINAL_ROW_ID_COLUMN in copied:
            raise ValueError(
                f"reserved column '{ORIGINAL_ROW_ID_COLUMN}' cannot be preprocessed"
            )
        return copied

    @field_validator("categorical_fill_value", mode="after")
    @classmethod
    def _validate_fill_value_type(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError(
                f"categorical_fill_value must be str, got {type(value).__name__}"
            )
        return value

    @model_validator(mode="after")
    def _validate_column_sets_and_fill_value(self) -> Self:
        overlap = set(self.numeric_columns) & set(self.categorical_columns)
        if overlap:
            raise ValueError(
                "columns cannot be both numeric and categorical: "
                + ", ".join(sorted(overlap))
            )
        if (
            self.categorical_imputation == "constant"
            and self.categorical_fill_value.strip() == ""
        ):
            raise ValueError(
                "categorical_fill_value must be non-empty when "
                "categorical_imputation is 'constant'"
            )
        return self


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    """Immutable container pairing a preprocessed frame with step events."""

    frame: pl.DataFrame
    events: tuple[PreprocessingEvent, ...]


@dataclass(frozen=True, slots=True)
class _FittedStatistics:
    numeric_fill_values: dict[str, float]
    categorical_fill_values: dict[str, str]
    scaling_centers: dict[str, float]
    scaling_scales: dict[str, float]
    warnings: list[str]


class DatasetPreprocessor:
    """Fit preprocessing statistics on train data and apply them via transform.

    The preprocessor never mutates input frames, never drops or reorders rows,
    and never updates fitted statistics during ``transform``.
    """

    def __init__(self, config: PreprocessorConfig) -> None:
        """Create a preprocessor from an explicit configuration.

        Args:
            config: Preprocessing column lists and strategies.

        Raises:
            TypeError: If ``config`` is not a ``PreprocessorConfig``.
        """
        if not isinstance(config, PreprocessorConfig):
            raise TypeError(
                "config must be PreprocessorConfig, "
                f"got {type(config).__name__}"
            )
        self._config = config.model_copy(deep=True)
        self._fitted: _FittedStatistics | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether ``fit`` has successfully completed."""
        return self._fitted is not None

    def fit(self, frame: pl.DataFrame) -> Self:
        """Learn imputation and scaling statistics from a training frame.

        Failed fits leave any previously fitted state unchanged.

        Args:
            frame: Training data as a Polars DataFrame.

        Returns:
            ``self`` for fluent chaining.

        Raises:
            TypeError: If ``frame`` is not a Polars DataFrame.
            DataValidationError: If configured columns are missing, have the
                wrong dtype, or contain non-finite numeric values, or if
                required statistics cannot be computed.
            InsufficientDataError: If configured columns exist but the frame
                has zero rows.
        """
        self._require_dataframe(frame, context="fit")
        statistics = self._compute_fit_statistics(frame)
        self._fitted = statistics
        return self

    def transform(self, frame: pl.DataFrame) -> PreprocessingResult:
        """Apply fitted statistics without recomputing them.

        Args:
            frame: Data to transform as a Polars DataFrame.

        Returns:
            A ``PreprocessingResult`` with the transformed frame and events.

        Raises:
            TypeError: If ``frame`` is not a Polars DataFrame.
            ProcessIntelligenceError: If the preprocessor has not been fitted.
            DataValidationError: If configured columns are missing, have the
                wrong dtype, or contain non-finite numeric values.
        """
        self._require_dataframe(frame, context="transform")
        if self._fitted is None:
            raise ProcessIntelligenceError(
                "DatasetPreprocessor.transform requires a successful fit first"
            )
        self._validate_configured_columns(frame)
        self._validate_numeric_finite(frame)

        fitted = self._fitted
        config = self._config
        rows_before = frame.height
        result_frame = self._apply_transform(frame, fitted)

        events = self._build_events(
            rows_before=rows_before,
            rows_after=result_frame.height,
            fitted=fitted,
            config=config,
        )
        return PreprocessingResult(frame=result_frame, events=events)

    def fit_transform(self, frame: pl.DataFrame) -> PreprocessingResult:
        """Fit on ``frame`` then transform the same frame.

        Args:
            frame: Training data used for both fit and transform.

        Returns:
            The ``PreprocessingResult`` from ``transform`` after a successful fit.
        """
        self.fit(frame)
        return self.transform(frame)

    def get_fitted_statistics(self) -> dict[str, Any]:
        """Return a deep copy of fitted imputation and scaling statistics.

        Returns:
            A new dict with keys ``numeric_fill_values``,
            ``categorical_fill_values``, ``scaling_centers``,
            ``scaling_scales``, and ``warnings``.

        Raises:
            ProcessIntelligenceError: If the preprocessor has not been fitted.
        """
        if self._fitted is None:
            raise ProcessIntelligenceError(
                "DatasetPreprocessor.get_fitted_statistics requires a successful fit first"
            )
        return {
            "numeric_fill_values": dict(self._fitted.numeric_fill_values),
            "categorical_fill_values": dict(self._fitted.categorical_fill_values),
            "scaling_centers": dict(self._fitted.scaling_centers),
            "scaling_scales": dict(self._fitted.scaling_scales),
            "warnings": list(self._fitted.warnings),
        }

    def _compute_fit_statistics(self, frame: pl.DataFrame) -> _FittedStatistics:
        config = self._config
        has_configured_columns = bool(config.numeric_columns or config.categorical_columns)
        if has_configured_columns and frame.height == 0:
            raise InsufficientDataError(
                "Cannot fit preprocessor on an empty frame when columns are configured"
            )

        self._validate_configured_columns(frame)
        self._validate_numeric_finite(frame)

        numeric_fill_values: dict[str, float] = {}
        categorical_fill_values: dict[str, str] = {}
        scaling_centers: dict[str, float] = {}
        scaling_scales: dict[str, float] = {}
        warnings: list[str] = []

        if config.numeric_imputation == "median":
            for column in config.numeric_columns:
                series = frame.get_column(column)
                value = _require_python_float(
                    series.median(),
                    error_message=(
                        f"Cannot compute median for all-null numeric column '{column}'"
                    ),
                )
                numeric_fill_values[column] = value
        elif config.numeric_imputation == "mean":
            for column in config.numeric_columns:
                series = frame.get_column(column)
                value = _require_python_float(
                    series.mean(),
                    error_message=(
                        f"Cannot compute mean for all-null numeric column '{column}'"
                    ),
                )
                numeric_fill_values[column] = value

        if config.categorical_imputation == "constant":
            for column in config.categorical_columns:
                categorical_fill_values[column] = config.categorical_fill_value
        elif config.categorical_imputation == "most_frequent":
            for column in config.categorical_columns:
                categorical_fill_values[column] = _most_frequent_value(
                    frame.get_column(column),
                    column_name=column,
                )

        if config.scaling in {"standard", "robust"}:
            for column in config.numeric_columns:
                series = frame.get_column(column)
                if column in numeric_fill_values:
                    series = series.fill_null(numeric_fill_values[column])

                if config.scaling == "standard":
                    stats_error = (
                        f"Cannot compute standard scaling statistics for column '{column}'"
                    )
                    center_f = _require_python_float(
                        series.mean(),
                        error_message=stats_error,
                    )
                    scale_f = _require_python_float(
                        series.std(ddof=0),
                        error_message=stats_error,
                    )
                    if scale_f == 0.0:
                        scale_f = 1.0
                        warnings.append(
                            f"Column '{column}' has zero variance; "
                            "using scale=1.0 for standard scaling."
                        )
                    scaling_centers[column] = center_f
                    scaling_scales[column] = scale_f
                else:
                    stats_error = (
                        f"Cannot compute robust scaling statistics for column '{column}'"
                    )
                    center_f = _require_python_float(
                        series.median(),
                        error_message=stats_error,
                    )
                    q1 = _require_python_float(
                        series.quantile(0.25),
                        error_message=stats_error,
                    )
                    q3 = _require_python_float(
                        series.quantile(0.75),
                        error_message=stats_error,
                    )
                    scale_f = q3 - q1
                    if scale_f == 0.0:
                        scale_f = 1.0
                        warnings.append(
                            f"Column '{column}' has zero IQR; "
                            "using scale=1.0 for robust scaling."
                        )
                    scaling_centers[column] = center_f
                    scaling_scales[column] = scale_f

        return _FittedStatistics(
            numeric_fill_values=numeric_fill_values,
            categorical_fill_values=categorical_fill_values,
            scaling_centers=scaling_centers,
            scaling_scales=scaling_scales,
            warnings=warnings,
        )

    def _apply_transform(
        self,
        frame: pl.DataFrame,
        fitted: _FittedStatistics,
    ) -> pl.DataFrame:
        config = self._config
        expressions: list[pl.Expr] = []

        for column in config.numeric_columns:
            expr: pl.Expr = pl.col(column)
            needs_float = (
                config.numeric_imputation in {"median", "mean"}
                or config.scaling in {"standard", "robust"}
            )
            if needs_float:
                expr = expr.cast(pl.Float64)

            if column in fitted.numeric_fill_values:
                expr = expr.fill_null(fitted.numeric_fill_values[column])

            if config.scaling in {"standard", "robust"}:
                center = fitted.scaling_centers[column]
                scale = fitted.scaling_scales[column]
                expr = (expr - center) / scale

            if needs_float or column in fitted.numeric_fill_values:
                expressions.append(expr.alias(column))

        for column in config.categorical_columns:
            if column not in fitted.categorical_fill_values:
                continue
            fill_value = fitted.categorical_fill_values[column]
            dtype = frame.schema[column]
            expressions.append(
                _categorical_fill_expr(column, fill_value, dtype).alias(column)
            )

        if not expressions:
            return frame.clone()
        return frame.with_columns(expressions)

    def _build_events(
        self,
        *,
        rows_before: int,
        rows_after: int,
        fitted: _FittedStatistics,
        config: PreprocessorConfig,
    ) -> tuple[PreprocessingEvent, ...]:
        events: list[PreprocessingEvent] = []

        impute_columns: list[str] = []
        if config.numeric_imputation != "none":
            impute_columns.extend(config.numeric_columns)
        if config.categorical_imputation != "none":
            impute_columns.extend(config.categorical_columns)

        if impute_columns:
            events.append(
                PreprocessingEvent(
                    step_name="impute_missing_values",
                    affected_columns=list(impute_columns),
                    rows_before=rows_before,
                    rows_after=rows_after,
                    parameters={
                        "numeric_strategy": config.numeric_imputation,
                        "categorical_strategy": config.categorical_imputation,
                        "numeric_fill_values": dict(fitted.numeric_fill_values),
                        "categorical_fill_values": dict(fitted.categorical_fill_values),
                        "fitted_on_training_data": True,
                    },
                    warnings=[],
                    timestamp=datetime.now(UTC),
                )
            )

        if config.scaling != "none" and config.numeric_columns:
            scaling_warnings = [
                warning
                for warning in fitted.warnings
                if any(column in warning for column in config.numeric_columns)
            ]
            events.append(
                PreprocessingEvent(
                    step_name="scale_numeric_features",
                    affected_columns=list(config.numeric_columns),
                    rows_before=rows_before,
                    rows_after=rows_after,
                    parameters={
                        "strategy": config.scaling,
                        "centers": dict(fitted.scaling_centers),
                        "scales": dict(fitted.scaling_scales),
                        "fitted_on_training_data": True,
                    },
                    warnings=list(scaling_warnings),
                    timestamp=datetime.now(UTC),
                )
            )

        if not events:
            events.append(
                PreprocessingEvent(
                    step_name="preprocess_noop",
                    affected_columns=[],
                    rows_before=rows_before,
                    rows_after=rows_after,
                    parameters={
                        "reason": "No preprocessing operation configured.",
                    },
                    warnings=[],
                    timestamp=datetime.now(UTC),
                )
            )

        return tuple(event.model_copy(deep=True) for event in events)

    def _validate_configured_columns(self, frame: pl.DataFrame) -> None:
        config = self._config
        available = set(frame.columns)
        missing = [
            name
            for name in [*config.numeric_columns, *config.categorical_columns]
            if name not in available
        ]
        if missing:
            raise DataValidationError(
                "Configured columns not found in frame: " + ", ".join(missing)
            )

        for column in config.numeric_columns:
            dtype = frame.schema[column]
            if dtype == pl.Boolean or not dtype.is_numeric():
                raise DataValidationError(
                    f"Column '{column}' must be numeric (Boolean not allowed), "
                    f"got {dtype}"
                )

        for column in config.categorical_columns:
            dtype = frame.schema[column]
            if not _is_categorical_dtype(dtype):
                raise DataValidationError(
                    f"Column '{column}' must be String or a supported categorical "
                    f"dtype, got {dtype}"
                )

    def _validate_numeric_finite(self, frame: pl.DataFrame) -> None:
        for column in self._config.numeric_columns:
            series = frame.get_column(column)
            if not series.dtype.is_float():
                continue

            kinds: list[str] = []
            if bool(series.is_nan().any()):
                kinds.append("NaN")
            if bool((series == float("inf")).any()):
                kinds.append("positive infinity")
            if bool((series == float("-inf")).any()):
                kinds.append("negative infinity")
            if kinds:
                raise DataValidationError(
                    f"Column '{column}' contains non-finite values: "
                    + ", ".join(kinds)
                )

    @staticmethod
    def _require_dataframe(frame: object, *, context: str) -> None:
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(
                f"{context} expects a polars.DataFrame, "
                f"got {type(frame).__name__}"
            )


def _is_categorical_dtype(dtype: pl.DataType) -> bool:
    if dtype == pl.String:
        return True
    base = dtype.base_type()
    return base == pl.Categorical or base == pl.Enum


def _require_python_float(value: object, *, error_message: str) -> float:
    if value is None:
        raise DataValidationError(error_message)
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, Integral):
        return float(int(value))
    if isinstance(value, Real):
        return float(value)
    raise DataValidationError(error_message)


def _most_frequent_value(series: pl.Series, *, column_name: str) -> str:
    non_null = series.drop_nulls()
    if non_null.len() == 0:
        raise DataValidationError(
            f"Cannot compute most-frequent value for all-null categorical "
            f"column '{column_name}'"
        )

    values = non_null.to_list()
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    max_count = max(counts.values())
    for value in values:
        if counts[value] == max_count:
            return str(value)
    raise DataValidationError(
        f"Cannot determine most-frequent value for column '{column_name}'"
    )


def _categorical_fill_expr(
    column: str,
    fill_value: str,
    dtype: pl.DataType,
) -> pl.Expr:
    if isinstance(dtype, pl.Enum):
        categories = set(dtype.categories.to_list())
        if fill_value not in categories:
            return pl.col(column).cast(pl.String).fill_null(fill_value)
    return pl.col(column).fill_null(fill_value)
