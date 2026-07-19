"""Lightweight dataset profiling for Polars DataFrames."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import polars as pl
from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    """Per-column summary statistics produced by lightweight profiling."""

    name: str
    dtype: str
    null_count: int = Field(ge=0)
    null_ratio: float = Field(ge=0.0, le=1.0)
    unique_count: int = Field(ge=0)
    cardinality_ratio: float = Field(ge=0.0, le=1.0)
    is_constant: bool
    minimum: float | int | None = None
    maximum: float | int | None = None
    mean: float | None = None
    sample_values: list[Any] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    """Dataset-level profile combining size, column stats, and a row preview."""

    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    columns: list[ColumnProfile] = Field(default_factory=list)
    preview_records: list[dict[str, Any]] = Field(default_factory=list)


class DatasetProfiler:
    """Compute a lightweight profile of a Polars DataFrame without mutating it."""

    def __init__(
        self,
        preview_rows: int = 20,
        sample_values_per_column: int = 5,
    ) -> None:
        if preview_rows < 1:
            raise ValueError(f"preview_rows must be >= 1, got {preview_rows}")
        if sample_values_per_column < 1:
            raise ValueError(
                "sample_values_per_column must be >= 1, "
                f"got {sample_values_per_column}"
            )
        self._preview_rows = preview_rows
        self._sample_values_per_column = sample_values_per_column

    def profile(self, frame: pl.DataFrame) -> DatasetProfile:
        """Profile a Polars DataFrame and return structured summary statistics.

        Args:
            frame: Input dataset in the internal canonical Polars format.

        Returns:
            A ``DatasetProfile`` describing size, per-column stats, and preview.

        Raises:
            TypeError: If ``frame`` is not a ``polars.DataFrame``.
        """
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(
                "Expected a polars.DataFrame, "
                f"got {type(frame).__name__}"
            )

        row_count = frame.height
        column_count = frame.width
        columns = [
            self._profile_column(frame, column_name, row_count)
            for column_name in frame.columns
        ]
        preview_records = frame.head(self._preview_rows).to_dicts()

        return DatasetProfile(
            row_count=row_count,
            column_count=column_count,
            columns=columns,
            preview_records=preview_records,
        )

    def _profile_column(
        self,
        frame: pl.DataFrame,
        column_name: str,
        row_count: int,
    ) -> ColumnProfile:
        series = frame.get_column(column_name)
        dtype = series.dtype
        null_count = int(series.null_count())
        unique_count = int(series.n_unique())

        if row_count == 0:
            null_ratio = 0.0
            cardinality_ratio = 0.0
            is_constant = False
        else:
            null_ratio = null_count / row_count
            cardinality_ratio = unique_count / row_count
            is_constant = unique_count <= 1

        minimum: float | int | None = None
        maximum: float | int | None = None
        mean: float | None = None

        if dtype.is_numeric():
            minimum = _to_python_number(series.min())
            maximum = _to_python_number(series.max())
            mean = _to_python_float(series.mean())

        sample_values = series.head(self._sample_values_per_column).to_list()

        return ColumnProfile(
            name=column_name,
            dtype=str(dtype),
            null_count=null_count,
            null_ratio=null_ratio,
            unique_count=unique_count,
            cardinality_ratio=cardinality_ratio,
            is_constant=is_constant,
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            sample_values=sample_values,
        )


def _to_python_number(value: object) -> float | int | None:
    """Convert a Polars scalar aggregate to a Python int, float, or None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    raise TypeError(f"Unsupported numeric aggregate type: {type(value)!r}")


def _to_python_float(value: object) -> float | None:
    """Convert a Polars mean aggregate to a Python float or None."""
    number = _to_python_number(value)
    if number is None:
        return None
    return float(number)
