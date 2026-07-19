"""Stable dataset sorting with preprocessing-event recording (Step 2E)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import polars as pl

from process_intelligence.core.exceptions import DataValidationError
from process_intelligence.core.schemas import PreprocessingEvent


@dataclass(frozen=True, slots=True)
class SortResult:
    """Immutable container pairing a sorted frame with its preprocessing event."""

    frame: pl.DataFrame
    event: PreprocessingEvent


class DatasetSorter:
    """Sort Polars DataFrames without mutating the input or retaining state."""

    def sort(
        self,
        frame: pl.DataFrame,
        *,
        by: str | Sequence[str],
        descending: bool | Sequence[bool] = False,
        nulls_last: bool = True,
    ) -> SortResult:
        """Sort a Polars DataFrame by one or more columns with a stable order.

        The input frame is never mutated. Column names, column order, dtypes,
        and row count are preserved. When ``_original_row_id`` is present its
        values are kept as-is. Equal sort keys retain their relative input
        order via Polars ``maintain_order=True``.

        Args:
            frame: Input dataset as a Polars DataFrame.
            by: Column name or ordered sequence of column names to sort by.
            descending: Single bool applied to all columns, or a per-column
                sequence of bools matching ``by``.
            nulls_last: When ``True``, place nulls last for each sort key;
                when ``False``, use Polars null-first ordering.

        Returns:
            A ``SortResult`` containing the sorted frame and a
            ``PreprocessingEvent`` describing the sort.

        Raises:
            TypeError: If ``frame``, ``by``, ``descending``, or ``nulls_last``
                have an unsupported type.
            DataValidationError: If ``by`` is empty, contains duplicates or
                empty names, references missing columns, or ``descending``
                length does not match ``by``.
        """
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(
                "Expected a polars.DataFrame, "
                f"got {type(frame).__name__}"
            )

        if not isinstance(nulls_last, bool):
            raise TypeError(
                "nulls_last must be a bool, "
                f"got {type(nulls_last).__name__}"
            )

        by_columns = _normalize_by(by)
        descending_flags = _normalize_descending(descending, len(by_columns))
        _validate_columns_exist(frame, by_columns)

        sorted_frame = frame.sort(
            by_columns,
            descending=descending_flags,
            nulls_last=nulls_last,
            maintain_order=True,
        )

        warnings = _build_null_warnings(frame, by_columns, nulls_last)
        row_count = frame.height
        event = PreprocessingEvent(
            step_name="sort_dataset",
            affected_columns=list(by_columns),
            rows_before=row_count,
            rows_after=sorted_frame.height,
            parameters={
                "by": list(by_columns),
                "descending": list(descending_flags),
                "nulls_last": nulls_last,
                "stable": True,
            },
            warnings=warnings,
            timestamp=datetime.now(UTC),
        )
        return SortResult(frame=sorted_frame, event=event)


def _normalize_by(by: str | Sequence[str]) -> list[str]:
    if isinstance(by, str):
        columns = [by]
    elif isinstance(by, (bytes, bytearray)):
        raise TypeError(
            "by must be a str or a sequence of str, "
            f"got {type(by).__name__}"
        )
    elif isinstance(by, Sequence):
        columns = list(by)
    else:
        raise TypeError(
            "by must be a str or a sequence of str, "
            f"got {type(by).__name__}"
        )

    if not columns:
        raise DataValidationError("by must contain at least one column name")

    for index, name in enumerate(columns):
        if isinstance(name, (bytes, bytearray)):
            raise TypeError(
                f"by[{index}] must be a str, got {type(name).__name__}"
            )
        if not isinstance(name, str):
            raise TypeError(
                f"by[{index}] must be a str, got {type(name).__name__}"
            )
        if name == "":
            raise DataValidationError("by column names must be non-empty strings")

    if len(set(columns)) != len(columns):
        raise DataValidationError(
            f"by contains duplicate column names: {columns}"
        )

    return columns


def _normalize_descending(
    descending: bool | Sequence[bool],
    column_count: int,
) -> list[bool]:
    if isinstance(descending, bool):
        return [descending] * column_count

    if isinstance(descending, (str, bytes, bytearray)):
        raise TypeError(
            "descending must be a bool or a sequence of bool, "
            f"got {type(descending).__name__}"
        )

    if not isinstance(descending, Sequence):
        raise TypeError(
            "descending must be a bool or a sequence of bool, "
            f"got {type(descending).__name__}"
        )

    flags = list(descending)
    if len(flags) != column_count:
        raise DataValidationError(
            "descending length must match the number of sort columns: "
            f"expected {column_count}, got {len(flags)}"
        )

    for index, flag in enumerate(flags):
        if not isinstance(flag, bool):
            raise TypeError(
                f"descending[{index}] must be a bool, "
                f"got {type(flag).__name__}"
            )

    return flags


def _validate_columns_exist(frame: pl.DataFrame, columns: Sequence[str]) -> None:
    available = set(frame.columns)
    missing = [name for name in columns if name not in available]
    if missing:
        raise DataValidationError(
            "Sort columns not found in frame: "
            + ", ".join(missing)
        )


def _build_null_warnings(
    frame: pl.DataFrame,
    columns: Sequence[str],
    nulls_last: bool,
) -> list[str]:
    placement = "last" if nulls_last else "first"
    warnings: list[str] = []
    for name in columns:
        null_count = frame.get_column(name).null_count()
        if null_count > 0:
            warnings.append(
                f"Sort column '{name}' contains null values; "
                f"nulls were placed {placement} according to nulls_last={nulls_last}."
            )
    return warnings
