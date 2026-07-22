"""Explicit numeric operating-cohort filtering (Step 11B.10).

Applies a user-confirmed inclusive/exclusive numeric range to the
preprocessed-input analysis frame after SORT and before PREPROCESS/SPLIT.
Does not mutate the input frame, invent bounds, or auto-select SOC/RSOC
columns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

from process_intelligence.core.exceptions import DataValidationError
from process_intelligence.data.loader import ORIGINAL_ROW_ID_COLUMN
from process_intelligence.workflow.schemas import NumericCohortFilter


@dataclass(frozen=True, slots=True)
class NumericCohortFilterOutcome:
    """Immutable result of applying a numeric operating cohort filter."""

    frame: pl.DataFrame
    source_row_count: int
    retained_row_count: int
    excluded_row_count: int
    null_excluded_count: int
    filter: NumericCohortFilter

    def metadata(self) -> dict[str, str | int | float | bool | None]:
        """Return JSON-safe scalar metadata for stage / report recording."""
        cohort_filter = self.filter
        return {
            "cohort_filter_configured": True,
            "cohort_filter_column": cohort_filter.column_name,
            "cohort_filter_lower_bound": cohort_filter.lower_bound,
            "cohort_filter_upper_bound": cohort_filter.upper_bound,
            "cohort_filter_include_lower": cohort_filter.include_lower,
            "cohort_filter_include_upper": cohort_filter.include_upper,
            "cohort_filter_exclude_column_from_features": (
                cohort_filter.exclude_filter_column_from_features
            ),
            "source_row_count": self.source_row_count,
            "retained_row_count": self.retained_row_count,
            "excluded_row_count": self.excluded_row_count,
            "null_excluded_count": self.null_excluded_count,
        }


def apply_numeric_cohort_filter(
    frame: pl.DataFrame,
    cohort_filter: NumericCohortFilter,
) -> NumericCohortFilterOutcome:
    """Filter ``frame`` to the user-confirmed numeric operating cohort.

    Null values in the filter column are excluded. The input frame is not
    mutated. Original row identifiers and relative row order are preserved.

    Args:
        frame: Analysis-order frame with original preprocessed-input values.
        cohort_filter: Validated numeric cohort filter configuration.

    Returns:
        Immutable outcome with the retained frame and filtering counts.

    Raises:
        TypeError: If ``frame`` or ``cohort_filter`` has an unexpected type.
        DataValidationError: If the filter column is missing, non-numeric,
            all-null, or retains zero rows.
    """
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"frame must be a polars.DataFrame, got {type(frame).__name__}")
    if not isinstance(cohort_filter, NumericCohortFilter):
        raise TypeError(
            "cohort_filter must be NumericCohortFilter, "
            f"got {type(cohort_filter).__name__}"
        )

    column_name = cohort_filter.column_name
    if column_name not in frame.columns:
        raise DataValidationError(
            f"Cohort filter column '{column_name}' was not found in the dataset."
        )

    series = frame.get_column(column_name)
    source_row_count = frame.height
    null_excluded_count = int(series.null_count())
    if null_excluded_count == source_row_count:
        raise DataValidationError(
            f"Cohort filter column '{column_name}' is all-null and cannot define "
            "an operating cohort."
        )
    if not bool(series.dtype.is_numeric()) or series.dtype == pl.Boolean:
        raise DataValidationError(
            f"Cohort filter column '{column_name}' must be numeric."
        )

    non_null = series.drop_nulls()
    if non_null.len() == 0:
        raise DataValidationError(
            f"Cohort filter column '{column_name}' is all-null and cannot define "
            "an operating cohort."
        )

    lower = cohort_filter.lower_bound
    upper = cohort_filter.upper_bound
    if cohort_filter.include_lower:
        lower_expr = pl.col(column_name) >= lower
    else:
        lower_expr = pl.col(column_name) > lower
    if cohort_filter.include_upper:
        upper_expr = pl.col(column_name) <= upper
    else:
        upper_expr = pl.col(column_name) < upper

    # Nulls fail numeric comparisons and are therefore excluded without coercion.
    retained = frame.filter(lower_expr & upper_expr)
    retained_row_count = retained.height
    if retained_row_count == 0:
        raise DataValidationError(
            "The configured cohort filter retained zero rows."
        )

    excluded_row_count = source_row_count - retained_row_count
    return NumericCohortFilterOutcome(
        frame=retained,
        source_row_count=source_row_count,
        retained_row_count=retained_row_count,
        excluded_row_count=excluded_row_count,
        null_excluded_count=null_excluded_count,
        filter=cohort_filter.model_copy(deep=True),
    )


def preview_numeric_cohort_filter_row_count(
    frame: pl.DataFrame,
    cohort_filter: NumericCohortFilter,
) -> int:
    """Return retained row count for UI preview without mutating ``frame``.

    Raises the same structured validation errors as
    ``apply_numeric_cohort_filter`` for invalid columns. Zero retained rows is
    returned as ``0`` for preview rather than raising, so the UI can warn.
    """
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"frame must be a polars.DataFrame, got {type(frame).__name__}")
    if not isinstance(cohort_filter, NumericCohortFilter):
        raise TypeError(
            "cohort_filter must be NumericCohortFilter, "
            f"got {type(cohort_filter).__name__}"
        )

    column_name = cohort_filter.column_name
    if column_name not in frame.columns:
        raise DataValidationError(
            f"Cohort filter column '{column_name}' was not found in the dataset."
        )
    series = frame.get_column(column_name)
    if int(series.null_count()) == frame.height:
        raise DataValidationError(
            f"Cohort filter column '{column_name}' is all-null and cannot define "
            "an operating cohort."
        )
    if not bool(series.dtype.is_numeric()) or series.dtype == pl.Boolean:
        raise DataValidationError(
            f"Cohort filter column '{column_name}' must be numeric."
        )

    lower = cohort_filter.lower_bound
    upper = cohort_filter.upper_bound
    if cohort_filter.include_lower:
        lower_expr = pl.col(column_name) >= lower
    else:
        lower_expr = pl.col(column_name) > lower
    if cohort_filter.include_upper:
        upper_expr = pl.col(column_name) <= upper
    else:
        upper_expr = pl.col(column_name) < upper
    return int(frame.filter(lower_expr & upper_expr).height)


def list_numeric_cohort_filter_candidates(
    frame: pl.DataFrame,
    *,
    active_feature_columns: list[str],
    identifier_columns: list[str] | None = None,
    timestamp_column: str | None = None,
) -> list[str]:
    """Return numeric, non-constant, non-all-null cohort column candidates.

    Candidates are drawn from ``active_feature_columns`` that exist on
    ``frame``, excluding identifier and timestamp columns. Constant and
    all-null columns are omitted. No SOC/RSOC name auto-selection is applied.
    """
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"frame must be a polars.DataFrame, got {type(frame).__name__}")
    if not isinstance(active_feature_columns, list):
        raise TypeError(
            "active_feature_columns must be a list[str], "
            f"got {type(active_feature_columns).__name__}"
        )

    blocked: set[str] = set()
    if identifier_columns is not None:
        blocked.update(identifier_columns)
    if timestamp_column is not None:
        blocked.add(timestamp_column)
    blocked.add(ORIGINAL_ROW_ID_COLUMN)

    candidates: list[str] = []
    seen: set[str] = set()
    for name in active_feature_columns:
        if not isinstance(name, str) or name == "" or name.strip() == "":
            continue
        if name in seen or name in blocked or name not in frame.columns:
            continue
        seen.add(name)
        series = frame.get_column(name)
        if not bool(series.dtype.is_numeric()) or series.dtype == pl.Boolean:
            continue
        non_null = series.drop_nulls()
        if non_null.len() == 0:
            continue
        if int(non_null.n_unique()) < 2:
            continue
        candidates.append(name)
    return candidates


def observed_numeric_range(
    frame: pl.DataFrame,
    column_name: str,
) -> tuple[float, float] | None:
    """Return finite observed min/max for caption display, or ``None``.

    Does not mutate ``frame`` and never invents default bounds for submission.
    """
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"frame must be a polars.DataFrame, got {type(frame).__name__}")
    if not isinstance(column_name, str) or column_name.strip() == "":
        raise ValueError("column_name must be a non-empty string")
    if column_name not in frame.columns:
        return None
    series = frame.get_column(column_name)
    if not bool(series.dtype.is_numeric()) or series.dtype == pl.Boolean:
        return None
    non_null = series.drop_nulls()
    if non_null.len() == 0:
        return None
    minimum = non_null.min()
    maximum = non_null.max()
    if not isinstance(minimum, (int, float)) or isinstance(minimum, bool):
        return None
    if not isinstance(maximum, (int, float)) or isinstance(maximum, bool):
        return None
    min_value = float(minimum)
    max_value = float(maximum)
    if not math.isfinite(min_value) or not math.isfinite(max_value):
        return None
    return min_value, max_value
