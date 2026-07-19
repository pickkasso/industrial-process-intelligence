"""Generic data-quality checks for Polars DataFrames (Step 2C)."""

from __future__ import annotations

from numbers import Real

import polars as pl

from process_intelligence.core.schemas import ValidationIssue
from process_intelligence.data.loader import ORIGINAL_ROW_ID_COLUMN

_ISSUE_EMPTY_DATASET = "EMPTY_DATASET"
_ISSUE_DUPLICATE_ROWS = "DUPLICATE_ROWS"
_ISSUE_ALL_NULL_COLUMN = "ALL_NULL_COLUMN"
_ISSUE_HIGH_MISSING_RATIO = "HIGH_MISSING_RATIO"
_ISSUE_CONSTANT_COLUMN = "CONSTANT_COLUMN"
_ISSUE_NEAR_CONSTANT_COLUMN = "NEAR_CONSTANT_COLUMN"
_ISSUE_NAN_VALUES = "NAN_VALUES"
_ISSUE_INFINITE_VALUES = "INFINITE_VALUES"
_ISSUE_NUMERIC_STRING_COLUMN = "NUMERIC_STRING_COLUMN"

_SEVERITY_ERROR = "ERROR"
_SEVERITY_WARNING = "WARNING"


class DatasetValidator:
    """Run industry-agnostic data-quality checks on a Polars DataFrame."""

    def __init__(
        self,
        high_missing_ratio: float = 0.30,
        near_constant_ratio: float = 0.95,
        numeric_string_ratio: float = 0.80,
    ) -> None:
        """Create a validator with configurable quality thresholds.

        Args:
            high_missing_ratio: Null-ratio threshold for HIGH_MISSING_RATIO.
            near_constant_ratio: Dominant-value ratio threshold for
                NEAR_CONSTANT_COLUMN.
            numeric_string_ratio: Castable-to-float ratio threshold for
                NUMERIC_STRING_COLUMN.

        Raises:
            ValueError: If any threshold is a bool or outside ``[0, 1]``.
        """
        self.high_missing_ratio = _validate_threshold(
            "high_missing_ratio",
            high_missing_ratio,
        )
        self.near_constant_ratio = _validate_threshold(
            "near_constant_ratio",
            near_constant_ratio,
        )
        self.numeric_string_ratio = _validate_threshold(
            "numeric_string_ratio",
            numeric_string_ratio,
        )

    def validate(self, frame: pl.DataFrame) -> list[ValidationIssue]:
        """Validate a Polars DataFrame and return quality issues.

        The input frame is never mutated. Issues are returned in a
        deterministic order: dataset-level checks first, then per-column
        checks in the original column order.

        Args:
            frame: Input dataset in the internal canonical Polars format.

        Returns:
            A list of ``ValidationIssue`` findings. Empty when no issues
            are detected.

        Raises:
            TypeError: If ``frame`` is not a ``polars.DataFrame``.
        """
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(
                "Expected a polars.DataFrame, "
                f"got {type(frame).__name__}"
            )

        if frame.height == 0:
            return [
                ValidationIssue(
                    issue_type=_ISSUE_EMPTY_DATASET,
                    column=None,
                    severity=_SEVERITY_ERROR,
                    message="Dataset has 0 rows and cannot be used for analysis.",
                    suggested_action=(
                        "Provide a non-empty dataset with at least one data row."
                    ),
                )
            ]

        issues: list[ValidationIssue] = []

        duplicate_issue = self._check_duplicate_rows(frame)
        if duplicate_issue is not None:
            issues.append(duplicate_issue)

        for column_name in frame.columns:
            if column_name == ORIGINAL_ROW_ID_COLUMN:
                continue
            issues.extend(self._check_column(frame, column_name))

        return issues

    def _check_duplicate_rows(self, frame: pl.DataFrame) -> ValidationIssue | None:
        data_columns = [
            name for name in frame.columns if name != ORIGINAL_ROW_ID_COLUMN
        ]
        if not data_columns:
            return None

        data_frame = frame.select(data_columns)
        duplicate_count = data_frame.height - data_frame.unique().height
        if duplicate_count < 1:
            return None

        return ValidationIssue(
            issue_type=_ISSUE_DUPLICATE_ROWS,
            column=None,
            severity=_SEVERITY_WARNING,
            message=(
                f"Found {duplicate_count} duplicate excess row(s) when "
                f"ignoring '{ORIGINAL_ROW_ID_COLUMN}'."
            ),
            suggested_action=(
                "Review the duplicate-row criteria and decide whether to "
                "remove or keep the duplicate excess rows."
            ),
        )

    def _check_column(
        self,
        frame: pl.DataFrame,
        column_name: str,
    ) -> list[ValidationIssue]:
        series = frame.get_column(column_name)
        row_count = frame.height
        null_count = int(series.null_count())
        issues: list[ValidationIssue] = []

        if null_count == row_count:
            issues.append(
                ValidationIssue(
                    issue_type=_ISSUE_ALL_NULL_COLUMN,
                    column=column_name,
                    severity=_SEVERITY_ERROR,
                    message=f"Column '{column_name}' contains only null values.",
                    suggested_action=(
                        "Drop the column or investigate why all values are missing."
                    ),
                )
            )
            # All-null columns skip constant / near-constant checks.
            issues.extend(self._check_float_anomalies(series, column_name))
            issues.extend(self._check_numeric_string(series, column_name))
            return issues

        null_ratio = null_count / row_count
        if null_ratio >= self.high_missing_ratio:
            issues.append(
                ValidationIssue(
                    issue_type=_ISSUE_HIGH_MISSING_RATIO,
                    column=column_name,
                    severity=_SEVERITY_WARNING,
                    message=(
                        f"Column '{column_name}' has missing ratio "
                        f"{null_ratio:.4f} (>= threshold {self.high_missing_ratio})."
                    ),
                    suggested_action=(
                        "Impute, drop, or investigate the high missingness "
                        "before modeling."
                    ),
                )
            )

        non_null = series.drop_nulls()
        non_null_unique = int(non_null.n_unique())
        if non_null_unique == 1:
            issues.append(
                ValidationIssue(
                    issue_type=_ISSUE_CONSTANT_COLUMN,
                    column=column_name,
                    severity=_SEVERITY_WARNING,
                    message=(
                        f"Column '{column_name}' has only one distinct "
                        "non-null value."
                    ),
                    suggested_action=(
                        "Consider dropping the constant column or verifying "
                        "that a single value is expected."
                    ),
                )
            )
        elif non_null_unique >= 2:
            non_null_count = non_null.len()
            count_values = non_null.value_counts().get_column("count").to_list()
            dominant_count = max(count_values)
            dominant_ratio = dominant_count / non_null_count
            if dominant_ratio >= self.near_constant_ratio:
                issues.append(
                    ValidationIssue(
                        issue_type=_ISSUE_NEAR_CONSTANT_COLUMN,
                        column=column_name,
                        severity=_SEVERITY_WARNING,
                        message=(
                            f"Column '{column_name}' has dominant-value ratio "
                            f"{dominant_ratio:.4f} "
                            f"(>= threshold {self.near_constant_ratio})."
                        ),
                        suggested_action=(
                            "Review whether the near-constant column is useful "
                            "as a modeling feature."
                        ),
                    )
                )

        issues.extend(self._check_float_anomalies(series, column_name))
        issues.extend(self._check_numeric_string(series, column_name))
        return issues

    def _check_float_anomalies(
        self,
        series: pl.Series,
        column_name: str,
    ) -> list[ValidationIssue]:
        if not series.dtype.is_float():
            return []

        issues: list[ValidationIssue] = []
        nan_count = int(series.is_nan().sum())
        if nan_count >= 1:
            issues.append(
                ValidationIssue(
                    issue_type=_ISSUE_NAN_VALUES,
                    column=column_name,
                    severity=_SEVERITY_WARNING,
                    message=(
                        f"Column '{column_name}' contains {nan_count} NaN value(s)."
                    ),
                    suggested_action=(
                        "Replace or remove NaN values; they are distinct from "
                        "Polars nulls."
                    ),
                )
            )

        infinite_count = int(series.is_infinite().sum())
        if infinite_count >= 1:
            issues.append(
                ValidationIssue(
                    issue_type=_ISSUE_INFINITE_VALUES,
                    column=column_name,
                    severity=_SEVERITY_ERROR,
                    message=(
                        f"Column '{column_name}' contains "
                        f"{infinite_count} infinite value(s)."
                    ),
                    suggested_action=(
                        "Remove or correct positive and negative infinity values "
                        "before modeling."
                    ),
                )
            )
        return issues

    def _check_numeric_string(
        self,
        series: pl.Series,
        column_name: str,
    ) -> list[ValidationIssue]:
        if series.dtype != pl.String and series.dtype != pl.Utf8:
            return []

        stripped = series.str.strip_chars()
        valid = stripped.filter(stripped.is_not_null() & (stripped != ""))
        if valid.len() == 0:
            return []

        casted = valid.cast(pl.Float64, strict=False)
        convertible_count = int(casted.is_not_null().sum())
        convertible_ratio = convertible_count / valid.len()
        if convertible_ratio < self.numeric_string_ratio:
            return []

        return [
            ValidationIssue(
                issue_type=_ISSUE_NUMERIC_STRING_COLUMN,
                column=column_name,
                severity=_SEVERITY_WARNING,
                message=(
                    f"Column '{column_name}' has numeric-like string ratio "
                    f"{convertible_ratio:.4f} "
                    f"(>= threshold {self.numeric_string_ratio})."
                ),
                suggested_action=(
                    "Confirm whether the string column should be cast to a "
                    "numeric type during preprocessing."
                ),
            )
        ]


def _validate_threshold(name: str, value: object) -> float:
    """Validate that a threshold is a non-bool real number in ``[0, 1]``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"{name} must be a real number in [0, 1] (bool not allowed), "
            f"got {value!r}"
        )
    ratio = float(value)
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {ratio}")
    return ratio
