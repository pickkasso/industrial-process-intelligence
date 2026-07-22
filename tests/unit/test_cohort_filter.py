"""Unit tests for explicit numeric operating cohort filtering (Step 11B.10)."""

from __future__ import annotations

import json

import polars as pl
import pytest

from process_intelligence.core.exceptions import DataValidationError
from process_intelligence.data.loader import ORIGINAL_ROW_ID_COLUMN
from process_intelligence.workflow import (
    NumericCohortFilter,
    apply_numeric_cohort_filter,
    list_numeric_cohort_filter_candidates,
    preview_numeric_cohort_filter_row_count,
)


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            ORIGINAL_ROW_ID_COLUMN: [10, 11, 12, 13, 14, 15],
            "RSOCavg": [95.0, 90.0, None, 85.0, 50.0, 40.0],
            "pressure": [1, 2, 3, 4, 5, 6],
            "label": ["a", "b", "c", "d", "e", "f"],
            "all_null": [None, None, None, None, None, None],
        }
    )


def test_inclusive_range() -> None:
    frame = _frame()
    outcome = apply_numeric_cohort_filter(
        frame,
        NumericCohortFilter(
            column_name="RSOCavg",
            lower_bound=85.0,
            upper_bound=95.0,
            include_lower=True,
            include_upper=True,
        ),
    )
    assert outcome.retained_row_count == 3
    assert outcome.frame.get_column("RSOCavg").to_list() == [95.0, 90.0, 85.0]


def test_exclusive_range() -> None:
    outcome = apply_numeric_cohort_filter(
        _frame(),
        NumericCohortFilter(
            column_name="RSOCavg",
            lower_bound=85.0,
            upper_bound=95.0,
            include_lower=False,
            include_upper=False,
        ),
    )
    assert outcome.frame.get_column("RSOCavg").to_list() == [90.0]


def test_mixed_inclusive_exclusive() -> None:
    outcome = apply_numeric_cohort_filter(
        _frame(),
        NumericCohortFilter(
            column_name="RSOCavg",
            lower_bound=85.0,
            upper_bound=95.0,
            include_lower=True,
            include_upper=False,
        ),
    )
    assert outcome.frame.get_column("RSOCavg").to_list() == [90.0, 85.0]


def test_null_rows_excluded() -> None:
    outcome = apply_numeric_cohort_filter(
        _frame(),
        NumericCohortFilter(
            column_name="RSOCavg",
            lower_bound=0.0,
            upper_bound=100.0,
        ),
    )
    assert outcome.null_excluded_count == 1
    assert None not in outcome.frame.get_column("RSOCavg").to_list()


def test_integer_and_float_numeric() -> None:
    int_outcome = apply_numeric_cohort_filter(
        _frame(),
        NumericCohortFilter(column_name="pressure", lower_bound=2, upper_bound=4),
    )
    assert int_outcome.retained_row_count == 3
    float_outcome = apply_numeric_cohort_filter(
        _frame(),
        NumericCohortFilter(column_name="RSOCavg", lower_bound=40.0, upper_bound=50.0),
    )
    assert float_outcome.retained_row_count == 2


def test_missing_column_refused() -> None:
    with pytest.raises(DataValidationError, match="not found"):
        apply_numeric_cohort_filter(
            _frame(),
            NumericCohortFilter(column_name="missing", lower_bound=0.0, upper_bound=1.0),
        )


def test_nonnumeric_column_refused() -> None:
    with pytest.raises(DataValidationError, match="numeric"):
        apply_numeric_cohort_filter(
            _frame(),
            NumericCohortFilter(column_name="label", lower_bound=0.0, upper_bound=1.0),
        )


def test_all_null_column_refused() -> None:
    with pytest.raises(DataValidationError, match="all-null"):
        apply_numeric_cohort_filter(
            _frame(),
            NumericCohortFilter(
                column_name="all_null",
                lower_bound=0.0,
                upper_bound=1.0,
            ),
        )


def test_zero_retained_rows_refused() -> None:
    with pytest.raises(DataValidationError, match="zero rows"):
        apply_numeric_cohort_filter(
            _frame(),
            NumericCohortFilter(
                column_name="RSOCavg",
                lower_bound=200.0,
                upper_bound=300.0,
            ),
        )


def test_original_row_id_and_order_preserved() -> None:
    frame = _frame()
    outcome = apply_numeric_cohort_filter(
        frame,
        NumericCohortFilter(
            column_name="RSOCavg",
            lower_bound=85.0,
            upper_bound=95.0,
        ),
    )
    assert outcome.frame.get_column(ORIGINAL_ROW_ID_COLUMN).to_list() == [10, 11, 13]


def test_input_frame_immutable_and_deterministic() -> None:
    frame = _frame()
    before = frame.to_dicts()
    cohort = NumericCohortFilter(
        column_name="RSOCavg",
        lower_bound=85.0,
        upper_bound=95.0,
    )
    first = apply_numeric_cohort_filter(frame, cohort)
    second = apply_numeric_cohort_filter(frame, cohort)
    assert frame.to_dicts() == before
    assert first.frame.to_dicts() == second.frame.to_dicts()
    assert first.metadata() == second.metadata()


def test_json_safe_metadata() -> None:
    outcome = apply_numeric_cohort_filter(
        _frame(),
        NumericCohortFilter(
            column_name="RSOCavg",
            lower_bound=85.0,
            upper_bound=95.0,
        ),
    )
    encoded = json.dumps(outcome.metadata())
    restored = json.loads(encoded)
    assert restored["retained_row_count"] == 3
    assert restored["cohort_filter_column"] == "RSOCavg"


def test_preview_zero_rows_returns_zero() -> None:
    count = preview_numeric_cohort_filter_row_count(
        _frame(),
        NumericCohortFilter(
            column_name="RSOCavg",
            lower_bound=200.0,
            upper_bound=300.0,
        ),
    )
    assert count == 0


def test_candidate_list_excludes_constant_identifier_timestamp() -> None:
    frame = pl.DataFrame(
        {
            "RSOCavg": [10.0, 20.0, 30.0],
            "SOH": [0.0, 0.0, 0.0],
            "SerialNumber": [1, 2, 3],
            "Date": [1.0, 2.0, 3.0],
            "pressure": [1.0, 2.0, 3.0],
        }
    )
    candidates = list_numeric_cohort_filter_candidates(
        frame,
        active_feature_columns=["RSOCavg", "SOH", "SerialNumber", "Date", "pressure"],
        identifier_columns=["SerialNumber"],
        timestamp_column="Date",
    )
    assert candidates == ["RSOCavg", "pressure"]
    assert "SOH" not in candidates
