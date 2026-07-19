"""Unit tests for DatasetValidator (Step 2C)."""

from __future__ import annotations

import math

import polars as pl
import pytest

from process_intelligence.data import (
    ORIGINAL_ROW_ID_COLUMN,
    DatasetValidator,
)

_KNOWN_ISSUE_TYPES = {
    "EMPTY_DATASET",
    "DUPLICATE_ROWS",
    "ALL_NULL_COLUMN",
    "HIGH_MISSING_RATIO",
    "CONSTANT_COLUMN",
    "NEAR_CONSTANT_COLUMN",
    "NAN_VALUES",
    "INFINITE_VALUES",
    "NUMERIC_STRING_COLUMN",
}


def test_dataset_validator_default_construction() -> None:
    validator = DatasetValidator()
    assert isinstance(validator, DatasetValidator)
    assert validator.high_missing_ratio == pytest.approx(0.30)
    assert validator.near_constant_ratio == pytest.approx(0.95)
    assert validator.numeric_string_ratio == pytest.approx(0.80)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"high_missing_ratio": -0.01},
        {"near_constant_ratio": -0.1},
        {"numeric_string_ratio": -1.0},
    ],
)
def test_threshold_below_zero_raises_value_error(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        DatasetValidator(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"high_missing_ratio": 1.01},
        {"near_constant_ratio": 1.5},
        {"numeric_string_ratio": 2.0},
    ],
)
def test_threshold_above_one_raises_value_error(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        DatasetValidator(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"high_missing_ratio": True},
        {"near_constant_ratio": False},
        {"numeric_string_ratio": True},
    ],
)
def test_bool_threshold_is_rejected(kwargs: dict[str, bool]) -> None:
    with pytest.raises(ValueError):
        DatasetValidator(**kwargs)


def test_non_polars_dataframe_raises_type_error() -> None:
    validator = DatasetValidator()
    with pytest.raises(TypeError):
        validator.validate({"a": [1, 2]})  # type: ignore[arg-type]


def test_clean_frame_returns_empty_issue_list() -> None:
    frame = pl.DataFrame(
        {
            "sensor_a": [1.0, 2.0, 3.0],
            "label": ["ok", "warn", "ok"],
            ORIGINAL_ROW_ID_COLUMN: [0, 1, 2],
        }
    )
    issues = DatasetValidator().validate(frame)
    assert issues == []


def test_empty_dataframe_produces_empty_dataset_issue() -> None:
    frame = pl.DataFrame()
    issues = DatasetValidator().validate(frame)

    assert len(issues) == 1
    assert issues[0].issue_type == "EMPTY_DATASET"
    assert issues[0].column is None
    assert issues[0].severity == "ERROR"


def test_zero_row_frame_with_columns_is_empty_dataset() -> None:
    frame = pl.DataFrame(schema={"a": pl.Int64, "b": pl.Float64})
    issues = DatasetValidator().validate(frame)

    assert len(issues) == 1
    assert issues[0].issue_type == "EMPTY_DATASET"
    assert issues[0].column is None


def test_empty_dataframe_does_not_emit_column_issues() -> None:
    frame = pl.DataFrame(schema={"all_null": pl.Float64, "const": pl.Int64})
    issues = DatasetValidator().validate(frame)

    assert [issue.issue_type for issue in issues] == ["EMPTY_DATASET"]
    assert all(issue.column is None for issue in issues)


def test_duplicate_rows_are_detected() -> None:
    frame = pl.DataFrame(
        {
            "a": [1, 1, 2],
            "b": [10, 10, 20],
            ORIGINAL_ROW_ID_COLUMN: [0, 1, 2],
        }
    )
    issues = DatasetValidator().validate(frame)
    duplicate_issues = [i for i in issues if i.issue_type == "DUPLICATE_ROWS"]

    assert len(duplicate_issues) == 1
    assert duplicate_issues[0].column is None
    assert duplicate_issues[0].severity == "WARNING"


def test_duplicate_excess_row_count_is_correct() -> None:
    frame = pl.DataFrame(
        {
            "a": [1, 1, 1, 2],
            "b": [10, 10, 10, 20],
        }
    )
    issues = DatasetValidator().validate(frame)
    duplicate_issue = next(i for i in issues if i.issue_type == "DUPLICATE_ROWS")

    assert "2" in duplicate_issue.message


def test_duplicate_detection_ignores_original_row_id() -> None:
    frame = pl.DataFrame(
        {
            "a": [1, 1],
            "b": [10, 10],
            ORIGINAL_ROW_ID_COLUMN: [0, 99],
        }
    )
    issues = DatasetValidator().validate(frame)
    duplicate_issues = [i for i in issues if i.issue_type == "DUPLICATE_ROWS"]

    assert len(duplicate_issues) == 1
    assert "1" in duplicate_issues[0].message


def test_original_row_id_only_frame_skips_duplicate_check() -> None:
    frame = pl.DataFrame({ORIGINAL_ROW_ID_COLUMN: [0, 0, 1]})
    issues = DatasetValidator().validate(frame)

    assert all(issue.issue_type != "DUPLICATE_ROWS" for issue in issues)


def test_all_null_column_is_detected() -> None:
    frame = pl.DataFrame(
        {
            "empty_col": [None, None, None],
            "ok": [1, 2, 3],
        }
    )
    issues = DatasetValidator().validate(frame)
    all_null = [i for i in issues if i.issue_type == "ALL_NULL_COLUMN"]

    assert len(all_null) == 1
    assert all_null[0].column == "empty_col"
    assert all_null[0].severity == "ERROR"


def test_all_null_column_does_not_also_emit_high_missing() -> None:
    frame = pl.DataFrame({"empty_col": [None, None, None]})
    issues = DatasetValidator().validate(frame)

    assert any(i.issue_type == "ALL_NULL_COLUMN" for i in issues)
    assert all(i.issue_type != "HIGH_MISSING_RATIO" for i in issues)


def test_high_missing_ratio_is_detected() -> None:
    frame = pl.DataFrame(
        {
            "sparse": [1.0, None, None, None, None, None, None, None, None, None],
            "ok": list(range(10)),
        }
    )
    issues = DatasetValidator(high_missing_ratio=0.30).validate(frame)
    high_missing = [i for i in issues if i.issue_type == "HIGH_MISSING_RATIO"]

    assert len(high_missing) == 1
    assert high_missing[0].column == "sparse"
    assert high_missing[0].severity == "WARNING"
    assert "0.9" in high_missing[0].message or "0.90" in high_missing[0].message


def test_high_missing_ratio_includes_boundary() -> None:
    frame = pl.DataFrame(
        {
            "border": [1, None, None, None, None, None, None, None, None, None],
        }
    )
    issues = DatasetValidator(high_missing_ratio=0.90).validate(frame)
    high_missing = [i for i in issues if i.issue_type == "HIGH_MISSING_RATIO"]

    assert len(high_missing) == 1
    assert high_missing[0].column == "border"
    assert "0.9" in high_missing[0].message


def test_missing_ratio_below_threshold_is_not_flagged() -> None:
    frame = pl.DataFrame(
        {
            "almost_full": [1, 2, 3, 4, 5, 6, 7, 8, 9, None],
        }
    )
    issues = DatasetValidator(high_missing_ratio=0.30).validate(frame)

    assert all(i.issue_type != "HIGH_MISSING_RATIO" for i in issues)


def test_nan_is_not_counted_in_null_ratio() -> None:
    frame = pl.DataFrame(
        {
            "mixed": [
                1.0,
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
            ],
        }
    )
    issues = DatasetValidator(high_missing_ratio=0.30).validate(frame)

    assert all(i.issue_type != "HIGH_MISSING_RATIO" for i in issues)
    assert any(i.issue_type == "NAN_VALUES" for i in issues)


def test_constant_column_is_detected() -> None:
    frame = pl.DataFrame({"const": [7, 7, 7, 7], "ok": [1, 2, 3, 4]})
    issues = DatasetValidator().validate(frame)
    constant = [i for i in issues if i.issue_type == "CONSTANT_COLUMN"]

    assert len(constant) == 1
    assert constant[0].column == "const"
    assert constant[0].severity == "WARNING"


def test_constant_column_with_partial_nulls() -> None:
    frame = pl.DataFrame({"const_nulls": [5, None, 5, None, 5]})
    issues = DatasetValidator().validate(frame)
    constant = [i for i in issues if i.issue_type == "CONSTANT_COLUMN"]

    assert len(constant) == 1
    assert constant[0].column == "const_nulls"


def test_near_constant_column_is_detected() -> None:
    values = [1] * 19 + [2]
    frame = pl.DataFrame({"almost": values})
    issues = DatasetValidator(near_constant_ratio=0.95).validate(frame)
    near = [i for i in issues if i.issue_type == "NEAR_CONSTANT_COLUMN"]

    assert len(near) == 1
    assert near[0].column == "almost"
    assert near[0].severity == "WARNING"
    assert "0.95" in near[0].message


def test_near_constant_ratio_includes_boundary() -> None:
    values = [1] * 19 + [2]
    frame = pl.DataFrame({"border": values})
    issues = DatasetValidator(near_constant_ratio=0.95).validate(frame)
    near = [i for i in issues if i.issue_type == "NEAR_CONSTANT_COLUMN"]

    assert len(near) == 1
    dominant_ratio = 19 / 20
    assert dominant_ratio == pytest.approx(0.95)
    assert "0.95" in near[0].message


def test_near_constant_below_threshold_not_flagged() -> None:
    values = [1] * 18 + [2, 3]
    frame = pl.DataFrame({"varied": values})
    issues = DatasetValidator(near_constant_ratio=0.95).validate(frame)

    assert all(i.issue_type != "NEAR_CONSTANT_COLUMN" for i in issues)


def test_constant_and_near_constant_are_mutually_exclusive() -> None:
    frame = pl.DataFrame({"const": [1, 1, 1, None]})
    issues = DatasetValidator().validate(frame)
    types = {i.issue_type for i in issues if i.column == "const"}

    assert "CONSTANT_COLUMN" in types
    assert "NEAR_CONSTANT_COLUMN" not in types


def test_float_nan_is_detected() -> None:
    frame = pl.DataFrame({"x": [1.0, float("nan"), 2.0]})
    issues = DatasetValidator().validate(frame)
    nan_issues = [i for i in issues if i.issue_type == "NAN_VALUES"]

    assert len(nan_issues) == 1
    assert nan_issues[0].column == "x"
    assert nan_issues[0].severity == "WARNING"
    assert "1" in nan_issues[0].message


def test_integer_column_skips_nan_check() -> None:
    frame = pl.DataFrame({"ints": [1, 2, 3, None]})
    issues = DatasetValidator().validate(frame)

    assert all(i.issue_type != "NAN_VALUES" for i in issues)


def test_positive_infinity_is_detected() -> None:
    frame = pl.DataFrame({"x": [1.0, math.inf, 2.0]})
    issues = DatasetValidator().validate(frame)
    inf_issues = [i for i in issues if i.issue_type == "INFINITE_VALUES"]

    assert len(inf_issues) == 1
    assert inf_issues[0].column == "x"
    assert inf_issues[0].severity == "ERROR"
    assert "1" in inf_issues[0].message


def test_negative_infinity_is_detected() -> None:
    frame = pl.DataFrame({"x": [1.0, -math.inf, 2.0]})
    issues = DatasetValidator().validate(frame)
    inf_issues = [i for i in issues if i.issue_type == "INFINITE_VALUES"]

    assert len(inf_issues) == 1
    assert inf_issues[0].severity == "ERROR"


def test_nan_and_infinity_issues_are_independent() -> None:
    frame = pl.DataFrame({"x": [1.0, float("nan"), math.inf, -math.inf]})
    issues = DatasetValidator().validate(frame)
    types = {i.issue_type for i in issues if i.column == "x"}

    assert "NAN_VALUES" in types
    assert "INFINITE_VALUES" in types
    nan_issue = next(i for i in issues if i.issue_type == "NAN_VALUES")
    inf_issue = next(i for i in issues if i.issue_type == "INFINITE_VALUES")
    assert "1" in nan_issue.message
    assert "2" in inf_issue.message


def test_numeric_string_column_is_detected() -> None:
    frame = pl.DataFrame({"vals": ["1", "2.5", "3", "4", "x"]})
    issues = DatasetValidator(numeric_string_ratio=0.80).validate(frame)
    numeric = [i for i in issues if i.issue_type == "NUMERIC_STRING_COLUMN"]

    assert len(numeric) == 1
    assert numeric[0].column == "vals"
    assert numeric[0].severity == "WARNING"


def test_numeric_string_strips_whitespace() -> None:
    frame = pl.DataFrame({"vals": [" 1 ", " 2.0", "3 ", "4", "5"]})
    issues = DatasetValidator(numeric_string_ratio=0.80).validate(frame)
    numeric = [i for i in issues if i.issue_type == "NUMERIC_STRING_COLUMN"]

    assert len(numeric) == 1
    assert numeric[0].column == "vals"


def test_null_and_blank_excluded_from_numeric_string_denominator() -> None:
    frame = pl.DataFrame(
        {
            "vals": ["1", "2", "3", "4", None, "", "  ", "text"],
        }
    )
    # Valid non-blank strings: 1,2,3,4,text -> 4/5 = 0.8
    issues = DatasetValidator(numeric_string_ratio=0.80).validate(frame)
    numeric = [i for i in issues if i.issue_type == "NUMERIC_STRING_COLUMN"]

    assert len(numeric) == 1
    assert "0.8" in numeric[0].message


def test_numeric_string_ratio_includes_boundary() -> None:
    frame = pl.DataFrame({"vals": ["1", "2", "3", "4", "nope"]})
    issues = DatasetValidator(numeric_string_ratio=0.80).validate(frame)
    numeric = [i for i in issues if i.issue_type == "NUMERIC_STRING_COLUMN"]

    assert len(numeric) == 1
    assert 4 / 5 == pytest.approx(0.80)


def test_low_numeric_string_ratio_not_flagged() -> None:
    frame = pl.DataFrame({"vals": ["1", "2", "abc", "def", "ghi"]})
    issues = DatasetValidator(numeric_string_ratio=0.80).validate(frame)

    assert all(i.issue_type != "NUMERIC_STRING_COLUMN" for i in issues)


def test_plain_text_column_not_flagged_as_numeric_string() -> None:
    frame = pl.DataFrame({"notes": ["alpha", "beta", "gamma", "delta"]})
    issues = DatasetValidator().validate(frame)

    assert all(i.issue_type != "NUMERIC_STRING_COLUMN" for i in issues)


def test_original_row_id_excluded_from_column_issues() -> None:
    frame = pl.DataFrame(
        {
            "ok": [1, 2, 3],
            ORIGINAL_ROW_ID_COLUMN: [0, 0, 0],
        }
    )
    issues = DatasetValidator().validate(frame)

    assert all(issue.column != ORIGINAL_ROW_ID_COLUMN for issue in issues)
    assert all(
        issue.issue_type
        not in {
            "CONSTANT_COLUMN",
            "NEAR_CONSTANT_COLUMN",
            "ALL_NULL_COLUMN",
            "HIGH_MISSING_RATIO",
            "NAN_VALUES",
            "INFINITE_VALUES",
            "NUMERIC_STRING_COLUMN",
        }
        or issue.column != ORIGINAL_ROW_ID_COLUMN
        for issue in issues
    )


def test_issue_types_match_defined_strings() -> None:
    frame = pl.DataFrame(
        {
            "all_null": [None, None],
            "const": [1, 1],
            "nan_col": [1.0, float("nan")],
            "inf_col": [1.0, math.inf],
            "num_str": ["1", "2"],
            "a": [1, 1],
            "b": [10, 10],
        }
    )
    issues = DatasetValidator(numeric_string_ratio=0.80).validate(frame)

    for issue in issues:
        assert issue.issue_type in _KNOWN_ISSUE_TYPES


def test_severity_is_error_or_warning() -> None:
    frame = pl.DataFrame(
        {
            "all_null": [None, None],
            "dup_a": [1, 1],
            "inf_col": [1.0, math.inf],
        }
    )
    issues = DatasetValidator().validate(frame)

    for issue in issues:
        assert issue.severity in {"ERROR", "WARNING"}


def test_all_messages_are_non_empty() -> None:
    frame = pl.DataFrame({"all_null": [None, None], "const": [1, 1]})
    issues = DatasetValidator().validate(frame)

    assert issues
    assert all(issue.message.strip() for issue in issues)


def test_all_suggested_actions_are_non_empty() -> None:
    frame = pl.DataFrame({"all_null": [None, None], "const": [1, 1]})
    issues = DatasetValidator().validate(frame)

    assert issues
    assert all(
        issue.suggested_action is not None and issue.suggested_action.strip()
        for issue in issues
    )


def test_issue_return_order_matches_specification() -> None:
    frame = pl.DataFrame(
        {
            "c_nan": [1.0, float("nan"), 1.0, 1.0, 1.0],
            "a_null": [None, None, None, None, None],
            "b_dup": [1, 1, 1, 1, 1],
        }
    )
    # Add a duplicate row for DUPLICATE_ROWS
    frame = pl.concat([frame, frame.head(1)])
    issues = DatasetValidator().validate(frame)
    types = [issue.issue_type for issue in issues]

    assert types[0] == "DUPLICATE_ROWS"
    # Column order: c_nan, a_null, b_dup
    assert "NAN_VALUES" in types
    assert types.index("DUPLICATE_ROWS") < types.index("NAN_VALUES")
    assert types.index("NAN_VALUES") < types.index("ALL_NULL_COLUMN")


def test_column_issue_order_follows_original_columns() -> None:
    frame = pl.DataFrame(
        {
            "second": [None, None, None],
            "first": [1, 1, 1],
        }
    )
    issues = DatasetValidator().validate(frame)
    column_issues = [i for i in issues if i.column is not None]

    assert [i.column for i in column_issues] == ["second", "first"]
    assert column_issues[0].issue_type == "ALL_NULL_COLUMN"
    assert column_issues[1].issue_type == "CONSTANT_COLUMN"


def test_validate_preserves_frame_shape() -> None:
    frame = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    before_shape = frame.shape
    DatasetValidator().validate(frame)
    assert frame.shape == before_shape


def test_validate_preserves_frame_columns() -> None:
    frame = pl.DataFrame(
        {
            "a": [1, 2],
            ORIGINAL_ROW_ID_COLUMN: [0, 1],
        }
    )
    before_columns = list(frame.columns)
    DatasetValidator().validate(frame)
    assert list(frame.columns) == before_columns


def test_validate_preserves_frame_data() -> None:
    frame = pl.DataFrame(
        {
            "a": [1, None, 3],
            "b": [1.0, float("nan"), math.inf],
            "c": ["x", "y", "z"],
            ORIGINAL_ROW_ID_COLUMN: [0, 1, 2],
        }
    )
    before = frame.clone()
    DatasetValidator().validate(frame)
    assert frame.equals(before)


def test_consecutive_validations_do_not_share_state() -> None:
    validator = DatasetValidator()
    empty = pl.DataFrame()
    clean = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})

    empty_issues = validator.validate(empty)
    clean_issues = validator.validate(clean)

    assert len(empty_issues) == 1
    assert empty_issues[0].issue_type == "EMPTY_DATASET"
    assert clean_issues == []

    dirty = pl.DataFrame({"const": [1, 1, 1]})
    dirty_issues = validator.validate(dirty)
    assert any(i.issue_type == "CONSTANT_COLUMN" for i in dirty_issues)
    assert validator.validate(clean) == []


def test_per_column_check_order_within_column() -> None:
    # One float column that is near-constant, has high missing, NaN, and Inf
    # null_ratio = 3/10 = 0.3 -> HIGH_MISSING at default 0.30
    # non-null values: 1.0, 1.0, 1.0, nan, inf, 2.0 -> unique >= 2, dominant 1.0
    values: list[float | None] = [
        1.0,
        1.0,
        1.0,
        float("nan"),
        math.inf,
        2.0,
        None,
        None,
        None,
        1.0,
    ]
    frame = pl.DataFrame({"mixed": values})
    issues = DatasetValidator(
        high_missing_ratio=0.30,
        near_constant_ratio=0.50,
    ).validate(frame)
    mixed_types = [i.issue_type for i in issues if i.column == "mixed"]

    assert mixed_types == [
        "HIGH_MISSING_RATIO",
        "NEAR_CONSTANT_COLUMN",
        "NAN_VALUES",
        "INFINITE_VALUES",
    ]
