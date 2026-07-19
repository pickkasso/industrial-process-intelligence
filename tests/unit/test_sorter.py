"""Unit tests for DatasetSorter (Step 2E)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
from datetime import UTC

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from process_intelligence.core.exceptions import DataValidationError
from process_intelligence.core.schemas import PreprocessingEvent
from process_intelligence.data import ORIGINAL_ROW_ID_COLUMN, DatasetSorter, SortResult
from process_intelligence.data.loader import ORIGINAL_ROW_ID_COLUMN as LOADER_ROW_ID
from process_intelligence.data.sorter import DatasetSorter as DirectSorter
from process_intelligence.data.sorter import SortResult as DirectSortResult


def _unsorted_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "batch_id": ["B", "A", "B", "A"],
            "timestamp": [2, 1, 1, 2],
            "value": [40, 10, 30, 20],
            ORIGINAL_ROW_ID_COLUMN: [0, 1, 2, 3],
        }
    )


def test_dataset_sorter_construction_succeeds() -> None:
    sorter = DatasetSorter()
    assert isinstance(sorter, DatasetSorter)
    assert isinstance(sorter, DirectSorter)


def test_non_polars_frame_raises_type_error() -> None:
    with pytest.raises(TypeError, match="polars.DataFrame"):
        DatasetSorter().sort({"a": [1]}, by="a")  # type: ignore[arg-type]


def test_by_string_sorts_single_column() -> None:
    frame = pl.DataFrame({"a": [3, 1, 2], "b": ["x", "y", "z"]})
    result = DatasetSorter().sort(frame, by="a")
    assert result.frame["a"].to_list() == [1, 2, 3]
    assert result.frame["b"].to_list() == ["y", "z", "x"]


def test_by_list_sorts_multiple_columns() -> None:
    frame = _unsorted_frame()
    result = DatasetSorter().sort(frame, by=["batch_id", "timestamp"])
    assert result.frame["batch_id"].to_list() == ["A", "A", "B", "B"]
    assert result.frame["timestamp"].to_list() == [1, 2, 1, 2]


def test_empty_by_sequence_rejected() -> None:
    frame = pl.DataFrame({"a": [1]})
    with pytest.raises(DataValidationError, match="at least one"):
        DatasetSorter().sort(frame, by=[])


def test_bytes_by_rejected() -> None:
    frame = pl.DataFrame({"a": [1]})
    with pytest.raises(TypeError, match="by"):
        DatasetSorter().sort(frame, by=b"a")  # type: ignore[arg-type]


def test_by_non_string_element_raises_type_error() -> None:
    frame = pl.DataFrame({"a": [1]})
    with pytest.raises(TypeError, match=r"by\[0\]"):
        DatasetSorter().sort(frame, by=[1])  # type: ignore[list-item]


def test_empty_string_column_name_rejected() -> None:
    frame = pl.DataFrame({"a": [1]})
    with pytest.raises(DataValidationError, match="non-empty"):
        DatasetSorter().sort(frame, by="")


def test_duplicate_sort_columns_rejected() -> None:
    frame = pl.DataFrame({"a": [1, 2]})
    with pytest.raises(DataValidationError, match="duplicate"):
        DatasetSorter().sort(frame, by=["a", "a"])


def test_missing_sort_column_rejected() -> None:
    frame = pl.DataFrame({"a": [1]})
    with pytest.raises(DataValidationError, match="missing_col"):
        DatasetSorter().sort(frame, by="missing_col")


def test_missing_column_name_included_in_error_message() -> None:
    frame = pl.DataFrame({"a": [1]})
    with pytest.raises(DataValidationError, match="not_here"):
        DatasetSorter().sort(frame, by=["a", "not_here"])


def test_descending_bool_applied() -> None:
    frame = pl.DataFrame({"a": [1, 2, 3]})
    result = DatasetSorter().sort(frame, by="a", descending=True)
    assert result.frame["a"].to_list() == [3, 2, 1]


def test_descending_sequence_applied() -> None:
    frame = pl.DataFrame({"a": [1, 1, 2], "b": [3, 1, 2]})
    result = DatasetSorter().sort(
        frame,
        by=["a", "b"],
        descending=[False, True],
    )
    assert result.frame["a"].to_list() == [1, 1, 2]
    assert result.frame["b"].to_list() == [3, 1, 2]


def test_descending_length_mismatch_rejected() -> None:
    frame = pl.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(DataValidationError, match="descending length"):
        DatasetSorter().sort(frame, by=["a", "b"], descending=[True])


def test_descending_sequence_non_bool_rejected() -> None:
    frame = pl.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(TypeError, match=r"descending\[1\]"):
        DatasetSorter().sort(
            frame,
            by=["a", "b"],
            descending=[True, 0],  # type: ignore[list-item]
        )


@pytest.mark.parametrize("bad_value", ["true", b"true"])
def test_descending_str_or_bytes_rejected(bad_value: object) -> None:
    frame = pl.DataFrame({"a": [1]})
    with pytest.raises(TypeError, match="descending"):
        DatasetSorter().sort(frame, by="a", descending=bad_value)  # type: ignore[arg-type]


def test_nulls_last_non_bool_rejected() -> None:
    frame = pl.DataFrame({"a": [1]})
    with pytest.raises(TypeError, match="nulls_last"):
        DatasetSorter().sort(frame, by="a", nulls_last=1)  # type: ignore[arg-type]


def test_single_column_ascending_order_correct() -> None:
    frame = pl.DataFrame({"score": [5, 1, 3, 2]})
    result = DatasetSorter().sort(frame, by="score", descending=False)
    assert result.frame["score"].to_list() == [1, 2, 3, 5]


def test_single_column_descending_order_correct() -> None:
    frame = pl.DataFrame({"score": [5, 1, 3, 2]})
    result = DatasetSorter().sort(frame, by="score", descending=True)
    assert result.frame["score"].to_list() == [5, 3, 2, 1]


def test_multi_column_priority_correct() -> None:
    frame = pl.DataFrame(
        {
            "g": ["b", "a", "b", "a"],
            "t": [2, 2, 1, 1],
            "label": ["b2", "a2", "b1", "a1"],
        }
    )
    result = DatasetSorter().sort(frame, by=["g", "t"])
    assert result.frame["label"].to_list() == ["a1", "a2", "b1", "b2"]


def test_per_column_descending_settings_correct() -> None:
    frame = pl.DataFrame(
        {
            "g": ["a", "a", "b", "b"],
            "t": [1, 2, 1, 2],
            "label": ["a1", "a2", "b1", "b2"],
        }
    )
    result = DatasetSorter().sort(
        frame,
        by=["g", "t"],
        descending=[False, True],
    )
    assert result.frame["label"].to_list() == ["a2", "a1", "b2", "b1"]


def test_group_and_timestamp_ascending_sort() -> None:
    frame = pl.DataFrame(
        {
            "batch_id": ["B", "A", "B", "A"],
            "timestamp": [20, 10, 10, 20],
        }
    )
    result = DatasetSorter().sort(
        frame,
        by=["batch_id", "timestamp"],
        descending=[False, False],
    )
    assert result.frame["batch_id"].to_list() == ["A", "A", "B", "B"]
    assert result.frame["timestamp"].to_list() == [10, 20, 10, 20]


def test_multiple_group_columns_plus_timestamp_sort() -> None:
    frame = pl.DataFrame(
        {
            "equipment_id": ["E2", "E1", "E2", "E1"],
            "run_id": ["R1", "R2", "R1", "R1"],
            "timestamp": [2, 1, 1, 1],
            "label": ["e2r1t2", "e1r2t1", "e2r1t1", "e1r1t1"],
        }
    )
    result = DatasetSorter().sort(
        frame,
        by=["equipment_id", "run_id", "timestamp"],
    )
    assert result.frame["label"].to_list() == [
        "e1r1t1",
        "e1r2t1",
        "e2r1t1",
        "e2r1t2",
    ]


def test_nulls_last_true_places_nulls_last() -> None:
    frame = pl.DataFrame({"a": [2, None, 1]})
    result = DatasetSorter().sort(frame, by="a", nulls_last=True)
    assert result.frame["a"].to_list() == [1, 2, None]


def test_nulls_last_false_places_nulls_first() -> None:
    frame = pl.DataFrame({"a": [2, None, 1]})
    result = DatasetSorter().sort(frame, by="a", nulls_last=False)
    assert result.frame["a"].to_list() == [None, 1, 2]


def test_stable_sort_preserves_relative_order_for_equal_keys() -> None:
    frame = pl.DataFrame(
        {
            "key": [1, 1, 1],
            "payload": ["first", "second", "third"],
        }
    )
    result = DatasetSorter().sort(frame, by="key")
    assert result.frame["payload"].to_list() == ["first", "second", "third"]


def test_original_row_id_values_preserved() -> None:
    frame = pl.DataFrame(
        {
            "a": [3, 1, 2],
            ORIGINAL_ROW_ID_COLUMN: [10, 20, 30],
        }
    )
    result = DatasetSorter().sort(frame, by="a")
    assert result.frame[ORIGINAL_ROW_ID_COLUMN].to_list() == [20, 30, 10]
    assert ORIGINAL_ROW_ID_COLUMN == LOADER_ROW_ID


def test_original_row_id_can_trace_back_to_source_rows() -> None:
    frame = pl.DataFrame(
        {
            "a": [30, 10, 20],
            "label": ["row0", "row1", "row2"],
            ORIGINAL_ROW_ID_COLUMN: [0, 1, 2],
        }
    )
    result = DatasetSorter().sort(frame, by="a")
    traced = [
        frame.filter(pl.col(ORIGINAL_ROW_ID_COLUMN) == row_id)["label"].item()
        for row_id in result.frame[ORIGINAL_ROW_ID_COLUMN].to_list()
    ]
    assert traced == ["row1", "row2", "row0"]
    assert result.frame["label"].to_list() == ["row1", "row2", "row0"]


def test_original_row_id_can_be_explicit_sort_key() -> None:
    frame = pl.DataFrame(
        {
            "a": [1, 1, 1],
            ORIGINAL_ROW_ID_COLUMN: [2, 0, 1],
        }
    )
    result = DatasetSorter().sort(frame, by=ORIGINAL_ROW_ID_COLUMN)
    assert result.frame[ORIGINAL_ROW_ID_COLUMN].to_list() == [0, 1, 2]


def test_frame_without_original_row_id_is_supported() -> None:
    frame = pl.DataFrame({"a": [2, 1], "b": ["x", "y"]})
    assert ORIGINAL_ROW_ID_COLUMN not in frame.columns
    result = DatasetSorter().sort(frame, by="a")
    assert result.frame["a"].to_list() == [1, 2]
    assert ORIGINAL_ROW_ID_COLUMN not in result.frame.columns


def test_returned_column_order_matches_input() -> None:
    frame = pl.DataFrame({"z": [2, 1], "a": [9, 8], "m": [5, 4]})
    result = DatasetSorter().sort(frame, by="z")
    assert result.frame.columns == frame.columns


def test_returned_dtypes_match_input() -> None:
    frame = pl.DataFrame(
        {
            "a": pl.Series([2, 1], dtype=pl.Int32),
            "b": pl.Series(["x", "y"], dtype=pl.Utf8),
            "c": pl.Series([1.5, 0.5], dtype=pl.Float64),
        }
    )
    result = DatasetSorter().sort(frame, by="a")
    assert result.frame.schema == frame.schema


def test_returned_row_count_matches_input() -> None:
    frame = pl.DataFrame({"a": [3, 1, 2, 4]})
    result = DatasetSorter().sort(frame, by="a")
    assert result.frame.height == frame.height


def test_input_frame_shape_unchanged_after_sort() -> None:
    frame = pl.DataFrame({"a": [3, 1, 2], "b": [9, 8, 7]})
    before_shape = frame.shape
    DatasetSorter().sort(frame, by="a")
    assert frame.shape == before_shape


def test_input_frame_columns_unchanged_after_sort() -> None:
    frame = pl.DataFrame({"a": [3, 1, 2], "b": [9, 8, 7]})
    before_columns = list(frame.columns)
    DatasetSorter().sort(frame, by="a")
    assert list(frame.columns) == before_columns


def test_input_frame_data_unchanged_after_sort() -> None:
    frame = pl.DataFrame({"a": [3, 1, 2], "b": ["c", "a", "b"]})
    before = frame.clone()
    DatasetSorter().sort(frame, by="a")
    assert_frame_equal(frame, before)


def test_empty_dataframe_sort_succeeds() -> None:
    frame = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)})
    result = DatasetSorter().sort(frame, by="a")
    assert result.frame.height == 0
    assert result.frame.columns == ["a"]
    assert result.event.rows_before == 0
    assert result.event.rows_after == 0


def test_single_row_dataframe_sort_succeeds() -> None:
    frame = pl.DataFrame({"a": [42], "b": ["only"]})
    result = DatasetSorter().sort(frame, by="a")
    assert_frame_equal(result.frame, frame)


def test_already_sorted_frame_sort_succeeds() -> None:
    frame = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = DatasetSorter().sort(frame, by="a")
    assert_frame_equal(result.frame, frame)


def test_sort_result_is_frozen_dataclass() -> None:
    frame = pl.DataFrame({"a": [2, 1]})
    result = DatasetSorter().sort(frame, by="a")
    assert is_dataclass(result)
    assert isinstance(result, SortResult)
    assert isinstance(result, DirectSortResult)
    with pytest.raises(FrozenInstanceError):
        result.frame = frame  # type: ignore[misc]


def test_sort_result_uses_slots() -> None:
    assert SortResult.__slots__ == ("frame", "event")
    field_names = {item.name for item in fields(SortResult)}
    assert field_names == {"frame", "event"}


def test_event_step_name_is_sort_dataset() -> None:
    result = DatasetSorter().sort(pl.DataFrame({"a": [1]}), by="a")
    assert result.event.step_name == "sort_dataset"
    assert isinstance(result.event, PreprocessingEvent)


def test_event_affected_columns_match_by_order() -> None:
    frame = pl.DataFrame({"b": [1], "a": [2], "c": [3]})
    result = DatasetSorter().sort(frame, by=["c", "a"])
    assert result.event.affected_columns == ["c", "a"]


def test_event_row_counts_are_correct() -> None:
    frame = pl.DataFrame({"a": [3, 1, 2]})
    result = DatasetSorter().sort(frame, by="a")
    assert result.event.rows_before == 3
    assert result.event.rows_after == 3


def test_event_parameters_preserve_normalized_values() -> None:
    frame = pl.DataFrame({"a": [1], "b": [2]})
    result = DatasetSorter().sort(
        frame,
        by="a",
        descending=True,
        nulls_last=False,
    )
    assert result.event.parameters["by"] == ["a"]
    assert result.event.parameters["descending"] == [True]
    assert result.event.parameters["nulls_last"] is False
    assert result.event.parameters["stable"] is True


def test_event_parameters_descending_always_per_column_list() -> None:
    frame = pl.DataFrame({"a": [1], "b": [2]})
    result = DatasetSorter().sort(frame, by=["a", "b"], descending=False)
    assert result.event.parameters["descending"] == [False, False]
    assert isinstance(result.event.parameters["descending"], list)


def test_event_parameters_stable_is_true() -> None:
    result = DatasetSorter().sort(pl.DataFrame({"a": [2, 1]}), by="a")
    assert result.event.parameters["stable"] is True


def test_event_timestamp_is_timezone_aware_utc() -> None:
    result = DatasetSorter().sort(pl.DataFrame({"a": [1]}), by="a")
    timestamp = result.event.timestamp
    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() == UTC.utcoffset(timestamp)
    assert timestamp.tzinfo == UTC


def test_null_sort_column_generates_warning() -> None:
    frame = pl.DataFrame({"a": [1, None, 2]})
    result = DatasetSorter().sort(frame, by="a", nulls_last=True)
    assert len(result.event.warnings) == 1
    assert "a" in result.event.warnings[0]
    assert "null" in result.event.warnings[0].lower()


def test_no_warning_when_sort_columns_have_no_nulls() -> None:
    frame = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = DatasetSorter().sort(frame, by=["a", "b"])
    assert result.event.warnings == []


def test_warning_order_follows_by_column_order() -> None:
    frame = pl.DataFrame(
        {
            "a": [1, None],
            "b": [None, 2],
            "c": [3, 4],
        }
    )
    result = DatasetSorter().sort(frame, by=["b", "c", "a"], nulls_last=False)
    assert len(result.event.warnings) == 2
    assert "b" in result.event.warnings[0]
    assert "a" in result.event.warnings[1]
    assert "first" in result.event.warnings[0]


def test_input_by_list_is_not_mutated() -> None:
    frame = pl.DataFrame({"a": [2, 1], "b": [4, 3]})
    by_columns = ["a", "b"]
    DatasetSorter().sort(frame, by=by_columns)
    assert by_columns == ["a", "b"]


def test_input_descending_list_is_not_mutated() -> None:
    frame = pl.DataFrame({"a": [2, 1], "b": [4, 3]})
    descending_flags = [True, False]
    DatasetSorter().sort(frame, by=["a", "b"], descending=descending_flags)
    assert descending_flags == [True, False]


def test_consecutive_sorts_do_not_share_state() -> None:
    sorter = DatasetSorter()
    first = pl.DataFrame({"a": [2, 1], "x": ["f1", "f2"]})
    second = pl.DataFrame({"a": [9, 8], "x": ["s1", "s2"]})
    first_result = sorter.sort(first, by="a")
    second_result = sorter.sort(second, by="a")
    assert first_result.frame["x"].to_list() == ["f2", "f1"]
    assert second_result.frame["x"].to_list() == ["s2", "s1"]
    assert first_result.event.parameters["by"] == ["a"]
    assert second_result.event.parameters["by"] == ["a"]


def test_same_input_produces_deterministic_result() -> None:
    frame = pl.DataFrame(
        {
            "key": [1, 1, 2, 1],
            "payload": ["a", "b", "c", "d"],
        }
    )
    sorter = DatasetSorter()
    first = sorter.sort(frame, by="key", descending=False, nulls_last=True)
    second = sorter.sort(frame, by="key", descending=False, nulls_last=True)
    assert_frame_equal(first.frame, second.frame)
    assert first.event.parameters == second.event.parameters
    assert first.event.warnings == second.event.warnings
    assert first.event.affected_columns == second.event.affected_columns
