"""Unit tests for LineageTracker (Step 2F)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, is_dataclass
from datetime import UTC, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from process_intelligence.core.exceptions import DataValidationError
from process_intelligence.core.schemas import PreprocessingEvent
from process_intelligence.data import (
    ORIGINAL_ROW_ID_COLUMN,
    PROCESSED_ROW_INDEX_COLUMN,
    LineageSnapshot,
    LineageTracker,
)
from process_intelligence.data.lineage import (
    PROCESSED_ROW_INDEX_COLUMN as DIRECT_PROCESSED_INDEX,
)
from process_intelligence.data.lineage import (
    LineageSnapshot as DirectSnapshot,
)
from process_intelligence.data.lineage import (
    LineageTracker as DirectTracker,
)


def _raw_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "value": [10, 20, 30, 40],
            ORIGINAL_ROW_ID_COLUMN: [0, 1, 2, 3],
        }
    )


def _make_event(
    *,
    step_name: str = "sort_dataset",
    rows_before: int,
    rows_after: int,
    affected_columns: list[str] | None = None,
    parameters: dict[str, object] | None = None,
    warnings: list[str] | None = None,
    timestamp: datetime | None = None,
) -> PreprocessingEvent:
    return PreprocessingEvent(
        step_name=step_name,
        affected_columns=affected_columns if affected_columns is not None else ["value"],
        rows_before=rows_before,
        rows_after=rows_after,
        parameters=parameters if parameters is not None else {"stable": True},
        warnings=warnings if warnings is not None else [],
        timestamp=timestamp if timestamp is not None else datetime.now(UTC),
    )


def test_tracker_construction_succeeds() -> None:
    tracker = LineageTracker(_raw_frame())
    assert isinstance(tracker, LineageTracker)
    assert isinstance(tracker, DirectTracker)


def test_non_polars_raw_raises_type_error() -> None:
    with pytest.raises(TypeError, match="polars.DataFrame"):
        LineageTracker({"a": [1]})  # type: ignore[arg-type]


def test_raw_missing_original_row_id_raises() -> None:
    frame = pl.DataFrame({"value": [1, 2]})
    with pytest.raises(DataValidationError, match=ORIGINAL_ROW_ID_COLUMN):
        LineageTracker(frame)


def test_raw_null_original_row_id_raises() -> None:
    frame = pl.DataFrame(
        {
            "value": [1, 2],
            ORIGINAL_ROW_ID_COLUMN: [0, None],
        }
    )
    with pytest.raises(DataValidationError, match="null"):
        LineageTracker(frame)


def test_raw_non_integer_original_row_id_raises() -> None:
    frame = pl.DataFrame(
        {
            "value": [1.0, 2.0],
            ORIGINAL_ROW_ID_COLUMN: [0.5, 1.5],
        }
    )
    with pytest.raises(DataValidationError, match="integer"):
        LineageTracker(frame)


def test_raw_boolean_original_row_id_raises() -> None:
    frame = pl.DataFrame(
        {
            "value": [1, 2],
            ORIGINAL_ROW_ID_COLUMN: [True, False],
        }
    )
    with pytest.raises(DataValidationError, match="Boolean|integer"):
        LineageTracker(frame)


def test_raw_duplicate_original_row_id_raises() -> None:
    frame = pl.DataFrame(
        {
            "value": [1, 2],
            ORIGINAL_ROW_ID_COLUMN: [1, 1],
        }
    )
    with pytest.raises(DataValidationError, match="unique"):
        LineageTracker(frame)


def test_raw_negative_original_row_id_raises() -> None:
    frame = pl.DataFrame(
        {
            "value": [1, 2],
            ORIGINAL_ROW_ID_COLUMN: [0, -1],
        }
    )
    with pytest.raises(DataValidationError, match="negative"):
        LineageTracker(frame)


def test_raw_non_contiguous_positive_ids_allowed() -> None:
    frame = pl.DataFrame(
        {
            "value": [10, 20, 30],
            ORIGINAL_ROW_ID_COLUMN: [3, 7, 11],
        }
    )
    tracker = LineageTracker(frame)
    assert tracker.get_raw_frame()[ORIGINAL_ROW_ID_COLUMN].to_list() == [3, 7, 11]


def test_empty_raw_frame_succeeds() -> None:
    frame = pl.DataFrame(
        {
            "value": pl.Series([], dtype=pl.Int64),
            ORIGINAL_ROW_ID_COLUMN: pl.Series([], dtype=pl.Int64),
        }
    )
    tracker = LineageTracker(frame)
    snap = tracker.snapshot()
    assert snap.raw_row_count == 0
    assert snap.processed_row_count == 0
    assert snap.mapping.height == 0


def test_constructor_does_not_mutate_input() -> None:
    frame = _raw_frame()
    before = frame.clone()
    LineageTracker(frame)
    assert_frame_equal(frame, before)


def test_initial_current_frame_matches_raw() -> None:
    frame = _raw_frame()
    tracker = LineageTracker(frame)
    assert_frame_equal(tracker.get_current_frame(), tracker.get_raw_frame())


def test_initial_event_history_is_empty() -> None:
    tracker = LineageTracker(_raw_frame())
    history = tracker.get_event_history()
    assert history == ()
    assert isinstance(history, tuple)


def test_get_raw_frame_returns_clone() -> None:
    tracker = LineageTracker(_raw_frame())
    first = tracker.get_raw_frame()
    second = tracker.get_raw_frame()
    assert first is not second
    mutated = first.with_columns(pl.lit(999).alias("value"))
    assert mutated["value"].to_list() == [999, 999, 999, 999]
    assert tracker.get_raw_frame()["value"].to_list() == [10, 20, 30, 40]


def test_get_current_frame_returns_clone() -> None:
    tracker = LineageTracker(_raw_frame())
    first = tracker.get_current_frame()
    second = tracker.get_current_frame()
    assert first is not second
    _ = first.with_columns(pl.lit(-1).alias("value"))
    assert tracker.get_current_frame()["value"].to_list() == [10, 20, 30, 40]


def test_initial_snapshot_row_counts() -> None:
    tracker = LineageTracker(_raw_frame())
    snap = tracker.snapshot()
    assert snap.raw_row_count == 4
    assert snap.processed_row_count == 4


def test_initial_snapshot_mapping_is_correct() -> None:
    tracker = LineageTracker(_raw_frame())
    snap = tracker.snapshot()
    expected = pl.DataFrame(
        {
            PROCESSED_ROW_INDEX_COLUMN: [0, 1, 2, 3],
            ORIGINAL_ROW_ID_COLUMN: [0, 1, 2, 3],
        }
    )
    assert_frame_equal(snap.mapping, expected)


def test_mapping_column_names_and_order() -> None:
    tracker = LineageTracker(_raw_frame())
    mapping = tracker.snapshot().mapping
    assert mapping.columns == [
        PROCESSED_ROW_INDEX_COLUMN,
        ORIGINAL_ROW_ID_COLUMN,
    ]
    assert PROCESSED_ROW_INDEX_COLUMN == "_processed_row_index"
    assert PROCESSED_ROW_INDEX_COLUMN == DIRECT_PROCESSED_INDEX


def test_mapping_processed_index_is_contiguous_from_zero() -> None:
    frame = pl.DataFrame(
        {
            "value": [30, 10, 40],
            ORIGINAL_ROW_ID_COLUMN: [3, 1, 4],
        }
    )
    tracker = LineageTracker(frame)
    indices = tracker.snapshot().mapping[PROCESSED_ROW_INDEX_COLUMN].to_list()
    assert indices == [0, 1, 2]


def test_record_sorted_processed_frame_succeeds() -> None:
    tracker = LineageTracker(_raw_frame())
    processed = tracker.get_current_frame().sort("value", descending=True)
    event = _make_event(rows_before=4, rows_after=4)
    snap = tracker.record(processed, event)
    assert snap.processed_row_count == 4
    assert tracker.get_current_frame()[ORIGINAL_ROW_ID_COLUMN].to_list() == [
        3,
        2,
        1,
        0,
    ]


def test_record_subset_processed_frame_succeeds() -> None:
    tracker = LineageTracker(_raw_frame())
    processed = tracker.get_current_frame().filter(
        pl.col(ORIGINAL_ROW_ID_COLUMN).is_in([1, 3])
    )
    event = _make_event(
        step_name="filter_rows",
        rows_before=4,
        rows_after=2,
    )
    snap = tracker.record(processed, event)
    assert snap.processed_row_count == 2
    assert tracker.removed_original_row_ids() == [0, 2]


def test_non_polars_processed_raises_type_error() -> None:
    tracker = LineageTracker(_raw_frame())
    event = _make_event(rows_before=4, rows_after=4)
    with pytest.raises(TypeError, match="polars.DataFrame"):
        tracker.record({"a": [1]}, event)  # type: ignore[arg-type]


def test_processed_missing_original_row_id_raises() -> None:
    tracker = LineageTracker(_raw_frame())
    processed = pl.DataFrame({"value": [1, 2, 3, 4]})
    event = _make_event(rows_before=4, rows_after=4)
    with pytest.raises(DataValidationError, match=ORIGINAL_ROW_ID_COLUMN):
        tracker.record(processed, event)


def test_processed_null_original_row_id_raises() -> None:
    tracker = LineageTracker(_raw_frame())
    processed = pl.DataFrame(
        {
            "value": [1, 2],
            ORIGINAL_ROW_ID_COLUMN: [0, None],
        }
    )
    event = _make_event(rows_before=4, rows_after=2)
    with pytest.raises(DataValidationError, match="null"):
        tracker.record(processed, event)


def test_processed_non_integer_original_row_id_raises() -> None:
    tracker = LineageTracker(_raw_frame())
    processed = pl.DataFrame(
        {
            "value": [1.0],
            ORIGINAL_ROW_ID_COLUMN: [0.5],
        }
    )
    event = _make_event(rows_before=4, rows_after=1)
    with pytest.raises(DataValidationError, match="integer"):
        tracker.record(processed, event)


def test_processed_boolean_original_row_id_raises() -> None:
    tracker = LineageTracker(_raw_frame())
    processed = pl.DataFrame(
        {
            "value": [1],
            ORIGINAL_ROW_ID_COLUMN: [True],
        }
    )
    event = _make_event(rows_before=4, rows_after=1)
    with pytest.raises(DataValidationError, match="Boolean|integer"):
        tracker.record(processed, event)


def test_processed_duplicate_original_row_id_raises() -> None:
    tracker = LineageTracker(_raw_frame())
    processed = pl.DataFrame(
        {
            "value": [1, 2],
            ORIGINAL_ROW_ID_COLUMN: [1, 1],
        }
    )
    event = _make_event(rows_before=4, rows_after=2)
    with pytest.raises(DataValidationError, match="unique"):
        tracker.record(processed, event)


def test_processed_negative_original_row_id_raises() -> None:
    tracker = LineageTracker(_raw_frame())
    processed = pl.DataFrame(
        {
            "value": [1],
            ORIGINAL_ROW_ID_COLUMN: [-1],
        }
    )
    event = _make_event(rows_before=4, rows_after=1)
    with pytest.raises(DataValidationError, match="negative"):
        tracker.record(processed, event)


def test_processed_unknown_original_row_id_raises() -> None:
    tracker = LineageTracker(_raw_frame())
    processed = pl.DataFrame(
        {
            "value": [1],
            ORIGINAL_ROW_ID_COLUMN: [99],
        }
    )
    event = _make_event(rows_before=4, rows_after=1)
    with pytest.raises(DataValidationError, match="not present"):
        tracker.record(processed, event)


def test_non_preprocessing_event_raises_type_error() -> None:
    tracker = LineageTracker(_raw_frame())
    with pytest.raises(TypeError, match="PreprocessingEvent"):
        tracker.record(tracker.get_current_frame(), {"step": "x"})  # type: ignore[arg-type]


def test_event_rows_before_mismatch_raises() -> None:
    tracker = LineageTracker(_raw_frame())
    event = _make_event(rows_before=3, rows_after=4)
    with pytest.raises(DataValidationError, match="rows_before"):
        tracker.record(tracker.get_current_frame(), event)


def test_event_rows_after_mismatch_raises() -> None:
    tracker = LineageTracker(_raw_frame())
    event = _make_event(rows_before=4, rows_after=2)
    with pytest.raises(DataValidationError, match="rows_after"):
        tracker.record(tracker.get_current_frame(), event)


def test_naive_timestamp_event_raises() -> None:
    tracker = LineageTracker(_raw_frame())
    event = _make_event(
        rows_before=4,
        rows_after=4,
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
    )
    with pytest.raises(DataValidationError, match="timezone-aware"):
        tracker.record(tracker.get_current_frame(), event)


def test_timezone_aware_utc_event_allowed() -> None:
    tracker = LineageTracker(_raw_frame())
    event = _make_event(
        rows_before=4,
        rows_after=4,
        timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
    )
    snap = tracker.record(tracker.get_current_frame(), event)
    assert snap.events[0].timestamp.tzinfo is not None


def test_record_updates_current_frame() -> None:
    tracker = LineageTracker(_raw_frame())
    processed = tracker.get_current_frame().filter(
        pl.col(ORIGINAL_ROW_ID_COLUMN) == 2
    )
    tracker.record(processed, _make_event(rows_before=4, rows_after=1))
    assert_frame_equal(tracker.get_current_frame(), processed)


def test_record_appends_event() -> None:
    tracker = LineageTracker(_raw_frame())
    event = _make_event(step_name="step_a", rows_before=4, rows_after=4)
    tracker.record(tracker.get_current_frame(), event)
    history = tracker.get_event_history()
    assert len(history) == 1
    assert history[0].step_name == "step_a"


def test_record_returns_lineage_snapshot() -> None:
    tracker = LineageTracker(_raw_frame())
    result = tracker.record(
        tracker.get_current_frame(),
        _make_event(rows_before=4, rows_after=4),
    )
    assert isinstance(result, LineageSnapshot)
    assert isinstance(result, DirectSnapshot)


def test_failed_record_does_not_change_current_frame() -> None:
    tracker = LineageTracker(_raw_frame())
    before = tracker.get_current_frame()
    bad = pl.DataFrame(
        {
            "value": [1],
            ORIGINAL_ROW_ID_COLUMN: [99],
        }
    )
    with pytest.raises(DataValidationError):
        tracker.record(bad, _make_event(rows_before=4, rows_after=1))
    assert_frame_equal(tracker.get_current_frame(), before)


def test_failed_record_does_not_change_event_history() -> None:
    tracker = LineageTracker(_raw_frame())
    tracker.record(
        tracker.get_current_frame(),
        _make_event(step_name="ok", rows_before=4, rows_after=4),
    )
    before = tracker.get_event_history()
    with pytest.raises(DataValidationError):
        tracker.record(
            tracker.get_current_frame(),
            _make_event(rows_before=3, rows_after=4),
        )
    after = tracker.get_event_history()
    assert len(after) == 1
    assert after[0].model_dump() == before[0].model_dump()


def test_record_does_not_mutate_processed_frame() -> None:
    tracker = LineageTracker(_raw_frame())
    processed = tracker.get_current_frame().sort("value", descending=True)
    before = processed.clone()
    tracker.record(processed, _make_event(rows_before=4, rows_after=4))
    assert_frame_equal(processed, before)


def test_record_does_not_mutate_event() -> None:
    tracker = LineageTracker(_raw_frame())
    event = _make_event(
        rows_before=4,
        rows_after=4,
        warnings=["existing"],
        parameters={"key": ["nested"]},
    )
    before = deepcopy(event.model_dump())
    tracker.record(tracker.get_current_frame(), event)
    assert event.model_dump() == before
    event.warnings.append("caller_change")
    assert tracker.get_event_history()[0].warnings == ["existing"]


def test_snapshot_frame_matches_current() -> None:
    tracker = LineageTracker(_raw_frame())
    processed = tracker.get_current_frame().filter(
        pl.col(ORIGINAL_ROW_ID_COLUMN).is_in([3, 1])
    )
    tracker.record(
        processed,
        _make_event(step_name="filter", rows_before=4, rows_after=2),
    )
    assert_frame_equal(tracker.snapshot().frame, tracker.get_current_frame())


def test_snapshot_mapping_reflects_current_order() -> None:
    tracker = LineageTracker(
        pl.DataFrame(
            {
                "value": [10, 20, 30],
                ORIGINAL_ROW_ID_COLUMN: [3, 1, 4],
            }
        )
    )
    mapping = tracker.snapshot().mapping
    assert mapping[PROCESSED_ROW_INDEX_COLUMN].to_list() == [0, 1, 2]
    assert mapping[ORIGINAL_ROW_ID_COLUMN].to_list() == [3, 1, 4]


def test_snapshot_events_is_tuple() -> None:
    tracker = LineageTracker(_raw_frame())
    tracker.record(
        tracker.get_current_frame(),
        _make_event(rows_before=4, rows_after=4),
    )
    events = tracker.snapshot().events
    assert isinstance(events, tuple)
    assert len(events) == 1


def test_lineage_snapshot_is_frozen_dataclass() -> None:
    tracker = LineageTracker(_raw_frame())
    snap = tracker.snapshot()
    assert is_dataclass(snap)
    assert isinstance(snap, LineageSnapshot)
    with pytest.raises(FrozenInstanceError):
        snap.raw_row_count = 99  # type: ignore[misc]


def test_lineage_snapshot_uses_slots() -> None:
    assert LineageSnapshot.__slots__ == (
        "frame",
        "mapping",
        "events",
        "raw_row_count",
        "processed_row_count",
    )
    field_names = {item.name for item in fields(LineageSnapshot)}
    assert field_names == {
        "frame",
        "mapping",
        "events",
        "raw_row_count",
        "processed_row_count",
    }


def test_trace_processed_row_returns_original_id() -> None:
    tracker = LineageTracker(
        pl.DataFrame(
            {
                "value": [10, 20, 30],
                ORIGINAL_ROW_ID_COLUMN: [3, 1, 4],
            }
        )
    )
    assert tracker.trace_processed_row(0) == 3
    assert tracker.trace_processed_row(1) == 1
    assert tracker.trace_processed_row(2) == 4


def test_trace_processed_row_returns_python_int() -> None:
    tracker = LineageTracker(_raw_frame())
    result = tracker.trace_processed_row(0)
    assert type(result) is int


def test_trace_processed_row_rejects_bool() -> None:
    tracker = LineageTracker(_raw_frame())
    with pytest.raises(TypeError, match="processed_row_index"):
        tracker.trace_processed_row(True)  # type: ignore[arg-type]


def test_trace_processed_row_rejects_non_int() -> None:
    tracker = LineageTracker(_raw_frame())
    with pytest.raises(TypeError, match="processed_row_index"):
        tracker.trace_processed_row(0.0)  # type: ignore[arg-type]


def test_trace_processed_row_negative_index_raises() -> None:
    tracker = LineageTracker(_raw_frame())
    with pytest.raises(IndexError):
        tracker.trace_processed_row(-1)


def test_trace_processed_row_out_of_range_raises() -> None:
    tracker = LineageTracker(_raw_frame())
    with pytest.raises(IndexError):
        tracker.trace_processed_row(4)


def test_find_processed_rows_returns_current_position() -> None:
    tracker = LineageTracker(
        pl.DataFrame(
            {
                "value": [10, 20, 30],
                ORIGINAL_ROW_ID_COLUMN: [3, 1, 4],
            }
        )
    )
    assert tracker.find_processed_rows(1) == [1]
    assert tracker.find_processed_rows(3) == [0]
    assert tracker.find_processed_rows(4) == [2]


def test_find_processed_rows_removed_id_returns_empty() -> None:
    tracker = LineageTracker(_raw_frame())
    processed = tracker.get_current_frame().filter(
        pl.col(ORIGINAL_ROW_ID_COLUMN) != 2
    )
    tracker.record(
        processed,
        _make_event(step_name="drop", rows_before=4, rows_after=3),
    )
    assert tracker.find_processed_rows(2) == []


def test_find_processed_rows_unknown_id_returns_empty() -> None:
    tracker = LineageTracker(_raw_frame())
    assert tracker.find_processed_rows(999) == []


def test_find_processed_rows_rejects_bool() -> None:
    tracker = LineageTracker(_raw_frame())
    with pytest.raises(TypeError, match="original_row_id"):
        tracker.find_processed_rows(True)  # type: ignore[arg-type]


def test_find_processed_rows_rejects_non_int() -> None:
    tracker = LineageTracker(_raw_frame())
    with pytest.raises(TypeError, match="original_row_id"):
        tracker.find_processed_rows("1")  # type: ignore[arg-type]


def test_find_processed_rows_negative_id_raises() -> None:
    tracker = LineageTracker(_raw_frame())
    with pytest.raises(ValueError, match="non-negative"):
        tracker.find_processed_rows(-1)


def test_removed_original_row_ids_preserves_raw_order() -> None:
    tracker = LineageTracker(_raw_frame())
    processed = tracker.get_current_frame().filter(
        pl.col(ORIGINAL_ROW_ID_COLUMN).is_in([3, 1])
    )
    tracker.record(
        processed,
        _make_event(step_name="subset", rows_before=4, rows_after=2),
    )
    assert tracker.removed_original_row_ids() == [0, 2]


def test_removed_original_row_ids_empty_when_none_removed() -> None:
    tracker = LineageTracker(_raw_frame())
    assert tracker.removed_original_row_ids() == []


def test_event_history_preserves_record_order() -> None:
    tracker = LineageTracker(_raw_frame())
    tracker.record(
        tracker.get_current_frame(),
        _make_event(step_name="first", rows_before=4, rows_after=4),
    )
    subset = tracker.get_current_frame().head(2)
    tracker.record(
        subset,
        _make_event(step_name="second", rows_before=4, rows_after=2),
    )
    names = [item.step_name for item in tracker.get_event_history()]
    assert names == ["first", "second"]


def test_event_history_step_name_filter() -> None:
    tracker = LineageTracker(_raw_frame())
    tracker.record(
        tracker.get_current_frame(),
        _make_event(step_name="sort_dataset", rows_before=4, rows_after=4),
    )
    subset = tracker.get_current_frame().head(3)
    tracker.record(
        subset,
        _make_event(step_name="filter_rows", rows_before=4, rows_after=3),
    )
    filtered = tracker.get_event_history(step_name="filter_rows")
    assert len(filtered) == 1
    assert filtered[0].step_name == "filter_rows"


def test_event_history_unknown_step_name_returns_empty() -> None:
    tracker = LineageTracker(_raw_frame())
    tracker.record(
        tracker.get_current_frame(),
        _make_event(rows_before=4, rows_after=4),
    )
    assert tracker.get_event_history(step_name="missing") == ()


def test_event_history_empty_step_name_raises() -> None:
    tracker = LineageTracker(_raw_frame())
    with pytest.raises(ValueError, match="non-empty"):
        tracker.get_event_history(step_name="")


def test_event_history_non_string_step_name_raises() -> None:
    tracker = LineageTracker(_raw_frame())
    with pytest.raises(TypeError, match="step_name"):
        tracker.get_event_history(step_name=1)  # type: ignore[arg-type]


def test_returned_events_are_independent_copies() -> None:
    tracker = LineageTracker(_raw_frame())
    tracker.record(
        tracker.get_current_frame(),
        _make_event(
            rows_before=4,
            rows_after=4,
            warnings=["a"],
            parameters={"items": [1]},
        ),
    )
    returned = tracker.get_event_history()
    returned[0].warnings.append("mutated")
    returned[0].parameters["items"].append(2)
    internal = tracker.get_event_history()
    assert internal[0].warnings == ["a"]
    assert internal[0].parameters["items"] == [1]


def test_snapshot_frame_mutation_does_not_affect_tracker() -> None:
    tracker = LineageTracker(_raw_frame())
    snap = tracker.snapshot()
    _ = snap.frame.with_columns(pl.lit(0).alias("value"))
    assert tracker.get_current_frame()["value"].to_list() == [10, 20, 30, 40]


def test_raw_frame_unchanged_after_multiple_records() -> None:
    tracker = LineageTracker(_raw_frame())
    raw_before = tracker.get_raw_frame()
    subset = tracker.get_current_frame().head(3)
    tracker.record(
        subset,
        _make_event(step_name="a", rows_before=4, rows_after=3),
    )
    subset2 = tracker.get_current_frame().head(1)
    tracker.record(
        subset2,
        _make_event(step_name="b", rows_before=3, rows_after=1),
    )
    assert_frame_equal(tracker.get_raw_frame(), raw_before)
    assert tracker.get_raw_frame().height == 4


def test_multiple_records_preserve_event_order() -> None:
    tracker = LineageTracker(_raw_frame())
    tracker.record(
        tracker.get_current_frame(),
        _make_event(step_name="one", rows_before=4, rows_after=4),
    )
    tracker.record(
        tracker.get_current_frame().head(2),
        _make_event(step_name="two", rows_before=4, rows_after=2),
    )
    tracker.record(
        tracker.get_current_frame().head(1),
        _make_event(step_name="three", rows_before=2, rows_after=1),
    )
    assert [e.step_name for e in tracker.get_event_history()] == [
        "one",
        "two",
        "three",
    ]


def test_separate_trackers_do_not_share_state() -> None:
    first = LineageTracker(_raw_frame())
    second = LineageTracker(_raw_frame())
    first.record(
        first.get_current_frame().head(1),
        _make_event(step_name="only_first", rows_before=4, rows_after=1),
    )
    assert first.get_event_history()[0].step_name == "only_first"
    assert second.get_event_history() == ()
    assert second.get_current_frame().height == 4
    assert first.get_current_frame().height == 1


def test_empty_processed_frame_record_supported() -> None:
    tracker = LineageTracker(_raw_frame())
    empty = tracker.get_current_frame().clear()
    snap = tracker.record(
        empty,
        _make_event(step_name="drop_all", rows_before=4, rows_after=0),
    )
    assert snap.processed_row_count == 0
    assert tracker.get_current_frame().height == 0


def test_empty_processed_mapping_is_empty() -> None:
    tracker = LineageTracker(_raw_frame())
    empty = tracker.get_current_frame().clear()
    snap = tracker.record(
        empty,
        _make_event(step_name="drop_all", rows_before=4, rows_after=0),
    )
    assert snap.mapping.height == 0
    assert snap.mapping.columns == [
        PROCESSED_ROW_INDEX_COLUMN,
        ORIGINAL_ROW_ID_COLUMN,
    ]


def test_all_rows_removed_returns_all_raw_ids() -> None:
    tracker = LineageTracker(_raw_frame())
    empty = tracker.get_current_frame().clear()
    tracker.record(
        empty,
        _make_event(step_name="drop_all", rows_before=4, rows_after=0),
    )
    assert tracker.removed_original_row_ids() == [0, 1, 2, 3]


def test_processed_row_index_not_added_to_current_frame() -> None:
    tracker = LineageTracker(_raw_frame())
    assert PROCESSED_ROW_INDEX_COLUMN not in tracker.get_current_frame().columns
    tracker.record(
        tracker.get_current_frame().sort("value", descending=True),
        _make_event(rows_before=4, rows_after=4),
    )
    assert PROCESSED_ROW_INDEX_COLUMN not in tracker.get_current_frame().columns
    assert PROCESSED_ROW_INDEX_COLUMN not in tracker.snapshot().frame.columns
    assert PROCESSED_ROW_INDEX_COLUMN in tracker.snapshot().mapping.columns


def test_find_processed_rows_returns_new_list() -> None:
    tracker = LineageTracker(_raw_frame())
    first = tracker.find_processed_rows(1)
    second = tracker.find_processed_rows(1)
    assert first == [1]
    assert first is not second
    first.append(99)
    assert tracker.find_processed_rows(1) == [1]
