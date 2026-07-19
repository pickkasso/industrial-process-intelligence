"""Raw/processed row lineage tracking via ``_original_row_id`` (Step 2F)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl

from process_intelligence.core.exceptions import DataValidationError
from process_intelligence.core.schemas import PreprocessingEvent
from process_intelligence.data.loader import ORIGINAL_ROW_ID_COLUMN

PROCESSED_ROW_INDEX_COLUMN = "_processed_row_index"


@dataclass(frozen=True, slots=True)
class LineageSnapshot:
    """Immutable snapshot of lineage state at a point in time."""

    frame: pl.DataFrame
    mapping: pl.DataFrame
    events: tuple[PreprocessingEvent, ...]
    raw_row_count: int
    processed_row_count: int


class LineageTracker:
    """Track raw ↔ processed row mapping and preprocessing event history.

    The raw frame is frozen at construction. Each successful ``record`` call
    updates the current processed frame and appends a deep-copied event.
    Failed ``record`` calls leave internal state unchanged.
    """

    def __init__(self, raw_frame: pl.DataFrame) -> None:
        """Initialize a tracker from a raw frame containing ``_original_row_id``.

        Args:
            raw_frame: Raw dataset as a Polars DataFrame. Must include a
                non-null, unique, non-negative integer ``_original_row_id``
                column. The input is not mutated.

        Raises:
            TypeError: If ``raw_frame`` is not a Polars DataFrame.
            DataValidationError: If ``_original_row_id`` is missing or invalid.
        """
        if not isinstance(raw_frame, pl.DataFrame):
            raise TypeError(
                "Expected a polars.DataFrame, "
                f"got {type(raw_frame).__name__}"
            )

        _validate_original_row_id_column(raw_frame, context="raw")

        self._raw = raw_frame.clone()
        self._current = raw_frame.clone()
        self._events: list[PreprocessingEvent] = []
        self._raw_row_count = self._raw.height
        self._raw_id_set: frozenset[int] = frozenset(
            int(value)
            for value in self._raw.get_column(ORIGINAL_ROW_ID_COLUMN).to_list()
        )

    def get_raw_frame(self) -> pl.DataFrame:
        """Return a clone of the immutable raw frame.

        Returns:
            An independent clone of the raw frame stored at construction.
        """
        return self._raw.clone()

    def get_current_frame(self) -> pl.DataFrame:
        """Return a clone of the current processed frame.

        Returns:
            An independent clone of the latest recorded processed frame,
            or the raw frame clone if no successful ``record`` has occurred.
        """
        return self._current.clone()

    def record(
        self,
        processed_frame: pl.DataFrame,
        event: PreprocessingEvent,
    ) -> LineageSnapshot:
        """Record a processed frame and its preprocessing event.

        Validation of ``processed_frame`` and ``event`` completes before any
        internal state is updated. On failure, the current frame and event
        history remain unchanged.

        Args:
            processed_frame: Result frame after a preprocessing step. Must
                include a valid ``_original_row_id`` column whose values are a
                subset of the original raw IDs. The input is not mutated.
            event: Structured log for the preprocessing step. Must be a
                ``PreprocessingEvent`` with timezone-aware timestamp and row
                counts matching tracker state and ``processed_frame``. The
                input is not mutated; a deep copy is stored.

        Returns:
            A ``LineageSnapshot`` reflecting state after the successful record.

        Raises:
            TypeError: If ``processed_frame`` or ``event`` has an unsupported
                type.
            DataValidationError: If frame IDs or event fields fail validation.
        """
        if not isinstance(processed_frame, pl.DataFrame):
            raise TypeError(
                "Expected a polars.DataFrame, "
                f"got {type(processed_frame).__name__}"
            )
        if not isinstance(event, PreprocessingEvent):
            raise TypeError(
                "event must be a PreprocessingEvent, "
                f"got {type(event).__name__}"
            )

        _validate_original_row_id_column(processed_frame, context="processed")
        _validate_processed_ids_subset(
            processed_frame,
            self._raw_id_set,
        )
        _validate_event_for_record(
            event,
            rows_before=self._current.height,
            rows_after=processed_frame.height,
        )

        self._current = processed_frame.clone()
        self._events.append(event.model_copy(deep=True))
        return self.snapshot()

    def snapshot(self) -> LineageSnapshot:
        """Return an independent snapshot of the current lineage state.

        Returns:
            A ``LineageSnapshot`` with cloned frame, mapping, and deep-copied
            events. Does not modify tracker state.
        """
        return LineageSnapshot(
            frame=self._current.clone(),
            mapping=_build_mapping(self._current),
            events=tuple(
                item.model_copy(deep=True) for item in self._events
            ),
            raw_row_count=self._raw_row_count,
            processed_row_count=self._current.height,
        )

    def trace_processed_row(self, processed_row_index: int) -> int:
        """Map a processed row position to its original row ID.

        Args:
            processed_row_index: Zero-based index into the current processed
                frame. Must be a Python ``int`` (not ``bool``).

        Returns:
            The ``_original_row_id`` at ``processed_row_index`` as a Python
            ``int``.

        Raises:
            TypeError: If ``processed_row_index`` is not an ``int``.
            IndexError: If the index is negative or out of range.
        """
        index = _require_strict_int(processed_row_index, "processed_row_index")
        if index < 0 or index >= self._current.height:
            raise IndexError(
                "processed_row_index out of range: "
                f"{index} (processed_row_count={self._current.height})"
            )
        value = self._current.get_column(ORIGINAL_ROW_ID_COLUMN)[index]
        return int(value)

    def find_processed_rows(self, original_row_id: int) -> list[int]:
        """Find current processed positions for an original row ID.

        Args:
            original_row_id: Original ``_original_row_id`` value. Must be a
                non-negative Python ``int`` (not ``bool``).

        Returns:
            A new list of processed row indices. Length is 0 when the ID is
            absent from the current frame (removed or never present), or 1
            when present (duplicate processed IDs are rejected at ``record``).

        Raises:
            TypeError: If ``original_row_id`` is not an ``int``.
            ValueError: If ``original_row_id`` is negative.
        """
        row_id = _require_strict_int(original_row_id, "original_row_id")
        if row_id < 0:
            raise ValueError(
                f"original_row_id must be non-negative, got {row_id}"
            )

        positions: list[int] = []
        for index, value in enumerate(
            self._current.get_column(ORIGINAL_ROW_ID_COLUMN).to_list()
        ):
            if int(value) == row_id:
                positions.append(index)
        return positions

    def removed_original_row_ids(self) -> list[int]:
        """Return original IDs present in raw but absent from current frame.

        Returns:
            Removed IDs in the order they appear in the raw frame. Empty when
            no rows have been removed.
        """
        current_ids = {
            int(value)
            for value in self._current.get_column(ORIGINAL_ROW_ID_COLUMN).to_list()
        }
        return [
            int(value)
            for value in self._raw.get_column(ORIGINAL_ROW_ID_COLUMN).to_list()
            if int(value) not in current_ids
        ]

    def get_event_history(
        self,
        step_name: str | None = None,
    ) -> tuple[PreprocessingEvent, ...]:
        """Return deep-copied preprocessing events, optionally filtered.

        Args:
            step_name: When ``None``, return the full history in record order.
                When a non-empty string, return only events whose
                ``step_name`` matches exactly.

        Returns:
            A tuple of deep-copied ``PreprocessingEvent`` instances. Empty when
            no events match.

        Raises:
            TypeError: If ``step_name`` is not ``str`` or ``None``.
            ValueError: If ``step_name`` is an empty string.
        """
        if step_name is not None and not isinstance(step_name, str):
            raise TypeError(
                "step_name must be a str or None, "
                f"got {type(step_name).__name__}"
            )
        if step_name == "":
            raise ValueError("step_name must be a non-empty string when provided")

        if step_name is None:
            selected = self._events
        else:
            selected = [
                item for item in self._events if item.step_name == step_name
            ]

        return tuple(item.model_copy(deep=True) for item in selected)


def _require_strict_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(
            f"{name} must be an int, got {type(value).__name__}"
        )
    return value


def _validate_original_row_id_column(frame: pl.DataFrame, *, context: str) -> None:
    if ORIGINAL_ROW_ID_COLUMN not in frame.columns:
        raise DataValidationError(
            f"{context} frame is missing required column "
            f"'{ORIGINAL_ROW_ID_COLUMN}'"
        )

    dtype = frame.schema[ORIGINAL_ROW_ID_COLUMN]
    if dtype == pl.Boolean:
        raise DataValidationError(
            f"{context} frame column '{ORIGINAL_ROW_ID_COLUMN}' "
            "must be an integer type, got Boolean"
        )
    if not dtype.is_integer():
        raise DataValidationError(
            f"{context} frame column '{ORIGINAL_ROW_ID_COLUMN}' "
            f"must be an integer type, got {dtype}"
        )

    series = frame.get_column(ORIGINAL_ROW_ID_COLUMN)
    if series.null_count() > 0:
        raise DataValidationError(
            f"{context} frame column '{ORIGINAL_ROW_ID_COLUMN}' "
            "must not contain null values"
        )

    if series.n_unique() != series.len():
        raise DataValidationError(
            f"{context} frame column '{ORIGINAL_ROW_ID_COLUMN}' "
            "must contain unique values"
        )

    if series.len() > 0 and bool((series < 0).any()):
        raise DataValidationError(
            f"{context} frame column '{ORIGINAL_ROW_ID_COLUMN}' "
            "must not contain negative values"
        )


def _validate_processed_ids_subset(
    processed_frame: pl.DataFrame,
    raw_id_set: frozenset[int],
) -> None:
    processed_ids = [
        int(value)
        for value in processed_frame.get_column(ORIGINAL_ROW_ID_COLUMN).to_list()
    ]
    unknown = [row_id for row_id in processed_ids if row_id not in raw_id_set]
    if unknown:
        raise DataValidationError(
            "processed frame contains _original_row_id values "
            f"not present in the raw frame: {unknown}"
        )


def _validate_event_for_record(
    event: PreprocessingEvent,
    *,
    rows_before: int,
    rows_after: int,
) -> None:
    if event.rows_before != rows_before:
        raise DataValidationError(
            "event.rows_before must equal the current processed row count: "
            f"expected {rows_before}, got {event.rows_before}"
        )
    if event.rows_after != rows_after:
        raise DataValidationError(
            "event.rows_after must equal processed_frame.height: "
            f"expected {rows_after}, got {event.rows_after}"
        )

    timestamp: datetime = event.timestamp
    if timestamp.tzinfo is None:
        raise DataValidationError(
            "event.timestamp must be timezone-aware; got a naive datetime"
        )


def _build_mapping(current: pl.DataFrame) -> pl.DataFrame:
    height = current.height
    if height == 0:
        return pl.DataFrame(
            {
                PROCESSED_ROW_INDEX_COLUMN: pl.Series(
                    name=PROCESSED_ROW_INDEX_COLUMN,
                    values=[],
                    dtype=pl.Int64,
                ),
                ORIGINAL_ROW_ID_COLUMN: pl.Series(
                    name=ORIGINAL_ROW_ID_COLUMN,
                    values=[],
                    dtype=pl.Int64,
                ),
            }
        )

    return pl.DataFrame(
        {
            PROCESSED_ROW_INDEX_COLUMN: list(range(height)),
            ORIGINAL_ROW_ID_COLUMN: [
                int(value)
                for value in current.get_column(ORIGINAL_ROW_ID_COLUMN).to_list()
            ],
        }
    )
