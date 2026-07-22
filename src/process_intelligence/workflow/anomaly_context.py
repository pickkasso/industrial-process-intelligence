"""Anomaly event context-window extraction for interpretation (Step 11B.9).

Builds compact, JSON-safe context windows around selected anomaly events using
the workflow analysis-order frame and original preprocessed-input values.
Context is interpretive only and must not feed modeling, scoring, or diagnosis.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import polars as pl

from process_intelligence.core.schemas import AnomalyEvent, RootCauseFactor
from process_intelligence.data.loader import ORIGINAL_ROW_ID_COLUMN
from process_intelligence.workflow.enums import AnomalyContextOrderBasis
from process_intelligence.workflow.schemas import (
    ANOMALY_CONTEXT_MAX_FEATURES,
    ANOMALY_CONTEXT_RADIUS,
    AnomalyContextIdentifierValue,
    AnomalyContextRow,
    AnomalyContextValue,
    AnomalyContextWindow,
    ScalarMetadataValue,
)


def build_anomaly_context_windows(
    *,
    analysis_frame: pl.DataFrame,
    anomaly_events: list[AnomalyEvent],
    diagnosis_factors: list[RootCauseFactor],
    order_basis: AnomalyContextOrderBasis,
    timestamp_column: str | None,
    identifier_columns: list[str],
    radius: int = ANOMALY_CONTEXT_RADIUS,
    max_features: int = ANOMALY_CONTEXT_MAX_FEATURES,
) -> tuple[list[AnomalyContextWindow], list[str]]:
    """Extract per-event analysis-order context windows from original values.

    Args:
        analysis_frame: Analysis-order frame with original (pre-preprocess)
            values and ``_original_row_id``.
        anomaly_events: Selected anomaly events in workflow order.
        diagnosis_factors: Ordered diagnosis factors used to choose features.
        order_basis: Explicit adjacency basis for the window rows.
        timestamp_column: Optional request timestamp column to include.
        identifier_columns: Request identifier columns; constant or all-null
            columns are omitted.
        radius: Fixed context radius (center ± radius). Defaults to 3.
        max_features: Maximum diagnosis features to include. Defaults to 5.

    Returns:
        Context windows for events whose center row was found, plus warnings
        for missing centers. Does not mutate ``analysis_frame`` or events.
    """
    if radius != ANOMALY_CONTEXT_RADIUS:
        raise ValueError(
            f"radius must equal ANOMALY_CONTEXT_RADIUS ({ANOMALY_CONTEXT_RADIUS}), "
            f"got {radius}"
        )
    if max_features < 0:
        raise ValueError(f"max_features must be >= 0, got {max_features}")
    if ORIGINAL_ROW_ID_COLUMN not in analysis_frame.columns:
        raise ValueError(
            f"analysis_frame requires reserved column {ORIGINAL_ROW_ID_COLUMN!r}"
        )

    feature_names = _select_context_feature_names(
        diagnosis_factors=diagnosis_factors,
        frame_columns=set(analysis_frame.columns),
        max_features=max_features,
    )
    included_identifiers = _select_identifier_columns(
        analysis_frame,
        identifier_columns=identifier_columns,
    )
    included_timestamp = (
        timestamp_column
        if timestamp_column is not None and timestamp_column in analysis_frame.columns
        else None
    )

    row_ids = [
        int(value) for value in analysis_frame.get_column(ORIGINAL_ROW_ID_COLUMN).to_list()
    ]
    position_by_id: dict[int, int] = {}
    for position, row_id in enumerate(row_ids):
        if row_id not in position_by_id:
            position_by_id[row_id] = position

    selected_ids = {_event_original_row_id(event) for event in anomaly_events}
    selected_ids.discard(None)

    windows: list[AnomalyContextWindow] = []
    warnings: list[str] = []
    for rank, event in enumerate(anomaly_events, start=1):
        center_id = _event_original_row_id(event)
        if center_id is None or center_id not in position_by_id:
            warnings.append(
                "Anomaly context window was not created because original row ID "
                f"{center_id!r} for event rank {rank} was not found in the "
                "analysis-order frame."
            )
            continue
        center_position = position_by_id[center_id]
        start = max(0, center_position - radius)
        end = min(analysis_frame.height, center_position + radius + 1)
        window_rows: list[AnomalyContextRow] = []
        for position in range(start, end):
            row = analysis_frame.row(position, named=True)
            original_row_id = int(row[ORIGINAL_ROW_ID_COLUMN])
            relative_offset = position - center_position
            is_center = relative_offset == 0
            is_selected = original_row_id in selected_ids
            identifier_values = tuple(
                AnomalyContextIdentifierValue(
                    column_name=name,
                    value=_to_json_safe_context_scalar(row.get(name)),
                )
                for name in included_identifiers
            )
            timestamp_value: ScalarMetadataValue = None
            if included_timestamp is not None:
                timestamp_value = _to_json_safe_context_scalar(row.get(included_timestamp))
            feature_values = tuple(
                AnomalyContextValue(
                    feature_name=name,
                    value=_to_json_safe_context_scalar(row.get(name)),
                )
                for name in feature_names
            )
            window_rows.append(
                AnomalyContextRow(
                    analysis_position=position,
                    original_row_id=original_row_id,
                    relative_offset=relative_offset,
                    is_center_event=is_center,
                    is_selected_anomaly_event=is_selected,
                    identifier_values=list(identifier_values),
                    timestamp_value=timestamp_value,
                    feature_values=list(feature_values),
                )
            )
        windows.append(
            AnomalyContextWindow(
                event_rank=rank,
                center_original_row_id=center_id,
                center_anomaly_score=float(event.anomaly_score),
                radius=radius,
                order_basis=order_basis,
                feature_names=list(feature_names),
                rows=window_rows,
            )
        )
    return windows, warnings


def _select_context_feature_names(
    *,
    diagnosis_factors: list[RootCauseFactor],
    frame_columns: set[str],
    max_features: int,
) -> tuple[str, ...]:
    selected: list[str] = []
    seen: set[str] = set()
    for factor in diagnosis_factors:
        name = factor.variable
        if name in seen:
            continue
        if name not in frame_columns:
            continue
        if name == ORIGINAL_ROW_ID_COLUMN:
            continue
        seen.add(name)
        selected.append(name)
        if len(selected) >= max_features:
            break
    return tuple(selected)


def _select_identifier_columns(
    frame: pl.DataFrame,
    *,
    identifier_columns: list[str],
) -> tuple[str, ...]:
    selected: list[str] = []
    for name in identifier_columns:
        if name not in frame.columns:
            continue
        series = frame.get_column(name)
        if series.len() == 0:
            continue
        if series.null_count() == series.len():
            continue
        non_null = series.drop_nulls()
        if non_null.n_unique() <= 1:
            continue
        selected.append(name)
    return tuple(selected)


def _event_original_row_id(event: AnomalyEvent) -> int | None:
    sample_id = event.sample_id
    if isinstance(sample_id, bool):
        return None
    if isinstance(sample_id, int):
        return sample_id
    if isinstance(sample_id, str):
        text = sample_id.strip()
        if text == "":
            return None
        try:
            return int(text)
        except ValueError:
            return None
    anomaly_id = event.anomaly_id.strip()
    if anomaly_id == "":
        return None
    try:
        return int(anomaly_id)
    except ValueError:
        return None


def _to_json_safe_context_scalar(value: Any) -> ScalarMetadataValue:
    """Convert a raw cell value to a JSON-safe scalar (NaN/Inf → None)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _to_json_safe_context_scalar(item())
        except (TypeError, ValueError):
            return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(as_float):
        return None
    if as_float.is_integer():
        return int(as_float)
    return as_float
