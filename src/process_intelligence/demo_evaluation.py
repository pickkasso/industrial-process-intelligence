"""Shared pure helpers for demo ground-truth anomaly overlap evaluation.

Presentation-only metrics for comparing displayed anomaly-event representatives
against synthetic ``injected_anomaly`` labels. Does not train models, alter
scores, or affect workflow readiness.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence, Set
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from process_intelligence.core.exceptions import DataValidationError

_INJECTED_ANOMALY_COLUMN = "injected_anomaly"
_ANOMALY_TYPE_COLUMN = "anomaly_type"


def overlap_precision_and_enrichment(
    *,
    detected_ids: Sequence[int],
    ground_truth_ids: Set[int],
    baseline_rate: float,
) -> tuple[float | None, float | None]:
    """Return precision and enrichment for detected IDs vs ground-truth IDs.

    Precision is matched / len(detected_ids). Enrichment is precision divided by
    ``baseline_rate`` when the baseline is a finite positive rate.

    Returns:
        ``(precision, enrichment)``. Either value may be ``None`` when the
        corresponding denominator is zero or non-finite.
    """
    if not detected_ids:
        return None, None
    matched = sum(1 for row_id in detected_ids if row_id in ground_truth_ids)
    precision = float(matched) / float(len(detected_ids))
    if (
        isinstance(baseline_rate, bool)
        or not isinstance(baseline_rate, (int, float))
        or not math.isfinite(float(baseline_rate))
        or float(baseline_rate) <= 0.0
    ):
        return precision, None
    return precision, precision / float(baseline_rate)


def injected_anomaly_row_ids(frame: pd.DataFrame) -> set[int]:
    """Return original row indices where ``injected_anomaly`` is true."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"frame must be a pandas.DataFrame, got {type(frame).__name__}"
        )
    if _INJECTED_ANOMALY_COLUMN not in frame.columns:
        raise DataValidationError(
            f"demo frame is missing required column {_INJECTED_ANOMALY_COLUMN!r}"
        )
    mask = frame[_INJECTED_ANOMALY_COLUMN].to_numpy(dtype=bool)
    return {int(index) for index, flag in enumerate(mask) if bool(flag)}


def dataset_anomaly_prevalence(
    *,
    anomaly_row_count: int,
    total_row_count: int,
) -> float | None:
    """Return injected-anomaly prevalence, or ``None`` when undefined."""
    if isinstance(anomaly_row_count, bool) or not isinstance(anomaly_row_count, int):
        raise DataValidationError(
            "anomaly_row_count must be int "
            f"(bool not allowed), got {type(anomaly_row_count).__name__}"
        )
    if isinstance(total_row_count, bool) or not isinstance(total_row_count, int):
        raise DataValidationError(
            "total_row_count must be int "
            f"(bool not allowed), got {type(total_row_count).__name__}"
        )
    if anomaly_row_count < 0 or total_row_count < 0:
        raise DataValidationError(
            "anomaly_row_count and total_row_count must be >= 0"
        )
    if total_row_count == 0:
        return None
    return float(anomaly_row_count) / float(total_row_count)


def available_anomaly_types(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return sorted unique non-empty ``anomaly_type`` values in ``frame``."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"frame must be a pandas.DataFrame, got {type(frame).__name__}"
        )
    if _ANOMALY_TYPE_COLUMN not in frame.columns:
        return ()
    values = frame[_ANOMALY_TYPE_COLUMN].tolist()
    unique: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if raw is None:
            continue
        text = str(raw).strip()
        if text == "" or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return tuple(sorted(unique))


def ground_truth_lookup(
    frame: pd.DataFrame,
    *,
    row_id: int,
) -> Mapping[str, Any] | None:
    """Return ground-truth fields for ``row_id``, or ``None`` if out of range."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"frame must be a pandas.DataFrame, got {type(frame).__name__}"
        )
    if isinstance(row_id, bool) or not isinstance(row_id, int):
        raise DataValidationError(
            f"row_id must be int (bool not allowed), got {type(row_id).__name__}"
        )
    if row_id < 0 or row_id >= len(frame):
        return None
    row = frame.iloc[row_id]
    injected = bool(row[_INJECTED_ANOMALY_COLUMN]) if (
        _INJECTED_ANOMALY_COLUMN in frame.columns
    ) else False
    anomaly_type: str | None = None
    if _ANOMALY_TYPE_COLUMN in frame.columns:
        raw_type = row[_ANOMALY_TYPE_COLUMN]
        if raw_type is not None:
            text = str(raw_type).strip()
            anomaly_type = text if text != "" else None
    timestamp: object | None = None
    if "timestamp" in frame.columns:
        timestamp = row["timestamp"]
    return {
        "injected_anomaly": injected,
        "anomaly_type": anomaly_type,
        "timestamp": timestamp,
    }
