"""Deterministic synthetic manufacturing-process demo dataset generator.

Ground-truth columns ``injected_anomaly`` and ``anomaly_type`` are evaluation /
demo metadata only. Quality outputs ``quality_score`` and ``defect_rate`` are
post-process targets: select at most one as a supervised target and do not use
either as anomaly-only model features. Application users must exclude metadata
and unused quality outputs from model features; this module does not integrate
them into the analysis workflow.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from process_intelligence.core.exceptions import DataValidationError

DEMO_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp",
    "batch_id",
    "equipment_id",
)

DEMO_CONTROLLABLE_COLUMNS: Final[tuple[str, ...]] = (
    "temperature_setpoint",
    "pressure_setpoint",
    "flow_rate_setpoint",
    "cycle_time_setpoint",
)

DEMO_MEASURED_COLUMNS: Final[tuple[str, ...]] = (
    "temperature_actual",
    "pressure_actual",
    "flow_rate_actual",
    "vibration",
    "motor_current",
    "chamber_humidity",
)


@dataclass(frozen=True, slots=True)
class DemoEngineeringBound:
    """Approved synthetic-demo engineering bounds for one controllable setpoint.

    These limits are demonstration-only. They are derived from the generator
    design ranges and must not be treated as real equipment operating limits.
    """

    variable: str
    lower_bound: float
    upper_bound: float

    def __post_init__(self) -> None:
        if not isinstance(self.variable, str) or self.variable.strip() == "":
            raise DataValidationError(
                "DemoEngineeringBound.variable must be a non-empty str"
            )
        if not math.isfinite(self.lower_bound) or not math.isfinite(self.upper_bound):
            raise DataValidationError(
                "DemoEngineeringBound bounds must be finite "
                f"(got lower={self.lower_bound!r}, upper={self.upper_bound!r})"
            )
        if self.lower_bound >= self.upper_bound:
            raise DataValidationError(
                "DemoEngineeringBound.lower_bound must be < upper_bound "
                f"(got {self.lower_bound} >= {self.upper_bound})"
            )


# Fixed generator clip ranges for controllable setpoints. Shared with demo UI
# recommendation constraints so bounds cannot drift from generation design.
DEMO_SETPOINT_ENGINEERING_BOUNDS: Final[tuple[DemoEngineeringBound, ...]] = (
    DemoEngineeringBound("temperature_setpoint", 170.0, 200.0),
    DemoEngineeringBound("pressure_setpoint", 1.8, 3.2),
    DemoEngineeringBound("flow_rate_setpoint", 90.0, 150.0),
    DemoEngineeringBound("cycle_time_setpoint", 40.0, 60.0),
)

_DEMO_SETPOINT_BOUND_LOOKUP: Final[dict[str, DemoEngineeringBound]] = {
    item.variable: item for item in DEMO_SETPOINT_ENGINEERING_BOUNDS
}

DEMO_QUALITY_COLUMNS: Final[tuple[str, ...]] = (
    "quality_score",
    "defect_rate",
)

# Declared semantic domain for the synthetic quality_score target. Applied only
# through built-in demo configuration / validation requests — never inferred for
# ordinary uploaded datasets.
DEMO_QUALITY_SCORE_DECLARED_MINIMUM: Final[float] = 40.0
DEMO_QUALITY_SCORE_DECLARED_MAXIMUM: Final[float] = 100.0

GROUND_TRUTH_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "injected_anomaly",
    "anomaly_type",
)

DEMO_DATASET_COLUMNS: Final[tuple[str, ...]] = (
    *DEMO_IDENTITY_COLUMNS,
    *DEMO_CONTROLLABLE_COLUMNS,
    *DEMO_MEASURED_COLUMNS,
    *DEMO_QUALITY_COLUMNS,
    *GROUND_TRUTH_METADATA_COLUMNS,
)

ANOMALY_TYPES: Final[tuple[str, ...]] = (
    "temperature_drift",
    "pressure_spike",
    "flow_rate_drop",
    "vibration_increase",
    "sensor_bias",
    "sensor_freeze",
)

_DEFAULT_ROW_COUNT: Final[int] = 1500
_DEFAULT_RANDOM_SEED: Final[int] = 42
_DEFAULT_ANOMALY_FRACTION: Final[float] = 0.03
_DEFAULT_START_TIMESTAMP: Final[datetime] = datetime(2024, 1, 1, 8, 0, 0, tzinfo=UTC)
_DEFAULT_SAMPLING_INTERVAL_MINUTES: Final[int] = 5

_MIN_ROW_COUNT: Final[int] = 50
_MAX_ROW_COUNT: Final[int] = 100_000
_MAX_ANOMALY_FRACTION: Final[float] = 0.25
_MIN_SAMPLING_INTERVAL_MINUTES: Final[float] = 0.1
_MAX_SAMPLING_INTERVAL_MINUTES: Final[float] = 24.0 * 60.0

_EQUIPMENT_IDS: Final[tuple[str, ...]] = ("EQ-A", "EQ-B")
_BATCH_SIZE: Final[int] = 40

_MIN_EVENT_LENGTH: Final[int] = 5
_MAX_EVENT_LENGTH: Final[int] = 20
_MIN_GAP_BETWEEN_EVENTS: Final[int] = 15


@dataclass(frozen=True, slots=True)
class DemoDatasetConfiguration:
    """Configuration for a reproducible manufacturing demo dataset."""

    row_count: int = _DEFAULT_ROW_COUNT
    random_seed: int = _DEFAULT_RANDOM_SEED
    anomaly_fraction: float = _DEFAULT_ANOMALY_FRACTION
    start_timestamp: datetime = _DEFAULT_START_TIMESTAMP
    sampling_interval_minutes: float = float(_DEFAULT_SAMPLING_INTERVAL_MINUTES)

    def __post_init__(self) -> None:
        _validate_row_count(self.row_count)
        _validate_random_seed(self.random_seed)
        _validate_anomaly_fraction(self.anomaly_fraction)
        _validate_start_timestamp(self.start_timestamp)
        _validate_sampling_interval_minutes(self.sampling_interval_minutes)


@dataclass(frozen=True, slots=True)
class _AnomalyEvent:
    start_index: int
    end_index: int  # exclusive
    anomaly_type: str


def generate_demo_dataset(configuration: DemoDatasetConfiguration) -> pd.DataFrame:
    """Generate a deterministic tabular manufacturing demo dataset.

    Parameters
    ----------
    configuration:
        Validated generation settings.

    Returns
    -------
    pandas.DataFrame
        Dataset with identity, controllable, measured, quality, and ground-truth
        metadata columns. Ground-truth columns must not be used as model features.
    """
    if not isinstance(configuration, DemoDatasetConfiguration):
        raise TypeError(
            "configuration must be DemoDatasetConfiguration, "
            f"got {type(configuration).__name__}"
        )

    rng = np.random.default_rng(configuration.random_seed)
    n_rows = configuration.row_count

    timestamps = _build_timestamps(
        start=configuration.start_timestamp,
        interval_minutes=configuration.sampling_interval_minutes,
        row_count=n_rows,
    )
    equipment_ids = _assign_equipment_ids(n_rows=n_rows, rng=rng)
    batch_ids = _assign_batch_ids(n_rows=n_rows, equipment_ids=equipment_ids)

    setpoints = _generate_setpoints(n_rows=n_rows, equipment_ids=equipment_ids, rng=rng)
    measurements = _generate_normal_measurements(setpoints=setpoints, rng=rng)

    events = _plan_anomaly_events(
        n_rows=n_rows,
        anomaly_fraction=configuration.anomaly_fraction,
        rng=rng,
    )
    injected_anomaly, anomaly_type = _apply_anomaly_events(
        measurements=measurements,
        events=events,
        rng=rng,
    )

    quality_score, defect_rate = _compute_quality_outputs(
        setpoints=setpoints,
        measurements=measurements,
        injected_anomaly=injected_anomaly,
        rng=rng,
    )

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "batch_id": batch_ids,
            "equipment_id": equipment_ids,
            "temperature_setpoint": setpoints["temperature_setpoint"],
            "pressure_setpoint": setpoints["pressure_setpoint"],
            "flow_rate_setpoint": setpoints["flow_rate_setpoint"],
            "cycle_time_setpoint": setpoints["cycle_time_setpoint"],
            "temperature_actual": measurements["temperature_actual"],
            "pressure_actual": measurements["pressure_actual"],
            "flow_rate_actual": measurements["flow_rate_actual"],
            "vibration": measurements["vibration"],
            "motor_current": measurements["motor_current"],
            "chamber_humidity": measurements["chamber_humidity"],
            "quality_score": quality_score,
            "defect_rate": defect_rate,
            "injected_anomaly": injected_anomaly,
            "anomaly_type": anomaly_type,
        }
    )
    return frame.loc[:, list(DEMO_DATASET_COLUMNS)].copy()


def summarize_demo_dataset(frame: pd.DataFrame) -> dict[str, object]:
    """Return concise summary statistics for a generated demo dataset."""
    _require_demo_columns(frame)
    anomaly_mask = frame["injected_anomaly"].to_numpy(dtype=bool)
    anomaly_row_count = int(anomaly_mask.sum())
    type_series = frame.loc[anomaly_mask, "anomaly_type"]
    type_counts = {str(name): int(count) for name, count in type_series.value_counts().items()}
    event_count = _count_contiguous_anomaly_events(anomaly_mask)
    return {
        "row_count": int(len(frame)),
        "column_count": int(frame.shape[1]),
        "anomaly_row_count": anomaly_row_count,
        "anomaly_event_count": event_count,
        "anomaly_type_counts": type_counts,
    }


def write_demo_workflow_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write a demo frame as a workflow-compatible CSV.

    Timestamps are serialized as UTC unix seconds so the existing TIME split
    path can consume the file without changing the DatasetLoader.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame must be a pandas.DataFrame, got {type(frame).__name__}")
    if not isinstance(path, Path):
        raise TypeError(f"path must be Path, got {type(path).__name__}")
    _require_demo_columns(frame)
    export = frame.copy()
    timestamps = pd.to_datetime(export["timestamp"], utc=True)
    export["timestamp"] = (timestamps.astype("int64") // 10**9).astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(path, index=False, encoding="utf-8")
    return path


def _validate_row_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(
            f"row_count must be an int (bool not allowed), got {type(value).__name__}"
        )
    if value < _MIN_ROW_COUNT or value > _MAX_ROW_COUNT:
        raise DataValidationError(
            f"row_count must be in [{_MIN_ROW_COUNT}, {_MAX_ROW_COUNT}], got {value}"
        )
    return value


def _validate_random_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(
            f"random_seed must be an int (bool not allowed), got {type(value).__name__}"
        )
    return value


def _validate_anomaly_fraction(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataValidationError(
            "anomaly_fraction must be a finite float in "
            f"[0.0, {_MAX_ANOMALY_FRACTION}] (bool not allowed), "
            f"got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise DataValidationError(f"anomaly_fraction must be finite, got {value!r}")
    if number < 0.0 or number > _MAX_ANOMALY_FRACTION:
        raise DataValidationError(
            f"anomaly_fraction must be in [0.0, {_MAX_ANOMALY_FRACTION}], got {number}"
        )
    return number


def _validate_start_timestamp(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise DataValidationError(
            f"start_timestamp must be datetime, got {type(value).__name__}"
        )
    if value.tzinfo is None:
        raise DataValidationError("start_timestamp must be timezone-aware")
    return value


def _validate_sampling_interval_minutes(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataValidationError(
            "sampling_interval_minutes must be a finite float > 0 "
            f"(bool not allowed), got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise DataValidationError(
            f"sampling_interval_minutes must be finite, got {value!r}"
        )
    if number < _MIN_SAMPLING_INTERVAL_MINUTES or number > _MAX_SAMPLING_INTERVAL_MINUTES:
        raise DataValidationError(
            "sampling_interval_minutes must be in "
            f"[{_MIN_SAMPLING_INTERVAL_MINUTES}, {_MAX_SAMPLING_INTERVAL_MINUTES}], "
            f"got {number}"
        )
    return number


def _build_timestamps(
    *,
    start: datetime,
    interval_minutes: float,
    row_count: int,
) -> list[datetime]:
    delta = timedelta(minutes=interval_minutes)
    return [start + (delta * index) for index in range(row_count)]


def _assign_equipment_ids(*, n_rows: int, rng: np.random.Generator) -> np.ndarray:
    # Stable-ish regime assignment with mild temporal blocks, not a perfect anomaly label.
    block_size = max(25, n_rows // 12)
    ids: list[str] = []
    current = str(rng.choice(_EQUIPMENT_IDS))
    remaining = block_size
    for _ in range(n_rows):
        if remaining <= 0:
            current = str(rng.choice(_EQUIPMENT_IDS))
            remaining = int(rng.integers(block_size // 2, block_size + 1))
        ids.append(current)
        remaining -= 1
    return np.asarray(ids, dtype=object)


def _assign_batch_ids(*, n_rows: int, equipment_ids: np.ndarray) -> np.ndarray:
    batch_ids: list[str] = []
    batch_counters = {equipment: 1 for equipment in _EQUIPMENT_IDS}
    for index in range(n_rows):
        equipment = str(equipment_ids[index])
        batch_number = batch_counters[equipment]
        batch_ids.append(f"{equipment}-B{batch_number:04d}")
        if (index + 1) % _BATCH_SIZE == 0:
            batch_counters[equipment] += 1
    return np.asarray(batch_ids, dtype=object)


def _generate_setpoints(
    *,
    n_rows: int,
    equipment_ids: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    is_eq_a = equipment_ids == "EQ-A"
    temperature = np.where(is_eq_a, 185.0, 182.0) + rng.normal(0.0, 0.8, size=n_rows)
    pressure = np.where(is_eq_a, 2.40, 2.55) + rng.normal(0.0, 0.04, size=n_rows)
    flow_rate = np.where(is_eq_a, 120.0, 115.0) + rng.normal(0.0, 1.5, size=n_rows)
    cycle_time = np.where(is_eq_a, 48.0, 50.0) + rng.normal(0.0, 0.6, size=n_rows)

    # Mild slow drift in setpoints (shared process schedule), not an anomaly label.
    schedule = np.linspace(0.0, 1.0, n_rows)
    temperature = temperature + 1.5 * np.sin(2.0 * math.pi * schedule)
    pressure = pressure + 0.05 * np.cos(2.0 * math.pi * schedule)
    flow_rate = flow_rate + 2.0 * np.sin(2.0 * math.pi * schedule + 0.4)

    temperature_bounds = _DEMO_SETPOINT_BOUND_LOOKUP["temperature_setpoint"]
    pressure_bounds = _DEMO_SETPOINT_BOUND_LOOKUP["pressure_setpoint"]
    flow_bounds = _DEMO_SETPOINT_BOUND_LOOKUP["flow_rate_setpoint"]
    cycle_bounds = _DEMO_SETPOINT_BOUND_LOOKUP["cycle_time_setpoint"]
    return {
        "temperature_setpoint": np.clip(
            temperature,
            temperature_bounds.lower_bound,
            temperature_bounds.upper_bound,
        ),
        "pressure_setpoint": np.clip(
            pressure,
            pressure_bounds.lower_bound,
            pressure_bounds.upper_bound,
        ),
        "flow_rate_setpoint": np.clip(
            flow_rate,
            flow_bounds.lower_bound,
            flow_bounds.upper_bound,
        ),
        "cycle_time_setpoint": np.clip(
            cycle_time,
            cycle_bounds.lower_bound,
            cycle_bounds.upper_bound,
        ),
    }


def demo_setpoint_engineering_bounds() -> tuple[DemoEngineeringBound, ...]:
    """Return the approved synthetic-demo setpoint engineering bounds."""
    return DEMO_SETPOINT_ENGINEERING_BOUNDS


def demo_setpoint_bound_map() -> dict[str, tuple[float, float]]:
    """Return ``{variable: (lower, upper)}`` for approved demo setpoint bounds."""
    return {
        item.variable: (item.lower_bound, item.upper_bound)
        for item in DEMO_SETPOINT_ENGINEERING_BOUNDS
    }


def _generate_normal_measurements(
    *,
    setpoints: Mapping[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    n_rows = int(setpoints["temperature_setpoint"].shape[0])
    temperature_noise = rng.normal(0.0, 0.9, size=n_rows)
    pressure_noise = rng.normal(0.0, 0.05, size=n_rows)
    flow_noise = rng.normal(0.0, 1.8, size=n_rows)

    # Correlated process behavior: higher temperature tends to raise motor current
    # and vibration; lower flow tends to raise humidity slightly.
    temperature_actual = setpoints["temperature_setpoint"] + temperature_noise
    pressure_actual = setpoints["pressure_setpoint"] + pressure_noise
    flow_rate_actual = setpoints["flow_rate_setpoint"] + flow_noise

    temp_delta = temperature_actual - setpoints["temperature_setpoint"]
    flow_delta = flow_rate_actual - setpoints["flow_rate_setpoint"]

    vibration = (
        0.35
        + 0.04 * np.abs(temp_delta)
        + 0.02 * np.abs(pressure_actual - setpoints["pressure_setpoint"])
        + rng.normal(0.0, 0.04, size=n_rows)
    )
    motor_current = (
        18.0
        + 0.12 * (temperature_actual - 180.0)
        + 0.08 * (pressure_actual - 2.4)
        + 0.4 * vibration
        + rng.normal(0.0, 0.25, size=n_rows)
    )
    chamber_humidity = (
        35.0
        - 0.08 * flow_delta
        + 0.05 * (temperature_actual - 183.0)
        + rng.normal(0.0, 0.8, size=n_rows)
    )

    return {
        "temperature_actual": np.clip(temperature_actual, 160.0, 220.0),
        "pressure_actual": np.clip(pressure_actual, 1.5, 3.8),
        "flow_rate_actual": np.clip(flow_rate_actual, 70.0, 170.0),
        "vibration": np.clip(vibration, 0.05, 3.0),
        "motor_current": np.clip(motor_current, 10.0, 40.0),
        "chamber_humidity": np.clip(chamber_humidity, 15.0, 70.0),
    }


def _plan_anomaly_events(
    *,
    n_rows: int,
    anomaly_fraction: float,
    rng: np.random.Generator,
) -> list[_AnomalyEvent]:
    target_rows = int(round(n_rows * anomaly_fraction))
    if target_rows <= 0:
        return []

    max_event_length = min(_MAX_EVENT_LENGTH, max(_MIN_EVENT_LENGTH, n_rows // 8))
    min_event_length = min(_MIN_EVENT_LENGTH, max_event_length)
    if min_event_length < 2:
        min_event_length = 2

    type_order = list(ANOMALY_TYPES)
    rng.shuffle(type_order)

    # Ensure every implemented anomaly type appears when the budget allows.
    if target_rows >= min_event_length * len(ANOMALY_TYPES):
        planned_types: list[str] = list(type_order)
    else:
        planned_types = [type_order[0]]

    remaining_after_base = target_rows - (len(planned_types) * min_event_length)
    while remaining_after_base >= min_event_length:
        planned_types.append(str(rng.choice(ANOMALY_TYPES)))
        remaining_after_base -= min_event_length

    # Spread events across the full timeline so TIME-split validation/test
    # partitions also contain injected anomalies for end-to-end evaluation.
    n_events = len(planned_types)
    usable_start = max(1, n_rows // 25)
    usable_end = max(usable_start + 2, n_rows - max(2, n_rows // 25))
    usable_span = max(1, usable_end - usable_start)

    lengths: list[int] = []
    remaining_budget = target_rows
    for index in range(n_events):
        events_left = n_events - index
        reserved = max(0, (events_left - 1) * min_event_length)
        available = max(min_event_length, remaining_budget - reserved)
        upper = min(max_event_length, available)
        if upper < 2:
            break
        lower = min(min_event_length, upper)
        length = int(rng.integers(lower, upper + 1))
        length = min(length, remaining_budget)
        if length < 2:
            break
        lengths.append(length)
        remaining_budget -= length

    if not lengths:
        return []

    centers = np.linspace(usable_start, usable_end - 1, num=len(lengths))
    jitter_scale = max(1.0, usable_span / (8.0 * len(lengths)))
    centers = centers + rng.normal(0.0, jitter_scale, size=len(lengths))
    centers = np.clip(centers, usable_start, usable_end - 1)

    tentative: list[tuple[int, int, str]] = []
    for index, length in enumerate(lengths):
        ideal_start = int(round(float(centers[index]) - (length / 2.0)))
        max_start = max(usable_start, n_rows - length)
        start = min(max(usable_start, ideal_start), max_start)
        end = min(n_rows, start + length)
        if end - start < 2:
            continue
        tentative.append((start, end, planned_types[index]))

    tentative.sort(key=lambda item: item[0])
    events: list[_AnomalyEvent] = []
    for start, end, anomaly_type in tentative:
        original_length = end - start
        if events and start < events[-1].end_index + _MIN_GAP_BETWEEN_EVENTS:
            start = events[-1].end_index + _MIN_GAP_BETWEEN_EVENTS
            end = start + original_length
            if end > n_rows:
                continue
        if end - start < 2:
            continue
        events.append(
            _AnomalyEvent(
                start_index=start,
                end_index=end,
                anomaly_type=anomaly_type,
            )
        )
    return events


def _apply_anomaly_events(
    *,
    measurements: dict[str, np.ndarray],
    events: Sequence[_AnomalyEvent],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = int(measurements["temperature_actual"].shape[0])
    injected = np.zeros(n_rows, dtype=bool)
    anomaly_type = np.full(n_rows, "", dtype=object)

    for event in events:
        start = event.start_index
        end = event.end_index
        injected[start:end] = True
        anomaly_type[start:end] = event.anomaly_type
        length = end - start
        progress = np.linspace(0.0, 1.0, length)

        if event.anomaly_type == "temperature_drift":
            measurements["temperature_actual"][start:end] = np.clip(
                measurements["temperature_actual"][start:end] + 8.0 * progress,
                160.0,
                220.0,
            )
            measurements["motor_current"][start:end] = np.clip(
                measurements["motor_current"][start:end] + 1.5 * progress,
                10.0,
                40.0,
            )
        elif event.anomaly_type == "pressure_spike":
            measurements["pressure_actual"][start:end] = np.clip(
                measurements["pressure_actual"][start:end] + 0.55 + 0.15 * progress,
                1.5,
                3.8,
            )
            measurements["vibration"][start:end] = np.clip(
                measurements["vibration"][start:end] + 0.25,
                0.05,
                3.0,
            )
        elif event.anomaly_type == "flow_rate_drop":
            measurements["flow_rate_actual"][start:end] = np.clip(
                measurements["flow_rate_actual"][start:end] - (18.0 + 6.0 * progress),
                70.0,
                170.0,
            )
            measurements["chamber_humidity"][start:end] = np.clip(
                measurements["chamber_humidity"][start:end] + 4.0 * progress,
                15.0,
                70.0,
            )
        elif event.anomaly_type == "vibration_increase":
            measurements["vibration"][start:end] = np.clip(
                measurements["vibration"][start:end] + 0.9 + 0.4 * progress,
                0.05,
                3.0,
            )
            measurements["motor_current"][start:end] = np.clip(
                measurements["motor_current"][start:end] + 2.0 * progress,
                10.0,
                40.0,
            )
        elif event.anomaly_type == "sensor_bias":
            bias = float(rng.uniform(4.0, 7.0))
            measurements["chamber_humidity"][start:end] = np.clip(
                measurements["chamber_humidity"][start:end] + bias,
                15.0,
                70.0,
            )
        elif event.anomaly_type == "sensor_freeze":
            frozen_value = float(measurements["temperature_actual"][start])
            measurements["temperature_actual"][start:end] = frozen_value
        else:  # pragma: no cover - defensive guard for enum drift
            raise DataValidationError(f"unsupported anomaly_type: {event.anomaly_type!r}")

    return injected, anomaly_type


def _compute_quality_outputs(
    *,
    setpoints: Mapping[str, np.ndarray],
    measurements: Mapping[str, np.ndarray],
    injected_anomaly: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = int(measurements["temperature_actual"].shape[0])

    # Near-linear process→quality mapping expressed directly in observed
    # feature space so default supervised candidates beat a mean baseline
    # under the workflow TIME split without algorithm-specific tuning.
    temp_gap = measurements["temperature_actual"] - setpoints["temperature_setpoint"]
    pressure_gap = measurements["pressure_actual"] - setpoints["pressure_setpoint"]
    flow_gap = measurements["flow_rate_actual"] - setpoints["flow_rate_setpoint"]
    cycle_gap = setpoints["cycle_time_setpoint"] - 49.0

    quality = (
        94.5
        - 0.55 * temp_gap
        - 12.0 * pressure_gap
        - 0.22 * flow_gap
        - 0.35 * cycle_gap
        - 18.0 * np.maximum(0.0, measurements["vibration"] - 0.45)
        - 0.45 * (measurements["motor_current"] - 20.0)
        - 0.18 * (measurements["chamber_humidity"] - 35.0)
        + rng.normal(0.0, 0.35, size=n_rows)
    )
    # Mild unexplained quality shock on injected anomalies keeps residual
    # anomaly detection informative even when the main mapping is well fit.
    anomaly_mask = np.asarray(injected_anomaly, dtype=bool)
    if bool(anomaly_mask.any()):
        shock = rng.normal(3.2, 0.35, size=int(anomaly_mask.sum()))
        quality = quality.copy()
        quality[anomaly_mask] = quality[anomaly_mask] - np.abs(shock)
    quality_score = np.clip(quality, 40.0, 100.0)

    defect_logits = (
        -4.0
        + 0.12 * np.abs(temp_gap)
        + 1.6 * np.abs(pressure_gap)
        + 0.05 * np.abs(flow_gap)
        + 2.2 * np.maximum(0.0, measurements["vibration"] - 0.45)
        + rng.normal(0.0, 0.12, size=n_rows)
    )
    defect_rate = 1.0 / (1.0 + np.exp(-defect_logits))
    defect_rate = np.clip(defect_rate, 0.001, 0.35)

    return quality_score, defect_rate


def _require_demo_columns(frame: pd.DataFrame) -> None:
    missing = [name for name in DEMO_DATASET_COLUMNS if name not in frame.columns]
    if missing:
        raise DataValidationError(
            f"frame is missing required demo columns: {missing}"
        )


def _count_contiguous_anomaly_events(anomaly_mask: np.ndarray) -> int:
    if anomaly_mask.size == 0:
        return 0
    transitions = np.diff(anomaly_mask.astype(np.int8), prepend=0)
    return int(np.sum(transitions == 1))
