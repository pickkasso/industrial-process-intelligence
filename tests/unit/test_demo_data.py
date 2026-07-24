"""Unit tests for the deterministic manufacturing demo dataset generator."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from process_intelligence.core.exceptions import DataValidationError
from process_intelligence.demo_data import (
    ANOMALY_TYPES,
    DEMO_CONTROLLABLE_COLUMNS,
    DEMO_DATASET_COLUMNS,
    DEMO_SETPOINT_ENGINEERING_BOUNDS,
    GROUND_TRUTH_METADATA_COLUMNS,
    DemoDatasetConfiguration,
    demo_setpoint_bound_map,
    generate_demo_dataset,
)


def _default_config(**overrides: object) -> DemoDatasetConfiguration:
    payload: dict[str, object] = {
        "row_count": 400,
        "random_seed": 42,
        "anomaly_fraction": 0.04,
        "start_timestamp": datetime(2024, 1, 1, 8, 0, 0, tzinfo=UTC),
        "sampling_interval_minutes": 5.0,
    }
    payload.update(overrides)
    return DemoDatasetConfiguration(**payload)  # type: ignore[arg-type]


def test_same_seed_produces_identical_dataframe() -> None:
    config = _default_config(random_seed=7)
    first = generate_demo_dataset(config)
    second = generate_demo_dataset(config)
    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_produce_different_numeric_data() -> None:
    first = generate_demo_dataset(_default_config(random_seed=1))
    second = generate_demo_dataset(_default_config(random_seed=2))
    numeric_columns = [
        name
        for name in DEMO_DATASET_COLUMNS
        if name not in {"timestamp", "batch_id", "equipment_id", *GROUND_TRUTH_METADATA_COLUMNS}
    ]
    assert any(
        not np.allclose(
            first[column].to_numpy(dtype=float),
            second[column].to_numpy(dtype=float),
            equal_nan=True,
        )
        for column in numeric_columns
        if pd.api.types.is_numeric_dtype(first[column])
    )


def test_expected_columns_and_row_count() -> None:
    frame = generate_demo_dataset(_default_config(row_count=220))
    assert list(frame.columns) == list(DEMO_DATASET_COLUMNS)
    assert len(frame) == 220


def test_timestamp_is_monotonic_increasing() -> None:
    frame = generate_demo_dataset(_default_config())
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    assert timestamps.is_monotonic_increasing
    assert timestamps.is_unique


def test_anomaly_fraction_within_tolerance() -> None:
    config = _default_config(row_count=800, anomaly_fraction=0.03)
    frame = generate_demo_dataset(config)
    observed = float(frame["injected_anomaly"].mean())
    assert observed == pytest.approx(config.anomaly_fraction, abs=0.02)


def test_anomalies_occur_in_contiguous_runs() -> None:
    frame = generate_demo_dataset(_default_config(row_count=600, anomaly_fraction=0.05))
    mask = frame["injected_anomaly"].to_numpy(dtype=bool)
    assert mask.any()
    assert (~mask).any()

    run_lengths: list[int] = []
    index = 0
    while index < len(mask):
        if not mask[index]:
            index += 1
            continue
        start = index
        while index < len(mask) and mask[index]:
            index += 1
        run_lengths.append(index - start)

    assert run_lengths
    assert all(length >= 2 for length in run_lengths)
    assert max(run_lengths) >= 2


def test_anomaly_types_consistent_with_flag() -> None:
    frame = generate_demo_dataset(_default_config())
    anomalous = frame["injected_anomaly"].astype(bool)
    assert set(frame.loc[anomalous, "anomaly_type"].unique()).issubset(set(ANOMALY_TYPES))
    assert (frame.loc[~anomalous, "anomaly_type"] == "").all()
    assert (frame.loc[anomalous, "anomaly_type"] != "").all()


def test_numeric_columns_are_finite() -> None:
    frame = generate_demo_dataset(_default_config())
    numeric = frame.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy(dtype=float)).all()


def test_configuration_validation_rejects_invalid_inputs() -> None:
    with pytest.raises(DataValidationError):
        DemoDatasetConfiguration(row_count=10)
    with pytest.raises(DataValidationError):
        DemoDatasetConfiguration(anomaly_fraction=-0.1)
    with pytest.raises(DataValidationError):
        DemoDatasetConfiguration(anomaly_fraction=0.9)
    with pytest.raises(DataValidationError):
        DemoDatasetConfiguration(sampling_interval_minutes=0.0)
    with pytest.raises(DataValidationError):
        DemoDatasetConfiguration(start_timestamp=datetime(2024, 1, 1, 8, 0, 0))
    with pytest.raises(DataValidationError):
        DemoDatasetConfiguration(row_count=True)  # type: ignore[arg-type]


def test_quality_outputs_are_not_copies_of_anomaly_flag() -> None:
    frame = generate_demo_dataset(_default_config(row_count=500, anomaly_fraction=0.04))
    anomaly = frame["injected_anomaly"].astype(float)
    quality = frame["quality_score"].astype(float)
    defect = frame["defect_rate"].astype(float)

    assert not np.allclose(quality, anomaly)
    assert not np.allclose(defect, anomaly)
    assert quality.nunique() > 2
    assert defect.nunique() > 2

    # Quality should vary within both normal and anomalous cohorts.
    assert frame.loc[~frame["injected_anomaly"], "quality_score"].nunique() > 1
    assert frame.loc[frame["injected_anomaly"], "quality_score"].nunique() > 1


def test_default_scale_includes_all_anomaly_types() -> None:
    frame = generate_demo_dataset(
        DemoDatasetConfiguration(
            row_count=1500,
            random_seed=42,
            anomaly_fraction=0.03,
            start_timestamp=datetime(2024, 1, 1, 8, 0, 0, tzinfo=UTC),
            sampling_interval_minutes=5.0,
        )
    )
    observed = set(frame.loc[frame["injected_anomaly"], "anomaly_type"].unique())
    assert observed == set(ANOMALY_TYPES)


def test_multiple_equipment_regimes_present() -> None:
    frame = generate_demo_dataset(_default_config(row_count=300))
    assert set(frame["equipment_id"].unique()) == {"EQ-A", "EQ-B"}


def test_zero_anomaly_fraction_produces_no_anomalies() -> None:
    frame = generate_demo_dataset(_default_config(anomaly_fraction=0.0))
    assert int(frame["injected_anomaly"].sum()) == 0
    assert (frame["anomaly_type"] == "").all()


def test_shared_setpoint_engineering_bounds_match_generator_design() -> None:
    bounds = demo_setpoint_bound_map()
    assert tuple(bounds.keys()) == DEMO_CONTROLLABLE_COLUMNS
    assert bounds == {
        "temperature_setpoint": (170.0, 200.0),
        "pressure_setpoint": (1.8, 3.2),
        "flow_rate_setpoint": (90.0, 150.0),
        "cycle_time_setpoint": (40.0, 60.0),
    }
    for item in DEMO_SETPOINT_ENGINEERING_BOUNDS:
        assert item.lower_bound < item.upper_bound


def test_generated_setpoints_stay_inside_shared_engineering_bounds() -> None:
    frame = generate_demo_dataset(_default_config(row_count=800, random_seed=7))
    bounds = demo_setpoint_bound_map()
    for name, (low, high) in bounds.items():
        values = frame[name].to_numpy(dtype=float)
        assert np.isfinite(values).all()
        assert float(values.min()) >= low
        assert float(values.max()) <= high
