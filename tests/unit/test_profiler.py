"""Unit tests for DatasetProfiler (Step 2B)."""

from __future__ import annotations

import polars as pl
import pytest

from process_intelligence.data import (
    ORIGINAL_ROW_ID_COLUMN,
    ColumnProfile,
    DatasetProfile,
    DatasetProfiler,
)


def _sample_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "lot_id": ["A", "B", "A", "C"],
            "temperature": [100.0, 110.0, None, 105.0],
            "status": [True, False, True, False],
            "constant_flag": [1, 1, 1, 1],
            ORIGINAL_ROW_ID_COLUMN: [0, 1, 2, 3],
        }
    )


def test_dataset_profiler_default_construction() -> None:
    profiler = DatasetProfiler()
    assert isinstance(profiler, DatasetProfiler)


def test_preview_rows_must_be_positive() -> None:
    with pytest.raises(ValueError):
        DatasetProfiler(preview_rows=0)
    with pytest.raises(ValueError):
        DatasetProfiler(preview_rows=-1)


def test_sample_values_per_column_must_be_positive() -> None:
    with pytest.raises(ValueError):
        DatasetProfiler(sample_values_per_column=0)
    with pytest.raises(ValueError):
        DatasetProfiler(sample_values_per_column=-3)


def test_non_polars_dataframe_raises_type_error() -> None:
    profiler = DatasetProfiler()
    with pytest.raises(TypeError):
        profiler.profile({"a": [1, 2]})  # type: ignore[arg-type]


def test_row_and_column_counts_are_accurate() -> None:
    frame = _sample_frame()
    profile = DatasetProfiler().profile(frame)

    assert profile.row_count == frame.height
    assert profile.column_count == frame.width


def test_original_column_order_is_preserved() -> None:
    frame = _sample_frame()
    profile = DatasetProfiler().profile(frame)

    assert [column.name for column in profile.columns] == frame.columns


def test_dtype_string_is_recorded() -> None:
    frame = _sample_frame()
    profile = DatasetProfiler().profile(frame)
    by_name = {column.name: column for column in profile.columns}

    assert by_name["lot_id"].dtype == str(frame.schema["lot_id"])
    assert by_name["temperature"].dtype == str(frame.schema["temperature"])
    assert by_name["status"].dtype == str(frame.schema["status"])


def test_null_count_is_accurate() -> None:
    frame = _sample_frame()
    profile = DatasetProfiler().profile(frame)
    by_name = {column.name: column for column in profile.columns}

    assert by_name["temperature"].null_count == 1
    assert by_name["lot_id"].null_count == 0


def test_null_ratio_is_accurate() -> None:
    frame = _sample_frame()
    profile = DatasetProfiler().profile(frame)
    by_name = {column.name: column for column in profile.columns}

    assert by_name["temperature"].null_ratio == pytest.approx(0.25)
    assert by_name["lot_id"].null_ratio == pytest.approx(0.0)


def test_unique_count_is_accurate() -> None:
    frame = _sample_frame()
    profile = DatasetProfiler().profile(frame)
    by_name = {column.name: column for column in profile.columns}

    assert by_name["lot_id"].unique_count == 3
    assert by_name["constant_flag"].unique_count == 1
    assert by_name["temperature"].unique_count == 4


def test_cardinality_ratio_is_accurate() -> None:
    frame = _sample_frame()
    profile = DatasetProfiler().profile(frame)
    by_name = {column.name: column for column in profile.columns}

    assert by_name["lot_id"].cardinality_ratio == pytest.approx(0.75)
    assert by_name["constant_flag"].cardinality_ratio == pytest.approx(0.25)


def test_constant_column_is_detected() -> None:
    frame = _sample_frame()
    profile = DatasetProfiler().profile(frame)
    by_name = {column.name: column for column in profile.columns}

    assert by_name["constant_flag"].is_constant is True


def test_non_constant_column_is_detected() -> None:
    frame = _sample_frame()
    profile = DatasetProfiler().profile(frame)
    by_name = {column.name: column for column in profile.columns}

    assert by_name["lot_id"].is_constant is False
    assert by_name["temperature"].is_constant is False


def test_numeric_column_statistics_are_accurate() -> None:
    frame = pl.DataFrame({"temperature": [100.0, 110.0, 90.0]})
    profile = DatasetProfiler().profile(frame)
    column = profile.columns[0]

    assert column.minimum == pytest.approx(90.0)
    assert column.maximum == pytest.approx(110.0)
    assert column.mean == pytest.approx(100.0)


def test_string_column_numeric_statistics_are_none() -> None:
    frame = pl.DataFrame({"lot_id": ["A", "B", "C"]})
    profile = DatasetProfiler().profile(frame)
    column = profile.columns[0]

    assert column.minimum is None
    assert column.maximum is None
    assert column.mean is None


def test_boolean_column_numeric_statistics_are_none() -> None:
    frame = pl.DataFrame({"status": [True, False, True]})
    profile = DatasetProfiler().profile(frame)
    column = profile.columns[0]

    assert column.minimum is None
    assert column.maximum is None
    assert column.mean is None


def test_sample_values_preserve_original_order() -> None:
    frame = pl.DataFrame({"lot_id": ["first", "second", "third", "fourth"]})
    profile = DatasetProfiler(sample_values_per_column=3).profile(frame)

    assert profile.columns[0].sample_values == ["first", "second", "third"]


def test_sample_values_respect_maximum_count() -> None:
    frame = pl.DataFrame({"value": [1, 2, 3, 4, 5, 6]})
    profile = DatasetProfiler(sample_values_per_column=2).profile(frame)

    assert len(profile.columns[0].sample_values) == 2
    assert profile.columns[0].sample_values == [1, 2]


def test_preview_records_preserve_original_row_order() -> None:
    frame = pl.DataFrame(
        {
            "seq": [10, 20, 30],
            "label": ["first", "second", "third"],
        }
    )
    profile = DatasetProfiler(preview_rows=2).profile(frame)

    assert profile.preview_records == [
        {"seq": 10, "label": "first"},
        {"seq": 20, "label": "second"},
    ]


def test_preview_records_respect_maximum_count() -> None:
    frame = pl.DataFrame({"seq": list(range(10))})
    profile = DatasetProfiler(preview_rows=3).profile(frame)

    assert len(profile.preview_records) == 3


def test_empty_dataframe_is_handled_safely() -> None:
    frame = pl.DataFrame(schema={"a": pl.Int64, "b": pl.String})
    profile = DatasetProfiler().profile(frame)

    assert profile.row_count == 0
    assert profile.column_count == 2
    assert len(profile.columns) == 2
    assert profile.preview_records == []


def test_empty_dataframe_ratios_are_zero() -> None:
    frame = pl.DataFrame(schema={"a": pl.Int64})
    profile = DatasetProfiler().profile(frame)
    column = profile.columns[0]

    assert column.null_ratio == pytest.approx(0.0)
    assert column.cardinality_ratio == pytest.approx(0.0)
    assert column.is_constant is False


def test_single_row_column_is_constant() -> None:
    frame = pl.DataFrame({"value": [42]})
    profile = DatasetProfiler().profile(frame)

    assert profile.columns[0].is_constant is True
    assert profile.columns[0].unique_count == 1


def test_all_null_non_empty_column_is_constant() -> None:
    frame = pl.DataFrame({"value": [None, None, None]}, schema={"value": pl.Float64})
    profile = DatasetProfiler().profile(frame)
    column = profile.columns[0]

    assert column.null_count == 3
    assert column.unique_count == 1
    assert column.is_constant is True


def test_all_null_numeric_column_statistics_are_none() -> None:
    frame = pl.DataFrame({"value": [None, None]}, schema={"value": pl.Float64})
    profile = DatasetProfiler().profile(frame)
    column = profile.columns[0]

    assert column.minimum is None
    assert column.maximum is None
    assert column.mean is None


def test_original_row_id_column_is_included() -> None:
    frame = _sample_frame()
    profile = DatasetProfiler().profile(frame)
    names = [column.name for column in profile.columns]

    assert ORIGINAL_ROW_ID_COLUMN in names
    by_name = {column.name: column for column in profile.columns}
    assert by_name[ORIGINAL_ROW_ID_COLUMN].sample_values == [0, 1, 2, 3, 4][:4]


def test_profile_preserves_frame_shape() -> None:
    frame = _sample_frame()
    shape_before = frame.shape
    DatasetProfiler().profile(frame)

    assert frame.shape == shape_before


def test_profile_preserves_frame_columns() -> None:
    frame = _sample_frame()
    columns_before = list(frame.columns)
    DatasetProfiler().profile(frame)

    assert list(frame.columns) == columns_before


def test_profile_preserves_frame_data() -> None:
    frame = _sample_frame()
    data_before = frame.to_dicts()
    DatasetProfiler().profile(frame)

    assert frame.to_dicts() == data_before


def test_profiler_has_no_shared_internal_state_across_frames() -> None:
    profiler = DatasetProfiler(preview_rows=2, sample_values_per_column=2)
    first = profiler.profile(pl.DataFrame({"a": [1, 2, 3]}))
    second = profiler.profile(pl.DataFrame({"b": ["x", "y"]}))

    assert first.column_count == 1
    assert first.columns[0].name == "a"
    assert second.column_count == 1
    assert second.columns[0].name == "b"
    assert first.columns[0].sample_values == [1, 2]
    assert second.columns[0].sample_values == ["x", "y"]


def test_column_profile_mutable_defaults_are_not_shared() -> None:
    first = ColumnProfile(
        name="a",
        dtype="Int64",
        null_count=0,
        null_ratio=0.0,
        unique_count=1,
        cardinality_ratio=1.0,
        is_constant=True,
    )
    second = ColumnProfile(
        name="b",
        dtype="Int64",
        null_count=0,
        null_ratio=0.0,
        unique_count=1,
        cardinality_ratio=1.0,
        is_constant=True,
    )
    first.sample_values.append(1)

    assert second.sample_values == []


def test_dataset_profile_mutable_defaults_are_not_shared() -> None:
    first = DatasetProfile(row_count=0, column_count=0)
    second = DatasetProfile(row_count=0, column_count=0)
    first.columns.append(
        ColumnProfile(
            name="a",
            dtype="Int64",
            null_count=0,
            null_ratio=0.0,
            unique_count=0,
            cardinality_ratio=0.0,
            is_constant=False,
        )
    )
    first.preview_records.append({"a": 1})

    assert second.columns == []
    assert second.preview_records == []


def test_dataset_profile_model_dump_validate_round_trip() -> None:
    profile = DatasetProfiler().profile(_sample_frame())
    restored = DatasetProfile.model_validate(profile.model_dump())

    assert restored == profile
