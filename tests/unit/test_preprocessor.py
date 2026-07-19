"""Unit tests for DatasetPreprocessor (Step 2G)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
from datetime import UTC
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from process_intelligence.core.exceptions import (
    DataValidationError,
    InsufficientDataError,
    ProcessIntelligenceError,
)
from process_intelligence.data import (
    MISSING_CATEGORY_TOKEN,
    ORIGINAL_ROW_ID_COLUMN,
    DatasetPreprocessor,
    PreprocessingResult,
    PreprocessorConfig,
)
from process_intelligence.data.preprocessor import (
    MISSING_CATEGORY_TOKEN as DIRECT_MISSING_TOKEN,
)
from process_intelligence.data.preprocessor import (
    DatasetPreprocessor as DirectPreprocessor,
)
from process_intelligence.data.preprocessor import (
    PreprocessingResult as DirectResult,
)
from process_intelligence.data.preprocessor import (
    PreprocessorConfig as DirectConfig,
)


def _train_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "temp": [1.0, None, 3.0, 5.0],
            "pressure": [10.0, 20.0, 30.0, 40.0],
            "status": ["ok", None, "ok", "warn"],
            "batch": ["A", "B", "A", "A"],
            "note": ["x", "y", "z", "w"],
            ORIGINAL_ROW_ID_COLUMN: [0, 1, 2, 3],
        }
    )


def test_missing_category_token_value() -> None:
    assert MISSING_CATEGORY_TOKEN == "__MISSING__"
    assert DIRECT_MISSING_TOKEN == "__MISSING__"


def test_preprocessor_config_default_construction() -> None:
    config = PreprocessorConfig()
    assert config.numeric_columns == []
    assert config.categorical_columns == []
    assert config.numeric_imputation == "median"
    assert config.categorical_imputation == "constant"
    assert config.categorical_fill_value == MISSING_CATEGORY_TOKEN
    assert config.scaling == "none"
    assert isinstance(config, DirectConfig)


def test_mutable_default_lists_are_not_shared() -> None:
    first = PreprocessorConfig()
    second = PreprocessorConfig()
    first.numeric_columns.append("temp")
    first.categorical_columns.append("status")
    assert second.numeric_columns == []
    assert second.categorical_columns == []


def test_empty_column_name_rejected() -> None:
    with pytest.raises(ValidationError):
        PreprocessorConfig(numeric_columns=[""])


def test_whitespace_only_column_name_rejected() -> None:
    with pytest.raises(ValidationError):
        PreprocessorConfig(categorical_columns=["   "])


def test_duplicate_numeric_columns_rejected() -> None:
    with pytest.raises(ValidationError):
        PreprocessorConfig(numeric_columns=["temp", "temp"])


def test_duplicate_categorical_columns_rejected() -> None:
    with pytest.raises(ValidationError):
        PreprocessorConfig(categorical_columns=["status", "status"])


def test_numeric_categorical_overlap_rejected() -> None:
    with pytest.raises(ValidationError):
        PreprocessorConfig(
            numeric_columns=["temp"],
            categorical_columns=["temp"],
        )


def test_original_row_id_as_numeric_rejected() -> None:
    with pytest.raises(ValidationError):
        PreprocessorConfig(numeric_columns=[ORIGINAL_ROW_ID_COLUMN])


def test_original_row_id_as_categorical_rejected() -> None:
    with pytest.raises(ValidationError):
        PreprocessorConfig(categorical_columns=[ORIGINAL_ROW_ID_COLUMN])


def test_constant_strategy_rejects_blank_fill_value() -> None:
    with pytest.raises(ValidationError):
        PreprocessorConfig(
            categorical_columns=["status"],
            categorical_imputation="constant",
            categorical_fill_value="   ",
        )


def test_dataset_preprocessor_construction() -> None:
    preprocessor = DatasetPreprocessor(PreprocessorConfig())
    assert isinstance(preprocessor, DatasetPreprocessor)
    assert isinstance(preprocessor, DirectPreprocessor)


def test_non_config_raises_type_error() -> None:
    with pytest.raises(TypeError, match="PreprocessorConfig"):
        DatasetPreprocessor({"numeric_columns": []})  # type: ignore[arg-type]


def test_is_fitted_false_after_construction() -> None:
    assert DatasetPreprocessor(PreprocessorConfig()).is_fitted is False


def test_fit_non_polars_raises_type_error() -> None:
    preprocessor = DatasetPreprocessor(PreprocessorConfig())
    with pytest.raises(TypeError, match="polars.DataFrame"):
        preprocessor.fit({"temp": [1.0]})  # type: ignore[arg-type]


def test_transform_non_polars_raises_type_error() -> None:
    preprocessor = DatasetPreprocessor(PreprocessorConfig())
    preprocessor.fit(pl.DataFrame())
    with pytest.raises(TypeError, match="polars.DataFrame"):
        preprocessor.transform({"temp": [1.0]})  # type: ignore[arg-type]


def test_missing_configured_column_rejected() -> None:
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(numeric_columns=["missing_col"])
    )
    with pytest.raises(DataValidationError, match="missing_col"):
        preprocessor.fit(pl.DataFrame({"temp": [1.0]}))


def test_invalid_numeric_dtype_rejected() -> None:
    frame = pl.DataFrame({"temp": ["a", "b"]})
    preprocessor = DatasetPreprocessor(PreprocessorConfig(numeric_columns=["temp"]))
    with pytest.raises(DataValidationError, match="temp"):
        preprocessor.fit(frame)


def test_boolean_numeric_column_rejected() -> None:
    frame = pl.DataFrame({"flag": [True, False]})
    preprocessor = DatasetPreprocessor(PreprocessorConfig(numeric_columns=["flag"]))
    with pytest.raises(DataValidationError, match="flag"):
        preprocessor.fit(frame)


def test_invalid_categorical_dtype_rejected() -> None:
    frame = pl.DataFrame({"status": [1, 2]})
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(categorical_columns=["status"])
    )
    with pytest.raises(DataValidationError, match="status"):
        preprocessor.fit(frame)


def test_nan_in_numeric_column_rejected() -> None:
    frame = pl.DataFrame({"temp": [1.0, float("nan")]})
    preprocessor = DatasetPreprocessor(PreprocessorConfig(numeric_columns=["temp"]))
    with pytest.raises(DataValidationError, match="(?i)nan"):
        preprocessor.fit(frame)


def test_positive_infinity_rejected() -> None:
    frame = pl.DataFrame({"temp": [1.0, float("inf")]})
    preprocessor = DatasetPreprocessor(PreprocessorConfig(numeric_columns=["temp"]))
    with pytest.raises(DataValidationError, match="(?i)infinity"):
        preprocessor.fit(frame)


def test_negative_infinity_rejected() -> None:
    frame = pl.DataFrame({"temp": [1.0, float("-inf")]})
    preprocessor = DatasetPreprocessor(PreprocessorConfig(numeric_columns=["temp"]))
    with pytest.raises(DataValidationError, match="(?i)infinity"):
        preprocessor.fit(frame)


def test_empty_frame_with_configured_columns_raises() -> None:
    frame = pl.DataFrame(schema={"temp": pl.Float64, "status": pl.String})
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            categorical_columns=["status"],
        )
    )
    with pytest.raises(InsufficientDataError):
        preprocessor.fit(frame)


def test_noop_config_fits_empty_frame() -> None:
    preprocessor = DatasetPreprocessor(PreprocessorConfig())
    result = preprocessor.fit(pl.DataFrame())
    assert result is preprocessor
    assert preprocessor.is_fitted is True


def test_fit_returns_self() -> None:
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(numeric_columns=["temp"], numeric_imputation="none")
    )
    frame = pl.DataFrame({"temp": [1.0, 2.0]})
    assert preprocessor.fit(frame) is preprocessor


def test_fit_sets_is_fitted_true() -> None:
    preprocessor = DatasetPreprocessor(PreprocessorConfig())
    preprocessor.fit(pl.DataFrame({"a": [1]}))
    assert preprocessor.is_fitted is True


def test_transform_before_fit_raises() -> None:
    preprocessor = DatasetPreprocessor(PreprocessorConfig())
    with pytest.raises(ProcessIntelligenceError):
        preprocessor.transform(pl.DataFrame())


def test_median_imputation_statistics() -> None:
    frame = pl.DataFrame({"temp": [1.0, None, 3.0, 5.0]})
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="median",
            categorical_imputation="none",
        )
    )
    preprocessor.fit(frame)
    stats = preprocessor.get_fitted_statistics()
    assert stats["numeric_fill_values"]["temp"] == pytest.approx(3.0)


def test_mean_imputation_statistics() -> None:
    frame = pl.DataFrame({"temp": [1.0, None, 3.0]})
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="mean",
            categorical_imputation="none",
        )
    )
    preprocessor.fit(frame)
    stats = preprocessor.get_fitted_statistics()
    assert stats["numeric_fill_values"]["temp"] == pytest.approx(2.0)


def test_numeric_imputation_none_keeps_nulls() -> None:
    frame = pl.DataFrame({"temp": [1.0, None, 3.0]})
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="none",
            categorical_imputation="none",
            scaling="none",
        )
    )
    result = preprocessor.fit_transform(frame)
    assert result.frame["temp"].to_list() == [1.0, None, 3.0]


def test_all_null_median_fit_rejected() -> None:
    frame = pl.DataFrame({"temp": [None, None]}, schema={"temp": pl.Float64})
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="median",
            categorical_imputation="none",
        )
    )
    with pytest.raises(DataValidationError, match="temp"):
        preprocessor.fit(frame)


def test_all_null_mean_fit_rejected() -> None:
    frame = pl.DataFrame({"temp": [None, None]}, schema={"temp": pl.Float64})
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="mean",
            categorical_imputation="none",
        )
    )
    with pytest.raises(DataValidationError, match="temp"):
        preprocessor.fit(frame)


def test_categorical_constant_imputation() -> None:
    frame = pl.DataFrame({"status": ["ok", None, "warn"]})
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            categorical_columns=["status"],
            categorical_imputation="constant",
            categorical_fill_value=MISSING_CATEGORY_TOKEN,
            numeric_imputation="none",
        )
    )
    result = preprocessor.fit_transform(frame)
    assert result.frame["status"].to_list() == ["ok", MISSING_CATEGORY_TOKEN, "warn"]


def test_categorical_most_frequent_imputation() -> None:
    frame = pl.DataFrame({"status": ["ok", "ok", None, "warn"]})
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            categorical_columns=["status"],
            categorical_imputation="most_frequent",
            numeric_imputation="none",
        )
    )
    result = preprocessor.fit_transform(frame)
    assert result.frame["status"].to_list() == ["ok", "ok", "ok", "warn"]


def test_most_frequent_tie_uses_first_occurrence() -> None:
    frame = pl.DataFrame({"status": ["b", "a", "a", "b", None]})
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            categorical_columns=["status"],
            categorical_imputation="most_frequent",
            numeric_imputation="none",
        )
    )
    preprocessor.fit(frame)
    stats = preprocessor.get_fitted_statistics()
    assert stats["categorical_fill_values"]["status"] == "b"
    result = preprocessor.transform(frame)
    assert result.frame["status"].to_list() == ["b", "a", "a", "b", "b"]


def test_categorical_imputation_none_keeps_nulls() -> None:
    frame = pl.DataFrame({"status": ["ok", None]})
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            categorical_columns=["status"],
            categorical_imputation="none",
            numeric_imputation="none",
        )
    )
    result = preprocessor.fit_transform(frame)
    assert result.frame["status"].to_list() == ["ok", None]


def test_categorical_all_null_most_frequent_rejected() -> None:
    frame = pl.DataFrame({"status": [None, None]}, schema={"status": pl.String})
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            categorical_columns=["status"],
            categorical_imputation="most_frequent",
            numeric_imputation="none",
        )
    )
    with pytest.raises(DataValidationError, match="status"):
        preprocessor.fit(frame)


def test_standard_scaling_center_and_scale() -> None:
    frame = pl.DataFrame({"temp": [0.0, 10.0]})
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="none",
            categorical_imputation="none",
            scaling="standard",
        )
    )
    preprocessor.fit(frame)
    stats = preprocessor.get_fitted_statistics()
    assert stats["scaling_centers"]["temp"] == pytest.approx(5.0)
    assert stats["scaling_scales"]["temp"] == pytest.approx(5.0)


def test_standard_scaling_result() -> None:
    frame = pl.DataFrame({"temp": [0.0, 10.0]})
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="none",
            categorical_imputation="none",
            scaling="standard",
        )
    )
    result = preprocessor.fit_transform(frame)
    assert result.frame["temp"].to_list() == pytest.approx([-1.0, 1.0])


def test_standard_scaling_zero_variance_scale_is_one() -> None:
    frame = pl.DataFrame({"temp": [4.0, 4.0, 4.0]})
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="none",
            categorical_imputation="none",
            scaling="standard",
        )
    )
    preprocessor.fit(frame)
    stats = preprocessor.get_fitted_statistics()
    assert stats["scaling_scales"]["temp"] == pytest.approx(1.0)
    result = preprocessor.transform(frame)
    assert result.frame["temp"].to_list() == pytest.approx([0.0, 0.0, 0.0])


def test_robust_scaling_center_and_iqr() -> None:
    frame = pl.DataFrame({"temp": [1.0, 2.0, 3.0, 4.0]})
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="none",
            categorical_imputation="none",
            scaling="robust",
        )
    )
    preprocessor.fit(frame)
    stats = preprocessor.get_fitted_statistics()
    assert stats["scaling_centers"]["temp"] == pytest.approx(2.5)
    q1 = float(frame.get_column("temp").quantile(0.25))  # type: ignore[arg-type]
    q3 = float(frame.get_column("temp").quantile(0.75))  # type: ignore[arg-type]
    assert stats["scaling_scales"]["temp"] == pytest.approx(q3 - q1)


def test_robust_scaling_result() -> None:
    frame = pl.DataFrame({"temp": [1.0, 2.0, 3.0, 4.0]})
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="none",
            categorical_imputation="none",
            scaling="robust",
        )
    )
    result = preprocessor.fit_transform(frame)
    stats = preprocessor.get_fitted_statistics()
    center = stats["scaling_centers"]["temp"]
    scale = stats["scaling_scales"]["temp"]
    expected = [(value - center) / scale for value in [1.0, 2.0, 3.0, 4.0]]
    assert result.frame["temp"].to_list() == pytest.approx(expected)


def test_robust_scaling_zero_iqr_scale_is_one() -> None:
    frame = pl.DataFrame({"temp": [7.0, 7.0, 7.0]})
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="none",
            categorical_imputation="none",
            scaling="robust",
        )
    )
    preprocessor.fit(frame)
    stats = preprocessor.get_fitted_statistics()
    assert stats["scaling_scales"]["temp"] == pytest.approx(1.0)


def test_scaling_statistics_use_imputed_training_values() -> None:
    frame = pl.DataFrame({"temp": [1.0, None, 3.0]})
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="mean",
            categorical_imputation="none",
            scaling="standard",
        )
    )
    preprocessor.fit(frame)
    stats = preprocessor.get_fitted_statistics()
    # imputed training values: [1.0, 2.0, 3.0]
    assert stats["numeric_fill_values"]["temp"] == pytest.approx(2.0)
    assert stats["scaling_centers"]["temp"] == pytest.approx(2.0)
    assert stats["scaling_scales"]["temp"] == pytest.approx(
        pl.Series([1.0, 2.0, 3.0]).std(ddof=0)
    )


def test_transform_uses_train_fit_statistics() -> None:
    train = pl.DataFrame({"temp": [0.0, 10.0]})
    test = pl.DataFrame({"temp": [100.0, 110.0]})
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="none",
            categorical_imputation="none",
            scaling="standard",
        )
    )
    preprocessor.fit(train)
    result = preprocessor.transform(test)
    assert result.frame["temp"].to_list() == pytest.approx([19.0, 21.0])


def test_transform_does_not_recompute_from_transform_data() -> None:
    train = pl.DataFrame({"temp": [0.0, 10.0]})
    test = pl.DataFrame({"temp": [100.0, 110.0]})
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="none",
            categorical_imputation="none",
            scaling="standard",
        )
    )
    preprocessor.fit(train)
    before = preprocessor.get_fitted_statistics()
    preprocessor.transform(test)
    after = preprocessor.get_fitted_statistics()
    assert after["scaling_centers"]["temp"] == pytest.approx(before["scaling_centers"]["temp"])
    assert after["scaling_scales"]["temp"] == pytest.approx(before["scaling_scales"]["temp"])
    assert after["scaling_centers"]["temp"] == pytest.approx(5.0)


def test_transform_does_not_mutate_fitted_statistics() -> None:
    train = pl.DataFrame({"temp": [0.0, 10.0], "status": ["a", None]})
    test = pl.DataFrame({"temp": [100.0, 110.0], "status": [None, "b"]})
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            categorical_columns=["status"],
            numeric_imputation="mean",
            categorical_imputation="constant",
            scaling="standard",
        )
    )
    preprocessor.fit(train)
    before = preprocessor.get_fitted_statistics()
    preprocessor.transform(test)
    after = preprocessor.get_fitted_statistics()
    assert after == before


def test_successful_refit_replaces_statistics() -> None:
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="median",
            categorical_imputation="none",
        )
    )
    preprocessor.fit(pl.DataFrame({"temp": [1.0, 3.0]}))
    first = preprocessor.get_fitted_statistics()["numeric_fill_values"]["temp"]
    preprocessor.fit(pl.DataFrame({"temp": [10.0, 30.0]}))
    second = preprocessor.get_fitted_statistics()["numeric_fill_values"]["temp"]
    assert first == pytest.approx(2.0)
    assert second == pytest.approx(20.0)


def test_failed_refit_keeps_fitted_flag() -> None:
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="median",
            categorical_imputation="none",
        )
    )
    preprocessor.fit(pl.DataFrame({"temp": [1.0, 3.0]}))
    with pytest.raises(DataValidationError):
        preprocessor.fit(pl.DataFrame({"other": [1.0]}))
    assert preprocessor.is_fitted is True


def test_failed_refit_keeps_previous_statistics() -> None:
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="median",
            categorical_imputation="none",
        )
    )
    preprocessor.fit(pl.DataFrame({"temp": [1.0, 3.0]}))
    before = preprocessor.get_fitted_statistics()
    with pytest.raises(DataValidationError):
        preprocessor.fit(
            pl.DataFrame({"temp": [None, None]}, schema={"temp": pl.Float64})
        )
    assert preprocessor.get_fitted_statistics() == before


def test_fit_transform_matches_fit_then_transform() -> None:
    frame = _train_frame()
    config = PreprocessorConfig(
        numeric_columns=["temp", "pressure"],
        categorical_columns=["status"],
        numeric_imputation="median",
        categorical_imputation="most_frequent",
        scaling="standard",
    )
    combined = DatasetPreprocessor(config).fit_transform(frame)
    separate = DatasetPreprocessor(config)
    separate.fit(frame)
    separate_result = separate.transform(frame)

    assert_frame_equal(combined.frame, separate_result.frame)
    assert [event.step_name for event in combined.events] == [
        event.step_name for event in separate_result.events
    ]
    assert combined.events[0].parameters == separate_result.events[0].parameters
    assert combined.events[1].parameters == separate_result.events[1].parameters


def test_result_frame_is_polars_dataframe() -> None:
    result = DatasetPreprocessor(PreprocessorConfig()).fit_transform(pl.DataFrame())
    assert isinstance(result.frame, pl.DataFrame)


def test_result_row_count_matches_input() -> None:
    frame = _train_frame()
    result = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            categorical_columns=["status"],
        )
    ).fit_transform(frame)
    assert result.frame.height == frame.height


def test_result_row_order_preserved() -> None:
    frame = pl.DataFrame(
        {
            "temp": [3.0, 1.0, 2.0],
            "label": ["c", "a", "b"],
            ORIGINAL_ROW_ID_COLUMN: [10, 20, 30],
        }
    )
    result = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="none",
            categorical_imputation="none",
            scaling="standard",
        )
    ).fit_transform(frame)
    assert result.frame["label"].to_list() == ["c", "a", "b"]
    assert result.frame[ORIGINAL_ROW_ID_COLUMN].to_list() == [10, 20, 30]


def test_result_column_names_and_order_preserved() -> None:
    frame = _train_frame()
    result = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            categorical_columns=["status"],
        )
    ).fit_transform(frame)
    assert result.frame.columns == frame.columns


def test_unconfigured_column_values_and_dtype_preserved() -> None:
    frame = _train_frame()
    note_before = frame.get_column("note").clone()
    result = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            categorical_columns=["status"],
        )
    ).fit_transform(frame)
    assert result.frame.schema["note"] == frame.schema["note"]
    assert result.frame["note"].to_list() == note_before.to_list()


def test_original_row_id_preserved() -> None:
    frame = _train_frame()
    result = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            categorical_columns=["status"],
            scaling="standard",
        )
    ).fit_transform(frame)
    assert result.frame.schema[ORIGINAL_ROW_ID_COLUMN] == frame.schema[ORIGINAL_ROW_ID_COLUMN]
    assert result.frame[ORIGINAL_ROW_ID_COLUMN].to_list() == [0, 1, 2, 3]


def test_frame_without_original_row_id_supported() -> None:
    frame = pl.DataFrame({"temp": [1.0, None, 3.0], "status": ["a", None, "a"]})
    result = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            categorical_columns=["status"],
        )
    ).fit_transform(frame)
    assert ORIGINAL_ROW_ID_COLUMN not in result.frame.columns
    assert result.frame.height == 3


def test_fit_does_not_mutate_input_frame() -> None:
    frame = pl.DataFrame({"temp": [1.0, None, 3.0]})
    before = frame.clone()
    DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="median",
            categorical_imputation="none",
        )
    ).fit(frame)
    assert_frame_equal(frame, before)


def test_transform_does_not_mutate_input_frame() -> None:
    train = pl.DataFrame({"temp": [1.0, 3.0]})
    test = pl.DataFrame({"temp": [1.0, None, 3.0]})
    before = test.clone()
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="median",
            categorical_imputation="none",
        )
    )
    preprocessor.fit(train)
    preprocessor.transform(test)
    assert_frame_equal(test, before)


def test_fit_transform_does_not_mutate_input_frame() -> None:
    frame = pl.DataFrame({"temp": [1.0, None, 3.0]})
    before = frame.clone()
    DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="median",
            categorical_imputation="none",
        )
    ).fit_transform(frame)
    assert_frame_equal(frame, before)


def test_preprocessing_result_is_frozen_dataclass() -> None:
    result = DatasetPreprocessor(PreprocessorConfig()).fit_transform(pl.DataFrame())
    assert is_dataclass(result)
    assert isinstance(result, PreprocessingResult)
    assert isinstance(result, DirectResult)
    with pytest.raises(FrozenInstanceError):
        result.events = ()  # type: ignore[misc]


def test_preprocessing_result_uses_slots() -> None:
    assert PreprocessingResult.__slots__ == ("frame", "events")
    assert {item.name for item in fields(PreprocessingResult)} == {"frame", "events"}


def test_events_are_tuple() -> None:
    result = DatasetPreprocessor(PreprocessorConfig()).fit_transform(pl.DataFrame())
    assert isinstance(result.events, tuple)


def test_imputation_event_step_name() -> None:
    result = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            categorical_columns=["status"],
            scaling="none",
        )
    ).fit_transform(_train_frame())
    assert result.events[0].step_name == "impute_missing_values"


def test_scaling_event_step_name() -> None:
    result = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="none",
            categorical_imputation="none",
            scaling="standard",
        )
    ).fit_transform(pl.DataFrame({"temp": [0.0, 10.0]}))
    assert result.events[0].step_name == "scale_numeric_features"


def test_imputation_then_scaling_event_order() -> None:
    result = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            categorical_columns=["status"],
            scaling="standard",
        )
    ).fit_transform(_train_frame())
    assert [event.step_name for event in result.events] == [
        "impute_missing_values",
        "scale_numeric_features",
    ]


def test_noop_config_emits_preprocess_noop_event() -> None:
    result = DatasetPreprocessor(PreprocessorConfig()).fit_transform(
        pl.DataFrame({"temp": [1.0]})
    )
    assert len(result.events) == 1
    assert result.events[0].step_name == "preprocess_noop"
    assert result.events[0].parameters["reason"] == "No preprocessing operation configured."


def test_event_affected_columns_order() -> None:
    result = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["pressure", "temp"],
            categorical_columns=["batch", "status"],
            scaling="none",
        )
    ).fit_transform(_train_frame())
    assert result.events[0].affected_columns == [
        "pressure",
        "temp",
        "batch",
        "status",
    ]


def test_event_rows_before_after() -> None:
    frame = _train_frame()
    result = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            categorical_columns=["status"],
            scaling="standard",
        )
    ).fit_transform(frame)
    for event in result.events:
        assert event.rows_before == frame.height
        assert event.rows_after == frame.height


def test_event_parameters_include_fitted_on_training_data() -> None:
    result = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            categorical_columns=["status"],
            scaling="standard",
        )
    ).fit_transform(_train_frame())
    assert result.events[0].parameters["fitted_on_training_data"] is True
    assert result.events[1].parameters["fitted_on_training_data"] is True


def test_event_timestamp_is_timezone_aware_utc() -> None:
    result = DatasetPreprocessor(PreprocessorConfig()).fit_transform(pl.DataFrame())
    timestamp = result.events[0].timestamp
    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() == UTC.utcoffset(timestamp)


def test_zero_variance_warning_created() -> None:
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="none",
            categorical_imputation="none",
            scaling="standard",
        )
    )
    result = preprocessor.fit_transform(pl.DataFrame({"temp": [2.0, 2.0]}))
    assert any("temp" in warning for warning in result.events[0].warnings)
    assert any(
        "temp" in warning for warning in preprocessor.get_fitted_statistics()["warnings"]
    )


def test_zero_iqr_warning_created() -> None:
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="none",
            categorical_imputation="none",
            scaling="robust",
        )
    )
    result = preprocessor.fit_transform(pl.DataFrame({"temp": [5.0, 5.0, 5.0]}))
    assert any("temp" in warning for warning in result.events[0].warnings)
    assert any("temp" in warning for warning in preprocessor.get_fitted_statistics()["warnings"])


def test_normal_columns_have_no_warnings() -> None:
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="none",
            categorical_imputation="none",
            scaling="standard",
        )
    )
    result = preprocessor.fit_transform(pl.DataFrame({"temp": [0.0, 10.0]}))
    assert result.events[0].warnings == []
    assert preprocessor.get_fitted_statistics()["warnings"] == []


def test_get_fitted_statistics_before_fit_raises() -> None:
    with pytest.raises(ProcessIntelligenceError):
        DatasetPreprocessor(PreprocessorConfig()).get_fitted_statistics()


def test_get_fitted_statistics_contains_required_keys() -> None:
    preprocessor = DatasetPreprocessor(PreprocessorConfig())
    preprocessor.fit(pl.DataFrame())
    stats = preprocessor.get_fitted_statistics()
    assert set(stats) == {
        "numeric_fill_values",
        "categorical_fill_values",
        "scaling_centers",
        "scaling_scales",
        "warnings",
    }


def test_returned_statistics_mutation_does_not_affect_internal_state() -> None:
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            categorical_columns=["status"],
            numeric_imputation="median",
            categorical_imputation="constant",
            scaling="standard",
        )
    )
    preprocessor.fit(
        pl.DataFrame({"temp": [1.0, 3.0], "status": ["a", None]})
    )
    stats = preprocessor.get_fitted_statistics()
    stats["numeric_fill_values"]["temp"] = 999.0
    stats["categorical_fill_values"]["status"] = "changed"
    stats["scaling_centers"]["temp"] = 999.0
    stats["scaling_scales"]["temp"] = 999.0
    stats["warnings"].append("injected")
    fresh = preprocessor.get_fitted_statistics()
    assert fresh["numeric_fill_values"]["temp"] == pytest.approx(2.0)
    assert fresh["categorical_fill_values"]["status"] == MISSING_CATEGORY_TOKEN
    assert "injected" not in fresh["warnings"]


def test_input_config_list_mutation_does_not_affect_internal_config() -> None:
    numeric = ["temp"]
    categorical = ["status"]
    config = PreprocessorConfig(
        numeric_columns=numeric,
        categorical_columns=categorical,
        scaling="standard",
    )
    preprocessor = DatasetPreprocessor(config)
    numeric.append("pressure")
    categorical.append("batch")
    config.numeric_columns.append("pressure")
    frame = pl.DataFrame({"temp": [1.0, 2.0], "status": ["a", "b"]})
    result = preprocessor.fit_transform(frame)
    assert result.events[0].affected_columns == ["temp", "status"]
    assert result.events[1].affected_columns == ["temp"]


def test_preprocessor_instances_do_not_share_state() -> None:
    config = PreprocessorConfig(
        numeric_columns=["temp"],
        numeric_imputation="median",
        categorical_imputation="none",
    )
    first = DatasetPreprocessor(config)
    second = DatasetPreprocessor(config)
    first.fit(pl.DataFrame({"temp": [1.0, 3.0]}))
    assert first.is_fitted is True
    assert second.is_fitted is False
    second.fit(pl.DataFrame({"temp": [10.0, 30.0]}))
    assert first.get_fitted_statistics()["numeric_fill_values"]["temp"] == pytest.approx(2.0)
    assert second.get_fitted_statistics()["numeric_fill_values"]["temp"] == pytest.approx(20.0)


def test_repeated_transform_does_not_accumulate() -> None:
    train = pl.DataFrame({"temp": [0.0, 10.0]})
    test = pl.DataFrame({"temp": [100.0, 110.0]})
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="none",
            categorical_imputation="none",
            scaling="standard",
        )
    )
    preprocessor.fit(train)
    first = preprocessor.transform(test).frame
    second = preprocessor.transform(test).frame
    assert_frame_equal(first, second)
    assert first["temp"].to_list() == pytest.approx([19.0, 21.0])


def test_extra_unconfigured_column_preserved_in_transform() -> None:
    train = pl.DataFrame({"temp": [1.0, 3.0], "extra": ["u", "v"]})
    test = pl.DataFrame({"temp": [1.0, None], "extra": ["p", "q"]})
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="median",
            categorical_imputation="none",
        )
    )
    preprocessor.fit(train)
    result = preprocessor.transform(test)
    assert result.frame["extra"].to_list() == ["p", "q"]
    assert result.frame.schema["extra"] == test.schema["extra"]


def test_missing_fitted_column_in_transform_raises() -> None:
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="median",
            categorical_imputation="none",
        )
    )
    preprocessor.fit(pl.DataFrame({"temp": [1.0, 3.0]}))
    with pytest.raises(DataValidationError, match="temp"):
        preprocessor.transform(pl.DataFrame({"other": [1.0]}))


def test_event_parameters_are_independent_copies() -> None:
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            numeric_imputation="median",
            categorical_imputation="none",
            scaling="standard",
        )
    )
    result = preprocessor.fit_transform(pl.DataFrame({"temp": [1.0, 3.0]}))
    params: dict[str, Any] = result.events[0].parameters
    params["numeric_fill_values"]["temp"] = 999.0
    fresh = preprocessor.transform(pl.DataFrame({"temp": [1.0, 3.0]}))
    assert fresh.events[0].parameters["numeric_fill_values"]["temp"] == pytest.approx(2.0)
