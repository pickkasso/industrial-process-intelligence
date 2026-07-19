"""Unit tests for leakage-safe dataset splitting (Step 5A)."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, fields, is_dataclass
from datetime import date, datetime
from math import ceil

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from process_intelligence.core.exceptions import (
    DataLeakageError,
    DataValidationError,
    InsufficientDataError,
)
from process_intelligence.data.loader import ORIGINAL_ROW_ID_COLUMN
from process_intelligence.evaluation import (
    DatasetSplit,
    DatasetSplitter,
    SplitConfig,
    SplitStrategy,
    SplitSummary,
)
from process_intelligence.evaluation.splitting import (
    DatasetSplit as DirectDatasetSplit,
)
from process_intelligence.evaluation.splitting import (
    DatasetSplitter as DirectDatasetSplitter,
)
from process_intelligence.evaluation.splitting import SplitConfig as DirectSplitConfig
from process_intelligence.evaluation.splitting import (
    SplitStrategy as DirectSplitStrategy,
)
from process_intelligence.evaluation.splitting import SplitSummary as DirectSplitSummary


def _frame_with_groups() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "batch": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"],
            "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ORIGINAL_ROW_ID_COLUMN: list(range(10)),
        }
    )


def _frame_with_dates() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": [
                date(2020, 1, 5),
                date(2020, 1, 1),
                date(2020, 1, 3),
                date(2020, 1, 2),
                date(2020, 1, 4),
                date(2020, 1, 6),
                date(2020, 1, 7),
                date(2020, 1, 8),
                date(2020, 1, 9),
                date(2020, 1, 10),
            ],
            "value": list(range(10)),
            ORIGINAL_ROW_ID_COLUMN: list(range(10)),
        }
    )


def _assert_partitions_cover(frame: pl.DataFrame, split: DatasetSplit) -> None:
    input_ids = set(frame.get_column(ORIGINAL_ROW_ID_COLUMN).to_list())
    train_ids = set(split.train.get_column(ORIGINAL_ROW_ID_COLUMN).to_list())
    validation_ids = set(split.validation.get_column(ORIGINAL_ROW_ID_COLUMN).to_list())
    test_ids = set(split.test.get_column(ORIGINAL_ROW_ID_COLUMN).to_list())
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert train_ids | validation_ids | test_ids == input_ids
    assert (
        split.train.height + split.validation.height + split.test.height == frame.height
    )


# --- SplitStrategy ---


def test_split_strategy_group_value() -> None:
    assert SplitStrategy.GROUP == "GROUP"
    assert SplitStrategy.GROUP.value == "GROUP"
    assert SplitStrategy.GROUP is DirectSplitStrategy.GROUP


def test_split_strategy_time_value() -> None:
    assert SplitStrategy.TIME == "TIME"
    assert SplitStrategy.TIME.value == "TIME"


def test_split_strategy_random_value() -> None:
    assert SplitStrategy.RANDOM == "RANDOM"
    assert SplitStrategy.RANDOM.value == "RANDOM"


def test_split_strategy_from_string() -> None:
    assert SplitStrategy("GROUP") is SplitStrategy.GROUP
    assert SplitStrategy("TIME") is SplitStrategy.TIME
    assert SplitStrategy("RANDOM") is SplitStrategy.RANDOM


def test_split_strategy_invalid_string() -> None:
    with pytest.raises(ValueError):
        SplitStrategy("STRATIFIED")


# --- SplitConfig ---


def test_split_config_group_ok() -> None:
    config = SplitConfig(strategy=SplitStrategy.GROUP, group_column="batch")
    assert config.strategy is SplitStrategy.GROUP
    assert config.group_column == "batch"
    assert config.time_column is None
    assert isinstance(config, DirectSplitConfig)


def test_split_config_time_ok() -> None:
    config = SplitConfig(strategy=SplitStrategy.TIME, time_column="ts")
    assert config.strategy is SplitStrategy.TIME
    assert config.time_column == "ts"
    assert config.group_column is None


def test_split_config_random_ok() -> None:
    config = SplitConfig(strategy=SplitStrategy.RANDOM, allow_random_split=True)
    assert config.strategy is SplitStrategy.RANDOM
    assert config.allow_random_split is True
    assert config.group_column is None
    assert config.time_column is None


def test_split_config_rejects_test_size_zero() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(strategy=SplitStrategy.RANDOM, test_size=0.0)


def test_split_config_rejects_test_size_one_or_more() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(strategy=SplitStrategy.RANDOM, test_size=1.0)
    with pytest.raises(ValidationError):
        SplitConfig(strategy=SplitStrategy.RANDOM, test_size=1.5)


def test_split_config_rejects_negative_validation_size() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(strategy=SplitStrategy.RANDOM, validation_size=-0.1)


def test_split_config_rejects_validation_size_one_or_more() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(strategy=SplitStrategy.RANDOM, validation_size=1.0)


def test_split_config_rejects_size_sum_at_least_one() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(
            strategy=SplitStrategy.RANDOM,
            test_size=0.6,
            validation_size=0.4,
        )


def test_split_config_rejects_bool_sizes() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(strategy=SplitStrategy.RANDOM, test_size=True)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        SplitConfig(
            strategy=SplitStrategy.RANDOM,
            validation_size=False,  # type: ignore[arg-type]
        )


def test_split_config_rejects_nan_sizes() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(strategy=SplitStrategy.RANDOM, test_size=math.nan)
    with pytest.raises(ValidationError):
        SplitConfig(strategy=SplitStrategy.RANDOM, validation_size=math.nan)


def test_split_config_rejects_positive_infinity_sizes() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(strategy=SplitStrategy.RANDOM, test_size=math.inf)
    with pytest.raises(ValidationError):
        SplitConfig(strategy=SplitStrategy.RANDOM, validation_size=math.inf)


def test_split_config_rejects_negative_infinity_sizes() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(strategy=SplitStrategy.RANDOM, test_size=-math.inf)
    with pytest.raises(ValidationError):
        SplitConfig(strategy=SplitStrategy.RANDOM, validation_size=-math.inf)


def test_split_config_rejects_negative_random_state() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(strategy=SplitStrategy.RANDOM, random_state=-1)


def test_split_config_rejects_bool_random_state() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(
            strategy=SplitStrategy.RANDOM,
            random_state=True,  # type: ignore[arg-type]
        )


def test_split_config_rejects_minimum_rows_below_one() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(strategy=SplitStrategy.RANDOM, minimum_train_rows=0)
    with pytest.raises(ValidationError):
        SplitConfig(strategy=SplitStrategy.RANDOM, minimum_test_rows=0)
    with pytest.raises(ValidationError):
        SplitConfig(strategy=SplitStrategy.RANDOM, minimum_validation_rows=0)


def test_split_config_rejects_bool_minimum_rows() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(
            strategy=SplitStrategy.RANDOM,
            minimum_train_rows=True,  # type: ignore[arg-type]
        )


def test_split_config_rejects_non_bool_allow_random_split() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(
            strategy=SplitStrategy.RANDOM,
            allow_random_split=1,  # type: ignore[arg-type]
        )


def test_split_config_rejects_whitespace_group_column() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(strategy=SplitStrategy.GROUP, group_column="   ")


def test_split_config_rejects_whitespace_time_column() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(strategy=SplitStrategy.TIME, time_column="   ")


def test_split_config_group_requires_group_column() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(strategy=SplitStrategy.GROUP)


def test_split_config_group_rejects_time_column() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(
            strategy=SplitStrategy.GROUP,
            group_column="batch",
            time_column="ts",
        )


def test_split_config_time_requires_time_column() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(strategy=SplitStrategy.TIME)


def test_split_config_time_rejects_group_column() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(
            strategy=SplitStrategy.TIME,
            time_column="ts",
            group_column="batch",
        )


def test_split_config_random_rejects_group_column() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(strategy=SplitStrategy.RANDOM, group_column="batch")


def test_split_config_random_rejects_time_column() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(strategy=SplitStrategy.RANDOM, time_column="ts")


# --- Common input validation ---


def test_dataset_splitter_construction() -> None:
    splitter = DatasetSplitter()
    assert isinstance(splitter, DirectDatasetSplitter)


def test_non_polars_frame_type_error() -> None:
    splitter = DatasetSplitter()
    config = SplitConfig(strategy=SplitStrategy.RANDOM, allow_random_split=True)
    with pytest.raises(TypeError):
        splitter.split([1, 2, 3], config)  # type: ignore[arg-type]


def test_non_split_config_type_error() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    with pytest.raises(TypeError):
        splitter.split(frame, {"strategy": "GROUP"})  # type: ignore[arg-type]


def test_missing_original_row_id_rejected() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame({"batch": ["A", "B"], "value": [1, 2]})
    config = SplitConfig(strategy=SplitStrategy.GROUP, group_column="batch")
    with pytest.raises(DataValidationError):
        splitter.split(frame, config)


def test_null_original_row_id_rejected() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "batch": ["A", "B"],
            "value": [1, 2],
            ORIGINAL_ROW_ID_COLUMN: pl.Series([0, None], dtype=pl.Int64),
        }
    )
    config = SplitConfig(strategy=SplitStrategy.GROUP, group_column="batch")
    with pytest.raises(DataValidationError):
        splitter.split(frame, config)


def test_string_original_row_id_rejected() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "batch": ["A", "B"],
            "value": [1, 2],
            ORIGINAL_ROW_ID_COLUMN: ["0", "1"],
        }
    )
    config = SplitConfig(strategy=SplitStrategy.GROUP, group_column="batch")
    with pytest.raises(DataValidationError):
        splitter.split(frame, config)


def test_boolean_original_row_id_rejected() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "batch": ["A", "B"],
            "value": [1, 2],
            ORIGINAL_ROW_ID_COLUMN: [True, False],
        }
    )
    config = SplitConfig(strategy=SplitStrategy.GROUP, group_column="batch")
    with pytest.raises(DataValidationError):
        splitter.split(frame, config)


def test_negative_original_row_id_rejected() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "batch": ["A", "B"],
            "value": [1, 2],
            ORIGINAL_ROW_ID_COLUMN: [-1, 0],
        }
    )
    config = SplitConfig(strategy=SplitStrategy.GROUP, group_column="batch")
    with pytest.raises(DataValidationError):
        splitter.split(frame, config)


def test_duplicate_original_row_id_rejected() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "batch": ["A", "B"],
            "value": [1, 2],
            ORIGINAL_ROW_ID_COLUMN: [1, 1],
        }
    )
    config = SplitConfig(strategy=SplitStrategy.GROUP, group_column="batch")
    with pytest.raises(DataValidationError):
        splitter.split(frame, config)


def test_non_contiguous_positive_original_ids_allowed() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "batch": ["A", "A", "B", "B", "C", "C"],
            "value": [1, 2, 3, 4, 5, 6],
            ORIGINAL_ROW_ID_COLUMN: [10, 20, 30, 40, 50, 60],
        }
    )
    config = SplitConfig(
        strategy=SplitStrategy.GROUP,
        group_column="batch",
        test_size=0.34,
        random_state=0,
    )
    result = splitter.split(frame, config)
    _assert_partitions_cover(frame, result)


def test_empty_frame_insufficient_data() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "batch": pl.Series([], dtype=pl.String),
            "value": pl.Series([], dtype=pl.Int64),
            ORIGINAL_ROW_ID_COLUMN: pl.Series([], dtype=pl.Int64),
        }
    )
    config = SplitConfig(strategy=SplitStrategy.GROUP, group_column="batch")
    with pytest.raises(InsufficientDataError):
        splitter.split(frame, config)


def test_train_minimum_insufficient() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "x": [1, 2, 3],
            ORIGINAL_ROW_ID_COLUMN: [0, 1, 2],
        }
    )
    config = SplitConfig(
        strategy=SplitStrategy.RANDOM,
        allow_random_split=True,
        test_size=0.34,
        minimum_train_rows=3,
        random_state=0,
    )
    with pytest.raises(InsufficientDataError):
        splitter.split(frame, config)


def test_test_minimum_insufficient() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            ORIGINAL_ROW_ID_COLUMN: list(range(5)),
        }
    )
    config = SplitConfig(
        strategy=SplitStrategy.RANDOM,
        allow_random_split=True,
        test_size=0.2,
        minimum_test_rows=2,
        random_state=0,
    )
    with pytest.raises(InsufficientDataError):
        splitter.split(frame, config)


def test_validation_minimum_insufficient() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "x": list(range(10)),
            ORIGINAL_ROW_ID_COLUMN: list(range(10)),
        }
    )
    config = SplitConfig(
        strategy=SplitStrategy.RANDOM,
        allow_random_split=True,
        test_size=0.2,
        validation_size=0.1,
        minimum_validation_rows=3,
        random_state=0,
    )
    with pytest.raises(InsufficientDataError):
        splitter.split(frame, config)


# --- GROUP split ---


def test_simple_group_split_success() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    config = SplitConfig(
        strategy=SplitStrategy.GROUP,
        group_column="batch",
        test_size=0.2,
        random_state=0,
    )
    result = splitter.split(frame, config)
    assert result.summary.strategy is SplitStrategy.GROUP
    _assert_partitions_cover(frame, result)
    assert result.test.height >= 1
    assert result.train.height >= 2


def test_group_column_missing_rejected() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    config = SplitConfig(strategy=SplitStrategy.GROUP, group_column="missing")
    with pytest.raises(DataValidationError):
        splitter.split(frame, config)


def test_group_null_rejected() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "batch": ["A", None, "B", "B"],
            "value": [1, 2, 3, 4],
            ORIGINAL_ROW_ID_COLUMN: [0, 1, 2, 3],
        }
    )
    config = SplitConfig(strategy=SplitStrategy.GROUP, group_column="batch")
    with pytest.raises(DataValidationError):
        splitter.split(frame, config)


def test_group_train_test_sets_disjoint() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    config = SplitConfig(
        strategy=SplitStrategy.GROUP,
        group_column="batch",
        test_size=0.2,
        random_state=1,
    )
    result = splitter.split(frame, config)
    assert set(result.summary.train_groups).isdisjoint(set(result.summary.test_groups))


def test_group_three_way_sets_disjoint() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    config = SplitConfig(
        strategy=SplitStrategy.GROUP,
        group_column="batch",
        test_size=0.2,
        validation_size=0.2,
        random_state=2,
    )
    result = splitter.split(frame, config)
    train_g = set(result.summary.train_groups)
    val_g = set(result.summary.validation_groups)
    test_g = set(result.summary.test_groups)
    assert train_g.isdisjoint(val_g)
    assert train_g.isdisjoint(test_g)
    assert val_g.isdisjoint(test_g)
    assert result.validation.height > 0


def test_same_group_not_in_two_partitions() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    config = SplitConfig(
        strategy=SplitStrategy.GROUP,
        group_column="batch",
        test_size=0.2,
        validation_size=0.2,
        random_state=3,
    )
    result = splitter.split(frame, config)
    for partition in (result.train, result.validation, result.test):
        if partition.height == 0:
            continue
        groups = set(partition.get_column("batch").to_list())
        for other in (result.train, result.validation, result.test):
            if other is partition or other.height == 0:
                continue
            assert groups.isdisjoint(set(other.get_column("batch").to_list()))


def test_all_groups_assigned_exactly_once() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    config = SplitConfig(
        strategy=SplitStrategy.GROUP,
        group_column="batch",
        test_size=0.2,
        validation_size=0.2,
        random_state=4,
    )
    result = splitter.split(frame, config)
    assigned = (
        set(result.summary.train_groups)
        | set(result.summary.validation_groups)
        | set(result.summary.test_groups)
    )
    assert assigned == set(frame.get_column("batch").unique().to_list())


def test_group_same_seed_same_membership() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    config = SplitConfig(
        strategy=SplitStrategy.GROUP,
        group_column="batch",
        test_size=0.2,
        random_state=11,
    )
    first = splitter.split(frame, config)
    second = splitter.split(frame, config)
    assert first.summary.test_groups == second.summary.test_groups
    assert first.summary.train_groups == second.summary.train_groups
    assert first.summary.test_original_row_ids == second.summary.test_original_row_ids


def test_group_different_seed_generally_different_membership() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    base = {
        "strategy": SplitStrategy.GROUP,
        "group_column": "batch",
        "test_size": 0.2,
    }
    first = splitter.split(frame, SplitConfig(**base, random_state=1))
    second = splitter.split(frame, SplitConfig(**base, random_state=2))
    assert set(first.summary.test_groups) != set(second.summary.test_groups) or set(
        first.summary.train_groups
    ) != set(second.summary.train_groups)


def test_group_preserves_relative_row_order() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    config = SplitConfig(
        strategy=SplitStrategy.GROUP,
        group_column="batch",
        test_size=0.2,
        random_state=5,
    )
    result = splitter.split(frame, config)
    input_ids = frame.get_column(ORIGINAL_ROW_ID_COLUMN).to_list()
    for partition in (result.train, result.validation, result.test):
        ids = partition.get_column(ORIGINAL_ROW_ID_COLUMN).to_list()
        id_set = set(ids)
        assert ids == [row_id for row_id in input_ids if row_id in id_set]


def test_insufficient_groups_error() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "batch": ["A", "A", "A"],
            "value": [1, 2, 3],
            ORIGINAL_ROW_ID_COLUMN: [0, 1, 2],
        }
    )
    config = SplitConfig(
        strategy=SplitStrategy.GROUP,
        group_column="batch",
        test_size=0.2,
    )
    with pytest.raises(InsufficientDataError):
        splitter.split(frame, config)


def test_validation_requires_at_least_three_groups() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "batch": ["A", "A", "B", "B"],
            "value": [1, 2, 3, 4],
            ORIGINAL_ROW_ID_COLUMN: [0, 1, 2, 3],
        }
    )
    config = SplitConfig(
        strategy=SplitStrategy.GROUP,
        group_column="batch",
        test_size=0.34,
        validation_size=0.34,
    )
    with pytest.raises(InsufficientDataError):
        splitter.split(frame, config)


def test_train_groups_match_train_frame() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    config = SplitConfig(
        strategy=SplitStrategy.GROUP,
        group_column="batch",
        test_size=0.2,
        random_state=6,
    )
    result = splitter.split(frame, config)
    assert set(result.summary.train_groups) == set(
        result.train.get_column("batch").unique().to_list()
    )


def test_validation_groups_match_validation_frame() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    config = SplitConfig(
        strategy=SplitStrategy.GROUP,
        group_column="batch",
        test_size=0.2,
        validation_size=0.2,
        random_state=7,
    )
    result = splitter.split(frame, config)
    assert set(result.summary.validation_groups) == set(
        result.validation.get_column("batch").unique().to_list()
    )


def test_test_groups_match_test_frame() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    config = SplitConfig(
        strategy=SplitStrategy.GROUP,
        group_column="batch",
        test_size=0.2,
        random_state=8,
    )
    result = splitter.split(frame, config)
    assert set(result.summary.test_groups) == set(
        result.test.get_column("batch").unique().to_list()
    )


def test_group_imbalance_produces_fraction_warning() -> None:
    splitter = DatasetSplitter()
    # One huge group and several tiny groups -> realized row fractions drift.
    frame = pl.DataFrame(
        {
            "batch": ["A"] * 20 + ["B", "C", "D", "E"],
            "value": list(range(24)),
            ORIGINAL_ROW_ID_COLUMN: list(range(24)),
        }
    )
    config = SplitConfig(
        strategy=SplitStrategy.GROUP,
        group_column="batch",
        test_size=0.2,
        random_state=0,
    )
    result = splitter.split(frame, config)
    actual = result.test.height / frame.height
    if abs(actual - config.test_size) > 0.05:
        assert any("test fraction" in warning.casefold() for warning in result.summary.warnings)
    else:
        # Force a known-imbalanced assignment by picking a seed that puts A in test.
        found_warning = False
        for seed in range(50):
            trial = splitter.split(
                frame,
                SplitConfig(
                    strategy=SplitStrategy.GROUP,
                    group_column="batch",
                    test_size=0.2,
                    random_state=seed,
                ),
            )
            trial_actual = trial.test.height / frame.height
            if abs(trial_actual - 0.2) > 0.05:
                assert any(
                    "test fraction" in warning.casefold()
                    for warning in trial.summary.warnings
                )
                found_warning = True
                break
        assert found_warning


def test_small_fraction_difference_no_warning() -> None:
    splitter = DatasetSplitter()
    # Equal-sized groups -> row fractions close to group fractions.
    frame = pl.DataFrame(
        {
            "batch": [g for g in ["A", "B", "C", "D", "E"] for _ in range(4)],
            "value": list(range(20)),
            ORIGINAL_ROW_ID_COLUMN: list(range(20)),
        }
    )
    config = SplitConfig(
        strategy=SplitStrategy.GROUP,
        group_column="batch",
        test_size=0.2,
        random_state=0,
    )
    result = splitter.split(frame, config)
    actual = result.test.height / frame.height
    assert abs(actual - 0.2) <= 0.05
    assert not any("test fraction" in warning.casefold() for warning in result.summary.warnings)


# --- TIME split ---


def test_date_time_split_success() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_dates()
    config = SplitConfig(strategy=SplitStrategy.TIME, time_column="ts", test_size=0.2)
    result = splitter.split(frame, config)
    assert result.summary.strategy is SplitStrategy.TIME
    _assert_partitions_cover(frame, result)


def test_datetime_time_split_success() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "ts": [datetime(2020, 1, i, 12, 0, 0) for i in range(1, 11)],
            "value": list(range(10)),
            ORIGINAL_ROW_ID_COLUMN: list(range(10)),
        }
    )
    config = SplitConfig(strategy=SplitStrategy.TIME, time_column="ts", test_size=0.2)
    result = splitter.split(frame, config)
    _assert_partitions_cover(frame, result)


def test_int_sequence_time_split_success() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "ts": [5, 1, 3, 2, 4, 6, 7, 8, 9, 10],
            "value": list(range(10)),
            ORIGINAL_ROW_ID_COLUMN: list(range(10)),
        }
    )
    config = SplitConfig(strategy=SplitStrategy.TIME, time_column="ts", test_size=0.2)
    result = splitter.split(frame, config)
    _assert_partitions_cover(frame, result)
    assert result.train.get_column("ts").to_list() == sorted(
        result.train.get_column("ts").to_list()
    )


def test_float_time_split_success() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "ts": [1.5, 0.5, 2.5, 3.5, 4.5, 5.5],
            "value": list(range(6)),
            ORIGINAL_ROW_ID_COLUMN: list(range(6)),
        }
    )
    config = SplitConfig(strategy=SplitStrategy.TIME, time_column="ts", test_size=0.34)
    result = splitter.split(frame, config)
    _assert_partitions_cover(frame, result)


def test_time_column_missing_rejected() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_dates()
    config = SplitConfig(strategy=SplitStrategy.TIME, time_column="missing")
    with pytest.raises(DataValidationError):
        splitter.split(frame, config)


def test_time_null_rejected() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "ts": pl.Series([date(2020, 1, 1), None], dtype=pl.Date),
            "value": [1, 2],
            ORIGINAL_ROW_ID_COLUMN: [0, 1],
        }
    )
    config = SplitConfig(strategy=SplitStrategy.TIME, time_column="ts")
    with pytest.raises(DataValidationError):
        splitter.split(frame, config)


def test_string_time_dtype_rejected() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "ts": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "value": [1, 2, 3],
            ORIGINAL_ROW_ID_COLUMN: [0, 1, 2],
        }
    )
    config = SplitConfig(strategy=SplitStrategy.TIME, time_column="ts")
    with pytest.raises(DataValidationError):
        splitter.split(frame, config)


def test_boolean_time_dtype_rejected() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "ts": [True, False, True],
            "value": [1, 2, 3],
            ORIGINAL_ROW_ID_COLUMN: [0, 1, 2],
        }
    )
    config = SplitConfig(strategy=SplitStrategy.TIME, time_column="ts")
    with pytest.raises(DataValidationError):
        splitter.split(frame, config)


def test_float_time_nan_rejected() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "ts": [1.0, math.nan, 3.0],
            "value": [1, 2, 3],
            ORIGINAL_ROW_ID_COLUMN: [0, 1, 2],
        }
    )
    config = SplitConfig(strategy=SplitStrategy.TIME, time_column="ts")
    with pytest.raises(DataValidationError):
        splitter.split(frame, config)


def test_float_time_positive_infinity_rejected() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "ts": [1.0, math.inf, 3.0],
            "value": [1, 2, 3],
            ORIGINAL_ROW_ID_COLUMN: [0, 1, 2],
        }
    )
    config = SplitConfig(strategy=SplitStrategy.TIME, time_column="ts")
    with pytest.raises(DataValidationError):
        splitter.split(frame, config)


def test_float_time_negative_infinity_rejected() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "ts": [1.0, -math.inf, 3.0],
            "value": [1, 2, 3],
            ORIGINAL_ROW_ID_COLUMN: [0, 1, 2],
        }
    )
    config = SplitConfig(strategy=SplitStrategy.TIME, time_column="ts")
    with pytest.raises(DataValidationError):
        splitter.split(frame, config)


def test_train_is_past_test_is_future() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_dates()
    config = SplitConfig(strategy=SplitStrategy.TIME, time_column="ts", test_size=0.2)
    result = splitter.split(frame, config)
    assert result.train.get_column("ts").max() <= result.test.get_column("ts").min()


def test_validation_between_train_and_test() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_dates()
    config = SplitConfig(
        strategy=SplitStrategy.TIME,
        time_column="ts",
        test_size=0.2,
        validation_size=0.2,
    )
    result = splitter.split(frame, config)
    assert result.train.get_column("ts").max() <= result.validation.get_column("ts").min()
    assert (
        result.validation.get_column("ts").max() <= result.test.get_column("ts").min()
    )


def test_identical_timestamp_stable_order() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "ts": [1, 2, 2, 2, 3, 4, 5, 6],
            "label": ["a", "b", "c", "d", "e", "f", "g", "h"],
            ORIGINAL_ROW_ID_COLUMN: list(range(8)),
        }
    )
    config = SplitConfig(strategy=SplitStrategy.TIME, time_column="ts", test_size=0.25)
    result = splitter.split(frame, config)
    # Relative order among equal timestamps follows input order.
    twin_ids_in_train = [
        row_id
        for row_id, ts in zip(
            result.train.get_column(ORIGINAL_ROW_ID_COLUMN).to_list(),
            result.train.get_column("ts").to_list(),
            strict=True,
        )
        if ts == 2
    ]
    assert twin_ids_in_train == sorted(twin_ids_in_train)


def test_identical_timestamp_boundary_warning() -> None:
    splitter = DatasetSplitter()
    # 10 rows, test_size=0.2 -> test_count=2, train=8. Boundary between index 7 and 8.
    frame = pl.DataFrame(
        {
            "ts": [1, 2, 3, 4, 5, 6, 7, 8, 8, 9],
            "value": list(range(10)),
            ORIGINAL_ROW_ID_COLUMN: list(range(10)),
        }
    )
    config = SplitConfig(strategy=SplitStrategy.TIME, time_column="ts", test_size=0.2)
    result = splitter.split(frame, config)
    assert any("identical timestamp" in warning.casefold() for warning in result.summary.warnings)
    assert result.summary.warnings.count(
        "identical timestamp values span a split boundary"
    ) <= 1


def test_no_boundary_duplicate_no_warning() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "ts": list(range(10)),
            "value": list(range(10)),
            ORIGINAL_ROW_ID_COLUMN: list(range(10)),
        }
    )
    config = SplitConfig(strategy=SplitStrategy.TIME, time_column="ts", test_size=0.2)
    result = splitter.split(frame, config)
    assert not any(
        "identical timestamp" in warning.casefold() for warning in result.summary.warnings
    )


def test_train_time_range_accuracy() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "ts": list(range(10)),
            "value": list(range(10)),
            ORIGINAL_ROW_ID_COLUMN: list(range(10)),
        }
    )
    config = SplitConfig(strategy=SplitStrategy.TIME, time_column="ts", test_size=0.2)
    result = splitter.split(frame, config)
    assert result.summary.train_time_range == (
        result.train.get_column("ts").min(),
        result.train.get_column("ts").max(),
    )


def test_validation_time_range_accuracy() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "ts": list(range(10)),
            "value": list(range(10)),
            ORIGINAL_ROW_ID_COLUMN: list(range(10)),
        }
    )
    config = SplitConfig(
        strategy=SplitStrategy.TIME,
        time_column="ts",
        test_size=0.2,
        validation_size=0.2,
    )
    result = splitter.split(frame, config)
    assert result.summary.validation_time_range == (
        result.validation.get_column("ts").min(),
        result.validation.get_column("ts").max(),
    )


def test_test_time_range_accuracy() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "ts": list(range(10)),
            "value": list(range(10)),
            ORIGINAL_ROW_ID_COLUMN: list(range(10)),
        }
    )
    config = SplitConfig(strategy=SplitStrategy.TIME, time_column="ts", test_size=0.2)
    result = splitter.split(frame, config)
    assert result.summary.test_time_range == (
        result.test.get_column("ts").min(),
        result.test.get_column("ts").max(),
    )


def test_no_validation_time_range_is_none() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_dates()
    config = SplitConfig(strategy=SplitStrategy.TIME, time_column="ts", test_size=0.2)
    result = splitter.split(frame, config)
    assert result.summary.validation_time_range is None
    assert result.validation.height == 0


def test_time_split_result_is_time_ascending() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_dates()
    config = SplitConfig(
        strategy=SplitStrategy.TIME,
        time_column="ts",
        test_size=0.2,
        validation_size=0.2,
    )
    result = splitter.split(frame, config)
    for partition in (result.train, result.validation, result.test):
        if partition.height == 0:
            continue
        values = partition.get_column("ts").to_list()
        assert values == sorted(values)


def test_time_split_does_not_reorder_input_frame() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_dates()
    original = frame.clone()
    config = SplitConfig(strategy=SplitStrategy.TIME, time_column="ts", test_size=0.2)
    splitter.split(frame, config)
    assert_frame_equal(frame, original)


# --- RANDOM split ---


def test_random_split_blocked_without_consent() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "x": list(range(10)),
            ORIGINAL_ROW_ID_COLUMN: list(range(10)),
        }
    )
    config = SplitConfig(strategy=SplitStrategy.RANDOM, allow_random_split=False)
    with pytest.raises(DataLeakageError):
        splitter.split(frame, config)


def test_random_split_succeeds_with_consent() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "x": list(range(10)),
            ORIGINAL_ROW_ID_COLUMN: list(range(10)),
        }
    )
    config = SplitConfig(
        strategy=SplitStrategy.RANDOM,
        allow_random_split=True,
        test_size=0.2,
        random_state=0,
    )
    result = splitter.split(frame, config)
    _assert_partitions_cover(frame, result)


def test_random_split_warning_present() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "x": list(range(10)),
            ORIGINAL_ROW_ID_COLUMN: list(range(10)),
        }
    )
    config = SplitConfig(
        strategy=SplitStrategy.RANDOM,
        allow_random_split=True,
        random_state=0,
    )
    result = splitter.split(frame, config)
    assert result.summary.warnings
    assert any("random" in warning.casefold() for warning in result.summary.warnings)


def test_random_split_warning_mentions_user_consent() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "x": list(range(10)),
            ORIGINAL_ROW_ID_COLUMN: list(range(10)),
        }
    )
    config = SplitConfig(
        strategy=SplitStrategy.RANDOM,
        allow_random_split=True,
        random_state=0,
    )
    result = splitter.split(frame, config)
    joined = " ".join(result.summary.warnings).casefold()
    assert "consent" in joined or "explicit" in joined


def test_random_same_seed_same_membership() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "x": list(range(20)),
            ORIGINAL_ROW_ID_COLUMN: list(range(20)),
        }
    )
    config = SplitConfig(
        strategy=SplitStrategy.RANDOM,
        allow_random_split=True,
        test_size=0.25,
        random_state=99,
    )
    first = splitter.split(frame, config)
    second = splitter.split(frame, config)
    assert first.summary.test_original_row_ids == second.summary.test_original_row_ids
    assert first.summary.train_original_row_ids == second.summary.train_original_row_ids


def test_random_different_seed_generally_different_membership() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "x": list(range(30)),
            ORIGINAL_ROW_ID_COLUMN: list(range(30)),
        }
    )
    first = splitter.split(
        frame,
        SplitConfig(
            strategy=SplitStrategy.RANDOM,
            allow_random_split=True,
            test_size=0.3,
            random_state=1,
        ),
    )
    second = splitter.split(
        frame,
        SplitConfig(
            strategy=SplitStrategy.RANDOM,
            allow_random_split=True,
            test_size=0.3,
            random_state=2,
        ),
    )
    assert set(first.summary.test_original_row_ids) != set(
        second.summary.test_original_row_ids
    )


def test_random_preserves_relative_row_order_within_partitions() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "x": list(range(20)),
            ORIGINAL_ROW_ID_COLUMN: list(range(20)),
        }
    )
    config = SplitConfig(
        strategy=SplitStrategy.RANDOM,
        allow_random_split=True,
        test_size=0.25,
        random_state=3,
    )
    result = splitter.split(frame, config)
    input_ids = frame.get_column(ORIGINAL_ROW_ID_COLUMN).to_list()
    for partition in (result.train, result.validation, result.test):
        ids = partition.get_column(ORIGINAL_ROW_ID_COLUMN).to_list()
        id_set = set(ids)
        assert ids == [row_id for row_id in input_ids if row_id in id_set]


def test_random_no_temp_columns_left() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "x": list(range(10)),
            ORIGINAL_ROW_ID_COLUMN: list(range(10)),
        }
    )
    config = SplitConfig(
        strategy=SplitStrategy.RANDOM,
        allow_random_split=True,
        random_state=0,
    )
    result = splitter.split(frame, config)
    for partition in (result.train, result.validation, result.test):
        assert partition.columns == frame.columns
        assert not any("internal_split" in name for name in partition.columns)


# --- Common results ---


def test_dataset_split_is_frozen_dataclass() -> None:
    assert is_dataclass(DatasetSplit)
    assert DatasetSplit is DirectDatasetSplit
    frame = pl.DataFrame({"x": [1, 2, 3], ORIGINAL_ROW_ID_COLUMN: [0, 1, 2]})
    splitter = DatasetSplitter()
    result = splitter.split(
        frame,
        SplitConfig(
            strategy=SplitStrategy.RANDOM,
            allow_random_split=True,
            test_size=0.34,
            minimum_train_rows=1,
            random_state=0,
        ),
    )
    with pytest.raises(FrozenInstanceError):
        result.train = frame  # type: ignore[misc]


def test_dataset_split_uses_slots() -> None:
    field_names = {item.name for item in fields(DatasetSplit)}
    assert field_names == {"train", "validation", "test", "summary"}
    assert hasattr(DatasetSplit, "__slots__")


def test_partitions_are_polars_dataframes() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    result = splitter.split(
        frame,
        SplitConfig(
            strategy=SplitStrategy.GROUP,
            group_column="batch",
            test_size=0.2,
            random_state=0,
        ),
    )
    assert isinstance(result.train, pl.DataFrame)
    assert isinstance(result.validation, pl.DataFrame)
    assert isinstance(result.test, pl.DataFrame)


def test_partition_column_order_matches_input() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    result = splitter.split(
        frame,
        SplitConfig(
            strategy=SplitStrategy.GROUP,
            group_column="batch",
            test_size=0.2,
            random_state=0,
        ),
    )
    for partition in (result.train, result.validation, result.test):
        assert partition.columns == frame.columns


def test_partition_dtypes_match_input() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    result = splitter.split(
        frame,
        SplitConfig(
            strategy=SplitStrategy.GROUP,
            group_column="batch",
            test_size=0.2,
            random_state=0,
        ),
    )
    for partition in (result.train, result.validation, result.test):
        assert partition.schema == frame.schema


def test_original_row_id_dtype_preserved() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "batch": ["A", "A", "B", "B", "C", "C"],
            ORIGINAL_ROW_ID_COLUMN: pl.Series([0, 1, 2, 3, 4, 5], dtype=pl.UInt32),
        }
    )
    result = splitter.split(
        frame,
        SplitConfig(
            strategy=SplitStrategy.GROUP,
            group_column="batch",
            test_size=0.34,
            random_state=0,
        ),
    )
    for partition in (result.train, result.validation, result.test):
        assert partition.schema[ORIGINAL_ROW_ID_COLUMN] == pl.UInt32


def test_partition_original_ids_disjoint_and_complete() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    result = splitter.split(
        frame,
        SplitConfig(
            strategy=SplitStrategy.GROUP,
            group_column="batch",
            test_size=0.2,
            validation_size=0.2,
            random_state=0,
        ),
    )
    _assert_partitions_cover(frame, result)


def test_no_row_duplication() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    result = splitter.split(
        frame,
        SplitConfig(
            strategy=SplitStrategy.GROUP,
            group_column="batch",
            test_size=0.2,
            random_state=0,
        ),
    )
    all_ids = (
        result.train.get_column(ORIGINAL_ROW_ID_COLUMN).to_list()
        + result.validation.get_column(ORIGINAL_ROW_ID_COLUMN).to_list()
        + result.test.get_column(ORIGINAL_ROW_ID_COLUMN).to_list()
    )
    assert len(all_ids) == len(set(all_ids)) == frame.height


def test_validation_size_zero_yields_empty_validation() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    result = splitter.split(
        frame,
        SplitConfig(
            strategy=SplitStrategy.GROUP,
            group_column="batch",
            test_size=0.2,
            validation_size=0.0,
            random_state=0,
        ),
    )
    assert result.validation.height == 0
    assert result.validation.schema == frame.schema
    assert result.validation.columns == frame.columns


def test_summary_row_counts_match_partitions() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    result = splitter.split(
        frame,
        SplitConfig(
            strategy=SplitStrategy.GROUP,
            group_column="batch",
            test_size=0.2,
            random_state=0,
        ),
    )
    assert result.summary.train_row_count == result.train.height
    assert result.summary.validation_row_count == result.validation.height
    assert result.summary.test_row_count == result.test.height


def test_summary_original_ids_match_frame_order() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    result = splitter.split(
        frame,
        SplitConfig(
            strategy=SplitStrategy.GROUP,
            group_column="batch",
            test_size=0.2,
            random_state=0,
        ),
    )
    assert result.summary.train_original_row_ids == result.train.get_column(
        ORIGINAL_ROW_ID_COLUMN
    ).to_list()
    assert result.summary.test_original_row_ids == result.test.get_column(
        ORIGINAL_ROW_ID_COLUMN
    ).to_list()


def test_summary_fractions_match_actual_ratios() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    result = splitter.split(
        frame,
        SplitConfig(
            strategy=SplitStrategy.GROUP,
            group_column="batch",
            test_size=0.2,
            random_state=0,
        ),
    )
    total = frame.height
    assert result.summary.train_fraction == pytest.approx(
        round(result.train.height / total, 6)
    )
    assert result.summary.test_fraction == pytest.approx(
        round(result.test.height / total, 6)
    )
    assert result.summary.validation_fraction == pytest.approx(
        round(result.validation.height / total, 6)
    )


def test_summary_requested_sizes_match_config() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    config = SplitConfig(
        strategy=SplitStrategy.GROUP,
        group_column="batch",
        test_size=0.25,
        validation_size=0.15,
        random_state=0,
    )
    result = splitter.split(frame, config)
    assert result.summary.requested_test_size == config.test_size
    assert result.summary.requested_validation_size == config.validation_size


def test_summary_group_time_columns_match_config() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    config = SplitConfig(
        strategy=SplitStrategy.GROUP,
        group_column="batch",
        test_size=0.2,
        random_state=0,
    )
    result = splitter.split(frame, config)
    assert result.summary.group_column == "batch"
    assert result.summary.time_column is None

    time_frame = _frame_with_dates()
    time_config = SplitConfig(
        strategy=SplitStrategy.TIME,
        time_column="ts",
        test_size=0.2,
    )
    time_result = splitter.split(time_frame, time_config)
    assert time_result.summary.time_column == "ts"
    assert time_result.summary.group_column is None


def test_summary_warnings_are_independent_mutable_lists() -> None:
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "x": list(range(10)),
            ORIGINAL_ROW_ID_COLUMN: list(range(10)),
        }
    )
    config = SplitConfig(
        strategy=SplitStrategy.RANDOM,
        allow_random_split=True,
        random_state=0,
    )
    first = splitter.split(frame, config)
    second = splitter.split(frame, config)
    first.summary.warnings.append("mutated")
    assert "mutated" not in second.summary.warnings


def test_split_summary_model_dump_round_trip() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    result = splitter.split(
        frame,
        SplitConfig(
            strategy=SplitStrategy.GROUP,
            group_column="batch",
            test_size=0.2,
            random_state=0,
        ),
    )
    dumped = result.summary.model_dump()
    restored = SplitSummary.model_validate(dumped)
    assert restored == result.summary
    assert isinstance(restored, DirectSplitSummary)


# --- Immutability and determinism ---


def test_split_does_not_mutate_input_frame() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    original = frame.clone()
    splitter.split(
        frame,
        SplitConfig(
            strategy=SplitStrategy.GROUP,
            group_column="batch",
            test_size=0.2,
            random_state=0,
        ),
    )
    assert_frame_equal(frame, original)


def test_split_does_not_mutate_config() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    config = SplitConfig(
        strategy=SplitStrategy.GROUP,
        group_column="batch",
        test_size=0.2,
        random_state=0,
    )
    before = config.model_dump()
    splitter.split(frame, config)
    assert config.model_dump() == before


def test_deterministic_results_for_same_inputs() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    config = SplitConfig(
        strategy=SplitStrategy.GROUP,
        group_column="batch",
        test_size=0.2,
        random_state=42,
    )
    first = splitter.split(frame, config)
    second = splitter.split(frame, config)
    assert_frame_equal(first.train, second.train)
    assert_frame_equal(first.test, second.test)
    assert first.summary.model_dump() == second.summary.model_dump()


def test_consecutive_splits_do_not_accumulate_state() -> None:
    splitter = DatasetSplitter()
    frame_a = _frame_with_groups()
    frame_b = pl.DataFrame(
        {
            "batch": ["X", "X", "Y", "Y", "Z", "Z"],
            "value": [1, 2, 3, 4, 5, 6],
            ORIGINAL_ROW_ID_COLUMN: [0, 1, 2, 3, 4, 5],
        }
    )
    config_a = SplitConfig(
        strategy=SplitStrategy.GROUP,
        group_column="batch",
        test_size=0.34,
        random_state=0,
    )
    config_b = SplitConfig(
        strategy=SplitStrategy.GROUP,
        group_column="batch",
        test_size=0.34,
        random_state=0,
    )
    first_a = splitter.split(frame_a, config_a)
    splitter.split(frame_b, config_b)
    second_a = splitter.split(frame_a, config_a)
    assert first_a.summary.test_original_row_ids == second_a.summary.test_original_row_ids


def test_different_splitter_instances_do_not_share_state() -> None:
    frame = _frame_with_groups()
    config = SplitConfig(
        strategy=SplitStrategy.GROUP,
        group_column="batch",
        test_size=0.2,
        random_state=0,
    )
    first = DatasetSplitter().split(frame, config)
    second = DatasetSplitter().split(frame, config)
    assert first.summary.test_original_row_ids == second.summary.test_original_row_ids
    assert_frame_equal(first.train, second.train)


def test_mutating_returned_partition_does_not_change_input() -> None:
    splitter = DatasetSplitter()
    frame = _frame_with_groups()
    original = frame.clone()
    result = splitter.split(
        frame,
        SplitConfig(
            strategy=SplitStrategy.GROUP,
            group_column="batch",
            test_size=0.2,
            random_state=0,
        ),
    )
    _ = result.train.with_columns(pl.lit(999).alias("value"))
    assert_frame_equal(frame, original)


def test_row_partition_counts_follow_ceil_rule() -> None:
    """TIME/RANDOM use ceil(total * size) for test/validation counts."""
    splitter = DatasetSplitter()
    frame = pl.DataFrame(
        {
            "ts": list(range(10)),
            "value": list(range(10)),
            ORIGINAL_ROW_ID_COLUMN: list(range(10)),
        }
    )
    test_size = 0.2
    validation_size = 0.15
    result = splitter.split(
        frame,
        SplitConfig(
            strategy=SplitStrategy.TIME,
            time_column="ts",
            test_size=test_size,
            validation_size=validation_size,
        ),
    )
    expected_test = ceil(10 * test_size)
    expected_validation = ceil(10 * validation_size)
    assert result.test.height == expected_test
    assert result.validation.height == expected_validation
    assert result.train.height == 10 - expected_test - expected_validation


def test_package_exports_match_direct_imports() -> None:
    assert DatasetSplit is DirectDatasetSplit
    assert DatasetSplitter is DirectDatasetSplitter
    assert SplitConfig is DirectSplitConfig
    assert SplitStrategy is DirectSplitStrategy
    assert SplitSummary is DirectSplitSummary


def test_group_and_random_time_ranges_are_none() -> None:
    splitter = DatasetSplitter()
    group_result = splitter.split(
        _frame_with_groups(),
        SplitConfig(
            strategy=SplitStrategy.GROUP,
            group_column="batch",
            test_size=0.2,
            random_state=0,
        ),
    )
    assert group_result.summary.train_time_range is None
    assert group_result.summary.test_time_range is None

    random_result = splitter.split(
        pl.DataFrame({"x": list(range(10)), ORIGINAL_ROW_ID_COLUMN: list(range(10))}),
        SplitConfig(
            strategy=SplitStrategy.RANDOM,
            allow_random_split=True,
            random_state=0,
        ),
    )
    assert random_result.summary.train_time_range is None
    assert random_result.summary.train_groups == []
    assert random_result.summary.test_groups == []
