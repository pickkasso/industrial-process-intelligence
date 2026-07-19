"""Leakage-safe dataset splitting (Step 5A).

Supports GROUP, TIME, and explicitly consented RANDOM splits only.
LeakageChecker, cross-validation, and model training are out of scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Self

import numpy as np
import polars as pl
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from process_intelligence.core.exceptions import (
    DataLeakageError,
    DataValidationError,
    InsufficientDataError,
    ProcessIntelligenceError,
)
from process_intelligence.data.loader import ORIGINAL_ROW_ID_COLUMN

_FRACTION_DIGITS = 6
_FRACTION_WARNING_TOLERANCE = 0.05
_BOUNDARY_WARNING = "identical timestamp values span a split boundary"
_RANDOM_SPLIT_WARNING = (
    "RANDOM split used with explicit user consent "
    "(allow_random_split=True); prefer GROUP or TIME split when available"
)
_INTERNAL_ROW_POS_COLUMN = "__pi_internal_split_row_pos__"


class SplitStrategy(StrEnum):
    """Supported leakage-aware dataset split strategies."""

    GROUP = "GROUP"
    TIME = "TIME"
    RANDOM = "RANDOM"


class SplitConfig(BaseModel):
    """Configuration for a single leakage-safe dataset split.

    RANDOM splits require ``allow_random_split=True`` at execution time.
    Config construction with ``allow_random_split=False`` is allowed so callers
    can present the option before obtaining explicit consent.
    """

    strategy: SplitStrategy
    test_size: float = 0.20
    validation_size: float = 0.0
    group_column: str | None = None
    time_column: str | None = None
    random_state: int = 42
    allow_random_split: bool = False
    minimum_train_rows: int = 2
    minimum_test_rows: int = 1
    minimum_validation_rows: int = 1

    @field_validator("test_size", mode="before")
    @classmethod
    def _validate_test_size(cls, value: object) -> float:
        return _validate_open_unit_interval(value, field_name="test_size")

    @field_validator("validation_size", mode="before")
    @classmethod
    def _validate_validation_size(cls, value: object) -> float:
        return _validate_half_open_unit_interval(value, field_name="validation_size")

    @field_validator("random_state", mode="before")
    @classmethod
    def _validate_random_state(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"random_state must be an int >= 0 (bool not allowed), got {value!r}"
            )
        if value < 0:
            raise ValueError(f"random_state must be >= 0, got {value}")
        return value

    @field_validator(
        "minimum_train_rows",
        "minimum_test_rows",
        "minimum_validation_rows",
        mode="before",
    )
    @classmethod
    def _validate_minimum_rows(cls, value: object, info: ValidationInfo) -> int:
        field_name = str(info.field_name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"{field_name} must be an int >= 1 (bool not allowed), got {value!r}"
            )
        if value < 1:
            raise ValueError(f"{field_name} must be >= 1, got {value}")
        return value

    @field_validator("allow_random_split", mode="before")
    @classmethod
    def _validate_allow_random_split(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError(
                f"allow_random_split must be a bool, got {type(value).__name__}"
            )
        return value

    @field_validator("group_column", "time_column", mode="before")
    @classmethod
    def _validate_optional_column_name(
        cls, value: object, info: ValidationInfo
    ) -> str | None:
        field_name = str(info.field_name)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(
                f"{field_name} must be None or str, got {type(value).__name__}"
            )
        if value == "" or value.strip() == "":
            raise ValueError(
                f"{field_name} must be a non-empty, non-whitespace string"
            )
        return value

    @model_validator(mode="after")
    def _validate_sizes_and_strategy_columns(self) -> Self:
        if self.test_size + self.validation_size >= 1.0:
            raise ValueError(
                "test_size + validation_size must be < 1, "
                f"got {self.test_size} + {self.validation_size}"
            )

        if self.strategy is SplitStrategy.GROUP:
            if self.group_column is None:
                raise ValueError("GROUP strategy requires group_column")
            if self.time_column is not None:
                raise ValueError("GROUP strategy does not allow time_column")
        elif self.strategy is SplitStrategy.TIME:
            if self.time_column is None:
                raise ValueError("TIME strategy requires time_column")
            if self.group_column is not None:
                raise ValueError("TIME strategy does not allow group_column")
        elif self.strategy is SplitStrategy.RANDOM:
            if self.group_column is not None:
                raise ValueError("RANDOM strategy does not allow group_column")
            if self.time_column is not None:
                raise ValueError("RANDOM strategy does not allow time_column")
        return self


class SplitSummary(BaseModel):
    """Structured summary of a completed dataset split.

    Records requested sizes, realized row counts and fractions, original-row
    membership, and strategy-specific group or time-range metadata.
    """

    strategy: SplitStrategy
    group_column: str | None = None
    time_column: str | None = None
    random_state: int
    requested_test_size: float
    requested_validation_size: float
    train_row_count: int = Field(ge=0)
    validation_row_count: int = Field(ge=0)
    test_row_count: int = Field(ge=0)
    train_fraction: float = Field(ge=0.0, le=1.0)
    validation_fraction: float = Field(ge=0.0, le=1.0)
    test_fraction: float = Field(ge=0.0, le=1.0)
    train_original_row_ids: list[int] = Field(default_factory=list)
    validation_original_row_ids: list[int] = Field(default_factory=list)
    test_original_row_ids: list[int] = Field(default_factory=list)
    train_groups: list[Any] = Field(default_factory=list)
    validation_groups: list[Any] = Field(default_factory=list)
    test_groups: list[Any] = Field(default_factory=list)
    train_time_range: tuple[Any | None, Any | None] | None = None
    validation_time_range: tuple[Any | None, Any | None] | None = None
    test_time_range: tuple[Any | None, Any | None] | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_collections_and_disjointness(self) -> Self:
        _reject_duplicates(
            self.train_original_row_ids,
            field_name="train_original_row_ids",
        )
        _reject_duplicates(
            self.validation_original_row_ids,
            field_name="validation_original_row_ids",
        )
        _reject_duplicates(
            self.test_original_row_ids,
            field_name="test_original_row_ids",
        )

        train_ids = set(self.train_original_row_ids)
        validation_ids = set(self.validation_original_row_ids)
        test_ids = set(self.test_original_row_ids)
        if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
            raise ValueError(
                "train/validation/test original row ID sets must be disjoint"
            )

        _reject_duplicates(self.train_groups, field_name="train_groups")
        _reject_duplicates(self.validation_groups, field_name="validation_groups")
        _reject_duplicates(self.test_groups, field_name="test_groups")

        if self.strategy is SplitStrategy.GROUP:
            train_groups = set(self.train_groups)
            validation_groups = set(self.validation_groups)
            test_groups = set(self.test_groups)
            if (
                train_groups & validation_groups
                or train_groups & test_groups
                or validation_groups & test_groups
            ):
                raise ValueError(
                    "GROUP strategy train/validation/test group sets must be disjoint"
                )
        return self


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Immutable container for train/validation/test partitions and summary."""

    train: pl.DataFrame
    validation: pl.DataFrame
    test: pl.DataFrame
    summary: SplitSummary


class DatasetSplitter:
    """Stateless splitter that produces leakage-aware dataset partitions.

    Instances hold no fitted state and never mutate input frames or configs.
    """

    def split(self, frame: pl.DataFrame, config: SplitConfig) -> DatasetSplit:
        """Split ``frame`` according to ``config`` without mutating inputs.

        Args:
            frame: Polars DataFrame containing ``_original_row_id``.
            config: Validated split configuration.

        Returns:
            A ``DatasetSplit`` with disjoint partitions covering every input row.

        Raises:
            TypeError: If ``frame`` or ``config`` has an unexpected type.
            DataValidationError: If required columns or ID constraints fail.
            DataLeakageError: If RANDOM is requested without explicit consent.
            InsufficientDataError: If partitions cannot meet minimum sizes.
            ProcessIntelligenceError: If TIME chronology invariants are violated.
        """
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(
                f"Expected a polars.DataFrame, got {type(frame).__name__}"
            )
        if not isinstance(config, SplitConfig):
            raise TypeError(
                f"Expected a SplitConfig, got {type(config).__name__}"
            )

        _validate_original_row_id_column(frame)

        if frame.height == 0:
            raise InsufficientDataError("Cannot split an empty DataFrame")

        if config.strategy is SplitStrategy.GROUP:
            return self._split_group(frame, config)
        if config.strategy is SplitStrategy.TIME:
            return self._split_time(frame, config)
        return self._split_random(frame, config)

    def _split_group(self, frame: pl.DataFrame, config: SplitConfig) -> DatasetSplit:
        group_column = config.group_column
        assert group_column is not None

        if group_column not in frame.columns:
            raise DataValidationError(
                f"group_column {group_column!r} is not present in the frame"
            )

        group_series = frame.get_column(group_column)
        if group_series.null_count() > 0:
            raise DataValidationError(
                f"group_column {group_column!r} must not contain null values"
            )

        unique_groups = group_series.unique(maintain_order=True).to_list()
        group_count = len(unique_groups)
        if group_count == 0:
            raise InsufficientDataError("No groups available for GROUP split")

        test_group_count = math.ceil(group_count * config.test_size)
        if config.validation_size > 0.0:
            validation_group_count = math.ceil(group_count * config.validation_size)
        else:
            validation_group_count = 0
        train_group_count = group_count - test_group_count - validation_group_count

        if test_group_count < 1 or train_group_count < 1:
            raise InsufficientDataError(
                "GROUP split requires at least one train group and one test group"
            )
        if config.validation_size > 0.0 and validation_group_count < 1:
            raise InsufficientDataError(
                "GROUP split with validation requires at least one validation group"
            )
        if config.validation_size > 0.0 and group_count < 3:
            raise InsufficientDataError(
                "GROUP split with validation requires at least three groups"
            )

        rng = np.random.default_rng(config.random_state)
        permutation = rng.permutation(group_count)

        test_groups = [unique_groups[int(i)] for i in permutation[:test_group_count]]
        validation_groups = [
            unique_groups[int(i)]
            for i in permutation[
                test_group_count : test_group_count + validation_group_count
            ]
        ]
        train_groups = [
            unique_groups[int(i)]
            for i in permutation[test_group_count + validation_group_count :]
        ]

        test_set = set(test_groups)
        validation_set = set(validation_groups)
        train_set = set(train_groups)

        train = frame.filter(pl.col(group_column).is_in(list(train_set)))
        validation = (
            frame.filter(pl.col(group_column).is_in(list(validation_set)))
            if validation_group_count > 0
            else frame.clear()
        )
        test = frame.filter(pl.col(group_column).is_in(list(test_set)))

        warnings: list[str] = []
        total_rows = frame.height
        actual_test_fraction = test.height / total_rows
        if abs(actual_test_fraction - config.test_size) > _FRACTION_WARNING_TOLERANCE:
            warnings.append(
                "Actual test fraction "
                f"{actual_test_fraction:.6f} differs from requested test_size "
                f"{config.test_size} by more than {_FRACTION_WARNING_TOLERANCE}"
            )
        if config.validation_size > 0.0:
            actual_validation_fraction = validation.height / total_rows
            if (
                abs(actual_validation_fraction - config.validation_size)
                > _FRACTION_WARNING_TOLERANCE
            ):
                warnings.append(
                    "Actual validation fraction "
                    f"{actual_validation_fraction:.6f} differs from requested "
                    f"validation_size {config.validation_size} by more than "
                    f"{_FRACTION_WARNING_TOLERANCE}"
                )

        _enforce_minimum_rows(
            train_rows=train.height,
            validation_rows=validation.height,
            test_rows=test.height,
            config=config,
        )

        summary = _build_summary(
            config=config,
            train=train,
            validation=validation,
            test=test,
            train_groups=train_groups,
            validation_groups=validation_groups,
            test_groups=test_groups,
            train_time_range=None,
            validation_time_range=None,
            test_time_range=None,
            warnings=warnings,
        )
        return DatasetSplit(
            train=train,
            validation=validation,
            test=test,
            summary=summary,
        )

    def _split_time(self, frame: pl.DataFrame, config: SplitConfig) -> DatasetSplit:
        time_column = config.time_column
        assert time_column is not None

        if time_column not in frame.columns:
            raise DataValidationError(
                f"time_column {time_column!r} is not present in the frame"
            )

        time_dtype = frame.schema[time_column]
        if not _is_allowed_time_dtype(time_dtype):
            raise DataValidationError(
                f"time_column {time_column!r} has unsupported dtype {time_dtype}"
            )

        time_series = frame.get_column(time_column)
        if time_series.null_count() > 0:
            raise DataValidationError(
                f"time_column {time_column!r} must not contain null values"
            )

        if time_dtype.is_float():
            if bool(time_series.is_nan().any()):
                raise DataValidationError(
                    f"time_column {time_column!r} must not contain NaN values"
                )
            if bool(time_series.is_infinite().any()):
                raise DataValidationError(
                    f"time_column {time_column!r} must not contain infinite values"
                )

        sorted_frame = frame.sort(time_column, maintain_order=True)
        total_rows = sorted_frame.height
        test_count, validation_count, train_count = _compute_row_partition_counts(
            total_rows=total_rows,
            test_size=config.test_size,
            validation_size=config.validation_size,
        )

        train = sorted_frame.slice(0, train_count)
        validation = (
            sorted_frame.slice(train_count, validation_count)
            if validation_count > 0
            else frame.clear()
        )
        test = sorted_frame.slice(train_count + validation_count, test_count)

        warnings: list[str] = []
        _check_time_chronology(
            train=train,
            validation=validation,
            test=test,
            time_column=time_column,
            has_validation=validation_count > 0,
            warnings=warnings,
        )

        _enforce_minimum_rows(
            train_rows=train.height,
            validation_rows=validation.height,
            test_rows=test.height,
            config=config,
        )

        train_time_range = _time_range(train, time_column)
        validation_time_range = (
            _time_range(validation, time_column) if validation_count > 0 else None
        )
        test_time_range = _time_range(test, time_column)

        summary = _build_summary(
            config=config,
            train=train,
            validation=validation,
            test=test,
            train_groups=[],
            validation_groups=[],
            test_groups=[],
            train_time_range=train_time_range,
            validation_time_range=validation_time_range,
            test_time_range=test_time_range,
            warnings=warnings,
        )
        return DatasetSplit(
            train=train,
            validation=validation,
            test=test,
            summary=summary,
        )

    def _split_random(self, frame: pl.DataFrame, config: SplitConfig) -> DatasetSplit:
        if not config.allow_random_split:
            raise DataLeakageError(
                "RANDOM split is blocked without explicit user consent; "
                "set allow_random_split=True only when GROUP/TIME splits "
                "are unavailable and the leakage risk is accepted"
            )

        total_rows = frame.height
        test_count, validation_count, train_count = _compute_row_partition_counts(
            total_rows=total_rows,
            test_size=config.test_size,
            validation_size=config.validation_size,
        )

        rng = np.random.default_rng(config.random_state)
        permutation = rng.permutation(total_rows)
        test_positions = {int(i) for i in permutation[:test_count]}
        validation_positions = {
            int(i)
            for i in permutation[test_count : test_count + validation_count]
        }
        train_positions = {
            int(i) for i in permutation[test_count + validation_count :]
        }

        temp_column = _INTERNAL_ROW_POS_COLUMN
        if temp_column in frame.columns:
            temp_column = f"{_INTERNAL_ROW_POS_COLUMN}_{config.random_state}"
            while temp_column in frame.columns:
                temp_column = f"{temp_column}_x"

        indexed = frame.with_columns(
            pl.int_range(0, pl.len(), dtype=pl.Int64).alias(temp_column)
        )
        train = indexed.filter(pl.col(temp_column).is_in(list(train_positions))).drop(
            temp_column
        )
        validation = (
            indexed.filter(pl.col(temp_column).is_in(list(validation_positions))).drop(
                temp_column
            )
            if validation_count > 0
            else frame.clear()
        )
        test = indexed.filter(pl.col(temp_column).is_in(list(test_positions))).drop(
            temp_column
        )

        # Ensure slice counts match assignment (guards against set/filter drift).
        if (
            train.height != train_count
            or validation.height != validation_count
            or test.height != test_count
        ):
            raise ProcessIntelligenceError(
                "RANDOM split produced unexpected partition sizes"
            )

        _enforce_minimum_rows(
            train_rows=train.height,
            validation_rows=validation.height,
            test_rows=test.height,
            config=config,
        )

        summary = _build_summary(
            config=config,
            train=train,
            validation=validation,
            test=test,
            train_groups=[],
            validation_groups=[],
            test_groups=[],
            train_time_range=None,
            validation_time_range=None,
            test_time_range=None,
            warnings=[_RANDOM_SPLIT_WARNING],
        )
        return DatasetSplit(
            train=train,
            validation=validation,
            test=test,
            summary=summary,
        )


def _validate_open_unit_interval(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a finite number in (0, 1) "
            f"(bool not allowed), got {value!r}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(
            f"{field_name} must be a finite number in (0, 1), got {value!r}"
        )
    if number <= 0.0 or number >= 1.0:
        raise ValueError(f"{field_name} must be in (0, 1), got {number}")
    return number


def _validate_half_open_unit_interval(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a finite number in [0, 1) "
            f"(bool not allowed), got {value!r}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(
            f"{field_name} must be a finite number in [0, 1), got {value!r}"
        )
    if number < 0.0 or number >= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1), got {number}")
    return number


def _reject_duplicates(values: list[Any], *, field_name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates")


def _validate_original_row_id_column(frame: pl.DataFrame) -> None:
    if ORIGINAL_ROW_ID_COLUMN not in frame.columns:
        raise DataValidationError(
            f"frame is missing required column '{ORIGINAL_ROW_ID_COLUMN}'"
        )

    dtype = frame.schema[ORIGINAL_ROW_ID_COLUMN]
    if dtype == pl.Boolean:
        raise DataValidationError(
            f"column '{ORIGINAL_ROW_ID_COLUMN}' must be an integer type, got Boolean"
        )
    if not dtype.is_integer():
        raise DataValidationError(
            f"column '{ORIGINAL_ROW_ID_COLUMN}' must be an integer type, got {dtype}"
        )

    series = frame.get_column(ORIGINAL_ROW_ID_COLUMN)
    if series.null_count() > 0:
        raise DataValidationError(
            f"column '{ORIGINAL_ROW_ID_COLUMN}' must not contain null values"
        )
    if series.n_unique() != series.len():
        raise DataValidationError(
            f"column '{ORIGINAL_ROW_ID_COLUMN}' must contain unique values"
        )
    if series.len() > 0 and bool((series < 0).any()):
        raise DataValidationError(
            f"column '{ORIGINAL_ROW_ID_COLUMN}' must not contain negative values"
        )


def _is_allowed_time_dtype(dtype: pl.DataType) -> bool:
    if dtype == pl.Boolean or dtype == pl.String:
        return False
    base = dtype.base_type()
    if base in (pl.Categorical, pl.Enum):
        return False
    if dtype.is_integer() or dtype.is_float() or dtype.is_decimal():
        return True
    return base in (pl.Date, pl.Datetime, pl.Time)


def _compute_row_partition_counts(
    *,
    total_rows: int,
    test_size: float,
    validation_size: float,
) -> tuple[int, int, int]:
    test_count = math.ceil(total_rows * test_size)
    if validation_size > 0.0:
        validation_count = math.ceil(total_rows * validation_size)
    else:
        validation_count = 0
    train_count = total_rows - test_count - validation_count
    if train_count < 0:
        raise InsufficientDataError(
            "Requested test_size and validation_size leave no rows for train"
        )
    return test_count, validation_count, train_count


def _enforce_minimum_rows(
    *,
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    config: SplitConfig,
) -> None:
    if train_rows < config.minimum_train_rows:
        raise InsufficientDataError(
            f"train partition has {train_rows} rows, "
            f"minimum_train_rows={config.minimum_train_rows}"
        )
    if test_rows < config.minimum_test_rows:
        raise InsufficientDataError(
            f"test partition has {test_rows} rows, "
            f"minimum_test_rows={config.minimum_test_rows}"
        )
    if config.validation_size > 0.0:
        if validation_rows < config.minimum_validation_rows:
            raise InsufficientDataError(
                f"validation partition has {validation_rows} rows, "
                f"minimum_validation_rows={config.minimum_validation_rows}"
            )
    elif validation_rows != 0:
        raise ProcessIntelligenceError(
            "validation_size is 0 but validation partition is not empty"
        )


def _original_row_ids(frame: pl.DataFrame) -> list[int]:
    return [int(value) for value in frame.get_column(ORIGINAL_ROW_ID_COLUMN).to_list()]


def _fraction(count: int, total: int) -> float:
    return round(count / total, _FRACTION_DIGITS)


def _time_range(
    frame: pl.DataFrame,
    time_column: str,
) -> tuple[Any | None, Any | None] | None:
    if frame.height == 0:
        return None
    series = frame.get_column(time_column)
    return (series.min(), series.max())


def _check_time_chronology(
    *,
    train: pl.DataFrame,
    validation: pl.DataFrame,
    test: pl.DataFrame,
    time_column: str,
    has_validation: bool,
    warnings: list[str],
) -> None:
    if train.height == 0 or test.height == 0:
        raise ProcessIntelligenceError(
            "TIME split requires non-empty train and test partitions"
        )

    train_max = train.get_column(time_column).max()
    test_min = test.get_column(time_column).min()

    if has_validation:
        if validation.height == 0:
            raise ProcessIntelligenceError(
                "TIME split with validation requires a non-empty validation partition"
            )
        validation_min = validation.get_column(time_column).min()
        validation_max = validation.get_column(time_column).max()
        if train_max > validation_min:  # type: ignore[operator]
            raise ProcessIntelligenceError(
                "TIME split chronology violated: train time max exceeds "
                "validation time min"
            )
        if validation_max > test_min:  # type: ignore[operator]
            raise ProcessIntelligenceError(
                "TIME split chronology violated: validation time max exceeds "
                "test time min"
            )
        _maybe_add_boundary_warning(
            left=train,
            right=validation,
            time_column=time_column,
            warnings=warnings,
        )
        _maybe_add_boundary_warning(
            left=validation,
            right=test,
            time_column=time_column,
            warnings=warnings,
        )
    else:
        if train_max > test_min:  # type: ignore[operator]
            raise ProcessIntelligenceError(
                "TIME split chronology violated: train time max exceeds test time min"
            )
        _maybe_add_boundary_warning(
            left=train,
            right=test,
            time_column=time_column,
            warnings=warnings,
        )


def _maybe_add_boundary_warning(
    *,
    left: pl.DataFrame,
    right: pl.DataFrame,
    time_column: str,
    warnings: list[str],
) -> None:
    if left.height == 0 or right.height == 0:
        return
    left_last = left.get_column(time_column)[-1]
    right_first = right.get_column(time_column)[0]
    if left_last == right_first and _BOUNDARY_WARNING not in warnings:
        warnings.append(_BOUNDARY_WARNING)


def _build_summary(
    *,
    config: SplitConfig,
    train: pl.DataFrame,
    validation: pl.DataFrame,
    test: pl.DataFrame,
    train_groups: list[Any],
    validation_groups: list[Any],
    test_groups: list[Any],
    train_time_range: tuple[Any | None, Any | None] | None,
    validation_time_range: tuple[Any | None, Any | None] | None,
    test_time_range: tuple[Any | None, Any | None] | None,
    warnings: list[str],
) -> SplitSummary:
    total_rows = train.height + validation.height + test.height
    return SplitSummary(
        strategy=config.strategy,
        group_column=config.group_column,
        time_column=config.time_column,
        random_state=config.random_state,
        requested_test_size=config.test_size,
        requested_validation_size=config.validation_size,
        train_row_count=train.height,
        validation_row_count=validation.height,
        test_row_count=test.height,
        train_fraction=_fraction(train.height, total_rows),
        validation_fraction=_fraction(validation.height, total_rows),
        test_fraction=_fraction(test.height, total_rows),
        train_original_row_ids=_original_row_ids(train),
        validation_original_row_ids=_original_row_ids(validation),
        test_original_row_ids=_original_row_ids(test),
        train_groups=list(train_groups),
        validation_groups=list(validation_groups),
        test_groups=list(test_groups),
        train_time_range=train_time_range,
        validation_time_range=validation_time_range,
        test_time_range=test_time_range,
        warnings=list(warnings),
    )
