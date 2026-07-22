"""Target suitability evaluation for supervised analysis safety gates (Step 11B.4).

Evaluates whether a selected target column has enough usable variation for
supervised modeling. Does not mutate the input DataFrame, invent synthetic
targets, or infer an analysis task from the data.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from process_intelligence.core.enums import AnalysisTask

ScalarMetadataValue = str | int | float | bool | None


class TargetSuitabilityRefusalCode(StrEnum):
    """Structured refusal codes for target suitability gates.

    Naming follows existing SCREAMING_SNAKE validation issue conventions.
    """

    TARGET_ALL_NULL = "TARGET_ALL_NULL"
    TARGET_CONSTANT = "TARGET_CONSTANT"
    TARGET_NON_NUMERIC_FOR_REGRESSION = "TARGET_NON_NUMERIC_FOR_REGRESSION"


class TargetSuitabilityAssessment(BaseModel):
    """Immutable assessment of whether a target column is suitable for modeling.

    ``suitable`` is False when supervised learning must be refused before split
    or model fitting. Classification currently requires at least two distinct
    non-null class labels; constant and all-null targets fail that requirement.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_column: str
    row_count: int = Field(ge=0)
    non_null_count: int = Field(ge=0)
    null_count: int = Field(ge=0)
    unique_non_null_count: int = Field(ge=0)
    is_all_null: bool
    is_constant: bool
    is_numeric: bool
    requested_task: AnalysisTask | None = None
    suitable: bool
    refusal_code: TargetSuitabilityRefusalCode | None = None
    message: str
    constant_value: ScalarMetadataValue = None

    @field_validator("target_column", "message", mode="before")
    @classmethod
    def _validate_non_empty_str(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(f"must be str, got {type(value).__name__}")
        if value == "" or value.strip() == "":
            raise ValueError("must be a non-empty, non-whitespace string")
        return value

    @field_validator(
        "is_all_null",
        "is_constant",
        "is_numeric",
        "suitable",
        mode="before",
    )
    @classmethod
    def _validate_bools(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError(f"must be a bool, got {type(value).__name__}")
        return value

    @field_validator("requested_task", mode="before")
    @classmethod
    def _validate_requested_task(cls, value: object) -> AnalysisTask | None:
        if value is None:
            return None
        if isinstance(value, AnalysisTask):
            return value
        if isinstance(value, str):
            try:
                return AnalysisTask(value)
            except ValueError as exc:
                raise ValueError(f"invalid AnalysisTask: {value!r}") from exc
        raise ValueError(
            f"requested_task must be AnalysisTask or None, got {type(value).__name__}"
        )

    @field_validator("refusal_code", mode="before")
    @classmethod
    def _validate_refusal_code(
        cls,
        value: object,
    ) -> TargetSuitabilityRefusalCode | None:
        if value is None:
            return None
        if isinstance(value, TargetSuitabilityRefusalCode):
            return value
        if isinstance(value, str):
            try:
                return TargetSuitabilityRefusalCode(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid TargetSuitabilityRefusalCode: {value!r}"
                ) from exc
        raise ValueError(
            "refusal_code must be TargetSuitabilityRefusalCode or None, "
            f"got {type(value).__name__}"
        )

    @field_validator("constant_value", mode="before")
    @classmethod
    def _validate_constant_value(cls, value: object) -> ScalarMetadataValue:
        if value is None or isinstance(value, (str, bool)):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(
                    f"constant_value float must be finite, got {value!r}"
                )
            return value
        raise ValueError(
            "constant_value must be str, int, float, bool, or None, "
            f"got {type(value).__name__}"
        )

    @model_validator(mode="after")
    def _validate_consistency(self) -> TargetSuitabilityAssessment:
        if self.null_count + self.non_null_count != self.row_count:
            raise ValueError(
                "null_count + non_null_count must equal row_count "
                f"({self.null_count} + {self.non_null_count} != {self.row_count})"
            )
        if self.is_all_null != (self.non_null_count == 0 and self.row_count > 0):
            raise ValueError(
                "is_all_null must equal (non_null_count == 0 and row_count > 0)"
            )
        if self.is_constant != (
            self.non_null_count > 0 and self.unique_non_null_count == 1
        ):
            raise ValueError(
                "is_constant must equal "
                "(non_null_count > 0 and unique_non_null_count == 1)"
            )
        if self.suitable and self.refusal_code is not None:
            raise ValueError("refusal_code must be None when suitable=True")
        if not self.suitable and self.refusal_code is None:
            raise ValueError("refusal_code is required when suitable=False")
        if self.constant_value is not None and not self.is_constant:
            raise ValueError("constant_value is only allowed when is_constant=True")
        return self


def _to_json_safe_scalar(value: Any) -> ScalarMetadataValue:
    """Convert a single cell value to a JSON-safe scalar when possible."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value) if isinstance(value, float) else int(value)
        if isinstance(number, float) and not math.isfinite(number):
            return None
        if isinstance(value, float):
            return float(value)
        return int(value)
    if isinstance(value, str):
        return value
    # NumPy / Polars scalar wrappers.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _to_json_safe_scalar(item())
        except (TypeError, ValueError):
            return None
    try:
        if math.isfinite(float(value)) and not isinstance(value, (str, bytes)):
            as_float = float(value)
            if as_float.is_integer():
                return int(as_float)
            return as_float
    except (TypeError, ValueError):
        return None
    return None


def evaluate_target_suitability(
    frame: pl.DataFrame,
    target_column: str,
    *,
    requested_task: AnalysisTask | None = None,
) -> TargetSuitabilityAssessment:
    """Evaluate whether ``target_column`` is suitable for supervised analysis.

    The input frame is never modified. Constant and all-null targets are always
    refused for supervised paths. Explicit REGRESSION additionally requires a
    numeric dtype. Classification suitability still requires at least two
    distinct non-null labels (covered by the constant / all-null gates).

    Args:
        frame: Loaded dataset frame.
        target_column: Selected target column name.
        requested_task: Caller-selected task, or ``None`` for AUTO routing.

    Returns:
        An immutable ``TargetSuitabilityAssessment``.

    Raises:
        TypeError: If ``frame`` is not a Polars DataFrame.
        ValueError: If ``target_column`` is empty or missing from ``frame``.
    """
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(
            f"frame must be a polars DataFrame, got {type(frame).__name__}"
        )
    if not isinstance(target_column, str) or target_column.strip() == "":
        raise ValueError("target_column must be a non-empty, non-whitespace string")
    if target_column not in frame.columns:
        raise ValueError(
            f"target_column {target_column!r} was not found in the dataset columns"
        )
    if requested_task is not None and not isinstance(requested_task, AnalysisTask):
        raise TypeError(
            "requested_task must be AnalysisTask or None, "
            f"got {type(requested_task).__name__}"
        )

    series = frame.get_column(target_column)
    row_count = int(frame.height)
    null_count = int(series.null_count())
    non_null = series.drop_nulls()
    non_null_count = int(non_null.len())
    unique_non_null_count = int(non_null.n_unique()) if non_null_count > 0 else 0
    is_all_null = row_count > 0 and non_null_count == 0
    is_constant = non_null_count > 0 and unique_non_null_count == 1
    is_numeric = bool(series.dtype.is_numeric()) and series.dtype != pl.Boolean

    constant_value: ScalarMetadataValue = None
    if is_constant:
        constant_value = _to_json_safe_scalar(non_null[0])

    refusal_code: TargetSuitabilityRefusalCode | None = None
    message: str
    suitable: bool

    if is_all_null:
        refusal_code = TargetSuitabilityRefusalCode.TARGET_ALL_NULL
        suitable = False
        message = (
            f"Target column '{target_column}' is entirely null "
            f"({null_count} null values across {row_count} rows). "
            "Supervised analysis requires at least one non-null target value "
            "and at least two distinct non-null values."
        )
    elif is_constant:
        refusal_code = TargetSuitabilityRefusalCode.TARGET_CONSTANT
        suitable = False
        if constant_value is None:
            message = (
                f"Target column '{target_column}' is constant: all "
                f"{non_null_count} non-null values are identical. "
                "Regression requires at least two distinct target values."
            )
        else:
            message = (
                f"Target column '{target_column}' is constant: all "
                f"{non_null_count} non-null values are {constant_value}. "
                "Regression requires at least two distinct target values."
            )
    elif (
        requested_task is AnalysisTask.REGRESSION
        and not is_numeric
    ):
        refusal_code = TargetSuitabilityRefusalCode.TARGET_NON_NUMERIC_FOR_REGRESSION
        suitable = False
        message = (
            f"Explicit REGRESSION requires a numeric target column, but "
            f"'{target_column}' has dtype '{series.dtype}'. "
            "Target values were not modified."
        )
    elif (
        requested_task is AnalysisTask.CLASSIFICATION
        and unique_non_null_count < 2
    ):
        # Defensive branch; all-null / constant already cover unique < 2.
        refusal_code = TargetSuitabilityRefusalCode.TARGET_CONSTANT
        suitable = False
        message = (
            f"Target column '{target_column}' does not provide at least two "
            "distinct non-null class labels required for classification."
        )
    else:
        suitable = True
        message = (
            f"Target column '{target_column}' has usable variation for the "
            "requested supervised analysis path."
        )

    return TargetSuitabilityAssessment(
        target_column=target_column,
        row_count=row_count,
        non_null_count=non_null_count,
        null_count=null_count,
        unique_non_null_count=unique_non_null_count,
        is_all_null=is_all_null,
        is_constant=is_constant,
        is_numeric=is_numeric,
        requested_task=requested_task,
        suitable=suitable,
        refusal_code=refusal_code,
        message=message,
        constant_value=constant_value,
    )
