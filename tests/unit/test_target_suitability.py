"""Unit tests for target suitability evaluation (Step 11B.4)."""

from __future__ import annotations

import copy
import json

import polars as pl
import pytest

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.data import (
    TargetSuitabilityAssessment,
    TargetSuitabilityRefusalCode,
    evaluate_target_suitability,
)


def _frame(**columns: list[object]) -> pl.DataFrame:
    return pl.DataFrame(columns)


def test_numeric_varying_target_suitable() -> None:
    frame = _frame(y=[1.0, 2.0, 3.0], x=[0.1, 0.2, 0.3])
    result = evaluate_target_suitability(
        frame,
        "y",
        requested_task=AnalysisTask.REGRESSION,
    )
    assert result.suitable is True
    assert result.refusal_code is None
    assert result.unique_non_null_count == 3
    assert result.is_constant is False
    assert result.is_numeric is True


def test_integer_varying_target_suitable() -> None:
    frame = _frame(y=[1, 2, 2, 3], x=[10, 11, 12, 13])
    result = evaluate_target_suitability(frame, "y")
    assert result.suitable is True
    assert result.is_numeric is True
    assert result.unique_non_null_count == 3


def test_float_varying_target_suitable() -> None:
    frame = _frame(SOH=[0.9, 0.8, 0.7], sensor=[1.0, 2.0, 3.0])
    result = evaluate_target_suitability(
        frame,
        "SOH",
        requested_task=AnalysisTask.REGRESSION,
    )
    assert result.suitable is True
    assert result.target_column == "SOH"


def test_constant_zero_target_refused() -> None:
    frame = _frame(SOH=[0, 0, 0, 0], sensor=[1.0, 2.0, 3.0, 4.0])
    result = evaluate_target_suitability(
        frame,
        "SOH",
        requested_task=AnalysisTask.REGRESSION,
    )
    assert result.suitable is False
    assert result.refusal_code is TargetSuitabilityRefusalCode.TARGET_CONSTANT
    assert result.is_constant is True
    assert result.unique_non_null_count == 1
    assert result.constant_value == 0
    assert "constant" in result.message.lower()
    assert "0" in result.message


def test_constant_nonzero_target_refused() -> None:
    frame = _frame(y=[5, 5, 5], x=[1, 2, 3])
    result = evaluate_target_suitability(frame, "y")
    assert result.suitable is False
    assert result.refusal_code is TargetSuitabilityRefusalCode.TARGET_CONSTANT
    assert result.constant_value == 5


def test_constant_float_target_refused() -> None:
    frame = _frame(y=[0.95, 0.95, 0.95], x=[1.0, 2.0, 3.0])
    result = evaluate_target_suitability(frame, "y")
    assert result.suitable is False
    assert result.refusal_code is TargetSuitabilityRefusalCode.TARGET_CONSTANT
    assert result.constant_value == pytest.approx(0.95)


def test_all_null_target_refused() -> None:
    frame = pl.DataFrame(
        {
            "y": pl.Series("y", [None, None, None], dtype=pl.Float64),
            "x": [1.0, 2.0, 3.0],
        }
    )
    result = evaluate_target_suitability(frame, "y")
    assert result.suitable is False
    assert result.refusal_code is TargetSuitabilityRefusalCode.TARGET_ALL_NULL
    assert result.is_all_null is True
    assert result.non_null_count == 0
    assert result.unique_non_null_count == 0


def test_nonnumeric_explicit_regression_refused() -> None:
    frame = _frame(y=["a", "b", "a"], x=[1.0, 2.0, 3.0])
    result = evaluate_target_suitability(
        frame,
        "y",
        requested_task=AnalysisTask.REGRESSION,
    )
    assert result.suitable is False
    assert (
        result.refusal_code
        is TargetSuitabilityRefusalCode.TARGET_NON_NUMERIC_FOR_REGRESSION
    )
    assert result.is_numeric is False


def test_nulls_with_two_distinct_non_null_values_suitable() -> None:
    frame = pl.DataFrame(
        {
            "y": pl.Series("y", [1.0, None, 2.0, None], dtype=pl.Float64),
            "x": [0.1, 0.2, 0.3, 0.4],
        }
    )
    result = evaluate_target_suitability(
        frame,
        "y",
        requested_task=AnalysisTask.REGRESSION,
    )
    assert result.suitable is True
    assert result.unique_non_null_count == 2
    assert result.null_count == 2
    assert result.is_constant is False


def test_input_dataframe_immutable() -> None:
    frame = _frame(y=[0.0, 0.0, 0.0], x=[1.0, 2.0, 3.0])
    before = frame.to_dicts()
    columns_before = list(frame.columns)
    evaluate_target_suitability(frame, "y")
    assert frame.to_dicts() == before
    assert list(frame.columns) == columns_before


def test_deterministic_result() -> None:
    frame = _frame(y=[1.0, 1.0, 1.0], x=[1.0, 2.0, 3.0])
    first = evaluate_target_suitability(frame, "y")
    second = evaluate_target_suitability(frame, "y")
    assert first == second
    assert first.model_dump() == second.model_dump()


def test_scalar_metadata_json_safe() -> None:
    frame = _frame(y=[0, 0, 0], x=[1, 2, 3])
    result = evaluate_target_suitability(frame, "y")
    payload = result.model_dump(mode="json")
    serialized = json.dumps(payload)
    restored = json.loads(serialized)
    assert restored["refusal_code"] == "TARGET_CONSTANT"
    assert restored["constant_value"] == 0
    assert restored["suitable"] is False
    round_trip = TargetSuitabilityAssessment.model_validate(restored)
    assert round_trip == result


def test_auto_path_allows_nonnumeric_when_varying() -> None:
    frame = _frame(y=["pass", "fail", "pass"], x=[1.0, 2.0, 3.0])
    result = evaluate_target_suitability(frame, "y", requested_task=None)
    assert result.suitable is True
    assert result.is_numeric is False


def test_classification_requires_at_least_two_classes_semantics() -> None:
    frame = _frame(y=[1, 1, 1], x=[0.1, 0.2, 0.3])
    result = evaluate_target_suitability(
        frame,
        "y",
        requested_task=AnalysisTask.CLASSIFICATION,
    )
    assert result.suitable is False
    assert result.unique_non_null_count < 2
    assert result.refusal_code is TargetSuitabilityRefusalCode.TARGET_CONSTANT


def test_assessment_copy_independence() -> None:
    frame = _frame(y=[1.0, 2.0], x=[3.0, 4.0])
    result = evaluate_target_suitability(frame, "y")
    dumped = result.model_dump()
    dumped_copy = copy.deepcopy(dumped)
    dumped["message"] = "mutated"
    assert result.message != "mutated"
    assert dumped_copy["message"] == result.message
