"""Unit tests for regression and classification metrics (Step 6B)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import pytest
from pydantic import ValidationError

from process_intelligence.core.exceptions import DataValidationError, InsufficientDataError
from process_intelligence.evaluation import (
    ClassificationMetrics,
    RegressionMetrics,
    evaluate_classification,
    evaluate_regression,
)


def test_regression_metrics_valid_creation() -> None:
    metrics = RegressionMetrics(sample_count=3, mae=0.1, rmse=0.2, r2=0.9)
    assert metrics.sample_count == 3
    assert metrics.warnings == []


def test_regression_metrics_rejects_zero_sample_count() -> None:
    with pytest.raises(ValidationError):
        RegressionMetrics(sample_count=0, mae=0.0, rmse=0.0, r2=None)


def test_regression_metrics_rejects_negative_mae() -> None:
    with pytest.raises(ValidationError):
        RegressionMetrics(sample_count=1, mae=-0.1, rmse=0.0, r2=None)


def test_regression_metrics_rejects_negative_rmse() -> None:
    with pytest.raises(ValidationError):
        RegressionMetrics(sample_count=1, mae=0.0, rmse=-0.1, r2=None)


def test_regression_metrics_rejects_nan() -> None:
    with pytest.raises(ValidationError):
        RegressionMetrics(sample_count=1, mae=float("nan"), rmse=0.0, r2=None)


def test_regression_metrics_rejects_infinity() -> None:
    with pytest.raises(ValidationError):
        RegressionMetrics(sample_count=1, mae=0.0, rmse=float("inf"), r2=None)


def test_regression_metrics_rejects_duplicate_warnings() -> None:
    with pytest.raises(ValidationError):
        RegressionMetrics(
            sample_count=1,
            mae=0.0,
            rmse=0.0,
            r2=None,
            warnings=["a", "a"],
        )


def test_regression_metrics_mutable_default_independence() -> None:
    first = RegressionMetrics(sample_count=1, mae=0.0, rmse=0.0, r2=None)
    second = RegressionMetrics(sample_count=1, mae=0.0, rmse=0.0, r2=None)
    first.warnings.append("only-first")
    assert second.warnings == []


def test_regression_metrics_round_trip() -> None:
    original = RegressionMetrics(
        sample_count=2,
        mae=1.0,
        rmse=1.5,
        r2=0.5,
        warnings=["note"],
    )
    restored = RegressionMetrics.model_validate(original.model_dump())
    assert restored == original


def test_classification_metrics_valid_creation() -> None:
    metrics = ClassificationMetrics(
        sample_count=4,
        class_count=2,
        accuracy=1.0,
        balanced_accuracy=1.0,
        precision_macro=1.0,
        recall_macro=1.0,
        f1_macro=1.0,
    )
    assert metrics.class_count == 2


def test_classification_metrics_rejects_zero_sample_count() -> None:
    with pytest.raises(ValidationError):
        ClassificationMetrics(
            sample_count=0,
            class_count=1,
            accuracy=0.0,
            balanced_accuracy=0.0,
            precision_macro=0.0,
            recall_macro=0.0,
            f1_macro=0.0,
        )


def test_classification_metrics_rejects_zero_class_count() -> None:
    with pytest.raises(ValidationError):
        ClassificationMetrics(
            sample_count=1,
            class_count=0,
            accuracy=0.0,
            balanced_accuracy=0.0,
            precision_macro=0.0,
            recall_macro=0.0,
            f1_macro=0.0,
        )


def test_classification_metrics_rejects_below_zero() -> None:
    with pytest.raises(ValidationError):
        ClassificationMetrics(
            sample_count=1,
            class_count=1,
            accuracy=-0.01,
            balanced_accuracy=0.0,
            precision_macro=0.0,
            recall_macro=0.0,
            f1_macro=0.0,
        )


def test_classification_metrics_rejects_above_one() -> None:
    with pytest.raises(ValidationError):
        ClassificationMetrics(
            sample_count=1,
            class_count=1,
            accuracy=1.01,
            balanced_accuracy=0.0,
            precision_macro=0.0,
            recall_macro=0.0,
            f1_macro=0.0,
        )


def test_classification_metrics_rejects_nan() -> None:
    with pytest.raises(ValidationError):
        ClassificationMetrics(
            sample_count=1,
            class_count=1,
            accuracy=float("nan"),
            balanced_accuracy=0.0,
            precision_macro=0.0,
            recall_macro=0.0,
            f1_macro=0.0,
        )


def test_classification_metrics_rejects_infinity() -> None:
    with pytest.raises(ValidationError):
        ClassificationMetrics(
            sample_count=1,
            class_count=1,
            accuracy=0.0,
            balanced_accuracy=float("inf"),
            precision_macro=0.0,
            recall_macro=0.0,
            f1_macro=0.0,
        )


def test_classification_metrics_rejects_duplicate_warnings() -> None:
    with pytest.raises(ValidationError):
        ClassificationMetrics(
            sample_count=1,
            class_count=1,
            accuracy=0.0,
            balanced_accuracy=0.0,
            precision_macro=0.0,
            recall_macro=0.0,
            f1_macro=0.0,
            warnings=["a", "a"],
        )


def test_classification_metrics_round_trip() -> None:
    original = ClassificationMetrics(
        sample_count=3,
        class_count=2,
        accuracy=0.5,
        balanced_accuracy=0.5,
        precision_macro=0.4,
        recall_macro=0.4,
        f1_macro=0.4,
        warnings=["note"],
    )
    restored = ClassificationMetrics.model_validate(original.model_dump())
    assert restored == original


def test_evaluate_regression_perfect_prediction() -> None:
    y = [1.0, 2.0, 3.0]
    metrics = evaluate_regression(y, y)
    assert metrics.mae == 0.0
    assert metrics.rmse == 0.0
    assert metrics.r2 == 1.0
    assert metrics.sample_count == 3


def test_evaluate_regression_mae_rmse_r2() -> None:
    y_true = [0.0, 1.0, 2.0]
    y_pred = [0.0, 1.0, 4.0]
    metrics = evaluate_regression(y_true, y_pred)
    assert metrics.mae == pytest.approx(2.0 / 3.0)
    assert metrics.rmse == pytest.approx(math.sqrt((0.0 + 0.0 + 4.0) / 3.0))
    assert metrics.r2 == pytest.approx(1.0 - (4.0 / 2.0))


def test_evaluate_regression_single_sample_r2_none_and_warning() -> None:
    metrics = evaluate_regression([1.0], [1.5])
    assert metrics.r2 is None
    assert metrics.sample_count == 1
    assert any("2" in warning or "R2" in warning or "r2" in warning.lower()
               for warning in metrics.warnings)


def test_evaluate_regression_accepts_polars_pandas_numpy_sequence() -> None:
    values = [1.0, 2.0, 3.0]
    for true_values, pred_values in (
        (pl.Series(values), pl.Series(values)),
        (pd.Series(values), pd.Series(values)),
        (np.asarray(values), np.asarray(values)),
        (values, values),
    ):
        metrics = evaluate_regression(true_values, pred_values)
        assert metrics.mae == 0.0


def test_evaluate_regression_rejects_invalid_inputs() -> None:
    with pytest.raises(TypeError):
        evaluate_regression(object(), [1.0])
    with pytest.raises(TypeError):
        evaluate_regression("abc", "abc")
    with pytest.raises(TypeError):
        evaluate_regression(b"abc", b"abc")
    with pytest.raises(DataValidationError):
        evaluate_regression(np.asarray([[1.0, 2.0]]), np.asarray([[1.0, 2.0]]))
    with pytest.raises(InsufficientDataError):
        evaluate_regression([], [])
    with pytest.raises(DataValidationError):
        evaluate_regression([1.0, 2.0], [1.0])


def test_evaluate_regression_rejects_non_numeric_and_non_finite() -> None:
    with pytest.raises(DataValidationError):
        evaluate_regression(["a", "b"], ["a", "b"])
    with pytest.raises(DataValidationError):
        evaluate_regression([True, False], [True, False])
    with pytest.raises(DataValidationError):
        evaluate_regression(pl.Series([1.0, None]), pl.Series([1.0, 2.0]))
    with pytest.raises(DataValidationError):
        evaluate_regression([1.0, float("nan")], [1.0, 2.0])
    with pytest.raises(DataValidationError):
        evaluate_regression([1.0, float("inf")], [1.0, 2.0])
    with pytest.raises(DataValidationError):
        evaluate_regression([1.0, float("-inf")], [1.0, 2.0])


def test_evaluate_regression_input_immutability() -> None:
    y_true = pl.Series([1.0, 2.0, 3.0])
    y_pred = pl.Series([1.0, 2.0, 4.0])
    before_true = y_true.to_list()
    before_pred = y_pred.to_list()
    evaluate_regression(y_true, y_pred)
    assert y_true.to_list() == before_true
    assert y_pred.to_list() == before_pred


def test_evaluate_classification_perfect_and_metrics() -> None:
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    metrics = evaluate_classification(y_true, y_pred)
    assert metrics.accuracy == 1.0
    assert metrics.balanced_accuracy == 1.0
    assert metrics.precision_macro == 1.0
    assert metrics.recall_macro == 1.0
    assert metrics.f1_macro == 1.0
    assert metrics.class_count == 2
    assert metrics.sample_count == 4


def test_evaluate_classification_metric_accuracy() -> None:
    metrics = evaluate_classification([0, 1, 0, 1], [0, 1, 1, 1])
    assert metrics.accuracy == pytest.approx(0.75)


def test_evaluate_classification_accepts_label_types_and_frames() -> None:
    assert evaluate_classification(["a", "b", "a"], ["a", "b", "a"]).accuracy == 1.0
    assert evaluate_classification([True, False, True], [True, False, True]).accuracy == 1.0
    assert evaluate_classification([0, 1, 0], [0, 1, 0]).accuracy == 1.0
    values = [0, 1, 0]
    for true_values, pred_values in (
        (pl.Series(values), pl.Series(values)),
        (pd.Series(values), pd.Series(values)),
        (np.asarray(values), np.asarray(values)),
        (values, values),
    ):
        assert evaluate_classification(true_values, pred_values).accuracy == 1.0


def test_evaluate_classification_rejects_invalid_inputs() -> None:
    with pytest.raises(TypeError):
        evaluate_classification(object(), [0])
    with pytest.raises(DataValidationError):
        evaluate_classification(np.asarray([[0, 1]]), np.asarray([[0, 1]]))
    with pytest.raises(InsufficientDataError):
        evaluate_classification([], [])
    with pytest.raises(DataValidationError):
        evaluate_classification([0, 1], [0])
    with pytest.raises(DataValidationError):
        evaluate_classification(pl.Series([0, None]), pl.Series([0, 1]))
    with pytest.raises(DataValidationError):
        evaluate_classification([0.0, float("nan")], [0.0, 1.0])
    with pytest.raises(DataValidationError):
        evaluate_classification([0.0, float("inf")], [0.0, 1.0])


def test_evaluate_classification_class_mismatch_warnings() -> None:
    missing_pred = evaluate_classification([0, 1, 0, 1], [0, 0, 0, 0])
    assert any("missing" in warning for warning in missing_pred.warnings)

    new_pred = evaluate_classification([0, 0, 0, 0], [0, 1, 0, 1])
    assert any("not present" in warning for warning in new_pred.warnings)


def test_evaluate_classification_zero_division_finite() -> None:
    metrics = evaluate_classification([0, 0, 0, 0], [0, 0, 0, 0])
    for value in (
        metrics.accuracy,
        metrics.balanced_accuracy,
        metrics.precision_macro,
        metrics.recall_macro,
        metrics.f1_macro,
    ):
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0


def test_evaluate_classification_input_immutability() -> None:
    y_true = pd.Series(["a", "b", "a"])
    y_pred = pd.Series(["a", "b", "b"])
    before_true = y_true.tolist()
    before_pred = y_pred.tolist()
    evaluate_classification(y_true, y_pred)
    assert y_true.tolist() == before_true
    assert y_pred.tolist() == before_pred
