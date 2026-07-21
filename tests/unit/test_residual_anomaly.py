"""Unit tests for residual anomaly detection (Step 7F)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import pytest
from pydantic import ValidationError

from process_intelligence.core.exceptions import (
    DataValidationError,
    InsufficientDataError,
    ProcessIntelligenceError,
)
from process_intelligence.models import (
    ResidualAnomalyConfig,
    ResidualAnomalyDetector,
    ResidualAnomalyResult,
    ResidualCalibration,
    ResidualThresholdMethod,
)
from process_intelligence.models.residual_anomaly import (
    _ALL_ANOMALY_WARNING,
    _ALL_NORMAL_WARNING,
    _CONSTANT_SCORE_WARNING,
    _IDENTICAL_CALIBRATION_SCORE_WARNING,
    _MAD_SCALE_CONSTANT,
    _MINIMUM_SCALE_WARNING,
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _mad_calibration(**overrides: Any) -> ResidualCalibration:
    payload: dict[str, Any] = {
        "method": ResidualThresholdMethod.MAD,
        "row_count": 4,
        "residual_center": 0.0,
        "residual_scale": 1.5,
        "threshold": 3.5,
        "mad_multiplier": 3.5,
        "quantile": None,
        "center_residuals": True,
        "threshold_inclusive": False,
        "calibration_score_min": 0.0,
        "calibration_score_max": 2.0,
        "calibration_score_mean": 1.0,
        "calibration_score_std": 0.5,
        "fitted_at": _utc_now(),
        "warnings": [],
    }
    payload.update(overrides)
    return ResidualCalibration(**payload)


def _quantile_calibration(**overrides: Any) -> ResidualCalibration:
    payload: dict[str, Any] = {
        "method": ResidualThresholdMethod.QUANTILE,
        "row_count": 4,
        "residual_center": 0.0,
        "residual_scale": None,
        "threshold": 1.2,
        "mad_multiplier": None,
        "quantile": 0.99,
        "center_residuals": True,
        "threshold_inclusive": False,
        "calibration_score_min": 0.0,
        "calibration_score_max": 2.0,
        "calibration_score_mean": 1.0,
        "calibration_score_std": 0.5,
        "fitted_at": _utc_now(),
        "warnings": [],
    }
    payload.update(overrides)
    return ResidualCalibration(**payload)


def _result(**overrides: Any) -> ResidualAnomalyResult:
    payload: dict[str, Any] = {
        "predictions": [1.0, 2.0],
        "residuals": [0.5, -0.5],
        "absolute_centered_residuals": [0.5, 0.5],
        "scores": [1.0, 1.0],
        "is_anomaly": [False, True],
        "raw_predictions": [1, -1],
        "threshold": 0.75,
        "row_count": 2,
        "anomaly_count": 1,
        "anomaly_fraction": 0.5,
        "residual_mean": 0.0,
        "residual_std": 0.5,
        "score_min": 1.0,
        "score_max": 1.0,
        "score_mean": 1.0,
        "score_std": 0.0,
        "warnings": [],
    }
    payload.update(overrides)
    return ResidualAnomalyResult(**payload)


def _empty_result(**overrides: Any) -> ResidualAnomalyResult:
    payload: dict[str, Any] = {
        "predictions": [],
        "residuals": [],
        "absolute_centered_residuals": [],
        "scores": [],
        "is_anomaly": [],
        "raw_predictions": [],
        "threshold": 3.5,
        "row_count": 0,
        "anomaly_count": 0,
        "anomaly_fraction": 0.0,
        "residual_mean": None,
        "residual_std": None,
        "score_min": None,
        "score_max": None,
        "score_mean": None,
        "score_std": None,
        "warnings": [],
    }
    payload.update(overrides)
    return ResidualAnomalyResult(**payload)


# ---------------------------------------------------------------------------
# ResidualThresholdMethod
# ---------------------------------------------------------------------------


def test_threshold_method_mad_value() -> None:
    assert ResidualThresholdMethod.MAD.value == "MAD"


def test_threshold_method_quantile_value() -> None:
    assert ResidualThresholdMethod.QUANTILE.value == "QUANTILE"


def test_threshold_method_from_string() -> None:
    assert ResidualThresholdMethod("MAD") is ResidualThresholdMethod.MAD
    assert ResidualThresholdMethod("QUANTILE") is ResidualThresholdMethod.QUANTILE


def test_threshold_method_invalid_string() -> None:
    with pytest.raises(ValueError):
        ResidualThresholdMethod("INVALID")


def test_threshold_method_no_extra_members() -> None:
    assert set(ResidualThresholdMethod) == {
        ResidualThresholdMethod.MAD,
        ResidualThresholdMethod.QUANTILE,
    }


# ---------------------------------------------------------------------------
# ResidualAnomalyConfig
# ---------------------------------------------------------------------------


def test_config_default_creation() -> None:
    config = ResidualAnomalyConfig()
    assert config.method is ResidualThresholdMethod.MAD
    assert config.mad_multiplier == pytest.approx(3.5)
    assert config.quantile == pytest.approx(0.99)
    assert config.minimum_scale == pytest.approx(1e-12)
    assert config.center_residuals is True
    assert config.threshold_inclusive is False


def test_config_rejects_invalid_method() -> None:
    with pytest.raises(ValidationError):
        ResidualAnomalyConfig(method="NOT_A_METHOD")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mad_multiplier", 0.0),
        ("mad_multiplier", -1.0),
        ("mad_multiplier", True),
        ("mad_multiplier", float("nan")),
        ("mad_multiplier", float("inf")),
        ("quantile", 0.0),
        ("quantile", 1.0),
        ("quantile", -0.1),
        ("quantile", 1.1),
        ("quantile", True),
        ("quantile", float("nan")),
        ("minimum_scale", 0.0),
        ("minimum_scale", -1e-6),
        ("minimum_scale", False),
    ],
)
def test_config_rejects_invalid_numeric_fields(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        ResidualAnomalyConfig(**{field: value})


@pytest.mark.parametrize("field", ["center_residuals", "threshold_inclusive"])
@pytest.mark.parametrize("value", [0, 1, "true", "false", None])
def test_config_rejects_non_bool_flags(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        ResidualAnomalyConfig(**{field: value})


def test_config_round_trip() -> None:
    original = ResidualAnomalyConfig(
        method=ResidualThresholdMethod.QUANTILE,
        mad_multiplier=2.5,
        quantile=0.95,
        minimum_scale=1e-8,
        center_residuals=False,
        threshold_inclusive=True,
    )
    restored = ResidualAnomalyConfig.model_validate(original.model_dump())
    assert restored == original


# ---------------------------------------------------------------------------
# ResidualCalibration
# ---------------------------------------------------------------------------


def test_calibration_mad_valid() -> None:
    calibration = _mad_calibration()
    assert calibration.method is ResidualThresholdMethod.MAD
    assert calibration.residual_scale == pytest.approx(1.5)
    assert calibration.quantile is None


def test_calibration_quantile_valid() -> None:
    calibration = _quantile_calibration()
    assert calibration.method is ResidualThresholdMethod.QUANTILE
    assert calibration.residual_scale is None
    assert calibration.mad_multiplier is None


def test_calibration_rejects_row_count_zero() -> None:
    with pytest.raises(ValidationError):
        _mad_calibration(row_count=0)


def test_calibration_rejects_residual_center_nan() -> None:
    with pytest.raises(ValidationError):
        _mad_calibration(residual_center=float("nan"))


def test_calibration_rejects_negative_threshold() -> None:
    with pytest.raises(ValidationError):
        _mad_calibration(threshold=-0.1, mad_multiplier=-0.1)


def test_calibration_rejects_threshold_nan() -> None:
    with pytest.raises(ValidationError):
        _mad_calibration(threshold=float("nan"), mad_multiplier=float("nan"))


def test_calibration_rejects_residual_scale_zero() -> None:
    with pytest.raises(ValidationError):
        _mad_calibration(residual_scale=0.0)


def test_calibration_mad_requires_residual_scale() -> None:
    with pytest.raises(ValidationError):
        _mad_calibration(residual_scale=None)


def test_calibration_mad_requires_multiplier() -> None:
    with pytest.raises(ValidationError):
        _mad_calibration(mad_multiplier=None)


def test_calibration_mad_rejects_quantile() -> None:
    with pytest.raises(ValidationError):
        _mad_calibration(quantile=0.99)


def test_calibration_mad_threshold_must_match_multiplier() -> None:
    with pytest.raises(ValidationError):
        _mad_calibration(threshold=3.5, mad_multiplier=2.0)


def test_calibration_quantile_rejects_residual_scale() -> None:
    with pytest.raises(ValidationError):
        _quantile_calibration(residual_scale=1.0)


def test_calibration_quantile_requires_quantile() -> None:
    with pytest.raises(ValidationError):
        _quantile_calibration(quantile=None)


def test_calibration_quantile_rejects_multiplier() -> None:
    with pytest.raises(ValidationError):
        _quantile_calibration(mad_multiplier=3.5)


def test_calibration_rejects_score_summary_order() -> None:
    with pytest.raises(ValidationError):
        _mad_calibration(
            calibration_score_min=2.0,
            calibration_score_mean=1.0,
            calibration_score_max=0.0,
        )


def test_calibration_rejects_negative_score_std() -> None:
    with pytest.raises(ValidationError):
        _mad_calibration(calibration_score_std=-0.1)


def test_calibration_rejects_naive_fitted_at() -> None:
    with pytest.raises(ValidationError):
        _mad_calibration(fitted_at=datetime(2024, 1, 1, 12, 0, 0))


def test_calibration_rejects_duplicate_warnings() -> None:
    with pytest.raises(ValidationError):
        _mad_calibration(warnings=["a", "a"])


def test_calibration_round_trip() -> None:
    original = _mad_calibration(warnings=["note"])
    restored = ResidualCalibration.model_validate(original.model_dump())
    assert restored == original


# ---------------------------------------------------------------------------
# ResidualAnomalyResult
# ---------------------------------------------------------------------------


def test_result_valid_creation() -> None:
    result = _result()
    assert result.row_count == 2
    assert result.anomaly_count == 1


def test_empty_result_valid_creation() -> None:
    result = _empty_result()
    assert result.row_count == 0
    assert result.anomaly_fraction == 0.0
    assert result.residual_mean is None


def test_result_rejects_negative_row_count() -> None:
    with pytest.raises(ValidationError):
        _result(row_count=-1)


def test_result_rejects_negative_anomaly_count() -> None:
    with pytest.raises(ValidationError):
        _result(anomaly_count=-1)


def test_result_rejects_anomaly_count_gt_row_count() -> None:
    with pytest.raises(ValidationError):
        _result(anomaly_count=3)


def test_result_rejects_collection_length_mismatch() -> None:
    with pytest.raises(ValidationError):
        _result(predictions=[1.0])


def test_result_rejects_prediction_nan() -> None:
    with pytest.raises(ValidationError):
        _result(predictions=[float("nan"), 2.0])


def test_result_rejects_residual_infinity() -> None:
    with pytest.raises(ValidationError):
        _result(residuals=[float("inf"), -0.5])


def test_result_rejects_negative_centered_residual() -> None:
    with pytest.raises(ValidationError):
        _result(absolute_centered_residuals=[-0.1, 0.5])


def test_result_rejects_negative_score() -> None:
    with pytest.raises(ValidationError):
        _result(scores=[-1.0, 1.0])


def test_result_rejects_score_nan() -> None:
    with pytest.raises(ValidationError):
        _result(scores=[float("nan"), 1.0])


def test_result_rejects_invalid_raw_prediction() -> None:
    with pytest.raises(ValidationError):
        _result(raw_predictions=[0, -1], is_anomaly=[False, True], anomaly_count=1)


def test_result_rejects_anomaly_flag_mismatch() -> None:
    with pytest.raises(ValidationError):
        _result(is_anomaly=[True, True], raw_predictions=[1, -1], anomaly_count=2)


def test_result_rejects_anomaly_count_mismatch() -> None:
    with pytest.raises(ValidationError):
        _result(anomaly_count=0, anomaly_fraction=0.0)


def test_result_rejects_anomaly_fraction_mismatch() -> None:
    with pytest.raises(ValidationError):
        _result(anomaly_fraction=0.25)


def test_empty_result_rejects_summary() -> None:
    with pytest.raises(ValidationError):
        _empty_result(residual_mean=0.0)


def test_result_rejects_missing_summary() -> None:
    with pytest.raises(ValidationError):
        _result(score_mean=None)


def test_result_rejects_score_order() -> None:
    with pytest.raises(ValidationError):
        _result(score_min=2.0, score_mean=1.0, score_max=0.0, scores=[2.0, 0.0])


def test_result_rejects_negative_residual_std() -> None:
    with pytest.raises(ValidationError):
        _result(residual_std=-0.1)


def test_result_rejects_negative_score_std() -> None:
    with pytest.raises(ValidationError):
        _result(score_std=-0.1)


def test_result_rejects_duplicate_warnings() -> None:
    with pytest.raises(ValidationError):
        _result(warnings=["a", "a"])


def test_result_mutable_default_independence() -> None:
    first = _empty_result()
    second = _empty_result()
    first.warnings.append("only-first")
    assert second.warnings == []


def test_result_round_trip() -> None:
    original = _result(warnings=["note"])
    restored = ResidualAnomalyResult.model_validate(original.model_dump())
    assert restored == original


# ---------------------------------------------------------------------------
# Detector construction
# ---------------------------------------------------------------------------


def test_detector_default_creation() -> None:
    detector = ResidualAnomalyDetector()
    assert detector.is_fitted is False
    assert detector.calibration is None
    assert detector.threshold is None
    assert detector.fitted_at is None


def test_detector_custom_config() -> None:
    config = ResidualAnomalyConfig(method=ResidualThresholdMethod.QUANTILE, quantile=0.9)
    detector = ResidualAnomalyDetector(config=config)
    metadata = detector.get_metadata()
    assert metadata["method"] == "QUANTILE"
    assert metadata["quantile"] == pytest.approx(0.9)


def test_detector_rejects_invalid_config_type() -> None:
    with pytest.raises(TypeError):
        ResidualAnomalyDetector(config={"method": "MAD"})  # type: ignore[arg-type]


def test_detector_config_isolation() -> None:
    config = ResidualAnomalyConfig(mad_multiplier=2.0)
    detector = ResidualAnomalyDetector(config=config)
    config.mad_multiplier = 9.0
    assert detector.get_metadata()["mad_multiplier"] == pytest.approx(2.0)


def test_detector_instance_state_isolation() -> None:
    first = ResidualAnomalyDetector()
    second = ResidualAnomalyDetector()
    first.fit([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert first.is_fitted is True
    assert second.is_fitted is False
    assert second.calibration is None


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_fit_accepts_polars_series() -> None:
    detector = ResidualAnomalyDetector()
    detector.fit(pl.Series([1.0, 2.0, 3.0]), pl.Series([0.9, 2.1, 2.8]))
    assert detector.is_fitted is True


def test_fit_accepts_pandas_series() -> None:
    detector = ResidualAnomalyDetector()
    detector.fit(pd.Series([1.0, 2.0, 3.0]), pd.Series([0.9, 2.1, 2.8]))
    assert detector.is_fitted is True


def test_fit_accepts_numpy_1d() -> None:
    detector = ResidualAnomalyDetector()
    detector.fit(np.array([1.0, 2.0, 3.0]), np.array([0.9, 2.1, 2.8]))
    assert detector.is_fitted is True


def test_fit_accepts_sequence() -> None:
    detector = ResidualAnomalyDetector()
    detector.fit([1.0, 2.0, 3.0], (0.9, 2.1, 2.8))
    assert detector.is_fitted is True


@pytest.mark.parametrize(
    "bad",
    [
        "1,2,3",
        b"123",
        pl.DataFrame({"a": [1.0, 2.0]}),
        pd.DataFrame({"a": [1.0, 2.0]}),
        1.0,
        {"a": 1.0},
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    ],
)
def test_fit_rejects_unsupported_inputs(bad: Any) -> None:
    detector = ResidualAnomalyDetector()
    with pytest.raises((TypeError, DataValidationError)):
        detector.fit(bad, [1.0, 2.0])


def test_fit_rejects_length_mismatch() -> None:
    detector = ResidualAnomalyDetector()
    with pytest.raises(DataValidationError):
        detector.fit([1.0, 2.0], [1.0])


def test_fit_rejects_empty() -> None:
    detector = ResidualAnomalyDetector()
    with pytest.raises(InsufficientDataError):
        detector.fit([], [])


def test_fit_rejects_boolean() -> None:
    detector = ResidualAnomalyDetector()
    with pytest.raises(DataValidationError):
        detector.fit([True, False, True], [False, True, False])


def test_fit_rejects_string_numbers() -> None:
    detector = ResidualAnomalyDetector()
    with pytest.raises(DataValidationError):
        detector.fit(["1.0", "2.0"], ["1.0", "2.0"])


def test_fit_rejects_null() -> None:
    detector = ResidualAnomalyDetector()
    with pytest.raises(DataValidationError):
        detector.fit(pl.Series([1.0, None, 3.0]), pl.Series([1.0, 2.0, 3.0]))


def test_fit_rejects_nan() -> None:
    detector = ResidualAnomalyDetector()
    with pytest.raises(DataValidationError):
        detector.fit([1.0, float("nan")], [1.0, 2.0])


def test_fit_rejects_positive_infinity() -> None:
    detector = ResidualAnomalyDetector()
    with pytest.raises(DataValidationError):
        detector.fit([1.0, float("inf")], [1.0, 2.0])


def test_fit_rejects_negative_infinity() -> None:
    detector = ResidualAnomalyDetector()
    with pytest.raises(DataValidationError):
        detector.fit([1.0, float("-inf")], [1.0, 2.0])


def test_fit_input_immutability() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    true_copy = y_true.copy()
    pred_copy = y_pred.copy()
    detector = ResidualAnomalyDetector()
    detector.fit(y_true, y_pred)
    np.testing.assert_array_equal(y_true, true_copy)
    np.testing.assert_array_equal(y_pred, pred_copy)


# ---------------------------------------------------------------------------
# MAD fit
# ---------------------------------------------------------------------------


def test_mad_fit_success_and_self_return() -> None:
    detector = ResidualAnomalyDetector()
    returned = detector.fit([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 2.5, 4.5])
    assert returned is detector
    assert detector.is_fitted is True


def test_mad_residual_center_and_scale() -> None:
    y_true = [0.0, 1.0, 2.0, 3.0, 10.0]
    y_pred = [0.0, 1.0, 2.0, 3.0, 3.0]
    residuals = np.asarray(y_true, dtype=np.float64) - np.asarray(
        y_pred, dtype=np.float64
    )
    center = float(np.median(residuals))
    absolute = np.abs(residuals - center)
    mad = float(np.median(absolute))
    expected_scale = max(_MAD_SCALE_CONSTANT * mad, 1e-12)

    detector = ResidualAnomalyDetector()
    detector.fit(y_true, y_pred)
    calibration = detector.calibration
    assert calibration is not None
    assert calibration.residual_center == pytest.approx(center)
    assert calibration.residual_scale == pytest.approx(expected_scale)
    assert calibration.threshold == pytest.approx(3.5)
    assert calibration.mad_multiplier == pytest.approx(3.5)
    assert calibration.row_count == 5
    assert calibration.fitted_at.tzinfo is not None
    assert calibration.fitted_at.utcoffset() is not None


def test_mad_center_residuals_false() -> None:
    detector = ResidualAnomalyDetector(
        config=ResidualAnomalyConfig(center_residuals=False)
    )
    detector.fit([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
    calibration = detector.calibration
    assert calibration is not None
    assert calibration.residual_center == pytest.approx(0.0)


def test_mad_minimum_scale_fallback_and_warning() -> None:
    detector = ResidualAnomalyDetector(
        config=ResidualAnomalyConfig(minimum_scale=1.0)
    )
    detector.fit([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    calibration = detector.calibration
    assert calibration is not None
    assert calibration.residual_scale == pytest.approx(1.0)
    assert _MINIMUM_SCALE_WARNING in calibration.warnings


def test_mad_calibration_score_summary() -> None:
    y_true = [0.0, 2.0, 4.0]
    y_pred = [0.0, 1.0, 2.0]
    detector = ResidualAnomalyDetector()
    detector.fit(y_true, y_pred)
    calibration = detector.calibration
    assert calibration is not None
    residuals = np.asarray(y_true, dtype=np.float64) - np.asarray(
        y_pred, dtype=np.float64
    )
    center = float(np.median(residuals))
    absolute = np.abs(residuals - center)
    scale = float(calibration.residual_scale)  # type: ignore[arg-type]
    scores = absolute / scale
    assert calibration.calibration_score_min == pytest.approx(float(np.min(scores)))
    assert calibration.calibration_score_max == pytest.approx(float(np.max(scores)))
    assert calibration.calibration_score_mean == pytest.approx(float(np.mean(scores)))
    assert calibration.calibration_score_std == pytest.approx(
        float(np.std(scores, ddof=0))
    )


# ---------------------------------------------------------------------------
# QUANTILE fit
# ---------------------------------------------------------------------------


def test_quantile_fit_success() -> None:
    detector = ResidualAnomalyDetector(
        config=ResidualAnomalyConfig(
            method=ResidualThresholdMethod.QUANTILE,
            quantile=0.75,
        )
    )
    y_true = [0.0, 1.0, 2.0, 3.0]
    y_pred = [0.0, 1.0, 1.0, 1.0]
    detector.fit(y_true, y_pred)
    calibration = detector.calibration
    assert calibration is not None
    residuals = np.asarray(y_true, dtype=np.float64) - np.asarray(
        y_pred, dtype=np.float64
    )
    center = float(np.median(residuals))
    absolute = np.abs(residuals - center)
    expected_threshold = float(np.quantile(absolute, 0.75))
    assert calibration.residual_center == pytest.approx(center)
    assert calibration.threshold == pytest.approx(expected_threshold)
    assert calibration.residual_scale is None
    assert calibration.quantile == pytest.approx(0.75)
    assert calibration.mad_multiplier is None


def test_quantile_identical_score_warning() -> None:
    detector = ResidualAnomalyDetector(
        config=ResidualAnomalyConfig(method=ResidualThresholdMethod.QUANTILE)
    )
    detector.fit([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    calibration = detector.calibration
    assert calibration is not None
    assert _IDENTICAL_CALIBRATION_SCORE_WARNING in calibration.warnings


def test_quantile_threshold_zero_allowed() -> None:
    detector = ResidualAnomalyDetector(
        config=ResidualAnomalyConfig(
            method=ResidualThresholdMethod.QUANTILE,
            quantile=0.5,
            center_residuals=True,
        )
    )
    detector.fit([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    calibration = detector.calibration
    assert calibration is not None
    assert calibration.threshold == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Fit atomicity
# ---------------------------------------------------------------------------


def test_failed_initial_fit_keeps_unfitted() -> None:
    detector = ResidualAnomalyDetector()
    with pytest.raises(InsufficientDataError):
        detector.fit([], [])
    assert detector.is_fitted is False
    assert detector.calibration is None


def test_successful_refit_replaces_calibration() -> None:
    detector = ResidualAnomalyDetector()
    detector.fit([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    first = detector.calibration
    assert first is not None
    detector.fit([10.0, 20.0, 30.0, 40.0], [9.0, 19.0, 29.0, 50.0])
    second = detector.calibration
    assert second is not None
    assert second.row_count == 4
    assert second.row_count != first.row_count


def test_failed_refit_preserves_calibration() -> None:
    detector = ResidualAnomalyDetector()
    detector.fit([1.0, 2.0, 3.0], [1.0, 2.0, 2.5])
    before = detector.calibration
    assert before is not None
    before_threshold = detector.threshold
    before_fitted_at = detector.fitted_at
    with pytest.raises(DataValidationError):
        detector.fit([1.0, float("nan")], [1.0, 2.0])
    after = detector.calibration
    assert detector.is_fitted is True
    assert after is not None
    assert after.row_count == before.row_count
    assert detector.threshold == before_threshold
    assert detector.fitted_at == before_fitted_at


# ---------------------------------------------------------------------------
# score_samples / predict / detect
# ---------------------------------------------------------------------------


def test_score_samples_before_fit_blocked() -> None:
    detector = ResidualAnomalyDetector()
    with pytest.raises(ProcessIntelligenceError):
        detector.score_samples([1.0], [1.0])


def test_mad_and_quantile_scores() -> None:
    mad = ResidualAnomalyDetector()
    mad.fit([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 2.0])
    mad_scores = mad.score_samples([0.0, 5.0], [0.0, 0.0])
    assert isinstance(mad_scores, np.ndarray)
    assert mad_scores.ndim == 1
    assert mad_scores.dtype == np.float64
    assert mad_scores.shape == (2,)
    assert np.isfinite(mad_scores).all()
    assert (mad_scores >= 0.0).all()

    quantile = ResidualAnomalyDetector(
        config=ResidualAnomalyConfig(method=ResidualThresholdMethod.QUANTILE)
    )
    quantile.fit([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 2.0])
    q_scores = quantile.score_samples([0.0, 5.0], [0.0, 0.0])
    calibration = quantile.calibration
    assert calibration is not None
    expected = np.abs(
        np.array([0.0, 5.0]) - np.array([0.0, 0.0]) - calibration.residual_center
    )
    np.testing.assert_allclose(q_scores, expected)


def test_score_samples_empty_and_immutability() -> None:
    detector = ResidualAnomalyDetector()
    detector.fit([1.0, 2.0], [1.0, 2.0])
    empty = detector.score_samples([], [])
    assert empty.shape == (0,)
    assert empty.dtype == np.float64

    y_true = np.array([1.0, 3.0])
    y_pred = np.array([1.0, 1.0])
    true_copy = y_true.copy()
    pred_copy = y_pred.copy()
    first = detector.score_samples(y_true, y_pred)
    second = detector.score_samples(y_true, y_pred)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(y_true, true_copy)
    np.testing.assert_array_equal(y_pred, pred_copy)


def test_predict_threshold_behavior() -> None:
    exclusive = ResidualAnomalyDetector(
        config=ResidualAnomalyConfig(
            method=ResidualThresholdMethod.QUANTILE,
            quantile=0.5,
            threshold_inclusive=False,
            center_residuals=False,
        )
    )
    exclusive.fit([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
    # absolute residuals: 0,1,2; threshold = median = 1.0
    preds = exclusive.predict([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(preds, np.array([1, 1, -1], dtype=np.int64))

    inclusive = ResidualAnomalyDetector(
        config=ResidualAnomalyConfig(
            method=ResidualThresholdMethod.QUANTILE,
            quantile=0.5,
            threshold_inclusive=True,
            center_residuals=False,
        )
    )
    inclusive.fit([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
    preds_inc = inclusive.predict([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(preds_inc, np.array([1, -1, -1], dtype=np.int64))


def test_predict_before_fit_and_empty() -> None:
    detector = ResidualAnomalyDetector()
    with pytest.raises(ProcessIntelligenceError):
        detector.predict([1.0], [1.0])
    detector.fit([1.0, 2.0], [1.0, 2.0])
    empty = detector.predict([], [])
    assert empty.shape == (0,)
    assert np.issubdtype(empty.dtype, np.integer)


def test_predict_matches_score_threshold() -> None:
    detector = ResidualAnomalyDetector(
        config=ResidualAnomalyConfig(threshold_inclusive=False)
    )
    detector.fit([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 2.5])
    y_true = [0.0, 10.0]
    y_pred = [0.0, 0.0]
    scores = detector.score_samples(y_true, y_pred)
    preds = detector.predict(y_true, y_pred)
    threshold = detector.threshold
    assert threshold is not None
    expected = np.where(scores > threshold, -1, 1)
    np.testing.assert_array_equal(preds, expected)


def test_detect_full_result() -> None:
    detector = ResidualAnomalyDetector(
        config=ResidualAnomalyConfig(
            method=ResidualThresholdMethod.QUANTILE,
            quantile=0.5,
            center_residuals=False,
            threshold_inclusive=False,
        )
    )
    detector.fit([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
    with pytest.raises(ProcessIntelligenceError):
        ResidualAnomalyDetector().detect([1.0], [1.0])

    y_true = [0.0, 1.0, 4.0]
    y_pred = [0.0, 0.0, 0.0]
    result = detector.detect(y_true, y_pred)
    assert isinstance(result, ResidualAnomalyResult)
    assert result.predictions == [0.0, 0.0, 0.0]
    assert result.residuals == pytest.approx([0.0, 1.0, 4.0])
    assert result.absolute_centered_residuals == pytest.approx([0.0, 1.0, 4.0])
    assert result.scores == pytest.approx([0.0, 1.0, 4.0])
    assert result.raw_predictions == [1, 1, -1]
    assert result.is_anomaly == [False, False, True]
    assert result.anomaly_count == 1
    assert result.anomaly_fraction == pytest.approx(1.0 / 3.0)
    assert result.residual_mean == pytest.approx(float(np.mean([0.0, 1.0, 4.0])))
    assert result.residual_std == pytest.approx(
        float(np.std([0.0, 1.0, 4.0], ddof=0))
    )
    assert result.score_min == pytest.approx(0.0)
    assert result.score_max == pytest.approx(4.0)
    assert result.score_mean == pytest.approx(float(np.mean([0.0, 1.0, 4.0])))
    assert result.score_std == pytest.approx(float(np.std([0.0, 1.0, 4.0], ddof=0)))


def test_detect_empty_result() -> None:
    detector = ResidualAnomalyDetector()
    detector.fit([1.0, 2.0], [1.0, 2.0])
    result = detector.detect([], [])
    assert result.row_count == 0
    assert result.anomaly_count == 0
    assert result.anomaly_fraction == 0.0
    assert result.predictions == []
    assert result.threshold == detector.threshold
    assert result.residual_mean is None
    assert result.score_std is None


def test_detect_warnings_deterministic() -> None:
    all_normal = ResidualAnomalyDetector(
        config=ResidualAnomalyConfig(
            method=ResidualThresholdMethod.QUANTILE,
            quantile=0.99,
            center_residuals=False,
            threshold_inclusive=False,
            mad_multiplier=100.0,
        )
    )
    all_normal.fit([0.0, 0.1, 0.2], [0.0, 0.0, 0.0])
    result_normal = all_normal.detect([0.0, 0.05], [0.0, 0.0])
    assert _ALL_NORMAL_WARNING in result_normal.warnings

    all_anomaly = ResidualAnomalyDetector(
        config=ResidualAnomalyConfig(
            method=ResidualThresholdMethod.QUANTILE,
            quantile=0.01,
            center_residuals=False,
            threshold_inclusive=True,
        )
    )
    all_anomaly.fit([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
    result_anomaly = all_anomaly.detect([5.0, 6.0], [0.0, 0.0])
    assert _ALL_ANOMALY_WARNING in result_anomaly.warnings

    constant = ResidualAnomalyDetector()
    constant.fit([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    result_constant = constant.detect([1.0, 1.0], [0.0, 0.0])
    assert _CONSTANT_SCORE_WARNING in result_constant.warnings
    assert len(result_constant.warnings) == len(set(result_constant.warnings))
    # deterministic order: all-normal / all-anomaly / constant as applicable
    if (
        _ALL_NORMAL_WARNING in result_constant.warnings
        and _CONSTANT_SCORE_WARNING in result_constant.warnings
    ):
        assert result_constant.warnings.index(_ALL_NORMAL_WARNING) < (
            result_constant.warnings.index(_CONSTANT_SCORE_WARNING)
        )


def test_detect_result_mutation_isolation() -> None:
    detector = ResidualAnomalyDetector()
    detector.fit([1.0, 2.0, 3.0], [1.0, 2.0, 2.0])
    first = detector.detect([1.0, 4.0], [1.0, 1.0])
    first.scores.append(999.0)
    first.warnings.append("mutated")
    second = detector.detect([1.0, 4.0], [1.0, 1.0])
    assert 999.0 not in second.scores
    assert "mutated" not in second.warnings


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_before_and_after_fit() -> None:
    detector = ResidualAnomalyDetector(
        config=ResidualAnomalyConfig(method=ResidualThresholdMethod.MAD, mad_multiplier=2.5)
    )
    before = detector.get_metadata()
    assert before["fitted"] is False
    assert before["fitted_at"] is None
    assert before["row_count"] == 0
    assert before["residual_center"] is None
    assert before["threshold"] is None
    assert before["mad_multiplier"] == pytest.approx(2.5)
    assert before["score_direction"] == "higher_is_more_anomalous"
    assert "y_true" not in before
    assert "y_pred" not in before
    assert "residuals" not in before

    detector.fit([1.0, 2.0, 3.0], [0.5, 2.0, 3.5])
    after = detector.get_metadata()
    calibration = detector.calibration
    assert calibration is not None
    assert after["method"] == "MAD"
    assert after["fitted"] is True
    assert after["fitted_at"] == calibration.fitted_at
    assert after["residual_center"] == pytest.approx(calibration.residual_center)
    assert after["residual_scale"] == pytest.approx(calibration.residual_scale)
    assert after["threshold"] == pytest.approx(calibration.threshold)
    assert after["mad_multiplier"] == pytest.approx(2.5)
    assert after["quantile"] is None
    assert after["score_direction"] == "higher_is_more_anomalous"

    after["threshold"] = -1.0
    again = detector.get_metadata()
    assert again["threshold"] == pytest.approx(calibration.threshold)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_across_detectors() -> None:
    config = ResidualAnomalyConfig(
        method=ResidualThresholdMethod.MAD,
        mad_multiplier=3.0,
        center_residuals=True,
    )
    y_true = [0.0, 1.0, 2.0, 8.0]
    y_pred = [0.0, 1.0, 1.5, 2.0]

    first = ResidualAnomalyDetector(config=config)
    second = ResidualAnomalyDetector(config=config)
    first.fit(y_true, y_pred)
    second.fit(y_true, y_pred)

    c1 = first.calibration
    c2 = second.calibration
    assert c1 is not None and c2 is not None
    assert c1.residual_center == pytest.approx(c2.residual_center)
    assert c1.residual_scale == pytest.approx(c2.residual_scale)
    assert c1.threshold == pytest.approx(c2.threshold)
    np.testing.assert_allclose(
        first.score_samples(y_true, y_pred),
        second.score_samples(y_true, y_pred),
    )
    np.testing.assert_array_equal(
        first.predict(y_true, y_pred),
        second.predict(y_true, y_pred),
    )


def test_quantile_threshold_determinism() -> None:
    config = ResidualAnomalyConfig(
        method=ResidualThresholdMethod.QUANTILE,
        quantile=0.8,
    )
    y_true = [0.0, 1.0, 2.0, 3.0, 10.0]
    y_pred = [0.0, 1.0, 2.0, 2.5, 3.0]
    a = ResidualAnomalyDetector(config=config).fit(y_true, y_pred)
    b = ResidualAnomalyDetector(config=config).fit(y_true, y_pred)
    assert a.threshold == pytest.approx(b.threshold)


def test_prediction_length_mismatch_on_score() -> None:
    detector = ResidualAnomalyDetector()
    detector.fit([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(DataValidationError):
        detector.score_samples([1.0], [])
