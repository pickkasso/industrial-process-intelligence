"""Residual-based anomaly scoring from regression actual vs predicted values (Step 7F)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
from pydantic import BaseModel, Field, field_validator, model_validator

from process_intelligence.core.exceptions import (
    DataValidationError,
    InsufficientDataError,
    ProcessIntelligenceError,
)

_SCORE_DIRECTION = "higher_is_more_anomalous"
_MAD_SCALE_CONSTANT = 1.4826
_MINIMUM_SCALE_WARNING = (
    "robust MAD scale was at or below minimum_scale; minimum_scale was used"
)
_IDENTICAL_CALIBRATION_SCORE_WARNING = (
    "calibration residual scores are identical; threshold discrimination is limited"
)
_ALL_NORMAL_WARNING = "all rows classified as normal"
_ALL_ANOMALY_WARNING = "all rows classified as anomaly"
_CONSTANT_SCORE_WARNING = "anomaly scores are constant across rows"


def _is_finite_number(value: object) -> bool:
    """Return True when ``value`` is a finite real number (bool excluded)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _require_finite_float(value: object, *, field_name: str) -> float:
    """Validate a finite non-bool float and return it as ``float``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a finite float (bool not allowed), "
            f"got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be a finite float, got {value!r}")
    return number


def _require_strict_bool(value: object, *, field_name: str) -> bool:
    """Validate an actual bool (reject int 0/1 and strings)."""
    if not isinstance(value, bool):
        raise ValueError(
            f"{field_name} must be a bool (int/str coercion not allowed), "
            f"got {type(value).__name__}"
        )
    return value


def _normalize_zero(value: float) -> float:
    """Normalize signed zero to ``0.0``."""
    return 0.0 if value == 0.0 else value


def _dedupe_warnings(warnings: Sequence[str]) -> list[str]:
    """Return unique warning strings preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for warning in warnings:
        if warning in seen:
            continue
        seen.add(warning)
        result.append(warning)
    return result


def _is_boolean_array(values: np.ndarray) -> bool:
    """Return True when values are boolean."""
    if values.dtype == np.bool_ or values.dtype.kind == "b":
        return True
    if values.dtype.kind == "O" and values.size > 0:
        return all(isinstance(value, (bool, np.bool_)) for value in values.tolist())
    return False


def _is_string_array(values: np.ndarray) -> bool:
    """Return True when values are string-like."""
    if values.dtype.kind in {"U", "S"}:
        return True
    if values.dtype.kind == "O" and values.size > 0:
        return all(isinstance(value, str) for value in values.tolist())
    return False


def _has_null_entries(values: np.ndarray) -> bool:
    """Return True when the array contains null-like entries."""
    if values.dtype.kind == "O":
        return any(
            value is None or (isinstance(value, float) and math.isnan(value))
            for value in values.tolist()
        )
    if values.dtype.kind == "f":
        return bool(np.isnan(values).any())
    return False


def _coerce_1d_array(values: Any, *, field_name: str) -> np.ndarray:
    """Convert supported 1D inputs to an independent NumPy array."""
    if isinstance(values, (str, bytes)):
        raise TypeError(
            f"{field_name} must not be str or bytes; got {type(values).__name__}"
        )
    if isinstance(values, (pl.DataFrame, pd.DataFrame)):
        raise TypeError(
            f"{field_name} must not be a DataFrame; got {type(values).__name__}"
        )
    if isinstance(values, Mapping):
        raise TypeError(
            f"{field_name} must not be a Mapping; got {type(values).__name__}"
        )

    if isinstance(values, pl.Series):
        array = np.array(values.to_numpy(), copy=True)
    elif isinstance(values, pd.Series):
        array = values.to_numpy(copy=True)
    elif isinstance(values, np.ndarray):
        array = np.array(values, copy=True)
    elif isinstance(values, Sequence):
        array = np.asarray(list(values))
    else:
        raise TypeError(
            f"{field_name} must be a polars.Series, pandas.Series, "
            f"1D numpy.ndarray, or Sequence (excluding str/bytes), "
            f"got {type(values).__name__}"
        )

    if array.ndim != 1:
        raise DataValidationError(
            f"{field_name} must be 1-dimensional, got shape {array.shape}"
        )
    return array


def _validate_numeric_values(values: np.ndarray, *, field_name: str) -> np.ndarray:
    """Validate finite non-boolean numeric values and return float64 copy."""
    if _has_null_entries(values):
        raise DataValidationError(f"{field_name} must not contain null values")
    if _is_boolean_array(values):
        raise DataValidationError(
            f"{field_name} must be numeric; boolean values are not allowed"
        )
    if _is_string_array(values):
        raise DataValidationError(
            f"{field_name} must be numeric; string values are not allowed"
        )
    if values.dtype.kind not in {"f", "i", "u"}:
        if values.dtype.kind == "O":
            try:
                numeric = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise DataValidationError(
                    f"{field_name} must be numeric"
                ) from exc
        else:
            raise DataValidationError(f"{field_name} must be numeric")
    else:
        numeric = values.astype(np.float64, copy=True)

    if not np.isfinite(numeric).all():
        raise DataValidationError(
            f"{field_name} must not contain NaN or infinite values"
        )
    return numeric


def _reject_polars_nulls(values: Any, *, field_name: str) -> None:
    """Reject Polars Series inputs that contain null values."""
    if isinstance(values, pl.Series) and int(values.null_count()) > 0:
        raise DataValidationError(f"{field_name} must not contain null values")


def _validate_pair(
    y_true: Any,
    y_pred: Any,
    *,
    allow_empty: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce aligned y_true / y_pred arrays without mutation."""
    _reject_polars_nulls(y_true, field_name="y_true")
    _reject_polars_nulls(y_pred, field_name="y_pred")
    true_raw = _coerce_1d_array(y_true, field_name="y_true")
    pred_raw = _coerce_1d_array(y_pred, field_name="y_pred")

    if true_raw.shape[0] != pred_raw.shape[0]:
        raise DataValidationError(
            f"y_true length ({true_raw.shape[0]}) must match "
            f"y_pred length ({pred_raw.shape[0]})"
        )

    if true_raw.size == 0:
        if not allow_empty:
            raise InsufficientDataError(
                "y_true and y_pred must contain at least one sample for fit"
            )
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    true_values = _validate_numeric_values(true_raw, field_name="y_true")
    pred_values = _validate_numeric_values(pred_raw, field_name="y_pred")
    return true_values, pred_values


def _compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute ``y_true - y_pred`` as float64."""
    return np.asarray(y_true - y_pred, dtype=np.float64)


def _absolute_centered(residuals: np.ndarray, residual_center: float) -> np.ndarray:
    """Return absolute residuals centered by ``residual_center``."""
    return np.abs(residuals - float(residual_center)).astype(np.float64, copy=False)


def _score_from_absolute(
    absolute_centered: np.ndarray,
    *,
    method: ResidualThresholdMethod,
    residual_scale: float | None,
) -> np.ndarray:
    """Compute anomaly scores from absolute centered residuals."""
    if method is ResidualThresholdMethod.MAD:
        if residual_scale is None:
            raise ProcessIntelligenceError(
                "MAD scoring requires a positive residual_scale"
            )
        scale = float(residual_scale)
        if scale <= 0.0 or not math.isfinite(scale):
            raise ProcessIntelligenceError(
                f"MAD residual_scale must be a positive finite float, got {scale!r}"
            )
        scores = absolute_centered / scale
    else:
        scores = absolute_centered.astype(np.float64, copy=True)
    return np.asarray(scores, dtype=np.float64)


def _raw_predictions_from_scores(
    scores: np.ndarray,
    *,
    threshold: float,
    threshold_inclusive: bool,
) -> np.ndarray:
    """Map scores to raw predictions (``-1`` anomaly, ``1`` normal)."""
    if scores.size == 0:
        return np.empty(0, dtype=np.int64)
    if threshold_inclusive:
        is_anomaly = scores >= float(threshold)
    else:
        is_anomaly = scores > float(threshold)
    raw = np.where(is_anomaly, -1, 1).astype(np.int64, copy=False)
    return raw


def _summary_stats(values: np.ndarray) -> tuple[float, float, float, float]:
    """Return min, max, mean, and population std as Python floats."""
    minimum = _normalize_zero(float(np.min(values)))
    maximum = _normalize_zero(float(np.max(values)))
    mean = _normalize_zero(float(np.mean(values)))
    # Floating-point mean can land slightly outside [min, max] for near-constant
    # arrays; clamp so schema ordering invariants remain satisfiable.
    mean = min(maximum, max(minimum, mean))
    std = _normalize_zero(float(np.std(values, ddof=0)))
    if std < 0.0:
        std = 0.0
    return minimum, maximum, mean, std


class ResidualThresholdMethod(StrEnum):
    """Supported residual anomaly threshold calibration methods."""

    MAD = "MAD"
    QUANTILE = "QUANTILE"


class ResidualAnomalyConfig(BaseModel):
    """Configuration for residual anomaly threshold calibration and scoring."""

    method: ResidualThresholdMethod = ResidualThresholdMethod.MAD
    mad_multiplier: float = 3.5
    quantile: float = 0.99
    minimum_scale: float = 1e-12
    center_residuals: bool = True
    threshold_inclusive: bool = False

    @field_validator("method", mode="before")
    @classmethod
    def _validate_method(cls, value: object) -> ResidualThresholdMethod:
        if isinstance(value, ResidualThresholdMethod):
            return value
        if isinstance(value, str):
            try:
                return ResidualThresholdMethod(value)
            except ValueError as exc:
                raise ValueError(
                    f"method must be a ResidualThresholdMethod, got {value!r}"
                ) from exc
        raise ValueError(
            f"method must be a ResidualThresholdMethod, got {type(value).__name__}"
        )

    @field_validator("mad_multiplier", mode="before")
    @classmethod
    def _validate_mad_multiplier(cls, value: object) -> float:
        number = _require_finite_float(value, field_name="mad_multiplier")
        if number <= 0.0:
            raise ValueError(f"mad_multiplier must be > 0, got {number}")
        return number

    @field_validator("quantile", mode="before")
    @classmethod
    def _validate_quantile(cls, value: object) -> float:
        number = _require_finite_float(value, field_name="quantile")
        if number <= 0.0 or number >= 1.0:
            raise ValueError(f"quantile must be in (0.0, 1.0), got {number}")
        return number

    @field_validator("minimum_scale", mode="before")
    @classmethod
    def _validate_minimum_scale(cls, value: object) -> float:
        number = _require_finite_float(value, field_name="minimum_scale")
        if number <= 0.0:
            raise ValueError(f"minimum_scale must be > 0, got {number}")
        return number

    @field_validator("center_residuals", "threshold_inclusive", mode="before")
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="bool field")


class ResidualCalibration(BaseModel):
    """Fitted residual anomaly calibration state derived from validation residuals."""

    method: ResidualThresholdMethod
    row_count: int
    residual_center: float
    residual_scale: float | None
    threshold: float
    mad_multiplier: float | None
    quantile: float | None
    center_residuals: bool
    threshold_inclusive: bool
    calibration_score_min: float
    calibration_score_max: float
    calibration_score_mean: float
    calibration_score_std: float
    fitted_at: datetime
    warnings: list[str] = Field(default_factory=list)

    @field_validator("center_residuals", "threshold_inclusive", mode="before")
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="bool field")

    @field_validator("fitted_at", mode="before")
    @classmethod
    def _validate_fitted_at(cls, value: object) -> datetime:
        if not isinstance(value, datetime):
            raise ValueError(
                f"fitted_at must be a datetime, got {type(value).__name__}"
            )
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("fitted_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_consistency(self) -> Self:
        if self.row_count < 1:
            raise ValueError(f"row_count must be >= 1, got {self.row_count}")
        if not _is_finite_number(self.residual_center):
            raise ValueError(
                f"residual_center must be a finite float, got {self.residual_center!r}"
            )
        if not _is_finite_number(self.threshold) or float(self.threshold) < 0.0:
            raise ValueError(
                f"threshold must be a finite float >= 0, got {self.threshold!r}"
            )
        if self.residual_scale is not None:
            if not _is_finite_number(self.residual_scale) or float(self.residual_scale) <= 0.0:
                raise ValueError(
                    "residual_scale must be None or a finite float > 0, "
                    f"got {self.residual_scale!r}"
                )
        for name, summary in (
            ("calibration_score_min", self.calibration_score_min),
            ("calibration_score_max", self.calibration_score_max),
            ("calibration_score_mean", self.calibration_score_mean),
            ("calibration_score_std", self.calibration_score_std),
        ):
            if not _is_finite_number(summary):
                raise ValueError(
                    f"{name} must be a finite float, got {summary!r}"
                )
        if not (
            self.calibration_score_min
            <= self.calibration_score_mean
            <= self.calibration_score_max
        ):
            raise ValueError(
                "calibration score summaries must satisfy "
                "calibration_score_min <= calibration_score_mean <= "
                "calibration_score_max"
            )
        if float(self.calibration_score_std) < 0.0:
            raise ValueError(
                f"calibration_score_std must be >= 0, got {self.calibration_score_std}"
            )
        if len(self.warnings) != len(set(self.warnings)):
            raise ValueError("warnings must not contain duplicate values")

        if self.method is ResidualThresholdMethod.MAD:
            if self.residual_scale is None:
                raise ValueError("MAD calibration requires residual_scale")
            if self.mad_multiplier is None:
                raise ValueError("MAD calibration requires mad_multiplier")
            if self.quantile is not None:
                raise ValueError("MAD calibration requires quantile to be None")
            if not _is_finite_number(self.mad_multiplier) or float(self.mad_multiplier) <= 0.0:
                raise ValueError(
                    "mad_multiplier must be a finite float > 0, "
                    f"got {self.mad_multiplier!r}"
                )
            if not math.isclose(
                float(self.threshold),
                float(self.mad_multiplier),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "MAD threshold must equal mad_multiplier: "
                    f"threshold={self.threshold}, mad_multiplier={self.mad_multiplier}"
                )
        elif self.method is ResidualThresholdMethod.QUANTILE:
            if self.residual_scale is not None:
                raise ValueError("QUANTILE calibration requires residual_scale to be None")
            if self.quantile is None:
                raise ValueError("QUANTILE calibration requires quantile")
            if self.mad_multiplier is not None:
                raise ValueError("QUANTILE calibration requires mad_multiplier to be None")
            if not _is_finite_number(self.quantile):
                raise ValueError(
                    f"quantile must be a finite float, got {self.quantile!r}"
                )
            q = float(self.quantile)
            if q <= 0.0 or q >= 1.0:
                raise ValueError(f"quantile must be in (0.0, 1.0), got {q}")
        else:
            raise ValueError(f"unsupported residual threshold method: {self.method!r}")
        return self


class ResidualAnomalyResult(BaseModel):
    """Structured residual anomaly scores and binary decisions for one input batch."""

    predictions: list[float] = Field(default_factory=list)
    residuals: list[float] = Field(default_factory=list)
    absolute_centered_residuals: list[float] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    is_anomaly: list[bool] = Field(default_factory=list)
    raw_predictions: list[int] = Field(default_factory=list)
    threshold: float
    row_count: int
    anomaly_count: int
    anomaly_fraction: float
    residual_mean: float | None = None
    residual_std: float | None = None
    score_min: float | None = None
    score_max: float | None = None
    score_mean: float | None = None
    score_std: float | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_consistency(self) -> Self:
        if self.row_count < 0:
            raise ValueError(f"row_count must be >= 0, got {self.row_count}")
        if self.anomaly_count < 0:
            raise ValueError(f"anomaly_count must be >= 0, got {self.anomaly_count}")
        if self.anomaly_count > self.row_count:
            raise ValueError(
                "anomaly_count cannot exceed row_count: "
                f"{self.anomaly_count} > {self.row_count}"
            )
        for name, collection in (
            ("predictions", self.predictions),
            ("residuals", self.residuals),
            ("absolute_centered_residuals", self.absolute_centered_residuals),
            ("scores", self.scores),
            ("is_anomaly", self.is_anomaly),
            ("raw_predictions", self.raw_predictions),
        ):
            if len(collection) != self.row_count:
                raise ValueError(
                    f"{name} length ({len(collection)}) must equal "
                    f"row_count ({self.row_count})"
                )
        if not _is_finite_number(self.threshold) or float(self.threshold) < 0.0:
            raise ValueError(
                f"threshold must be a finite float >= 0, got {self.threshold!r}"
            )
        for index, value in enumerate(self.predictions):
            if not _is_finite_number(value):
                raise ValueError(
                    f"predictions[{index}] must be a finite float, got {value!r}"
                )
        for index, value in enumerate(self.residuals):
            if not _is_finite_number(value):
                raise ValueError(
                    f"residuals[{index}] must be a finite float, got {value!r}"
                )
        for index, value in enumerate(self.absolute_centered_residuals):
            if not _is_finite_number(value) or float(value) < 0.0:
                raise ValueError(
                    f"absolute_centered_residuals[{index}] must be a finite "
                    f"float >= 0, got {value!r}"
                )
        for index, value in enumerate(self.scores):
            if not _is_finite_number(value) or float(value) < 0.0:
                raise ValueError(
                    f"scores[{index}] must be a finite float >= 0, got {value!r}"
                )
        for index, raw in enumerate(self.raw_predictions):
            if raw not in (-1, 1):
                raise ValueError(
                    f"raw_predictions[{index}] must be -1 or 1, got {raw!r}"
                )
            expected_anomaly = raw == -1
            if bool(self.is_anomaly[index]) is not expected_anomaly:
                raise ValueError(
                    "is_anomaly must match raw_predictions == -1 at "
                    f"index {index}"
                )
        true_count = sum(1 for flag in self.is_anomaly if flag)
        if true_count != self.anomaly_count:
            raise ValueError(
                "anomaly_count must equal the number of True values in "
                f"is_anomaly ({true_count}), got {self.anomaly_count}"
            )
        if not _is_finite_number(self.anomaly_fraction):
            raise ValueError(
                "anomaly_fraction must be a finite float in [0.0, 1.0], "
                f"got {self.anomaly_fraction!r}"
            )
        fraction = float(self.anomaly_fraction)
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError(
                f"anomaly_fraction must be in [0.0, 1.0], got {fraction}"
            )
        if self.row_count == 0:
            if fraction != 0.0:
                raise ValueError(
                    "anomaly_fraction must be 0.0 when row_count is 0"
                )
            for name, summary in (
                ("residual_mean", self.residual_mean),
                ("residual_std", self.residual_std),
                ("score_min", self.score_min),
                ("score_max", self.score_max),
                ("score_mean", self.score_mean),
                ("score_std", self.score_std),
            ):
                if summary is not None:
                    raise ValueError(
                        f"{name} must be None when row_count is 0, got {summary!r}"
                    )
        else:
            expected_fraction = self.anomaly_count / self.row_count
            if not math.isclose(
                fraction,
                expected_fraction,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "anomaly_fraction must equal anomaly_count / row_count "
                    f"({expected_fraction}), got {fraction}"
                )
            for name, summary in (
                ("residual_mean", self.residual_mean),
                ("residual_std", self.residual_std),
                ("score_min", self.score_min),
                ("score_max", self.score_max),
                ("score_mean", self.score_mean),
                ("score_std", self.score_std),
            ):
                if summary is None or not _is_finite_number(summary):
                    raise ValueError(
                        f"{name} must be a finite float when row_count >= 1, "
                        f"got {summary!r}"
                    )
            assert self.residual_std is not None
            assert self.score_std is not None
            assert self.score_min is not None
            assert self.score_max is not None
            assert self.score_mean is not None
            if float(self.residual_std) < 0.0:
                raise ValueError(
                    f"residual_std must be >= 0, got {self.residual_std}"
                )
            if float(self.score_std) < 0.0:
                raise ValueError(f"score_std must be >= 0, got {self.score_std}")
            if not (self.score_min <= self.score_mean <= self.score_max):
                raise ValueError(
                    "score summaries must satisfy "
                    "score_min <= score_mean <= score_max"
                )
        if len(self.warnings) != len(set(self.warnings)):
            raise ValueError("warnings must not contain duplicate values")
        return self


class ResidualAnomalyDetector:
    """Calibrate and score residual anomalies from regression actual vs predicted."""

    def __init__(
        self,
        *,
        config: ResidualAnomalyConfig | None = None,
    ) -> None:
        """Create a residual anomaly detector.

        Args:
            config: Residual anomaly configuration. ``None`` uses defaults.

        Raises:
            TypeError: If ``config`` is not ``ResidualAnomalyConfig`` or ``None``.
        """
        if config is None:
            resolved = ResidualAnomalyConfig()
        elif isinstance(config, ResidualAnomalyConfig):
            resolved = config.model_copy(deep=True)
        else:
            raise TypeError(
                "config must be ResidualAnomalyConfig or None, "
                f"got {type(config).__name__}"
            )
        self._config: ResidualAnomalyConfig = resolved
        self._calibration: ResidualCalibration | None = None
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Whether the detector has a successfully fitted calibration."""
        return self._is_fitted

    @property
    def calibration(self) -> ResidualCalibration | None:
        """Deep copy of the fitted calibration, or ``None`` before fit."""
        if self._calibration is None:
            return None
        return self._calibration.model_copy(deep=True)

    @property
    def threshold(self) -> float | None:
        """Decision threshold from the fitted calibration, or ``None`` before fit."""
        if self._calibration is None:
            return None
        return float(self._calibration.threshold)

    @property
    def fitted_at(self) -> datetime | None:
        """Timezone-aware UTC timestamp of the last successful fit."""
        if self._calibration is None:
            return None
        return self._calibration.fitted_at

    def _require_fitted(self, *, operation: str) -> ResidualCalibration:
        """Return calibration or raise if the detector is not fitted."""
        if not self._is_fitted or self._calibration is None:
            raise ProcessIntelligenceError(
                f"{operation} requires a successful fit before it can be called"
            )
        return self._calibration

    def _build_calibration(
        self,
        residuals: np.ndarray,
        *,
        fitted_at: datetime,
    ) -> ResidualCalibration:
        """Compute a candidate calibration from residuals (does not mutate state)."""
        config = self._config
        if config.center_residuals:
            residual_center = _normalize_zero(float(np.median(residuals)))
        else:
            residual_center = 0.0

        absolute = _absolute_centered(residuals, residual_center)
        warnings: list[str] = []

        if config.method is ResidualThresholdMethod.MAD:
            mad = float(np.median(absolute))
            robust_scale = _MAD_SCALE_CONSTANT * mad
            effective_scale = max(robust_scale, float(config.minimum_scale))
            if robust_scale <= float(config.minimum_scale):
                warnings.append(_MINIMUM_SCALE_WARNING)
            scores = absolute / effective_scale
            threshold = float(config.mad_multiplier)
            residual_scale: float | None = float(effective_scale)
            mad_multiplier: float | None = float(config.mad_multiplier)
            quantile: float | None = None
        else:
            scores = absolute.astype(np.float64, copy=True)
            threshold = float(np.quantile(scores, float(config.quantile)))
            if not math.isfinite(threshold) or threshold < 0.0:
                raise ProcessIntelligenceError(
                    f"QUANTILE threshold must be a finite float >= 0, got {threshold!r}"
                )
            if scores.size > 0 and float(np.min(scores)) == float(np.max(scores)):
                warnings.append(_IDENTICAL_CALIBRATION_SCORE_WARNING)
            residual_scale = None
            mad_multiplier = None
            quantile = float(config.quantile)

        if not np.isfinite(scores).all():
            raise ProcessIntelligenceError("calibration scores must be finite")
        score_min, score_max, score_mean, score_std = _summary_stats(scores)

        return ResidualCalibration(
            method=config.method,
            row_count=int(residuals.shape[0]),
            residual_center=float(residual_center),
            residual_scale=residual_scale,
            threshold=_normalize_zero(float(threshold)),
            mad_multiplier=mad_multiplier,
            quantile=quantile,
            center_residuals=bool(config.center_residuals),
            threshold_inclusive=bool(config.threshold_inclusive),
            calibration_score_min=score_min,
            calibration_score_max=score_max,
            calibration_score_mean=score_mean,
            calibration_score_std=score_std,
            fitted_at=fitted_at,
            warnings=_dedupe_warnings(warnings),
        )

    def fit(self, y_true: Any, y_pred: Any) -> Self:
        """Calibrate residual anomaly thresholds from actual vs predicted values.

        Args:
            y_true: Observed regression targets.
            y_pred: Regression predictions aligned with ``y_true``.

        Returns:
            The fitted detector instance.

        Raises:
            InsufficientDataError: If inputs are empty.
            DataValidationError: If inputs fail validation.
            TypeError: If input types are unsupported.
        """
        true_values, pred_values = _validate_pair(
            y_true,
            y_pred,
            allow_empty=False,
        )
        residuals = _compute_residuals(true_values, pred_values)
        fitted_at = datetime.now(UTC)
        candidate = self._build_calibration(residuals, fitted_at=fitted_at)

        self._calibration = candidate
        self._is_fitted = True
        return self

    def score_samples(self, y_true: Any, y_pred: Any) -> np.ndarray:
        """Compute residual anomaly scores (higher means more anomalous).

        Args:
            y_true: Observed regression targets.
            y_pred: Regression predictions aligned with ``y_true``.

        Returns:
            One-dimensional float ndarray of anomaly scores.

        Raises:
            ProcessIntelligenceError: If called before a successful fit.
            DataValidationError: If inputs fail validation.
            TypeError: If input types are unsupported.
        """
        calibration = self._require_fitted(operation="score_samples")
        true_values, pred_values = _validate_pair(
            y_true,
            y_pred,
            allow_empty=True,
        )
        if true_values.size == 0:
            return np.empty(0, dtype=np.float64)

        residuals = _compute_residuals(true_values, pred_values)
        absolute = _absolute_centered(residuals, calibration.residual_center)
        scores = _score_from_absolute(
            absolute,
            method=calibration.method,
            residual_scale=calibration.residual_scale,
        )
        if not np.isfinite(scores).all():
            raise ProcessIntelligenceError("score_samples results must be finite")
        if (scores < 0.0).any():
            raise ProcessIntelligenceError("score_samples results must be >= 0")
        return scores

    def predict(self, y_true: Any, y_pred: Any) -> np.ndarray:
        """Return raw residual anomaly labels (``1`` normal, ``-1`` anomaly).

        Args:
            y_true: Observed regression targets.
            y_pred: Regression predictions aligned with ``y_true``.

        Returns:
            One-dimensional integer ndarray of ``-1`` / ``1`` labels.

        Raises:
            ProcessIntelligenceError: If called before a successful fit.
            DataValidationError: If inputs fail validation.
            TypeError: If input types are unsupported.
        """
        calibration = self._require_fitted(operation="predict")
        scores = self.score_samples(y_true, y_pred)
        return _raw_predictions_from_scores(
            scores,
            threshold=float(calibration.threshold),
            threshold_inclusive=bool(calibration.threshold_inclusive),
        )

    def detect(self, y_true: Any, y_pred: Any) -> ResidualAnomalyResult:
        """Score residuals and classify anomalies, preserving input order.

        Args:
            y_true: Observed regression targets.
            y_pred: Regression predictions aligned with ``y_true``.

        Returns:
            Structured residual anomaly detection result.

        Raises:
            ProcessIntelligenceError: If called before a successful fit.
            DataValidationError: If inputs fail validation.
            TypeError: If input types are unsupported.
        """
        calibration = self._require_fitted(operation="detect")
        true_values, pred_values = _validate_pair(
            y_true,
            y_pred,
            allow_empty=True,
        )
        threshold = float(calibration.threshold)
        row_count = int(true_values.shape[0])

        if row_count == 0:
            return ResidualAnomalyResult(
                predictions=[],
                residuals=[],
                absolute_centered_residuals=[],
                scores=[],
                is_anomaly=[],
                raw_predictions=[],
                threshold=threshold,
                row_count=0,
                anomaly_count=0,
                anomaly_fraction=0.0,
                residual_mean=None,
                residual_std=None,
                score_min=None,
                score_max=None,
                score_mean=None,
                score_std=None,
                warnings=[],
            )

        residuals = _compute_residuals(true_values, pred_values)
        absolute = _absolute_centered(residuals, calibration.residual_center)
        scores = _score_from_absolute(
            absolute,
            method=calibration.method,
            residual_scale=calibration.residual_scale,
        )
        raw = _raw_predictions_from_scores(
            scores,
            threshold=threshold,
            threshold_inclusive=bool(calibration.threshold_inclusive),
        )
        is_anomaly = [bool(value == -1) for value in raw.tolist()]
        anomaly_count = sum(1 for flag in is_anomaly if flag)
        residual_mean = _normalize_zero(float(np.mean(residuals)))
        residual_std = _normalize_zero(float(np.std(residuals, ddof=0)))
        score_min, score_max, score_mean, score_std = _summary_stats(scores)

        warnings: list[str] = []
        if anomaly_count == 0:
            warnings.append(_ALL_NORMAL_WARNING)
        if anomaly_count == row_count:
            warnings.append(_ALL_ANOMALY_WARNING)
        if math.isclose(score_min, score_max, rel_tol=0.0, abs_tol=0.0):
            warnings.append(_CONSTANT_SCORE_WARNING)

        return ResidualAnomalyResult(
            predictions=[float(value) for value in pred_values.tolist()],
            residuals=[float(value) for value in residuals.tolist()],
            absolute_centered_residuals=[float(value) for value in absolute.tolist()],
            scores=[float(value) for value in scores.tolist()],
            is_anomaly=is_anomaly,
            raw_predictions=[int(value) for value in raw.tolist()],
            threshold=threshold,
            row_count=row_count,
            anomaly_count=anomaly_count,
            anomaly_fraction=float(anomaly_count) / float(row_count),
            residual_mean=residual_mean,
            residual_std=residual_std,
            score_min=score_min,
            score_max=score_max,
            score_mean=score_mean,
            score_std=score_std,
            warnings=_dedupe_warnings(warnings),
        )

    def get_metadata(self) -> dict[str, Any]:
        """Return independent detector metadata without input arrays.

        Returns:
            A new dictionary describing configuration and calibration state.
        """
        config = self._config
        calibration = self._calibration
        if calibration is None:
            return {
                "method": config.method.value,
                "fitted": False,
                "fitted_at": None,
                "row_count": 0,
                "residual_center": None,
                "residual_scale": None,
                "threshold": None,
                "mad_multiplier": float(config.mad_multiplier),
                "quantile": float(config.quantile),
                "center_residuals": bool(config.center_residuals),
                "threshold_inclusive": bool(config.threshold_inclusive),
                "score_direction": _SCORE_DIRECTION,
            }
        return {
            "method": calibration.method.value,
            "fitted": True,
            "fitted_at": calibration.fitted_at,
            "row_count": int(calibration.row_count),
            "residual_center": float(calibration.residual_center),
            "residual_scale": (
                None
                if calibration.residual_scale is None
                else float(calibration.residual_scale)
            ),
            "threshold": float(calibration.threshold),
            "mad_multiplier": (
                None
                if calibration.mad_multiplier is None
                else float(calibration.mad_multiplier)
            ),
            "quantile": (
                None if calibration.quantile is None else float(calibration.quantile)
            ),
            "center_residuals": bool(calibration.center_residuals),
            "threshold_inclusive": bool(calibration.threshold_inclusive),
            "score_direction": _SCORE_DIRECTION,
        }
