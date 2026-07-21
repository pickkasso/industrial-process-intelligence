"""Residual-association root-cause diagnoser (Step 8C).

Ranks residual-associated features by Spearman correlation with residual
anomaly scores and robust anomaly/reference deviation. Results describe
association with model error patterns only—not process causation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Self

import numpy as np
import polars as pl
from pydantic import BaseModel, Field, field_validator, model_validator

from process_intelligence.core.enums import ColumnRole
from process_intelligence.core.exceptions import (
    DataValidationError,
    InsufficientDataError,
)
from process_intelligence.core.schemas import (
    AnomalyEvent,
    ExplanationResult,
    RootCauseFactor,
)
from process_intelligence.diagnosis.enums import DiagnosisMethod, DiagnosisScope
from process_intelligence.diagnosis.protocols import BaseRootCauseDiagnoser
from process_intelligence.diagnosis.schemas import (
    DiagnosisBatchResult,
    DiagnosisRequest,
    DiagnosisResult,
    ScalarMetadataValue,
)

_ROBUST_SCALE_CONSTANT = 1.4826
_ORIGINAL_ROW_ID = "_original_row_id"
_SUPPORTED_SCOPES = frozenset(
    {
        DiagnosisScope.SINGLE_EVENT,
        DiagnosisScope.TOP_ANOMALIES,
        DiagnosisScope.ANOMALY_GROUP,
        DiagnosisScope.GLOBAL,
    }
)
_DIRECTION_POSITIVE = "POSITIVE"
_DIRECTION_NEGATIVE = "NEGATIVE"
_DIRECTION_MIXED = "MIXED"
_DIRECTION_UNKNOWN = "UNKNOWN"

_WARNING_NEAR_ZERO_MAD = (
    "reference MAD was near zero; minimum_scale was applied for robust deviation"
)
_WARNING_CONSTANT_RANK = (
    "rank association was undefined because one input was nearly constant"
)

_CAVEAT_ASSOCIATION = (
    "Residual association does not establish process causation; treat factors "
    "as residual-associated candidates requiring model and process review."
)
_CAVEAT_MISSPECIFICATION = (
    "Residual patterns may reflect sensor error, omitted variables, "
    "distribution shift, or model misspecification."
)
_CAVEAT_DETECTOR = (
    "Associations depend on the selected validation-calibrated residual "
    "detector and reference group."
)
_CAVEAT_NEAR_ZERO = (
    "At least one feature had near-zero reference MAD or nearly constant "
    "rank association input; minimum_scale / zero-correlation fallback applied."
)
_CAVEAT_EXPLANATION = (
    "Supplied model explanation was not combined with residual association "
    "analysis."
)
_CAVEAT_NO_FACTORS = (
    "No residual-associated factors met the scoring criteria for this diagnosis."
)

_BATCH_WARNING_ASSOCIATION = (
    "Residual association does not establish process causation."
)
_BATCH_WARNING_AGGREGATED = (
    "Results aggregate individually diagnosed residual anomaly events."
)
_BATCH_WARNING_EFFECTS = (
    "Residual anomalies can reflect process, sensor, data-shift, or "
    "model-fit effects."
)
_BATCH_WARNING_EMPTY_FACTORS = (
    "At least one diagnosed event returned no residual-associated factors."
)


@dataclass
class _AggregateBucket:
    normalized_scores: list[float] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    abs_score_correlations: list[float] = field(default_factory=list)
    prevalences: list[float] = field(default_factory=list)
    directions: list[str] = field(default_factory=list)
    deviations: list[float] = field(default_factory=list)
    count: int = 0


def _require_strict_bool(value: object, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a bool, got {type(value).__name__}")
    return value


def _require_positive_finite_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a finite float > 0 "
            f"(bool not allowed), got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    if number <= 0.0:
        raise ValueError(f"{field_name} must be > 0, got {number}")
    return number


def _require_non_negative_finite_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a finite float >= 0 "
            f"(bool not allowed), got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    if number < 0.0:
        raise ValueError(f"{field_name} must be >= 0, got {number}")
    return number


def _require_finite_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a finite float "
            f"(bool not allowed), got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return number


def _require_unit_interval(value: object, *, field_name: str) -> float:
    number = _require_finite_float(value, field_name=field_name)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{field_name} must be in [0.0, 1.0], got {number}")
    return number


def _require_correlation(value: object, *, field_name: str) -> float:
    number = _require_finite_float(value, field_name=field_name)
    if number < -1.0 or number > 1.0:
        raise ValueError(f"{field_name} must be in [-1.0, 1.0], got {number}")
    return number


def _require_strict_int_ge1(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{field_name} must be an int >= 1 "
            f"(bool not allowed), got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1, got {value}")
    return value


def _require_non_empty_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be str, got {type(value).__name__}")
    if value == "" or value.strip() == "":
        raise ValueError(f"{field_name} must be a non-empty, non-whitespace string")
    return value


def _require_weight(value: object, *, field_name: str) -> float:
    return _require_unit_interval(value, field_name=field_name)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Assign 1-based average ranks with stable tie handling (no SciPy)."""
    n = int(values.shape[0])
    if n == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and values[order[j]] == values[order[i]]:
            j += 1
        # Average of 1-based positions i+1 .. j inclusive.
        average = 0.5 * (float(i + 1) + float(j))
        for k in range(i, j):
            ranks[order[k]] = average
        i = j
    return ranks


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    denom_x = float(np.sqrt(np.sum(np.square(x_centered))))
    denom_y = float(np.sqrt(np.sum(np.square(y_centered))))
    if denom_x == 0.0 or denom_y == 0.0:
        return 0.0
    corr = float(np.sum(x_centered * y_centered) / (denom_x * denom_y))
    if corr > 1.0:
        return 1.0
    if corr < -1.0:
        return -1.0
    return corr


def _spearman_rank_correlation(
    left: np.ndarray,
    right: np.ndarray,
    *,
    minimum_scale: float,
) -> tuple[float, bool]:
    """Spearman via average ranks + Pearson; never returns NaN."""
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_std = float(np.std(left_ranks))
    right_std = float(np.std(right_ranks))
    if left_std <= minimum_scale or right_std <= minimum_scale:
        return 0.0, True
    correlation = _pearson_correlation(left_ranks, right_ranks)
    if not math.isfinite(correlation):
        return 0.0, True
    return float(correlation), False


class ResidualAssociationConfig(BaseModel):
    """Configuration for residual-association diagnosis.

    Controls residual column names, minimum row counts, robust scale floors,
    deviation thresholds, association score weights, and ranking behavior.
    Weights and scores are associative ranking heuristics only.
    """

    residual_column: str = "_regression_residual"
    absolute_residual_column: str = "_absolute_centered_residual"
    anomaly_score_column: str = "_residual_anomaly_score"
    anomaly_indicator_column: str = "_is_residual_anomaly"
    minimum_total_rows: int = 5
    minimum_reference_rows: int = 2
    minimum_anomaly_rows: int = 1
    minimum_scale: float = 1e-12
    deviation_z_threshold: float = 3.5
    direction_tolerance: float = 1e-12
    score_correlation_weight: float = 0.45
    group_deviation_weight: float = 0.35
    prevalence_weight: float = 0.20
    include_zero_score_factors: bool = False
    normalize_factor_scores: bool = True
    use_anomaly_score_ordering: bool = True

    @field_validator(
        "residual_column",
        "absolute_residual_column",
        "anomaly_score_column",
        "anomaly_indicator_column",
        mode="before",
    )
    @classmethod
    def _validate_column_names(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="column name")

    @field_validator(
        "minimum_total_rows",
        "minimum_reference_rows",
        "minimum_anomaly_rows",
        mode="before",
    )
    @classmethod
    def _validate_row_counts(cls, value: object) -> int:
        return _require_strict_int_ge1(value, field_name="row count")

    @field_validator("minimum_scale", "deviation_z_threshold", mode="before")
    @classmethod
    def _validate_positive_floats(cls, value: object) -> float:
        return _require_positive_finite_float(value, field_name="positive float")

    @field_validator("direction_tolerance", mode="before")
    @classmethod
    def _validate_direction_tolerance(cls, value: object) -> float:
        return _require_non_negative_finite_float(
            value,
            field_name="direction_tolerance",
        )

    @field_validator(
        "score_correlation_weight",
        "group_deviation_weight",
        "prevalence_weight",
        mode="before",
    )
    @classmethod
    def _validate_weights(cls, value: object) -> float:
        return _require_weight(value, field_name="weight")

    @field_validator(
        "include_zero_score_factors",
        "normalize_factor_scores",
        "use_anomaly_score_ordering",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="bool field")

    @model_validator(mode="after")
    def _validate_config_consistency(self) -> Self:
        columns = [
            self.residual_column,
            self.absolute_residual_column,
            self.anomaly_score_column,
            self.anomaly_indicator_column,
        ]
        if len(columns) != len(set(columns)):
            raise ValueError(
                "residual_column, absolute_residual_column, "
                "anomaly_score_column, and anomaly_indicator_column "
                "must be distinct"
            )
        weight_sum = (
            self.score_correlation_weight
            + self.group_deviation_weight
            + self.prevalence_weight
        )
        if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "score_correlation_weight, group_deviation_weight, and "
                f"prevalence_weight must sum to 1.0, got {weight_sum}"
            )
        if (
            self.score_correlation_weight == 0.0
            and self.group_deviation_weight == 0.0
            and self.prevalence_weight == 0.0
        ):
            raise ValueError("association weights must not all be zero")
        return self


class ResidualFeatureStatistic(BaseModel):
    """Per-feature residual association statistics.

    Stores scalar summaries used to rank residual-associated candidates.
    Does not embed estimators, DataFrames, or full arrays.
    """

    feature_name: str
    total_row_count: int
    anomaly_row_count: int
    reference_row_count: int
    feature_residual_rank_correlation: float
    feature_score_rank_correlation: float
    reference_median: float
    anomaly_median: float
    signed_location_difference: float
    median_absolute_deviation: float
    effective_scale: float
    group_deviation_z: float
    deviation_prevalence: float
    bounded_group_strength: float
    raw_association_score: float
    normalized_association_score: float
    direction: str
    confidence: float
    warnings: list[str] = Field(default_factory=list)

    @field_validator("feature_name", mode="before")
    @classmethod
    def _validate_feature_name(cls, value: object) -> str:
        name = _require_non_empty_str(value, field_name="feature_name")
        if name == _ORIGINAL_ROW_ID:
            raise ValueError(f"feature_name must not be {_ORIGINAL_ROW_ID!r}")
        return name

    @field_validator(
        "total_row_count",
        "anomaly_row_count",
        "reference_row_count",
        mode="before",
    )
    @classmethod
    def _validate_row_counts(cls, value: object) -> int:
        return _require_strict_int_ge1(value, field_name="row count")

    @field_validator(
        "feature_residual_rank_correlation",
        "feature_score_rank_correlation",
        mode="before",
    )
    @classmethod
    def _validate_correlations(cls, value: object) -> float:
        return _require_correlation(value, field_name="rank correlation")

    @field_validator(
        "reference_median",
        "anomaly_median",
        "signed_location_difference",
        mode="before",
    )
    @classmethod
    def _validate_finite_stats(cls, value: object) -> float:
        return _require_finite_float(value, field_name="statistic")

    @field_validator("median_absolute_deviation", mode="before")
    @classmethod
    def _validate_mad(cls, value: object) -> float:
        return _require_non_negative_finite_float(
            value,
            field_name="median_absolute_deviation",
        )

    @field_validator("effective_scale", mode="before")
    @classmethod
    def _validate_effective_scale(cls, value: object) -> float:
        return _require_positive_finite_float(value, field_name="effective_scale")

    @field_validator("group_deviation_z", "raw_association_score", mode="before")
    @classmethod
    def _validate_non_negative_scores(cls, value: object) -> float:
        return _require_non_negative_finite_float(value, field_name="score")

    @field_validator(
        "deviation_prevalence",
        "bounded_group_strength",
        "normalized_association_score",
        "confidence",
        mode="before",
    )
    @classmethod
    def _validate_unit_interval_fields(cls, value: object) -> float:
        return _require_unit_interval(value, field_name="unit-interval field")

    @field_validator("direction", mode="before")
    @classmethod
    def _validate_direction(cls, value: object) -> str:
        text = _require_non_empty_str(value, field_name="direction")
        allowed = {
            _DIRECTION_POSITIVE,
            _DIRECTION_NEGATIVE,
            _DIRECTION_MIXED,
            _DIRECTION_UNKNOWN,
        }
        if text not in allowed:
            raise ValueError(
                f"direction must be one of {sorted(allowed)}, got {text!r}"
            )
        return text

    @field_validator("warnings", mode="before")
    @classmethod
    def _validate_warnings_before(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"warnings must be a list[str], got {type(value).__name__}"
            )
        return list(value)

    @field_validator("warnings", mode="after")
    @classmethod
    def _validate_warnings(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = _require_non_empty_str(item, field_name="warnings")
            if text in seen:
                raise ValueError(f"warnings must not contain duplicates: {text!r}")
            seen.add(text)
            cleaned.append(text)
        return cleaned

    @model_validator(mode="after")
    def _validate_row_count_consistency(self) -> Self:
        if self.anomaly_row_count > self.total_row_count:
            raise ValueError(
                "anomaly_row_count must be <= total_row_count "
                f"({self.anomaly_row_count} > {self.total_row_count})"
            )
        if self.reference_row_count > self.total_row_count:
            raise ValueError(
                "reference_row_count must be <= total_row_count "
                f"({self.reference_row_count} > {self.total_row_count})"
            )
        if self.anomaly_row_count + self.reference_row_count > self.total_row_count:
            raise ValueError(
                "anomaly_row_count + reference_row_count must be <= total_row_count "
                f"({self.anomaly_row_count} + {self.reference_row_count} > "
                f"{self.total_row_count})"
            )
        return self


@dataclass(frozen=True, slots=True)
class _EventDiagnosisPayload:
    result: DiagnosisResult
    statistics: tuple[ResidualFeatureStatistic, ...]


class ResidualAssociationDiagnoser(BaseRootCauseDiagnoser):
    """Concrete diagnoser ranking residual-associated feature candidates.

    Uses Spearman rank association with residual anomaly scores and robust
    anomaly/reference deviation. Does not claim causation, does not fit
    models, and does not cache inputs or outputs between ``diagnose`` calls.
    """

    def __init__(
        self,
        *,
        config: ResidualAssociationConfig | None = None,
    ) -> None:
        if config is None:
            stored = ResidualAssociationConfig()
        elif isinstance(config, ResidualAssociationConfig):
            stored = config.model_copy(deep=True)
        else:
            raise TypeError(
                "config must be ResidualAssociationConfig or None, "
                f"got {type(config).__name__}"
            )
        self._config = stored

    @property
    def method(self) -> DiagnosisMethod:
        return DiagnosisMethod.RESIDUAL_ASSOCIATION

    def get_metadata(self) -> dict[str, ScalarMetadataValue]:
        return {
            "method": DiagnosisMethod.RESIDUAL_ASSOCIATION.value,
            "supported_scopes": ",".join(
                scope.value for scope in sorted(_SUPPORTED_SCOPES, key=str)
            ),
            "residual_column": self._config.residual_column,
            "absolute_residual_column": self._config.absolute_residual_column,
            "anomaly_score_column": self._config.anomaly_score_column,
            "anomaly_indicator_column": self._config.anomaly_indicator_column,
            "minimum_total_rows": int(self._config.minimum_total_rows),
            "minimum_reference_rows": int(self._config.minimum_reference_rows),
            "minimum_anomaly_rows": int(self._config.minimum_anomaly_rows),
            "minimum_scale": float(self._config.minimum_scale),
            "deviation_z_threshold": float(self._config.deviation_z_threshold),
            "direction_tolerance": float(self._config.direction_tolerance),
            "score_correlation_weight": float(self._config.score_correlation_weight),
            "group_deviation_weight": float(self._config.group_deviation_weight),
            "prevalence_weight": float(self._config.prevalence_weight),
            "include_zero_score_factors": bool(
                self._config.include_zero_score_factors
            ),
            "normalize_factor_scores": bool(self._config.normalize_factor_scores),
            "use_anomaly_score_ordering": bool(
                self._config.use_anomaly_score_ordering
            ),
            "fitted": False,
            "requires_training": False,
            "association_not_causation": True,
        }

    def diagnose(
        self,
        data: pl.DataFrame,
        *,
        request: DiagnosisRequest,
        explanation: ExplanationResult | None = None,
    ) -> DiagnosisResult | DiagnosisBatchResult:
        if not isinstance(data, pl.DataFrame):
            raise TypeError(
                f"data must be a polars.DataFrame, got {type(data).__name__}"
            )
        if not isinstance(request, DiagnosisRequest):
            raise TypeError(
                f"request must be DiagnosisRequest, got {type(request).__name__}"
            )
        if explanation is not None and not isinstance(explanation, ExplanationResult):
            raise TypeError(
                "explanation must be ExplanationResult or None, "
                f"got {type(explanation).__name__}"
            )

        if request.method is not DiagnosisMethod.RESIDUAL_ASSOCIATION:
            raise DataValidationError(
                "request.method must be RESIDUAL_ASSOCIATION, "
                f"got {request.method!r}"
            )
        if request.scope not in _SUPPORTED_SCOPES:
            raise DataValidationError(
                f"request.scope is not supported by this diagnoser: {request.scope!r}"
            )

        if data.height < self._config.minimum_total_rows:
            raise InsufficientDataError(
                "data has "
                f"{data.height} rows; requires at least "
                f"{self._config.minimum_total_rows}"
            )

        score_column = self._effective_score_column(request)
        indicator_column = self._effective_indicator_column(request)
        self._validate_required_columns(
            data,
            request=request,
            score_column=score_column,
            indicator_column=indicator_column,
        )
        row_ids = self._extract_row_ids(data, request.row_id_column)
        feature_matrix = self._extract_feature_matrix(data, request.feature_columns)
        residual_values = self._extract_signed_numeric_column(
            data,
            self._config.residual_column,
            column_label="residual",
            require_non_negative=False,
        )
        self._extract_signed_numeric_column(
            data,
            self._config.absolute_residual_column,
            column_label="absolute residual",
            require_non_negative=True,
        )
        score_values = self._extract_signed_numeric_column(
            data,
            score_column,
            column_label="anomaly score",
            require_non_negative=True,
        )
        indicator_anomaly_indices = self._indices_from_indicator(
            data,
            indicator_column,
        )
        reference_indices = self._reference_indices_from_indicator(
            data,
            indicator_column,
        )
        anomaly_indices, selected_events = self._resolve_anomaly_indices(
            request=request,
            row_ids=row_ids,
            indicator_anomaly_indices=indicator_anomaly_indices,
        )
        explanation_unused = explanation is not None

        if request.scope is DiagnosisScope.SINGLE_EVENT:
            return self._diagnose_single_or_group(
                feature_matrix=feature_matrix,
                residual_values=residual_values,
                score_values=score_values,
                anomaly_indices=anomaly_indices,
                reference_indices=reference_indices,
                request=request,
                score_column=score_column,
                anomaly_id=selected_events[0].anomaly_id,
                analyzed_row_count=1,
                scope=request.scope,
                explanation_unused=explanation_unused,
                allow_single_anomaly_row=True,
            ).result

        if request.scope in {DiagnosisScope.ANOMALY_GROUP, DiagnosisScope.GLOBAL}:
            return self._diagnose_single_or_group(
                feature_matrix=feature_matrix,
                residual_values=residual_values,
                score_values=score_values,
                anomaly_indices=anomaly_indices,
                reference_indices=reference_indices,
                request=request,
                score_column=score_column,
                anomaly_id=None,
                analyzed_row_count=len(anomaly_indices),
                scope=request.scope,
                explanation_unused=explanation_unused,
                allow_single_anomaly_row=False,
            ).result

        ordered_events = self._select_top_events(selected_events, request=request)
        return self._diagnose_top_anomalies(
            feature_matrix=feature_matrix,
            residual_values=residual_values,
            score_values=score_values,
            row_ids=row_ids,
            reference_indices=reference_indices,
            ordered_events=ordered_events,
            request=request,
            score_column=score_column,
            explanation_unused=explanation_unused,
        )

    def _effective_score_column(self, request: DiagnosisRequest) -> str:
        if request.anomaly_score_column is not None:
            return request.anomaly_score_column
        return self._config.anomaly_score_column

    def _effective_indicator_column(self, request: DiagnosisRequest) -> str:
        if request.anomaly_indicator_column is not None:
            return request.anomaly_indicator_column
        return self._config.anomaly_indicator_column

    def _validate_required_columns(
        self,
        data: pl.DataFrame,
        *,
        request: DiagnosisRequest,
        score_column: str,
        indicator_column: str,
    ) -> None:
        required = [
            request.row_id_column,
            *request.feature_columns,
            self._config.residual_column,
            self._config.absolute_residual_column,
            score_column,
            indicator_column,
        ]
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise DataValidationError(
                f"data is missing required columns: {missing}"
            )

    def _extract_row_ids(self, data: pl.DataFrame, row_id_column: str) -> list[object]:
        series = data.get_column(row_id_column)
        dtype = series.dtype
        if dtype == pl.Boolean:
            raise DataValidationError(
                f"row ID column {row_id_column!r} must not be Boolean"
            )
        if dtype in {pl.Float32, pl.Float64} or (
            dtype.is_float() if hasattr(dtype, "is_float") else False
        ):
            raise DataValidationError(
                f"row ID column {row_id_column!r} must not be float"
            )
        if not (dtype.is_integer() or dtype == pl.String):
            raise DataValidationError(
                f"row ID column {row_id_column!r} must be integer or string, "
                f"got {dtype}"
            )

        values = series.to_list()
        if any(value is None for value in values):
            raise DataValidationError(
                f"row ID column {row_id_column!r} must not contain nulls"
            )
        if len(values) != len(set(values)):
            raise DataValidationError(
                f"row ID column {row_id_column!r} must contain unique values"
            )
        for value in values:
            if isinstance(value, bool):
                raise DataValidationError(
                    f"row ID column {row_id_column!r} values must be int or str"
                )
            if isinstance(value, float):
                raise DataValidationError(
                    f"row ID column {row_id_column!r} values must be int or str"
                )
            if not isinstance(value, (int, str)):
                raise DataValidationError(
                    f"row ID column {row_id_column!r} values must be int or str, "
                    f"got {type(value).__name__}"
                )
        return values

    def _extract_feature_matrix(
        self,
        data: pl.DataFrame,
        feature_columns: list[str],
    ) -> dict[str, np.ndarray]:
        matrix: dict[str, np.ndarray] = {}
        for column in feature_columns:
            matrix[column] = self._extract_signed_numeric_column(
                data,
                column,
                column_label=f"feature {column}",
                require_non_negative=False,
                feature_name=column,
            )
        return matrix

    def _extract_signed_numeric_column(
        self,
        data: pl.DataFrame,
        column: str,
        *,
        column_label: str,
        require_non_negative: bool,
        feature_name: str | None = None,
    ) -> np.ndarray:
        series = data.get_column(column)
        dtype = series.dtype
        error_name = feature_name if feature_name is not None else column
        if dtype == pl.Boolean or not dtype.is_numeric():
            raise DataValidationError(
                f"{column_label} {error_name!r} must be numeric "
                f"(integer/float/Decimal); got dtype {dtype}"
            )

        values = series.to_numpy()
        array = np.asarray(values, dtype=float)
        if array.ndim != 1:
            raise DataValidationError(
                f"{column_label} {error_name!r} must be one-dimensional"
            )
        if np.isnan(array).any():
            raise DataValidationError(
                f"{column_label} {error_name!r} must not contain null or NaN values"
            )
        if np.isinf(array).any():
            raise DataValidationError(
                f"{column_label} {error_name!r} must not contain infinite values"
            )
        if require_non_negative and bool(np.any(array < 0.0)):
            raise DataValidationError(
                f"{column_label} {error_name!r} must contain only non-negative values"
            )
        return array

    def _indices_from_indicator(
        self,
        data: pl.DataFrame,
        column: str,
    ) -> list[int]:
        series = data.get_column(column)
        dtype = series.dtype
        values = series.to_list()
        indices: list[int] = []

        if dtype == pl.Boolean:
            for index, value in enumerate(values):
                if value is None:
                    raise DataValidationError(
                        f"anomaly indicator column {column!r} must not contain nulls"
                    )
                if value is True:
                    indices.append(index)
                elif value is False:
                    continue
                else:
                    raise DataValidationError(
                        f"anomaly indicator column {column!r} has invalid Boolean value"
                    )
            return indices

        if dtype.is_integer() and dtype != pl.Boolean:
            for index, value in enumerate(values):
                if value is None or isinstance(value, bool):
                    raise DataValidationError(
                        f"anomaly indicator column {column!r} must contain only 0/1"
                    )
                if value == 1:
                    indices.append(index)
                elif value == 0:
                    continue
                else:
                    raise DataValidationError(
                        f"anomaly indicator column {column!r} must contain only 0/1, "
                        f"got {value!r}"
                    )
            return indices

        raise DataValidationError(
            f"anomaly indicator column {column!r} must be Boolean or integer 0/1, "
            f"got dtype {dtype}"
        )

    def _reference_indices_from_indicator(
        self,
        data: pl.DataFrame,
        column: str,
    ) -> list[int]:
        series = data.get_column(column)
        dtype = series.dtype
        values = series.to_list()
        indices: list[int] = []

        if dtype == pl.Boolean:
            for index, value in enumerate(values):
                if value is None:
                    raise DataValidationError(
                        f"anomaly indicator column {column!r} must not contain nulls"
                    )
                if value is False:
                    indices.append(index)
                elif value is True:
                    continue
                else:
                    raise DataValidationError(
                        f"anomaly indicator column {column!r} has invalid Boolean value"
                    )
            return indices

        if dtype.is_integer() and dtype != pl.Boolean:
            for index, value in enumerate(values):
                if value is None or isinstance(value, bool):
                    raise DataValidationError(
                        f"anomaly indicator column {column!r} must contain only 0/1"
                    )
                if value == 0:
                    indices.append(index)
                elif value == 1:
                    continue
                else:
                    raise DataValidationError(
                        f"anomaly indicator column {column!r} must contain only 0/1, "
                        f"got {value!r}"
                    )
            return indices

        raise DataValidationError(
            f"anomaly indicator column {column!r} must be Boolean or integer 0/1, "
            f"got dtype {dtype}"
        )

    def _resolve_anomaly_indices(
        self,
        *,
        request: DiagnosisRequest,
        row_ids: list[object],
        indicator_anomaly_indices: list[int],
    ) -> tuple[list[int], list[AnomalyEvent]]:
        indicator_set = set(indicator_anomaly_indices)

        if request.anomaly_events:
            events = [event.model_copy(deep=True) for event in request.anomaly_events]
            indices = self._match_events_to_indices(events, row_ids)
            if not set(indices).issubset(indicator_set):
                raise DataValidationError(
                    "selected anomaly events must be residual anomalies according "
                    "to the effective anomaly indicator column "
                    "(event IDs must be a subset of indicator anomaly rows)"
                )
            return indices, events

        if request.scope in {DiagnosisScope.SINGLE_EVENT, DiagnosisScope.TOP_ANOMALIES}:
            raise DataValidationError(
                "anomaly_events must be provided for SINGLE_EVENT and "
                "TOP_ANOMALIES scopes"
            )
        if not indicator_anomaly_indices:
            raise InsufficientDataError(
                "anomaly subset must contain at least one residual anomaly row"
            )
        return list(indicator_anomaly_indices), []

    def _match_events_to_indices(
        self,
        events: list[AnomalyEvent],
        row_ids: list[object],
    ) -> list[int]:
        by_str: dict[str, int] = {}
        for index, row_id in enumerate(row_ids):
            key = str(row_id)
            if key in by_str:
                raise DataValidationError(
                    f"duplicate stringified row ID {key!r} prevents anomaly matching"
                )
            by_str[key] = index

        indices: list[int] = []
        seen_event_ids: set[str] = set()
        sample_is_int = bool(row_ids) and isinstance(row_ids[0], int) and not isinstance(
            row_ids[0], bool
        )

        for event in events:
            anomaly_id = event.anomaly_id
            if anomaly_id in seen_event_ids:
                raise DataValidationError(
                    f"duplicate anomaly_id in anomaly_events: {anomaly_id!r}"
                )
            seen_event_ids.add(anomaly_id)

            if anomaly_id not in by_str:
                if sample_is_int:
                    raise DataValidationError(
                        f"anomaly event ID {anomaly_id!r} is incompatible with "
                        "integer row IDs (or was not found in data)"
                    )
                raise DataValidationError(
                    f"anomaly event ID {anomaly_id!r} was not found in data "
                    f"row ID column"
                )
            indices.append(by_str[anomaly_id])

        if len(indices) != len(set(indices)):
            raise DataValidationError(
                "each anomaly event must match exactly one unique data row"
            )
        return indices

    def _select_top_events(
        self,
        events: list[AnomalyEvent],
        *,
        request: DiagnosisRequest,
    ) -> list[AnomalyEvent]:
        if not events:
            raise DataValidationError(
                "TOP_ANOMALIES requires at least one anomaly event"
            )
        indexed = list(enumerate(events))
        if self._config.use_anomaly_score_ordering:
            for _, event in indexed:
                score = event.anomaly_score
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    raise DataValidationError(
                        "anomaly_score must be a finite number for ordering"
                    )
                if not math.isfinite(float(score)):
                    raise DataValidationError(
                        "anomaly_score must be finite for ordering"
                    )
            indexed.sort(
                key=lambda item: (-float(item[1].anomaly_score), item[0]),
            )
        selected = [event for _, event in indexed[: request.top_k_events]]
        return selected

    def _diagnose_top_anomalies(
        self,
        *,
        feature_matrix: dict[str, np.ndarray],
        residual_values: np.ndarray,
        score_values: np.ndarray,
        row_ids: list[object],
        reference_indices: list[int],
        ordered_events: list[AnomalyEvent],
        request: DiagnosisRequest,
        score_column: str,
        explanation_unused: bool,
    ) -> DiagnosisBatchResult:
        id_to_index = {str(row_id): index for index, row_id in enumerate(row_ids)}
        payloads: list[_EventDiagnosisPayload] = []
        for event in ordered_events:
            index = id_to_index[str(event.anomaly_id)]
            payload = self._diagnose_single_or_group(
                feature_matrix=feature_matrix,
                residual_values=residual_values,
                score_values=score_values,
                anomaly_indices=[index],
                reference_indices=reference_indices,
                request=request,
                score_column=score_column,
                anomaly_id=event.anomaly_id,
                analyzed_row_count=1,
                scope=DiagnosisScope.TOP_ANOMALIES,
                explanation_unused=explanation_unused,
                allow_single_anomaly_row=True,
            )
            payloads.append(payload)

        results = [payload.result for payload in payloads]
        aggregate_factors = self._aggregate_factors_from_statistics(
            payloads,
            request=request,
        )
        warnings = [
            _BATCH_WARNING_ASSOCIATION,
            _BATCH_WARNING_AGGREGATED,
            _BATCH_WARNING_EFFECTS,
        ]
        if any(not result.factors for result in results):
            warnings.append(_BATCH_WARNING_EMPTY_FACTORS)

        return DiagnosisBatchResult(
            results=results,
            aggregate_factors=aggregate_factors,
            requested_event_count=len(ordered_events),
            diagnosed_event_count=len(results),
            failed_event_count=0,
            generated_at=datetime.now(UTC),
            warnings=warnings,
            metadata={
                "anomaly_row_count": len(ordered_events),
                "returned_result_count": len(results),
                "returned_aggregate_factor_count": len(aggregate_factors),
                "ranking_is_heuristic": True,
                "association_not_causation": True,
            },
        )

    def _diagnose_single_or_group(
        self,
        *,
        feature_matrix: dict[str, np.ndarray],
        residual_values: np.ndarray,
        score_values: np.ndarray,
        anomaly_indices: list[int],
        reference_indices: list[int],
        request: DiagnosisRequest,
        score_column: str,
        anomaly_id: str | None,
        analyzed_row_count: int,
        scope: DiagnosisScope,
        explanation_unused: bool,
        allow_single_anomaly_row: bool,
    ) -> _EventDiagnosisPayload:
        total_row_count = int(next(iter(feature_matrix.values())).shape[0])
        anomaly_set = set(anomaly_indices)
        if not anomaly_indices:
            raise InsufficientDataError("anomaly subset must contain at least one row")
        if len(anomaly_indices) != len(anomaly_set):
            raise DataValidationError("anomaly subset indices must be unique")
        if anomaly_set.intersection(reference_indices):
            raise DataValidationError(
                "anomaly subset and reference group must not overlap"
            )
        if not reference_indices:
            raise InsufficientDataError("reference group must not be empty")
        if len(reference_indices) < self._config.minimum_reference_rows:
            raise InsufficientDataError(
                "reference group has "
                f"{len(reference_indices)} rows; requires at least "
                f"{self._config.minimum_reference_rows}"
            )

        anomaly_row_count = len(anomaly_indices)
        if allow_single_anomaly_row:
            if anomaly_row_count < 1:
                raise InsufficientDataError(
                    "anomaly subset must contain at least one row"
                )
        elif anomaly_row_count < self._config.minimum_anomaly_rows:
            raise InsufficientDataError(
                "anomaly group has "
                f"{anomaly_row_count} rows; requires at least "
                f"{self._config.minimum_anomaly_rows}"
            )

        statistics = [
            self._compute_feature_statistic(
                feature_name=feature_name,
                values=feature_matrix[feature_name],
                residual_values=residual_values,
                score_values=score_values,
                anomaly_indices=anomaly_indices,
                reference_indices=reference_indices,
                total_row_count=total_row_count,
                request=request,
            )
            for feature_name in request.feature_columns
        ]

        near_zero = any(
            (
                _WARNING_NEAR_ZERO_MAD in statistic.warnings
                or _WARNING_CONSTANT_RANK in statistic.warnings
            )
            for statistic in statistics
        )
        factors = self._statistics_to_factors(
            statistics,
            request=request,
            feature_order=list(request.feature_columns),
        )
        caveats = self._build_caveats(
            near_zero_or_constant=near_zero,
            explanation_unused=explanation_unused,
            has_factors=bool(factors),
        )
        if factors:
            result_confidence = float(
                sum(factor.confidence for factor in factors) / len(factors)
            )
        else:
            result_confidence = 0.0

        result = DiagnosisResult(
            anomaly_id=anomaly_id,
            task=request.task,
            method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
            scope=scope,
            factors=factors,
            confidence=result_confidence,
            analyzed_row_count=analyzed_row_count,
            reference_row_count=len(reference_indices),
            caveats=caveats,
            generated_at=datetime.now(UTC),
            metadata={
                "anomaly_row_count": anomaly_row_count,
                "reference_row_count": len(reference_indices),
                "total_row_count": total_row_count,
                "evaluated_feature_count": len(request.feature_columns),
                "returned_factor_count": len(factors),
                "residual_column": self._config.residual_column,
                "anomaly_score_column": score_column,
                "ranking_is_heuristic": True,
                "association_not_causation": True,
            },
        )
        return _EventDiagnosisPayload(
            result=result,
            statistics=tuple(statistics),
        )

    def _compute_feature_statistic(
        self,
        *,
        feature_name: str,
        values: np.ndarray,
        residual_values: np.ndarray,
        score_values: np.ndarray,
        anomaly_indices: list[int],
        reference_indices: list[int],
        total_row_count: int,
        request: DiagnosisRequest,
    ) -> ResidualFeatureStatistic:
        anomaly_values = values[np.asarray(anomaly_indices, dtype=int)]
        reference_values = values[np.asarray(reference_indices, dtype=int)]

        residual_corr, residual_constant = _spearman_rank_correlation(
            values,
            residual_values,
            minimum_scale=self._config.minimum_scale,
        )
        score_corr, score_constant = _spearman_rank_correlation(
            values,
            score_values,
            minimum_scale=self._config.minimum_scale,
        )

        reference_median = float(np.median(reference_values))
        anomaly_median = float(np.median(anomaly_values))
        signed_location_difference = anomaly_median - reference_median
        median_absolute_deviation = float(
            np.median(np.abs(reference_values - reference_median))
        )
        robust_scale = _ROBUST_SCALE_CONSTANT * median_absolute_deviation
        warnings: list[str] = []
        if residual_constant or score_constant:
            warnings.append(_WARNING_CONSTANT_RANK)
        if robust_scale <= self._config.minimum_scale:
            effective_scale = float(self._config.minimum_scale)
            warnings.append(_WARNING_NEAR_ZERO_MAD)
        else:
            effective_scale = float(robust_scale)

        group_deviation_z = abs(signed_location_difference) / effective_scale
        if not math.isfinite(group_deviation_z):
            raise DataValidationError(
                f"feature {feature_name!r} produced a non-finite group deviation z"
            )
        row_deviation = np.abs(anomaly_values - reference_median) / effective_scale
        deviation_prevalence = float(
            np.mean(row_deviation >= self._config.deviation_z_threshold)
        )
        bounded_group_strength = float(
            group_deviation_z / (group_deviation_z + 1.0)
        )
        score_correlation_strength = abs(score_corr)
        raw_association_score = (
            self._config.score_correlation_weight * score_correlation_strength
            + self._config.group_deviation_weight * bounded_group_strength
            + self._config.prevalence_weight * deviation_prevalence
        )
        if not math.isfinite(raw_association_score):
            raise DataValidationError(
                f"feature {feature_name!r} produced a non-finite association score"
            )
        raw_association_score = float(min(1.0, max(0.0, raw_association_score)))

        direction = self._resolve_direction(
            anomaly_values=anomaly_values,
            reference_median=reference_median,
            residual_correlation=residual_corr,
        )
        confidence = self._compute_confidence(
            raw_association_score=raw_association_score,
            total_row_count=total_row_count,
            reference_row_count=len(reference_indices),
            anomaly_row_count=len(anomaly_indices),
            request=request,
        )

        # Deduplicate warnings while preserving order.
        unique_warnings: list[str] = []
        seen_warnings: set[str] = set()
        for warning in warnings:
            if warning not in seen_warnings:
                seen_warnings.add(warning)
                unique_warnings.append(warning)

        return ResidualFeatureStatistic(
            feature_name=feature_name,
            total_row_count=total_row_count,
            anomaly_row_count=len(anomaly_indices),
            reference_row_count=len(reference_indices),
            feature_residual_rank_correlation=float(residual_corr),
            feature_score_rank_correlation=float(score_corr),
            reference_median=reference_median,
            anomaly_median=anomaly_median,
            signed_location_difference=float(signed_location_difference),
            median_absolute_deviation=median_absolute_deviation,
            effective_scale=effective_scale,
            group_deviation_z=float(group_deviation_z),
            deviation_prevalence=float(deviation_prevalence),
            bounded_group_strength=float(bounded_group_strength),
            raw_association_score=float(raw_association_score),
            normalized_association_score=0.0,
            direction=direction,
            confidence=float(confidence),
            warnings=unique_warnings,
        )

    def _resolve_direction(
        self,
        *,
        anomaly_values: np.ndarray,
        reference_median: float,
        residual_correlation: float,
    ) -> str:
        tolerance = self._config.direction_tolerance
        has_high = bool(np.any(anomaly_values > reference_median + tolerance))
        has_low = bool(np.any(anomaly_values < reference_median - tolerance))
        if (
            has_high
            and has_low
            and abs(residual_correlation) <= tolerance
        ):
            return _DIRECTION_MIXED
        if residual_correlation > tolerance:
            return _DIRECTION_POSITIVE
        if residual_correlation < -tolerance:
            return _DIRECTION_NEGATIVE
        return _DIRECTION_UNKNOWN

    def _compute_confidence(
        self,
        *,
        raw_association_score: float,
        total_row_count: int,
        reference_row_count: int,
        anomaly_row_count: int,
        request: DiagnosisRequest,
    ) -> float:
        association_strength = float(raw_association_score)
        total_support = min(
            1.0,
            total_row_count / self._config.minimum_total_rows,
        )
        reference_support = min(
            1.0,
            reference_row_count / self._config.minimum_reference_rows,
        )
        if request.scope in {
            DiagnosisScope.SINGLE_EVENT,
            DiagnosisScope.TOP_ANOMALIES,
        }:
            anomaly_support = 1.0
        else:
            anomaly_support = min(
                1.0,
                anomaly_row_count / self._config.minimum_anomaly_rows,
            )
        confidence = (
            association_strength
            * total_support
            * reference_support
            * anomaly_support
        )
        return float(min(1.0, max(0.0, confidence)))

    def _statistics_to_factors(
        self,
        statistics: list[ResidualFeatureStatistic],
        *,
        request: DiagnosisRequest,
        feature_order: list[str],
    ) -> list[RootCauseFactor]:
        order_index = {name: index for index, name in enumerate(feature_order)}
        ranked = list(statistics)
        if not self._config.include_zero_score_factors:
            ranked = [
                item for item in ranked if item.raw_association_score != 0.0
            ]
        if not ranked:
            return []

        max_raw = max(item.raw_association_score for item in ranked)
        normalized_stats: list[ResidualFeatureStatistic] = []
        for item in ranked:
            if self._config.normalize_factor_scores:
                if max_raw > 0.0:
                    normalized = item.raw_association_score / max_raw
                else:
                    normalized = 0.0
            else:
                # Raw score is already in [0, 1]; reuse it as the schema score.
                normalized = item.raw_association_score
            normalized_stats.append(
                item.model_copy(
                    update={"normalized_association_score": float(normalized)}
                )
            )

        normalized_stats.sort(
            key=lambda item: (
                -item.raw_association_score,
                -abs(item.feature_score_rank_correlation),
                -item.deviation_prevalence,
                -item.confidence,
                order_index.get(item.feature_name, len(order_index)),
            )
        )
        selected = normalized_stats[: request.top_k_factors]
        return [self._to_root_cause_factor(item) for item in selected]

    def _to_root_cause_factor(
        self,
        statistic: ResidualFeatureStatistic,
    ) -> RootCauseFactor:
        summary = self._direction_summary(statistic.direction)
        evidence = (
            f"{summary} "
            f"feature_residual_rank_correlation="
            f"{statistic.feature_residual_rank_correlation:.6g}; "
            f"feature_score_rank_correlation="
            f"{statistic.feature_score_rank_correlation:.6g}; "
            f"reference_median={statistic.reference_median:.6g}; "
            f"anomaly_median={statistic.anomaly_median:.6g}; "
            f"signed_location_difference={statistic.signed_location_difference:.6g}; "
            f"group_deviation_z={statistic.group_deviation_z:.6g}; "
            f"deviation_prevalence={statistic.deviation_prevalence:.6g}; "
            f"raw_association_score={statistic.raw_association_score:.6g}; "
            f"normalized_association_score="
            f"{statistic.normalized_association_score:.6g}; "
            f"anomaly_row_count={statistic.anomaly_row_count}; "
            f"reference_row_count={statistic.reference_row_count}. "
            "This association may reflect process behavior, sensor effects, "
            "omitted variables, or model misspecification and requires verification."
        )
        return RootCauseFactor(
            variable=statistic.feature_name,
            direction=statistic.direction,
            deviation=float(statistic.signed_location_difference),
            role=ColumnRole.UNKNOWN,
            controllable=False,
            evidence=evidence,
            confidence=float(statistic.confidence),
            needs_verification=True,
        )

    @staticmethod
    def _direction_summary(direction: str) -> str:
        base = (
            "This feature was associated with residual anomaly score variation "
            "and showed repeated deviation from the normal reference group."
        )
        if direction == _DIRECTION_POSITIVE:
            return (
                f"{base} Higher values were associated with more positive "
                "regression residuals."
            )
        if direction == _DIRECTION_NEGATIVE:
            return (
                f"{base} Higher values were associated with more negative "
                "regression residuals."
            )
        if direction == _DIRECTION_MIXED:
            return (
                f"{base} Both high and low deviations appeared among residual "
                "anomalies."
            )
        return (
            f"{base} Signed residual association direction was not stable."
        )

    def _build_caveats(
        self,
        *,
        near_zero_or_constant: bool,
        explanation_unused: bool,
        has_factors: bool,
    ) -> list[str]:
        caveats = [
            _CAVEAT_ASSOCIATION,
            _CAVEAT_MISSPECIFICATION,
            _CAVEAT_DETECTOR,
        ]
        if near_zero_or_constant:
            caveats.append(_CAVEAT_NEAR_ZERO)
        if explanation_unused:
            caveats.append(_CAVEAT_EXPLANATION)
        if not has_factors:
            caveats.append(_CAVEAT_NO_FACTORS)
        unique: list[str] = []
        seen: set[str] = set()
        for item in caveats:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        return unique

    def _aggregate_factors_from_statistics(
        self,
        payloads: list[_EventDiagnosisPayload],
        *,
        request: DiagnosisRequest,
    ) -> list[RootCauseFactor]:
        feature_order = {
            name: index for index, name in enumerate(request.feature_columns)
        }
        buckets: dict[str, _AggregateBucket] = {}
        for payload in payloads:
            ranked_stats = self._ranked_statistics_for_result(
                list(payload.statistics),
                request=request,
            )
            for statistic in ranked_stats:
                if statistic.feature_name not in {
                    factor.variable for factor in payload.result.factors
                }:
                    continue
                bucket = buckets.setdefault(statistic.feature_name, _AggregateBucket())
                bucket.normalized_scores.append(
                    float(statistic.normalized_association_score)
                )
                bucket.confidences.append(float(statistic.confidence))
                bucket.abs_score_correlations.append(
                    abs(float(statistic.feature_score_rank_correlation))
                )
                bucket.prevalences.append(float(statistic.deviation_prevalence))
                bucket.directions.append(str(statistic.direction))
                bucket.deviations.append(float(statistic.signed_location_difference))
                bucket.count += 1

        aggregates: list[
            tuple[float, float, float, int, int, RootCauseFactor]
        ] = []
        for variable, bucket in buckets.items():
            mean_score = (
                float(sum(bucket.normalized_scores) / len(bucket.normalized_scores))
                if bucket.normalized_scores
                else 0.0
            )
            mean_confidence = (
                float(sum(bucket.confidences) / len(bucket.confidences))
                if bucket.confidences
                else 0.0
            )
            mean_abs_corr = (
                float(
                    sum(bucket.abs_score_correlations)
                    / len(bucket.abs_score_correlations)
                )
                if bucket.abs_score_correlations
                else 0.0
            )
            mean_prevalence = (
                float(sum(bucket.prevalences) / len(bucket.prevalences))
                if bucket.prevalences
                else 0.0
            )
            direction = self._aggregate_direction(bucket.directions)
            mean_deviation = (
                float(sum(bucket.deviations) / len(bucket.deviations))
                if bucket.deviations
                else None
            )
            evidence = (
                f"Aggregated residual-associated factor across {bucket.count} "
                f"diagnosed residual anomaly events. "
                f"mean_normalized_association_score={mean_score:.6g}; "
                f"mean_abs_feature_score_rank_correlation={mean_abs_corr:.6g}; "
                f"mean_deviation_prevalence={mean_prevalence:.6g}; "
                f"mean_confidence={mean_confidence:.6g}; "
                f"occurrence_count={bucket.count}. "
                "Residual association does not establish process causation."
            )
            factor = RootCauseFactor(
                variable=variable,
                direction=direction,
                deviation=mean_deviation,
                role=ColumnRole.UNKNOWN,
                controllable=False,
                evidence=evidence,
                confidence=mean_confidence,
                needs_verification=True,
            )
            aggregates.append(
                (
                    mean_score,
                    mean_abs_corr,
                    mean_confidence,
                    bucket.count,
                    feature_order.get(variable, len(feature_order)),
                    factor,
                )
            )

        aggregates.sort(
            key=lambda item: (-item[0], -item[1], -item[2], -item[3], item[4])
        )
        return [item[5] for item in aggregates[: request.top_k_factors]]

    def _ranked_statistics_for_result(
        self,
        statistics: list[ResidualFeatureStatistic],
        *,
        request: DiagnosisRequest,
    ) -> list[ResidualFeatureStatistic]:
        """Apply the same filtering/normalization used for public factors."""
        order_index = {
            name: index for index, name in enumerate(request.feature_columns)
        }
        ranked = list(statistics)
        if not self._config.include_zero_score_factors:
            ranked = [
                item for item in ranked if item.raw_association_score != 0.0
            ]
        if not ranked:
            return []
        max_raw = max(item.raw_association_score for item in ranked)
        normalized_stats: list[ResidualFeatureStatistic] = []
        for item in ranked:
            if self._config.normalize_factor_scores:
                if max_raw > 0.0:
                    normalized = item.raw_association_score / max_raw
                else:
                    normalized = 0.0
            else:
                normalized = item.raw_association_score
            normalized_stats.append(
                item.model_copy(
                    update={"normalized_association_score": float(normalized)}
                )
            )
        normalized_stats.sort(
            key=lambda item: (
                -item.raw_association_score,
                -abs(item.feature_score_rank_correlation),
                -item.deviation_prevalence,
                -item.confidence,
                order_index.get(item.feature_name, len(order_index)),
            )
        )
        return normalized_stats[: request.top_k_factors]

    @staticmethod
    def _aggregate_direction(directions: list[str]) -> str:
        known = [
            direction
            for direction in directions
            if direction in {_DIRECTION_POSITIVE, _DIRECTION_NEGATIVE}
        ]
        if any(direction == _DIRECTION_MIXED for direction in directions):
            return _DIRECTION_MIXED
        if not known:
            return _DIRECTION_UNKNOWN
        unique = set(known)
        if unique == {_DIRECTION_POSITIVE}:
            return _DIRECTION_POSITIVE
        if unique == {_DIRECTION_NEGATIVE}:
            return _DIRECTION_NEGATIVE
        return _DIRECTION_MIXED
