"""Robust z-score and group-comparison root-cause diagnoser (Step 8B).

Associates likely driver features with anomaly rows by comparing them to a
normal reference group. Results describe association only—not causation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import polars as pl
from pydantic import BaseModel, Field, field_validator

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
_SUPPORTED_METHODS = frozenset(
    {
        DiagnosisMethod.ROBUST_Z_SCORE,
        DiagnosisMethod.GROUP_COMPARISON,
        DiagnosisMethod.ENSEMBLE,
    }
)
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

_CAVEAT_ASSOCIATION = (
    "Association does not establish causation; treat factors as likely "
    "drivers requiring process review."
)
_CAVEAT_REFERENCE = (
    "Robust scores depend on the selected normal reference group."
)
_CAVEAT_NEAR_ZERO = (
    "At least one feature had near-zero reference dispersion; "
    "minimum_scale was applied."
)
_CAVEAT_EXPLANATION = (
    "Model explanation was supplied but not combined in this method."
)
_CAVEAT_NO_FACTORS = (
    "No associated factors met the scoring criteria for this diagnosis."
)
_CAVEAT_ENSEMBLE_HEURISTIC = (
    "Ensemble association scores are a deterministic ranking heuristic, "
    "not a causal estimate or probability."
)
_WARNING_NEAR_ZERO_DISPERSION = "reference dispersion was near zero"

_BATCH_WARNING_ASSOCIATION = (
    "Association does not establish causation across diagnosed events."
)
_BATCH_WARNING_AGGREGATED = (
    "Results are aggregated across individually diagnosed anomaly events."
)
_BATCH_WARNING_EMPTY_FACTORS = (
    "At least one diagnosed event returned no associated factors."
)


@dataclass
class _AggregateBucket:
    scores: list[float] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
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


class RobustGroupComparisonConfig(BaseModel):
    """Configuration for robust z-score and group-comparison diagnosis.

    Controls scale floors, deviation thresholds, direction tolerance, and
    ranking / normalization behavior. Values describe associative ranking
    heuristics only.
    """

    minimum_scale: float = 1e-12
    z_score_threshold: float = 3.5
    direction_tolerance: float = 1e-12
    minimum_anomaly_rows: int = 1
    include_zero_score_factors: bool = False
    normalize_factor_scores: bool = True
    use_anomaly_score_ordering: bool = True

    @field_validator("minimum_scale", mode="before")
    @classmethod
    def _validate_minimum_scale(cls, value: object) -> float:
        return _require_positive_finite_float(value, field_name="minimum_scale")

    @field_validator("z_score_threshold", mode="before")
    @classmethod
    def _validate_z_score_threshold(cls, value: object) -> float:
        return _require_positive_finite_float(value, field_name="z_score_threshold")

    @field_validator("direction_tolerance", mode="before")
    @classmethod
    def _validate_direction_tolerance(cls, value: object) -> float:
        return _require_non_negative_finite_float(
            value,
            field_name="direction_tolerance",
        )

    @field_validator("minimum_anomaly_rows", mode="before")
    @classmethod
    def _validate_minimum_anomaly_rows(cls, value: object) -> int:
        return _require_strict_int_ge1(value, field_name="minimum_anomaly_rows")

    @field_validator(
        "include_zero_score_factors",
        "normalize_factor_scores",
        "use_anomaly_score_ordering",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="bool field")


class RobustFeatureStatistic(BaseModel):
    """Per-feature robust association statistics for group comparison.

    Stores scalar summaries used to rank likely drivers. Does not embed
    estimators, DataFrames, or full arrays. Not required on public diagnosis
    payloads; useful for tests and optional scalar metadata.
    """

    feature_name: str
    reference_row_count: int
    anomaly_row_count: int
    reference_median: float
    anomaly_median: float
    signed_location_difference: float
    median_absolute_deviation: float
    effective_scale: float
    robust_z_score: float
    deviation_prevalence: float
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

    @field_validator("reference_row_count", "anomaly_row_count", mode="before")
    @classmethod
    def _validate_row_counts(cls, value: object) -> int:
        return _require_strict_int_ge1(value, field_name="row count")

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

    @field_validator("robust_z_score", "raw_association_score", mode="before")
    @classmethod
    def _validate_non_negative_scores(cls, value: object) -> float:
        return _require_non_negative_finite_float(value, field_name="score")

    @field_validator(
        "deviation_prevalence",
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


class RobustGroupComparisonDiagnoser(BaseRootCauseDiagnoser):
    """Concrete diagnoser using robust z-score and normal-vs-anomaly comparison.

    Ranks features by associative deviation from a normal reference group.
    Does not claim causation, does not fit models, and does not cache inputs
    or outputs between ``diagnose`` calls.
    """

    def __init__(
        self,
        *,
        config: RobustGroupComparisonConfig | None = None,
    ) -> None:
        if config is None:
            stored = RobustGroupComparisonConfig()
        elif isinstance(config, RobustGroupComparisonConfig):
            stored = config.model_copy(deep=True)
        else:
            raise TypeError(
                "config must be RobustGroupComparisonConfig or None, "
                f"got {type(config).__name__}"
            )
        self._config = stored

    @property
    def method(self) -> DiagnosisMethod:
        return DiagnosisMethod.ENSEMBLE

    def get_metadata(self) -> dict[str, ScalarMetadataValue]:
        return {
            "method": DiagnosisMethod.ENSEMBLE.value,
            "supported_methods": ",".join(
                method.value for method in sorted(_SUPPORTED_METHODS, key=str)
            ),
            "supported_scopes": ",".join(
                scope.value for scope in sorted(_SUPPORTED_SCOPES, key=str)
            ),
            "minimum_scale": float(self._config.minimum_scale),
            "z_score_threshold": float(self._config.z_score_threshold),
            "direction_tolerance": float(self._config.direction_tolerance),
            "minimum_anomaly_rows": int(self._config.minimum_anomaly_rows),
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

        if request.method not in _SUPPORTED_METHODS:
            raise DataValidationError(
                "request.method must be ROBUST_Z_SCORE, GROUP_COMPARISON, or "
                f"ENSEMBLE, got {request.method!r}"
            )
        if request.scope not in _SUPPORTED_SCOPES:
            raise DataValidationError(
                f"request.scope is not supported by this diagnoser: {request.scope!r}"
            )

        if data.height < 1:
            raise InsufficientDataError("data must contain at least one row")

        self._validate_required_columns(data, request)
        row_ids = self._extract_row_ids(data, request.row_id_column)
        feature_matrix = self._extract_feature_matrix(data, request.feature_columns)

        anomaly_indices, selected_events = self._resolve_anomaly_indices(
            data,
            request=request,
            row_ids=row_ids,
        )
        explanation_unused = explanation is not None

        if request.scope is DiagnosisScope.SINGLE_EVENT:
            return self._diagnose_single_or_group(
                feature_matrix=feature_matrix,
                anomaly_indices=anomaly_indices,
                request=request,
                anomaly_id=selected_events[0].anomaly_id,
                analyzed_row_count=1,
                scope=request.scope,
                explanation_unused=explanation_unused,
                allow_single_anomaly_row=True,
            )

        if request.scope in {DiagnosisScope.ANOMALY_GROUP, DiagnosisScope.GLOBAL}:
            return self._diagnose_single_or_group(
                feature_matrix=feature_matrix,
                anomaly_indices=anomaly_indices,
                request=request,
                anomaly_id=None,
                analyzed_row_count=len(anomaly_indices),
                scope=request.scope,
                explanation_unused=explanation_unused,
                allow_single_anomaly_row=False,
            )

        # TOP_ANOMALIES
        ordered_events = self._select_top_events(selected_events, request=request)
        return self._diagnose_top_anomalies(
            feature_matrix=feature_matrix,
            row_ids=row_ids,
            ordered_events=ordered_events,
            request=request,
            explanation_unused=explanation_unused,
        )

    def _validate_required_columns(
        self,
        data: pl.DataFrame,
        request: DiagnosisRequest,
    ) -> None:
        required = [request.row_id_column, *request.feature_columns]
        if request.anomaly_events:
            # events are authoritative; indicator optional but if present checked later
            pass
        elif request.scope in {DiagnosisScope.ANOMALY_GROUP, DiagnosisScope.GLOBAL}:
            if request.anomaly_indicator_column is None:
                raise DataValidationError(
                    "anomaly_indicator_column is required when anomaly_events "
                    "is empty for ANOMALY_GROUP or GLOBAL scope"
                )
            required.append(request.anomaly_indicator_column)
        else:
            raise DataValidationError(
                "anomaly_events must be provided for SINGLE_EVENT and "
                "TOP_ANOMALIES scopes"
            )

        if request.anomaly_score_column is not None:
            # optional; only required if present in request naming (must exist if set)
            required.append(request.anomaly_score_column)
        if request.anomaly_indicator_column is not None and request.anomaly_events:
            required.append(request.anomaly_indicator_column)
        if request.target_column is not None:
            # target is not used for scoring but must exist if declared
            required.append(request.target_column)

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
            series = data.get_column(column)
            dtype = series.dtype
            if dtype == pl.Boolean or not dtype.is_numeric():
                raise DataValidationError(
                    f"feature {column!r} must be numeric "
                    f"(integer/float); got dtype {dtype}"
                )

            values = series.to_numpy()
            array = np.asarray(values, dtype=float)
            if array.ndim != 1:
                raise DataValidationError(
                    f"feature {column!r} must be one-dimensional"
                )
            if np.isnan(array).any():
                raise DataValidationError(
                    f"feature {column!r} must not contain null or NaN values"
                )
            if np.isinf(array).any():
                raise DataValidationError(
                    f"feature {column!r} must not contain infinite values"
                )
            matrix[column] = array
        return matrix

    def _resolve_anomaly_indices(
        self,
        data: pl.DataFrame,
        *,
        request: DiagnosisRequest,
        row_ids: list[object],
    ) -> tuple[list[int], list[AnomalyEvent]]:
        if request.anomaly_events:
            events = [event.model_copy(deep=True) for event in request.anomaly_events]
            indices = self._match_events_to_indices(events, row_ids)
            if request.anomaly_indicator_column is not None:
                indicator_indices = self._indices_from_indicator(
                    data,
                    request.anomaly_indicator_column,
                )
                if set(indices) != set(indicator_indices):
                    raise DataValidationError(
                        "anomaly_events row IDs must exactly match "
                        "anomaly_indicator_column True/1 rows when both are provided"
                    )
            return indices, events

        if request.anomaly_indicator_column is None:
            raise DataValidationError(
                "anomaly_indicator_column is required when anomaly_events is empty"
            )
        indices = self._indices_from_indicator(
            data,
            request.anomaly_indicator_column,
        )
        return indices, []

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
        row_ids: list[object],
        ordered_events: list[AnomalyEvent],
        request: DiagnosisRequest,
        explanation_unused: bool,
    ) -> DiagnosisBatchResult:
        id_to_index = {str(row_id): index for index, row_id in enumerate(row_ids)}
        results: list[DiagnosisResult] = []
        for event in ordered_events:
            index = id_to_index[str(event.anomaly_id)]
            result = self._diagnose_single_or_group(
                feature_matrix=feature_matrix,
                anomaly_indices=[index],
                request=request,
                anomaly_id=event.anomaly_id,
                analyzed_row_count=1,
                scope=DiagnosisScope.TOP_ANOMALIES,
                explanation_unused=explanation_unused,
                allow_single_anomaly_row=True,
            )
            results.append(result)

        aggregate_factors = self._aggregate_factors(results, request=request)
        warnings = [
            _BATCH_WARNING_ASSOCIATION,
            _BATCH_WARNING_AGGREGATED,
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
            },
        )

    def _diagnose_single_or_group(
        self,
        *,
        feature_matrix: dict[str, np.ndarray],
        anomaly_indices: list[int],
        request: DiagnosisRequest,
        anomaly_id: str | None,
        analyzed_row_count: int,
        scope: DiagnosisScope,
        explanation_unused: bool,
        allow_single_anomaly_row: bool,
    ) -> DiagnosisResult:
        row_count = next(iter(feature_matrix.values())).shape[0]
        anomaly_set = set(anomaly_indices)
        if not anomaly_indices:
            raise InsufficientDataError("anomaly subset must contain at least one row")
        if len(anomaly_indices) != len(anomaly_set):
            raise DataValidationError("anomaly subset indices must be unique")

        reference_indices = [
            index for index in range(row_count) if index not in anomaly_set
        ]
        if not reference_indices:
            raise InsufficientDataError("reference group must not be empty")
        if len(reference_indices) < request.minimum_reference_rows:
            raise InsufficientDataError(
                "reference group has "
                f"{len(reference_indices)} rows; requires at least "
                f"{request.minimum_reference_rows}"
            )

        anomaly_row_count = len(anomaly_indices)
        if allow_single_anomaly_row:
            if anomaly_row_count < 1:
                raise InsufficientDataError("anomaly subset must contain at least one row")
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
                anomaly_indices=anomaly_indices,
                reference_indices=reference_indices,
                request=request,
            )
            for feature_name in request.feature_columns
        ]

        near_zero = any(
            _WARNING_NEAR_ZERO_DISPERSION in statistic.warnings
            for statistic in statistics
        )
        factors = self._statistics_to_factors(
            statistics,
            request=request,
            feature_order=list(request.feature_columns),
        )
        method_used = self._method_used(request.method)
        caveats = self._build_caveats(
            near_zero_dispersion=near_zero,
            explanation_unused=explanation_unused,
            has_factors=bool(factors),
            method=request.method,
        )
        if factors:
            result_confidence = float(
                sum(factor.confidence for factor in factors) / len(factors)
            )
        else:
            result_confidence = 0.0

        return DiagnosisResult(
            anomaly_id=anomaly_id,
            task=request.task,
            method_used=method_used,
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
                "evaluated_feature_count": len(request.feature_columns),
                "returned_factor_count": len(factors),
                "z_score_threshold": float(self._config.z_score_threshold),
                "minimum_scale": float(self._config.minimum_scale),
                "ranking_is_heuristic": True,
            },
        )

    def _compute_feature_statistic(
        self,
        *,
        feature_name: str,
        values: np.ndarray,
        anomaly_indices: list[int],
        reference_indices: list[int],
        request: DiagnosisRequest,
    ) -> RobustFeatureStatistic:
        anomaly_values = values[np.asarray(anomaly_indices, dtype=int)]
        reference_values = values[np.asarray(reference_indices, dtype=int)]

        reference_median = float(np.median(reference_values))
        anomaly_median = float(np.median(anomaly_values))
        signed_location_difference = anomaly_median - reference_median
        median_absolute_deviation = float(
            np.median(np.abs(reference_values - reference_median))
        )
        robust_scale = _ROBUST_SCALE_CONSTANT * median_absolute_deviation
        warnings: list[str] = []
        if robust_scale <= self._config.minimum_scale:
            effective_scale = float(self._config.minimum_scale)
            warnings.append(_WARNING_NEAR_ZERO_DISPERSION)
        else:
            effective_scale = float(robust_scale)

        row_robust_z = np.abs(anomaly_values - reference_median) / effective_scale
        robust_z_score = abs(signed_location_difference) / effective_scale
        if not math.isfinite(robust_z_score):
            raise DataValidationError(
                f"feature {feature_name!r} produced a non-finite robust z-score"
            )
        deviation_prevalence = float(
            np.mean(row_robust_z >= self._config.z_score_threshold)
        )
        raw_association_score = self._raw_association_score(
            robust_z_score=robust_z_score,
            deviation_prevalence=deviation_prevalence,
            method=request.method,
        )
        if not math.isfinite(raw_association_score):
            raise DataValidationError(
                f"feature {feature_name!r} produced a non-finite association score"
            )

        direction = self._resolve_direction(
            anomaly_values=anomaly_values,
            reference_median=reference_median,
            signed_location_difference=signed_location_difference,
        )
        single_event_like = request.scope in {
            DiagnosisScope.SINGLE_EVENT,
            DiagnosisScope.TOP_ANOMALIES,
        } and len(anomaly_indices) == 1
        confidence = self._compute_confidence(
            robust_z_score=robust_z_score,
            deviation_prevalence=deviation_prevalence,
            reference_row_count=len(reference_indices),
            anomaly_row_count=len(anomaly_indices),
            request=request,
            allow_single_anomaly_row=single_event_like,
        )

        return RobustFeatureStatistic(
            feature_name=feature_name,
            reference_row_count=len(reference_indices),
            anomaly_row_count=len(anomaly_indices),
            reference_median=reference_median,
            anomaly_median=anomaly_median,
            signed_location_difference=signed_location_difference,
            median_absolute_deviation=median_absolute_deviation,
            effective_scale=effective_scale,
            robust_z_score=float(robust_z_score),
            deviation_prevalence=float(deviation_prevalence),
            raw_association_score=float(raw_association_score),
            normalized_association_score=0.0,
            direction=direction,
            confidence=float(confidence),
            warnings=warnings,
        )

    def _raw_association_score(
        self,
        *,
        robust_z_score: float,
        deviation_prevalence: float,
        method: DiagnosisMethod,
    ) -> float:
        if method is DiagnosisMethod.ENSEMBLE:
            return robust_z_score * (0.5 + 0.5 * deviation_prevalence)
        return float(robust_z_score)

    def _resolve_direction(
        self,
        *,
        anomaly_values: np.ndarray,
        reference_median: float,
        signed_location_difference: float,
    ) -> str:
        tolerance = self._config.direction_tolerance
        has_high = bool(np.any(anomaly_values > reference_median + tolerance))
        has_low = bool(np.any(anomaly_values < reference_median - tolerance))
        if (
            has_high
            and has_low
            and abs(signed_location_difference) <= tolerance
        ):
            return _DIRECTION_MIXED
        if signed_location_difference > tolerance:
            return _DIRECTION_POSITIVE
        if signed_location_difference < -tolerance:
            return _DIRECTION_NEGATIVE
        return _DIRECTION_UNKNOWN

    def _compute_confidence(
        self,
        *,
        robust_z_score: float,
        deviation_prevalence: float,
        reference_row_count: int,
        anomaly_row_count: int,
        request: DiagnosisRequest,
        allow_single_anomaly_row: bool,
    ) -> float:
        effect_strength = robust_z_score / (robust_z_score + 1.0)
        reference_support = min(
            1.0,
            reference_row_count / request.minimum_reference_rows,
        )
        if allow_single_anomaly_row or request.scope in {
            DiagnosisScope.SINGLE_EVENT,
            DiagnosisScope.TOP_ANOMALIES,
        }:
            anomaly_support = 1.0
        else:
            anomaly_support = min(
                1.0,
                anomaly_row_count / self._config.minimum_anomaly_rows,
            )
        if request.method in {
            DiagnosisMethod.GROUP_COMPARISON,
            DiagnosisMethod.ENSEMBLE,
        }:
            repeatability = 0.5 + 0.5 * deviation_prevalence
        else:
            repeatability = 1.0
        confidence = (
            effect_strength * reference_support * anomaly_support * repeatability
        )
        return float(min(1.0, max(0.0, confidence)))

    def _statistics_to_factors(
        self,
        statistics: list[RobustFeatureStatistic],
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
        normalized_stats: list[RobustFeatureStatistic] = []
        for item in ranked:
            if self._config.normalize_factor_scores:
                if max_raw > 0.0:
                    normalized = item.raw_association_score / max_raw
                else:
                    normalized = 0.0
            else:
                normalized = item.raw_association_score / (
                    item.raw_association_score + 1.0
                )
            normalized_stats.append(
                item.model_copy(
                    update={"normalized_association_score": float(normalized)}
                )
            )

        normalized_stats.sort(
            key=lambda item: (
                -item.raw_association_score,
                -item.deviation_prevalence,
                -item.confidence,
                order_index.get(item.feature_name, len(order_index)),
            )
        )
        selected = normalized_stats[: request.top_k_factors]
        return [self._to_root_cause_factor(item) for item in selected]

    def _to_root_cause_factor(
        self,
        statistic: RobustFeatureStatistic,
    ) -> RootCauseFactor:
        summary = self._direction_summary(statistic.direction)
        evidence = (
            f"{summary} "
            f"reference_median={statistic.reference_median:.6g}; "
            f"anomaly_median={statistic.anomaly_median:.6g}; "
            f"signed_location_difference={statistic.signed_location_difference:.6g}; "
            f"MAD={statistic.median_absolute_deviation:.6g}; "
            f"effective_scale={statistic.effective_scale:.6g}; "
            f"robust_z_score={statistic.robust_z_score:.6g}; "
            f"deviation_prevalence={statistic.deviation_prevalence:.6g}; "
            f"raw_association_score={statistic.raw_association_score:.6g}; "
            f"normalized_association_score={statistic.normalized_association_score:.6g}; "
            f"reference_row_count={statistic.reference_row_count}; "
            f"anomaly_row_count={statistic.anomaly_row_count}. "
            "The association should be reviewed with process context before "
            "action is taken."
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
        if direction == _DIRECTION_POSITIVE:
            return (
                "This feature showed one of the largest robust deviations from "
                "the reference group. Higher values were associated with the "
                "analyzed anomaly subset."
            )
        if direction == _DIRECTION_NEGATIVE:
            return (
                "This feature showed one of the largest robust deviations from "
                "the reference group. Lower values were associated with the "
                "analyzed anomaly subset."
            )
        if direction == _DIRECTION_MIXED:
            return (
                "This feature showed one of the largest robust deviations from "
                "the reference group. The direction was mixed across the "
                "analyzed anomaly rows."
            )
        return (
            "This feature showed a robust deviation from the reference group, "
            "but the direction relative to the reference median was unclear."
        )

    def _method_used(self, method: DiagnosisMethod) -> list[DiagnosisMethod]:
        if method is DiagnosisMethod.ROBUST_Z_SCORE:
            return [DiagnosisMethod.ROBUST_Z_SCORE]
        if method is DiagnosisMethod.GROUP_COMPARISON:
            return [DiagnosisMethod.GROUP_COMPARISON]
        return [
            DiagnosisMethod.ROBUST_Z_SCORE,
            DiagnosisMethod.GROUP_COMPARISON,
        ]

    def _build_caveats(
        self,
        *,
        near_zero_dispersion: bool,
        explanation_unused: bool,
        has_factors: bool,
        method: DiagnosisMethod,
    ) -> list[str]:
        caveats = [_CAVEAT_ASSOCIATION, _CAVEAT_REFERENCE]
        if method is DiagnosisMethod.ENSEMBLE:
            caveats.append(_CAVEAT_ENSEMBLE_HEURISTIC)
        if near_zero_dispersion:
            caveats.append(_CAVEAT_NEAR_ZERO)
        if explanation_unused:
            caveats.append(_CAVEAT_EXPLANATION)
        if not has_factors:
            caveats.append(_CAVEAT_NO_FACTORS)
        # Deduplicate while preserving order.
        unique: list[str] = []
        seen: set[str] = set()
        for item in caveats:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        return unique

    def _aggregate_factors(
        self,
        results: list[DiagnosisResult],
        *,
        request: DiagnosisRequest,
    ) -> list[RootCauseFactor]:
        feature_order = {
            name: index for index, name in enumerate(request.feature_columns)
        }
        buckets: dict[str, _AggregateBucket] = {}
        for result in results:
            for factor in result.factors:
                bucket = buckets.setdefault(factor.variable, _AggregateBucket())
                # RootCauseFactor has no normalized score field; recover from evidence.
                normalized = self._parse_normalized_score(factor.evidence)
                bucket.scores.append(normalized)
                bucket.confidences.append(float(factor.confidence))
                bucket.directions.append(str(factor.direction))
                if factor.deviation is not None:
                    bucket.deviations.append(float(factor.deviation))
                bucket.count += 1

        aggregates: list[tuple[float, float, int, int, RootCauseFactor]] = []
        for variable, bucket in buckets.items():
            mean_score = (
                float(sum(bucket.scores) / len(bucket.scores))
                if bucket.scores
                else 0.0
            )
            mean_confidence = (
                float(sum(bucket.confidences) / len(bucket.confidences))
                if bucket.confidences
                else 0.0
            )
            direction = self._aggregate_direction(bucket.directions)
            mean_deviation = (
                float(sum(bucket.deviations) / len(bucket.deviations))
                if bucket.deviations
                else None
            )
            evidence = (
                f"Aggregated associated factor across {bucket.count} diagnosed "
                f"anomaly events. mean_normalized_association_score={mean_score:.6g}; "
                f"mean_confidence={mean_confidence:.6g}; "
                f"occurrence_count={bucket.count}. "
                "Association does not establish causation."
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
                    mean_confidence,
                    bucket.count,
                    feature_order.get(variable, len(feature_order)),
                    factor,
                )
            )

        aggregates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
        return [item[4] for item in aggregates[: request.top_k_factors]]

    @staticmethod
    def _parse_normalized_score(evidence: str) -> float:
        marker = "normalized_association_score="
        if marker not in evidence:
            return 0.0
        fragment = evidence.split(marker, 1)[1]
        token = fragment.split(";", 1)[0].strip()
        try:
            value = float(token)
        except ValueError:
            return 0.0
        if not math.isfinite(value):
            return 0.0
        return float(min(1.0, max(0.0, value)))

    @staticmethod
    def _aggregate_direction(directions: list[str]) -> str:
        known = [
            direction
            for direction in directions
            if direction in {_DIRECTION_POSITIVE, _DIRECTION_NEGATIVE}
        ]
        if not known:
            if any(direction == _DIRECTION_MIXED for direction in directions):
                return _DIRECTION_MIXED
            return _DIRECTION_UNKNOWN
        unique = set(known)
        if unique == {_DIRECTION_POSITIVE}:
            return _DIRECTION_POSITIVE
        if unique == {_DIRECTION_NEGATIVE}:
            return _DIRECTION_NEGATIVE
        return _DIRECTION_MIXED
