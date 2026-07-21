"""Diagnosis ensemble aggregator combining robust and residual diagnosers (Step 8D).

Fuses per-feature rankings from child diagnosers with weighted reciprocal-rank
fusion. Results describe association only—not causation.
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
    ProcessIntelligenceError,
)
from process_intelligence.core.schemas import ExplanationResult, RootCauseFactor
from process_intelligence.diagnosis.enums import DiagnosisMethod, DiagnosisScope
from process_intelligence.diagnosis.protocols import BaseRootCauseDiagnoser
from process_intelligence.diagnosis.residual_association import (
    ResidualAssociationDiagnoser,
)
from process_intelligence.diagnosis.robust_group_comparison import (
    RobustGroupComparisonDiagnoser,
)
from process_intelligence.diagnosis.schemas import (
    DiagnosisBatchResult,
    DiagnosisRequest,
    DiagnosisResult,
    ScalarMetadataValue,
)

_SOURCE_ROBUST = "ROBUST_GROUP_COMPARISON"
_SOURCE_RESIDUAL = "RESIDUAL_ASSOCIATION"
_ORIGINAL_ROW_ID = "_original_row_id"

_DIRECTION_POSITIVE = "POSITIVE"
_DIRECTION_NEGATIVE = "NEGATIVE"
_DIRECTION_MIXED = "MIXED"
_DIRECTION_UNKNOWN = "UNKNOWN"
_ALLOWED_DIRECTIONS = frozenset(
    {
        _DIRECTION_POSITIVE,
        _DIRECTION_NEGATIVE,
        _DIRECTION_MIXED,
        _DIRECTION_UNKNOWN,
    }
)

_CANONICAL_METHOD_ORDER = (
    DiagnosisMethod.ROBUST_Z_SCORE,
    DiagnosisMethod.GROUP_COMPARISON,
    DiagnosisMethod.RESIDUAL_ASSOCIATION,
)

_CAVEAT_ASSOCIATION = (
    "ensemble attribution reflects association and does not establish causation"
)
_CAVEAT_RRF = (
    "ranking combines method-specific factor order using weighted "
    "reciprocal-rank fusion"
)
_CAVEAT_ROBUST_REF = (
    "robust comparison depends on the selected reference group"
)
_CAVEAT_RESIDUAL = (
    "residual association can reflect process behavior, sensor effects, "
    "omitted variables, data shift, or model misspecification"
)
_CAVEAT_NO_SUPPORT = (
    "no factors met the configured minimum method support for this ensemble"
)
_CAVEAT_NO_FACTORS = (
    "no associated factors were returned by the ensemble diagnosis"
)
_CAVEAT_DIRECTION_CONFLICT = (
    "at least one ensemble factor has conflicting direction evidence across methods"
)
_CAVEAT_PARTIAL_SUPPORT = (
    "at least one ensemble factor has only partial method support"
)

_BATCH_WARNING_ASSOCIATION = (
    "ensemble attribution does not establish causation"
)
_BATCH_WARNING_AGGREGATED = (
    "results aggregate individually diagnosed anomaly events"
)
_BATCH_WARNING_RRF = (
    "weighted reciprocal-rank fusion uses ranking order rather than "
    "calibrated probabilities"
)
_BATCH_WARNING_PARTIAL = "partial child method success"
_BATCH_WARNING_DIRECTION = "direction conflict present among ensemble factors"
_BATCH_WARNING_EMPTY_EVENT = "at least one diagnosed event returned no factors"
_BATCH_WARNING_EMPTY_AGGREGATE = "no aggregate factors were returned"

_ISOLATABLE_EXCEPTIONS = (
    ProcessIntelligenceError,
    ValueError,
    TypeError,
    ArithmeticError,
    np.linalg.LinAlgError,
)
_NON_ISOLATABLE_EXCEPTIONS = (
    KeyboardInterrupt,
    SystemExit,
    GeneratorExit,
    MemoryError,
)


@dataclass
class _SourceFactorView:
    source_key: str
    method: DiagnosisMethod
    rank: int
    factor: RootCauseFactor
    weight: float


@dataclass
class _FeatureBucket:
    sources: list[_SourceFactorView] = field(default_factory=list)


@dataclass
class _ChildOutcome:
    source_key: str
    method: DiagnosisMethod
    weight: float
    result: DiagnosisResult | DiagnosisBatchResult | None = None
    error: BaseException | None = None

    @property
    def succeeded(self) -> bool:
        return self.result is not None and self.error is None


@dataclass
class _AggregateEventBucket:
    ranks: list[int] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    deviations: list[float] = field(default_factory=list)
    deviation_weights: list[float] = field(default_factory=list)
    directions: list[str] = field(default_factory=list)
    roles: list[ColumnRole] = field(default_factory=list)
    controllables: list[bool] = field(default_factory=list)
    needs_verification_flags: list[bool] = field(default_factory=list)
    direction_conflicts: list[bool] = field(default_factory=list)


def _require_strict_bool(value: object, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a bool, got {type(value).__name__}")
    return value


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


def _require_non_negative_finite_float(value: object, *, field_name: str) -> float:
    number = _require_finite_float(value, field_name=field_name)
    if number < 0.0:
        raise ValueError(f"{field_name} must be >= 0, got {number}")
    return number


def _require_positive_finite_float(value: object, *, field_name: str) -> float:
    number = _require_finite_float(value, field_name=field_name)
    if number <= 0.0:
        raise ValueError(f"{field_name} must be > 0, got {number}")
    return number


def _require_non_empty_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be str, got {type(value).__name__}")
    if value == "" or value.strip() == "":
        raise ValueError(f"{field_name} must be a non-empty, non-whitespace string")
    return value


def _require_strict_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{field_name} must be an int (bool not allowed), "
            f"got {type(value).__name__}"
        )
    return value


def _normalize_direction(value: str) -> str:
    text = value.strip()
    if text in _ALLOWED_DIRECTIONS:
        return text
    return _DIRECTION_UNKNOWN


def _combine_directions(directions: list[str]) -> tuple[str, bool]:
    normalized = [_normalize_direction(item) for item in directions]
    has_mixed = _DIRECTION_MIXED in normalized
    has_positive = _DIRECTION_POSITIVE in normalized
    has_negative = _DIRECTION_NEGATIVE in normalized
    known = {
        item
        for item in normalized
        if item in {_DIRECTION_POSITIVE, _DIRECTION_NEGATIVE, _DIRECTION_MIXED}
    }

    conflict = has_mixed or (has_positive and has_negative)

    if has_mixed or (has_positive and has_negative):
        return _DIRECTION_MIXED, conflict
    if known == {_DIRECTION_POSITIVE} or (
        has_positive and not has_negative and not has_mixed
    ):
        return _DIRECTION_POSITIVE, conflict
    if known == {_DIRECTION_NEGATIVE} or (
        has_negative and not has_positive and not has_mixed
    ):
        return _DIRECTION_NEGATIVE, conflict
    if not known:
        return _DIRECTION_UNKNOWN, conflict
    return _DIRECTION_MIXED, conflict


def _combine_roles(roles: list[ColumnRole]) -> ColumnRole:
    if not roles:
        return ColumnRole.UNKNOWN
    first = roles[0]
    if all(role is first for role in roles):
        return first
    return ColumnRole.UNKNOWN


def _dedupe_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _short_error_message(exc: BaseException) -> str:
    text = str(exc).strip()
    if not text:
        return type(exc).__name__
    if len(text) > 200:
        return text[:197] + "..."
    return text


def _assert_scalar_metadata(metadata: dict[str, ScalarMetadataValue]) -> None:
    for key, value in metadata.items():
        if isinstance(value, (pl.DataFrame, np.ndarray)):
            raise ProcessIntelligenceError(
                f"child metadata[{key!r}] must not contain DataFrame or ndarray"
            )
        if value is None or isinstance(value, (str, bool)):
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            continue
        if isinstance(value, float) and math.isfinite(value):
            continue
        raise ProcessIntelligenceError(
            f"child metadata[{key!r}] must be scalar-only, "
            f"got {type(value).__name__}"
        )


class DiagnosisEnsembleConfig(BaseModel):
    """Configuration for weighted reciprocal-rank fusion ensemble diagnosis.

    Weights and fusion constants control associative ranking across child
    diagnosers. They are not causal probabilities or calibrated accuracies.
    """

    robust_method_weight: float = 0.5
    residual_method_weight: float = 0.5
    reciprocal_rank_constant: float = 60.0
    minimum_method_support: int = 1
    continue_on_method_failure: bool = True
    require_residual_method: bool = False
    include_child_evidence: bool = True
    normalize_ensemble_scores: bool = True
    penalize_partial_method_support: bool = True

    @field_validator("robust_method_weight", "residual_method_weight", mode="before")
    @classmethod
    def _validate_weights(cls, value: object) -> float:
        return _require_unit_interval(value, field_name="method weight")

    @field_validator("reciprocal_rank_constant", mode="before")
    @classmethod
    def _validate_rrf_constant(cls, value: object) -> float:
        return _require_positive_finite_float(
            value,
            field_name="reciprocal_rank_constant",
        )

    @field_validator("minimum_method_support", mode="before")
    @classmethod
    def _validate_minimum_support(cls, value: object) -> int:
        number = _require_strict_int(value, field_name="minimum_method_support")
        if number not in {1, 2}:
            raise ValueError(
                f"minimum_method_support must be 1 or 2, got {number}"
            )
        return number

    @field_validator(
        "continue_on_method_failure",
        "require_residual_method",
        "include_child_evidence",
        "normalize_ensemble_scores",
        "penalize_partial_method_support",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="bool field")

    @model_validator(mode="after")
    def _validate_weight_sum(self) -> Self:
        if self.robust_method_weight == 0.0 and self.residual_method_weight == 0.0:
            raise ValueError(
                "robust_method_weight and residual_method_weight must not both be 0"
            )
        return self


class EnsembleFactorStatistic(BaseModel):
    """Per-feature ensemble fusion statistics across child diagnoser sources.

    Stores scalar ranking and agreement summaries. Does not embed estimators,
    DataFrames, or full arrays. Ranking authority remains the factor list order
    on diagnosis results; this model is for tests and optional introspection.
    """

    feature_name: str
    source_method_count: int
    source_methods: list[DiagnosisMethod]
    source_ranks: dict[str, int]
    source_confidences: dict[str, float]
    weighted_rrf_score: float
    normalized_ensemble_score: float
    combined_confidence: float
    combined_direction: str
    combined_deviation: float
    direction_conflict: bool
    needs_verification: bool
    warnings: list[str] = Field(default_factory=list)

    @field_validator("feature_name", mode="before")
    @classmethod
    def _validate_feature_name(cls, value: object) -> str:
        name = _require_non_empty_str(value, field_name="feature_name")
        if name == _ORIGINAL_ROW_ID:
            raise ValueError(f"feature_name must not be {_ORIGINAL_ROW_ID!r}")
        return name

    @field_validator("source_method_count", mode="before")
    @classmethod
    def _validate_source_count(cls, value: object) -> int:
        number = _require_strict_int(value, field_name="source_method_count")
        if number < 1 or number > 2:
            raise ValueError(
                f"source_method_count must be between 1 and 2, got {number}"
            )
        return number

    @field_validator("source_methods", mode="before")
    @classmethod
    def _validate_source_methods_before(cls, value: object) -> list[object]:
        if not isinstance(value, list):
            raise ValueError(
                f"source_methods must be a list[DiagnosisMethod], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("source_methods", mode="after")
    @classmethod
    def _validate_source_methods(
        cls,
        value: list[DiagnosisMethod],
    ) -> list[DiagnosisMethod]:
        cleaned: list[DiagnosisMethod] = []
        seen: set[DiagnosisMethod] = set()
        for item in value:
            if not isinstance(item, DiagnosisMethod):
                raise ValueError(
                    "source_methods entries must be DiagnosisMethod, "
                    f"got {type(item).__name__}"
                )
            if item in seen:
                raise ValueError(
                    f"source_methods must not contain duplicates: {item!r}"
                )
            seen.add(item)
            cleaned.append(item)
        return cleaned

    @field_validator("source_ranks", mode="before")
    @classmethod
    def _validate_source_ranks_before(cls, value: object) -> dict[str, int]:
        if not isinstance(value, dict):
            raise ValueError(
                f"source_ranks must be a dict[str, int], got {type(value).__name__}"
            )
        cleaned: dict[str, int] = {}
        for key, raw in value.items():
            text = _require_non_empty_str(key, field_name="source_ranks key")
            rank = _require_strict_int(raw, field_name="source rank")
            if rank < 1:
                raise ValueError(f"source rank must be >= 1, got {rank}")
            cleaned[text] = rank
        return cleaned

    @field_validator("source_confidences", mode="before")
    @classmethod
    def _validate_source_confidences_before(
        cls,
        value: object,
    ) -> dict[str, float]:
        if not isinstance(value, dict):
            raise ValueError(
                "source_confidences must be a dict[str, float], "
                f"got {type(value).__name__}"
            )
        cleaned: dict[str, float] = {}
        for key, raw in value.items():
            text = _require_non_empty_str(key, field_name="source_confidences key")
            cleaned[text] = _require_unit_interval(raw, field_name="source confidence")
        return cleaned

    @field_validator("weighted_rrf_score", mode="before")
    @classmethod
    def _validate_weighted_score(cls, value: object) -> float:
        return _require_non_negative_finite_float(
            value,
            field_name="weighted_rrf_score",
        )

    @field_validator(
        "normalized_ensemble_score",
        "combined_confidence",
        mode="before",
    )
    @classmethod
    def _validate_unit_fields(cls, value: object) -> float:
        return _require_unit_interval(value, field_name="unit-interval field")

    @field_validator("combined_direction", mode="before")
    @classmethod
    def _validate_direction(cls, value: object) -> str:
        text = _require_non_empty_str(value, field_name="combined_direction")
        if text not in _ALLOWED_DIRECTIONS:
            raise ValueError(
                f"combined_direction must be one of {sorted(_ALLOWED_DIRECTIONS)}, "
                f"got {text!r}"
            )
        return text

    @field_validator("combined_deviation", mode="before")
    @classmethod
    def _validate_deviation(cls, value: object) -> float:
        return _require_finite_float(value, field_name="combined_deviation")

    @field_validator("direction_conflict", "needs_verification", mode="before")
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="bool field")

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
    def _validate_source_consistency(self) -> Self:
        if self.source_method_count != len(self.source_methods):
            raise ValueError(
                "source_method_count must equal the number of source_methods "
                f"({self.source_method_count} != {len(self.source_methods)})"
            )
        if set(self.source_ranks) != set(self.source_confidences):
            raise ValueError(
                "source_ranks and source_confidences keys must match"
            )
        if len(self.source_ranks) != self.source_method_count:
            raise ValueError(
                "source_method_count must equal the number of source rank entries"
            )
        allowed_keys = {_SOURCE_ROBUST, _SOURCE_RESIDUAL}
        for key in self.source_ranks:
            if key not in allowed_keys:
                raise ValueError(
                    f"source key must be one of {sorted(allowed_keys)}, got {key!r}"
                )
        return self


class DiagnosisEnsembleDiagnoser(BaseRootCauseDiagnoser):
    """Ensemble diagnoser fusing robust-group and residual-association results.

    Combines child factor rankings with weighted reciprocal-rank fusion.
    Does not claim causation, does not retrain models, and does not cache
    inputs or outputs between ``diagnose`` calls.
    """

    def __init__(
        self,
        *,
        config: DiagnosisEnsembleConfig | None = None,
        robust_diagnoser: BaseRootCauseDiagnoser | None = None,
        residual_diagnoser: BaseRootCauseDiagnoser | None = None,
    ) -> None:
        if config is None:
            stored_config = DiagnosisEnsembleConfig()
        elif isinstance(config, DiagnosisEnsembleConfig):
            stored_config = config.model_copy(deep=True)
        else:
            raise TypeError(
                "config must be DiagnosisEnsembleConfig or None, "
                f"got {type(config).__name__}"
            )

        if robust_diagnoser is None:
            stored_robust: BaseRootCauseDiagnoser = RobustGroupComparisonDiagnoser()
        else:
            if not isinstance(robust_diagnoser, BaseRootCauseDiagnoser):
                raise TypeError(
                    "robust_diagnoser must be a BaseRootCauseDiagnoser instance, "
                    f"got {type(robust_diagnoser).__name__}"
                )
            if robust_diagnoser.method is not DiagnosisMethod.ENSEMBLE:
                raise TypeError(
                    "robust_diagnoser.method must be DiagnosisMethod.ENSEMBLE, "
                    f"got {robust_diagnoser.method!r}"
                )
            stored_robust = robust_diagnoser

        if residual_diagnoser is None:
            stored_residual: BaseRootCauseDiagnoser = ResidualAssociationDiagnoser()
        else:
            if not isinstance(residual_diagnoser, BaseRootCauseDiagnoser):
                raise TypeError(
                    "residual_diagnoser must be a BaseRootCauseDiagnoser instance, "
                    f"got {type(residual_diagnoser).__name__}"
                )
            if residual_diagnoser.method is not DiagnosisMethod.RESIDUAL_ASSOCIATION:
                raise TypeError(
                    "residual_diagnoser.method must be "
                    "DiagnosisMethod.RESIDUAL_ASSOCIATION, "
                    f"got {residual_diagnoser.method!r}"
                )
            stored_residual = residual_diagnoser

        self._config = stored_config
        self._robust_diagnoser = stored_robust
        self._residual_diagnoser = stored_residual

    @property
    def method(self) -> DiagnosisMethod:
        return DiagnosisMethod.ENSEMBLE

    def get_metadata(self) -> dict[str, ScalarMetadataValue]:
        return {
            "method": DiagnosisMethod.ENSEMBLE.value,
            "robust_source": _SOURCE_ROBUST,
            "residual_source": _SOURCE_RESIDUAL,
            "robust_method_weight": float(self._config.robust_method_weight),
            "residual_method_weight": float(self._config.residual_method_weight),
            "reciprocal_rank_constant": float(self._config.reciprocal_rank_constant),
            "minimum_method_support": int(self._config.minimum_method_support),
            "continue_on_method_failure": bool(
                self._config.continue_on_method_failure
            ),
            "require_residual_method": bool(self._config.require_residual_method),
            "include_child_evidence": bool(self._config.include_child_evidence),
            "normalize_ensemble_scores": bool(self._config.normalize_ensemble_scores),
            "penalize_partial_method_support": bool(
                self._config.penalize_partial_method_support
            ),
            "fitted": False,
            "requires_training": False,
            "ranking_method": "weighted_reciprocal_rank_fusion",
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
        if request.method is not DiagnosisMethod.ENSEMBLE:
            raise DataValidationError(
                "request.method must be ENSEMBLE, "
                f"got {request.method!r}"
            )
        if data.height < 1:
            raise DataValidationError("data must contain at least one row")
        if len(request.feature_columns) < 1:
            raise DataValidationError("feature_columns must contain at least one item")

        robust_request = self._build_child_request(
            request,
            method=DiagnosisMethod.ENSEMBLE,
        )
        residual_request = self._build_child_request(
            request,
            method=DiagnosisMethod.RESIDUAL_ASSOCIATION,
        )

        outcomes = self._run_children(
            data=data,
            request=request,
            robust_request=robust_request,
            residual_request=residual_request,
            explanation=explanation,
        )

        if request.scope is DiagnosisScope.TOP_ANOMALIES:
            return self._combine_batch(
                request=request,
                outcomes=outcomes,
                explanation=explanation,
            )
        return self._combine_single(
            request=request,
            outcomes=outcomes,
            explanation=explanation,
        )

    def _build_child_request(
        self,
        request: DiagnosisRequest,
        *,
        method: DiagnosisMethod,
    ) -> DiagnosisRequest:
        return DiagnosisRequest(
            task=request.task,
            method=method,
            scope=request.scope,
            feature_columns=list(request.feature_columns),
            anomaly_events=[
                event.model_copy(deep=True) for event in request.anomaly_events
            ],
            target_column=request.target_column,
            anomaly_indicator_column=request.anomaly_indicator_column,
            anomaly_score_column=request.anomaly_score_column,
            row_id_column=request.row_id_column,
            top_k_events=request.top_k_events,
            top_k_factors=request.top_k_factors,
            minimum_reference_rows=request.minimum_reference_rows,
            metadata=dict(request.metadata),
        )

    def _run_children(
        self,
        *,
        data: pl.DataFrame,
        request: DiagnosisRequest,
        robust_request: DiagnosisRequest,
        residual_request: DiagnosisRequest,
        explanation: ExplanationResult | None,
    ) -> list[_ChildOutcome]:
        plan: list[tuple[str, DiagnosisMethod, float, BaseRootCauseDiagnoser, DiagnosisRequest]] = [
            (
                _SOURCE_ROBUST,
                DiagnosisMethod.ENSEMBLE,
                float(self._config.robust_method_weight),
                self._robust_diagnoser,
                robust_request,
            ),
            (
                _SOURCE_RESIDUAL,
                DiagnosisMethod.RESIDUAL_ASSOCIATION,
                float(self._config.residual_method_weight),
                self._residual_diagnoser,
                residual_request,
            ),
        ]

        outcomes: list[_ChildOutcome] = []
        for source_key, method, weight, diagnoser, child_request in plan:
            outcome = _ChildOutcome(
                source_key=source_key,
                method=method,
                weight=weight,
            )
            try:
                result = diagnoser.diagnose(
                    data,
                    request=child_request,
                    explanation=explanation,
                )
                self._validate_child_result(
                    result,
                    request=request,
                    source_key=source_key,
                )
                outcome.result = result
            except _NON_ISOLATABLE_EXCEPTIONS:
                raise
            except _ISOLATABLE_EXCEPTIONS as exc:
                outcome.error = exc
                if not self._config.continue_on_method_failure:
                    raise
            outcomes.append(outcome)

        self._enforce_failure_policy(outcomes)
        return outcomes

    def _enforce_failure_policy(self, outcomes: list[_ChildOutcome]) -> None:
        successes = [item for item in outcomes if item.succeeded]
        failures = [item for item in outcomes if not item.succeeded]

        residual = next(
            item for item in outcomes if item.source_key == _SOURCE_RESIDUAL
        )
        if self._config.require_residual_method and not residual.succeeded:
            message = self._format_failure_message(failures)
            raise ProcessIntelligenceError(
                "residual method is required but failed: " + message
            )

        if not successes:
            raise ProcessIntelligenceError(
                "all ensemble child methods failed: "
                + self._format_failure_message(failures)
            )

        if len(successes) < self._config.minimum_method_support:
            raise ProcessIntelligenceError(
                "successful ensemble method count "
                f"{len(successes)} is below minimum_method_support "
                f"{self._config.minimum_method_support}"
            )

    def _format_failure_message(self, failures: list[_ChildOutcome]) -> str:
        parts: list[str] = []
        for item in failures:
            exc = item.error
            if exc is None:
                parts.append(f"{item.source_key}: unknown failure")
            else:
                parts.append(
                    f"{item.source_key}: {type(exc).__name__}: "
                    f"{_short_error_message(exc)}"
                )
        return "; ".join(parts)

    def _validate_child_result(
        self,
        result: DiagnosisResult | DiagnosisBatchResult,
        *,
        request: DiagnosisRequest,
        source_key: str,
    ) -> None:
        if request.scope is DiagnosisScope.TOP_ANOMALIES:
            if not isinstance(result, DiagnosisBatchResult):
                raise ProcessIntelligenceError(
                    f"{source_key} must return DiagnosisBatchResult for "
                    f"TOP_ANOMALIES, got {type(result).__name__}"
                )
            self._validate_batch_child(result, request=request, source_key=source_key)
            return

        if not isinstance(result, DiagnosisResult):
            raise ProcessIntelligenceError(
                f"{source_key} must return DiagnosisResult for "
                f"{request.scope.value}, got {type(result).__name__}"
            )
        self._validate_single_child(result, request=request, source_key=source_key)

    def _validate_single_child(
        self,
        result: DiagnosisResult,
        *,
        request: DiagnosisRequest,
        source_key: str,
    ) -> None:
        if result.task is not request.task:
            raise ProcessIntelligenceError(
                f"{source_key} result.task mismatch: {result.task!r}"
            )
        if result.scope is not request.scope:
            raise ProcessIntelligenceError(
                f"{source_key} result.scope mismatch: {result.scope!r}"
            )
        if not math.isfinite(float(result.confidence)):
            raise ProcessIntelligenceError(
                f"{source_key} result.confidence must be finite"
            )
        if result.confidence < 0.0 or result.confidence > 1.0:
            raise ProcessIntelligenceError(
                f"{source_key} result.confidence must be in [0.0, 1.0]"
            )
        if (
            result.generated_at.tzinfo is None
            or result.generated_at.tzinfo.utcoffset(result.generated_at) is None
        ):
            raise ProcessIntelligenceError(
                f"{source_key} generated_at must be timezone-aware"
            )
        _assert_scalar_metadata(result.metadata)
        self._validate_factors(result.factors, request=request, source_key=source_key)

        if request.scope is DiagnosisScope.SINGLE_EVENT:
            expected = request.anomaly_events[0].anomaly_id
            if result.anomaly_id != expected:
                raise ProcessIntelligenceError(
                    f"{source_key} anomaly_id must equal {expected!r}, "
                    f"got {result.anomaly_id!r}"
                )

    def _validate_batch_child(
        self,
        result: DiagnosisBatchResult,
        *,
        request: DiagnosisRequest,
        source_key: str,
    ) -> None:
        if (
            result.generated_at.tzinfo is None
            or result.generated_at.tzinfo.utcoffset(result.generated_at) is None
        ):
            raise ProcessIntelligenceError(
                f"{source_key} generated_at must be timezone-aware"
            )
        _assert_scalar_metadata(result.metadata)

        requested_ids = {event.anomaly_id for event in request.anomaly_events}
        seen: set[str | None] = set()
        for item in result.results:
            if item.anomaly_id in seen:
                raise ProcessIntelligenceError(
                    f"{source_key} batch results contain duplicate anomaly_id "
                    f"{item.anomaly_id!r}"
                )
            seen.add(item.anomaly_id)
            if item.anomaly_id not in requested_ids:
                raise ProcessIntelligenceError(
                    f"{source_key} batch anomaly_id {item.anomaly_id!r} "
                    "is not in the request events"
                )
            if item.task is not request.task:
                raise ProcessIntelligenceError(
                    f"{source_key} batch result.task mismatch"
                )
            if item.scope is not request.scope:
                raise ProcessIntelligenceError(
                    f"{source_key} batch result.scope mismatch"
                )
            if not math.isfinite(float(item.confidence)):
                raise ProcessIntelligenceError(
                    f"{source_key} batch result.confidence must be finite"
                )
            if item.confidence < 0.0 or item.confidence > 1.0:
                raise ProcessIntelligenceError(
                    f"{source_key} batch result.confidence must be in [0.0, 1.0]"
                )
            if (
                item.generated_at.tzinfo is None
                or item.generated_at.tzinfo.utcoffset(item.generated_at) is None
            ):
                raise ProcessIntelligenceError(
                    f"{source_key} batch generated_at must be timezone-aware"
                )
            _assert_scalar_metadata(item.metadata)
            self._validate_factors(
                item.factors,
                request=request,
                source_key=source_key,
            )

    def _validate_factors(
        self,
        factors: list[RootCauseFactor],
        *,
        request: DiagnosisRequest,
        source_key: str,
    ) -> None:
        seen: set[str] = set()
        feature_set = set(request.feature_columns)
        for factor in factors:
            if factor.variable in seen:
                raise ProcessIntelligenceError(
                    f"{source_key} factors contain duplicate variable "
                    f"{factor.variable!r}"
                )
            seen.add(factor.variable)
            if factor.variable not in feature_set:
                raise ProcessIntelligenceError(
                    f"{source_key} factor variable {factor.variable!r} "
                    "is not in request.feature_columns"
                )
            if not math.isfinite(float(factor.confidence)):
                raise ProcessIntelligenceError(
                    f"{source_key} factor confidence must be finite"
                )
            if factor.confidence < 0.0 or factor.confidence > 1.0:
                raise ProcessIntelligenceError(
                    f"{source_key} factor confidence must be in [0.0, 1.0]"
                )

    def _successful_outcomes(
        self,
        outcomes: list[_ChildOutcome],
    ) -> list[_ChildOutcome]:
        return [item for item in outcomes if item.succeeded]

    def _normalized_weights(
        self,
        successes: list[_ChildOutcome],
    ) -> dict[str, float]:
        weight_sum = float(sum(item.weight for item in successes))
        if weight_sum <= 0.0:
            raise ProcessIntelligenceError(
                "successful ensemble method weight sum must be > 0"
            )
        if len(successes) == 1:
            return {successes[0].source_key: 1.0}
        return {
            item.source_key: float(item.weight) / weight_sum for item in successes
        }

    def _combine_single(
        self,
        *,
        request: DiagnosisRequest,
        outcomes: list[_ChildOutcome],
        explanation: ExplanationResult | None,
    ) -> DiagnosisResult:
        successes = self._successful_outcomes(outcomes)
        child_results: list[tuple[_ChildOutcome, DiagnosisResult]] = []
        for item in successes:
            result = item.result
            if not isinstance(result, DiagnosisResult):
                raise ProcessIntelligenceError(
                    f"{item.source_key} expected DiagnosisResult"
                )
            child_results.append((item, result))

        self._assert_aligned_single_results(child_results)

        factors, statistics, fusion_notes = self._fuse_factor_lists(
            [
                (
                    outcome,
                    list(result.factors),
                )
                for outcome, result in child_results
            ],
            request=request,
        )

        first = child_results[0][1]
        method_used = self._merge_method_used(
            [result.method_used for _, result in child_results]
        )
        if factors:
            confidence = float(
                sum(factor.confidence for factor in factors) / len(factors)
            )
        else:
            confidence = 0.0

        caveats = self._build_result_caveats(
            outcomes=outcomes,
            child_results=[result for _, result in child_results],
            statistics=statistics,
            explanation=explanation,
            has_factors=bool(factors),
            fusion_notes=fusion_notes,
        )
        metadata = self._build_result_metadata(
            outcomes=outcomes,
            request=request,
            factors=factors,
            statistics=statistics,
        )

        return DiagnosisResult(
            anomaly_id=first.anomaly_id,
            task=request.task,
            method_used=method_used,
            scope=request.scope,
            factors=factors,
            confidence=confidence,
            analyzed_row_count=first.analyzed_row_count,
            reference_row_count=first.reference_row_count,
            caveats=caveats,
            generated_at=datetime.now(UTC),
            metadata=metadata,
        )

    def _assert_aligned_single_results(
        self,
        child_results: list[tuple[_ChildOutcome, DiagnosisResult]],
    ) -> None:
        if not child_results:
            raise ProcessIntelligenceError("no successful child results to combine")
        first = child_results[0][1]
        for outcome, result in child_results[1:]:
            if result.anomaly_id != first.anomaly_id:
                raise ProcessIntelligenceError(
                    f"{outcome.source_key} anomaly_id mismatch across children"
                )
            if result.task is not first.task:
                raise ProcessIntelligenceError(
                    f"{outcome.source_key} task mismatch across children"
                )
            if result.scope is not first.scope:
                raise ProcessIntelligenceError(
                    f"{outcome.source_key} scope mismatch across children"
                )
            if result.analyzed_row_count != first.analyzed_row_count:
                raise ProcessIntelligenceError(
                    f"{outcome.source_key} analyzed_row_count mismatch across children"
                )
            if result.reference_row_count != first.reference_row_count:
                raise ProcessIntelligenceError(
                    f"{outcome.source_key} reference_row_count mismatch across children"
                )

    def _combine_batch(
        self,
        *,
        request: DiagnosisRequest,
        outcomes: list[_ChildOutcome],
        explanation: ExplanationResult | None,
    ) -> DiagnosisBatchResult:
        successes = self._successful_outcomes(outcomes)
        batches: list[tuple[_ChildOutcome, DiagnosisBatchResult]] = []
        for item in successes:
            result = item.result
            if not isinstance(result, DiagnosisBatchResult):
                raise ProcessIntelligenceError(
                    f"{item.source_key} expected DiagnosisBatchResult"
                )
            batches.append((item, result))

        self._assert_aligned_batches(batches)

        event_ids = [item.anomaly_id for item in batches[0][1].results]
        per_event: list[DiagnosisResult] = []
        all_statistics: list[EnsembleFactorStatistic] = []
        for anomaly_id in event_ids:
            child_factor_lists: list[tuple[_ChildOutcome, list[RootCauseFactor]]] = []
            template: DiagnosisResult | None = None
            method_lists: list[list[DiagnosisMethod]] = []
            for outcome, batch in batches:
                match = next(
                    (
                        result
                        for result in batch.results
                        if result.anomaly_id == anomaly_id
                    ),
                    None,
                )
                if match is None:
                    raise ProcessIntelligenceError(
                        f"{outcome.source_key} missing anomaly_id {anomaly_id!r}"
                    )
                if template is None:
                    template = match
                child_factor_lists.append((outcome, list(match.factors)))
                method_lists.append(list(match.method_used))

            assert template is not None
            factors, statistics, fusion_notes = self._fuse_factor_lists(
                child_factor_lists,
                request=request,
            )
            all_statistics.extend(statistics)
            if factors:
                confidence = float(
                    sum(factor.confidence for factor in factors) / len(factors)
                )
            else:
                confidence = 0.0
            caveats = self._build_result_caveats(
                outcomes=outcomes,
                child_results=[
                    result
                    for _, batch in batches
                    for result in batch.results
                    if result.anomaly_id == anomaly_id
                ],
                statistics=statistics,
                explanation=explanation,
                has_factors=bool(factors),
                fusion_notes=fusion_notes,
            )
            metadata = self._build_result_metadata(
                outcomes=outcomes,
                request=request,
                factors=factors,
                statistics=statistics,
            )
            per_event.append(
                DiagnosisResult(
                    anomaly_id=anomaly_id,
                    task=request.task,
                    method_used=self._merge_method_used(method_lists),
                    scope=request.scope,
                    factors=factors,
                    confidence=confidence,
                    analyzed_row_count=template.analyzed_row_count,
                    reference_row_count=template.reference_row_count,
                    caveats=caveats,
                    generated_at=datetime.now(UTC),
                    metadata=metadata,
                )
            )

        aggregate_factors = self._aggregate_event_factors(per_event, request=request)
        warnings = self._build_batch_warnings(
            outcomes=outcomes,
            results=per_event,
            aggregate_factors=aggregate_factors,
            statistics=all_statistics,
        )
        failed_count = batches[0][1].failed_event_count
        requested_count = batches[0][1].requested_event_count
        diagnosed_count = batches[0][1].diagnosed_event_count

        return DiagnosisBatchResult(
            results=per_event,
            aggregate_factors=aggregate_factors,
            requested_event_count=requested_count,
            diagnosed_event_count=diagnosed_count,
            failed_event_count=failed_count,
            generated_at=datetime.now(UTC),
            warnings=warnings,
            metadata={
                "successful_method_count": len(successes),
                "failed_method_count": len(outcomes) - len(successes),
                "diagnosed_event_count": diagnosed_count,
                "aggregate_factor_count": len(aggregate_factors),
                "partial_ensemble": len(successes) < len(outcomes),
                "direction_conflict_count": sum(
                    1 for item in all_statistics if item.direction_conflict
                ),
                "ranking_method": "weighted_reciprocal_rank_fusion",
                "association_not_causation": True,
            },
        )

    def _assert_aligned_batches(
        self,
        batches: list[tuple[_ChildOutcome, DiagnosisBatchResult]],
    ) -> None:
        if not batches:
            raise ProcessIntelligenceError("no successful child batches to combine")
        first_outcome, first = batches[0]
        first_ids = [item.anomaly_id for item in first.results]
        for outcome, batch in batches[1:]:
            if batch.requested_event_count != first.requested_event_count:
                raise ProcessIntelligenceError(
                    f"{outcome.source_key} requested_event_count mismatch"
                )
            if batch.diagnosed_event_count != first.diagnosed_event_count:
                raise ProcessIntelligenceError(
                    f"{outcome.source_key} diagnosed_event_count mismatch"
                )
            if batch.failed_event_count != first.failed_event_count:
                raise ProcessIntelligenceError(
                    f"{outcome.source_key} failed_event_count mismatch"
                )
            other_ids = [item.anomaly_id for item in batch.results]
            if other_ids != first_ids:
                raise ProcessIntelligenceError(
                    f"{outcome.source_key} anomaly_id order mismatch versus "
                    f"{first_outcome.source_key}"
                )

    def _fuse_factor_lists(
        self,
        child_factor_lists: list[tuple[_ChildOutcome, list[RootCauseFactor]]],
        *,
        request: DiagnosisRequest,
    ) -> tuple[list[RootCauseFactor], list[EnsembleFactorStatistic], list[str]]:
        normalized_weights = self._normalized_weights(
            [outcome for outcome, _ in child_factor_lists]
        )
        buckets: dict[str, _FeatureBucket] = {}
        for outcome, factors in child_factor_lists:
            for rank, factor in enumerate(factors, start=1):
                bucket = buckets.setdefault(factor.variable, _FeatureBucket())
                bucket.sources.append(
                    _SourceFactorView(
                        source_key=outcome.source_key,
                        method=outcome.method,
                        rank=rank,
                        factor=factor,
                        weight=normalized_weights[outcome.source_key],
                    )
                )

        total_success_weight = float(sum(normalized_weights.values()))
        statistics: list[EnsembleFactorStatistic] = []
        fusion_notes: list[str] = []

        for feature_name, bucket in buckets.items():
            source_count = len(bucket.sources)
            if source_count < self._config.minimum_method_support:
                continue

            weighted_rrf = 0.0
            source_ranks: dict[str, int] = {}
            source_confidences: dict[str, float] = {}
            source_methods: list[DiagnosisMethod] = []
            for view in bucket.sources:
                contribution = view.weight / (
                    float(self._config.reciprocal_rank_constant) + float(view.rank)
                )
                if not math.isfinite(contribution) or contribution < 0.0:
                    raise ProcessIntelligenceError(
                        "weighted RRF contribution must be finite and >= 0"
                    )
                weighted_rrf += contribution
                source_ranks[view.source_key] = view.rank
                source_confidences[view.source_key] = float(view.factor.confidence)
                source_methods.append(view.method)

            if not math.isfinite(weighted_rrf) or weighted_rrf < 0.0:
                raise ProcessIntelligenceError(
                    "weighted_rrf_score must be finite and >= 0"
                )

            base_confidence = self._weighted_confidence(bucket.sources)
            coverage_weight = float(sum(view.weight for view in bucket.sources))
            method_coverage = coverage_weight / total_success_weight
            if self._config.penalize_partial_method_support:
                combined_confidence = base_confidence * method_coverage
            else:
                combined_confidence = base_confidence
            combined_confidence = min(1.0, max(0.0, float(combined_confidence)))

            combined_deviation = self._combine_deviation(bucket.sources)
            directions = [view.factor.direction for view in bucket.sources]
            combined_direction, direction_conflict = _combine_directions(directions)
            partial_support = source_count < len(child_factor_lists)
            needs_verification = (
                any(view.factor.needs_verification for view in bucket.sources)
                or direction_conflict
                or (
                    partial_support
                    and self._config.penalize_partial_method_support
                )
            )

            warnings: list[str] = []
            if direction_conflict:
                warnings.append("direction conflict across ensemble sources")
            if partial_support:
                warnings.append("partial method support for this feature")

            # normalized score filled after max is known
            statistic = EnsembleFactorStatistic(
                feature_name=feature_name,
                source_method_count=source_count,
                source_methods=source_methods,
                source_ranks=source_ranks,
                source_confidences=source_confidences,
                weighted_rrf_score=float(weighted_rrf),
                normalized_ensemble_score=0.0,
                combined_confidence=combined_confidence,
                combined_direction=combined_direction,
                combined_deviation=combined_deviation,
                direction_conflict=direction_conflict,
                needs_verification=needs_verification,
                warnings=warnings,
            )
            statistics.append(statistic)

        if not statistics:
            fusion_notes.append(_CAVEAT_NO_SUPPORT)
            return [], [], fusion_notes

        if self._config.normalize_ensemble_scores:
            max_score = max(item.weighted_rrf_score for item in statistics)
            for index, statistic in enumerate(statistics):
                if max_score > 0.0:
                    normalized = statistic.weighted_rrf_score / max_score
                else:
                    normalized = 0.0
                statistics[index] = statistic.model_copy(
                    update={"normalized_ensemble_score": float(normalized)}
                )
        else:
            for index, statistic in enumerate(statistics):
                normalized = statistic.weighted_rrf_score / (
                    statistic.weighted_rrf_score + 1.0
                )
                statistics[index] = statistic.model_copy(
                    update={"normalized_ensemble_score": float(normalized)}
                )

        feature_order = {
            name: index for index, name in enumerate(request.feature_columns)
        }
        ordered = sorted(
            statistics,
            key=lambda item: (
                -item.weighted_rrf_score,
                -item.source_method_count,
                -item.combined_confidence,
                item.direction_conflict,
                feature_order.get(item.feature_name, 10**9),
            ),
        )
        ordered = ordered[: request.top_k_factors]

        fused_factors: list[RootCauseFactor] = []
        for statistic in ordered:
            bucket = buckets[statistic.feature_name]
            fused_factors.append(
                self._statistic_to_factor(statistic, bucket.sources)
            )

        return fused_factors, ordered, fusion_notes

    def _weighted_confidence(self, sources: list[_SourceFactorView]) -> float:
        weight_sum = float(sum(view.weight for view in sources))
        if weight_sum <= 0.0:
            return 0.0
        return float(
            sum(view.weight * float(view.factor.confidence) for view in sources)
            / weight_sum
        )

    def _combine_deviation(self, sources: list[_SourceFactorView]) -> float:
        weighted_num = 0.0
        weighted_den = 0.0
        plain: list[float] = []
        for view in sources:
            deviation = view.factor.deviation
            value = 0.0 if deviation is None else float(deviation)
            if not math.isfinite(value):
                raise ProcessIntelligenceError(
                    "source deviation must be finite"
                )
            plain.append(value)
            confidence = float(view.factor.confidence)
            weighted_num += confidence * value
            weighted_den += confidence
        if weighted_den > 0.0:
            combined = weighted_num / weighted_den
        else:
            combined = float(sum(plain) / len(plain))
        if not math.isfinite(combined):
            raise ProcessIntelligenceError("combined_deviation must be finite")
        return combined

    def _statistic_to_factor(
        self,
        statistic: EnsembleFactorStatistic,
        sources: list[_SourceFactorView],
    ) -> RootCauseFactor:
        roles = [view.factor.role for view in sources]
        role = _combine_roles(roles)
        controllable = all(view.factor.controllable for view in sources)
        evidence = self._build_factor_evidence(statistic, sources)
        return RootCauseFactor(
            variable=statistic.feature_name,
            direction=statistic.combined_direction,
            deviation=float(statistic.combined_deviation),
            role=role,
            controllable=controllable,
            evidence=evidence,
            confidence=float(statistic.combined_confidence),
            needs_verification=bool(statistic.needs_verification),
        )

    def _build_factor_evidence(
        self,
        statistic: EnsembleFactorStatistic,
        sources: list[_SourceFactorView],
    ) -> str:
        source_bits: list[str] = []
        ordered_sources = sorted(
            sources,
            key=lambda view: 0 if view.source_key == _SOURCE_ROBUST else 1,
        )
        for view in ordered_sources:
            source_bits.append(
                f"{view.source_key} rank={view.rank} "
                f"confidence={float(view.factor.confidence):.6g}"
            )
        parts = [
            "ensemble association ranking via weighted reciprocal-rank fusion",
            "sources: " + "; ".join(source_bits),
            f"normalized_ensemble_score={statistic.normalized_ensemble_score:.6g}",
            f"combined_confidence={statistic.combined_confidence:.6g}",
            f"combined_direction={statistic.combined_direction}",
            f"source_support={statistic.source_method_count}",
            f"direction_conflict={statistic.direction_conflict}",
            "association does not establish causation",
        ]
        if self._config.include_child_evidence:
            for view in ordered_sources:
                child_text = view.factor.evidence.strip()
                if not child_text:
                    continue
                labeled = f"{view.source_key} evidence: {child_text}"
                parts.append(labeled)
        return "; ".join(_dedupe_preserve(parts))

    def _merge_method_used(
        self,
        method_lists: list[list[DiagnosisMethod]],
    ) -> list[DiagnosisMethod]:
        present: set[DiagnosisMethod] = set()
        for methods in method_lists:
            present.update(methods)
        merged = [method for method in _CANONICAL_METHOD_ORDER if method in present]
        if not merged:
            # Fallback should not happen for valid children; keep contract valid.
            merged = [DiagnosisMethod.ENSEMBLE]
        return merged

    def _build_result_caveats(
        self,
        *,
        outcomes: list[_ChildOutcome],
        child_results: list[DiagnosisResult],
        statistics: list[EnsembleFactorStatistic],
        explanation: ExplanationResult | None,
        has_factors: bool,
        fusion_notes: list[str],
    ) -> list[str]:
        caveats: list[str] = [
            _CAVEAT_ASSOCIATION,
            _CAVEAT_RRF,
            _CAVEAT_ROBUST_REF,
            _CAVEAT_RESIDUAL,
        ]

        failures = [item for item in outcomes if not item.succeeded]
        for item in failures:
            if item.error is None:
                caveats.append(f"child method unavailable: {item.source_key}")
            else:
                caveats.append(
                    "child method failure: "
                    f"{item.source_key} {type(item.error).__name__}: "
                    f"{_short_error_message(item.error)}"
                )

        if any(item.direction_conflict for item in statistics):
            caveats.append(_CAVEAT_DIRECTION_CONFLICT)
        if any(
            item.source_method_count < sum(1 for o in outcomes if o.succeeded)
            for item in statistics
        ):
            caveats.append(_CAVEAT_PARTIAL_SUPPORT)

        if explanation is not None:
            for result in child_results:
                for caveat in result.caveats:
                    if "explanation" in caveat.lower():
                        caveats.append(caveat)

        for note in fusion_notes:
            caveats.append(note)

        if not has_factors:
            caveats.append(_CAVEAT_NO_FACTORS)

        return _dedupe_preserve(caveats)

    def _build_result_metadata(
        self,
        *,
        outcomes: list[_ChildOutcome],
        request: DiagnosisRequest,
        factors: list[RootCauseFactor],
        statistics: list[EnsembleFactorStatistic],
    ) -> dict[str, ScalarMetadataValue]:
        successes = self._successful_outcomes(outcomes)
        robust_ok = any(
            item.source_key == _SOURCE_ROBUST and item.succeeded for item in outcomes
        )
        residual_ok = any(
            item.source_key == _SOURCE_RESIDUAL and item.succeeded
            for item in outcomes
        )
        partial_support_count = sum(
            1
            for item in statistics
            if item.source_method_count < len(successes)
        )
        return {
            "successful_method_count": len(successes),
            "failed_method_count": len(outcomes) - len(successes),
            "robust_method_succeeded": robust_ok,
            "residual_method_succeeded": residual_ok,
            "evaluated_feature_count": len(request.feature_columns),
            "returned_factor_count": len(factors),
            "reciprocal_rank_constant": float(self._config.reciprocal_rank_constant),
            "minimum_method_support": int(self._config.minimum_method_support),
            "ranking_method": "weighted_reciprocal_rank_fusion",
            "association_not_causation": True,
            "partial_ensemble": len(successes) < len(outcomes),
            "direction_conflict_count": sum(
                1 for item in statistics if item.direction_conflict
            ),
            "partial_support_factor_count": partial_support_count,
        }

    def _aggregate_event_factors(
        self,
        results: list[DiagnosisResult],
        *,
        request: DiagnosisRequest,
    ) -> list[RootCauseFactor]:
        buckets: dict[str, _AggregateEventBucket] = {}
        for result in results:
            for rank, factor in enumerate(result.factors, start=1):
                bucket = buckets.setdefault(factor.variable, _AggregateEventBucket())
                bucket.ranks.append(rank)
                bucket.confidences.append(float(factor.confidence))
                deviation = 0.0 if factor.deviation is None else float(factor.deviation)
                bucket.deviations.append(deviation)
                bucket.deviation_weights.append(float(factor.confidence))
                bucket.directions.append(factor.direction)
                bucket.roles.append(factor.role)
                bucket.controllables.append(bool(factor.controllable))
                bucket.needs_verification_flags.append(bool(factor.needs_verification))
                conflict = (
                    _normalize_direction(factor.direction) == _DIRECTION_MIXED
                )
                bucket.direction_conflicts.append(conflict)

        feature_order = {
            name: index for index, name in enumerate(request.feature_columns)
        }
        aggregates: list[tuple[float, int, float, bool, int, RootCauseFactor]] = []
        for feature_name, bucket in buckets.items():
            rrf = 0.0
            for rank in bucket.ranks:
                rrf += 1.0 / (
                    float(self._config.reciprocal_rank_constant) + float(rank)
                )
            support = len(bucket.ranks)
            mean_confidence = float(sum(bucket.confidences) / support)
            weight_sum = float(sum(bucket.deviation_weights))
            if weight_sum > 0.0:
                mean_deviation = float(
                    sum(
                        weight * value
                        for weight, value in zip(
                            bucket.deviation_weights,
                            bucket.deviations,
                            strict=True,
                        )
                    )
                    / weight_sum
                )
            else:
                mean_deviation = float(sum(bucket.deviations) / support)
            direction, conflict = _combine_directions(bucket.directions)
            role = _combine_roles(bucket.roles)
            controllable = all(bucket.controllables)
            needs_verification = (
                any(bucket.needs_verification_flags)
                or conflict
                or any(bucket.direction_conflicts)
            )
            evidence = (
                "aggregate of individually diagnosed event ensemble rankings; "
                f"event_support={support}; "
                f"aggregate_rank_score={rrf:.6g}; "
                f"mean_confidence={mean_confidence:.6g}; "
                f"direction={direction}; "
                "association does not establish causation"
            )
            factor = RootCauseFactor(
                variable=feature_name,
                direction=direction,
                deviation=mean_deviation,
                role=role,
                controllable=controllable,
                evidence=evidence,
                confidence=mean_confidence,
                needs_verification=needs_verification,
            )
            aggregates.append(
                (
                    rrf,
                    support,
                    mean_confidence,
                    conflict,
                    feature_order.get(feature_name, 10**9),
                    factor,
                )
            )

        aggregates.sort(
            key=lambda item: (-item[0], -item[1], -item[2], item[3], item[4])
        )
        return [item[5] for item in aggregates[: request.top_k_factors]]

    def _build_batch_warnings(
        self,
        *,
        outcomes: list[_ChildOutcome],
        results: list[DiagnosisResult],
        aggregate_factors: list[RootCauseFactor],
        statistics: list[EnsembleFactorStatistic],
    ) -> list[str]:
        warnings = [
            _BATCH_WARNING_ASSOCIATION,
            _BATCH_WARNING_AGGREGATED,
            _BATCH_WARNING_RRF,
        ]
        successes = self._successful_outcomes(outcomes)
        if len(successes) < len(outcomes):
            warnings.append(_BATCH_WARNING_PARTIAL)
        if any(item.direction_conflict for item in statistics):
            warnings.append(_BATCH_WARNING_DIRECTION)
        if any(not result.factors for result in results):
            warnings.append(_BATCH_WARNING_EMPTY_EVENT)
        if not aggregate_factors:
            warnings.append(_BATCH_WARNING_EMPTY_AGGREGATE)
        return _dedupe_preserve(warnings)
