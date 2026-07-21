"""Score candidate-grid scenarios with fitted models (Step 9D).

Uses a fitted supervised regression model for predicted quality and a fitted
unsupervised anomaly model for feature-space anomaly scores. Does not rank
scenarios, generate recommendations, refit models, or score residual anomalies.
"""

from __future__ import annotations

import copy
import math
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

import numpy as np
import polars as pl
from pydantic import BaseModel, Field, field_validator, model_validator

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.exceptions import (
    DataValidationError,
    ProcessIntelligenceError,
)
from process_intelligence.core.protocols import BaseAnalysisModel, BaseAnomalyModel
from process_intelligence.core.schemas import ModelMetadata, ModelSpec
from process_intelligence.recommendation.candidate_grid import (
    CandidateGridReport,
    CandidateGridStatus,
    CandidateScenario,
    CandidateScenarioType,
)
from process_intelligence.recommendation.enums import RecommendationObjective
from process_intelligence.recommendation.schemas import ScalarMetadataValue

_ORIGINAL_ROW_ID = "_original_row_id"
_SCENARIO_ID_RE = re.compile(r"^SCN-(\d{6})$")
_TIMING_ABS_TOL = 1e-6

_WARNING_GRID_REFUSED = "Candidate grid status is REFUSED; scenario scoring was refused."
_WARNING_NO_SCENARIOS = "Candidate grid contains no scenarios to score."
_WARNING_BATCH_LIMIT = (
    "Scenario count exceeds maximum_batch_rows; scenario scoring was refused."
)
_WARNING_QUALITY_MODEL_MISSING = (
    "Required quality regression model is missing for the requested objective."
)
_WARNING_ANOMALY_MODEL_MISSING = (
    "Required anomaly model is missing for the requested objective."
)
_WARNING_FEATURE_METADATA = (
    "Model feature metadata is unavailable; exact feature-set validation "
    "could not be completed."
)
_WARNING_PARTIAL = (
    "Partial scoring applied; some required model outputs are unavailable."
)
_WARNING_RESIDUAL = (
    "Residual anomaly was not computed because actual target values are unavailable"
)
_WARNING_UNRANKED = "Scenarios remain unranked"
_WARNING_NO_CAUSATION = (
    "Model scoring does not establish process causation or guarantee improvement"
)
_WARNING_VERIFICATION = (
    "Model, extrapolation, uncertainty, domain, safety, and operational "
    "verification required"
)

_ISOLATED_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ProcessIntelligenceError,
    ValueError,
    TypeError,
    ArithmeticError,
    np.linalg.LinAlgError,
)


def _require_non_empty_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be str, got {type(value).__name__}")
    if value == "" or value.strip() == "":
        raise ValueError(f"{field_name} must be a non-empty, non-whitespace string")
    return value


def _require_optional_non_empty_str(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_non_empty_str(value, field_name=field_name)


def _require_strict_bool(value: object, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(
            f"{field_name} must be a bool (0/1 and strings rejected), "
            f"got {type(value).__name__}"
        )
    return value


def _require_strict_int_ge(
    value: object,
    *,
    field_name: str,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{field_name} must be an int >= {minimum} "
            f"(bool not allowed), got {type(value).__name__}"
        )
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {value}")
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


def _require_optional_finite_float(
    value: object,
    *,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    return _require_finite_float(value, field_name=field_name)


def _require_non_negative_finite_float(
    value: object,
    *,
    field_name: str,
) -> float:
    number = _require_finite_float(value, field_name=field_name)
    if number < 0.0:
        raise ValueError(f"{field_name} must be >= 0, got {number}")
    return number


def _require_timezone_aware(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _validate_unique_non_empty_strings(
    values: list[str],
    *,
    field_name: str,
) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _require_non_empty_str(item, field_name=field_name)
        if text in seen:
            raise ValueError(f"{field_name} must not contain duplicates: {text!r}")
        seen.add(text)
        cleaned.append(text)
    return cleaned


def _validate_scalar_metadata(
    value: object,
    *,
    field_name: str = "metadata",
) -> dict[str, ScalarMetadataValue]:
    if not isinstance(value, dict):
        raise ValueError(
            f"{field_name} must be a dict[str, scalar], got {type(value).__name__}"
        )
    cleaned: dict[str, ScalarMetadataValue] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or key == "" or key.strip() == "":
            raise ValueError(f"{field_name} keys must be non-empty strings")
        if raw is None or isinstance(raw, (str, bool)):
            cleaned[key] = raw
            continue
        if isinstance(raw, int) and not isinstance(raw, bool):
            cleaned[key] = raw
            continue
        if isinstance(raw, float):
            if not math.isfinite(raw):
                raise ValueError(
                    f"{field_name}[{key!r}] float must be finite, got {raw!r}"
                )
            cleaned[key] = raw
            continue
        raise ValueError(
            f"{field_name}[{key!r}] must be str, int, float, bool, or None "
            f"(no DataFrame, ndarray, estimator, or nested objects); "
            f"got {type(raw).__name__}"
        )
    return cleaned


def _append_unique_warning(warnings: list[str], message: str) -> None:
    if message and message not in warnings:
        warnings.append(message)


def _failure_warning(*, role: str, exc: BaseException) -> str:
    message = str(exc).strip() or type(exc).__name__
    if len(message) > 200:
        message = message[:197] + "..."
    return f"{role} scoring failed ({type(exc).__name__}): {message}"


class ScenarioScoringStatus(StrEnum):
    """Outcome status for candidate scenario model scoring.

    ``SCORED`` means every objective-required model output was computed for all
    scenarios. ``PARTIAL`` means policy allowed incomplete required outputs.
    ``REFUSED`` means scoring could not produce usable scenario scores.
    """

    SCORED = "SCORED"
    PARTIAL = "PARTIAL"
    REFUSED = "REFUSED"


class ScenarioScoringPolicy(BaseModel):
    """Tunable policy for batch scenario scoring without model refitting.

    Controls batch limits, feature-schema checks, fitted-state requirements,
    and whether partial objective outputs are allowed.
    """

    maximum_batch_rows: int = 5000
    require_regression_task: bool = True
    require_fitted_models: bool = True
    require_exact_feature_set: bool = True
    allow_partial_scoring: bool = False
    preserve_scenario_order: bool = True
    require_baseline_scenario: bool = True
    require_finite_predictions: bool = True
    require_finite_anomaly_scores: bool = True

    @field_validator("maximum_batch_rows", mode="before")
    @classmethod
    def _validate_maximum_batch_rows(cls, value: object) -> int:
        return _require_strict_int_ge(
            value,
            field_name="maximum_batch_rows",
            minimum=1,
        )

    @field_validator(
        "require_regression_task",
        "require_fitted_models",
        "require_exact_feature_set",
        "allow_partial_scoring",
        "preserve_scenario_order",
        "require_baseline_scenario",
        "require_finite_predictions",
        "require_finite_anomaly_scores",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="policy bool field")


class ScenarioScoringRequest(BaseModel):
    """Request contract for scoring a candidate grid with fitted models.

    Uses ``grid.objective`` as the scoring objective. Does not embed models,
    estimators, DataFrames, or ranked recommendation outputs.
    """

    grid: CandidateGridReport
    feature_columns: list[str]
    baseline_features: dict[str, float]
    target_column: str | None = None
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)

    @field_validator("grid", mode="before")
    @classmethod
    def _validate_grid(cls, value: object) -> CandidateGridReport:
        if isinstance(value, CandidateGridReport):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return CandidateGridReport.model_validate(value)
        raise ValueError(
            f"grid must be CandidateGridReport, got {type(value).__name__}"
        )

    @field_validator("feature_columns", mode="before")
    @classmethod
    def _validate_feature_columns_before(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            raise ValueError(
                f"feature_columns must be a list[str], got {type(value).__name__}"
            )
        return list(value)

    @field_validator("feature_columns", mode="after")
    @classmethod
    def _validate_feature_columns(cls, value: list[str]) -> list[str]:
        cleaned = _validate_unique_non_empty_strings(
            value,
            field_name="feature_columns",
        )
        if not cleaned:
            raise ValueError("feature_columns must contain at least one feature")
        for name in cleaned:
            if name == _ORIGINAL_ROW_ID:
                raise ValueError(
                    f"feature_columns cannot include reserved column "
                    f"'{_ORIGINAL_ROW_ID}'"
                )
        return cleaned

    @field_validator("baseline_features", mode="before")
    @classmethod
    def _validate_baseline_features_before(cls, value: object) -> dict[str, float]:
        if not isinstance(value, dict):
            raise ValueError(
                f"baseline_features must be a dict[str, float], "
                f"got {type(value).__name__}"
            )
        cleaned: dict[str, float] = {}
        for key, raw in value.items():
            name = _require_non_empty_str(key, field_name="baseline_features key")
            if name in cleaned:
                raise ValueError(
                    f"baseline_features must not contain duplicate keys: {name!r}"
                )
            cleaned[name] = _require_finite_float(
                raw,
                field_name=f"baseline_features[{name!r}]",
            )
        return cleaned

    @field_validator("target_column", mode="before")
    @classmethod
    def _validate_target_column(cls, value: object) -> str | None:
        return _require_optional_non_empty_str(value, field_name="target_column")

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> dict[str, ScalarMetadataValue]:
        if value is None:
            return {}
        return _validate_scalar_metadata(value)

    @model_validator(mode="after")
    def _validate_request_consistency(self) -> Self:
        if self.target_column is not None and self.target_column in self.feature_columns:
            raise ValueError(
                "target_column must not appear in feature_columns "
                f"(got {self.target_column!r})"
            )

        feature_set = set(self.feature_columns)
        baseline_keys = set(self.baseline_features)
        if baseline_keys != feature_set:
            raise ValueError(
                "baseline_features keys must exactly match feature_columns "
                f"(features={list(self.feature_columns)}, "
                f"baseline={sorted(baseline_keys)})"
            )

        for variable in self.grid.candidate_variables:
            if variable not in feature_set:
                raise ValueError(
                    f"grid candidate variable {variable!r} is absent from "
                    "feature_columns"
                )
            current = None
            for grid in self.grid.variable_grids:
                if grid.variable == variable:
                    current = grid.current_value
                    break
            if current is None:
                raise ValueError(
                    f"grid candidate variable {variable!r} has no variable grid"
                )
            baseline_value = self.baseline_features[variable]
            if not math.isclose(
                baseline_value,
                current,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"baseline_features[{variable!r}] must match grid current "
                    f"value (baseline={baseline_value}, current={current})"
                )
        return self


class CandidateScenarioScore(BaseModel):
    """Per-scenario quality prediction and/or feature-space anomaly score.

    Stores deltas from the baseline scenario when the corresponding output was
    scored. Does not rank scenarios or assert process improvement.
    """

    scenario_id: str
    scenario_index: int
    scenario_type: CandidateScenarioType
    change_count: int
    normalized_change_magnitude: float
    quality_prediction: float | None
    anomaly_score: float | None
    quality_delta_from_baseline: float | None
    anomaly_score_delta_from_baseline: float | None
    quality_scored: bool
    anomaly_scored: bool
    warnings: list[str] = Field(default_factory=list)

    @field_validator("scenario_id", mode="before")
    @classmethod
    def _validate_scenario_id(cls, value: object) -> str:
        text = _require_non_empty_str(value, field_name="scenario_id")
        if _SCENARIO_ID_RE.fullmatch(text) is None:
            raise ValueError(
                f"scenario_id must match SCN-000000 format, got {text!r}"
            )
        return text

    @field_validator("scenario_index", mode="before")
    @classmethod
    def _validate_scenario_index(cls, value: object) -> int:
        return _require_strict_int_ge(value, field_name="scenario_index", minimum=0)

    @field_validator("scenario_type", mode="before")
    @classmethod
    def _validate_scenario_type(cls, value: object) -> CandidateScenarioType:
        if isinstance(value, CandidateScenarioType):
            return value
        if isinstance(value, str):
            try:
                return CandidateScenarioType(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid CandidateScenarioType: {value!r}"
                ) from exc
        raise ValueError(
            f"scenario_type must be CandidateScenarioType, got {type(value).__name__}"
        )

    @field_validator("change_count", mode="before")
    @classmethod
    def _validate_change_count(cls, value: object) -> int:
        return _require_strict_int_ge(value, field_name="change_count", minimum=0)

    @field_validator("normalized_change_magnitude", mode="before")
    @classmethod
    def _validate_normalized_change_magnitude(cls, value: object) -> float:
        return _require_non_negative_finite_float(
            value,
            field_name="normalized_change_magnitude",
        )

    @field_validator(
        "quality_prediction",
        "anomaly_score",
        "quality_delta_from_baseline",
        "anomaly_score_delta_from_baseline",
        mode="before",
    )
    @classmethod
    def _validate_optional_outputs(cls, value: object) -> float | None:
        # Adapter contract: anomaly scores are higher-is-more-anomalous and may
        # be negative (normal points have negative -decision_function scores).
        return _require_optional_finite_float(value, field_name="model output")

    @field_validator("quality_scored", "anomaly_scored", mode="before")
    @classmethod
    def _validate_scored_flags(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="scored flag")

    @field_validator("warnings", mode="before")
    @classmethod
    def _validate_warnings_before(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(f"warnings must be a list[str], got {type(value).__name__}")
        return list(value)

    @field_validator("warnings", mode="after")
    @classmethod
    def _validate_warnings(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(value, field_name="warnings")

    @model_validator(mode="after")
    def _validate_score_consistency(self) -> Self:
        match = _SCENARIO_ID_RE.fullmatch(self.scenario_id)
        assert match is not None
        if int(match.group(1)) != self.scenario_index:
            raise ValueError("scenario_id numeric suffix must equal scenario_index")

        if self.quality_scored:
            if self.quality_prediction is None:
                raise ValueError(
                    "quality_scored=True requires quality_prediction to be present"
                )
        elif self.quality_prediction is not None or (
            self.quality_delta_from_baseline is not None
        ):
            raise ValueError(
                "quality_scored=False requires quality_prediction and "
                "quality_delta_from_baseline to be None"
            )

        if self.anomaly_scored:
            if self.anomaly_score is None:
                raise ValueError(
                    "anomaly_scored=True requires anomaly_score to be present"
                )
        elif self.anomaly_score is not None or (
            self.anomaly_score_delta_from_baseline is not None
        ):
            raise ValueError(
                "anomaly_scored=False requires anomaly_score and "
                "anomaly_score_delta_from_baseline to be None"
            )

        if self.scenario_index == 0:
            if self.quality_scored and self.quality_delta_from_baseline is not None:
                if not math.isclose(
                    self.quality_delta_from_baseline,
                    0.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "baseline quality_delta_from_baseline must be 0 when scored"
                    )
            if self.anomaly_scored and self.anomaly_score_delta_from_baseline is not None:
                if not math.isclose(
                    self.anomaly_score_delta_from_baseline,
                    0.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "baseline anomaly_score_delta_from_baseline must be 0 "
                        "when scored"
                    )
        return self


class ScenarioScoringReport(BaseModel):
    """Structured report of unranked scenario model scores.

    Records quality predictions and/or feature-space anomaly scores without
    selecting a best scenario or emitting recommendations.
    """

    status: ScenarioScoringStatus
    objective: RecommendationObjective
    task: AnalysisTask
    feature_columns: list[str]
    target_column: str | None
    quality_model_name: str | None
    quality_estimator_key: str | None
    anomaly_model_name: str | None
    anomaly_estimator_key: str | None
    baseline_scenario_id: str | None
    scores: list[CandidateScenarioScore]
    requested_scenario_count: int
    scored_scenario_count: int
    quality_scored_count: int
    anomaly_scored_count: int
    prediction_seconds: float
    anomaly_scoring_seconds: float
    total_seconds: float
    evaluated_at: datetime
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)

    @field_validator("status", mode="before")
    @classmethod
    def _validate_status(cls, value: object) -> ScenarioScoringStatus:
        if isinstance(value, ScenarioScoringStatus):
            return value
        if isinstance(value, str):
            try:
                return ScenarioScoringStatus(value)
            except ValueError as exc:
                raise ValueError(f"invalid ScenarioScoringStatus: {value!r}") from exc
        raise ValueError(
            f"status must be ScenarioScoringStatus, got {type(value).__name__}"
        )

    @field_validator("objective", mode="before")
    @classmethod
    def _validate_objective(cls, value: object) -> RecommendationObjective:
        if isinstance(value, RecommendationObjective):
            return value
        if isinstance(value, str):
            try:
                return RecommendationObjective(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid RecommendationObjective: {value!r}"
                ) from exc
        raise ValueError(
            f"objective must be RecommendationObjective, got {type(value).__name__}"
        )

    @field_validator("task", mode="before")
    @classmethod
    def _validate_task(cls, value: object) -> AnalysisTask:
        if isinstance(value, AnalysisTask):
            return value
        if isinstance(value, str):
            try:
                return AnalysisTask(value)
            except ValueError as exc:
                raise ValueError(f"invalid AnalysisTask: {value!r}") from exc
        raise ValueError(f"task must be AnalysisTask, got {type(value).__name__}")

    @field_validator("feature_columns", mode="before")
    @classmethod
    def _validate_feature_columns_before(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            raise ValueError(
                f"feature_columns must be a list[str], got {type(value).__name__}"
            )
        return list(value)

    @field_validator("feature_columns", mode="after")
    @classmethod
    def _validate_feature_columns(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(
            value,
            field_name="feature_columns",
        )

    @field_validator("target_column", mode="before")
    @classmethod
    def _validate_target_column(cls, value: object) -> str | None:
        return _require_optional_non_empty_str(value, field_name="target_column")

    @field_validator(
        "quality_model_name",
        "quality_estimator_key",
        "anomaly_model_name",
        "anomaly_estimator_key",
        "baseline_scenario_id",
        mode="before",
    )
    @classmethod
    def _validate_optional_identity(cls, value: object) -> str | None:
        return _require_optional_non_empty_str(value, field_name="identity field")

    @field_validator("scores", mode="before")
    @classmethod
    def _validate_scores_before(cls, value: object) -> list[CandidateScenarioScore]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"scores must be a list[CandidateScenarioScore], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("scores", mode="after")
    @classmethod
    def _validate_scores_after(
        cls,
        value: list[CandidateScenarioScore],
    ) -> list[CandidateScenarioScore]:
        return [item.model_copy(deep=True) for item in value]

    @field_validator(
        "requested_scenario_count",
        "scored_scenario_count",
        "quality_scored_count",
        "anomaly_scored_count",
        mode="before",
    )
    @classmethod
    def _validate_counts(cls, value: object) -> int:
        return _require_strict_int_ge(value, field_name="count field", minimum=0)

    @field_validator(
        "prediction_seconds",
        "anomaly_scoring_seconds",
        "total_seconds",
        mode="before",
    )
    @classmethod
    def _validate_timing(cls, value: object) -> float:
        return _require_non_negative_finite_float(value, field_name="timing field")

    @field_validator("evaluated_at", mode="after")
    @classmethod
    def _validate_evaluated_at(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value, field_name="evaluated_at")

    @field_validator("warnings", mode="before")
    @classmethod
    def _validate_warnings_before(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(f"warnings must be a list[str], got {type(value).__name__}")
        return list(value)

    @field_validator("warnings", mode="after")
    @classmethod
    def _validate_warnings(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(value, field_name="warnings")

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> dict[str, ScalarMetadataValue]:
        if value is None:
            return {}
        return _validate_scalar_metadata(value)

    @model_validator(mode="after")
    def _validate_report_consistency(self) -> Self:
        quality_pair = (self.quality_model_name, self.quality_estimator_key)
        if (quality_pair[0] is None) != (quality_pair[1] is None):
            raise ValueError(
                "quality_model_name and quality_estimator_key must both be "
                "present or both be None"
            )
        anomaly_pair = (self.anomaly_model_name, self.anomaly_estimator_key)
        if (anomaly_pair[0] is None) != (anomaly_pair[1] is None):
            raise ValueError(
                "anomaly_model_name and anomaly_estimator_key must both be "
                "present or both be None"
            )

        score_ids = [item.scenario_id for item in self.scores]
        if len(score_ids) != len(set(score_ids)):
            raise ValueError("score scenario IDs must be unique")
        score_indexes = [item.scenario_index for item in self.scores]
        if len(score_indexes) != len(set(score_indexes)):
            raise ValueError("score scenario indexes must be unique")
        if score_indexes != sorted(score_indexes):
            raise ValueError("scores must be ordered by ascending scenario_index")
        for expected_index, score in enumerate(self.scores):
            if score.scenario_index != expected_index:
                raise ValueError(
                    "scores must preserve contiguous scenario_index order from 0"
                )

        if self.scored_scenario_count != len(self.scores):
            raise ValueError("scored_scenario_count must equal len(scores)")
        quality_count = sum(1 for item in self.scores if item.quality_scored)
        anomaly_count = sum(1 for item in self.scores if item.anomaly_scored)
        if self.quality_scored_count != quality_count:
            raise ValueError(
                "quality_scored_count must equal the number of quality_scored scores"
            )
        if self.anomaly_scored_count != anomaly_count:
            raise ValueError(
                "anomaly_scored_count must equal the number of anomaly_scored scores"
            )
        if self.quality_scored_count > self.scored_scenario_count:
            raise ValueError("quality_scored_count must be <= scored_scenario_count")
        if self.anomaly_scored_count > self.scored_scenario_count:
            raise ValueError("anomaly_scored_count must be <= scored_scenario_count")

        expected_parts = self.prediction_seconds + self.anomaly_scoring_seconds
        if self.total_seconds + _TIMING_ABS_TOL < expected_parts:
            raise ValueError(
                "total_seconds must be at least prediction_seconds + "
                "anomaly_scoring_seconds within timing tolerance"
            )

        if self.scores:
            if self.baseline_scenario_id is not None:
                if self.scores[0].scenario_id != self.baseline_scenario_id:
                    raise ValueError(
                        "baseline_scenario_id must match the first score scenario_id"
                    )
                if self.scores[0].scenario_type is not CandidateScenarioType.BASELINE:
                    raise ValueError(
                        "first score must be BASELINE when baseline_scenario_id "
                        "is set"
                    )

        if self.status is ScenarioScoringStatus.REFUSED:
            if self.scores:
                raise ValueError("REFUSED report must have empty scores")
            if self.scored_scenario_count != 0:
                raise ValueError("REFUSED report must have scored_scenario_count=0")
            if self.quality_scored_count != 0 or self.anomaly_scored_count != 0:
                raise ValueError("REFUSED report must have scored counts of 0")
            if self.baseline_scenario_id is not None:
                raise ValueError("REFUSED report must have baseline_scenario_id=None")
        elif self.status is ScenarioScoringStatus.PARTIAL:
            if not self.scores:
                raise ValueError("PARTIAL report requires at least one score")
            if not self._has_partial_required_outputs():
                raise ValueError(
                    "PARTIAL report requires some objective-required outputs "
                    "to be missing"
                )
        elif self.status is ScenarioScoringStatus.SCORED:
            if not self.scores:
                raise ValueError("SCORED report requires at least one score")
            if not self._has_complete_required_outputs():
                raise ValueError(
                    "SCORED report requires all objective-required outputs "
                    "for every score"
                )
        else:
            raise ValueError(f"unsupported scoring status: {self.status!r}")

        return self

    def _requires_quality(self) -> bool:
        return self.objective in {
            RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
            RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
        }

    def _requires_anomaly(self) -> bool:
        return self.objective in {
            RecommendationObjective.REDUCE_ANOMALY_SCORE,
            RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
        }

    def _has_complete_required_outputs(self) -> bool:
        if self._requires_quality() and not all(
            item.quality_scored for item in self.scores
        ):
            return False
        if self._requires_anomaly() and not all(
            item.anomaly_scored for item in self.scores
        ):
            return False
        return True

    def _has_partial_required_outputs(self) -> bool:
        if not self.scores:
            return False
        if self._requires_quality() and self._requires_anomaly():
            quality_ok = all(item.quality_scored for item in self.scores)
            anomaly_ok = all(item.anomaly_scored for item in self.scores)
            return (quality_ok or anomaly_ok) and not (quality_ok and anomaly_ok)
        if self._requires_quality():
            return any(item.quality_scored for item in self.scores) and not all(
                item.quality_scored for item in self.scores
            )
        if self._requires_anomaly():
            return any(item.anomaly_scored for item in self.scores) and not all(
                item.anomaly_scored for item in self.scores
            )
        return False


@dataclass(frozen=True, slots=True)
class ScenarioScoringOutcome:
    """Immutable wrapper around a scenario scoring report.

    Holds only ``ScenarioScoringReport``. Does not store models, estimators,
    or DataFrames, and contains no business logic.
    """

    report: ScenarioScoringReport


def _is_raw_sklearn_estimator(value: object) -> bool:
    fit_method = getattr(value, "fit", None)
    predict_method = getattr(value, "predict", None)
    if not callable(fit_method) or not callable(predict_method):
        return False
    module = type(value).__module__
    return module.startswith("sklearn")


def _validate_quality_model(model: object | None) -> BaseAnalysisModel | None:
    if model is None:
        return None
    if _is_raw_sklearn_estimator(model) and not isinstance(model, BaseAnalysisModel):
        raise TypeError(
            "quality_model must be a project supervised model adapter/protocol, "
            f"not a raw sklearn estimator ({type(model).__name__})"
        )
    if not isinstance(model, BaseAnalysisModel):
        raise TypeError(
            "quality_model must be BaseAnalysisModel or None, "
            f"got {type(model).__name__}"
        )
    if isinstance(model, BaseAnomalyModel):
        raise TypeError(
            "quality_model must be a supervised analysis model, "
            f"got anomaly model {type(model).__name__}"
        )
    return model


def _validate_anomaly_model(model: object | None) -> BaseAnomalyModel | None:
    if model is None:
        return None
    if _is_raw_sklearn_estimator(model) and not isinstance(model, BaseAnomalyModel):
        raise TypeError(
            "anomaly_model must be a project anomaly adapter/protocol, "
            f"not a raw sklearn estimator ({type(model).__name__})"
        )
    if not isinstance(model, BaseAnomalyModel):
        raise TypeError(
            "anomaly_model must be BaseAnomalyModel or None, "
            f"got {type(model).__name__}"
        )
    if not callable(getattr(model, "score_samples", None)):
        raise TypeError(
            "anomaly_model must expose a callable score_samples method"
        )
    return model


def _resolve_model_identity(
    model: BaseAnalysisModel | BaseAnomalyModel | None,
) -> tuple[str | None, str | None]:
    if model is None:
        return None, None
    metadata = model.get_metadata()
    if not isinstance(metadata, ModelMetadata):
        raise ProcessIntelligenceError(
            "model.get_metadata must return ModelMetadata, "
            f"got {type(metadata).__name__}"
        )
    name = metadata.model_name
    estimator_key = getattr(metadata, "estimator_key", None)
    if not isinstance(estimator_key, str) or estimator_key.strip() == "":
        model_spec = getattr(model, "_spec", None)
        if isinstance(model_spec, ModelSpec):
            estimator_key = model_spec.estimator_key
        else:
            estimator_key = None
    if not isinstance(name, str) or name.strip() == "":
        return None, None
    if not isinstance(estimator_key, str) or estimator_key.strip() == "":
        return None, None
    return name, estimator_key


def _resolve_model_features(
    model: BaseAnalysisModel | BaseAnomalyModel,
) -> list[str] | None:
    feature_names = getattr(model, "feature_names", None)
    if isinstance(feature_names, tuple) and feature_names:
        return list(feature_names)
    if isinstance(feature_names, list) and feature_names:
        return list(feature_names)
    metadata = model.get_metadata()
    features = list(metadata.features)
    if features:
        return features
    return None


def _snapshot_model_state(
    model: BaseAnalysisModel | BaseAnomalyModel,
) -> dict[str, Any]:
    metadata = model.get_metadata()
    fitted = bool(getattr(model, "is_fitted", False))
    feature_names = getattr(model, "feature_names", None)
    if isinstance(feature_names, tuple):
        features: list[str] = list(feature_names)
    elif isinstance(feature_names, list):
        features = list(feature_names)
    else:
        features = list(metadata.features)
    estimator_key = getattr(metadata, "estimator_key", None)
    if not isinstance(estimator_key, str):
        model_spec = getattr(model, "_spec", None)
        estimator_key = (
            model_spec.estimator_key if isinstance(model_spec, ModelSpec) else None
        )
    parameters = getattr(metadata, "parameters", None)
    stable_parameters = (
        copy.deepcopy(parameters) if isinstance(parameters, dict) else None
    )
    return {
        "fitted": fitted,
        "task": metadata.task,
        "features": features,
        "model_name": metadata.model_name,
        "estimator_key": estimator_key,
        "parameters": stable_parameters,
        "seed": metadata.seed,
        "version": metadata.version,
    }


def _assert_model_state_unchanged(
    model: BaseAnalysisModel | BaseAnomalyModel,
    before: dict[str, Any],
    *,
    role: str,
) -> None:
    after = _snapshot_model_state(model)
    if after != before:
        raise ProcessIntelligenceError(
            f"{role} model state changed during scenario scoring"
        )


def _coerce_prediction_array(
    values: object,
    *,
    expected_length: int,
    require_finite: bool,
    role: str,
) -> list[float]:
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array.reshape(-1)
    if array.ndim != 1:
        if array.ndim == 2 and array.shape[1] > 1:
            raise ProcessIntelligenceError(
                f"{role} multi-output predictions are not supported in Step 9D, "
                f"got shape {array.shape}"
            )
        raise ProcessIntelligenceError(
            f"{role} predictions must be 1D, got shape {array.shape}"
        )
    if array.shape[0] != expected_length:
        raise ProcessIntelligenceError(
            f"{role} prediction length ({array.shape[0]}) must match scenario "
            f"count ({expected_length})"
        )
    result: list[float] = []
    for index, raw in enumerate(array.tolist()):
        if isinstance(raw, (str, bytes)):
            raise ProcessIntelligenceError(
                f"{role} predictions must be numeric, got string at index {index}"
            )
        if isinstance(raw, bool) or not isinstance(raw, (int, float, np.integer, np.floating)):
            raise ProcessIntelligenceError(
                f"{role} predictions must be numeric, got {type(raw).__name__} "
                f"at index {index}"
            )
        number = float(raw)
        if require_finite and not math.isfinite(number):
            raise ProcessIntelligenceError(
                f"{role} predictions must be finite, got {number!r} at index {index}"
            )
        result.append(number)
    return result


def _coerce_score_array(
    values: object,
    *,
    expected_length: int,
    require_finite: bool,
    role: str,
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    array = np.reshape(array, -1)
    if array.ndim != 1:
        raise ProcessIntelligenceError(
            f"{role} scores must be 1D, got shape {array.shape}"
        )
    if array.shape[0] != expected_length:
        raise ProcessIntelligenceError(
            f"{role} score length ({array.shape[0]}) must match scenario "
            f"count ({expected_length})"
        )
    result: list[float] = []
    for index, raw in enumerate(array.tolist()):
        number = float(raw)
        if require_finite and not math.isfinite(number):
            raise ProcessIntelligenceError(
                f"{role} scores must be finite, got {number!r} at index {index}"
            )
        result.append(number)
    return result


def _objective_task(objective: RecommendationObjective) -> AnalysisTask:
    if objective is RecommendationObjective.REDUCE_ANOMALY_SCORE:
        return AnalysisTask.UNSUPERVISED_ANOMALY
    return AnalysisTask.REGRESSION


def _build_feature_matrix(
    *,
    scenarios: list[CandidateScenario],
    feature_columns: list[str],
    baseline_features: dict[str, float],
    candidate_variables: list[str],
) -> pl.DataFrame:
    candidate_set = set(candidate_variables)
    columns: dict[str, list[float]] = {name: [] for name in feature_columns}
    for scenario in scenarios:
        row = dict(baseline_features)
        for name, value in scenario.variable_values.items():
            if name not in candidate_set:
                raise DataValidationError(
                    f"scenario {scenario.scenario_id} variable {name!r} is not "
                    "a grid candidate variable"
                )
            if name not in row:
                raise DataValidationError(
                    f"scenario {scenario.scenario_id} variable {name!r} is absent "
                    "from baseline_features / feature_columns"
                )
            number = _require_finite_float(
                value,
                field_name=f"scenario.variable_values[{name!r}]",
            )
            row[name] = number
        for name in feature_columns:
            if name not in row:
                raise DataValidationError(
                    f"feature {name!r} is missing from the constructed scenario row"
                )
            number = _require_finite_float(
                row[name],
                field_name=f"feature_row[{name!r}]",
            )
            columns[name].append(number)
    return pl.DataFrame(columns)


class CandidateScenarioScorer:
    """Score candidate scenarios with fitted quality and/or anomaly models.

    Performs a single batch ``predict`` and/or ``score_samples`` call after
    validation. Does not fit, refit, rank scenarios, or emit recommendations.
    """

    def __init__(
        self,
        *,
        quality_model: BaseAnalysisModel | None = None,
        anomaly_model: BaseAnomalyModel | None = None,
        policy: ScenarioScoringPolicy | None = None,
    ) -> None:
        if policy is None:
            resolved_policy = ScenarioScoringPolicy()
        elif isinstance(policy, ScenarioScoringPolicy):
            resolved_policy = policy.model_copy(deep=True)
        else:
            raise TypeError(
                "policy must be ScenarioScoringPolicy or None, "
                f"got {type(policy).__name__}"
            )

        self._policy = resolved_policy
        self._quality_model = _validate_quality_model(quality_model)
        self._anomaly_model = _validate_anomaly_model(anomaly_model)

    def get_metadata(self) -> dict[str, ScalarMetadataValue]:
        """Return scalar scorer capability metadata for the current instance."""
        return {
            "maximum_batch_rows": self._policy.maximum_batch_rows,
            "require_regression_task": self._policy.require_regression_task,
            "require_fitted_models": self._policy.require_fitted_models,
            "require_exact_feature_set": self._policy.require_exact_feature_set,
            "allow_partial_scoring": self._policy.allow_partial_scoring,
            "preserve_scenario_order": self._policy.preserve_scenario_order,
            "require_baseline_scenario": self._policy.require_baseline_scenario,
            "quality_model_available": self._quality_model is not None,
            "anomaly_model_available": self._anomaly_model is not None,
            "scores_quality_prediction": self._quality_model is not None,
            "scores_feature_space_anomaly": self._anomaly_model is not None,
            "scores_residual_anomaly": False,
            "performs_model_refit": False,
            "performs_scenario_ranking": False,
            "generates_recommendation": False,
        }

    def score(self, request: ScenarioScoringRequest) -> ScenarioScoringOutcome:
        """Score all grid scenarios in grid order without ranking.

        Args:
            request: Validated scenario scoring request.

        Returns:
            Outcome wrapping a ``ScenarioScoringReport``.

        Raises:
            TypeError: If ``request`` is not a ``ScenarioScoringRequest``.
            DataValidationError: If baseline, fitted-state, task, or feature
                schema contracts fail.
            ProcessIntelligenceError: If model state mutates during scoring.
        """
        if not isinstance(request, ScenarioScoringRequest):
            raise TypeError(
                "request must be ScenarioScoringRequest, "
                f"got {type(request).__name__}"
            )

        total_started = time.perf_counter()
        warnings: list[str] = []
        objective = request.grid.objective
        task = _objective_task(objective)
        quality_name, quality_key = _resolve_model_identity(self._quality_model)
        anomaly_name, anomaly_key = _resolve_model_identity(self._anomaly_model)
        requested_count = len(request.grid.scenarios)

        def _refused(
            *,
            warning: str | None = None,
            extra_warnings: list[str] | None = None,
            prediction_seconds: float = 0.0,
            anomaly_scoring_seconds: float = 0.0,
            quality_call_count: int = 0,
            anomaly_call_count: int = 0,
            quality_performed: bool = False,
            anomaly_performed: bool = False,
        ) -> ScenarioScoringOutcome:
            local_warnings = list(warnings)
            if warning is not None:
                _append_unique_warning(local_warnings, warning)
            if extra_warnings:
                for item in extra_warnings:
                    _append_unique_warning(local_warnings, item)
            self._append_standard_caveats(local_warnings)
            total_seconds = max(
                0.0,
                time.perf_counter() - total_started,
            )
            total_seconds = max(
                total_seconds,
                prediction_seconds + anomaly_scoring_seconds,
            )
            report = ScenarioScoringReport(
                status=ScenarioScoringStatus.REFUSED,
                objective=objective,
                task=task,
                feature_columns=list(request.feature_columns),
                target_column=request.target_column,
                quality_model_name=quality_name,
                quality_estimator_key=quality_key,
                anomaly_model_name=anomaly_name,
                anomaly_estimator_key=anomaly_key,
                baseline_scenario_id=None,
                scores=[],
                requested_scenario_count=requested_count,
                scored_scenario_count=0,
                quality_scored_count=0,
                anomaly_scored_count=0,
                prediction_seconds=float(prediction_seconds),
                anomaly_scoring_seconds=float(anomaly_scoring_seconds),
                total_seconds=float(total_seconds),
                evaluated_at=datetime.now(tz=UTC),
                warnings=local_warnings,
                metadata=self._build_metadata(
                    request=request,
                    scored_count=0,
                    quality_performed=quality_performed,
                    anomaly_performed=anomaly_performed,
                    quality_call_count=quality_call_count,
                    anomaly_call_count=anomaly_call_count,
                ),
            )
            return ScenarioScoringOutcome(report=report)

        if request.grid.status is CandidateGridStatus.REFUSED:
            return _refused(warning=_WARNING_GRID_REFUSED)

        if not request.grid.scenarios:
            return _refused(warning=_WARNING_NO_SCENARIOS)

        if requested_count > self._policy.maximum_batch_rows:
            return _refused(warning=_WARNING_BATCH_LIMIT)

        if self._policy.require_baseline_scenario:
            first = request.grid.scenarios[0]
            if first.scenario_type is not CandidateScenarioType.BASELINE:
                raise DataValidationError(
                    "require_baseline_scenario=True requires the first scenario "
                    "to be BASELINE"
                )
            if first.scenario_index != 0:
                raise DataValidationError(
                    "require_baseline_scenario=True requires baseline "
                    "scenario_index=0"
                )
            if request.grid.baseline_scenario_id != first.scenario_id:
                raise DataValidationError(
                    "grid.baseline_scenario_id must match the baseline scenario_id"
                )

        requires_quality = objective in {
            RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
            RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
        }
        requires_anomaly = objective in {
            RecommendationObjective.REDUCE_ANOMALY_SCORE,
            RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
        }

        if requires_quality and self._quality_model is None:
            return _refused(warning=_WARNING_QUALITY_MODEL_MISSING)
        if requires_anomaly and self._anomaly_model is None:
            return _refused(warning=_WARNING_ANOMALY_MODEL_MISSING)

        score_quality = self._quality_model is not None
        score_anomaly = self._anomaly_model is not None

        if score_quality:
            assert self._quality_model is not None
            self._validate_quality_ready(
                self._quality_model,
                request=request,
                warnings=warnings,
            )
            task = self._quality_model.get_metadata().task
        if score_anomaly:
            assert self._anomaly_model is not None
            self._validate_anomaly_ready(
                self._anomaly_model,
                request=request,
                warnings=warnings,
            )
            if not score_quality:
                task = self._anomaly_model.get_metadata().task

        feature_frame = _build_feature_matrix(
            scenarios=list(request.grid.scenarios),
            feature_columns=list(request.feature_columns),
            baseline_features=dict(request.baseline_features),
            candidate_variables=list(request.grid.candidate_variables),
        )

        quality_values: list[float] | None = None
        anomaly_values: list[float] | None = None
        prediction_seconds = 0.0
        anomaly_scoring_seconds = 0.0
        quality_call_count = 0
        anomaly_call_count = 0
        quality_error: BaseException | None = None
        anomaly_error: BaseException | None = None

        quality_snapshot: dict[str, Any] | None = None
        anomaly_snapshot: dict[str, Any] | None = None
        if score_quality and self._quality_model is not None:
            quality_snapshot = _snapshot_model_state(self._quality_model)
        if score_anomaly and self._anomaly_model is not None:
            anomaly_snapshot = _snapshot_model_state(self._anomaly_model)

        if score_quality and self._quality_model is not None:
            started = time.perf_counter()
            try:
                predictions = self._quality_model.predict(feature_frame)
                quality_call_count = 1
                quality_values = _coerce_prediction_array(
                    predictions,
                    expected_length=requested_count,
                    require_finite=self._policy.require_finite_predictions,
                    role="quality",
                )
            except _ISOLATED_EXCEPTIONS as exc:
                quality_error = exc
                _append_unique_warning(
                    warnings,
                    _failure_warning(role="quality", exc=exc),
                )
            finally:
                prediction_seconds = time.perf_counter() - started
            if quality_snapshot is not None:
                _assert_model_state_unchanged(
                    self._quality_model,
                    quality_snapshot,
                    role="quality",
                )

        if score_anomaly and self._anomaly_model is not None:
            started = time.perf_counter()
            try:
                raw_anomaly_scores = self._anomaly_model.score_samples(feature_frame)
                anomaly_call_count = 1
                anomaly_values = _coerce_score_array(
                    raw_anomaly_scores,
                    expected_length=requested_count,
                    require_finite=self._policy.require_finite_anomaly_scores,
                    role="anomaly",
                )
            except _ISOLATED_EXCEPTIONS as exc:
                anomaly_error = exc
                _append_unique_warning(
                    warnings,
                    _failure_warning(role="anomaly", exc=exc),
                )
            finally:
                anomaly_scoring_seconds = time.perf_counter() - started
            if anomaly_snapshot is not None:
                _assert_model_state_unchanged(
                    self._anomaly_model,
                    anomaly_snapshot,
                    role="anomaly",
                )

        quality_ok = quality_values is not None
        anomaly_ok = anomaly_values is not None
        required_quality_failed = requires_quality and not quality_ok
        required_anomaly_failed = requires_anomaly and not anomaly_ok

        if required_quality_failed or required_anomaly_failed:
            if not self._policy.allow_partial_scoring:
                return _refused(
                    prediction_seconds=prediction_seconds,
                    anomaly_scoring_seconds=anomaly_scoring_seconds,
                    quality_call_count=quality_call_count,
                    anomaly_call_count=anomaly_call_count,
                    quality_performed=quality_ok,
                    anomaly_performed=anomaly_ok,
                )
            # Partial allowed only when at least one required path succeeded
            # or optional success provides some scores for a single-requirement
            # objective with incomplete coverage. For balance, one of two.
            has_any_success = quality_ok or anomaly_ok
            if not has_any_success:
                return _refused(
                    prediction_seconds=prediction_seconds,
                    anomaly_scoring_seconds=anomaly_scoring_seconds,
                    quality_call_count=quality_call_count,
                    anomaly_call_count=anomaly_call_count,
                )
            if requires_quality and requires_anomaly:
                if not (quality_ok or anomaly_ok):
                    return _refused(
                        prediction_seconds=prediction_seconds,
                        anomaly_scoring_seconds=anomaly_scoring_seconds,
                        quality_call_count=quality_call_count,
                        anomaly_call_count=anomaly_call_count,
                    )
            elif requires_quality and not quality_ok:
                return _refused(
                    prediction_seconds=prediction_seconds,
                    anomaly_scoring_seconds=anomaly_scoring_seconds,
                    quality_call_count=quality_call_count,
                    anomaly_call_count=anomaly_call_count,
                    anomaly_performed=anomaly_ok,
                )
            elif requires_anomaly and not anomaly_ok:
                return _refused(
                    prediction_seconds=prediction_seconds,
                    anomaly_scoring_seconds=anomaly_scoring_seconds,
                    quality_call_count=quality_call_count,
                    anomaly_call_count=anomaly_call_count,
                    quality_performed=quality_ok,
                )

        if not quality_ok and not anomaly_ok:
            return _refused(
                prediction_seconds=prediction_seconds,
                anomaly_scoring_seconds=anomaly_scoring_seconds,
                quality_call_count=quality_call_count,
                anomaly_call_count=anomaly_call_count,
            )

        baseline_quality = quality_values[0] if quality_ok and quality_values else None
        baseline_anomaly = anomaly_values[0] if anomaly_ok and anomaly_values else None
        scores: list[CandidateScenarioScore] = []
        for index, scenario in enumerate(request.grid.scenarios):
            quality_prediction = (
                quality_values[index] if quality_ok and quality_values else None
            )
            anomaly_score = (
                anomaly_values[index] if anomaly_ok and anomaly_values else None
            )
            quality_delta: float | None = None
            anomaly_delta: float | None = None
            if quality_prediction is not None and baseline_quality is not None:
                quality_delta = float(quality_prediction - baseline_quality)
            if anomaly_score is not None and baseline_anomaly is not None:
                anomaly_delta = float(anomaly_score - baseline_anomaly)
            scores.append(
                CandidateScenarioScore(
                    scenario_id=scenario.scenario_id,
                    scenario_index=scenario.scenario_index,
                    scenario_type=scenario.scenario_type,
                    change_count=scenario.change_count,
                    normalized_change_magnitude=scenario.normalized_change_magnitude,
                    quality_prediction=quality_prediction,
                    anomaly_score=anomaly_score,
                    quality_delta_from_baseline=quality_delta,
                    anomaly_score_delta_from_baseline=anomaly_delta,
                    quality_scored=quality_prediction is not None,
                    anomaly_scored=anomaly_score is not None,
                    warnings=[],
                )
            )

        required_complete = True
        if requires_quality and not all(item.quality_scored for item in scores):
            required_complete = False
        if requires_anomaly and not all(item.anomaly_scored for item in scores):
            required_complete = False

        if required_complete:
            status = ScenarioScoringStatus.SCORED
        else:
            if not self._policy.allow_partial_scoring:
                return _refused(
                    prediction_seconds=prediction_seconds,
                    anomaly_scoring_seconds=anomaly_scoring_seconds,
                    quality_call_count=quality_call_count,
                    anomaly_call_count=anomaly_call_count,
                    quality_performed=quality_ok,
                    anomaly_performed=anomaly_ok,
                )
            status = ScenarioScoringStatus.PARTIAL
            _append_unique_warning(warnings, _WARNING_PARTIAL)

        if quality_error is not None and quality_ok is False:
            pass  # already warned
        if anomaly_error is not None and anomaly_ok is False:
            pass

        self._append_standard_caveats(warnings)
        total_seconds = max(
            time.perf_counter() - total_started,
            prediction_seconds + anomaly_scoring_seconds,
        )

        report = ScenarioScoringReport(
            status=status,
            objective=objective,
            task=task,
            feature_columns=list(request.feature_columns),
            target_column=request.target_column,
            quality_model_name=quality_name,
            quality_estimator_key=quality_key,
            anomaly_model_name=anomaly_name,
            anomaly_estimator_key=anomaly_key,
            baseline_scenario_id=(
                scores[0].scenario_id
                if scores
                and scores[0].scenario_type is CandidateScenarioType.BASELINE
                else None
            ),
            scores=scores,
            requested_scenario_count=requested_count,
            scored_scenario_count=len(scores),
            quality_scored_count=sum(1 for item in scores if item.quality_scored),
            anomaly_scored_count=sum(1 for item in scores if item.anomaly_scored),
            prediction_seconds=float(prediction_seconds),
            anomaly_scoring_seconds=float(anomaly_scoring_seconds),
            total_seconds=float(total_seconds),
            evaluated_at=datetime.now(tz=UTC),
            warnings=warnings,
            metadata=self._build_metadata(
                request=request,
                scored_count=len(scores),
                quality_performed=quality_ok,
                anomaly_performed=anomaly_ok,
                quality_call_count=quality_call_count,
                anomaly_call_count=anomaly_call_count,
            ),
        )
        return ScenarioScoringOutcome(report=report)

    def _validate_quality_ready(
        self,
        model: BaseAnalysisModel,
        *,
        request: ScenarioScoringRequest,
        warnings: list[str],
    ) -> None:
        if self._policy.require_fitted_models:
            if not bool(getattr(model, "is_fitted", False)):
                raise DataValidationError(
                    "quality_model must be fitted when require_fitted_models=True"
                )
        if self._policy.require_regression_task:
            metadata = model.get_metadata()
            if metadata.task is not AnalysisTask.REGRESSION:
                raise DataValidationError(
                    "quality_model task must be REGRESSION when "
                    f"require_regression_task=True, got {metadata.task!r}"
                )
        features = _resolve_model_features(model)
        if features is None:
            _append_unique_warning(warnings, _WARNING_FEATURE_METADATA)
            if self._policy.require_exact_feature_set:
                raise DataValidationError(
                    "quality_model feature metadata is unavailable; cannot "
                    "satisfy require_exact_feature_set=True"
                )
            return
        if self._policy.require_exact_feature_set:
            if features != list(request.feature_columns):
                raise DataValidationError(
                    "quality_model features must exactly match "
                    "request.feature_columns in order: "
                    f"expected {list(request.feature_columns)}, got {features}"
                )

    def _validate_anomaly_ready(
        self,
        model: BaseAnomalyModel,
        *,
        request: ScenarioScoringRequest,
        warnings: list[str],
    ) -> None:
        if self._policy.require_fitted_models:
            if not bool(getattr(model, "is_fitted", False)):
                raise DataValidationError(
                    "anomaly_model must be fitted when require_fitted_models=True"
                )
        features = _resolve_model_features(model)
        if features is None:
            _append_unique_warning(warnings, _WARNING_FEATURE_METADATA)
            if self._policy.require_exact_feature_set:
                raise DataValidationError(
                    "anomaly_model feature metadata is unavailable; cannot "
                    "satisfy require_exact_feature_set=True"
                )
            return
        if self._policy.require_exact_feature_set:
            if features != list(request.feature_columns):
                raise DataValidationError(
                    "anomaly_model features must exactly match "
                    "request.feature_columns in order: "
                    f"expected {list(request.feature_columns)}, got {features}"
                )

    @staticmethod
    def _append_standard_caveats(warnings: list[str]) -> None:
        for message in (
            _WARNING_RESIDUAL,
            _WARNING_UNRANKED,
            _WARNING_NO_CAUSATION,
            _WARNING_VERIFICATION,
        ):
            _append_unique_warning(warnings, message)

    def _build_metadata(
        self,
        *,
        request: ScenarioScoringRequest,
        scored_count: int,
        quality_performed: bool,
        anomaly_performed: bool,
        quality_call_count: int,
        anomaly_call_count: int,
    ) -> dict[str, ScalarMetadataValue]:
        return {
            "grid_status": request.grid.status.value,
            "grid_scenario_count": len(request.grid.scenarios),
            "scored_scenario_count": scored_count,
            "quality_model_available": self._quality_model is not None,
            "anomaly_model_available": self._anomaly_model is not None,
            "quality_scoring_performed": quality_performed,
            "anomaly_scoring_performed": anomaly_performed,
            "residual_scoring_performed": False,
            "model_refit_performed": False,
            "scenario_ranking_performed": False,
            "recommendation_generated": False,
            "baseline_feature_count": len(request.feature_columns),
            "candidate_variable_count": len(request.grid.candidate_variables),
            "batch_prediction_call_count": quality_call_count,
            "batch_anomaly_call_count": anomaly_call_count,
            "scores_preserve_grid_order": True,
            "actual_target_available": False,
        }
