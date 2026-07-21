"""Recommendation pipeline orchestrator connecting Steps 9A–9F (Step 9G).

Wires existing concrete recommendation components in a fixed stage order.
Does not implement scoring, ranking, constraint merging, or model fit/refit.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from process_intelligence.core.exceptions import ProcessIntelligenceError
from process_intelligence.core.schemas import VariableConstraint
from process_intelligence.recommendation.candidate_grid import (
    CandidateGridGenerator,
    CandidateGridReport,
    CandidateGridStatus,
)
from process_intelligence.recommendation.candidate_selection import (
    CandidateVariableSelector,
    CandidateVariableSet,
)
from process_intelligence.recommendation.constraint_resolution import (
    ConstraintResolutionOutcome,
    ConstraintResolutionReport,
    ConstraintResolutionStatus,
    ConstraintResolver,
)
from process_intelligence.recommendation.enums import (
    RecommendationObjective,
    RecommendationSafetyStatus,
    RecommendationStatus,
)
from process_intelligence.recommendation.recommendation_generation import (
    RankedScenarioRecommendationGenerator,
    RecommendationGenerationRequest,
)
from process_intelligence.recommendation.safety_gate import RecommendationSafetyGate
from process_intelligence.recommendation.scenario_ranking import (
    CandidateScenarioRanker,
    QualityOptimizationDirection,
    ScenarioRankingReport,
    ScenarioRankingRequest,
    ScenarioRankingStatus,
)
from process_intelligence.recommendation.scenario_scoring import (
    CandidateScenarioScorer,
    ScenarioScoringReport,
    ScenarioScoringRequest,
    ScenarioScoringStatus,
)
from process_intelligence.recommendation.schemas import (
    DEFAULT_RECOMMENDATION_DISCLAIMER,
    RecommendationRequest,
    RecommendationResult,
    RecommendationSafetyContext,
    RecommendationSafetyDecision,
    ScalarMetadataValue,
)

_ORIGINAL_ROW_ID = "_original_row_id"

_OMISSION_WARNING = (
    "additional pipeline warnings were omitted due to the configured limit"
)

_PRESERVE_OUTPUTS_WARNING = (
    "preserve_stage_outputs=False is recorded in metadata only; "
    "executed stage outputs are always retained in the pipeline report"
)


class RecommendationPipelineStage(StrEnum):
    """Canonical orchestration stages for the recommendation pipeline.

    Stages mirror Steps 9A–9F in execution order. The pipeline invokes each
    stage's existing concrete component and does not reimplement algorithms.
    """

    SAFETY = "SAFETY"
    CONSTRAINT_RESOLUTION = "CONSTRAINT_RESOLUTION"
    CANDIDATE_SELECTION = "CANDIDATE_SELECTION"
    GRID_GENERATION = "GRID_GENERATION"
    SCENARIO_SCORING = "SCENARIO_SCORING"
    SCENARIO_RANKING = "SCENARIO_RANKING"
    RECOMMENDATION_GENERATION = "RECOMMENDATION_GENERATION"


_CANONICAL_STAGES: tuple[RecommendationPipelineStage, ...] = (
    RecommendationPipelineStage.SAFETY,
    RecommendationPipelineStage.CONSTRAINT_RESOLUTION,
    RecommendationPipelineStage.CANDIDATE_SELECTION,
    RecommendationPipelineStage.GRID_GENERATION,
    RecommendationPipelineStage.SCENARIO_SCORING,
    RecommendationPipelineStage.SCENARIO_RANKING,
    RecommendationPipelineStage.RECOMMENDATION_GENERATION,
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


def _require_non_negative_finite_float(value: object, *, field_name: str) -> float:
    number = _require_finite_float(value, field_name=field_name)
    if number < 0.0:
        raise ValueError(f"{field_name} must be >= 0, got {number}")
    return number


def _require_timezone_aware(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


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


def _append_unique(target: list[str], message: str) -> None:
    if message not in target:
        target.append(message)


def _stable_metadata_snapshot(
    metadata: dict[str, ScalarMetadataValue],
) -> dict[str, ScalarMetadataValue]:
    """Return scalar metadata entries suitable for before/after equality checks."""
    stable: dict[str, ScalarMetadataValue] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, bool)) or value is None:
            stable[key] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            stable[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            stable[key] = value
    return stable


def _stages_after(
    terminal: RecommendationPipelineStage,
) -> list[RecommendationPipelineStage]:
    index = _CANONICAL_STAGES.index(terminal)
    return list(_CANONICAL_STAGES[index + 1 :])


def _extract_baselines_from_scoring(
    scoring: ScenarioScoringReport | None,
) -> tuple[float | None, float | None]:
    if scoring is None:
        return None, None
    baseline_id = scoring.baseline_scenario_id
    if baseline_id is None:
        return None, None
    for score in scoring.scores:
        if score.scenario_id == baseline_id:
            return score.quality_prediction, score.anomaly_score
    return None, None


def _extract_baselines_from_ranking(
    ranking: ScenarioRankingReport | None,
) -> tuple[float | None, float | None]:
    if ranking is None:
        return None, None
    return ranking.baseline_quality_prediction, ranking.baseline_anomaly_score


class RecommendationPipelinePolicy(BaseModel):
    """Tunable early-stop and warning aggregation policy for pipeline orchestration.

    Stop flags terminate after structured stage outcomes. They never invent
    placeholder schemas. Stage algorithms remain in the concrete components.
    """

    stop_on_safety_refusal: bool = True
    stop_on_constraint_refusal: bool = True
    stop_on_empty_candidates: bool = True
    stop_on_grid_refusal_or_empty: bool = True
    stop_on_scoring_refusal: bool = True
    stop_on_ranking_refusal: bool = True
    preserve_stage_outputs: bool = True
    include_stage_warnings: bool = True
    maximum_aggregated_warnings: int = 100

    @field_validator(
        "stop_on_safety_refusal",
        "stop_on_constraint_refusal",
        "stop_on_empty_candidates",
        "stop_on_grid_refusal_or_empty",
        "stop_on_scoring_refusal",
        "stop_on_ranking_refusal",
        "preserve_stage_outputs",
        "include_stage_warnings",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="policy bool field")

    @field_validator("maximum_aggregated_warnings", mode="before")
    @classmethod
    def _validate_maximum_aggregated_warnings(cls, value: object) -> int:
        return _require_strict_int_ge(
            value,
            field_name="maximum_aggregated_warnings",
            minimum=1,
        )


class RecommendationPipelineRequest(BaseModel):
    """Validated inputs for end-to-end recommendation pipeline orchestration.

    Carries the Step 9A request/context plus feature-space and quality-direction
    fields needed by later stages. Does not embed models, estimators, or
    DataFrames.
    """

    recommendation_request: RecommendationRequest
    safety_context: RecommendationSafetyContext
    industry_constraints: list[VariableConstraint]
    user_overrides: list[VariableConstraint]
    feature_columns: list[str]
    baseline_features: dict[str, float]
    target_column: str | None = None
    quality_direction: QualityOptimizationDirection | None = None
    quality_target: float | None = None
    extrapolation_evaluated: bool = False
    extrapolation_flag: bool = False
    uncertainty_available: bool = False
    uncertainty_acceptable: bool | None = None
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)

    @field_validator("recommendation_request", mode="before")
    @classmethod
    def _validate_recommendation_request(cls, value: object) -> RecommendationRequest:
        if isinstance(value, RecommendationRequest):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return RecommendationRequest.model_validate(value)
        raise ValueError(
            "recommendation_request must be RecommendationRequest, "
            f"got {type(value).__name__}"
        )

    @field_validator("safety_context", mode="before")
    @classmethod
    def _validate_safety_context(cls, value: object) -> RecommendationSafetyContext:
        if isinstance(value, RecommendationSafetyContext):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return RecommendationSafetyContext.model_validate(value)
        raise ValueError(
            "safety_context must be RecommendationSafetyContext, "
            f"got {type(value).__name__}"
        )

    @field_validator("industry_constraints", "user_overrides", mode="before")
    @classmethod
    def _validate_constraint_lists_before(cls, value: object) -> list[VariableConstraint]:
        if not isinstance(value, list):
            raise ValueError(
                f"constraint list must be a list[VariableConstraint], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("industry_constraints", "user_overrides", mode="after")
    @classmethod
    def _validate_constraint_lists(
        cls,
        value: list[VariableConstraint],
    ) -> list[VariableConstraint]:
        seen: set[str] = set()
        copied: list[VariableConstraint] = []
        for item in value:
            if not isinstance(item, VariableConstraint):
                raise ValueError(
                    "constraint list entries must be VariableConstraint, "
                    f"got {type(item).__name__}"
                )
            name = item.variable
            if name in seen:
                raise ValueError(
                    f"constraint list must not contain duplicate variables: {name!r}"
                )
            seen.add(name)
            copied.append(item.model_copy(deep=True))
        return copied

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

    @field_validator("quality_direction", mode="before")
    @classmethod
    def _validate_quality_direction(
        cls,
        value: object,
    ) -> QualityOptimizationDirection | None:
        if value is None:
            return None
        if isinstance(value, QualityOptimizationDirection):
            return value
        if isinstance(value, str):
            try:
                return QualityOptimizationDirection(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid QualityOptimizationDirection: {value!r}"
                ) from exc
        raise ValueError(
            "quality_direction must be QualityOptimizationDirection or None, "
            f"got {type(value).__name__}"
        )

    @field_validator("quality_target", mode="before")
    @classmethod
    def _validate_quality_target(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_finite_float(value, field_name="quality_target")

    @field_validator(
        "extrapolation_evaluated",
        "extrapolation_flag",
        "uncertainty_available",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="request bool field")

    @field_validator("uncertainty_acceptable", mode="before")
    @classmethod
    def _validate_uncertainty_acceptable(cls, value: object) -> bool | None:
        if value is None:
            return None
        return _require_strict_bool(value, field_name="uncertainty_acceptable")

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> dict[str, ScalarMetadataValue]:
        if value is None:
            return {}
        return _validate_scalar_metadata(value)

    @model_validator(mode="after")
    def _validate_request_consistency(self) -> Self:
        if (
            self.target_column is not None
            and self.target_column in self.feature_columns
        ):
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

        current_values = self.recommendation_request.current_values
        for name, current in current_values.items():
            if name not in self.baseline_features:
                raise ValueError(
                    "recommendation_request.current_values variables must exist in "
                    f"baseline_features; missing {name!r}"
                )
            baseline_value = self.baseline_features[name]
            if not math.isclose(
                current,
                baseline_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"current_values[{name!r}] must match baseline_features "
                    f"(current={current}, baseline={baseline_value})"
                )

        objective = self.recommendation_request.objective
        needs_quality = objective in {
            RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
            RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
        }
        if needs_quality and self.quality_direction is None:
            raise ValueError(
                "quality_direction is required for IMPROVE_PREDICTED_QUALITY "
                "and BALANCE_QUALITY_AND_ANOMALY objectives"
            )

        if self.quality_direction is QualityOptimizationDirection.TARGET:
            if self.quality_target is None:
                raise ValueError(
                    "quality_target is required when quality_direction is TARGET"
                )
        elif self.quality_target is not None:
            raise ValueError(
                "quality_target must be None unless quality_direction is TARGET"
            )

        if not self.extrapolation_evaluated and self.extrapolation_flag:
            raise ValueError(
                "extrapolation_flag must be False when extrapolation_evaluated=False"
            )

        if not self.uncertainty_available and self.uncertainty_acceptable is not None:
            raise ValueError(
                "uncertainty_acceptable must be None when uncertainty_available=False"
            )

        return self


class RecommendationPipelineReport(BaseModel):
    """Structured report for a single recommendation pipeline run.

    Records executed and skipped stages, optional stage outputs, aggregated
    warnings, and the terminal ``RecommendationResult``. Does not embed models
    or DataFrames.
    """

    status: RecommendationStatus
    terminal_stage: RecommendationPipelineStage
    final_result: RecommendationResult
    safety_decision: RecommendationSafetyDecision
    constraint_report: ConstraintResolutionReport | None = None
    candidate_set: CandidateVariableSet | None = None
    grid_report: CandidateGridReport | None = None
    scoring_report: ScenarioScoringReport | None = None
    ranking_report: ScenarioRankingReport | None = None
    executed_stages: list[RecommendationPipelineStage]
    skipped_stages: list[RecommendationPipelineStage]
    started_at: datetime
    completed_at: datetime
    total_seconds: float
    warnings: list[str]
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)

    @field_validator("status", mode="before")
    @classmethod
    def _validate_status(cls, value: object) -> RecommendationStatus:
        if isinstance(value, RecommendationStatus):
            return value
        if isinstance(value, str):
            try:
                return RecommendationStatus(value)
            except ValueError as exc:
                raise ValueError(f"invalid RecommendationStatus: {value!r}") from exc
        raise ValueError(
            f"status must be RecommendationStatus, got {type(value).__name__}"
        )

    @field_validator("terminal_stage", mode="before")
    @classmethod
    def _validate_terminal_stage(cls, value: object) -> RecommendationPipelineStage:
        if isinstance(value, RecommendationPipelineStage):
            return value
        if isinstance(value, str):
            try:
                return RecommendationPipelineStage(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid RecommendationPipelineStage: {value!r}"
                ) from exc
        raise ValueError(
            "terminal_stage must be RecommendationPipelineStage, "
            f"got {type(value).__name__}"
        )

    @field_validator("final_result", mode="before")
    @classmethod
    def _validate_final_result(cls, value: object) -> RecommendationResult:
        if isinstance(value, RecommendationResult):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return RecommendationResult.model_validate(value)
        raise ValueError(
            f"final_result must be RecommendationResult, got {type(value).__name__}"
        )

    @field_validator("safety_decision", mode="before")
    @classmethod
    def _validate_safety_decision(
        cls,
        value: object,
    ) -> RecommendationSafetyDecision:
        if isinstance(value, RecommendationSafetyDecision):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return RecommendationSafetyDecision.model_validate(value)
        raise ValueError(
            "safety_decision must be RecommendationSafetyDecision, "
            f"got {type(value).__name__}"
        )

    @field_validator("constraint_report", mode="before")
    @classmethod
    def _validate_constraint_report(
        cls,
        value: object,
    ) -> ConstraintResolutionReport | None:
        if value is None:
            return None
        if isinstance(value, ConstraintResolutionReport):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return ConstraintResolutionReport.model_validate(value)
        raise ValueError(
            "constraint_report must be ConstraintResolutionReport or None, "
            f"got {type(value).__name__}"
        )

    @field_validator("candidate_set", mode="before")
    @classmethod
    def _validate_candidate_set(cls, value: object) -> CandidateVariableSet | None:
        if value is None:
            return None
        if isinstance(value, CandidateVariableSet):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return CandidateVariableSet.model_validate(value)
        raise ValueError(
            "candidate_set must be CandidateVariableSet or None, "
            f"got {type(value).__name__}"
        )

    @field_validator("grid_report", mode="before")
    @classmethod
    def _validate_grid_report(cls, value: object) -> CandidateGridReport | None:
        if value is None:
            return None
        if isinstance(value, CandidateGridReport):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return CandidateGridReport.model_validate(value)
        raise ValueError(
            f"grid_report must be CandidateGridReport or None, got {type(value).__name__}"
        )

    @field_validator("scoring_report", mode="before")
    @classmethod
    def _validate_scoring_report(cls, value: object) -> ScenarioScoringReport | None:
        if value is None:
            return None
        if isinstance(value, ScenarioScoringReport):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return ScenarioScoringReport.model_validate(value)
        raise ValueError(
            "scoring_report must be ScenarioScoringReport or None, "
            f"got {type(value).__name__}"
        )

    @field_validator("ranking_report", mode="before")
    @classmethod
    def _validate_ranking_report(cls, value: object) -> ScenarioRankingReport | None:
        if value is None:
            return None
        if isinstance(value, ScenarioRankingReport):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return ScenarioRankingReport.model_validate(value)
        raise ValueError(
            "ranking_report must be ScenarioRankingReport or None, "
            f"got {type(value).__name__}"
        )

    @field_validator("executed_stages", "skipped_stages", mode="before")
    @classmethod
    def _validate_stage_lists_before(cls, value: object) -> list[object]:
        if not isinstance(value, list):
            raise ValueError(
                f"stage list must be a list[RecommendationPipelineStage], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("executed_stages", "skipped_stages", mode="after")
    @classmethod
    def _validate_stage_lists(
        cls,
        value: list[RecommendationPipelineStage],
    ) -> list[RecommendationPipelineStage]:
        cleaned: list[RecommendationPipelineStage] = []
        seen: set[RecommendationPipelineStage] = set()
        for item in value:
            if isinstance(item, RecommendationPipelineStage):
                stage = item
            elif isinstance(item, str):
                try:
                    stage = RecommendationPipelineStage(item)
                except ValueError as exc:
                    raise ValueError(
                        f"invalid RecommendationPipelineStage: {item!r}"
                    ) from exc
            else:
                raise ValueError(
                    "stage list entries must be RecommendationPipelineStage, "
                    f"got {type(item).__name__}"
                )
            if stage in seen:
                raise ValueError(f"stage list must not contain duplicates: {stage!r}")
            seen.add(stage)
            cleaned.append(stage)
        return cleaned

    @field_validator("started_at", "completed_at", mode="after")
    @classmethod
    def _validate_timestamps(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value, field_name="timestamp")

    @field_validator("total_seconds", mode="before")
    @classmethod
    def _validate_total_seconds(cls, value: object) -> float:
        return _require_non_negative_finite_float(value, field_name="total_seconds")

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
        executed = self.executed_stages
        skipped = self.skipped_stages
        executed_set = set(executed)
        skipped_set = set(skipped)

        overlap = executed_set & skipped_set
        if overlap:
            raise ValueError(
                "executed_stages and skipped_stages must be disjoint; "
                f"overlap={sorted(item.value for item in overlap)}"
            )

        union = executed_set | skipped_set
        canonical_set = set(_CANONICAL_STAGES)
        if union != canonical_set:
            raise ValueError(
                "executed_stages and skipped_stages must cover all canonical stages "
                f"(got {sorted(item.value for item in union)})"
            )

        if RecommendationPipelineStage.SAFETY not in executed_set:
            raise ValueError("SAFETY must always be executed")

        if not executed:
            raise ValueError("executed_stages must not be empty")

        expected_executed = list(_CANONICAL_STAGES[: len(executed)])
        if executed != expected_executed:
            raise ValueError(
                "executed_stages must be a contiguous prefix of the canonical order"
            )

        expected_skipped = list(_CANONICAL_STAGES[len(executed) :])
        if skipped != expected_skipped:
            raise ValueError(
                "skipped_stages must be the contiguous suffix after executed_stages"
            )

        if self.terminal_stage != executed[-1]:
            raise ValueError(
                "terminal_stage must equal the last executed stage "
                f"(got {self.terminal_stage!r}, last={executed[-1]!r})"
            )

        if self.final_result.status is not self.status:
            raise ValueError("final_result.status must equal report.status")

        if self.final_result.safety_decision != self.safety_decision:
            raise ValueError(
                "final_result.safety_decision must match report.safety_decision"
            )

        stage_output_rules: list[
            tuple[RecommendationPipelineStage, object | None, str]
        ] = [
            (
                RecommendationPipelineStage.CONSTRAINT_RESOLUTION,
                self.constraint_report,
                "constraint_report",
            ),
            (
                RecommendationPipelineStage.CANDIDATE_SELECTION,
                self.candidate_set,
                "candidate_set",
            ),
            (
                RecommendationPipelineStage.GRID_GENERATION,
                self.grid_report,
                "grid_report",
            ),
            (
                RecommendationPipelineStage.SCENARIO_SCORING,
                self.scoring_report,
                "scoring_report",
            ),
            (
                RecommendationPipelineStage.SCENARIO_RANKING,
                self.ranking_report,
                "ranking_report",
            ),
        ]
        for stage, output, field_name in stage_output_rules:
            if stage in executed_set:
                if output is None:
                    raise ValueError(
                        f"{field_name} must be present when {stage.value} is executed"
                    )
            elif output is not None:
                raise ValueError(
                    f"{field_name} must be None when {stage.value} is skipped"
                )

        if self.completed_at < self.started_at:
            raise ValueError("completed_at must be >= started_at")

        return self


@dataclass(frozen=True, slots=True)
class RecommendationPipelineOutcome:
    """Immutable wrapper around a recommendation pipeline report.

    Holds only ``RecommendationPipelineReport``. Contains no business logic and
    does not store models, estimators, or DataFrames.
    """

    report: RecommendationPipelineReport


class RecommendationPipeline:
    """Orchestrate Steps 9A–9F into a single recommendation pipeline run.

    Requires a fitted ``CandidateScenarioScorer``. Other stage components default
    to fresh concrete instances. The pipeline does not fit models, call predict
    directly, or reimplement stage algorithms.
    """

    def __init__(
        self,
        *,
        scenario_scorer: CandidateScenarioScorer,
        safety_gate: RecommendationSafetyGate | None = None,
        constraint_resolver: ConstraintResolver | None = None,
        candidate_selector: CandidateVariableSelector | None = None,
        grid_generator: CandidateGridGenerator | None = None,
        scenario_ranker: CandidateScenarioRanker | None = None,
        recommendation_generator: RankedScenarioRecommendationGenerator | None = None,
        policy: RecommendationPipelinePolicy | None = None,
    ) -> None:
        if not isinstance(scenario_scorer, CandidateScenarioScorer):
            raise TypeError(
                "scenario_scorer must be CandidateScenarioScorer, "
                f"got {type(scenario_scorer).__name__}"
            )
        self._scenario_scorer = scenario_scorer

        if safety_gate is None:
            self._safety_gate = RecommendationSafetyGate()
        elif isinstance(safety_gate, RecommendationSafetyGate):
            self._safety_gate = safety_gate
        else:
            raise TypeError(
                "safety_gate must be RecommendationSafetyGate or None, "
                f"got {type(safety_gate).__name__}"
            )

        if constraint_resolver is None:
            self._constraint_resolver = ConstraintResolver()
        elif isinstance(constraint_resolver, ConstraintResolver):
            self._constraint_resolver = constraint_resolver
        else:
            raise TypeError(
                "constraint_resolver must be ConstraintResolver or None, "
                f"got {type(constraint_resolver).__name__}"
            )

        if candidate_selector is None:
            self._candidate_selector = CandidateVariableSelector()
        elif isinstance(candidate_selector, CandidateVariableSelector):
            self._candidate_selector = candidate_selector
        else:
            raise TypeError(
                "candidate_selector must be CandidateVariableSelector or None, "
                f"got {type(candidate_selector).__name__}"
            )

        if grid_generator is None:
            self._grid_generator = CandidateGridGenerator()
        elif isinstance(grid_generator, CandidateGridGenerator):
            self._grid_generator = grid_generator
        else:
            raise TypeError(
                "grid_generator must be CandidateGridGenerator or None, "
                f"got {type(grid_generator).__name__}"
            )

        if scenario_ranker is None:
            self._scenario_ranker = CandidateScenarioRanker()
        elif isinstance(scenario_ranker, CandidateScenarioRanker):
            self._scenario_ranker = scenario_ranker
        else:
            raise TypeError(
                "scenario_ranker must be CandidateScenarioRanker or None, "
                f"got {type(scenario_ranker).__name__}"
            )

        if recommendation_generator is None:
            self._recommendation_generator = RankedScenarioRecommendationGenerator()
        elif isinstance(
            recommendation_generator,
            RankedScenarioRecommendationGenerator,
        ):
            self._recommendation_generator = recommendation_generator
        else:
            raise TypeError(
                "recommendation_generator must be "
                "RankedScenarioRecommendationGenerator or None, "
                f"got {type(recommendation_generator).__name__}"
            )

        if policy is None:
            self._policy = RecommendationPipelinePolicy()
        elif isinstance(policy, RecommendationPipelinePolicy):
            self._policy = policy.model_copy(deep=True)
        else:
            raise TypeError(
                "policy must be RecommendationPipelinePolicy or None, "
                f"got {type(policy).__name__}"
            )

    def get_metadata(self) -> dict[str, ScalarMetadataValue]:
        """Return scalar-only orchestration metadata for the current policy."""
        policy = self._policy
        return {
            "stop_on_safety_refusal": policy.stop_on_safety_refusal,
            "stop_on_constraint_refusal": policy.stop_on_constraint_refusal,
            "stop_on_empty_candidates": policy.stop_on_empty_candidates,
            "stop_on_grid_refusal_or_empty": policy.stop_on_grid_refusal_or_empty,
            "stop_on_scoring_refusal": policy.stop_on_scoring_refusal,
            "stop_on_ranking_refusal": policy.stop_on_ranking_refusal,
            "preserve_stage_outputs": policy.preserve_stage_outputs,
            "include_stage_warnings": policy.include_stage_warnings,
            "maximum_aggregated_warnings": policy.maximum_aggregated_warnings,
            "has_scenario_scorer": True,
            "orchestrates_safety": True,
            "orchestrates_constraint_resolution": True,
            "orchestrates_candidate_selection": True,
            "orchestrates_grid_generation": True,
            "orchestrates_scenario_scoring": True,
            "orchestrates_scenario_ranking": True,
            "orchestrates_recommendation_generation": True,
            "performs_model_fit": False,
            "performs_model_refit": False,
            "computes_residual_anomaly": False,
            "computes_extrapolation": False,
            "computes_uncertainty": False,
        }

    def run(
        self,
        request: RecommendationPipelineRequest,
    ) -> RecommendationPipelineOutcome:
        """Execute the recommendation pipeline for a validated request.

        Args:
            request: Validated pipeline request.

        Returns:
            Outcome wrapping an immutable pipeline report.
        """
        if not isinstance(request, RecommendationPipelineRequest):
            raise TypeError(
                "request must be RecommendationPipelineRequest, "
                f"got {type(request).__name__}"
            )

        started_at = datetime.now(tz=UTC)
        perf_started = time.perf_counter()
        policy = self._policy.model_copy(deep=True)

        before_snapshots = self._snapshot_component_metadata()

        safety_decision: RecommendationSafetyDecision | None = None
        constraint_report: ConstraintResolutionReport | None = None
        constraint_outcome: ConstraintResolutionOutcome | None = None
        candidate_set: CandidateVariableSet | None = None
        grid_report: CandidateGridReport | None = None
        scoring_report: ScenarioScoringReport | None = None
        ranking_report: ScenarioRankingReport | None = None
        final_result: RecommendationResult | None = None
        terminal_warning: str | None = None
        generation_executed = False

        executed: list[RecommendationPipelineStage] = []

        # --- Stage 1: SAFETY ---
        safety_decision = self._safety_gate.evaluate(
            request.recommendation_request,
            context=request.safety_context,
        )
        executed.append(RecommendationPipelineStage.SAFETY)

        if (
            safety_decision.status is RecommendationSafetyStatus.REFUSED
            and policy.stop_on_safety_refusal
        ):
            terminal_warning = (
                "Pipeline terminated after SAFETY because safety status is REFUSED."
            )
            final_result = self._build_terminal_result(
                request=request,
                safety_decision=safety_decision,
                terminal_stage=RecommendationPipelineStage.SAFETY,
                terminal_warning=terminal_warning,
                scoring_report=None,
                ranking_report=None,
            )
            return self._finalize_outcome(
                request=request,
                policy=policy,
                started_at=started_at,
                perf_started=perf_started,
                before_snapshots=before_snapshots,
                safety_decision=safety_decision,
                constraint_report=None,
                candidate_set=None,
                grid_report=None,
                scoring_report=None,
                ranking_report=None,
                final_result=final_result,
                executed=executed,
                terminal_stage=RecommendationPipelineStage.SAFETY,
                terminal_warning=terminal_warning,
                generation_executed=False,
            )

        # --- Stage 2: CONSTRAINT_RESOLUTION ---
        constraint_outcome = self._constraint_resolver.resolve(
            request.recommendation_request,
            safety_decision=safety_decision,
            industry_constraints=request.industry_constraints,
            user_overrides=request.user_overrides,
        )
        constraint_report = constraint_outcome.report
        executed.append(RecommendationPipelineStage.CONSTRAINT_RESOLUTION)

        if (
            constraint_report.status is ConstraintResolutionStatus.REFUSED
            and policy.stop_on_constraint_refusal
        ):
            terminal_warning = (
                "Pipeline terminated after CONSTRAINT_RESOLUTION because "
                "constraint resolution status is REFUSED."
            )
            final_result = self._build_terminal_result(
                request=request,
                safety_decision=safety_decision,
                terminal_stage=RecommendationPipelineStage.CONSTRAINT_RESOLUTION,
                terminal_warning=terminal_warning,
                scoring_report=None,
                ranking_report=None,
            )
            return self._finalize_outcome(
                request=request,
                policy=policy,
                started_at=started_at,
                perf_started=perf_started,
                before_snapshots=before_snapshots,
                safety_decision=safety_decision,
                constraint_report=constraint_report,
                candidate_set=None,
                grid_report=None,
                scoring_report=None,
                ranking_report=None,
                final_result=final_result,
                executed=executed,
                terminal_stage=RecommendationPipelineStage.CONSTRAINT_RESOLUTION,
                terminal_warning=terminal_warning,
                generation_executed=False,
            )

        # --- Stage 3: CANDIDATE_SELECTION ---
        selection_outcome = self._candidate_selector.select(
            request.recommendation_request,
            safety_decision=safety_decision,
            resolution=constraint_outcome,
        )
        candidate_set = selection_outcome.candidate_set
        executed.append(RecommendationPipelineStage.CANDIDATE_SELECTION)

        if (
            len(candidate_set.candidates) == 0
            and policy.stop_on_empty_candidates
        ):
            terminal_warning = (
                "Pipeline terminated after CANDIDATE_SELECTION because no "
                "candidate variables were selected."
            )
            final_result = self._build_terminal_result(
                request=request,
                safety_decision=safety_decision,
                terminal_stage=RecommendationPipelineStage.CANDIDATE_SELECTION,
                terminal_warning=terminal_warning,
                scoring_report=None,
                ranking_report=None,
            )
            return self._finalize_outcome(
                request=request,
                policy=policy,
                started_at=started_at,
                perf_started=perf_started,
                before_snapshots=before_snapshots,
                safety_decision=safety_decision,
                constraint_report=constraint_report,
                candidate_set=candidate_set,
                grid_report=None,
                scoring_report=None,
                ranking_report=None,
                final_result=final_result,
                executed=executed,
                terminal_stage=RecommendationPipelineStage.CANDIDATE_SELECTION,
                terminal_warning=terminal_warning,
                generation_executed=False,
            )

        # --- Stage 4: GRID_GENERATION ---
        grid_outcome = self._grid_generator.generate(candidate_set)
        grid_report = grid_outcome.report
        executed.append(RecommendationPipelineStage.GRID_GENERATION)

        grid_should_stop = (
            grid_report.status is CandidateGridStatus.REFUSED
            or grid_report.status is CandidateGridStatus.EMPTY
            or len(grid_report.scenarios) == 0
        )
        if grid_should_stop and policy.stop_on_grid_refusal_or_empty:
            terminal_warning = (
                "Pipeline terminated after GRID_GENERATION because the grid "
                f"status is {grid_report.status.value} or scenarios are empty."
            )
            final_result = self._build_terminal_result(
                request=request,
                safety_decision=safety_decision,
                terminal_stage=RecommendationPipelineStage.GRID_GENERATION,
                terminal_warning=terminal_warning,
                scoring_report=None,
                ranking_report=None,
            )
            return self._finalize_outcome(
                request=request,
                policy=policy,
                started_at=started_at,
                perf_started=perf_started,
                before_snapshots=before_snapshots,
                safety_decision=safety_decision,
                constraint_report=constraint_report,
                candidate_set=candidate_set,
                grid_report=grid_report,
                scoring_report=None,
                ranking_report=None,
                final_result=final_result,
                executed=executed,
                terminal_stage=RecommendationPipelineStage.GRID_GENERATION,
                terminal_warning=terminal_warning,
                generation_executed=False,
            )

        # --- Stage 5: SCENARIO_SCORING ---
        scoring_request = self._try_build_scoring_request(
            request=request,
            grid_report=grid_report,
        )
        if scoring_request is None:
            terminal_warning = (
                "Pipeline terminated after GRID_GENERATION because a valid "
                "ScenarioScoringRequest could not be constructed."
            )
            final_result = self._build_terminal_result(
                request=request,
                safety_decision=safety_decision,
                terminal_stage=RecommendationPipelineStage.GRID_GENERATION,
                terminal_warning=terminal_warning,
                scoring_report=None,
                ranking_report=None,
            )
            return self._finalize_outcome(
                request=request,
                policy=policy,
                started_at=started_at,
                perf_started=perf_started,
                before_snapshots=before_snapshots,
                safety_decision=safety_decision,
                constraint_report=constraint_report,
                candidate_set=candidate_set,
                grid_report=grid_report,
                scoring_report=None,
                ranking_report=None,
                final_result=final_result,
                executed=executed,
                terminal_stage=RecommendationPipelineStage.GRID_GENERATION,
                terminal_warning=terminal_warning,
                generation_executed=False,
            )

        scoring_outcome = self._scenario_scorer.score(scoring_request)
        scoring_report = scoring_outcome.report
        executed.append(RecommendationPipelineStage.SCENARIO_SCORING)

        if (
            scoring_report.status is ScenarioScoringStatus.REFUSED
            and policy.stop_on_scoring_refusal
        ):
            terminal_warning = (
                "Pipeline terminated after SCENARIO_SCORING because scoring "
                "status is REFUSED."
            )
            final_result = self._build_terminal_result(
                request=request,
                safety_decision=safety_decision,
                terminal_stage=RecommendationPipelineStage.SCENARIO_SCORING,
                terminal_warning=terminal_warning,
                scoring_report=scoring_report,
                ranking_report=None,
            )
            return self._finalize_outcome(
                request=request,
                policy=policy,
                started_at=started_at,
                perf_started=perf_started,
                before_snapshots=before_snapshots,
                safety_decision=safety_decision,
                constraint_report=constraint_report,
                candidate_set=candidate_set,
                grid_report=grid_report,
                scoring_report=scoring_report,
                ranking_report=None,
                final_result=final_result,
                executed=executed,
                terminal_stage=RecommendationPipelineStage.SCENARIO_SCORING,
                terminal_warning=terminal_warning,
                generation_executed=False,
            )

        # --- Stage 6: SCENARIO_RANKING ---
        ranking_request = self._try_build_ranking_request(
            request=request,
            grid_report=grid_report,
            scoring_report=scoring_report,
        )
        if ranking_request is None:
            terminal_warning = (
                "Pipeline terminated after SCENARIO_SCORING because a valid "
                "ScenarioRankingRequest could not be constructed."
            )
            final_result = self._build_terminal_result(
                request=request,
                safety_decision=safety_decision,
                terminal_stage=RecommendationPipelineStage.SCENARIO_SCORING,
                terminal_warning=terminal_warning,
                scoring_report=scoring_report,
                ranking_report=None,
            )
            return self._finalize_outcome(
                request=request,
                policy=policy,
                started_at=started_at,
                perf_started=perf_started,
                before_snapshots=before_snapshots,
                safety_decision=safety_decision,
                constraint_report=constraint_report,
                candidate_set=candidate_set,
                grid_report=grid_report,
                scoring_report=scoring_report,
                ranking_report=None,
                final_result=final_result,
                executed=executed,
                terminal_stage=RecommendationPipelineStage.SCENARIO_SCORING,
                terminal_warning=terminal_warning,
                generation_executed=False,
            )

        ranking_outcome = self._scenario_ranker.rank(ranking_request)
        ranking_report = ranking_outcome.report
        executed.append(RecommendationPipelineStage.SCENARIO_RANKING)

        if (
            ranking_report.status is ScenarioRankingStatus.REFUSED
            and policy.stop_on_ranking_refusal
        ):
            terminal_warning = (
                "Pipeline terminated after SCENARIO_RANKING because ranking "
                "status is REFUSED."
            )
            final_result = self._build_terminal_result(
                request=request,
                safety_decision=safety_decision,
                terminal_stage=RecommendationPipelineStage.SCENARIO_RANKING,
                terminal_warning=terminal_warning,
                scoring_report=scoring_report,
                ranking_report=ranking_report,
            )
            return self._finalize_outcome(
                request=request,
                policy=policy,
                started_at=started_at,
                perf_started=perf_started,
                before_snapshots=before_snapshots,
                safety_decision=safety_decision,
                constraint_report=constraint_report,
                candidate_set=candidate_set,
                grid_report=grid_report,
                scoring_report=scoring_report,
                ranking_report=ranking_report,
                final_result=final_result,
                executed=executed,
                terminal_stage=RecommendationPipelineStage.SCENARIO_RANKING,
                terminal_warning=terminal_warning,
                generation_executed=False,
            )

        # --- Stage 7: RECOMMENDATION_GENERATION ---
        generation_request = self._try_build_generation_request(
            request=request,
            safety_decision=safety_decision,
            grid_report=grid_report,
            ranking_report=ranking_report,
        )
        if generation_request is None:
            terminal_warning = (
                "Pipeline terminated after SCENARIO_RANKING because a valid "
                "RecommendationGenerationRequest could not be constructed."
            )
            final_result = self._build_terminal_result(
                request=request,
                safety_decision=safety_decision,
                terminal_stage=RecommendationPipelineStage.SCENARIO_RANKING,
                terminal_warning=terminal_warning,
                scoring_report=scoring_report,
                ranking_report=ranking_report,
            )
            return self._finalize_outcome(
                request=request,
                policy=policy,
                started_at=started_at,
                perf_started=perf_started,
                before_snapshots=before_snapshots,
                safety_decision=safety_decision,
                constraint_report=constraint_report,
                candidate_set=candidate_set,
                grid_report=grid_report,
                scoring_report=scoring_report,
                ranking_report=ranking_report,
                final_result=final_result,
                executed=executed,
                terminal_stage=RecommendationPipelineStage.SCENARIO_RANKING,
                terminal_warning=terminal_warning,
                generation_executed=False,
            )

        generation_outcome = self._recommendation_generator.generate(generation_request)
        final_result = generation_outcome.result
        executed.append(RecommendationPipelineStage.RECOMMENDATION_GENERATION)
        generation_executed = True

        return self._finalize_outcome(
            request=request,
            policy=policy,
            started_at=started_at,
            perf_started=perf_started,
            before_snapshots=before_snapshots,
            safety_decision=safety_decision,
            constraint_report=constraint_report,
            candidate_set=candidate_set,
            grid_report=grid_report,
            scoring_report=scoring_report,
            ranking_report=ranking_report,
            final_result=final_result,
            executed=executed,
            terminal_stage=RecommendationPipelineStage.RECOMMENDATION_GENERATION,
            terminal_warning=None,
            generation_executed=generation_executed,
        )

    def _snapshot_component_metadata(
        self,
    ) -> dict[str, dict[str, ScalarMetadataValue]]:
        return {
            "safety_gate": _stable_metadata_snapshot(self._safety_gate.get_metadata()),
            "constraint_resolver": _stable_metadata_snapshot(
                self._constraint_resolver.get_metadata()
            ),
            "candidate_selector": _stable_metadata_snapshot(
                self._candidate_selector.get_metadata()
            ),
            "grid_generator": _stable_metadata_snapshot(
                self._grid_generator.get_metadata()
            ),
            "scenario_scorer": _stable_metadata_snapshot(
                self._scenario_scorer.get_metadata()
            ),
            "scenario_ranker": _stable_metadata_snapshot(
                self._scenario_ranker.get_metadata()
            ),
            "recommendation_generator": _stable_metadata_snapshot(
                self._recommendation_generator.get_metadata()
            ),
        }

    def _assert_component_metadata_unchanged(
        self,
        before: dict[str, dict[str, ScalarMetadataValue]],
    ) -> None:
        after = self._snapshot_component_metadata()
        for name, before_meta in before.items():
            after_meta = after[name]
            if after_meta != before_meta:
                raise ProcessIntelligenceError(
                    f"{name} stable metadata changed during pipeline execution"
                )

    def _try_build_scoring_request(
        self,
        *,
        request: RecommendationPipelineRequest,
        grid_report: CandidateGridReport,
    ) -> ScenarioScoringRequest | None:
        try:
            return ScenarioScoringRequest(
                grid=grid_report,
                feature_columns=list(request.feature_columns),
                baseline_features=dict(request.baseline_features),
                target_column=request.target_column,
                metadata={
                    "pipeline_stage": RecommendationPipelineStage.SCENARIO_SCORING.value,
                },
            )
        except (ValidationError, ValueError, TypeError):
            return None

    def _try_build_ranking_request(
        self,
        *,
        request: RecommendationPipelineRequest,
        grid_report: CandidateGridReport,
        scoring_report: ScenarioScoringReport,
    ) -> ScenarioRankingRequest | None:
        try:
            return ScenarioRankingRequest(
                grid=grid_report,
                scoring=scoring_report,
                quality_direction=request.quality_direction,
                quality_target=request.quality_target,
                metadata={
                    "pipeline_stage": RecommendationPipelineStage.SCENARIO_RANKING.value,
                },
            )
        except (ValidationError, ValueError, TypeError):
            return None

    def _try_build_generation_request(
        self,
        *,
        request: RecommendationPipelineRequest,
        safety_decision: RecommendationSafetyDecision,
        grid_report: CandidateGridReport,
        ranking_report: ScenarioRankingReport,
    ) -> RecommendationGenerationRequest | None:
        try:
            return RecommendationGenerationRequest(
                request=request.recommendation_request,
                safety_decision=safety_decision,
                grid=grid_report,
                ranking=ranking_report,
                extrapolation_evaluated=request.extrapolation_evaluated,
                extrapolation_flag=request.extrapolation_flag,
                uncertainty_available=request.uncertainty_available,
                uncertainty_acceptable=request.uncertainty_acceptable,
                metadata={
                    "pipeline_stage": (
                        RecommendationPipelineStage.RECOMMENDATION_GENERATION.value
                    ),
                },
            )
        except (ValidationError, ValueError, TypeError):
            return None

    def _build_terminal_result(
        self,
        *,
        request: RecommendationPipelineRequest,
        safety_decision: RecommendationSafetyDecision,
        terminal_stage: RecommendationPipelineStage,
        terminal_warning: str,
        scoring_report: ScenarioScoringReport | None,
        ranking_report: ScenarioRankingReport | None,
    ) -> RecommendationResult:
        if safety_decision.status is RecommendationSafetyStatus.REFUSED:
            status = RecommendationStatus.REFUSED
        else:
            status = RecommendationStatus.READY_FOR_OPTIMIZATION

        baseline_prediction: float | None = None
        baseline_anomaly_score: float | None = None
        ranking_baseline = _extract_baselines_from_ranking(ranking_report)
        scoring_baseline = _extract_baselines_from_scoring(scoring_report)
        if ranking_baseline[0] is not None or ranking_baseline[1] is not None:
            baseline_prediction, baseline_anomaly_score = ranking_baseline
        elif scoring_baseline[0] is not None or scoring_baseline[1] is not None:
            baseline_prediction, baseline_anomaly_score = scoring_baseline

        disclaimer = safety_decision.disclaimer
        if not disclaimer:
            disclaimer = DEFAULT_RECOMMENDATION_DISCLAIMER

        warnings = [terminal_warning]
        return RecommendationResult(
            status=status,
            objective=request.recommendation_request.objective,
            safety_decision=safety_decision,
            changes=[],
            baseline_prediction=baseline_prediction,
            proposed_prediction=None,
            baseline_anomaly_score=baseline_anomaly_score,
            proposed_anomaly_score=None,
            confidence=0.0,
            extrapolation_flag=request.extrapolation_flag,
            uncertainty_available=request.uncertainty_available,
            disclaimer=disclaimer,
            generated_at=datetime.now(tz=UTC),
            warnings=warnings,
            metadata={
                "recommendation_generated": False,
                "model_refit_performed": False,
                "pipeline_terminal_stage": terminal_stage.value,
                "pipeline_orchestration_only": True,
            },
        )

    def _aggregate_warnings(
        self,
        *,
        policy: RecommendationPipelinePolicy,
        safety_decision: RecommendationSafetyDecision,
        constraint_report: ConstraintResolutionReport | None,
        candidate_set: CandidateVariableSet | None,
        grid_report: CandidateGridReport | None,
        scoring_report: ScenarioScoringReport | None,
        ranking_report: ScenarioRankingReport | None,
        final_result: RecommendationResult,
        terminal_warning: str | None,
    ) -> list[str]:
        collected: list[str] = []

        if policy.include_stage_warnings:
            for message in safety_decision.messages:
                _append_unique(collected, message)
            if constraint_report is not None:
                for message in constraint_report.warnings:
                    _append_unique(collected, message)
            if candidate_set is not None:
                for message in candidate_set.warnings:
                    _append_unique(collected, message)
            if grid_report is not None:
                for message in grid_report.warnings:
                    _append_unique(collected, message)
            if scoring_report is not None:
                for message in scoring_report.warnings:
                    _append_unique(collected, message)
            if ranking_report is not None:
                for message in ranking_report.warnings:
                    _append_unique(collected, message)
            for message in final_result.warnings:
                _append_unique(collected, message)
        else:
            for message in final_result.warnings:
                _append_unique(collected, message)

        if terminal_warning is not None:
            _append_unique(collected, terminal_warning)

        if not policy.preserve_stage_outputs:
            _append_unique(collected, _PRESERVE_OUTPUTS_WARNING)

        limit = policy.maximum_aggregated_warnings
        if len(collected) <= limit:
            return collected

        trimmed = collected[: max(limit - 1, 0)]
        if _OMISSION_WARNING not in trimmed and limit >= 1:
            trimmed.append(_OMISSION_WARNING)
        return trimmed[:limit]

    def _build_report_metadata(
        self,
        *,
        request: RecommendationPipelineRequest,
        policy: RecommendationPipelinePolicy,
        executed: list[RecommendationPipelineStage],
        skipped: list[RecommendationPipelineStage],
        terminal_stage: RecommendationPipelineStage,
        safety_decision: RecommendationSafetyDecision,
        final_result: RecommendationResult,
        generation_executed: bool,
    ) -> dict[str, ScalarMetadataValue]:
        executed_set = set(executed)
        return {
            "executed_stage_count": len(executed),
            "skipped_stage_count": len(skipped),
            "terminal_stage": terminal_stage.value,
            "safety_status": safety_decision.status.value,
            "final_recommendation_status": final_result.status.value,
            "constraint_resolution_executed": (
                RecommendationPipelineStage.CONSTRAINT_RESOLUTION in executed_set
            ),
            "candidate_selection_executed": (
                RecommendationPipelineStage.CANDIDATE_SELECTION in executed_set
            ),
            "grid_generation_executed": (
                RecommendationPipelineStage.GRID_GENERATION in executed_set
            ),
            "scenario_scoring_executed": (
                RecommendationPipelineStage.SCENARIO_SCORING in executed_set
            ),
            "scenario_ranking_executed": (
                RecommendationPipelineStage.SCENARIO_RANKING in executed_set
            ),
            "recommendation_generation_executed": (
                RecommendationPipelineStage.RECOMMENDATION_GENERATION in executed_set
            ),
            "recommendation_generated": (
                generation_executed
                and final_result.status is RecommendationStatus.GENERATED
            ),
            "model_fit_performed": False,
            "model_refit_performed": False,
            "residual_scoring_performed": False,
            "extrapolation_evaluated": request.extrapolation_evaluated,
            "uncertainty_available": request.uncertainty_available,
            "stage_outputs_preserved": True,
            "preserve_stage_outputs_requested": policy.preserve_stage_outputs,
            "pipeline_orchestration_only": True,
        }

    def _finalize_outcome(
        self,
        *,
        request: RecommendationPipelineRequest,
        policy: RecommendationPipelinePolicy,
        started_at: datetime,
        perf_started: float,
        before_snapshots: dict[str, dict[str, ScalarMetadataValue]],
        safety_decision: RecommendationSafetyDecision,
        constraint_report: ConstraintResolutionReport | None,
        candidate_set: CandidateVariableSet | None,
        grid_report: CandidateGridReport | None,
        scoring_report: ScenarioScoringReport | None,
        ranking_report: ScenarioRankingReport | None,
        final_result: RecommendationResult,
        executed: list[RecommendationPipelineStage],
        terminal_stage: RecommendationPipelineStage,
        terminal_warning: str | None,
        generation_executed: bool,
    ) -> RecommendationPipelineOutcome:
        self._assert_component_metadata_unchanged(before_snapshots)

        skipped = _stages_after(terminal_stage)
        completed_at = datetime.now(tz=UTC)
        total_seconds = max(0.0, time.perf_counter() - perf_started)

        warnings = self._aggregate_warnings(
            policy=policy,
            safety_decision=safety_decision,
            constraint_report=constraint_report,
            candidate_set=candidate_set,
            grid_report=grid_report,
            scoring_report=scoring_report,
            ranking_report=ranking_report,
            final_result=final_result,
            terminal_warning=terminal_warning,
        )

        report = RecommendationPipelineReport(
            status=final_result.status,
            terminal_stage=terminal_stage,
            final_result=final_result,
            safety_decision=safety_decision,
            constraint_report=constraint_report,
            candidate_set=candidate_set,
            grid_report=grid_report,
            scoring_report=scoring_report,
            ranking_report=ranking_report,
            executed_stages=list(executed),
            skipped_stages=list(skipped),
            started_at=started_at,
            completed_at=completed_at,
            total_seconds=total_seconds,
            warnings=warnings,
            metadata=self._build_report_metadata(
                request=request,
                policy=policy,
                executed=executed,
                skipped=skipped,
                terminal_stage=terminal_stage,
                safety_decision=safety_decision,
                final_result=final_result,
                generation_executed=generation_executed,
            ),
        )
        return RecommendationPipelineOutcome(report=report)
