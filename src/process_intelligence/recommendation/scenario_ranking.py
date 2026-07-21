"""Deterministic candidate scenario ranking by objective (Step 9E).

Normalizes Step 9D quality predictions and anomaly scores into objective
benefits, applies a change-magnitude penalty, and ranks scenarios without
emitting recommendations or re-scoring models.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator

from process_intelligence.core.exceptions import DataValidationError
from process_intelligence.recommendation.candidate_grid import (
    CandidateGridReport,
    CandidateGridStatus,
    CandidateScenario,
    CandidateScenarioType,
)
from process_intelligence.recommendation.enums import RecommendationObjective
from process_intelligence.recommendation.scenario_scoring import (
    CandidateScenarioScore,
    ScenarioScoringReport,
    ScenarioScoringStatus,
)
from process_intelligence.recommendation.schemas import ScalarMetadataValue

_SCENARIO_ID_RE = re.compile(r"^SCN-(\d{6})$")

_WARNING_GRID_REFUSED = (
    "Candidate grid status is REFUSED; scenario ranking was refused."
)
_WARNING_SCORING_REFUSED = (
    "Scenario scoring status is REFUSED; scenario ranking was refused."
)
_WARNING_NO_SCENARIOS = "Scoring report contains no scenarios to rank."
_WARNING_PARTIAL_REFUSED = (
    "Partial scoring report is not allowed; scenario ranking was refused."
)
_WARNING_PARTIAL_USED = (
    "Partial scoring report was used; ranking is limited to scenarios with "
    "required model outputs."
)
_WARNING_REQUIRED_OUTPUT_EXCLUDED = (
    "One or more scenarios were excluded because required model outputs "
    "were missing."
)
_WARNING_BASELINE_OUTPUT_MISSING = (
    "Baseline is missing required model outputs for the requested objective; "
    "scenario ranking was refused."
)
_WARNING_ALL_OUTPUTS_MISSING = (
    "Required model outputs are unavailable for all scenarios; "
    "scenario ranking was refused."
)
_WARNING_QUALITY_DIRECTION = (
    "Quality optimization direction was applied when ranking predicted quality."
)
_WARNING_QUALITY_TARGET = (
    "Quality target distance was used for TARGET quality optimization."
)
_WARNING_COMPONENT_WORSENING = (
    "One or more scenarios were excluded because a required objective "
    "component worsened relative to baseline."
)
_WARNING_MINIMUM_IMPROVEMENT = (
    "One or more scenarios did not meet the minimum improvement thresholds."
)
_WARNING_CHANGE_PENALTY = (
    "Change-magnitude penalty was applied to non-baseline composite scores."
)
_WARNING_TRUNCATION = (
    "Ranked scenario list was truncated to the maximum_ranked_scenarios limit."
)
_WARNING_NO_IMPROVEMENT = (
    "No scenario improved sufficiently over baseline for the requested objective."
)
_WARNING_RELATIVE_HEURISTIC = (
    "Ranking scores are relative heuristics, not probabilities or causal effects."
)
_WARNING_VERIFICATION = (
    "Ranked scenarios require extrapolation, uncertainty, domain, safety, and "
    "operational verification."
)
_WARNING_NO_GUARANTEE = (
    "Model scoring does not guarantee real-process improvement."
)

_SCENARIO_WARNING_QUALITY_MISSING = "required quality output missing"
_SCENARIO_WARNING_ANOMALY_MISSING = "required anomaly output missing"
_SCENARIO_WARNING_QUALITY_REQUIREMENT = "quality requirement not met"
_SCENARIO_WARNING_ANOMALY_REQUIREMENT = "anomaly requirement not met"
_SCENARIO_WARNING_COMPONENT_WORSENED = "required component worsened"
_SCENARIO_WARNING_COMPOSITE_THRESHOLD = "composite advantage below threshold"
_SCENARIO_WARNING_BASELINE = "scenario is the unchanged baseline"
_SCENARIO_WARNING_INELIGIBLE = (
    "scenario is not eligible for recommendation selection"
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


def _require_unit_interval_float(value: object, *, field_name: str) -> float:
    number = _require_finite_float(value, field_name=field_name)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{field_name} must be in [0.0, 1.0], got {number}")
    return number


def _require_optional_unit_interval_float(
    value: object,
    *,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    return _require_unit_interval_float(value, field_name=field_name)


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


def _requires_quality(objective: RecommendationObjective) -> bool:
    return objective in {
        RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
    }


def _requires_anomaly(objective: RecommendationObjective) -> bool:
    return objective in {
        RecommendationObjective.REDUCE_ANOMALY_SCORE,
        RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
    }


def _score_has_required_outputs(
    score: CandidateScenarioScore,
    *,
    objective: RecommendationObjective,
) -> bool:
    if _requires_quality(objective) and not score.quality_scored:
        return False
    if _requires_anomaly(objective) and not score.anomaly_scored:
        return False
    return True


class QualityOptimizationDirection(StrEnum):
    """Direction used to interpret predicted quality as improvement.

    ``MAXIMIZE`` treats higher predictions as better. ``MINIMIZE`` treats lower
    predictions as better. ``TARGET`` treats smaller absolute distance to a
    target value as better.
    """

    MAXIMIZE = "MAXIMIZE"
    MINIMIZE = "MINIMIZE"
    TARGET = "TARGET"


class ScenarioRankingStatus(StrEnum):
    """Outcome status for objective-based scenario ranking.

    ``RANKED`` means at least one eligible non-baseline candidate exists under a
    complete scoring report. ``PARTIAL`` means a partial scoring report was
    allowed and at least one eligible candidate remains. ``NO_IMPROVEMENT`` means
    ranking was possible but no candidate beat baseline under the objective.
    ``REFUSED`` means ranking could not be performed.
    """

    RANKED = "RANKED"
    PARTIAL = "PARTIAL"
    NO_IMPROVEMENT = "NO_IMPROVEMENT"
    REFUSED = "REFUSED"


class ScenarioRankingPolicy(BaseModel):
    """Tunable policy for deterministic objective-based scenario ranking.

    Controls benefit weights, change penalties, improvement thresholds, and
    ranking inclusion limits. Does not score models or emit recommendations.
    """

    quality_weight: float = 0.5
    anomaly_weight: float = 0.5
    change_penalty_weight: float = 0.10
    minimum_quality_improvement: float = 0.0
    minimum_anomaly_improvement: float = 0.0
    minimum_composite_advantage: float = 1e-12
    allow_required_component_worsening: bool = False
    allow_partial_scoring_report: bool = False
    prefer_fewer_changes: bool = True
    prefer_smaller_changes: bool = True
    require_baseline_scenario: bool = True
    include_ineligible_scenarios: bool = True
    maximum_ranked_scenarios: int = 5000

    @field_validator(
        "quality_weight",
        "anomaly_weight",
        "change_penalty_weight",
        mode="before",
    )
    @classmethod
    def _validate_unit_weights(cls, value: object) -> float:
        return _require_unit_interval_float(value, field_name="weight field")

    @field_validator(
        "minimum_quality_improvement",
        "minimum_anomaly_improvement",
        "minimum_composite_advantage",
        mode="before",
    )
    @classmethod
    def _validate_thresholds(cls, value: object) -> float:
        return _require_non_negative_finite_float(value, field_name="threshold field")

    @field_validator(
        "allow_required_component_worsening",
        "allow_partial_scoring_report",
        "prefer_fewer_changes",
        "prefer_smaller_changes",
        "require_baseline_scenario",
        "include_ineligible_scenarios",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="policy bool field")

    @field_validator("maximum_ranked_scenarios", mode="before")
    @classmethod
    def _validate_maximum_ranked_scenarios(cls, value: object) -> int:
        return _require_strict_int_ge(
            value,
            field_name="maximum_ranked_scenarios",
            minimum=1,
        )

    @model_validator(mode="after")
    def _validate_weight_presence(self) -> Self:
        if self.quality_weight == 0.0 and self.anomaly_weight == 0.0:
            raise ValueError(
                "quality_weight and anomaly_weight cannot both be 0"
            )
        return self


class ScenarioRankingRequest(BaseModel):
    """Request contract pairing a candidate grid with its scoring report.

    Requires aligned scenario identity fields between grid and scoring. Does not
    embed models, estimators, DataFrames, or final recommendations.
    """

    grid: CandidateGridReport
    scoring: ScenarioScoringReport
    quality_direction: QualityOptimizationDirection | None = None
    quality_target: float | None = None
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

    @field_validator("scoring", mode="before")
    @classmethod
    def _validate_scoring(cls, value: object) -> ScenarioScoringReport:
        if isinstance(value, ScenarioScoringReport):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return ScenarioScoringReport.model_validate(value)
        raise ValueError(
            f"scoring must be ScenarioScoringReport, got {type(value).__name__}"
        )

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
        return _require_optional_finite_float(value, field_name="quality_target")

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> dict[str, ScalarMetadataValue]:
        if value is None:
            return {}
        return _validate_scalar_metadata(value)

    @model_validator(mode="after")
    def _validate_request_consistency(self) -> Self:
        if self.grid.objective != self.scoring.objective:
            raise ValueError(
                "grid.objective must equal scoring.objective "
                f"(grid={self.grid.objective!r}, scoring={self.scoring.objective!r})"
            )

        scenarios = self.grid.scenarios
        scores = self.scoring.scores
        if len(scenarios) != self.scoring.requested_scenario_count:
            raise ValueError(
                "grid scenario count must equal scoring.requested_scenario_count "
                f"(grid={len(scenarios)}, requested={self.scoring.requested_scenario_count})"
            )
        if len(scenarios) != len(scores):
            # REFUSED scoring has empty scores; requested may still be > 0.
            if not (
                self.scoring.status is ScenarioScoringStatus.REFUSED
                and not scores
            ):
                raise ValueError(
                    "grid scenario count must equal scoring score count "
                    f"(grid={len(scenarios)}, scores={len(scores)})"
                )

        if scores:
            for scenario, score in zip(scenarios, scores, strict=True):
                if scenario.scenario_id != score.scenario_id:
                    raise ValueError(
                        "score scenario_id must match grid scenario_id "
                        f"(grid={scenario.scenario_id!r}, score={score.scenario_id!r})"
                    )
                if scenario.scenario_index != score.scenario_index:
                    raise ValueError(
                        "score scenario_index must match grid scenario_index "
                        f"(grid={scenario.scenario_index}, score={score.scenario_index})"
                    )
                if scenario.scenario_type != score.scenario_type:
                    raise ValueError(
                        "score scenario_type must match grid scenario_type "
                        f"(grid={scenario.scenario_type!r}, score={score.scenario_type!r})"
                    )
                if scenario.change_count != score.change_count:
                    raise ValueError(
                        "score change_count must match grid change_count "
                        f"(grid={scenario.change_count}, score={score.change_count})"
                    )
                if not math.isclose(
                    scenario.normalized_change_magnitude,
                    score.normalized_change_magnitude,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "score normalized_change_magnitude must match grid "
                        "normalized_change_magnitude "
                        f"(grid={scenario.normalized_change_magnitude}, "
                        f"score={score.normalized_change_magnitude})"
                    )

        needs_quality = _requires_quality(self.grid.objective)
        if needs_quality and self.quality_direction is None:
            raise ValueError(
                "quality_direction is required for objectives that use "
                "predicted quality"
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

        return self


class RankedCandidateScenario(BaseModel):
    """One ranked scenario with objective benefits and change costs.

    Preserves scenario values and deltas from the candidate grid and scoring
    report. Does not assert proven process improvement.
    """

    rank: int
    scenario_id: str
    scenario_index: int
    scenario_type: CandidateScenarioType
    variable_values: dict[str, float]
    changed_variables: list[str]
    deltas: dict[str, float]
    relative_deltas: dict[str, float | None]
    change_count: int
    normalized_change_magnitude: float
    quality_prediction: float | None
    anomaly_score: float | None
    quality_improvement_from_baseline: float | None
    anomaly_improvement_from_baseline: float | None
    normalized_quality_benefit: float | None
    normalized_anomaly_benefit: float | None
    objective_benefit_score: float
    normalized_change_cost: float
    change_penalty: float
    composite_score: float
    quality_requirement_met: bool | None
    anomaly_requirement_met: bool | None
    objective_requirements_met: bool
    selection_eligible: bool
    warnings: list[str] = Field(default_factory=list)

    @field_validator("rank", mode="before")
    @classmethod
    def _validate_rank(cls, value: object) -> int:
        return _require_strict_int_ge(value, field_name="rank", minimum=1)

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

    @field_validator("variable_values", mode="before")
    @classmethod
    def _validate_variable_values_before(cls, value: object) -> dict[str, float]:
        if not isinstance(value, dict):
            raise ValueError(
                f"variable_values must be a dict[str, float], got {type(value).__name__}"
            )
        cleaned: dict[str, float] = {}
        for key, raw in value.items():
            name = _require_non_empty_str(key, field_name="variable_values key")
            cleaned[name] = _require_finite_float(
                raw,
                field_name=f"variable_values[{name!r}]",
            )
        return cleaned

    @field_validator("changed_variables", mode="before")
    @classmethod
    def _validate_changed_variables_before(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"changed_variables must be a list[str], got {type(value).__name__}"
            )
        return list(value)

    @field_validator("changed_variables", mode="after")
    @classmethod
    def _validate_changed_variables(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(
            value,
            field_name="changed_variables",
        )

    @field_validator("deltas", mode="before")
    @classmethod
    def _validate_deltas_before(cls, value: object) -> dict[str, float]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(
                f"deltas must be a dict[str, float], got {type(value).__name__}"
            )
        cleaned: dict[str, float] = {}
        for key, raw in value.items():
            name = _require_non_empty_str(key, field_name="deltas key")
            cleaned[name] = _require_finite_float(raw, field_name=f"deltas[{name!r}]")
        return cleaned

    @field_validator("relative_deltas", mode="before")
    @classmethod
    def _validate_relative_deltas_before(
        cls,
        value: object,
    ) -> dict[str, float | None]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(
                f"relative_deltas must be a dict[str, float | None], "
                f"got {type(value).__name__}"
            )
        cleaned: dict[str, float | None] = {}
        for key, raw in value.items():
            name = _require_non_empty_str(key, field_name="relative_deltas key")
            cleaned[name] = _require_optional_finite_float(
                raw,
                field_name=f"relative_deltas[{name!r}]",
            )
        return cleaned

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
        "quality_improvement_from_baseline",
        "anomaly_improvement_from_baseline",
        mode="before",
    )
    @classmethod
    def _validate_optional_numeric(cls, value: object) -> float | None:
        return _require_optional_finite_float(value, field_name="numeric field")

    @field_validator(
        "normalized_quality_benefit",
        "normalized_anomaly_benefit",
        mode="before",
    )
    @classmethod
    def _validate_optional_unit_benefits(cls, value: object) -> float | None:
        return _require_optional_unit_interval_float(
            value,
            field_name="normalized benefit",
        )

    @field_validator(
        "objective_benefit_score",
        "normalized_change_cost",
        mode="before",
    )
    @classmethod
    def _validate_unit_scores(cls, value: object) -> float:
        return _require_unit_interval_float(value, field_name="unit score field")

    @field_validator("change_penalty", mode="before")
    @classmethod
    def _validate_change_penalty(cls, value: object) -> float:
        return _require_non_negative_finite_float(
            value,
            field_name="change_penalty",
        )

    @field_validator("composite_score", mode="before")
    @classmethod
    def _validate_composite_score(cls, value: object) -> float:
        return _require_finite_float(value, field_name="composite_score")

    @field_validator(
        "quality_requirement_met",
        "anomaly_requirement_met",
        mode="before",
    )
    @classmethod
    def _validate_optional_requirement_flags(cls, value: object) -> bool | None:
        if value is None:
            return None
        return _require_strict_bool(value, field_name="requirement flag")

    @field_validator(
        "objective_requirements_met",
        "selection_eligible",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="ranked bool field")

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
    def _validate_ranked_consistency(self) -> Self:
        match = _SCENARIO_ID_RE.fullmatch(self.scenario_id)
        assert match is not None
        if int(match.group(1)) != self.scenario_index:
            raise ValueError("scenario_id numeric suffix must equal scenario_index")

        for name in self.changed_variables:
            if name not in self.variable_values:
                raise ValueError(
                    f"changed_variables entry {name!r} missing from variable_values"
                )

        changed_set = set(self.changed_variables)
        if set(self.deltas.keys()) != changed_set:
            raise ValueError("deltas keys must exactly match changed_variables")
        if set(self.relative_deltas.keys()) != changed_set:
            raise ValueError(
                "relative_deltas keys must exactly match changed_variables"
            )
        if self.change_count != len(self.changed_variables):
            raise ValueError("change_count must equal len(changed_variables)")

        if self.scenario_type is CandidateScenarioType.BASELINE:
            if self.scenario_index != 0:
                raise ValueError("BASELINE requires scenario_index=0")
            if self.changed_variables:
                raise ValueError("BASELINE requires empty changed_variables")
            if self.selection_eligible:
                raise ValueError("BASELINE cannot be selection_eligible=True")
        elif self.scenario_type is CandidateScenarioType.SINGLE_VARIABLE:
            if self.change_count != 1:
                raise ValueError("SINGLE_VARIABLE requires change_count=1")
        elif self.scenario_type is CandidateScenarioType.MULTI_VARIABLE:
            if self.change_count < 2:
                raise ValueError("MULTI_VARIABLE requires change_count>=2")
        else:
            raise ValueError(f"unsupported scenario_type: {self.scenario_type!r}")

        required_flags: list[bool] = []
        if self.quality_requirement_met is not None:
            required_flags.append(self.quality_requirement_met)
        if self.anomaly_requirement_met is not None:
            required_flags.append(self.anomaly_requirement_met)
        if required_flags:
            expected_met = all(required_flags)
            if self.objective_requirements_met != expected_met:
                raise ValueError(
                    "objective_requirements_met must equal all present "
                    "component requirement flags"
                )

        if self.selection_eligible:
            if self.scenario_type is CandidateScenarioType.BASELINE:
                raise ValueError("selection_eligible=True cannot be BASELINE")
            if not self.objective_requirements_met:
                raise ValueError(
                    "selection_eligible=True requires objective_requirements_met=True"
                )
        return self


class ScenarioRankingReport(BaseModel):
    """Structured report of objective-ranked candidate scenarios.

    Identifies a best eligible non-baseline scenario when one exists. Does not
    generate ``RecommendationChange`` objects or claim causal improvement.
    """

    status: ScenarioRankingStatus
    objective: RecommendationObjective
    quality_direction: QualityOptimizationDirection | None
    quality_target: float | None
    baseline_scenario_id: str | None
    baseline_quality_prediction: float | None
    baseline_anomaly_score: float | None
    ranked_scenarios: list[RankedCandidateScenario]
    best_scenario_id: str | None
    best_nonbaseline_scenario_id: str | None
    baseline_is_top_ranked: bool
    requested_scenario_count: int
    evaluated_scenario_count: int
    returned_scenario_count: int
    eligible_scenario_count: int
    ineligible_scenario_count: int
    truncated_scenario_count: int
    evaluated_at: datetime
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)

    @field_validator("status", mode="before")
    @classmethod
    def _validate_status(cls, value: object) -> ScenarioRankingStatus:
        if isinstance(value, ScenarioRankingStatus):
            return value
        if isinstance(value, str):
            try:
                return ScenarioRankingStatus(value)
            except ValueError as exc:
                raise ValueError(f"invalid ScenarioRankingStatus: {value!r}") from exc
        raise ValueError(
            f"status must be ScenarioRankingStatus, got {type(value).__name__}"
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

    @field_validator(
        "quality_target",
        "baseline_quality_prediction",
        "baseline_anomaly_score",
        mode="before",
    )
    @classmethod
    def _validate_optional_numeric(cls, value: object) -> float | None:
        return _require_optional_finite_float(value, field_name="numeric field")

    @field_validator(
        "baseline_scenario_id",
        "best_scenario_id",
        "best_nonbaseline_scenario_id",
        mode="before",
    )
    @classmethod
    def _validate_optional_ids(cls, value: object) -> str | None:
        return _require_optional_non_empty_str(value, field_name="scenario id")

    @field_validator("ranked_scenarios", mode="before")
    @classmethod
    def _validate_ranked_before(cls, value: object) -> list[RankedCandidateScenario]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"ranked_scenarios must be a list[RankedCandidateScenario], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("ranked_scenarios", mode="after")
    @classmethod
    def _validate_ranked_after(
        cls,
        value: list[RankedCandidateScenario],
    ) -> list[RankedCandidateScenario]:
        return [item.model_copy(deep=True) for item in value]

    @field_validator("baseline_is_top_ranked", mode="before")
    @classmethod
    def _validate_baseline_is_top_ranked(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="baseline_is_top_ranked")

    @field_validator(
        "requested_scenario_count",
        "evaluated_scenario_count",
        "returned_scenario_count",
        "eligible_scenario_count",
        "ineligible_scenario_count",
        "truncated_scenario_count",
        mode="before",
    )
    @classmethod
    def _validate_counts(cls, value: object) -> int:
        return _require_strict_int_ge(value, field_name="count field", minimum=0)

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
        ranks = [item.rank for item in self.ranked_scenarios]
        if len(ranks) != len(set(ranks)):
            raise ValueError("ranked_scenarios ranks must be unique")
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(
                "ranked_scenarios ranks must be contiguous starting at 1 "
                "in ascending order"
            )
        if [item.rank for item in self.ranked_scenarios] != sorted(
            item.rank for item in self.ranked_scenarios
        ):
            raise ValueError("ranked_scenarios must be ordered by ascending rank")

        scenario_ids = [item.scenario_id for item in self.ranked_scenarios]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("ranked_scenarios scenario IDs must be unique")
        scenario_indexes = [item.scenario_index for item in self.ranked_scenarios]
        if len(scenario_indexes) != len(set(scenario_indexes)):
            raise ValueError("ranked_scenarios scenario indexes must be unique")

        if self.returned_scenario_count != len(self.ranked_scenarios):
            raise ValueError(
                "returned_scenario_count must equal len(ranked_scenarios)"
            )
        returned_eligible = sum(
            1 for item in self.ranked_scenarios if item.selection_eligible
        )
        if returned_eligible != self.eligible_scenario_count:
            raise ValueError(
                "eligible_scenario_count must equal the number of "
                "selection_eligible ranked scenarios"
            )

        if not (
            self.requested_scenario_count
            >= self.evaluated_scenario_count
            >= self.returned_scenario_count
        ):
            raise ValueError(
                "requested_scenario_count >= evaluated_scenario_count >= "
                "returned_scenario_count is required"
            )
        expected_truncated = (
            self.evaluated_scenario_count - self.returned_scenario_count
        )
        if self.truncated_scenario_count != expected_truncated:
            raise ValueError(
                "truncated_scenario_count must equal "
                "evaluated_scenario_count - returned_scenario_count"
            )

        if self.ranked_scenarios:
            if self.best_scenario_id != self.ranked_scenarios[0].scenario_id:
                raise ValueError(
                    "best_scenario_id must equal the rank-1 scenario_id when "
                    "ranked_scenarios is non-empty"
                )
        elif self.best_scenario_id is not None:
            raise ValueError(
                "best_scenario_id must be None when ranked_scenarios is empty"
            )

        eligible_ranked = [
            item
            for item in self.ranked_scenarios
            if item.selection_eligible
            and item.scenario_type is not CandidateScenarioType.BASELINE
        ]
        expected_best_nonbaseline = (
            eligible_ranked[0].scenario_id if eligible_ranked else None
        )
        if self.best_nonbaseline_scenario_id != expected_best_nonbaseline:
            raise ValueError(
                "best_nonbaseline_scenario_id must be the highest-ranked "
                "selection-eligible non-baseline scenario"
            )

        top_is_baseline = bool(
            self.ranked_scenarios
            and self.ranked_scenarios[0].scenario_type
            is CandidateScenarioType.BASELINE
        )
        if self.baseline_is_top_ranked != top_is_baseline:
            raise ValueError(
                "baseline_is_top_ranked must match whether rank 1 is BASELINE"
            )

        if self.baseline_scenario_id is not None:
            ranked_ids = {item.scenario_id for item in self.ranked_scenarios}
            if (
                self.baseline_scenario_id not in ranked_ids
                and self.status is not ScenarioRankingStatus.REFUSED
            ):
                # Baseline may be evaluated but omitted only if include policy
                # somehow dropped it; contract requires baseline preservation
                # when ranking succeeds.
                raise ValueError(
                    "baseline_scenario_id must appear in ranked_scenarios when set"
                )
            for item in self.ranked_scenarios:
                if item.scenario_id == self.baseline_scenario_id:
                    if item.selection_eligible:
                        raise ValueError("baseline scenario cannot be selection_eligible")
                    if item.scenario_type is not CandidateScenarioType.BASELINE:
                        raise ValueError(
                            "baseline_scenario_id must refer to a BASELINE scenario"
                        )
                    break

        if self.status is ScenarioRankingStatus.REFUSED:
            if self.ranked_scenarios:
                raise ValueError("REFUSED report must have empty ranked_scenarios")
            if self.best_scenario_id is not None:
                raise ValueError("REFUSED report must have best_scenario_id=None")
            if self.best_nonbaseline_scenario_id is not None:
                raise ValueError(
                    "REFUSED report must have best_nonbaseline_scenario_id=None"
                )
            if self.evaluated_scenario_count != 0:
                raise ValueError("REFUSED report must have evaluated_scenario_count=0")
            if self.returned_scenario_count != 0:
                raise ValueError("REFUSED report must have returned_scenario_count=0")
            if self.eligible_scenario_count != 0:
                raise ValueError("REFUSED report must have eligible_scenario_count=0")
        elif self.status is ScenarioRankingStatus.RANKED:
            if self.eligible_scenario_count < 1:
                raise ValueError("RANKED requires eligible_scenario_count >= 1")
            if self.best_nonbaseline_scenario_id is None:
                raise ValueError("RANKED requires best_nonbaseline_scenario_id")
        elif self.status is ScenarioRankingStatus.PARTIAL:
            if self.eligible_scenario_count < 1:
                raise ValueError("PARTIAL requires eligible_scenario_count >= 1")
            if self.best_nonbaseline_scenario_id is None:
                raise ValueError("PARTIAL requires best_nonbaseline_scenario_id")
        elif self.status is ScenarioRankingStatus.NO_IMPROVEMENT:
            if self.eligible_scenario_count != 0:
                raise ValueError(
                    "NO_IMPROVEMENT requires eligible_scenario_count == 0"
                )
            if self.best_nonbaseline_scenario_id is not None:
                raise ValueError(
                    "NO_IMPROVEMENT requires best_nonbaseline_scenario_id=None"
                )
        else:
            raise ValueError(f"unsupported ranking status: {self.status!r}")

        return self


@dataclass(frozen=True, slots=True)
class ScenarioRankingOutcome:
    """Immutable wrapper around a scenario ranking report.

    Holds only ``ScenarioRankingReport``. Does not store models, estimators,
    or DataFrames, and contains no business logic.
    """

    report: ScenarioRankingReport


@dataclass(frozen=True, slots=True)
class _ScoredPair:
    scenario: CandidateScenario
    score: CandidateScenarioScore


@dataclass
class _WorkingRank:
    scenario: CandidateScenario
    score: CandidateScenarioScore
    is_baseline: bool
    quality_improvement: float | None
    anomaly_improvement: float | None
    normalized_quality_benefit: float | None
    normalized_anomaly_benefit: float | None
    objective_benefit_score: float
    normalized_change_cost: float
    change_penalty: float
    composite_score: float
    quality_requirement_met: bool | None
    anomaly_requirement_met: bool | None
    objective_requirements_met: bool
    selection_eligible: bool
    warnings: list[str]


class CandidateScenarioRanker:
    """Rank scored candidate scenarios by objective without model calls.

    Applies quality-direction improvements, anomaly-score reductions, benefit
    normalization, and change penalties. Does not generate recommendations.
    """

    def __init__(
        self,
        *,
        policy: ScenarioRankingPolicy | None = None,
    ) -> None:
        if policy is None:
            resolved = ScenarioRankingPolicy()
        elif isinstance(policy, ScenarioRankingPolicy):
            resolved = policy.model_copy(deep=True)
        else:
            raise TypeError(
                "policy must be ScenarioRankingPolicy or None, "
                f"got {type(policy).__name__}"
            )
        self._policy = resolved

    def get_metadata(self) -> dict[str, ScalarMetadataValue]:
        """Return scalar ranker capability metadata for the current instance."""
        return {
            "quality_weight": self._policy.quality_weight,
            "anomaly_weight": self._policy.anomaly_weight,
            "change_penalty_weight": self._policy.change_penalty_weight,
            "minimum_quality_improvement": self._policy.minimum_quality_improvement,
            "minimum_anomaly_improvement": self._policy.minimum_anomaly_improvement,
            "minimum_composite_advantage": self._policy.minimum_composite_advantage,
            "allow_required_component_worsening": (
                self._policy.allow_required_component_worsening
            ),
            "allow_partial_scoring_report": self._policy.allow_partial_scoring_report,
            "prefer_fewer_changes": self._policy.prefer_fewer_changes,
            "prefer_smaller_changes": self._policy.prefer_smaller_changes,
            "require_baseline_scenario": self._policy.require_baseline_scenario,
            "include_ineligible_scenarios": self._policy.include_ineligible_scenarios,
            "maximum_ranked_scenarios": self._policy.maximum_ranked_scenarios,
            "performs_model_scoring": False,
            "performs_model_refit": False,
            "performs_scenario_ranking": True,
            "generates_recommendation": False,
            "ranking_is_relative_heuristic": True,
        }

    def rank(self, request: ScenarioRankingRequest) -> ScenarioRankingOutcome:
        """Rank scored scenarios by objective benefit minus change penalty.

        Args:
            request: Validated ranking request with aligned grid and scoring.

        Returns:
            Outcome wrapping a ``ScenarioRankingReport``.

        Raises:
            TypeError: If ``request`` is not a ``ScenarioRankingRequest``.
            DataValidationError: If valid objects contradict ranking contracts.
        """
        if not isinstance(request, ScenarioRankingRequest):
            raise TypeError(
                "request must be ScenarioRankingRequest, "
                f"got {type(request).__name__}"
            )

        policy = self._policy
        objective = request.grid.objective
        warnings: list[str] = []
        requested_count = len(request.grid.scenarios)
        used_partial = False

        self._validate_objective_weights(objective=objective, policy=policy)

        if request.grid.status is CandidateGridStatus.REFUSED:
            _append_unique_warning(warnings, _WARNING_GRID_REFUSED)
            return self._refused_outcome(
                request=request,
                warnings=warnings,
                requested_count=requested_count,
            )

        if request.scoring.status is ScenarioScoringStatus.REFUSED:
            _append_unique_warning(warnings, _WARNING_SCORING_REFUSED)
            return self._refused_outcome(
                request=request,
                warnings=warnings,
                requested_count=requested_count,
            )

        if not request.scoring.scores:
            _append_unique_warning(warnings, _WARNING_NO_SCENARIOS)
            return self._refused_outcome(
                request=request,
                warnings=warnings,
                requested_count=requested_count,
            )

        if request.scoring.status is ScenarioScoringStatus.PARTIAL:
            if not policy.allow_partial_scoring_report:
                _append_unique_warning(warnings, _WARNING_PARTIAL_REFUSED)
                return self._refused_outcome(
                    request=request,
                    warnings=warnings,
                    requested_count=requested_count,
                )
            used_partial = True
            _append_unique_warning(warnings, _WARNING_PARTIAL_USED)

        pairs = self._aligned_pairs(request)
        if policy.require_baseline_scenario:
            self._validate_required_baseline(pairs=pairs, request=request)

        baseline_pair = pairs[0] if pairs else None
        if (
            baseline_pair is not None
            and baseline_pair.scenario.scenario_type is CandidateScenarioType.BASELINE
        ):
            if not _score_has_required_outputs(
                baseline_pair.score,
                objective=objective,
            ):
                _append_unique_warning(warnings, _WARNING_BASELINE_OUTPUT_MISSING)
                return self._refused_outcome(
                    request=request,
                    warnings=warnings,
                    requested_count=requested_count,
                )
        elif policy.require_baseline_scenario:
            raise DataValidationError(
                "require_baseline_scenario=True requires the first scenario "
                "to be BASELINE"
            )

        evaluable: list[_ScoredPair] = []
        excluded_missing = 0
        for pair in pairs:
            if _score_has_required_outputs(pair.score, objective=objective):
                evaluable.append(pair)
            else:
                excluded_missing += 1

        if excluded_missing:
            _append_unique_warning(warnings, _WARNING_REQUIRED_OUTPUT_EXCLUDED)

        if not evaluable:
            _append_unique_warning(warnings, _WARNING_ALL_OUTPUTS_MISSING)
            return self._refused_outcome(
                request=request,
                warnings=warnings,
                requested_count=requested_count,
            )

        if request.quality_direction is not None:
            _append_unique_warning(warnings, _WARNING_QUALITY_DIRECTION)
        if request.quality_direction is QualityOptimizationDirection.TARGET:
            _append_unique_warning(warnings, _WARNING_QUALITY_TARGET)

        baseline_quality, baseline_anomaly = self._baseline_outputs(
            pairs=evaluable,
            objective=objective,
        )

        working = self._build_working_ranks(
            evaluable=evaluable,
            request=request,
            policy=policy,
            baseline_quality=baseline_quality,
            baseline_anomaly=baseline_anomaly,
            warnings=warnings,
        )

        evaluated_count = len(working)
        ineligible_count = sum(
            1
            for item in working
            if (not item.is_baseline) and (not item.selection_eligible)
        )

        sorted_working = self._sort_working(working, policy=policy)
        retained, _ = self._apply_cap_and_inclusion(
            sorted_working,
            policy=policy,
        )
        retained_sorted = self._sort_working(retained, policy=policy)
        ranked = [
            self._to_ranked(item, rank=index)
            for index, item in enumerate(retained_sorted, start=1)
        ]
        truncated_count = evaluated_count - len(ranked)
        if truncated_count > 0:
            _append_unique_warning(warnings, _WARNING_TRUNCATION)

        returned_eligible_count = sum(
            1 for item in ranked if item.selection_eligible
        )

        if returned_eligible_count == 0:
            _append_unique_warning(warnings, _WARNING_NO_IMPROVEMENT)
            status = ScenarioRankingStatus.NO_IMPROVEMENT
        elif used_partial:
            status = ScenarioRankingStatus.PARTIAL
        else:
            status = ScenarioRankingStatus.RANKED

        self._append_standard_caveats(warnings)

        baseline_scenario_id = None
        for item in working:
            if item.is_baseline:
                baseline_scenario_id = item.scenario.scenario_id
                break

        best_scenario_id = ranked[0].scenario_id if ranked else None
        best_nonbaseline = next(
            (
                item.scenario_id
                for item in ranked
                if item.selection_eligible
                and item.scenario_type is not CandidateScenarioType.BASELINE
            ),
            None,
        )
        baseline_is_top = bool(
            ranked and ranked[0].scenario_type is CandidateScenarioType.BASELINE
        )

        report = ScenarioRankingReport(
            status=status,
            objective=objective,
            quality_direction=request.quality_direction,
            quality_target=request.quality_target,
            baseline_scenario_id=baseline_scenario_id,
            baseline_quality_prediction=baseline_quality,
            baseline_anomaly_score=baseline_anomaly,
            ranked_scenarios=ranked,
            best_scenario_id=best_scenario_id,
            best_nonbaseline_scenario_id=best_nonbaseline,
            baseline_is_top_ranked=baseline_is_top,
            requested_scenario_count=requested_count,
            evaluated_scenario_count=evaluated_count,
            returned_scenario_count=len(ranked),
            eligible_scenario_count=returned_eligible_count,
            ineligible_scenario_count=ineligible_count,
            truncated_scenario_count=truncated_count,
            evaluated_at=datetime.now(tz=UTC),
            warnings=warnings,
            metadata=self._build_metadata(
                request=request,
                used_partial=used_partial,
                evaluated_count=evaluated_count,
                returned_count=len(ranked),
                eligible_count=returned_eligible_count,
                ineligible_count=ineligible_count,
                truncated_count=truncated_count,
            ),
        )
        return ScenarioRankingOutcome(report=report)

    def _validate_objective_weights(
        self,
        *,
        objective: RecommendationObjective,
        policy: ScenarioRankingPolicy,
    ) -> None:
        if (
            objective is RecommendationObjective.IMPROVE_PREDICTED_QUALITY
            and policy.quality_weight <= 0.0
        ):
            raise DataValidationError(
                "IMPROVE_PREDICTED_QUALITY requires quality_weight > 0"
            )
        if (
            objective is RecommendationObjective.REDUCE_ANOMALY_SCORE
            and policy.anomaly_weight <= 0.0
        ):
            raise DataValidationError(
                "REDUCE_ANOMALY_SCORE requires anomaly_weight > 0"
            )
        if objective is RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY:
            if policy.quality_weight <= 0.0 or policy.anomaly_weight <= 0.0:
                raise DataValidationError(
                    "BALANCE_QUALITY_AND_ANOMALY requires quality_weight > 0 "
                    "and anomaly_weight > 0"
                )

    def _aligned_pairs(self, request: ScenarioRankingRequest) -> list[_ScoredPair]:
        scenarios = request.grid.scenarios
        scores = request.scoring.scores
        if len(scenarios) != len(scores):
            raise DataValidationError(
                "grid scenarios and scoring scores must have equal length for ranking"
            )
        return [
            _ScoredPair(scenario=scenario, score=score)
            for scenario, score in zip(scenarios, scores, strict=True)
        ]

    def _validate_required_baseline(
        self,
        *,
        pairs: list[_ScoredPair],
        request: ScenarioRankingRequest,
    ) -> None:
        if not pairs:
            raise DataValidationError(
                "require_baseline_scenario=True requires at least one scenario"
            )
        first = pairs[0]
        if first.scenario.scenario_type is not CandidateScenarioType.BASELINE:
            raise DataValidationError(
                "require_baseline_scenario=True requires the first grid scenario "
                "to be BASELINE"
            )
        if first.score.scenario_type is not CandidateScenarioType.BASELINE:
            raise DataValidationError(
                "require_baseline_scenario=True requires the first score "
                "to be BASELINE"
            )
        baseline_ids = {
            request.grid.baseline_scenario_id,
            request.scoring.baseline_scenario_id,
            first.scenario.scenario_id,
            first.score.scenario_id,
        }
        baseline_ids.discard(None)
        if len(baseline_ids) != 1:
            raise DataValidationError(
                "baseline scenario IDs must agree between grid and scoring"
            )

    def _baseline_outputs(
        self,
        *,
        pairs: list[_ScoredPair],
        objective: RecommendationObjective,
    ) -> tuple[float | None, float | None]:
        del objective  # outputs are taken from scored baseline as-is
        baseline = next(
            (
                pair
                for pair in pairs
                if pair.scenario.scenario_type is CandidateScenarioType.BASELINE
            ),
            None,
        )
        if baseline is None:
            return None, None
        return baseline.score.quality_prediction, baseline.score.anomaly_score

    def _quality_improvement(
        self,
        *,
        scenario_quality: float,
        baseline_quality: float,
        direction: QualityOptimizationDirection,
        quality_target: float | None,
    ) -> float:
        if direction is QualityOptimizationDirection.MAXIMIZE:
            return scenario_quality - baseline_quality
        if direction is QualityOptimizationDirection.MINIMIZE:
            return baseline_quality - scenario_quality
        if direction is QualityOptimizationDirection.TARGET:
            if quality_target is None:
                raise DataValidationError(
                    "TARGET quality direction requires quality_target"
                )
            return abs(baseline_quality - quality_target) - abs(
                scenario_quality - quality_target
            )
        raise DataValidationError(f"unsupported quality direction: {direction!r}")

    def _build_working_ranks(
        self,
        *,
        evaluable: list[_ScoredPair],
        request: ScenarioRankingRequest,
        policy: ScenarioRankingPolicy,
        baseline_quality: float | None,
        baseline_anomaly: float | None,
        warnings: list[str],
    ) -> list[_WorkingRank]:
        objective = request.grid.objective
        needs_quality = _requires_quality(objective)
        needs_anomaly = _requires_anomaly(objective)

        raw_quality: list[float | None] = []
        raw_anomaly: list[float | None] = []
        for pair in evaluable:
            is_baseline = (
                pair.scenario.scenario_type is CandidateScenarioType.BASELINE
            )
            if is_baseline:
                raw_quality.append(0.0 if needs_quality else None)
                raw_anomaly.append(0.0 if needs_anomaly else None)
                continue

            q_imp: float | None = None
            a_imp: float | None = None
            if needs_quality:
                if (
                    pair.score.quality_prediction is None
                    or baseline_quality is None
                    or request.quality_direction is None
                ):
                    raise DataValidationError(
                        "quality improvement requires scored quality outputs "
                        "and quality_direction"
                    )
                q_imp = self._quality_improvement(
                    scenario_quality=pair.score.quality_prediction,
                    baseline_quality=baseline_quality,
                    direction=request.quality_direction,
                    quality_target=request.quality_target,
                )
            if needs_anomaly:
                if pair.score.anomaly_score is None or baseline_anomaly is None:
                    raise DataValidationError(
                        "anomaly improvement requires scored anomaly outputs"
                    )
                a_imp = baseline_anomaly - pair.score.anomaly_score
            raw_quality.append(q_imp)
            raw_anomaly.append(a_imp)

        max_pos_quality = 0.0
        max_pos_anomaly = 0.0
        for value in raw_quality:
            if value is not None and value > max_pos_quality:
                max_pos_quality = value
        for value in raw_anomaly:
            if value is not None and value > max_pos_anomaly:
                max_pos_anomaly = value

        max_change_magnitude = 0.0
        for pair in evaluable:
            if pair.scenario.scenario_type is CandidateScenarioType.BASELINE:
                continue
            magnitude = pair.scenario.normalized_change_magnitude
            if magnitude > max_change_magnitude:
                max_change_magnitude = magnitude

        if policy.change_penalty_weight > 0.0 and max_change_magnitude > 0.0:
            _append_unique_warning(warnings, _WARNING_CHANGE_PENALTY)

        if needs_quality and needs_anomaly:
            weight_sum = policy.quality_weight + policy.anomaly_weight
            norm_quality_weight = policy.quality_weight / weight_sum
            norm_anomaly_weight = policy.anomaly_weight / weight_sum
        else:
            norm_quality_weight = 1.0
            norm_anomaly_weight = 1.0

        saw_worsening = False
        saw_minimum_miss = False
        working: list[_WorkingRank] = []
        for pair, q_imp, a_imp in zip(
            evaluable,
            raw_quality,
            raw_anomaly,
            strict=True,
        ):
            is_baseline = (
                pair.scenario.scenario_type is CandidateScenarioType.BASELINE
            )
            scenario_warnings: list[str] = []

            if is_baseline:
                _append_unique_warning(scenario_warnings, _SCENARIO_WARNING_BASELINE)
                _append_unique_warning(scenario_warnings, _SCENARIO_WARNING_INELIGIBLE)
                baseline_quality_met = (
                    0.0 >= policy.minimum_quality_improvement
                    if needs_quality
                    else None
                )
                baseline_anomaly_met = (
                    0.0 >= policy.minimum_anomaly_improvement
                    if needs_anomaly
                    else None
                )
                baseline_flags = [
                    flag
                    for flag in (baseline_quality_met, baseline_anomaly_met)
                    if flag is not None
                ]
                working.append(
                    _WorkingRank(
                        scenario=pair.scenario,
                        score=pair.score,
                        is_baseline=True,
                        quality_improvement=0.0 if needs_quality else None,
                        anomaly_improvement=0.0 if needs_anomaly else None,
                        normalized_quality_benefit=0.0 if needs_quality else None,
                        normalized_anomaly_benefit=0.0 if needs_anomaly else None,
                        objective_benefit_score=0.0,
                        normalized_change_cost=0.0,
                        change_penalty=0.0,
                        composite_score=0.0,
                        quality_requirement_met=baseline_quality_met,
                        anomaly_requirement_met=baseline_anomaly_met,
                        objective_requirements_met=bool(baseline_flags)
                        and all(baseline_flags),
                        selection_eligible=False,
                        warnings=scenario_warnings,
                    )
                )
                continue

            if needs_quality:
                assert q_imp is not None
                if max_pos_quality > 0.0:
                    n_q = max(0.0, q_imp) / max_pos_quality
                else:
                    n_q = 0.0
            else:
                n_q = None

            if needs_anomaly:
                assert a_imp is not None
                if max_pos_anomaly > 0.0:
                    n_a = max(0.0, a_imp) / max_pos_anomaly
                else:
                    n_a = 0.0
            else:
                n_a = None

            if objective is RecommendationObjective.IMPROVE_PREDICTED_QUALITY:
                assert n_q is not None
                benefit = n_q
            elif objective is RecommendationObjective.REDUCE_ANOMALY_SCORE:
                assert n_a is not None
                benefit = n_a
            else:
                assert n_q is not None and n_a is not None
                benefit = (
                    norm_quality_weight * n_q + norm_anomaly_weight * n_a
                )

            if max_change_magnitude > 0.0:
                change_cost = (
                    pair.scenario.normalized_change_magnitude / max_change_magnitude
                )
            else:
                change_cost = 0.0
            change_penalty = policy.change_penalty_weight * change_cost
            composite = benefit - change_penalty
            if not math.isfinite(composite):
                raise DataValidationError("composite_score must be finite")

            quality_met: bool | None = None
            anomaly_met: bool | None = None
            if needs_quality:
                assert q_imp is not None
                quality_met = q_imp >= policy.minimum_quality_improvement
                if not quality_met:
                    _append_unique_warning(
                        scenario_warnings,
                        _SCENARIO_WARNING_QUALITY_REQUIREMENT,
                    )
                    saw_minimum_miss = True
            if needs_anomaly:
                assert a_imp is not None
                anomaly_met = a_imp >= policy.minimum_anomaly_improvement
                if not anomaly_met:
                    _append_unique_warning(
                        scenario_warnings,
                        _SCENARIO_WARNING_ANOMALY_REQUIREMENT,
                    )
                    saw_minimum_miss = True

            component_flags = [
                flag for flag in (quality_met, anomaly_met) if flag is not None
            ]
            requirements_met = bool(component_flags) and all(component_flags)

            worsened = False
            if not policy.allow_required_component_worsening:
                if needs_quality and q_imp is not None and q_imp < 0.0:
                    worsened = True
                if needs_anomaly and a_imp is not None and a_imp < 0.0:
                    worsened = True
            if worsened:
                _append_unique_warning(
                    scenario_warnings,
                    _SCENARIO_WARNING_COMPONENT_WORSENED,
                )
                saw_worsening = True

            positive_raw = False
            if needs_quality and q_imp is not None and q_imp > 0.0:
                positive_raw = True
            if needs_anomaly and a_imp is not None and a_imp > 0.0:
                positive_raw = True

            eligible = (
                requirements_met
                and (not worsened)
                and composite >= policy.minimum_composite_advantage
                and positive_raw
            )
            if composite < policy.minimum_composite_advantage:
                _append_unique_warning(
                    scenario_warnings,
                    _SCENARIO_WARNING_COMPOSITE_THRESHOLD,
                )
            if not eligible:
                _append_unique_warning(scenario_warnings, _SCENARIO_WARNING_INELIGIBLE)

            working.append(
                _WorkingRank(
                    scenario=pair.scenario,
                    score=pair.score,
                    is_baseline=False,
                    quality_improvement=q_imp,
                    anomaly_improvement=a_imp,
                    normalized_quality_benefit=n_q,
                    normalized_anomaly_benefit=n_a,
                    objective_benefit_score=benefit,
                    normalized_change_cost=change_cost,
                    change_penalty=change_penalty,
                    composite_score=composite,
                    quality_requirement_met=quality_met,
                    anomaly_requirement_met=anomaly_met,
                    objective_requirements_met=requirements_met,
                    selection_eligible=eligible,
                    warnings=scenario_warnings,
                )
            )

        if saw_worsening:
            _append_unique_warning(warnings, _WARNING_COMPONENT_WORSENING)
        if saw_minimum_miss:
            _append_unique_warning(warnings, _WARNING_MINIMUM_IMPROVEMENT)
        return working

    def _sort_key(
        self,
        item: _WorkingRank,
        *,
        policy: ScenarioRankingPolicy,
    ) -> tuple[object, ...]:
        fewer_changes = (
            item.scenario.change_count if policy.prefer_fewer_changes else 0
        )
        smaller_changes = (
            item.scenario.normalized_change_magnitude
            if policy.prefer_smaller_changes
            else 0.0
        )
        return (
            -item.composite_score,
            -item.objective_benefit_score,
            0 if item.objective_requirements_met else 1,
            0 if item.selection_eligible else 1,
            fewer_changes,
            smaller_changes,
            item.scenario.scenario_index,
        )

    def _sort_working(
        self,
        items: list[_WorkingRank],
        *,
        policy: ScenarioRankingPolicy,
    ) -> list[_WorkingRank]:
        return sorted(items, key=lambda item: self._sort_key(item, policy=policy))

    def _apply_cap_and_inclusion(
        self,
        sorted_working: list[_WorkingRank],
        *,
        policy: ScenarioRankingPolicy,
    ) -> tuple[list[_WorkingRank], int]:
        baseline_items = [item for item in sorted_working if item.is_baseline]
        non_baseline = [item for item in sorted_working if not item.is_baseline]

        if not policy.include_ineligible_scenarios:
            non_baseline = [item for item in non_baseline if item.selection_eligible]

        maximum = policy.maximum_ranked_scenarios
        retained: list[_WorkingRank] = []
        slots_for_non_baseline = maximum
        if baseline_items:
            if maximum >= 1:
                retained.extend(baseline_items[:1])
                slots_for_non_baseline = maximum - 1
        retained.extend(non_baseline[: max(0, slots_for_non_baseline)])
        # Truncated count is computed by the caller as evaluated - returned.
        return retained, 0

    def _to_ranked(self, item: _WorkingRank, *, rank: int) -> RankedCandidateScenario:
        scenario = item.scenario
        score = item.score
        return RankedCandidateScenario(
            rank=rank,
            scenario_id=scenario.scenario_id,
            scenario_index=scenario.scenario_index,
            scenario_type=scenario.scenario_type,
            variable_values=dict(scenario.variable_values),
            changed_variables=list(scenario.changed_variables),
            deltas=dict(scenario.deltas),
            relative_deltas=dict(scenario.relative_deltas),
            change_count=scenario.change_count,
            normalized_change_magnitude=scenario.normalized_change_magnitude,
            quality_prediction=score.quality_prediction,
            anomaly_score=score.anomaly_score,
            quality_improvement_from_baseline=item.quality_improvement,
            anomaly_improvement_from_baseline=item.anomaly_improvement,
            normalized_quality_benefit=item.normalized_quality_benefit,
            normalized_anomaly_benefit=item.normalized_anomaly_benefit,
            objective_benefit_score=item.objective_benefit_score,
            normalized_change_cost=item.normalized_change_cost,
            change_penalty=item.change_penalty,
            composite_score=item.composite_score,
            quality_requirement_met=item.quality_requirement_met,
            anomaly_requirement_met=item.anomaly_requirement_met,
            objective_requirements_met=item.objective_requirements_met,
            selection_eligible=item.selection_eligible,
            warnings=list(item.warnings),
        )

    def _append_standard_caveats(self, warnings: list[str]) -> None:
        _append_unique_warning(warnings, _WARNING_RELATIVE_HEURISTIC)
        _append_unique_warning(warnings, _WARNING_VERIFICATION)
        _append_unique_warning(warnings, _WARNING_NO_GUARANTEE)

    def _refused_outcome(
        self,
        *,
        request: ScenarioRankingRequest,
        warnings: list[str],
        requested_count: int,
    ) -> ScenarioRankingOutcome:
        self._append_standard_caveats(warnings)
        report = ScenarioRankingReport(
            status=ScenarioRankingStatus.REFUSED,
            objective=request.grid.objective,
            quality_direction=request.quality_direction,
            quality_target=request.quality_target,
            baseline_scenario_id=None,
            baseline_quality_prediction=None,
            baseline_anomaly_score=None,
            ranked_scenarios=[],
            best_scenario_id=None,
            best_nonbaseline_scenario_id=None,
            baseline_is_top_ranked=False,
            requested_scenario_count=requested_count,
            evaluated_scenario_count=0,
            returned_scenario_count=0,
            eligible_scenario_count=0,
            ineligible_scenario_count=0,
            truncated_scenario_count=0,
            evaluated_at=datetime.now(tz=UTC),
            warnings=list(warnings),
            metadata=self._build_metadata(
                request=request,
                used_partial=False,
                evaluated_count=0,
                returned_count=0,
                eligible_count=0,
                ineligible_count=0,
                truncated_count=0,
                refused=True,
            ),
        )
        return ScenarioRankingOutcome(report=report)

    def _build_metadata(
        self,
        *,
        request: ScenarioRankingRequest,
        used_partial: bool,
        evaluated_count: int,
        returned_count: int,
        eligible_count: int,
        ineligible_count: int,
        truncated_count: int,
        refused: bool = False,
    ) -> dict[str, ScalarMetadataValue]:
        del used_partial  # reserved for future status-linked metadata
        return {
            "grid_status": request.grid.status.value,
            "scoring_status": request.scoring.status.value,
            "objective": request.grid.objective.value,
            "evaluated_scenario_count": evaluated_count,
            "returned_scenario_count": returned_count,
            "eligible_scenario_count": eligible_count,
            "ineligible_scenario_count": ineligible_count,
            "truncated_scenario_count": truncated_count,
            "quality_direction": (
                request.quality_direction.value
                if request.quality_direction is not None
                else None
            ),
            "quality_target_available": request.quality_target is not None,
            "quality_weight": self._policy.quality_weight,
            "anomaly_weight": self._policy.anomaly_weight,
            "change_penalty_weight": self._policy.change_penalty_weight,
            "minimum_quality_improvement": self._policy.minimum_quality_improvement,
            "minimum_anomaly_improvement": self._policy.minimum_anomaly_improvement,
            "minimum_composite_advantage": self._policy.minimum_composite_advantage,
            "baseline_preserved": not refused,
            "scenario_ranking_performed": not refused,
            "model_scoring_performed": False,
            "model_refit_performed": False,
            "recommendation_generated": False,
            "ranking_is_relative_heuristic": True,
            "association_or_prediction_not_causation": True,
        }
