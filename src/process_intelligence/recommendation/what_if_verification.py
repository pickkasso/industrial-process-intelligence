"""Local what-if verification for generated recommendations (Step 11B.13).

Scores the baseline, proposed center, and one-factor-at-a-time adjacent
constraint-grid neighbors with the same fitted models used for recommendation.
Does not refit models, re-optimize recommendations, or assert physical safety.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator

from process_intelligence.core.exceptions import (
    DataValidationError,
    ProcessIntelligenceError,
)
from process_intelligence.recommendation.candidate_grid import (
    CandidateGridReport,
    CandidateGridStatus,
    CandidateScenario,
    CandidateScenarioType,
    CandidateValuePoint,
    VariableCandidateGrid,
)
from process_intelligence.recommendation.enums import (
    RecommendationObjective,
    RecommendationStatus,
    WhatIfPerturbationDirection,
    WhatIfStabilityClassification,
    WhatIfVerificationScenarioType,
    WhatIfVerificationStatus,
)
from process_intelligence.recommendation.scenario_ranking import (
    QualityOptimizationDirection,
)
from process_intelligence.recommendation.scenario_scoring import (
    CandidateScenarioScorer,
    ScenarioScoringRequest,
    ScenarioScoringStatus,
)
from process_intelligence.recommendation.schemas import (
    RecommendationChange,
    RecommendationResult,
    ScalarMetadataValue,
)

_ABS_TOL = 1e-12
_SCENARIO_ID_RE = re.compile(r"^WIF-(\d{6})$")

_WARNING_MIXED = (
    "Recommendation is locally mixed under adjacent grid perturbations."
)
_WARNING_ISOLATED = (
    "Recommendation improvement was isolated to the proposed grid point."
)
_WARNING_NOT_APPLICABLE = (
    "What-if verification was not applicable because no recommendation "
    "was generated."
)
_WARNING_MODEL_LOCAL = (
    "What-if verification describes model-local adjacent-grid stability only "
    "and does not establish physical safety or causation."
)
_WARNING_EXTRAPOLATION_PASS_THROUGH = (
    "Per-scenario extrapolation was not independently recomputed; "
    "extrapolation flags follow the recommendation evaluation contract."
)

_RATIONALE_COMPLETED = (
    "Adjacent constraint-grid scenarios were scored with the same fitted "
    "model used for recommendation generation."
)
_RATIONALE_NOT_APPLICABLE = (
    "Local what-if verification requires a GENERATED recommendation with "
    "proposed changes."
)
_RATIONALE_UNAVAILABLE = (
    "Local what-if verification could not be completed with the available "
    "fitted model, constraint grid, and operating-point inputs."
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


def _append_unique(target: list[str], message: str) -> None:
    if message and message not in target:
        target.append(message)


def _floats_close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=_ABS_TOL)


def _scenario_id_for_index(index: int) -> str:
    return f"WIF-{index:06d}"


def _find_grid_point_index(
    points: list[CandidateValuePoint],
    *,
    value: float,
) -> int | None:
    for index, point in enumerate(points):
        if _floats_close(point.value, value):
            return index
    return None


def _adjacent_neighbors(
    grid: VariableCandidateGrid,
    *,
    proposed_value: float,
) -> tuple[float | None, float | None]:
    index = _find_grid_point_index(grid.points, value=proposed_value)
    if index is None:
        return None, None
    lower = grid.points[index - 1].value if index > 0 else None
    upper = (
        grid.points[index + 1].value if index + 1 < len(grid.points) else None
    )
    return lower, upper


def _quality_improvement(
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


def _quality_improves(
    *,
    scenario_quality: float,
    baseline_quality: float,
    direction: QualityOptimizationDirection,
    quality_target: float | None,
) -> bool:
    improvement = _quality_improvement(
        scenario_quality=scenario_quality,
        baseline_quality=baseline_quality,
        direction=direction,
        quality_target=quality_target,
    )
    return improvement > 0.0 and not _floats_close(improvement, 0.0)


def _quality_improves_or_matches(
    *,
    scenario_quality: float,
    reference_quality: float,
    direction: QualityOptimizationDirection,
    quality_target: float | None,
) -> bool:
    improvement = _quality_improvement(
        scenario_quality=scenario_quality,
        baseline_quality=reference_quality,
        direction=direction,
        quality_target=quality_target,
    )
    return improvement > 0.0 or _floats_close(improvement, 0.0)


def _anomaly_improves(*, scenario_score: float, baseline_score: float) -> bool:
    improvement = baseline_score - scenario_score
    return improvement > 0.0 and not _floats_close(improvement, 0.0)


def _anomaly_improves_or_matches(
    *,
    scenario_score: float,
    reference_score: float,
) -> bool:
    improvement = reference_score - scenario_score
    return improvement > 0.0 or _floats_close(improvement, 0.0)


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


def classify_what_if_stability(
    *,
    proposed_improves_over_baseline: bool,
    neighbor_improves: list[bool],
) -> WhatIfStabilityClassification:
    """Classify local stability from deterministic neighbor improvement flags."""
    if not proposed_improves_over_baseline:
        return WhatIfStabilityClassification.NOT_IMPROVING
    if not neighbor_improves:
        return WhatIfStabilityClassification.NO_NEIGHBORS
    improving = sum(1 for flag in neighbor_improves if flag)
    if improving == len(neighbor_improves):
        return WhatIfStabilityClassification.STABLE
    if improving == 0:
        return WhatIfStabilityClassification.ISOLATED
    return WhatIfStabilityClassification.MIXED


class WhatIfVerificationScenario(BaseModel):
    """One compact local what-if scenario scored against the fitted model."""

    scenario_id: str
    scenario_type: WhatIfVerificationScenarioType
    perturbed_variable: str | None = None
    perturbation_direction: WhatIfPerturbationDirection | None = None
    variable_values: dict[str, float]
    predicted_quality: float | None = None
    anomaly_score: float | None = None
    objective_value: float
    improves_over_baseline: bool
    improves_or_matches_proposed: bool
    extrapolated: bool | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("scenario_id", mode="before")
    @classmethod
    def _validate_scenario_id(cls, value: object) -> str:
        text = _require_non_empty_str(value, field_name="scenario_id")
        if _SCENARIO_ID_RE.fullmatch(text) is None:
            raise ValueError(
                f"scenario_id must match WIF-000000 format, got {text!r}"
            )
        return text

    @field_validator("scenario_type", mode="before")
    @classmethod
    def _validate_scenario_type(
        cls,
        value: object,
    ) -> WhatIfVerificationScenarioType:
        if isinstance(value, WhatIfVerificationScenarioType):
            return value
        if isinstance(value, str):
            try:
                return WhatIfVerificationScenarioType(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid WhatIfVerificationScenarioType: {value!r}"
                ) from exc
        raise ValueError(
            "scenario_type must be WhatIfVerificationScenarioType, "
            f"got {type(value).__name__}"
        )

    @field_validator("perturbed_variable", mode="before")
    @classmethod
    def _validate_perturbed_variable(cls, value: object) -> str | None:
        return _require_optional_non_empty_str(
            value,
            field_name="perturbed_variable",
        )

    @field_validator("perturbation_direction", mode="before")
    @classmethod
    def _validate_perturbation_direction(
        cls,
        value: object,
    ) -> WhatIfPerturbationDirection | None:
        if value is None:
            return None
        if isinstance(value, WhatIfPerturbationDirection):
            return value
        if isinstance(value, str):
            try:
                return WhatIfPerturbationDirection(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid WhatIfPerturbationDirection: {value!r}"
                ) from exc
        raise ValueError(
            "perturbation_direction must be WhatIfPerturbationDirection or None, "
            f"got {type(value).__name__}"
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

    @field_validator(
        "predicted_quality",
        "anomaly_score",
        mode="before",
    )
    @classmethod
    def _validate_optional_scores(cls, value: object) -> float | None:
        return _require_optional_finite_float(value, field_name="score field")

    @field_validator("objective_value", mode="before")
    @classmethod
    def _validate_objective_value(cls, value: object) -> float:
        return _require_finite_float(value, field_name="objective_value")

    @field_validator(
        "improves_over_baseline",
        "improves_or_matches_proposed",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="scenario bool field")

    @field_validator("extrapolated", mode="before")
    @classmethod
    def _validate_extrapolated(cls, value: object) -> bool | None:
        if value is None:
            return None
        return _require_strict_bool(value, field_name="extrapolated")

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
    def _validate_scenario_consistency(self) -> Self:
        if self.scenario_type is WhatIfVerificationScenarioType.BASELINE:
            if self.perturbed_variable is not None:
                raise ValueError("BASELINE requires perturbed_variable=None")
            if self.perturbation_direction is not None:
                raise ValueError("BASELINE requires perturbation_direction=None")
            if self.improves_over_baseline:
                raise ValueError("BASELINE cannot improve over baseline")
        elif self.scenario_type is WhatIfVerificationScenarioType.PROPOSED_CENTER:
            if self.perturbed_variable is not None:
                raise ValueError(
                    "PROPOSED_CENTER requires perturbed_variable=None"
                )
            if self.perturbation_direction is not None:
                raise ValueError(
                    "PROPOSED_CENTER requires perturbation_direction=None"
                )
            if not self.variable_values:
                raise ValueError(
                    "PROPOSED_CENTER requires non-empty variable_values"
                )
        elif self.scenario_type in {
            WhatIfVerificationScenarioType.LOWER_NEIGHBOR,
            WhatIfVerificationScenarioType.UPPER_NEIGHBOR,
        }:
            if self.perturbed_variable is None:
                raise ValueError(
                    "neighbor scenarios require perturbed_variable"
                )
            if self.perturbation_direction is None:
                raise ValueError(
                    "neighbor scenarios require perturbation_direction"
                )
            expected_direction = (
                WhatIfPerturbationDirection.LOWER
                if self.scenario_type
                is WhatIfVerificationScenarioType.LOWER_NEIGHBOR
                else WhatIfPerturbationDirection.UPPER
            )
            if self.perturbation_direction is not expected_direction:
                raise ValueError(
                    "perturbation_direction must match neighbor scenario_type"
                )
            if self.perturbed_variable not in self.variable_values:
                raise ValueError(
                    "perturbed_variable must appear in variable_values"
                )
        else:
            raise ValueError(f"unsupported scenario_type: {self.scenario_type!r}")
        return self


class RecommendationWhatIfVerificationResult(BaseModel):
    """Compact local what-if verification summary for one recommendation."""

    status: WhatIfVerificationStatus
    objective: RecommendationObjective
    baseline_objective_value: float | None = None
    proposed_objective_value: float | None = None
    scenario_count: int
    neighbor_scenario_count: int
    improving_neighbor_count: int
    non_improving_neighbor_count: int
    extrapolated_scenario_count: int
    stability_classification: WhatIfStabilityClassification
    scenarios: list[WhatIfVerificationScenario]
    warnings: list[str] = Field(default_factory=list)
    rationale: str
    evaluated_at: datetime
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)

    @field_validator("status", mode="before")
    @classmethod
    def _validate_status(cls, value: object) -> WhatIfVerificationStatus:
        if isinstance(value, WhatIfVerificationStatus):
            return value
        if isinstance(value, str):
            try:
                return WhatIfVerificationStatus(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid WhatIfVerificationStatus: {value!r}"
                ) from exc
        raise ValueError(
            f"status must be WhatIfVerificationStatus, got {type(value).__name__}"
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

    @field_validator(
        "baseline_objective_value",
        "proposed_objective_value",
        mode="before",
    )
    @classmethod
    def _validate_optional_objectives(cls, value: object) -> float | None:
        return _require_optional_finite_float(value, field_name="objective value")

    @field_validator(
        "scenario_count",
        "neighbor_scenario_count",
        "improving_neighbor_count",
        "non_improving_neighbor_count",
        "extrapolated_scenario_count",
        mode="before",
    )
    @classmethod
    def _validate_counts(cls, value: object) -> int:
        return _require_strict_int_ge(value, field_name="count field", minimum=0)

    @field_validator("stability_classification", mode="before")
    @classmethod
    def _validate_stability(
        cls,
        value: object,
    ) -> WhatIfStabilityClassification:
        if isinstance(value, WhatIfStabilityClassification):
            return value
        if isinstance(value, str):
            try:
                return WhatIfStabilityClassification(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid WhatIfStabilityClassification: {value!r}"
                ) from exc
        raise ValueError(
            "stability_classification must be WhatIfStabilityClassification, "
            f"got {type(value).__name__}"
        )

    @field_validator("scenarios", mode="before")
    @classmethod
    def _validate_scenarios_before(
        cls,
        value: object,
    ) -> list[WhatIfVerificationScenario]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                "scenarios must be a list[WhatIfVerificationScenario], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("scenarios", mode="after")
    @classmethod
    def _validate_scenarios_after(
        cls,
        value: list[WhatIfVerificationScenario],
    ) -> list[WhatIfVerificationScenario]:
        return [item.model_copy(deep=True) for item in value]

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

    @field_validator("rationale", mode="before")
    @classmethod
    def _validate_rationale(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="rationale")

    @field_validator("evaluated_at", mode="after")
    @classmethod
    def _validate_evaluated_at(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value, field_name="evaluated_at")

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> dict[str, ScalarMetadataValue]:
        if value is None:
            return {}
        return _validate_scalar_metadata(value)

    @model_validator(mode="after")
    def _validate_result_consistency(self) -> Self:
        if self.scenario_count != len(self.scenarios):
            raise ValueError("scenario_count must equal len(scenarios)")

        neighbors = [
            item
            for item in self.scenarios
            if item.scenario_type
            in {
                WhatIfVerificationScenarioType.LOWER_NEIGHBOR,
                WhatIfVerificationScenarioType.UPPER_NEIGHBOR,
            }
        ]
        if self.neighbor_scenario_count != len(neighbors):
            raise ValueError(
                "neighbor_scenario_count must equal the number of neighbor scenarios"
            )
        improving = sum(1 for item in neighbors if item.improves_over_baseline)
        non_improving = len(neighbors) - improving
        if self.improving_neighbor_count != improving:
            raise ValueError(
                "improving_neighbor_count must match neighbor improvement flags"
            )
        if self.non_improving_neighbor_count != non_improving:
            raise ValueError(
                "non_improving_neighbor_count must match neighbor improvement flags"
            )

        extrapolated = sum(
            1 for item in self.scenarios if item.extrapolated is True
        )
        if self.extrapolated_scenario_count != extrapolated:
            raise ValueError(
                "extrapolated_scenario_count must equal True extrapolated flags"
            )

        scenario_ids = [item.scenario_id for item in self.scenarios]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("scenario IDs must be unique")

        if self.status is WhatIfVerificationStatus.NOT_APPLICABLE:
            if self.scenarios:
                raise ValueError("NOT_APPLICABLE requires empty scenarios")
            if self.stability_classification is not (
                WhatIfStabilityClassification.UNAVAILABLE
            ):
                raise ValueError(
                    "NOT_APPLICABLE requires stability_classification UNAVAILABLE"
                )
            if (
                self.baseline_objective_value is not None
                or self.proposed_objective_value is not None
            ):
                raise ValueError(
                    "NOT_APPLICABLE requires baseline/proposed objective values None"
                )
        elif self.status is WhatIfVerificationStatus.UNAVAILABLE:
            if self.stability_classification is not (
                WhatIfStabilityClassification.UNAVAILABLE
            ):
                raise ValueError(
                    "UNAVAILABLE requires stability_classification UNAVAILABLE"
                )
        elif self.status is WhatIfVerificationStatus.COMPLETED:
            if not self.scenarios:
                raise ValueError("COMPLETED requires at least one scenario")
            if self.stability_classification is (
                WhatIfStabilityClassification.UNAVAILABLE
            ):
                raise ValueError(
                    "COMPLETED cannot use stability_classification UNAVAILABLE"
                )
            if self.baseline_objective_value is None:
                raise ValueError(
                    "COMPLETED requires baseline_objective_value"
                )
            if self.proposed_objective_value is None:
                raise ValueError(
                    "COMPLETED requires proposed_objective_value"
                )
            types = {item.scenario_type for item in self.scenarios}
            if WhatIfVerificationScenarioType.BASELINE not in types:
                raise ValueError("COMPLETED requires a BASELINE scenario")
            if WhatIfVerificationScenarioType.PROPOSED_CENTER not in types:
                raise ValueError("COMPLETED requires a PROPOSED_CENTER scenario")
        else:
            raise ValueError(f"unsupported verification status: {self.status!r}")

        return self


@dataclass(frozen=True, slots=True)
class RecommendationWhatIfVerificationOutcome:
    """Immutable wrapper around a what-if verification result."""

    result: RecommendationWhatIfVerificationResult


@dataclass(frozen=True, slots=True)
class _NeighborSpec:
    variable: str
    direction: WhatIfPerturbationDirection
    value: float


@dataclass(frozen=True, slots=True)
class _BuiltScenario:
    scenario_type: WhatIfVerificationScenarioType
    perturbed_variable: str | None
    perturbation_direction: WhatIfPerturbationDirection | None
    compact_values: dict[str, float]
    full_candidate_values: dict[str, float]
    # None when this built scenario is numerically identical (on every
    # candidate variable) to an earlier built scenario. ``CandidateGridReport``
    # requires exactly one BASELINE scenario at index 0 and forbids any other
    # scenario from declaring an unchanged variable as "changed", so an exact
    # duplicate cannot be represented as its own scored candidate scenario.
    # ``duplicate_of`` then points at the earlier ``built`` list index whose
    # score should be reused instead of re-scoring an identical feature row.
    candidate_scenario: CandidateScenario | None
    duplicate_of: int | None = None


class RecommendationWhatIfVerifier:
    """Verify generated recommendations with adjacent constraint-grid neighbors.

    Reuses the fitted scoring models and recommendation constraint grids. Does
    not refit, reselect, or mutate recommendation change values.
    """

    def __init__(
        self,
        *,
        scenario_scorer: CandidateScenarioScorer,
    ) -> None:
        if not isinstance(scenario_scorer, CandidateScenarioScorer):
            raise TypeError(
                "scenario_scorer must be CandidateScenarioScorer, "
                f"got {type(scenario_scorer).__name__}"
            )
        self._scenario_scorer = scenario_scorer

    def get_metadata(self) -> dict[str, ScalarMetadataValue]:
        """Return scalar verifier capability metadata."""
        return {
            "performs_model_refit": False,
            "performs_model_reselection": False,
            "performs_recommendation_reoptimization": False,
            "uses_adjacent_constraint_grid_neighbors": True,
            "uses_one_factor_at_a_time_perturbation": True,
            "uses_random_perturbation": False,
            "asserts_physical_safety": False,
            "asserts_causation": False,
            "model_local_stability_only": True,
        }

    def not_applicable(
        self,
        *,
        objective: RecommendationObjective,
        reason: str | None = None,
    ) -> RecommendationWhatIfVerificationOutcome:
        """Build a structured NOT_APPLICABLE verification result."""
        warnings = [_WARNING_NOT_APPLICABLE, _WARNING_MODEL_LOCAL]
        rationale = reason or _RATIONALE_NOT_APPLICABLE
        result = RecommendationWhatIfVerificationResult(
            status=WhatIfVerificationStatus.NOT_APPLICABLE,
            objective=objective,
            baseline_objective_value=None,
            proposed_objective_value=None,
            scenario_count=0,
            neighbor_scenario_count=0,
            improving_neighbor_count=0,
            non_improving_neighbor_count=0,
            extrapolated_scenario_count=0,
            stability_classification=WhatIfStabilityClassification.UNAVAILABLE,
            scenarios=[],
            warnings=warnings,
            rationale=rationale,
            evaluated_at=datetime.now(tz=UTC),
            metadata={
                "verification_executed": False,
                "model_refit_performed": False,
                "recommendation_mutated": False,
            },
        )
        return RecommendationWhatIfVerificationOutcome(result=result)

    def unavailable(
        self,
        *,
        objective: RecommendationObjective,
        reason: str,
        warnings: list[str] | None = None,
    ) -> RecommendationWhatIfVerificationOutcome:
        """Build a structured UNAVAILABLE verification result."""
        local_warnings = list(warnings or [])
        _append_unique(local_warnings, reason)
        _append_unique(local_warnings, _WARNING_MODEL_LOCAL)
        result = RecommendationWhatIfVerificationResult(
            status=WhatIfVerificationStatus.UNAVAILABLE,
            objective=objective,
            baseline_objective_value=None,
            proposed_objective_value=None,
            scenario_count=0,
            neighbor_scenario_count=0,
            improving_neighbor_count=0,
            non_improving_neighbor_count=0,
            extrapolated_scenario_count=0,
            stability_classification=WhatIfStabilityClassification.UNAVAILABLE,
            scenarios=[],
            warnings=local_warnings,
            rationale=_RATIONALE_UNAVAILABLE,
            evaluated_at=datetime.now(tz=UTC),
            metadata={
                "verification_executed": True,
                "model_refit_performed": False,
                "recommendation_mutated": False,
            },
        )
        return RecommendationWhatIfVerificationOutcome(result=result)

    def verify(
        self,
        *,
        recommendation: RecommendationResult,
        grid_report: CandidateGridReport,
        feature_columns: list[str],
        baseline_features: dict[str, float],
        quality_direction: QualityOptimizationDirection | None = None,
        quality_target: float | None = None,
        target_column: str | None = None,
        extrapolation_flag: bool | None = None,
    ) -> RecommendationWhatIfVerificationOutcome:
        """Score baseline/proposed/neighbor scenarios without mutating inputs.

        Args:
            recommendation: Generated recommendation to verify.
            grid_report: Candidate grid used for the same recommendation run.
            feature_columns: Model feature order.
            baseline_features: Operating-point feature map.
            quality_direction: Quality improvement direction when required.
            quality_target: Target value for TARGET quality direction.
            target_column: Optional target column name for scoring request.
            extrapolation_flag: Pass-through extrapolation signal when no
                per-scenario checker is available.

        Returns:
            Outcome wrapping ``RecommendationWhatIfVerificationResult``.
        """
        if not isinstance(recommendation, RecommendationResult):
            raise TypeError(
                "recommendation must be RecommendationResult, "
                f"got {type(recommendation).__name__}"
            )
        if not isinstance(grid_report, CandidateGridReport):
            raise TypeError(
                "grid_report must be CandidateGridReport, "
                f"got {type(grid_report).__name__}"
            )

        objective = recommendation.objective
        if recommendation.status is not RecommendationStatus.GENERATED:
            return self.not_applicable(objective=objective)

        if not recommendation.changes:
            return self.unavailable(
                objective=objective,
                reason=(
                    "Generated recommendation has no proposed changes for "
                    "what-if verification."
                ),
            )

        if not grid_report.variable_grids or not grid_report.candidate_variables:
            return self.unavailable(
                objective=objective,
                reason=(
                    "Candidate constraint grid is unavailable for local "
                    "neighbor generation."
                ),
            )

        if not feature_columns:
            return self.unavailable(
                objective=objective,
                reason="Model feature schema is unavailable for verification scoring.",
            )

        if set(baseline_features.keys()) != set(feature_columns):
            return self.unavailable(
                objective=objective,
                reason=(
                    "Baseline operating row keys do not match the model "
                    "feature schema."
                ),
            )

        for variable in grid_report.candidate_variables:
            if variable not in feature_columns:
                return self.unavailable(
                    objective=objective,
                    reason=(
                        "Recommendation scenario schema does not match the "
                        "model feature schema."
                    ),
                )

        if _requires_quality(objective) and quality_direction is None:
            return self.unavailable(
                objective=objective,
                reason=(
                    "Quality optimization direction is required for quality "
                    "objective verification."
                ),
            )

        try:
            built = self._build_scenarios(
                recommendation=recommendation,
                grid_report=grid_report,
            )
        except DataValidationError as exc:
            return self.unavailable(
                objective=objective,
                reason=str(exc),
            )

        try:
            scored = self._score_scenarios(
                built=built,
                grid_report=grid_report,
                feature_columns=list(feature_columns),
                baseline_features=dict(baseline_features),
                target_column=target_column,
            )
        except (DataValidationError, ProcessIntelligenceError, TypeError, ValueError) as exc:
            return self.unavailable(
                objective=objective,
                reason=(
                    "Verification scoring failed with the fitted recommendation "
                    f"model ({type(exc).__name__}): {exc}"
                ),
            )

        return self._assemble_completed(
            recommendation=recommendation,
            built=built,
            scored=scored,
            quality_direction=quality_direction,
            quality_target=quality_target,
            extrapolation_flag=extrapolation_flag,
        )

    def _build_scenarios(
        self,
        *,
        recommendation: RecommendationResult,
        grid_report: CandidateGridReport,
    ) -> list[_BuiltScenario]:
        grid_by_variable = {
            grid.variable: grid for grid in grid_report.variable_grids
        }
        candidate_variables = list(grid_report.candidate_variables)
        current_values = {
            name: grid_by_variable[name].current_value for name in candidate_variables
        }
        proposed_values = dict(current_values)
        change_order: list[RecommendationChange] = list(recommendation.changes)
        for change in change_order:
            if change.variable not in grid_by_variable:
                raise DataValidationError(
                    f"Proposed change variable {change.variable!r} is absent "
                    "from the recommendation constraint grid."
                )
            grid = grid_by_variable[change.variable]
            proposed_index = _find_grid_point_index(
                grid.points, value=change.proposed_value
            )
            if proposed_index is None:
                raise DataValidationError(
                    f"Proposed value for {change.variable!r} is not a grid point."
                )
            # Snap to the grid point's exact stored value (not the raw
            # ``change.proposed_value``) so downstream ``CandidateGridReport``
            # validation -- which requires exact membership in the grid's
            # point-value set -- succeeds even when the recommendation's
            # proposed value differs from the grid point by a
            # sub-tolerance floating point epsilon.
            proposed_values[change.variable] = grid.points[proposed_index].value

        neighbor_specs: list[_NeighborSpec] = []
        for change in change_order:
            grid = grid_by_variable[change.variable]
            lower, upper = _adjacent_neighbors(
                grid,
                proposed_value=change.proposed_value,
            )
            if lower is not None:
                neighbor_specs.append(
                    _NeighborSpec(
                        variable=change.variable,
                        direction=WhatIfPerturbationDirection.LOWER,
                        value=lower,
                    )
                )
            if upper is not None:
                neighbor_specs.append(
                    _NeighborSpec(
                        variable=change.variable,
                        direction=WhatIfPerturbationDirection.UPPER,
                        value=upper,
                    )
                )

        max_scenarios = 1 + 2 * len(change_order)
        # baseline + proposed + neighbors; neighbors already capped by OFAT.
        if 2 + len(neighbor_specs) > max_scenarios + 1:
            raise DataValidationError(
                "Neighbor generation exceeded 1 + 2 * change_count scenarios."
            )

        compact_keys = [change.variable for change in change_order]
        built: list[_BuiltScenario] = []
        seen_values: dict[tuple[tuple[str, float], ...], int] = {}

        def _append(
            *,
            scenario_type: WhatIfVerificationScenarioType,
            values: dict[str, float],
            perturbed_variable: str | None,
            perturbation_direction: WhatIfPerturbationDirection | None,
        ) -> None:
            key = tuple(
                (name, round(values[name], 12)) for name in candidate_variables
            )
            duplicate_of = seen_values.get(key)
            if duplicate_of is not None:
                built.append(
                    self._make_duplicate_built(
                        scenario_type=scenario_type,
                        values=values,
                        compact_keys=compact_keys,
                        perturbed_variable=perturbed_variable,
                        perturbation_direction=perturbation_direction,
                        duplicate_of=duplicate_of,
                    )
                )
                return
            seen_values[key] = len(built)
            built.append(
                self._make_built(
                    index=sum(1 for item in built if item.candidate_scenario is not None),
                    scenario_type=scenario_type,
                    values=values,
                    # "changed" is always computed relative to the true
                    # process baseline (current_values), matching the
                    # invariant enforced by ``CandidateGridReport``: a
                    # variable may only be declared "changed" when it differs
                    # from the grid's current_value, and any candidate
                    # variable that coincidentally still equals current_value
                    # (e.g. a neighbor grid point identical to the current
                    # operating value) must be reported as unchanged even
                    # though it was deliberately perturbed away from the
                    # proposed center. Exact duplicates of an earlier built
                    # scenario (every candidate variable equal to baseline)
                    # are handled separately above and never reach this
                    # branch.
                    reference_values=current_values,
                    candidate_variables=candidate_variables,
                    grid_by_variable=grid_by_variable,
                    compact_keys=compact_keys,
                    perturbed_variable=perturbed_variable,
                    perturbation_direction=perturbation_direction,
                )
            )

        _append(
            scenario_type=WhatIfVerificationScenarioType.BASELINE,
            values=current_values,
            perturbed_variable=None,
            perturbation_direction=None,
        )
        _append(
            scenario_type=WhatIfVerificationScenarioType.PROPOSED_CENTER,
            values=proposed_values,
            perturbed_variable=None,
            perturbation_direction=None,
        )
        for neighbor in neighbor_specs:
            values = dict(proposed_values)
            values[neighbor.variable] = neighbor.value
            scenario_type = (
                WhatIfVerificationScenarioType.LOWER_NEIGHBOR
                if neighbor.direction is WhatIfPerturbationDirection.LOWER
                else WhatIfVerificationScenarioType.UPPER_NEIGHBOR
            )
            _append(
                scenario_type=scenario_type,
                values=values,
                perturbed_variable=neighbor.variable,
                perturbation_direction=neighbor.direction,
            )
        return built

    @staticmethod
    def _compact_values(
        *,
        values: dict[str, float],
        compact_keys: list[str],
        perturbed_variable: str | None,
    ) -> dict[str, float]:
        compact = {name: values[name] for name in compact_keys if name in values}
        if (
            perturbed_variable is not None
            and perturbed_variable not in compact
            and perturbed_variable in values
        ):
            compact[perturbed_variable] = values[perturbed_variable]
        return compact

    def _make_duplicate_built(
        self,
        *,
        scenario_type: WhatIfVerificationScenarioType,
        values: dict[str, float],
        compact_keys: list[str],
        perturbed_variable: str | None,
        perturbation_direction: WhatIfPerturbationDirection | None,
        duplicate_of: int,
    ) -> _BuiltScenario:
        compact = self._compact_values(
            values=values,
            compact_keys=compact_keys,
            perturbed_variable=perturbed_variable,
        )
        return _BuiltScenario(
            scenario_type=scenario_type,
            perturbed_variable=perturbed_variable,
            perturbation_direction=perturbation_direction,
            compact_values=compact,
            full_candidate_values=dict(values),
            candidate_scenario=None,
            duplicate_of=duplicate_of,
        )

    def _make_built(
        self,
        *,
        index: int,
        scenario_type: WhatIfVerificationScenarioType,
        values: dict[str, float],
        reference_values: dict[str, float],
        candidate_variables: list[str],
        grid_by_variable: dict[str, VariableCandidateGrid],
        compact_keys: list[str],
        perturbed_variable: str | None,
        perturbation_direction: WhatIfPerturbationDirection | None,
    ) -> _BuiltScenario:
        changed = [
            name
            for name in candidate_variables
            if not _floats_close(values[name], reference_values[name])
        ]
        if not changed:
            candidate_type = CandidateScenarioType.BASELINE
        elif len(changed) == 1:
            candidate_type = CandidateScenarioType.SINGLE_VARIABLE
        else:
            candidate_type = CandidateScenarioType.MULTI_VARIABLE

        deltas = {
            name: values[name] - reference_values[name] for name in changed
        }
        relative_deltas: dict[str, float | None] = {}
        magnitude = 0.0
        for name in changed:
            reference = reference_values[name]
            delta = deltas[name]
            if reference == 0.0:
                relative_deltas[name] = None
            else:
                relative_deltas[name] = delta / abs(reference)
            magnitude += abs(delta) / grid_by_variable[name].effective_span

        candidate = CandidateScenario(
            scenario_id=f"SCN-{index:06d}",
            scenario_index=index,
            scenario_type=candidate_type,
            variable_values={name: values[name] for name in candidate_variables},
            changed_variables=changed,
            deltas=deltas,
            relative_deltas=relative_deltas,
            change_count=len(changed),
            normalized_change_magnitude=float(magnitude),
        )
        compact = self._compact_values(
            values=values,
            compact_keys=compact_keys,
            perturbed_variable=perturbed_variable,
        )
        return _BuiltScenario(
            scenario_type=scenario_type,
            perturbed_variable=perturbed_variable,
            perturbation_direction=perturbation_direction,
            compact_values=compact,
            full_candidate_values=dict(values),
            candidate_scenario=candidate,
        )

    def _score_scenarios(
        self,
        *,
        built: list[_BuiltScenario],
        grid_report: CandidateGridReport,
        feature_columns: list[str],
        baseline_features: dict[str, float],
        target_column: str | None,
    ) -> list[tuple[float | None, float | None]]:
        # Exact duplicates (identical values on every candidate variable as an
        # earlier built scenario) are excluded from the scored batch; their
        # scores are copied from the referenced earlier scenario afterward.
        scoreable = [item for item in built if item.candidate_scenario is not None]
        scenarios: list[CandidateScenario] = []
        for item in scoreable:
            candidate = item.candidate_scenario
            assert candidate is not None
            scenarios.append(candidate)
        change_count = sum(
            1
            for item in scenarios
            if item.scenario_type is not CandidateScenarioType.BASELINE
        )
        max_change = max((item.change_count for item in scenarios), default=0)
        effective_limit = max(
            grid_report.effective_combination_limit,
            max_change,
            1,
        )
        temporary_grid = CandidateGridReport(
            status=(
                CandidateGridStatus.READY
                if change_count >= 1
                else CandidateGridStatus.EMPTY
            ),
            objective=grid_report.objective,
            safety_status=grid_report.safety_status,
            resolution_status=grid_report.resolution_status,
            candidate_variables=list(grid_report.candidate_variables),
            variable_grids=[
                grid.model_copy(deep=True) for grid in grid_report.variable_grids
            ],
            scenarios=scenarios,
            baseline_scenario_id=scenarios[0].scenario_id if scenarios else None,
            potential_scenario_count=len(scenarios),
            generated_scenario_count=len(scenarios),
            change_scenario_count=change_count,
            truncated_scenario_count=0,
            maximum_scenarios=max(len(scenarios), 1),
            effective_combination_limit=effective_limit,
            generated_at=datetime.now(tz=UTC),
            warnings=[],
            metadata={
                "verification_temporary_grid": True,
                "full_candidate_grid_retained": False,
            },
        )

        request = ScenarioScoringRequest(
            grid=temporary_grid,
            feature_columns=feature_columns,
            baseline_features=baseline_features,
            target_column=target_column,
            metadata={"what_if_verification": True},
        )
        outcome = self._scenario_scorer.score(request)
        report = outcome.report
        if report.status is ScenarioScoringStatus.REFUSED or not report.scores:
            raise ProcessIntelligenceError(
                "Candidate scenario scoring refused verification scenarios."
            )
        if len(report.scores) != len(scoreable):
            raise ProcessIntelligenceError(
                "Verification score count does not match scenario count."
            )
        score_iterator = iter(report.scores)
        scored_by_built_index: dict[int, tuple[float | None, float | None]] = {}
        results: list[tuple[float | None, float | None]] = []
        for index, item in enumerate(built):
            if item.candidate_scenario is not None:
                score = next(score_iterator)
                pair = (score.quality_prediction, score.anomaly_score)
                scored_by_built_index[index] = pair
            else:
                assert item.duplicate_of is not None
                pair = scored_by_built_index[item.duplicate_of]
            results.append(pair)
        return results

    def _assemble_completed(
        self,
        *,
        recommendation: RecommendationResult,
        built: list[_BuiltScenario],
        scored: list[tuple[float | None, float | None]],
        quality_direction: QualityOptimizationDirection | None,
        quality_target: float | None,
        extrapolation_flag: bool | None,
    ) -> RecommendationWhatIfVerificationOutcome:
        objective = recommendation.objective
        warnings: list[str] = [_WARNING_MODEL_LOCAL]
        if extrapolation_flag is not None:
            _append_unique(warnings, _WARNING_EXTRAPOLATION_PASS_THROUGH)

        baseline_quality, baseline_anomaly = scored[0]
        proposed_quality, proposed_anomaly = scored[1]

        baseline_objective = self._objective_value(
            objective=objective,
            quality=baseline_quality,
            anomaly=baseline_anomaly,
        )
        proposed_objective = self._objective_value(
            objective=objective,
            quality=proposed_quality,
            anomaly=proposed_anomaly,
        )

        scenarios: list[WhatIfVerificationScenario] = []
        for index, (item, (quality, anomaly)) in enumerate(
            zip(built, scored, strict=True)
        ):
            objective_value = self._objective_value(
                objective=objective,
                quality=quality,
                anomaly=anomaly,
            )
            improves_baseline = self._improves_over_reference(
                objective=objective,
                quality=quality,
                anomaly=anomaly,
                reference_quality=baseline_quality,
                reference_anomaly=baseline_anomaly,
                quality_direction=quality_direction,
                quality_target=quality_target,
            )
            improves_proposed = self._improves_or_matches_reference(
                objective=objective,
                quality=quality,
                anomaly=anomaly,
                reference_quality=proposed_quality,
                reference_anomaly=proposed_anomaly,
                quality_direction=quality_direction,
                quality_target=quality_target,
            )
            if item.scenario_type is WhatIfVerificationScenarioType.BASELINE:
                improves_baseline = False
                improves_proposed = self._improves_or_matches_reference(
                    objective=objective,
                    quality=quality,
                    anomaly=anomaly,
                    reference_quality=proposed_quality,
                    reference_anomaly=proposed_anomaly,
                    quality_direction=quality_direction,
                    quality_target=quality_target,
                )
            scenarios.append(
                WhatIfVerificationScenario(
                    scenario_id=_scenario_id_for_index(index),
                    scenario_type=item.scenario_type,
                    perturbed_variable=item.perturbed_variable,
                    perturbation_direction=item.perturbation_direction,
                    variable_values=dict(item.compact_values),
                    predicted_quality=quality,
                    anomaly_score=anomaly,
                    objective_value=objective_value,
                    improves_over_baseline=improves_baseline,
                    improves_or_matches_proposed=improves_proposed,
                    extrapolated=extrapolation_flag,
                    warnings=[],
                )
            )

        neighbors = [
            item
            for item in scenarios
            if item.scenario_type
            in {
                WhatIfVerificationScenarioType.LOWER_NEIGHBOR,
                WhatIfVerificationScenarioType.UPPER_NEIGHBOR,
            }
        ]
        proposed = next(
            item
            for item in scenarios
            if item.scenario_type is WhatIfVerificationScenarioType.PROPOSED_CENTER
        )
        stability = classify_what_if_stability(
            proposed_improves_over_baseline=proposed.improves_over_baseline,
            neighbor_improves=[item.improves_over_baseline for item in neighbors],
        )
        if stability is WhatIfStabilityClassification.MIXED:
            _append_unique(warnings, _WARNING_MIXED)
        elif stability is WhatIfStabilityClassification.ISOLATED:
            _append_unique(warnings, _WARNING_ISOLATED)

        result = RecommendationWhatIfVerificationResult(
            status=WhatIfVerificationStatus.COMPLETED,
            objective=objective,
            baseline_objective_value=baseline_objective,
            proposed_objective_value=proposed_objective,
            scenario_count=len(scenarios),
            neighbor_scenario_count=len(neighbors),
            improving_neighbor_count=sum(
                1 for item in neighbors if item.improves_over_baseline
            ),
            non_improving_neighbor_count=sum(
                1 for item in neighbors if not item.improves_over_baseline
            ),
            extrapolated_scenario_count=sum(
                1 for item in scenarios if item.extrapolated is True
            ),
            stability_classification=stability,
            scenarios=scenarios,
            warnings=warnings,
            rationale=_RATIONALE_COMPLETED,
            evaluated_at=datetime.now(tz=UTC),
            metadata={
                "verification_executed": True,
                "model_refit_performed": False,
                "recommendation_mutated": False,
                "neighbor_generation": "adjacent_constraint_grid_ofat",
                "batch_scored": True,
            },
        )
        return RecommendationWhatIfVerificationOutcome(result=result)

    def _objective_value(
        self,
        *,
        objective: RecommendationObjective,
        quality: float | None,
        anomaly: float | None,
    ) -> float:
        if objective is RecommendationObjective.REDUCE_ANOMALY_SCORE:
            if anomaly is None:
                raise ProcessIntelligenceError(
                    "Anomaly objective verification requires anomaly scores."
                )
            return float(anomaly)
        if objective is RecommendationObjective.IMPROVE_PREDICTED_QUALITY:
            if quality is None:
                raise ProcessIntelligenceError(
                    "Quality objective verification requires quality predictions."
                )
            return float(quality)
        if quality is None or anomaly is None:
            raise ProcessIntelligenceError(
                "Balance objective verification requires quality and anomaly outputs."
            )
        # Compact display value prefers quality; improvement uses both components.
        return float(quality)

    def _improves_over_reference(
        self,
        *,
        objective: RecommendationObjective,
        quality: float | None,
        anomaly: float | None,
        reference_quality: float | None,
        reference_anomaly: float | None,
        quality_direction: QualityOptimizationDirection | None,
        quality_target: float | None,
    ) -> bool:
        needs_quality = _requires_quality(objective)
        needs_anomaly = _requires_anomaly(objective)
        quality_ok = True
        anomaly_ok = True
        if needs_quality:
            if (
                quality is None
                or reference_quality is None
                or quality_direction is None
            ):
                raise ProcessIntelligenceError(
                    "Quality improvement comparison requires scored quality outputs."
                )
            quality_ok = _quality_improves(
                scenario_quality=quality,
                baseline_quality=reference_quality,
                direction=quality_direction,
                quality_target=quality_target,
            )
        if needs_anomaly:
            if anomaly is None or reference_anomaly is None:
                raise ProcessIntelligenceError(
                    "Anomaly improvement comparison requires scored anomaly outputs."
                )
            anomaly_ok = _anomaly_improves(
                scenario_score=anomaly,
                baseline_score=reference_anomaly,
            )
        if needs_quality and needs_anomaly:
            return quality_ok and anomaly_ok
        if needs_quality:
            return quality_ok
        return anomaly_ok

    def _improves_or_matches_reference(
        self,
        *,
        objective: RecommendationObjective,
        quality: float | None,
        anomaly: float | None,
        reference_quality: float | None,
        reference_anomaly: float | None,
        quality_direction: QualityOptimizationDirection | None,
        quality_target: float | None,
    ) -> bool:
        needs_quality = _requires_quality(objective)
        needs_anomaly = _requires_anomaly(objective)
        quality_ok = True
        anomaly_ok = True
        if needs_quality:
            if (
                quality is None
                or reference_quality is None
                or quality_direction is None
            ):
                raise ProcessIntelligenceError(
                    "Quality comparison requires scored quality outputs."
                )
            quality_ok = _quality_improves_or_matches(
                scenario_quality=quality,
                reference_quality=reference_quality,
                direction=quality_direction,
                quality_target=quality_target,
            )
        if needs_anomaly:
            if anomaly is None or reference_anomaly is None:
                raise ProcessIntelligenceError(
                    "Anomaly comparison requires scored anomaly outputs."
                )
            anomaly_ok = _anomaly_improves_or_matches(
                scenario_score=anomaly,
                reference_score=reference_anomaly,
            )
        if needs_quality and needs_anomaly:
            return quality_ok and anomaly_ok
        if needs_quality:
            return quality_ok
        return anomaly_ok


def recommendation_warnings_for_stability(
    classification: WhatIfStabilityClassification,
) -> list[str]:
    """Return recommendation warning strings for MIXED/ISOLATED classifications."""
    if classification is WhatIfStabilityClassification.MIXED:
        return [_WARNING_MIXED]
    if classification is WhatIfStabilityClassification.ISOLATED:
        return [_WARNING_ISOLATED]
    return []
