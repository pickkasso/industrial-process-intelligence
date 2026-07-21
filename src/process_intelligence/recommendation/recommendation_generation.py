"""Convert ranked scenarios into RecommendationChange / Result (Step 9F).

Stateless result generator over validated ranking, grid, and safety inputs.
Does not score models, re-rank scenarios, or claim causal improvement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Self

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from process_intelligence.core.exceptions import DataValidationError
from process_intelligence.core.schemas import RootCauseFactor
from process_intelligence.recommendation.candidate_grid import (
    CandidateGridReport,
    CandidateScenario,
    CandidateScenarioType,
    VariableCandidateGrid,
)
from process_intelligence.recommendation.enums import (
    RecommendationSafetyStatus,
    RecommendationStatus,
)
from process_intelligence.recommendation.scenario_ranking import (
    RankedCandidateScenario,
    ScenarioRankingReport,
    ScenarioRankingStatus,
)
from process_intelligence.recommendation.schemas import (
    RecommendationChange,
    RecommendationRequest,
    RecommendationResult,
    RecommendationSafetyDecision,
    ScalarMetadataValue,
)

_ABS_TOL = 1e-12

_DISCLAIMER = (
    "This recommendation is model-based decision support. "
    "The selected scenario is not proven optimal. "
    "Diagnosis and prediction reflect association and do not establish "
    "causation. Proposed changes require domain, process safety, "
    "operational, and experimental verification. "
    "Real-process improvement is not guaranteed."
)

_WARNING_SAFETY_REFUSED = (
    "Safety decision status is REFUSED; recommendation generation was refused."
)
_WARNING_SAFETY_CAUTION = (
    "Safety decision status is CAUTION."
)
_WARNING_CAUTION_BLOCKED = (
    "Caution safety decision blocked recommendation generation by policy."
)
_WARNING_RANKING_REFUSED = (
    "Scenario ranking status is REFUSED and cannot be used for "
    "recommendation generation."
)
_WARNING_RANKING_PARTIAL = (
    "Scenario ranking status is PARTIAL."
)
_WARNING_PARTIAL_BLOCKED = (
    "Partial ranking is not allowed for recommendation generation by policy."
)
_WARNING_NO_IMPROVEMENT = (
    "No sufficiently improved non-baseline scenario is available for "
    "recommendation generation."
)
_WARNING_EXTRAPOLATION_UNEVALUATED = (
    "Extrapolation evaluation is unavailable; recommendation generation "
    "requires extrapolation evaluation by policy."
)
_WARNING_EXTRAPOLATION_DETECTED = (
    "Extrapolation was detected; recommendation generation was blocked "
    "by policy."
)
_WARNING_UNCERTAINTY_UNAVAILABLE = (
    "Uncertainty is unavailable; recommendation generation requires "
    "uncertainty by policy."
)
_WARNING_UNCERTAINTY_UNACCEPTABLE = (
    "Uncertainty is unavailable or unacceptable; recommendation generation "
    "was blocked by policy."
)
_WARNING_NO_BEST_NONBASELINE = (
    "No eligible best non-baseline scenario is available for "
    "recommendation generation."
)
_WARNING_LOW_FACTOR_CONFIDENCE = (
    "One or more changed variables have diagnosis factor confidence below "
    "the configured minimum; recommendation generation was blocked."
)
_WARNING_FACTOR_MISSING = (
    "One or more changed variables lack matching diagnosis factors; "
    "recommendation generation was blocked."
)
_WARNING_UNSAFE_CHANGED_VARIABLE = (
    "One or more changed variables are not fully safety-eligible; "
    "recommendation generation was blocked without partial application."
)
_WARNING_MODEL_RANKED = (
    "Selected scenario is model-ranked and not proven optimal."
)
_WARNING_RESIDUAL_UNEVALUATED = (
    "Residual anomaly was not evaluated because actual target values "
    "are unavailable."
)
_WARNING_NO_GUARANTEE = (
    "Real-process improvement is not guaranteed."
)
_WARNING_VERIFICATION_REQUIRED = (
    "Process, domain, safety, operational, and experimental verification "
    "is required."
)


def _require_strict_bool(value: object, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(
            f"{field_name} must be a bool (0/1 and strings rejected), "
            f"got {type(value).__name__}"
        )
    return value


def _require_confidence(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a finite float in [0.0, 1.0] "
            f"(bool not allowed), got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{field_name} must be in [0.0, 1.0], got {number}")
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


def _floats_close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=_ABS_TOL)


def _float_dicts_close(left: dict[str, float], right: dict[str, float]) -> bool:
    if set(left) != set(right):
        return False
    return all(_floats_close(left[key], right[key]) for key in left)


def _optional_float_dicts_close(
    left: dict[str, float | None],
    right: dict[str, float | None],
) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if left_value is None and right_value is None:
            continue
        if left_value is None or right_value is None:
            return False
        if not _floats_close(left_value, right_value):
            return False
    return True


def _append_unique(warnings: list[str], message: str) -> None:
    if message and message not in warnings:
        warnings.append(message)


def _build_change_rationale(
    *,
    variable: str,
    current_value: float,
    proposed_value: float,
) -> str:
    return (
        f"Model-based scenario ranking identified changing {variable} from "
        f"{current_value} to {proposed_value} as part of a candidate scenario "
        "with improved objective-relative model outputs versus the baseline. "
        "This association and prediction do not establish causation and "
        "require process, domain, safety, and operational verification."
    )


class RecommendationGenerationPolicy(BaseModel):
    """Policy controls for ranked-scenario recommendation generation.

    Tunable gates for caution/partial ranking, extrapolation, uncertainty,
    diagnosis confidence, and safety eligibility. Does not score models or
    invent proposed values.
    """

    allow_caution_safety_decision: bool = True
    allow_partial_ranking: bool = False
    generate_ready_result_on_no_improvement: bool = True
    require_extrapolation_evaluation: bool = False
    block_on_extrapolation: bool = True
    require_uncertainty_available: bool = False
    require_acceptable_uncertainty: bool = False
    minimum_factor_confidence: float = 0.0
    require_all_changed_variables_in_diagnosis: bool = True
    require_all_changed_variables_safety_eligible: bool = True
    require_values_within_grid_bounds: bool = True
    include_safety_messages: bool = True
    include_ranking_warnings: bool = True

    @field_validator(
        "allow_caution_safety_decision",
        "allow_partial_ranking",
        "generate_ready_result_on_no_improvement",
        "require_extrapolation_evaluation",
        "block_on_extrapolation",
        "require_uncertainty_available",
        "require_acceptable_uncertainty",
        "require_all_changed_variables_in_diagnosis",
        "require_all_changed_variables_safety_eligible",
        "require_values_within_grid_bounds",
        "include_safety_messages",
        "include_ranking_warnings",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="policy bool field")

    @field_validator("minimum_factor_confidence", mode="before")
    @classmethod
    def _validate_minimum_factor_confidence(cls, value: object) -> float:
        return _require_confidence(value, field_name="minimum_factor_confidence")

    @model_validator(mode="after")
    def _validate_uncertainty_relationship(self) -> Self:
        if self.require_acceptable_uncertainty and not self.require_uncertainty_available:
            raise ValueError(
                "require_acceptable_uncertainty=True requires "
                "require_uncertainty_available=True"
            )
        return self


class RecommendationGenerationRequest(BaseModel):
    """Validated inputs for converting a ranked scenario into a recommendation.

    Requires objective-aligned request, safety decision, grid, and ranking
    objects. Does not invoke model scoring or mutate caller-owned state.
    """

    request: RecommendationRequest
    safety_decision: RecommendationSafetyDecision
    grid: CandidateGridReport
    ranking: ScenarioRankingReport
    extrapolation_evaluated: bool = False
    extrapolation_flag: bool = False
    uncertainty_available: bool = False
    uncertainty_acceptable: bool | None = None
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)

    @field_validator("request", mode="before")
    @classmethod
    def _validate_request(cls, value: object) -> RecommendationRequest:
        if isinstance(value, RecommendationRequest):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return RecommendationRequest.model_validate(value)
        raise ValueError(
            f"request must be RecommendationRequest, got {type(value).__name__}"
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

    @field_validator("ranking", mode="before")
    @classmethod
    def _validate_ranking(cls, value: object) -> ScenarioRankingReport:
        if isinstance(value, ScenarioRankingReport):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return ScenarioRankingReport.model_validate(value)
        raise ValueError(
            f"ranking must be ScenarioRankingReport, got {type(value).__name__}"
        )

    @field_validator(
        "extrapolation_evaluated",
        "extrapolation_flag",
        "uncertainty_available",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="generation bool field")

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
    def _validate_generation_consistency(self) -> Self:
        objectives = {
            self.request.objective,
            self.safety_decision.objective,
            self.grid.objective,
            self.ranking.objective,
        }
        if len(objectives) != 1:
            raise ValueError(
                "request, safety_decision, grid, and ranking objectives must match "
                f"(got request={self.request.objective!r}, "
                f"safety={self.safety_decision.objective!r}, "
                f"grid={self.grid.objective!r}, "
                f"ranking={self.ranking.objective!r})"
            )

        current_keys = set(self.request.current_values)
        for name in self.safety_decision.eligible_variables:
            if name not in current_keys:
                raise ValueError(
                    "safety_decision.eligible_variables must exist in "
                    f"request.current_values; missing {name!r}"
                )

        factors_by_variable = self._diagnosis_factor_map(self.request.diagnosis.factors)
        for assessment in self.safety_decision.variable_assessments:
            factor = factors_by_variable.get(assessment.variable)
            if factor is None:
                # Allow assessments without a matching diagnosis factor so the
                # generator can apply require_all_changed_variables_in_diagnosis.
                continue
            expected_rank = next(
                index
                for index, item in enumerate(self.request.diagnosis.factors, start=1)
                if item.variable == assessment.variable
            )
            if assessment.factor_rank != expected_rank:
                raise ValueError(
                    "safety assessment factor_rank must match diagnosis factor "
                    f"order for {assessment.variable!r}"
                )
            if not _floats_close(assessment.factor_confidence, factor.confidence):
                raise ValueError(
                    "safety assessment factor_confidence must match diagnosis "
                    f"factor confidence for {assessment.variable!r}"
                )

        if self.ranking.requested_scenario_count != self.grid.generated_scenario_count:
            raise ValueError(
                "ranking.requested_scenario_count must equal "
                "grid.generated_scenario_count "
                f"(got ranking={self.ranking.requested_scenario_count}, "
                f"grid={self.grid.generated_scenario_count})"
            )

        if self.ranking.status is ScenarioRankingStatus.REFUSED:
            # REFUSED rankings omit scenarios; only count alignment is required.
            pass
        else:
            if self.ranking.baseline_scenario_id != self.grid.baseline_scenario_id:
                raise ValueError(
                    "ranking.baseline_scenario_id must equal grid.baseline_scenario_id "
                    f"(got ranking={self.ranking.baseline_scenario_id!r}, "
                    f"grid={self.grid.baseline_scenario_id!r})"
                )

            grid_by_id = {item.scenario_id: item for item in self.grid.scenarios}
            ranked_by_id = {
                item.scenario_id: item for item in self.ranking.ranked_scenarios
            }

            for ranked in self.ranking.ranked_scenarios:
                grid_scenario = grid_by_id.get(ranked.scenario_id)
                if grid_scenario is None:
                    raise ValueError(
                        f"ranked scenario {ranked.scenario_id!r} is absent from grid"
                    )
                self._assert_scenario_alignment(ranked, grid_scenario)

            if self.ranking.best_scenario_id is not None:
                if self.ranking.best_scenario_id not in ranked_by_id:
                    raise ValueError(
                        "ranking.best_scenario_id must exist in ranked_scenarios"
                    )

            if self.ranking.best_nonbaseline_scenario_id is not None:
                selected = ranked_by_id.get(self.ranking.best_nonbaseline_scenario_id)
                if selected is None:
                    raise ValueError(
                        "ranking.best_nonbaseline_scenario_id must exist in "
                        "ranked_scenarios"
                    )
                if not selected.selection_eligible:
                    raise ValueError(
                        "ranking.best_nonbaseline_scenario_id must refer to a "
                        "selection_eligible scenario"
                    )
                if selected.scenario_type is CandidateScenarioType.BASELINE:
                    raise ValueError(
                        "ranking.best_nonbaseline_scenario_id must not be BASELINE"
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

    @staticmethod
    def _diagnosis_factor_map(
        factors: list[RootCauseFactor],
    ) -> dict[str, RootCauseFactor]:
        mapping: dict[str, RootCauseFactor] = {}
        for factor in factors:
            if factor.variable in mapping:
                raise ValueError(
                    f"diagnosis factors contain duplicate variable {factor.variable!r}"
                )
            mapping[factor.variable] = factor
        return mapping

    @staticmethod
    def _assert_scenario_alignment(
        ranked: RankedCandidateScenario,
        grid_scenario: CandidateScenario,
    ) -> None:
        if ranked.scenario_index != grid_scenario.scenario_index:
            raise ValueError(
                f"scenario_index mismatch for {ranked.scenario_id!r}"
            )
        if ranked.scenario_type != grid_scenario.scenario_type:
            raise ValueError(
                f"scenario_type mismatch for {ranked.scenario_id!r}"
            )
        if ranked.change_count != grid_scenario.change_count:
            raise ValueError(
                f"change_count mismatch for {ranked.scenario_id!r}"
            )
        if not _floats_close(
            ranked.normalized_change_magnitude,
            grid_scenario.normalized_change_magnitude,
        ):
            raise ValueError(
                f"normalized_change_magnitude mismatch for {ranked.scenario_id!r}"
            )
        if not _float_dicts_close(
            ranked.variable_values,
            grid_scenario.variable_values,
        ):
            raise ValueError(
                f"variable_values mismatch for {ranked.scenario_id!r}"
            )
        if ranked.changed_variables != grid_scenario.changed_variables:
            raise ValueError(
                f"changed_variables mismatch for {ranked.scenario_id!r}"
            )
        if not _float_dicts_close(ranked.deltas, grid_scenario.deltas):
            raise ValueError(f"deltas mismatch for {ranked.scenario_id!r}")
        if not _optional_float_dicts_close(
            ranked.relative_deltas,
            grid_scenario.relative_deltas,
        ):
            raise ValueError(
                f"relative_deltas mismatch for {ranked.scenario_id!r}"
            )


@dataclass(frozen=True, slots=True)
class RecommendationGenerationOutcome:
    """Immutable wrapper around a generated recommendation result.

    Holds only ``RecommendationResult``. Does not store models, estimators,
    or DataFrames, and contains no business logic.
    """

    result: RecommendationResult


class RankedScenarioRecommendationGenerator:
    """Convert a ranked eligible scenario into RecommendationChange objects.

    Stateless converter over validated generation requests. Does not inherit
    ``BaseRecommendationEngine``, call models, or re-rank scenarios.
    """

    def __init__(
        self,
        *,
        policy: RecommendationGenerationPolicy | None = None,
    ) -> None:
        """Create a generator with an isolated policy copy.

        Args:
            policy: Optional generation policy. When ``None``, defaults are used.
                The provided policy is deep-copied so later mutations do not
                affect this generator.

        Raises:
            TypeError: If ``policy`` is not ``None`` or a
                ``RecommendationGenerationPolicy``.
        """
        if policy is None:
            resolved = RecommendationGenerationPolicy()
        elif isinstance(policy, RecommendationGenerationPolicy):
            resolved = policy.model_copy(deep=True)
        else:
            raise TypeError(
                "policy must be RecommendationGenerationPolicy or None, "
                f"got {type(policy).__name__}"
            )
        self._policy = resolved

    def get_metadata(self) -> dict[str, ScalarMetadataValue]:
        """Return scalar-only generator metadata for the current policy.

        Returns an independent dict on every call. Does not include estimators,
        models, DataFrames, or ndarray values.
        """
        policy = self._policy
        return {
            "allow_caution_safety_decision": policy.allow_caution_safety_decision,
            "allow_partial_ranking": policy.allow_partial_ranking,
            "generate_ready_result_on_no_improvement": (
                policy.generate_ready_result_on_no_improvement
            ),
            "require_extrapolation_evaluation": (
                policy.require_extrapolation_evaluation
            ),
            "block_on_extrapolation": policy.block_on_extrapolation,
            "require_uncertainty_available": policy.require_uncertainty_available,
            "require_acceptable_uncertainty": policy.require_acceptable_uncertainty,
            "minimum_factor_confidence": policy.minimum_factor_confidence,
            "require_all_changed_variables_in_diagnosis": (
                policy.require_all_changed_variables_in_diagnosis
            ),
            "require_all_changed_variables_safety_eligible": (
                policy.require_all_changed_variables_safety_eligible
            ),
            "require_values_within_grid_bounds": (
                policy.require_values_within_grid_bounds
            ),
            "include_safety_messages": policy.include_safety_messages,
            "include_ranking_warnings": policy.include_ranking_warnings,
            "performs_model_scoring": False,
            "performs_model_refit": False,
            "performs_scenario_ranking": False,
            "generates_recommendation": True,
            "evaluates_residual_anomaly": False,
            "recommendation_is_model_based": True,
            "recommendation_is_not_causal": True,
        }

    def generate(
        self,
        generation_request: RecommendationGenerationRequest,
    ) -> RecommendationGenerationOutcome:
        """Generate a recommendation result from a ranked scenario package.

        Expected policy blocks return structured ``READY_FOR_OPTIMIZATION`` or
        ``REFUSED`` results. Invalid types raise ``TypeError``. Contradictory
        valid objects raise ``DataValidationError``. Inputs are not mutated.

        Args:
            generation_request: Validated generation request package.

        Returns:
            A ``RecommendationGenerationOutcome`` wrapping ``RecommendationResult``.
        """
        if not isinstance(generation_request, RecommendationGenerationRequest):
            raise TypeError(
                "generation_request must be RecommendationGenerationRequest, "
                f"got {type(generation_request).__name__}"
            )

        policy = self._policy
        request = generation_request.request
        safety = generation_request.safety_decision
        grid = generation_request.grid
        ranking = generation_request.ranking

        warnings: list[str] = []
        self._append_safety_status_warnings(warnings, safety=safety, policy=policy)
        if policy.include_safety_messages:
            for message in safety.messages:
                _append_unique(warnings, message)

        if safety.status is RecommendationSafetyStatus.REFUSED:
            result = self._build_result(
                status=RecommendationStatus.REFUSED,
                generation_request=generation_request,
                changes=[],
                baseline_prediction=None,
                proposed_prediction=None,
                baseline_anomaly_score=None,
                proposed_anomaly_score=None,
                confidence=0.0,
                warnings=warnings,
                selected_scenario_id=None,
                selected_change_count=0,
                recommendation_generated=False,
            )
            return RecommendationGenerationOutcome(result=result)

        if (
            safety.status is RecommendationSafetyStatus.CAUTION
            and not policy.allow_caution_safety_decision
        ):
            _append_unique(warnings, _WARNING_CAUTION_BLOCKED)
            result = self._build_ready_result(
                generation_request=generation_request,
                warnings=warnings,
                baseline_prediction=None,
                baseline_anomaly_score=None,
            )
            return RecommendationGenerationOutcome(result=result)

        self._append_ranking_status_warnings(warnings, ranking=ranking, policy=policy)
        if policy.include_ranking_warnings:
            for message in ranking.warnings:
                _append_unique(warnings, message)

        if ranking.status is ScenarioRankingStatus.REFUSED:
            result = self._build_ready_result(
                generation_request=generation_request,
                warnings=warnings,
                baseline_prediction=None,
                baseline_anomaly_score=None,
            )
            return RecommendationGenerationOutcome(result=result)

        if ranking.status is ScenarioRankingStatus.NO_IMPROVEMENT:
            # Always READY_FOR_OPTIMIZATION to honor RecommendationResult contracts.
            _ = policy.generate_ready_result_on_no_improvement
            result = self._build_ready_result(
                generation_request=generation_request,
                warnings=warnings,
                baseline_prediction=ranking.baseline_quality_prediction,
                baseline_anomaly_score=ranking.baseline_anomaly_score,
            )
            return RecommendationGenerationOutcome(result=result)

        if (
            ranking.status is ScenarioRankingStatus.PARTIAL
            and not policy.allow_partial_ranking
        ):
            _append_unique(warnings, _WARNING_PARTIAL_BLOCKED)
            result = self._build_ready_result(
                generation_request=generation_request,
                warnings=warnings,
                baseline_prediction=ranking.baseline_quality_prediction,
                baseline_anomaly_score=ranking.baseline_anomaly_score,
            )
            return RecommendationGenerationOutcome(result=result)

        if (
            policy.require_extrapolation_evaluation
            and not generation_request.extrapolation_evaluated
        ):
            _append_unique(warnings, _WARNING_EXTRAPOLATION_UNEVALUATED)
            result = self._build_ready_result(
                generation_request=generation_request,
                warnings=warnings,
                baseline_prediction=ranking.baseline_quality_prediction,
                baseline_anomaly_score=ranking.baseline_anomaly_score,
            )
            return RecommendationGenerationOutcome(result=result)

        if (
            policy.block_on_extrapolation
            and generation_request.extrapolation_flag
        ):
            _append_unique(warnings, _WARNING_EXTRAPOLATION_DETECTED)
            result = self._build_ready_result(
                generation_request=generation_request,
                warnings=warnings,
                baseline_prediction=ranking.baseline_quality_prediction,
                baseline_anomaly_score=ranking.baseline_anomaly_score,
            )
            return RecommendationGenerationOutcome(result=result)

        if not generation_request.extrapolation_evaluated:
            _append_unique(warnings, _WARNING_EXTRAPOLATION_UNEVALUATED)
        elif generation_request.extrapolation_flag:
            _append_unique(warnings, _WARNING_EXTRAPOLATION_DETECTED)

        if (
            policy.require_uncertainty_available
            and not generation_request.uncertainty_available
        ):
            _append_unique(warnings, _WARNING_UNCERTAINTY_UNAVAILABLE)
            result = self._build_ready_result(
                generation_request=generation_request,
                warnings=warnings,
                baseline_prediction=ranking.baseline_quality_prediction,
                baseline_anomaly_score=ranking.baseline_anomaly_score,
            )
            return RecommendationGenerationOutcome(result=result)

        if (
            policy.require_acceptable_uncertainty
            and generation_request.uncertainty_acceptable is not True
        ):
            _append_unique(warnings, _WARNING_UNCERTAINTY_UNACCEPTABLE)
            result = self._build_ready_result(
                generation_request=generation_request,
                warnings=warnings,
                baseline_prediction=ranking.baseline_quality_prediction,
                baseline_anomaly_score=ranking.baseline_anomaly_score,
            )
            return RecommendationGenerationOutcome(result=result)

        if not generation_request.uncertainty_available:
            _append_unique(warnings, _WARNING_UNCERTAINTY_UNAVAILABLE)
        elif generation_request.uncertainty_acceptable is False:
            _append_unique(warnings, _WARNING_UNCERTAINTY_UNACCEPTABLE)

        selected_id = ranking.best_nonbaseline_scenario_id
        if selected_id is None:
            _append_unique(warnings, _WARNING_NO_BEST_NONBASELINE)
            result = self._build_ready_result(
                generation_request=generation_request,
                warnings=warnings,
                baseline_prediction=ranking.baseline_quality_prediction,
                baseline_anomaly_score=ranking.baseline_anomaly_score,
            )
            return RecommendationGenerationOutcome(result=result)

        selected = next(
            (
                item
                for item in ranking.ranked_scenarios
                if item.scenario_id == selected_id
            ),
            None,
        )
        if selected is None:
            raise DataValidationError(
                "best_nonbaseline_scenario_id is absent from ranked_scenarios"
            )
        self._assert_selected_scenario_usable(selected)

        unsafe_variables = [
            variable
            for variable in selected.changed_variables
            if not self._is_changed_variable_safety_eligible(
                variable=variable,
                request=request,
                safety=safety,
            )
        ]
        if unsafe_variables:
            _append_unique(warnings, _WARNING_UNSAFE_CHANGED_VARIABLE)
            if policy.require_all_changed_variables_safety_eligible:
                raise DataValidationError(
                    "changed variables are not fully safety-eligible: "
                    f"{unsafe_variables}"
                )
            result = self._build_ready_result(
                generation_request=generation_request,
                warnings=warnings,
                baseline_prediction=ranking.baseline_quality_prediction,
                baseline_anomaly_score=ranking.baseline_anomaly_score,
            )
            return RecommendationGenerationOutcome(result=result)

        try:
            factor_map = self._build_factor_map(request.diagnosis.factors)
        except DataValidationError:
            raise

        missing_factors = [
            variable
            for variable in selected.changed_variables
            if variable not in factor_map
        ]
        if missing_factors:
            if policy.require_all_changed_variables_in_diagnosis:
                raise DataValidationError(
                    "changed variables missing from diagnosis factors: "
                    f"{missing_factors}"
                )
            _append_unique(warnings, _WARNING_FACTOR_MISSING)
            result = self._build_ready_result(
                generation_request=generation_request,
                warnings=warnings,
                baseline_prediction=ranking.baseline_quality_prediction,
                baseline_anomaly_score=ranking.baseline_anomaly_score,
            )
            return RecommendationGenerationOutcome(result=result)

        low_confidence = [
            variable
            for variable in selected.changed_variables
            if factor_map[variable].confidence < policy.minimum_factor_confidence
        ]
        if low_confidence:
            _append_unique(warnings, _WARNING_LOW_FACTOR_CONFIDENCE)
            result = self._build_ready_result(
                generation_request=generation_request,
                warnings=warnings,
                baseline_prediction=ranking.baseline_quality_prediction,
                baseline_anomaly_score=ranking.baseline_anomaly_score,
            )
            return RecommendationGenerationOutcome(result=result)

        for variable in selected.changed_variables:
            factor = factor_map[variable]
            if factor.needs_verification:
                _append_unique(
                    warnings,
                    (
                        f"Diagnosis factor {variable} needs verification before "
                        "operational application."
                    ),
                )

        changes = self._build_changes(
            selected=selected,
            request=request,
            factor_map=factor_map,
            grid=grid,
            policy=policy,
        )

        for message in (
            _WARNING_MODEL_RANKED,
            _WARNING_RESIDUAL_UNEVALUATED,
            _WARNING_NO_GUARANTEE,
            _WARNING_VERIFICATION_REQUIRED,
        ):
            _append_unique(warnings, message)

        confidence = sum(change.confidence for change in changes) / len(changes)
        confidence = min(1.0, max(0.0, confidence))

        result = self._build_result(
            status=RecommendationStatus.GENERATED,
            generation_request=generation_request,
            changes=changes,
            baseline_prediction=ranking.baseline_quality_prediction,
            proposed_prediction=selected.quality_prediction,
            baseline_anomaly_score=ranking.baseline_anomaly_score,
            proposed_anomaly_score=selected.anomaly_score,
            confidence=confidence,
            warnings=warnings,
            selected_scenario_id=selected.scenario_id,
            selected_change_count=len(changes),
            recommendation_generated=True,
        )
        return RecommendationGenerationOutcome(result=result)

    def _append_safety_status_warnings(
        self,
        warnings: list[str],
        *,
        safety: RecommendationSafetyDecision,
        policy: RecommendationGenerationPolicy,
    ) -> None:
        if safety.status is RecommendationSafetyStatus.REFUSED:
            _append_unique(warnings, _WARNING_SAFETY_REFUSED)
        elif safety.status is RecommendationSafetyStatus.CAUTION:
            _append_unique(warnings, _WARNING_SAFETY_CAUTION)
            if not policy.allow_caution_safety_decision:
                # Blocking message is appended by the caller path.
                pass

    def _append_ranking_status_warnings(
        self,
        warnings: list[str],
        *,
        ranking: ScenarioRankingReport,
        policy: RecommendationGenerationPolicy,
    ) -> None:
        if ranking.status is ScenarioRankingStatus.REFUSED:
            _append_unique(warnings, _WARNING_RANKING_REFUSED)
        elif ranking.status is ScenarioRankingStatus.PARTIAL:
            _append_unique(warnings, _WARNING_RANKING_PARTIAL)
            if not policy.allow_partial_ranking:
                pass
        elif ranking.status is ScenarioRankingStatus.NO_IMPROVEMENT:
            _append_unique(warnings, _WARNING_NO_IMPROVEMENT)

    def _assert_selected_scenario_usable(
        self,
        selected: RankedCandidateScenario,
    ) -> None:
        if not selected.selection_eligible:
            raise DataValidationError(
                "selected best non-baseline scenario must be selection_eligible"
            )
        if not selected.objective_requirements_met:
            raise DataValidationError(
                "selected best non-baseline scenario must have "
                "objective_requirements_met=True"
            )
        if selected.scenario_type is CandidateScenarioType.BASELINE:
            raise DataValidationError(
                "selected best non-baseline scenario must not be BASELINE"
            )
        if selected.change_count < 1:
            raise DataValidationError(
                "selected scenario must have change_count >= 1"
            )
        if not selected.changed_variables:
            raise DataValidationError(
                "selected scenario must have non-empty changed_variables"
            )
        if selected.change_count != len(selected.changed_variables):
            raise DataValidationError(
                "selected scenario change_count must equal len(changed_variables)"
            )

    def _is_changed_variable_safety_eligible(
        self,
        *,
        variable: str,
        request: RecommendationRequest,
        safety: RecommendationSafetyDecision,
    ) -> bool:
        if variable not in request.current_values:
            return False
        if variable not in safety.eligible_variables:
            return False
        if variable in safety.blocked_variables:
            return False
        assessment = next(
            (
                item
                for item in safety.variable_assessments
                if item.variable == variable
            ),
            None,
        )
        if assessment is None:
            return False
        return assessment.eligible is True

    @staticmethod
    def _build_factor_map(
        factors: list[RootCauseFactor],
    ) -> dict[str, RootCauseFactor]:
        mapping: dict[str, RootCauseFactor] = {}
        for factor in factors:
            if factor.variable in mapping:
                raise DataValidationError(
                    f"diagnosis factors contain duplicate variable {factor.variable!r}"
                )
            mapping[factor.variable] = factor
        return mapping

    def _build_changes(
        self,
        *,
        selected: RankedCandidateScenario,
        request: RecommendationRequest,
        factor_map: dict[str, RootCauseFactor],
        grid: CandidateGridReport,
        policy: RecommendationGenerationPolicy,
    ) -> list[RecommendationChange]:
        if selected.change_count > request.max_simultaneous_changes:
            raise DataValidationError(
                "selected scenario change_count exceeds "
                "request.max_simultaneous_changes"
            )

        grid_by_variable = {
            item.variable: item for item in grid.variable_grids
        }
        changes: list[RecommendationChange] = []
        for variable in selected.changed_variables:
            if variable not in selected.deltas:
                raise DataValidationError(
                    f"changed variable {variable!r} missing from deltas"
                )
            if variable not in selected.relative_deltas:
                raise DataValidationError(
                    f"changed variable {variable!r} missing from relative_deltas"
                )
            if variable not in request.current_values:
                raise DataValidationError(
                    f"changed variable {variable!r} missing from current_values"
                )
            if variable not in selected.variable_values:
                raise DataValidationError(
                    f"changed variable {variable!r} missing from variable_values"
                )

            current_value = _require_finite_float(
                request.current_values[variable],
                field_name=f"current_values[{variable!r}]",
            )
            proposed_value = _require_finite_float(
                selected.variable_values[variable],
                field_name=f"proposed_value[{variable!r}]",
            )
            delta = _require_finite_float(
                selected.deltas[variable],
                field_name=f"delta[{variable!r}]",
            )
            relative_delta = selected.relative_deltas[variable]
            if relative_delta is not None:
                relative_delta = _require_finite_float(
                    relative_delta,
                    field_name=f"relative_delta[{variable!r}]",
                )

            expected_delta = proposed_value - current_value
            if not _floats_close(delta, expected_delta):
                raise DataValidationError(
                    f"delta for {variable!r} must equal proposed - current"
                )
            if _floats_close(proposed_value, current_value):
                raise DataValidationError(
                    f"proposed value for {variable!r} must differ from current"
                )

            if current_value == 0.0:
                # None is allowed; finite values are accepted as provided.
                pass
            else:
                if relative_delta is None:
                    raise DataValidationError(
                        f"relative_delta for {variable!r} must be present when "
                        "current_value != 0"
                    )
                expected_relative = delta / abs(current_value)
                if not _floats_close(relative_delta, expected_relative):
                    raise DataValidationError(
                        f"relative_delta for {variable!r} must equal "
                        "delta / abs(current_value)"
                    )

            variable_grid = grid_by_variable.get(variable)
            if variable_grid is None:
                raise DataValidationError(
                    f"changed variable {variable!r} missing from grid.variable_grids"
                )
            self._assert_value_within_grid(
                variable=variable,
                proposed_value=proposed_value,
                variable_grid=variable_grid,
                require_bounds=policy.require_values_within_grid_bounds,
            )

            factor = factor_map[variable]
            changes.append(
                RecommendationChange(
                    variable=variable,
                    current_value=current_value,
                    proposed_value=proposed_value,
                    delta=delta,
                    relative_delta=relative_delta,
                    rationale=_build_change_rationale(
                        variable=variable,
                        current_value=current_value,
                        proposed_value=proposed_value,
                    ),
                    confidence=factor.confidence,
                    requires_verification=True,
                )
            )
        return changes

    @staticmethod
    def _assert_value_within_grid(
        *,
        variable: str,
        proposed_value: float,
        variable_grid: VariableCandidateGrid,
        require_bounds: bool,
    ) -> None:
        if not require_bounds:
            return
        lower_ok = proposed_value >= variable_grid.minimum or _floats_close(
            proposed_value,
            variable_grid.minimum,
        )
        upper_ok = proposed_value <= variable_grid.maximum or _floats_close(
            proposed_value,
            variable_grid.maximum,
        )
        if not lower_ok or not upper_ok:
            raise DataValidationError(
                f"proposed value for {variable!r} is outside resolved grid bounds"
            )
        point_values = [point.value for point in variable_grid.points]
        if not any(_floats_close(proposed_value, point) for point in point_values):
            raise DataValidationError(
                f"proposed value for {variable!r} is not a grid point"
            )

    def _build_ready_result(
        self,
        *,
        generation_request: RecommendationGenerationRequest,
        warnings: list[str],
        baseline_prediction: float | None,
        baseline_anomaly_score: float | None,
    ) -> RecommendationResult:
        return self._build_result(
            status=RecommendationStatus.READY_FOR_OPTIMIZATION,
            generation_request=generation_request,
            changes=[],
            baseline_prediction=baseline_prediction,
            proposed_prediction=None,
            baseline_anomaly_score=baseline_anomaly_score,
            proposed_anomaly_score=None,
            confidence=0.0,
            warnings=warnings,
            selected_scenario_id=None,
            selected_change_count=0,
            recommendation_generated=False,
        )

    def _build_result(
        self,
        *,
        status: RecommendationStatus,
        generation_request: RecommendationGenerationRequest,
        changes: list[RecommendationChange],
        baseline_prediction: float | None,
        proposed_prediction: float | None,
        baseline_anomaly_score: float | None,
        proposed_anomaly_score: float | None,
        confidence: float,
        warnings: list[str],
        selected_scenario_id: str | None,
        selected_change_count: int,
        recommendation_generated: bool,
    ) -> RecommendationResult:
        policy = self._policy
        request = generation_request.request
        safety = generation_request.safety_decision
        ranking = generation_request.ranking

        # Preserve anomaly scores as finite floats from ranking (including
        # negatives) under the higher=more-anomalous adapter contract.
        controlled_keys = {
            "safety_status",
            "ranking_status",
            "recommendation_status",
            "selected_scenario_available",
            "selected_scenario_id",
            "selected_change_count",
            "eligible_variable_count",
            "diagnosis_factor_count",
            "minimum_factor_confidence",
            "extrapolation_evaluated",
            "extrapolation_flag",
            "uncertainty_available",
            "uncertainty_acceptable",
            "model_scoring_performed",
            "model_refit_performed",
            "scenario_ranking_performed",
            "recommendation_generated",
            "residual_scoring_performed",
            "actual_target_available",
            "confidence_is_heuristic",
            "recommendation_is_model_based",
            "recommendation_is_not_causal",
            "real_process_improvement_not_guaranteed",
        }
        metadata: dict[str, ScalarMetadataValue] = {
            "safety_status": safety.status.value,
            "ranking_status": ranking.status.value,
            "recommendation_status": status.value,
            "selected_scenario_available": selected_scenario_id is not None,
            "selected_scenario_id": selected_scenario_id,
            "selected_change_count": selected_change_count,
            "eligible_variable_count": len(safety.eligible_variables),
            "diagnosis_factor_count": len(request.diagnosis.factors),
            "minimum_factor_confidence": policy.minimum_factor_confidence,
            "extrapolation_evaluated": generation_request.extrapolation_evaluated,
            "extrapolation_flag": generation_request.extrapolation_flag,
            "uncertainty_available": generation_request.uncertainty_available,
            "uncertainty_acceptable": generation_request.uncertainty_acceptable,
            "model_scoring_performed": False,
            "model_refit_performed": False,
            "scenario_ranking_performed": False,
            "recommendation_generated": recommendation_generated,
            "residual_scoring_performed": False,
            "actual_target_available": False,
            "confidence_is_heuristic": True,
            "recommendation_is_model_based": True,
            "recommendation_is_not_causal": True,
            "real_process_improvement_not_guaranteed": True,
        }
        for key, value in generation_request.metadata.items():
            if key not in controlled_keys:
                metadata[key] = value

        try:
            return RecommendationResult(
                status=status,
                objective=request.objective,
                safety_decision=safety.model_copy(deep=True),
                changes=list(changes),
                baseline_prediction=baseline_prediction,
                proposed_prediction=proposed_prediction,
                baseline_anomaly_score=baseline_anomaly_score,
                proposed_anomaly_score=proposed_anomaly_score,
                confidence=confidence,
                extrapolation_flag=generation_request.extrapolation_flag,
                uncertainty_available=generation_request.uncertainty_available,
                disclaimer=_DISCLAIMER,
                generated_at=datetime.now(tz=UTC),
                warnings=list(warnings),
                metadata=dict(metadata),
            )
        except ValidationError as exc:
            raise DataValidationError(
                f"recommendation result validation failed: {exc}"
            ) from exc
