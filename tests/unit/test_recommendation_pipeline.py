"""Unit tests for RecommendationPipeline orchestrator (Step 9G)."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from typing import Any

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError
from sklearn.linear_model import LinearRegression

from process_intelligence.core.enums import AnalysisTask, ColumnRole
from process_intelligence.core.exceptions import (
    DataValidationError,
    ProcessIntelligenceError,
)
from process_intelligence.core.protocols import BaseAnalysisModel, BaseAnomalyModel
from process_intelligence.core.schemas import RootCauseFactor, VariableConstraint
from process_intelligence.diagnosis.enums import DiagnosisMethod, DiagnosisScope
from process_intelligence.diagnosis.schemas import DiagnosisResult
from process_intelligence.evaluation.leakage import (
    LeakageIssue,
    LeakageIssueType,
    LeakageReport,
    LeakageSeverity,
)
from process_intelligence.models import (
    create_isolation_forest_anomaly_model,
    create_linear_regression,
)
from process_intelligence.models.anomaly import IsolationForestConfig
from process_intelligence.recommendation import (
    CandidateGridGenerator,
    CandidateGridOutcome,
    CandidateGridPolicy,
    CandidateGridReport,
    CandidateGridStatus,
    CandidateScenarioRanker,
    CandidateScenarioScorer,
    CandidateVariableSelector,
    CandidateVariableSet,
    ConstraintResolutionOutcome,
    ConstraintResolutionReport,
    ConstraintResolutionStatus,
    ConstraintResolver,
    QualityOptimizationDirection,
    RankedScenarioRecommendationGenerator,
    RecommendationGenerationPolicy,
    RecommendationGenerationRequest,
    RecommendationObjective,
    RecommendationPipeline,
    RecommendationPipelineOutcome,
    RecommendationPipelinePolicy,
    RecommendationPipelineReport,
    RecommendationPipelineRequest,
    RecommendationPipelineStage,
    RecommendationReasonCode,
    RecommendationRequest,
    RecommendationResult,
    RecommendationSafetyContext,
    RecommendationSafetyDecision,
    RecommendationSafetyGate,
    RecommendationSafetyPolicy,
    RecommendationSafetyStatus,
    RecommendationStatus,
    ScenarioRankingOutcome,
    ScenarioRankingPolicy,
    ScenarioRankingReport,
    ScenarioRankingRequest,
    ScenarioRankingStatus,
    ScenarioScoringOutcome,
    ScenarioScoringReport,
    ScenarioScoringRequest,
    ScenarioScoringStatus,
    VariableEligibilityAssessment,
)
from process_intelligence.recommendation.candidate_selection import (
    CandidateSelectionOutcome,
)
from process_intelligence.recommendation.schemas import DEFAULT_RECOMMENDATION_DISCLAIMER

FEATURE_COLUMNS = ["pressure", "temperature", "humidity"]
BASELINE_FEATURES = {"pressure": 50.0, "temperature": 80.0, "humidity": 45.0}

_OMISSION_WARNING = (
    "additional pipeline warnings were omitted due to the configured limit"
)
_PRESERVE_OUTPUTS_WARNING = (
    "preserve_stage_outputs=False is recorded in metadata only; "
    "executed stage outputs are always retained in the pipeline report"
)
_CANONICAL_STAGES = list(RecommendationPipelineStage)


# --- Shared builders (patterns from test_recommendation_safety_gate / scenario_scoring) ---


def _factor(
    variable: str,
    *,
    role: ColumnRole = ColumnRole.CONTROLLABLE_PROCESS,
    controllable: bool = True,
    confidence: float = 0.8,
    needs_verification: bool = False,
) -> RootCauseFactor:
    return RootCauseFactor(
        variable=variable,
        direction="increase",
        deviation=1.0,
        role=role,
        controllable=controllable,
        evidence="Associated driver; association only.",
        confidence=confidence,
        needs_verification=needs_verification,
    )


def _constraint(
    variable: str,
    *,
    minimum: float | None = 0.0,
    maximum: float | None = 100.0,
    fixed: bool = False,
) -> VariableConstraint:
    return VariableConstraint(
        variable=variable,
        adjustable=True,
        minimum=minimum,
        maximum=maximum,
        fixed=fixed,
    )


def _diagnosis(
    factors: list[RootCauseFactor] | None = None,
    *,
    task: AnalysisTask = AnalysisTask.REGRESSION,
) -> DiagnosisResult:
    return DiagnosisResult(
        anomaly_id="a-1",
        task=task,
        method_used=[DiagnosisMethod.GROUP_COMPARISON],
        scope=DiagnosisScope.SINGLE_EVENT,
        factors=factors or [_factor("pressure"), _factor("temperature")],
        confidence=0.8,
        analyzed_row_count=2,
        reference_row_count=10,
        caveats=["Association only; causation is not established."],
        generated_at=datetime(2026, 7, 21, 9, 0, tzinfo=UTC),
    )


def _safe_leakage(*, warnings: list[LeakageIssue] | None = None) -> LeakageReport:
    issues = list(warnings or [])
    blocker_count = sum(
        1 for issue in issues if issue.severity is LeakageSeverity.BLOCKER
    )
    warning_count = sum(
        1 for issue in issues if issue.severity is LeakageSeverity.WARNING
    )
    return LeakageReport(
        is_safe=blocker_count == 0,
        issues=issues,
        blocker_count=blocker_count,
        warning_count=warning_count,
        checked_feature_columns=list(FEATURE_COLUMNS),
        checked_preprocessing_event_count=0,
    )


def _blocker_leakage() -> LeakageReport:
    return LeakageReport(
        is_safe=False,
        issues=[
            LeakageIssue(
                issue_type=LeakageIssueType.TARGET_INCLUDED_AS_FEATURE,
                severity=LeakageSeverity.BLOCKER,
                columns=["quality"],
                partitions=[],
                message="Target included as feature",
                suggested_action="Remove target from features",
            )
        ],
        blocker_count=1,
        warning_count=0,
        checked_feature_columns=["pressure"],
        checked_preprocessing_event_count=0,
    )


def _recommendation_request(**overrides: Any) -> RecommendationRequest:
    factors = overrides.pop("factors", None)
    task = overrides.pop("task", AnalysisTask.REGRESSION)
    diagnosis = overrides.pop("diagnosis", _diagnosis(factors, task=task))
    current_values = overrides.pop(
        "current_values",
        {"pressure": 50.0, "temperature": 80.0},
    )
    constraints = overrides.pop(
        "constraints",
        [_constraint("pressure"), _constraint("temperature")],
    )
    payload: dict[str, Any] = {
        "task": task,
        "diagnosis": diagnosis,
        "objective": RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        "current_values": current_values,
        "constraints": constraints,
        "user_confirmed_controllable_variables": list(
            overrides.pop(
                "user_confirmed_controllable_variables",
                list(current_values.keys()),
            )
        ),
        "user_verified_variables": list(
            overrides.pop("user_verified_variables", list(current_values.keys()))
        ),
        "max_simultaneous_changes": overrides.pop(
            "max_simultaneous_changes",
            min(3, len(current_values)),
        ),
        "metadata": overrides.pop("metadata", {}),
    }
    payload.update(overrides)
    return RecommendationRequest(**payload)


def _safety_context(**overrides: Any) -> RecommendationSafetyContext:
    payload: dict[str, Any] = {
        "leakage_report": _safe_leakage(),
        "final_evaluation_available": True,
        "model_performance_acceptable": True,
        "model_performance_reason": None,
        "extrapolation_detected": False,
        "uncertainty_available": False,
        "uncertainty_acceptable": None,
        "metadata": {},
    }
    payload.update(overrides)
    return RecommendationSafetyContext(**payload)


def _approved_safety_policy(**overrides: Any) -> RecommendationSafetyPolicy:
    payload: dict[str, Any] = {
        "block_unverified_factors": True,
        "require_user_controllability_confirmation": True,
    }
    payload.update(overrides)
    return RecommendationSafetyPolicy(**payload)


def _assessment(
    variable: str,
    *,
    factor_rank: int,
    eligible: bool = True,
    current_value: float | None = None,
) -> VariableEligibilityAssessment:
    defaults = {"pressure": 50.0, "temperature": 80.0, "humidity": 45.0}
    return VariableEligibilityAssessment(
        variable=variable,
        factor_rank=factor_rank,
        factor_confidence=0.8,
        factor_role=ColumnRole.CONTROLLABLE_PROCESS.value,
        factor_controllable=True,
        factor_needs_verification=False,
        current_value=(
            current_value if current_value is not None else defaults[variable]
        ),
        constraint_present=True,
        user_confirmed_controllable=True,
        user_verified=True,
        eligible=eligible,
        reason_codes=(
            []
            if eligible
            else [RecommendationReasonCode.NON_CONTROLLABLE_VARIABLE]
        ),
        warnings=[],
    )


def _safety_decision(**overrides: Any) -> RecommendationSafetyDecision:
    assessments = overrides.pop(
        "variable_assessments",
        [
            _assessment("pressure", factor_rank=1),
            _assessment("temperature", factor_rank=2),
        ],
    )
    eligible = [item.variable for item in assessments if item.eligible]
    blocked = [item.variable for item in assessments if not item.eligible]
    status = overrides.pop(
        "status",
        (
            RecommendationSafetyStatus.APPROVED
            if eligible and not blocked
            else (
                RecommendationSafetyStatus.CAUTION
                if eligible
                else RecommendationSafetyStatus.REFUSED
            )
        ),
    )
    global_codes = list(overrides.pop("global_reason_codes", []))
    if status is RecommendationSafetyStatus.REFUSED and not global_codes:
        global_codes = [RecommendationReasonCode.NO_ELIGIBLE_VARIABLES]
    payload: dict[str, Any] = {
        "status": status,
        "objective": RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        "eligible_variables": eligible,
        "blocked_variables": blocked,
        "variable_assessments": assessments,
        "global_reason_codes": global_codes,
        "messages": overrides.pop("messages", ["Safety evaluation completed."]),
        "disclaimer": DEFAULT_RECOMMENDATION_DISCLAIMER,
        "evaluated_at": datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
        "metadata": {},
    }
    payload.update(overrides)
    return RecommendationSafetyDecision(**payload)


def _pipeline_request(**overrides: Any) -> RecommendationPipelineRequest:
    rec_overrides = overrides.pop("rec_overrides", {})
    ctx_overrides = overrides.pop("ctx_overrides", {})
    objective = overrides.pop(
        "objective",
        rec_overrides.pop("objective", RecommendationObjective.IMPROVE_PREDICTED_QUALITY),
    )
    task = overrides.pop("task", rec_overrides.pop("task", AnalysisTask.REGRESSION))
    if objective is RecommendationObjective.REDUCE_ANOMALY_SCORE:
        task = AnalysisTask.UNSUPERVISED_ANOMALY
        quality_direction = overrides.pop("quality_direction", None)
    else:
        quality_direction = overrides.pop(
            "quality_direction",
            QualityOptimizationDirection.MAXIMIZE,
        )
    rec_overrides.setdefault("objective", objective)
    rec_overrides.setdefault("task", task)
    payload: dict[str, Any] = {
        "recommendation_request": _recommendation_request(**rec_overrides),
        "safety_context": _safety_context(**ctx_overrides),
        "industry_constraints": overrides.pop("industry_constraints", []),
        "user_overrides": overrides.pop("user_overrides", []),
        "feature_columns": overrides.pop("feature_columns", list(FEATURE_COLUMNS)),
        "baseline_features": overrides.pop("baseline_features", dict(BASELINE_FEATURES)),
        "target_column": overrides.pop(
            "target_column",
            None if objective is RecommendationObjective.REDUCE_ANOMALY_SCORE else "quality",
        ),
        "quality_direction": quality_direction,
        "quality_target": overrides.pop("quality_target", None),
        "extrapolation_evaluated": overrides.pop("extrapolation_evaluated", False),
        "extrapolation_flag": overrides.pop("extrapolation_flag", False),
        "uncertainty_available": overrides.pop("uncertainty_available", False),
        "uncertainty_acceptable": overrides.pop("uncertainty_acceptable", None),
        "metadata": overrides.pop("metadata", {}),
    }
    payload.update(overrides)
    return RecommendationPipelineRequest(**payload)


def _training_frame(*, rows: int = 40) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    return pl.DataFrame(
        {
            "pressure": rng.uniform(0.0, 100.0, rows).tolist(),
            "temperature": rng.uniform(0.0, 100.0, rows).tolist(),
            "humidity": rng.uniform(20.0, 60.0, rows).tolist(),
        }
    )


def _quality_target(frame: pl.DataFrame) -> pl.Series:
    values = (
        frame["pressure"] * 0.1
        + frame["temperature"] * 0.05
        + frame["humidity"] * 0.02
    )
    return pl.Series("quality", values.to_list())


def _fitted_quality_model() -> BaseAnalysisModel:
    frame = _training_frame()
    model = create_linear_regression(random_state=7)
    model.fit(frame.select(FEATURE_COLUMNS), _quality_target(frame))
    return model


def _fitted_anomaly_model() -> BaseAnomalyModel:
    frame = _training_frame()
    model = create_isolation_forest_anomaly_model(
        config=IsolationForestConfig(random_state=7),
    )
    model.fit(frame.select(FEATURE_COLUMNS))
    return model


class _QualityModelSpy(BaseAnalysisModel):
    def __init__(self, inner: BaseAnalysisModel) -> None:
        self._inner = inner
        self.predict_call_count = 0
        self.last_predict_frame: pl.DataFrame | None = None

    @property
    def is_fitted(self) -> bool:
        return self._inner.is_fitted

    @property
    def feature_names(self) -> tuple[str, ...] | list[str]:
        return self._inner.feature_names

    def get_metadata(self) -> object:
        return self._inner.get_metadata()

    def fit(
        self,
        X: pl.DataFrame,
        y: pl.Series | None = None,
    ) -> _QualityModelSpy:
        self._inner.fit(X, y)
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        self.predict_call_count += 1
        self.last_predict_frame = frame
        return self._inner.predict(frame)

    def evaluate(self, X: pl.DataFrame, y: pl.Series | None = None) -> object:
        return self._inner.evaluate(X, y)

    def explain(self, X: pl.DataFrame) -> object:
        return self._inner.explain(X)


class _AnomalyModelSpy(BaseAnomalyModel):
    def __init__(self, inner: BaseAnomalyModel) -> None:
        self._inner = inner
        self.score_call_count = 0
        self.last_score_frame: pl.DataFrame | None = None

    @property
    def is_fitted(self) -> bool:
        return self._inner.is_fitted

    @property
    def feature_names(self) -> tuple[str, ...] | list[str]:
        return self._inner.feature_names

    def get_metadata(self) -> object:
        return self._inner.get_metadata()

    def fit(
        self,
        X: pl.DataFrame,
        y: pl.Series | None = None,
    ) -> _AnomalyModelSpy:
        self._inner.fit(X, y)
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return self._inner.predict(frame)

    def score_samples(self, frame: pl.DataFrame) -> np.ndarray:
        self.score_call_count += 1
        self.last_score_frame = frame
        return self._inner.score_samples(frame)

    def evaluate(self, X: pl.DataFrame, y: pl.Series | None = None) -> object:
        return self._inner.evaluate(X, y)

    def explain(self, X: pl.DataFrame) -> object:
        return self._inner.explain(X)

    def classify_anomalies(self, X: pl.DataFrame) -> object:
        return self._inner.classify_anomalies(X)


class _SafetyGateSpy(RecommendationSafetyGate):
    def __init__(self, calls: list[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._calls = calls
        self.last_kwargs: dict[str, Any] = {}

    def evaluate(
        self,
        request: RecommendationRequest,
        *,
        context: RecommendationSafetyContext,
    ) -> RecommendationSafetyDecision:
        self._calls.append("SAFETY")
        self.last_kwargs = {"request": request, "context": context}
        return super().evaluate(request, context=context)


class _ConstraintResolverSpy(ConstraintResolver):
    def __init__(self, calls: list[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._calls = calls
        self.last_kwargs: dict[str, Any] = {}
        self.override_outcome: ConstraintResolutionOutcome | None = None

    def resolve(
        self,
        request: RecommendationRequest,
        *,
        safety_decision: RecommendationSafetyDecision,
        industry_constraints: list[VariableConstraint] | None = None,
        user_overrides: list[VariableConstraint] | None = None,
    ) -> ConstraintResolutionOutcome:
        self._calls.append("CONSTRAINT_RESOLUTION")
        self.last_kwargs = {
            "request": request,
            "safety_decision": safety_decision,
            "industry_constraints": industry_constraints,
            "user_overrides": user_overrides,
        }
        if self.override_outcome is not None:
            return self.override_outcome
        return super().resolve(
            request,
            safety_decision=safety_decision,
            industry_constraints=industry_constraints,
            user_overrides=user_overrides,
        )


class _CandidateSelectorSpy(CandidateVariableSelector):
    def __init__(self, calls: list[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._calls = calls
        self.last_kwargs: dict[str, Any] = {}
        self.override_outcome: CandidateVariableSet | None = None

    def select(
        self,
        request: RecommendationRequest,
        *,
        safety_decision: RecommendationSafetyDecision,
        resolution: ConstraintResolutionOutcome,
    ) -> Any:
        self._calls.append("CANDIDATE_SELECTION")
        self.last_kwargs = {
            "request": request,
            "safety_decision": safety_decision,
            "resolution": resolution,
        }
        outcome = super().select(
            request,
            safety_decision=safety_decision,
            resolution=resolution,
        )
        if self.override_outcome is not None:
            return CandidateSelectionOutcome(candidate_set=self.override_outcome)
        return outcome


class _GridGeneratorSpy(CandidateGridGenerator):
    def __init__(self, calls: list[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._calls = calls
        self.last_candidate_set: CandidateVariableSet | None = None
        self.override_outcome: CandidateGridOutcome | None = None

    def generate(self, candidate_set: CandidateVariableSet) -> CandidateGridOutcome:
        self._calls.append("GRID_GENERATION")
        self.last_candidate_set = candidate_set
        if self.override_outcome is not None:
            return self.override_outcome
        return super().generate(candidate_set)


class _ScenarioScorerSpy(CandidateScenarioScorer):
    def __init__(
        self,
        calls: list[str],
        *,
        quality_model: BaseAnalysisModel | None = None,
        anomaly_model: BaseAnomalyModel | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            quality_model=quality_model,
            anomaly_model=anomaly_model,
            **kwargs,
        )
        self._calls = calls
        self.last_request: ScenarioScoringRequest | None = None
        self.override_outcome: ScenarioScoringOutcome | None = None

    def score(self, request: ScenarioScoringRequest) -> ScenarioScoringOutcome:
        self._calls.append("SCENARIO_SCORING")
        self.last_request = request
        if self.override_outcome is not None:
            return self.override_outcome
        return super().score(request)


class _ScenarioRankerSpy(CandidateScenarioRanker):
    def __init__(self, calls: list[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._calls = calls
        self.last_request: ScenarioRankingRequest | None = None
        self.override_outcome: ScenarioRankingOutcome | None = None

    def rank(self, request: ScenarioRankingRequest) -> ScenarioRankingOutcome:
        self._calls.append("SCENARIO_RANKING")
        self.last_request = request
        if self.override_outcome is not None:
            return self.override_outcome
        return super().rank(request)


class _RecommendationGeneratorSpy(RankedScenarioRecommendationGenerator):
    def __init__(self, calls: list[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._calls = calls
        self.last_request: RecommendationGenerationRequest | None = None

    def generate(self, request: RecommendationGenerationRequest) -> Any:
        self._calls.append("RECOMMENDATION_GENERATION")
        self.last_request = request
        return super().generate(request)


def _scorer(
    *,
    objective: RecommendationObjective = RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
    quality: BaseAnalysisModel | _QualityModelSpy | None = None,
    anomaly: BaseAnomalyModel | _AnomalyModelSpy | None = None,
) -> CandidateScenarioScorer:
    if objective is RecommendationObjective.REDUCE_ANOMALY_SCORE:
        return CandidateScenarioScorer(anomaly_model=anomaly or _fitted_anomaly_model())
    if objective is RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY:
        return CandidateScenarioScorer(
            quality_model=quality or _fitted_quality_model(),
            anomaly_model=anomaly or _fitted_anomaly_model(),
        )
    return CandidateScenarioScorer(quality_model=quality or _fitted_quality_model())


def _pipeline_with_spies(
    *,
    calls: list[str] | None = None,
    scorer: CandidateScenarioScorer | None = None,
    policy: RecommendationPipelinePolicy | None = None,
    **component_kwargs: Any,
) -> tuple[RecommendationPipeline, list[str]]:
    call_log = calls if calls is not None else []
    quality_spy = component_kwargs.pop("quality_spy", None)
    anomaly_spy = component_kwargs.pop("anomaly_spy", None)
    if scorer is None:
        scorer = _ScenarioScorerSpy(
            call_log,
            quality_model=quality_spy or _fitted_quality_model(),
            anomaly_model=anomaly_spy,
        )
    elif isinstance(scorer, _ScenarioScorerSpy):
        scorer._calls = call_log
    else:
        # Wrap a plain scorer so call-order tests still see SCENARIO_SCORING.
        wrapped = _ScenarioScorerSpy(
            call_log,
            quality_model=getattr(scorer, "_quality_model", quality_spy),
            anomaly_model=getattr(scorer, "_anomaly_model", anomaly_spy),
            policy=getattr(scorer, "_policy", None),
        )
        scorer = wrapped
    pipeline = RecommendationPipeline(
        scenario_scorer=scorer,
        safety_gate=_SafetyGateSpy(
            call_log,
            policy=component_kwargs.pop(
                "safety_policy",
                _approved_safety_policy(),
            ),
        ),
        constraint_resolver=_ConstraintResolverSpy(
            call_log,
            **component_kwargs.pop("resolver_kwargs", {}),
        ),
        candidate_selector=_CandidateSelectorSpy(
            call_log,
            **component_kwargs.pop("selector_kwargs", {}),
        ),
        grid_generator=_GridGeneratorSpy(
            call_log,
            **component_kwargs.pop("grid_kwargs", {}),
        ),
        scenario_ranker=_ScenarioRankerSpy(
            call_log,
            **component_kwargs.pop("ranker_kwargs", {}),
        ),
        recommendation_generator=_RecommendationGeneratorSpy(
            call_log,
            **component_kwargs.pop("generator_kwargs", {}),
        ),
        policy=policy,
    )
    return pipeline, call_log


def _terminal_result(
    *,
    status: RecommendationStatus,
    safety: RecommendationSafetyDecision,
    terminal_stage: RecommendationPipelineStage,
) -> RecommendationResult:
    return RecommendationResult(
        status=status,
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        safety_decision=safety,
        changes=[],
        baseline_prediction=None,
        proposed_prediction=None,
        baseline_anomaly_score=None,
        proposed_anomaly_score=None,
        confidence=0.0,
        extrapolation_flag=False,
        uncertainty_available=False,
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        generated_at=datetime(2026, 7, 21, 18, 0, tzinfo=UTC),
        warnings=[f"terminated at {terminal_stage.value}"],
        metadata={
            "recommendation_generated": False,
            "pipeline_terminal_stage": terminal_stage.value,
        },
    )


def _refused_constraint_report() -> ConstraintResolutionReport:
    return ConstraintResolutionReport(
        status=ConstraintResolutionStatus.REFUSED,
        safety_status=RecommendationSafetyStatus.APPROVED,
        requested_eligible_variables=["pressure", "temperature"],
        resolved_variables=[],
        unresolved_variables=["pressure", "temperature"],
        resolved_constraints=[],
        issues=[],
        evaluated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
        warnings=["Constraint resolution refused."],
        metadata={},
    )


def _pipeline_report(**overrides: Any) -> RecommendationPipelineReport:
    safety = overrides.pop("safety_decision", _safety_decision())
    executed = overrides.pop(
        "executed_stages",
        [RecommendationPipelineStage.SAFETY],
    )
    terminal = executed[-1]
    skipped = _CANONICAL_STAGES[len(executed) :]
    status = overrides.pop("status", RecommendationStatus.READY_FOR_OPTIMIZATION)
    final_result = overrides.pop(
        "final_result",
        _terminal_result(
            status=status,
            safety=safety,
            terminal_stage=terminal,
        ),
    )
    executed_set = set(executed)
    _unset = object()
    constraint_report = overrides.pop("constraint_report", _unset)
    candidate_set = overrides.pop("candidate_set", _unset)
    grid_report = overrides.pop("grid_report", _unset)
    scoring_report = overrides.pop("scoring_report", _unset)
    ranking_report = overrides.pop("ranking_report", _unset)
    if constraint_report is _unset:
        constraint_report = (
            _refused_constraint_report()
            if RecommendationPipelineStage.CONSTRAINT_RESOLUTION in executed_set
            else None
        )
    if candidate_set is _unset:
        candidate_set = (
            _empty_candidate_set()
            if RecommendationPipelineStage.CANDIDATE_SELECTION in executed_set
            else None
        )
    if grid_report is _unset:
        grid_report = None
    if scoring_report is _unset:
        scoring_report = None
    if ranking_report is _unset:
        ranking_report = None
    payload: dict[str, Any] = {
        "status": status,
        "terminal_stage": terminal,
        "final_result": final_result,
        "safety_decision": safety,
        "constraint_report": constraint_report,
        "candidate_set": candidate_set,
        "grid_report": grid_report,
        "scoring_report": scoring_report,
        "ranking_report": ranking_report,
        "executed_stages": executed,
        "skipped_stages": skipped,
        "started_at": datetime(2026, 7, 21, 17, 0, tzinfo=UTC),
        "completed_at": datetime(2026, 7, 21, 17, 1, tzinfo=UTC),
        "total_seconds": 0.5,
        "warnings": overrides.pop("warnings", ["pipeline warning"]),
        "metadata": overrides.pop("metadata", {"pipeline_orchestration_only": True}),
    }
    payload.update(overrides)
    return RecommendationPipelineReport(**payload)


def _refused_constraint_outcome(
    request: RecommendationRequest,
) -> ConstraintResolutionOutcome:
    del request
    return ConstraintResolutionOutcome(report=_refused_constraint_report())


def _empty_candidate_set() -> CandidateVariableSet:
    return CandidateVariableSet(
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        safety_status=RecommendationSafetyStatus.APPROVED,
        resolution_status=ConstraintResolutionStatus.REFUSED,
        candidates=[],
        candidate_variables=[],
        max_simultaneous_changes=1,
        effective_change_budget=0,
        generated_at=datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
        warnings=["No candidates."],
        metadata={},
    )


def _refused_grid_outcome() -> CandidateGridOutcome:
    return CandidateGridOutcome(
        report=CandidateGridReport(
            status=CandidateGridStatus.REFUSED,
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
            safety_status=RecommendationSafetyStatus.REFUSED,
            resolution_status=ConstraintResolutionStatus.REFUSED,
            candidate_variables=[],
            variable_grids=[],
            scenarios=[],
            baseline_scenario_id=None,
            potential_scenario_count=0,
            generated_scenario_count=0,
            change_scenario_count=0,
            truncated_scenario_count=0,
            maximum_scenarios=500,
            effective_combination_limit=0,
            generated_at=datetime(2026, 7, 21, 15, 0, tzinfo=UTC),
            warnings=["Grid refused."],
            metadata={},
        )
    )


def _empty_grid_outcome() -> CandidateGridOutcome:
    return CandidateGridOutcome(
        report=CandidateGridReport(
            status=CandidateGridStatus.EMPTY,
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
            safety_status=RecommendationSafetyStatus.APPROVED,
            resolution_status=ConstraintResolutionStatus.READY,
            candidate_variables=[],
            variable_grids=[],
            scenarios=[],
            baseline_scenario_id=None,
            potential_scenario_count=0,
            generated_scenario_count=0,
            change_scenario_count=0,
            truncated_scenario_count=0,
            maximum_scenarios=500,
            effective_combination_limit=0,
            generated_at=datetime(2026, 7, 21, 15, 0, tzinfo=UTC),
            warnings=["Grid empty."],
            metadata={},
        )
    )


def _refused_scoring_outcome() -> ScenarioScoringOutcome:
    return ScenarioScoringOutcome(
        report=ScenarioScoringReport(
            status=ScenarioScoringStatus.REFUSED,
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
            task=AnalysisTask.REGRESSION,
            feature_columns=list(FEATURE_COLUMNS),
            target_column="quality",
            quality_model_name=None,
            quality_estimator_key=None,
            anomaly_model_name=None,
            anomaly_estimator_key=None,
            baseline_scenario_id=None,
            scores=[],
            requested_scenario_count=0,
            scored_scenario_count=0,
            quality_scored_count=0,
            anomaly_scored_count=0,
            prediction_seconds=0.0,
            anomaly_scoring_seconds=0.0,
            total_seconds=0.0,
            evaluated_at=datetime(2026, 7, 21, 16, 0, tzinfo=UTC),
            warnings=["Scoring refused."],
            metadata={},
        )
    )


def _refused_ranking_outcome() -> ScenarioRankingOutcome:
    return ScenarioRankingOutcome(
        report=ScenarioRankingReport(
            status=ScenarioRankingStatus.REFUSED,
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
            quality_direction=QualityOptimizationDirection.MAXIMIZE,
            quality_target=None,
            baseline_scenario_id=None,
            baseline_quality_prediction=None,
            baseline_anomaly_score=None,
            ranked_scenarios=[],
            best_scenario_id=None,
            best_nonbaseline_scenario_id=None,
            baseline_is_top_ranked=False,
            requested_scenario_count=0,
            evaluated_scenario_count=0,
            returned_scenario_count=0,
            eligible_scenario_count=0,
            ineligible_scenario_count=0,
            truncated_scenario_count=0,
            evaluated_at=datetime(2026, 7, 21, 17, 0, tzinfo=UTC),
            warnings=["Ranking refused."],
            metadata={},
        )
    )


# --- 1-8: Stage enum and policy ---


def test_pipeline_stage_enum_values() -> None:
    assert RecommendationPipelineStage.SAFETY == "SAFETY"
    assert RecommendationPipelineStage.CONSTRAINT_RESOLUTION == "CONSTRAINT_RESOLUTION"
    assert RecommendationPipelineStage.CANDIDATE_SELECTION == "CANDIDATE_SELECTION"
    assert RecommendationPipelineStage.GRID_GENERATION == "GRID_GENERATION"
    assert RecommendationPipelineStage.SCENARIO_SCORING == "SCENARIO_SCORING"
    assert RecommendationPipelineStage.SCENARIO_RANKING == "SCENARIO_RANKING"
    assert (
        RecommendationPipelineStage.RECOMMENDATION_GENERATION
        == "RECOMMENDATION_GENERATION"
    )


def test_pipeline_stage_no_extra_members() -> None:
    assert len(RecommendationPipelineStage) == 7


def test_pipeline_stage_canonical_order() -> None:
    assert _CANONICAL_STAGES == [
        RecommendationPipelineStage.SAFETY,
        RecommendationPipelineStage.CONSTRAINT_RESOLUTION,
        RecommendationPipelineStage.CANDIDATE_SELECTION,
        RecommendationPipelineStage.GRID_GENERATION,
        RecommendationPipelineStage.SCENARIO_SCORING,
        RecommendationPipelineStage.SCENARIO_RANKING,
        RecommendationPipelineStage.RECOMMENDATION_GENERATION,
    ]


def test_default_pipeline_policy() -> None:
    policy = RecommendationPipelinePolicy()
    assert policy.stop_on_safety_refusal is True
    assert policy.stop_on_constraint_refusal is True
    assert policy.stop_on_empty_candidates is True
    assert policy.stop_on_grid_refusal_or_empty is True
    assert policy.stop_on_scoring_refusal is True
    assert policy.stop_on_ranking_refusal is True
    assert policy.preserve_stage_outputs is True
    assert policy.include_stage_warnings is True
    assert policy.maximum_aggregated_warnings == 100


@pytest.mark.parametrize(
    "field",
    [
        "stop_on_safety_refusal",
        "stop_on_constraint_refusal",
        "stop_on_empty_candidates",
        "stop_on_grid_refusal_or_empty",
        "stop_on_scoring_refusal",
        "stop_on_ranking_refusal",
        "preserve_stage_outputs",
        "include_stage_warnings",
    ],
)
@pytest.mark.parametrize("bad", [0, 1, "true", None])
def test_policy_bool_strict_validation(field: str, bad: object) -> None:
    with pytest.raises(ValidationError):
        RecommendationPipelinePolicy(**{field: bad})


def test_maximum_aggregated_warnings_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        RecommendationPipelinePolicy(maximum_aggregated_warnings=0)


def test_policy_round_trip() -> None:
    policy = RecommendationPipelinePolicy(
        stop_on_safety_refusal=False,
        maximum_aggregated_warnings=50,
    )
    restored = RecommendationPipelinePolicy.model_validate(policy.model_dump())
    assert restored == policy


# --- 9-29: PipelineRequest validation ---


def test_pipeline_request_valid_happy_path() -> None:
    request = _pipeline_request()
    assert request.feature_columns == FEATURE_COLUMNS
    assert request.baseline_features == BASELINE_FEATURES
    assert request.quality_direction is QualityOptimizationDirection.MAXIMIZE


def test_pipeline_request_recommendation_request_type_error() -> None:
    with pytest.raises(ValidationError):
        RecommendationPipelineRequest(
            recommendation_request="bad",  # type: ignore[arg-type]
            safety_context=_safety_context(),
            industry_constraints=[],
            user_overrides=[],
            feature_columns=list(FEATURE_COLUMNS),
            baseline_features=dict(BASELINE_FEATURES),
            quality_direction=QualityOptimizationDirection.MAXIMIZE,
        )


def test_pipeline_request_safety_context_type_error() -> None:
    with pytest.raises(ValidationError):
        RecommendationPipelineRequest(
            recommendation_request=_recommendation_request(),
            safety_context="bad",  # type: ignore[arg-type]
            industry_constraints=[],
            user_overrides=[],
            feature_columns=list(FEATURE_COLUMNS),
            baseline_features=dict(BASELINE_FEATURES),
            quality_direction=QualityOptimizationDirection.MAXIMIZE,
        )


def test_pipeline_request_constraint_list_type_error() -> None:
    with pytest.raises(ValidationError):
        RecommendationPipelineRequest(
            recommendation_request=_recommendation_request(),
            safety_context=_safety_context(),
            industry_constraints="bad",  # type: ignore[arg-type]
            user_overrides=[],
            feature_columns=list(FEATURE_COLUMNS),
            baseline_features=dict(BASELINE_FEATURES),
            quality_direction=QualityOptimizationDirection.MAXIMIZE,
        )


def test_pipeline_request_duplicate_industry_constraints_rejected() -> None:
    with pytest.raises(ValidationError):
        _pipeline_request(
            industry_constraints=[_constraint("pressure"), _constraint("pressure")],
        )


def test_pipeline_request_empty_feature_columns_rejected() -> None:
    with pytest.raises(ValidationError):
        _pipeline_request(feature_columns=[])


def test_pipeline_request_duplicate_feature_columns_rejected() -> None:
    with pytest.raises(ValidationError):
        _pipeline_request(feature_columns=["pressure", "pressure", "humidity"])


def test_pipeline_request_reserved_feature_column_rejected() -> None:
    with pytest.raises(ValidationError):
        _pipeline_request(feature_columns=["pressure", "_original_row_id", "humidity"])


def test_pipeline_request_baseline_key_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        _pipeline_request(
            baseline_features={"pressure": 50.0, "temperature": 80.0},
        )


def test_pipeline_request_current_values_missing_in_baseline_rejected() -> None:
    with pytest.raises(ValidationError):
        _pipeline_request(
            rec_overrides={"current_values": {"pressure": 50.0, "unknown": 1.0}},
        )


def test_pipeline_request_current_values_mismatch_baseline_rejected() -> None:
    with pytest.raises(ValidationError):
        _pipeline_request(
            rec_overrides={"current_values": {"pressure": 51.0, "temperature": 80.0}},
        )


def test_pipeline_request_quality_direction_required_for_quality_objective() -> None:
    with pytest.raises(ValidationError):
        RecommendationPipelineRequest(
            recommendation_request=_recommendation_request(
                objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
            ),
            safety_context=_safety_context(),
            industry_constraints=[],
            user_overrides=[],
            feature_columns=list(FEATURE_COLUMNS),
            baseline_features=dict(BASELINE_FEATURES),
            quality_direction=None,
        )


def test_pipeline_request_quality_direction_required_for_balance_objective() -> None:
    with pytest.raises(ValidationError):
        RecommendationPipelineRequest(
            recommendation_request=_recommendation_request(
                objective=RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
            ),
            safety_context=_safety_context(),
            industry_constraints=[],
            user_overrides=[],
            feature_columns=list(FEATURE_COLUMNS),
            baseline_features=dict(BASELINE_FEATURES),
            quality_direction=None,
        )


def test_pipeline_request_quality_direction_none_ok_for_anomaly_objective() -> None:
    request = _pipeline_request(
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        quality_direction=None,
    )
    assert request.quality_direction is None


def test_pipeline_request_quality_target_required_for_target_direction() -> None:
    with pytest.raises(ValidationError):
        _pipeline_request(
            quality_direction=QualityOptimizationDirection.TARGET,
            quality_target=None,
        )


def test_pipeline_request_quality_target_forbidden_without_target_direction() -> None:
    with pytest.raises(ValidationError):
        _pipeline_request(
            quality_direction=QualityOptimizationDirection.MAXIMIZE,
            quality_target=10.0,
        )


def test_pipeline_request_extrapolation_flag_without_evaluated_rejected() -> None:
    with pytest.raises(ValidationError):
        _pipeline_request(extrapolation_evaluated=False, extrapolation_flag=True)


def test_pipeline_request_uncertainty_acceptable_without_available_rejected() -> None:
    with pytest.raises(ValidationError):
        _pipeline_request(
            uncertainty_available=False,
            uncertainty_acceptable=True,
        )


def test_pipeline_request_metadata_scalar_validation() -> None:
    with pytest.raises(ValidationError):
        _pipeline_request(metadata={"bad": {"nested": 1}})


def test_pipeline_request_target_column_in_features_rejected() -> None:
    with pytest.raises(ValidationError):
        _pipeline_request(target_column="pressure")


def test_pipeline_request_input_immutability_on_construction() -> None:
    features = list(FEATURE_COLUMNS)
    baseline = dict(BASELINE_FEATURES)
    industry = [_constraint("pressure")]
    request = _pipeline_request(
        feature_columns=features,
        baseline_features=baseline,
        industry_constraints=industry,
    )
    features.append("extra")
    baseline["pressure"] = 999.0
    industry[0].minimum = -1.0
    assert request.feature_columns == FEATURE_COLUMNS
    assert request.baseline_features["pressure"] == 50.0
    assert request.industry_constraints[0].minimum == 0.0


def test_pipeline_request_round_trip() -> None:
    request = _pipeline_request(metadata={"run_id": "abc"})
    restored = RecommendationPipelineRequest.model_validate(request.model_dump())
    assert restored == request


# --- 30-49: PipelineReport validation ---


def test_pipeline_report_valid_full_execution() -> None:
    outcome = RecommendationPipeline(scenario_scorer=_scorer()).run(_pipeline_request())
    report = outcome.report
    assert report.executed_stages == list(_CANONICAL_STAGES)
    assert report.skipped_stages == []
    assert report.constraint_report is not None
    assert report.candidate_set is not None
    assert report.grid_report is not None
    assert report.scoring_report is not None
    assert report.ranking_report is not None


def test_pipeline_report_safety_terminal() -> None:
    safety = _safety_decision(status=RecommendationSafetyStatus.REFUSED)
    report = _pipeline_report(
        safety_decision=safety,
        executed_stages=[RecommendationPipelineStage.SAFETY],
        status=RecommendationStatus.REFUSED,
        final_result=_terminal_result(
            status=RecommendationStatus.REFUSED,
            safety=safety,
            terminal_stage=RecommendationPipelineStage.SAFETY,
        ),
    )
    assert report.constraint_report is None
    assert report.candidate_set is None


def test_pipeline_report_mid_stage_terminal() -> None:
    safety = _safety_decision()
    constraint = _refused_constraint_report()
    report = _pipeline_report(
        safety_decision=safety,
        executed_stages=[
            RecommendationPipelineStage.SAFETY,
            RecommendationPipelineStage.CONSTRAINT_RESOLUTION,
        ],
        constraint_report=constraint,
        status=RecommendationStatus.READY_FOR_OPTIMIZATION,
    )
    assert report.grid_report is None
    assert report.terminal_stage is RecommendationPipelineStage.CONSTRAINT_RESOLUTION


def test_pipeline_report_executed_skipped_disjoint_and_cover_all() -> None:
    report = _pipeline_report(
        executed_stages=_CANONICAL_STAGES[:3],
    )
    assert set(report.executed_stages).isdisjoint(set(report.skipped_stages))
    assert set(report.executed_stages) | set(report.skipped_stages) == set(
        _CANONICAL_STAGES
    )


def test_pipeline_report_executed_must_be_prefix() -> None:
    with pytest.raises(ValidationError):
        _pipeline_report(
            executed_stages=[
                RecommendationPipelineStage.SAFETY,
                RecommendationPipelineStage.GRID_GENERATION,
            ],
        )


def test_pipeline_report_safety_must_always_execute() -> None:
    with pytest.raises(ValidationError):
        _pipeline_report(
            executed_stages=[RecommendationPipelineStage.CONSTRAINT_RESOLUTION],
        )


def test_pipeline_report_terminal_stage_matches_last_executed() -> None:
    with pytest.raises(ValidationError):
        _pipeline_report(
            executed_stages=[RecommendationPipelineStage.SAFETY],
            terminal_stage=RecommendationPipelineStage.CONSTRAINT_RESOLUTION,
        )


def test_pipeline_report_missing_stage_output_rejected() -> None:
    with pytest.raises(ValidationError):
        _pipeline_report(
            executed_stages=[
                RecommendationPipelineStage.SAFETY,
                RecommendationPipelineStage.CONSTRAINT_RESOLUTION,
            ],
            constraint_report=None,
        )


def test_pipeline_report_skipped_stage_output_must_be_none() -> None:
    with pytest.raises(ValidationError):
        _pipeline_report(
            executed_stages=[RecommendationPipelineStage.SAFETY],
            constraint_report=_refused_constraint_report(),
        )


def test_pipeline_report_timestamps_timezone_aware() -> None:
    with pytest.raises(ValidationError):
        _pipeline_report(started_at=datetime(2026, 7, 21, 17, 0))


def test_pipeline_report_completed_before_started_rejected() -> None:
    with pytest.raises(ValidationError):
        _pipeline_report(
            started_at=datetime(2026, 7, 21, 18, 0, tzinfo=UTC),
            completed_at=datetime(2026, 7, 21, 17, 0, tzinfo=UTC),
        )


def test_pipeline_report_total_seconds_negative_rejected() -> None:
    with pytest.raises(ValidationError):
        _pipeline_report(total_seconds=-0.1)


def test_pipeline_report_warning_duplicates_rejected() -> None:
    with pytest.raises(ValidationError):
        _pipeline_report(warnings=["a", "a"])


def test_pipeline_report_final_result_status_mismatch_rejected() -> None:
    safety = _safety_decision()
    with pytest.raises(ValidationError):
        _pipeline_report(
            safety_decision=safety,
            status=RecommendationStatus.GENERATED,
            final_result=_terminal_result(
                status=RecommendationStatus.REFUSED,
                safety=safety,
                terminal_stage=RecommendationPipelineStage.SAFETY,
            ),
        )


def test_pipeline_report_metadata_scalar_validation() -> None:
    with pytest.raises(ValidationError):
        _pipeline_report(metadata={"frame": pl.DataFrame({"x": [1]})})


def test_pipeline_report_round_trip() -> None:
    report = _pipeline_report()
    restored = RecommendationPipelineReport.model_validate(report.model_dump())
    assert restored == report


# --- 50-51: Outcome frozen/slots ---


def test_pipeline_outcome_frozen_and_slots() -> None:
    outcome = RecommendationPipelineOutcome(report=_pipeline_report())
    with pytest.raises(FrozenInstanceError):
        outcome.report = outcome.report  # type: ignore[misc]
    assert RecommendationPipelineOutcome.__slots__ == ("report",)


# --- 52-68: Pipeline construction ---


def test_pipeline_scorer_required() -> None:
    with pytest.raises(TypeError):
        RecommendationPipeline(scenario_scorer=LinearRegression())  # type: ignore[arg-type]


def test_pipeline_scorer_type_error() -> None:
    with pytest.raises(TypeError, match="CandidateScenarioScorer"):
        RecommendationPipeline(scenario_scorer=object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "component,cls_name",
    [
        ("safety_gate", "RecommendationSafetyGate"),
        ("constraint_resolver", "ConstraintResolver"),
        ("candidate_selector", "CandidateVariableSelector"),
        ("grid_generator", "CandidateGridGenerator"),
        ("scenario_ranker", "CandidateScenarioRanker"),
        ("recommendation_generator", "RankedScenarioRecommendationGenerator"),
        ("policy", "RecommendationPipelinePolicy"),
    ],
)
def test_pipeline_component_type_errors(component: str, cls_name: str) -> None:
    kwargs = {"scenario_scorer": _scorer(), component: object()}
    with pytest.raises(TypeError, match=cls_name):
        RecommendationPipeline(**kwargs)  # type: ignore[arg-type]


def test_pipeline_default_instance_isolation() -> None:
    pipeline_a = RecommendationPipeline(scenario_scorer=_scorer())
    pipeline_b = RecommendationPipeline(scenario_scorer=_scorer())
    assert pipeline_a.get_metadata() == pipeline_b.get_metadata()
    assert pipeline_a.get_metadata() is not pipeline_b.get_metadata()


def test_pipeline_does_not_deepcopy_dependencies() -> None:
    scorer = _scorer()
    gate = RecommendationSafetyGate()
    pipeline = RecommendationPipeline(
        scenario_scorer=scorer,
        safety_gate=gate,
    )
    assert pipeline._scenario_scorer is scorer  # noqa: SLF001
    assert pipeline._safety_gate is gate  # noqa: SLF001


def test_pipeline_policy_immutability_from_external_mutation() -> None:
    policy = RecommendationPipelinePolicy(stop_on_safety_refusal=False)
    pipeline = RecommendationPipeline(
        scenario_scorer=_scorer(),
        policy=policy,
    )
    policy.stop_on_safety_refusal = True  # type: ignore[misc]
    assert pipeline.get_metadata()["stop_on_safety_refusal"] is False


def test_pipeline_get_metadata_scalar_and_flags() -> None:
    meta = RecommendationPipeline(
        scenario_scorer=_scorer(),
    ).get_metadata()
    assert meta["has_scenario_scorer"] is True
    assert meta["performs_model_fit"] is False
    assert meta["performs_model_refit"] is False
    assert meta["computes_residual_anomaly"] is False
    for value in meta.values():
        assert value is None or isinstance(value, (str, int, float, bool))


def test_pipeline_run_request_type_error() -> None:
    pipeline = RecommendationPipeline(scenario_scorer=_scorer())
    with pytest.raises(TypeError):
        pipeline.run(_pipeline_request().model_dump())  # type: ignore[arg-type]


# --- 69-78: Call order with spies ---


def test_pipeline_happy_path_call_order() -> None:
    pipeline, calls = _pipeline_with_spies(scorer=_scorer())
    outcome = pipeline.run(_pipeline_request())
    assert calls == [
        "SAFETY",
        "CONSTRAINT_RESOLUTION",
        "CANDIDATE_SELECTION",
        "GRID_GENERATION",
        "SCENARIO_SCORING",
        "SCENARIO_RANKING",
        "RECOMMENDATION_GENERATION",
    ]
    assert outcome.report.executed_stages == list(_CANONICAL_STAGES)


def test_pipeline_each_stage_called_once_on_happy_path() -> None:
    pipeline, calls = _pipeline_with_spies(scorer=_scorer())
    pipeline.run(_pipeline_request())
    assert len(calls) == len(set(calls)) == 7


def test_pipeline_no_direct_model_calls_on_pipeline_object() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    anomaly = _AnomalyModelSpy(_fitted_anomaly_model())
    scorer = _ScenarioScorerSpy([], quality_model=quality, anomaly_model=anomaly)
    pipeline = RecommendationPipeline(scenario_scorer=scorer)
    pipeline.run(
        _pipeline_request(objective=RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY)
    )
    assert not hasattr(pipeline, "predict")
    assert not hasattr(pipeline, "score_samples")
    assert quality.predict_call_count >= 1
    assert anomaly.score_call_count >= 1


def test_scorer_spy_wraps_model_counters() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    anomaly = _AnomalyModelSpy(_fitted_anomaly_model())
    calls: list[str] = []
    pipeline, calls = _pipeline_with_spies(
        calls=calls,
        scorer=_ScenarioScorerSpy(
            calls,
            quality_model=quality,
            anomaly_model=anomaly,
        ),
    )
    pipeline.run(
        _pipeline_request(objective=RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY)
    )
    assert "SCENARIO_SCORING" in calls
    assert quality.predict_call_count >= 1
    assert anomaly.score_call_count >= 1


# --- 79-87: Safety REFUSED early exit ---


def test_safety_refused_early_exit() -> None:
    pipeline, calls = _pipeline_with_spies(scorer=_scorer())
    outcome = pipeline.run(
        _pipeline_request(ctx_overrides={"leakage_report": _blocker_leakage()}),
    )
    assert outcome.report.terminal_stage is RecommendationPipelineStage.SAFETY
    assert outcome.report.status is RecommendationStatus.REFUSED
    assert outcome.report.final_result.status is RecommendationStatus.REFUSED
    assert outcome.report.final_result.changes == []
    assert outcome.report.final_result.proposed_prediction is None
    assert outcome.report.constraint_report is None
    assert calls == ["SAFETY"]


def test_safety_refused_executed_and_skipped_stages() -> None:
    outcome = RecommendationPipeline(scenario_scorer=_scorer()).run(
        _pipeline_request(ctx_overrides={"leakage_report": _blocker_leakage()}),
    )
    report = outcome.report
    assert report.executed_stages == [RecommendationPipelineStage.SAFETY]
    assert report.skipped_stages == _CANONICAL_STAGES[1:]


def test_safety_refused_disclaimer_default() -> None:
    outcome = RecommendationPipeline(scenario_scorer=_scorer()).run(
        _pipeline_request(ctx_overrides={"leakage_report": _blocker_leakage()}),
    )
    assert outcome.report.final_result.disclaimer == DEFAULT_RECOMMENDATION_DISCLAIMER


def test_stop_on_safety_refusal_false_continues() -> None:
    pipeline, calls = _pipeline_with_spies(
        scorer=_scorer(),
        policy=RecommendationPipelinePolicy(stop_on_safety_refusal=False),
    )
    outcome = pipeline.run(
        _pipeline_request(ctx_overrides={"leakage_report": _blocker_leakage()}),
    )
    assert "CONSTRAINT_RESOLUTION" in calls
    assert outcome.report.executed_stages[0] is RecommendationPipelineStage.SAFETY


# --- 88-92: Constraint REFUSED terminal ---


def test_constraint_refused_terminal_with_approved_safety() -> None:
    pipeline, calls = _pipeline_with_spies(scorer=_scorer())
    resolver = pipeline._constraint_resolver  # noqa: SLF001
    assert isinstance(resolver, _ConstraintResolverSpy)
    resolver.override_outcome = _refused_constraint_outcome(_recommendation_request())
    outcome = pipeline.run(_pipeline_request())
    assert outcome.report.terminal_stage is (
        RecommendationPipelineStage.CONSTRAINT_RESOLUTION
    )
    assert outcome.report.constraint_report is not None
    assert outcome.report.constraint_report.status is ConstraintResolutionStatus.REFUSED
    assert outcome.report.status is RecommendationStatus.READY_FOR_OPTIMIZATION
    assert outcome.report.candidate_set is None
    assert calls == ["SAFETY", "CONSTRAINT_RESOLUTION"]


def test_constraint_refused_via_spy_override() -> None:
    pipeline, calls = _pipeline_with_spies(scorer=_scorer())
    resolver = pipeline._constraint_resolver  # noqa: SLF001
    assert isinstance(resolver, _ConstraintResolverSpy)
    resolver.override_outcome = _refused_constraint_outcome(
        _recommendation_request(),
    )
    outcome = pipeline.run(_pipeline_request())
    assert outcome.report.terminal_stage is RecommendationPipelineStage.CONSTRAINT_RESOLUTION
    assert calls == ["SAFETY", "CONSTRAINT_RESOLUTION"]


# --- 93-96: Empty candidates ---


def test_empty_candidates_terminal() -> None:
    pipeline, calls = _pipeline_with_spies(scorer=_scorer())
    selector = pipeline._candidate_selector  # noqa: SLF001
    assert isinstance(selector, _CandidateSelectorSpy)
    selector.override_outcome = _empty_candidate_set()
    outcome = pipeline.run(_pipeline_request())
    assert outcome.report.terminal_stage is RecommendationPipelineStage.CANDIDATE_SELECTION
    assert outcome.report.candidate_set is not None
    assert outcome.report.candidate_set.candidates == []
    assert outcome.report.grid_report is None
    assert calls == [
        "SAFETY",
        "CONSTRAINT_RESOLUTION",
        "CANDIDATE_SELECTION",
    ]


# --- 97-101: Grid REFUSED/EMPTY ---


def test_grid_refused_terminal() -> None:
    pipeline, calls = _pipeline_with_spies(scorer=_scorer())
    grid = pipeline._grid_generator  # noqa: SLF001
    assert isinstance(grid, _GridGeneratorSpy)
    grid.override_outcome = _refused_grid_outcome()
    outcome = pipeline.run(_pipeline_request())
    assert outcome.report.terminal_stage is RecommendationPipelineStage.GRID_GENERATION
    assert outcome.report.grid_report is not None
    assert outcome.report.grid_report.status is CandidateGridStatus.REFUSED
    assert "SCENARIO_SCORING" not in calls


def test_grid_empty_terminal() -> None:
    pipeline, calls = _pipeline_with_spies(scorer=_scorer())
    grid = pipeline._grid_generator  # noqa: SLF001
    assert isinstance(grid, _GridGeneratorSpy)
    grid.override_outcome = _empty_grid_outcome()
    outcome = pipeline.run(_pipeline_request())
    assert outcome.report.grid_report is not None
    assert outcome.report.grid_report.status is CandidateGridStatus.EMPTY
    assert "SCENARIO_SCORING" not in calls


# --- 102-106: Scoring REFUSED ---


def test_scoring_refused_terminal() -> None:
    pipeline, calls = _pipeline_with_spies(
        scorer=_ScenarioScorerSpy([], quality_model=_fitted_quality_model()),
    )
    scorer = pipeline._scenario_scorer  # noqa: SLF001
    assert isinstance(scorer, _ScenarioScorerSpy)
    scorer.override_outcome = _refused_scoring_outcome()
    outcome = pipeline.run(_pipeline_request())
    assert outcome.report.terminal_stage is RecommendationPipelineStage.SCENARIO_SCORING
    assert outcome.report.scoring_report is not None
    assert outcome.report.scoring_report.status is ScenarioScoringStatus.REFUSED
    assert "SCENARIO_RANKING" not in calls


def test_scoring_refused_with_no_models() -> None:
    outcome = RecommendationPipeline(
        scenario_scorer=CandidateScenarioScorer(),
    ).run(_pipeline_request())
    assert outcome.report.terminal_stage is RecommendationPipelineStage.SCENARIO_SCORING
    assert outcome.report.scoring_report is not None
    assert outcome.report.scoring_report.status is ScenarioScoringStatus.REFUSED


# --- 107-111: Ranking REFUSED ---


def test_ranking_refused_terminal() -> None:
    pipeline, calls = _pipeline_with_spies(scorer=_scorer())
    ranker = pipeline._scenario_ranker  # noqa: SLF001
    assert isinstance(ranker, _ScenarioRankerSpy)
    ranker.override_outcome = _refused_ranking_outcome()
    outcome = pipeline.run(_pipeline_request())
    assert outcome.report.terminal_stage is RecommendationPipelineStage.SCENARIO_RANKING
    assert outcome.report.ranking_report is not None
    assert outcome.report.ranking_report.status is ScenarioRankingStatus.REFUSED
    assert "RECOMMENDATION_GENERATION" not in calls


# --- 112-122: Full GENERATED e2e ---


def test_full_generated_e2e_quality_objective() -> None:
    outcome = RecommendationPipeline(scenario_scorer=_scorer()).run(_pipeline_request())
    report = outcome.report
    assert report.status is RecommendationStatus.GENERATED
    assert report.terminal_stage is RecommendationPipelineStage.RECOMMENDATION_GENERATION
    assert report.final_result.status is RecommendationStatus.GENERATED
    assert report.constraint_report is not None
    assert report.candidate_set is not None
    assert report.grid_report is not None
    assert report.scoring_report is not None
    assert report.ranking_report is not None
    assert len(report.final_result.changes) >= 0


def test_full_generated_e2e_anomaly_objective() -> None:
    outcome = RecommendationPipeline(
        scenario_scorer=_scorer(objective=RecommendationObjective.REDUCE_ANOMALY_SCORE),
    ).run(
        _pipeline_request(objective=RecommendationObjective.REDUCE_ANOMALY_SCORE),
    )
    assert outcome.report.status is RecommendationStatus.GENERATED
    assert outcome.report.final_result.objective is RecommendationObjective.REDUCE_ANOMALY_SCORE


def test_full_generated_e2e_balance_objective() -> None:
    outcome = RecommendationPipeline(
        scenario_scorer=_scorer(
            objective=RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
        ),
    ).run(
        _pipeline_request(
            objective=RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
            rec_overrides={
                "objective": RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
            },
        ),
    )
    assert outcome.report.status is RecommendationStatus.GENERATED


def test_ready_for_optimization_requires_approved_or_caution_with_eligible() -> None:
    pipeline, _calls = _pipeline_with_spies(scorer=_scorer())
    resolver = pipeline._constraint_resolver  # noqa: SLF001
    assert isinstance(resolver, _ConstraintResolverSpy)
    resolver.override_outcome = _refused_constraint_outcome(_recommendation_request())
    outcome = pipeline.run(_pipeline_request())
    assert outcome.report.status is RecommendationStatus.READY_FOR_OPTIMIZATION
    assert outcome.report.safety_decision.status in {
        RecommendationSafetyStatus.APPROVED,
        RecommendationSafetyStatus.CAUTION,
    }


# --- 123-129: NO_IMPROVEMENT, PARTIAL, TRUNCATED, PARTIAL constraint ---


def test_no_improvement_passed_to_generation() -> None:
    pipeline, calls = _pipeline_with_spies(
        scorer=_scorer(),
        ranker_kwargs={
            "policy": ScenarioRankingPolicy(minimum_quality_improvement=1e12),
        },
    )
    outcome = pipeline.run(_pipeline_request())
    assert "RECOMMENDATION_GENERATION" in calls
    assert outcome.report.ranking_report is not None
    assert (
        outcome.report.ranking_report.status is ScenarioRankingStatus.NO_IMPROVEMENT
    )
    assert outcome.report.status is RecommendationStatus.READY_FOR_OPTIMIZATION
    assert outcome.report.final_result.changes == []


def test_partial_ranking_with_allow_policy_generates() -> None:
    pipeline = RecommendationPipeline(
        scenario_scorer=_scorer(),
        recommendation_generator=RankedScenarioRecommendationGenerator(
            policy=RecommendationGenerationPolicy(allow_partial_ranking=True),
        ),
    )
    outcome = pipeline.run(_pipeline_request())
    assert outcome.report.status in {
        RecommendationStatus.GENERATED,
        RecommendationStatus.READY_FOR_OPTIMIZATION,
    }


def test_truncated_grid_status_reaches_generation_or_stops_by_policy() -> None:
    pipeline = RecommendationPipeline(
        scenario_scorer=_scorer(),
        grid_generator=CandidateGridGenerator(
            policy=CandidateGridPolicy(maximum_scenarios=2),
        ),
    )
    outcome = pipeline.run(_pipeline_request())
    assert outcome.report.grid_report is not None
    assert outcome.report.grid_report.status in {
        CandidateGridStatus.TRUNCATED,
        CandidateGridStatus.READY,
    }


def test_partial_constraint_resolution_continues() -> None:
    request = _pipeline_request(
        rec_overrides={
            "factors": [_factor("pressure"), _factor("temperature")],
            "current_values": {"pressure": 50.0, "temperature": 80.0},
            "constraints": [_constraint("pressure")],
            "user_confirmed_controllable_variables": ["pressure", "temperature"],
            "user_verified_variables": ["pressure", "temperature"],
        },
    )
    outcome = RecommendationPipeline(scenario_scorer=_scorer()).run(request)
    assert outcome.report.constraint_report is not None
    assert outcome.report.constraint_report.status in {
        ConstraintResolutionStatus.PARTIAL,
        ConstraintResolutionStatus.READY,
    }


# --- 130-140: Input forwarding via spies ---


def test_safety_gate_receives_request_and_context() -> None:
    pipeline, _ = _pipeline_with_spies(scorer=_scorer())
    request = _pipeline_request(metadata={"forward": "test"})
    pipeline.run(request)
    gate = pipeline._safety_gate  # noqa: SLF001
    assert isinstance(gate, _SafetyGateSpy)
    assert gate.last_kwargs["request"].objective == request.recommendation_request.objective
    assert gate.last_kwargs["context"].leakage_report.is_safe is True


def test_constraint_resolver_receives_constraints_and_safety() -> None:
    pipeline, _ = _pipeline_with_spies(scorer=_scorer())
    industry = [_constraint("pressure", minimum=10.0, maximum=90.0)]
    overrides = [_constraint("temperature", minimum=5.0, maximum=95.0)]
    request = _pipeline_request(
        industry_constraints=industry,
        user_overrides=overrides,
    )
    pipeline.run(request)
    resolver = pipeline._constraint_resolver  # noqa: SLF001
    assert isinstance(resolver, _ConstraintResolverSpy)
    assert resolver.last_kwargs["industry_constraints"][0].minimum == 10.0
    assert resolver.last_kwargs["user_overrides"][0].minimum == 5.0
    assert resolver.last_kwargs["safety_decision"].status in {
        RecommendationSafetyStatus.APPROVED,
        RecommendationSafetyStatus.CAUTION,
    }


def test_candidate_selector_receives_resolution_outcome() -> None:
    pipeline, _ = _pipeline_with_spies(scorer=_scorer())
    pipeline.run(_pipeline_request())
    selector = pipeline._candidate_selector  # noqa: SLF001
    assert isinstance(selector, _CandidateSelectorSpy)
    assert selector.last_kwargs["resolution"].report.status in {
        ConstraintResolutionStatus.READY,
        ConstraintResolutionStatus.PARTIAL,
    }


def test_grid_generator_receives_candidate_set() -> None:
    pipeline, _ = _pipeline_with_spies(scorer=_scorer())
    pipeline.run(_pipeline_request())
    grid = pipeline._grid_generator  # noqa: SLF001
    assert isinstance(grid, _GridGeneratorSpy)
    assert grid.last_candidate_set is not None
    assert len(grid.last_candidate_set.candidates) >= 1


def test_scorer_receives_scoring_request_with_features() -> None:
    pipeline, _ = _pipeline_with_spies(
        scorer=_ScenarioScorerSpy([], quality_model=_fitted_quality_model()),
    )
    pipeline.run(_pipeline_request())
    scorer = pipeline._scenario_scorer  # noqa: SLF001
    assert isinstance(scorer, _ScenarioScorerSpy)
    assert scorer.last_request is not None
    assert scorer.last_request.feature_columns == FEATURE_COLUMNS
    assert scorer.last_request.baseline_features == BASELINE_FEATURES


def test_ranker_receives_quality_direction() -> None:
    pipeline, _ = _pipeline_with_spies(scorer=_scorer())
    pipeline.run(_pipeline_request(quality_direction=QualityOptimizationDirection.MAXIMIZE))
    ranker = pipeline._scenario_ranker  # noqa: SLF001
    assert isinstance(ranker, _ScenarioRankerSpy)
    assert ranker.last_request is not None
    assert ranker.last_request.quality_direction is QualityOptimizationDirection.MAXIMIZE


def test_generator_receives_extrapolation_and_uncertainty_flags() -> None:
    pipeline, _ = _pipeline_with_spies(scorer=_scorer())
    pipeline.run(
        _pipeline_request(
            extrapolation_evaluated=True,
            extrapolation_flag=True,
            uncertainty_available=True,
            uncertainty_acceptable=True,
        )
    )
    generator = pipeline._recommendation_generator  # noqa: SLF001
    assert isinstance(generator, _RecommendationGeneratorSpy)
    assert generator.last_request is not None
    assert generator.last_request.extrapolation_evaluated is True
    assert generator.last_request.extrapolation_flag is True
    assert generator.last_request.uncertainty_available is True
    assert generator.last_request.uncertainty_acceptable is True


# --- 141-150: Terminal helper properties ---


def test_report_terminal_stage_matches_executed_suffix() -> None:
    outcome = RecommendationPipeline(scenario_scorer=_scorer()).run(_pipeline_request())
    report = outcome.report
    assert report.terminal_stage == report.executed_stages[-1]


def test_report_metadata_executed_stage_flags() -> None:
    outcome = RecommendationPipeline(scenario_scorer=_scorer()).run(_pipeline_request())
    meta = outcome.report.metadata
    assert meta["executed_stage_count"] == len(_CANONICAL_STAGES)
    assert meta["skipped_stage_count"] == 0
    assert meta["terminal_stage"] == RecommendationPipelineStage.RECOMMENDATION_GENERATION.value
    assert meta["constraint_resolution_executed"] is True
    assert meta["recommendation_generation_executed"] is True


def test_report_metadata_safety_and_final_status() -> None:
    outcome = RecommendationPipeline(scenario_scorer=_scorer()).run(_pipeline_request())
    meta = outcome.report.metadata
    assert meta["safety_status"] in {
        RecommendationSafetyStatus.APPROVED.value,
        RecommendationSafetyStatus.CAUTION.value,
    }
    assert meta["final_recommendation_status"] == outcome.report.status.value


def test_report_metadata_orchestration_flags() -> None:
    outcome = RecommendationPipeline(scenario_scorer=_scorer()).run(_pipeline_request())
    meta = outcome.report.metadata
    assert meta["model_fit_performed"] is False
    assert meta["model_refit_performed"] is False
    assert meta["residual_scoring_performed"] is False
    assert meta["pipeline_orchestration_only"] is True
    assert meta["stage_outputs_preserved"] is True


def test_report_metadata_recommendation_generated_flag() -> None:
    outcome = RecommendationPipeline(scenario_scorer=_scorer()).run(_pipeline_request())
    meta = outcome.report.metadata
    if outcome.report.status is RecommendationStatus.GENERATED:
        assert meta["recommendation_generated"] is True
    else:
        assert meta["recommendation_generated"] is False


def test_safety_terminal_metadata_stage_counts() -> None:
    outcome = RecommendationPipeline(scenario_scorer=_scorer()).run(
        _pipeline_request(ctx_overrides={"leakage_report": _blocker_leakage()}),
    )
    meta = outcome.report.metadata
    assert meta["executed_stage_count"] == 1
    assert meta["skipped_stage_count"] == len(_CANONICAL_STAGES) - 1
    assert meta["constraint_resolution_executed"] is False


# --- 151-157: Warning aggregation ---


def test_warnings_aggregate_from_stages() -> None:
    outcome = RecommendationPipeline(scenario_scorer=_scorer()).run(_pipeline_request())
    report = outcome.report
    assert isinstance(report.warnings, list)
    assert len(report.warnings) >= 1


def test_warnings_respect_maximum_aggregated_limit() -> None:
    pipeline = RecommendationPipeline(
        scenario_scorer=_scorer(),
        policy=RecommendationPipelinePolicy(maximum_aggregated_warnings=2),
    )
    outcome = pipeline.run(_pipeline_request())
    assert len(outcome.report.warnings) <= 2


def test_warnings_omission_message_when_truncated() -> None:
    pipeline = RecommendationPipeline(
        scenario_scorer=_scorer(),
        policy=RecommendationPipelinePolicy(maximum_aggregated_warnings=1),
    )
    outcome = pipeline.run(_pipeline_request())
    if len(outcome.report.warnings) == 1:
        assert _OMISSION_WARNING in outcome.report.warnings


def test_include_stage_warnings_false_uses_final_only() -> None:
    pipeline = RecommendationPipeline(
        scenario_scorer=_scorer(),
        policy=RecommendationPipelinePolicy(include_stage_warnings=False),
    )
    outcome = pipeline.run(_pipeline_request())
    safety_messages = set(outcome.report.safety_decision.messages)
    overlap = [item for item in outcome.report.warnings if item in safety_messages]
    assert overlap == []


def test_preserve_stage_outputs_false_adds_metadata_warning() -> None:
    pipeline = RecommendationPipeline(
        scenario_scorer=_scorer(),
        policy=RecommendationPipelinePolicy(preserve_stage_outputs=False),
    )
    outcome = pipeline.run(_pipeline_request())
    assert _PRESERVE_OUTPUTS_WARNING in outcome.report.warnings
    assert outcome.report.constraint_report is not None


def test_terminal_warning_included_on_early_exit() -> None:
    outcome = RecommendationPipeline(scenario_scorer=_scorer()).run(
        _pipeline_request(ctx_overrides={"leakage_report": _blocker_leakage()}),
    )
    assert any("SAFETY" in warning for warning in outcome.report.warnings)


# --- 158-169: Metadata and timing ---


def test_report_timing_fields_present_and_valid() -> None:
    outcome = RecommendationPipeline(scenario_scorer=_scorer()).run(_pipeline_request())
    report = outcome.report
    assert report.started_at.tzinfo is not None
    assert report.completed_at.tzinfo is not None
    assert report.completed_at >= report.started_at
    assert report.total_seconds >= 0.0
    assert math.isfinite(report.total_seconds)


def test_report_request_metadata_not_merged_into_report_by_default() -> None:
    outcome = RecommendationPipeline(scenario_scorer=_scorer()).run(
        _pipeline_request(metadata={"client": "test-client"}),
    )
    assert "client" not in outcome.report.metadata


def test_pipeline_get_metadata_matches_policy() -> None:
    policy = RecommendationPipelinePolicy(
        stop_on_safety_refusal=False,
        maximum_aggregated_warnings=25,
    )
    meta = RecommendationPipeline(
        scenario_scorer=_scorer(),
        policy=policy,
    ).get_metadata()
    assert meta["stop_on_safety_refusal"] is False
    assert meta["maximum_aggregated_warnings"] == 25


def test_report_extrapolation_and_uncertainty_metadata() -> None:
    outcome = RecommendationPipeline(scenario_scorer=_scorer()).run(
        _pipeline_request(
            extrapolation_evaluated=True,
            extrapolation_flag=True,
            uncertainty_available=True,
            uncertainty_acceptable=False,
        )
    )
    meta = outcome.report.metadata
    assert meta["extrapolation_evaluated"] is True
    assert meta["uncertainty_available"] is True


# --- 170-178: Exception propagation ---


@pytest.mark.parametrize(
    "exc",
    [
        TypeError("bad type"),
        DataValidationError("bad data"),
        ProcessIntelligenceError("process error"),
        ValueError("bad value"),
        ArithmeticError("math error"),
        AssertionError("assert failed"),
        KeyboardInterrupt(),
        SystemExit(1),
    ],
)
def test_exceptions_propagate_from_stage_components(exc: BaseException) -> None:
    class _RaisingSafetyGate(_SafetyGateSpy):
        def evaluate(self, request, *, context):
            raise exc

    pipeline = RecommendationPipeline(
        scenario_scorer=_scorer(),
        safety_gate=_RaisingSafetyGate([]),
    )
    with pytest.raises(type(exc)):
        pipeline.run(_pipeline_request())


def test_metadata_mutation_raises_process_intelligence_error() -> None:
    class _MutatingGate(RecommendationSafetyGate):
        def __init__(self) -> None:
            super().__init__()
            self._flip = False

        def evaluate(self, request, *, context):
            self._flip = True
            return super().evaluate(request, context=context)

        def get_metadata(self):
            meta = dict(super().get_metadata())
            if self._flip:
                meta["block_unverified_factors"] = not meta["block_unverified_factors"]
            return meta

    pipeline = RecommendationPipeline(
        scenario_scorer=_scorer(),
        safety_gate=_MutatingGate(),
    )
    with pytest.raises(ProcessIntelligenceError):
        pipeline.run(_pipeline_request())


# --- 179-198: Immutability and determinism ---


def test_mutate_request_lists_after_construction_do_not_affect_run() -> None:
    request = _pipeline_request()
    request.metadata["mutated"] = True
    request.recommendation_request.metadata["mutated"] = True
    outcome = RecommendationPipeline(scenario_scorer=_scorer()).run(request)
    assert "mutated" not in outcome.report.metadata


def test_mutate_returned_report_does_not_affect_next_run() -> None:
    pipeline = RecommendationPipeline(scenario_scorer=_scorer())
    first = pipeline.run(_pipeline_request())
    first.report.warnings.append("mutated")
    second = pipeline.run(_pipeline_request())
    assert "mutated" not in second.report.warnings


def test_consecutive_runs_do_not_accumulate_warnings() -> None:
    pipeline = RecommendationPipeline(scenario_scorer=_scorer())
    first = pipeline.run(_pipeline_request())
    second = pipeline.run(_pipeline_request())
    assert len(first.report.warnings) == len(second.report.warnings)


def test_consecutive_runs_produce_deterministic_business_results() -> None:
    pipeline = RecommendationPipeline(scenario_scorer=_scorer())
    first = pipeline.run(_pipeline_request())
    second = pipeline.run(_pipeline_request())
    assert first.report.status == second.report.status
    assert first.report.terminal_stage == second.report.terminal_stage
    assert len(first.report.final_result.changes) == len(second.report.final_result.changes)


def test_pipeline_request_deep_copy_isolation() -> None:
    original = _pipeline_request()
    copied = RecommendationPipelineRequest.model_validate(original.model_dump())
    copied.recommendation_request.metadata["changed"] = True
    assert "changed" not in original.recommendation_request.metadata


def test_outcome_report_is_immutable_via_pydantic_copy() -> None:
    outcome = RecommendationPipeline(scenario_scorer=_scorer()).run(_pipeline_request())
    copied = outcome.report.model_copy(deep=True)
    copied.warnings.append("extra")
    assert "extra" not in outcome.report.warnings


# --- 199-208: Real fitted models e2e ---


def test_real_quality_model_e2e_predict_once_per_run() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    pipeline = RecommendationPipeline(
        scenario_scorer=CandidateScenarioScorer(quality_model=quality),
    )
    pipeline.run(_pipeline_request())
    assert quality.predict_call_count >= 1


def test_real_anomaly_model_e2e_score_once_per_run() -> None:
    anomaly = _AnomalyModelSpy(_fitted_anomaly_model())
    pipeline = RecommendationPipeline(
        scenario_scorer=CandidateScenarioScorer(anomaly_model=anomaly),
    )
    pipeline.run(
        _pipeline_request(objective=RecommendationObjective.REDUCE_ANOMALY_SCORE),
    )
    assert anomaly.score_call_count >= 1


def test_negative_anomaly_score_preserved_in_e2e() -> None:
    outcome = RecommendationPipeline(
        scenario_scorer=_scorer(objective=RecommendationObjective.REDUCE_ANOMALY_SCORE),
    ).run(
        _pipeline_request(objective=RecommendationObjective.REDUCE_ANOMALY_SCORE),
    )
    scoring = outcome.report.scoring_report
    assert scoring is not None
    for score in scoring.scores:
        if score.anomaly_scored and score.anomaly_score is not None:
            assert math.isfinite(score.anomaly_score)


def test_balance_e2e_uses_both_models_without_refit() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    anomaly = _AnomalyModelSpy(_fitted_anomaly_model())
    pipeline = RecommendationPipeline(
        scenario_scorer=CandidateScenarioScorer(
            quality_model=quality,
            anomaly_model=anomaly,
        ),
    )
    outcome = pipeline.run(
        _pipeline_request(
            objective=RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
            rec_overrides={
                "objective": RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
            },
        ),
    )
    assert quality.predict_call_count >= 1
    assert anomaly.score_call_count >= 1
    assert outcome.report.metadata["model_refit_performed"] is False
    assert outcome.report.metadata["residual_scoring_performed"] is False


def test_quality_only_scorer_for_quality_objective() -> None:
    quality = _QualityModelSpy(_fitted_quality_model())
    pipeline = RecommendationPipeline(
        scenario_scorer=CandidateScenarioScorer(quality_model=quality),
    )
    outcome = pipeline.run(_pipeline_request())
    assert quality.predict_call_count >= 1
    assert outcome.report.scoring_report is not None
    assert outcome.report.scoring_report.quality_scored_count >= 1


def test_anomaly_only_scorer_for_anomaly_objective() -> None:
    anomaly = _AnomalyModelSpy(_fitted_anomaly_model())
    pipeline = RecommendationPipeline(
        scenario_scorer=CandidateScenarioScorer(anomaly_model=anomaly),
    )
    outcome = pipeline.run(
        _pipeline_request(objective=RecommendationObjective.REDUCE_ANOMALY_SCORE),
    )
    assert anomaly.score_call_count >= 1
    assert outcome.report.scoring_report is not None
    assert outcome.report.scoring_report.anomaly_scored_count >= 1


# --- Additional request/report coverage ---


@pytest.mark.parametrize("bad", [0, 1, "true", None])
def test_pipeline_request_bool_strict_validation(bad: object) -> None:
    with pytest.raises(ValidationError):
        _pipeline_request(extrapolation_evaluated=bad)  # type: ignore[arg-type]


def test_pipeline_request_baseline_nan_rejected() -> None:
    with pytest.raises(ValidationError):
        _pipeline_request(
            baseline_features={
                "pressure": float("nan"),
                "temperature": 80.0,
                "humidity": 45.0,
            },
        )


def test_pipeline_request_duplicate_user_overrides_rejected() -> None:
    with pytest.raises(ValidationError):
        _pipeline_request(
            user_overrides=[_constraint("pressure"), _constraint("pressure")],
        )


def test_pipeline_request_target_quality_direction_valid() -> None:
    request = _pipeline_request(
        quality_direction=QualityOptimizationDirection.TARGET,
        quality_target=12.5,
    )
    assert request.quality_target == pytest.approx(12.5)


def test_current_values_match_baseline_via_isclose() -> None:
    request = _pipeline_request(
        rec_overrides={
            "current_values": {"pressure": 50.0 + 1e-15, "temperature": 80.0},
        },
    )
    assert math.isclose(
        request.recommendation_request.current_values["pressure"],
        request.baseline_features["pressure"],
        abs_tol=1e-12,
    )


@pytest.mark.parametrize(
    "stop_field",
    [
        "stop_on_grid_refusal_or_empty",
        "stop_on_scoring_refusal",
        "stop_on_ranking_refusal",
        "stop_on_empty_candidates",
        "stop_on_constraint_refusal",
    ],
)
def test_stop_flags_false_continue_past_terminal_status(stop_field: str) -> None:
    policy = RecommendationPipelinePolicy(**{stop_field: False})
    calls: list[str] = []
    pipeline, calls = _pipeline_with_spies(
        calls=calls,
        scorer=_ScenarioScorerSpy(calls, quality_model=_fitted_quality_model()),
        policy=policy,
    )
    if stop_field == "stop_on_constraint_refusal":
        resolver = pipeline._constraint_resolver  # noqa: SLF001
        assert isinstance(resolver, _ConstraintResolverSpy)
        resolver.override_outcome = _refused_constraint_outcome(_recommendation_request())
    elif stop_field == "stop_on_empty_candidates":
        selector = pipeline._candidate_selector  # noqa: SLF001
        assert isinstance(selector, _CandidateSelectorSpy)
        selector.override_outcome = _empty_candidate_set()
    elif stop_field == "stop_on_grid_refusal_or_empty":
        grid = pipeline._grid_generator  # noqa: SLF001
        assert isinstance(grid, _GridGeneratorSpy)
        grid.override_outcome = _refused_grid_outcome()
    elif stop_field == "stop_on_scoring_refusal":
        scorer = pipeline._scenario_scorer  # noqa: SLF001
        assert isinstance(scorer, _ScenarioScorerSpy)
        scorer.override_outcome = _refused_scoring_outcome()
    elif stop_field == "stop_on_ranking_refusal":
        ranker = pipeline._scenario_ranker  # noqa: SLF001
        assert isinstance(ranker, _ScenarioRankerSpy)
        ranker.override_outcome = _refused_ranking_outcome()
    pipeline.run(_pipeline_request())
    assert len(calls) > 1


