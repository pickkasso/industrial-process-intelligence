"""End-to-end industrial process analysis workflow orchestrator (Step 10C).

Wires existing public components from the data, routing, industries, evaluation,
models, diagnosis, and recommendation packages into a single leakage-safe
raw-CSV analysis path. The orchestrator does not reimplement modeling,
diagnosis, ranking, or recommendation algorithms; it only sequences existing
concrete components and aggregates their structured outputs.

Design contracts:
    * Load raw CSV, profile, validate, quality-score, sort, then defer
      preprocessing fit until after the leakage-safe split.
    * Preprocessing is fit on the training partition only and applied to
      validation and test; the full dataset is never fit.
    * Test partitions are never used for model selection or threshold
      calibration.
    * Diagnosis and recommendation express association, not causation, and
      surface uncertainty rather than guaranteed outcomes.
    * Every ``run`` call is stateless; no caches are retained between runs.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import NoReturn, TypeVar

import polars as pl

from process_intelligence.core.enums import AnalysisTask, AnomalyType, ColumnRole
from process_intelligence.core.exceptions import (
    DataValidationError,
    InsufficientDataError,
    ProcessIntelligenceError,
)
from process_intelligence.core.protocols import BaseIndustryProfile
from process_intelligence.core.schemas import (
    AnomalyEvent,
    RootCauseFactor,
    VariableConstraint,
)
from process_intelligence.data import (
    ORIGINAL_ROW_ID_COLUMN,
    DataQualityScorer,
    DatasetLoader,
    DatasetPreprocessor,
    DatasetProfiler,
    DatasetSorter,
    DatasetValidator,
    PreprocessorConfig,
    TargetSuitabilityAssessment,
    evaluate_target_suitability,
)
from process_intelligence.diagnosis import (
    DiagnosisEnsembleConfig,
    DiagnosisEnsembleDiagnoser,
    DiagnosisMethod,
    DiagnosisRequest,
    DiagnosisResult,
    DiagnosisScope,
    RobustGroupComparisonDiagnoser,
)
from process_intelligence.evaluation import (
    DatasetSplit,
    DatasetSplitter,
    FinalEvaluationOutcome,
    FinalModelEvaluator,
    LeakageChecker,
    LeakageReport,
    ModelPerformanceAcceptanceReport,
    ModelPerformanceAcceptanceStatus,
    ModelPerformanceAssessor,
    SplitConfig,
    SplitStrategy,
)
from process_intelligence.industries import (
    AutomotiveIndustryProfile,
    BatteryIndustryProfile,
    GenericIndustryProfile,
    IndustryRegistry,
    SemiconductorIndustryProfile,
)
from process_intelligence.models import (
    AnomalyFinalEvaluationOutcome,
    AnomalyFinalEvaluator,
    ModelRegistry,
    ResidualAnomalyFinalEvaluationOutcome,
    ResidualAnomalyFinalEvaluator,
    ResidualAnomalyPipeline,
    SupervisedModelScreener,
    UnsupervisedAnomalyModelScreener,
    create_default_anomaly_model_registry,
    create_default_supervised_model_registry,
)
from process_intelligence.recommendation import (
    CandidateScenarioScorer,
    QualityOptimizationDirection,
    RecommendationObjective,
    RecommendationPipeline,
    RecommendationPipelineRequest,
    RecommendationReasonCode,
    RecommendationRequest,
    RecommendationResult,
    RecommendationSafetyContext,
    RecommendationSafetyDecision,
    RecommendationSafetyGate,
    RecommendationSafetyPolicy,
    RecommendationSafetyStatus,
    RecommendationStatus,
    RecommendationWhatIfVerifier,
    WhatIfVerificationStatus,
    recommendation_warnings_for_stability,
)
from process_intelligence.recommendation.candidate_grid import CandidateGridReport
from process_intelligence.recommendation.schemas import DEFAULT_RECOMMENDATION_DISCLAIMER
from process_intelligence.recommendation.what_if_verification import (
    RecommendationWhatIfVerificationResult,
)
from process_intelligence.routing import (
    AnalysisTaskRouter,
    ColumnRoleMapper,
    IndustryRouter,
    create_default_industry_router,
    create_default_task_router,
)
from process_intelligence.workflow.anomaly_context import build_anomaly_context_windows
from process_intelligence.workflow.cohort_filter import apply_numeric_cohort_filter
from process_intelligence.workflow.dataset_fingerprint import (
    compute_dataset_content_fingerprint,
)
from process_intelligence.workflow.enums import (
    AnalysisExecutionMode,
    AnalysisWorkflowStage,
    AnalysisWorkflowStatus,
    AnomalyContextOrderBasis,
    OperatingPointSelectionMode,
    TaskSelectionSource,
)
from process_intelligence.workflow.schemas import (
    AnalysisWorkflowOutcome,
    AnalysisWorkflowPolicy,
    AnalysisWorkflowReport,
    AnalysisWorkflowRequest,
    AnalysisWorkflowStageRecord,
    AnomalyContextWindow,
    CohortFilterSummary,
    ScalarMetadataValue,
)

_RESIDUAL_INDICATOR_COLUMN = "_is_residual_anomaly"
_RESIDUAL_SCORE_COLUMN = "_residual_anomaly_score"
_ANOMALY_INDICATOR_COLUMN = "_is_anomaly"
_ANOMALY_SCORE_COLUMN = "_anomaly_score"

_SEMICONDUCTOR_INDUSTRY = "semiconductor"

_SKIPPED_NOT_APPLICABLE = "Not applicable for anomaly-only analysis."
_ANOMALY_ONLY_OVERVIEW = (
    "Anomaly-only analysis completed. Recommendation generation is not "
    "enabled for this analysis mode."
)
_ANOMALY_ONLY_RECOMMENDATION_DISABLED = (
    "Recommendation generation is not enabled for anomaly-only analysis."
)
_ANOMALY_ONLY_NO_ELIGIBLE = (
    "No eligible verified controllable variable was available for "
    "anomaly-score reduction recommendation."
)
_ANOMALY_RECOMMENDATION_ASSOCIATION_NOTE = (
    "Diagnosis factors are associations, not established causes; "
    "recommendation candidates are limited to the intersection of diagnosis "
    "factors with explicitly confirmed, verified, and constrained controllable "
    "process inputs."
)
_CONTROLLABLE_PROCESS_ROLES = frozenset({ColumnRole.CONTROLLABLE_PROCESS})
_SELECTION_SOURCE_UNSUPERVISED = "UNSUPERVISED_ANOMALY_SCORE"
_DIAGNOSIS_SOURCE_ROBUST = "ROBUST_GROUP_COMPARISON"

_WARNING_OMISSION = (
    "Additional workflow warnings were omitted due to the configured limit."
)


class _WorkflowHalt(Exception):
    """Internal control-flow signal to stop a run after a terminal stage.

    This is a private orchestration signal, not a domain error. It carries the
    terminal business status and is always caught inside ``run``.
    """

    def __init__(self, status: AnalysisWorkflowStatus) -> None:
        super().__init__(status.value)
        self.status = status


@dataclass
class _RunState:
    """Mutable per-run bookkeeping for stage records, counts, and flags.

    Holds only scalar report inputs and the executed stage records. Does not
    store DataFrames, estimators, or intermediate model objects.
    """

    records: list[AnalysisWorkflowStageRecord] = field(default_factory=list)
    status: AnalysisWorkflowStatus = AnalysisWorkflowStatus.REFUSED
    raw_row_count: int = 0
    processed_row_count: int = 0
    cohort_row_count: int = 0
    train_row_count: int = 0
    validation_row_count: int = 0
    test_row_count: int = 0
    cohort_filter_summary: CohortFilterSummary | None = None
    selected_industry: str | None = None
    selected_task: AnalysisTask | None = None
    inferred_task: AnalysisTask | None = None
    task_selection_source: TaskSelectionSource | None = None
    task_override_applied: bool = False
    selected_supervised_model_key: str | None = None
    selected_anomaly_model_key: str | None = None
    selected_operating_row_id: int | str | None = None
    anomaly_event_count: int = 0
    diagnosis_factor_count: int = 0
    anomaly_events: list[AnomalyEvent] = field(default_factory=list)
    diagnosis_factors: list[RootCauseFactor] = field(default_factory=list)
    anomaly_context_windows: list[AnomalyContextWindow] = field(default_factory=list)
    context_warnings: list[str] = field(default_factory=list)
    raw_csv_loaded: bool = False
    dataset_fingerprint: str | None = None
    supervised_model_available: bool = False
    anomaly_model_available: bool = False
    residual_calibration_performed: bool = False
    independent_test_evaluation_performed: bool = False
    diagnosis_performed: bool = False
    recommendation_pipeline_executed: bool = False
    recommendation_generated: bool = False
    row_identity_preserved: bool = False
    model_performance_assessment: ModelPerformanceAcceptanceReport | None = None
    final_recommendation: RecommendationResult | None = None
    recommendation_verification: RecommendationWhatIfVerificationResult | None = None
    analysis_mode: AnalysisExecutionMode = AnalysisExecutionMode.SUPERVISED
    recommendation_applicable: bool = True
    diagnosis_source: str | None = None
    anomaly_event_selection_source: str | None = None
    target_column: str | None = None
    feature_count: int | None = None
    target_suitable: bool | None = None
    target_unique_non_null_count: int | None = None
    target_refusal_code: str | None = None
    target_suitability_message: str | None = None


def _row_id_set(frame: pl.DataFrame) -> set[int]:
    """Return the set of original row identifiers in a frame."""
    return {int(value) for value in frame.get_column(ORIGINAL_ROW_ID_COLUMN).to_list()}


def _finite_float(value: object) -> float:
    """Coerce a scalar to a finite Python float or raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected a finite number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"expected a finite number, got {value!r}")
    return number


class IndustrialProcessAnalysisWorkflow:
    """Orchestrate the raw-CSV → recommendation industrial analysis path.

    The workflow sequences existing public components in a fixed canonical
    stage order and returns a single structured ``AnalysisWorkflowOutcome``.
    It performs model fit and refit through the existing components, keeps the
    test partition strictly for independent evaluation, and never claims
    causal guarantees. Each ``run`` call is independent; the instance holds no
    per-run state between calls.
    """

    def __init__(
        self,
        *,
        loader: DatasetLoader | None = None,
        profiler: DatasetProfiler | None = None,
        validator: DatasetValidator | None = None,
        quality_scorer: DataQualityScorer | None = None,
        sorter: DatasetSorter | None = None,
        preprocessor: DatasetPreprocessor | None = None,
        industry_router: IndustryRouter | None = None,
        industry_registry: IndustryRegistry | None = None,
        task_router: AnalysisTaskRouter | None = None,
        role_mapper: ColumnRoleMapper | None = None,
        splitter: DatasetSplitter | None = None,
        leakage_checker: LeakageChecker | None = None,
        supervised_registry: ModelRegistry | None = None,
        anomaly_registry: ModelRegistry | None = None,
        residual_pipeline: ResidualAnomalyPipeline | None = None,
        residual_final_evaluator: ResidualAnomalyFinalEvaluator | None = None,
        diagnosis_ensemble: DiagnosisEnsembleDiagnoser | None = None,
        policy: AnalysisWorkflowPolicy | None = None,
    ) -> None:
        self._loader = _resolve_dependency(loader, DatasetLoader, DatasetLoader)
        self._profiler = _resolve_dependency(profiler, DatasetProfiler, DatasetProfiler)
        self._validator = _resolve_dependency(
            validator, DatasetValidator, DatasetValidator
        )
        self._quality_scorer = _resolve_dependency(
            quality_scorer, DataQualityScorer, DataQualityScorer
        )
        self._sorter = _resolve_dependency(sorter, DatasetSorter, DatasetSorter)

        if preprocessor is not None and not isinstance(
            preprocessor, DatasetPreprocessor
        ):
            raise TypeError(
                "preprocessor must be DatasetPreprocessor or None, "
                f"got {type(preprocessor).__name__}"
            )
        # A fresh preprocessor is always created per run from the request
        # configuration to keep runs stateless and leakage-safe. Any injected
        # instance is accepted only for type validation.
        self._preprocessor = preprocessor

        self._industry_router = _resolve_dependency(
            industry_router,
            IndustryRouter,
            create_default_industry_router,
        )

        if industry_registry is None:
            self._industry_registry = IndustryRegistry(
                profiles=[
                    SemiconductorIndustryProfile(),
                    BatteryIndustryProfile(),
                    AutomotiveIndustryProfile(),
                    GenericIndustryProfile(),
                ]
            )
        elif isinstance(industry_registry, IndustryRegistry):
            self._industry_registry = industry_registry
        else:
            raise TypeError(
                "industry_registry must be IndustryRegistry or None, "
                f"got {type(industry_registry).__name__}"
            )

        self._task_router = _resolve_dependency(
            task_router,
            AnalysisTaskRouter,
            create_default_task_router,
        )
        self._role_mapper = _resolve_dependency(
            role_mapper, ColumnRoleMapper, ColumnRoleMapper
        )
        self._splitter = _resolve_dependency(splitter, DatasetSplitter, DatasetSplitter)
        self._leakage_checker = _resolve_dependency(
            leakage_checker, LeakageChecker, LeakageChecker
        )

        if supervised_registry is None:
            self._supervised_registry: ModelRegistry = (
                create_default_supervised_model_registry()
            )
        elif isinstance(supervised_registry, ModelRegistry):
            self._supervised_registry = supervised_registry
        else:
            raise TypeError(
                "supervised_registry must be ModelRegistry or None, "
                f"got {type(supervised_registry).__name__}"
            )

        if anomaly_registry is None:
            self._anomaly_registry: ModelRegistry = (
                create_default_anomaly_model_registry()
            )
        elif isinstance(anomaly_registry, ModelRegistry):
            self._anomaly_registry = anomaly_registry
        else:
            raise TypeError(
                "anomaly_registry must be ModelRegistry or None, "
                f"got {type(anomaly_registry).__name__}"
            )

        self._residual_pipeline = _resolve_dependency(
            residual_pipeline,
            ResidualAnomalyPipeline,
            ResidualAnomalyPipeline,
        )
        self._residual_final_evaluator = _resolve_dependency(
            residual_final_evaluator,
            ResidualAnomalyFinalEvaluator,
            ResidualAnomalyFinalEvaluator,
        )

        if diagnosis_ensemble is not None and not isinstance(
            diagnosis_ensemble, DiagnosisEnsembleDiagnoser
        ):
            raise TypeError(
                "diagnosis_ensemble must be DiagnosisEnsembleDiagnoser or None, "
                f"got {type(diagnosis_ensemble).__name__}"
            )
        self._diagnosis_ensemble = diagnosis_ensemble

        if policy is None:
            self._policy = AnalysisWorkflowPolicy()
        elif isinstance(policy, AnalysisWorkflowPolicy):
            self._policy = policy.model_copy(deep=True)
        else:
            raise TypeError(
                "policy must be AnalysisWorkflowPolicy or None, "
                f"got {type(policy).__name__}"
            )

    def get_metadata(self) -> dict[str, ScalarMetadataValue]:
        """Return scalar orchestration metadata for the configured policy."""
        policy = self._policy
        return {
            "stage_count": len(list(AnalysisWorkflowStage)),
            "stop_on_validation_blocker": policy.stop_on_validation_blocker,
            "stop_on_leakage_blocker": policy.stop_on_leakage_blocker,
            "require_semiconductor_industry": policy.require_semiconductor_industry,
            "require_regression_task": policy.require_regression_task,
            "require_residual_diagnosis": policy.require_residual_diagnosis,
            "allow_partial_diagnosis_ensemble": (
                policy.allow_partial_diagnosis_ensemble
            ),
            "maximum_anomaly_events": policy.maximum_anomaly_events,
            "minimum_anomaly_events": policy.minimum_anomaly_events,
            "preserve_stage_outputs": policy.preserve_stage_outputs,
            "include_stage_warnings": policy.include_stage_warnings,
            "maximum_aggregated_warnings": policy.maximum_aggregated_warnings,
            "loads_raw_csv": True,
            "performs_model_screening": True,
            "performs_independent_test_evaluation": True,
            "performs_residual_calibration": True,
            "performs_diagnosis": True,
            "performs_recommendation": True,
            "computes_extrapolation": False,
            "computes_uncertainty": False,
            "supports_datetime_string_parsing": False,
        }

    def run(self, request: AnalysisWorkflowRequest) -> AnalysisWorkflowOutcome:
        """Execute the workflow for a validated request.

        Args:
            request: Validated analysis workflow request.

        Returns:
            An immutable outcome wrapping a validated workflow report.

        Raises:
            TypeError: If ``request`` is not an ``AnalysisWorkflowRequest``.
            DataValidationError | ValueError | ProcessIntelligenceError |
            ArithmeticError | AssertionError: Propagated from underlying
                components for genuine failures that are not structured refusals.
        """
        if not isinstance(request, AnalysisWorkflowRequest):
            raise TypeError(
                "request must be AnalysisWorkflowRequest, "
                f"got {type(request).__name__}"
            )

        started_at = datetime.now(tz=UTC)
        perf_started = time.perf_counter()
        policy = self._policy.model_copy(deep=True)
        state = _RunState()

        try:
            self._execute(request, policy=policy, state=state)
        except _WorkflowHalt as halt:
            state.status = halt.status

        completed_at = datetime.now(tz=UTC)
        total_seconds = max(0.0, time.perf_counter() - perf_started)
        report = self._build_report(
            policy=policy,
            state=state,
            started_at=started_at,
            completed_at=completed_at,
            total_seconds=total_seconds,
        )
        return AnalysisWorkflowOutcome(report=report)

    # ------------------------------------------------------------------
    # Stage orchestration
    # ------------------------------------------------------------------

    def _execute(
        self,
        request: AnalysisWorkflowRequest,
        *,
        policy: AnalysisWorkflowPolicy,
        state: _RunState,
    ) -> None:
        feature_columns = list(request.feature_columns)
        state.analysis_mode = request.analysis_mode
        state.target_column = request.target_column
        state.feature_count = len(feature_columns)
        anomaly_only = request.analysis_mode is AnalysisExecutionMode.ANOMALY_ONLY
        if anomaly_only:
            state.recommendation_applicable = bool(
                request.anomaly_recommendation.enabled
            )
            state.target_suitable = None
            state.target_suitability_message = "NOT_APPLICABLE"
        else:
            state.recommendation_applicable = True

        # --- LOAD ---
        csv_path = request.csv_path
        if not csv_path.exists() or not csv_path.is_file():
            self._refuse(
                state,
                AnalysisWorkflowStage.LOAD,
                "The requested CSV file does not exist or is not a regular file.",
            )
        load_warnings: list[str] = []
        try:
            state.dataset_fingerprint = compute_dataset_content_fingerprint(csv_path)
        except (OSError, ValueError, TypeError) as exc:
            state.dataset_fingerprint = None
            load_warnings.append(
                "Dataset content fingerprint could not be computed "
                f"({type(exc).__name__}); run comparison will be unavailable."
            )
        loaded = self._loader.load(csv_path)
        loaded_frame = loaded.frame
        metadata = loaded.metadata
        if loaded_frame.height < 1:
            self._refuse(
                state,
                AnalysisWorkflowStage.LOAD,
                "The loaded CSV dataset contains no rows.",
            )
        state.raw_row_count = loaded_frame.height
        state.processed_row_count = loaded_frame.height
        state.cohort_row_count = loaded_frame.height
        state.raw_csv_loaded = True
        self._ok(
            state,
            AnalysisWorkflowStage.LOAD,
            f"Loaded raw CSV dataset '{metadata.file_name}' with "
            f"{loaded_frame.height} rows.",
            row_count=loaded_frame.height,
            warnings=load_warnings,
            metadata={"column_count": len(loaded_frame.columns)},
        )

        # --- PROFILE ---
        profile = self._profiler.profile(loaded_frame)
        self._ok(
            state,
            AnalysisWorkflowStage.PROFILE,
            "Profiled the loaded dataset column statistics.",
            row_count=profile.row_count,
            metadata={"column_count": profile.column_count},
        )

        # --- VALIDATE ---
        issues = self._validator.validate(loaded_frame)
        blockers = [issue for issue in issues if issue.severity == "ERROR"]
        validation_warnings = [
            _format_validation_warning(issue)
            for issue in issues
            if issue.severity != "ERROR"
        ]
        if anomaly_only:
            if blockers and policy.stop_on_validation_blocker:
                self._refuse(
                    state,
                    AnalysisWorkflowStage.VALIDATE,
                    "Validation found blocking (ERROR) data quality issues; "
                    "the workflow stopped before analysis.",
                    warnings=validation_warnings,
                    metadata={"blocker_count": len(blockers)},
                )
            self._ok(
                state,
                AnalysisWorkflowStage.VALIDATE,
                "Validated the loaded dataset for structural quality issues. "
                "Target suitability is not applicable for anomaly-only analysis.",
                warnings=validation_warnings,
                metadata={
                    "issue_count": len(issues),
                    "blocker_count": len(blockers),
                    "target_suitable": None,
                    "target_suitability": "NOT_APPLICABLE",
                    "analysis_mode": request.analysis_mode.value,
                },
            )
        else:
            if request.target_column is None:
                self._refuse(
                    state,
                    AnalysisWorkflowStage.VALIDATE,
                    "Target column is required for supervised analysis.",
                )
            if request.target_column not in loaded_frame.columns:
                self._refuse(
                    state,
                    AnalysisWorkflowStage.VALIDATE,
                    f"Target column '{request.target_column}' was not found in the "
                    "loaded dataset. Supervised analysis cannot continue.",
                    warnings=validation_warnings,
                    metadata={
                        "issue_count": len(issues),
                        "blocker_count": len(blockers),
                        "target_column": request.target_column,
                        "target_suitable": False,
                    },
                )
            target_assessment = evaluate_target_suitability(
                loaded_frame,
                request.target_column,
                requested_task=None,
            )
            self._store_target_suitability(state, target_assessment)
            if not target_assessment.suitable:
                refusal_metadata: dict[str, ScalarMetadataValue] = {
                    "issue_count": len(issues),
                    "blocker_count": len(blockers),
                    "target_column": target_assessment.target_column,
                    "target_suitable": False,
                    "target_unique_non_null_count": (
                        target_assessment.unique_non_null_count
                    ),
                    "target_refusal_code": (
                        None
                        if target_assessment.refusal_code is None
                        else target_assessment.refusal_code.value
                    ),
                    "row_count": target_assessment.row_count,
                    "non_null_count": target_assessment.non_null_count,
                    "null_count": target_assessment.null_count,
                    "is_all_null": target_assessment.is_all_null,
                    "is_constant": target_assessment.is_constant,
                    "is_numeric": target_assessment.is_numeric,
                }
                if target_assessment.constant_value is not None:
                    refusal_metadata["constant_value"] = (
                        target_assessment.constant_value
                    )
                self._refuse(
                    state,
                    AnalysisWorkflowStage.VALIDATE,
                    target_assessment.message,
                    warnings=validation_warnings,
                    metadata=refusal_metadata,
                )
            if blockers and policy.stop_on_validation_blocker:
                self._refuse(
                    state,
                    AnalysisWorkflowStage.VALIDATE,
                    "Validation found blocking (ERROR) data quality issues; "
                    "the workflow stopped before analysis.",
                    warnings=validation_warnings,
                    metadata={"blocker_count": len(blockers)},
                )
            self._ok(
                state,
                AnalysisWorkflowStage.VALIDATE,
                "Validated the loaded dataset for structural quality issues and "
                "target suitability.",
                warnings=validation_warnings,
                metadata={
                    "issue_count": len(issues),
                    "blocker_count": len(blockers),
                    "target_column": target_assessment.target_column,
                    "target_suitable": True,
                    "target_unique_non_null_count": (
                        target_assessment.unique_non_null_count
                    ),
                },
            )

        # --- QUALITY_SCORE ---
        quality = self._quality_scorer.score(issues)
        self._ok(
            state,
            AnalysisWorkflowStage.QUALITY_SCORE,
            "Computed the overall data quality score.",
            metadata={"total_score": _finite_float(quality.total_score)},
        )

        # --- SORT ---
        if request.timestamp_column is not None:
            sort_result = self._sorter.sort(
                loaded_frame,
                by=request.timestamp_column,
            )
            sorted_frame = sort_result.frame
            context_order_basis = AnomalyContextOrderBasis.SORTED_ANALYSIS_ORDER
            sort_message = (
                "Applied a stable chronological sort using the timestamp column."
            )
        else:
            sorted_frame = loaded_frame
            context_order_basis = AnomalyContextOrderBasis.LOADED_ROW_ORDER
            sort_message = (
                "No timestamp column provided; preserved the loaded row order "
                "without sorting."
            )
        state.row_identity_preserved = _row_id_set(sorted_frame) == _row_id_set(
            loaded_frame
        )
        self._ok(
            state,
            AnalysisWorkflowStage.SORT,
            sort_message,
            row_count=sorted_frame.height,
        )

        # --- COHORT_FILTER ---
        analysis_frame = sorted_frame
        if request.cohort_filter is None:
            state.cohort_row_count = analysis_frame.height
            state.cohort_filter_summary = CohortFilterSummary(
                configured=False,
                column_name=None,
                lower_bound=None,
                upper_bound=None,
                include_lower=None,
                include_upper=None,
                exclude_filter_column_from_features=None,
                source_row_count=analysis_frame.height,
                retained_row_count=analysis_frame.height,
                excluded_row_count=0,
                null_excluded_count=0,
            )
            self._skip(
                state,
                AnalysisWorkflowStage.COHORT_FILTER,
                "No operating cohort filter was configured.",
            )
        else:
            try:
                cohort_outcome = apply_numeric_cohort_filter(
                    analysis_frame,
                    request.cohort_filter,
                )
            except DataValidationError as exc:
                self._refuse(
                    state,
                    AnalysisWorkflowStage.COHORT_FILTER,
                    str(exc),
                    metadata={
                        "cohort_filter_column": request.cohort_filter.column_name,
                        "cohort_filter_lower_bound": (
                            request.cohort_filter.lower_bound
                        ),
                        "cohort_filter_upper_bound": (
                            request.cohort_filter.upper_bound
                        ),
                    },
                )
            analysis_frame = cohort_outcome.frame
            state.cohort_row_count = cohort_outcome.retained_row_count
            state.cohort_filter_summary = CohortFilterSummary(
                configured=True,
                column_name=cohort_outcome.filter.column_name,
                lower_bound=cohort_outcome.filter.lower_bound,
                upper_bound=cohort_outcome.filter.upper_bound,
                include_lower=cohort_outcome.filter.include_lower,
                include_upper=cohort_outcome.filter.include_upper,
                exclude_filter_column_from_features=(
                    cohort_outcome.filter.exclude_filter_column_from_features
                ),
                source_row_count=cohort_outcome.source_row_count,
                retained_row_count=cohort_outcome.retained_row_count,
                excluded_row_count=cohort_outcome.excluded_row_count,
                null_excluded_count=cohort_outcome.null_excluded_count,
            )
            if request.cohort_filter.exclude_filter_column_from_features:
                filter_column = request.cohort_filter.column_name
                if filter_column in feature_columns:
                    feature_columns = [
                        name for name in feature_columns if name != filter_column
                    ]
                    state.feature_count = len(feature_columns)
                if not feature_columns:
                    self._refuse(
                        state,
                        AnalysisWorkflowStage.COHORT_FILTER,
                        "Excluding the cohort filter column left no modeling "
                        "features for anomaly analysis.",
                        metadata=cohort_outcome.metadata(),
                    )

            # Validate the filtered cohort against the same splitter contract
            # used later for train/validation/test partitioning.
            if request.timestamp_column is not None:
                probe_split_config = SplitConfig(
                    strategy=SplitStrategy.TIME,
                    time_column=request.timestamp_column,
                    test_size=0.20,
                    validation_size=0.20,
                    random_state=42,
                )
            else:
                probe_split_config = SplitConfig(
                    strategy=SplitStrategy.RANDOM,
                    allow_random_split=True,
                    test_size=0.20,
                    validation_size=0.20,
                    random_state=42,
                )
            try:
                self._splitter.split(analysis_frame, probe_split_config)
            except InsufficientDataError:
                self._refuse(
                    state,
                    AnalysisWorkflowStage.COHORT_FILTER,
                    "The configured cohort filter retained too few rows for the "
                    "required train/validation/test split.",
                    metadata=cohort_outcome.metadata(),
                )

            self._ok(
                state,
                AnalysisWorkflowStage.COHORT_FILTER,
                (
                    "Applied the user-confirmed numeric operating cohort filter "
                    f"on '{request.cohort_filter.column_name}'."
                ),
                row_count=cohort_outcome.retained_row_count,
                metadata=cohort_outcome.metadata(),
            )

        # Keep the historical local name for the analysis-order frame used by
        # split, diagnosis context, and later stages.
        sorted_frame = analysis_frame

        # --- PREPROCESS (deferred fit until after SPLIT) ---
        preprocessor_config = PreprocessorConfig(
            numeric_columns=list(feature_columns),
            categorical_columns=[],
            numeric_imputation="median",
            categorical_imputation="none",
            scaling="none",
        )
        preprocessor = DatasetPreprocessor(preprocessor_config)
        self._ok(
            state,
            AnalysisWorkflowStage.PREPROCESS,
            "Configured leakage-safe preprocessing; the training-only fit is "
            "deferred until after the dataset split.",
            metadata={"numeric_feature_count": len(feature_columns)},
        )

        # --- INDUSTRY_ROUTING ---
        industry_result = self._industry_router.route(metadata)
        selected_industry = industry_result.selected_industry
        state.selected_industry = selected_industry
        if (
            policy.require_semiconductor_industry
            and selected_industry.strip().casefold() != _SEMICONDUCTOR_INDUSTRY
        ):
            self._refuse(
                state,
                AnalysisWorkflowStage.INDUSTRY_ROUTING,
                "Policy requires the semiconductor industry, but routing selected "
                f"'{selected_industry}'.",
                metadata={"selected_industry": selected_industry},
            )
        industry_profile = self._industry_registry.get(selected_industry)
        self._ok(
            state,
            AnalysisWorkflowStage.INDUSTRY_ROUTING,
            f"Routed the dataset to the '{selected_industry}' industry profile.",
            metadata={
                "selected_industry": selected_industry,
                "requires_user_confirmation": (
                    industry_result.requires_user_confirmation
                ),
            },
        )

        # Column-role mapping is required for task routing; it is computed here
        # but its stage record is emitted after TASK_ROUTING to honour the
        # canonical stage order.
        sorted_profile = self._profiler.profile(sorted_frame)
        role_mapping = self._role_mapper.map_roles(
            sorted_profile,
            industry_profile.get_schema_hints(),
            overrides=request.column_role_overrides
            if request.column_role_overrides
            else None,
        )

        # --- TASK_ROUTING ---
        if anomaly_only:
            self._skip(
                state,
                AnalysisWorkflowStage.TASK_ROUTING,
                _SKIPPED_NOT_APPLICABLE,
                metadata={"analysis_mode": request.analysis_mode.value},
            )
        else:
            if request.target_column is None:
                self._refuse(
                    state,
                    AnalysisWorkflowStage.TASK_ROUTING,
                    "Target column is required for supervised task routing.",
                )
            task_result = self._task_router.route(
                sorted_profile,
                role_mapping,
                confirmed_target=request.target_column,
                confirmed_time_column=request.timestamp_column,
            )
            inferred_task = task_result.selected_task
            state.inferred_task = inferred_task
            routing_warnings = list(task_result.warnings)

            if request.requested_task is None:
                selected_task = inferred_task
                task_selection_source = TaskSelectionSource.ROUTER
                task_override_applied = False
            else:
                selected_task = request.requested_task
                task_selection_source = TaskSelectionSource.USER_OVERRIDE
                task_override_applied = True
                if selected_task is not inferred_task:
                    routing_warnings.append(
                        "The task router inferred "
                        f"{inferred_task.value}, while the user explicitly selected "
                        f"{selected_task.value}. The explicit selection was used."
                    )

            state.selected_task = selected_task
            state.task_selection_source = task_selection_source
            state.task_override_applied = task_override_applied

            task_routing_metadata: dict[str, ScalarMetadataValue] = {
                "inferred_task": inferred_task.value,
                "selected_task": selected_task.value,
                "task_selection_source": task_selection_source.value,
                "task_override_applied": task_override_applied,
                "selected_target": task_result.selected_target,
            }

            if selected_task is AnalysisTask.REGRESSION and (
                request.target_column in sorted_frame.columns
            ):
                regression_assessment = evaluate_target_suitability(
                    sorted_frame,
                    request.target_column,
                    requested_task=AnalysisTask.REGRESSION,
                )
                self._store_target_suitability(state, regression_assessment)
                if not regression_assessment.suitable:
                    task_routing_metadata["target_suitable"] = False
                    task_routing_metadata["target_unique_non_null_count"] = (
                        regression_assessment.unique_non_null_count
                    )
                    task_routing_metadata["target_refusal_code"] = (
                        None
                        if regression_assessment.refusal_code is None
                        else regression_assessment.refusal_code.value
                    )
                    self._refuse(
                        state,
                        AnalysisWorkflowStage.TASK_ROUTING,
                        regression_assessment.message,
                        warnings=routing_warnings,
                        metadata=task_routing_metadata,
                    )

            if selected_task is AnalysisTask.CLASSIFICATION:
                self._refuse(
                    state,
                    AnalysisWorkflowStage.TASK_ROUTING,
                    "Classification modeling is not yet supported by this workflow. "
                    "The analysis was refused because CLASSIFICATION was selected.",
                    warnings=routing_warnings,
                    metadata=task_routing_metadata,
                )

            if (
                policy.require_regression_task
                and selected_task is not AnalysisTask.REGRESSION
            ):
                self._refuse(
                    state,
                    AnalysisWorkflowStage.TASK_ROUTING,
                    "Policy requires a regression task, but routing selected "
                    f"'{selected_task.value}'. Classification is not supported by "
                    "this workflow.",
                    warnings=routing_warnings,
                    metadata=task_routing_metadata,
                )
            self._ok(
                state,
                AnalysisWorkflowStage.TASK_ROUTING,
                f"Routed the analysis to the '{selected_task.value}' task.",
                warnings=routing_warnings,
                metadata=task_routing_metadata,
            )

        # --- ROLE_MAPPING ---
        self._ok(
            state,
            AnalysisWorkflowStage.ROLE_MAPPING,
            "Mapped column roles; explicit request feature columns are used for "
            "modeling.",
            warnings=list(role_mapping.warnings),
            metadata={"feature_count": len(feature_columns)},
        )

        # --- SPLIT (and deferred training-only preprocessing) ---
        if request.timestamp_column is not None:
            split_config = SplitConfig(
                strategy=SplitStrategy.TIME,
                time_column=request.timestamp_column,
                test_size=0.20,
                validation_size=0.20,
                random_state=42,
            )
        else:
            split_config = SplitConfig(
                strategy=SplitStrategy.RANDOM,
                allow_random_split=True,
                test_size=0.20,
                validation_size=0.20,
                random_state=42,
            )
        raw_split = self._splitter.split(sorted_frame, split_config)

        preprocessor.fit(raw_split.train)
        train_pp = preprocessor.transform(raw_split.train)
        validation_pp = preprocessor.transform(raw_split.validation)
        test_pp = preprocessor.transform(raw_split.test)
        split = DatasetSplit(
            train=train_pp.frame,
            validation=validation_pp.frame,
            test=test_pp.frame,
            summary=raw_split.summary,
        )
        preprocessing_events = tuple(train_pp.events)

        train_ids = set(split.summary.train_original_row_ids)
        validation_ids = set(split.summary.validation_original_row_ids)
        test_ids = set(split.summary.test_original_row_ids)
        if (
            split.train.height < 1
            or split.validation.height < 1
            or split.test.height < 1
        ):
            self._refuse(
                state,
                AnalysisWorkflowStage.SPLIT,
                "One or more split partitions were empty; the workflow cannot "
                "continue safely.",
            )
        if (
            train_ids & validation_ids
            or train_ids & test_ids
            or validation_ids & test_ids
        ):
            self._refuse(
                state,
                AnalysisWorkflowStage.SPLIT,
                "Split partitions shared original row identifiers; refusing to "
                "continue to avoid leakage.",
            )
        state.train_row_count = split.train.height
        state.validation_row_count = split.validation.height
        state.test_row_count = split.test.height
        self._ok(
            state,
            AnalysisWorkflowStage.SPLIT,
            "Created disjoint train/validation/test partitions and fit "
            "preprocessing on the training partition only.",
            metadata={
                "strategy": split.summary.strategy.value,
                "train_row_count": split.train.height,
                "validation_row_count": split.validation.height,
                "test_row_count": split.test.height,
            },
        )

        # --- LEAKAGE_CHECK ---
        leakage_report = self._leakage_checker.check(
            split,
            role_mapping,
            target_column=request.target_column,
            feature_columns=feature_columns,
            preprocessing_events=preprocessing_events,
        )
        if policy.stop_on_leakage_blocker and not leakage_report.is_safe:
            self._refuse(
                state,
                AnalysisWorkflowStage.LEAKAGE_CHECK,
                "Leakage checking reported blocking issues; the workflow stopped "
                "before modeling.",
                metadata={"blocker_count": int(leakage_report.blocker_count)},
            )
        self._ok(
            state,
            AnalysisWorkflowStage.LEAKAGE_CHECK,
            "Verified that the split, roles, and preprocessing are leakage-safe.",
            metadata={
                "is_safe": bool(leakage_report.is_safe),
                "blocker_count": int(leakage_report.blocker_count),
            },
        )

        supervised_final: FinalEvaluationOutcome | None = None
        residual_final: ResidualAnomalyFinalEvaluationOutcome | None = None

        # --- SUPERVISED_SCREENING / SUPERVISED_FINAL_EVALUATION ---
        if anomaly_only:
            self._skip(
                state,
                AnalysisWorkflowStage.SUPERVISED_SCREENING,
                _SKIPPED_NOT_APPLICABLE,
            )
            self._skip(
                state,
                AnalysisWorkflowStage.SUPERVISED_FINAL_EVALUATION,
                _SKIPPED_NOT_APPLICABLE,
            )
        else:
            if request.target_column is None:
                self._refuse(
                    state,
                    AnalysisWorkflowStage.SUPERVISED_SCREENING,
                    "Target column is required for supervised screening.",
                )
            if request.model_performance_policy is None:
                self._refuse(
                    state,
                    AnalysisWorkflowStage.SUPERVISED_FINAL_EVALUATION,
                    "Model performance policy is required for supervised analysis.",
                )
            supervised_screening = SupervisedModelScreener(
                self._supervised_registry
            ).screen(
                split,
                task=AnalysisTask.REGRESSION,
                target_column=request.target_column,
                feature_columns=feature_columns,
                leakage_report=leakage_report,
                industry_profile=industry_profile,
            )
            state.selected_supervised_model_key = (
                supervised_screening.summary.selected_estimator_key
            )
            self._ok(
                state,
                AnalysisWorkflowStage.SUPERVISED_SCREENING,
                "Screened supervised regression candidates on train/validation.",
                metadata={
                    "selected_model_name": (
                        supervised_screening.summary.selected_model_name
                    ),
                    "selected_estimator_key": (
                        supervised_screening.summary.selected_estimator_key
                    ),
                },
            )

            supervised_final = FinalModelEvaluator(self._supervised_registry).evaluate(
                split,
                supervised_screening,
                task=AnalysisTask.REGRESSION,
                target_column=request.target_column,
                feature_columns=feature_columns,
                leakage_report=leakage_report,
            )
            state.supervised_model_available = True
            state.independent_test_evaluation_performed = True
            performance_outcome = ModelPerformanceAssessor(
                policy=request.model_performance_policy,
            ).assess(supervised_final)
            assessment = performance_outcome.report
            state.model_performance_assessment = assessment
            assessment_warnings = list(assessment.warnings)
            self._ok(
                state,
                AnalysisWorkflowStage.SUPERVISED_FINAL_EVALUATION,
                "Refit the selected regression model on train+validation and "
                "evaluated once on the independent test partition.",
                warnings=assessment_warnings,
                metadata={
                    "refit_on_train_validation": bool(
                        supervised_final.report.refit_on_train_validation
                    ),
                    "performance_degraded": bool(
                        supervised_final.report.performance_degraded
                    ),
                    "model_performance_status": assessment.status.value,
                    "model_performance_rule_count": len(assessment.metric_results),
                    "model_performance_required_rule_count": (
                        assessment.required_rule_count
                    ),
                    "model_performance_failed_rule_count": (
                        assessment.failed_required_rule_count
                    ),
                    "model_performance_unavailable_rule_count": (
                        assessment.unavailable_required_rule_count
                    ),
                },
            )

        # --- ANOMALY_SCREENING ---
        anomaly_screening = UnsupervisedAnomalyModelScreener(
            self._anomaly_registry
        ).screen(
            split,
            feature_columns=feature_columns,
            leakage_report=leakage_report,
            industry_profile=industry_profile,
        )
        state.selected_anomaly_model_key = (
            anomaly_screening.summary.selected_estimator_key
        )
        self._ok(
            state,
            AnalysisWorkflowStage.ANOMALY_SCREENING,
            "Screened unsupervised anomaly detectors on train/validation.",
            metadata={
                "selected_model_name": (
                    anomaly_screening.summary.selected_model_name
                ),
                "selected_estimator_key": (
                    anomaly_screening.summary.selected_estimator_key
                ),
            },
        )

        # --- ANOMALY_FINAL_EVALUATION ---
        anomaly_final = AnomalyFinalEvaluator(self._anomaly_registry).evaluate(
            split,
            anomaly_screening,
            feature_columns=feature_columns,
            leakage_report=leakage_report,
        )
        state.anomaly_model_available = True
        if anomaly_only:
            state.independent_test_evaluation_performed = True
        self._ok(
            state,
            AnalysisWorkflowStage.ANOMALY_FINAL_EVALUATION,
            "Evaluated the selected anomaly detector on the independent test "
            "partition.",
            row_count=anomaly_final.test_scored.height,
        )

        # --- RESIDUAL_CALIBRATION / RESIDUAL_FINAL_EVALUATION ---
        if anomaly_only:
            self._skip(
                state,
                AnalysisWorkflowStage.RESIDUAL_CALIBRATION,
                _SKIPPED_NOT_APPLICABLE,
            )
            self._skip(
                state,
                AnalysisWorkflowStage.RESIDUAL_FINAL_EVALUATION,
                _SKIPPED_NOT_APPLICABLE,
            )
        else:
            if request.target_column is None or supervised_final is None:
                self._refuse(
                    state,
                    AnalysisWorkflowStage.RESIDUAL_CALIBRATION,
                    "Supervised outputs are required for residual calibration.",
                )
            residual_pipeline_outcome = self._residual_pipeline.run(
                split,
                supervised_screening,
                target_column=request.target_column,
                feature_columns=feature_columns,
                leakage_report=leakage_report,
            )
            state.residual_calibration_performed = True
            self._ok(
                state,
                AnalysisWorkflowStage.RESIDUAL_CALIBRATION,
                "Calibrated the residual anomaly detector using the screening "
                "regression model.",
            )

            residual_final = self._residual_final_evaluator.evaluate(
                split,
                residual_pipeline_outcome,
                target_column=request.target_column,
                feature_columns=feature_columns,
                leakage_report=leakage_report,
            )
            self._ok(
                state,
                AnalysisWorkflowStage.RESIDUAL_FINAL_EVALUATION,
                "Evaluated the calibrated residual anomaly detector on the "
                "independent test partition.",
                row_count=residual_final.test_scored.height,
            )

        # --- ANOMALY_EVENT_SELECTION ---
        if anomaly_only:
            anomaly_events, event_warnings = _select_unsupervised_anomaly_events(
                anomaly_test=anomaly_final.test_scored,
                feature_columns=feature_columns,
                maximum_events=policy.maximum_anomaly_events,
                minimum_events=policy.minimum_anomaly_events,
            )
            state.anomaly_event_selection_source = _SELECTION_SOURCE_UNSUPERVISED
            event_message = (
                "Selected anomaly events for diagnosis using unsupervised "
                "anomaly scores."
            )
            event_metadata: dict[str, ScalarMetadataValue] = {
                "anomaly_event_count": len(anomaly_events),
                "selection_source": _SELECTION_SOURCE_UNSUPERVISED,
            }
        else:
            assert residual_final is not None
            anomaly_events, event_warnings = _select_anomaly_events(
                residual_test=residual_final.test_scored,
                anomaly_test=anomaly_final.test_scored,
                feature_columns=feature_columns,
                maximum_events=policy.maximum_anomaly_events,
                minimum_events=policy.minimum_anomaly_events,
            )
            event_message = (
                "Selected anomaly events for diagnosis using residual and "
                "unsupervised indicators."
            )
            event_metadata = {
                "anomaly_event_count": len(anomaly_events),
            }
        state.anomaly_event_count = len(anomaly_events)
        state.anomaly_events = [
            event.model_copy(
                update={"contributing_variables": []},
                deep=True,
            )
            for event in anomaly_events
        ]
        if len(anomaly_events) < policy.minimum_anomaly_events:
            self._refuse(
                state,
                AnalysisWorkflowStage.ANOMALY_EVENT_SELECTION,
                "Fewer anomaly events were available than the configured minimum; "
                "the workflow stopped before diagnosis.",
                metadata={
                    "anomaly_event_count": len(anomaly_events),
                    "minimum_anomaly_events": policy.minimum_anomaly_events,
                },
            )
        self._ok(
            state,
            AnalysisWorkflowStage.ANOMALY_EVENT_SELECTION,
            event_message,
            warnings=event_warnings,
            metadata=event_metadata,
        )

        # --- DIAGNOSIS ---
        if anomaly_only:
            diagnosis_frame, diagnosis_warnings = _build_unsupervised_diagnosis_frame(
                anomaly_final.test_scored
            )
            diagnoser = RobustGroupComparisonDiagnoser()
            diagnosis_request = DiagnosisRequest(
                task=AnalysisTask.UNSUPERVISED_ANOMALY,
                method=DiagnosisMethod.GROUP_COMPARISON,
                scope=DiagnosisScope.ANOMALY_GROUP,
                feature_columns=list(feature_columns),
                anomaly_events=[],
                anomaly_indicator_column=_ANOMALY_INDICATOR_COLUMN,
                anomaly_score_column=_ANOMALY_SCORE_COLUMN,
                row_id_column=ORIGINAL_ROW_ID_COLUMN,
                minimum_reference_rows=5,
                metadata={
                    "stage": AnalysisWorkflowStage.DIAGNOSIS.value,
                    "diagnosis_source": _DIAGNOSIS_SOURCE_ROBUST,
                    "association_not_causation": True,
                    "target_based_diagnosis": False,
                },
            )
            diagnosis_result = diagnoser.diagnose(
                diagnosis_frame,
                request=diagnosis_request,
            )
            if not isinstance(diagnosis_result, DiagnosisResult):
                self._refuse(
                    state,
                    AnalysisWorkflowStage.DIAGNOSIS,
                    "Diagnosis returned an unexpected batch result for a single "
                    "anomaly group.",
                )
            state.diagnosis_source = _DIAGNOSIS_SOURCE_ROBUST
            state.diagnosis_performed = True
            state.diagnosis_factor_count = len(diagnosis_result.factors)
            state.diagnosis_factors = [
                factor.model_copy(deep=True) for factor in diagnosis_result.factors
            ]
            diagnosis_warnings = list(diagnosis_warnings)
            diagnosis_warnings.append(
                "Diagnosis compares anomaly and normal group distributions "
                "(association, not causation; not target-based)."
            )
            self._ok(
                state,
                AnalysisWorkflowStage.DIAGNOSIS,
                "Diagnosed likely associated factors using robust "
                "normal-vs-anomaly group comparison (association, not causation).",
                warnings=_dedupe(diagnosis_warnings),
                metadata={
                    "diagnosis_factor_count": len(diagnosis_result.factors),
                    "diagnosis_source": _DIAGNOSIS_SOURCE_ROBUST,
                    "target_based_diagnosis": False,
                    "association_not_causation": True,
                },
            )
        else:
            assert residual_final is not None
            diagnosis_frame, diagnosis_warnings = _build_diagnosis_frame(
                residual_final.test_scored
            )
            ensemble = self._resolve_diagnoser(policy)
            diagnosis_request = DiagnosisRequest(
                task=AnalysisTask.RESIDUAL_ANOMALY,
                method=DiagnosisMethod.ENSEMBLE,
                scope=DiagnosisScope.ANOMALY_GROUP,
                feature_columns=list(feature_columns),
                anomaly_events=[],
                anomaly_indicator_column=_RESIDUAL_INDICATOR_COLUMN,
                anomaly_score_column=_RESIDUAL_SCORE_COLUMN,
                row_id_column=ORIGINAL_ROW_ID_COLUMN,
                minimum_reference_rows=5,
                metadata={"stage": AnalysisWorkflowStage.DIAGNOSIS.value},
            )
            diagnosis_result = ensemble.diagnose(
                diagnosis_frame,
                request=diagnosis_request,
            )
            if not isinstance(diagnosis_result, DiagnosisResult):
                self._refuse(
                    state,
                    AnalysisWorkflowStage.DIAGNOSIS,
                    "Diagnosis returned an unexpected batch result for a single "
                    "anomaly group.",
                )
            residual_succeeded = diagnosis_result.metadata.get(
                "residual_method_succeeded"
            )
            partial_ensemble = bool(diagnosis_result.metadata.get("partial_ensemble"))
            if policy.require_residual_diagnosis and residual_succeeded is not True:
                self._refuse(
                    state,
                    AnalysisWorkflowStage.DIAGNOSIS,
                    "Residual diagnosis was required but the residual method did not "
                    "succeed.",
                )
            if partial_ensemble and not policy.allow_partial_diagnosis_ensemble:
                self._refuse(
                    state,
                    AnalysisWorkflowStage.DIAGNOSIS,
                    "The diagnosis ensemble produced only partial method support and "
                    "partial ensembles are not permitted by policy.",
                )
            state.diagnosis_source = "RESIDUAL_ROBUST_ENSEMBLE"
            state.diagnosis_performed = True
            state.diagnosis_factor_count = len(diagnosis_result.factors)
            state.diagnosis_factors = [
                factor.model_copy(deep=True) for factor in diagnosis_result.factors
            ]
            self._ok(
                state,
                AnalysisWorkflowStage.DIAGNOSIS,
                "Diagnosed likely associated factors using the residual/robust "
                "ensemble (association, not causation).",
                warnings=diagnosis_warnings,
                metadata={
                    "diagnosis_factor_count": len(diagnosis_result.factors),
                    "residual_method_succeeded": bool(residual_succeeded),
                    "partial_ensemble": partial_ensemble,
                },
            )

        # --- ANOMALY CONTEXT WINDOWS (interpretation only) ---
        context_windows, context_warnings = build_anomaly_context_windows(
            analysis_frame=sorted_frame,
            anomaly_events=state.anomaly_events,
            diagnosis_factors=state.diagnosis_factors,
            order_basis=context_order_basis,
            timestamp_column=request.timestamp_column,
            identifier_columns=list(request.identifier_columns),
        )
        state.anomaly_context_windows = [
            window.model_copy(deep=True) for window in context_windows
        ]
        state.context_warnings = list(context_warnings)

        # --- RECOMMENDATION ---
        if anomaly_only:
            if not request.anomaly_recommendation.enabled:
                operating_request = request
                if (
                    request.operating_point_selection
                    is OperatingPointSelectionMode.TOP_RESIDUAL_ANOMALY
                ):
                    operating_request = request.model_copy(
                        update={
                            "operating_point_selection": (
                                OperatingPointSelectionMode.TOP_UNSUPERVISED_ANOMALY
                            )
                        }
                    )
                selected_row_id, _baseline, operating_warnings = _select_operating_point(
                    request=operating_request,
                    residual_test=anomaly_final.test_scored,
                    anomaly_test=anomaly_final.test_scored,
                    test_frame=anomaly_final.test_scored,
                    feature_columns=feature_columns,
                )
                state.selected_operating_row_id = selected_row_id
                self._skip(
                    state,
                    AnalysisWorkflowStage.RECOMMENDATION,
                    _ANOMALY_ONLY_RECOMMENDATION_DISABLED,
                    warnings=operating_warnings,
                    metadata={
                        "recommendation_applicable": False,
                        "operating_row_id": _scalar_row_id(selected_row_id),
                        "overview_message": _ANOMALY_ONLY_OVERVIEW,
                    },
                )
                self._skip(
                    state,
                    AnalysisWorkflowStage.WHAT_IF_VERIFICATION,
                    "What-if verification was not applicable because no "
                    "recommendation was generated.",
                    metadata={"recommendation_applicable": False},
                )
                state.status = AnalysisWorkflowStatus.PARTIAL
            else:
                self._run_anomaly_recommendation(
                    request,
                    policy=policy,
                    state=state,
                    feature_columns=feature_columns,
                    industry_profile=industry_profile,
                    leakage_report=leakage_report,
                    anomaly_final=anomaly_final,
                    diagnosis_result=diagnosis_result,
                )
        else:
            assert supervised_final is not None
            assert residual_final is not None
            self._run_recommendation(
                request,
                policy=policy,
                state=state,
                feature_columns=feature_columns,
                industry_profile=industry_profile,
                leakage_report=leakage_report,
                supervised_final=supervised_final,
                anomaly_final=anomaly_final,
                residual_final=residual_final,
                diagnosis_result=diagnosis_result,
            )

    def _run_anomaly_recommendation(
        self,
        request: AnalysisWorkflowRequest,
        *,
        policy: AnalysisWorkflowPolicy,
        state: _RunState,
        feature_columns: list[str],
        industry_profile: BaseIndustryProfile,
        leakage_report: LeakageReport,
        anomaly_final: AnomalyFinalEvaluationOutcome,
        diagnosis_result: DiagnosisResult,
    ) -> None:
        """Run REDUCE_ANOMALY_SCORE recommendation for ANOMALY_ONLY mode.

        Uses the already-fitted selected anomaly model. Does not refit or
        reselect models and does not require supervised performance acceptance.
        """
        del policy  # stage-output retention remains on RecommendationPipelinePolicy

        operating_request = request
        if (
            request.operating_point_selection
            is OperatingPointSelectionMode.TOP_RESIDUAL_ANOMALY
        ):
            operating_request = request.model_copy(
                update={
                    "operating_point_selection": (
                        OperatingPointSelectionMode.TOP_UNSUPERVISED_ANOMALY
                    )
                }
            )
        selected_row_id, baseline, operating_warnings = _select_operating_point(
            request=operating_request,
            residual_test=anomaly_final.test_scored,
            anomaly_test=anomaly_final.test_scored,
            test_frame=anomaly_final.test_scored,
            feature_columns=feature_columns,
        )
        state.selected_operating_row_id = selected_row_id

        fitted_model = anomaly_final.final_model
        if fitted_model is None:
            self._refuse(
                state,
                AnalysisWorkflowStage.RECOMMENDATION,
                "A fitted selected anomaly model is required before anomaly-only "
                "recommendation generation.",
                warnings=operating_warnings,
                metadata={"operating_row_id": _scalar_row_id(selected_row_id)},
            )

        eligible_variables = _select_anomaly_recommendation_candidates(
            request=request,
            feature_columns=feature_columns,
            diagnosis_result=diagnosis_result,
            baseline=baseline,
        )
        association_warnings = list(operating_warnings)
        _append_unique_warning(
            association_warnings,
            _ANOMALY_RECOMMENDATION_ASSOCIATION_NOTE,
        )

        factors_for_request = _apply_role_overrides_to_diagnosis_factors(
            diagnosis_result.factors,
            role_overrides=request.column_role_overrides,
        )
        diagnosis_for_request = DiagnosisResult(
            anomaly_id=diagnosis_result.anomaly_id,
            task=AnalysisTask.UNSUPERVISED_ANOMALY,
            method_used=list(diagnosis_result.method_used),
            scope=diagnosis_result.scope,
            factors=factors_for_request,
            confidence=float(diagnosis_result.confidence),
            analyzed_row_count=diagnosis_result.analyzed_row_count,
            reference_row_count=max(diagnosis_result.reference_row_count, 5),
            caveats=list(diagnosis_result.caveats),
            generated_at=datetime.now(tz=UTC),
            metadata=dict(diagnosis_result.metadata),
        )

        if not eligible_variables:
            safety_context = RecommendationSafetyContext(
                leakage_report=leakage_report,
                final_evaluation_available=True,
                model_performance_acceptable=False,
                model_performance_reason=(
                    "Model performance acceptance: NOT_APPLICABLE for "
                    "anomaly-only recommendation."
                ),
                extrapolation_detected=False,
                uncertainty_available=False,
                uncertainty_acceptable=None,
                metadata={
                    "workflow_stage": AnalysisWorkflowStage.RECOMMENDATION.value,
                    "model_performance_status": "NOT_APPLICABLE",
                    "model_performance_gate_bypassed": True,
                    "anomaly_model_selected_label_free": True,
                    "recommendation_safety_depends_on_user_constraints": True,
                },
            )
            placeholder_values = {
                name: baseline[name]
                for name in feature_columns
                if name in baseline
            }
            if not placeholder_values:
                self._refuse(
                    state,
                    AnalysisWorkflowStage.RECOMMENDATION,
                    "No operating-point feature values were available for "
                    "anomaly-only recommendation eligibility evaluation.",
                    warnings=association_warnings,
                    metadata={"operating_row_id": _scalar_row_id(selected_row_id)},
                )
            recommendation_request = RecommendationRequest(
                task=AnalysisTask.UNSUPERVISED_ANOMALY,
                diagnosis=diagnosis_for_request,
                objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
                current_values=placeholder_values,
                constraints=[],
                user_confirmed_controllable_variables=[],
                user_verified_variables=[],
                max_simultaneous_changes=1,
                metadata={
                    "workflow_stage": AnalysisWorkflowStage.RECOMMENDATION.value,
                    "anomaly_recommendation_enabled": True,
                },
            )
            safety_gate = RecommendationSafetyGate(
                policy=RecommendationSafetyPolicy(
                    require_acceptable_model_performance=False,
                )
            )
            safety_decision = safety_gate.evaluate(
                recommendation_request,
                context=safety_context,
            )
            if safety_decision.status is not RecommendationSafetyStatus.REFUSED:
                reason_codes = list(safety_decision.global_reason_codes)
                if RecommendationReasonCode.NO_ELIGIBLE_VARIABLES not in reason_codes:
                    reason_codes.append(RecommendationReasonCode.NO_ELIGIBLE_VARIABLES)
                safety_decision = RecommendationSafetyDecision(
                    status=RecommendationSafetyStatus.REFUSED,
                    objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
                    eligible_variables=[],
                    blocked_variables=list(safety_decision.blocked_variables),
                    variable_assessments=[
                        item.model_copy(deep=True)
                        for item in safety_decision.variable_assessments
                    ],
                    global_reason_codes=reason_codes,
                    messages=[
                        *list(safety_decision.messages),
                        _ANOMALY_ONLY_NO_ELIGIBLE,
                    ],
                    disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
                    evaluated_at=datetime.now(tz=UTC),
                    metadata=dict(safety_decision.metadata),
                )
            final_result = RecommendationResult(
                status=RecommendationStatus.REFUSED,
                objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
                safety_decision=safety_decision,
                changes=[],
                baseline_prediction=None,
                proposed_prediction=None,
                baseline_anomaly_score=None,
                proposed_anomaly_score=None,
                confidence=0.0,
                extrapolation_flag=False,
                uncertainty_available=False,
                disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
                generated_at=datetime.now(tz=UTC),
                warnings=list(association_warnings),
                metadata={
                    "recommendation_generated": False,
                    "model_refit_performed": False,
                    "anomaly_recommendation_enabled": True,
                    "eligible_candidate_count": 0,
                },
            )
            state.final_recommendation = final_result
            state.recommendation_pipeline_executed = False
            state.status = AnalysisWorkflowStatus.REFUSED
            self._record(
                state,
                AnalysisWorkflowStage.RECOMMENDATION,
                succeeded=False,
                structured_refusal=True,
                message=_ANOMALY_ONLY_NO_ELIGIBLE,
                warnings=association_warnings,
                metadata={
                    "recommendation_status": final_result.status.value,
                    "operating_row_id": _scalar_row_id(selected_row_id),
                    "eligible_candidate_count": 0,
                    "model_performance_status": "NOT_APPLICABLE",
                },
            )
            self._skip_what_if_verification(
                state,
                reason=(
                    "What-if verification was not applicable because "
                    "recommendation was refused."
                ),
            )
            return

        current_values = {name: baseline[name] for name in eligible_variables}
        recommendation_max_changes = min(
            request.max_simultaneous_changes,
            len(current_values),
        )
        constraints_for_request = [
            constraint.model_copy(deep=True)
            for constraint in request.request_constraints
            if constraint.variable in current_values
        ]
        confirmed_controllable = [
            name
            for name in request.user_confirmed_controllable_variables
            if name in current_values
        ]
        verified_variables = [
            name
            for name in request.user_verified_variables
            if name in current_values
        ]
        recommendation_request = RecommendationRequest(
            task=AnalysisTask.UNSUPERVISED_ANOMALY,
            diagnosis=diagnosis_for_request,
            objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
            current_values=current_values,
            constraints=constraints_for_request,
            user_confirmed_controllable_variables=confirmed_controllable,
            user_verified_variables=verified_variables,
            max_simultaneous_changes=recommendation_max_changes,
            metadata={
                "workflow_stage": AnalysisWorkflowStage.RECOMMENDATION.value,
                "anomaly_recommendation_enabled": True,
            },
        )
        safety_context = RecommendationSafetyContext(
            leakage_report=leakage_report,
            final_evaluation_available=True,
            model_performance_acceptable=False,
            model_performance_reason=(
                "Model performance acceptance: NOT_APPLICABLE for "
                "anomaly-only recommendation. Anomaly model was selected "
                "through label-free validation screening. Recommendation "
                "safety depends on explicit user constraints and "
                "model-domain checks."
            ),
            extrapolation_detected=False,
            uncertainty_available=False,
            uncertainty_acceptable=None,
            metadata={
                "workflow_stage": AnalysisWorkflowStage.RECOMMENDATION.value,
                "model_performance_status": "NOT_APPLICABLE",
                "model_performance_gate_bypassed": True,
                "anomaly_model_selected_label_free": True,
                "recommendation_safety_depends_on_user_constraints": True,
            },
        )
        industry_constraints = _resolve_industry_constraints(
            request=request,
            industry_profile=industry_profile,
            feature_columns=feature_columns,
        )
        user_overrides = [
            constraint.model_copy(deep=True) for constraint in request.user_overrides
        ]
        scorer = CandidateScenarioScorer(
            quality_model=None,
            anomaly_model=fitted_model,
        )
        safety_gate = RecommendationSafetyGate(
            policy=RecommendationSafetyPolicy(
                require_acceptable_model_performance=False,
            )
        )
        pipeline = RecommendationPipeline(
            scenario_scorer=scorer,
            safety_gate=safety_gate,
        )
        pipeline_request = RecommendationPipelineRequest(
            recommendation_request=recommendation_request,
            safety_context=safety_context,
            industry_constraints=industry_constraints,
            user_overrides=user_overrides,
            feature_columns=list(feature_columns),
            baseline_features=dict(baseline),
            target_column=None,
            quality_direction=None,
            quality_target=None,
            extrapolation_evaluated=False,
            extrapolation_flag=False,
            uncertainty_available=False,
            uncertainty_acceptable=None,
            metadata={
                "workflow_stage": AnalysisWorkflowStage.RECOMMENDATION.value,
                "anomaly_recommendation_enabled": True,
            },
        )
        outcome = pipeline.run(pipeline_request)
        final_result = outcome.report.final_result
        state.recommendation_pipeline_executed = True
        state.final_recommendation = final_result

        pipeline_warnings = list(association_warnings)
        pipeline_warnings.extend(outcome.report.warnings)

        if final_result.status is RecommendationStatus.GENERATED:
            state.status = AnalysisWorkflowStatus.COMPLETED
            state.recommendation_generated = True
            self._ok(
                state,
                AnalysisWorkflowStage.RECOMMENDATION,
                "Generated anomaly-score reduction recommendations with the "
                "fitted anomaly model.",
                warnings=pipeline_warnings,
                metadata={
                    "recommendation_status": final_result.status.value,
                    "operating_row_id": _scalar_row_id(selected_row_id),
                    "change_count": len(final_result.changes),
                    "eligible_candidate_count": len(eligible_variables),
                    "model_performance_status": "NOT_APPLICABLE",
                    "objective": RecommendationObjective.REDUCE_ANOMALY_SCORE.value,
                },
            )
            self._run_what_if_verification(
                state,
                recommendation=final_result,
                grid_report=outcome.report.grid_report,
                scenario_scorer=scorer,
                feature_columns=list(feature_columns),
                baseline_features=dict(baseline),
                quality_direction=None,
                quality_target=None,
                target_column=None,
            )
        elif final_result.status is RecommendationStatus.READY_FOR_OPTIMIZATION:
            state.status = AnalysisWorkflowStatus.PARTIAL
            self._ok(
                state,
                AnalysisWorkflowStage.RECOMMENDATION,
                "The anomaly-only recommendation pipeline determined the case "
                "is ready for optimization but did not emit concrete changes.",
                warnings=pipeline_warnings,
                metadata={
                    "recommendation_status": final_result.status.value,
                    "operating_row_id": _scalar_row_id(selected_row_id),
                    "eligible_candidate_count": len(eligible_variables),
                    "model_performance_status": "NOT_APPLICABLE",
                    "objective": RecommendationObjective.REDUCE_ANOMALY_SCORE.value,
                },
            )
            self._skip_what_if_verification(
                state,
                reason=(
                    "What-if verification was not applicable because no "
                    "recommendation was generated."
                ),
            )
        else:
            state.status = AnalysisWorkflowStatus.REFUSED
            self._record(
                state,
                AnalysisWorkflowStage.RECOMMENDATION,
                succeeded=False,
                structured_refusal=True,
                message=(
                    "The anomaly-only recommendation pipeline refused to produce "
                    "changes for safety reasons."
                ),
                warnings=pipeline_warnings,
                metadata={
                    "recommendation_status": final_result.status.value,
                    "operating_row_id": _scalar_row_id(selected_row_id),
                    "eligible_candidate_count": len(eligible_variables),
                    "model_performance_status": "NOT_APPLICABLE",
                    "objective": RecommendationObjective.REDUCE_ANOMALY_SCORE.value,
                },
            )
            self._skip_what_if_verification(
                state,
                reason=(
                    "What-if verification was not applicable because "
                    "recommendation was refused."
                ),
            )

    def _run_recommendation(
        self,
        request: AnalysisWorkflowRequest,
        *,
        policy: AnalysisWorkflowPolicy,
        state: _RunState,
        feature_columns: list[str],
        industry_profile: BaseIndustryProfile,
        leakage_report: LeakageReport,
        supervised_final: FinalEvaluationOutcome,
        anomaly_final: AnomalyFinalEvaluationOutcome,
        residual_final: ResidualAnomalyFinalEvaluationOutcome,
        diagnosis_result: DiagnosisResult,
    ) -> None:
        objective = request.objective
        if objective is None:
            self._refuse(
                state,
                AnalysisWorkflowStage.RECOMMENDATION,
                "Recommendation objective is required for supervised analysis.",
            )
        if objective is RecommendationObjective.REDUCE_ANOMALY_SCORE:
            recommendation_task = AnalysisTask.UNSUPERVISED_ANOMALY
            target_column: str | None = None
            quality_direction = None
        else:
            recommendation_task = AnalysisTask.REGRESSION
            target_column = request.target_column
            quality_direction = request.quality_direction

        selected_row_id, baseline, operating_warnings = _select_operating_point(
            request=request,
            residual_test=residual_final.test_scored,
            anomaly_test=anomaly_final.test_scored,
            test_frame=residual_final.test_scored,
            feature_columns=feature_columns,
        )
        state.selected_operating_row_id = selected_row_id

        constraint_variables = {
            constraint.variable for constraint in request.request_constraints
        }
        confirmed_variables = set(request.user_confirmed_controllable_variables)
        candidate_variables = [
            name
            for name in feature_columns
            if name in constraint_variables or name in confirmed_variables
        ]
        if not candidate_variables:
            self._refuse(
                state,
                AnalysisWorkflowStage.RECOMMENDATION,
                "No operating-point variables were available for recommendation "
                "because the request supplied neither constraints nor confirmed "
                "controllable variables.",
                warnings=operating_warnings,
                metadata={"operating_row_id": _scalar_row_id(selected_row_id)},
            )

        current_values = {name: baseline[name] for name in candidate_variables}
        recommendation_max_changes = min(
            request.max_simultaneous_changes,
            len(current_values),
        )
        constraints_for_request = [
            constraint.model_copy(deep=True)
            for constraint in request.request_constraints
            if constraint.variable in current_values
        ]
        confirmed_controllable = [
            name
            for name in request.user_confirmed_controllable_variables
            if name in current_values
        ]
        verified_variables = [
            name
            for name in request.user_verified_variables
            if name in current_values
        ]

        diagnosis_for_request = DiagnosisResult(
            anomaly_id=diagnosis_result.anomaly_id,
            task=recommendation_task,
            method_used=list(diagnosis_result.method_used),
            scope=diagnosis_result.scope,
            factors=[
                factor.model_copy(deep=True) for factor in diagnosis_result.factors
            ],
            confidence=float(diagnosis_result.confidence),
            analyzed_row_count=diagnosis_result.analyzed_row_count,
            reference_row_count=max(diagnosis_result.reference_row_count, 5),
            caveats=list(diagnosis_result.caveats),
            generated_at=datetime.now(tz=UTC),
            metadata=dict(diagnosis_result.metadata),
        )

        recommendation_request = RecommendationRequest(
            task=recommendation_task,
            diagnosis=diagnosis_for_request,
            objective=objective,
            current_values=current_values,
            constraints=constraints_for_request,
            user_confirmed_controllable_variables=confirmed_controllable,
            user_verified_variables=verified_variables,
            max_simultaneous_changes=recommendation_max_changes,
            metadata={"workflow_stage": AnalysisWorkflowStage.RECOMMENDATION.value},
        )

        assessment = state.model_performance_assessment
        if assessment is None:
            self._refuse(
                state,
                AnalysisWorkflowStage.RECOMMENDATION,
                "Model performance assessment is required before recommendation "
                "but was not produced by supervised final evaluation.",
                warnings=operating_warnings,
            )
        final_evaluation_available = (
            assessment.evaluation_available
            and assessment.independent_test_evaluation
        )
        model_performance_acceptable = (
            assessment.status is ModelPerformanceAcceptanceStatus.ACCEPTABLE
        )
        if model_performance_acceptable:
            model_performance_reason = None
        elif assessment.warnings:
            model_performance_reason = assessment.warnings[0]
        else:
            model_performance_reason = (
                f"Model performance status is {assessment.status.value}."
            )
        safety_context = RecommendationSafetyContext(
            leakage_report=leakage_report,
            final_evaluation_available=final_evaluation_available,
            model_performance_acceptable=model_performance_acceptable,
            model_performance_reason=model_performance_reason,
            extrapolation_detected=False,
            uncertainty_available=False,
            uncertainty_acceptable=None,
            metadata={
                "workflow_stage": AnalysisWorkflowStage.RECOMMENDATION.value,
                "model_performance_status": assessment.status.value,
                "model_performance_gate_bypassed": False,
            },
        )

        industry_constraints = _resolve_industry_constraints(
            request=request,
            industry_profile=industry_profile,
            feature_columns=feature_columns,
        )
        user_overrides = [
            constraint.model_copy(deep=True) for constraint in request.user_overrides
        ]

        if objective is RecommendationObjective.REDUCE_ANOMALY_SCORE:
            quality_model = None
        else:
            quality_model = supervised_final.final_model
        scorer = CandidateScenarioScorer(
            quality_model=quality_model,
            anomaly_model=anomaly_final.final_model,
        )
        safety_gate = RecommendationSafetyGate()
        pipeline = RecommendationPipeline(
            scenario_scorer=scorer,
            safety_gate=safety_gate,
        )

        pipeline_request = RecommendationPipelineRequest(
            recommendation_request=recommendation_request,
            safety_context=safety_context,
            industry_constraints=industry_constraints,
            user_overrides=user_overrides,
            feature_columns=list(feature_columns),
            baseline_features=dict(baseline),
            target_column=target_column,
            quality_direction=quality_direction,
            quality_target=request.quality_target,
            extrapolation_evaluated=False,
            extrapolation_flag=False,
            uncertainty_available=False,
            uncertainty_acceptable=None,
            metadata={"workflow_stage": AnalysisWorkflowStage.RECOMMENDATION.value},
        )

        outcome = pipeline.run(pipeline_request)
        final_result = outcome.report.final_result
        state.recommendation_pipeline_executed = True
        state.final_recommendation = final_result

        pipeline_warnings = list(operating_warnings)
        pipeline_warnings.extend(outcome.report.warnings)

        if final_result.status is RecommendationStatus.GENERATED:
            state.status = AnalysisWorkflowStatus.COMPLETED
            state.recommendation_generated = True
            self._ok(
                state,
                AnalysisWorkflowStage.RECOMMENDATION,
                "Generated recommended process changes with the recommendation "
                "pipeline.",
                warnings=pipeline_warnings,
                metadata={
                    "recommendation_status": final_result.status.value,
                    "operating_row_id": _scalar_row_id(selected_row_id),
                    "change_count": len(final_result.changes),
                },
            )
            self._run_what_if_verification(
                state,
                recommendation=final_result,
                grid_report=outcome.report.grid_report,
                scenario_scorer=scorer,
                feature_columns=list(feature_columns),
                baseline_features=dict(baseline),
                quality_direction=quality_direction,
                quality_target=request.quality_target,
                target_column=target_column,
            )
        elif final_result.status is RecommendationStatus.READY_FOR_OPTIMIZATION:
            state.status = AnalysisWorkflowStatus.PARTIAL
            self._ok(
                state,
                AnalysisWorkflowStage.RECOMMENDATION,
                "The recommendation pipeline determined the case is ready for "
                "optimization but did not emit concrete changes.",
                warnings=pipeline_warnings,
                metadata={
                    "recommendation_status": final_result.status.value,
                    "operating_row_id": _scalar_row_id(selected_row_id),
                },
            )
            self._skip_what_if_verification(
                state,
                reason=(
                    "What-if verification was not applicable because no "
                    "recommendation was generated."
                ),
            )
        else:
            state.status = AnalysisWorkflowStatus.REFUSED
            self._record(
                state,
                AnalysisWorkflowStage.RECOMMENDATION,
                succeeded=False,
                structured_refusal=True,
                message=(
                    "The recommendation pipeline refused to produce changes for "
                    "safety reasons."
                ),
                warnings=pipeline_warnings,
                metadata={
                    "recommendation_status": final_result.status.value,
                    "operating_row_id": _scalar_row_id(selected_row_id),
                },
            )
            self._skip_what_if_verification(
                state,
                reason=(
                    "What-if verification was not applicable because "
                    "recommendation was refused."
                ),
            )

    def _skip_what_if_verification(
        self,
        state: _RunState,
        *,
        reason: str,
    ) -> None:
        verifier = RecommendationWhatIfVerifier(
            scenario_scorer=CandidateScenarioScorer(),
        )
        objective = (
            state.final_recommendation.objective
            if state.final_recommendation is not None
            else RecommendationObjective.REDUCE_ANOMALY_SCORE
        )
        outcome = verifier.not_applicable(objective=objective, reason=reason)
        state.recommendation_verification = outcome.result
        self._skip(
            state,
            AnalysisWorkflowStage.WHAT_IF_VERIFICATION,
            reason,
            metadata={
                "verification_status": outcome.result.status.value,
                "stability_classification": (
                    outcome.result.stability_classification.value
                ),
            },
        )

    def _run_what_if_verification(
        self,
        state: _RunState,
        *,
        recommendation: RecommendationResult,
        grid_report: CandidateGridReport | None,
        scenario_scorer: CandidateScenarioScorer,
        feature_columns: list[str],
        baseline_features: dict[str, float],
        quality_direction: QualityOptimizationDirection | None,
        quality_target: float | None,
        target_column: str | None,
    ) -> None:
        verifier = RecommendationWhatIfVerifier(scenario_scorer=scenario_scorer)
        try:
            if grid_report is None:
                outcome = verifier.unavailable(
                    objective=recommendation.objective,
                    reason=(
                        "Candidate constraint grid is unavailable for local "
                        "what-if verification."
                    ),
                )
            else:
                outcome = verifier.verify(
                    recommendation=recommendation,
                    grid_report=grid_report,
                    feature_columns=feature_columns,
                    baseline_features=baseline_features,
                    quality_direction=quality_direction,
                    quality_target=quality_target,
                    target_column=target_column,
                    extrapolation_flag=recommendation.extrapolation_flag,
                )
        except (
            DataValidationError,
            ProcessIntelligenceError,
            TypeError,
            ValueError,
            ArithmeticError,
        ) as exc:
            outcome = verifier.unavailable(
                objective=recommendation.objective,
                reason=(
                    "What-if verification raised a structured error "
                    f"({type(exc).__name__}): {exc}"
                ),
            )

        result = outcome.result
        state.recommendation_verification = result

        # Append MIXED/ISOLATED warnings onto the recommendation without mutating
        # status, changes, or scores.
        extra_warnings = recommendation_warnings_for_stability(
            result.stability_classification
        )
        if extra_warnings and state.final_recommendation is not None:
            merged = list(state.final_recommendation.warnings)
            for warning in extra_warnings:
                if warning not in merged:
                    merged.append(warning)
            state.final_recommendation = state.final_recommendation.model_copy(
                update={"warnings": merged}
            )

        if result.status is WhatIfVerificationStatus.COMPLETED:
            message = (
                "Completed local what-if verification for the generated "
                "recommendation using adjacent constraint-grid neighbors."
            )
        elif result.status is WhatIfVerificationStatus.UNAVAILABLE:
            message = (
                "Local what-if verification was unavailable for the generated "
                "recommendation."
            )
        else:
            message = (
                "What-if verification was not applicable for the current "
                "recommendation result."
            )
        self._ok(
            state,
            AnalysisWorkflowStage.WHAT_IF_VERIFICATION,
            message,
            warnings=list(result.warnings),
            metadata={
                "verification_status": result.status.value,
                "stability_classification": result.stability_classification.value,
                "scenario_count": result.scenario_count,
                "neighbor_scenario_count": result.neighbor_scenario_count,
                "model_refit_performed": False,
            },
        )

    def _resolve_diagnoser(
        self,
        policy: AnalysisWorkflowPolicy,
    ) -> DiagnosisEnsembleDiagnoser:
        if self._diagnosis_ensemble is not None:
            return self._diagnosis_ensemble
        return DiagnosisEnsembleDiagnoser(
            config=DiagnosisEnsembleConfig(
                require_residual_method=policy.require_residual_diagnosis,
                continue_on_method_failure=True,
            )
        )

    # ------------------------------------------------------------------
    # Stage record helpers
    # ------------------------------------------------------------------

    def _store_target_suitability(
        self,
        state: _RunState,
        assessment: TargetSuitabilityAssessment,
    ) -> None:
        state.target_column = assessment.target_column
        state.target_suitable = assessment.suitable
        state.target_unique_non_null_count = assessment.unique_non_null_count
        state.target_refusal_code = (
            None
            if assessment.refusal_code is None
            else assessment.refusal_code.value
        )
        state.target_suitability_message = assessment.message

    def _record(
        self,
        state: _RunState,
        stage: AnalysisWorkflowStage,
        *,
        succeeded: bool,
        structured_refusal: bool,
        message: str,
        row_count: int | None = None,
        warnings: list[str] | None = None,
        metadata: dict[str, ScalarMetadataValue] | None = None,
    ) -> None:
        state.records.append(
            AnalysisWorkflowStageRecord(
                stage=stage,
                executed=True,
                succeeded=succeeded,
                structured_refusal=structured_refusal,
                row_count=row_count,
                message=message,
                warnings=_dedupe(warnings or []),
                metadata=metadata or {},
            )
        )

    def _skip(
        self,
        state: _RunState,
        stage: AnalysisWorkflowStage,
        message: str,
        *,
        warnings: list[str] | None = None,
        metadata: dict[str, ScalarMetadataValue] | None = None,
    ) -> None:
        """Record a stage as explicitly skipped / not applicable."""
        state.records.append(
            AnalysisWorkflowStageRecord(
                stage=stage,
                executed=False,
                succeeded=False,
                structured_refusal=False,
                row_count=None,
                message=message,
                warnings=_dedupe(warnings or []),
                metadata=metadata or {},
            )
        )

    def _ok(
        self,
        state: _RunState,
        stage: AnalysisWorkflowStage,
        message: str,
        *,
        row_count: int | None = None,
        warnings: list[str] | None = None,
        metadata: dict[str, ScalarMetadataValue] | None = None,
    ) -> None:
        self._record(
            state,
            stage,
            succeeded=True,
            structured_refusal=False,
            message=message,
            row_count=row_count,
            warnings=warnings,
            metadata=metadata,
        )

    def _refuse(
        self,
        state: _RunState,
        stage: AnalysisWorkflowStage,
        message: str,
        *,
        warnings: list[str] | None = None,
        metadata: dict[str, ScalarMetadataValue] | None = None,
    ) -> NoReturn:
        self._record(
            state,
            stage,
            succeeded=False,
            structured_refusal=True,
            message=message,
            warnings=warnings,
            metadata=metadata,
        )
        raise _WorkflowHalt(AnalysisWorkflowStatus.REFUSED)

    # ------------------------------------------------------------------
    # Report assembly
    # ------------------------------------------------------------------

    def _build_report(
        self,
        *,
        policy: AnalysisWorkflowPolicy,
        state: _RunState,
        started_at: datetime,
        completed_at: datetime,
        total_seconds: float,
    ) -> AnalysisWorkflowReport:
        records = state.records
        executed = [record for record in records if record.executed]
        terminal_stage = executed[-1].stage if executed else records[-1].stage
        warnings = self._aggregate_warnings(
            policy=policy,
            records=records,
            extra_warnings=state.context_warnings,
        )
        report_metadata = _build_report_metadata(state)

        return AnalysisWorkflowReport(
            status=state.status,
            terminal_stage=terminal_stage,
            stage_records=records,
            analysis_mode=state.analysis_mode,
            model_performance_assessment=state.model_performance_assessment,
            final_recommendation=state.final_recommendation,
            recommendation_verification=state.recommendation_verification,
            selected_industry=state.selected_industry,
            selected_task=state.selected_task,
            inferred_task=state.inferred_task,
            task_selection_source=state.task_selection_source,
            task_override_applied=state.task_override_applied,
            selected_supervised_model_key=state.selected_supervised_model_key,
            selected_anomaly_model_key=state.selected_anomaly_model_key,
            selected_operating_row_id=state.selected_operating_row_id,
            anomaly_event_count=state.anomaly_event_count,
            diagnosis_factor_count=state.diagnosis_factor_count,
            anomaly_events=[
                event.model_copy(deep=True) for event in state.anomaly_events
            ],
            diagnosis_factors=[
                factor.model_copy(deep=True) for factor in state.diagnosis_factors
            ],
            anomaly_context_windows=[
                window.model_copy(deep=True)
                for window in state.anomaly_context_windows
            ],
            raw_row_count=state.raw_row_count,
            processed_row_count=state.processed_row_count,
            cohort_row_count=state.cohort_row_count,
            train_row_count=state.train_row_count,
            validation_row_count=state.validation_row_count,
            test_row_count=state.test_row_count,
            cohort_filter_summary=(
                state.cohort_filter_summary.model_copy(deep=True)
                if state.cohort_filter_summary is not None
                else CohortFilterSummary(
                    configured=False,
                    source_row_count=state.processed_row_count,
                    retained_row_count=state.cohort_row_count,
                    excluded_row_count=0,
                    null_excluded_count=0,
                )
            ),
            dataset_fingerprint=state.dataset_fingerprint,
            started_at=started_at,
            completed_at=completed_at,
            total_seconds=total_seconds,
            warnings=warnings,
            metadata=report_metadata,
        )

    def _aggregate_warnings(
        self,
        *,
        policy: AnalysisWorkflowPolicy,
        records: list[AnalysisWorkflowStageRecord],
        extra_warnings: list[str] | None = None,
    ) -> list[str]:
        if not policy.include_stage_warnings:
            if not records:
                return []
            terminal = records[-1]
            kept: list[str] = []
            seen_terminal: set[str] = set()
            for message in terminal.warnings:
                if message not in seen_terminal:
                    seen_terminal.add(message)
                    kept.append(message)
            if terminal.structured_refusal and terminal.message not in seen_terminal:
                kept.append(terminal.message)
            if extra_warnings:
                for message in extra_warnings:
                    if message not in seen_terminal:
                        seen_terminal.add(message)
                        kept.append(message)
            return kept

        collected: list[str] = []
        seen_all: set[str] = set()
        for record in records:
            for message in record.warnings:
                if message not in seen_all:
                    seen_all.add(message)
                    collected.append(message)
        if extra_warnings:
            for message in extra_warnings:
                if message not in seen_all:
                    seen_all.add(message)
                    collected.append(message)

        limit = policy.maximum_aggregated_warnings
        if len(collected) <= limit:
            return collected
        trimmed = collected[: max(limit - 1, 0)]
        if limit >= 1 and _WARNING_OMISSION not in trimmed:
            trimmed.append(_WARNING_OMISSION)
        return trimmed[:limit]


_T = TypeVar("_T")


def _resolve_dependency(
    value: _T | None,
    expected_type: type[_T],
    factory: Callable[[], _T],
) -> _T:
    """Return ``value`` when a valid instance, else build a default.

    Args:
        value: Injected dependency or ``None``.
        expected_type: Required type for an injected dependency.
        factory: Zero-argument callable producing a default instance.

    Returns:
        The validated injected instance or a freshly created default.

    Raises:
        TypeError: If ``value`` is not ``None`` and not ``expected_type``.
    """
    if value is None:
        return factory()
    if isinstance(value, expected_type):
        return value
    raise TypeError(
        f"{expected_type.__name__} dependency must be "
        f"{expected_type.__name__} or None, got {type(value).__name__}"
    )


def _dedupe(messages: list[str]) -> list[str]:
    """Return messages in order without duplicates."""
    seen: set[str] = set()
    result: list[str] = []
    for message in messages:
        if message not in seen:
            seen.add(message)
            result.append(message)
    return result


def _format_validation_warning(issue: object) -> str:
    """Return a concrete validation warning from an existing issue payload."""
    message = getattr(issue, "message", None)
    if isinstance(message, str) and message.strip() != "":
        return message
    issue_type = getattr(issue, "issue_type", None)
    if isinstance(issue_type, str) and issue_type.strip() != "":
        return f"Validation issue ({issue_type})"
    return "Validation issue"


def _scalar_row_id(row_id: int | str | None) -> ScalarMetadataValue:
    """Return a scalar-safe representation of an operating row identifier."""
    if row_id is None or isinstance(row_id, (int, str)):
        return row_id
    return str(row_id)


def _select_unsupervised_anomaly_events(
    *,
    anomaly_test: pl.DataFrame,
    feature_columns: list[str],
    maximum_events: int,
    minimum_events: int,
) -> tuple[list[AnomalyEvent], list[str]]:
    """Select anomaly events from unsupervised scores only (anomaly-only mode).

    Uses unsupervised indicator-positive rows first, then high unsupervised
    scores when below the configured minimum. Does not treat missing residual
    indicators as a fallback condition.
    """
    warnings: list[str] = []
    candidates: list[tuple[bool, float, int, str, str]] = []
    seen_ids: set[int] = set()

    unsupervised_hits = anomaly_test.filter(pl.col(_ANOMALY_INDICATOR_COLUMN)).sort(
        [_ANOMALY_SCORE_COLUMN, ORIGINAL_ROW_ID_COLUMN],
        descending=[True, False],
    )
    for row in unsupervised_hits.iter_rows(named=True):
        row_id = int(row[ORIGINAL_ROW_ID_COLUMN])
        if row_id in seen_ids:
            continue
        seen_ids.add(row_id)
        candidates.append(
            (
                True,
                _finite_float(row[_ANOMALY_SCORE_COLUMN]),
                row_id,
                "unsupervised_anomaly_model",
                (
                    "Unsupervised anomaly indicator was True for this row "
                    f"(selection_source={_SELECTION_SOURCE_UNSUPERVISED})."
                ),
            )
        )

    if len(candidates) < minimum_events:
        ordered_scores = anomaly_test.sort(
            [_ANOMALY_SCORE_COLUMN, ORIGINAL_ROW_ID_COLUMN],
            descending=[True, False],
        )
        for row in ordered_scores.iter_rows(named=True):
            if len(candidates) >= maximum_events:
                break
            row_id = int(row[ORIGINAL_ROW_ID_COLUMN])
            if row_id in seen_ids:
                continue
            seen_ids.add(row_id)
            candidates.append(
                (
                    False,
                    _finite_float(row[_ANOMALY_SCORE_COLUMN]),
                    row_id,
                    "top_anomaly_score_candidate",
                    (
                        "Selected as a high unsupervised anomaly-score diagnostic "
                        "candidate; not asserted as a confirmed anomaly."
                    ),
                )
            )
            warnings.append(
                "Anomaly event selected as a high unsupervised anomaly-score "
                "diagnostic candidate; not asserted as a confirmed anomaly."
            )

    candidates.sort(key=lambda item: (not item[0], -item[1], item[2]))
    selected = candidates[:maximum_events]

    events: list[AnomalyEvent] = []
    for _is_flagged, score, row_id, detector, rationale in selected:
        events.append(
            AnomalyEvent(
                anomaly_id=str(row_id),
                anomaly_type=AnomalyType.PROCESS_INPUT,
                anomaly_score=score,
                severity="high" if score >= 0.0 else "moderate",
                sample_id=row_id,
                model_confidence=0.75,
                detector=detector,
                rationale=rationale,
                contributing_variables=list(feature_columns),
            )
        )
    return events, _dedupe(warnings)


def _build_unsupervised_diagnosis_frame(
    anomaly_test: pl.DataFrame,
) -> tuple[pl.DataFrame, list[str]]:
    """Return a diagnosis frame with at least one unsupervised anomaly-group row."""
    warnings: list[str] = []
    positive = int(anomaly_test.get_column(_ANOMALY_INDICATOR_COLUMN).sum())
    if positive > 0:
        return anomaly_test, warnings

    top_id = anomaly_test.sort(_ANOMALY_SCORE_COLUMN, descending=True)[
        ORIGINAL_ROW_ID_COLUMN
    ][0]
    frame = anomaly_test.with_columns(
        pl.when(pl.col(ORIGINAL_ROW_ID_COLUMN) == top_id)
        .then(True)
        .otherwise(False)
        .alias(_ANOMALY_INDICATOR_COLUMN)
    )
    warnings.append(
        "No unsupervised anomaly indicators were present on the test partition; "
        "the highest unsupervised anomaly-score row was marked as the diagnostic "
        "anomaly group."
    )
    return frame, warnings


def _select_anomaly_events(
    *,
    residual_test: pl.DataFrame,
    anomaly_test: pl.DataFrame,
    feature_columns: list[str],
    maximum_events: int,
    minimum_events: int,
) -> tuple[list[AnomalyEvent], list[str]]:
    """Select anomaly events, preferring residual then unsupervised indicators.

    Priority:
        1. residual indicator True rows (by residual score descending)
        2. unsupervised indicator True rows not already selected
        3. top unsupervised-score diagnostic candidates when below minimum

    Ordering places indicator-positive rows first, then higher scores, then
    ascending row identifiers. Duplicate row identifiers are not produced.
    """
    warnings: list[str] = []
    candidates: list[tuple[bool, float, int, str, str, str]] = []
    seen_ids: set[int] = set()

    residual_hits = residual_test.filter(pl.col(_RESIDUAL_INDICATOR_COLUMN)).sort(
        [_RESIDUAL_SCORE_COLUMN, ORIGINAL_ROW_ID_COLUMN],
        descending=[True, False],
    )
    for row in residual_hits.iter_rows(named=True):
        row_id = int(row[ORIGINAL_ROW_ID_COLUMN])
        if row_id in seen_ids:
            continue
        seen_ids.add(row_id)
        candidates.append(
            (
                True,
                _finite_float(row[_RESIDUAL_SCORE_COLUMN]),
                row_id,
                "residual_anomaly_detector",
                "residual_anomaly",
                "Residual or unsupervised anomaly indicator was True for this row.",
            )
        )

    unsupervised_hits = anomaly_test.filter(pl.col(_ANOMALY_INDICATOR_COLUMN)).sort(
        [_ANOMALY_SCORE_COLUMN, ORIGINAL_ROW_ID_COLUMN],
        descending=[True, False],
    )
    for row in unsupervised_hits.iter_rows(named=True):
        row_id = int(row[ORIGINAL_ROW_ID_COLUMN])
        if row_id in seen_ids:
            continue
        seen_ids.add(row_id)
        candidates.append(
            (
                True,
                _finite_float(row[_ANOMALY_SCORE_COLUMN]),
                row_id,
                "unsupervised_anomaly_model",
                "unsupervised_anomaly",
                "Residual or unsupervised anomaly indicator was True for this row.",
            )
        )

    if len(candidates) < minimum_events:
        ordered_scores = anomaly_test.sort(
            [_ANOMALY_SCORE_COLUMN, ORIGINAL_ROW_ID_COLUMN],
            descending=[True, False],
        )
        for row in ordered_scores.iter_rows(named=True):
            if len(candidates) >= maximum_events:
                break
            row_id = int(row[ORIGINAL_ROW_ID_COLUMN])
            if row_id in seen_ids:
                continue
            seen_ids.add(row_id)
            candidates.append(
                (
                    False,
                    _finite_float(row[_ANOMALY_SCORE_COLUMN]),
                    row_id,
                    "top_anomaly_score_candidate",
                    "diagnostic_candidate",
                    (
                        "Selected as a high-score diagnostic candidate; not "
                        "asserted as a confirmed anomaly."
                    ),
                )
            )
            warnings.append(
                "Anomaly event selected as a high-score diagnostic candidate; "
                "not asserted as a confirmed anomaly."
            )

    candidates.sort(key=lambda item: (not item[0], -item[1], item[2]))
    selected = candidates[:maximum_events]

    events: list[AnomalyEvent] = []
    for _is_flagged, score, row_id, detector, _source, rationale in selected:
        events.append(
            AnomalyEvent(
                anomaly_id=str(row_id),
                anomaly_type=AnomalyType.PROCESS_INPUT,
                anomaly_score=score,
                severity="high" if score >= 0.0 else "moderate",
                sample_id=row_id,
                model_confidence=0.75,
                detector=detector,
                rationale=rationale,
                contributing_variables=list(feature_columns),
            )
        )
    return events, _dedupe(warnings)


def _build_diagnosis_frame(
    residual_test: pl.DataFrame,
) -> tuple[pl.DataFrame, list[str]]:
    """Return a diagnosis frame guaranteeing at least one anomaly-group row.

    When no residual anomaly indicators are present, a new frame is returned
    that marks the highest residual-score row as the anomaly group. The input
    frame is never mutated.
    """
    warnings: list[str] = []
    positive = int(residual_test.get_column(_RESIDUAL_INDICATOR_COLUMN).sum())
    if positive > 0:
        return residual_test, warnings

    top_id = residual_test.sort(_RESIDUAL_SCORE_COLUMN, descending=True)[
        ORIGINAL_ROW_ID_COLUMN
    ][0]
    frame = residual_test.with_columns(
        pl.when(pl.col(ORIGINAL_ROW_ID_COLUMN) == top_id)
        .then(True)
        .otherwise(False)
        .alias(_RESIDUAL_INDICATOR_COLUMN)
    )
    warnings.append(
        "No residual anomaly indicators were present on the test partition; the "
        "highest residual-score row was marked as the diagnostic anomaly group."
    )
    return frame, warnings


def _append_unique_warning(warnings: list[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)


def _apply_role_overrides_to_diagnosis_factors(
    factors: list[RootCauseFactor],
    *,
    role_overrides: dict[str, ColumnRole],
) -> list[RootCauseFactor]:
    """Return diagnosis factors with explicit role overrides applied.

    Role override does not auto-confirm controllability or verification.
    """
    updated: list[RootCauseFactor] = []
    for factor in factors:
        override = role_overrides.get(factor.variable)
        if override is None:
            updated.append(factor.model_copy(deep=True))
        else:
            updated.append(factor.model_copy(update={"role": override}, deep=True))
    return updated


def _select_anomaly_recommendation_candidates(
    *,
    request: AnalysisWorkflowRequest,
    feature_columns: list[str],
    diagnosis_result: DiagnosisResult,
    baseline: dict[str, float],
) -> list[str]:
    """Select ANOMALY_ONLY recommendation candidates by explicit intersection.

    Candidates must appear in diagnosis factors and active features, use an
    explicitly controllable process role, be confirmed/verified/constrained,
    and must not be identifier, timestamp, excluded, or cohort-filter columns.
    """
    feature_set = set(feature_columns)
    diagnosis_variables = {factor.variable for factor in diagnosis_result.factors}
    confirmed = set(request.user_confirmed_controllable_variables)
    verified = set(request.user_verified_variables)
    constrained: dict[str, VariableConstraint] = {}
    for constraint in request.request_constraints:
        if (
            constraint.variable in feature_set
            and constraint.minimum is not None
            and constraint.maximum is not None
            and math.isfinite(float(constraint.minimum))
            and math.isfinite(float(constraint.maximum))
        ):
            constrained[constraint.variable] = constraint

    blocked_columns: set[str] = set(request.identifier_columns)
    blocked_columns.update(request.excluded_columns)
    if request.timestamp_column is not None:
        blocked_columns.add(request.timestamp_column)
    if request.cohort_filter is not None:
        # Cohort conditioning columns are never anomaly-only recommendation
        # candidates, regardless of exclude_filter_column_from_features.
        blocked_columns.add(request.cohort_filter.column_name)

    eligible: list[str] = []
    for name in feature_columns:
        if name not in diagnosis_variables:
            continue
        if name not in confirmed or name not in verified:
            continue
        if name not in constrained:
            continue
        if name in blocked_columns:
            continue
        role = request.column_role_overrides.get(name)
        if role not in _CONTROLLABLE_PROCESS_ROLES:
            continue
        if name not in baseline:
            continue
        current_value = baseline[name]
        constraint = constrained[name]
        assert constraint.minimum is not None
        assert constraint.maximum is not None
        if current_value < float(constraint.minimum) or current_value > float(
            constraint.maximum
        ):
            continue
        eligible.append(name)
    return eligible


def _select_operating_point(
    *,
    request: AnalysisWorkflowRequest,
    residual_test: pl.DataFrame,
    anomaly_test: pl.DataFrame,
    test_frame: pl.DataFrame,
    feature_columns: list[str],
) -> tuple[int, dict[str, float], list[str]]:
    """Select the operating-point row identifier and baseline feature values.

    Returns the selected original row identifier, a baseline mapping of every
    feature column to a finite float, and any advisory warnings.
    """
    warnings: list[str] = []
    mode = request.operating_point_selection

    if mode is OperatingPointSelectionMode.EXPLICIT_ROW_ID:
        candidate = _coerce_row_id(request.explicit_operating_row_id)
        if candidate is None:
            raise DataValidationError(
                "explicit_operating_row_id must resolve to a single integer "
                "row identifier"
            )
        matched = test_frame.filter(pl.col(ORIGINAL_ROW_ID_COLUMN) == candidate)
        if matched.height != 1:
            raise DataValidationError(
                "explicit_operating_row_id must match exactly one row in the "
                f"test partition (got {matched.height} matches for "
                f"{candidate!r})"
            )
        row_id = candidate
    elif mode is OperatingPointSelectionMode.TOP_RESIDUAL_ANOMALY:
        residual_hits = residual_test.filter(pl.col(_RESIDUAL_INDICATOR_COLUMN))
        if residual_hits.height > 0:
            row_id = _top_scored_row_id(
                residual_test,
                indicator_column=_RESIDUAL_INDICATOR_COLUMN,
                score_column=_RESIDUAL_SCORE_COLUMN,
            )
        else:
            warnings.append(
                "No residual anomaly indicators were available; falling back to "
                "top unsupervised anomaly operating-point selection."
            )
            row_id = _top_scored_row_id(
                anomaly_test,
                indicator_column=_ANOMALY_INDICATOR_COLUMN,
                score_column=_ANOMALY_SCORE_COLUMN,
            )
    elif mode is OperatingPointSelectionMode.TOP_UNSUPERVISED_ANOMALY:
        unsupervised_hits = anomaly_test.filter(pl.col(_ANOMALY_INDICATOR_COLUMN))
        if unsupervised_hits.height > 0:
            row_id = _top_scored_row_id(
                anomaly_test,
                indicator_column=_ANOMALY_INDICATOR_COLUMN,
                score_column=_ANOMALY_SCORE_COLUMN,
            )
        else:
            warnings.append(
                "No unsupervised anomaly indicators were available; selecting "
                "the highest unsupervised anomaly score as a diagnostic "
                "candidate (not asserted as a confirmed anomaly)."
            )
            row_id = _top_scored_row_id(
                anomaly_test,
                indicator_column=_ANOMALY_INDICATOR_COLUMN,
                score_column=_ANOMALY_SCORE_COLUMN,
            )
    else:  # LATEST_ROW
        if request.timestamp_column is not None and (
            request.timestamp_column in test_frame.columns
        ):
            ordered = test_frame.sort(
                [request.timestamp_column, ORIGINAL_ROW_ID_COLUMN],
                descending=[True, False],
            )
            row_id = int(ordered.get_column(ORIGINAL_ROW_ID_COLUMN)[0])
        else:
            row_id = _latest_row_id(test_frame)

    matched = test_frame.filter(pl.col(ORIGINAL_ROW_ID_COLUMN) == row_id)
    if matched.height != 1:
        raise DataValidationError(
            "Selected operating row must match exactly one test row "
            f"(got {matched.height} matches for {row_id!r})"
        )

    baseline = {
        name: _finite_float(matched.get_column(name)[0]) for name in feature_columns
    }
    return int(row_id), baseline, warnings


def _coerce_row_id(value: int | str | None) -> int | None:
    """Coerce an explicit operating row identifier to an int when possible."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _top_scored_row_id(
    frame: pl.DataFrame,
    *,
    indicator_column: str,
    score_column: str,
) -> int:
    """Return the row identifier of the top-ranked (optionally flagged) row."""
    hits = frame.filter(pl.col(indicator_column))
    ordered = hits if hits.height > 0 else frame
    ordered = ordered.sort(
        [score_column, ORIGINAL_ROW_ID_COLUMN],
        descending=[True, False],
    )
    return int(ordered.get_column(ORIGINAL_ROW_ID_COLUMN)[0])


def _latest_row_id(frame: pl.DataFrame) -> int:
    """Return the row identifier of the last row of a partition."""
    return int(frame.get_column(ORIGINAL_ROW_ID_COLUMN)[-1])


def _resolve_industry_constraints(
    *,
    request: AnalysisWorkflowRequest,
    industry_profile: BaseIndustryProfile,
    feature_columns: list[str],
) -> list[VariableConstraint]:
    """Return industry constraints from the request or the industry profile.

    Uses the request-supplied industry constraints when present; otherwise the
    profile's recommendation constraints filtered to modeling features. Entries
    are deep-copied and de-duplicated by variable name.
    """
    if request.industry_constraints:
        return [
            constraint.model_copy(deep=True)
            for constraint in request.industry_constraints
        ]

    feature_set = set(feature_columns)
    resolved: list[VariableConstraint] = []
    seen: set[str] = set()
    profile_constraints = industry_profile.get_recommendation_constraints()
    for constraint in profile_constraints:
        variable = constraint.variable
        if variable not in feature_set or variable in seen:
            continue
        seen.add(variable)
        resolved.append(constraint.model_copy(deep=True))
    return resolved


def _build_report_metadata(state: _RunState) -> dict[str, ScalarMetadataValue]:
    """Build the scalar-only report metadata dictionary for a run."""
    assessment = state.model_performance_assessment
    metadata: dict[str, ScalarMetadataValue] = {
        "raw_csv_loaded": state.raw_csv_loaded,
        "raw_row_count": state.raw_row_count,
        "processed_row_count": state.processed_row_count,
        "cohort_row_count": state.cohort_row_count,
        "train_row_count": state.train_row_count,
        "validation_row_count": state.validation_row_count,
        "test_row_count": state.test_row_count,
        "analysis_mode": state.analysis_mode.value,
        "selected_industry": state.selected_industry,
        "selected_task": (
            state.selected_task.value if state.selected_task is not None else None
        ),
        "inferred_task": (
            state.inferred_task.value if state.inferred_task is not None else None
        ),
        "task_selection_source": (
            state.task_selection_source.value
            if state.task_selection_source is not None
            else None
        ),
        "task_override_applied": state.task_override_applied,
        "selected_supervised_model_available": state.supervised_model_available,
        "selected_anomaly_model_available": state.anomaly_model_available,
        "residual_calibration_performed": state.residual_calibration_performed,
        "independent_test_evaluation_performed": (
            state.independent_test_evaluation_performed
        ),
        "anomaly_event_count": state.anomaly_event_count,
        "anomaly_event_selection_source": state.anomaly_event_selection_source,
        "diagnosis_performed": state.diagnosis_performed,
        "diagnosis_source": state.diagnosis_source,
        "recommendation_pipeline_executed": state.recommendation_pipeline_executed,
        "recommendation_generated": state.recommendation_generated,
        "recommendation_applicable": state.recommendation_applicable,
        "row_identity_preserved": state.row_identity_preserved,
        "test_used_for_model_selection": False,
        "test_used_for_threshold_calibration": False,
        "anomaly_score_direction": "higher_is_more_anomalous",
        "model_fit_performed": True,
        "model_refit_performed": (
            state.analysis_mode is AnalysisExecutionMode.SUPERVISED
        ),
        "residual_model_refit_after_calibration": False,
        "extrapolation_computed": False,
        "uncertainty_computed": False,
        "workflow_orchestration_only": True,
        "model_performance_gate_bypassed": False,
        "target_column": state.target_column,
        "feature_count": state.feature_count,
        "target_suitable": state.target_suitable,
        "target_unique_non_null_count": state.target_unique_non_null_count,
        "target_refusal_code": state.target_refusal_code,
        "target_suitability_message": state.target_suitability_message,
    }
    if assessment is None:
        metadata["model_performance_assessed"] = False
        metadata["model_performance_status"] = (
            "NOT_APPLICABLE"
            if state.analysis_mode is AnalysisExecutionMode.ANOMALY_ONLY
            else None
        )
        metadata["model_performance_rule_count"] = None
        metadata["model_performance_required_rule_count"] = None
        metadata["model_performance_failed_rule_count"] = None
        metadata["model_performance_unavailable_rule_count"] = None
    else:
        metadata["model_performance_assessed"] = True
        metadata["model_performance_status"] = assessment.status.value
        metadata["model_performance_rule_count"] = len(assessment.metric_results)
        metadata["model_performance_required_rule_count"] = (
            assessment.required_rule_count
        )
        metadata["model_performance_failed_rule_count"] = (
            assessment.failed_required_rule_count
        )
        metadata["model_performance_unavailable_rule_count"] = (
            assessment.unavailable_required_rule_count
        )
    return metadata
