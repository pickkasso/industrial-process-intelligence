"""Unit tests for AnalysisWorkflowReportBuilder (Step 11A)."""

from __future__ import annotations

import copy
import json
import math
from datetime import UTC, datetime
from typing import Any

import pytest

from process_intelligence.core.enums import AnalysisTask, AnomalyType, ColumnRole
from process_intelligence.core.schemas import AnomalyEvent, RootCauseFactor
from process_intelligence.evaluation import (
    MetricAcceptanceDirection,
    MetricAcceptanceResult,
    ModelPerformanceAcceptanceReport,
    ModelPerformanceAcceptanceStatus,
)
from process_intelligence.recommendation import (
    RecommendationChange,
    RecommendationObjective,
    RecommendationReasonCode,
    RecommendationResult,
    RecommendationSafetyDecision,
    RecommendationSafetyStatus,
    RecommendationStatus,
    VariableEligibilityAssessment,
)
from process_intelligence.recommendation.schemas import DEFAULT_RECOMMENDATION_DISCLAIMER
from process_intelligence.reporting import AnalysisWorkflowReportBuilder
from process_intelligence.workflow import (
    AnalysisExecutionMode,
    AnalysisWorkflowReport,
    AnalysisWorkflowStage,
    AnalysisWorkflowStageRecord,
    AnalysisWorkflowStatus,
    AnomalyContextOrderBasis,
    AnomalyContextRow,
    AnomalyContextValue,
    AnomalyContextWindow,
)

_UTC_START = datetime(2026, 7, 21, 10, 0, tzinfo=UTC)
_UTC_END = datetime(2026, 7, 21, 10, 5, tzinfo=UTC)
_ASSESSED_AT = datetime(2026, 7, 21, 10, 2, tzinfo=UTC)
_GENERATED_AT = datetime(2026, 7, 21, 13, 0, tzinfo=UTC)
_EVALUATED_AT = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)

_OMISSION = (
    "Additional presentation warnings were omitted due to the configured limit."
)
_UNSAFE = "Unsafe wording detected in recommendation rationale text."


def _rec_assessment(
    variable: str = "pressure",
    *,
    eligible: bool = True,
    reason_codes: list[RecommendationReasonCode] | None = None,
    **overrides: Any,
) -> VariableEligibilityAssessment:
    payload: dict[str, Any] = {
        "variable": variable,
        "factor_rank": 1,
        "factor_confidence": 0.7,
        "factor_role": str(ColumnRole.CONTROLLABLE_PROCESS),
        "factor_controllable": True,
        "factor_needs_verification": False,
        "current_value": 50.0,
        "constraint_present": True,
        "user_confirmed_controllable": True,
        "user_verified": False,
        "eligible": eligible,
        "reason_codes": list(reason_codes or []),
        "warnings": [],
    }
    payload.update(overrides)
    return VariableEligibilityAssessment(**payload)


def _rec_decision(**overrides: Any) -> RecommendationSafetyDecision:
    payload: dict[str, Any] = {
        "status": RecommendationSafetyStatus.APPROVED,
        "objective": RecommendationObjective.REDUCE_ANOMALY_SCORE,
        "eligible_variables": ["pressure", "temperature"],
        "blocked_variables": [],
        "variable_assessments": [
            _rec_assessment("pressure", eligible=True, factor_rank=1),
            _rec_assessment("temperature", eligible=True, factor_rank=2),
        ],
        "global_reason_codes": [],
        "messages": ["Safety checks passed."],
        "disclaimer": DEFAULT_RECOMMENDATION_DISCLAIMER,
        "evaluated_at": _EVALUATED_AT,
        "metadata": {},
    }
    payload.update(overrides)
    return RecommendationSafetyDecision(**payload)


def _rec_refused_decision() -> RecommendationSafetyDecision:
    return _rec_decision(
        status=RecommendationSafetyStatus.REFUSED,
        eligible_variables=[],
        blocked_variables=["pressure"],
        variable_assessments=[
            _rec_assessment(
                "pressure",
                eligible=False,
                reason_codes=[RecommendationReasonCode.NO_ELIGIBLE_VARIABLES],
                current_value=None,
            )
        ],
        global_reason_codes=[RecommendationReasonCode.NO_ELIGIBLE_VARIABLES],
        messages=["Recommendation refused by safety gate."],
    )


def _rec_change(**overrides: Any) -> RecommendationChange:
    payload: dict[str, Any] = {
        "variable": "pressure",
        "current_value": 50.0,
        "proposed_value": 48.0,
        "delta": -2.0,
        "relative_delta": -0.04,
        "rationale": "Candidate change within observed support; association only.",
        "confidence": 0.6,
        "requires_verification": True,
    }
    payload.update(overrides)
    return RecommendationChange(**payload)


def _generated_result(**overrides: Any) -> RecommendationResult:
    payload: dict[str, Any] = {
        "status": RecommendationStatus.GENERATED,
        "objective": RecommendationObjective.REDUCE_ANOMALY_SCORE,
        "safety_decision": _rec_decision(),
        "changes": [
            _rec_change(),
            _rec_change(
                variable="temperature",
                current_value=220.0,
                proposed_value=215.0,
                delta=-5.0,
                relative_delta=-5.0 / 220.0,
                confidence=0.55,
            ),
        ],
        "baseline_prediction": 80.0,
        "proposed_prediction": 82.0,
        "baseline_anomaly_score": -0.35,
        "proposed_anomaly_score": -0.55,
        "confidence": 0.5,
        "extrapolation_flag": False,
        "uncertainty_available": False,
        "disclaimer": DEFAULT_RECOMMENDATION_DISCLAIMER,
        "generated_at": _GENERATED_AT,
        "warnings": ["Verify operationally before applying."],
    }
    payload.update(overrides)
    return RecommendationResult(**payload)


def _ready_result() -> RecommendationResult:
    return RecommendationResult(
        status=RecommendationStatus.READY_FOR_OPTIMIZATION,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        safety_decision=_rec_decision(),
        changes=[],
        confidence=0.5,
        extrapolation_flag=False,
        uncertainty_available=False,
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        generated_at=_GENERATED_AT,
    )


def _refused_result() -> RecommendationResult:
    return RecommendationResult(
        status=RecommendationStatus.REFUSED,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        safety_decision=_rec_refused_decision(),
        changes=[],
        confidence=0.0,
        extrapolation_flag=False,
        uncertainty_available=False,
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        generated_at=_GENERATED_AT,
    )


def _performance_assessment(
    **overrides: Any,
) -> ModelPerformanceAcceptanceReport:
    payload: dict[str, Any] = {
        "status": ModelPerformanceAcceptanceStatus.ACCEPTABLE,
        "task": AnalysisTask.REGRESSION,
        "evaluation_available": True,
        "independent_test_evaluation": True,
        "test_row_count": 20,
        "metric_results": [
            MetricAcceptanceResult(
                metric_name="rmse",
                observed_value=1.2,
                threshold=10.0,
                direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
                required=True,
                available=True,
                passed=True,
                message="Metric rmse=1.2 meets LOWER_IS_BETTER threshold 10.0.",
            ),
            MetricAcceptanceResult(
                metric_name="mae",
                observed_value=0.9,
                threshold=10.0,
                direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
                required=True,
                available=True,
                passed=True,
                message="Metric mae=0.9 meets LOWER_IS_BETTER threshold 10.0.",
            ),
        ],
        "required_rule_count": 2,
        "passed_required_rule_count": 2,
        "failed_required_rule_count": 0,
        "unavailable_required_rule_count": 0,
        "assessed_at": _ASSESSED_AT,
        "warnings": ["Performance warning A."],
        "metadata": {},
    }
    payload.update(overrides)
    return ModelPerformanceAcceptanceReport(**payload)


def _stage_record(
    stage: AnalysisWorkflowStage,
    *,
    executed: bool = True,
    succeeded: bool = True,
    structured_refusal: bool = False,
    row_count: int | None = None,
    message: str | None = None,
    warnings: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AnalysisWorkflowStageRecord:
    return AnalysisWorkflowStageRecord(
        stage=stage,
        executed=executed,
        succeeded=succeeded,
        structured_refusal=structured_refusal,
        row_count=row_count,
        message=message or f"{stage.value} stage completed.",
        warnings=list(warnings or []),
        metadata=dict(metadata or {}),
    )


def _prefix_records(
    terminal: AnalysisWorkflowStage,
    *,
    refuse_last: bool = False,
    extra_stage_warnings: dict[AnalysisWorkflowStage, list[str]] | None = None,
    extra_stage_metadata: dict[AnalysisWorkflowStage, dict[str, Any]] | None = None,
) -> list[AnalysisWorkflowStageRecord]:
    stages = list(AnalysisWorkflowStage)
    terminal_index = stages.index(terminal)
    records: list[AnalysisWorkflowStageRecord] = []
    for index, stage in enumerate(stages):
        if index <= terminal_index:
            is_last = index == terminal_index
            warnings = list((extra_stage_warnings or {}).get(stage, []))
            metadata = dict((extra_stage_metadata or {}).get(stage, {}))
            if is_last and refuse_last:
                records.append(
                    _stage_record(
                        stage,
                        executed=True,
                        succeeded=False,
                        structured_refusal=True,
                        warnings=warnings,
                        metadata=metadata,
                    )
                )
            else:
                records.append(
                    _stage_record(
                        stage,
                        executed=True,
                        succeeded=True,
                        warnings=warnings,
                        metadata=metadata,
                    )
                )
        else:
            records.append(
                _stage_record(
                    stage,
                    executed=False,
                    succeeded=False,
                    structured_refusal=False,
                    message=f"{stage.value} stage skipped.",
                )
            )
    return records


def _default_metadata(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "raw_csv_loaded": True,
        "raw_row_count": 100,
        "processed_row_count": 100,
        "cohort_row_count": 100,
        "train_row_count": 60,
        "validation_row_count": 20,
        "test_row_count": 20,
        "selected_industry": "semiconductor",
        "selected_task": "REGRESSION",
        "inferred_task": "REGRESSION",
        "task_selection_source": "ROUTER",
        "task_override_applied": False,
        "selected_supervised_model_available": True,
        "selected_anomaly_model_available": True,
        "residual_calibration_performed": True,
        "independent_test_evaluation_performed": True,
        "anomaly_event_count": 2,
        "diagnosis_performed": True,
        "recommendation_pipeline_executed": True,
        "recommendation_generated": True,
        "row_identity_preserved": True,
        "test_used_for_model_selection": False,
        "test_used_for_threshold_calibration": False,
        "anomaly_score_direction": "higher_is_more_anomalous",
        "model_performance_assessed": True,
        "model_performance_status": "ACCEPTABLE",
        "model_performance_rule_count": 2,
        "model_performance_required_rule_count": 2,
        "model_performance_failed_rule_count": 0,
        "model_performance_unavailable_rule_count": 0,
        "model_performance_gate_bypassed": False,
        "extrapolation_computed": False,
        "uncertainty_computed": False,
        "workflow_orchestration_only": True,
        "model_fit_performed": True,
        "secret_internal_flag": True,
        "csv_path": "C:/secret/data.csv",
    }
    payload.update(overrides)
    return payload


def _completed_report(**overrides: Any) -> AnalysisWorkflowReport:
    payload: dict[str, Any] = {
        "status": AnalysisWorkflowStatus.COMPLETED,
        "terminal_stage": AnalysisWorkflowStage.RECOMMENDATION,
        "stage_records": _prefix_records(
            AnalysisWorkflowStage.RECOMMENDATION,
            extra_stage_warnings={
                AnalysisWorkflowStage.LOAD: ["Stage warning LOAD."],
                AnalysisWorkflowStage.DIAGNOSIS: ["Stage warning DIAGNOSIS."],
            },
            extra_stage_metadata={
                AnalysisWorkflowStage.LOAD: {
                    "column_count": 12,
                    "csv_path": "C:/secret/raw.csv",
                    "traceback": "Traceback...",
                    "model": "should-remove",
                    "supervised_model_key": "ridge",
                }
            },
        ),
        "model_performance_assessment": _performance_assessment(),
        "final_recommendation": _generated_result(),
        "selected_industry": "semiconductor",
        "selected_task": AnalysisTask.REGRESSION,
        "inferred_task": AnalysisTask.REGRESSION,
        "task_selection_source": "ROUTER",
        "task_override_applied": False,
        "selected_supervised_model_key": "ridge",
        "selected_anomaly_model_key": "isolation_forest",
        "selected_operating_row_id": 7,
        "anomaly_event_count": 2,
        "diagnosis_factor_count": 3,
        "raw_row_count": 100,
        "processed_row_count": 100,
        "cohort_row_count": 100,
        "train_row_count": 60,
        "validation_row_count": 20,
        "test_row_count": 20,
        "cohort_filter_summary": {
            "configured": False,
            "source_row_count": 100,
            "retained_row_count": 100,
            "excluded_row_count": 0,
            "null_excluded_count": 0,
        },
        "started_at": _UTC_START,
        "completed_at": _UTC_END,
        "total_seconds": 300.0,
        "warnings": ["Workflow warning W1.", "Performance warning A."],
        "metadata": _default_metadata(),
    }
    payload.update(overrides)
    return AnalysisWorkflowReport(**payload)


def _partial_report(**overrides: Any) -> AnalysisWorkflowReport:
    payload = {
        "status": AnalysisWorkflowStatus.PARTIAL,
        "terminal_stage": AnalysisWorkflowStage.RECOMMENDATION,
        "stage_records": _prefix_records(AnalysisWorkflowStage.RECOMMENDATION),
        "model_performance_assessment": _performance_assessment(warnings=[]),
        "final_recommendation": _ready_result(),
        "selected_industry": "semiconductor",
        "selected_task": AnalysisTask.REGRESSION,
        "inferred_task": AnalysisTask.REGRESSION,
        "task_selection_source": "ROUTER",
        "task_override_applied": False,
        "selected_supervised_model_key": "ridge",
        "selected_anomaly_model_key": "isolation_forest",
        "selected_operating_row_id": 7,
        "anomaly_event_count": 2,
        "diagnosis_factor_count": 3,
        "raw_row_count": 100,
        "processed_row_count": 100,
        "cohort_row_count": 100,
        "train_row_count": 60,
        "validation_row_count": 20,
        "test_row_count": 20,
        "cohort_filter_summary": {
            "configured": False,
            "source_row_count": 100,
            "retained_row_count": 100,
            "excluded_row_count": 0,
            "null_excluded_count": 0,
        },
        "started_at": _UTC_START,
        "completed_at": _UTC_END,
        "total_seconds": 300.0,
        "warnings": [],
        "metadata": _default_metadata(recommendation_generated=False),
    }
    payload.update(overrides)
    return AnalysisWorkflowReport(**payload)


def _refused_report(**overrides: Any) -> AnalysisWorkflowReport:
    payload = {
        "status": AnalysisWorkflowStatus.REFUSED,
        "terminal_stage": AnalysisWorkflowStage.VALIDATE,
        "stage_records": _prefix_records(
            AnalysisWorkflowStage.VALIDATE,
            refuse_last=True,
        ),
        "model_performance_assessment": None,
        "final_recommendation": None,
        "selected_industry": None,
        "selected_task": None,
        "selected_supervised_model_key": None,
        "selected_anomaly_model_key": None,
        "selected_operating_row_id": None,
        "anomaly_event_count": 0,
        "diagnosis_factor_count": 0,
        "raw_row_count": 100,
        "processed_row_count": 100,
        "cohort_row_count": 100,
        "train_row_count": 0,
        "validation_row_count": 0,
        "test_row_count": 0,
        "cohort_filter_summary": {
            "configured": False,
            "source_row_count": 100,
            "retained_row_count": 100,
            "excluded_row_count": 0,
            "null_excluded_count": 0,
        },
        "started_at": _UTC_START,
        "completed_at": _UTC_END,
        "total_seconds": 12.0,
        "warnings": ["Validation refused."],
        "metadata": {
            "raw_csv_loaded": True,
            "row_identity_preserved": True,
            "test_used_for_model_selection": False,
            "test_used_for_threshold_calibration": False,
            "model_performance_gate_bypassed": False,
            "workflow_orchestration_only": True,
        },
    }
    payload.update(overrides)
    return AnalysisWorkflowReport(**payload)


def test_completed_conversion() -> None:
    outcome = AnalysisWorkflowReportBuilder().build(_completed_report())
    report = outcome.report
    assert report.overview.status is AnalysisWorkflowStatus.COMPLETED
    assert report.recommendation is not None
    assert report.recommendation.status is RecommendationStatus.GENERATED
    assert report.model_performance is not None
    assert report.model_performance.status is ModelPerformanceAcceptanceStatus.ACCEPTABLE
    assert report.data_summary.cohort_row_count == 100
    assert report.cohort_filter_summary.configured is False
    assert report.cohort_filter_summary.range_display == "Not configured"


def test_cohort_filter_summary_mapping() -> None:
    source = _anomaly_only_report(
        cohort_row_count=40,
        train_row_count=24,
        validation_row_count=8,
        test_row_count=8,
        cohort_filter_summary={
            "configured": True,
            "column_name": "RSOCavg",
            "lower_bound": 80.0,
            "upper_bound": 100.0,
            "include_lower": True,
            "include_upper": True,
            "exclude_filter_column_from_features": True,
            "source_row_count": 90,
            "retained_row_count": 40,
            "excluded_row_count": 50,
            "null_excluded_count": 2,
        },
    )
    report = AnalysisWorkflowReportBuilder().build(source).report
    assert report.cohort_filter_summary.configured is True
    assert report.cohort_filter_summary.column_name == "RSOCavg"
    assert report.cohort_filter_summary.range_display == "80.0 ≤ x ≤ 100.0"
    assert report.cohort_filter_summary.filter_column_used_as_feature_display == "No"
    assert report.data_summary.cohort_row_count == 40
    assert report.cohort_filter_summary.retained_row_count == 40
    encoded = report.model_dump_json()
    assert "RSOCavg" in encoded
    assert "DataFrame" not in encoded


def test_partial_conversion() -> None:
    outcome = AnalysisWorkflowReportBuilder().build(_partial_report())
    report = outcome.report
    assert report.overview.status is AnalysisWorkflowStatus.PARTIAL
    assert report.recommendation is not None
    assert report.recommendation.status is RecommendationStatus.READY_FOR_OPTIMIZATION
    assert report.recommendation.changes == []


def test_refused_conversion() -> None:
    outcome = AnalysisWorkflowReportBuilder().build(_refused_report())
    report = outcome.report
    assert report.overview.status is AnalysisWorkflowStatus.REFUSED
    assert report.recommendation is None
    assert report.model_performance is None


def test_invalid_input_type() -> None:
    builder = AnalysisWorkflowReportBuilder()
    with pytest.raises(TypeError, match="AnalysisWorkflowReport"):
        builder.build("not-a-report")  # type: ignore[arg-type]


def test_overview_status_and_headline_determinism() -> None:
    completed = AnalysisWorkflowReportBuilder().build(_completed_report()).report
    partial = AnalysisWorkflowReportBuilder().build(_partial_report()).report
    refused = AnalysisWorkflowReportBuilder().build(_refused_report()).report
    assert completed.overview.headline == (
        "Analysis completed with a generated recommendation"
    )
    assert partial.overview.headline == (
        "Analysis completed without an executable recommendation"
    )
    assert refused.overview.headline == (
        "Analysis was stopped by a safety or validation gate"
    )
    again = AnalysisWorkflowReportBuilder().build(_completed_report()).report
    assert again.overview.headline == completed.overview.headline
    assert again.overview.summary == completed.overview.summary


def test_data_routing_model_preservation() -> None:
    source = _completed_report()
    report = AnalysisWorkflowReportBuilder().build(source).report
    assert report.data_summary.raw_row_count == source.raw_row_count
    assert report.data_summary.test_row_count == source.test_row_count
    assert report.data_summary.selected_operating_row_id == 7
    assert report.data_summary.row_identity_preserved is True
    assert report.routing_summary.selected_industry == "semiconductor"
    assert report.routing_summary.selected_task is AnalysisTask.REGRESSION
    assert report.routing_summary.inferred_task is AnalysisTask.REGRESSION
    assert report.routing_summary.task_selection_source == "ROUTER"
    assert report.routing_summary.task_override_applied is False
    assert report.routing_summary.target_column is None
    assert report.routing_summary.feature_count is None
    assert report.model_summary.supervised_model_key == "ridge"
    assert report.model_summary.anomaly_model_key == "isolation_forest"
    assert report.model_summary.independent_test_evaluation_performed is True
    assert report.model_summary.residual_calibration_performed is True
    assert report.model_summary.test_used_for_model_selection is False
    assert report.model_summary.test_used_for_threshold_calibration is False
    assert report.model_summary.anomaly_score_direction == "higher_is_more_anomalous"


def test_performance_and_metric_preservation() -> None:
    source = _completed_report()
    report = AnalysisWorkflowReportBuilder().build(source).report
    assert report.model_performance is not None
    assert source.model_performance_assessment is not None
    assert report.model_performance.status is source.model_performance_assessment.status
    assert report.model_performance.assessed_at == _ASSESSED_AT
    source_metric = source.model_performance_assessment.metric_results[0]
    view_metric = report.model_performance.metrics[0]
    assert view_metric.metric_name == source_metric.metric_name
    assert view_metric.observed_value == pytest.approx(source_metric.observed_value)
    assert view_metric.threshold == pytest.approx(source_metric.threshold)


def test_stage_order_sequence_and_labels() -> None:
    source = _completed_report()
    report = AnalysisWorkflowReportBuilder().build(source).report
    assert [stage.stage for stage in report.stages] == [
        record.stage for record in source.stage_records
    ]
    assert [stage.sequence for stage in report.stages] == list(
        range(1, len(report.stages) + 1)
    )
    # Completed path executes PROFILE; use refused report for SKIPPED labels.
    refused = AnalysisWorkflowReportBuilder().build(_refused_report()).report
    skipped_stages = [stage for stage in refused.stages if not stage.executed]
    assert skipped_stages
    assert all(stage.status_label == "SKIPPED" for stage in skipped_stages)
    validate = next(
        stage
        for stage in refused.stages
        if stage.stage is AnalysisWorkflowStage.VALIDATE
    )
    assert validate.status_label == "REFUSED"
    assert validate.structured_refusal is True


def test_recommendation_mapping_preservation() -> None:
    source = _completed_report()
    report = AnalysisWorkflowReportBuilder().build(source).report
    assert report.recommendation is not None
    assert source.final_recommendation is not None
    assert [c.variable for c in report.recommendation.changes] == [
        c.variable for c in source.final_recommendation.changes
    ]
    first_src = source.final_recommendation.changes[0]
    first_view = report.recommendation.changes[0]
    assert first_view.current_value == pytest.approx(first_src.current_value)
    assert first_view.proposed_value == pytest.approx(first_src.proposed_value)
    assert first_view.delta == pytest.approx(first_src.delta)
    assert first_view.confidence == pytest.approx(first_src.confidence)
    assert report.recommendation.baseline_anomaly_score == pytest.approx(-0.35)
    assert report.recommendation.proposed_anomaly_score == pytest.approx(-0.55)
    assert report.recommendation.baseline_anomaly_score < 0
    assert report.recommendation.safety_status is RecommendationSafetyStatus.APPROVED
    assert report.recommendation.safety_messages == ["Safety checks passed."]


def test_no_clipping_of_negative_scores() -> None:
    source = _completed_report(
        final_recommendation=_generated_result(
            baseline_anomaly_score=-12.5,
            proposed_anomaly_score=-20.75,
        )
    )
    report = AnalysisWorkflowReportBuilder().build(source).report
    assert report.recommendation is not None
    assert report.recommendation.baseline_anomaly_score == pytest.approx(-12.5)
    assert report.recommendation.proposed_anomaly_score == pytest.approx(-20.75)
    baseline = report.recommendation.baseline_anomaly_score
    assert baseline is not None
    assert abs(baseline) != baseline


def test_warning_aggregation_order_dedupe_and_cap() -> None:
    source = _completed_report()
    report = AnalysisWorkflowReportBuilder().build(source).report
    assert report.warnings[0] == "Workflow warning W1."
    assert report.warnings[1] == "Performance warning A."
    assert "Stage warning LOAD." in report.warnings
    assert "Stage warning DIAGNOSIS." in report.warnings
    assert "Safety checks passed." in report.warnings
    assert "Verify operationally before applying." in report.warnings
    # performance warning appears once despite also being in workflow warnings
    assert report.warnings.count("Performance warning A.") == 1

    capped = AnalysisWorkflowReportBuilder(maximum_warnings=3).build(source).report
    assert len(capped.warnings) == 3
    assert capped.warnings[-1] == _OMISSION
    assert _OMISSION not in capped.warnings[:-1]


def test_stage_warning_exclusion_policy() -> None:
    source = _completed_report()
    report = AnalysisWorkflowReportBuilder(
        include_stage_warnings=False
    ).build(source).report
    assert "Stage warning LOAD." not in report.warnings
    assert "Stage warning DIAGNOSIS." not in report.warnings
    assert report.warnings[0] == "Workflow warning W1."


def test_unsafe_rationale_detection_without_rewrite() -> None:
    unsafe_text = (
        "Candidate remains association-only and is not guaranteed in isolation."
    )
    source = _completed_report(
        final_recommendation=_generated_result(
            changes=[
                _rec_change(rationale=unsafe_text),
                _rec_change(
                    variable="temperature",
                    current_value=220.0,
                    proposed_value=215.0,
                    delta=-5.0,
                    relative_delta=-5.0 / 220.0,
                    confidence=0.55,
                ),
            ]
        )
    )
    report = AnalysisWorkflowReportBuilder().build(source).report
    assert report.recommendation is not None
    assert report.recommendation.changes[0].rationale == unsafe_text
    assert _UNSAFE in report.warnings


def test_disclaimer_aggregation() -> None:
    report = AnalysisWorkflowReportBuilder().build(_completed_report()).report
    assert report.disclaimers[0] == DEFAULT_RECOMMENDATION_DISCLAIMER
    joined = " ".join(report.disclaimers).lower()
    assert "model-based" in joined
    assert "causation" in joined
    assert "verification" in joined
    assert "not guaranteed" in joined
    assert "metric thresholds" in joined
    assert "deployment safety" in joined
    assert "proven optimal" not in joined
    assert "safe to deploy" not in joined


def test_metadata_allowlist_and_sensitive_removal() -> None:
    report = AnalysisWorkflowReportBuilder().build(_completed_report()).report
    assert "csv_path" not in report.metadata
    assert "secret_internal_flag" not in report.metadata
    assert "model_fit_performed" not in report.metadata
    assert report.metadata["row_identity_preserved"] is True
    assert report.metadata["model_performance_gate_bypassed"] is False
    assert report.metadata["test_used_for_model_selection"] is False

    load_stage = next(
        stage for stage in report.stages if stage.stage is AnalysisWorkflowStage.LOAD
    )
    assert "csv_path" not in load_stage.metadata
    assert "traceback" not in load_stage.metadata
    assert "model" not in load_stage.metadata
    assert load_stage.metadata.get("column_count") == 12
    assert load_stage.metadata.get("supervised_model_key") == "ridge"


def test_json_safe_and_datetime_preservation() -> None:
    source = _completed_report()
    report = AnalysisWorkflowReportBuilder().build(source).report
    dumped = report.model_dump(mode="json")
    json.dumps(dumped)
    assert report.overview.started_at == _UTC_START
    assert report.overview.completed_at == _UTC_END
    assert report.overview.total_seconds == pytest.approx(300.0)
    assert report.model_performance is not None
    assert report.model_performance.assessed_at == _ASSESSED_AT
    assert report.recommendation is not None
    assert report.recommendation.generated_at == _GENERATED_AT
    assert "DataFrame" not in str(dumped)
    assert "ndarray" not in str(dumped)
    assert "C:/secret" not in json.dumps(dumped)


def test_input_immutability_and_nested_immutability() -> None:
    source = _completed_report()
    before = copy.deepcopy(source.model_dump(mode="python"))
    warnings_before = list(source.warnings)
    metadata_before = dict(source.metadata)
    stage_meta_before = dict(source.stage_records[0].metadata)
    rec_before = source.final_recommendation
    assert rec_before is not None
    change_rationale_before = rec_before.changes[0].rationale
    safety_messages_before = list(rec_before.safety_decision.messages)

    AnalysisWorkflowReportBuilder().build(source)

    assert source.model_dump(mode="python") == before
    assert source.warnings == warnings_before
    assert source.metadata == metadata_before
    assert source.stage_records[0].metadata == stage_meta_before
    assert source.final_recommendation is not None
    assert source.final_recommendation.changes[0].rationale == change_rationale_before
    assert source.final_recommendation.safety_decision.messages == safety_messages_before


def test_no_cache_determinism_and_instance_isolation() -> None:
    source = _completed_report()
    builder_a = AnalysisWorkflowReportBuilder(maximum_warnings=50)
    builder_b = AnalysisWorkflowReportBuilder(maximum_warnings=3)
    first = builder_a.build(source).report.model_dump(mode="json")
    second = builder_a.build(source).report.model_dump(mode="json")
    assert first == second
    capped = builder_b.build(source).report
    assert len(capped.warnings) == 3
    assert builder_a.maximum_warnings == 50
    assert builder_b.maximum_warnings == 3


def test_returned_dto_mutation_does_not_affect_next_call() -> None:
    source = _completed_report()
    builder = AnalysisWorkflowReportBuilder()
    first = builder.build(source).report
    first.warnings.append("mutated-local-warning")
    first.metadata["mutated"] = True
    second = builder.build(source).report
    assert "mutated-local-warning" not in second.warnings
    assert "mutated" not in second.metadata


def test_get_metadata_scalar_only_and_capabilities() -> None:
    builder = AnalysisWorkflowReportBuilder(
        maximum_warnings=10,
        include_stage_metadata=False,
        include_stage_warnings=False,
    )
    meta = builder.get_metadata()
    assert meta["maximum_warnings"] == 10
    assert meta["include_stage_metadata"] is False
    assert meta["include_stage_warnings"] is False
    assert meta["produces_json_safe_dto"] is True
    assert meta["performs_model_scoring"] is False
    assert meta["performs_model_fit"] is False
    assert meta["performs_model_refit"] is False
    assert meta["performs_metric_recalculation"] is False
    assert meta["performs_diagnosis"] is False
    assert meta["performs_recommendation"] is False
    assert meta["includes_raw_dataframe"] is False
    assert meta["includes_model_object"] is False
    assert meta["includes_estimator_object"] is False
    assert meta["preserves_negative_anomaly_scores"] is True
    assert meta["rewrites_backend_results"] is False
    meta["maximum_warnings"] = 999
    assert builder.get_metadata()["maximum_warnings"] == 10


def test_builder_config_validation() -> None:
    with pytest.raises(ValueError):
        AnalysisWorkflowReportBuilder(maximum_warnings=0)
    with pytest.raises(ValueError):
        AnalysisWorkflowReportBuilder(maximum_warnings=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AnalysisWorkflowReportBuilder(include_stage_metadata=1)  # type: ignore[arg-type]


def test_refused_with_recommendation_mapping() -> None:
    source = _completed_report(
        status=AnalysisWorkflowStatus.REFUSED,
        terminal_stage=AnalysisWorkflowStage.RECOMMENDATION,
        stage_records=_prefix_records(AnalysisWorkflowStage.RECOMMENDATION),
        final_recommendation=_refused_result(),
        metadata=_default_metadata(recommendation_generated=False),
    )
    report = AnalysisWorkflowReportBuilder().build(source).report
    assert report.overview.status is AnalysisWorkflowStatus.REFUSED
    assert report.recommendation is not None
    assert report.recommendation.status is RecommendationStatus.REFUSED
    assert report.recommendation.changes == []


# ---------------------------------------------------------------------------
# ANOMALY_ONLY analysis-mode presentation (Step 11B.5)
# ---------------------------------------------------------------------------

_ANOMALY_ONLY_OVERVIEW = (
    "Anomaly-only analysis completed. Recommendation generation is not "
    "enabled for this analysis mode."
)

_ANOMALY_ONLY_SKIPPED_STAGES = frozenset(
    {
        AnalysisWorkflowStage.TASK_ROUTING,
        AnalysisWorkflowStage.SUPERVISED_SCREENING,
        AnalysisWorkflowStage.SUPERVISED_FINAL_EVALUATION,
        AnalysisWorkflowStage.RESIDUAL_CALIBRATION,
        AnalysisWorkflowStage.RESIDUAL_FINAL_EVALUATION,
        AnalysisWorkflowStage.RECOMMENDATION,
    }
)


def _anomaly_only_stage_records() -> list[AnalysisWorkflowStageRecord]:
    records: list[AnalysisWorkflowStageRecord] = []
    for stage in list(AnalysisWorkflowStage):
        if stage is AnalysisWorkflowStage.DIAGNOSIS:
            break
        if stage in _ANOMALY_ONLY_SKIPPED_STAGES:
            records.append(
                _stage_record(
                    stage,
                    executed=False,
                    succeeded=False,
                    structured_refusal=False,
                    message="Not applicable for anomaly-only analysis.",
                )
            )
        else:
            records.append(_stage_record(stage, executed=True, succeeded=True))
    records.append(
        _stage_record(
            AnalysisWorkflowStage.DIAGNOSIS,
            executed=True,
            succeeded=True,
            metadata={
                "diagnosis_factor_count": 2,
                "diagnosis_source": "ROBUST_GROUP_COMPARISON",
                "target_based_diagnosis": False,
                "association_not_causation": True,
            },
        )
    )
    records.append(
        _stage_record(
            AnalysisWorkflowStage.RECOMMENDATION,
            executed=False,
            succeeded=False,
            structured_refusal=False,
            message="Not applicable for anomaly-only analysis.",
            metadata={"recommendation_applicable": False},
        )
    )
    return records


def _anomaly_only_metadata(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "raw_csv_loaded": True,
        "raw_row_count": 90,
        "processed_row_count": 90,
        "cohort_row_count": 90,
        "train_row_count": 54,
        "validation_row_count": 18,
        "test_row_count": 18,
        "analysis_mode": "ANOMALY_ONLY",
        "selected_industry": "battery",
        "selected_task": None,
        "inferred_task": None,
        "task_selection_source": None,
        "task_override_applied": False,
        "selected_supervised_model_available": False,
        "selected_anomaly_model_available": True,
        "residual_calibration_performed": False,
        "independent_test_evaluation_performed": True,
        "anomaly_event_count": 2,
        "anomaly_event_selection_source": "UNSUPERVISED_ANOMALY_SCORE",
        "diagnosis_performed": True,
        "diagnosis_source": "ROBUST_GROUP_COMPARISON",
        "recommendation_pipeline_executed": False,
        "recommendation_generated": False,
        "recommendation_applicable": False,
        "row_identity_preserved": True,
        "test_used_for_model_selection": False,
        "test_used_for_threshold_calibration": False,
        "anomaly_score_direction": "higher_is_more_anomalous",
        "model_performance_assessed": False,
        "model_performance_status": "NOT_APPLICABLE",
        "model_performance_gate_bypassed": False,
        "extrapolation_computed": False,
        "uncertainty_computed": False,
        "workflow_orchestration_only": True,
        "target_column": None,
        "feature_count": 5,
        "target_suitable": None,
        "target_unique_non_null_count": None,
        "target_refusal_code": None,
        "target_suitability_message": "NOT_APPLICABLE",
        "model_fit_performed": True,
        "csv_path": "C:/secret/battery.csv",
    }
    payload.update(overrides)
    return payload


def _anomaly_only_report(**overrides: Any) -> AnalysisWorkflowReport:
    payload: dict[str, Any] = {
        "status": AnalysisWorkflowStatus.PARTIAL,
        "terminal_stage": AnalysisWorkflowStage.DIAGNOSIS,
        "stage_records": _anomaly_only_stage_records(),
        "analysis_mode": AnalysisExecutionMode.ANOMALY_ONLY,
        "model_performance_assessment": None,
        "final_recommendation": None,
        "selected_industry": "battery",
        "selected_task": None,
        "inferred_task": None,
        "task_selection_source": None,
        "task_override_applied": False,
        "selected_supervised_model_key": None,
        "selected_anomaly_model_key": "isolation_forest",
        "selected_operating_row_id": 5,
        "anomaly_event_count": 2,
        "diagnosis_factor_count": 2,
        "raw_row_count": 90,
        "processed_row_count": 90,
        "cohort_row_count": 90,
        "train_row_count": 54,
        "validation_row_count": 18,
        "test_row_count": 18,
        "cohort_filter_summary": {
            "configured": False,
            "source_row_count": 90,
            "retained_row_count": 90,
            "excluded_row_count": 0,
            "null_excluded_count": 0,
        },
        "started_at": _UTC_START,
        "completed_at": _UTC_END,
        "total_seconds": 8.0,
        "warnings": [],
        "metadata": _anomaly_only_metadata(),
    }
    payload.update(overrides)
    return AnalysisWorkflowReport(**payload)


def test_anomaly_only_overview_summary_and_headline() -> None:
    report = AnalysisWorkflowReportBuilder().build(_anomaly_only_report()).report
    assert report.overview.status is AnalysisWorkflowStatus.PARTIAL
    assert report.overview.summary == _ANOMALY_ONLY_OVERVIEW
    assert report.overview.headline == (
        "Analysis completed without an executable recommendation"
    )


def test_anomaly_only_no_generic_refusal_banner() -> None:
    report = AnalysisWorkflowReportBuilder().build(_anomaly_only_report()).report
    assert report.overview.status is not AnalysisWorkflowStatus.REFUSED
    assert "safety or validation gate" not in report.overview.headline
    assert "safety or validation gate" not in report.overview.summary


def test_anomaly_only_analysis_mode_shown_in_metadata() -> None:
    report = AnalysisWorkflowReportBuilder().build(_anomaly_only_report()).report
    assert report.metadata["analysis_mode"] == "ANOMALY_ONLY"
    assert report.metadata["recommendation_applicable"] is False


def test_anomaly_only_target_and_task_not_applicable() -> None:
    report = AnalysisWorkflowReportBuilder().build(_anomaly_only_report()).report
    assert report.routing_summary.target_column is None
    assert report.routing_summary.selected_task is None
    assert report.routing_summary.inferred_task is None
    assert report.routing_summary.target_suitable is None
    assert report.routing_summary.target_suitability_message == "NOT_APPLICABLE"
    assert report.routing_summary.feature_count == 5


def test_anomaly_only_supervised_model_not_applicable() -> None:
    report = AnalysisWorkflowReportBuilder().build(_anomaly_only_report()).report
    assert report.model_summary.supervised_model_key is None


def test_anomaly_only_anomaly_model_displayed() -> None:
    report = AnalysisWorkflowReportBuilder().build(_anomaly_only_report()).report
    assert report.model_summary.anomaly_model_key == "isolation_forest"
    assert report.model_summary.independent_test_evaluation_performed is True


def test_anomaly_only_performance_not_applicable() -> None:
    report = AnalysisWorkflowReportBuilder().build(_anomaly_only_report()).report
    assert report.model_performance is None


def test_anomaly_only_residual_not_applicable() -> None:
    report = AnalysisWorkflowReportBuilder().build(_anomaly_only_report()).report
    assert report.model_summary.residual_calibration_performed is False
    residual_stages = [
        stage
        for stage in report.stages
        if stage.stage
        in {
            AnalysisWorkflowStage.RESIDUAL_CALIBRATION,
            AnalysisWorkflowStage.RESIDUAL_FINAL_EVALUATION,
        }
    ]
    assert len(residual_stages) == 2
    assert all(stage.status_label == "SKIPPED" for stage in residual_stages)


def test_anomaly_only_recommendation_not_enabled() -> None:
    report = AnalysisWorkflowReportBuilder().build(_anomaly_only_report()).report
    assert report.recommendation is None
    assert report.overview.recommendation_status is None
    recommendation_stage = next(
        stage
        for stage in report.stages
        if stage.stage is AnalysisWorkflowStage.RECOMMENDATION
    )
    assert recommendation_stage.executed is False
    assert recommendation_stage.status_label == "SKIPPED"
    assert recommendation_stage.metadata.get("recommendation_applicable") is False


def test_anomaly_only_events_and_diagnosis_factors_displayed() -> None:
    report = AnalysisWorkflowReportBuilder().build(_anomaly_only_report()).report
    assert report.data_summary.anomaly_event_count == 2
    assert report.data_summary.diagnosis_factor_count == 2
    diagnosis_stage = next(
        stage
        for stage in report.stages
        if stage.stage is AnalysisWorkflowStage.DIAGNOSIS
    )
    assert diagnosis_stage.executed is True
    assert diagnosis_stage.status_label == "SUCCEEDED"
    assert diagnosis_stage.metadata.get("diagnosis_source") == "ROBUST_GROUP_COMPARISON"
    assert diagnosis_stage.metadata.get("target_based_diagnosis") is False


def test_anomaly_only_task_routing_and_supervised_stages_skipped() -> None:
    report = AnalysisWorkflowReportBuilder().build(_anomaly_only_report()).report
    skipped_stages = {
        stage.stage for stage in report.stages if stage.status_label == "SKIPPED"
    }
    assert AnalysisWorkflowStage.TASK_ROUTING in skipped_stages
    assert AnalysisWorkflowStage.SUPERVISED_SCREENING in skipped_stages
    assert AnalysisWorkflowStage.SUPERVISED_FINAL_EVALUATION in skipped_stages


def test_anomaly_only_json_safe_serialization() -> None:
    report = AnalysisWorkflowReportBuilder().build(_anomaly_only_report()).report
    dumped = report.model_dump(mode="json")
    json.dumps(dumped)
    assert "csv_path" not in report.metadata
    assert "DataFrame" not in str(dumped)
    assert "ndarray" not in str(dumped)


def test_anomaly_only_metadata_sensitive_removal() -> None:
    report = AnalysisWorkflowReportBuilder().build(_anomaly_only_report()).report
    assert "csv_path" not in report.metadata
    assert "model_fit_performed" not in report.metadata
    assert report.metadata["row_identity_preserved"] is True


def test_supervised_report_regression_still_passes_with_analysis_mode() -> None:
    # Default SUPERVISED analysis_mode must continue to build correctly and
    # must not trigger the anomaly-only overview message.
    source = _completed_report()
    assert source.analysis_mode is AnalysisExecutionMode.SUPERVISED
    report = AnalysisWorkflowReportBuilder().build(source).report
    assert report.overview.summary != _ANOMALY_ONLY_OVERVIEW
    assert report.model_summary.supervised_model_key == "ridge"
    assert report.model_performance is not None
    assert "Proposed changes require" in " ".join(report.disclaimers)
    assert "Performance acceptance only confirms" in " ".join(report.disclaimers)


def _backend_event(
    *,
    row_id: int,
    score: float,
    detector: str = "unsupervised_anomaly_model",
) -> AnomalyEvent:
    return AnomalyEvent(
        anomaly_id=str(row_id),
        anomaly_type=AnomalyType.PROCESS_INPUT,
        anomaly_score=score,
        severity="high",
        sample_id=row_id,
        model_confidence=0.75,
        detector=detector,
        rationale="Selected anomaly event for presentation mapping tests.",
        contributing_variables=[],
    )


def _backend_factor(
    *,
    variable: str,
    direction: str = "POSITIVE",
    score: float = 0.8,
    anomaly_median: float = 4.1,
    reference_median: float = 3.7,
    confidence: float = 0.8,
    robust_z_score: float | None = None,
    robust_scale: float | None = None,
    robust_scale_status: str = "AVAILABLE",
) -> RootCauseFactor:
    if robust_z_score is None and robust_scale_status == "AVAILABLE":
        robust_z_score = 2.5
    if robust_scale is None:
        robust_scale = 0.0 if robust_scale_status == "ZERO_VARIANCE" else 1.4826
    if robust_z_score is None:
        robust_z_text = "robust_z_score=None"
    else:
        robust_z_text = f"robust_z_score={robust_z_score:.6g}"
    signed = anomaly_median - reference_median
    evidence = (
        "Higher values were associated with the analyzed anomaly subset. "
        f"reference_median={reference_median:.6g}; "
        f"anomaly_median={anomaly_median:.6g}; "
        f"signed_location_difference={signed:.6g}; "
        f"robust_scale={robust_scale:.6g}; "
        f"robust_scale_status={robust_scale_status}; "
        f"{robust_z_text}; "
        f"raw_association_score={score:.6g}; "
        f"normalized_association_score={min(score / (score + 1.0), 1.0):.6g}."
    )
    return RootCauseFactor(
        variable=variable,
        direction=direction,
        deviation=signed,
        role=ColumnRole.UNKNOWN,
        controllable=False,
        evidence=evidence,
        confidence=confidence,
        needs_verification=True,
    )


def _anomaly_result_report(**overrides: Any) -> AnalysisWorkflowReport:
    events = [
        _backend_event(row_id=13, score=-0.55),
        _backend_event(row_id=21, score=0.42, detector="top_anomaly_score_candidate"),
    ]
    factors = [
        _backend_factor(variable="voltage", direction="POSITIVE"),
        _backend_factor(
            variable="current",
            direction="NEGATIVE",
            score=0.7,
            anomaly_median=1.1,
            reference_median=1.5,
            confidence=0.7,
        ),
    ]
    payload: dict[str, Any] = {
        "status": AnalysisWorkflowStatus.PARTIAL,
        "terminal_stage": AnalysisWorkflowStage.DIAGNOSIS,
        "stage_records": _anomaly_only_stage_records(),
        "analysis_mode": AnalysisExecutionMode.ANOMALY_ONLY,
        "model_performance_assessment": None,
        "final_recommendation": None,
        "selected_industry": "battery",
        "selected_task": None,
        "inferred_task": None,
        "task_selection_source": None,
        "task_override_applied": False,
        "selected_supervised_model_key": None,
        "selected_anomaly_model_key": "isolation_forest",
        "selected_operating_row_id": 13,
        "anomaly_event_count": 2,
        "diagnosis_factor_count": 2,
        "anomaly_events": events,
        "diagnosis_factors": factors,
        "raw_row_count": 90,
        "processed_row_count": 90,
        "cohort_row_count": 90,
        "train_row_count": 54,
        "validation_row_count": 18,
        "test_row_count": 18,
        "cohort_filter_summary": {
            "configured": False,
            "source_row_count": 90,
            "retained_row_count": 90,
            "excluded_row_count": 0,
            "null_excluded_count": 0,
        },
        "started_at": _UTC_START,
        "completed_at": _UTC_END,
        "total_seconds": 8.0,
        "warnings": [],
        "metadata": _anomaly_only_metadata(
            anomaly_event_selection_source="UNSUPERVISED_ANOMALY_SCORE",
            diagnosis_source="ROBUST_GROUP_COMPARISON",
        ),
    }
    payload.update(overrides)
    return AnalysisWorkflowReport(**payload)


def test_selected_anomaly_events_mapping() -> None:
    report = AnalysisWorkflowReportBuilder().build(_anomaly_result_report()).report
    assert len(report.anomaly_events) == 2
    assert report.anomaly_events[0].original_row_id == 13
    assert report.anomaly_events[1].original_row_id == 21


def test_original_row_id_preserved() -> None:
    report = AnalysisWorkflowReportBuilder().build(_anomaly_result_report()).report
    assert [event.original_row_id for event in report.anomaly_events] == [13, 21]


def test_anomaly_score_preserved_including_negative() -> None:
    report = AnalysisWorkflowReportBuilder().build(_anomaly_result_report()).report
    assert report.anomaly_events[0].anomaly_score == pytest.approx(-0.55)
    assert report.anomaly_events[1].anomaly_score == pytest.approx(0.42)


def test_selected_order_and_rank_preserved() -> None:
    report = AnalysisWorkflowReportBuilder().build(_anomaly_result_report()).report
    assert [event.rank for event in report.anomaly_events] == [1, 2]
    assert [event.original_row_id for event in report.anomaly_events] == [13, 21]


def test_operating_row_marked() -> None:
    report = AnalysisWorkflowReportBuilder().build(_anomaly_result_report()).report
    assert report.anomaly_events[0].is_operating_row is True
    assert report.anomaly_events[1].is_operating_row is False


def test_score_direction_and_selection_source_preserved() -> None:
    report = AnalysisWorkflowReportBuilder().build(_anomaly_result_report()).report
    assert report.anomaly_events[0].score_direction == "higher_is_more_anomalous"
    assert report.anomaly_events[0].selection_source == "unsupervised_anomaly_model"
    assert report.anomaly_events[1].selection_source == "top_anomaly_score_candidate"


def test_diagnosis_factor_mapping_preserves_fields() -> None:
    report = AnalysisWorkflowReportBuilder().build(_anomaly_result_report()).report
    assert len(report.diagnosis_factors) == 2
    first = report.diagnosis_factors[0]
    assert first.rank == 1
    assert first.feature_name == "voltage"
    assert first.diagnostic_score == pytest.approx(0.8)
    assert first.direction == "POSITIVE"
    assert first.anomaly_group_value == pytest.approx(4.1)
    assert first.normal_group_value == pytest.approx(3.7)
    assert first.raw_group_difference == pytest.approx(0.4)
    assert first.robust_scale == pytest.approx(1.4826)
    assert first.robust_z_score == pytest.approx(2.5)
    assert first.robust_scale_status == "AVAILABLE"
    assert first.effect_size == pytest.approx(0.4)
    assert first.confidence == pytest.approx(0.8)
    assert first.source == "ROBUST_GROUP_COMPARISON"
    assert report.diagnosis_factors[1].feature_name == "current"
    assert report.diagnosis_factors[1].direction == "NEGATIVE"


def test_diagnosis_factor_zero_variance_mapping() -> None:
    factors = [
        _backend_factor(
            variable="Current",
            score=0.75,
            anomaly_median=106.0,
            reference_median=0.0,
            confidence=0.75,
            robust_z_score=None,
            robust_scale=0.0,
            robust_scale_status="ZERO_VARIANCE",
        ),
        _backend_factor(
            variable="Power",
            direction="NEGATIVE",
            score=0.7,
            anomaly_median=0.0,
            reference_median=10.0,
            confidence=0.7,
            robust_z_score=None,
            robust_scale=0.0,
            robust_scale_status="ZERO_VARIANCE",
        ),
    ]
    source = _anomaly_result_report(
        diagnosis_factor_count=2,
        diagnosis_factors=factors,
    )
    before = source.model_dump(mode="json")
    report = AnalysisWorkflowReportBuilder().build(source).report
    assert source.model_dump(mode="json") == before
    assert len(report.diagnosis_factors) == 2
    first = report.diagnosis_factors[0]
    assert first.feature_name == "Current"
    assert first.robust_z_score is None
    assert first.robust_scale == pytest.approx(0.0)
    assert first.robust_scale_status == "ZERO_VARIANCE"
    assert first.anomaly_group_value == pytest.approx(106.0)
    assert first.normal_group_value == pytest.approx(0.0)
    assert first.raw_group_difference == pytest.approx(106.0)
    assert first.diagnostic_score == pytest.approx(0.75)
    assert math.isfinite(first.diagnostic_score)
    assert 0.0 <= first.diagnostic_score <= 1.0
    assert first.direction == "POSITIVE"
    assert first.confidence == pytest.approx(0.75)
    second = report.diagnosis_factors[1]
    assert second.direction == "NEGATIVE"
    assert second.robust_z_score is None
    dumped = report.model_dump(mode="json")
    encoded = json.dumps(dumped)
    assert "null" in encoded
    assert "1e+14" not in encoded.lower()
    assert "100000000000000" not in encoded
    assert dumped["diagnosis_factors"][0]["robust_scale_status"] == "ZERO_VARIANCE"
    assert dumped["diagnosis_factors"][0]["robust_scale"] == pytest.approx(0.0)


def test_empty_events_and_factors() -> None:
    report = AnalysisWorkflowReportBuilder().build(
        _anomaly_result_report(
            anomaly_event_count=0,
            diagnosis_factor_count=0,
            anomaly_events=[],
            diagnosis_factors=[],
            selected_operating_row_id=None,
        )
    ).report
    assert report.anomaly_events == []
    assert report.diagnosis_factors == []


def test_operating_row_missing_emits_warning() -> None:
    report = AnalysisWorkflowReportBuilder().build(
        _anomaly_result_report(selected_operating_row_id=999)
    ).report
    assert all(not event.is_operating_row for event in report.anomaly_events)
    assert (
        "Selected operating row ID was not present among selected anomaly events."
        in report.warnings
    )


def test_presentation_excludes_raw_dataframe_model_and_path() -> None:
    report = AnalysisWorkflowReportBuilder().build(_anomaly_result_report()).report
    dumped = report.model_dump(mode="json")
    encoded = json.dumps(dumped)
    assert "DataFrame" not in encoded
    assert "IsolationForest" not in encoded
    assert "csv_path" not in report.metadata
    assert "C:/" not in encoded
    assert "contributing_variables" not in encoded


def test_anomaly_only_disclaimer_filtering() -> None:
    report = AnalysisWorkflowReportBuilder().build(_anomaly_result_report()).report
    joined = " ".join(report.disclaimers)
    assert "model-based decision support" in joined
    assert "do not establish causation" in joined
    assert "does not automatically mean a defect" in joined
    assert "Domain verification is required" in joined
    assert "Proposed changes require" not in joined
    assert "Performance acceptance only confirms" not in joined
    assert "Absence of extrapolation" not in joined


def test_supervised_disclaimer_regression() -> None:
    report = AnalysisWorkflowReportBuilder().build(_completed_report()).report
    joined = " ".join(report.disclaimers)
    assert "Proposed changes require" in joined
    assert "Performance acceptance only confirms" in joined
    assert "Absence of extrapolation" in joined


def test_anomaly_presentation_json_safe_and_input_immutable() -> None:
    source = _anomaly_result_report()
    before = copy.deepcopy(source.model_dump(mode="json"))
    report = AnalysisWorkflowReportBuilder().build(source).report
    after = source.model_dump(mode="json")
    assert before == after
    dumped = report.model_dump(mode="json")
    json.dumps(dumped)
    assert report.anomaly_events[0].anomaly_score == pytest.approx(-0.55)


def test_anomaly_context_windows_mapped_from_workflow_report() -> None:
    source = _anomaly_only_report(
        anomaly_events=[
            AnomalyEvent(
                anomaly_id="13",
                anomaly_type=AnomalyType.PROCESS_INPUT,
                anomaly_score=-0.2,
                severity="high",
                sample_id=13,
                model_confidence=0.75,
                detector="unsupervised_anomaly_model",
                rationale="Selected unsupervised anomaly.",
                contributing_variables=[],
            )
        ],
        anomaly_event_count=1,
        selected_operating_row_id=13,
        anomaly_context_windows=[
            AnomalyContextWindow(
                event_rank=1,
                center_original_row_id=13,
                center_anomaly_score=-0.2,
                radius=3,
                order_basis=AnomalyContextOrderBasis.SORTED_ANALYSIS_ORDER,
                feature_names=["RSOCmin", "ChgPmax"],
                rows=[
                    AnomalyContextRow(
                        analysis_position=10,
                        original_row_id=13,
                        relative_offset=0,
                        is_center_event=True,
                        is_selected_anomaly_event=True,
                        feature_values=[
                            AnomalyContextValue(feature_name="RSOCmin", value=61.0),
                            AnomalyContextValue(feature_name="ChgPmax", value=0.0),
                        ],
                    )
                ],
            )
        ],
    )
    before = copy.deepcopy(source.model_dump(mode="json"))
    report = AnalysisWorkflowReportBuilder().build(source).report
    assert source.model_dump(mode="json") == before
    assert len(report.anomaly_context_windows) == 1
    window = report.anomaly_context_windows[0]
    assert window.event_rank == 1
    assert window.center_original_row_id == 13
    assert window.center_anomaly_score == pytest.approx(-0.2)
    assert window.radius == 3
    assert window.order_basis is AnomalyContextOrderBasis.SORTED_ANALYSIS_ORDER
    assert window.feature_names == ["RSOCmin", "ChgPmax"]
    assert window.rows[0].feature_values[0].value == pytest.approx(61.0)
    dumped = report.model_dump(mode="json")
    json.dumps(dumped)
    total_rows = sum(len(item.rows) for item in report.anomaly_context_windows)
    assert total_rows <= 35
    assert report.anomaly_context_windows or report.anomaly_context_windows == []


def test_empty_anomaly_context_windows_mapped() -> None:
    report = AnalysisWorkflowReportBuilder().build(
        _anomaly_only_report(anomaly_context_windows=[])
    ).report
    assert report.anomaly_context_windows == []
