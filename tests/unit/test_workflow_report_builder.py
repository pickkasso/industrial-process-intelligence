"""Unit tests for AnalysisWorkflowReportBuilder (Step 11A)."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from typing import Any

import pytest

from process_intelligence.core.enums import AnalysisTask, ColumnRole
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
    AnalysisWorkflowReport,
    AnalysisWorkflowStage,
    AnalysisWorkflowStageRecord,
    AnalysisWorkflowStatus,
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
        "train_row_count": 60,
        "validation_row_count": 20,
        "test_row_count": 20,
        "selected_industry": "semiconductor",
        "selected_task": "REGRESSION",
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
        "selected_supervised_model_key": "ridge",
        "selected_anomaly_model_key": "isolation_forest",
        "selected_operating_row_id": 7,
        "anomaly_event_count": 2,
        "diagnosis_factor_count": 3,
        "raw_row_count": 100,
        "processed_row_count": 100,
        "train_row_count": 60,
        "validation_row_count": 20,
        "test_row_count": 20,
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
        "selected_supervised_model_key": "ridge",
        "selected_anomaly_model_key": "isolation_forest",
        "selected_operating_row_id": 7,
        "anomaly_event_count": 2,
        "diagnosis_factor_count": 3,
        "raw_row_count": 100,
        "processed_row_count": 100,
        "train_row_count": 60,
        "validation_row_count": 20,
        "test_row_count": 20,
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
        "train_row_count": 0,
        "validation_row_count": 0,
        "test_row_count": 0,
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
