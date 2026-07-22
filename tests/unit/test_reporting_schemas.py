"""Unit tests for workflow presentation reporting schemas (Step 11A)."""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.evaluation import (
    MetricAcceptanceDirection,
    ModelPerformanceAcceptanceStatus,
)
from process_intelligence.recommendation import (
    RecommendationSafetyStatus,
    RecommendationStatus,
)
from process_intelligence.reporting import (
    AnomalyContextRowView,
    AnomalyContextValueView,
    AnomalyContextWindowView,
    AnomalyEventView,
    DiagnosisFactorView,
    ModelPerformanceView,
    PerformanceMetricView,
    RecommendationChangeView,
    RecommendationView,
    WorkflowCohortFilterSummaryView,
    WorkflowDataSummaryView,
    WorkflowModelSummaryView,
    WorkflowOverviewView,
    WorkflowPresentationOutcome,
    WorkflowPresentationReport,
    WorkflowRoutingSummaryView,
    WorkflowStageView,
)
from process_intelligence.workflow import (
    AnalysisWorkflowStage,
    AnalysisWorkflowStatus,
    AnomalyContextOrderBasis,
)

_UTC_START = datetime(2026, 7, 21, 10, 0, tzinfo=UTC)
_UTC_END = datetime(2026, 7, 21, 10, 5, tzinfo=UTC)


def _overview(**overrides: Any) -> WorkflowOverviewView:
    payload: dict[str, Any] = {
        "status": AnalysisWorkflowStatus.COMPLETED,
        "terminal_stage": AnalysisWorkflowStage.RECOMMENDATION,
        "headline": "Analysis completed with a generated recommendation",
        "summary": (
            "Workflow terminated at stage RECOMMENDATION with status COMPLETED. "
            "Recommendation status: GENERATED."
        ),
        "started_at": _UTC_START,
        "completed_at": _UTC_END,
        "total_seconds": 300.0,
        "recommendation_status": RecommendationStatus.GENERATED,
    }
    payload.update(overrides)
    return WorkflowOverviewView(**payload)


def _data_summary(**overrides: Any) -> WorkflowDataSummaryView:
    payload: dict[str, Any] = {
        "raw_row_count": 100,
        "processed_row_count": 100,
        "cohort_row_count": 100,
        "train_row_count": 60,
        "validation_row_count": 20,
        "test_row_count": 20,
        "anomaly_event_count": 2,
        "diagnosis_factor_count": 3,
        "selected_operating_row_id": 7,
        "row_identity_preserved": True,
    }
    payload.update(overrides)
    return WorkflowDataSummaryView(**payload)


def _cohort_filter_summary(**overrides: Any) -> WorkflowCohortFilterSummaryView:
    payload: dict[str, Any] = {
        "configured": False,
        "column_name": None,
        "lower_bound": None,
        "upper_bound": None,
        "include_lower": None,
        "include_upper": None,
        "exclude_filter_column_from_features": None,
        "source_row_count": 100,
        "retained_row_count": 100,
        "excluded_row_count": 0,
        "null_excluded_count": 0,
        "range_display": "Not configured",
        "filter_column_used_as_feature_display": "Not applicable",
    }
    payload.update(overrides)
    return WorkflowCohortFilterSummaryView(**payload)


def _routing(**overrides: Any) -> WorkflowRoutingSummaryView:
    payload: dict[str, Any] = {
        "selected_industry": "semiconductor",
        "selected_task": AnalysisTask.REGRESSION,
        "inferred_task": AnalysisTask.REGRESSION,
        "task_selection_source": "ROUTER",
        "task_override_applied": False,
        "target_column": None,
        "feature_count": None,
    }
    payload.update(overrides)
    return WorkflowRoutingSummaryView(**payload)


def _model_summary(**overrides: Any) -> WorkflowModelSummaryView:
    payload: dict[str, Any] = {
        "supervised_model_key": "ridge",
        "anomaly_model_key": "isolation_forest",
        "independent_test_evaluation_performed": True,
        "residual_calibration_performed": True,
        "test_used_for_model_selection": False,
        "test_used_for_threshold_calibration": False,
        "anomaly_score_direction": "higher_is_more_anomalous",
    }
    payload.update(overrides)
    return WorkflowModelSummaryView(**payload)


def _metric(**overrides: Any) -> PerformanceMetricView:
    payload: dict[str, Any] = {
        "metric_name": "rmse",
        "observed_value": 1.2,
        "threshold": 10.0,
        "direction": MetricAcceptanceDirection.LOWER_IS_BETTER,
        "required": True,
        "available": True,
        "passed": True,
        "message": "Metric rmse=1.2 meets LOWER_IS_BETTER threshold 10.0.",
    }
    payload.update(overrides)
    return PerformanceMetricView(**payload)


def _performance(**overrides: Any) -> ModelPerformanceView:
    payload: dict[str, Any] = {
        "status": ModelPerformanceAcceptanceStatus.ACCEPTABLE,
        "evaluation_available": True,
        "independent_test_evaluation": True,
        "test_row_count": 20,
        "metrics": [
            _metric(),
            _metric(
                metric_name="mae",
                observed_value=0.9,
                message="Metric mae=0.9 meets LOWER_IS_BETTER threshold 10.0.",
            ),
        ],
        "required_rule_count": 2,
        "passed_required_rule_count": 2,
        "failed_required_rule_count": 0,
        "unavailable_required_rule_count": 0,
        "assessed_at": datetime(2026, 7, 21, 10, 2, tzinfo=UTC),
        "warnings": [],
    }
    payload.update(overrides)
    return ModelPerformanceView(**payload)


def _stage(
    sequence: int = 1,
    stage: AnalysisWorkflowStage = AnalysisWorkflowStage.LOAD,
    **overrides: Any,
) -> WorkflowStageView:
    payload: dict[str, Any] = {
        "sequence": sequence,
        "stage": stage,
        "executed": True,
        "succeeded": True,
        "structured_refusal": False,
        "status_label": "SUCCEEDED",
        "row_count": 100,
        "message": f"{stage.value} stage completed.",
        "warnings": [],
        "metadata": {"column_count": 10},
    }
    payload.update(overrides)
    return WorkflowStageView(**payload)


def _change(**overrides: Any) -> RecommendationChangeView:
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
    return RecommendationChangeView(**payload)


def _recommendation(**overrides: Any) -> RecommendationView:
    payload: dict[str, Any] = {
        "status": RecommendationStatus.GENERATED,
        "changes": [_change()],
        "confidence": 0.5,
        "baseline_prediction": 80.0,
        "proposed_prediction": 82.0,
        "baseline_anomaly_score": -0.35,
        "proposed_anomaly_score": -0.55,
        "extrapolation_flag": False,
        "uncertainty_available": False,
        "generated_at": datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
        "disclaimer": (
            "These recommendations are model-based decision support. "
            "Diagnosis reflects association and not established causation. "
            "Proposed process changes require domain, safety, and operational "
            "verification."
        ),
        "warnings": ["Verify operationally before applying."],
        "safety_status": RecommendationSafetyStatus.APPROVED,
        "safety_messages": ["Safety checks passed."],
    }
    payload.update(overrides)
    return RecommendationView(**payload)


def _presentation(**overrides: Any) -> WorkflowPresentationReport:
    payload: dict[str, Any] = {
        "overview": _overview(),
        "data_summary": _data_summary(),
        "cohort_filter_summary": _cohort_filter_summary(),
        "routing_summary": _routing(),
        "model_summary": _model_summary(),
        "model_performance": _performance(),
        "stages": [
            _stage(1, AnalysisWorkflowStage.LOAD),
            _stage(2, AnalysisWorkflowStage.PROFILE),
        ],
        "recommendation": _recommendation(),
        "warnings": ["Recommendations are decision support, not proven causes."],
        "disclaimers": [
            "These outputs are model-based decision support and not operational "
            "commands."
        ],
        "metadata": {"row_identity_preserved": True},
    }
    payload.update(overrides)
    return WorkflowPresentationReport(**payload)


def test_each_view_constructs_normally() -> None:
    assert _overview().status is AnalysisWorkflowStatus.COMPLETED
    assert _data_summary().raw_row_count == 100
    assert _routing().selected_industry == "semiconductor"
    assert _model_summary().supervised_model_key == "ridge"
    assert _metric().metric_name == "rmse"
    assert _performance().status is ModelPerformanceAcceptanceStatus.ACCEPTABLE
    assert _stage().status_label == "SUCCEEDED"
    assert _change().delta == -2.0
    assert _recommendation().baseline_anomaly_score == pytest.approx(-0.35)
    report = _presentation()
    assert report.overview.status is AnalysisWorkflowStatus.COMPLETED


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        WorkflowOverviewView(
            **{
                **_overview().model_dump(),
                "unexpected": True,
            }
        )
    with pytest.raises(ValidationError):
        WorkflowDataSummaryView(
            **{
                **_data_summary().model_dump(),
                "extra": 1,
            }
        )


def test_bool_int_strict_validation() -> None:
    with pytest.raises(ValidationError):
        _data_summary(raw_row_count=True)
    with pytest.raises(ValidationError):
        _data_summary(row_identity_preserved=1)
    with pytest.raises(ValidationError):
        _stage(sequence=True)
    with pytest.raises(ValidationError):
        _model_summary(test_used_for_model_selection=1)


def test_nan_inf_rejected() -> None:
    with pytest.raises(ValidationError):
        _overview(total_seconds=float("nan"))
    with pytest.raises(ValidationError):
        _change(current_value=float("inf"))
    with pytest.raises(ValidationError):
        _recommendation(baseline_anomaly_score=float("nan"))
    with pytest.raises(ValidationError):
        _metric(threshold=float("-inf"))


def test_empty_strings_rejected() -> None:
    with pytest.raises(ValidationError):
        _overview(headline="   ")
    with pytest.raises(ValidationError):
        _stage(message="")
    with pytest.raises(ValidationError):
        _change(rationale="")
    with pytest.raises(ValidationError):
        _presentation(warnings=[""])


def test_mutable_state_independence() -> None:
    warnings = ["alpha"]
    metadata = {"row_identity_preserved": True}
    report = _presentation(warnings=warnings, metadata=metadata)
    warnings.append("beta")
    metadata["injected"] = "nope"
    assert report.warnings == ["alpha"]
    assert "injected" not in report.metadata


def test_data_summary_count_relationship() -> None:
    with pytest.raises(ValidationError):
        _data_summary(processed_row_count=101)
    with pytest.raises(ValidationError):
        _data_summary(train_row_count=50, validation_row_count=20, test_row_count=20)
    early = _data_summary(
        train_row_count=0,
        validation_row_count=0,
        test_row_count=0,
        processed_row_count=100,
    )
    assert early.processed_row_count == 100


def test_performance_metric_relationship() -> None:
    with pytest.raises(ValidationError):
        _metric(available=False, observed_value=1.0, passed=None)
    with pytest.raises(ValidationError):
        _metric(available=True, observed_value=None, passed=True)
    unavailable = _metric(
        available=False,
        observed_value=None,
        passed=None,
        message="Required metric missing.",
    )
    assert unavailable.passed is None


def test_stage_status_label_relationships() -> None:
    skipped = _stage(
        executed=False,
        succeeded=False,
        structured_refusal=False,
        status_label="SKIPPED",
    )
    assert skipped.status_label == "SKIPPED"
    refused = _stage(
        executed=True,
        succeeded=False,
        structured_refusal=True,
        status_label="REFUSED",
    )
    assert refused.status_label == "REFUSED"
    incomplete = _stage(
        executed=True,
        succeeded=False,
        structured_refusal=False,
        status_label="INCOMPLETE",
    )
    assert incomplete.status_label == "INCOMPLETE"
    with pytest.raises(ValidationError):
        _stage(
            executed=False,
            succeeded=False,
            structured_refusal=False,
            status_label="SUCCEEDED",
        )


def test_stage_sequence_rejects_zero_and_bool() -> None:
    with pytest.raises(ValidationError):
        _stage(sequence=0)
    with pytest.raises(ValidationError):
        _stage(sequence=False)


def test_recommendation_negative_anomaly_score_allowed() -> None:
    view = _recommendation(
        baseline_anomaly_score=-12.5,
        proposed_anomaly_score=-20.0,
    )
    assert view.baseline_anomaly_score == pytest.approx(-12.5)
    assert view.proposed_anomaly_score == pytest.approx(-20.0)


def test_recommendation_nan_inf_rejected() -> None:
    with pytest.raises(ValidationError):
        _recommendation(proposed_prediction=float("inf"))
    with pytest.raises(ValidationError):
        _change(delta=float("nan"))


def test_generated_requires_changes() -> None:
    with pytest.raises(ValidationError):
        _recommendation(changes=[])
    ready = _recommendation(
        status=RecommendationStatus.READY_FOR_OPTIMIZATION,
        changes=[],
        proposed_prediction=None,
        proposed_anomaly_score=None,
    )
    assert ready.changes == []


def test_presentation_stage_sequence_and_duplicates() -> None:
    with pytest.raises(ValidationError):
        _presentation(
            stages=[
                _stage(1, AnalysisWorkflowStage.LOAD),
                _stage(3, AnalysisWorkflowStage.PROFILE),
            ]
        )
    with pytest.raises(ValidationError):
        _presentation(
            stages=[
                _stage(1, AnalysisWorkflowStage.LOAD),
                _stage(2, AnalysisWorkflowStage.LOAD),
            ]
        )


def test_completed_requires_generated_recommendation() -> None:
    with pytest.raises(ValidationError):
        _presentation(recommendation=None)
    with pytest.raises(ValidationError):
        _presentation(
            recommendation=_recommendation(
                status=RecommendationStatus.READY_FOR_OPTIMIZATION,
                changes=[],
                proposed_prediction=None,
                proposed_anomaly_score=None,
            )
        )


def test_partial_recommendation_relationship() -> None:
    report = _presentation(
        overview=_overview(
            status=AnalysisWorkflowStatus.PARTIAL,
            headline="Analysis completed without an executable recommendation",
            summary=(
                "Workflow terminated at stage RECOMMENDATION with status PARTIAL. "
                "Recommendation status: READY_FOR_OPTIMIZATION."
            ),
            recommendation_status=RecommendationStatus.READY_FOR_OPTIMIZATION,
        ),
        recommendation=_recommendation(
            status=RecommendationStatus.READY_FOR_OPTIMIZATION,
            changes=[],
            proposed_prediction=None,
            proposed_anomaly_score=None,
        ),
    )
    assert report.recommendation is not None
    assert report.recommendation.status is RecommendationStatus.READY_FOR_OPTIMIZATION
    with pytest.raises(ValidationError):
        _presentation(
            overview=_overview(
                status=AnalysisWorkflowStatus.PARTIAL,
                headline="Analysis completed without an executable recommendation",
                summary="Partial without recommendation.",
                recommendation_status=None,
            ),
            recommendation=_recommendation(),
        )


def test_refused_recommendation_relationship() -> None:
    report = _presentation(
        overview=_overview(
            status=AnalysisWorkflowStatus.REFUSED,
            terminal_stage=AnalysisWorkflowStage.VALIDATE,
            headline="Analysis was stopped by a safety or validation gate",
            summary=(
                "Workflow terminated at stage VALIDATE with status REFUSED. "
                "Recommendation status: none."
            ),
            recommendation_status=None,
        ),
        recommendation=None,
        model_performance=None,
    )
    assert report.recommendation is None
    refused = _presentation(
        overview=_overview(
            status=AnalysisWorkflowStatus.REFUSED,
            headline="Analysis was stopped by a safety or validation gate",
            summary=(
                "Workflow terminated at stage RECOMMENDATION with status REFUSED. "
                "Recommendation status: REFUSED."
            ),
            recommendation_status=RecommendationStatus.REFUSED,
        ),
        recommendation=_recommendation(
            status=RecommendationStatus.REFUSED,
            changes=[],
            proposed_prediction=None,
            proposed_anomaly_score=None,
            safety_status=RecommendationSafetyStatus.REFUSED,
            safety_messages=["Refused."],
            confidence=0.0,
        ),
    )
    assert refused.recommendation is not None
    assert refused.recommendation.status is RecommendationStatus.REFUSED


def test_warning_and_disclaimer_duplicates_rejected() -> None:
    with pytest.raises(ValidationError):
        _presentation(warnings=["alpha", "alpha"])
    with pytest.raises(ValidationError):
        _presentation(disclaimers=["one", "one"])


def test_routing_summary_preserves_override_fields() -> None:
    routing = _routing(
        inferred_task=AnalysisTask.CLASSIFICATION,
        selected_task=AnalysisTask.REGRESSION,
        task_selection_source="USER_OVERRIDE",
        task_override_applied=True,
    )
    assert routing.inferred_task is AnalysisTask.CLASSIFICATION
    assert routing.selected_task is AnalysisTask.REGRESSION
    assert routing.task_selection_source == "USER_OVERRIDE"
    assert routing.task_override_applied is True
    restored = WorkflowRoutingSummaryView.model_validate(routing.model_dump())
    assert restored == routing


def test_metadata_scalar_only() -> None:
    with pytest.raises(ValidationError):
        _presentation(metadata={"bad": {"nested": 1}})
    with pytest.raises(ValidationError):
        _stage(metadata={"arr": [1, 2]})


def test_model_dump_json_and_round_trip() -> None:
    report = _presentation()
    dumped = report.model_dump(mode="json")
    assert isinstance(dumped["overview"]["started_at"], str)
    assert dumped["overview"]["status"] == "COMPLETED"
    assert dumped["recommendation"]["baseline_anomaly_score"] == pytest.approx(-0.35)
    encoded = json.dumps(dumped)
    decoded = json.loads(encoded)
    restored = WorkflowPresentationReport.model_validate(decoded)
    assert restored.overview.status is AnalysisWorkflowStatus.COMPLETED
    assert restored.recommendation is not None
    assert restored.recommendation.baseline_anomaly_score == pytest.approx(-0.35)


def test_json_serialization_enums_and_datetimes() -> None:
    report = _presentation()
    payload = report.model_dump(mode="json")
    assert payload["overview"]["terminal_stage"] == "RECOMMENDATION"
    assert payload["model_performance"]["status"] == "ACCEPTABLE"
    assert "T" in payload["overview"]["completed_at"]


def test_frozen_outcome_and_slots() -> None:
    outcome = WorkflowPresentationOutcome(report=_presentation())
    assert outcome.report.overview.status is AnalysisWorkflowStatus.COMPLETED
    assert dataclasses.is_dataclass(outcome)
    assert outcome.__slots__ == ("report",)
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.report = _presentation(  # type: ignore[misc]
            warnings=["mutated"],
        )


def test_model_summary_safety_invariants() -> None:
    with pytest.raises(ValidationError):
        _model_summary(test_used_for_model_selection=True)
    with pytest.raises(ValidationError):
        _model_summary(test_used_for_threshold_calibration=True)
    with pytest.raises(ValidationError):
        _model_summary(anomaly_score_direction="lower_is_more_anomalous")


def test_overview_forbidden_wording() -> None:
    with pytest.raises(ValidationError):
        _overview(summary="This will improve the process.")
    with pytest.raises(ValidationError):
        _overview(
            headline="Analysis completed with a generated recommendation",
            summary="Confirmed root cause identified.",
        )


def test_finite_float_relationships() -> None:
    change = _change(current_value=10.0, proposed_value=12.5, delta=2.5)
    assert math.isclose(change.delta, change.proposed_value - change.current_value)
    with pytest.raises(ValidationError):
        _change(current_value=10.0, proposed_value=12.5, delta=1.0)


# ---------------------------------------------------------------------------
# ANOMALY_ONLY analysis-mode presentation schema coverage (Step 11B.5)
# ---------------------------------------------------------------------------

_ANOMALY_ONLY_OVERVIEW = (
    "Anomaly-only analysis completed. Recommendation generation is not "
    "enabled for this analysis mode."
)


def test_anomaly_only_overview_message_accepted() -> None:
    overview = _overview(
        status=AnalysisWorkflowStatus.PARTIAL,
        terminal_stage=AnalysisWorkflowStage.DIAGNOSIS,
        headline="Analysis completed without an executable recommendation",
        summary=_ANOMALY_ONLY_OVERVIEW,
        recommendation_status=None,
    )
    assert overview.summary == _ANOMALY_ONLY_OVERVIEW
    assert overview.status is AnalysisWorkflowStatus.PARTIAL


def test_anomaly_only_routing_and_model_summary_not_applicable() -> None:
    routing = _routing(
        selected_task=None,
        inferred_task=None,
        task_selection_source=None,
        target_column=None,
        feature_count=5,
        target_suitable=None,
        target_suitability_message="NOT_APPLICABLE",
    )
    assert routing.selected_task is None
    assert routing.target_column is None
    assert routing.target_suitability_message == "NOT_APPLICABLE"

    model_summary = _model_summary(
        supervised_model_key=None,
        anomaly_model_key="isolation_forest",
        residual_calibration_performed=False,
    )
    assert model_summary.supervised_model_key is None
    assert model_summary.anomaly_model_key == "isolation_forest"
    assert model_summary.residual_calibration_performed is False


def test_anomaly_only_presentation_report_without_recommendation() -> None:
    report = _presentation(
        overview=_overview(
            status=AnalysisWorkflowStatus.PARTIAL,
            terminal_stage=AnalysisWorkflowStage.DIAGNOSIS,
            headline="Analysis completed without an executable recommendation",
            summary=_ANOMALY_ONLY_OVERVIEW,
            recommendation_status=None,
        ),
        routing_summary=_routing(
            selected_task=None,
            inferred_task=None,
            task_selection_source=None,
            target_column=None,
            feature_count=5,
        ),
        model_summary=_model_summary(
            supervised_model_key=None,
            anomaly_model_key="isolation_forest",
            residual_calibration_performed=False,
        ),
        model_performance=None,
        stages=[
            _stage(1, AnalysisWorkflowStage.LOAD),
            _stage(
                2,
                AnalysisWorkflowStage.TASK_ROUTING,
                executed=False,
                succeeded=False,
                structured_refusal=False,
                status_label="SKIPPED",
                row_count=None,
                message="Not applicable for anomaly-only analysis.",
                metadata={},
            ),
        ],
        recommendation=None,
        metadata={
            "row_identity_preserved": True,
            "analysis_mode": "ANOMALY_ONLY",
            "recommendation_applicable": False,
        },
    )
    assert report.overview.summary == _ANOMALY_ONLY_OVERVIEW
    assert report.recommendation is None
    assert report.model_performance is None
    assert report.metadata["analysis_mode"] == "ANOMALY_ONLY"
    assert report.metadata["recommendation_applicable"] is False
    skipped = next(
        stage for stage in report.stages if stage.stage is AnalysisWorkflowStage.TASK_ROUTING
    )
    assert skipped.status_label == "SKIPPED"
    assert skipped.executed is False


def test_anomaly_only_metadata_round_trip_json() -> None:
    report = _presentation(
        overview=_overview(
            status=AnalysisWorkflowStatus.PARTIAL,
            terminal_stage=AnalysisWorkflowStage.DIAGNOSIS,
            headline="Analysis completed without an executable recommendation",
            summary=_ANOMALY_ONLY_OVERVIEW,
            recommendation_status=None,
        ),
        model_performance=None,
        recommendation=None,
        metadata={"row_identity_preserved": True, "analysis_mode": "ANOMALY_ONLY"},
    )
    dumped = report.model_dump(mode="json")
    encoded = json.dumps(dumped)
    decoded = json.loads(encoded)
    restored = WorkflowPresentationReport.model_validate(decoded)
    assert restored.metadata["analysis_mode"] == "ANOMALY_ONLY"
    assert restored.overview.summary == _ANOMALY_ONLY_OVERVIEW
    assert restored.recommendation is None


def test_supervised_presentation_regression_still_passes() -> None:
    # Guards against ANOMALY_ONLY schema additions affecting the default
    # SUPERVISED-shaped presentation report exercised throughout this module.
    report = _presentation()
    assert report.overview.status is AnalysisWorkflowStatus.COMPLETED
    assert report.recommendation is not None
    assert report.model_performance is not None
    assert report.model_summary.supervised_model_key == "ridge"


# ---------------------------------------------------------------------------
# Anomaly result presentation views (Step 11B.6)
# ---------------------------------------------------------------------------


def _anomaly_event(**overrides: Any) -> AnomalyEventView:
    payload: dict[str, Any] = {
        "rank": 1,
        "original_row_id": 13,
        "anomaly_score": 0.42,
        "is_operating_row": True,
        "selection_source": "unsupervised_anomaly_model",
        "score_direction": "higher_is_more_anomalous",
        "is_anomaly_flagged": True,
    }
    payload.update(overrides)
    return AnomalyEventView(**payload)


def _diagnosis_factor(**overrides: Any) -> DiagnosisFactorView:
    payload: dict[str, Any] = {
        "rank": 1,
        "feature_name": "voltage",
        "diagnostic_score": 0.8,
        "direction": "POSITIVE",
        "anomaly_group_value": 4.1,
        "normal_group_value": 3.7,
        "raw_group_difference": 0.4,
        "robust_scale": 1.4826,
        "robust_z_score": 2.5,
        "robust_scale_status": "AVAILABLE",
        "effect_size": 0.4,
        "confidence": 0.8,
        "source": "ROBUST_GROUP_COMPARISON",
    }
    payload.update(overrides)
    return DiagnosisFactorView(**payload)


def test_valid_anomaly_event_view() -> None:
    event = _anomaly_event()
    assert event.rank == 1
    assert event.original_row_id == 13
    assert event.anomaly_score == pytest.approx(0.42)
    assert event.is_operating_row is True


def test_finite_anomaly_score_enforced() -> None:
    event = _anomaly_event(anomaly_score=-1.25)
    assert event.anomaly_score == pytest.approx(-1.25)


def test_invalid_nan_anomaly_score_rejected() -> None:
    with pytest.raises(ValidationError):
        _anomaly_event(anomaly_score=math.nan)


def test_original_row_id_json_safe() -> None:
    assert _anomaly_event(original_row_id="LOT-13").original_row_id == "LOT-13"
    with pytest.raises(ValidationError):
        _anomaly_event(original_row_id=True)
    with pytest.raises(ValidationError):
        _anomaly_event(original_row_id=None)


def test_operating_row_bool_strict() -> None:
    with pytest.raises(ValidationError):
        _anomaly_event(is_operating_row=1)
    with pytest.raises(ValidationError):
        _anomaly_event(is_operating_row="yes")


def test_diagnosis_factor_view_valid() -> None:
    factor = _diagnosis_factor()
    assert factor.feature_name == "voltage"
    assert factor.direction == "POSITIVE"
    assert factor.source == "ROBUST_GROUP_COMPARISON"


def test_diagnosis_factor_optional_values_none() -> None:
    factor = _diagnosis_factor(
        diagnostic_score=None,
        direction=None,
        anomaly_group_value=None,
        normal_group_value=None,
        raw_group_difference=None,
        robust_scale=None,
        robust_z_score=None,
        robust_scale_status=None,
        effect_size=None,
        confidence=None,
        source=None,
    )
    assert factor.diagnostic_score is None
    assert factor.direction is None
    assert factor.confidence is None
    assert factor.robust_scale is None
    assert factor.robust_scale_status is None
    assert factor.raw_group_difference is None


def test_diagnosis_factor_zero_variance_fields() -> None:
    factor = _diagnosis_factor(
        diagnostic_score=0.75,
        robust_scale=0.0,
        robust_z_score=None,
        robust_scale_status="ZERO_VARIANCE",
        raw_group_difference=106.0,
        anomaly_group_value=106.0,
        normal_group_value=0.0,
    )
    assert factor.robust_z_score is None
    assert factor.robust_scale == pytest.approx(0.0)
    assert factor.robust_scale_status == "ZERO_VARIANCE"
    assert factor.raw_group_difference == pytest.approx(106.0)
    dumped = factor.model_dump(mode="json")
    assert dumped["robust_z_score"] is None
    assert dumped["robust_scale"] == pytest.approx(0.0)
    assert dumped["robust_scale_status"] == "ZERO_VARIANCE"
    assert math.isfinite(dumped["diagnostic_score"])
    assert 0.0 <= dumped["diagnostic_score"] <= 1.0


def test_diagnosis_factor_ranking_score_bounded() -> None:
    factor = _diagnosis_factor(diagnostic_score=0.9936, confidence=0.9936)
    assert 0.0 <= factor.diagnostic_score <= 1.0
    with pytest.raises(ValidationError):
        _diagnosis_factor(diagnostic_score=2.5)
    with pytest.raises(ValidationError):
        _diagnosis_factor(diagnostic_score=-0.1)
    with pytest.raises(ValidationError):
        _diagnosis_factor(robust_scale=-1.0)


def test_invalid_infinity_rejected_on_factor_fields() -> None:
    with pytest.raises(ValidationError):
        _diagnosis_factor(diagnostic_score=math.inf)
    with pytest.raises(ValidationError):
        _diagnosis_factor(robust_z_score=-math.inf)
    with pytest.raises(ValidationError):
        _diagnosis_factor(robust_scale=math.inf)
    with pytest.raises(ValidationError):
        _anomaly_event(anomaly_score=math.inf)


def test_anomaly_presentation_views_immutable_schema() -> None:
    event = _anomaly_event()
    factor = _diagnosis_factor()
    with pytest.raises(ValidationError):
        event.model_validate({**event.model_dump(), "extra": 1})
    with pytest.raises(ValidationError):
        factor.model_validate({**factor.model_dump(), "extra": True})


def test_anomaly_presentation_json_serialization() -> None:
    report = _presentation(
        anomaly_events=[
            _anomaly_event(rank=1, original_row_id=13, anomaly_score=-0.5),
            _anomaly_event(
                rank=2,
                original_row_id=21,
                anomaly_score=0.1,
                is_operating_row=False,
            ),
        ],
        diagnosis_factors=[
            _diagnosis_factor(rank=1),
            _diagnosis_factor(rank=2, feature_name="current", direction="NEGATIVE"),
        ],
    )
    dumped = report.model_dump(mode="json")
    encoded = json.dumps(dumped)
    restored = WorkflowPresentationReport.model_validate(json.loads(encoded))
    assert restored.anomaly_events[0].anomaly_score == pytest.approx(-0.5)
    assert restored.diagnosis_factors[1].feature_name == "current"
    assert dataclasses.is_dataclass(WorkflowPresentationOutcome(report=restored))


def _context_window_view(**overrides: Any) -> AnomalyContextWindowView:
    payload: dict[str, Any] = {
        "event_rank": 1,
        "center_original_row_id": 13,
        "center_anomaly_score": -0.2,
        "radius": 3,
        "order_basis": AnomalyContextOrderBasis.LOADED_ROW_ORDER,
        "feature_names": ["RSOCmin"],
        "rows": [
            AnomalyContextRowView(
                analysis_position=10,
                original_row_id=13,
                relative_offset=0,
                is_center_event=True,
                is_selected_anomaly_event=True,
                feature_values=[
                    AnomalyContextValueView(feature_name="RSOCmin", value=55.0)
                ],
            )
        ],
    }
    payload.update(overrides)
    return AnomalyContextWindowView(**payload)


def test_anomaly_context_window_view_round_trip() -> None:
    window = _context_window_view()
    dumped = window.model_dump(mode="json")
    restored = AnomalyContextWindowView.model_validate(dumped)
    assert restored.radius == 3
    assert restored.rows[0].relative_offset == 0
    report = _presentation(anomaly_context_windows=[window])
    encoded = json.dumps(report.model_dump(mode="json"))
    assert "RSOCmin" in encoded
    assert "DataFrame" not in encoded
