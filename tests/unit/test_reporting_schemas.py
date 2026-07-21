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
    ModelPerformanceView,
    PerformanceMetricView,
    RecommendationChangeView,
    RecommendationView,
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


def _routing(**overrides: Any) -> WorkflowRoutingSummaryView:
    payload: dict[str, Any] = {
        "selected_industry": "semiconductor",
        "selected_task": AnalysisTask.REGRESSION,
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
