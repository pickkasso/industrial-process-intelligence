"""Unit tests for built-in-demo ground-truth evaluation helpers."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pandas as pd  # type: ignore[import-untyped]
import pytest

from process_intelligence.demo_evaluation import overlap_precision_and_enrichment
from process_intelligence.reporting.schemas import AnomalyEventView
from process_intelligence.ui.demo_configuration import (
    DATA_SOURCE_BUILTIN_DEMO,
    DATA_SOURCE_UPLOAD_CSV,
)
from process_intelligence.ui.demo_evaluation import (
    DemoAnomalyEvaluationStatus,
    evaluate_demo_anomaly_events,
    evaluate_demo_anomaly_presentation,
    format_enrichment,
    format_percentage,
    is_demo_evaluation_panel_eligible,
    select_displayed_events_for_evaluation,
    summary_contains_no_runtime_objects,
)
from process_intelligence.workflow.enums import AnalysisExecutionMode

_SCORE_DIRECTION = "higher_is_more_anomalous"
_FINGERPRINT_A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
_FINGERPRINT_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def _event(
    *,
    rank: int,
    original_row_id: int | str,
    anomaly_score: float = 0.5,
    selection_source: str = "unsupervised_anomaly_model",
) -> AnomalyEventView:
    return AnomalyEventView(
        rank=rank,
        original_row_id=original_row_id,
        anomaly_score=anomaly_score,
        is_operating_row=False,
        selection_source=selection_source,
        score_direction=_SCORE_DIRECTION,
        is_anomaly_flagged=True,
    )


def _demo_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 10, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 15, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 20, tzinfo=UTC),
            ],
            "temperature_setpoint": [180.0, 181.0, 182.0, 183.0, 184.0],
            "injected_anomaly": [False, True, False, True, False],
            "anomaly_type": ["", "temperature_drift", "", "pressure_spike", ""],
        }
    )


def test_exact_matching_by_original_row_id() -> None:
    summary = evaluate_demo_anomaly_events(
        events=[_event(rank=1, original_row_id=1), _event(rank=2, original_row_id=2)],
        demo_frame=_demo_frame(),
    )
    assert summary.evaluation_status is DemoAnomalyEvaluationStatus.AVAILABLE
    assert summary.event_details[0].ground_truth_match is True
    assert summary.event_details[0].anomaly_type == "temperature_drift"
    assert summary.event_details[1].ground_truth_match is False


def test_matched_and_unmatched_event_representatives() -> None:
    summary = evaluate_demo_anomaly_events(
        events=[
            _event(rank=1, original_row_id=1),
            _event(rank=2, original_row_id=0),
            _event(rank=3, original_row_id=3),
        ],
        demo_frame=_demo_frame(),
    )
    assert summary.evaluated_event_count == 3
    assert summary.matched_event_count == 2
    assert summary.unmatched_event_count == 1


def test_selected_event_precision_calculation() -> None:
    summary = evaluate_demo_anomaly_events(
        events=[
            _event(rank=1, original_row_id=1),
            _event(rank=2, original_row_id=0),
            _event(rank=3, original_row_id=2),
            _event(rank=4, original_row_id=3),
        ],
        demo_frame=_demo_frame(),
    )
    assert summary.selected_event_precision == pytest.approx(0.5)


def test_dataset_anomaly_prevalence_calculation() -> None:
    summary = evaluate_demo_anomaly_events(
        events=[_event(rank=1, original_row_id=1)],
        demo_frame=_demo_frame(),
    )
    assert summary.dataset_anomaly_row_count == 2
    assert summary.dataset_anomaly_prevalence == pytest.approx(0.4)


def test_enrichment_calculation() -> None:
    summary = evaluate_demo_anomaly_events(
        events=[
            _event(rank=1, original_row_id=1),
            _event(rank=2, original_row_id=3),
        ],
        demo_frame=_demo_frame(),
    )
    assert summary.selected_event_precision == pytest.approx(1.0)
    assert summary.enrichment_factor == pytest.approx(1.0 / 0.4)


def test_shared_overlap_helper_matches_ui_metrics() -> None:
    precision, enrichment = overlap_precision_and_enrichment(
        detected_ids=[1, 3],
        ground_truth_ids={1, 3},
        baseline_rate=0.4,
    )
    summary = evaluate_demo_anomaly_events(
        events=[
            _event(rank=1, original_row_id=1),
            _event(rank=2, original_row_id=3),
        ],
        demo_frame=_demo_frame(),
    )
    assert precision == summary.selected_event_precision
    assert enrichment == summary.enrichment_factor


def test_zero_displayed_events() -> None:
    summary = evaluate_demo_anomaly_events(events=[], demo_frame=_demo_frame())
    assert summary.evaluation_status is DemoAnomalyEvaluationStatus.NO_DISPLAYED_EVENTS
    assert summary.selected_event_precision is None
    assert summary.enrichment_factor is None
    assert any("No anomaly-event representatives" in msg for msg in summary.explanatory_messages)


def test_zero_injected_anomaly_rows() -> None:
    frame = _demo_frame()
    frame["injected_anomaly"] = False
    frame["anomaly_type"] = ""
    summary = evaluate_demo_anomaly_events(
        events=[_event(rank=1, original_row_id=0)],
        demo_frame=frame,
    )
    assert (
        summary.evaluation_status
        is DemoAnomalyEvaluationStatus.NO_GROUND_TRUTH_ANOMALIES
    )
    assert summary.selected_event_precision is None
    assert summary.enrichment_factor is None


def test_invalid_sample_ids() -> None:
    summary = evaluate_demo_anomaly_events(
        events=[
            _event(rank=1, original_row_id="row-1"),
            _event(rank=2, original_row_id=99),
        ],
        demo_frame=_demo_frame(),
    )
    assert summary.evaluation_status is DemoAnomalyEvaluationStatus.INVALID_SAMPLE_IDS
    assert summary.evaluated_event_count == 0
    assert all(not detail.sample_id_valid for detail in summary.event_details)


def test_duplicate_displayed_sample_ids_handled_deterministically() -> None:
    summary = evaluate_demo_anomaly_events(
        events=[
            _event(rank=1, original_row_id=1, anomaly_score=0.9),
            _event(rank=2, original_row_id=1, anomaly_score=0.1),
            _event(rank=3, original_row_id=0, anomaly_score=0.2),
        ],
        demo_frame=_demo_frame(),
    )
    assert summary.evaluated_event_count == 2
    assert summary.matched_event_count == 1
    assert summary.selected_event_precision == pytest.approx(0.5)
    assert summary.event_details[0].anomaly_score == pytest.approx(0.9)
    assert "Duplicate" in summary.event_details[1].evaluation_note


def test_anomaly_type_collection() -> None:
    summary = evaluate_demo_anomaly_events(
        events=[
            _event(rank=1, original_row_id=1),
            _event(rank=2, original_row_id=3),
        ],
        demo_frame=_demo_frame(),
    )
    assert summary.matched_anomaly_types == ("pressure_spike", "temperature_drift")
    assert "temperature_drift" in summary.available_anomaly_types
    assert "pressure_spike" in summary.available_anomaly_types


def test_timestamps_preserved_when_available() -> None:
    summary = evaluate_demo_anomaly_events(
        events=[_event(rank=1, original_row_id=1)],
        demo_frame=_demo_frame(),
    )
    assert summary.event_details[0].timestamp is not None
    assert "2024-01-01" in summary.event_details[0].timestamp


def test_non_finite_event_scores_handled_safely() -> None:
    # AnomalyEventView forbids non-finite scores; helper still sanitizes values.
    detail_score = evaluate_demo_anomaly_events(
        events=[_event(rank=1, original_row_id=1, anomaly_score=1.5)],
        demo_frame=_demo_frame(),
    ).event_details[0].anomaly_score
    assert detail_score is not None
    assert math.isfinite(detail_score)

    from process_intelligence.ui.demo_evaluation import format_optional_score

    assert format_optional_score(float("nan")) == "N/A"
    assert format_optional_score(float("inf")) == "N/A"


def test_deterministic_result_serialization() -> None:
    events = [
        _event(rank=1, original_row_id=1),
        _event(rank=2, original_row_id=0),
    ]
    first = evaluate_demo_anomaly_events(events=events, demo_frame=_demo_frame())
    second = evaluate_demo_anomaly_events(events=events, demo_frame=_demo_frame())
    assert first.to_dict() == second.to_dict()
    assert first.evaluation_status.value == "AVAILABLE"


def test_result_contains_no_dataframe_or_model_object() -> None:
    summary = evaluate_demo_anomaly_events(
        events=[_event(rank=1, original_row_id=1)],
        demo_frame=_demo_frame(),
    )
    assert summary_contains_no_runtime_objects(summary)
    payload = summary.to_dict()
    assert "DataFrame" not in repr(type(payload))
    for value in payload.values():
        assert not isinstance(value, pd.DataFrame)


def _presentation_report(
    *,
    analysis_mode: str,
    anomaly_events: list[AnomalyEventView],
    dataset_fingerprint: str = _FINGERPRINT_A,
) -> object:
    from process_intelligence.reporting.schemas import (
        WorkflowCohortFilterSummaryView,
        WorkflowDataSummaryView,
        WorkflowModelSummaryView,
        WorkflowOverviewView,
        WorkflowPresentationReport,
        WorkflowRoutingSummaryView,
        WorkflowStageView,
    )
    from process_intelligence.workflow import (
        AnalysisWorkflowStage,
        AnalysisWorkflowStatus,
    )

    return WorkflowPresentationReport(
        overview=WorkflowOverviewView(
            status=AnalysisWorkflowStatus.PARTIAL,
            terminal_stage=AnalysisWorkflowStage.DIAGNOSIS,
            headline="Analysis completed without an executable recommendation",
            summary="Demo evaluation unit-test report.",
            started_at=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            completed_at=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            total_seconds=1.0,
        ),
        data_summary=WorkflowDataSummaryView(
            raw_row_count=5,
            processed_row_count=5,
            cohort_row_count=5,
            train_row_count=3,
            validation_row_count=1,
            test_row_count=1,
            anomaly_event_count=len(anomaly_events),
            diagnosis_factor_count=0,
            selected_operating_row_id=1,
            row_identity_preserved=True,
        ),
        cohort_filter_summary=WorkflowCohortFilterSummaryView(
            configured=False,
            column_name=None,
            lower_bound=None,
            upper_bound=None,
            include_lower=None,
            include_upper=None,
            exclude_filter_column_from_features=None,
            source_row_count=5,
            retained_row_count=5,
            excluded_row_count=0,
            null_excluded_count=0,
            range_display="Not configured",
            filter_column_used_as_feature_display="Not applicable",
        ),
        routing_summary=WorkflowRoutingSummaryView(
            selected_industry="semiconductor",
            selected_task=None,
            inferred_task=None,
            task_selection_source=None,
            task_override_applied=False,
            target_column=None,
            feature_count=4,
        ),
        model_summary=WorkflowModelSummaryView(
            supervised_model_key=None,
            anomaly_model_key="isolation_forest",
            independent_test_evaluation_performed=True,
            residual_calibration_performed=False,
            test_used_for_model_selection=False,
            test_used_for_threshold_calibration=False,
            anomaly_score_direction=_SCORE_DIRECTION,
        ),
        stages=[
            WorkflowStageView(
                sequence=1,
                stage=AnalysisWorkflowStage.DIAGNOSIS,
                executed=True,
                succeeded=True,
                structured_refusal=False,
                status_label="SUCCEEDED",
                message="DIAGNOSIS stage completed.",
            )
        ],
        anomaly_events=anomaly_events,
        dataset_fingerprint=dataset_fingerprint,
        metadata={"analysis_mode": analysis_mode},
    )


def test_panel_eligible_only_for_matching_builtin_demo_fingerprint() -> None:
    report = _presentation_report(
        analysis_mode=AnalysisExecutionMode.ANOMALY_ONLY.value,
        anomaly_events=[_event(rank=1, original_row_id=1)],
    )
    assert (
        is_demo_evaluation_panel_eligible(
            data_source=DATA_SOURCE_BUILTIN_DEMO,
            report=report,  # type: ignore[arg-type]
            current_dataset_fingerprint=_FINGERPRINT_A,
        )
        is True
    )
    assert (
        is_demo_evaluation_panel_eligible(
            data_source=DATA_SOURCE_UPLOAD_CSV,
            report=report,  # type: ignore[arg-type]
            current_dataset_fingerprint=_FINGERPRINT_A,
        )
        is False
    )
    assert (
        is_demo_evaluation_panel_eligible(
            data_source=DATA_SOURCE_BUILTIN_DEMO,
            report=report,  # type: ignore[arg-type]
            current_dataset_fingerprint=_FINGERPRINT_B,
        )
        is False
    )


def test_supervised_selects_residual_events_only() -> None:
    report = _presentation_report(
        analysis_mode=AnalysisExecutionMode.SUPERVISED.value,
        anomaly_events=[
            _event(
                rank=1,
                original_row_id=1,
                selection_source="residual_anomaly_detector",
            ),
            _event(
                rank=2,
                original_row_id=0,
                selection_source="unsupervised_anomaly_model",
            ),
        ],
    )
    selected, mode = select_displayed_events_for_evaluation(report)  # type: ignore[arg-type]
    assert mode == AnalysisExecutionMode.SUPERVISED.value
    assert len(selected) == 1
    assert selected[0].original_row_id == 1
    summary = evaluate_demo_anomaly_presentation(
        report=report,  # type: ignore[arg-type]
        demo_frame=_demo_frame(),
    )
    assert summary.evaluation_status is DemoAnomalyEvaluationStatus.AVAILABLE
    assert summary.evaluated_event_count == 1
    assert summary.matched_event_count == 1


def test_evaluation_does_not_mutate_events_or_frame() -> None:
    frame = _demo_frame()
    events = [_event(rank=1, original_row_id=1)]
    before_frame = frame.copy(deep=True)
    before_score = events[0].anomaly_score
    evaluate_demo_anomaly_events(events=events, demo_frame=frame)
    assert frame.equals(before_frame)
    assert events[0].anomaly_score == before_score
    assert "injected_anomaly" not in ["temperature_setpoint"]


def test_formatters() -> None:
    assert format_percentage(0.5) == "50.0%"
    assert format_percentage(None) == "N/A"
    assert format_enrichment(2.5) == "2.50×"
    assert format_enrichment(None) == "N/A"
