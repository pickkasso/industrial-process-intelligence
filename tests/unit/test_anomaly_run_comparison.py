"""Unit tests for anomaly-run comparison DTOs and builder (Step 11B.11)."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from process_intelligence.reporting import (
    AnomalyEventView,
    AnomalyRunComparisonView,
    DiagnosisFactorComparisonView,
    DiagnosisFactorPresenceStatus,
    DiagnosisFactorView,
    EventOverlapEntryView,
    EventOverlapPresenceStatus,
    EventOverlapView,
    RunConfigurationComparisonView,
    WorkflowCohortFilterSummaryView,
    WorkflowDataSummaryView,
    WorkflowModelSummaryView,
    WorkflowOverviewView,
    WorkflowPresentationReport,
    WorkflowRoutingSummaryView,
    WorkflowStageView,
    build_anomaly_run_comparison,
)
from process_intelligence.workflow import (
    AnalysisWorkflowStage,
    AnalysisWorkflowStatus,
)

_UTC_START = datetime(2026, 7, 21, 10, 0, tzinfo=UTC)
_UTC_END = datetime(2026, 7, 21, 10, 5, tzinfo=UTC)
_FINGERPRINT_A = hashlib.sha256(b"dataset-a").hexdigest()
_FINGERPRINT_B = hashlib.sha256(b"dataset-b").hexdigest()


def _overview(**overrides: Any) -> WorkflowOverviewView:
    payload: dict[str, Any] = {
        "status": AnalysisWorkflowStatus.PARTIAL,
        "terminal_stage": AnalysisWorkflowStage.DIAGNOSIS,
        "headline": "Analysis completed without an executable recommendation",
        "summary": (
            "Anomaly-only analysis completed. Recommendation generation is not "
            "enabled for this analysis mode."
        ),
        "started_at": _UTC_START,
        "completed_at": _UTC_END,
        "total_seconds": 1.0,
        "recommendation_status": None,
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
        "diagnosis_factor_count": 2,
        "selected_operating_row_id": 13,
        "row_identity_preserved": True,
    }
    payload.update(overrides)
    return WorkflowDataSummaryView(**payload)


def _cohort(**overrides: Any) -> WorkflowCohortFilterSummaryView:
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
        "selected_industry": "battery",
        "selected_task": None,
        "inferred_task": None,
        "task_selection_source": None,
        "task_override_applied": False,
        "target_column": None,
        "feature_count": 10,
    }
    payload.update(overrides)
    return WorkflowRoutingSummaryView(**payload)


def _model(**overrides: Any) -> WorkflowModelSummaryView:
    payload: dict[str, Any] = {
        "supervised_model_key": None,
        "anomaly_model_key": "isolation_forest",
        "independent_test_evaluation_performed": True,
        "residual_calibration_performed": False,
        "test_used_for_model_selection": False,
        "test_used_for_threshold_calibration": False,
        "anomaly_score_direction": "higher_is_more_anomalous",
    }
    payload.update(overrides)
    return WorkflowModelSummaryView(**payload)


def _stage(**overrides: Any) -> WorkflowStageView:
    payload: dict[str, Any] = {
        "sequence": 1,
        "stage": AnalysisWorkflowStage.LOAD,
        "executed": True,
        "succeeded": True,
        "structured_refusal": False,
        "status_label": "SUCCEEDED",
        "row_count": 100,
        "message": "LOAD stage completed.",
        "warnings": [],
        "metadata": {},
    }
    payload.update(overrides)
    return WorkflowStageView(**payload)


def _event(**overrides: Any) -> AnomalyEventView:
    payload: dict[str, Any] = {
        "rank": 1,
        "original_row_id": 13,
        "anomaly_score": 0.9,
        "is_operating_row": True,
        "selection_source": "unsupervised_anomaly_model",
        "score_direction": "higher_is_more_anomalous",
        "is_anomaly_flagged": True,
    }
    payload.update(overrides)
    return AnomalyEventView(**payload)


def _factor(**overrides: Any) -> DiagnosisFactorView:
    payload: dict[str, Any] = {
        "rank": 1,
        "feature_name": "RSOCavg",
        "diagnostic_score": 0.8,
        "direction": "POSITIVE",
    }
    payload.update(overrides)
    return DiagnosisFactorView(**payload)


def _anomaly_report(**overrides: Any) -> WorkflowPresentationReport:
    payload: dict[str, Any] = {
        "overview": _overview(),
        "data_summary": _data_summary(),
        "cohort_filter_summary": _cohort(),
        "routing_summary": _routing(),
        "model_summary": _model(),
        "model_performance": None,
        "stages": [_stage()],
        "anomaly_events": [
            _event(rank=1, original_row_id=13, is_operating_row=True),
            _event(rank=2, original_row_id=11, is_operating_row=False),
        ],
        "diagnosis_factors": [
            _factor(rank=1, feature_name="RSOCavg", direction="POSITIVE"),
            _factor(rank=2, feature_name="Current", direction="NEGATIVE"),
        ],
        "recommendation": None,
        "warnings": [],
        "disclaimers": ["Decision support only."],
        "metadata": {"analysis_mode": "ANOMALY_ONLY"},
        "dataset_fingerprint": _FINGERPRINT_A,
    }
    payload.update(overrides)
    return WorkflowPresentationReport(**payload)


def _empty_overlap(**overrides: Any) -> EventOverlapView:
    payload: dict[str, Any] = {
        "baseline_event_count": 0,
        "current_event_count": 0,
        "shared_event_count": 0,
        "shared_original_row_ids": [],
        "baseline_only_original_row_ids": [],
        "current_only_original_row_ids": [],
        "entries": [],
    }
    payload.update(overrides)
    return EventOverlapView(**payload)


def _configuration(**overrides: Any) -> RunConfigurationComparisonView:
    payload: dict[str, Any] = {
        "baseline_analysis_mode": "ANOMALY_ONLY",
        "current_analysis_mode": "ANOMALY_ONLY",
        "baseline_industry": "battery",
        "current_industry": "battery",
        "baseline_cohort_configured": False,
        "current_cohort_configured": False,
        "baseline_cohort_description": "Not configured",
        "current_cohort_description": "Not configured",
        "baseline_analysis_rows": 100,
        "current_analysis_rows": 100,
        "baseline_feature_count": 10,
        "current_feature_count": 10,
        "baseline_train_row_count": 60,
        "current_train_row_count": 60,
        "baseline_validation_row_count": 20,
        "current_validation_row_count": 20,
        "baseline_test_row_count": 20,
        "current_test_row_count": 20,
        "baseline_anomaly_model": "isolation_forest",
        "current_anomaly_model": "isolation_forest",
        "baseline_anomaly_event_count": 2,
        "current_anomaly_event_count": 2,
        "baseline_diagnosis_factor_count": 2,
        "current_diagnosis_factor_count": 2,
        "baseline_operating_row_id": 13,
        "current_operating_row_id": 13,
    }
    payload.update(overrides)
    return RunConfigurationComparisonView(**payload)


# ---------------------------------------------------------------------------
# Comparison DTO tests
# ---------------------------------------------------------------------------


def test_valid_comparison_view() -> None:
    view = AnomalyRunComparisonView(
        compatible=True,
        compatibility_message="ok",
        same_dataset=True,
        configuration=_configuration(),
        event_overlap=_empty_overlap(
            baseline_event_count=1,
            current_event_count=1,
            shared_event_count=1,
            shared_original_row_ids=[13],
            entries=[
                EventOverlapEntryView(
                    presence_status=EventOverlapPresenceStatus.SHARED,
                    original_row_id=13,
                    baseline_rank=1,
                    current_rank=1,
                )
            ],
        ),
        factor_comparison=[
            DiagnosisFactorComparisonView(
                feature_name="RSOCavg",
                baseline_rank=1,
                current_rank=1,
                baseline_direction="POSITIVE",
                current_direction="POSITIVE",
                presence_status=DiagnosisFactorPresenceStatus.SHARED,
            )
        ],
        warnings=[],
        disclaimers=["Anomaly score magnitudes are not directly comparable."],
    )
    assert view.compatible is True
    assert view.event_overlap.shared_event_count == 1


def test_incompatible_comparison_view() -> None:
    view = AnomalyRunComparisonView(
        compatible=False,
        compatibility_message="different datasets",
        same_dataset=False,
        configuration=_configuration(),
        event_overlap=_empty_overlap(),
        factor_comparison=[],
        warnings=[],
        disclaimers=["Shared factors indicate repeated associations, not causation."],
    )
    assert view.compatible is False
    assert view.event_overlap.entries == []


def test_comparison_views_are_immutable() -> None:
    view = AnomalyRunComparisonView(
        compatible=True,
        compatibility_message="ok",
        same_dataset=True,
        configuration=_configuration(),
        event_overlap=_empty_overlap(),
        factor_comparison=[],
        warnings=[],
        disclaimers=["Decision support."],
    )
    with pytest.raises(ValidationError):
        view.compatible = False  # type: ignore[misc]
    with pytest.raises(ValidationError):
        view.configuration.baseline_analysis_rows = 1  # type: ignore[misc]


def test_empty_event_overlap() -> None:
    overlap = _empty_overlap()
    assert overlap.shared_event_count == 0
    assert overlap.shared_original_row_ids == []
    assert overlap.entries == []


def test_json_safe_row_ids() -> None:
    overlap = _empty_overlap(
        baseline_event_count=2,
        current_event_count=1,
        shared_event_count=1,
        shared_original_row_ids=[13],
        baseline_only_original_row_ids=["LOT-7"],
        current_only_original_row_ids=[],
        entries=[
            EventOverlapEntryView(
                presence_status=EventOverlapPresenceStatus.SHARED,
                original_row_id=13,
                baseline_rank=1,
                current_rank=1,
            ),
            EventOverlapEntryView(
                presence_status=EventOverlapPresenceStatus.BASELINE_ONLY,
                original_row_id="LOT-7",
                baseline_rank=2,
                current_rank=None,
            ),
        ],
    )
    dumped = overlap.model_dump(mode="json")
    encoded = json.dumps(dumped)
    restored = EventOverlapView.model_validate(json.loads(encoded))
    assert restored.shared_original_row_ids == [13]
    assert restored.baseline_only_original_row_ids == ["LOT-7"]


def test_factor_rank_optional_fields() -> None:
    shared = DiagnosisFactorComparisonView(
        feature_name="a",
        baseline_rank=1,
        current_rank=2,
        baseline_direction=None,
        current_direction="POSITIVE",
        presence_status=DiagnosisFactorPresenceStatus.SHARED,
    )
    baseline_only = DiagnosisFactorComparisonView(
        feature_name="b",
        baseline_rank=3,
        current_rank=None,
        baseline_direction="NEGATIVE",
        current_direction=None,
        presence_status=DiagnosisFactorPresenceStatus.BASELINE_ONLY,
    )
    assert shared.current_direction == "POSITIVE"
    assert baseline_only.current_rank is None


def test_presence_status_validation() -> None:
    with pytest.raises(ValidationError):
        DiagnosisFactorComparisonView(
            feature_name="a",
            baseline_rank=1,
            current_rank=1,
            presence_status="BOTH",  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError):
        DiagnosisFactorComparisonView(
            feature_name="a",
            baseline_rank=1,
            current_rank=None,
            presence_status=DiagnosisFactorPresenceStatus.SHARED,
        )


def test_comparison_rejects_nan_infinity() -> None:
    with pytest.raises(ValidationError):
        _configuration(baseline_analysis_rows=math.nan)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        _data_summary(raw_row_count=math.inf)  # type: ignore[arg-type]


def test_comparison_json_round_trip() -> None:
    view = AnomalyRunComparisonView(
        compatible=True,
        compatibility_message="ok",
        same_dataset=True,
        configuration=_configuration(),
        event_overlap=_empty_overlap(),
        factor_comparison=[
            DiagnosisFactorComparisonView(
                feature_name="Current",
                baseline_rank=None,
                current_rank=1,
                baseline_direction=None,
                current_direction="NEGATIVE",
                presence_status=DiagnosisFactorPresenceStatus.CURRENT_ONLY,
            )
        ],
        warnings=["note"],
        disclaimers=["Shared events indicate row-selection stability, not confirmed defects."],
    )
    dumped = view.model_dump(mode="json")
    restored = AnomalyRunComparisonView.model_validate(json.loads(json.dumps(dumped)))
    assert restored.model_dump(mode="json") == dumped


def test_independent_default_state() -> None:
    first = AnomalyRunComparisonView(
        compatible=False,
        compatibility_message="a",
        same_dataset=False,
        configuration=_configuration(),
        event_overlap=_empty_overlap(),
        warnings=[],
        disclaimers=["a"],
    )
    second = AnomalyRunComparisonView(
        compatible=False,
        compatibility_message="b",
        same_dataset=False,
        configuration=_configuration(),
        event_overlap=_empty_overlap(),
        warnings=[],
        disclaimers=["b"],
    )
    first.warnings.append("mutated")  # type: ignore[attr-defined]
    assert second.warnings == []


# ---------------------------------------------------------------------------
# Comparison builder tests
# ---------------------------------------------------------------------------


def test_same_dataset_compatible() -> None:
    baseline = _anomaly_report()
    current = _anomaly_report(
        anomaly_events=[
            _event(rank=1, original_row_id=13),
            _event(rank=2, original_row_id=8, is_operating_row=False),
        ]
    )
    comparison = build_anomaly_run_comparison(baseline, current)
    assert comparison.compatible is True
    assert comparison.same_dataset is True


def test_different_dataset_incompatible() -> None:
    baseline = _anomaly_report(dataset_fingerprint=_FINGERPRINT_A)
    current = _anomaly_report(dataset_fingerprint=_FINGERPRINT_B)
    comparison = build_anomaly_run_comparison(baseline, current)
    assert comparison.compatible is False
    assert comparison.same_dataset is False
    assert "different datasets" in comparison.compatibility_message
    assert comparison.event_overlap.entries == []
    assert comparison.factor_comparison == []


def test_missing_fingerprint_incompatible() -> None:
    baseline = _anomaly_report(dataset_fingerprint=None)
    current = _anomaly_report()
    comparison = build_anomaly_run_comparison(baseline, current)
    assert comparison.compatible is False
    assert "fingerprint" in comparison.compatibility_message.lower()


def test_supervised_baseline_incompatible() -> None:
    baseline = _anomaly_report(metadata={"analysis_mode": "SUPERVISED"})
    current = _anomaly_report()
    comparison = build_anomaly_run_comparison(baseline, current)
    assert comparison.compatible is False
    assert "ANOMALY_ONLY" in comparison.compatibility_message


def test_refused_baseline_incompatible() -> None:
    baseline = _anomaly_report(
        overview=_overview(
            status=AnalysisWorkflowStatus.REFUSED,
            terminal_stage=AnalysisWorkflowStage.LOAD,
            headline="Analysis was stopped by a safety or validation gate",
            summary="Refused at LOAD.",
            recommendation_status=None,
        )
    )
    current = _anomaly_report()
    comparison = build_anomaly_run_comparison(baseline, current)
    assert comparison.compatible is False
    assert "refused" in comparison.compatibility_message.lower()


def test_event_shared_baseline_only_current_only_and_ranks() -> None:
    baseline = _anomaly_report(
        data_summary=_data_summary(anomaly_event_count=3, diagnosis_factor_count=1),
        anomaly_events=[
            _event(rank=1, original_row_id=13),
            _event(rank=2, original_row_id=11, is_operating_row=False),
            _event(rank=3, original_row_id=16, is_operating_row=False),
        ],
        diagnosis_factors=[_factor(rank=1, feature_name="RSOCavg")],
    )
    current = _anomaly_report(
        data_summary=_data_summary(
            anomaly_event_count=2,
            diagnosis_factor_count=1,
            selected_operating_row_id=11,
            cohort_row_count=40,
            train_row_count=24,
            validation_row_count=8,
            test_row_count=8,
        ),
        anomaly_events=[
            _event(rank=1, original_row_id=11),
            _event(rank=2, original_row_id=0, is_operating_row=False),
        ],
        diagnosis_factors=[_factor(rank=1, feature_name="Current")],
    )
    comparison = build_anomaly_run_comparison(baseline, current)
    overlap = comparison.event_overlap
    assert overlap.shared_original_row_ids == [11]
    assert overlap.baseline_only_original_row_ids == [13, 16]
    assert overlap.current_only_original_row_ids == [0]
    shared_entry = next(
        item
        for item in overlap.entries
        if item.presence_status is EventOverlapPresenceStatus.SHARED
    )
    assert shared_entry.baseline_rank == 2
    assert shared_entry.current_rank == 1


def test_factor_shared_baseline_only_current_only_direction_and_order() -> None:
    baseline = _anomaly_report(
        data_summary=_data_summary(diagnosis_factor_count=3, anomaly_event_count=1),
        anomaly_events=[_event(rank=1, original_row_id=1)],
        diagnosis_factors=[
            _factor(rank=1, feature_name="RSOCavg", direction="POSITIVE"),
            _factor(rank=2, feature_name="Power", direction="NEGATIVE"),
            _factor(rank=3, feature_name="Voltage", direction="POSITIVE"),
        ],
    )
    current = _anomaly_report(
        data_summary=_data_summary(diagnosis_factor_count=3, anomaly_event_count=1),
        anomaly_events=[_event(rank=1, original_row_id=1)],
        diagnosis_factors=[
            _factor(rank=1, feature_name="Current", direction="NEGATIVE"),
            _factor(rank=2, feature_name="RSOCavg", direction="POSITIVE"),
            _factor(rank=3, feature_name="Temp", direction="POSITIVE"),
        ],
    )
    comparison = build_anomaly_run_comparison(baseline, current)
    names = [item.feature_name for item in comparison.factor_comparison]
    statuses = [item.presence_status for item in comparison.factor_comparison]
    assert names[0] == "RSOCavg"
    assert statuses[0] is DiagnosisFactorPresenceStatus.SHARED
    assert comparison.factor_comparison[0].baseline_direction == "POSITIVE"
    assert comparison.factor_comparison[0].current_direction == "POSITIVE"
    assert names[1:] == ["Current", "Temp", "Power", "Voltage"]
    assert statuses[1] is DiagnosisFactorPresenceStatus.CURRENT_ONLY
    assert statuses[2] is DiagnosisFactorPresenceStatus.CURRENT_ONLY
    assert statuses[3] is DiagnosisFactorPresenceStatus.BASELINE_ONLY
    assert statuses[4] is DiagnosisFactorPresenceStatus.BASELINE_ONLY


def test_deterministic_ordering() -> None:
    baseline = _anomaly_report()
    current = _anomaly_report(
        anomaly_events=[
            _event(rank=1, original_row_id=11, is_operating_row=False),
            _event(rank=2, original_row_id=13),
        ]
    )
    first = build_anomaly_run_comparison(baseline, current).model_dump(mode="json")
    second = build_anomaly_run_comparison(baseline, current).model_dump(mode="json")
    assert first == second


def test_no_score_deltas_in_comparison_payload() -> None:
    comparison = build_anomaly_run_comparison(_anomaly_report(), _anomaly_report())
    payload = json.dumps(comparison.model_dump(mode="json"))
    assert "anomaly_score" not in payload
    assert "diagnostic_score" not in payload
    assert "score_delta" not in payload
    assert "ranking_score" not in payload


def test_input_reports_immutable() -> None:
    baseline = _anomaly_report()
    current = _anomaly_report(
        anomaly_events=[_event(rank=1, original_row_id=0)],
        data_summary=_data_summary(anomaly_event_count=1, diagnosis_factor_count=2),
    )
    baseline_before = baseline.model_dump(mode="json")
    current_before = current.model_dump(mode="json")
    build_anomaly_run_comparison(baseline, current)
    assert baseline.model_dump(mode="json") == baseline_before
    assert current.model_dump(mode="json") == current_before


def test_comparison_result_json_safe_without_path_or_model() -> None:
    comparison = build_anomaly_run_comparison(_anomaly_report(), _anomaly_report())
    payload = comparison.model_dump(mode="json")
    encoded = json.dumps(payload)
    assert "C:\\" not in encoded
    assert "/Users/" not in encoded
    assert "DataFrame" not in encoded
    assert "IsolationForest" not in encoded
    restored = AnomalyRunComparisonView.model_validate(json.loads(encoded))
    assert restored.compatible is True


def test_global_vs_filtered_synthetic_scenario() -> None:
    baseline = _anomaly_report(
        data_summary=_data_summary(
            anomaly_event_count=5,
            diagnosis_factor_count=2,
            cohort_row_count=100,
            train_row_count=60,
            validation_row_count=20,
            test_row_count=20,
            selected_operating_row_id=13,
        ),
        anomaly_events=[
            _event(rank=1, original_row_id=13),
            _event(rank=2, original_row_id=11, is_operating_row=False),
            _event(rank=3, original_row_id=16, is_operating_row=False),
            _event(rank=4, original_row_id=17, is_operating_row=False),
            _event(rank=5, original_row_id=8, is_operating_row=False),
        ],
        diagnosis_factors=[
            _factor(rank=1, feature_name="RSOCmin"),
            _factor(rank=2, feature_name="RSOCmax"),
        ],
    )
    current = _anomaly_report(
        data_summary=_data_summary(
            anomaly_event_count=1,
            diagnosis_factor_count=2,
            cohort_row_count=40,
            train_row_count=24,
            validation_row_count=8,
            test_row_count=8,
            selected_operating_row_id=0,
        ),
        cohort_filter_summary=_cohort(
            configured=True,
            column_name="RSOCavg",
            lower_bound=80.0,
            upper_bound=100.0,
            include_lower=True,
            include_upper=True,
            exclude_filter_column_from_features=True,
            source_row_count=100,
            retained_row_count=40,
            excluded_row_count=60,
            null_excluded_count=0,
            range_display="80.0 ≤ x ≤ 100.0",
            filter_column_used_as_feature_display="No",
        ),
        anomaly_events=[_event(rank=1, original_row_id=0)],
        diagnosis_factors=[
            _factor(rank=1, feature_name="Current"),
            _factor(rank=2, feature_name="Power"),
        ],
    )
    comparison = build_anomaly_run_comparison(baseline, current)
    assert comparison.compatible is True
    assert comparison.event_overlap.shared_event_count == 0
    assert comparison.event_overlap.baseline_only_original_row_ids == [13, 11, 16, 17, 8]
    assert comparison.event_overlap.current_only_original_row_ids == [0]
    assert comparison.configuration.baseline_cohort_description == "Not configured"
    assert comparison.configuration.current_cohort_description.startswith("RSOCavg:")
