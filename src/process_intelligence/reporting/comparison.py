"""Pure builders for anomaly-run baseline comparison presentation (Step 11B.11).

Compares two ``WorkflowPresentationReport`` instances using presentation fields
only. Does not access DataFrames, models, estimators, raw CSV content, or
recompute anomaly / diagnosis scores.
"""

from __future__ import annotations

from process_intelligence.reporting.schemas import (
    AnomalyRunComparisonView,
    DiagnosisFactorComparisonView,
    DiagnosisFactorPresenceStatus,
    EventOverlapEntryView,
    EventOverlapPresenceStatus,
    EventOverlapView,
    RunConfigurationComparisonView,
    WorkflowPresentationReport,
)
from process_intelligence.workflow.enums import (
    AnalysisExecutionMode,
    AnalysisWorkflowStatus,
)

_COMPATIBLE_MESSAGE = (
    "Baseline and current anomaly reports are comparable for configuration, "
    "selected-event overlap, and diagnosis-factor rank changes."
)

_DISCLAIMERS: tuple[str, ...] = (
    "Different cohort runs use separately fitted anomaly models.",
    "Anomaly score magnitudes are not directly comparable.",
    "Shared events indicate row-selection stability, not confirmed defects.",
    "Shared factors indicate repeated associations, not causation.",
)

_EMPTY_EVENT_OVERLAP = EventOverlapView(
    baseline_event_count=0,
    current_event_count=0,
    shared_event_count=0,
    shared_original_row_ids=[],
    baseline_only_original_row_ids=[],
    current_only_original_row_ids=[],
    entries=[],
)


def build_anomaly_run_comparison(
    baseline: WorkflowPresentationReport,
    current: WorkflowPresentationReport,
) -> AnomalyRunComparisonView:
    """Build a deterministic anomaly-run comparison from two presentation reports.

    Args:
        baseline: Previously saved comparison baseline presentation report.
        current: Current presentation report to compare against the baseline.

    Returns:
        Immutable JSON-safe comparison view. When incompatible, event and factor
        overlap fields are empty and ``compatible`` is False.

    Raises:
        TypeError: If either argument is not a ``WorkflowPresentationReport``.
    """
    if not isinstance(baseline, WorkflowPresentationReport):
        raise TypeError(
            "baseline must be WorkflowPresentationReport, "
            f"got {type(baseline).__name__}"
        )
    if not isinstance(current, WorkflowPresentationReport):
        raise TypeError(
            "current must be WorkflowPresentationReport, "
            f"got {type(current).__name__}"
        )

    configuration = _build_configuration_comparison(baseline, current)
    same_dataset = _same_dataset(baseline, current)
    compatible, message, warnings = _evaluate_compatibility(
        baseline,
        current,
        same_dataset=same_dataset,
    )

    if not compatible:
        return AnomalyRunComparisonView(
            compatible=False,
            compatibility_message=message,
            same_dataset=same_dataset,
            configuration=configuration,
            event_overlap=_EMPTY_EVENT_OVERLAP.model_copy(deep=True),
            factor_comparison=[],
            warnings=list(warnings),
            disclaimers=list(_DISCLAIMERS),
        )

    event_overlap = _build_event_overlap(baseline, current)
    factor_comparison = _build_factor_comparison(baseline, current)
    return AnomalyRunComparisonView(
        compatible=True,
        compatibility_message=_COMPATIBLE_MESSAGE,
        same_dataset=True,
        configuration=configuration,
        event_overlap=event_overlap,
        factor_comparison=factor_comparison,
        warnings=list(warnings),
        disclaimers=list(_DISCLAIMERS),
    )


def _analysis_mode(report: WorkflowPresentationReport) -> str | None:
    value = report.metadata.get("analysis_mode")
    if isinstance(value, str) and value != "" and value.strip() != "":
        return value
    return None


def _cohort_description(report: WorkflowPresentationReport) -> str:
    summary = report.cohort_filter_summary
    if not summary.configured:
        return "Not configured"
    column_name = summary.column_name
    if column_name is None:
        return summary.range_display
    return f"{column_name}: {summary.range_display}"


def _build_configuration_comparison(
    baseline: WorkflowPresentationReport,
    current: WorkflowPresentationReport,
) -> RunConfigurationComparisonView:
    return RunConfigurationComparisonView(
        baseline_analysis_mode=_analysis_mode(baseline),
        current_analysis_mode=_analysis_mode(current),
        baseline_industry=baseline.routing_summary.selected_industry,
        current_industry=current.routing_summary.selected_industry,
        baseline_cohort_configured=baseline.cohort_filter_summary.configured,
        current_cohort_configured=current.cohort_filter_summary.configured,
        baseline_cohort_description=_cohort_description(baseline),
        current_cohort_description=_cohort_description(current),
        baseline_analysis_rows=baseline.data_summary.cohort_row_count,
        current_analysis_rows=current.data_summary.cohort_row_count,
        baseline_feature_count=baseline.routing_summary.feature_count,
        current_feature_count=current.routing_summary.feature_count,
        baseline_train_row_count=baseline.data_summary.train_row_count,
        current_train_row_count=current.data_summary.train_row_count,
        baseline_validation_row_count=baseline.data_summary.validation_row_count,
        current_validation_row_count=current.data_summary.validation_row_count,
        baseline_test_row_count=baseline.data_summary.test_row_count,
        current_test_row_count=current.data_summary.test_row_count,
        baseline_anomaly_model=baseline.model_summary.anomaly_model_key,
        current_anomaly_model=current.model_summary.anomaly_model_key,
        baseline_anomaly_event_count=baseline.data_summary.anomaly_event_count,
        current_anomaly_event_count=current.data_summary.anomaly_event_count,
        baseline_diagnosis_factor_count=baseline.data_summary.diagnosis_factor_count,
        current_diagnosis_factor_count=current.data_summary.diagnosis_factor_count,
        baseline_operating_row_id=baseline.data_summary.selected_operating_row_id,
        current_operating_row_id=current.data_summary.selected_operating_row_id,
    )


def _same_dataset(
    baseline: WorkflowPresentationReport,
    current: WorkflowPresentationReport,
) -> bool:
    baseline_fp = baseline.dataset_fingerprint
    current_fp = current.dataset_fingerprint
    if baseline_fp is None or current_fp is None:
        return False
    return baseline_fp == current_fp


def _evaluate_compatibility(
    baseline: WorkflowPresentationReport,
    current: WorkflowPresentationReport,
    *,
    same_dataset: bool,
) -> tuple[bool, str, list[str]]:
    warnings: list[str] = []
    baseline_mode = _analysis_mode(baseline)
    current_mode = _analysis_mode(current)

    if baseline.overview.status is AnalysisWorkflowStatus.REFUSED:
        return (
            False,
            "Comparison is unavailable because the baseline report was refused.",
            warnings,
        )
    if current.overview.status is AnalysisWorkflowStatus.REFUSED:
        return (
            False,
            "Comparison is unavailable because the current report was refused.",
            warnings,
        )
    if baseline.dataset_fingerprint is None or current.dataset_fingerprint is None:
        return (
            False,
            (
                "Comparison is unavailable because one or both reports are missing "
                "a dataset fingerprint."
            ),
            warnings,
        )
    if not same_dataset:
        return (
            False,
            (
                "Comparison is unavailable because the baseline and current "
                "reports were generated from different datasets."
            ),
            warnings,
        )
    if baseline_mode != AnalysisExecutionMode.ANOMALY_ONLY.value:
        return (
            False,
            (
                "Comparison is unavailable because the baseline report is not an "
                "ANOMALY_ONLY analysis."
            ),
            warnings,
        )
    if current_mode != AnalysisExecutionMode.ANOMALY_ONLY.value:
        return (
            False,
            (
                "Comparison is unavailable because the current report is not an "
                "ANOMALY_ONLY analysis."
            ),
            warnings,
        )
    if not baseline.anomaly_events:
        return (
            False,
            (
                "Comparison is unavailable because the baseline report does not "
                "include anomaly event results."
            ),
            warnings,
        )
    if not current.anomaly_events:
        return (
            False,
            (
                "Comparison is unavailable because the current report does not "
                "include anomaly event results."
            ),
            warnings,
        )
    return True, _COMPATIBLE_MESSAGE, warnings


def _build_event_overlap(
    baseline: WorkflowPresentationReport,
    current: WorkflowPresentationReport,
) -> EventOverlapView:
    baseline_rank_by_id = {
        event.original_row_id: event.rank for event in baseline.anomaly_events
    }
    current_rank_by_id = {
        event.original_row_id: event.rank for event in current.anomaly_events
    }
    baseline_ids = [event.original_row_id for event in baseline.anomaly_events]
    current_ids = [event.original_row_id for event in current.anomaly_events]
    current_id_set = set(current_ids)
    baseline_id_set = set(baseline_ids)

    shared_ids = [row_id for row_id in baseline_ids if row_id in current_id_set]
    baseline_only_ids = [
        row_id for row_id in baseline_ids if row_id not in current_id_set
    ]
    current_only_ids = [
        row_id for row_id in current_ids if row_id not in baseline_id_set
    ]

    entries: list[EventOverlapEntryView] = []
    for row_id in shared_ids:
        entries.append(
            EventOverlapEntryView(
                presence_status=EventOverlapPresenceStatus.SHARED,
                original_row_id=row_id,
                baseline_rank=baseline_rank_by_id[row_id],
                current_rank=current_rank_by_id[row_id],
            )
        )
    for row_id in baseline_only_ids:
        entries.append(
            EventOverlapEntryView(
                presence_status=EventOverlapPresenceStatus.BASELINE_ONLY,
                original_row_id=row_id,
                baseline_rank=baseline_rank_by_id[row_id],
                current_rank=None,
            )
        )
    for row_id in current_only_ids:
        entries.append(
            EventOverlapEntryView(
                presence_status=EventOverlapPresenceStatus.CURRENT_ONLY,
                original_row_id=row_id,
                baseline_rank=None,
                current_rank=current_rank_by_id[row_id],
            )
        )

    return EventOverlapView(
        baseline_event_count=len(baseline.anomaly_events),
        current_event_count=len(current.anomaly_events),
        shared_event_count=len(shared_ids),
        shared_original_row_ids=shared_ids,
        baseline_only_original_row_ids=baseline_only_ids,
        current_only_original_row_ids=current_only_ids,
        entries=entries,
    )


def _build_factor_comparison(
    baseline: WorkflowPresentationReport,
    current: WorkflowPresentationReport,
) -> list[DiagnosisFactorComparisonView]:
    baseline_by_name = {
        factor.feature_name: factor for factor in baseline.diagnosis_factors
    }
    current_by_name = {
        factor.feature_name: factor for factor in current.diagnosis_factors
    }
    shared_names = sorted(
        set(baseline_by_name) & set(current_by_name),
        key=lambda name: (
            current_by_name[name].rank,
            baseline_by_name[name].rank,
            name,
        ),
    )
    current_only_names = sorted(
        set(current_by_name) - set(baseline_by_name),
        key=lambda name: (current_by_name[name].rank, name),
    )
    baseline_only_names = sorted(
        set(baseline_by_name) - set(current_by_name),
        key=lambda name: (baseline_by_name[name].rank, name),
    )

    factors: list[DiagnosisFactorComparisonView] = []
    for name in shared_names:
        baseline_factor = baseline_by_name[name]
        current_factor = current_by_name[name]
        factors.append(
            DiagnosisFactorComparisonView(
                feature_name=name,
                baseline_rank=baseline_factor.rank,
                current_rank=current_factor.rank,
                baseline_direction=baseline_factor.direction,
                current_direction=current_factor.direction,
                presence_status=DiagnosisFactorPresenceStatus.SHARED,
            )
        )
    for name in current_only_names:
        current_factor = current_by_name[name]
        factors.append(
            DiagnosisFactorComparisonView(
                feature_name=name,
                baseline_rank=None,
                current_rank=current_factor.rank,
                baseline_direction=None,
                current_direction=current_factor.direction,
                presence_status=DiagnosisFactorPresenceStatus.CURRENT_ONLY,
            )
        )
    for name in baseline_only_names:
        baseline_factor = baseline_by_name[name]
        factors.append(
            DiagnosisFactorComparisonView(
                feature_name=name,
                baseline_rank=baseline_factor.rank,
                current_rank=None,
                baseline_direction=baseline_factor.direction,
                current_direction=None,
                presence_status=DiagnosisFactorPresenceStatus.BASELINE_ONLY,
            )
        )
    return factors
