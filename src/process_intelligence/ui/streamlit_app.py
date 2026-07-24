"""Streamlit MVP application for industrial process analysis (Step 11B.2).

Collects user inputs, builds ``AnalysisWorkflowRequest``, runs
``IndustrialProcessAnalysisWorkflow``, converts the backend report through
``AnalysisWorkflowReportBuilder``, and renders ``WorkflowPresentationReport``.
Does not implement modeling, scoring, ranking, or recommendation algorithms.
"""

from __future__ import annotations

import csv
import math
import re
import tempfile
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, cast

import altair as alt
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import streamlit as st
from pydantic import ValidationError

from process_intelligence.core.enums import AnalysisTask, ColumnRole
from process_intelligence.core.exceptions import (
    DataValidationError,
    ProcessIntelligenceError,
)
from process_intelligence.data import evaluate_target_suitability
from process_intelligence.demo_data import (
    DEMO_QUALITY_SCORE_DECLARED_MAXIMUM,
    DEMO_QUALITY_SCORE_DECLARED_MINIMUM,
)
from process_intelligence.evaluation.performance_acceptance import (
    MetricAcceptanceDirection,
    ModelPerformanceAcceptanceStatus,
)
from process_intelligence.recommendation import (
    QualityOptimizationDirection,
    RecommendationObjective,
    RecommendationStatus,
    TargetPredictionPlausibilityStatus,
    WhatIfVerificationStatus,
)
from process_intelligence.reporting import AnalysisWorkflowReportBuilder
from process_intelligence.reporting.comparison import build_anomaly_run_comparison
from process_intelligence.reporting.schemas import (
    AnomalyContextWindowView,
    AnomalyEventView,
    AnomalyRunComparisonView,
    DiagnosisFactorComparisonView,
    DiagnosisFactorPresenceStatus,
    DiagnosisFactorView,
    EventOverlapEntryView,
    EventOverlapPresenceStatus,
    EventOverlapView,
    RecommendationWhatIfVerificationView,
    WorkflowPresentationReport,
)
from process_intelligence.ui.column_configuration import (
    AutomaticColumnConfigurator,
    UiColumnConfigurationReport,
    resolve_active_feature_columns,
)
from process_intelligence.ui.configuration_preset import (
    CONFIGURATION_PRESET_FILENAME,
    SESSION_ANOMALY_CONSTRAINT_VARIABLES_KEY,
    SESSION_ANOMALY_RECOMMENDATION_ENABLED_KEY,
    SESSION_CONFIRMED_CONTROLLABLE_KEY,
    SESSION_CONSTRAINT_VARIABLES_KEY,
    SESSION_EXPLICIT_FEATURES_KEY,
    SESSION_EXPLICIT_ROW_ID_KEY,
    SESSION_MAX_CHANGES_KEY,
    SESSION_OBJECTIVE_KEY,
    SESSION_OPERATING_MODE_KEY,
    SESSION_PRESET_APPLY_SUMMARY_KEY,
    SESSION_PRESET_MISSING_COLUMNS_KEY,
    SESSION_PRESET_PARSED_KEY,
    SESSION_PRESET_PENDING_APPLY_KEY,
    SESSION_PRESET_VALIDATION_MESSAGE_KEY,
    SESSION_PRESET_VALIDATION_OK_KEY,
    SESSION_QUALITY_DIRECTION_KEY,
    SESSION_QUALITY_TARGET_KEY,
    SESSION_REQUESTED_TASK_KEY,
    SESSION_ROLE_OVERRIDE_COLUMNS_KEY,
    SESSION_RULE_COUNT_KEY,
    SESSION_USE_RECOMMENDED_FEATURES_KEY,
    SESSION_VERIFIED_VARIABLES_KEY,
    ConfigurationPresetBuildResult,
    WorkflowUiConfigurationPreset,
    build_configuration_preset_session_updates,
    build_configuration_preset_summary,
    build_exportable_configuration_preset,
    check_configuration_preset_column_compatibility,
    clear_configuration_preset_widget_prefixes,
    configuration_preset_to_json,
    parse_configuration_preset_json,
)
from process_intelligence.ui.demo_configuration import (
    DATA_SOURCE_BUILTIN_DEMO,
    DATA_SOURCE_UPLOAD_CSV,
    SESSION_DATA_SOURCE_KEY,
    SESSION_DATA_SOURCE_PREVIOUS_KEY,
    SESSION_DEMO_APPLY_SUCCESS_KEY,
    SESSION_DEMO_APPLY_SUMMARY_KEY,
    SESSION_DEMO_TEMPLATE_KEY,
    build_demo_configuration_preset,
    clear_demo_recommendation_session_state,
    clear_stale_analysis_session_state,
    demo_configuration_apply_summary,
    demo_dataset_to_workflow_csv_bytes,
    demo_template_options,
    filter_ground_truth_from_recommended_features,
    ground_truth_metadata_columns_present,
    is_builtin_demo_source,
    load_builtin_demo_dataset,
    validate_demo_feature_guard,
)
from process_intelligence.ui.request_builder import WorkflowUiRequestBuilder
from process_intelligence.ui.schemas import (
    StreamlitUiConfig,
    UiMetricRuleInput,
    UiVariableConstraintInput,
    WorkflowUiSubmission,
)
from process_intelligence.ui.workflow_factory import create_default_analysis_workflow
from process_intelligence.workflow import (
    ANOMALY_CONTEXT_RADIUS,
    AnalysisExecutionMode,
    AnalysisWorkflowOutcome,
    AnalysisWorkflowStage,
    AnalysisWorkflowStatus,
    IndustrialProcessAnalysisWorkflow,
    NumericCohortFilter,
    OperatingPointSelectionMode,
    list_numeric_cohort_filter_candidates,
    observed_numeric_range,
    preview_numeric_cohort_filter_row_count,
)

_ORIGINAL_ROW_ID = "_original_row_id"
_SESSION_REPORT_KEY = "last_presentation_report_json"
_SESSION_BASELINE_REPORT_KEY = "comparison_baseline_report_json"
_SESSION_BASELINE_LABEL_KEY = "comparison_baseline_label"
_SESSION_COLUMNS_KEY = "ui_column_schema_fingerprint"
_SESSION_ANALYSIS_MODE_KEY = "ui_analysis_mode"
_SESSION_TARGET_KEY = "ui_target_column"
_SESSION_TIMESTAMP_KEY = "ui_timestamp_column"
_SESSION_IDENTIFIERS_KEY = "ui_identifier_columns"
_SESSION_EXCLUDED_KEY = "ui_excluded_columns"
_NOT_AVAILABLE = "Not available"
_NOT_APPLICABLE = "Not applicable"
_ROLE_NONE = "(no override)"
_TARGET_PLACEHOLDER = "(select target)"
_TIMESTAMP_NONE = "(none)"
_OBJECTIVE_PLACEHOLDER = "(select objective)"
_DIRECTION_PLACEHOLDER = "(select direction)"
_TASK_AUTO = "AUTO"
_TASK_HELP = (
    "AUTO uses the task router. "
    "REGRESSION predicts a continuous numeric target. "
    "CLASSIFICATION predicts discrete classes. "
    "Unsupported tasks are refused by the current workflow."
)
_TARGET_CONSTANT_OPERATOR_MESSAGE = (
    "The selected target has no usable variation. Choose a different target "
    "or use anomaly-only analysis."
)
_TASK_UNSUPPORTED_OPERATOR_MESSAGE = (
    "The selected analysis task is not supported by the current workflow."
)
_PERFORMANCE_RULE_INCOMPLETE_OPERATOR_MESSAGE = (
    "Complete the metric name, direction, and finite threshold."
)
_CONSTRAINT_INCOMPLETE_OPERATOR_MESSAGE = (
    "Complete or clear the selected process-variable constraint."
)
_COHORT_TOO_SMALL_OPERATOR_MESSAGE = (
    "The selected operating cohort does not contain enough rows for train, "
    "validation, and test partitions."
)
_RECOMMENDATION_REFUSED_AFTER_ANOMALY_MESSAGE = (
    "Anomaly analysis completed, but recommendation generation was refused "
    "by the safety gate."
)
_WINDOWS_ABS_PATH_PATTERN = re.compile(
    r"[A-Za-z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]*"
)
_UNIX_ABS_PATH_PATTERN = re.compile(
    r"/(?:Users|home|tmp|var|opt|private|mnt)/[^\s\"']+"
)
_KNOWN_OPERATOR_ERROR_FRAGMENTS: tuple[tuple[str, str], ...] = (
    (
        "retained too few rows for the required train/validation/test",
        _COHORT_TOO_SMALL_OPERATOR_MESSAGE,
    ),
    (
        "Classification modeling is not yet supported",
        _TASK_UNSUPPORTED_OPERATOR_MESSAGE,
    ),
    (
        "Classification is not supported by",
        _TASK_UNSUPPORTED_OPERATOR_MESSAGE,
    ),
)
_QUALITY_DIRECTION_PLACEHOLDER = "(select quality direction)"
_ANOMALY_ONLY_RECOMMENDATION_NOTE = (
    "Recommendation generation is not enabled for anomaly-only analysis."
)
_ANOMALY_ONLY_PERFORMANCE_NOTE = "Not applicable for anomaly-only analysis."
_ANOMALY_RECOMMENDATION_SECTION_CAPTION = (
    "Generate a constrained what-if suggestion for the selected operating row "
    "using the fitted anomaly model. Recommendations are model-based scenarios, "
    "not operational commands."
)
_ANOMALY_RECOMMENDATION_RESULT_CAPTION = (
    "Lower anomaly score means the scenario is less unusual according to the "
    "same fitted anomaly model. It does not prove improved quality or process "
    "safety."
)
_ANOMALY_RECOMMENDATION_UNCERTAINTY_NOTE = (
    "Uncertainty unavailable. Absence of uncertainty does not establish safety."
)
_WHAT_IF_SECTION_CAPTION = (
    "Adjacent constraint-grid scenarios are scored with the same fitted model "
    "to show whether the proposed improvement persists locally. This is "
    "model-local stability evidence, not proof of physical safety or causation."
)
_WHAT_IF_NOT_APPLICABLE = (
    "What-if verification was not applicable because no recommendation was generated."
)
_WHAT_IF_ANOMALY_CAPTION = (
    "Objective value is the anomaly score; lower values are less anomalous "
    "under REDUCE_ANOMALY_SCORE."
)
_WHAT_IF_QUALITY_CAPTION = (
    "Objective value is the predicted quality under the configured quality "
    "optimization direction."
)
_OBSERVED_RANGE_CAPTION_TEMPLATE = (
    "Observed range: {low} to {high}. This is not an approved operating limit."
)
_COHORT_COLUMN_PLACEHOLDER = "(select cohort column)"
_COHORT_FILTER_INTERPRETATION_NOTE = (
    "Anomaly scores are relative to the selected operating cohort and should "
    "not be compared directly with scores from a different cohort run."
)
_ANOMALY_EVENTS_CAPTION = (
    "Rows with the highest unsupervised anomaly scores in the independent test "
    "partition. These are statistical outliers and are not automatically defects. "
    "Original row ID is the workflow-preserved CSV row identity (0-based from the "
    "loaded file) and may differ from a user-provided identifier column."
)
_ANOMALY_CONTEXT_CAPTION = (
    "Context rows show the selected event and its neighboring records in the "
    "workflow analysis order. Values are original preprocessed-input values and "
    "are provided for interpretation only."
)
_ANOMALY_CONTEXT_ORDER_CAPTION = (
    "Original row ID is the workflow-preserved 0-based CSV row identity. "
    "Neighboring rows are based on the workflow analysis order, not arithmetic "
    "differences between original row IDs. When timestamp sorting was used, "
    "original row ID values may not be contiguous."
)
_ANOMALY_CONTEXT_FILTERED_CAPTION = (
    "When an operating cohort filter is configured, neighboring rows are "
    "adjacent only within the filtered analysis-order cohort. Rows outside "
    "the selected operating range are not re-inserted into the context window."
)
_ANOMALY_CONTEXT_CHART_MISSING_CAPTION = (
    "Selected context feature has no finite numeric values; chart is omitted."
)
_DIAGNOSIS_FACTORS_CAPTION = (
    "Features that differ most strongly between the selected anomaly group and "
    "the normal comparison group. These are associations, not proven causes. "
    "Factors are ranked using a common bounded association score. Robust "
    "z-scores are shown only as supporting statistics when the comparison-group "
    "scale is available."
)
_COMPARISON_SECTION_CAPTION = (
    "This comparison shows configuration, selected-row overlap, and "
    "diagnosis-factor rank changes. Anomaly score magnitudes are not compared "
    "across separately trained runs."
)
_COMPARISON_INTERPRETATION_NOTES = (
    "Different cohort runs use separately fitted anomaly models.\n"
    "Anomaly score magnitudes are not directly comparable.\n"
    "Shared events indicate row-selection stability, not confirmed defects.\n"
    "Shared factors indicate repeated associations, not causation."
)

WorkflowFactory = Callable[[], IndustrialProcessAnalysisWorkflow]


def _render_quick_start() -> None:
    """Compact operator guidance shown before or above CSV upload."""
    st.subheader("Quick start")
    st.markdown(
        "1. Upload a CSV file, or select the built-in manufacturing demo.\n"
        "2. Select supervised or anomaly-only analysis "
        "(or apply a demo template).\n"
        "3. Review the suggested columns and safety-critical settings.\n"
        "4. Run the analysis and inspect the report.\n"
        "5. Export reusable configuration when needed."
    )
    with st.expander("More operator guidance", expanded=False):
        st.markdown(
            "- Suggested columns are starting points; confirm roles before running.\n"
            "- SUPERVISED needs a target, supported task, and performance rules.\n"
            "- ANOMALY_ONLY does not need a target; statistical outliers are not "
            "automatically defects.\n"
            "- Recommendations use only confirmed controllable, verified, and "
            "constrained variables.\n"
            "- See `docs/USER_GUIDE.md` for a full walkthrough."
        )


def _render_analysis_mode_guidance(analysis_mode: AnalysisExecutionMode) -> None:
    """Explain mode differences without auto-selecting UI values."""
    if analysis_mode is AnalysisExecutionMode.SUPERVISED:
        with st.expander("SUPERVISED mode details", expanded=False):
            st.markdown(
                "- Target column is required.\n"
                "- Regression or another supported task is required.\n"
                "- Independent test performance acceptance rules are required.\n"
                "- Constant or all-null targets cannot be run.\n"
                "- Recommendations use only actual controllable, verified, and "
                "constrained variables."
            )
    else:
        with st.expander("ANOMALY_ONLY mode details", expanded=False):
            st.markdown(
                "- Target column is not required.\n"
                "- Runs unsupervised anomaly detection and related-variable "
                "diagnosis.\n"
                "- Statistical anomalies do not automatically mean defects.\n"
                "- Cohort filter is optional.\n"
                "- Anomaly recommendation is disabled by default.\n"
                "- When recommendation is enabled, only truly controllable "
                "variables are allowed."
            )


def _render_run_configuration_summary(
    *,
    analysis_mode: AnalysisExecutionMode,
    selected_target: str | None,
    feature_count: int,
    selected_timestamp: str | None,
    identifier_count: int,
    cohort_filter: NumericCohortFilter | None,
    restrict_operating_cohort: bool,
    recommendation_enabled: bool,
    constraint_variable_count: int,
    performance_rule_count: int | None,
) -> None:
    """Compact summary of current UI selections before readiness checks."""
    st.subheader("Current configuration summary")
    if analysis_mode is AnalysisExecutionMode.ANOMALY_ONLY:
        target_display = "Not required"
    elif selected_target is None:
        target_display = "Not selected"
    else:
        target_display = selected_target
    if not restrict_operating_cohort:
        cohort_display = "Disabled"
    elif cohort_filter is None:
        cohort_display = "Enabled (incomplete)"
    else:
        cohort_display = (
            f"{cohort_filter.column_name} "
            f"[{cohort_filter.lower_bound}, {cohort_filter.upper_bound}]"
        )
    performance_display = (
        "Not applicable"
        if performance_rule_count is None
        else str(performance_rule_count)
    )
    st.write(
        f"- Analysis mode: `{analysis_mode.value}`\n"
        f"- Target: `{target_display}`\n"
        f"- Feature count: `{feature_count}`\n"
        f"- Timestamp: `"
        f"{'Not selected' if selected_timestamp is None else selected_timestamp}`\n"
        f"- Identifier count: `{identifier_count}`\n"
        f"- Cohort filter: `{cohort_display}`\n"
        f"- Recommendation: `{'enabled' if recommendation_enabled else 'disabled'}`\n"
        f"- Constraint variable count: `{constraint_variable_count}`\n"
        f"- Performance rule count: `{performance_display}`"
    )


def _csv_bytes_from_rows(rows: list[dict[str, Any]]) -> bytes:
    """Serialize presentation rows to UTF-8 CSV bytes without temp files."""
    buffer = StringIO()
    if not rows:
        return b""
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _render_what_if_verification_section(report: WorkflowPresentationReport) -> None:
    """Render compact what-if verification under Recommendation when applicable."""
    recommendation = report.recommendation
    verification = report.recommendation_verification
    show_full = (
        recommendation is not None
        and recommendation.status is RecommendationStatus.GENERATED
        and verification is not None
        and verification.status is not WhatIfVerificationStatus.NOT_APPLICABLE
    )
    if not show_full:
        if recommendation is None or recommendation.status is not (
            RecommendationStatus.GENERATED
        ):
            st.caption(_WHAT_IF_NOT_APPLICABLE)
        return

    assert verification is not None
    st.subheader("What-if verification")
    st.write(_WHAT_IF_SECTION_CAPTION)
    st.write(
        {
            "Verification status": verification.status.value,
            "Stability classification": verification.stability_classification.value,
            "Baseline objective value": verification.baseline_objective_value,
            "Proposed objective value": verification.proposed_objective_value,
            "Neighbor scenarios": verification.neighbor_scenario_count,
            "Improving neighbors": verification.improving_neighbor_count,
            "Non-improving neighbors": verification.non_improving_neighbor_count,
            "Extrapolated scenarios": verification.extrapolated_scenario_count,
        }
    )
    st.info(verification.stability_message)
    if verification.objective is RecommendationObjective.REDUCE_ANOMALY_SCORE:
        st.caption(_WHAT_IF_ANOMALY_CAPTION)
        objective_column = "Anomaly score"
    else:
        st.caption(_WHAT_IF_QUALITY_CAPTION)
        objective_column = "Predicted quality"

    table_rows: list[dict[str, Any]] = []
    chart_labels: list[str] = []
    chart_values: list[float] = []
    for scenario in verification.scenarios:
        label = scenario.scenario_type.value
        if scenario.perturbed_variable is not None:
            label = f"{scenario.scenario_type.value}:{scenario.perturbed_variable}"
        table_rows.append(
            {
                "Scenario": scenario.scenario_id,
                "Type": scenario.scenario_type.value,
                "Perturbed variable": scenario.perturbed_variable,
                "Direction": (
                    None
                    if scenario.perturbation_direction is None
                    else scenario.perturbation_direction.value
                ),
                "Perturbed value": scenario.perturbed_value,
                "Objective value": scenario.objective_value,
                objective_column: (
                    scenario.anomaly_score
                    if verification.objective
                    is RecommendationObjective.REDUCE_ANOMALY_SCORE
                    else scenario.predicted_quality
                ),
                "Improved vs baseline": scenario.improves_over_baseline,
                "Matched/improved proposed": scenario.improves_or_matches_proposed,
                "Extrapolated": scenario.extrapolated,
            }
        )
        chart_labels.append(label)
        chart_values.append(scenario.objective_value)
    if table_rows:
        st.dataframe(table_rows, use_container_width=True)
    if chart_labels:
        chart_frame = pd.DataFrame(
            {
                "scenario": chart_labels,
                "objective_value": chart_values,
            }
        )
        st.bar_chart(chart_frame, x="scenario", y="objective_value")

    download_rows = _what_if_verification_download_rows(verification)
    st.download_button(
        "Download what-if verification CSV",
        data=_csv_bytes_from_rows(download_rows),
        file_name="recommendation_what_if_verification.csv",
        mime="text/csv",
        key="download_what_if_verification_csv",
    )
    for warning in verification.warnings:
        st.warning(warning)


def _what_if_verification_download_rows(
    verification: RecommendationWhatIfVerificationView,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not verification.scenarios:
        rows.append(
            {
                "verification_status": verification.status.value,
                "stability_classification": (
                    verification.stability_classification.value
                ),
                "baseline_objective_value": verification.baseline_objective_value,
                "proposed_objective_value": verification.proposed_objective_value,
                "scenario_id": None,
                "scenario_type": None,
                "perturbed_variable": None,
                "perturbation_direction": None,
                "perturbed_value": None,
                "objective_value": None,
                "improves_over_baseline": None,
                "improves_or_matches_proposed": None,
                "extrapolated": None,
                "warnings": (
                    verification.warnings[0] if verification.warnings else None
                ),
            }
        )
        return rows
    for scenario in verification.scenarios:
        rows.append(
            {
                "verification_status": verification.status.value,
                "stability_classification": (
                    verification.stability_classification.value
                ),
                "baseline_objective_value": verification.baseline_objective_value,
                "proposed_objective_value": verification.proposed_objective_value,
                "scenario_id": scenario.scenario_id,
                "scenario_type": scenario.scenario_type.value,
                "perturbed_variable": scenario.perturbed_variable,
                "perturbation_direction": (
                    None
                    if scenario.perturbation_direction is None
                    else scenario.perturbation_direction.value
                ),
                "perturbed_value": scenario.perturbed_value,
                "objective_value": scenario.objective_value,
                "improves_over_baseline": scenario.improves_over_baseline,
                "improves_or_matches_proposed": scenario.improves_or_matches_proposed,
                "extrapolated": scenario.extrapolated,
                "warnings": (
                    scenario.warnings[0]
                    if scenario.warnings
                    else (
                        verification.warnings[0] if verification.warnings else None
                    )
                ),
            }
        )
    return rows


def _is_anomaly_only_report(report: WorkflowPresentationReport) -> bool:
    return report.metadata.get("analysis_mode") == AnalysisExecutionMode.ANOMALY_ONLY.value


def _report_eligible_as_comparison_baseline(report: WorkflowPresentationReport) -> bool:
    if not _is_anomaly_only_report(report):
        return False
    if report.overview.status is AnalysisWorkflowStatus.REFUSED:
        return False
    if report.dataset_fingerprint is None:
        return False
    return bool(report.anomaly_events)


def _cohort_filter_display(report: WorkflowPresentationReport) -> str:
    summary = report.cohort_filter_summary
    if not summary.configured:
        return "Not configured"
    column_name = summary.column_name
    if column_name is None:
        return summary.range_display
    return f"{column_name}: {summary.range_display}"


def _load_baseline_report_from_session() -> WorkflowPresentationReport | None:
    cached = st.session_state.get(_SESSION_BASELINE_REPORT_KEY)
    if cached is None:
        return None
    if not isinstance(cached, dict):
        st.session_state.pop(_SESSION_BASELINE_REPORT_KEY, None)
        st.session_state.pop(_SESSION_BASELINE_LABEL_KEY, None)
        return None
    try:
        return WorkflowPresentationReport.model_validate(cached)
    except ValidationError:
        st.session_state.pop(_SESSION_BASELINE_REPORT_KEY, None)
        st.session_state.pop(_SESSION_BASELINE_LABEL_KEY, None)
        return None


def _save_comparison_baseline(report: WorkflowPresentationReport) -> None:
    st.session_state[_SESSION_BASELINE_REPORT_KEY] = report.model_dump(mode="json")
    st.session_state[_SESSION_BASELINE_LABEL_KEY] = (
        f"{_analysis_mode_label(report)} | {_cohort_filter_display(report)}"
    )


def _clear_comparison_baseline() -> None:
    st.session_state.pop(_SESSION_BASELINE_REPORT_KEY, None)
    st.session_state.pop(_SESSION_BASELINE_LABEL_KEY, None)


def _analysis_mode_label(report: WorkflowPresentationReport) -> str:
    mode = report.metadata.get("analysis_mode")
    if isinstance(mode, str) and mode.strip() != "":
        return mode
    return _NOT_AVAILABLE


def _event_overlap_presence_label(status: EventOverlapPresenceStatus) -> str:
    if status is EventOverlapPresenceStatus.SHARED:
        return "Shared"
    if status is EventOverlapPresenceStatus.BASELINE_ONLY:
        return "Baseline only"
    return "Current only"


def _factor_presence_label(status: DiagnosisFactorPresenceStatus) -> str:
    if status is DiagnosisFactorPresenceStatus.SHARED:
        return "Shared"
    if status is DiagnosisFactorPresenceStatus.BASELINE_ONLY:
        return "Baseline only"
    return "Current only"


def _event_overlap_table_rows(
    entries: Sequence[EventOverlapEntryView],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        rows.append(
            {
                "Status": _event_overlap_presence_label(entry.presence_status),
                "Original row ID": str(entry.original_row_id),
                "Baseline rank": (
                    _NOT_AVAILABLE
                    if entry.baseline_rank is None
                    else str(entry.baseline_rank)
                ),
                "Current rank": (
                    _NOT_AVAILABLE
                    if entry.current_rank is None
                    else str(entry.current_rank)
                ),
            }
        )
    return rows


def _event_overlap_download_rows(overlap: EventOverlapView) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in overlap.entries:
        rows.append(
            {
                "status": entry.presence_status.value,
                "original_row_id": entry.original_row_id,
                "baseline_rank": entry.baseline_rank,
                "current_rank": entry.current_rank,
            }
        )
    return rows


def _factor_comparison_table_rows(
    factors: Sequence[DiagnosisFactorComparisonView],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for factor in factors:
        rows.append(
            {
                "Feature": factor.feature_name,
                "Presence": _factor_presence_label(factor.presence_status),
                "Baseline rank": (
                    _NOT_AVAILABLE
                    if factor.baseline_rank is None
                    else str(factor.baseline_rank)
                ),
                "Current rank": (
                    _NOT_AVAILABLE
                    if factor.current_rank is None
                    else str(factor.current_rank)
                ),
                "Baseline direction": _display_optional(factor.baseline_direction),
                "Current direction": _display_optional(factor.current_direction),
            }
        )
    return rows


def _factor_comparison_download_rows(
    factors: Sequence[DiagnosisFactorComparisonView],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for factor in factors:
        rows.append(
            {
                "feature_name": factor.feature_name,
                "presence_status": factor.presence_status.value,
                "baseline_rank": factor.baseline_rank,
                "current_rank": factor.current_rank,
                "baseline_direction": factor.baseline_direction,
                "current_direction": factor.current_direction,
            }
        )
    return rows


def _configuration_comparison_table_rows(
    comparison: AnomalyRunComparisonView,
) -> list[dict[str, Any]]:
    config = comparison.configuration
    return [
        {
            "Metric": "Cohort filter",
            "Baseline": config.baseline_cohort_description,
            "Current": config.current_cohort_description,
        },
        {
            "Metric": "Analysis rows",
            "Baseline": str(config.baseline_analysis_rows),
            "Current": str(config.current_analysis_rows),
        },
        {
            "Metric": "Feature count",
            "Baseline": _display_optional(config.baseline_feature_count),
            "Current": _display_optional(config.current_feature_count),
        },
        {
            "Metric": "Train rows",
            "Baseline": str(config.baseline_train_row_count),
            "Current": str(config.current_train_row_count),
        },
        {
            "Metric": "Validation rows",
            "Baseline": str(config.baseline_validation_row_count),
            "Current": str(config.current_validation_row_count),
        },
        {
            "Metric": "Test rows",
            "Baseline": str(config.baseline_test_row_count),
            "Current": str(config.current_test_row_count),
        },
        {
            "Metric": "Anomaly model",
            "Baseline": _display_optional(config.baseline_anomaly_model),
            "Current": _display_optional(config.current_anomaly_model),
        },
        {
            "Metric": "Anomaly event count",
            "Baseline": str(config.baseline_anomaly_event_count),
            "Current": str(config.current_anomaly_event_count),
        },
        {
            "Metric": "Diagnosis factor count",
            "Baseline": str(config.baseline_diagnosis_factor_count),
            "Current": str(config.current_diagnosis_factor_count),
        },
        {
            "Metric": "Operating row ID",
            "Baseline": _display_optional(config.baseline_operating_row_id),
            "Current": _display_optional(config.current_operating_row_id),
        },
    ]


def _render_baseline_summary(baseline: WorkflowPresentationReport) -> None:
    st.write("Baseline available")
    st.markdown(
        f"- Analysis mode: `{_analysis_mode_label(baseline)}`\n"
        f"- Cohort filter: `{_cohort_filter_display(baseline)}`\n"
        f"- Analysis rows: `{baseline.data_summary.cohort_row_count}`\n"
        f"- Anomaly events: `{baseline.data_summary.anomaly_event_count}`\n"
        f"- Anomaly model: `"
        f"{_display_optional(baseline.model_summary.anomaly_model_key)}`"
    )
    fingerprint = baseline.dataset_fingerprint
    if fingerprint is not None:
        with st.expander("Baseline debugging details", expanded=False):
            st.caption(f"Dataset fingerprint prefix: `{fingerprint[:12]}`")


def _render_run_comparison_section(report: WorkflowPresentationReport) -> None:
    st.header("Run comparison")
    baseline = _load_baseline_report_from_session()
    eligible = _report_eligible_as_comparison_baseline(report)

    if eligible:
        save_label = (
            "Replace comparison baseline"
            if baseline is not None
            else "Save current report as comparison baseline"
        )
        if st.button(save_label, key="save_comparison_baseline"):
            _save_comparison_baseline(report)
            st.success("Current presentation report saved as comparison baseline.")
            baseline = report

    if baseline is not None:
        _render_baseline_summary(baseline)
        if st.button(
            "Clear comparison baseline",
            key="clear_comparison_baseline",
        ):
            _clear_comparison_baseline()
            st.success("Comparison baseline cleared. The current report is unchanged.")
            return

    if baseline is None:
        return

    comparison = build_anomaly_run_comparison(baseline, report)
    if not comparison.compatible:
        st.warning(comparison.compatibility_message)
        return

    st.header("Baseline vs current anomaly analysis")
    st.caption(_COMPARISON_SECTION_CAPTION)

    st.subheader("Configuration comparison")
    st.dataframe(
        _configuration_comparison_table_rows(comparison),
        use_container_width=True,
        hide_index=True,
    )

    overlap = comparison.event_overlap
    st.subheader("Event overlap")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Shared events", overlap.shared_event_count)
    metric_cols[1].metric(
        "Baseline-only events",
        len(overlap.baseline_only_original_row_ids),
    )
    metric_cols[2].metric(
        "Current-only events",
        len(overlap.current_only_original_row_ids),
    )
    st.dataframe(
        _event_overlap_table_rows(overlap.entries),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        label="Download event overlap CSV",
        data=_csv_bytes_from_rows(_event_overlap_download_rows(overlap)),
        file_name="anomaly_event_overlap.csv",
        mime="text/csv",
        key="download_anomaly_event_overlap_csv",
    )

    st.subheader("Diagnosis factor comparison")
    st.dataframe(
        _factor_comparison_table_rows(comparison.factor_comparison),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        label="Download factor comparison CSV",
        data=_csv_bytes_from_rows(
            _factor_comparison_download_rows(comparison.factor_comparison)
        ),
        file_name="diagnosis_factor_comparison.csv",
        mime="text/csv",
        key="download_diagnosis_factor_comparison_csv",
    )

    st.subheader("Interpretation note")
    st.info(_COMPARISON_INTERPRETATION_NOTES)


def _anomaly_event_table_rows(
    events: Sequence[AnomalyEventView],
) -> list[dict[str, Any]]:
    return [
        {
            "Rank": event.rank,
            "Original row ID": event.original_row_id,
            "Anomaly score": event.anomaly_score,
            "Operating row": "Yes" if event.is_operating_row else "No",
        }
        for event in events
    ]


def _anomaly_event_download_rows(
    events: Sequence[AnomalyEventView],
) -> list[dict[str, Any]]:
    return [
        {
            "rank": event.rank,
            "original_row_id": event.original_row_id,
            "anomaly_score": event.anomaly_score,
            "is_operating_row": event.is_operating_row,
            "selection_source": event.selection_source,
            "score_direction": event.score_direction,
            "is_anomaly_flagged": event.is_anomaly_flagged,
        }
        for event in events
    ]


def _diagnosis_factor_table_rows(
    factors: Sequence[DiagnosisFactorView],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for factor in factors:
        difference = factor.raw_group_difference
        if difference is None:
            difference = factor.effect_size
        robust_z_display: float | str
        if factor.robust_z_score is None:
            robust_z_display = _NOT_AVAILABLE
        else:
            robust_z_display = factor.robust_z_score
        rows.append(
            {
                "Rank": factor.rank,
                "Feature": factor.feature_name,
                "Ranking score": factor.diagnostic_score,
                "Direction": factor.direction,
                "Anomaly group": factor.anomaly_group_value,
                "Normal group": factor.normal_group_value,
                "Difference": difference,
                "Robust scale": factor.robust_scale,
                "Robust z-score": robust_z_display,
                "Scale status": factor.robust_scale_status,
                "Confidence": factor.confidence,
                "Source": factor.source,
            }
        )
    return rows


def _diagnosis_factor_download_rows(
    factors: Sequence[DiagnosisFactorView],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for factor in factors:
        difference = factor.raw_group_difference
        if difference is None:
            difference = factor.effect_size
        rows.append(
            {
                "rank": factor.rank,
                "feature_name": factor.feature_name,
                "ranking_score": factor.diagnostic_score,
                "direction": factor.direction,
                "anomaly_group_value": factor.anomaly_group_value,
                "normal_group_value": factor.normal_group_value,
                "raw_group_difference": difference,
                "robust_scale": factor.robust_scale,
                "robust_z_score": factor.robust_z_score,
                "robust_scale_status": factor.robust_scale_status,
                "confidence": factor.confidence,
                "source": factor.source,
            }
        )
    return rows


def _context_event_option_label(
    window: AnomalyContextWindowView,
    *,
    is_operating: bool,
) -> str:
    suffix = " (operating row)" if is_operating else ""
    return (
        f"Rank {window.event_rank} — Original row ID "
        f"{window.center_original_row_id}{suffix}"
    )


def _default_context_window_index(
    windows: Sequence[AnomalyContextWindowView],
    events: Sequence[AnomalyEventView],
) -> int:
    operating_ids = {
        event.original_row_id for event in events if event.is_operating_row
    }
    for index, window in enumerate(windows):
        if window.center_original_row_id in operating_ids:
            return index
    return 0


def _context_table_rows(window: AnomalyContextWindowView) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in window.rows:
        item: dict[str, Any] = {
            "Relative offset": row.relative_offset,
            "Analysis position": row.analysis_position,
            "Original row ID": row.original_row_id,
            "Center event": "Yes" if row.is_center_event else "No",
            "Selected anomaly event": (
                "Yes" if row.is_selected_anomaly_event else "No"
            ),
        }
        if row.timestamp_value is not None or any(
            other.timestamp_value is not None for other in window.rows
        ):
            item["Timestamp"] = row.timestamp_value
        for identifier in row.identifier_values:
            item[identifier.column_name] = identifier.value
        feature_lookup = {value.feature_name: value.value for value in row.feature_values}
        for feature_name in window.feature_names:
            item[feature_name] = feature_lookup.get(feature_name)
        rows.append(item)
    return rows


def _context_download_rows(window: AnomalyContextWindowView) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in window.rows:
        item: dict[str, Any] = {
            "event_rank": window.event_rank,
            "center_original_row_id": window.center_original_row_id,
            "center_anomaly_score": window.center_anomaly_score,
            "order_basis": window.order_basis.value,
            "radius": window.radius,
            "analysis_position": row.analysis_position,
            "original_row_id": row.original_row_id,
            "relative_offset": row.relative_offset,
            "is_center_event": row.is_center_event,
            "is_selected_anomaly_event": row.is_selected_anomaly_event,
            "timestamp_value": row.timestamp_value,
        }
        for identifier in row.identifier_values:
            item[f"identifier_{identifier.column_name}"] = identifier.value
        feature_lookup = {value.feature_name: value.value for value in row.feature_values}
        for feature_name in window.feature_names:
            item[feature_name] = feature_lookup.get(feature_name)
        rows.append(item)
    return rows


def _feature_values_are_numeric(window: AnomalyContextWindowView, feature_name: str) -> bool:
    saw_value = False
    for row in window.rows:
        for item in row.feature_values:
            if item.feature_name != feature_name:
                continue
            if item.value is None:
                continue
            saw_value = True
            if isinstance(item.value, bool) or not isinstance(item.value, (int, float)):
                return False
    return saw_value


def _context_chart_rows(
    window: AnomalyContextWindowView,
    feature_name: str,
) -> list[dict[str, Any]]:
    chart_rows: list[dict[str, Any]] = []
    for row in window.rows:
        value = next(
            (
                item.value
                for item in row.feature_values
                if item.feature_name == feature_name
            ),
            None,
        )
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            continue
        chart_rows.append(
            {
                "relative_offset": row.relative_offset,
                "value": numeric,
            }
        )
    return chart_rows


def _context_chart_y_domain(values: Sequence[float]) -> tuple[float, float] | None:
    """Return an explicit Y-axis domain that keeps constant and narrow series visible.

    Does not force zero into the domain. Padding is deterministic.
    """
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    lower = min(finite)
    upper = max(finite)
    if lower == upper:
        padding = max(abs(lower) * 0.02, 1e-6)
        return lower - padding, upper + padding
    padding = max((upper - lower) * 0.05, 1e-6)
    return lower - padding, upper + padding


def _build_anomaly_context_feature_chart(
    chart_rows: Sequence[Mapping[str, Any]],
    feature_name: str,
) -> alt.Chart | None:
    """Build an Altair line chart with an explicit Y domain for context values."""
    values = [float(row["value"]) for row in chart_rows]
    domain = _context_chart_y_domain(values)
    if domain is None:
        return None
    frame = pd.DataFrame(
        [
            {
                "relative_offset": int(row["relative_offset"]),
                "value": float(row["value"]),
            }
            for row in chart_rows
        ]
    ).sort_values("relative_offset", kind="mergesort")
    chart = cast(
        alt.Chart,
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(
            x=alt.X("relative_offset:Q", title="Relative offset"),
            y=alt.Y(
                "value:Q",
                title=feature_name,
                scale=alt.Scale(domain=list(domain), zero=False, nice=False),
            ),
        ),
    )
    return chart


def render_app(
    *,
    workflow_factory: WorkflowFactory = create_default_analysis_workflow,
    request_builder: WorkflowUiRequestBuilder | None = None,
    report_builder: AnalysisWorkflowReportBuilder | None = None,
    config: StreamlitUiConfig | None = None,
    column_configurator: AutomaticColumnConfigurator | None = None,
) -> None:
    """Render the Streamlit MVP page and execute the analysis workflow on submit.

    Args:
        workflow_factory: Callable that returns a fresh workflow instance.
        request_builder: Optional request builder. Created per call when omitted.
        report_builder: Optional presentation builder. Created per call when omitted.
        config: Optional UI configuration. Created per call when omitted.
        column_configurator: Optional column suggester. Created per call when omitted.
    """
    ui_config = (
        config.model_copy(deep=True) if config is not None else StreamlitUiConfig()
    )
    active_request_builder = (
        request_builder
        if request_builder is not None
        else WorkflowUiRequestBuilder()
    )
    active_report_builder = (
        report_builder
        if report_builder is not None
        else AnalysisWorkflowReportBuilder()
    )
    active_configurator = (
        column_configurator
        if column_configurator is not None
        else AutomaticColumnConfigurator()
    )

    st.set_page_config(
        page_title=ui_config.page_title,
        page_icon=ui_config.page_icon,
        layout="wide",
    )

    st.title(ui_config.page_title)
    st.markdown(
        "Upload a process CSV or load the built-in manufacturing demo, configure "
        "columns and acceptance rules, then run the industrial analysis workflow."
    )
    st.info(
        "Model-based decision support only. These outputs do not guarantee "
        "real-process improvement and are not operational commands."
    )
    _render_quick_start()

    if SESSION_DATA_SOURCE_KEY not in st.session_state:
        st.session_state[SESSION_DATA_SOURCE_KEY] = DATA_SOURCE_UPLOAD_CSV
    data_source = st.radio(
        "Data source",
        options=[DATA_SOURCE_UPLOAD_CSV, DATA_SOURCE_BUILTIN_DEMO],
        key=SESSION_DATA_SOURCE_KEY,
        horizontal=True,
        help=(
            "Upload CSV keeps the existing file workflow. Built-in manufacturing "
            "demo loads a synthetic in-memory dataset without writing a CSV file."
        ),
    )
    previous_data_source = st.session_state.get(SESSION_DATA_SOURCE_PREVIOUS_KEY)
    if (
        previous_data_source is not None
        and previous_data_source != data_source
    ):
        clear_stale_analysis_session_state(
            cast(MutableMapping[str, Any], st.session_state)
        )
        st.session_state.pop(SESSION_DEMO_APPLY_SUCCESS_KEY, None)
        st.session_state.pop(SESSION_DEMO_APPLY_SUMMARY_KEY, None)
        if is_builtin_demo_source(previous_data_source) and not is_builtin_demo_source(
            data_source
        ):
            clear_demo_recommendation_session_state(
                cast(MutableMapping[str, Any], st.session_state)
            )
    st.session_state[SESSION_DATA_SOURCE_PREVIOUS_KEY] = data_source
    demo_source_active = is_builtin_demo_source(data_source)

    columns: list[str] = []
    analysis_frame: pl.DataFrame | None = None
    preview_frame: pl.DataFrame | None = None
    upload_bytes: bytes | None = None
    config_report: UiColumnConfigurationReport | None = None

    if demo_source_active:
        demo_summary = None
        try:
            demo_frame, demo_summary = load_builtin_demo_dataset()
            upload_bytes = demo_dataset_to_workflow_csv_bytes(demo_frame)
            columns, analysis_frame, preview_frame = _read_csv_for_ui(
                upload_bytes,
                preview_row_count=ui_config.preview_row_count,
            )
            columns = [name for name in columns if name != _ORIGINAL_ROW_ID]
            if analysis_frame is not None and _ORIGINAL_ROW_ID in analysis_frame.columns:
                analysis_frame = analysis_frame.drop(_ORIGINAL_ROW_ID)
            config_report = active_configurator.analyze(analysis_frame)
        except (pl.exceptions.PolarsError, ValueError, TypeError) as exc:
            _display_user_error(exc, area="Built-in demo dataset")
            upload_bytes = None
            columns = []
            analysis_frame = None
            preview_frame = None
            config_report = None
            demo_summary = None
        if demo_summary is not None:
            st.info(
                "Source: synthetic built-in manufacturing demo data "
                "(not production data)."
            )
            meta_cols = st.columns(4)
            meta_cols[0].metric("Rows", demo_summary.row_count)
            meta_cols[1].metric("Columns", demo_summary.column_count)
            meta_cols[2].metric("Seed", demo_summary.random_seed)
            meta_cols[3].metric("Anomaly rows", demo_summary.anomaly_row_count)
            st.caption(
                "Analysis on this dataset does not represent production accuracy. "
                "Selecting the demo source or changing a template does not run "
                "analysis."
            )
            if SESSION_DEMO_TEMPLATE_KEY not in st.session_state:
                st.session_state[SESSION_DEMO_TEMPLATE_KEY] = (
                    demo_template_options()[0]
                )
            st.selectbox(
                "Demo analysis template",
                options=list(demo_template_options()),
                key=SESSION_DEMO_TEMPLATE_KEY,
                help=(
                    "Choose a leakage-safe template, then click "
                    "Apply demo configuration. Changing the template alone "
                    "does not overwrite the current configuration."
                ),
            )
            if st.button(
                "Apply demo configuration",
                type="secondary",
                key="apply_demo_configuration",
            ):
                template_label = str(
                    st.session_state.get(SESSION_DEMO_TEMPLATE_KEY)
                )
                demo_preset = build_demo_configuration_preset(template_label)
                clear_stale_analysis_session_state(
                    cast(MutableMapping[str, Any], st.session_state)
                )
                st.session_state[SESSION_PRESET_PENDING_APPLY_KEY] = (
                    build_configuration_preset_session_updates(demo_preset)
                )
                st.session_state[SESSION_DEMO_APPLY_SUCCESS_KEY] = True
                st.session_state[SESSION_DEMO_APPLY_SUMMARY_KEY] = (
                    demo_configuration_apply_summary(demo_preset)
                )
                st.rerun()
            if st.session_state.pop(SESSION_DEMO_APPLY_SUCCESS_KEY, False):
                apply_summary = st.session_state.get(SESSION_DEMO_APPLY_SUMMARY_KEY)
                st.success("Demo configuration applied.")
                if isinstance(apply_summary, dict):
                    st.caption(
                        "mode="
                        f"{apply_summary.get('analysis_mode')}; "
                        f"target={apply_summary.get('target')}; "
                        f"features={apply_summary.get('feature_count')}; "
                        "operating_point="
                        f"{apply_summary.get('operating_point_selection_mode')}; "
                        "recommendation="
                        f"{apply_summary.get('anomaly_recommendation_enabled')}."
                    )
                    controllable_count = apply_summary.get(
                        "verified_controllable_count", 0
                    )
                    if (
                        isinstance(controllable_count, int)
                        and controllable_count > 0
                    ):
                        controllable_names = apply_summary.get(
                            "verified_controllable_variables", []
                        )
                        bounds = apply_summary.get("approved_synthetic_bounds", [])
                        bound_parts: list[str] = []
                        if isinstance(bounds, list):
                            for item in bounds:
                                if not isinstance(item, dict):
                                    continue
                                variable = item.get("variable")
                                lower = item.get("lower_bound")
                                upper = item.get("upper_bound")
                                if (
                                    isinstance(variable, str)
                                    and isinstance(lower, (int, float))
                                    and isinstance(upper, (int, float))
                                ):
                                    bound_parts.append(
                                        f"{variable}=[{lower}, {upper}]"
                                    )
                        names_text = (
                            ", ".join(str(name) for name in controllable_names)
                            if isinstance(controllable_names, list)
                            else ""
                        )
                        st.caption(
                            f"Verified controllable variables: {controllable_count}"
                            + (f" ({names_text})" if names_text else "")
                            + (
                                f"; approved synthetic bounds: "
                                f"{'; '.join(bound_parts)}"
                                if bound_parts
                                else ""
                            )
                            + ". These limits are demonstration-only and must "
                            "not be reused as real equipment operating limits."
                        )
    else:
        uploaded = st.file_uploader(
            "CSV upload",
            type=["csv"],
            accept_multiple_files=False,
            help="CSV files only. Uploaded content is not stored permanently.",
        )
        if uploaded is not None:
            try:
                upload_bytes = bytes(uploaded.getvalue())
                if len(upload_bytes) == 0:
                    st.error("Uploaded CSV is empty. Provide a non-empty CSV file.")
                    upload_bytes = None
                elif len(upload_bytes) > ui_config.maximum_upload_bytes:
                    st.error(
                        "Uploaded CSV exceeds the configured maximum upload size. "
                        "Reduce the file size and try again."
                    )
                    upload_bytes = None
                else:
                    columns, analysis_frame, preview_frame = _read_csv_for_ui(
                        upload_bytes,
                        preview_row_count=ui_config.preview_row_count,
                    )
                    columns = [name for name in columns if name != _ORIGINAL_ROW_ID]
                    if (
                        analysis_frame is not None
                        and _ORIGINAL_ROW_ID in analysis_frame.columns
                    ):
                        analysis_frame = analysis_frame.drop(_ORIGINAL_ROW_ID)
                    config_report = active_configurator.analyze(analysis_frame)
            except (pl.exceptions.PolarsError, UnicodeDecodeError, ValueError) as exc:
                _display_user_error(exc, area="CSV upload / parsing")
                upload_bytes = None
                columns = []
                analysis_frame = None
                preview_frame = None
                config_report = None

    if preview_frame is not None:
        preview_label = (
            "Show dataset preview" if demo_source_active else "Show CSV preview"
        )
        show_preview = st.checkbox(preview_label, value=False)
        if show_preview:
            st.caption(
                f"Showing up to {ui_config.preview_row_count} rows for column review."
            )
            st.dataframe(preview_frame.to_dicts(), use_container_width=True)

    if demo_source_active and columns:
        ground_truth_present = ground_truth_metadata_columns_present(columns)
        if ground_truth_present:
            st.markdown("Ground-truth demo metadata")
            st.caption(
                "The following columns are synthetic evaluation labels only. "
                "They are retained for preview and later evaluation, must not be "
                "selected as model features, and are not used in workflow "
                "execution: "
                + ", ".join(ground_truth_present)
                + "."
            )

    if not columns or config_report is None:
        if demo_source_active:
            st.warning(
                "Built-in manufacturing demo could not be loaded. "
                "Try Upload CSV or reload the page."
            )
        else:
            st.warning(
                "Upload a valid CSV or select the built-in manufacturing demo "
                "to configure columns and run the analysis workflow."
            )
        _render_cached_report_if_any()
        return

    _reset_column_widget_state_if_schema_changed(columns)
    _consume_pending_configuration_preset_apply()

    st.subheader("Analysis mode")
    if _SESSION_ANALYSIS_MODE_KEY not in st.session_state:
        st.session_state[_SESSION_ANALYSIS_MODE_KEY] = (
            AnalysisExecutionMode.SUPERVISED.value
        )
    previous_mode = str(st.session_state.get(_SESSION_ANALYSIS_MODE_KEY))
    analysis_mode = AnalysisExecutionMode(
        st.selectbox(
            "Analysis mode",
            options=[
                AnalysisExecutionMode.SUPERVISED.value,
                AnalysisExecutionMode.ANOMALY_ONLY.value,
            ],
            key=_SESSION_ANALYSIS_MODE_KEY,
            help=(
                "SUPERVISED requires a target and recommendation settings. "
                "ANOMALY_ONLY runs label-free anomaly detection without a target."
            ),
        )
    )
    _render_analysis_mode_guidance(analysis_mode)
    if (
        previous_mode == AnalysisExecutionMode.SUPERVISED.value
        and analysis_mode is AnalysisExecutionMode.ANOMALY_ONLY
    ):
        st.session_state[_SESSION_TARGET_KEY] = _TARGET_PLACEHOLDER
    anomaly_only = analysis_mode is AnalysisExecutionMode.ANOMALY_ONLY

    st.subheader("Column configuration")
    st.caption(
        "Suggested target candidates are based on column structure and names. "
        "The user must confirm the final prediction target for supervised mode."
    )

    _render_configuration_summary(config_report)

    selected_target: str | None = None
    target_has_usable_variation = True
    if anomaly_only:
        st.info("Target column: not required for anomaly-only analysis.")
    else:
        target_options = _build_target_options(columns, config_report.target_candidates)
        if _SESSION_TARGET_KEY not in st.session_state:
            st.session_state[_SESSION_TARGET_KEY] = _TARGET_PLACEHOLDER
        target_selection = st.selectbox(
            "Target column",
            options=target_options,
            key=_SESSION_TARGET_KEY,
            help="Automatic suggestions are not final. Confirm the prediction target.",
        )
        selected_target = (
            None if target_selection == _TARGET_PLACEHOLDER else target_selection
        )
        if selected_target is None:
            st.warning("Select a target column before running analysis.")
        elif analysis_frame is not None and selected_target in analysis_frame.columns:
            target_assessment = evaluate_target_suitability(
                analysis_frame,
                selected_target,
                requested_task=None,
            )
            target_has_usable_variation = target_assessment.suitable
            if not target_has_usable_variation:
                if target_assessment.is_constant or target_assessment.is_all_null:
                    st.error(_TARGET_CONSTANT_OPERATOR_MESSAGE)
                else:
                    st.error(target_assessment.message)

    timestamp_help = _timestamp_help_text(config_report)
    timestamp_options = _build_timestamp_options(
        columns,
        config_report.timestamp_candidates,
    )
    if _SESSION_TIMESTAMP_KEY not in st.session_state:
        st.session_state[_SESSION_TIMESTAMP_KEY] = _TIMESTAMP_NONE
    _prune_selection_conflicts(
        selected_target=selected_target,
        timestamp_options=timestamp_options,
    )
    timestamp_selection = st.selectbox(
        "Timestamp column (optional)",
        options=timestamp_options,
        key=_SESSION_TIMESTAMP_KEY,
        help=timestamp_help,
    )
    selected_timestamp = (
        None if timestamp_selection == _TIMESTAMP_NONE else timestamp_selection
    )

    st.caption("Identifier suggestions are provisional. Review before running.")
    identifier_default = [
        name
        for name in config_report.identifier_candidates
        if name != selected_target and name != selected_timestamp
    ]
    if _SESSION_IDENTIFIERS_KEY not in st.session_state:
        st.session_state[_SESSION_IDENTIFIERS_KEY] = identifier_default
    identifier_columns = st.multiselect(
        "Identifier columns",
        options=columns,
        key=_SESSION_IDENTIFIERS_KEY,
        help=(
            "Suggested only when an identifier-like column has at least two "
            "distinct non-null values. Constant or all-null identifier-like "
            "columns are not selected by default."
        ),
    )
    for warning in config_report.warnings:
        if "resembles an identifier but is constant" in warning:
            st.info(warning)

    excluded_default = [
        name
        for name in config_report.excluded_candidates
        if name != selected_target
        and name != selected_timestamp
        and name not in identifier_columns
        and name not in config_report.target_candidates
        and name not in config_report.identifier_candidates
        and name not in config_report.timestamp_candidates
    ]
    if _SESSION_EXCLUDED_KEY not in st.session_state:
        st.session_state[_SESSION_EXCLUDED_KEY] = excluded_default
    excluded_columns = st.multiselect(
        "Excluded columns",
        options=columns,
        key=_SESSION_EXCLUDED_KEY,
        help=(
            "Explicit exclusions only. Target, identifier, and timestamp suggestions "
            "are omitted from features automatically and are not pre-selected here."
        ),
    )

    use_recommended_features = st.checkbox(
        "Use recommended numeric feature set",
        value=True,
        key=SESSION_USE_RECOMMENDED_FEATURES_KEY,
        help=(
            "When enabled, the compact recommended feature set is used and feature "
            "tags are not expanded on the main form."
        ),
    )

    base_recommended = list(config_report.recommended_feature_columns)
    if demo_source_active:
        base_recommended = filter_ground_truth_from_recommended_features(
            base_recommended
        )
    active_features = resolve_active_feature_columns(
        base_recommended,
        selected_target=selected_target,
        selected_timestamp=selected_timestamp,
        selected_identifiers=identifier_columns,
        selected_excluded=excluded_columns,
    )

    if use_recommended_features:
        feature_columns = list(active_features)
        st.write(f"Recommended features in use: {len(feature_columns)}")
        with st.expander("Review recommended feature columns", expanded=False):
            st.dataframe(
                [{"column": name} for name in feature_columns],
                use_container_width=True,
            )
    else:
        with st.expander("Manual feature selection", expanded=True):
            select_all = st.checkbox(
                "Select all recommended features",
                value=True,
                key="manual_select_all_recommended",
            )
            manual_default = list(active_features) if select_all else []
            if SESSION_EXPLICIT_FEATURES_KEY not in st.session_state:
                st.session_state[SESSION_EXPLICIT_FEATURES_KEY] = manual_default
            feature_columns = st.multiselect(
                "Feature columns",
                options=columns,
                key=SESSION_EXPLICIT_FEATURES_KEY,
            )

    cohort_filter: NumericCohortFilter | None = None
    cohort_filter_ready = True
    restrict_operating_cohort = False
    modeling_feature_columns = list(feature_columns)
    if anomaly_only:
        st.subheader("Operating cohort filter")
        st.caption(
            "Restrict anomaly analysis to a user-confirmed numeric operating "
            "range. This can reduce false positives caused by comparing "
            "different operating regimes."
        )
        restrict_operating_cohort = st.checkbox(
            "Restrict analysis to an operating range",
            value=False,
            key="ui_restrict_operating_cohort",
        )
        if restrict_operating_cohort:
            cohort_candidates = (
                list_numeric_cohort_filter_candidates(
                    analysis_frame,
                    active_feature_columns=list(feature_columns),
                    identifier_columns=list(identifier_columns),
                    timestamp_column=selected_timestamp,
                )
                if analysis_frame is not None
                else []
            )
            cohort_options = [_COHORT_COLUMN_PLACEHOLDER, *cohort_candidates]
            selected_cohort_column = st.selectbox(
                "Cohort column",
                options=cohort_options,
                index=0,
                key="ui_cohort_filter_column",
            )
            lower_text = st.text_input(
                "Lower bound",
                value="",
                key="ui_cohort_filter_lower",
                help="Enter a finite number. No default bound is inferred.",
            )
            upper_text = st.text_input(
                "Upper bound",
                value="",
                key="ui_cohort_filter_upper",
                help="Enter a finite number. No default bound is inferred.",
            )
            include_lower = st.checkbox(
                "Include lower bound",
                value=True,
                key="ui_cohort_filter_include_lower",
            )
            include_upper = st.checkbox(
                "Include upper bound",
                value=True,
                key="ui_cohort_filter_include_upper",
            )
            exclude_filter_column = st.checkbox(
                "Exclude cohort column from anomaly features",
                value=True,
                key="ui_cohort_filter_exclude_feature",
            )

            if (
                selected_cohort_column != _COHORT_COLUMN_PLACEHOLDER
                and analysis_frame is not None
            ):
                observed = observed_numeric_range(
                    analysis_frame,
                    selected_cohort_column,
                )
                if observed is not None:
                    st.caption(
                        "Observed range in uploaded data: "
                        f"{observed[0]} to {observed[1]}"
                    )

            lower_value: float | None = None
            upper_value: float | None = None
            bounds_ok = True
            try:
                if str(lower_text).strip() == "" or str(upper_text).strip() == "":
                    bounds_ok = False
                else:
                    lower_value = float(str(lower_text).strip())
                    upper_value = float(str(upper_text).strip())
                    if not math.isfinite(lower_value) or not math.isfinite(upper_value):
                        bounds_ok = False
                    elif lower_value > upper_value:
                        bounds_ok = False
            except ValueError:
                bounds_ok = False

            preview_retained: int | None = None
            if (
                selected_cohort_column != _COHORT_COLUMN_PLACEHOLDER
                and bounds_ok
                and lower_value is not None
                and upper_value is not None
                and analysis_frame is not None
            ):
                try:
                    preview_filter = NumericCohortFilter(
                        column_name=selected_cohort_column,
                        lower_bound=lower_value,
                        upper_bound=upper_value,
                        include_lower=include_lower,
                        include_upper=include_upper,
                        exclude_filter_column_from_features=exclude_filter_column,
                    )
                    preview_retained = preview_numeric_cohort_filter_row_count(
                        analysis_frame,
                        preview_filter,
                    )
                    st.caption(
                        f"Preview: {preview_retained:,} of "
                        f"{analysis_frame.height:,} rows would be included."
                    )
                    cohort_filter = preview_filter
                except (DataValidationError, ValidationError, ValueError) as exc:
                    cohort_filter_ready = False
                    st.warning(_operator_facing_error_message(exc, area="cohort filter"))
            else:
                cohort_filter_ready = False

            if preview_retained is not None and preview_retained <= 0:
                cohort_filter_ready = False
                st.warning(
                    "The configured operating range retains zero rows. "
                    "Adjust the bounds before running analysis."
                )

            if (
                cohort_filter is not None
                and cohort_filter.exclude_filter_column_from_features
                and cohort_filter.column_name in modeling_feature_columns
            ):
                modeling_feature_columns = [
                    name
                    for name in modeling_feature_columns
                    if name != cohort_filter.column_name
                ]
                st.caption(
                    "Active modeling features after excluding the cohort "
                    f"column: {len(modeling_feature_columns)}"
                )

            if selected_cohort_column == _COHORT_COLUMN_PLACEHOLDER:
                st.warning("Select a cohort column before running analysis.")
            if not bounds_ok:
                st.warning(
                    "Provide finite lower and upper bounds with lower <= upper."
                )
        else:
            cohort_filter = None
            cohort_filter_ready = True

    st.subheader("Column role overrides")
    st.caption(
        "Overrides are optional. Controllability is not confirmed by selecting "
        "a role. Choose columns explicitly; wide schemas do not render one widget "
        "per feature."
    )
    override_candidates = list(feature_columns)
    if SESSION_ROLE_OVERRIDE_COLUMNS_KEY not in st.session_state:
        st.session_state[SESSION_ROLE_OVERRIDE_COLUMNS_KEY] = []
    override_selected = st.multiselect(
        "Columns with explicit role overrides",
        options=override_candidates,
        key=SESSION_ROLE_OVERRIDE_COLUMNS_KEY,
    )
    role_options = [_ROLE_NONE, *[role.value for role in ColumnRole]]
    column_role_overrides: dict[str, ColumnRole] = {}
    for feature_name in override_selected:
        selected_role = st.selectbox(
            f"Role override for {feature_name}",
            options=role_options,
            index=0,
            key=f"role_override_{feature_name}",
        )
        if selected_role != _ROLE_NONE:
            column_role_overrides[feature_name] = ColumnRole(selected_role)

    requested_task: AnalysisTask | None = None
    analysis_task_selection_valid = True
    selected_objective: RecommendationObjective | None = None
    quality_direction: QualityOptimizationDirection | None = None
    quality_target: float | None = None
    quality_direction_selected = True
    performance_rule_rows: list[dict[str, Any]] = []
    performance_rule_drafts: list[dict[str, Any]] = []
    complete_rule_count = 0
    constraint_rows: list[dict[str, float | str]] = []
    recommendation_constraint_drafts: list[dict[str, Any]] = []
    confirmed_controllable: list[str] = []
    verified_variables: list[str] = []
    anomaly_recommendation_enabled = False
    anomaly_recommendation_ready = True

    if anomaly_only:
        st.subheader("Analysis task")
        st.info("Analysis task: not applicable for anomaly-only analysis.")
        st.subheader("Model performance acceptance rules")
        st.info(_ANOMALY_ONLY_PERFORMANCE_NOTE)

        st.subheader("Anomaly-score reduction recommendation")
        st.caption(_ANOMALY_RECOMMENDATION_SECTION_CAPTION)
        anomaly_recommendation_enabled = bool(
            st.checkbox(
                "Enable anomaly-score reduction recommendation",
                value=False,
                key=SESSION_ANOMALY_RECOMMENDATION_ENABLED_KEY,
            )
        )
        if anomaly_recommendation_enabled:
            selected_objective = RecommendationObjective.REDUCE_ANOMALY_SCORE
            st.write(
                f"Objective (fixed): {RecommendationObjective.REDUCE_ANOMALY_SCORE.value}"
            )
            st.caption(
                "Diagnosis factors are associations, not proven causes. "
                "Candidate variables must be selected from active anomaly features."
            )
            recommendation_feature_options = list(modeling_feature_columns)
            cohort_filter_column = (
                None if cohort_filter is None else cohort_filter.column_name
            )
            if cohort_filter_column is not None:
                recommendation_feature_options = [
                    name
                    for name in recommendation_feature_options
                    if name != cohort_filter_column
                ]
                st.caption(
                    "Cohort filter columns are excluded from recommendation "
                    "candidate options."
                )

            review_variables = st.multiselect(
                "Candidate variables with explicit role overrides",
                options=recommendation_feature_options,
                default=[],
                key="anomaly_recommendation_review_variables",
            )
            role_options = [_ROLE_NONE, *[role.value for role in ColumnRole]]
            for feature_name in review_variables:
                selected_role = st.selectbox(
                    f"Recommendation role override for {feature_name}",
                    options=role_options,
                    index=0,
                    key=f"anomaly_rec_role_{feature_name}",
                )
                if selected_role != _ROLE_NONE:
                    column_role_overrides[feature_name] = ColumnRole(selected_role)

            confirmed_controllable = st.multiselect(
                "Confirmed controllable variables",
                options=review_variables,
                default=[],
                key="anomaly_rec_confirmed_controllable",
            )
            verified_variables = st.multiselect(
                "Verified variables",
                options=review_variables,
                default=[],
                key="anomaly_rec_verified_variables",
            )
            constrained_variables = st.multiselect(
                "Variables with recommendation constraints",
                options=review_variables,
                default=[],
                key=SESSION_ANOMALY_CONSTRAINT_VARIABLES_KEY,
            )
            constraint_incomplete = False
            for variable in constrained_variables:
                observed_caption = None
                if analysis_frame is not None and variable in analysis_frame.columns:
                    series = analysis_frame.get_column(variable)
                    if series.dtype.is_numeric():
                        low = series.min()
                        high = series.max()
                        if low is not None and high is not None:
                            observed_caption = _OBSERVED_RANGE_CAPTION_TEMPLATE.format(
                                low=low,
                                high=high,
                            )
                if observed_caption is not None:
                    st.caption(observed_caption)
                minimum_text = st.text_input(
                    f"Lower bound for {variable}",
                    value="",
                    key=f"anomaly_rec_constraint_min_{variable}",
                )
                maximum_text = st.text_input(
                    f"Upper bound for {variable}",
                    value="",
                    key=f"anomaly_rec_constraint_max_{variable}",
                )
                min_stripped = str(minimum_text).strip()
                max_stripped = str(maximum_text).strip()
                draft_minimum: float | str | None = (
                    None if min_stripped == "" else min_stripped
                )
                draft_maximum: float | str | None = (
                    None if max_stripped == "" else max_stripped
                )
                # Record every selected constraint for preset export validation.
                recommendation_constraint_drafts.append(
                    {
                        "variable": variable,
                        "minimum": draft_minimum,
                        "maximum": draft_maximum,
                    }
                )
                if min_stripped == "" or max_stripped == "":
                    constraint_incomplete = True
                    st.warning(_CONSTRAINT_INCOMPLETE_OPERATOR_MESSAGE)
                    continue
                try:
                    minimum_value = float(min_stripped)
                    maximum_value = float(max_stripped)
                except ValueError:
                    constraint_incomplete = True
                    st.warning(_CONSTRAINT_INCOMPLETE_OPERATOR_MESSAGE)
                    continue
                if not math.isfinite(minimum_value) or not math.isfinite(maximum_value):
                    constraint_incomplete = True
                    st.warning(_CONSTRAINT_INCOMPLETE_OPERATOR_MESSAGE)
                    continue
                if minimum_value >= maximum_value:
                    constraint_incomplete = True
                    st.warning(_CONSTRAINT_INCOMPLETE_OPERATOR_MESSAGE)
                    continue
                constraint_rows.append(
                    {
                        "variable": variable,
                        "minimum": minimum_value,
                        "maximum": maximum_value,
                    }
                )

            role_overrides_complete = all(
                name in column_role_overrides for name in review_variables
            )
            eligible_intersection = [
                name
                for name in review_variables
                if name in confirmed_controllable
                and name in verified_variables
                and any(row["variable"] == name for row in constraint_rows)
                and column_role_overrides.get(name) is ColumnRole.CONTROLLABLE_PROCESS
                and (cohort_filter_column is None or name != cohort_filter_column)
            ]
            anomaly_recommendation_ready = (
                len(review_variables) >= 1
                and role_overrides_complete
                and len(confirmed_controllable) >= 1
                and len(verified_variables) >= 1
                and len(constraint_rows) >= 1
                and not constraint_incomplete
                and len(eligible_intersection) >= 1
            )
            if not anomaly_recommendation_ready:
                st.warning(
                    "Complete recommendation review variables, CONTROLLABLE_PROCESS "
                    "role overrides, confirmed controllable, verified, and finite "
                    "constraint bounds before running."
                )
        else:
            st.info(_ANOMALY_ONLY_RECOMMENDATION_NOTE)
    else:
        st.subheader("Analysis task")
        task_options = [
            _TASK_AUTO,
            AnalysisTask.REGRESSION.value,
            AnalysisTask.CLASSIFICATION.value,
        ]
        if SESSION_REQUESTED_TASK_KEY not in st.session_state:
            st.session_state[SESSION_REQUESTED_TASK_KEY] = _TASK_AUTO
        task_selection = st.selectbox(
            "Analysis task",
            options=task_options,
            key=SESSION_REQUESTED_TASK_KEY,
            help=_TASK_HELP,
        )
        if task_selection == AnalysisTask.REGRESSION.value:
            requested_task = AnalysisTask.REGRESSION
        elif task_selection == AnalysisTask.CLASSIFICATION.value:
            requested_task = AnalysisTask.CLASSIFICATION
            st.info(_TASK_UNSUPPORTED_OPERATOR_MESSAGE)
        analysis_task_selection_valid = task_selection in task_options
        if (
            requested_task is AnalysisTask.REGRESSION
            and selected_target is not None
            and analysis_frame is not None
            and selected_target in analysis_frame.columns
        ):
            regression_target_assessment = evaluate_target_suitability(
                analysis_frame,
                selected_target,
                requested_task=AnalysisTask.REGRESSION,
            )
            if not regression_target_assessment.suitable:
                target_has_usable_variation = False
                if (
                    regression_target_assessment.is_constant
                    or regression_target_assessment.is_all_null
                ):
                    st.error(_TARGET_CONSTANT_OPERATOR_MESSAGE)
                else:
                    st.error(regression_target_assessment.message)

        st.subheader("Recommendation objective")
        st.caption(
            "Objective is not inferred. Select an objective explicitly before running."
        )
        objective_options = [
            _OBJECTIVE_PLACEHOLDER,
            *[item.value for item in RecommendationObjective],
        ]
        if SESSION_OBJECTIVE_KEY not in st.session_state:
            st.session_state[SESSION_OBJECTIVE_KEY] = _OBJECTIVE_PLACEHOLDER
        objective_selection = st.selectbox(
            "Objective",
            options=objective_options,
            key=SESSION_OBJECTIVE_KEY,
        )
        if objective_selection != _OBJECTIVE_PLACEHOLDER:
            selected_objective = RecommendationObjective(objective_selection)
        else:
            st.warning("Select a recommendation objective before running analysis.")

        needs_quality = selected_objective in {
            RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
            RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
        }
        if needs_quality:
            quality_options = [
                _QUALITY_DIRECTION_PLACEHOLDER,
                *[item.value for item in QualityOptimizationDirection],
            ]
            if SESSION_QUALITY_DIRECTION_KEY not in st.session_state:
                st.session_state[SESSION_QUALITY_DIRECTION_KEY] = (
                    _QUALITY_DIRECTION_PLACEHOLDER
                )
            quality_selection = st.selectbox(
                "Quality direction",
                options=quality_options,
                key=SESSION_QUALITY_DIRECTION_KEY,
            )
            if quality_selection == _QUALITY_DIRECTION_PLACEHOLDER:
                quality_direction_selected = False
                st.warning("Select a quality optimization direction before running.")
            else:
                quality_direction = QualityOptimizationDirection(quality_selection)
                if quality_direction is QualityOptimizationDirection.TARGET:
                    if SESSION_QUALITY_TARGET_KEY not in st.session_state:
                        st.session_state[SESSION_QUALITY_TARGET_KEY] = 0.0
                    quality_target = float(
                        st.number_input(
                            "Quality target",
                            format="%.6f",
                            key=SESSION_QUALITY_TARGET_KEY,
                        )
                    )

        st.subheader("Model performance acceptance rules")
        st.caption(
            "Provide at least one complete metric rule. Direction is never inferred "
            "from the metric name. Arbitrary defaults are not submitted."
        )
        st.markdown(
            "- R² generally uses HIGHER_IS_BETTER.\n"
            "- RMSE and MAE generally use LOWER_IS_BETTER.\n"
            "- The user must confirm the metric direction and threshold."
        )
        if SESSION_RULE_COUNT_KEY not in st.session_state:
            st.session_state[SESSION_RULE_COUNT_KEY] = 1
        rule_count = int(
            st.number_input(
                "Number of performance rules",
                min_value=1,
                max_value=20,
                step=1,
                key=SESSION_RULE_COUNT_KEY,
            )
        )
        rule_issues: list[str] = []
        direction_options = [
            _DIRECTION_PLACEHOLDER,
            *[item.value for item in MetricAcceptanceDirection],
        ]
        for index in range(rule_count):
            st.markdown(f"Rule {index + 1}")
            metric_name = st.text_input(
                f"Metric name #{index + 1}",
                value="",
                key=f"metric_name_{index}",
            )
            direction_selection = st.selectbox(
                f"Direction #{index + 1}",
                options=direction_options,
                index=0,
                key=f"metric_direction_{index}",
            )
            threshold_text = st.text_input(
                f"Threshold #{index + 1}",
                value="",
                key=f"metric_threshold_{index}",
                help="Enter a finite numeric threshold. Leave blank until confirmed.",
            )
            required = st.checkbox(
                f"Required #{index + 1}",
                value=True,
                key=f"metric_required_{index}",
            )
            missing = _performance_rule_missing_fields(
                metric_name=metric_name,
                direction_selection=direction_selection,
                threshold_text=threshold_text,
            )
            performance_rule_drafts.append(
                {
                    "metric_name": metric_name,
                    "direction": (
                        None
                        if direction_selection == _DIRECTION_PLACEHOLDER
                        else direction_selection
                    ),
                    "threshold": threshold_text,
                    "required": bool(required),
                }
            )
            if missing:
                rule_issues.append(_PERFORMANCE_RULE_INCOMPLETE_OPERATOR_MESSAGE)
                continue
            threshold_value = float(str(threshold_text).strip())
            performance_rule_rows.append(
                {
                    "metric_name": str(metric_name).strip(),
                    "direction": direction_selection,
                    "threshold": threshold_value,
                    "required": bool(required),
                }
            )
            complete_rule_count += 1

        if complete_rule_count < 1 and not rule_issues:
            st.warning(_PERFORMANCE_RULE_INCOMPLETE_OPERATOR_MESSAGE)
        for issue in dict.fromkeys(rule_issues):
            st.warning(issue)

        st.subheader("Process variable constraints")
        st.caption(
            "Constraints are not inferred from dataset min/max. Variables without "
            "constraints may be excluded from recommendation candidates."
        )
        if SESSION_CONSTRAINT_VARIABLES_KEY not in st.session_state:
            st.session_state[SESSION_CONSTRAINT_VARIABLES_KEY] = []
        constrained_variables = st.multiselect(
            "Variables with recommendation constraints",
            options=feature_columns,
            key=SESSION_CONSTRAINT_VARIABLES_KEY,
        )
        for variable in constrained_variables:
            minimum = st.number_input(
                f"Minimum for {variable}",
                value=0.0,
                format="%.6f",
                key=f"constraint_min_{variable}",
            )
            maximum = st.number_input(
                f"Maximum for {variable}",
                value=1.0,
                format="%.6f",
                key=f"constraint_max_{variable}",
            )
            recommendation_constraint_drafts.append(
                {
                    "variable": variable,
                    "minimum": float(minimum),
                    "maximum": float(maximum),
                }
            )
            if float(minimum) >= float(maximum):
                st.warning(_CONSTRAINT_INCOMPLETE_OPERATOR_MESSAGE)
            constraint_rows.append(
                {
                    "variable": variable,
                    "minimum": float(minimum),
                    "maximum": float(maximum),
                }
            )

        st.subheader("Controllability and verification")
        st.caption(
            "Role override, controllability confirmation, and verification are "
            "independent inputs and are not auto-aligned. Recommended features are "
            "not auto-selected as controllable or verified."
        )
        if SESSION_CONFIRMED_CONTROLLABLE_KEY not in st.session_state:
            st.session_state[SESSION_CONFIRMED_CONTROLLABLE_KEY] = []
        confirmed_controllable = st.multiselect(
            "Confirmed controllable variables",
            options=feature_columns,
            key=SESSION_CONFIRMED_CONTROLLABLE_KEY,
        )
        if SESSION_VERIFIED_VARIABLES_KEY not in st.session_state:
            st.session_state[SESSION_VERIFIED_VARIABLES_KEY] = []
        verified_variables = st.multiselect(
            "Verified variables",
            options=feature_columns,
            key=SESSION_VERIFIED_VARIABLES_KEY,
        )

    st.subheader("Operating point and change limits")
    max_feature_count = max(1, len(feature_columns) or 1)
    if SESSION_MAX_CHANGES_KEY not in st.session_state:
        st.session_state[SESSION_MAX_CHANGES_KEY] = min(3, max_feature_count)
    elif (
        isinstance(st.session_state[SESSION_MAX_CHANGES_KEY], int)
        and st.session_state[SESSION_MAX_CHANGES_KEY] > max_feature_count
    ):
        st.session_state[SESSION_MAX_CHANGES_KEY] = max_feature_count
    max_simultaneous_changes = int(
        st.number_input(
            "Maximum simultaneous changes",
            min_value=1,
            max_value=max_feature_count,
            step=1,
            key=SESSION_MAX_CHANGES_KEY,
        )
    )
    default_operating = (
        OperatingPointSelectionMode.TOP_UNSUPERVISED_ANOMALY
        if anomaly_only
        else OperatingPointSelectionMode.TOP_RESIDUAL_ANOMALY
    )
    if SESSION_OPERATING_MODE_KEY not in st.session_state:
        st.session_state[SESSION_OPERATING_MODE_KEY] = default_operating.value
    operating_mode = OperatingPointSelectionMode(
        st.selectbox(
            "Operating-point selection mode",
            options=[item.value for item in OperatingPointSelectionMode],
            key=SESSION_OPERATING_MODE_KEY,
        )
    )
    explicit_operating_row_id: int | str | None = None
    explicit_row_id_ok = True
    if operating_mode is OperatingPointSelectionMode.EXPLICIT_ROW_ID:
        if SESSION_EXPLICIT_ROW_ID_KEY not in st.session_state:
            st.session_state[SESSION_EXPLICIT_ROW_ID_KEY] = ""
        row_id_text = st.text_input(
            "Explicit operating row ID",
            key=SESSION_EXPLICIT_ROW_ID_KEY,
        )
        explicit_operating_row_id = row_id_text
        if str(row_id_text).strip() == "":
            explicit_row_id_ok = False
            st.warning("Provide an explicit operating row ID for this mode.")

    max_changes_valid = (
        len(feature_columns) >= 1
        and 1 <= max_simultaneous_changes <= len(feature_columns)
    )

    preset_export_result = build_exportable_configuration_preset(
        analysis_mode=analysis_mode,
        requested_task=requested_task,
        target_column=selected_target,
        use_recommended_numeric_feature_set=bool(use_recommended_features),
        explicit_feature_columns=(
            [] if use_recommended_features else list(feature_columns)
        ),
        timestamp_column=selected_timestamp,
        identifier_columns=list(identifier_columns),
        excluded_columns=list(excluded_columns),
        role_overrides=dict(column_role_overrides),
        recommendation_objective=selected_objective,
        quality_direction=quality_direction,
        quality_target=quality_target,
        performance_rule_drafts=performance_rule_drafts,
        cohort_filter=cohort_filter,
        operating_cohort_restricted=bool(restrict_operating_cohort),
        anomaly_recommendation_enabled=bool(anomaly_recommendation_enabled),
        recommendation_constraint_drafts=recommendation_constraint_drafts,
        confirmed_controllable_variables=list(confirmed_controllable),
        verified_variables=list(verified_variables),
        maximum_simultaneous_changes=int(max_simultaneous_changes),
        operating_point_selection_mode=operating_mode,
        explicit_operating_row_id=(
            None
            if explicit_operating_row_id is None
            or str(explicit_operating_row_id).strip() == ""
            else explicit_operating_row_id
        ),
    )
    _render_configuration_preset_section(
        columns=columns,
        export_result=preset_export_result,
    )

    if anomaly_only:
        readiness = {
            "CSV parsed": True,
            "feature count >= 1": len(modeling_feature_columns) >= 1,
            "analysis mode valid": analysis_mode is AnalysisExecutionMode.ANOMALY_ONLY,
            "identifier/timestamp conflicts absent": (
                selected_timestamp not in identifier_columns
                if selected_timestamp is not None
                else True
            ),
            "feature-only split row count available": (
                analysis_frame is not None and analysis_frame.height >= 5
            ),
            "operating cohort filter valid": cohort_filter_ready,
            "operating-point selection valid": explicit_row_id_ok,
            "max simultaneous changes valid": max_changes_valid,
        }
        if anomaly_recommendation_enabled:
            readiness["anomaly recommendation inputs complete"] = (
                anomaly_recommendation_ready
            )
    else:
        readiness = {
            "CSV parsed": True,
            "target selected": selected_target is not None,
            "selected target has usable variation": target_has_usable_variation,
            "feature count >= 1": len(feature_columns) >= 1,
            "analysis task selection valid": analysis_task_selection_valid,
            "objective selected": selected_objective is not None,
            "required quality direction selected when applicable": (
                quality_direction_selected
            ),
            "at least one complete performance rule": complete_rule_count >= 1,
            "max simultaneous changes valid": max_changes_valid,
            "explicit row ID supplied when required": explicit_row_id_ok,
        }

    demo_feature_guard = validate_demo_feature_guard(
        modeling_feature_columns if anomaly_only else feature_columns,
        data_source=data_source,
    )
    if demo_source_active:
        readiness["demo features exclude forbidden columns"] = demo_feature_guard.ok
        if not demo_feature_guard.ok and demo_feature_guard.message is not None:
            st.error(demo_feature_guard.message)

    recommendation_enabled_summary = (
        bool(anomaly_recommendation_enabled)
        if anomaly_only
        else selected_objective is not None
    )
    _render_run_configuration_summary(
        analysis_mode=analysis_mode,
        selected_target=selected_target,
        feature_count=(
            len(modeling_feature_columns) if anomaly_only else len(feature_columns)
        ),
        selected_timestamp=selected_timestamp,
        identifier_count=len(identifier_columns),
        cohort_filter=cohort_filter,
        restrict_operating_cohort=bool(restrict_operating_cohort),
        recommendation_enabled=recommendation_enabled_summary,
        constraint_variable_count=len(recommendation_constraint_drafts),
        performance_rule_count=None if anomaly_only else complete_rule_count,
    )

    st.subheader("Run readiness")
    for label, ready in readiness.items():
        st.write(f"{'True' if ready else 'False'}: {label}")
    if anomaly_only and not anomaly_recommendation_enabled:
        st.caption(_ANOMALY_ONLY_RECOMMENDATION_NOTE)
    elif anomaly_only:
        st.caption(
            "Incomplete recommendation fields disable Run analysis. Complete "
            "safety inputs are required when anomaly-score reduction "
            "recommendation is enabled."
        )
    else:
        st.caption(
            "Empty constraints, controllability, or verification inputs can still allow "
            "analysis to run, but recommendation may remain READY or REFUSED."
        )

    run_disabled = not all(readiness.values())
    submitted = st.button(
        "Run analysis",
        type="primary",
        disabled=run_disabled,
    )

    if submitted:
        if upload_bytes is None:
            st.error(
                "A loaded dataset is required before running the workflow."
                if demo_source_active
                else "CSV upload is required before running the workflow."
            )
        elif anomaly_only:
            _run_workflow_from_upload(
                upload_bytes=upload_bytes,
                analysis_mode=analysis_mode,
                target_column=None,
                feature_columns=list(feature_columns),
                timestamp_selection=(
                    _TIMESTAMP_NONE
                    if selected_timestamp is None
                    else selected_timestamp
                ),
                identifier_columns=list(identifier_columns),
                excluded_columns=list(excluded_columns),
                column_role_overrides=dict(column_role_overrides),
                requested_task=None,
                objective=(
                    RecommendationObjective.REDUCE_ANOMALY_SCORE
                    if anomaly_recommendation_enabled
                    else None
                ),
                quality_direction=None,
                quality_target=None,
                declared_target_minimum=None,
                declared_target_maximum=None,
                performance_rule_rows=[],
                constraint_rows=(
                    list(constraint_rows) if anomaly_recommendation_enabled else []
                ),
                confirmed_controllable=(
                    list(confirmed_controllable)
                    if anomaly_recommendation_enabled
                    else []
                ),
                verified_variables=(
                    list(verified_variables) if anomaly_recommendation_enabled else []
                ),
                max_simultaneous_changes=max_simultaneous_changes,
                operating_mode=operating_mode,
                explicit_operating_row_id=explicit_operating_row_id,
                cohort_filter=cohort_filter,
                anomaly_recommendation_enabled=anomaly_recommendation_enabled,
                workflow_factory=workflow_factory,
                request_builder=active_request_builder,
                report_builder=active_report_builder,
            )
        elif selected_target is None:
            st.error("Select a target column before running the workflow.")
        elif not target_has_usable_variation:
            st.error(_TARGET_CONSTANT_OPERATOR_MESSAGE)
        elif selected_objective is None:
            st.error("Select a recommendation objective before running the workflow.")
        elif complete_rule_count < 1:
            st.error(_PERFORMANCE_RULE_INCOMPLETE_OPERATOR_MESSAGE)
        else:
            _run_workflow_from_upload(
                upload_bytes=upload_bytes,
                analysis_mode=analysis_mode,
                target_column=selected_target,
                feature_columns=list(feature_columns),
                timestamp_selection=(
                    _TIMESTAMP_NONE
                    if selected_timestamp is None
                    else selected_timestamp
                ),
                identifier_columns=list(identifier_columns),
                excluded_columns=list(excluded_columns),
                column_role_overrides=dict(column_role_overrides),
                requested_task=requested_task,
                objective=selected_objective,
                quality_direction=quality_direction,
                quality_target=quality_target,
                declared_target_minimum=(
                    DEMO_QUALITY_SCORE_DECLARED_MINIMUM
                    if demo_source_active
                    else None
                ),
                declared_target_maximum=(
                    DEMO_QUALITY_SCORE_DECLARED_MAXIMUM
                    if demo_source_active
                    else None
                ),
                performance_rule_rows=performance_rule_rows,
                constraint_rows=constraint_rows,
                confirmed_controllable=list(confirmed_controllable),
                verified_variables=list(verified_variables),
                max_simultaneous_changes=max_simultaneous_changes,
                operating_mode=operating_mode,
                explicit_operating_row_id=explicit_operating_row_id,
                cohort_filter=None,
                anomaly_recommendation_enabled=False,
                workflow_factory=workflow_factory,
                request_builder=active_request_builder,
                report_builder=active_report_builder,
            )

    _render_cached_report_if_any()
    return

def render_presentation_report(report: WorkflowPresentationReport) -> None:
    """Render a ``WorkflowPresentationReport`` in section order.

    Args:
        report: Presentation DTO produced by ``AnalysisWorkflowReportBuilder``.

    Raises:
        TypeError: If ``report`` is not a ``WorkflowPresentationReport``.
    """
    if not isinstance(report, WorkflowPresentationReport):
        raise TypeError(
            "report must be WorkflowPresentationReport, "
            f"got {type(report).__name__}"
        )

    _render_run_comparison_section(report)

    overview = report.overview
    st.header("Overview")
    operator_headline = _operator_overview_headline(report)
    if overview.status is AnalysisWorkflowStatus.COMPLETED:
        st.success(operator_headline)
    elif (
        overview.status is AnalysisWorkflowStatus.PARTIAL
        or operator_headline == _RECOMMENDATION_REFUSED_AFTER_ANOMALY_MESSAGE
    ):
        st.warning(operator_headline)
    else:
        st.error(operator_headline)
    st.write(overview.summary)
    refused_stage = next(
        (
            stage
            for stage in report.stages
            if stage.structured_refusal and stage.executed
        ),
        None,
    )
    if refused_stage is not None:
        st.caption(
            f"Refused stage: `{refused_stage.stage.value}`. "
            f"Cause: {_sanitize_operator_text(refused_stage.message)}"
        )
    st.caption(
        f"Terminal stage: {overview.terminal_stage.value} | "
        f"Duration: {overview.total_seconds:.3f}s"
    )

    data = report.data_summary
    st.header("Data summary")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Raw rows", data.raw_row_count)
    metric_cols[1].metric("Processed rows", data.processed_row_count)
    metric_cols[2].metric("Train", data.train_row_count)
    metric_cols[3].metric("Validation", data.validation_row_count)
    metric_cols2 = st.columns(4)
    metric_cols2[0].metric("Test", data.test_row_count)
    metric_cols2[1].metric("Anomaly events", data.anomaly_event_count)
    metric_cols2[2].metric("Diagnosis factors", data.diagnosis_factor_count)
    operating_id = (
        _NOT_AVAILABLE
        if data.selected_operating_row_id is None
        else str(data.selected_operating_row_id)
    )
    metric_cols2[3].metric("Operating row ID", operating_id)
    st.metric("Cohort rows", data.cohort_row_count)

    cohort = report.cohort_filter_summary
    st.subheader("Operating cohort filter")
    if cohort.configured:
        st.write(
            {
                "Cohort filter": cohort.column_name,
                "Range": cohort.range_display,
                "Rows retained": (
                    f"{cohort.retained_row_count:,} / {cohort.source_row_count:,}"
                ),
                "Filter column used as model feature": (
                    cohort.filter_column_used_as_feature_display
                ),
            }
        )
        st.info(_COHORT_FILTER_INTERPRETATION_NOTE)
    else:
        st.write(
            {
                "Cohort filter": "Not configured",
                "Rows retained": f"{cohort.retained_row_count:,}",
            }
        )

    routing = report.routing_summary
    model = report.model_summary
    analysis_mode_value = report.metadata.get("analysis_mode")
    anomaly_only_report = analysis_mode_value == AnalysisExecutionMode.ANOMALY_ONLY.value
    st.header("Routing summary")
    if anomaly_only_report:
        st.write(
            {
                "analysis_mode": AnalysisExecutionMode.ANOMALY_ONLY.value,
                "industry": _display_optional(routing.selected_industry),
                "target": "Not required",
                "analysis_task": _NOT_APPLICABLE,
                "feature_count": _display_optional(routing.feature_count),
            }
        )
    else:
        st.write(
            {
                "analysis_mode": _display_optional(analysis_mode_value),
                "industry": _display_optional(routing.selected_industry),
                "inferred_task": _display_optional(
                    None
                    if routing.inferred_task is None
                    else routing.inferred_task.value
                ),
                "selected_task": _display_optional(
                    None
                    if routing.selected_task is None
                    else routing.selected_task.value
                ),
                "selection_source": _display_optional(routing.task_selection_source),
                "override_applied": routing.task_override_applied,
                "target_column": _display_optional(routing.target_column),
                "feature_count": _display_optional(routing.feature_count),
                "target_suitable": _display_optional(routing.target_suitable),
                "target_unique_non_null_count": _display_optional(
                    routing.target_unique_non_null_count
                ),
                "target_refusal_code": _display_optional(routing.target_refusal_code),
                "target_suitability_message": _display_optional(
                    routing.target_suitability_message
                ),
            }
        )

    st.header("Model summary")
    if anomaly_only_report:
        st.write(
            {
                "supervised_model": _NOT_APPLICABLE,
                "anomaly_model": _display_optional(model.anomaly_model_key),
                "residual_calibration": _NOT_APPLICABLE,
                "independent_test_evaluation": (
                    model.independent_test_evaluation_performed
                ),
                "anomaly_score_direction": _display_optional(
                    model.anomaly_score_direction
                ),
                "recommendation": "Not enabled for anomaly-only mode",
            }
        )
    else:
        st.write(
            {
                "supervised_model_key": _display_optional(model.supervised_model_key),
                "anomaly_model_key": _display_optional(model.anomaly_model_key),
                "independent_test_evaluation": (
                    model.independent_test_evaluation_performed
                ),
                "residual_calibration": model.residual_calibration_performed,
                "anomaly_score_direction": _display_optional(
                    model.anomaly_score_direction
                ),
            }
        )

    st.header("Detected anomaly events")
    st.caption(_ANOMALY_EVENTS_CAPTION)
    if not report.anomaly_events:
        st.write("No anomaly events were selected.")
    else:
        selection_sources = sorted(
            {event.selection_source for event in report.anomaly_events}
        )
        st.caption(f"Selection source: {', '.join(selection_sources)}")
        st.dataframe(
            _anomaly_event_table_rows(report.anomaly_events),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            label="Download anomaly events CSV",
            data=_csv_bytes_from_rows(
                _anomaly_event_download_rows(report.anomaly_events)
            ),
            file_name="anomaly_events.csv",
            mime="text/csv",
            key="download_anomaly_events_csv",
        )

    st.header("Anomaly context explorer")
    st.caption(_ANOMALY_CONTEXT_CAPTION)
    st.caption(_ANOMALY_CONTEXT_ORDER_CAPTION)
    if report.cohort_filter_summary.configured:
        st.caption(_ANOMALY_CONTEXT_FILTERED_CAPTION)
    context_windows = report.anomaly_context_windows
    if not context_windows:
        st.write("No anomaly context windows are available.")
    else:
        operating_ids = {
            event.original_row_id
            for event in report.anomaly_events
            if event.is_operating_row
        }
        option_labels = [
            _context_event_option_label(
                window,
                is_operating=window.center_original_row_id in operating_ids,
            )
            for window in context_windows
        ]
        default_index = _default_context_window_index(
            context_windows,
            report.anomaly_events,
        )
        selected_label = st.selectbox(
            "Select anomaly event",
            options=option_labels,
            index=default_index,
            key="anomaly_context_event_selector",
        )
        selected_window = context_windows[option_labels.index(selected_label)]
        st.markdown(
            f"- Center original row ID: `{selected_window.center_original_row_id}`\n"
            f"- Anomaly score: `{selected_window.center_anomaly_score}`\n"
            f"- Analysis order basis: `{selected_window.order_basis.value}`\n"
            f"- Window radius: `{selected_window.radius}`\n"
            f"- Included features: `{', '.join(selected_window.feature_names)}`"
        )
        st.caption(f"Fixed context radius contract: {ANOMALY_CONTEXT_RADIUS}")
        st.dataframe(
            _context_table_rows(selected_window),
            use_container_width=True,
            hide_index=True,
        )
        if selected_window.feature_names:
            feature_name = st.selectbox(
                "Context feature",
                options=list(selected_window.feature_names),
                index=0,
                key=(
                    "anomaly_context_feature_selector_"
                    f"{selected_window.event_rank}"
                ),
            )
            if _feature_values_are_numeric(selected_window, feature_name):
                chart_rows = _context_chart_rows(selected_window, feature_name)
                chart = (
                    _build_anomaly_context_feature_chart(chart_rows, feature_name)
                    if chart_rows
                    else None
                )
                if chart is None:
                    st.caption(_ANOMALY_CONTEXT_CHART_MISSING_CAPTION)
                else:
                    st.altair_chart(chart, use_container_width=True)
                    st.caption(
                        "Center event is at relative offset 0 on the analysis-order "
                        "axis."
                    )
            else:
                st.caption(
                    "Selected context feature is non-numeric; chart is omitted."
                )
        st.download_button(
            label="Download selected context CSV",
            data=_csv_bytes_from_rows(_context_download_rows(selected_window)),
            file_name=f"anomaly_context_rank_{selected_window.event_rank}.csv",
            mime="text/csv",
            key=f"download_anomaly_context_rank_{selected_window.event_rank}",
        )

    st.header("Likely associated factors")
    st.caption(_DIAGNOSIS_FACTORS_CAPTION)
    if not report.diagnosis_factors:
        st.write("No diagnosis factors were produced.")
    else:
        st.dataframe(
            _diagnosis_factor_table_rows(report.diagnosis_factors),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            label="Download diagnosis factors CSV",
            data=_csv_bytes_from_rows(
                _diagnosis_factor_download_rows(report.diagnosis_factors)
            ),
            file_name="diagnosis_factors.csv",
            mime="text/csv",
            key="download_diagnosis_factors_csv",
        )

    st.header("Model performance")
    performance = report.model_performance
    if anomaly_only_report and performance is None:
        st.write(_ANOMALY_ONLY_PERFORMANCE_NOTE)
    elif performance is None:
        st.write(_NOT_AVAILABLE)
    else:
        if performance.status is ModelPerformanceAcceptanceStatus.ACCEPTABLE:
            st.success(f"Status: {performance.status.value}")
        elif performance.status is ModelPerformanceAcceptanceStatus.UNACCEPTABLE:
            st.error(f"Status: {performance.status.value}")
        else:
            st.warning(f"Status: {performance.status.value}")
        st.write(
            {
                "evaluation_available": performance.evaluation_available,
                "independent_test_evaluation": performance.independent_test_evaluation,
                "test_row_count": performance.test_row_count,
                "required_rule_count": performance.required_rule_count,
                "passed_required_rule_count": performance.passed_required_rule_count,
                "failed_required_rule_count": performance.failed_required_rule_count,
                "unavailable_required_rule_count": (
                    performance.unavailable_required_rule_count
                ),
            }
        )
        st.dataframe(
            [
                {
                    "metric": item.metric_name,
                    "observed": item.observed_value,
                    "threshold": item.threshold,
                    "direction": item.direction.value,
                    "required": item.required,
                    "available": item.available,
                    "passed": item.passed,
                    "message": item.message,
                }
                for item in performance.metrics
            ],
            use_container_width=True,
        )

    st.header("Stage execution")
    stage_rows = [
        {
            "sequence": stage.sequence,
            "stage": stage.stage.value,
            "status_label": stage.status_label,
            "row_count": stage.row_count,
            "message": stage.message,
        }
        for stage in report.stages
    ]
    st.dataframe(stage_rows, use_container_width=True)
    for stage in report.stages:
        if stage.metadata:
            with st.expander(f"Stage metadata: {stage.stage.value}"):
                limited_items = list(stage.metadata.items())[:20]
                st.write(dict(limited_items))

    st.header("Recommendation")
    recommendation = report.recommendation
    if anomaly_only_report and recommendation is None:
        st.write(_ANOMALY_ONLY_RECOMMENDATION_NOTE)
    elif recommendation is None:
        st.write(_NOT_AVAILABLE)
    elif anomaly_only_report:
        st.subheader("Anomaly-score reduction recommendation")
        st.write(f"Status: {recommendation.status.value}")
        st.write(f"Objective: {recommendation.objective.value}")
        operating_row_display = (
            _NOT_AVAILABLE
            if report.data_summary.selected_operating_row_id is None
            else str(report.data_summary.selected_operating_row_id)
        )
        baseline_score = recommendation.baseline_anomaly_score
        proposed_score = recommendation.proposed_anomaly_score
        score_change: float | None = None
        if baseline_score is not None and proposed_score is not None:
            score_change = proposed_score - baseline_score
        if recommendation.status is RecommendationStatus.GENERATED:
            st.success(f"Status: {recommendation.status.value}")
            st.write(
                {
                    "Operating row ID": operating_row_display,
                    "Baseline anomaly score": baseline_score,
                    "Proposed anomaly score": proposed_score,
                    "Score change": score_change,
                    "Number of proposed changes": len(recommendation.changes),
                    "Extrapolation status": recommendation.extrapolation_flag,
                    "Uncertainty status": (
                        "available"
                        if recommendation.uncertainty_available
                        else "unavailable"
                    ),
                    "Safety status": recommendation.safety_status.value,
                }
            )
            if baseline_score is not None and proposed_score is not None:
                chart_frame = pd.DataFrame(
                    {
                        "scenario": ["Baseline", "Proposed"],
                        "anomaly_score": [baseline_score, proposed_score],
                    }
                )
                st.bar_chart(chart_frame, x="scenario", y="anomaly_score")
            constraint_lookup = {
                change.variable: change for change in recommendation.changes
            }
            st.dataframe(
                [
                    {
                        "Variable": change.variable,
                        "Current value": change.current_value,
                        "Proposed value": change.proposed_value,
                        "Absolute change": change.delta,
                        "Relative change": change.relative_delta,
                        "Verification status": (
                            "required" if change.requires_verification else "recorded"
                        ),
                        "Confidence": change.confidence,
                        "Rationale": change.rationale,
                    }
                    for change in constraint_lookup.values()
                ],
                use_container_width=True,
            )
            st.caption(_ANOMALY_RECOMMENDATION_RESULT_CAPTION)
            if not recommendation.uncertainty_available:
                st.caption(_ANOMALY_RECOMMENDATION_UNCERTAINTY_NOTE)
            download_rows: list[dict[str, Any]] = []
            if recommendation.changes:
                for change in recommendation.changes:
                    download_rows.append(
                        {
                            "recommendation_status": recommendation.status.value,
                            "operating_row_id": (
                                report.data_summary.selected_operating_row_id
                            ),
                            "objective": recommendation.objective.value,
                            "baseline_anomaly_score": baseline_score,
                            "proposed_anomaly_score": proposed_score,
                            "score_change": score_change,
                            "variable": change.variable,
                            "current_value": change.current_value,
                            "proposed_value": change.proposed_value,
                            "absolute_change": change.delta,
                            "relative_change": change.relative_delta,
                            "extrapolation_flag": recommendation.extrapolation_flag,
                            "uncertainty_available": (
                                recommendation.uncertainty_available
                            ),
                            "safety_status": recommendation.safety_status.value,
                            "warning": (
                                recommendation.warnings[0]
                                if recommendation.warnings
                                else None
                            ),
                        }
                    )
            else:
                download_rows.append(
                    {
                        "recommendation_status": recommendation.status.value,
                        "operating_row_id": (
                            report.data_summary.selected_operating_row_id
                        ),
                        "objective": recommendation.objective.value,
                        "baseline_anomaly_score": baseline_score,
                        "proposed_anomaly_score": proposed_score,
                        "score_change": score_change,
                        "variable": None,
                        "current_value": None,
                        "proposed_value": None,
                        "absolute_change": None,
                        "relative_change": None,
                        "extrapolation_flag": recommendation.extrapolation_flag,
                        "uncertainty_available": recommendation.uncertainty_available,
                        "safety_status": recommendation.safety_status.value,
                        "warning": (
                            recommendation.warnings[0]
                            if recommendation.warnings
                            else None
                        ),
                    }
                )
            st.download_button(
                "Download anomaly recommendation CSV",
                data=_csv_bytes_from_rows(download_rows),
                file_name="anomaly_recommendation.csv",
                mime="text/csv",
                key="download_anomaly_recommendation_csv",
            )
        elif recommendation.status is RecommendationStatus.READY_FOR_OPTIMIZATION:
            st.warning(f"Status: {recommendation.status.value}")
            st.write(
                {
                    "Operating row ID": operating_row_display,
                    "Baseline anomaly score": baseline_score,
                    "Proposed anomaly score": proposed_score,
                    "Extrapolation status": recommendation.extrapolation_flag,
                    "Uncertainty status": (
                        "available"
                        if recommendation.uncertainty_available
                        else "unavailable"
                    ),
                    "Safety status": recommendation.safety_status.value,
                }
            )
            for warning in recommendation.warnings:
                st.warning(warning)
            for message in recommendation.safety_messages:
                st.info(message)
            if not recommendation.uncertainty_available:
                st.caption(_ANOMALY_RECOMMENDATION_UNCERTAINTY_NOTE)
        else:
            st.error(f"Status: {recommendation.status.value}")
            for message in recommendation.safety_messages:
                st.error(message)
            for warning in recommendation.warnings:
                st.warning(warning)
    elif recommendation.status is RecommendationStatus.GENERATED:
        plausibility = recommendation.target_prediction_plausibility
        proposed_outside_declared = (
            plausibility is not None
            and plausibility.proposed_status
            is TargetPredictionPlausibilityStatus.OUTSIDE_DECLARED_DOMAIN
        )
        if proposed_outside_declared:
            st.warning(
                f"Status: {recommendation.status.value} — predicted quality is "
                "outside the declared target domain and must not be treated as a "
                "normal verified improvement."
            )
        else:
            st.success(f"Status: {recommendation.status.value}")
        st.write(f"Objective: {recommendation.objective.value}")
        st.dataframe(
            [
                {
                    "variable": change.variable,
                    "current": change.current_value,
                    "proposed": change.proposed_value,
                    "delta": change.delta,
                    "relative_delta": change.relative_delta,
                    "confidence": change.confidence,
                    "verification_required": change.requires_verification,
                    "rationale": change.rationale,
                }
                for change in recommendation.changes
            ],
            use_container_width=True,
        )
        prediction_summary: dict[str, object] = {
            "raw_baseline_quality_prediction": recommendation.baseline_prediction,
            "raw_proposed_quality_prediction": recommendation.proposed_prediction,
            "baseline_anomaly_score": recommendation.baseline_anomaly_score,
            "proposed_anomaly_score": recommendation.proposed_anomaly_score,
            "overall_confidence": recommendation.confidence,
            "safety_status": recommendation.safety_status.value,
            "extrapolation_flag": recommendation.extrapolation_flag,
        }
        if plausibility is not None:
            domain = plausibility.domain
            prediction_summary.update(
                {
                    "observed_training_target_minimum": (
                        None if domain is None else domain.observed_minimum
                    ),
                    "observed_training_target_maximum": (
                        None if domain is None else domain.observed_maximum
                    ),
                    "declared_target_minimum": (
                        None if domain is None else domain.declared_minimum
                    ),
                    "declared_target_maximum": (
                        None if domain is None else domain.declared_maximum
                    ),
                    "baseline_prediction_plausibility": (
                        None
                        if plausibility.baseline_status is None
                        else plausibility.baseline_status.value
                    ),
                    "proposed_prediction_plausibility": (
                        None
                        if plausibility.proposed_status is None
                        else plausibility.proposed_status.value
                    ),
                }
            )
        st.write(prediction_summary)
        if proposed_outside_declared:
            domain = None if plausibility is None else plausibility.domain
            if (
                domain is not None
                and domain.declared_minimum is not None
                and domain.declared_maximum is not None
            ):
                domain_text = (
                    f"[{domain.declared_minimum:g}, {domain.declared_maximum:g}]"
                )
            else:
                domain_text = (
                    f"[{DEMO_QUALITY_SCORE_DECLARED_MINIMUM:g}, "
                    f"{DEMO_QUALITY_SCORE_DECLARED_MAXIMUM:g}]"
                )
            st.error(
                "The raw proposed quality prediction is outside the declared "
                f"target domain {domain_text}. Do not interpret the estimated "
                "improvement quantitatively as an attainable quality score."
            )
        if plausibility is not None:
            for message in plausibility.warning_messages:
                st.warning(message)
        for warning in recommendation.warnings:
            if (
                plausibility is None
                or warning not in plausibility.warning_messages
            ):
                st.warning(warning)
        st.caption(
            "Association does not establish causation. Proposed changes require "
            "domain, safety, and operational verification."
        )
    elif recommendation.status is RecommendationStatus.READY_FOR_OPTIMIZATION:
        st.warning(f"Status: {recommendation.status.value}")
        st.write(
            "No executable changes were generated. The workflow is ready for "
            "optimization under the current safety status."
        )
        st.write(
            {
                "baseline_quality_prediction": recommendation.baseline_prediction,
                "proposed_quality_prediction": recommendation.proposed_prediction,
                "baseline_anomaly_score": recommendation.baseline_anomaly_score,
                "proposed_anomaly_score": recommendation.proposed_anomaly_score,
                "overall_confidence": recommendation.confidence,
                "safety_status": recommendation.safety_status.value,
            }
        )
        for warning in recommendation.warnings:
            st.warning(warning)
    else:
        st.error(f"Status: {recommendation.status.value}")
        st.write(
            "Safety Gate refused recommendation generation. No proposed changes "
            "are available."
        )
        st.write(
            {
                "safety_status": recommendation.safety_status.value,
                "overall_confidence": recommendation.confidence,
            }
        )
        for message in recommendation.safety_messages:
            st.error(message)

    _render_what_if_verification_section(report)

    st.header("Warnings")
    if not report.warnings:
        st.write("No warnings.")
    else:
        for warning in report.warnings:
            st.warning(warning)

    st.header("Disclaimers")
    if not report.disclaimers:
        st.write("No disclaimers.")
    else:
        with st.expander("Disclaimers", expanded=True):
            for disclaimer in report.disclaimers:
                st.info(disclaimer)


def _run_workflow_from_upload(
    *,
    upload_bytes: bytes,
    analysis_mode: AnalysisExecutionMode,
    target_column: str | None,
    feature_columns: list[str],
    timestamp_selection: str,
    identifier_columns: list[str],
    excluded_columns: list[str],
    column_role_overrides: dict[str, ColumnRole],
    requested_task: AnalysisTask | None,
    objective: RecommendationObjective | None,
    quality_direction: QualityOptimizationDirection | None,
    quality_target: float | None,
    declared_target_minimum: float | None,
    declared_target_maximum: float | None,
    performance_rule_rows: Sequence[Mapping[str, Any]],
    constraint_rows: Sequence[Mapping[str, float | str]],
    confirmed_controllable: list[str],
    verified_variables: list[str],
    max_simultaneous_changes: int,
    operating_mode: OperatingPointSelectionMode,
    explicit_operating_row_id: int | str | None,
    cohort_filter: NumericCohortFilter | None,
    anomaly_recommendation_enabled: bool,
    workflow_factory: WorkflowFactory,
    request_builder: WorkflowUiRequestBuilder,
    report_builder: AnalysisWorkflowReportBuilder,
) -> None:
    timestamp_column = (
        None if timestamp_selection == _TIMESTAMP_NONE else timestamp_selection
    )
    row_id: int | str | None = None
    if operating_mode is OperatingPointSelectionMode.EXPLICIT_ROW_ID:
        if explicit_operating_row_id is None:
            row_id = None
        elif isinstance(explicit_operating_row_id, int):
            row_id = explicit_operating_row_id
        else:
            text = str(explicit_operating_row_id).strip()
            if text == "":
                row_id = None
            else:
                try:
                    row_id = int(text)
                except ValueError:
                    row_id = text

    try:
        performance_rules = [
            UiMetricRuleInput(
                metric_name=str(row["metric_name"]),
                direction=MetricAcceptanceDirection(str(row["direction"])),
                threshold=float(row["threshold"]),
                required=bool(row["required"]),
            )
            for row in performance_rule_rows
        ]
        constraints = [
            UiVariableConstraintInput(
                variable=str(row["variable"]),
                minimum=float(row["minimum"]),
                maximum=float(row["maximum"]),
            )
            for row in constraint_rows
        ]
        submission = WorkflowUiSubmission(
            analysis_mode=analysis_mode,
            target_column=target_column,
            feature_columns=feature_columns,
            timestamp_column=timestamp_column,
            identifier_columns=identifier_columns,
            excluded_columns=excluded_columns,
            column_role_overrides=column_role_overrides,
            requested_task=requested_task,
            objective=objective,
            quality_direction=quality_direction,
            quality_target=quality_target,
            declared_target_minimum=declared_target_minimum,
            declared_target_maximum=declared_target_maximum,
            performance_rules=performance_rules,
            constraints=constraints,
            user_confirmed_controllable_variables=confirmed_controllable,
            user_verified_variables=verified_variables,
            max_simultaneous_changes=max_simultaneous_changes,
            operating_point_selection=operating_mode,
            explicit_operating_row_id=row_id,
            cohort_filter=(
                None if cohort_filter is None else cohort_filter.model_copy(deep=True)
            ),
            anomaly_recommendation_enabled=bool(anomaly_recommendation_enabled),
            metadata={"ui_entry": "streamlit_form"},
        )
    except (ValidationError, ValueError, TypeError) as exc:
        _display_user_error(exc, area="UI submission / validation")
        return

    with tempfile.TemporaryDirectory(prefix="ipi_ui_") as temp_dir:
        csv_path = Path(temp_dir) / "upload.csv"
        try:
            csv_path.write_bytes(upload_bytes)
            request = request_builder.build(
                csv_path=csv_path,
                submission=submission,
            )
            workflow = workflow_factory()
            outcome = workflow.run(request)
            if not isinstance(outcome, AnalysisWorkflowOutcome):
                raise TypeError(
                    "workflow.run must return AnalysisWorkflowOutcome, "
                    f"got {type(outcome).__name__}"
                )
            presentation = report_builder.build(outcome.report)
            st.session_state[_SESSION_REPORT_KEY] = presentation.report.model_dump(
                mode="json"
            )
        except (
            ValidationError,
            DataValidationError,
            ProcessIntelligenceError,
            FileNotFoundError,
            ValueError,
            TypeError,
            pl.exceptions.PolarsError,
        ) as exc:
            _display_user_error(exc, area="Workflow execution")


def _read_csv_for_ui(
    upload_bytes: bytes,
    *,
    preview_row_count: int,
) -> tuple[list[str], pl.DataFrame, pl.DataFrame]:
    """Parse uploaded CSV bytes with full-schema inference.

    Uses ``infer_schema_length=None`` so mixed integer/float columns remain a
    single numeric dtype instead of becoming null-filled after a short sample.
    The displayed preview is limited to ``preview_row_count`` rows. The returned
    frames are ephemeral render locals and are never written to session state.
    """
    frame = pl.read_csv(BytesIO(upload_bytes), infer_schema_length=None)
    if frame.height < 1:
        raise ValueError("Uploaded CSV has no data rows.")
    if frame.width < 1:
        raise ValueError("Uploaded CSV has no columns.")
    preview = frame.head(preview_row_count)
    return list(frame.columns), frame, preview


def _build_target_options(
    columns: Sequence[str],
    target_candidates: Sequence[str],
) -> list[str]:
    options: list[str] = [_TARGET_PLACEHOLDER]
    seen = {_TARGET_PLACEHOLDER}
    for name in target_candidates:
        if name not in seen and name in columns:
            options.append(name)
            seen.add(name)
    for name in columns:
        if name not in seen:
            options.append(name)
            seen.add(name)
    return options


def _build_timestamp_options(
    columns: Sequence[str],
    timestamp_candidates: Sequence[str],
) -> list[str]:
    options: list[str] = [_TIMESTAMP_NONE]
    seen = {_TIMESTAMP_NONE}
    for name in timestamp_candidates:
        if name not in seen and name in columns:
            options.append(name)
            seen.add(name)
    for name in columns:
        if name not in seen:
            options.append(name)
            seen.add(name)
    return options


def _timestamp_help_text(report: UiColumnConfigurationReport) -> str:
    parts: list[str] = [
        "Timestamp is not auto-confirmed. Date strings are not parsed automatically."
    ]
    for name in report.timestamp_candidates:
        suggestion = next(
            (item for item in report.suggestions if item.column == name),
            None,
        )
        if suggestion is None:
            continue
        mono = suggestion.monotonic_non_decreasing
        mono_text = "unknown" if mono is None else str(mono)
        parts.append(
            f"{name}: dtype={suggestion.dtype}, numeric={suggestion.numeric}, "
            f"monotonic_non_decreasing={mono_text}."
        )
    return " ".join(parts)


def _render_configuration_summary(report: UiColumnConfigurationReport) -> None:
    st.markdown("Configuration summary")
    cols = st.columns(4)
    cols[0].metric("Total columns", report.column_count)
    cols[1].metric("Numeric columns", len(report.numeric_columns))
    cols[2].metric("Recommended features", len(report.recommended_feature_columns))
    cols[3].metric("Target candidates", len(report.target_candidates))
    cols2 = st.columns(3)
    cols2[0].metric("Identifier candidates", len(report.identifier_candidates))
    cols2[1].metric("Timestamp candidates", len(report.timestamp_candidates))
    cols2[2].metric("Review-required columns", len(report.review_required_columns))


def _render_configuration_preset_section(
    *,
    columns: Sequence[str],
    export_result: ConfigurationPresetBuildResult,
) -> None:
    """Render analysis configuration preset download / upload / apply controls."""
    st.subheader("Analysis configuration preset")
    st.caption(
        "Download the current analysis settings as JSON, or upload a previously "
        "saved configuration. Applying a preset restores UI inputs only and does "
        "not run analysis."
    )

    if export_result.is_exportable and export_result.preset is not None:
        summary = build_configuration_preset_summary(export_result.preset)
        st.caption(
            "Export summary: "
            f"mode={summary['analysis_mode']}, "
            f"target={summary['target']}, "
            f"constraints={summary['constraint_variable_count']}."
        )
        try:
            preset_json = configuration_preset_to_json(export_result.preset)
        except (TypeError, ValueError, ValidationError):
            st.warning(
                "Configuration preset export is unavailable. "
                "Review the current settings and try again."
            )
            preset_json = None
        if preset_json is not None:
            st.download_button(
                "Download configuration JSON",
                data=preset_json,
                file_name=CONFIGURATION_PRESET_FILENAME,
                mime="application/json",
                key="download_configuration_preset_json",
            )
    else:
        st.warning(
            "Complete or clear the highlighted configuration rows before "
            "exporting the preset."
        )
        for issue in export_result.issues:
            st.warning(issue)

    uploaded_preset = st.file_uploader(
        "Upload configuration JSON",
        type=["json"],
        accept_multiple_files=False,
        key="upload_configuration_preset_json",
        help="Validate first. Apply configuration only after successful validation.",
    )

    if uploaded_preset is None:
        st.session_state.pop(SESSION_PRESET_PARSED_KEY, None)
        st.session_state.pop(SESSION_PRESET_VALIDATION_OK_KEY, None)
        st.session_state.pop(SESSION_PRESET_VALIDATION_MESSAGE_KEY, None)
        st.session_state.pop(SESSION_PRESET_MISSING_COLUMNS_KEY, None)
    else:
        try:
            raw_text = uploaded_preset.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            st.error("Configuration JSON must be UTF-8 text.")
            st.session_state[SESSION_PRESET_VALIDATION_OK_KEY] = False
            st.session_state[SESSION_PRESET_VALIDATION_MESSAGE_KEY] = (
                "Configuration JSON must be UTF-8 text."
            )
            st.session_state.pop(SESSION_PRESET_PARSED_KEY, None)
            raw_text = None

        if raw_text is not None:
            parse_result = parse_configuration_preset_json(raw_text)
            if not parse_result.ok or parse_result.preset is None:
                message = (
                    parse_result.error_message or "Configuration validation failed."
                )
                st.error(message)
                st.session_state[SESSION_PRESET_VALIDATION_OK_KEY] = False
                st.session_state[SESSION_PRESET_VALIDATION_MESSAGE_KEY] = message
                st.session_state.pop(SESSION_PRESET_PARSED_KEY, None)
                st.session_state.pop(SESSION_PRESET_MISSING_COLUMNS_KEY, None)
            else:
                compatibility = check_configuration_preset_column_compatibility(
                    parse_result.preset,
                    available_columns=columns,
                )
                if not compatibility.compatible:
                    st.error(compatibility.message)
                    st.session_state[SESSION_PRESET_VALIDATION_OK_KEY] = False
                    st.session_state[SESSION_PRESET_VALIDATION_MESSAGE_KEY] = (
                        compatibility.message
                    )
                    st.session_state[SESSION_PRESET_MISSING_COLUMNS_KEY] = list(
                        compatibility.missing_columns
                    )
                    st.session_state.pop(SESSION_PRESET_PARSED_KEY, None)
                else:
                    st.success("Validate configuration: succeeded.")
                    st.session_state[SESSION_PRESET_VALIDATION_OK_KEY] = True
                    st.session_state[SESSION_PRESET_VALIDATION_MESSAGE_KEY] = (
                        compatibility.message
                    )
                    st.session_state[SESSION_PRESET_MISSING_COLUMNS_KEY] = []
                    st.session_state[SESSION_PRESET_PARSED_KEY] = (
                        parse_result.parsed_dict
                    )

    apply_enabled = bool(st.session_state.get(SESSION_PRESET_VALIDATION_OK_KEY))
    if st.button(
        "Apply configuration",
        type="secondary",
        disabled=not apply_enabled,
        key="apply_configuration_preset",
    ):
        pending = st.session_state.get(SESSION_PRESET_PARSED_KEY)
        if not isinstance(pending, dict):
            st.error("No validated configuration is ready to apply.")
        else:
            try:
                pending_preset = WorkflowUiConfigurationPreset.model_validate(pending)
            except ValidationError:
                st.error(
                    "Validated configuration could not be reloaded. "
                    "Re-upload the configuration JSON and validate again."
                )
            else:
                compatibility = check_configuration_preset_column_compatibility(
                    pending_preset,
                    available_columns=columns,
                )
                if not compatibility.compatible:
                    st.error(compatibility.message)
                    st.session_state[SESSION_PRESET_MISSING_COLUMNS_KEY] = list(
                        compatibility.missing_columns
                    )
                else:
                    # Stage updates for the next run before widgets instantiate.
                    st.session_state[SESSION_PRESET_PENDING_APPLY_KEY] = (
                        build_configuration_preset_session_updates(pending_preset)
                    )
                    st.rerun()

    apply_summary_raw = st.session_state.get(SESSION_PRESET_APPLY_SUMMARY_KEY)
    if isinstance(apply_summary_raw, dict):
        apply_summary = dict(apply_summary_raw)
        st.success("Configuration applied.")
        st.write(
            f"schema version: {apply_summary.get('schema_version')}; "
            f"analysis mode: {apply_summary.get('analysis_mode')}; "
            f"target: {apply_summary.get('target')}; "
            f"feature configuration: {apply_summary.get('feature_configuration')}; "
            f"cohort filter: "
            f"{'yes' if apply_summary.get('cohort_filter_configured') else 'no'}; "
            f"anomaly recommendation: "
            f"{'yes' if apply_summary.get('anomaly_recommendation_enabled') else 'no'}; "
            f"constraint variables: {apply_summary.get('constraint_variable_count')}"
        )


def _consume_pending_configuration_preset_apply() -> None:
    """Apply a staged configuration preset before widgets are created."""
    pending = st.session_state.pop(SESSION_PRESET_PENDING_APPLY_KEY, None)
    if not isinstance(pending, dict):
        return
    # SessionStateProxy is MutableMapping[str | int, Any]; helper keys are str.
    clear_configuration_preset_widget_prefixes(
        cast(MutableMapping[str, Any], st.session_state)
    )
    for key, value in pending.items():
        st.session_state[key] = value


def _reset_column_widget_state_if_schema_changed(columns: Sequence[str]) -> None:
    fingerprint = "\0".join(columns)
    previous = st.session_state.get(_SESSION_COLUMNS_KEY)
    if previous == fingerprint:
        return
    st.session_state[_SESSION_COLUMNS_KEY] = fingerprint
    for key in (
        _SESSION_TARGET_KEY,
        _SESSION_TIMESTAMP_KEY,
        _SESSION_IDENTIFIERS_KEY,
        _SESSION_EXCLUDED_KEY,
    ):
        st.session_state.pop(key, None)


def _prune_selection_conflicts(
    *,
    selected_target: str | None,
    timestamp_options: Sequence[str],
) -> None:
    """Remove selected target from identifier / timestamp / excluded widget state.

    Uses only the public ``st.session_state`` API. Does not call ``st.rerun``.
    """
    if selected_target is None:
        return

    if _SESSION_IDENTIFIERS_KEY in st.session_state:
        current = st.session_state[_SESSION_IDENTIFIERS_KEY]
        if isinstance(current, list):
            st.session_state[_SESSION_IDENTIFIERS_KEY] = [
                name for name in current if name != selected_target
            ]

    if _SESSION_EXCLUDED_KEY in st.session_state:
        current = st.session_state[_SESSION_EXCLUDED_KEY]
        if isinstance(current, list):
            st.session_state[_SESSION_EXCLUDED_KEY] = [
                name for name in current if name != selected_target
            ]

    if st.session_state.get(_SESSION_TIMESTAMP_KEY) == selected_target:
        if _TIMESTAMP_NONE in timestamp_options:
            st.session_state[_SESSION_TIMESTAMP_KEY] = _TIMESTAMP_NONE


def _performance_rule_missing_fields(
    *,
    metric_name: object,
    direction_selection: object,
    threshold_text: object,
) -> list[str]:
    missing: list[str] = []
    name = "" if metric_name is None else str(metric_name).strip()
    if name == "":
        missing.append("metric name")
    direction = "" if direction_selection is None else str(direction_selection)
    if direction == "" or direction == _DIRECTION_PLACEHOLDER:
        missing.append("direction")
    raw_threshold = "" if threshold_text is None else str(threshold_text).strip()
    if raw_threshold == "":
        missing.append("threshold")
    else:
        try:
            number = float(raw_threshold)
        except ValueError:
            missing.append("threshold (finite float)")
        else:
            if not math.isfinite(number):
                missing.append("threshold (finite float)")
    return missing


def _render_cached_report_if_any() -> None:
    cached = st.session_state.get(_SESSION_REPORT_KEY)
    if cached is None:
        return
    if not isinstance(cached, dict):
        st.session_state.pop(_SESSION_REPORT_KEY, None)
        return
    try:
        report = WorkflowPresentationReport.model_validate(cached)
    except ValidationError:
        st.session_state.pop(_SESSION_REPORT_KEY, None)
        return
    st.divider()
    st.subheader("Last presentation report")
    render_presentation_report(report)


def _display_optional(value: object) -> str:
    if value is None:
        return _NOT_AVAILABLE
    return str(value)


def _sanitize_operator_text(text: str) -> str:
    cleaned = _WINDOWS_ABS_PATH_PATTERN.sub("<path>", text)
    cleaned = _UNIX_ABS_PATH_PATTERN.sub("<path>", cleaned)
    return cleaned


def _operator_facing_error_message(exc: BaseException, *, area: str) -> str:
    """Build an operator-facing error without exception types or absolute paths."""
    raw = _sanitize_operator_text(str(exc).strip())
    for fragment, message in _KNOWN_OPERATOR_ERROR_FRAGMENTS:
        if fragment in raw:
            return message
    if isinstance(exc, ValidationError):
        return (
            f"One or more inputs are invalid in {area}. "
            "Review the highlighted fields and try again."
        )
    if isinstance(exc, (pl.exceptions.PolarsError, UnicodeDecodeError)):
        return (
            "The CSV could not be parsed. Confirm the file is a valid UTF-8 CSV "
            "and try again."
        )
    if isinstance(exc, FileNotFoundError):
        return "A required input file was not found. Re-upload the CSV and try again."
    if raw:
        return f"{raw} Review the {area} settings and try again."
    return f"Something went wrong in {area}. Review the inputs and try again."


def _operator_overview_headline(report: WorkflowPresentationReport) -> str:
    """Choose an operator headline from structured presentation status fields."""
    overview = report.overview
    analysis_mode = report.metadata.get("analysis_mode")
    recommendation_refused = (
        overview.recommendation_status is RecommendationStatus.REFUSED
        or (
            report.recommendation is not None
            and report.recommendation.status is RecommendationStatus.REFUSED
        )
    )
    diagnosis_succeeded = any(
        stage.stage is AnalysisWorkflowStage.DIAGNOSIS and stage.succeeded
        for stage in report.stages
    )
    if (
        analysis_mode == AnalysisExecutionMode.ANOMALY_ONLY.value
        and overview.status is AnalysisWorkflowStatus.REFUSED
        and overview.terminal_stage is AnalysisWorkflowStage.RECOMMENDATION
        and recommendation_refused
        and diagnosis_succeeded
    ):
        return _RECOMMENDATION_REFUSED_AFTER_ANOMALY_MESSAGE
    if overview.terminal_stage is AnalysisWorkflowStage.TASK_ROUTING and (
        overview.status is AnalysisWorkflowStatus.REFUSED
    ):
        return _TASK_UNSUPPORTED_OPERATOR_MESSAGE
    return overview.headline


def _display_user_error(exc: BaseException, *, area: str) -> None:
    st.error(_operator_facing_error_message(exc, area=area))
