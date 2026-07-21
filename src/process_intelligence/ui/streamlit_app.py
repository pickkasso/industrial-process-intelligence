"""Streamlit MVP application for industrial process analysis (Step 11B.2).

Collects user inputs, builds ``AnalysisWorkflowRequest``, runs
``IndustrialProcessAnalysisWorkflow``, converts the backend report through
``AnalysisWorkflowReportBuilder``, and renders ``WorkflowPresentationReport``.
Does not implement modeling, scoring, ranking, or recommendation algorithms.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Callable, Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st
from pydantic import ValidationError

from process_intelligence.core.enums import ColumnRole
from process_intelligence.core.exceptions import (
    DataValidationError,
    ProcessIntelligenceError,
)
from process_intelligence.evaluation.performance_acceptance import (
    MetricAcceptanceDirection,
    ModelPerformanceAcceptanceStatus,
)
from process_intelligence.recommendation import (
    QualityOptimizationDirection,
    RecommendationObjective,
    RecommendationStatus,
)
from process_intelligence.reporting import AnalysisWorkflowReportBuilder
from process_intelligence.reporting.schemas import WorkflowPresentationReport
from process_intelligence.ui.column_configuration import (
    AutomaticColumnConfigurator,
    UiColumnConfigurationReport,
    resolve_active_feature_columns,
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
    AnalysisWorkflowStatus,
    IndustrialProcessAnalysisWorkflow,
    OperatingPointSelectionMode,
)
from process_intelligence.workflow.schemas import AnalysisWorkflowOutcome

_ORIGINAL_ROW_ID = "_original_row_id"
_SESSION_REPORT_KEY = "last_presentation_report_json"
_SESSION_COLUMNS_KEY = "ui_column_schema_fingerprint"
_SESSION_TARGET_KEY = "ui_target_column"
_SESSION_TIMESTAMP_KEY = "ui_timestamp_column"
_SESSION_IDENTIFIERS_KEY = "ui_identifier_columns"
_SESSION_EXCLUDED_KEY = "ui_excluded_columns"
_NOT_AVAILABLE = "Not available"
_ROLE_NONE = "(no override)"
_TARGET_PLACEHOLDER = "(select target)"
_TIMESTAMP_NONE = "(none)"
_OBJECTIVE_PLACEHOLDER = "(select objective)"
_DIRECTION_PLACEHOLDER = "(select direction)"
_QUALITY_DIRECTION_PLACEHOLDER = "(select quality direction)"

WorkflowFactory = Callable[[], IndustrialProcessAnalysisWorkflow]


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
        "Upload a process CSV, configure columns and acceptance rules, then run "
        "the industrial analysis workflow."
    )
    st.info(
        "Model-based decision support only. These outputs do not guarantee "
        "real-process improvement and are not operational commands."
    )

    uploaded = st.file_uploader(
        "CSV upload",
        type=["csv"],
        accept_multiple_files=False,
        help="CSV files only. Uploaded content is not stored permanently.",
    )

    columns: list[str] = []
    analysis_frame: pl.DataFrame | None = None
    preview_frame: pl.DataFrame | None = None
    upload_bytes: bytes | None = None
    config_report: UiColumnConfigurationReport | None = None

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
                if _ORIGINAL_ROW_ID in analysis_frame.columns:
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
        show_preview = st.checkbox("Show CSV preview", value=False)
        if show_preview:
            st.caption(
                f"Showing up to {ui_config.preview_row_count} rows for column review."
            )
            st.dataframe(preview_frame.to_dicts(), use_container_width=True)

    if not columns or config_report is None:
        st.warning(
            "Upload a valid CSV to configure columns and run the analysis workflow."
        )
        _render_cached_report_if_any()
        return

    _reset_column_widget_state_if_schema_changed(columns)

    st.subheader("Column configuration")
    st.caption(
        "Suggested target candidates are based on column structure and names. "
        "The user must confirm the final prediction target."
    )

    _render_configuration_summary(config_report)

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
        help="Suggested; review before running.",
    )

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
        help=(
            "When enabled, the compact recommended feature set is used and feature "
            "tags are not expanded on the main form."
        ),
    )

    base_recommended = list(config_report.recommended_feature_columns)
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
            feature_columns = st.multiselect(
                "Feature columns",
                options=columns,
                default=manual_default,
            )

    st.subheader("Column role overrides")
    st.caption(
        "Overrides are optional. Controllability is not confirmed by selecting "
        "a role. Choose columns explicitly; wide schemas do not render one widget "
        "per feature."
    )
    override_candidates = list(feature_columns)
    override_selected = st.multiselect(
        "Columns with explicit role overrides",
        options=override_candidates,
        default=[],
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

    st.subheader("Recommendation objective")
    st.caption(
        "Objective is not inferred. Select an objective explicitly before running."
    )
    objective_options = [
        _OBJECTIVE_PLACEHOLDER,
        *[item.value for item in RecommendationObjective],
    ]
    objective_selection = st.selectbox(
        "Objective",
        options=objective_options,
        index=0,
    )
    selected_objective: RecommendationObjective | None = None
    if objective_selection != _OBJECTIVE_PLACEHOLDER:
        selected_objective = RecommendationObjective(objective_selection)
    else:
        st.warning("Select a recommendation objective before running analysis.")

    quality_direction: QualityOptimizationDirection | None = None
    quality_target: float | None = None
    needs_quality = selected_objective in {
        RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
    }
    quality_direction_selected = True
    if needs_quality:
        quality_options = [
            _QUALITY_DIRECTION_PLACEHOLDER,
            *[item.value for item in QualityOptimizationDirection],
        ]
        quality_selection = st.selectbox(
            "Quality direction",
            options=quality_options,
            index=0,
        )
        if quality_selection == _QUALITY_DIRECTION_PLACEHOLDER:
            quality_direction_selected = False
            st.warning("Select a quality optimization direction before running.")
        else:
            quality_direction = QualityOptimizationDirection(quality_selection)
            if quality_direction is QualityOptimizationDirection.TARGET:
                quality_target = float(
                    st.number_input("Quality target", value=0.0, format="%.6f")
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
    rule_count = int(
        st.number_input(
            "Number of performance rules",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
        )
    )
    performance_rule_rows: list[dict[str, Any]] = []
    rule_issues: list[str] = []
    complete_rule_count = 0
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
        if missing:
            rule_issues.append(f"Rule {index + 1} missing: {', '.join(missing)}")
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

    if complete_rule_count < 1:
        st.warning(
            "Configure at least one complete performance rule "
            "(metric name, direction, and finite threshold)."
        )
    for issue in rule_issues:
        st.warning(issue)

    st.subheader("Process variable constraints")
    st.caption(
        "Constraints are not inferred from dataset min/max. Variables without "
        "constraints may be excluded from recommendation candidates."
    )
    constrained_variables = st.multiselect(
        "Variables with recommendation constraints",
        options=feature_columns,
        default=[],
    )
    constraint_rows: list[dict[str, float | str]] = []
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
    confirmed_controllable = st.multiselect(
        "Confirmed controllable variables",
        options=feature_columns,
        default=[],
    )
    verified_variables = st.multiselect(
        "Verified variables",
        options=feature_columns,
        default=[],
    )

    st.subheader("Operating point and change limits")
    max_feature_count = max(1, len(feature_columns) or 1)
    max_simultaneous_changes = int(
        st.number_input(
            "Maximum simultaneous changes",
            min_value=1,
            max_value=max_feature_count,
            value=min(3, max_feature_count),
            step=1,
        )
    )
    operating_mode = OperatingPointSelectionMode(
        st.selectbox(
            "Operating-point selection mode",
            options=[item.value for item in OperatingPointSelectionMode],
            index=list(OperatingPointSelectionMode).index(
                OperatingPointSelectionMode.TOP_RESIDUAL_ANOMALY
            ),
        )
    )
    explicit_operating_row_id: int | str | None = None
    explicit_row_id_ok = True
    if operating_mode is OperatingPointSelectionMode.EXPLICIT_ROW_ID:
        row_id_text = st.text_input("Explicit operating row ID", value="")
        explicit_operating_row_id = row_id_text
        if str(row_id_text).strip() == "":
            explicit_row_id_ok = False
            st.warning("Provide an explicit operating row ID for this mode.")

    max_changes_valid = (
        len(feature_columns) >= 1
        and 1 <= max_simultaneous_changes <= len(feature_columns)
    )
    readiness = {
        "CSV parsed": True,
        "target selected": selected_target is not None,
        "feature count >= 1": len(feature_columns) >= 1,
        "objective selected": selected_objective is not None,
        "required quality direction selected when applicable": (
            quality_direction_selected
        ),
        "at least one complete performance rule": complete_rule_count >= 1,
        "max simultaneous changes valid": max_changes_valid,
        "explicit row ID supplied when required": explicit_row_id_ok,
    }
    st.subheader("Run readiness")
    for label, ready in readiness.items():
        st.write(f"{'True' if ready else 'False'}: {label}")
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
            st.error("CSV upload is required before running the workflow.")
        elif selected_target is None:
            st.error("Select a target column before running the workflow.")
        elif selected_objective is None:
            st.error("Select a recommendation objective before running the workflow.")
        elif complete_rule_count < 1:
            st.error("Configure at least one complete performance rule.")
        else:
            _run_workflow_from_upload(
                upload_bytes=upload_bytes,
                target_column=selected_target,
                feature_columns=list(feature_columns),
                timestamp_selection=(
                    _TIMESTAMP_NONE
                    if selected_timestamp is None
                    else selected_timestamp
                ),
                identifier_columns=list(identifier_columns),
                excluded_columns=list(excluded_columns),
                column_role_overrides=column_role_overrides,
                objective=selected_objective,
                quality_direction=quality_direction,
                quality_target=quality_target,
                performance_rule_rows=performance_rule_rows,
                constraint_rows=constraint_rows,
                confirmed_controllable=list(confirmed_controllable),
                verified_variables=list(verified_variables),
                max_simultaneous_changes=max_simultaneous_changes,
                operating_mode=operating_mode,
                explicit_operating_row_id=explicit_operating_row_id,
                workflow_factory=workflow_factory,
                request_builder=active_request_builder,
                report_builder=active_report_builder,
            )

    _render_cached_report_if_any()


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

    overview = report.overview
    st.header("Overview")
    if overview.status is AnalysisWorkflowStatus.COMPLETED:
        st.success(overview.headline)
    elif overview.status is AnalysisWorkflowStatus.PARTIAL:
        st.warning(overview.headline)
    else:
        st.error(overview.headline)
    st.write(overview.summary)
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

    routing = report.routing_summary
    model = report.model_summary
    st.header("Routing summary")
    st.write(
        {
            "industry": _display_optional(routing.selected_industry),
            "task": _display_optional(
                None if routing.selected_task is None else routing.selected_task.value
            ),
            "target_column": _display_optional(routing.target_column),
            "feature_count": _display_optional(routing.feature_count),
        }
    )

    st.header("Model summary")
    st.write(
        {
            "supervised_model_key": _display_optional(model.supervised_model_key),
            "anomaly_model_key": _display_optional(model.anomaly_model_key),
            "independent_test_evaluation": model.independent_test_evaluation_performed,
            "residual_calibration": model.residual_calibration_performed,
            "anomaly_score_direction": _display_optional(model.anomaly_score_direction),
        }
    )

    st.header("Model performance")
    performance = report.model_performance
    if performance is None:
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
    if recommendation is None:
        st.write(_NOT_AVAILABLE)
    elif recommendation.status is RecommendationStatus.GENERATED:
        st.success(f"Status: {recommendation.status.value}")
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
    target_column: str,
    feature_columns: list[str],
    timestamp_selection: str,
    identifier_columns: list[str],
    excluded_columns: list[str],
    column_role_overrides: dict[str, ColumnRole],
    objective: RecommendationObjective,
    quality_direction: QualityOptimizationDirection | None,
    quality_target: float | None,
    performance_rule_rows: Sequence[Mapping[str, Any]],
    constraint_rows: Sequence[Mapping[str, float | str]],
    confirmed_controllable: list[str],
    verified_variables: list[str],
    max_simultaneous_changes: int,
    operating_mode: OperatingPointSelectionMode,
    explicit_operating_row_id: int | str | None,
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
            target_column=target_column,
            feature_columns=feature_columns,
            timestamp_column=timestamp_column,
            identifier_columns=identifier_columns,
            excluded_columns=excluded_columns,
            column_role_overrides=column_role_overrides,
            objective=objective,
            quality_direction=quality_direction,
            quality_target=quality_target,
            performance_rules=performance_rules,
            constraints=constraints,
            user_confirmed_controllable_variables=confirmed_controllable,
            user_verified_variables=verified_variables,
            max_simultaneous_changes=max_simultaneous_changes,
            operating_point_selection=operating_mode,
            explicit_operating_row_id=row_id,
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


def _display_user_error(exc: BaseException, *, area: str) -> None:
    st.error(
        f"{type(exc).__name__}: {exc}. Review the {area} inputs and try again."
    )
