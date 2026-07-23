"""Public API for the Streamlit MVP UI (Step 11B / 11B.2)."""

from process_intelligence.ui.column_configuration import (
    AutomaticColumnConfigurator,
    UiColumnConfigurationReport,
    UiColumnSuggestion,
    UiColumnSuggestionCategory,
    resolve_active_feature_columns,
)
from process_intelligence.ui.configuration_preset import (
    WorkflowUiConfigurationPreset,
    apply_configuration_preset_to_session_state,
    build_configuration_preset,
    check_configuration_preset_column_compatibility,
    configuration_preset_to_json,
    parse_configuration_preset_json,
)
from process_intelligence.ui.request_builder import WorkflowUiRequestBuilder
from process_intelligence.ui.schemas import (
    StreamlitUiConfig,
    UiMetricRuleInput,
    UiVariableConstraintInput,
    WorkflowUiSubmission,
)
from process_intelligence.ui.streamlit_app import (
    render_app,
    render_presentation_report,
)
from process_intelligence.ui.workflow_factory import create_default_analysis_workflow

__all__ = [
    "AutomaticColumnConfigurator",
    "StreamlitUiConfig",
    "UiColumnConfigurationReport",
    "UiColumnSuggestion",
    "UiColumnSuggestionCategory",
    "UiMetricRuleInput",
    "UiVariableConstraintInput",
    "WorkflowUiConfigurationPreset",
    "WorkflowUiRequestBuilder",
    "WorkflowUiSubmission",
    "apply_configuration_preset_to_session_state",
    "build_configuration_preset",
    "check_configuration_preset_column_compatibility",
    "configuration_preset_to_json",
    "create_default_analysis_workflow",
    "parse_configuration_preset_json",
    "render_app",
    "render_presentation_report",
    "resolve_active_feature_columns",
]
