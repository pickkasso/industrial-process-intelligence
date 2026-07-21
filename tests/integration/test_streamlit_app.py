"""Integration tests for the Streamlit MVP UI (Step 11B)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

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
    RecommendationResult,
    RecommendationSafetyDecision,
    RecommendationSafetyStatus,
    RecommendationStatus,
    VariableEligibilityAssessment,
)
from process_intelligence.recommendation.schemas import DEFAULT_RECOMMENDATION_DISCLAIMER
from process_intelligence.reporting import (
    AnalysisWorkflowReportBuilder,
    WorkflowPresentationReport,
)
from process_intelligence.ui import render_presentation_report
from process_intelligence.workflow import (
    AnalysisWorkflowOutcome,
    AnalysisWorkflowReport,
    AnalysisWorkflowStage,
    AnalysisWorkflowStageRecord,
    AnalysisWorkflowStatus,
    IndustrialProcessAnalysisWorkflow,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = PROJECT_ROOT / "app.py"

_UTC_START = datetime(2026, 7, 21, 10, 0, tzinfo=UTC)
_UTC_END = datetime(2026, 7, 21, 10, 5, tzinfo=UTC)
_ASSESSED_AT = datetime(2026, 7, 21, 10, 2, tzinfo=UTC)
_GENERATED_AT = datetime(2026, 7, 21, 13, 0, tzinfo=UTC)
_EVALUATED_AT = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)

_CSV_BYTES = (
    b"timestamp,lot_id,pressure,temperature,quality,notes\n"
    b"1,LOT-1,50.0,220.0,80.0,ok\n"
    b"2,LOT-1,51.0,221.0,81.0,ok\n"
    b"3,LOT-2,52.0,219.0,79.5,ok\n"
)


def _completed_report() -> AnalysisWorkflowReport:
    stages = list(AnalysisWorkflowStage)
    terminal = AnalysisWorkflowStage.RECOMMENDATION
    terminal_index = stages.index(terminal)
    records: list[AnalysisWorkflowStageRecord] = []
    for index, stage in enumerate(stages):
        executed = index <= terminal_index
        records.append(
            AnalysisWorkflowStageRecord(
                stage=stage,
                executed=executed,
                succeeded=executed,
                structured_refusal=False,
                row_count=3 if executed else None,
                message=(
                    f"{stage.value} stage completed."
                    if executed
                    else f"{stage.value} stage skipped."
                ),
                warnings=[],
                metadata={},
            )
        )

    assessment = VariableEligibilityAssessment(
        variable="pressure",
        factor_rank=1,
        factor_confidence=0.7,
        factor_role=str(ColumnRole.CONTROLLABLE_PROCESS),
        factor_controllable=True,
        factor_needs_verification=False,
        current_value=50.0,
        constraint_present=True,
        user_confirmed_controllable=True,
        user_verified=False,
        eligible=True,
        reason_codes=[],
        warnings=[],
    )
    decision = RecommendationSafetyDecision(
        status=RecommendationSafetyStatus.APPROVED,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        eligible_variables=["pressure"],
        blocked_variables=[],
        variable_assessments=[assessment],
        global_reason_codes=[],
        messages=["Safety checks passed."],
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        evaluated_at=_EVALUATED_AT,
    )
    recommendation = RecommendationResult(
        status=RecommendationStatus.GENERATED,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        safety_decision=decision,
        changes=[
            RecommendationChange(
                variable="pressure",
                current_value=50.0,
                proposed_value=48.0,
                delta=-2.0,
                relative_delta=-0.04,
                rationale=(
                    "Candidate change within observed support; association only."
                ),
                confidence=0.6,
                requires_verification=True,
            )
        ],
        baseline_prediction=80.0,
        proposed_prediction=82.0,
        baseline_anomaly_score=-0.35,
        proposed_anomaly_score=-0.55,
        confidence=0.5,
        extrapolation_flag=False,
        uncertainty_available=False,
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        generated_at=_GENERATED_AT,
        warnings=["Verify operationally before applying."],
    )
    performance = ModelPerformanceAcceptanceReport(
        status=ModelPerformanceAcceptanceStatus.ACCEPTABLE,
        task=AnalysisTask.REGRESSION,
        evaluation_available=True,
        independent_test_evaluation=True,
        test_row_count=20,
        metric_results=[
            MetricAcceptanceResult(
                metric_name="rmse",
                observed_value=1.2,
                threshold=10.0,
                direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
                required=True,
                available=True,
                passed=True,
                message="Metric rmse=1.2 meets LOWER_IS_BETTER threshold 10.0.",
            )
        ],
        required_rule_count=1,
        passed_required_rule_count=1,
        failed_required_rule_count=0,
        unavailable_required_rule_count=0,
        assessed_at=_ASSESSED_AT,
        warnings=[],
        metadata={},
    )
    return AnalysisWorkflowReport(
        status=AnalysisWorkflowStatus.COMPLETED,
        terminal_stage=terminal,
        stage_records=records,
        model_performance_assessment=performance,
        final_recommendation=recommendation,
        selected_industry="semiconductor",
        selected_task=AnalysisTask.REGRESSION,
        selected_supervised_model_key="ridge",
        selected_anomaly_model_key="isolation_forest",
        selected_operating_row_id=7,
        anomaly_event_count=2,
        diagnosis_factor_count=3,
        raw_row_count=100,
        processed_row_count=100,
        train_row_count=60,
        validation_row_count=20,
        test_row_count=20,
        started_at=_UTC_START,
        completed_at=_UTC_END,
        total_seconds=300.0,
        warnings=["Workflow warning W1."],
        metadata={
            "raw_csv_loaded": True,
            "workflow_orchestration_only": True,
            "test_used_for_model_selection": False,
            "test_used_for_threshold_calibration": False,
            "anomaly_score_direction": "higher_is_more_anomalous",
            "row_identity_preserved": True,
        },
    )


class _DeterministicWorkflow:
    """UI test double that returns a fixed backend workflow report."""

    def run(self, request: object) -> AnalysisWorkflowOutcome:
        assert request is not None
        return AnalysisWorkflowOutcome(report=_completed_report())


def _ui_entry(*, workflow_factory: object) -> None:
    """Self-contained AppTest entry; imports must live inside the function body."""
    from process_intelligence.ui import render_app

    render_app(workflow_factory=workflow_factory)  # type: ignore[arg-type]


def _render_entry(*, report_json: dict[str, object]) -> None:
    from process_intelligence.reporting import WorkflowPresentationReport
    from process_intelligence.ui import render_presentation_report as render_report

    render_report(WorkflowPresentationReport.model_validate(report_json))


def _make_app() -> AppTest:
    return AppTest.from_function(
        _ui_entry,
        default_timeout=30,
        kwargs={"workflow_factory": lambda: _DeterministicWorkflow()},
    )


def _upload_csv(at: AppTest) -> AppTest:
    at.file_uploader[0].set_value([("sample.csv", _CSV_BYTES, "text/csv")])
    return at.run()


def _text_blob(at: AppTest) -> str:
    parts: list[str] = []
    for attr in (
        "title",
        "header",
        "subheader",
        "markdown",
        "text",
        "caption",
        "info",
        "success",
        "warning",
        "error",
    ):
        elements = getattr(at, attr, [])
        for element in elements:
            value = getattr(element, "value", None)
            if value is not None:
                parts.append(str(value))
    return "\n".join(parts)


def test_app_py_loads() -> None:
    at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not at.exception
    assert len(at.title) >= 1
    assert "Industrial Process Intelligence" in at.title[0].value


def test_initial_header_and_uploader() -> None:
    at = _make_app().run()
    assert not at.exception
    assert "Industrial Process Intelligence" in at.title[0].value
    assert len(at.file_uploader) == 1
    blob = _text_blob(at)
    assert "Model-based decision support" in blob
    assert "guarantee" in blob.lower()


def test_run_without_upload_shows_guidance() -> None:
    at = _make_app().run()
    assert not at.exception
    assert len(at.button) == 0
    blob = _text_blob(at)
    assert "Upload a valid CSV" in blob


def _select_target(at: AppTest, target: str = "quality") -> AppTest:
    target_box = next(item for item in at.selectbox if item.label == "Target column")
    return target_box.select(target).run()


def _select_objective(at: AppTest, value: str = "REDUCE_ANOMALY_SCORE") -> AppTest:
    objective_box = next(item for item in at.selectbox if item.label == "Objective")
    return objective_box.select(value).run()


def _complete_performance_rule(
    at: AppTest,
    *,
    metric_name: str = "rmse",
    direction: str = "LOWER_IS_BETTER",
    threshold: str = "10.0",
) -> AppTest:
    metric_box = next(
        item for item in at.text_input if item.label.startswith("Metric name #")
    )
    at = metric_box.set_value(metric_name).run()
    direction_box = next(
        item for item in at.selectbox if item.label.startswith("Direction #")
    )
    at = direction_box.select(direction).run()
    threshold_box = next(
        item for item in at.text_input if item.label.startswith("Threshold #")
    )
    return threshold_box.set_value(threshold).run()


def _prepare_runnable_form(at: AppTest, *, target: str = "quality") -> AppTest:
    at = _select_target(at, target=target)
    at = _select_objective(at)
    return _complete_performance_rule(at)


def test_upload_creates_column_and_objective_controls() -> None:
    at = _upload_csv(_make_app().run())
    assert not at.exception
    labels = [item.label for item in at.selectbox]
    assert "Target column" in labels
    assert "Objective" in labels
    target_box = next(item for item in at.selectbox if item.label == "Target column")
    assert target_box.value == "(select target)"
    assert list(target_box.options)[0] == "(select target)"
    objective_box = next(item for item in at.selectbox if item.label == "Objective")
    assert objective_box.value == "(select objective)"
    assert list(objective_box.options)[0] == "(select objective)"
    assert any(
        item.label == "Use recommended numeric feature set" for item in at.checkbox
    )
    run_button = next(item for item in at.button if item.label == "Run analysis")
    assert run_button.disabled is True


def test_performance_constraint_controllability_inputs_present() -> None:
    at = _upload_csv(_make_app().run())
    text_labels = [item.label for item in at.text_input]
    assert any(label.startswith("Metric name") for label in text_labels)
    metric_box = next(
        item for item in at.text_input if item.label.startswith("Metric name #")
    )
    assert metric_box.value == ""
    direction_box = next(
        item for item in at.selectbox if item.label.startswith("Direction #")
    )
    assert direction_box.value == "(select direction)"
    threshold_box = next(
        item for item in at.text_input if item.label.startswith("Threshold #")
    )
    assert threshold_box.value == ""
    assert any(
        item.label == "Variables with recommendation constraints"
        for item in at.multiselect
    )
    assert any(
        item.label == "Confirmed controllable variables" for item in at.multiselect
    )
    assert any(item.label == "Verified variables" for item in at.multiselect)


def test_run_renders_presentation_sections() -> None:
    at = _prepare_runnable_form(_upload_csv(_make_app().run()))
    assert not at.exception
    run_button = next(item for item in at.button if item.label == "Run analysis")
    assert run_button.disabled is False
    at = run_button.click().run()
    assert not at.exception
    blob = _text_blob(at)
    assert "Analysis completed with a generated recommendation" in blob
    headers = [item.value for item in at.header]
    assert "Model performance" in headers
    assert "Stage execution" in headers
    assert "Recommendation" in headers
    assert "Warnings" in headers
    assert "Disclaimers" in headers


def test_negative_anomaly_score_displayed() -> None:
    at = _prepare_runnable_form(_upload_csv(_make_app().run()))
    at = next(item for item in at.button if item.label == "Run analysis").click().run()
    assert not at.exception
    # Scores are rendered via st.write(dict(...)); inspect JSON/text elements.
    serialized = str(at)
    assert "-0.35" in serialized or "-0.55" in serialized


def test_no_absolute_path_traceback_or_estimator_repr() -> None:
    at = _prepare_runnable_form(_upload_csv(_make_app().run()))
    at = next(item for item in at.button if item.label == "Run analysis").click().run()
    assert not at.exception
    blob = _text_blob(at) + str(at)
    assert "Traceback" not in blob
    assert "C:\\Users\\" not in blob
    assert "/Users/" not in blob
    assert "RandomForestRegressor(" not in blob
    assert "IsolationForest(" not in blob
    assert "sklearn." not in blob


def test_session_state_has_no_raw_bytes_dataframe_or_model() -> None:
    at = _prepare_runnable_form(_upload_csv(_make_app().run()))
    at = next(item for item in at.button if item.label == "Run analysis").click().run()
    assert not at.exception
    state = at.session_state.filtered_state
    assert "last_presentation_report_json" in state
    report_json = state["last_presentation_report_json"]
    assert isinstance(report_json, dict)
    for _key, value in state.items():
        assert not isinstance(value, (bytes, bytearray))
        type_name = type(value).__name__
        assert "DataFrame" not in type_name
        assert "Estimator" not in type_name
        assert not hasattr(value, "predict")


def test_presentation_rerender_is_deterministic() -> None:
    report = AnalysisWorkflowReportBuilder().build(_completed_report()).report
    first = report.model_dump(mode="json")
    second = WorkflowPresentationReport.model_validate(first).model_dump(mode="json")
    assert first == second

    at = AppTest.from_function(
        _render_entry,
        default_timeout=30,
        kwargs={"report_json": first},
    ).run()
    assert not at.exception
    blob = _text_blob(at)
    assert "Analysis completed with a generated recommendation" in blob
    again = WorkflowPresentationReport.model_validate(first).model_dump(mode="json")
    assert again == first


def test_default_factory_is_real_workflow() -> None:
    from process_intelligence.ui import create_default_analysis_workflow

    workflow = create_default_analysis_workflow()
    assert isinstance(workflow, IndustrialProcessAnalysisWorkflow)


def test_render_presentation_report_type_guard() -> None:
    with pytest.raises(TypeError):
        render_presentation_report({"overview": {}})  # type: ignore[arg-type]


def test_backend_disclaimer_visible_on_initial_page() -> None:
    at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    blob = _text_blob(at)
    assert "decision support" in blob.lower()
