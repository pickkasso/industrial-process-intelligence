"""Integration tests for the Streamlit MVP UI (Step 11B)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from process_intelligence.core.enums import AnalysisTask, AnomalyType, ColumnRole
from process_intelligence.core.schemas import AnomalyEvent, RootCauseFactor
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
    RecommendationTargetPlausibility,
    RecommendationWhatIfVerificationResult,
    TargetPredictionDomain,
    TargetPredictionPlausibilityStatus,
    VariableEligibilityAssessment,
    WhatIfPerturbationDirection,
    WhatIfStabilityClassification,
    WhatIfVerificationScenario,
    WhatIfVerificationScenarioType,
    WhatIfVerificationStatus,
)
from process_intelligence.recommendation.schemas import DEFAULT_RECOMMENDATION_DISCLAIMER
from process_intelligence.reporting import (
    AnalysisWorkflowReportBuilder,
    WorkflowPresentationReport,
)
from process_intelligence.ui import render_presentation_report
from process_intelligence.workflow import (
    AnalysisExecutionMode,
    AnalysisWorkflowOutcome,
    AnalysisWorkflowReport,
    AnalysisWorkflowStage,
    AnalysisWorkflowStageRecord,
    AnalysisWorkflowStatus,
    AnomalyContextOrderBasis,
    AnomalyContextRow,
    AnomalyContextValue,
    AnomalyContextWindow,
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


def _wif_scenario(**overrides: object) -> WhatIfVerificationScenario:
    payload: dict[str, object] = {
        "scenario_id": "WIF-000000",
        "scenario_type": WhatIfVerificationScenarioType.BASELINE,
        "perturbed_variable": None,
        "perturbation_direction": None,
        "variable_values": {"pressure": 50.0},
        "predicted_quality": None,
        "anomaly_score": -0.35,
        "objective_value": -0.35,
        "improves_over_baseline": False,
        "improves_or_matches_proposed": False,
        "extrapolated": False,
        "warnings": [],
    }
    payload.update(overrides)
    return WhatIfVerificationScenario(**payload)


def _wif_result(**overrides: object) -> RecommendationWhatIfVerificationResult:
    scenarios = overrides.pop(
        "scenarios",
        [
            _wif_scenario(),
            _wif_scenario(
                scenario_id="WIF-000001",
                scenario_type=WhatIfVerificationScenarioType.PROPOSED_CENTER,
                variable_values={"pressure": 48.0},
                anomaly_score=-0.55,
                objective_value=-0.55,
                improves_over_baseline=True,
                improves_or_matches_proposed=True,
            ),
            _wif_scenario(
                scenario_id="WIF-000002",
                scenario_type=WhatIfVerificationScenarioType.LOWER_NEIGHBOR,
                perturbed_variable="pressure",
                perturbation_direction=WhatIfPerturbationDirection.LOWER,
                variable_values={"pressure": 44.0},
                anomaly_score=-0.6,
                objective_value=-0.6,
                improves_over_baseline=True,
                improves_or_matches_proposed=True,
            ),
            _wif_scenario(
                scenario_id="WIF-000003",
                scenario_type=WhatIfVerificationScenarioType.UPPER_NEIGHBOR,
                perturbed_variable="pressure",
                perturbation_direction=WhatIfPerturbationDirection.UPPER,
                variable_values={"pressure": 52.0},
                anomaly_score=-0.1,
                objective_value=-0.1,
                improves_over_baseline=False,
                improves_or_matches_proposed=False,
            ),
        ],
    )
    payload: dict[str, object] = {
        "status": WhatIfVerificationStatus.COMPLETED,
        "objective": RecommendationObjective.REDUCE_ANOMALY_SCORE,
        "baseline_objective_value": -0.35,
        "proposed_objective_value": -0.55,
        "scenario_count": len(scenarios),
        "neighbor_scenario_count": 2,
        "improving_neighbor_count": 1,
        "non_improving_neighbor_count": 1,
        "extrapolated_scenario_count": 0,
        "stability_classification": WhatIfStabilityClassification.MIXED,
        "scenarios": scenarios,
        "warnings": [],
        "rationale": (
            "Local what-if verification scored adjacent constraint-grid "
            "neighbors with the same fitted model."
        ),
        "evaluated_at": _GENERATED_AT,
        "metadata": {
            "verification_executed": True,
            "model_refit_performed": False,
        },
    }
    payload.update(overrides)
    return RecommendationWhatIfVerificationResult(**payload)


def _completed_report(
    *,
    terminal: AnalysisWorkflowStage = AnalysisWorkflowStage.RECOMMENDATION,
    recommendation_verification: RecommendationWhatIfVerificationResult | None = None,
) -> AnalysisWorkflowReport:
    stages = list(AnalysisWorkflowStage)
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
        recommendation_verification=recommendation_verification,
        selected_industry="semiconductor",
        selected_task=AnalysisTask.REGRESSION,
        selected_supervised_model_key="ridge",
        selected_anomaly_model_key="isolation_forest",
        selected_operating_row_id=7,
        anomaly_event_count=2,
        diagnosis_factor_count=3,
        raw_row_count=100,
        processed_row_count=100,
        cohort_row_count=100,
        train_row_count=60,
        validation_row_count=20,
        test_row_count=20,
        cohort_filter_summary={
            "configured": False,
            "source_row_count": 100,
            "retained_row_count": 100,
            "excluded_row_count": 0,
            "null_excluded_count": 0,
        },
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
        dataset_fingerprint=(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        ),
    )


_ANOMALY_ONLY_SKIPPED_STAGES = frozenset(
    {
        AnalysisWorkflowStage.TASK_ROUTING,
        AnalysisWorkflowStage.SUPERVISED_SCREENING,
        AnalysisWorkflowStage.SUPERVISED_FINAL_EVALUATION,
        AnalysisWorkflowStage.RESIDUAL_CALIBRATION,
        AnalysisWorkflowStage.RESIDUAL_FINAL_EVALUATION,
        AnalysisWorkflowStage.RECOMMENDATION,
    }
)

_ANOMALY_CSV_BYTES = (
    b"timestamp,lot_id,pressure,temperature,SOH\n"
    b"1,LOT-1,50.0,220.0,0\n"
    b"2,LOT-1,51.0,221.0,0\n"
    b"3,LOT-2,52.0,219.0,0\n"
    b"4,LOT-2,53.0,222.0,0\n"
    b"5,LOT-3,54.0,223.0,0\n"
    b"6,LOT-3,55.0,224.0,0\n"
)


def _anomaly_only_report() -> AnalysisWorkflowReport:
    records: list[AnalysisWorkflowStageRecord] = []
    for stage in list(AnalysisWorkflowStage):
        if stage is AnalysisWorkflowStage.DIAGNOSIS:
            break
        if stage in _ANOMALY_ONLY_SKIPPED_STAGES:
            records.append(
                AnalysisWorkflowStageRecord(
                    stage=stage,
                    executed=False,
                    succeeded=False,
                    structured_refusal=False,
                    row_count=None,
                    message="Not applicable for anomaly-only analysis.",
                    warnings=[],
                    metadata={},
                )
            )
        else:
            records.append(
                AnalysisWorkflowStageRecord(
                    stage=stage,
                    executed=True,
                    succeeded=True,
                    structured_refusal=False,
                    row_count=3,
                    message=f"{stage.value} stage completed.",
                    warnings=[],
                    metadata={},
                )
            )
    records.append(
        AnalysisWorkflowStageRecord(
            stage=AnalysisWorkflowStage.DIAGNOSIS,
            executed=True,
            succeeded=True,
            structured_refusal=False,
            row_count=3,
            message="DIAGNOSIS stage completed.",
            warnings=[],
            metadata={
                "diagnosis_factor_count": 3,
                "diagnosis_source": "ROBUST_GROUP_COMPARISON",
            },
        )
    )
    records.append(
        AnalysisWorkflowStageRecord(
            stage=AnalysisWorkflowStage.RECOMMENDATION,
            executed=False,
            succeeded=False,
            structured_refusal=False,
            row_count=None,
            message="Not applicable for anomaly-only analysis.",
            warnings=[],
            metadata={"recommendation_applicable": False},
        )
    )
    return AnalysisWorkflowReport(
        status=AnalysisWorkflowStatus.PARTIAL,
        terminal_stage=AnalysisWorkflowStage.DIAGNOSIS,
        stage_records=records,
        analysis_mode=AnalysisExecutionMode.ANOMALY_ONLY,
        model_performance_assessment=None,
        final_recommendation=None,
        selected_industry="semiconductor",
        selected_task=None,
        selected_supervised_model_key=None,
        selected_anomaly_model_key="isolation_forest",
        selected_operating_row_id=2,
        anomaly_event_count=2,
        diagnosis_factor_count=3,
        anomaly_events=[
            AnomalyEvent(
                anomaly_id="2",
                anomaly_type=AnomalyType.PROCESS_INPUT,
                anomaly_score=-0.55,
                severity="high",
                sample_id=2,
                model_confidence=0.75,
                detector="unsupervised_anomaly_model",
                rationale="Unsupervised anomaly indicator was True.",
                contributing_variables=[],
            ),
            AnomalyEvent(
                anomaly_id="5",
                anomaly_type=AnomalyType.PROCESS_INPUT,
                anomaly_score=0.42,
                severity="high",
                sample_id=5,
                model_confidence=0.75,
                detector="top_anomaly_score_candidate",
                rationale="High unsupervised anomaly-score candidate.",
                contributing_variables=[],
            ),
        ],
        diagnosis_factors=[
            RootCauseFactor(
                variable="pressure",
                direction="POSITIVE",
                deviation=0.4,
                role=ColumnRole.UNKNOWN,
                controllable=False,
                evidence=(
                    "Higher values were associated with the analyzed anomaly "
                    "subset. reference_median=3.7; anomaly_median=4.1; "
                    "signed_location_difference=0.4; "
                    "robust_scale=1.4826; "
                    "robust_scale_status=AVAILABLE; "
                    "robust_z_score=2.5; raw_association_score=0.8."
                ),
                confidence=0.8,
                needs_verification=True,
            ),
            RootCauseFactor(
                variable="temperature",
                direction="NEGATIVE",
                deviation=-1.2,
                role=ColumnRole.UNKNOWN,
                controllable=False,
                evidence=(
                    "Lower values were associated with the analyzed anomaly "
                    "subset. reference_median=220; anomaly_median=218.8; "
                    "signed_location_difference=-1.2; "
                    "robust_scale=0.74; "
                    "robust_scale_status=AVAILABLE; "
                    "robust_z_score=1.8; raw_association_score=0.7."
                ),
                confidence=0.7,
                needs_verification=True,
            ),
            RootCauseFactor(
                variable="Current",
                direction="POSITIVE",
                deviation=106.0,
                role=ColumnRole.UNKNOWN,
                controllable=False,
                evidence=(
                    "Higher values were associated with the analyzed anomaly "
                    "subset. reference_median=0; anomaly_median=106; "
                    "signed_location_difference=106; "
                    "robust_scale=0; "
                    "robust_scale_status=ZERO_VARIANCE; "
                    "robust_z_score=None; raw_association_score=0.75. "
                    "Robust standardized score is unavailable because the "
                    "reference group has zero robust variance."
                ),
                confidence=0.75,
                needs_verification=True,
            ),
        ],
        anomaly_context_windows=[
            AnomalyContextWindow(
                event_rank=1,
                center_original_row_id=2,
                center_anomaly_score=-0.55,
                radius=3,
                order_basis=AnomalyContextOrderBasis.LOADED_ROW_ORDER,
                feature_names=["pressure", "temperature"],
                rows=[
                    AnomalyContextRow(
                        analysis_position=0,
                        original_row_id=0,
                        relative_offset=-2,
                        is_center_event=False,
                        is_selected_anomaly_event=False,
                        feature_values=[
                            AnomalyContextValue(feature_name="pressure", value=50.0),
                            AnomalyContextValue(
                                feature_name="temperature", value=220.0
                            ),
                        ],
                    ),
                    AnomalyContextRow(
                        analysis_position=1,
                        original_row_id=1,
                        relative_offset=-1,
                        is_center_event=False,
                        is_selected_anomaly_event=False,
                        feature_values=[
                            AnomalyContextValue(feature_name="pressure", value=51.0),
                            AnomalyContextValue(
                                feature_name="temperature", value=221.0
                            ),
                        ],
                    ),
                    AnomalyContextRow(
                        analysis_position=2,
                        original_row_id=2,
                        relative_offset=0,
                        is_center_event=True,
                        is_selected_anomaly_event=True,
                        feature_values=[
                            AnomalyContextValue(feature_name="pressure", value=52.0),
                            AnomalyContextValue(
                                feature_name="temperature", value=219.0
                            ),
                        ],
                    ),
                    AnomalyContextRow(
                        analysis_position=3,
                        original_row_id=3,
                        relative_offset=1,
                        is_center_event=False,
                        is_selected_anomaly_event=False,
                        feature_values=[
                            AnomalyContextValue(feature_name="pressure", value=53.0),
                            AnomalyContextValue(
                                feature_name="temperature", value=222.0
                            ),
                        ],
                    ),
                ],
            ),
            AnomalyContextWindow(
                event_rank=2,
                center_original_row_id=5,
                center_anomaly_score=0.42,
                radius=3,
                order_basis=AnomalyContextOrderBasis.LOADED_ROW_ORDER,
                feature_names=["pressure", "temperature"],
                rows=[
                    AnomalyContextRow(
                        analysis_position=5,
                        original_row_id=5,
                        relative_offset=0,
                        is_center_event=True,
                        is_selected_anomaly_event=True,
                        feature_values=[
                            AnomalyContextValue(feature_name="pressure", value=55.0),
                            AnomalyContextValue(
                                feature_name="temperature", value=224.0
                            ),
                        ],
                    ),
                ],
            ),
        ],
        raw_row_count=6,
        processed_row_count=6,
        cohort_row_count=6,
        train_row_count=4,
        validation_row_count=1,
        test_row_count=1,
        cohort_filter_summary={
            "configured": False,
            "source_row_count": 6,
            "retained_row_count": 6,
            "excluded_row_count": 0,
            "null_excluded_count": 0,
        },
        started_at=_UTC_START,
        completed_at=_UTC_END,
        total_seconds=1.0,
        warnings=[],
        metadata={
            "raw_csv_loaded": True,
            "workflow_orchestration_only": True,
            "test_used_for_model_selection": False,
            "test_used_for_threshold_calibration": False,
            "anomaly_score_direction": "higher_is_more_anomalous",
            "row_identity_preserved": True,
            "analysis_mode": "ANOMALY_ONLY",
            "recommendation_applicable": False,
            "independent_test_evaluation_performed": True,
            "residual_calibration_performed": False,
            "anomaly_event_selection_source": "UNSUPERVISED_ANOMALY_SCORE",
            "diagnosis_source": "ROBUST_GROUP_COMPARISON",
        },
        dataset_fingerprint=(
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        ),
    )


class _DeterministicWorkflow:
    """UI test double that returns a fixed backend workflow report."""

    def run(self, request: object) -> AnalysisWorkflowOutcome:
        assert request is not None
        return AnalysisWorkflowOutcome(report=_completed_report())


class _AnomalyOnlyWorkflow:
    """UI test double that returns a fixed ANOMALY_ONLY backend workflow report."""

    def run(self, request: object) -> AnalysisWorkflowOutcome:
        assert request is not None
        return AnalysisWorkflowOutcome(report=_anomaly_only_report())


_ANOMALY_WORKFLOW_RUN_COUNT = {"count": 0}


class _CountingAnomalyOnlyWorkflow:
    """ANOMALY_ONLY workflow double that counts ``run`` invocations."""

    def run(self, request: object) -> AnalysisWorkflowOutcome:
        assert request is not None
        _ANOMALY_WORKFLOW_RUN_COUNT["count"] += 1
        return AnalysisWorkflowOutcome(report=_anomaly_only_report())


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


def _make_anomaly_app() -> AppTest:
    return AppTest.from_function(
        _ui_entry,
        default_timeout=30,
        kwargs={"workflow_factory": lambda: _AnomalyOnlyWorkflow()},
    )


def _upload_anomaly_csv(at: AppTest) -> AppTest:
    at.file_uploader[0].set_value([("anomaly.csv", _ANOMALY_CSV_BYTES, "text/csv")])
    return at.run()


def _select_analysis_mode(at: AppTest, mode: str) -> AppTest:
    mode_box = next(item for item in at.selectbox if item.label == "Analysis mode")
    return mode_box.select(mode).run()


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
    assert "Quick start" in blob
    assert "Upload a CSV file" in blob
    assert "built-in manufacturing demo" in blob.lower()
    assert "Export reusable configuration when needed." in blob
    radio = next(item for item in at.radio if item.label == "Data source")
    assert radio.value == "Upload CSV"


def test_run_without_upload_shows_guidance() -> None:
    at = _make_app().run()
    assert not at.exception
    assert len(at.button) == 0
    blob = _text_blob(at)
    assert "Upload a valid CSV" in blob
    assert "built-in manufacturing demo" in blob.lower()
    assert "Quick start" in blob


def test_analysis_mode_guidance_and_configuration_summary() -> None:
    at = _upload_csv(_make_app().run())
    assert not at.exception
    blob = _text_blob(at)
    assert "Current configuration summary" in blob
    assert "Feature count:" in blob
    assert "Performance rule count:" in blob
    assert any(item.label == "SUPERVISED mode details" for item in at.expander)
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    blob_anomaly = _text_blob(at)
    assert "Not required" in blob_anomaly
    assert "Not applicable" in blob_anomaly
    assert any(item.label == "ANOMALY_ONLY mode details" for item in at.expander)


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
    assert "Analysis task" in labels
    assert "Objective" in labels
    target_box = next(item for item in at.selectbox if item.label == "Target column")
    assert target_box.value == "(select target)"
    assert list(target_box.options)[0] == "(select target)"
    task_box = next(item for item in at.selectbox if item.label == "Analysis task")
    assert task_box.value == "AUTO"
    assert list(task_box.options) == ["AUTO", "REGRESSION", "CLASSIFICATION"]
    assert "AUTO uses the task router" in str(task_box.help)
    objective_box = next(item for item in at.selectbox if item.label == "Objective")
    assert objective_box.value == "(select objective)"
    assert list(objective_box.options)[0] == "(select objective)"
    assert any(
        item.label == "Use recommended numeric feature set" for item in at.checkbox
    )
    run_button = next(item for item in at.button if item.label == "Run analysis")
    assert run_button.disabled is True


def test_analysis_task_stays_auto_after_soh_target_selection() -> None:
    at = _upload_csv(_make_app().run())
    at = _select_target(at, target="quality")
    task_box = next(item for item in at.selectbox if item.label == "Analysis task")
    assert task_box.value == "AUTO"
    blob = _text_blob(at)
    assert "True: analysis task selection valid" in blob


class _CapturingWorkflow:
    """Capture the workflow request for UI contract assertions."""

    last_request: object | None = None

    def run(self, request: object) -> AnalysisWorkflowOutcome:
        type(self).last_request = request
        return AnalysisWorkflowOutcome(report=_completed_report())


def _make_capturing_app() -> AppTest:
    _CapturingWorkflow.last_request = None
    return AppTest.from_function(
        _ui_entry,
        default_timeout=30,
        kwargs={"workflow_factory": lambda: _CapturingWorkflow()},
    )


class _CountingCapturingWorkflow:
    """Capture the workflow request and count how many times run() was called."""

    last_request: object | None = None
    run_count: int = 0

    def run(self, request: object) -> AnalysisWorkflowOutcome:
        type(self).last_request = request
        type(self).run_count += 1
        return AnalysisWorkflowOutcome(report=_anomaly_only_report())


def _make_anomaly_capturing_app() -> AppTest:
    _CountingCapturingWorkflow.last_request = None
    _CountingCapturingWorkflow.run_count = 0
    return AppTest.from_function(
        _ui_entry,
        default_timeout=30,
        kwargs={"workflow_factory": lambda: _CountingCapturingWorkflow()},
    )


def test_regression_task_selection_reaches_request() -> None:
    from process_intelligence.core.enums import AnalysisTask
    from process_intelligence.workflow import AnalysisWorkflowRequest

    at = _prepare_runnable_form(_upload_csv(_make_capturing_app().run()))
    task_box = next(item for item in at.selectbox if item.label == "Analysis task")
    at = task_box.select("REGRESSION").run()
    run_button = next(item for item in at.button if item.label == "Run analysis")
    at = run_button.click().run()
    assert not at.exception
    request = _CapturingWorkflow.last_request
    assert isinstance(request, AnalysisWorkflowRequest)
    assert request.requested_task is AnalysisTask.REGRESSION


def test_classification_task_selection_shows_unsupported_guidance() -> None:
    at = _upload_csv(_make_app().run())
    task_box = next(item for item in at.selectbox if item.label == "Analysis task")
    at = task_box.select("CLASSIFICATION").run()
    blob = _text_blob(at)
    assert "not supported by the current workflow" in blob.lower()
    assert "True: analysis task selection valid" in blob


def test_operator_error_helpers_hide_exception_types_and_paths() -> None:
    from process_intelligence.ui.streamlit_app import (
        _operator_facing_error_message,
        _sanitize_operator_text,
    )

    assert "C:\\Users\\" not in _sanitize_operator_text(
        r"Failed at C:\Users\tangt\data.csv"
    )
    assert "/Users/" not in _sanitize_operator_text("Failed at /Users/tangt/data.csv")
    message = _operator_facing_error_message(
        ValueError(
            "The configured cohort filter retained too few rows for the "
            "required train/validation/test split."
        ),
        area="Workflow execution",
    )
    assert "ValueError" not in message
    assert "enough rows for train, validation, and test partitions" in message
    assert "Traceback" not in message


def test_recommendation_refused_after_anomaly_uses_operator_headline() -> None:
    from process_intelligence.ui.streamlit_app import _operator_overview_headline

    source = _anomaly_only_report().model_copy(
        update={
            "status": AnalysisWorkflowStatus.REFUSED,
            "terminal_stage": AnalysisWorkflowStage.RECOMMENDATION,
            "final_recommendation": RecommendationResult(
                status=RecommendationStatus.REFUSED,
                objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
                safety_decision=RecommendationSafetyDecision(
                    status=RecommendationSafetyStatus.REFUSED,
                    objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
                    eligible_variables=[],
                    blocked_variables=[],
                    variable_assessments=[],
                    global_reason_codes=[],
                    messages=["Safety gate refused recommendation."],
                    disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
                    evaluated_at=_EVALUATED_AT,
                ),
                changes=[],
                baseline_prediction=None,
                proposed_prediction=None,
                baseline_anomaly_score=None,
                proposed_anomaly_score=None,
                confidence=0.0,
                extrapolation_flag=False,
                uncertainty_available=False,
                disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
                generated_at=_GENERATED_AT,
                warnings=[],
            ),
            "stage_records": [
                *[
                    record
                    for record in _anomaly_only_report().stage_records
                    if record.stage is not AnalysisWorkflowStage.RECOMMENDATION
                ],
                AnalysisWorkflowStageRecord(
                    stage=AnalysisWorkflowStage.RECOMMENDATION,
                    executed=True,
                    succeeded=False,
                    structured_refusal=True,
                    row_count=None,
                    message="Recommendation refused by safety gate.",
                    warnings=[],
                    metadata={},
                ),
            ],
        }
    )
    report = AnalysisWorkflowReportBuilder().build(source).report
    headline = _operator_overview_headline(report)
    assert "recommendation generation was refused by the safety gate" in headline
    assert "Analysis was stopped by a safety or validation gate" not in headline

    at = AppTest.from_function(
        _render_entry,
        default_timeout=30,
        kwargs={"report_json": report.model_dump(mode="json")},
    ).run()
    assert not at.exception
    blob = _text_blob(at)
    assert (
        "Anomaly analysis completed, but recommendation generation was refused"
        in blob
    )
    assert "Refused stage: `RECOMMENDATION`" in blob


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


def test_streamlit_displays_declared_domain_extrapolation_warning() -> None:
    source = _completed_report()
    assert source.final_recommendation is not None
    recommendation = source.final_recommendation.model_copy(
        update={
            "baseline_prediction": 96.5,
            "proposed_prediction": 115.55,
            "objective": RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
            "target_prediction_plausibility": RecommendationTargetPlausibility(
                domain=TargetPredictionDomain(
                    observed_minimum=45.0,
                    observed_maximum=98.0,
                    declared_minimum=40.0,
                    declared_maximum=100.0,
                ),
                baseline_raw_prediction=96.5,
                proposed_raw_prediction=115.55,
                baseline_status=TargetPredictionPlausibilityStatus.WITHIN_OBSERVED_RANGE,
                proposed_status=(
                    TargetPredictionPlausibilityStatus.OUTSIDE_DECLARED_DOMAIN
                ),
                warning_messages=[
                    "Raw predicted target is outside the declared semantic "
                    "target domain and must not be interpreted as a "
                    "quantitatively attainable quality value."
                ],
            ),
            "warnings": [
                "Raw predicted target is outside the declared semantic target "
                "domain and must not be interpreted as a quantitatively "
                "attainable quality value."
            ],
        },
        deep=True,
    )
    report = AnalysisWorkflowReportBuilder().build(
        source.model_copy(update={"final_recommendation": recommendation}, deep=True)
    ).report
    at = AppTest.from_function(
        _render_entry,
        default_timeout=30,
        kwargs={"report_json": report.model_dump(mode="json")},
    ).run()
    assert not at.exception
    blob = _text_blob(at) + str(at)
    assert "OUTSIDE_DECLARED_DOMAIN" in blob
    assert "outside the declared target domain" in blob.lower()
    assert "must not be treated as a normal verified improvement" in blob.lower()
    assert "115.55" in blob
    assert "Association does not establish causation" in blob


# ---------------------------------------------------------------------------
# What-if verification section (Step 11B.13)
# ---------------------------------------------------------------------------


def _verified_presentation_report_json() -> dict[str, object]:
    source = _completed_report(
        terminal=AnalysisWorkflowStage.WHAT_IF_VERIFICATION,
        recommendation_verification=_wif_result(),
    )
    report = AnalysisWorkflowReportBuilder().build(source).report
    return report.model_dump(mode="json")


def test_what_if_verification_section_shown_for_generated_recommendation() -> None:
    at = AppTest.from_function(
        _render_entry,
        default_timeout=30,
        kwargs={"report_json": _verified_presentation_report_json()},
    ).run()
    assert not at.exception
    headers = [item.value for item in at.subheader]
    assert "What-if verification" in headers
    blob = _text_blob(at)
    assert "lower values are less anomalous" in blob
    assert "not proof of physical safety or causation" in blob


def test_what_if_verification_csv_download_key_present() -> None:
    at = AppTest.from_function(
        _render_entry,
        default_timeout=30,
        kwargs={"report_json": _verified_presentation_report_json()},
    ).run()
    assert not at.exception
    download = next(
        item
        for item in at.download_button
        if item.label == "Download what-if verification CSV"
    )
    assert download.key == "download_what_if_verification_csv"


def test_what_if_verification_absent_when_no_verification_result() -> None:
    # A GENERATED recommendation with no recommendation_verification (e.g. a
    # workflow report generated before Step 11B.13 or with an unavailable
    # constraint grid) renders neither the full section nor a compact
    # not-applicable caption: the section is silently omitted.
    source = _completed_report()  # default: no recommendation_verification
    report = AnalysisWorkflowReportBuilder().build(source).report
    at = AppTest.from_function(
        _render_entry,
        default_timeout=30,
        kwargs={"report_json": report.model_dump(mode="json")},
    ).run()
    assert not at.exception
    headers = [item.value for item in at.subheader]
    assert "What-if verification" not in headers
    blob = _text_blob(at)
    assert "What-if verification was not applicable" not in blob


def test_what_if_verification_compact_caption_when_recommendation_not_generated() -> (
    None
):
    at = _upload_anomaly_csv(_make_anomaly_app().run())
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    at = next(item for item in at.button if item.label == "Run analysis").click().run()
    assert not at.exception
    blob = _text_blob(at)
    # Recommendation generation is disabled for this ANOMALY_ONLY fixture, so
    # final_recommendation is None and the compact not-applicable caption is
    # shown instead of the full what-if verification section.
    assert (
        "What-if verification was not applicable because no recommendation "
        "was generated." in blob
    )
    headers = [item.value for item in at.subheader]
    assert "What-if verification" not in headers


def test_what_if_verification_no_full_feature_dump() -> None:
    at = AppTest.from_function(
        _render_entry,
        default_timeout=30,
        kwargs={"report_json": _verified_presentation_report_json()},
    ).run()
    assert not at.exception
    blob = _text_blob(at) + str(at)
    # Only the compact controllable-variable mapping used by the fixture
    # ("pressure") is present in the what-if scenarios; there is no raw
    # feature-row dump of the full dataset's other columns.
    assert "temperature" not in blob
    assert "sensor" not in blob
    assert "DataFrame" not in blob


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


# ---------------------------------------------------------------------------
# ANOMALY_ONLY analysis mode (Step 11B.5)
# ---------------------------------------------------------------------------


def test_analysis_mode_control_defaults_to_supervised() -> None:
    at = _upload_csv(_make_app().run())
    assert not at.exception
    mode_box = next(item for item in at.selectbox if item.label == "Analysis mode")
    assert mode_box.value == "SUPERVISED"
    assert list(mode_box.options) == ["SUPERVISED", "ANOMALY_ONLY"]


def test_anomaly_only_selectable_and_hides_target() -> None:
    at = _upload_csv(_make_app().run())
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    assert not at.exception
    mode_box = next(item for item in at.selectbox if item.label == "Analysis mode")
    assert mode_box.value == "ANOMALY_ONLY"
    labels = [item.label for item in at.selectbox]
    assert "Target column" not in labels
    assert "Objective" not in labels
    blob = _text_blob(at)
    assert "not required for anomaly-only analysis" in blob.lower()


def test_anomaly_only_does_not_require_target_task_objective_performance() -> None:
    at = _upload_anomaly_csv(_make_anomaly_app().run())
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    assert not at.exception
    blob = _text_blob(at)
    assert "True: analysis mode valid" in blob
    assert "target selected" not in blob
    assert "objective selected" not in blob
    assert "at least one complete performance rule" not in blob
    run_button = next(item for item in at.button if item.label == "Run analysis")
    assert run_button.disabled is False


def test_anomaly_only_constant_soh_still_ready() -> None:
    # SOH is numeric and constant zero; ANOMALY_ONLY must not block readiness
    # on it since no target suitability evaluation is performed.
    at = _upload_anomaly_csv(_make_anomaly_app().run())
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    assert not at.exception
    run_button = next(item for item in at.button if item.label == "Run analysis")
    assert run_button.disabled is False
    blob = _text_blob(at)
    assert "True: feature count >= 1" in blob


def test_anomaly_only_feature_count_required() -> None:
    at = _upload_anomaly_csv(_make_anomaly_app().run())
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    at = next(
        item
        for item in at.checkbox
        if item.label == "Use recommended numeric feature set"
    ).set_value(False).run()
    select_all_box = next(
        item
        for item in at.checkbox
        if item.label == "Select all recommended features"
    )
    at = select_all_box.set_value(False).run()
    feature_box = next(
        item for item in at.multiselect if item.label == "Feature columns"
    )
    at = feature_box.set_value([]).run()
    assert not at.exception
    blob = _text_blob(at)
    assert "False: feature count >= 1" in blob
    run_button = next(item for item in at.button if item.label == "Run analysis")
    assert run_button.disabled is True


def test_anomaly_only_complete_submit_runs_workflow_with_target_none() -> None:
    from process_intelligence.workflow import AnalysisWorkflowRequest

    at = _upload_anomaly_csv(_make_anomaly_capturing_app().run())
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    run_button = next(item for item in at.button if item.label == "Run analysis")
    assert run_button.disabled is False
    at = run_button.click().run()
    assert not at.exception
    assert _CountingCapturingWorkflow.run_count == 1
    request = _CountingCapturingWorkflow.last_request
    assert isinstance(request, AnalysisWorkflowRequest)
    assert request.analysis_mode is AnalysisExecutionMode.ANOMALY_ONLY
    assert request.target_column is None
    assert request.objective is None
    assert request.model_performance_policy is None


def test_anomaly_only_stale_target_not_leaked_on_mode_switch() -> None:
    at = _upload_csv(_make_app().run())
    at = _select_target(at, target="quality")
    target_box = next(item for item in at.selectbox if item.label == "Target column")
    assert target_box.value == "quality"

    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    assert "Target column" not in [item.label for item in at.selectbox]

    at = _select_analysis_mode(at, "SUPERVISED")
    target_box_after = next(
        item for item in at.selectbox if item.label == "Target column"
    )
    assert target_box_after.value == "(select target)"


def test_anomaly_only_recommendation_not_applicable_message() -> None:
    at = _upload_anomaly_csv(_make_anomaly_app().run())
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    run_button = next(item for item in at.button if item.label == "Run analysis")
    assert run_button.disabled is False
    at = run_button.click().run()
    assert not at.exception
    blob = _text_blob(at)
    assert "Recommendation generation is not enabled for anomaly-only analysis." in blob
    assert "Not applicable for anomaly-only analysis." in blob


def test_supervised_regression_still_works_after_anomaly_only_additions() -> None:
    # Guards against ANOMALY_ONLY UI additions breaking the default SUPERVISED
    # submission path exercised throughout the rest of this module.
    at = _prepare_runnable_form(_upload_csv(_make_app().run()))
    assert not at.exception
    mode_box = next(item for item in at.selectbox if item.label == "Analysis mode")
    assert mode_box.value == "SUPERVISED"
    run_button = next(item for item in at.button if item.label == "Run analysis")
    assert run_button.disabled is False
    at = run_button.click().run()
    assert not at.exception
    blob = _text_blob(at)
    assert "Analysis completed with a generated recommendation" in blob
    assert "Proposed changes require" in blob
    assert "Performance acceptance only confirms" in blob


# ---------------------------------------------------------------------------
# Anomaly result presentation (Step 11B.6)
# ---------------------------------------------------------------------------


def test_detected_anomaly_events_section_and_table() -> None:
    at = _upload_anomaly_csv(_make_anomaly_app().run())
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    at = next(item for item in at.button if item.label == "Run analysis").click().run()
    assert not at.exception
    headers = [item.value for item in at.header]
    assert "Detected anomaly events" in headers
    blob = _text_blob(at) + str(at)
    assert "statistical outliers" in blob.lower()
    assert "not automatically defects" in blob.lower()
    report_json = at.session_state.filtered_state["last_presentation_report_json"]
    assert isinstance(report_json, dict)
    events = report_json["anomaly_events"]
    assert isinstance(events, list)
    assert len(events) == 2
    assert events[0]["original_row_id"] == 2
    assert events[0]["anomaly_score"] == pytest.approx(-0.55)
    assert events[0]["is_operating_row"] is True
    assert events[1]["is_operating_row"] is False


def test_likely_associated_factors_section_and_table() -> None:
    at = _upload_anomaly_csv(_make_anomaly_app().run())
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    at = next(item for item in at.button if item.label == "Run analysis").click().run()
    assert not at.exception
    headers = [item.value for item in at.header]
    assert "Likely associated factors" in headers
    blob = _text_blob(at)
    assert "associations, not proven causes" in blob.lower()
    assert "common bounded association score" in blob.lower()
    assert "supporting statistics" in blob.lower()
    assert "diagnostic score" not in blob.lower()
    assert "original row id is the workflow-preserved" in blob.lower()
    report_json = at.session_state.filtered_state["last_presentation_report_json"]
    assert isinstance(report_json, dict)
    factors = report_json["diagnosis_factors"]
    assert isinstance(factors, list)
    assert factors[0]["feature_name"] == "pressure"
    assert factors[0]["direction"] == "POSITIVE"
    assert factors[0]["robust_scale_status"] == "AVAILABLE"
    assert factors[0]["robust_z_score"] == pytest.approx(2.5)
    assert factors[0]["robust_scale"] == pytest.approx(1.4826)
    assert factors[0]["diagnostic_score"] == pytest.approx(0.8)
    assert factors[1]["feature_name"] == "temperature"
    assert factors[1]["direction"] == "NEGATIVE"
    assert factors[2]["feature_name"] == "Current"
    assert factors[2]["robust_scale_status"] == "ZERO_VARIANCE"
    assert factors[2]["robust_z_score"] is None
    assert factors[2]["robust_scale"] == pytest.approx(0.0)
    assert factors[2]["anomaly_group_value"] == pytest.approx(106.0)
    assert factors[2]["normal_group_value"] == pytest.approx(0.0)
    assert factors[2]["raw_group_difference"] == pytest.approx(106.0)
    assert factors[2]["direction"] == "POSITIVE"
    assert factors[2]["confidence"] == pytest.approx(0.75)
    assert factors[2]["diagnostic_score"] == pytest.approx(0.75)
    encoded = json.dumps(report_json)
    assert "100000000000000" not in encoded
    assert "1e+14" not in encoded.lower()


def test_diagnosis_factor_table_and_csv_include_scale_status() -> None:
    from process_intelligence.ui.streamlit_app import (
        _diagnosis_factor_download_rows,
        _diagnosis_factor_table_rows,
    )

    report = AnalysisWorkflowReportBuilder().build(_anomaly_only_report()).report
    table_rows = _diagnosis_factor_table_rows(report.diagnosis_factors)
    assert "Ranking score" in table_rows[0]
    assert "Diagnostic score" not in table_rows[0]
    assert "Robust scale" in table_rows[0]
    assert table_rows[0]["Ranking score"] == pytest.approx(0.8)
    assert table_rows[0]["Robust scale"] == pytest.approx(1.4826)
    assert table_rows[0]["Robust z-score"] == pytest.approx(2.5)
    assert table_rows[0]["Scale status"] == "AVAILABLE"
    assert table_rows[2]["Scale status"] == "ZERO_VARIANCE"
    assert table_rows[2]["Robust z-score"] == "Not available"
    assert table_rows[2]["Robust scale"] == pytest.approx(0.0)
    assert table_rows[2]["Ranking score"] == pytest.approx(0.75)
    assert table_rows[2]["Anomaly group"] == pytest.approx(106.0)
    assert table_rows[2]["Normal group"] == pytest.approx(0.0)
    assert table_rows[2]["Difference"] == pytest.approx(106.0)
    assert table_rows[2]["Direction"] == "POSITIVE"
    assert table_rows[2]["Confidence"] == pytest.approx(0.75)
    download_rows = _diagnosis_factor_download_rows(report.diagnosis_factors)
    assert "ranking_score" in download_rows[0]
    assert "diagnostic_score" not in download_rows[0]
    assert download_rows[0]["ranking_score"] == pytest.approx(0.8)
    assert download_rows[0]["robust_scale"] == pytest.approx(1.4826)
    assert download_rows[2]["robust_scale_status"] == "ZERO_VARIANCE"
    assert download_rows[2]["robust_z_score"] is None
    assert download_rows[2]["robust_scale"] == pytest.approx(0.0)
    assert download_rows[2]["ranking_score"] == pytest.approx(0.75)
    csv_text = json.dumps(download_rows)
    assert "ZERO_VARIANCE" in csv_text
    assert "ranking_score" in csv_text
    assert "robust_scale" in csv_text
    assert "100000000000000" not in csv_text



def test_empty_events_and_factors_handling() -> None:
    empty_source = _anomaly_only_report().model_copy(
        update={
            "anomaly_event_count": 0,
            "diagnosis_factor_count": 0,
            "anomaly_events": [],
            "diagnosis_factors": [],
            "selected_operating_row_id": None,
        },
        deep=True,
    )
    empty_report = AnalysisWorkflowReportBuilder().build(empty_source).report
    at = AppTest.from_function(
        _render_entry,
        default_timeout=30,
        kwargs={"report_json": empty_report.model_dump(mode="json")},
    ).run()
    assert not at.exception
    blob = _text_blob(at)
    assert "No anomaly events were selected." in blob
    assert "No diagnosis factors were produced." in blob


def test_anomaly_and_diagnosis_csv_downloads() -> None:
    at = _upload_anomaly_csv(_make_anomaly_app().run())
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    at = next(item for item in at.button if item.label == "Run analysis").click().run()
    assert not at.exception
    labels = [item.label for item in at.download_button]
    assert "Download anomaly events CSV" in labels
    assert "Download diagnosis factors CSV" in labels


def test_anomaly_only_mode_aware_disclaimers() -> None:
    at = _upload_anomaly_csv(_make_anomaly_app().run())
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    at = next(item for item in at.button if item.label == "Run analysis").click().run()
    assert not at.exception
    blob = _text_blob(at)
    assert "model-based decision support" in blob.lower()
    assert "do not establish causation" in blob.lower()
    assert "does not automatically mean a defect" in blob.lower()
    assert "Domain verification is required" in blob
    assert "Proposed changes require" not in blob
    assert "Performance acceptance only confirms" not in blob
    assert "Absence of extrapolation" not in blob


def test_no_full_feature_dump_in_anomaly_presentation() -> None:
    at = _upload_anomaly_csv(_make_anomaly_app().run())
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    at = next(item for item in at.button if item.label == "Run analysis").click().run()
    assert not at.exception
    state = at.session_state.filtered_state
    report_json = state["last_presentation_report_json"]
    assert isinstance(report_json, dict)
    encoded = json.dumps(report_json)
    assert "contributing_variables" not in encoded
    factors = report_json.get("diagnosis_factors", [])
    assert isinstance(factors, list)
    assert len(factors) == 3
    assert "Traceback" not in encoded
    assert "C:\\Users\\" not in encoded
    for _key, value in state.items():
        assert not isinstance(value, (bytes, bytearray))
        assert "DataFrame" not in type(value).__name__


# ---------------------------------------------------------------------------
# Anomaly context explorer (Step 11B.9)
# ---------------------------------------------------------------------------


def test_anomaly_context_explorer_section_and_selector() -> None:
    at = _upload_anomaly_csv(_make_anomaly_app().run())
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    at = next(item for item in at.button if item.label == "Run analysis").click().run()
    assert not at.exception
    headers = [item.value for item in at.header]
    assert "Anomaly context explorer" in headers
    blob = _text_blob(at)
    assert "interpretation only" in blob.lower()
    assert "workflow analysis order" in blob.lower()
    selector = next(
        item for item in at.selectbox if item.label == "Select anomaly event"
    )
    assert "Rank 1" in str(selector.value)
    assert "Original row ID 2" in str(selector.value)


def test_anomaly_context_defaults_to_operating_row() -> None:
    at = _upload_anomaly_csv(_make_anomaly_app().run())
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    at = next(item for item in at.button if item.label == "Run analysis").click().run()
    assert not at.exception
    selector = next(
        item for item in at.selectbox if item.label == "Select anomaly event"
    )
    assert "operating row" in str(selector.value).lower() or "Rank 1" in str(
        selector.value
    )


def test_anomaly_context_summary_table_and_chart() -> None:
    at = _upload_anomaly_csv(_make_anomaly_app().run())
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    at = next(item for item in at.button if item.label == "Run analysis").click().run()
    assert not at.exception
    blob = _text_blob(at) + str(at)
    assert "Center original row ID" in blob
    assert "Anomaly score" in blob
    assert "Analysis order basis" in blob
    assert "Window radius" in blob
    assert "Included features" in blob
    assert "relative offset 0" in blob.lower()
    feature_box = next(item for item in at.selectbox if item.label == "Context feature")
    assert feature_box.value == "pressure"
    report_json = at.session_state.filtered_state["last_presentation_report_json"]
    window = report_json["anomaly_context_windows"][0]
    assert any(row["relative_offset"] == 0 for row in window["rows"])
    assert any(row["analysis_position"] == 2 for row in window["rows"])
    assert "pressure" in window["feature_names"]


def test_anomaly_context_csv_download_and_no_rerun() -> None:
    at = _upload_anomaly_csv(_make_anomaly_app().run())
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    at = next(item for item in at.button if item.label == "Run analysis").click().run()
    assert not at.exception
    downloads = [item.label for item in at.download_button]
    assert "Download selected context CSV" in downloads
    report_before = at.session_state.filtered_state["last_presentation_report_json"]
    selector = next(
        item for item in at.selectbox if item.label == "Select anomaly event"
    )
    options = list(selector.options)
    if len(options) > 1:
        at = selector.select(options[1]).run()
        assert not at.exception
        report_after = at.session_state.filtered_state["last_presentation_report_json"]
        assert report_after == report_before


def test_anomaly_context_session_state_remains_presentation_only() -> None:
    at = _upload_anomaly_csv(_make_anomaly_app().run())
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    at = next(item for item in at.button if item.label == "Run analysis").click().run()
    assert not at.exception
    state = at.session_state.filtered_state
    report_json = state["last_presentation_report_json"]
    assert isinstance(report_json, dict)
    windows = report_json["anomaly_context_windows"]
    assert isinstance(windows, list)
    assert len(windows) == 2
    assert windows[0]["radius"] == 3
    assert len(windows[0]["feature_names"]) <= 5
    total_rows = sum(len(window["rows"]) for window in windows)
    assert total_rows <= 35
    encoded = json.dumps(report_json)
    assert "DataFrame" not in encoded
    assert "Traceback" not in encoded
    assert "C:\\Users\\" not in encoded
    for _key, value in state.items():
        assert not isinstance(value, (bytes, bytearray))
        assert "DataFrame" not in type(value).__name__


def test_anomaly_context_empty_windows_safe() -> None:
    empty_report = AnalysisWorkflowReportBuilder().build(
        _anomaly_only_report().model_copy(update={"anomaly_context_windows": []})
    ).report
    at = AppTest.from_function(
        _render_entry,
        default_timeout=30,
        kwargs={"report_json": empty_report.model_dump(mode="json")},
    ).run()
    assert not at.exception
    blob = _text_blob(at)
    assert "No anomaly context windows are available." in blob


def test_anomaly_context_does_not_break_event_or_diagnosis_sections() -> None:
    at = _upload_anomaly_csv(_make_anomaly_app().run())
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    at = next(item for item in at.button if item.label == "Run analysis").click().run()
    assert not at.exception
    headers = [item.value for item in at.header]
    assert "Detected anomaly events" in headers
    assert "Anomaly context explorer" in headers
    assert "Likely associated factors" in headers


def _context_window_with_feature_values(
    values_by_offset: list[tuple[int, object]],
    *,
    feature_name: str = "RSOCmin",
) -> AnomalyContextWindow:
    rows = [
        AnomalyContextRow(
            analysis_position=max(0, offset + 10),
            original_row_id=max(0, offset + 10),
            relative_offset=offset,
            is_center_event=offset == 0,
            is_selected_anomaly_event=offset == 0,
            feature_values=[
                AnomalyContextValue(feature_name=feature_name, value=value),
            ],
        )
        for offset, value in values_by_offset
    ]
    return AnomalyContextWindow(
        event_rank=1,
        center_original_row_id=10,
        center_anomaly_score=-0.4,
        radius=3,
        order_basis=AnomalyContextOrderBasis.LOADED_ROW_ORDER,
        feature_names=[feature_name],
        rows=rows,
    )


def _presentation_report_with_context(
    window: AnomalyContextWindow,
) -> WorkflowPresentationReport:
    return AnalysisWorkflowReportBuilder().build(
        _anomaly_only_report().model_copy(
            update={"anomaly_context_windows": [window]}
        )
    ).report


def test_context_chart_y_domain_constant_positive_includes_value() -> None:
    from process_intelligence.ui.streamlit_app import (
        _build_anomaly_context_feature_chart,
        _context_chart_y_domain,
    )

    domain = _context_chart_y_domain([89.6, 89.6, 89.6])
    assert domain is not None
    lower, upper = domain
    assert lower < 89.6 < upper
    padding = max(abs(89.6) * 0.02, 1e-6)
    assert lower == pytest.approx(89.6 - padding)
    assert upper == pytest.approx(89.6 + padding)
    chart = _build_anomaly_context_feature_chart(
        [
            {"relative_offset": -1, "value": 89.6},
            {"relative_offset": 0, "value": 89.6},
            {"relative_offset": 1, "value": 89.6},
        ],
        "RSOCmin",
    )
    assert chart is not None
    encoded = chart.to_dict()["encoding"]["y"]["scale"]
    assert encoded["domain"][0] < 89.6 < encoded["domain"][1]
    assert encoded.get("zero") is False


def test_context_chart_y_domain_constant_zero_visible() -> None:
    from process_intelligence.ui.streamlit_app import _context_chart_y_domain

    domain = _context_chart_y_domain([0.0, 0.0, 0.0])
    assert domain is not None
    lower, upper = domain
    assert lower < 0.0 < upper
    assert lower == pytest.approx(-1e-6)
    assert upper == pytest.approx(1e-6)


def test_context_chart_y_domain_constant_negative_visible() -> None:
    from process_intelligence.ui.streamlit_app import _context_chart_y_domain

    domain = _context_chart_y_domain([-12.5, -12.5])
    assert domain is not None
    lower, upper = domain
    assert lower < -12.5 < upper
    padding = max(abs(-12.5) * 0.02, 1e-6)
    assert lower == pytest.approx(-12.5 - padding)
    assert upper == pytest.approx(-12.5 + padding)


def test_context_chart_y_domain_non_constant_keeps_range() -> None:
    from process_intelligence.ui.streamlit_app import _context_chart_y_domain

    domain = _context_chart_y_domain([50.0, 51.0, 52.0, 53.0])
    assert domain is not None
    lower, upper = domain
    assert lower < 50.0
    assert upper > 53.0
    assert lower > 0.0


def test_context_chart_y_domain_small_variation_excludes_forced_zero() -> None:
    from process_intelligence.ui.streamlit_app import _context_chart_y_domain

    domain = _context_chart_y_domain([89.5, 89.6, 89.7])
    assert domain is not None
    lower, upper = domain
    assert lower < 89.5
    assert upper > 89.7
    assert lower > 80.0


def test_context_chart_rows_preserve_relative_offset_order() -> None:
    from process_intelligence.ui.streamlit_app import _context_chart_rows

    report = _presentation_report_with_context(
        _context_window_with_feature_values(
            [(1, 91.0), (-2, 89.6), (0, 90.0), (-1, 89.8), (2, 91.2)],
        )
    )
    window = report.anomaly_context_windows[0]
    rows = _context_chart_rows(window, "RSOCmin")
    assert [row["relative_offset"] for row in rows] == [1, -2, 0, -1, 2]
    chart_rows = [
        {"relative_offset": row["relative_offset"], "value": row["value"]}
        for row in rows
    ]
    from process_intelligence.ui.streamlit_app import (
        _build_anomaly_context_feature_chart,
    )

    chart = _build_anomaly_context_feature_chart(chart_rows, "RSOCmin")
    assert chart is not None
    frame = chart.data
    assert list(frame["relative_offset"]) == [-2, -1, 0, 1, 2]


def test_context_chart_empty_finite_values_safe_message() -> None:
    from process_intelligence.reporting.schemas import (
        AnomalyContextRowView,
        AnomalyContextValueView,
        AnomalyContextWindowView,
    )
    from process_intelligence.ui.streamlit_app import (
        _ANOMALY_CONTEXT_CHART_MISSING_CAPTION,
        _build_anomaly_context_feature_chart,
        _context_chart_rows,
        _context_chart_y_domain,
        _feature_values_are_numeric,
    )

    assert _context_chart_y_domain([]) is None
    assert _build_anomaly_context_feature_chart([], "RSOCmin") is None

    window = AnomalyContextWindowView(
        event_rank=1,
        center_original_row_id=10,
        center_anomaly_score=-0.4,
        radius=3,
        order_basis=AnomalyContextOrderBasis.LOADED_ROW_ORDER,
        feature_names=["RSOCmin"],
        rows=[
            AnomalyContextRowView(
                analysis_position=10,
                original_row_id=10,
                relative_offset=offset,
                is_center_event=offset == 0,
                is_selected_anomaly_event=offset == 0,
                feature_values=[
                    AnomalyContextValueView.model_construct(
                        feature_name="RSOCmin",
                        value=float("nan"),
                    )
                ],
            )
            for offset in (-1, 0, 1)
        ],
    )
    assert _feature_values_are_numeric(window, "RSOCmin")
    assert _context_chart_rows(window, "RSOCmin") == []

    source = _anomaly_only_report().model_copy(
        update={
            "anomaly_context_windows": [
                _context_window_with_feature_values(
                    [(-1, None), (0, None), (1, None)],
                )
            ]
        }
    )
    report = AnalysisWorkflowReportBuilder().build(source).report
    at = AppTest.from_function(
        _render_entry,
        default_timeout=30,
        kwargs={"report_json": report.model_dump(mode="json")},
    ).run()
    assert not at.exception
    blob = _text_blob(at) + str(at)
    assert "Traceback" not in blob
    assert "C:\\Users\\" not in blob
    assert (
        "non-numeric; chart is omitted" in blob
        or _ANOMALY_CONTEXT_CHART_MISSING_CAPTION in blob
    )


def test_anomaly_context_constant_series_chart_and_table_regression() -> None:
    report = _presentation_report_with_context(
        _context_window_with_feature_values(
            [(-3, 89.6), (-2, 89.6), (-1, 89.6), (0, 89.6), (1, 89.6), (2, 89.6), (3, 89.6)],
        )
    )
    at = AppTest.from_function(
        _render_entry,
        default_timeout=30,
        kwargs={"report_json": report.model_dump(mode="json")},
    ).run()
    assert not at.exception
    blob = _text_blob(at) + str(at)
    assert "Anomaly context explorer" in blob
    assert "relative offset 0" in blob.lower()
    assert "Traceback" not in blob
    assert "C:\\Users\\" not in blob
    assert any(item.label == "Download selected context CSV" for item in at.download_button)
    table_blob = str(at.dataframe)
    assert "89.6" in table_blob or "RSOCmin" in blob
    from process_intelligence.ui.streamlit_app import (
        _context_chart_rows,
        _context_chart_y_domain,
        _context_download_rows,
        _context_table_rows,
    )

    window = report.anomaly_context_windows[0]
    chart_rows = _context_chart_rows(window, "RSOCmin")
    domain = _context_chart_y_domain([float(row["value"]) for row in chart_rows])
    assert domain is not None
    assert domain[0] < 89.6 < domain[1]
    table_rows = _context_table_rows(window)
    assert [row["Relative offset"] for row in table_rows] == list(range(-3, 4))
    assert all(row["RSOCmin"] == 89.6 for row in table_rows)
    download_rows = _context_download_rows(window)
    assert [row["relative_offset"] for row in download_rows] == list(range(-3, 4))
    assert all(row["RSOCmin"] == 89.6 for row in download_rows)


# ---------------------------------------------------------------------------
# Step 11B.10 Operating cohort filter UI
# ---------------------------------------------------------------------------


def test_operating_cohort_filter_section_defaults_off() -> None:
    at = _upload_anomaly_csv(_make_anomaly_app().run())
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    assert not at.exception
    blob = _text_blob(at)
    assert "Operating cohort filter" in blob
    restrict = next(
        item
        for item in at.checkbox
        if item.label == "Restrict analysis to an operating range"
    )
    assert restrict.value is False
    labels = [item.label for item in at.selectbox]
    assert "Cohort column" not in labels


def test_cohort_filter_candidates_exclude_constant_soh_and_identifiers() -> None:
    at = _upload_anomaly_csv(_make_anomaly_app().run())
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    restrict = next(
        item
        for item in at.checkbox
        if item.label == "Restrict analysis to an operating range"
    )
    at = restrict.check().run()
    cohort_box = next(item for item in at.selectbox if item.label == "Cohort column")
    assert cohort_box.value == "(select cohort column)"
    assert "SOH" not in cohort_box.options
    assert "lot_id" not in cohort_box.options
    assert "timestamp" not in cohort_box.options
    assert "pressure" in cohort_box.options or "temperature" in cohort_box.options
    assert "(select cohort column)" in cohort_box.options


def test_cohort_filter_blank_bounds_block_readiness() -> None:
    at = _upload_anomaly_csv(_make_anomaly_app().run())
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    restrict = next(
        item
        for item in at.checkbox
        if item.label == "Restrict analysis to an operating range"
    )
    at = restrict.check().run()
    blob = _text_blob(at)
    assert "False: operating cohort filter valid" in blob
    run_button = next(item for item in at.button if item.label == "Run analysis")
    assert run_button.disabled is True


def test_cohort_filter_complete_submit_preserves_request() -> None:
    from process_intelligence.workflow import AnalysisWorkflowRequest

    at = _upload_anomaly_csv(_make_anomaly_capturing_app().run())
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    restrict = next(
        item
        for item in at.checkbox
        if item.label == "Restrict analysis to an operating range"
    )
    at = restrict.check().run()
    cohort_box = next(item for item in at.selectbox if item.label == "Cohort column")
    candidate = next(
        name
        for name in cohort_box.options
        if name not in {"(select cohort column)"}
    )
    at = cohort_box.select(candidate).run()
    lower = next(item for item in at.text_input if item.label == "Lower bound")
    upper = next(item for item in at.text_input if item.label == "Upper bound")
    at = lower.set_value("0").run()
    at = upper.set_value("1000").run()
    exclude = next(
        item
        for item in at.checkbox
        if item.label == "Exclude cohort column from anomaly features"
    )
    assert exclude.value is True
    blob = _text_blob(at)
    assert "Preview:" in blob
    assert "True: operating cohort filter valid" in blob
    run_button = next(item for item in at.button if item.label == "Run analysis")
    at = run_button.click().run()
    assert not at.exception
    assert _CountingCapturingWorkflow.run_count == 1
    request = _CountingCapturingWorkflow.last_request
    assert isinstance(request, AnalysisWorkflowRequest)
    assert request.cohort_filter is not None
    assert request.cohort_filter.column_name == candidate
    assert request.cohort_filter.lower_bound == pytest.approx(0.0)
    assert request.cohort_filter.upper_bound == pytest.approx(1000.0)
    assert request.cohort_filter.exclude_filter_column_from_features is True


def test_cohort_filter_off_does_not_submit_stale_filter() -> None:
    from process_intelligence.workflow import AnalysisWorkflowRequest

    at = _upload_anomaly_csv(_make_anomaly_capturing_app().run())
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    restrict = next(
        item
        for item in at.checkbox
        if item.label == "Restrict analysis to an operating range"
    )
    at = restrict.check().run()
    cohort_box = next(item for item in at.selectbox if item.label == "Cohort column")
    candidate = next(
        name
        for name in cohort_box.options
        if name not in {"(select cohort column)"}
    )
    at = cohort_box.select(candidate).run()
    lower = next(item for item in at.text_input if item.label == "Lower bound")
    upper = next(item for item in at.text_input if item.label == "Upper bound")
    at = lower.set_value("0").run()
    at = upper.set_value("1000").run()
    restrict = next(
        item
        for item in at.checkbox
        if item.label == "Restrict analysis to an operating range"
    )
    at = restrict.uncheck().run()
    run_button = next(item for item in at.button if item.label == "Run analysis")
    at = run_button.click().run()
    assert not at.exception
    request = _CountingCapturingWorkflow.last_request
    assert isinstance(request, AnalysisWorkflowRequest)
    assert request.cohort_filter is None


def test_supervised_mode_does_not_submit_cohort_filter() -> None:
    from process_intelligence.workflow import AnalysisWorkflowRequest

    at = _prepare_runnable_form(_upload_csv(_make_capturing_app().run()))
    run_button = next(item for item in at.button if item.label == "Run analysis")
    at = run_button.click().run()
    assert not at.exception
    request = _CapturingWorkflow.last_request
    assert isinstance(request, AnalysisWorkflowRequest)
    assert request.cohort_filter is None


def test_filtered_result_shows_cohort_interpretation_note() -> None:
    from process_intelligence.workflow import CohortFilterSummary

    filtered_source = _anomaly_only_report().model_copy(
        update={
            "cohort_row_count": 4,
            "train_row_count": 2,
            "validation_row_count": 1,
            "test_row_count": 1,
            "cohort_filter_summary": CohortFilterSummary(
                configured=True,
                column_name="pressure",
                lower_bound=10.0,
                upper_bound=100.0,
                include_lower=True,
                include_upper=True,
                exclude_filter_column_from_features=True,
                source_row_count=6,
                retained_row_count=4,
                excluded_row_count=2,
                null_excluded_count=0,
            ),
        }
    )
    filtered_report = AnalysisWorkflowReportBuilder().build(filtered_source).report

    def _render_entry(*, report_json: dict[str, object]) -> None:
        from process_intelligence.reporting.schemas import WorkflowPresentationReport
        from process_intelligence.ui.streamlit_app import render_presentation_report

        render_presentation_report(
            WorkflowPresentationReport.model_validate(report_json)
        )

    at = AppTest.from_function(
        _render_entry,
        default_timeout=30,
        kwargs={"report_json": filtered_report.model_dump(mode="json")},
    ).run()
    assert not at.exception
    blob = _text_blob(at)
    assert "Operating cohort filter" in blob
    assert "pressure" in blob
    assert (
        "Anomaly scores are relative to the selected operating cohort"
        in blob
    )
    assert "adjacent only within the filtered analysis-order cohort" in blob


# ---------------------------------------------------------------------------
# Anomaly run baseline comparison (Step 11B.11)
# ---------------------------------------------------------------------------


_COMPARISON_FINGERPRINT = (
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
)


def _presentation_from_anomaly_report(
    source: AnalysisWorkflowReport | None = None,
) -> WorkflowPresentationReport:
    report_source = _anomaly_only_report() if source is None else source
    return AnalysisWorkflowReportBuilder().build(report_source).report


def _render_with_optional_baseline(
    *,
    report_json: dict[str, object],
    baseline_json: dict[str, object] | None = None,
) -> None:
    import streamlit as st

    from process_intelligence.reporting.schemas import WorkflowPresentationReport
    from process_intelligence.ui.streamlit_app import render_presentation_report

    if baseline_json is not None:
        st.session_state["comparison_baseline_report_json"] = baseline_json
        st.session_state["comparison_baseline_label"] = "test-baseline"
    render_presentation_report(
        WorkflowPresentationReport.model_validate(report_json)
    )


def test_run_comparison_section_and_save_baseline_button() -> None:
    report = _presentation_from_anomaly_report()
    at = AppTest.from_function(
        _render_with_optional_baseline,
        default_timeout=30,
        kwargs={"report_json": report.model_dump(mode="json")},
    ).run()
    assert not at.exception
    blob = _text_blob(at)
    assert "Run comparison" in blob
    save_button = next(
        item
        for item in at.button
        if item.label == "Save current report as comparison baseline"
    )
    at = save_button.click().run()
    assert not at.exception
    state = at.session_state.filtered_state
    assert "comparison_baseline_report_json" in state
    baseline_json = state["comparison_baseline_report_json"]
    assert isinstance(baseline_json, dict)
    assert "anomaly_events" in baseline_json
    for value in state.values():
        assert not isinstance(value, (bytes, bytearray))
        assert "DataFrame" not in type(value).__name__
    assert "Baseline available" in _text_blob(at)


def test_save_baseline_does_not_rerun_workflow() -> None:
    _ANOMALY_WORKFLOW_RUN_COUNT["count"] = 0
    at = AppTest.from_function(
        _ui_entry,
        default_timeout=30,
        kwargs={"workflow_factory": lambda: _CountingAnomalyOnlyWorkflow()},
    ).run()
    at = _upload_anomaly_csv(at)
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    at = next(item for item in at.button if item.label == "Run analysis").click().run()
    assert _ANOMALY_WORKFLOW_RUN_COUNT["count"] == 1
    save_button = next(
        item
        for item in at.button
        if item.label == "Save current report as comparison baseline"
    )
    at = save_button.click().run()
    assert not at.exception
    assert _ANOMALY_WORKFLOW_RUN_COUNT["count"] == 1


def test_new_run_does_not_auto_replace_baseline() -> None:
    report = _presentation_from_anomaly_report()
    baseline_json = report.model_dump(mode="json")
    at = AppTest.from_function(
        _render_with_optional_baseline,
        default_timeout=30,
        kwargs={
            "report_json": report.model_dump(mode="json"),
            "baseline_json": baseline_json,
        },
    ).run()
    assert not at.exception
    labels = [item.label for item in at.button]
    assert "Replace comparison baseline" in labels
    assert "Clear comparison baseline" in labels
    assert "Save current report as comparison baseline" not in labels


def test_clear_comparison_baseline() -> None:
    report = _presentation_from_anomaly_report()
    at = AppTest.from_function(
        _render_with_optional_baseline,
        default_timeout=30,
        kwargs={
            "report_json": report.model_dump(mode="json"),
            "baseline_json": report.model_dump(mode="json"),
        },
    ).run()
    clear_button = next(
        item for item in at.button if item.label == "Clear comparison baseline"
    )
    at = clear_button.click().run()
    assert not at.exception
    state = at.session_state.filtered_state
    assert "comparison_baseline_report_json" not in state
    blob = _text_blob(at)
    assert "Baseline vs current anomaly analysis" not in blob
    assert "Overview" in blob


def test_same_dataset_comparison_sections_and_downloads() -> None:
    baseline_source = _anomaly_only_report()
    current_source = baseline_source.model_copy(
        update={
            "anomaly_event_count": 1,
            "diagnosis_factor_count": 1,
            "selected_operating_row_id": 5,
            "anomaly_events": [
                baseline_source.anomaly_events[1].model_copy(deep=True),
            ],
            "diagnosis_factors": [
                baseline_source.diagnosis_factors[0].model_copy(deep=True),
            ],
        }
    )
    baseline = _presentation_from_anomaly_report(baseline_source)
    current = _presentation_from_anomaly_report(current_source)
    at = AppTest.from_function(
        _render_with_optional_baseline,
        default_timeout=30,
        kwargs={
            "report_json": current.model_dump(mode="json"),
            "baseline_json": baseline.model_dump(mode="json"),
        },
    ).run()
    assert not at.exception
    blob = _text_blob(at)
    assert "Baseline vs current anomaly analysis" in blob
    assert "Configuration comparison" in blob
    assert "Event overlap" in blob
    assert "Diagnosis factor comparison" in blob
    assert "Interpretation note" in blob
    assert "Anomaly score magnitudes are not directly comparable" in blob
    assert "Shared events indicate row-selection stability" in blob
    download_labels = [item.label for item in at.download_button]
    assert "Download event overlap CSV" in download_labels
    assert "Download factor comparison CSV" in download_labels
    serialized = str(at)
    assert "score_delta" not in serialized
    assert "anomaly_score_delta" not in serialized


def test_different_dataset_compatibility_warning() -> None:
    baseline = _presentation_from_anomaly_report()
    current = _presentation_from_anomaly_report(
        _anomaly_only_report().model_copy(
            update={"dataset_fingerprint": _COMPARISON_FINGERPRINT}
        )
    )
    at = AppTest.from_function(
        _render_with_optional_baseline,
        default_timeout=30,
        kwargs={
            "report_json": current.model_dump(mode="json"),
            "baseline_json": baseline.model_dump(mode="json"),
        },
    ).run()
    assert not at.exception
    blob = _text_blob(at)
    assert "different datasets" in blob.lower()
    assert "Baseline vs current anomaly analysis" not in blob


def test_supervised_anomaly_mismatch_warning() -> None:
    anomaly = _presentation_from_anomaly_report()
    shared_fingerprint = anomaly.dataset_fingerprint
    assert shared_fingerprint is not None
    supervised_source = _completed_report().model_copy(
        update={
            "dataset_fingerprint": shared_fingerprint,
            "metadata": {
                **_completed_report().metadata,
                "analysis_mode": "SUPERVISED",
            },
        }
    )
    supervised = AnalysisWorkflowReportBuilder().build(supervised_source).report
    at = AppTest.from_function(
        _render_with_optional_baseline,
        default_timeout=30,
        kwargs={
            "report_json": anomaly.model_dump(mode="json"),
            "baseline_json": supervised.model_dump(mode="json"),
        },
    ).run()
    assert not at.exception
    blob = _text_blob(at)
    assert "ANOMALY_ONLY" in blob
    assert "Baseline vs current anomaly analysis" not in blob


def test_comparison_page_has_no_absolute_path_or_traceback() -> None:
    report = _presentation_from_anomaly_report()
    at = AppTest.from_function(
        _render_with_optional_baseline,
        default_timeout=30,
        kwargs={
            "report_json": report.model_dump(mode="json"),
            "baseline_json": report.model_dump(mode="json"),
        },
    ).run()
    assert not at.exception
    blob = _text_blob(at)
    assert "Traceback" not in blob
    assert "C:\\Users\\" not in blob
    assert "/Users/" not in blob
    assert report.dataset_fingerprint is not None
    assert report.dataset_fingerprint not in blob


def test_baseline_summary_without_full_fingerprint() -> None:
    report = _presentation_from_anomaly_report()
    at = AppTest.from_function(
        _render_with_optional_baseline,
        default_timeout=30,
        kwargs={
            "report_json": report.model_dump(mode="json"),
            "baseline_json": report.model_dump(mode="json"),
        },
    ).run()
    blob = _text_blob(at)
    assert "Baseline available" in blob
    assert "Analysis mode" in blob
    assert "ANOMALY_ONLY" in blob
    assert report.dataset_fingerprint is not None
    assert report.dataset_fingerprint not in blob


# ---------------------------------------------------------------------------
# Analysis configuration preset import/export
# ---------------------------------------------------------------------------


def _compatible_supervised_preset_json() -> bytes:
    from process_intelligence.ui.configuration_preset import (
        build_configuration_preset,
        configuration_preset_to_json,
    )
    from process_intelligence.ui.schemas import UiMetricRuleInput
    from process_intelligence.workflow import AnalysisExecutionMode

    preset = build_configuration_preset(
        analysis_mode=AnalysisExecutionMode.SUPERVISED,
        target_column="quality",
        use_recommended_numeric_feature_set=True,
        timestamp_column="timestamp",
        identifier_columns=["lot_id"],
        excluded_columns=["notes"],
        recommendation_objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        performance_rules=[
            UiMetricRuleInput(
                metric_name="rmse",
                direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
                threshold=12.5,
                required=True,
            )
        ],
        maximum_simultaneous_changes=2,
    )
    return configuration_preset_to_json(preset).encode("utf-8")


def _incompatible_preset_json() -> bytes:
    from process_intelligence.ui.configuration_preset import (
        build_configuration_preset,
        configuration_preset_to_json,
    )
    from process_intelligence.workflow import AnalysisExecutionMode

    preset = build_configuration_preset(
        analysis_mode=AnalysisExecutionMode.SUPERVISED,
        target_column="absent_target",
        maximum_simultaneous_changes=1,
    )
    return configuration_preset_to_json(preset).encode("utf-8")


def _upload_preset_json(
    at: AppTest,
    payload: bytes,
    *,
    name: str = "process_intelligence_configuration.json",
) -> AppTest:
    preset_uploader = next(
        item for item in at.file_uploader if item.label == "Upload configuration JSON"
    )
    return preset_uploader.set_value([(name, payload, "application/json")]).run()


def test_configuration_preset_section_rendering() -> None:
    at = _upload_csv(_make_app().run())
    assert not at.exception
    blob = _text_blob(at)
    assert "Analysis configuration preset" in blob


def test_configuration_preset_download_button_present() -> None:
    at = _upload_csv(_make_app().run())
    labels = [item.label for item in at.download_button]
    assert "Download configuration JSON" in labels


def test_configuration_preset_uploader_present() -> None:
    at = _upload_csv(_make_app().run())
    labels = [item.label for item in at.file_uploader]
    assert "CSV upload" in labels
    assert "Upload configuration JSON" in labels
    assert len(at.file_uploader) == 2


def test_configuration_preset_apply_before_click_does_not_change_ui_state() -> None:
    at = _upload_csv(_make_app().run())
    before_mode = next(
        item for item in at.selectbox if item.label == "Analysis mode"
    ).value
    before_target = next(
        item for item in at.selectbox if item.label == "Target column"
    ).value
    at = _upload_preset_json(at, _compatible_supervised_preset_json())
    assert not at.exception
    after_mode = next(
        item for item in at.selectbox if item.label == "Analysis mode"
    ).value
    after_target = next(
        item for item in at.selectbox if item.label == "Target column"
    ).value
    assert after_mode == before_mode
    assert after_target == before_target
    apply_button = next(
        item for item in at.button if item.label == "Apply configuration"
    )
    assert apply_button.disabled is False


def test_configuration_preset_invalid_shows_error() -> None:
    at = _upload_csv(_make_app().run())
    at = _upload_preset_json(at, b"{not-json", name="bad.json")
    assert not at.exception
    blob = _text_blob(at)
    assert "JSON parsing failed" in blob or "validation failed" in blob.lower()
    apply_button = next(
        item for item in at.button if item.label == "Apply configuration"
    )
    assert apply_button.disabled is True


def test_configuration_preset_compatible_apply_success() -> None:
    at = _upload_csv(_make_app().run())
    at = _upload_preset_json(at, _compatible_supervised_preset_json())
    apply_button = next(
        item for item in at.button if item.label == "Apply configuration"
    )
    at = apply_button.click().run()
    assert not at.exception
    blob = _text_blob(at)
    assert "Configuration applied" in blob
    target_box = next(item for item in at.selectbox if item.label == "Target column")
    assert target_box.value == "quality"
    objective_box = next(item for item in at.selectbox if item.label == "Objective")
    assert objective_box.value == RecommendationObjective.REDUCE_ANOMALY_SCORE.value


def test_configuration_preset_incompatible_columns_rejected() -> None:
    at = _upload_csv(_make_app().run())
    before_target = next(
        item for item in at.selectbox if item.label == "Target column"
    ).value
    at = _upload_preset_json(at, _incompatible_preset_json(), name="bad_columns.json")
    assert not at.exception
    blob = _text_blob(at)
    assert "absent_target" in blob
    apply_button = next(
        item for item in at.button if item.label == "Apply configuration"
    )
    assert apply_button.disabled is True
    after_target = next(
        item for item in at.selectbox if item.label == "Target column"
    ).value
    assert after_target == before_target


def test_configuration_preset_import_does_not_auto_run_workflow() -> None:
    at = _upload_anomaly_csv(_make_anomaly_capturing_app().run())
    from process_intelligence.ui.configuration_preset import (
        build_configuration_preset,
        configuration_preset_to_json,
    )
    from process_intelligence.workflow import AnalysisExecutionMode

    assert _CountingCapturingWorkflow.run_count == 0
    preset_bytes = configuration_preset_to_json(
        build_configuration_preset(
            analysis_mode=AnalysisExecutionMode.ANOMALY_ONLY,
            use_recommended_numeric_feature_set=True,
            maximum_simultaneous_changes=2,
        )
    ).encode("utf-8")
    at = _upload_preset_json(at, preset_bytes)
    apply_button = next(
        item for item in at.button if item.label == "Apply configuration"
    )
    at = apply_button.click().run()
    assert not at.exception
    assert _CountingCapturingWorkflow.run_count == 0
    blob = _text_blob(at)
    assert "Run readiness" in blob


def test_configuration_preset_session_state_has_no_forbidden_runtime_objects() -> None:
    at = _upload_csv(_make_app().run())
    at = _upload_preset_json(at, _compatible_supervised_preset_json())
    apply_button = next(
        item for item in at.button if item.label == "Apply configuration"
    )
    at = apply_button.click().run()
    assert not at.exception
    state = at.session_state.filtered_state
    preset_keys = {
        "ui_configuration_preset_parsed",
        "ui_configuration_preset_validation_message",
        "ui_configuration_preset_validation_ok",
        "ui_configuration_preset_missing_columns",
        "ui_configuration_preset_apply_summary",
        "ui_configuration_preset_pending_apply",
    }
    for key, value in state.items():
        assert not isinstance(value, (bytes, bytearray, memoryview)), key
        type_name = type(value).__name__
        assert "DataFrame" not in type_name, key
        assert "Estimator" not in type_name, key
        assert not hasattr(value, "predict"), key
        if key in preset_keys:
            assert type_name not in {
                "UploadedFile",
                "Path",
                "WindowsPath",
                "PosixPath",
                "AnalysisWorkflowRequest",
                "WorkflowPresentationReport",
            }, key
            if isinstance(value, dict):
                json.dumps(value)


def test_configuration_preset_download_present_when_valid() -> None:
    at = _upload_csv(_make_app().run())
    labels = [item.label for item in at.download_button]
    assert "Download configuration JSON" in labels
    blob = _text_blob(at)
    assert "Complete or clear the highlighted configuration rows" not in blob


def test_configuration_preset_download_blocked_for_incomplete_rule() -> None:
    at = _upload_csv(_make_app().run())
    metric_box = next(
        item for item in at.text_input if item.label.startswith("Metric name #")
    )
    at = metric_box.set_value("rmse").run()
    assert not at.exception
    labels = [item.label for item in at.download_button]
    assert "Download configuration JSON" not in labels
    blob = _text_blob(at)
    assert (
        "Complete or clear the highlighted configuration rows before "
        "exporting the preset."
    ) in blob
    assert "Performance rule 1 is incomplete" in blob


def test_configuration_preset_download_blocked_for_incomplete_constraint() -> None:
    at = _upload_csv(_make_app().run())
    constraint_box = next(
        item
        for item in at.multiselect
        if item.label == "Variables with recommendation constraints"
    )
    at = constraint_box.set_value(["pressure"]).run()
    min_box = next(
        item for item in at.number_input if item.label == "Minimum for pressure"
    )
    at = min_box.set_value(10.0).run()
    max_box = next(
        item for item in at.number_input if item.label == "Maximum for pressure"
    )
    at = max_box.set_value(1.0).run()
    assert not at.exception
    labels = [item.label for item in at.download_button]
    assert "Download configuration JSON" not in labels
    blob = _text_blob(at)
    assert "Constraint for 'pressure' is invalid" in blob
    assert "lower bound must not exceed upper bound" in blob


def test_configuration_preset_export_issues_do_not_auto_correct_settings() -> None:
    at = _upload_csv(_make_app().run())
    metric_box = next(
        item for item in at.text_input if item.label.startswith("Metric name #")
    )
    at = metric_box.set_value("rmse").run()
    metric_after = next(
        item for item in at.text_input if item.label.startswith("Metric name #")
    )
    assert metric_after.value == "rmse"
    direction_box = next(
        item for item in at.selectbox if item.label.startswith("Direction #")
    )
    assert direction_box.value == "(select direction)"
    threshold_box = next(
        item for item in at.text_input if item.label.startswith("Threshold #")
    )
    assert threshold_box.value == ""


def test_configuration_preset_export_validation_does_not_auto_run_workflow() -> None:
    at = _upload_anomaly_csv(_make_anomaly_capturing_app().run())
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    assert _CountingCapturingWorkflow.run_count == 0
    enable_box = next(
        item
        for item in at.checkbox
        if item.label == "Enable anomaly-score reduction recommendation"
    )
    at = enable_box.check().run()
    review_box = next(
        item
        for item in at.multiselect
        if item.label == "Candidate variables with explicit role overrides"
    )
    at = review_box.set_value(["pressure"]).run()
    constraint_box = next(
        item
        for item in at.multiselect
        if item.label == "Variables with recommendation constraints"
    )
    at = constraint_box.set_value(["pressure"]).run()
    assert not at.exception
    assert _CountingCapturingWorkflow.run_count == 0
    labels = [item.label for item in at.download_button]
    assert "Download configuration JSON" not in labels
    blob = _text_blob(at)
    assert "Constraint for 'pressure' is incomplete" in blob
    assert "Run readiness" in blob


def test_configuration_preset_import_still_works_with_export_validation() -> None:
    at = _upload_csv(_make_app().run())
    at = _upload_preset_json(at, _compatible_supervised_preset_json())
    apply_button = next(
        item for item in at.button if item.label == "Apply configuration"
    )
    at = apply_button.click().run()
    assert not at.exception
    blob = _text_blob(at)
    assert "Configuration applied" in blob
    labels = [item.label for item in at.download_button]
    assert "Download configuration JSON" in labels


def test_configuration_preset_download_blocked_for_incomplete_cohort_filter() -> None:
    at = _upload_anomaly_csv(_make_anomaly_app().run())
    at = _select_analysis_mode(at, "ANOMALY_ONLY")
    restrict = next(
        item
        for item in at.checkbox
        if item.label == "Restrict analysis to an operating range"
    )
    at = restrict.check().run()
    assert not at.exception
    labels = [item.label for item in at.download_button]
    assert "Download configuration JSON" not in labels
    blob = _text_blob(at)
    assert "Operating cohort filter is incomplete" in blob
    assert "Current configuration summary" in blob
    assert "Enabled (incomplete)" in blob


def test_configuration_preset_export_keeps_wide_csv_ui() -> None:
    at = _upload_csv(_make_app().run())
    labels = [item.label for item in at.selectbox]
    assert "Target column" in labels
    assert "Analysis mode" in labels
    assert "Objective" in labels
    assert any(item.label == "CSV upload" for item in at.file_uploader)
    assert "Analysis configuration preset" in _text_blob(at)


# ---------------------------------------------------------------------------
# Built-in manufacturing demo data source
# ---------------------------------------------------------------------------


def _select_builtin_demo(at: AppTest) -> AppTest:
    radio = next(item for item in at.radio if item.label == "Data source")
    return radio.set_value("Built-in manufacturing demo").run()


def _select_demo_template(at: AppTest, template: str) -> AppTest:
    box = next(
        item for item in at.selectbox if item.label == "Demo analysis template"
    )
    return box.select(template).run()


def _apply_demo_configuration(at: AppTest) -> AppTest:
    button = next(
        item for item in at.button if item.label == "Apply demo configuration"
    )
    return button.click().run()


def _run_readiness_flags(at: AppTest) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    for line in _text_blob(at).splitlines():
        value = line.strip()
        if value.startswith("True: "):
            flags[value.removeprefix("True: ")] = True
        elif value.startswith("False: "):
            flags[value.removeprefix("False: ")] = False
    return flags


def test_builtin_demo_default_source_remains_upload_csv() -> None:
    at = _make_app().run()
    assert not at.exception
    radio = next(item for item in at.radio if item.label == "Data source")
    assert radio.value == "Upload CSV"
    assert any(item.label == "CSV upload" for item in at.file_uploader)


def test_builtin_demo_loads_in_memory_dataset() -> None:
    at = _select_builtin_demo(_make_app().run())
    assert not at.exception
    blob = _text_blob(at)
    assert "synthetic built-in manufacturing demo" in blob.lower()
    assert "Ground-truth demo metadata" in blob
    assert "injected_anomaly" in blob
    assert "anomaly_type" in blob
    metrics = {item.label: item.value for item in at.metric}
    assert int(metrics.get("Rows", -1)) == 1500
    assert int(metrics.get("Columns", -1)) == 17
    assert int(metrics.get("Seed", -1)) == 42
    assert int(metrics.get("Anomaly rows", -1)) > 0
    assert not any(item.label == "CSV upload" for item in at.file_uploader)
    state = at.session_state.filtered_state
    assert "last_presentation_report_json" not in state


def test_builtin_demo_template_change_alone_does_not_apply() -> None:
    at = _select_builtin_demo(_make_app().run())
    at = _select_demo_template(at, "Anomaly-only process monitoring")
    assert not at.exception
    mode_box = next(item for item in at.selectbox if item.label == "Analysis mode")
    # Default UI mode remains SUPERVISED until Apply is clicked.
    assert mode_box.value == "SUPERVISED"
    assert "Demo configuration applied." not in _text_blob(at)
    state = at.session_state.filtered_state
    assert "last_presentation_report_json" not in state


def test_builtin_demo_apply_supervised_template() -> None:
    at = _select_builtin_demo(_make_app().run())
    at = _select_demo_template(at, "Supervised quality prediction")
    at = _apply_demo_configuration(at)
    assert not at.exception
    blob = _text_blob(at)
    assert "Demo configuration applied." in blob
    assert "Verified controllable variables: 4" in blob
    assert "temperature_setpoint" in blob
    assert "demonstration-only" in blob.lower()
    mode_box = next(item for item in at.selectbox if item.label == "Analysis mode")
    assert mode_box.value == "SUPERVISED"
    target_box = next(item for item in at.selectbox if item.label == "Target column")
    assert target_box.value == "quality_score"
    task_box = next(item for item in at.selectbox if item.label == "Analysis task")
    assert task_box.value == "REGRESSION"
    timestamp_box = next(
        item for item in at.selectbox if item.label == "Timestamp column (optional)"
    )
    assert timestamp_box.value == "timestamp"
    state = at.session_state.filtered_state
    assert state.get("ui_explicit_feature_columns") == [
        "temperature_setpoint",
        "pressure_setpoint",
        "flow_rate_setpoint",
        "cycle_time_setpoint",
        "temperature_actual",
        "pressure_actual",
        "flow_rate_actual",
        "vibration",
        "motor_current",
        "chamber_humidity",
    ]
    assert state.get("ui_use_recommended_numeric_feature_set") is False
    assert state.get("ui_confirmed_controllable_variables") == [
        "temperature_setpoint",
        "pressure_setpoint",
        "flow_rate_setpoint",
        "cycle_time_setpoint",
    ]
    assert state.get("ui_verified_variables") == [
        "temperature_setpoint",
        "pressure_setpoint",
        "flow_rate_setpoint",
        "cycle_time_setpoint",
    ]
    assert state.get("ui_recommendation_constraint_columns") == [
        "temperature_setpoint",
        "pressure_setpoint",
        "flow_rate_setpoint",
        "cycle_time_setpoint",
    ]
    assert state.get("constraint_min_temperature_setpoint") == 170.0
    assert state.get("constraint_max_temperature_setpoint") == 200.0
    assert state.get("constraint_min_pressure_setpoint") == 1.8
    assert state.get("constraint_max_pressure_setpoint") == 3.2
    assert state.get("constraint_min_flow_rate_setpoint") == 90.0
    assert state.get("constraint_max_flow_rate_setpoint") == 150.0
    assert state.get("constraint_min_cycle_time_setpoint") == 40.0
    assert state.get("constraint_max_cycle_time_setpoint") == 60.0
    assert "last_presentation_report_json" not in state
    run_button = next(item for item in at.button if item.label == "Run analysis")
    assert run_button.disabled is False
    flags = _run_readiness_flags(at)
    assert flags.get("demo features exclude forbidden columns") is True
    assert all(flags.values())
    summary_blob = _text_blob(at)
    assert "Constraint variable count: `4`" in summary_blob or (
        "constraint" in summary_blob.lower() and "4" in summary_blob
    )


def test_builtin_demo_template_change_alone_does_not_change_controllables() -> None:
    at = _select_builtin_demo(_make_app().run())
    at = _apply_demo_configuration(at)
    state = at.session_state.filtered_state
    assert len(state.get("ui_confirmed_controllable_variables", [])) == 4
    at = _select_demo_template(at, "Anomaly-only process monitoring")
    assert not at.exception
    state = at.session_state.filtered_state
    # Selecting a template without Apply must not clear/apply controllables.
    assert state.get("ui_confirmed_controllable_variables") == [
        "temperature_setpoint",
        "pressure_setpoint",
        "flow_rate_setpoint",
        "cycle_time_setpoint",
    ]
    assert "Demo configuration applied." not in _text_blob(at)
    assert "last_presentation_report_json" not in state


def test_builtin_demo_apply_anomaly_only_template() -> None:
    at = _select_builtin_demo(_make_app().run())
    at = _select_demo_template(at, "Anomaly-only process monitoring")
    at = _apply_demo_configuration(at)
    assert not at.exception
    mode_box = next(item for item in at.selectbox if item.label == "Analysis mode")
    assert mode_box.value == "ANOMALY_ONLY"
    assert "Target column" not in [item.label for item in at.selectbox]
    timestamp_box = next(
        item for item in at.selectbox if item.label == "Timestamp column (optional)"
    )
    assert timestamp_box.value == "timestamp"
    operating_box = next(
        item
        for item in at.selectbox
        if item.label == "Operating-point selection mode"
    )
    assert operating_box.value == "TOP_UNSUPERVISED_ANOMALY"
    state = at.session_state.filtered_state
    assert state.get("anomaly_recommendation_enabled") is False
    assert state.get("ui_restrict_operating_cohort") is False
    assert state.get("ui_confirmed_controllable_variables") == []
    assert state.get("ui_verified_variables") == []
    assert state.get("ui_recommendation_constraint_columns") == []
    assert len(state.get("ui_explicit_feature_columns", [])) == 10
    assert "last_presentation_report_json" not in state
    run_button = next(item for item in at.button if item.label == "Run analysis")
    assert run_button.disabled is False
    flags = _run_readiness_flags(at)
    assert flags.get("demo features exclude forbidden columns") is True
    assert all(flags.values())


def test_builtin_demo_switch_anomaly_clears_supervised_controllables() -> None:
    at = _select_builtin_demo(_make_app().run())
    at = _select_demo_template(at, "Supervised quality prediction")
    at = _apply_demo_configuration(at)
    assert at.session_state.filtered_state.get(
        "ui_confirmed_controllable_variables"
    ) == [
        "temperature_setpoint",
        "pressure_setpoint",
        "flow_rate_setpoint",
        "cycle_time_setpoint",
    ]
    at = _select_demo_template(at, "Anomaly-only process monitoring")
    at = _apply_demo_configuration(at)
    state = at.session_state.filtered_state
    assert state.get("ui_confirmed_controllable_variables") == []
    assert state.get("ui_verified_variables") == []
    assert state.get("ui_recommendation_constraint_columns") == []
    assert state.get("anomaly_recommendation_enabled") is False


def test_builtin_demo_switch_to_upload_clears_demo_controllables() -> None:
    at = _select_builtin_demo(_make_app().run())
    at = _apply_demo_configuration(at)
    assert len(
        at.session_state.filtered_state.get("ui_confirmed_controllable_variables", [])
    ) == 4
    radio = next(item for item in at.radio if item.label == "Data source")
    at = radio.set_value("Upload CSV").run()
    state = at.session_state.filtered_state
    assert state.get("ui_confirmed_controllable_variables") == []
    assert state.get("ui_verified_variables") == []
    assert state.get("ui_recommendation_constraint_columns") == []
    assert "constraint_min_temperature_setpoint" not in state
    assert any(item.label == "CSV upload" for item in at.file_uploader)


def test_builtin_demo_no_auto_run_on_source_or_apply() -> None:
    at = _select_builtin_demo(_make_app().run())
    assert "last_presentation_report_json" not in at.session_state.filtered_state
    at = _apply_demo_configuration(at)
    assert "last_presentation_report_json" not in at.session_state.filtered_state
    # Deterministic workflow factory would populate a report only after Run.
    assert "Overview" not in [item.value for item in at.header]


def test_builtin_demo_source_switch_clears_stale_results() -> None:
    at = _prepare_runnable_form(_upload_csv(_make_app().run()))
    at = next(item for item in at.button if item.label == "Run analysis").click().run()
    assert "last_presentation_report_json" in at.session_state.filtered_state

    at = _select_builtin_demo(at)
    state = at.session_state.filtered_state
    assert "last_presentation_report_json" not in state
    assert "comparison_baseline_report_json" not in state

    at = _apply_demo_configuration(at)
    at = next(item for item in at.button if item.label == "Run analysis").click().run()
    assert "last_presentation_report_json" in at.session_state.filtered_state

    radio = next(item for item in at.radio if item.label == "Data source")
    at = radio.set_value("Upload CSV").run()
    state = at.session_state.filtered_state
    assert "last_presentation_report_json" not in state
    assert "comparison_baseline_report_json" not in state
    # Upload flow is active again; demo frame is not silently retained.
    assert any(item.label == "CSV upload" for item in at.file_uploader)
    assert "Upload a valid CSV" in _text_blob(at)


def test_builtin_demo_forbidden_feature_disables_run() -> None:
    at = _select_builtin_demo(_make_app().run())
    at = _apply_demo_configuration(at)
    assert not at.exception
    feature_box = next(
        item for item in at.multiselect if item.label == "Feature columns"
    )
    at = feature_box.set_value(
        ["temperature_setpoint", "pressure_setpoint", "injected_anomaly"]
    ).run()
    assert not at.exception
    blob = _text_blob(at)
    assert "injected_anomaly" in blob
    assert "must not include" in blob.lower() or "Ground-truth" in blob
    run_button = next(item for item in at.button if item.label == "Run analysis")
    assert run_button.disabled is True
    flags = _run_readiness_flags(at)
    assert flags.get("demo features exclude forbidden columns") is False

