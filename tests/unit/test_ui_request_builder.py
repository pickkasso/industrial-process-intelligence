"""Unit tests for WorkflowUiRequestBuilder (Step 11B)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from process_intelligence.core.enums import AnalysisTask, ColumnRole
from process_intelligence.evaluation import MetricAcceptanceDirection
from process_intelligence.recommendation import (
    QualityOptimizationDirection,
    RecommendationObjective,
)
from process_intelligence.ui import (
    UiMetricRuleInput,
    UiVariableConstraintInput,
    WorkflowUiRequestBuilder,
    WorkflowUiSubmission,
)
from process_intelligence.workflow import (
    AnalysisExecutionMode,
    AnalysisWorkflowPolicy,
    AnalysisWorkflowRequest,
    OperatingPointSelectionMode,
)


def _rule(**overrides: Any) -> UiMetricRuleInput:
    payload: dict[str, Any] = {
        "metric_name": "rmse",
        "direction": MetricAcceptanceDirection.LOWER_IS_BETTER,
        "threshold": 12.5,
        "required": True,
    }
    payload.update(overrides)
    return UiMetricRuleInput(**payload)


def _submission(**overrides: Any) -> WorkflowUiSubmission:
    payload: dict[str, Any] = {
        "target_column": "quality",
        "feature_columns": ["pressure", "temperature"],
        "timestamp_column": "timestamp",
        "identifier_columns": ["lot_id"],
        "excluded_columns": ["notes"],
        "column_role_overrides": {"pressure": ColumnRole.CONTROLLABLE_PROCESS},
        "objective": RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        "quality_direction": QualityOptimizationDirection.MAXIMIZE,
        "quality_target": None,
        "performance_rules": [
            _rule(),
            _rule(
                metric_name="mae",
                direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
                threshold=9.0,
                required=False,
            ),
        ],
        "constraints": [
            UiVariableConstraintInput(
                variable="pressure",
                minimum=10.0,
                maximum=90.0,
            )
        ],
        "user_confirmed_controllable_variables": ["pressure"],
        "user_verified_variables": ["temperature"],
        "max_simultaneous_changes": 2,
        "operating_point_selection": OperatingPointSelectionMode.TOP_RESIDUAL_ANOMALY,
        "explicit_operating_row_id": None,
        "metadata": {"caller": "unit"},
    }
    payload.update(overrides)
    return WorkflowUiSubmission(**payload)


def _write_csv(path: Path) -> Path:
    path.write_text("pressure,temperature,quality\n1,2,3\n", encoding="utf-8")
    return path


def test_builds_valid_request(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "sample.csv")
    builder = WorkflowUiRequestBuilder()
    request = builder.build(csv_path=csv_path, submission=_submission())
    assert isinstance(request, AnalysisWorkflowRequest)
    assert request.csv_path == csv_path
    assert request.target_column == "quality"
    assert request.feature_columns == ["pressure", "temperature"]
    assert request.requested_task is None
    assert request.industry_constraints == []
    assert request.user_overrides == []


def test_builder_preserves_requested_task_auto(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "sample.csv")
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=_submission(requested_task=None, target_column="SOH"),
    )
    assert request.target_column == "SOH"
    assert request.requested_task is None


def test_builder_preserves_requested_task_regression(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "sample.csv")
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=_submission(requested_task=AnalysisTask.REGRESSION),
    )
    assert request.requested_task is AnalysisTask.REGRESSION


def test_builder_preserves_requested_task_classification(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "sample.csv")
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=_submission(requested_task=AnalysisTask.CLASSIFICATION),
    )
    assert request.requested_task is AnalysisTask.CLASSIFICATION


def test_builder_does_not_infer_task_from_soh_name(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "sample.csv")
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=_submission(target_column="SOH", requested_task=None),
    )
    assert request.target_column == "SOH"
    assert request.requested_task is None


def test_performance_rule_conversion(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "sample.csv")
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=_submission(),
    )
    rules = request.model_performance_policy.rules
    assert len(rules) == 2
    assert rules[0].metric_name == "rmse"
    assert rules[0].threshold == 12.5
    assert rules[0].direction is MetricAcceptanceDirection.LOWER_IS_BETTER
    assert rules[1].required is False


def test_constraint_conversion(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "sample.csv")
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=_submission(),
    )
    assert len(request.request_constraints) == 1
    item = request.request_constraints[0]
    assert item.variable == "pressure"
    assert item.minimum == 10.0
    assert item.maximum == 90.0
    assert item.adjustable is True
    assert item.fixed is False


def test_objective_preserved(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "sample.csv")
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=_submission(),
    )
    assert request.objective is RecommendationObjective.IMPROVE_PREDICTED_QUALITY


def test_quality_direction_preserved(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "sample.csv")
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=_submission(),
    )
    assert request.quality_direction is QualityOptimizationDirection.MAXIMIZE


def test_confirmation_preserved(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "sample.csv")
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=_submission(),
    )
    assert request.user_confirmed_controllable_variables == ["pressure"]


def test_verification_preserved(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "sample.csv")
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=_submission(),
    )
    assert request.user_verified_variables == ["temperature"]


def test_operating_mode_preserved(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "sample.csv")
    submission = _submission(
        operating_point_selection=OperatingPointSelectionMode.EXPLICIT_ROW_ID,
        explicit_operating_row_id="row-7",
    )
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=submission,
    )
    assert (
        request.operating_point_selection
        is OperatingPointSelectionMode.EXPLICIT_ROW_ID
    )
    assert request.explicit_operating_row_id == "row-7"


def test_missing_csv_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        WorkflowUiRequestBuilder().build(
            csv_path=missing,
            submission=_submission(),
        )


def test_directory_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "not_a_file.csv"
    directory.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        WorkflowUiRequestBuilder().build(
            csv_path=directory,
            submission=_submission(),
        )


def test_non_csv_rejected(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="\\.csv"):
        WorkflowUiRequestBuilder().build(
            csv_path=path,
            submission=_submission(),
        )


def test_submission_type_error(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "sample.csv")
    with pytest.raises(TypeError):
        WorkflowUiRequestBuilder().build(
            csv_path=csv_path,
            submission={"target_column": "quality"},  # type: ignore[arg-type]
        )


def test_no_threshold_inference(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "sample.csv")
    submission = _submission(
        performance_rules=[
            _rule(
                metric_name="r2",
                direction=MetricAcceptanceDirection.HIGHER_IS_BETTER,
                threshold=0.01,
            )
        ]
    )
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=submission,
    )
    assert request.model_performance_policy.rules[0].threshold == 0.01
    assert (
        request.model_performance_policy.rules[0].direction
        is MetricAcceptanceDirection.HIGHER_IS_BETTER
    )


def test_no_constraint_inference(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "sample.csv")
    submission = _submission(constraints=[])
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=submission,
    )
    assert request.request_constraints == []


def test_no_controllability_inference(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "sample.csv")
    submission = _submission(
        user_confirmed_controllable_variables=[],
        user_verified_variables=[],
    )
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=submission,
    )
    assert request.user_confirmed_controllable_variables == []
    assert request.user_verified_variables == []


def test_metadata_has_no_path(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "secret_name.csv")
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=_submission(),
    )
    serialized = str(request.metadata)
    assert "secret_name.csv" not in serialized
    assert str(csv_path) not in serialized
    assert "\\" not in serialized or "ui_source" in request.metadata
    assert request.metadata.get("ui_source") == "streamlit_mvp"


def test_input_immutability(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "sample.csv")
    submission = _submission()
    before = submission.model_dump(mode="python")
    WorkflowUiRequestBuilder().build(csv_path=csv_path, submission=submission)
    assert submission.model_dump(mode="python") == before


def test_builder_state_independence(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "sample.csv")
    policy = AnalysisWorkflowPolicy(maximum_aggregated_warnings=11)
    builder_a = WorkflowUiRequestBuilder(workflow_policy=policy)
    builder_b = WorkflowUiRequestBuilder(workflow_policy=policy)
    policy.maximum_aggregated_warnings = 99
    meta_a = builder_a.get_metadata()
    meta_b = builder_b.get_metadata()
    assert meta_a["workflow_policy_stop_on_validation_blocker"] is True
    assert meta_b["workflow_policy_stop_on_validation_blocker"] is True
    request = builder_a.build(csv_path=csv_path, submission=_submission())
    assert isinstance(request, AnalysisWorkflowRequest)


def test_get_metadata_scalar_only() -> None:
    metadata = WorkflowUiRequestBuilder().get_metadata()
    assert metadata["builds_analysis_workflow_request"] is True
    assert metadata["infers_constraints"] is False
    assert metadata["infers_controllability"] is False
    assert metadata["infers_metric_direction"] is False
    assert metadata["stores_uploaded_file"] is False
    assert metadata["stores_raw_dataframe"] is False
    assert metadata["performs_modeling"] is False
    for key, value in metadata.items():
        assert isinstance(key, str)
        assert value is None or isinstance(value, (str, int, float, bool))
    assert copy.deepcopy(metadata) == metadata


# ---------------------------------------------------------------------------
# ANOMALY_ONLY analysis mode (Step 11B.5)
# ---------------------------------------------------------------------------


def _anomaly_submission(**overrides: Any) -> WorkflowUiSubmission:
    payload: dict[str, Any] = {
        "analysis_mode": AnalysisExecutionMode.ANOMALY_ONLY,
        "feature_columns": ["pressure", "temperature"],
        "timestamp_column": "timestamp",
        "identifier_columns": ["lot_id"],
        "excluded_columns": ["notes"],
        "max_simultaneous_changes": 2,
        "operating_point_selection": (
            OperatingPointSelectionMode.TOP_UNSUPERVISED_ANOMALY
        ),
        "metadata": {"caller": "unit"},
    }
    payload.update(overrides)
    return WorkflowUiSubmission(**payload)


def test_builder_passes_analysis_mode_anomaly_only(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "sample.csv")
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=_anomaly_submission(),
    )
    assert request.analysis_mode is AnalysisExecutionMode.ANOMALY_ONLY
    assert request.feature_columns == ["pressure", "temperature"]


def test_builder_forces_supervised_fields_none_for_anomaly_only(
    tmp_path: Path,
) -> None:
    csv_path = _write_csv(tmp_path / "sample.csv")
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=_anomaly_submission(),
    )
    assert request.target_column is None
    assert request.model_performance_policy is None
    assert request.requested_task is None
    assert request.objective is None
    assert request.quality_direction is None
    assert request.quality_target is None
    assert request.request_constraints == []
    assert request.user_confirmed_controllable_variables == []
    assert request.user_verified_variables == []


def test_builder_metadata_records_analysis_mode_anomaly_only(
    tmp_path: Path,
) -> None:
    csv_path = _write_csv(tmp_path / "sample.csv")
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=_anomaly_submission(),
    )
    assert request.metadata["analysis_mode"] == "ANOMALY_ONLY"


def test_builder_preserves_identifier_and_excluded_columns_for_anomaly_only(
    tmp_path: Path,
) -> None:
    csv_path = _write_csv(tmp_path / "sample.csv")
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=_anomaly_submission(),
    )
    assert request.identifier_columns == ["lot_id"]
    assert request.excluded_columns == ["notes"]
    assert request.timestamp_column == "timestamp"


def test_builder_supervised_analysis_mode_default_regression(
    tmp_path: Path,
) -> None:
    # Guards against ANOMALY_ONLY additions affecting the default SUPERVISED
    # request-building path exercised throughout the rest of this module.
    csv_path = _write_csv(tmp_path / "sample.csv")
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=_submission(),
    )
    assert request.analysis_mode is AnalysisExecutionMode.SUPERVISED
    assert request.metadata["analysis_mode"] == "SUPERVISED"
    assert request.target_column == "quality"
    assert request.model_performance_policy is not None


def test_builder_passes_cohort_filter_for_anomaly_only(tmp_path: Path) -> None:
    from process_intelligence.workflow import NumericCohortFilter

    csv_path = _write_csv(tmp_path / "sample.csv")
    cohort = NumericCohortFilter(
        column_name="pressure",
        lower_bound=10.0,
        upper_bound=50.0,
        exclude_filter_column_from_features=True,
    )
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=_anomaly_submission(cohort_filter=cohort),
    )
    assert request.cohort_filter is not None
    assert request.cohort_filter.column_name == "pressure"
    assert request.cohort_filter.lower_bound == pytest.approx(10.0)
    assert request.cohort_filter.upper_bound == pytest.approx(50.0)


def test_builder_forces_cohort_filter_none_for_supervised(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "sample.csv")
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=_submission(),
    )
    assert request.cohort_filter is None

def test_builder_anomaly_recommendation_disabled_default(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "sample.csv")
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=_anomaly_submission(),
    )
    assert request.anomaly_recommendation.enabled is False
    assert request.objective is None
    assert request.request_constraints == []


def test_builder_anomaly_recommendation_enabled_fixed_objective(tmp_path: Path) -> None:
    from process_intelligence.core.enums import ColumnRole
    from process_intelligence.ui.schemas import UiVariableConstraintInput

    csv_path = _write_csv(tmp_path / "sample.csv")
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=_anomaly_submission(
            anomaly_recommendation_enabled=True,
            objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
            column_role_overrides={"temperature": ColumnRole.CONTROLLABLE_PROCESS},
            constraints=[
                UiVariableConstraintInput(
                    variable="temperature",
                    minimum=60.0,
                    maximum=160.0,
                )
            ],
            user_confirmed_controllable_variables=["temperature"],
            user_verified_variables=["temperature"],
            max_simultaneous_changes=1,
        ),
    )
    assert request.anomaly_recommendation.enabled is True
    assert request.objective is RecommendationObjective.REDUCE_ANOMALY_SCORE
    assert request.target_column is None
    assert request.requested_task is None
    assert request.model_performance_policy is None
    assert request.quality_direction is None
    assert [item.variable for item in request.request_constraints] == ["temperature"]
    assert request.user_confirmed_controllable_variables == ["temperature"]
    assert request.user_verified_variables == ["temperature"]
    assert request.max_simultaneous_changes == 1
    assert request.column_role_overrides["temperature"] is ColumnRole.CONTROLLABLE_PROCESS


def test_builder_anomaly_recommendation_excludes_filter_column_candidates(
    tmp_path: Path,
) -> None:
    from process_intelligence.core.enums import ColumnRole
    from process_intelligence.ui.schemas import UiVariableConstraintInput
    from process_intelligence.workflow import NumericCohortFilter

    csv_path = _write_csv(tmp_path / "sample.csv")
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=_anomaly_submission(
            anomaly_recommendation_enabled=True,
            objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
            cohort_filter=NumericCohortFilter(
                column_name="pressure",
                lower_bound=10.0,
                upper_bound=50.0,
            ),
            column_role_overrides={
                "temperature": ColumnRole.CONTROLLABLE_PROCESS,
                "pressure": ColumnRole.CONTROLLABLE_PROCESS,
            },
            constraints=[
                UiVariableConstraintInput(
                    variable="temperature",
                    minimum=60.0,
                    maximum=160.0,
                ),
                UiVariableConstraintInput(
                    variable="pressure",
                    minimum=10.0,
                    maximum=50.0,
                ),
            ],
            user_confirmed_controllable_variables=["temperature", "pressure"],
            user_verified_variables=["temperature", "pressure"],
        ),
    )
    assert [item.variable for item in request.request_constraints] == ["temperature"]
    assert request.user_confirmed_controllable_variables == ["temperature"]
    assert request.user_verified_variables == ["temperature"]
    assert "pressure" not in request.column_role_overrides


def test_builder_supervised_does_not_submit_anomaly_recommendation_flag(
    tmp_path: Path,
) -> None:
    csv_path = _write_csv(tmp_path / "sample.csv")
    request = WorkflowUiRequestBuilder().build(
        csv_path=csv_path,
        submission=_submission(),
    )
    assert request.anomaly_recommendation.enabled is False
