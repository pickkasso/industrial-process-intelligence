"""Unit tests for analysis workflow enums and Pydantic schemas (Step 10C)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from process_intelligence.core.enums import AnalysisTask, ColumnRole
from process_intelligence.core.schemas import VariableConstraint
from process_intelligence.evaluation import (
    MetricAcceptanceDirection,
    MetricAcceptanceResult,
    MetricAcceptanceRule,
    ModelPerformanceAcceptancePolicy,
    ModelPerformanceAcceptanceReport,
    ModelPerformanceAcceptanceStatus,
)
from process_intelligence.recommendation import (
    QualityOptimizationDirection,
    RecommendationChange,
    RecommendationObjective,
    RecommendationReasonCode,
    RecommendationResult,
    RecommendationSafetyDecision,
    RecommendationSafetyStatus,
    RecommendationStatus,
    VariableEligibilityAssessment,
)
from process_intelligence.recommendation.schemas import DEFAULT_RECOMMENDATION_DISCLAIMER
from process_intelligence.workflow import (
    AnalysisWorkflowOutcome,
    AnalysisWorkflowPolicy,
    AnalysisWorkflowReport,
    AnalysisWorkflowRequest,
    AnalysisWorkflowStage,
    AnalysisWorkflowStageRecord,
    AnalysisWorkflowStatus,
    IndustrialProcessAnalysisWorkflow,
    OperatingPointSelectionMode,
)

_UTC_START = datetime(2026, 7, 21, 10, 0, tzinfo=UTC)
_UTC_END = datetime(2026, 7, 21, 10, 5, tzinfo=UTC)

_CANONICAL_STAGE_NAMES = [
    "LOAD",
    "PROFILE",
    "VALIDATE",
    "QUALITY_SCORE",
    "SORT",
    "PREPROCESS",
    "INDUSTRY_ROUTING",
    "TASK_ROUTING",
    "ROLE_MAPPING",
    "SPLIT",
    "LEAKAGE_CHECK",
    "SUPERVISED_SCREENING",
    "SUPERVISED_FINAL_EVALUATION",
    "ANOMALY_SCREENING",
    "ANOMALY_FINAL_EVALUATION",
    "RESIDUAL_CALIBRATION",
    "RESIDUAL_FINAL_EVALUATION",
    "ANOMALY_EVENT_SELECTION",
    "DIAGNOSIS",
    "RECOMMENDATION",
]


# --- recommendation helpers (used to build final_recommendation) ---


def _constraint(
    variable: str = "pressure",
    *,
    minimum: float | None = 0.0,
    maximum: float | None = 100.0,
) -> VariableConstraint:
    return VariableConstraint(
        variable=variable,
        adjustable=True,
        minimum=minimum,
        maximum=maximum,
        fixed=False,
    )


def _rec_assessment(
    variable: str = "pressure",
    *,
    eligible: bool = True,
    reason_codes: list[RecommendationReasonCode] | None = None,
    factor_rank: int = 1,
    **overrides: Any,
) -> VariableEligibilityAssessment:
    payload: dict[str, Any] = {
        "variable": variable,
        "factor_rank": factor_rank,
        "factor_confidence": 0.7,
        "factor_role": str(ColumnRole.CONTROLLABLE_PROCESS),
        "factor_controllable": True,
        "factor_needs_verification": False,
        "current_value": 50.0,
        "constraint_present": True,
        "user_confirmed_controllable": True,
        "user_verified": False,
        "eligible": eligible,
        "reason_codes": list(reason_codes or []),
        "warnings": [],
    }
    payload.update(overrides)
    return VariableEligibilityAssessment(**payload)


def _rec_decision(**overrides: Any) -> RecommendationSafetyDecision:
    payload: dict[str, Any] = {
        "status": RecommendationSafetyStatus.APPROVED,
        "objective": RecommendationObjective.REDUCE_ANOMALY_SCORE,
        "eligible_variables": ["pressure"],
        "blocked_variables": [],
        "variable_assessments": [_rec_assessment("pressure", eligible=True)],
        "global_reason_codes": [],
        "messages": ["Safety checks passed."],
        "disclaimer": DEFAULT_RECOMMENDATION_DISCLAIMER,
        "evaluated_at": datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
        "metadata": {"eligible_variable_count": 1},
    }
    payload.update(overrides)
    return RecommendationSafetyDecision(**payload)


def _rec_refused_decision() -> RecommendationSafetyDecision:
    return _rec_decision(
        status=RecommendationSafetyStatus.REFUSED,
        eligible_variables=[],
        blocked_variables=["pressure"],
        variable_assessments=[
            _rec_assessment(
                "pressure",
                eligible=False,
                reason_codes=[RecommendationReasonCode.CURRENT_VALUE_MISSING],
                current_value=None,
            )
        ],
        global_reason_codes=[RecommendationReasonCode.NO_ELIGIBLE_VARIABLES],
        messages=["Refused."],
    )


def _rec_change(**overrides: Any) -> RecommendationChange:
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
    return RecommendationChange(**payload)


def _generated_result() -> RecommendationResult:
    return RecommendationResult(
        status=RecommendationStatus.GENERATED,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        safety_decision=_rec_decision(),
        changes=[_rec_change()],
        confidence=0.5,
        extrapolation_flag=False,
        uncertainty_available=False,
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
        warnings=["Verify operationally before applying."],
    )


def _ready_result() -> RecommendationResult:
    return RecommendationResult(
        status=RecommendationStatus.READY_FOR_OPTIMIZATION,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        safety_decision=_rec_decision(),
        changes=[],
        confidence=0.5,
        extrapolation_flag=False,
        uncertainty_available=False,
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
    )


def _refused_result() -> RecommendationResult:
    return RecommendationResult(
        status=RecommendationStatus.REFUSED,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        safety_decision=_rec_refused_decision(),
        changes=[],
        confidence=0.0,
        extrapolation_flag=False,
        uncertainty_available=False,
        disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
        generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
    )


# --- request helpers ---


def _performance_policy(**overrides: Any) -> ModelPerformanceAcceptancePolicy:
    payload: dict[str, Any] = {
        "rules": [
            MetricAcceptanceRule(
                metric_name="rmse",
                direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
                threshold=10.0,
            ),
            MetricAcceptanceRule(
                metric_name="mae",
                direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
                threshold=10.0,
            ),
        ],
    }
    payload.update(overrides)
    return ModelPerformanceAcceptancePolicy(**payload)


def _performance_assessment(
    **overrides: Any,
) -> ModelPerformanceAcceptanceReport:
    payload: dict[str, Any] = {
        "status": ModelPerformanceAcceptanceStatus.ACCEPTABLE,
        "task": AnalysisTask.REGRESSION,
        "evaluation_available": True,
        "independent_test_evaluation": True,
        "test_row_count": 20,
        "metric_results": [
            MetricAcceptanceResult(
                metric_name="rmse",
                observed_value=1.2,
                threshold=10.0,
                direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
                required=True,
                available=True,
                passed=True,
                message="Metric rmse=1.2 meets LOWER_IS_BETTER threshold 10.0.",
            ),
            MetricAcceptanceResult(
                metric_name="mae",
                observed_value=0.9,
                threshold=10.0,
                direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
                required=True,
                available=True,
                passed=True,
                message="Metric mae=0.9 meets LOWER_IS_BETTER threshold 10.0.",
            ),
        ],
        "required_rule_count": 2,
        "passed_required_rule_count": 2,
        "failed_required_rule_count": 0,
        "unavailable_required_rule_count": 0,
        "assessed_at": datetime(2026, 7, 21, 10, 2, tzinfo=UTC),
        "warnings": [],
        "metadata": {"estimator_key": "ridge"},
    }
    payload.update(overrides)
    return ModelPerformanceAcceptanceReport(**payload)


def _request_kwargs(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "csv_path": Path("data.csv"),
        "target_column": "quality",
        "feature_columns": ["pressure", "temperature", "flow"],
        "model_performance_policy": _performance_policy(),
        "objective": RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        "quality_direction": QualityOptimizationDirection.MAXIMIZE,
    }
    payload.update(overrides)
    return payload


def _request(**overrides: Any) -> AnalysisWorkflowRequest:
    return AnalysisWorkflowRequest(**_request_kwargs(**overrides))


# --- stage record helpers ---


def _stage_record(
    stage: AnalysisWorkflowStage,
    *,
    executed: bool = True,
    succeeded: bool = True,
    structured_refusal: bool = False,
    row_count: int | None = None,
    message: str | None = None,
    warnings: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AnalysisWorkflowStageRecord:
    return AnalysisWorkflowStageRecord(
        stage=stage,
        executed=executed,
        succeeded=succeeded,
        structured_refusal=structured_refusal,
        row_count=row_count,
        message=message or f"{stage.value} stage completed.",
        warnings=list(warnings or []),
        metadata=dict(metadata or {}),
    )


def _prefix_records(
    terminal: AnalysisWorkflowStage,
    *,
    refuse_last: bool = False,
) -> list[AnalysisWorkflowStageRecord]:
    stages = list(AnalysisWorkflowStage)
    terminal_index = stages.index(terminal)
    records: list[AnalysisWorkflowStageRecord] = []
    for index, stage in enumerate(stages[: terminal_index + 1]):
        is_last = index == terminal_index
        if is_last and refuse_last:
            records.append(
                _stage_record(
                    stage,
                    executed=True,
                    succeeded=False,
                    structured_refusal=True,
                )
            )
        else:
            records.append(_stage_record(stage, executed=True, succeeded=True))
    return records


def _base_report_kwargs(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": AnalysisWorkflowStatus.COMPLETED,
        "terminal_stage": AnalysisWorkflowStage.RECOMMENDATION,
        "stage_records": _prefix_records(AnalysisWorkflowStage.RECOMMENDATION),
        "model_performance_assessment": _performance_assessment(),
        "final_recommendation": _generated_result(),
        "selected_industry": "semiconductor",
        "selected_task": AnalysisTask.REGRESSION,
        "selected_supervised_model_key": "ridge",
        "selected_anomaly_model_key": "isolation_forest",
        "selected_operating_row_id": 7,
        "anomaly_event_count": 2,
        "diagnosis_factor_count": 3,
        "raw_row_count": 100,
        "processed_row_count": 100,
        "train_row_count": 60,
        "validation_row_count": 20,
        "test_row_count": 20,
        "started_at": _UTC_START,
        "completed_at": _UTC_END,
        "total_seconds": 300.0,
        "warnings": ["Recommendations are decision support, not proven causes."],
        "metadata": {"seed": 1},
    }
    payload.update(overrides)
    return payload


def _completed_report(**overrides: Any) -> AnalysisWorkflowReport:
    return AnalysisWorkflowReport(**_base_report_kwargs(**overrides))


def _partial_ready_report(**overrides: Any) -> AnalysisWorkflowReport:
    payload = _base_report_kwargs(
        status=AnalysisWorkflowStatus.PARTIAL,
        terminal_stage=AnalysisWorkflowStage.RECOMMENDATION,
        stage_records=_prefix_records(AnalysisWorkflowStage.RECOMMENDATION),
        final_recommendation=_ready_result(),
    )
    payload.update(overrides)
    return AnalysisWorkflowReport(**payload)


def _partial_early_report(**overrides: Any) -> AnalysisWorkflowReport:
    payload = _base_report_kwargs(
        status=AnalysisWorkflowStatus.PARTIAL,
        terminal_stage=AnalysisWorkflowStage.DIAGNOSIS,
        stage_records=_prefix_records(AnalysisWorkflowStage.DIAGNOSIS),
        model_performance_assessment=_performance_assessment(),
        final_recommendation=None,
    )
    payload.update(overrides)
    return AnalysisWorkflowReport(**payload)


def _refused_report(**overrides: Any) -> AnalysisWorkflowReport:
    payload = _base_report_kwargs(
        status=AnalysisWorkflowStatus.REFUSED,
        terminal_stage=AnalysisWorkflowStage.VALIDATE,
        stage_records=_prefix_records(
            AnalysisWorkflowStage.VALIDATE, refuse_last=True
        ),
        model_performance_assessment=None,
        final_recommendation=None,
        selected_industry=None,
        selected_task=None,
        selected_supervised_model_key=None,
        selected_anomaly_model_key=None,
        selected_operating_row_id=None,
        anomaly_event_count=0,
        diagnosis_factor_count=0,
        train_row_count=0,
        validation_row_count=0,
        test_row_count=0,
    )
    payload.update(overrides)
    return AnalysisWorkflowReport(**payload)


# --- 1-4: enums ---


def test_stage_enum_values_and_canonical_order() -> None:
    assert [member.name for member in AnalysisWorkflowStage] == _CANONICAL_STAGE_NAMES
    assert [member.value for member in AnalysisWorkflowStage] == _CANONICAL_STAGE_NAMES
    assert list(AnalysisWorkflowStage) == [
        AnalysisWorkflowStage(name) for name in _CANONICAL_STAGE_NAMES
    ]
    assert len(AnalysisWorkflowStage) == 20
    assert AnalysisWorkflowStage.__doc__


def test_status_enum_values() -> None:
    assert {member.name: member.value for member in AnalysisWorkflowStatus} == {
        "COMPLETED": "COMPLETED",
        "PARTIAL": "PARTIAL",
        "REFUSED": "REFUSED",
    }
    assert AnalysisWorkflowStatus.__doc__


def test_operating_point_selection_mode_values() -> None:
    assert {
        member.name: member.value for member in OperatingPointSelectionMode
    } == {
        "EXPLICIT_ROW_ID": "EXPLICIT_ROW_ID",
        "TOP_RESIDUAL_ANOMALY": "TOP_RESIDUAL_ANOMALY",
        "TOP_UNSUPERVISED_ANOMALY": "TOP_UNSUPERVISED_ANOMALY",
        "LATEST_ROW": "LATEST_ROW",
    }
    assert OperatingPointSelectionMode.__doc__


def test_enums_have_no_extra_members() -> None:
    assert len(AnalysisWorkflowStage) == 20
    assert len(AnalysisWorkflowStatus) == 3
    assert len(OperatingPointSelectionMode) == 4
    with pytest.raises(ValueError):
        AnalysisWorkflowStage("NOT_A_STAGE")
    with pytest.raises(ValueError):
        AnalysisWorkflowStatus("NOT_A_STATUS")
    with pytest.raises(ValueError):
        OperatingPointSelectionMode("NOT_A_MODE")


# --- 5-11: policy ---


def test_policy_defaults() -> None:
    policy = AnalysisWorkflowPolicy()
    assert policy.stop_on_validation_blocker is True
    assert policy.stop_on_leakage_blocker is True
    assert policy.require_semiconductor_industry is False
    assert policy.require_regression_task is True
    assert policy.require_residual_diagnosis is True
    assert policy.allow_partial_diagnosis_ensemble is False
    assert policy.maximum_anomaly_events == 5
    assert policy.minimum_anomaly_events == 1
    assert policy.preserve_stage_outputs is True
    assert policy.include_stage_warnings is True
    assert policy.maximum_aggregated_warnings == 200


@pytest.mark.parametrize("bad_value", [0, 1, "true"])
def test_policy_bool_strict(bad_value: object) -> None:
    with pytest.raises(ValidationError):
        AnalysisWorkflowPolicy(stop_on_validation_blocker=bad_value)  # type: ignore[arg-type]


def test_policy_event_count_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        AnalysisWorkflowPolicy(minimum_anomaly_events=0)
    with pytest.raises(ValidationError):
        AnalysisWorkflowPolicy(maximum_anomaly_events=0)


def test_policy_event_count_bool_rejected() -> None:
    with pytest.raises(ValidationError):
        AnalysisWorkflowPolicy(minimum_anomaly_events=True)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        AnalysisWorkflowPolicy(maximum_anomaly_events=True)  # type: ignore[arg-type]


def test_policy_minimum_greater_than_maximum_rejected() -> None:
    with pytest.raises(ValidationError):
        AnalysisWorkflowPolicy(minimum_anomaly_events=5, maximum_anomaly_events=2)


def test_policy_warning_maximum_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        AnalysisWorkflowPolicy(maximum_aggregated_warnings=0)


def test_policy_round_trip() -> None:
    policy = AnalysisWorkflowPolicy(
        maximum_anomaly_events=8,
        minimum_anomaly_events=2,
        require_semiconductor_industry=True,
    )
    restored = AnalysisWorkflowPolicy.model_validate(policy.model_dump())
    assert restored == policy


# --- 12-31: request ---


def test_request_valid() -> None:
    request = _request()
    assert request.csv_path == Path("data.csv")
    assert request.target_column == "quality"
    assert request.feature_columns == ["pressure", "temperature", "flow"]
    assert request.objective is RecommendationObjective.IMPROVE_PREDICTED_QUALITY
    assert request.quality_direction is QualityOptimizationDirection.MAXIMIZE
    assert (
        request.operating_point_selection
        is OperatingPointSelectionMode.TOP_RESIDUAL_ANOMALY
    )
    assert isinstance(
        request.model_performance_policy, ModelPerformanceAcceptancePolicy
    )
    assert len(request.model_performance_policy.rules) >= 1


def test_request_requires_model_performance_policy() -> None:
    payload = _request_kwargs()
    del payload["model_performance_policy"]
    with pytest.raises(ValidationError):
        AnalysisWorkflowRequest(**payload)


def test_request_model_performance_policy_copy_independence() -> None:
    policy = _performance_policy()
    request = _request(model_performance_policy=policy)
    policy.rules[0].threshold = 0.0001
    assert request.model_performance_policy.rules[0].threshold == pytest.approx(10.0)


def test_report_requires_assessment_after_supervised_final() -> None:
    with pytest.raises(ValidationError):
        _completed_report(model_performance_assessment=None)


def test_report_rejects_assessment_before_supervised_final() -> None:
    with pytest.raises(ValidationError):
        _refused_report(model_performance_assessment=_performance_assessment())


def test_request_non_csv_extension_rejected() -> None:
    with pytest.raises(ValidationError):
        _request(csv_path=Path("data.txt"))
    with pytest.raises(ValidationError):
        _request(csv_path="data.parquet")


def test_request_empty_target_rejected() -> None:
    with pytest.raises(ValidationError):
        _request(target_column="")
    with pytest.raises(ValidationError):
        _request(target_column="   ")


def test_request_empty_feature_list_rejected() -> None:
    with pytest.raises(ValidationError):
        _request(feature_columns=[])


def test_request_duplicate_features_rejected() -> None:
    with pytest.raises(ValidationError):
        _request(feature_columns=["pressure", "pressure"])


def test_request_target_in_features_rejected() -> None:
    with pytest.raises(ValidationError):
        _request(target_column="pressure")


def test_request_identifier_excluded_overlap_rejected() -> None:
    with pytest.raises(ValidationError):
        _request(identifier_columns=["shared"], excluded_columns=["shared"])


def test_request_timestamp_conflict_with_features_rejected() -> None:
    with pytest.raises(ValidationError):
        _request(timestamp_column="pressure")


def test_request_invalid_override_type_rejected() -> None:
    with pytest.raises(ValidationError):
        _request(column_role_overrides={"pressure": "NOT_A_ROLE"})
    with pytest.raises(ValidationError):
        _request(column_role_overrides={"pressure": 123})  # type: ignore[dict-item]


def test_request_quality_direction_requirements() -> None:
    with pytest.raises(ValidationError):
        _request(
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
            quality_direction=None,
        )
    ok = _request(
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        quality_direction=None,
    )
    assert ok.quality_direction is None
    assert ok.objective is RecommendationObjective.REDUCE_ANOMALY_SCORE


def test_request_quality_target_relationship() -> None:
    with pytest.raises(ValidationError):
        _request(
            quality_direction=QualityOptimizationDirection.TARGET,
            quality_target=None,
        )
    with pytest.raises(ValidationError):
        _request(
            quality_direction=QualityOptimizationDirection.MAXIMIZE,
            quality_target=5.0,
        )
    ok = _request(
        quality_direction=QualityOptimizationDirection.TARGET,
        quality_target=5.0,
    )
    assert ok.quality_target == pytest.approx(5.0)


def test_request_duplicate_constraint_variable_rejected() -> None:
    with pytest.raises(ValidationError):
        _request(
            request_constraints=[_constraint("pressure"), _constraint("pressure")]
        )


def test_request_constraint_variable_not_in_features_rejected() -> None:
    with pytest.raises(ValidationError):
        _request(request_constraints=[_constraint("missing_var")])


def test_request_confirmation_variable_not_in_features_rejected() -> None:
    with pytest.raises(ValidationError):
        _request(user_confirmed_controllable_variables=["missing_var"])


def test_request_duplicate_verification_rejected() -> None:
    with pytest.raises(ValidationError):
        _request(user_verified_variables=["pressure", "pressure"])


def test_request_max_simultaneous_changes_zero_and_bool_rejected() -> None:
    with pytest.raises(ValidationError):
        _request(max_simultaneous_changes=0)
    with pytest.raises(ValidationError):
        _request(max_simultaneous_changes=True)  # type: ignore[arg-type]


def test_request_explicit_row_id_mode_rules() -> None:
    with pytest.raises(ValidationError):
        _request(
            operating_point_selection=OperatingPointSelectionMode.EXPLICIT_ROW_ID,
            explicit_operating_row_id=None,
        )
    with pytest.raises(ValidationError):
        _request(
            operating_point_selection=(
                OperatingPointSelectionMode.TOP_RESIDUAL_ANOMALY
            ),
            explicit_operating_row_id=5,
        )
    ok = _request(
        operating_point_selection=OperatingPointSelectionMode.EXPLICIT_ROW_ID,
        explicit_operating_row_id=5,
    )
    assert ok.explicit_operating_row_id == 5


def test_request_non_scalar_metadata_rejected() -> None:
    with pytest.raises(ValidationError):
        _request(metadata={"bad": [1, 2]})  # type: ignore[dict-item]
    with pytest.raises(ValidationError):
        _request(metadata={"bad": {"nested": 1}})  # type: ignore[dict-item]


def test_request_input_list_copy_independence() -> None:
    features = ["pressure", "temperature", "flow"]
    constraints = [_constraint("pressure")]
    request = _request(
        feature_columns=features,
        request_constraints=constraints,
    )
    features.append("gas_flow")
    constraints.append(_constraint("temperature"))
    assert request.feature_columns == ["pressure", "temperature", "flow"]
    assert [item.variable for item in request.request_constraints] == ["pressure"]


def test_request_round_trip() -> None:
    request = _request(
        identifier_columns=["lot_id"],
        request_constraints=[_constraint("pressure")],
        metadata={"seed": 7, "note": "run"},
    )
    restored = AnalysisWorkflowRequest.model_validate(request.model_dump())
    assert restored.target_column == request.target_column
    assert restored.feature_columns == request.feature_columns
    assert restored.objective is request.objective
    assert restored.quality_direction is request.quality_direction


# --- 32-40: stage record ---


def test_stage_record_success() -> None:
    record = _stage_record(
        AnalysisWorkflowStage.LOAD,
        executed=True,
        succeeded=True,
        structured_refusal=False,
        row_count=100,
    )
    assert record.executed is True
    assert record.succeeded is True
    assert record.structured_refusal is False
    assert record.row_count == 100


def test_stage_record_refusal() -> None:
    record = _stage_record(
        AnalysisWorkflowStage.VALIDATE,
        executed=True,
        succeeded=False,
        structured_refusal=True,
    )
    assert record.executed is True
    assert record.succeeded is False
    assert record.structured_refusal is True


def test_stage_record_not_executed_relationships() -> None:
    with pytest.raises(ValidationError):
        _stage_record(
            AnalysisWorkflowStage.SPLIT,
            executed=False,
            succeeded=True,
            structured_refusal=False,
        )
    with pytest.raises(ValidationError):
        _stage_record(
            AnalysisWorkflowStage.SPLIT,
            executed=False,
            succeeded=False,
            structured_refusal=True,
        )
    ok = _stage_record(
        AnalysisWorkflowStage.SPLIT,
        executed=False,
        succeeded=False,
        structured_refusal=False,
    )
    assert ok.executed is False
    assert ok.succeeded is False
    assert ok.structured_refusal is False


def test_stage_record_succeeded_and_refusal_both_true_rejected() -> None:
    with pytest.raises(ValidationError):
        _stage_record(
            AnalysisWorkflowStage.LOAD,
            executed=True,
            succeeded=True,
            structured_refusal=True,
        )


def test_stage_record_invalid_row_count_rejected() -> None:
    with pytest.raises(ValidationError):
        _stage_record(AnalysisWorkflowStage.LOAD, row_count=-1)
    with pytest.raises(ValidationError):
        _stage_record(AnalysisWorkflowStage.LOAD, row_count=True)  # type: ignore[arg-type]


def test_stage_record_empty_message_rejected() -> None:
    with pytest.raises(ValidationError):
        _stage_record(AnalysisWorkflowStage.LOAD, message="   ")


def test_stage_record_duplicate_warnings_rejected() -> None:
    with pytest.raises(ValidationError):
        _stage_record(AnalysisWorkflowStage.LOAD, warnings=["dup", "dup"])


def test_stage_record_non_scalar_metadata_rejected() -> None:
    with pytest.raises(ValidationError):
        _stage_record(AnalysisWorkflowStage.LOAD, metadata={"bad": [1, 2]})


def test_stage_record_round_trip() -> None:
    record = _stage_record(
        AnalysisWorkflowStage.LOAD,
        row_count=50,
        warnings=["note"],
        metadata={"seed": 1, "flag": True},
    )
    restored = AnalysisWorkflowStageRecord.model_validate(record.model_dump())
    assert restored == record


# --- 41-55: report ---


def test_report_completed_valid() -> None:
    report = _completed_report()
    assert report.status is AnalysisWorkflowStatus.COMPLETED
    assert report.terminal_stage is AnalysisWorkflowStage.RECOMMENDATION
    assert report.final_recommendation is not None
    assert report.final_recommendation.status is RecommendationStatus.GENERATED
    assert len(report.stage_records) == 20


def test_report_partial_valid() -> None:
    ready = _partial_ready_report()
    assert ready.status is AnalysisWorkflowStatus.PARTIAL
    assert ready.final_recommendation is not None
    assert (
        ready.final_recommendation.status
        is RecommendationStatus.READY_FOR_OPTIMIZATION
    )

    early = _partial_early_report()
    assert early.status is AnalysisWorkflowStatus.PARTIAL
    assert early.terminal_stage is AnalysisWorkflowStage.DIAGNOSIS
    assert early.final_recommendation is None


def test_report_refused_valid() -> None:
    report = _refused_report()
    assert report.status is AnalysisWorkflowStatus.REFUSED
    assert report.terminal_stage is AnalysisWorkflowStage.VALIDATE
    assert report.stage_records[-1].structured_refusal is True
    assert report.final_recommendation is None


def test_report_duplicate_stages_rejected() -> None:
    records = _prefix_records(AnalysisWorkflowStage.PROFILE)
    records.append(_stage_record(AnalysisWorkflowStage.LOAD))
    with pytest.raises(ValidationError):
        _completed_report(
            status=AnalysisWorkflowStatus.PARTIAL,
            terminal_stage=AnalysisWorkflowStage.PROFILE,
            stage_records=records,
            final_recommendation=None,
        )


def test_report_out_of_order_stages_rejected() -> None:
    records = [
        _stage_record(AnalysisWorkflowStage.LOAD),
        _stage_record(AnalysisWorkflowStage.VALIDATE),
        _stage_record(AnalysisWorkflowStage.PROFILE),
    ]
    with pytest.raises(ValidationError):
        _completed_report(
            status=AnalysisWorkflowStatus.PARTIAL,
            terminal_stage=AnalysisWorkflowStage.PROFILE,
            stage_records=records,
            final_recommendation=None,
        )


def test_report_executed_prefix_violation_rejected() -> None:
    records = [
        _stage_record(AnalysisWorkflowStage.LOAD, executed=True),
        _stage_record(
            AnalysisWorkflowStage.PROFILE,
            executed=False,
            succeeded=False,
            structured_refusal=False,
        ),
        _stage_record(AnalysisWorkflowStage.VALIDATE, executed=True),
    ]
    with pytest.raises(ValidationError):
        _completed_report(
            status=AnalysisWorkflowStatus.PARTIAL,
            terminal_stage=AnalysisWorkflowStage.VALIDATE,
            stage_records=records,
            final_recommendation=None,
        )


def test_report_terminal_stage_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        _completed_report(terminal_stage=AnalysisWorkflowStage.DIAGNOSIS)


def test_report_invalid_counts_rejected() -> None:
    with pytest.raises(ValidationError):
        _completed_report(raw_row_count=-1)
    with pytest.raises(ValidationError):
        _completed_report(anomaly_event_count=True)  # type: ignore[arg-type]


def test_report_split_count_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        _completed_report(
            train_row_count=60,
            validation_row_count=20,
            test_row_count=10,
        )


def test_report_status_relationship_violations_rejected() -> None:
    # COMPLETED must terminate at RECOMMENDATION.
    with pytest.raises(ValidationError):
        _completed_report(
            terminal_stage=AnalysisWorkflowStage.DIAGNOSIS,
            stage_records=_prefix_records(AnalysisWorkflowStage.DIAGNOSIS),
        )
    # COMPLETED requires a final_recommendation.
    with pytest.raises(ValidationError):
        _completed_report(final_recommendation=None)
    # COMPLETED requires GENERATED recommendation status.
    with pytest.raises(ValidationError):
        _completed_report(final_recommendation=_ready_result())
    # PARTIAL with a recommendation must be READY_FOR_OPTIMIZATION.
    with pytest.raises(ValidationError):
        _partial_ready_report(final_recommendation=_generated_result())
    # PARTIAL without recommendation cannot terminate at RECOMMENDATION.
    with pytest.raises(ValidationError):
        _partial_ready_report(final_recommendation=None)
    # REFUSED without a structured refusal or REFUSED recommendation is invalid.
    with pytest.raises(ValidationError):
        _refused_report(
            stage_records=_prefix_records(AnalysisWorkflowStage.VALIDATE),
        )


def test_report_completed_before_started_rejected() -> None:
    with pytest.raises(ValidationError):
        _completed_report(started_at=_UTC_END, completed_at=_UTC_START)


def test_report_naive_datetime_rejected() -> None:
    with pytest.raises(ValidationError):
        _completed_report(started_at=datetime(2026, 7, 21, 10, 0))
    with pytest.raises(ValidationError):
        _completed_report(completed_at=datetime(2026, 7, 21, 10, 5))


def test_report_duplicate_warnings_rejected() -> None:
    with pytest.raises(ValidationError):
        _completed_report(warnings=["dup", "dup"])


def test_report_non_scalar_metadata_rejected() -> None:
    with pytest.raises(ValidationError):
        _completed_report(metadata={"bad": {"nested": 1}})


def test_report_round_trip() -> None:
    report = _completed_report()
    restored = AnalysisWorkflowReport.model_validate(report.model_dump())
    assert restored.status is report.status
    assert restored.terminal_stage is report.terminal_stage
    assert len(restored.stage_records) == len(report.stage_records)
    assert restored.final_recommendation is not None
    assert (
        restored.final_recommendation.status is RecommendationStatus.GENERATED
    )


# --- 56-57: outcome ---


def test_outcome_is_frozen() -> None:
    assert dataclasses.is_dataclass(AnalysisWorkflowOutcome)
    assert AnalysisWorkflowOutcome.__dataclass_params__.frozen is True  # type: ignore[attr-defined]
    outcome = AnalysisWorkflowOutcome(report=_completed_report())
    assert outcome.report.status is AnalysisWorkflowStatus.COMPLETED
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.report = _completed_report()  # type: ignore[misc]


def test_outcome_uses_slots() -> None:
    assert AnalysisWorkflowOutcome.__slots__ == ("report",)
    outcome = AnalysisWorkflowOutcome(report=_completed_report())
    assert not hasattr(outcome, "__dict__")


# --- 58: public export ---


def test_public_package_exports() -> None:
    import process_intelligence.workflow as workflow

    assert workflow.AnalysisWorkflowStage is AnalysisWorkflowStage
    assert workflow.AnalysisWorkflowStatus is AnalysisWorkflowStatus
    assert workflow.OperatingPointSelectionMode is OperatingPointSelectionMode
    assert workflow.AnalysisWorkflowPolicy is AnalysisWorkflowPolicy
    assert workflow.AnalysisWorkflowRequest is AnalysisWorkflowRequest
    assert workflow.AnalysisWorkflowStageRecord is AnalysisWorkflowStageRecord
    assert workflow.AnalysisWorkflowReport is AnalysisWorkflowReport
    assert workflow.AnalysisWorkflowOutcome is AnalysisWorkflowOutcome
    assert workflow.IndustrialProcessAnalysisWorkflow is (
        IndustrialProcessAnalysisWorkflow
    )
    assert set(workflow.__all__) == {
        "AnalysisWorkflowOutcome",
        "AnalysisWorkflowPolicy",
        "AnalysisWorkflowReport",
        "AnalysisWorkflowRequest",
        "AnalysisWorkflowStage",
        "AnalysisWorkflowStageRecord",
        "AnalysisWorkflowStatus",
        "IndustrialProcessAnalysisWorkflow",
        "OperatingPointSelectionMode",
    }
