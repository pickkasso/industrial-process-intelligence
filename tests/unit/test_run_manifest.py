"""Unit tests for analysis-run manifest fingerprinting (Step 14A)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from process_intelligence.core.enums import AnalysisTask, ColumnRole
from process_intelligence.core.schemas import VariableConstraint
from process_intelligence.evaluation import (
    MetricAcceptanceDirection,
    MetricAcceptanceRule,
    ModelPerformanceAcceptancePolicy,
)
from process_intelligence.recommendation import (
    QualityOptimizationDirection,
    RecommendationObjective,
    RecommendationStatus,
)
from process_intelligence.workflow import (
    AnalysisExecutionMode,
    AnalysisWorkflowPolicy,
    AnalysisWorkflowRequest,
    AnalysisWorkflowStage,
    AnalysisWorkflowStageRecord,
    AnalysisWorkflowStatus,
    NumericCohortFilter,
    OperatingPointSelectionMode,
    analysis_run_manifest_to_json,
    build_analysis_run_manifest,
    compute_configuration_fingerprint,
    compute_run_signature,
    get_application_version,
)
from process_intelligence.workflow.run_manifest import (
    MANIFEST_SCHEMA_VERSION,
    assert_manifest_has_no_forbidden_objects,
    build_configuration_fingerprint_payload,
)


def _performance_policy() -> ModelPerformanceAcceptancePolicy:
    return ModelPerformanceAcceptancePolicy(
        rules=[
            MetricAcceptanceRule(
                metric_name="rmse",
                direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
                threshold=10.0,
            )
        ]
    )


def _constraint(variable: str = "pressure") -> VariableConstraint:
    return VariableConstraint(
        variable=variable,
        adjustable=True,
        minimum=10.0,
        maximum=70.0,
        fixed=False,
    )


def _request(**overrides: object) -> AnalysisWorkflowRequest:
    payload: dict[str, object] = {
        "csv_path": Path("data.csv"),
        "target_column": "quality",
        "feature_columns": ["pressure", "temperature", "flow"],
        "model_performance_policy": _performance_policy(),
        "objective": RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        "quality_direction": QualityOptimizationDirection.MAXIMIZE,
        "requested_task": AnalysisTask.REGRESSION,
        "timestamp_column": "timestamp",
        "identifier_columns": ["lot_id"],
        "excluded_columns": ["injected_anomaly"],
        "column_role_overrides": {
            "pressure": ColumnRole.CONTROLLABLE_PROCESS,
            "quality": ColumnRole.TARGET_QUALITY,
        },
        "request_constraints": [_constraint()],
        "user_confirmed_controllable_variables": ["pressure"],
        "user_verified_variables": ["pressure"],
        "operating_point_selection": OperatingPointSelectionMode.TOP_RESIDUAL_ANOMALY,
        "declared_target_minimum": 0.0,
        "declared_target_maximum": 100.0,
        "metadata": {"ui_source": "streamlit_mvp", "transient": True},
    }
    payload.update(overrides)
    return AnalysisWorkflowRequest(**payload)  # type: ignore[arg-type]


def _policy(**overrides: object) -> AnalysisWorkflowPolicy:
    return AnalysisWorkflowPolicy(**overrides)  # type: ignore[arg-type]


def _stage(
    stage: AnalysisWorkflowStage,
    *,
    executed: bool = True,
    succeeded: bool = True,
    structured_refusal: bool = False,
    message: str | None = None,
    metadata: dict[str, object] | None = None,
) -> AnalysisWorkflowStageRecord:
    return AnalysisWorkflowStageRecord(
        stage=stage,
        executed=executed,
        succeeded=succeeded,
        structured_refusal=structured_refusal,
        message=message or f"{stage.value} stage record.",
        warnings=[],
        metadata=metadata or {},
    )


def test_same_effective_request_policy_same_configuration_fingerprint() -> None:
    first = compute_configuration_fingerprint(_request(), _policy())
    second = compute_configuration_fingerprint(_request(), _policy())
    assert first == second
    assert len(first) == 64


def test_mapping_insertion_order_does_not_change_fingerprint() -> None:
    left = _request(
        column_role_overrides={
            "pressure": ColumnRole.CONTROLLABLE_PROCESS,
            "quality": ColumnRole.TARGET_QUALITY,
        }
    )
    right = _request(
        column_role_overrides={
            "quality": ColumnRole.TARGET_QUALITY,
            "pressure": ColumnRole.CONTROLLABLE_PROCESS,
        }
    )
    assert compute_configuration_fingerprint(left, _policy()) == (
        compute_configuration_fingerprint(right, _policy())
    )


def test_feature_order_change_alters_fingerprint() -> None:
    left = _request(feature_columns=["pressure", "temperature", "flow"])
    right = _request(feature_columns=["flow", "temperature", "pressure"])
    assert compute_configuration_fingerprint(left, _policy()) != (
        compute_configuration_fingerprint(right, _policy())
    )


def test_target_change_alters_fingerprint() -> None:
    left = compute_configuration_fingerprint(_request(target_column="quality"), _policy())
    right = compute_configuration_fingerprint(
        _request(target_column="defect_rate"),
        _policy(),
    )
    assert left != right


def test_constraint_change_alters_fingerprint() -> None:
    left = compute_configuration_fingerprint(_request(), _policy())
    right = compute_configuration_fingerprint(
        _request(request_constraints=[_constraint("temperature")]),
        _policy(),
    )
    assert left != right


def test_cohort_settings_change_alters_fingerprint() -> None:
    base = AnalysisWorkflowRequest(
        csv_path=Path("data.csv"),
        analysis_mode=AnalysisExecutionMode.ANOMALY_ONLY,
        feature_columns=["pressure", "temperature", "flow"],
        timestamp_column="timestamp",
        operating_point_selection=OperatingPointSelectionMode.TOP_UNSUPERVISED_ANOMALY,
    )
    left = compute_configuration_fingerprint(base, _policy())
    right = compute_configuration_fingerprint(
        base.model_copy(
            update={
                "cohort_filter": NumericCohortFilter(
                    column_name="pressure",
                    lower_bound=20.0,
                    upper_bound=60.0,
                )
            }
        ),
        _policy(),
    )
    assert left != right


def test_declared_target_domain_change_alters_fingerprint() -> None:
    left = compute_configuration_fingerprint(_request(), _policy())
    right = compute_configuration_fingerprint(
        _request(declared_target_maximum=99.0),
        _policy(),
    )
    assert left != right


def test_transient_metadata_and_path_excluded_from_fingerprint() -> None:
    payload = build_configuration_fingerprint_payload(_request(), _policy())
    assert "metadata" not in payload
    assert "csv_path" not in payload
    encoded = json.dumps(payload, default=str)
    assert "streamlit_mvp" not in encoded
    assert "data.csv" not in encoded


def test_same_dataset_config_version_schema_same_run_signature() -> None:
    config = compute_configuration_fingerprint(_request(), _policy())
    first = compute_run_signature(
        dataset_fingerprint="a" * 64,
        configuration_fingerprint=config,
        application_version="0.1.0",
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
    )
    second = compute_run_signature(
        dataset_fingerprint="a" * 64,
        configuration_fingerprint=config,
        application_version="0.1.0",
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
    )
    assert first == second
    assert len(first) == 64


def test_dataset_fingerprint_change_alters_run_signature() -> None:
    config = compute_configuration_fingerprint(_request(), _policy())
    left = compute_run_signature(
        dataset_fingerprint="a" * 64,
        configuration_fingerprint=config,
        application_version="0.1.0",
    )
    right = compute_run_signature(
        dataset_fingerprint="b" * 64,
        configuration_fingerprint=config,
        application_version="0.1.0",
    )
    assert left != right


def test_application_version_change_alters_run_signature() -> None:
    config = compute_configuration_fingerprint(_request(), _policy())
    left = compute_run_signature(
        dataset_fingerprint="a" * 64,
        configuration_fingerprint=config,
        application_version="0.1.0",
    )
    right = compute_run_signature(
        dataset_fingerprint="a" * 64,
        configuration_fingerprint=config,
        application_version="0.2.0",
    )
    assert left != right


def test_manifest_contains_no_forbidden_objects_and_agrees_on_features() -> None:
    started = datetime(2026, 7, 26, 1, 0, tzinfo=UTC)
    completed = started + timedelta(seconds=1.5)
    features = ["pressure", "temperature", "flow"]
    manifest = build_analysis_run_manifest(
        request=_request(),
        policy=_policy(),
        stage_records=[
            _stage(AnalysisWorkflowStage.LOAD),
            _stage(AnalysisWorkflowStage.PROFILE),
        ],
        workflow_status=AnalysisWorkflowStatus.PARTIAL,
        dataset_fingerprint="c" * 64,
        feature_columns=features,
        analysis_mode=AnalysisExecutionMode.SUPERVISED,
        requested_task=AnalysisTask.REGRESSION,
        resolved_task=AnalysisTask.REGRESSION,
        target_column="quality",
        selected_model="ridge",
        warning_count=1,
        started_at_utc=started,
        completed_at_utc=completed,
        duration_seconds=1.5,
        application_version="0.1.0",
    )
    dumped = manifest.model_dump(mode="json")
    assert_manifest_has_no_forbidden_objects(dumped)
    assert manifest.feature_count == len(features)
    assert manifest.feature_columns == features
    assert manifest.started_at_utc.tzinfo is not None
    assert manifest.completed_at_utc.tzinfo is not None
    assert manifest.duration_seconds >= 0.0
    assert "C:\\" not in json.dumps(dumped)
    assert "DataFrame" not in json.dumps(dumped)


def test_serialization_is_deterministic() -> None:
    started = datetime(2026, 7, 26, 2, 0, tzinfo=UTC)
    completed = started + timedelta(seconds=2.0)
    manifest = build_analysis_run_manifest(
        request=_request(),
        policy=_policy(),
        stage_records=[_stage(AnalysisWorkflowStage.LOAD)],
        workflow_status=AnalysisWorkflowStatus.REFUSED,
        dataset_fingerprint="d" * 64,
        feature_columns=["pressure", "temperature", "flow"],
        analysis_mode=AnalysisExecutionMode.SUPERVISED,
        requested_task=AnalysisTask.REGRESSION,
        resolved_task=None,
        target_column="quality",
        selected_model=None,
        warning_count=0,
        started_at_utc=started,
        completed_at_utc=completed,
        duration_seconds=2.0,
        application_version="0.1.0",
        recommendation_status=RecommendationStatus.REFUSED,
        recommendation_reason_codes=["LOW_CONFIDENCE"],
    )
    first = analysis_run_manifest_to_json(manifest)
    second = analysis_run_manifest_to_json(manifest)
    assert first == second
    assert first.endswith("\n")
    assert manifest.refusal_or_failure_code == "LOW_CONFIDENCE"


def test_recommendation_refusal_code_not_modeling_failure() -> None:
    started = datetime(2026, 7, 26, 3, 0, tzinfo=UTC)
    completed = started + timedelta(seconds=3.0)
    manifest = build_analysis_run_manifest(
        request=_request(),
        policy=_policy(),
        stage_records=[
            _stage(AnalysisWorkflowStage.LOAD),
            _stage(AnalysisWorkflowStage.SUPERVISED_FINAL_EVALUATION),
            _stage(
                AnalysisWorkflowStage.RECOMMENDATION,
                succeeded=False,
                structured_refusal=True,
                message="Recommendation refused by confidence gate.",
                metadata={"recommendation_status": "REFUSED"},
            ),
        ],
        workflow_status=AnalysisWorkflowStatus.REFUSED,
        dataset_fingerprint="e" * 64,
        feature_columns=["pressure", "temperature", "flow"],
        analysis_mode=AnalysisExecutionMode.SUPERVISED,
        requested_task=AnalysisTask.REGRESSION,
        resolved_task=AnalysisTask.REGRESSION,
        target_column="quality",
        selected_model="ridge",
        warning_count=0,
        started_at_utc=started,
        completed_at_utc=completed,
        duration_seconds=3.0,
        application_version="0.1.0",
        recommendation_status=RecommendationStatus.REFUSED,
        recommendation_reason_codes=["BELOW_CONFIDENCE_THRESHOLD"],
    )
    assert manifest.selected_model == "ridge"
    assert manifest.stage_entries[1].status == "SUCCEEDED"
    assert manifest.stage_entries[2].status == "REFUSED"
    assert manifest.refusal_or_failure_code == "BELOW_CONFIDENCE_THRESHOLD"


def test_get_application_version_is_non_empty() -> None:
    version = get_application_version()
    assert isinstance(version, str)
    assert version.strip()
    assert "\\" not in version
    assert not version.lower().endswith(".py")


def test_final_feature_override_used_for_fingerprint() -> None:
    request = _request(feature_columns=["pressure", "temperature", "flow", "batch"])
    with_override = compute_configuration_fingerprint(
        request,
        _policy(),
        feature_columns=["pressure", "temperature", "flow"],
    )
    without_override = compute_configuration_fingerprint(request, _policy())
    assert with_override != without_override
