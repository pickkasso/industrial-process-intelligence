"""Unit tests for current-setup reproducibility comparison (Step 14B)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.evaluation import (
    MetricAcceptanceDirection,
    MetricAcceptanceRule,
    ModelPerformanceAcceptancePolicy,
)
from process_intelligence.recommendation import (
    QualityOptimizationDirection,
    RecommendationObjective,
)
from process_intelligence.reporting import (
    ReproducibilityComparison,
    ReproducibilityComponentStatus,
    ReproducibilityMatchStatus,
    assert_reproducibility_comparison_has_no_forbidden_objects,
    compare_setup_to_run_manifest,
    reproducibility_comparison_to_json,
)
from process_intelligence.workflow import (
    AnalysisRunManifest,
    AnalysisWorkflowPolicy,
    AnalysisWorkflowRequest,
    AnalysisWorkflowStage,
    AnalysisWorkflowStageRecord,
    AnalysisWorkflowStatus,
    OperatingPointSelectionMode,
    build_analysis_run_manifest,
    compute_configuration_fingerprint,
    compute_run_signature,
    get_application_version,
)
from process_intelligence.workflow.run_manifest import MANIFEST_SCHEMA_VERSION


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
        "operating_point_selection": OperatingPointSelectionMode.TOP_RESIDUAL_ANOMALY,
        "max_simultaneous_changes": 2,
        "metadata": {"ui_source": "streamlit_mvp", "presentation_only": True},
    }
    payload.update(overrides)
    if "feature_columns" in overrides and "max_simultaneous_changes" not in overrides:
        features = payload["feature_columns"]
        if isinstance(features, list) and features:
            payload["max_simultaneous_changes"] = min(2, len(features))
    return AnalysisWorkflowRequest(**payload)  # type: ignore[arg-type]


def _policy(**overrides: object) -> AnalysisWorkflowPolicy:
    return AnalysisWorkflowPolicy(**overrides)  # type: ignore[arg-type]


def _manifest(
    *,
    request: AnalysisWorkflowRequest | None = None,
    policy: AnalysisWorkflowPolicy | None = None,
    dataset_fingerprint: str | None = "a" * 64,
    application_version: str | None = None,
    feature_columns: list[str] | None = None,
) -> AnalysisRunManifest:
    active_request = _request() if request is None else request
    active_policy = _policy() if policy is None else policy
    features = (
        list(active_request.feature_columns)
        if feature_columns is None
        else list(feature_columns)
    )
    started = datetime(2026, 7, 26, 10, 0, tzinfo=UTC)
    completed = started + timedelta(seconds=12.5)
    return build_analysis_run_manifest(
        request=active_request,
        policy=active_policy,
        stage_records=[
            AnalysisWorkflowStageRecord(
                stage=AnalysisWorkflowStage.LOAD,
                executed=True,
                succeeded=True,
                structured_refusal=False,
                message="LOAD completed.",
                warnings=[],
                metadata={"duration_seconds": 0.1},
            )
        ],
        workflow_status=AnalysisWorkflowStatus.COMPLETED,
        dataset_fingerprint=dataset_fingerprint,
        feature_columns=features,
        analysis_mode=active_request.analysis_mode,
        requested_task=active_request.requested_task,
        resolved_task=AnalysisTask.REGRESSION,
        target_column=active_request.target_column,
        selected_model="ridge",
        warning_count=0,
        started_at_utc=started,
        completed_at_utc=completed,
        duration_seconds=12.5,
        application_version=application_version,
    )


def test_exact_component_match() -> None:
    request = _request()
    policy = _policy()
    manifest = _manifest(request=request, policy=policy)
    before = manifest.model_dump(mode="json")
    comparison = compare_setup_to_run_manifest(
        stored_manifest=manifest,
        current_dataset_fingerprint=manifest.dataset_fingerprint,
        current_request=request,
        current_policy=policy,
        current_feature_columns=list(request.feature_columns),
        current_application_version=manifest.application_version,
        current_manifest_schema_version=manifest.manifest_schema_version,
    )
    assert comparison.overall_status is ReproducibilityMatchStatus.EXACT_MATCH
    assert comparison.dataset_matches is ReproducibilityComponentStatus.MATCH
    assert comparison.configuration_matches is ReproducibilityComponentStatus.MATCH
    assert comparison.application_version_matches is ReproducibilityComponentStatus.MATCH
    assert comparison.manifest_schema_matches is ReproducibilityComponentStatus.MATCH
    assert comparison.comparison_available is True
    assert comparison.current_candidate_run_signature == manifest.run_signature
    assert "match this stored run" in comparison.explanatory_messages[0].lower()
    assert before == manifest.model_dump(mode="json")


def test_dataset_only_mismatch() -> None:
    request = _request()
    policy = _policy()
    manifest = _manifest(request=request, policy=policy, dataset_fingerprint="a" * 64)
    comparison = compare_setup_to_run_manifest(
        stored_manifest=manifest,
        current_dataset_fingerprint="b" * 64,
        current_request=request,
        current_policy=policy,
        current_application_version=manifest.application_version,
    )
    assert comparison.overall_status is ReproducibilityMatchStatus.DATASET_CHANGED
    assert comparison.dataset_matches is ReproducibilityComponentStatus.CHANGED
    assert comparison.configuration_matches is ReproducibilityComponentStatus.MATCH
    assert comparison.current_candidate_run_signature != manifest.run_signature


def test_configuration_only_mismatch() -> None:
    request = _request()
    policy = _policy()
    manifest = _manifest(request=request, policy=policy)
    changed = _request(feature_columns=["pressure", "temperature"])
    comparison = compare_setup_to_run_manifest(
        stored_manifest=manifest,
        current_dataset_fingerprint=manifest.dataset_fingerprint,
        current_request=changed,
        current_policy=policy,
        current_application_version=manifest.application_version,
    )
    assert comparison.overall_status is ReproducibilityMatchStatus.CONFIGURATION_CHANGED
    assert comparison.configuration_matches is ReproducibilityComponentStatus.CHANGED
    assert comparison.dataset_matches is ReproducibilityComponentStatus.MATCH
    assert comparison.current_candidate_run_signature != manifest.run_signature


def test_version_only_mismatch() -> None:
    request = _request()
    policy = _policy()
    manifest = _manifest(
        request=request,
        policy=policy,
        application_version="0.1.0",
    )
    comparison = compare_setup_to_run_manifest(
        stored_manifest=manifest,
        current_dataset_fingerprint=manifest.dataset_fingerprint,
        current_request=request,
        current_policy=policy,
        current_application_version="0.2.0",
    )
    assert (
        comparison.overall_status
        is ReproducibilityMatchStatus.APPLICATION_VERSION_CHANGED
    )
    assert (
        comparison.application_version_matches
        is ReproducibilityComponentStatus.CHANGED
    )
    assert comparison.configuration_matches is ReproducibilityComponentStatus.MATCH


def test_schema_only_mismatch() -> None:
    request = _request()
    policy = _policy()
    manifest = _manifest(request=request, policy=policy)
    comparison = compare_setup_to_run_manifest(
        stored_manifest=manifest,
        current_dataset_fingerprint=manifest.dataset_fingerprint,
        current_request=request,
        current_policy=policy,
        current_application_version=manifest.application_version,
        current_manifest_schema_version=manifest.manifest_schema_version + 1,
    )
    assert (
        comparison.overall_status is ReproducibilityMatchStatus.MANIFEST_SCHEMA_CHANGED
    )
    assert comparison.manifest_schema_matches is ReproducibilityComponentStatus.CHANGED


def test_multiple_mismatches() -> None:
    request = _request()
    policy = _policy()
    manifest = _manifest(
        request=request,
        policy=policy,
        dataset_fingerprint="a" * 64,
        application_version="0.1.0",
    )
    comparison = compare_setup_to_run_manifest(
        stored_manifest=manifest,
        current_dataset_fingerprint="b" * 64,
        current_request=_request(target_column="defect_rate"),
        current_policy=policy,
        current_application_version="0.2.0",
    )
    assert (
        comparison.overall_status
        is ReproducibilityMatchStatus.MULTIPLE_COMPONENTS_CHANGED
    )
    assert comparison.dataset_matches is ReproducibilityComponentStatus.CHANGED
    assert comparison.configuration_matches is ReproducibilityComponentStatus.CHANGED
    assert (
        comparison.application_version_matches
        is ReproducibilityComponentStatus.CHANGED
    )


def test_incomplete_current_configuration() -> None:
    request = _request()
    policy = _policy()
    manifest = _manifest(request=request, policy=policy)
    before = manifest.model_dump(mode="json")
    comparison = compare_setup_to_run_manifest(
        stored_manifest=manifest,
        current_dataset_fingerprint=manifest.dataset_fingerprint,
        current_configuration_incomplete=True,
        incompleteness_reasons=["target selected", "invalid performance rule"],
    )
    assert (
        comparison.overall_status
        is ReproducibilityMatchStatus.CURRENT_CONFIGURATION_INCOMPLETE
    )
    assert comparison.configuration_matches is ReproducibilityComponentStatus.INCOMPLETE
    assert comparison.current_candidate_run_signature is None
    assert "target selected" in comparison.incompleteness_reasons
    assert "cannot be fingerprinted" in comparison.explanatory_messages[0]
    assert before == manifest.model_dump(mode="json")


def test_unavailable_stored_manifest() -> None:
    comparison = compare_setup_to_run_manifest(
        stored_manifest=None,
        current_dataset_fingerprint="a" * 64,
        current_request=_request(),
        current_policy=_policy(),
    )
    assert comparison.overall_status is ReproducibilityMatchStatus.UNAVAILABLE
    assert comparison.comparison_available is False


def test_unavailable_active_dataset() -> None:
    manifest = _manifest()
    comparison = compare_setup_to_run_manifest(
        stored_manifest=manifest,
        current_dataset_fingerprint=None,
        current_request=_request(),
        current_policy=_policy(),
    )
    assert comparison.overall_status is ReproducibilityMatchStatus.UNAVAILABLE
    assert comparison.comparison_available is False


def test_candidate_signature_equals_stored_for_exact_match() -> None:
    request = _request()
    policy = _policy()
    manifest = _manifest(request=request, policy=policy)
    comparison = compare_setup_to_run_manifest(
        stored_manifest=manifest,
        current_dataset_fingerprint=manifest.dataset_fingerprint,
        current_request=request,
        current_policy=policy,
        current_application_version=manifest.application_version,
    )
    assert comparison.current_candidate_run_signature == comparison.stored_run_signature
    assert comparison.current_candidate_run_signature == manifest.run_signature


@pytest.mark.parametrize(
    ("override",),
    [
        ({"feature_columns": ["pressure", "flow"]},),
        ({"target_column": "defect_rate"},),
        ({"operating_point_selection": OperatingPointSelectionMode.LATEST_ROW},),
    ],
)
def test_candidate_signature_differs_after_material_change(
    override: dict[str, object],
) -> None:
    request = _request()
    policy = _policy()
    manifest = _manifest(request=request, policy=policy)
    comparison = compare_setup_to_run_manifest(
        stored_manifest=manifest,
        current_dataset_fingerprint=manifest.dataset_fingerprint,
        current_request=_request(**override),
        current_policy=policy,
        current_application_version=manifest.application_version,
    )
    assert comparison.current_candidate_run_signature is not None
    assert comparison.current_candidate_run_signature != manifest.run_signature


def test_timestamps_and_duration_do_not_affect_comparison() -> None:
    request = _request()
    policy = _policy()
    first = _manifest(request=request, policy=policy)
    second_started = datetime(2026, 7, 26, 18, 0, tzinfo=UTC)
    second = build_analysis_run_manifest(
        request=request,
        policy=policy,
        stage_records=[
            AnalysisWorkflowStageRecord(
                stage=AnalysisWorkflowStage.LOAD,
                executed=True,
                succeeded=True,
                structured_refusal=False,
                message="LOAD completed later.",
                warnings=[],
                metadata={"duration_seconds": 99.0},
            )
        ],
        workflow_status=AnalysisWorkflowStatus.COMPLETED,
        dataset_fingerprint=first.dataset_fingerprint,
        feature_columns=list(request.feature_columns),
        analysis_mode=request.analysis_mode,
        requested_task=request.requested_task,
        resolved_task=AnalysisTask.REGRESSION,
        target_column=request.target_column,
        selected_model="ridge",
        warning_count=0,
        started_at_utc=second_started,
        completed_at_utc=second_started + timedelta(seconds=99.0),
        duration_seconds=99.0,
        application_version=first.application_version,
    )
    assert first.run_signature == second.run_signature
    comparison = compare_setup_to_run_manifest(
        stored_manifest=second,
        current_dataset_fingerprint=first.dataset_fingerprint,
        current_request=request,
        current_policy=policy,
        current_application_version=first.application_version,
    )
    assert comparison.overall_status is ReproducibilityMatchStatus.EXACT_MATCH


def test_comparison_serialization_is_deterministic() -> None:
    request = _request()
    policy = _policy()
    manifest = _manifest(request=request, policy=policy)
    comparison = compare_setup_to_run_manifest(
        stored_manifest=manifest,
        current_dataset_fingerprint=manifest.dataset_fingerprint,
        current_request=request,
        current_policy=policy,
        current_application_version=manifest.application_version,
    )
    first = reproducibility_comparison_to_json(comparison)
    second = reproducibility_comparison_to_json(comparison)
    assert first == second
    payload = json.loads(first)
    assert_reproducibility_comparison_has_no_forbidden_objects(payload)


def test_stored_manifest_unchanged_by_comparison() -> None:
    request = _request()
    policy = _policy()
    manifest = _manifest(request=request, policy=policy)
    before = manifest.model_dump(mode="json")
    compare_setup_to_run_manifest(
        stored_manifest=manifest,
        current_dataset_fingerprint="b" * 64,
        current_request=_request(feature_columns=["pressure"]),
        current_policy=policy,
        current_application_version="9.9.9",
        current_manifest_schema_version=99,
    )
    assert manifest.model_dump(mode="json") == before


def test_no_dataframe_or_model_objects_stored() -> None:
    request = _request()
    policy = _policy()
    manifest = _manifest(request=request, policy=policy)
    comparison = compare_setup_to_run_manifest(
        stored_manifest=manifest,
        current_dataset_fingerprint=manifest.dataset_fingerprint,
        current_request=request,
        current_policy=policy,
        current_application_version=manifest.application_version,
    )
    payload = comparison.model_dump(mode="python")
    assert_reproducibility_comparison_has_no_forbidden_objects(payload)
    assert isinstance(comparison, ReproducibilityComparison)


def test_presentation_only_metadata_does_not_change_match() -> None:
    request = _request(metadata={"ui_entry": "streamlit_form", "page": 1})
    policy = _policy()
    manifest = _manifest(request=request, policy=policy)
    altered = _request(
        metadata={
            "ui_entry": "streamlit_form",
            "page": 2,
            "expander_open": True,
            "selected_row_for_display": 12,
        }
    )
    left = compute_configuration_fingerprint(request, policy)
    right = compute_configuration_fingerprint(altered, policy)
    assert left == right
    comparison = compare_setup_to_run_manifest(
        stored_manifest=manifest,
        current_dataset_fingerprint=manifest.dataset_fingerprint,
        current_request=altered,
        current_policy=policy,
        current_application_version=manifest.application_version,
    )
    assert comparison.overall_status is ReproducibilityMatchStatus.EXACT_MATCH


def test_reuses_public_fingerprint_helpers() -> None:
    request = _request()
    policy = _policy()
    config = compute_configuration_fingerprint(request, policy)
    version = get_application_version()
    signature = compute_run_signature(
        dataset_fingerprint="c" * 64,
        configuration_fingerprint=config,
        application_version=version,
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
    )
    manifest = _manifest(
        request=request,
        policy=policy,
        dataset_fingerprint="c" * 64,
        application_version=version,
    )
    assert manifest.configuration_fingerprint == config
    assert manifest.run_signature == signature
    comparison = compare_setup_to_run_manifest(
        stored_manifest=manifest,
        current_dataset_fingerprint="c" * 64,
        current_request=request,
        current_policy=policy,
        current_application_version=version,
    )
    assert comparison.overall_status is ReproducibilityMatchStatus.EXACT_MATCH
