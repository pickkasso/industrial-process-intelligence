"""Unit tests for reproducibility bundle export (Step 14C)."""

from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path

from process_intelligence.core.enums import AnalysisTask, ColumnRole
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
    ReproducibilityBundle,
    assert_bundle_has_no_forbidden_objects,
    build_effective_configuration_payload,
    build_environment_payload,
    build_feature_schema_payload,
    build_reproducibility_bundle,
    compute_reproducibility_bundle_signature,
    reproducibility_bundle_download_filename,
    serialize_reproducibility_bundle,
)
from process_intelligence.reporting.reproducibility_bundle import (
    bundle_contains_raw_dataset_rows,
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
        "excluded_columns": ["notes"],
        "column_role_overrides": {"notes": ColumnRole.CONTEXT},
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


def _feature_schema(
    request: AnalysisWorkflowRequest | None = None,
    *,
    feature_columns: list[str] | None = None,
) -> dict[str, object]:
    active = _request() if request is None else request
    features = (
        list(active.feature_columns) if feature_columns is None else list(feature_columns)
    )
    return build_feature_schema_payload(
        target_column=active.target_column,
        feature_columns=features,
        column_roles={
            name: role.value for name, role in active.column_role_overrides.items()
        },
        dtypes={
            "quality": "Float64",
            "pressure": "Float64",
            "temperature": "Float64",
            "flow": "Float64",
            "timestamp": "Utf8",
            "lot_id": "Utf8",
            "notes": "Utf8",
        },
        excluded_columns=list(active.excluded_columns),
        identifier_columns=list(active.identifier_columns),
        timestamp_column=active.timestamp_column,
    )


def _environment(
    *,
    application_version: str | None = None,
    manifest_schema_version: int = MANIFEST_SCHEMA_VERSION,
) -> dict[str, object]:
    return build_environment_payload(
        application_version=application_version or get_application_version(),
        manifest_schema_version=manifest_schema_version,
        python_version="3.11.0",
        platform_system="TestOS",
        platform_release="1.0",
        platform_machine="x86_64",
        package_versions={"pydantic": "2.0.0", "polars": "1.0.0"},
    )


def _build_bundle(
    *,
    request: AnalysisWorkflowRequest | None = None,
    policy: AnalysisWorkflowPolicy | None = None,
    manifest: AnalysisRunManifest | None = None,
    effective_configuration: dict[str, object] | None = None,
    feature_schema: dict[str, object] | None = None,
    environment: dict[str, object] | None = None,
    generated_at: datetime | None = None,
) -> ReproducibilityBundle:
    active_request = _request() if request is None else request
    active_policy = _policy() if policy is None else policy
    active_manifest = (
        _manifest(request=active_request, policy=active_policy)
        if manifest is None
        else manifest
    )
    config = (
        build_effective_configuration_payload(
            active_request,
            active_policy,
            feature_columns=list(active_manifest.feature_columns),
        )
        if effective_configuration is None
        else effective_configuration
    )
    schema = (
        _feature_schema(active_request, feature_columns=list(active_manifest.feature_columns))
        if feature_schema is None
        else feature_schema
    )
    env = (
        _environment(application_version=active_manifest.application_version)
        if environment is None
        else environment
    )
    return build_reproducibility_bundle(
        run_manifest=active_manifest,
        effective_configuration=config,
        feature_schema=schema,
        environment=env,
        generated_at=generated_at,
    )


def test_same_inputs_produce_same_canonical_json_files() -> None:
    first = _build_bundle()
    second = _build_bundle()
    assert first.file_map() == second.file_map()
    for path, content in first.file_map().items():
        if not path.endswith(".json"):
            continue
        parsed = json.loads(content)
        assert content.endswith("\n")
        assert content == (
            json.dumps(
                parsed,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )


def test_same_inputs_produce_same_bundle_signature() -> None:
    first = _build_bundle()
    second = _build_bundle()
    assert first.bundle_manifest.bundle_signature == second.bundle_manifest.bundle_signature
    assert compute_reproducibility_bundle_signature(
        first.signed_file_map()
    ) == first.bundle_manifest.bundle_signature


def test_dataset_fingerprint_change_changes_bundle_signature() -> None:
    base = _build_bundle()
    changed_manifest = _manifest(dataset_fingerprint="b" * 64)
    changed = _build_bundle(manifest=changed_manifest)
    assert (
        changed.bundle_manifest.bundle_signature
        != base.bundle_manifest.bundle_signature
    )


def test_configuration_fingerprint_change_changes_bundle_signature() -> None:
    base = _build_bundle()
    request = _request(feature_columns=["pressure", "temperature"])
    changed = _build_bundle(request=request)
    assert (
        changed.bundle_manifest.configuration_fingerprint
        != base.bundle_manifest.configuration_fingerprint
    )
    assert (
        changed.bundle_manifest.bundle_signature
        != base.bundle_manifest.bundle_signature
    )


def test_application_version_change_changes_bundle_signature() -> None:
    base = _build_bundle()
    changed_manifest = _manifest(application_version="9.9.9-test")
    changed = _build_bundle(
        manifest=changed_manifest,
        environment=_environment(application_version="9.9.9-test"),
    )
    assert (
        changed.bundle_manifest.bundle_signature
        != base.bundle_manifest.bundle_signature
    )


def test_manifest_schema_change_changes_bundle_signature() -> None:
    base = _build_bundle()
    env = _environment(manifest_schema_version=MANIFEST_SCHEMA_VERSION + 1)
    changed = _build_bundle(environment=env)
    assert (
        changed.bundle_manifest.bundle_signature
        != base.bundle_manifest.bundle_signature
    )


def test_generated_timestamp_does_not_change_bundle_signature() -> None:
    first = _build_bundle(generated_at=datetime(2026, 7, 26, 12, 0, tzinfo=UTC))
    second = _build_bundle(generated_at=datetime(2026, 7, 26, 18, 30, tzinfo=UTC))
    assert first.bundle_manifest.bundle_signature == second.bundle_manifest.bundle_signature
    assert first.bundle_manifest.generated_at != second.bundle_manifest.generated_at
    first_manifest = json.loads(
        first.file_map()["reproducibility_bundle/bundle_manifest.json"]
    )
    second_manifest = json.loads(
        second.file_map()["reproducibility_bundle/bundle_manifest.json"]
    )
    assert first_manifest["generated_at"] != second_manifest["generated_at"]
    assert first_manifest["bundle_signature"] == second_manifest["bundle_signature"]


def test_presentation_only_metadata_does_not_change_bundle_signature() -> None:
    request_a = _request(metadata={"ui_source": "a", "presentation_only": True})
    request_b = _request(
        metadata={
            "ui_source": "b",
            "presentation_only": False,
            "selected_expander": "Run provenance",
        }
    )
    first = _build_bundle(request=request_a)
    second = _build_bundle(request=request_b)
    assert first.signed_file_map() == second.signed_file_map()
    assert first.bundle_manifest.bundle_signature == second.bundle_manifest.bundle_signature
    config = json.loads(
        first.file_map()["reproducibility_bundle/effective_configuration.json"]
    )
    assert "metadata" not in config
    assert "csv_path" not in config


def test_raw_dataset_rows_are_not_included() -> None:
    bundle = _build_bundle()
    assert bundle_contains_raw_dataset_rows(bundle) is False
    joined = "\n".join(bundle.file_map().values())
    assert "50.0" not in joined
    assert "LOT-1" not in joined
    assert "csv_bytes" not in joined
    assert "dataset_bytes" not in joined


def test_unicode_and_korean_column_names_serialize_correctly() -> None:
    request = _request(
        target_column="품질",
        feature_columns=["압력", "온도", "유량_α"],
        identifier_columns=["로트"],
        excluded_columns=["메모"],
        column_role_overrides={"메모": ColumnRole.CONTEXT},
        timestamp_column="시각",
        max_simultaneous_changes=2,
    )
    schema = build_feature_schema_payload(
        target_column="품질",
        feature_columns=["압력", "온도", "유량_α"],
        column_roles={"메모": ColumnRole.CONTEXT.value},
        dtypes={"품질": "Float64", "압력": "Float64", "온도": "Float64", "유량_α": "Float64"},
        excluded_columns=["메모"],
        identifier_columns=["로트"],
        timestamp_column="시각",
    )
    bundle = _build_bundle(request=request, feature_schema=schema)
    feature_json = bundle.file_map()["reproducibility_bundle/feature_schema.json"]
    assert "품질" in feature_json
    assert "압력" in feature_json
    assert "유량_α" in feature_json
    parsed = json.loads(feature_json)
    assert parsed["target_column"] == "품질"
    assert parsed["feature_columns"] == ["압력", "온도", "유량_α"]


def test_missing_optional_schema_fields_serialize_safely() -> None:
    schema = build_feature_schema_payload(
        target_column=None,
        feature_columns=["pressure", "temperature"],
        column_roles=None,
        dtypes=None,
        excluded_columns=None,
        identifier_columns=None,
        timestamp_column=None,
    )
    bundle = build_reproducibility_bundle(
        run_manifest=_manifest(),
        effective_configuration={"analysis_mode": "ANOMALY_ONLY"},
        feature_schema=schema,
        environment=_environment(),
    )
    parsed = json.loads(bundle.file_map()["reproducibility_bundle/feature_schema.json"])
    assert parsed["target_column"] is None
    assert parsed["timestamp_column"] is None
    assert parsed["column_roles"] == {}
    assert parsed["dtypes"] == {}
    assert parsed["excluded_columns"] == []
    assert parsed["identifier_columns"] == []


def test_report_and_run_manifest_remain_unchanged_after_export() -> None:
    request = _request()
    policy = _policy()
    manifest = _manifest(request=request, policy=policy)
    before_manifest = manifest.model_dump(mode="json")
    config = build_effective_configuration_payload(
        request,
        policy,
        feature_columns=list(manifest.feature_columns),
    )
    before_config = json.dumps(config, sort_keys=True)
    bundle = build_reproducibility_bundle(
        run_manifest=manifest,
        effective_configuration=config,
        feature_schema=_feature_schema(request),
        environment=_environment(application_version=manifest.application_version),
        generated_at=datetime(2026, 7, 26, 15, 0, tzinfo=UTC),
    )
    zip_bytes = serialize_reproducibility_bundle(bundle)
    assert isinstance(zip_bytes, (bytes, bytearray))
    assert manifest.model_dump(mode="json") == before_manifest
    assert json.dumps(config, sort_keys=True) == before_config
    dumped = bundle.bundle_manifest.model_dump(mode="json")
    assert_bundle_has_no_forbidden_objects(dumped)
    with zipfile.ZipFile(BytesIO(zip_bytes)) as archive:
        names = set(archive.namelist())
        assert "reproducibility_bundle/bundle_manifest.json" in names
        assert "reproducibility_bundle/run_manifest.json" in names
        assert "reproducibility_bundle/effective_configuration.json" in names
        assert "reproducibility_bundle/feature_schema.json" in names
        assert "reproducibility_bundle/environment.json" in names
        assert "reproducibility_bundle/README.txt" in names


def test_download_filename_includes_safe_run_identifier() -> None:
    manifest = _manifest()
    filename = reproducibility_bundle_download_filename(manifest)
    assert filename.startswith("reproducibility_bundle_")
    assert filename.endswith(".zip")
    assert manifest.run_signature[:12] in filename
    assert "/" not in filename
    assert "\\" not in filename


def test_signed_files_exclude_generated_at_from_signature_inputs() -> None:
    bundle = _build_bundle(generated_at=datetime(2026, 7, 26, 12, 0, tzinfo=UTC))
    signed = bundle.signed_file_map()
    assert "bundle_manifest.json" not in signed
    for name in (
        "run_manifest.json",
        "effective_configuration.json",
        "feature_schema.json",
        "environment.json",
    ):
        assert "generated_at" not in json.loads(signed[name])
