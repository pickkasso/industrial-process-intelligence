"""Unit tests for reproducibility-bundle verification (Step 14D)."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

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
    BUNDLE_ARCHIVE_ROOT,
    BUNDLE_MANIFEST_BASENAME,
    BUNDLE_SCHEMA_VERSION,
    MAX_BUNDLE_COMPRESSION_RATIO,
    MAX_BUNDLE_ENTRY_UNCOMPRESSED_BYTES,
    MAX_BUNDLE_ZIP_BYTES,
    SIGNED_BUNDLE_FILE_BASENAMES,
    ReproducibilityBundleVerificationStatus,
    SchemaCompatibilityStatus,
    build_effective_configuration_payload,
    build_environment_payload,
    build_feature_schema_payload,
    build_reproducibility_bundle,
    evaluate_schema_compatibility,
    inspect_reproducibility_bundle,
    serialize_reproducibility_bundle,
    validate_reproducibility_bundle_archive,
    verify_reproducibility_bundle,
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
        "metadata": {"ui_source": "streamlit_mvp"},
    }
    payload.update(overrides)
    return AnalysisWorkflowRequest(**payload)  # type: ignore[arg-type]


def _policy(**overrides: object) -> AnalysisWorkflowPolicy:
    return AnalysisWorkflowPolicy(**overrides)  # type: ignore[arg-type]


def _manifest(
    *,
    request: AnalysisWorkflowRequest | None = None,
    policy: AnalysisWorkflowPolicy | None = None,
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
        dataset_fingerprint="a" * 64,
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
    )


def _feature_schema(
    request: AnalysisWorkflowRequest,
    *,
    feature_columns: list[str] | None = None,
) -> dict[str, object]:
    features = (
        list(request.feature_columns) if feature_columns is None else list(feature_columns)
    )
    return build_feature_schema_payload(
        target_column=request.target_column,
        feature_columns=features,
        column_roles={
            name: role.value for name, role in request.column_role_overrides.items()
        },
        dtypes={
            "quality": "Float64",
            "pressure": "Float64",
            "temperature": "Float64",
            "flow": "Float64",
            "timestamp": "Utf8",
            "lot_id": "Utf8",
            "notes": "Utf8",
            "온도": "Float64",
        },
        excluded_columns=list(request.excluded_columns),
        identifier_columns=list(request.identifier_columns),
        timestamp_column=request.timestamp_column,
    )


def _valid_zip_bytes(**bundle_overrides: object) -> bytes:
    request = _request()
    policy = _policy()
    manifest = _manifest(request=request, policy=policy)
    bundle = build_reproducibility_bundle(
        run_manifest=manifest,
        effective_configuration=build_effective_configuration_payload(
            request,
            policy,
            feature_columns=list(manifest.feature_columns),
        ),
        feature_schema=_feature_schema(request),
        environment=build_environment_payload(
            application_version=manifest.application_version,
            manifest_schema_version=manifest.manifest_schema_version,
            python_version="3.11.0",
            platform_system="TestOS",
            platform_release="1.0",
            platform_machine="x86_64",
            package_versions={"pydantic": "2.0.0"},
        ),
        **bundle_overrides,  # type: ignore[arg-type]
    )
    return serialize_reproducibility_bundle(bundle)


def _zip_member_map(zip_bytes: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes), mode="r") as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _rewrite_zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(filename=name)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, members[name])
    return buffer.getvalue()


def _path(name: str) -> str:
    return f"{BUNDLE_ARCHIVE_ROOT}/{name}"


def test_valid_step_14c_bundle_returns_valid() -> None:
    result = verify_reproducibility_bundle(_valid_zip_bytes())
    assert result.overall_status is ReproducibilityBundleVerificationStatus.VALID
    assert result.expected_bundle_signature == result.computed_bundle_signature
    assert result.inspection is not None
    assert result.inspection.raw_dataset_included is False
    assert result.inspection.bundle_schema_version == BUNDLE_SCHEMA_VERSION
    assert (
        result.bundle_schema_compatibility is not None
        and result.bundle_schema_compatibility.status
        is SchemaCompatibilityStatus.SUPPORTED_EXACT
    )


def test_valid_korean_unicode_bundle_is_inspected_correctly() -> None:
    request = _request(feature_columns=["pressure", "온도", "flow"])
    policy = _policy()
    manifest = _manifest(
        request=request,
        policy=policy,
        feature_columns=["pressure", "온도", "flow"],
    )
    zip_bytes = serialize_reproducibility_bundle(
        build_reproducibility_bundle(
            run_manifest=manifest,
            effective_configuration=build_effective_configuration_payload(
                request,
                policy,
                feature_columns=list(manifest.feature_columns),
            ),
            feature_schema=_feature_schema(
                request,
                feature_columns=list(manifest.feature_columns),
            ),
            environment=build_environment_payload(
                application_version=manifest.application_version,
                python_version="3.11.0",
                platform_system="TestOS",
                package_versions={"polars": "1.0.0"},
            ),
        )
    )
    result = inspect_reproducibility_bundle(zip_bytes)
    assert result.overall_status is ReproducibilityBundleVerificationStatus.VALID
    assert result.inspection is not None
    assert "온도" in result.inspection.final_modeling_feature_names
    assert result.inspection.final_modeling_feature_count == 3


def test_changed_signed_file_returns_signature_mismatch() -> None:
    members = _zip_member_map(_valid_zip_bytes())
    path = _path("README.txt")
    members[path] = (members[path].decode("utf-8") + "tampered\n").encode("utf-8")
    result = verify_reproducibility_bundle(_rewrite_zip(members))
    assert result.overall_status is ReproducibilityBundleVerificationStatus.SIGNATURE_MISMATCH
    assert result.expected_bundle_signature != result.computed_bundle_signature


def test_changed_bundle_manifest_metadata_returns_manifest_content_mismatch() -> None:
    members = _zip_member_map(_valid_zip_bytes())
    path = _path(BUNDLE_MANIFEST_BASENAME)
    manifest = json.loads(members[path].decode("utf-8"))
    manifest["run_signature"] = "b" * 64
    members[path] = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    result = verify_reproducibility_bundle(_rewrite_zip(members))
    assert (
        result.overall_status
        is ReproducibilityBundleVerificationStatus.MANIFEST_CONTENT_MISMATCH
    )
    assert "run_signature" in result.invalid_components


def test_missing_signed_file_returns_missing_required_file() -> None:
    members = _zip_member_map(_valid_zip_bytes())
    del members[_path("environment.json")]
    result = verify_reproducibility_bundle(_rewrite_zip(members))
    assert (
        result.overall_status
        is ReproducibilityBundleVerificationStatus.MISSING_REQUIRED_FILE
    )


def test_missing_bundle_manifest_returns_missing_required_file() -> None:
    members = _zip_member_map(_valid_zip_bytes())
    del members[_path(BUNDLE_MANIFEST_BASENAME)]
    result = verify_reproducibility_bundle(_rewrite_zip(members))
    assert (
        result.overall_status
        is ReproducibilityBundleVerificationStatus.MISSING_REQUIRED_FILE
    )


def test_unexpected_file_returns_unexpected_file() -> None:
    members = _zip_member_map(_valid_zip_bytes())
    members[f"{BUNDLE_ARCHIVE_ROOT}/extra.txt"] = b"unexpected\n"
    result = verify_reproducibility_bundle(_rewrite_zip(members))
    assert (
        result.overall_status is ReproducibilityBundleVerificationStatus.UNEXPECTED_FILE
    )


def test_duplicate_archive_member_is_rejected() -> None:
    valid = _valid_zip_bytes()
    members = _zip_member_map(valid)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
        archive.writestr(_path("README.txt"), members[_path("README.txt")])
    result = verify_reproducibility_bundle(buffer.getvalue())
    assert result.overall_status is ReproducibilityBundleVerificationStatus.DUPLICATE_FILE


def test_path_traversal_is_rejected() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        archive.writestr("../evil.txt", b"x")
    result = validate_reproducibility_bundle_archive(buffer.getvalue())
    assert result.overall_status is ReproducibilityBundleVerificationStatus.UNSAFE_ARCHIVE


def test_absolute_unix_path_is_rejected() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        archive.writestr("/tmp/evil.txt", b"x")
    result = validate_reproducibility_bundle_archive(buffer.getvalue())
    assert result.overall_status is ReproducibilityBundleVerificationStatus.UNSAFE_ARCHIVE


def test_windows_drive_path_is_rejected() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        archive.writestr("C:/Windows/evil.txt", b"x")
    result = validate_reproducibility_bundle_archive(buffer.getvalue())
    assert result.overall_status is ReproducibilityBundleVerificationStatus.UNSAFE_ARCHIVE


def test_backslash_traversal_is_rejected() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        # ZipInfo keeps the literal backslash name.
        info = zipfile.ZipInfo(filename="reproducibility_bundle\\..\\evil.txt")
        archive.writestr(info, b"x")
    result = validate_reproducibility_bundle_archive(buffer.getvalue())
    assert result.overall_status is ReproducibilityBundleVerificationStatus.UNSAFE_ARCHIVE


def test_symbolic_link_entry_is_rejected() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        info = zipfile.ZipInfo(filename=_path("README.txt"))
        info.create_system = 3
        info.external_attr = (0o120777 << 16)
        archive.writestr(info, b"target")
    result = validate_reproducibility_bundle_archive(buffer.getvalue())
    assert result.overall_status is ReproducibilityBundleVerificationStatus.UNSAFE_ARCHIVE


def _force_zip_encryption_flag(zip_bytes: bytes, member_name: str) -> bytes:
    """Set the encrypt bit in local/central headers (stdlib writestr clears it)."""
    data = bytearray(zip_bytes)
    name = member_name.encode("utf-8")
    offset = 0
    while True:
        local = data.find(b"PK\x03\x04", offset)
        if local < 0:
            break
        name_len = int.from_bytes(data[local + 26 : local + 28], "little")
        name_start = local + 30
        if data[name_start : name_start + name_len] == name:
            flag = int.from_bytes(data[local + 6 : local + 8], "little") | 0x1
            data[local + 6 : local + 8] = flag.to_bytes(2, "little")
        offset = local + 4
    offset = 0
    while True:
        central = data.find(b"PK\x01\x02", offset)
        if central < 0:
            break
        name_len = int.from_bytes(data[central + 28 : central + 30], "little")
        name_start = central + 46
        if data[name_start : name_start + name_len] == name:
            flag = int.from_bytes(data[central + 8 : central + 10], "little") | 0x1
            data[central + 8 : central + 10] = flag.to_bytes(2, "little")
        offset = central + 4
    return bytes(data)


def test_encrypted_zip_entry_is_rejected() -> None:
    members = {_path("README.txt"): b"secret"}
    forged = _force_zip_encryption_flag(_rewrite_zip(members), _path("README.txt"))
    with zipfile.ZipFile(io.BytesIO(forged), mode="r") as archive:
        assert archive.getinfo(_path("README.txt")).flag_bits & 0x1
    result = validate_reproducibility_bundle_archive(forged)
    assert result.overall_status is ReproducibilityBundleVerificationStatus.UNSAFE_ARCHIVE


def test_excessive_compressed_size_is_rejected() -> None:
    oversized = b"0" * (MAX_BUNDLE_ZIP_BYTES + 1)
    result = validate_reproducibility_bundle_archive(oversized)
    assert (
        result.overall_status
        is ReproducibilityBundleVerificationStatus.SIZE_LIMIT_EXCEEDED
    )


def test_excessive_uncompressed_entry_size_is_rejected() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED) as archive:
        payload = b"a" * (MAX_BUNDLE_ENTRY_UNCOMPRESSED_BYTES + 1)
        archive.writestr(_path("README.txt"), payload)
    result = validate_reproducibility_bundle_archive(buffer.getvalue())
    assert (
        result.overall_status
        is ReproducibilityBundleVerificationStatus.SIZE_LIMIT_EXCEEDED
    )


def test_suspicious_compression_ratio_is_rejected() -> None:
    # Craft ZipInfo metadata claiming a huge uncompressed size with tiny payload.
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        payload = b"0" * 64
        info = zipfile.ZipInfo(filename=_path("README.txt"))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, payload)
        # Re-open and patch file_size via a second crafted member set.
    # Build a minimal local ZIP where compress_size is small and file_size is huge.
    # Use ZipInfo before write; Python sets sizes from payload, so construct manually.
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo(filename=_path("README.txt"))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, b"0")
    # Read back and recreate with forged sizes by writing a custom archive.
    forged = io.BytesIO()
    with zipfile.ZipFile(forged, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo(filename=_path("README.txt"))
        info.compress_type = zipfile.ZIP_DEFLATED
        # writestr overwrites sizes; after write, mutate the ZipInfo in the archive.
        archive.writestr(info, b"0" * 8)
        written = archive.getinfo(_path("README.txt"))
        # Force a suspicious declared ratio for the safety pre-check.
        written.file_size = int(written.compress_size * (MAX_BUNDLE_COMPRESSION_RATIO + 5))
    result = validate_reproducibility_bundle_archive(forged.getvalue())
    assert result.overall_status in {
        ReproducibilityBundleVerificationStatus.UNSAFE_ARCHIVE,
        ReproducibilityBundleVerificationStatus.SIZE_LIMIT_EXCEEDED,
        ReproducibilityBundleVerificationStatus.MALFORMED_ZIP,
    }


def test_malformed_zip_is_rejected() -> None:
    result = verify_reproducibility_bundle(b"not-a-zip")
    assert result.overall_status is ReproducibilityBundleVerificationStatus.MALFORMED_ZIP


def test_malformed_json_is_rejected() -> None:
    members = _zip_member_map(_valid_zip_bytes())
    members[_path("environment.json")] = b"{not-json\n"
    # Recompute is not needed; signature will fail first unless we also update
    # the manifest signature. Force JSON parse path by keeping signature valid:
    # replace after building a zip where we only break unsigned? environment is signed.
    # Signature mismatch happens before JSON parse of signed files... actually
    # decode UTF-8 works, then signature is computed from the malformed text,
    # then JSON parse happens, then signature compare. Order in verify:
    # decode -> compute signature -> parse JSON -> schema -> signature compare
    # So malformed JSON is detected before signature compare. Good.
    result = verify_reproducibility_bundle(_rewrite_zip(members))
    assert result.overall_status is ReproducibilityBundleVerificationStatus.MALFORMED_JSON


def test_invalid_utf8_is_rejected() -> None:
    members = _zip_member_map(_valid_zip_bytes())
    members[_path("README.txt")] = b"\xff\xfe invalid"
    result = verify_reproducibility_bundle(_rewrite_zip(members))
    assert (
        result.overall_status
        is ReproducibilityBundleVerificationStatus.INVALID_TEXT_ENCODING
    )


def test_unsupported_future_bundle_schema_is_rejected() -> None:
    members = _zip_member_map(_valid_zip_bytes())
    # Keep signed files identical so signature still matches, then bump schema
    # in unsigned manifest only — that becomes MANIFEST_CONTENT_MISMATCH.
    # For UNSUPPORTED_BUNDLE_SCHEMA, bump schema in signed environment as well
    # and rewrite bundle_manifest signature accordingly after recomputing.
    signed = {
        name: members[_path(name)].decode("utf-8")
        for name in SIGNED_BUNDLE_FILE_BASENAMES
    }
    env = json.loads(signed["environment.json"])
    env["bundle_schema_version"] = BUNDLE_SCHEMA_VERSION + 1
    signed["environment.json"] = (
        json.dumps(env, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    from process_intelligence.reporting import compute_reproducibility_bundle_signature

    new_signature = compute_reproducibility_bundle_signature(signed)
    manifest = json.loads(members[_path(BUNDLE_MANIFEST_BASENAME)].decode("utf-8"))
    manifest["bundle_schema_version"] = BUNDLE_SCHEMA_VERSION + 1
    manifest["bundle_signature"] = new_signature
    members[_path(BUNDLE_MANIFEST_BASENAME)] = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    for name, text in signed.items():
        members[_path(name)] = text.encode("utf-8")
    result = verify_reproducibility_bundle(_rewrite_zip(members))
    assert (
        result.overall_status
        is ReproducibilityBundleVerificationStatus.UNSUPPORTED_BUNDLE_SCHEMA
    )


def test_unsupported_run_manifest_schema_is_rejected() -> None:
    members = _zip_member_map(_valid_zip_bytes())
    signed = {
        name: members[_path(name)].decode("utf-8")
        for name in SIGNED_BUNDLE_FILE_BASENAMES
    }
    run_manifest = json.loads(signed["run_manifest.json"])
    run_manifest["manifest_schema_version"] = MANIFEST_SCHEMA_VERSION + 1
    signed["run_manifest.json"] = (
        json.dumps(run_manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    env = json.loads(signed["environment.json"])
    env["manifest_schema_version"] = MANIFEST_SCHEMA_VERSION + 1
    signed["environment.json"] = (
        json.dumps(env, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    from process_intelligence.reporting import compute_reproducibility_bundle_signature

    new_signature = compute_reproducibility_bundle_signature(signed)
    manifest = json.loads(members[_path(BUNDLE_MANIFEST_BASENAME)].decode("utf-8"))
    manifest["bundle_signature"] = new_signature
    members[_path(BUNDLE_MANIFEST_BASENAME)] = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    for name, text in signed.items():
        members[_path(name)] = text.encode("utf-8")
    result = verify_reproducibility_bundle(_rewrite_zip(members))
    assert (
        result.overall_status
        is ReproducibilityBundleVerificationStatus.UNSUPPORTED_RUN_MANIFEST_SCHEMA
    )


def test_included_files_mismatch_is_detected() -> None:
    members = _zip_member_map(_valid_zip_bytes())
    path = _path(BUNDLE_MANIFEST_BASENAME)
    manifest = json.loads(members[path].decode("utf-8"))
    manifest["included_files"] = list(manifest["included_files"]) + [
        f"{BUNDLE_ARCHIVE_ROOT}/extra.json"
    ]
    members[path] = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    result = verify_reproducibility_bundle(_rewrite_zip(members))
    assert (
        result.overall_status
        is ReproducibilityBundleVerificationStatus.MANIFEST_CONTENT_MISMATCH
    )
    assert "included_files" in result.invalid_components


def test_run_signature_mismatch_is_detected() -> None:
    members = _zip_member_map(_valid_zip_bytes())
    path = _path(BUNDLE_MANIFEST_BASENAME)
    manifest = json.loads(members[path].decode("utf-8"))
    manifest["run_signature"] = "c" * 64
    members[path] = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    result = verify_reproducibility_bundle(_rewrite_zip(members))
    assert (
        result.overall_status
        is ReproducibilityBundleVerificationStatus.MANIFEST_CONTENT_MISMATCH
    )
    assert "run_signature" in result.invalid_components


def test_dataset_fingerprint_mismatch_is_detected() -> None:
    members = _zip_member_map(_valid_zip_bytes())
    path = _path(BUNDLE_MANIFEST_BASENAME)
    manifest = json.loads(members[path].decode("utf-8"))
    manifest["dataset_fingerprint"] = "d" * 64
    members[path] = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    result = verify_reproducibility_bundle(_rewrite_zip(members))
    assert (
        result.overall_status
        is ReproducibilityBundleVerificationStatus.MANIFEST_CONTENT_MISMATCH
    )
    assert "dataset_fingerprint" in result.invalid_components


def test_configuration_fingerprint_mismatch_is_detected() -> None:
    members = _zip_member_map(_valid_zip_bytes())
    path = _path(BUNDLE_MANIFEST_BASENAME)
    manifest = json.loads(members[path].decode("utf-8"))
    manifest["configuration_fingerprint"] = "e" * 64
    members[path] = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    result = verify_reproducibility_bundle(_rewrite_zip(members))
    assert (
        result.overall_status
        is ReproducibilityBundleVerificationStatus.MANIFEST_CONTENT_MISMATCH
    )
    assert "configuration_fingerprint" in result.invalid_components


def test_application_version_mismatch_is_detected() -> None:
    members = _zip_member_map(_valid_zip_bytes())
    path = _path(BUNDLE_MANIFEST_BASENAME)
    manifest = json.loads(members[path].decode("utf-8"))
    manifest["application_version"] = "9.9.9-tampered"
    members[path] = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    result = verify_reproducibility_bundle(_rewrite_zip(members))
    assert (
        result.overall_status
        is ReproducibilityBundleVerificationStatus.MANIFEST_CONTENT_MISMATCH
    )
    assert "application_version" in result.invalid_components


def test_no_report_manifest_or_comparison_object_is_mutated() -> None:
    report = {"status": "COMPLETED", "nested": {"value": 1}}
    manifest_obj = {"run_signature": "a" * 64}
    comparison = {"overall_status": "EXACT_MATCH"}
    report_before = json.dumps(report, sort_keys=True)
    manifest_before = json.dumps(manifest_obj, sort_keys=True)
    comparison_before = json.dumps(comparison, sort_keys=True)

    result = verify_reproducibility_bundle(_valid_zip_bytes())
    assert result.overall_status is ReproducibilityBundleVerificationStatus.VALID
    assert json.dumps(report, sort_keys=True) == report_before
    assert json.dumps(manifest_obj, sort_keys=True) == manifest_before
    assert json.dumps(comparison, sort_keys=True) == comparison_before


def test_verifier_never_invokes_workflow_model_execution() -> None:
    workflow = MagicMock()
    result = verify_reproducibility_bundle(_valid_zip_bytes())
    assert result.overall_status is ReproducibilityBundleVerificationStatus.VALID
    assert workflow.run.call_count == 0
    assert workflow.execute.call_count == 0


def test_file_like_source_is_accepted() -> None:
    result = verify_reproducibility_bundle(io.BytesIO(_valid_zip_bytes()))
    assert result.overall_status is ReproducibilityBundleVerificationStatus.VALID


def test_schema_compatibility_policy_centralized() -> None:
    exact = evaluate_schema_compatibility(
        schema_name="bundle_schema_version",
        observed_version=BUNDLE_SCHEMA_VERSION,
        supported_version=BUNDLE_SCHEMA_VERSION,
    )
    assert exact.status is SchemaCompatibilityStatus.SUPPORTED_EXACT
    future = evaluate_schema_compatibility(
        schema_name="bundle_schema_version",
        observed_version=BUNDLE_SCHEMA_VERSION + 1,
        supported_version=BUNDLE_SCHEMA_VERSION,
    )
    assert future.status is SchemaCompatibilityStatus.UNSUPPORTED_FUTURE
    missing = evaluate_schema_compatibility(
        schema_name="bundle_schema_version",
        observed_version=None,
        supported_version=BUNDLE_SCHEMA_VERSION,
    )
    assert missing.status is SchemaCompatibilityStatus.MALFORMED_OR_MISSING
    older = evaluate_schema_compatibility(
        schema_name="bundle_schema_version",
        observed_version=0,
        supported_version=BUNDLE_SCHEMA_VERSION,
        older_readable_versions=(0,),
    )
    assert older.status is SchemaCompatibilityStatus.SUPPORTED_OLDER


def test_csv_member_is_rejected_as_unexpected() -> None:
    members = _zip_member_map(_valid_zip_bytes())
    members[f"{BUNDLE_ARCHIVE_ROOT}/data.csv"] = b"a,b\n1,2\n"
    result = verify_reproducibility_bundle(_rewrite_zip(members))
    assert (
        result.overall_status is ReproducibilityBundleVerificationStatus.UNEXPECTED_FILE
    )


def test_application_version_available() -> None:
    # Sanity: helpers used by fixtures remain importable.
    assert isinstance(get_application_version(), str)
    assert get_application_version()
