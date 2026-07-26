"""Deterministic reproducibility-bundle export for completed workflow reports.

Step 14C: builds a downloadable ZIP describing declared-input provenance for a
completed analysis. Does not embed raw dataset rows, does not rerun analysis,
and does not mutate stored reports or run manifests.

ZIP container metadata (entry timestamps, compressor details) may differ across
environments. Canonical file contents and the bundle signature are deterministic
for identical provenance inputs.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import platform
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from process_intelligence.workflow.run_manifest import (
    MANIFEST_SCHEMA_VERSION,
    AnalysisRunManifest,
    analysis_run_manifest_to_json,
    build_configuration_fingerprint_payload,
    canonicalize_fingerprint_value,
    deterministic_fingerprint_json,
    get_application_version,
)
from process_intelligence.workflow.schemas import (
    AnalysisWorkflowPolicy,
    AnalysisWorkflowRequest,
)

BUNDLE_SCHEMA_VERSION: Literal[1] = 1

# Public layout contract shared with Step 14D verification.
BUNDLE_ARCHIVE_ROOT: Final[str] = "reproducibility_bundle"
SIGNED_BUNDLE_FILE_BASENAMES: Final[tuple[str, ...]] = (
    "run_manifest.json",
    "effective_configuration.json",
    "feature_schema.json",
    "environment.json",
    "README.txt",
)
BUNDLE_MANIFEST_BASENAME: Final[str] = "bundle_manifest.json"

_BUNDLE_ROOT: Final[str] = BUNDLE_ARCHIVE_ROOT
_SIGNED_FILE_NAMES: Final[tuple[str, ...]] = SIGNED_BUNDLE_FILE_BASENAMES
_BUNDLE_MANIFEST_NAME: Final[str] = BUNDLE_MANIFEST_BASENAME
_DEFAULT_PACKAGE_NAMES: Final[tuple[str, ...]] = (
    "industrial-process-intelligence",
    "pydantic",
    "polars",
    "numpy",
    "scikit-learn",
    "scipy",
    "pandas",
    "pyarrow",
    "streamlit",
    "joblib",
)

_README_TEXT: Final[str] = """\
Industrial Process Intelligence — Reproducibility Bundle
========================================================

This archive exports declared-input provenance for one completed analysis run.

What this bundle proves
-----------------------
- The stored immutable run manifest (fingerprints, run signature, feature list)
- The effective workflow configuration used for that completed run
- The final modeling feature schema metadata (roles/dtypes when available)
- Application and runtime environment metadata recorded for the run
- A deterministic bundle signature over the signed file contents

What this bundle does NOT prove
-------------------------------
- Causal correctness of diagnosis or recommendations
- Bit-for-bit deterministic model training or scoring replay
- That re-running analysis on another machine will produce identical metrics
- That the operator applied the recommended actions safely

Dataset privacy
---------------
The raw dataset is intentionally NOT included. Only the dataset fingerprint
(SHA-256 of the CSV bytes used for the run) is recorded. Matching fingerprints
indicate the same declared input bytes; they do not embed or recover row values.

How to interpret identifiers
----------------------------
- dataset_fingerprint: digest of the CSV bytes used for the completed run
- configuration_fingerprint: digest of the effective workflow request/policy
- run_signature: digest of dataset + configuration + application version +
  manifest schema version
- bundle_signature: digest of the canonical signed files in this archive

Relation to the Streamlit UI
----------------------------
Run provenance shows the stored manifest. The current-setup comparison
(Step 14B) checks whether the live UI dataset/configuration still match that
stored run. This bundle (Step 14C) exports the stored run's provenance for
offline inspection. It does not restore configuration or replay execution.

Determinism note
----------------
Canonical JSON/text file contents and bundle_signature are deterministic for
identical provenance inputs. ZIP container metadata (for example entry
timestamps) may differ and is not part of the signature.
"""


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_non_empty_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be str, got {type(value).__name__}")
    if value == "" or value.strip() == "":
        raise ValueError(f"{field_name} must be a non-empty, non-whitespace string")
    return value


def _canonical_pretty_json(payload: Mapping[str, object]) -> str:
    canonical = canonicalize_fingerprint_value(dict(payload))
    if not isinstance(canonical, dict):
        raise ValueError("payload must canonicalize to a mapping")
    return (
        json.dumps(
            canonical,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _package_version(distribution_name: str) -> str | None:
    try:
        version = importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    if isinstance(version, str) and version.strip():
        return version.strip()
    return None


def collect_relevant_package_versions(
    package_names: Sequence[str] | None = None,
) -> dict[str, str]:
    """Return installed versions for relevant packages (missing packages omitted)."""
    names = (
        list(_DEFAULT_PACKAGE_NAMES)
        if package_names is None
        else [str(name) for name in package_names]
    )
    versions: dict[str, str] = {}
    for name in sorted(set(names)):
        version = _package_version(name)
        if version is not None:
            versions[name] = version
    return versions


def build_environment_payload(
    *,
    application_version: str | None = None,
    manifest_schema_version: int = MANIFEST_SCHEMA_VERSION,
    bundle_schema_version: int = BUNDLE_SCHEMA_VERSION,
    python_version: str | None = None,
    platform_system: str | None = None,
    platform_release: str | None = None,
    platform_machine: str | None = None,
    package_versions: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build runtime/environment provenance for the bundle (no paths)."""
    version = (
        get_application_version()
        if application_version is None
        else application_version.strip()
    )
    if not version:
        raise ValueError("application_version must be a non-empty str")
    packages = (
        collect_relevant_package_versions()
        if package_versions is None
        else {str(key): str(value) for key, value in package_versions.items()}
    )
    return {
        "application_version": version,
        "bundle_schema_version": int(bundle_schema_version),
        "manifest_schema_version": int(manifest_schema_version),
        "package_versions": dict(sorted(packages.items(), key=lambda item: item[0])),
        "platform_machine": (
            platform.machine() if platform_machine is None else platform_machine
        ),
        "platform_release": (
            platform.release() if platform_release is None else platform_release
        ),
        "platform_system": (
            platform.system() if platform_system is None else platform_system
        ),
        "python_version": (
            platform.python_version() if python_version is None else python_version
        ),
    }


def build_effective_configuration_payload(
    request: AnalysisWorkflowRequest,
    policy: AnalysisWorkflowPolicy,
    *,
    feature_columns: Sequence[str] | None = None,
) -> dict[str, object]:
    """Return the canonical effective configuration used for fingerprinting.

    Excludes ``csv_path``, request metadata, DataFrames, models, timestamps,
    and presentation-only / transient UI state.
    """
    return build_configuration_fingerprint_payload(
        request,
        policy,
        feature_columns=feature_columns,
    )


def build_feature_schema_payload(
    *,
    target_column: str | None = None,
    feature_columns: Sequence[str],
    column_roles: Mapping[str, str] | None = None,
    dtypes: Mapping[str, str] | None = None,
    excluded_columns: Sequence[str] | None = None,
    identifier_columns: Sequence[str] | None = None,
    timestamp_column: str | None = None,
) -> dict[str, object]:
    """Build schema metadata for modeling columns (no row-level values)."""
    features = [str(name) for name in feature_columns]
    roles = {
        str(name): str(role)
        for name, role in sorted((column_roles or {}).items(), key=lambda item: item[0])
    }
    dtype_map = {
        str(name): str(dtype)
        for name, dtype in sorted((dtypes or {}).items(), key=lambda item: item[0])
    }
    return {
        "column_roles": roles,
        "dtypes": dtype_map,
        "excluded_columns": [str(name) for name in (excluded_columns or ())],
        "feature_columns": features,
        "identifier_columns": [str(name) for name in (identifier_columns or ())],
        "target_column": target_column,
        "timestamp_column": timestamp_column,
    }


def select_relevant_column_dtypes(
    all_dtypes: Mapping[str, str],
    *,
    feature_columns: Sequence[str],
    target_column: str | None = None,
    timestamp_column: str | None = None,
    identifier_columns: Sequence[str] | None = None,
    excluded_columns: Sequence[str] | None = None,
) -> dict[str, str]:
    """Keep dtype summaries only for columns referenced by the run schema."""
    relevant: set[str] = set(feature_columns)
    if target_column is not None:
        relevant.add(target_column)
    if timestamp_column is not None:
        relevant.add(timestamp_column)
    relevant.update(identifier_columns or ())
    relevant.update(excluded_columns or ())
    return {
        name: str(all_dtypes[name])
        for name in sorted(relevant)
        if name in all_dtypes
    }


class ReproducibilityBundleFile(BaseModel):
    """One text file inside a reproducibility bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str
    content: str
    signed: bool

    @field_validator("relative_path", "content", mode="before")
    @classmethod
    def _validate_strings(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(
                f"bundle file string fields must be str, got {type(value).__name__}"
            )
        return value

    @field_validator("relative_path", mode="after")
    @classmethod
    def _validate_relative_path(cls, value: str) -> str:
        text = value.strip()
        if text == "":
            raise ValueError("relative_path must be non-empty")
        if text.startswith("/") or text.startswith("\\") or ".." in Path(text).parts:
            raise ValueError(f"relative_path must be a safe relative path, got {value!r}")
        return text

    @field_validator("signed", mode="before")
    @classmethod
    def _validate_signed(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError(
                "signed must be a bool (0/1 and strings rejected), "
                f"got {type(value).__name__}"
            )
        return value


class ReproducibilityBundleManifest(BaseModel):
    """Top-level metadata for one reproducibility bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle_schema_version: Literal[1] = BUNDLE_SCHEMA_VERSION
    bundle_signature: str
    included_files: list[str] = Field(default_factory=list)
    application_version: str
    run_signature: str
    dataset_fingerprint: str | None = None
    configuration_fingerprint: str
    generated_at: str | None = None

    @field_validator("bundle_schema_version", mode="before")
    @classmethod
    def _validate_schema_version(cls, value: object) -> int:
        if value != BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                f"bundle_schema_version must be {BUNDLE_SCHEMA_VERSION}, got {value!r}"
            )
        return BUNDLE_SCHEMA_VERSION

    @field_validator(
        "bundle_signature",
        "application_version",
        "run_signature",
        "configuration_fingerprint",
        mode="before",
    )
    @classmethod
    def _validate_required_strings(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="bundle manifest string")

    @field_validator("dataset_fingerprint", "generated_at", mode="before")
    @classmethod
    def _validate_optional_strings(cls, value: object) -> str | None:
        if value is None:
            return None
        return _require_non_empty_str(value, field_name="optional bundle string")

    @field_validator("included_files", mode="before")
    @classmethod
    def _validate_included_files_before(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"included_files must be list[str], got {type(value).__name__}"
            )
        return list(value)

    @field_validator("included_files", mode="after")
    @classmethod
    def _validate_included_files(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = _require_non_empty_str(item, field_name="included_files")
            if text in seen:
                raise ValueError(f"included_files must not contain duplicates: {text!r}")
            seen.add(text)
            cleaned.append(text)
        return cleaned

    @model_validator(mode="after")
    def _validate_digest_lengths(self) -> Self:
        for field_name in (
            "bundle_signature",
            "run_signature",
            "configuration_fingerprint",
        ):
            digest = getattr(self, field_name)
            if len(digest) != 64:
                raise ValueError(f"{field_name} must be a 64-character SHA-256 hex digest")
        if self.dataset_fingerprint is not None and len(self.dataset_fingerprint) != 64:
            raise ValueError(
                "dataset_fingerprint must be a 64-character SHA-256 hex digest when set"
            )
        return self


class ReproducibilityBundle(BaseModel):
    """In-memory reproducibility bundle with deterministic file contents."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle_manifest: ReproducibilityBundleManifest
    files: list[ReproducibilityBundleFile] = Field(default_factory=list)

    @field_validator("files", mode="before")
    @classmethod
    def _validate_files_before(cls, value: object) -> list[ReproducibilityBundleFile]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"files must be list[ReproducibilityBundleFile], got {type(value).__name__}"
            )
        return list(value)

    @field_validator("files", mode="after")
    @classmethod
    def _validate_files(
        cls,
        value: list[ReproducibilityBundleFile],
    ) -> list[ReproducibilityBundleFile]:
        copied: list[ReproducibilityBundleFile] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, ReproducibilityBundleFile):
                raise ValueError(
                    "files entries must be ReproducibilityBundleFile, "
                    f"got {type(item).__name__}"
                )
            if item.relative_path in seen:
                raise ValueError(
                    f"files must not contain duplicate paths: {item.relative_path!r}"
                )
            seen.add(item.relative_path)
            copied.append(item)
        return copied

    def file_map(self) -> dict[str, str]:
        """Return ``relative_path -> content`` for all bundle files."""
        return {item.relative_path: item.content for item in self.files}

    def signed_file_map(self) -> dict[str, str]:
        """Return ``basename -> content`` for signed bundle files only."""
        mapping: dict[str, str] = {}
        for item in self.files:
            if not item.signed:
                continue
            mapping[Path(item.relative_path).name] = item.content
        return mapping


def compute_reproducibility_bundle_signature(
    signed_file_contents: Mapping[str, str],
) -> str:
    """Compute the bundle signature from canonical signed file contents.

    ``generated_at``, ZIP filenames, temporary directories, UI labels, and
    download timestamps must not be included in ``signed_file_contents``.
    """
    if not isinstance(signed_file_contents, Mapping):
        raise TypeError(
            "signed_file_contents must be a mapping, "
            f"got {type(signed_file_contents).__name__}"
        )
    expected = set(_SIGNED_FILE_NAMES)
    provided = set(signed_file_contents.keys())
    if provided != expected:
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        raise ValueError(
            "signed_file_contents keys must exactly match the signed bundle "
            f"files; missing={missing}, unexpected={unexpected}"
        )
    digests = {
        name: hashlib.sha256(signed_file_contents[name].encode("utf-8")).hexdigest()
        for name in _SIGNED_FILE_NAMES
    }
    return _sha256_hex(deterministic_fingerprint_json(digests))


def _bundle_relative_path(filename: str) -> str:
    return f"{_BUNDLE_ROOT}/{filename}"


def _format_generated_at(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError("generated_at must be timezone-aware when provided")
    return value.isoformat()


def build_reproducibility_bundle(
    *,
    run_manifest: AnalysisRunManifest,
    effective_configuration: Mapping[str, object],
    feature_schema: Mapping[str, object],
    environment: Mapping[str, object] | None = None,
    application_version: str | None = None,
    generated_at: datetime | None = None,
    readme_text: str | None = None,
) -> ReproducibilityBundle:
    """Build a deterministic reproducibility bundle from stored provenance inputs.

    Does not mutate ``run_manifest`` or the provided mappings. Does not execute
    workflow stages or embed raw dataset rows.
    """
    if not isinstance(run_manifest, AnalysisRunManifest):
        raise TypeError(
            "run_manifest must be AnalysisRunManifest, "
            f"got {type(run_manifest).__name__}"
        )
    if not isinstance(effective_configuration, Mapping):
        raise TypeError(
            "effective_configuration must be a mapping, "
            f"got {type(effective_configuration).__name__}"
        )
    if not isinstance(feature_schema, Mapping):
        raise TypeError(
            "feature_schema must be a mapping, "
            f"got {type(feature_schema).__name__}"
        )

    # Snapshot before any serialization so callers retain original objects.
    manifest_copy = run_manifest.model_copy(deep=True)
    config_payload = dict(effective_configuration)
    schema_payload = dict(feature_schema)

    app_version = (
        manifest_copy.application_version
        if application_version is None
        else application_version.strip()
    )
    if not app_version:
        raise ValueError("application_version must be a non-empty str")

    env_payload = (
        build_environment_payload(application_version=app_version)
        if environment is None
        else dict(environment)
    )
    # Keep environment identity fields aligned with the stored run when omitted.
    env_payload.setdefault("application_version", app_version)
    env_payload.setdefault(
        "manifest_schema_version",
        manifest_copy.manifest_schema_version,
    )
    env_payload.setdefault("bundle_schema_version", BUNDLE_SCHEMA_VERSION)

    run_manifest_json = analysis_run_manifest_to_json(manifest_copy)
    effective_configuration_json = _canonical_pretty_json(config_payload)
    feature_schema_json = _canonical_pretty_json(schema_payload)
    environment_json = _canonical_pretty_json(env_payload)
    readme = _README_TEXT if readme_text is None else readme_text
    if not isinstance(readme, str) or readme == "":
        raise ValueError("readme_text must be a non-empty str when provided")
    if not readme.endswith("\n"):
        readme = readme + "\n"

    signed_contents = {
        "run_manifest.json": run_manifest_json,
        "effective_configuration.json": effective_configuration_json,
        "feature_schema.json": feature_schema_json,
        "environment.json": environment_json,
        "README.txt": readme,
    }
    bundle_signature = compute_reproducibility_bundle_signature(signed_contents)

    included_files = [
        _bundle_relative_path(_BUNDLE_MANIFEST_NAME),
        *[_bundle_relative_path(name) for name in _SIGNED_FILE_NAMES],
    ]
    bundle_manifest = ReproducibilityBundleManifest(
        bundle_schema_version=BUNDLE_SCHEMA_VERSION,
        bundle_signature=bundle_signature,
        included_files=included_files,
        application_version=app_version,
        run_signature=manifest_copy.run_signature,
        dataset_fingerprint=manifest_copy.dataset_fingerprint,
        configuration_fingerprint=manifest_copy.configuration_fingerprint,
        generated_at=_format_generated_at(generated_at),
    )
    bundle_manifest_json = _canonical_pretty_json(
        bundle_manifest.model_dump(mode="json")
    )

    files = [
        ReproducibilityBundleFile(
            relative_path=_bundle_relative_path(_BUNDLE_MANIFEST_NAME),
            content=bundle_manifest_json,
            signed=False,
        )
    ]
    for name in _SIGNED_FILE_NAMES:
        files.append(
            ReproducibilityBundleFile(
                relative_path=_bundle_relative_path(name),
                content=signed_contents[name],
                signed=True,
            )
        )
    return ReproducibilityBundle(bundle_manifest=bundle_manifest, files=files)


def serialize_reproducibility_bundle(bundle: ReproducibilityBundle) -> bytes:
    """Serialize a bundle to ZIP bytes.

    File contents are deterministic. ZIP entry metadata is written with a fixed
    DOS epoch timestamp, but byte-for-byte ZIP identity across compressor
    implementations is not guaranteed and is not part of the bundle signature.
    """
    if not isinstance(bundle, ReproducibilityBundle):
        raise TypeError(
            "bundle must be ReproducibilityBundle, "
            f"got {type(bundle).__name__}"
        )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(bundle.files, key=lambda file: file.relative_path):
            info = zipfile.ZipInfo(filename=item.relative_path)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, item.content.encode("utf-8"))
    return buffer.getvalue()


def reproducibility_bundle_download_filename(run_manifest: AnalysisRunManifest) -> str:
    """Return a filesystem-safe ZIP filename that includes a run identifier."""
    if not isinstance(run_manifest, AnalysisRunManifest):
        raise TypeError(
            "run_manifest must be AnalysisRunManifest, "
            f"got {type(run_manifest).__name__}"
        )
    run_id = run_manifest.run_signature[:12]
    return f"reproducibility_bundle_{run_id}.zip"


def assert_bundle_has_no_forbidden_objects(payload: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` when dumped bundle data retains forbidden objects."""
    forbidden_type_names = {
        "DataFrame",
        "Series",
        "ndarray",
        "Path",
        "WindowsPath",
        "PosixPath",
    }

    def _walk(value: object, *, path: str) -> None:
        type_name = type(value).__name__
        if type_name in forbidden_type_names or isinstance(value, Path):
            raise ValueError(f"forbidden object at {path}: {type_name}")
        if isinstance(value, BaseModel):
            raise ValueError(f"raw BaseModel retained at {path}: {type_name}")
        if isinstance(value, Mapping):
            for key, child in value.items():
                _walk(child, path=f"{path}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                _walk(child, path=f"{path}[{index}]")

    _walk(dict(payload), path="bundle")


def bundle_contains_raw_dataset_rows(bundle: ReproducibilityBundle) -> bool:
    """Heuristic guard used by tests: reject obvious row-payload leakage."""
    forbidden_keys = {
        "rows",
        "row_values",
        "records",
        "csv_bytes",
        "dataset_bytes",
        "raw_csv",
        "data_rows",
    }
    for item in bundle.files:
        if item.relative_path.endswith(".json"):
            try:
                parsed = json.loads(item.content)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                if any(key in parsed for key in forbidden_keys):
                    return True
                # Nested dataset-like blobs.
                text = json.dumps(parsed, ensure_ascii=False)
                if '"csv_bytes"' in text or '"dataset_bytes"' in text:
                    return True
        lower = item.content.lower()
        if "csv_bytes" in lower or "dataset_bytes" in lower:
            return True
    return False
