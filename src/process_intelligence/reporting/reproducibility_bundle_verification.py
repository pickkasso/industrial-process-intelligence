"""Secure, read-only verification and inspection of reproducibility bundles.

Step 14D: treats uploaded ZIP bytes as untrusted input. Validates archive
safety, required layout, signature integrity, manifest cross-checks, and
schema compatibility. Does not execute analysis, restore configuration,
import datasets, or mutate active workflow state.
"""

from __future__ import annotations

import io
import json
import re
import stat
import zipfile
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import BinaryIO, Final, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from process_intelligence.reporting.reproducibility_bundle import (
    BUNDLE_ARCHIVE_ROOT,
    BUNDLE_MANIFEST_BASENAME,
    BUNDLE_SCHEMA_VERSION,
    SIGNED_BUNDLE_FILE_BASENAMES,
    compute_reproducibility_bundle_signature,
)
from process_intelligence.workflow.run_manifest import MANIFEST_SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Archive safety limits (named constants; treat uploaded ZIPs as untrusted)
# ---------------------------------------------------------------------------

#: Maximum compressed ZIP payload size accepted by the verifier.
MAX_BUNDLE_ZIP_BYTES: Final[int] = 5 * 1024 * 1024

#: Maximum uncompressed size allowed for any single archive member.
MAX_BUNDLE_ENTRY_UNCOMPRESSED_BYTES: Final[int] = 2 * 1024 * 1024

#: Maximum total uncompressed size across all archive members.
MAX_BUNDLE_TOTAL_UNCOMPRESSED_BYTES: Final[int] = 8 * 1024 * 1024

#: Maximum number of archive members (including directories).
MAX_BUNDLE_ARCHIVE_MEMBER_COUNT: Final[int] = 32

#: Reject entries whose uncompressed/compressed ratio exceeds this threshold
#: when the compressed size is positive (ZIP-bomb heuristic).
MAX_BUNDLE_COMPRESSION_RATIO: Final[float] = 100.0

_FORBIDDEN_ROW_PAYLOAD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "rows",
        "row_values",
        "records",
        "csv_bytes",
        "dataset_bytes",
        "raw_csv",
        "data_rows",
    }
)

_FORBIDDEN_DATA_SUFFIXES: Final[tuple[str, ...]] = (
    ".csv",
    ".tsv",
    ".parquet",
    ".pq",
    ".xlsx",
    ".xls",
    ".jsonl",
    ".ndjson",
    ".pkl",
    ".pickle",
    ".joblib",
    ".npy",
    ".npz",
    ".h5",
    ".hdf5",
    ".feather",
    ".arrow",
)

_FORBIDDEN_EXECUTABLE_SUFFIXES: Final[tuple[str, ...]] = (
    ".py",
    ".pyc",
    ".pyo",
    ".pyw",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bat",
    ".cmd",
    ".ps1",
    ".sh",
    ".bash",
    ".js",
    ".mjs",
    ".wasm",
)

_WINDOWS_DRIVE_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z]:")

_EXPECTED_BASENAMES: Final[frozenset[str]] = frozenset(
    (BUNDLE_MANIFEST_BASENAME, *SIGNED_BUNDLE_FILE_BASENAMES)
)

_EXPECTED_RELATIVE_PATHS: Final[tuple[str, ...]] = tuple(
    f"{BUNDLE_ARCHIVE_ROOT}/{name}"
    for name in (BUNDLE_MANIFEST_BASENAME, *SIGNED_BUNDLE_FILE_BASENAMES)
)

_ABSENCE_LIMITATION_NOTE: Final[str] = (
    "Absence of raw dataset files reduces risk but cannot mathematically prove "
    "that arbitrary text values contain no sensitive data."
)

_MESSAGE_VALID = (
    "The reproducibility bundle archive is structurally valid, the signed file "
    "contents match the stored bundle signature, and bundle manifest metadata "
    "matches the signed provenance files."
)
_MESSAGE_VALID_WITH_WARNINGS = (
    "The reproducibility bundle verified successfully with warnings. Review "
    "warnings before trusting the inspected provenance summary."
)
_MESSAGE_SIGNATURE_MISMATCH = (
    "The recomputed bundle signature does not match bundle_manifest."
    "bundle_signature. Signed file contents may have been altered."
)
_MESSAGE_MANIFEST_CONTENT_MISMATCH = (
    "The cryptographic signature of signed files is valid, but "
    "bundle_manifest.json metadata does not match the signed provenance."
)
_MESSAGE_UNSUPPORTED_BUNDLE_SCHEMA = (
    "The bundle schema version is not supported by this application."
)
_MESSAGE_UNSUPPORTED_RUN_MANIFEST_SCHEMA = (
    "The run-manifest schema version is not supported by this application."
)
_MESSAGE_MISSING_REQUIRED_FILE = (
    "One or more required reproducibility-bundle files are missing."
)
_MESSAGE_UNEXPECTED_FILE = (
    "The archive contains unexpected members beyond the Step 14C layout."
)
_MESSAGE_DUPLICATE_FILE = (
    "The archive contains duplicate member names and cannot be trusted."
)
_MESSAGE_MALFORMED_ZIP = "The uploaded bytes are not a readable ZIP archive."
_MESSAGE_MALFORMED_JSON = (
    "One or more required JSON files could not be parsed as strict UTF-8 JSON."
)
_MESSAGE_INVALID_TEXT_ENCODING = (
    "One or more required text files are not valid UTF-8."
)
_MESSAGE_UNSAFE_ARCHIVE = (
    "The archive was rejected by safety checks "
    "(path traversal, symlink, encryption, non-regular entry, or similar)."
)
_MESSAGE_SIZE_LIMIT_EXCEEDED = (
    "The archive exceeded configured compressed or uncompressed size limits."
)
_MESSAGE_UNAVAILABLE = (
    "Reproducibility-bundle verification is unavailable for this input."
)


class ReproducibilityBundleVerificationStatus(StrEnum):
    """Overall outcome of verifying an uploaded reproducibility bundle."""

    VALID = "VALID"
    VALID_WITH_WARNINGS = "VALID_WITH_WARNINGS"
    SIGNATURE_MISMATCH = "SIGNATURE_MISMATCH"
    MANIFEST_CONTENT_MISMATCH = "MANIFEST_CONTENT_MISMATCH"
    UNSUPPORTED_BUNDLE_SCHEMA = "UNSUPPORTED_BUNDLE_SCHEMA"
    UNSUPPORTED_RUN_MANIFEST_SCHEMA = "UNSUPPORTED_RUN_MANIFEST_SCHEMA"
    MISSING_REQUIRED_FILE = "MISSING_REQUIRED_FILE"
    UNEXPECTED_FILE = "UNEXPECTED_FILE"
    DUPLICATE_FILE = "DUPLICATE_FILE"
    MALFORMED_ZIP = "MALFORMED_ZIP"
    MALFORMED_JSON = "MALFORMED_JSON"
    INVALID_TEXT_ENCODING = "INVALID_TEXT_ENCODING"
    UNSAFE_ARCHIVE = "UNSAFE_ARCHIVE"
    SIZE_LIMIT_EXCEEDED = "SIZE_LIMIT_EXCEEDED"
    UNAVAILABLE = "UNAVAILABLE"


class ReproducibilityBundleFileStatus(StrEnum):
    """Per-file / per-component verification status."""

    PRESENT = "PRESENT"
    MISSING = "MISSING"
    UNEXPECTED = "UNEXPECTED"
    DUPLICATE = "DUPLICATE"
    INVALID_ENCODING = "INVALID_ENCODING"
    MALFORMED_JSON = "MALFORMED_JSON"
    UNSAFE = "UNSAFE"
    SIZE_LIMIT_EXCEEDED = "SIZE_LIMIT_EXCEEDED"
    SIGNATURE_INPUT = "SIGNATURE_INPUT"
    NOT_CHECKED = "NOT_CHECKED"


class SchemaCompatibilityStatus(StrEnum):
    """Centralized schema compatibility classification."""

    SUPPORTED_EXACT = "SUPPORTED_EXACT"
    SUPPORTED_OLDER = "SUPPORTED_OLDER"
    UNSUPPORTED_FUTURE = "UNSUPPORTED_FUTURE"
    MALFORMED_OR_MISSING = "MALFORMED_OR_MISSING"


class ReproducibilityBundleFileVerification(BaseModel):
    """Verification outcome for one archive member or expected path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str
    status: ReproducibilityBundleFileStatus
    detail: str | None = None

    @field_validator("relative_path", mode="before")
    @classmethod
    def _validate_path(cls, value: object) -> str:
        if not isinstance(value, str) or value.strip() == "":
            raise ValueError("relative_path must be a non-empty str")
        return value

    @field_validator("detail", mode="before")
    @classmethod
    def _validate_detail(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or value.strip() == "":
            raise ValueError("detail must be a non-empty str when provided")
        return value


class SchemaCompatibilityResult(BaseModel):
    """Compatibility of one schema version field against supported versions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_name: str
    observed_version: int | None
    supported_version: int
    status: SchemaCompatibilityStatus
    detail: str

    @field_validator("schema_name", "detail", mode="before")
    @classmethod
    def _validate_strings(cls, value: object) -> str:
        if not isinstance(value, str) or value.strip() == "":
            raise ValueError("schema compatibility strings must be non-empty str")
        return value


class ReproducibilityBundleInspectionSummary(BaseModel):
    """Safe, read-only provenance summary derived from a verified bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle_schema_version: int
    run_signature: str
    dataset_fingerprint: str | None = None
    configuration_fingerprint: str
    application_version: str
    run_manifest_schema_version: int
    analysis_task: str | None = None
    target_column: str | None = None
    final_modeling_feature_count: int
    final_modeling_feature_names: list[str] = Field(default_factory=list)
    column_roles: dict[str, str] = Field(default_factory=dict)
    platform_system: str | None = None
    python_version: str | None = None
    package_versions: dict[str, str] = Field(default_factory=dict)
    raw_dataset_included: bool = False
    readme_text: str
    absence_check_note: str = _ABSENCE_LIMITATION_NOTE

    @field_validator(
        "run_signature",
        "configuration_fingerprint",
        "application_version",
        "readme_text",
        mode="before",
    )
    @classmethod
    def _validate_required_strings(cls, value: object) -> str:
        if not isinstance(value, str) or value == "":
            raise ValueError("inspection string fields must be non-empty str")
        return value


class ReproducibilityBundleVerificationResult(BaseModel):
    """Informational, read-only result of verifying an uploaded bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    overall_status: ReproducibilityBundleVerificationStatus
    summary: str
    file_verifications: list[ReproducibilityBundleFileVerification] = Field(
        default_factory=list
    )
    expected_bundle_signature: str | None = None
    computed_bundle_signature: str | None = None
    bundle_schema_compatibility: SchemaCompatibilityResult | None = None
    run_manifest_schema_compatibility: SchemaCompatibilityResult | None = None
    warnings: list[str] = Field(default_factory=list)
    inspection: ReproducibilityBundleInspectionSummary | None = None
    invalid_components: list[str] = Field(default_factory=list)

    @field_validator("summary", mode="before")
    @classmethod
    def _validate_summary(cls, value: object) -> str:
        if not isinstance(value, str) or value.strip() == "":
            raise ValueError("summary must be a non-empty str")
        return value

    @field_validator("warnings", "invalid_components", mode="before")
    @classmethod
    def _validate_string_lists(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("string list fields must be list[str]")
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str) or item.strip() == "":
                raise ValueError("string list entries must be non-empty str")
            cleaned.append(item)
        return cleaned

    @field_validator(
        "expected_bundle_signature",
        "computed_bundle_signature",
        mode="before",
    )
    @classmethod
    def _validate_optional_digest(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or value.strip() == "":
            raise ValueError("signature fields must be non-empty str when provided")
        return value

    @model_validator(mode="after")
    def _validate_inspection_consistency(self) -> Self:
        if self.inspection is not None and self.overall_status not in {
            ReproducibilityBundleVerificationStatus.VALID,
            ReproducibilityBundleVerificationStatus.VALID_WITH_WARNINGS,
        }:
            raise ValueError(
                "inspection summary is only allowed for VALID or VALID_WITH_WARNINGS"
            )
        return self


def evaluate_schema_compatibility(
    *,
    schema_name: str,
    observed_version: object,
    supported_version: int,
    older_readable_versions: Sequence[int] = (),
) -> SchemaCompatibilityResult:
    """Classify a schema version against the centralized compatibility policy.

    Exact supported versions are accepted. Explicitly listed older readable
    versions are accepted with ``SUPPORTED_OLDER``. Future / unknown versions
    are rejected (``UNSUPPORTED_FUTURE``). Missing or non-integer versions are
    ``MALFORMED_OR_MISSING``.
    """
    if not isinstance(schema_name, str) or schema_name.strip() == "":
        raise ValueError("schema_name must be a non-empty str")
    if not isinstance(supported_version, int) or isinstance(supported_version, bool):
        raise ValueError("supported_version must be an int")

    if observed_version is None:
        return SchemaCompatibilityResult(
            schema_name=schema_name,
            observed_version=None,
            supported_version=supported_version,
            status=SchemaCompatibilityStatus.MALFORMED_OR_MISSING,
            detail=f"{schema_name} version is missing.",
        )
    if isinstance(observed_version, bool) or not isinstance(observed_version, int):
        return SchemaCompatibilityResult(
            schema_name=schema_name,
            observed_version=None,
            supported_version=supported_version,
            status=SchemaCompatibilityStatus.MALFORMED_OR_MISSING,
            detail=(
                f"{schema_name} version must be an int, "
                f"got {type(observed_version).__name__}."
            ),
        )
    if observed_version == supported_version:
        return SchemaCompatibilityResult(
            schema_name=schema_name,
            observed_version=observed_version,
            supported_version=supported_version,
            status=SchemaCompatibilityStatus.SUPPORTED_EXACT,
            detail=(
                f"{schema_name} version {observed_version} matches the supported "
                f"version {supported_version}."
            ),
        )
    if observed_version in older_readable_versions and observed_version < supported_version:
        return SchemaCompatibilityResult(
            schema_name=schema_name,
            observed_version=observed_version,
            supported_version=supported_version,
            status=SchemaCompatibilityStatus.SUPPORTED_OLDER,
            detail=(
                f"{schema_name} version {observed_version} is older than the "
                f"current supported version {supported_version} but remains readable."
            ),
        )
    if observed_version > supported_version:
        return SchemaCompatibilityResult(
            schema_name=schema_name,
            observed_version=observed_version,
            supported_version=supported_version,
            status=SchemaCompatibilityStatus.UNSUPPORTED_FUTURE,
            detail=(
                f"{schema_name} version {observed_version} is newer than supported "
                f"version {supported_version} and is rejected."
            ),
        )
    return SchemaCompatibilityResult(
        schema_name=schema_name,
        observed_version=observed_version,
        supported_version=supported_version,
        status=SchemaCompatibilityStatus.UNSUPPORTED_FUTURE,
        detail=(
            f"{schema_name} version {observed_version} is not in the supported "
            f"compatibility set (current={supported_version})."
        ),
    )


def _read_source_bytes(source: bytes | bytearray | memoryview | BinaryIO) -> bytes:
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source)
    elif hasattr(source, "read"):
        raw = source.read(MAX_BUNDLE_ZIP_BYTES + 1)
        if not isinstance(raw, (bytes, bytearray)):
            raise TypeError(
                "file-like source must yield bytes, "
                f"got {type(raw).__name__}"
            )
        data = bytes(raw)
    else:
        raise TypeError(
            "source must be bytes or a binary file-like object, "
            f"got {type(source).__name__}"
        )
    return data


def _is_unsafe_member_name(name: str) -> str | None:
    """Return a reason string when ``name`` is an unsafe archive path."""
    if name == "":
        return "empty archive member name"
    if "\x00" in name:
        return "null byte in archive member name"
    if "\\" in name:
        return "backslash path separator is not allowed"
    if name.startswith("/") or name.startswith("\\"):
        return "absolute archive member path is not allowed"
    if _WINDOWS_DRIVE_RE.match(name) is not None:
        return "Windows drive path is not allowed"
    if ":" in name:
        return "colon in archive member path is not allowed"
    parts = name.split("/")
    if any(part == ".." for part in parts):
        return "path traversal ('..') is not allowed"
    if any(part == "" for part in parts[:-1]):
        return "empty path segment is not allowed"
    return None


def _is_symlink_entry(info: zipfile.ZipInfo) -> bool:
    # Unix symbolic links store the mode in the high 16 bits of external_attr.
    mode = (info.external_attr >> 16) & 0xFFFF
    if mode and stat.S_ISLNK(mode):
        return True
    return False


def _is_directory_entry(info: zipfile.ZipInfo) -> bool:
    if info.filename.endswith("/"):
        return True
    mode = (info.external_attr >> 16) & 0xFFFF
    if mode and stat.S_ISDIR(mode):
        return True
    return False


def _is_encrypted_entry(info: zipfile.ZipInfo) -> bool:
    return bool(info.flag_bits & 0x1)


def _suffix_is_forbidden(filename: str) -> str | None:
    lower = filename.lower()
    for suffix in _FORBIDDEN_DATA_SUFFIXES:
        if lower.endswith(suffix):
            return f"forbidden data file type: {suffix}"
    for suffix in _FORBIDDEN_EXECUTABLE_SUFFIXES:
        if lower.endswith(suffix):
            return f"forbidden executable or script type: {suffix}"
    return None


def _result(
    status: ReproducibilityBundleVerificationStatus,
    summary: str,
    *,
    file_verifications: Sequence[ReproducibilityBundleFileVerification] = (),
    expected_bundle_signature: str | None = None,
    computed_bundle_signature: str | None = None,
    bundle_schema_compatibility: SchemaCompatibilityResult | None = None,
    run_manifest_schema_compatibility: SchemaCompatibilityResult | None = None,
    warnings: Sequence[str] = (),
    inspection: ReproducibilityBundleInspectionSummary | None = None,
    invalid_components: Sequence[str] = (),
) -> ReproducibilityBundleVerificationResult:
    return ReproducibilityBundleVerificationResult(
        overall_status=status,
        summary=summary,
        file_verifications=list(file_verifications),
        expected_bundle_signature=expected_bundle_signature,
        computed_bundle_signature=computed_bundle_signature,
        bundle_schema_compatibility=bundle_schema_compatibility,
        run_manifest_schema_compatibility=run_manifest_schema_compatibility,
        warnings=list(warnings),
        inspection=inspection,
        invalid_components=list(invalid_components),
    )


def validate_reproducibility_bundle_archive(
    source: bytes | bytearray | memoryview | BinaryIO,
) -> ReproducibilityBundleVerificationResult:
    """Validate archive safety and Step 14C layout without semantic parsing.

    Reads members in memory through ``zipfile``. Does not extract to disk and
    does not execute or import any uploaded content.
    """
    try:
        data = _read_source_bytes(source)
    except TypeError as exc:
        return _result(
            ReproducibilityBundleVerificationStatus.UNAVAILABLE,
            str(exc),
            invalid_components=["source"],
        )

    if len(data) == 0:
        return _result(
            ReproducibilityBundleVerificationStatus.MALFORMED_ZIP,
            _MESSAGE_MALFORMED_ZIP,
            invalid_components=["archive"],
        )
    if len(data) > MAX_BUNDLE_ZIP_BYTES:
        return _result(
            ReproducibilityBundleVerificationStatus.SIZE_LIMIT_EXCEEDED,
            (
                f"{_MESSAGE_SIZE_LIMIT_EXCEEDED} Compressed size "
                f"{len(data)} exceeds MAX_BUNDLE_ZIP_BYTES="
                f"{MAX_BUNDLE_ZIP_BYTES}."
            ),
            invalid_components=["archive_size"],
        )

    try:
        archive = zipfile.ZipFile(io.BytesIO(data), mode="r")
    except zipfile.BadZipFile:
        return _result(
            ReproducibilityBundleVerificationStatus.MALFORMED_ZIP,
            _MESSAGE_MALFORMED_ZIP,
            invalid_components=["archive"],
        )

    with archive:
        try:
            infos = archive.infolist()
        except (zipfile.BadZipFile, RuntimeError, ValueError):
            return _result(
                ReproducibilityBundleVerificationStatus.MALFORMED_ZIP,
                _MESSAGE_MALFORMED_ZIP,
                invalid_components=["archive"],
            )

        if len(infos) > MAX_BUNDLE_ARCHIVE_MEMBER_COUNT:
            return _result(
                ReproducibilityBundleVerificationStatus.SIZE_LIMIT_EXCEEDED,
                (
                    f"{_MESSAGE_SIZE_LIMIT_EXCEEDED} Member count {len(infos)} "
                    f"exceeds MAX_BUNDLE_ARCHIVE_MEMBER_COUNT="
                    f"{MAX_BUNDLE_ARCHIVE_MEMBER_COUNT}."
                ),
                invalid_components=["archive_member_count"],
            )

        file_verifications: list[ReproducibilityBundleFileVerification] = []
        seen_names: set[str] = set()
        duplicates: list[str] = []
        total_uncompressed = 0

        for info in infos:
            name = info.filename
            if name in seen_names:
                duplicates.append(name)
                file_verifications.append(
                    ReproducibilityBundleFileVerification(
                        relative_path=name,
                        status=ReproducibilityBundleFileStatus.DUPLICATE,
                        detail="duplicate archive member name",
                    )
                )
                continue
            seen_names.add(name)

            unsafe_reason = _is_unsafe_member_name(name)
            if unsafe_reason is not None:
                file_verifications.append(
                    ReproducibilityBundleFileVerification(
                        relative_path=name,
                        status=ReproducibilityBundleFileStatus.UNSAFE,
                        detail=unsafe_reason,
                    )
                )
                return _result(
                    ReproducibilityBundleVerificationStatus.UNSAFE_ARCHIVE,
                    f"{_MESSAGE_UNSAFE_ARCHIVE} ({unsafe_reason}: {name!r})",
                    file_verifications=file_verifications,
                    invalid_components=[name],
                )

            if _is_encrypted_entry(info):
                file_verifications.append(
                    ReproducibilityBundleFileVerification(
                        relative_path=name,
                        status=ReproducibilityBundleFileStatus.UNSAFE,
                        detail="encrypted ZIP entry",
                    )
                )
                return _result(
                    ReproducibilityBundleVerificationStatus.UNSAFE_ARCHIVE,
                    f"{_MESSAGE_UNSAFE_ARCHIVE} (encrypted entry: {name!r})",
                    file_verifications=file_verifications,
                    invalid_components=[name],
                )

            if _is_symlink_entry(info):
                file_verifications.append(
                    ReproducibilityBundleFileVerification(
                        relative_path=name,
                        status=ReproducibilityBundleFileStatus.UNSAFE,
                        detail="symbolic-link archive entry",
                    )
                )
                return _result(
                    ReproducibilityBundleVerificationStatus.UNSAFE_ARCHIVE,
                    f"{_MESSAGE_UNSAFE_ARCHIVE} (symlink: {name!r})",
                    file_verifications=file_verifications,
                    invalid_components=[name],
                )

            if _is_directory_entry(info):
                file_verifications.append(
                    ReproducibilityBundleFileVerification(
                        relative_path=name,
                        status=ReproducibilityBundleFileStatus.UNSAFE,
                        detail="non-regular directory archive entry",
                    )
                )
                return _result(
                    ReproducibilityBundleVerificationStatus.UNSAFE_ARCHIVE,
                    f"{_MESSAGE_UNSAFE_ARCHIVE} (directory entry: {name!r})",
                    file_verifications=file_verifications,
                    invalid_components=[name],
                )

            # Reject non-regular types only when a Unix file-type bit is present.
            # Python's zipfile on Windows often stores permission bits (e.g. 0o600)
            # without S_IFREG; those remain acceptable regular file payloads.
            mode = (info.external_attr >> 16) & 0xFFFF
            file_type = stat.S_IFMT(mode)
            if file_type and not stat.S_ISREG(mode):
                file_verifications.append(
                    ReproducibilityBundleFileVerification(
                        relative_path=name,
                        status=ReproducibilityBundleFileStatus.UNSAFE,
                        detail=f"non-regular archive entry mode={oct(mode)}",
                    )
                )
                return _result(
                    ReproducibilityBundleVerificationStatus.UNSAFE_ARCHIVE,
                    (
                        f"{_MESSAGE_UNSAFE_ARCHIVE} "
                        f"(non-regular entry: {name!r})"
                    ),
                    file_verifications=file_verifications,
                    invalid_components=[name],
                )

            uncompressed = int(info.file_size)
            compressed = int(info.compress_size)
            if uncompressed < 0 or compressed < 0:
                return _result(
                    ReproducibilityBundleVerificationStatus.UNSAFE_ARCHIVE,
                    f"{_MESSAGE_UNSAFE_ARCHIVE} (negative size for {name!r})",
                    invalid_components=[name],
                )
            if uncompressed > MAX_BUNDLE_ENTRY_UNCOMPRESSED_BYTES:
                file_verifications.append(
                    ReproducibilityBundleFileVerification(
                        relative_path=name,
                        status=ReproducibilityBundleFileStatus.SIZE_LIMIT_EXCEEDED,
                        detail=(
                            f"uncompressed size {uncompressed} exceeds "
                            f"MAX_BUNDLE_ENTRY_UNCOMPRESSED_BYTES="
                            f"{MAX_BUNDLE_ENTRY_UNCOMPRESSED_BYTES}"
                        ),
                    )
                )
                return _result(
                    ReproducibilityBundleVerificationStatus.SIZE_LIMIT_EXCEEDED,
                    (
                        f"{_MESSAGE_SIZE_LIMIT_EXCEEDED} Entry {name!r} "
                        f"uncompressed size {uncompressed} exceeds limit."
                    ),
                    file_verifications=file_verifications,
                    invalid_components=[name],
                )

            total_uncompressed += uncompressed
            if total_uncompressed > MAX_BUNDLE_TOTAL_UNCOMPRESSED_BYTES:
                return _result(
                    ReproducibilityBundleVerificationStatus.SIZE_LIMIT_EXCEEDED,
                    (
                        f"{_MESSAGE_SIZE_LIMIT_EXCEEDED} Total uncompressed size "
                        f"exceeds MAX_BUNDLE_TOTAL_UNCOMPRESSED_BYTES="
                        f"{MAX_BUNDLE_TOTAL_UNCOMPRESSED_BYTES}."
                    ),
                    file_verifications=file_verifications,
                    invalid_components=["total_uncompressed_size"],
                )

            if compressed > 0:
                ratio = uncompressed / compressed
                if ratio > MAX_BUNDLE_COMPRESSION_RATIO:
                    file_verifications.append(
                        ReproducibilityBundleFileVerification(
                            relative_path=name,
                            status=ReproducibilityBundleFileStatus.UNSAFE,
                            detail=(
                                f"suspicious compression ratio {ratio:.1f} exceeds "
                                f"MAX_BUNDLE_COMPRESSION_RATIO="
                                f"{MAX_BUNDLE_COMPRESSION_RATIO}"
                            ),
                        )
                    )
                    return _result(
                        ReproducibilityBundleVerificationStatus.UNSAFE_ARCHIVE,
                        (
                            f"{_MESSAGE_UNSAFE_ARCHIVE} "
                            f"(suspicious compression ratio for {name!r})"
                        ),
                        file_verifications=file_verifications,
                        invalid_components=[name],
                    )

            forbidden = _suffix_is_forbidden(name)
            if forbidden is not None:
                file_verifications.append(
                    ReproducibilityBundleFileVerification(
                        relative_path=name,
                        status=ReproducibilityBundleFileStatus.UNEXPECTED,
                        detail=forbidden,
                    )
                )
                return _result(
                    ReproducibilityBundleVerificationStatus.UNEXPECTED_FILE,
                    f"{_MESSAGE_UNEXPECTED_FILE} ({forbidden}: {name!r})",
                    file_verifications=file_verifications,
                    invalid_components=[name],
                )

            # Read bytes to enforce claimed sizes and detect truncated members.
            try:
                payload = archive.read(name)
            except (RuntimeError, zipfile.BadZipFile, ValueError) as exc:
                detail = str(exc)
                if "encrypted" in detail.lower() or "password" in detail.lower():
                    return _result(
                        ReproducibilityBundleVerificationStatus.UNSAFE_ARCHIVE,
                        f"{_MESSAGE_UNSAFE_ARCHIVE} (encrypted entry: {name!r})",
                        invalid_components=[name],
                    )
                return _result(
                    ReproducibilityBundleVerificationStatus.MALFORMED_ZIP,
                    _MESSAGE_MALFORMED_ZIP,
                    invalid_components=[name],
                )
            if len(payload) > MAX_BUNDLE_ENTRY_UNCOMPRESSED_BYTES:
                return _result(
                    ReproducibilityBundleVerificationStatus.SIZE_LIMIT_EXCEEDED,
                    (
                        f"{_MESSAGE_SIZE_LIMIT_EXCEEDED} Entry {name!r} "
                        "expanded beyond the per-entry limit."
                    ),
                    invalid_components=[name],
                )

        if duplicates:
            return _result(
                ReproducibilityBundleVerificationStatus.DUPLICATE_FILE,
                f"{_MESSAGE_DUPLICATE_FILE} Duplicates: {sorted(set(duplicates))}.",
                file_verifications=file_verifications,
                invalid_components=sorted(set(duplicates)),
            )

        expected_set = set(_EXPECTED_RELATIVE_PATHS)
        present = set(seen_names)
        missing = sorted(expected_set - present)
        unexpected = sorted(present - expected_set)

        for path in _EXPECTED_RELATIVE_PATHS:
            if path in present:
                file_verifications.append(
                    ReproducibilityBundleFileVerification(
                        relative_path=path,
                        status=ReproducibilityBundleFileStatus.PRESENT,
                    )
                )
            else:
                file_verifications.append(
                    ReproducibilityBundleFileVerification(
                        relative_path=path,
                        status=ReproducibilityBundleFileStatus.MISSING,
                        detail="required file missing",
                    )
                )
        for path in unexpected:
            file_verifications.append(
                ReproducibilityBundleFileVerification(
                    relative_path=path,
                    status=ReproducibilityBundleFileStatus.UNEXPECTED,
                    detail="unexpected archive member",
                )
            )

        if missing:
            return _result(
                ReproducibilityBundleVerificationStatus.MISSING_REQUIRED_FILE,
                f"{_MESSAGE_MISSING_REQUIRED_FILE} Missing: {missing}.",
                file_verifications=file_verifications,
                invalid_components=missing,
            )
        if unexpected:
            return _result(
                ReproducibilityBundleVerificationStatus.UNEXPECTED_FILE,
                f"{_MESSAGE_UNEXPECTED_FILE} Unexpected: {unexpected}.",
                file_verifications=file_verifications,
                invalid_components=unexpected,
            )

        # Successful structural validation marker (no semantic checks yet).
        return _result(
            ReproducibilityBundleVerificationStatus.VALID,
            "Archive structure and safety checks passed.",
            file_verifications=file_verifications,
        )


def _decode_utf8(payload: bytes, *, path: str) -> tuple[str | None, str | None]:
    try:
        return payload.decode("utf-8"), None
    except UnicodeDecodeError as exc:
        return None, f"{path}: invalid UTF-8 ({exc.reason})"


def _parse_json_object(
    text: str,
    *,
    path: str,
) -> tuple[dict[str, object] | None, str | None]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"{path}: malformed JSON ({exc.msg})"
    if not isinstance(parsed, dict):
        return None, f"{path}: JSON root must be an object"
    # json.loads keys are str for object roots.
    return {str(key): value for key, value in parsed.items()}, None


def _walk_forbidden_row_keys(
    value: object,
    *,
    path: str,
    findings: list[str],
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text in _FORBIDDEN_ROW_PAYLOAD_KEYS:
                findings.append(child_path)
            _walk_forbidden_row_keys(child, path=child_path, findings=findings)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _walk_forbidden_row_keys(child, path=f"{path}[{index}]", findings=findings)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _require_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be int, got {type(value).__name__}")
    return value


def _build_inspection_summary(
    *,
    bundle_manifest: Mapping[str, object],
    run_manifest: Mapping[str, object],
    feature_schema: Mapping[str, object],
    environment: Mapping[str, object],
    readme_text: str,
) -> ReproducibilityBundleInspectionSummary:
    feature_names_raw = feature_schema.get("feature_columns")
    if isinstance(feature_names_raw, list):
        feature_names = [str(item) for item in feature_names_raw]
    else:
        feature_names_raw = run_manifest.get("feature_columns")
        feature_names = (
            [str(item) for item in feature_names_raw]
            if isinstance(feature_names_raw, list)
            else []
        )

    roles_raw = feature_schema.get("column_roles")
    column_roles = (
        {str(key): str(value) for key, value in roles_raw.items()}
        if isinstance(roles_raw, Mapping)
        else {}
    )

    packages_raw = environment.get("package_versions")
    package_versions = (
        {str(key): str(value) for key, value in packages_raw.items()}
        if isinstance(packages_raw, Mapping)
        else {}
    )

    analysis_task = _optional_str(run_manifest.get("resolved_task"))
    if analysis_task is None:
        analysis_task = _optional_str(run_manifest.get("requested_task"))

    feature_count_raw = run_manifest.get("feature_count")
    if isinstance(feature_count_raw, int) and not isinstance(feature_count_raw, bool):
        feature_count = feature_count_raw
    else:
        feature_count = len(feature_names)

    return ReproducibilityBundleInspectionSummary(
        bundle_schema_version=_require_int(
            bundle_manifest["bundle_schema_version"],
            field_name="bundle_schema_version",
        ),
        run_signature=str(bundle_manifest["run_signature"]),
        dataset_fingerprint=_optional_str(bundle_manifest.get("dataset_fingerprint")),
        configuration_fingerprint=str(bundle_manifest["configuration_fingerprint"]),
        application_version=str(bundle_manifest["application_version"]),
        run_manifest_schema_version=_require_int(
            run_manifest["manifest_schema_version"],
            field_name="manifest_schema_version",
        ),
        analysis_task=analysis_task,
        target_column=_optional_str(
            feature_schema.get("target_column", run_manifest.get("target_column"))
        ),
        final_modeling_feature_count=feature_count,
        final_modeling_feature_names=feature_names,
        column_roles=column_roles,
        platform_system=_optional_str(environment.get("platform_system")),
        python_version=_optional_str(environment.get("python_version")),
        package_versions=package_versions,
        raw_dataset_included=False,
        readme_text=readme_text,
        absence_check_note=_ABSENCE_LIMITATION_NOTE,
    )


def parse_reproducibility_bundle(
    source: bytes | bytearray | memoryview | BinaryIO,
) -> ReproducibilityBundleVerificationResult:
    """Validate the archive and decode required members as strict UTF-8/JSON.

    On success, ``overall_status`` is ``VALID`` and decoded contents are not
    returned directly; use ``verify_reproducibility_bundle`` for the full
    semantic verification and inspection summary. This helper is retained as a
    focused structural+encoding gate for callers that only need that stage.
    """
    try:
        data = _read_source_bytes(source)
    except TypeError as exc:
        return _result(
            ReproducibilityBundleVerificationStatus.UNAVAILABLE,
            str(exc),
            invalid_components=["source"],
        )

    structural = validate_reproducibility_bundle_archive(data)
    if (
        structural.overall_status
        is not ReproducibilityBundleVerificationStatus.VALID
    ):
        return structural

    file_verifications = list(structural.file_verifications)
    with zipfile.ZipFile(io.BytesIO(data), mode="r") as archive:
        for basename in (BUNDLE_MANIFEST_BASENAME, *SIGNED_BUNDLE_FILE_BASENAMES):
            path = f"{BUNDLE_ARCHIVE_ROOT}/{basename}"
            payload = archive.read(path)
            text, decode_error = _decode_utf8(payload, path=path)
            if decode_error is not None:
                return _result(
                    ReproducibilityBundleVerificationStatus.INVALID_TEXT_ENCODING,
                    f"{_MESSAGE_INVALID_TEXT_ENCODING} ({decode_error})",
                    file_verifications=[
                        *file_verifications,
                        ReproducibilityBundleFileVerification(
                            relative_path=path,
                            status=ReproducibilityBundleFileStatus.INVALID_ENCODING,
                            detail=decode_error,
                        ),
                    ],
                    invalid_components=[path],
                )
            assert text is not None
            if basename.endswith(".json"):
                _, json_error = _parse_json_object(text, path=path)
                if json_error is not None:
                    return _result(
                        ReproducibilityBundleVerificationStatus.MALFORMED_JSON,
                        f"{_MESSAGE_MALFORMED_JSON} ({json_error})",
                        file_verifications=[
                            *file_verifications,
                            ReproducibilityBundleFileVerification(
                                relative_path=path,
                                status=ReproducibilityBundleFileStatus.MALFORMED_JSON,
                                detail=json_error,
                            ),
                        ],
                        invalid_components=[path],
                    )
    return structural


def verify_reproducibility_bundle(
    source: bytes | bytearray | memoryview | BinaryIO,
) -> ReproducibilityBundleVerificationResult:
    """Verify archive safety, signature, schema compatibility, and metadata.

    Sequence:
    1. validate archive structure and safety limits
    2. read exact signed file bytes and decode as UTF-8
    3. recompute the Step 14C bundle signature from those bytes
    4. compare against ``bundle_manifest.bundle_signature``
    5. parse and semantically validate JSON
    6. cross-check manifest metadata against signed provenance
    """
    # Materialize once so file-like sources can be rewound for staged checks.
    try:
        data = _read_source_bytes(source)
    except TypeError as exc:
        return _result(
            ReproducibilityBundleVerificationStatus.UNAVAILABLE,
            str(exc),
            invalid_components=["source"],
        )

    structural = validate_reproducibility_bundle_archive(data)
    if (
        structural.overall_status
        is not ReproducibilityBundleVerificationStatus.VALID
    ):
        return structural

    file_verifications = list(structural.file_verifications)
    warnings: list[str] = []

    try:
        with zipfile.ZipFile(io.BytesIO(data), mode="r") as archive:
            decoded: dict[str, str] = {}
            for basename in (BUNDLE_MANIFEST_BASENAME, *SIGNED_BUNDLE_FILE_BASENAMES):
                path = f"{BUNDLE_ARCHIVE_ROOT}/{basename}"
                payload = archive.read(path)
                text, decode_error = _decode_utf8(payload, path=path)
                if decode_error is not None:
                    return _result(
                        ReproducibilityBundleVerificationStatus.INVALID_TEXT_ENCODING,
                        f"{_MESSAGE_INVALID_TEXT_ENCODING} ({decode_error})",
                        file_verifications=[
                            *file_verifications,
                            ReproducibilityBundleFileVerification(
                                relative_path=path,
                                status=ReproducibilityBundleFileStatus.INVALID_ENCODING,
                                detail=decode_error,
                            ),
                        ],
                        invalid_components=[path],
                    )
                assert text is not None
                decoded[basename] = text
    except (zipfile.BadZipFile, RuntimeError, ValueError):
        return _result(
            ReproducibilityBundleVerificationStatus.MALFORMED_ZIP,
            _MESSAGE_MALFORMED_ZIP,
            invalid_components=["archive"],
        )

    # Signature uses exact canonical signed file text (not re-serialized JSON).
    signed_contents = {
        name: decoded[name] for name in SIGNED_BUNDLE_FILE_BASENAMES
    }
    try:
        computed_signature = compute_reproducibility_bundle_signature(signed_contents)
    except (TypeError, ValueError) as exc:
        return _result(
            ReproducibilityBundleVerificationStatus.UNAVAILABLE,
            f"{_MESSAGE_UNAVAILABLE} ({exc})",
            invalid_components=["bundle_signature"],
        )

    for name in SIGNED_BUNDLE_FILE_BASENAMES:
        file_verifications.append(
            ReproducibilityBundleFileVerification(
                relative_path=f"{BUNDLE_ARCHIVE_ROOT}/{name}",
                status=ReproducibilityBundleFileStatus.SIGNATURE_INPUT,
                detail="included in bundle signature input set",
            )
        )

    parsed_objects: dict[str, dict[str, object]] = {}
    for basename in (BUNDLE_MANIFEST_BASENAME, *SIGNED_BUNDLE_FILE_BASENAMES):
        if not basename.endswith(".json"):
            continue
        path = f"{BUNDLE_ARCHIVE_ROOT}/{basename}"
        parsed, json_error = _parse_json_object(decoded[basename], path=path)
        if json_error is not None:
            return _result(
                ReproducibilityBundleVerificationStatus.MALFORMED_JSON,
                f"{_MESSAGE_MALFORMED_JSON} ({json_error})",
                file_verifications=[
                    *file_verifications,
                    ReproducibilityBundleFileVerification(
                        relative_path=path,
                        status=ReproducibilityBundleFileStatus.MALFORMED_JSON,
                        detail=json_error,
                    ),
                ],
                computed_bundle_signature=computed_signature,
                invalid_components=[path],
            )
        assert parsed is not None
        parsed_objects[basename] = parsed

    bundle_manifest = parsed_objects[BUNDLE_MANIFEST_BASENAME]
    run_manifest = parsed_objects["run_manifest.json"]
    feature_schema = parsed_objects["feature_schema.json"]
    environment = parsed_objects["environment.json"]
    effective_configuration = parsed_objects["effective_configuration.json"]
    readme_text = decoded["README.txt"]

    expected_signature_raw = bundle_manifest.get("bundle_signature")
    expected_signature = (
        expected_signature_raw
        if isinstance(expected_signature_raw, str)
        else None
    )

    bundle_schema_compat = evaluate_schema_compatibility(
        schema_name="bundle_schema_version",
        observed_version=bundle_manifest.get("bundle_schema_version"),
        supported_version=BUNDLE_SCHEMA_VERSION,
    )
    # Prefer environment / run-manifest versions for run schema when present.
    run_schema_observed = run_manifest.get("manifest_schema_version")
    if run_schema_observed is None:
        run_schema_observed = environment.get("manifest_schema_version")
    run_schema_compat = evaluate_schema_compatibility(
        schema_name="manifest_schema_version",
        observed_version=run_schema_observed,
        supported_version=MANIFEST_SCHEMA_VERSION,
    )

    if bundle_schema_compat.status is SchemaCompatibilityStatus.UNSUPPORTED_FUTURE:
        return _result(
            ReproducibilityBundleVerificationStatus.UNSUPPORTED_BUNDLE_SCHEMA,
            (
                f"{_MESSAGE_UNSUPPORTED_BUNDLE_SCHEMA} "
                f"{bundle_schema_compat.detail}"
            ),
            file_verifications=file_verifications,
            expected_bundle_signature=expected_signature,
            computed_bundle_signature=computed_signature,
            bundle_schema_compatibility=bundle_schema_compat,
            run_manifest_schema_compatibility=run_schema_compat,
            invalid_components=["bundle_schema_version"],
        )
    if bundle_schema_compat.status is SchemaCompatibilityStatus.MALFORMED_OR_MISSING:
        return _result(
            ReproducibilityBundleVerificationStatus.UNSUPPORTED_BUNDLE_SCHEMA,
            (
                f"{_MESSAGE_UNSUPPORTED_BUNDLE_SCHEMA} "
                f"{bundle_schema_compat.detail}"
            ),
            file_verifications=file_verifications,
            expected_bundle_signature=expected_signature,
            computed_bundle_signature=computed_signature,
            bundle_schema_compatibility=bundle_schema_compat,
            run_manifest_schema_compatibility=run_schema_compat,
            invalid_components=["bundle_schema_version"],
        )
    if run_schema_compat.status is SchemaCompatibilityStatus.UNSUPPORTED_FUTURE:
        return _result(
            ReproducibilityBundleVerificationStatus.UNSUPPORTED_RUN_MANIFEST_SCHEMA,
            (
                f"{_MESSAGE_UNSUPPORTED_RUN_MANIFEST_SCHEMA} "
                f"{run_schema_compat.detail}"
            ),
            file_verifications=file_verifications,
            expected_bundle_signature=expected_signature,
            computed_bundle_signature=computed_signature,
            bundle_schema_compatibility=bundle_schema_compat,
            run_manifest_schema_compatibility=run_schema_compat,
            invalid_components=["manifest_schema_version"],
        )
    if run_schema_compat.status is SchemaCompatibilityStatus.MALFORMED_OR_MISSING:
        return _result(
            ReproducibilityBundleVerificationStatus.UNSUPPORTED_RUN_MANIFEST_SCHEMA,
            (
                f"{_MESSAGE_UNSUPPORTED_RUN_MANIFEST_SCHEMA} "
                f"{run_schema_compat.detail}"
            ),
            file_verifications=file_verifications,
            expected_bundle_signature=expected_signature,
            computed_bundle_signature=computed_signature,
            bundle_schema_compatibility=bundle_schema_compat,
            run_manifest_schema_compatibility=run_schema_compat,
            invalid_components=["manifest_schema_version"],
        )
    if bundle_schema_compat.status is SchemaCompatibilityStatus.SUPPORTED_OLDER:
        warnings.append(bundle_schema_compat.detail)
    if run_schema_compat.status is SchemaCompatibilityStatus.SUPPORTED_OLDER:
        warnings.append(run_schema_compat.detail)

    if expected_signature is None or expected_signature != computed_signature:
        return _result(
            ReproducibilityBundleVerificationStatus.SIGNATURE_MISMATCH,
            _MESSAGE_SIGNATURE_MISMATCH,
            file_verifications=file_verifications,
            expected_bundle_signature=expected_signature,
            computed_bundle_signature=computed_signature,
            bundle_schema_compatibility=bundle_schema_compat,
            run_manifest_schema_compatibility=run_schema_compat,
            invalid_components=["bundle_signature"],
        )

    # Signature matched: cross-check unsigned bundle_manifest against signed files.
    mismatches: list[str] = []

    def _expect_equal(label: str, manifest_value: object, actual: object) -> None:
        if manifest_value != actual:
            mismatches.append(label)

    _expect_equal(
        "bundle_schema_version",
        bundle_manifest.get("bundle_schema_version"),
        BUNDLE_SCHEMA_VERSION,
    )
    env_bundle_schema = environment.get("bundle_schema_version")
    if env_bundle_schema is not None:
        _expect_equal(
            "environment.bundle_schema_version",
            env_bundle_schema,
            bundle_manifest.get("bundle_schema_version"),
        )

    _expect_equal(
        "run_signature",
        bundle_manifest.get("run_signature"),
        run_manifest.get("run_signature"),
    )
    _expect_equal(
        "dataset_fingerprint",
        bundle_manifest.get("dataset_fingerprint"),
        run_manifest.get("dataset_fingerprint"),
    )
    _expect_equal(
        "configuration_fingerprint",
        bundle_manifest.get("configuration_fingerprint"),
        run_manifest.get("configuration_fingerprint"),
    )
    _expect_equal(
        "application_version",
        bundle_manifest.get("application_version"),
        run_manifest.get("application_version"),
    )
    env_app_version = environment.get("application_version")
    if env_app_version is not None:
        _expect_equal(
            "environment.application_version",
            env_app_version,
            bundle_manifest.get("application_version"),
        )

    env_manifest_schema = environment.get("manifest_schema_version")
    if env_manifest_schema is not None:
        _expect_equal(
            "environment.manifest_schema_version",
            env_manifest_schema,
            run_manifest.get("manifest_schema_version"),
        )

    included_files = bundle_manifest.get("included_files")
    actual_included = list(_EXPECTED_RELATIVE_PATHS)
    if not isinstance(included_files, list):
        mismatches.append("included_files")
    else:
        included_normalized = [str(item) for item in included_files]
        if included_normalized != actual_included:
            mismatches.append("included_files")

    # Ensure expected basenames cover only the Step 14C contract.
    if set(_EXPECTED_BASENAMES) != {
        BUNDLE_MANIFEST_BASENAME,
        *SIGNED_BUNDLE_FILE_BASENAMES,
    }:
        mismatches.append("expected_basenames")

    if mismatches:
        return _result(
            ReproducibilityBundleVerificationStatus.MANIFEST_CONTENT_MISMATCH,
            (
                f"{_MESSAGE_MANIFEST_CONTENT_MISMATCH} "
                f"Mismatched fields: {mismatches}."
            ),
            file_verifications=file_verifications,
            expected_bundle_signature=expected_signature,
            computed_bundle_signature=computed_signature,
            bundle_schema_compatibility=bundle_schema_compat,
            run_manifest_schema_compatibility=run_schema_compat,
            invalid_components=mismatches,
        )

    # Conservative raw-data exclusion inspection on expected JSON payloads.
    forbidden_findings: list[str] = []
    json_payloads: tuple[tuple[str, Mapping[str, object]], ...] = (
        (BUNDLE_MANIFEST_BASENAME, bundle_manifest),
        ("run_manifest.json", run_manifest),
        ("effective_configuration.json", effective_configuration),
        ("feature_schema.json", feature_schema),
        ("environment.json", environment),
    )
    for json_basename, json_payload in json_payloads:
        _walk_forbidden_row_keys(
            json_payload,
            path=f"{BUNDLE_ARCHIVE_ROOT}/{json_basename}",
            findings=forbidden_findings,
        )
    if forbidden_findings:
        warnings.append(
            "Prohibited row-level payload fields were found in signed JSON: "
            + ", ".join(forbidden_findings)
        )

    try:
        inspection = _build_inspection_summary(
            bundle_manifest=bundle_manifest,
            run_manifest=run_manifest,
            feature_schema=feature_schema,
            environment=environment,
            readme_text=readme_text,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return _result(
            ReproducibilityBundleVerificationStatus.MANIFEST_CONTENT_MISMATCH,
            (
                f"{_MESSAGE_MANIFEST_CONTENT_MISMATCH} "
                f"Inspection summary could not be built ({exc})."
            ),
            file_verifications=file_verifications,
            expected_bundle_signature=expected_signature,
            computed_bundle_signature=computed_signature,
            bundle_schema_compatibility=bundle_schema_compat,
            run_manifest_schema_compatibility=run_schema_compat,
            invalid_components=["inspection"],
        )

    if warnings:
        return _result(
            ReproducibilityBundleVerificationStatus.VALID_WITH_WARNINGS,
            _MESSAGE_VALID_WITH_WARNINGS,
            file_verifications=file_verifications,
            expected_bundle_signature=expected_signature,
            computed_bundle_signature=computed_signature,
            bundle_schema_compatibility=bundle_schema_compat,
            run_manifest_schema_compatibility=run_schema_compat,
            warnings=warnings,
            inspection=inspection,
        )

    return _result(
        ReproducibilityBundleVerificationStatus.VALID,
        _MESSAGE_VALID,
        file_verifications=file_verifications,
        expected_bundle_signature=expected_signature,
        computed_bundle_signature=computed_signature,
        bundle_schema_compatibility=bundle_schema_compat,
        run_manifest_schema_compatibility=run_schema_compat,
        warnings=[],
        inspection=inspection,
    )


def inspect_reproducibility_bundle(
    source: bytes | bytearray | memoryview | BinaryIO,
) -> ReproducibilityBundleVerificationResult:
    """Verify a bundle and return the read-only inspection result.

    Equivalent to ``verify_reproducibility_bundle``. Successful results include
    a safe ``inspection`` summary. Never restores configuration, imports a
    dataset, or executes analysis.
    """
    return verify_reproducibility_bundle(source)
