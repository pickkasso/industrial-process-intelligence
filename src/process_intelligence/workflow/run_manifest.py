"""Typed analysis-run manifest for reproducibility and auditability (Step 14A).

Fingerprints the effective workflow request/policy, dataset content digest,
application version, and manifest schema. Does not embed raw data, trained
models, secrets, local absolute paths, or uploaded file contents.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.recommendation.enums import RecommendationStatus
from process_intelligence.workflow.enums import (
    AnalysisExecutionMode,
    AnalysisWorkflowStage,
    AnalysisWorkflowStatus,
    OperatingPointSelectionMode,
)
from process_intelligence.workflow.schemas import (
    AnalysisWorkflowPolicy,
    AnalysisWorkflowRequest,
    AnalysisWorkflowStageRecord,
)

MANIFEST_SCHEMA_VERSION: Literal[1] = 1
APPLICATION_VERSION_FALLBACK = "0+unknown"
PACKAGE_DISTRIBUTION_NAME = "industrial-process-intelligence"
_SHORT_MESSAGE_LIMIT = 240
_STAGE_CODE_KEYS: tuple[str, ...] = (
    "recommendation_status",
    "target_refusal_code",
    "refusal_code",
    "reason_code",
)


def get_application_version() -> str:
    """Return the installed project version, or a deterministic fallback.

    Prefers ``importlib.metadata.version`` for the distribution name. Does not
    read Git state, branch names, or filesystem paths.
    """
    try:
        version = importlib.metadata.version(PACKAGE_DISTRIBUTION_NAME)
    except importlib.metadata.PackageNotFoundError:
        try:
            from process_intelligence import __version__ as package_version
        except ImportError:
            return APPLICATION_VERSION_FALLBACK
        if isinstance(package_version, str) and package_version.strip():
            return package_version.strip()
        return APPLICATION_VERSION_FALLBACK
    if isinstance(version, str) and version.strip():
        return version.strip()
    return APPLICATION_VERSION_FALLBACK


def _require_non_empty_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be str, got {type(value).__name__}")
    if value == "" or value.strip() == "":
        raise ValueError(f"{field_name} must be a non-empty, non-whitespace string")
    return value


def _require_optional_non_empty_str(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_non_empty_str(value, field_name=field_name)


def _require_strict_bool(value: object, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(
            f"{field_name} must be a bool (0/1 and strings rejected), "
            f"got {type(value).__name__}"
        )
    return value


def _require_strict_int_ge0(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{field_name} must be an int >= 0 (bool not allowed), "
            f"got {type(value).__name__}"
        )
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0, got {value}")
    return value


def _require_finite_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a finite float "
            f"(bool not allowed), got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return number


def _require_non_negative_finite_float(value: object, *, field_name: str) -> float:
    number = _require_finite_float(value, field_name=field_name)
    if number < 0.0:
        raise ValueError(f"{field_name} must be >= 0, got {number}")
    return number


def _require_timezone_aware(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonicalize_fingerprint_value(value: object) -> object:
    """Normalize a value for deterministic fingerprint JSON.

    Enums become their stable string values. Mapping keys are sorted. List and
    tuple order is preserved when order is semantically meaningful. Paths and
    other non-JSON-safe objects are rejected.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"fingerprint float must be finite, got {value!r}")
        return value
    if isinstance(value, Enum):
        enum_value = value.value
        if isinstance(enum_value, (str, int, float, bool)) or enum_value is None:
            return canonicalize_fingerprint_value(enum_value)
        return str(enum_value)
    if isinstance(value, Path):
        raise ValueError(
            "fingerprint payload must not include filesystem paths "
            f"(got Path {value.name!r})"
        )
    if isinstance(value, BaseModel):
        return canonicalize_fingerprint_value(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        items: list[tuple[str, object]] = []
        for key, raw in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    "fingerprint mapping keys must be str, "
                    f"got {type(key).__name__}"
                )
            items.append((key, canonicalize_fingerprint_value(raw)))
        items.sort(key=lambda item: item[0])
        return {key: item for key, item in items}
    if isinstance(value, tuple):
        return [canonicalize_fingerprint_value(item) for item in value]
    if isinstance(value, list):
        return [canonicalize_fingerprint_value(item) for item in value]
    raise ValueError(
        "fingerprint payload contains unsupported type "
        f"{type(value).__name__}; DataFrames, models, and paths are forbidden"
    )


def deterministic_fingerprint_json(payload: Mapping[str, object]) -> str:
    """Serialize a fingerprint payload as canonical JSON text."""
    canonical = canonicalize_fingerprint_value(dict(payload))
    if not isinstance(canonical, dict):
        raise ValueError("fingerprint payload must canonicalize to a mapping")
    return json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def build_configuration_fingerprint_payload(
    request: AnalysisWorkflowRequest,
    policy: AnalysisWorkflowPolicy,
    *,
    feature_columns: Sequence[str] | None = None,
) -> dict[str, object]:
    """Build the effective configuration payload used for fingerprinting.

    Excludes transient execution fields, ``csv_path``, request metadata,
    DataFrames, models, timestamps, and UI/session-only state.
    """
    if not isinstance(request, AnalysisWorkflowRequest):
        raise TypeError(
            "request must be AnalysisWorkflowRequest, "
            f"got {type(request).__name__}"
        )
    if not isinstance(policy, AnalysisWorkflowPolicy):
        raise TypeError(
            "policy must be AnalysisWorkflowPolicy, "
            f"got {type(policy).__name__}"
        )

    effective_features = (
        list(feature_columns)
        if feature_columns is not None
        else list(request.feature_columns)
    )
    role_overrides = {
        name: role.value if isinstance(role, Enum) else str(role)
        for name, role in sorted(request.column_role_overrides.items())
    }
    performance_policy = (
        None
        if request.model_performance_policy is None
        else request.model_performance_policy.model_dump(mode="python")
    )
    cohort_filter = (
        None
        if request.cohort_filter is None
        else request.cohort_filter.model_dump(mode="python")
    )

    return {
        "analysis_mode": request.analysis_mode.value,
        "anomaly_recommendation": request.anomaly_recommendation.model_dump(
            mode="python"
        ),
        "cohort_filter": cohort_filter,
        "column_role_overrides": role_overrides,
        "declared_target_maximum": request.declared_target_maximum,
        "declared_target_minimum": request.declared_target_minimum,
        "excluded_columns": list(request.excluded_columns),
        "explicit_operating_row_id": request.explicit_operating_row_id,
        "feature_columns": effective_features,
        "identifier_columns": list(request.identifier_columns),
        "industry_constraints": [
            item.model_dump(mode="python") for item in request.industry_constraints
        ],
        "max_simultaneous_changes": request.max_simultaneous_changes,
        "model_performance_policy": performance_policy,
        "objective": (
            None if request.objective is None else request.objective.value
        ),
        "operating_point_selection": request.operating_point_selection.value,
        "quality_direction": (
            None
            if request.quality_direction is None
            else request.quality_direction.value
        ),
        "quality_target": request.quality_target,
        "request_constraints": [
            item.model_dump(mode="python") for item in request.request_constraints
        ],
        "requested_task": (
            None if request.requested_task is None else request.requested_task.value
        ),
        "target_column": request.target_column,
        "timestamp_column": request.timestamp_column,
        "user_confirmed_controllable_variables": list(
            request.user_confirmed_controllable_variables
        ),
        "user_overrides": [
            item.model_dump(mode="python") for item in request.user_overrides
        ],
        "user_verified_variables": list(request.user_verified_variables),
        "workflow_policy": policy.model_dump(mode="python"),
    }


def compute_configuration_fingerprint(
    request: AnalysisWorkflowRequest,
    policy: AnalysisWorkflowPolicy,
    *,
    feature_columns: Sequence[str] | None = None,
) -> str:
    """Return a SHA-256 hex digest of the effective workflow configuration."""
    payload = build_configuration_fingerprint_payload(
        request,
        policy,
        feature_columns=feature_columns,
    )
    return _sha256_hex(deterministic_fingerprint_json(payload))


def compute_run_signature(
    *,
    dataset_fingerprint: str | None,
    configuration_fingerprint: str,
    application_version: str,
    manifest_schema_version: int = MANIFEST_SCHEMA_VERSION,
) -> str:
    """Return a SHA-256 reproducibility signature for one analysis run.

    The signature identifies matching dataset content, effective configuration,
    software version, and manifest schema. It is not a cryptographic proof of
    model correctness or causality. Timing values are intentionally excluded.
    """
    if not isinstance(configuration_fingerprint, str) or not configuration_fingerprint:
        raise ValueError("configuration_fingerprint must be a non-empty str")
    if not isinstance(application_version, str) or not application_version.strip():
        raise ValueError("application_version must be a non-empty str")
    if (
        isinstance(manifest_schema_version, bool)
        or not isinstance(manifest_schema_version, int)
        or manifest_schema_version < 1
    ):
        raise ValueError(
            "manifest_schema_version must be an int >= 1 "
            f"(bool not allowed), got {manifest_schema_version!r}"
        )
    if dataset_fingerprint is not None and (
        not isinstance(dataset_fingerprint, str) or not dataset_fingerprint
    ):
        raise ValueError(
            "dataset_fingerprint must be a non-empty str or None, "
            f"got {dataset_fingerprint!r}"
        )

    payload = {
        "application_version": application_version.strip(),
        "configuration_fingerprint": configuration_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "manifest_schema_version": manifest_schema_version,
    }
    return _sha256_hex(deterministic_fingerprint_json(payload))


def stage_manifest_status(
    *,
    executed: bool,
    succeeded: bool,
    structured_refusal: bool,
) -> str:
    """Derive a compact stage status label for the run manifest."""
    if not executed:
        return "SKIPPED"
    if structured_refusal:
        return "REFUSED"
    if succeeded:
        return "SUCCEEDED"
    return "INCOMPLETE"


def _shorten_message(message: str) -> str:
    text = message.strip()
    if len(text) <= _SHORT_MESSAGE_LIMIT:
        return text
    return text[: _SHORT_MESSAGE_LIMIT - 3].rstrip() + "..."


def _stage_code_or_reason(record: AnalysisWorkflowStageRecord) -> str | None:
    for key in _STAGE_CODE_KEYS:
        value = record.metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


class AnalysisStageManifestEntry(BaseModel):
    """Compact stage summary entry for one analysis-run manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: AnalysisWorkflowStage
    status: str
    code: str | None = None
    duration_seconds: float | None = None
    message: str

    @field_validator("stage", mode="before")
    @classmethod
    def _validate_stage(cls, value: object) -> AnalysisWorkflowStage:
        if isinstance(value, AnalysisWorkflowStage):
            return value
        if isinstance(value, str):
            try:
                return AnalysisWorkflowStage(value)
            except ValueError as exc:
                raise ValueError(f"invalid AnalysisWorkflowStage: {value!r}") from exc
        raise ValueError(
            f"stage must be AnalysisWorkflowStage, got {type(value).__name__}"
        )

    @field_validator("status", "message", mode="before")
    @classmethod
    def _validate_non_empty_str(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="stage manifest string")

    @field_validator("code", mode="before")
    @classmethod
    def _validate_code(cls, value: object) -> str | None:
        return _require_optional_non_empty_str(value, field_name="code")

    @field_validator("duration_seconds", mode="before")
    @classmethod
    def _validate_duration(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_non_negative_finite_float(
            value,
            field_name="duration_seconds",
        )


class AnalysisRunManifest(BaseModel):
    """Immutable reproducibility manifest for one analysis workflow report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_schema_version: Literal[1] = MANIFEST_SCHEMA_VERSION
    run_signature: str
    configuration_fingerprint: str
    dataset_fingerprint: str | None = None
    application_version: str
    started_at_utc: datetime
    completed_at_utc: datetime
    duration_seconds: float
    analysis_mode: AnalysisExecutionMode
    requested_task: AnalysisTask | None = None
    resolved_task: AnalysisTask | None = None
    target_column: str | None = None
    timestamp_column: str | None = None
    feature_count: int
    feature_columns: list[str] = Field(default_factory=list)
    operating_point_selection_mode: OperatingPointSelectionMode
    workflow_status: AnalysisWorkflowStatus
    selected_model: str | None = None
    stage_entries: list[AnalysisStageManifestEntry] = Field(default_factory=list)
    warning_count: int = 0
    refusal_or_failure_code: str | None = None

    @field_validator("manifest_schema_version", mode="before")
    @classmethod
    def _validate_schema_version(cls, value: object) -> int:
        if value != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"manifest_schema_version must be {MANIFEST_SCHEMA_VERSION}, "
                f"got {value!r}"
            )
        return MANIFEST_SCHEMA_VERSION

    @field_validator(
        "run_signature",
        "configuration_fingerprint",
        "application_version",
        mode="before",
    )
    @classmethod
    def _validate_required_strings(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="manifest string field")

    @field_validator("dataset_fingerprint", mode="before")
    @classmethod
    def _validate_dataset_fingerprint(cls, value: object) -> str | None:
        if value is None:
            return None
        text = _require_non_empty_str(value, field_name="dataset_fingerprint")
        if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
            raise ValueError(
                "dataset_fingerprint must be a 64-character lowercase hex "
                f"SHA-256 digest, got {text!r}"
            )
        return text

    @field_validator("started_at_utc", "completed_at_utc", mode="before")
    @classmethod
    def _validate_timestamps_before(cls, value: object) -> object:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            text = value.strip()
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(text)
            except ValueError as exc:
                raise ValueError(
                    f"timestamp must be an ISO-8601 datetime string, got {value!r}"
                ) from exc
        return value

    @field_validator("started_at_utc", "completed_at_utc", mode="after")
    @classmethod
    def _validate_timestamps_after(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value, field_name="manifest timestamp")

    @field_validator("duration_seconds", mode="before")
    @classmethod
    def _validate_duration(cls, value: object) -> float:
        return _require_non_negative_finite_float(
            value,
            field_name="duration_seconds",
        )

    @field_validator("analysis_mode", mode="before")
    @classmethod
    def _validate_analysis_mode(cls, value: object) -> AnalysisExecutionMode:
        if isinstance(value, AnalysisExecutionMode):
            return value
        if isinstance(value, str):
            try:
                return AnalysisExecutionMode(value)
            except ValueError as exc:
                raise ValueError(f"invalid AnalysisExecutionMode: {value!r}") from exc
        raise ValueError(
            "analysis_mode must be AnalysisExecutionMode, "
            f"got {type(value).__name__}"
        )

    @field_validator("requested_task", "resolved_task", mode="before")
    @classmethod
    def _validate_optional_task(cls, value: object) -> AnalysisTask | None:
        if value is None:
            return None
        if isinstance(value, AnalysisTask):
            return value
        if isinstance(value, str):
            try:
                return AnalysisTask(value)
            except ValueError as exc:
                raise ValueError(f"invalid AnalysisTask: {value!r}") from exc
        raise ValueError(
            f"task must be AnalysisTask or None, got {type(value).__name__}"
        )

    @field_validator("target_column", "timestamp_column", "selected_model", mode="before")
    @classmethod
    def _validate_optional_strings(cls, value: object) -> str | None:
        return _require_optional_non_empty_str(value, field_name="optional string")

    @field_validator("feature_count", "warning_count", mode="before")
    @classmethod
    def _validate_counts(cls, value: object) -> int:
        return _require_strict_int_ge0(value, field_name="count field")

    @field_validator("feature_columns", mode="before")
    @classmethod
    def _validate_feature_columns_before(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"feature_columns must be a list[str], got {type(value).__name__}"
            )
        return list(value)

    @field_validator("feature_columns", mode="after")
    @classmethod
    def _validate_feature_columns(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = _require_non_empty_str(item, field_name="feature_columns")
            if text in seen:
                raise ValueError(
                    f"feature_columns must not contain duplicates: {text!r}"
                )
            seen.add(text)
            cleaned.append(text)
        return cleaned

    @field_validator("operating_point_selection_mode", mode="before")
    @classmethod
    def _validate_operating_point(
        cls,
        value: object,
    ) -> OperatingPointSelectionMode:
        if isinstance(value, OperatingPointSelectionMode):
            return value
        if isinstance(value, str):
            try:
                return OperatingPointSelectionMode(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid OperatingPointSelectionMode: {value!r}"
                ) from exc
        raise ValueError(
            "operating_point_selection_mode must be OperatingPointSelectionMode, "
            f"got {type(value).__name__}"
        )

    @field_validator("workflow_status", mode="before")
    @classmethod
    def _validate_workflow_status(cls, value: object) -> AnalysisWorkflowStatus:
        if isinstance(value, AnalysisWorkflowStatus):
            return value
        if isinstance(value, str):
            try:
                return AnalysisWorkflowStatus(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid AnalysisWorkflowStatus: {value!r}"
                ) from exc
        raise ValueError(
            "workflow_status must be AnalysisWorkflowStatus, "
            f"got {type(value).__name__}"
        )

    @field_validator("stage_entries", mode="before")
    @classmethod
    def _validate_stage_entries_before(
        cls,
        value: object,
    ) -> list[AnalysisStageManifestEntry]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                "stage_entries must be a list[AnalysisStageManifestEntry], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("stage_entries", mode="after")
    @classmethod
    def _validate_stage_entries(
        cls,
        value: list[AnalysisStageManifestEntry],
    ) -> list[AnalysisStageManifestEntry]:
        copied: list[AnalysisStageManifestEntry] = []
        for item in value:
            if not isinstance(item, AnalysisStageManifestEntry):
                raise ValueError(
                    "stage_entries entries must be AnalysisStageManifestEntry, "
                    f"got {type(item).__name__}"
                )
            copied.append(item.model_copy(deep=True))
        return copied

    @field_validator("refusal_or_failure_code", mode="before")
    @classmethod
    def _validate_refusal_code(cls, value: object) -> str | None:
        return _require_optional_non_empty_str(
            value,
            field_name="refusal_or_failure_code",
        )

    @model_validator(mode="after")
    def _validate_consistency(self) -> Self:
        if self.completed_at_utc < self.started_at_utc:
            raise ValueError("completed_at_utc must be >= started_at_utc")
        if self.feature_count != len(self.feature_columns):
            raise ValueError(
                "feature_count must equal len(feature_columns) "
                f"(got {self.feature_count} != {len(self.feature_columns)})"
            )
        if len(self.run_signature) != 64:
            raise ValueError("run_signature must be a 64-character SHA-256 hex digest")
        if len(self.configuration_fingerprint) != 64:
            raise ValueError(
                "configuration_fingerprint must be a 64-character SHA-256 hex digest"
            )
        return self


def build_stage_manifest_entries(
    stage_records: Sequence[AnalysisWorkflowStageRecord],
) -> list[AnalysisStageManifestEntry]:
    """Build compact stage entries while preserving existing stage order."""
    entries: list[AnalysisStageManifestEntry] = []
    for record in stage_records:
        if not isinstance(record, AnalysisWorkflowStageRecord):
            raise TypeError(
                "stage_records entries must be AnalysisWorkflowStageRecord, "
                f"got {type(record).__name__}"
            )
        duration_raw = record.metadata.get("duration_seconds")
        duration: float | None
        if duration_raw is None:
            duration = None
        else:
            duration = _require_non_negative_finite_float(
                duration_raw,
                field_name="duration_seconds",
            )
        entries.append(
            AnalysisStageManifestEntry(
                stage=record.stage,
                status=stage_manifest_status(
                    executed=record.executed,
                    succeeded=record.succeeded,
                    structured_refusal=record.structured_refusal,
                ),
                code=_stage_code_or_reason(record),
                duration_seconds=duration,
                message=_shorten_message(record.message),
            )
        )
    return entries


def resolve_refusal_or_failure_code(
    *,
    stage_records: Sequence[AnalysisWorkflowStageRecord],
    target_refusal_code: str | None,
    recommendation_status: RecommendationStatus | None,
    recommendation_reason_codes: Sequence[str] | None = None,
) -> str | None:
    """Resolve a compact refusal/failure code when one is available."""
    if recommendation_status is RecommendationStatus.REFUSED:
        if recommendation_reason_codes:
            for reason_code in recommendation_reason_codes:
                if isinstance(reason_code, str) and reason_code.strip():
                    return reason_code.strip()
        return RecommendationStatus.REFUSED.value
    if target_refusal_code is not None and target_refusal_code.strip():
        return target_refusal_code.strip()
    for record in reversed(list(stage_records)):
        if record.structured_refusal:
            stage_code = _stage_code_or_reason(record)
            if stage_code is not None:
                return stage_code
            return record.stage.value
    return None


def build_analysis_run_manifest(
    *,
    request: AnalysisWorkflowRequest,
    policy: AnalysisWorkflowPolicy,
    stage_records: Sequence[AnalysisWorkflowStageRecord],
    workflow_status: AnalysisWorkflowStatus,
    dataset_fingerprint: str | None,
    feature_columns: Sequence[str],
    analysis_mode: AnalysisExecutionMode,
    requested_task: AnalysisTask | None,
    resolved_task: AnalysisTask | None,
    target_column: str | None,
    selected_model: str | None,
    warning_count: int,
    started_at_utc: datetime,
    completed_at_utc: datetime,
    duration_seconds: float,
    application_version: str | None = None,
    target_refusal_code: str | None = None,
    recommendation_status: RecommendationStatus | None = None,
    recommendation_reason_codes: Sequence[str] | None = None,
) -> AnalysisRunManifest:
    """Construct an immutable analysis-run manifest for one workflow report."""
    version = (
        get_application_version()
        if application_version is None
        else application_version.strip()
    )
    features = list(feature_columns)
    configuration_fingerprint = compute_configuration_fingerprint(
        request,
        policy,
        feature_columns=features,
    )
    run_signature = compute_run_signature(
        dataset_fingerprint=dataset_fingerprint,
        configuration_fingerprint=configuration_fingerprint,
        application_version=version,
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
    )
    return AnalysisRunManifest(
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
        run_signature=run_signature,
        configuration_fingerprint=configuration_fingerprint,
        dataset_fingerprint=dataset_fingerprint,
        application_version=version,
        started_at_utc=started_at_utc,
        completed_at_utc=completed_at_utc,
        duration_seconds=duration_seconds,
        analysis_mode=analysis_mode,
        requested_task=requested_task,
        resolved_task=resolved_task,
        target_column=target_column,
        timestamp_column=request.timestamp_column,
        feature_count=len(features),
        feature_columns=features,
        operating_point_selection_mode=request.operating_point_selection,
        workflow_status=workflow_status,
        selected_model=selected_model,
        stage_entries=build_stage_manifest_entries(stage_records),
        warning_count=warning_count,
        refusal_or_failure_code=resolve_refusal_or_failure_code(
            stage_records=stage_records,
            target_refusal_code=target_refusal_code,
            recommendation_status=recommendation_status,
            recommendation_reason_codes=recommendation_reason_codes,
        ),
    )


def analysis_run_manifest_to_json(manifest: AnalysisRunManifest) -> str:
    """Serialize a run manifest with deterministic key ordering."""
    if not isinstance(manifest, AnalysisRunManifest):
        raise TypeError(
            "manifest must be AnalysisRunManifest, "
            f"got {type(manifest).__name__}"
        )
    payload = manifest.model_dump(mode="json")
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def assert_manifest_has_no_forbidden_objects(payload: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` when a dumped manifest contains forbidden objects."""
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

    _walk(dict(payload), path="manifest")


def _rebuild_workflow_report_forward_refs() -> None:
    """Resolve the ``AnalysisWorkflowReport.run_manifest`` forward reference."""
    from process_intelligence.workflow.schemas import AnalysisWorkflowReport

    AnalysisWorkflowReport.model_rebuild(
        _types_namespace={"AnalysisRunManifest": AnalysisRunManifest}
    )


_rebuild_workflow_report_forward_refs()
