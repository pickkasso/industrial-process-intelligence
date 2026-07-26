"""Compare the current declared setup against a stored run manifest (Step 14B).

Informational only: does not modify stored reports, manifests, workflow status,
metrics, anomaly events, diagnosis, recommendation, or demo evaluation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from process_intelligence.workflow.run_manifest import (
    MANIFEST_SCHEMA_VERSION,
    AnalysisRunManifest,
    compute_configuration_fingerprint,
    compute_run_signature,
    get_application_version,
)
from process_intelligence.workflow.schemas import (
    AnalysisWorkflowPolicy,
    AnalysisWorkflowRequest,
)

_MESSAGE_EXACT_MATCH = (
    "The currently loaded dataset and effective analysis configuration match "
    "this stored run."
)
_MESSAGE_CONFIGURATION_CHANGED = (
    "The current analysis configuration differs from the configuration that "
    "produced this result. Run analysis again before interpreting the result "
    "as current."
)
_MESSAGE_DATASET_CHANGED = (
    "The active dataset differs from the dataset that produced this result."
)
_MESSAGE_APPLICATION_VERSION_CHANGED = (
    "The application version differs from the version recorded for this stored "
    "run."
)
_MESSAGE_MANIFEST_SCHEMA_CHANGED = (
    "The manifest schema version differs from the schema recorded for this "
    "stored run."
)
_MESSAGE_MULTIPLE_CHANGED = (
    "Multiple declared inputs differ from this stored run "
    "(dataset, configuration, application version, and/or manifest schema)."
)
_MESSAGE_CONFIGURATION_INCOMPLETE = (
    "The current configuration cannot be fingerprinted until the listed "
    "readiness issues are resolved."
)
_MESSAGE_UNAVAILABLE = (
    "Current-setup comparison is unavailable because the stored run manifest "
    "or the active dataset fingerprint is missing."
)
_MESSAGE_SIGNATURE_DISCLAIMER = (
    "Matching signatures identify matching declared inputs and software "
    "metadata; they do not prove model correctness, causality, or bit-for-bit "
    "deterministic execution."
)


class ReproducibilityMatchStatus(StrEnum):
    """Overall current-setup vs stored-manifest comparison outcome."""

    EXACT_MATCH = "EXACT_MATCH"
    DATASET_CHANGED = "DATASET_CHANGED"
    CONFIGURATION_CHANGED = "CONFIGURATION_CHANGED"
    APPLICATION_VERSION_CHANGED = "APPLICATION_VERSION_CHANGED"
    MANIFEST_SCHEMA_CHANGED = "MANIFEST_SCHEMA_CHANGED"
    MULTIPLE_COMPONENTS_CHANGED = "MULTIPLE_COMPONENTS_CHANGED"
    CURRENT_CONFIGURATION_INCOMPLETE = "CURRENT_CONFIGURATION_INCOMPLETE"
    UNAVAILABLE = "UNAVAILABLE"


class ReproducibilityComponentStatus(StrEnum):
    """Per-component match status for a current-setup comparison."""

    MATCH = "MATCH"
    CHANGED = "CHANGED"
    UNAVAILABLE = "UNAVAILABLE"
    INCOMPLETE = "INCOMPLETE"


class ReproducibilityComparison(BaseModel):
    """Serializable current-setup comparison against one stored run manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    overall_status: ReproducibilityMatchStatus
    dataset_matches: ReproducibilityComponentStatus
    configuration_matches: ReproducibilityComponentStatus
    application_version_matches: ReproducibilityComponentStatus
    manifest_schema_matches: ReproducibilityComponentStatus
    stored_run_signature: str | None = None
    current_candidate_run_signature: str | None = None
    explanatory_messages: list[str] = Field(default_factory=list)
    comparison_available: bool
    incompleteness_reasons: list[str] = Field(default_factory=list)

    @field_validator("stored_run_signature", "current_candidate_run_signature", mode="before")
    @classmethod
    def _validate_optional_signature(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(
                "run signature fields must be str or None, "
                f"got {type(value).__name__}"
            )
        text = value.strip()
        if text == "":
            raise ValueError("run signature fields must be non-empty when provided")
        return text

    @field_validator("explanatory_messages", "incompleteness_reasons", mode="before")
    @classmethod
    def _validate_string_lists(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"string list fields must be list[str], got {type(value).__name__}"
            )
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError(
                    "string list entries must be str, "
                    f"got {type(item).__name__}"
                )
            text = item.strip()
            if text == "":
                raise ValueError("string list entries must be non-empty")
            cleaned.append(text)
        return cleaned

    @field_validator("comparison_available", mode="before")
    @classmethod
    def _validate_bool(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError(
                "comparison_available must be a bool "
                f"(0/1 and strings rejected), got {type(value).__name__}"
            )
        return value

    @model_validator(mode="after")
    def _validate_consistency(self) -> Self:
        if (
            self.overall_status is ReproducibilityMatchStatus.UNAVAILABLE
            and self.comparison_available
        ):
            raise ValueError(
                "comparison_available must be False when overall_status is UNAVAILABLE"
            )
        if (
            self.overall_status
            is ReproducibilityMatchStatus.CURRENT_CONFIGURATION_INCOMPLETE
            and self.configuration_matches is not ReproducibilityComponentStatus.INCOMPLETE
        ):
            raise ValueError(
                "configuration_matches must be INCOMPLETE when overall_status is "
                "CURRENT_CONFIGURATION_INCOMPLETE"
            )
        return self


def reproducibility_comparison_to_json(comparison: ReproducibilityComparison) -> str:
    """Serialize a comparison with deterministic key ordering."""
    if not isinstance(comparison, ReproducibilityComparison):
        raise TypeError(
            "comparison must be ReproducibilityComparison, "
            f"got {type(comparison).__name__}"
        )
    payload = comparison.model_dump(mode="json")
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


def assert_reproducibility_comparison_has_no_forbidden_objects(
    payload: Mapping[str, Any],
) -> None:
    """Raise ``ValueError`` when a dumped comparison contains forbidden objects."""
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

    _walk(dict(payload), path="comparison")


def _component_status(matches: bool | None) -> ReproducibilityComponentStatus:
    if matches is None:
        return ReproducibilityComponentStatus.UNAVAILABLE
    if matches:
        return ReproducibilityComponentStatus.MATCH
    return ReproducibilityComponentStatus.CHANGED


def _overall_from_changed_components(
    changed: Sequence[str],
) -> ReproducibilityMatchStatus:
    if not changed:
        return ReproducibilityMatchStatus.EXACT_MATCH
    if len(changed) >= 2:
        return ReproducibilityMatchStatus.MULTIPLE_COMPONENTS_CHANGED
    sole = changed[0]
    mapping = {
        "dataset": ReproducibilityMatchStatus.DATASET_CHANGED,
        "configuration": ReproducibilityMatchStatus.CONFIGURATION_CHANGED,
        "application_version": ReproducibilityMatchStatus.APPLICATION_VERSION_CHANGED,
        "manifest_schema": ReproducibilityMatchStatus.MANIFEST_SCHEMA_CHANGED,
    }
    return mapping[sole]


def compare_setup_to_run_manifest(
    *,
    stored_manifest: AnalysisRunManifest | None,
    current_dataset_fingerprint: str | None,
    current_request: AnalysisWorkflowRequest | None = None,
    current_policy: AnalysisWorkflowPolicy | None = None,
    current_feature_columns: Sequence[str] | None = None,
    current_configuration_incomplete: bool = False,
    incompleteness_reasons: Sequence[str] | None = None,
    current_application_version: str | None = None,
    current_manifest_schema_version: int | None = None,
) -> ReproducibilityComparison:
    """Compare the current declared setup against a stored analysis-run manifest.

    Timing fields (started/completed timestamps and duration) are intentionally
    ignored. Selected model, metrics, anomaly scores, and stage statuses are not
    treated as current-setup inputs.
    """
    reasons = [
        item.strip()
        for item in (incompleteness_reasons or ())
        if isinstance(item, str) and item.strip()
    ]
    if stored_manifest is not None and not isinstance(
        stored_manifest, AnalysisRunManifest
    ):
        raise TypeError(
            "stored_manifest must be AnalysisRunManifest or None, "
            f"got {type(stored_manifest).__name__}"
        )
    if current_request is not None and not isinstance(
        current_request, AnalysisWorkflowRequest
    ):
        raise TypeError(
            "current_request must be AnalysisWorkflowRequest or None, "
            f"got {type(current_request).__name__}"
        )
    if current_policy is not None and not isinstance(
        current_policy, AnalysisWorkflowPolicy
    ):
        raise TypeError(
            "current_policy must be AnalysisWorkflowPolicy or None, "
            f"got {type(current_policy).__name__}"
        )

    if stored_manifest is None or current_dataset_fingerprint is None:
        return ReproducibilityComparison(
            overall_status=ReproducibilityMatchStatus.UNAVAILABLE,
            dataset_matches=ReproducibilityComponentStatus.UNAVAILABLE,
            configuration_matches=(
                ReproducibilityComponentStatus.INCOMPLETE
                if current_configuration_incomplete
                else ReproducibilityComponentStatus.UNAVAILABLE
            ),
            application_version_matches=ReproducibilityComponentStatus.UNAVAILABLE,
            manifest_schema_matches=ReproducibilityComponentStatus.UNAVAILABLE,
            stored_run_signature=(
                None if stored_manifest is None else stored_manifest.run_signature
            ),
            current_candidate_run_signature=None,
            explanatory_messages=[_MESSAGE_UNAVAILABLE],
            comparison_available=False,
            incompleteness_reasons=reasons,
        )

    version = (
        get_application_version()
        if current_application_version is None
        else current_application_version.strip()
    )
    schema_version = (
        MANIFEST_SCHEMA_VERSION
        if current_manifest_schema_version is None
        else current_manifest_schema_version
    )

    dataset_matches = (
        current_dataset_fingerprint == stored_manifest.dataset_fingerprint
    )
    version_matches = version == stored_manifest.application_version
    schema_matches = schema_version == stored_manifest.manifest_schema_version

    if current_configuration_incomplete:
        messages = [_MESSAGE_CONFIGURATION_INCOMPLETE]
        if reasons:
            messages.extend(reasons)
        messages.append(_MESSAGE_SIGNATURE_DISCLAIMER)
        return ReproducibilityComparison(
            overall_status=ReproducibilityMatchStatus.CURRENT_CONFIGURATION_INCOMPLETE,
            dataset_matches=_component_status(dataset_matches),
            configuration_matches=ReproducibilityComponentStatus.INCOMPLETE,
            application_version_matches=_component_status(version_matches),
            manifest_schema_matches=_component_status(schema_matches),
            stored_run_signature=stored_manifest.run_signature,
            current_candidate_run_signature=None,
            explanatory_messages=messages,
            comparison_available=True,
            incompleteness_reasons=reasons,
        )

    if current_request is None or current_policy is None:
        return ReproducibilityComparison(
            overall_status=ReproducibilityMatchStatus.CURRENT_CONFIGURATION_INCOMPLETE,
            dataset_matches=_component_status(dataset_matches),
            configuration_matches=ReproducibilityComponentStatus.INCOMPLETE,
            application_version_matches=_component_status(version_matches),
            manifest_schema_matches=_component_status(schema_matches),
            stored_run_signature=stored_manifest.run_signature,
            current_candidate_run_signature=None,
            explanatory_messages=[
                _MESSAGE_CONFIGURATION_INCOMPLETE,
                _MESSAGE_SIGNATURE_DISCLAIMER,
            ],
            comparison_available=True,
            incompleteness_reasons=reasons or ["current request could not be built"],
        )

    current_config_fp = compute_configuration_fingerprint(
        current_request,
        current_policy,
        feature_columns=current_feature_columns,
    )
    configuration_matches = (
        current_config_fp == stored_manifest.configuration_fingerprint
    )
    candidate_signature = compute_run_signature(
        dataset_fingerprint=current_dataset_fingerprint,
        configuration_fingerprint=current_config_fp,
        application_version=version,
        manifest_schema_version=schema_version,
    )

    changed: list[str] = []
    if not dataset_matches:
        changed.append("dataset")
    if not configuration_matches:
        changed.append("configuration")
    if not version_matches:
        changed.append("application_version")
    if not schema_matches:
        changed.append("manifest_schema")

    overall = _overall_from_changed_components(changed)
    if overall is ReproducibilityMatchStatus.EXACT_MATCH:
        messages = [_MESSAGE_EXACT_MATCH, _MESSAGE_SIGNATURE_DISCLAIMER]
    elif overall is ReproducibilityMatchStatus.DATASET_CHANGED:
        messages = [_MESSAGE_DATASET_CHANGED, _MESSAGE_SIGNATURE_DISCLAIMER]
    elif overall is ReproducibilityMatchStatus.CONFIGURATION_CHANGED:
        messages = [_MESSAGE_CONFIGURATION_CHANGED, _MESSAGE_SIGNATURE_DISCLAIMER]
    elif overall is ReproducibilityMatchStatus.APPLICATION_VERSION_CHANGED:
        messages = [
            _MESSAGE_APPLICATION_VERSION_CHANGED,
            _MESSAGE_SIGNATURE_DISCLAIMER,
        ]
    elif overall is ReproducibilityMatchStatus.MANIFEST_SCHEMA_CHANGED:
        messages = [_MESSAGE_MANIFEST_SCHEMA_CHANGED, _MESSAGE_SIGNATURE_DISCLAIMER]
    else:
        messages = [_MESSAGE_MULTIPLE_CHANGED, _MESSAGE_SIGNATURE_DISCLAIMER]

    return ReproducibilityComparison(
        overall_status=overall,
        dataset_matches=_component_status(dataset_matches),
        configuration_matches=_component_status(configuration_matches),
        application_version_matches=_component_status(version_matches),
        manifest_schema_matches=_component_status(schema_matches),
        stored_run_signature=stored_manifest.run_signature,
        current_candidate_run_signature=candidate_signature,
        explanatory_messages=messages,
        comparison_available=True,
        incompleteness_reasons=[],
    )
