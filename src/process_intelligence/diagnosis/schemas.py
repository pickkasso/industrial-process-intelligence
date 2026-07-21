"""Pydantic contracts for root-cause diagnosis requests and results (Step 8A).

Reuses core ``AnomalyEvent`` and ``RootCauseFactor``. Does not redefine those
schemas and does not perform attribution calculations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.schemas import AnomalyEvent, RootCauseFactor
from process_intelligence.diagnosis.enums import DiagnosisMethod, DiagnosisScope

ScalarMetadataValue = str | int | float | bool | None
"""Allowed scalar types for diagnosis metadata dictionaries."""


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


def _require_strict_int_ge(
    value: object,
    *,
    field_name: str,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{field_name} must be an int >= {minimum} "
            f"(bool not allowed), got {type(value).__name__}"
        )
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {value}")
    return value


def _require_strict_int_ge0(value: object, *, field_name: str) -> int:
    return _require_strict_int_ge(value, field_name=field_name, minimum=0)


def _require_confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "confidence must be a finite float in [0.0, 1.0] "
            f"(bool not allowed), got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"confidence must be finite, got {value!r}")
    if number < 0.0 or number > 1.0:
        raise ValueError(f"confidence must be in [0.0, 1.0], got {number}")
    return number


def _require_timezone_aware(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _validate_scalar_metadata(
    value: object,
    *,
    field_name: str = "metadata",
) -> dict[str, ScalarMetadataValue]:
    if not isinstance(value, dict):
        raise ValueError(
            f"{field_name} must be a dict[str, scalar], got {type(value).__name__}"
        )
    cleaned: dict[str, ScalarMetadataValue] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or key == "" or key.strip() == "":
            raise ValueError(f"{field_name} keys must be non-empty strings")
        if raw is None or isinstance(raw, (str, bool)):
            cleaned[key] = raw
            continue
        if isinstance(raw, int) and not isinstance(raw, bool):
            cleaned[key] = raw
            continue
        if isinstance(raw, float):
            if not math.isfinite(raw):
                raise ValueError(
                    f"{field_name}[{key!r}] float must be finite, got {raw!r}"
                )
            cleaned[key] = raw
            continue
        raise ValueError(
            f"{field_name}[{key!r}] must be str, int, float, bool, or None "
            f"(no DataFrame, ndarray, estimator, or nested objects); "
            f"got {type(raw).__name__}"
        )
    return cleaned


def _validate_non_empty_unique_strings(
    values: list[str],
    *,
    field_name: str,
) -> list[str]:
    if not values:
        raise ValueError(f"{field_name} must contain at least one item")
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _require_non_empty_str(item, field_name=field_name)
        if text in seen:
            raise ValueError(f"{field_name} must not contain duplicates: {text!r}")
        seen.add(text)
        cleaned.append(text)
    return cleaned


def _validate_unique_non_empty_strings(
    values: list[str],
    *,
    field_name: str,
) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _require_non_empty_str(item, field_name=field_name)
        if text in seen:
            raise ValueError(f"{field_name} must not contain duplicates: {text!r}")
        seen.add(text)
        cleaned.append(text)
    return cleaned


def _validate_root_cause_factors(
    factors: list[RootCauseFactor],
    *,
    field_name: str,
) -> list[RootCauseFactor]:
    """Copy factors and reject duplicate ``variable`` identifiers.

    ``RootCauseFactor`` has no rank field; list order is the ranking order.
    """
    seen_variables: set[str] = set()
    copied: list[RootCauseFactor] = []
    for factor in factors:
        if not isinstance(factor, RootCauseFactor):
            raise ValueError(
                f"{field_name} entries must be RootCauseFactor, "
                f"got {type(factor).__name__}"
            )
        variable = factor.variable
        if variable in seen_variables:
            raise ValueError(
                f"{field_name} must not contain duplicate variables: {variable!r}"
            )
        seen_variables.add(variable)
        copied.append(factor.model_copy(deep=True))
    return copied


class DiagnosisRequest(BaseModel):
    """Request contract for root-cause diagnosis over anomaly events.

    Specifies analysis task, attribution method, diagnosis scope, feature
    columns, and optional role columns. Holds ``AnomalyEvent`` instances from
    core; does not embed DataFrames, estimators, or prediction arrays.
    """

    task: AnalysisTask
    method: DiagnosisMethod
    scope: DiagnosisScope
    feature_columns: list[str]
    anomaly_events: list[AnomalyEvent] = Field(default_factory=list)
    target_column: str | None = None
    anomaly_indicator_column: str | None = None
    anomaly_score_column: str | None = None
    row_id_column: str = "_original_row_id"
    top_k_events: int = 20
    top_k_factors: int = 10
    minimum_reference_rows: int = 5
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)

    @field_validator("task", mode="before")
    @classmethod
    def _validate_task(cls, value: object) -> AnalysisTask:
        if isinstance(value, AnalysisTask):
            return value
        if isinstance(value, str):
            try:
                return AnalysisTask(value)
            except ValueError as exc:
                raise ValueError(f"invalid AnalysisTask: {value!r}") from exc
        raise ValueError(f"task must be AnalysisTask, got {type(value).__name__}")

    @field_validator("method", mode="before")
    @classmethod
    def _validate_method(cls, value: object) -> DiagnosisMethod:
        if isinstance(value, DiagnosisMethod):
            return value
        if isinstance(value, str):
            try:
                return DiagnosisMethod(value)
            except ValueError as exc:
                raise ValueError(f"invalid DiagnosisMethod: {value!r}") from exc
        raise ValueError(
            f"method must be DiagnosisMethod, got {type(value).__name__}"
        )

    @field_validator("scope", mode="before")
    @classmethod
    def _validate_scope(cls, value: object) -> DiagnosisScope:
        if isinstance(value, DiagnosisScope):
            return value
        if isinstance(value, str):
            try:
                return DiagnosisScope(value)
            except ValueError as exc:
                raise ValueError(f"invalid DiagnosisScope: {value!r}") from exc
        raise ValueError(
            f"scope must be DiagnosisScope, got {type(value).__name__}"
        )

    @field_validator("feature_columns", mode="before")
    @classmethod
    def _validate_feature_columns_before(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            raise ValueError(
                f"feature_columns must be a list[str], got {type(value).__name__}"
            )
        return list(value)

    @field_validator("feature_columns", mode="after")
    @classmethod
    def _validate_feature_columns(cls, value: list[str]) -> list[str]:
        return _validate_non_empty_unique_strings(
            value,
            field_name="feature_columns",
        )

    @field_validator("anomaly_events", mode="before")
    @classmethod
    def _validate_anomaly_events_before(cls, value: object) -> list[AnomalyEvent]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"anomaly_events must be a list[AnomalyEvent], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("anomaly_events", mode="after")
    @classmethod
    def _validate_anomaly_events(
        cls,
        value: list[AnomalyEvent],
    ) -> list[AnomalyEvent]:
        seen_ids: set[str] = set()
        copied: list[AnomalyEvent] = []
        for event in value:
            if not isinstance(event, AnomalyEvent):
                raise ValueError(
                    "anomaly_events entries must be AnomalyEvent, "
                    f"got {type(event).__name__}"
                )
            anomaly_id = event.anomaly_id
            if anomaly_id in seen_ids:
                raise ValueError(
                    "anomaly_events must not contain duplicate anomaly_id values: "
                    f"{anomaly_id!r}"
                )
            seen_ids.add(anomaly_id)
            copied.append(event.model_copy(deep=True))
        return copied

    @field_validator(
        "target_column",
        "anomaly_indicator_column",
        "anomaly_score_column",
        mode="before",
    )
    @classmethod
    def _validate_optional_columns(cls, value: object) -> str | None:
        return _require_optional_non_empty_str(value, field_name="column name")

    @field_validator("row_id_column", mode="before")
    @classmethod
    def _validate_row_id_column(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="row_id_column")

    @field_validator(
        "top_k_events",
        "top_k_factors",
        "minimum_reference_rows",
        mode="before",
    )
    @classmethod
    def _validate_positive_counts(cls, value: object) -> int:
        return _require_strict_int_ge(
            value,
            field_name="top-k / reference count",
            minimum=1,
        )

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> dict[str, ScalarMetadataValue]:
        if value is None:
            return {}
        return _validate_scalar_metadata(value)

    @model_validator(mode="after")
    def _validate_request_consistency(self) -> Self:
        role_columns: dict[str, str] = {"row_id_column": self.row_id_column}
        for name, column in (
            ("target_column", self.target_column),
            ("anomaly_indicator_column", self.anomaly_indicator_column),
            ("anomaly_score_column", self.anomaly_score_column),
        ):
            if column is not None:
                role_columns[name] = column

        role_values = list(role_columns.values())
        if len(role_values) != len(set(role_values)):
            raise ValueError(
                "row_id_column, target_column, anomaly_indicator_column, and "
                "anomaly_score_column must be unique when provided"
            )

        reserved = set(role_values)
        overlap = [column for column in self.feature_columns if column in reserved]
        if overlap:
            raise ValueError(
                "feature_columns must not include row ID, target, anomaly "
                f"indicator, or anomaly score columns; conflicting: {overlap}"
            )

        event_count = len(self.anomaly_events)
        if self.scope is DiagnosisScope.SINGLE_EVENT and event_count != 1:
            raise ValueError(
                "anomaly_events must contain exactly 1 event when "
                f"scope is SINGLE_EVENT, got {event_count}"
            )
        if self.scope is DiagnosisScope.TOP_ANOMALIES and event_count < 1:
            raise ValueError(
                "anomaly_events must contain at least 1 event when "
                "scope is TOP_ANOMALIES"
            )

        return self


class DiagnosisResult(BaseModel):
    """Per-event or scoped root-cause diagnosis result.

    Factors reuse core ``RootCauseFactor``. List order is ranking order because
    ``RootCauseFactor`` has no rank field. Describes association / likely
    drivers only—not causal claims.
    """

    anomaly_id: str | None = None
    task: AnalysisTask
    method_used: list[DiagnosisMethod]
    scope: DiagnosisScope
    factors: list[RootCauseFactor] = Field(default_factory=list)
    confidence: float
    analyzed_row_count: int
    reference_row_count: int
    caveats: list[str] = Field(default_factory=list)
    generated_at: datetime
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)

    @field_validator("anomaly_id", mode="before")
    @classmethod
    def _validate_anomaly_id(cls, value: object) -> str | None:
        if value is None:
            return None
        return _require_non_empty_str(value, field_name="anomaly_id")

    @field_validator("task", mode="before")
    @classmethod
    def _validate_task(cls, value: object) -> AnalysisTask:
        if isinstance(value, AnalysisTask):
            return value
        if isinstance(value, str):
            try:
                return AnalysisTask(value)
            except ValueError as exc:
                raise ValueError(f"invalid AnalysisTask: {value!r}") from exc
        raise ValueError(f"task must be AnalysisTask, got {type(value).__name__}")

    @field_validator("scope", mode="before")
    @classmethod
    def _validate_scope(cls, value: object) -> DiagnosisScope:
        if isinstance(value, DiagnosisScope):
            return value
        if isinstance(value, str):
            try:
                return DiagnosisScope(value)
            except ValueError as exc:
                raise ValueError(f"invalid DiagnosisScope: {value!r}") from exc
        raise ValueError(
            f"scope must be DiagnosisScope, got {type(value).__name__}"
        )

    @field_validator("method_used", mode="before")
    @classmethod
    def _validate_method_used_before(cls, value: object) -> list[object]:
        if not isinstance(value, list):
            raise ValueError(
                f"method_used must be a list[DiagnosisMethod], "
                f"got {type(value).__name__}"
            )
        cleaned: list[object] = []
        for item in value:
            if isinstance(item, DiagnosisMethod):
                cleaned.append(item)
            elif isinstance(item, str):
                try:
                    cleaned.append(DiagnosisMethod(item))
                except ValueError as exc:
                    raise ValueError(
                        f"invalid DiagnosisMethod: {item!r}"
                    ) from exc
            else:
                raise ValueError(
                    "method_used entries must be DiagnosisMethod, "
                    f"got {type(item).__name__}"
                )
        return cleaned

    @field_validator("method_used", mode="after")
    @classmethod
    def _validate_method_used(
        cls,
        value: list[DiagnosisMethod],
    ) -> list[DiagnosisMethod]:
        if not value:
            raise ValueError("method_used must contain at least one method")
        seen: set[DiagnosisMethod] = set()
        cleaned: list[DiagnosisMethod] = []
        for method in value:
            if not isinstance(method, DiagnosisMethod):
                raise ValueError(
                    "method_used entries must be DiagnosisMethod, "
                    f"got {type(method).__name__}"
                )
            if method in seen:
                raise ValueError(
                    f"method_used must not contain duplicates: {method!r}"
                )
            seen.add(method)
            cleaned.append(method)
        return cleaned

    @field_validator("factors", mode="before")
    @classmethod
    def _validate_factors_before(cls, value: object) -> list[RootCauseFactor]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"factors must be a list[RootCauseFactor], got {type(value).__name__}"
            )
        return list(value)

    @field_validator("factors", mode="after")
    @classmethod
    def _validate_factors(cls, value: list[RootCauseFactor]) -> list[RootCauseFactor]:
        return _validate_root_cause_factors(value, field_name="factors")

    @field_validator("confidence", mode="before")
    @classmethod
    def _validate_confidence(cls, value: object) -> float:
        return _require_confidence(value)

    @field_validator("analyzed_row_count", mode="before")
    @classmethod
    def _validate_analyzed_row_count(cls, value: object) -> int:
        return _require_strict_int_ge(
            value,
            field_name="analyzed_row_count",
            minimum=1,
        )

    @field_validator("reference_row_count", mode="before")
    @classmethod
    def _validate_reference_row_count(cls, value: object) -> int:
        return _require_strict_int_ge0(value, field_name="reference_row_count")

    @field_validator("caveats", mode="before")
    @classmethod
    def _validate_caveats_before(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"caveats must be a list[str], got {type(value).__name__}"
            )
        return list(value)

    @field_validator("caveats", mode="after")
    @classmethod
    def _validate_caveats(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(value, field_name="caveats")

    @field_validator("generated_at", mode="after")
    @classmethod
    def _validate_generated_at(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value, field_name="generated_at")

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> dict[str, ScalarMetadataValue]:
        if value is None:
            return {}
        return _validate_scalar_metadata(value)

class DiagnosisBatchResult(BaseModel):
    """Batch-level diagnosis outcome with optional aggregate factors.

    ``diagnosed_event_count + failed_event_count`` must equal
    ``requested_event_count`` (no missing events in this contract).
    """

    results: list[DiagnosisResult] = Field(default_factory=list)
    aggregate_factors: list[RootCauseFactor] = Field(default_factory=list)
    requested_event_count: int
    diagnosed_event_count: int
    failed_event_count: int
    generated_at: datetime
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)

    @field_validator("results", mode="before")
    @classmethod
    def _validate_results_before(cls, value: object) -> list[DiagnosisResult]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"results must be a list[DiagnosisResult], got {type(value).__name__}"
            )
        return list(value)

    @field_validator("results", mode="after")
    @classmethod
    def _validate_results(cls, value: list[DiagnosisResult]) -> list[DiagnosisResult]:
        seen_ids: set[str | None] = set()
        copied: list[DiagnosisResult] = []
        for result in value:
            if not isinstance(result, DiagnosisResult):
                raise ValueError(
                    "results entries must be DiagnosisResult, "
                    f"got {type(result).__name__}"
                )
            if result.anomaly_id in seen_ids:
                raise ValueError(
                    "results must not contain duplicate anomaly_id values: "
                    f"{result.anomaly_id!r}"
                )
            seen_ids.add(result.anomaly_id)
            copied.append(result.model_copy(deep=True))
        return copied

    @field_validator("aggregate_factors", mode="before")
    @classmethod
    def _validate_aggregate_factors_before(
        cls,
        value: object,
    ) -> list[RootCauseFactor]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"aggregate_factors must be a list[RootCauseFactor], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("aggregate_factors", mode="after")
    @classmethod
    def _validate_aggregate_factors(
        cls,
        value: list[RootCauseFactor],
    ) -> list[RootCauseFactor]:
        return _validate_root_cause_factors(
            value,
            field_name="aggregate_factors",
        )

    @field_validator(
        "requested_event_count",
        "diagnosed_event_count",
        "failed_event_count",
        mode="before",
    )
    @classmethod
    def _validate_event_counts(cls, value: object) -> int:
        return _require_strict_int_ge0(value, field_name="event count")

    @field_validator("generated_at", mode="after")
    @classmethod
    def _validate_generated_at(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value, field_name="generated_at")

    @field_validator("warnings", mode="before")
    @classmethod
    def _validate_warnings_before(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"warnings must be a list[str], got {type(value).__name__}"
            )
        return list(value)

    @field_validator("warnings", mode="after")
    @classmethod
    def _validate_warnings(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(value, field_name="warnings")

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> dict[str, ScalarMetadataValue]:
        if value is None:
            return {}
        return _validate_scalar_metadata(value)

    @model_validator(mode="after")
    def _validate_batch_consistency(self) -> Self:
        accounted = self.diagnosed_event_count + self.failed_event_count
        if accounted != self.requested_event_count:
            raise ValueError(
                "diagnosed_event_count + failed_event_count must equal "
                "requested_event_count "
                f"(got {self.diagnosed_event_count} + {self.failed_event_count} "
                f"!= {self.requested_event_count})"
            )

        return self


@dataclass(frozen=True, slots=True)
class DiagnosisOutcome:
    """Immutable wrapper around a single or batch diagnosis result.

    Holds only ``DiagnosisResult`` or ``DiagnosisBatchResult``. Does not store
    models, estimators, or DataFrames, and contains no business logic.
    """

    result: DiagnosisResult | DiagnosisBatchResult
