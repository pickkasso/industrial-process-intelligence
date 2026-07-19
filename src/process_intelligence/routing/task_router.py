"""Analysis-task routing from DatasetProfile and column-role mapping."""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from process_intelligence.core.enums import AnalysisTask, ColumnRole
from process_intelligence.core.exceptions import DataValidationError, InsufficientDataError
from process_intelligence.data.profiler import ColumnProfile, DatasetProfile
from process_intelligence.routing.schema_mapper import ColumnRoleMappingResult

_ORIGINAL_ROW_ID = "_original_row_id"

DecisionLiteral = Literal[
    "AUTO_SELECTED",
    "CONFIRMATION_REQUIRED",
    "USER_CONFIRMED",
]


def _validate_int_at_least_two(value: object, *, field_name: str) -> int:
    """Reject bools and non-integers; require value >= 2."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{field_name} must be an int >= 2 (bool not allowed), got {value!r}"
        )
    if value < 2:
        raise ValueError(f"{field_name} must be >= 2, got {value}")
    return value


def _validate_unit_interval(value: object, *, field_name: str) -> float:
    """Reject bools, non-numbers, NaN, infinities, and values outside [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a finite number in [0, 1] "
            f"(bool not allowed), got {value!r}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(
            f"{field_name} must be a finite number in [0, 1], got {value!r}"
        )
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {number}")
    return number


def _is_numeric_dtype(dtype: str) -> bool:
    """Return True when a profiler dtype string is numeric (not Boolean)."""
    lowered = dtype.casefold()
    return (
        lowered.startswith("int")
        or lowered.startswith("uint")
        or lowered.startswith("float")
        or lowered.startswith("decimal")
    )


def _is_integer_dtype(dtype: str) -> bool:
    """Return True when a profiler dtype string is integer-like."""
    lowered = dtype.casefold()
    return lowered.startswith("int") or lowered.startswith("uint")


def _is_string_like_dtype(dtype: str) -> bool:
    """Return True when a profiler dtype string is string or categorical."""
    lowered = dtype.casefold()
    return (
        lowered == "string"
        or lowered == "utf8"
        or lowered.startswith("categorical")
        or lowered.startswith("enum")
    )


def _is_boolean_dtype(dtype: str) -> bool:
    """Return True when a profiler dtype string is Boolean/Bool."""
    lowered = dtype.casefold()
    return lowered in {"boolean", "bool"}


def _is_temporal_dtype(dtype: str) -> bool:
    """Return True when a profiler dtype string is temporal."""
    lowered = dtype.casefold()
    return (
        lowered == "date"
        or lowered.startswith("datetime")
        or lowered == "time"
    )


def _append_unique(items: list[str], value: str) -> None:
    """Append ``value`` when it is not already present."""
    if value not in items:
        items.append(value)


def _append_unique_task(items: list[AnalysisTask], value: AnalysisTask) -> None:
    """Append ``value`` when it is not already present."""
    if value not in items:
        items.append(value)


def _append_unique_warning(warnings: list[str], message: str) -> None:
    """Append a warning message when it is not already present."""
    if message not in warnings:
        warnings.append(message)


class TaskRoutingPolicy(BaseModel):
    """Thresholds that control analysis-task selection vs confirmation."""

    classification_max_unique_values: int = 20
    classification_max_cardinality_ratio: float = 0.05
    minimum_target_non_null_values: int = 2
    prefer_time_series_anomaly: bool = True

    @field_validator("classification_max_unique_values", mode="before")
    @classmethod
    def _validate_classification_max_unique_values(cls, value: object) -> int:
        return _validate_int_at_least_two(
            value, field_name="classification_max_unique_values"
        )

    @field_validator("minimum_target_non_null_values", mode="before")
    @classmethod
    def _validate_minimum_target_non_null_values(cls, value: object) -> int:
        return _validate_int_at_least_two(
            value, field_name="minimum_target_non_null_values"
        )

    @field_validator("classification_max_cardinality_ratio", mode="before")
    @classmethod
    def _validate_classification_max_cardinality_ratio(cls, value: object) -> float:
        return _validate_unit_interval(
            value, field_name="classification_max_cardinality_ratio"
        )

    @field_validator("prefer_time_series_anomaly", mode="before")
    @classmethod
    def _validate_prefer_time_series_anomaly(cls, value: object) -> bool:
        if not isinstance(value, bool):
            raise ValueError(
                "prefer_time_series_anomaly must be a bool, "
                f"got {type(value).__name__}"
            )
        return value


class TaskRoutingResult(BaseModel):
    """Outcome of analysis-task routing, including candidates and decision."""

    decision: DecisionLiteral
    selected_task: AnalysisTask
    selected_target: str | None = None
    selected_time_column: str | None = None
    candidate_tasks: list[AnalysisTask] = Field(default_factory=list)
    target_candidates: list[str] = Field(default_factory=list)
    time_candidates: list[str] = Field(default_factory=list)
    requires_user_confirmation: bool
    user_confirmed: bool
    reason: str
    evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("reason", mode="after")
    @classmethod
    def _reject_blank_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason must not be empty or whitespace-only")
        return value

    @model_validator(mode="after")
    def _validate_result_consistency(self) -> Self:
        """Enforce decision flags, candidate membership, and task requirements."""
        if self.decision == "AUTO_SELECTED":
            if self.requires_user_confirmation or self.user_confirmed:
                raise ValueError(
                    "AUTO_SELECTED requires requires_user_confirmation=False "
                    "and user_confirmed=False"
                )
        elif self.decision == "CONFIRMATION_REQUIRED":
            if not self.requires_user_confirmation or self.user_confirmed:
                raise ValueError(
                    "CONFIRMATION_REQUIRED requires "
                    "requires_user_confirmation=True and user_confirmed=False"
                )
        elif self.decision == "USER_CONFIRMED":
            if self.requires_user_confirmation or not self.user_confirmed:
                raise ValueError(
                    "USER_CONFIRMED requires requires_user_confirmation=False "
                    "and user_confirmed=True"
                )

        if self.selected_task not in self.candidate_tasks:
            raise ValueError("selected_task must be included in candidate_tasks")

        if len(self.candidate_tasks) != len(set(self.candidate_tasks)):
            raise ValueError("candidate_tasks must not contain duplicates")

        if len(self.target_candidates) != len(set(self.target_candidates)):
            raise ValueError("target_candidates must not contain duplicates")

        if len(self.time_candidates) != len(set(self.time_candidates)):
            raise ValueError("time_candidates must not contain duplicates")

        if (
            self.selected_target is not None
            and self.selected_target not in self.target_candidates
        ):
            raise ValueError("selected_target must be included in target_candidates")

        if (
            self.selected_time_column is not None
            and self.selected_time_column not in self.time_candidates
        ):
            raise ValueError(
                "selected_time_column must be included in time_candidates"
            )

        if self.selected_task in {
            AnalysisTask.REGRESSION,
            AnalysisTask.CLASSIFICATION,
            AnalysisTask.RESIDUAL_ANOMALY,
        }:
            if self.selected_target is None:
                raise ValueError(
                    f"{self.selected_task} requires selected_target"
                )

        if self.selected_task in {
            AnalysisTask.TIME_SERIES_ANOMALY,
            AnalysisTask.DRIFT_DETECTION,
        }:
            if self.selected_time_column is None:
                raise ValueError(
                    f"{self.selected_task} requires selected_time_column"
                )

        return self


class AnalysisTaskRouter:
    """Select an analysis task from profile statistics and role mapping."""

    def __init__(self, policy: TaskRoutingPolicy | None = None) -> None:
        """Create a router with an isolated deep-copied policy.

        Args:
            policy: Routing thresholds. When ``None``, defaults are used.
                The provided policy is deep-copied so later mutations do not
                affect this router.

        Raises:
            TypeError: If ``policy`` is not ``None`` or a ``TaskRoutingPolicy``.
        """
        if policy is None:
            self._policy = TaskRoutingPolicy()
        elif not isinstance(policy, TaskRoutingPolicy):
            raise TypeError(
                f"policy must be TaskRoutingPolicy, got {type(policy).__name__}"
            )
        else:
            self._policy = policy.model_copy(deep=True)

    def route(
        self,
        dataset_profile: DatasetProfile,
        role_mapping: ColumnRoleMappingResult,
        *,
        confirmed_task: AnalysisTask | None = None,
        confirmed_target: str | None = None,
        confirmed_time_column: str | None = None,
    ) -> TaskRoutingResult:
        """Route an analysis task from profile and role-mapping evidence.

        Args:
            dataset_profile: Lightweight profile with per-column statistics.
            role_mapping: Column-role assignments aligned to profile columns.
            confirmed_task: Optional user-confirmed analysis task.
            confirmed_target: Optional user-confirmed target column name.
            confirmed_time_column: Optional user-confirmed time column name.

        Returns:
            A ``TaskRoutingResult`` describing the selected path and candidates.

        Raises:
            TypeError: If input types are invalid.
            ValueError: If confirmed column names are blank.
            DataValidationError: If profile/mapping consistency or confirmed
                column constraints fail.
            InsufficientDataError: If a confirmed supervised path lacks enough
                non-null target values.
        """
        self._validate_dataset_profile(dataset_profile)
        self._validate_role_mapping(dataset_profile, role_mapping)
        self._validate_confirmed_task(confirmed_task)
        column_by_name = {column.name: column for column in dataset_profile.columns}
        confirmed_target = self._validate_confirmed_column(
            confirmed_target,
            column_by_name=column_by_name,
            field_name="confirmed_target",
        )
        confirmed_time_column = self._validate_confirmed_column(
            confirmed_time_column,
            column_by_name=column_by_name,
            field_name="confirmed_time_column",
        )

        target_candidates = self._collect_role_candidates(
            role_mapping,
            role=ColumnRole.TARGET_QUALITY,
        )
        time_candidates = self._collect_role_candidates(
            role_mapping,
            role=ColumnRole.TIME,
        )
        if confirmed_target is not None:
            _append_unique(target_candidates, confirmed_target)
        if confirmed_time_column is not None:
            _append_unique(time_candidates, confirmed_time_column)

        low_confidence = set(role_mapping.low_confidence_columns)
        policy = self._policy

        if confirmed_task is not None:
            return self._route_user_confirmed(
                dataset_profile=dataset_profile,
                column_by_name=column_by_name,
                policy=policy,
                confirmed_task=confirmed_task,
                confirmed_target=confirmed_target,
                confirmed_time_column=confirmed_time_column,
                target_candidates=target_candidates,
                time_candidates=time_candidates,
            )

        if confirmed_target is not None:
            return self._route_single_target(
                dataset_profile=dataset_profile,
                column_by_name=column_by_name,
                policy=policy,
                target_name=confirmed_target,
                target_candidates=target_candidates,
                time_candidates=time_candidates,
                low_confidence=low_confidence,
                confirmed_time_column=confirmed_time_column,
            )

        if len(target_candidates) >= 2:
            return self._route_multiple_targets(
                dataset_profile=dataset_profile,
                column_by_name=column_by_name,
                policy=policy,
                target_candidates=target_candidates,
                time_candidates=time_candidates,
                low_confidence=low_confidence,
            )

        if len(target_candidates) == 1:
            return self._route_single_target(
                dataset_profile=dataset_profile,
                column_by_name=column_by_name,
                policy=policy,
                target_name=target_candidates[0],
                target_candidates=target_candidates,
                time_candidates=time_candidates,
                low_confidence=low_confidence,
                confirmed_time_column=confirmed_time_column,
            )

        return self._route_anomaly_path(
            policy=policy,
            target_candidates=target_candidates,
            time_candidates=time_candidates,
            low_confidence=low_confidence,
            preferred_time=confirmed_time_column,
            ignore_multiple_time_ambiguity=confirmed_time_column is not None,
            force_confirmation=False,
            extra_warnings=[],
            extra_evidence=[],
        )

    def _route_user_confirmed(
        self,
        *,
        dataset_profile: DatasetProfile,
        column_by_name: dict[str, ColumnProfile],
        policy: TaskRoutingPolicy,
        confirmed_task: AnalysisTask,
        confirmed_target: str | None,
        confirmed_time_column: str | None,
        target_candidates: list[str],
        time_candidates: list[str],
    ) -> TaskRoutingResult:
        selected_target = confirmed_target
        selected_time = confirmed_time_column
        warnings: list[str] = []
        evidence: list[str] = [
            f"Target candidates: {target_candidates}",
            f"Time candidates: {time_candidates}",
        ]

        if confirmed_task in {
            AnalysisTask.REGRESSION,
            AnalysisTask.CLASSIFICATION,
            AnalysisTask.RESIDUAL_ANOMALY,
        }:
            if confirmed_target is None:
                raise DataValidationError(
                    f"{confirmed_task} requires confirmed_target"
                )
            target_profile = column_by_name[confirmed_target]
            non_null = self._non_null_count(dataset_profile, target_profile)
            if non_null < policy.minimum_target_non_null_values:
                raise InsufficientDataError(
                    f"Target {confirmed_target!r} has insufficient non-null "
                    f"values ({non_null}) for supervised task {confirmed_task}"
                )
            self._validate_confirmed_target_dtype(
                confirmed_task,
                target_profile=target_profile,
            )
            evidence.extend(
                [
                    f"Selected target dtype: {target_profile.dtype}",
                    (
                        f"Selected target unique_count="
                        f"{target_profile.unique_count}, "
                        f"cardinality_ratio={target_profile.cardinality_ratio}"
                    ),
                    (
                        "Policy thresholds: "
                        f"classification_max_unique_values="
                        f"{policy.classification_max_unique_values}, "
                        f"classification_max_cardinality_ratio="
                        f"{policy.classification_max_cardinality_ratio}, "
                        f"minimum_target_non_null_values="
                        f"{policy.minimum_target_non_null_values}"
                    ),
                ]
            )
        elif confirmed_task in {
            AnalysisTask.TIME_SERIES_ANOMALY,
            AnalysisTask.DRIFT_DETECTION,
        }:
            if confirmed_time_column is None:
                raise DataValidationError(
                    f"{confirmed_task} requires confirmed_time_column"
                )
        elif confirmed_task == AnalysisTask.UNSUPERVISED_ANOMALY:
            pass
        else:
            raise DataValidationError(f"Unsupported confirmed task: {confirmed_task}")

        candidate_tasks = self._user_confirmed_candidate_tasks(
            confirmed_task,
            has_time=bool(time_candidates),
        )
        evidence.append(
            f"User explicitly confirmed analysis task {confirmed_task}"
        )
        return TaskRoutingResult(
            decision="USER_CONFIRMED",
            selected_task=confirmed_task,
            selected_target=selected_target,
            selected_time_column=selected_time,
            candidate_tasks=candidate_tasks,
            target_candidates=list(target_candidates),
            time_candidates=list(time_candidates),
            requires_user_confirmation=False,
            user_confirmed=True,
            reason=(
                "User explicitly confirmed the analysis task; "
                "user confirmation overrides automatic inference"
            ),
            evidence=evidence,
            warnings=warnings,
        )

    def _route_multiple_targets(
        self,
        *,
        dataset_profile: DatasetProfile,
        column_by_name: dict[str, ColumnProfile],
        policy: TaskRoutingPolicy,
        target_candidates: list[str],
        time_candidates: list[str],
        low_confidence: set[str],
    ) -> TaskRoutingResult:
        provisional_target = target_candidates[0]
        target_profile = column_by_name[provisional_target]
        inference = self._infer_supervised_from_target(
            target_profile=target_profile,
            policy=policy,
        )
        warnings = [
            "Multiple TARGET_QUALITY columns were found; target selection is required"
        ]
        if inference.warning is not None:
            _append_unique_warning(warnings, inference.warning)
        if provisional_target in low_confidence:
            _append_unique_warning(
                warnings,
                "Selected target has low role-mapping confidence",
            )

        evidence = self._build_supervised_evidence(
            target_candidates=target_candidates,
            time_candidates=time_candidates,
            target_profile=target_profile,
            policy=policy,
            selection_note=(
                f"Provisional task {inference.selected_task} from first "
                f"target candidate {provisional_target!r}"
            ),
        )
        return TaskRoutingResult(
            decision="CONFIRMATION_REQUIRED",
            selected_task=inference.selected_task,
            selected_target=provisional_target,
            selected_time_column=None,
            candidate_tasks=list(inference.candidate_tasks),
            target_candidates=list(target_candidates),
            time_candidates=list(time_candidates),
            requires_user_confirmation=True,
            user_confirmed=False,
            reason="Multiple target candidates require explicit target selection",
            evidence=evidence,
            warnings=warnings,
        )

    def _route_single_target(
        self,
        *,
        dataset_profile: DatasetProfile,
        column_by_name: dict[str, ColumnProfile],
        policy: TaskRoutingPolicy,
        target_name: str,
        target_candidates: list[str],
        time_candidates: list[str],
        low_confidence: set[str],
        confirmed_time_column: str | None,
    ) -> TaskRoutingResult:
        target_profile = column_by_name[target_name]
        non_null = self._non_null_count(dataset_profile, target_profile)
        if non_null < policy.minimum_target_non_null_values:
            insufficient_warnings = [
                (
                    f"Target {target_name!r} has insufficient non-null values "
                    f"({non_null}) for supervised routing"
                )
            ]
            extra_evidence = [
                (
                    f"Target {target_name!r} non-null count {non_null} is below "
                    f"minimum_target_non_null_values="
                    f"{policy.minimum_target_non_null_values}"
                )
            ]
            return self._route_anomaly_path(
                policy=policy,
                target_candidates=target_candidates,
                time_candidates=time_candidates,
                low_confidence=low_confidence,
                preferred_time=confirmed_time_column,
                ignore_multiple_time_ambiguity=confirmed_time_column is not None,
                force_confirmation=True,
                extra_warnings=insufficient_warnings,
                extra_evidence=extra_evidence,
            )

        inference = self._infer_supervised_from_target(
            target_profile=target_profile,
            policy=policy,
        )
        warnings: list[str] = []
        if inference.warning is not None:
            _append_unique_warning(warnings, inference.warning)

        decision: DecisionLiteral
        requires_confirmation: bool
        if inference.requires_confirmation:
            decision = "CONFIRMATION_REQUIRED"
            requires_confirmation = True
        else:
            decision = "AUTO_SELECTED"
            requires_confirmation = False

        if target_name in low_confidence:
            decision = "CONFIRMATION_REQUIRED"
            requires_confirmation = True
            _append_unique_warning(
                warnings,
                "Selected target has low role-mapping confidence",
            )

        reason = inference.reason
        if decision == "CONFIRMATION_REQUIRED" and not inference.requires_confirmation:
            reason = (
                f"Selected target {target_name!r} requires user confirmation "
                "due to low role-mapping confidence"
            )

        evidence = self._build_supervised_evidence(
            target_candidates=target_candidates,
            time_candidates=time_candidates,
            target_profile=target_profile,
            policy=policy,
            selection_note=inference.reason,
        )
        return TaskRoutingResult(
            decision=decision,
            selected_task=inference.selected_task,
            selected_target=target_name,
            selected_time_column=None,
            candidate_tasks=list(inference.candidate_tasks),
            target_candidates=list(target_candidates),
            time_candidates=list(time_candidates),
            requires_user_confirmation=requires_confirmation,
            user_confirmed=False,
            reason=reason,
            evidence=evidence,
            warnings=warnings,
        )

    def _route_anomaly_path(
        self,
        *,
        policy: TaskRoutingPolicy,
        target_candidates: list[str],
        time_candidates: list[str],
        low_confidence: set[str],
        preferred_time: str | None,
        ignore_multiple_time_ambiguity: bool,
        force_confirmation: bool,
        extra_warnings: list[str],
        extra_evidence: list[str],
    ) -> TaskRoutingResult:
        warnings = list(extra_warnings)
        evidence = [
            f"Target candidates: {target_candidates}",
            f"Time candidates: {time_candidates}",
            (
                "Policy thresholds: "
                f"prefer_time_series_anomaly="
                f"{policy.prefer_time_series_anomaly}, "
                f"minimum_target_non_null_values="
                f"{policy.minimum_target_non_null_values}"
            ),
            *extra_evidence,
        ]

        if not time_candidates:
            evidence.append(
                "No target or time information; selecting unsupervised anomaly"
            )
            return TaskRoutingResult(
                decision="AUTO_SELECTED"
                if not force_confirmation
                else "CONFIRMATION_REQUIRED",
                selected_task=AnalysisTask.UNSUPERVISED_ANOMALY,
                selected_target=None,
                selected_time_column=None,
                candidate_tasks=[AnalysisTask.UNSUPERVISED_ANOMALY],
                target_candidates=list(target_candidates),
                time_candidates=[],
                requires_user_confirmation=force_confirmation,
                user_confirmed=False,
                reason=(
                    "No target or time information available; "
                    "selected generic unsupervised anomaly detection"
                    if not force_confirmation
                    else (
                        "Supervised target data is insufficient and no time "
                        "column is available; selected unsupervised anomaly"
                    )
                ),
                evidence=evidence,
                warnings=warnings,
            )

        selected_time = (
            preferred_time if preferred_time is not None else time_candidates[0]
        )
        multiple_times = len(time_candidates) >= 2 and not ignore_multiple_time_ambiguity
        if multiple_times:
            _append_unique_warning(
                warnings,
                "Multiple TIME columns were found; time column selection is required",
            )

        if policy.prefer_time_series_anomaly:
            selected_task = AnalysisTask.TIME_SERIES_ANOMALY
            candidate_tasks = [
                AnalysisTask.TIME_SERIES_ANOMALY,
                AnalysisTask.UNSUPERVISED_ANOMALY,
                AnalysisTask.DRIFT_DETECTION,
            ]
            reason = (
                f"No usable target; selected time-series anomaly using "
                f"time column {selected_time!r}"
            )
        else:
            selected_task = AnalysisTask.UNSUPERVISED_ANOMALY
            candidate_tasks = [
                AnalysisTask.UNSUPERVISED_ANOMALY,
                AnalysisTask.TIME_SERIES_ANOMALY,
                AnalysisTask.DRIFT_DETECTION,
            ]
            reason = (
                f"No usable target; selected unsupervised anomaly with "
                f"available time column {selected_time!r}"
            )

        decision: DecisionLiteral = "AUTO_SELECTED"
        requires_confirmation = False
        if force_confirmation or multiple_times:
            decision = "CONFIRMATION_REQUIRED"
            requires_confirmation = True

        if selected_time in low_confidence:
            decision = "CONFIRMATION_REQUIRED"
            requires_confirmation = True
            _append_unique_warning(
                warnings,
                "Selected time column has low role-mapping confidence",
            )

        evidence.append(reason)
        return TaskRoutingResult(
            decision=decision,
            selected_task=selected_task,
            selected_target=None,
            selected_time_column=selected_time,
            candidate_tasks=candidate_tasks,
            target_candidates=list(target_candidates),
            time_candidates=list(time_candidates),
            requires_user_confirmation=requires_confirmation,
            user_confirmed=False,
            reason=reason
            if not multiple_times
            else (
                "Multiple time candidates require explicit time column selection"
            ),
            evidence=evidence,
            warnings=warnings,
        )

    def _infer_supervised_from_target(
        self,
        *,
        target_profile: ColumnProfile,
        policy: TaskRoutingPolicy,
    ) -> _SupervisedInference:
        dtype = target_profile.dtype
        if _is_boolean_dtype(dtype):
            return _SupervisedInference(
                selected_task=AnalysisTask.CLASSIFICATION,
                candidate_tasks=[
                    AnalysisTask.CLASSIFICATION,
                    AnalysisTask.RESIDUAL_ANOMALY,
                ],
                requires_confirmation=False,
                reason=(
                    f"Boolean target {target_profile.name!r} selected "
                    "classification"
                ),
                warning=None,
            )

        if _is_string_like_dtype(dtype):
            return _SupervisedInference(
                selected_task=AnalysisTask.CLASSIFICATION,
                candidate_tasks=[
                    AnalysisTask.CLASSIFICATION,
                    AnalysisTask.RESIDUAL_ANOMALY,
                ],
                requires_confirmation=False,
                reason=(
                    f"String/categorical target {target_profile.name!r} "
                    "selected classification"
                ),
                warning=None,
            )

        if _is_numeric_dtype(dtype):
            low_cardinality = (
                target_profile.unique_count
                <= policy.classification_max_unique_values
                and target_profile.cardinality_ratio
                <= policy.classification_max_cardinality_ratio
            )
            if low_cardinality:
                return _SupervisedInference(
                    selected_task=AnalysisTask.CLASSIFICATION,
                    candidate_tasks=[
                        AnalysisTask.CLASSIFICATION,
                        AnalysisTask.REGRESSION,
                        AnalysisTask.RESIDUAL_ANOMALY,
                    ],
                    requires_confirmation=True,
                    reason=(
                        f"Numeric target {target_profile.name!r} has low "
                        "cardinality and may be either a classification label "
                        "or a continuous regression target"
                    ),
                    warning=(
                        "Ambiguous numeric target: low cardinality may indicate "
                        "classification labels or continuous values"
                    ),
                )
            return _SupervisedInference(
                selected_task=AnalysisTask.REGRESSION,
                candidate_tasks=[
                    AnalysisTask.REGRESSION,
                    AnalysisTask.RESIDUAL_ANOMALY,
                ],
                requires_confirmation=False,
                reason=(
                    f"Continuous numeric target {target_profile.name!r} "
                    "selected regression"
                ),
                warning=None,
            )

        if _is_temporal_dtype(dtype):
            return _SupervisedInference(
                selected_task=AnalysisTask.REGRESSION,
                candidate_tasks=[
                    AnalysisTask.REGRESSION,
                    AnalysisTask.CLASSIFICATION,
                ],
                requires_confirmation=True,
                reason=(
                    f"Temporal target {target_profile.name!r} has unclear "
                    "analysis-task meaning"
                ),
                warning=(
                    "Temporal target dtype is ambiguous for supervised routing"
                ),
            )

        return _SupervisedInference(
            selected_task=AnalysisTask.REGRESSION,
            candidate_tasks=[
                AnalysisTask.REGRESSION,
                AnalysisTask.CLASSIFICATION,
            ],
            requires_confirmation=True,
            reason=(
                f"Target {target_profile.name!r} dtype {dtype!r} cannot "
                "determine the analysis task"
            ),
            warning="Unknown target dtype prevents automatic task confirmation",
        )

    @staticmethod
    def _user_confirmed_candidate_tasks(
        confirmed_task: AnalysisTask,
        *,
        has_time: bool,
    ) -> list[AnalysisTask]:
        related: list[AnalysisTask]
        if confirmed_task == AnalysisTask.REGRESSION:
            related = [
                AnalysisTask.REGRESSION,
                AnalysisTask.RESIDUAL_ANOMALY,
            ]
        elif confirmed_task == AnalysisTask.CLASSIFICATION:
            related = [
                AnalysisTask.CLASSIFICATION,
                AnalysisTask.RESIDUAL_ANOMALY,
            ]
        elif confirmed_task == AnalysisTask.RESIDUAL_ANOMALY:
            related = [
                AnalysisTask.RESIDUAL_ANOMALY,
                AnalysisTask.REGRESSION,
            ]
        elif confirmed_task == AnalysisTask.TIME_SERIES_ANOMALY:
            related = [
                AnalysisTask.TIME_SERIES_ANOMALY,
                AnalysisTask.UNSUPERVISED_ANOMALY,
                AnalysisTask.DRIFT_DETECTION,
            ]
        elif confirmed_task == AnalysisTask.DRIFT_DETECTION:
            related = [
                AnalysisTask.DRIFT_DETECTION,
                AnalysisTask.TIME_SERIES_ANOMALY,
                AnalysisTask.UNSUPERVISED_ANOMALY,
            ]
        elif confirmed_task == AnalysisTask.UNSUPERVISED_ANOMALY:
            if has_time:
                related = [
                    AnalysisTask.UNSUPERVISED_ANOMALY,
                    AnalysisTask.TIME_SERIES_ANOMALY,
                    AnalysisTask.DRIFT_DETECTION,
                ]
            else:
                related = [AnalysisTask.UNSUPERVISED_ANOMALY]
        else:
            related = [confirmed_task]

        candidates: list[AnalysisTask] = []
        _append_unique_task(candidates, confirmed_task)
        for task in related:
            _append_unique_task(candidates, task)
        return candidates

    @staticmethod
    def _validate_confirmed_target_dtype(
        confirmed_task: AnalysisTask,
        *,
        target_profile: ColumnProfile,
    ) -> None:
        dtype = target_profile.dtype
        name = target_profile.name
        if confirmed_task == AnalysisTask.REGRESSION:
            if not _is_numeric_dtype(dtype):
                raise DataValidationError(
                    f"REGRESSION requires a numeric target, got {name!r} "
                    f"with dtype {dtype!r}"
                )
            return

        if confirmed_task == AnalysisTask.CLASSIFICATION:
            if (
                _is_boolean_dtype(dtype)
                or _is_string_like_dtype(dtype)
                or _is_integer_dtype(dtype)
                or _is_numeric_dtype(dtype)
            ):
                return
            raise DataValidationError(
                f"CLASSIFICATION does not accept target {name!r} "
                f"with dtype {dtype!r}"
            )

        if confirmed_task == AnalysisTask.RESIDUAL_ANOMALY:
            if not _is_numeric_dtype(dtype):
                raise DataValidationError(
                    f"RESIDUAL_ANOMALY requires a numeric target, got "
                    f"{name!r} with dtype {dtype!r}"
                )

    @staticmethod
    def _build_supervised_evidence(
        *,
        target_candidates: list[str],
        time_candidates: list[str],
        target_profile: ColumnProfile,
        policy: TaskRoutingPolicy,
        selection_note: str,
    ) -> list[str]:
        return [
            f"Target candidates: {target_candidates}",
            f"Time candidates: {time_candidates}",
            f"Selected target dtype: {target_profile.dtype}",
            (
                f"Selected target unique_count={target_profile.unique_count}, "
                f"cardinality_ratio={target_profile.cardinality_ratio}"
            ),
            (
                "Policy thresholds: "
                f"classification_max_unique_values="
                f"{policy.classification_max_unique_values}, "
                f"classification_max_cardinality_ratio="
                f"{policy.classification_max_cardinality_ratio}, "
                f"minimum_target_non_null_values="
                f"{policy.minimum_target_non_null_values}"
            ),
            selection_note,
        ]

    @staticmethod
    def _collect_role_candidates(
        role_mapping: ColumnRoleMappingResult,
        *,
        role: ColumnRole,
    ) -> list[str]:
        candidates: list[str] = []
        for assignment in role_mapping.assignments:
            if assignment.role != role:
                continue
            if assignment.column_name == _ORIGINAL_ROW_ID:
                continue
            _append_unique(candidates, assignment.column_name)
        return candidates

    @staticmethod
    def _non_null_count(
        dataset_profile: DatasetProfile,
        column_profile: ColumnProfile,
    ) -> int:
        return dataset_profile.row_count - column_profile.null_count

    @staticmethod
    def _validate_dataset_profile(dataset_profile: object) -> None:
        if not isinstance(dataset_profile, DatasetProfile):
            raise TypeError(
                "dataset_profile must be DatasetProfile, "
                f"got {type(dataset_profile).__name__}"
            )
        names = [column.name for column in dataset_profile.columns]
        if len(names) != len(set(names)):
            raise DataValidationError(
                "dataset_profile.columns must not contain duplicate names"
            )

    @staticmethod
    def _validate_role_mapping(
        dataset_profile: DatasetProfile,
        role_mapping: object,
    ) -> None:
        if not isinstance(role_mapping, ColumnRoleMappingResult):
            raise TypeError(
                "role_mapping must be ColumnRoleMappingResult, "
                f"got {type(role_mapping).__name__}"
            )
        profile_names = [column.name for column in dataset_profile.columns]
        assignment_names = [
            assignment.column_name for assignment in role_mapping.assignments
        ]
        if profile_names != assignment_names:
            raise DataValidationError(
                "role_mapping.assignments column_name order must exactly "
                "match dataset_profile.columns order"
            )

    @staticmethod
    def _validate_confirmed_task(confirmed_task: object) -> None:
        if confirmed_task is None:
            return
        if not isinstance(confirmed_task, AnalysisTask):
            raise TypeError(
                "confirmed_task must be AnalysisTask or None, "
                f"got {type(confirmed_task).__name__}"
            )

    @staticmethod
    def _validate_confirmed_column(
        value: object,
        *,
        column_by_name: dict[str, ColumnProfile],
        field_name: str,
    ) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} must be str or None, got {type(value).__name__}"
            )
        if not value.strip():
            raise ValueError(
                f"{field_name} must be a non-empty non-whitespace string"
            )
        if value == _ORIGINAL_ROW_ID:
            raise DataValidationError(
                f"{field_name} cannot be {_ORIGINAL_ROW_ID!r}"
            )
        if value not in column_by_name:
            raise DataValidationError(
                f"{field_name} {value!r} is not present in dataset_profile"
            )
        return value


class _SupervisedInference:
    """Internal supervised-task inference bundle."""

    __slots__ = (
        "selected_task",
        "candidate_tasks",
        "requires_confirmation",
        "reason",
        "warning",
    )

    def __init__(
        self,
        *,
        selected_task: AnalysisTask,
        candidate_tasks: list[AnalysisTask],
        requires_confirmation: bool,
        reason: str,
        warning: str | None,
    ) -> None:
        self.selected_task = selected_task
        self.candidate_tasks = candidate_tasks
        self.requires_confirmation = requires_confirmation
        self.reason = reason
        self.warning = warning


def create_default_task_router(
    policy: TaskRoutingPolicy | None = None,
) -> AnalysisTaskRouter:
    """Create a new ``AnalysisTaskRouter`` with an optional policy.

    Args:
        policy: Optional routing policy. Deep-copied by ``AnalysisTaskRouter``.

    Returns:
        A newly constructed ``AnalysisTaskRouter``. Each call returns an
        independent instance rather than a shared singleton.
    """
    return AnalysisTaskRouter(policy=policy)
