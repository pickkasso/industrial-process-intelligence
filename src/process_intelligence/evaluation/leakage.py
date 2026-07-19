"""Structural data-leakage checks over splits, features, and preprocessing (Step 5B).

Inspects partition membership, role mapping, feature names, and preprocessing
fit-scope metadata. Does not scan feature/target values, train models, or
mutate inputs.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Self

import polars as pl
from pydantic import BaseModel, Field, field_validator, model_validator

from process_intelligence.core.enums import ColumnRole
from process_intelligence.core.exceptions import DataLeakageError, DataValidationError
from process_intelligence.core.schemas import PreprocessingEvent
from process_intelligence.data.loader import ORIGINAL_ROW_ID_COLUMN
from process_intelligence.evaluation.splitting import (
    DatasetSplit,
    SplitStrategy,
)
from process_intelligence.routing.schema_mapper import ColumnRoleMappingResult

_FIT_SENSITIVE_STEP_NAMES = frozenset(
    {
        "impute_missing_values",
        "scale_numeric_features",
    }
)
_PARTITION_ORDER = ("train", "validation", "test")
_NON_TOKEN_CHARS = re.compile(r"[^0-9a-z\uac00-\ud7a3]+")

_DEFAULT_FUTURE_FEATURE_TOKENS: tuple[str, ...] = (
    "future",
    "next",
    "lead",
    "post",
    "after",
)
_DEFAULT_TARGET_DERIVED_TOKENS: tuple[str, ...] = (
    "actual",
    "observed",
    "residual",
    "error",
    "difference",
    "delta",
    "future",
    "next",
)
_DEFAULT_OUTCOME_TOKENS: tuple[str, ...] = (
    "outcome",
    "failure",
    "completion",
    "completed",
    "end",
    "result",
)
_DEFAULT_TEMPORAL_TOKENS: tuple[str, ...] = (
    "time",
    "timestamp",
    "date",
    "datetime",
)


class LeakageIssueType(StrEnum):
    """Deterministic taxonomy of leakage risks reported by ``LeakageChecker``."""

    ORIGINAL_ROW_ID_OVERLAP = "ORIGINAL_ROW_ID_OVERLAP"
    SPLIT_SUMMARY_MISMATCH = "SPLIT_SUMMARY_MISMATCH"
    GROUP_PARTITION_OVERLAP = "GROUP_PARTITION_OVERLAP"
    TIME_ORDER_VIOLATION = "TIME_ORDER_VIOLATION"
    TARGET_INCLUDED_AS_FEATURE = "TARGET_INCLUDED_AS_FEATURE"
    IDENTIFIER_INCLUDED_AS_FEATURE = "IDENTIFIER_INCLUDED_AS_FEATURE"
    TARGET_DERIVED_FEATURE = "TARGET_DERIVED_FEATURE"
    FUTURE_DERIVED_FEATURE = "FUTURE_DERIVED_FEATURE"
    OUTCOME_TIMESTAMP_FEATURE = "OUTCOME_TIMESTAMP_FEATURE"
    PREPROCESSING_FIT_SCOPE_UNKNOWN = "PREPROCESSING_FIT_SCOPE_UNKNOWN"
    PREPROCESSING_NOT_FIT_ON_TRAINING_DATA = "PREPROCESSING_NOT_FIT_ON_TRAINING_DATA"


class LeakageSeverity(StrEnum):
    """Severity of a leakage finding: blockers fail safety, warnings do not."""

    BLOCKER = "BLOCKER"
    WARNING = "WARNING"


class LeakageCheckConfig(BaseModel):
    """Tunable rules for structural leakage inspection.

    Token collections are normalized (strip + casefold), stored as independent
    tuples, and rejected when empty, whitespace-only, or duplicated.
    """

    allow_identifier_features: bool = False
    require_split_summary_match: bool = True
    require_preprocessing_fit_scope: bool = True
    future_feature_tokens: tuple[str, ...] = _DEFAULT_FUTURE_FEATURE_TOKENS
    target_derived_tokens: tuple[str, ...] = _DEFAULT_TARGET_DERIVED_TOKENS
    outcome_tokens: tuple[str, ...] = _DEFAULT_OUTCOME_TOKENS
    temporal_tokens: tuple[str, ...] = _DEFAULT_TEMPORAL_TOKENS

    @field_validator(
        "allow_identifier_features",
        "require_split_summary_match",
        "require_preprocessing_fit_scope",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError(f"must be a bool, got {type(value).__name__}")
        return value

    @field_validator(
        "future_feature_tokens",
        "target_derived_tokens",
        "outcome_tokens",
        "temporal_tokens",
        mode="before",
    )
    @classmethod
    def _normalize_token_tuple(cls, value: object) -> tuple[str, ...]:
        if isinstance(value, (str, bytes)):
            raise ValueError(
                "token collection must be a sequence of strings, "
                f"got {type(value).__name__}"
            )
        if not isinstance(value, Sequence):
            raise ValueError(
                "token collection must be a sequence of strings, "
                f"got {type(value).__name__}"
            )

        normalized: list[str] = []
        seen: set[str] = set()
        for index, item in enumerate(value):
            if not isinstance(item, str) or isinstance(item, bytes):
                raise ValueError(
                    f"token at index {index} must be a non-empty str, "
                    f"got {type(item).__name__}"
                )
            if item == "" or item.strip() == "":
                raise ValueError(
                    f"token at index {index} must be a non-empty, "
                    "non-whitespace string"
                )
            token = item.strip().casefold()
            if token in seen:
                raise ValueError(f"duplicate token {token!r} is not allowed")
            seen.add(token)
            normalized.append(token)
        return tuple(normalized)


class LeakageIssue(BaseModel):
    """Single structured leakage finding with remediation guidance."""

    issue_type: LeakageIssueType
    severity: LeakageSeverity
    columns: list[str] = Field(default_factory=list)
    partitions: list[str] = Field(default_factory=list)
    message: str
    suggested_action: str

    @field_validator("columns", "partitions", mode="after")
    @classmethod
    def _reject_duplicate_names(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("must not contain duplicate values")
        return list(value)

    @field_validator("message", "suggested_action", mode="after")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"must be str, got {type(value).__name__}")
        if value == "" or value.strip() == "":
            raise ValueError("must be a non-empty, non-whitespace string")
        return value


class LeakageReport(BaseModel):
    """Aggregated leakage inspection result with safety and count summaries."""

    is_safe: bool
    issues: list[LeakageIssue] = Field(default_factory=list)
    blocker_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    checked_feature_columns: list[str] = Field(default_factory=list)
    checked_preprocessing_event_count: int = Field(ge=0)

    @field_validator("checked_feature_columns", mode="after")
    @classmethod
    def _reject_duplicate_features(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("checked_feature_columns must not contain duplicates")
        return list(value)

    @model_validator(mode="after")
    def _validate_counts_and_safety(self) -> Self:
        expected_blockers = sum(
            1 for issue in self.issues if issue.severity is LeakageSeverity.BLOCKER
        )
        expected_warnings = sum(
            1 for issue in self.issues if issue.severity is LeakageSeverity.WARNING
        )
        if self.blocker_count != expected_blockers:
            raise ValueError(
                "blocker_count must equal the number of BLOCKER issues "
                f"({expected_blockers}), got {self.blocker_count}"
            )
        if self.warning_count != expected_warnings:
            raise ValueError(
                "warning_count must equal the number of WARNING issues "
                f"({expected_warnings}), got {self.warning_count}"
            )
        expected_safe = self.blocker_count == 0
        if self.is_safe != expected_safe:
            raise ValueError(
                "is_safe must be True only when blocker_count is 0, "
                f"got is_safe={self.is_safe} with blocker_count={self.blocker_count}"
            )
        return self


class LeakageChecker:
    """Stateless structural leakage inspector for splits and feature metadata.

    Holds an isolated deep copy of ``LeakageCheckConfig``. Does not cache
    check results or mutate caller-owned inputs.
    """

    def __init__(self, config: LeakageCheckConfig | None = None) -> None:
        """Create a checker with an isolated configuration copy.

        Args:
            config: Optional check configuration. When ``None``, defaults are
                used. The provided config is deep-copied so later mutations do
                not affect this checker.

        Raises:
            TypeError: If ``config`` is not ``None`` or a ``LeakageCheckConfig``.
        """
        if config is None:
            self._config = LeakageCheckConfig()
        elif not isinstance(config, LeakageCheckConfig):
            raise TypeError(
                "config must be LeakageCheckConfig, "
                f"got {type(config).__name__}"
            )
        else:
            self._config = config.model_copy(deep=True)

    def check(
        self,
        split: DatasetSplit,
        role_mapping: ColumnRoleMappingResult,
        *,
        target_column: str | None = None,
        feature_columns: Sequence[str] | None = None,
        preprocessing_events: Sequence[PreprocessingEvent] = (),
    ) -> LeakageReport:
        """Inspect split membership, features, and preprocessing for leakage.

        Args:
            split: Train/validation/test partitions with summary metadata.
            role_mapping: Column role assignments used for identifier checks.
            target_column: Optional target column name.
            feature_columns: Explicit feature list, or ``None`` to use
                ``use_in_model=True`` assignments in original order.
            preprocessing_events: Preprocessing lineage events to inspect.

        Returns:
            A ``LeakageReport`` describing blockers and warnings. Warning-only
            reports are considered safe.

        Raises:
            TypeError: If an input has an unexpected type.
            DataValidationError: If split schema, target, or features are invalid.
            ValueError: If ``target_column`` is blank.
        """
        if not isinstance(split, DatasetSplit):
            raise TypeError(
                f"split must be DatasetSplit, got {type(split).__name__}"
            )
        if not isinstance(role_mapping, ColumnRoleMappingResult):
            raise TypeError(
                "role_mapping must be ColumnRoleMappingResult, "
                f"got {type(role_mapping).__name__}"
            )

        _validate_partition_frames(split)
        schema_columns = list(split.train.columns)
        resolved_target = _resolve_target_column(target_column, schema_columns)
        resolved_features = _resolve_feature_columns(
            feature_columns,
            role_mapping=role_mapping,
            schema_columns=schema_columns,
        )
        events = _validate_preprocessing_events(preprocessing_events)

        issues: list[LeakageIssue] = []
        issues.extend(_check_original_row_id_overlap(split))
        if self._config.require_split_summary_match:
            mismatch = _check_split_summary_mismatch(split)
            if mismatch is not None:
                issues.append(mismatch)
        if split.summary.strategy is SplitStrategy.GROUP:
            group_issue = _check_group_partition_overlap(split)
            if group_issue is not None:
                issues.append(group_issue)
        if split.summary.strategy is SplitStrategy.TIME:
            time_issue = _check_time_order_violation(split)
            if time_issue is not None:
                issues.append(time_issue)

        issues.extend(
            _check_target_included_as_feature(
                resolved_target,
                resolved_features,
            )
        )
        issues.extend(
            _check_identifier_features(
                resolved_features,
                role_mapping=role_mapping,
                allow_identifier_features=self._config.allow_identifier_features,
            )
        )
        issues.extend(
            _check_target_derived_features(
                resolved_target,
                resolved_features,
                target_derived_tokens=self._config.target_derived_tokens,
            )
        )
        issues.extend(
            _check_future_derived_features(
                resolved_features,
                future_feature_tokens=self._config.future_feature_tokens,
            )
        )
        issues.extend(
            _check_outcome_timestamp_features(
                resolved_features,
                outcome_tokens=self._config.outcome_tokens,
                temporal_tokens=self._config.temporal_tokens,
            )
        )
        if self._config.require_preprocessing_fit_scope:
            issues.extend(_check_preprocessing_fit_scope(events))

        blocker_count = sum(
            1 for issue in issues if issue.severity is LeakageSeverity.BLOCKER
        )
        warning_count = sum(
            1 for issue in issues if issue.severity is LeakageSeverity.WARNING
        )
        return LeakageReport(
            is_safe=blocker_count == 0,
            issues=issues,
            blocker_count=blocker_count,
            warning_count=warning_count,
            checked_feature_columns=list(resolved_features),
            checked_preprocessing_event_count=len(events),
        )

    def assert_safe(self, report: LeakageReport) -> None:
        """Raise ``DataLeakageError`` when a report contains blocker issues.

        Args:
            report: Leakage report previously produced by ``check``.

        Raises:
            TypeError: If ``report`` is not a ``LeakageReport``.
            DataLeakageError: If one or more BLOCKER issues are present.
        """
        if not isinstance(report, LeakageReport):
            raise TypeError(
                f"report must be LeakageReport, got {type(report).__name__}"
            )
        if report.blocker_count == 0:
            return

        blocker_types: list[str] = []
        seen_types: set[str] = set()
        for issue in report.issues:
            if issue.severity is not LeakageSeverity.BLOCKER:
                continue
            type_name = str(issue.issue_type)
            if type_name not in seen_types:
                seen_types.add(type_name)
                blocker_types.append(type_name)

        raise DataLeakageError(
            f"Data leakage blockers detected: count={report.blocker_count}, "
            f"types={', '.join(blocker_types)}"
        )


def _normalize_name_tokens(name: str) -> list[str]:
    """Normalize a column name into casefolded alphanumeric/Hangul tokens."""
    folded = name.casefold()
    spaced = _NON_TOKEN_CHARS.sub(" ", folded)
    collapsed = " ".join(spaced.split())
    if collapsed == "":
        return []
    return collapsed.split(" ")


def _validate_partition_frames(split: DatasetSplit) -> None:
    partitions = {
        "train": split.train,
        "validation": split.validation,
        "test": split.test,
    }
    for name, frame in partitions.items():
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(
                f"split.{name} must be a polars.DataFrame, "
                f"got {type(frame).__name__}"
            )

    train_columns = list(split.train.columns)
    for name in ("validation", "test"):
        other_columns = list(partitions[name].columns)
        if other_columns != train_columns:
            raise DataValidationError(
                f"partition schemas must match train column names and order; "
                f"{name} differs from train"
            )

    for name, frame in partitions.items():
        if ORIGINAL_ROW_ID_COLUMN not in frame.columns:
            raise DataValidationError(
                f"partition {name!r} is missing required column "
                f"'{ORIGINAL_ROW_ID_COLUMN}'"
            )


def _resolve_target_column(
    target_column: str | None,
    schema_columns: Sequence[str],
) -> str | None:
    if target_column is None:
        return None
    if not isinstance(target_column, str) or isinstance(target_column, bytes):
        raise TypeError(
            f"target_column must be None or str, got {type(target_column).__name__}"
        )
    if target_column == "" or target_column.strip() == "":
        raise ValueError("target_column must be a non-empty, non-whitespace string")
    if target_column == ORIGINAL_ROW_ID_COLUMN:
        raise DataValidationError(
            f"target_column cannot be reserved column '{ORIGINAL_ROW_ID_COLUMN}'"
        )
    if target_column not in schema_columns:
        raise DataValidationError(
            f"target_column {target_column!r} is not present in partition schema"
        )
    return target_column


def _resolve_feature_columns(
    feature_columns: Sequence[str] | None,
    *,
    role_mapping: ColumnRoleMappingResult,
    schema_columns: Sequence[str],
) -> list[str]:
    if feature_columns is None:
        return [
            assignment.column_name
            for assignment in role_mapping.assignments
            if assignment.use_in_model
        ]

    if isinstance(feature_columns, (str, bytes)):
        raise TypeError(
            "feature_columns must be a sequence of str, "
            f"got {type(feature_columns).__name__}"
        )
    if not isinstance(feature_columns, Sequence):
        raise TypeError(
            "feature_columns must be a sequence of str, "
            f"got {type(feature_columns).__name__}"
        )

    resolved: list[str] = []
    seen: set[str] = set()
    for index, name in enumerate(feature_columns):
        if not isinstance(name, str) or isinstance(name, bytes):
            raise TypeError(
                f"feature_columns[{index}] must be str, got {type(name).__name__}"
            )
        if name == "" or name.strip() == "":
            raise ValueError(
                f"feature_columns[{index}] must be a non-empty, "
                "non-whitespace string"
            )
        if name in seen:
            raise DataValidationError(
                f"feature_columns contains duplicate column {name!r}"
            )
        seen.add(name)
        resolved.append(name)

    missing = [name for name in resolved if name not in schema_columns]
    if missing:
        raise DataValidationError(
            "feature_columns contains columns not present in partition schema: "
            + ", ".join(missing)
        )
    return resolved


def _validate_preprocessing_events(
    preprocessing_events: Sequence[PreprocessingEvent],
) -> tuple[PreprocessingEvent, ...]:
    if isinstance(preprocessing_events, (str, bytes)):
        raise TypeError(
            "preprocessing_events must be a sequence of PreprocessingEvent, "
            f"got {type(preprocessing_events).__name__}"
        )
    if not isinstance(preprocessing_events, Sequence):
        raise TypeError(
            "preprocessing_events must be a sequence of PreprocessingEvent, "
            f"got {type(preprocessing_events).__name__}"
        )

    events: list[PreprocessingEvent] = []
    for index, event in enumerate(preprocessing_events):
        if not isinstance(event, PreprocessingEvent):
            raise TypeError(
                f"preprocessing_events[{index}] must be PreprocessingEvent, "
                f"got {type(event).__name__}"
            )
        events.append(event)
    return tuple(events)


def _partition_row_ids(frame: pl.DataFrame) -> list[Any]:
    return frame.get_column(ORIGINAL_ROW_ID_COLUMN).to_list()


def _ordered_overlap_partitions(*names: str) -> list[str]:
    selected = set(names)
    return [name for name in _PARTITION_ORDER if name in selected]


def _check_original_row_id_overlap(split: DatasetSplit) -> list[LeakageIssue]:
    train_ids = set(_partition_row_ids(split.train))
    validation_ids = set(_partition_row_ids(split.validation))
    test_ids = set(_partition_row_ids(split.test))

    pairs = (
        ("train", train_ids, "validation", validation_ids),
        ("train", train_ids, "test", test_ids),
        ("validation", validation_ids, "test", test_ids),
    )
    issues: list[LeakageIssue] = []
    for left_name, left_ids, right_name, right_ids in pairs:
        overlap = left_ids & right_ids
        if not overlap:
            continue
        issues.append(
            LeakageIssue(
                issue_type=LeakageIssueType.ORIGINAL_ROW_ID_OVERLAP,
                severity=LeakageSeverity.BLOCKER,
                columns=[ORIGINAL_ROW_ID_COLUMN],
                partitions=_ordered_overlap_partitions(left_name, right_name),
                message=(
                    f"Detected {len(overlap)} overlapping `_original_row_id` "
                    f"values between {left_name} and {right_name}"
                ),
                suggested_action=(
                    "Regenerate the dataset split so train/validation/test "
                    "partitions have disjoint `_original_row_id` values"
                ),
            )
        )
    return issues


def _check_split_summary_mismatch(split: DatasetSplit) -> LeakageIssue | None:
    comparisons = (
        ("train", _partition_row_ids(split.train), split.summary.train_original_row_ids),
        (
            "validation",
            _partition_row_ids(split.validation),
            split.summary.validation_original_row_ids,
        ),
        ("test", _partition_row_ids(split.test), split.summary.test_original_row_ids),
    )
    mismatched: list[str] = []
    for name, frame_ids, summary_ids in comparisons:
        if list(frame_ids) != list(summary_ids):
            mismatched.append(name)
    if not mismatched:
        return None
    return LeakageIssue(
        issue_type=LeakageIssueType.SPLIT_SUMMARY_MISMATCH,
        severity=LeakageSeverity.BLOCKER,
        columns=[ORIGINAL_ROW_ID_COLUMN],
        partitions=_ordered_overlap_partitions(*mismatched),
        message=(
            "Actual partition `_original_row_id` sequences do not match "
            "SplitSummary membership lists"
        ),
        suggested_action=(
            "Rebuild the split summary from the current partitions so "
            "summary IDs match the frames exactly"
        ),
    )


def _check_group_partition_overlap(split: DatasetSplit) -> LeakageIssue | None:
    train_groups = set(split.summary.train_groups)
    validation_groups = set(split.summary.validation_groups)
    test_groups = set(split.summary.test_groups)

    overlapping_partitions: set[str] = set()
    if train_groups & validation_groups:
        overlapping_partitions.update({"train", "validation"})
    if train_groups & test_groups:
        overlapping_partitions.update({"train", "test"})
    if validation_groups & test_groups:
        overlapping_partitions.update({"validation", "test"})
    if not overlapping_partitions:
        return None

    columns: list[str] = []
    if split.summary.group_column is not None:
        columns = [split.summary.group_column]

    return LeakageIssue(
        issue_type=LeakageIssueType.GROUP_PARTITION_OVERLAP,
        severity=LeakageSeverity.BLOCKER,
        columns=columns,
        partitions=_ordered_overlap_partitions(*overlapping_partitions),
        message=(
            "GROUP split has overlapping groups across partitions, "
            "which leaks sample identity across train/validation/test"
        ),
        suggested_action=(
            "Re-split the dataset at the group level so each group appears "
            "in only one partition"
        ),
    )


def _check_time_order_violation(split: DatasetSplit) -> LeakageIssue | None:
    has_validation = split.validation.height > 0
    train_range = split.summary.train_time_range
    validation_range = split.summary.validation_time_range
    test_range = split.summary.test_time_range

    if train_range is None or test_range is None:
        raise DataValidationError(
            "TIME split requires train_time_range and test_time_range "
            "in SplitSummary to verify chronology"
        )
    if has_validation and validation_range is None:
        raise DataValidationError(
            "TIME split with a non-empty validation partition requires "
            "validation_time_range in SplitSummary"
        )

    train_max = train_range[1]
    test_min = test_range[0]
    if train_max is None or test_min is None:
        raise DataValidationError(
            "TIME split time ranges must provide non-null bounds for "
            "chronology verification"
        )

    violated: set[str] = set()
    if has_validation:
        assert validation_range is not None
        validation_min = validation_range[0]
        validation_max = validation_range[1]
        if validation_min is None or validation_max is None:
            raise DataValidationError(
                "TIME split validation_time_range must provide non-null bounds"
            )
        if train_max > validation_min:
            violated.update({"train", "validation"})
        if validation_max > test_min:
            violated.update({"validation", "test"})
    elif train_max > test_min:
        violated.update({"train", "test"})

    if not violated:
        return None

    columns: list[str] = []
    if split.summary.time_column is not None:
        columns = [split.summary.time_column]

    return LeakageIssue(
        issue_type=LeakageIssueType.TIME_ORDER_VIOLATION,
        severity=LeakageSeverity.BLOCKER,
        columns=columns,
        partitions=_ordered_overlap_partitions(*violated),
        message=(
            "TIME split chronology is violated: future data appears in an "
            "earlier partition or partitions are temporally mixed"
        ),
        suggested_action=(
            "Re-run a chronological TIME split so earlier partitions only "
            "contain earlier timestamps"
        ),
    )


def _check_target_included_as_feature(
    target_column: str | None,
    feature_columns: Sequence[str],
) -> list[LeakageIssue]:
    if target_column is None:
        return []
    if target_column not in feature_columns:
        return []
    return [
        LeakageIssue(
            issue_type=LeakageIssueType.TARGET_INCLUDED_AS_FEATURE,
            severity=LeakageSeverity.BLOCKER,
            columns=[target_column],
            partitions=[],
            message=(
                f"Target column {target_column!r} is included in the feature list"
            ),
            suggested_action=(
                "Remove the target column from the model feature list"
            ),
        )
    ]


def _check_identifier_features(
    feature_columns: Sequence[str],
    *,
    role_mapping: ColumnRoleMappingResult,
    allow_identifier_features: bool,
) -> list[LeakageIssue]:
    if allow_identifier_features:
        return []

    identifier_names = {
        assignment.column_name
        for assignment in role_mapping.assignments
        if assignment.role is ColumnRole.IDENTIFIER
    }
    identifier_names.add(ORIGINAL_ROW_ID_COLUMN)

    issues: list[LeakageIssue] = []
    for column in feature_columns:
        if column not in identifier_names:
            continue
        issues.append(
            LeakageIssue(
                issue_type=LeakageIssueType.IDENTIFIER_INCLUDED_AS_FEATURE,
                severity=LeakageSeverity.BLOCKER,
                columns=[column],
                partitions=[],
                message=(
                    f"Identifier column {column!r} is used as a model feature, "
                    "risking ID memorization or sample identity leakage"
                ),
                suggested_action=(
                    "Remove identifier columns from the model feature list"
                ),
            )
        )
    return issues


def _check_target_derived_features(
    target_column: str | None,
    feature_columns: Sequence[str],
    *,
    target_derived_tokens: Sequence[str],
) -> list[LeakageIssue]:
    if target_column is None:
        return []

    target_tokens = _normalize_name_tokens(target_column)
    if not target_tokens:
        return []

    derived_token_set = set(target_derived_tokens)
    issues: list[LeakageIssue] = []
    for column in feature_columns:
        if column == target_column:
            continue
        feature_tokens = _normalize_name_tokens(column)
        if not feature_tokens:
            continue
        if not all(token in feature_tokens for token in target_tokens):
            continue
        if not any(token in derived_token_set for token in feature_tokens):
            continue
        issues.append(
            LeakageIssue(
                issue_type=LeakageIssueType.TARGET_DERIVED_FEATURE,
                severity=LeakageSeverity.BLOCKER,
                columns=[column],
                partitions=[],
                message=(
                    f"Feature {column!r} may be derived from the target or "
                    "outcome rather than available at prediction time"
                ),
                suggested_action=(
                    "Review when the feature was created and whether it used "
                    "the target or post-outcome information"
                ),
            )
        )
    return issues


def _check_future_derived_features(
    feature_columns: Sequence[str],
    *,
    future_feature_tokens: Sequence[str],
) -> list[LeakageIssue]:
    future_token_set = set(future_feature_tokens)
    issues: list[LeakageIssue] = []
    for column in feature_columns:
        feature_tokens = _normalize_name_tokens(column)
        if not any(token in future_token_set for token in feature_tokens):
            continue
        issues.append(
            LeakageIssue(
                issue_type=LeakageIssueType.FUTURE_DERIVED_FEATURE,
                severity=LeakageSeverity.BLOCKER,
                columns=[column],
                partitions=[],
                message=(
                    f"Feature {column!r} may use information available only "
                    "after the prediction cutoff"
                ),
                suggested_action=(
                    "Use only information known before the prediction cutoff"
                ),
            )
        )
    return issues


def _check_outcome_timestamp_features(
    feature_columns: Sequence[str],
    *,
    outcome_tokens: Sequence[str],
    temporal_tokens: Sequence[str],
) -> list[LeakageIssue]:
    outcome_token_set = set(outcome_tokens)
    temporal_token_set = set(temporal_tokens)
    issues: list[LeakageIssue] = []
    for column in feature_columns:
        feature_tokens = _normalize_name_tokens(column)
        has_outcome = any(token in outcome_token_set for token in feature_tokens)
        has_temporal = any(token in temporal_token_set for token in feature_tokens)
        if not (has_outcome and has_temporal):
            continue
        issues.append(
            LeakageIssue(
                issue_type=LeakageIssueType.OUTCOME_TIMESTAMP_FEATURE,
                severity=LeakageSeverity.BLOCKER,
                columns=[column],
                partitions=[],
                message=(
                    f"Feature {column!r} may be a timestamp knowable only after "
                    "the outcome occurs"
                ),
                suggested_action=(
                    "Verify whether the timestamp is available at prediction "
                    "time before including it as a feature"
                ),
            )
        )
    return issues


def _check_preprocessing_fit_scope(
    events: Sequence[PreprocessingEvent],
) -> list[LeakageIssue]:
    unknown_issues: list[LeakageIssue] = []
    not_fit_issues: list[LeakageIssue] = []

    for event in events:
        if event.step_name not in _FIT_SENSITIVE_STEP_NAMES:
            continue
        parameters = event.parameters
        if "fitted_on_training_data" not in parameters:
            unknown_issues.append(
                LeakageIssue(
                    issue_type=LeakageIssueType.PREPROCESSING_FIT_SCOPE_UNKNOWN,
                    severity=LeakageSeverity.WARNING,
                    columns=list(event.affected_columns),
                    partitions=list(_PARTITION_ORDER),
                    message=(
                        f"Preprocessing step {event.step_name!r} does not record "
                        "whether statistics were fitted on training data only"
                    ),
                    suggested_action=(
                        "Record that fit-sensitive preprocessing statistics "
                        "were fitted on the training partition only"
                    ),
                )
            )
            continue

        fitted_flag = parameters["fitted_on_training_data"]
        if fitted_flag is not True:
            not_fit_issues.append(
                LeakageIssue(
                    issue_type=LeakageIssueType.PREPROCESSING_NOT_FIT_ON_TRAINING_DATA,
                    severity=LeakageSeverity.BLOCKER,
                    columns=list(event.affected_columns),
                    partitions=list(_PARTITION_ORDER),
                    message=(
                        f"Preprocessing step {event.step_name!r} may have fitted "
                        "statistics using validation/test information"
                    ),
                    suggested_action=(
                        "Fit the preprocessor on the training partition only "
                        "before transforming validation and test data"
                    ),
                )
            )

    return unknown_issues + not_fit_issues
