"""Unit tests for structural LeakageChecker (Step 5B)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl
import pytest
from pydantic import ValidationError

from process_intelligence.core.enums import ColumnRole
from process_intelligence.core.exceptions import DataLeakageError, DataValidationError
from process_intelligence.core.schemas import ColumnRoleAssignment, PreprocessingEvent
from process_intelligence.data import (
    ORIGINAL_ROW_ID_COLUMN,
    DatasetPreprocessor,
    PreprocessorConfig,
)
from process_intelligence.data.profiler import ColumnProfile, DatasetProfile
from process_intelligence.evaluation import (
    DatasetSplit,
    DatasetSplitter,
    LeakageCheckConfig,
    LeakageChecker,
    LeakageIssue,
    LeakageIssueType,
    LeakageReport,
    LeakageSeverity,
    SplitConfig,
    SplitStrategy,
    SplitSummary,
)
from process_intelligence.evaluation.leakage import (
    LeakageCheckConfig as DirectLeakageCheckConfig,
)
from process_intelligence.evaluation.leakage import LeakageChecker as DirectLeakageChecker
from process_intelligence.evaluation.leakage import LeakageIssue as DirectLeakageIssue
from process_intelligence.evaluation.leakage import (
    LeakageIssueType as DirectLeakageIssueType,
)
from process_intelligence.evaluation.leakage import LeakageReport as DirectLeakageReport
from process_intelligence.evaluation.leakage import (
    LeakageSeverity as DirectLeakageSeverity,
)
from process_intelligence.routing import ColumnRoleMapper, ColumnRoleMappingResult
from process_intelligence.routing.schema_mapper import ColumnRoleMappingResult as DirectMapping


def _assignment(
    column_name: str,
    role: ColumnRole = ColumnRole.STATE_SENSOR,
    *,
    use_in_model: bool | None = None,
    confidence: float = 0.9,
) -> ColumnRoleAssignment:
    if use_in_model is None:
        use_in_model = role in {
            ColumnRole.CONTROLLABLE_PROCESS,
            ColumnRole.STATE_SENSOR,
            ColumnRole.CONTEXT,
            ColumnRole.DERIVED_FEATURE,
        }
    return ColumnRoleAssignment(
        column_name=column_name,
        role=role,
        confidence=confidence,
        evidence=["test"],
        alternative_roles=[],
        use_in_model=use_in_model,
    )


def _zero_counts() -> dict[ColumnRole, int]:
    return {role: 0 for role in ColumnRole}


def _role_mapping(
    *assignments: ColumnRoleAssignment,
) -> ColumnRoleMappingResult:
    counts = _zero_counts()
    for assignment in assignments:
        counts[assignment.role] += 1
    unresolved = [
        assignment.column_name
        for assignment in assignments
        if assignment.role == ColumnRole.UNKNOWN
    ]
    return ColumnRoleMappingResult(
        assignments=list(assignments),
        unresolved_columns=unresolved,
        low_confidence_columns=[],
        requires_user_confirmation=False,
        warnings=[],
        role_counts=counts,
    )


def _frame(
    row_ids: list[int],
    *,
    extra: dict[str, list[object]] | None = None,
) -> pl.DataFrame:
    data: dict[str, list[object]] = {
        "temp": [float(i) for i in row_ids],
        "pressure": [float(i) * 10 for i in row_ids],
        "yield": [float(i) * 0.1 for i in row_ids],
        ORIGINAL_ROW_ID_COLUMN: list(row_ids),
    }
    if extra:
        data.update(extra)
    return pl.DataFrame(data)


def _summary(
    *,
    strategy: SplitStrategy = SplitStrategy.RANDOM,
    train_ids: list[int],
    validation_ids: list[int] | None = None,
    test_ids: list[int],
    group_column: str | None = None,
    time_column: str | None = None,
    train_groups: list[object] | None = None,
    validation_groups: list[object] | None = None,
    test_groups: list[object] | None = None,
    train_time_range: tuple[object | None, object | None] | None = None,
    validation_time_range: tuple[object | None, object | None] | None = None,
    test_time_range: tuple[object | None, object | None] | None = None,
    bypass_validation: bool = False,
) -> SplitSummary:
    validation_ids = [] if validation_ids is None else validation_ids
    payload = {
        "strategy": strategy,
        "group_column": group_column,
        "time_column": time_column,
        "random_state": 42,
        "requested_test_size": 0.2,
        "requested_validation_size": 0.0 if not validation_ids else 0.2,
        "train_row_count": len(train_ids),
        "validation_row_count": len(validation_ids),
        "test_row_count": len(test_ids),
        "train_fraction": 0.6,
        "validation_fraction": 0.2 if validation_ids else 0.0,
        "test_fraction": 0.2,
        "train_original_row_ids": list(train_ids),
        "validation_original_row_ids": list(validation_ids),
        "test_original_row_ids": list(test_ids),
        "train_groups": [] if train_groups is None else list(train_groups),
        "validation_groups": (
            [] if validation_groups is None else list(validation_groups)
        ),
        "test_groups": [] if test_groups is None else list(test_groups),
        "train_time_range": train_time_range,
        "validation_time_range": validation_time_range,
        "test_time_range": test_time_range,
        "warnings": [],
    }
    if bypass_validation:
        return SplitSummary.model_construct(**payload)
    return SplitSummary(**payload)


def _split_from_ids(
    train_ids: list[int],
    test_ids: list[int],
    validation_ids: list[int] | None = None,
    *,
    summary: SplitSummary | None = None,
    extra_columns: dict[str, dict[str, list[object]]] | None = None,
) -> DatasetSplit:
    validation_ids = [] if validation_ids is None else validation_ids

    def _with_extra(ids: list[int], partition: str) -> pl.DataFrame:
        extra = None
        if extra_columns is not None and partition in extra_columns:
            extra = extra_columns[partition]
        return _frame(ids, extra=extra)

    train = _with_extra(train_ids, "train")
    validation = (
        _with_extra(validation_ids, "validation")
        if validation_ids
        else train.clear()
    )
    test = _with_extra(test_ids, "test")
    if summary is None:
        summary = _summary(
            train_ids=train_ids,
            validation_ids=validation_ids,
            test_ids=test_ids,
        )
    return DatasetSplit(
        train=train,
        validation=validation,
        test=test,
        summary=summary,
    )


def _safe_role_mapping() -> ColumnRoleMappingResult:
    return _role_mapping(
        _assignment("temp", ColumnRole.STATE_SENSOR),
        _assignment("pressure", ColumnRole.CONTROLLABLE_PROCESS),
        _assignment("yield", ColumnRole.TARGET_QUALITY, use_in_model=False),
        _assignment(ORIGINAL_ROW_ID_COLUMN, ColumnRole.IDENTIFIER, use_in_model=False),
    )


def _event(
    step_name: str,
    *,
    affected_columns: list[str] | None = None,
    parameters: dict[str, object] | None = None,
) -> PreprocessingEvent:
    return PreprocessingEvent(
        step_name=step_name,
        affected_columns=[] if affected_columns is None else list(affected_columns),
        rows_before=4,
        rows_after=4,
        parameters={} if parameters is None else dict(parameters),
        warnings=[],
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
    )


def _issue_types(report: LeakageReport) -> list[LeakageIssueType]:
    return [issue.issue_type for issue in report.issues]


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


def test_leakage_issue_type_values() -> None:
    expected = {
        "ORIGINAL_ROW_ID_OVERLAP",
        "SPLIT_SUMMARY_MISMATCH",
        "GROUP_PARTITION_OVERLAP",
        "TIME_ORDER_VIOLATION",
        "TARGET_INCLUDED_AS_FEATURE",
        "IDENTIFIER_INCLUDED_AS_FEATURE",
        "TARGET_DERIVED_FEATURE",
        "FUTURE_DERIVED_FEATURE",
        "OUTCOME_TIMESTAMP_FEATURE",
        "PREPROCESSING_FIT_SCOPE_UNKNOWN",
        "PREPROCESSING_NOT_FIT_ON_TRAINING_DATA",
    }
    assert {member.value for member in LeakageIssueType} == expected
    assert set(LeakageIssueType) == set(DirectLeakageIssueType)
    for member in LeakageIssueType:
        assert member.value == member.name


def test_leakage_severity_values() -> None:
    assert LeakageSeverity.BLOCKER == "BLOCKER"
    assert LeakageSeverity.WARNING == "WARNING"
    assert set(LeakageSeverity) == {"BLOCKER", "WARNING"}
    assert LeakageSeverity.BLOCKER is DirectLeakageSeverity.BLOCKER


def test_leakage_enums_from_string() -> None:
    assert LeakageIssueType("TARGET_INCLUDED_AS_FEATURE") is (
        LeakageIssueType.TARGET_INCLUDED_AS_FEATURE
    )
    assert LeakageSeverity("WARNING") is LeakageSeverity.WARNING


def test_leakage_enums_reject_invalid_string() -> None:
    with pytest.raises(ValueError):
        LeakageIssueType("NOT_A_REAL_ISSUE")
    with pytest.raises(ValueError):
        LeakageSeverity("INFO")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_default_leakage_check_config() -> None:
    config = LeakageCheckConfig()
    assert config.allow_identifier_features is False
    assert config.require_split_summary_match is True
    assert config.require_preprocessing_fit_scope is True
    assert config.future_feature_tokens == (
        "future",
        "next",
        "lead",
        "post",
        "after",
    )
    assert config.target_derived_tokens == (
        "actual",
        "observed",
        "residual",
        "error",
        "difference",
        "delta",
        "future",
        "next",
    )
    assert config.outcome_tokens == (
        "outcome",
        "failure",
        "completion",
        "completed",
        "end",
        "result",
    )
    assert config.temporal_tokens == ("time", "timestamp", "date", "datetime")
    assert isinstance(config, DirectLeakageCheckConfig)


@pytest.mark.parametrize(
    "field_name",
    [
        "allow_identifier_features",
        "require_split_summary_match",
        "require_preprocessing_fit_scope",
    ],
)
def test_config_rejects_non_bool(field_name: str) -> None:
    with pytest.raises(ValidationError):
        LeakageCheckConfig(**{field_name: 1})


def test_config_rejects_empty_and_blank_tokens() -> None:
    with pytest.raises(ValidationError):
        LeakageCheckConfig(future_feature_tokens=("future", ""))
    with pytest.raises(ValidationError):
        LeakageCheckConfig(target_derived_tokens=("  ",))


def test_config_rejects_duplicate_tokens() -> None:
    with pytest.raises(ValidationError):
        LeakageCheckConfig(outcome_tokens=("failure", "Failure"))


def test_config_normalizes_tokens() -> None:
    config = LeakageCheckConfig(temporal_tokens=(" Time ", "TIMESTAMP"))
    assert config.temporal_tokens == ("time", "timestamp")


def test_config_token_collections_are_independent() -> None:
    source = ["future", "next"]
    config = LeakageCheckConfig(future_feature_tokens=source)
    source.append("lead")
    assert config.future_feature_tokens == ("future", "next")


def test_config_round_trip() -> None:
    config = LeakageCheckConfig(allow_identifier_features=True)
    restored = LeakageCheckConfig.model_validate(config.model_dump())
    assert restored == config


# ---------------------------------------------------------------------------
# LeakageIssue and LeakageReport
# ---------------------------------------------------------------------------


def test_leakage_issue_creation() -> None:
    issue = LeakageIssue(
        issue_type=LeakageIssueType.TARGET_INCLUDED_AS_FEATURE,
        severity=LeakageSeverity.BLOCKER,
        columns=["yield"],
        partitions=[],
        message="target in features",
        suggested_action="remove target",
    )
    assert isinstance(issue, DirectLeakageIssue)
    assert issue.columns == ["yield"]


def test_leakage_issue_rejects_duplicate_columns_and_partitions() -> None:
    with pytest.raises(ValidationError):
        LeakageIssue(
            issue_type=LeakageIssueType.ORIGINAL_ROW_ID_OVERLAP,
            severity=LeakageSeverity.BLOCKER,
            columns=["a", "a"],
            partitions=[],
            message="overlap",
            suggested_action="resplit",
        )
    with pytest.raises(ValidationError):
        LeakageIssue(
            issue_type=LeakageIssueType.ORIGINAL_ROW_ID_OVERLAP,
            severity=LeakageSeverity.BLOCKER,
            columns=[],
            partitions=["train", "train"],
            message="overlap",
            suggested_action="resplit",
        )


def test_leakage_issue_rejects_blank_message_and_action() -> None:
    with pytest.raises(ValidationError):
        LeakageIssue(
            issue_type=LeakageIssueType.FUTURE_DERIVED_FEATURE,
            severity=LeakageSeverity.BLOCKER,
            message="   ",
            suggested_action="fix",
        )
    with pytest.raises(ValidationError):
        LeakageIssue(
            issue_type=LeakageIssueType.FUTURE_DERIVED_FEATURE,
            severity=LeakageSeverity.BLOCKER,
            message="future feature",
            suggested_action="",
        )


def test_safe_and_unsafe_leakage_report() -> None:
    safe = LeakageReport(
        is_safe=True,
        issues=[],
        blocker_count=0,
        warning_count=0,
        checked_feature_columns=["temp"],
        checked_preprocessing_event_count=0,
    )
    assert safe.is_safe is True
    assert isinstance(safe, DirectLeakageReport)

    issue = LeakageIssue(
        issue_type=LeakageIssueType.TARGET_INCLUDED_AS_FEATURE,
        severity=LeakageSeverity.BLOCKER,
        columns=["yield"],
        message="target in features",
        suggested_action="remove target",
    )
    unsafe = LeakageReport(
        is_safe=False,
        issues=[issue],
        blocker_count=1,
        warning_count=0,
        checked_feature_columns=["yield"],
        checked_preprocessing_event_count=0,
    )
    assert unsafe.is_safe is False


def test_report_rejects_count_and_safety_mismatches() -> None:
    issue = LeakageIssue(
        issue_type=LeakageIssueType.TARGET_INCLUDED_AS_FEATURE,
        severity=LeakageSeverity.BLOCKER,
        columns=["yield"],
        message="target in features",
        suggested_action="remove target",
    )
    with pytest.raises(ValidationError):
        LeakageReport(
            is_safe=False,
            issues=[issue],
            blocker_count=0,
            warning_count=0,
            checked_feature_columns=["yield"],
            checked_preprocessing_event_count=0,
        )
    warning = LeakageIssue(
        issue_type=LeakageIssueType.PREPROCESSING_FIT_SCOPE_UNKNOWN,
        severity=LeakageSeverity.WARNING,
        message="unknown fit scope",
        suggested_action="record fit scope",
    )
    with pytest.raises(ValidationError):
        LeakageReport(
            is_safe=True,
            issues=[warning],
            blocker_count=0,
            warning_count=0,
            checked_feature_columns=[],
            checked_preprocessing_event_count=1,
        )
    with pytest.raises(ValidationError):
        LeakageReport(
            is_safe=True,
            issues=[issue],
            blocker_count=1,
            warning_count=0,
            checked_feature_columns=["yield"],
            checked_preprocessing_event_count=0,
        )


def test_report_rejects_duplicate_features_and_negative_event_count() -> None:
    with pytest.raises(ValidationError):
        LeakageReport(
            is_safe=True,
            issues=[],
            blocker_count=0,
            warning_count=0,
            checked_feature_columns=["temp", "temp"],
            checked_preprocessing_event_count=0,
        )
    with pytest.raises(ValidationError):
        LeakageReport(
            is_safe=True,
            issues=[],
            blocker_count=0,
            warning_count=0,
            checked_feature_columns=[],
            checked_preprocessing_event_count=-1,
        )


def test_report_mutable_defaults_are_independent() -> None:
    first = LeakageReport(
        is_safe=True,
        blocker_count=0,
        warning_count=0,
        checked_preprocessing_event_count=0,
    )
    second = LeakageReport(
        is_safe=True,
        blocker_count=0,
        warning_count=0,
        checked_preprocessing_event_count=0,
    )
    first.issues.append(
        LeakageIssue(
            issue_type=LeakageIssueType.FUTURE_DERIVED_FEATURE,
            severity=LeakageSeverity.WARNING,
            message="x",
            suggested_action="y",
        )
    )
    first.checked_feature_columns.append("temp")
    assert second.issues == []
    assert second.checked_feature_columns == []


def test_report_round_trip() -> None:
    report = LeakageReport(
        is_safe=True,
        issues=[],
        blocker_count=0,
        warning_count=0,
        checked_feature_columns=["temp"],
        checked_preprocessing_event_count=2,
    )
    restored = LeakageReport.model_validate(report.model_dump())
    assert restored == report


# ---------------------------------------------------------------------------
# Checker input validation
# ---------------------------------------------------------------------------


def test_checker_construction_and_config_isolation() -> None:
    default_checker = LeakageChecker()
    assert isinstance(default_checker, DirectLeakageChecker)

    config = LeakageCheckConfig(allow_identifier_features=True)
    checker = LeakageChecker(config)
    config.allow_identifier_features = False
    split = _split_from_ids(
        [0, 1, 2],
        [3, 4],
        extra_columns={
            "train": {"sample_id": ["a", "b", "c"]},
            "test": {"sample_id": ["d", "e"]},
        },
    )
    split = DatasetSplit(
        train=split.train,
        validation=split.train.clear(),
        test=split.test,
        summary=split.summary,
    )
    mapping = _role_mapping(
        _assignment("sample_id", ColumnRole.IDENTIFIER, use_in_model=True),
        _assignment("temp", ColumnRole.STATE_SENSOR),
        _assignment(ORIGINAL_ROW_ID_COLUMN, ColumnRole.IDENTIFIER, use_in_model=False),
    )
    report = checker.check(
        split,
        mapping,
        feature_columns=["sample_id", "temp"],
    )
    assert LeakageIssueType.IDENTIFIER_INCLUDED_AS_FEATURE not in _issue_types(report)

    with pytest.raises(TypeError):
        LeakageChecker("not-a-config")  # type: ignore[arg-type]


def test_check_rejects_invalid_split_and_role_mapping_types() -> None:
    checker = LeakageChecker()
    mapping = _safe_role_mapping()
    with pytest.raises(TypeError):
        checker.check("split", mapping)  # type: ignore[arg-type]
    split = _split_from_ids([0, 1], [2, 3])
    with pytest.raises(TypeError):
        checker.check(split, "mapping")  # type: ignore[arg-type]


def test_check_rejects_schema_mismatch_and_missing_row_id() -> None:
    checker = LeakageChecker()
    mapping = _safe_role_mapping()
    train = _frame([0, 1])
    validation = train.clear()
    test = pl.DataFrame(
        {
            "temp": [1.0],
            "pressure": [10.0],
            ORIGINAL_ROW_ID_COLUMN: [2],
        }
    )
    split = DatasetSplit(
        train=train,
        validation=validation,
        test=test,
        summary=_summary(train_ids=[0, 1], test_ids=[2]),
    )
    with pytest.raises(DataValidationError):
        checker.check(split, mapping, feature_columns=["temp", "pressure"])

    bad_train = pl.DataFrame({"temp": [1.0], "pressure": [2.0]})
    bad_split = DatasetSplit(
        train=bad_train,
        validation=bad_train.clear(),
        test=bad_train.clear().vstack(
            pl.DataFrame({"temp": [3.0], "pressure": [4.0]})
        ),
        summary=_summary(train_ids=[0], test_ids=[1], bypass_validation=True),
    )
    with pytest.raises(DataValidationError):
        checker.check(bad_split, mapping, feature_columns=["temp"])


def test_check_rejects_invalid_target_and_features() -> None:
    checker = LeakageChecker()
    split = _split_from_ids([0, 1, 2], [3, 4])
    mapping = _safe_role_mapping()

    with pytest.raises(TypeError):
        checker.check(split, mapping, target_column=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        checker.check(split, mapping, target_column="  ")
    with pytest.raises(DataValidationError):
        checker.check(split, mapping, target_column="missing_target")
    with pytest.raises(DataValidationError):
        checker.check(split, mapping, target_column=ORIGINAL_ROW_ID_COLUMN)

    with pytest.raises(TypeError):
        checker.check(split, mapping, feature_columns="temp")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        checker.check(split, mapping, feature_columns=b"temp")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        checker.check(split, mapping, feature_columns=["temp", 1])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        checker.check(split, mapping, feature_columns=["temp", ""])
    with pytest.raises(DataValidationError):
        checker.check(split, mapping, feature_columns=["temp", "temp"])
    with pytest.raises(DataValidationError, match="ghost"):
        checker.check(split, mapping, feature_columns=["temp", "ghost"])


def test_feature_none_uses_use_in_model_order() -> None:
    checker = LeakageChecker()
    split = _split_from_ids([0, 1, 2], [3, 4])
    mapping = _role_mapping(
        _assignment("pressure", ColumnRole.CONTROLLABLE_PROCESS, use_in_model=True),
        _assignment("temp", ColumnRole.STATE_SENSOR, use_in_model=True),
        _assignment("yield", ColumnRole.TARGET_QUALITY, use_in_model=False),
        _assignment(ORIGINAL_ROW_ID_COLUMN, ColumnRole.IDENTIFIER, use_in_model=False),
    )
    report = checker.check(split, mapping)
    assert report.checked_feature_columns == ["pressure", "temp"]


def test_check_rejects_invalid_preprocessing_events() -> None:
    checker = LeakageChecker()
    split = _split_from_ids([0, 1], [2, 3])
    mapping = _safe_role_mapping()
    with pytest.raises(TypeError):
        checker.check(
            split,
            mapping,
            feature_columns=["temp"],
            preprocessing_events="impute",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        checker.check(
            split,
            mapping,
            feature_columns=["temp"],
            preprocessing_events=[object()],  # type: ignore[list-item]
        )


# ---------------------------------------------------------------------------
# Partition leakage
# ---------------------------------------------------------------------------


def test_disjoint_partitions_have_no_id_overlap_issue() -> None:
    checker = LeakageChecker()
    split = _split_from_ids([0, 1, 2], [3, 4], [5])
    report = checker.check(
        split,
        _safe_role_mapping(),
        feature_columns=["temp", "pressure"],
    )
    assert LeakageIssueType.ORIGINAL_ROW_ID_OVERLAP not in _issue_types(report)


@pytest.mark.parametrize(
    ("train_ids", "validation_ids", "test_ids", "expected_partitions"),
    [
        ([0, 1, 2], [2, 3], [4, 5], ["train", "validation"]),
        ([0, 1, 2], [], [2, 3], ["train", "test"]),
        ([0, 1], [2, 3], [3, 4], ["validation", "test"]),
    ],
)
def test_original_row_id_overlap_detection(
    train_ids: list[int],
    validation_ids: list[int],
    test_ids: list[int],
    expected_partitions: list[str],
) -> None:
    checker = LeakageChecker()
    summary = _summary(
        train_ids=list(range(100, 100 + len(train_ids))),
        validation_ids=list(range(200, 200 + len(validation_ids))),
        test_ids=list(range(300, 300 + len(test_ids))),
        bypass_validation=True,
    )
    split = _split_from_ids(
        train_ids,
        test_ids,
        validation_ids,
        summary=summary,
    )
    report = checker.check(
        split,
        _safe_role_mapping(),
        feature_columns=["temp", "pressure"],
    )
    overlap_issues = [
        issue
        for issue in report.issues
        if issue.issue_type is LeakageIssueType.ORIGINAL_ROW_ID_OVERLAP
    ]
    assert len(overlap_issues) == 1
    assert overlap_issues[0].severity is LeakageSeverity.BLOCKER
    assert overlap_issues[0].columns == [ORIGINAL_ROW_ID_COLUMN]
    assert overlap_issues[0].partitions == expected_partitions
    assert "overlap" in overlap_issues[0].message.lower()


def test_empty_validation_is_safe_for_overlap() -> None:
    checker = LeakageChecker()
    split = _split_from_ids([0, 1, 2], [3, 4])
    report = checker.check(
        split,
        _safe_role_mapping(),
        feature_columns=["temp"],
    )
    assert LeakageIssueType.ORIGINAL_ROW_ID_OVERLAP not in _issue_types(report)


@pytest.mark.parametrize("partition", ["train", "validation", "test"])
def test_split_summary_mismatch_detection(partition: str) -> None:
    checker = LeakageChecker()
    train_ids = [0, 1, 2]
    validation_ids = [3]
    test_ids = [4, 5]
    summary_train = list(train_ids)
    summary_validation = list(validation_ids)
    summary_test = list(test_ids)
    if partition == "train":
        summary_train = [0, 2, 1]
    elif partition == "validation":
        summary_validation = [99]
    else:
        summary_test = [5, 4]
    summary = _summary(
        train_ids=summary_train,
        validation_ids=summary_validation,
        test_ids=summary_test,
        bypass_validation=True,
    )
    split = _split_from_ids(
        train_ids,
        test_ids,
        validation_ids,
        summary=summary,
    )
    report = checker.check(
        split,
        _safe_role_mapping(),
        feature_columns=["temp"],
    )
    mismatch = [
        issue
        for issue in report.issues
        if issue.issue_type is LeakageIssueType.SPLIT_SUMMARY_MISMATCH
    ]
    assert len(mismatch) == 1
    assert mismatch[0].severity is LeakageSeverity.BLOCKER
    assert partition in mismatch[0].partitions


def test_summary_mismatch_skipped_when_config_false() -> None:
    checker = LeakageChecker(
        LeakageCheckConfig(require_split_summary_match=False)
    )
    summary = _summary(train_ids=[0, 2, 1], test_ids=[3, 4], bypass_validation=True)
    split = _split_from_ids([0, 1, 2], [3, 4], summary=summary)
    report = checker.check(
        split,
        _safe_role_mapping(),
        feature_columns=["temp"],
    )
    assert LeakageIssueType.SPLIT_SUMMARY_MISMATCH not in _issue_types(report)


def test_group_partition_overlap_detection() -> None:
    checker = LeakageChecker()
    summary = _summary(
        strategy=SplitStrategy.GROUP,
        train_ids=[0, 1],
        test_ids=[2, 3],
        group_column="batch",
        train_groups=["A", "B"],
        test_groups=["B", "C"],
        bypass_validation=True,
    )
    split = _split_from_ids([0, 1], [2, 3], summary=summary)
    report = checker.check(
        split,
        _safe_role_mapping(),
        feature_columns=["temp"],
    )
    issues = [
        issue
        for issue in report.issues
        if issue.issue_type is LeakageIssueType.GROUP_PARTITION_OVERLAP
    ]
    assert len(issues) == 1
    assert issues[0].columns == ["batch"]
    assert issues[0].partitions == ["train", "test"]


def test_group_overlap_skipped_for_non_group_strategy() -> None:
    checker = LeakageChecker()
    summary = _summary(
        strategy=SplitStrategy.RANDOM,
        train_ids=[0, 1],
        test_ids=[2, 3],
        train_groups=["A"],
        test_groups=["A"],
        bypass_validation=True,
    )
    split = _split_from_ids([0, 1], [2, 3], summary=summary)
    report = checker.check(
        split,
        _safe_role_mapping(),
        feature_columns=["temp"],
    )
    assert LeakageIssueType.GROUP_PARTITION_OVERLAP not in _issue_types(report)


@pytest.mark.parametrize(
    ("train_range", "validation_range", "test_range", "expected"),
    [
        (
            (date(2020, 1, 1), date(2020, 1, 10)),
            None,
            (date(2020, 1, 5), date(2020, 1, 20)),
            ["train", "test"],
        ),
        (
            (date(2020, 1, 1), date(2020, 1, 10)),
            (date(2020, 1, 5), date(2020, 1, 12)),
            (date(2020, 1, 13), date(2020, 1, 20)),
            ["train", "validation"],
        ),
        (
            (date(2020, 1, 1), date(2020, 1, 5)),
            (date(2020, 1, 6), date(2020, 1, 15)),
            (date(2020, 1, 10), date(2020, 1, 20)),
            ["validation", "test"],
        ),
    ],
)
def test_time_order_violation_detection(
    train_range: tuple[date, date],
    validation_range: tuple[date, date] | None,
    test_range: tuple[date, date],
    expected: list[str],
) -> None:
    checker = LeakageChecker()
    validation_ids = [3] if validation_range is not None else []
    summary = _summary(
        strategy=SplitStrategy.TIME,
        train_ids=[0, 1, 2],
        validation_ids=validation_ids,
        test_ids=[4, 5],
        time_column="ts",
        train_time_range=train_range,
        validation_time_range=validation_range,
        test_time_range=test_range,
    )
    split = _split_from_ids([0, 1, 2], [4, 5], validation_ids, summary=summary)
    report = checker.check(
        split,
        _safe_role_mapping(),
        feature_columns=["temp"],
    )
    issues = [
        issue
        for issue in report.issues
        if issue.issue_type is LeakageIssueType.TIME_ORDER_VIOLATION
    ]
    assert len(issues) == 1
    assert issues[0].columns == ["ts"]
    assert issues[0].partitions == expected


def test_identical_time_boundary_is_allowed() -> None:
    checker = LeakageChecker()
    boundary = date(2020, 1, 10)
    summary = _summary(
        strategy=SplitStrategy.TIME,
        train_ids=[0, 1, 2],
        test_ids=[3, 4],
        time_column="ts",
        train_time_range=(date(2020, 1, 1), boundary),
        test_time_range=(boundary, date(2020, 1, 20)),
    )
    split = _split_from_ids([0, 1, 2], [3, 4], summary=summary)
    report = checker.check(
        split,
        _safe_role_mapping(),
        feature_columns=["temp"],
    )
    assert LeakageIssueType.TIME_ORDER_VIOLATION not in _issue_types(report)


def test_time_order_skipped_for_non_time_strategy() -> None:
    checker = LeakageChecker()
    summary = _summary(
        strategy=SplitStrategy.RANDOM,
        train_ids=[0, 1],
        test_ids=[2, 3],
        train_time_range=(date(2020, 1, 10), date(2020, 1, 20)),
        test_time_range=(date(2020, 1, 1), date(2020, 1, 5)),
    )
    split = _split_from_ids([0, 1], [2, 3], summary=summary)
    report = checker.check(
        split,
        _safe_role_mapping(),
        feature_columns=["temp"],
    )
    assert LeakageIssueType.TIME_ORDER_VIOLATION not in _issue_types(report)


# ---------------------------------------------------------------------------
# Target and identifier leakage
# ---------------------------------------------------------------------------


def test_target_included_as_feature() -> None:
    checker = LeakageChecker()
    split = _split_from_ids([0, 1], [2, 3])
    report = checker.check(
        split,
        _safe_role_mapping(),
        target_column="yield",
        feature_columns=["temp", "yield"],
    )
    issues = [
        issue
        for issue in report.issues
        if issue.issue_type is LeakageIssueType.TARGET_INCLUDED_AS_FEATURE
    ]
    assert len(issues) == 1
    assert issues[0].severity is LeakageSeverity.BLOCKER
    assert issues[0].columns == ["yield"]


def test_target_not_in_features_has_no_target_issue() -> None:
    checker = LeakageChecker()
    split = _split_from_ids([0, 1], [2, 3])
    report = checker.check(
        split,
        _safe_role_mapping(),
        target_column="yield",
        feature_columns=["temp", "pressure"],
    )
    assert LeakageIssueType.TARGET_INCLUDED_AS_FEATURE not in _issue_types(report)


def test_identifier_feature_detection_and_order() -> None:
    checker = LeakageChecker()
    split = _split_from_ids(
        [0, 1],
        [2, 3],
        extra_columns={
            "train": {"wafer_id": ["a", "b"], "lot_id": ["x", "y"]},
            "test": {"wafer_id": ["c", "d"], "lot_id": ["u", "v"]},
        },
    )
    # Align schemas for empty validation
    split = DatasetSplit(
        train=split.train,
        validation=split.train.clear(),
        test=split.test,
        summary=split.summary,
    )
    mapping = _role_mapping(
        _assignment("wafer_id", ColumnRole.IDENTIFIER, use_in_model=False),
        _assignment("lot_id", ColumnRole.IDENTIFIER, use_in_model=False),
        _assignment("temp", ColumnRole.STATE_SENSOR),
        _assignment(ORIGINAL_ROW_ID_COLUMN, ColumnRole.IDENTIFIER, use_in_model=False),
    )
    report = checker.check(
        split,
        mapping,
        feature_columns=["temp", "lot_id", "wafer_id", ORIGINAL_ROW_ID_COLUMN],
    )
    id_issues = [
        issue
        for issue in report.issues
        if issue.issue_type is LeakageIssueType.IDENTIFIER_INCLUDED_AS_FEATURE
    ]
    assert [issue.columns[0] for issue in id_issues] == [
        "lot_id",
        "wafer_id",
        ORIGINAL_ROW_ID_COLUMN,
    ]
    assert all(issue.severity is LeakageSeverity.BLOCKER for issue in id_issues)
    assert "identity" in id_issues[0].message.lower() or "memorization" in (
        id_issues[0].message.lower()
    )


def test_allow_identifier_features_skips_id_issues() -> None:
    checker = LeakageChecker(LeakageCheckConfig(allow_identifier_features=True))
    split = _split_from_ids(
        [0, 1],
        [2, 3],
        extra_columns={
            "train": {"wafer_id": ["a", "b"]},
            "test": {"wafer_id": ["c", "d"]},
        },
    )
    split = DatasetSplit(
        train=split.train,
        validation=split.train.clear(),
        test=split.test,
        summary=split.summary,
    )
    mapping = _role_mapping(
        _assignment("wafer_id", ColumnRole.IDENTIFIER, use_in_model=True),
        _assignment("temp", ColumnRole.STATE_SENSOR),
        _assignment(ORIGINAL_ROW_ID_COLUMN, ColumnRole.IDENTIFIER, use_in_model=False),
    )
    report = checker.check(
        split,
        mapping,
        feature_columns=["wafer_id", "temp", ORIGINAL_ROW_ID_COLUMN],
    )
    assert LeakageIssueType.IDENTIFIER_INCLUDED_AS_FEATURE not in _issue_types(report)


def test_non_identifier_feature_has_no_id_issue() -> None:
    checker = LeakageChecker()
    split = _split_from_ids([0, 1], [2, 3])
    report = checker.check(
        split,
        _safe_role_mapping(),
        feature_columns=["temp", "pressure"],
    )
    assert LeakageIssueType.IDENTIFIER_INCLUDED_AS_FEATURE not in _issue_types(report)


# ---------------------------------------------------------------------------
# Name-based leakage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "feature_name",
    ["actual_yield", "yield_residual", "future_yield", "yield_error"],
)
def test_target_derived_feature_patterns(feature_name: str) -> None:
    checker = LeakageChecker()
    split = _split_from_ids(
        [0, 1],
        [2, 3],
        extra_columns={
            "train": {feature_name: [1.0, 2.0]},
            "test": {feature_name: [3.0, 4.0]},
        },
    )
    split = DatasetSplit(
        train=split.train,
        validation=split.train.clear(),
        test=split.test,
        summary=split.summary,
    )
    report = checker.check(
        split,
        _safe_role_mapping(),
        target_column="yield",
        feature_columns=["temp", feature_name],
    )
    assert LeakageIssueType.TARGET_DERIVED_FEATURE in _issue_types(report)


def test_target_token_only_is_not_target_derived() -> None:
    checker = LeakageChecker()
    split = _split_from_ids(
        [0, 1],
        [2, 3],
        extra_columns={
            "train": {"yield_temperature": [1.0, 2.0]},
            "test": {"yield_temperature": [3.0, 4.0]},
        },
    )
    split = DatasetSplit(
        train=split.train,
        validation=split.train.clear(),
        test=split.test,
        summary=split.summary,
    )
    report = checker.check(
        split,
        _safe_role_mapping(),
        target_column="yield",
        feature_columns=["temp", "yield_temperature"],
    )
    assert LeakageIssueType.TARGET_DERIVED_FEATURE not in _issue_types(report)


def test_target_derived_requires_target_column() -> None:
    checker = LeakageChecker()
    split = _split_from_ids(
        [0, 1],
        [2, 3],
        extra_columns={
            "train": {"actual_yield": [1.0, 2.0]},
            "test": {"actual_yield": [3.0, 4.0]},
        },
    )
    split = DatasetSplit(
        train=split.train,
        validation=split.train.clear(),
        test=split.test,
        summary=split.summary,
    )
    report = checker.check(
        split,
        _safe_role_mapping(),
        feature_columns=["temp", "actual_yield"],
    )
    assert LeakageIssueType.TARGET_DERIVED_FEATURE not in _issue_types(report)


@pytest.mark.parametrize(
    "feature_name",
    [
        "future_temperature",
        "next_cycle_capacity",
        "lead_failure_signal",
        "post_process_result",
        "after_event_pressure",
    ],
)
def test_future_feature_token_detection(feature_name: str) -> None:
    checker = LeakageChecker()
    split = _split_from_ids(
        [0, 1],
        [2, 3],
        extra_columns={
            "train": {feature_name: [1.0, 2.0]},
            "test": {feature_name: [3.0, 4.0]},
        },
    )
    split = DatasetSplit(
        train=split.train,
        validation=split.train.clear(),
        test=split.test,
        summary=split.summary,
    )
    report = checker.check(
        split,
        _safe_role_mapping(),
        feature_columns=["temp", feature_name],
    )
    assert LeakageIssueType.FUTURE_DERIVED_FEATURE in _issue_types(report)


def test_manufacture_is_not_future_false_positive() -> None:
    checker = LeakageChecker()
    split = _split_from_ids(
        [0, 1],
        [2, 3],
        extra_columns={
            "train": {"manufacture_date_code": [1.0, 2.0]},
            "test": {"manufacture_date_code": [3.0, 4.0]},
        },
    )
    split = DatasetSplit(
        train=split.train,
        validation=split.train.clear(),
        test=split.test,
        summary=split.summary,
    )
    report = checker.check(
        split,
        _safe_role_mapping(),
        feature_columns=["temp", "manufacture_date_code"],
    )
    assert LeakageIssueType.FUTURE_DERIVED_FEATURE not in _issue_types(report)


@pytest.mark.parametrize(
    "feature_name",
    [
        "failure_timestamp",
        "completion_time",
        "outcome_date",
        "result_datetime",
        "end_time",
    ],
)
def test_outcome_timestamp_detection(feature_name: str) -> None:
    checker = LeakageChecker()
    split = _split_from_ids(
        [0, 1],
        [2, 3],
        extra_columns={
            "train": {feature_name: [1.0, 2.0]},
            "test": {feature_name: [3.0, 4.0]},
        },
    )
    split = DatasetSplit(
        train=split.train,
        validation=split.train.clear(),
        test=split.test,
        summary=split.summary,
    )
    report = checker.check(
        split,
        _safe_role_mapping(),
        feature_columns=["temp", feature_name],
    )
    assert LeakageIssueType.OUTCOME_TIMESTAMP_FEATURE in _issue_types(report)


@pytest.mark.parametrize(
    "feature_name",
    ["timestamp", "event_time", "measurement_time", "start_time"],
)
def test_generic_timestamps_are_not_outcome_timestamp(feature_name: str) -> None:
    checker = LeakageChecker()
    split = _split_from_ids(
        [0, 1],
        [2, 3],
        extra_columns={
            "train": {feature_name: [1.0, 2.0]},
            "test": {feature_name: [3.0, 4.0]},
        },
    )
    split = DatasetSplit(
        train=split.train,
        validation=split.train.clear(),
        test=split.test,
        summary=split.summary,
    )
    report = checker.check(
        split,
        _safe_role_mapping(),
        feature_columns=["temp", feature_name],
    )
    assert LeakageIssueType.OUTCOME_TIMESTAMP_FEATURE not in _issue_types(report)


def test_name_normalization_consistency_and_casefold() -> None:
    checker = LeakageChecker()
    for feature_name in ("future_yield", "Future-Yield", "future yield"):
        split = _split_from_ids(
            [0, 1],
            [2, 3],
            extra_columns={
                "train": {feature_name: [1.0, 2.0]},
                "test": {feature_name: [3.0, 4.0]},
            },
        )
        split = DatasetSplit(
            train=split.train,
            validation=split.train.clear(),
            test=split.test,
            summary=split.summary,
        )
        report = checker.check(
            split,
            _safe_role_mapping(),
            target_column="yield",
            feature_columns=["temp", feature_name],
        )
        types = _issue_types(report)
        assert LeakageIssueType.TARGET_DERIVED_FEATURE in types
        assert LeakageIssueType.FUTURE_DERIVED_FEATURE in types


def test_feature_issue_order_and_no_duplicates() -> None:
    checker = LeakageChecker()
    split = _split_from_ids(
        [0, 1],
        [2, 3],
        extra_columns={
            "train": {
                "next_temp": [1.0, 2.0],
                "future_pressure": [1.0, 2.0],
                "after_event": [1.0, 2.0],
            },
            "test": {
                "next_temp": [3.0, 4.0],
                "future_pressure": [3.0, 4.0],
                "after_event": [3.0, 4.0],
            },
        },
    )
    split = DatasetSplit(
        train=split.train,
        validation=split.train.clear(),
        test=split.test,
        summary=split.summary,
    )
    features = ["temp", "after_event", "next_temp", "future_pressure"]
    report = checker.check(
        split,
        _safe_role_mapping(),
        feature_columns=features,
    )
    future_issues = [
        issue
        for issue in report.issues
        if issue.issue_type is LeakageIssueType.FUTURE_DERIVED_FEATURE
    ]
    assert [issue.columns[0] for issue in future_issues] == [
        "after_event",
        "next_temp",
        "future_pressure",
    ]
    keys = {
        (issue.issue_type, tuple(issue.columns), tuple(issue.partitions))
        for issue in report.issues
    }
    assert len(keys) == len(report.issues)


def test_different_issue_types_can_coexist_on_same_feature() -> None:
    checker = LeakageChecker()
    split = _split_from_ids(
        [0, 1],
        [2, 3],
        extra_columns={
            "train": {"future_yield": [1.0, 2.0]},
            "test": {"future_yield": [3.0, 4.0]},
        },
    )
    split = DatasetSplit(
        train=split.train,
        validation=split.train.clear(),
        test=split.test,
        summary=split.summary,
    )
    report = checker.check(
        split,
        _safe_role_mapping(),
        target_column="yield",
        feature_columns=["temp", "future_yield"],
    )
    types = _issue_types(report)
    assert LeakageIssueType.TARGET_DERIVED_FEATURE in types
    assert LeakageIssueType.FUTURE_DERIVED_FEATURE in types


# ---------------------------------------------------------------------------
# Preprocessing fit scope
# ---------------------------------------------------------------------------


def test_preprocessing_true_flags_are_safe() -> None:
    checker = LeakageChecker()
    split = _split_from_ids([0, 1], [2, 3])
    events = [
        _event(
            "impute_missing_values",
            affected_columns=["temp"],
            parameters={"fitted_on_training_data": True},
        ),
        _event(
            "scale_numeric_features",
            affected_columns=["pressure"],
            parameters={"fitted_on_training_data": True},
        ),
    ]
    report = checker.check(
        split,
        _safe_role_mapping(),
        feature_columns=["temp", "pressure"],
        preprocessing_events=events,
    )
    assert report.checked_preprocessing_event_count == 2
    assert LeakageIssueType.PREPROCESSING_FIT_SCOPE_UNKNOWN not in _issue_types(report)
    assert (
        LeakageIssueType.PREPROCESSING_NOT_FIT_ON_TRAINING_DATA
        not in _issue_types(report)
    )


def test_preprocessing_missing_key_warning_only() -> None:
    checker = LeakageChecker()
    split = _split_from_ids([0, 1], [2, 3])
    events = [
        _event("impute_missing_values", affected_columns=["temp"], parameters={}),
    ]
    report = checker.check(
        split,
        _safe_role_mapping(),
        feature_columns=["temp"],
        preprocessing_events=events,
    )
    types = _issue_types(report)
    assert LeakageIssueType.PREPROCESSING_FIT_SCOPE_UNKNOWN in types
    assert LeakageIssueType.PREPROCESSING_NOT_FIT_ON_TRAINING_DATA not in types
    warning = next(
        issue
        for issue in report.issues
        if issue.issue_type is LeakageIssueType.PREPROCESSING_FIT_SCOPE_UNKNOWN
    )
    assert warning.severity is LeakageSeverity.WARNING
    assert warning.columns == ["temp"]
    assert warning.partitions == ["train", "validation", "test"]


@pytest.mark.parametrize("bad_value", [False, 0, "true", None])
def test_preprocessing_non_true_fit_flag_is_blocker(bad_value: object) -> None:
    checker = LeakageChecker()
    split = _split_from_ids([0, 1], [2, 3])
    events = [
        _event(
            "scale_numeric_features",
            affected_columns=["pressure"],
            parameters={"fitted_on_training_data": bad_value},
        )
    ]
    report = checker.check(
        split,
        _safe_role_mapping(),
        feature_columns=["temp", "pressure"],
        preprocessing_events=events,
    )
    assert LeakageIssueType.PREPROCESSING_NOT_FIT_ON_TRAINING_DATA in _issue_types(
        report
    )
    assert LeakageIssueType.PREPROCESSING_FIT_SCOPE_UNKNOWN not in _issue_types(report)


def test_sort_and_noop_events_are_ignored() -> None:
    checker = LeakageChecker()
    split = _split_from_ids([0, 1], [2, 3])
    events = [
        _event("sort_dataset", parameters={}),
        _event("preprocess_noop", parameters={}),
    ]
    report = checker.check(
        split,
        _safe_role_mapping(),
        feature_columns=["temp"],
        preprocessing_events=events,
    )
    assert report.checked_preprocessing_event_count == 2
    assert LeakageIssueType.PREPROCESSING_FIT_SCOPE_UNKNOWN not in _issue_types(report)
    assert (
        LeakageIssueType.PREPROCESSING_NOT_FIT_ON_TRAINING_DATA
        not in _issue_types(report)
    )


def test_preprocessing_fit_scope_config_false_skips() -> None:
    checker = LeakageChecker(
        LeakageCheckConfig(require_preprocessing_fit_scope=False)
    )
    split = _split_from_ids([0, 1], [2, 3])
    events = [
        _event("impute_missing_values", affected_columns=["temp"], parameters={}),
        _event(
            "scale_numeric_features",
            affected_columns=["pressure"],
            parameters={"fitted_on_training_data": False},
        ),
    ]
    report = checker.check(
        split,
        _safe_role_mapping(),
        feature_columns=["temp", "pressure"],
        preprocessing_events=events,
    )
    assert LeakageIssueType.PREPROCESSING_FIT_SCOPE_UNKNOWN not in _issue_types(report)
    assert (
        LeakageIssueType.PREPROCESSING_NOT_FIT_ON_TRAINING_DATA
        not in _issue_types(report)
    )


def test_preprocessing_issue_order_follows_events() -> None:
    checker = LeakageChecker()
    split = _split_from_ids([0, 1], [2, 3])
    events = [
        _event(
            "scale_numeric_features",
            affected_columns=["pressure"],
            parameters={},
        ),
        _event(
            "impute_missing_values",
            affected_columns=["temp"],
            parameters={},
        ),
        _event(
            "scale_numeric_features",
            affected_columns=["pressure"],
            parameters={"fitted_on_training_data": False},
        ),
        _event(
            "impute_missing_values",
            affected_columns=["temp"],
            parameters={"fitted_on_training_data": 0},
        ),
    ]
    report = checker.check(
        split,
        _safe_role_mapping(),
        feature_columns=["temp", "pressure"],
        preprocessing_events=events,
    )
    unknown = [
        issue
        for issue in report.issues
        if issue.issue_type is LeakageIssueType.PREPROCESSING_FIT_SCOPE_UNKNOWN
    ]
    not_fit = [
        issue
        for issue in report.issues
        if issue.issue_type
        is LeakageIssueType.PREPROCESSING_NOT_FIT_ON_TRAINING_DATA
    ]
    assert [issue.columns for issue in unknown] == [["pressure"], ["temp"]]
    assert [issue.columns for issue in not_fit] == [["pressure"], ["temp"]]
    assert report.checked_preprocessing_event_count == 4


# ---------------------------------------------------------------------------
# Report and assert_safe
# ---------------------------------------------------------------------------


def test_report_safety_counts_and_feature_order() -> None:
    checker = LeakageChecker()
    split = _split_from_ids([0, 1], [2, 3])
    safe = checker.check(
        split,
        _safe_role_mapping(),
        feature_columns=["pressure", "temp"],
    )
    assert safe.is_safe is True
    assert safe.blocker_count == 0
    assert safe.warning_count == 0
    assert safe.checked_feature_columns == ["pressure", "temp"]

    warning_only = checker.check(
        split,
        _safe_role_mapping(),
        feature_columns=["temp"],
        preprocessing_events=[
            _event("impute_missing_values", affected_columns=["temp"], parameters={})
        ],
    )
    assert warning_only.is_safe is True
    assert warning_only.blocker_count == 0
    assert warning_only.warning_count == 1

    blocked = checker.check(
        split,
        _safe_role_mapping(),
        target_column="yield",
        feature_columns=["temp", "yield"],
    )
    assert blocked.is_safe is False
    assert blocked.blocker_count >= 1


def test_assert_safe_behavior() -> None:
    checker = LeakageChecker()
    with pytest.raises(TypeError):
        checker.assert_safe("report")  # type: ignore[arg-type]

    safe = LeakageReport(
        is_safe=True,
        issues=[],
        blocker_count=0,
        warning_count=0,
        checked_feature_columns=[],
        checked_preprocessing_event_count=0,
    )
    checker.assert_safe(safe)

    warning = LeakageIssue(
        issue_type=LeakageIssueType.PREPROCESSING_FIT_SCOPE_UNKNOWN,
        severity=LeakageSeverity.WARNING,
        message="unknown",
        suggested_action="record",
    )
    warning_report = LeakageReport(
        is_safe=True,
        issues=[warning],
        blocker_count=0,
        warning_count=1,
        checked_feature_columns=[],
        checked_preprocessing_event_count=1,
    )
    checker.assert_safe(warning_report)

    blocker = LeakageIssue(
        issue_type=LeakageIssueType.TARGET_INCLUDED_AS_FEATURE,
        severity=LeakageSeverity.BLOCKER,
        columns=["yield"],
        message="target in features",
        suggested_action="remove",
    )
    blocked = LeakageReport(
        is_safe=False,
        issues=[blocker],
        blocker_count=1,
        warning_count=0,
        checked_feature_columns=["yield"],
        checked_preprocessing_event_count=0,
    )
    with pytest.raises(DataLeakageError) as exc_info:
        checker.assert_safe(blocked)
    message = str(exc_info.value)
    assert "1" in message
    assert "TARGET_INCLUDED_AS_FEATURE" in message


# ---------------------------------------------------------------------------
# Determinism, immutability, integration
# ---------------------------------------------------------------------------


def test_issue_order_matches_specification() -> None:
    checker = LeakageChecker()
    summary = _summary(
        strategy=SplitStrategy.GROUP,
        train_ids=[0, 1],
        validation_ids=[1],
        test_ids=[2, 1],
        group_column="batch",
        train_groups=["A", "B"],
        validation_groups=["B"],
        test_groups=["B", "C"],
        bypass_validation=True,
    )
    # Frames intentionally overlap IDs; summary IDs also mismatch frames.
    train = _frame(
        [0, 1],
        extra={
            "batch": ["A", "B"],
            "sample_id": ["s0", "s1"],
            "actual_yield": [1.0, 2.0],
            "future_temp": [1.0, 2.0],
            "failure_timestamp": [1.0, 2.0],
            "yield": [0.1, 0.2],
        },
    )
    validation = _frame(
        [1],
        extra={
            "batch": ["B"],
            "sample_id": ["s1"],
            "actual_yield": [3.0],
            "future_temp": [3.0],
            "failure_timestamp": [3.0],
            "yield": [0.3],
        },
    )
    test = _frame(
        [2, 1],
        extra={
            "batch": ["C", "B"],
            "sample_id": ["s2", "s1"],
            "actual_yield": [4.0, 5.0],
            "future_temp": [4.0, 5.0],
            "failure_timestamp": [4.0, 5.0],
            "yield": [0.4, 0.5],
        },
    )
    # Force summary mismatch vs frames while keeping GROUP overlap metadata.
    mismatched_summary = summary.model_copy(
        update={
            "train_original_row_ids": [0, 9],
            "validation_original_row_ids": [8],
            "test_original_row_ids": [7, 6],
        },
        deep=True,
    )
    # model_copy still validates; use model_construct for intentionally bad IDs.
    mismatched_summary = SplitSummary.model_construct(
        **{
            **mismatched_summary.model_dump(),
            "train_original_row_ids": [0, 9],
            "validation_original_row_ids": [8],
            "test_original_row_ids": [7, 6],
            "train_groups": ["A", "B"],
            "validation_groups": ["B"],
            "test_groups": ["B", "C"],
            "strategy": SplitStrategy.GROUP,
            "group_column": "batch",
        }
    )
    split = DatasetSplit(
        train=train,
        validation=validation,
        test=test,
        summary=mismatched_summary,
    )
    mapping = _role_mapping(
        _assignment("sample_id", ColumnRole.IDENTIFIER, use_in_model=False),
        _assignment("temp", ColumnRole.STATE_SENSOR),
        _assignment("yield", ColumnRole.TARGET_QUALITY, use_in_model=False),
        _assignment(ORIGINAL_ROW_ID_COLUMN, ColumnRole.IDENTIFIER, use_in_model=False),
    )
    events = [
        _event("impute_missing_values", affected_columns=["temp"], parameters={}),
        _event(
            "scale_numeric_features",
            affected_columns=["pressure"],
            parameters={"fitted_on_training_data": False},
        ),
    ]
    report = checker.check(
        split,
        mapping,
        target_column="yield",
        feature_columns=[
            "temp",
            "yield",
            "sample_id",
            "actual_yield",
            "future_temp",
            "failure_timestamp",
        ],
        preprocessing_events=events,
    )
    order = _issue_types(report)
    expected_order = [
        LeakageIssueType.ORIGINAL_ROW_ID_OVERLAP,
        LeakageIssueType.ORIGINAL_ROW_ID_OVERLAP,
        LeakageIssueType.ORIGINAL_ROW_ID_OVERLAP,
        LeakageIssueType.SPLIT_SUMMARY_MISMATCH,
        LeakageIssueType.GROUP_PARTITION_OVERLAP,
        LeakageIssueType.TARGET_INCLUDED_AS_FEATURE,
        LeakageIssueType.IDENTIFIER_INCLUDED_AS_FEATURE,
        LeakageIssueType.TARGET_DERIVED_FEATURE,
        LeakageIssueType.FUTURE_DERIVED_FEATURE,
        LeakageIssueType.OUTCOME_TIMESTAMP_FEATURE,
        LeakageIssueType.PREPROCESSING_FIT_SCOPE_UNKNOWN,
        LeakageIssueType.PREPROCESSING_NOT_FIT_ON_TRAINING_DATA,
    ]
    # Filter to unique type progression while preserving relative order.
    seen: list[LeakageIssueType] = []
    for issue_type in order:
        if not seen or seen[-1] != issue_type:
            seen.append(issue_type)
    assert seen == [
        LeakageIssueType.ORIGINAL_ROW_ID_OVERLAP,
        LeakageIssueType.SPLIT_SUMMARY_MISMATCH,
        LeakageIssueType.GROUP_PARTITION_OVERLAP,
        LeakageIssueType.TARGET_INCLUDED_AS_FEATURE,
        LeakageIssueType.IDENTIFIER_INCLUDED_AS_FEATURE,
        LeakageIssueType.TARGET_DERIVED_FEATURE,
        LeakageIssueType.FUTURE_DERIVED_FEATURE,
        LeakageIssueType.OUTCOME_TIMESTAMP_FEATURE,
        LeakageIssueType.PREPROCESSING_FIT_SCOPE_UNKNOWN,
        LeakageIssueType.PREPROCESSING_NOT_FIT_ON_TRAINING_DATA,
    ]
    assert order[:3] == expected_order[:3]


def test_check_does_not_mutate_inputs() -> None:
    checker = LeakageChecker()
    split = _split_from_ids([0, 1, 2], [3, 4])
    mapping = _safe_role_mapping()
    features = ["temp", "pressure"]
    events = [
        _event(
            "impute_missing_values",
            affected_columns=["temp"],
            parameters={"fitted_on_training_data": True},
        )
    ]
    train_before = split.train.clone()
    summary_before = split.summary.model_copy(deep=True)
    mapping_before = mapping.model_copy(deep=True)
    features_before = list(features)
    events_before = [event.model_copy(deep=True) for event in events]

    checker.check(
        split,
        mapping,
        feature_columns=features,
        preprocessing_events=events,
    )

    assert split.train.equals(train_before)
    assert split.summary == summary_before
    assert mapping == mapping_before
    assert features == features_before
    assert events == events_before


def test_repeated_checks_do_not_accumulate_and_are_deterministic() -> None:
    checker = LeakageChecker()
    split = _split_from_ids([0, 1], [2, 3])
    mapping = _safe_role_mapping()
    first = checker.check(
        split,
        mapping,
        target_column="yield",
        feature_columns=["temp", "yield"],
    )
    second = checker.check(
        split,
        mapping,
        target_column="yield",
        feature_columns=["temp", "yield"],
    )
    assert first.model_dump() == second.model_dump()
    assert first.blocker_count == second.blocker_count


def test_checker_instances_do_not_share_state() -> None:
    first = LeakageChecker(LeakageCheckConfig(allow_identifier_features=True))
    second = LeakageChecker(LeakageCheckConfig(allow_identifier_features=False))
    split = _split_from_ids(
        [0, 1],
        [2, 3],
        extra_columns={
            "train": {"wafer_id": ["a", "b"]},
            "test": {"wafer_id": ["c", "d"]},
        },
    )
    split = DatasetSplit(
        train=split.train,
        validation=split.train.clear(),
        test=split.test,
        summary=split.summary,
    )
    mapping = _role_mapping(
        _assignment("wafer_id", ColumnRole.IDENTIFIER, use_in_model=True),
        _assignment("temp", ColumnRole.STATE_SENSOR),
        _assignment(ORIGINAL_ROW_ID_COLUMN, ColumnRole.IDENTIFIER, use_in_model=False),
    )
    first_report = first.check(split, mapping, feature_columns=["wafer_id", "temp"])
    second_report = second.check(split, mapping, feature_columns=["wafer_id", "temp"])
    assert (
        LeakageIssueType.IDENTIFIER_INCLUDED_AS_FEATURE
        not in _issue_types(first_report)
    )
    assert (
        LeakageIssueType.IDENTIFIER_INCLUDED_AS_FEATURE in _issue_types(second_report)
    )


def test_datasetsplitter_result_passes_partition_leakage() -> None:
    frame = pl.DataFrame(
        {
            "batch": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"],
            "temp": list(range(10)),
            "pressure": list(range(10, 20)),
            "yield": [0.1 * i for i in range(10)],
            ORIGINAL_ROW_ID_COLUMN: list(range(10)),
        }
    )
    split = DatasetSplitter().split(
        frame,
        SplitConfig(
            strategy=SplitStrategy.GROUP,
            group_column="batch",
            test_size=0.2,
            validation_size=0.2,
        ),
    )
    mapping = _role_mapping(
        _assignment("temp", ColumnRole.STATE_SENSOR),
        _assignment("pressure", ColumnRole.CONTROLLABLE_PROCESS),
        _assignment("yield", ColumnRole.TARGET_QUALITY, use_in_model=False),
        _assignment("batch", ColumnRole.CONTEXT, use_in_model=False),
        _assignment(ORIGINAL_ROW_ID_COLUMN, ColumnRole.IDENTIFIER, use_in_model=False),
    )
    report = LeakageChecker().check(
        split,
        mapping,
        target_column="yield",
        feature_columns=["temp", "pressure"],
    )
    assert LeakageIssueType.ORIGINAL_ROW_ID_OVERLAP not in _issue_types(report)
    assert LeakageIssueType.SPLIT_SUMMARY_MISMATCH not in _issue_types(report)
    assert LeakageIssueType.GROUP_PARTITION_OVERLAP not in _issue_types(report)
    assert report.is_safe is True


def test_preprocessor_transform_events_pass_fit_scope() -> None:
    train = pl.DataFrame(
        {
            "temp": [1.0, None, 3.0, 5.0],
            "status": ["ok", None, "ok", "warn"],
            ORIGINAL_ROW_ID_COLUMN: [0, 1, 2, 3],
        }
    )
    test = pl.DataFrame(
        {
            "temp": [2.0, None],
            "status": ["ok", None],
            ORIGINAL_ROW_ID_COLUMN: [4, 5],
        }
    )
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=["temp"],
            categorical_columns=["status"],
            numeric_imputation="median",
            categorical_imputation="constant",
            scaling="standard",
        )
    )
    preprocessor.fit(train)
    result = preprocessor.transform(test)
    split = DatasetSplit(
        train=train,
        validation=train.clear(),
        test=test,
        summary=_summary(train_ids=[0, 1, 2, 3], test_ids=[4, 5]),
    )
    mapping = _role_mapping(
        _assignment("temp", ColumnRole.STATE_SENSOR),
        _assignment("status", ColumnRole.CONTEXT),
        _assignment(ORIGINAL_ROW_ID_COLUMN, ColumnRole.IDENTIFIER, use_in_model=False),
    )
    report = LeakageChecker().check(
        split,
        mapping,
        feature_columns=["temp", "status"],
        preprocessing_events=result.events,
    )
    assert LeakageIssueType.PREPROCESSING_FIT_SCOPE_UNKNOWN not in _issue_types(report)
    assert (
        LeakageIssueType.PREPROCESSING_NOT_FIT_ON_TRAINING_DATA
        not in _issue_types(report)
    )


def test_column_role_mapper_identifier_connects_to_id_leakage() -> None:
    profile = DatasetProfile(
        row_count=4,
        column_count=3,
        columns=[
            ColumnProfile(
                name="wafer_id",
                dtype="String",
                null_count=0,
                null_ratio=0.0,
                unique_count=4,
                cardinality_ratio=1.0,
                is_constant=False,
            ),
            ColumnProfile(
                name="temp",
                dtype="Float64",
                null_count=0,
                null_ratio=0.0,
                unique_count=4,
                cardinality_ratio=1.0,
                is_constant=False,
            ),
            ColumnProfile(
                name=ORIGINAL_ROW_ID_COLUMN,
                dtype="Int64",
                null_count=0,
                null_ratio=0.0,
                unique_count=4,
                cardinality_ratio=1.0,
                is_constant=False,
            ),
        ],
        preview_records=[],
    )
    from process_intelligence.core.schemas import SchemaHints

    mapping = ColumnRoleMapper().map_roles(
        profile,
        SchemaHints(
            default_roles={
                "wafer_id": ColumnRole.IDENTIFIER,
                "temp": ColumnRole.STATE_SENSOR,
            }
        ),
    )
    assert isinstance(mapping, DirectMapping)
    split = _split_from_ids(
        [0, 1],
        [2, 3],
        extra_columns={
            "train": {"wafer_id": ["a", "b"]},
            "test": {"wafer_id": ["c", "d"]},
        },
    )
    split = DatasetSplit(
        train=split.train,
        validation=split.train.clear(),
        test=split.test,
        summary=split.summary,
    )
    report = LeakageChecker().check(
        split,
        mapping,
        feature_columns=["wafer_id", "temp"],
    )
    assert LeakageIssueType.IDENTIFIER_INCLUDED_AS_FEATURE in _issue_types(report)


def test_package_exports_include_leakage_objects() -> None:
    from process_intelligence import evaluation

    assert evaluation.LeakageChecker is LeakageChecker
    assert evaluation.LeakageIssueType is LeakageIssueType
    assert "LeakageChecker" in evaluation.__all__
