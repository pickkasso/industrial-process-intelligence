"""Unit tests for unsupervised anomaly pipeline (Step 7B)."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, fields, is_dataclass
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.exceptions import (
    DataLeakageError,
    DataValidationError,
    InsufficientDataError,
    ProcessIntelligenceError,
)
from process_intelligence.data.loader import ORIGINAL_ROW_ID_COLUMN
from process_intelligence.evaluation import (
    DatasetSplit,
    DatasetSplitter,
    LeakageIssue,
    LeakageIssueType,
    LeakageReport,
    LeakageSeverity,
    SplitConfig,
    SplitStrategy,
    SplitSummary,
)
from process_intelligence.models import (
    AnomalyDetectionResult,
    AnomalyPartitionSummary,
    AnomalyPipelinePolicy,
    AnomalyPipelineReport,
    IsolationForestAnomalyModel,
    IsolationForestConfig,
    UnsupervisedAnomalyPipeline,
)
from process_intelligence.models.anomaly_pipeline import (
    _DATA_PARTITION_COLUMN,
    _IS_ANOMALY_COLUMN,
    _RAW_PREDICTION_COLUMN,
    _SCORE_COLUMN,
)

_FEATURES = ["f1", "f2"]
_RESULT_COLUMNS = [
    _SCORE_COLUMN,
    _IS_ANOMALY_COLUMN,
    _RAW_PREDICTION_COLUMN,
    _DATA_PARTITION_COLUMN,
]


def _cluster_frame(*, rows: int = 40, include_outlier: bool = True) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    normal = rng.normal(loc=0.0, scale=1.0, size=(rows, 2))
    if include_outlier and rows > 0:
        normal[-1] = np.array([25.0, -25.0])
    return pl.DataFrame(
        {
            "f1": normal[:, 0].tolist() if rows else [],
            "f2": normal[:, 1].tolist() if rows else [],
            ORIGINAL_ROW_ID_COLUMN: list(range(rows)),
        }
    )


def _make_split(
    *,
    rows: int = 40,
    validation_size: float = 0.2,
    test_size: float = 0.2,
    include_outlier: bool = True,
) -> DatasetSplit:
    frame = _cluster_frame(rows=rows, include_outlier=include_outlier)
    config = SplitConfig(
        strategy=SplitStrategy.RANDOM,
        test_size=test_size,
        validation_size=validation_size,
        random_state=42,
        allow_random_split=True,
        minimum_train_rows=1,
        minimum_test_rows=1,
        minimum_validation_rows=1 if validation_size > 0.0 else 1,
    )
    return DatasetSplitter().split(frame, config)


def _split_with_empty_validation(*, rows: int = 40) -> DatasetSplit:
    """Build a split with an empty validation partition."""
    frame = _cluster_frame(rows=rows)
    config = SplitConfig(
        strategy=SplitStrategy.RANDOM,
        test_size=0.2,
        validation_size=0.0,
        random_state=42,
        allow_random_split=True,
        minimum_train_rows=1,
        minimum_test_rows=1,
        minimum_validation_rows=1,
    )
    return DatasetSplitter().split(frame, config)


def _split_with_empty_test(*, rows: int = 40) -> DatasetSplit:
    """Build a split with an empty test partition via manual construction."""
    split = _make_split(rows=rows, validation_size=0.2, test_size=0.2)
    empty = split.test.clear()
    return DatasetSplit(
        train=split.train,
        validation=split.validation,
        test=empty,
        summary=split.summary.model_copy(
            update={
                "test_row_count": 0,
                "test_original_row_ids": [],
                "test_fraction": 0.0,
            }
        ),
    )


def _train_only_split(*, rows: int = 30) -> DatasetSplit:
    """Build a split with empty validation and test partitions."""
    train = _cluster_frame(rows=rows)
    empty = train.clear()
    return _manual_split(
        train,
        empty,
        empty,
        mutate_summary={
            "requested_test_size": 0.2,
            "requested_validation_size": 0.0,
            "train_row_count": train.height,
            "validation_row_count": 0,
            "test_row_count": 0,
            "train_fraction": 1.0,
            "validation_fraction": 0.0,
            "test_fraction": 0.0,
            "validation_original_row_ids": [],
            "test_original_row_ids": [],
        },
    )


def _manual_split(
    train: pl.DataFrame,
    validation: pl.DataFrame,
    test: pl.DataFrame,
    *,
    mutate_summary: dict[str, Any] | None = None,
) -> DatasetSplit:
    summary_kwargs: dict[str, Any] = {
        "strategy": SplitStrategy.RANDOM,
        "random_state": 42,
        "requested_test_size": 0.2,
        "requested_validation_size": 0.2,
        "train_row_count": train.height,
        "validation_row_count": validation.height,
        "test_row_count": test.height,
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "test_fraction": 0.2,
        "train_original_row_ids": train[ORIGINAL_ROW_ID_COLUMN].to_list(),
        "validation_original_row_ids": validation[ORIGINAL_ROW_ID_COLUMN].to_list(),
        "test_original_row_ids": test[ORIGINAL_ROW_ID_COLUMN].to_list(),
    }
    if mutate_summary:
        summary_kwargs.update(mutate_summary)
    return DatasetSplit(
        train=train,
        validation=validation,
        test=test,
        summary=SplitSummary(**summary_kwargs),
    )


def _safe_report(features: list[str] | None = None) -> LeakageReport:
    return LeakageReport(
        is_safe=True,
        issues=[],
        blocker_count=0,
        warning_count=0,
        checked_feature_columns=list(features or _FEATURES),
        checked_preprocessing_event_count=0,
    )


def _warning_report(features: list[str] | None = None) -> LeakageReport:
    issue = LeakageIssue(
        issue_type=LeakageIssueType.IDENTIFIER_INCLUDED_AS_FEATURE,
        severity=LeakageSeverity.WARNING,
        columns=["lot_id"],
        partitions=["train"],
        message="Identifier-like feature detected",
        suggested_action="Review whether the identifier should remain a feature",
    )
    return LeakageReport(
        is_safe=True,
        issues=[issue],
        blocker_count=0,
        warning_count=1,
        checked_feature_columns=list(features or _FEATURES),
        checked_preprocessing_event_count=0,
    )


def _blocker_report(features: list[str] | None = None) -> LeakageReport:
    issue = LeakageIssue(
        issue_type=LeakageIssueType.ORIGINAL_ROW_ID_OVERLAP,
        severity=LeakageSeverity.BLOCKER,
        columns=[ORIGINAL_ROW_ID_COLUMN],
        partitions=["train", "test"],
        message="Overlapping original row IDs",
        suggested_action="Rebuild the split",
    )
    return LeakageReport(
        is_safe=False,
        issues=[issue],
        blocker_count=1,
        warning_count=0,
        checked_feature_columns=list(features or _FEATURES),
        checked_preprocessing_event_count=0,
    )


def _partition_summary(
    *,
    partition: str = "train",
    input_row_count: int = 10,
    scored: bool = True,
    scored_row_count: int | None = None,
    anomaly_count: int = 1,
    anomaly_fraction: float | None = None,
    score_min: float | None = 0.1,
    score_max: float | None = 0.9,
    score_mean: float | None = 0.5,
    scoring_seconds: float = 0.01,
    warnings: list[str] | None = None,
) -> AnomalyPartitionSummary:
    if scored_row_count is None:
        scored_row_count = input_row_count if scored else 0
    if anomaly_fraction is None:
        anomaly_fraction = (
            0.0 if scored_row_count == 0 else anomaly_count / scored_row_count
        )
    if not scored or scored_row_count == 0:
        score_min = None
        score_max = None
        score_mean = None
        anomaly_count = 0
        anomaly_fraction = 0.0
    return AnomalyPartitionSummary(
        partition=partition,  # type: ignore[arg-type]
        input_row_count=input_row_count,
        scored=scored,
        scored_row_count=scored_row_count,
        anomaly_count=anomaly_count,
        anomaly_fraction=anomaly_fraction,
        score_min=score_min,
        score_max=score_max,
        score_mean=score_mean,
        scoring_seconds=scoring_seconds,
        warnings=[] if warnings is None else warnings,
    )


def _valid_report(**overrides: Any) -> AnomalyPipelineReport:
    summaries = [
        _partition_summary(partition="train", input_row_count=10, anomaly_count=1),
        _partition_summary(partition="validation", input_row_count=5, anomaly_count=1),
        _partition_summary(partition="test", input_row_count=5, anomaly_count=0),
    ]
    total_scored = sum(item.scored_row_count for item in summaries)
    total_anomaly = sum(item.anomaly_count for item in summaries)
    payload: dict[str, Any] = {
        "task": AnalysisTask.UNSUPERVISED_ANOMALY,
        "model_name": "Isolation Forest",
        "estimator_key": "isolation_forest",
        "feature_columns": list(_FEATURES),
        "fit_partition": "train",
        "fit_row_count": 10,
        "partition_summaries": summaries,
        "total_scored_row_count": total_scored,
        "total_anomaly_count": total_anomaly,
        "total_anomaly_fraction": total_anomaly / total_scored,
        "fit_seconds": 0.1,
        "total_scoring_seconds": 0.2,
        "total_seconds": 0.3,
        "combined_sorted_by_original_row_id": True,
        "created_at": datetime.now(UTC),
        "warnings": [],
    }
    payload.update(overrides)
    return AnomalyPipelineReport(**payload)


# --- AnomalyPipelinePolicy ---


def test_policy_defaults() -> None:
    policy = AnomalyPipelinePolicy()
    assert policy.require_safe_leakage_report is True
    assert policy.require_split_summary_match is True
    assert policy.score_train_partition is True
    assert policy.score_validation_partition is True
    assert policy.score_test_partition is True
    assert policy.sort_combined_by_original_row_id is True


@pytest.mark.parametrize(
    "field",
    [
        "require_safe_leakage_report",
        "require_split_summary_match",
        "score_train_partition",
        "score_validation_partition",
        "score_test_partition",
        "sort_combined_by_original_row_id",
    ],
)
@pytest.mark.parametrize("bad", [0, 1, "true", "false", None])
def test_policy_rejects_non_bool(field: str, bad: object) -> None:
    with pytest.raises(ValidationError):
        AnomalyPipelinePolicy(**{field: bad})


def test_policy_rejects_all_score_flags_false() -> None:
    with pytest.raises(ValidationError):
        AnomalyPipelinePolicy(
            score_train_partition=False,
            score_validation_partition=False,
            score_test_partition=False,
        )


def test_policy_round_trip() -> None:
    policy = AnomalyPipelinePolicy(score_test_partition=False)
    restored = AnomalyPipelinePolicy.model_validate(policy.model_dump())
    assert restored == policy


# --- AnomalyPartitionSummary ---


def test_partition_summary_scored_ok() -> None:
    summary = _partition_summary()
    assert summary.scored is True
    assert summary.score_min is not None


def test_partition_summary_empty_scored_ok() -> None:
    summary = _partition_summary(
        input_row_count=0,
        scored=True,
        scored_row_count=0,
        anomaly_count=0,
    )
    assert summary.score_mean is None


def test_partition_summary_disabled_ok() -> None:
    summary = _partition_summary(
        input_row_count=8,
        scored=False,
        scored_row_count=0,
        warnings=["scoring was disabled by policy"],
    )
    assert summary.scored is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_row_count": -1},
        {"scored_row_count": -1},
        {"scored_row_count": 11, "input_row_count": 10},
        {"anomaly_count": -1},
        {"anomaly_count": 11, "input_row_count": 10, "scored_row_count": 10},
        {"anomaly_fraction": -0.1, "input_row_count": 10, "anomaly_count": 0},
        {"anomaly_fraction": 1.1, "input_row_count": 10, "anomaly_count": 10},
        {"anomaly_fraction": float("nan")},
        {"scoring_seconds": -0.1},
        {"scoring_seconds": float("nan")},
    ],
)
def test_partition_summary_rejects_invalid_counts(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _partition_summary(**kwargs)


def test_partition_summary_disabled_with_scored_rows_rejected() -> None:
    with pytest.raises(ValidationError):
        AnomalyPartitionSummary(
            partition="train",
            input_row_count=5,
            scored=False,
            scored_row_count=5,
            anomaly_count=0,
            anomaly_fraction=0.0,
            scoring_seconds=0.0,
        )


def test_partition_summary_disabled_with_scores_rejected() -> None:
    with pytest.raises(ValidationError):
        AnomalyPartitionSummary(
            partition="train",
            input_row_count=5,
            scored=False,
            scored_row_count=0,
            anomaly_count=0,
            anomaly_fraction=0.0,
            score_min=0.1,
            score_max=0.2,
            score_mean=0.15,
            scoring_seconds=0.0,
        )


def test_partition_summary_missing_score_summary_rejected() -> None:
    with pytest.raises(ValidationError):
        AnomalyPartitionSummary(
            partition="train",
            input_row_count=5,
            scored=True,
            scored_row_count=5,
            anomaly_count=1,
            anomaly_fraction=0.2,
            score_min=None,
            score_max=0.5,
            score_mean=0.3,
            scoring_seconds=0.0,
        )


def test_partition_summary_score_order_rejected() -> None:
    with pytest.raises(ValidationError):
        _partition_summary(score_min=0.9, score_mean=0.5, score_max=0.1)


def test_partition_summary_fraction_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        _partition_summary(anomaly_count=1, anomaly_fraction=0.5)


def test_partition_summary_duplicate_warnings_rejected() -> None:
    with pytest.raises(ValidationError):
        _partition_summary(warnings=["a", "a"])


def test_partition_summary_mutable_default_independence() -> None:
    first = _partition_summary()
    second = _partition_summary()
    first.warnings.append("x")
    assert second.warnings == []


def test_partition_summary_round_trip() -> None:
    summary = _partition_summary(warnings=["note"])
    restored = AnomalyPartitionSummary.model_validate(summary.model_dump())
    assert restored == summary


# --- AnomalyPipelineReport ---


def test_report_ok() -> None:
    report = _valid_report()
    assert report.task is AnalysisTask.UNSUPERVISED_ANOMALY
    assert report.fit_partition == "train"


def test_report_rejects_wrong_task() -> None:
    with pytest.raises(ValidationError):
        _valid_report(task=AnalysisTask.REGRESSION)


@pytest.mark.parametrize("field", ["model_name", "estimator_key"])
@pytest.mark.parametrize("bad", ["", "   "])
def test_report_rejects_blank_identity(field: str, bad: str) -> None:
    with pytest.raises(ValidationError):
        _valid_report(**{field: bad})


def test_report_rejects_empty_features() -> None:
    with pytest.raises(ValidationError):
        _valid_report(feature_columns=[])


def test_report_rejects_duplicate_features() -> None:
    with pytest.raises(ValidationError):
        _valid_report(feature_columns=["f1", "f1"])


def test_report_rejects_original_row_id_feature() -> None:
    with pytest.raises(ValidationError):
        _valid_report(feature_columns=[ORIGINAL_ROW_ID_COLUMN, "f1"])


def test_report_rejects_zero_fit_rows() -> None:
    with pytest.raises(ValidationError):
        _valid_report(fit_row_count=0)


def test_report_rejects_wrong_summary_count() -> None:
    with pytest.raises(ValidationError):
        _valid_report(partition_summaries=[_partition_summary()])


def test_report_rejects_wrong_summary_order() -> None:
    summaries = [
        _partition_summary(partition="test"),
        _partition_summary(partition="validation"),
        _partition_summary(partition="train"),
    ]
    with pytest.raises(ValidationError):
        _valid_report(
            partition_summaries=summaries,
            total_scored_row_count=sum(s.scored_row_count for s in summaries),
            total_anomaly_count=sum(s.anomaly_count for s in summaries),
            total_anomaly_fraction=0.1,
        )


def test_report_rejects_duplicate_partition() -> None:
    summaries = [
        _partition_summary(partition="train"),
        _partition_summary(partition="train"),
        _partition_summary(partition="test"),
    ]
    with pytest.raises(ValidationError):
        _valid_report(partition_summaries=summaries)


def test_report_rejects_total_scored_mismatch() -> None:
    with pytest.raises(ValidationError):
        _valid_report(total_scored_row_count=1)


def test_report_rejects_total_anomaly_mismatch() -> None:
    with pytest.raises(ValidationError):
        _valid_report(total_anomaly_count=99)


def test_report_rejects_fraction_mismatch() -> None:
    with pytest.raises(ValidationError):
        _valid_report(total_anomaly_fraction=0.99)


def test_report_rejects_negative_fit_seconds() -> None:
    with pytest.raises(ValidationError):
        _valid_report(fit_seconds=-0.1, total_seconds=0.1)


def test_report_rejects_nan_scoring_seconds() -> None:
    with pytest.raises(ValidationError):
        _valid_report(total_scoring_seconds=float("nan"), total_seconds=0.1)


def test_report_rejects_total_seconds_mismatch() -> None:
    with pytest.raises(ValidationError):
        _valid_report(total_seconds=9.9)


def test_report_rejects_naive_created_at() -> None:
    with pytest.raises(ValidationError):
        _valid_report(created_at=datetime(2024, 1, 1))


def test_report_rejects_duplicate_warnings() -> None:
    with pytest.raises(ValidationError):
        _valid_report(warnings=["a", "a"])


def test_report_round_trip() -> None:
    report = _valid_report(warnings=["note"])
    restored = AnomalyPipelineReport.model_validate(report.model_dump(mode="json"))
    assert restored.model_name == report.model_name
    assert restored.total_anomaly_count == report.total_anomaly_count


# --- Outcome / construction ---


def test_outcome_frozen_slots_and_types() -> None:
    split = _make_split()
    pipeline = UnsupervisedAnomalyPipeline(
        model_config=IsolationForestConfig(n_estimators=20, random_state=7)
    )
    outcome = pipeline.run(
        split,
        feature_columns=_FEATURES,
        leakage_report=_safe_report(),
    )
    assert is_dataclass(outcome)
    assert outcome.__slots__ == (
        "fitted_model",
        "train_scored",
        "validation_scored",
        "test_scored",
        "combined_scored",
        "report",
    )
    assert {item.name for item in fields(outcome)} == set(outcome.__slots__)
    with pytest.raises(FrozenInstanceError):
        outcome.report = outcome.report  # type: ignore[misc]
    assert isinstance(outcome.fitted_model, IsolationForestAnomalyModel)
    assert isinstance(outcome.train_scored, pl.DataFrame)
    assert isinstance(outcome.validation_scored, pl.DataFrame)
    assert isinstance(outcome.test_scored, pl.DataFrame)
    assert isinstance(outcome.combined_scored, pl.DataFrame)
    assert isinstance(outcome.report, AnomalyPipelineReport)


def test_pipeline_construction_and_isolation() -> None:
    config = IsolationForestConfig(n_estimators=15, random_state=3)
    policy = AnomalyPipelinePolicy(score_test_partition=False)
    pipeline = UnsupervisedAnomalyPipeline(model_config=config, policy=policy)
    with pytest.raises(TypeError):
        UnsupervisedAnomalyPipeline(model_config="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        UnsupervisedAnomalyPipeline(policy="bad")  # type: ignore[arg-type]

    config.n_estimators = 99
    policy.score_test_partition = True
    split = _make_split()
    first = pipeline.run(
        split, feature_columns=_FEATURES, leakage_report=_safe_report()
    )
    assert first.fitted_model.get_metadata().n_estimators == 15
    assert first.report.partition_summaries[2].scored is False

    other = UnsupervisedAnomalyPipeline(
        model_config=IsolationForestConfig(n_estimators=11, random_state=9)
    )
    second = other.run(
        split, feature_columns=_FEATURES, leakage_report=_safe_report()
    )
    assert first.fitted_model is not second.fitted_model
    assert first.fitted_model.get_metadata().n_estimators != (
        second.fitted_model.get_metadata().n_estimators
    )


# --- run validation ---


def test_run_rejects_bad_split_type() -> None:
    pipeline = UnsupervisedAnomalyPipeline()
    with pytest.raises(TypeError):
        pipeline.run(
            "split",  # type: ignore[arg-type]
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_run_rejects_non_polars_partition() -> None:
    split = _make_split()
    bad = DatasetSplit(
        train=split.train,
        validation="bad",  # type: ignore[arg-type]
        test=split.test,
        summary=split.summary,
    )
    pipeline = UnsupervisedAnomalyPipeline()
    with pytest.raises(TypeError):
        pipeline.run(bad, feature_columns=_FEATURES, leakage_report=_safe_report())


def test_run_rejects_column_order_mismatch() -> None:
    split = _make_split()
    reordered = split.validation.select(["f2", "f1", ORIGINAL_ROW_ID_COLUMN])
    bad = DatasetSplit(
        train=split.train,
        validation=reordered,
        test=split.test,
        summary=split.summary,
    )
    pipeline = UnsupervisedAnomalyPipeline()
    with pytest.raises(DataValidationError):
        pipeline.run(bad, feature_columns=_FEATURES, leakage_report=_safe_report())


def test_run_rejects_dtype_mismatch() -> None:
    split = _make_split()
    casted = split.validation.with_columns(pl.col("f1").cast(pl.Float32))
    bad = DatasetSplit(
        train=split.train,
        validation=casted,
        test=split.test,
        summary=split.summary,
    )
    pipeline = UnsupervisedAnomalyPipeline()
    with pytest.raises(DataValidationError):
        pipeline.run(bad, feature_columns=_FEATURES, leakage_report=_safe_report())


def test_run_rejects_missing_original_row_id() -> None:
    split = _make_split()
    dropped = split.train.drop(ORIGINAL_ROW_ID_COLUMN)
    bad = DatasetSplit(
        train=dropped,
        validation=split.validation.drop(ORIGINAL_ROW_ID_COLUMN),
        test=split.test.drop(ORIGINAL_ROW_ID_COLUMN),
        summary=split.summary,
    )
    pipeline = UnsupervisedAnomalyPipeline()
    with pytest.raises(DataValidationError):
        pipeline.run(bad, feature_columns=_FEATURES, leakage_report=_safe_report())


@pytest.mark.parametrize("bad", ["f1", b"f1"])
def test_run_rejects_str_or_bytes_features(bad: object) -> None:
    split = _make_split()
    pipeline = UnsupervisedAnomalyPipeline()
    with pytest.raises(TypeError):
        pipeline.run(
            split,
            feature_columns=bad,  # type: ignore[arg-type]
            leakage_report=_safe_report(),
        )


@pytest.mark.parametrize(
    "features",
    [
        [1, "f2"],
        ["", "f2"],
        ["  ", "f2"],
        [],
        ["f1", "f1"],
        [ORIGINAL_ROW_ID_COLUMN, "f1"],
        [_SCORE_COLUMN, "f1"],
        ["missing", "f2"],
    ],
)
def test_run_rejects_invalid_features(features: list[Any]) -> None:
    split = _make_split()
    pipeline = UnsupervisedAnomalyPipeline()
    expected = (
        TypeError
        if features and not isinstance(features[0], str)
        else DataValidationError
    )
    with pytest.raises(expected):
        pipeline.run(
            split,
            feature_columns=features,
            leakage_report=_safe_report(),
        )


def test_run_rejects_bad_leakage_type() -> None:
    split = _make_split()
    pipeline = UnsupervisedAnomalyPipeline()
    with pytest.raises(TypeError):
        pipeline.run(
            split,
            feature_columns=_FEATURES,
            leakage_report="bad",  # type: ignore[arg-type]
        )


def test_run_rejects_checked_feature_order_and_set_mismatch() -> None:
    split = _make_split()
    pipeline = UnsupervisedAnomalyPipeline()
    with pytest.raises(DataValidationError):
        pipeline.run(
            split,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(["f2", "f1"]),
        )
    with pytest.raises(DataValidationError):
        pipeline.run(
            split,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(["f1"]),
        )


# --- Leakage gate ---


def test_leakage_gate_blocker_and_warning_and_override() -> None:
    split = _make_split()
    pipeline = UnsupervisedAnomalyPipeline()
    with pytest.raises(DataLeakageError, match="count=1"):
        pipeline.run(
            split,
            feature_columns=_FEATURES,
            leakage_report=_blocker_report(),
        )

    warning_outcome = pipeline.run(
        split,
        feature_columns=_FEATURES,
        leakage_report=_warning_report(),
    )
    assert any("WARNING" in item for item in warning_outcome.report.warnings)

    safe_outcome = pipeline.run(
        split,
        feature_columns=_FEATURES,
        leakage_report=_safe_report(),
    )
    assert safe_outcome.fitted_model.is_fitted

    allow = UnsupervisedAnomalyPipeline(
        policy=AnomalyPipelinePolicy(require_safe_leakage_report=False)
    )
    blocked_allowed = allow.run(
        split,
        feature_columns=_FEATURES,
        leakage_report=_blocker_report(),
    )
    assert blocked_allowed.fitted_model.is_fitted

    report = _blocker_report()
    before = report.model_dump()
    allow.run(split, feature_columns=_FEATURES, leakage_report=report)
    assert report.model_dump() == before


# --- original row id ---


def test_original_row_id_validation_cases() -> None:
    split = _make_split()
    pipeline = UnsupervisedAnomalyPipeline()

    null_train = split.train.with_columns(
        pl.when(pl.col(ORIGINAL_ROW_ID_COLUMN) == split.train[ORIGINAL_ROW_ID_COLUMN][0])
        .then(None)
        .otherwise(pl.col(ORIGINAL_ROW_ID_COLUMN))
        .alias(ORIGINAL_ROW_ID_COLUMN)
    )
    with pytest.raises(DataValidationError):
        pipeline.run(
            DatasetSplit(
                train=null_train,
                validation=split.validation,
                test=split.test,
                summary=split.summary,
            ),
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )

    string_ids = split.train.with_columns(
        pl.col(ORIGINAL_ROW_ID_COLUMN).cast(pl.String)
    )
    with pytest.raises(DataValidationError):
        pipeline.run(
            _manual_split(
                string_ids,
                split.validation.with_columns(
                    pl.col(ORIGINAL_ROW_ID_COLUMN).cast(pl.String)
                ),
                split.test.with_columns(pl.col(ORIGINAL_ROW_ID_COLUMN).cast(pl.String)),
            ),
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )

    bool_ids = pl.DataFrame(
        {
            "f1": [0.0, 1.0],
            "f2": [0.0, 1.0],
            ORIGINAL_ROW_ID_COLUMN: [True, False],
        }
    )
    with pytest.raises(DataValidationError):
        pipeline.run(
            _manual_split(bool_ids, bool_ids.clear(), bool_ids.clear()),
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )

    negative = split.train.with_columns(pl.lit(-1).alias(ORIGINAL_ROW_ID_COLUMN))
    with pytest.raises(DataValidationError):
        pipeline.run(
            DatasetSplit(
                train=negative,
                validation=split.validation,
                test=split.test,
                summary=split.summary,
            ),
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )

    dup = split.train.with_columns(
        pl.lit(split.train[ORIGINAL_ROW_ID_COLUMN][0]).alias(ORIGINAL_ROW_ID_COLUMN)
    )
    with pytest.raises(DataValidationError):
        pipeline.run(
            DatasetSplit(
                train=dup,
                validation=split.validation,
                test=split.test,
                summary=split.summary,
            ),
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_original_row_id_overlap_cases() -> None:
    split = _make_split()
    pipeline = UnsupervisedAnomalyPipeline()
    shared = int(split.train[ORIGINAL_ROW_ID_COLUMN][0])
    id_dtype = split.train.schema[ORIGINAL_ROW_ID_COLUMN]

    val_ids = split.validation[ORIGINAL_ROW_ID_COLUMN].to_list()
    val_ids[0] = shared
    val_overlap = split.validation.with_columns(
        pl.Series(ORIGINAL_ROW_ID_COLUMN, val_ids, dtype=id_dtype)
    )
    with pytest.raises(DataLeakageError, match="train/validation"):
        pipeline.run(
            DatasetSplit(
                train=split.train,
                validation=val_overlap,
                test=split.test,
                summary=split.summary,
            ),
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )

    test_ids = split.test[ORIGINAL_ROW_ID_COLUMN].to_list()
    test_ids[0] = shared
    test_overlap = split.test.with_columns(
        pl.Series(ORIGINAL_ROW_ID_COLUMN, test_ids, dtype=id_dtype)
    )
    with pytest.raises(DataLeakageError, match="train/test"):
        pipeline.run(
            DatasetSplit(
                train=split.train,
                validation=split.validation,
                test=test_overlap,
                summary=split.summary,
            ),
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )

    shared_vt = int(split.validation[ORIGINAL_ROW_ID_COLUMN][0])
    test_ids_vt = split.test[ORIGINAL_ROW_ID_COLUMN].to_list()
    test_ids_vt[0] = shared_vt
    test_vt = split.test.with_columns(
        pl.Series(ORIGINAL_ROW_ID_COLUMN, test_ids_vt, dtype=id_dtype)
    )
    with pytest.raises(DataLeakageError, match="validation/test"):
        pipeline.run(
            DatasetSplit(
                train=split.train,
                validation=split.validation,
                test=test_vt,
                summary=split.summary,
            ),
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_noncontiguous_ids_and_dtype_preserved() -> None:
    train = pl.DataFrame(
        {
            "f1": [0.0, 0.1, 0.2, 25.0],
            "f2": [0.0, -0.1, 0.05, -25.0],
            ORIGINAL_ROW_ID_COLUMN: pl.Series([10, 20, 30, 40], dtype=pl.Int32),
        }
    )
    validation = pl.DataFrame(
        {
            "f1": [0.05],
            "f2": [0.02],
            ORIGINAL_ROW_ID_COLUMN: pl.Series([50], dtype=pl.Int32),
        }
    )
    test = pl.DataFrame(
        {
            "f1": [0.03],
            "f2": [-0.02],
            ORIGINAL_ROW_ID_COLUMN: pl.Series([60], dtype=pl.Int32),
        }
    )
    split = _manual_split(train, validation, test)
    outcome = UnsupervisedAnomalyPipeline(
        model_config=IsolationForestConfig(n_estimators=20, random_state=1)
    ).run(split, feature_columns=_FEATURES, leakage_report=_safe_report())
    assert outcome.train_scored[ORIGINAL_ROW_ID_COLUMN].dtype == pl.Int32
    assert outcome.combined_scored[ORIGINAL_ROW_ID_COLUMN].to_list() == [
        10,
        20,
        30,
        40,
        50,
        60,
    ]


# --- SplitSummary ---


def test_split_summary_mismatch_and_opt_out() -> None:
    split = _make_split()
    pipeline = UnsupervisedAnomalyPipeline()
    bad_summary = split.summary.model_copy(
        update={"train_original_row_ids": list(reversed(split.summary.train_original_row_ids))}
    )
    bad = DatasetSplit(
        train=split.train,
        validation=split.validation,
        test=split.test,
        summary=bad_summary,
    )
    with pytest.raises(DataValidationError, match="train"):
        pipeline.run(bad, feature_columns=_FEATURES, leakage_report=_safe_report())

    for field, message in (
        (
            "validation_original_row_ids",
            "validation",
        ),
        (
            "test_original_row_ids",
            "test",
        ),
    ):
        values = list(getattr(split.summary, field))
        values = list(reversed(values)) if values else values
        mismatched = DatasetSplit(
            train=split.train,
            validation=split.validation,
            test=split.test,
            summary=split.summary.model_copy(update={field: values}),
        )
        with pytest.raises(DataValidationError, match=message):
            pipeline.run(
                mismatched,
                feature_columns=_FEATURES,
                leakage_report=_safe_report(),
            )

    allow = UnsupervisedAnomalyPipeline(
        policy=AnomalyPipelinePolicy(require_split_summary_match=False)
    )
    outcome = allow.run(bad, feature_columns=_FEATURES, leakage_report=_safe_report())
    assert outcome.fitted_model.is_fitted

    before = split.summary.model_dump()
    pipeline.run(split, feature_columns=_FEATURES, leakage_report=_safe_report())
    assert split.summary.model_dump() == before


# --- reserved columns / minimum data ---


@pytest.mark.parametrize("column", _RESULT_COLUMNS)
def test_reserved_column_collision(column: str) -> None:
    split = _make_split()
    colliding = split.train.with_columns(pl.lit(1).alias(column))
    bad = DatasetSplit(
        train=colliding,
        validation=split.validation.with_columns(pl.lit(1).alias(column)),
        test=split.test.with_columns(pl.lit(1).alias(column)),
        summary=split.summary,
    )
    with pytest.raises(DataValidationError, match=column):
        UnsupervisedAnomalyPipeline().run(
            bad,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_minimum_data_rules() -> None:
    empty_train = pl.DataFrame(
        schema={
            "f1": pl.Float64,
            "f2": pl.Float64,
            ORIGINAL_ROW_ID_COLUMN: pl.Int64,
        }
    )
    validation = pl.DataFrame(
        {
            "f1": [0.0],
            "f2": [0.0],
            ORIGINAL_ROW_ID_COLUMN: [1],
        }
    )
    test = pl.DataFrame(
        {
            "f1": [0.1],
            "f2": [0.1],
            ORIGINAL_ROW_ID_COLUMN: [2],
        }
    )
    with pytest.raises(InsufficientDataError):
        UnsupervisedAnomalyPipeline().run(
            _manual_split(empty_train, validation, test),
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )

    split = _split_with_empty_validation()
    assert split.validation.height == 0
    outcome = UnsupervisedAnomalyPipeline(
        model_config=IsolationForestConfig(n_estimators=15, random_state=2)
    ).run(split, feature_columns=_FEATURES, leakage_report=_safe_report())
    assert outcome.validation_scored.height == 0
    assert outcome.validation_scored.columns[-4:] == _RESULT_COLUMNS

    split_empty_test = _split_with_empty_test()
    assert split_empty_test.test.height == 0
    outcome_test = UnsupervisedAnomalyPipeline(
        model_config=IsolationForestConfig(n_estimators=15, random_state=2)
    ).run(split_empty_test, feature_columns=_FEATURES, leakage_report=_safe_report())
    assert outcome_test.test_scored.height == 0

    forced = _train_only_split()
    policy = AnomalyPipelinePolicy(
        score_train_partition=False,
        score_validation_partition=True,
        score_test_partition=True,
    )
    with pytest.raises(InsufficientDataError):
        UnsupervisedAnomalyPipeline(policy=policy).run(
            forced,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


# --- fit / scoring behavior ---


def test_fit_train_only_and_feature_order() -> None:
    split = _make_split()
    train_before = split.train.clone()
    features = ["f2", "f1"]
    pipeline = UnsupervisedAnomalyPipeline(
        model_config=IsolationForestConfig(n_estimators=20, random_state=5)
    )
    outcome = pipeline.run(
        split,
        feature_columns=features,
        leakage_report=_safe_report(features),
    )
    assert outcome.fitted_model.is_fitted is True
    assert list(outcome.fitted_model.feature_names) == features
    assert outcome.fitted_model.fit_row_count == split.train.height
    assert split.train.equals(train_before)


def test_fit_called_once_and_partitions_not_used() -> None:
    split = _make_split()
    pipeline = UnsupervisedAnomalyPipeline(
        model_config=IsolationForestConfig(n_estimators=15, random_state=4)
    )
    fit_frames: list[pl.DataFrame] = []
    detect_calls: list[str] = []

    original_create = __import__(
        "process_intelligence.models.anomaly_pipeline",
        fromlist=["create_isolation_forest_anomaly_model"],
    ).create_isolation_forest_anomaly_model

    def _factory(*, config: IsolationForestConfig | None = None) -> IsolationForestAnomalyModel:
        model = original_create(config=config)
        original_fit = model.fit
        original_detect = model.detect

        def fit_wrapper(X: Any, y: Any = None) -> IsolationForestAnomalyModel:
            fit_frames.append(X.clone() if isinstance(X, pl.DataFrame) else X)
            return original_fit(X, y)

        def detect_wrapper(X: Any) -> AnomalyDetectionResult:
            detect_calls.append("detect")
            return original_detect(X)

        model.fit = fit_wrapper  # type: ignore[method-assign]
        model.detect = detect_wrapper  # type: ignore[method-assign]
        return model

    with patch(
        "process_intelligence.models.anomaly_pipeline.create_isolation_forest_anomaly_model",
        side_effect=_factory,
    ):
        outcome = pipeline.run(
            split,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )

    assert len(fit_frames) == 1
    assert list(fit_frames[0].columns) == _FEATURES
    assert ORIGINAL_ROW_ID_COLUMN not in fit_frames[0].columns
    assert fit_frames[0].height == split.train.height
    assert detect_calls.count("detect") == 3
    assert outcome.train_scored.height == split.train.height


def test_train_validation_test_scoring_basics() -> None:
    split = _make_split()
    train_ids = split.train[ORIGINAL_ROW_ID_COLUMN].to_list()
    val_ids = split.validation[ORIGINAL_ROW_ID_COLUMN].to_list()
    test_ids = split.test[ORIGINAL_ROW_ID_COLUMN].to_list()
    outcome = UnsupervisedAnomalyPipeline(
        model_config=IsolationForestConfig(n_estimators=25, random_state=8)
    ).run(split, feature_columns=_FEATURES, leakage_report=_safe_report())

    assert outcome.train_scored.height == split.train.height
    assert outcome.train_scored[ORIGINAL_ROW_ID_COLUMN].to_list() == train_ids
    assert outcome.train_scored[_DATA_PARTITION_COLUMN].unique().to_list() == ["train"]
    assert outcome.train_scored[_SCORE_COLUMN].dtype == pl.Float64
    assert outcome.train_scored[_IS_ANOMALY_COLUMN].dtype == pl.Boolean
    raw = outcome.train_scored[_RAW_PREDICTION_COLUMN].to_list()
    assert set(raw).issubset({-1, 1})
    flags = outcome.train_scored[_IS_ANOMALY_COLUMN].to_list()
    assert flags == [value == -1 for value in raw]

    assert outcome.validation_scored.height == split.validation.height
    assert outcome.validation_scored[ORIGINAL_ROW_ID_COLUMN].to_list() == val_ids
    assert outcome.validation_scored[_DATA_PARTITION_COLUMN].unique().to_list() == [
        "validation"
    ]
    assert outcome.test_scored.height == split.test.height
    assert outcome.test_scored[ORIGINAL_ROW_ID_COLUMN].to_list() == test_ids
    assert outcome.test_scored[_DATA_PARTITION_COLUMN].unique().to_list() == ["test"]


def test_empty_validation_and_test_scored_schema() -> None:
    forced = _train_only_split()
    outcome = UnsupervisedAnomalyPipeline(
        model_config=IsolationForestConfig(n_estimators=15, random_state=3)
    ).run(forced, feature_columns=_FEATURES, leakage_report=_safe_report())
    assert outcome.validation_scored.height == 0
    assert outcome.test_scored.height == 0
    assert list(outcome.validation_scored.columns) == list(outcome.train_scored.columns)
    assert list(outcome.test_scored.dtypes) == list(outcome.train_scored.dtypes)


def test_scoring_disabled_partitions() -> None:
    split = _make_split()
    detect_counts = {"n": 0}
    original_create = __import__(
        "process_intelligence.models.anomaly_pipeline",
        fromlist=["create_isolation_forest_anomaly_model"],
    ).create_isolation_forest_anomaly_model

    def _factory(*, config: IsolationForestConfig | None = None) -> IsolationForestAnomalyModel:
        model = original_create(config=config)
        original_detect = model.detect

        def detect_wrapper(X: Any) -> AnomalyDetectionResult:
            detect_counts["n"] += 1
            return original_detect(X)

        model.detect = detect_wrapper  # type: ignore[method-assign]
        return model

    policy = AnomalyPipelinePolicy(
        score_train_partition=False,
        score_validation_partition=False,
        score_test_partition=True,
    )
    with patch(
        "process_intelligence.models.anomaly_pipeline.create_isolation_forest_anomaly_model",
        side_effect=_factory,
    ):
        outcome = UnsupervisedAnomalyPipeline(
            model_config=IsolationForestConfig(n_estimators=15, random_state=6),
            policy=policy,
        ).run(split, feature_columns=_FEATURES, leakage_report=_safe_report())

    assert detect_counts["n"] == 1
    assert outcome.train_scored.height == 0
    assert outcome.validation_scored.height == 0
    assert outcome.test_scored.height == split.test.height
    assert list(outcome.train_scored.columns[-4:]) == _RESULT_COLUMNS
    assert outcome.report.partition_summaries[0].scored is False
    assert outcome.report.partition_summaries[1].scored is False
    assert any("disabled" in w.lower() for w in outcome.report.warnings)
    assert outcome.combined_scored.height == split.test.height


def test_scored_dataframe_schema_and_immutability() -> None:
    split = _make_split()
    train_before = split.train.clone()
    outcome = UnsupervisedAnomalyPipeline(
        model_config=IsolationForestConfig(n_estimators=20, random_state=11)
    ).run(split, feature_columns=_FEATURES, leakage_report=_safe_report())

    scored = outcome.train_scored
    assert list(scored.columns[:-4]) == list(split.train.columns)
    assert list(scored.columns[-4:]) == _RESULT_COLUMNS
    assert scored[_SCORE_COLUMN].dtype == pl.Float64
    assert scored[_IS_ANOMALY_COLUMN].dtype == pl.Boolean
    assert scored[_RAW_PREDICTION_COLUMN].dtype.is_signed_integer()
    assert scored[_DATA_PARTITION_COLUMN].dtype == pl.String
    for column in split.train.columns:
        assert scored[column].dtype == split.train[column].dtype
        assert scored[column].to_list() == split.train[column].to_list()
    assert scored.height == split.train.height
    assert scored[ORIGINAL_ROW_ID_COLUMN].n_unique() == scored.height
    assert split.train.equals(train_before)

    scored = scored.with_columns(pl.lit(999).alias("mutated"))
    assert "mutated" not in split.train.columns


def test_detect_result_validation_failures() -> None:
    split = _make_split()
    pipeline = UnsupervisedAnomalyPipeline(
        model_config=IsolationForestConfig(n_estimators=10, random_state=1)
    )
    original_create = __import__(
        "process_intelligence.models.anomaly_pipeline",
        fromlist=["create_isolation_forest_anomaly_model"],
    ).create_isolation_forest_anomaly_model

    def _bad_length(*, config: IsolationForestConfig | None = None) -> IsolationForestAnomalyModel:
        model = original_create(config=config)
        model.detect = MagicMock(  # type: ignore[method-assign]
            return_value=AnomalyDetectionResult(
                scores=[0.1],
                is_anomaly=[False],
                raw_predictions=[1],
                threshold=0.0,
                row_count=1,
                anomaly_count=0,
                anomaly_fraction=0.0,
                score_min=0.1,
                score_max=0.1,
                score_mean=0.1,
            )
        )
        return model

    with patch(
        "process_intelligence.models.anomaly_pipeline.create_isolation_forest_anomaly_model",
        side_effect=_bad_length,
    ):
        with pytest.raises(ProcessIntelligenceError):
            pipeline.run(
                split,
                feature_columns=_FEATURES,
                leakage_report=_safe_report(),
            )

    def _bad_raw(*, config: IsolationForestConfig | None = None) -> IsolationForestAnomalyModel:
        model = original_create(config=config)
        height = split.train.height

        def detect_bad(X: Any) -> AnomalyDetectionResult:
            n = X.height if hasattr(X, "height") else len(X)
            # Bypass AnomalyDetectionResult validator by returning a plain object
            # that looks wrong after construction with valid values then mutate via mock.
            result = AnomalyDetectionResult(
                scores=[0.0] * n,
                is_anomaly=[False] * n,
                raw_predictions=[1] * n,
                threshold=0.0,
                row_count=n,
                anomaly_count=0,
                anomaly_fraction=0.0,
                score_min=0.0,
                score_max=0.0,
                score_mean=0.0,
            )
            object.__setattr__(result, "raw_predictions", [0] * n)
            return result

        _ = height
        model.detect = detect_bad  # type: ignore[method-assign]
        return model

    with patch(
        "process_intelligence.models.anomaly_pipeline.create_isolation_forest_anomaly_model",
        side_effect=_bad_raw,
    ):
        with pytest.raises(ProcessIntelligenceError):
            pipeline.run(
                split,
                feature_columns=_FEATURES,
                leakage_report=_safe_report(),
            )

    def _bad_flags(*, config: IsolationForestConfig | None = None) -> IsolationForestAnomalyModel:
        model = original_create(config=config)

        def detect_bad(X: Any) -> AnomalyDetectionResult:
            n = X.height if hasattr(X, "height") else len(X)
            result = AnomalyDetectionResult(
                scores=[0.0] * n,
                is_anomaly=[False] * n,
                raw_predictions=[1] * n,
                threshold=0.0,
                row_count=n,
                anomaly_count=0,
                anomaly_fraction=0.0,
                score_min=0.0,
                score_max=0.0,
                score_mean=0.0,
            )
            object.__setattr__(result, "is_anomaly", [True] + [False] * (n - 1))
            object.__setattr__(result, "raw_predictions", [1] * n)
            object.__setattr__(result, "anomaly_count", 1)
            return result

        model.detect = detect_bad  # type: ignore[method-assign]
        return model

    with patch(
        "process_intelligence.models.anomaly_pipeline.create_isolation_forest_anomaly_model",
        side_effect=_bad_flags,
    ):
        with pytest.raises(ProcessIntelligenceError):
            pipeline.run(
                split,
                feature_columns=_FEATURES,
                leakage_report=_safe_report(),
            )


# --- combined / report / metadata / determinism ---


def test_combined_scored_sorting_and_counts() -> None:
    split = _make_split()
    pipeline = UnsupervisedAnomalyPipeline(
        model_config=IsolationForestConfig(n_estimators=20, random_state=12)
    )
    outcome = pipeline.run(
        split, feature_columns=_FEATURES, leakage_report=_safe_report()
    )
    expected_rows = split.train.height + split.validation.height + split.test.height
    assert outcome.combined_scored.height == expected_rows
    assert outcome.combined_scored.height == outcome.report.total_scored_row_count
    ids = outcome.combined_scored[ORIGINAL_ROW_ID_COLUMN].to_list()
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)
    all_ids = (
        split.train[ORIGINAL_ROW_ID_COLUMN].to_list()
        + split.validation[ORIGINAL_ROW_ID_COLUMN].to_list()
        + split.test[ORIGINAL_ROW_ID_COLUMN].to_list()
    )
    assert set(ids) == set(all_ids)
    anomaly_count = int(outcome.combined_scored[_IS_ANOMALY_COLUMN].sum())
    assert anomaly_count == outcome.report.total_anomaly_count
    assert list(outcome.combined_scored.columns) == list(outcome.train_scored.columns)

    unsorted = UnsupervisedAnomalyPipeline(
        model_config=IsolationForestConfig(n_estimators=20, random_state=12),
        policy=AnomalyPipelinePolicy(sort_combined_by_original_row_id=False),
    ).run(split, feature_columns=_FEATURES, leakage_report=_safe_report())
    concat_ids = (
        unsorted.train_scored[ORIGINAL_ROW_ID_COLUMN].to_list()
        + unsorted.validation_scored[ORIGINAL_ROW_ID_COLUMN].to_list()
        + unsorted.test_scored[ORIGINAL_ROW_ID_COLUMN].to_list()
    )
    assert unsorted.combined_scored[ORIGINAL_ROW_ID_COLUMN].to_list() == concat_ids
    assert unsorted.report.combined_sorted_by_original_row_id is False


def test_disabled_partition_excluded_from_combined() -> None:
    split = _make_split()
    outcome = UnsupervisedAnomalyPipeline(
        model_config=IsolationForestConfig(n_estimators=15, random_state=2),
        policy=AnomalyPipelinePolicy(score_validation_partition=False),
    ).run(split, feature_columns=_FEATURES, leakage_report=_safe_report())
    assert outcome.combined_scored.height == split.train.height + split.test.height
    assert "validation" not in outcome.combined_scored[_DATA_PARTITION_COLUMN].to_list()


def test_report_and_metadata_consistency() -> None:
    split = _make_split()
    outcome = UnsupervisedAnomalyPipeline(
        model_config=IsolationForestConfig(n_estimators=18, random_state=13)
    ).run(split, feature_columns=["f2", "f1"], leakage_report=_safe_report(["f2", "f1"]))
    report = outcome.report
    metadata = outcome.fitted_model.get_metadata()

    assert report.model_name == metadata.model_name == "Isolation Forest"
    assert report.estimator_key == metadata.estimator_key == "isolation_forest"
    assert report.task is AnalysisTask.UNSUPERVISED_ANOMALY
    assert report.feature_columns == ["f2", "f1"]
    assert report.fit_partition == "train"
    assert report.fit_row_count == split.train.height
    assert [item.partition for item in report.partition_summaries] == [
        "train",
        "validation",
        "test",
    ]
    assert report.total_scored_row_count == (
        report.partition_summaries[0].scored_row_count
        + report.partition_summaries[1].scored_row_count
        + report.partition_summaries[2].scored_row_count
    )
    assert report.total_anomaly_count == sum(
        item.anomaly_count for item in report.partition_summaries
    )
    if report.total_scored_row_count:
        assert math.isclose(
            report.total_anomaly_fraction,
            report.total_anomaly_count / report.total_scored_row_count,
        )
    assert report.fit_seconds >= 0.0
    assert report.total_scoring_seconds >= 0.0
    assert math.isclose(
        report.total_seconds,
        report.fit_seconds + report.total_scoring_seconds,
    )
    assert math.isfinite(report.fit_seconds)
    assert report.created_at.tzinfo is not None
    assert report.combined_sorted_by_original_row_id is True
    assert list(metadata.features) == ["f2", "f1"]
    assert metadata.fit_row_count == split.train.height
    assert not hasattr(metadata, "estimator") or getattr(metadata, "estimator", None) is None
    dumped = metadata.model_dump()
    assert "estimator" not in dumped
    assert "training_data" not in dumped

    for summary in report.partition_summaries:
        assert summary.scoring_seconds >= 0.0
        assert summary.input_row_count >= 0
        if summary.scored and summary.scored_row_count > 0:
            assert summary.score_min is not None
            assert summary.score_max is not None
            assert summary.score_mean is not None


def test_warning_cases() -> None:
    split = _make_split()
    warning_outcome = UnsupervisedAnomalyPipeline(
        model_config=IsolationForestConfig(n_estimators=15, random_state=1)
    ).run(split, feature_columns=_FEATURES, leakage_report=_warning_report())
    assert warning_outcome.report.warnings[0].startswith("LeakageReport contains WARNING")

    empty_val = _split_with_empty_validation()
    empty_outcome = UnsupervisedAnomalyPipeline(
        model_config=IsolationForestConfig(n_estimators=15, random_state=1)
    ).run(empty_val, feature_columns=_FEATURES, leakage_report=_safe_report())
    assert any("empty" in item.lower() for item in empty_outcome.report.warnings)

    disabled = UnsupervisedAnomalyPipeline(
        model_config=IsolationForestConfig(n_estimators=15, random_state=1),
        policy=AnomalyPipelinePolicy(score_test_partition=False),
    ).run(split, feature_columns=_FEATURES, leakage_report=_safe_report())
    assert any("disabled" in item.lower() for item in disabled.report.warnings)
    assert len(disabled.report.warnings) == len(set(disabled.report.warnings))


def test_zero_and_all_anomaly_warnings_via_mock() -> None:
    split = _make_split()
    original_create = __import__(
        "process_intelligence.models.anomaly_pipeline",
        fromlist=["create_isolation_forest_anomaly_model"],
    ).create_isolation_forest_anomaly_model

    def _factory_zero(
        *, config: IsolationForestConfig | None = None
    ) -> IsolationForestAnomalyModel:
        model = original_create(config=config)
        original_detect = model.detect

        def detect_zero(X: Any) -> AnomalyDetectionResult:
            n = X.height if hasattr(X, "height") else len(X)
            if n == 0:
                return original_detect(X)
            return AnomalyDetectionResult(
                scores=[0.1] * n,
                is_anomaly=[False] * n,
                raw_predictions=[1] * n,
                threshold=0.0,
                row_count=n,
                anomaly_count=0,
                anomaly_fraction=0.0,
                score_min=0.1,
                score_max=0.1,
                score_mean=0.1,
            )

        model.detect = detect_zero  # type: ignore[method-assign]
        return model

    zero = UnsupervisedAnomalyPipeline(
        model_config=IsolationForestConfig(n_estimators=10, random_state=1)
    )
    with patch(
        "process_intelligence.models.anomaly_pipeline.create_isolation_forest_anomaly_model",
        side_effect=_factory_zero,
    ):
        outcome = zero.run(
            split, feature_columns=_FEATURES, leakage_report=_safe_report()
        )
    assert any("No scored rows were flagged" in w for w in outcome.report.warnings)

    def _factory_all(*, config: IsolationForestConfig | None = None) -> IsolationForestAnomalyModel:
        model = original_create(config=config)

        def detect_all(X: Any) -> AnomalyDetectionResult:
            n = X.height if hasattr(X, "height") else len(X)
            if n == 0:
                return AnomalyDetectionResult(
                    scores=[],
                    is_anomaly=[],
                    raw_predictions=[],
                    threshold=0.0,
                    row_count=0,
                    anomaly_count=0,
                    anomaly_fraction=0.0,
                )
            return AnomalyDetectionResult(
                scores=[1.0] * n,
                is_anomaly=[True] * n,
                raw_predictions=[-1] * n,
                threshold=0.0,
                row_count=n,
                anomaly_count=n,
                anomaly_fraction=1.0,
                score_min=1.0,
                score_max=1.0,
                score_mean=1.0,
            )

        model.detect = detect_all  # type: ignore[method-assign]
        return model

    with patch(
        "process_intelligence.models.anomaly_pipeline.create_isolation_forest_anomaly_model",
        side_effect=_factory_all,
    ):
        all_outcome = zero.run(
            split, feature_columns=_FEATURES, leakage_report=_safe_report()
        )
    assert any("All scored rows were flagged" in w for w in all_outcome.report.warnings)


def test_outcome_reuse_and_no_estimator_leak() -> None:
    split = _make_split()
    outcome = UnsupervisedAnomalyPipeline(
        model_config=IsolationForestConfig(n_estimators=20, random_state=14)
    ).run(split, feature_columns=_FEATURES, leakage_report=_safe_report())
    reused = outcome.fitted_model.detect(split.test.select(_FEATURES))
    assert isinstance(reused, AnomalyDetectionResult)
    scores = outcome.fitted_model.score_samples(split.test.select(_FEATURES))
    assert scores.shape[0] == split.test.height
    assert not hasattr(outcome, "estimator")
    assert outcome.__slots__ == (
        "fitted_model",
        "train_scored",
        "validation_scored",
        "test_scored",
        "combined_scored",
        "report",
    )


def test_immutability_and_determinism() -> None:
    split = _make_split()
    train_before = split.train.clone()
    val_before = split.validation.clone()
    test_before = split.test.clone()
    features = list(_FEATURES)
    config = IsolationForestConfig(n_estimators=20, random_state=21)
    policy = AnomalyPipelinePolicy()
    config_before = config.model_dump()
    policy_before = policy.model_dump()
    report = _safe_report()
    report_before = report.model_dump()

    pipeline = UnsupervisedAnomalyPipeline(model_config=config, policy=policy)
    first = pipeline.run(split, feature_columns=features, leakage_report=report)
    second = pipeline.run(split, feature_columns=features, leakage_report=report)
    other = UnsupervisedAnomalyPipeline(
        model_config=IsolationForestConfig(n_estimators=20, random_state=21),
        policy=AnomalyPipelinePolicy(),
    ).run(split, feature_columns=features, leakage_report=_safe_report())

    assert first.fitted_model is not second.fitted_model
    assert first.train_scored[_SCORE_COLUMN].to_list() == second.train_scored[
        _SCORE_COLUMN
    ].to_list()
    assert first.train_scored[_RAW_PREDICTION_COLUMN].to_list() == second.train_scored[
        _RAW_PREDICTION_COLUMN
    ].to_list()
    assert other.train_scored[_SCORE_COLUMN].to_list() == first.train_scored[
        _SCORE_COLUMN
    ].to_list()

    with pytest.raises(FrozenInstanceError):
        first.report = first.report  # type: ignore[misc]
    mutated = first.combined_scored.with_columns(pl.lit(0.0).alias(_SCORE_COLUMN))
    third = pipeline.run(split, feature_columns=features, leakage_report=report)
    assert third.combined_scored[_SCORE_COLUMN].to_list() == first.combined_scored[
        _SCORE_COLUMN
    ].to_list()
    assert mutated[_SCORE_COLUMN].to_list() != third.combined_scored[_SCORE_COLUMN].to_list()

    assert split.train.equals(train_before)
    assert split.validation.equals(val_before)
    assert split.test.equals(test_before)
    assert features == _FEATURES
    assert config.model_dump() == config_before
    assert policy.model_dump() == policy_before
    assert report.model_dump() == report_before

    score_policy = AnomalyPipelinePolicy(score_test_partition=False)
    scored_diff_policy = UnsupervisedAnomalyPipeline(
        model_config=IsolationForestConfig(n_estimators=20, random_state=21),
        policy=score_policy,
    ).run(split, feature_columns=features, leakage_report=_safe_report())
    assert scored_diff_policy.train_scored[_SCORE_COLUMN].to_list() == first.train_scored[
        _SCORE_COLUMN
    ].to_list()


def test_package_exports() -> None:
    from process_intelligence import models

    for name in (
        "AnomalyPipelinePolicy",
        "AnomalyPartitionSummary",
        "AnomalyPipelineReport",
        "AnomalyPipelineOutcome",
        "UnsupervisedAnomalyPipeline",
    ):
        assert name in models.__all__
        assert getattr(models, name) is not None
