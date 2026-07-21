"""Unit tests for residual anomaly pipeline (Step 7G)."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import FrozenInstanceError, fields, is_dataclass
from datetime import UTC, datetime
from typing import Any, ClassVar, Self
from unittest.mock import patch

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
from process_intelligence.core.protocols import BaseAnalysisModel, DataFrameLike, SeriesLike
from process_intelligence.core.schemas import (
    ExplanationResult,
    ModelEvaluation,
    ModelMetadata,
    ModelSpec,
)
from process_intelligence.data.loader import ORIGINAL_ROW_ID_COLUMN
from process_intelligence.evaluation.leakage import (
    LeakageIssue,
    LeakageIssueType,
    LeakageReport,
    LeakageSeverity,
)
from process_intelligence.evaluation.splitting import DatasetSplit, SplitStrategy, SplitSummary
from process_intelligence.models import (
    CandidateRunStatus,
    CandidateScreeningResult,
    ModelScreeningOutcome,
    ModelScreeningSummary,
    ResidualAnomalyConfig,
    ResidualAnomalyDetector,
    ResidualAnomalyPipeline,
    ResidualAnomalyPipelineOutcome,
    ResidualAnomalyPipelinePolicy,
    ResidualAnomalyPipelineReport,
    ResidualAnomalyResult,
    ResidualPartitionSummary,
    ResidualThresholdMethod,
    SupervisedModelScreener,
    create_default_supervised_model_registry,
)
from process_intelligence.models.residual_anomaly_pipeline import (
    _DATA_PARTITION_COLUMN,
    _IS_RESIDUAL_ANOMALY_COLUMN,
    _PARTITION_DISABLED_WARNING,
    _REGRESSION_PREDICTION_COLUMN,
    _RESERVED_RESULT_COLUMNS,
    _RESIDUAL_ANOMALY_RAW_PREDICTION_COLUMN,
    _WARNING_FRACTION_SHIFT,
    _WARNING_LEAKAGE,
    _WARNING_TRAIN_SCORING_DISABLED,
    _WARNING_VALIDATION_ALL_ANOMALIES,
    _WARNING_VALIDATION_ZERO_ANOMALIES,
)

_FEATURES = ["f1", "f2"]
_TARGET = "y"
_RESULT_COLUMNS = list(_RESERVED_RESULT_COLUMNS)


def _summary(
    *,
    train_ids: list[int],
    validation_ids: list[int],
    test_ids: list[int],
) -> SplitSummary:
    total = len(train_ids) + len(validation_ids) + len(test_ids)
    return SplitSummary(
        strategy=SplitStrategy.RANDOM,
        random_state=42,
        requested_test_size=0.2,
        requested_validation_size=0.2,
        train_row_count=len(train_ids),
        validation_row_count=len(validation_ids),
        test_row_count=len(test_ids),
        train_fraction=len(train_ids) / total if total else 0.0,
        validation_fraction=len(validation_ids) / total if total else 0.0,
        test_fraction=len(test_ids) / total if total else 0.0,
        train_original_row_ids=list(train_ids),
        validation_original_row_ids=list(validation_ids),
        test_original_row_ids=list(test_ids),
    )


def _regression_split(
    *,
    test_target_offset: float = 1000.0,
    test_feature_offset: float = 0.0,
) -> DatasetSplit:
    train = pl.DataFrame(
        {
            ORIGINAL_ROW_ID_COLUMN: [0, 1, 2, 3, 4, 5],
            "f1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "f2": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            "y": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        }
    )
    validation = pl.DataFrame(
        {
            ORIGINAL_ROW_ID_COLUMN: [6, 7, 8],
            "f1": [0.5, 1.5, 2.5],
            "f2": [1.25, 1.75, 2.25],
            "y": [1.0, 3.0, 5.0],
        }
    )
    test = pl.DataFrame(
        {
            ORIGINAL_ROW_ID_COLUMN: [9, 10, 11],
            "f1": [
                0.5 + test_feature_offset,
                1.5 + test_feature_offset,
                2.5 + test_feature_offset,
            ],
            "f2": [
                1.25 + test_feature_offset,
                1.75 + test_feature_offset,
                2.25 + test_feature_offset,
            ],
            "y": [
                1.0 + test_target_offset,
                3.0 + test_target_offset,
                5.0 + test_target_offset,
            ],
        }
    )
    return DatasetSplit(
        train=train,
        validation=validation,
        test=test,
        summary=_summary(
            train_ids=[0, 1, 2, 3, 4, 5],
            validation_ids=[6, 7, 8],
            test_ids=[9, 10, 11],
        ),
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


def _split_with_empty_test(*, rows: int = 12) -> DatasetSplit:
    split = _regression_split()
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


def _safe_report(features: list[str] | None = None) -> LeakageReport:
    cols = list(features or _FEATURES)
    return LeakageReport(
        is_safe=True,
        issues=[],
        blocker_count=0,
        warning_count=0,
        checked_feature_columns=cols,
        checked_preprocessing_event_count=0,
    )


def _warning_report(features: list[str] | None = None) -> LeakageReport:
    cols = list(features or _FEATURES)
    issue = LeakageIssue(
        issue_type=LeakageIssueType.PREPROCESSING_FIT_SCOPE_UNKNOWN,
        severity=LeakageSeverity.WARNING,
        message="unknown fit scope",
        suggested_action="record fit scope",
    )
    return LeakageReport(
        is_safe=True,
        issues=[issue],
        blocker_count=0,
        warning_count=1,
        checked_feature_columns=cols,
        checked_preprocessing_event_count=1,
    )


def _blocker_report(features: list[str] | None = None) -> LeakageReport:
    cols = list(features or _FEATURES)
    issue = LeakageIssue(
        issue_type=LeakageIssueType.TARGET_INCLUDED_AS_FEATURE,
        severity=LeakageSeverity.BLOCKER,
        columns=["y"],
        message="target in features",
        suggested_action="remove target",
    )
    return LeakageReport(
        is_safe=False,
        issues=[issue],
        blocker_count=1,
        warning_count=0,
        checked_feature_columns=cols,
        checked_preprocessing_event_count=0,
    )


def _spec(
    *,
    name: str,
    task: AnalysisTask = AnalysisTask.REGRESSION,
    estimator_key: str = "linear_regression",
    priority: int = 10,
) -> ModelSpec:
    return ModelSpec(
        name=name,
        task=task,
        estimator_key=estimator_key,
        optional_dependencies=[],
        priority=priority,
        time_budget_seconds=5.0,
    )


def _success_result(
    *,
    name: str = "Linear Regression",
    estimator_key: str = "linear_regression",
    registry_rank: int = 1,
    metrics: dict[str, float] | None = None,
    task: AnalysisTask = AnalysisTask.REGRESSION,
    warnings: list[str] | None = None,
) -> CandidateScreeningResult:
    payload = metrics or {"rmse": 1.0, "mae": 0.5, "r2": 0.9}
    primary = "rmse" if task is AnalysisTask.REGRESSION else "f1_macro"
    return CandidateScreeningResult(
        spec=_spec(name=name, estimator_key=estimator_key, task=task),
        status=CandidateRunStatus.SUCCESS,
        registry_rank=registry_rank,
        metrics=dict(payload),
        primary_metric_name=primary,
        primary_metric_value=payload[primary],
        fit_seconds=0.01,
        evaluation_seconds=0.02,
        total_seconds=0.03,
        warnings=[] if warnings is None else list(warnings),
    )


def _valid_summary(**overrides: object) -> ModelScreeningSummary:
    candidates = [
        _success_result(
            name="Dummy Regressor",
            estimator_key="dummy_regressor",
            registry_rank=0,
            metrics={"rmse": 2.0, "mae": 1.5, "r2": 0.0},
        ),
        _success_result(
            name="Linear Regression",
            registry_rank=1,
            metrics={"rmse": 1.0, "mae": 0.5, "r2": 0.9},
        ),
    ]
    payload: dict[str, object] = {
        "task": AnalysisTask.REGRESSION,
        "target_column": _TARGET,
        "feature_columns": list(_FEATURES),
        "candidate_results": candidates,
        "selected_model_name": "Linear Regression",
        "selected_estimator_key": "linear_regression",
        "selected_metrics": {"rmse": 1.0, "mae": 0.5, "r2": 0.9},
        "selected_registry_rank": 1,
        "ranking_metric": "rmse",
        "higher_is_better": False,
        "baseline_model_name": "Dummy Regressor",
        "baseline_metrics": {"rmse": 2.0, "mae": 1.5, "r2": 0.0},
        "selected_beats_baseline": True,
        "successful_model_names": ["Dummy Regressor", "Linear Regression"],
        "failed_model_names": [],
        "train_row_count": 6,
        "validation_row_count": 3,
        "test_row_count": 3,
        "warnings": [],
    }
    payload.update(overrides)
    return ModelScreeningSummary(**payload)  # type: ignore[arg-type]


def _summary_with_overrides(**override: object) -> ModelScreeningSummary:
    summary = _valid_summary()
    payload = summary.model_dump()
    payload.update(override)
    if "candidate_results" not in override:
        payload["candidate_results"] = summary.candidate_results
    return ModelScreeningSummary.model_construct(**payload)  # type: ignore[arg-type]


def _default_prediction(X: DataFrameLike) -> np.ndarray:
    if isinstance(X, pl.DataFrame):
        return (
            X.get_column("f1").to_numpy()
            + X.get_column("f2").to_numpy()
        ).astype(float)
    frame = pl.DataFrame(X)
    return (
        frame.get_column("f1").to_numpy() + frame.get_column("f2").to_numpy()
    ).astype(float)


class SpyRegressionModel(BaseAnalysisModel):
    """Fitted regression stub that records predict calls without refitting."""

    def __init__(
        self,
        *,
        name: str = "Linear Regression",
        task: AnalysisTask = AnalysisTask.REGRESSION,
        estimator_key: str = "linear_regression",
        feature_names: list[str] | None = None,
        fit_row_count: int = 6,
        is_fitted: bool = True,
        predictions: np.ndarray | None = None,
        prediction_fn: Callable[[DataFrameLike], np.ndarray] | None = None,
        metadata_features: list[str] | None = None,
    ) -> None:
        self._name = name
        self._task = task
        self._is_fitted = is_fitted
        self._fit_row_count = fit_row_count
        self._feature_names = list(feature_names or _FEATURES)
        self._metadata_features = list(metadata_features or self._feature_names)
        self._spec = _spec(name=name, task=task, estimator_key=estimator_key)
        self.predict_calls: list[tuple[list[str], int]] = []
        self.fit_calls = 0
        self._predictions = predictions
        self._prediction_fn = prediction_fn or _default_prediction
        self._state_token = object()

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def fit_row_count(self) -> int:
        return self._fit_row_count

    def fit(self, X: DataFrameLike, y: SeriesLike | None = None) -> Self:
        _ = (X, y)
        self.fit_calls += 1
        self._is_fitted = True
        return self

    def predict(self, X: DataFrameLike) -> np.ndarray:
        if isinstance(X, pl.DataFrame):
            columns = list(X.columns)
            height = int(X.height)
        else:
            columns = [str(c) for c in X.columns]
            height = len(X)
        self.predict_calls.append((columns, height))
        if self._predictions is not None:
            return self._predictions
        return self._prediction_fn(X)

    def evaluate(
        self,
        X: DataFrameLike,
        y: SeriesLike | None = None,
    ) -> ModelEvaluation:
        _ = (X, y)
        return ModelEvaluation(metrics={"rmse": 1.0}, notes=[])

    def explain(self, X: DataFrameLike) -> ExplanationResult:
        _ = X
        return ExplanationResult(method="stub", feature_importances={})

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_name=self._name,
            version="0",
            task=self._task,
            features=list(self._metadata_features),
        )


def _fitted_spy(**overrides: object) -> SpyRegressionModel:
    return SpyRegressionModel(**overrides)  # type: ignore[arg-type]


def _outcome_from_summary(
    summary: ModelScreeningSummary,
    selected: SpyRegressionModel | None = None,
) -> ModelScreeningOutcome:
    model = selected or _fitted_spy(
        name=summary.selected_model_name,
        estimator_key=summary.selected_estimator_key,
        fit_row_count=summary.train_row_count,
        feature_names=list(summary.feature_columns),
        metadata_features=list(summary.feature_columns),
    )
    return ModelScreeningOutcome(selected_model=model, summary=summary)


class SpyDetector(ResidualAnomalyDetector):
    """Records fit/detect calls while delegating to the real detector."""

    created: ClassVar[list[SpyDetector]] = []

    def __init__(self, config: ResidualAnomalyConfig | None = None) -> None:
        super().__init__(config=config)
        self.fit_calls: list[tuple[Any, Any]] = []
        self.detect_calls: list[tuple[Any, Any]] = []
        SpyDetector.created.append(self)

    def fit(self, y: Any, predictions: Any) -> Self:
        self.fit_calls.append((y, predictions))
        return super().fit(y, predictions)

    def detect(self, y: Any, predictions: Any) -> ResidualAnomalyResult:
        self.detect_calls.append((y, predictions))
        return super().detect(y, predictions)


@pytest.fixture(autouse=True)
def _reset_spy_detector_created() -> None:
    SpyDetector.created.clear()


def _detector_patch() -> patch:
    return patch(
        "process_intelligence.models.residual_anomaly_pipeline.ResidualAnomalyDetector",
        SpyDetector,
    )


def _default_screening_outcome(
    split: DatasetSplit,
    spy_model: SpyRegressionModel | None = None,
) -> ModelScreeningOutcome:
    summary = _valid_summary(
        train_row_count=split.train.height,
        validation_row_count=max(split.validation.height, 1),
        test_row_count=split.test.height,
    )
    return _outcome_from_summary(
        summary,
        spy_model or _fitted_spy(fit_row_count=split.train.height),
    )


def _run_pipeline(
    split: DatasetSplit,
    screening_outcome: ModelScreeningOutcome,
    *,
    target_column: str = _TARGET,
    feature_columns: Sequence[str] | None = None,
    leakage_report: LeakageReport | None = None,
    policy: ResidualAnomalyPipelinePolicy | None = None,
    detector_config: ResidualAnomalyConfig | None = None,
    use_detector_spy: bool = True,
) -> ResidualAnomalyPipelineOutcome:
    features = list(feature_columns or _FEATURES)
    report = _safe_report(features) if leakage_report is None else leakage_report
    pipeline = ResidualAnomalyPipeline(
        detector_config=detector_config,
        policy=policy,
    )
    if use_detector_spy:
        with _detector_patch():
            return pipeline.run(
                split,
                screening_outcome,
                target_column=target_column,
                feature_columns=features,
                leakage_report=report,
            )
    return pipeline.run(
        split,
        screening_outcome,
        target_column=target_column,
        feature_columns=features,
        leakage_report=report,
    )


def _run_default(
    *,
    split: DatasetSplit | None = None,
    screening_outcome: ModelScreeningOutcome | None = None,
    target_column: str = _TARGET,
    feature_columns: Sequence[str] | None = None,
    leakage_report: LeakageReport | None = None,
    policy: ResidualAnomalyPipelinePolicy | None = None,
    detector_config: ResidualAnomalyConfig | None = None,
    spy_model: SpyRegressionModel | None = None,
    pipeline: ResidualAnomalyPipeline | None = None,
    use_detector_spy: bool = True,
) -> ResidualAnomalyPipelineOutcome:
    used_split = _regression_split() if split is None else split
    outcome = screening_outcome or _default_screening_outcome(used_split, spy_model)
    features = list(feature_columns or _FEATURES)
    report = _safe_report(features) if leakage_report is None else leakage_report
    runner_pipeline = pipeline or ResidualAnomalyPipeline(
        detector_config=detector_config,
        policy=policy,
    )
    if use_detector_spy:
        with _detector_patch():
            return runner_pipeline.run(
                used_split,
                outcome,
                target_column=target_column,
                feature_columns=features,
                leakage_report=report,
            )
    return runner_pipeline.run(
        used_split,
        outcome,
        target_column=target_column,
        feature_columns=features,
        leakage_report=report,
    )


def _partition_summary(
    *,
    partition: str = "train",
    input_row_count: int = 6,
    scored: bool = True,
    prediction_count: int | None = None,
    anomaly_count: int = 1,
    anomaly_fraction: float | None = None,
    residual_mean: float | None = 0.0,
    residual_std: float | None = 0.5,
    score_min: float | None = 0.1,
    score_max: float | None = 0.9,
    score_mean: float | None = 0.5,
    score_std: float | None = 0.2,
    prediction_seconds: float = 0.01,
    scoring_seconds: float = 0.02,
    warnings: list[str] | None = None,
) -> ResidualPartitionSummary:
    if prediction_count is None:
        prediction_count = input_row_count if scored else 0
    if anomaly_fraction is None:
        anomaly_fraction = (
            0.0 if prediction_count == 0 else anomaly_count / prediction_count
        )
    if not scored or prediction_count == 0:
        residual_mean = None
        residual_std = None
        score_min = None
        score_max = None
        score_mean = None
        score_std = None
        anomaly_count = 0
        anomaly_fraction = 0.0
        prediction_seconds = 0.0
        scoring_seconds = 0.0
    total_seconds = prediction_seconds + scoring_seconds
    return ResidualPartitionSummary(
        partition=partition,  # type: ignore[arg-type]
        input_row_count=input_row_count,
        scored=scored,
        prediction_count=prediction_count,
        anomaly_count=anomaly_count,
        anomaly_fraction=anomaly_fraction,
        residual_mean=residual_mean,
        residual_std=residual_std,
        score_min=score_min,
        score_max=score_max,
        score_mean=score_mean,
        score_std=score_std,
        prediction_seconds=prediction_seconds,
        scoring_seconds=scoring_seconds,
        total_seconds=total_seconds,
        warnings=[] if warnings is None else warnings,
    )


def _valid_report(**overrides: Any) -> ResidualAnomalyPipelineReport:
    train_summary = _partition_summary(
        partition="train",
        input_row_count=6,
        anomaly_count=1,
    )
    validation_summary = _partition_summary(
        partition="validation",
        input_row_count=3,
        anomaly_count=0,
        anomaly_fraction=0.0,
    )
    total_scored = train_summary.prediction_count + validation_summary.prediction_count
    total_anomaly = train_summary.anomaly_count + validation_summary.anomaly_count
    payload: dict[str, Any] = {
        "task": AnalysisTask.REGRESSION,
        "model_name": "Linear Regression",
        "estimator_key": "linear_regression",
        "target_column": _TARGET,
        "feature_columns": list(_FEATURES),
        "detector_method": ResidualThresholdMethod.MAD,
        "calibration_partition": "validation",
        "calibration_row_count": 3,
        "residual_center": 0.0,
        "residual_scale": 1.0,
        "threshold": 3.5,
        "train_row_count": 6,
        "validation_row_count": 3,
        "test_row_count": 3,
        "partition_summaries": [train_summary, validation_summary],
        "total_scored_row_count": total_scored,
        "total_anomaly_count": total_anomaly,
        "total_anomaly_fraction": total_anomaly / total_scored,
        "train_validation_anomaly_fraction_shift": abs(
            validation_summary.anomaly_fraction - train_summary.anomaly_fraction
        ),
        "calibration_seconds": 0.01,
        "total_prediction_seconds": 0.02,
        "total_scoring_seconds": 0.03,
        "total_seconds": 0.06,
        "combined_sorted_by_original_row_id": True,
        "created_at": datetime.now(UTC),
        "warnings": [],
    }
    payload.update(overrides)
    return ResidualAnomalyPipelineReport(**payload)


def _forced_anomaly_result(
    base: ResidualAnomalyResult,
    *,
    anomaly_count: int,
) -> ResidualAnomalyResult:
    row_count = base.row_count
    flags = [True] * anomaly_count + [False] * (row_count - anomaly_count)
    raws = [-1 if flag else 1 for flag in flags]
    payload = base.model_dump()
    payload.update(
        {
            "anomaly_count": anomaly_count,
            "anomaly_fraction": 0.0 if row_count == 0 else anomaly_count / row_count,
            "is_anomaly": flags,
            "raw_predictions": raws,
        }
    )
    return ResidualAnomalyResult(**payload)


def _bad_result(
    split: DatasetSplit,
    *,
    overrides: dict[str, Any],
) -> ResidualAnomalyResult:
    model = _fitted_spy()
    preds = model.predict(split.validation.select(_FEATURES))
    y = split.validation.get_column(_TARGET)
    detector = ResidualAnomalyDetector()
    detector.fit(y, preds)
    good = detector.detect(y, preds)
    payload = good.model_dump()
    payload.update(overrides)
    return ResidualAnomalyResult.model_construct(**payload)


# --- ResidualAnomalyPipelinePolicy (1-11) ---


def test_policy_defaults() -> None:
    policy = ResidualAnomalyPipelinePolicy()
    assert policy.require_safe_leakage_report is True
    assert policy.require_screening_consistency is True
    assert policy.require_split_summary_match is True
    assert policy.minimum_validation_rows == 1
    assert policy.score_train_partition is True
    assert policy.sort_combined_by_original_row_id is True
    assert policy.maximum_anomaly_fraction_shift == pytest.approx(0.10)
    assert policy.warn_on_fraction_shift is True


@pytest.mark.parametrize(
    "field",
    [
        "require_safe_leakage_report",
        "require_screening_consistency",
        "require_split_summary_match",
        "score_train_partition",
        "sort_combined_by_original_row_id",
        "warn_on_fraction_shift",
    ],
)
@pytest.mark.parametrize("bad", [0, 1, "true", "false", None])
def test_policy_rejects_non_bool(field: str, bad: object) -> None:
    with pytest.raises(ValidationError):
        ResidualAnomalyPipelinePolicy(**{field: bad})


def test_policy_rejects_minimum_validation_rows_zero() -> None:
    with pytest.raises(ValidationError):
        ResidualAnomalyPipelinePolicy(minimum_validation_rows=0)


@pytest.mark.parametrize("bad", [True, False, "1"])
def test_policy_rejects_minimum_validation_rows_bool(bad: object) -> None:
    with pytest.raises(ValidationError):
        ResidualAnomalyPipelinePolicy(minimum_validation_rows=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [-0.1, 1.1, True, float("nan"), float("inf")],
)
def test_policy_rejects_invalid_maximum_anomaly_fraction_shift(value: object) -> None:
    with pytest.raises(ValidationError):
        ResidualAnomalyPipelinePolicy(maximum_anomaly_fraction_shift=value)  # type: ignore[arg-type]


def test_policy_round_trip() -> None:
    policy = ResidualAnomalyPipelinePolicy(
        score_train_partition=False,
        maximum_anomaly_fraction_shift=0.25,
    )
    restored = ResidualAnomalyPipelinePolicy.model_validate(policy.model_dump())
    assert restored == policy


# --- ResidualPartitionSummary (12-30) ---


def test_partition_summary_scored_ok() -> None:
    summary = _partition_summary()
    assert summary.scored is True
    assert summary.score_min is not None


def test_partition_summary_disabled_ok() -> None:
    summary = _partition_summary(
        input_row_count=6,
        scored=False,
        prediction_count=0,
        warnings=[_PARTITION_DISABLED_WARNING],
    )
    assert summary.scored is False
    assert summary.prediction_count == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_row_count": -1},
        {"prediction_count": -1},
        {"anomaly_count": -1},
        {"prediction_count": 7, "input_row_count": 6},
        {"anomaly_count": 7, "prediction_count": 6, "input_row_count": 6},
        {"anomaly_fraction": -0.1},
        {"anomaly_fraction": 1.1},
        {"prediction_seconds": -0.1},
        {"prediction_seconds": float("nan")},
    ],
)
def test_partition_summary_rejects_invalid_counts(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _partition_summary(**kwargs)


def test_partition_summary_rejects_total_seconds_mismatch() -> None:
    with pytest.raises(ValidationError):
        ResidualPartitionSummary(
            partition="train",
            input_row_count=6,
            scored=True,
            prediction_count=6,
            anomaly_count=1,
            anomaly_fraction=1 / 6,
            residual_mean=0.0,
            residual_std=0.5,
            score_min=0.1,
            score_max=0.9,
            score_mean=0.5,
            score_std=0.2,
            prediction_seconds=0.01,
            scoring_seconds=0.02,
            total_seconds=0.5,
        )


def test_partition_summary_fraction_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        _partition_summary(anomaly_count=1, anomaly_fraction=0.5)


def test_partition_summary_scored_false_with_counts_rejected() -> None:
    with pytest.raises(ValidationError):
        ResidualPartitionSummary(
            partition="train",
            input_row_count=5,
            scored=False,
            prediction_count=5,
            anomaly_count=0,
            anomaly_fraction=0.0,
            prediction_seconds=0.0,
            scoring_seconds=0.0,
            total_seconds=0.0,
        )


def test_partition_summary_scored_false_with_summaries_rejected() -> None:
    with pytest.raises(ValidationError):
        ResidualPartitionSummary(
            partition="train",
            input_row_count=5,
            scored=False,
            prediction_count=0,
            anomaly_count=0,
            anomaly_fraction=0.0,
            residual_mean=0.0,
            prediction_seconds=0.0,
            scoring_seconds=0.0,
            total_seconds=0.0,
        )


def test_partition_summary_scored_true_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        _partition_summary(
            input_row_count=6,
            prediction_count=5,
        )


def test_partition_summary_score_order_rejected() -> None:
    with pytest.raises(ValidationError):
        _partition_summary(score_min=0.9, score_mean=0.5, score_max=0.1)


def test_partition_summary_validation_scored_false_rejected() -> None:
    with pytest.raises(ValidationError):
        ResidualPartitionSummary(
            partition="validation",
            input_row_count=3,
            scored=False,
            prediction_count=0,
            anomaly_count=0,
            anomaly_fraction=0.0,
            prediction_seconds=0.0,
            scoring_seconds=0.0,
            total_seconds=0.0,
        )


def test_partition_summary_duplicate_warnings_rejected() -> None:
    with pytest.raises(ValidationError):
        _partition_summary(warnings=["a", "a"])


def test_partition_summary_round_trip() -> None:
    summary = _partition_summary(warnings=["note"])
    restored = ResidualPartitionSummary.model_validate(summary.model_dump())
    assert restored == summary


# --- ResidualAnomalyPipelineReport (31-59) ---


def test_report_ok() -> None:
    report = _valid_report()
    assert report.task is AnalysisTask.REGRESSION
    assert report.calibration_partition == "validation"


def test_report_rejects_wrong_task() -> None:
    with pytest.raises(ValidationError):
        _valid_report(task=AnalysisTask.CLASSIFICATION)


@pytest.mark.parametrize("field", ["model_name", "estimator_key", "target_column"])
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


def test_report_rejects_target_in_features() -> None:
    with pytest.raises(ValidationError):
        _valid_report(feature_columns=["f1", "y"])


def test_report_rejects_bad_calibration_partition() -> None:
    with pytest.raises(ValidationError):
        _valid_report(calibration_partition="train")  # type: ignore[arg-type]


def test_report_rejects_calibration_row_count_zero() -> None:
    with pytest.raises(ValidationError):
        _valid_report(calibration_row_count=0)


def test_report_rejects_negative_threshold() -> None:
    with pytest.raises(ValidationError):
        _valid_report(threshold=-0.1)


def test_report_rejects_nan_residual_center() -> None:
    with pytest.raises(ValidationError):
        _valid_report(residual_center=float("nan"))


def test_report_rejects_residual_scale_zero() -> None:
    with pytest.raises(ValidationError):
        _valid_report(residual_scale=0.0)


def test_report_rejects_zero_train_or_validation_rows() -> None:
    with pytest.raises(ValidationError):
        _valid_report(train_row_count=0)
    with pytest.raises(ValidationError):
        _valid_report(validation_row_count=0)


def test_report_rejects_wrong_summary_count() -> None:
    with pytest.raises(ValidationError):
        _valid_report(partition_summaries=[_partition_summary()])


def test_report_rejects_wrong_summary_order() -> None:
    summaries = [
        _partition_summary(partition="validation", input_row_count=3, anomaly_count=0),
        _partition_summary(partition="train"),
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


def test_report_shift_none_when_train_disabled() -> None:
    train_disabled = _partition_summary(
        partition="train",
        scored=False,
        prediction_count=0,
        warnings=[_PARTITION_DISABLED_WARNING],
    )
    val = _partition_summary(
        partition="validation",
        input_row_count=3,
        anomaly_count=0,
        anomaly_fraction=0.0,
    )
    report = _valid_report(
        partition_summaries=[train_disabled, val],
        total_scored_row_count=3,
        total_anomaly_count=0,
        total_anomaly_fraction=0.0,
        train_validation_anomaly_fraction_shift=None,
    )
    assert report.train_validation_anomaly_fraction_shift is None


def test_report_shift_required_when_train_scored() -> None:
    with pytest.raises(ValidationError):
        _valid_report(train_validation_anomaly_fraction_shift=None)


def test_report_rejects_negative_timing() -> None:
    with pytest.raises(ValidationError):
        _valid_report(calibration_seconds=-0.1, total_seconds=0.05)


def test_report_rejects_nan_timing() -> None:
    with pytest.raises(ValidationError):
        _valid_report(total_scoring_seconds=float("nan"), total_seconds=0.06)


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
    restored = ResidualAnomalyPipelineReport.model_validate(report.model_dump(mode="json"))
    assert restored.model_name == report.model_name
    assert restored.total_anomaly_count == report.total_anomaly_count


# --- Outcome (60-65) ---


def test_outcome_frozen_slots_and_types() -> None:
    outcome = _run_default()
    assert is_dataclass(outcome)
    assert outcome.__slots__ == (
        "regression_model",
        "residual_detector",
        "train_scored",
        "validation_scored",
        "combined_scored",
        "report",
    )
    assert {item.name for item in fields(outcome)} == set(outcome.__slots__)
    with pytest.raises(FrozenInstanceError):
        outcome.report = outcome.report  # type: ignore[misc]
    assert isinstance(outcome.regression_model, BaseAnalysisModel)
    assert isinstance(outcome.residual_detector, ResidualAnomalyDetector)
    assert isinstance(outcome.train_scored, pl.DataFrame)
    assert isinstance(outcome.validation_scored, pl.DataFrame)
    assert isinstance(outcome.combined_scored, pl.DataFrame)
    assert isinstance(outcome.report, ResidualAnomalyPipelineReport)


# --- Pipeline constructor (66-73) ---


def test_pipeline_construction_defaults_and_isolation() -> None:
    config = ResidualAnomalyConfig(mad_multiplier=4.0)
    policy = ResidualAnomalyPipelinePolicy(score_train_partition=False)
    pipeline = ResidualAnomalyPipeline(detector_config=config, policy=policy)
    with pytest.raises(TypeError):
        ResidualAnomalyPipeline(detector_config="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ResidualAnomalyPipeline(policy="bad")  # type: ignore[arg-type]

    config.mad_multiplier = 99.0
    policy.score_train_partition = True
    outcome = _run_default(pipeline=pipeline)
    assert outcome.report.partition_summaries[0].scored is False

    other_outcome = _run_default(use_detector_spy=False)
    assert outcome.residual_detector is not other_outcome.residual_detector


def test_pipeline_instance_separation() -> None:
    second = ResidualAnomalyPipeline()
    a = _run_default(use_detector_spy=False)
    b = second.run(
        _regression_split(),
        _outcome_from_summary(_valid_summary()),
        target_column=_TARGET,
        feature_columns=_FEATURES,
        leakage_report=_safe_report(),
    )
    assert a.residual_detector is not b.residual_detector


# --- Split / schema (74-81) ---


def test_run_rejects_bad_split_type() -> None:
    pipeline = ResidualAnomalyPipeline()
    with pytest.raises(TypeError):
        pipeline.run(
            "split",  # type: ignore[arg-type]
            _outcome_from_summary(_valid_summary()),
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_run_rejects_non_polars_partition() -> None:
    split = _regression_split()
    bad = DatasetSplit(
        train=split.train,
        validation="bad",  # type: ignore[arg-type]
        test=split.test,
        summary=split.summary,
    )
    pipeline = ResidualAnomalyPipeline()
    with pytest.raises(TypeError):
        pipeline.run(
            bad,
            _default_screening_outcome(split),
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_run_rejects_column_order_mismatch() -> None:
    split = _regression_split()
    reordered = split.validation.select(["f2", "f1", ORIGINAL_ROW_ID_COLUMN, "y"])
    bad = DatasetSplit(
        train=split.train,
        validation=reordered,
        test=split.test,
        summary=split.summary,
    )
    with pytest.raises(DataValidationError):
        _run_default(split=bad)


def test_run_rejects_dtype_mismatch() -> None:
    split = _regression_split()
    casted = split.validation.with_columns(pl.col("f1").cast(pl.Float32))
    bad = DatasetSplit(
        train=split.train,
        validation=casted,
        test=split.test,
        summary=split.summary,
    )
    with pytest.raises(DataValidationError):
        _run_default(split=bad)


def test_run_rejects_missing_original_row_id() -> None:
    split = _regression_split()
    bad = DatasetSplit(
        train=split.train.drop(ORIGINAL_ROW_ID_COLUMN),
        validation=split.validation.drop(ORIGINAL_ROW_ID_COLUMN),
        test=split.test.drop(ORIGINAL_ROW_ID_COLUMN),
        summary=split.summary,
    )
    with pytest.raises(DataValidationError):
        _run_default(split=bad)


def test_run_rejects_empty_train() -> None:
    split = _regression_split()
    empty = split.train.clear()
    bad = _manual_split(empty, split.validation, split.test)
    with pytest.raises(InsufficientDataError):
        _run_default(split=bad)


def test_run_rejects_validation_below_minimum() -> None:
    split = _regression_split()
    empty_val = split.validation.clear()
    bad = _manual_split(split.train, empty_val, split.test)
    pipeline = ResidualAnomalyPipeline()
    with pytest.raises(InsufficientDataError):
        pipeline.run(
            bad,
            _default_screening_outcome(split),
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_run_allows_empty_test() -> None:
    split = _split_with_empty_test()
    outcome = _run_default(split=split)
    assert outcome.report.test_row_count == 0
    assert outcome.combined_scored.height == split.train.height + split.validation.height


# --- Target (82-85) ---


@pytest.mark.parametrize("bad", [123, None])
def test_run_rejects_non_str_target(bad: object) -> None:
    with pytest.raises(TypeError):
        _run_default(target_column=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["", "   "])
def test_run_rejects_empty_target(bad: str) -> None:
    with pytest.raises(DataValidationError):
        _run_default(target_column=bad)


def test_run_rejects_original_row_id_target() -> None:
    with pytest.raises(DataValidationError):
        _run_default(target_column=ORIGINAL_ROW_ID_COLUMN)


def test_run_rejects_missing_target_column() -> None:
    split = _regression_split()
    with pytest.raises(DataValidationError):
        _run_default(split=split, target_column="missing")


# --- Features (86-95) ---


@pytest.mark.parametrize("bad", ["f1", b"f1"])
def test_run_rejects_str_or_bytes_features(bad: object) -> None:
    split = _regression_split()
    pipeline = ResidualAnomalyPipeline()
    with pytest.raises(TypeError):
        pipeline.run(
            split,
            _default_screening_outcome(split),
            target_column=_TARGET,
            feature_columns=bad,  # type: ignore[arg-type]
            leakage_report=_safe_report(),
        )


@pytest.mark.parametrize(
    "features,expected",
    [
        ([1, "f2"], TypeError),
        (["", "f2"], DataValidationError),
        (["  ", "f2"], DataValidationError),
        ([], DataValidationError),
        (["f1", "f1"], DataValidationError),
        ([ORIGINAL_ROW_ID_COLUMN, "f1"], DataValidationError),
        ([_REGRESSION_PREDICTION_COLUMN, "f1"], DataValidationError),
        (["missing", "f2"], DataValidationError),
        (["f1", "y"], DataValidationError),
    ],
)
def test_run_rejects_invalid_features(
    features: list[Any],
    expected: type[Exception],
) -> None:
    split = _regression_split()
    pipeline = ResidualAnomalyPipeline()
    with pytest.raises(expected):
        pipeline.run(
            split,
            _default_screening_outcome(split),
            target_column=_TARGET,
            feature_columns=features,
            leakage_report=_safe_report(_FEATURES),
        )


# --- Leakage (96-103) ---


def test_run_rejects_bad_leakage_type() -> None:
    split = _regression_split()
    pipeline = ResidualAnomalyPipeline()
    with pytest.raises(TypeError):
        pipeline.run(
            split,
            _outcome_from_summary(_valid_summary()),
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report="bad",  # type: ignore[arg-type]
        )


def test_leakage_gate_blocker_warning_safe_and_override() -> None:
    split = _regression_split()
    pipeline = ResidualAnomalyPipeline()
    with pytest.raises(DataLeakageError, match="count=1"):
        pipeline.run(
            split,
            _outcome_from_summary(_valid_summary()),
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_blocker_report(),
        )

    warning_outcome = _run_default(leakage_report=_warning_report())
    assert _WARNING_LEAKAGE in warning_outcome.report.warnings

    _run_default(leakage_report=_safe_report())

    allow = ResidualAnomalyPipeline(
        policy=ResidualAnomalyPipelinePolicy(require_safe_leakage_report=False)
    )
    blocked = allow.run(
        split,
        _outcome_from_summary(_valid_summary()),
        target_column=_TARGET,
        feature_columns=_FEATURES,
        leakage_report=_blocker_report(),
    )
    assert blocked.residual_detector.is_fitted

    report = _blocker_report()
    before = report.model_dump()
    allow.run(
        split,
        _outcome_from_summary(_valid_summary()),
        target_column=_TARGET,
        feature_columns=_FEATURES,
        leakage_report=report,
    )
    assert report.model_dump() == before


def test_run_rejects_checked_feature_order_and_set_mismatch() -> None:
    with pytest.raises(DataValidationError):
        _run_default(leakage_report=_safe_report(["f2", "f1"]))
    with pytest.raises(DataValidationError):
        _run_default(leakage_report=_safe_report(["f1"]))


# --- IDs / SplitSummary (104-115) ---


def test_original_row_id_validation_cases() -> None:
    split = _regression_split()
    null_train = split.train.with_columns(
        pl.when(pl.col(ORIGINAL_ROW_ID_COLUMN) == split.train[ORIGINAL_ROW_ID_COLUMN][0])
        .then(None)
        .otherwise(pl.col(ORIGINAL_ROW_ID_COLUMN))
        .alias(ORIGINAL_ROW_ID_COLUMN)
    )
    with pytest.raises(DataValidationError):
        _run_default(
            split=DatasetSplit(
                train=null_train,
                validation=split.validation,
                test=split.test,
                summary=split.summary,
            )
        )

    string_ids = split.train.with_columns(
        pl.col(ORIGINAL_ROW_ID_COLUMN).cast(pl.String)
    )
    with pytest.raises(DataValidationError):
        _run_default(
            split=_manual_split(
                string_ids,
                split.validation.with_columns(
                    pl.col(ORIGINAL_ROW_ID_COLUMN).cast(pl.String)
                ),
                split.test.with_columns(pl.col(ORIGINAL_ROW_ID_COLUMN).cast(pl.String)),
            )
        )

    bool_ids = pl.DataFrame(
        {
            "f1": [0.0, 1.0],
            "f2": [0.0, 1.0],
            ORIGINAL_ROW_ID_COLUMN: [True, False],
            "y": [0.0, 1.0],
        }
    )
    val_ok = pl.DataFrame(
        {
            "f1": [0.5],
            "f2": [0.5],
            ORIGINAL_ROW_ID_COLUMN: [2],
            "y": [1.0],
        }
    )
    with pytest.raises(DataValidationError):
        _run_default(
            split=_manual_split(bool_ids, val_ok, val_ok.clear()),
        )

    negative = split.train.with_columns(pl.lit(-1).alias(ORIGINAL_ROW_ID_COLUMN))
    with pytest.raises(DataValidationError):
        _run_default(
            split=DatasetSplit(
                train=negative,
                validation=split.validation,
                test=split.test,
                summary=split.summary,
            )
        )

    dup = split.train.with_columns(
        pl.lit(split.train[ORIGINAL_ROW_ID_COLUMN][0]).alias(ORIGINAL_ROW_ID_COLUMN)
    )
    with pytest.raises(DataValidationError):
        _run_default(
            split=DatasetSplit(
                train=dup,
                validation=split.validation,
                test=split.test,
                summary=split.summary,
            )
        )


def test_original_row_id_train_validation_overlap() -> None:
    split = _regression_split()
    shared = int(split.train[ORIGINAL_ROW_ID_COLUMN][0])
    id_dtype = split.train.schema[ORIGINAL_ROW_ID_COLUMN]
    val_ids = split.validation[ORIGINAL_ROW_ID_COLUMN].to_list()
    val_ids[0] = shared
    val_overlap = split.validation.with_columns(
        pl.Series(ORIGINAL_ROW_ID_COLUMN, val_ids, dtype=id_dtype)
    )
    with pytest.raises(DataValidationError, match="Overlapping"):
        _run_default(
            split=DatasetSplit(
                train=split.train,
                validation=val_overlap,
                test=split.test,
                summary=split.summary,
            )
        )


def test_split_summary_mismatch_and_opt_out() -> None:
    split = _regression_split()
    bad_summary = split.summary.model_copy(
        update={
            "train_original_row_ids": list(
                reversed(split.summary.train_original_row_ids)
            )
        }
    )
    bad = DatasetSplit(
        train=split.train,
        validation=split.validation,
        test=split.test,
        summary=bad_summary,
    )
    with pytest.raises(DataValidationError, match="train"):
        _run_default(split=bad)

    val_mismatch = DatasetSplit(
        train=split.train,
        validation=split.validation,
        test=split.test,
        summary=split.summary.model_copy(
            update={
                "validation_original_row_ids": list(
                    reversed(split.summary.validation_original_row_ids)
                )
            }
        ),
    )
    with pytest.raises(DataValidationError, match="validation"):
        _run_default(split=val_mismatch)

    allow = ResidualAnomalyPipelinePolicy(require_split_summary_match=False)
    _run_default(split=bad, policy=allow)

    before = split.summary.model_dump()
    _run_default(split=split)
    assert split.summary.model_dump() == before


def test_test_ids_not_read_by_pipeline() -> None:
    baseline = _run_default()
    mutated = _regression_split(test_target_offset=9999.0, test_feature_offset=888.0)
    altered = _run_default(split=mutated)
    assert baseline.report.test_row_count == altered.report.test_row_count == 3
    assert (
        baseline.report.total_anomaly_count
        == altered.report.total_anomaly_count
    )
    assert math.isclose(
        baseline.report.threshold,
        altered.report.threshold,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


# --- Screening consistency (116-128) ---


def test_run_rejects_bad_screening_outcome_type() -> None:
    split = _regression_split()
    pipeline = ResidualAnomalyPipeline()
    with pytest.raises(TypeError):
        pipeline.run(
            split,
            "bad",  # type: ignore[arg-type]
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


@pytest.mark.parametrize(
    "override,message",
    [
        ({"task": AnalysisTask.CLASSIFICATION}, "REGRESSION"),
        ({"target_column": "bad"}, "target_column"),
        ({"feature_columns": ["f2", "f1"]}, "feature_columns"),
        ({"train_row_count": 99}, "train_row_count"),
        ({"validation_row_count": 99}, "validation_row_count"),
        ({"test_row_count": 99}, "test_row_count"),
        ({"selected_model_name": ""}, "selected_model_name"),
        ({"selected_estimator_key": "   "}, "selected_estimator_key"),
        ({"selected_metrics": {}}, "selected_metrics"),
        ({"selected_registry_rank": -1}, "selected_registry_rank"),
    ],
)
def test_screening_consistency_mismatches(
    override: dict[str, object],
    message: str,
) -> None:
    base = _valid_summary()
    payload = base.model_dump()
    payload.update(override)
    if "candidate_results" not in override:
        payload["candidate_results"] = base.candidate_results
    summary = ModelScreeningSummary.model_construct(**payload)  # type: ignore[arg-type]
    with pytest.raises(DataValidationError, match=message):
        _run_default(screening_outcome=_outcome_from_summary(summary))


def test_screening_consistency_false_skips_detail_but_rejects_non_regression() -> None:
    policy = ResidualAnomalyPipelinePolicy(require_screening_consistency=False)
    mismatched = _valid_summary().model_copy(update={"target_column": "other"})
    _run_default(
        screening_outcome=_outcome_from_summary(mismatched),
        policy=policy,
    )
    cls_summary = _valid_summary().model_copy(update={"task": AnalysisTask.CLASSIFICATION})
    with pytest.raises(DataValidationError, match="REGRESSION"):
        _run_default(
            screening_outcome=_outcome_from_summary(cls_summary),
            policy=policy,
        )


# --- Selected model (129-140) ---


def test_selected_model_must_be_fitted_with_predict_and_metadata() -> None:
    unfitted = _fitted_spy(is_fitted=False)
    with pytest.raises(ProcessIntelligenceError, match="fitted"):
        _run_default(screening_outcome=_outcome_from_summary(_valid_summary(), unfitted))

    class NoPredict(SpyRegressionModel):
        def predict(self, X: DataFrameLike) -> np.ndarray:  # type: ignore[override]
            raise RuntimeError("should not be called")

    broken = _fitted_spy()
    broken.predict = None  # type: ignore[assignment]
    with pytest.raises(ProcessIntelligenceError, match="predict"):
        _run_default(
            screening_outcome=_outcome_from_summary(_valid_summary(), broken)
        )


@pytest.mark.parametrize(
    "override,message",
    [
        ({"name": "Wrong Name"}, "metadata name"),
        ({"estimator_key": "wrong_key"}, "estimator_key"),
        ({"task": AnalysisTask.CLASSIFICATION}, "REGRESSION"),
        ({"metadata_features": ["f1"]}, "features"),
        ({"fit_row_count": 99}, "fit_row_count"),
        ({"feature_names": ["f2", "f1"]}, "feature_names"),
    ],
)
def test_selected_model_metadata_mismatches(
    override: dict[str, object],
    message: str,
) -> None:
    model = _fitted_spy(**override)  # type: ignore[arg-type]
    with pytest.raises(
        ProcessIntelligenceError,
        match="metadata|estimator_key|REGRESSION|fit_row_count|feature_names",
    ):
        _run_default(screening_outcome=_outcome_from_summary(_valid_summary(), model))


def test_selected_candidate_matching_paths() -> None:
    _run_default()

    missing = _valid_summary().model_copy(update={"selected_model_name": "Missing Model"})
    with pytest.raises(DataValidationError, match="No SUCCESS candidate"):
        _run_default(screening_outcome=_outcome_from_summary(missing))

    duplicate = _valid_summary(
        candidate_results=[
            _success_result(
                name="Dummy Regressor",
                estimator_key="dummy_regressor",
                registry_rank=0,
                metrics={"rmse": 2.0, "mae": 1.5, "r2": 0.0},
            ),
            _success_result(name="Linear Regression", registry_rank=1),
            _success_result(name="Linear Regression", registry_rank=1),
        ]
    )
    with pytest.raises(DataValidationError, match="Multiple SUCCESS"):
        _run_default(screening_outcome=_outcome_from_summary(duplicate))

    failed_candidate = CandidateScreeningResult(
        spec=_spec(name="Linear Regression"),
        status=CandidateRunStatus.FAILED,
        registry_rank=1,
        metrics={},
        primary_metric_name=None,
        primary_metric_value=None,
        fit_seconds=0.0,
        evaluation_seconds=0.0,
        total_seconds=0.0,
        warnings=["failed"],
        error_type="ValueError",
        error_message="fit failed",
    )
    failed_payload = _valid_summary().model_dump()
    failed_payload.update(
        {
            "candidate_results": [failed_candidate],
            "successful_model_names": [],
            "failed_model_names": ["Linear Regression"],
        }
    )
    failed_only = ModelScreeningSummary.model_construct(**failed_payload)
    with pytest.raises(DataValidationError, match="No SUCCESS candidate"):
        _run_default(screening_outcome=_outcome_from_summary(failed_only))


# --- Reserved columns (141-148) ---


@pytest.mark.parametrize("column", _RESULT_COLUMNS)
def test_reserved_column_collision(column: str) -> None:
    split = _regression_split()
    colliding = split.train.with_columns(pl.lit(1.0).alias(column))
    bad = DatasetSplit(
        train=colliding,
        validation=split.validation.with_columns(pl.lit(1.0).alias(column)),
        test=split.test.with_columns(pl.lit(1.0).alias(column)),
        summary=split.summary,
    )
    with pytest.raises(DataValidationError, match=column):
        _run_default(split=bad)


# --- Test partition unused (149-158) ---


def test_test_partition_unused_for_scoring_and_calibration() -> None:
    split = _regression_split()
    spy = _fitted_spy()
    with _detector_patch():
        _run_default(split=split, spy_model=spy)
    assert len(spy.predict_calls) == 2
    assert spy.fit_calls == 0
    assert len(SpyDetector.created) == 1
    detector = SpyDetector.created[0]
    assert len(detector.fit_calls) == 1
    y_fit, pred_fit = detector.fit_calls[0]
    assert len(y_fit) == split.validation.height
    assert len(pred_fit) == split.validation.height
    assert all(
        call[0] is not split.test.get_column(_TARGET)
        for call in detector.detect_calls
    )


def test_mutating_test_does_not_change_scored_outputs() -> None:
    base_split = _regression_split(test_target_offset=0.0)
    mutated_split = _regression_split(test_target_offset=5000.0, test_feature_offset=7000.0)
    a = _run_default(split=base_split, use_detector_spy=False)
    b = _run_default(split=mutated_split, use_detector_spy=False)
    assert a.validation_scored.equals(b.validation_scored)
    assert a.train_scored.equals(b.train_scored)
    assert a.report.test_row_count == b.report.test_row_count


# --- Prediction (159-169) ---


def test_validation_predict_once_with_feature_order() -> None:
    split = _regression_split()
    features = ["f2", "f1"]
    summary = _valid_summary(feature_columns=features)
    spy = _fitted_spy(
        feature_names=features,
        metadata_features=features,
    )
    token_before = spy._state_token
    with _detector_patch():
        _run_default(
            split=split,
            spy_model=spy,
            screening_outcome=_outcome_from_summary(summary, spy),
            feature_columns=features,
            leakage_report=_safe_report(features),
        )
    assert spy.fit_calls == 0
    assert spy._state_token is token_before
    assert len(spy.predict_calls) == 2
    val_call = spy.predict_calls[0]
    assert val_call == (features, split.validation.height)
    train_call = spy.predict_calls[1]
    assert train_call == (features, split.train.height)


@pytest.mark.parametrize(
    "predictions,message",
    [
        (np.array([1.0, 2.0]), "length"),
        (np.array([[1.0], [2.0], [3.0]]), "1-dimensional"),
        (np.array([1.0, float("nan"), 3.0]), "finite"),
        (np.array([1.0, float("inf"), 3.0]), "finite"),
    ],
)
def test_run_rejects_bad_validation_predictions(
    predictions: np.ndarray,
    message: str,
) -> None:
    split = _regression_split()
    spy = _fitted_spy(predictions=predictions)
    with pytest.raises(ProcessIntelligenceError, match=message):
        _run_default(split=split, spy_model=spy)


def test_train_predict_once_when_enabled_never_when_disabled() -> None:
    split = _regression_split()
    spy = _fitted_spy()
    with _detector_patch():
        _run_default(split=split, spy_model=spy)
    assert len(spy.predict_calls) == 2

    spy_disabled = _fitted_spy()
    policy = ResidualAnomalyPipelinePolicy(score_train_partition=False)
    with _detector_patch():
        _run_default(split=split, spy_model=spy_disabled, policy=policy)
    assert len(spy_disabled.predict_calls) == 1
    assert spy_disabled.predict_calls[0][1] == split.validation.height


# --- Detector calibration (170-180) ---


def test_detector_calibration_on_validation_only() -> None:
    split = _regression_split()
    with _detector_patch():
        outcome = _run_default(split=split)
    assert len(SpyDetector.created) == 1
    detector = SpyDetector.created[0]
    assert len(detector.fit_calls) == 1
    y_val, pred_val = detector.fit_calls[0]
    assert len(y_val) == split.validation.height
    assert len(pred_val) == split.validation.height
    calibration = outcome.residual_detector.calibration
    assert calibration is not None
    assert calibration.row_count == split.validation.height
    assert calibration.method is ResidualThresholdMethod.MAD
    assert math.isfinite(float(calibration.threshold))
    assert calibration.fitted_at.tzinfo is not None
    assert calibration.fitted_at.utcoffset() is not None


def test_new_detector_each_run() -> None:
    with _detector_patch():
        _run_default()
        _run_default()
    assert len(SpyDetector.created) == 2
    assert SpyDetector.created[0] is not SpyDetector.created[1]


# --- Validation scoring (181-188) ---


def test_validation_detect_once_reusing_prediction() -> None:
    split = _regression_split()
    with _detector_patch():
        _run_default(split=split)
    detector = SpyDetector.created[0]
    assert len(detector.detect_calls) == 2
    val_y, val_pred = detector.detect_calls[0]
    assert len(val_y) == split.validation.height
    assert len(val_pred) == split.validation.height


def test_validation_scored_immutability() -> None:
    split = _regression_split()
    before = split.validation.clone()
    outcome = _run_default(split=split)
    assert split.validation.equals(before)
    assert outcome.validation_scored.height == split.validation.height


# --- Train scoring (189-197) ---


def test_train_scoring_default_and_disable() -> None:
    split = _regression_split()
    with _detector_patch():
        enabled = _run_default(split=split)
    assert enabled.train_scored.height == split.train.height
    assert enabled.report.partition_summaries[0].scored is True

    with _detector_patch():
        disabled = _run_default(
            split=split,
            policy=ResidualAnomalyPipelinePolicy(score_train_partition=False),
        )
    assert disabled.train_scored.height == 0
    assert disabled.report.partition_summaries[0].scored is False
    assert _PARTITION_DISABLED_WARNING in disabled.report.partition_summaries[0].warnings
    assert _WARNING_TRAIN_SCORING_DISABLED in disabled.report.warnings
    assert len(SpyDetector.created[-1].detect_calls) == 1


def test_disabled_train_scored_preserves_schema() -> None:
    outcome = _run_default(
        policy=ResidualAnomalyPipelinePolicy(score_train_partition=False)
    )
    assert list(outcome.train_scored.columns) == list(outcome.validation_scored.columns)


# --- Result validation (198-209) ---


@pytest.mark.parametrize(
    "override,message",
    [
        ({"predictions": [1.0]}, "predictions"),
        ({"residuals": [float("nan"), 0.0, 0.0]}, "residuals"),
        ({"absolute_centered_residuals": [-0.1, 0.1, 0.1]}, "absolute_centered"),
        ({"scores": [float("inf"), 1.0, 1.0]}, "scores"),
        ({"raw_predictions": [0, 1, 1]}, "raw_predictions"),
        ({"is_anomaly": [True, False, False]}, "is_anomaly"),
        ({"anomaly_count": 99}, "anomaly_count"),
        ({"threshold": 999.0}, "threshold"),
    ],
)
def test_run_rejects_bad_detect_results(
    override: dict[str, Any],
    message: str,
) -> None:
    split = _regression_split()

    class BadDetector(SpyDetector):
        detect_call = 0

        def detect(self, y: Any, predictions: Any) -> ResidualAnomalyResult:
            self.detect_call += 1
            if self.detect_call == 1:
                return _bad_result(split, overrides=override)
            return super().detect(y, predictions)

    with patch(
        "process_intelligence.models.residual_anomaly_pipeline.ResidualAnomalyDetector",
        BadDetector,
    ):
        with pytest.raises(ProcessIntelligenceError, match=message):
            _run_default(split=split, use_detector_spy=False)


def test_run_rejects_non_result_detect_return() -> None:
    split = _regression_split()

    class WrongTypeDetector(SpyDetector):
        def detect(self, y: Any, predictions: Any) -> ResidualAnomalyResult:  # type: ignore[override]
            return {"bad": True}  # type: ignore[return-value]

    with patch(
        "process_intelligence.models.residual_anomaly_pipeline.ResidualAnomalyDetector",
        WrongTypeDetector,
    ):
        with pytest.raises(ProcessIntelligenceError, match="ResidualAnomalyResult"):
            _run_default(split=split, use_detector_spy=False)


# --- Scored DataFrame (210-227) ---


def test_scored_frame_columns_dtypes_and_partitions() -> None:
    split = _regression_split()
    outcome = _run_default(split=split)
    expected_cols = list(split.train.columns) + _RESULT_COLUMNS
    assert list(outcome.validation_scored.columns) == expected_cols
    assert outcome.validation_scored[_DATA_PARTITION_COLUMN].unique().to_list() == [
        "validation"
    ]
    assert outcome.train_scored[_DATA_PARTITION_COLUMN].unique().to_list() == ["train"]
    assert outcome.validation_scored[_REGRESSION_PREDICTION_COLUMN].dtype == pl.Float64
    assert outcome.validation_scored[_IS_RESIDUAL_ANOMALY_COLUMN].dtype == pl.Boolean
    assert outcome.validation_scored[_RESIDUAL_ANOMALY_RAW_PREDICTION_COLUMN].dtype == pl.Int64


def test_scored_frame_preserves_original_values_and_no_input_mutation() -> None:
    split = _regression_split()
    train_before = split.train.clone()
    val_before = split.validation.clone()
    outcome = _run_default(split=split)
    for col in ("f1", "f2", "y", ORIGINAL_ROW_ID_COLUMN):
        assert outcome.validation_scored.get_column(col).to_list() == (
            split.validation.get_column(col).to_list()
        )
    assert split.train.equals(train_before)
    assert split.validation.equals(val_before)
    assert outcome.validation_scored.height == split.validation.height
    assert outcome.validation_scored[ORIGINAL_ROW_ID_COLUMN].n_unique() == (
        split.validation.height
    )


# --- Combined (228-236) ---


def test_combined_concat_sort_and_disabled_train() -> None:
    split = _regression_split()
    outcome = _run_default(split=split)
    assert outcome.combined_scored.height == split.train.height + split.validation.height
    ids = outcome.combined_scored[ORIGINAL_ROW_ID_COLUMN].to_list()
    assert ids == sorted(ids)
    assert outcome.combined_scored[ORIGINAL_ROW_ID_COLUMN].n_unique() == len(ids)

    unsorted = _run_default(
        split=split,
        policy=ResidualAnomalyPipelinePolicy(sort_combined_by_original_row_id=False),
    )
    expected = (
        outcome.train_scored[ORIGINAL_ROW_ID_COLUMN].to_list()
        + outcome.validation_scored[ORIGINAL_ROW_ID_COLUMN].to_list()
    )
    assert unsorted.combined_scored[ORIGINAL_ROW_ID_COLUMN].to_list() == expected

    disabled = _run_default(
        split=split,
        policy=ResidualAnomalyPipelinePolicy(score_train_partition=False),
    )
    assert disabled.combined_scored.height == split.validation.height
    assert list(disabled.combined_scored.columns) == list(
        outcome.validation_scored.columns
    )


# --- Partition summary / report fields (237-273) ---


def test_report_partition_summaries_and_totals() -> None:
    outcome = _run_default()
    report = outcome.report
    assert [item.partition for item in report.partition_summaries] == [
        "train",
        "validation",
    ]
    train_summary = report.partition_summaries[0]
    val_summary = report.partition_summaries[1]
    assert train_summary.input_row_count == outcome.train_scored.height
    assert val_summary.input_row_count == outcome.validation_scored.height
    assert report.total_scored_row_count == (
        train_summary.prediction_count + val_summary.prediction_count
    )
    assert report.total_anomaly_count == (
        train_summary.anomaly_count + val_summary.anomaly_count
    )
    for summary in report.partition_summaries:
        if summary.scored:
            assert summary.prediction_seconds >= 0.0
            assert summary.scoring_seconds >= 0.0
            assert summary.total_seconds >= 0.0
            assert math.isfinite(summary.total_seconds)


def test_report_identity_and_calibration_fields() -> None:
    outcome = _run_default()
    report = outcome.report
    metadata = outcome.regression_model.get_metadata()
    assert report.task is AnalysisTask.REGRESSION
    assert report.model_name == metadata.model_name
    assert report.estimator_key == "linear_regression"
    assert report.feature_columns == list(_FEATURES)
    assert report.detector_method is outcome.residual_detector.calibration.method
    assert report.calibration_partition == "validation"
    assert report.calibration_row_count == outcome.validation_scored.height
    assert report.created_at.tzinfo is not None
    assert report.combined_sorted_by_original_row_id is True
    assert len(report.warnings) == len(set(report.warnings))


def test_report_fraction_shift_warning_when_large() -> None:
    split = _regression_split()

    class FractionShiftDetector(SpyDetector):
        detect_index = 0

        def detect(self, y: Any, predictions: Any) -> ResidualAnomalyResult:
            self.detect_index += 1
            base = super().detect(y, predictions)
            if self.detect_index == 1:
                return _forced_anomaly_result(base, anomaly_count=base.row_count)
            return _forced_anomaly_result(base, anomaly_count=0)

    policy = ResidualAnomalyPipelinePolicy(
        maximum_anomaly_fraction_shift=0.0,
        warn_on_fraction_shift=True,
    )
    with patch(
        "process_intelligence.models.residual_anomaly_pipeline.ResidualAnomalyDetector",
        FractionShiftDetector,
    ):
        outcome = _run_default(split=split, policy=policy, use_detector_spy=False)
    assert _WARNING_FRACTION_SHIFT in outcome.report.warnings


def test_validation_all_or_zero_anomaly_warnings() -> None:
    split = _regression_split()
    outcome = _run_default(split=split)
    assert _WARNING_VALIDATION_ZERO_ANOMALIES in outcome.report.warnings

    class AllValidationAnomaliesDetector(SpyDetector):
        detect_index = 0

        def detect(self, y: Any, predictions: Any) -> ResidualAnomalyResult:
            self.detect_index += 1
            base = super().detect(y, predictions)
            if self.detect_index == 1:
                return _forced_anomaly_result(base, anomaly_count=base.row_count)
            return base

    with patch(
        "process_intelligence.models.residual_anomaly_pipeline.ResidualAnomalyDetector",
        AllValidationAnomaliesDetector,
    ):
        outcome_all = _run_default(split=split, use_detector_spy=False)
    assert _WARNING_VALIDATION_ALL_ANOMALIES in outcome_all.report.warnings


# --- Outcome immutability / determinism (274-293) ---


def test_outcome_reuses_selected_model_only() -> None:
    split = _regression_split()
    spy = _fitted_spy()
    outcome = _run_default(split=split, spy_model=spy)
    assert outcome.regression_model is spy
    assert isinstance(outcome.residual_detector, ResidualAnomalyDetector)
    assert not hasattr(outcome.report, "estimator_object")


def test_calibration_property_independence() -> None:
    outcome = _run_default(use_detector_spy=False)
    cal_a = outcome.residual_detector.calibration
    cal_b = outcome.residual_detector.calibration
    assert cal_a is not cal_b
    assert cal_a is not None and cal_b is not None
    assert cal_a.threshold == cal_b.threshold == outcome.report.threshold


def test_inputs_not_mutated_across_runs() -> None:
    split = _regression_split()
    split_dump = split.summary.model_dump()
    outcome = _outcome_from_summary(_valid_summary())
    model = outcome.selected_model
    model_token = getattr(model, "_state_token", None)
    features = list(_FEATURES)
    config = ResidualAnomalyConfig()
    policy = ResidualAnomalyPipelinePolicy()
    config_dump = config.model_dump()
    policy_dump = policy.model_dump()
    candidate_dump = copy.deepcopy(outcome.summary.candidate_results)

    pipeline = ResidualAnomalyPipeline(detector_config=config, policy=policy)
    with _detector_patch():
        pipeline.run(
            split,
            outcome,
            target_column=_TARGET,
            feature_columns=features,
            leakage_report=_safe_report(),
        )
        pipeline.run(
            split,
            outcome,
            target_column=_TARGET,
            feature_columns=features,
            leakage_report=_safe_report(),
        )

    assert split.summary.model_dump() == split_dump
    assert config.model_dump() == config_dump
    assert policy.model_dump() == policy_dump
    assert outcome.summary.candidate_results == candidate_dump
    if isinstance(model, SpyRegressionModel):
        assert model._state_token is model_token


def test_deterministic_results_and_returned_df_mutation_isolation() -> None:
    split = _regression_split()
    spy = _fitted_spy()
    a = _run_default(split=split, spy_model=spy, use_detector_spy=False)
    b = _run_default(split=split, spy_model=spy, use_detector_spy=False)
    assert a.validation_scored.equals(b.validation_scored)
    assert math.isclose(a.report.threshold, b.report.threshold, rel_tol=0.0, abs_tol=1e-12)

    mutated = a.combined_scored.with_columns(pl.lit(99.0).alias("f1"))
    c = _run_default(split=split, spy_model=spy, use_detector_spy=False)
    assert c.combined_scored.equals(b.combined_scored)
    assert mutated.height == c.combined_scored.height


def test_no_accumulation_across_runs_on_spy_model() -> None:
    spy = _fitted_spy()
    _run_default(spy_model=spy)
    first_calls = len(spy.predict_calls)
    _run_default(spy_model=spy)
    assert len(spy.predict_calls) == first_calls + 2


# --- Integration ---


def test_integration_with_supervised_screener_and_default_registry() -> None:
    split = _regression_split()
    registry = create_default_supervised_model_registry(random_state=42)
    screener = SupervisedModelScreener(registry)
    screening = screener.screen(
        split,
        task=AnalysisTask.REGRESSION,
        target_column=_TARGET,
        feature_columns=_FEATURES,
        leakage_report=_safe_report(),
    )
    outcome = ResidualAnomalyPipeline().run(
        split,
        screening,
        target_column=_TARGET,
        feature_columns=_FEATURES,
        leakage_report=_safe_report(),
    )
    assert outcome.regression_model.is_fitted
    assert outcome.residual_detector.is_fitted
    assert outcome.validation_scored.height == split.validation.height
    assert outcome.combined_scored.height == split.train.height + split.validation.height
    assert outcome.report.task is AnalysisTask.REGRESSION
    assert outcome.report.model_name == screening.summary.selected_model_name
