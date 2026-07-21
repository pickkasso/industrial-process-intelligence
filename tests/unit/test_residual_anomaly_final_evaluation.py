"""Unit tests for residual anomaly final evaluation (Step 7H)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import FrozenInstanceError, fields, is_dataclass
from datetime import UTC, datetime
from typing import Any, Self
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
    ResidualAnomalyFinalEvaluationOutcome,
    ResidualAnomalyFinalEvaluationPolicy,
    ResidualAnomalyFinalEvaluationReport,
    ResidualAnomalyFinalEvaluator,
    ResidualAnomalyPipeline,
    ResidualAnomalyPipelineOutcome,
    ResidualAnomalyPipelineReport,
    ResidualAnomalyResult,
    ResidualPartitionSummary,
    ResidualThresholdMethod,
    SupervisedModelScreener,
    create_default_supervised_model_registry,
)
from process_intelligence.models.residual_anomaly_final_evaluation import (
    _ABSOLUTE_CENTERED_RESIDUAL_COLUMN,
    _DATA_PARTITION_COLUMN,
    _FLAG_DEGENERATE,
    _FLAG_FRACTION_ABOVE,
    _FLAG_FRACTION_BELOW,
    _FLAG_FRACTION_SHIFT,
    _FLAG_NEGATIVE_SEPARATION,
    _FLAG_SCORE_MEAN_SHIFT,
    _FLAG_SCORE_NEARLY_CONSTANT,
    _FLAG_SCORE_STD_CONTRACTED,
    _FLAG_SCORE_STD_EXPANDED,
    _IS_RESIDUAL_ANOMALY_COLUMN,
    _REGRESSION_PREDICTION_COLUMN,
    _REGRESSION_RESIDUAL_COLUMN,
    _RESERVED_RESULT_COLUMNS,
    _RESIDUAL_ANOMALY_RAW_PREDICTION_COLUMN,
    _RESIDUAL_ANOMALY_SCORE_COLUMN,
    _TEST_METRIC_KEYS,
    _VALIDATION_METRIC_KEYS,
    _WARNING_ALL_ANOMALIES,
    _WARNING_DISTRIBUTION_SHIFT,
    _WARNING_LEAKAGE,
    _WARNING_NO_ANOMALIES,
    _WARNING_QUALITY_FLAGS,
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
    test_target_offset: float = 0.0,
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
    name: str = "Linear Regression",
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
) -> CandidateScreeningResult:
    payload = metrics or {"rmse": 1.0, "mae": 0.5, "r2": 0.9}
    return CandidateScreeningResult(
        spec=_spec(name=name, estimator_key=estimator_key),
        status=CandidateRunStatus.SUCCESS,
        registry_rank=registry_rank,
        metrics=dict(payload),
        primary_metric_name="rmse",
        primary_metric_value=payload["rmse"],
        fit_seconds=0.01,
        evaluation_seconds=0.02,
        total_seconds=0.03,
        warnings=[],
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


def _default_prediction(X: DataFrameLike) -> np.ndarray:
    if isinstance(X, pl.DataFrame):
        return (
            X.get_column("f1").to_numpy() + X.get_column("f2").to_numpy()
        ).astype(float)
    frame = pl.DataFrame(X)
    return (
        frame.get_column("f1").to_numpy() + frame.get_column("f2").to_numpy()
    ).astype(float)


class SpyRegressionModel(BaseAnalysisModel):
    """Fitted regression stub that records predict/fit/evaluate calls."""

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
        no_predict: bool = False,
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
        self.evaluate_calls = 0
        self._predictions = predictions
        self._prediction_fn = prediction_fn or _default_prediction
        self._no_predict = no_predict

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
        if self._no_predict:
            raise AttributeError("predict disabled")
        if isinstance(X, pl.DataFrame):
            columns = list(X.columns)
            height = int(X.height)
        else:
            columns = [str(c) for c in X.columns]
            height = len(X)
        self.predict_calls.append((columns, height))
        if self._predictions is not None:
            return np.asarray(self._predictions)
        return self._prediction_fn(X)

    def evaluate(
        self,
        X: DataFrameLike,
        y: SeriesLike | None = None,
    ) -> ModelEvaluation:
        _ = (X, y)
        self.evaluate_calls += 1
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


class SpyDetector(ResidualAnomalyDetector):
    """Records fit/detect/score_samples/predict calls."""

    def __init__(self, config: ResidualAnomalyConfig | None = None) -> None:
        super().__init__(config=config)
        self.fit_calls: list[tuple[Any, Any]] = []
        self.detect_calls: list[tuple[Any, Any]] = []
        self.score_samples_calls = 0
        self.predict_calls = 0

    def fit(self, y: Any, predictions: Any) -> Self:
        self.fit_calls.append((y, predictions))
        return super().fit(y, predictions)

    def detect(self, y: Any, predictions: Any) -> ResidualAnomalyResult:
        self.detect_calls.append((y, predictions))
        return super().detect(y, predictions)

    def score_samples(self, y_true: Any, y_pred: Any) -> np.ndarray:
        self.score_samples_calls += 1
        return super().score_samples(y_true, y_pred)

    def predict(self, y_true: Any, y_pred: Any) -> np.ndarray:
        self.predict_calls += 1
        return super().predict(y_true, y_pred)


def _fitted_spy(**overrides: object) -> SpyRegressionModel:
    return SpyRegressionModel(**overrides)  # type: ignore[arg-type]


def _partition_summary(
    *,
    partition: str = "validation",
    input_row_count: int = 3,
    scored: bool = True,
    prediction_count: int | None = None,
    anomaly_count: int = 0,
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
        total_seconds=prediction_seconds + scoring_seconds,
        warnings=[] if warnings is None else warnings,
    )


def _valid_pipeline_report(**overrides: Any) -> ResidualAnomalyPipelineReport:
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


def _calibrate_spy_detector(
    model: SpyRegressionModel,
    split: DatasetSplit,
    *,
    config: ResidualAnomalyConfig | None = None,
) -> SpyDetector:
    detector = SpyDetector(config=config)
    x_val = split.validation.select(_FEATURES)
    y_val = split.validation.get_column(_TARGET)
    preds = model.predict(x_val)
    model.predict_calls.clear()
    detector.fit(y_val, preds)
    detector.fit_calls.clear()
    return detector


def _empty_scored_like(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.clear().with_columns(
        [
            pl.Series(_REGRESSION_PREDICTION_COLUMN, [], dtype=pl.Float64),
            pl.Series(_REGRESSION_RESIDUAL_COLUMN, [], dtype=pl.Float64),
            pl.Series(_ABSOLUTE_CENTERED_RESIDUAL_COLUMN, [], dtype=pl.Float64),
            pl.Series(_RESIDUAL_ANOMALY_SCORE_COLUMN, [], dtype=pl.Float64),
            pl.Series(_IS_RESIDUAL_ANOMALY_COLUMN, [], dtype=pl.Boolean),
            pl.Series(_RESIDUAL_ANOMALY_RAW_PREDICTION_COLUMN, [], dtype=pl.Int64),
            pl.Series(_DATA_PARTITION_COLUMN, [], dtype=pl.String),
        ]
    )


def _build_pipeline_outcome(
    *,
    split: DatasetSplit | None = None,
    model: SpyRegressionModel | None = None,
    detector: SpyDetector | None = None,
    report: ResidualAnomalyPipelineReport | None = None,
) -> tuple[DatasetSplit, ResidualAnomalyPipelineOutcome, SpyRegressionModel, SpyDetector]:
    used_split = _regression_split() if split is None else split
    used_model = model or _fitted_spy(fit_row_count=used_split.train.height)
    used_detector = detector or _calibrate_spy_detector(used_model, used_split)
    calibration = used_detector.calibration
    assert calibration is not None
    val_result = used_detector.detect(
        used_split.validation.get_column(_TARGET),
        used_model.predict(used_split.validation.select(_FEATURES)),
    )
    used_model.predict_calls.clear()
    used_detector.detect_calls.clear()
    used_detector.fit_calls.clear()

    train_summary = _partition_summary(
        partition="train",
        input_row_count=used_split.train.height,
        anomaly_count=0,
        anomaly_fraction=0.0,
        residual_mean=0.0,
        residual_std=0.1,
        score_min=0.0,
        score_max=0.1,
        score_mean=0.05,
        score_std=0.01,
    )
    validation_summary = _partition_summary(
        partition="validation",
        input_row_count=used_split.validation.height,
        anomaly_count=val_result.anomaly_count,
        anomaly_fraction=val_result.anomaly_fraction,
        residual_mean=val_result.residual_mean,
        residual_std=val_result.residual_std,
        score_min=val_result.score_min,
        score_max=val_result.score_max,
        score_mean=val_result.score_mean,
        score_std=val_result.score_std,
    )
    used_report = report or _valid_pipeline_report(
        train_row_count=used_split.train.height,
        validation_row_count=used_split.validation.height,
        test_row_count=used_split.test.height,
        calibration_row_count=used_split.validation.height,
        residual_center=float(calibration.residual_center),
        residual_scale=(
            None
            if calibration.residual_scale is None
            else float(calibration.residual_scale)
        ),
        threshold=float(calibration.threshold),
        detector_method=calibration.method,
        model_name=used_model.get_metadata().model_name,
        estimator_key=used_model._spec.estimator_key,
        partition_summaries=[train_summary, validation_summary],
        total_scored_row_count=(
            train_summary.prediction_count + validation_summary.prediction_count
        ),
        total_anomaly_count=(
            train_summary.anomaly_count + validation_summary.anomaly_count
        ),
        total_anomaly_fraction=(
            (train_summary.anomaly_count + validation_summary.anomaly_count)
            / (train_summary.prediction_count + validation_summary.prediction_count)
        ),
        train_validation_anomaly_fraction_shift=abs(
            validation_summary.anomaly_fraction - train_summary.anomaly_fraction
        ),
    )
    outcome = ResidualAnomalyPipelineOutcome(
        regression_model=used_model,
        residual_detector=used_detector,
        train_scored=_empty_scored_like(used_split.train),
        validation_scored=_empty_scored_like(used_split.validation),
        combined_scored=_empty_scored_like(used_split.train),
        report=used_report,
    )
    return used_split, outcome, used_model, used_detector


def _run_pipeline_integration(
    *,
    split: DatasetSplit | None = None,
    spy_model: SpyRegressionModel | None = None,
) -> tuple[DatasetSplit, ResidualAnomalyPipelineOutcome]:
    used_split = _regression_split() if split is None else split
    model = spy_model or _fitted_spy(fit_row_count=used_split.train.height)
    summary = _valid_summary(
        train_row_count=used_split.train.height,
        validation_row_count=used_split.validation.height,
        test_row_count=used_split.test.height,
    )
    screening = ModelScreeningOutcome(selected_model=model, summary=summary)
    pipeline = ResidualAnomalyPipeline()
    outcome = pipeline.run(
        used_split,
        screening,
        target_column=_TARGET,
        feature_columns=_FEATURES,
        leakage_report=_safe_report(),
    )
    return used_split, outcome


def _evaluate(
    *,
    split: DatasetSplit | None = None,
    pipeline_outcome: ResidualAnomalyPipelineOutcome | None = None,
    target_column: str = _TARGET,
    feature_columns: Sequence[str] | None = None,
    leakage_report: LeakageReport | None = None,
    policy: ResidualAnomalyFinalEvaluationPolicy | None = None,
) -> ResidualAnomalyFinalEvaluationOutcome:
    used_split, outcome, _, _ = _build_pipeline_outcome(split=split)
    if pipeline_outcome is not None:
        outcome = pipeline_outcome
        used_split = split or used_split
    features = list(feature_columns or _FEATURES)
    report = _safe_report(features) if leakage_report is None else leakage_report
    evaluator = ResidualAnomalyFinalEvaluator(policy=policy)
    return evaluator.evaluate(
        used_split,
        outcome,
        target_column=target_column,
        feature_columns=features,
        leakage_report=report,
    )


def _valid_final_report(**overrides: Any) -> ResidualAnomalyFinalEvaluationReport:
    validation_metrics = {
        "validation_anomaly_fraction": 0.0,
        "validation_residual_mean": 0.0,
        "validation_residual_std": 0.5,
        "validation_score_min": 0.1,
        "validation_score_max": 0.9,
        "validation_score_mean": 0.5,
        "validation_score_std": 0.2,
        "validation_score_range": 0.8,
    }
    test_metrics = {
        "test_anomaly_fraction": 0.1,
        "test_residual_mean": 0.1,
        "test_residual_std": 0.4,
        "test_score_min": 0.0,
        "test_score_max": 1.0,
        "test_score_mean": 0.4,
        "test_score_std": 0.3,
        "test_score_range": 1.0,
    }
    payload: dict[str, Any] = {
        "task": AnalysisTask.REGRESSION,
        "model_name": "Linear Regression",
        "estimator_key": "linear_regression",
        "target_column": _TARGET,
        "feature_columns": list(_FEATURES),
        "detector_method": ResidualThresholdMethod.MAD,
        "calibration_partition": "validation",
        "test_partition": "test",
        "train_row_count": 6,
        "validation_row_count": 3,
        "test_row_count": 3,
        "regression_fit_row_count": 6,
        "calibration_row_count": 3,
        "residual_center": 0.0,
        "residual_scale": 1.0,
        "threshold": 3.5,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_has_normal_and_anomaly": True,
        "test_score_separation": 1.5,
        "anomaly_fraction_shift": 0.1,
        "score_mean_shift": -0.1,
        "standardized_score_mean_shift": 0.5,
        "score_std_ratio": 1.5,
        "quality_flags": [],
        "prediction_seconds": 0.01,
        "scoring_seconds": 0.02,
        "total_seconds": 0.03,
        "evaluated_at": datetime.now(UTC),
        "warnings": [],
    }
    payload.update(overrides)
    return ResidualAnomalyFinalEvaluationReport(**payload)


# --- Policy (1-19) ---


def test_policy_defaults() -> None:
    policy = ResidualAnomalyFinalEvaluationPolicy()
    assert policy.require_safe_leakage_report is True
    assert policy.require_pipeline_consistency is True
    assert policy.require_split_summary_match is True
    assert policy.minimum_test_rows == 1
    assert policy.minimum_anomaly_fraction == pytest.approx(0.001)
    assert policy.maximum_anomaly_fraction == pytest.approx(0.25)
    assert policy.maximum_anomaly_fraction_shift == pytest.approx(0.10)
    assert policy.minimum_score_std == pytest.approx(1e-12)
    assert policy.maximum_standardized_score_mean_shift == pytest.approx(1.0)
    assert policy.minimum_score_std_ratio == pytest.approx(0.5)
    assert policy.maximum_score_std_ratio == pytest.approx(2.0)
    assert policy.warn_on_distribution_shift is True


@pytest.mark.parametrize(
    "field",
    [
        "require_safe_leakage_report",
        "require_pipeline_consistency",
        "require_split_summary_match",
        "warn_on_distribution_shift",
    ],
)
@pytest.mark.parametrize("bad", [0, 1, "true", "false", None])
def test_policy_rejects_non_bool(field: str, bad: object) -> None:
    with pytest.raises(ValidationError):
        ResidualAnomalyFinalEvaluationPolicy(**{field: bad})


def test_policy_rejects_minimum_test_rows_zero() -> None:
    with pytest.raises(ValidationError):
        ResidualAnomalyFinalEvaluationPolicy(minimum_test_rows=0)


@pytest.mark.parametrize("bad", [True, False])
def test_policy_rejects_minimum_test_rows_bool(bad: object) -> None:
    with pytest.raises(ValidationError):
        ResidualAnomalyFinalEvaluationPolicy(minimum_test_rows=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field",
    [
        "minimum_anomaly_fraction",
        "maximum_anomaly_fraction",
        "maximum_anomaly_fraction_shift",
    ],
)
@pytest.mark.parametrize("bad", [-0.1, 1.1, True, float("nan"), float("inf")])
def test_policy_rejects_bad_fractions(field: str, bad: object) -> None:
    with pytest.raises(ValidationError):
        ResidualAnomalyFinalEvaluationPolicy(**{field: bad})


def test_policy_rejects_minimum_fraction_ge_maximum() -> None:
    with pytest.raises(ValidationError):
        ResidualAnomalyFinalEvaluationPolicy(
            minimum_anomaly_fraction=0.2,
            maximum_anomaly_fraction=0.2,
        )


@pytest.mark.parametrize("bad", [0.0, True, float("nan")])
def test_policy_rejects_bad_minimum_score_std(bad: object) -> None:
    with pytest.raises(ValidationError):
        ResidualAnomalyFinalEvaluationPolicy(minimum_score_std=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0.0, True, float("nan")])
def test_policy_rejects_bad_standardized_mean_shift(bad: object) -> None:
    with pytest.raises(ValidationError):
        ResidualAnomalyFinalEvaluationPolicy(
            maximum_standardized_score_mean_shift=bad  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("bad", [-0.1, True])
def test_policy_rejects_bad_std_ratio_bounds(bad: object) -> None:
    with pytest.raises(ValidationError):
        ResidualAnomalyFinalEvaluationPolicy(minimum_score_std_ratio=bad)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ResidualAnomalyFinalEvaluationPolicy(maximum_score_std_ratio=bad)  # type: ignore[arg-type]


def test_policy_rejects_minimum_std_ratio_ge_maximum() -> None:
    with pytest.raises(ValidationError):
        ResidualAnomalyFinalEvaluationPolicy(
            minimum_score_std_ratio=2.0,
            maximum_score_std_ratio=2.0,
        )


def test_policy_round_trip() -> None:
    policy = ResidualAnomalyFinalEvaluationPolicy(
        require_safe_leakage_report=False,
        minimum_test_rows=2,
        maximum_anomaly_fraction=0.4,
    )
    restored = ResidualAnomalyFinalEvaluationPolicy.model_validate(
        policy.model_dump()
    )
    assert restored == policy


# --- Report (20-61) ---


def test_report_valid() -> None:
    report = _valid_final_report()
    assert report.task is AnalysisTask.REGRESSION
    assert list(report.validation_metrics.keys()) == list(_VALIDATION_METRIC_KEYS)
    assert list(report.test_metrics.keys()) == list(_TEST_METRIC_KEYS)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"task": AnalysisTask.CLASSIFICATION},
        {"model_name": " "},
        {"estimator_key": ""},
        {"target_column": "  "},
        {"feature_columns": []},
        {"feature_columns": ["f1", "f1"]},
        {"feature_columns": [ORIGINAL_ROW_ID_COLUMN]},
        {"feature_columns": ["f1", "y"], "target_column": "y"},
        {"calibration_partition": "train"},
        {"test_partition": "validation"},
        {"train_row_count": 0},
        {"validation_row_count": 0},
        {"test_row_count": 0},
        {"regression_fit_row_count": 5},
        {"calibration_row_count": 2},
        {"residual_center": float("nan")},
        {"residual_scale": 0.0},
        {"threshold": -0.1},
        {"validation_metrics": {}},
        {"test_metrics": {}},
        {"anomaly_fraction_shift": -0.1},
        {"standardized_score_mean_shift": -0.1},
        {"score_std_ratio": -0.1},
        {"quality_flags": ["a", "a"]},
        {"prediction_seconds": -0.1},
        {"prediction_seconds": float("nan")},
        {"total_seconds": 1.0},
        {"evaluated_at": datetime(2024, 1, 1)},
        {"warnings": ["w", "w"]},
    ],
)
def test_report_rejects_invalid(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _valid_final_report(**kwargs)


def test_report_rejects_missing_validation_key() -> None:
    metrics = dict(_valid_final_report().validation_metrics)
    del metrics["validation_score_range"]
    with pytest.raises(ValidationError):
        _valid_final_report(validation_metrics=metrics)


def test_report_rejects_missing_test_key() -> None:
    metrics = dict(_valid_final_report().test_metrics)
    del metrics["test_score_range"]
    with pytest.raises(ValidationError):
        _valid_final_report(test_metrics=metrics)


def test_report_rejects_empty_metric_key() -> None:
    metrics = dict(_valid_final_report().test_metrics)
    metrics[""] = 1.0
    with pytest.raises(ValidationError):
        _valid_final_report(test_metrics=metrics)


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf")])
def test_report_rejects_bad_metric_values(bad: object) -> None:
    metrics = dict(_valid_final_report().test_metrics)
    metrics["test_score_mean"] = bad  # type: ignore[assignment]
    with pytest.raises(ValidationError):
        _valid_final_report(test_metrics=metrics)


def test_report_rejects_score_range_mismatch() -> None:
    metrics = dict(_valid_final_report().test_metrics)
    metrics["test_score_range"] = 999.0
    with pytest.raises(ValidationError):
        _valid_final_report(test_metrics=metrics)


def test_report_rejects_anomaly_fraction_out_of_range() -> None:
    metrics = dict(_valid_final_report().test_metrics)
    metrics["test_anomaly_fraction"] = 1.5
    with pytest.raises(ValidationError):
        _valid_final_report(test_metrics=metrics)


def test_report_rejects_negative_std() -> None:
    metrics = dict(_valid_final_report().test_metrics)
    metrics["test_score_std"] = -0.1
    with pytest.raises(ValidationError):
        _valid_final_report(test_metrics=metrics)


def test_report_rejects_missing_separation_when_groups() -> None:
    with pytest.raises(ValidationError):
        _valid_final_report(
            test_has_normal_and_anomaly=True,
            test_score_separation=None,
        )


def test_report_mutable_defaults_independent() -> None:
    a = _valid_final_report()
    b = _valid_final_report()
    a.quality_flags.append("x")
    assert b.quality_flags == []


def test_report_round_trip() -> None:
    report = _valid_final_report(warnings=["warn"])
    restored = ResidualAnomalyFinalEvaluationReport.model_validate(
        report.model_dump()
    )
    assert restored == report


# --- Outcome (62-67) ---


def test_outcome_frozen_dataclass() -> None:
    split, pipeline_outcome, model, detector = _build_pipeline_outcome()
    outcome = _evaluate(split=split, pipeline_outcome=pipeline_outcome)
    assert is_dataclass(outcome)
    assert {f.name for f in fields(outcome)} == {
        "regression_model",
        "residual_detector",
        "test_scored",
        "report",
    }
    with pytest.raises(FrozenInstanceError):
        outcome.report = outcome.report  # type: ignore[misc]
    assert isinstance(outcome.regression_model, BaseAnalysisModel)
    assert isinstance(outcome.residual_detector, ResidualAnomalyDetector)
    assert isinstance(outcome.test_scored, pl.DataFrame)
    assert isinstance(outcome.report, ResidualAnomalyFinalEvaluationReport)
    assert outcome.regression_model is model
    assert outcome.residual_detector is detector


# --- Evaluator construction (68-72) ---


def test_evaluator_defaults() -> None:
    evaluator = ResidualAnomalyFinalEvaluator()
    assert isinstance(evaluator._policy, ResidualAnomalyFinalEvaluationPolicy)


def test_evaluator_custom_policy() -> None:
    policy = ResidualAnomalyFinalEvaluationPolicy(minimum_test_rows=2)
    evaluator = ResidualAnomalyFinalEvaluator(policy=policy)
    assert evaluator._policy.minimum_test_rows == 2


def test_evaluator_rejects_bad_policy_type() -> None:
    with pytest.raises(TypeError):
        ResidualAnomalyFinalEvaluator(policy={"minimum_test_rows": 1})  # type: ignore[arg-type]


def test_evaluator_policy_isolation() -> None:
    policy = ResidualAnomalyFinalEvaluationPolicy(minimum_test_rows=2)
    evaluator = ResidualAnomalyFinalEvaluator(policy=policy)
    mutated = policy.model_copy(update={"minimum_test_rows": 9})
    assert evaluator._policy.minimum_test_rows == 2
    assert mutated.minimum_test_rows == 9
    a = ResidualAnomalyFinalEvaluator(
        policy=ResidualAnomalyFinalEvaluationPolicy(minimum_test_rows=3)
    )
    b = ResidualAnomalyFinalEvaluator(
        policy=ResidualAnomalyFinalEvaluationPolicy(minimum_test_rows=4)
    )
    assert a._policy.minimum_test_rows != b._policy.minimum_test_rows


# --- Input validation (73-98) ---


def test_evaluate_rejects_bad_split_type() -> None:
    _, outcome, _, _ = _build_pipeline_outcome()
    with pytest.raises(TypeError):
        ResidualAnomalyFinalEvaluator().evaluate(
            "bad",  # type: ignore[arg-type]
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_evaluate_rejects_bad_pipeline_outcome_type() -> None:
    split = _regression_split()
    with pytest.raises(TypeError):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            "bad",  # type: ignore[arg-type]
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_evaluate_rejects_non_polars_partition() -> None:
    split = _regression_split()
    bad = DatasetSplit(
        train=split.train,
        validation=split.validation,
        test="bad",  # type: ignore[arg-type]
        summary=split.summary,
    )
    _, outcome, _, _ = _build_pipeline_outcome(split=split)
    with pytest.raises(TypeError):
        ResidualAnomalyFinalEvaluator().evaluate(
            bad,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_evaluate_rejects_schema_order_mismatch() -> None:
    split = _regression_split()
    bad_test = split.test.select(["f1", ORIGINAL_ROW_ID_COLUMN, "f2", "y"])
    bad = DatasetSplit(
        train=split.train,
        validation=split.validation,
        test=bad_test,
        summary=split.summary,
    )
    _, outcome, _, _ = _build_pipeline_outcome(split=split)
    with pytest.raises(DataValidationError):
        ResidualAnomalyFinalEvaluator().evaluate(
            bad,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_evaluate_rejects_dtype_mismatch() -> None:
    split = _regression_split()
    bad_test = split.test.with_columns(pl.col("f1").cast(pl.Int64))
    bad = DatasetSplit(
        train=split.train,
        validation=split.validation,
        test=bad_test,
        summary=split.summary,
    )
    _, outcome, _, _ = _build_pipeline_outcome(split=split)
    with pytest.raises(DataValidationError):
        ResidualAnomalyFinalEvaluator().evaluate(
            bad,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_evaluate_rejects_missing_original_row_id() -> None:
    split = _regression_split()
    bad_test = split.test.drop(ORIGINAL_ROW_ID_COLUMN)
    # schemas must match - drop from all
    bad = DatasetSplit(
        train=split.train.drop(ORIGINAL_ROW_ID_COLUMN),
        validation=split.validation.drop(ORIGINAL_ROW_ID_COLUMN),
        test=bad_test,
        summary=split.summary,
    )
    _, outcome, _, _ = _build_pipeline_outcome(split=split)
    with pytest.raises(DataValidationError):
        ResidualAnomalyFinalEvaluator().evaluate(
            bad,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_evaluate_rejects_empty_train() -> None:
    split = _regression_split()
    empty_train = split.train.clear()
    bad = _manual_split(empty_train, split.validation, split.test)
    model = _fitted_spy(fit_row_count=0)
    _, outcome, _, _ = _build_pipeline_outcome(split=split, model=model)
    with pytest.raises(InsufficientDataError):
        ResidualAnomalyFinalEvaluator().evaluate(
            bad,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_evaluate_rejects_empty_validation() -> None:
    split = _regression_split()
    empty_val = split.validation.clear()
    bad = _manual_split(split.train, empty_val, split.test)
    _, outcome, _, _ = _build_pipeline_outcome(split=split)
    with pytest.raises(InsufficientDataError):
        ResidualAnomalyFinalEvaluator().evaluate(
            bad,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_evaluate_rejects_insufficient_test_rows() -> None:
    split = _regression_split()
    _, outcome, _, _ = _build_pipeline_outcome(split=split)
    with pytest.raises(InsufficientDataError):
        ResidualAnomalyFinalEvaluator(
            policy=ResidualAnomalyFinalEvaluationPolicy(minimum_test_rows=10)
        ).evaluate(
            split,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


@pytest.mark.parametrize("bad", [None, 1, b"y"])
def test_evaluate_rejects_bad_target(bad: object) -> None:
    split, outcome, _, _ = _build_pipeline_outcome()
    with pytest.raises((TypeError, DataValidationError)):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            outcome,
            target_column=bad,  # type: ignore[arg-type]
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_evaluate_rejects_blank_target() -> None:
    split, outcome, _, _ = _build_pipeline_outcome()
    with pytest.raises(DataValidationError):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            outcome,
            target_column="  ",
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_evaluate_rejects_original_row_id_target() -> None:
    split, outcome, _, _ = _build_pipeline_outcome()
    with pytest.raises(DataValidationError):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            outcome,
            target_column=ORIGINAL_ROW_ID_COLUMN,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_evaluate_rejects_missing_target() -> None:
    split, outcome, _, _ = _build_pipeline_outcome()
    with pytest.raises(DataValidationError):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            outcome,
            target_column="missing",
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


@pytest.mark.parametrize("bad", ["f1", b"f1"])
def test_evaluate_rejects_str_or_bytes_features(bad: object) -> None:
    split, outcome, _, _ = _build_pipeline_outcome()
    with pytest.raises(TypeError):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            outcome,
            target_column=_TARGET,
            feature_columns=bad,  # type: ignore[arg-type]
            leakage_report=_safe_report(),
        )


def test_evaluate_rejects_non_str_feature_element() -> None:
    split, outcome, _, _ = _build_pipeline_outcome()
    with pytest.raises(TypeError):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            outcome,
            target_column=_TARGET,
            feature_columns=["f1", 2],  # type: ignore[list-item]
            leakage_report=_safe_report(),
        )


@pytest.mark.parametrize(
    "features",
    [
        [" "],
        [],
        ["f1", "f1"],
        [ORIGINAL_ROW_ID_COLUMN],
        ["f1", "y"],
        [_REGRESSION_PREDICTION_COLUMN],
        ["missing"],
    ],
)
def test_evaluate_rejects_bad_features(features: list[str]) -> None:
    split, outcome, _, _ = _build_pipeline_outcome()
    with pytest.raises(DataValidationError):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            outcome,
            target_column=_TARGET,
            feature_columns=features,
            leakage_report=_safe_report(),
        )


def test_evaluate_rejects_bad_leakage_type() -> None:
    split, outcome, _, _ = _build_pipeline_outcome()
    with pytest.raises(TypeError):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report="bad",  # type: ignore[arg-type]
        )


def test_evaluate_rejects_checked_feature_order_mismatch() -> None:
    split, outcome, _, _ = _build_pipeline_outcome()
    with pytest.raises(DataValidationError):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(["f2", "f1"]),
        )


def test_evaluate_rejects_checked_feature_set_mismatch() -> None:
    split, outcome, _, _ = _build_pipeline_outcome()
    with pytest.raises(DataValidationError):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(["f1", "other"]),
        )


# --- Leakage gate (99-103) ---


def test_blocker_report_raises() -> None:
    split, outcome, _, _ = _build_pipeline_outcome()
    with pytest.raises(DataLeakageError, match="BLOCKER|blocker|TARGET"):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_blocker_report(),
        )


def test_warning_only_report_allowed() -> None:
    split, outcome, _, _ = _build_pipeline_outcome()
    result = ResidualAnomalyFinalEvaluator().evaluate(
        split,
        outcome,
        target_column=_TARGET,
        feature_columns=_FEATURES,
        leakage_report=_warning_report(),
    )
    assert _WARNING_LEAKAGE in result.report.warnings


def test_safe_report_allowed() -> None:
    outcome = _evaluate()
    assert isinstance(outcome.report, ResidualAnomalyFinalEvaluationReport)


def test_require_safe_false_allows_blocker() -> None:
    split, outcome, _, _ = _build_pipeline_outcome()
    result = ResidualAnomalyFinalEvaluator(
        policy=ResidualAnomalyFinalEvaluationPolicy(
            require_safe_leakage_report=False
        )
    ).evaluate(
        split,
        outcome,
        target_column=_TARGET,
        feature_columns=_FEATURES,
        leakage_report=_blocker_report(),
    )
    assert result.report.test_row_count == split.test.height


def test_leakage_report_immutability() -> None:
    split, outcome, _, _ = _build_pipeline_outcome()
    report = _warning_report()
    before = report.model_dump()
    ResidualAnomalyFinalEvaluator().evaluate(
        split,
        outcome,
        target_column=_TARGET,
        feature_columns=_FEATURES,
        leakage_report=report,
    )
    assert report.model_dump() == before


# --- Reserved columns (104-111) ---


@pytest.mark.parametrize("column", list(_RESERVED_RESULT_COLUMNS))
def test_reserved_column_conflict(column: str) -> None:
    split = _regression_split()
    frames = []
    for frame in (split.train, split.validation, split.test):
        frames.append(frame.with_columns(pl.lit(0.0).alias(column)))
    bad = _manual_split(frames[0], frames[1], frames[2])
    _, outcome, _, _ = _build_pipeline_outcome(split=split)
    with pytest.raises(DataValidationError, match=column):
        ResidualAnomalyFinalEvaluator().evaluate(
            bad,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


# --- original ID / SplitSummary (112-123) ---


def test_rejects_null_original_id() -> None:
    split = _regression_split()
    bad_test = split.test.with_columns(
        pl.Series(ORIGINAL_ROW_ID_COLUMN, [9, None, 11], dtype=pl.Int64)
    )
    bad_train = split.train
    bad_val = split.validation
    # dtypes must match - keep Int64 with null
    bad = DatasetSplit(
        train=bad_train,
        validation=bad_val,
        test=bad_test,
        summary=split.summary,
    )
    _, outcome, _, _ = _build_pipeline_outcome(split=split)
    with pytest.raises(DataValidationError):
        ResidualAnomalyFinalEvaluator().evaluate(
            bad,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_rejects_string_original_id() -> None:
    split = _regression_split()
    frames = []
    for frame in (split.train, split.validation, split.test):
        frames.append(
            frame.with_columns(
                pl.col(ORIGINAL_ROW_ID_COLUMN).cast(pl.Utf8)
            )
        )
    bad = _manual_split(frames[0], frames[1], frames[2])
    _, outcome, _, _ = _build_pipeline_outcome(split=split)
    with pytest.raises(DataValidationError):
        ResidualAnomalyFinalEvaluator().evaluate(
            bad,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_rejects_boolean_original_id() -> None:
    split = _regression_split()
    frames = []
    for frame in (split.train, split.validation, split.test):
        frames.append(
            frame.with_columns(
                pl.Series(
                    ORIGINAL_ROW_ID_COLUMN,
                    [False] * frame.height,
                    dtype=pl.Boolean,
                )
            )
        )
    bad = DatasetSplit(
        train=frames[0],
        validation=frames[1],
        test=frames[2],
        summary=split.summary,
    )
    _, outcome, _, _ = _build_pipeline_outcome(split=split)
    with pytest.raises(DataValidationError):
        ResidualAnomalyFinalEvaluator().evaluate(
            bad,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_rejects_negative_original_id() -> None:
    split = _regression_split()
    bad_test = split.test.with_columns(
        pl.Series(ORIGINAL_ROW_ID_COLUMN, [-1, 10, 11], dtype=pl.Int64)
    )
    bad = DatasetSplit(
        train=split.train,
        validation=split.validation,
        test=bad_test,
        summary=split.summary.model_copy(
            update={"test_original_row_ids": [-1, 10, 11]}
        ),
    )
    _, outcome, _, _ = _build_pipeline_outcome(split=split)
    with pytest.raises(DataValidationError):
        ResidualAnomalyFinalEvaluator().evaluate(
            bad,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_rejects_duplicate_original_id_within_partition() -> None:
    split = _regression_split()
    bad_test = split.test.with_columns(
        pl.Series(ORIGINAL_ROW_ID_COLUMN, [9, 9, 11], dtype=pl.Int64)
    )
    bad = DatasetSplit(
        train=split.train,
        validation=split.validation,
        test=bad_test,
        summary=split.summary.model_copy(
            update={"test_original_row_ids": [9, 9, 11]}
        ),
    )
    _, outcome, _, _ = _build_pipeline_outcome(split=split)
    with pytest.raises(DataValidationError):
        ResidualAnomalyFinalEvaluator().evaluate(
            bad,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda s: DatasetSplit(
                train=s.train,
                validation=s.validation.with_columns(
                    pl.Series(ORIGINAL_ROW_ID_COLUMN, [0, 7, 8], dtype=pl.Int64)
                ),
                test=s.test,
                summary=s.summary.model_copy(
                    update={"validation_original_row_ids": [0, 7, 8]}
                ),
            ),
            "train and validation",
        ),
        (
            lambda s: DatasetSplit(
                train=s.train,
                validation=s.validation,
                test=s.test.with_columns(
                    pl.Series(ORIGINAL_ROW_ID_COLUMN, [0, 10, 11], dtype=pl.Int64)
                ),
                summary=s.summary.model_copy(
                    update={"test_original_row_ids": [0, 10, 11]}
                ),
            ),
            "train and test",
        ),
        (
            lambda s: DatasetSplit(
                train=s.train,
                validation=s.validation,
                test=s.test.with_columns(
                    pl.Series(ORIGINAL_ROW_ID_COLUMN, [6, 10, 11], dtype=pl.Int64)
                ),
                summary=s.summary.model_copy(
                    update={"test_original_row_ids": [6, 10, 11]}
                ),
            ),
            "validation and test",
        ),
    ],
)
def test_rejects_overlapping_ids(
    mutate: Callable[[DatasetSplit], DatasetSplit],
    match: str,
) -> None:
    split = _regression_split()
    bad = mutate(split)
    _, outcome, _, _ = _build_pipeline_outcome(split=split)
    with pytest.raises(DataValidationError, match=match):
        ResidualAnomalyFinalEvaluator().evaluate(
            bad,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


@pytest.mark.parametrize(
    ("field", "ids", "partition"),
    [
        ("train_original_row_ids", [5, 4, 3, 2, 1, 0], "train"),
        ("validation_original_row_ids", [8, 7, 6], "validation"),
        ("test_original_row_ids", [11, 10, 9], "test"),
    ],
)
def test_rejects_summary_id_mismatch(
    field: str,
    ids: list[int],
    partition: str,
) -> None:
    split = _regression_split()
    bad = DatasetSplit(
        train=split.train,
        validation=split.validation,
        test=split.test,
        summary=split.summary.model_copy(update={field: ids}),
    )
    _, outcome, _, _ = _build_pipeline_outcome(split=split)
    with pytest.raises(DataValidationError, match=partition):
        ResidualAnomalyFinalEvaluator().evaluate(
            bad,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_summary_match_false_allows_mismatch() -> None:
    split = _regression_split()
    bad = DatasetSplit(
        train=split.train,
        validation=split.validation,
        test=split.test,
        summary=split.summary.model_copy(
            update={"test_original_row_ids": [11, 10, 9]}
        ),
    )
    _, outcome, _, _ = _build_pipeline_outcome(split=split)
    result = ResidualAnomalyFinalEvaluator(
        policy=ResidualAnomalyFinalEvaluationPolicy(
            require_split_summary_match=False
        )
    ).evaluate(
        bad,
        outcome,
        target_column=_TARGET,
        feature_columns=_FEATURES,
        leakage_report=_safe_report(),
    )
    assert result.report.test_row_count == 3


# --- Pipeline consistency (124-152) ---


def test_normal_pipeline_outcome_allowed() -> None:
    result = _evaluate()
    assert result.report.model_name == "Linear Regression"


def test_rejects_report_task_mismatch() -> None:
    split, outcome, model, detector = _build_pipeline_outcome()
    bad_report = outcome.report.model_copy(deep=True)
    object.__setattr__(
        ResidualAnomalyPipelineOutcome(
            regression_model=model,
            residual_detector=detector,
            train_scored=outcome.train_scored,
            validation_scored=outcome.validation_scored,
            combined_scored=outcome.combined_scored,
            report=bad_report.model_construct(
                **{
                    **bad_report.model_dump(),
                    "task": AnalysisTask.CLASSIFICATION,
                }
            ),
        ),
        "report",
        bad_report.model_construct(
            **{**bad_report.model_dump(), "task": AnalysisTask.CLASSIFICATION}
        ),
    )
    bad_outcome = ResidualAnomalyPipelineOutcome(
        regression_model=model,
        residual_detector=detector,
        train_scored=outcome.train_scored,
        validation_scored=outcome.validation_scored,
        combined_scored=outcome.combined_scored,
        report=ResidualAnomalyPipelineReport.model_construct(
            **{**outcome.report.model_dump(), "task": AnalysisTask.CLASSIFICATION}
        ),
    )
    with pytest.raises((DataValidationError, ValidationError, ProcessIntelligenceError)):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            bad_outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_rejects_target_mismatch() -> None:
    split, outcome, model, detector = _build_pipeline_outcome()
    bad_outcome = ResidualAnomalyPipelineOutcome(
        regression_model=model,
        residual_detector=detector,
        train_scored=outcome.train_scored,
        validation_scored=outcome.validation_scored,
        combined_scored=outcome.combined_scored,
        report=outcome.report.model_copy(update={"target_column": "other"}),
    )
    with pytest.raises(DataValidationError, match="target_column"):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            bad_outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_rejects_feature_order_mismatch() -> None:
    split, outcome, model, detector = _build_pipeline_outcome()
    bad_outcome = ResidualAnomalyPipelineOutcome(
        regression_model=model,
        residual_detector=detector,
        train_scored=outcome.train_scored,
        validation_scored=outcome.validation_scored,
        combined_scored=outcome.combined_scored,
        report=outcome.report.model_copy(
            update={"feature_columns": ["f2", "f1"]}
        ),
    )
    with pytest.raises(DataValidationError, match="feature_columns"):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            bad_outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("train_row_count", 99),
        ("validation_row_count", 99),
        ("test_row_count", 99),
        ("calibration_row_count", 99),
    ],
)
def test_rejects_row_count_mismatch(field: str, value: int) -> None:
    split, outcome, model, detector = _build_pipeline_outcome()
    bad_outcome = ResidualAnomalyPipelineOutcome(
        regression_model=model,
        residual_detector=detector,
        train_scored=outcome.train_scored,
        validation_scored=outcome.validation_scored,
        combined_scored=outcome.combined_scored,
        report=outcome.report.model_copy(update={field: value}),
    )
    with pytest.raises(DataValidationError):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            bad_outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_rejects_calibration_partition_mismatch() -> None:
    split, outcome, model, detector = _build_pipeline_outcome()
    bad_report = ResidualAnomalyPipelineReport.model_construct(
        **{
            **outcome.report.model_dump(),
            "calibration_partition": "train",
        }
    )
    bad_outcome = ResidualAnomalyPipelineOutcome(
        regression_model=model,
        residual_detector=detector,
        train_scored=outcome.train_scored,
        validation_scored=outcome.validation_scored,
        combined_scored=outcome.combined_scored,
        report=bad_report,
    )
    with pytest.raises(DataValidationError, match="calibration_partition"):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            bad_outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_name", " "),
        ("estimator_key", ""),
    ],
)
def test_rejects_blank_model_identity(field: str, value: str) -> None:
    split, outcome, model, detector = _build_pipeline_outcome()
    bad_report = ResidualAnomalyPipelineReport.model_construct(
        **{**outcome.report.model_dump(), field: value}
    )
    bad_outcome = ResidualAnomalyPipelineOutcome(
        regression_model=model,
        residual_detector=detector,
        train_scored=outcome.train_scored,
        validation_scored=outcome.validation_scored,
        combined_scored=outcome.combined_scored,
        report=bad_report,
    )
    with pytest.raises(DataValidationError):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            bad_outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_rejects_unfitted_model() -> None:
    split = _regression_split()
    model = _fitted_spy(is_fitted=False, fit_row_count=split.train.height)
    _, outcome, _, _ = _build_pipeline_outcome(split=split, model=model)
    with pytest.raises(ProcessIntelligenceError, match="fitted"):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_rejects_model_without_predict() -> None:
    split, outcome, model, detector = _build_pipeline_outcome()
    model.predict = "not-callable"  # type: ignore[assignment]
    with pytest.raises(ProcessIntelligenceError, match="predict"):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_rejects_metadata_name_mismatch() -> None:
    split, outcome, model, detector = _build_pipeline_outcome()
    model._name = "Other"
    with pytest.raises(ProcessIntelligenceError, match="name"):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_rejects_metadata_estimator_key_mismatch() -> None:
    split, outcome, model, detector = _build_pipeline_outcome()
    model._spec = _spec(estimator_key="ridge_regression")
    with pytest.raises(ProcessIntelligenceError, match="estimator_key"):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_rejects_metadata_task_mismatch() -> None:
    split = _regression_split()
    model = _fitted_spy(
        task=AnalysisTask.CLASSIFICATION,
        fit_row_count=split.train.height,
    )
    _, outcome, _, _ = _build_pipeline_outcome(split=split, model=model)
    with pytest.raises(ProcessIntelligenceError, match="REGRESSION"):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_rejects_metadata_feature_order_mismatch() -> None:
    split = _regression_split()
    model = _fitted_spy(
        feature_names=["f2", "f1"],
        metadata_features=["f2", "f1"],
        fit_row_count=split.train.height,
    )
    _, outcome, _, _ = _build_pipeline_outcome(split=split, model=model)
    with pytest.raises(ProcessIntelligenceError, match="feature"):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_rejects_fit_row_count_mismatch() -> None:
    split = _regression_split()
    model = _fitted_spy(fit_row_count=99)
    _, outcome, _, _ = _build_pipeline_outcome(split=split, model=model)
    with pytest.raises(ProcessIntelligenceError, match="fit_row_count"):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_rejects_unfitted_detector() -> None:
    split = _regression_split()
    model = _fitted_spy(fit_row_count=split.train.height)
    detector = SpyDetector()
    report = _valid_pipeline_report(
        train_row_count=split.train.height,
        validation_row_count=split.validation.height,
        test_row_count=split.test.height,
    )
    outcome = ResidualAnomalyPipelineOutcome(
        regression_model=model,
        residual_detector=detector,
        train_scored=_empty_scored_like(split.train),
        validation_scored=_empty_scored_like(split.validation),
        combined_scored=_empty_scored_like(split.train),
        report=report,
    )
    with pytest.raises(ProcessIntelligenceError, match="fitted"):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_rejects_detector_method_mismatch() -> None:
    split, outcome, model, detector = _build_pipeline_outcome()
    bad_outcome = ResidualAnomalyPipelineOutcome(
        regression_model=model,
        residual_detector=detector,
        train_scored=outcome.train_scored,
        validation_scored=outcome.validation_scored,
        combined_scored=outcome.combined_scored,
        report=outcome.report.model_copy(
            update={"detector_method": ResidualThresholdMethod.QUANTILE}
        ),
    )
    with pytest.raises(ProcessIntelligenceError, match="method"):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            bad_outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_rejects_detector_calibration_row_count_mismatch() -> None:
    split, outcome, model, detector = _build_pipeline_outcome()
    # Force report calibration_row_count match but detector has different
    # by swapping to a detector calibrated on fewer rows is hard; mutate report.
    bad_outcome = ResidualAnomalyPipelineOutcome(
        regression_model=model,
        residual_detector=detector,
        train_scored=outcome.train_scored,
        validation_scored=outcome.validation_scored,
        combined_scored=outcome.combined_scored,
        report=outcome.report.model_copy(
            update={
                "calibration_row_count": split.validation.height,
                "validation_row_count": split.validation.height,
            }
        ),
    )
    # Mutate detector calibration row_count via reconstruct
    calibration = detector.calibration
    assert calibration is not None
    detector._calibration = calibration.model_copy(update={"row_count": 1})
    with pytest.raises(ProcessIntelligenceError, match="row_count"):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            bad_outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_rejects_residual_center_mismatch() -> None:
    split, outcome, model, detector = _build_pipeline_outcome()
    bad_outcome = ResidualAnomalyPipelineOutcome(
        regression_model=model,
        residual_detector=detector,
        train_scored=outcome.train_scored,
        validation_scored=outcome.validation_scored,
        combined_scored=outcome.combined_scored,
        report=outcome.report.model_copy(update={"residual_center": 999.0}),
    )
    with pytest.raises(ProcessIntelligenceError, match="residual_center"):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            bad_outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_rejects_residual_scale_mismatch() -> None:
    split, outcome, model, detector = _build_pipeline_outcome()
    bad_outcome = ResidualAnomalyPipelineOutcome(
        regression_model=model,
        residual_detector=detector,
        train_scored=outcome.train_scored,
        validation_scored=outcome.validation_scored,
        combined_scored=outcome.combined_scored,
        report=outcome.report.model_copy(update={"residual_scale": 999.0}),
    )
    with pytest.raises(ProcessIntelligenceError, match="residual_scale"):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            bad_outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_rejects_threshold_mismatch() -> None:
    split, outcome, model, detector = _build_pipeline_outcome()
    bad_outcome = ResidualAnomalyPipelineOutcome(
        regression_model=model,
        residual_detector=detector,
        train_scored=outcome.train_scored,
        validation_scored=outcome.validation_scored,
        combined_scored=outcome.combined_scored,
        report=outcome.report.model_copy(update={"threshold": 999.0}),
    )
    with pytest.raises(ProcessIntelligenceError, match="threshold"):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            bad_outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_rejects_missing_validation_summary() -> None:
    split, outcome, model, detector = _build_pipeline_outcome()
    train_only = [
        s for s in outcome.report.partition_summaries if s.partition == "train"
    ]
    bad_report = ResidualAnomalyPipelineReport.model_construct(
        **{
            **outcome.report.model_dump(),
            "partition_summaries": train_only + train_only,
        }
    )
    bad_outcome = ResidualAnomalyPipelineOutcome(
        regression_model=model,
        residual_detector=detector,
        train_scored=outcome.train_scored,
        validation_scored=outcome.validation_scored,
        combined_scored=outcome.combined_scored,
        report=bad_report,
    )
    with pytest.raises(DataValidationError, match="validation"):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            bad_outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_rejects_duplicate_validation_summary() -> None:
    split, outcome, model, detector = _build_pipeline_outcome()
    val = next(
        s for s in outcome.report.partition_summaries if s.partition == "validation"
    )
    bad_report = ResidualAnomalyPipelineReport.model_construct(
        **{
            **outcome.report.model_dump(),
            "partition_summaries": [val, val],
        }
    )
    bad_outcome = ResidualAnomalyPipelineOutcome(
        regression_model=model,
        residual_detector=detector,
        train_scored=outcome.train_scored,
        validation_scored=outcome.validation_scored,
        combined_scored=outcome.combined_scored,
        report=bad_report,
    )
    with pytest.raises(DataValidationError, match="multiple validation"):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            bad_outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_rejects_validation_summary_not_scored() -> None:
    split, outcome, model, detector = _build_pipeline_outcome()
    train = next(
        s for s in outcome.report.partition_summaries if s.partition == "train"
    )
    unscored = ResidualPartitionSummary.model_construct(
        partition="validation",
        input_row_count=split.validation.height,
        scored=False,
        prediction_count=0,
        anomaly_count=0,
        anomaly_fraction=0.0,
        residual_mean=None,
        residual_std=None,
        score_min=None,
        score_max=None,
        score_mean=None,
        score_std=None,
        prediction_seconds=0.0,
        scoring_seconds=0.0,
        total_seconds=0.0,
        warnings=[],
    )
    bad_report = ResidualAnomalyPipelineReport.model_construct(
        **{
            **outcome.report.model_dump(),
            "partition_summaries": [train, unscored],
            "total_scored_row_count": train.prediction_count,
            "total_anomaly_count": train.anomaly_count,
            "total_anomaly_fraction": (
                0.0
                if train.prediction_count == 0
                else train.anomaly_count / train.prediction_count
            ),
            "train_validation_anomaly_fraction_shift": (
                abs(0.0 - train.anomaly_fraction) if train.scored else None
            ),
        }
    )
    bad_outcome = ResidualAnomalyPipelineOutcome(
        regression_model=model,
        residual_detector=detector,
        train_scored=outcome.train_scored,
        validation_scored=outcome.validation_scored,
        combined_scored=outcome.combined_scored,
        report=bad_report,
    )
    with pytest.raises(DataValidationError, match="scored"):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            bad_outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_consistency_false_skips_detailed_compare() -> None:
    split, outcome, model, detector = _build_pipeline_outcome()
    # Change residual_center in report; with consistency=False should still pass
    # basics if model/detector fitted.
    bad_outcome = ResidualAnomalyPipelineOutcome(
        regression_model=model,
        residual_detector=detector,
        train_scored=outcome.train_scored,
        validation_scored=outcome.validation_scored,
        combined_scored=outcome.combined_scored,
        report=outcome.report.model_copy(update={"residual_center": 123.0}),
    )
    result = ResidualAnomalyFinalEvaluator(
        policy=ResidualAnomalyFinalEvaluationPolicy(
            require_pipeline_consistency=False
        )
    ).evaluate(
        split,
        bad_outcome,
        target_column=_TARGET,
        feature_columns=_FEATURES,
        leakage_report=_safe_report(),
    )
    assert result.report.test_row_count == 3


# --- State immutability / prediction / scoring ---


def test_predict_and_detect_once_no_fit() -> None:
    split, outcome, model, detector = _build_pipeline_outcome()
    before_threshold = detector.threshold
    before_fitted_at = detector.fitted_at
    before_calibration = detector.calibration
    before_features = list(model.feature_names)
    before_fit_rows = model.fit_row_count
    before_fitted = model.is_fitted

    model.predict_calls.clear()
    detector.detect_calls.clear()
    detector.fit_calls.clear()
    detector.score_samples_calls = 0
    detector.predict_calls = 0

    result = ResidualAnomalyFinalEvaluator().evaluate(
        split,
        outcome,
        target_column=_TARGET,
        feature_columns=_FEATURES,
        leakage_report=_safe_report(),
    )

    assert len(model.predict_calls) == 1
    assert model.predict_calls[0][0] == _FEATURES
    assert model.fit_calls == 0
    assert model.evaluate_calls == 0
    assert len(detector.detect_calls) == 1
    assert detector.fit_calls == []
    assert detector.score_samples_calls == 0
    assert detector.predict_calls == 0
    assert model.is_fitted is before_fitted
    assert model.feature_names == before_features
    assert model.fit_row_count == before_fit_rows
    assert detector.threshold == before_threshold
    assert detector.fitted_at == before_fitted_at
    assert detector.calibration == before_calibration
    assert result.regression_model is model
    assert result.residual_detector is detector


def test_rejects_boolean_prediction() -> None:
    split, outcome, model, _ = _build_pipeline_outcome()
    model._predictions = np.array([True, False, True])
    with pytest.raises(ProcessIntelligenceError, match="boolean|numeric"):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


@pytest.mark.parametrize(
    "preds",
    [
        np.array([1.0, float("nan"), 3.0]),
        np.array([1.0, float("inf"), 3.0]),
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        np.array([1.0, 2.0]),
    ],
)
def test_rejects_bad_prediction_arrays(preds: np.ndarray) -> None:
    split, outcome, model, _ = _build_pipeline_outcome()
    model._predictions = preds
    with pytest.raises(ProcessIntelligenceError):
        ResidualAnomalyFinalEvaluator().evaluate(
            split,
            outcome,
            target_column=_TARGET,
            feature_columns=_FEATURES,
            leakage_report=_safe_report(),
        )


def test_no_registry_instantiation() -> None:
    with patch(
        "process_intelligence.models.registry.ModelRegistry.instantiate"
    ) as mocked:
        _evaluate()
        mocked.assert_not_called()


def test_rejects_bad_detect_result_row_count() -> None:
    split, outcome, model, detector = _build_pipeline_outcome()

    def _bad_detect(y: Any, pred: Any) -> ResidualAnomalyResult:
        good = ResidualAnomalyDetector.detect(detector, y, pred)
        payload = good.model_dump()
        payload["row_count"] = good.row_count + 1
        return ResidualAnomalyResult.model_construct(**payload)

    with patch.object(detector, "detect", side_effect=_bad_detect):
        with pytest.raises(ProcessIntelligenceError, match="row_count"):
            ResidualAnomalyFinalEvaluator().evaluate(
                split,
                outcome,
                target_column=_TARGET,
                feature_columns=_FEATURES,
                leakage_report=_safe_report(),
            )


def test_rejects_mismatched_result_predictions() -> None:
    split, outcome, model, detector = _build_pipeline_outcome()

    def _bad_detect(y: Any, pred: Any) -> ResidualAnomalyResult:
        good = ResidualAnomalyDetector.detect(detector, y, pred)
        payload = good.model_dump()
        payload["predictions"] = [v + 1.0 for v in good.predictions]
        return ResidualAnomalyResult.model_construct(**payload)

    with patch.object(detector, "detect", side_effect=_bad_detect):
        with pytest.raises(ProcessIntelligenceError, match="prediction"):
            ResidualAnomalyFinalEvaluator().evaluate(
                split,
                outcome,
                target_column=_TARGET,
                feature_columns=_FEATURES,
                leakage_report=_safe_report(),
            )


@pytest.mark.parametrize(
    ("field", "mutator"),
    [
        (
            "residuals",
            lambda good: [v + 1.0 for v in good.residuals],
        ),
        (
            "absolute_centered_residuals",
            lambda good: [v + 1.0 for v in good.absolute_centered_residuals],
        ),
        (
            "scores",
            lambda good: [float("nan")] + list(good.scores[1:]),
        ),
        (
            "scores",
            lambda good: [-1.0] + list(good.scores[1:]),
        ),
        (
            "raw_predictions",
            lambda good: [0] + list(good.raw_predictions[1:]),
        ),
        (
            "is_anomaly",
            lambda good: [not good.is_anomaly[0]] + list(good.is_anomaly[1:]),
        ),
        (
            "anomaly_count",
            lambda good: good.anomaly_count + 1,
        ),
        (
            "threshold",
            lambda good: good.threshold + 1.0,
        ),
    ],
)
def test_rejects_inconsistent_detect_result(
    field: str,
    mutator: Callable[[ResidualAnomalyResult], Any],
) -> None:
    split, outcome, model, detector = _build_pipeline_outcome()

    def _bad_detect(y: Any, pred: Any) -> ResidualAnomalyResult:
        good = ResidualAnomalyDetector.detect(detector, y, pred)
        payload = good.model_dump()
        payload[field] = mutator(good)
        return ResidualAnomalyResult.model_construct(**payload)

    with patch.object(detector, "detect", side_effect=_bad_detect):
        with pytest.raises(ProcessIntelligenceError):
            ResidualAnomalyFinalEvaluator().evaluate(
                split,
                outcome,
                target_column=_TARGET,
                feature_columns=_FEATURES,
                leakage_report=_safe_report(),
            )


# --- Test scored DataFrame ---


def test_test_scored_dataframe_contract() -> None:
    split, outcome, _, _ = _build_pipeline_outcome()
    before = split.test.clone()
    result = ResidualAnomalyFinalEvaluator().evaluate(
        split,
        outcome,
        target_column=_TARGET,
        feature_columns=_FEATURES,
        leakage_report=_safe_report(),
    )
    scored = result.test_scored
    assert list(scored.columns[: before.width]) == list(before.columns)
    assert list(scored.columns[-7:]) == _RESULT_COLUMNS
    assert scored[_REGRESSION_PREDICTION_COLUMN].dtype == pl.Float64
    assert scored[_REGRESSION_RESIDUAL_COLUMN].dtype == pl.Float64
    assert scored[_ABSOLUTE_CENTERED_RESIDUAL_COLUMN].dtype == pl.Float64
    assert scored[_RESIDUAL_ANOMALY_SCORE_COLUMN].dtype == pl.Float64
    assert scored[_IS_RESIDUAL_ANOMALY_COLUMN].dtype == pl.Boolean
    assert scored[_RESIDUAL_ANOMALY_RAW_PREDICTION_COLUMN].dtype == pl.Int64
    assert scored[_DATA_PARTITION_COLUMN].dtype == pl.String
    assert scored[_DATA_PARTITION_COLUMN].to_list() == ["test"] * before.height
    for column in before.columns:
        assert scored[column].dtype == before[column].dtype
        assert scored[column].to_list() == before[column].to_list()
    assert scored[ORIGINAL_ROW_ID_COLUMN].to_list() == before[
        ORIGINAL_ROW_ID_COLUMN
    ].to_list()
    assert scored.height == before.height
    assert scored[ORIGINAL_ROW_ID_COLUMN].n_unique() == scored.height
    assert split.test.to_dict(as_series=False) == before.to_dict(as_series=False)


# --- Metrics / separation / shifts / flags / warnings ---


def test_validation_metrics_copied_without_repredict() -> None:
    split, outcome, model, detector = _build_pipeline_outcome()
    model.predict_calls.clear()
    detector.detect_calls.clear()
    result = ResidualAnomalyFinalEvaluator().evaluate(
        split,
        outcome,
        target_column=_TARGET,
        feature_columns=_FEATURES,
        leakage_report=_safe_report(),
    )
    assert len(model.predict_calls) == 1
    assert len(detector.detect_calls) == 1
    val_summary = next(
        s for s in outcome.report.partition_summaries if s.partition == "validation"
    )
    metrics = result.report.validation_metrics
    assert list(metrics.keys()) == list(_VALIDATION_METRIC_KEYS)
    assert metrics["validation_anomaly_fraction"] == pytest.approx(
        val_summary.anomaly_fraction
    )
    assert metrics["validation_residual_mean"] == pytest.approx(
        val_summary.residual_mean  # type: ignore[arg-type]
    )
    assert metrics["validation_score_range"] == pytest.approx(
        val_summary.score_max - val_summary.score_min  # type: ignore[operator]
    )
    before = outcome.report.model_dump()
    assert outcome.report.model_dump() == before


def test_test_metrics_and_shifts() -> None:
    result = _evaluate()
    metrics = result.report.test_metrics
    assert list(metrics.keys()) == list(_TEST_METRIC_KEYS)
    assert all(math.isfinite(v) for v in metrics.values())
    assert metrics["test_score_range"] == pytest.approx(
        metrics["test_score_max"] - metrics["test_score_min"]
    )
    assert result.report.anomaly_fraction_shift == pytest.approx(
        abs(
            metrics["test_anomaly_fraction"]
            - result.report.validation_metrics["validation_anomaly_fraction"]
        )
    )
    assert result.report.score_mean_shift == pytest.approx(
        metrics["test_score_mean"]
        - result.report.validation_metrics["validation_score_mean"]
    )
    assert result.report.standardized_score_mean_shift >= 0.0
    assert result.report.score_std_ratio >= 0.0
    assert math.isfinite(result.report.score_mean_shift)


def test_score_separation_both_groups() -> None:
    # Force large residuals on test so some anomalies appear.
    split = _regression_split(test_target_offset=100.0)
    result = _evaluate(split=split)
    if result.report.test_has_normal_and_anomaly:
        assert result.report.test_score_separation is not None
        assert math.isfinite(result.report.test_score_separation)


def test_score_separation_none_when_single_group() -> None:
    # Use identical train/val/test residuals likely all-normal with MAD.
    split = _regression_split(test_target_offset=0.0)
    result = _evaluate(split=split)
    if not result.report.test_has_normal_and_anomaly:
        assert result.report.test_score_separation is None
        assert _FLAG_DEGENERATE in result.report.quality_flags


def test_quality_flags_and_warnings_order() -> None:
    policy = ResidualAnomalyFinalEvaluationPolicy(
        minimum_anomaly_fraction=0.5,
        maximum_anomaly_fraction=0.6,
        maximum_anomaly_fraction_shift=0.0,
        maximum_standardized_score_mean_shift=1e-12,
        minimum_score_std_ratio=10.0,
        maximum_score_std_ratio=20.0,
        warn_on_distribution_shift=True,
    )
    split = _regression_split(test_target_offset=0.0)
    result = _evaluate(split=split, policy=policy)
    flags = result.report.quality_flags
    assert len(flags) == len(set(flags))
    # Fraction below should appear early when anomaly fraction is low.
    if result.report.test_metrics["test_anomaly_fraction"] < 0.5:
        assert _FLAG_FRACTION_BELOW in flags
    assert _WARNING_QUALITY_FLAGS in result.report.warnings
    assert _WARNING_DISTRIBUTION_SHIFT in result.report.warnings
    assert len(result.report.warnings) == len(set(result.report.warnings))


def test_warn_on_distribution_shift_false_omits_warning() -> None:
    policy = ResidualAnomalyFinalEvaluationPolicy(
        maximum_anomaly_fraction_shift=0.0,
        warn_on_distribution_shift=False,
    )
    result = _evaluate(
        split=_regression_split(test_target_offset=50.0),
        policy=policy,
    )
    assert _WARNING_DISTRIBUTION_SHIFT not in result.report.warnings


def test_pipeline_and_result_warnings_reflected() -> None:
    split, outcome, model, detector = _build_pipeline_outcome()
    report = outcome.report.model_copy(update={"warnings": ["pipeline warn"]})
    outcome = ResidualAnomalyPipelineOutcome(
        regression_model=model,
        residual_detector=detector,
        train_scored=outcome.train_scored,
        validation_scored=outcome.validation_scored,
        combined_scored=outcome.combined_scored,
        report=report,
    )
    result = ResidualAnomalyFinalEvaluator().evaluate(
        split,
        outcome,
        target_column=_TARGET,
        feature_columns=_FEATURES,
        leakage_report=_warning_report(),
    )
    assert _WARNING_LEAKAGE in result.report.warnings
    assert "pipeline warn" in result.report.warnings


def test_zero_and_all_anomaly_warnings() -> None:
    # Zero anomalies expected for small well-fit residuals.
    result = _evaluate(split=_regression_split(test_target_offset=0.0))
    if result.report.test_metrics["test_anomaly_fraction"] == 0.0:
        assert _WARNING_NO_ANOMALIES in result.report.warnings

    # Huge offset should mark all as anomalies with low threshold.
    split = _regression_split(test_target_offset=1e6)
    model = _fitted_spy(fit_row_count=split.train.height)
    detector = _calibrate_spy_detector(
        model,
        split,
        config=ResidualAnomalyConfig(
            method=ResidualThresholdMethod.QUANTILE,
            quantile=0.01,
        ),
    )
    _, outcome, _, _ = _build_pipeline_outcome(
        split=split, model=model, detector=detector
    )
    result = ResidualAnomalyFinalEvaluator().evaluate(
        split,
        outcome,
        target_column=_TARGET,
        feature_columns=_FEATURES,
        leakage_report=_safe_report(),
    )
    if result.report.test_metrics["test_anomaly_fraction"] == 1.0:
        assert _WARNING_ALL_ANOMALIES in result.report.warnings


def test_negative_separation_flag_when_inverted() -> None:
    # Construct synthetic result via patched detect with inverted groups.
    split, outcome, model, detector = _build_pipeline_outcome()

    def _inverted(y: Any, pred: Any) -> ResidualAnomalyResult:
        good = ResidualAnomalyDetector.detect(detector, y, pred)
        # Force both groups with inverted score means.
        scores = [10.0, 0.0, 0.0]
        raw = [1, -1, -1]  # normal has higher score -> negative separation
        flags = [False, True, True]
        payload = good.model_dump()
        payload.update(
            {
                "scores": scores,
                "raw_predictions": raw,
                "is_anomaly": flags,
                "anomaly_count": 2,
                "anomaly_fraction": 2 / 3,
                "score_min": 0.0,
                "score_max": 10.0,
                "score_mean": float(np.mean(scores)),
                "score_std": float(np.std(scores, ddof=0)),
                "warnings": [],
            }
        )
        return ResidualAnomalyResult(**payload)

    with patch.object(detector, "detect", side_effect=_inverted):
        # Also need predictions/residuals to match; patch further to pass
        # validation by aligning predictions to model output.
        def _aligned(y: Any, pred: Any) -> ResidualAnomalyResult:
            pred_arr = np.asarray(pred, dtype=float)
            y_arr = (
                y.to_numpy() if isinstance(y, pl.Series) else np.asarray(y, dtype=float)
            )
            residuals = y_arr - pred_arr
            center = float(detector.calibration.residual_center)  # type: ignore[union-attr]
            centered = np.abs(residuals - center)
            scores = np.array([10.0, 0.0, 0.0], dtype=float)
            raw = [1, -1, -1]
            flags = [False, True, True]
            return ResidualAnomalyResult(
                predictions=[float(v) for v in pred_arr.tolist()],
                residuals=[float(v) for v in residuals.tolist()],
                absolute_centered_residuals=[float(v) for v in centered.tolist()],
                scores=[float(v) for v in scores.tolist()],
                is_anomaly=flags,
                raw_predictions=raw,
                threshold=float(detector.threshold),  # type: ignore[arg-type]
                row_count=3,
                anomaly_count=2,
                anomaly_fraction=2 / 3,
                residual_mean=float(np.mean(residuals)),
                residual_std=float(np.std(residuals, ddof=0)),
                score_min=0.0,
                score_max=10.0,
                score_mean=float(np.mean(scores)),
                score_std=float(np.std(scores, ddof=0)),
                warnings=[],
            )

        with patch.object(detector, "detect", side_effect=_aligned):
            result = ResidualAnomalyFinalEvaluator().evaluate(
                split,
                outcome,
                target_column=_TARGET,
                feature_columns=_FEATURES,
                leakage_report=_safe_report(),
            )
    assert result.report.test_score_separation is not None
    assert result.report.test_score_separation < 0.0
    assert _FLAG_NEGATIVE_SEPARATION in result.report.quality_flags


def test_quality_flags_do_not_change_threshold() -> None:
    split, outcome, _, detector = _build_pipeline_outcome()
    before = detector.threshold
    ResidualAnomalyFinalEvaluator(
        policy=ResidualAnomalyFinalEvaluationPolicy(
            minimum_anomaly_fraction=0.5,
            maximum_anomaly_fraction=0.9,
            maximum_anomaly_fraction_shift=0.0,
        )
    ).evaluate(
        split,
        outcome,
        target_column=_TARGET,
        feature_columns=_FEATURES,
        leakage_report=_safe_report(),
    )
    assert detector.threshold == before


# --- Timing / report fields / immutability ---


def test_timings_and_report_fields() -> None:
    split, outcome, model, detector = _build_pipeline_outcome()
    result = ResidualAnomalyFinalEvaluator().evaluate(
        split,
        outcome,
        target_column=_TARGET,
        feature_columns=_FEATURES,
        leakage_report=_safe_report(),
    )
    report = result.report
    assert report.prediction_seconds >= 0.0
    assert report.scoring_seconds >= 0.0
    assert report.total_seconds == pytest.approx(
        report.prediction_seconds + report.scoring_seconds
    )
    assert math.isfinite(report.total_seconds)
    assert report.evaluated_at.tzinfo is not None
    assert report.evaluated_at.utcoffset() is not None
    assert report.model_name == model.get_metadata().model_name
    assert report.estimator_key == model._spec.estimator_key
    assert report.target_column == _TARGET
    assert report.feature_columns == _FEATURES
    assert report.detector_method == detector.calibration.method  # type: ignore[union-attr]
    assert report.calibration_partition == "validation"
    assert report.test_partition == "test"
    assert report.train_row_count == split.train.height
    assert report.validation_row_count == split.validation.height
    assert report.test_row_count == split.test.height
    assert report.regression_fit_row_count == split.train.height
    assert report.calibration_row_count == split.validation.height
    assert report.threshold == pytest.approx(float(detector.threshold))  # type: ignore[arg-type]
    dump = report.model_dump()
    assert "estimator" not in dump
    assert "predictions" not in dump
    assert "residuals" not in dump


def test_outcome_reuse_and_immutability() -> None:
    split, outcome, model, detector = _build_pipeline_outcome()
    features = list(_FEATURES)
    leakage = _safe_report()
    split_before = (
        split.train.clone(),
        split.validation.clone(),
        split.test.clone(),
    )
    pipeline_before = (
        outcome.report.model_dump(),
        outcome.train_scored.clone(),
        outcome.validation_scored.clone(),
    )
    leakage_before = leakage.model_dump()
    calibration_before = detector.calibration
    threshold_before = detector.threshold

    evaluator = ResidualAnomalyFinalEvaluator()
    first = evaluator.evaluate(
        split,
        outcome,
        target_column=_TARGET,
        feature_columns=features,
        leakage_report=leakage,
    )
    mutated = first.test_scored.with_columns(pl.lit(1).alias("_mut"))
    assert "_mut" in mutated.columns
    assert "_mut" not in first.test_scored.columns

    second = evaluator.evaluate(
        split,
        outcome,
        target_column=_TARGET,
        feature_columns=features,
        leakage_report=leakage,
    )
    other = ResidualAnomalyFinalEvaluator().evaluate(
        split,
        outcome,
        target_column=_TARGET,
        feature_columns=features,
        leakage_report=leakage,
    )

    assert first.report.test_metrics == second.report.test_metrics
    assert first.test_scored.to_dict(as_series=False) == second.test_scored.to_dict(
        as_series=False
    )
    assert other.report.test_metrics == first.report.test_metrics
    assert model.fit_calls == 0
    assert detector.fit_calls == []
    assert detector.calibration == calibration_before
    assert detector.threshold == threshold_before
    assert features == _FEATURES
    assert leakage.model_dump() == leakage_before
    assert split.train.to_dict(as_series=False) == split_before[0].to_dict(
        as_series=False
    )
    assert split.validation.to_dict(as_series=False) == split_before[1].to_dict(
        as_series=False
    )
    assert split.test.to_dict(as_series=False) == split_before[2].to_dict(
        as_series=False
    )
    assert outcome.report.model_dump() == pipeline_before[0]
    assert outcome.train_scored.to_dict(as_series=False) == pipeline_before[1].to_dict(
        as_series=False
    )
    # Reuse model/detector after evaluation
    preds = model.predict(split.test.select(_FEATURES))
    detect = detector.detect(split.test.get_column(_TARGET), preds)
    assert isinstance(detect, ResidualAnomalyResult)


def test_integration_with_real_pipeline() -> None:
    split, pipeline_outcome = _run_pipeline_integration()
    result = ResidualAnomalyFinalEvaluator().evaluate(
        split,
        pipeline_outcome,
        target_column=_TARGET,
        feature_columns=_FEATURES,
        leakage_report=_safe_report(),
    )
    assert result.report.task is AnalysisTask.REGRESSION
    assert result.test_scored.height == split.test.height
    assert result.regression_model is pipeline_outcome.regression_model
    assert result.residual_detector is pipeline_outcome.residual_detector


def test_integration_with_supervised_screener() -> None:
    split = _regression_split()
    registry = create_default_supervised_model_registry()
    screener = SupervisedModelScreener(registry)
    screening = screener.screen(
        split,
        task=AnalysisTask.REGRESSION,
        target_column=_TARGET,
        feature_columns=_FEATURES,
        leakage_report=_safe_report(),
    )
    pipeline_outcome = ResidualAnomalyPipeline().run(
        split,
        screening,
        target_column=_TARGET,
        feature_columns=_FEATURES,
        leakage_report=_safe_report(),
    )
    result = ResidualAnomalyFinalEvaluator().evaluate(
        split,
        pipeline_outcome,
        target_column=_TARGET,
        feature_columns=_FEATURES,
        leakage_report=_safe_report(),
    )
    assert result.report.estimator_key == screening.summary.selected_estimator_key


def test_flag_constants_present() -> None:
    assert "residual anomaly" in _FLAG_FRACTION_BELOW
    assert "residual anomaly" in _FLAG_FRACTION_ABOVE
    assert "residual anomaly" in _FLAG_FRACTION_SHIFT
    assert "nearly constant" in _FLAG_SCORE_NEARLY_CONSTANT
    assert "degenerate" in _FLAG_DEGENERATE
    assert "inconsistent" in _FLAG_NEGATIVE_SEPARATION
    assert "shifted" in _FLAG_SCORE_MEAN_SHIFT
    assert "contracted" in _FLAG_SCORE_STD_CONTRACTED
    assert "expanded" in _FLAG_SCORE_STD_EXPANDED
