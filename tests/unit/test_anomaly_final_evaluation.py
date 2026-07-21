"""Unit tests for unsupervised anomaly final evaluation (Step 7E)."""

from __future__ import annotations

import copy
import dataclasses
import math
from datetime import UTC, datetime
from typing import Any, Self

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
from process_intelligence.core.protocols import BaseAnomalyModel, DataFrameLike, SeriesLike
from process_intelligence.core.schemas import (
    AnomalyEvent,
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
from process_intelligence.evaluation.splitting import (
    DatasetSplit,
    SplitStrategy,
    SplitSummary,
)
from process_intelligence.models import (
    AnomalyCandidateRunStatus,
    AnomalyCandidateScreeningResult,
    AnomalyDetectionResult,
    AnomalyFinalEvaluationOutcome,
    AnomalyFinalEvaluationPolicy,
    AnomalyFinalEvaluationReport,
    AnomalyFinalEvaluator,
    AnomalyScreeningOutcome,
    AnomalyScreeningSummary,
    ModelRegistry,
    UnsupervisedAnomalyModelScreener,
    create_default_anomaly_model_registry,
)


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


def _anomaly_split(
    *,
    test_feature_offset: float = 0.0,
    empty_test: bool = False,
    empty_train: bool = False,
    empty_validation: bool = False,
) -> DatasetSplit:
    if empty_train:
        train = pl.DataFrame(
            schema={
                ORIGINAL_ROW_ID_COLUMN: pl.Int64,
                "f1": pl.Float64,
                "f2": pl.Float64,
            }
        )
        train_ids: list[int] = []
    else:
        train = pl.DataFrame(
            {
                ORIGINAL_ROW_ID_COLUMN: [0, 1, 2, 3, 4, 5, 6, 7],
                "f1": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 5.0, 5.5],
                "f2": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 4.0, 4.5],
            }
        )
        train_ids = [0, 1, 2, 3, 4, 5, 6, 7]

    if empty_validation:
        validation = pl.DataFrame(
            schema={
                ORIGINAL_ROW_ID_COLUMN: pl.Int64,
                "f1": pl.Float64,
                "f2": pl.Float64,
            }
        )
        validation_ids: list[int] = []
    else:
        validation = pl.DataFrame(
            {
                ORIGINAL_ROW_ID_COLUMN: [8, 9, 10, 11],
                "f1": [0.05, 0.25, 4.8, 5.2],
                "f2": [0.02, 0.12, 3.9, 4.2],
            }
        )
        validation_ids = [8, 9, 10, 11]

    if empty_test:
        test = pl.DataFrame(
            schema={
                ORIGINAL_ROW_ID_COLUMN: pl.Int64,
                "f1": pl.Float64,
                "f2": pl.Float64,
            }
        )
        test_ids: list[int] = []
    else:
        test = pl.DataFrame(
            {
                ORIGINAL_ROW_ID_COLUMN: [12, 13, 14, 15],
                "f1": [
                    0.1 + test_feature_offset,
                    0.2 + test_feature_offset,
                    5.0 + test_feature_offset,
                    5.2 + test_feature_offset,
                ],
                "f2": [
                    0.05 + test_feature_offset,
                    0.1 + test_feature_offset,
                    4.0 + test_feature_offset,
                    4.2 + test_feature_offset,
                ],
            }
        )
        test_ids = [12, 13, 14, 15]

    return DatasetSplit(
        train=train,
        validation=validation,
        test=test,
        summary=_summary(
            train_ids=train_ids,
            validation_ids=validation_ids,
            test_ids=test_ids,
        ),
    )


def _safe_report(feature_columns: list[str] | None = None) -> LeakageReport:
    features = ["f1", "f2"] if feature_columns is None else list(feature_columns)
    return LeakageReport(
        is_safe=True,
        issues=[],
        blocker_count=0,
        warning_count=0,
        checked_feature_columns=features,
        checked_preprocessing_event_count=0,
    )


def _warning_report(feature_columns: list[str] | None = None) -> LeakageReport:
    features = ["f1", "f2"] if feature_columns is None else list(feature_columns)
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
        checked_feature_columns=features,
        checked_preprocessing_event_count=1,
    )


def _blocker_report(feature_columns: list[str] | None = None) -> LeakageReport:
    features = ["f1", "f2"] if feature_columns is None else list(feature_columns)
    issue = LeakageIssue(
        issue_type=LeakageIssueType.IDENTIFIER_INCLUDED_AS_FEATURE,
        severity=LeakageSeverity.BLOCKER,
        columns=["id"],
        message="identifier in features",
        suggested_action="remove identifier",
    )
    return LeakageReport(
        is_safe=False,
        issues=[issue],
        blocker_count=1,
        warning_count=0,
        checked_feature_columns=features,
        checked_preprocessing_event_count=0,
    )


def _spec(
    *,
    name: str = "Spy Anomaly",
    estimator_key: str = "spy_anomaly",
    priority: int = 10,
) -> ModelSpec:
    return ModelSpec(
        name=name,
        task=AnalysisTask.UNSUPERVISED_ANOMALY,
        estimator_key=estimator_key,
        optional_dependencies=[],
        priority=priority,
        time_budget_seconds=5.0,
    )


def _success_metrics(
    *,
    validation_fraction: float = 0.25,
    score_mean: float = 0.5,
    score_std: float = 0.4,
) -> dict[str, float]:
    return {
        "train_anomaly_fraction": 0.125,
        "validation_anomaly_fraction": validation_fraction,
        "anomaly_fraction_gap": abs(validation_fraction - 0.125),
        "validation_score_min": 0.0,
        "validation_score_max": 1.0,
        "validation_score_mean": score_mean,
        "validation_score_std": score_std,
        "validation_score_range": 1.0,
    }


def _success_result(
    *,
    name: str = "Spy Anomaly",
    estimator_key: str = "spy_anomaly",
    registry_rank: int = 0,
    metrics: dict[str, float] | None = None,
    score_separation: float | None = 1.5,
    has_normal_and_anomaly: bool = True,
    quality_flags: list[str] | None = None,
    warnings: list[str] | None = None,
) -> AnomalyCandidateScreeningResult:
    flags = [] if quality_flags is None else list(quality_flags)
    return AnomalyCandidateScreeningResult(
        spec=_spec(name=name, estimator_key=estimator_key),
        status=AnomalyCandidateRunStatus.SUCCESS,
        registry_rank=registry_rank,
        metrics=_success_metrics() if metrics is None else dict(metrics),
        score_separation=score_separation,
        has_normal_and_anomaly=has_normal_and_anomaly,
        quality_penalty_count=len(flags),
        quality_flags=flags,
        fit_seconds=0.1,
        train_scoring_seconds=0.05,
        validation_scoring_seconds=0.05,
        total_seconds=0.2,
        warnings=[] if warnings is None else list(warnings),
        error_type=None,
        error_message=None,
    )


def _failed_result(
    *,
    name: str = "Broken",
    estimator_key: str = "broken",
    registry_rank: int = 1,
) -> AnomalyCandidateScreeningResult:
    return AnomalyCandidateScreeningResult(
        spec=_spec(name=name, estimator_key=estimator_key),
        status=AnomalyCandidateRunStatus.FAILED,
        registry_rank=registry_rank,
        metrics={},
        score_separation=None,
        has_normal_and_anomaly=False,
        quality_penalty_count=0,
        quality_flags=[],
        fit_seconds=0.01,
        train_scoring_seconds=0.0,
        validation_scoring_seconds=0.0,
        total_seconds=0.01,
        warnings=[],
        error_type="ValueError",
        error_message="boom",
    )


def _valid_report(**overrides: object) -> AnomalyFinalEvaluationReport:
    payload: dict[str, object] = {
        "task": AnalysisTask.UNSUPERVISED_ANOMALY,
        "model_name": "Spy Anomaly",
        "estimator_key": "spy_anomaly",
        "feature_columns": ["f1", "f2"],
        "refit_on_train_validation": True,
        "fit_partitions": ["train", "validation"],
        "train_row_count": 8,
        "validation_row_count": 4,
        "test_row_count": 4,
        "final_fit_row_count": 12,
        "validation_metrics": _success_metrics(),
        "test_metrics": {
            "test_anomaly_fraction": 0.25,
            "test_score_min": 0.0,
            "test_score_max": 1.0,
            "test_score_mean": 0.5,
            "test_score_std": 0.4,
            "test_score_range": 1.0,
        },
        "validation_score_separation": 1.5,
        "test_score_separation": 1.2,
        "test_has_normal_and_anomaly": True,
        "anomaly_fraction_shift": 0.0,
        "score_mean_shift": 0.0,
        "score_std_ratio": 1.0,
        "quality_flags": [],
        "fit_seconds": 0.1,
        "test_scoring_seconds": 0.05,
        "total_seconds": 0.15,
        "evaluated_at": datetime.now(UTC),
        "warnings": [],
    }
    payload.update(overrides)
    return AnomalyFinalEvaluationReport(**payload)  # type: ignore[arg-type]


class _ExtendedMetadata(ModelMetadata):
    estimator_key: str = ""
    fit_row_count: int = 0
    fitted: bool = False


class _SpyAnomalyModel(BaseAnomalyModel):
    """Controllable anomaly model for final-evaluation unit tests."""

    instances: list[_SpyAnomalyModel] = []

    def __init__(
        self,
        *,
        name: str = "Spy Anomaly",
        estimator_key: str = "spy_anomaly",
        test_scores: list[float] | None = None,
        test_raw: list[int] | None = None,
        detect_warnings: list[str] | None = None,
        detect_override: Any | None = None,
        mark_fitted: bool = False,
        fit_row_count: int = 0,
        feature_names: tuple[str, ...] = (),
        task: AnalysisTask = AnalysisTask.UNSUPERVISED_ANOMALY,
        metadata_name: str | None = None,
        metadata_estimator_key: str | None = None,
        metadata_features: list[str] | None = None,
        metadata_fit_row_count: int | None = None,
        metadata_task: AnalysisTask | None = None,
    ) -> None:
        self._spec = _spec(name=name, estimator_key=estimator_key)
        self._test_scores = (
            [0.1, 0.2, 1.5, 1.8] if test_scores is None else list(test_scores)
        )
        self._test_raw = [1, 1, -1, -1] if test_raw is None else list(test_raw)
        self._detect_warnings = (
            [] if detect_warnings is None else list(detect_warnings)
        )
        self._detect_override = detect_override
        self._is_fitted = mark_fitted
        self._feature_names = feature_names
        self._fit_row_count = fit_row_count
        self._task = task
        self._metadata_name = metadata_name
        self._metadata_estimator_key = metadata_estimator_key
        self._metadata_features = metadata_features
        self._metadata_fit_row_count = metadata_fit_row_count
        self._metadata_task = metadata_task
        self.fit_calls = 0
        self.detect_calls = 0
        self.seen_fit_columns: list[str] | None = None
        self.seen_fit_row_ids: list[int] | None = None
        self.seen_fit_height: int | None = None
        self.seen_detect_columns: list[str] | None = None
        self.seen_detect_height: int | None = None
        type(self).instances.append(self)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def fit_row_count(self) -> int:
        return self._fit_row_count

    def fit(self, X: DataFrameLike, y: SeriesLike | None = None) -> Self:
        _ = y
        self.fit_calls += 1
        assert isinstance(X, pl.DataFrame)
        self.seen_fit_columns = list(X.columns)
        self.seen_fit_height = int(X.height)
        if ORIGINAL_ROW_ID_COLUMN in X.columns:
            self.seen_fit_row_ids = [
                int(v) for v in X.get_column(ORIGINAL_ROW_ID_COLUMN).to_list()
            ]
        else:
            self.seen_fit_row_ids = []
        self._feature_names = tuple(X.columns)
        self._fit_row_count = int(X.height)
        self._is_fitted = True
        return self

    def _build_result(
        self,
        *,
        scores: list[float],
        raw: list[int],
    ) -> AnomalyDetectionResult:
        is_anomaly = [value == -1 for value in raw]
        anomaly_count = sum(1 for flag in is_anomaly if flag)
        row_count = len(scores)
        score_min = float(min(scores)) if scores else None
        score_max = float(max(scores)) if scores else None
        if scores:
            score_mean = float(sum(scores) / row_count)
            score_mean = min(max(score_mean, score_min), score_max)  # type: ignore[arg-type]
        else:
            score_mean = None
        return AnomalyDetectionResult(
            scores=list(scores),
            is_anomaly=is_anomaly,
            raw_predictions=list(raw),
            threshold=0.0,
            row_count=row_count,
            anomaly_count=anomaly_count,
            anomaly_fraction=(
                float(anomaly_count) / float(row_count) if row_count else 0.0
            ),
            score_min=score_min,
            score_max=score_max,
            score_mean=score_mean,
            warnings=list(self._detect_warnings),
        )

    def detect(self, X: DataFrameLike) -> AnomalyDetectionResult:
        self.detect_calls += 1
        assert isinstance(X, pl.DataFrame)
        self.seen_detect_columns = list(X.columns)
        self.seen_detect_height = int(X.height)
        if self._detect_override is not None:
            return self._detect_override(X)
        scores = list(self._test_scores[: X.height])
        raw = list(self._test_raw[: X.height])
        if len(scores) != X.height:
            scores = [0.1] * X.height
            raw = [1] * X.height
        return self._build_result(scores=scores, raw=raw)

    def predict(self, X: DataFrameLike) -> np.ndarray:
        result = self.detect(X)
        return np.asarray(result.raw_predictions, dtype=np.int64)

    def score_samples(self, X: DataFrameLike) -> np.ndarray:
        result = self.detect(X)
        return np.asarray(result.scores, dtype=np.float64)

    def classify_anomalies(self, X: DataFrameLike) -> list[AnomalyEvent]:
        _ = X
        return []

    def evaluate(
        self,
        X: DataFrameLike,
        y: SeriesLike | None = None,
    ) -> ModelEvaluation:
        _ = (X, y)
        raise ProcessIntelligenceError("not implemented")

    def explain(self, X: DataFrameLike) -> ExplanationResult:
        _ = X
        return ExplanationResult(method="none", feature_importances={}, notes=[])

    def get_metadata(self) -> ModelMetadata:
        return _ExtendedMetadata(
            model_name=(
                self._metadata_name
                if self._metadata_name is not None
                else self._spec.name
            ),
            version="test",
            task=(
                self._metadata_task
                if self._metadata_task is not None
                else self._task
            ),
            features=(
                list(self._metadata_features)
                if self._metadata_features is not None
                else list(self._feature_names)
            ),
            estimator_key=(
                self._metadata_estimator_key
                if self._metadata_estimator_key is not None
                else self._spec.estimator_key
            ),
            fit_row_count=(
                self._metadata_fit_row_count
                if self._metadata_fit_row_count is not None
                else self._fit_row_count
            ),
            fitted=self._is_fitted,
        )


def _factory_for(template: _SpyAnomalyModel):
    def _factory() -> BaseAnomalyModel:
        return _SpyAnomalyModel(
            name=template._spec.name,
            estimator_key=template._spec.estimator_key,
            test_scores=template._test_scores,
            test_raw=template._test_raw,
            detect_warnings=template._detect_warnings,
            detect_override=template._detect_override,
        )

    return _factory


def _registry_from_spy(template: _SpyAnomalyModel) -> ModelRegistry:
    registry = ModelRegistry()
    registry.register_factory(template._spec.estimator_key, _factory_for(template))
    registry.register(template._spec)
    return registry


def _make_screening_outcome(
    *,
    selected: _SpyAnomalyModel | None = None,
    candidate: AnomalyCandidateScreeningResult | None = None,
    extra_candidates: list[AnomalyCandidateScreeningResult] | None = None,
    train_row_count: int = 8,
    validation_row_count: int = 4,
    test_row_count: int = 4,
    feature_columns: list[str] | None = None,
    summary_overrides: dict[str, object] | None = None,
) -> AnomalyScreeningOutcome:
    features = ["f1", "f2"] if feature_columns is None else list(feature_columns)
    selected_result = candidate if candidate is not None else _success_result()
    if selected is None:
        selected_model = _SpyAnomalyModel(
            name=selected_result.spec.name,
            estimator_key=selected_result.spec.estimator_key,
            mark_fitted=True,
            fit_row_count=train_row_count,
            feature_names=tuple(features),
        )
    else:
        selected_model = selected

    results = [selected_result]
    if extra_candidates:
        results.extend(extra_candidates)

    payload: dict[str, object] = {
        "task": AnalysisTask.UNSUPERVISED_ANOMALY,
        "feature_columns": features,
        "candidate_results": results,
        "selected_model_name": selected_result.spec.name,
        "selected_estimator_key": selected_result.spec.estimator_key,
        "selected_registry_rank": selected_result.registry_rank,
        "selected_metrics": dict(selected_result.metrics),
        "selected_score_separation": selected_result.score_separation,
        "selected_quality_penalty_count": selected_result.quality_penalty_count,
        "ranking_method": (
            "quality penalties, non-degenerate detection, standardized score "
            "separation, fraction stability, score variation, registry order"
        ),
        "successful_model_names": [
            item.spec.name
            for item in results
            if item.status is AnomalyCandidateRunStatus.SUCCESS
        ],
        "failed_model_names": [
            item.spec.name
            for item in results
            if item.status is AnomalyCandidateRunStatus.FAILED
        ],
        "train_row_count": train_row_count,
        "validation_row_count": validation_row_count,
        "test_row_count": test_row_count,
        "warnings": [],
    }
    if summary_overrides:
        payload.update(summary_overrides)
    summary = AnomalyScreeningSummary(**payload)  # type: ignore[arg-type]
    return AnomalyScreeningOutcome(selected_model=selected_model, summary=summary)


def _evaluate_default(
    *,
    policy: AnomalyFinalEvaluationPolicy | None = None,
    split: DatasetSplit | None = None,
    outcome: AnomalyScreeningOutcome | None = None,
    registry: ModelRegistry | None = None,
    feature_columns: list[str] | None = None,
    leakage_report: LeakageReport | None = None,
) -> AnomalyFinalEvaluationOutcome:
    features = ["f1", "f2"] if feature_columns is None else list(feature_columns)
    used_split = _anomaly_split() if split is None else split
    used_outcome = (
        _make_screening_outcome(
            train_row_count=used_split.train.height,
            validation_row_count=used_split.validation.height,
            test_row_count=used_split.test.height,
            feature_columns=features,
        )
        if outcome is None
        else outcome
    )
    template = _SpyAnomalyModel(
        name=used_outcome.summary.selected_model_name,
        estimator_key=used_outcome.summary.selected_estimator_key,
        test_scores=[0.1, 0.2, 1.5, 1.8],
        test_raw=[1, 1, -1, -1],
    )
    used_registry = (
        _registry_from_spy(template) if registry is None else registry
    )
    evaluator = AnomalyFinalEvaluator(used_registry, policy=policy)
    return evaluator.evaluate(
        used_split,
        used_outcome,
        feature_columns=features,
        leakage_report=(
            _safe_report(features) if leakage_report is None else leakage_report
        ),
    )


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def test_policy_defaults() -> None:
    policy = AnomalyFinalEvaluationPolicy()
    assert policy.refit_on_train_validation is True
    assert policy.require_safe_leakage_report is True
    assert policy.require_screening_consistency is True
    assert policy.minimum_test_rows == 1
    assert policy.minimum_anomaly_fraction == 0.001
    assert policy.maximum_anomaly_fraction == 0.25
    assert policy.maximum_fraction_shift == 0.10
    assert policy.minimum_score_std == 1e-12
    assert policy.warn_on_distribution_shift is True


@pytest.mark.parametrize(
    "field_name",
    [
        "refit_on_train_validation",
        "require_safe_leakage_report",
        "require_screening_consistency",
        "warn_on_distribution_shift",
    ],
)
def test_policy_bool_fields_reject_non_bool(field_name: str) -> None:
    with pytest.raises(ValidationError):
        AnomalyFinalEvaluationPolicy(**{field_name: 1})  # type: ignore[arg-type]


def test_policy_minimum_test_rows_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        AnomalyFinalEvaluationPolicy(minimum_test_rows=0)


def test_policy_minimum_test_rows_rejects_bool() -> None:
    with pytest.raises(ValidationError):
        AnomalyFinalEvaluationPolicy(minimum_test_rows=True)  # type: ignore[arg-type]


@pytest.mark.parametrize("field_name", [
    "minimum_anomaly_fraction",
    "maximum_anomaly_fraction",
    "maximum_fraction_shift",
])
def test_policy_fraction_rejects_negative(field_name: str) -> None:
    with pytest.raises(ValidationError):
        AnomalyFinalEvaluationPolicy(**{field_name: -0.1})


@pytest.mark.parametrize("field_name", [
    "minimum_anomaly_fraction",
    "maximum_anomaly_fraction",
    "maximum_fraction_shift",
])
def test_policy_fraction_rejects_above_one(field_name: str) -> None:
    with pytest.raises(ValidationError):
        AnomalyFinalEvaluationPolicy(**{field_name: 1.1})


@pytest.mark.parametrize("field_name", [
    "minimum_anomaly_fraction",
    "maximum_anomaly_fraction",
    "maximum_fraction_shift",
])
def test_policy_fraction_rejects_bool(field_name: str) -> None:
    with pytest.raises(ValidationError):
        AnomalyFinalEvaluationPolicy(**{field_name: True})  # type: ignore[arg-type]


@pytest.mark.parametrize("field_name", [
    "minimum_anomaly_fraction",
    "maximum_anomaly_fraction",
    "maximum_fraction_shift",
])
def test_policy_fraction_rejects_nan(field_name: str) -> None:
    with pytest.raises(ValidationError):
        AnomalyFinalEvaluationPolicy(**{field_name: float("nan")})


@pytest.mark.parametrize("field_name", [
    "minimum_anomaly_fraction",
    "maximum_anomaly_fraction",
    "maximum_fraction_shift",
])
def test_policy_fraction_rejects_inf(field_name: str) -> None:
    with pytest.raises(ValidationError):
        AnomalyFinalEvaluationPolicy(**{field_name: float("inf")})


def test_policy_minimum_fraction_must_be_less_than_maximum() -> None:
    with pytest.raises(ValidationError):
        AnomalyFinalEvaluationPolicy(
            minimum_anomaly_fraction=0.2,
            maximum_anomaly_fraction=0.2,
        )


def test_policy_minimum_score_std_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        AnomalyFinalEvaluationPolicy(minimum_score_std=0.0)


def test_policy_minimum_score_std_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        AnomalyFinalEvaluationPolicy(minimum_score_std=-1.0)


def test_policy_minimum_score_std_rejects_bool() -> None:
    with pytest.raises(ValidationError):
        AnomalyFinalEvaluationPolicy(minimum_score_std=True)  # type: ignore[arg-type]


def test_policy_round_trip() -> None:
    policy = AnomalyFinalEvaluationPolicy(
        refit_on_train_validation=False,
        minimum_test_rows=3,
        minimum_anomaly_fraction=0.01,
        maximum_anomaly_fraction=0.2,
    )
    restored = AnomalyFinalEvaluationPolicy.model_validate(policy.model_dump())
    assert restored == policy


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_report_valid_creation() -> None:
    report = _valid_report()
    assert report.task is AnalysisTask.UNSUPERVISED_ANOMALY
    assert report.model_name == "Spy Anomaly"


def test_report_rejects_wrong_task() -> None:
    with pytest.raises(ValidationError):
        _valid_report(task=AnalysisTask.REGRESSION)


def test_report_rejects_empty_model_name() -> None:
    with pytest.raises(ValidationError):
        _valid_report(model_name="")


def test_report_rejects_empty_estimator_key() -> None:
    with pytest.raises(ValidationError):
        _valid_report(estimator_key="")


def test_report_rejects_empty_features() -> None:
    with pytest.raises(ValidationError):
        _valid_report(feature_columns=[])


def test_report_rejects_duplicate_features() -> None:
    with pytest.raises(ValidationError):
        _valid_report(feature_columns=["f1", "f1"])


def test_report_rejects_original_row_id_feature() -> None:
    with pytest.raises(ValidationError):
        _valid_report(feature_columns=[ORIGINAL_ROW_ID_COLUMN, "f1"])


def test_report_rejects_duplicate_fit_partitions() -> None:
    with pytest.raises(ValidationError):
        _valid_report(fit_partitions=["train", "train"])


def test_report_rejects_refit_true_partition_mismatch() -> None:
    with pytest.raises(ValidationError):
        _valid_report(
            refit_on_train_validation=True,
            fit_partitions=["train"],
            final_fit_row_count=8,
        )


def test_report_rejects_refit_false_partition_mismatch() -> None:
    with pytest.raises(ValidationError):
        _valid_report(
            refit_on_train_validation=False,
            fit_partitions=["train", "validation"],
            final_fit_row_count=8,
        )


def test_report_rejects_train_row_zero() -> None:
    with pytest.raises(ValidationError):
        _valid_report(train_row_count=0, final_fit_row_count=4)


def test_report_rejects_validation_row_zero() -> None:
    with pytest.raises(ValidationError):
        _valid_report(validation_row_count=0, final_fit_row_count=8)


def test_report_rejects_test_row_zero() -> None:
    with pytest.raises(ValidationError):
        _valid_report(test_row_count=0)


def test_report_rejects_final_fit_row_zero() -> None:
    with pytest.raises(ValidationError):
        _valid_report(final_fit_row_count=0)


def test_report_rejects_refit_true_final_count_mismatch() -> None:
    with pytest.raises(ValidationError):
        _valid_report(final_fit_row_count=10)


def test_report_rejects_refit_false_final_count_mismatch() -> None:
    with pytest.raises(ValidationError):
        _valid_report(
            refit_on_train_validation=False,
            fit_partitions=["train"],
            final_fit_row_count=7,
        )


def test_report_rejects_empty_validation_metrics() -> None:
    with pytest.raises(ValidationError):
        _valid_report(validation_metrics={})


def test_report_rejects_empty_test_metrics() -> None:
    with pytest.raises(ValidationError):
        _valid_report(test_metrics={})


def test_report_rejects_missing_test_metric_key() -> None:
    metrics = {
        "test_anomaly_fraction": 0.25,
        "test_score_min": 0.0,
        "test_score_max": 1.0,
        "test_score_mean": 0.5,
        "test_score_std": 0.4,
    }
    with pytest.raises(ValidationError):
        _valid_report(test_metrics=metrics)


def test_report_rejects_missing_validation_metric_key() -> None:
    metrics = dict(_success_metrics())
    del metrics["validation_anomaly_fraction"]
    with pytest.raises(ValidationError):
        _valid_report(validation_metrics=metrics)


def test_report_rejects_empty_metric_key() -> None:
    with pytest.raises(ValidationError):
        _valid_report(test_metrics={"": 1.0, **{
            "test_anomaly_fraction": 0.25,
            "test_score_min": 0.0,
            "test_score_max": 1.0,
            "test_score_mean": 0.5,
            "test_score_std": 0.4,
            "test_score_range": 1.0,
        }})


def test_report_rejects_bool_metric() -> None:
    metrics = {
        "test_anomaly_fraction": True,
        "test_score_min": 0.0,
        "test_score_max": 1.0,
        "test_score_mean": 0.5,
        "test_score_std": 0.4,
        "test_score_range": 1.0,
    }
    with pytest.raises(ValidationError):
        _valid_report(test_metrics=metrics)


def test_report_rejects_nan_metric() -> None:
    metrics = {
        "test_anomaly_fraction": float("nan"),
        "test_score_min": 0.0,
        "test_score_max": 1.0,
        "test_score_mean": 0.5,
        "test_score_std": 0.4,
        "test_score_range": 1.0,
    }
    with pytest.raises(ValidationError):
        _valid_report(test_metrics=metrics)


def test_report_rejects_inf_metric() -> None:
    metrics = {
        "test_anomaly_fraction": float("inf"),
        "test_score_min": 0.0,
        "test_score_max": 1.0,
        "test_score_mean": 0.5,
        "test_score_std": 0.4,
        "test_score_range": 1.0,
    }
    with pytest.raises(ValidationError):
        _valid_report(test_metrics=metrics)


def test_report_rejects_missing_separation_when_groups() -> None:
    with pytest.raises(ValidationError):
        _valid_report(
            test_has_normal_and_anomaly=True,
            test_score_separation=None,
        )


def test_report_rejects_negative_fraction_shift() -> None:
    with pytest.raises(ValidationError):
        _valid_report(anomaly_fraction_shift=-0.1)


def test_report_rejects_nan_score_mean_shift() -> None:
    with pytest.raises(ValidationError):
        _valid_report(score_mean_shift=float("nan"))


def test_report_rejects_negative_score_std_ratio() -> None:
    with pytest.raises(ValidationError):
        _valid_report(score_std_ratio=-1.0)


def test_report_rejects_duplicate_quality_flags() -> None:
    with pytest.raises(ValidationError):
        _valid_report(quality_flags=["a", "a"])


def test_report_rejects_negative_timing() -> None:
    with pytest.raises(ValidationError):
        _valid_report(fit_seconds=-0.1, total_seconds=-0.05)


def test_report_rejects_nan_timing() -> None:
    with pytest.raises(ValidationError):
        _valid_report(fit_seconds=float("nan"), total_seconds=float("nan"))


def test_report_rejects_total_time_mismatch() -> None:
    with pytest.raises(ValidationError):
        _valid_report(fit_seconds=0.1, test_scoring_seconds=0.05, total_seconds=0.2)


def test_report_rejects_naive_evaluated_at() -> None:
    with pytest.raises(ValidationError):
        _valid_report(evaluated_at=datetime(2024, 1, 1))


def test_report_rejects_duplicate_warnings() -> None:
    with pytest.raises(ValidationError):
        _valid_report(warnings=["a", "a"])


def test_report_mutable_default_independence() -> None:
    left = _valid_report()
    right = _valid_report()
    left.quality_flags.append("x")
    assert right.quality_flags == []


def test_report_round_trip() -> None:
    report = _valid_report()
    restored = AnomalyFinalEvaluationReport.model_validate(report.model_dump())
    assert restored.model_name == report.model_name
    assert restored.test_metrics == report.test_metrics


# ---------------------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------------------


def test_outcome_frozen_dataclass() -> None:
    model = _SpyAnomalyModel(mark_fitted=True, fit_row_count=8, feature_names=("f1", "f2"))
    scored = pl.DataFrame({ORIGINAL_ROW_ID_COLUMN: [1], "f1": [0.0]})
    report = _valid_report()
    outcome = AnomalyFinalEvaluationOutcome(
        final_model=model,
        test_scored=scored,
        report=report,
    )
    assert dataclasses.is_dataclass(outcome)
    assert outcome.__dataclass_params__.frozen  # type: ignore[attr-defined]
    assert "final_model" in outcome.__slots__
    assert isinstance(outcome.final_model, BaseAnomalyModel)
    assert isinstance(outcome.test_scored, pl.DataFrame)
    assert isinstance(outcome.report, AnomalyFinalEvaluationReport)


# ---------------------------------------------------------------------------
# Evaluator construction
# ---------------------------------------------------------------------------


def test_evaluator_construction() -> None:
    registry = ModelRegistry()
    evaluator = AnomalyFinalEvaluator(registry)
    assert isinstance(evaluator, AnomalyFinalEvaluator)


def test_evaluator_rejects_bad_registry_type() -> None:
    with pytest.raises(TypeError):
        AnomalyFinalEvaluator("not-a-registry")  # type: ignore[arg-type]


def test_evaluator_rejects_bad_policy_type() -> None:
    with pytest.raises(TypeError):
        AnomalyFinalEvaluator(ModelRegistry(), policy="bad")  # type: ignore[arg-type]


def test_evaluator_policy_isolation() -> None:
    policy = AnomalyFinalEvaluationPolicy(minimum_test_rows=2)
    evaluator = AnomalyFinalEvaluator(ModelRegistry(), policy=policy)
    policy.minimum_test_rows = 99
    assert evaluator._policy.minimum_test_rows == 2


def test_evaluator_instances_do_not_share_policy_state() -> None:
    policy = AnomalyFinalEvaluationPolicy(minimum_test_rows=3)
    left = AnomalyFinalEvaluator(ModelRegistry(), policy=policy)
    right = AnomalyFinalEvaluator(ModelRegistry(), policy=policy)
    left._policy.minimum_test_rows = 7
    assert right._policy.minimum_test_rows == 3


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_evaluate_rejects_bad_split_type() -> None:
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(TypeError):
        evaluator.evaluate(
            "split",  # type: ignore[arg-type]
            _make_screening_outcome(),
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )


def test_evaluate_rejects_bad_screening_outcome_type() -> None:
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(TypeError):
        evaluator.evaluate(
            _anomaly_split(),
            "outcome",  # type: ignore[arg-type]
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )


def test_evaluate_rejects_non_polars_partition() -> None:
    split = _anomaly_split()
    bad = DatasetSplit(
        train=split.train,
        validation="bad",  # type: ignore[arg-type]
        test=split.test,
        summary=split.summary,
    )
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(TypeError):
        evaluator.evaluate(
            bad,
            _make_screening_outcome(),
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )


def test_evaluate_rejects_schema_mismatch() -> None:
    split = _anomaly_split()
    bad_test = split.test.rename({"f1": "fx"})
    bad = DatasetSplit(
        train=split.train,
        validation=split.validation,
        test=bad_test,
        summary=split.summary,
    )
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(DataValidationError):
        evaluator.evaluate(
            bad,
            _make_screening_outcome(),
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )


def test_evaluate_rejects_dtype_mismatch() -> None:
    split = _anomaly_split()
    bad_test = split.test.with_columns(pl.col("f1").cast(pl.Int64))
    bad = DatasetSplit(
        train=split.train,
        validation=split.validation,
        test=bad_test,
        summary=split.summary,
    )
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(DataValidationError):
        evaluator.evaluate(
            bad,
            _make_screening_outcome(),
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )


def test_evaluate_rejects_missing_original_row_id() -> None:
    split = _anomaly_split()
    train = split.train.drop(ORIGINAL_ROW_ID_COLUMN)
    validation = split.validation.drop(ORIGINAL_ROW_ID_COLUMN)
    test = split.test.drop(ORIGINAL_ROW_ID_COLUMN)
    bad = DatasetSplit(
        train=train,
        validation=validation,
        test=test,
        summary=split.summary,
    )
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(DataValidationError):
        evaluator.evaluate(
            bad,
            _make_screening_outcome(),
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )


def test_evaluate_rejects_feature_columns_str() -> None:
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(TypeError):
        evaluator.evaluate(
            _anomaly_split(),
            _make_screening_outcome(),
            feature_columns="f1",  # type: ignore[arg-type]
            leakage_report=_safe_report(),
        )


def test_evaluate_rejects_feature_columns_bytes() -> None:
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(TypeError):
        evaluator.evaluate(
            _anomaly_split(),
            _make_screening_outcome(),
            feature_columns=b"f1",  # type: ignore[arg-type]
            leakage_report=_safe_report(),
        )


def test_evaluate_rejects_non_string_feature() -> None:
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(TypeError):
        evaluator.evaluate(
            _anomaly_split(),
            _make_screening_outcome(),
            feature_columns=["f1", 2],  # type: ignore[list-item]
            leakage_report=_safe_report(),
        )


def test_evaluate_rejects_blank_feature_name() -> None:
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(DataValidationError):
        evaluator.evaluate(
            _anomaly_split(),
            _make_screening_outcome(),
            feature_columns=["f1", "  "],
            leakage_report=_safe_report(["f1", "  "]),
        )


def test_evaluate_rejects_empty_feature_list() -> None:
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(DataValidationError):
        evaluator.evaluate(
            _anomaly_split(),
            _make_screening_outcome(),
            feature_columns=[],
            leakage_report=_safe_report([]),
        )


def test_evaluate_rejects_duplicate_features() -> None:
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(DataValidationError):
        evaluator.evaluate(
            _anomaly_split(),
            _make_screening_outcome(),
            feature_columns=["f1", "f1"],
            leakage_report=_safe_report(["f1", "f2"]),
        )


def test_evaluate_rejects_original_row_id_feature() -> None:
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(DataValidationError):
        evaluator.evaluate(
            _anomaly_split(),
            _make_screening_outcome(),
            feature_columns=[ORIGINAL_ROW_ID_COLUMN, "f1"],
            leakage_report=_safe_report([ORIGINAL_ROW_ID_COLUMN, "f1"]),
        )


def test_evaluate_rejects_reserved_result_feature() -> None:
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(DataValidationError):
        evaluator.evaluate(
            _anomaly_split(),
            _make_screening_outcome(),
            feature_columns=["f1", "_anomaly_score"],
            leakage_report=_safe_report(["f1", "_anomaly_score"]),
        )


def test_evaluate_rejects_missing_feature() -> None:
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(DataValidationError):
        evaluator.evaluate(
            _anomaly_split(),
            _make_screening_outcome(),
            feature_columns=["f1", "missing"],
            leakage_report=_safe_report(["f1", "missing"]),
        )


def test_evaluate_rejects_bad_leakage_report_type() -> None:
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(TypeError):
        evaluator.evaluate(
            _anomaly_split(),
            _make_screening_outcome(),
            feature_columns=["f1", "f2"],
            leakage_report="bad",  # type: ignore[arg-type]
        )


def test_evaluate_rejects_checked_feature_order_mismatch() -> None:
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(DataValidationError):
        evaluator.evaluate(
            _anomaly_split(),
            _make_screening_outcome(),
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(["f2", "f1"]),
        )


def test_evaluate_rejects_checked_feature_set_mismatch() -> None:
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(DataValidationError):
        evaluator.evaluate(
            _anomaly_split(),
            _make_screening_outcome(),
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(["f1", "fx"]),
        )


# ---------------------------------------------------------------------------
# Leakage gate
# ---------------------------------------------------------------------------


def test_blocker_report_raises_data_leakage_error() -> None:
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(DataLeakageError, match="IDENTIFIER_INCLUDED_AS_FEATURE"):
        evaluator.evaluate(
            _anomaly_split(),
            _make_screening_outcome(),
            feature_columns=["f1", "f2"],
            leakage_report=_blocker_report(),
        )


def test_warning_only_report_allowed() -> None:
    outcome = _evaluate_default(leakage_report=_warning_report())
    assert any("WARNING" in warning for warning in outcome.report.warnings)


def test_safe_report_allowed() -> None:
    outcome = _evaluate_default(leakage_report=_safe_report())
    assert outcome.report.test_row_count == 4


def test_require_safe_false_allows_blocker() -> None:
    policy = AnomalyFinalEvaluationPolicy(require_safe_leakage_report=False)
    outcome = _evaluate_default(policy=policy, leakage_report=_blocker_report())
    assert outcome.report.model_name == "Spy Anomaly"


def test_leakage_report_input_immutability() -> None:
    report = _warning_report()
    before = report.model_copy(deep=True)
    _ = _evaluate_default(leakage_report=report)
    assert report.model_dump() == before.model_dump()


# ---------------------------------------------------------------------------
# Minimum partitions
# ---------------------------------------------------------------------------


def test_empty_train_rejected() -> None:
    split = _anomaly_split(empty_train=True)
    policy = AnomalyFinalEvaluationPolicy(require_screening_consistency=False)
    evaluator = AnomalyFinalEvaluator(
        _registry_from_spy(_SpyAnomalyModel()),
        policy=policy,
    )
    with pytest.raises(InsufficientDataError):
        evaluator.evaluate(
            split,
            _make_screening_outcome(),
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )


def test_empty_validation_rejected() -> None:
    split = _anomaly_split(empty_validation=True)
    policy = AnomalyFinalEvaluationPolicy(require_screening_consistency=False)
    evaluator = AnomalyFinalEvaluator(
        _registry_from_spy(_SpyAnomalyModel()),
        policy=policy,
    )
    with pytest.raises(InsufficientDataError):
        evaluator.evaluate(
            split,
            _make_screening_outcome(),
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )


def test_test_below_minimum_rejected() -> None:
    policy = AnomalyFinalEvaluationPolicy(minimum_test_rows=5)
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()), policy=policy)
    with pytest.raises(InsufficientDataError):
        evaluator.evaluate(
            _anomaly_split(),
            _make_screening_outcome(),
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )


def test_test_not_used_as_validation_fallback() -> None:
    split = _anomaly_split(empty_validation=True)
    policy = AnomalyFinalEvaluationPolicy(require_screening_consistency=False)
    evaluator = AnomalyFinalEvaluator(
        _registry_from_spy(_SpyAnomalyModel()),
        policy=policy,
    )
    with pytest.raises(InsufficientDataError):
        evaluator.evaluate(
            split,
            _make_screening_outcome(),
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )
    assert split.test.height == 4


# ---------------------------------------------------------------------------
# Reserved columns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "column",
    [
        "_anomaly_score",
        "_is_anomaly",
        "_anomaly_raw_prediction",
        "_data_partition",
    ],
)
def test_reserved_column_conflict_rejected(column: str) -> None:
    split = _anomaly_split()
    train = split.train.with_columns(pl.lit(0).alias(column))
    validation = split.validation.with_columns(pl.lit(0).alias(column))
    test = split.test.with_columns(pl.lit(0).alias(column))
    bad = DatasetSplit(
        train=train,
        validation=validation,
        test=test,
        summary=split.summary,
    )
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(DataValidationError, match=column):
        evaluator.evaluate(
            bad,
            _make_screening_outcome(),
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )


# ---------------------------------------------------------------------------
# Screening consistency
# ---------------------------------------------------------------------------


def test_consistency_feature_order_mismatch() -> None:
    split = _anomaly_split()
    outcome = _make_screening_outcome(feature_columns=["f2", "f1"])
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(DataValidationError, match="feature_columns"):
        evaluator.evaluate(
            split,
            outcome,
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )


def test_consistency_train_row_mismatch() -> None:
    outcome = _make_screening_outcome(train_row_count=7)
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(DataValidationError, match="train_row_count"):
        evaluator.evaluate(
            _anomaly_split(),
            outcome,
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )


def test_consistency_validation_row_mismatch() -> None:
    outcome = _make_screening_outcome(validation_row_count=3)
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(DataValidationError, match="validation_row_count"):
        evaluator.evaluate(
            _anomaly_split(),
            outcome,
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )


def test_consistency_test_row_mismatch() -> None:
    outcome = _make_screening_outcome(test_row_count=3)
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(DataValidationError, match="test_row_count"):
        evaluator.evaluate(
            _anomaly_split(),
            outcome,
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )


def test_consistency_metadata_name_mismatch() -> None:
    selected = _SpyAnomalyModel(
        mark_fitted=True,
        fit_row_count=8,
        feature_names=("f1", "f2"),
        metadata_name="Other",
    )
    outcome = _make_screening_outcome(selected=selected)
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(ProcessIntelligenceError, match="metadata name"):
        evaluator.evaluate(
            _anomaly_split(),
            outcome,
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )


def test_consistency_metadata_estimator_mismatch() -> None:
    selected = _SpyAnomalyModel(
        mark_fitted=True,
        fit_row_count=8,
        feature_names=("f1", "f2"),
        metadata_estimator_key="other_key",
    )
    outcome = _make_screening_outcome(selected=selected)
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(ProcessIntelligenceError, match="estimator_key"):
        evaluator.evaluate(
            _anomaly_split(),
            outcome,
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )


def test_consistency_metadata_task_mismatch() -> None:
    selected = _SpyAnomalyModel(
        mark_fitted=True,
        fit_row_count=8,
        feature_names=("f1", "f2"),
        metadata_task=AnalysisTask.REGRESSION,
    )
    outcome = _make_screening_outcome(selected=selected)
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(ProcessIntelligenceError, match="task"):
        evaluator.evaluate(
            _anomaly_split(),
            outcome,
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )


def test_consistency_metadata_features_mismatch() -> None:
    selected = _SpyAnomalyModel(
        mark_fitted=True,
        fit_row_count=8,
        feature_names=("f1", "f2"),
        metadata_features=["f2", "f1"],
    )
    outcome = _make_screening_outcome(selected=selected)
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(ProcessIntelligenceError, match="features"):
        evaluator.evaluate(
            _anomaly_split(),
            outcome,
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )


def test_consistency_metadata_fit_row_mismatch() -> None:
    selected = _SpyAnomalyModel(
        mark_fitted=True,
        fit_row_count=8,
        feature_names=("f1", "f2"),
        metadata_fit_row_count=7,
    )
    outcome = _make_screening_outcome(selected=selected)
    evaluator = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    with pytest.raises(ProcessIntelligenceError, match="fit_row_count"):
        evaluator.evaluate(
            _anomaly_split(),
            outcome,
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )


def test_consistency_false_skips_detailed_checks() -> None:
    policy = AnomalyFinalEvaluationPolicy(require_screening_consistency=False)
    outcome = _make_screening_outcome(train_row_count=7)
    result = _evaluate_default(policy=policy, outcome=outcome)
    assert result.report.train_row_count == 8


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


def test_selected_success_candidate_found() -> None:
    outcome = _evaluate_default()
    assert outcome.report.model_name == "Spy Anomaly"
    assert outcome.report.estimator_key == "spy_anomaly"


def test_missing_selected_candidate_rejected() -> None:
    selected = _success_result(name="A", estimator_key="a", registry_rank=0)
    other = _success_result(name="B", estimator_key="b", registry_rank=1)
    # Build summary that claims A but only contains B after mutation workaround:
    # use summary with matching candidate then replace candidate_results via
    # constructing mismatched selection through extra failed-only path.
    results = [other]
    with pytest.raises(ValidationError):
        AnomalyScreeningSummary(
            task=AnalysisTask.UNSUPERVISED_ANOMALY,
            feature_columns=["f1", "f2"],
            candidate_results=results,
            selected_model_name=selected.spec.name,
            selected_estimator_key=selected.spec.estimator_key,
            selected_registry_rank=selected.registry_rank,
            selected_metrics=dict(selected.metrics),
            selected_score_separation=selected.score_separation,
            selected_quality_penalty_count=0,
            ranking_method="x",
            successful_model_names=["B"],
            failed_model_names=[],
            train_row_count=8,
            validation_row_count=4,
            test_row_count=4,
            warnings=[],
        )


def test_duplicate_selected_candidates_rejected() -> None:
    first = _success_result(name="Spy Anomaly", registry_rank=0)
    duplicate = _success_result(name="Spy Anomaly", registry_rank=0)
    with pytest.raises(ValidationError):
        AnomalyScreeningSummary(
            task=AnalysisTask.UNSUPERVISED_ANOMALY,
            feature_columns=["f1", "f2"],
            candidate_results=[first, duplicate],
            selected_model_name="Spy Anomaly",
            selected_estimator_key="spy_anomaly",
            selected_registry_rank=0,
            selected_metrics=dict(first.metrics),
            selected_score_separation=first.score_separation,
            selected_quality_penalty_count=0,
            ranking_method="x",
            successful_model_names=["Spy Anomaly"],
            failed_model_names=[],
            train_row_count=8,
            validation_row_count=4,
            test_row_count=4,
            warnings=[],
        )


def test_failed_candidate_not_selected() -> None:
    success = _success_result(name="Good", estimator_key="good", registry_rank=0)
    failed = _failed_result(name="Bad", estimator_key="bad", registry_rank=1)
    outcome = _make_screening_outcome(
        candidate=success,
        extra_candidates=[failed],
    )
    result = _evaluate_default(outcome=outcome)
    assert result.report.model_name == "Good"


# ---------------------------------------------------------------------------
# Final refit
# ---------------------------------------------------------------------------


def test_default_policy_creates_new_model_instance() -> None:
    _SpyAnomalyModel.instances.clear()
    screening = _make_screening_outcome()
    selected_before = screening.selected_model
    result = _evaluate_default(outcome=screening)
    assert result.final_model is not selected_before
    assert result.final_model.is_fitted is True
    assert result.report.final_fit_row_count == 12
    assert result.report.fit_partitions == ["train", "validation"]


def test_refit_uses_train_then_validation_order() -> None:
    _SpyAnomalyModel.instances.clear()
    result = _evaluate_default()
    final = result.final_model
    assert isinstance(final, _SpyAnomalyModel)
    assert final.fit_calls == 1
    assert final.seen_fit_height == 12
    assert final.seen_fit_columns == ["f1", "f2"]
    # row ids are absent from feature-only fit frame
    assert final.seen_fit_row_ids == []


def test_refit_concat_order_train_before_validation() -> None:
    split = _anomaly_split()
    template = _SpyAnomalyModel()
    registry = _registry_from_spy(template)
    _SpyAnomalyModel.instances.clear()

    class _OrderSpy(_SpyAnomalyModel):
        def fit(self, X: DataFrameLike, y: SeriesLike | None = None) -> Self:
            assert isinstance(X, pl.DataFrame)
            # Reconstruct expected concat by selecting features from train+val
            expected = pl.concat(
                [split.train.select(["f1", "f2"]), split.validation.select(["f1", "f2"])],
                how="vertical",
            )
            assert X.to_dict(as_series=False) == expected.to_dict(as_series=False)
            return super().fit(X, y)

    order_template = _OrderSpy()
    registry = _registry_from_spy(order_template)
    outcome = _make_screening_outcome()
    evaluator = AnomalyFinalEvaluator(registry)
    result = evaluator.evaluate(
        split,
        outcome,
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    assert result.final_model.fit_calls == 1  # type: ignore[attr-defined]


def test_test_not_used_in_fit() -> None:
    result = _evaluate_default()
    final = result.final_model
    assert isinstance(final, _SpyAnomalyModel)
    assert final.seen_fit_height == 12
    assert final.detect_calls == 1
    assert final.seen_detect_height == 4


def test_screening_selected_model_unchanged_after_refit() -> None:
    selected = _SpyAnomalyModel(
        mark_fitted=True,
        fit_row_count=8,
        feature_names=("f1", "f2"),
    )
    outcome = _make_screening_outcome(selected=selected)
    before_fit_calls = selected.fit_calls
    before_detect_calls = selected.detect_calls
    _ = _evaluate_default(outcome=outcome)
    assert selected.fit_calls == before_fit_calls
    assert selected.detect_calls == before_detect_calls
    assert selected.fit_row_count == 8


def test_registry_unchanged_after_evaluate() -> None:
    template = _SpyAnomalyModel()
    registry = _registry_from_spy(template)
    before = list(registry.list_specs(AnalysisTask.UNSUPERVISED_ANOMALY))
    _ = _evaluate_default(registry=registry)
    after = list(registry.list_specs(AnalysisTask.UNSUPERVISED_ANOMALY))
    assert before == after


# ---------------------------------------------------------------------------
# Refit disabled
# ---------------------------------------------------------------------------


def test_refit_false_uses_selected_model() -> None:
    selected = _SpyAnomalyModel(
        mark_fitted=True,
        fit_row_count=8,
        feature_names=("f1", "f2"),
        test_scores=[0.1, 0.2, 1.5, 1.8],
        test_raw=[1, 1, -1, -1],
    )
    outcome = _make_screening_outcome(selected=selected)
    policy = AnomalyFinalEvaluationPolicy(refit_on_train_validation=False)
    result = _evaluate_default(policy=policy, outcome=outcome)
    assert result.final_model is selected
    assert selected.fit_calls == 0
    assert result.report.fit_seconds == 0.0
    assert result.report.final_fit_row_count == 8
    assert result.report.fit_partitions == ["train"]
    assert any("not refit" in warning for warning in result.report.warnings)


def test_refit_false_rejects_unfitted_selected_model() -> None:
    selected = _SpyAnomalyModel(
        mark_fitted=False,
        fit_row_count=8,
        feature_names=("f1", "f2"),
    )
    outcome = _make_screening_outcome(selected=selected)
    policy = AnomalyFinalEvaluationPolicy(
        refit_on_train_validation=False,
        require_screening_consistency=False,
    )
    with pytest.raises(ProcessIntelligenceError, match="fitted"):
        _evaluate_default(policy=policy, outcome=outcome)


# ---------------------------------------------------------------------------
# Test detect and scored frame
# ---------------------------------------------------------------------------


def test_test_detect_exactly_once_and_scored_frame() -> None:
    split = _anomaly_split()
    split_before = copy.deepcopy(split.test)
    result = _evaluate_default(split=split)
    final = result.final_model
    assert isinstance(final, _SpyAnomalyModel)
    assert final.detect_calls == 1
    assert final.fit_calls == 1
    assert final.seen_detect_columns == ["f1", "f2"]

    scored = result.test_scored
    assert scored.height == split.test.height
    assert list(scored.columns) == [
        ORIGINAL_ROW_ID_COLUMN,
        "f1",
        "f2",
        "_anomaly_score",
        "_is_anomaly",
        "_anomaly_raw_prediction",
        "_data_partition",
    ]
    assert scored.schema["_anomaly_score"] == pl.Float64
    assert scored.schema["_is_anomaly"] == pl.Boolean
    assert scored.schema["_anomaly_raw_prediction"] == pl.Int64
    assert scored.schema["_data_partition"] == pl.String
    assert scored.get_column("_data_partition").to_list() == ["test"] * 4
    assert scored.get_column(ORIGINAL_ROW_ID_COLUMN).to_list() == [12, 13, 14, 15]
    assert scored.get_column(ORIGINAL_ROW_ID_COLUMN).dtype == pl.Int64
    assert len(set(scored.get_column(ORIGINAL_ROW_ID_COLUMN).to_list())) == 4
    assert split.test.equals(split_before)


def test_test_metrics_and_separation() -> None:
    result = _evaluate_default()
    metrics = result.report.test_metrics
    assert list(metrics.keys()) == [
        "test_anomaly_fraction",
        "test_score_min",
        "test_score_max",
        "test_score_mean",
        "test_score_std",
        "test_score_range",
    ]
    scores = np.asarray([0.1, 0.2, 1.5, 1.8], dtype=np.float64)
    assert metrics["test_anomaly_fraction"] == 0.5
    assert metrics["test_score_min"] == float(np.min(scores))
    assert metrics["test_score_max"] == float(np.max(scores))
    assert metrics["test_score_mean"] == float(np.mean(scores))
    assert metrics["test_score_std"] == float(np.std(scores, ddof=0))
    assert metrics["test_score_range"] == float(np.max(scores) - np.min(scores))
    assert all(math.isfinite(value) for value in metrics.values())

    mean_anomaly = (1.5 + 1.8) / 2.0
    mean_normal = (0.1 + 0.2) / 2.0
    expected_sep = (mean_anomaly - mean_normal) / max(
        metrics["test_score_std"],
        1e-12,
    )
    assert result.report.test_has_normal_and_anomaly is True
    assert result.report.test_score_separation == pytest.approx(expected_sep)


def test_separation_none_when_single_group() -> None:
    template = _SpyAnomalyModel(
        test_scores=[0.1, 0.2, 0.3, 0.4],
        test_raw=[1, 1, 1, 1],
    )
    registry = _registry_from_spy(template)
    result = _evaluate_default(registry=registry)
    assert result.report.test_has_normal_and_anomaly is False
    assert result.report.test_score_separation is None
    assert "test detection is degenerate" in result.report.quality_flags


def test_negative_separation_allowed() -> None:
    template = _SpyAnomalyModel(
        test_scores=[1.8, 1.5, 0.1, 0.2],
        test_raw=[-1, -1, 1, 1],
    )
    # Wait - with these scores, anomaly mean is high, normal is low => positive.
    # For negative: anomaly scores lower than normal.
    template = _SpyAnomalyModel(
        test_scores=[0.1, 0.2, 1.5, 1.8],
        test_raw=[-1, -1, 1, 1],
    )
    registry = _registry_from_spy(template)
    result = _evaluate_default(registry=registry)
    assert result.report.test_score_separation is not None
    assert result.report.test_score_separation < 0.0
    assert (
        "anomaly and normal score direction is inconsistent"
        in result.report.quality_flags
    )


def test_validation_metrics_copied_not_recomputed() -> None:
    candidate = _success_result(
        metrics=_success_metrics(validation_fraction=0.3, score_mean=0.7, score_std=0.2)
    )
    outcome = _make_screening_outcome(candidate=candidate)
    metrics_before = dict(outcome.summary.selected_metrics)
    result = _evaluate_default(outcome=outcome)
    assert result.report.validation_metrics == metrics_before
    assert result.report.validation_metrics is not outcome.summary.selected_metrics
    assert result.report.validation_score_separation == candidate.score_separation
    assert result.report.anomaly_fraction_shift == pytest.approx(
        abs(0.5 - 0.3)
    )
    assert result.report.score_mean_shift == pytest.approx(
        result.report.test_metrics["test_score_mean"] - 0.7
    )
    assert result.report.score_std_ratio == pytest.approx(
        result.report.test_metrics["test_score_std"] / max(0.2, 1e-12)
    )
    assert outcome.summary.selected_metrics == metrics_before


# ---------------------------------------------------------------------------
# Quality flags and warnings
# ---------------------------------------------------------------------------


def test_quality_flags_fraction_bounds_and_order() -> None:
    policy = AnomalyFinalEvaluationPolicy(
        minimum_anomaly_fraction=0.6,
        maximum_anomaly_fraction=0.9,
        maximum_fraction_shift=0.01,
    )
    # fraction 0.5 < 0.6, shift from validation 0.25 is 0.25 > 0.01
    result = _evaluate_default(policy=policy)
    flags = result.report.quality_flags
    assert flags[0] == "test anomaly fraction below expected minimum"
    assert "validation-test anomaly fraction shift is large" in flags


def test_quality_flag_above_maximum() -> None:
    policy = AnomalyFinalEvaluationPolicy(
        minimum_anomaly_fraction=0.001,
        maximum_anomaly_fraction=0.1,
    )
    result = _evaluate_default(policy=policy)
    assert "test anomaly fraction above expected maximum" in result.report.quality_flags


def test_quality_flag_nearly_constant_scores() -> None:
    template = _SpyAnomalyModel(
        test_scores=[0.5, 0.5, 0.5, 0.5],
        test_raw=[1, 1, -1, -1],
    )
    registry = _registry_from_spy(template)
    policy = AnomalyFinalEvaluationPolicy(minimum_score_std=0.1)
    result = _evaluate_default(registry=registry, policy=policy)
    assert "test anomaly scores are nearly constant" in result.report.quality_flags


def test_warning_order_and_contents() -> None:
    selected_candidate = _success_result(
        quality_flags=["anomaly fraction below expected minimum"],
        warnings=["candidate note"],
    )
    selected = _SpyAnomalyModel(
        mark_fitted=True,
        fit_row_count=8,
        feature_names=("f1", "f2"),
        test_scores=[0.1, 0.2, 0.3, 0.4],
        test_raw=[1, 1, 1, 1],
        detect_warnings=["detect note"],
    )
    outcome = _make_screening_outcome(
        selected=selected,
        candidate=selected_candidate,
    )
    policy = AnomalyFinalEvaluationPolicy(
        refit_on_train_validation=False,
        maximum_fraction_shift=0.01,
        warn_on_distribution_shift=True,
    )
    result = _evaluate_default(
        policy=policy,
        outcome=outcome,
        leakage_report=_warning_report(),
    )
    warnings = result.report.warnings
    assert warnings[0].startswith("LeakageReport contains WARNING")
    assert any("not refit" in item for item in warnings)
    assert any("quality flags" in item for item in warnings)
    assert any("fraction shift exceeds" in item for item in warnings)
    assert any("quality penalties" in item for item in warnings)
    assert "candidate note" in warnings
    assert "detect note" in warnings
    assert any("no detected anomalies" in item for item in warnings)
    assert len(warnings) == len(set(warnings))


def test_all_anomaly_warning() -> None:
    template = _SpyAnomalyModel(
        test_scores=[1.0, 1.1, 1.2, 1.3],
        test_raw=[-1, -1, -1, -1],
    )
    registry = _registry_from_spy(template)
    result = _evaluate_default(registry=registry)
    assert any("every row as an anomaly" in item for item in result.report.warnings)


def test_warn_on_distribution_shift_false_omits_warning() -> None:
    policy = AnomalyFinalEvaluationPolicy(
        maximum_fraction_shift=0.01,
        warn_on_distribution_shift=False,
    )
    result = _evaluate_default(policy=policy)
    assert all("fraction shift exceeds" not in item for item in result.report.warnings)


def test_quality_flags_do_not_change_model_name() -> None:
    policy = AnomalyFinalEvaluationPolicy(maximum_anomaly_fraction=0.1)
    result = _evaluate_default(policy=policy)
    assert result.report.quality_flags
    assert result.report.model_name == "Spy Anomaly"


# ---------------------------------------------------------------------------
# Detect validation errors
# ---------------------------------------------------------------------------


def test_detect_wrong_type_rejected() -> None:
    def _bad(_X: pl.DataFrame) -> object:
        return {"scores": [1.0]}

    template = _SpyAnomalyModel(detect_override=_bad)
    registry = _registry_from_spy(template)
    with pytest.raises(ProcessIntelligenceError, match="AnomalyDetectionResult"):
        _evaluate_default(registry=registry)


def test_detect_row_count_mismatch_rejected() -> None:
    def _bad(_X: pl.DataFrame) -> AnomalyDetectionResult:
        return AnomalyDetectionResult(
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

    template = _SpyAnomalyModel(detect_override=_bad)
    registry = _registry_from_spy(template)
    with pytest.raises(ProcessIntelligenceError, match="row_count"):
        _evaluate_default(registry=registry)


def test_detect_nan_score_rejected() -> None:
    def _bad(X: pl.DataFrame) -> AnomalyDetectionResult:
        n = X.height
        return AnomalyDetectionResult(
            scores=[float("nan")] + [0.1] * (n - 1),
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

    template = _SpyAnomalyModel(detect_override=_bad)
    registry = _registry_from_spy(template)
    with pytest.raises((ProcessIntelligenceError, ValidationError)):
        _evaluate_default(registry=registry)


def test_detect_bad_raw_prediction_rejected() -> None:
    def _bad(X: pl.DataFrame) -> AnomalyDetectionResult:
        n = X.height
        # Bypass AnomalyDetectionResult validation by constructing invalid via
        # model_construct if needed — but validation happens in pydantic.
        # Use override that returns a patched object.
        result = AnomalyDetectionResult(
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
        object.__setattr__(result, "raw_predictions", [0] + [1] * (n - 1))
        return result

    template = _SpyAnomalyModel(detect_override=_bad)
    registry = _registry_from_spy(template)
    with pytest.raises(ProcessIntelligenceError, match="raw_predictions"):
        _evaluate_default(registry=registry)


# ---------------------------------------------------------------------------
# Timing / metadata / outcome
# ---------------------------------------------------------------------------


def test_timings_and_evaluated_at() -> None:
    result = _evaluate_default()
    assert result.report.fit_seconds >= 0.0
    assert result.report.test_scoring_seconds >= 0.0
    assert result.report.total_seconds == pytest.approx(
        result.report.fit_seconds + result.report.test_scoring_seconds
    )
    assert math.isfinite(result.report.fit_seconds)
    assert result.report.evaluated_at.tzinfo is not None
    assert result.report.evaluated_at.utcoffset() is not None


def test_final_model_metadata_matches_report() -> None:
    result = _evaluate_default()
    metadata = result.final_model.get_metadata()
    assert metadata.model_name == result.report.model_name
    assert metadata.estimator_key == result.report.estimator_key  # type: ignore[attr-defined]
    assert metadata.task is AnalysisTask.UNSUPERVISED_ANOMALY
    assert list(metadata.features) == result.report.feature_columns
    assert metadata.fit_row_count == result.report.final_fit_row_count  # type: ignore[attr-defined]
    assert "estimator" not in result.report.model_fields
    reuse = result.final_model.detect(  # type: ignore[attr-defined]
        result.test_scored.select(["f1", "f2"])
    )
    assert isinstance(reuse, AnomalyDetectionResult)
    scores = result.final_model.score_samples(result.test_scored.select(["f1", "f2"]))
    assert len(scores) == result.report.test_row_count


def test_outcome_contains_only_final_model() -> None:
    result = _evaluate_default()
    fields = {field.name for field in dataclasses.fields(result)}
    assert fields == {"final_model", "test_scored", "report"}
    anomaly_count = int(result.test_scored.get_column("_is_anomaly").sum())
    assert anomaly_count / result.report.test_row_count == pytest.approx(
        result.report.test_metrics["test_anomaly_fraction"]
    )


# ---------------------------------------------------------------------------
# Immutability and determinism
# ---------------------------------------------------------------------------


def test_inputs_immutable_across_evaluate() -> None:
    split = _anomaly_split()
    features = ["f1", "f2"]
    outcome = _make_screening_outcome()
    leakage = _safe_report()
    split_before = copy.deepcopy(split)
    features_before = list(features)
    outcome_before = copy.deepcopy(outcome.summary)
    selected_before_fit = outcome.selected_model.fit_calls  # type: ignore[attr-defined]
    candidate_before = outcome.summary.candidate_results[0].model_copy(deep=True)
    registry = _registry_from_spy(_SpyAnomalyModel())
    specs_before = list(registry.list_specs(AnalysisTask.UNSUPERVISED_ANOMALY))

    evaluator = AnomalyFinalEvaluator(registry)
    first = evaluator.evaluate(
        split,
        outcome,
        feature_columns=features,
        leakage_report=leakage,
    )
    mutated = first.test_scored.with_columns(pl.lit(1).alias("_tmp"))
    assert "_tmp" in mutated.columns
    assert "_tmp" not in first.test_scored.columns

    second = evaluator.evaluate(
        split,
        outcome,
        feature_columns=features,
        leakage_report=leakage,
    )

    assert split.train.equals(split_before.train)
    assert split.validation.equals(split_before.validation)
    assert split.test.equals(split_before.test)
    assert features == features_before
    assert outcome.summary.model_dump() == outcome_before.model_dump()
    assert outcome.selected_model.fit_calls == selected_before_fit  # type: ignore[attr-defined]
    assert (
        outcome.summary.candidate_results[0].model_dump()
        == candidate_before.model_dump()
    )
    assert list(registry.list_specs(AnalysisTask.UNSUPERVISED_ANOMALY)) == specs_before
    assert first.final_model is not second.final_model
    assert first.report.test_metrics == second.report.test_metrics
    assert first.test_scored.select(
        ["_anomaly_score", "_is_anomaly", "_anomaly_raw_prediction"]
    ).equals(
        second.test_scored.select(
            ["_anomaly_score", "_is_anomaly", "_anomaly_raw_prediction"]
        )
    )


def test_test_value_change_does_not_change_selected_model_name() -> None:
    split_a = _anomaly_split()
    split_b = _anomaly_split(test_feature_offset=10.0)
    outcome = _make_screening_outcome()
    registry = _registry_from_spy(_SpyAnomalyModel())
    evaluator = AnomalyFinalEvaluator(registry)
    left = evaluator.evaluate(
        split_a,
        outcome,
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    right = evaluator.evaluate(
        split_b,
        outcome,
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    assert left.report.model_name == right.report.model_name == "Spy Anomaly"


def test_separate_evaluators_do_not_share_fitted_state() -> None:
    left = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    right = AnomalyFinalEvaluator(_registry_from_spy(_SpyAnomalyModel()))
    outcome_left = left.evaluate(
        _anomaly_split(),
        _make_screening_outcome(),
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    outcome_right = right.evaluate(
        _anomaly_split(),
        _make_screening_outcome(),
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    assert outcome_left.final_model is not outcome_right.final_model


# ---------------------------------------------------------------------------
# Integration with default anomaly registry + screening
# ---------------------------------------------------------------------------


def test_default_registry_screening_then_final_evaluation() -> None:
    split = _anomaly_split()
    features = ["f1", "f2"]
    registry = create_default_anomaly_model_registry(random_state=42)
    screener = UnsupervisedAnomalyModelScreener(registry)
    screening = screener.screen(
        split,
        feature_columns=features,
        leakage_report=_safe_report(features),
    )
    evaluator = AnomalyFinalEvaluator(registry)
    result = evaluator.evaluate(
        split,
        screening,
        feature_columns=features,
        leakage_report=_safe_report(features),
    )
    assert result.final_model is not screening.selected_model
    assert result.final_model.is_fitted is True
    assert result.report.final_fit_row_count == split.train.height + split.validation.height
    assert result.test_scored.height == split.test.height
    assert result.report.model_name == screening.summary.selected_model_name
    assert "_anomaly_score" in result.test_scored.columns
    # second call remains deterministic for same seed/registry factories
    result2 = evaluator.evaluate(
        split,
        screening,
        feature_columns=features,
        leakage_report=_safe_report(features),
    )
    assert result.test_scored.get_column("_anomaly_score").to_list() == (
        result2.test_scored.get_column("_anomaly_score").to_list()
    )
