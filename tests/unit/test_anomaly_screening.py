"""Unit tests for unsupervised anomaly model screening (Step 7D)."""

from __future__ import annotations

import copy
import dataclasses
import math
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
from process_intelligence.core.protocols import (
    BaseAnomalyModel,
    BaseIndustryProfile,
    DataFrameLike,
    SeriesLike,
)
from process_intelligence.core.schemas import (
    AnomalyEvent,
    DatasetMetadata,
    ExplanationResult,
    IndustryScore,
    ModelEvaluation,
    ModelMetadata,
    ModelSpec,
    PreprocessingRules,
    SchemaHints,
    ValidationIssue,
    VariableConstraint,
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
    AnomalyScreeningOutcome,
    AnomalyScreeningPolicy,
    AnomalyScreeningSummary,
    ModelRegistry,
    UnsupervisedAnomalyModelScreener,
    create_default_anomaly_model_registry,
)
from process_intelligence.models.anomaly_screening import (
    AnomalyCandidateRunStatus as DirectStatus,
)
from process_intelligence.models.anomaly_screening import (
    _compare_anomaly_candidates,
    _compute_label_free_metrics,
    _compute_quality_flags,
    _select_best_candidate,
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
) -> DatasetSplit:
    train = pl.DataFrame(
        {
            ORIGINAL_ROW_ID_COLUMN: [0, 1, 2, 3, 4, 5, 6, 7],
            "f1": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 5.0, 5.5],
            "f2": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 4.0, 4.5],
        }
    )
    validation = pl.DataFrame(
        {
            ORIGINAL_ROW_ID_COLUMN: [8, 9, 10, 11],
            "f1": [0.05, 0.25, 4.8, 5.2],
            "f2": [0.02, 0.12, 3.9, 4.2],
        }
    )
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
                ORIGINAL_ROW_ID_COLUMN: [12, 13],
                "f1": [0.1 + test_feature_offset, 5.0 + test_feature_offset],
                "f2": [0.05 + test_feature_offset, 4.0 + test_feature_offset],
            }
        )
        test_ids = [12, 13]
    return DatasetSplit(
        train=train,
        validation=validation,
        test=test,
        summary=_summary(
            train_ids=[0, 1, 2, 3, 4, 5, 6, 7],
            validation_ids=[8, 9, 10, 11],
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
    name: str,
    estimator_key: str,
    priority: int = 10,
    time_budget_seconds: float | None = 5.0,
) -> ModelSpec:
    return ModelSpec(
        name=name,
        task=AnalysisTask.UNSUPERVISED_ANOMALY,
        estimator_key=estimator_key,
        optional_dependencies=[],
        priority=priority,
        time_budget_seconds=time_budget_seconds,
    )


def _success_metrics(
    *,
    train_fraction: float = 0.125,
    validation_fraction: float = 0.25,
    gap: float | None = None,
    score_min: float = 0.0,
    score_max: float = 1.0,
    score_mean: float = 0.5,
    score_std: float = 0.4,
) -> dict[str, float]:
    if gap is None:
        gap = abs(validation_fraction - train_fraction)
    return {
        "train_anomaly_fraction": train_fraction,
        "validation_anomaly_fraction": validation_fraction,
        "anomaly_fraction_gap": gap,
        "validation_score_min": score_min,
        "validation_score_max": score_max,
        "validation_score_mean": score_mean,
        "validation_score_std": score_std,
        "validation_score_range": score_max - score_min,
    }


def _success_result(
    *,
    name: str = "Isolation Forest",
    estimator_key: str = "isolation_forest",
    registry_rank: int = 0,
    metrics: dict[str, float] | None = None,
    score_separation: float | None = 1.5,
    has_normal_and_anomaly: bool = True,
    quality_flags: list[str] | None = None,
    fit_seconds: float = 0.1,
    train_scoring_seconds: float = 0.05,
    validation_scoring_seconds: float = 0.05,
) -> AnomalyCandidateScreeningResult:
    flags = [] if quality_flags is None else list(quality_flags)
    metric_values = _success_metrics() if metrics is None else dict(metrics)
    return AnomalyCandidateScreeningResult(
        spec=_spec(name=name, estimator_key=estimator_key),
        status=AnomalyCandidateRunStatus.SUCCESS,
        registry_rank=registry_rank,
        metrics=metric_values,
        score_separation=score_separation,
        has_normal_and_anomaly=has_normal_and_anomaly,
        quality_penalty_count=len(flags),
        quality_flags=flags,
        fit_seconds=fit_seconds,
        train_scoring_seconds=train_scoring_seconds,
        validation_scoring_seconds=validation_scoring_seconds,
        total_seconds=fit_seconds + train_scoring_seconds + validation_scoring_seconds,
        warnings=[],
        error_type=None,
        error_message=None,
    )


def _failed_result(
    *,
    name: str = "Broken Model",
    estimator_key: str = "broken",
    registry_rank: int = 1,
    fit_seconds: float = 0.01,
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
        fit_seconds=fit_seconds,
        train_scoring_seconds=0.0,
        validation_scoring_seconds=0.0,
        total_seconds=fit_seconds,
        warnings=[],
        error_type="ValueError",
        error_message="boom",
    )


def _summary_from_results(
    results: list[AnomalyCandidateScreeningResult],
    *,
    selected: AnomalyCandidateScreeningResult,
) -> AnomalyScreeningSummary:
    successful = [
        item.spec.name
        for item in results
        if item.status is AnomalyCandidateRunStatus.SUCCESS
    ]
    failed = [
        item.spec.name
        for item in results
        if item.status is AnomalyCandidateRunStatus.FAILED
    ]
    return AnomalyScreeningSummary(
        task=AnalysisTask.UNSUPERVISED_ANOMALY,
        feature_columns=["f1", "f2"],
        candidate_results=list(results),
        selected_model_name=selected.spec.name,
        selected_estimator_key=selected.spec.estimator_key,
        selected_registry_rank=selected.registry_rank,
        selected_metrics=dict(selected.metrics),
        selected_score_separation=selected.score_separation,
        selected_quality_penalty_count=selected.quality_penalty_count,
        ranking_method=(
            "quality penalties, non-degenerate detection, standardized score "
            "separation, fraction stability, score variation, registry order"
        ),
        successful_model_names=successful,
        failed_model_names=failed,
        train_row_count=8,
        validation_row_count=4,
        test_row_count=2,
        warnings=[],
    )


class _CountingAnomalyModel(BaseAnomalyModel):
    """Deterministic test-double anomaly model with call counters."""

    instances: list[_CountingAnomalyModel] = []

    def __init__(
        self,
        *,
        name: str,
        estimator_key: str,
        train_raw: list[int],
        validation_raw: list[int],
        train_scores: list[float],
        validation_scores: list[float],
        fail_on: str | None = None,
        fail_exception: BaseException | None = None,
        custom_detect: Any | None = None,
    ) -> None:
        self._spec = _spec(name=name, estimator_key=estimator_key)
        self._train_raw = list(train_raw)
        self._validation_raw = list(validation_raw)
        self._train_scores = list(train_scores)
        self._validation_scores = list(validation_scores)
        self._fail_on = fail_on
        self._fail_exception = fail_exception or ValueError("forced failure")
        self._custom_detect = custom_detect
        self._is_fitted = False
        self._feature_names: tuple[str, ...] = ()
        self._fit_row_count = 0
        self.fit_calls = 0
        self.train_detect_calls = 0
        self.validation_detect_calls = 0
        self.test_detect_calls = 0
        self.predict_calls = 0
        self.seen_fit_columns: list[str] | None = None
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
        self.fit_calls += 1
        if self._fail_on == "fit":
            raise self._fail_exception
        assert y is None
        assert isinstance(X, pl.DataFrame)
        self.seen_fit_columns = list(X.columns)
        self._feature_names = tuple(X.columns)
        self._fit_row_count = X.height
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
            # Keep summaries consistent under float noise for constant scores.
            score_mean = min(max(score_mean, score_min), score_max)
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
            warnings=[],
        )

    def detect(self, X: DataFrameLike) -> AnomalyDetectionResult:
        assert isinstance(X, pl.DataFrame)
        if X.height == self._fit_row_count and self.train_detect_calls == 0:
            self.train_detect_calls += 1
            if self._fail_on == "train_detect":
                raise self._fail_exception
            if self._custom_detect is not None:
                return self._custom_detect("train", X)
            return self._build_result(
                scores=self._train_scores,
                raw=self._train_raw,
            )
        self.validation_detect_calls += 1
        if self._fail_on == "validation_detect":
            raise self._fail_exception
        if self._custom_detect is not None:
            return self._custom_detect("validation", X)
        return self._build_result(
            scores=self._validation_scores,
            raw=self._validation_raw,
        )

    def predict(self, X: DataFrameLike) -> np.ndarray:
        self.predict_calls += 1
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
            model_name=self._spec.name,
            version="test",
            task=AnalysisTask.UNSUPERVISED_ANOMALY,
            features=list(self._feature_names),
            estimator_key=self._spec.estimator_key,
            fit_row_count=self._fit_row_count,
            fitted=self._is_fitted,
        )


class _ExtendedMetadata(ModelMetadata):
    estimator_key: str = ""
    fit_row_count: int = 0
    fitted: bool = False


def _factory_for(model: _CountingAnomalyModel):
    def _factory() -> BaseAnomalyModel:
        clone = _CountingAnomalyModel(
            name=model._spec.name,
            estimator_key=model._spec.estimator_key,
            train_raw=model._train_raw,
            validation_raw=model._validation_raw,
            train_scores=model._train_scores,
            validation_scores=model._validation_scores,
            fail_on=model._fail_on,
            fail_exception=model._fail_exception,
            custom_detect=model._custom_detect,
        )
        return clone

    return _factory


def _registry_from_models(
    models: list[_CountingAnomalyModel],
) -> ModelRegistry:
    registry = ModelRegistry()
    for model in models:
        registry.register_factory(model._spec.estimator_key, _factory_for(model))
        registry.register(model._spec)
    return registry


class _DummyProfile(BaseIndustryProfile):
    @property
    def industry_name(self) -> str:
        return "dummy"

    def score_industry(self, metadata: DatasetMetadata) -> IndustryScore:
        _ = metadata
        return IndustryScore(
            industry_name="dummy",
            score=0.0,
            confidence=0.0,
            evidence=[],
            uncertain_factors=[],
            requires_user_confirmation=True,
        )

    def get_schema_hints(self) -> SchemaHints:
        return SchemaHints()

    def get_preprocessing_rules(self) -> PreprocessingRules:
        return PreprocessingRules()

    def get_default_model_candidates(self, task: AnalysisTask) -> list[ModelSpec]:
        _ = task
        return []

    def validate_physical_ranges(self, frame: DataFrameLike) -> list[ValidationIssue]:
        _ = frame
        return []

    def get_recommendation_constraints(self) -> list[VariableConstraint]:
        return []


# ---------------------------------------------------------------------------
# AnomalyCandidateRunStatus
# ---------------------------------------------------------------------------


def test_status_success_value() -> None:
    assert AnomalyCandidateRunStatus.SUCCESS == "SUCCESS"
    assert AnomalyCandidateRunStatus.SUCCESS.value == "SUCCESS"


def test_status_failed_value() -> None:
    assert AnomalyCandidateRunStatus.FAILED == "FAILED"
    assert DirectStatus.FAILED.value == "FAILED"


def test_status_from_string() -> None:
    assert AnomalyCandidateRunStatus("SUCCESS") is AnomalyCandidateRunStatus.SUCCESS
    assert AnomalyCandidateRunStatus("FAILED") is AnomalyCandidateRunStatus.FAILED


def test_status_invalid_string() -> None:
    with pytest.raises(ValueError):
        AnomalyCandidateRunStatus("SKIPPED")


def test_status_no_extra_members() -> None:
    assert set(AnomalyCandidateRunStatus) == {
        AnomalyCandidateRunStatus.SUCCESS,
        AnomalyCandidateRunStatus.FAILED,
    }


# ---------------------------------------------------------------------------
# AnomalyScreeningPolicy
# ---------------------------------------------------------------------------


def test_policy_defaults() -> None:
    policy = AnomalyScreeningPolicy()
    assert policy.maximum_candidates is None
    assert policy.candidate_time_budget_seconds is None
    assert policy.require_safe_leakage_report is True
    assert policy.require_validation_partition is True
    assert policy.continue_on_candidate_failure is True
    assert policy.minimum_validation_rows == 2
    assert policy.minimum_anomaly_fraction == 0.001
    assert policy.maximum_anomaly_fraction == 0.25
    assert policy.maximum_fraction_gap == 0.10
    assert policy.minimum_score_std == 1e-12
    assert policy.prefer_non_degenerate_detection is True


@pytest.mark.parametrize("value", [0, -1, True])
def test_policy_maximum_candidates_rejects(value: object) -> None:
    with pytest.raises(ValidationError):
        AnomalyScreeningPolicy(maximum_candidates=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0.0, -1.0, True, float("nan"), float("inf")])
def test_policy_time_budget_rejects(value: object) -> None:
    with pytest.raises(ValidationError):
        AnomalyScreeningPolicy(candidate_time_budget_seconds=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [1, True])
def test_policy_minimum_validation_rows_rejects(value: object) -> None:
    with pytest.raises(ValidationError):
        AnomalyScreeningPolicy(minimum_validation_rows=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minimum_anomaly_fraction", -0.1),
        ("maximum_anomaly_fraction", 1.1),
        ("maximum_fraction_gap", True),
        ("minimum_anomaly_fraction", float("nan")),
        ("maximum_anomaly_fraction", float("inf")),
    ],
)
def test_policy_fraction_rejects(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        AnomalyScreeningPolicy(**{field: value})


def test_policy_minimum_fraction_not_less_than_maximum() -> None:
    with pytest.raises(ValidationError):
        AnomalyScreeningPolicy(
            minimum_anomaly_fraction=0.3,
            maximum_anomaly_fraction=0.2,
        )


@pytest.mark.parametrize("value", [0.0, -1.0, True])
def test_policy_minimum_score_std_rejects(value: object) -> None:
    with pytest.raises(ValidationError):
        AnomalyScreeningPolicy(minimum_score_std=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field",
    [
        "require_safe_leakage_report",
        "require_validation_partition",
        "continue_on_candidate_failure",
        "prefer_non_degenerate_detection",
    ],
)
def test_policy_bool_fields_reject_non_bool(field: str) -> None:
    with pytest.raises(ValidationError):
        AnomalyScreeningPolicy(**{field: 1})


def test_policy_round_trip() -> None:
    policy = AnomalyScreeningPolicy(
        maximum_candidates=2,
        candidate_time_budget_seconds=12.5,
        minimum_validation_rows=3,
    )
    restored = AnomalyScreeningPolicy.model_validate(policy.model_dump())
    assert restored == policy


# ---------------------------------------------------------------------------
# AnomalyCandidateScreeningResult
# ---------------------------------------------------------------------------


def test_candidate_result_success_and_failed() -> None:
    success = _success_result()
    failed = _failed_result()
    assert success.status is AnomalyCandidateRunStatus.SUCCESS
    assert failed.status is AnomalyCandidateRunStatus.FAILED


def test_candidate_result_rejects_negative_rank() -> None:
    with pytest.raises(ValidationError):
        _success_result(registry_rank=-1)


def test_candidate_result_rejects_bad_metrics() -> None:
    with pytest.raises(ValidationError):
        _success_result(metrics={"": 1.0})
    with pytest.raises(ValidationError):
        _success_result(metrics={"train_anomaly_fraction": True})  # type: ignore[dict-item]
    with pytest.raises(ValidationError):
        metrics = _success_metrics()
        metrics["train_anomaly_fraction"] = float("nan")
        _success_result(metrics=metrics)
    with pytest.raises(ValidationError):
        metrics = _success_metrics()
        metrics["train_anomaly_fraction"] = float("inf")
        _success_result(metrics=metrics)


def test_candidate_result_rejects_bad_timing() -> None:
    with pytest.raises(ValidationError):
        _success_result(fit_seconds=-0.1)
    with pytest.raises(ValidationError):
        AnomalyCandidateScreeningResult(
            spec=_spec(name="A", estimator_key="a"),
            status=AnomalyCandidateRunStatus.SUCCESS,
            registry_rank=0,
            metrics=_success_metrics(),
            score_separation=1.0,
            has_normal_and_anomaly=True,
            quality_penalty_count=0,
            fit_seconds=0.1,
            train_scoring_seconds=0.1,
            validation_scoring_seconds=0.1,
            total_seconds=0.5,
        )


def test_candidate_result_success_consistency() -> None:
    metrics = _success_metrics()
    del metrics["validation_score_std"]
    with pytest.raises(ValidationError):
        _success_result(metrics=metrics)
    with pytest.raises(ValidationError):
        AnomalyCandidateScreeningResult(
            spec=_spec(name="A", estimator_key="a"),
            status=AnomalyCandidateRunStatus.SUCCESS,
            registry_rank=0,
            metrics=_success_metrics(),
            score_separation=1.0,
            has_normal_and_anomaly=True,
            quality_penalty_count=0,
            fit_seconds=0.1,
            train_scoring_seconds=0.1,
            validation_scoring_seconds=0.1,
            total_seconds=0.3,
            error_type="ValueError",
            error_message="x",
        )
    with pytest.raises(ValidationError):
        AnomalyCandidateScreeningResult(
            spec=_spec(name="A", estimator_key="a"),
            status=AnomalyCandidateRunStatus.SUCCESS,
            registry_rank=0,
            metrics=_success_metrics(),
            score_separation=1.0,
            has_normal_and_anomaly=True,
            quality_penalty_count=0,
            quality_flags=["a"],
            fit_seconds=0.1,
            train_scoring_seconds=0.1,
            validation_scoring_seconds=0.1,
            total_seconds=0.3,
        )
    with pytest.raises(ValidationError):
        _success_result(has_normal_and_anomaly=True, score_separation=None)


def test_candidate_result_failed_consistency() -> None:
    with pytest.raises(ValidationError):
        AnomalyCandidateScreeningResult(
            spec=_spec(name="A", estimator_key="a"),
            status=AnomalyCandidateRunStatus.FAILED,
            registry_rank=0,
            metrics=_success_metrics(),
            score_separation=None,
            has_normal_and_anomaly=False,
            quality_penalty_count=0,
            fit_seconds=0.0,
            train_scoring_seconds=0.0,
            validation_scoring_seconds=0.0,
            total_seconds=0.0,
            error_type="ValueError",
            error_message="x",
        )
    with pytest.raises(ValidationError):
        AnomalyCandidateScreeningResult(
            spec=_spec(name="A", estimator_key="a"),
            status=AnomalyCandidateRunStatus.FAILED,
            registry_rank=0,
            metrics={},
            score_separation=1.0,
            has_normal_and_anomaly=False,
            quality_penalty_count=0,
            fit_seconds=0.0,
            train_scoring_seconds=0.0,
            validation_scoring_seconds=0.0,
            total_seconds=0.0,
            error_type="ValueError",
            error_message="x",
        )
    with pytest.raises(ValidationError):
        AnomalyCandidateScreeningResult(
            spec=_spec(name="A", estimator_key="a"),
            status=AnomalyCandidateRunStatus.FAILED,
            registry_rank=0,
            metrics={},
            score_separation=None,
            has_normal_and_anomaly=False,
            quality_penalty_count=0,
            quality_flags=["x"],
            fit_seconds=0.0,
            train_scoring_seconds=0.0,
            validation_scoring_seconds=0.0,
            total_seconds=0.0,
            error_type="ValueError",
            error_message="x",
        )
    with pytest.raises(ValidationError):
        AnomalyCandidateScreeningResult(
            spec=_spec(name="A", estimator_key="a"),
            status=AnomalyCandidateRunStatus.FAILED,
            registry_rank=0,
            metrics={},
            score_separation=None,
            has_normal_and_anomaly=False,
            quality_penalty_count=0,
            fit_seconds=0.0,
            train_scoring_seconds=0.0,
            validation_scoring_seconds=0.0,
            total_seconds=0.0,
            error_type=None,
            error_message="x",
        )
    with pytest.raises(ValidationError):
        AnomalyCandidateScreeningResult(
            spec=_spec(name="A", estimator_key="a"),
            status=AnomalyCandidateRunStatus.FAILED,
            registry_rank=0,
            metrics={},
            score_separation=None,
            has_normal_and_anomaly=False,
            quality_penalty_count=0,
            fit_seconds=0.0,
            train_scoring_seconds=0.0,
            validation_scoring_seconds=0.0,
            total_seconds=0.0,
            error_type="ValueError",
            error_message="",
        )


def test_candidate_result_rejects_duplicate_lists() -> None:
    with pytest.raises(ValidationError):
        _success_result(quality_flags=["a", "a"])
    with pytest.raises(ValidationError):
        result = _success_result()
        AnomalyCandidateScreeningResult(
            **{
                **result.model_dump(),
                "warnings": ["w", "w"],
            }
        )


def test_candidate_result_round_trip() -> None:
    result = _success_result(quality_flags=["validation detection is degenerate"])
    restored = AnomalyCandidateScreeningResult.model_validate(result.model_dump())
    assert restored == result


# ---------------------------------------------------------------------------
# AnomalyScreeningSummary / Outcome
# ---------------------------------------------------------------------------


def test_summary_valid_and_rejects() -> None:
    selected = _success_result()
    summary = _summary_from_results([selected], selected=selected)
    assert summary.selected_model_name == "Isolation Forest"

    with pytest.raises(ValidationError):
        AnomalyScreeningSummary(
            **{**summary.model_dump(), "task": AnalysisTask.REGRESSION}
        )
    with pytest.raises(ValidationError):
        AnomalyScreeningSummary(**{**summary.model_dump(), "feature_columns": []})
    with pytest.raises(ValidationError):
        AnomalyScreeningSummary(
            **{**summary.model_dump(), "feature_columns": ["f1", "f1"]}
        )
    with pytest.raises(ValidationError):
        AnomalyScreeningSummary(
            **{
                **summary.model_dump(),
                "feature_columns": [ORIGINAL_ROW_ID_COLUMN, "f1"],
            }
        )
    with pytest.raises(ValidationError):
        AnomalyScreeningSummary(**{**summary.model_dump(), "selected_model_name": " "})
    with pytest.raises(ValidationError):
        AnomalyScreeningSummary(
            **{**summary.model_dump(), "selected_estimator_key": ""}
        )
    with pytest.raises(ValidationError):
        AnomalyScreeningSummary(**{**summary.model_dump(), "ranking_method": ""})
    with pytest.raises(ValidationError):
        AnomalyScreeningSummary(**{**summary.model_dump(), "selected_metrics": {}})
    with pytest.raises(ValidationError):
        AnomalyScreeningSummary(
            **{**summary.model_dump(), "selected_model_name": "Missing"}
        )
    with pytest.raises(ValidationError):
        AnomalyScreeningSummary(
            **{**summary.model_dump(), "selected_registry_rank": 9}
        )
    with pytest.raises(ValidationError):
        AnomalyScreeningSummary(
            **{
                **summary.model_dump(),
                "selected_metrics": {
                    **selected.metrics,
                    "train_anomaly_fraction": 0.999,
                },
            }
        )
    with pytest.raises(ValidationError):
        AnomalyScreeningSummary(
            **{**summary.model_dump(), "selected_score_separation": 99.0}
        )
    with pytest.raises(ValidationError):
        AnomalyScreeningSummary(
            **{
                **summary.model_dump(),
                "successful_model_names": ["Isolation Forest", "Isolation Forest"],
            }
        )
    with pytest.raises(ValidationError):
        failed = _failed_result(name="Broken")
        AnomalyScreeningSummary(
            **{
                **_summary_from_results(
                    [selected, failed], selected=selected
                ).model_dump(),
                "failed_model_names": ["Broken", "Broken"],
            }
        )
    with pytest.raises(ValidationError):
        AnomalyScreeningSummary(
            **{
                **summary.model_dump(),
                "successful_model_names": ["Isolation Forest"],
                "failed_model_names": ["Isolation Forest"],
            }
        )
    with pytest.raises(ValidationError):
        AnomalyScreeningSummary(**{**summary.model_dump(), "train_row_count": 0})
    with pytest.raises(ValidationError):
        AnomalyScreeningSummary(**{**summary.model_dump(), "validation_row_count": 0})
    with pytest.raises(ValidationError):
        AnomalyScreeningSummary(**{**summary.model_dump(), "warnings": ["a", "a"]})


def test_summary_round_trip() -> None:
    selected = _success_result()
    summary = _summary_from_results([selected], selected=selected)
    restored = AnomalyScreeningSummary.model_validate(summary.model_dump())
    assert restored == summary


def test_outcome_frozen_slots() -> None:
    model = create_default_anomaly_model_registry().instantiate(
        _spec(name="Isolation Forest", estimator_key="isolation_forest")
    )
    assert isinstance(model, BaseAnomalyModel)
    selected = _success_result()
    summary = _summary_from_results([selected], selected=selected)
    outcome = AnomalyScreeningOutcome(selected_model=model, summary=summary)
    assert dataclasses.is_dataclass(outcome)
    assert outcome.__slots__ == ("selected_model", "summary")
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.summary = summary  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Screener construction and input validation
# ---------------------------------------------------------------------------


def test_screener_construction_and_policy_isolation() -> None:
    registry = create_default_anomaly_model_registry()
    policy = AnomalyScreeningPolicy(maximum_candidates=2)
    screener = UnsupervisedAnomalyModelScreener(registry, policy=policy)
    policy.maximum_candidates = 1
    split = _anomaly_split()
    outcome = screener.screen(
        split,
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    assert len(outcome.summary.candidate_results) == 2

    with pytest.raises(TypeError):
        UnsupervisedAnomalyModelScreener("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        UnsupervisedAnomalyModelScreener(registry, policy="bad")  # type: ignore[arg-type]


def test_screen_input_validation() -> None:
    registry = create_default_anomaly_model_registry()
    screener = UnsupervisedAnomalyModelScreener(registry)
    split = _anomaly_split()
    report = _safe_report()

    with pytest.raises(TypeError):
        screener.screen(
            "bad",  # type: ignore[arg-type]
            feature_columns=["f1", "f2"],
            leakage_report=report,
        )

    bad_split = DatasetSplit(
        train=split.train,
        validation={"not": "frame"},  # type: ignore[arg-type]
        test=split.test,
        summary=split.summary,
    )
    with pytest.raises(TypeError):
        screener.screen(
            bad_split,
            feature_columns=["f1", "f2"],
            leakage_report=report,
        )

    schema_mismatch = DatasetSplit(
        train=split.train,
        validation=split.validation.rename({"f1": "fx"}),
        test=split.test,
        summary=split.summary,
    )
    with pytest.raises(DataValidationError):
        screener.screen(
            schema_mismatch,
            feature_columns=["f1", "f2"],
            leakage_report=report,
        )

    dtype_mismatch = DatasetSplit(
        train=split.train,
        validation=split.validation.with_columns(pl.col("f1").cast(pl.Int64)),
        test=split.test.with_columns(pl.col("f1").cast(pl.Int64)),
        summary=split.summary,
    )
    with pytest.raises(DataValidationError):
        screener.screen(
            dtype_mismatch,
            feature_columns=["f1", "f2"],
            leakage_report=report,
        )

    no_id = DatasetSplit(
        train=split.train.drop(ORIGINAL_ROW_ID_COLUMN),
        validation=split.validation.drop(ORIGINAL_ROW_ID_COLUMN),
        test=split.test.drop(ORIGINAL_ROW_ID_COLUMN),
        summary=split.summary,
    )
    with pytest.raises(DataValidationError):
        screener.screen(
            no_id,
            feature_columns=["f1", "f2"],
            leakage_report=report,
        )

    with pytest.raises(TypeError):
        screener.screen(split, feature_columns="f1", leakage_report=report)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        screener.screen(split, feature_columns=b"f1", leakage_report=report)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        screener.screen(split, feature_columns=[1], leakage_report=report)  # type: ignore[list-item]
    with pytest.raises(DataValidationError):
        screener.screen(split, feature_columns=[" "], leakage_report=report)
    with pytest.raises(DataValidationError):
        screener.screen(split, feature_columns=[], leakage_report=report)
    with pytest.raises(DataValidationError):
        screener.screen(
            split,
            feature_columns=["f1", "f1"],
            leakage_report=report,
        )
    with pytest.raises(DataValidationError):
        screener.screen(
            split,
            feature_columns=[ORIGINAL_ROW_ID_COLUMN],
            leakage_report=report,
        )
    with pytest.raises(DataValidationError):
        screener.screen(
            split,
            feature_columns=["missing"],
            leakage_report=_safe_report(["missing"]),
        )
    with pytest.raises(TypeError):
        screener.screen(
            split,
            feature_columns=["f1", "f2"],
            leakage_report="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        screener.screen(
            split,
            feature_columns=["f1", "f2"],
            leakage_report=report,
            industry_profile="bad",  # type: ignore[arg-type]
        )


def test_leakage_gate() -> None:
    registry = create_default_anomaly_model_registry()
    screener = UnsupervisedAnomalyModelScreener(registry)
    split = _anomaly_split()

    with pytest.raises(DataLeakageError, match="IDENTIFIER_INCLUDED_AS_FEATURE"):
        screener.screen(
            split,
            feature_columns=["f1", "f2"],
            leakage_report=_blocker_report(),
        )

    outcome = screener.screen(
        split,
        feature_columns=["f1", "f2"],
        leakage_report=_warning_report(),
    )
    assert any("WARNING" in warning for warning in outcome.summary.warnings)

    outcome_safe = screener.screen(
        split,
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    assert outcome_safe.summary.selected_model_name

    permissive = UnsupervisedAnomalyModelScreener(
        registry,
        policy=AnomalyScreeningPolicy(require_safe_leakage_report=False),
    )
    outcome_blocker = permissive.screen(
        split,
        feature_columns=["f1", "f2"],
        leakage_report=_blocker_report(),
    )
    assert outcome_blocker.summary.selected_model_name

    with pytest.raises(DataValidationError):
        screener.screen(
            split,
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(["f2", "f1"]),
        )
    with pytest.raises(DataValidationError):
        screener.screen(
            split,
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(["f1"]),
        )

    report = _safe_report()
    before = report.model_dump()
    screener.screen(split, feature_columns=["f1", "f2"], leakage_report=report)
    assert report.model_dump() == before


def test_minimum_data_conditions() -> None:
    registry = create_default_anomaly_model_registry()
    screener = UnsupervisedAnomalyModelScreener(registry)
    split = _anomaly_split()

    empty_train = DatasetSplit(
        train=split.train.clear(),
        validation=split.validation,
        test=split.test,
        summary=split.summary,
    )
    with pytest.raises(InsufficientDataError):
        screener.screen(
            empty_train,
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )

    short_validation = DatasetSplit(
        train=split.train,
        validation=split.validation.head(1),
        test=split.test,
        summary=split.summary,
    )
    with pytest.raises(InsufficientDataError):
        screener.screen(
            short_validation,
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )

    empty_test_split = _anomaly_split(empty_test=True)
    outcome = screener.screen(
        empty_test_split,
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    assert outcome.summary.test_row_count == 0

    # validation shortfall is not rescued by a large test partition
    with pytest.raises(InsufficientDataError):
        UnsupervisedAnomalyModelScreener(
            registry,
            policy=AnomalyScreeningPolicy(require_validation_partition=False),
        ).screen(
            short_validation,
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )


# ---------------------------------------------------------------------------
# Candidate lookup / execution
# ---------------------------------------------------------------------------


def test_candidate_lookup_and_limits() -> None:
    registry = create_default_anomaly_model_registry()
    registry.register(
        ModelSpec(
            name="Linear Regression",
            task=AnalysisTask.REGRESSION,
            estimator_key="linear_regression",
            optional_dependencies=[],
            priority=1,
            time_budget_seconds=5.0,
        )
    )
    before = registry.list_specs()
    screener = UnsupervisedAnomalyModelScreener(
        registry,
        policy=AnomalyScreeningPolicy(maximum_candidates=2),
    )
    outcome = screener.screen(
        _anomaly_split(),
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    assert len(outcome.summary.candidate_results) == 2
    assert all(
        item.spec.task is AnalysisTask.UNSUPERVISED_ANOMALY
        for item in outcome.summary.candidate_results
    )
    assert outcome.summary.candidate_results[0].registry_rank == 0
    assert outcome.summary.candidate_results[1].registry_rank == 1
    assert registry.list_specs() == before

    empty = ModelRegistry()
    with pytest.raises(ProcessIntelligenceError):
        UnsupervisedAnomalyModelScreener(empty).screen(
            _anomaly_split(),
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )


def test_time_budget_is_forwarded() -> None:
    registry = create_default_anomaly_model_registry()
    seen: dict[str, object] = {}

    original = registry.get_candidates

    def _wrapped(
        task: AnalysisTask,
        *,
        industry_profile: BaseIndustryProfile | None = None,
        time_budget_seconds: float | None = None,
    ) -> tuple[ModelSpec, ...]:
        seen["task"] = task
        seen["time_budget_seconds"] = time_budget_seconds
        return original(
            task,
            industry_profile=industry_profile,
            time_budget_seconds=time_budget_seconds,
        )

    registry.get_candidates = _wrapped  # type: ignore[method-assign]
    UnsupervisedAnomalyModelScreener(
        registry,
        policy=AnomalyScreeningPolicy(candidate_time_budget_seconds=16.0),
    ).screen(
        _anomaly_split(),
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    assert seen["task"] is AnalysisTask.UNSUPERVISED_ANOMALY
    assert seen["time_budget_seconds"] == 16.0


def test_candidate_execution_counts_and_feature_order() -> None:
    _CountingAnomalyModel.instances.clear()
    template = _CountingAnomalyModel(
        name="Counter",
        estimator_key="counter",
        train_raw=[1, 1, 1, 1, -1, 1, 1, 1],
        validation_raw=[1, 1, -1, 1],
        train_scores=[0.1, 0.2, 0.15, 0.12, 0.9, 0.2, 0.18, 0.11],
        validation_scores=[0.1, 0.2, 0.85, 0.15],
    )
    registry = _registry_from_models([template])
    split = _anomaly_split()
    features = ["f2", "f1"]
    features_before = list(features)
    train_before = split.train.clone()
    validation_before = split.validation.clone()
    test_before = split.test.clone()

    outcome = UnsupervisedAnomalyModelScreener(registry).screen(
        split,
        feature_columns=features,
        leakage_report=_safe_report(features),
    )
    created = [
        item
        for item in _CountingAnomalyModel.instances
        if item.fit_calls > 0
    ]
    assert len(created) == 1
    model = created[0]
    assert model.fit_calls == 1
    assert model.train_detect_calls == 1
    assert model.validation_detect_calls == 1
    assert model.test_detect_calls == 0
    assert model.predict_calls == 0
    assert model.seen_fit_columns == ["f2", "f1"]
    assert ORIGINAL_ROW_ID_COLUMN not in (model.seen_fit_columns or [])
    assert features == features_before
    assert split.train.equals(train_before)
    assert split.validation.equals(validation_before)
    assert split.test.equals(test_before)
    assert isinstance(outcome.selected_model, BaseAnomalyModel)


def test_detect_result_validation_failures() -> None:
    def _bad_type(_partition: str, _frame: pl.DataFrame) -> object:
        return {"not": "result"}

    def _bad_row_count(_partition: str, frame: pl.DataFrame) -> AnomalyDetectionResult:
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
            warnings=[],
        )

    def _nan_score(_partition: str, frame: pl.DataFrame) -> AnomalyDetectionResult:
        n = frame.height
        scores = [0.1] * n
        scores[0] = float("nan")
        return AnomalyDetectionResult.model_construct(
            scores=scores,
            is_anomaly=[False] * n,
            raw_predictions=[1] * n,
            threshold=0.0,
            row_count=n,
            anomaly_count=0,
            anomaly_fraction=0.0,
            score_min=0.1,
            score_max=0.1,
            score_mean=0.1,
            warnings=[],
        )

    good = _CountingAnomalyModel(
        name="Good",
        estimator_key="good",
        train_raw=[1, 1, 1, 1, -1, 1, 1, 1],
        validation_raw=[1, 1, -1, 1],
        train_scores=[0.1, 0.2, 0.15, 0.12, 0.9, 0.2, 0.18, 0.11],
        validation_scores=[0.1, 0.2, 0.85, 0.15],
    )

    for custom, key in (
        (_bad_type, "bad_type"),
        (_bad_row_count, "bad_rows"),
        (_nan_score, "nan_score"),
    ):
        bad = _CountingAnomalyModel(
            name=f"Bad-{key}",
            estimator_key=key,
            train_raw=[1] * 8,
            validation_raw=[1] * 4,
            train_scores=[0.1] * 8,
            validation_scores=[0.1] * 4,
            custom_detect=custom,
        )
        registry = _registry_from_models([bad, good])
        outcome = UnsupervisedAnomalyModelScreener(registry).screen(
            _anomaly_split(),
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )
        assert outcome.summary.failed_model_names[0].startswith("Bad-")
        assert outcome.summary.selected_model_name == "Good"


def test_metric_and_separation_helpers() -> None:
    train = AnomalyDetectionResult(
        scores=[0.1, 0.2, 0.9, 0.15],
        is_anomaly=[False, False, True, False],
        raw_predictions=[1, 1, -1, 1],
        threshold=0.0,
        row_count=4,
        anomaly_count=1,
        anomaly_fraction=0.25,
        score_min=0.1,
        score_max=0.9,
        score_mean=0.3375,
        warnings=[],
    )
    validation = AnomalyDetectionResult(
        scores=[0.0, 1.0, 2.0, 3.0],
        is_anomaly=[False, False, True, True],
        raw_predictions=[1, 1, -1, -1],
        threshold=0.0,
        row_count=4,
        anomaly_count=2,
        anomaly_fraction=0.5,
        score_min=0.0,
        score_max=3.0,
        score_mean=1.5,
        warnings=[],
    )
    metrics, separation, has_groups = _compute_label_free_metrics(
        train_result=train,
        validation_result=validation,
        minimum_score_std=1e-12,
    )
    assert list(metrics.keys()) == [
        "train_anomaly_fraction",
        "validation_anomaly_fraction",
        "anomaly_fraction_gap",
        "validation_score_min",
        "validation_score_max",
        "validation_score_mean",
        "validation_score_std",
        "validation_score_range",
    ]
    assert metrics["train_anomaly_fraction"] == 0.25
    assert metrics["validation_anomaly_fraction"] == 0.5
    assert metrics["anomaly_fraction_gap"] == 0.25
    assert metrics["validation_score_min"] == 0.0
    assert metrics["validation_score_max"] == 3.0
    assert metrics["validation_score_mean"] == 1.5
    assert metrics["validation_score_std"] == float(
        np.std([0.0, 1.0, 2.0, 3.0], ddof=0)
    )
    assert metrics["validation_score_range"] == 3.0
    assert has_groups is True
    assert separation is not None
    mean_anomaly = 2.5
    mean_normal = 0.5
    expected = (mean_anomaly - mean_normal) / metrics["validation_score_std"]
    assert math.isclose(separation, expected, abs_tol=1e-12)
    assert all(math.isfinite(value) for value in metrics.values())

    constant = AnomalyDetectionResult(
        scores=[1.0, 1.0, 1.0, 1.0],
        is_anomaly=[False, False, True, True],
        raw_predictions=[1, 1, -1, -1],
        threshold=0.0,
        row_count=4,
        anomaly_count=2,
        anomaly_fraction=0.5,
        score_min=1.0,
        score_max=1.0,
        score_mean=1.0,
        warnings=[],
    )
    _, sep_const, _ = _compute_label_free_metrics(
        train_result=train,
        validation_result=constant,
        minimum_score_std=1e-6,
    )
    assert sep_const == 0.0

    all_normal = AnomalyDetectionResult(
        scores=[0.1, 0.2, 0.3, 0.4],
        is_anomaly=[False, False, False, False],
        raw_predictions=[1, 1, 1, 1],
        threshold=0.0,
        row_count=4,
        anomaly_count=0,
        anomaly_fraction=0.0,
        score_min=0.1,
        score_max=0.4,
        score_mean=0.25,
        warnings=[],
    )
    _, sep_none, has_none = _compute_label_free_metrics(
        train_result=train,
        validation_result=all_normal,
        minimum_score_std=1e-12,
    )
    assert sep_none is None
    assert has_none is False


def test_quality_flags_order_and_penalty() -> None:
    policy = AnomalyScreeningPolicy(
        minimum_anomaly_fraction=0.1,
        maximum_anomaly_fraction=0.2,
        maximum_fraction_gap=0.05,
        minimum_score_std=0.5,
    )
    metrics = _success_metrics(
        train_fraction=0.0,
        validation_fraction=0.01,
        gap=0.2,
        score_std=0.01,
    )
    flags = _compute_quality_flags(
        metrics=metrics,
        has_normal_and_anomaly=False,
        score_separation=-1.0,
        policy=policy,
    )
    assert flags == [
        "anomaly fraction below expected minimum",
        "train-validation anomaly fraction unstable",
        "validation anomaly scores nearly constant",
        "validation detection is degenerate",
        "anomaly and normal score direction is inconsistent",
    ]
    high = _success_metrics(validation_fraction=0.9, gap=0.0, score_std=1.0)
    high_flags = _compute_quality_flags(
        metrics=high,
        has_normal_and_anomaly=True,
        score_separation=1.0,
        policy=policy,
    )
    assert high_flags[0] == "anomaly fraction above expected maximum"


def test_ranking_rules() -> None:
    low_penalty = _success_result(
        name="LowPenalty",
        estimator_key="low",
        registry_rank=1,
        quality_flags=[],
        score_separation=1.0,
    )
    high_penalty = _success_result(
        name="HighPenalty",
        estimator_key="high",
        registry_rank=0,
        quality_flags=["validation detection is degenerate"],
        score_separation=5.0,
        has_normal_and_anomaly=False,
    )
    assert (
        _select_best_candidate(
            [high_penalty, low_penalty],
            prefer_non_degenerate=True,
        ).spec.name
        == "LowPenalty"
    )

    degenerate = _success_result(
        name="Degenerate",
        estimator_key="deg",
        registry_rank=0,
        score_separation=1.0,
        has_normal_and_anomaly=False,
        quality_flags=["validation detection is degenerate"],
        metrics=_success_metrics(score_std=2.0, gap=0.0),
    )
    non_deg = _success_result(
        name="NonDeg",
        estimator_key="non",
        registry_rank=1,
        score_separation=1.0,
        has_normal_and_anomaly=True,
        quality_flags=["validation detection is degenerate"],
        metrics=_success_metrics(score_std=0.1, gap=0.2),
    )
    assert (
        _select_best_candidate(
            [degenerate, non_deg],
            prefer_non_degenerate=True,
        ).spec.name
        == "NonDeg"
    )
    assert (
        _select_best_candidate(
            [degenerate, non_deg],
            prefer_non_degenerate=False,
        ).spec.name
        == "Degenerate"
    )

    larger_sep = _success_result(
        name="LargeSep",
        estimator_key="large",
        registry_rank=1,
        score_separation=3.0,
        metrics=_success_metrics(gap=0.2, score_std=0.1),
    )
    smaller_sep = _success_result(
        name="SmallSep",
        estimator_key="small",
        registry_rank=0,
        score_separation=1.0,
        metrics=_success_metrics(gap=0.0, score_std=2.0),
    )
    assert (
        _select_best_candidate(
            [smaller_sep, larger_sep],
            prefer_non_degenerate=True,
        ).spec.name
        == "LargeSep"
    )

    none_sep = _success_result(
        name="NoneSep",
        estimator_key="none",
        registry_rank=0,
        score_separation=None,
        has_normal_and_anomaly=False,
        quality_flags=[],
    )
    assert (
        _compare_anomaly_candidates(
            larger_sep,
            none_sep,
            prefer_non_degenerate=False,
        )
        < 0
    )

    tight_gap = _success_result(
        name="Tight",
        estimator_key="tight",
        registry_rank=1,
        score_separation=1.0,
        metrics=_success_metrics(gap=0.01, score_std=0.1),
    )
    loose_gap = _success_result(
        name="Loose",
        estimator_key="loose",
        registry_rank=0,
        score_separation=1.0,
        metrics=_success_metrics(gap=0.2, score_std=2.0),
    )
    assert (
        _select_best_candidate(
            [loose_gap, tight_gap],
            prefer_non_degenerate=True,
        ).spec.name
        == "Tight"
    )

    high_std = _success_result(
        name="HighStd",
        estimator_key="hstd",
        registry_rank=1,
        score_separation=1.0,
        metrics=_success_metrics(gap=0.1, score_std=2.0),
    )
    low_std = _success_result(
        name="LowStd",
        estimator_key="lstd",
        registry_rank=0,
        score_separation=1.0,
        metrics=_success_metrics(gap=0.1, score_std=0.2),
    )
    assert (
        _select_best_candidate(
            [low_std, high_std],
            prefer_non_degenerate=True,
        ).spec.name
        == "HighStd"
    )

    early = _success_result(
        name="Early",
        estimator_key="early",
        registry_rank=0,
        score_separation=1.0,
        metrics=_success_metrics(gap=0.1, score_std=1.0),
    )
    late = _success_result(
        name="Late",
        estimator_key="late",
        registry_rank=1,
        score_separation=1.0,
        metrics=_success_metrics(gap=0.1, score_std=1.0),
    )
    assert (
        _select_best_candidate(
            [late, early],
            prefer_non_degenerate=True,
        ).spec.name
        == "Early"
    )


def test_failure_isolation_and_all_failed() -> None:
    good = _CountingAnomalyModel(
        name="Good",
        estimator_key="good",
        train_raw=[1, 1, 1, 1, -1, 1, 1, 1],
        validation_raw=[1, 1, -1, 1],
        train_scores=[0.1, 0.2, 0.15, 0.12, 0.9, 0.2, 0.18, 0.11],
        validation_scores=[0.1, 0.2, 0.85, 0.15],
    )
    fit_fail = _CountingAnomalyModel(
        name="FitFail",
        estimator_key="fit_fail",
        train_raw=[1] * 8,
        validation_raw=[1] * 4,
        train_scores=[0.1] * 8,
        validation_scores=[0.1] * 4,
        fail_on="fit",
        fail_exception=ValueError("fit exploded"),
    )
    train_detect_fail = _CountingAnomalyModel(
        name="TrainDetectFail",
        estimator_key="train_fail",
        train_raw=[1] * 8,
        validation_raw=[1] * 4,
        train_scores=[0.1] * 8,
        validation_scores=[0.1] * 4,
        fail_on="train_detect",
        fail_exception=TypeError("train detect exploded"),
    )
    validation_detect_fail = _CountingAnomalyModel(
        name="ValDetectFail",
        estimator_key="val_fail",
        train_raw=[1] * 8,
        validation_raw=[1] * 4,
        train_scores=[0.1] * 8,
        validation_scores=[0.1] * 4,
        fail_on="validation_detect",
        fail_exception=ArithmeticError("val detect exploded"),
    )

    registry = _registry_from_models(
        [fit_fail, train_detect_fail, validation_detect_fail, good]
    )
    outcome = UnsupervisedAnomalyModelScreener(registry).screen(
        _anomaly_split(),
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    assert outcome.summary.selected_model_name == "Good"
    assert outcome.summary.failed_model_names == [
        "FitFail",
        "TrainDetectFail",
        "ValDetectFail",
    ]
    failed = [
        item
        for item in outcome.summary.candidate_results
        if item.status is AnomalyCandidateRunStatus.FAILED
    ]
    assert [item.error_type for item in failed] == [
        "ValueError",
        "TypeError",
        "ArithmeticError",
    ]
    assert all(item.error_message for item in failed)
    assert all(item.total_seconds >= 0.0 for item in failed)
    assert outcome.summary.candidate_results[-1].spec.name == "Good"

    strict = UnsupervisedAnomalyModelScreener(
        _registry_from_models([fit_fail, good]),
        policy=AnomalyScreeningPolicy(continue_on_candidate_failure=False),
    )
    with pytest.raises(ValueError, match="fit exploded"):
        strict.screen(
            _anomaly_split(),
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )

    all_bad = _registry_from_models([fit_fail, train_detect_fail])
    with pytest.raises(ProcessIntelligenceError) as exc_info:
        UnsupervisedAnomalyModelScreener(all_bad).screen(
            _anomaly_split(),
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )
    message = str(exc_info.value)
    assert "FitFail" in message
    assert "TrainDetectFail" in message
    assert "ValueError" in message
    assert "TypeError" in message
    assert "candidate_count=2" in message


def test_summary_warnings_and_selected_fields() -> None:
    good = _CountingAnomalyModel(
        name="Good",
        estimator_key="good",
        train_raw=[1] * 8,
        validation_raw=[1, 1, 1, 1],
        train_scores=[0.1] * 8,
        validation_scores=[0.1, 0.1, 0.1, 0.1],
    )
    bad = _CountingAnomalyModel(
        name="Bad",
        estimator_key="bad",
        train_raw=[1] * 8,
        validation_raw=[1] * 4,
        train_scores=[0.1] * 8,
        validation_scores=[0.1] * 4,
        fail_on="fit",
    )
    outcome = UnsupervisedAnomalyModelScreener(
        _registry_from_models([bad, good])
    ).screen(
        _anomaly_split(),
        feature_columns=["f1", "f2"],
        leakage_report=_warning_report(),
    )
    warnings = outcome.summary.warnings
    assert warnings[0].startswith("LeakageReport contains WARNING")
    assert "failed" in warnings[1].lower()
    assert any("quality penalties" in warning.lower() for warning in warnings)
    assert any("degenerate" in warning.lower() for warning in warnings)
    assert any("score separation" in warning.lower() for warning in warnings)
    assert len(warnings) == len(set(warnings))
    assert outcome.summary.selected_metrics == (
        outcome.summary.candidate_results[1].metrics
    )
    assert outcome.summary.selected_score_separation is None
    assert outcome.summary.selected_quality_penalty_count >= 1
    assert outcome.summary.successful_model_names == ["Good"]
    assert outcome.summary.failed_model_names == ["Bad"]
    assert outcome.summary.train_row_count == 8
    assert outcome.summary.validation_row_count == 4
    assert outcome.summary.test_row_count == 2
    assert outcome.summary.ranking_method


def test_test_partition_unused() -> None:
    registry = create_default_anomaly_model_registry(random_state=7)
    screener = UnsupervisedAnomalyModelScreener(registry)
    base = _anomaly_split(test_feature_offset=0.0)
    shifted = _anomaly_split(test_feature_offset=1000.0)
    outcome_a = screener.screen(
        base,
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    outcome_b = screener.screen(
        shifted,
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    assert outcome_a.summary.selected_model_name == (
        outcome_b.summary.selected_model_name
    )
    assert outcome_a.summary.selected_metrics == outcome_b.summary.selected_metrics
    assert [
        (item.spec.name, item.status, item.metrics, item.error_type)
        for item in outcome_a.summary.candidate_results
    ] == [
        (item.spec.name, item.status, item.metrics, item.error_type)
        for item in outcome_b.summary.candidate_results
    ]
    assert outcome_a.summary.test_row_count == 2
    assert outcome_b.summary.test_row_count == 2


def test_selected_model_consistency_and_reuse() -> None:
    registry = create_default_anomaly_model_registry(random_state=3)
    outcome = UnsupervisedAnomalyModelScreener(registry).screen(
        _anomaly_split(),
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    model = outcome.selected_model
    assert isinstance(model, BaseAnomalyModel)
    assert getattr(model, "is_fitted", False) is True
    metadata = model.get_metadata()
    assert metadata.model_name == outcome.summary.selected_model_name
    assert metadata.estimator_key == outcome.summary.selected_estimator_key  # type: ignore[attr-defined]
    assert metadata.task is AnalysisTask.UNSUPERVISED_ANOMALY
    assert list(metadata.features) == ["f1", "f2"]
    assert metadata.fit_row_count == outcome.summary.train_row_count  # type: ignore[attr-defined]
    detect = model.detect(_anomaly_split().validation.select(["f1", "f2"]))
    assert isinstance(detect, AnomalyDetectionResult)
    scores = model.score_samples(_anomaly_split().validation.select(["f1", "f2"]))
    assert scores.shape[0] == 4


def test_immutability_determinism_and_isolation() -> None:
    registry = create_default_anomaly_model_registry(random_state=11)
    profile = _DummyProfile()
    features = ["f1", "f2"]
    features_before = list(features)
    split = _anomaly_split()
    report = _safe_report()

    screener_a = UnsupervisedAnomalyModelScreener(registry)
    outcome_1 = screener_a.screen(
        split,
        feature_columns=features,
        leakage_report=report,
        industry_profile=profile,
    )
    specs_before = [
        item.spec.model_dump() for item in outcome_1.summary.candidate_results
    ]
    metrics_before = copy.deepcopy(outcome_1.summary.selected_metrics)
    outcome_1.summary.selected_metrics["train_anomaly_fraction"] = -1.0
    for item in outcome_1.summary.candidate_results:
        if item.status is AnomalyCandidateRunStatus.SUCCESS:
            item.metrics["train_anomaly_fraction"] = -2.0

    outcome_2 = screener_a.screen(
        split,
        feature_columns=features,
        leakage_report=report,
        industry_profile=profile,
    )
    assert features == features_before
    assert [
        item.spec.model_dump() for item in outcome_2.summary.candidate_results
    ] == specs_before
    assert outcome_2.summary.selected_metrics == metrics_before
    assert outcome_1.summary.selected_model_name == (
        outcome_2.summary.selected_model_name
    )
    assert outcome_1.selected_model is not outcome_2.selected_model

    screener_b = UnsupervisedAnomalyModelScreener(registry)
    outcome_3 = screener_b.screen(
        split,
        feature_columns=features,
        leakage_report=report,
    )
    assert outcome_3.summary.selected_model_name == (
        outcome_1.summary.selected_model_name
    )
    assert outcome_3.selected_model is not outcome_1.selected_model

    dumped = outcome_3.summary.model_dump()
    assert "estimator" not in dumped
    for item in dumped["candidate_results"]:
        assert "estimator" not in item
        assert set(item.keys()) >= {
            "spec",
            "status",
            "registry_rank",
            "metrics",
            "quality_flags",
        }


def test_default_registry_integration_smoke() -> None:
    registry = create_default_anomaly_model_registry(random_state=42)
    outcome = UnsupervisedAnomalyModelScreener(registry).screen(
        _anomaly_split(),
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    assert outcome.summary.task is AnalysisTask.UNSUPERVISED_ANOMALY
    assert outcome.summary.selected_model_name in {
        "Isolation Forest",
        "One-Class SVM",
        "Elliptic Envelope",
    }
    assert len(outcome.summary.candidate_results) == 3
    for item in outcome.summary.candidate_results:
        assert item.fit_seconds >= 0.0
        assert item.train_scoring_seconds >= 0.0
        assert item.validation_scoring_seconds >= 0.0
        assert math.isclose(
            item.total_seconds,
            item.fit_seconds
            + item.train_scoring_seconds
            + item.validation_scoring_seconds,
            abs_tol=1e-12,
        )
        assert math.isfinite(item.total_seconds)
