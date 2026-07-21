"""Unit tests for final supervised model evaluation (Step 6D)."""

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
from process_intelligence.core.protocols import BaseAnalysisModel, DataFrameLike, SeriesLike
from process_intelligence.core.schemas import (
    ExplanationResult,
    ModelEvaluation,
    ModelMetadata,
    ModelSpec,
)
from process_intelligence.data.loader import ORIGINAL_ROW_ID_COLUMN
from process_intelligence.evaluation import (
    FinalEvaluationOutcome,
    FinalEvaluationPolicy,
    FinalEvaluationReport,
    FinalModelEvaluator,
)
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
    CandidateRunStatus,
    CandidateScreeningResult,
    ModelRegistry,
    ModelScreeningOutcome,
    ModelScreeningSummary,
    SupervisedModelScreener,
    create_default_supervised_model_registry,
)
from process_intelligence.models.screening import ModelScreeningPolicy


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


def _classification_split() -> DatasetSplit:
    train = pl.DataFrame(
        {
            ORIGINAL_ROW_ID_COLUMN: [0, 1, 2, 3, 4, 5, 6, 7],
            "f1": [0.0, 0.1, 0.2, 0.3, 3.0, 3.1, 3.2, 3.3],
            "f2": [0.0, 0.0, 0.1, 0.1, 1.0, 1.0, 1.1, 1.1],
            "y": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    validation = pl.DataFrame(
        {
            ORIGINAL_ROW_ID_COLUMN: [8, 9, 10, 11],
            "f1": [0.05, 0.25, 3.05, 3.25],
            "f2": [0.0, 0.1, 1.0, 1.1],
            "y": [0, 0, 1, 1],
        }
    )
    test = pl.DataFrame(
        {
            ORIGINAL_ROW_ID_COLUMN: [12, 13, 14, 15],
            "f1": [0.05, 0.25, 3.05, 3.25],
            "f2": [0.0, 0.1, 1.0, 1.1],
            "y": [0, 0, 1, 1],
        }
    )
    return DatasetSplit(
        train=train,
        validation=validation,
        test=test,
        summary=_summary(
            train_ids=[0, 1, 2, 3, 4, 5, 6, 7],
            validation_ids=[8, 9, 10, 11],
            test_ids=[12, 13, 14, 15],
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
        checked_feature_columns=features,
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
    if metrics is None:
        if task is AnalysisTask.REGRESSION:
            payload = {"rmse": 1.0, "mae": 0.5, "r2": 0.9}
            primary_name = "rmse"
        else:
            payload = {"f1_macro": 0.8, "balanced_accuracy": 0.75, "accuracy": 0.7}
            primary_name = "f1_macro"
    else:
        payload = dict(metrics)
        primary_name = "rmse" if task is AnalysisTask.REGRESSION else "f1_macro"
    return CandidateScreeningResult(
        spec=_spec(name=name, estimator_key=estimator_key, task=task),
        status=CandidateRunStatus.SUCCESS,
        registry_rank=registry_rank,
        metrics=payload,
        primary_metric_name=primary_name,
        primary_metric_value=payload[primary_name],
        fit_seconds=0.01,
        evaluation_seconds=0.02,
        total_seconds=0.03,
        warnings=[] if warnings is None else list(warnings),
    )


def _valid_summary(**overrides: object) -> ModelScreeningSummary:
    candidate_results = [
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
        "target_column": "y",
        "feature_columns": ["f1", "f2"],
        "candidate_results": candidate_results,
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


def _classification_summary(**overrides: object) -> ModelScreeningSummary:
    candidate_results = [
        _success_result(
            name="Dummy Classifier",
            estimator_key="dummy_classifier",
            registry_rank=0,
            task=AnalysisTask.CLASSIFICATION,
            metrics={"f1_macro": 0.4, "balanced_accuracy": 0.4, "accuracy": 0.4},
        ),
        _success_result(
            name="Logistic Regression",
            estimator_key="logistic_regression",
            registry_rank=1,
            task=AnalysisTask.CLASSIFICATION,
            metrics={"f1_macro": 0.9, "balanced_accuracy": 0.85, "accuracy": 0.8},
        ),
    ]
    payload: dict[str, object] = {
        "task": AnalysisTask.CLASSIFICATION,
        "target_column": "y",
        "feature_columns": ["f1", "f2"],
        "candidate_results": candidate_results,
        "selected_model_name": "Logistic Regression",
        "selected_estimator_key": "logistic_regression",
        "selected_metrics": {
            "f1_macro": 0.9,
            "balanced_accuracy": 0.85,
            "accuracy": 0.8,
        },
        "selected_registry_rank": 1,
        "ranking_metric": "f1_macro",
        "higher_is_better": True,
        "baseline_model_name": "Dummy Classifier",
        "baseline_metrics": {
            "f1_macro": 0.4,
            "balanced_accuracy": 0.4,
            "accuracy": 0.4,
        },
        "selected_beats_baseline": True,
        "successful_model_names": ["Dummy Classifier", "Logistic Regression"],
        "failed_model_names": [],
        "train_row_count": 8,
        "validation_row_count": 4,
        "test_row_count": 4,
        "warnings": [],
    }
    payload.update(overrides)
    return ModelScreeningSummary(**payload)  # type: ignore[arg-type]


class ControllableModel(BaseAnalysisModel):
    """Deterministic stub model with injectable fit/evaluate behavior."""

    def __init__(
        self,
        *,
        name: str,
        task: AnalysisTask = AnalysisTask.REGRESSION,
        metrics: dict[str, float] | None = None,
        fit_error: Exception | None = None,
        evaluate_error: Exception | None = None,
        evaluate_return: object | None = None,
        estimator_key: str | None = None,
    ) -> None:
        self._name = name
        self._task = task
        self._metrics = (
            {"rmse": 0.5, "mae": 0.4, "r2": 0.8} if metrics is None else dict(metrics)
        )
        self._fit_error = fit_error
        self._evaluate_error = evaluate_error
        self._evaluate_return = evaluate_return
        self._is_fitted = False
        self.fit_calls = 0
        self.evaluate_calls = 0
        self.fit_row_counts: list[int] = []
        self.fit_feature_names: list[list[str]] = []
        self.fit_row_ids: list[list[int]] = []
        self.evaluate_feature_names: list[list[str]] = []
        if estimator_key is not None:
            self._spec = _spec(name=name, task=task, estimator_key=estimator_key)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, X: DataFrameLike, y: SeriesLike | None = None) -> Self:
        _ = y
        if self._fit_error is not None:
            raise self._fit_error
        self.fit_calls += 1
        if isinstance(X, pl.DataFrame):
            self.fit_row_counts.append(int(X.height))
            self.fit_feature_names.append(list(X.columns))
            if ORIGINAL_ROW_ID_COLUMN in X.columns:
                self.fit_row_ids.append(
                    [int(v) for v in X.get_column(ORIGINAL_ROW_ID_COLUMN).to_list()]
                )
            else:
                self.fit_row_ids.append([])
        else:
            self.fit_row_counts.append(int(len(X)))
            self.fit_feature_names.append([str(c) for c in X.columns])
            self.fit_row_ids.append([])
        self._is_fitted = True
        return self

    def predict(self, X: DataFrameLike) -> np.ndarray:
        height = int(X.height) if isinstance(X, pl.DataFrame) else len(X)
        return np.zeros(height, dtype=float)

    def evaluate(
        self,
        X: DataFrameLike,
        y: SeriesLike | None = None,
    ) -> ModelEvaluation:
        _ = y
        self.evaluate_calls += 1
        if isinstance(X, pl.DataFrame):
            self.evaluate_feature_names.append(list(X.columns))
        if self._evaluate_error is not None:
            raise self._evaluate_error
        if self._evaluate_return is not None:
            return self._evaluate_return  # type: ignore[return-value]
        return ModelEvaluation(metrics=dict(self._metrics), notes=[])

    def explain(self, X: DataFrameLike) -> ExplanationResult:
        _ = X
        return ExplanationResult(method="stub", feature_importances={})

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(model_name=self._name, version="0", task=self._task)


def _registry_with_models(
    entries: list[tuple[ModelSpec, ControllableModel]],
) -> ModelRegistry:
    registry = ModelRegistry()
    for spec, model in entries:
        captured = model
        captured_spec = spec

        def _factory(
            captured_model: ControllableModel = captured,
            bound_spec: ModelSpec = captured_spec,
        ) -> BaseAnalysisModel:
            clone = ControllableModel(
                name=captured_model._name,
                task=captured_model._task,
                metrics=dict(captured_model._metrics),
                fit_error=captured_model._fit_error,
                evaluate_error=captured_model._evaluate_error,
                evaluate_return=captured_model._evaluate_return,
                estimator_key=bound_spec.estimator_key,
            )
            clone._spec = bound_spec.model_copy(deep=True)
            return clone

        registry.register_factory(spec.estimator_key, _factory, replace=True)
        registry.register(spec)
    return registry


def _fitted_selected(
    *,
    name: str = "Linear Regression",
    task: AnalysisTask = AnalysisTask.REGRESSION,
    metrics: dict[str, float] | None = None,
    estimator_key: str = "linear_regression",
) -> ControllableModel:
    model = ControllableModel(
        name=name,
        task=task,
        metrics=metrics,
        estimator_key=estimator_key,
    )
    model._is_fitted = True
    return model


def _outcome_from_summary(
    summary: ModelScreeningSummary,
    selected: ControllableModel | None = None,
) -> ModelScreeningOutcome:
    model = selected
    if model is None:
        model = _fitted_selected(
            name=summary.selected_model_name,
            task=summary.task,
            metrics=dict(summary.selected_metrics),
            estimator_key=summary.selected_estimator_key,
        )
    return ModelScreeningOutcome(selected_model=model, summary=summary)


def _valid_report_kwargs(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "task": AnalysisTask.REGRESSION,
        "model_name": "Linear Regression",
        "estimator_key": "linear_regression",
        "target_column": "y",
        "feature_columns": ["f1", "f2"],
        "refit_on_train_validation": True,
        "train_row_count": 6,
        "validation_row_count": 3,
        "test_row_count": 3,
        "final_fit_row_count": 9,
        "validation_metrics": {"rmse": 1.0, "mae": 0.5, "r2": 0.9},
        "test_metrics": {"rmse": 1.2, "mae": 0.6, "r2": 0.8},
        "primary_metric_name": "rmse",
        "higher_is_better": False,
        "validation_primary_metric": 1.0,
        "test_primary_metric": 1.2,
        "generalization_gap": 0.2,
        "performance_degraded": True,
        "fit_seconds": 0.01,
        "test_evaluation_seconds": 0.02,
        "total_seconds": 0.03,
        "evaluated_at": datetime.now(UTC),
        "warnings": [],
    }
    payload.update(overrides)
    return payload


def _evaluate_default(
    *,
    registry: ModelRegistry | None = None,
    split: DatasetSplit | None = None,
    screening_outcome: ModelScreeningOutcome | None = None,
    task: AnalysisTask = AnalysisTask.REGRESSION,
    target_column: str = "y",
    feature_columns: list[str] | None = None,
    leakage_report: LeakageReport | None = None,
    policy: FinalEvaluationPolicy | None = None,
) -> FinalEvaluationOutcome:
    features = ["f1", "f2"] if feature_columns is None else feature_columns
    if registry is None:
        selected = _fitted_selected()
        registry = _registry_with_models(
            [
                (
                    _spec(name="Dummy Regressor", estimator_key="dummy_regressor"),
                    ControllableModel(
                        name="Dummy Regressor",
                        metrics={"rmse": 2.0, "mae": 1.5, "r2": 0.0},
                        estimator_key="dummy_regressor",
                    ),
                ),
                (
                    _spec(name="Linear Regression", estimator_key="linear_regression"),
                    ControllableModel(
                        name="Linear Regression",
                        metrics={"rmse": 0.5, "mae": 0.4, "r2": 0.8},
                        estimator_key="linear_regression",
                    ),
                ),
            ]
        )
        _ = selected
    used_split = _regression_split() if split is None else split
    used_outcome = (
        _outcome_from_summary(_valid_summary())
        if screening_outcome is None
        else screening_outcome
    )
    used_report = _safe_report(features) if leakage_report is None else leakage_report
    evaluator = FinalModelEvaluator(registry, policy=policy)
    return evaluator.evaluate(
        used_split,
        used_outcome,
        task=task,
        target_column=target_column,
        feature_columns=features,
        leakage_report=used_report,
    )


# --- FinalEvaluationPolicy ---


def test_policy_defaults_and_bool_rejection() -> None:
    policy = FinalEvaluationPolicy()
    assert policy.refit_on_train_validation is True
    assert policy.require_safe_leakage_report is True
    assert policy.require_screening_consistency is True
    assert policy.warn_on_test_degradation is True

    for field in (
        "refit_on_train_validation",
        "require_safe_leakage_report",
        "require_screening_consistency",
        "warn_on_test_degradation",
    ):
        with pytest.raises(ValidationError):
            FinalEvaluationPolicy(**{field: 1})  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            FinalEvaluationPolicy(**{field: "true"})  # type: ignore[arg-type]

    dumped = policy.model_dump()
    restored = FinalEvaluationPolicy.model_validate(dumped)
    assert restored == policy


def test_policy_isolation_from_external_mutation() -> None:
    external = FinalEvaluationPolicy(refit_on_train_validation=True)
    registry = ModelRegistry()
    evaluator = FinalModelEvaluator(registry, policy=external)
    external.refit_on_train_validation = False
    assert evaluator._policy.refit_on_train_validation is True

    other = FinalModelEvaluator(
        registry,
        policy=FinalEvaluationPolicy(refit_on_train_validation=False),
    )
    assert evaluator._policy.refit_on_train_validation is True
    assert other._policy.refit_on_train_validation is False


# --- FinalEvaluationReport ---


def test_report_valid_regression_and_classification() -> None:
    reg = FinalEvaluationReport(**_valid_report_kwargs())  # type: ignore[arg-type]
    assert reg.primary_metric_name == "rmse"
    assert reg.higher_is_better is False

    clf_kwargs = _valid_report_kwargs(
        task=AnalysisTask.CLASSIFICATION,
        model_name="Logistic Regression",
        estimator_key="logistic_regression",
        validation_metrics={"f1_macro": 0.9, "balanced_accuracy": 0.85, "accuracy": 0.8},
        test_metrics={"f1_macro": 0.7, "balanced_accuracy": 0.7, "accuracy": 0.7},
        primary_metric_name="f1_macro",
        higher_is_better=True,
        validation_primary_metric=0.9,
        test_primary_metric=0.7,
        generalization_gap=0.2,
        performance_degraded=True,
    )
    clf = FinalEvaluationReport(**clf_kwargs)  # type: ignore[arg-type]
    assert clf.primary_metric_name == "f1_macro"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_name", ""),
        ("model_name", "   "),
        ("estimator_key", ""),
        ("target_column", ""),
        ("primary_metric_name", ""),
        ("feature_columns", []),
        ("feature_columns", ["f1", "f1"]),
        ("feature_columns", ["f1", "y"]),
        ("feature_columns", [ORIGINAL_ROW_ID_COLUMN, "f1"]),
        ("train_row_count", -1),
        ("test_row_count", 0),
        ("final_fit_row_count", 0),
        ("validation_metrics", {}),
        ("test_metrics", {}),
        ("fit_seconds", -0.1),
        ("fit_seconds", float("nan")),
        ("fit_seconds", float("inf")),
        ("total_seconds", 0.99),
        ("evaluated_at", datetime(2026, 1, 1)),
        ("warnings", ["a", "a"]),
    ],
)
def test_report_rejects_invalid_fields(field: str, value: object) -> None:
    kwargs = _valid_report_kwargs(**{field: value})
    with pytest.raises(ValidationError):
        FinalEvaluationReport(**kwargs)  # type: ignore[arg-type]


def test_report_rejects_missing_primary_and_bad_metrics() -> None:
    with pytest.raises(ValidationError):
        FinalEvaluationReport(
            **_valid_report_kwargs(
                validation_metrics={"mae": 0.5},
                primary_metric_name="rmse",
            )
        )  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        FinalEvaluationReport(
            **_valid_report_kwargs(
                test_metrics={"mae": 0.5},
                primary_metric_name="rmse",
            )
        )  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        FinalEvaluationReport(
            **_valid_report_kwargs(test_metrics={"": 1.0, "rmse": 1.0, "mae": 0.5})
        )  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        FinalEvaluationReport(
            **_valid_report_kwargs(
                test_metrics={"rmse": True, "mae": 0.5}  # type: ignore[dict-item]
            )
        )
    with pytest.raises(ValidationError):
        FinalEvaluationReport(
            **_valid_report_kwargs(test_metrics={"rmse": float("nan"), "mae": 0.5})
        )  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        FinalEvaluationReport(
            **_valid_report_kwargs(test_metrics={"rmse": float("inf"), "mae": 0.5})
        )  # type: ignore[arg-type]


def test_report_mutable_independence_and_round_trip() -> None:
    features = ["f1", "f2"]
    metrics = {"rmse": 1.0, "mae": 0.5}
    warnings = ["w1"]
    report = FinalEvaluationReport(
        **_valid_report_kwargs(
            feature_columns=features,
            validation_metrics=metrics,
            test_metrics=dict(metrics),
            warnings=warnings,
            generalization_gap=0.0,
            performance_degraded=False,
            validation_primary_metric=1.0,
            test_primary_metric=1.0,
        )
    )  # type: ignore[arg-type]
    features.append("f3")
    metrics["rmse"] = 99.0
    warnings.append("w2")
    assert report.feature_columns == ["f1", "f2"]
    assert report.validation_metrics["rmse"] == 1.0
    assert report.warnings == ["w1"]

    restored = FinalEvaluationReport.model_validate(report.model_dump())
    assert restored.model_name == report.model_name
    assert restored.test_metrics == report.test_metrics


# --- FinalEvaluationOutcome ---


def test_outcome_frozen_slots_and_types() -> None:
    assert dataclasses.is_dataclass(FinalEvaluationOutcome)
    assert FinalEvaluationOutcome.__dataclass_params__.frozen is True  # type: ignore[attr-defined]
    assert FinalEvaluationOutcome.__slots__ == ("final_model", "report")

    model = _fitted_selected()
    report = FinalEvaluationReport(**_valid_report_kwargs())  # type: ignore[arg-type]
    outcome = FinalEvaluationOutcome(final_model=model, report=report)
    assert isinstance(outcome.final_model, BaseAnalysisModel)
    assert isinstance(outcome.report, FinalEvaluationReport)
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.report = report  # type: ignore[misc]


# --- Evaluator construction ---


def test_evaluator_construction_type_checks_and_isolation() -> None:
    with pytest.raises(TypeError):
        FinalModelEvaluator("not-registry")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        FinalModelEvaluator(ModelRegistry(), policy="bad")  # type: ignore[arg-type]

    registry = ModelRegistry()
    policy = FinalEvaluationPolicy(warn_on_test_degradation=False)
    evaluator = FinalModelEvaluator(registry, policy=policy)
    assert isinstance(evaluator, FinalModelEvaluator)
    policy.warn_on_test_degradation = True
    assert evaluator._policy.warn_on_test_degradation is False

    other = FinalModelEvaluator(registry)
    assert other._policy.warn_on_test_degradation is True
    assert evaluator._policy is not other._policy


# --- Input validation ---


def test_input_type_and_schema_validation() -> None:
    registry = _registry_with_models(
        [
            (
                _spec(name="Linear Regression"),
                ControllableModel(name="Linear Regression"),
            )
        ]
    )
    evaluator = FinalModelEvaluator(registry)
    outcome = _outcome_from_summary(_valid_summary())
    split = _regression_split()
    features = ["f1", "f2"]
    report = _safe_report()

    with pytest.raises(TypeError):
        evaluator.evaluate(
            "bad",  # type: ignore[arg-type]
            outcome,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=features,
            leakage_report=report,
        )
    with pytest.raises(TypeError):
        evaluator.evaluate(
            split,
            "bad",  # type: ignore[arg-type]
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=features,
            leakage_report=report,
        )

    bad_split = DatasetSplit(
        train=split.train,
        validation=split.validation,
        test=[1, 2, 3],  # type: ignore[arg-type]
        summary=split.summary,
    )
    with pytest.raises(TypeError):
        evaluator.evaluate(
            bad_split,
            outcome,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=features,
            leakage_report=report,
        )

    mismatched = DatasetSplit(
        train=split.train,
        validation=split.validation.select(["f1", "f2", "y", ORIGINAL_ROW_ID_COLUMN]),
        test=split.test,
        summary=split.summary,
    )
    with pytest.raises(DataValidationError):
        evaluator.evaluate(
            mismatched,
            outcome,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=features,
            leakage_report=report,
        )

    no_id = DatasetSplit(
        train=split.train.drop(ORIGINAL_ROW_ID_COLUMN),
        validation=split.validation.drop(ORIGINAL_ROW_ID_COLUMN),
        test=split.test.drop(ORIGINAL_ROW_ID_COLUMN),
        summary=split.summary,
    )
    with pytest.raises(DataValidationError):
        evaluator.evaluate(
            no_id,
            outcome,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=features,
            leakage_report=report,
        )

    with pytest.raises(TypeError):
        evaluator.evaluate(
            split,
            outcome,
            task="REGRESSION",  # type: ignore[arg-type]
            target_column="y",
            feature_columns=features,
            leakage_report=report,
        )
    with pytest.raises(DataValidationError):
        evaluator.evaluate(
            split,
            outcome,
            task=AnalysisTask.UNSUPERVISED_ANOMALY,
            target_column="y",
            feature_columns=features,
            leakage_report=report,
        )
    with pytest.raises(TypeError):
        evaluator.evaluate(
            split,
            outcome,
            task=AnalysisTask.REGRESSION,
            target_column=1,  # type: ignore[arg-type]
            feature_columns=features,
            leakage_report=report,
        )
    with pytest.raises(DataValidationError):
        evaluator.evaluate(
            split,
            outcome,
            task=AnalysisTask.REGRESSION,
            target_column="  ",
            feature_columns=features,
            leakage_report=report,
        )
    with pytest.raises(DataValidationError):
        evaluator.evaluate(
            split,
            outcome,
            task=AnalysisTask.REGRESSION,
            target_column="missing",
            feature_columns=features,
            leakage_report=report,
        )
    with pytest.raises(DataValidationError):
        evaluator.evaluate(
            split,
            outcome,
            task=AnalysisTask.REGRESSION,
            target_column=ORIGINAL_ROW_ID_COLUMN,
            feature_columns=features,
            leakage_report=report,
        )


def test_feature_column_validation() -> None:
    registry = _registry_with_models(
        [
            (
                _spec(name="Linear Regression"),
                ControllableModel(name="Linear Regression"),
            )
        ]
    )
    evaluator = FinalModelEvaluator(registry)
    outcome = _outcome_from_summary(_valid_summary())
    split = _regression_split()
    report = _safe_report()

    with pytest.raises(TypeError):
        evaluator.evaluate(
            split,
            outcome,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns="f1",  # type: ignore[arg-type]
            leakage_report=report,
        )
    with pytest.raises(TypeError):
        evaluator.evaluate(
            split,
            outcome,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=b"f1",  # type: ignore[arg-type]
            leakage_report=report,
        )
    with pytest.raises(TypeError):
        evaluator.evaluate(
            split,
            outcome,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=[1],  # type: ignore[list-item]
            leakage_report=report,
        )
    with pytest.raises(DataValidationError):
        evaluator.evaluate(
            split,
            outcome,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=[" "],
            leakage_report=report,
        )
    with pytest.raises(DataValidationError):
        evaluator.evaluate(
            split,
            outcome,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=[],
            leakage_report=_safe_report([]),
        )
    with pytest.raises(DataValidationError):
        evaluator.evaluate(
            split,
            outcome,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=["f1", "f1"],
            leakage_report=report,
        )
    with pytest.raises(DataValidationError):
        evaluator.evaluate(
            split,
            outcome,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=["f1", "y"],
            leakage_report=report,
        )
    with pytest.raises(DataValidationError):
        evaluator.evaluate(
            split,
            outcome,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=[ORIGINAL_ROW_ID_COLUMN, "f1"],
            leakage_report=report,
        )
    with pytest.raises(DataValidationError):
        evaluator.evaluate(
            split,
            outcome,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=["f1", "missing"],
            leakage_report=_safe_report(["f1", "missing"]),
        )


def test_leakage_gate_and_immutability() -> None:
    features = ["f1", "f2"]
    with pytest.raises(TypeError):
        _evaluate_default(leakage_report="bad")  # type: ignore[arg-type]
    with pytest.raises(DataValidationError):
        _evaluate_default(leakage_report=_safe_report(["f2", "f1"]))
    with pytest.raises(DataValidationError):
        _evaluate_default(leakage_report=_safe_report(["f1"]))

    with pytest.raises(DataLeakageError):
        _evaluate_default(leakage_report=_blocker_report())

    outcome = _evaluate_default(leakage_report=_warning_report())
    assert any("LeakageReport" in warning for warning in outcome.report.warnings)

    outcome_safe = _evaluate_default(leakage_report=_safe_report())
    assert isinstance(outcome_safe.report, FinalEvaluationReport)

    allowed = _evaluate_default(
        leakage_report=_blocker_report(),
        policy=FinalEvaluationPolicy(require_safe_leakage_report=False),
    )
    assert allowed.report.model_name == "Linear Regression"

    report = _safe_report()
    before = report.model_dump()
    _evaluate_default(leakage_report=report)
    assert report.model_dump() == before
    _ = features


def test_minimum_partition_sizes() -> None:
    base = _regression_split()
    empty_train = DatasetSplit(
        train=base.train.head(0),
        validation=base.validation,
        test=base.test,
        summary=base.summary,
    )
    with pytest.raises(InsufficientDataError):
        _evaluate_default(
            split=empty_train,
            screening_outcome=_outcome_from_summary(_valid_summary()),
            policy=FinalEvaluationPolicy(require_screening_consistency=False),
        )

    empty_validation = DatasetSplit(
        train=base.train,
        validation=base.validation.head(0),
        test=base.test,
        summary=base.summary,
    )
    with pytest.raises(InsufficientDataError):
        _evaluate_default(
            split=empty_validation,
            screening_outcome=_outcome_from_summary(_valid_summary()),
            policy=FinalEvaluationPolicy(require_screening_consistency=False),
        )

    empty_test = DatasetSplit(
        train=base.train,
        validation=base.validation,
        test=base.test.head(0),
        summary=base.summary,
    )
    with pytest.raises(InsufficientDataError):
        _evaluate_default(
            split=empty_test,
            screening_outcome=_outcome_from_summary(_valid_summary()),
            policy=FinalEvaluationPolicy(require_screening_consistency=False),
        )


def test_screening_consistency_checks() -> None:
    split = _regression_split()
    base_summary = _valid_summary()

    cases: list[dict[str, Any]] = [
        {"task": AnalysisTask.CLASSIFICATION},
        {"target_column": "other"},
        {"feature_columns": ["f2", "f1"]},
        {"train_row_count": 5},
        {"validation_row_count": 2},
        {"test_row_count": 2},
    ]
    for overrides in cases:
        with pytest.raises(DataValidationError):
            _evaluate_default(
                split=split,
                screening_outcome=_outcome_from_summary(
                    _valid_summary(**overrides)
                ),
            )

    # consistency False skips detailed mismatch checks
    mismatched = _outcome_from_summary(_valid_summary(train_row_count=5))
    ok = _evaluate_default(
        split=split,
        screening_outcome=mismatched,
        policy=FinalEvaluationPolicy(require_screening_consistency=False),
    )
    assert ok.report.train_row_count == split.train.height
    _ = base_summary


def test_selected_candidate_resolution() -> None:
    outcome = _evaluate_default()
    assert outcome.report.model_name == "Linear Regression"
    assert outcome.report.estimator_key == "linear_regression"

    valid = _valid_summary()
    no_match = ModelScreeningSummary.model_construct(
        task=valid.task,
        target_column=valid.target_column,
        feature_columns=list(valid.feature_columns),
        candidate_results=[
            _success_result(
                name="Dummy Regressor",
                estimator_key="dummy_regressor",
                registry_rank=0,
                metrics={"rmse": 2.0, "mae": 1.5},
            ),
            _success_result(
                name="Other",
                estimator_key="other",
                registry_rank=2,
                metrics={"rmse": 0.1, "mae": 0.1},
            ),
        ],
        selected_model_name="Linear Regression",
        selected_estimator_key="linear_regression",
        selected_metrics=dict(valid.selected_metrics),
        selected_registry_rank=1,
        ranking_metric=valid.ranking_metric,
        higher_is_better=valid.higher_is_better,
        baseline_model_name=valid.baseline_model_name,
        baseline_metrics=dict(valid.baseline_metrics),
        selected_beats_baseline=valid.selected_beats_baseline,
        successful_model_names=["Dummy Regressor", "Other"],
        failed_model_names=[],
        train_row_count=valid.train_row_count,
        validation_row_count=valid.validation_row_count,
        test_row_count=valid.test_row_count,
        warnings=[],
    )
    with pytest.raises(DataValidationError):
        _evaluate_default(screening_outcome=_outcome_from_summary(no_match))

    duplicate = ModelScreeningSummary.model_construct(
        task=valid.task,
        target_column=valid.target_column,
        feature_columns=list(valid.feature_columns),
        candidate_results=[
            _success_result(
                name="Dummy Regressor",
                estimator_key="dummy_regressor",
                registry_rank=0,
                metrics={"rmse": 2.0, "mae": 1.5},
            ),
            _success_result(
                name="Linear Regression",
                registry_rank=1,
                metrics={"rmse": 1.0, "mae": 0.5},
            ),
            _success_result(
                name="Linear Regression",
                registry_rank=1,
                metrics={"rmse": 1.1, "mae": 0.6},
            ),
        ],
        selected_model_name=valid.selected_model_name,
        selected_estimator_key=valid.selected_estimator_key,
        selected_metrics=dict(valid.selected_metrics),
        selected_registry_rank=1,
        ranking_metric=valid.ranking_metric,
        higher_is_better=valid.higher_is_better,
        baseline_model_name=valid.baseline_model_name,
        baseline_metrics=dict(valid.baseline_metrics),
        selected_beats_baseline=valid.selected_beats_baseline,
        successful_model_names=list(valid.successful_model_names),
        failed_model_names=[],
        train_row_count=valid.train_row_count,
        validation_row_count=valid.validation_row_count,
        test_row_count=valid.test_row_count,
        warnings=[],
    )
    with pytest.raises(DataValidationError):
        _evaluate_default(screening_outcome=_outcome_from_summary(duplicate))

    failed_only_wrong = ModelScreeningSummary.model_construct(
        task=valid.task,
        target_column=valid.target_column,
        feature_columns=list(valid.feature_columns),
        candidate_results=[
            _success_result(
                name="Dummy Regressor",
                estimator_key="dummy_regressor",
                registry_rank=0,
                metrics={"rmse": 2.0, "mae": 1.5},
            ),
            CandidateScreeningResult(
                spec=_spec(name="Linear Regression"),
                status=CandidateRunStatus.FAILED,
                registry_rank=1,
                metrics={},
                primary_metric_name=None,
                primary_metric_value=None,
                fit_seconds=0.01,
                evaluation_seconds=0.0,
                total_seconds=0.01,
                warnings=[],
                error_type="ValueError",
                error_message="boom",
            ),
            _success_result(
                name="Linear Regression",
                registry_rank=2,
                metrics={"rmse": 1.0, "mae": 0.5},
            ),
        ],
        selected_model_name=valid.selected_model_name,
        selected_estimator_key=valid.selected_estimator_key,
        selected_metrics=dict(valid.selected_metrics),
        selected_registry_rank=1,
        ranking_metric=valid.ranking_metric,
        higher_is_better=valid.higher_is_better,
        baseline_model_name=valid.baseline_model_name,
        baseline_metrics=dict(valid.baseline_metrics),
        selected_beats_baseline=valid.selected_beats_baseline,
        successful_model_names=["Dummy Regressor", "Linear Regression"],
        failed_model_names=[],
        train_row_count=valid.train_row_count,
        validation_row_count=valid.validation_row_count,
        test_row_count=valid.test_row_count,
        warnings=[],
    )
    with pytest.raises(DataValidationError):
        _evaluate_default(screening_outcome=_outcome_from_summary(failed_only_wrong))


# --- Refit behavior ---


def test_default_refit_creates_new_instance_and_uses_train_validation() -> None:
    tracking = ControllableModel(
        name="Linear Regression",
        metrics={"rmse": 0.5, "mae": 0.4, "r2": 0.8},
        estimator_key="linear_regression",
    )
    registry = _registry_with_models(
        [
            (
                _spec(name="Dummy Regressor", estimator_key="dummy_regressor"),
                ControllableModel(
                    name="Dummy Regressor",
                    metrics={"rmse": 2.0, "mae": 1.5},
                    estimator_key="dummy_regressor",
                ),
            ),
            (
                _spec(name="Linear Regression", estimator_key="linear_regression"),
                tracking,
            ),
        ]
    )
    selected = _fitted_selected()
    selected_before_fitted = selected.is_fitted
    split = _regression_split()
    screening = _outcome_from_summary(_valid_summary(), selected=selected)
    before_specs = registry.list_specs(AnalysisTask.REGRESSION)

    result = FinalModelEvaluator(registry).evaluate(
        split,
        screening,
        task=AnalysisTask.REGRESSION,
        target_column="y",
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )

    assert result.final_model is not selected
    assert result.final_model.is_fitted is True
    assert selected.is_fitted is selected_before_fitted
    assert result.report.final_fit_row_count == 9
    assert result.report.refit_on_train_validation is True
    assert isinstance(result.final_model, ControllableModel)
    assert result.final_model.fit_calls == 1
    assert result.final_model.fit_row_counts == [9]
    assert result.final_model.fit_feature_names == [["f1", "f2"]]
    assert result.final_model.evaluate_calls == 1
    assert ORIGINAL_ROW_ID_COLUMN not in result.final_model.fit_feature_names[0]
    assert registry.list_specs(AnalysisTask.REGRESSION) == before_specs


def test_refit_disabled_uses_selected_model() -> None:
    selected = _fitted_selected(
        metrics={"rmse": 0.25, "mae": 0.2, "r2": 0.95},
    )
    registry = _registry_with_models(
        [
            (
                _spec(name="Linear Regression"),
                ControllableModel(
                    name="Linear Regression",
                    metrics={"rmse": 9.0, "mae": 9.0},
                ),
            )
        ]
    )
    screening = _outcome_from_summary(_valid_summary(), selected=selected)
    result = FinalModelEvaluator(
        registry,
        policy=FinalEvaluationPolicy(refit_on_train_validation=False),
    ).evaluate(
        _regression_split(),
        screening,
        task=AnalysisTask.REGRESSION,
        target_column="y",
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    assert result.final_model is selected
    assert result.report.fit_seconds == 0.0
    assert result.report.final_fit_row_count == 6
    assert any("not refit" in warning.lower() for warning in result.report.warnings)

    unfitted = ControllableModel(name="Linear Regression")
    with pytest.raises(ProcessIntelligenceError):
        FinalModelEvaluator(
            registry,
            policy=FinalEvaluationPolicy(refit_on_train_validation=False),
        ).evaluate(
            _regression_split(),
            _outcome_from_summary(_valid_summary(), selected=unfitted),
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )


def test_test_partition_not_used_for_fit_or_reselection() -> None:
    class SpyModel(ControllableModel):
        def fit(self, X: DataFrameLike, y: SeriesLike | None = None) -> Self:
            super().fit(X, y)
            if isinstance(X, pl.DataFrame) and X.height == 3:
                raise AssertionError("test-sized frame must not be used for fit")
            return self

    registry = _registry_with_models(
        [
            (
                _spec(name="Dummy Regressor", estimator_key="dummy_regressor"),
                ControllableModel(
                    name="Dummy Regressor",
                    metrics={"rmse": 2.0, "mae": 1.5},
                    estimator_key="dummy_regressor",
                ),
            ),
            (
                _spec(name="Linear Regression"),
                SpyModel(
                    name="Linear Regression",
                    metrics={"rmse": 0.5, "mae": 0.4, "r2": 0.8},
                    estimator_key="linear_regression",
                ),
            ),
        ]
    )
    split_a = _regression_split(test_target_offset=0.0)
    split_b = _regression_split(test_target_offset=1000.0)
    screening = _outcome_from_summary(_valid_summary())

    result_a = FinalModelEvaluator(registry).evaluate(
        split_a,
        screening,
        task=AnalysisTask.REGRESSION,
        target_column="y",
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    result_b = FinalModelEvaluator(registry).evaluate(
        split_b,
        screening,
        task=AnalysisTask.REGRESSION,
        target_column="y",
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    assert result_a.report.model_name == result_b.report.model_name
    assert result_a.report.estimator_key == result_b.report.estimator_key
    assert isinstance(result_a.final_model, ControllableModel)
    assert result_a.final_model.evaluate_calls == 1
    assert result_a.final_model.fit_calls == 1

    test_before = split_a.test.clone()
    _ = result_a
    assert split_a.test.equals(test_before)


# --- Metric results ---


def test_regression_metrics_gap_and_ranges() -> None:
    registry = _registry_with_models(
        [
            (
                _spec(name="Linear Regression"),
                ControllableModel(
                    name="Linear Regression",
                    metrics={"rmse": 1.5, "mae": 1.0, "r2": -0.2},
                    estimator_key="linear_regression",
                ),
            )
        ]
    )
    screening = _outcome_from_summary(
        _valid_summary(selected_metrics={"rmse": 1.0, "mae": 0.5, "r2": 0.9})
    )
    result = FinalModelEvaluator(registry).evaluate(
        _regression_split(),
        screening,
        task=AnalysisTask.REGRESSION,
        target_column="y",
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    assert "rmse" in result.report.test_metrics
    assert "mae" in result.report.test_metrics
    assert result.report.test_metrics["r2"] == -0.2
    assert result.report.primary_metric_name == "rmse"
    assert result.report.higher_is_better is False
    assert result.report.validation_primary_metric == 1.0
    assert result.report.test_primary_metric == 1.5
    assert result.report.generalization_gap == pytest.approx(0.5)
    assert result.report.performance_degraded is True

    improved = FinalModelEvaluator(
        _registry_with_models(
            [
                (
                    _spec(name="Linear Regression"),
                    ControllableModel(
                        name="Linear Regression",
                        metrics={"rmse": 0.5, "mae": 0.4, "r2": 0.95},
                        estimator_key="linear_regression",
                    ),
                )
            ]
        )
    ).evaluate(
        _regression_split(),
        screening,
        task=AnalysisTask.REGRESSION,
        target_column="y",
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    assert improved.report.generalization_gap == pytest.approx(-0.5)
    assert improved.report.performance_degraded is False

    with pytest.raises(ProcessIntelligenceError):
        FinalModelEvaluator(
            _registry_with_models(
                [
                    (
                        _spec(name="Linear Regression"),
                        ControllableModel(
                            name="Linear Regression",
                            metrics={"rmse": -0.1, "mae": 0.4},
                            estimator_key="linear_regression",
                        ),
                    )
                ]
            )
        ).evaluate(
            _regression_split(),
            screening,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )

    with pytest.raises(ProcessIntelligenceError):
        FinalModelEvaluator(
            _registry_with_models(
                [
                    (
                        _spec(name="Linear Regression"),
                        ControllableModel(
                            name="Linear Regression",
                            metrics={"rmse": 0.1, "mae": -0.4},
                            estimator_key="linear_regression",
                        ),
                    )
                ]
            )
        ).evaluate(
            _regression_split(),
            screening,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )

    with pytest.raises(ProcessIntelligenceError):
        FinalModelEvaluator(
            _registry_with_models(
                [
                    (
                        _spec(name="Linear Regression"),
                        ControllableModel(
                            name="Linear Regression",
                            metrics={"mae": 0.4},
                            estimator_key="linear_regression",
                        ),
                    )
                ]
            )
        ).evaluate(
            _regression_split(),
            screening,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )


def test_classification_metrics_gap_and_ranges() -> None:
    split = _classification_split()
    summary = _classification_summary()
    screening = _outcome_from_summary(
        summary,
        selected=_fitted_selected(
            name="Logistic Regression",
            task=AnalysisTask.CLASSIFICATION,
            metrics=dict(summary.selected_metrics),
            estimator_key="logistic_regression",
        ),
    )
    registry = _registry_with_models(
        [
            (
                _spec(
                    name="Logistic Regression",
                    task=AnalysisTask.CLASSIFICATION,
                    estimator_key="logistic_regression",
                ),
                ControllableModel(
                    name="Logistic Regression",
                    task=AnalysisTask.CLASSIFICATION,
                    metrics={
                        "f1_macro": 0.7,
                        "balanced_accuracy": 0.7,
                        "accuracy": 0.7,
                    },
                    estimator_key="logistic_regression",
                ),
            )
        ]
    )
    result = FinalModelEvaluator(registry).evaluate(
        split,
        screening,
        task=AnalysisTask.CLASSIFICATION,
        target_column="y",
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    assert "f1_macro" in result.report.test_metrics
    assert "balanced_accuracy" in result.report.test_metrics
    assert "accuracy" in result.report.test_metrics
    assert result.report.primary_metric_name == "f1_macro"
    assert result.report.higher_is_better is True
    assert result.report.generalization_gap == pytest.approx(0.2)
    assert result.report.performance_degraded is True

    improved_registry = _registry_with_models(
        [
            (
                _spec(
                    name="Logistic Regression",
                    task=AnalysisTask.CLASSIFICATION,
                    estimator_key="logistic_regression",
                ),
                ControllableModel(
                    name="Logistic Regression",
                    task=AnalysisTask.CLASSIFICATION,
                    metrics={
                        "f1_macro": 0.95,
                        "balanced_accuracy": 0.9,
                        "accuracy": 0.9,
                    },
                    estimator_key="logistic_regression",
                ),
            )
        ]
    )
    improved = FinalModelEvaluator(improved_registry).evaluate(
        split,
        screening,
        task=AnalysisTask.CLASSIFICATION,
        target_column="y",
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    assert improved.report.performance_degraded is False

    with pytest.raises(ProcessIntelligenceError):
        FinalModelEvaluator(
            _registry_with_models(
                [
                    (
                        _spec(
                            name="Logistic Regression",
                            task=AnalysisTask.CLASSIFICATION,
                            estimator_key="logistic_regression",
                        ),
                        ControllableModel(
                            name="Logistic Regression",
                            task=AnalysisTask.CLASSIFICATION,
                            metrics={
                                "f1_macro": -0.1,
                                "balanced_accuracy": 0.5,
                                "accuracy": 0.5,
                            },
                            estimator_key="logistic_regression",
                        ),
                    )
                ]
            )
        ).evaluate(
            split,
            screening,
            task=AnalysisTask.CLASSIFICATION,
            target_column="y",
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )

    with pytest.raises(ProcessIntelligenceError):
        FinalModelEvaluator(
            _registry_with_models(
                [
                    (
                        _spec(
                            name="Logistic Regression",
                            task=AnalysisTask.CLASSIFICATION,
                            estimator_key="logistic_regression",
                        ),
                        ControllableModel(
                            name="Logistic Regression",
                            task=AnalysisTask.CLASSIFICATION,
                            metrics={
                                "f1_macro": 1.1,
                                "balanced_accuracy": 0.5,
                                "accuracy": 0.5,
                            },
                            estimator_key="logistic_regression",
                        ),
                    )
                ]
            )
        ).evaluate(
            split,
            screening,
            task=AnalysisTask.CLASSIFICATION,
            target_column="y",
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )

    with pytest.raises(ProcessIntelligenceError):
        FinalModelEvaluator(
            _registry_with_models(
                [
                    (
                        _spec(
                            name="Logistic Regression",
                            task=AnalysisTask.CLASSIFICATION,
                            estimator_key="logistic_regression",
                        ),
                        ControllableModel(
                            name="Logistic Regression",
                            task=AnalysisTask.CLASSIFICATION,
                            metrics={"balanced_accuracy": 0.5, "accuracy": 0.5},
                            estimator_key="logistic_regression",
                        ),
                    )
                ]
            )
        ).evaluate(
            split,
            screening,
            task=AnalysisTask.CLASSIFICATION,
            target_column="y",
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )


def test_metric_payload_validation() -> None:
    screening = _outcome_from_summary(_valid_summary())
    with pytest.raises(ProcessIntelligenceError):
        FinalModelEvaluator(
            _registry_with_models(
                [
                    (
                        _spec(name="Linear Regression"),
                        ControllableModel(
                            name="Linear Regression",
                            evaluate_return={"rmse": 1.0},
                            estimator_key="linear_regression",
                        ),
                    )
                ]
            )
        ).evaluate(
            _regression_split(),
            screening,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )

    for bad_metrics in (
        {"": 1.0, "rmse": 1.0, "mae": 0.5},
        {"rmse": True, "mae": 0.5},
        {"rmse": float("nan"), "mae": 0.5},
        {"rmse": float("inf"), "mae": 0.5},
    ):
        with pytest.raises(ProcessIntelligenceError):
            FinalModelEvaluator(
                _registry_with_models(
                    [
                        (
                            _spec(name="Linear Regression"),
                            ControllableModel(
                                name="Linear Regression",
                                evaluate_return=ModelEvaluation.model_construct(
                                    metrics=bad_metrics,  # type: ignore[arg-type]
                                    notes=[],
                                ),
                                estimator_key="linear_regression",
                            ),
                        )
                    ]
                )
            ).evaluate(
                _regression_split(),
                screening,
                task=AnalysisTask.REGRESSION,
                target_column="y",
                feature_columns=["f1", "f2"],
                leakage_report=_safe_report(),
            )

    valid = _valid_summary()
    nan_summary = ModelScreeningSummary.model_construct(
        task=valid.task,
        target_column=valid.target_column,
        feature_columns=list(valid.feature_columns),
        candidate_results=list(valid.candidate_results),
        selected_model_name=valid.selected_model_name,
        selected_estimator_key=valid.selected_estimator_key,
        selected_metrics={"rmse": float("nan"), "mae": 0.5},
        selected_registry_rank=valid.selected_registry_rank,
        ranking_metric=valid.ranking_metric,
        higher_is_better=valid.higher_is_better,
        baseline_model_name=valid.baseline_model_name,
        baseline_metrics=dict(valid.baseline_metrics),
        selected_beats_baseline=valid.selected_beats_baseline,
        successful_model_names=list(valid.successful_model_names),
        failed_model_names=list(valid.failed_model_names),
        train_row_count=valid.train_row_count,
        validation_row_count=valid.validation_row_count,
        test_row_count=valid.test_row_count,
        warnings=list(valid.warnings),
    )
    with pytest.raises(ProcessIntelligenceError):
        FinalModelEvaluator(
            _registry_with_models(
                [
                    (
                        _spec(name="Linear Regression"),
                        ControllableModel(
                            name="Linear Regression",
                            metrics={"rmse": 0.5, "mae": 0.4},
                            estimator_key="linear_regression",
                        ),
                    )
                ]
            )
        ).evaluate(
            _regression_split(),
            _outcome_from_summary(nan_summary),
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=["f1", "f2"],
            leakage_report=_safe_report(),
        )


# --- Timing, warnings, consistency, immutability ---


def test_timings_warnings_and_model_consistency() -> None:
    registry = _registry_with_models(
        [
            (
                _spec(name="Linear Regression"),
                ControllableModel(
                    name="Linear Regression",
                    metrics={"rmse": 1.5, "mae": 1.0, "r2": 0.5},
                    estimator_key="linear_regression",
                    evaluate_return=ModelEvaluation(
                        metrics={"rmse": 1.5, "mae": 1.0, "r2": 0.5},
                        notes=["eval note"],
                    ),
                ),
            )
        ]
    )
    screening = _outcome_from_summary(
        _valid_summary(warnings=["screening warning"])
    )
    result = FinalModelEvaluator(registry).evaluate(
        _regression_split(),
        screening,
        task=AnalysisTask.REGRESSION,
        target_column="y",
        feature_columns=["f1", "f2"],
        leakage_report=_warning_report(),
    )
    assert result.report.fit_seconds >= 0.0
    assert result.report.test_evaluation_seconds >= 0.0
    assert result.report.total_seconds == (
        result.report.fit_seconds + result.report.test_evaluation_seconds
    )
    assert math.isfinite(result.report.fit_seconds)
    assert result.report.evaluated_at.tzinfo is not None
    assert result.report.evaluated_at.utcoffset() is not None

    warnings = result.report.warnings
    assert warnings[0].startswith("LeakageReport")
    assert any("degraded" in warning.lower() for warning in warnings)
    assert "screening warning" in warnings
    assert "eval note" in warnings
    assert len(warnings) == len(set(warnings))

    metadata = result.final_model.get_metadata()
    assert metadata.model_name == result.report.model_name
    assert metadata.task == result.report.task
    assert result.final_model._spec.estimator_key == result.report.estimator_key  # type: ignore[attr-defined]
    preds = result.final_model.predict(_regression_split().test.select(["f1", "f2"]))
    assert preds.shape[0] == 3
    dumped = result.report.model_dump()
    assert "estimator" not in dumped
    assert "train" not in dumped
    assert "predictions" not in dumped

    no_degrade = FinalModelEvaluator(
        registry,
        policy=FinalEvaluationPolicy(warn_on_test_degradation=False),
    ).evaluate(
        _regression_split(),
        screening,
        task=AnalysisTask.REGRESSION,
        target_column="y",
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    assert not any("degraded" in warning.lower() for warning in no_degrade.report.warnings)

    improved = FinalModelEvaluator(
        _registry_with_models(
            [
                (
                    _spec(name="Linear Regression"),
                    ControllableModel(
                        name="Linear Regression",
                        metrics={"rmse": 0.1, "mae": 0.1, "r2": 0.99},
                        estimator_key="linear_regression",
                    ),
                )
            ]
        )
    ).evaluate(
        _regression_split(),
        screening,
        task=AnalysisTask.REGRESSION,
        target_column="y",
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    assert not any("degraded" in warning.lower() for warning in improved.report.warnings)


def test_immutability_determinism_and_integration() -> None:
    registry = create_default_supervised_model_registry(random_state=42)
    split = _regression_split()
    train_before = split.train.clone()
    validation_before = split.validation.clone()
    test_before = split.test.clone()
    features = ["f1", "f2"]
    features_before = list(features)

    screening = SupervisedModelScreener(
        registry,
        policy=ModelScreeningPolicy(maximum_candidates=3),
    ).screen(
        split,
        task=AnalysisTask.REGRESSION,
        target_column="y",
        feature_columns=features,
        leakage_report=_safe_report(),
    )
    selected_before = copy.deepcopy(screening.selected_model.get_metadata())
    summary_before = screening.summary.model_dump()
    specs_before = registry.list_specs(AnalysisTask.REGRESSION)
    candidate_spec_before = screening.summary.candidate_results[
        screening.summary.selected_registry_rank
    ].spec.model_dump()

    evaluator = FinalModelEvaluator(registry)
    first = evaluator.evaluate(
        split,
        screening,
        task=AnalysisTask.REGRESSION,
        target_column="y",
        feature_columns=features,
        leakage_report=_safe_report(),
    )
    second = evaluator.evaluate(
        split,
        screening,
        task=AnalysisTask.REGRESSION,
        target_column="y",
        feature_columns=features,
        leakage_report=_safe_report(),
    )

    assert split.train.equals(train_before)
    assert split.validation.equals(validation_before)
    assert split.test.equals(test_before)
    assert features == features_before
    assert screening.summary.model_dump() == summary_before
    assert screening.selected_model.get_metadata() == selected_before
    assert registry.list_specs(AnalysisTask.REGRESSION) == specs_before
    assert (
        screening.summary.candidate_results[
            screening.summary.selected_registry_rank
        ].spec.model_dump()
        == candidate_spec_before
    )

    assert first.report.model_name == second.report.model_name
    assert first.report.test_metrics == second.report.test_metrics
    assert first.report.validation_metrics is not first.report.test_metrics
    first.report.test_metrics["rmse"] = 999.0
    assert second.report.test_metrics["rmse"] != 999.0

    other = FinalModelEvaluator(registry)
    assert other._policy is not evaluator._policy

    # Changing only test values after screening must not change selected model name
    altered = _regression_split(test_target_offset=500.0)
    altered_screening = ModelScreeningOutcome(
        selected_model=screening.selected_model,
        summary=screening.summary.model_copy(
            update={"test_row_count": altered.test.height}
        ),
    )
    altered_result = FinalModelEvaluator(registry).evaluate(
        altered,
        altered_screening,
        task=AnalysisTask.REGRESSION,
        target_column="y",
        feature_columns=features,
        leakage_report=_safe_report(),
    )
    assert altered_result.report.model_name == first.report.model_name
