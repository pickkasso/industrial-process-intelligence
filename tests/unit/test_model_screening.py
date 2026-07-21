"""Unit tests for supervised model screening (Step 6C)."""

from __future__ import annotations

import copy
import dataclasses
import math
from typing import Self

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
    BaseAnalysisModel,
    BaseIndustryProfile,
    DataFrameLike,
    SeriesLike,
)
from process_intelligence.core.schemas import (
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
    CandidateRunStatus,
    CandidateScreeningResult,
    ModelRegistry,
    ModelScreeningOutcome,
    ModelScreeningPolicy,
    ModelScreeningSummary,
    SupervisedModelScreener,
    create_default_supervised_model_registry,
)
from process_intelligence.models.screening import CandidateRunStatus as DirectStatus


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
            "y": [1, 1, 0, 0],
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
    time_budget_seconds: float | None = 5.0,
) -> ModelSpec:
    return ModelSpec(
        name=name,
        task=task,
        estimator_key=estimator_key,
        optional_dependencies=[],
        priority=priority,
        time_budget_seconds=time_budget_seconds,
    )


def _success_result(
    *,
    name: str = "Linear Regression",
    estimator_key: str = "linear_regression",
    registry_rank: int = 0,
    metrics: dict[str, float] | None = None,
    primary_metric_name: str = "rmse",
    primary_metric_value: float | None = None,
    task: AnalysisTask = AnalysisTask.REGRESSION,
) -> CandidateScreeningResult:
    payload = {"rmse": 1.0, "mae": 0.5, "r2": 0.9} if metrics is None else dict(metrics)
    primary = (
        payload[primary_metric_name]
        if primary_metric_value is None
        else primary_metric_value
    )
    return CandidateScreeningResult(
        spec=_spec(name=name, estimator_key=estimator_key, task=task),
        status=CandidateRunStatus.SUCCESS,
        registry_rank=registry_rank,
        metrics=payload,
        primary_metric_name=primary_metric_name,
        primary_metric_value=primary,
        fit_seconds=0.01,
        evaluation_seconds=0.02,
        total_seconds=0.03,
        warnings=[],
    )


def _failed_result(
    *,
    name: str = "Broken",
    registry_rank: int = 0,
) -> CandidateScreeningResult:
    return CandidateScreeningResult(
        spec=_spec(name=name, estimator_key="broken"),
        status=CandidateRunStatus.FAILED,
        registry_rank=registry_rank,
        metrics={},
        primary_metric_name=None,
        primary_metric_value=None,
        fit_seconds=0.01,
        evaluation_seconds=0.0,
        total_seconds=0.01,
        warnings=[],
        error_type="ValueError",
        error_message="boom",
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
    ) -> None:
        self._name = name
        self._task = task
        self._metrics = (
            {"rmse": 1.0, "mae": 0.5, "r2": 0.8} if metrics is None else dict(metrics)
        )
        self._fit_error = fit_error
        self._evaluate_error = evaluate_error
        self._evaluate_return = evaluate_return
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, X: DataFrameLike, y: SeriesLike | None = None) -> Self:
        _ = (X, y)
        if self._fit_error is not None:
            raise self._fit_error
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
        _ = (X, y)
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


class StubIndustryProfile(BaseIndustryProfile):
    """Minimal industry profile stub."""

    def __init__(self, candidates: list[ModelSpec] | None = None) -> None:
        self._candidates = [] if candidates is None else list(candidates)
        self.mutated = False

    @property
    def industry_name(self) -> str:
        return "stub"

    def score_industry(self, metadata: DatasetMetadata) -> IndustryScore:
        _ = metadata
        return IndustryScore(
            industry_name="stub",
            score=0.5,
            confidence=0.5,
            evidence=[],
            uncertain_factors=[],
            requires_user_confirmation=True,
        )

    def get_schema_hints(self) -> SchemaHints:
        return SchemaHints()

    def get_preprocessing_rules(self) -> PreprocessingRules:
        return PreprocessingRules()

    def get_default_model_candidates(self, task: AnalysisTask) -> list[ModelSpec]:
        return [spec.model_copy(deep=True) for spec in self._candidates if spec.task is task]

    def validate_physical_ranges(self, frame: DataFrameLike) -> list[ValidationIssue]:
        _ = frame
        return []

    def get_recommendation_constraints(self) -> list[VariableConstraint]:
        return []


def _registry_with_models(
    entries: list[tuple[ModelSpec, ControllableModel]],
) -> ModelRegistry:
    registry = ModelRegistry()
    for spec, model in entries:
        captured = model

        def _factory(
            captured_model: ControllableModel = captured,
        ) -> BaseAnalysisModel:
            return ControllableModel(
                name=captured_model._name,
                task=captured_model._task,
                metrics=dict(captured_model._metrics),
                fit_error=captured_model._fit_error,
                evaluate_error=captured_model._evaluate_error,
                evaluate_return=captured_model._evaluate_return,
            )

        registry.register_factory(spec.estimator_key, _factory, replace=True)
        registry.register(spec)
    return registry


def _screen_default(
    *,
    task: AnalysisTask = AnalysisTask.REGRESSION,
    policy: ModelScreeningPolicy | None = None,
    leakage_report: LeakageReport | None = None,
    split: DatasetSplit | None = None,
    registry: ModelRegistry | None = None,
    feature_columns: list[str] | None = None,
    industry_profile: BaseIndustryProfile | None = None,
) -> ModelScreeningOutcome:
    features = ["f1", "f2"] if feature_columns is None else list(feature_columns)
    if split is None:
        split = (
            _regression_split()
            if task is AnalysisTask.REGRESSION
            else _classification_split()
        )
    if registry is None:
        registry = create_default_supervised_model_registry(random_state=42)
    screener = SupervisedModelScreener(registry, policy=policy)
    return screener.screen(
        split,
        task=task,
        target_column="y",
        feature_columns=features,
        leakage_report=(
            _safe_report(features) if leakage_report is None else leakage_report
        ),
        industry_profile=industry_profile,
    )


def test_candidate_run_status_values_and_members() -> None:
    assert CandidateRunStatus.SUCCESS == "SUCCESS"
    assert CandidateRunStatus.FAILED == "FAILED"
    assert CandidateRunStatus("SUCCESS") is CandidateRunStatus.SUCCESS
    assert CandidateRunStatus("FAILED") is CandidateRunStatus.FAILED
    with pytest.raises(ValueError):
        CandidateRunStatus("SKIPPED")
    assert {member.name for member in CandidateRunStatus} == {"SUCCESS", "FAILED"}
    assert DirectStatus is CandidateRunStatus


def test_policy_defaults_and_validation() -> None:
    policy = ModelScreeningPolicy()
    assert policy.maximum_candidates is None
    assert policy.candidate_time_budget_seconds is None
    assert policy.require_safe_leakage_report is True
    assert policy.require_validation_partition is True
    assert policy.continue_on_candidate_failure is True
    assert policy.require_successful_baseline is True

    with pytest.raises(ValidationError):
        ModelScreeningPolicy(maximum_candidates=0)
    with pytest.raises(ValidationError):
        ModelScreeningPolicy(maximum_candidates=-1)
    with pytest.raises(ValidationError):
        ModelScreeningPolicy(maximum_candidates=True)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ModelScreeningPolicy(candidate_time_budget_seconds=0)
    with pytest.raises(ValidationError):
        ModelScreeningPolicy(candidate_time_budget_seconds=-1.0)
    with pytest.raises(ValidationError):
        ModelScreeningPolicy(candidate_time_budget_seconds=True)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ModelScreeningPolicy(candidate_time_budget_seconds=float("nan"))
    with pytest.raises(ValidationError):
        ModelScreeningPolicy(candidate_time_budget_seconds=float("inf"))
    with pytest.raises(ValidationError):
        ModelScreeningPolicy(candidate_time_budget_seconds=float("-inf"))
    with pytest.raises(ValidationError):
        ModelScreeningPolicy(require_safe_leakage_report=1)  # type: ignore[arg-type]

    restored = ModelScreeningPolicy.model_validate(policy.model_dump())
    assert restored == policy


def test_candidate_result_success_and_failed_validation() -> None:
    success = _success_result()
    failed = _failed_result()
    assert success.status is CandidateRunStatus.SUCCESS
    assert failed.status is CandidateRunStatus.FAILED

    with pytest.raises(ValidationError):
        CandidateScreeningResult(
            spec=_spec(name="A"),
            status=CandidateRunStatus.FAILED,
            registry_rank=-1,
            metrics={},
            fit_seconds=0.0,
            evaluation_seconds=0.0,
            total_seconds=0.0,
            error_type="ValueError",
            error_message="x",
        )
    with pytest.raises(ValidationError):
        CandidateScreeningResult(
            spec=_spec(name="A"),
            status=CandidateRunStatus.FAILED,
            registry_rank=0,
            metrics={},
            fit_seconds=-0.1,
            evaluation_seconds=0.0,
            total_seconds=0.0,
            error_type="ValueError",
            error_message="x",
        )
    with pytest.raises(ValidationError):
        CandidateScreeningResult(
            spec=_spec(name="A"),
            status=CandidateRunStatus.FAILED,
            registry_rank=0,
            metrics={},
            fit_seconds=float("nan"),
            evaluation_seconds=0.0,
            total_seconds=0.0,
            error_type="ValueError",
            error_message="x",
        )
    with pytest.raises(ValidationError):
        CandidateScreeningResult(
            spec=_spec(name="A"),
            status=CandidateRunStatus.FAILED,
            registry_rank=0,
            metrics={},
            fit_seconds=float("inf"),
            evaluation_seconds=0.0,
            total_seconds=0.0,
            error_type="ValueError",
            error_message="x",
        )
    with pytest.raises(ValidationError):
        CandidateScreeningResult(
            spec=_spec(name="A"),
            status=CandidateRunStatus.SUCCESS,
            registry_rank=0,
            metrics={"": 1.0},
            primary_metric_name="rmse",
            primary_metric_value=1.0,
            fit_seconds=0.0,
            evaluation_seconds=0.0,
            total_seconds=0.0,
        )
    with pytest.raises(ValidationError):
        CandidateScreeningResult(
            spec=_spec(name="A"),
            status=CandidateRunStatus.SUCCESS,
            registry_rank=0,
            metrics={"rmse": float("nan"), "mae": 0.1},
            primary_metric_name="rmse",
            primary_metric_value=1.0,
            fit_seconds=0.0,
            evaluation_seconds=0.0,
            total_seconds=0.0,
        )
    with pytest.raises(ValidationError):
        CandidateScreeningResult(
            spec=_spec(name="A"),
            status=CandidateRunStatus.SUCCESS,
            registry_rank=0,
            metrics={"rmse": float("inf"), "mae": 0.1},
            primary_metric_name="rmse",
            primary_metric_value=1.0,
            fit_seconds=0.0,
            evaluation_seconds=0.0,
            total_seconds=0.0,
        )
    with pytest.raises(ValidationError):
        CandidateScreeningResult(
            spec=_spec(name="A"),
            status=CandidateRunStatus.SUCCESS,
            registry_rank=0,
            metrics={},
            primary_metric_name="rmse",
            primary_metric_value=1.0,
            fit_seconds=0.0,
            evaluation_seconds=0.0,
            total_seconds=0.0,
        )
    with pytest.raises(ValidationError):
        CandidateScreeningResult(
            spec=_spec(name="A"),
            status=CandidateRunStatus.SUCCESS,
            registry_rank=0,
            metrics={"rmse": 1.0, "mae": 0.5},
            primary_metric_name=None,
            primary_metric_value=1.0,
            fit_seconds=0.0,
            evaluation_seconds=0.0,
            total_seconds=0.0,
        )
    with pytest.raises(ValidationError):
        CandidateScreeningResult(
            spec=_spec(name="A"),
            status=CandidateRunStatus.SUCCESS,
            registry_rank=0,
            metrics={"rmse": 1.0, "mae": 0.5},
            primary_metric_name="rmse",
            primary_metric_value=1.0,
            fit_seconds=0.0,
            evaluation_seconds=0.0,
            total_seconds=0.0,
            error_type="ValueError",
        )
    with pytest.raises(ValidationError):
        CandidateScreeningResult(
            spec=_spec(name="A"),
            status=CandidateRunStatus.FAILED,
            registry_rank=0,
            metrics={"rmse": 1.0},
            fit_seconds=0.0,
            evaluation_seconds=0.0,
            total_seconds=0.0,
            error_type="ValueError",
            error_message="x",
        )
    with pytest.raises(ValidationError):
        CandidateScreeningResult(
            spec=_spec(name="A"),
            status=CandidateRunStatus.FAILED,
            registry_rank=0,
            metrics={},
            fit_seconds=0.0,
            evaluation_seconds=0.0,
            total_seconds=0.0,
            error_type=None,
            error_message="x",
        )
    with pytest.raises(ValidationError):
        CandidateScreeningResult(
            spec=_spec(name="A"),
            status=CandidateRunStatus.FAILED,
            registry_rank=0,
            metrics={},
            fit_seconds=0.0,
            evaluation_seconds=0.0,
            total_seconds=0.0,
            error_type="ValueError",
            error_message=None,
        )
    with pytest.raises(ValidationError):
        CandidateScreeningResult(
            spec=_spec(name="A"),
            status=CandidateRunStatus.FAILED,
            registry_rank=0,
            metrics={},
            fit_seconds=0.0,
            evaluation_seconds=0.0,
            total_seconds=0.0,
            warnings=["a", "a"],
            error_type="ValueError",
            error_message="x",
        )
    assert CandidateScreeningResult.model_validate(success.model_dump()) == success


def test_summary_validation_and_round_trip() -> None:
    summary = _valid_summary()
    assert summary.selected_model_name == "Linear Regression"

    with pytest.raises(ValidationError):
        _valid_summary(target_column="   ")
    with pytest.raises(ValidationError):
        _valid_summary(feature_columns=[])
    with pytest.raises(ValidationError):
        _valid_summary(feature_columns=["f1", "f1"])
    with pytest.raises(ValidationError):
        _valid_summary(selected_metrics={})
    with pytest.raises(ValidationError):
        _valid_summary(ranking_metric="missing")
    with pytest.raises(ValidationError):
        _valid_summary(selected_model_name="Missing Model")
    with pytest.raises(ValidationError):
        _valid_summary(selected_registry_rank=99)
    with pytest.raises(ValidationError):
        _valid_summary(baseline_model_name="Not Present")
    with pytest.raises(ValidationError):
        _valid_summary(
            baseline_model_name=None,
            baseline_metrics={"rmse": 1.0},
            selected_beats_baseline=None,
        )
    with pytest.raises(ValidationError):
        _valid_summary(successful_model_names=["A", "A"])
    with pytest.raises(ValidationError):
        _valid_summary(
            candidate_results=[_success_result(), _failed_result(name="Broken")],
            selected_model_name="Linear Regression",
            selected_registry_rank=0,
            successful_model_names=["Linear Regression"],
            failed_model_names=["Broken", "Broken"],
            baseline_model_name=None,
            baseline_metrics={},
            selected_beats_baseline=None,
        )
    with pytest.raises(ValidationError):
        _valid_summary(
            candidate_results=[_success_result(), _failed_result(name="Broken")],
            selected_model_name="Linear Regression",
            selected_registry_rank=0,
            successful_model_names=["Linear Regression", "Broken"],
            failed_model_names=["Broken"],
            baseline_model_name=None,
            baseline_metrics={},
            selected_beats_baseline=None,
        )
    with pytest.raises(ValidationError):
        _valid_summary(validation_row_count=0)
    with pytest.raises(ValidationError):
        _valid_summary(warnings=["dup", "dup"])

    assert ModelScreeningSummary.model_validate(summary.model_dump()) == summary


def test_outcome_frozen_dataclass() -> None:
    model = ControllableModel(name="Linear Regression")
    model.fit(pl.DataFrame({"f1": [1.0]}), pl.Series("y", [1.0]))
    summary = _valid_summary(
        candidate_results=[_success_result()],
        selected_registry_rank=0,
        baseline_model_name=None,
        baseline_metrics={},
        selected_beats_baseline=None,
        successful_model_names=["Linear Regression"],
        failed_model_names=[],
    )
    outcome = ModelScreeningOutcome(selected_model=model, summary=summary)
    assert dataclasses.is_dataclass(outcome)
    assert outcome.__slots__ == ("selected_model", "summary")
    assert isinstance(outcome.selected_model, BaseAnalysisModel)
    assert isinstance(outcome.summary, ModelScreeningSummary)
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.selected_model = model  # type: ignore[misc]


def test_screener_construction_and_policy_isolation() -> None:
    registry = create_default_supervised_model_registry()
    policy = ModelScreeningPolicy(maximum_candidates=2)
    screener = SupervisedModelScreener(registry, policy=policy)
    assert isinstance(screener, SupervisedModelScreener)

    with pytest.raises(TypeError):
        SupervisedModelScreener("not-registry")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        SupervisedModelScreener(registry, policy="bad")  # type: ignore[arg-type]

    policy.maximum_candidates = 1
    outcome = screener.screen(
        _regression_split(),
        task=AnalysisTask.REGRESSION,
        target_column="y",
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    assert len(outcome.summary.candidate_results) == 2


def test_screen_input_type_and_schema_validation() -> None:
    registry = create_default_supervised_model_registry()
    screener = SupervisedModelScreener(registry)
    split = _regression_split()
    report = _safe_report()

    with pytest.raises(TypeError):
        screener.screen(
            "bad",  # type: ignore[arg-type]
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=["f1", "f2"],
            leakage_report=report,
        )

    bad_split = DatasetSplit(
        train=split.train,
        validation="bad",  # type: ignore[arg-type]
        test=split.test,
        summary=split.summary,
    )
    with pytest.raises(TypeError):
        screener.screen(
            bad_split,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=["f1", "f2"],
            leakage_report=report,
        )

    mismatched = DatasetSplit(
        train=split.train,
        validation=split.validation.rename({"f1": "fx"}),
        test=split.test,
        summary=split.summary,
    )
    with pytest.raises(DataValidationError):
        screener.screen(
            mismatched,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=["f1", "f2"],
            leakage_report=report,
        )

    missing_id = DatasetSplit(
        train=split.train.drop(ORIGINAL_ROW_ID_COLUMN),
        validation=split.validation.drop(ORIGINAL_ROW_ID_COLUMN),
        test=split.test.drop(ORIGINAL_ROW_ID_COLUMN),
        summary=split.summary,
    )
    with pytest.raises(DataValidationError):
        screener.screen(
            missing_id,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=["f1", "f2"],
            leakage_report=report,
        )

    with pytest.raises(TypeError):
        screener.screen(
            split,
            task="REGRESSION",  # type: ignore[arg-type]
            target_column="y",
            feature_columns=["f1", "f2"],
            leakage_report=report,
        )
    with pytest.raises(DataValidationError):
        screener.screen(
            split,
            task=AnalysisTask.UNSUPERVISED_ANOMALY,
            target_column="y",
            feature_columns=["f1", "f2"],
            leakage_report=report,
        )
    with pytest.raises(TypeError):
        screener.screen(
            split,
            task=AnalysisTask.REGRESSION,
            target_column=1,  # type: ignore[arg-type]
            feature_columns=["f1", "f2"],
            leakage_report=report,
        )
    with pytest.raises(DataValidationError):
        screener.screen(
            split,
            task=AnalysisTask.REGRESSION,
            target_column="  ",
            feature_columns=["f1", "f2"],
            leakage_report=report,
        )
    with pytest.raises(DataValidationError):
        screener.screen(
            split,
            task=AnalysisTask.REGRESSION,
            target_column="missing",
            feature_columns=["f1", "f2"],
            leakage_report=report,
        )
    with pytest.raises(DataValidationError):
        screener.screen(
            split,
            task=AnalysisTask.REGRESSION,
            target_column=ORIGINAL_ROW_ID_COLUMN,
            feature_columns=["f1", "f2"],
            leakage_report=report,
        )
    with pytest.raises(TypeError):
        screener.screen(
            split,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns="f1",  # type: ignore[arg-type]
            leakage_report=report,
        )
    with pytest.raises(TypeError):
        screener.screen(
            split,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=b"f1",  # type: ignore[arg-type]
            leakage_report=report,
        )
    with pytest.raises(TypeError):
        screener.screen(
            split,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=["f1", 1],  # type: ignore[list-item]
            leakage_report=report,
        )
    with pytest.raises(DataValidationError):
        screener.screen(
            split,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=["f1", "  "],
            leakage_report=report,
        )
    with pytest.raises(DataValidationError):
        screener.screen(
            split,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=["f1", "f1"],
            leakage_report=report,
        )
    with pytest.raises(DataValidationError):
        screener.screen(
            split,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=[],
            leakage_report=report,
        )
    with pytest.raises(DataValidationError):
        screener.screen(
            split,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=[ORIGINAL_ROW_ID_COLUMN, "f1"],
            leakage_report=report,
        )
    with pytest.raises(DataValidationError):
        screener.screen(
            split,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=["f1", "y"],
            leakage_report=report,
        )
    with pytest.raises(DataValidationError):
        screener.screen(
            split,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=["f1", "missing"],
            leakage_report=report,
        )
    with pytest.raises(TypeError):
        screener.screen(
            split,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=["f1", "f2"],
            leakage_report="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        screener.screen(
            split,
            task=AnalysisTask.REGRESSION,
            target_column="y",
            feature_columns=["f1", "f2"],
            leakage_report=report,
            industry_profile="bad",  # type: ignore[arg-type]
        )


def test_leakage_gate_and_feature_alignment() -> None:
    registry = create_default_supervised_model_registry()
    with pytest.raises(DataLeakageError):
        _screen_default(registry=registry, leakage_report=_blocker_report())

    outcome = _screen_default(registry=registry, leakage_report=_warning_report())
    assert any("warning" in warning.casefold() for warning in outcome.summary.warnings)

    safe_outcome = _screen_default(
        registry=registry,
        leakage_report=_safe_report(),
    )
    assert safe_outcome.summary.candidate_results

    assert _screen_default(
        registry=registry,
        leakage_report=_blocker_report(),
        policy=ModelScreeningPolicy(require_safe_leakage_report=False),
    ).summary.candidate_results

    with pytest.raises(DataValidationError):
        _screen_default(registry=registry, leakage_report=_safe_report(["f2", "f1"]))
    with pytest.raises(DataValidationError):
        _screen_default(registry=registry, leakage_report=_safe_report(["f1"]))

    report = _safe_report()
    before = report.model_copy(deep=True)
    _screen_default(registry=registry, leakage_report=report)
    assert report.model_dump() == before.model_dump()


def test_partition_usage_and_test_isolation() -> None:
    registry = create_default_supervised_model_registry()
    split = _regression_split()

    empty_train = DatasetSplit(
        train=split.train.clear(),
        validation=split.validation,
        test=split.test,
        summary=split.summary,
    )
    with pytest.raises(InsufficientDataError):
        _screen_default(registry=registry, split=empty_train)

    empty_validation = DatasetSplit(
        train=split.train,
        validation=split.validation.clear(),
        test=split.test,
        summary=split.summary,
    )
    with pytest.raises(InsufficientDataError):
        _screen_default(registry=registry, split=empty_validation)
    with pytest.raises(InsufficientDataError):
        _screen_default(
            registry=registry,
            split=empty_validation,
            policy=ModelScreeningPolicy(require_validation_partition=False),
        )

    features = ["f2", "f1"]
    outcome = _screen_default(
        registry=registry,
        feature_columns=features,
        leakage_report=_safe_report(features),
    )
    assert outcome.summary.feature_columns == ["f2", "f1"]

    selected_a = _screen_default(
        registry=registry,
        split=_regression_split(test_target_offset=1000.0),
    )
    selected_b = _screen_default(
        registry=registry,
        split=_regression_split(test_target_offset=-500.0),
    )
    assert selected_a.summary.selected_model_name == selected_b.summary.selected_model_name
    assert selected_a.summary.selected_metrics == selected_b.summary.selected_metrics

    selected_c = _screen_default(
        registry=registry,
        split=_regression_split(test_feature_offset=50.0),
    )
    assert selected_c.summary.selected_model_name == selected_a.summary.selected_model_name
    assert selected_c.summary.test_row_count == 3
    assert selected_c.summary.validation_row_count == 3

    train_before = split.train.clone()
    validation_before = split.validation.clone()
    test_before = split.test.clone()
    _screen_default(registry=registry, split=split)
    assert split.train.equals(train_before)
    assert split.validation.equals(validation_before)
    assert split.test.equals(test_before)


def test_candidate_lookup_limits_and_registry_immutability() -> None:
    registry = create_default_supervised_model_registry()
    before_specs = registry.list_specs(AnalysisTask.REGRESSION)

    outcome = _screen_default(registry=registry)
    assert len(outcome.summary.candidate_results) == 4
    assert all(
        item.spec.task is AnalysisTask.REGRESSION
        for item in outcome.summary.candidate_results
    )

    with pytest.raises(ProcessIntelligenceError):
        _screen_default(registry=ModelRegistry())

    limited = _screen_default(
        registry=registry,
        policy=ModelScreeningPolicy(maximum_candidates=2),
    )
    assert len(limited.summary.candidate_results) == 2
    assert [item.registry_rank for item in limited.summary.candidate_results] == [0, 1]

    expensive = ModelRegistry()
    expensive.register_factory(
        "dummy_regressor",
        lambda: ControllableModel(
            name="Dummy Regressor",
            metrics={"rmse": 2.0, "mae": 1.5, "r2": 0.0},
        ),
    )
    expensive.register_factory(
        "linear_regression",
        lambda: ControllableModel(
            name="Linear Regression",
            metrics={"rmse": 1.0, "mae": 0.5, "r2": 0.9},
        ),
    )
    expensive.register(
        _spec(
            name="Dummy Regressor",
            estimator_key="dummy_regressor",
            priority=10,
            time_budget_seconds=1.0,
        )
    )
    expensive.register(
        _spec(
            name="Linear Regression",
            estimator_key="linear_regression",
            priority=20,
            time_budget_seconds=100.0,
        )
    )
    filtered = _screen_default(
        registry=expensive,
        policy=ModelScreeningPolicy(candidate_time_budget_seconds=5.0),
    )
    assert [item.spec.name for item in filtered.summary.candidate_results] == [
        "Dummy Regressor"
    ]

    assert [item.spec.name for item in outcome.summary.candidate_results] == [
        spec.name for spec in before_specs
    ]
    assert registry.list_specs(AnalysisTask.REGRESSION) == before_specs


def test_regression_screening_ranking_and_baseline() -> None:
    registry = create_default_supervised_model_registry(random_state=42)
    outcome = _screen_default(registry=registry)
    assert outcome.selected_model.is_fitted is True
    success = [
        item
        for item in outcome.summary.candidate_results
        if item.status is CandidateRunStatus.SUCCESS
    ]
    assert "rmse" in success[0].metrics
    assert "mae" in success[0].metrics
    assert outcome.summary.ranking_metric == "rmse"
    assert outcome.summary.higher_is_better is False
    assert (
        outcome.summary.selected_model_name
        == outcome.selected_model.get_metadata().model_name
    )
    assert outcome.summary.baseline_model_name == "Dummy Regressor"
    assert outcome.summary.baseline_metrics
    assert isinstance(outcome.summary.selected_beats_baseline, bool)

    registry = _registry_with_models(
        [
            (
                _spec(name="Dummy Regressor", estimator_key="dummy_regressor", priority=10),
                ControllableModel(
                    name="Dummy Regressor",
                    metrics={"rmse": 3.0, "mae": 2.0, "r2": 0.0},
                ),
            ),
            (
                _spec(name="High RMSE", estimator_key="high", priority=20),
                ControllableModel(
                    name="High RMSE",
                    metrics={"rmse": 2.0, "mae": 0.1, "r2": 0.99},
                ),
            ),
            (
                _spec(name="Low RMSE", estimator_key="low", priority=30),
                ControllableModel(
                    name="Low RMSE",
                    metrics={"rmse": 1.0, "mae": 0.9, "r2": 0.1},
                ),
            ),
        ]
    )
    outcome = _screen_default(registry=registry)
    assert outcome.summary.selected_model_name == "Low RMSE"
    assert outcome.summary.selected_beats_baseline is True

    registry = _registry_with_models(
        [
            (
                _spec(name="Dummy Regressor", estimator_key="dummy_regressor", priority=10),
                ControllableModel(
                    name="Dummy Regressor",
                    metrics={"rmse": 5.0, "mae": 4.0, "r2": 0.0},
                ),
            ),
            (
                _spec(name="Worse MAE", estimator_key="a", priority=20),
                ControllableModel(
                    name="Worse MAE",
                    metrics={"rmse": 1.0, "mae": 0.8, "r2": 0.5},
                ),
            ),
            (
                _spec(name="Better MAE", estimator_key="b", priority=30),
                ControllableModel(
                    name="Better MAE",
                    metrics={"rmse": 1.0, "mae": 0.2, "r2": 0.1},
                ),
            ),
        ]
    )
    assert _screen_default(registry=registry).summary.selected_model_name == "Better MAE"

    registry = _registry_with_models(
        [
            (
                _spec(name="Dummy Regressor", estimator_key="dummy_regressor", priority=10),
                ControllableModel(
                    name="Dummy Regressor",
                    metrics={"rmse": 5.0, "mae": 4.0, "r2": 0.0},
                ),
            ),
            (
                _spec(name="Low R2", estimator_key="a", priority=20),
                ControllableModel(
                    name="Low R2",
                    metrics={"rmse": 1.0, "mae": 0.5, "r2": 0.2},
                ),
            ),
            (
                _spec(name="High R2", estimator_key="b", priority=30),
                ControllableModel(
                    name="High R2",
                    metrics={"rmse": 1.0, "mae": 0.5, "r2": 0.9},
                ),
            ),
        ]
    )
    assert _screen_default(registry=registry).summary.selected_model_name == "High R2"

    registry = _registry_with_models(
        [
            (
                _spec(name="Dummy Regressor", estimator_key="dummy_regressor", priority=10),
                ControllableModel(
                    name="Dummy Regressor",
                    metrics={"rmse": 5.0, "mae": 4.0, "r2": 0.0},
                ),
            ),
            (
                _spec(name="First", estimator_key="a", priority=20),
                ControllableModel(
                    name="First",
                    metrics={"rmse": 1.0, "mae": 0.5, "r2": 0.5},
                ),
            ),
            (
                _spec(name="Second", estimator_key="b", priority=30),
                ControllableModel(
                    name="Second",
                    metrics={"rmse": 1.0, "mae": 0.5, "r2": 0.5},
                ),
            ),
        ]
    )
    tied = _screen_default(registry=registry)
    assert tied.summary.selected_model_name == "First"
    assert tied.summary.selected_registry_rank == 1
    assert tied.summary.selected_metrics["rmse"] == 1.0

    registry = _registry_with_models(
        [
            (
                _spec(name="Dummy Regressor", estimator_key="dummy_regressor", priority=10),
                ControllableModel(
                    name="Dummy Regressor",
                    metrics={"rmse": 1.0, "mae": 0.5, "r2": 0.0},
                ),
            ),
            (
                _spec(name="Same", estimator_key="same", priority=20),
                ControllableModel(
                    name="Same",
                    metrics={"rmse": 1.0, "mae": 0.4, "r2": 0.5},
                ),
            ),
        ]
    )
    equal = _screen_default(registry=registry)
    assert equal.summary.selected_model_name == "Same"
    assert equal.summary.selected_beats_baseline is False

    registry = _registry_with_models(
        [
            (
                _spec(name="Dummy Regressor", estimator_key="dummy_regressor", priority=10),
                ControllableModel(
                    name="Dummy Regressor",
                    metrics={"rmse": 0.1, "mae": 0.1, "r2": 0.99},
                ),
            ),
            (
                _spec(name="Worse", estimator_key="worse", priority=20),
                ControllableModel(
                    name="Worse",
                    metrics={"rmse": 2.0, "mae": 1.0, "r2": 0.1},
                ),
            ),
        ]
    )
    baseline_selected = _screen_default(registry=registry)
    assert baseline_selected.summary.selected_model_name == "Dummy Regressor"
    assert baseline_selected.summary.selected_beats_baseline is False


def test_classification_screening_ranking_and_baseline() -> None:
    registry = create_default_supervised_model_registry(random_state=42)
    outcome = _screen_default(task=AnalysisTask.CLASSIFICATION, registry=registry)
    success = [
        item
        for item in outcome.summary.candidate_results
        if item.status is CandidateRunStatus.SUCCESS
    ]
    assert "f1_macro" in success[0].metrics
    assert "balanced_accuracy" in success[0].metrics
    assert "accuracy" in success[0].metrics
    assert outcome.summary.ranking_metric == "f1_macro"
    assert outcome.summary.higher_is_better is True
    assert outcome.summary.baseline_model_name == "Dummy Classifier"

    def _cls(
        name: str,
        key: str,
        priority: int,
        metrics: dict[str, float],
    ) -> tuple[ModelSpec, ControllableModel]:
        return (
            _spec(
                name=name,
                task=AnalysisTask.CLASSIFICATION,
                estimator_key=key,
                priority=priority,
            ),
            ControllableModel(
                name=name,
                task=AnalysisTask.CLASSIFICATION,
                metrics=metrics,
            ),
        )

    registry = _registry_with_models(
        [
            _cls(
                "Dummy Classifier",
                "dummy_classifier",
                10,
                {"f1_macro": 0.2, "balanced_accuracy": 0.2, "accuracy": 0.2},
            ),
            _cls(
                "Low F1",
                "low",
                20,
                {"f1_macro": 0.5, "balanced_accuracy": 0.9, "accuracy": 0.9},
            ),
            _cls(
                "High F1",
                "high",
                30,
                {"f1_macro": 0.8, "balanced_accuracy": 0.4, "accuracy": 0.4},
            ),
        ]
    )
    selected = _screen_default(task=AnalysisTask.CLASSIFICATION, registry=registry)
    assert selected.summary.selected_model_name == "High F1"
    assert selected.summary.selected_beats_baseline is True

    registry = _registry_with_models(
        [
            _cls(
                "Dummy Classifier",
                "dummy_classifier",
                10,
                {"f1_macro": 0.1, "balanced_accuracy": 0.1, "accuracy": 0.1},
            ),
            _cls(
                "Worse BA",
                "a",
                20,
                {"f1_macro": 0.7, "balanced_accuracy": 0.4, "accuracy": 0.9},
            ),
            _cls(
                "Better BA",
                "b",
                30,
                {"f1_macro": 0.7, "balanced_accuracy": 0.8, "accuracy": 0.2},
            ),
        ]
    )
    ba_outcome = _screen_default(
        task=AnalysisTask.CLASSIFICATION,
        registry=registry,
    )
    assert ba_outcome.summary.selected_model_name == "Better BA"

    registry = _registry_with_models(
        [
            _cls(
                "Dummy Classifier",
                "dummy_classifier",
                10,
                {"f1_macro": 0.1, "balanced_accuracy": 0.1, "accuracy": 0.1},
            ),
            _cls(
                "Worse Acc",
                "a",
                20,
                {"f1_macro": 0.7, "balanced_accuracy": 0.7, "accuracy": 0.4},
            ),
            _cls(
                "Better Acc",
                "b",
                30,
                {"f1_macro": 0.7, "balanced_accuracy": 0.7, "accuracy": 0.9},
            ),
        ]
    )
    acc_outcome = _screen_default(
        task=AnalysisTask.CLASSIFICATION,
        registry=registry,
    )
    assert acc_outcome.summary.selected_model_name == "Better Acc"

    registry = _registry_with_models(
        [
            _cls(
                "Dummy Classifier",
                "dummy_classifier",
                10,
                {"f1_macro": 0.1, "balanced_accuracy": 0.1, "accuracy": 0.1},
            ),
            _cls(
                "First",
                "a",
                20,
                {"f1_macro": 0.7, "balanced_accuracy": 0.7, "accuracy": 0.7},
            ),
            _cls(
                "Second",
                "b",
                30,
                {"f1_macro": 0.7, "balanced_accuracy": 0.7, "accuracy": 0.7},
            ),
        ]
    )
    tied = _screen_default(task=AnalysisTask.CLASSIFICATION, registry=registry)
    assert tied.summary.selected_model_name == "First"
    assert tied.summary.selected_registry_rank == 1

    registry = _registry_with_models(
        [
            _cls(
                "Dummy Classifier",
                "dummy_classifier",
                10,
                {"f1_macro": 0.7, "balanced_accuracy": 0.2, "accuracy": 0.2},
            ),
            _cls(
                "Same F1",
                "same",
                20,
                {"f1_macro": 0.7, "balanced_accuracy": 0.9, "accuracy": 0.9},
            ),
        ]
    )
    equal = _screen_default(task=AnalysisTask.CLASSIFICATION, registry=registry)
    assert equal.summary.selected_model_name == "Same F1"
    assert equal.summary.selected_beats_baseline is False


def test_metric_validation_and_candidate_failure_isolation() -> None:
    registry = _registry_with_models(
        [
            (
                _spec(name="Dummy Regressor", estimator_key="dummy_regressor", priority=10),
                ControllableModel(
                    name="Dummy Regressor",
                    metrics={"rmse": 2.0, "mae": 1.0, "r2": 0.0},
                ),
            ),
            (
                _spec(name="Bad Type", estimator_key="bad_type", priority=20),
                ControllableModel(name="Bad Type", evaluate_return={"rmse": 0.1}),
            ),
        ]
    )
    outcome = _screen_default(registry=registry)
    failed = next(
        item for item in outcome.summary.candidate_results if item.spec.name == "Bad Type"
    )
    assert failed.status is CandidateRunStatus.FAILED
    assert failed.error_type
    assert failed.error_message

    registry = _registry_with_models(
        [
            (
                _spec(name="Dummy Regressor", estimator_key="dummy_regressor", priority=10),
                ControllableModel(
                    name="Dummy Regressor",
                    metrics={"rmse": 2.0, "mae": 1.0, "r2": 0.0},
                ),
            ),
            (
                _spec(name="Missing RMSE", estimator_key="missing", priority=20),
                ControllableModel(
                    name="Missing RMSE",
                    metrics={"mae": 0.1, "r2": 0.9},
                ),
            ),
        ]
    )
    assert "Missing RMSE" in _screen_default(registry=registry).summary.failed_model_names

    registry = _registry_with_models(
        [
            (
                _spec(
                    name="Dummy Classifier",
                    task=AnalysisTask.CLASSIFICATION,
                    estimator_key="dummy_classifier",
                    priority=10,
                ),
                ControllableModel(
                    name="Dummy Classifier",
                    task=AnalysisTask.CLASSIFICATION,
                    metrics={
                        "f1_macro": 0.2,
                        "balanced_accuracy": 0.2,
                        "accuracy": 0.2,
                    },
                ),
            ),
            (
                _spec(
                    name="Missing F1",
                    task=AnalysisTask.CLASSIFICATION,
                    estimator_key="missing",
                    priority=20,
                ),
                ControllableModel(
                    name="Missing F1",
                    task=AnalysisTask.CLASSIFICATION,
                    metrics={"balanced_accuracy": 0.9, "accuracy": 0.9},
                ),
            ),
        ]
    )
    missing_f1 = _screen_default(
        task=AnalysisTask.CLASSIFICATION,
        registry=registry,
    )
    assert "Missing F1" in missing_f1.summary.failed_model_names

    for bad in (float("nan"), float("inf")):
        registry = _registry_with_models(
            [
                (
                    _spec(
                        name="Dummy Regressor",
                        estimator_key="dummy_regressor",
                        priority=10,
                    ),
                    ControllableModel(
                        name="Dummy Regressor",
                        metrics={"rmse": 2.0, "mae": 1.0, "r2": 0.0},
                    ),
                ),
                (
                    _spec(name="Bad Metric", estimator_key="bad", priority=20),
                    ControllableModel(
                        name="Bad Metric",
                        metrics={"rmse": bad, "mae": 0.1, "r2": 0.1},
                    ),
                ),
            ]
        )
        assert "Bad Metric" in _screen_default(registry=registry).summary.failed_model_names

    for metrics in (
        {"rmse": -1.0, "mae": 0.1, "r2": 0.1},
        {"rmse": 1.0, "mae": -0.1, "r2": 0.1},
    ):
        registry = _registry_with_models(
            [
                (
                    _spec(
                        name="Dummy Regressor",
                        estimator_key="dummy_regressor",
                        priority=10,
                    ),
                    ControllableModel(
                        name="Dummy Regressor",
                        metrics={"rmse": 2.0, "mae": 1.0, "r2": 0.0},
                    ),
                ),
                (
                    _spec(name="Neg", estimator_key="neg", priority=20),
                    ControllableModel(name="Neg", metrics=metrics),
                ),
            ]
        )
        assert "Neg" in _screen_default(registry=registry).summary.failed_model_names

    for metrics in (
        {"f1_macro": -0.1, "balanced_accuracy": 0.5, "accuracy": 0.5},
        {"f1_macro": 1.1, "balanced_accuracy": 0.5, "accuracy": 0.5},
    ):
        registry = _registry_with_models(
            [
                (
                    _spec(
                        name="Dummy Classifier",
                        task=AnalysisTask.CLASSIFICATION,
                        estimator_key="dummy_classifier",
                        priority=10,
                    ),
                    ControllableModel(
                        name="Dummy Classifier",
                        task=AnalysisTask.CLASSIFICATION,
                        metrics={
                            "f1_macro": 0.2,
                            "balanced_accuracy": 0.2,
                            "accuracy": 0.2,
                        },
                    ),
                ),
                (
                    _spec(
                        name="Out",
                        task=AnalysisTask.CLASSIFICATION,
                        estimator_key="out",
                        priority=20,
                    ),
                    ControllableModel(
                        name="Out",
                        task=AnalysisTask.CLASSIFICATION,
                        metrics=metrics,
                    ),
                ),
            ]
        )
        assert (
            "Out"
            in _screen_default(
                task=AnalysisTask.CLASSIFICATION, registry=registry
            ).summary.failed_model_names
        )

    registry = _registry_with_models(
        [
            (
                _spec(name="Dummy Regressor", estimator_key="dummy_regressor", priority=10),
                ControllableModel(
                    name="Dummy Regressor",
                    metrics={"rmse": 2.0, "mae": 1.0, "r2": 0.0},
                ),
            ),
            (
                _spec(name="Neg R2", estimator_key="neg_r2", priority=20),
                ControllableModel(
                    name="Neg R2",
                    metrics={"rmse": 0.5, "mae": 0.4, "r2": -1.5},
                ),
            ),
        ]
    )
    outcome = _screen_default(registry=registry)
    assert outcome.summary.selected_model_name == "Neg R2"
    assert outcome.summary.selected_metrics["r2"] == -1.5

    registry = _registry_with_models(
        [
            (
                _spec(name="Dummy Regressor", estimator_key="dummy_regressor", priority=10),
                ControllableModel(
                    name="Dummy Regressor",
                    metrics={"rmse": 2.0, "mae": 1.0, "r2": 0.0},
                ),
            ),
            (
                _spec(name="Fit Boom", estimator_key="fit_boom", priority=20),
                ControllableModel(name="Fit Boom", fit_error=ValueError("fit failed")),
            ),
            (
                _spec(name="Eval Boom", estimator_key="eval_boom", priority=30),
                ControllableModel(
                    name="Eval Boom",
                    evaluate_error=DataValidationError("eval failed"),
                ),
            ),
            (
                _spec(name="Good", estimator_key="good", priority=40),
                ControllableModel(
                    name="Good",
                    metrics={"rmse": 0.2, "mae": 0.1, "r2": 0.95},
                ),
            ),
        ]
    )
    outcome = _screen_default(registry=registry)
    assert outcome.summary.selected_model_name == "Good"
    assert "Fit Boom" in outcome.summary.failed_model_names
    assert "Eval Boom" in outcome.summary.failed_model_names
    assert "Good" in outcome.summary.successful_model_names
    assert [item.spec.name for item in outcome.summary.candidate_results] == [
        "Dummy Regressor",
        "Fit Boom",
        "Eval Boom",
        "Good",
    ]
    fit_failed = next(
        item for item in outcome.summary.candidate_results if item.spec.name == "Fit Boom"
    )
    assert fit_failed.error_type == "ValueError"
    assert fit_failed.error_message
    assert fit_failed.fit_seconds >= 0.0
    assert math.isfinite(fit_failed.total_seconds)

    registry = _registry_with_models(
        [
            (
                _spec(name="Dummy Regressor", estimator_key="dummy_regressor", priority=10),
                ControllableModel(
                    name="Dummy Regressor",
                    fit_error=ValueError("stop"),
                ),
            ),
        ]
    )
    with pytest.raises(ValueError, match="stop"):
        _screen_default(
            registry=registry,
            policy=ModelScreeningPolicy(continue_on_candidate_failure=False),
        )

    registry = _registry_with_models(
        [
            (
                _spec(name="A", estimator_key="a", priority=10),
                ControllableModel(name="A", fit_error=ValueError("a")),
            ),
            (
                _spec(name="B", estimator_key="b", priority=20),
                ControllableModel(name="B", fit_error=TypeError("b")),
            ),
        ]
    )
    with pytest.raises(ProcessIntelligenceError, match="A|B|ValueError|TypeError"):
        _screen_default(
            registry=registry,
            policy=ModelScreeningPolicy(require_successful_baseline=False),
        )


def test_baseline_detection_and_optional_requirement() -> None:
    registry = _registry_with_models(
        [
            (
                _spec(name="Not Dummy Name", estimator_key="dummy_regressor", priority=10),
                ControllableModel(
                    name="Not Dummy Name",
                    metrics={"rmse": 2.0, "mae": 1.0, "r2": 0.0},
                ),
            ),
            (
                _spec(name="Other", estimator_key="other", priority=20),
                ControllableModel(
                    name="Other",
                    metrics={"rmse": 0.5, "mae": 0.4, "r2": 0.9},
                ),
            ),
        ]
    )
    assert _screen_default(registry=registry).summary.baseline_model_name == "Not Dummy Name"

    registry = _registry_with_models(
        [
            (
                _spec(name="Dummy Fallback", estimator_key="custom_baseline", priority=10),
                ControllableModel(
                    name="Dummy Fallback",
                    metrics={"rmse": 2.0, "mae": 1.0, "r2": 0.0},
                ),
            ),
            (
                _spec(name="Other", estimator_key="other", priority=20),
                ControllableModel(
                    name="Other",
                    metrics={"rmse": 0.5, "mae": 0.4, "r2": 0.9},
                ),
            ),
        ]
    )
    assert _screen_default(registry=registry).summary.baseline_model_name == "Dummy Fallback"

    registry = _registry_with_models(
        [
            (
                _spec(name="Dummy Regressor", estimator_key="dummy_regressor", priority=10),
                ControllableModel(
                    name="Dummy Regressor",
                    fit_error=ValueError("baseline down"),
                ),
            ),
            (
                _spec(name="Other", estimator_key="other", priority=20),
                ControllableModel(
                    name="Other",
                    metrics={"rmse": 0.5, "mae": 0.4, "r2": 0.9},
                ),
            ),
        ]
    )
    with pytest.raises(ProcessIntelligenceError, match="[Bb]aseline"):
        _screen_default(registry=registry)

    registry = _registry_with_models(
        [
            (
                _spec(name="Other", estimator_key="other", priority=10),
                ControllableModel(
                    name="Other",
                    metrics={"rmse": 0.5, "mae": 0.4, "r2": 0.9},
                ),
            ),
        ]
    )
    with pytest.raises(ProcessIntelligenceError, match="[Bb]aseline"):
        _screen_default(registry=registry)

    outcome = _screen_default(
        registry=registry,
        policy=ModelScreeningPolicy(require_successful_baseline=False),
    )
    assert outcome.summary.baseline_model_name is None
    assert outcome.summary.baseline_metrics == {}
    assert outcome.summary.selected_beats_baseline is None
    assert any("baseline" in warning.casefold() for warning in outcome.summary.warnings)


def test_timing_immutability_determinism_and_selected_model_api() -> None:
    registry = create_default_supervised_model_registry(random_state=42)
    outcome = _screen_default(registry=registry)
    for item in outcome.summary.candidate_results:
        assert item.fit_seconds >= 0.0
        assert item.evaluation_seconds >= 0.0
        assert math.isfinite(item.fit_seconds)
        assert math.isfinite(item.evaluation_seconds)
        assert item.total_seconds == pytest.approx(
            item.fit_seconds + item.evaluation_seconds
        )

    dumped = outcome.summary.model_dump()
    assert "selected_model" not in dumped
    assert "estimator" not in dumped
    for item in dumped["candidate_results"]:
        assert "model" not in item
        assert "estimator" not in item

    features = ["f1", "f2"]
    features_before = list(features)
    profile = StubIndustryProfile()
    profile_before = copy.deepcopy(profile._candidates)
    _screen_default(
        registry=registry,
        feature_columns=features,
        industry_profile=profile,
    )
    assert features == features_before
    assert profile._candidates == profile_before
    assert profile.mutated is False

    screener = SupervisedModelScreener(registry)
    first = screener.screen(
        _regression_split(),
        task=AnalysisTask.REGRESSION,
        target_column="y",
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    second = screener.screen(
        _regression_split(),
        task=AnalysisTask.REGRESSION,
        target_column="y",
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    assert len(first.summary.candidate_results) == len(second.summary.candidate_results)
    assert first.summary.selected_model_name == second.summary.selected_model_name

    other = SupervisedModelScreener(
        create_default_supervised_model_registry(random_state=42)
    )
    other_outcome = other.screen(
        _regression_split(),
        task=AnalysisTask.REGRESSION,
        target_column="y",
        feature_columns=["f1", "f2"],
        leakage_report=_safe_report(),
    )
    assert other_outcome.selected_model is not first.selected_model
    assert other_outcome.summary.selected_model_name == first.summary.selected_model_name

    selected_metrics = dict(first.summary.selected_metrics)
    first.summary.selected_metrics["rmse"] = -999.0
    assert selected_metrics["rmse"] != -999.0

    if first.summary.candidate_results[0].metrics:
        key = next(iter(first.summary.candidate_results[0].metrics))
        candidate_metrics = dict(first.summary.candidate_results[0].metrics)
        first.summary.candidate_results[0].metrics[key] = -123.0
        assert candidate_metrics[key] != -123.0

    preds = first.selected_model.predict(
        _regression_split().validation.select(["f1", "f2"])
    )
    assert isinstance(preds, np.ndarray)
    evaluation = first.selected_model.evaluate(
        _regression_split().validation.select(["f1", "f2"]),
        _regression_split().validation["y"],
    )
    assert isinstance(evaluation, ModelEvaluation)
    assert evaluation.metrics
