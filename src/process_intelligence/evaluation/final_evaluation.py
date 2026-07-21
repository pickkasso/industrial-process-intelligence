"""Final supervised model evaluation on the held-out test partition (Step 6D).

Refits the screening-selected ``ModelSpec`` on train+validation (by default),
then evaluates exactly once on the test partition. Test data is never used for
fitting, model selection, or hyperparameter decisions.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Self

import polars as pl
from pydantic import BaseModel, Field, field_validator, model_validator

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.exceptions import (
    DataLeakageError,
    DataValidationError,
    InsufficientDataError,
    ProcessIntelligenceError,
)
from process_intelligence.core.protocols import BaseAnalysisModel
from process_intelligence.core.schemas import ModelEvaluation, ModelMetadata, ModelSpec
from process_intelligence.data.loader import ORIGINAL_ROW_ID_COLUMN
from process_intelligence.evaluation.leakage import LeakageReport, LeakageSeverity
from process_intelligence.evaluation.splitting import DatasetSplit
from process_intelligence.models.registry import ModelRegistry
from process_intelligence.models.screening import (
    CandidateRunStatus,
    CandidateScreeningResult,
    ModelScreeningOutcome,
    ModelScreeningSummary,
)

_SUPPORTED_TASKS = frozenset({AnalysisTask.REGRESSION, AnalysisTask.CLASSIFICATION})
_REGRESSION_REQUIRED_METRICS = ("rmse", "mae")
_CLASSIFICATION_REQUIRED_METRICS = ("f1_macro", "balanced_accuracy", "accuracy")

_WARNING_LEAKAGE = (
    "LeakageReport contains WARNING issues; proceed with caution"
)
_WARNING_NO_REFIT = (
    "Final model was not refit on train+validation; "
    "using the screening-selected fitted model"
)


class FinalEvaluationPolicy(BaseModel):
    """Configurable rules for final supervised model evaluation.

    Controls whether the selected specification is refit on train+validation,
    whether leakage blockers and screening consistency are enforced, and whether
    test-metric degradation relative to validation produces a warning.
    """

    refit_on_train_validation: bool = True
    require_safe_leakage_report: bool = True
    require_screening_consistency: bool = True
    warn_on_test_degradation: bool = True

    @field_validator(
        "refit_on_train_validation",
        "require_safe_leakage_report",
        "require_screening_consistency",
        "warn_on_test_degradation",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError(f"must be a bool, got {type(value).__name__}")
        return value


class FinalEvaluationReport(BaseModel):
    """Structured report of a single final train+validation refit and test evaluation.

    Records selected model identity, partition sizes, validation and test metrics,
    generalization gap, timings, and deterministic warnings. Does not store
    estimators, training frames, or full prediction arrays.
    """

    task: AnalysisTask
    model_name: str
    estimator_key: str
    target_column: str
    feature_columns: list[str]
    refit_on_train_validation: bool
    train_row_count: int
    validation_row_count: int
    test_row_count: int
    final_fit_row_count: int
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    primary_metric_name: str
    higher_is_better: bool
    validation_primary_metric: float
    test_primary_metric: float
    generalization_gap: float
    performance_degraded: bool
    fit_seconds: float
    test_evaluation_seconds: float
    total_seconds: float
    evaluated_at: datetime
    warnings: list[str] = Field(default_factory=list)

    @field_validator(
        "model_name",
        "estimator_key",
        "target_column",
        "primary_metric_name",
        mode="before",
    )
    @classmethod
    def _validate_non_empty_str(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(f"must be str, got {type(value).__name__}")
        if value == "" or value.strip() == "":
            raise ValueError("must be a non-empty, non-whitespace string")
        return value

    @field_validator("feature_columns", mode="after")
    @classmethod
    def _validate_feature_columns(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("feature_columns must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("feature_columns must not contain duplicates")
        if ORIGINAL_ROW_ID_COLUMN in value:
            raise ValueError(
                f"feature_columns must not include {ORIGINAL_ROW_ID_COLUMN!r}"
            )
        return list(value)

    @field_validator(
        "train_row_count",
        "validation_row_count",
        mode="before",
    )
    @classmethod
    def _validate_non_negative_row_count(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"row counts must be int >= 0 (bool not allowed), "
                f"got {type(value).__name__}"
            )
        if value < 0:
            raise ValueError(f"row counts must be >= 0, got {value}")
        return value

    @field_validator("test_row_count", "final_fit_row_count", mode="before")
    @classmethod
    def _validate_positive_row_count(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"row counts must be int >= 1 (bool not allowed), "
                f"got {type(value).__name__}"
            )
        if value < 1:
            raise ValueError(f"row counts must be >= 1, got {value}")
        return value

    @field_validator("validation_metrics", "test_metrics", mode="before")
    @classmethod
    def _validate_metrics_dict(cls, value: object) -> dict[str, float]:
        if not isinstance(value, dict):
            raise ValueError(
                f"metrics must be a dict[str, float], got {type(value).__name__}"
            )
        if not value:
            raise ValueError("metrics must not be empty")
        cleaned: dict[str, float] = {}
        for key, raw in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("metrics keys must be non-empty strings")
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(
                    f"metrics[{key!r}] must be a finite number "
                    f"(bool not allowed), got {type(raw).__name__}"
                )
            number = float(raw)
            if not math.isfinite(number):
                raise ValueError(f"metrics[{key!r}] must be finite, got {raw!r}")
            cleaned[key] = number
        return cleaned

    @field_validator(
        "validation_primary_metric",
        "test_primary_metric",
        "generalization_gap",
        mode="before",
    )
    @classmethod
    def _validate_finite_float(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"must be a finite number (bool not allowed), "
                f"got {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"must be finite, got {value!r}")
        return number

    @field_validator(
        "fit_seconds",
        "test_evaluation_seconds",
        "total_seconds",
        mode="before",
    )
    @classmethod
    def _validate_non_negative_timing(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "timing fields must be finite numbers >= 0 "
                f"(bool not allowed), got {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"timing fields must be finite, got {value!r}")
        if number < 0.0:
            raise ValueError(f"timing fields must be >= 0, got {number}")
        return number

    @field_validator(
        "refit_on_train_validation",
        "higher_is_better",
        "performance_degraded",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError(f"must be a bool, got {type(value).__name__}")
        return value

    @field_validator("evaluated_at", mode="after")
    @classmethod
    def _validate_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("evaluated_at must be timezone-aware")
        return value

    @field_validator("warnings", mode="after")
    @classmethod
    def _reject_duplicate_warnings(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("warnings must not contain duplicates")
        return list(value)

    @model_validator(mode="after")
    def _validate_report_consistency(self) -> Self:
        if self.target_column in self.feature_columns:
            raise ValueError(
                f"target_column {self.target_column!r} must not appear in "
                "feature_columns"
            )
        if self.primary_metric_name not in self.validation_metrics:
            raise ValueError(
                f"primary_metric_name {self.primary_metric_name!r} must exist in "
                "validation_metrics"
            )
        if self.primary_metric_name not in self.test_metrics:
            raise ValueError(
                f"primary_metric_name {self.primary_metric_name!r} must exist in "
                "test_metrics"
            )
        expected_total = self.fit_seconds + self.test_evaluation_seconds
        if self.total_seconds != expected_total:
            raise ValueError(
                "total_seconds must equal fit_seconds + test_evaluation_seconds, "
                f"got total_seconds={self.total_seconds}, expected={expected_total}"
            )
        return self


@dataclass(frozen=True, slots=True)
class FinalEvaluationOutcome:
    """Fitted final model paired with an immutable final-evaluation report."""

    final_model: BaseAnalysisModel
    report: FinalEvaluationReport


class FinalModelEvaluator:
    """Refit a screening-selected model and evaluate once on the test partition.

    Uses the registry only to instantiate a fresh model for optional refit.
    Never mutates the registry, screening outcome, split partitions, or leakage
    report. Test data is used solely for the single final evaluation call.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        policy: FinalEvaluationPolicy | None = None,
    ) -> None:
        """Create an evaluator with a read-only registry and isolated policy.

        Args:
            registry: Model registry used to instantiate the selected specification.
            policy: Optional evaluation policy. When ``None``, defaults are used.

        Raises:
            TypeError: If ``registry`` or ``policy`` has an invalid type.
        """
        if not isinstance(registry, ModelRegistry):
            raise TypeError(
                f"registry must be ModelRegistry, got {type(registry).__name__}"
            )
        if policy is None:
            stored_policy = FinalEvaluationPolicy()
        elif isinstance(policy, FinalEvaluationPolicy):
            stored_policy = policy.model_copy(deep=True)
        else:
            raise TypeError(
                "policy must be FinalEvaluationPolicy or None, "
                f"got {type(policy).__name__}"
            )
        self._registry = registry
        self._policy = stored_policy

    def evaluate(
        self,
        split: DatasetSplit,
        screening_outcome: ModelScreeningOutcome,
        *,
        task: AnalysisTask,
        target_column: str,
        feature_columns: Sequence[str],
        leakage_report: LeakageReport,
    ) -> FinalEvaluationOutcome:
        """Refit the selected model and evaluate exactly once on test.

        Args:
            split: Train/validation/test partitions. Test is used only for the
                final evaluation call.
            screening_outcome: Screening result providing the selected specification
                and validation metrics.
            task: ``REGRESSION`` or ``CLASSIFICATION``.
            target_column: Target column name present in all partitions.
            feature_columns: Feature column names in evaluation order.
            leakage_report: Structural leakage report for the feature set.

        Returns:
            Outcome containing the fitted final model and evaluation report.

        Raises:
            TypeError: If inputs have invalid types.
            DataValidationError: If inputs fail structural or consistency checks.
            DataLeakageError: If blockers are present and the policy requires a
                safe leakage report.
            InsufficientDataError: If any partition is empty.
            ProcessIntelligenceError: If evaluation results or model metadata are
                inconsistent with the final-evaluation contract.
        """
        validated_task = _validate_task(task)
        validated_target = _validate_target_column(target_column)
        validated_features = _validate_feature_columns(
            feature_columns,
            target_column=validated_target,
        )
        validated_split = _validate_split(
            split,
            target_column=validated_target,
            feature_columns=validated_features,
        )
        validated_outcome = _validate_screening_outcome(screening_outcome)
        _validate_leakage_report(
            leakage_report,
            feature_columns=validated_features,
            require_safe=self._policy.require_safe_leakage_report,
        )

        train = validated_split.train
        validation = validated_split.validation
        test = validated_split.test
        if train.height < 1:
            raise InsufficientDataError(
                "train partition must contain at least one row for final evaluation"
            )
        if validation.height < 1:
            raise InsufficientDataError(
                "validation partition must contain at least one row for final evaluation"
            )
        if test.height < 1:
            raise InsufficientDataError(
                "test partition must contain at least one row for final evaluation"
            )

        if self._policy.require_screening_consistency:
            _validate_screening_consistency(
                summary=validated_outcome.summary,
                task=validated_task,
                target_column=validated_target,
                feature_columns=validated_features,
                split=validated_split,
            )

        selected_candidate = _find_selected_candidate(validated_outcome.summary)
        selected_spec = selected_candidate.spec
        validation_metrics = _copy_and_validate_metrics(
            validated_outcome.summary.selected_metrics,
            task=validated_task,
            source="validation",
        )

        fit_seconds = 0.0
        if self._policy.refit_on_train_validation:
            final_model = self._registry.instantiate(selected_spec)
            fit_frame = pl.concat([train, validation], how="vertical")
            x_fit = fit_frame.select(validated_features)
            y_fit = fit_frame[validated_target]
            fit_started = time.perf_counter()
            final_model.fit(x_fit, y_fit)
            fit_seconds = time.perf_counter() - fit_started
            final_fit_row_count = int(train.height + validation.height)
        else:
            final_model = validated_outcome.selected_model
            if hasattr(final_model, "is_fitted") and not bool(final_model.is_fitted):
                raise ProcessIntelligenceError(
                    "screening selected_model must be fitted when "
                    "refit_on_train_validation is False"
                )
            final_fit_row_count = int(validated_outcome.summary.train_row_count)

        x_test = test.select(validated_features)
        y_test = test[validated_target]
        eval_started = time.perf_counter()
        evaluation = final_model.evaluate(x_test, y_test)
        test_evaluation_seconds = time.perf_counter() - eval_started

        test_metrics = _extract_and_validate_metrics(
            evaluation,
            task=validated_task,
        )
        primary_metric_name, higher_is_better = _primary_config(validated_task)
        validation_primary = validation_metrics[primary_metric_name]
        test_primary = test_metrics[primary_metric_name]
        generalization_gap = _generalization_gap(
            task=validated_task,
            validation_primary=validation_primary,
            test_primary=test_primary,
        )
        performance_degraded = generalization_gap > 0.0

        warnings = _build_warnings(
            leakage_report=leakage_report,
            refit_on_train_validation=self._policy.refit_on_train_validation,
            warn_on_test_degradation=self._policy.warn_on_test_degradation,
            performance_degraded=performance_degraded,
            primary_metric_name=primary_metric_name,
            validation_primary=validation_primary,
            test_primary=test_primary,
            screening_warnings=validated_outcome.summary.warnings,
            selected_candidate=selected_candidate,
            evaluation=evaluation,
        )

        report = FinalEvaluationReport(
            task=validated_task,
            model_name=selected_spec.name,
            estimator_key=selected_spec.estimator_key,
            target_column=validated_target,
            feature_columns=list(validated_features),
            refit_on_train_validation=self._policy.refit_on_train_validation,
            train_row_count=int(train.height),
            validation_row_count=int(validation.height),
            test_row_count=int(test.height),
            final_fit_row_count=final_fit_row_count,
            validation_metrics=dict(validation_metrics),
            test_metrics=dict(test_metrics),
            primary_metric_name=primary_metric_name,
            higher_is_better=higher_is_better,
            validation_primary_metric=float(validation_primary),
            test_primary_metric=float(test_primary),
            generalization_gap=float(generalization_gap),
            performance_degraded=performance_degraded,
            fit_seconds=float(fit_seconds),
            test_evaluation_seconds=float(test_evaluation_seconds),
            total_seconds=float(fit_seconds + test_evaluation_seconds),
            evaluated_at=datetime.now(UTC),
            warnings=warnings,
        )
        _assert_final_model_consistency(
            final_model=final_model,
            report=report,
            selected_spec=selected_spec,
        )
        return FinalEvaluationOutcome(final_model=final_model, report=report)


def _primary_config(task: AnalysisTask) -> tuple[str, bool]:
    """Return primary metric name and direction for ``task``."""
    if task is AnalysisTask.REGRESSION:
        return "rmse", False
    return "f1_macro", True


def _generalization_gap(
    *,
    task: AnalysisTask,
    validation_primary: float,
    test_primary: float,
) -> float:
    """Return gap where larger values mean worse test performance vs validation."""
    if task is AnalysisTask.REGRESSION:
        return float(test_primary - validation_primary)
    return float(validation_primary - test_primary)


def _validate_task(task: object) -> AnalysisTask:
    """Validate that ``task`` is a supported supervised ``AnalysisTask``."""
    if not isinstance(task, AnalysisTask):
        raise TypeError(f"task must be AnalysisTask, got {type(task).__name__}")
    if task not in _SUPPORTED_TASKS:
        raise DataValidationError(
            "FinalModelEvaluator supports REGRESSION and CLASSIFICATION only, "
            f"got {task!r}"
        )
    return task


def _validate_target_column(target_column: object) -> str:
    """Validate a non-empty target column name string."""
    if not isinstance(target_column, str):
        raise TypeError(
            f"target_column must be str, got {type(target_column).__name__}"
        )
    if target_column == "" or target_column.strip() == "":
        raise DataValidationError(
            "target_column must be a non-empty, non-whitespace string"
        )
    if target_column == ORIGINAL_ROW_ID_COLUMN:
        raise DataValidationError(
            f"target_column must not be reserved column {ORIGINAL_ROW_ID_COLUMN!r}"
        )
    return target_column


def _validate_feature_columns(
    feature_columns: object,
    *,
    target_column: str,
) -> list[str]:
    """Validate feature column names while preserving input order."""
    if isinstance(feature_columns, (str, bytes)):
        raise TypeError(
            "feature_columns must be a Sequence[str], not "
            f"{type(feature_columns).__name__}"
        )
    if not isinstance(feature_columns, Sequence):
        raise TypeError(
            "feature_columns must be a Sequence[str], "
            f"got {type(feature_columns).__name__}"
        )

    validated: list[str] = []
    seen: set[str] = set()
    for index, name in enumerate(feature_columns):
        if not isinstance(name, str):
            raise TypeError(
                f"feature_columns[{index}] must be str, got {type(name).__name__}"
            )
        if name == "" or name.strip() == "":
            raise DataValidationError(
                f"feature_columns[{index}] must be a non-empty, non-whitespace string"
            )
        if name == ORIGINAL_ROW_ID_COLUMN:
            raise DataValidationError(
                "feature_columns must not include reserved column "
                f"{ORIGINAL_ROW_ID_COLUMN!r}"
            )
        if name == target_column:
            raise DataValidationError(
                f"target_column {target_column!r} must not appear in feature_columns"
            )
        if name in seen:
            raise DataValidationError(
                f"feature_columns must not contain duplicates, found {name!r}"
            )
        seen.add(name)
        validated.append(name)

    if not validated:
        raise DataValidationError("feature_columns must contain at least one feature")
    return validated


def _validate_split(
    split: object,
    *,
    target_column: str,
    feature_columns: Sequence[str],
) -> DatasetSplit:
    """Validate split type, partition frames, schema, and required columns."""
    if not isinstance(split, DatasetSplit):
        raise TypeError(f"split must be DatasetSplit, got {type(split).__name__}")

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
    if list(split.validation.columns) != train_columns:
        raise DataValidationError(
            "train and validation partitions must share identical column names "
            "and order"
        )
    if list(split.test.columns) != train_columns:
        raise DataValidationError(
            "train and test partitions must share identical column names and order"
        )

    if ORIGINAL_ROW_ID_COLUMN not in train_columns:
        raise DataValidationError(
            f"all partitions require reserved column {ORIGINAL_ROW_ID_COLUMN!r}"
        )
    if target_column not in train_columns:
        raise DataValidationError(
            f"target_column {target_column!r} is missing from split partitions"
        )
    missing_features = [name for name in feature_columns if name not in train_columns]
    if missing_features:
        raise DataValidationError(
            "feature_columns missing from split partitions: "
            f"{', '.join(missing_features)}"
        )
    return split


def _validate_screening_outcome(screening_outcome: object) -> ModelScreeningOutcome:
    """Validate screening outcome type and nested selected model / summary."""
    if not isinstance(screening_outcome, ModelScreeningOutcome):
        raise TypeError(
            "screening_outcome must be ModelScreeningOutcome, "
            f"got {type(screening_outcome).__name__}"
        )
    if not isinstance(screening_outcome.selected_model, BaseAnalysisModel):
        raise TypeError(
            "screening_outcome.selected_model must be BaseAnalysisModel, "
            f"got {type(screening_outcome.selected_model).__name__}"
        )
    if not isinstance(screening_outcome.summary, ModelScreeningSummary):
        raise TypeError(
            "screening_outcome.summary must be ModelScreeningSummary, "
            f"got {type(screening_outcome.summary).__name__}"
        )
    return screening_outcome


def _validate_leakage_report(
    leakage_report: object,
    *,
    feature_columns: Sequence[str],
    require_safe: bool,
) -> None:
    """Validate leakage report type, feature alignment, and optional safety gate."""
    if not isinstance(leakage_report, LeakageReport):
        raise TypeError(
            f"leakage_report must be LeakageReport, got {type(leakage_report).__name__}"
        )
    checked = list(leakage_report.checked_feature_columns)
    expected = list(feature_columns)
    if checked != expected:
        raise DataValidationError(
            "leakage_report.checked_feature_columns must exactly match "
            f"feature_columns in order; expected {expected!r}, got {checked!r}"
        )
    if require_safe and leakage_report.blocker_count >= 1:
        blocker_types: list[str] = []
        seen: set[str] = set()
        for issue in leakage_report.issues:
            if issue.severity is not LeakageSeverity.BLOCKER:
                continue
            type_name = str(issue.issue_type)
            if type_name not in seen:
                seen.add(type_name)
                blocker_types.append(type_name)
        raise DataLeakageError(
            "Data leakage blockers detected during final evaluation: "
            f"count={leakage_report.blocker_count}, "
            f"types={', '.join(blocker_types)}"
        )


def _validate_screening_consistency(
    *,
    summary: ModelScreeningSummary,
    task: AnalysisTask,
    target_column: str,
    feature_columns: Sequence[str],
    split: DatasetSplit,
) -> None:
    """Require screening summary to match the current evaluation inputs."""
    mismatches: list[str] = []
    if summary.task != task:
        mismatches.append(f"task (summary={summary.task!r}, expected={task!r})")
    if summary.target_column != target_column:
        mismatches.append(
            "target_column "
            f"(summary={summary.target_column!r}, expected={target_column!r})"
        )
    if list(summary.feature_columns) != list(feature_columns):
        mismatches.append(
            "feature_columns "
            f"(summary={list(summary.feature_columns)!r}, "
            f"expected={list(feature_columns)!r})"
        )
    if summary.train_row_count != split.train.height:
        mismatches.append(
            "train_row_count "
            f"(summary={summary.train_row_count}, expected={split.train.height})"
        )
    if summary.validation_row_count != split.validation.height:
        mismatches.append(
            "validation_row_count "
            f"(summary={summary.validation_row_count}, "
            f"expected={split.validation.height})"
        )
    if summary.test_row_count != split.test.height:
        mismatches.append(
            "test_row_count "
            f"(summary={summary.test_row_count}, expected={split.test.height})"
        )
    if summary.selected_model_name == "" or summary.selected_model_name.strip() == "":
        mismatches.append("selected_model_name is missing or blank")
    if (
        summary.selected_estimator_key == ""
        or summary.selected_estimator_key.strip() == ""
    ):
        mismatches.append("selected_estimator_key is missing or blank")
    if not summary.selected_metrics:
        mismatches.append("selected_metrics is missing or empty")
    elif summary.ranking_metric not in summary.selected_metrics:
        mismatches.append(
            f"ranking_metric {summary.ranking_metric!r} missing from selected_metrics"
        )
    if mismatches:
        raise DataValidationError(
            "screening_outcome is inconsistent with final evaluation inputs: "
            + "; ".join(mismatches)
        )


def _find_selected_candidate(
    summary: ModelScreeningSummary,
) -> CandidateScreeningResult:
    """Locate the unique SUCCESS candidate matching the screening selection."""
    matches = [
        result
        for result in summary.candidate_results
        if (
            result.status is CandidateRunStatus.SUCCESS
            and result.registry_rank == summary.selected_registry_rank
            and result.spec.name == summary.selected_model_name
            and result.spec.estimator_key == summary.selected_estimator_key
        )
    ]
    if len(matches) == 0:
        raise DataValidationError(
            "No SUCCESS candidate matches the screening selection: "
            f"name={summary.selected_model_name!r}, "
            f"estimator_key={summary.selected_estimator_key!r}, "
            f"registry_rank={summary.selected_registry_rank}"
        )
    if len(matches) > 1:
        raise DataValidationError(
            "Multiple SUCCESS candidates match the screening selection: "
            f"name={summary.selected_model_name!r}, "
            f"estimator_key={summary.selected_estimator_key!r}, "
            f"registry_rank={summary.selected_registry_rank}, "
            f"match_count={len(matches)}"
        )
    return matches[0]


def _copy_and_validate_metrics(
    metrics: dict[str, float],
    *,
    task: AnalysisTask,
    source: str,
) -> dict[str, float]:
    """Copy and validate task-specific metric maps without mutating the input."""
    if not isinstance(metrics, dict):
        raise ProcessIntelligenceError(
            f"{source} metrics must be a dict[str, float], "
            f"got {type(metrics).__name__}"
        )
    cleaned: dict[str, float] = {}
    for key, raw in metrics.items():
        if not isinstance(key, str) or not key:
            raise ProcessIntelligenceError(
                f"{source} metric keys must be non-empty strings"
            )
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ProcessIntelligenceError(
                f"{source} metric {key!r} must be a finite number "
                f"(bool not allowed), got {type(raw).__name__}"
            )
        number = float(raw)
        if not math.isfinite(number):
            raise ProcessIntelligenceError(
                f"{source} metric {key!r} must be finite, got {raw!r}"
            )
        cleaned[key] = number
    _validate_task_metric_ranges(cleaned, task=task, source=source)
    return cleaned


def _extract_and_validate_metrics(
    evaluation: object,
    *,
    task: AnalysisTask,
) -> dict[str, float]:
    """Extract and validate finite metrics from a model evaluation payload."""
    if not isinstance(evaluation, ModelEvaluation):
        raise ProcessIntelligenceError(
            "model.evaluate must return ModelEvaluation, "
            f"got {type(evaluation).__name__}"
        )
    if not isinstance(evaluation.metrics, dict):
        raise ProcessIntelligenceError(
            "ModelEvaluation.metrics must be a dict[str, float], "
            f"got {type(evaluation.metrics).__name__}"
        )

    cleaned: dict[str, float] = {}
    for key, raw in evaluation.metrics.items():
        if not isinstance(key, str) or not key:
            raise ProcessIntelligenceError("metric keys must be non-empty strings")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ProcessIntelligenceError(
                f"metric {key!r} must be a finite number "
                f"(bool not allowed), got {type(raw).__name__}"
            )
        number = float(raw)
        if not math.isfinite(number):
            raise ProcessIntelligenceError(
                f"metric {key!r} must be finite, got {raw!r}"
            )
        cleaned[key] = number

    _validate_task_metric_ranges(cleaned, task=task, source="test")
    return cleaned


def _validate_task_metric_ranges(
    metrics: dict[str, float],
    *,
    task: AnalysisTask,
    source: str,
) -> None:
    """Validate required metrics and value ranges for the analysis task."""
    if task is AnalysisTask.REGRESSION:
        for name in _REGRESSION_REQUIRED_METRICS:
            if name not in metrics:
                raise ProcessIntelligenceError(
                    f"{source} regression evaluation missing required metric {name!r}"
                )
        if metrics["rmse"] < 0.0:
            raise ProcessIntelligenceError(
                f"{source} rmse must be >= 0, got {metrics['rmse']}"
            )
        if metrics["mae"] < 0.0:
            raise ProcessIntelligenceError(
                f"{source} mae must be >= 0, got {metrics['mae']}"
            )
        if "r2" in metrics and not math.isfinite(metrics["r2"]):
            raise ProcessIntelligenceError(
                f"{source} r2 must be finite when present, got {metrics['r2']!r}"
            )
        return

    for name in _CLASSIFICATION_REQUIRED_METRICS:
        if name not in metrics:
            raise ProcessIntelligenceError(
                f"{source} classification evaluation missing required metric {name!r}"
            )
    for name in _CLASSIFICATION_REQUIRED_METRICS:
        value = metrics[name]
        if value < 0.0 or value > 1.0:
            raise ProcessIntelligenceError(
                f"{source} {name} must be within [0, 1], got {value}"
            )


def _build_warnings(
    *,
    leakage_report: LeakageReport,
    refit_on_train_validation: bool,
    warn_on_test_degradation: bool,
    performance_degraded: bool,
    primary_metric_name: str,
    validation_primary: float,
    test_primary: float,
    screening_warnings: Sequence[str],
    selected_candidate: CandidateScreeningResult,
    evaluation: ModelEvaluation,
) -> list[str]:
    """Build deterministic, de-duplicated final-evaluation warnings."""
    warnings: list[str] = []

    if any(
        issue.severity is LeakageSeverity.WARNING for issue in leakage_report.issues
    ):
        warnings.append(_WARNING_LEAKAGE)

    if not refit_on_train_validation:
        warnings.append(_WARNING_NO_REFIT)

    if warn_on_test_degradation and performance_degraded:
        warnings.append(
            "Test primary metric degraded relative to validation: "
            f"{primary_metric_name} validation={validation_primary}, "
            f"test={test_primary}"
        )

    for warning in screening_warnings:
        if warning and warning not in warnings:
            warnings.append(warning)
    for warning in selected_candidate.warnings:
        if warning and warning not in warnings:
            warnings.append(warning)
    for note in evaluation.notes:
        if note and note not in warnings:
            warnings.append(note)

    return warnings


def _assert_final_model_consistency(
    *,
    final_model: BaseAnalysisModel,
    report: FinalEvaluationReport,
    selected_spec: ModelSpec,
) -> None:
    """Verify the fitted final model matches the evaluation report identity."""
    if not isinstance(final_model, BaseAnalysisModel):
        raise ProcessIntelligenceError(
            "final_model must be BaseAnalysisModel, "
            f"got {type(final_model).__name__}"
        )
    if hasattr(final_model, "is_fitted") and not bool(final_model.is_fitted):
        raise ProcessIntelligenceError("final_model must be in a fitted state")

    metadata = final_model.get_metadata()
    if not isinstance(metadata, ModelMetadata):
        raise ProcessIntelligenceError(
            "final_model.get_metadata must return ModelMetadata, "
            f"got {type(metadata).__name__}"
        )
    if metadata.model_name != report.model_name:
        raise ProcessIntelligenceError(
            "final model metadata model_name does not match report: "
            f"metadata={metadata.model_name!r}, report={report.model_name!r}"
        )
    if metadata.task != report.task:
        raise ProcessIntelligenceError(
            "final model metadata task does not match report: "
            f"metadata={metadata.task!r}, report={report.task!r}"
        )
    if selected_spec.estimator_key != report.estimator_key:
        raise ProcessIntelligenceError(
            "selected ModelSpec estimator_key does not match report: "
            f"spec={selected_spec.estimator_key!r}, report={report.estimator_key!r}"
        )
    model_spec = getattr(final_model, "_spec", None)
    if isinstance(model_spec, ModelSpec):
        if model_spec.estimator_key != report.estimator_key:
            raise ProcessIntelligenceError(
                "final model spec estimator_key does not match report: "
                f"model={model_spec.estimator_key!r}, report={report.estimator_key!r}"
            )
        if model_spec.name != report.model_name:
            raise ProcessIntelligenceError(
                "final model spec name does not match report: "
                f"model={model_spec.name!r}, report={report.model_name!r}"
            )
