"""Supervised model screening on train/validation partitions (Step 6C).

Candidates from ``ModelRegistry`` are fit on the train partition and ranked on
the validation partition only. The test partition is never used for fitting,
prediction, evaluation, or model selection.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import cmp_to_key
from typing import Self

import numpy as np
import polars as pl
from pydantic import BaseModel, Field, field_validator, model_validator

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.exceptions import (
    DataLeakageError,
    DataValidationError,
    InsufficientDataError,
    ProcessIntelligenceError,
)
from process_intelligence.core.protocols import BaseAnalysisModel, BaseIndustryProfile
from process_intelligence.core.schemas import ModelEvaluation, ModelSpec
from process_intelligence.data.loader import ORIGINAL_ROW_ID_COLUMN
from process_intelligence.evaluation.leakage import LeakageReport, LeakageSeverity
from process_intelligence.evaluation.splitting import DatasetSplit
from process_intelligence.models.registry import ModelRegistry

_SUPPORTED_TASKS = frozenset({AnalysisTask.REGRESSION, AnalysisTask.CLASSIFICATION})
_REGRESSION_REQUIRED_METRICS = ("rmse", "mae")
_CLASSIFICATION_REQUIRED_METRICS = ("f1_macro", "balanced_accuracy", "accuracy")
_CANDIDATE_FAILURE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ProcessIntelligenceError,
    ValueError,
    TypeError,
    ArithmeticError,
    np.linalg.LinAlgError,
)

_WARNING_LEAKAGE = (
    "LeakageReport contains WARNING issues; proceed with caution"
)
_WARNING_FAILED_CANDIDATES = (
    "One or more screening candidates failed during fit or evaluation"
)
_WARNING_NO_BASELINE = (
    "No successful Dummy baseline candidate was available for comparison"
)
_WARNING_DID_NOT_BEAT_BASELINE = (
    "Selected model did not outperform the Dummy baseline on the ranking metric"
)


class CandidateRunStatus(StrEnum):
    """Execution outcome for a single screening candidate."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class ModelScreeningPolicy(BaseModel):
    """Configurable rules for supervised model screening.

    Controls candidate limits, leakage/validation gates, failure isolation,
    and whether a successful Dummy baseline is required.
    """

    maximum_candidates: int | None = None
    candidate_time_budget_seconds: float | None = None
    require_safe_leakage_report: bool = True
    require_validation_partition: bool = True
    continue_on_candidate_failure: bool = True
    require_successful_baseline: bool = True

    @field_validator("maximum_candidates", mode="before")
    @classmethod
    def _validate_maximum_candidates(cls, value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "maximum_candidates must be None or an int >= 1 "
                f"(bool not allowed), got {type(value).__name__}"
            )
        if value < 1:
            raise ValueError(f"maximum_candidates must be >= 1, got {value}")
        return value

    @field_validator("candidate_time_budget_seconds", mode="before")
    @classmethod
    def _validate_time_budget(cls, value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "candidate_time_budget_seconds must be None or a finite number > 0 "
                f"(bool not allowed), got {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(
                "candidate_time_budget_seconds must be a finite number > 0, "
                f"got {value!r}"
            )
        if number <= 0.0:
            raise ValueError(
                f"candidate_time_budget_seconds must be > 0, got {number}"
            )
        return number

    @field_validator(
        "require_safe_leakage_report",
        "require_validation_partition",
        "continue_on_candidate_failure",
        "require_successful_baseline",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError(f"must be a bool, got {type(value).__name__}")
        return value


class CandidateScreeningResult(BaseModel):
    """Per-candidate screening outcome with metrics or failure details."""

    spec: ModelSpec
    status: CandidateRunStatus
    registry_rank: int
    metrics: dict[str, float] = Field(default_factory=dict)
    primary_metric_name: str | None = None
    primary_metric_value: float | None = None
    fit_seconds: float
    evaluation_seconds: float
    total_seconds: float
    warnings: list[str] = Field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None

    @field_validator("registry_rank", mode="before")
    @classmethod
    def _validate_registry_rank(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "registry_rank must be an int >= 0 (bool not allowed), "
                f"got {type(value).__name__}"
            )
        if value < 0:
            raise ValueError(f"registry_rank must be >= 0, got {value}")
        return value

    @field_validator(
        "fit_seconds",
        "evaluation_seconds",
        "total_seconds",
        mode="before",
    )
    @classmethod
    def _validate_non_negative_finite(cls, value: object) -> float:
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

    @field_validator("metrics", mode="before")
    @classmethod
    def _validate_metrics_dict(cls, value: object) -> dict[str, float]:
        if not isinstance(value, dict):
            raise ValueError(
                f"metrics must be a dict[str, float], got {type(value).__name__}"
            )
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

    @field_validator("primary_metric_value", mode="before")
    @classmethod
    def _validate_primary_metric_value(cls, value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "primary_metric_value must be None or a finite number "
                f"(bool not allowed), got {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"primary_metric_value must be finite, got {value!r}")
        return number

    @field_validator("warnings", mode="after")
    @classmethod
    def _reject_duplicate_warnings(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("warnings must not contain duplicates")
        return list(value)

    @model_validator(mode="after")
    def _validate_status_consistency(self) -> Self:
        if self.status is CandidateRunStatus.SUCCESS:
            if not self.metrics:
                raise ValueError("SUCCESS results require non-empty metrics")
            if self.primary_metric_name is None or not str(
                self.primary_metric_name
            ).strip():
                raise ValueError("SUCCESS results require primary_metric_name")
            if self.primary_metric_value is None:
                raise ValueError("SUCCESS results require primary_metric_value")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError(
                    "SUCCESS results must not include error_type or error_message"
                )
        elif self.status is CandidateRunStatus.FAILED:
            if self.metrics:
                raise ValueError("FAILED results require empty metrics")
            if self.primary_metric_name is not None:
                raise ValueError("FAILED results require primary_metric_name=None")
            if self.primary_metric_value is not None:
                raise ValueError("FAILED results require primary_metric_value=None")
            if self.error_type is None or not str(self.error_type).strip():
                raise ValueError("FAILED results require a non-empty error_type")
            if self.error_message is None or not str(self.error_message).strip():
                raise ValueError("FAILED results require a non-empty error_message")
        return self


class ModelScreeningSummary(BaseModel):
    """Aggregate screening summary including ranking and baseline comparison."""

    task: AnalysisTask
    target_column: str
    feature_columns: list[str] = Field(default_factory=list)
    candidate_results: list[CandidateScreeningResult] = Field(default_factory=list)
    selected_model_name: str
    selected_estimator_key: str
    selected_metrics: dict[str, float] = Field(default_factory=dict)
    selected_registry_rank: int
    ranking_metric: str
    higher_is_better: bool
    baseline_model_name: str | None = None
    baseline_metrics: dict[str, float] = Field(default_factory=dict)
    selected_beats_baseline: bool | None = None
    successful_model_names: list[str] = Field(default_factory=list)
    failed_model_names: list[str] = Field(default_factory=list)
    train_row_count: int
    validation_row_count: int
    test_row_count: int
    warnings: list[str] = Field(default_factory=list)

    @field_validator("target_column", mode="before")
    @classmethod
    def _validate_target_column(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(
                f"target_column must be str, got {type(value).__name__}"
            )
        if value == "" or value.strip() == "":
            raise ValueError("target_column must be a non-empty, non-whitespace string")
        return value

    @field_validator("feature_columns", mode="after")
    @classmethod
    def _validate_feature_columns(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("feature_columns must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("feature_columns must not contain duplicates")
        return list(value)

    @field_validator("selected_model_name", "selected_estimator_key", mode="before")
    @classmethod
    def _validate_selected_identity(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(f"must be str, got {type(value).__name__}")
        if value == "" or value.strip() == "":
            raise ValueError("must be a non-empty, non-whitespace string")
        return value

    @field_validator("selected_metrics", "baseline_metrics", mode="before")
    @classmethod
    def _validate_metric_maps(cls, value: object) -> dict[str, float]:
        if not isinstance(value, dict):
            raise ValueError(
                f"metric maps must be dict[str, float], got {type(value).__name__}"
            )
        cleaned: dict[str, float] = {}
        for key, raw in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("metric keys must be non-empty strings")
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(
                    f"metric {key!r} must be a finite number "
                    f"(bool not allowed), got {type(raw).__name__}"
                )
            number = float(raw)
            if not math.isfinite(number):
                raise ValueError(f"metric {key!r} must be finite, got {raw!r}")
            cleaned[key] = number
        return cleaned

    @field_validator("selected_registry_rank", mode="before")
    @classmethod
    def _validate_selected_rank(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "selected_registry_rank must be an int >= 0 "
                f"(bool not allowed), got {type(value).__name__}"
            )
        if value < 0:
            raise ValueError(f"selected_registry_rank must be >= 0, got {value}")
        return value

    @field_validator("higher_is_better", mode="before")
    @classmethod
    def _validate_higher_is_better(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError(f"higher_is_better must be bool, got {type(value).__name__}")
        return value

    @field_validator(
        "train_row_count",
        "validation_row_count",
        "test_row_count",
        mode="before",
    )
    @classmethod
    def _validate_row_counts(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "row counts must be int >= 0 (bool not allowed), "
                f"got {type(value).__name__}"
            )
        if value < 0:
            raise ValueError(f"row counts must be >= 0, got {value}")
        return value

    @field_validator(
        "successful_model_names",
        "failed_model_names",
        "warnings",
        mode="after",
    )
    @classmethod
    def _reject_duplicate_names(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("list must not contain duplicates")
        return list(value)

    @model_validator(mode="after")
    def _validate_summary_consistency(self) -> Self:
        if not self.selected_metrics:
            raise ValueError("selected_metrics must not be empty")
        if self.ranking_metric not in self.selected_metrics:
            raise ValueError(
                f"ranking_metric {self.ranking_metric!r} must exist in selected_metrics"
            )
        if self.validation_row_count < 1:
            raise ValueError(
                "validation_row_count must be >= 1 for a successful screening summary"
            )

        success_results = [
            result
            for result in self.candidate_results
            if result.status is CandidateRunStatus.SUCCESS
        ]
        if not success_results:
            raise ValueError("candidate_results must include at least one SUCCESS")

        success_names = {result.spec.name for result in success_results}
        if self.selected_model_name not in success_names:
            raise ValueError(
                f"selected_model_name {self.selected_model_name!r} "
                "must be a SUCCESS candidate"
            )

        selected_matches = [
            result
            for result in success_results
            if result.spec.name == self.selected_model_name
        ]
        if selected_matches[0].registry_rank != self.selected_registry_rank:
            raise ValueError(
                "selected_registry_rank must match the selected candidate registry_rank"
            )

        if self.baseline_model_name is None:
            if self.baseline_metrics:
                raise ValueError(
                    "baseline_metrics must be empty when baseline_model_name is None"
                )
        elif self.baseline_model_name not in success_names:
            raise ValueError(
                f"baseline_model_name {self.baseline_model_name!r} "
                "must be a SUCCESS candidate"
            )

        overlap = set(self.successful_model_names) & set(self.failed_model_names)
        if overlap:
            raise ValueError(
                "successful_model_names and failed_model_names must be disjoint, "
                f"overlap={sorted(overlap)}"
            )
        return self


@dataclass(frozen=True, slots=True)
class ModelScreeningOutcome:
    """Selected fitted model paired with an immutable screening summary."""

    selected_model: BaseAnalysisModel
    summary: ModelScreeningSummary


class SupervisedModelScreener:
    """Screen supervised registry candidates on train/validation partitions.

    Fits each available candidate on the train partition, evaluates on the
    validation partition, and selects a winner by deterministic ranking rules.
    The test partition is never used for model selection.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        policy: ModelScreeningPolicy | None = None,
    ) -> None:
        """Create a screener with a read-only registry and isolated policy.

        Args:
            registry: Model registry used to list and instantiate candidates.
            policy: Optional screening policy. When ``None``, defaults are used.

        Raises:
            TypeError: If ``registry`` or ``policy`` has an invalid type.
        """
        if not isinstance(registry, ModelRegistry):
            raise TypeError(
                f"registry must be ModelRegistry, got {type(registry).__name__}"
            )
        if policy is None:
            stored_policy = ModelScreeningPolicy()
        elif isinstance(policy, ModelScreeningPolicy):
            stored_policy = policy.model_copy(deep=True)
        else:
            raise TypeError(
                "policy must be ModelScreeningPolicy or None, "
                f"got {type(policy).__name__}"
            )
        self._registry = registry
        self._policy = stored_policy

    def screen(
        self,
        split: DatasetSplit,
        *,
        task: AnalysisTask,
        target_column: str,
        feature_columns: Sequence[str],
        leakage_report: LeakageReport,
        industry_profile: BaseIndustryProfile | None = None,
    ) -> ModelScreeningOutcome:
        """Fit and rank supervised candidates without using the test partition.

        Args:
            split: Train/validation/test partitions. Only train and validation
                are used for fitting and ranking.
            task: ``REGRESSION`` or ``CLASSIFICATION``.
            target_column: Target column name present in all partitions.
            feature_columns: Feature column names in evaluation order.
            leakage_report: Structural leakage report for the feature set.
            industry_profile: Optional industry profile forwarded to the registry.

        Returns:
            Outcome containing the selected fitted model and screening summary.

        Raises:
            TypeError: If inputs have invalid types.
            DataValidationError: If inputs fail structural validation.
            DataLeakageError: If blockers are present and the policy requires a
                safe leakage report.
            InsufficientDataError: If train or validation partitions are empty.
            ProcessIntelligenceError: If no candidates are available, all
                candidates fail, or a required baseline is missing.
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
        _validate_leakage_report(
            leakage_report,
            feature_columns=validated_features,
            require_safe=self._policy.require_safe_leakage_report,
        )
        if industry_profile is not None and not isinstance(
            industry_profile, BaseIndustryProfile
        ):
            raise TypeError(
                "industry_profile must be BaseIndustryProfile or None, "
                f"got {type(industry_profile).__name__}"
            )

        train = validated_split.train
        validation = validated_split.validation
        if train.height == 0:
            raise InsufficientDataError(
                "train partition must contain at least one row for screening"
            )
        if validation.height == 0:
            raise InsufficientDataError(
                "validation partition must contain at least one row for screening"
            )

        x_train = train.select(validated_features)
        y_train = train[validated_target]
        x_validation = validation.select(validated_features)
        y_validation = validation[validated_target]

        candidates = self._registry.get_candidates(
            validated_task,
            industry_profile=industry_profile,
            time_budget_seconds=self._policy.candidate_time_budget_seconds,
        )
        if not candidates:
            raise ProcessIntelligenceError(
                f"No available screening candidates for task={validated_task!r}"
            )

        ranking_metric, higher_is_better = _ranking_config(validated_task)
        candidate_results: list[CandidateScreeningResult] = []
        successful_models: dict[str, BaseAnalysisModel] = {}
        successful_names: list[str] = []
        failed_names: list[str] = []
        max_candidates = self._policy.maximum_candidates

        for rank, spec in enumerate(candidates):
            if max_candidates is not None and rank >= max_candidates:
                break

            result, fitted = self._run_candidate(
                spec=spec,
                registry_rank=rank,
                task=validated_task,
                ranking_metric=ranking_metric,
                x_train=x_train,
                y_train=y_train,
                x_validation=x_validation,
                y_validation=y_validation,
            )
            candidate_results.append(result)
            if result.status is CandidateRunStatus.SUCCESS and fitted is not None:
                successful_models[result.spec.name] = fitted
                successful_names.append(result.spec.name)
            else:
                failed_names.append(result.spec.name)

        if not successful_names:
            detail_parts = [
                f"{item.spec.name}({item.error_type})"
                for item in candidate_results
                if item.status is CandidateRunStatus.FAILED
            ]
            raise ProcessIntelligenceError(
                "All screening candidates failed: "
                f"candidate_count={len(candidate_results)}, "
                f"failed_models={', '.join(failed_names)}, "
                f"error_types={', '.join(detail_parts)}"
            )

        success_results = [
            item
            for item in candidate_results
            if item.status is CandidateRunStatus.SUCCESS
        ]
        selected_result = _select_best_candidate(success_results, validated_task)
        selected_model = successful_models[selected_result.spec.name]

        baseline_result = _find_baseline(success_results)
        baseline_model_name: str | None = None
        baseline_metrics: dict[str, float] = {}
        selected_beats_baseline: bool | None = None
        summary_warnings: list[str] = []

        if baseline_result is None:
            if self._policy.require_successful_baseline:
                raise ProcessIntelligenceError(
                    "Successful Dummy baseline candidate is required for screening "
                    "but none succeeded"
                )
            summary_warnings.append(_WARNING_NO_BASELINE)
        else:
            baseline_model_name = baseline_result.spec.name
            baseline_metrics = dict(baseline_result.metrics)
            selected_beats_baseline = _beats_baseline(
                task=validated_task,
                selected=selected_result,
                baseline=baseline_result,
            )
            if selected_beats_baseline is False:
                summary_warnings.append(_WARNING_DID_NOT_BEAT_BASELINE)

        if any(
            issue.severity is LeakageSeverity.WARNING
            for issue in leakage_report.issues
        ):
            summary_warnings.append(_WARNING_LEAKAGE)
        if failed_names:
            summary_warnings.append(_WARNING_FAILED_CANDIDATES)

        summary_warnings = _dedupe_preserve_order(summary_warnings)

        metadata = selected_model.get_metadata()
        if metadata.model_name != selected_result.spec.name:
            raise ProcessIntelligenceError(
                "Selected model metadata name does not match screening selection: "
                f"metadata={metadata.model_name!r}, "
                f"selected={selected_result.spec.name!r}"
            )
        if hasattr(selected_model, "is_fitted") and not bool(selected_model.is_fitted):
            raise ProcessIntelligenceError(
                "Selected model is not in a fitted state after screening"
            )

        summary = ModelScreeningSummary(
            task=validated_task,
            target_column=validated_target,
            feature_columns=list(validated_features),
            candidate_results=list(candidate_results),
            selected_model_name=selected_result.spec.name,
            selected_estimator_key=selected_result.spec.estimator_key,
            selected_metrics=dict(selected_result.metrics),
            selected_registry_rank=selected_result.registry_rank,
            ranking_metric=ranking_metric,
            higher_is_better=higher_is_better,
            baseline_model_name=baseline_model_name,
            baseline_metrics=baseline_metrics,
            selected_beats_baseline=selected_beats_baseline,
            successful_model_names=list(successful_names),
            failed_model_names=list(failed_names),
            train_row_count=int(train.height),
            validation_row_count=int(validation.height),
            test_row_count=int(validated_split.test.height),
            warnings=summary_warnings,
        )
        return ModelScreeningOutcome(selected_model=selected_model, summary=summary)

    def _run_candidate(
        self,
        *,
        spec: ModelSpec,
        registry_rank: int,
        task: AnalysisTask,
        ranking_metric: str,
        x_train: pl.DataFrame,
        y_train: pl.Series,
        x_validation: pl.DataFrame,
        y_validation: pl.Series,
    ) -> tuple[CandidateScreeningResult, BaseAnalysisModel | None]:
        """Instantiate, fit, and evaluate one candidate with failure isolation."""
        fit_seconds = 0.0
        evaluation_seconds = 0.0
        try:
            model = self._registry.instantiate(spec)
            fit_started = time.perf_counter()
            model.fit(x_train, y_train)
            fit_seconds = time.perf_counter() - fit_started

            eval_started = time.perf_counter()
            evaluation = model.evaluate(x_validation, y_validation)
            evaluation_seconds = time.perf_counter() - eval_started

            metrics = _extract_and_validate_metrics(evaluation, task)
            primary_value = metrics[ranking_metric]
            result = CandidateScreeningResult(
                spec=spec.model_copy(deep=True),
                status=CandidateRunStatus.SUCCESS,
                registry_rank=registry_rank,
                metrics=dict(metrics),
                primary_metric_name=ranking_metric,
                primary_metric_value=primary_value,
                fit_seconds=fit_seconds,
                evaluation_seconds=evaluation_seconds,
                total_seconds=fit_seconds + evaluation_seconds,
                warnings=[],
                error_type=None,
                error_message=None,
            )
            return result, model
        except _CANDIDATE_FAILURE_EXCEPTIONS as exc:
            if not self._policy.continue_on_candidate_failure:
                raise
            failed = CandidateScreeningResult(
                spec=spec.model_copy(deep=True),
                status=CandidateRunStatus.FAILED,
                registry_rank=registry_rank,
                metrics={},
                primary_metric_name=None,
                primary_metric_value=None,
                fit_seconds=fit_seconds,
                evaluation_seconds=evaluation_seconds,
                total_seconds=fit_seconds + evaluation_seconds,
                warnings=[],
                error_type=type(exc).__name__,
                error_message=str(exc) if str(exc) else type(exc).__name__,
            )
            return failed, None


def _ranking_config(task: AnalysisTask) -> tuple[str, bool]:
    """Return ranking metric name and direction for ``task``."""
    if task is AnalysisTask.REGRESSION:
        return "rmse", False
    return "f1_macro", True


def _validate_task(task: object) -> AnalysisTask:
    """Validate that ``task`` is a supported supervised ``AnalysisTask``."""
    if not isinstance(task, AnalysisTask):
        raise TypeError(f"task must be AnalysisTask, got {type(task).__name__}")
    if task not in _SUPPORTED_TASKS:
        raise DataValidationError(
            "SupervisedModelScreener supports REGRESSION and CLASSIFICATION only, "
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
            "Data leakage blockers detected during screening: "
            f"count={leakage_report.blocker_count}, "
            f"types={', '.join(blocker_types)}"
        )


def _extract_and_validate_metrics(
    evaluation: object,
    task: AnalysisTask,
) -> dict[str, float]:
    """Extract and validate finite metrics from a model evaluation payload."""
    if not isinstance(evaluation, ModelEvaluation):
        raise DataValidationError(
            "model.evaluate must return ModelEvaluation, "
            f"got {type(evaluation).__name__}"
        )
    if not isinstance(evaluation.metrics, dict):
        raise DataValidationError(
            "ModelEvaluation.metrics must be a dict[str, float], "
            f"got {type(evaluation.metrics).__name__}"
        )

    cleaned: dict[str, float] = {}
    for key, raw in evaluation.metrics.items():
        if not isinstance(key, str) or not key:
            raise DataValidationError("metric keys must be non-empty strings")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise DataValidationError(
                f"metric {key!r} must be a finite number "
                f"(bool not allowed), got {type(raw).__name__}"
            )
        number = float(raw)
        if not math.isfinite(number):
            raise DataValidationError(
                f"metric {key!r} must be finite, got {raw!r}"
            )
        cleaned[key] = number

    if task is AnalysisTask.REGRESSION:
        for name in _REGRESSION_REQUIRED_METRICS:
            if name not in cleaned:
                raise DataValidationError(
                    f"regression evaluation missing required metric {name!r}"
                )
        if cleaned["rmse"] < 0.0:
            raise DataValidationError(
                f"rmse must be >= 0, got {cleaned['rmse']}"
            )
        if cleaned["mae"] < 0.0:
            raise DataValidationError(
                f"mae must be >= 0, got {cleaned['mae']}"
            )
        if "r2" in cleaned and not math.isfinite(cleaned["r2"]):
            raise DataValidationError(
                f"r2 must be finite when present, got {cleaned['r2']!r}"
            )
    else:
        for name in _CLASSIFICATION_REQUIRED_METRICS:
            if name not in cleaned:
                raise DataValidationError(
                    f"classification evaluation missing required metric {name!r}"
                )
        for name in _CLASSIFICATION_REQUIRED_METRICS:
            value = cleaned[name]
            if value < 0.0 or value > 1.0:
                raise DataValidationError(
                    f"{name} must be within [0, 1], got {value}"
                )
    return cleaned


def _compare_regression_candidates(
    left: CandidateScreeningResult,
    right: CandidateScreeningResult,
) -> int:
    """Compare SUCCESS regression candidates; negative means ``left`` ranks better."""
    left_rmse = left.metrics["rmse"]
    right_rmse = right.metrics["rmse"]
    if left_rmse != right_rmse:
        return -1 if left_rmse < right_rmse else 1

    left_mae = left.metrics["mae"]
    right_mae = right.metrics["mae"]
    if left_mae != right_mae:
        return -1 if left_mae < right_mae else 1

    left_r2 = left.metrics.get("r2")
    right_r2 = right.metrics.get("r2")
    if left_r2 is not None and right_r2 is not None and left_r2 != right_r2:
        return -1 if left_r2 > right_r2 else 1

    if left.registry_rank != right.registry_rank:
        return -1 if left.registry_rank < right.registry_rank else 1
    return 0


def _classification_sort_key(
    result: CandidateScreeningResult,
) -> tuple[float, float, float, int]:
    """Sort key for classification candidates (descending metrics, ascending rank)."""
    return (
        -result.metrics["f1_macro"],
        -result.metrics["balanced_accuracy"],
        -result.metrics["accuracy"],
        result.registry_rank,
    )


def _select_best_candidate(
    success_results: Sequence[CandidateScreeningResult],
    task: AnalysisTask,
) -> CandidateScreeningResult:
    """Select the best SUCCESS candidate using deterministic ranking rules."""
    if not success_results:
        raise ProcessIntelligenceError("No successful candidates available for selection")
    if task is AnalysisTask.REGRESSION:
        ordered = sorted(
            success_results,
            key=cmp_to_key(_compare_regression_candidates),
        )
        return ordered[0]
    ordered = sorted(success_results, key=_classification_sort_key)
    return ordered[0]


def _is_baseline_spec(spec: ModelSpec) -> bool:
    """Return whether ``spec`` identifies a Dummy baseline candidate."""
    if spec.estimator_key.casefold().startswith("dummy_"):
        return True
    return spec.name.strip().casefold().startswith("dummy")


def _find_baseline(
    success_results: Sequence[CandidateScreeningResult],
) -> CandidateScreeningResult | None:
    """Return the first SUCCESS Dummy baseline in candidate execution order."""
    for result in success_results:
        if _is_baseline_spec(result.spec):
            return result
    return None


def _beats_baseline(
    *,
    task: AnalysisTask,
    selected: CandidateScreeningResult,
    baseline: CandidateScreeningResult,
) -> bool:
    """Return whether the selected candidate strictly beats the baseline."""
    if selected.spec.name == baseline.spec.name:
        return False
    if task is AnalysisTask.REGRESSION:
        return selected.metrics["rmse"] < baseline.metrics["rmse"]
    return selected.metrics["f1_macro"] > baseline.metrics["f1_macro"]


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    """Return unique strings preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
