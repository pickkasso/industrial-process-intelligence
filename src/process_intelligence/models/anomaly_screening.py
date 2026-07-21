"""Unsupervised anomaly model screening on train/validation partitions (Step 7D).

Candidates from ``ModelRegistry`` are fit on the train partition and compared on
the validation partition using label-free quality metrics. The test partition is
never used for fitting, scoring, or model selection.
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
from process_intelligence.core.protocols import BaseAnomalyModel, BaseIndustryProfile
from process_intelligence.core.schemas import ModelSpec
from process_intelligence.data.loader import ORIGINAL_ROW_ID_COLUMN
from process_intelligence.evaluation.leakage import LeakageReport, LeakageSeverity
from process_intelligence.evaluation.splitting import DatasetSplit
from process_intelligence.models.anomaly import AnomalyDetectionResult
from process_intelligence.models.registry import ModelRegistry

_SUCCESS_REQUIRED_METRICS: tuple[str, ...] = (
    "train_anomaly_fraction",
    "validation_anomaly_fraction",
    "anomaly_fraction_gap",
    "validation_score_min",
    "validation_score_max",
    "validation_score_mean",
    "validation_score_std",
    "validation_score_range",
)

_CANDIDATE_FAILURE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ProcessIntelligenceError,
    ValueError,
    TypeError,
    ArithmeticError,
    np.linalg.LinAlgError,
)

_RANKING_METHOD = (
    "quality penalties, non-degenerate detection, standardized score "
    "separation, fraction stability, score variation, registry order"
)

_FLAG_FRACTION_BELOW = "anomaly fraction below expected minimum"
_FLAG_FRACTION_ABOVE = "anomaly fraction above expected maximum"
_FLAG_FRACTION_UNSTABLE = "train-validation anomaly fraction unstable"
_FLAG_SCORE_NEARLY_CONSTANT = "validation anomaly scores nearly constant"
_FLAG_DEGENERATE = "validation detection is degenerate"
_FLAG_NEGATIVE_SEPARATION = (
    "anomaly and normal score direction is inconsistent"
)

_WARNING_LEAKAGE = (
    "LeakageReport contains WARNING issues; proceed with caution"
)
_WARNING_FAILED_CANDIDATES = (
    "One or more screening candidates failed during fit or scoring"
)
_WARNING_SELECTED_QUALITY = (
    "Selected anomaly candidate has one or more quality penalties"
)
_WARNING_SELECTED_DEGENERATE = (
    "Selected anomaly candidate has degenerate validation detection"
)
_WARNING_SELECTED_NO_SEPARATION = (
    "Selected anomaly candidate has no validation score separation"
)


class AnomalyCandidateRunStatus(StrEnum):
    """Execution outcome for a single unsupervised anomaly screening candidate."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class AnomalyScreeningPolicy(BaseModel):
    """Configurable rules for unsupervised anomaly model screening.

    Controls candidate limits, leakage/validation gates, failure isolation,
    and label-free quality thresholds used for ranking.
    """

    maximum_candidates: int | None = None
    candidate_time_budget_seconds: float | None = None
    require_safe_leakage_report: bool = True
    require_validation_partition: bool = True
    continue_on_candidate_failure: bool = True
    minimum_validation_rows: int = 2
    minimum_anomaly_fraction: float = 0.001
    maximum_anomaly_fraction: float = 0.25
    maximum_fraction_gap: float = 0.10
    minimum_score_std: float = 1e-12
    prefer_non_degenerate_detection: bool = True

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

    @field_validator("minimum_validation_rows", mode="before")
    @classmethod
    def _validate_minimum_validation_rows(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "minimum_validation_rows must be an int >= 2 "
                f"(bool not allowed), got {type(value).__name__}"
            )
        if value < 2:
            raise ValueError(f"minimum_validation_rows must be >= 2, got {value}")
        return value

    @field_validator(
        "minimum_anomaly_fraction",
        "maximum_anomaly_fraction",
        "maximum_fraction_gap",
        mode="before",
    )
    @classmethod
    def _validate_fraction_fields(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "fraction fields must be finite floats in [0.0, 1.0] "
                f"(bool not allowed), got {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"fraction fields must be finite, got {value!r}")
        if number < 0.0 or number > 1.0:
            raise ValueError(
                f"fraction fields must be in [0.0, 1.0], got {number}"
            )
        return number

    @field_validator("minimum_score_std", mode="before")
    @classmethod
    def _validate_minimum_score_std(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "minimum_score_std must be a finite float > 0 "
                f"(bool not allowed), got {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"minimum_score_std must be finite, got {value!r}")
        if number <= 0.0:
            raise ValueError(f"minimum_score_std must be > 0, got {number}")
        return number

    @field_validator(
        "require_safe_leakage_report",
        "require_validation_partition",
        "continue_on_candidate_failure",
        "prefer_non_degenerate_detection",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError(f"must be a bool, got {type(value).__name__}")
        return value

    @model_validator(mode="after")
    def _validate_fraction_relationship(self) -> Self:
        if self.minimum_anomaly_fraction >= self.maximum_anomaly_fraction:
            raise ValueError(
                "minimum_anomaly_fraction must be < maximum_anomaly_fraction, "
                f"got {self.minimum_anomaly_fraction} >= "
                f"{self.maximum_anomaly_fraction}"
            )
        return self


class AnomalyCandidateScreeningResult(BaseModel):
    """Per-candidate unsupervised screening outcome with metrics or failure details."""

    spec: ModelSpec
    status: AnomalyCandidateRunStatus
    registry_rank: int
    metrics: dict[str, float] = Field(default_factory=dict)
    score_separation: float | None = None
    has_normal_and_anomaly: bool
    quality_penalty_count: int
    quality_flags: list[str] = Field(default_factory=list)
    fit_seconds: float
    train_scoring_seconds: float
    validation_scoring_seconds: float
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

    @field_validator("has_normal_and_anomaly", mode="before")
    @classmethod
    def _validate_has_groups(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError(
                f"has_normal_and_anomaly must be a bool, got {type(value).__name__}"
            )
        return value

    @field_validator("quality_penalty_count", mode="before")
    @classmethod
    def _validate_penalty_count(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "quality_penalty_count must be an int >= 0 (bool not allowed), "
                f"got {type(value).__name__}"
            )
        if value < 0:
            raise ValueError(f"quality_penalty_count must be >= 0, got {value}")
        return value

    @field_validator(
        "fit_seconds",
        "train_scoring_seconds",
        "validation_scoring_seconds",
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

    @field_validator("score_separation", mode="before")
    @classmethod
    def _validate_score_separation(cls, value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "score_separation must be None or a finite number "
                f"(bool not allowed), got {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"score_separation must be finite, got {value!r}")
        return number

    @field_validator("quality_flags", "warnings", mode="after")
    @classmethod
    def _reject_duplicate_strings(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("list must not contain duplicates")
        return list(value)

    @model_validator(mode="after")
    def _validate_status_consistency(self) -> Self:
        expected_total = (
            self.fit_seconds
            + self.train_scoring_seconds
            + self.validation_scoring_seconds
        )
        if not math.isclose(
            self.total_seconds,
            expected_total,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "total_seconds must equal fit_seconds + train_scoring_seconds "
                "+ validation_scoring_seconds"
            )

        if self.status is AnomalyCandidateRunStatus.SUCCESS:
            for key in _SUCCESS_REQUIRED_METRICS:
                if key not in self.metrics:
                    raise ValueError(
                        f"SUCCESS results require metric {key!r}"
                    )
            if self.error_type is not None or self.error_message is not None:
                raise ValueError(
                    "SUCCESS results must not include error_type or error_message"
                )
            if self.quality_penalty_count != len(self.quality_flags):
                raise ValueError(
                    "quality_penalty_count must equal len(quality_flags)"
                )
            if self.has_normal_and_anomaly and self.score_separation is None:
                raise ValueError(
                    "SUCCESS results with has_normal_and_anomaly=True "
                    "require score_separation"
                )
        elif self.status is AnomalyCandidateRunStatus.FAILED:
            if self.metrics:
                raise ValueError("FAILED results require empty metrics")
            if self.score_separation is not None:
                raise ValueError("FAILED results require score_separation=None")
            if self.has_normal_and_anomaly:
                raise ValueError(
                    "FAILED results require has_normal_and_anomaly=False"
                )
            if self.quality_penalty_count != 0:
                raise ValueError(
                    "FAILED results require quality_penalty_count=0"
                )
            if self.quality_flags:
                raise ValueError("FAILED results require empty quality_flags")
            if self.error_type is None or not str(self.error_type).strip():
                raise ValueError("FAILED results require a non-empty error_type")
            if self.error_message is None or not str(self.error_message).strip():
                raise ValueError(
                    "FAILED results require a non-empty error_message"
                )
        return self


class AnomalyScreeningSummary(BaseModel):
    """Aggregate unsupervised anomaly screening summary and selection metadata."""

    task: AnalysisTask
    feature_columns: list[str] = Field(default_factory=list)
    candidate_results: list[AnomalyCandidateScreeningResult] = Field(
        default_factory=list
    )
    selected_model_name: str
    selected_estimator_key: str
    selected_registry_rank: int
    selected_metrics: dict[str, float] = Field(default_factory=dict)
    selected_score_separation: float | None = None
    selected_quality_penalty_count: int
    ranking_method: str
    successful_model_names: list[str] = Field(default_factory=list)
    failed_model_names: list[str] = Field(default_factory=list)
    train_row_count: int
    validation_row_count: int
    test_row_count: int
    warnings: list[str] = Field(default_factory=list)

    @field_validator("task", mode="before")
    @classmethod
    def _validate_task(cls, value: object) -> AnalysisTask:
        if not isinstance(value, AnalysisTask):
            raise ValueError(
                f"task must be AnalysisTask, got {type(value).__name__}"
            )
        if value is not AnalysisTask.UNSUPERVISED_ANOMALY:
            raise ValueError(
                "task must be AnalysisTask.UNSUPERVISED_ANOMALY, "
                f"got {value!r}"
            )
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
                "feature_columns must not include reserved column "
                f"{ORIGINAL_ROW_ID_COLUMN!r}"
            )
        return list(value)

    @field_validator(
        "selected_model_name",
        "selected_estimator_key",
        "ranking_method",
        mode="before",
    )
    @classmethod
    def _validate_non_empty_strings(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(f"must be str, got {type(value).__name__}")
        if value == "" or value.strip() == "":
            raise ValueError("must be a non-empty, non-whitespace string")
        return value

    @field_validator("selected_metrics", mode="before")
    @classmethod
    def _validate_selected_metrics(cls, value: object) -> dict[str, float]:
        if not isinstance(value, dict):
            raise ValueError(
                f"selected_metrics must be dict[str, float], "
                f"got {type(value).__name__}"
            )
        if not value:
            raise ValueError("selected_metrics must not be empty")
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

    @field_validator("selected_score_separation", mode="before")
    @classmethod
    def _validate_selected_separation(cls, value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "selected_score_separation must be None or a finite number "
                f"(bool not allowed), got {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(
                f"selected_score_separation must be finite, got {value!r}"
            )
        return number

    @field_validator("selected_registry_rank", "selected_quality_penalty_count", mode="before")
    @classmethod
    def _validate_non_negative_int(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "must be an int >= 0 (bool not allowed), "
                f"got {type(value).__name__}"
            )
        if value < 0:
            raise ValueError(f"must be >= 0, got {value}")
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
        if self.train_row_count < 1:
            raise ValueError("train_row_count must be >= 1")
        if self.validation_row_count < 1:
            raise ValueError("validation_row_count must be >= 1")

        success_results = [
            result
            for result in self.candidate_results
            if result.status is AnomalyCandidateRunStatus.SUCCESS
        ]
        if not success_results:
            raise ValueError("candidate_results must include at least one SUCCESS")

        selected_matches = [
            result
            for result in success_results
            if (
                result.spec.name == self.selected_model_name
                and result.spec.estimator_key == self.selected_estimator_key
                and result.registry_rank == self.selected_registry_rank
            )
        ]
        if len(selected_matches) != 1:
            raise ValueError(
                "selected model must match exactly one SUCCESS candidate "
                "by name, estimator_key, and registry_rank"
            )
        selected = selected_matches[0]
        if selected.metrics != self.selected_metrics:
            raise ValueError(
                "selected_metrics must match the selected SUCCESS candidate metrics"
            )
        if selected.score_separation != self.selected_score_separation:
            raise ValueError(
                "selected_score_separation must match the selected SUCCESS candidate"
            )
        if selected.quality_penalty_count != self.selected_quality_penalty_count:
            raise ValueError(
                "selected_quality_penalty_count must match the selected "
                "SUCCESS candidate"
            )

        overlap = set(self.successful_model_names) & set(self.failed_model_names)
        if overlap:
            raise ValueError(
                "successful_model_names and failed_model_names must be disjoint, "
                f"overlap={sorted(overlap)}"
            )
        return self


@dataclass(frozen=True, slots=True)
class AnomalyScreeningOutcome:
    """Selected fitted anomaly model paired with an immutable screening summary."""

    selected_model: BaseAnomalyModel
    summary: AnomalyScreeningSummary


class UnsupervisedAnomalyModelScreener:
    """Screen unsupervised anomaly registry candidates on train/validation.

    Fits each available candidate on the train partition, scores train and
    validation with ``detect``, and selects a winner by deterministic
    label-free ranking rules. The test partition is never used for selection.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        policy: AnomalyScreeningPolicy | None = None,
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
            stored_policy = AnomalyScreeningPolicy()
        elif isinstance(policy, AnomalyScreeningPolicy):
            stored_policy = policy.model_copy(deep=True)
        else:
            raise TypeError(
                "policy must be AnomalyScreeningPolicy or None, "
                f"got {type(policy).__name__}"
            )
        self._registry = registry
        self._policy = stored_policy

    def screen(
        self,
        split: DatasetSplit,
        *,
        feature_columns: Sequence[str],
        leakage_report: LeakageReport,
        industry_profile: BaseIndustryProfile | None = None,
    ) -> AnomalyScreeningOutcome:
        """Fit and rank unsupervised anomaly candidates without using test data.

        Args:
            split: Train/validation/test partitions. Only train and validation
                are used for fitting and ranking.
            feature_columns: Feature column names in evaluation order.
            leakage_report: Structural leakage report for the feature set.
            industry_profile: Optional industry profile forwarded to the registry.

        Returns:
            Outcome containing the selected fitted anomaly model and summary.

        Raises:
            TypeError: If inputs have invalid types.
            DataValidationError: If inputs fail structural validation.
            DataLeakageError: If blockers are present and the policy requires a
                safe leakage report.
            InsufficientDataError: If train is empty or validation is too small.
            ProcessIntelligenceError: If no candidates are available, all
                candidates fail, or the selected model is inconsistent.
        """
        validated_features = _validate_feature_columns(feature_columns)
        validated_split = _validate_split(
            split,
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
                "train partition must contain at least one row for "
                "anomaly screening"
            )
        if validation.height < self._policy.minimum_validation_rows:
            raise InsufficientDataError(
                "validation partition must contain at least "
                f"{self._policy.minimum_validation_rows} rows for anomaly "
                f"screening, got {validation.height}"
            )

        x_train = train.select(validated_features)
        x_validation = validation.select(validated_features)

        candidates = self._registry.get_candidates(
            AnalysisTask.UNSUPERVISED_ANOMALY,
            industry_profile=industry_profile,
            time_budget_seconds=self._policy.candidate_time_budget_seconds,
        )
        if not candidates:
            raise ProcessIntelligenceError(
                "No available anomaly screening candidates for task="
                f"{AnalysisTask.UNSUPERVISED_ANOMALY!r}"
            )

        candidate_results: list[AnomalyCandidateScreeningResult] = []
        successful_models: dict[str, BaseAnomalyModel] = {}
        successful_names: list[str] = []
        failed_names: list[str] = []
        max_candidates = self._policy.maximum_candidates

        for rank, spec in enumerate(candidates):
            if max_candidates is not None and rank >= max_candidates:
                break

            result, fitted = self._run_candidate(
                spec=spec,
                registry_rank=rank,
                x_train=x_train,
                x_validation=x_validation,
            )
            candidate_results.append(result)
            if (
                result.status is AnomalyCandidateRunStatus.SUCCESS
                and fitted is not None
            ):
                successful_models[result.spec.name] = fitted
                successful_names.append(result.spec.name)
            else:
                failed_names.append(result.spec.name)

        if not successful_names:
            detail_parts = [
                f"{item.spec.name}({item.error_type})"
                for item in candidate_results
                if item.status is AnomalyCandidateRunStatus.FAILED
            ]
            raise ProcessIntelligenceError(
                "All anomaly screening candidates failed: "
                f"candidate_count={len(candidate_results)}, "
                f"failed_models={', '.join(failed_names)}, "
                f"error_types={', '.join(detail_parts)}"
            )

        success_results = [
            item
            for item in candidate_results
            if item.status is AnomalyCandidateRunStatus.SUCCESS
        ]
        selected_result = _select_best_candidate(
            success_results,
            prefer_non_degenerate=self._policy.prefer_non_degenerate_detection,
        )
        selected_model = successful_models[selected_result.spec.name]

        summary_warnings = _build_summary_warnings(
            leakage_report=leakage_report,
            failed_names=failed_names,
            selected_result=selected_result,
        )

        _validate_selected_model_consistency(
            selected_model=selected_model,
            selected_result=selected_result,
            feature_columns=validated_features,
            train_row_count=int(train.height),
        )

        summary = AnomalyScreeningSummary(
            task=AnalysisTask.UNSUPERVISED_ANOMALY,
            feature_columns=list(validated_features),
            candidate_results=list(candidate_results),
            selected_model_name=selected_result.spec.name,
            selected_estimator_key=selected_result.spec.estimator_key,
            selected_registry_rank=selected_result.registry_rank,
            selected_metrics=dict(selected_result.metrics),
            selected_score_separation=selected_result.score_separation,
            selected_quality_penalty_count=selected_result.quality_penalty_count,
            ranking_method=_RANKING_METHOD,
            successful_model_names=list(successful_names),
            failed_model_names=list(failed_names),
            train_row_count=int(train.height),
            validation_row_count=int(validation.height),
            test_row_count=int(validated_split.test.height),
            warnings=summary_warnings,
        )
        return AnomalyScreeningOutcome(
            selected_model=selected_model,
            summary=summary,
        )

    def _run_candidate(
        self,
        *,
        spec: ModelSpec,
        registry_rank: int,
        x_train: pl.DataFrame,
        x_validation: pl.DataFrame,
    ) -> tuple[AnomalyCandidateScreeningResult, BaseAnomalyModel | None]:
        """Instantiate, fit, and score one candidate with failure isolation."""
        fit_seconds = 0.0
        train_scoring_seconds = 0.0
        validation_scoring_seconds = 0.0
        try:
            model = self._registry.instantiate(spec)
            if not isinstance(model, BaseAnomalyModel):
                raise DataValidationError(
                    "registry.instantiate must return BaseAnomalyModel for "
                    "anomaly screening, "
                    f"got {type(model).__name__}"
                )

            fit_started = time.perf_counter()
            model.fit(x_train)
            fit_seconds = time.perf_counter() - fit_started

            detect = getattr(model, "detect", None)
            if not callable(detect):
                raise DataValidationError(
                    "anomaly screening candidates must expose a callable detect "
                    f"method, got {type(model).__name__}"
                )

            train_started = time.perf_counter()
            train_result = detect(x_train)
            train_scoring_seconds = time.perf_counter() - train_started
            _validate_detection_result(
                train_result,
                expected_row_count=x_train.height,
                partition_name="train",
            )

            validation_started = time.perf_counter()
            validation_result = detect(x_validation)
            validation_scoring_seconds = (
                time.perf_counter() - validation_started
            )
            _validate_detection_result(
                validation_result,
                expected_row_count=x_validation.height,
                partition_name="validation",
            )

            metrics, score_separation, has_groups = _compute_label_free_metrics(
                train_result=train_result,
                validation_result=validation_result,
                minimum_score_std=self._policy.minimum_score_std,
            )
            quality_flags = _compute_quality_flags(
                metrics=metrics,
                has_normal_and_anomaly=has_groups,
                score_separation=score_separation,
                policy=self._policy,
            )
            result = AnomalyCandidateScreeningResult(
                spec=spec.model_copy(deep=True),
                status=AnomalyCandidateRunStatus.SUCCESS,
                registry_rank=registry_rank,
                metrics=dict(metrics),
                score_separation=score_separation,
                has_normal_and_anomaly=has_groups,
                quality_penalty_count=len(quality_flags),
                quality_flags=list(quality_flags),
                fit_seconds=fit_seconds,
                train_scoring_seconds=train_scoring_seconds,
                validation_scoring_seconds=validation_scoring_seconds,
                total_seconds=(
                    fit_seconds
                    + train_scoring_seconds
                    + validation_scoring_seconds
                ),
                warnings=[],
                error_type=None,
                error_message=None,
            )
            return result, model
        except _CANDIDATE_FAILURE_EXCEPTIONS as exc:
            if not self._policy.continue_on_candidate_failure:
                raise
            failed = AnomalyCandidateScreeningResult(
                spec=spec.model_copy(deep=True),
                status=AnomalyCandidateRunStatus.FAILED,
                registry_rank=registry_rank,
                metrics={},
                score_separation=None,
                has_normal_and_anomaly=False,
                quality_penalty_count=0,
                quality_flags=[],
                fit_seconds=fit_seconds,
                train_scoring_seconds=train_scoring_seconds,
                validation_scoring_seconds=validation_scoring_seconds,
                total_seconds=(
                    fit_seconds
                    + train_scoring_seconds
                    + validation_scoring_seconds
                ),
                warnings=[],
                error_type=type(exc).__name__,
                error_message=str(exc) if str(exc) else type(exc).__name__,
            )
            return failed, None


def _validate_feature_columns(feature_columns: object) -> list[str]:
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
    feature_columns: Sequence[str],
) -> DatasetSplit:
    """Validate split type, partition frames, schema, dtypes, and columns."""
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
            "train and validation partitions must share identical column "
            "names and order"
        )
    if list(split.test.columns) != train_columns:
        raise DataValidationError(
            "train and test partitions must share identical column names "
            "and order"
        )

    train_dtypes = list(split.train.dtypes)
    if list(split.validation.dtypes) != train_dtypes:
        raise DataValidationError(
            "train and validation partitions must share identical dtypes"
        )
    if list(split.test.dtypes) != train_dtypes:
        raise DataValidationError(
            "train and test partitions must share identical dtypes"
        )

    if ORIGINAL_ROW_ID_COLUMN not in train_columns:
        raise DataValidationError(
            f"all partitions require reserved column {ORIGINAL_ROW_ID_COLUMN!r}"
        )

    missing_features = [
        name for name in feature_columns if name not in train_columns
    ]
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
            "Data leakage blockers detected during anomaly screening: "
            f"count={leakage_report.blocker_count}, "
            f"types={', '.join(blocker_types)}"
        )


def _validate_detection_result(
    result: object,
    *,
    expected_row_count: int,
    partition_name: str,
) -> AnomalyDetectionResult:
    """Validate an ``AnomalyDetectionResult`` against partition expectations."""
    if not isinstance(result, AnomalyDetectionResult):
        raise DataValidationError(
            f"{partition_name} detect must return AnomalyDetectionResult, "
            f"got {type(result).__name__}"
        )
    if result.row_count != expected_row_count:
        raise DataValidationError(
            f"{partition_name} detect row_count ({result.row_count}) must "
            f"equal partition height ({expected_row_count})"
        )
    if len(result.scores) != expected_row_count:
        raise DataValidationError(
            f"{partition_name} detect scores length must equal partition height"
        )
    if len(result.is_anomaly) != expected_row_count:
        raise DataValidationError(
            f"{partition_name} detect is_anomaly length must equal "
            "partition height"
        )
    if len(result.raw_predictions) != expected_row_count:
        raise DataValidationError(
            f"{partition_name} detect raw_predictions length must equal "
            "partition height"
        )
    for index, score in enumerate(result.scores):
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise DataValidationError(
                f"{partition_name} detect scores[{index}] must be a finite number"
            )
        if not math.isfinite(float(score)):
            raise DataValidationError(
                f"{partition_name} detect scores[{index}] must be finite, "
                f"got {score!r}"
            )
    for index, raw in enumerate(result.raw_predictions):
        if raw not in (-1, 1):
            raise DataValidationError(
                f"{partition_name} detect raw_predictions[{index}] must be "
                f"-1 or 1, got {raw!r}"
            )
        expected_anomaly = raw == -1
        if bool(result.is_anomaly[index]) is not expected_anomaly:
            raise DataValidationError(
                f"{partition_name} detect is_anomaly must match "
                f"raw_predictions == -1 at index {index}"
            )
    true_count = sum(1 for flag in result.is_anomaly if flag)
    if true_count != result.anomaly_count:
        raise DataValidationError(
            f"{partition_name} detect anomaly_count must equal True flags "
            f"({true_count}), got {result.anomaly_count}"
        )
    if expected_row_count > 0:
        expected_fraction = result.anomaly_count / expected_row_count
        if not math.isclose(
            float(result.anomaly_fraction),
            expected_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise DataValidationError(
                f"{partition_name} detect anomaly_fraction must equal "
                "anomaly_count / row_count"
            )
        if (
            result.score_min is None
            or result.score_max is None
            or result.score_mean is None
        ):
            raise DataValidationError(
                f"{partition_name} detect requires score_min/max/mean when "
                "row_count >= 1"
            )
        for name, summary in (
            ("score_min", result.score_min),
            ("score_max", result.score_max),
            ("score_mean", result.score_mean),
        ):
            if isinstance(summary, bool) or not isinstance(summary, (int, float)):
                raise DataValidationError(
                    f"{partition_name} detect {name} must be a finite number"
                )
            if not math.isfinite(float(summary)):
                raise DataValidationError(
                    f"{partition_name} detect {name} must be finite"
                )
        recomputed_min = float(min(result.scores))
        recomputed_max = float(max(result.scores))
        recomputed_mean = float(sum(result.scores) / expected_row_count)
        if not math.isclose(
            float(result.score_min),
            recomputed_min,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise DataValidationError(
                f"{partition_name} detect score_min does not match scores"
            )
        if not math.isclose(
            float(result.score_max),
            recomputed_max,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise DataValidationError(
                f"{partition_name} detect score_max does not match scores"
            )
        if not math.isclose(
            float(result.score_mean),
            recomputed_mean,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise DataValidationError(
                f"{partition_name} detect score_mean does not match scores"
            )
    if isinstance(result.threshold, bool) or not isinstance(
        result.threshold, (int, float)
    ):
        raise DataValidationError(
            f"{partition_name} detect threshold must be a finite number"
        )
    if not math.isfinite(float(result.threshold)):
        raise DataValidationError(
            f"{partition_name} detect threshold must be finite"
        )
    return result


def _compute_label_free_metrics(
    *,
    train_result: AnomalyDetectionResult,
    validation_result: AnomalyDetectionResult,
    minimum_score_std: float,
) -> tuple[dict[str, float], float | None, bool]:
    """Compute deterministic label-free metrics from train/validation detects."""
    train_fraction = float(train_result.anomaly_fraction)
    validation_fraction = float(validation_result.anomaly_fraction)
    fraction_gap = abs(validation_fraction - train_fraction)

    scores = np.asarray(validation_result.scores, dtype=np.float64)
    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    score_mean = float(np.mean(scores))
    score_std = float(np.std(scores, ddof=0))
    score_range = score_max - score_min

    metrics: dict[str, float] = {
        "train_anomaly_fraction": train_fraction,
        "validation_anomaly_fraction": validation_fraction,
        "anomaly_fraction_gap": fraction_gap,
        "validation_score_min": score_min,
        "validation_score_max": score_max,
        "validation_score_mean": score_mean,
        "validation_score_std": score_std,
        "validation_score_range": score_range,
    }
    for key, value in metrics.items():
        if not math.isfinite(value):
            raise DataValidationError(
                f"computed metric {key!r} must be finite, got {value!r}"
            )

    raw = list(validation_result.raw_predictions)
    anomaly_scores = [
        float(score)
        for score, prediction in zip(validation_result.scores, raw, strict=True)
        if prediction == -1
    ]
    normal_scores = [
        float(score)
        for score, prediction in zip(validation_result.scores, raw, strict=True)
        if prediction == 1
    ]
    has_groups = bool(anomaly_scores) and bool(normal_scores)
    if not has_groups:
        return metrics, None, False

    mean_anomaly = float(sum(anomaly_scores) / len(anomaly_scores))
    mean_normal = float(sum(normal_scores) / len(normal_scores))
    denominator = max(score_std, minimum_score_std)
    separation = (mean_anomaly - mean_normal) / denominator
    if not math.isfinite(separation):
        raise DataValidationError(
            f"score_separation must be finite, got {separation!r}"
        )
    return metrics, float(separation), True


def _compute_quality_flags(
    *,
    metrics: dict[str, float],
    has_normal_and_anomaly: bool,
    score_separation: float | None,
    policy: AnomalyScreeningPolicy,
) -> list[str]:
    """Build ordered, deduplicated quality flags for a SUCCESS candidate."""
    flags: list[str] = []
    validation_fraction = metrics["validation_anomaly_fraction"]
    if validation_fraction < policy.minimum_anomaly_fraction:
        flags.append(_FLAG_FRACTION_BELOW)
    if validation_fraction > policy.maximum_anomaly_fraction:
        flags.append(_FLAG_FRACTION_ABOVE)
    if metrics["anomaly_fraction_gap"] > policy.maximum_fraction_gap:
        flags.append(_FLAG_FRACTION_UNSTABLE)
    if metrics["validation_score_std"] < policy.minimum_score_std:
        flags.append(_FLAG_SCORE_NEARLY_CONSTANT)
    if not has_normal_and_anomaly:
        flags.append(_FLAG_DEGENERATE)
    if score_separation is not None and score_separation < 0.0:
        flags.append(_FLAG_NEGATIVE_SEPARATION)
    return flags


def _compare_anomaly_candidates(
    left: AnomalyCandidateScreeningResult,
    right: AnomalyCandidateScreeningResult,
    *,
    prefer_non_degenerate: bool,
) -> int:
    """Compare SUCCESS candidates; negative means ``left`` ranks better."""
    if left.quality_penalty_count != right.quality_penalty_count:
        return (
            -1
            if left.quality_penalty_count < right.quality_penalty_count
            else 1
        )

    if prefer_non_degenerate:
        if left.has_normal_and_anomaly != right.has_normal_and_anomaly:
            return -1 if left.has_normal_and_anomaly else 1

    left_sep = left.score_separation
    right_sep = right.score_separation
    if left_sep is None and right_sep is not None:
        return 1
    if left_sep is not None and right_sep is None:
        return -1
    if (
        left_sep is not None
        and right_sep is not None
        and left_sep != right_sep
    ):
        return -1 if left_sep > right_sep else 1

    left_gap = left.metrics["anomaly_fraction_gap"]
    right_gap = right.metrics["anomaly_fraction_gap"]
    if left_gap != right_gap:
        return -1 if left_gap < right_gap else 1

    left_std = left.metrics["validation_score_std"]
    right_std = right.metrics["validation_score_std"]
    if left_std != right_std:
        return -1 if left_std > right_std else 1

    if left.registry_rank != right.registry_rank:
        return -1 if left.registry_rank < right.registry_rank else 1
    return 0


def _select_best_candidate(
    success_results: Sequence[AnomalyCandidateScreeningResult],
    *,
    prefer_non_degenerate: bool,
) -> AnomalyCandidateScreeningResult:
    """Select the best SUCCESS candidate using deterministic ranking rules."""
    if not success_results:
        raise ProcessIntelligenceError(
            "No successful anomaly candidates available for selection"
        )

    def _cmp(
        left: AnomalyCandidateScreeningResult,
        right: AnomalyCandidateScreeningResult,
    ) -> int:
        return _compare_anomaly_candidates(
            left,
            right,
            prefer_non_degenerate=prefer_non_degenerate,
        )

    ordered = sorted(success_results, key=cmp_to_key(_cmp))
    return ordered[0]


def _build_summary_warnings(
    *,
    leakage_report: LeakageReport,
    failed_names: Sequence[str],
    selected_result: AnomalyCandidateScreeningResult,
) -> list[str]:
    """Build ordered, deduplicated summary warnings."""
    warnings: list[str] = []
    if any(
        issue.severity is LeakageSeverity.WARNING
        for issue in leakage_report.issues
    ):
        warnings.append(_WARNING_LEAKAGE)
    if failed_names:
        warnings.append(_WARNING_FAILED_CANDIDATES)
    if selected_result.quality_penalty_count >= 1:
        warnings.append(_WARNING_SELECTED_QUALITY)
    if not selected_result.has_normal_and_anomaly:
        warnings.append(_WARNING_SELECTED_DEGENERATE)
    if selected_result.score_separation is None:
        warnings.append(_WARNING_SELECTED_NO_SEPARATION)
    return _dedupe_preserve_order(warnings)


def _validate_selected_model_consistency(
    *,
    selected_model: BaseAnomalyModel,
    selected_result: AnomalyCandidateScreeningResult,
    feature_columns: Sequence[str],
    train_row_count: int,
) -> None:
    """Validate selected fitted model metadata against the screening summary."""
    if not isinstance(selected_model, BaseAnomalyModel):
        raise ProcessIntelligenceError(
            "selected_model must be BaseAnomalyModel, "
            f"got {type(selected_model).__name__}"
        )
    if hasattr(selected_model, "is_fitted") and not bool(selected_model.is_fitted):
        raise ProcessIntelligenceError(
            "Selected anomaly model is not in a fitted state after screening"
        )

    metadata = selected_model.get_metadata()
    if metadata.model_name != selected_result.spec.name:
        raise ProcessIntelligenceError(
            "Selected model metadata name does not match screening selection: "
            f"metadata={metadata.model_name!r}, "
            f"selected={selected_result.spec.name!r}"
        )

    estimator_key = getattr(metadata, "estimator_key", None)
    if estimator_key != selected_result.spec.estimator_key:
        raise ProcessIntelligenceError(
            "Selected model metadata estimator_key does not match selection: "
            f"metadata={estimator_key!r}, "
            f"selected={selected_result.spec.estimator_key!r}"
        )
    if metadata.task is not AnalysisTask.UNSUPERVISED_ANOMALY:
        raise ProcessIntelligenceError(
            "Selected model metadata task must be UNSUPERVISED_ANOMALY, "
            f"got {metadata.task!r}"
        )
    if list(metadata.features) != list(feature_columns):
        raise ProcessIntelligenceError(
            "Selected model metadata features must match feature_columns: "
            f"expected {list(feature_columns)}, got {list(metadata.features)}"
        )
    fit_row_count = getattr(metadata, "fit_row_count", None)
    if fit_row_count != train_row_count:
        raise ProcessIntelligenceError(
            "Selected model metadata fit_row_count must equal train height "
            f"({train_row_count}), got {fit_row_count}"
        )


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
