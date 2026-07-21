"""Model performance acceptance against independent test metrics (Step 10D).

Compares existing public metrics from a supervised ``FinalEvaluationReport``
(or ``FinalEvaluationOutcome``) to an explicit user-provided acceptance policy.
Does not recompute metrics, call predict, or access models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.evaluation.final_evaluation import (
    FinalEvaluationOutcome,
    FinalEvaluationReport,
)

ScalarMetadataValue = str | int | float | bool | None
"""Allowed scalar types for acceptance report metadata dictionaries."""


class MetricAcceptanceDirection(StrEnum):
    """Direction used to compare an observed metric against a threshold."""

    HIGHER_IS_BETTER = "HIGHER_IS_BETTER"
    LOWER_IS_BETTER = "LOWER_IS_BETTER"


class ModelPerformanceAcceptanceStatus(StrEnum):
    """Overall model-performance acceptance outcome for one assessment."""

    ACCEPTABLE = "ACCEPTABLE"
    UNACCEPTABLE = "UNACCEPTABLE"
    UNAVAILABLE = "UNAVAILABLE"


def _require_non_empty_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be str, got {type(value).__name__}")
    if value == "" or value.strip() == "":
        raise ValueError(f"{field_name} must be a non-empty, non-whitespace string")
    return value


def _require_strict_bool(value: object, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(
            f"{field_name} must be a bool (0/1 and strings rejected), "
            f"got {type(value).__name__}"
        )
    return value


def _require_strict_int_ge(
    value: object,
    *,
    field_name: str,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{field_name} must be an int >= {minimum} "
            f"(bool not allowed), got {type(value).__name__}"
        )
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {value}")
    return value


def _require_finite_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a finite float "
            f"(bool not allowed), got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return number


def _require_timezone_aware(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _validate_scalar_metadata(
    value: object,
    *,
    field_name: str = "metadata",
) -> dict[str, ScalarMetadataValue]:
    if not isinstance(value, dict):
        raise ValueError(
            f"{field_name} must be a dict[str, scalar], got {type(value).__name__}"
        )
    cleaned: dict[str, ScalarMetadataValue] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or key == "" or key.strip() == "":
            raise ValueError(f"{field_name} keys must be non-empty strings")
        if raw is None or isinstance(raw, (str, bool)):
            cleaned[key] = raw
            continue
        if isinstance(raw, int) and not isinstance(raw, bool):
            cleaned[key] = raw
            continue
        if isinstance(raw, float):
            if not math.isfinite(raw):
                raise ValueError(
                    f"{field_name}[{key!r}] float must be finite, got {raw!r}"
                )
            cleaned[key] = raw
            continue
        raise ValueError(
            f"{field_name}[{key!r}] must be str, int, float, bool, or None "
            f"(no DataFrame, ndarray, estimator, or nested objects); "
            f"got {type(raw).__name__}"
        )
    return cleaned


def _validate_unique_non_empty_strings(
    values: list[str],
    *,
    field_name: str,
) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _require_non_empty_str(item, field_name=field_name)
        if text in seen:
            raise ValueError(f"{field_name} must not contain duplicates: {text!r}")
        seen.add(text)
        cleaned.append(text)
    return cleaned


class MetricAcceptanceRule(BaseModel):
    """Single metric threshold rule for independent-test performance acceptance.

    ``metric_name`` must match a public key in ``FinalEvaluationReport.test_metrics``
    exactly. Direction is never inferred from the metric name.
    """

    metric_name: str
    direction: MetricAcceptanceDirection
    threshold: float
    required: bool = True

    @field_validator("metric_name", mode="before")
    @classmethod
    def _validate_metric_name(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="metric_name")

    @field_validator("direction", mode="before")
    @classmethod
    def _validate_direction(cls, value: object) -> MetricAcceptanceDirection:
        if isinstance(value, MetricAcceptanceDirection):
            return value
        if isinstance(value, str):
            try:
                return MetricAcceptanceDirection(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid MetricAcceptanceDirection: {value!r}"
                ) from exc
        raise ValueError(
            f"direction must be MetricAcceptanceDirection, got {type(value).__name__}"
        )

    @field_validator("threshold", mode="before")
    @classmethod
    def _validate_threshold(cls, value: object) -> float:
        return _require_finite_float(value, field_name="threshold")

    @field_validator("required", mode="before")
    @classmethod
    def _validate_required(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="required")


class ModelPerformanceAcceptancePolicy(BaseModel):
    """Explicit acceptance criteria for independent supervised test metrics.

    Callers must supply thresholds. This policy does not embed default R2/MAE
    cutoffs.
    """

    rules: list[MetricAcceptanceRule]
    require_all_required_rules: bool = True
    minimum_test_rows: int = 1
    unavailable_metric_is_failure: bool = True

    @field_validator("rules", mode="before")
    @classmethod
    def _validate_rules_before(cls, value: object) -> list[MetricAcceptanceRule]:
        if not isinstance(value, list):
            raise ValueError(
                f"rules must be a list[MetricAcceptanceRule], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("rules", mode="after")
    @classmethod
    def _validate_rules(cls, value: list[MetricAcceptanceRule]) -> list[MetricAcceptanceRule]:
        if not value:
            raise ValueError("rules must contain at least one MetricAcceptanceRule")
        seen: set[str] = set()
        copied: list[MetricAcceptanceRule] = []
        for item in value:
            if not isinstance(item, MetricAcceptanceRule):
                raise ValueError(
                    "rules entries must be MetricAcceptanceRule, "
                    f"got {type(item).__name__}"
                )
            name = item.metric_name
            if name in seen:
                raise ValueError(
                    f"rules must not contain duplicate metric_name: {name!r}"
                )
            seen.add(name)
            copied.append(item.model_copy(deep=True))
        return copied

    @field_validator(
        "require_all_required_rules",
        "unavailable_metric_is_failure",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="policy bool field")

    @field_validator("minimum_test_rows", mode="before")
    @classmethod
    def _validate_minimum_test_rows(cls, value: object) -> int:
        return _require_strict_int_ge(
            value,
            field_name="minimum_test_rows",
            minimum=1,
        )


class MetricAcceptanceResult(BaseModel):
    """Per-metric comparison of an observed independent-test value to a rule."""

    metric_name: str
    observed_value: float | None
    threshold: float
    direction: MetricAcceptanceDirection
    required: bool
    available: bool
    passed: bool | None
    message: str

    @field_validator("metric_name", "message", mode="before")
    @classmethod
    def _validate_non_empty_str(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="string field")

    @field_validator("threshold", mode="before")
    @classmethod
    def _validate_threshold(cls, value: object) -> float:
        return _require_finite_float(value, field_name="threshold")

    @field_validator("observed_value", mode="before")
    @classmethod
    def _validate_observed_value(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_finite_float(value, field_name="observed_value")

    @field_validator("direction", mode="before")
    @classmethod
    def _validate_direction(cls, value: object) -> MetricAcceptanceDirection:
        if isinstance(value, MetricAcceptanceDirection):
            return value
        if isinstance(value, str):
            try:
                return MetricAcceptanceDirection(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid MetricAcceptanceDirection: {value!r}"
                ) from exc
        raise ValueError(
            f"direction must be MetricAcceptanceDirection, got {type(value).__name__}"
        )

    @field_validator("required", "available", mode="before")
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="result bool field")

    @field_validator("passed", mode="before")
    @classmethod
    def _validate_passed(cls, value: object) -> bool | None:
        if value is None:
            return None
        return _require_strict_bool(value, field_name="passed")

    @model_validator(mode="after")
    def _validate_availability_relationship(self) -> Self:
        if not self.available:
            if self.observed_value is not None:
                raise ValueError(
                    "observed_value must be None when available=False"
                )
            if self.passed is not None:
                raise ValueError("passed must be None when available=False")
        else:
            if self.observed_value is None:
                raise ValueError(
                    "observed_value must be a finite float when available=True"
                )
            if self.passed is None:
                raise ValueError("passed must be a bool when available=True")
        return self


class ModelPerformanceAcceptanceReport(BaseModel):
    """Structured acceptance assessment against independent test metrics."""

    status: ModelPerformanceAcceptanceStatus
    task: AnalysisTask
    evaluation_available: bool
    independent_test_evaluation: bool
    test_row_count: int
    metric_results: list[MetricAcceptanceResult]
    required_rule_count: int
    passed_required_rule_count: int
    failed_required_rule_count: int
    unavailable_required_rule_count: int
    assessed_at: datetime
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)

    @field_validator("status", mode="before")
    @classmethod
    def _validate_status(cls, value: object) -> ModelPerformanceAcceptanceStatus:
        if isinstance(value, ModelPerformanceAcceptanceStatus):
            return value
        if isinstance(value, str):
            try:
                return ModelPerformanceAcceptanceStatus(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid ModelPerformanceAcceptanceStatus: {value!r}"
                ) from exc
        raise ValueError(
            "status must be ModelPerformanceAcceptanceStatus, "
            f"got {type(value).__name__}"
        )

    @field_validator("task", mode="before")
    @classmethod
    def _validate_task(cls, value: object) -> AnalysisTask:
        if isinstance(value, AnalysisTask):
            return value
        if isinstance(value, str):
            try:
                return AnalysisTask(value)
            except ValueError as exc:
                raise ValueError(f"invalid AnalysisTask: {value!r}") from exc
        raise ValueError(
            f"task must be AnalysisTask, got {type(value).__name__}"
        )

    @field_validator(
        "evaluation_available",
        "independent_test_evaluation",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="report bool field")

    @field_validator(
        "test_row_count",
        "required_rule_count",
        "passed_required_rule_count",
        "failed_required_rule_count",
        "unavailable_required_rule_count",
        mode="before",
    )
    @classmethod
    def _validate_counts(cls, value: object) -> int:
        return _require_strict_int_ge(value, field_name="count field", minimum=0)

    @field_validator("metric_results", mode="before")
    @classmethod
    def _validate_metric_results_before(
        cls,
        value: object,
    ) -> list[MetricAcceptanceResult]:
        if not isinstance(value, list):
            raise ValueError(
                f"metric_results must be a list[MetricAcceptanceResult], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("metric_results", mode="after")
    @classmethod
    def _validate_metric_results(
        cls,
        value: list[MetricAcceptanceResult],
    ) -> list[MetricAcceptanceResult]:
        seen: set[str] = set()
        copied: list[MetricAcceptanceResult] = []
        for item in value:
            if not isinstance(item, MetricAcceptanceResult):
                raise ValueError(
                    "metric_results entries must be MetricAcceptanceResult, "
                    f"got {type(item).__name__}"
                )
            if item.metric_name in seen:
                raise ValueError(
                    "metric_results must not contain duplicate metric_name: "
                    f"{item.metric_name!r}"
                )
            seen.add(item.metric_name)
            copied.append(item.model_copy(deep=True))
        return copied

    @field_validator("assessed_at", mode="after")
    @classmethod
    def _validate_assessed_at(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value, field_name="assessed_at")

    @field_validator("warnings", mode="before")
    @classmethod
    def _validate_warnings_before(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(f"warnings must be a list[str], got {type(value).__name__}")
        return list(value)

    @field_validator("warnings", mode="after")
    @classmethod
    def _validate_warnings(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(value, field_name="warnings")

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> dict[str, ScalarMetadataValue]:
        if value is None:
            return {}
        return _validate_scalar_metadata(value)

    @model_validator(mode="after")
    def _validate_count_consistency(self) -> Self:
        required_results = [item for item in self.metric_results if item.required]
        if self.required_rule_count != len(required_results):
            raise ValueError(
                "required_rule_count must equal the number of required "
                f"metric_results (got {self.required_rule_count} != "
                f"{len(required_results)})"
            )

        passed = sum(
            1 for item in required_results if item.available and item.passed is True
        )
        failed = sum(
            1 for item in required_results if item.available and item.passed is False
        )
        unavailable = sum(1 for item in required_results if not item.available)

        if self.passed_required_rule_count != passed:
            raise ValueError(
                "passed_required_rule_count must match required metric_results "
                f"(got {self.passed_required_rule_count} != {passed})"
            )
        if self.failed_required_rule_count != failed:
            raise ValueError(
                "failed_required_rule_count must match required metric_results "
                f"(got {self.failed_required_rule_count} != {failed})"
            )
        if self.unavailable_required_rule_count != unavailable:
            raise ValueError(
                "unavailable_required_rule_count must match required metric_results "
                f"(got {self.unavailable_required_rule_count} != {unavailable})"
            )

        if (
            self.passed_required_rule_count
            + self.failed_required_rule_count
            + self.unavailable_required_rule_count
            != self.required_rule_count
        ):
            raise ValueError(
                "passed + failed + unavailable required counts must equal "
                "required_rule_count"
            )
        return self


@dataclass(frozen=True, slots=True)
class ModelPerformanceAcceptanceOutcome:
    """Immutable container for a completed model-performance acceptance report."""

    report: ModelPerformanceAcceptanceReport


class ModelPerformanceAssessor:
    """Assess independent-test metrics against an explicit acceptance policy.

    Uses only public fields on ``FinalEvaluationReport`` / ``FinalEvaluationOutcome``.
    Does not recompute metrics, call predict, or access estimators.
    """

    def __init__(self, *, policy: ModelPerformanceAcceptancePolicy) -> None:
        """Create an assessor with an isolated deep copy of ``policy``.

        Args:
            policy: Required acceptance policy. Deep-copied so later mutations
                do not affect this assessor.

        Raises:
            TypeError: If ``policy`` is not a ``ModelPerformanceAcceptancePolicy``.
        """
        if not isinstance(policy, ModelPerformanceAcceptancePolicy):
            raise TypeError(
                "policy must be ModelPerformanceAcceptancePolicy, "
                f"got {type(policy).__name__}"
            )
        self._policy = policy.model_copy(deep=True)

    def assess(
        self,
        final_evaluation: FinalEvaluationReport | FinalEvaluationOutcome,
    ) -> ModelPerformanceAcceptanceOutcome:
        """Compare independent test metrics to the configured acceptance policy.

        Args:
            final_evaluation: Supervised final-evaluation report or outcome.
                Validation, screening, and training metrics are never used.

        Returns:
            An outcome containing the structured acceptance report.

        Raises:
            TypeError: If ``final_evaluation`` is not a supported public type.
        """
        report = _resolve_final_evaluation_report(final_evaluation)
        policy = self._policy.model_copy(deep=True)
        assessed_at = datetime.now(tz=UTC)

        evaluation_available = True
        independent_test_evaluation = _confirm_independent_test_evaluation(report)
        test_row_count = int(report.test_row_count)
        test_metrics = dict(report.test_metrics)

        warnings: list[str] = []
        metric_results: list[MetricAcceptanceResult] = []
        for rule in policy.rules:
            metric_results.append(
                _evaluate_rule(rule=rule, test_metrics=test_metrics)
            )

        required_results = [item for item in metric_results if item.required]
        required_rule_count = len(required_results)
        passed_required = sum(
            1 for item in required_results if item.available and item.passed is True
        )
        failed_required = sum(
            1 for item in required_results if item.available and item.passed is False
        )
        unavailable_required = sum(
            1 for item in required_results if not item.available
        )

        status, status_warnings = _determine_status(
            policy=policy,
            evaluation_available=evaluation_available,
            independent_test_evaluation=independent_test_evaluation,
            test_row_count=test_row_count,
            required_rule_count=required_rule_count,
            passed_required=passed_required,
            failed_required=failed_required,
            unavailable_required=unavailable_required,
            metric_results=metric_results,
        )
        warnings.extend(status_warnings)

        acceptance_report = ModelPerformanceAcceptanceReport(
            status=status,
            task=report.task,
            evaluation_available=evaluation_available,
            independent_test_evaluation=independent_test_evaluation,
            test_row_count=test_row_count,
            metric_results=metric_results,
            required_rule_count=required_rule_count,
            passed_required_rule_count=passed_required,
            failed_required_rule_count=failed_required,
            unavailable_required_rule_count=unavailable_required,
            assessed_at=assessed_at,
            warnings=warnings,
            metadata={
                "model_name": report.model_name,
                "estimator_key": report.estimator_key,
                "primary_metric_name": report.primary_metric_name,
                "rule_count": len(policy.rules),
                "minimum_test_rows": policy.minimum_test_rows,
                "require_all_required_rules": policy.require_all_required_rules,
                "unavailable_metric_is_failure": policy.unavailable_metric_is_failure,
            },
        )
        return ModelPerformanceAcceptanceOutcome(report=acceptance_report)


def _resolve_final_evaluation_report(
    final_evaluation: object,
) -> FinalEvaluationReport:
    """Extract a ``FinalEvaluationReport`` from a supported public input type."""
    if isinstance(final_evaluation, FinalEvaluationOutcome):
        report = final_evaluation.report
        if not isinstance(report, FinalEvaluationReport):
            raise TypeError(
                "FinalEvaluationOutcome.report must be FinalEvaluationReport, "
                f"got {type(report).__name__}"
            )
        return report
    if isinstance(final_evaluation, FinalEvaluationReport):
        return final_evaluation
    raise TypeError(
        "final_evaluation must be FinalEvaluationReport or "
        "FinalEvaluationOutcome, "
        f"got {type(final_evaluation).__name__}"
    )


def _confirm_independent_test_evaluation(report: FinalEvaluationReport) -> bool:
    """Confirm independent-test evaluation via the public report contract.

    ``FinalEvaluationReport`` exposes ``test_metrics`` and ``test_row_count`` from
    a single held-out test evaluation. Both must be present and usable. This
    method never infers independence from validation or screening metrics.
    """
    if not isinstance(report.test_metrics, dict) or not report.test_metrics:
        return False
    if not isinstance(report.test_row_count, int) or isinstance(
        report.test_row_count, bool
    ):
        return False
    if report.test_row_count < 1:
        return False
    for key, raw in report.test_metrics.items():
        if not isinstance(key, str) or key == "" or key.strip() == "":
            return False
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return False
        if not math.isfinite(float(raw)):
            return False
    return True


def _evaluate_rule(
    *,
    rule: MetricAcceptanceRule,
    test_metrics: dict[str, float],
) -> MetricAcceptanceResult:
    """Evaluate one acceptance rule against public independent-test metrics."""
    if rule.metric_name not in test_metrics:
        if rule.required:
            message = (
                f"Required metric {rule.metric_name!r} is unavailable in "
                "independent test metrics."
            )
        else:
            message = (
                f"Optional metric {rule.metric_name!r} is unavailable in "
                "independent test metrics."
            )
        return MetricAcceptanceResult(
            metric_name=rule.metric_name,
            observed_value=None,
            threshold=float(rule.threshold),
            direction=rule.direction,
            required=rule.required,
            available=False,
            passed=None,
            message=message,
        )

    observed = float(test_metrics[rule.metric_name])
    if rule.direction is MetricAcceptanceDirection.HIGHER_IS_BETTER:
        passed = observed >= rule.threshold
        relation = "meets" if passed else "does not meet"
        message = (
            f"Metric {rule.metric_name}={observed} {relation} "
            f"HIGHER_IS_BETTER threshold {rule.threshold}."
        )
    else:
        passed = observed <= rule.threshold
        relation = "meets" if passed else "does not meet"
        message = (
            f"Metric {rule.metric_name}={observed} {relation} "
            f"LOWER_IS_BETTER threshold {rule.threshold}."
        )
    return MetricAcceptanceResult(
        metric_name=rule.metric_name,
        observed_value=observed,
        threshold=float(rule.threshold),
        direction=rule.direction,
        required=rule.required,
        available=True,
        passed=passed,
        message=message,
    )


def _determine_status(
    *,
    policy: ModelPerformanceAcceptancePolicy,
    evaluation_available: bool,
    independent_test_evaluation: bool,
    test_row_count: int,
    required_rule_count: int,
    passed_required: int,
    failed_required: int,
    unavailable_required: int,
    metric_results: list[MetricAcceptanceResult],
) -> tuple[ModelPerformanceAcceptanceStatus, list[str]]:
    """Derive overall acceptance status and deterministic warning messages."""
    warnings: list[str] = []

    if not evaluation_available:
        warnings.append(
            "Model performance acceptance is UNAVAILABLE because a supervised "
            "final evaluation was not available."
        )
        return ModelPerformanceAcceptanceStatus.UNAVAILABLE, warnings

    if not independent_test_evaluation:
        warnings.append(
            "Model performance acceptance is UNAVAILABLE because independent "
            "test evaluation could not be confirmed from the public final "
            "evaluation report contract."
        )
        return ModelPerformanceAcceptanceStatus.UNAVAILABLE, warnings

    if test_row_count < policy.minimum_test_rows:
        warnings.append(
            "Model performance acceptance is UNAVAILABLE because test_row_count "
            f"({test_row_count}) is below minimum_test_rows "
            f"({policy.minimum_test_rows})."
        )
        return ModelPerformanceAcceptanceStatus.UNAVAILABLE, warnings

    if required_rule_count < 1:
        warnings.append(
            "Model performance acceptance is UNAVAILABLE because the policy "
            "contains no required metric rules."
        )
        return ModelPerformanceAcceptanceStatus.UNAVAILABLE, warnings

    if unavailable_required > 0:
        missing = [
            item.metric_name
            for item in metric_results
            if item.required and not item.available
        ]
        if policy.unavailable_metric_is_failure:
            warnings.append(
                "Model performance acceptance is UNAVAILABLE because required "
                f"independent test metric(s) are missing: {', '.join(missing)}."
            )
            return ModelPerformanceAcceptanceStatus.UNAVAILABLE, warnings
        warnings.append(
            "Required independent test metric(s) are missing but "
            "unavailable_metric_is_failure=False: "
            f"{', '.join(missing)}."
        )

    if failed_required > 0:
        failed_names = [
            item.metric_name
            for item in metric_results
            if item.required and item.available and item.passed is False
        ]
        warnings.append(
            "Model performance is UNACCEPTABLE because required independent "
            f"test metric(s) failed acceptance: {', '.join(failed_names)}."
        )
        return ModelPerformanceAcceptanceStatus.UNACCEPTABLE, warnings

    if policy.require_all_required_rules:
        if (
            passed_required == required_rule_count
            and unavailable_required == 0
            and failed_required == 0
        ):
            return ModelPerformanceAcceptanceStatus.ACCEPTABLE, warnings
        warnings.append(
            "Model performance acceptance is UNAVAILABLE because not all "
            "required independent test metric rules could be confirmed as "
            "passed."
        )
        return ModelPerformanceAcceptanceStatus.UNAVAILABLE, warnings

    if passed_required >= 1 and failed_required == 0 and unavailable_required == 0:
        return ModelPerformanceAcceptanceStatus.ACCEPTABLE, warnings

    warnings.append(
        "Model performance acceptance is UNAVAILABLE because no required "
        "independent test metric rule passed under "
        "require_all_required_rules=False."
    )
    return ModelPerformanceAcceptanceStatus.UNAVAILABLE, warnings
