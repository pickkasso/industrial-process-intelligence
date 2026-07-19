"""Transparent, configurable Data Quality Score from ValidationIssue lists."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator

from process_intelligence.core.exceptions import DataValidationError
from process_intelligence.core.schemas import ValidationIssue

_REQUIRED_SEVERITIES: tuple[str, ...] = ("ERROR", "WARNING", "INFO")
_DEFAULT_DIMENSIONS: tuple[str, ...] = (
    "COMPLETENESS",
    "UNIQUENESS",
    "VALIDITY",
    "VARIABILITY",
    "OTHER",
)
_DEFAULT_MULTIPLIER = 1.0
_DEFAULT_MODELING_IMPACT = (
    "Unknown impact on downstream modeling; review required."
)
_STARTING_SCORE = 100.0


def _freeze_str_float_mapping(mapping: Mapping[str, float]) -> MappingProxyType[str, float]:
    return MappingProxyType(dict(mapping))


def _freeze_str_str_mapping(mapping: Mapping[str, str]) -> MappingProxyType[str, str]:
    return MappingProxyType(dict(mapping))


DEFAULT_SEVERITY_WEIGHTS: Mapping[str, float] = _freeze_str_float_mapping(
    {
        "ERROR": 20.0,
        "WARNING": 8.0,
        "INFO": 2.0,
    }
)

DEFAULT_ISSUE_TYPE_MULTIPLIERS: Mapping[str, float] = _freeze_str_float_mapping(
    {
        "EMPTY_DATASET": 5.0,
        "DUPLICATE_ROWS": 1.0,
        "ALL_NULL_COLUMN": 1.5,
        "HIGH_MISSING_RATIO": 1.0,
        "CONSTANT_COLUMN": 0.75,
        "NEAR_CONSTANT_COLUMN": 0.5,
        "NAN_VALUES": 1.0,
        "INFINITE_VALUES": 1.5,
        "NUMERIC_STRING_COLUMN": 0.75,
    }
)

ISSUE_TYPE_DIMENSIONS: Mapping[str, str] = _freeze_str_str_mapping(
    {
        "EMPTY_DATASET": "COMPLETENESS",
        "ALL_NULL_COLUMN": "COMPLETENESS",
        "HIGH_MISSING_RATIO": "COMPLETENESS",
        "DUPLICATE_ROWS": "UNIQUENESS",
        "NAN_VALUES": "VALIDITY",
        "INFINITE_VALUES": "VALIDITY",
        "NUMERIC_STRING_COLUMN": "VALIDITY",
        "CONSTANT_COLUMN": "VARIABILITY",
        "NEAR_CONSTANT_COLUMN": "VARIABILITY",
    }
)

ISSUE_TYPE_MODELING_IMPACTS: Mapping[str, str] = _freeze_str_str_mapping(
    {
        "EMPTY_DATASET": (
            "Empty datasets cannot be used for model training or evaluation."
        ),
        "DUPLICATE_ROWS": (
            "Duplicate rows can bias training and inflate performance estimates."
        ),
        "ALL_NULL_COLUMN": (
            "All-null columns provide no predictive signal and should be removed."
        ),
        "HIGH_MISSING_RATIO": (
            "High missingness reduces effective sample size and can distort estimates."
        ),
        "CONSTANT_COLUMN": (
            "Constant columns add no variance and are typically uninformative features."
        ),
        "NEAR_CONSTANT_COLUMN": (
            "Near-constant columns contribute little variance and may harm stability."
        ),
        "NAN_VALUES": (
            "NaN values can break numeric estimators or require special handling."
        ),
        "INFINITE_VALUES": (
            "Infinite values are invalid for most estimators and must be corrected."
        ),
        "NUMERIC_STRING_COLUMN": (
            "Numeric-like strings may be mistyped and need casting before modeling."
        ),
    }
)


def _default_severity_weights() -> dict[str, float]:
    return dict(DEFAULT_SEVERITY_WEIGHTS)


def _default_issue_type_multipliers() -> dict[str, float]:
    return dict(DEFAULT_ISSUE_TYPE_MULTIPLIERS)


def _validate_non_negative_finite_weight(value: object, *, field_name: str) -> float:
    """Reject bools, non-numbers, negatives, NaN, and infinities."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a non-negative finite number "
            f"(bool not allowed), got {value!r}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(
            f"{field_name} must be a non-negative finite number, got {value!r}"
        )
    if number < 0.0:
        raise ValueError(f"{field_name} must be >= 0, got {number}")
    return number


class DataQualityWeights(BaseModel):
    """Configurable severity weights and issue-type multipliers for scoring."""

    severity_weights: dict[str, float] = Field(default_factory=_default_severity_weights)
    issue_type_multipliers: dict[str, float] = Field(
        default_factory=_default_issue_type_multipliers
    )

    @field_validator("severity_weights", mode="before")
    @classmethod
    def _copy_and_validate_severity_weights(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value

        normalized_severity: dict[str, float] = {}
        for key, raw_weight in value.items():
            upper_key = str(key).upper()
            normalized_severity[upper_key] = _validate_non_negative_finite_weight(
                raw_weight,
                field_name=f"severity_weights[{key!r}]",
            )

        missing = [
            severity
            for severity in _REQUIRED_SEVERITIES
            if severity not in normalized_severity
        ]
        if missing:
            raise ValueError(
                "severity_weights must include ERROR, WARNING, and INFO "
                f"(compared case-insensitively); missing: {missing}"
            )
        return normalized_severity

    @field_validator("issue_type_multipliers", mode="before")
    @classmethod
    def _copy_and_validate_issue_type_multipliers(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value

        validated_multipliers: dict[str, float] = {}
        for key, raw_multiplier in value.items():
            validated_multipliers[str(key)] = _validate_non_negative_finite_weight(
                raw_multiplier,
                field_name=f"issue_type_multipliers[{key!r}]",
            )
        return validated_multipliers


class QualityPenalty(BaseModel):
    """Per-issue deduction detail contributing to a Data Quality Score."""

    issue_type: str
    column: str | None = None
    severity: str
    dimension: str
    base_weight: float = Field(ge=0.0)
    multiplier: float = Field(ge=0.0)
    deducted_points: float = Field(ge=0.0)
    reason: str
    modeling_impact: str
    recommended_action: str | None = None


class DataQualityScore(BaseModel):
    """Aggregated Data Quality Score with transparent penalty breakdown."""

    total_score: float = Field(ge=0.0, le=100.0)
    starting_score: float
    total_deduction: float = Field(ge=0.0)
    issue_count: int = Field(ge=0)
    penalties: list[QualityPenalty] = Field(default_factory=list)
    severity_counts: dict[str, int] = Field(default_factory=dict)
    issue_type_counts: dict[str, int] = Field(default_factory=dict)
    dimension_deductions: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_starting_score(self) -> Self:
        if self.starting_score != _STARTING_SCORE:
            raise ValueError(f"starting_score must be exactly {_STARTING_SCORE}")
        return self


class DataQualityScorer:
    """Compute a transparent 0–100 Data Quality Score from validation issues."""

    def __init__(self, weights: DataQualityWeights | None = None) -> None:
        """Create a scorer with optional custom weights.

        Args:
            weights: Scoring weights. When ``None``, a fresh default
                ``DataQualityWeights`` instance is created.
        """
        self._weights = DataQualityWeights() if weights is None else weights

    def score(self, issues: Sequence[ValidationIssue]) -> DataQualityScore:
        """Score a sequence of validation issues without mutating inputs.

        Args:
            issues: Validation findings in input order. ``str`` and ``bytes``
                are rejected even though they are sequences.

        Returns:
            A ``DataQualityScore`` with penalties, counts, and dimension
            deductions. Results are not cached on the scorer instance.

        Raises:
            TypeError: If ``issues`` is ``str``/``bytes`` or contains a
                non-``ValidationIssue`` element.
            DataValidationError: If an issue has an unknown severity.
        """
        if isinstance(issues, (str, bytes)):
            raise TypeError(
                "issues must be a Sequence of ValidationIssue, "
                f"got {type(issues).__name__}"
            )
        if not isinstance(issues, Sequence):
            raise TypeError(
                "issues must be a Sequence of ValidationIssue, "
                f"got {type(issues).__name__}"
            )

        severity_weights = dict(self._weights.severity_weights)
        multipliers = dict(self._weights.issue_type_multipliers)

        penalties: list[QualityPenalty] = []
        severity_counts: dict[str, int] = {name: 0 for name in _REQUIRED_SEVERITIES}
        issue_type_counts: dict[str, int] = {}
        dimension_deductions: dict[str, float] = {
            name: 0.0 for name in _DEFAULT_DIMENSIONS
        }
        raw_total_deduction = 0.0

        for index, issue in enumerate(issues):
            if not isinstance(issue, ValidationIssue):
                raise TypeError(
                    "Each element of issues must be a ValidationIssue, "
                    f"got {type(issue).__name__} at index {index}"
                )

            severity = issue.severity.upper()
            if severity not in _REQUIRED_SEVERITIES:
                raise DataValidationError(
                    f"Unknown severity {issue.severity!r}; "
                    "expected ERROR, WARNING, or INFO"
                )

            base_weight = severity_weights[severity]
            multiplier = multipliers.get(issue.issue_type, _DEFAULT_MULTIPLIER)
            deducted_points = round(base_weight * multiplier, 2)
            dimension = ISSUE_TYPE_DIMENSIONS.get(issue.issue_type, "OTHER")
            modeling_impact = ISSUE_TYPE_MODELING_IMPACTS.get(
                issue.issue_type,
                _DEFAULT_MODELING_IMPACT,
            )

            penalties.append(
                QualityPenalty(
                    issue_type=issue.issue_type,
                    column=issue.column,
                    severity=severity,
                    dimension=dimension,
                    base_weight=base_weight,
                    multiplier=multiplier,
                    deducted_points=deducted_points,
                    reason=issue.message,
                    modeling_impact=modeling_impact,
                    recommended_action=issue.suggested_action,
                )
            )

            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            if issue.issue_type in issue_type_counts:
                issue_type_counts[issue.issue_type] += 1
            else:
                issue_type_counts[issue.issue_type] = 1
            dimension_deductions[dimension] = (
                dimension_deductions.get(dimension, 0.0) + deducted_points
            )
            raw_total_deduction += deducted_points

        total_deduction = round(raw_total_deduction, 2)
        total_score = round(max(0.0, _STARTING_SCORE - total_deduction), 2)
        rounded_dimensions = {
            name: round(value, 2) for name, value in dimension_deductions.items()
        }

        return DataQualityScore(
            total_score=total_score,
            starting_score=_STARTING_SCORE,
            total_deduction=total_deduction,
            issue_count=len(penalties),
            penalties=penalties,
            severity_counts=severity_counts,
            issue_type_counts=issue_type_counts,
            dimension_deductions=rounded_dimensions,
        )
