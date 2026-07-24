"""Target-prediction domain and plausibility evaluation for recommendations.

Derives observed training-target ranges and optional caller-declared semantic
bounds, then classifies raw model predictions without clipping values.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator

from process_intelligence.recommendation.enums import TargetPredictionPlausibilityStatus

_WARNING_OUTSIDE_DECLARED = (
    "Raw predicted target is outside the declared semantic target domain and "
    "must not be interpreted as a quantitatively attainable quality value."
)
_WARNING_OUTSIDE_OBSERVED = (
    "Raw predicted target is outside the observed training-target range "
    "(model extrapolation). This does not by itself mark the proposal as an "
    "unsafe engineering change."
)
_WARNING_DOMAIN_UNAVAILABLE = (
    "Target prediction domain was unavailable; plausibility could not be assessed."
)


def _require_finite_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a finite float, got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be a finite float, got {number}")
    return number


def _require_optional_finite_float(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    return _require_finite_float(value, field_name=field_name)


class TargetPredictionDomain(BaseModel):
    """Observed training range and optional declared semantic target bounds.

    ``observed_*`` come from the supervised training partition. ``declared_*``
    are optional caller-supplied semantic limits (for example a demo quality
    score domain). Ordinary uploaded datasets do not receive declared bounds
    unless the caller supplies them explicitly.
    """

    observed_minimum: float | None = None
    observed_maximum: float | None = None
    declared_minimum: float | None = None
    declared_maximum: float | None = None

    @field_validator(
        "observed_minimum",
        "observed_maximum",
        "declared_minimum",
        "declared_maximum",
        mode="before",
    )
    @classmethod
    def _validate_optional_bounds(cls, value: object) -> float | None:
        return _require_optional_finite_float(value, field_name="domain bound")

    @model_validator(mode="after")
    def _validate_bound_pairs(self) -> Self:
        if (self.observed_minimum is None) ^ (self.observed_maximum is None):
            raise ValueError(
                "observed_minimum and observed_maximum must both be set or both None"
            )
        if (
            self.observed_minimum is not None
            and self.observed_maximum is not None
            and self.observed_minimum > self.observed_maximum
        ):
            raise ValueError(
                "observed_minimum must be <= observed_maximum "
                f"(got {self.observed_minimum} > {self.observed_maximum})"
            )
        if (self.declared_minimum is None) ^ (self.declared_maximum is None):
            raise ValueError(
                "declared_minimum and declared_maximum must both be set or both None"
            )
        if (
            self.declared_minimum is not None
            and self.declared_maximum is not None
            and self.declared_minimum > self.declared_maximum
        ):
            raise ValueError(
                "declared_minimum must be <= declared_maximum "
                f"(got {self.declared_minimum} > {self.declared_maximum})"
            )
        return self

    @property
    def has_observed_range(self) -> bool:
        """Return True when an observed training range is present."""
        return self.observed_minimum is not None and self.observed_maximum is not None

    @property
    def has_declared_domain(self) -> bool:
        """Return True when a declared semantic domain is present."""
        return self.declared_minimum is not None and self.declared_maximum is not None


class TargetPredictionPlausibilityAssessment(BaseModel):
    """Classification of one raw target prediction against a domain contract."""

    raw_prediction: float
    status: TargetPredictionPlausibilityStatus
    warning_message: str | None = None

    @field_validator("raw_prediction", mode="before")
    @classmethod
    def _validate_raw_prediction(cls, value: object) -> float:
        return _require_finite_float(value, field_name="raw_prediction")

    @field_validator("status", mode="before")
    @classmethod
    def _validate_status(cls, value: object) -> TargetPredictionPlausibilityStatus:
        if isinstance(value, TargetPredictionPlausibilityStatus):
            return value
        if isinstance(value, str):
            try:
                return TargetPredictionPlausibilityStatus(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid TargetPredictionPlausibilityStatus: {value!r}"
                ) from exc
        raise ValueError(
            "status must be TargetPredictionPlausibilityStatus, "
            f"got {type(value).__name__}"
        )

    @field_validator("warning_message", mode="before")
    @classmethod
    def _validate_warning_message(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(
                f"warning_message must be str or None, got {type(value).__name__}"
            )
        if value == "" or value.strip() == "":
            raise ValueError("warning_message must be non-empty when provided")
        return value


class RecommendationTargetPlausibility(BaseModel):
    """Paired baseline/proposed plausibility for a supervised recommendation."""

    domain: TargetPredictionDomain | None = None
    baseline_raw_prediction: float | None = None
    proposed_raw_prediction: float | None = None
    baseline_status: TargetPredictionPlausibilityStatus | None = None
    proposed_status: TargetPredictionPlausibilityStatus | None = None
    warning_messages: list[str] = Field(default_factory=list)

    @field_validator("domain", mode="before")
    @classmethod
    def _validate_domain(cls, value: object) -> TargetPredictionDomain | None:
        if value is None:
            return None
        if isinstance(value, TargetPredictionDomain):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return TargetPredictionDomain.model_validate(value)
        raise ValueError(
            "domain must be TargetPredictionDomain or None, "
            f"got {type(value).__name__}"
        )

    @field_validator(
        "baseline_raw_prediction",
        "proposed_raw_prediction",
        mode="before",
    )
    @classmethod
    def _validate_optional_predictions(cls, value: object) -> float | None:
        return _require_optional_finite_float(value, field_name="raw prediction")

    @field_validator("baseline_status", "proposed_status", mode="before")
    @classmethod
    def _validate_optional_status(
        cls,
        value: object,
    ) -> TargetPredictionPlausibilityStatus | None:
        if value is None:
            return None
        if isinstance(value, TargetPredictionPlausibilityStatus):
            return value
        if isinstance(value, str):
            try:
                return TargetPredictionPlausibilityStatus(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid TargetPredictionPlausibilityStatus: {value!r}"
                ) from exc
        raise ValueError(
            "plausibility status must be TargetPredictionPlausibilityStatus "
            f"or None, got {type(value).__name__}"
        )

    @field_validator("warning_messages", mode="before")
    @classmethod
    def _validate_warning_messages_before(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"warning_messages must be a list[str], got {type(value).__name__}"
            )
        return list(value)

    @field_validator("warning_messages", mode="after")
    @classmethod
    def _validate_warning_messages(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                raise ValueError(
                    "warning_messages entries must be str, "
                    f"got {type(item).__name__}"
                )
            if item == "" or item.strip() == "":
                raise ValueError("warning_messages entries must be non-empty")
            if item in seen:
                continue
            seen.add(item)
            cleaned.append(item)
        return cleaned


def build_target_prediction_domain(
    training_targets: Sequence[float] | None,
    *,
    declared_minimum: float | None = None,
    declared_maximum: float | None = None,
) -> TargetPredictionDomain | None:
    """Build a domain contract from training targets and optional declared bounds.

    Returns ``None`` when neither an observed range nor a declared domain can
    be formed. Does not infer universal quality bounds.
    """
    observed_minimum: float | None = None
    observed_maximum: float | None = None
    if training_targets is not None:
        finite_values = [
            _require_finite_float(value, field_name="training target")
            for value in training_targets
        ]
        if finite_values:
            observed_minimum = float(min(finite_values))
            observed_maximum = float(max(finite_values))

    declared_min = _require_optional_finite_float(
        declared_minimum,
        field_name="declared_minimum",
    )
    declared_max = _require_optional_finite_float(
        declared_maximum,
        field_name="declared_maximum",
    )
    if (declared_min is None) ^ (declared_max is None):
        raise ValueError(
            "declared_minimum and declared_maximum must both be set or both None"
        )
    if declared_min is not None and declared_max is not None and declared_min > declared_max:
        raise ValueError(
            "declared_minimum must be <= declared_maximum "
            f"(got {declared_min} > {declared_max})"
        )

    if (
        observed_minimum is None
        and observed_maximum is None
        and declared_min is None
        and declared_max is None
    ):
        return None
    return TargetPredictionDomain(
        observed_minimum=observed_minimum,
        observed_maximum=observed_maximum,
        declared_minimum=declared_min,
        declared_maximum=declared_max,
    )


def evaluate_target_prediction_plausibility(
    prediction: float,
    *,
    domain: TargetPredictionDomain | None,
) -> TargetPredictionPlausibilityAssessment:
    """Classify one finite raw prediction against the supplied domain.

    Raises:
        ValueError: If ``prediction`` is non-finite.
    """
    raw = _require_finite_float(prediction, field_name="prediction")
    if domain is None or (
        not domain.has_observed_range and not domain.has_declared_domain
    ):
        return TargetPredictionPlausibilityAssessment(
            raw_prediction=raw,
            status=TargetPredictionPlausibilityStatus.DOMAIN_UNAVAILABLE,
            warning_message=_WARNING_DOMAIN_UNAVAILABLE,
        )

    if domain.has_declared_domain:
        assert domain.declared_minimum is not None
        assert domain.declared_maximum is not None
        declared_min = float(domain.declared_minimum)
        declared_max = float(domain.declared_maximum)
        if raw < declared_min or raw > declared_max:
            return TargetPredictionPlausibilityAssessment(
                raw_prediction=raw,
                status=TargetPredictionPlausibilityStatus.OUTSIDE_DECLARED_DOMAIN,
                warning_message=_WARNING_OUTSIDE_DECLARED,
            )

    if domain.has_observed_range:
        assert domain.observed_minimum is not None
        assert domain.observed_maximum is not None
        observed_min = float(domain.observed_minimum)
        observed_max = float(domain.observed_maximum)
        if raw < observed_min or raw > observed_max:
            return TargetPredictionPlausibilityAssessment(
                raw_prediction=raw,
                status=TargetPredictionPlausibilityStatus.OUTSIDE_OBSERVED_RANGE,
                warning_message=_WARNING_OUTSIDE_OBSERVED,
            )
        return TargetPredictionPlausibilityAssessment(
            raw_prediction=raw,
            status=TargetPredictionPlausibilityStatus.WITHIN_OBSERVED_RANGE,
            warning_message=None,
        )

    # Declared domain only, and prediction is inside it.
    return TargetPredictionPlausibilityAssessment(
        raw_prediction=raw,
        status=TargetPredictionPlausibilityStatus.WITHIN_OBSERVED_RANGE,
        warning_message=None,
    )


def assess_recommendation_target_plausibility(
    *,
    baseline_prediction: float | None,
    proposed_prediction: float | None,
    domain: TargetPredictionDomain | None,
) -> RecommendationTargetPlausibility:
    """Evaluate baseline and proposed raw predictions against a domain contract."""
    warnings: list[str] = []
    baseline_status: TargetPredictionPlausibilityStatus | None = None
    proposed_status: TargetPredictionPlausibilityStatus | None = None
    baseline_raw = (
        None
        if baseline_prediction is None
        else _require_finite_float(baseline_prediction, field_name="baseline_prediction")
    )
    proposed_raw = (
        None
        if proposed_prediction is None
        else _require_finite_float(proposed_prediction, field_name="proposed_prediction")
    )

    if baseline_raw is not None:
        baseline_assessment = evaluate_target_prediction_plausibility(
            baseline_raw,
            domain=domain,
        )
        baseline_status = baseline_assessment.status
        if baseline_assessment.warning_message is not None:
            warnings.append(baseline_assessment.warning_message)

    if proposed_raw is not None:
        proposed_assessment = evaluate_target_prediction_plausibility(
            proposed_raw,
            domain=domain,
        )
        proposed_status = proposed_assessment.status
        if (
            proposed_assessment.warning_message is not None
            and proposed_assessment.warning_message not in warnings
        ):
            warnings.append(proposed_assessment.warning_message)

    return RecommendationTargetPlausibility(
        domain=None if domain is None else domain.model_copy(deep=True),
        baseline_raw_prediction=baseline_raw,
        proposed_raw_prediction=proposed_raw,
        baseline_status=baseline_status,
        proposed_status=proposed_status,
        warning_messages=warnings,
    )
