"""Recommendation-layer enumerations for objectives, safety, and reason codes."""

from enum import StrEnum


class RecommendationObjective(StrEnum):
    """Optimization objective for actionable process recommendations.

    Selects whether recommendation search prioritizes anomaly-score reduction,
    predicted quality improvement, or a balance of both. Objectives describe
    decision-support goals, not guaranteed process outcomes.
    """

    REDUCE_ANOMALY_SCORE = "REDUCE_ANOMALY_SCORE"
    IMPROVE_PREDICTED_QUALITY = "IMPROVE_PREDICTED_QUALITY"
    BALANCE_QUALITY_AND_ANOMALY = "BALANCE_QUALITY_AND_ANOMALY"


class RecommendationSafetyStatus(StrEnum):
    """Outcome of recommendation safety-gate evaluation.

    ``APPROVED`` means all checked variables are eligible without caution
    signals. ``CAUTION`` allows optimization with partial eligibility or
    user overrides. ``REFUSED`` blocks strong recommendations.
    """

    APPROVED = "APPROVED"
    CAUTION = "CAUTION"
    REFUSED = "REFUSED"


class RecommendationStatus(StrEnum):
    """Lifecycle status of a recommendation result.

    Safety Gate (Step 9A) may produce ``REFUSED`` or
    ``READY_FOR_OPTIMIZATION``. ``GENERATED`` is reserved for later
    optimizer steps that emit concrete proposed values.
    """

    REFUSED = "REFUSED"
    READY_FOR_OPTIMIZATION = "READY_FOR_OPTIMIZATION"
    GENERATED = "GENERATED"


class RecommendationReasonCode(StrEnum):
    """Deterministic reason codes for recommendation refusal or caution.

    Codes identify structured safety or eligibility findings. They describe
    association-aware decision support limits, not proven causal mechanisms.
    """

    LEAKAGE_BLOCKER = "LEAKAGE_BLOCKER"
    FINAL_EVALUATION_MISSING = "FINAL_EVALUATION_MISSING"
    MODEL_PERFORMANCE_UNACCEPTABLE = "MODEL_PERFORMANCE_UNACCEPTABLE"
    DIAGNOSIS_CONFIDENCE_TOO_LOW = "DIAGNOSIS_CONFIDENCE_TOO_LOW"
    REFERENCE_SAMPLE_TOO_SMALL = "REFERENCE_SAMPLE_TOO_SMALL"
    EXTRAPOLATION_RISK = "EXTRAPOLATION_RISK"
    UNCERTAINTY_UNAVAILABLE = "UNCERTAINTY_UNAVAILABLE"
    UNCERTAINTY_UNACCEPTABLE = "UNCERTAINTY_UNACCEPTABLE"
    CURRENT_VALUE_MISSING = "CURRENT_VALUE_MISSING"
    CONSTRAINT_MISSING = "CONSTRAINT_MISSING"
    NON_CONTROLLABLE_VARIABLE = "NON_CONTROLLABLE_VARIABLE"
    USER_CONFIRMATION_REQUIRED = "USER_CONFIRMATION_REQUIRED"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
    CURRENT_VALUE_OUTSIDE_CONSTRAINT = "CURRENT_VALUE_OUTSIDE_CONSTRAINT"
    CANDIDATE_LIMIT_EXCEEDED = "CANDIDATE_LIMIT_EXCEEDED"
    NO_ELIGIBLE_VARIABLES = "NO_ELIGIBLE_VARIABLES"
    PARTIAL_ELIGIBILITY = "PARTIAL_ELIGIBILITY"


class WhatIfVerificationScenarioType(StrEnum):
    """Scenario role inside local recommendation what-if verification.

    ``BASELINE`` is the unchanged operating point. ``PROPOSED_CENTER`` is the
    generated recommendation. Neighbor types are one-factor-at-a-time adjacent
    constraint-grid points around the proposed center.
    """

    BASELINE = "BASELINE"
    PROPOSED_CENTER = "PROPOSED_CENTER"
    LOWER_NEIGHBOR = "LOWER_NEIGHBOR"
    UPPER_NEIGHBOR = "UPPER_NEIGHBOR"


class WhatIfPerturbationDirection(StrEnum):
    """Direction of an adjacent-grid neighbor relative to the proposed value."""

    LOWER = "LOWER"
    UPPER = "UPPER"


class WhatIfVerificationStatus(StrEnum):
    """Outcome status for recommendation local what-if verification.

    ``COMPLETED`` means local neighbor scenarios were scored. ``NOT_APPLICABLE``
    means verification was intentionally skipped because no generated
    recommendation was available. ``UNAVAILABLE`` means required inputs could
    not be assembled or scoring failed structurally.
    """

    COMPLETED = "COMPLETED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNAVAILABLE = "UNAVAILABLE"


class TargetPredictionPlausibilityStatus(StrEnum):
    """Plausibility of a supervised recommendation target prediction.

    Statuses describe whether a raw model prediction lies inside the observed
    training-target range and an optional caller-declared semantic domain.
    They do not clip predictions or assert physical attainability.
    """

    WITHIN_OBSERVED_RANGE = "WITHIN_OBSERVED_RANGE"
    OUTSIDE_OBSERVED_RANGE = "OUTSIDE_OBSERVED_RANGE"
    OUTSIDE_DECLARED_DOMAIN = "OUTSIDE_DECLARED_DOMAIN"
    DOMAIN_UNAVAILABLE = "DOMAIN_UNAVAILABLE"


class WhatIfStabilityClassification(StrEnum):
    """Deterministic local stability label for a generated recommendation.

    Labels describe model-local adjacent-grid behavior only. They do not assert
    physical robustness, process safety, causation, or deployment readiness.
    """

    STABLE = "STABLE"
    MIXED = "MIXED"
    ISOLATED = "ISOLATED"
    NOT_IMPROVING = "NOT_IMPROVING"
    NO_NEIGHBORS = "NO_NEIGHBORS"
    UNAVAILABLE = "UNAVAILABLE"
