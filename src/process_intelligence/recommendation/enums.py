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
