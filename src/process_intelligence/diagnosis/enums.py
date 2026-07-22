"""Diagnosis-layer enumerations for root-cause analysis contracts."""

from enum import StrEnum


class DiagnosisMethod(StrEnum):
    """Attribution or association method used for root-cause diagnosis.

    Values identify how variable contributions were estimated. They describe
    associative evidence relative to anomaly scores or group differences, not
    proven causal mechanisms.
    """

    ROBUST_Z_SCORE = "ROBUST_Z_SCORE"
    GROUP_COMPARISON = "GROUP_COMPARISON"
    MODEL_EXPLANATION = "MODEL_EXPLANATION"
    RESIDUAL_ASSOCIATION = "RESIDUAL_ASSOCIATION"
    DISTRIBUTION_SHIFT = "DISTRIBUTION_SHIFT"
    RULE_BASED = "RULE_BASED"
    ENSEMBLE = "ENSEMBLE"


class DiagnosisScope(StrEnum):
    """Scope of anomaly events covered by a diagnosis request or result.

    Controls whether diagnosis targets a single event, a top-N score slice,
    a named anomaly group, or a global/batch aggregation over events.
    """

    SINGLE_EVENT = "SINGLE_EVENT"
    TOP_ANOMALIES = "TOP_ANOMALIES"
    ANOMALY_GROUP = "ANOMALY_GROUP"
    GLOBAL = "GLOBAL"


class RobustScaleStatus(StrEnum):
    """Availability of the reference-group robust scale used for standardization.

    ``AVAILABLE`` means a positive robust scale was observed and a finite
    robust z-score may be reported. ``ZERO_VARIANCE`` means the reference
    robust scale is at or below the configured numerical floor, so a
    standardized score is undefined. ``INSUFFICIENT_DATA`` reserves the case
    where scale cannot be estimated from the available reference values.
    """

    AVAILABLE = "AVAILABLE"
    ZERO_VARIANCE = "ZERO_VARIANCE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
