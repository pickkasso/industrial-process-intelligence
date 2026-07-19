"""String-based domain enumerations for the process intelligence platform."""

from enum import StrEnum


class ColumnRole(StrEnum):
    """Role assigned to a dataset column during automatic classification."""

    IDENTIFIER = "IDENTIFIER"
    TIME = "TIME"
    CONTROLLABLE_PROCESS = "CONTROLLABLE_PROCESS"
    STATE_SENSOR = "STATE_SENSOR"
    CONTEXT = "CONTEXT"
    TARGET_QUALITY = "TARGET_QUALITY"
    DERIVED_FEATURE = "DERIVED_FEATURE"
    UNKNOWN = "UNKNOWN"


class AnalysisTask(StrEnum):
    """Supported analysis path selected by the routing layer."""

    REGRESSION = "REGRESSION"
    CLASSIFICATION = "CLASSIFICATION"
    UNSUPERVISED_ANOMALY = "UNSUPERVISED_ANOMALY"
    TIME_SERIES_ANOMALY = "TIME_SERIES_ANOMALY"
    RESIDUAL_ANOMALY = "RESIDUAL_ANOMALY"
    DRIFT_DETECTION = "DRIFT_DETECTION"


class AnomalyType(StrEnum):
    """Supported anomaly type labels for detection and reporting."""

    DATA_QUALITY = "DATA_QUALITY"
    PROCESS_INPUT = "PROCESS_INPUT"
    STATE_SENSOR = "STATE_SENSOR"
    QUALITY_OUTPUT = "QUALITY_OUTPUT"
    RELATIONSHIP = "RELATIONSHIP"
    DRIFT = "DRIFT"
    MULTIVARIATE_COMBINATION = "MULTIVARIATE_COMBINATION"
