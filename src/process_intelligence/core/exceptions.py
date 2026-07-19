"""Domain exception hierarchy for the process intelligence platform."""


class ProcessIntelligenceError(Exception):
    """Base exception for all process intelligence domain errors."""


class DataValidationError(ProcessIntelligenceError):
    """Raised when input data fails domain validation checks."""


class DataLeakageError(ProcessIntelligenceError):
    """Raised when a potential data leakage risk is detected."""


class InsufficientDataError(ProcessIntelligenceError):
    """Raised when available data is insufficient for the requested operation."""
