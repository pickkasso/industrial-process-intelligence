"""Core domain types for the process intelligence platform."""

from process_intelligence.core.enums import AnalysisTask, AnomalyType, ColumnRole
from process_intelligence.core.exceptions import (
    DataLeakageError,
    DataValidationError,
    InsufficientDataError,
    ProcessIntelligenceError,
)
from process_intelligence.core.protocols import (
    BaseAnalysisModel,
    BaseAnomalyModel,
    BaseIndustryProfile,
    DataFrameLike,
    SeriesLike,
)
from process_intelligence.core.schemas import (
    AnomalyEvent,
    ColumnRoleAssignment,
    DatasetMetadata,
    ExplanationResult,
    IndustryScore,
    ModelEvaluation,
    ModelMetadata,
    ModelSpec,
    PreprocessingEvent,
    PreprocessingRules,
    RootCauseFactor,
    SchemaHints,
    ValidationIssue,
    VariableConstraint,
)

__all__ = [
    "AnalysisTask",
    "AnomalyEvent",
    "AnomalyType",
    "BaseAnalysisModel",
    "BaseAnomalyModel",
    "BaseIndustryProfile",
    "ColumnRole",
    "ColumnRoleAssignment",
    "DataFrameLike",
    "DataLeakageError",
    "DataValidationError",
    "DatasetMetadata",
    "ExplanationResult",
    "IndustryScore",
    "InsufficientDataError",
    "ModelEvaluation",
    "ModelMetadata",
    "ModelSpec",
    "PreprocessingEvent",
    "PreprocessingRules",
    "ProcessIntelligenceError",
    "RootCauseFactor",
    "SchemaHints",
    "SeriesLike",
    "ValidationIssue",
    "VariableConstraint",
]
