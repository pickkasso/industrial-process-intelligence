"""Core Pydantic schemas for industry profiles, models, and analysis results."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from process_intelligence.core.enums import AnalysisTask, AnomalyType, ColumnRole


class PreprocessingEvent(BaseModel):
    """Structured log entry for a single preprocessing step."""

    step_name: str
    affected_columns: list[str]
    rows_before: int = Field(ge=0)
    rows_after: int = Field(ge=0)
    parameters: dict[str, Any]
    warnings: list[str]
    timestamp: datetime


class DatasetMetadata(BaseModel):
    """Minimal dataset metadata used as input to industry scoring."""

    file_name: str
    file_format: str
    row_count: int = Field(ge=0)
    column_names: list[str]
    dtypes: dict[str, str]
    user_description: str | None = None


class IndustryScore(BaseModel):
    """Industry estimation result with confidence and supporting evidence."""

    industry_name: str
    score: float
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str]
    uncertain_factors: list[str]
    requires_user_confirmation: bool


class SchemaHints(BaseModel):
    """Schema hints supplied by an industry profile."""

    keywords: list[str] = Field(default_factory=list)
    synonyms: dict[str, list[str]] = Field(default_factory=dict)
    expected_units: dict[str, str] = Field(default_factory=dict)
    default_roles: dict[str, ColumnRole] = Field(default_factory=dict)
    physical_ranges: dict[str, tuple[float | None, float | None]] = Field(default_factory=dict)


class PreprocessingRules(BaseModel):
    """Default preprocessing rules provided by an industry profile."""

    imputation: dict[str, str] = Field(default_factory=dict)
    scaling: dict[str, str] = Field(default_factory=dict)
    outlier_handling: dict[str, str] = Field(default_factory=dict)
    industry_specific: dict[str, Any] = Field(default_factory=dict)


class ColumnRoleAssignment(BaseModel):
    """Automatic column-role classification result for a single column."""

    column_name: str
    role: ColumnRole
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str]
    alternative_roles: list[ColumnRole]
    use_in_model: bool


class ValidationIssue(BaseModel):
    """Minimal validation finding returned by industry-profile checks."""

    issue_type: str
    column: str | None = None
    severity: str
    message: str
    suggested_action: str | None = None


class ModelSpec(BaseModel):
    """Minimal model specification used by profiles and the model registry."""

    name: str
    task: AnalysisTask
    estimator_key: str
    optional_dependencies: list[str] = Field(default_factory=list)
    priority: int = Field(ge=0)
    time_budget_seconds: float | None = Field(default=None, gt=0.0)


class AnomalyEvent(BaseModel):
    """Structured anomaly event produced by detectors and classifiers."""

    anomaly_id: str
    anomaly_type: AnomalyType
    anomaly_score: float
    severity: str
    sample_id: str | int | None = None
    timestamp: datetime | None = None
    group_id: str | int | None = None
    deviation: float | None = None
    contributing_variables: list[str] = Field(default_factory=list)
    model_confidence: float = Field(ge=0.0, le=1.0)
    detector: str
    rationale: str


class RootCauseFactor(BaseModel):
    """Single factor in a root-cause diagnosis result."""

    variable: str
    direction: str
    deviation: float | None = None
    role: ColumnRole
    controllable: bool
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)
    needs_verification: bool


class VariableConstraint(BaseModel):
    """Per-variable constraint used by the recommendation engine."""

    variable: str
    adjustable: bool
    minimum: float | None = None
    maximum: float | None = None
    max_change_ratio: float | None = Field(default=None, ge=0.0)
    unit: str | None = None
    cost: float | None = Field(default=None, ge=0.0)
    safety_constraints: list[str] = Field(default_factory=list)
    equipment_operating_range: tuple[float | None, float | None] | None = None
    max_simultaneous_changes: int | None = Field(default=None, ge=1)
    fixed: bool

    @model_validator(mode="after")
    def validate_range_bounds(self) -> Self:
        """Reject inverted numeric bounds when both ends are provided."""
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum cannot be greater than maximum")

        if self.equipment_operating_range is not None:
            lower, upper = self.equipment_operating_range
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(
                    "equipment_operating_range lower bound cannot exceed upper bound"
                )

        return self


class ModelMetadata(BaseModel):
    """Traceable metadata returned by analysis models."""

    model_name: str
    version: str
    task: AnalysisTask
    features: list[str] = Field(default_factory=list)
    training_timestamp: datetime | None = None
    seed: int | None = None
    optional_dependencies_used: list[str] = Field(default_factory=list)


class ModelEvaluation(BaseModel):
    """Evaluation metrics and notes returned by analysis models."""

    metrics: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ExplanationResult(BaseModel):
    """Feature-importance explanation returned by analysis models."""

    method: str
    feature_importances: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
