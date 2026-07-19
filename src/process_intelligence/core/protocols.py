"""Abstract base interfaces for industry profiles and analysis models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self, TypeAlias

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.schemas import (
    AnomalyEvent,
    DatasetMetadata,
    ExplanationResult,
    IndustryScore,
    ModelEvaluation,
    ModelMetadata,
    ModelSpec,
    PreprocessingRules,
    SchemaHints,
    ValidationIssue,
    VariableConstraint,
)

DataFrameLike: TypeAlias = pl.DataFrame | pd.DataFrame
"""Union of supported tabular frame types (Polars or Pandas)."""

SeriesLike: TypeAlias = pl.Series | pd.Series
"""Union of supported series types (Polars or Pandas)."""


class BaseIndustryProfile(ABC):
    """Plugin interface for industry-specific scoring, hints, and constraints."""

    @property
    @abstractmethod
    def industry_name(self) -> str:
        """Canonical name of the industry represented by this profile."""

    @abstractmethod
    def score_industry(self, metadata: DatasetMetadata) -> IndustryScore:
        """Estimate how well the dataset matches this industry profile."""

    @abstractmethod
    def get_schema_hints(self) -> SchemaHints:
        """Return column naming, unit, role, and range hints for this industry."""

    @abstractmethod
    def get_preprocessing_rules(self) -> PreprocessingRules:
        """Return default preprocessing rules for this industry."""

    @abstractmethod
    def get_default_model_candidates(self, task: AnalysisTask) -> list[ModelSpec]:
        """Return prioritized default model candidates for the given analysis task."""

    @abstractmethod
    def validate_physical_ranges(self, frame: DataFrameLike) -> list[ValidationIssue]:
        """Validate physical ranges against industry expectations."""

    @abstractmethod
    def get_recommendation_constraints(self) -> list[VariableConstraint]:
        """Return default recommendation constraints for controllable variables."""


class BaseAnalysisModel(ABC):
    """Common interface shared by regression, classification, and related models."""

    @abstractmethod
    def fit(self, X: DataFrameLike, y: SeriesLike | None = None) -> Self:
        """Fit the model and return the fitted instance."""

    @abstractmethod
    def predict(self, X: DataFrameLike) -> np.ndarray:
        """Generate model predictions for the provided features."""

    @abstractmethod
    def evaluate(
        self,
        X: DataFrameLike,
        y: SeriesLike | None = None,
    ) -> ModelEvaluation:
        """Evaluate model performance and return structured metrics."""

    @abstractmethod
    def explain(self, X: DataFrameLike) -> ExplanationResult:
        """Explain model predictions for the provided features."""

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Return traceable model metadata."""


class BaseAnomalyModel(BaseAnalysisModel):
    """Analysis model interface extended with anomaly scoring and classification."""

    @abstractmethod
    def score_samples(self, X: DataFrameLike) -> np.ndarray:
        """Compute per-sample anomaly scores."""

    @abstractmethod
    def classify_anomalies(self, X: DataFrameLike) -> list[AnomalyEvent]:
        """Classify anomalies and return structured anomaly events."""
