"""Generic fallback industry profile and default registry factory."""

from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]
import polars as pl

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.protocols import BaseIndustryProfile, DataFrameLike
from process_intelligence.core.schemas import (
    DatasetMetadata,
    IndustryScore,
    ModelSpec,
    PreprocessingRules,
    SchemaHints,
    ValidationIssue,
    VariableConstraint,
)
from process_intelligence.industries.registry import IndustryRegistry


class GenericIndustryProfile(BaseIndustryProfile):
    """Minimal fallback industry profile used when no industry is identified."""

    industry_name: str = "generic"

    def score_industry(self, metadata: DatasetMetadata) -> IndustryScore:
        """Return a low-confidence fallback score requiring user confirmation.

        Args:
            metadata: Dataset metadata used for industry estimation.

        Returns:
            An ``IndustryScore`` with zero score/confidence that requires user
            confirmation.

        Raises:
            TypeError: If ``metadata`` is not a ``DatasetMetadata`` instance.
        """
        if not isinstance(metadata, DatasetMetadata):
            raise TypeError(
                f"metadata must be DatasetMetadata, got {type(metadata).__name__}"
            )

        return IndustryScore(
            industry_name=self.industry_name,
            score=0.0,
            confidence=0.0,
            evidence=[
                "No industry-specific evidence was found in the dataset metadata."
            ],
            uncertain_factors=[
                "No industry-specific profile was selected; using the generic fallback."
            ],
            requires_user_confirmation=True,
        )

    def get_schema_hints(self) -> SchemaHints:
        """Return empty default schema hints."""
        return SchemaHints()

    def get_preprocessing_rules(self) -> PreprocessingRules:
        """Return empty default preprocessing rules."""
        return PreprocessingRules()

    def get_default_model_candidates(self, task: AnalysisTask) -> list[ModelSpec]:
        """Return no default model candidates for the generic profile.

        Args:
            task: Analysis task for which candidates are requested.

        Returns:
            An empty list of model specifications.

        Raises:
            TypeError: If ``task`` is not an ``AnalysisTask`` instance.
        """
        if not isinstance(task, AnalysisTask):
            raise TypeError(f"task must be AnalysisTask, got {type(task).__name__}")

        return []

    def validate_physical_ranges(self, frame: DataFrameLike) -> list[ValidationIssue]:
        """Return no physical-range issues for the generic profile.

        Args:
            frame: Polars or Pandas DataFrame to validate.

        Returns:
            An empty list of validation issues.

        Raises:
            TypeError: If ``frame`` is not a Polars or Pandas DataFrame.
        """
        if not isinstance(frame, (pl.DataFrame, pd.DataFrame)):
            raise TypeError(
                f"frame must be a Polars or Pandas DataFrame, "
                f"got {type(frame).__name__}"
            )

        return []

    def get_recommendation_constraints(self) -> list[VariableConstraint]:
        """Return no recommendation constraints for the generic profile."""
        return []


def create_default_registry() -> IndustryRegistry:
    """Create a new registry containing only the generic industry profile.

    Returns:
        A newly constructed ``IndustryRegistry`` instance. Each call returns an
        independent registry rather than a shared singleton.
    """
    return IndustryRegistry(profiles=[GenericIndustryProfile()])
