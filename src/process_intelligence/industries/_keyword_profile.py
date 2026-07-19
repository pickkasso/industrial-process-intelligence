"""Shared keyword-based industry profile base class (internal)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import ClassVar

import pandas as pd  # type: ignore[import-untyped]
import polars as pl

from process_intelligence.core.enums import AnalysisTask, ColumnRole
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

_NON_ALNUM_HANGUL = re.compile(r"[^0-9a-z\uac00-\ud7a3]+")

_SOURCE_WEIGHTS: Mapping[str, float] = {
    "user_description": 4.0,
    "column_names": 2.0,
    "file_name": 1.0,
}

_SOURCE_ORDER: tuple[str, ...] = (
    "user_description",
    "column_names",
    "file_name",
)


def _normalize_text(value: str) -> str:
    """Normalize text for deterministic keyword matching."""
    if not value:
        return ""
    folded = value.casefold()
    replaced = _NON_ALNUM_HANGUL.sub(" ", folded)
    return " ".join(replaced.split())


def _keyword_matches(normalized_text: str, normalized_keyword: str) -> bool:
    """Return True when keyword matches as whole words or a phrase."""
    if not normalized_text or not normalized_keyword:
        return False

    text_words = normalized_text.split()
    keyword_words = normalized_keyword.split()
    if not keyword_words:
        return False

    if len(keyword_words) == 1:
        return keyword_words[0] in text_words

    window = len(keyword_words)
    for index in range(len(text_words) - window + 1):
        if text_words[index : index + window] == keyword_words:
            return True
    return False


class _KeywordIndustryProfile(BaseIndustryProfile):
    """Common keyword-scoring base for concrete industry profiles.

    Subclasses supply immutable class-level keyword and schema-hint data.
    This class is internal and is not part of the public industries API.
    """

    industry_name: ClassVar[str]
    keywords: ClassVar[tuple[str, ...]]
    synonyms: ClassVar[Mapping[str, tuple[str, ...]]]
    expected_units: ClassVar[Mapping[str, str]]
    default_roles: ClassVar[Mapping[str, ColumnRole]]
    physical_ranges: ClassVar[Mapping[str, tuple[float | None, float | None]]]

    def score_industry(self, metadata: DatasetMetadata) -> IndustryScore:
        """Score dataset metadata using keyword signals from three sources.

        Args:
            metadata: Dataset metadata used for industry estimation.

        Returns:
            An ``IndustryScore`` with deterministic score, confidence, and
            evidence derived from keyword matches.

        Raises:
            TypeError: If ``metadata`` is not a ``DatasetMetadata`` instance.
        """
        if not isinstance(metadata, DatasetMetadata):
            raise TypeError(
                f"metadata must be DatasetMetadata, got {type(metadata).__name__}"
            )

        matches_by_source = self._collect_matches(metadata)
        score = 0.0
        evidence: list[str] = []

        for source in _SOURCE_ORDER:
            weight = _SOURCE_WEIGHTS[source]
            for keyword in matches_by_source[source]:
                score += weight
                evidence_item = f"Matched keyword '{keyword}' in {source}."
                if evidence_item not in evidence:
                    evidence.append(evidence_item)

        confidence = round(min(1.0, score / 12.0), 4)
        requires_user_confirmation = confidence < 0.60
        uncertain_factors = self._build_uncertain_factors(score, confidence)

        return IndustryScore(
            industry_name=self.industry_name,
            score=score,
            confidence=confidence,
            evidence=evidence,
            uncertain_factors=uncertain_factors,
            requires_user_confirmation=requires_user_confirmation,
        )

    def get_schema_hints(self) -> SchemaHints:
        """Return a fresh SchemaHints copy built from class-level settings."""
        return SchemaHints(
            keywords=list(self.keywords),
            synonyms={key: list(values) for key, values in self.synonyms.items()},
            expected_units=dict(self.expected_units),
            default_roles=dict(self.default_roles),
            physical_ranges=dict(self.physical_ranges),
        )

    def get_preprocessing_rules(self) -> PreprocessingRules:
        """Return empty preprocessing rules for this stage."""
        return PreprocessingRules()

    def get_default_model_candidates(self, task: AnalysisTask) -> list[ModelSpec]:
        """Return no model candidates for this stage.

        Args:
            task: Analysis task for which candidates are requested.

        Returns:
            A new empty list of model specifications.

        Raises:
            TypeError: If ``task`` is not an ``AnalysisTask`` instance.
        """
        if not isinstance(task, AnalysisTask):
            raise TypeError(f"task must be AnalysisTask, got {type(task).__name__}")

        return []

    def validate_physical_ranges(self, frame: DataFrameLike) -> list[ValidationIssue]:
        """Return no physical-range issues for this stage.

        Args:
            frame: Polars or Pandas DataFrame to validate.

        Returns:
            A new empty list of validation issues.

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
        """Return no recommendation constraints for this stage."""
        return []

    def _collect_matches(
        self,
        metadata: DatasetMetadata,
    ) -> dict[str, list[str]]:
        """Collect matched keywords per source in declaration order."""
        description = metadata.user_description or ""
        return {
            "user_description": self._matched_keywords_in_text(description),
            "column_names": self._matched_keywords_in_columns(metadata.column_names),
            "file_name": self._matched_keywords_in_text(metadata.file_name),
        }

    def _matched_keywords_in_text(self, text: str) -> list[str]:
        """Return keywords matched in a single text value."""
        normalized_text = _normalize_text(text)
        matched: list[str] = []
        for keyword in self.keywords:
            if _keyword_matches(normalized_text, _normalize_text(keyword)):
                matched.append(keyword)
        return matched

    def _matched_keywords_in_columns(self, column_names: list[str]) -> list[str]:
        """Return keywords matched at least once across column names."""
        normalized_columns = [_normalize_text(name) for name in column_names]
        matched: list[str] = []
        for keyword in self.keywords:
            normalized_keyword = _normalize_text(keyword)
            if any(
                _keyword_matches(column_text, normalized_keyword)
                for column_text in normalized_columns
            ):
                matched.append(keyword)
        return matched

    @staticmethod
    def _build_uncertain_factors(score: float, confidence: float) -> list[str]:
        """Build uncertain-factor messages from score and confidence."""
        if score == 0.0:
            return ["No industry-specific signals were detected."]
        if confidence < 0.60:
            return [
                "Available metadata provides limited industry-specific evidence."
            ]
        return []
