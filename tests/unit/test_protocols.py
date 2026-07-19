"""Unit tests for core abstract interfaces."""

from typing import Self, get_args

import numpy as np
import pandas as pd
import polars as pl
import pytest

from process_intelligence.core import (
    AnalysisTask,
    AnomalyEvent,
    AnomalyType,
    BaseAnalysisModel,
    BaseAnomalyModel,
    BaseIndustryProfile,
    DataFrameLike,
    DatasetMetadata,
    ExplanationResult,
    IndustryScore,
    ModelEvaluation,
    ModelMetadata,
    ModelSpec,
    PreprocessingRules,
    SchemaHints,
    SeriesLike,
    ValidationIssue,
    VariableConstraint,
)


class MinimalIndustryProfile(BaseIndustryProfile):
    """Minimal concrete industry profile used for interface tests."""

    @property
    def industry_name(self) -> str:
        return "test_industry"

    def score_industry(self, metadata: DatasetMetadata) -> IndustryScore:
        return IndustryScore(
            industry_name=self.industry_name,
            score=1.0,
            confidence=1.0,
            evidence=[metadata.file_name],
            uncertain_factors=[],
            requires_user_confirmation=False,
        )

    def get_schema_hints(self) -> SchemaHints:
        return SchemaHints()

    def get_preprocessing_rules(self) -> PreprocessingRules:
        return PreprocessingRules()

    def get_default_model_candidates(self, task: AnalysisTask) -> list[ModelSpec]:
        return [
            ModelSpec(
                name="dummy",
                task=task,
                estimator_key="dummy",
                priority=0,
            )
        ]

    def validate_physical_ranges(self, frame: DataFrameLike) -> list[ValidationIssue]:
        _ = frame
        return []

    def get_recommendation_constraints(self) -> list[VariableConstraint]:
        return []


class MinimalAnalysisModel(BaseAnalysisModel):
    """Minimal concrete analysis model used for interface tests."""

    def fit(self, X: DataFrameLike, y: SeriesLike | None = None) -> Self:
        _ = X, y
        return self

    def predict(self, X: DataFrameLike) -> np.ndarray:
        return np.zeros(len(X), dtype=float)

    def evaluate(
        self,
        X: DataFrameLike,
        y: SeriesLike | None = None,
    ) -> ModelEvaluation:
        _ = X, y
        return ModelEvaluation()

    def explain(self, X: DataFrameLike) -> ExplanationResult:
        _ = X
        return ExplanationResult(method="none")

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_name="minimal",
            version="0.0.0",
            task=AnalysisTask.REGRESSION,
        )


class MinimalAnomalyModel(BaseAnomalyModel):
    """Minimal concrete anomaly model used for interface tests."""

    def fit(self, X: DataFrameLike, y: SeriesLike | None = None) -> Self:
        _ = X, y
        return self

    def predict(self, X: DataFrameLike) -> np.ndarray:
        return np.zeros(len(X), dtype=float)

    def evaluate(
        self,
        X: DataFrameLike,
        y: SeriesLike | None = None,
    ) -> ModelEvaluation:
        _ = X, y
        return ModelEvaluation()

    def explain(self, X: DataFrameLike) -> ExplanationResult:
        _ = X
        return ExplanationResult(method="none")

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_name="minimal-anomaly",
            version="0.0.0",
            task=AnalysisTask.UNSUPERVISED_ANOMALY,
        )

    def score_samples(self, X: DataFrameLike) -> np.ndarray:
        return np.zeros(len(X), dtype=float)

    def classify_anomalies(self, X: DataFrameLike) -> list[AnomalyEvent]:
        _ = X
        return [
            AnomalyEvent(
                anomaly_id="stub",
                anomaly_type=AnomalyType.DATA_QUALITY,
                anomaly_score=0.0,
                severity="low",
                model_confidence=0.0,
                detector="stub",
                rationale="stub",
            )
        ]


class AnomalyModelMissingScoreSamples(BaseAnomalyModel):
    """BaseAnomalyModel stub missing score_samples."""

    def fit(self, X: DataFrameLike, y: SeriesLike | None = None) -> Self:
        _ = X, y
        return self

    def predict(self, X: DataFrameLike) -> np.ndarray:
        return np.zeros(len(X), dtype=float)

    def evaluate(
        self,
        X: DataFrameLike,
        y: SeriesLike | None = None,
    ) -> ModelEvaluation:
        _ = X, y
        return ModelEvaluation()

    def explain(self, X: DataFrameLike) -> ExplanationResult:
        _ = X
        return ExplanationResult(method="none")

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_name="missing-score",
            version="0.0.0",
            task=AnalysisTask.UNSUPERVISED_ANOMALY,
        )

    def classify_anomalies(self, X: DataFrameLike) -> list[AnomalyEvent]:
        _ = X
        return []


class AnomalyModelMissingClassify(BaseAnomalyModel):
    """BaseAnomalyModel stub missing classify_anomalies."""

    def fit(self, X: DataFrameLike, y: SeriesLike | None = None) -> Self:
        _ = X, y
        return self

    def predict(self, X: DataFrameLike) -> np.ndarray:
        return np.zeros(len(X), dtype=float)

    def evaluate(
        self,
        X: DataFrameLike,
        y: SeriesLike | None = None,
    ) -> ModelEvaluation:
        _ = X, y
        return ModelEvaluation()

    def explain(self, X: DataFrameLike) -> ExplanationResult:
        _ = X
        return ExplanationResult(method="none")

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_name="missing-classify",
            version="0.0.0",
            task=AnalysisTask.UNSUPERVISED_ANOMALY,
        )

    def score_samples(self, X: DataFrameLike) -> np.ndarray:
        return np.zeros(len(X), dtype=float)


def test_base_industry_profile_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseIndustryProfile()  # type: ignore[abstract]


def test_base_analysis_model_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseAnalysisModel()  # type: ignore[abstract]


def test_base_anomaly_model_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseAnomalyModel()  # type: ignore[abstract]


def test_minimal_industry_profile_stub_can_be_instantiated() -> None:
    profile = MinimalIndustryProfile()
    assert profile.industry_name == "test_industry"


def test_minimal_analysis_model_stub_can_be_instantiated() -> None:
    model = MinimalAnalysisModel()
    assert isinstance(model, BaseAnalysisModel)


def test_anomaly_model_missing_score_samples_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        AnomalyModelMissingScoreSamples()  # type: ignore[abstract]


def test_anomaly_model_missing_classify_anomalies_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        AnomalyModelMissingClassify()  # type: ignore[abstract]


def test_minimal_anomaly_model_stub_can_be_instantiated() -> None:
    model = MinimalAnomalyModel()
    assert isinstance(model, BaseAnomalyModel)
    assert isinstance(model, BaseAnalysisModel)


def test_dataframe_like_includes_polars_and_pandas() -> None:
    frame_types = get_args(DataFrameLike)
    series_types = get_args(SeriesLike)
    assert pl.DataFrame in frame_types
    assert pd.DataFrame in frame_types
    assert pl.Series in series_types
    assert pd.Series in series_types
