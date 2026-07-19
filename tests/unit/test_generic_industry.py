"""Unit tests for GenericIndustryProfile and create_default_registry (Step 3A)."""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.protocols import BaseIndustryProfile
from process_intelligence.core.schemas import DatasetMetadata, PreprocessingRules, SchemaHints
from process_intelligence.industries import (
    GenericIndustryProfile,
    IndustryRegistry,
    create_default_registry,
)


def _metadata(**overrides: object) -> DatasetMetadata:
    payload: dict[str, object] = {
        "file_name": "sample.csv",
        "file_format": "csv",
        "row_count": 2,
        "column_names": ["temp", "pressure"],
        "dtypes": {"temp": "Float64", "pressure": "Float64"},
        "user_description": None,
    }
    payload.update(overrides)
    return DatasetMetadata(**payload)  # type: ignore[arg-type]


def test_generic_profile_instantiate() -> None:
    profile = GenericIndustryProfile()
    assert isinstance(profile, GenericIndustryProfile)


def test_generic_profile_is_base_industry_profile() -> None:
    assert isinstance(GenericIndustryProfile(), BaseIndustryProfile)


def test_generic_industry_name() -> None:
    assert GenericIndustryProfile().industry_name == "generic"


def test_score_industry_runs() -> None:
    score = GenericIndustryProfile().score_industry(_metadata())
    assert score.industry_name == "generic"


def test_score_is_zero() -> None:
    score = GenericIndustryProfile().score_industry(_metadata())
    assert score.score == 0.0


def test_confidence_is_zero() -> None:
    score = GenericIndustryProfile().score_industry(_metadata())
    assert score.confidence == 0.0


def test_requires_user_confirmation_true() -> None:
    score = GenericIndustryProfile().score_industry(_metadata())
    assert score.requires_user_confirmation is True


def test_evidence_not_empty() -> None:
    score = GenericIndustryProfile().score_industry(_metadata())
    assert len(score.evidence) >= 1
    assert all(isinstance(item, str) and item.strip() for item in score.evidence)


def test_uncertain_factors_not_empty() -> None:
    score = GenericIndustryProfile().score_industry(_metadata())
    assert len(score.uncertain_factors) >= 1
    assert all(
        isinstance(item, str) and item.strip() for item in score.uncertain_factors
    )


def test_score_industry_rejects_invalid_metadata_type() -> None:
    with pytest.raises(TypeError, match="DatasetMetadata"):
        GenericIndustryProfile().score_industry("not-metadata")  # type: ignore[arg-type]


def test_score_industry_does_not_mutate_metadata() -> None:
    metadata = _metadata(column_names=["temp", "pressure"])
    original = metadata.model_dump()
    GenericIndustryProfile().score_industry(metadata)
    assert metadata.model_dump() == original


def test_get_schema_hints_returns_schema_hints() -> None:
    hints = GenericIndustryProfile().get_schema_hints()
    assert isinstance(hints, SchemaHints)


def test_schema_hints_default_collections_empty() -> None:
    hints = GenericIndustryProfile().get_schema_hints()
    assert hints.keywords == []
    assert hints.synonyms == {}
    assert hints.expected_units == {}
    assert hints.default_roles == {}
    assert hints.physical_ranges == {}


def test_get_schema_hints_returns_independent_instances() -> None:
    profile = GenericIndustryProfile()
    first = profile.get_schema_hints()
    second = profile.get_schema_hints()
    assert first is not second
    first.keywords.append("temp")
    assert second.keywords == []


def test_get_preprocessing_rules_returns_preprocessing_rules() -> None:
    rules = GenericIndustryProfile().get_preprocessing_rules()
    assert isinstance(rules, PreprocessingRules)


def test_preprocessing_rules_default_collections_empty() -> None:
    rules = GenericIndustryProfile().get_preprocessing_rules()
    assert rules.imputation == {}
    assert rules.scaling == {}
    assert rules.outlier_handling == {}
    assert rules.industry_specific == {}


def test_get_preprocessing_rules_returns_independent_instances() -> None:
    profile = GenericIndustryProfile()
    first = profile.get_preprocessing_rules()
    second = profile.get_preprocessing_rules()
    assert first is not second
    first.imputation["temp"] = "median"
    assert second.imputation == {}


def test_get_default_model_candidates_returns_empty_list() -> None:
    candidates = GenericIndustryProfile().get_default_model_candidates(
        AnalysisTask.REGRESSION
    )
    assert candidates == []


def test_get_default_model_candidates_rejects_invalid_task() -> None:
    with pytest.raises(TypeError, match="AnalysisTask"):
        GenericIndustryProfile().get_default_model_candidates(
            "REGRESSION"  # type: ignore[arg-type]
        )


def test_get_default_model_candidates_returns_independent_lists() -> None:
    profile = GenericIndustryProfile()
    first = profile.get_default_model_candidates(AnalysisTask.CLASSIFICATION)
    second = profile.get_default_model_candidates(AnalysisTask.CLASSIFICATION)
    assert first is not second
    first.append("mutated")  # type: ignore[arg-type]
    assert second == []


def test_validate_physical_ranges_accepts_polars_dataframe() -> None:
    frame = pl.DataFrame({"temp": [1.0, 2.0]})
    issues = GenericIndustryProfile().validate_physical_ranges(frame)
    assert issues == []


def test_validate_physical_ranges_accepts_pandas_dataframe() -> None:
    frame = pd.DataFrame({"temp": [1.0, 2.0]})
    issues = GenericIndustryProfile().validate_physical_ranges(frame)
    assert issues == []


def test_validate_physical_ranges_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        GenericIndustryProfile().validate_physical_ranges([1, 2, 3])  # type: ignore[arg-type]


def test_validate_physical_ranges_does_not_mutate_frame() -> None:
    frame = pl.DataFrame({"temp": [1.0, 2.0]})
    before = frame.clone()
    GenericIndustryProfile().validate_physical_ranges(frame)
    assert frame.equals(before)


def test_validate_physical_ranges_returns_independent_lists() -> None:
    profile = GenericIndustryProfile()
    frame = pl.DataFrame({"temp": [1.0]})
    first = profile.validate_physical_ranges(frame)
    second = profile.validate_physical_ranges(frame)
    assert first is not second
    first.append("mutated")  # type: ignore[arg-type]
    assert second == []


def test_get_recommendation_constraints_returns_empty_list() -> None:
    assert GenericIndustryProfile().get_recommendation_constraints() == []


def test_get_recommendation_constraints_returns_independent_lists() -> None:
    profile = GenericIndustryProfile()
    first = profile.get_recommendation_constraints()
    second = profile.get_recommendation_constraints()
    assert first is not second
    first.append("mutated")  # type: ignore[arg-type]
    assert second == []


def test_create_default_registry_returns_industry_registry() -> None:
    registry = create_default_registry()
    assert isinstance(registry, IndustryRegistry)


def test_default_registry_names_are_generic_only() -> None:
    registry = create_default_registry()
    assert registry.list_names() == ("generic",)
    assert len(registry) == 1
    assert isinstance(registry.get("generic"), GenericIndustryProfile)


def test_create_default_registry_returns_independent_instances() -> None:
    first = create_default_registry()
    second = create_default_registry()
    assert first is not second
    assert first.get("generic") is not second.get("generic")


def test_default_registry_mutations_are_isolated() -> None:
    first = create_default_registry()
    second = create_default_registry()
    first.unregister("generic")
    assert len(first) == 0
    assert second.list_names() == ("generic",)
    assert second.contains("generic") is True
