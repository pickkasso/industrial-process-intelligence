"""Unit tests for IndustryRegistry (Step 3A)."""

from __future__ import annotations

from typing import Any

import pytest

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.exceptions import DataValidationError, InsufficientDataError
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
from process_intelligence.industries import IndustryRegistry


class StubIndustryProfile(BaseIndustryProfile):
    """Minimal BaseIndustryProfile stub for registry tests."""

    def __init__(
        self,
        industry_name: Any = "stub",
        *,
        score: float = 0.5,
        confidence: float = 0.5,
        score_result: IndustryScore | object | None = None,
        raise_on_score: Exception | None = None,
    ) -> None:
        self._industry_name = industry_name
        self._score = score
        self._confidence = confidence
        self._score_result = score_result
        self._raise_on_score = raise_on_score
        self.score_calls: list[DatasetMetadata] = []

    @property
    def industry_name(self) -> Any:
        return self._industry_name

    def score_industry(self, metadata: DatasetMetadata) -> IndustryScore:
        self.score_calls.append(metadata)
        if self._raise_on_score is not None:
            raise self._raise_on_score
        if self._score_result is not None:
            return self._score_result  # type: ignore[return-value]
        return IndustryScore(
            industry_name=str(self._industry_name),
            score=self._score,
            confidence=self._confidence,
            evidence=["stub evidence"],
            uncertain_factors=[],
            requires_user_confirmation=False,
        )

    def get_schema_hints(self) -> SchemaHints:
        return SchemaHints()

    def get_preprocessing_rules(self) -> PreprocessingRules:
        return PreprocessingRules()

    def get_default_model_candidates(self, task: AnalysisTask) -> list[ModelSpec]:
        _ = task
        return []

    def validate_physical_ranges(self, frame: DataFrameLike) -> list[ValidationIssue]:
        _ = frame
        return []

    def get_recommendation_constraints(self) -> list[VariableConstraint]:
        return []


def _metadata(**overrides: Any) -> DatasetMetadata:
    payload = {
        "file_name": "sample.csv",
        "file_format": "csv",
        "row_count": 3,
        "column_names": ["a", "b"],
        "dtypes": {"a": "Float64", "b": "Utf8"},
        "user_description": None,
    }
    payload.update(overrides)
    return DatasetMetadata(**payload)


def test_empty_registry_creation() -> None:
    registry = IndustryRegistry()
    assert len(registry) == 0
    assert registry.list_names() == ()
    assert registry.list_profiles() == ()


def test_initial_length_is_zero() -> None:
    assert len(IndustryRegistry()) == 0


def test_constructor_registers_profiles_in_order() -> None:
    first = StubIndustryProfile("alpha")
    second = StubIndustryProfile("beta")
    registry = IndustryRegistry(profiles=[first, second])
    assert registry.list_names() == ("alpha", "beta")
    assert registry.list_profiles() == (first, second)
    assert len(registry) == 2


def test_constructor_does_not_mutate_input_sequence() -> None:
    profiles = [StubIndustryProfile("alpha"), StubIndustryProfile("beta")]
    original = list(profiles)
    IndustryRegistry(profiles=profiles)
    assert profiles == original


def test_constructor_rejects_non_profile_elements() -> None:
    with pytest.raises(TypeError, match="BaseIndustryProfile"):
        IndustryRegistry(profiles=["not-a-profile"])  # type: ignore[list-item]


def test_constructor_rejects_duplicate_names() -> None:
    with pytest.raises(DataValidationError, match="alpha"):
        IndustryRegistry(
            profiles=[
                StubIndustryProfile("alpha"),
                StubIndustryProfile("alpha"),
            ]
        )


def test_register_success() -> None:
    registry = IndustryRegistry()
    profile = StubIndustryProfile("alpha")
    registry.register(profile)
    assert registry.get("alpha") is profile


def test_register_increases_length() -> None:
    registry = IndustryRegistry()
    assert len(registry) == 0
    registry.register(StubIndustryProfile("alpha"))
    assert len(registry) == 1


def test_register_rejects_non_profile() -> None:
    registry = IndustryRegistry()
    with pytest.raises(TypeError, match="BaseIndustryProfile"):
        registry.register("not-a-profile")  # type: ignore[arg-type]


def test_register_rejects_non_string_industry_name() -> None:
    registry = IndustryRegistry()
    with pytest.raises(TypeError, match="industry_name"):
        registry.register(StubIndustryProfile(123))  # type: ignore[arg-type]


def test_register_rejects_empty_industry_name() -> None:
    registry = IndustryRegistry()
    with pytest.raises(DataValidationError):
        registry.register(StubIndustryProfile(""))


def test_register_rejects_whitespace_only_industry_name() -> None:
    registry = IndustryRegistry()
    with pytest.raises(DataValidationError):
        registry.register(StubIndustryProfile("   "))


def test_register_rejects_duplicate_name() -> None:
    registry = IndustryRegistry(profiles=[StubIndustryProfile("alpha")])
    with pytest.raises(DataValidationError, match="alpha"):
        registry.register(StubIndustryProfile("alpha"))


def test_register_rejects_case_insensitive_duplicate() -> None:
    registry = IndustryRegistry(profiles=[StubIndustryProfile("Semiconductor")])
    with pytest.raises(DataValidationError):
        registry.register(StubIndustryProfile("semiconductor"))


def test_register_rejects_whitespace_variant_duplicate() -> None:
    registry = IndustryRegistry(profiles=[StubIndustryProfile("Semiconductor")])
    with pytest.raises(DataValidationError):
        registry.register(StubIndustryProfile(" Semiconductor "))


def test_register_replace_false_rejects_duplicate() -> None:
    registry = IndustryRegistry(profiles=[StubIndustryProfile("alpha")])
    with pytest.raises(DataValidationError):
        registry.register(StubIndustryProfile("alpha"), replace=False)


def test_register_replace_true_succeeds() -> None:
    original = StubIndustryProfile("alpha", score=0.1)
    replacement = StubIndustryProfile("alpha", score=0.9)
    registry = IndustryRegistry(profiles=[original])
    registry.register(replacement, replace=True)
    assert registry.get("alpha") is replacement
    assert registry.get("alpha") is not original


def test_register_replace_requires_bool() -> None:
    registry = IndustryRegistry()
    with pytest.raises(TypeError, match="replace"):
        registry.register(StubIndustryProfile("alpha"), replace=1)  # type: ignore[arg-type]


def test_replace_preserves_existing_order_position() -> None:
    first = StubIndustryProfile("alpha")
    second = StubIndustryProfile("beta")
    third = StubIndustryProfile("gamma")
    replacement = StubIndustryProfile("beta")
    registry = IndustryRegistry(profiles=[first, second, third])
    registry.register(replacement, replace=True)
    assert registry.list_names() == ("alpha", "beta", "gamma")
    assert registry.list_profiles() == (first, replacement, third)


def test_new_registration_appends_to_end() -> None:
    registry = IndustryRegistry(profiles=[StubIndustryProfile("alpha")])
    registry.register(StubIndustryProfile("beta"))
    assert registry.list_names() == ("alpha", "beta")


def test_get_returns_registered_profile() -> None:
    profile = StubIndustryProfile("alpha")
    registry = IndustryRegistry(profiles=[profile])
    assert registry.get("alpha") is profile


def test_get_is_case_insensitive() -> None:
    profile = StubIndustryProfile("Semiconductor")
    registry = IndustryRegistry(profiles=[profile])
    assert registry.get("SEMICONDUCTOR") is profile
    assert registry.get("semiconductor") is profile


def test_get_ignores_surrounding_whitespace() -> None:
    profile = StubIndustryProfile("Semiconductor")
    registry = IndustryRegistry(profiles=[profile])
    assert registry.get(" Semiconductor ") is profile


def test_get_rejects_non_string_name() -> None:
    registry = IndustryRegistry()
    with pytest.raises(TypeError, match="industry_name"):
        registry.get(123)  # type: ignore[arg-type]


def test_get_rejects_empty_name() -> None:
    registry = IndustryRegistry()
    with pytest.raises(ValueError):
        registry.get("")
    with pytest.raises(ValueError):
        registry.get("   ")


def test_get_missing_name_raises_key_error() -> None:
    registry = IndustryRegistry()
    with pytest.raises(KeyError):
        registry.get("missing")


def test_get_key_error_includes_lookup_name() -> None:
    registry = IndustryRegistry()
    with pytest.raises(KeyError, match="missing-industry"):
        registry.get("missing-industry")


def test_contains_registered_name_true() -> None:
    registry = IndustryRegistry(profiles=[StubIndustryProfile("alpha")])
    assert registry.contains("alpha") is True
    assert registry.contains(" ALPHA ") is True


def test_contains_unregistered_name_false() -> None:
    registry = IndustryRegistry(profiles=[StubIndustryProfile("alpha")])
    assert registry.contains("beta") is False


def test_contains_applies_name_validation() -> None:
    registry = IndustryRegistry()
    with pytest.raises(TypeError):
        registry.contains(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        registry.contains("")


def test_unregister_success() -> None:
    profile = StubIndustryProfile("alpha")
    registry = IndustryRegistry(profiles=[profile])
    removed = registry.unregister("alpha")
    assert removed is profile
    assert len(registry) == 0
    assert registry.contains("alpha") is False


def test_unregister_returns_removed_profile() -> None:
    profile = StubIndustryProfile("alpha")
    registry = IndustryRegistry(profiles=[profile])
    assert registry.unregister(" ALPHA ") is profile


def test_unregister_decreases_length() -> None:
    registry = IndustryRegistry(
        profiles=[StubIndustryProfile("alpha"), StubIndustryProfile("beta")]
    )
    assert len(registry) == 2
    registry.unregister("alpha")
    assert len(registry) == 1


def test_unregister_preserves_remaining_order() -> None:
    first = StubIndustryProfile("alpha")
    second = StubIndustryProfile("beta")
    third = StubIndustryProfile("gamma")
    registry = IndustryRegistry(profiles=[first, second, third])
    registry.unregister("beta")
    assert registry.list_names() == ("alpha", "gamma")
    assert registry.list_profiles() == (first, third)


def test_unregister_missing_name_raises_key_error() -> None:
    registry = IndustryRegistry()
    with pytest.raises(KeyError, match="missing"):
        registry.unregister("missing")


def test_list_names_preserves_registration_order() -> None:
    registry = IndustryRegistry(
        profiles=[
            StubIndustryProfile("alpha"),
            StubIndustryProfile("beta"),
            StubIndustryProfile("gamma"),
        ]
    )
    assert registry.list_names() == ("alpha", "beta", "gamma")


def test_list_names_returns_tuple() -> None:
    registry = IndustryRegistry(profiles=[StubIndustryProfile("alpha")])
    names = registry.list_names()
    assert isinstance(names, tuple)
    assert names == ("alpha",)


def test_list_profiles_preserves_registration_order() -> None:
    first = StubIndustryProfile("alpha")
    second = StubIndustryProfile("beta")
    registry = IndustryRegistry(profiles=[first, second])
    assert registry.list_profiles() == (first, second)


def test_list_profiles_returns_tuple() -> None:
    registry = IndustryRegistry(profiles=[StubIndustryProfile("alpha")])
    profiles = registry.list_profiles()
    assert isinstance(profiles, tuple)


def test_score_all_calls_profiles_in_registration_order() -> None:
    first = StubIndustryProfile("alpha", score=0.1)
    second = StubIndustryProfile("beta", score=0.2)
    registry = IndustryRegistry(profiles=[first, second])
    metadata = _metadata()
    scores = registry.score_all(metadata)
    assert [score.industry_name for score in scores] == ["alpha", "beta"]
    assert first.score_calls == [metadata]
    assert second.score_calls == [metadata]


def test_score_all_returns_tuple() -> None:
    registry = IndustryRegistry(profiles=[StubIndustryProfile("alpha")])
    result = registry.score_all(_metadata())
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert isinstance(result[0], IndustryScore)


def test_score_all_empty_registry_returns_empty_tuple() -> None:
    registry = IndustryRegistry()
    assert registry.score_all(_metadata()) == ()


def test_score_all_rejects_invalid_metadata_type() -> None:
    registry = IndustryRegistry(profiles=[StubIndustryProfile("alpha")])
    with pytest.raises(TypeError, match="DatasetMetadata"):
        registry.score_all({"file_name": "x.csv"})  # type: ignore[arg-type]


def test_score_all_rejects_non_industry_score_return() -> None:
    registry = IndustryRegistry(
        profiles=[StubIndustryProfile("alpha", score_result={"score": 1.0})]
    )
    with pytest.raises(DataValidationError, match="IndustryScore"):
        registry.score_all(_metadata())


def test_score_all_rejects_empty_industry_name_in_score() -> None:
    bad_score = IndustryScore(
        industry_name="",
        score=0.1,
        confidence=0.1,
        evidence=["e"],
        uncertain_factors=[],
        requires_user_confirmation=True,
    )
    registry = IndustryRegistry(
        profiles=[StubIndustryProfile("alpha", score_result=bad_score)]
    )
    with pytest.raises(DataValidationError, match="empty industry_name"):
        registry.score_all(_metadata())


def test_score_all_propagates_domain_exceptions() -> None:
    registry = IndustryRegistry(
        profiles=[
            StubIndustryProfile(
                "alpha",
                raise_on_score=InsufficientDataError("not enough rows"),
            )
        ]
    )
    with pytest.raises(InsufficientDataError, match="not enough rows"):
        registry.score_all(_metadata())


def test_best_match_empty_registry_returns_none() -> None:
    assert IndustryRegistry().best_match(_metadata()) is None


def test_best_match_selects_highest_score() -> None:
    low = StubIndustryProfile("low", score=0.2, confidence=0.9)
    high = StubIndustryProfile("high", score=0.8, confidence=0.1)
    registry = IndustryRegistry(profiles=[low, high])
    best = registry.best_match(_metadata())
    assert best is not None
    assert best.industry_name == "high"


def test_best_match_breaks_score_tie_by_confidence() -> None:
    lower_confidence = StubIndustryProfile("a", score=0.5, confidence=0.2)
    higher_confidence = StubIndustryProfile("b", score=0.5, confidence=0.8)
    registry = IndustryRegistry(profiles=[lower_confidence, higher_confidence])
    best = registry.best_match(_metadata())
    assert best is not None
    assert best.industry_name == "b"


def test_best_match_breaks_full_tie_by_registration_order() -> None:
    first = StubIndustryProfile("first", score=0.5, confidence=0.5)
    second = StubIndustryProfile("second", score=0.5, confidence=0.5)
    registry = IndustryRegistry(profiles=[first, second])
    best = registry.best_match(_metadata())
    assert best is not None
    assert best.industry_name == "first"


def test_best_match_does_not_change_registry_order() -> None:
    first = StubIndustryProfile("first", score=0.1, confidence=0.1)
    second = StubIndustryProfile("second", score=0.9, confidence=0.9)
    registry = IndustryRegistry(profiles=[first, second])
    registry.best_match(_metadata())
    assert registry.list_names() == ("first", "second")
    assert registry.list_profiles() == (first, second)


def test_score_all_does_not_mutate_metadata() -> None:
    registry = IndustryRegistry(profiles=[StubIndustryProfile("alpha")])
    metadata = _metadata(column_names=["a", "b"])
    original_columns = list(metadata.column_names)
    original_dump = metadata.model_dump()
    registry.score_all(metadata)
    assert metadata.column_names == original_columns
    assert metadata.model_dump() == original_dump


def test_score_all_does_not_change_registry_state() -> None:
    first = StubIndustryProfile("alpha")
    second = StubIndustryProfile("beta")
    registry = IndustryRegistry(profiles=[first, second])
    before_names = registry.list_names()
    before_profiles = registry.list_profiles()
    before_len = len(registry)
    registry.score_all(_metadata())
    assert registry.list_names() == before_names
    assert registry.list_profiles() == before_profiles
    assert len(registry) == before_len


def test_registry_instances_do_not_share_state() -> None:
    first = IndustryRegistry()
    second = IndustryRegistry()
    first.register(StubIndustryProfile("alpha"))
    assert first.contains("alpha") is True
    assert second.contains("alpha") is False
    assert len(second) == 0


def test_best_match_is_deterministic_for_same_input() -> None:
    profiles = [
        StubIndustryProfile("alpha", score=0.4, confidence=0.4),
        StubIndustryProfile("beta", score=0.7, confidence=0.2),
        StubIndustryProfile("gamma", score=0.7, confidence=0.9),
    ]
    registry = IndustryRegistry(profiles=profiles)
    metadata = _metadata()
    first = registry.best_match(metadata)
    second = registry.best_match(metadata)
    assert first is not None
    assert second is not None
    assert first.model_dump() == second.model_dump()
    assert first.industry_name == "gamma"
