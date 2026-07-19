"""Unit tests for IndustryRouter (Step 3C)."""

from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import ValidationError

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.exceptions import DataValidationError
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
from process_intelligence.industries import (
    GenericIndustryProfile,
    IndustryRegistry,
    create_default_registry,
)
from process_intelligence.routing import (
    IndustryRouter,
    IndustryRoutingPolicy,
    IndustryRoutingResult,
    create_default_industry_router,
)


class StubIndustryProfile(BaseIndustryProfile):
    """Minimal BaseIndustryProfile stub for router tests."""

    def __init__(
        self,
        industry_name: Any = "stub",
        *,
        score: float = 0.5,
        confidence: float = 0.5,
        requires_user_confirmation: bool = False,
        score_result: IndustryScore | object | None = None,
        evidence: list[str] | None = None,
    ) -> None:
        self._industry_name = industry_name
        self._score = score
        self._confidence = confidence
        self._requires_user_confirmation = requires_user_confirmation
        self._score_result = score_result
        self._evidence = evidence if evidence is not None else ["stub evidence"]
        self.score_calls: list[DatasetMetadata] = []

    @property
    def industry_name(self) -> Any:
        return self._industry_name

    def score_industry(self, metadata: DatasetMetadata) -> IndustryScore:
        self.score_calls.append(metadata)
        if self._score_result is not None:
            return self._score_result  # type: ignore[return-value]
        return IndustryScore(
            industry_name=str(self._industry_name),
            score=self._score,
            confidence=self._confidence,
            evidence=list(self._evidence),
            uncertain_factors=[],
            requires_user_confirmation=self._requires_user_confirmation,
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
    payload: dict[str, Any] = {
        "file_name": "sample.csv",
        "file_format": "csv",
        "row_count": 3,
        "column_names": ["a", "b"],
        "dtypes": {"a": "Float64", "b": "Utf8"},
        "user_description": None,
    }
    payload.update(overrides)
    return DatasetMetadata(**payload)


def _score(
    industry_name: str,
    *,
    score: float = 5.0,
    confidence: float = 0.8,
    requires_user_confirmation: bool = False,
) -> IndustryScore:
    return IndustryScore(
        industry_name=industry_name,
        score=score,
        confidence=confidence,
        evidence=["evidence"],
        uncertain_factors=[],
        requires_user_confirmation=requires_user_confirmation,
    )


def _result(
    decision: str,
    *,
    selected_industry: str = "alpha",
    selected_score: IndustryScore | None = None,
    requires_user_confirmation: bool,
    user_confirmed: bool = False,
    reason: str = "test reason",
    score_margin: float | None = None,
    ranked_candidates: list[IndustryScore] | None = None,
    alternatives: list[IndustryScore] | None = None,
) -> IndustryRoutingResult:
    score = selected_score or _score(selected_industry)
    return IndustryRoutingResult(
        decision=decision,  # type: ignore[arg-type]
        selected_industry=selected_industry,
        selected_score=score,
        ranked_candidates=ranked_candidates if ranked_candidates is not None else [],
        alternatives=alternatives if alternatives is not None else [],
        requires_user_confirmation=requires_user_confirmation,
        reason=reason,
        score_margin=score_margin,
        user_confirmed=user_confirmed,
    )


def _router(
    *profiles: BaseIndustryProfile,
    policy: IndustryRoutingPolicy | None = None,
    fallback_profile: BaseIndustryProfile | None = None,
) -> IndustryRouter:
    return IndustryRouter(
        IndustryRegistry(profiles=list(profiles)),
        policy=policy,
        fallback_profile=fallback_profile,
    )


# --- IndustryRoutingPolicy ---


def test_policy_default_creation() -> None:
    policy = IndustryRoutingPolicy()
    assert isinstance(policy, IndustryRoutingPolicy)


def test_policy_default_values() -> None:
    policy = IndustryRoutingPolicy()
    assert policy.minimum_candidate_score == pytest.approx(1.0)
    assert policy.minimum_auto_confidence == pytest.approx(0.60)
    assert policy.minimum_score_margin == pytest.approx(2.0)


def test_policy_rejects_negative_minimum_candidate_score() -> None:
    with pytest.raises(ValidationError):
        IndustryRoutingPolicy(minimum_candidate_score=-0.1)


def test_policy_rejects_negative_minimum_auto_confidence() -> None:
    with pytest.raises(ValidationError):
        IndustryRoutingPolicy(minimum_auto_confidence=-0.1)


def test_policy_rejects_minimum_auto_confidence_above_one() -> None:
    with pytest.raises(ValidationError):
        IndustryRoutingPolicy(minimum_auto_confidence=1.1)


def test_policy_rejects_negative_minimum_score_margin() -> None:
    with pytest.raises(ValidationError):
        IndustryRoutingPolicy(minimum_score_margin=-1.0)


def test_policy_rejects_bool_numeric_inputs() -> None:
    with pytest.raises(ValidationError):
        IndustryRoutingPolicy(minimum_candidate_score=True)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        IndustryRoutingPolicy(minimum_auto_confidence=False)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        IndustryRoutingPolicy(minimum_score_margin=True)  # type: ignore[arg-type]


def test_policy_rejects_nan() -> None:
    with pytest.raises(ValidationError):
        IndustryRoutingPolicy(minimum_candidate_score=math.nan)


def test_policy_rejects_positive_infinity() -> None:
    with pytest.raises(ValidationError):
        IndustryRoutingPolicy(minimum_score_margin=math.inf)


def test_policy_rejects_negative_infinity() -> None:
    with pytest.raises(ValidationError):
        IndustryRoutingPolicy(minimum_auto_confidence=-math.inf)


# --- IndustryRoutingResult ---


def test_result_auto_selected() -> None:
    result = _result(
        "AUTO_SELECTED",
        requires_user_confirmation=False,
        user_confirmed=False,
    )
    assert result.decision == "AUTO_SELECTED"


def test_result_confirmation_required() -> None:
    result = _result(
        "CONFIRMATION_REQUIRED",
        requires_user_confirmation=True,
        user_confirmed=False,
    )
    assert result.decision == "CONFIRMATION_REQUIRED"


def test_result_fallback_generic() -> None:
    result = _result(
        "FALLBACK_GENERIC",
        selected_industry="generic",
        selected_score=_score("generic", score=0.0, confidence=0.0),
        requires_user_confirmation=True,
        user_confirmed=False,
    )
    assert result.decision == "FALLBACK_GENERIC"


def test_result_user_confirmed() -> None:
    result = _result(
        "USER_CONFIRMED",
        requires_user_confirmation=False,
        user_confirmed=True,
    )
    assert result.decision == "USER_CONFIRMED"
    assert result.user_confirmed is True


def test_result_rejects_blank_selected_industry() -> None:
    with pytest.raises(ValidationError):
        _result(
            "AUTO_SELECTED",
            selected_industry="   ",
            requires_user_confirmation=False,
        )


def test_result_rejects_blank_reason() -> None:
    with pytest.raises(ValidationError):
        _result(
            "AUTO_SELECTED",
            requires_user_confirmation=False,
            reason="",
        )


def test_result_rejects_negative_score_margin() -> None:
    with pytest.raises(ValidationError):
        _result(
            "AUTO_SELECTED",
            requires_user_confirmation=False,
            score_margin=-0.01,
        )


def test_result_rejects_decision_requires_confirmation_mismatch() -> None:
    with pytest.raises(ValidationError):
        _result(
            "AUTO_SELECTED",
            requires_user_confirmation=True,
            user_confirmed=False,
        )
    with pytest.raises(ValidationError):
        _result(
            "CONFIRMATION_REQUIRED",
            requires_user_confirmation=False,
            user_confirmed=False,
        )


def test_result_rejects_decision_user_confirmed_mismatch() -> None:
    with pytest.raises(ValidationError):
        _result(
            "AUTO_SELECTED",
            requires_user_confirmation=False,
            user_confirmed=True,
        )
    with pytest.raises(ValidationError):
        _result(
            "USER_CONFIRMED",
            requires_user_confirmation=False,
            user_confirmed=False,
        )


def test_result_list_defaults_are_not_shared() -> None:
    first = _result("AUTO_SELECTED", requires_user_confirmation=False)
    second = _result("AUTO_SELECTED", requires_user_confirmation=False)
    first.ranked_candidates.append(_score("beta"))
    assert second.ranked_candidates == []
    first.alternatives.append(_score("gamma"))
    assert second.alternatives == []


def test_result_model_dump_validate_round_trip() -> None:
    original = _result(
        "CONFIRMATION_REQUIRED",
        requires_user_confirmation=True,
        score_margin=1.5,
        ranked_candidates=[_score("alpha"), _score("beta", score=3.0)],
        alternatives=[_score("beta", score=3.0)],
    )
    restored = IndustryRoutingResult.model_validate(original.model_dump())
    assert restored == original


# --- Router construction ---


def test_router_creation() -> None:
    router = _router(StubIndustryProfile("alpha", score=5.0, confidence=0.9))
    assert isinstance(router, IndustryRouter)


def test_router_rejects_invalid_registry_type() -> None:
    with pytest.raises(TypeError, match="IndustryRegistry"):
        IndustryRouter("not-a-registry")  # type: ignore[arg-type]


def test_router_rejects_invalid_policy_type() -> None:
    with pytest.raises(TypeError, match="IndustryRoutingPolicy"):
        IndustryRouter(
            IndustryRegistry(),
            policy="bad",  # type: ignore[arg-type]
        )


def test_router_rejects_invalid_fallback_type() -> None:
    with pytest.raises(TypeError, match="BaseIndustryProfile"):
        IndustryRouter(
            IndustryRegistry(),
            fallback_profile="bad",  # type: ignore[arg-type]
        )


def test_router_rejects_blank_fallback_industry_name() -> None:
    with pytest.raises(DataValidationError, match="industry_name"):
        IndustryRouter(
            IndustryRegistry(),
            fallback_profile=StubIndustryProfile("   "),
        )


def test_router_rejects_fallback_name_collision() -> None:
    with pytest.raises(DataValidationError, match="collides"):
        IndustryRouter(
            IndustryRegistry(profiles=[StubIndustryProfile("generic")]),
            fallback_profile=GenericIndustryProfile(),
        )


def test_external_registry_change_does_not_affect_router() -> None:
    registry = IndustryRegistry(profiles=[StubIndustryProfile("alpha")])
    router = IndustryRouter(registry)
    registry.register(StubIndustryProfile("beta"))
    assert router.list_candidate_industries() == ("alpha",)
    assert registry.list_names() == ("alpha", "beta")


def test_external_policy_change_does_not_affect_router() -> None:
    policy = IndustryRoutingPolicy(minimum_auto_confidence=0.9)
    router = _router(
        StubIndustryProfile("alpha", score=5.0, confidence=0.7),
        policy=policy,
    )
    policy.minimum_auto_confidence = 0.1
    result = router.route(_metadata())
    assert result.decision == "CONFIRMATION_REQUIRED"
    assert "confidence" in result.reason.lower()


def test_list_candidate_industries_returns_tuple() -> None:
    router = _router(
        StubIndustryProfile("alpha"),
        StubIndustryProfile("beta"),
    )
    names = router.list_candidate_industries()
    assert isinstance(names, tuple)


def test_list_candidate_industries_preserves_registration_order() -> None:
    router = _router(
        StubIndustryProfile("alpha"),
        StubIndustryProfile("beta"),
        StubIndustryProfile("gamma"),
    )
    assert router.list_candidate_industries() == ("alpha", "beta", "gamma")


# --- route input validation ---


def test_route_rejects_invalid_metadata_type() -> None:
    router = _router(StubIndustryProfile("alpha"))
    with pytest.raises(TypeError, match="DatasetMetadata"):
        router.route("not-metadata")  # type: ignore[arg-type]


def test_route_rejects_non_string_confirmed_industry() -> None:
    router = _router(StubIndustryProfile("alpha"))
    with pytest.raises(TypeError, match="confirmed_industry"):
        router.route(_metadata(), confirmed_industry=123)  # type: ignore[arg-type]


def test_route_rejects_empty_confirmed_industry() -> None:
    router = _router(StubIndustryProfile("alpha"))
    with pytest.raises(ValueError, match="confirmed_industry"):
        router.route(_metadata(), confirmed_industry="")


def test_route_rejects_whitespace_confirmed_industry() -> None:
    router = _router(StubIndustryProfile("alpha"))
    with pytest.raises(ValueError, match="confirmed_industry"):
        router.route(_metadata(), confirmed_industry="   ")


def test_route_rejects_unknown_confirmed_industry() -> None:
    router = _router(StubIndustryProfile("alpha"))
    with pytest.raises(DataValidationError, match="unknown-industry"):
        router.route(_metadata(), confirmed_industry="unknown-industry")


def test_route_unknown_confirmed_industry_message_includes_name() -> None:
    router = _router(StubIndustryProfile("alpha"))
    with pytest.raises(DataValidationError, match="missing-name"):
        router.route(_metadata(), confirmed_industry="missing-name")


def test_route_does_not_mutate_metadata() -> None:
    metadata = _metadata(column_names=["a", "b"], dtypes={"a": "Float64", "b": "Utf8"})
    before = metadata.model_dump()
    router = _router(StubIndustryProfile("alpha", score=5.0, confidence=0.9))
    router.route(metadata)
    assert metadata.model_dump() == before


# --- candidate scoring and ranking ---


def test_rank_by_score_descending() -> None:
    router = _router(
        StubIndustryProfile("low", score=2.0, confidence=0.9),
        StubIndustryProfile("high", score=8.0, confidence=0.5),
    )
    result = router.route(_metadata())
    assert [item.industry_name for item in result.ranked_candidates] == [
        "high",
        "low",
    ]


def test_rank_ties_prefer_higher_confidence() -> None:
    router = _router(
        StubIndustryProfile("low_conf", score=5.0, confidence=0.4),
        StubIndustryProfile("high_conf", score=5.0, confidence=0.9),
    )
    result = router.route(_metadata())
    assert [item.industry_name for item in result.ranked_candidates] == [
        "high_conf",
        "low_conf",
    ]


def test_rank_ties_prefer_registration_order() -> None:
    router = _router(
        StubIndustryProfile("first", score=5.0, confidence=0.8),
        StubIndustryProfile("second", score=5.0, confidence=0.8),
    )
    result = router.route(_metadata())
    assert [item.industry_name for item in result.ranked_candidates] == [
        "first",
        "second",
    ]


def test_rank_order_is_deterministic() -> None:
    registry = IndustryRegistry(
        profiles=[
            StubIndustryProfile("a", score=4.0, confidence=0.7),
            StubIndustryProfile("b", score=6.0, confidence=0.5),
            StubIndustryProfile("c", score=6.0, confidence=0.9),
        ]
    )
    router = IndustryRouter(registry)
    first = [item.industry_name for item in router.route(_metadata()).ranked_candidates]
    second = [item.industry_name for item in router.route(_metadata()).ranked_candidates]
    assert first == second == ["c", "b", "a"]


def test_registry_registration_order_unchanged_after_route() -> None:
    registry = IndustryRegistry(
        profiles=[
            StubIndustryProfile("alpha", score=1.0, confidence=0.9),
            StubIndustryProfile("beta", score=9.0, confidence=0.9),
        ]
    )
    router = IndustryRouter(registry)
    router.route(_metadata())
    assert registry.list_names() == ("alpha", "beta")


def test_score_margin_calculation() -> None:
    router = _router(
        StubIndustryProfile("alpha", score=7.12345, confidence=0.9),
        StubIndustryProfile("beta", score=3.0, confidence=0.9),
    )
    result = router.route(_metadata())
    assert result.score_margin == pytest.approx(4.1235)


def test_score_margin_none_for_single_candidate() -> None:
    router = _router(StubIndustryProfile("alpha", score=5.0, confidence=0.9))
    result = router.route(_metadata())
    assert result.score_margin is None


def test_score_margin_none_for_empty_candidates() -> None:
    router = IndustryRouter(IndustryRegistry())
    result = router.route(_metadata())
    assert result.decision == "FALLBACK_GENERIC"
    assert result.score_margin is None


def test_rejects_negative_profile_score() -> None:
    router = _router(StubIndustryProfile("alpha", score=-1.0, confidence=0.9))
    with pytest.raises(DataValidationError, match="alpha"):
        router.route(_metadata())


def test_rejects_nan_profile_score() -> None:
    router = _router(
        StubIndustryProfile(
            "alpha",
            score_result=_score("alpha", score=math.nan, confidence=0.9),
        )
    )
    with pytest.raises(DataValidationError):
        router.route(_metadata())


def test_rejects_infinity_profile_score() -> None:
    router = _router(
        StubIndustryProfile(
            "alpha",
            score_result=_score("alpha", score=math.inf, confidence=0.9),
        )
    )
    with pytest.raises(DataValidationError):
        router.route(_metadata())


def test_rejects_non_industry_score_result() -> None:
    router = _router(
        StubIndustryProfile("alpha", score_result={"not": "an IndustryScore"})
    )
    with pytest.raises(DataValidationError, match="IndustryScore"):
        router.route(_metadata())


# --- automatic routing ---


def test_empty_registry_falls_back_generic() -> None:
    result = IndustryRouter(IndustryRegistry()).route(_metadata())
    assert result.decision == "FALLBACK_GENERIC"
    assert result.selected_industry == "generic"
    assert result.ranked_candidates == []
    assert result.alternatives == []


def test_all_zero_scores_fall_back_generic() -> None:
    router = _router(
        StubIndustryProfile("alpha", score=0.0, confidence=0.0),
        StubIndustryProfile("beta", score=0.0, confidence=0.0),
    )
    result = router.route(_metadata())
    assert result.decision == "FALLBACK_GENERIC"
    assert result.selected_industry == "generic"
    assert len(result.ranked_candidates) == 2
    assert len(result.alternatives) == 2


def test_below_minimum_candidate_score_falls_back() -> None:
    policy = IndustryRoutingPolicy(minimum_candidate_score=5.0)
    router = _router(
        StubIndustryProfile("alpha", score=2.0, confidence=0.9),
        policy=policy,
    )
    result = router.route(_metadata())
    assert result.decision == "FALLBACK_GENERIC"
    assert "score" in result.reason.lower() or "candidate" in result.reason.lower()


def test_fallback_requires_user_confirmation() -> None:
    result = IndustryRouter(IndustryRegistry()).route(_metadata())
    assert result.requires_user_confirmation is True
    assert result.user_confirmed is False


def test_low_confidence_requires_confirmation() -> None:
    policy = IndustryRoutingPolicy(minimum_auto_confidence=0.8)
    router = _router(
        StubIndustryProfile("alpha", score=5.0, confidence=0.5),
        policy=policy,
    )
    result = router.route(_metadata())
    assert result.decision == "CONFIRMATION_REQUIRED"
    assert result.selected_industry == "alpha"
    assert "confidence" in result.reason.lower()


def test_profile_requires_confirmation_flag() -> None:
    policy = IndustryRoutingPolicy(
        minimum_auto_confidence=0.5,
        minimum_score_margin=0.0,
    )
    router = _router(
        StubIndustryProfile(
            "alpha",
            score=5.0,
            confidence=0.9,
            requires_user_confirmation=True,
        ),
        policy=policy,
    )
    result = router.route(_metadata())
    assert result.decision == "CONFIRMATION_REQUIRED"
    assert "confirmation" in result.reason.lower() or "profile" in result.reason.lower()


def test_small_score_margin_requires_confirmation() -> None:
    policy = IndustryRoutingPolicy(
        minimum_auto_confidence=0.5,
        minimum_score_margin=3.0,
    )
    router = _router(
        StubIndustryProfile("alpha", score=5.0, confidence=0.9),
        StubIndustryProfile("beta", score=3.5, confidence=0.9),
        policy=policy,
    )
    result = router.route(_metadata())
    assert result.decision == "CONFIRMATION_REQUIRED"
    assert result.score_margin == pytest.approx(1.5)
    assert "margin" in result.reason.lower() or "difference" in result.reason.lower()


def test_score_margin_equal_to_minimum_passes() -> None:
    policy = IndustryRoutingPolicy(
        minimum_auto_confidence=0.5,
        minimum_score_margin=2.0,
    )
    router = _router(
        StubIndustryProfile("alpha", score=5.0, confidence=0.9),
        StubIndustryProfile("beta", score=3.0, confidence=0.9),
        policy=policy,
    )
    result = router.route(_metadata())
    assert result.decision == "AUTO_SELECTED"
    assert result.score_margin == pytest.approx(2.0)


def test_all_criteria_met_auto_selected() -> None:
    policy = IndustryRoutingPolicy(
        minimum_candidate_score=1.0,
        minimum_auto_confidence=0.6,
        minimum_score_margin=2.0,
    )
    router = _router(
        StubIndustryProfile("alpha", score=8.0, confidence=0.9),
        StubIndustryProfile("beta", score=3.0, confidence=0.4),
        policy=policy,
    )
    result = router.route(_metadata())
    assert result.decision == "AUTO_SELECTED"


def test_auto_selected_does_not_require_confirmation() -> None:
    router = _router(
        StubIndustryProfile("alpha", score=8.0, confidence=0.9),
        StubIndustryProfile("beta", score=3.0, confidence=0.4),
        policy=IndustryRoutingPolicy(minimum_auto_confidence=0.5),
    )
    result = router.route(_metadata())
    assert result.decision == "AUTO_SELECTED"
    assert result.requires_user_confirmation is False
    assert result.user_confirmed is False


def test_auto_selected_chooses_top_candidate() -> None:
    router = _router(
        StubIndustryProfile("alpha", score=3.0, confidence=0.9),
        StubIndustryProfile("beta", score=9.0, confidence=0.9),
        policy=IndustryRoutingPolicy(minimum_auto_confidence=0.5, minimum_score_margin=1.0),
    )
    result = router.route(_metadata())
    assert result.decision == "AUTO_SELECTED"
    assert result.selected_industry == "beta"


def test_confirmation_required_uses_top_as_provisional() -> None:
    router = _router(
        StubIndustryProfile("alpha", score=5.0, confidence=0.2),
        StubIndustryProfile("beta", score=1.0, confidence=0.1),
    )
    result = router.route(_metadata())
    assert result.decision == "CONFIRMATION_REQUIRED"
    assert result.selected_industry == "alpha"


def test_alternatives_exclude_selected_candidate() -> None:
    router = _router(
        StubIndustryProfile("alpha", score=8.0, confidence=0.9),
        StubIndustryProfile("beta", score=3.0, confidence=0.4),
        policy=IndustryRoutingPolicy(minimum_auto_confidence=0.5),
    )
    result = router.route(_metadata())
    assert [item.industry_name for item in result.alternatives] == ["beta"]
    assert all(item.industry_name != result.selected_industry for item in result.alternatives)


def test_fallback_includes_candidates_in_alternatives() -> None:
    router = _router(
        StubIndustryProfile("alpha", score=0.0, confidence=0.0),
        StubIndustryProfile("beta", score=0.0, confidence=0.0),
    )
    result = router.route(_metadata())
    assert result.decision == "FALLBACK_GENERIC"
    assert {item.industry_name for item in result.alternatives} == {"alpha", "beta"}


def test_confidence_reason_priority_over_margin() -> None:
    policy = IndustryRoutingPolicy(
        minimum_auto_confidence=0.9,
        minimum_score_margin=10.0,
    )
    router = _router(
        StubIndustryProfile("alpha", score=5.0, confidence=0.5),
        StubIndustryProfile("beta", score=4.5, confidence=0.5),
        policy=policy,
    )
    result = router.route(_metadata())
    assert result.decision == "CONFIRMATION_REQUIRED"
    assert "confidence" in result.reason.lower()


def test_profile_confirmation_reason_priority_over_margin() -> None:
    policy = IndustryRoutingPolicy(
        minimum_auto_confidence=0.5,
        minimum_score_margin=10.0,
    )
    router = _router(
        StubIndustryProfile(
            "alpha",
            score=5.0,
            confidence=0.9,
            requires_user_confirmation=True,
        ),
        StubIndustryProfile("beta", score=4.5, confidence=0.9),
        policy=policy,
    )
    result = router.route(_metadata())
    assert result.decision == "CONFIRMATION_REQUIRED"
    assert "margin" not in result.reason.lower()


# --- user confirmation ---


def test_user_confirmed_decision() -> None:
    router = _router(StubIndustryProfile("alpha", score=5.0, confidence=0.9))
    result = router.route(_metadata(), confirmed_industry="alpha")
    assert result.decision == "USER_CONFIRMED"


def test_user_confirmed_does_not_require_confirmation() -> None:
    router = _router(StubIndustryProfile("alpha", score=5.0, confidence=0.9))
    result = router.route(_metadata(), confirmed_industry="alpha")
    assert result.requires_user_confirmation is False


def test_user_confirmed_sets_user_confirmed_true() -> None:
    router = _router(StubIndustryProfile("alpha", score=5.0, confidence=0.9))
    result = router.route(_metadata(), confirmed_industry="alpha")
    assert result.user_confirmed is True


def test_user_confirmed_ignores_case() -> None:
    router = _router(StubIndustryProfile("alpha", score=5.0, confidence=0.9))
    result = router.route(_metadata(), confirmed_industry="ALPHA")
    assert result.selected_industry == "alpha"
    assert result.decision == "USER_CONFIRMED"


def test_user_confirmed_ignores_surrounding_whitespace() -> None:
    router = _router(StubIndustryProfile("alpha", score=5.0, confidence=0.9))
    result = router.route(_metadata(), confirmed_industry="  alpha  ")
    assert result.selected_industry == "alpha"


def test_user_can_confirm_non_top_candidate() -> None:
    router = _router(
        StubIndustryProfile("alpha", score=9.0, confidence=0.9),
        StubIndustryProfile("beta", score=2.0, confidence=0.4),
    )
    result = router.route(_metadata(), confirmed_industry="beta")
    assert result.decision == "USER_CONFIRMED"
    assert result.selected_industry == "beta"
    assert result.selected_score.industry_name == "beta"


def test_user_can_confirm_fallback_generic() -> None:
    router = _router(StubIndustryProfile("alpha", score=9.0, confidence=0.9))
    result = router.route(_metadata(), confirmed_industry="generic")
    assert result.decision == "USER_CONFIRMED"
    assert result.selected_industry == "generic"
    assert result.selected_score.industry_name == "generic"
    assert [item.industry_name for item in result.alternatives] == ["alpha"]


def test_user_confirmed_alternatives_exclude_selected() -> None:
    router = _router(
        StubIndustryProfile("alpha", score=9.0, confidence=0.9),
        StubIndustryProfile("beta", score=2.0, confidence=0.4),
    )
    result = router.route(_metadata(), confirmed_industry="beta")
    assert [item.industry_name for item in result.alternatives] == ["alpha"]


def test_user_confirmed_keeps_full_ranked_candidates() -> None:
    router = _router(
        StubIndustryProfile("alpha", score=9.0, confidence=0.9),
        StubIndustryProfile("beta", score=2.0, confidence=0.4),
    )
    result = router.route(_metadata(), confirmed_industry="beta")
    assert [item.industry_name for item in result.ranked_candidates] == [
        "alpha",
        "beta",
    ]


# --- default router ---


def test_create_default_industry_router_returns_router() -> None:
    router = create_default_industry_router()
    assert isinstance(router, IndustryRouter)


def test_default_router_candidate_order() -> None:
    router = create_default_industry_router()
    assert router.list_candidate_industries() == (
        "semiconductor",
        "battery",
        "automotive",
    )


def test_default_router_fallback_is_generic() -> None:
    router = create_default_industry_router()
    result = router.route(_metadata(), confirmed_industry="generic")
    assert result.selected_industry == "generic"
    assert result.decision == "USER_CONFIRMED"


def test_default_router_wafer_metadata_selects_or_proposes_semiconductor() -> None:
    router = create_default_industry_router()
    metadata = _metadata(
        file_name="wafer_lot.csv",
        user_description="semiconductor fab etch deposition",
        column_names=["wafer_id", "lot_id", "chamber_id"],
        dtypes={
            "wafer_id": "Utf8",
            "lot_id": "Utf8",
            "chamber_id": "Utf8",
        },
    )
    result = router.route(metadata)
    assert result.decision in {"AUTO_SELECTED", "CONFIRMATION_REQUIRED"}
    assert result.selected_industry == "semiconductor"


def test_default_router_battery_metadata_selects_or_proposes_battery() -> None:
    router = create_default_industry_router()
    metadata = _metadata(
        file_name="battery_cell.csv",
        user_description="battery cell discharge_capacity state_of_health",
        column_names=["cell_id", "cycle_index", "soc"],
        dtypes={
            "cell_id": "Utf8",
            "cycle_index": "Int64",
            "soc": "Float64",
        },
    )
    result = router.route(metadata)
    assert result.decision in {"AUTO_SELECTED", "CONFIRMATION_REQUIRED"}
    assert result.selected_industry == "battery"


def test_default_router_vehicle_metadata_selects_or_proposes_automotive() -> None:
    router = create_default_industry_router()
    metadata = _metadata(
        file_name="vehicle_engine.csv",
        user_description="automotive vehicle_speed engine_rpm throttle_position",
        column_names=["vehicle_id", "vin", "brake_pressure"],
        dtypes={
            "vehicle_id": "Utf8",
            "vin": "Utf8",
            "brake_pressure": "Float64",
        },
    )
    result = router.route(metadata)
    assert result.decision in {"AUTO_SELECTED", "CONFIRMATION_REQUIRED"}
    assert result.selected_industry == "automotive"


def test_default_router_no_signal_falls_back_generic() -> None:
    router = create_default_industry_router()
    result = router.route(_metadata())
    assert result.decision == "FALLBACK_GENERIC"
    assert result.selected_industry == "generic"


def test_create_default_industry_router_is_independent() -> None:
    first = create_default_industry_router()
    second = create_default_industry_router()
    assert first is not second
    assert first.list_candidate_industries() == second.list_candidate_industries()


def test_default_routers_do_not_share_registry_state() -> None:
    first = create_default_industry_router()
    second = create_default_industry_router()
    # Mutating via private snapshot would be incorrect API usage; instead verify
    # that route results from one router do not affect the other.
    metadata = _metadata(
        file_name="wafer_lot.csv",
        user_description="semiconductor fab etch deposition",
        column_names=["wafer_id", "lot_id", "chamber_id"],
        dtypes={
            "wafer_id": "Utf8",
            "lot_id": "Utf8",
            "chamber_id": "Utf8",
        },
    )
    first_result = first.route(metadata)
    second_result = second.route(_metadata())
    assert first_result.selected_industry == "semiconductor"
    assert second_result.decision == "FALLBACK_GENERIC"
    assert second.list_candidate_industries() == (
        "semiconductor",
        "battery",
        "automotive",
    )


def test_create_default_registry_still_generic_only() -> None:
    registry = create_default_registry()
    assert registry.list_names() == ("generic",)


# --- immutability and regression ---


def test_route_does_not_mutate_input_registry() -> None:
    registry = IndustryRegistry(
        profiles=[
            StubIndustryProfile("alpha", score=5.0, confidence=0.9),
            StubIndustryProfile("beta", score=2.0, confidence=0.4),
        ]
    )
    before = registry.list_names()
    router = IndustryRouter(registry)
    router.route(_metadata())
    assert registry.list_names() == before


def test_route_does_not_mutate_input_profiles() -> None:
    profile = StubIndustryProfile("alpha", score=5.0, confidence=0.9)
    before_name = profile.industry_name
    before_score = profile._score
    router = IndustryRouter(IndustryRegistry(profiles=[profile]))
    router.route(_metadata())
    assert profile.industry_name == before_name
    assert profile._score == before_score


def test_consecutive_routes_do_not_accumulate() -> None:
    router = _router(
        StubIndustryProfile("alpha", score=8.0, confidence=0.9),
        StubIndustryProfile("beta", score=1.0, confidence=0.2),
        policy=IndustryRoutingPolicy(minimum_auto_confidence=0.5),
    )
    first = router.route(_metadata())
    second = router.route(_metadata())
    assert len(first.ranked_candidates) == 2
    assert len(second.ranked_candidates) == 2
    assert first.model_dump() == second.model_dump()


def test_separate_router_instances_do_not_share_state() -> None:
    registry_a = IndustryRegistry(
        profiles=[StubIndustryProfile("alpha", score=8.0, confidence=0.9)]
    )
    registry_b = IndustryRegistry(
        profiles=[StubIndustryProfile("beta", score=8.0, confidence=0.9)]
    )
    router_a = IndustryRouter(
        registry_a,
        policy=IndustryRoutingPolicy(minimum_auto_confidence=0.5),
    )
    router_b = IndustryRouter(
        registry_b,
        policy=IndustryRoutingPolicy(minimum_auto_confidence=0.5),
    )
    result_a = router_a.route(_metadata())
    result_b = router_b.route(_metadata())
    assert result_a.selected_industry == "alpha"
    assert result_b.selected_industry == "beta"


def test_same_input_returns_deterministic_result() -> None:
    router = _router(
        StubIndustryProfile("alpha", score=4.0, confidence=0.7),
        StubIndustryProfile("beta", score=6.0, confidence=0.5),
        StubIndustryProfile("gamma", score=6.0, confidence=0.9),
        policy=IndustryRoutingPolicy(minimum_auto_confidence=0.5, minimum_score_margin=0.0),
    )
    metadata = _metadata()
    first = router.route(metadata)
    second = router.route(metadata)
    assert first.model_dump() == second.model_dump()
