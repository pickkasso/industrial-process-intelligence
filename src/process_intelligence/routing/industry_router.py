"""Industry routing policy, result models, and IndustryRouter."""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from process_intelligence.core.exceptions import DataValidationError
from process_intelligence.core.protocols import BaseIndustryProfile
from process_intelligence.core.schemas import DatasetMetadata, IndustryScore
from process_intelligence.industries.automotive import AutomotiveIndustryProfile
from process_intelligence.industries.battery import BatteryIndustryProfile
from process_intelligence.industries.generic import GenericIndustryProfile
from process_intelligence.industries.registry import IndustryRegistry
from process_intelligence.industries.semiconductor import SemiconductorIndustryProfile

DecisionLiteral = Literal[
    "AUTO_SELECTED",
    "CONFIRMATION_REQUIRED",
    "FALLBACK_GENERIC",
    "USER_CONFIRMED",
]


def _normalize_industry_name(industry_name: str) -> str:
    """Normalize an industry name for case-insensitive lookup."""
    return industry_name.strip().casefold()


def _validate_non_negative_finite(value: object, *, field_name: str) -> float:
    """Reject bools, non-numbers, negatives, NaN, and infinities."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a non-negative finite number "
            f"(bool not allowed), got {value!r}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(
            f"{field_name} must be a non-negative finite number, got {value!r}"
        )
    if number < 0.0:
        raise ValueError(f"{field_name} must be >= 0, got {number}")
    return number


def _validate_score_value(score: float, *, industry_name: str) -> None:
    """Ensure an industry score is a finite non-negative number."""
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise DataValidationError(
            f"Industry score for {industry_name!r} must be a finite number, "
            f"got {score!r}"
        )
    number = float(score)
    if not math.isfinite(number):
        raise DataValidationError(
            f"Industry score for {industry_name!r} must be a finite number, "
            f"got {score!r}"
        )
    if number < 0.0:
        raise DataValidationError(
            f"Industry score for {industry_name!r} must be >= 0, got {number}"
        )


class IndustryRoutingPolicy(BaseModel):
    """Thresholds that control automatic industry selection vs confirmation."""

    minimum_candidate_score: float = 1.0
    minimum_auto_confidence: float = 0.60
    minimum_score_margin: float = 2.0

    @field_validator("minimum_candidate_score", mode="before")
    @classmethod
    def _validate_minimum_candidate_score(cls, value: object) -> float:
        return _validate_non_negative_finite(
            value, field_name="minimum_candidate_score"
        )

    @field_validator("minimum_auto_confidence", mode="before")
    @classmethod
    def _validate_minimum_auto_confidence(cls, value: object) -> float:
        number = _validate_non_negative_finite(
            value, field_name="minimum_auto_confidence"
        )
        if number > 1.0:
            raise ValueError(
                f"minimum_auto_confidence must be <= 1, got {number}"
            )
        return number

    @field_validator("minimum_score_margin", mode="before")
    @classmethod
    def _validate_minimum_score_margin(cls, value: object) -> float:
        return _validate_non_negative_finite(
            value, field_name="minimum_score_margin"
        )


class IndustryRoutingResult(BaseModel):
    """Outcome of industry routing, including ranked candidates and decision."""

    decision: DecisionLiteral
    selected_industry: str
    selected_score: IndustryScore
    ranked_candidates: list[IndustryScore] = Field(default_factory=list)
    alternatives: list[IndustryScore] = Field(default_factory=list)
    requires_user_confirmation: bool
    reason: str
    score_margin: float | None = None
    user_confirmed: bool = False

    @field_validator("selected_industry", "reason", mode="after")
    @classmethod
    def _reject_blank_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty or whitespace-only")
        return value

    @field_validator("score_margin", mode="after")
    @classmethod
    def _validate_score_margin(cls, value: float | None) -> float | None:
        if value is not None and value < 0.0:
            raise ValueError(f"score_margin must be >= 0, got {value}")
        return value

    @model_validator(mode="after")
    def _validate_decision_flags(self) -> Self:
        """Enforce consistency between decision and confirmation flags."""
        if self.decision == "AUTO_SELECTED":
            if self.requires_user_confirmation or self.user_confirmed:
                raise ValueError(
                    "AUTO_SELECTED requires requires_user_confirmation=False "
                    "and user_confirmed=False"
                )
        elif self.decision == "CONFIRMATION_REQUIRED":
            if not self.requires_user_confirmation or self.user_confirmed:
                raise ValueError(
                    "CONFIRMATION_REQUIRED requires "
                    "requires_user_confirmation=True and user_confirmed=False"
                )
        elif self.decision == "FALLBACK_GENERIC":
            if not self.requires_user_confirmation or self.user_confirmed:
                raise ValueError(
                    "FALLBACK_GENERIC requires "
                    "requires_user_confirmation=True and user_confirmed=False"
                )
        elif self.decision == "USER_CONFIRMED":
            if self.requires_user_confirmation or not self.user_confirmed:
                raise ValueError(
                    "USER_CONFIRMED requires requires_user_confirmation=False "
                    "and user_confirmed=True"
                )
        return self


class IndustryRouter:
    """Select an industry from scored profiles using an explicit routing policy."""

    def __init__(
        self,
        registry: IndustryRegistry,
        *,
        policy: IndustryRoutingPolicy | None = None,
        fallback_profile: BaseIndustryProfile | None = None,
    ) -> None:
        """Create a router with a snapshot of the candidate registry.

        Args:
            registry: Candidate industry profiles. A snapshot is stored; the
                input registry is not modified.
            policy: Routing thresholds. When ``None``, defaults are used. The
                provided policy is deep-copied.
            fallback_profile: Profile used when no industry candidate is
                selected. When ``None``, ``GenericIndustryProfile`` is used.

        Raises:
            TypeError: If ``registry``, ``policy``, or ``fallback_profile`` has
                an invalid type.
            DataValidationError: If the fallback industry name is empty or
                collides with a candidate name.
        """
        if not isinstance(registry, IndustryRegistry):
            raise TypeError(
                f"registry must be IndustryRegistry, got {type(registry).__name__}"
            )

        if policy is None:
            self._policy = IndustryRoutingPolicy()
        elif not isinstance(policy, IndustryRoutingPolicy):
            raise TypeError(
                f"policy must be IndustryRoutingPolicy, "
                f"got {type(policy).__name__}"
            )
        else:
            self._policy = policy.model_copy(deep=True)

        if fallback_profile is None:
            resolved_fallback: BaseIndustryProfile = GenericIndustryProfile()
        elif not isinstance(fallback_profile, BaseIndustryProfile):
            raise TypeError(
                f"fallback_profile must be a BaseIndustryProfile instance, "
                f"got {type(fallback_profile).__name__}"
            )
        else:
            resolved_fallback = fallback_profile

        fallback_name = resolved_fallback.industry_name
        if not isinstance(fallback_name, str) or not fallback_name.strip():
            raise DataValidationError(
                "fallback_profile.industry_name must not be empty or "
                "whitespace-only"
            )

        snapshot = IndustryRegistry(profiles=registry.list_profiles())
        fallback_key = _normalize_industry_name(fallback_name)
        for profile in snapshot.list_profiles():
            if _normalize_industry_name(profile.industry_name) == fallback_key:
                raise DataValidationError(
                    f"fallback_profile industry name "
                    f"{fallback_name!r} collides with a candidate profile"
                )

        self._registry = snapshot
        self._fallback_profile = resolved_fallback

    def list_candidate_industries(self) -> tuple[str, ...]:
        """Return candidate industry names in registration order."""
        return self._registry.list_names()

    def route(
        self,
        metadata: DatasetMetadata,
        *,
        confirmed_industry: str | None = None,
    ) -> IndustryRoutingResult:
        """Route dataset metadata to an industry selection decision.

        Args:
            metadata: Dataset metadata used for industry scoring. Not modified.
            confirmed_industry: Optional user-confirmed industry name. Matching
                ignores case and surrounding whitespace.

        Returns:
            An ``IndustryRoutingResult`` describing the routing decision.

        Raises:
            TypeError: If ``metadata`` or ``confirmed_industry`` has an invalid
                type.
            ValueError: If ``confirmed_industry`` is empty or whitespace-only.
            DataValidationError: If confirmed industry is unknown, or if a
                profile returns an invalid score.
        """
        if not isinstance(metadata, DatasetMetadata):
            raise TypeError(
                f"metadata must be DatasetMetadata, got {type(metadata).__name__}"
            )

        if confirmed_industry is not None and not isinstance(confirmed_industry, str):
            raise TypeError(
                f"confirmed_industry must be str or None, "
                f"got {type(confirmed_industry).__name__}"
            )

        if confirmed_industry is not None:
            if not confirmed_industry.strip():
                raise ValueError(
                    "confirmed_industry must not be empty or whitespace-only"
                )
            return self._route_user_confirmed(metadata, confirmed_industry)

        return self._route_automatic(metadata)

    def _route_automatic(self, metadata: DatasetMetadata) -> IndustryRoutingResult:
        """Apply automatic routing when no industry has been confirmed."""
        candidate_scores = self._score_candidates(metadata)
        ranked_candidates = self._rank_candidates(candidate_scores)
        score_margin = self._compute_score_margin(ranked_candidates)

        if not ranked_candidates:
            fallback_score = self._score_fallback(metadata)
            return IndustryRoutingResult(
                decision="FALLBACK_GENERIC",
                selected_industry=self._fallback_profile.industry_name,
                selected_score=fallback_score,
                ranked_candidates=[],
                alternatives=[],
                requires_user_confirmation=True,
                reason=(
                    "No industry candidates are registered; "
                    "using the generic fallback profile."
                ),
                score_margin=None,
                user_confirmed=False,
            )

        top = ranked_candidates[0]

        if top.score == 0.0:
            fallback_score = self._score_fallback(metadata)
            return IndustryRoutingResult(
                decision="FALLBACK_GENERIC",
                selected_industry=self._fallback_profile.industry_name,
                selected_score=fallback_score,
                ranked_candidates=list(ranked_candidates),
                alternatives=list(ranked_candidates),
                requires_user_confirmation=True,
                reason=(
                    "No industry-specific signals were found in the dataset "
                    "metadata; using the generic fallback profile."
                ),
                score_margin=score_margin,
                user_confirmed=False,
            )

        if top.score < self._policy.minimum_candidate_score:
            fallback_score = self._score_fallback(metadata)
            return IndustryRoutingResult(
                decision="FALLBACK_GENERIC",
                selected_industry=self._fallback_profile.industry_name,
                selected_score=fallback_score,
                ranked_candidates=list(ranked_candidates),
                alternatives=list(ranked_candidates),
                requires_user_confirmation=True,
                reason=(
                    f"Top candidate score {top.score} is below the minimum "
                    f"candidate score {self._policy.minimum_candidate_score}."
                ),
                score_margin=score_margin,
                user_confirmed=False,
            )

        if top.confidence < self._policy.minimum_auto_confidence:
            return IndustryRoutingResult(
                decision="CONFIRMATION_REQUIRED",
                selected_industry=top.industry_name,
                selected_score=top,
                ranked_candidates=list(ranked_candidates),
                alternatives=list(ranked_candidates[1:]),
                requires_user_confirmation=True,
                reason=(
                    f"Top candidate confidence {top.confidence} is below the "
                    f"automatic confirmation threshold "
                    f"{self._policy.minimum_auto_confidence}."
                ),
                score_margin=score_margin,
                user_confirmed=False,
            )

        if top.requires_user_confirmation:
            return IndustryRoutingResult(
                decision="CONFIRMATION_REQUIRED",
                selected_industry=top.industry_name,
                selected_score=top,
                ranked_candidates=list(ranked_candidates),
                alternatives=list(ranked_candidates[1:]),
                requires_user_confirmation=True,
                reason=(
                    "The top industry profile itself requires user "
                    "confirmation before selection."
                ),
                score_margin=score_margin,
                user_confirmed=False,
            )

        if (
            len(ranked_candidates) >= 2
            and score_margin is not None
            and score_margin < self._policy.minimum_score_margin
        ):
            return IndustryRoutingResult(
                decision="CONFIRMATION_REQUIRED",
                selected_industry=top.industry_name,
                selected_score=top,
                ranked_candidates=list(ranked_candidates),
                alternatives=list(ranked_candidates[1:]),
                requires_user_confirmation=True,
                reason=(
                    f"Score margin {score_margin} between the top two "
                    f"candidates is below the minimum required margin "
                    f"{self._policy.minimum_score_margin}."
                ),
                score_margin=score_margin,
                user_confirmed=False,
            )

        return IndustryRoutingResult(
            decision="AUTO_SELECTED",
            selected_industry=top.industry_name,
            selected_score=top,
            ranked_candidates=list(ranked_candidates),
            alternatives=list(ranked_candidates[1:]),
            requires_user_confirmation=False,
            reason=(
                "Top candidate meets confidence and score-margin thresholds "
                "for automatic industry selection."
            ),
            score_margin=score_margin,
            user_confirmed=False,
        )

    def _route_user_confirmed(
        self,
        metadata: DatasetMetadata,
        confirmed_industry: str,
    ) -> IndustryRoutingResult:
        """Apply an explicitly user-confirmed industry selection."""
        profile, is_fallback = self._resolve_confirmed_industry(confirmed_industry)
        candidate_scores = self._score_candidates(metadata)
        ranked_candidates = self._rank_candidates(candidate_scores)
        score_margin = self._compute_score_margin(ranked_candidates)

        if is_fallback:
            selected_score = self._score_fallback(metadata)
            alternatives = list(ranked_candidates)
        else:
            selected_key = _normalize_industry_name(profile.industry_name)
            selected_score = next(
                score
                for score in ranked_candidates
                if _normalize_industry_name(score.industry_name) == selected_key
            )
            alternatives = [
                score
                for score in ranked_candidates
                if _normalize_industry_name(score.industry_name) != selected_key
            ]

        return IndustryRoutingResult(
            decision="USER_CONFIRMED",
            selected_industry=profile.industry_name,
            selected_score=selected_score,
            ranked_candidates=list(ranked_candidates),
            alternatives=alternatives,
            requires_user_confirmation=False,
            reason="The user explicitly confirmed the selected industry.",
            score_margin=score_margin,
            user_confirmed=True,
        )

    def _resolve_confirmed_industry(
        self,
        confirmed_industry: str,
    ) -> tuple[BaseIndustryProfile, bool]:
        """Resolve a confirmed industry name to a candidate or fallback profile."""
        key = _normalize_industry_name(confirmed_industry)
        for profile in self._registry.list_profiles():
            if _normalize_industry_name(profile.industry_name) == key:
                return profile, False

        if _normalize_industry_name(self._fallback_profile.industry_name) == key:
            return self._fallback_profile, True

        raise DataValidationError(
            f"Unknown confirmed industry {confirmed_industry!r}: "
            f"not found in candidate registry or fallback profile"
        )

    def _score_candidates(
        self,
        metadata: DatasetMetadata,
    ) -> tuple[IndustryScore, ...]:
        """Score all candidate profiles and validate returned scores."""
        scores = self._registry.score_all(metadata)
        for score in scores:
            _validate_score_value(score.score, industry_name=score.industry_name)
        return scores

    def _score_fallback(self, metadata: DatasetMetadata) -> IndustryScore:
        """Score the fallback profile and validate the returned IndustryScore."""
        score = self._fallback_profile.score_industry(metadata)
        if not isinstance(score, IndustryScore):
            raise DataValidationError(
                f"fallback_profile must return IndustryScore, "
                f"got {type(score).__name__}"
            )
        if not score.industry_name.strip():
            raise DataValidationError(
                "fallback_profile returned IndustryScore with an empty "
                "industry_name"
            )
        _validate_score_value(score.score, industry_name=score.industry_name)
        return score

    @staticmethod
    def _rank_candidates(
        scores: tuple[IndustryScore, ...],
    ) -> list[IndustryScore]:
        """Rank scores by score, confidence, then registration order."""
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda item: (-item[1].score, -item[1].confidence, item[0]))
        return [score for _, score in indexed]

    @staticmethod
    def _compute_score_margin(
        ranked_candidates: list[IndustryScore],
    ) -> float | None:
        """Return rounded score difference between first and second place."""
        if len(ranked_candidates) < 2:
            return None
        margin = ranked_candidates[0].score - ranked_candidates[1].score
        return round(margin, 4)


def create_default_industry_router(
    policy: IndustryRoutingPolicy | None = None,
) -> IndustryRouter:
    """Create a router with semiconductor, battery, and automotive candidates.

    Args:
        policy: Optional routing policy. Deep-copied by ``IndustryRouter``.

    Returns:
        A newly constructed ``IndustryRouter``. Each call returns an independent
        registry and router rather than a shared singleton.
    """
    registry = IndustryRegistry(
        profiles=[
            SemiconductorIndustryProfile(),
            BatteryIndustryProfile(),
            AutomotiveIndustryProfile(),
        ]
    )
    return IndustryRouter(
        registry,
        policy=policy,
        fallback_profile=GenericIndustryProfile(),
    )
