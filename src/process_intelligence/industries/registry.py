"""In-memory registry for BaseIndustryProfile implementations."""

from __future__ import annotations

from collections.abc import Sequence

from process_intelligence.core.exceptions import DataValidationError
from process_intelligence.core.protocols import BaseIndustryProfile
from process_intelligence.core.schemas import DatasetMetadata, IndustryScore


def _normalize_industry_name(industry_name: str) -> str:
    """Normalize an industry name for registry key lookup."""
    return industry_name.strip().casefold()


def _validate_lookup_name(industry_name: object) -> str:
    """Validate a caller-provided industry name used for lookup operations."""
    if not isinstance(industry_name, str):
        raise TypeError(
            f"industry_name must be str, got {type(industry_name).__name__}"
        )

    if not industry_name.strip():
        raise ValueError("industry_name must not be empty or whitespace-only")

    return industry_name


def _validate_profile(profile: object) -> tuple[str, BaseIndustryProfile]:
    """Validate a profile and return its normalized key with the profile."""
    if not isinstance(profile, BaseIndustryProfile):
        raise TypeError(
            f"profile must be a BaseIndustryProfile instance, "
            f"got {type(profile).__name__}"
        )

    industry_name = profile.industry_name
    if not isinstance(industry_name, str):
        raise TypeError(
            f"profile.industry_name must be str, got {type(industry_name).__name__}"
        )

    if not industry_name.strip():
        raise DataValidationError(
            "profile.industry_name must not be empty or whitespace-only"
        )

    return _normalize_industry_name(industry_name), profile


class IndustryRegistry:
    """Register and look up BaseIndustryProfile implementations in memory."""

    def __init__(
        self,
        profiles: Sequence[BaseIndustryProfile] | None = None,
    ) -> None:
        """Create a registry, optionally registering profiles in order.

        Args:
            profiles: Optional sequence of profiles to register. When ``None``,
                the registry starts empty. The input sequence is not modified.
        """
        staged: dict[str, BaseIndustryProfile] = {}
        if profiles is not None:
            for profile in profiles:
                key, validated = _validate_profile(profile)
                if key in staged:
                    raise DataValidationError(
                        f"Duplicate industry profile registration for "
                        f"{validated.industry_name!r}"
                    )
                staged[key] = validated

        self._profiles: dict[str, BaseIndustryProfile] = staged

    def register(
        self,
        profile: BaseIndustryProfile,
        *,
        replace: bool = False,
    ) -> None:
        """Register a profile, optionally replacing an existing entry.

        Args:
            profile: Profile implementation to register.
            replace: When ``True``, replace an existing profile with the same
                normalized industry name while preserving registry order.

        Raises:
            TypeError: If ``profile`` or ``replace`` has an invalid type, or if
                ``profile.industry_name`` is not a string.
            DataValidationError: If the industry name is empty/whitespace-only,
                or if a duplicate name is registered without ``replace=True``.
        """
        if not isinstance(replace, bool):
            raise TypeError(f"replace must be bool, got {type(replace).__name__}")

        key, validated = _validate_profile(profile)
        if key in self._profiles and not replace:
            raise DataValidationError(
                f"Duplicate industry profile registration for "
                f"{validated.industry_name!r}"
            )

        self._profiles[key] = validated

    def get(self, industry_name: str) -> BaseIndustryProfile:
        """Return the registered profile for the given industry name.

        Args:
            industry_name: Industry name to look up. Matching ignores case and
                surrounding whitespace.

        Returns:
            The registered profile instance.

        Raises:
            TypeError: If ``industry_name`` is not a string.
            ValueError: If ``industry_name`` is empty or whitespace-only.
            KeyError: If no profile is registered for the name.
        """
        validated_name = _validate_lookup_name(industry_name)
        key = _normalize_industry_name(validated_name)
        try:
            return self._profiles[key]
        except KeyError as exc:
            raise KeyError(validated_name) from exc

    def contains(self, industry_name: str) -> bool:
        """Return whether a profile is registered for the industry name.

        Args:
            industry_name: Industry name to look up. Matching ignores case and
                surrounding whitespace.

        Returns:
            ``True`` if registered, otherwise ``False``.

        Raises:
            TypeError: If ``industry_name`` is not a string.
            ValueError: If ``industry_name`` is empty or whitespace-only.
        """
        validated_name = _validate_lookup_name(industry_name)
        key = _normalize_industry_name(validated_name)
        return key in self._profiles

    def unregister(self, industry_name: str) -> BaseIndustryProfile:
        """Remove and return the registered profile for the industry name.

        Args:
            industry_name: Industry name to remove. Matching ignores case and
                surrounding whitespace.

        Returns:
            The removed profile instance.

        Raises:
            TypeError: If ``industry_name`` is not a string.
            ValueError: If ``industry_name`` is empty or whitespace-only.
            KeyError: If no profile is registered for the name.
        """
        validated_name = _validate_lookup_name(industry_name)
        key = _normalize_industry_name(validated_name)
        try:
            return self._profiles.pop(key)
        except KeyError as exc:
            raise KeyError(validated_name) from exc

    def list_names(self) -> tuple[str, ...]:
        """Return original industry names in registration order."""
        return tuple(profile.industry_name for profile in self._profiles.values())

    def list_profiles(self) -> tuple[BaseIndustryProfile, ...]:
        """Return registered profiles in registration order."""
        return tuple(self._profiles.values())

    def score_all(self, metadata: DatasetMetadata) -> tuple[IndustryScore, ...]:
        """Score the dataset metadata against every registered profile.

        Args:
            metadata: Dataset metadata passed to each profile.

        Returns:
            Industry scores in registration order.

        Raises:
            TypeError: If ``metadata`` is not a ``DatasetMetadata`` instance.
            DataValidationError: If a profile returns a non-``IndustryScore``
                value or an ``IndustryScore`` with an empty industry name.
        """
        if not isinstance(metadata, DatasetMetadata):
            raise TypeError(
                f"metadata must be DatasetMetadata, got {type(metadata).__name__}"
            )

        scores: list[IndustryScore] = []
        for profile in self._profiles.values():
            score = profile.score_industry(metadata)
            if not isinstance(score, IndustryScore):
                raise DataValidationError(
                    f"Profile {profile.industry_name!r} must return IndustryScore, "
                    f"got {type(score).__name__}"
                )
            if score.industry_name == "":
                raise DataValidationError(
                    f"Profile {profile.industry_name!r} returned IndustryScore "
                    f"with an empty industry_name"
                )
            scores.append(score)

        return tuple(scores)

    def best_match(self, metadata: DatasetMetadata) -> IndustryScore | None:
        """Return the best-matching industry score for the metadata.

        Selection prefers higher ``score``, then higher ``confidence``, then the
        earliest registered profile when both values are tied.

        Args:
            metadata: Dataset metadata passed to each profile.

        Returns:
            The best ``IndustryScore``, or ``None`` when the registry is empty.
        """
        scores = self.score_all(metadata)
        if not scores:
            return None

        best = scores[0]
        for score in scores[1:]:
            if score.score > best.score or (
                score.score == best.score and score.confidence > best.confidence
            ):
                best = score

        return best

    def __len__(self) -> int:
        """Return the number of registered profiles."""
        return len(self._profiles)
