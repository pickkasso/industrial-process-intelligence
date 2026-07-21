"""Abstract recommendation engine contract (Step 9A).

Defines the interface only. Concrete optimizers and fallback engines are
implemented in later steps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from process_intelligence.recommendation.schemas import (
    RecommendationRequest,
    RecommendationResult,
    RecommendationSafetyDecision,
    ScalarMetadataValue,
)


class BaseRecommendationEngine(ABC):
    """Abstract recommendation engine over validated requests and safety decisions.

    Concrete engines may generate proposed process changes only when the safety
    decision is not ``REFUSED``. When ``safety_decision.status`` is ``REFUSED``,
    implementations must not produce strong recommendations (no non-empty
    ``GENERATED`` change list). Metadata returned by ``get_metadata`` must
    contain only scalar values and must not include estimators, fitted models,
    or full input DataFrames.
    """

    @abstractmethod
    def recommend(
        self,
        request: RecommendationRequest,
        *,
        safety_decision: RecommendationSafetyDecision,
    ) -> RecommendationResult:
        """Produce a recommendation result under the given safety decision.

        Args:
            request: Validated recommendation request.
            safety_decision: Prior safety-gate decision for the same request.

        Returns:
            A ``RecommendationResult``. When ``safety_decision.status`` is
            ``REFUSED``, the result must remain refused without generating
            strong proposed changes.
        """

    @abstractmethod
    def get_metadata(self) -> dict[str, ScalarMetadataValue]:
        """Return scalar-only engine metadata for traceability.

        Must not return estimators, model objects, DataFrames, or full
        prediction arrays—only ``str``, ``int``, ``float``, ``bool``, or
        ``None`` values.
        """
