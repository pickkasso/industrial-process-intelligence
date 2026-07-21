"""Abstract diagnoser contract for root-cause diagnosis (Step 8A).

Defines the interface only. Concrete attribution algorithms (robust z-score,
group comparison, and later optional methods) are implemented in later steps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl

from process_intelligence.core.schemas import ExplanationResult
from process_intelligence.diagnosis.enums import DiagnosisMethod
from process_intelligence.diagnosis.schemas import (
    DiagnosisBatchResult,
    DiagnosisRequest,
    DiagnosisResult,
    ScalarMetadataValue,
)


class BaseRootCauseDiagnoser(ABC):
    """Abstract root-cause diagnoser over Polars frames and diagnosis requests.

    Implementations associate likely driver variables with anomaly events.
    They must not claim causation. Metadata returned by ``get_metadata`` must
    contain only scalar values and must not include estimators, fitted models,
    or full input DataFrames.
    """

    @property
    @abstractmethod
    def method(self) -> DiagnosisMethod:
        """Primary diagnosis method implemented by this diagnoser."""

    @abstractmethod
    def diagnose(
        self,
        data: pl.DataFrame,
        *,
        request: DiagnosisRequest,
        explanation: ExplanationResult | None = None,
    ) -> DiagnosisResult | DiagnosisBatchResult:
        """Diagnose likely drivers for the requested anomaly scope.

        Args:
            data: Feature frame used for association analysis (Polars only).
            request: Validated diagnosis request describing scope and columns.
            explanation: Optional model explanation from
                ``BaseAnalysisModel.explain``; unused methods may ignore it.

        Returns:
            A per-scope ``DiagnosisResult`` or a ``DiagnosisBatchResult`` when
            multiple events are diagnosed together.
        """

    @abstractmethod
    def get_metadata(self) -> dict[str, ScalarMetadataValue]:
        """Return scalar-only diagnoser metadata for traceability.

        Must not return estimators, model objects, DataFrames, or full
        prediction arrays—only ``str``, ``int``, ``float``, ``bool``, or
        ``None`` values.
        """
