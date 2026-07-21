"""Unit tests for BaseRootCauseDiagnoser abstract contract (Step 8A)."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from process_intelligence.core.enums import AnalysisTask, AnomalyType, ColumnRole
from process_intelligence.core.schemas import (
    AnomalyEvent,
    ExplanationResult,
    RootCauseFactor,
)
from process_intelligence.diagnosis import (
    BaseRootCauseDiagnoser,
    DiagnosisBatchResult,
    DiagnosisMethod,
    DiagnosisRequest,
    DiagnosisResult,
    DiagnosisScope,
)


def _event(anomaly_id: str = "a-1") -> AnomalyEvent:
    return AnomalyEvent(
        anomaly_id=anomaly_id,
        anomaly_type=AnomalyType.PROCESS_INPUT,
        anomaly_score=0.9,
        severity="high",
        model_confidence=0.8,
        detector="isolation_forest",
        rationale="score above threshold",
    )


def _request(*, scope: DiagnosisScope = DiagnosisScope.SINGLE_EVENT) -> DiagnosisRequest:
    events = [_event("a-1")]
    if scope is DiagnosisScope.TOP_ANOMALIES:
        events = [_event("a-1"), _event("a-2")]
    elif scope in {DiagnosisScope.ANOMALY_GROUP, DiagnosisScope.GLOBAL}:
        events = []
    return DiagnosisRequest(
        task=AnalysisTask.UNSUPERVISED_ANOMALY,
        method=DiagnosisMethod.GROUP_COMPARISON,
        scope=scope,
        feature_columns=["pressure", "temperature"],
        anomaly_events=events,
        anomaly_indicator_column="is_anomaly",
        anomaly_score_column="anomaly_score",
    )


def _result(anomaly_id: str = "a-1") -> DiagnosisResult:
    return DiagnosisResult(
        anomaly_id=anomaly_id,
        task=AnalysisTask.UNSUPERVISED_ANOMALY,
        method_used=[DiagnosisMethod.GROUP_COMPARISON],
        scope=DiagnosisScope.SINGLE_EVENT,
        factors=[
            RootCauseFactor(
                variable="pressure",
                direction="increase",
                deviation=2.0,
                role=ColumnRole.CONTROLLABLE_PROCESS,
                controllable=True,
                evidence=(
                    "Likely driver by normal-vs-anomaly group comparison; "
                    "association only."
                ),
                confidence=0.8,
                needs_verification=True,
            )
        ],
        confidence=0.8,
        analyzed_row_count=1,
        reference_row_count=5,
        caveats=["Association only; causation is not established."],
        generated_at=datetime(2026, 7, 21, 10, 0, tzinfo=UTC),
        metadata={"deterministic": True},
    )


class _ConcreteDiagnoser(BaseRootCauseDiagnoser):
    """Minimal concrete diagnoser used only for contract tests."""

    def __init__(self) -> None:
        self._meta: dict[str, str | int | float | bool | None] = {
            "name": "concrete",
            "version": 1,
        }

    @property
    def method(self) -> DiagnosisMethod:
        return DiagnosisMethod.GROUP_COMPARISON

    def diagnose(
        self,
        data: pl.DataFrame,
        *,
        request: DiagnosisRequest,
        explanation: ExplanationResult | None = None,
    ) -> DiagnosisResult | DiagnosisBatchResult:
        _ = explanation
        if request.scope is DiagnosisScope.SINGLE_EVENT:
            return _result(request.anomaly_events[0].anomaly_id)
        results = [
            _result(event.anomaly_id) for event in request.anomaly_events
        ]
        if not results and data.height >= 0:
            return DiagnosisBatchResult(
                results=[],
                aggregate_factors=[],
                requested_event_count=0,
                diagnosed_event_count=0,
                failed_event_count=0,
                generated_at=datetime(2026, 7, 21, 10, 0, tzinfo=UTC),
                warnings=["No events requested."],
                metadata={"rows_seen": data.height},
            )
        return DiagnosisBatchResult(
            results=results,
            aggregate_factors=[
                RootCauseFactor(
                    variable="pressure",
                    direction="increase",
                    deviation=2.0,
                    role=ColumnRole.CONTROLLABLE_PROCESS,
                    controllable=True,
                    evidence="Batch aggregate association only.",
                    confidence=0.7,
                    needs_verification=True,
                )
            ],
            requested_event_count=len(results),
            diagnosed_event_count=len(results),
            failed_event_count=0,
            generated_at=datetime(2026, 7, 21, 10, 0, tzinfo=UTC),
            warnings=["Deterministic test batch."],
            metadata={"rows_seen": data.height},
        )

    def get_metadata(self) -> dict[str, str | int | float | bool | None]:
        return dict(self._meta)


class _MissingMethod(BaseRootCauseDiagnoser):
    def diagnose(
        self,
        data: pl.DataFrame,
        *,
        request: DiagnosisRequest,
        explanation: ExplanationResult | None = None,
    ) -> DiagnosisResult | DiagnosisBatchResult:
        _ = data, request, explanation
        return _result()

    def get_metadata(self) -> dict[str, str | int | float | bool | None]:
        return {}


class _MissingDiagnose(BaseRootCauseDiagnoser):
    @property
    def method(self) -> DiagnosisMethod:
        return DiagnosisMethod.ROBUST_Z_SCORE

    def get_metadata(self) -> dict[str, str | int | float | bool | None]:
        return {}


class _MissingMetadata(BaseRootCauseDiagnoser):
    @property
    def method(self) -> DiagnosisMethod:
        return DiagnosisMethod.ROBUST_Z_SCORE

    def diagnose(
        self,
        data: pl.DataFrame,
        *,
        request: DiagnosisRequest,
        explanation: ExplanationResult | None = None,
    ) -> DiagnosisResult | DiagnosisBatchResult:
        _ = data, request, explanation
        return _result()


def test_cannot_instantiate_abstract_diagnoser() -> None:
    with pytest.raises(TypeError):
        BaseRootCauseDiagnoser()  # type: ignore[abstract]


@pytest.mark.parametrize(
    "incomplete_cls",
    [_MissingMethod, _MissingDiagnose, _MissingMetadata],
)
def test_incomplete_subclass_cannot_instantiate(incomplete_cls: type) -> None:
    with pytest.raises(TypeError):
        incomplete_cls()  # type: ignore[abstract]


def test_concrete_subclass_instantiates() -> None:
    diagnoser = _ConcreteDiagnoser()
    assert isinstance(diagnoser, BaseRootCauseDiagnoser)
    assert diagnoser.method is DiagnosisMethod.GROUP_COMPARISON


def test_diagnose_return_types_and_optional_explanation() -> None:
    diagnoser = _ConcreteDiagnoser()
    frame = pl.DataFrame(
        {
            "_original_row_id": [0, 1, 2],
            "pressure": [1.0, 2.0, 3.0],
            "temperature": [10.0, 11.0, 12.0],
            "is_anomaly": [0, 1, 0],
            "anomaly_score": [0.1, 0.9, 0.2],
        }
    )
    single = diagnoser.diagnose(frame, request=_request())
    assert isinstance(single, DiagnosisResult)
    assert single.anomaly_id == "a-1"

    explanation = ExplanationResult(
        method="none",
        feature_importances={"pressure": 0.5},
        notes=["optional"],
        confidence=0.5,
    )
    with_explanation = diagnoser.diagnose(
        frame,
        request=_request(),
        explanation=explanation,
    )
    assert isinstance(with_explanation, DiagnosisResult)
    assert with_explanation.model_dump() == single.model_dump()

    batch = diagnoser.diagnose(
        frame,
        request=_request(scope=DiagnosisScope.TOP_ANOMALIES),
    )
    assert isinstance(batch, DiagnosisBatchResult)
    assert batch.diagnosed_event_count == 2


def test_diagnose_does_not_mutate_inputs() -> None:
    diagnoser = _ConcreteDiagnoser()
    frame = pl.DataFrame(
        {
            "pressure": [1.0, 2.0],
            "temperature": [10.0, 11.0],
            "is_anomaly": [0, 1],
            "anomaly_score": [0.1, 0.9],
        }
    )
    request = _request()
    original_columns = list(frame.columns)
    original_height = frame.height
    original_event_ids = [event.anomaly_id for event in request.anomaly_events]
    original_features = list(request.feature_columns)

    _ = diagnoser.diagnose(frame, request=request)

    assert frame.columns == original_columns
    assert frame.height == original_height
    assert [event.anomaly_id for event in request.anomaly_events] == original_event_ids
    assert request.feature_columns == original_features


def test_metadata_independence_and_no_heavy_objects() -> None:
    diagnoser = _ConcreteDiagnoser()
    first = diagnoser.get_metadata()
    second = diagnoser.get_metadata()
    assert first == {"name": "concrete", "version": 1}
    assert first is not second
    first["name"] = "mutated"
    assert diagnoser.get_metadata()["name"] == "concrete"

    for value in second.values():
        assert value is None or isinstance(value, (str, int, float, bool))
        assert not isinstance(value, pl.DataFrame)


def test_diagnose_is_deterministic() -> None:
    diagnoser = _ConcreteDiagnoser()
    frame = pl.DataFrame(
        {
            "pressure": [1.0, 2.0],
            "temperature": [10.0, 11.0],
            "is_anomaly": [0, 1],
            "anomaly_score": [0.1, 0.9],
        }
    )
    request = _request()
    first = diagnoser.diagnose(frame, request=request)
    second = diagnoser.diagnose(frame, request=request)
    assert isinstance(first, DiagnosisResult)
    assert isinstance(second, DiagnosisResult)
    assert first.model_dump() == second.model_dump()


def test_abstract_contract_docstrings_exist() -> None:
    assert BaseRootCauseDiagnoser.__doc__
    assert BaseRootCauseDiagnoser.method.__doc__
    assert BaseRootCauseDiagnoser.diagnose.__doc__
    assert BaseRootCauseDiagnoser.get_metadata.__doc__
    assert "estimator" in (BaseRootCauseDiagnoser.get_metadata.__doc__ or "").lower()
    assert "dataframe" in (BaseRootCauseDiagnoser.get_metadata.__doc__ or "").lower()
