"""Unit tests for DiagnosisEnsembleDiagnoser (Step 8D)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from process_intelligence.core.enums import AnalysisTask, AnomalyType, ColumnRole
from process_intelligence.core.exceptions import (
    DataValidationError,
    ProcessIntelligenceError,
)
from process_intelligence.core.schemas import (
    AnomalyEvent,
    ExplanationResult,
    RootCauseFactor,
)
from process_intelligence.diagnosis import (
    BaseRootCauseDiagnoser,
    DiagnosisBatchResult,
    DiagnosisEnsembleConfig,
    DiagnosisEnsembleDiagnoser,
    DiagnosisMethod,
    DiagnosisRequest,
    DiagnosisResult,
    DiagnosisScope,
    EnsembleFactorStatistic,
)

_SOURCE_ROBUST = "ROBUST_GROUP_COMPARISON"
_SOURCE_RESIDUAL = "RESIDUAL_ASSOCIATION"


def _event(anomaly_id: str, *, score: float = 0.9) -> AnomalyEvent:
    return AnomalyEvent(
        anomaly_id=anomaly_id,
        anomaly_type=AnomalyType.PROCESS_INPUT,
        anomaly_score=score,
        severity="high",
        model_confidence=0.8,
        detector="ensemble_test",
        rationale="score above threshold",
    )


def _factor(
    variable: str,
    *,
    direction: str = "POSITIVE",
    deviation: float = 1.0,
    confidence: float = 0.8,
    role: ColumnRole = ColumnRole.CONTROLLABLE_PROCESS,
    controllable: bool = True,
    needs_verification: bool = False,
    evidence: str = "child association evidence",
) -> RootCauseFactor:
    return RootCauseFactor(
        variable=variable,
        direction=direction,
        deviation=deviation,
        role=role,
        controllable=controllable,
        evidence=evidence,
        confidence=confidence,
        needs_verification=needs_verification,
    )


def _result(
    *,
    anomaly_id: str | None = "5",
    factors: list[RootCauseFactor] | None = None,
    scope: DiagnosisScope = DiagnosisScope.SINGLE_EVENT,
    task: AnalysisTask = AnalysisTask.RESIDUAL_ANOMALY,
    method_used: list[DiagnosisMethod] | None = None,
    analyzed_row_count: int = 1,
    reference_row_count: int = 5,
    confidence: float | None = None,
    caveats: list[str] | None = None,
) -> DiagnosisResult:
    resolved_factors = factors if factors is not None else [_factor("pressure")]
    if confidence is None:
        confidence = (
            float(sum(item.confidence for item in resolved_factors) / len(resolved_factors))
            if resolved_factors
            else 0.0
        )
    return DiagnosisResult(
        anomaly_id=anomaly_id,
        task=task,
        method_used=method_used
        or [DiagnosisMethod.ROBUST_Z_SCORE, DiagnosisMethod.GROUP_COMPARISON],
        scope=scope,
        factors=resolved_factors,
        confidence=confidence,
        analyzed_row_count=analyzed_row_count,
        reference_row_count=reference_row_count,
        caveats=caveats
        or [
            "Association does not establish causation; treat factors as likely "
            "drivers requiring process review."
        ],
        generated_at=datetime.now(UTC),
        metadata={"ranking_is_heuristic": True},
    )


def _batch(
    results: list[DiagnosisResult],
    *,
    failed_event_count: int = 0,
) -> DiagnosisBatchResult:
    return DiagnosisBatchResult(
        results=results,
        aggregate_factors=[],
        requested_event_count=len(results) + failed_event_count,
        diagnosed_event_count=len(results),
        failed_event_count=failed_event_count,
        generated_at=datetime.now(UTC),
        warnings=["Association does not establish causation across diagnosed events."],
        metadata={"returned_result_count": len(results)},
    )


class RecordingDiagnoser(BaseRootCauseDiagnoser):
    """Test double that records diagnose calls and returns configured payloads."""

    def __init__(
        self,
        *,
        method: DiagnosisMethod,
        payload: DiagnosisResult | DiagnosisBatchResult | Exception | None = None,
    ) -> None:
        self._method = method
        self._payload = payload
        self.calls: list[dict[str, Any]] = []
        self.mutable_marker = {"value": 0}

    @property
    def method(self) -> DiagnosisMethod:
        return self._method

    def set_payload(
        self,
        payload: DiagnosisResult | DiagnosisBatchResult | Exception,
    ) -> None:
        self._payload = payload

    def diagnose(
        self,
        data: pl.DataFrame,
        *,
        request: DiagnosisRequest,
        explanation: ExplanationResult | None = None,
    ) -> DiagnosisResult | DiagnosisBatchResult:
        self.calls.append(
            {
                "data_id": id(data),
                "request": request,
                "explanation_id": id(explanation) if explanation is not None else None,
                "method": request.method,
                "events_id": id(request.anomaly_events),
                "features_id": id(request.feature_columns),
            }
        )
        if isinstance(self._payload, Exception):
            raise self._payload
        if self._payload is None:
            raise RuntimeError("no payload configured")
        if isinstance(self._payload, DiagnosisResult):
            return self._payload.model_copy(deep=True)
        return self._payload.model_copy(deep=True)

    def get_metadata(self) -> dict[str, Any]:
        return {"method": self._method.value}


def _request(
    *,
    scope: DiagnosisScope = DiagnosisScope.SINGLE_EVENT,
    method: DiagnosisMethod = DiagnosisMethod.ENSEMBLE,
    events: list[AnomalyEvent] | None = None,
    feature_columns: list[str] | None = None,
    top_k_factors: int = 10,
    minimum_reference_rows: int = 2,
    anomaly_indicator_column: str | None = "_is_residual_anomaly",
    anomaly_score_column: str | None = "_residual_anomaly_score",
    task: AnalysisTask = AnalysisTask.RESIDUAL_ANOMALY,
) -> DiagnosisRequest:
    if events is None:
        if scope is DiagnosisScope.SINGLE_EVENT:
            events = [_event("5", score=0.95)]
        elif scope is DiagnosisScope.TOP_ANOMALIES:
            events = [
                _event("5", score=0.95),
                _event("6", score=0.80),
                _event("7", score=0.70),
            ]
        else:
            events = []
    return DiagnosisRequest(
        task=task,
        method=method,
        scope=scope,
        feature_columns=feature_columns or ["pressure", "temperature", "stable"],
        anomaly_events=events,
        anomaly_indicator_column=anomaly_indicator_column,
        anomaly_score_column=anomaly_score_column,
        row_id_column="_original_row_id",
        top_k_factors=top_k_factors,
        minimum_reference_rows=minimum_reference_rows,
    )


def _residual_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "_original_row_id": ["0", "1", "2", "3", "4", "5", "6", "7"],
            "pressure": [1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 8.0, 7.0],
            "temperature": [10.0, 10.0, 10.0, 10.0, 10.0, 30.0, 0.0, 10.0],
            "stable": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            "_regression_residual": [-0.1, 0.0, 0.1, 0.2, 0.3, 2.0, 1.5, 1.0],
            "_absolute_centered_residual": [0.2, 0.1, 0.0, 0.1, 0.2, 1.9, 1.4, 0.9],
            "_residual_anomaly_score": [0.05, 0.08, 0.10, 0.12, 0.15, 0.95, 0.80, 0.70],
            "_is_residual_anomaly": [0, 0, 0, 0, 0, 1, 1, 1],
        }
    )


def _robust_only_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "_original_row_id": ["0", "1", "2", "3", "4", "5", "6", "7"],
            "pressure": [10.0, 10.0, 10.0, 10.0, 10.0, 40.0, 0.0, 10.0],
            "temperature": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 35.0],
            "stable": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            "is_anomaly": [0, 0, 0, 0, 0, 1, 1, 1],
            "anomaly_score": [0.1, 0.1, 0.1, 0.1, 0.1, 0.95, 0.8, 0.7],
        }
    )


def _valid_statistic(**overrides: Any) -> EnsembleFactorStatistic:
    payload: dict[str, Any] = {
        "feature_name": "pressure",
        "source_method_count": 2,
        "source_methods": [
            DiagnosisMethod.ENSEMBLE,
            DiagnosisMethod.RESIDUAL_ASSOCIATION,
        ],
        "source_ranks": {_SOURCE_ROBUST: 1, _SOURCE_RESIDUAL: 2},
        "source_confidences": {_SOURCE_ROBUST: 0.8, _SOURCE_RESIDUAL: 0.7},
        "weighted_rrf_score": 0.02,
        "normalized_ensemble_score": 1.0,
        "combined_confidence": 0.75,
        "combined_direction": "POSITIVE",
        "combined_deviation": 3.0,
        "direction_conflict": False,
        "needs_verification": False,
        "warnings": [],
    }
    payload.update(overrides)
    return EnsembleFactorStatistic(**payload)


def _ensemble_with_doubles(
    *,
    robust_payload: DiagnosisResult | DiagnosisBatchResult | Exception,
    residual_payload: DiagnosisResult | DiagnosisBatchResult | Exception,
    config: DiagnosisEnsembleConfig | None = None,
) -> tuple[DiagnosisEnsembleDiagnoser, RecordingDiagnoser, RecordingDiagnoser]:
    robust = RecordingDiagnoser(
        method=DiagnosisMethod.ENSEMBLE,
        payload=robust_payload,
    )
    residual = RecordingDiagnoser(
        method=DiagnosisMethod.RESIDUAL_ASSOCIATION,
        payload=residual_payload,
    )
    diagnoser = DiagnosisEnsembleDiagnoser(
        config=config,
        robust_diagnoser=robust,
        residual_diagnoser=residual,
    )
    return diagnoser, robust, residual


# --- Config ---


def test_config_defaults() -> None:
    config = DiagnosisEnsembleConfig()
    assert config.robust_method_weight == pytest.approx(0.5)
    assert config.residual_method_weight == pytest.approx(0.5)
    assert config.reciprocal_rank_constant == pytest.approx(60.0)
    assert config.minimum_method_support == 1
    assert config.continue_on_method_failure is True
    assert config.require_residual_method is False
    assert config.include_child_evidence is True
    assert config.normalize_ensemble_scores is True
    assert config.penalize_partial_method_support is True


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("robust_method_weight", -0.1),
        ("residual_method_weight", 1.1),
        ("robust_method_weight", True),
        ("residual_method_weight", float("nan")),
        ("robust_method_weight", float("inf")),
        ("reciprocal_rank_constant", 0.0),
        ("reciprocal_rank_constant", True),
        ("minimum_method_support", 0),
        ("minimum_method_support", 3),
        ("minimum_method_support", True),
        ("continue_on_method_failure", 1),
        ("require_residual_method", "true"),
        ("include_child_evidence", 0),
        ("normalize_ensemble_scores", "yes"),
        ("penalize_partial_method_support", 1),
    ],
)
def test_config_rejects_invalid_values(field_name: str, bad_value: object) -> None:
    with pytest.raises(ValidationError):
        DiagnosisEnsembleConfig(**{field_name: bad_value})


def test_config_rejects_both_weights_zero() -> None:
    with pytest.raises(ValidationError):
        DiagnosisEnsembleConfig(robust_method_weight=0.0, residual_method_weight=0.0)


def test_config_round_trip() -> None:
    config = DiagnosisEnsembleConfig(
        robust_method_weight=0.7,
        residual_method_weight=0.3,
        reciprocal_rank_constant=10.0,
        minimum_method_support=2,
        continue_on_method_failure=False,
        require_residual_method=True,
        include_child_evidence=False,
        normalize_ensemble_scores=False,
        penalize_partial_method_support=False,
    )
    restored = DiagnosisEnsembleConfig.model_validate(config.model_dump())
    assert restored == config


# --- EnsembleFactorStatistic ---


def test_statistic_valid_and_round_trip() -> None:
    statistic = _valid_statistic(warnings=["partial method support for this feature"])
    restored = EnsembleFactorStatistic.model_validate(statistic.model_dump())
    assert restored.feature_name == "pressure"
    assert restored.source_method_count == 2
    assert restored.warnings == ["partial method support for this feature"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"feature_name": ""},
        {"feature_name": "_original_row_id"},
        {"source_method_count": 0},
        {"source_method_count": True},
        {
            "source_method_count": 1,
            "source_methods": [
                DiagnosisMethod.ENSEMBLE,
                DiagnosisMethod.RESIDUAL_ASSOCIATION,
            ],
        },
        {
            "source_methods": [
                DiagnosisMethod.ENSEMBLE,
                DiagnosisMethod.ENSEMBLE,
            ]
        },
        {"source_ranks": {_SOURCE_ROBUST: 0, _SOURCE_RESIDUAL: 1}},
        {"source_ranks": {_SOURCE_ROBUST: True, _SOURCE_RESIDUAL: 1}},
        {
            "source_ranks": {_SOURCE_ROBUST: 1},
            "source_confidences": {_SOURCE_ROBUST: 0.8, _SOURCE_RESIDUAL: 0.7},
            "source_method_count": 1,
            "source_methods": [DiagnosisMethod.ENSEMBLE],
        },
        {"source_confidences": {_SOURCE_ROBUST: 1.5, _SOURCE_RESIDUAL: 0.7}},
        {"weighted_rrf_score": -0.1},
        {"normalized_ensemble_score": 1.5},
        {"combined_confidence": -0.1},
        {"combined_direction": "UP"},
        {"combined_deviation": float("nan")},
        {"direction_conflict": 1},
        {"needs_verification": "true"},
        {"warnings": [""]},
        {"warnings": ["a", "a"]},
    ],
)
def test_statistic_rejects_invalid(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _valid_statistic(**overrides)


def test_statistic_mutable_collections_independent() -> None:
    warnings = ["direction conflict across ensemble sources"]
    ranks = {_SOURCE_ROBUST: 1}
    confidences = {_SOURCE_ROBUST: 0.5}
    statistic = _valid_statistic(
        source_method_count=1,
        source_methods=[DiagnosisMethod.ENSEMBLE],
        source_ranks=ranks,
        source_confidences=confidences,
        warnings=warnings,
    )
    warnings.append("extra")
    ranks[_SOURCE_RESIDUAL] = 2
    confidences[_SOURCE_RESIDUAL] = 0.1
    assert statistic.warnings == ["direction conflict across ensemble sources"]
    assert _SOURCE_RESIDUAL not in statistic.source_ranks
    assert _SOURCE_RESIDUAL not in statistic.source_confidences


# --- Construction / contract ---


def test_default_construction_and_method() -> None:
    diagnoser = DiagnosisEnsembleDiagnoser()
    assert isinstance(diagnoser, BaseRootCauseDiagnoser)
    assert diagnoser.method is DiagnosisMethod.ENSEMBLE


def test_config_type_error() -> None:
    with pytest.raises(TypeError):
        DiagnosisEnsembleDiagnoser(config={"robust_method_weight": 0.5})  # type: ignore[arg-type]


def test_child_type_and_method_errors() -> None:
    with pytest.raises(TypeError):
        DiagnosisEnsembleDiagnoser(robust_diagnoser=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        DiagnosisEnsembleDiagnoser(residual_diagnoser=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        DiagnosisEnsembleDiagnoser(
            robust_diagnoser=RecordingDiagnoser(
                method=DiagnosisMethod.RESIDUAL_ASSOCIATION
            )
        )
    with pytest.raises(TypeError):
        DiagnosisEnsembleDiagnoser(
            residual_diagnoser=RecordingDiagnoser(method=DiagnosisMethod.ENSEMBLE)
        )


def test_external_config_immutability_and_child_isolation() -> None:
    config = DiagnosisEnsembleConfig(robust_method_weight=0.6)
    first = DiagnosisEnsembleDiagnoser(config=config)
    second = DiagnosisEnsembleDiagnoser(config=config)
    config.robust_method_weight = 0.1
    assert first.get_metadata()["robust_method_weight"] == pytest.approx(0.6)
    assert second.get_metadata()["robust_method_weight"] == pytest.approx(0.6)
    assert first.get_metadata() is not second.get_metadata()
    assert first._robust_diagnoser is not second._robust_diagnoser
    assert first._residual_diagnoser is not second._residual_diagnoser


def test_metadata_scalar_only_and_independent() -> None:
    diagnoser = DiagnosisEnsembleDiagnoser()
    first = diagnoser.get_metadata()
    second = diagnoser.get_metadata()
    assert first["method"] == "ENSEMBLE"
    assert first["robust_source"] == _SOURCE_ROBUST
    assert first["residual_source"] == _SOURCE_RESIDUAL
    assert first["ranking_method"] == "weighted_reciprocal_rank_fusion"
    assert first["association_not_causation"] is True
    assert first["fitted"] is False
    assert first is not second
    first["fitted"] = True
    assert second["fitted"] is False
    for value in first.values():
        assert value is None or isinstance(value, (str, int, float, bool))


# --- Input / child request ---


def test_request_method_and_input_validation() -> None:
    diagnoser, _, _ = _ensemble_with_doubles(
        robust_payload=_result(),
        residual_payload=_result(
            method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
            factors=[_factor("temperature")],
        ),
    )
    data = _residual_frame()
    ok = _request()
    out = diagnoser.diagnose(data, request=ok)
    assert isinstance(out, DiagnosisResult)

    with pytest.raises(DataValidationError):
        diagnoser.diagnose(
            data,
            request=_request(method=DiagnosisMethod.RESIDUAL_ASSOCIATION),
        )
    with pytest.raises(TypeError):
        diagnoser.diagnose({"pressure": [1]}, request=ok)  # type: ignore[arg-type]
    with pytest.raises(DataValidationError):
        diagnoser.diagnose(data.clear(), request=ok)
    with pytest.raises(TypeError):
        diagnoser.diagnose(data, request={"method": "ENSEMBLE"})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        diagnoser.diagnose(data, request=ok, explanation={"method": "x"})  # type: ignore[arg-type]


def test_input_and_child_request_independence() -> None:
    robust_payload = _result(factors=[_factor("pressure"), _factor("temperature")])
    residual_payload = _result(
        method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
        factors=[_factor("pressure", confidence=0.6)],
    )
    diagnoser, robust, residual = _ensemble_with_doubles(
        robust_payload=robust_payload,
        residual_payload=residual_payload,
    )
    data = _residual_frame()
    explanation = ExplanationResult(method="test", feature_importances={"pressure": 0.5})
    request = _request()
    original_events = list(request.anomaly_events)
    original_features = list(request.feature_columns)
    original_meta = dict(request.metadata)

    result = diagnoser.diagnose(data, request=request, explanation=explanation)
    assert isinstance(result, DiagnosisResult)

    assert request.anomaly_events == original_events
    assert request.feature_columns == original_features
    assert request.metadata == original_meta
    assert explanation.feature_importances == {"pressure": 0.5}

    assert len(robust.calls) == 1
    assert len(residual.calls) == 1
    assert robust.calls[0]["method"] is DiagnosisMethod.ENSEMBLE
    assert residual.calls[0]["method"] is DiagnosisMethod.RESIDUAL_ASSOCIATION
    assert robust.calls[0]["data_id"] == id(data)
    assert residual.calls[0]["data_id"] == id(data)
    assert robust.calls[0]["explanation_id"] == id(explanation)
    assert residual.calls[0]["explanation_id"] == id(explanation)
    assert robust.calls[0]["events_id"] != id(request.anomaly_events)
    assert residual.calls[0]["events_id"] != id(request.anomaly_events)
    assert robust.calls[0]["events_id"] != residual.calls[0]["events_id"]


def test_child_call_order_and_once() -> None:
    diagnoser, robust, residual = _ensemble_with_doubles(
        robust_payload=_result(factors=[_factor("pressure")]),
        residual_payload=_result(
            method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
            factors=[_factor("temperature")],
        ),
    )
    diagnoser.diagnose(_residual_frame(), request=_request())
    assert [call["method"] for call in robust.calls + residual.calls] == [
        DiagnosisMethod.ENSEMBLE,
        DiagnosisMethod.RESIDUAL_ASSOCIATION,
    ]
    assert len(robust.calls) == 1
    assert len(residual.calls) == 1


# --- Child result validation ---


def test_child_result_contract_failures() -> None:
    cases = [
        _result(task=AnalysisTask.CLASSIFICATION),
        _result(scope=DiagnosisScope.GLOBAL),
        _result(factors=[_factor("missing_feature")]),
    ]
    for bad in cases:
        residual_bad = bad.model_copy(
            update={"method_used": [DiagnosisMethod.RESIDUAL_ASSOCIATION]},
            deep=True,
        )
        diagnoser, _, _ = _ensemble_with_doubles(
            robust_payload=bad,
            residual_payload=residual_bad,
        )
        with pytest.raises(ProcessIntelligenceError):
            diagnoser.diagnose(_residual_frame(), request=_request())

    naive = DiagnosisResult.model_construct(
        anomaly_id="5",
        task=AnalysisTask.RESIDUAL_ANOMALY,
        method_used=[DiagnosisMethod.ROBUST_Z_SCORE],
        scope=DiagnosisScope.SINGLE_EVENT,
        factors=[_factor("pressure")],
        confidence=0.8,
        analyzed_row_count=1,
        reference_row_count=5,
        caveats=["x"],
        generated_at=datetime(2024, 1, 1),
        metadata={},
    )
    naive_residual = DiagnosisResult.model_construct(
        anomaly_id="5",
        task=AnalysisTask.RESIDUAL_ANOMALY,
        method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
        scope=DiagnosisScope.SINGLE_EVENT,
        factors=[_factor("pressure")],
        confidence=0.8,
        analyzed_row_count=1,
        reference_row_count=5,
        caveats=["x"],
        generated_at=datetime(2024, 1, 1),
        metadata={},
    )
    diagnoser, _, _ = _ensemble_with_doubles(
        robust_payload=naive,
        residual_payload=naive_residual,
    )
    with pytest.raises(ProcessIntelligenceError):
        diagnoser.diagnose(_residual_frame(), request=_request())


def test_child_duplicate_factor_failure() -> None:
    duplicate = DiagnosisResult.model_construct(
        anomaly_id="5",
        task=AnalysisTask.RESIDUAL_ANOMALY,
        method_used=[DiagnosisMethod.ROBUST_Z_SCORE],
        scope=DiagnosisScope.SINGLE_EVENT,
        factors=[_factor("pressure"), _factor("pressure")],
        confidence=0.8,
        analyzed_row_count=1,
        reference_row_count=5,
        caveats=["x"],
        generated_at=datetime.now(UTC),
        metadata={},
    )
    duplicate_residual = DiagnosisResult.model_construct(
        anomaly_id="5",
        task=AnalysisTask.RESIDUAL_ANOMALY,
        method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
        scope=DiagnosisScope.SINGLE_EVENT,
        factors=[_factor("pressure"), _factor("pressure")],
        confidence=0.8,
        analyzed_row_count=1,
        reference_row_count=5,
        caveats=["x"],
        generated_at=datetime.now(UTC),
        metadata={},
    )
    diagnoser, _, _ = _ensemble_with_doubles(
        robust_payload=duplicate,
        residual_payload=duplicate_residual,
    )
    with pytest.raises(ProcessIntelligenceError):
        diagnoser.diagnose(_residual_frame(), request=_request())


# --- Failure isolation ---


def test_partial_success_continue_true() -> None:
    diagnoser, _, _ = _ensemble_with_doubles(
        robust_payload=ValueError("robust failed"),
        residual_payload=_result(
            method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
            factors=[_factor("pressure")],
        ),
        config=DiagnosisEnsembleConfig(continue_on_method_failure=True),
    )
    result = diagnoser.diagnose(_residual_frame(), request=_request())
    assert isinstance(result, DiagnosisResult)
    assert result.metadata["partial_ensemble"] is True
    assert result.metadata["robust_method_succeeded"] is False
    assert result.metadata["residual_method_succeeded"] is True
    assert any("ROBUST_GROUP_COMPARISON" in caveat for caveat in result.caveats)


def test_continue_false_propagates() -> None:
    diagnoser, _, _ = _ensemble_with_doubles(
        robust_payload=ValueError("robust failed"),
        residual_payload=_result(method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION]),
        config=DiagnosisEnsembleConfig(continue_on_method_failure=False),
    )
    with pytest.raises(ValueError, match="robust failed"):
        diagnoser.diagnose(_residual_frame(), request=_request())


def test_both_children_fail_message() -> None:
    diagnoser, _, _ = _ensemble_with_doubles(
        robust_payload=ValueError("robust boom"),
        residual_payload=TypeError("residual boom"),
    )
    with pytest.raises(ProcessIntelligenceError) as exc_info:
        diagnoser.diagnose(_residual_frame(), request=_request())
    message = str(exc_info.value)
    assert _SOURCE_ROBUST in message
    assert _SOURCE_RESIDUAL in message
    assert "ValueError" in message
    assert "TypeError" in message


def test_require_residual_true_fails_even_if_robust_ok() -> None:
    diagnoser, _, _ = _ensemble_with_doubles(
        robust_payload=_result(),
        residual_payload=ValueError("no residual"),
        config=DiagnosisEnsembleConfig(require_residual_method=True),
    )
    with pytest.raises(ProcessIntelligenceError):
        diagnoser.diagnose(_residual_frame(), request=_request())


def test_residual_columns_missing_partial_and_required() -> None:
    data = _robust_only_frame()
    request = _request(
        task=AnalysisTask.UNSUPERVISED_ANOMALY,
        anomaly_indicator_column=None,
        anomaly_score_column="anomaly_score",
    )
    diagnoser = DiagnosisEnsembleDiagnoser(
        config=DiagnosisEnsembleConfig(require_residual_method=False)
    )
    result = diagnoser.diagnose(data, request=request)
    assert isinstance(result, DiagnosisResult)
    assert result.metadata["robust_method_succeeded"] is True
    assert result.metadata["residual_method_succeeded"] is False
    assert result.metadata["partial_ensemble"] is True

    required = DiagnosisEnsembleDiagnoser(
        config=DiagnosisEnsembleConfig(require_residual_method=True)
    )
    with pytest.raises(ProcessIntelligenceError):
        required.diagnose(data, request=request)


def test_minimum_method_support_insufficient_sources() -> None:
    diagnoser, _, _ = _ensemble_with_doubles(
        robust_payload=ValueError("robust failed"),
        residual_payload=_result(method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION]),
        config=DiagnosisEnsembleConfig(minimum_method_support=2),
    )
    with pytest.raises(ProcessIntelligenceError):
        diagnoser.diagnose(_residual_frame(), request=_request())


def test_system_exception_not_isolated() -> None:
    diagnoser, _, _ = _ensemble_with_doubles(
        robust_payload=MemoryError("oom"),
        residual_payload=_result(method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION]),
    )
    with pytest.raises(MemoryError):
        diagnoser.diagnose(_residual_frame(), request=_request())


# --- Weighted RRF ---


def test_weighted_rrf_math_and_normalization() -> None:
    config = DiagnosisEnsembleConfig(
        robust_method_weight=0.75,
        residual_method_weight=0.25,
        reciprocal_rank_constant=10.0,
        normalize_ensemble_scores=True,
    )
    robust_payload = _result(
        factors=[
            _factor("pressure", confidence=0.9),
            _factor("temperature", confidence=0.5),
        ]
    )
    residual_payload = _result(
        method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
        factors=[
            _factor("temperature", confidence=0.8, deviation=-2.0, direction="NEGATIVE"),
            _factor("pressure", confidence=0.4),
        ],
    )
    diagnoser, _, _ = _ensemble_with_doubles(
        robust_payload=robust_payload,
        residual_payload=residual_payload,
        config=config,
    )
    result = diagnoser.diagnose(_residual_frame(), request=_request())
    assert isinstance(result, DiagnosisResult)
    assert [factor.variable for factor in result.factors][0] == "pressure"

    robust_w = 0.75 / 1.0
    residual_w = 0.25 / 1.0
    pressure_score = robust_w / (10.0 + 1) + residual_w / (10.0 + 2)
    temperature_score = robust_w / (10.0 + 2) + residual_w / (10.0 + 1)
    assert pressure_score > temperature_score

    pressure = next(factor for factor in result.factors if factor.variable == "pressure")
    assert "ROBUST_GROUP_COMPARISON" in pressure.evidence
    assert "rank=1" in pressure.evidence
    assert "normalized_ensemble_score=" in pressure.evidence
    assert "association does not establish causation" in pressure.evidence
    assert "proven root cause" not in pressure.evidence.lower()
    assert "will fix" not in pressure.evidence.lower()
    assert "반드시" not in pressure.evidence


def test_single_source_weight_one_and_zero_weight_contribution() -> None:
    diagnoser, _, _ = _ensemble_with_doubles(
        robust_payload=_result(factors=[_factor("pressure")]),
        residual_payload=ValueError("skip residual"),
        config=DiagnosisEnsembleConfig(
            robust_method_weight=0.3,
            residual_method_weight=0.7,
            reciprocal_rank_constant=5.0,
        ),
    )
    result = diagnoser.diagnose(_residual_frame(), request=_request())
    assert isinstance(result, DiagnosisResult)
    assert result.factors[0].variable == "pressure"
    assert "source_support=1" in result.factors[0].evidence

    both = DiagnosisEnsembleConfig(
        robust_method_weight=0.0,
        residual_method_weight=1.0,
        reciprocal_rank_constant=5.0,
    )
    diagnoser2, _, _ = _ensemble_with_doubles(
        robust_payload=_result(
            factors=[_factor("pressure"), _factor("temperature", confidence=0.2)]
        ),
        residual_payload=_result(
            method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
            factors=[_factor("temperature", confidence=0.9)],
        ),
        config=both,
    )
    fused = diagnoser2.diagnose(_residual_frame(), request=_request())
    assert isinstance(fused, DiagnosisResult)
    # robust weight 0 => pressure only from robust gets 0 contribution and may
    # still remain if support=1; temperature gets residual-only contribution.
    variables = [factor.variable for factor in fused.factors]
    assert "temperature" in variables


def test_normalize_false_and_support_filters() -> None:
    config = DiagnosisEnsembleConfig(
        normalize_ensemble_scores=False,
        minimum_method_support=2,
        reciprocal_rank_constant=10.0,
    )
    diagnoser, _, _ = _ensemble_with_doubles(
        robust_payload=_result(
            factors=[_factor("pressure"), _factor("temperature"), _factor("stable")]
        ),
        residual_payload=_result(
            method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
            factors=[_factor("pressure")],
        ),
        config=config,
    )
    result = diagnoser.diagnose(_residual_frame(), request=_request())
    assert isinstance(result, DiagnosisResult)
    assert [factor.variable for factor in result.factors] == ["pressure"]
    assert "normalized_ensemble_score=" in result.factors[0].evidence

    empty = DiagnosisEnsembleDiagnoser(
        config=DiagnosisEnsembleConfig(minimum_method_support=2),
        robust_diagnoser=RecordingDiagnoser(
            method=DiagnosisMethod.ENSEMBLE,
            payload=_result(factors=[_factor("pressure")]),
        ),
        residual_diagnoser=RecordingDiagnoser(
            method=DiagnosisMethod.RESIDUAL_ASSOCIATION,
            payload=_result(
                method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
                factors=[_factor("temperature")],
            ),
        ),
    )
    empty_result = empty.diagnose(_residual_frame(), request=_request())
    assert isinstance(empty_result, DiagnosisResult)
    assert empty_result.factors == []
    assert any("minimum method support" in caveat for caveat in empty_result.caveats)


# --- Confidence / deviation / direction ---


def test_confidence_penalty_and_deviation_direction() -> None:
    config = DiagnosisEnsembleConfig(
        robust_method_weight=0.5,
        residual_method_weight=0.5,
        penalize_partial_method_support=True,
        reciprocal_rank_constant=10.0,
    )
    diagnoser, _, _ = _ensemble_with_doubles(
        robust_payload=_result(
            factors=[
                _factor("pressure", confidence=1.0, deviation=4.0, direction="POSITIVE"),
                _factor("temperature", confidence=0.5, deviation=2.0, direction="POSITIVE"),
            ]
        ),
        residual_payload=_result(
            method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
            factors=[
                _factor(
                    "pressure",
                    confidence=0.0,
                    deviation=8.0,
                    direction="NEGATIVE",
                )
            ],
        ),
        config=config,
    )
    result = diagnoser.diagnose(_residual_frame(), request=_request())
    assert isinstance(result, DiagnosisResult)
    pressure = next(factor for factor in result.factors if factor.variable == "pressure")
    temperature = next(
        factor for factor in result.factors if factor.variable == "temperature"
    )
    assert pressure.direction == "MIXED"
    assert pressure.needs_verification is True
    assert "direction_conflict=True" in pressure.evidence
    # residual confidence 0 => weighted uses only robust deviation
    assert pressure.deviation == pytest.approx(4.0)

    zero_conf, _, _ = _ensemble_with_doubles(
        robust_payload=_result(
            factors=[_factor("pressure", confidence=0.0, deviation=4.0)]
        ),
        residual_payload=_result(
            method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
            factors=[_factor("pressure", confidence=0.0, deviation=8.0)],
        ),
        config=config,
    )
    zero_result = zero_conf.diagnose(_residual_frame(), request=_request())
    assert isinstance(zero_result, DiagnosisResult)
    assert zero_result.factors[0].deviation == pytest.approx(6.0)
    assert 0.0 <= pressure.confidence <= 1.0
    assert temperature.needs_verification is True  # partial support + penalty
    assert "causal probability" not in pressure.evidence.lower()

    no_penalty = DiagnosisEnsembleConfig(
        penalize_partial_method_support=False,
        reciprocal_rank_constant=10.0,
    )
    diagnoser2, _, _ = _ensemble_with_doubles(
        robust_payload=_result(factors=[_factor("temperature", confidence=0.8)]),
        residual_payload=_result(
            method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
            factors=[_factor("pressure", confidence=0.8)],
        ),
        config=no_penalty,
    )
    result2 = diagnoser2.diagnose(_residual_frame(), request=_request())
    assert isinstance(result2, DiagnosisResult)
    temp2 = next(factor for factor in result2.factors if factor.variable == "temperature")
    assert temp2.confidence == pytest.approx(0.8)
    assert temp2.needs_verification is False


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("POSITIVE", "POSITIVE", "POSITIVE"),
        ("NEGATIVE", "NEGATIVE", "NEGATIVE"),
        ("UNKNOWN", "POSITIVE", "POSITIVE"),
        ("UNKNOWN", "NEGATIVE", "NEGATIVE"),
        ("POSITIVE", "NEGATIVE", "MIXED"),
        ("MIXED", "POSITIVE", "MIXED"),
        ("UNKNOWN", "UNKNOWN", "UNKNOWN"),
    ],
)
def test_direction_consensus(left: str, right: str, expected: str) -> None:
    diagnoser, _, _ = _ensemble_with_doubles(
        robust_payload=_result(factors=[_factor("pressure", direction=left)]),
        residual_payload=_result(
            method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
            factors=[_factor("pressure", direction=right)],
        ),
    )
    result = diagnoser.diagnose(_residual_frame(), request=_request())
    assert isinstance(result, DiagnosisResult)
    assert result.factors[0].direction == expected


def test_role_controllable_and_verification_rules() -> None:
    diagnoser, _, _ = _ensemble_with_doubles(
        robust_payload=_result(
            factors=[
                _factor(
                    "pressure",
                    role=ColumnRole.CONTROLLABLE_PROCESS,
                    controllable=True,
                    needs_verification=False,
                )
            ]
        ),
        residual_payload=_result(
            method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
            factors=[
                _factor(
                    "pressure",
                    role=ColumnRole.STATE_SENSOR,
                    controllable=False,
                    needs_verification=True,
                )
            ],
        ),
    )
    result = diagnoser.diagnose(_residual_frame(), request=_request())
    assert isinstance(result, DiagnosisResult)
    factor = result.factors[0]
    assert factor.role is ColumnRole.UNKNOWN
    assert factor.controllable is False
    assert factor.needs_verification is True
    assert set(RootCauseFactor.model_fields) >= {
        "variable",
        "direction",
        "deviation",
        "role",
        "controllable",
        "evidence",
        "confidence",
        "needs_verification",
    }
    assert "score" not in RootCauseFactor.model_fields
    assert "rank" not in RootCauseFactor.model_fields


def test_child_evidence_include_exclude_and_order() -> None:
    robust_payload = _result(
        factors=[_factor("pressure", evidence="robust-only-evidence")]
    )
    residual_payload = _result(
        method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
        factors=[_factor("pressure", evidence="residual-only-evidence")],
    )
    include, _, _ = _ensemble_with_doubles(
        robust_payload=robust_payload,
        residual_payload=residual_payload,
        config=DiagnosisEnsembleConfig(include_child_evidence=True),
    )
    included = include.diagnose(_residual_frame(), request=_request())
    assert isinstance(included, DiagnosisResult)
    evidence = included.factors[0].evidence
    assert evidence.index(_SOURCE_ROBUST) < evidence.index(_SOURCE_RESIDUAL)
    assert "robust-only-evidence" in evidence
    assert "residual-only-evidence" in evidence

    exclude, _, _ = _ensemble_with_doubles(
        robust_payload=robust_payload,
        residual_payload=residual_payload,
        config=DiagnosisEnsembleConfig(include_child_evidence=False),
    )
    excluded = exclude.diagnose(_residual_frame(), request=_request())
    assert isinstance(excluded, DiagnosisResult)
    assert "robust-only-evidence" not in excluded.factors[0].evidence
    assert "residual-only-evidence" not in excluded.factors[0].evidence


def test_ranking_tie_breaks_and_top_k() -> None:
    config = DiagnosisEnsembleConfig(reciprocal_rank_constant=10.0)
    request = _request(
        feature_columns=["pressure", "temperature", "stable"],
        top_k_factors=2,
    )
    diagnoser, _, _ = _ensemble_with_doubles(
        robust_payload=_result(
            factors=[
                _factor("temperature", confidence=0.9),
                _factor("pressure", confidence=0.5, direction="POSITIVE"),
                _factor("stable", confidence=0.5, direction="POSITIVE"),
            ]
        ),
        residual_payload=_result(
            method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
            factors=[
                _factor("pressure", confidence=0.5, direction="NEGATIVE"),
                _factor("stable", confidence=0.9, direction="POSITIVE"),
            ],
        ),
        config=config,
    )
    result = diagnoser.diagnose(_residual_frame(), request=request)
    assert isinstance(result, DiagnosisResult)
    assert len(result.factors) == 2
    variables = [factor.variable for factor in result.factors]

    temp_score = 0.5 / 11.0
    pressure_score = 0.5 / 12.0 + 0.5 / 11.0
    stable_score = 0.5 / 13.0 + 0.5 / 12.0
    assert pressure_score > stable_score > temp_score
    assert variables == ["pressure", "stable"]
    assert len({factor.variable for factor in result.factors}) == len(result.factors)


# --- Result scopes ---


def test_single_event_result_contract() -> None:
    diagnoser, _, _ = _ensemble_with_doubles(
        robust_payload=_result(
            factors=[_factor("pressure")],
            method_used=[DiagnosisMethod.ROBUST_Z_SCORE, DiagnosisMethod.GROUP_COMPARISON],
        ),
        residual_payload=_result(
            factors=[_factor("pressure")],
            method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
        ),
    )
    result = diagnoser.diagnose(_residual_frame(), request=_request())
    assert isinstance(result, DiagnosisResult)
    assert result.anomaly_id == "5"
    assert result.task is AnalysisTask.RESIDUAL_ANOMALY
    assert result.scope is DiagnosisScope.SINGLE_EVENT
    assert result.analyzed_row_count == 1
    assert result.reference_row_count == 5
    assert result.method_used == [
        DiagnosisMethod.ROBUST_Z_SCORE,
        DiagnosisMethod.GROUP_COMPARISON,
        DiagnosisMethod.RESIDUAL_ASSOCIATION,
    ]
    assert result.generated_at.tzinfo is not None
    assert result.caveats[0] == (
        "ensemble attribution reflects association and does not establish causation"
    )
    assert result.caveats[1].startswith("ranking combines method-specific")
    assert len(result.caveats) == len(set(result.caveats))
    for value in result.metadata.values():
        assert value is None or isinstance(value, (str, int, float, bool))


def test_group_and_global_scopes() -> None:
    for scope in (DiagnosisScope.ANOMALY_GROUP, DiagnosisScope.GLOBAL):
        diagnoser, _, _ = _ensemble_with_doubles(
            robust_payload=_result(
                anomaly_id=None,
                scope=scope,
                analyzed_row_count=3,
                factors=[_factor("pressure")],
            ),
            residual_payload=_result(
                anomaly_id=None,
                scope=scope,
                analyzed_row_count=3,
                method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
                factors=[_factor("pressure")],
            ),
        )
        request = _request(scope=scope, events=[])
        result = diagnoser.diagnose(_residual_frame(), request=request)
        assert isinstance(result, DiagnosisResult)
        assert result.scope is scope
        assert result.anomaly_id is None
        assert result.analyzed_row_count == 3


# --- TOP_ANOMALIES / aggregate ---


def test_top_anomalies_alignment_and_aggregate() -> None:
    events = [_event("5"), _event("6"), _event("7")]
    robust_batch = _batch(
        [
            _result(
                anomaly_id="5",
                scope=DiagnosisScope.TOP_ANOMALIES,
                factors=[_factor("pressure", confidence=0.9), _factor("temperature")],
            ),
            _result(
                anomaly_id="6",
                scope=DiagnosisScope.TOP_ANOMALIES,
                factors=[_factor("pressure", confidence=0.8)],
            ),
            _result(
                anomaly_id="7",
                scope=DiagnosisScope.TOP_ANOMALIES,
                factors=[_factor("temperature", confidence=0.7, direction="NEGATIVE")],
            ),
        ]
    )
    residual_batch = _batch(
        [
            _result(
                anomaly_id="5",
                scope=DiagnosisScope.TOP_ANOMALIES,
                method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
                factors=[_factor("pressure", confidence=0.6)],
            ),
            _result(
                anomaly_id="6",
                scope=DiagnosisScope.TOP_ANOMALIES,
                method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
                factors=[_factor("stable", confidence=0.5)],
            ),
            _result(
                anomaly_id="7",
                scope=DiagnosisScope.TOP_ANOMALIES,
                method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
                factors=[_factor("temperature", confidence=0.7, direction="POSITIVE")],
            ),
        ]
    )
    diagnoser, _, _ = _ensemble_with_doubles(
        robust_payload=robust_batch,
        residual_payload=residual_batch,
    )
    request = _request(scope=DiagnosisScope.TOP_ANOMALIES, events=events)
    batch = diagnoser.diagnose(_residual_frame(), request=request)
    assert isinstance(batch, DiagnosisBatchResult)
    assert [item.anomaly_id for item in batch.results] == ["5", "6", "7"]
    assert batch.requested_event_count == 3
    assert batch.diagnosed_event_count == 3
    assert batch.failed_event_count == 0
    assert batch.warnings[0] == "ensemble attribution does not establish causation"
    assert len(batch.warnings) == len(set(batch.warnings))
    assert batch.generated_at.tzinfo is not None
    assert batch.metadata["ranking_method"] == "weighted_reciprocal_rank_fusion"
    assert batch.metadata["association_not_causation"] is True
    assert batch.aggregate_factors
    assert "association does not establish causation" in batch.aggregate_factors[0].evidence
    assert len({factor.variable for factor in batch.aggregate_factors}) == len(
        batch.aggregate_factors
    )


def test_top_anomalies_mismatch_failures() -> None:
    events = [_event("5"), _event("6")]
    robust_batch = _batch(
        [
            _result(anomaly_id="5", scope=DiagnosisScope.TOP_ANOMALIES),
            _result(anomaly_id="6", scope=DiagnosisScope.TOP_ANOMALIES),
        ]
    )
    residual_order = _batch(
        [
            _result(
                anomaly_id="6",
                scope=DiagnosisScope.TOP_ANOMALIES,
                method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
            ),
            _result(
                anomaly_id="5",
                scope=DiagnosisScope.TOP_ANOMALIES,
                method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
            ),
        ]
    )
    diagnoser, _, _ = _ensemble_with_doubles(
        robust_payload=robust_batch,
        residual_payload=residual_order,
    )
    with pytest.raises(ProcessIntelligenceError):
        diagnoser.diagnose(
            _residual_frame(),
            request=_request(scope=DiagnosisScope.TOP_ANOMALIES, events=events),
        )

    residual_count = _batch(
        [
            _result(
                anomaly_id="5",
                scope=DiagnosisScope.TOP_ANOMALIES,
                method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
            )
        ]
    )
    diagnoser2, _, _ = _ensemble_with_doubles(
        robust_payload=robust_batch,
        residual_payload=residual_count,
    )
    with pytest.raises(ProcessIntelligenceError):
        diagnoser2.diagnose(
            _residual_frame(),
            request=_request(scope=DiagnosisScope.TOP_ANOMALIES, events=events),
        )


# --- Real integration ---


def test_real_integration_residual_scored_data() -> None:
    diagnoser = DiagnosisEnsembleDiagnoser()
    data = _residual_frame()
    request = _request()
    snapshot = data.to_dicts()
    result = diagnoser.diagnose(data, request=request)
    assert isinstance(result, DiagnosisResult)
    assert result.scope is DiagnosisScope.SINGLE_EVENT
    assert result.factors
    assert data.to_dicts() == snapshot
    variables = [factor.variable for factor in result.factors]
    assert len(variables) == len(set(variables))
    assert any(
        "source_support=2" in factor.evidence or "source_support=1" in factor.evidence
        for factor in result.factors
    )

    group = diagnoser.diagnose(
        data,
        request=_request(scope=DiagnosisScope.ANOMALY_GROUP, events=[]),
    )
    assert isinstance(group, DiagnosisResult)
    assert group.scope is DiagnosisScope.ANOMALY_GROUP

    batch = diagnoser.diagnose(
        data,
        request=_request(
            scope=DiagnosisScope.TOP_ANOMALIES,
            events=[_event("5", score=0.95), _event("6", score=0.8), _event("7", score=0.7)],
        ),
    )
    assert isinstance(batch, DiagnosisBatchResult)
    assert batch.diagnosed_event_count == 3


def test_real_robust_only_without_residual_columns() -> None:
    diagnoser = DiagnosisEnsembleDiagnoser()
    data = _robust_only_frame()
    request = _request(
        task=AnalysisTask.UNSUPERVISED_ANOMALY,
        anomaly_indicator_column=None,
        anomaly_score_column="anomaly_score",
    )
    result = diagnoser.diagnose(data, request=request)
    assert isinstance(result, DiagnosisResult)
    assert result.metadata["robust_method_succeeded"] is True
    assert result.metadata["residual_method_succeeded"] is False


# --- State / regression ---


def test_no_cache_determinism_and_isolation() -> None:
    robust = RecordingDiagnoser(
        method=DiagnosisMethod.ENSEMBLE,
        payload=_result(factors=[_factor("pressure"), _factor("temperature")]),
    )
    residual = RecordingDiagnoser(
        method=DiagnosisMethod.RESIDUAL_ASSOCIATION,
        payload=_result(
            method_used=[DiagnosisMethod.RESIDUAL_ASSOCIATION],
            factors=[_factor("pressure")],
        ),
    )
    diagnoser = DiagnosisEnsembleDiagnoser(
        robust_diagnoser=robust,
        residual_diagnoser=residual,
    )
    data = _residual_frame()
    request = _request()
    first = diagnoser.diagnose(data, request=request)
    second = diagnoser.diagnose(data, request=request)
    assert isinstance(first, DiagnosisResult)
    assert isinstance(second, DiagnosisResult)
    assert [factor.variable for factor in first.factors] == [
        factor.variable for factor in second.factors
    ]
    assert [factor.confidence for factor in first.factors] == pytest.approx(
        [factor.confidence for factor in second.factors]
    )
    assert len(robust.calls) == 2
    assert len(residual.calls) == 2
    first.factors[0].confidence = 0.0
    third = diagnoser.diagnose(data, request=request)
    assert isinstance(third, DiagnosisResult)
    assert third.factors[0].confidence != 0.0
    assert robust.mutable_marker == {"value": 0}
    meta = diagnoser.get_metadata()
    assert "diagnoser" not in meta
    assert not any(isinstance(value, (pl.DataFrame, np.ndarray)) for value in meta.values())
