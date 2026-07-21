"""Unit tests for diagnosis enums and Pydantic schemas (Step 8A)."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from process_intelligence.core.enums import AnalysisTask, AnomalyType, ColumnRole
from process_intelligence.core.schemas import AnomalyEvent, RootCauseFactor
from process_intelligence.diagnosis import (
    BaseRootCauseDiagnoser,
    DiagnosisBatchResult,
    DiagnosisMethod,
    DiagnosisOutcome,
    DiagnosisRequest,
    DiagnosisResult,
    DiagnosisScope,
)


def _event(anomaly_id: str = "a-1", *, score: float = 0.9) -> AnomalyEvent:
    return AnomalyEvent(
        anomaly_id=anomaly_id,
        anomaly_type=AnomalyType.PROCESS_INPUT,
        anomaly_score=score,
        severity="high",
        model_confidence=0.8,
        detector="isolation_forest",
        rationale="score above threshold",
    )


def _factor(variable: str = "pressure") -> RootCauseFactor:
    return RootCauseFactor(
        variable=variable,
        direction="increase",
        deviation=1.5,
        role=ColumnRole.CONTROLLABLE_PROCESS,
        controllable=True,
        evidence="Associated with higher anomaly score in the analyzed cohort.",
        confidence=0.7,
        needs_verification=True,
    )


def _valid_request_kwargs(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task": AnalysisTask.UNSUPERVISED_ANOMALY,
        "method": DiagnosisMethod.ROBUST_Z_SCORE,
        "scope": DiagnosisScope.TOP_ANOMALIES,
        "feature_columns": ["pressure", "temperature"],
        "anomaly_events": [_event("a-1"), _event("a-2")],
        "target_column": None,
        "anomaly_indicator_column": "is_anomaly",
        "anomaly_score_column": "anomaly_score",
        "row_id_column": "_original_row_id",
        "top_k_events": 20,
        "top_k_factors": 10,
        "minimum_reference_rows": 5,
        "metadata": {"seed": 7},
    }
    payload.update(overrides)
    return payload


def _valid_result_kwargs(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "anomaly_id": "a-1",
        "task": AnalysisTask.UNSUPERVISED_ANOMALY,
        "method_used": [DiagnosisMethod.ROBUST_Z_SCORE],
        "scope": DiagnosisScope.SINGLE_EVENT,
        "factors": [_factor("pressure"), _factor("temperature")],
        "confidence": 0.75,
        "analyzed_row_count": 3,
        "reference_row_count": 10,
        "caveats": [
            "Association only; causation is not established.",
        ],
        "generated_at": datetime(2026, 7, 21, 9, 0, tzinfo=UTC),
        "metadata": {"version": 1},
    }
    payload.update(overrides)
    return payload


def _valid_batch_kwargs(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "results": [
            DiagnosisResult(**_valid_result_kwargs(anomaly_id="a-1")),
            DiagnosisResult(**_valid_result_kwargs(anomaly_id="a-2")),
        ],
        "aggregate_factors": [_factor("pressure")],
        "requested_event_count": 2,
        "diagnosed_event_count": 2,
        "failed_event_count": 0,
        "generated_at": datetime(2026, 7, 21, 9, 5, tzinfo=UTC),
        "warnings": ["Top-N truncated to requested events."],
        "metadata": {"batch": True},
    }
    payload.update(overrides)
    return payload


# --- enums ---


def test_diagnosis_method_members_and_values() -> None:
    expected = {
        "ROBUST_Z_SCORE": "ROBUST_Z_SCORE",
        "GROUP_COMPARISON": "GROUP_COMPARISON",
        "MODEL_EXPLANATION": "MODEL_EXPLANATION",
        "RESIDUAL_ASSOCIATION": "RESIDUAL_ASSOCIATION",
        "DISTRIBUTION_SHIFT": "DISTRIBUTION_SHIFT",
        "RULE_BASED": "RULE_BASED",
        "ENSEMBLE": "ENSEMBLE",
    }
    assert {member.name: member.value for member in DiagnosisMethod} == expected
    assert len(DiagnosisMethod) == 7
    assert DiagnosisMethod.__doc__


def test_diagnosis_scope_members_and_values() -> None:
    expected = {
        "SINGLE_EVENT": "SINGLE_EVENT",
        "TOP_ANOMALIES": "TOP_ANOMALIES",
        "ANOMALY_GROUP": "ANOMALY_GROUP",
        "GLOBAL": "GLOBAL",
    }
    assert {member.name: member.value for member in DiagnosisScope} == expected
    assert len(DiagnosisScope) == 4
    assert DiagnosisScope.__doc__


# --- DiagnosisRequest ---


def test_request_valid_top_anomalies() -> None:
    request = DiagnosisRequest(**_valid_request_kwargs())
    assert request.scope is DiagnosisScope.TOP_ANOMALIES
    assert len(request.anomaly_events) == 2
    assert isinstance(request.anomaly_events[0], AnomalyEvent)
    assert request.feature_columns == ["pressure", "temperature"]


def test_request_single_event_requires_exactly_one() -> None:
    ok = DiagnosisRequest(
        **_valid_request_kwargs(
            scope=DiagnosisScope.SINGLE_EVENT,
            anomaly_events=[_event("only")],
        )
    )
    assert len(ok.anomaly_events) == 1

    with pytest.raises(ValidationError):
        DiagnosisRequest(
            **_valid_request_kwargs(
                scope=DiagnosisScope.SINGLE_EVENT,
                anomaly_events=[_event("a"), _event("b")],
            )
        )
    with pytest.raises(ValidationError):
        DiagnosisRequest(
            **_valid_request_kwargs(
                scope=DiagnosisScope.SINGLE_EVENT,
                anomaly_events=[],
            )
        )


def test_request_top_anomalies_requires_at_least_one() -> None:
    with pytest.raises(ValidationError):
        DiagnosisRequest(
            **_valid_request_kwargs(
                scope=DiagnosisScope.TOP_ANOMALIES,
                anomaly_events=[],
            )
        )


@pytest.mark.parametrize(
    "scope",
    [DiagnosisScope.ANOMALY_GROUP, DiagnosisScope.GLOBAL],
)
def test_request_group_or_global_allows_empty_events(scope: DiagnosisScope) -> None:
    request = DiagnosisRequest(
        **_valid_request_kwargs(scope=scope, anomaly_events=[])
    )
    assert request.anomaly_events == []


def test_request_rejects_feature_role_column_overlap() -> None:
    with pytest.raises(ValidationError):
        DiagnosisRequest(
            **_valid_request_kwargs(
                feature_columns=["pressure", "anomaly_score"],
                anomaly_score_column="anomaly_score",
            )
        )
    with pytest.raises(ValidationError):
        DiagnosisRequest(
            **_valid_request_kwargs(
                feature_columns=["pressure", "_original_row_id"],
            )
        )
    with pytest.raises(ValidationError):
        DiagnosisRequest(
            **_valid_request_kwargs(
                target_column="quality",
                anomaly_score_column="quality",
            )
        )


def test_request_rejects_empty_or_duplicate_features() -> None:
    with pytest.raises(ValidationError):
        DiagnosisRequest(**_valid_request_kwargs(feature_columns=[]))
    with pytest.raises(ValidationError):
        DiagnosisRequest(**_valid_request_kwargs(feature_columns=["", "x"]))
    with pytest.raises(ValidationError):
        DiagnosisRequest(
            **_valid_request_kwargs(feature_columns=["pressure", "pressure"])
        )


def test_request_rejects_blank_optional_columns() -> None:
    with pytest.raises(ValidationError):
        DiagnosisRequest(**_valid_request_kwargs(target_column="   "))
    with pytest.raises(ValidationError):
        DiagnosisRequest(**_valid_request_kwargs(row_id_column=""))


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("top_k_events", 0),
        ("top_k_factors", -1),
        ("minimum_reference_rows", 0),
        ("top_k_events", True),
        ("top_k_factors", 1.5),
        ("minimum_reference_rows", "5"),
    ],
)
def test_request_rejects_invalid_topk_and_counts(
    field_name: str,
    bad_value: object,
) -> None:
    with pytest.raises(ValidationError):
        DiagnosisRequest(**_valid_request_kwargs(**{field_name: bad_value}))


def test_request_rejects_non_scalar_or_nonfinite_metadata() -> None:
    with pytest.raises(ValidationError):
        DiagnosisRequest(**_valid_request_kwargs(metadata={"x": math.nan}))
    with pytest.raises(ValidationError):
        DiagnosisRequest(**_valid_request_kwargs(metadata={"x": math.inf}))
    with pytest.raises(ValidationError):
        DiagnosisRequest(**_valid_request_kwargs(metadata={"x": [1, 2]}))
    with pytest.raises(ValidationError):
        DiagnosisRequest(**_valid_request_kwargs(metadata={"x": {"a": 1}}))


def test_request_rejects_duplicate_anomaly_ids() -> None:
    with pytest.raises(ValidationError):
        DiagnosisRequest(
            **_valid_request_kwargs(
                anomaly_events=[_event("dup"), _event("dup")],
            )
        )


def test_request_uses_core_anomaly_event() -> None:
    request = DiagnosisRequest(**_valid_request_kwargs())
    assert type(request.anomaly_events[0]) is AnomalyEvent
    assert request.anomaly_events[0].anomaly_id == "a-1"


def test_request_mutable_collection_independence() -> None:
    features = ["pressure", "temperature"]
    events = [_event("a-1"), _event("a-2")]
    metadata = {"seed": 1}
    request = DiagnosisRequest(
        **_valid_request_kwargs(
            feature_columns=features,
            anomaly_events=events,
            metadata=metadata,
        )
    )
    features.append("gas_flow")
    events.append(_event("a-3"))
    metadata["seed"] = 99
    assert request.feature_columns == ["pressure", "temperature"]
    assert [event.anomaly_id for event in request.anomaly_events] == ["a-1", "a-2"]
    assert request.metadata == {"seed": 1}

    first = DiagnosisRequest(**_valid_request_kwargs())
    second = DiagnosisRequest(**_valid_request_kwargs())
    first.metadata["mutated"] = True
    assert "mutated" not in second.metadata


def test_request_round_trip() -> None:
    request = DiagnosisRequest(**_valid_request_kwargs())
    restored = DiagnosisRequest.model_validate(request.model_dump())
    assert restored.model_dump() == request.model_dump()
    assert restored.anomaly_events[0].anomaly_id == "a-1"


# --- DiagnosisResult ---


def test_result_valid_and_uses_core_root_cause_factor() -> None:
    result = DiagnosisResult(**_valid_result_kwargs())
    assert type(result.factors[0]) is RootCauseFactor
    assert result.factors[0].variable == "pressure"
    assert result.method_used == [DiagnosisMethod.ROBUST_Z_SCORE]


def test_result_rejects_duplicate_methods_and_factors() -> None:
    with pytest.raises(ValidationError):
        DiagnosisResult(
            **_valid_result_kwargs(
                method_used=[
                    DiagnosisMethod.ROBUST_Z_SCORE,
                    DiagnosisMethod.ROBUST_Z_SCORE,
                ],
            )
        )
    with pytest.raises(ValidationError):
        DiagnosisResult(
            **_valid_result_kwargs(
                factors=[_factor("pressure"), _factor("pressure")],
            )
        )
    with pytest.raises(ValidationError):
        DiagnosisResult(**_valid_result_kwargs(method_used=[]))


def test_result_treats_factor_order_as_ranking() -> None:
    result = DiagnosisResult(
        **_valid_result_kwargs(
            factors=[_factor("temperature"), _factor("pressure")],
        )
    )
    assert [factor.variable for factor in result.factors] == [
        "temperature",
        "pressure",
    ]
    assert not hasattr(result.factors[0], "rank")


@pytest.mark.parametrize("bad_confidence", [-0.01, 1.01, math.nan, math.inf, True])
def test_result_rejects_invalid_confidence(bad_confidence: object) -> None:
    with pytest.raises(ValidationError):
        DiagnosisResult(**_valid_result_kwargs(confidence=bad_confidence))


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("analyzed_row_count", 0),
        ("analyzed_row_count", True),
        ("reference_row_count", -1),
        ("reference_row_count", False),
        ("analyzed_row_count", 1.5),
    ],
)
def test_result_rejects_invalid_row_counts(
    field_name: str,
    bad_value: object,
) -> None:
    with pytest.raises(ValidationError):
        DiagnosisResult(**_valid_result_kwargs(**{field_name: bad_value}))


def test_result_rejects_naive_datetime_and_duplicate_caveats() -> None:
    with pytest.raises(ValidationError):
        DiagnosisResult(
            **_valid_result_kwargs(
                generated_at=datetime(2026, 7, 21, 9, 0),
            )
        )
    with pytest.raises(ValidationError):
        DiagnosisResult(**_valid_result_kwargs(caveats=["a", "a"]))
    with pytest.raises(ValidationError):
        DiagnosisResult(**_valid_result_kwargs(caveats=[""]))


def test_result_timezone_aware_generated_at() -> None:
    result = DiagnosisResult(**_valid_result_kwargs())
    assert result.generated_at.tzinfo is not None
    assert result.generated_at.utcoffset() is not None


def test_result_mutable_independence_and_round_trip() -> None:
    factors = [_factor("pressure")]
    caveats = ["Association only."]
    metadata = {"k": 1}
    result = DiagnosisResult(
        **_valid_result_kwargs(
            factors=factors,
            caveats=caveats,
            metadata=metadata,
        )
    )
    factors.append(_factor("temperature"))
    caveats.append("extra")
    metadata["k"] = 9
    assert [factor.variable for factor in result.factors] == ["pressure"]
    assert result.caveats == ["Association only."]
    assert result.metadata == {"k": 1}

    restored = DiagnosisResult.model_validate(result.model_dump())
    assert restored.model_dump() == result.model_dump()


# --- DiagnosisBatchResult ---


def test_batch_valid() -> None:
    batch = DiagnosisBatchResult(**_valid_batch_kwargs())
    assert batch.requested_event_count == 2
    assert batch.diagnosed_event_count + batch.failed_event_count == 2


def test_batch_count_relationship_exact() -> None:
    ok = DiagnosisBatchResult(
        **_valid_batch_kwargs(
            results=[DiagnosisResult(**_valid_result_kwargs(anomaly_id="a-1"))],
            requested_event_count=3,
            diagnosed_event_count=1,
            failed_event_count=2,
        )
    )
    assert ok.failed_event_count == 2

    with pytest.raises(ValidationError):
        DiagnosisBatchResult(
            **_valid_batch_kwargs(
                requested_event_count=3,
                diagnosed_event_count=1,
                failed_event_count=1,
            )
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("requested_event_count", True),
        ("diagnosed_event_count", -1),
        ("failed_event_count", 1.0),
    ],
)
def test_batch_rejects_invalid_counts(field_name: str, bad_value: object) -> None:
    with pytest.raises(ValidationError):
        DiagnosisBatchResult(**_valid_batch_kwargs(**{field_name: bad_value}))


def test_batch_rejects_duplicate_result_anomaly_ids() -> None:
    with pytest.raises(ValidationError):
        DiagnosisBatchResult(
            **_valid_batch_kwargs(
                results=[
                    DiagnosisResult(**_valid_result_kwargs(anomaly_id="same")),
                    DiagnosisResult(**_valid_result_kwargs(anomaly_id="same")),
                ],
            )
        )


def test_batch_rejects_duplicate_aggregate_factors_and_warnings() -> None:
    with pytest.raises(ValidationError):
        DiagnosisBatchResult(
            **_valid_batch_kwargs(
                aggregate_factors=[_factor("pressure"), _factor("pressure")],
            )
        )
    with pytest.raises(ValidationError):
        DiagnosisBatchResult(**_valid_batch_kwargs(warnings=["a", "a"]))
    with pytest.raises(ValidationError):
        DiagnosisBatchResult(**_valid_batch_kwargs(warnings=[""]))


def test_batch_mutable_independence_and_round_trip() -> None:
    warnings = ["note"]
    metadata = {"batch": True}
    batch = DiagnosisBatchResult(
        **_valid_batch_kwargs(warnings=warnings, metadata=metadata)
    )
    warnings.append("extra")
    metadata["batch"] = False
    assert batch.warnings == ["note"]
    assert batch.metadata == {"batch": True}

    first = DiagnosisBatchResult(**_valid_batch_kwargs())
    second = DiagnosisBatchResult(**_valid_batch_kwargs())
    first.warnings.append("mutated")
    assert second.warnings == ["Top-N truncated to requested events."]

    restored = DiagnosisBatchResult.model_validate(batch.model_dump())
    assert restored.model_dump() == batch.model_dump()


# --- DiagnosisOutcome ---


def test_outcome_frozen_slots_and_types() -> None:
    import dataclasses

    assert dataclasses.is_dataclass(DiagnosisOutcome)
    assert DiagnosisOutcome.__dataclass_params__.frozen is True  # type: ignore[attr-defined]
    assert DiagnosisOutcome.__slots__ == ("result",)

    result = DiagnosisResult(**_valid_result_kwargs())
    outcome = DiagnosisOutcome(result=result)
    assert outcome.result is result

    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.result = DiagnosisResult(  # type: ignore[misc]
            **_valid_result_kwargs(anomaly_id="other")
        )

    batch = DiagnosisBatchResult(**_valid_batch_kwargs())
    batch_outcome = DiagnosisOutcome(result=batch)
    assert isinstance(batch_outcome.result, DiagnosisBatchResult)


# --- public import ---


def test_public_package_exports() -> None:
    import process_intelligence.diagnosis as diagnosis

    assert diagnosis.DiagnosisMethod is DiagnosisMethod
    assert diagnosis.DiagnosisScope is DiagnosisScope
    assert diagnosis.DiagnosisRequest is DiagnosisRequest
    assert diagnosis.DiagnosisResult is DiagnosisResult
    assert diagnosis.DiagnosisBatchResult is DiagnosisBatchResult
    assert diagnosis.DiagnosisOutcome is DiagnosisOutcome
    assert diagnosis.BaseRootCauseDiagnoser is BaseRootCauseDiagnoser
    assert "AnomalyEvent" not in diagnosis.__all__
    assert "RootCauseFactor" not in diagnosis.__all__
    assert "ExplanationResult" not in diagnosis.__all__
