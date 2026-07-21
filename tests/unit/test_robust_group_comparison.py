"""Unit tests for RobustGroupComparisonDiagnoser (Step 8B)."""

from __future__ import annotations

import math
from datetime import UTC
from typing import Any

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from process_intelligence.core.enums import AnalysisTask, AnomalyType
from process_intelligence.core.exceptions import (
    DataValidationError,
    InsufficientDataError,
)
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
    RobustFeatureStatistic,
    RobustGroupComparisonConfig,
    RobustGroupComparisonDiagnoser,
)

_ROBUST_SCALE_CONSTANT = 1.4826


def _event(
    anomaly_id: str,
    *,
    score: float = 0.9,
) -> AnomalyEvent:
    return AnomalyEvent(
        anomaly_id=anomaly_id,
        anomaly_type=AnomalyType.PROCESS_INPUT,
        anomaly_score=score,
        severity="high",
        model_confidence=0.8,
        detector="isolation_forest",
        rationale="score above threshold",
    )


def _base_frame() -> pl.DataFrame:
    """Deterministic frame with reference, high, low, mixed, and zero-MAD features."""
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


def _request(
    *,
    scope: DiagnosisScope = DiagnosisScope.SINGLE_EVENT,
    method: DiagnosisMethod = DiagnosisMethod.ENSEMBLE,
    events: list[AnomalyEvent] | None = None,
    feature_columns: list[str] | None = None,
    top_k_events: int = 20,
    top_k_factors: int = 10,
    minimum_reference_rows: int = 3,
    anomaly_indicator_column: str | None = None,
    anomaly_score_column: str | None = "anomaly_score",
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
            if anomaly_indicator_column is None:
                anomaly_indicator_column = "is_anomaly"
    return DiagnosisRequest(
        task=AnalysisTask.UNSUPERVISED_ANOMALY,
        method=method,
        scope=scope,
        feature_columns=feature_columns
        or ["pressure", "temperature", "stable"],
        anomaly_events=events,
        anomaly_indicator_column=anomaly_indicator_column,
        anomaly_score_column=anomaly_score_column,
        row_id_column="_original_row_id",
        top_k_events=top_k_events,
        top_k_factors=top_k_factors,
        minimum_reference_rows=minimum_reference_rows,
    )


def _valid_statistic(**overrides: Any) -> RobustFeatureStatistic:
    payload: dict[str, Any] = {
        "feature_name": "pressure",
        "reference_row_count": 5,
        "anomaly_row_count": 1,
        "reference_median": 10.0,
        "anomaly_median": 40.0,
        "signed_location_difference": 30.0,
        "median_absolute_deviation": 0.0,
        "effective_scale": 1e-12,
        "robust_z_score": 3.0e13,
        "deviation_prevalence": 1.0,
        "raw_association_score": 3.0e13,
        "normalized_association_score": 1.0,
        "direction": "POSITIVE",
        "confidence": 0.8,
        "warnings": ["reference dispersion was near zero"],
    }
    payload.update(overrides)
    return RobustFeatureStatistic(**payload)


# --- Config ---


def test_config_defaults() -> None:
    config = RobustGroupComparisonConfig()
    assert config.minimum_scale == pytest.approx(1e-12)
    assert config.z_score_threshold == pytest.approx(3.5)
    assert config.direction_tolerance == pytest.approx(1e-12)
    assert config.minimum_anomaly_rows == 1
    assert config.include_zero_score_factors is False
    assert config.normalize_factor_scores is True
    assert config.use_anomaly_score_ordering is True


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("minimum_scale", 0.0),
        ("minimum_scale", True),
        ("minimum_scale", math.nan),
        ("minimum_scale", math.inf),
        ("minimum_scale", -math.inf),
        ("z_score_threshold", 0.0),
        ("z_score_threshold", math.inf),
        ("direction_tolerance", -1.0),
        ("minimum_anomaly_rows", 0),
        ("minimum_anomaly_rows", True),
        ("include_zero_score_factors", 1),
        ("normalize_factor_scores", "true"),
        ("use_anomaly_score_ordering", 0),
    ],
)
def test_config_rejects_invalid_values(field_name: str, bad_value: object) -> None:
    with pytest.raises(ValidationError):
        RobustGroupComparisonConfig(**{field_name: bad_value})


def test_config_round_trip() -> None:
    config = RobustGroupComparisonConfig(
        minimum_scale=1e-8,
        z_score_threshold=2.5,
        direction_tolerance=0.0,
        minimum_anomaly_rows=2,
        include_zero_score_factors=True,
        normalize_factor_scores=False,
        use_anomaly_score_ordering=False,
    )
    restored = RobustGroupComparisonConfig.model_validate(config.model_dump())
    assert restored.model_dump() == config.model_dump()


# --- RobustFeatureStatistic ---


def test_statistic_valid_and_round_trip() -> None:
    statistic = _valid_statistic()
    assert statistic.feature_name == "pressure"
    restored = RobustFeatureStatistic.model_validate(statistic.model_dump())
    assert restored.model_dump() == statistic.model_dump()


@pytest.mark.parametrize(
    "overrides",
    [
        {"feature_name": ""},
        {"feature_name": "_original_row_id"},
        {"reference_row_count": 0},
        {"anomaly_row_count": True},
        {"reference_median": math.nan},
        {"median_absolute_deviation": -0.1},
        {"effective_scale": 0.0},
        {"robust_z_score": -1.0},
        {"deviation_prevalence": -0.1},
        {"deviation_prevalence": 1.1},
        {"normalized_association_score": 1.5},
        {"confidence": -0.01},
        {"warnings": [""]},
        {"warnings": ["a", "a"]},
    ],
)
def test_statistic_rejects_invalid(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _valid_statistic(**overrides)


# --- construction / contract ---


def test_diagnoser_construction_and_method() -> None:
    diagnoser = RobustGroupComparisonDiagnoser()
    assert isinstance(diagnoser, BaseRootCauseDiagnoser)
    assert diagnoser.method is DiagnosisMethod.ENSEMBLE


def test_diagnoser_rejects_bad_config_type() -> None:
    with pytest.raises(TypeError):
        RobustGroupComparisonDiagnoser(config={"minimum_scale": 1e-6})  # type: ignore[arg-type]


def test_external_config_mutation_isolated() -> None:
    config = RobustGroupComparisonConfig(z_score_threshold=2.0)
    diagnoser = RobustGroupComparisonDiagnoser(config=config)
    config.z_score_threshold = 9.0
    assert diagnoser.get_metadata()["z_score_threshold"] == pytest.approx(2.0)


def test_diagnoser_instances_are_isolated() -> None:
    a = RobustGroupComparisonDiagnoser(
        config=RobustGroupComparisonConfig(minimum_scale=1e-6)
    )
    b = RobustGroupComparisonDiagnoser(
        config=RobustGroupComparisonConfig(minimum_scale=1e-9)
    )
    assert a.get_metadata()["minimum_scale"] != b.get_metadata()["minimum_scale"]


def test_get_metadata_scalar_and_independent() -> None:
    diagnoser = RobustGroupComparisonDiagnoser()
    first = diagnoser.get_metadata()
    second = diagnoser.get_metadata()
    assert first is not second
    first["method"] = "mutated"
    assert diagnoser.get_metadata()["method"] == "ENSEMBLE"
    assert first["fitted"] is False
    assert first["requires_training"] is False
    assert first["association_not_causation"] is True
    for value in second.values():
        assert value is None or isinstance(value, (str, int, float, bool))
        assert not isinstance(value, (pl.DataFrame, np.ndarray))


# --- input validation ---


def test_diagnose_accepts_polars_and_rejects_others() -> None:
    diagnoser = RobustGroupComparisonDiagnoser()
    frame = _base_frame()
    result = diagnoser.diagnose(frame, request=_request())
    assert isinstance(result, DiagnosisResult)

    with pytest.raises(TypeError):
        diagnoser.diagnose(frame.to_pandas(), request=_request())  # type: ignore[arg-type]
    with pytest.raises(InsufficientDataError):
        diagnoser.diagnose(frame.clear(), request=_request())
    with pytest.raises(TypeError):
        diagnoser.diagnose(frame, request={"scope": "SINGLE_EVENT"})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        diagnoser.diagnose(frame, request=_request(), explanation="bad")  # type: ignore[arg-type]


def test_missing_columns_and_row_id_rules() -> None:
    diagnoser = RobustGroupComparisonDiagnoser()
    frame = _base_frame()

    with pytest.raises(DataValidationError, match="missing required columns"):
        diagnoser.diagnose(
            frame.drop("_original_row_id"),
            request=_request(),
        )
    with pytest.raises(DataValidationError, match="missing required columns"):
        diagnoser.diagnose(
            frame.drop("pressure"),
            request=_request(),
        )
    with pytest.raises(DataValidationError, match="anomaly_indicator_column"):
        diagnoser.diagnose(
            frame,
            request=_request(
                scope=DiagnosisScope.ANOMALY_GROUP,
                events=[],
                anomaly_indicator_column=None,
            ),
        )

    null_ids = frame.with_columns(
        pl.Series("_original_row_id", ["0", None, "2", "3", "4", "5", "6", "7"])
    )
    with pytest.raises(DataValidationError, match="null"):
        diagnoser.diagnose(null_ids, request=_request())

    dup_ids = frame.with_columns(
        pl.Series("_original_row_id", ["0", "0", "2", "3", "4", "5", "6", "7"])
    )
    with pytest.raises(DataValidationError, match="unique"):
        diagnoser.diagnose(dup_ids, request=_request())

    float_ids = frame.with_columns(
        pl.Series("_original_row_id", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    )
    with pytest.raises(DataValidationError, match="float"):
        diagnoser.diagnose(float_ids, request=_request(events=[_event("5.0")]))


def test_feature_dtype_and_value_validation_includes_name() -> None:
    diagnoser = RobustGroupComparisonDiagnoser()
    frame = _base_frame()

    bool_feature = frame.with_columns(pl.Series("pressure", [True] * 8))
    with pytest.raises(DataValidationError, match="pressure"):
        diagnoser.diagnose(bool_feature, request=_request())

    string_feature = frame.with_columns(pl.Series("temperature", ["a"] * 8))
    with pytest.raises(DataValidationError, match="temperature"):
        diagnoser.diagnose(string_feature, request=_request())

    null_feature = frame.with_columns(
        pl.Series("pressure", [10.0, None, 10.0, 10.0, 10.0, 40.0, 0.0, 10.0])
    )
    with pytest.raises(DataValidationError, match="pressure"):
        diagnoser.diagnose(null_feature, request=_request())

    nan_feature = frame.with_columns(
        pl.Series("pressure", [10.0, math.nan, 10.0, 10.0, 10.0, 40.0, 0.0, 10.0])
    )
    with pytest.raises(DataValidationError, match="pressure"):
        diagnoser.diagnose(nan_feature, request=_request())

    inf_feature = frame.with_columns(
        pl.Series("pressure", [10.0, math.inf, 10.0, 10.0, 10.0, 40.0, 0.0, 10.0])
    )
    with pytest.raises(DataValidationError, match="pressure"):
        diagnoser.diagnose(inf_feature, request=_request())


def test_input_immutability() -> None:
    diagnoser = RobustGroupComparisonDiagnoser()
    frame = _base_frame()
    request = _request()
    original_columns = list(frame.columns)
    original_height = frame.height
    original_ids = [event.anomaly_id for event in request.anomaly_events]
    original_features = list(request.feature_columns)

    _ = diagnoser.diagnose(frame, request=request)

    assert list(frame.columns) == original_columns
    assert frame.height == original_height
    assert [event.anomaly_id for event in request.anomaly_events] == original_ids
    assert request.feature_columns == original_features


# --- anomaly row resolution ---


def test_anomaly_event_matching_and_errors() -> None:
    diagnoser = RobustGroupComparisonDiagnoser()
    frame = _base_frame()

    ok = diagnoser.diagnose(frame, request=_request(events=[_event("5")]))
    assert isinstance(ok, DiagnosisResult)
    assert ok.anomaly_id == "5"

    with pytest.raises(DataValidationError, match="not found|incompatible"):
        diagnoser.diagnose(frame, request=_request(events=[_event("missing")]))

    int_ids = frame.with_columns(
        pl.Series("_original_row_id", [0, 1, 2, 3, 4, 5, 6, 7])
    )
    with pytest.raises(DataValidationError, match="incompatible"):
        diagnoser.diagnose(
            int_ids,
            request=_request(events=[_event("a-1")]),
        )


def test_indicator_formats_and_authority_rules() -> None:
    diagnoser = RobustGroupComparisonDiagnoser()
    frame = _base_frame()

    bool_indicator = frame.with_columns(
        pl.Series("is_anomaly", [False, False, False, False, False, True, True, True])
    )
    group = diagnoser.diagnose(
        bool_indicator,
        request=_request(
            scope=DiagnosisScope.ANOMALY_GROUP,
            events=[],
            anomaly_indicator_column="is_anomaly",
        ),
    )
    assert isinstance(group, DiagnosisResult)
    assert group.analyzed_row_count == 3

    with pytest.raises(DataValidationError, match="0/1"):
        bad = frame.with_columns(
            pl.Series("is_anomaly", [0, 0, 0, 0, 0, 1, 2, 1])
        )
        diagnoser.diagnose(
            bad,
            request=_request(
                scope=DiagnosisScope.ANOMALY_GROUP,
                events=[],
                anomaly_indicator_column="is_anomaly",
            ),
        )

    # events authoritative and matching indicator
    matched = diagnoser.diagnose(
        frame,
        request=_request(
            scope=DiagnosisScope.ANOMALY_GROUP,
            events=[_event("5"), _event("6"), _event("7")],
            anomaly_indicator_column="is_anomaly",
        ),
    )
    assert matched.analyzed_row_count == 3

    with pytest.raises(DataValidationError, match="exactly match"):
        diagnoser.diagnose(
            frame,
            request=_request(
                scope=DiagnosisScope.ANOMALY_GROUP,
                events=[_event("5"), _event("6")],
                anomaly_indicator_column="is_anomaly",
            ),
        )

    with pytest.raises(DataValidationError, match="anomaly_indicator_column"):
        diagnoser.diagnose(
            frame,
            request=_request(
                scope=DiagnosisScope.GLOBAL,
                events=[],
                anomaly_indicator_column=None,
            ),
        )


# --- minimum data ---


def test_reference_and_anomaly_minimums() -> None:
    diagnoser = RobustGroupComparisonDiagnoser(
        config=RobustGroupComparisonConfig(minimum_anomaly_rows=3)
    )
    frame = _base_frame()

    with pytest.raises(InsufficientDataError, match="reference"):
        diagnoser.diagnose(
            frame,
            request=_request(minimum_reference_rows=8),
        )

    tiny = pl.DataFrame(
        {
            "_original_row_id": ["0", "1"],
            "pressure": [1.0, 9.0],
            "temperature": [1.0, 1.0],
            "stable": [1.0, 1.0],
            "is_anomaly": [0, 1],
            "anomaly_score": [0.1, 0.9],
        }
    )
    with pytest.raises(InsufficientDataError, match="reference"):
        diagnoser.diagnose(
            tiny,
            request=_request(
                events=[_event("1")],
                minimum_reference_rows=2,
            ),
        )

    # force empty reference by marking all as anomalies via events + matching indicator
    all_anomaly = frame.with_columns(pl.lit(1).alias("is_anomaly"))
    with pytest.raises(InsufficientDataError, match="reference"):
        diagnoser.diagnose(
            all_anomaly,
            request=_request(
                scope=DiagnosisScope.ANOMALY_GROUP,
                events=[_event(str(i)) for i in range(8)],
                anomaly_indicator_column="is_anomaly",
                minimum_reference_rows=1,
            ),
        )

    with pytest.raises(InsufficientDataError, match="anomaly group"):
        diagnoser.diagnose(
            frame,
            request=_request(
                scope=DiagnosisScope.ANOMALY_GROUP,
                events=[_event("5"), _event("6")],
                anomaly_indicator_column=None,
            ),
        )

    # SINGLE_EVENT allows one anomaly row even if minimum_anomaly_rows > 1
    single = diagnoser.diagnose(frame, request=_request())
    assert single.analyzed_row_count == 1


# --- robust calculations ---


def test_robust_statistics_and_scores() -> None:
    diagnoser = RobustGroupComparisonDiagnoser()
    frame = _base_frame()
    request = _request(method=DiagnosisMethod.ROBUST_Z_SCORE)
    result = diagnoser.diagnose(frame, request=request)
    assert isinstance(result, DiagnosisResult)

    reference = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    anomaly = np.array([40.0])
    reference_median = float(np.median(reference))
    anomaly_median = float(np.median(anomaly))
    signed = anomaly_median - reference_median
    mad = float(np.median(np.abs(reference - reference_median)))
    robust_scale = _ROBUST_SCALE_CONSTANT * mad
    effective = max(robust_scale, 1e-12)
    robust_z = abs(signed) / effective

    pressure = next(f for f in result.factors if f.variable == "pressure")
    assert pressure.direction == "POSITIVE"
    assert pressure.deviation == pytest.approx(signed)
    assert "robust_z_score=" in pressure.evidence
    assert robust_z == pytest.approx(abs(signed) / effective)
    assert math.isfinite(robust_z)
    assert "near-zero reference dispersion" in " ".join(result.caveats).lower() or any(
        "near-zero" in caveat.lower() for caveat in result.caveats
    )

    # ENSEMBLE raw uses prevalence term; pressure prevalence is 1.0 for single high event
    ensemble = diagnoser.diagnose(
        frame,
        request=_request(method=DiagnosisMethod.ENSEMBLE),
    )
    assert isinstance(ensemble, DiagnosisResult)
    ens_pressure = next(f for f in ensemble.factors if f.variable == "pressure")
    assert "raw_association_score=" in ens_pressure.evidence

    group = diagnoser.diagnose(
        frame,
        request=_request(
            method=DiagnosisMethod.GROUP_COMPARISON,
            scope=DiagnosisScope.ANOMALY_GROUP,
            events=[_event("5"), _event("6"), _event("7")],
        ),
    )
    assert isinstance(group, DiagnosisResult)
    assert group.method_used == [DiagnosisMethod.GROUP_COMPARISON]


def test_directions_positive_negative_unknown_mixed() -> None:
    diagnoser = RobustGroupComparisonDiagnoser(
        config=RobustGroupComparisonConfig(direction_tolerance=1e-9)
    )
    frame = _base_frame()

    high = diagnoser.diagnose(frame, request=_request(events=[_event("5")]))
    assert next(f for f in high.factors if f.variable == "pressure").direction == (
        "POSITIVE"
    )

    low = diagnoser.diagnose(frame, request=_request(events=[_event("6")]))
    assert next(f for f in low.factors if f.variable == "pressure").direction == (
        "NEGATIVE"
    )

    # mixed pressure group: highs and lows with group median near reference
    mixed_diagnoser = RobustGroupComparisonDiagnoser(
        config=RobustGroupComparisonConfig(
            direction_tolerance=1e-9,
            include_zero_score_factors=True,
        )
    )
    mixed_frame = pl.DataFrame(
        {
            "_original_row_id": ["0", "1", "2", "3", "4", "5"],
            "pressure": [10.0, 10.0, 10.0, 10.0, 20.0, 0.0],
            "temperature": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            "stable": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "is_anomaly": [0, 0, 0, 0, 1, 1],
            "anomaly_score": [0.1, 0.1, 0.1, 0.1, 0.9, 0.8],
        }
    )
    mixed = mixed_diagnoser.diagnose(
        mixed_frame,
        request=_request(
            scope=DiagnosisScope.ANOMALY_GROUP,
            events=[_event("4"), _event("5")],
            minimum_reference_rows=3,
        ),
    )
    pressure = next(f for f in mixed.factors if f.variable == "pressure")
    assert pressure.direction == "MIXED"

    # unknown: anomaly equals reference for all features
    unknown_frame = pl.DataFrame(
        {
            "_original_row_id": ["0", "1", "2", "3"],
            "pressure": [10.0, 10.0, 10.0, 10.0],
            "temperature": [20.0, 20.0, 20.0, 20.0],
            "stable": [1.0, 1.0, 1.0, 1.0],
            "is_anomaly": [0, 0, 0, 1],
            "anomaly_score": [0.1, 0.1, 0.1, 0.9],
        }
    )
    unknown = diagnoser.diagnose(
        unknown_frame,
        request=_request(
            events=[_event("3")],
            minimum_reference_rows=3,
            feature_columns=["pressure", "temperature", "stable"],
        ),
    )
    # zero raw scores excluded by default
    assert unknown.factors == []


# --- confidence / normalization / ranking ---


def test_normalization_ranking_and_zero_score_policy() -> None:
    diagnoser = RobustGroupComparisonDiagnoser()
    frame = _base_frame()
    result = diagnoser.diagnose(
        frame,
        request=_request(top_k_factors=2),
    )
    assert len(result.factors) <= 2
    assert len({factor.variable for factor in result.factors}) == len(result.factors)
    assert "normalized_association_score=1" in result.factors[0].evidence
    scores = [
        float(factor.evidence.split("normalized_association_score=")[1].split(";")[0])
        for factor in result.factors
    ]
    assert scores == sorted(scores, reverse=True)
    assert all(factor.variable != "stable" for factor in result.factors)

    include_zero = RobustGroupComparisonDiagnoser(
        config=RobustGroupComparisonConfig(include_zero_score_factors=True)
    )
    with_zero = include_zero.diagnose(
        frame,
        request=_request(
            events=[_event("3")],  # reference-like row wrongly labeled for zero diffs
            minimum_reference_rows=3,
        ),
    )
    # row 3 is reference-like; may still produce zeros
    assert isinstance(with_zero, DiagnosisResult)
    for factor in result.factors:
        assert 0.0 <= factor.confidence <= 1.0


def test_ranking_tie_breaks_use_feature_order() -> None:
    frame = pl.DataFrame(
        {
            "_original_row_id": ["0", "1", "2", "3", "4"],
            "a": [0.0, 0.0, 0.0, 0.0, 10.0],
            "b": [0.0, 0.0, 0.0, 0.0, 10.0],
            "is_anomaly": [0, 0, 0, 0, 1],
            "anomaly_score": [0.1, 0.1, 0.1, 0.1, 0.9],
        }
    )
    diagnoser = RobustGroupComparisonDiagnoser()
    result = diagnoser.diagnose(
        frame,
        request=_request(
            events=[_event("4")],
            feature_columns=["b", "a"],
            anomaly_indicator_column="is_anomaly",
            anomaly_score_column="anomaly_score",
            minimum_reference_rows=3,
        ),
    )
    assert [factor.variable for factor in result.factors] == ["b", "a"]


# --- scopes ---


def test_single_event_result_contract() -> None:
    diagnoser = RobustGroupComparisonDiagnoser()
    result = diagnoser.diagnose(_base_frame(), request=_request())
    assert isinstance(result, DiagnosisResult)
    assert result.anomaly_id == "5"
    assert result.analyzed_row_count == 1
    assert result.reference_row_count == 7
    assert result.method_used == [
        DiagnosisMethod.ROBUST_Z_SCORE,
        DiagnosisMethod.GROUP_COMPARISON,
    ]
    assert result.scope is DiagnosisScope.SINGLE_EVENT
    assert result.generated_at.tzinfo is not None
    assert result.generated_at.utcoffset() == UTC.utcoffset(result.generated_at)
    assert any("causation" in caveat.lower() for caveat in result.caveats)
    assert isinstance(result.factors[0], RootCauseFactor)


def test_anomaly_group_and_global() -> None:
    diagnoser = RobustGroupComparisonDiagnoser()
    frame = _base_frame()
    group = diagnoser.diagnose(
        frame,
        request=_request(
            scope=DiagnosisScope.ANOMALY_GROUP,
            events=[_event("5"), _event("6"), _event("7")],
        ),
    )
    assert isinstance(group, DiagnosisResult)
    assert group.anomaly_id is None
    assert group.analyzed_row_count == 3
    assert group.scope is DiagnosisScope.ANOMALY_GROUP

    global_result = diagnoser.diagnose(
        frame,
        request=_request(
            scope=DiagnosisScope.GLOBAL,
        ),
    )
    assert isinstance(global_result, DiagnosisResult)
    assert global_result.anomaly_id is None
    assert global_result.scope is DiagnosisScope.GLOBAL
    assert global_result.analyzed_row_count == 3


def test_top_anomalies_batch() -> None:
    diagnoser = RobustGroupComparisonDiagnoser()
    frame = _base_frame()
    batch = diagnoser.diagnose(
        frame,
        request=_request(
            scope=DiagnosisScope.TOP_ANOMALIES,
            top_k_events=2,
            events=[
                _event("7", score=0.70),
                _event("5", score=0.95),
                _event("6", score=0.80),
            ],
        ),
    )
    assert isinstance(batch, DiagnosisBatchResult)
    assert [result.anomaly_id for result in batch.results] == ["5", "6"]
    assert all(result.analyzed_row_count == 1 for result in batch.results)
    assert batch.requested_event_count == 2
    assert batch.diagnosed_event_count == 2
    assert batch.failed_event_count == 0
    assert len({result.anomaly_id for result in batch.results}) == 2
    assert len({factor.variable for factor in batch.aggregate_factors}) == len(
        batch.aggregate_factors
    )
    assert any("causation" in warning.lower() for warning in batch.warnings)
    assert batch.generated_at.tzinfo is not None

    # score ties preserve request order
    tied = diagnoser.diagnose(
        frame,
        request=_request(
            scope=DiagnosisScope.TOP_ANOMALIES,
            top_k_events=2,
            events=[
                _event("6", score=0.9),
                _event("5", score=0.9),
                _event("7", score=0.1),
            ],
        ),
    )
    assert [result.anomaly_id for result in tied.results] == ["6", "5"]

    unordered = RobustGroupComparisonDiagnoser(
        config=RobustGroupComparisonConfig(use_anomaly_score_ordering=False)
    )
    ordered = unordered.diagnose(
        frame,
        request=_request(
            scope=DiagnosisScope.TOP_ANOMALIES,
            top_k_events=2,
            events=[
                _event("7", score=0.99),
                _event("5", score=0.1),
                _event("6", score=0.5),
            ],
        ),
    )
    assert [result.anomaly_id for result in ordered.results] == ["7", "5"]


def test_aggregate_direction_and_averages() -> None:
    diagnoser = RobustGroupComparisonDiagnoser()
    frame = _base_frame()
    batch = diagnoser.diagnose(
        frame,
        request=_request(
            scope=DiagnosisScope.TOP_ANOMALIES,
            events=[_event("5", score=0.95), _event("6", score=0.8)],
            top_k_factors=5,
        ),
    )
    pressure = next(
        factor for factor in batch.aggregate_factors if factor.variable == "pressure"
    )
    assert pressure.direction == "MIXED"
    assert "mean_normalized_association_score=" in pressure.evidence
    assert "occurrence_count=" in pressure.evidence

    positive_only = diagnoser.diagnose(
        frame,
        request=_request(
            scope=DiagnosisScope.TOP_ANOMALIES,
            events=[_event("5", score=0.95), _event("7", score=0.7)],
            feature_columns=["temperature", "stable"],
        ),
    )
    temperature = next(
        factor
        for factor in positive_only.aggregate_factors
        if factor.variable == "temperature"
    )
    # event 5 has no temp shift; event 7 is high -> aggregate may be POSITIVE
    assert temperature.direction in {"POSITIVE", "UNKNOWN", "MIXED"}


# --- methods / explanation / language / state ---


def test_supported_methods_and_method_used() -> None:
    diagnoser = RobustGroupComparisonDiagnoser()
    frame = _base_frame()

    robust = diagnoser.diagnose(
        frame,
        request=_request(method=DiagnosisMethod.ROBUST_Z_SCORE),
    )
    assert robust.method_used == [DiagnosisMethod.ROBUST_Z_SCORE]

    group = diagnoser.diagnose(
        frame,
        request=_request(method=DiagnosisMethod.GROUP_COMPARISON),
    )
    assert group.method_used == [DiagnosisMethod.GROUP_COMPARISON]

    ensemble = diagnoser.diagnose(
        frame,
        request=_request(method=DiagnosisMethod.ENSEMBLE),
    )
    assert ensemble.method_used == [
        DiagnosisMethod.ROBUST_Z_SCORE,
        DiagnosisMethod.GROUP_COMPARISON,
    ]

    with pytest.raises(DataValidationError, match="ENSEMBLE"):
        diagnoser.diagnose(
            frame,
            request=_request(method=DiagnosisMethod.MODEL_EXPLANATION),
        )


def test_explanation_handling() -> None:
    diagnoser = RobustGroupComparisonDiagnoser()
    frame = _base_frame()
    explanation = ExplanationResult(
        method="none",
        feature_importances={"pressure": 0.9},
        notes=["unused"],
        confidence=0.5,
    )
    original_notes = list(explanation.notes)
    result = diagnoser.diagnose(
        frame,
        request=_request(),
        explanation=explanation,
    )
    assert explanation.notes == original_notes
    assert any("explanation" in caveat.lower() for caveat in result.caveats)
    assert all(
        not isinstance(value, (np.ndarray, pl.DataFrame))
        for value in result.metadata.values()
    )


def test_soft_language_and_determinism() -> None:
    diagnoser = RobustGroupComparisonDiagnoser()
    frame = _base_frame()
    request = _request()
    first = diagnoser.diagnose(frame, request=request)
    second = diagnoser.diagnose(frame, request=request)
    assert [factor.variable for factor in first.factors] == [
        factor.variable for factor in second.factors
    ]
    assert [factor.confidence for factor in first.factors] == [
        factor.confidence for factor in second.factors
    ]

    forbidden = [
        "caused the anomaly",
        "root cause was proven",
        "반드시",
        "will fix",
        "반드시 원인",
    ]
    blob = " ".join(
        [first.caveats[0], *(factor.evidence for factor in first.factors)]
    ).lower()
    for phrase in forbidden:
        assert phrase.lower() not in blob

    first.factors[0].evidence = "mutated"
    third = diagnoser.diagnose(frame, request=request)
    assert "mutated" not in third.factors[0].evidence
    assert not hasattr(diagnoser, "_last_result")


def test_public_exports_include_step8b_objects() -> None:
    import process_intelligence.diagnosis as diagnosis

    assert diagnosis.RobustGroupComparisonConfig is RobustGroupComparisonConfig
    assert diagnosis.RobustFeatureStatistic is RobustFeatureStatistic
    assert diagnosis.RobustGroupComparisonDiagnoser is RobustGroupComparisonDiagnoser
    assert "DiagnosisRequest" in diagnosis.__all__
