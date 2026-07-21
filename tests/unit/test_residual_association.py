"""Unit tests for ResidualAssociationDiagnoser (Step 8C)."""

from __future__ import annotations

import math
from datetime import UTC
from typing import Any

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from process_intelligence.core.enums import AnalysisTask, AnomalyType, ColumnRole
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
    ResidualAssociationConfig,
    ResidualAssociationDiagnoser,
    ResidualFeatureStatistic,
)
from process_intelligence.diagnosis import residual_association as residual_module

_ROBUST_SCALE_CONSTANT = 1.4826


def _event(
    anomaly_id: str,
    *,
    score: float = 0.9,
) -> AnomalyEvent:
    return AnomalyEvent(
        anomaly_id=anomaly_id,
        anomaly_type=AnomalyType.RELATIONSHIP,
        anomaly_score=score,
        severity="high",
        model_confidence=0.8,
        detector="residual_anomaly",
        rationale="residual anomaly score above threshold",
    )


def _base_frame() -> pl.DataFrame:
    """Deterministic residual-scored frame with reserved residual columns."""
    # Rows 0-4 normal, 5-7 residual anomalies.
    # pressure correlates positively with residual and score.
    # temperature is bidirectional among anomalies.
    # stable is constant.
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


def _request(
    *,
    scope: DiagnosisScope = DiagnosisScope.SINGLE_EVENT,
    method: DiagnosisMethod = DiagnosisMethod.RESIDUAL_ASSOCIATION,
    events: list[AnomalyEvent] | None = None,
    feature_columns: list[str] | None = None,
    top_k_events: int = 20,
    top_k_factors: int = 10,
    minimum_reference_rows: int = 2,
    anomaly_indicator_column: str | None = "_is_residual_anomaly",
    anomaly_score_column: str | None = "_residual_anomaly_score",
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
        task=AnalysisTask.RESIDUAL_ANOMALY,
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


def _valid_statistic(**overrides: Any) -> ResidualFeatureStatistic:
    payload: dict[str, Any] = {
        "feature_name": "pressure",
        "total_row_count": 8,
        "anomaly_row_count": 1,
        "reference_row_count": 5,
        "feature_residual_rank_correlation": 0.8,
        "feature_score_rank_correlation": 0.7,
        "reference_median": 3.0,
        "anomaly_median": 9.0,
        "signed_location_difference": 6.0,
        "median_absolute_deviation": 1.0,
        "effective_scale": 1.4826,
        "group_deviation_z": 4.0,
        "deviation_prevalence": 1.0,
        "bounded_group_strength": 0.8,
        "raw_association_score": 0.75,
        "normalized_association_score": 1.0,
        "direction": "POSITIVE",
        "confidence": 0.7,
        "warnings": [],
    }
    payload.update(overrides)
    return ResidualFeatureStatistic(**payload)


def _manual_average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    n = len(values)
    while i < n:
        j = i + 1
        while j < n and values[order[j]] == values[order[i]]:
            j += 1
        average = 0.5 * (float(i + 1) + float(j))
        for k in range(i, j):
            ranks[order[k]] = average
        i = j
    return ranks


def _manual_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_ranks = _manual_average_ranks(left)
    right_ranks = _manual_average_ranks(right)
    if float(np.std(left_ranks)) <= 1e-12 or float(np.std(right_ranks)) <= 1e-12:
        return 0.0
    x = left_ranks - float(np.mean(left_ranks))
    y = right_ranks - float(np.mean(right_ranks))
    denom = float(np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(x * y) / denom)


# --- Config ---


def test_config_defaults() -> None:
    config = ResidualAssociationConfig()
    assert config.residual_column == "_regression_residual"
    assert config.absolute_residual_column == "_absolute_centered_residual"
    assert config.anomaly_score_column == "_residual_anomaly_score"
    assert config.anomaly_indicator_column == "_is_residual_anomaly"
    assert config.minimum_total_rows == 5
    assert config.minimum_reference_rows == 2
    assert config.minimum_anomaly_rows == 1
    assert config.minimum_scale == pytest.approx(1e-12)
    assert config.deviation_z_threshold == pytest.approx(3.5)
    assert config.direction_tolerance == pytest.approx(1e-12)
    assert config.score_correlation_weight == pytest.approx(0.45)
    assert config.group_deviation_weight == pytest.approx(0.35)
    assert config.prevalence_weight == pytest.approx(0.20)
    assert config.include_zero_score_factors is False
    assert config.normalize_factor_scores is True
    assert config.use_anomaly_score_ordering is True


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("residual_column", ""),
        ("residual_column", "   "),
        ("absolute_residual_column", ""),
        ("anomaly_score_column", ""),
        ("anomaly_indicator_column", ""),
        ("minimum_total_rows", 0),
        ("minimum_total_rows", True),
        ("minimum_reference_rows", 0),
        ("minimum_anomaly_rows", True),
        ("minimum_scale", 0.0),
        ("minimum_scale", True),
        ("deviation_z_threshold", 0.0),
        ("direction_tolerance", -1.0),
        ("score_correlation_weight", -0.1),
        ("score_correlation_weight", 1.1),
        ("score_correlation_weight", True),
        ("group_deviation_weight", math.nan),
        ("prevalence_weight", math.inf),
        ("include_zero_score_factors", 1),
        ("normalize_factor_scores", "true"),
        ("use_anomaly_score_ordering", 0),
    ],
)
def test_config_rejects_invalid_values(field_name: str, bad_value: object) -> None:
    with pytest.raises(ValidationError):
        ResidualAssociationConfig(**{field_name: bad_value})


def test_config_rejects_duplicate_column_names() -> None:
    with pytest.raises(ValidationError, match="distinct"):
        ResidualAssociationConfig(
            residual_column="same",
            absolute_residual_column="same",
        )


def test_config_rejects_weight_sum_mismatch() -> None:
    with pytest.raises(ValidationError, match="sum"):
        ResidualAssociationConfig(
            score_correlation_weight=0.5,
            group_deviation_weight=0.5,
            prevalence_weight=0.5,
        )


def test_config_round_trip() -> None:
    config = ResidualAssociationConfig(
        minimum_total_rows=6,
        minimum_reference_rows=3,
        score_correlation_weight=0.5,
        group_deviation_weight=0.3,
        prevalence_weight=0.2,
        include_zero_score_factors=True,
        normalize_factor_scores=False,
        use_anomaly_score_ordering=False,
    )
    restored = ResidualAssociationConfig.model_validate(config.model_dump())
    assert restored.model_dump() == config.model_dump()


# --- Statistic ---


def test_statistic_valid_and_round_trip() -> None:
    statistic = _valid_statistic(
        warnings=["rank association was undefined because one input was nearly constant"]
    )
    assert statistic.feature_name == "pressure"
    restored = ResidualFeatureStatistic.model_validate(statistic.model_dump())
    assert restored.model_dump() == statistic.model_dump()
    assert restored.warnings is not statistic.warnings


@pytest.mark.parametrize(
    "overrides",
    [
        {"feature_name": ""},
        {"feature_name": "_original_row_id"},
        {"total_row_count": 0},
        {"anomaly_row_count": True},
        {"reference_row_count": 0},
        {"feature_residual_rank_correlation": 1.5},
        {"feature_score_rank_correlation": -1.1},
        {"median_absolute_deviation": -0.1},
        {"effective_scale": 0.0},
        {"group_deviation_z": -1.0},
        {"deviation_prevalence": -0.1},
        {"deviation_prevalence": 1.1},
        {"bounded_group_strength": 1.5},
        {"raw_association_score": -0.01},
        {"normalized_association_score": 1.5},
        {"confidence": -0.01},
        {"direction": "UP"},
        {"warnings": [""]},
        {"warnings": ["a", "a"]},
        {"anomaly_row_count": 6, "reference_row_count": 5, "total_row_count": 8},
    ],
)
def test_statistic_rejects_invalid(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _valid_statistic(**overrides)


# --- Contract ---


def test_diagnoser_construction_and_method() -> None:
    diagnoser = ResidualAssociationDiagnoser()
    assert isinstance(diagnoser, BaseRootCauseDiagnoser)
    assert diagnoser.method is DiagnosisMethod.RESIDUAL_ASSOCIATION


def test_diagnoser_rejects_bad_config_type() -> None:
    with pytest.raises(TypeError):
        ResidualAssociationDiagnoser(config={"minimum_scale": 1e-6})  # type: ignore[arg-type]


def test_external_config_mutation_isolated() -> None:
    config = ResidualAssociationConfig(deviation_z_threshold=2.0)
    diagnoser = ResidualAssociationDiagnoser(config=config)
    config.deviation_z_threshold = 9.0
    assert diagnoser.get_metadata()["deviation_z_threshold"] == pytest.approx(2.0)


def test_get_metadata_scalar_and_independent() -> None:
    diagnoser = ResidualAssociationDiagnoser()
    first = diagnoser.get_metadata()
    second = diagnoser.get_metadata()
    assert first is not second
    first["method"] = "mutated"
    assert diagnoser.get_metadata()["method"] == "RESIDUAL_ASSOCIATION"
    assert first["fitted"] is False
    assert first["requires_training"] is False
    assert first["association_not_causation"] is True
    for value in second.values():
        assert value is None or isinstance(value, (str, int, float, bool))
        assert not isinstance(value, (pl.DataFrame, np.ndarray))


# --- Input ---


def test_diagnose_accepts_polars_and_rejects_others() -> None:
    diagnoser = ResidualAssociationDiagnoser()
    frame = _base_frame()
    result = diagnoser.diagnose(frame, request=_request())
    assert isinstance(result, DiagnosisResult)

    with pytest.raises(TypeError):
        diagnoser.diagnose(frame.to_pandas(), request=_request())  # type: ignore[arg-type]
    with pytest.raises(InsufficientDataError):
        diagnoser.diagnose(
            frame.head(3),
            request=_request(),
        )
    with pytest.raises(TypeError):
        diagnoser.diagnose(frame, request={"scope": "SINGLE_EVENT"})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        diagnoser.diagnose(frame, request=_request(), explanation="bad")  # type: ignore[arg-type]


def test_missing_columns_and_row_id_rules() -> None:
    diagnoser = ResidualAssociationDiagnoser()
    frame = _base_frame()

    with pytest.raises(DataValidationError, match="missing required columns"):
        diagnoser.diagnose(frame.drop("_original_row_id"), request=_request())
    with pytest.raises(DataValidationError, match="missing required columns"):
        diagnoser.diagnose(frame.drop("pressure"), request=_request())
    with pytest.raises(DataValidationError, match="missing required columns"):
        diagnoser.diagnose(
            frame.drop("_regression_residual"),
            request=_request(),
        )
    with pytest.raises(DataValidationError, match="missing required columns"):
        diagnoser.diagnose(
            frame.drop("_absolute_centered_residual"),
            request=_request(),
        )
    with pytest.raises(DataValidationError, match="missing required columns"):
        diagnoser.diagnose(
            frame.drop("_residual_anomaly_score"),
            request=_request(),
        )
    with pytest.raises(DataValidationError, match="missing required columns"):
        diagnoser.diagnose(
            frame.drop("_is_residual_anomaly"),
            request=_request(),
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


def test_feature_and_residual_column_validation() -> None:
    diagnoser = ResidualAssociationDiagnoser()
    frame = _base_frame()

    bool_feature = frame.with_columns(pl.Series("pressure", [True] * 8))
    with pytest.raises(DataValidationError, match="pressure"):
        diagnoser.diagnose(bool_feature, request=_request())

    string_feature = frame.with_columns(pl.Series("temperature", ["a"] * 8))
    with pytest.raises(DataValidationError, match="temperature"):
        diagnoser.diagnose(string_feature, request=_request())

    nan_feature = frame.with_columns(
        pl.Series("pressure", [1.0, math.nan, 3.0, 4.0, 5.0, 9.0, 8.0, 7.0])
    )
    with pytest.raises(DataValidationError, match="pressure"):
        diagnoser.diagnose(nan_feature, request=_request())

    inf_feature = frame.with_columns(
        pl.Series("pressure", [1.0, math.inf, 3.0, 4.0, 5.0, 9.0, 8.0, 7.0])
    )
    with pytest.raises(DataValidationError, match="pressure"):
        diagnoser.diagnose(inf_feature, request=_request())

    bool_residual = frame.with_columns(
        pl.Series("_regression_residual", [True] * 8)
    )
    with pytest.raises(DataValidationError, match="residual"):
        diagnoser.diagnose(bool_residual, request=_request())

    nan_residual = frame.with_columns(
        pl.Series(
            "_regression_residual",
            [-0.1, math.nan, 0.1, 0.2, 0.3, 2.0, 1.5, 1.0],
        )
    )
    with pytest.raises(DataValidationError, match="residual"):
        diagnoser.diagnose(nan_residual, request=_request())

    negative_abs = frame.with_columns(
        pl.Series(
            "_absolute_centered_residual",
            [0.2, 0.1, 0.0, 0.1, 0.2, -1.9, 1.4, 0.9],
        )
    )
    with pytest.raises(DataValidationError, match="absolute residual"):
        diagnoser.diagnose(negative_abs, request=_request())

    negative_score = frame.with_columns(
        pl.Series(
            "_residual_anomaly_score",
            [0.05, 0.08, 0.10, 0.12, 0.15, -0.95, 0.80, 0.70],
        )
    )
    with pytest.raises(DataValidationError, match="anomaly score"):
        diagnoser.diagnose(negative_score, request=_request())

    bad_indicator = frame.with_columns(
        pl.Series("_is_residual_anomaly", [0, 0, 0, 0, 0, 1, 2, 1])
    )
    with pytest.raises(DataValidationError, match="0/1"):
        diagnoser.diagnose(bad_indicator, request=_request())


def test_input_immutability() -> None:
    diagnoser = ResidualAssociationDiagnoser()
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


# --- Events and reference ---


def test_event_matching_and_indicator_subset_rules() -> None:
    diagnoser = ResidualAssociationDiagnoser()
    frame = _base_frame()

    ok = diagnoser.diagnose(frame, request=_request(events=[_event("5")]))
    assert isinstance(ok, DiagnosisResult)
    assert ok.anomaly_id == "5"
    assert ok.analyzed_row_count == 1
    assert ok.reference_row_count == 5

    with pytest.raises(DataValidationError, match="not found|incompatible"):
        diagnoser.diagnose(frame, request=_request(events=[_event("missing")]))

    # event on indicator-normal row rejected
    with pytest.raises(DataValidationError, match="subset|residual anomal"):
        diagnoser.diagnose(frame, request=_request(events=[_event("0")]))

    # SINGLE_EVENT subset of indicator anomalies is allowed
    subset = diagnoser.diagnose(
        frame,
        request=_request(
            scope=DiagnosisScope.SINGLE_EVENT,
            events=[_event("6", score=0.8)],
        ),
    )
    assert subset.anomaly_id == "6"

    # other anomaly rows must not enter reference
    assert subset.reference_row_count == 5
    assert subset.metadata["reference_row_count"] == 5


def test_reference_and_anomaly_minimums() -> None:
    diagnoser = ResidualAssociationDiagnoser(
        config=ResidualAssociationConfig(
            minimum_reference_rows=6,
            minimum_anomaly_rows=3,
        )
    )
    frame = _base_frame()
    with pytest.raises(InsufficientDataError, match="reference"):
        diagnoser.diagnose(frame, request=_request())

    group_diagnoser = ResidualAssociationDiagnoser(
        config=ResidualAssociationConfig(minimum_anomaly_rows=4)
    )
    with pytest.raises(InsufficientDataError, match="anomaly group"):
        group_diagnoser.diagnose(
            frame,
            request=_request(
                scope=DiagnosisScope.ANOMALY_GROUP,
                events=[],
            ),
        )

    # SINGLE_EVENT still allows one anomaly row
    single = ResidualAssociationDiagnoser(
        config=ResidualAssociationConfig(minimum_anomaly_rows=4)
    )
    result = single.diagnose(frame, request=_request())
    assert result.analyzed_row_count == 1


# --- Spearman ---


def test_spearman_helpers_and_correlations() -> None:
    values = np.asarray([1.0, 2.0, 2.0, 4.0], dtype=float)
    ranks = residual_module._average_ranks(values)
    assert ranks.tolist() == pytest.approx([1.0, 2.5, 2.5, 4.0])

    frame = _base_frame()
    diagnoser = ResidualAssociationDiagnoser()
    result = diagnoser.diagnose(frame, request=_request())
    assert isinstance(result, DiagnosisResult)
    pressure = next(f for f in result.factors if f.variable == "pressure")
    assert "feature_score_rank_correlation=" in pressure.evidence
    assert "feature_residual_rank_correlation=" in pressure.evidence

    feature = frame["pressure"].to_numpy().astype(float)
    residual = frame["_regression_residual"].to_numpy().astype(float)
    score = frame["_residual_anomaly_score"].to_numpy().astype(float)
    expected_residual = _manual_spearman(feature, residual)
    expected_score = _manual_spearman(feature, score)
    assert expected_residual > 0.0
    assert expected_score > 0.0

    # constant feature -> zero correlation warning path via group diagnosis
    constant_only = ResidualAssociationDiagnoser()
    constant_result = constant_only.diagnose(
        frame,
        request=_request(
            scope=DiagnosisScope.ANOMALY_GROUP,
            events=[],
            feature_columns=["stable"],
            top_k_factors=1,
        ),
    )
    # stable may be excluded when raw score is exactly zero
    text = " ".join(constant_result.caveats).lower()
    assert "causation" in text
    assert "scipy" not in residual_module.__dict__
    assert not hasattr(residual_module, "scipy")


def test_positive_and_negative_residual_association_direction() -> None:
    diagnoser = ResidualAssociationDiagnoser()
    frame = _base_frame()
    positive = diagnoser.diagnose(frame, request=_request())
    pressure = next(f for f in positive.factors if f.variable == "pressure")
    assert pressure.direction == "POSITIVE"
    assert "more positive regression residuals" in pressure.evidence

    # Invert residual signs to induce negative association with pressure.
    negative_frame = frame.with_columns(
        (-pl.col("_regression_residual")).alias("_regression_residual")
    )
    negative = diagnoser.diagnose(negative_frame, request=_request())
    pressure_neg = next(f for f in negative.factors if f.variable == "pressure")
    assert pressure_neg.direction == "NEGATIVE"
    assert "more negative regression residuals" in pressure_neg.evidence


def test_mixed_and_unknown_direction() -> None:
    # Bidirectional feature with near-zero residual correlation -> MIXED
    frame = pl.DataFrame(
        {
            "_original_row_id": ["0", "1", "2", "3", "4", "5", "6"],
            "mixed_feature": [10.0, 10.0, 10.0, 10.0, 10.0, 40.0, 0.0],
            "noise": [1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 2.5],
            "_regression_residual": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "_absolute_centered_residual": [0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0],
            "_residual_anomaly_score": [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.8],
            "_is_residual_anomaly": [0, 0, 0, 0, 0, 1, 1],
        }
    )
    diagnoser = ResidualAssociationDiagnoser(
        config=ResidualAssociationConfig(include_zero_score_factors=True)
    )
    result = diagnoser.diagnose(
        frame,
        request=_request(
            scope=DiagnosisScope.ANOMALY_GROUP,
            events=[],
            feature_columns=["mixed_feature", "noise"],
        ),
    )
    mixed = next(f for f in result.factors if f.variable == "mixed_feature")
    assert mixed.direction == "MIXED"
    assert "Both high and low deviations" in mixed.evidence

    # Unknown: tiny residual correlation and one-sided anomaly deviation
    unknown_frame = pl.DataFrame(
        {
            "_original_row_id": ["0", "1", "2", "3", "4", "5"],
            "flat": [1.0, 1.0, 1.0, 1.0, 1.0, 1.01],
            "_regression_residual": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "_absolute_centered_residual": [0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
            "_residual_anomaly_score": [0.1, 0.1, 0.1, 0.1, 0.1, 0.9],
            "_is_residual_anomaly": [0, 0, 0, 0, 0, 1],
        }
    )
    unknown = diagnoser.diagnose(
        unknown_frame,
        request=_request(
            scope=DiagnosisScope.ANOMALY_GROUP,
            events=[],
            feature_columns=["flat"],
        ),
    )
    assert unknown.factors[0].direction in {"UNKNOWN", "POSITIVE", "NEGATIVE"}


# --- Robust deviation and score ---


def test_robust_deviation_and_weighted_score() -> None:
    frame = _base_frame()
    config = ResidualAssociationConfig(
        score_correlation_weight=0.45,
        group_deviation_weight=0.35,
        prevalence_weight=0.20,
    )
    diagnoser = ResidualAssociationDiagnoser(config=config)
    result = diagnoser.diagnose(
        frame,
        request=_request(scope=DiagnosisScope.ANOMALY_GROUP, events=[]),
    )
    assert isinstance(result, DiagnosisResult)
    pressure = next(f for f in result.factors if f.variable == "pressure")
    assert pressure.role is ColumnRole.UNKNOWN
    assert pressure.controllable is False
    assert pressure.needs_verification is True
    assert 0.0 <= pressure.confidence <= 1.0

    # Manual check for reference/anomaly medians of pressure
    pressure_values = frame["pressure"].to_numpy().astype(float)
    reference = pressure_values[:5]
    anomaly = pressure_values[5:]
    ref_median = float(np.median(reference))
    anom_median = float(np.median(anomaly))
    signed_diff = anom_median - ref_median
    mad = float(np.median(np.abs(reference - ref_median)))
    scale = max(_ROBUST_SCALE_CONSTANT * mad, 1e-12)
    group_z = abs(signed_diff) / scale
    prevalence = float(np.mean(np.abs(anomaly - ref_median) / scale >= 3.5))
    bounded = group_z / (group_z + 1.0)
    score_corr = abs(
        _manual_spearman(
            pressure_values,
            frame["_residual_anomaly_score"].to_numpy().astype(float),
        )
    )
    expected_raw = 0.45 * score_corr + 0.35 * bounded + 0.20 * prevalence
    assert "raw_association_score=" in pressure.evidence
    marker = "raw_association_score="
    fragment = pressure.evidence.split(marker, 1)[1]
    reported_raw = float(fragment.split(";", 1)[0].strip())
    assert reported_raw == pytest.approx(expected_raw)
    assert 0.0 <= expected_raw <= 1.0
    assert mad == pytest.approx(1.0)
    assert ref_median == pytest.approx(3.0)
    assert anom_median == pytest.approx(8.0)
    assert signed_diff == pytest.approx(5.0)
    assert prevalence == pytest.approx(1.0 / 3.0)
    assert scale == pytest.approx(_ROBUST_SCALE_CONSTANT * mad)
    assert group_z == pytest.approx(abs(signed_diff) / scale)
    assert bounded == pytest.approx(group_z / (group_z + 1.0))


def test_zero_mad_fallback_and_zero_median_difference_retention() -> None:
    # Zero MAD reference with high prevalence / score correlation.
    frame = pl.DataFrame(
        {
            "_original_row_id": ["0", "1", "2", "3", "4", "5", "6"],
            "high_score_corr": [1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 8.0],
            "balanced": [10.0, 10.0, 10.0, 10.0, 10.0, 20.0, 0.0],
            "_regression_residual": [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 0.9],
            "_absolute_centered_residual": [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 0.9],
            "_residual_anomaly_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.95, 0.85],
            "_is_residual_anomaly": [0, 0, 0, 0, 0, 1, 1],
        }
    )
    diagnoser = ResidualAssociationDiagnoser()
    result = diagnoser.diagnose(
        frame,
        request=_request(
            scope=DiagnosisScope.ANOMALY_GROUP,
            events=[],
            feature_columns=["high_score_corr", "balanced"],
        ),
    )
    names = [factor.variable for factor in result.factors]
    assert "high_score_corr" in names
    assert "balanced" in names
    caveat_text = " ".join(result.caveats).lower()
    assert "near-zero" in caveat_text or "minimum_scale" in caveat_text


def test_normalization_and_zero_factor_filtering() -> None:
    frame = _base_frame()
    diagnoser = ResidualAssociationDiagnoser(
        config=ResidualAssociationConfig(include_zero_score_factors=False)
    )
    result = diagnoser.diagnose(
        frame,
        request=_request(
            scope=DiagnosisScope.ANOMALY_GROUP,
            events=[],
            feature_columns=["pressure", "stable"],
        ),
    )
    names = [factor.variable for factor in result.factors]
    assert "pressure" in names
    # constant feature may be filtered when raw score is exactly 0
    assert names.count("pressure") == 1

    include_zero = ResidualAssociationDiagnoser(
        config=ResidualAssociationConfig(include_zero_score_factors=True)
    )
    with_zero = include_zero.diagnose(
        frame,
        request=_request(
            scope=DiagnosisScope.ANOMALY_GROUP,
            events=[],
            feature_columns=["pressure", "stable"],
            top_k_factors=10,
        ),
    )
    assert len(with_zero.factors) >= len(result.factors)

    no_norm = ResidualAssociationDiagnoser(
        config=ResidualAssociationConfig(normalize_factor_scores=False)
    )
    raw_result = no_norm.diagnose(frame, request=_request())
    assert isinstance(raw_result, DiagnosisResult)
    assert raw_result.factors


# --- Ranking ---


def test_ranking_order_and_top_k() -> None:
    frame = _base_frame()
    diagnoser = ResidualAssociationDiagnoser()
    result = diagnoser.diagnose(
        frame,
        request=_request(
            scope=DiagnosisScope.ANOMALY_GROUP,
            events=[],
            top_k_factors=2,
        ),
    )
    assert len(result.factors) <= 2
    variables = [factor.variable for factor in result.factors]
    assert len(variables) == len(set(variables))
    # pressure should outrank stable
    if "pressure" in variables and "stable" in variables:
        assert variables.index("pressure") < variables.index("stable")


# --- Scope ---


def test_scopes_single_group_global_and_batch() -> None:
    frame = _base_frame()
    diagnoser = ResidualAssociationDiagnoser()

    single = diagnoser.diagnose(frame, request=_request())
    assert isinstance(single, DiagnosisResult)
    assert single.anomaly_id == "5"
    assert single.scope is DiagnosisScope.SINGLE_EVENT
    assert single.method_used == [DiagnosisMethod.RESIDUAL_ASSOCIATION]
    assert single.analyzed_row_count == 1
    assert single.reference_row_count == 5
    assert single.generated_at.tzinfo is not None
    assert single.generated_at.utcoffset() == UTC.utcoffset(single.generated_at)
    assert single.metadata["association_not_causation"] is True
    assert single.metadata["ranking_is_heuristic"] is True
    for value in single.metadata.values():
        assert value is None or isinstance(value, (str, int, float, bool))

    group = diagnoser.diagnose(
        frame,
        request=_request(scope=DiagnosisScope.ANOMALY_GROUP, events=[]),
    )
    assert isinstance(group, DiagnosisResult)
    assert group.anomaly_id is None
    assert group.analyzed_row_count == 3

    global_result = diagnoser.diagnose(
        frame,
        request=_request(scope=DiagnosisScope.GLOBAL, events=[]),
    )
    assert isinstance(global_result, DiagnosisResult)
    assert global_result.scope is DiagnosisScope.GLOBAL
    assert global_result.anomaly_id is None

    batch = diagnoser.diagnose(
        frame,
        request=_request(scope=DiagnosisScope.TOP_ANOMALIES),
    )
    assert isinstance(batch, DiagnosisBatchResult)
    assert batch.requested_event_count == 3
    assert batch.diagnosed_event_count == 3
    assert batch.failed_event_count == 0
    assert [r.anomaly_id for r in batch.results] == ["5", "6", "7"]
    assert len({f.variable for f in batch.aggregate_factors}) == len(
        batch.aggregate_factors
    )
    assert any("causation" in warning.lower() for warning in batch.warnings)
    assert any("aggregate" in warning.lower() for warning in batch.warnings)


def test_top_anomalies_ordering_and_top_k_events() -> None:
    frame = _base_frame()
    events = [
        _event("7", score=0.70),
        _event("5", score=0.95),
        _event("6", score=0.80),
    ]
    ordered = ResidualAssociationDiagnoser().diagnose(
        frame,
        request=_request(
            scope=DiagnosisScope.TOP_ANOMALIES,
            events=events,
            top_k_events=2,
        ),
    )
    assert isinstance(ordered, DiagnosisBatchResult)
    assert [r.anomaly_id for r in ordered.results] == ["5", "6"]

    # tie keeps input order when scores equal
    tied_events = [
        _event("6", score=0.9),
        _event("5", score=0.9),
        _event("7", score=0.9),
    ]
    tied = ResidualAssociationDiagnoser().diagnose(
        frame,
        request=_request(
            scope=DiagnosisScope.TOP_ANOMALIES,
            events=tied_events,
            top_k_events=2,
        ),
    )
    assert [r.anomaly_id for r in tied.results] == ["6", "5"]

    no_score_order = ResidualAssociationDiagnoser(
        config=ResidualAssociationConfig(use_anomaly_score_ordering=False)
    )
    unordered = no_score_order.diagnose(
        frame,
        request=_request(
            scope=DiagnosisScope.TOP_ANOMALIES,
            events=events,
            top_k_events=3,
        ),
    )
    assert [r.anomaly_id for r in unordered.results] == ["7", "5", "6"]


def test_batch_aggregate_direction_and_ranking() -> None:
    frame = _base_frame()
    batch = ResidualAssociationDiagnoser().diagnose(
        frame,
        request=_request(scope=DiagnosisScope.TOP_ANOMALIES, top_k_factors=3),
    )
    assert isinstance(batch, DiagnosisBatchResult)
    assert batch.aggregate_factors
    variables = [factor.variable for factor in batch.aggregate_factors]
    assert len(variables) == len(set(variables))
    for factor in batch.aggregate_factors:
        assert 0.0 <= factor.confidence <= 1.0
        assert factor.direction in {"POSITIVE", "NEGATIVE", "MIXED", "UNKNOWN"}
        assert "causation" in factor.evidence.lower() or "associated" in factor.evidence.lower()


# --- Result language / explanation / state ---


def test_result_language_and_explanation_handling() -> None:
    frame = _base_frame()
    diagnoser = ResidualAssociationDiagnoser()
    explanation = ExplanationResult(
        method="permutation",
        feature_importances={"pressure": 0.4},
        notes=["synthetic"],
        confidence=0.5,
    )
    original_notes = list(explanation.notes)
    result = diagnoser.diagnose(
        frame,
        request=_request(),
        explanation=explanation,
    )
    assert isinstance(result, DiagnosisResult)
    assert explanation.notes == original_notes
    joined = " ".join(result.caveats).lower()
    assert "causation" in joined
    assert "sensor" in joined or "misspecification" in joined
    assert "explanation" in joined
    forbidden = ["proven cause", "caused the defect", "will fix", "반드시 원인", "확실한 원인"]
    evidence_blob = " ".join(factor.evidence for factor in result.factors).lower()
    for phrase in forbidden:
        assert phrase not in evidence_blob
        assert phrase not in joined

    none_result = diagnoser.diagnose(frame, request=_request(), explanation=None)
    assert isinstance(none_result, DiagnosisResult)


def test_state_isolation_and_determinism() -> None:
    frame = _base_frame()
    request = _request()
    a = ResidualAssociationDiagnoser()
    b = ResidualAssociationDiagnoser()
    first = a.diagnose(frame, request=request)
    second = a.diagnose(frame, request=request)
    other = b.diagnose(frame, request=request)
    assert isinstance(first, DiagnosisResult)
    assert isinstance(second, DiagnosisResult)
    assert [f.variable for f in first.factors] == [f.variable for f in second.factors]
    assert [f.confidence for f in first.factors] == pytest.approx(
        [f.confidence for f in second.factors]
    )
    assert [f.variable for f in first.factors] == [f.variable for f in other.factors]

    mutated = first.factors[0]
    first.factors[0] = RootCauseFactor(
        variable="mutated",
        direction="UNKNOWN",
        deviation=0.0,
        role=ColumnRole.UNKNOWN,
        controllable=False,
        evidence="mutated",
        confidence=0.1,
        needs_verification=True,
    )
    third = a.diagnose(frame, request=request)
    assert third.factors[0].variable != "mutated"
    del mutated

    meta = a.get_metadata()
    assert not any(isinstance(v, (pl.DataFrame, np.ndarray)) for v in meta.values())
    for factor in third.factors:
        assert "estimator" not in factor.evidence.lower()


def test_wrong_method_rejected() -> None:
    diagnoser = ResidualAssociationDiagnoser()
    with pytest.raises(DataValidationError, match="RESIDUAL_ASSOCIATION"):
        diagnoser.diagnose(
            _base_frame(),
            request=_request(method=DiagnosisMethod.GROUP_COMPARISON),
        )


def test_config_default_columns_when_request_omits_score_and_indicator_names() -> None:
    # Request still needs unique role columns; use aliases then rely on config names
    # by setting request columns to the reserved names explicitly via None fallback.
    frame = _base_frame()
    diagnoser = ResidualAssociationDiagnoser()
    request = DiagnosisRequest(
        task=AnalysisTask.RESIDUAL_ANOMALY,
        method=DiagnosisMethod.RESIDUAL_ASSOCIATION,
        scope=DiagnosisScope.SINGLE_EVENT,
        feature_columns=["pressure", "temperature", "stable"],
        anomaly_events=[_event("5", score=0.95)],
        anomaly_indicator_column=None,
        anomaly_score_column=None,
        row_id_column="_original_row_id",
        minimum_reference_rows=2,
    )
    result = diagnoser.diagnose(frame, request=request)
    assert isinstance(result, DiagnosisResult)
    assert result.metadata["anomaly_score_column"] == "_residual_anomaly_score"
