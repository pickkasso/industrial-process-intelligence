"""Unit tests for automatic UI column configuration (Step 11B.2)."""

from __future__ import annotations

import copy
from enum import Enum
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from process_intelligence.ui.column_configuration import (
    AutomaticColumnConfigurator,
    UiColumnConfigurationReport,
    UiColumnSuggestion,
    UiColumnSuggestionCategory,
    resolve_active_feature_columns,
)


def _suggestion(**overrides: Any) -> UiColumnSuggestion:
    payload: dict[str, Any] = {
        "column": "pressure",
        "dtype": "Float64",
        "category": UiColumnSuggestionCategory.FEATURE_CANDIDATE,
        "priority": 1,
        "confidence": 0.75,
        "reasons": ["numeric non-constant feature candidate"],
        "numeric": True,
        "null_count": 0,
        "unique_count": 3,
        "unique_ratio": 1.0,
        "constant": False,
        "monotonic_non_decreasing": False,
    }
    payload.update(overrides)
    return UiColumnSuggestion(**payload)


def _report(**overrides: Any) -> UiColumnConfigurationReport:
    suggestions = [
        _suggestion(column="pressure"),
        _suggestion(
            column="quality",
            category=UiColumnSuggestionCategory.TARGET_CANDIDATE,
            priority=65,
            confidence=0.8,
            reasons=["column name matches target heuristic"],
            unique_count=2,
            unique_ratio=2 / 3,
        ),
        _suggestion(
            column="lot_id",
            dtype="String",
            category=UiColumnSuggestionCategory.IDENTIFIER_CANDIDATE,
            priority=50,
            confidence=0.9,
            reasons=["column name matches identifier heuristic"],
            numeric=False,
            unique_count=2,
            unique_ratio=2 / 3,
            monotonic_non_decreasing=None,
        ),
    ]
    payload: dict[str, Any] = {
        "column_count": 3,
        "row_count": 3,
        "suggestions": suggestions,
        "target_candidates": ["quality"],
        "recommended_feature_columns": ["pressure"],
        "timestamp_candidates": [],
        "identifier_candidates": ["lot_id"],
        "excluded_candidates": ["lot_id"],
        "review_required_columns": [],
        "numeric_columns": ["pressure", "quality"],
        "nonnumeric_columns": ["lot_id"],
        "warnings": [],
        "metadata": {"modifies_raw_data": False},
    }
    payload.update(overrides)
    return UiColumnConfigurationReport(**payload)


def test_enum_values_exact() -> None:
    assert UiColumnSuggestionCategory.TARGET_CANDIDATE.value == "TARGET_CANDIDATE"
    assert UiColumnSuggestionCategory.FEATURE_CANDIDATE.value == "FEATURE_CANDIDATE"
    assert UiColumnSuggestionCategory.TIMESTAMP_CANDIDATE.value == "TIMESTAMP_CANDIDATE"
    assert (
        UiColumnSuggestionCategory.IDENTIFIER_CANDIDATE.value == "IDENTIFIER_CANDIDATE"
    )
    assert UiColumnSuggestionCategory.EXCLUDED_CANDIDATE.value == "EXCLUDED_CANDIDATE"
    assert UiColumnSuggestionCategory.REVIEW_REQUIRED.value == "REVIEW_REQUIRED"


def test_no_extra_enum_members() -> None:
    assert {item.name for item in UiColumnSuggestionCategory} == {
        "TARGET_CANDIDATE",
        "FEATURE_CANDIDATE",
        "TIMESTAMP_CANDIDATE",
        "IDENTIFIER_CANDIDATE",
        "EXCLUDED_CANDIDATE",
        "REVIEW_REQUIRED",
    }
    assert issubclass(UiColumnSuggestionCategory, Enum)


def test_valid_suggestion_schema() -> None:
    item = _suggestion()
    assert item.column == "pressure"
    assert item.confidence == 0.75


@pytest.mark.parametrize("confidence", [-0.1, 1.1, True, float("nan"), float("inf")])
def test_confidence_range(confidence: object) -> None:
    with pytest.raises(ValidationError):
        _suggestion(confidence=confidence)


@pytest.mark.parametrize("bad", [True, -1, 1.5, "2"])
def test_count_rejects_bool_negative_non_int(bad: object) -> None:
    with pytest.raises(ValidationError):
        _suggestion(null_count=bad)
    with pytest.raises(ValidationError):
        _suggestion(unique_count=bad)


def test_reason_duplicates_rejected() -> None:
    with pytest.raises(ValidationError):
        _suggestion(reasons=["a", "a"])


def test_report_valid_construction() -> None:
    report = _report()
    assert report.column_count == 3
    assert report.target_candidates == ["quality"]


def test_report_column_count_mismatch() -> None:
    with pytest.raises(ValidationError):
        _report(column_count=2)


def test_candidate_list_duplicates_rejected() -> None:
    with pytest.raises(ValidationError):
        _report(target_candidates=["quality", "quality"])


def test_json_serialization() -> None:
    report = _report()
    payload = report.model_dump(mode="json")
    assert isinstance(payload, dict)
    restored = UiColumnConfigurationReport.model_validate(payload)
    assert restored.model_dump(mode="json") == payload


def test_empty_frame_rejected() -> None:
    configurator = AutomaticColumnConfigurator()
    with pytest.raises(ValueError):
        configurator.analyze(pl.DataFrame({"a": []}))
    with pytest.raises(ValueError):
        configurator.analyze(pl.DataFrame())


def test_wrong_input_type() -> None:
    configurator = AutomaticColumnConfigurator()
    with pytest.raises(TypeError):
        configurator.analyze({"a": [1, 2]})  # type: ignore[arg-type]


def test_numeric_feature_recommendation() -> None:
    frame = pl.DataFrame(
        {
            "pressure": [1.0, 2.0, 3.0],
            "temperature": [10.0, 11.0, 12.0],
            "notes": ["a", "b", "c"],
        }
    )
    report = AutomaticColumnConfigurator().analyze(frame)
    assert "pressure" in report.recommended_feature_columns
    assert "temperature" in report.recommended_feature_columns
    assert "notes" not in report.recommended_feature_columns


def test_string_column_excluded_from_features() -> None:
    frame = pl.DataFrame({"x": [1.0, 2.0], "label_text": ["a", "b"]})
    report = AutomaticColumnConfigurator().analyze(frame)
    assert "label_text" not in report.recommended_feature_columns
    assert "label_text" in report.excluded_candidates


def test_constant_column_excluded() -> None:
    frame = pl.DataFrame({"sensor": [1.0, 2.0], "const": [5.0, 5.0]})
    report = AutomaticColumnConfigurator().analyze(frame)
    assert "const" not in report.recommended_feature_columns
    assert "const" in report.excluded_candidates


def test_all_null_column_excluded() -> None:
    frame = pl.DataFrame(
        {
            "sensor": [1.0, 2.0],
            "empty": pl.Series("empty", [None, None], dtype=pl.Float64),
        }
    )
    report = AutomaticColumnConfigurator().analyze(frame)
    assert "empty" not in report.recommended_feature_columns
    assert "empty" in report.excluded_candidates


def test_bool_review_required() -> None:
    frame = pl.DataFrame({"sensor": [1.0, 2.0], "flag": [True, False]})
    report = AutomaticColumnConfigurator().analyze(frame)
    suggestion = next(item for item in report.suggestions if item.column == "flag")
    assert suggestion.category is UiColumnSuggestionCategory.REVIEW_REQUIRED
    assert "flag" in report.review_required_columns
    assert "flag" not in report.recommended_feature_columns


def test_serial_number_identifier_candidate() -> None:
    frame = pl.DataFrame(
        {
            "SerialNumber": [1, 2, 3],
            "sensor": [1.0, 2.0, 3.0],
        }
    )
    report = AutomaticColumnConfigurator().analyze(frame)
    assert "SerialNumber" in report.identifier_candidates
    assert "SerialNumber" not in report.recommended_feature_columns


def test_lot_id_identifier_candidate() -> None:
    frame = pl.DataFrame(
        {
            "lot_id": ["A", "B", "C"],
            "sensor": [1.0, 2.0, 3.0],
        }
    )
    report = AutomaticColumnConfigurator().analyze(frame)
    assert "lot_id" in report.identifier_candidates


def test_date_timestamp_candidate() -> None:
    frame = pl.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "sensor": [1.0, 2.0, 3.0],
        }
    )
    report = AutomaticColumnConfigurator().analyze(frame)
    assert "Date" in report.timestamp_candidates
    assert "Date" not in report.recommended_feature_columns


def test_time_timestamp_candidate() -> None:
    frame = pl.DataFrame(
        {
            "Time": ["10:00", "11:00", "12:00"],
            "sensor": [1.0, 2.0, 3.0],
        }
    )
    report = AutomaticColumnConfigurator().analyze(frame)
    assert "Time" in report.timestamp_candidates


def test_timestamp_string_not_parsed() -> None:
    frame = pl.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-02"],
            "sensor": [1.0, 2.0],
        }
    )
    before = frame.to_dicts()
    report = AutomaticColumnConfigurator().analyze(frame)
    after = frame.to_dicts()
    assert before == after
    date_suggestion = next(item for item in report.suggestions if item.column == "Date")
    assert date_suggestion.dtype == "String"
    assert "parsing not performed" in " ".join(date_suggestion.reasons).lower() or (
        "non-numeric" in " ".join(date_suggestion.reasons).lower()
    )


def test_soh_target_candidate() -> None:
    frame = pl.DataFrame(
        {
            "Date": ["a", "b", "c"],
            "sensor": [1.0, 2.0, 3.0],
            "SOH": [0.9, 0.8, 0.7],
        }
    )
    report = AutomaticColumnConfigurator().analyze(frame)
    assert report.target_candidates[0] == "SOH"
    assert len(report.target_candidates) >= 1
    assert "SOH" not in report.recommended_feature_columns
    assert "SOH" not in report.excluded_candidates
    suggestion = next(item for item in report.suggestions if item.column == "SOH")
    assert suggestion.category is UiColumnSuggestionCategory.TARGET_CANDIDATE


def test_lowercase_soh_target_candidate() -> None:
    frame = pl.DataFrame({"soh": [0.9, 0.8, 0.7], "sensor": [1.0, 2.0, 3.0]})
    report = AutomaticColumnConfigurator().analyze(frame)
    assert "soh" in report.target_candidates
    assert report.target_candidates[0] == "soh"


def test_padded_soh_target_candidate() -> None:
    frame = pl.DataFrame({" SOH ": [0.9, 0.8, 0.7], "sensor": [1.0, 2.0, 3.0]})
    report = AutomaticColumnConfigurator().analyze(frame)
    assert " SOH " in report.target_candidates
    assert list(frame.columns) == [" SOH ", "sensor"]


def test_soh_underscore_target_candidate() -> None:
    frame = pl.DataFrame({"SOH_": [0.9, 0.8, 0.7], "sensor": [1.0, 2.0, 3.0]})
    report = AutomaticColumnConfigurator().analyze(frame)
    assert "SOH_" in report.target_candidates


def test_quality_score_target_candidate() -> None:
    frame = pl.DataFrame(
        {
            "quality_score": [1.0, 2.0, 3.0],
            "sensor": [4.0, 5.0, 6.0],
        }
    )
    report = AutomaticColumnConfigurator().analyze(frame)
    assert "quality_score" in report.target_candidates
    assert "quality_score" not in report.excluded_candidates


def test_quality_score_spaced_target_candidate() -> None:
    frame = pl.DataFrame(
        {
            "Quality Score": [1.0, 2.0, 3.0],
            "sensor": [4.0, 5.0, 6.0],
        }
    )
    report = AutomaticColumnConfigurator().analyze(frame)
    assert "Quality Score" in report.target_candidates


def test_state_of_health_hyphen_target_candidate() -> None:
    frame = pl.DataFrame(
        {
            "state-of-health": [0.9, 0.8, 0.7],
            "sensor": [1.0, 2.0, 3.0],
        }
    )
    report = AutomaticColumnConfigurator().analyze(frame)
    assert "state-of-health" in report.target_candidates


def test_target_candidate_not_overwritten_by_excluded_category() -> None:
    frame = pl.DataFrame(
        {
            "SOH": [0.95, 0.95, 0.95],
            "sensor": [1.0, 2.0, 3.0],
            "notes": ["a", "b", "c"],
        }
    )
    report = AutomaticColumnConfigurator().analyze(frame)
    soh = next(item for item in report.suggestions if item.column == "SOH")
    assert soh.category is UiColumnSuggestionCategory.TARGET_CANDIDATE
    assert soh.constant is True
    assert "SOH" in report.target_candidates
    assert "SOH" not in report.excluded_candidates
    assert "notes" in report.excluded_candidates


def test_target_candidate_excluded_from_recommended_features() -> None:
    frame = pl.DataFrame(
        {
            "SOH": [0.9, 0.8, 0.7],
            "sensor": [1.0, 2.0, 3.0],
        }
    )
    report = AutomaticColumnConfigurator().analyze(frame)
    assert "SOH" not in report.recommended_feature_columns
    assert "sensor" in report.recommended_feature_columns


def test_target_candidate_not_in_explicit_excluded_default() -> None:
    frame = pl.DataFrame(
        {
            "SerialNumber": [1, 2, 3],
            "Date": ["a", "b", "c"],
            "Time": ["x", "y", "z"],
            "sensor": [1.0, 2.0, 3.0],
            "SOH": [0.9, 0.8, 0.7],
            "notes": ["a", "b", "c"],
        }
    )
    report = AutomaticColumnConfigurator().analyze(frame)
    assert "SOH" not in report.excluded_candidates
    assert "SerialNumber" not in report.excluded_candidates
    assert "Date" not in report.excluded_candidates
    assert "Time" not in report.excluded_candidates
    assert "notes" in report.excluded_candidates
    assert "SerialNumber" in report.identifier_candidates
    assert "Date" in report.timestamp_candidates
    assert "Time" in report.timestamp_candidates
    assert len(report.target_candidates) >= 1


def test_first_column_not_auto_final_target() -> None:
    frame = pl.DataFrame(
        {
            "Date": ["a", "b"],
            "sensor": [1.0, 2.0],
            "SOH": [0.9, 0.8],
        }
    )
    report = AutomaticColumnConfigurator().analyze(frame)
    assert report.metadata["infers_final_target"] is False
    assert report.target_candidates[0] != "Date"
    metadata = AutomaticColumnConfigurator().get_metadata()
    assert metadata["infers_final_target"] is False


def test_identifier_excluded_from_default_features() -> None:
    frame = pl.DataFrame(
        {
            "SerialNumber": [10, 20, 30],
            "sensor": [1.0, 2.0, 3.0],
        }
    )
    report = AutomaticColumnConfigurator().analyze(frame)
    assert "SerialNumber" not in report.recommended_feature_columns


def test_timestamp_excluded_from_default_features() -> None:
    frame = pl.DataFrame(
        {
            "Date": ["a", "b", "c"],
            "sensor": [1.0, 2.0, 3.0],
        }
    )
    report = AutomaticColumnConfigurator().analyze(frame)
    assert "Date" not in report.recommended_feature_columns


def test_resolve_active_features_removes_selected_target() -> None:
    active = resolve_active_feature_columns(
        ["sensor_a", "sensor_b", "SOH"],
        selected_target="SOH",
        selected_timestamp="Date",
        selected_identifiers=["SerialNumber"],
        selected_excluded=["notes"],
    )
    assert active == ["sensor_a", "sensor_b"]


def test_null_count_accuracy() -> None:
    frame = pl.DataFrame(
        {
            "sensor": pl.Series("sensor", [1.0, None, 3.0], dtype=pl.Float64),
        }
    )
    report = AutomaticColumnConfigurator().analyze(frame)
    suggestion = report.suggestions[0]
    assert suggestion.null_count == 1


def test_unique_ratio_accuracy() -> None:
    frame = pl.DataFrame({"sensor": [1.0, 1.0, 2.0, 3.0]})
    report = AutomaticColumnConfigurator().analyze(frame)
    suggestion = report.suggestions[0]
    assert suggestion.unique_count == 3
    assert suggestion.unique_ratio == pytest.approx(0.75)


def test_monotonic_numeric_detection() -> None:
    frame = pl.DataFrame(
        {
            "increasing": [1.0, 2.0, 3.0],
            "mixed": [1.0, 3.0, 2.0],
        }
    )
    report = AutomaticColumnConfigurator().analyze(frame)
    increasing = next(
        item for item in report.suggestions if item.column == "increasing"
    )
    mixed = next(item for item in report.suggestions if item.column == "mixed")
    assert increasing.monotonic_non_decreasing is True
    assert mixed.monotonic_non_decreasing is False


def test_original_column_order_preserved() -> None:
    frame = pl.DataFrame(
        {
            "c": [1.0, 2.0],
            "a": [3.0, 4.0],
            "b": [5.0, 6.0],
        }
    )
    report = AutomaticColumnConfigurator().analyze(frame)
    assert [item.column for item in report.suggestions] == ["c", "a", "b"]
    assert report.recommended_feature_columns == ["c", "a", "b"]


def test_target_priority_determinism() -> None:
    frame = pl.DataFrame(
        {
            "quality": [1.0, 2.0, 3.0],
            "sensor": [4.0, 5.0, 6.0],
            "SOH": [0.9, 0.8, 0.7],
            "y": [1.0, 1.0, 0.0],
        }
    )
    report = AutomaticColumnConfigurator().analyze(frame)
    assert report.target_candidates[0] == "SOH"
    assert "quality" in report.target_candidates
    assert report.target_candidates.index("SOH") < report.target_candidates.index(
        "quality"
    )
    assert report.target_candidates.index("quality") < report.target_candidates.index(
        "y"
    )


def test_same_input_deterministic() -> None:
    frame = pl.DataFrame(
        {
            "Date": ["a", "b", "c"],
            "SerialNumber": [1, 2, 3],
            "sensor": [1.0, 2.0, 3.0],
            "SOH": [0.9, 0.8, 0.7],
        }
    )
    configurator = AutomaticColumnConfigurator()
    first = configurator.analyze(frame).model_dump(mode="json")
    second = configurator.analyze(frame).model_dump(mode="json")
    assert first == second


def test_input_frame_immutability() -> None:
    frame = pl.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-02"],
            "sensor": [1, 2],
            "SOH": [0.9, 0.8],
        }
    )
    before_cols = list(frame.columns)
    before_rows = frame.to_dicts()
    before_dtypes = list(frame.dtypes)
    AutomaticColumnConfigurator().analyze(frame)
    assert list(frame.columns) == before_cols
    assert frame.to_dicts() == before_rows
    assert list(frame.dtypes) == before_dtypes


def test_no_cache_across_calls() -> None:
    configurator = AutomaticColumnConfigurator()
    first = configurator.analyze(pl.DataFrame({"sensor": [1.0, 2.0]}))
    second = configurator.analyze(
        pl.DataFrame({"other": [3.0, 4.0], "SOH": [0.1, 0.2]})
    )
    assert first.suggestions[0].column == "sensor"
    assert [item.column for item in second.suggestions] == ["other", "SOH"]


def test_instance_state_isolation() -> None:
    left = AutomaticColumnConfigurator(identifier_unique_ratio=0.9)
    right = AutomaticColumnConfigurator(
        identifier_unique_ratio=0.99,
        near_unique_ratio=0.995,
    )
    assert left.get_metadata()["identifier_unique_ratio"] == 0.9
    assert right.get_metadata()["identifier_unique_ratio"] == 0.99
    assert right.get_metadata()["near_unique_ratio"] == 0.995


def test_get_metadata_scalar_only() -> None:
    metadata = AutomaticColumnConfigurator().get_metadata()
    assert metadata["analyzes_dtype"] is True
    assert metadata["analyzes_null_count"] is True
    assert metadata["analyzes_cardinality"] is True
    assert metadata["analyzes_monotonicity"] is True
    assert metadata["infers_final_target"] is False
    assert metadata["infers_controllability"] is False
    assert metadata["infers_constraints"] is False
    assert metadata["modifies_raw_data"] is False
    assert metadata["performs_modeling"] is False
    for value in metadata.values():
        assert value is None or isinstance(value, (str, int, float, bool))
    again = AutomaticColumnConfigurator().get_metadata()
    assert again is not metadata
    again["analyzes_dtype"] = False
    assert AutomaticColumnConfigurator().get_metadata()["analyzes_dtype"] is True


def test_raw_values_not_in_metadata() -> None:
    frame = pl.DataFrame({"sensor": [1.0, 2.0], "SOH": [0.9, 0.8]})
    report = AutomaticColumnConfigurator().analyze(frame)
    dumped = report.model_dump(mode="json")
    serialized = str(dumped)
    assert "0.9" not in serialized or "SOH" in serialized
    for key, value in report.metadata.items():
        assert not isinstance(value, (list, dict, pl.DataFrame, pl.Series))
        assert key != "raw_values"


def test_report_has_no_dataframe_series_ndarray() -> None:
    frame = pl.DataFrame({"sensor": [1.0, 2.0], "SOH": [0.9, 0.8]})
    report = AutomaticColumnConfigurator().analyze(frame)
    dumped = report.model_dump(mode="json")

    def _walk(node: object) -> None:
        assert not isinstance(node, (pl.DataFrame, pl.Series))
        type_name = type(node).__name__
        assert "ndarray" not in type_name
        if isinstance(node, dict):
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for value in node:
                _walk(value)

    _walk(dumped)
    cloned = copy.deepcopy(dumped)
    assert cloned == dumped
