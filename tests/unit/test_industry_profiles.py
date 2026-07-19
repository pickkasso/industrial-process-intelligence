"""Unit tests for keyword-based industry profiles (Step 3B)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import polars as pl
import pytest

from process_intelligence.core.enums import AnalysisTask, ColumnRole
from process_intelligence.core.protocols import BaseIndustryProfile
from process_intelligence.core.schemas import DatasetMetadata, PreprocessingRules, SchemaHints
from process_intelligence.industries import (
    AutomotiveIndustryProfile,
    BatteryIndustryProfile,
    GenericIndustryProfile,
    IndustryRegistry,
    SemiconductorIndustryProfile,
    create_default_registry,
)

PROFILES = (
    SemiconductorIndustryProfile,
    BatteryIndustryProfile,
    AutomotiveIndustryProfile,
)


def _metadata(**overrides: Any) -> DatasetMetadata:
    payload: dict[str, Any] = {
        "file_name": "sample.csv",
        "file_format": "csv",
        "row_count": 3,
        "column_names": ["temp", "pressure"],
        "dtypes": {"temp": "Float64", "pressure": "Float64"},
        "user_description": None,
    }
    payload.update(overrides)
    return DatasetMetadata(**payload)


def _evidence_sources(evidence: list[str]) -> list[str]:
    sources: list[str] = []
    for item in evidence:
        for source in ("user_description", "column_names", "file_name"):
            if f" in {source}." in item:
                sources.append(source)
                break
    return sources


def _evidence_keywords(evidence: list[str]) -> list[str]:
    keywords: list[str] = []
    for item in evidence:
        start = item.find("'")
        end = item.find("'", start + 1)
        if start != -1 and end != -1:
            keywords.append(item[start + 1 : end])
    return keywords


@pytest.mark.parametrize("profile_cls", PROFILES)
def test_profile_instantiate(profile_cls: type[BaseIndustryProfile]) -> None:
    assert isinstance(profile_cls(), profile_cls)


@pytest.mark.parametrize("profile_cls", PROFILES)
def test_profile_is_base_industry_profile(
    profile_cls: type[BaseIndustryProfile],
) -> None:
    assert isinstance(profile_cls(), BaseIndustryProfile)


def test_industry_names() -> None:
    assert SemiconductorIndustryProfile().industry_name == "semiconductor"
    assert BatteryIndustryProfile().industry_name == "battery"
    assert AutomotiveIndustryProfile().industry_name == "automotive"


@pytest.mark.parametrize("profile_cls", PROFILES)
def test_keywords_have_no_duplicates(profile_cls: type[Any]) -> None:
    keywords = profile_cls.keywords
    assert len(keywords) == len(set(keywords))


@pytest.mark.parametrize("profile_cls", PROFILES)
def test_keywords_have_no_empty_strings(profile_cls: type[Any]) -> None:
    assert all(isinstance(item, str) and item.strip() for item in profile_cls.keywords)


@pytest.mark.parametrize("profile_cls", PROFILES)
def test_score_industry_rejects_invalid_metadata(
    profile_cls: type[BaseIndustryProfile],
) -> None:
    with pytest.raises(TypeError, match="DatasetMetadata"):
        profile_cls().score_industry("not-metadata")  # type: ignore[arg-type]


@pytest.mark.parametrize("profile_cls", PROFILES)
def test_get_default_model_candidates_rejects_invalid_task(
    profile_cls: type[BaseIndustryProfile],
) -> None:
    with pytest.raises(TypeError, match="AnalysisTask"):
        profile_cls().get_default_model_candidates("REGRESSION")  # type: ignore[arg-type]


@pytest.mark.parametrize("profile_cls", PROFILES)
def test_validate_physical_ranges_accepts_polars(
    profile_cls: type[BaseIndustryProfile],
) -> None:
    frame = pl.DataFrame({"temp": [1.0, 2.0]})
    assert profile_cls().validate_physical_ranges(frame) == []


@pytest.mark.parametrize("profile_cls", PROFILES)
def test_validate_physical_ranges_accepts_pandas(
    profile_cls: type[BaseIndustryProfile],
) -> None:
    frame = pd.DataFrame({"temp": [1.0, 2.0]})
    assert profile_cls().validate_physical_ranges(frame) == []


@pytest.mark.parametrize("profile_cls", PROFILES)
def test_validate_physical_ranges_rejects_non_dataframe(
    profile_cls: type[BaseIndustryProfile],
) -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        profile_cls().validate_physical_ranges([1, 2, 3])  # type: ignore[arg-type]


@pytest.mark.parametrize("profile_cls", PROFILES)
def test_validate_physical_ranges_does_not_mutate_frame(
    profile_cls: type[BaseIndustryProfile],
) -> None:
    frame = pl.DataFrame({"temp": [1.0, 2.0]})
    before = frame.clone()
    profile_cls().validate_physical_ranges(frame)
    assert frame.equals(before)


@pytest.mark.parametrize("profile_cls", PROFILES)
def test_get_preprocessing_rules_returns_independent_empty_instances(
    profile_cls: type[BaseIndustryProfile],
) -> None:
    profile = profile_cls()
    first = profile.get_preprocessing_rules()
    second = profile.get_preprocessing_rules()
    assert isinstance(first, PreprocessingRules)
    assert first is not second
    first.imputation["temp"] = "median"
    assert second.imputation == {}


@pytest.mark.parametrize("profile_cls", PROFILES)
def test_get_default_model_candidates_returns_independent_empty_lists(
    profile_cls: type[BaseIndustryProfile],
) -> None:
    profile = profile_cls()
    first = profile.get_default_model_candidates(AnalysisTask.REGRESSION)
    second = profile.get_default_model_candidates(AnalysisTask.REGRESSION)
    assert first == []
    assert first is not second
    first.append("mutated")  # type: ignore[arg-type]
    assert second == []


@pytest.mark.parametrize("profile_cls", PROFILES)
def test_get_recommendation_constraints_returns_independent_empty_lists(
    profile_cls: type[BaseIndustryProfile],
) -> None:
    profile = profile_cls()
    first = profile.get_recommendation_constraints()
    second = profile.get_recommendation_constraints()
    assert first == []
    assert first is not second
    first.append("mutated")  # type: ignore[arg-type]
    assert second == []


def test_no_keyword_score_is_zero() -> None:
    score = SemiconductorIndustryProfile().score_industry(_metadata())
    assert score.score == pytest.approx(0.0)


def test_no_keyword_confidence_is_zero() -> None:
    score = SemiconductorIndustryProfile().score_industry(_metadata())
    assert score.confidence == pytest.approx(0.0)


def test_no_keyword_requires_user_confirmation() -> None:
    score = SemiconductorIndustryProfile().score_industry(_metadata())
    assert score.requires_user_confirmation is True


def test_no_keyword_uncertain_factors_not_empty() -> None:
    score = SemiconductorIndustryProfile().score_industry(_metadata())
    assert len(score.uncertain_factors) >= 1


def test_file_name_keyword_detection() -> None:
    score = SemiconductorIndustryProfile().score_industry(
        _metadata(file_name="wafer_run.csv")
    )
    assert score.score == pytest.approx(1.0)
    assert any("file_name" in item and "wafer" in item for item in score.evidence)


def test_user_description_keyword_detection() -> None:
    score = SemiconductorIndustryProfile().score_industry(
        _metadata(user_description="Contains semiconductor process metrics")
    )
    assert score.score == pytest.approx(4.0)
    assert any("user_description" in item for item in score.evidence)


def test_column_names_keyword_detection() -> None:
    score = SemiconductorIndustryProfile().score_industry(
        _metadata(column_names=["temp", "lot_id"], dtypes={"temp": "Float64", "lot_id": "Utf8"})
    )
    assert score.score == pytest.approx(2.0)
    assert any("column_names" in item and "lot_id" in item for item in score.evidence)


def test_user_description_weight_is_four() -> None:
    score = BatteryIndustryProfile().score_industry(
        _metadata(user_description="battery dataset")
    )
    assert score.score == pytest.approx(4.0)


def test_column_names_weight_is_two() -> None:
    score = BatteryIndustryProfile().score_industry(
        _metadata(
            column_names=["cell_id"],
            dtypes={"cell_id": "Utf8"},
        )
    )
    assert score.score == pytest.approx(2.0)


def test_file_name_weight_is_one() -> None:
    score = AutomotiveIndustryProfile().score_industry(
        _metadata(file_name="automotive_log.csv")
    )
    assert score.score == pytest.approx(1.0)


def test_duplicate_column_keyword_scored_once() -> None:
    score = SemiconductorIndustryProfile().score_industry(
        _metadata(
            column_names=["wafer_a", "wafer_b", "wafer_c"],
            dtypes={"wafer_a": "Utf8", "wafer_b": "Utf8", "wafer_c": "Utf8"},
        )
    )
    assert score.score == pytest.approx(2.0)


def test_same_keyword_across_sources_adds_per_source() -> None:
    score = SemiconductorIndustryProfile().score_industry(
        _metadata(
            file_name="wafer.csv",
            user_description="wafer process",
            column_names=["wafer"],
            dtypes={"wafer": "Utf8"},
        )
    )
    assert score.score == pytest.approx(7.0)


def test_case_insensitive_matching() -> None:
    score = SemiconductorIndustryProfile().score_industry(
        _metadata(file_name="WAFER_RUN.CSV")
    )
    assert score.score == pytest.approx(1.0)


def test_underscore_and_hyphen_equivalence() -> None:
    profile = SemiconductorIndustryProfile()
    underscore = profile.score_industry(
        _metadata(
            column_names=["wafer_id"],
            dtypes={"wafer_id": "Utf8"},
        )
    )
    hyphen = profile.score_industry(
        _metadata(
            column_names=["Wafer-ID"],
            dtypes={"Wafer-ID": "Utf8"},
        )
    )
    spaced = profile.score_industry(
        _metadata(
            column_names=["wafer id"],
            dtypes={"wafer id": "Utf8"},
        )
    )
    assert underscore.score == pytest.approx(hyphen.score)
    assert hyphen.score == pytest.approx(spaced.score)
    assert underscore.score > 0.0


def test_substring_inside_unrelated_word_does_not_match() -> None:
    # "etch" must not match inside "sketch"; "soc" must not match inside "social".
    semi = SemiconductorIndustryProfile().score_industry(
        _metadata(
            file_name="sketch.csv",
            user_description="sketch quality samples",
            column_names=["sketch_score"],
            dtypes={"sketch_score": "Float64"},
        )
    )
    battery = BatteryIndustryProfile().score_industry(
        _metadata(
            file_name="social.csv",
            user_description="social metrics only",
            column_names=["social_score"],
            dtypes={"social_score": "Float64"},
        )
    )
    assert semi.score == pytest.approx(0.0)
    assert battery.score == pytest.approx(0.0)


def test_english_keyword_detection() -> None:
    score = AutomotiveIndustryProfile().score_industry(
        _metadata(user_description="engine_rpm telemetry")
    )
    assert score.score == pytest.approx(4.0)


def test_korean_keyword_detection() -> None:
    score = SemiconductorIndustryProfile().score_industry(
        _metadata(user_description="반도체 공정 데이터")
    )
    assert score.score == pytest.approx(4.0)


def test_confidence_formula() -> None:
    score = SemiconductorIndustryProfile().score_industry(
        _metadata(file_name="wafer.csv")
    )
    assert score.confidence == pytest.approx(round(min(1.0, score.score / 12.0), 4))


def test_confidence_never_exceeds_one() -> None:
    score = SemiconductorIndustryProfile().score_industry(
        _metadata(
            file_name="wafer_lot_die_etch_deposition_fab.csv",
            user_description=(
                "semiconductor wafer lithography photoresist overlay "
                "critical_dimension film_thickness sheet_resistance 반도체 웨이퍼"
            ),
            column_names=["wafer_id", "lot_id", "die_id", "chamber_id", "etch"],
            dtypes={
                "wafer_id": "Utf8",
                "lot_id": "Utf8",
                "die_id": "Utf8",
                "chamber_id": "Utf8",
                "etch": "Float64",
            },
        )
    )
    assert score.confidence <= 1.0
    assert score.confidence == pytest.approx(1.0)


def test_confidence_below_threshold_requires_confirmation() -> None:
    score = SemiconductorIndustryProfile().score_industry(
        _metadata(file_name="wafer.csv")
    )
    assert score.confidence < 0.60
    assert score.requires_user_confirmation is True


def test_confidence_at_or_above_threshold_no_confirmation() -> None:
    score = SemiconductorIndustryProfile().score_industry(
        _metadata(
            user_description="semiconductor wafer etch",
            column_names=["lot_id"],
            dtypes={"lot_id": "Utf8"},
        )
    )
    # 3 description keywords * 4 + 1 column * 2 = 14 -> confidence 1.0
    assert score.confidence >= 0.60
    assert score.requires_user_confirmation is False


def test_evidence_follows_source_order() -> None:
    score = SemiconductorIndustryProfile().score_industry(
        _metadata(
            file_name="fab.csv",
            user_description="wafer process",
            column_names=["etch"],
            dtypes={"etch": "Float64"},
        )
    )
    sources = _evidence_sources(score.evidence)
    assert sources == sorted(sources, key=["user_description", "column_names", "file_name"].index)


def test_evidence_follows_keyword_declaration_order() -> None:
    profile = SemiconductorIndustryProfile()
    score = profile.score_industry(
        _metadata(
            user_description="etch wafer deposition",
        )
    )
    keywords = _evidence_keywords(
        [item for item in score.evidence if "user_description" in item]
    )
    declared = [word for word in profile.keywords if word in keywords]
    assert keywords == declared


def test_evidence_has_no_duplicates() -> None:
    score = SemiconductorIndustryProfile().score_industry(
        _metadata(
            file_name="wafer.csv",
            user_description="wafer wafer wafer",
            column_names=["wafer", "wafer"],
            dtypes={"wafer": "Utf8"},
        )
    )
    assert len(score.evidence) == len(set(score.evidence))


def test_score_industry_is_deterministic() -> None:
    profile = SemiconductorIndustryProfile()
    metadata = _metadata(
        file_name="wafer.csv",
        user_description="semiconductor fab",
        column_names=["lot_id"],
        dtypes={"lot_id": "Utf8"},
    )
    first = profile.score_industry(metadata)
    second = profile.score_industry(metadata)
    assert first.model_dump() == second.model_dump()


def test_score_industry_does_not_mutate_metadata() -> None:
    metadata = _metadata(
        column_names=["wafer_id"],
        dtypes={"wafer_id": "Utf8"},
        user_description="semiconductor",
    )
    original = metadata.model_dump()
    SemiconductorIndustryProfile().score_industry(metadata)
    assert metadata.model_dump() == original


def test_wafer_metadata_prefers_semiconductor() -> None:
    metadata = _metadata(
        file_name="wafer_lot.csv",
        user_description="semiconductor fab etch deposition",
        column_names=["wafer_id", "lot_id", "chamber_id"],
        dtypes={
            "wafer_id": "Utf8",
            "lot_id": "Utf8",
            "chamber_id": "Utf8",
        },
    )
    scores = {
        "semiconductor": SemiconductorIndustryProfile().score_industry(metadata).score,
        "battery": BatteryIndustryProfile().score_industry(metadata).score,
        "automotive": AutomotiveIndustryProfile().score_industry(metadata).score,
    }
    assert scores["semiconductor"] > scores["battery"]
    assert scores["semiconductor"] > scores["automotive"]


def test_battery_metadata_prefers_battery() -> None:
    metadata = _metadata(
        file_name="battery_cell.csv",
        user_description="battery cell discharge_capacity state_of_health",
        column_names=["cell_id", "cycle_index", "soc"],
        dtypes={
            "cell_id": "Utf8",
            "cycle_index": "Int64",
            "soc": "Float64",
        },
    )
    scores = {
        "semiconductor": SemiconductorIndustryProfile().score_industry(metadata).score,
        "battery": BatteryIndustryProfile().score_industry(metadata).score,
        "automotive": AutomotiveIndustryProfile().score_industry(metadata).score,
    }
    assert scores["battery"] > scores["semiconductor"]
    assert scores["battery"] > scores["automotive"]


def test_vehicle_metadata_prefers_automotive() -> None:
    metadata = _metadata(
        file_name="vehicle_engine.csv",
        user_description="automotive vehicle_speed engine_rpm throttle_position",
        column_names=["vehicle_id", "vin", "brake_pressure"],
        dtypes={
            "vehicle_id": "Utf8",
            "vin": "Utf8",
            "brake_pressure": "Float64",
        },
    )
    scores = {
        "semiconductor": SemiconductorIndustryProfile().score_industry(metadata).score,
        "battery": BatteryIndustryProfile().score_industry(metadata).score,
        "automotive": AutomotiveIndustryProfile().score_industry(metadata).score,
    }
    assert scores["automotive"] > scores["semiconductor"]
    assert scores["automotive"] > scores["battery"]


def test_korean_semiconductor_description() -> None:
    score = SemiconductorIndustryProfile().score_industry(
        _metadata(user_description="반도체 웨이퍼 식각 데이터")
    )
    assert score.score > 0.0
    assert any("반도체" in item or "웨이퍼" in item or "식각" in item for item in score.evidence)


def test_korean_battery_description() -> None:
    score = BatteryIndustryProfile().score_industry(
        _metadata(user_description="배터리 양극 음극 전해질")
    )
    assert score.score > 0.0


def test_korean_automotive_description() -> None:
    score = AutomotiveIndustryProfile().score_industry(
        _metadata(user_description="자동차 차량 차속 조향각")
    )
    assert score.score > 0.0


def test_registry_can_register_three_profiles() -> None:
    registry = IndustryRegistry(
        profiles=[
            SemiconductorIndustryProfile(),
            BatteryIndustryProfile(),
            AutomotiveIndustryProfile(),
        ]
    )
    assert registry.list_names() == ("semiconductor", "battery", "automotive")


def test_registry_best_match_semiconductor() -> None:
    registry = IndustryRegistry(
        profiles=[
            SemiconductorIndustryProfile(),
            BatteryIndustryProfile(),
            AutomotiveIndustryProfile(),
        ]
    )
    best = registry.best_match(
        _metadata(
            file_name="wafer.csv",
            user_description="semiconductor lithography",
            column_names=["wafer_id", "chamber_id"],
            dtypes={"wafer_id": "Utf8", "chamber_id": "Utf8"},
        )
    )
    assert best is not None
    assert best.industry_name == "semiconductor"


def test_registry_best_match_battery() -> None:
    registry = IndustryRegistry(
        profiles=[
            SemiconductorIndustryProfile(),
            BatteryIndustryProfile(),
            AutomotiveIndustryProfile(),
        ]
    )
    best = registry.best_match(
        _metadata(
            file_name="battery.csv",
            user_description="battery cell cathode anode",
            column_names=["cell_id", "discharge_capacity"],
            dtypes={"cell_id": "Utf8", "discharge_capacity": "Float64"},
        )
    )
    assert best is not None
    assert best.industry_name == "battery"


def test_registry_best_match_automotive() -> None:
    registry = IndustryRegistry(
        profiles=[
            SemiconductorIndustryProfile(),
            BatteryIndustryProfile(),
            AutomotiveIndustryProfile(),
        ]
    )
    best = registry.best_match(
        _metadata(
            file_name="vehicle.csv",
            user_description="automotive engine_rpm vehicle_speed",
            column_names=["vehicle_id", "vin"],
            dtypes={"vehicle_id": "Utf8", "vin": "Utf8"},
        )
    )
    assert best is not None
    assert best.industry_name == "automotive"


@pytest.mark.parametrize("profile_cls", PROFILES)
def test_get_schema_hints_returns_schema_hints(
    profile_cls: type[BaseIndustryProfile],
) -> None:
    hints = profile_cls().get_schema_hints()
    assert isinstance(hints, SchemaHints)


@pytest.mark.parametrize("profile_cls", PROFILES)
def test_schema_hints_keywords_not_empty(profile_cls: type[BaseIndustryProfile]) -> None:
    assert len(profile_cls().get_schema_hints().keywords) > 0


@pytest.mark.parametrize("profile_cls", PROFILES)
def test_schema_hints_synonyms_not_empty(profile_cls: type[BaseIndustryProfile]) -> None:
    assert len(profile_cls().get_schema_hints().synonyms) > 0


@pytest.mark.parametrize("profile_cls", PROFILES)
def test_schema_hints_expected_units_not_empty(
    profile_cls: type[BaseIndustryProfile],
) -> None:
    assert len(profile_cls().get_schema_hints().expected_units) > 0


@pytest.mark.parametrize("profile_cls", PROFILES)
def test_schema_hints_default_roles_not_empty(
    profile_cls: type[BaseIndustryProfile],
) -> None:
    assert len(profile_cls().get_schema_hints().default_roles) > 0


@pytest.mark.parametrize("profile_cls", PROFILES)
def test_default_roles_are_column_roles(
    profile_cls: type[BaseIndustryProfile],
) -> None:
    roles = profile_cls().get_schema_hints().default_roles
    assert all(isinstance(role, ColumnRole) for role in roles.values())


def test_semiconductor_role_mappings() -> None:
    roles = SemiconductorIndustryProfile().get_schema_hints().default_roles
    assert roles["wafer_id"] == ColumnRole.IDENTIFIER
    assert roles["timestamp"] == ColumnRole.TIME
    assert roles["yield"] == ColumnRole.TARGET_QUALITY


def test_battery_role_mappings() -> None:
    roles = BatteryIndustryProfile().get_schema_hints().default_roles
    assert roles["cell_id"] == ColumnRole.IDENTIFIER
    assert roles["cell_voltage"] == ColumnRole.STATE_SENSOR
    assert roles["discharge_capacity"] == ColumnRole.TARGET_QUALITY


def test_automotive_role_mappings() -> None:
    roles = AutomotiveIndustryProfile().get_schema_hints().default_roles
    assert roles["vehicle_id"] == ColumnRole.IDENTIFIER
    assert roles["engine_rpm"] == ColumnRole.STATE_SENSOR
    assert roles["emission_rate"] == ColumnRole.TARGET_QUALITY


@pytest.mark.parametrize("profile_cls", PROFILES)
def test_physical_ranges_empty_for_this_step(
    profile_cls: type[BaseIndustryProfile],
) -> None:
    assert profile_cls().get_schema_hints().physical_ranges == {}


@pytest.mark.parametrize("profile_cls", PROFILES)
def test_get_schema_hints_returns_independent_instances(
    profile_cls: type[BaseIndustryProfile],
) -> None:
    profile = profile_cls()
    first = profile.get_schema_hints()
    second = profile.get_schema_hints()
    assert first is not second


@pytest.mark.parametrize("profile_cls", PROFILES)
def test_schema_hints_mutation_does_not_affect_next_call(
    profile_cls: type[BaseIndustryProfile],
) -> None:
    profile = profile_cls()
    first = profile.get_schema_hints()
    original_keyword_count = len(first.keywords)
    first.keywords.append("mutated_keyword")
    first.synonyms["mutated"] = ["x"]
    first.expected_units["mutated"] = "unit"
    first.default_roles["mutated"] = ColumnRole.UNKNOWN
    first.physical_ranges["mutated"] = (0.0, 1.0)

    second = profile.get_schema_hints()
    assert len(second.keywords) == original_keyword_count
    assert "mutated_keyword" not in second.keywords
    assert "mutated" not in second.synonyms
    assert "mutated" not in second.expected_units
    assert "mutated" not in second.default_roles
    assert "mutated" not in second.physical_ranges


def test_generic_profile_behavior_preserved() -> None:
    score = GenericIndustryProfile().score_industry(_metadata())
    assert score.industry_name == "generic"
    assert score.score == pytest.approx(0.0)
    assert score.confidence == pytest.approx(0.0)
    assert score.requires_user_confirmation is True


def test_create_default_registry_still_generic_only() -> None:
    assert create_default_registry().list_names() == ("generic",)
