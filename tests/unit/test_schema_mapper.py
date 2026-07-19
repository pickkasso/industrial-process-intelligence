"""Unit tests for ColumnRoleMapper (Step 3D)."""

from __future__ import annotations

import math
from copy import deepcopy

import pytest
from pydantic import ValidationError

from process_intelligence.core.enums import ColumnRole
from process_intelligence.core.exceptions import DataValidationError
from process_intelligence.core.schemas import ColumnRoleAssignment, SchemaHints
from process_intelligence.data.profiler import ColumnProfile, DatasetProfile
from process_intelligence.industries.automotive import AutomotiveIndustryProfile
from process_intelligence.industries.battery import BatteryIndustryProfile
from process_intelligence.industries.semiconductor import SemiconductorIndustryProfile
from process_intelligence.routing import (
    ColumnRoleMapper,
    ColumnRoleMappingPolicy,
    ColumnRoleMappingResult,
)


def _column(
    name: str,
    *,
    dtype: str = "Float64",
    cardinality_ratio: float = 1.0,
    unique_count: int = 10,
    null_count: int = 0,
    null_ratio: float = 0.0,
    is_constant: bool = False,
) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        dtype=dtype,
        null_count=null_count,
        null_ratio=null_ratio,
        unique_count=unique_count,
        cardinality_ratio=cardinality_ratio,
        is_constant=is_constant,
    )


def _profile(
    *columns: ColumnProfile,
    row_count: int | None = None,
) -> DatasetProfile:
    resolved_row_count = 10 if row_count is None else row_count
    return DatasetProfile(
        row_count=resolved_row_count,
        column_count=len(columns),
        columns=list(columns),
        preview_records=[],
    )


def _hints(
    *,
    default_roles: dict[str, ColumnRole] | None = None,
    synonyms: dict[str, list[str]] | None = None,
) -> SchemaHints:
    return SchemaHints(
        default_roles=default_roles or {},
        synonyms=synonyms or {},
    )


def _assignment(
    column_name: str,
    role: ColumnRole = ColumnRole.UNKNOWN,
    *,
    confidence: float = 0.0,
) -> ColumnRoleAssignment:
    return ColumnRoleAssignment(
        column_name=column_name,
        role=role,
        confidence=confidence,
        evidence=["test evidence"],
        alternative_roles=[],
        use_in_model=role
        in {
            ColumnRole.CONTROLLABLE_PROCESS,
            ColumnRole.STATE_SENSOR,
            ColumnRole.CONTEXT,
            ColumnRole.DERIVED_FEATURE,
        },
    )


def _zero_counts() -> dict[ColumnRole, int]:
    return {role: 0 for role in ColumnRole}


def _by_name(result: ColumnRoleMappingResult, name: str) -> ColumnRoleAssignment:
    for assignment in result.assignments:
        if assignment.column_name == name:
            return assignment
    raise AssertionError(f"column {name!r} not found in assignments")


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def test_policy_default_creation() -> None:
    policy = ColumnRoleMappingPolicy()
    assert policy.minimum_auto_confidence == pytest.approx(0.80)
    assert policy.context_cardinality_ratio == pytest.approx(0.20)
    assert policy.identifier_cardinality_ratio == pytest.approx(0.90)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("minimum_auto_confidence", -0.1),
        ("minimum_auto_confidence", 1.1),
        ("context_cardinality_ratio", -0.01),
        ("context_cardinality_ratio", 1.01),
        ("identifier_cardinality_ratio", -1.0),
        ("identifier_cardinality_ratio", 2.0),
    ],
)
def test_policy_rejects_out_of_range(field_name: str, value: float) -> None:
    with pytest.raises(ValidationError):
        ColumnRoleMappingPolicy(**{field_name: value})


@pytest.mark.parametrize("field_name", [
    "minimum_auto_confidence",
    "context_cardinality_ratio",
    "identifier_cardinality_ratio",
])
def test_policy_rejects_bool(field_name: str) -> None:
    with pytest.raises(ValidationError):
        ColumnRoleMappingPolicy(**{field_name: True})


@pytest.mark.parametrize("field_name", [
    "minimum_auto_confidence",
    "context_cardinality_ratio",
    "identifier_cardinality_ratio",
])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_policy_rejects_non_finite(field_name: str, value: float) -> None:
    with pytest.raises(ValidationError):
        ColumnRoleMappingPolicy(**{field_name: value})


def test_policy_rejects_context_threshold_not_below_identifier() -> None:
    with pytest.raises(ValidationError):
        ColumnRoleMappingPolicy(
            context_cardinality_ratio=0.90,
            identifier_cardinality_ratio=0.90,
        )
    with pytest.raises(ValidationError):
        ColumnRoleMappingPolicy(
            context_cardinality_ratio=0.95,
            identifier_cardinality_ratio=0.90,
        )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


def test_result_valid_creation() -> None:
    counts = _zero_counts()
    counts[ColumnRole.UNKNOWN] = 1
    result = ColumnRoleMappingResult(
        assignments=[_assignment("a")],
        unresolved_columns=["a"],
        low_confidence_columns=[],
        requires_user_confirmation=True,
        warnings=[],
        role_counts=counts,
    )
    assert result.assignments[0].column_name == "a"
    assert result.unresolved_columns == ["a"]


def test_result_mutable_defaults_not_shared() -> None:
    first = ColumnRoleMappingResult(requires_user_confirmation=False)
    second = ColumnRoleMappingResult(requires_user_confirmation=False)
    first.assignments.append(_assignment("x"))
    first.warnings.append("w")
    assert second.assignments == []
    assert second.warnings == []


def test_result_rejects_duplicate_assignment_names() -> None:
    counts = _zero_counts()
    counts[ColumnRole.UNKNOWN] = 2
    with pytest.raises(ValidationError):
        ColumnRoleMappingResult(
            assignments=[_assignment("a"), _assignment("a")],
            unresolved_columns=["a", "a"],
            role_counts=counts,
            requires_user_confirmation=True,
        )


def test_result_unresolved_must_match_unknown_assignments() -> None:
    counts = _zero_counts()
    counts[ColumnRole.UNKNOWN] = 1
    with pytest.raises(ValidationError):
        ColumnRoleMappingResult(
            assignments=[_assignment("a")],
            unresolved_columns=[],
            role_counts=counts,
            requires_user_confirmation=False,
        )


def test_result_rejects_unknown_low_confidence_column() -> None:
    counts = _zero_counts()
    counts[ColumnRole.CONTEXT] = 1
    with pytest.raises(ValidationError):
        ColumnRoleMappingResult(
            assignments=[
                _assignment("a", ColumnRole.CONTEXT, confidence=0.5),
            ],
            unresolved_columns=[],
            low_confidence_columns=["missing"],
            role_counts=counts,
            requires_user_confirmation=True,
        )


def test_result_model_dump_round_trip() -> None:
    counts = _zero_counts()
    counts[ColumnRole.TIME] = 1
    original = ColumnRoleMappingResult(
        assignments=[_assignment("ts", ColumnRole.TIME, confidence=0.9)],
        unresolved_columns=[],
        low_confidence_columns=[],
        requires_user_confirmation=False,
        warnings=["note"],
        role_counts=counts,
    )
    restored = ColumnRoleMappingResult.model_validate(original.model_dump())
    assert restored == original


# ---------------------------------------------------------------------------
# Mapper construction and input validation
# ---------------------------------------------------------------------------


def test_mapper_default_construction() -> None:
    mapper = ColumnRoleMapper()
    result = mapper.map_roles(_profile(_column("x")), _hints())
    assert len(result.assignments) == 1


def test_mapper_accepts_policy() -> None:
    policy = ColumnRoleMappingPolicy(minimum_auto_confidence=0.99)
    mapper = ColumnRoleMapper(policy=policy)
    result = mapper.map_roles(
        _profile(_column("flag", dtype="Boolean")),
        _hints(),
    )
    assert result.requires_user_confirmation is True
    assert "flag" in result.low_confidence_columns


def test_mapper_rejects_invalid_policy_type() -> None:
    with pytest.raises(TypeError, match="ColumnRoleMappingPolicy"):
        ColumnRoleMapper(policy={"minimum_auto_confidence": 0.8})  # type: ignore[arg-type]


def test_external_policy_mutation_does_not_affect_mapper() -> None:
    policy = ColumnRoleMappingPolicy(minimum_auto_confidence=0.99)
    mapper = ColumnRoleMapper(policy=policy)
    policy.minimum_auto_confidence = 0.50
    result = mapper.map_roles(
        _profile(_column("flag", dtype="Boolean")),
        _hints(),
    )
    assert "flag" in result.low_confidence_columns


def test_map_roles_rejects_invalid_dataset_profile_type() -> None:
    with pytest.raises(TypeError, match="DatasetProfile"):
        ColumnRoleMapper().map_roles({"columns": []}, _hints())  # type: ignore[arg-type]


def test_map_roles_rejects_invalid_schema_hints_type() -> None:
    with pytest.raises(TypeError, match="SchemaHints"):
        ColumnRoleMapper().map_roles(_profile(_column("x")), {})  # type: ignore[arg-type]


def test_map_roles_rejects_non_mapping_overrides() -> None:
    with pytest.raises(TypeError, match="Mapping"):
        ColumnRoleMapper().map_roles(
            _profile(_column("x")),
            _hints(),
            overrides=["x"],  # type: ignore[arg-type]
        )


def test_override_key_must_be_str() -> None:
    with pytest.raises(TypeError, match="str"):
        ColumnRoleMapper().map_roles(
            _profile(_column("x")),
            _hints(),
            overrides={1: ColumnRole.CONTEXT},  # type: ignore[dict-item]
        )


def test_override_rejects_empty_key() -> None:
    with pytest.raises(DataValidationError):
        ColumnRoleMapper().map_roles(
            _profile(_column("x")),
            _hints(),
            overrides={"": ColumnRole.CONTEXT},
        )


def test_override_rejects_whitespace_key() -> None:
    with pytest.raises(DataValidationError):
        ColumnRoleMapper().map_roles(
            _profile(_column("x")),
            _hints(),
            overrides={"   ": ColumnRole.CONTEXT},
        )


def test_override_value_must_be_column_role() -> None:
    with pytest.raises(TypeError, match="ColumnRole"):
        ColumnRoleMapper().map_roles(
            _profile(_column("x")),
            _hints(),
            overrides={"x": "CONTEXT"},  # type: ignore[dict-item]
        )


def test_override_missing_column_rejected_with_name() -> None:
    with pytest.raises(DataValidationError, match="missing_col"):
        ColumnRoleMapper().map_roles(
            _profile(_column("x")),
            _hints(),
            overrides={"missing_col": ColumnRole.CONTEXT},
        )


def test_override_rejects_original_row_id() -> None:
    with pytest.raises(DataValidationError, match="_original_row_id"):
        ColumnRoleMapper().map_roles(
            _profile(_column("_original_row_id", dtype="Int64")),
            _hints(),
            overrides={"_original_row_id": ColumnRole.CONTEXT},
        )


def test_override_rejects_duplicate_normalized_keys() -> None:
    profile = _profile(_column("wafer_id"), _column("Wafer-ID"))
    with pytest.raises(DataValidationError, match="normalization"):
        ColumnRoleMapper().map_roles(
            profile,
            _hints(),
            overrides={
                "wafer_id": ColumnRole.IDENTIFIER,
                "Wafer-ID": ColumnRole.CONTEXT,
            },
        )


def test_rejects_duplicate_dataset_column_names() -> None:
    profile = DatasetProfile(
        row_count=2,
        column_count=2,
        columns=[_column("dup"), _column("dup")],
        preview_records=[],
    )
    with pytest.raises(DataValidationError, match="duplicate"):
        ColumnRoleMapper().map_roles(profile, _hints())


# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------


def test_override_beats_canonical_and_synonym() -> None:
    hints = _hints(
        default_roles={"wafer_id": ColumnRole.IDENTIFIER},
        synonyms={"wafer_id": ["wafer"]},
    )
    profile = _profile(_column("wafer_id"), _column("wafer"))
    result = ColumnRoleMapper().map_roles(
        profile,
        hints,
        overrides={
            "wafer_id": ColumnRole.CONTEXT,
            "wafer": ColumnRole.STATE_SENSOR,
        },
    )
    assert _by_name(result, "wafer_id").role == ColumnRole.CONTEXT
    assert _by_name(result, "wafer").role == ColumnRole.STATE_SENSOR
    assert _by_name(result, "wafer_id").confidence == pytest.approx(1.0)
    assert _by_name(result, "wafer_id").evidence
    assert _by_name(result, "wafer_id").alternative_roles == []


def test_original_row_id_is_identifier() -> None:
    result = ColumnRoleMapper().map_roles(
        _profile(_column("_original_row_id", dtype="Int64")),
        _hints(),
    )
    assignment = _by_name(result, "_original_row_id")
    assert assignment.role == ColumnRole.IDENTIFIER
    assert assignment.confidence == pytest.approx(1.0)
    assert assignment.use_in_model is False
    assert "lineage" in " ".join(assignment.evidence).casefold() or (
        "_original_row_id" in " ".join(assignment.evidence)
    )


def test_canonical_beats_synonym() -> None:
    hints = _hints(
        default_roles={
            "pressure": ColumnRole.STATE_SENSOR,
            "setpoint": ColumnRole.CONTROLLABLE_PROCESS,
        },
        synonyms={"setpoint": ["pressure"]},
    )
    result = ColumnRoleMapper().map_roles(
        _profile(_column("pressure")),
        hints,
    )
    assignment = _by_name(result, "pressure")
    assert assignment.role == ColumnRole.STATE_SENSOR
    assert assignment.confidence == pytest.approx(0.98)


def test_synonym_match_confidence() -> None:
    hints = _hints(
        default_roles={"wafer_id": ColumnRole.IDENTIFIER},
        synonyms={"wafer_id": ["wafer"]},
    )
    result = ColumnRoleMapper().map_roles(
        _profile(_column("wafer")),
        hints,
    )
    assignment = _by_name(result, "wafer")
    assert assignment.role == ColumnRole.IDENTIFIER
    assert assignment.confidence == pytest.approx(0.95)
    assert "synonym" in " ".join(assignment.evidence).casefold()
    assert "wafer_id" in " ".join(assignment.evidence)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_canonical_match_ignores_case_and_separators() -> None:
    hints = _hints(default_roles={"wafer_id": ColumnRole.IDENTIFIER})
    for name in ("Wafer-ID", "WAFER_ID", "wafer id"):
        result = ColumnRoleMapper().map_roles(_profile(_column(name)), hints)
        assert _by_name(result, name).role == ColumnRole.IDENTIFIER
        assert _by_name(result, name).confidence == pytest.approx(0.98)


def test_synonym_match_ignores_space_underscore() -> None:
    hints = _hints(
        default_roles={"lot_id": ColumnRole.IDENTIFIER},
        synonyms={"lot_id": ["batch id"]},
    )
    result = ColumnRoleMapper().map_roles(
        _profile(_column("batch_id")),
        hints,
    )
    assert _by_name(result, "batch_id").role == ColumnRole.IDENTIFIER


def test_hangul_canonical_normalization() -> None:
    hints = _hints(default_roles={"웨이퍼_id": ColumnRole.IDENTIFIER})
    result = ColumnRoleMapper().map_roles(
        _profile(_column("웨이퍼-id")),
        hints,
    )
    assert _by_name(result, "웨이퍼-id").role == ColumnRole.IDENTIFIER


def test_substring_does_not_match_canonical_or_synonym() -> None:
    hints = _hints(
        default_roles={"wafer_id": ColumnRole.IDENTIFIER},
        synonyms={"wafer_id": ["wafer"]},
    )
    result = ColumnRoleMapper().map_roles(
        _profile(
            _column("super_wafer_id_extra", dtype="Utf8"),
            _column("wafer_name", dtype="Utf8"),
        ),
        hints,
    )
    assert _by_name(result, "super_wafer_id_extra").role != ColumnRole.IDENTIFIER or (
        _by_name(result, "super_wafer_id_extra").confidence
        != pytest.approx(0.98)
    )
    # "wafer_name" must not synonym-match "wafer"
    assignment = _by_name(result, "wafer_name")
    assert assignment.confidence != pytest.approx(0.95) or (
        assignment.role != ColumnRole.IDENTIFIER
    )
    assert "canonical" not in " ".join(assignment.evidence).casefold()
    assert "synonym" not in " ".join(assignment.evidence).casefold()


# ---------------------------------------------------------------------------
# Synonym rules
# ---------------------------------------------------------------------------


def test_synonym_resolves_canonical_role() -> None:
    hints = _hints(
        default_roles={"cell_id": ColumnRole.IDENTIFIER},
        synonyms={"cell_id": ["cellid"]},
    )
    result = ColumnRoleMapper().map_roles(_profile(_column("cellid")), hints)
    assert _by_name(result, "cellid").role == ColumnRole.IDENTIFIER


def test_same_role_multiple_synonym_matches_allowed() -> None:
    hints = _hints(
        default_roles={
            "a_id": ColumnRole.IDENTIFIER,
            "b_id": ColumnRole.IDENTIFIER,
        },
        synonyms={
            "a_id": ["shared"],
            "b_id": ["shared"],
        },
    )
    result = ColumnRoleMapper().map_roles(_profile(_column("shared")), hints)
    assignment = _by_name(result, "shared")
    assert assignment.role == ColumnRole.IDENTIFIER
    assert assignment.confidence == pytest.approx(0.95)
    assert assignment.role != ColumnRole.UNKNOWN


def test_synonym_role_conflict_is_unknown() -> None:
    hints = _hints(
        default_roles={
            "pressure": ColumnRole.STATE_SENSOR,
            "setpoint": ColumnRole.CONTROLLABLE_PROCESS,
        },
        synonyms={
            "pressure": ["p1"],
            "setpoint": ["p1"],
        },
    )
    result = ColumnRoleMapper().map_roles(_profile(_column("p1")), hints)
    assignment = _by_name(result, "p1")
    assert assignment.role == ColumnRole.UNKNOWN
    assert assignment.confidence == pytest.approx(0.0)
    assert assignment.alternative_roles == [
        ColumnRole.STATE_SENSOR,
        ColumnRole.CONTROLLABLE_PROCESS,
    ]
    assert any("ambiguous" in warning.casefold() for warning in result.warnings)
    assert result.requires_user_confirmation is True


def test_synonym_without_default_role_ignored() -> None:
    hints = _hints(
        default_roles={},
        synonyms={"orphan": ["orphan_col"]},
    )
    result = ColumnRoleMapper().map_roles(
        _profile(_column("orphan_col", dtype="Float64")),
        hints,
    )
    assert _by_name(result, "orphan_col").role == ColumnRole.UNKNOWN


def test_synonym_result_is_deterministic() -> None:
    hints = _hints(
        default_roles={
            "pressure": ColumnRole.STATE_SENSOR,
            "setpoint": ColumnRole.CONTROLLABLE_PROCESS,
        },
        synonyms={
            "pressure": ["p1"],
            "setpoint": ["p1"],
        },
    )
    mapper = ColumnRoleMapper()
    profile = _profile(_column("p1"))
    first = mapper.map_roles(profile, hints)
    second = mapper.map_roles(profile, hints)
    assert first.model_dump() == second.model_dump()


# ---------------------------------------------------------------------------
# dtype and name rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dtype", "confidence"),
    [
        ("Date", 0.90),
        ("Datetime(time_unit='us', time_zone=None)", 0.90),
        ("Time", 0.90),
    ],
)
def test_temporal_dtype_maps_to_time(dtype: str, confidence: float) -> None:
    result = ColumnRoleMapper().map_roles(
        _profile(_column("col", dtype=dtype)),
        _hints(),
    )
    assignment = _by_name(result, "col")
    assert assignment.role == ColumnRole.TIME
    assert assignment.confidence == pytest.approx(confidence)


@pytest.mark.parametrize("name", ["timestamp", "event_time"])
def test_exact_time_names(name: str) -> None:
    result = ColumnRoleMapper().map_roles(
        _profile(_column(name, dtype="Float64")),
        _hints(),
    )
    assignment = _by_name(result, name)
    assert assignment.role == ColumnRole.TIME
    assert assignment.confidence == pytest.approx(0.88)


def test_etch_time_not_time_by_name_alone() -> None:
    result = ColumnRoleMapper().map_roles(
        _profile(_column("etch_time", dtype="Float64")),
        _hints(),
    )
    assert _by_name(result, "etch_time").role != ColumnRole.TIME


@pytest.mark.parametrize(
    "name",
    ["wafer_id", "batch id", "uuid", "vin", "serial_number"],
)
def test_identifier_name_patterns(name: str) -> None:
    result = ColumnRoleMapper().map_roles(
        _profile(_column(name, dtype="Utf8")),
        _hints(),
    )
    assert _by_name(result, name).role == ColumnRole.IDENTIFIER
    assert _by_name(result, name).confidence == pytest.approx(0.88)


def test_identity_substring_not_identifier() -> None:
    result = ColumnRoleMapper().map_roles(
        _profile(_column("identity_score", dtype="Float64")),
        _hints(),
    )
    assert _by_name(result, "identity_score").role != ColumnRole.IDENTIFIER


@pytest.mark.parametrize("name", ["target", "label"])
def test_exact_target_names(name: str) -> None:
    result = ColumnRoleMapper().map_roles(
        _profile(_column(name, dtype="Float64")),
        _hints(),
    )
    assignment = _by_name(result, name)
    assert assignment.role == ColumnRole.TARGET_QUALITY
    assert assignment.confidence == pytest.approx(0.85)


def test_combined_target_name_not_auto_target() -> None:
    result = ColumnRoleMapper().map_roles(
        _profile(_column("target_value", dtype="Float64")),
        _hints(),
    )
    assert _by_name(result, "target_value").role != ColumnRole.TARGET_QUALITY


def test_boolean_dtype_is_context_with_target_alternative() -> None:
    result = ColumnRoleMapper().map_roles(
        _profile(_column("flag", dtype="Boolean")),
        _hints(),
    )
    assignment = _by_name(result, "flag")
    assert assignment.role == ColumnRole.CONTEXT
    assert assignment.confidence == pytest.approx(0.80)
    assert assignment.alternative_roles == [ColumnRole.TARGET_QUALITY]


# ---------------------------------------------------------------------------
# Cardinality
# ---------------------------------------------------------------------------


def test_low_cardinality_string_is_context() -> None:
    result = ColumnRoleMapper().map_roles(
        _profile(
            _column("grp", dtype="Utf8", cardinality_ratio=0.10, unique_count=1),
        ),
        _hints(),
    )
    assignment = _by_name(result, "grp")
    assert assignment.role == ColumnRole.CONTEXT
    assert assignment.confidence == pytest.approx(0.75)


def test_context_cardinality_boundary_inclusive() -> None:
    result = ColumnRoleMapper().map_roles(
        _profile(
            _column("grp", dtype="String", cardinality_ratio=0.20, unique_count=2),
        ),
        _hints(),
    )
    assert _by_name(result, "grp").role == ColumnRole.CONTEXT


def test_high_cardinality_string_is_identifier() -> None:
    result = ColumnRoleMapper().map_roles(
        _profile(
            _column("code", dtype="Utf8", cardinality_ratio=0.95, unique_count=95),
        ),
        _hints(),
    )
    assignment = _by_name(result, "code")
    assert assignment.role == ColumnRole.IDENTIFIER
    assert assignment.confidence == pytest.approx(0.70)


def test_identifier_cardinality_boundary_inclusive() -> None:
    result = ColumnRoleMapper().map_roles(
        _profile(
            _column("code", dtype="Utf8", cardinality_ratio=0.90, unique_count=90),
        ),
        _hints(),
    )
    assert _by_name(result, "code").role == ColumnRole.IDENTIFIER


def test_mid_cardinality_string_is_unknown() -> None:
    result = ColumnRoleMapper().map_roles(
        _profile(
            _column("mid", dtype="Utf8", cardinality_ratio=0.50, unique_count=5),
        ),
        _hints(),
    )
    assert _by_name(result, "mid").role == ColumnRole.UNKNOWN


def test_empty_profile_skips_cardinality_inference() -> None:
    result = ColumnRoleMapper().map_roles(
        _profile(
            _column("grp", dtype="Utf8", cardinality_ratio=0.0, unique_count=0),
            row_count=0,
        ),
        _hints(),
    )
    assert _by_name(result, "grp").role == ColumnRole.UNKNOWN


def test_numeric_column_skips_string_cardinality_rules() -> None:
    result = ColumnRoleMapper().map_roles(
        _profile(
            _column("x", dtype="Float64", cardinality_ratio=0.05, unique_count=1),
        ),
        _hints(),
    )
    assert _by_name(result, "x").role == ColumnRole.UNKNOWN


@pytest.mark.parametrize("dtype", ["String", "Utf8", "Categorical", "Enum"])
def test_string_like_dtypes(dtype: str) -> None:
    result = ColumnRoleMapper().map_roles(
        _profile(
            _column("grp", dtype=dtype, cardinality_ratio=0.05, unique_count=1),
        ),
        _hints(),
    )
    assert _by_name(result, "grp").role == ColumnRole.CONTEXT


# ---------------------------------------------------------------------------
# UNKNOWN and alternatives
# ---------------------------------------------------------------------------


def test_unclassified_numeric_unknown_alternatives() -> None:
    result = ColumnRoleMapper().map_roles(
        _profile(_column("sensor_x", dtype="Float64")),
        _hints(),
    )
    assignment = _by_name(result, "sensor_x")
    assert assignment.role == ColumnRole.UNKNOWN
    assert assignment.alternative_roles == [
        ColumnRole.CONTROLLABLE_PROCESS,
        ColumnRole.STATE_SENSOR,
        ColumnRole.TARGET_QUALITY,
    ]


def test_unclassified_string_alternatives() -> None:
    result = ColumnRoleMapper().map_roles(
        _profile(
            _column("mid", dtype="Utf8", cardinality_ratio=0.5, unique_count=5),
        ),
        _hints(),
    )
    assert _by_name(result, "mid").alternative_roles == [
        ColumnRole.CONTEXT,
        ColumnRole.IDENTIFIER,
    ]


def test_other_dtype_unknown_alternatives_empty() -> None:
    result = ColumnRoleMapper().map_roles(
        _profile(_column("blob", dtype="Object")),
        _hints(),
    )
    assert _by_name(result, "blob").alternative_roles == []


def test_alternative_roles_exclude_selected_and_duplicates() -> None:
    result = ColumnRoleMapper().map_roles(
        _profile(_column("flag", dtype="Boolean")),
        _hints(),
    )
    assignment = _by_name(result, "flag")
    assert assignment.role not in assignment.alternative_roles
    assert len(assignment.alternative_roles) == len(set(assignment.alternative_roles))


# ---------------------------------------------------------------------------
# Confidence and confirmation
# ---------------------------------------------------------------------------


def test_unresolved_and_low_confidence_ordering() -> None:
    policy = ColumnRoleMappingPolicy(minimum_auto_confidence=0.80)
    result = ColumnRoleMapper(policy=policy).map_roles(
        _profile(
            _column("a", dtype="Float64"),
            _column("b", dtype="Utf8", cardinality_ratio=0.05, unique_count=1),
            _column("c", dtype="Float64"),
            _column("d", dtype="Utf8", cardinality_ratio=0.95, unique_count=95),
        ),
        _hints(),
    )
    assert result.unresolved_columns == ["a", "c"]
    assert result.low_confidence_columns == ["b", "d"]
    assert result.requires_user_confirmation is True


def test_confidence_equal_to_threshold_not_low() -> None:
    result = ColumnRoleMapper().map_roles(
        _profile(_column("flag", dtype="Boolean")),
        _hints(),
    )
    assert "flag" not in result.low_confidence_columns
    assert _by_name(result, "flag").confidence == pytest.approx(0.80)


def test_requires_confirmation_false_when_all_high_confidence() -> None:
    hints = _hints(
        default_roles={
            "wafer_id": ColumnRole.IDENTIFIER,
            "rf_power": ColumnRole.CONTROLLABLE_PROCESS,
        },
    )
    result = ColumnRoleMapper().map_roles(
        _profile(_column("wafer_id"), _column("rf_power")),
        hints,
    )
    assert result.unresolved_columns == []
    assert result.low_confidence_columns == []
    assert result.requires_user_confirmation is False


def test_unresolved_alone_requires_confirmation() -> None:
    result = ColumnRoleMapper().map_roles(
        _profile(_column("x", dtype="Float64")),
        _hints(),
    )
    assert result.requires_user_confirmation is True


# ---------------------------------------------------------------------------
# use_in_model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        (ColumnRole.CONTROLLABLE_PROCESS, True),
        (ColumnRole.STATE_SENSOR, True),
        (ColumnRole.CONTEXT, True),
        (ColumnRole.DERIVED_FEATURE, True),
        (ColumnRole.IDENTIFIER, False),
        (ColumnRole.TIME, False),
        (ColumnRole.TARGET_QUALITY, False),
        (ColumnRole.UNKNOWN, False),
    ],
)
def test_use_in_model_by_role(role: ColumnRole, expected: bool) -> None:
    result = ColumnRoleMapper().map_roles(
        _profile(_column("col")),
        _hints(),
        overrides={"col": role},
    )
    assert _by_name(result, "col").use_in_model is expected


# ---------------------------------------------------------------------------
# Order and aggregation
# ---------------------------------------------------------------------------


def test_assignment_order_and_role_counts() -> None:
    hints = _hints(
        default_roles={
            "id_col": ColumnRole.IDENTIFIER,
            "t": ColumnRole.TIME,
            "p": ColumnRole.CONTROLLABLE_PROCESS,
        },
    )
    result = ColumnRoleMapper().map_roles(
        _profile(
            _column("p"),
            _column("id_col"),
            _column("t"),
            _column("x", dtype="Float64"),
        ),
        hints,
    )
    assert [a.column_name for a in result.assignments] == [
        "p",
        "id_col",
        "t",
        "x",
    ]
    assert list(result.role_counts.keys()) == list(ColumnRole)
    assert result.role_counts[ColumnRole.CONTROLLABLE_PROCESS] == 1
    assert result.role_counts[ColumnRole.IDENTIFIER] == 1
    assert result.role_counts[ColumnRole.TIME] == 1
    assert result.role_counts[ColumnRole.UNKNOWN] == 1
    for role in ColumnRole:
        assert result.role_counts[role] == sum(
            1 for a in result.assignments if a.role == role
        )


def test_warnings_preserve_column_order_without_duplicates() -> None:
    hints = _hints(
        default_roles={
            "pressure": ColumnRole.STATE_SENSOR,
            "setpoint": ColumnRole.CONTROLLABLE_PROCESS,
        },
        synonyms={
            "pressure": ["amb"],
            "setpoint": ["amb"],
        },
    )
    result = ColumnRoleMapper().map_roles(
        _profile(_column("amb"), _column("other", dtype="Float64"), _column("amb2")),
        _hints(
            default_roles=hints.default_roles,
            synonyms={
                "pressure": ["amb", "amb2"],
                "setpoint": ["amb", "amb2"],
            },
        ),
    )
    assert len(result.warnings) == 2
    assert "amb" in result.warnings[0]
    assert "amb2" in result.warnings[1]
    assert len(result.warnings) == len(set(result.warnings))


# ---------------------------------------------------------------------------
# Industry SchemaHints integration
# ---------------------------------------------------------------------------


def test_semiconductor_schema_hints_roles() -> None:
    hints = SemiconductorIndustryProfile().get_schema_hints()
    result = ColumnRoleMapper().map_roles(
        _profile(
            _column("wafer_id"),
            _column("timestamp"),
            _column("rf_power"),
            _column("chamber_pressure"),
            _column("yield"),
        ),
        hints,
    )
    assert _by_name(result, "wafer_id").role == ColumnRole.IDENTIFIER
    assert _by_name(result, "timestamp").role == ColumnRole.TIME
    assert _by_name(result, "rf_power").role == ColumnRole.CONTROLLABLE_PROCESS
    assert _by_name(result, "chamber_pressure").role == ColumnRole.STATE_SENSOR
    assert _by_name(result, "yield").role == ColumnRole.TARGET_QUALITY


def test_battery_schema_hints_roles() -> None:
    hints = BatteryIndustryProfile().get_schema_hints()
    result = ColumnRoleMapper().map_roles(
        _profile(
            _column("cell_id"),
            _column("cycle_index"),
            _column("charge_current"),
            _column("cell_voltage"),
            _column("discharge_capacity"),
        ),
        hints,
    )
    assert _by_name(result, "cell_id").role == ColumnRole.IDENTIFIER
    assert _by_name(result, "cycle_index").role == ColumnRole.TIME
    assert _by_name(result, "charge_current").role == ColumnRole.CONTROLLABLE_PROCESS
    assert _by_name(result, "cell_voltage").role == ColumnRole.STATE_SENSOR
    assert _by_name(result, "discharge_capacity").role == ColumnRole.TARGET_QUALITY


def test_automotive_schema_hints_roles() -> None:
    hints = AutomotiveIndustryProfile().get_schema_hints()
    result = ColumnRoleMapper().map_roles(
        _profile(
            _column("vehicle_id"),
            _column("throttle_position"),
            _column("engine_rpm"),
            _column("road_type", dtype="Utf8", cardinality_ratio=0.1),
            _column("emission_rate"),
        ),
        hints,
    )
    assert _by_name(result, "vehicle_id").role == ColumnRole.IDENTIFIER
    assert _by_name(result, "throttle_position").role == ColumnRole.CONTROLLABLE_PROCESS
    assert _by_name(result, "engine_rpm").role == ColumnRole.STATE_SENSOR
    assert _by_name(result, "road_type").role == ColumnRole.CONTEXT
    assert _by_name(result, "emission_rate").role == ColumnRole.TARGET_QUALITY


def test_industry_synonym_maps_real_column_name() -> None:
    hints = SemiconductorIndustryProfile().get_schema_hints()
    result = ColumnRoleMapper().map_roles(
        _profile(_column("process_yield", dtype="Float64")),
        hints,
    )
    assert _by_name(result, "process_yield").role == ColumnRole.TARGET_QUALITY
    assert _by_name(result, "process_yield").confidence == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Immutability and regression
# ---------------------------------------------------------------------------


def test_map_roles_does_not_mutate_inputs() -> None:
    profile = _profile(_column("wafer_id"), _column("rf_power"))
    hints = SemiconductorIndustryProfile().get_schema_hints()
    overrides: dict[str, ColumnRole] = {"rf_power": ColumnRole.STATE_SENSOR}

    profile_before = deepcopy(profile.model_dump())
    hints_before = deepcopy(hints.model_dump())
    overrides_before = dict(overrides)

    ColumnRoleMapper().map_roles(profile, hints, overrides=overrides)

    assert profile.model_dump() == profile_before
    assert hints.model_dump() == hints_before
    assert overrides == overrides_before


def test_consecutive_calls_do_not_accumulate() -> None:
    mapper = ColumnRoleMapper()
    first = mapper.map_roles(_profile(_column("a")), _hints())
    second = mapper.map_roles(_profile(_column("b"), _column("c")), _hints())
    assert len(first.assignments) == 1
    assert len(second.assignments) == 2
    assert [a.column_name for a in second.assignments] == ["b", "c"]


def test_separate_mapper_instances_do_not_share_state() -> None:
    left = ColumnRoleMapper(
        policy=ColumnRoleMappingPolicy(minimum_auto_confidence=0.99),
    )
    right = ColumnRoleMapper(
        policy=ColumnRoleMappingPolicy(minimum_auto_confidence=0.50),
    )
    profile = _profile(_column("flag", dtype="Boolean"))
    left_result = left.map_roles(profile, _hints())
    right_result = right.map_roles(profile, _hints())
    assert left_result.requires_user_confirmation is True
    assert right_result.requires_user_confirmation is False


def test_identical_inputs_are_deterministic() -> None:
    mapper = ColumnRoleMapper()
    profile = _profile(_column("x"), _column("y", dtype="Boolean"))
    hints = _hints(default_roles={"x": ColumnRole.STATE_SENSOR})
    assert mapper.map_roles(profile, hints).model_dump() == mapper.map_roles(
        profile, hints
    ).model_dump()


def test_mutating_returned_assignments_does_not_affect_next_call() -> None:
    mapper = ColumnRoleMapper()
    profile = _profile(_column("x"))
    first = mapper.map_roles(profile, _hints())
    first.assignments.clear()
    first.warnings.append("mutated")
    second = mapper.map_roles(profile, _hints())
    assert len(second.assignments) == 1
    assert second.warnings == []


def test_routing_exports_include_schema_mapper() -> None:
    from process_intelligence import routing

    assert routing.ColumnRoleMapper is ColumnRoleMapper
    assert routing.ColumnRoleMappingPolicy is ColumnRoleMappingPolicy
    assert routing.ColumnRoleMappingResult is ColumnRoleMappingResult
