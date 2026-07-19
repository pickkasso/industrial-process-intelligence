"""Unit tests for AnalysisTaskRouter (Step 3E)."""

from __future__ import annotations

import math
from copy import deepcopy

import pytest
from pydantic import ValidationError

from process_intelligence.core.enums import AnalysisTask, ColumnRole
from process_intelligence.core.exceptions import DataValidationError, InsufficientDataError
from process_intelligence.core.schemas import ColumnRoleAssignment
from process_intelligence.data.profiler import ColumnProfile, DatasetProfile
from process_intelligence.industries.automotive import AutomotiveIndustryProfile
from process_intelligence.industries.battery import BatteryIndustryProfile
from process_intelligence.industries.semiconductor import SemiconductorIndustryProfile
from process_intelligence.routing import (
    AnalysisTaskRouter,
    ColumnRoleMapper,
    TaskRoutingPolicy,
    TaskRoutingResult,
    create_default_task_router,
)
from process_intelligence.routing.schema_mapper import ColumnRoleMappingResult


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
    resolved_row_count = 100 if row_count is None else row_count
    return DatasetProfile(
        row_count=resolved_row_count,
        column_count=len(columns),
        columns=list(columns),
        preview_records=[],
    )


def _assignment(
    column_name: str,
    role: ColumnRole = ColumnRole.UNKNOWN,
    *,
    confidence: float = 0.95,
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


def _mapping(
    *assignments: ColumnRoleAssignment,
    low_confidence_columns: list[str] | None = None,
) -> ColumnRoleMappingResult:
    counts = _zero_counts()
    for assignment in assignments:
        counts[assignment.role] += 1
    unresolved = [
        assignment.column_name
        for assignment in assignments
        if assignment.role == ColumnRole.UNKNOWN
    ]
    low = list(low_confidence_columns or [])
    return ColumnRoleMappingResult(
        assignments=list(assignments),
        unresolved_columns=unresolved,
        low_confidence_columns=low,
        requires_user_confirmation=bool(unresolved or low),
        warnings=[],
        role_counts=counts,
    )


def _result(
    *,
    decision: str,
    selected_task: AnalysisTask,
    selected_target: str | None = None,
    selected_time_column: str | None = None,
    candidate_tasks: list[AnalysisTask] | None = None,
    target_candidates: list[str] | None = None,
    time_candidates: list[str] | None = None,
    requires_user_confirmation: bool,
    user_confirmed: bool,
    reason: str = "test reason",
    evidence: list[str] | None = None,
    warnings: list[str] | None = None,
) -> TaskRoutingResult:
    tasks = candidate_tasks if candidate_tasks is not None else [selected_task]
    return TaskRoutingResult(
        decision=decision,  # type: ignore[arg-type]
        selected_task=selected_task,
        selected_target=selected_target,
        selected_time_column=selected_time_column,
        candidate_tasks=tasks,
        target_candidates=list(target_candidates or []),
        time_candidates=list(time_candidates or []),
        requires_user_confirmation=requires_user_confirmation,
        user_confirmed=user_confirmed,
        reason=reason,
        evidence=list(evidence or ["evidence"]),
        warnings=list(warnings or []),
    )


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def test_policy_default_creation() -> None:
    policy = TaskRoutingPolicy()
    assert isinstance(policy, TaskRoutingPolicy)


def test_policy_default_values() -> None:
    policy = TaskRoutingPolicy()
    assert policy.classification_max_unique_values == 20
    assert policy.classification_max_cardinality_ratio == pytest.approx(0.05)
    assert policy.minimum_target_non_null_values == 2
    assert policy.prefer_time_series_anomaly is True


def test_policy_rejects_classification_max_unique_below_two() -> None:
    with pytest.raises(ValidationError):
        TaskRoutingPolicy(classification_max_unique_values=1)


def test_policy_rejects_minimum_target_non_null_below_two() -> None:
    with pytest.raises(ValidationError):
        TaskRoutingPolicy(minimum_target_non_null_values=1)


@pytest.mark.parametrize(
    "field_name",
    ["classification_max_unique_values", "minimum_target_non_null_values"],
)
def test_policy_rejects_bool_for_int_fields(field_name: str) -> None:
    with pytest.raises(ValidationError):
        TaskRoutingPolicy(**{field_name: True})


def test_policy_rejects_negative_cardinality_ratio() -> None:
    with pytest.raises(ValidationError):
        TaskRoutingPolicy(classification_max_cardinality_ratio=-0.01)


def test_policy_rejects_cardinality_ratio_above_one() -> None:
    with pytest.raises(ValidationError):
        TaskRoutingPolicy(classification_max_cardinality_ratio=1.01)


def test_policy_rejects_bool_cardinality_ratio() -> None:
    with pytest.raises(ValidationError):
        TaskRoutingPolicy(classification_max_cardinality_ratio=True)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_policy_rejects_non_finite_cardinality_ratio(value: float) -> None:
    with pytest.raises(ValidationError):
        TaskRoutingPolicy(classification_max_cardinality_ratio=value)


def test_policy_rejects_non_bool_prefer_time_series() -> None:
    with pytest.raises(ValidationError):
        TaskRoutingPolicy(prefer_time_series_anomaly=1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


def test_result_auto_selected() -> None:
    result = _result(
        decision="AUTO_SELECTED",
        selected_task=AnalysisTask.REGRESSION,
        selected_target="y",
        candidate_tasks=[AnalysisTask.REGRESSION, AnalysisTask.RESIDUAL_ANOMALY],
        target_candidates=["y"],
        requires_user_confirmation=False,
        user_confirmed=False,
    )
    assert result.decision == "AUTO_SELECTED"
    assert result.requires_user_confirmation is False
    assert result.user_confirmed is False


def test_result_confirmation_required() -> None:
    result = _result(
        decision="CONFIRMATION_REQUIRED",
        selected_task=AnalysisTask.CLASSIFICATION,
        selected_target="y",
        candidate_tasks=[
            AnalysisTask.CLASSIFICATION,
            AnalysisTask.REGRESSION,
            AnalysisTask.RESIDUAL_ANOMALY,
        ],
        target_candidates=["y"],
        requires_user_confirmation=True,
        user_confirmed=False,
    )
    assert result.decision == "CONFIRMATION_REQUIRED"
    assert result.requires_user_confirmation is True


def test_result_user_confirmed() -> None:
    result = _result(
        decision="USER_CONFIRMED",
        selected_task=AnalysisTask.TIME_SERIES_ANOMALY,
        selected_time_column="ts",
        candidate_tasks=[
            AnalysisTask.TIME_SERIES_ANOMALY,
            AnalysisTask.UNSUPERVISED_ANOMALY,
        ],
        time_candidates=["ts"],
        requires_user_confirmation=False,
        user_confirmed=True,
    )
    assert result.decision == "USER_CONFIRMED"
    assert result.user_confirmed is True


def test_result_rejects_blank_reason() -> None:
    with pytest.raises(ValidationError):
        _result(
            decision="AUTO_SELECTED",
            selected_task=AnalysisTask.UNSUPERVISED_ANOMALY,
            requires_user_confirmation=False,
            user_confirmed=False,
            reason="   ",
        )


def test_result_rejects_selected_task_missing_from_candidates() -> None:
    with pytest.raises(ValidationError):
        TaskRoutingResult(
            decision="AUTO_SELECTED",
            selected_task=AnalysisTask.REGRESSION,
            selected_target="y",
            candidate_tasks=[AnalysisTask.CLASSIFICATION],
            target_candidates=["y"],
            requires_user_confirmation=False,
            user_confirmed=False,
            reason="bad",
        )


def test_result_rejects_duplicate_candidate_tasks() -> None:
    with pytest.raises(ValidationError):
        TaskRoutingResult(
            decision="AUTO_SELECTED",
            selected_task=AnalysisTask.REGRESSION,
            selected_target="y",
            candidate_tasks=[AnalysisTask.REGRESSION, AnalysisTask.REGRESSION],
            target_candidates=["y"],
            requires_user_confirmation=False,
            user_confirmed=False,
            reason="bad",
        )


def test_result_rejects_duplicate_target_candidates() -> None:
    with pytest.raises(ValidationError):
        _result(
            decision="AUTO_SELECTED",
            selected_task=AnalysisTask.REGRESSION,
            selected_target="y",
            target_candidates=["y", "y"],
            requires_user_confirmation=False,
            user_confirmed=False,
        )


def test_result_rejects_duplicate_time_candidates() -> None:
    with pytest.raises(ValidationError):
        _result(
            decision="AUTO_SELECTED",
            selected_task=AnalysisTask.TIME_SERIES_ANOMALY,
            selected_time_column="ts",
            time_candidates=["ts", "ts"],
            requires_user_confirmation=False,
            user_confirmed=False,
        )


def test_result_rejects_selected_target_not_in_candidates() -> None:
    with pytest.raises(ValidationError):
        _result(
            decision="AUTO_SELECTED",
            selected_task=AnalysisTask.REGRESSION,
            selected_target="y",
            target_candidates=["other"],
            requires_user_confirmation=False,
            user_confirmed=False,
        )


def test_result_rejects_selected_time_not_in_candidates() -> None:
    with pytest.raises(ValidationError):
        _result(
            decision="AUTO_SELECTED",
            selected_task=AnalysisTask.TIME_SERIES_ANOMALY,
            selected_time_column="ts",
            time_candidates=["other"],
            requires_user_confirmation=False,
            user_confirmed=False,
        )


@pytest.mark.parametrize(
    ("decision", "requires", "confirmed"),
    [
        ("AUTO_SELECTED", True, False),
        ("AUTO_SELECTED", False, True),
        ("CONFIRMATION_REQUIRED", False, False),
        ("CONFIRMATION_REQUIRED", True, True),
        ("USER_CONFIRMED", True, True),
        ("USER_CONFIRMED", False, False),
    ],
)
def test_result_rejects_decision_flag_mismatch(
    decision: str,
    requires: bool,
    confirmed: bool,
) -> None:
    with pytest.raises(ValidationError):
        _result(
            decision=decision,
            selected_task=AnalysisTask.UNSUPERVISED_ANOMALY,
            requires_user_confirmation=requires,
            user_confirmed=confirmed,
        )


def test_result_rejects_regression_without_target() -> None:
    with pytest.raises(ValidationError):
        _result(
            decision="AUTO_SELECTED",
            selected_task=AnalysisTask.REGRESSION,
            requires_user_confirmation=False,
            user_confirmed=False,
        )


def test_result_rejects_classification_without_target() -> None:
    with pytest.raises(ValidationError):
        _result(
            decision="AUTO_SELECTED",
            selected_task=AnalysisTask.CLASSIFICATION,
            requires_user_confirmation=False,
            user_confirmed=False,
        )


def test_result_rejects_residual_without_target() -> None:
    with pytest.raises(ValidationError):
        _result(
            decision="AUTO_SELECTED",
            selected_task=AnalysisTask.RESIDUAL_ANOMALY,
            requires_user_confirmation=False,
            user_confirmed=False,
        )


def test_result_rejects_time_series_without_time() -> None:
    with pytest.raises(ValidationError):
        _result(
            decision="AUTO_SELECTED",
            selected_task=AnalysisTask.TIME_SERIES_ANOMALY,
            requires_user_confirmation=False,
            user_confirmed=False,
        )


def test_result_rejects_drift_without_time() -> None:
    with pytest.raises(ValidationError):
        _result(
            decision="AUTO_SELECTED",
            selected_task=AnalysisTask.DRIFT_DETECTION,
            requires_user_confirmation=False,
            user_confirmed=False,
        )


def test_result_mutable_default_collections_independent() -> None:
    first = TaskRoutingResult(
        decision="AUTO_SELECTED",
        selected_task=AnalysisTask.UNSUPERVISED_ANOMALY,
        candidate_tasks=[AnalysisTask.UNSUPERVISED_ANOMALY],
        requires_user_confirmation=False,
        user_confirmed=False,
        reason="ok",
    )
    second = TaskRoutingResult(
        decision="AUTO_SELECTED",
        selected_task=AnalysisTask.UNSUPERVISED_ANOMALY,
        candidate_tasks=[AnalysisTask.UNSUPERVISED_ANOMALY],
        requires_user_confirmation=False,
        user_confirmed=False,
        reason="ok",
    )
    first.warnings.append("w")
    first.target_candidates.append("x")
    assert second.warnings == []
    assert second.target_candidates == []


def test_result_model_dump_validate_round_trip() -> None:
    original = _result(
        decision="CONFIRMATION_REQUIRED",
        selected_task=AnalysisTask.CLASSIFICATION,
        selected_target="y",
        candidate_tasks=[
            AnalysisTask.CLASSIFICATION,
            AnalysisTask.REGRESSION,
            AnalysisTask.RESIDUAL_ANOMALY,
        ],
        target_candidates=["y"],
        requires_user_confirmation=True,
        user_confirmed=False,
        evidence=["a", "b"],
        warnings=["ambiguous"],
    )
    restored = TaskRoutingResult.model_validate(original.model_dump())
    assert restored.model_dump() == original.model_dump()


# ---------------------------------------------------------------------------
# Router input validation
# ---------------------------------------------------------------------------


def test_router_default_creation() -> None:
    router = AnalysisTaskRouter()
    assert isinstance(router, AnalysisTaskRouter)


def test_router_with_policy() -> None:
    policy = TaskRoutingPolicy(prefer_time_series_anomaly=False)
    router = AnalysisTaskRouter(policy=policy)
    profile = _profile(_column("x"))
    mapping = _mapping(_assignment("x", ColumnRole.STATE_SENSOR))
    result = router.route(profile, mapping)
    assert result.selected_task == AnalysisTask.UNSUPERVISED_ANOMALY


def test_router_rejects_non_policy() -> None:
    with pytest.raises(TypeError):
        AnalysisTaskRouter(policy={"prefer_time_series_anomaly": True})  # type: ignore[arg-type]


def test_external_policy_mutation_does_not_affect_router() -> None:
    policy = TaskRoutingPolicy(prefer_time_series_anomaly=True)
    router = AnalysisTaskRouter(policy=policy)
    policy.prefer_time_series_anomaly = False
    profile = _profile(_column("ts", dtype="Datetime"), _column("x"))
    mapping = _mapping(
        _assignment("ts", ColumnRole.TIME),
        _assignment("x", ColumnRole.STATE_SENSOR),
    )
    result = router.route(profile, mapping)
    assert result.selected_task == AnalysisTask.TIME_SERIES_ANOMALY


def test_route_rejects_non_dataset_profile() -> None:
    with pytest.raises(TypeError):
        AnalysisTaskRouter().route(
            {"row_count": 1},  # type: ignore[arg-type]
            _mapping(_assignment("x")),
        )


def test_route_rejects_non_role_mapping() -> None:
    with pytest.raises(TypeError):
        AnalysisTaskRouter().route(_profile(_column("x")), {"assignments": []})  # type: ignore[arg-type]


def test_route_rejects_profile_assignment_order_mismatch() -> None:
    profile = _profile(_column("a"), _column("b"))
    mapping = _mapping(
        _assignment("b", ColumnRole.STATE_SENSOR),
        _assignment("a", ColumnRole.TARGET_QUALITY),
    )
    with pytest.raises(DataValidationError):
        AnalysisTaskRouter().route(profile, mapping)


def test_route_rejects_duplicate_profile_column_names() -> None:
    profile = DatasetProfile(
        row_count=10,
        column_count=2,
        columns=[_column("a"), _column("a")],
        preview_records=[],
    )
    # Bypass mapping validators so profile duplicate detection is exercised.
    mapping = ColumnRoleMappingResult.model_construct(
        assignments=[
            _assignment("a", ColumnRole.STATE_SENSOR),
            _assignment("a", ColumnRole.TARGET_QUALITY),
        ],
        unresolved_columns=[],
        low_confidence_columns=[],
        requires_user_confirmation=False,
        warnings=[],
        role_counts=_zero_counts(),
    )
    with pytest.raises(DataValidationError):
        AnalysisTaskRouter().route(profile, mapping)


def test_route_rejects_non_analysis_task_confirmed() -> None:
    profile = _profile(_column("y"), _column("x"))
    mapping = _mapping(
        _assignment("y", ColumnRole.TARGET_QUALITY),
        _assignment("x", ColumnRole.STATE_SENSOR),
    )
    with pytest.raises(TypeError):
        AnalysisTaskRouter().route(
            profile,
            mapping,
            confirmed_task="REGRESSION",  # type: ignore[arg-type]
        )


def test_route_rejects_non_string_confirmed_target() -> None:
    profile = _profile(_column("y"))
    mapping = _mapping(_assignment("y", ColumnRole.TARGET_QUALITY))
    with pytest.raises(TypeError):
        AnalysisTaskRouter().route(
            profile,
            mapping,
            confirmed_target=123,  # type: ignore[arg-type]
        )


def test_route_rejects_blank_confirmed_target() -> None:
    profile = _profile(_column("y"))
    mapping = _mapping(_assignment("y", ColumnRole.TARGET_QUALITY))
    with pytest.raises(ValueError):
        AnalysisTaskRouter().route(profile, mapping, confirmed_target="  ")


def test_route_rejects_missing_confirmed_target() -> None:
    profile = _profile(_column("y"))
    mapping = _mapping(_assignment("y", ColumnRole.TARGET_QUALITY))
    with pytest.raises(DataValidationError):
        AnalysisTaskRouter().route(profile, mapping, confirmed_target="missing")


def test_route_rejects_original_row_id_as_target() -> None:
    profile = _profile(_column("_original_row_id"), _column("y"))
    mapping = _mapping(
        _assignment("_original_row_id", ColumnRole.IDENTIFIER),
        _assignment("y", ColumnRole.TARGET_QUALITY),
    )
    with pytest.raises(DataValidationError):
        AnalysisTaskRouter().route(
            profile,
            mapping,
            confirmed_target="_original_row_id",
        )


def test_route_rejects_non_string_confirmed_time() -> None:
    profile = _profile(_column("ts", dtype="Datetime"))
    mapping = _mapping(_assignment("ts", ColumnRole.TIME))
    with pytest.raises(TypeError):
        AnalysisTaskRouter().route(
            profile,
            mapping,
            confirmed_time_column=1,  # type: ignore[arg-type]
        )


def test_route_rejects_blank_confirmed_time() -> None:
    profile = _profile(_column("ts", dtype="Datetime"))
    mapping = _mapping(_assignment("ts", ColumnRole.TIME))
    with pytest.raises(ValueError):
        AnalysisTaskRouter().route(profile, mapping, confirmed_time_column="")


def test_route_rejects_missing_confirmed_time() -> None:
    profile = _profile(_column("ts", dtype="Datetime"))
    mapping = _mapping(_assignment("ts", ColumnRole.TIME))
    with pytest.raises(DataValidationError):
        AnalysisTaskRouter().route(
            profile,
            mapping,
            confirmed_time_column="missing",
        )


def test_route_rejects_original_row_id_as_time() -> None:
    profile = _profile(_column("_original_row_id"), _column("ts", dtype="Datetime"))
    mapping = _mapping(
        _assignment("_original_row_id", ColumnRole.IDENTIFIER),
        _assignment("ts", ColumnRole.TIME),
    )
    with pytest.raises(DataValidationError):
        AnalysisTaskRouter().route(
            profile,
            mapping,
            confirmed_time_column="_original_row_id",
        )


# ---------------------------------------------------------------------------
# Automatic supervised routing
# ---------------------------------------------------------------------------


def test_boolean_target_classification() -> None:
    profile = _profile(_column("flag", dtype="Boolean", unique_count=2, cardinality_ratio=0.02))
    mapping = _mapping(_assignment("flag", ColumnRole.TARGET_QUALITY))
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.selected_task == AnalysisTask.CLASSIFICATION
    assert result.decision == "AUTO_SELECTED"
    assert result.candidate_tasks == [
        AnalysisTask.CLASSIFICATION,
        AnalysisTask.RESIDUAL_ANOMALY,
    ]


def test_string_target_classification() -> None:
    profile = _profile(
        _column("label", dtype="Utf8", unique_count=3, cardinality_ratio=0.03)
    )
    mapping = _mapping(_assignment("label", ColumnRole.TARGET_QUALITY))
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.selected_task == AnalysisTask.CLASSIFICATION
    assert result.decision == "AUTO_SELECTED"


def test_categorical_target_classification() -> None:
    profile = _profile(
        _column("grade", dtype="Categorical", unique_count=4, cardinality_ratio=0.04)
    )
    mapping = _mapping(_assignment("grade", ColumnRole.TARGET_QUALITY))
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.selected_task == AnalysisTask.CLASSIFICATION


def test_continuous_numeric_target_regression() -> None:
    profile = _profile(
        _column("yield", dtype="Float64", unique_count=80, cardinality_ratio=0.8)
    )
    mapping = _mapping(_assignment("yield", ColumnRole.TARGET_QUALITY))
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.selected_task == AnalysisTask.REGRESSION
    assert result.decision == "AUTO_SELECTED"
    assert result.candidate_tasks == [
        AnalysisTask.REGRESSION,
        AnalysisTask.RESIDUAL_ANOMALY,
    ]


def test_low_cardinality_int_target_ambiguous() -> None:
    profile = _profile(
        _column("class_id", dtype="Int64", unique_count=5, cardinality_ratio=0.05)
    )
    mapping = _mapping(_assignment("class_id", ColumnRole.TARGET_QUALITY))
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.selected_task == AnalysisTask.CLASSIFICATION
    assert result.decision == "CONFIRMATION_REQUIRED"
    assert result.candidate_tasks == [
        AnalysisTask.CLASSIFICATION,
        AnalysisTask.REGRESSION,
        AnalysisTask.RESIDUAL_ANOMALY,
    ]
    assert any(
        "ambiguous" in w.casefold() or "cardinality" in w.casefold()
        for w in result.warnings
    )


def test_low_cardinality_float_target_ambiguous() -> None:
    profile = _profile(
        _column("bin", dtype="Float64", unique_count=4, cardinality_ratio=0.04)
    )
    mapping = _mapping(_assignment("bin", ColumnRole.TARGET_QUALITY))
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.decision == "CONFIRMATION_REQUIRED"
    assert result.selected_task == AnalysisTask.CLASSIFICATION


def test_numeric_ambiguity_requires_confirmation() -> None:
    profile = _profile(
        _column("code", dtype="Int32", unique_count=10, cardinality_ratio=0.05)
    )
    mapping = _mapping(_assignment("code", ColumnRole.TARGET_QUALITY))
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.decision == "CONFIRMATION_REQUIRED"
    assert result.requires_user_confirmation is True


def test_numeric_ambiguity_candidate_order() -> None:
    profile = _profile(
        _column("code", dtype="Int32", unique_count=3, cardinality_ratio=0.03)
    )
    mapping = _mapping(_assignment("code", ColumnRole.TARGET_QUALITY))
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.candidate_tasks == [
        AnalysisTask.CLASSIFICATION,
        AnalysisTask.REGRESSION,
        AnalysisTask.RESIDUAL_ANOMALY,
    ]


def test_temporal_target_requires_confirmation() -> None:
    profile = _profile(_column("event_time", dtype="Datetime"))
    mapping = _mapping(_assignment("event_time", ColumnRole.TARGET_QUALITY))
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.decision == "CONFIRMATION_REQUIRED"
    assert result.selected_task == AnalysisTask.REGRESSION
    assert result.candidate_tasks == [
        AnalysisTask.REGRESSION,
        AnalysisTask.CLASSIFICATION,
    ]
    assert any("temporal" in w.casefold() for w in result.warnings)


def test_unknown_dtype_target_requires_confirmation() -> None:
    profile = _profile(_column("blob", dtype="Object"))
    mapping = _mapping(_assignment("blob", ColumnRole.TARGET_QUALITY))
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.decision == "CONFIRMATION_REQUIRED"
    assert result.selected_task == AnalysisTask.REGRESSION
    assert any("unknown" in w.casefold() or "dtype" in w.casefold() for w in result.warnings)


def test_single_classification_target_auto_selected() -> None:
    profile = _profile(_column("ok", dtype="Boolean", unique_count=2, cardinality_ratio=0.02))
    mapping = _mapping(_assignment("ok", ColumnRole.TARGET_QUALITY))
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.decision == "AUTO_SELECTED"
    assert result.selected_task == AnalysisTask.CLASSIFICATION


def test_single_regression_target_auto_selected() -> None:
    profile = _profile(
        _column("quality", dtype="Float64", unique_count=90, cardinality_ratio=0.9)
    )
    mapping = _mapping(_assignment("quality", ColumnRole.TARGET_QUALITY))
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.decision == "AUTO_SELECTED"
    assert result.selected_task == AnalysisTask.REGRESSION


def test_low_confidence_target_requires_confirmation() -> None:
    profile = _profile(
        _column("quality", dtype="Float64", unique_count=90, cardinality_ratio=0.9)
    )
    mapping = _mapping(
        _assignment("quality", ColumnRole.TARGET_QUALITY, confidence=0.4),
        low_confidence_columns=["quality"],
    )
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.decision == "CONFIRMATION_REQUIRED"
    assert any("confidence" in w.casefold() for w in result.warnings)


def test_insufficient_target_skips_supervised_auto() -> None:
    profile = _profile(
        _column("y", dtype="Float64", null_count=99, unique_count=1, cardinality_ratio=0.01),
        row_count=100,
    )
    mapping = _mapping(_assignment("y", ColumnRole.TARGET_QUALITY))
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.selected_task == AnalysisTask.UNSUPERVISED_ANOMALY
    assert result.decision == "CONFIRMATION_REQUIRED"


def test_insufficient_target_sets_selected_target_none() -> None:
    profile = _profile(
        _column("y", dtype="Float64", null_count=99),
        row_count=100,
    )
    mapping = _mapping(_assignment("y", ColumnRole.TARGET_QUALITY))
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.selected_target is None
    assert "y" in result.target_candidates


def test_insufficient_target_with_time_uses_time_series() -> None:
    profile = _profile(
        _column("ts", dtype="Datetime"),
        _column("y", dtype="Float64", null_count=99),
        row_count=100,
    )
    mapping = _mapping(
        _assignment("ts", ColumnRole.TIME),
        _assignment("y", ColumnRole.TARGET_QUALITY),
    )
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.selected_task == AnalysisTask.TIME_SERIES_ANOMALY
    assert result.selected_time_column == "ts"
    assert result.decision == "CONFIRMATION_REQUIRED"


def test_insufficient_target_without_time_uses_unsupervised() -> None:
    profile = _profile(
        _column("y", dtype="Float64", null_count=99),
        row_count=100,
    )
    mapping = _mapping(_assignment("y", ColumnRole.TARGET_QUALITY))
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.selected_task == AnalysisTask.UNSUPERVISED_ANOMALY
    assert result.selected_time_column is None


def test_multiple_targets_require_confirmation() -> None:
    profile = _profile(
        _column("y1", dtype="Float64", unique_count=80, cardinality_ratio=0.8),
        _column("y2", dtype="Float64", unique_count=70, cardinality_ratio=0.7),
    )
    mapping = _mapping(
        _assignment("y1", ColumnRole.TARGET_QUALITY),
        _assignment("y2", ColumnRole.TARGET_QUALITY),
    )
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.decision == "CONFIRMATION_REQUIRED"
    assert any("multiple" in w.casefold() and "target" in w.casefold() for w in result.warnings)


def test_multiple_targets_use_first_provisional() -> None:
    profile = _profile(
        _column("first", dtype="Float64", unique_count=80, cardinality_ratio=0.8),
        _column("second", dtype="Boolean", unique_count=2, cardinality_ratio=0.02),
    )
    mapping = _mapping(
        _assignment("first", ColumnRole.TARGET_QUALITY),
        _assignment("second", ColumnRole.TARGET_QUALITY),
    )
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.selected_target == "first"
    assert result.selected_task == AnalysisTask.REGRESSION


def test_target_candidate_order_follows_profile() -> None:
    profile = _profile(
        _column("x"),
        _column("y_b", dtype="Float64", unique_count=80, cardinality_ratio=0.8),
        _column("y_a", dtype="Float64", unique_count=70, cardinality_ratio=0.7),
    )
    mapping = _mapping(
        _assignment("x", ColumnRole.STATE_SENSOR),
        _assignment("y_b", ColumnRole.TARGET_QUALITY),
        _assignment("y_a", ColumnRole.TARGET_QUALITY),
    )
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.target_candidates == ["y_b", "y_a"]


# ---------------------------------------------------------------------------
# Automatic anomaly routing
# ---------------------------------------------------------------------------


def test_time_only_selects_time_series_anomaly() -> None:
    profile = _profile(_column("ts", dtype="Datetime"), _column("x"))
    mapping = _mapping(
        _assignment("ts", ColumnRole.TIME),
        _assignment("x", ColumnRole.STATE_SENSOR),
    )
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.selected_task == AnalysisTask.TIME_SERIES_ANOMALY
    assert result.selected_time_column == "ts"
    assert result.decision == "AUTO_SELECTED"


def test_prefer_time_series_false_selects_unsupervised() -> None:
    router = AnalysisTaskRouter(
        policy=TaskRoutingPolicy(prefer_time_series_anomaly=False)
    )
    profile = _profile(_column("ts", dtype="Datetime"), _column("x"))
    mapping = _mapping(
        _assignment("ts", ColumnRole.TIME),
        _assignment("x", ColumnRole.STATE_SENSOR),
    )
    result = router.route(profile, mapping)
    assert result.selected_task == AnalysisTask.UNSUPERVISED_ANOMALY
    assert result.selected_time_column == "ts"
    assert result.decision == "AUTO_SELECTED"


def test_no_target_no_time_selects_unsupervised() -> None:
    profile = _profile(_column("x"), _column("z"))
    mapping = _mapping(
        _assignment("x", ColumnRole.STATE_SENSOR),
        _assignment("z", ColumnRole.CONTEXT),
    )
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.selected_task == AnalysisTask.UNSUPERVISED_ANOMALY
    assert result.selected_target is None
    assert result.selected_time_column is None
    assert result.decision == "AUTO_SELECTED"
    assert "target" in result.reason.casefold() or "time" in result.reason.casefold()


def test_multiple_time_columns_require_confirmation() -> None:
    profile = _profile(
        _column("ts1", dtype="Datetime"),
        _column("ts2", dtype="Datetime"),
        _column("x"),
    )
    mapping = _mapping(
        _assignment("ts1", ColumnRole.TIME),
        _assignment("ts2", ColumnRole.TIME),
        _assignment("x", ColumnRole.STATE_SENSOR),
    )
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.decision == "CONFIRMATION_REQUIRED"
    assert any("multiple" in w.casefold() and "time" in w.casefold() for w in result.warnings)


def test_multiple_time_uses_first_provisional() -> None:
    profile = _profile(
        _column("first_ts", dtype="Datetime"),
        _column("second_ts", dtype="Datetime"),
    )
    mapping = _mapping(
        _assignment("first_ts", ColumnRole.TIME),
        _assignment("second_ts", ColumnRole.TIME),
    )
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.selected_time_column == "first_ts"


def test_time_candidate_order_preserved() -> None:
    profile = _profile(
        _column("b_ts", dtype="Datetime"),
        _column("a_ts", dtype="Datetime"),
    )
    mapping = _mapping(
        _assignment("b_ts", ColumnRole.TIME),
        _assignment("a_ts", ColumnRole.TIME),
    )
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.time_candidates == ["b_ts", "a_ts"]


def test_low_confidence_time_requires_confirmation() -> None:
    profile = _profile(_column("ts", dtype="Datetime"), _column("x"))
    mapping = _mapping(
        _assignment("ts", ColumnRole.TIME, confidence=0.3),
        _assignment("x", ColumnRole.STATE_SENSOR),
        low_confidence_columns=["ts"],
    )
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.decision == "CONFIRMATION_REQUIRED"
    assert any("confidence" in w.casefold() for w in result.warnings)


def test_time_series_candidate_task_order() -> None:
    profile = _profile(_column("ts", dtype="Datetime"))
    mapping = _mapping(_assignment("ts", ColumnRole.TIME))
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.candidate_tasks == [
        AnalysisTask.TIME_SERIES_ANOMALY,
        AnalysisTask.UNSUPERVISED_ANOMALY,
        AnalysisTask.DRIFT_DETECTION,
    ]


def test_unsupervised_preferred_candidate_task_order() -> None:
    router = AnalysisTaskRouter(
        policy=TaskRoutingPolicy(prefer_time_series_anomaly=False)
    )
    profile = _profile(_column("ts", dtype="Datetime"))
    mapping = _mapping(_assignment("ts", ColumnRole.TIME))
    result = router.route(profile, mapping)
    assert result.candidate_tasks == [
        AnalysisTask.UNSUPERVISED_ANOMALY,
        AnalysisTask.TIME_SERIES_ANOMALY,
        AnalysisTask.DRIFT_DETECTION,
    ]


# ---------------------------------------------------------------------------
# User confirmation
# ---------------------------------------------------------------------------


def test_confirmed_regression_success() -> None:
    profile = _profile(
        _column("y", dtype="Float64", unique_count=80, cardinality_ratio=0.8),
        _column("x"),
    )
    mapping = _mapping(
        _assignment("y", ColumnRole.TARGET_QUALITY),
        _assignment("x", ColumnRole.STATE_SENSOR),
    )
    result = AnalysisTaskRouter().route(
        profile,
        mapping,
        confirmed_task=AnalysisTask.REGRESSION,
        confirmed_target="y",
    )
    assert result.decision == "USER_CONFIRMED"
    assert result.selected_task == AnalysisTask.REGRESSION
    assert result.selected_target == "y"
    assert result.user_confirmed is True
    assert result.requires_user_confirmation is False


def test_confirmed_regression_missing_target_rejected() -> None:
    profile = _profile(_column("y", dtype="Float64", unique_count=80, cardinality_ratio=0.8))
    mapping = _mapping(_assignment("y", ColumnRole.TARGET_QUALITY))
    with pytest.raises(DataValidationError):
        AnalysisTaskRouter().route(
            profile,
            mapping,
            confirmed_task=AnalysisTask.REGRESSION,
        )


def test_confirmed_regression_string_target_rejected() -> None:
    profile = _profile(_column("y", dtype="Utf8", unique_count=3, cardinality_ratio=0.03))
    mapping = _mapping(_assignment("y", ColumnRole.TARGET_QUALITY))
    with pytest.raises(DataValidationError):
        AnalysisTaskRouter().route(
            profile,
            mapping,
            confirmed_task=AnalysisTask.REGRESSION,
            confirmed_target="y",
        )


def test_confirmed_classification_success() -> None:
    profile = _profile(_column("flag", dtype="Boolean", unique_count=2, cardinality_ratio=0.02))
    mapping = _mapping(_assignment("flag", ColumnRole.TARGET_QUALITY))
    result = AnalysisTaskRouter().route(
        profile,
        mapping,
        confirmed_task=AnalysisTask.CLASSIFICATION,
        confirmed_target="flag",
    )
    assert result.decision == "USER_CONFIRMED"
    assert result.selected_task == AnalysisTask.CLASSIFICATION


def test_confirmed_classification_missing_target_rejected() -> None:
    profile = _profile(_column("flag", dtype="Boolean"))
    mapping = _mapping(_assignment("flag", ColumnRole.TARGET_QUALITY))
    with pytest.raises(DataValidationError):
        AnalysisTaskRouter().route(
            profile,
            mapping,
            confirmed_task=AnalysisTask.CLASSIFICATION,
        )


def test_confirmed_classification_allows_float_target() -> None:
    profile = _profile(
        _column("score", dtype="Float64", unique_count=50, cardinality_ratio=0.5)
    )
    mapping = _mapping(_assignment("score", ColumnRole.TARGET_QUALITY))
    result = AnalysisTaskRouter().route(
        profile,
        mapping,
        confirmed_task=AnalysisTask.CLASSIFICATION,
        confirmed_target="score",
    )
    assert result.decision == "USER_CONFIRMED"
    assert result.selected_task == AnalysisTask.CLASSIFICATION


def test_confirmed_residual_anomaly_numeric_success() -> None:
    profile = _profile(
        _column("y", dtype="Float64", unique_count=80, cardinality_ratio=0.8)
    )
    mapping = _mapping(_assignment("y", ColumnRole.TARGET_QUALITY))
    result = AnalysisTaskRouter().route(
        profile,
        mapping,
        confirmed_task=AnalysisTask.RESIDUAL_ANOMALY,
        confirmed_target="y",
    )
    assert result.decision == "USER_CONFIRMED"
    assert result.selected_task == AnalysisTask.RESIDUAL_ANOMALY


def test_confirmed_residual_missing_target_rejected() -> None:
    profile = _profile(_column("y", dtype="Float64", unique_count=80, cardinality_ratio=0.8))
    mapping = _mapping(_assignment("y", ColumnRole.TARGET_QUALITY))
    with pytest.raises(DataValidationError):
        AnalysisTaskRouter().route(
            profile,
            mapping,
            confirmed_task=AnalysisTask.RESIDUAL_ANOMALY,
        )


def test_confirmed_residual_string_target_rejected() -> None:
    profile = _profile(_column("y", dtype="Utf8", unique_count=3, cardinality_ratio=0.03))
    mapping = _mapping(_assignment("y", ColumnRole.TARGET_QUALITY))
    with pytest.raises(DataValidationError):
        AnalysisTaskRouter().route(
            profile,
            mapping,
            confirmed_task=AnalysisTask.RESIDUAL_ANOMALY,
            confirmed_target="y",
        )


def test_confirmed_time_series_success() -> None:
    profile = _profile(_column("ts", dtype="Datetime"), _column("x"))
    mapping = _mapping(
        _assignment("ts", ColumnRole.TIME),
        _assignment("x", ColumnRole.STATE_SENSOR),
    )
    result = AnalysisTaskRouter().route(
        profile,
        mapping,
        confirmed_task=AnalysisTask.TIME_SERIES_ANOMALY,
        confirmed_time_column="ts",
    )
    assert result.decision == "USER_CONFIRMED"
    assert result.selected_task == AnalysisTask.TIME_SERIES_ANOMALY
    assert result.selected_time_column == "ts"


def test_confirmed_time_series_missing_time_rejected() -> None:
    profile = _profile(_column("ts", dtype="Datetime"))
    mapping = _mapping(_assignment("ts", ColumnRole.TIME))
    with pytest.raises(DataValidationError):
        AnalysisTaskRouter().route(
            profile,
            mapping,
            confirmed_task=AnalysisTask.TIME_SERIES_ANOMALY,
        )


def test_confirmed_time_series_allows_non_temporal_column() -> None:
    profile = _profile(_column("cycle", dtype="Int64"), _column("x"))
    mapping = _mapping(
        _assignment("cycle", ColumnRole.CONTEXT),
        _assignment("x", ColumnRole.STATE_SENSOR),
    )
    result = AnalysisTaskRouter().route(
        profile,
        mapping,
        confirmed_task=AnalysisTask.TIME_SERIES_ANOMALY,
        confirmed_time_column="cycle",
    )
    assert result.decision == "USER_CONFIRMED"
    assert result.selected_time_column == "cycle"
    assert "cycle" in result.time_candidates


def test_confirmed_drift_success() -> None:
    profile = _profile(_column("ts", dtype="Datetime"))
    mapping = _mapping(_assignment("ts", ColumnRole.TIME))
    result = AnalysisTaskRouter().route(
        profile,
        mapping,
        confirmed_task=AnalysisTask.DRIFT_DETECTION,
        confirmed_time_column="ts",
    )
    assert result.decision == "USER_CONFIRMED"
    assert result.selected_task == AnalysisTask.DRIFT_DETECTION


def test_confirmed_drift_missing_time_rejected() -> None:
    profile = _profile(_column("ts", dtype="Datetime"))
    mapping = _mapping(_assignment("ts", ColumnRole.TIME))
    with pytest.raises(DataValidationError):
        AnalysisTaskRouter().route(
            profile,
            mapping,
            confirmed_task=AnalysisTask.DRIFT_DETECTION,
        )


def test_confirmed_unsupervised_success() -> None:
    profile = _profile(_column("x"), _column("z"))
    mapping = _mapping(
        _assignment("x", ColumnRole.STATE_SENSOR),
        _assignment("z", ColumnRole.CONTEXT),
    )
    result = AnalysisTaskRouter().route(
        profile,
        mapping,
        confirmed_task=AnalysisTask.UNSUPERVISED_ANOMALY,
    )
    assert result.decision == "USER_CONFIRMED"
    assert result.selected_task == AnalysisTask.UNSUPERVISED_ANOMALY


def test_user_confirmed_flags() -> None:
    profile = _profile(_column("y", dtype="Float64", unique_count=80, cardinality_ratio=0.8))
    mapping = _mapping(_assignment("y", ColumnRole.TARGET_QUALITY))
    result = AnalysisTaskRouter().route(
        profile,
        mapping,
        confirmed_task=AnalysisTask.REGRESSION,
        confirmed_target="y",
    )
    assert result.decision == "USER_CONFIRMED"
    assert result.user_confirmed is True
    assert result.requires_user_confirmation is False
    assert "confirm" in result.reason.casefold()


def test_user_can_select_non_primary_target() -> None:
    profile = _profile(
        _column("y1", dtype="Float64", unique_count=80, cardinality_ratio=0.8),
        _column("y2", dtype="Boolean", unique_count=2, cardinality_ratio=0.02),
    )
    mapping = _mapping(
        _assignment("y1", ColumnRole.TARGET_QUALITY),
        _assignment("y2", ColumnRole.TARGET_QUALITY),
    )
    result = AnalysisTaskRouter().route(
        profile,
        mapping,
        confirmed_task=AnalysisTask.CLASSIFICATION,
        confirmed_target="y2",
    )
    assert result.selected_target == "y2"
    assert result.selected_task == AnalysisTask.CLASSIFICATION


def test_user_can_select_non_time_role_as_time() -> None:
    profile = _profile(_column("cycle_index", dtype="Int64"), _column("x"))
    mapping = _mapping(
        _assignment("cycle_index", ColumnRole.CONTEXT),
        _assignment("x", ColumnRole.STATE_SENSOR),
    )
    result = AnalysisTaskRouter().route(
        profile,
        mapping,
        confirmed_task=AnalysisTask.DRIFT_DETECTION,
        confirmed_time_column="cycle_index",
    )
    assert result.selected_time_column == "cycle_index"
    assert "cycle_index" in result.time_candidates


def test_confirmed_target_only_prioritizes_that_target() -> None:
    profile = _profile(
        _column("y1", dtype="Float64", unique_count=80, cardinality_ratio=0.8),
        _column("y2", dtype="Boolean", unique_count=2, cardinality_ratio=0.02),
    )
    mapping = _mapping(
        _assignment("y1", ColumnRole.TARGET_QUALITY),
        _assignment("y2", ColumnRole.TARGET_QUALITY),
    )
    result = AnalysisTaskRouter().route(profile, mapping, confirmed_target="y2")
    assert result.selected_target == "y2"
    assert result.selected_task == AnalysisTask.CLASSIFICATION
    assert result.decision == "AUTO_SELECTED"
    assert not any("multiple" in w.casefold() for w in result.warnings)


def test_confirmed_time_only_prioritizes_that_time() -> None:
    profile = _profile(
        _column("ts1", dtype="Datetime"),
        _column("ts2", dtype="Datetime"),
        _column("x"),
    )
    mapping = _mapping(
        _assignment("ts1", ColumnRole.TIME),
        _assignment("ts2", ColumnRole.TIME),
        _assignment("x", ColumnRole.STATE_SENSOR),
    )
    result = AnalysisTaskRouter().route(
        profile,
        mapping,
        confirmed_time_column="ts2",
    )
    assert result.selected_time_column == "ts2"
    assert result.selected_task == AnalysisTask.TIME_SERIES_ANOMALY
    assert result.decision == "AUTO_SELECTED"
    assert not any("multiple" in w.casefold() for w in result.warnings)


def test_confirmed_supervised_insufficient_target_raises() -> None:
    profile = _profile(
        _column("y", dtype="Float64", null_count=99),
        row_count=100,
    )
    mapping = _mapping(_assignment("y", ColumnRole.TARGET_QUALITY))
    with pytest.raises(InsufficientDataError):
        AnalysisTaskRouter().route(
            profile,
            mapping,
            confirmed_task=AnalysisTask.REGRESSION,
            confirmed_target="y",
        )


# ---------------------------------------------------------------------------
# Candidates and evidence
# ---------------------------------------------------------------------------


def test_candidate_tasks_have_no_duplicates() -> None:
    profile = _profile(
        _column("y", dtype="Float64", unique_count=80, cardinality_ratio=0.8)
    )
    mapping = _mapping(_assignment("y", ColumnRole.TARGET_QUALITY))
    result = AnalysisTaskRouter().route(profile, mapping)
    assert len(result.candidate_tasks) == len(set(result.candidate_tasks))


def test_selected_task_is_first_or_included() -> None:
    profile = _profile(
        _column("y", dtype="Float64", unique_count=80, cardinality_ratio=0.8)
    )
    mapping = _mapping(_assignment("y", ColumnRole.TARGET_QUALITY))
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.selected_task in result.candidate_tasks
    assert result.candidate_tasks[0] == result.selected_task


def test_evidence_not_empty() -> None:
    profile = _profile(_column("x"))
    mapping = _mapping(_assignment("x", ColumnRole.STATE_SENSOR))
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.evidence


def test_multiple_target_warning() -> None:
    profile = _profile(
        _column("y1", dtype="Float64", unique_count=80, cardinality_ratio=0.8),
        _column("y2", dtype="Float64", unique_count=70, cardinality_ratio=0.7),
    )
    mapping = _mapping(
        _assignment("y1", ColumnRole.TARGET_QUALITY),
        _assignment("y2", ColumnRole.TARGET_QUALITY),
    )
    result = AnalysisTaskRouter().route(profile, mapping)
    assert any("target" in w.casefold() for w in result.warnings)


def test_multiple_time_warning() -> None:
    profile = _profile(
        _column("ts1", dtype="Datetime"),
        _column("ts2", dtype="Datetime"),
    )
    mapping = _mapping(
        _assignment("ts1", ColumnRole.TIME),
        _assignment("ts2", ColumnRole.TIME),
    )
    result = AnalysisTaskRouter().route(profile, mapping)
    assert any("time" in w.casefold() for w in result.warnings)


def test_low_confidence_warning() -> None:
    profile = _profile(
        _column("y", dtype="Float64", unique_count=80, cardinality_ratio=0.8)
    )
    mapping = _mapping(
        _assignment("y", ColumnRole.TARGET_QUALITY, confidence=0.2),
        low_confidence_columns=["y"],
    )
    result = AnalysisTaskRouter().route(profile, mapping)
    assert any("confidence" in w.casefold() for w in result.warnings)


def test_insufficient_target_warning() -> None:
    profile = _profile(_column("y", dtype="Float64", null_count=99), row_count=100)
    mapping = _mapping(_assignment("y", ColumnRole.TARGET_QUALITY))
    result = AnalysisTaskRouter().route(profile, mapping)
    assert any(
        "insufficient" in w.casefold() or "non-null" in w.casefold()
        for w in result.warnings
    )


def test_ambiguous_numeric_warning() -> None:
    profile = _profile(
        _column("code", dtype="Int64", unique_count=4, cardinality_ratio=0.04)
    )
    mapping = _mapping(_assignment("code", ColumnRole.TARGET_QUALITY))
    result = AnalysisTaskRouter().route(profile, mapping)
    assert any(
        "ambiguous" in w.casefold() or "cardinality" in w.casefold()
        for w in result.warnings
    )


def test_warnings_have_no_duplicates() -> None:
    profile = _profile(
        _column("code", dtype="Int64", unique_count=4, cardinality_ratio=0.04)
    )
    mapping = _mapping(
        _assignment("code", ColumnRole.TARGET_QUALITY, confidence=0.2),
        low_confidence_columns=["code"],
    )
    result = AnalysisTaskRouter().route(profile, mapping)
    assert len(result.warnings) == len(set(result.warnings))


def test_identical_inputs_are_deterministic() -> None:
    router = AnalysisTaskRouter()
    profile = _profile(
        _column("y", dtype="Float64", unique_count=80, cardinality_ratio=0.8),
        _column("ts", dtype="Datetime"),
    )
    mapping = _mapping(
        _assignment("y", ColumnRole.TARGET_QUALITY),
        _assignment("ts", ColumnRole.TIME),
    )
    assert router.route(profile, mapping).model_dump() == router.route(
        profile, mapping
    ).model_dump()


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


def test_column_role_mapper_result_accepted() -> None:
    profile = _profile(
        _column("yield", dtype="Float64", unique_count=80, cardinality_ratio=0.8),
        _column("rf_power"),
    )
    mapping = ColumnRoleMapper().map_roles(
        profile,
        SemiconductorIndustryProfile().get_schema_hints(),
    )
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.selected_task == AnalysisTask.REGRESSION
    assert result.selected_target == "yield"


def test_semiconductor_yield_routes_to_regression() -> None:
    profile = _profile(
        _column("wafer_id", dtype="Utf8", unique_count=100, cardinality_ratio=1.0),
        _column("timestamp", dtype="Datetime"),
        _column("rf_power"),
        _column("yield", dtype="Float64", unique_count=90, cardinality_ratio=0.9),
    )
    mapping = ColumnRoleMapper().map_roles(
        profile,
        SemiconductorIndustryProfile().get_schema_hints(),
    )
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.selected_task == AnalysisTask.REGRESSION
    assert result.selected_target == "yield"


def test_battery_discharge_capacity_routes_to_regression() -> None:
    profile = _profile(
        _column("cell_id", dtype="Utf8", unique_count=100, cardinality_ratio=1.0),
        _column("cycle_index", dtype="Int64", unique_count=100, cardinality_ratio=1.0),
        _column("charge_current"),
        _column(
            "discharge_capacity",
            dtype="Float64",
            unique_count=90,
            cardinality_ratio=0.9,
        ),
    )
    mapping = ColumnRoleMapper().map_roles(
        profile,
        BatteryIndustryProfile().get_schema_hints(),
    )
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.selected_task == AnalysisTask.REGRESSION
    assert result.selected_target == "discharge_capacity"


def test_automotive_failure_flag_routes_to_classification() -> None:
    profile = _profile(
        _column("vehicle_id", dtype="Utf8", unique_count=100, cardinality_ratio=1.0),
        _column("throttle_position"),
        _column("failure_flag", dtype="Boolean", unique_count=2, cardinality_ratio=0.02),
    )
    mapping = ColumnRoleMapper().map_roles(
        profile,
        AutomotiveIndustryProfile().get_schema_hints(),
    )
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.selected_task == AnalysisTask.CLASSIFICATION
    assert result.selected_target == "failure_flag"


def test_timestamp_only_routes_to_time_series_anomaly() -> None:
    profile = _profile(
        _column("timestamp", dtype="Datetime"),
        _column("sensor_a"),
        _column("sensor_b"),
    )
    mapping = ColumnRoleMapper().map_roles(
        profile,
        SemiconductorIndustryProfile().get_schema_hints(),
    )
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.selected_task == AnalysisTask.TIME_SERIES_ANOMALY
    assert result.selected_time_column == "timestamp"


def test_no_target_no_time_routes_to_unsupervised() -> None:
    profile = _profile(_column("sensor_a"), _column("sensor_b"))
    mapping = ColumnRoleMapper().map_roles(
        profile,
        SemiconductorIndustryProfile().get_schema_hints(),
    )
    result = AnalysisTaskRouter().route(profile, mapping)
    assert result.selected_task == AnalysisTask.UNSUPERVISED_ANOMALY


def test_create_default_task_router_returns_router() -> None:
    router = create_default_task_router()
    assert isinstance(router, AnalysisTaskRouter)


def test_create_default_task_router_independent_instances() -> None:
    left = create_default_task_router()
    right = create_default_task_router()
    assert left is not right


def test_factory_external_policy_mutation_isolated() -> None:
    policy = TaskRoutingPolicy(prefer_time_series_anomaly=True)
    router = create_default_task_router(policy=policy)
    policy.prefer_time_series_anomaly = False
    profile = _profile(_column("ts", dtype="Datetime"))
    mapping = _mapping(_assignment("ts", ColumnRole.TIME))
    result = router.route(profile, mapping)
    assert result.selected_task == AnalysisTask.TIME_SERIES_ANOMALY


# ---------------------------------------------------------------------------
# Immutability and isolation
# ---------------------------------------------------------------------------


def test_route_does_not_mutate_dataset_profile() -> None:
    profile = _profile(
        _column("y", dtype="Float64", unique_count=80, cardinality_ratio=0.8)
    )
    mapping = _mapping(_assignment("y", ColumnRole.TARGET_QUALITY))
    before = deepcopy(profile.model_dump())
    AnalysisTaskRouter().route(profile, mapping)
    assert profile.model_dump() == before


def test_route_does_not_mutate_role_mapping() -> None:
    profile = _profile(
        _column("y", dtype="Float64", unique_count=80, cardinality_ratio=0.8)
    )
    mapping = _mapping(_assignment("y", ColumnRole.TARGET_QUALITY))
    before = deepcopy(mapping.model_dump())
    AnalysisTaskRouter().route(profile, mapping)
    assert mapping.model_dump() == before


def test_consecutive_routes_do_not_accumulate() -> None:
    router = AnalysisTaskRouter()
    first = router.route(
        _profile(_column("y", dtype="Float64", unique_count=80, cardinality_ratio=0.8)),
        _mapping(_assignment("y", ColumnRole.TARGET_QUALITY)),
    )
    second = router.route(
        _profile(_column("ts", dtype="Datetime")),
        _mapping(_assignment("ts", ColumnRole.TIME)),
    )
    assert first.selected_task == AnalysisTask.REGRESSION
    assert second.selected_task == AnalysisTask.TIME_SERIES_ANOMALY
    assert first.target_candidates == ["y"]
    assert second.target_candidates == []


def test_separate_router_instances_do_not_share_state() -> None:
    left = AnalysisTaskRouter(
        policy=TaskRoutingPolicy(prefer_time_series_anomaly=True)
    )
    right = AnalysisTaskRouter(
        policy=TaskRoutingPolicy(prefer_time_series_anomaly=False)
    )
    profile = _profile(_column("ts", dtype="Datetime"))
    mapping = _mapping(_assignment("ts", ColumnRole.TIME))
    assert left.route(profile, mapping).selected_task == AnalysisTask.TIME_SERIES_ANOMALY
    assert right.route(profile, mapping).selected_task == AnalysisTask.UNSUPERVISED_ANOMALY
