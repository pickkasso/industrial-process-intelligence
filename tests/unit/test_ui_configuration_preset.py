"""Unit tests for analysis configuration preset import/export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from process_intelligence.core.enums import AnalysisTask, ColumnRole
from process_intelligence.evaluation import MetricAcceptanceDirection
from process_intelligence.recommendation import (
    QualityOptimizationDirection,
    RecommendationObjective,
)
from process_intelligence.ui.configuration_preset import (
    SESSION_ANOMALY_RECOMMENDATION_ENABLED_KEY,
    SESSION_OBJECTIVE_KEY,
    SESSION_PRESET_APPLY_SUMMARY_KEY,
    SESSION_QUALITY_DIRECTION_KEY,
    SESSION_REQUESTED_TASK_KEY,
    SESSION_RULE_COUNT_KEY,
    SESSION_TARGET_KEY,
    WorkflowUiConfigurationPreset,
    apply_configuration_preset_to_session_state,
    build_configuration_preset,
    build_exportable_configuration_preset,
    check_configuration_preset_column_compatibility,
    collect_referenced_columns,
    configuration_preset_to_json,
    parse_configuration_preset_json,
)
from process_intelligence.ui.schemas import (
    UiMetricRuleInput,
    UiVariableConstraintInput,
)
from process_intelligence.workflow import (
    AnalysisExecutionMode,
    NumericCohortFilter,
    OperatingPointSelectionMode,
)

_COLUMNS = (
    "timestamp",
    "lot_id",
    "pressure",
    "temperature",
    "quality",
    "notes",
)


def _rule(**overrides: Any) -> UiMetricRuleInput:
    payload: dict[str, Any] = {
        "metric_name": "rmse",
        "direction": MetricAcceptanceDirection.LOWER_IS_BETTER,
        "threshold": 10.0,
        "required": True,
    }
    payload.update(overrides)
    return UiMetricRuleInput(**payload)


def _constraint(**overrides: Any) -> UiVariableConstraintInput:
    payload: dict[str, Any] = {
        "variable": "pressure",
        "minimum": 0.0,
        "maximum": 100.0,
    }
    payload.update(overrides)
    return UiVariableConstraintInput(**payload)


def _supervised_preset(**overrides: Any) -> WorkflowUiConfigurationPreset:
    payload: dict[str, Any] = {
        "analysis_mode": AnalysisExecutionMode.SUPERVISED,
        "requested_task": AnalysisTask.REGRESSION,
        "target_column": "quality",
        "use_recommended_numeric_feature_set": True,
        "explicit_feature_columns": [],
        "timestamp_column": "timestamp",
        "identifier_columns": ["lot_id"],
        "excluded_columns": ["notes"],
        "role_overrides": {"pressure": ColumnRole.CONTROLLABLE_PROCESS},
        "recommendation_objective": RecommendationObjective.REDUCE_ANOMALY_SCORE,
        "quality_direction": None,
        "quality_target": None,
        "performance_rules": [_rule()],
        "cohort_filter": None,
        "anomaly_recommendation_enabled": False,
        "recommendation_constraints": [_constraint()],
        "confirmed_controllable_variables": ["pressure"],
        "verified_variables": ["pressure"],
        "maximum_simultaneous_changes": 2,
        "operating_point_selection_mode": (
            OperatingPointSelectionMode.TOP_RESIDUAL_ANOMALY
        ),
        "explicit_operating_row_id": None,
    }
    payload.update(overrides)
    return WorkflowUiConfigurationPreset(**payload)


def _anomaly_preset(**overrides: Any) -> WorkflowUiConfigurationPreset:
    payload: dict[str, Any] = {
        "analysis_mode": AnalysisExecutionMode.ANOMALY_ONLY,
        "requested_task": None,
        "target_column": None,
        "use_recommended_numeric_feature_set": True,
        "explicit_feature_columns": [],
        "timestamp_column": "timestamp",
        "identifier_columns": ["lot_id"],
        "excluded_columns": ["notes"],
        "role_overrides": {},
        "recommendation_objective": None,
        "quality_direction": None,
        "quality_target": None,
        "performance_rules": [],
        "cohort_filter": None,
        "anomaly_recommendation_enabled": False,
        "recommendation_constraints": [],
        "confirmed_controllable_variables": [],
        "verified_variables": [],
        "maximum_simultaneous_changes": 2,
        "operating_point_selection_mode": (
            OperatingPointSelectionMode.TOP_UNSUPERVISED_ANOMALY
        ),
        "explicit_operating_row_id": None,
    }
    payload.update(overrides)
    return WorkflowUiConfigurationPreset(**payload)


def test_preset_json_round_trip() -> None:
    preset = _supervised_preset(
        recommendation_objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        quality_direction=QualityOptimizationDirection.MAXIMIZE,
    )
    text = configuration_preset_to_json(preset)
    result = parse_configuration_preset_json(text)
    assert result.ok is True
    assert result.preset is not None
    assert result.preset == preset


def test_deterministic_serialization() -> None:
    preset = _supervised_preset()
    first = configuration_preset_to_json(preset)
    second = configuration_preset_to_json(preset)
    assert first == second
    loaded = json.loads(first)
    assert list(loaded.keys()) == sorted(loaded.keys())


def test_optional_missing_ui_selection_round_trip() -> None:
    preset = build_configuration_preset(
        analysis_mode=AnalysisExecutionMode.SUPERVISED,
        requested_task=None,
        target_column=None,
        use_recommended_numeric_feature_set=True,
        explicit_feature_columns=[],
        timestamp_column=None,
        identifier_columns=[],
        excluded_columns=[],
        role_overrides={},
        recommendation_objective=None,
        quality_direction=None,
        quality_target=None,
        performance_rules=[],
        cohort_filter=None,
        anomaly_recommendation_enabled=False,
        recommendation_constraints=[],
        confirmed_controllable_variables=[],
        verified_variables=[],
        maximum_simultaneous_changes=1,
        operating_point_selection_mode=OperatingPointSelectionMode.LATEST_ROW,
        explicit_operating_row_id=None,
    )
    restored = parse_configuration_preset_json(configuration_preset_to_json(preset))
    assert restored.ok is True
    assert restored.preset is not None
    assert restored.preset.target_column is None
    assert restored.preset.recommendation_objective is None
    assert restored.preset.performance_rules == []
    assert restored.preset.role_overrides == {}


def test_forbidden_runtime_objects_rejected() -> None:
    frame = pd.DataFrame({"pressure": [1.0]})
    with pytest.raises((ValidationError, ValueError, TypeError)):
        WorkflowUiConfigurationPreset(
            analysis_mode=AnalysisExecutionMode.SUPERVISED,
            identifier_columns=frame,  # type: ignore[arg-type]
        )
    with pytest.raises((ValidationError, ValueError, TypeError)):
        WorkflowUiConfigurationPreset.model_validate(
            {
                "schema_version": 1,
                "analysis_mode": AnalysisExecutionMode.SUPERVISED,
                "role_overrides": {Path("secret.csv"): ColumnRole.STATE_SENSOR},
            }
        )
    with pytest.raises((ValidationError, ValueError, TypeError)):
        WorkflowUiConfigurationPreset.model_validate(
            {
                "schema_version": 1,
                "analysis_mode": AnalysisExecutionMode.SUPERVISED,
                "explicit_feature_columns": [{"nested": "object"}],
            }
        )
    with pytest.raises((ValidationError, ValueError, TypeError)):
        WorkflowUiConfigurationPreset.model_validate(
            {
                "schema_version": 1,
                "analysis_mode": AnalysisExecutionMode.SUPERVISED,
                "csv_bytes": b"pressure,temperature\n1,2\n",
            }
        )
    preset = _supervised_preset()
    dumped = preset.model_dump(mode="json")
    assert isinstance(dumped, dict)
    assert b"raw" not in dumped.values()
    assert all(
        value is None or isinstance(value, (str, int, float, bool, list, dict))
        for value in dumped.values()
    )


def test_valid_schema_version_one() -> None:
    preset = _supervised_preset()
    assert preset.schema_version == 1
    dumped = preset.model_dump(mode="json")
    assert dumped["schema_version"] == 1


def test_unsupported_schema_version_rejection() -> None:
    result = parse_configuration_preset_json(
        json.dumps({"schema_version": 2, "analysis_mode": "SUPERVISED"})
    )
    assert result.ok is False
    assert result.preset is None
    assert result.error_message is not None
    assert "schema_version" in result.error_message


def test_invalid_json_rejection() -> None:
    result = parse_configuration_preset_json("{not-json")
    assert result.ok is False
    assert result.preset is None
    assert result.error_message is not None
    assert "JSON parsing failed" in result.error_message

    missing_version = parse_configuration_preset_json(
        json.dumps({"analysis_mode": "SUPERVISED"})
    )
    assert missing_version.ok is False
    assert missing_version.error_message is not None
    assert "schema_version" in missing_version.error_message


def test_same_column_schema_compatibility_success() -> None:
    preset = _supervised_preset()
    result = check_configuration_preset_column_compatibility(
        preset,
        available_columns=_COLUMNS,
    )
    assert result.compatible is True
    assert result.missing_columns == ()


def test_extra_current_csv_columns_allowed() -> None:
    preset = _supervised_preset()
    result = check_configuration_preset_column_compatibility(
        preset,
        available_columns=[*_COLUMNS, "extra_sensor", "batch_id"],
    )
    assert result.compatible is True
    assert result.missing_columns == ()


def test_missing_referenced_column_rejection() -> None:
    preset = _supervised_preset(target_column="missing_quality")
    result = check_configuration_preset_column_compatibility(
        preset,
        available_columns=_COLUMNS,
    )
    assert result.compatible is False
    assert "missing_quality" in result.missing_columns


def test_missing_columns_are_sorted() -> None:
    preset = _supervised_preset(
        target_column="zeta_missing",
        identifier_columns=["alpha_missing"],
        excluded_columns=["mu_missing"],
        role_overrides={"beta_missing": ColumnRole.STATE_SENSOR},
    )
    result = check_configuration_preset_column_compatibility(
        preset,
        available_columns=["pressure", "temperature", "timestamp", "notes", "lot_id"],
    )
    assert result.compatible is False
    assert result.missing_columns == tuple(sorted(result.missing_columns))
    assert result.missing_columns == (
        "alpha_missing",
        "beta_missing",
        "mu_missing",
        "zeta_missing",
    )
    assert collect_referenced_columns(preset) == tuple(
        sorted(collect_referenced_columns(preset))
    )


def test_invalid_preset_does_not_mutate_existing_state() -> None:
    state: dict[str, Any] = {
        SESSION_TARGET_KEY: "quality",
        SESSION_OBJECTIVE_KEY: RecommendationObjective.REDUCE_ANOMALY_SCORE.value,
        "marker": "keep-me",
    }
    before = dict(state)
    result = apply_configuration_preset_to_session_state(
        state,
        preset=_supervised_preset(target_column="absent_target"),
        available_columns=_COLUMNS,
    )
    assert result.applied is False
    assert state == before


def test_valid_preset_atomic_apply() -> None:
    state: dict[str, Any] = {
        SESSION_TARGET_KEY: "(select target)",
        SESSION_OBJECTIVE_KEY: "(select objective)",
    }
    preset = _supervised_preset(
        recommendation_objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        quality_direction=QualityOptimizationDirection.MINIMIZE,
    )
    result = apply_configuration_preset_to_session_state(
        state,
        preset=preset,
        available_columns=_COLUMNS,
    )
    assert result.applied is True
    assert state[SESSION_TARGET_KEY] == "quality"
    assert (
        state[SESSION_OBJECTIVE_KEY]
        == RecommendationObjective.IMPROVE_PREDICTED_QUALITY.value
    )
    assert (
        state[SESSION_QUALITY_DIRECTION_KEY]
        == QualityOptimizationDirection.MINIMIZE.value
    )
    assert state[SESSION_RULE_COUNT_KEY] == 1
    assert state["metric_name_0"] == "rmse"
    assert state[SESSION_PRESET_APPLY_SUMMARY_KEY]["analysis_mode"] == "SUPERVISED"


def test_anomaly_only_apply_clears_supervised_stale_state() -> None:
    state: dict[str, Any] = {
        SESSION_TARGET_KEY: "quality",
        SESSION_REQUESTED_TASK_KEY: AnalysisTask.REGRESSION.value,
        SESSION_OBJECTIVE_KEY: RecommendationObjective.REDUCE_ANOMALY_SCORE.value,
        SESSION_QUALITY_DIRECTION_KEY: QualityOptimizationDirection.MAXIMIZE.value,
        SESSION_RULE_COUNT_KEY: 2,
        "metric_name_0": "rmse",
        "metric_direction_0": MetricAcceptanceDirection.LOWER_IS_BETTER.value,
        "metric_threshold_0": "1.0",
        "metric_required_0": True,
        SESSION_ANOMALY_RECOMMENDATION_ENABLED_KEY: True,
    }
    result = apply_configuration_preset_to_session_state(
        state,
        preset=_anomaly_preset(),
        available_columns=_COLUMNS,
    )
    assert result.applied is True
    assert state[SESSION_TARGET_KEY] == "(select target)"
    assert state[SESSION_REQUESTED_TASK_KEY] == "AUTO"
    assert state[SESSION_OBJECTIVE_KEY] == "(select objective)"
    assert state[SESSION_QUALITY_DIRECTION_KEY] == "(select quality direction)"
    assert state[SESSION_RULE_COUNT_KEY] == 1
    assert "metric_name_0" not in state
    assert state[SESSION_ANOMALY_RECOMMENDATION_ENABLED_KEY] is False


def test_supervised_apply_clears_anomaly_only_stale_state() -> None:
    state: dict[str, Any] = {
        "ui_restrict_operating_cohort": True,
        "ui_cohort_filter_column": "pressure",
        "ui_cohort_filter_lower": "10",
        "ui_cohort_filter_upper": "50",
        SESSION_ANOMALY_RECOMMENDATION_ENABLED_KEY: True,
        "anomaly_recommendation_review_variables": ["pressure"],
        "anomaly_rec_confirmed_controllable": ["pressure"],
        "anomaly_rec_verified_variables": ["pressure"],
        "anomaly_rec_constrained_variables": ["pressure"],
        "anomaly_rec_constraint_min_pressure": "1",
        "anomaly_rec_constraint_max_pressure": "2",
        "anomaly_rec_role_pressure": ColumnRole.CONTROLLABLE_PROCESS.value,
    }
    result = apply_configuration_preset_to_session_state(
        state,
        preset=_supervised_preset(),
        available_columns=_COLUMNS,
    )
    assert result.applied is True
    assert state["ui_restrict_operating_cohort"] is False
    assert state[SESSION_ANOMALY_RECOMMENDATION_ENABLED_KEY] is False
    assert state["anomaly_recommendation_review_variables"] == []
    assert state["anomaly_rec_confirmed_controllable"] == []
    assert "anomaly_rec_constraint_min_pressure" not in state
    assert "anomaly_rec_role_pressure" not in state
    assert state[SESSION_TARGET_KEY] == "quality"


def test_nan_inf_rejection() -> None:
    with pytest.raises(ValidationError):
        WorkflowUiConfigurationPreset(
            analysis_mode=AnalysisExecutionMode.SUPERVISED,
            quality_target=float("nan"),
        )
    with pytest.raises(ValidationError):
        WorkflowUiConfigurationPreset(
            analysis_mode=AnalysisExecutionMode.SUPERVISED,
            quality_target=float("inf"),
        )
    with pytest.raises(ValidationError):
        WorkflowUiConfigurationPreset(
            analysis_mode=AnalysisExecutionMode.SUPERVISED,
            performance_rules=[
                UiMetricRuleInput(
                    metric_name="rmse",
                    direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
                    threshold=float("nan"),
                    required=True,
                )
            ],
        )


def test_nan_inf_rejection_via_json_constants() -> None:
    nan_text = (
        '{"schema_version": 1, "analysis_mode": "SUPERVISED", '
        '"quality_target": NaN}'
    )
    inf_text = (
        '{"schema_version": 1, "analysis_mode": "SUPERVISED", '
        '"quality_target": Infinity}'
    )
    for text in (nan_text, inf_text):
        result = parse_configuration_preset_json(text)
        assert result.ok is False
        assert result.preset is None


def test_build_configuration_preset_preserves_cohort_filter() -> None:
    preset = build_configuration_preset(
        analysis_mode=AnalysisExecutionMode.ANOMALY_ONLY,
        cohort_filter=NumericCohortFilter(
            column_name="pressure",
            lower_bound=10.0,
            upper_bound=50.0,
        ),
        maximum_simultaneous_changes=1,
    )
    assert preset.cohort_filter is not None
    assert preset.cohort_filter.column_name == "pressure"


def _rule_draft(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "metric_name": "rmse",
        "direction": MetricAcceptanceDirection.LOWER_IS_BETTER,
        "threshold": 10.0,
        "required": True,
    }
    payload.update(overrides)
    return payload


def _constraint_draft(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "variable": "pressure",
        "minimum": 0.0,
        "maximum": 100.0,
    }
    payload.update(overrides)
    return payload


def _exportable_kwargs(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "analysis_mode": AnalysisExecutionMode.SUPERVISED,
        "requested_task": AnalysisTask.REGRESSION,
        "target_column": "quality",
        "use_recommended_numeric_feature_set": True,
        "explicit_feature_columns": [],
        "timestamp_column": "timestamp",
        "identifier_columns": ["lot_id"],
        "excluded_columns": ["notes"],
        "role_overrides": {"pressure": ColumnRole.CONTROLLABLE_PROCESS},
        "recommendation_objective": RecommendationObjective.REDUCE_ANOMALY_SCORE,
        "quality_direction": None,
        "quality_target": None,
        "performance_rule_drafts": [_rule_draft()],
        "cohort_filter": None,
        "anomaly_recommendation_enabled": False,
        "recommendation_constraint_drafts": [_constraint_draft()],
        "confirmed_controllable_variables": ["pressure"],
        "verified_variables": ["pressure"],
        "maximum_simultaneous_changes": 2,
        "operating_point_selection_mode": (
            OperatingPointSelectionMode.TOP_RESIDUAL_ANOMALY
        ),
        "explicit_operating_row_id": None,
    }
    payload.update(overrides)
    return payload


def test_exportable_when_all_values_valid() -> None:
    result = build_exportable_configuration_preset(**_exportable_kwargs())
    assert result.is_exportable is True
    assert result.preset is not None
    assert result.issues == ()
    assert len(result.preset.performance_rules) == 1
    assert len(result.preset.recommendation_constraints) == 1


def test_empty_optional_performance_row_allowed() -> None:
    result = build_exportable_configuration_preset(
        **_exportable_kwargs(
            performance_rule_drafts=[
                {
                    "metric_name": "",
                    "direction": None,
                    "threshold": "",
                    "required": True,
                }
            ],
            recommendation_constraint_drafts=[],
        )
    )
    assert result.is_exportable is True
    assert result.preset is not None
    assert result.preset.performance_rules == []
    assert result.issues == ()


def test_partial_performance_row_blocks_export() -> None:
    result = build_exportable_configuration_preset(
        **_exportable_kwargs(
            performance_rule_drafts=[
                _rule_draft(direction=None, threshold=""),
            ]
        )
    )
    assert result.is_exportable is False
    assert result.preset is None
    assert len(result.issues) == 1
    assert "Performance rule 1 is incomplete" in result.issues[0]
    assert "direction" in result.issues[0]
    assert "threshold" in result.issues[0]


def test_metric_only_rule_blocks_export() -> None:
    result = build_exportable_configuration_preset(
        **_exportable_kwargs(
            performance_rule_drafts=[
                _rule_draft(direction=None, threshold=""),
            ]
        )
    )
    assert result.is_exportable is False
    assert "direction and threshold are required" in result.issues[0]


def test_direction_only_rule_blocks_export() -> None:
    result = build_exportable_configuration_preset(
        **_exportable_kwargs(
            performance_rule_drafts=[
                _rule_draft(metric_name="", threshold=""),
            ]
        )
    )
    assert result.is_exportable is False
    assert "metric name and threshold are required" in result.issues[0]


def test_threshold_only_rule_blocks_export() -> None:
    result = build_exportable_configuration_preset(
        **_exportable_kwargs(
            performance_rule_drafts=[
                _rule_draft(metric_name="", direction=None),
            ]
        )
    )
    assert result.is_exportable is False
    assert "metric name and direction are required" in result.issues[0]


def test_non_finite_threshold_blocks_export() -> None:
    result = build_exportable_configuration_preset(
        **_exportable_kwargs(
            performance_rule_drafts=[
                _rule_draft(threshold=float("nan")),
            ]
        )
    )
    assert result.is_exportable is False
    assert result.preset is None
    assert result.issues == (
        "Threshold for performance rule 1 must be finite.",
    )


def test_complete_constraint_included() -> None:
    result = build_exportable_configuration_preset(
        **_exportable_kwargs(
            recommendation_constraint_drafts=[
                _constraint_draft(variable="temperature", minimum=1.0, maximum=2.0),
            ]
        )
    )
    assert result.is_exportable is True
    assert result.preset is not None
    assert len(result.preset.recommendation_constraints) == 1
    assert result.preset.recommendation_constraints[0].variable == "temperature"


def test_constraint_column_only_selected_blocks_export() -> None:
    result = build_exportable_configuration_preset(
        **_exportable_kwargs(
            recommendation_constraint_drafts=[
                _constraint_draft(minimum=None, maximum=None),
            ]
        )
    )
    assert result.is_exportable is False
    assert result.preset is None
    assert result.issues == (
        "Constraint for 'pressure' is incomplete: "
        "lower and upper bounds are required.",
    )


def test_lower_only_constraint_blocks_export() -> None:
    result = build_exportable_configuration_preset(
        **_exportable_kwargs(
            recommendation_constraint_drafts=[
                _constraint_draft(maximum=None),
            ]
        )
    )
    assert result.is_exportable is False
    assert result.issues == (
        "Constraint for 'pressure' is incomplete: upper bound is required.",
    )


def test_upper_only_constraint_blocks_export() -> None:
    result = build_exportable_configuration_preset(
        **_exportable_kwargs(
            recommendation_constraint_drafts=[
                _constraint_draft(minimum=None),
            ]
        )
    )
    assert result.is_exportable is False
    assert result.issues == (
        "Constraint for 'pressure' is incomplete: lower bound is required.",
    )


def test_lower_greater_than_upper_blocks_export() -> None:
    result = build_exportable_configuration_preset(
        **_exportable_kwargs(
            recommendation_constraint_drafts=[
                _constraint_draft(minimum=10.0, maximum=1.0),
            ]
        )
    )
    assert result.is_exportable is False
    assert result.issues == (
        "Constraint for 'pressure' is invalid: "
        "lower bound must not exceed upper bound.",
    )


def test_non_finite_constraint_blocks_export() -> None:
    result = build_exportable_configuration_preset(
        **_exportable_kwargs(
            recommendation_constraint_drafts=[
                _constraint_draft(minimum=float("inf"), maximum=100.0),
            ]
        )
    )
    assert result.is_exportable is False
    assert result.issues == (
        "Constraint for 'pressure' is invalid: bounds must be finite.",
    )


def test_multiple_issues_are_deterministic_ordered() -> None:
    result = build_exportable_configuration_preset(
        **_exportable_kwargs(
            performance_rule_drafts=[
                _rule_draft(metric_name="rmse", direction=None, threshold=""),
                _rule_draft(
                    metric_name="mae",
                    direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
                    threshold=float("nan"),
                ),
            ],
            recommendation_constraint_drafts=[
                _constraint_draft(variable="Current", minimum=None, maximum=None),
                _constraint_draft(variable="Power", minimum=5.0, maximum=1.0),
            ],
        )
    )
    assert result.is_exportable is False
    assert result.preset is None
    assert result.issues == (
        "Performance rule 1 is incomplete: direction and threshold are required.",
        "Threshold for performance rule 2 must be finite.",
        "Constraint for 'Current' is incomplete: "
        "lower and upper bounds are required.",
        "Constraint for 'Power' is invalid: "
        "lower bound must not exceed upper bound.",
    )


def test_invalid_export_build_does_not_mutate_input_state() -> None:
    rule_drafts = [
        _rule_draft(direction=None, threshold=""),
    ]
    constraint_drafts = [
        _constraint_draft(minimum=None, maximum=None),
    ]
    identifiers = ["lot_id"]
    features = ["pressure"]
    before_rules = [dict(item) for item in rule_drafts]
    before_constraints = [dict(item) for item in constraint_drafts]
    before_identifiers = list(identifiers)
    before_features = list(features)

    result = build_exportable_configuration_preset(
        **_exportable_kwargs(
            performance_rule_drafts=rule_drafts,
            recommendation_constraint_drafts=constraint_drafts,
            identifier_columns=identifiers,
            confirmed_controllable_variables=features,
        )
    )
    assert result.is_exportable is False
    assert rule_drafts == before_rules
    assert constraint_drafts == before_constraints
    assert identifiers == before_identifiers
    assert features == before_features


def test_valid_export_preset_json_is_deterministic() -> None:
    result = build_exportable_configuration_preset(**_exportable_kwargs())
    assert result.is_exportable is True
    assert result.preset is not None
    first = configuration_preset_to_json(result.preset)
    second = configuration_preset_to_json(result.preset)
    assert first == second
    loaded = json.loads(first)
    assert list(loaded.keys()) == sorted(loaded.keys())
    assert loaded["performance_rules"][0]["metric_name"] == "rmse"
    assert loaded["recommendation_constraints"][0]["variable"] == "pressure"


def test_export_import_round_trip_regression() -> None:
    result = build_exportable_configuration_preset(**_exportable_kwargs())
    assert result.is_exportable is True
    assert result.preset is not None
    restored = parse_configuration_preset_json(
        configuration_preset_to_json(result.preset)
    )
    assert restored.ok is True
    assert restored.preset == result.preset


def test_export_allows_unset_objective_and_auto_task() -> None:
    result = build_exportable_configuration_preset(
        **_exportable_kwargs(
            recommendation_objective=None,
            requested_task=None,
            performance_rule_drafts=[],
        )
    )
    assert result.is_exportable is True
    assert result.preset is not None
    assert result.preset.recommendation_objective is None
    assert result.preset.requested_task is None


def test_export_omits_disabled_cohort_filter() -> None:
    result = build_exportable_configuration_preset(
        **_exportable_kwargs(
            analysis_mode=AnalysisExecutionMode.ANOMALY_ONLY,
            recommendation_objective=None,
            performance_rule_drafts=[],
            cohort_filter=None,
            operating_cohort_restricted=False,
        )
    )
    assert result.is_exportable is True
    assert result.preset is not None
    assert result.preset.cohort_filter is None


def test_export_blocks_incomplete_enabled_cohort_filter() -> None:
    result = build_exportable_configuration_preset(
        **_exportable_kwargs(
            analysis_mode=AnalysisExecutionMode.ANOMALY_ONLY,
            recommendation_objective=None,
            performance_rule_drafts=[],
            cohort_filter=None,
            operating_cohort_restricted=True,
        )
    )
    assert result.is_exportable is False
    assert result.preset is None
    assert any("Operating cohort filter is incomplete" in issue for issue in result.issues)


def test_export_includes_complete_enabled_cohort_filter() -> None:
    cohort = NumericCohortFilter(
        column_name="pressure",
        lower_bound=10.0,
        upper_bound=50.0,
        include_lower=True,
        include_upper=True,
        exclude_filter_column_from_features=True,
    )
    result = build_exportable_configuration_preset(
        **_exportable_kwargs(
            analysis_mode=AnalysisExecutionMode.ANOMALY_ONLY,
            recommendation_objective=None,
            performance_rule_drafts=[],
            cohort_filter=cohort,
            operating_cohort_restricted=True,
        )
    )
    assert result.is_exportable is True
    assert result.preset is not None
    assert result.preset.cohort_filter is not None
    assert result.preset.cohort_filter.column_name == "pressure"

