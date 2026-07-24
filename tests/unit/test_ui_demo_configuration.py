"""Unit tests for Streamlit built-in demo configuration helpers."""

from __future__ import annotations

from process_intelligence.core.enums import AnalysisTask, ColumnRole
from process_intelligence.demo_data import (
    DEMO_CONTROLLABLE_COLUMNS,
    DEMO_IDENTITY_COLUMNS,
    DEMO_MEASURED_COLUMNS,
    DEMO_QUALITY_COLUMNS,
    DEMO_SETPOINT_ENGINEERING_BOUNDS,
    GROUND_TRUTH_METADATA_COLUMNS,
    DemoDatasetConfiguration,
)
from process_intelligence.demo_validation import default_demo_feature_columns
from process_intelligence.ui.configuration_preset import (
    SESSION_CONFIRMED_CONTROLLABLE_KEY,
    SESSION_CONSTRAINT_VARIABLES_KEY,
    SESSION_VERIFIED_VARIABLES_KEY,
    build_configuration_preset,
)
from process_intelligence.ui.demo_configuration import (
    DATA_SOURCE_BUILTIN_DEMO,
    DATA_SOURCE_UPLOAD_CSV,
    FORBIDDEN_DEMO_MODEL_FEATURE_COLUMNS,
    DemoAnalysisTemplate,
    build_demo_configuration_preset,
    build_demo_dataset_ui_summary,
    clear_demo_recommendation_session_state,
    demo_controllable_variable_specs,
    demo_dataset_to_workflow_csv_bytes,
    demo_recommendation_constraints,
    find_forbidden_demo_feature_columns,
    load_builtin_demo_dataset,
    validate_demo_feature_guard,
    validate_demo_recommendation_constraints,
)
from process_intelligence.workflow.enums import (
    AnalysisExecutionMode,
    OperatingPointSelectionMode,
)

_EXPECTED_FEATURES = tuple(default_demo_feature_columns())


def test_supervised_template_uses_exact_process_features() -> None:
    preset = build_demo_configuration_preset(
        DemoAnalysisTemplate.SUPERVISED_QUALITY_PREDICTION
    )
    assert preset.explicit_feature_columns == list(_EXPECTED_FEATURES)
    assert len(preset.explicit_feature_columns) == 10
    assert preset.use_recommended_numeric_feature_set is False


def test_anomaly_only_template_uses_exact_process_features() -> None:
    preset = build_demo_configuration_preset(
        DemoAnalysisTemplate.ANOMALY_ONLY_PROCESS_MONITORING
    )
    assert preset.explicit_feature_columns == list(_EXPECTED_FEATURES)
    assert len(preset.explicit_feature_columns) == 10
    assert preset.use_recommended_numeric_feature_set is False


def test_supervised_template_target_and_task() -> None:
    preset = build_demo_configuration_preset(
        DemoAnalysisTemplate.SUPERVISED_QUALITY_PREDICTION
    )
    assert preset.analysis_mode is AnalysisExecutionMode.SUPERVISED
    assert preset.target_column == "quality_score"
    assert preset.requested_task is AnalysisTask.REGRESSION
    assert preset.timestamp_column == "timestamp"


def test_anomaly_only_template_has_no_target() -> None:
    preset = build_demo_configuration_preset(
        DemoAnalysisTemplate.ANOMALY_ONLY_PROCESS_MONITORING
    )
    assert preset.analysis_mode is AnalysisExecutionMode.ANOMALY_ONLY
    assert preset.target_column is None
    assert preset.requested_task is None
    assert preset.timestamp_column == "timestamp"
    assert preset.performance_rules == []
    assert preset.recommendation_objective is None


def test_templates_exclude_identity_quality_and_ground_truth_from_features() -> None:
    forbidden = set(FORBIDDEN_DEMO_MODEL_FEATURE_COLUMNS)
    for template in DemoAnalysisTemplate:
        preset = build_demo_configuration_preset(template)
        assert forbidden.isdisjoint(preset.explicit_feature_columns)
        # Identity / quality / ground-truth are kept out of features via role
        # fields that satisfy existing UI contracts.
        assert "timestamp" not in preset.explicit_feature_columns
        assert "batch_id" in preset.identifier_columns
        assert "equipment_id" in preset.identifier_columns
        for name in GROUND_TRUTH_METADATA_COLUMNS:
            assert name in preset.excluded_columns
        for name in DEMO_QUALITY_COLUMNS:
            if preset.target_column == name:
                assert name not in preset.excluded_columns
                assert name not in preset.explicit_feature_columns
            else:
                assert name in preset.excluded_columns
        for name in DEMO_IDENTITY_COLUMNS:
            assert name not in preset.explicit_feature_columns


def test_operating_point_modes_per_template() -> None:
    supervised = build_demo_configuration_preset(
        DemoAnalysisTemplate.SUPERVISED_QUALITY_PREDICTION
    )
    anomaly = build_demo_configuration_preset(
        DemoAnalysisTemplate.ANOMALY_ONLY_PROCESS_MONITORING
    )
    assert (
        supervised.operating_point_selection_mode
        is OperatingPointSelectionMode.TOP_RESIDUAL_ANOMALY
    )
    assert (
        anomaly.operating_point_selection_mode
        is OperatingPointSelectionMode.TOP_UNSUPERVISED_ANOMALY
    )


def test_anomaly_recommendation_and_cohort_disabled_for_both_templates() -> None:
    for template in DemoAnalysisTemplate:
        preset = build_demo_configuration_preset(template)
        assert preset.anomaly_recommendation_enabled is False
        assert preset.cohort_filter is None


def test_exactly_four_demo_controllable_setpoints() -> None:
    specs = demo_controllable_variable_specs()
    assert len(specs) == 4
    assert tuple(item.variable for item in specs) == DEMO_CONTROLLABLE_COLUMNS
    for item in specs:
        assert item.column_role is ColumnRole.CONTROLLABLE_PROCESS
        assert item.confirmed_controllable is True
        assert item.engineering_verified is True
        assert item.lower_bound < item.upper_bound
    validate_demo_recommendation_constraints(specs)


def test_measured_variables_are_not_controllable() -> None:
    controllable = {item.variable for item in demo_controllable_variable_specs()}
    assert controllable.isdisjoint(DEMO_MEASURED_COLUMNS)


def test_supervised_template_contains_complete_recommendation_constraints() -> None:
    preset = build_demo_configuration_preset(
        DemoAnalysisTemplate.SUPERVISED_QUALITY_PREDICTION
    )
    assert preset.confirmed_controllable_variables == list(DEMO_CONTROLLABLE_COLUMNS)
    assert preset.verified_variables == list(DEMO_CONTROLLABLE_COLUMNS)
    assert len(preset.recommendation_constraints) == 4
    by_name = {item.variable: item for item in preset.recommendation_constraints}
    for bound in DEMO_SETPOINT_ENGINEERING_BOUNDS:
        constraint = by_name[bound.variable]
        assert constraint.minimum == bound.lower_bound
        assert constraint.maximum == bound.upper_bound
        assert constraint.minimum < constraint.maximum
    for name in DEMO_CONTROLLABLE_COLUMNS:
        assert preset.role_overrides[name] is ColumnRole.CONTROLLABLE_PROCESS
    for name in DEMO_MEASURED_COLUMNS:
        assert preset.role_overrides[name] is ColumnRole.STATE_SENSOR
        assert name not in preset.confirmed_controllable_variables


def test_anomaly_only_template_has_no_controllable_recommendation_config() -> None:
    preset = build_demo_configuration_preset(
        DemoAnalysisTemplate.ANOMALY_ONLY_PROCESS_MONITORING
    )
    assert preset.confirmed_controllable_variables == []
    assert preset.verified_variables == []
    assert preset.recommendation_constraints == []
    assert preset.role_overrides == {}
    assert preset.anomaly_recommendation_enabled is False


def test_uploaded_csv_configuration_not_assigned_demo_constraints() -> None:
    # Ordinary preset builder without demo helpers stays free of demo bounds.
    preset = build_configuration_preset(
        analysis_mode=AnalysisExecutionMode.SUPERVISED,
        requested_task=AnalysisTask.REGRESSION,
        target_column="y",
        explicit_feature_columns=["x1", "x2"],
    )
    assert preset.confirmed_controllable_variables == []
    assert preset.verified_variables == []
    assert preset.recommendation_constraints == []
    assert preset.role_overrides == {}


def test_demo_template_construction_is_deterministic() -> None:
    first = build_demo_configuration_preset(
        DemoAnalysisTemplate.SUPERVISED_QUALITY_PREDICTION
    )
    second = build_demo_configuration_preset(
        DemoAnalysisTemplate.SUPERVISED_QUALITY_PREDICTION
    )
    assert first.model_dump() == second.model_dump()
    assert demo_recommendation_constraints() == demo_recommendation_constraints()


def test_clear_demo_recommendation_session_state() -> None:
    state: dict[str, object] = {
        SESSION_CONFIRMED_CONTROLLABLE_KEY: list(DEMO_CONTROLLABLE_COLUMNS),
        SESSION_VERIFIED_VARIABLES_KEY: list(DEMO_CONTROLLABLE_COLUMNS),
        SESSION_CONSTRAINT_VARIABLES_KEY: list(DEMO_CONTROLLABLE_COLUMNS),
        "constraint_min_temperature_setpoint": 170.0,
        "constraint_max_temperature_setpoint": 200.0,
        "role_override_temperature_setpoint": "CONTROLLABLE_PROCESS",
    }
    clear_demo_recommendation_session_state(state)
    assert state[SESSION_CONFIRMED_CONTROLLABLE_KEY] == []
    assert state[SESSION_VERIFIED_VARIABLES_KEY] == []
    assert state[SESSION_CONSTRAINT_VARIABLES_KEY] == []
    assert "constraint_min_temperature_setpoint" not in state
    assert "role_override_temperature_setpoint" not in state


def test_forbidden_feature_validator_names_all_offending_columns() -> None:
    features = [
        *_EXPECTED_FEATURES,
        "injected_anomaly",
        "quality_score",
        "timestamp",
        "batch_id",
    ]
    offending = find_forbidden_demo_feature_columns(features)
    assert offending == (
        "injected_anomaly",
        "quality_score",
        "timestamp",
        "batch_id",
    )
    guard = validate_demo_feature_guard(
        features,
        data_source=DATA_SOURCE_BUILTIN_DEMO,
    )
    assert guard.ok is False
    assert guard.forbidden_selected == offending
    assert guard.message is not None
    for name in offending:
        assert name in guard.message


def test_ordinary_upload_not_subjected_to_demo_guard() -> None:
    features = ["injected_anomaly", "quality_score", "timestamp"]
    guard = validate_demo_feature_guard(
        features,
        data_source=DATA_SOURCE_UPLOAD_CSV,
    )
    assert guard.ok is True
    assert guard.forbidden_selected == ()
    assert guard.message is None


def test_deterministic_demo_metadata_summary() -> None:
    frame_a, summary_a = load_builtin_demo_dataset()
    frame_b, summary_b = load_builtin_demo_dataset(DemoDatasetConfiguration())
    assert summary_a == summary_b
    assert summary_a.row_count == 1500
    assert summary_a.column_count == 17
    assert summary_a.random_seed == 42
    assert summary_a.anomaly_row_count == int(frame_a["injected_anomaly"].sum())
    assert summary_a.is_synthetic is True
    rebuilt = build_demo_dataset_ui_summary(
        frame_b,
        configuration=DemoDatasetConfiguration(),
    )
    assert rebuilt == summary_a
    csv_bytes = demo_dataset_to_workflow_csv_bytes(frame_a)
    assert isinstance(csv_bytes, bytes)
    assert csv_bytes.startswith(b"timestamp,")
    assert b"injected_anomaly" in csv_bytes.splitlines()[0]
