"""Built-in manufacturing demo dataset helpers for the Streamlit UI.

Provides leakage-safe analysis templates and demo-only feature guards. Does not
run analysis, persist CSV files, or alter model algorithms.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from io import StringIO
from typing import Any, Final

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from process_intelligence.core.enums import AnalysisTask, ColumnRole
from process_intelligence.demo_data import (
    DEMO_CONTROLLABLE_COLUMNS,
    DEMO_IDENTITY_COLUMNS,
    DEMO_MEASURED_COLUMNS,
    DEMO_QUALITY_COLUMNS,
    DEMO_QUALITY_SCORE_DECLARED_MAXIMUM,
    DEMO_QUALITY_SCORE_DECLARED_MINIMUM,
    DEMO_SETPOINT_ENGINEERING_BOUNDS,
    GROUND_TRUTH_METADATA_COLUMNS,
    DemoDatasetConfiguration,
    generate_demo_dataset,
    summarize_demo_dataset,
)
from process_intelligence.demo_validation import (
    IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    default_demo_excluded_columns,
    default_demo_feature_columns,
)
from process_intelligence.evaluation import MetricAcceptanceDirection
from process_intelligence.recommendation import (
    QualityOptimizationDirection,
    RecommendationObjective,
)
from process_intelligence.ui.configuration_preset import (
    SESSION_CONFIRMED_CONTROLLABLE_KEY,
    SESSION_CONSTRAINT_VARIABLES_KEY,
    SESSION_ROLE_OVERRIDE_COLUMNS_KEY,
    SESSION_VERIFIED_VARIABLES_KEY,
    WorkflowUiConfigurationPreset,
    build_configuration_preset,
)
from process_intelligence.ui.schemas import UiMetricRuleInput, UiVariableConstraintInput
from process_intelligence.workflow.enums import (
    AnalysisExecutionMode,
    OperatingPointSelectionMode,
)

DATA_SOURCE_UPLOAD_CSV: Final[str] = "Upload CSV"
DATA_SOURCE_BUILTIN_DEMO: Final[str] = "Built-in manufacturing demo"

SESSION_DATA_SOURCE_KEY: Final[str] = "ui_data_source"
SESSION_DATA_SOURCE_PREVIOUS_KEY: Final[str] = "ui_data_source_previous"
SESSION_DEMO_TEMPLATE_KEY: Final[str] = "ui_demo_analysis_template"
SESSION_DEMO_APPLY_SUCCESS_KEY: Final[str] = "ui_demo_configuration_apply_success"
SESSION_DEMO_APPLY_SUMMARY_KEY: Final[str] = "ui_demo_configuration_apply_summary"

# Presentation-only report / comparison keys mirrored from streamlit_app.
SESSION_REPORT_KEY: Final[str] = "last_presentation_report_json"
SESSION_BASELINE_REPORT_KEY: Final[str] = "comparison_baseline_report_json"
SESSION_BASELINE_LABEL_KEY: Final[str] = "comparison_baseline_label"
SESSION_REPRODUCIBILITY_BUNDLE_INPUTS_KEY: Final[str] = (
    "last_reproducibility_bundle_inputs_json"
)

# Columns that must never appear as model features for the built-in demo.
FORBIDDEN_DEMO_MODEL_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    *DEMO_IDENTITY_COLUMNS,
    *DEMO_QUALITY_COLUMNS,
    *GROUND_TRUTH_METADATA_COLUMNS,
)

_DEMO_PERFORMANCE_RULES: Final[tuple[UiMetricRuleInput, ...]] = (
    UiMetricRuleInput(
        metric_name="rmse",
        direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
        threshold=1_000_000.0,
        required=True,
    ),
    UiMetricRuleInput(
        metric_name="mae",
        direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
        threshold=1_000_000.0,
        required=True,
    ),
    UiMetricRuleInput(
        metric_name="r2",
        direction=MetricAcceptanceDirection.HIGHER_IS_BETTER,
        threshold=-1_000_000.0,
        required=True,
    ),
)


class DemoAnalysisTemplate(StrEnum):
    """Operator-facing demo analysis templates."""

    SUPERVISED_QUALITY_PREDICTION = "Supervised quality prediction"
    ANOMALY_ONLY_PROCESS_MONITORING = "Anomaly-only process monitoring"


@dataclass(frozen=True, slots=True)
class DemoDatasetUiSummary:
    """Concise synthetic-demo metadata for Streamlit display."""

    row_count: int
    column_count: int
    random_seed: int
    anomaly_row_count: int
    is_synthetic: bool = True

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe summary dictionary."""
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "random_seed": self.random_seed,
            "anomaly_row_count": self.anomaly_row_count,
            "is_synthetic": self.is_synthetic,
        }


@dataclass(frozen=True, slots=True)
class DemoFeatureGuardResult:
    """Result of the demo-only forbidden-feature guard."""

    ok: bool
    forbidden_selected: tuple[str, ...]
    message: str | None


@dataclass(frozen=True, slots=True)
class DemoControllableVariableSpec:
    """Verified controllable setpoint specification for the supervised demo.

    Bounds are synthetic demonstration limits from the generator design, not
    real equipment operating limits.
    """

    variable: str
    lower_bound: float
    upper_bound: float
    column_role: ColumnRole = ColumnRole.CONTROLLABLE_PROCESS
    confirmed_controllable: bool = True
    engineering_verified: bool = True

    def __post_init__(self) -> None:
        if self.variable not in DEMO_CONTROLLABLE_COLUMNS:
            raise ValueError(
                "DemoControllableVariableSpec.variable must be a demo "
                f"controllable setpoint, got {self.variable!r}"
            )
        if self.column_role is not ColumnRole.CONTROLLABLE_PROCESS:
            raise ValueError(
                "DemoControllableVariableSpec.column_role must be "
                "CONTROLLABLE_PROCESS"
            )
        if not self.confirmed_controllable or not self.engineering_verified:
            raise ValueError(
                "DemoControllableVariableSpec requires confirmed_controllable "
                "and engineering_verified to be True"
            )
        if self.lower_bound >= self.upper_bound:
            raise ValueError(
                "DemoControllableVariableSpec lower_bound must be < upper_bound "
                f"(got {self.lower_bound} >= {self.upper_bound})"
            )


def demo_controllable_variable_specs() -> tuple[DemoControllableVariableSpec, ...]:
    """Return the four verified controllable setpoints for the supervised demo."""
    return tuple(
        DemoControllableVariableSpec(
            variable=item.variable,
            lower_bound=item.lower_bound,
            upper_bound=item.upper_bound,
        )
        for item in DEMO_SETPOINT_ENGINEERING_BOUNDS
    )


def demo_supervised_role_overrides() -> dict[str, ColumnRole]:
    """Return supervised demo column-role overrides for recommendation safety."""
    roles: dict[str, ColumnRole] = {
        name: ColumnRole.CONTROLLABLE_PROCESS for name in DEMO_CONTROLLABLE_COLUMNS
    }
    roles.update({name: ColumnRole.STATE_SENSOR for name in DEMO_MEASURED_COLUMNS})
    roles[TARGET_COLUMN] = ColumnRole.TARGET_QUALITY
    return roles


def demo_recommendation_constraints() -> list[UiVariableConstraintInput]:
    """Build UI constraint DTOs from approved synthetic setpoint bounds."""
    return [
        UiVariableConstraintInput(
            variable=spec.variable,
            minimum=spec.lower_bound,
            maximum=spec.upper_bound,
        )
        for spec in demo_controllable_variable_specs()
    ]


def validate_demo_recommendation_constraints(
    specs: Sequence[DemoControllableVariableSpec] | None = None,
) -> None:
    """Validate that supervised demo controllable constraints are complete."""
    active = (
        demo_controllable_variable_specs() if specs is None else tuple(specs)
    )
    if len(active) != len(DEMO_CONTROLLABLE_COLUMNS):
        raise ValueError(
            "supervised demo must define exactly "
            f"{len(DEMO_CONTROLLABLE_COLUMNS)} controllable variables, "
            f"got {len(active)}"
        )
    names = [item.variable for item in active]
    if tuple(names) != DEMO_CONTROLLABLE_COLUMNS:
        raise ValueError(
            "supervised demo controllable variables must be exactly "
            f"{DEMO_CONTROLLABLE_COLUMNS}, got {tuple(names)}"
        )
    measured = set(DEMO_MEASURED_COLUMNS)
    for item in active:
        if item.variable in measured:
            raise ValueError(
                f"measured variable {item.variable!r} must not be controllable"
            )
        if item.lower_bound >= item.upper_bound:
            raise ValueError(
                f"invalid bounds for {item.variable!r}: "
                f"{item.lower_bound} >= {item.upper_bound}"
            )
        approved = next(
            bound
            for bound in DEMO_SETPOINT_ENGINEERING_BOUNDS
            if bound.variable == item.variable
        )
        if (
            item.lower_bound != approved.lower_bound
            or item.upper_bound != approved.upper_bound
        ):
            raise ValueError(
                f"demo bounds for {item.variable!r} must match generator "
                f"design ranges ({approved.lower_bound}, {approved.upper_bound})"
            )


def is_builtin_demo_source(value: object) -> bool:
    """Return True when the active data source is the built-in demo."""
    return value == DATA_SOURCE_BUILTIN_DEMO


def demo_template_options() -> tuple[str, ...]:
    """Return the exact operator-facing demo template labels."""
    return tuple(item.value for item in DemoAnalysisTemplate)


def parse_demo_analysis_template(value: object) -> DemoAnalysisTemplate:
    """Parse a demo template label or enum member."""
    if isinstance(value, DemoAnalysisTemplate):
        return value
    if isinstance(value, str):
        try:
            return DemoAnalysisTemplate(value)
        except ValueError as exc:
            raise ValueError(f"invalid DemoAnalysisTemplate: {value!r}") from exc
    raise TypeError(
        "demo template must be DemoAnalysisTemplate or str, "
        f"got {type(value).__name__}"
    )


def load_builtin_demo_dataset(
    configuration: DemoDatasetConfiguration | None = None,
) -> tuple[pd.DataFrame, DemoDatasetUiSummary]:
    """Generate the deterministic built-in demo dataset in memory.

    Does not write a CSV file. Callers may convert the frame to workflow CSV
    bytes for ephemeral UI / temp-path execution.
    """
    active = (
        DemoDatasetConfiguration()
        if configuration is None
        else configuration
    )
    if not isinstance(active, DemoDatasetConfiguration):
        raise TypeError(
            "configuration must be DemoDatasetConfiguration or None, "
            f"got {type(active).__name__}"
        )
    frame = generate_demo_dataset(active)
    return frame, build_demo_dataset_ui_summary(frame, configuration=active)


def build_demo_dataset_ui_summary(
    frame: pd.DataFrame,
    *,
    configuration: DemoDatasetConfiguration | None = None,
) -> DemoDatasetUiSummary:
    """Build a deterministic operator-facing demo metadata summary."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"frame must be a pandas.DataFrame, got {type(frame).__name__}"
        )
    active = (
        DemoDatasetConfiguration()
        if configuration is None
        else configuration
    )
    summary = summarize_demo_dataset(frame)
    row_count = summary["row_count"]
    column_count = summary["column_count"]
    anomaly_row_count = summary["anomaly_row_count"]
    if not isinstance(row_count, int):
        raise TypeError(
            f"row_count must be int, got {type(row_count).__name__}"
        )
    if not isinstance(column_count, int):
        raise TypeError(
            f"column_count must be int, got {type(column_count).__name__}"
        )
    if not isinstance(anomaly_row_count, int):
        raise TypeError(
            f"anomaly_row_count must be int, got {type(anomaly_row_count).__name__}"
        )
    return DemoDatasetUiSummary(
        row_count=row_count,
        column_count=column_count,
        random_seed=int(active.random_seed),
        anomaly_row_count=anomaly_row_count,
        is_synthetic=True,
    )


def demo_dataset_to_workflow_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a demo frame to workflow-compatible CSV bytes in memory.

    Timestamps are encoded as UTC unix seconds, matching
    ``write_demo_workflow_csv``. No file is written.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"frame must be a pandas.DataFrame, got {type(frame).__name__}"
        )
    export = frame.copy()
    timestamps = pd.to_datetime(export["timestamp"], utc=True)
    export["timestamp"] = (timestamps.astype("int64") // 10**9).astype(np.int64)
    buffer = StringIO()
    export.to_csv(buffer, index=False, encoding="utf-8")
    return buffer.getvalue().encode("utf-8")


def build_demo_configuration_preset(
    template: DemoAnalysisTemplate | str,
) -> WorkflowUiConfigurationPreset:
    """Build a leakage-safe configuration preset for a demo analysis template.

    Feature columns are exactly ``default_demo_feature_columns()``. Identity,
    quality-output, and ground-truth columns are kept out of model features via
    timestamp / identifier / excluded / target fields that satisfy existing UI
    contracts. Does not execute analysis.
    """
    selected = parse_demo_analysis_template(template)
    feature_columns = default_demo_feature_columns()

    if selected is DemoAnalysisTemplate.SUPERVISED_QUALITY_PREDICTION:
        controllable_specs = demo_controllable_variable_specs()
        validate_demo_recommendation_constraints(controllable_specs)
        return build_configuration_preset(
            analysis_mode=AnalysisExecutionMode.SUPERVISED,
            requested_task=AnalysisTask.REGRESSION,
            target_column=TARGET_COLUMN,
            use_recommended_numeric_feature_set=False,
            explicit_feature_columns=feature_columns,
            timestamp_column=TIMESTAMP_COLUMN,
            identifier_columns=list(IDENTIFIER_COLUMNS),
            excluded_columns=default_demo_excluded_columns(
                analysis_mode=AnalysisExecutionMode.SUPERVISED,
            ),
            role_overrides=demo_supervised_role_overrides(),
            recommendation_objective=(
                RecommendationObjective.IMPROVE_PREDICTED_QUALITY
            ),
            quality_direction=QualityOptimizationDirection.MAXIMIZE,
            declared_target_minimum=DEMO_QUALITY_SCORE_DECLARED_MINIMUM,
            declared_target_maximum=DEMO_QUALITY_SCORE_DECLARED_MAXIMUM,
            performance_rules=list(_DEMO_PERFORMANCE_RULES),
            cohort_filter=None,
            anomaly_recommendation_enabled=False,
            recommendation_constraints=demo_recommendation_constraints(),
            confirmed_controllable_variables=list(DEMO_CONTROLLABLE_COLUMNS),
            verified_variables=list(DEMO_CONTROLLABLE_COLUMNS),
            operating_point_selection_mode=(
                OperatingPointSelectionMode.TOP_RESIDUAL_ANOMALY
            ),
            explicit_operating_row_id=None,
        )

    return build_configuration_preset(
        analysis_mode=AnalysisExecutionMode.ANOMALY_ONLY,
        requested_task=None,
        target_column=None,
        use_recommended_numeric_feature_set=False,
        explicit_feature_columns=feature_columns,
        timestamp_column=TIMESTAMP_COLUMN,
        identifier_columns=list(IDENTIFIER_COLUMNS),
        excluded_columns=default_demo_excluded_columns(
            analysis_mode=AnalysisExecutionMode.ANOMALY_ONLY,
        ),
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
        operating_point_selection_mode=(
            OperatingPointSelectionMode.TOP_UNSUPERVISED_ANOMALY
        ),
        explicit_operating_row_id=None,
    )


def find_forbidden_demo_feature_columns(
    feature_columns: Sequence[str],
) -> tuple[str, ...]:
    """Return forbidden demo columns present in ``feature_columns`` (sorted)."""
    if not isinstance(feature_columns, Sequence) or isinstance(
        feature_columns,
        (str, bytes),
    ):
        raise TypeError(
            "feature_columns must be a sequence of str, "
            f"got {type(feature_columns).__name__}"
        )
    forbidden = set(FORBIDDEN_DEMO_MODEL_FEATURE_COLUMNS)
    selected = [str(name) for name in feature_columns if str(name) in forbidden]
    # Preserve first-seen order while remaining deterministic for duplicates.
    seen: set[str] = set()
    ordered: list[str] = []
    for name in selected:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return tuple(ordered)


def validate_demo_feature_guard(
    feature_columns: Sequence[str],
    *,
    data_source: object,
) -> DemoFeatureGuardResult:
    """Validate demo-only leakage rules for the final feature list.

    Ordinary uploaded CSV configurations are not subjected to this guard.
    Offending columns are reported; they are never silently removed.
    """
    if not is_builtin_demo_source(data_source):
        return DemoFeatureGuardResult(
            ok=True,
            forbidden_selected=(),
            message=None,
        )
    offending = find_forbidden_demo_feature_columns(feature_columns)
    if not offending:
        return DemoFeatureGuardResult(
            ok=True,
            forbidden_selected=(),
            message=None,
        )
    named = ", ".join(offending)
    return DemoFeatureGuardResult(
        ok=False,
        forbidden_selected=offending,
        message=(
            "Built-in demo model features must not include identity, "
            "quality-output, or ground-truth metadata columns: "
            f"{named}. Remove them from the feature list before running "
            "analysis."
        ),
    )


def clear_stale_analysis_session_state(
    session_state: MutableMapping[str, Any],
) -> None:
    """Clear cached analysis report and run-comparison baseline state."""
    session_state.pop(SESSION_REPORT_KEY, None)
    session_state.pop(SESSION_BASELINE_REPORT_KEY, None)
    session_state.pop(SESSION_BASELINE_LABEL_KEY, None)
    session_state.pop(SESSION_REPRODUCIBILITY_BUNDLE_INPUTS_KEY, None)


def clear_demo_recommendation_session_state(
    session_state: MutableMapping[str, Any],
) -> None:
    """Clear demo-applied controllable / constraint / role-override widget state.

    Used when leaving the built-in demo source so uploaded CSV runs do not
    inherit synthetic demonstration constraints.
    """
    session_state[SESSION_CONSTRAINT_VARIABLES_KEY] = []
    session_state[SESSION_CONFIRMED_CONTROLLABLE_KEY] = []
    session_state[SESSION_VERIFIED_VARIABLES_KEY] = []
    session_state[SESSION_ROLE_OVERRIDE_COLUMNS_KEY] = []
    for key in list(session_state.keys()):
        if not isinstance(key, str):
            continue
        if (
            key.startswith("constraint_min_")
            or key.startswith("constraint_max_")
            or key.startswith("role_override_")
        ):
            session_state.pop(key, None)
    session_state.pop(SESSION_DEMO_APPLY_SUCCESS_KEY, None)
    session_state.pop(SESSION_DEMO_APPLY_SUMMARY_KEY, None)


def demo_configuration_apply_summary(
    preset: WorkflowUiConfigurationPreset,
) -> dict[str, Any]:
    """Build a concise success summary after applying a demo template."""
    if not isinstance(preset, WorkflowUiConfigurationPreset):
        raise TypeError(
            "preset must be WorkflowUiConfigurationPreset, "
            f"got {type(preset).__name__}"
        )
    controllable_names = list(preset.confirmed_controllable_variables)
    bound_summary = [
        {
            "variable": item.variable,
            "lower_bound": item.minimum,
            "upper_bound": item.maximum,
        }
        for item in preset.recommendation_constraints
    ]
    return {
        "analysis_mode": preset.analysis_mode.value,
        "target": (
            "Not required"
            if preset.analysis_mode is AnalysisExecutionMode.ANOMALY_ONLY
            else (preset.target_column or "Not selected")
        ),
        "requested_task": (
            None if preset.requested_task is None else preset.requested_task.value
        ),
        "feature_count": len(preset.explicit_feature_columns),
        "timestamp_column": preset.timestamp_column,
        "operating_point_selection_mode": (
            preset.operating_point_selection_mode.value
        ),
        "anomaly_recommendation_enabled": preset.anomaly_recommendation_enabled,
        "cohort_filter_configured": preset.cohort_filter is not None,
        "verified_controllable_count": len(controllable_names),
        "verified_controllable_variables": controllable_names,
        "approved_synthetic_bounds": bound_summary,
        "bounds_are_demonstration_only": True,
    }


def filter_ground_truth_from_recommended_features(
    recommended_features: Sequence[str],
) -> list[str]:
    """Remove ground-truth demo metadata from recommended feature suggestions."""
    blocked = set(GROUND_TRUTH_METADATA_COLUMNS)
    return [name for name in recommended_features if name not in blocked]


def ground_truth_metadata_columns_present(
    columns: Sequence[str],
) -> tuple[str, ...]:
    """Return ground-truth metadata columns present in ``columns``."""
    available = set(columns)
    return tuple(name for name in GROUND_TRUTH_METADATA_COLUMNS if name in available)


def demo_non_feature_columns_for_template(
    template: DemoAnalysisTemplate | str,
) -> tuple[str, ...]:
    """Return columns that must stay out of model features for a template."""
    selected = parse_demo_analysis_template(template)
    if selected is DemoAnalysisTemplate.SUPERVISED_QUALITY_PREDICTION:
        # Target is quality_score; still forbidden as a feature.
        return FORBIDDEN_DEMO_MODEL_FEATURE_COLUMNS
    return FORBIDDEN_DEMO_MODEL_FEATURE_COLUMNS


def assert_preset_excludes_forbidden_features(
    preset: WorkflowUiConfigurationPreset,
) -> None:
    """Raise ``AssertionError`` when a demo preset leaks forbidden features."""
    leaked = find_forbidden_demo_feature_columns(preset.explicit_feature_columns)
    if leaked:
        raise AssertionError(
            f"demo preset feature columns include forbidden names: {leaked}"
        )


def session_indicates_completed_analysis(
    session_state: Mapping[str, Any],
) -> bool:
    """Return True when a cached presentation report is present."""
    return isinstance(session_state.get(SESSION_REPORT_KEY), dict)
