"""End-to-end integration tests for the Step 10C analysis workflow orchestrator.

These tests exercise the public ``IndustrialProcessAnalysisWorkflow`` against a
synthetic raw CSV. They do not import from the Step 10B smoke module; the
synthetic fixture is recreated here independently.

Scope notes:
    * The full test suite, ``ruff``, and ``mypy`` are expected to be run
      externally by the developer / CI; nothing here suppresses warnings or
      relaxes leakage / independent-test contracts.
    * Floating point comparisons use ``math.isclose`` / ``pytest.approx``.
    * No estimator class names or exact metric values are hard-coded.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from process_intelligence.core.enums import AnalysisTask, ColumnRole
from process_intelligence.core.schemas import VariableConstraint
from process_intelligence.evaluation import (
    MetricAcceptanceDirection,
    MetricAcceptanceRule,
    ModelPerformanceAcceptancePolicy,
    ModelPerformanceAcceptanceStatus,
)
from process_intelligence.recommendation import (
    QualityOptimizationDirection,
    RecommendationObjective,
    RecommendationStatus,
)
from process_intelligence.workflow import (
    ANOMALY_CONTEXT_MAX_FEATURES,
    ANOMALY_CONTEXT_RADIUS,
    AnalysisExecutionMode,
    AnalysisWorkflowOutcome,
    AnalysisWorkflowPolicy,
    AnalysisWorkflowReport,
    AnalysisWorkflowRequest,
    AnalysisWorkflowStage,
    AnalysisWorkflowStatus,
    AnomalyContextOrderBasis,
    IndustrialProcessAnalysisWorkflow,
    NumericCohortFilter,
    OperatingPointSelectionMode,
    TaskSelectionSource,
)

# ---------------------------------------------------------------------------
# Synthetic dataset definition
# ---------------------------------------------------------------------------

CSV_NAME = "semiconductor_process_smoke.csv"
N_ROWS = 180
RNG_SEED = 20260721

PROCESS_FEATURES = [
    "chamber_temperature",
    "chamber_pressure",
    "gas_flow_rate",
    "rf_power",
    "deposition_time",
]
TARGET_COLUMN = "quality_score"
TIME_COLUMN = "timestamp"
ID_COLUMNS = ["lot_id", "wafer_id"]
EXCLUDED_COLUMNS = ["film_thickness", "injected_anomaly"]
ALL_CSV_COLUMNS = [
    TIME_COLUMN,
    *ID_COLUMNS,
    *PROCESS_FEATURES,
    "film_thickness",
    TARGET_COLUMN,
    "injected_anomaly",
]

# Fixed, deliberately wide constraint bounds. The generation formula (including
# the injected-anomaly perturbations) keeps every feature comfortably inside
# these ranges, so the operating-point baseline is always within bounds.
CONSTRAINT_BOUNDS: dict[str, tuple[float, float]] = {
    "chamber_temperature": (150.0, 300.0),
    "chamber_pressure": (10.0, 70.0),
    "gas_flow_rate": (60.0, 200.0),
    "rf_power": (200.0, 500.0),
    "deposition_time": (30.0, 80.0),
}

_FORBIDDEN_OUTCOME_PATTERNS = (
    "guaranteed improvement",
    "proven root cause",
    "this will fix",
    "proven optimal",
)


def _build_synthetic_rows(*, n_rows: int = N_ROWS) -> list[dict[str, Any]]:
    """Build deterministic synthetic semiconductor-style process rows."""
    rng = np.random.default_rng(RNG_SEED)
    rows: list[dict[str, Any]] = []

    # ~6.7% anomalies spaced across the whole timeline so temporal splits place
    # some anomalies into train, validation, and test partitions.
    anomaly_every = 15
    anomaly_indices = {i for i in range(7, n_rows, anomaly_every)}

    for i in range(n_rows):
        temperature = 220.0 + rng.normal(0.0, 4.0)
        pressure = 40.0 + 0.12 * (temperature - 220.0) + rng.normal(0.0, 1.2)
        gas_flow = 120.0 + 0.35 * (pressure - 40.0) + rng.normal(0.0, 2.5)
        rf_power = float(rng.uniform(280.0, 420.0))
        deposition_time = 55.0 + rng.normal(0.0, 2.0)

        injected = 0
        if i in anomaly_indices:
            injected = 1
            temperature += 18.0
            pressure -= 8.0
            gas_flow += 22.0

        film_thickness = (
            0.06 * temperature
            + 0.35 * pressure
            + 0.03 * gas_flow
            + 0.015 * rf_power
            + 0.08 * deposition_time
            + float(rng.normal(0.0, 0.35))
        )
        if injected:
            film_thickness += 6.0

        quality = (
            88.0
            - 0.045 * (temperature - 220.0) ** 2
            - 0.08 * (pressure - 40.0) ** 2
            - 0.004 * (gas_flow - 120.0) ** 2
            - 0.0008 * (rf_power - 350.0) ** 2
            - 0.06 * (deposition_time - 55.0) ** 2
            + 0.02 * film_thickness
            + float(rng.normal(0.0, 0.4))
        )
        if injected:
            quality -= 12.0

        rows.append(
            {
                TIME_COLUMN: 1_700_000_000 + i * 3600,
                "lot_id": f"LOT-{i // 12:03d}",
                "wafer_id": f"W-{i:04d}",
                "chamber_temperature": float(temperature),
                "chamber_pressure": float(pressure),
                "gas_flow_rate": float(gas_flow),
                "rf_power": float(rf_power),
                "deposition_time": float(deposition_time),
                "film_thickness": float(film_thickness),
                TARGET_COLUMN: float(quality),
                "injected_anomaly": int(injected),
            }
        )
    return rows


def write_synthetic_csv(path: Path, *, n_rows: int = N_ROWS) -> Path:
    """Write the canonical continuous-target synthetic CSV via polars."""
    rows = _build_synthetic_rows(n_rows=n_rows)
    frame = pl.DataFrame(rows).select(ALL_CSV_COLUMNS)
    assert frame.height == n_rows
    assert frame.select(pl.all().is_null().any()).row(0) == tuple(
        [False] * len(ALL_CSV_COLUMNS)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path)
    return path


def write_classification_csv(path: Path, *, n_rows: int = N_ROWS) -> Path:
    """Write a synthetic CSV whose target is a low-cardinality binary label.

    Task routing selects CLASSIFICATION for this target, which lets the
    regression-required policy exercise its structured refusal path.
    """
    rows = _build_synthetic_rows(n_rows=n_rows)
    for i, row in enumerate(rows):
        row[TARGET_COLUMN] = int(i % 2)
    frame = pl.DataFrame(rows).select(ALL_CSV_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path)
    return path


def write_soh_like_csv(path: Path, *, n_rows: int = N_ROWS) -> Path:
    """Write a numeric SOH target with repeated values (classification-prone).

    The target remains a float dtype, but unique-value cardinality is low enough
    that the task router may infer CLASSIFICATION. Values are not cast or
    expanded to force regression.
    """
    rows = _build_synthetic_rows(n_rows=n_rows)
    # Five repeated SOH levels keep cardinality low enough for CLASSIFICATION
    # inference while preserving a float dtype. Values are not expanded.
    soh_levels = [0.70, 0.75, 0.80, 0.85, 0.90]
    for i, row in enumerate(rows):
        row["SOH"] = float(soh_levels[i % len(soh_levels)])
        del row[TARGET_COLUMN]
    columns = [
        TIME_COLUMN,
        *ID_COLUMNS,
        *PROCESS_FEATURES,
        "film_thickness",
        "SOH",
        "injected_anomaly",
    ]
    frame = pl.DataFrame(rows).select(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path)
    return path


def write_constant_soh_csv(path: Path, *, n_rows: int = 48) -> Path:
    """Write a battery-like CSV where SOH is numeric and entirely constant zero.

    Row count is intentionally smaller than production CSVs. Values are not
    mutated after write; SOH remains 0 for every row.
    """
    rows = _build_synthetic_rows(n_rows=n_rows)
    for row in rows:
        row["SOH"] = 0
        del row[TARGET_COLUMN]
    columns = [
        TIME_COLUMN,
        *ID_COLUMNS,
        *PROCESS_FEATURES,
        "film_thickness",
        "SOH",
        "injected_anomaly",
    ]
    frame = pl.DataFrame(rows).select(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path)
    return path


def write_invalid_csv(path: Path, *, n_rows: int = N_ROWS) -> Path:
    """Write a synthetic CSV containing an all-null column (VALIDATE blocker)."""
    rows = _build_synthetic_rows(n_rows=n_rows)
    frame = pl.DataFrame(rows).select(ALL_CSV_COLUMNS)
    frame = frame.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("all_null_sensor")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path)
    return path


# ---------------------------------------------------------------------------
# Battery-like synthetic dataset for ANOMALY_ONLY analysis-mode coverage
# ---------------------------------------------------------------------------

BATTERY_N_ROWS = 90
BATTERY_RNG_SEED = 20260722
BATTERY_ID_COLUMN = "SerialNumber"
BATTERY_DATE_COLUMN = "Date"
BATTERY_TIME_COLUMN = "Time"
BATTERY_TARGET_COLUMN = "SOH"
BATTERY_FEATURES = [
    "voltage",
    "current",
    "temperature",
    "internal_resistance",
    "cycle_count",
]


def _build_battery_rows(*, n_rows: int = BATTERY_N_ROWS) -> list[dict[str, Any]]:
    """Build deterministic synthetic battery-style rows with clear outliers.

    ``SOH`` is a numeric column that is constant zero for every row; it is
    never used as a target in ANOMALY_ONLY analysis. Roughly one row in six
    is perturbed into a clear multivariate outlier so unsupervised anomaly
    detection has real signal to find.
    """
    rng = np.random.default_rng(BATTERY_RNG_SEED)
    rows: list[dict[str, Any]] = []
    for i in range(n_rows):
        voltage = float(3.7 + rng.normal(0.0, 0.03))
        current = float(1.2 + rng.normal(0.0, 0.04))
        temperature = float(25.0 + rng.normal(0.0, 1.0))
        internal_resistance = float(0.05 + rng.normal(0.0, 0.004))
        cycle_count = float(i)
        if i % 6 == 3:
            # Clear outlier: low voltage, hot, high internal resistance.
            voltage -= 0.8
            temperature += 18.0
            internal_resistance += 0.06
        rows.append(
            {
                BATTERY_ID_COLUMN: f"BATT-{i:04d}",
                BATTERY_DATE_COLUMN: f"2026-01-{(i % 28) + 1:02d}",
                BATTERY_TIME_COLUMN: f"{(i % 24):02d}:00:00",
                "voltage": voltage,
                "current": current,
                "temperature": temperature,
                "internal_resistance": internal_resistance,
                "cycle_count": cycle_count,
                BATTERY_TARGET_COLUMN: 0,
            }
        )
    return rows


def write_battery_anomaly_csv(path: Path, *, n_rows: int = BATTERY_N_ROWS) -> Path:
    """Write the battery-like ANOMALY_ONLY fixture CSV."""
    rows = _build_battery_rows(n_rows=n_rows)
    columns = [
        BATTERY_ID_COLUMN,
        BATTERY_DATE_COLUMN,
        BATTERY_TIME_COLUMN,
        *BATTERY_FEATURES,
        BATTERY_TARGET_COLUMN,
    ]
    frame = pl.DataFrame(rows).select(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path)
    return path


# ---------------------------------------------------------------------------
# Request / workflow helpers
# ---------------------------------------------------------------------------


def _lenient_performance_policy() -> ModelPerformanceAcceptancePolicy:
    """Return deliberately lenient independent-test acceptance thresholds."""
    return ModelPerformanceAcceptancePolicy(
        rules=[
            MetricAcceptanceRule(
                metric_name="rmse",
                direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
                threshold=1_000_000.0,
            ),
            MetricAcceptanceRule(
                metric_name="mae",
                direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
                threshold=1_000_000.0,
            ),
            MetricAcceptanceRule(
                metric_name="r2",
                direction=MetricAcceptanceDirection.HIGHER_IS_BETTER,
                threshold=-1_000_000.0,
            ),
        ],
        minimum_test_rows=1,
    )


def build_request(
    csv_path: Path,
    objective: RecommendationObjective,
    **overrides: Any,
) -> AnalysisWorkflowRequest:
    """Build a validated ``AnalysisWorkflowRequest`` for the synthetic dataset.

    Constraints, confirmations, and verifications cover every process feature
    with wide bounds so the safety gate can treat them as eligible change
    candidates. ``overrides`` replace any top-level request field.
    """
    if objective is RecommendationObjective.REDUCE_ANOMALY_SCORE:
        quality_direction: QualityOptimizationDirection | None = None
    else:
        quality_direction = QualityOptimizationDirection.MAXIMIZE

    constraints = [
        VariableConstraint(
            variable=name,
            adjustable=True,
            minimum=low,
            maximum=high,
            fixed=False,
        )
        for name, (low, high) in CONSTRAINT_BOUNDS.items()
    ]
    role_overrides: dict[str, ColumnRole] = {
        name: ColumnRole.CONTROLLABLE_PROCESS for name in PROCESS_FEATURES
    }
    role_overrides[TARGET_COLUMN] = ColumnRole.TARGET_QUALITY

    params: dict[str, Any] = {
        "csv_path": csv_path,
        "target_column": TARGET_COLUMN,
        "feature_columns": list(PROCESS_FEATURES),
        "timestamp_column": TIME_COLUMN,
        "identifier_columns": list(ID_COLUMNS),
        "excluded_columns": list(EXCLUDED_COLUMNS),
        "column_role_overrides": role_overrides,
        "objective": objective,
        "quality_direction": quality_direction,
        "request_constraints": constraints,
        "user_confirmed_controllable_variables": list(PROCESS_FEATURES),
        "user_verified_variables": list(PROCESS_FEATURES),
        "max_simultaneous_changes": 2,
        "operating_point_selection": OperatingPointSelectionMode.TOP_RESIDUAL_ANOMALY,
        "model_performance_policy": _lenient_performance_policy(),
    }
    params.update(overrides)
    return AnalysisWorkflowRequest(**params)


def _happy_policy(**overrides: Any) -> AnalysisWorkflowPolicy:
    """Return the shared happy-path policy, optionally overriding fields."""
    params: dict[str, Any] = {
        "allow_partial_diagnosis_ensemble": True,
        "require_semiconductor_industry": False,
        "require_regression_task": True,
        "require_residual_diagnosis": True,
    }
    params.update(overrides)
    return AnalysisWorkflowPolicy(**params)


def _make_workflow(**policy_overrides: Any) -> IndustrialProcessAnalysisWorkflow:
    return IndustrialProcessAnalysisWorkflow(policy=_happy_policy(**policy_overrides))


def _run(
    csv_path: Path,
    objective: RecommendationObjective,
    *,
    policy_overrides: dict[str, Any] | None = None,
    **request_overrides: Any,
) -> AnalysisWorkflowOutcome:
    workflow = _make_workflow(**(policy_overrides or {}))
    return workflow.run(build_request(csv_path, objective, **request_overrides))


def build_anomaly_request(
    csv_path: Path,
    **overrides: Any,
) -> AnalysisWorkflowRequest:
    """Build a validated ANOMALY_ONLY ``AnalysisWorkflowRequest``.

    No target, task, performance policy, objective, quality direction, or
    quality target is set: ANOMALY_ONLY rejects all of those fields.
    ``overrides`` replace any top-level request field.
    """
    params: dict[str, Any] = {
        "csv_path": csv_path,
        "analysis_mode": AnalysisExecutionMode.ANOMALY_ONLY,
        "feature_columns": list(BATTERY_FEATURES),
        "identifier_columns": [BATTERY_ID_COLUMN],
        "excluded_columns": [
            BATTERY_DATE_COLUMN,
            BATTERY_TIME_COLUMN,
            BATTERY_TARGET_COLUMN,
        ],
        "operating_point_selection": OperatingPointSelectionMode.TOP_UNSUPERVISED_ANOMALY,
    }
    params.update(overrides)
    return AnalysisWorkflowRequest(**params)


def _anomaly_policy(**overrides: Any) -> AnalysisWorkflowPolicy:
    """Return the shared ANOMALY_ONLY policy, optionally overriding fields."""
    params: dict[str, Any] = {
        "require_regression_task": False,
        "require_residual_diagnosis": False,
        "allow_partial_diagnosis_ensemble": True,
        "require_semiconductor_industry": False,
    }
    params.update(overrides)
    return AnalysisWorkflowPolicy(**params)


def _make_anomaly_workflow(
    **policy_overrides: Any,
) -> IndustrialProcessAnalysisWorkflow:
    return IndustrialProcessAnalysisWorkflow(policy=_anomaly_policy(**policy_overrides))


def _run_anomaly(
    csv_path: Path,
    *,
    policy_overrides: dict[str, Any] | None = None,
    **request_overrides: Any,
) -> AnalysisWorkflowOutcome:
    workflow = _make_anomaly_workflow(**(policy_overrides or {}))
    return workflow.run(build_anomaly_request(csv_path, **request_overrides))


def _business_fingerprint(report: AnalysisWorkflowReport) -> dict[str, Any]:
    """Extract a run-independent business fingerprint for determinism checks."""
    recommendation = report.final_recommendation
    if recommendation is None:
        changes: list[tuple[str, float, float]] = []
        recommendation_status: str | None = None
        recommendation_confidence: float | None = None
    else:
        changes = [
            (change.variable, change.proposed_value, change.current_value)
            for change in recommendation.changes
        ]
        recommendation_status = recommendation.status.value
        recommendation_confidence = recommendation.confidence
    return {
        "status": report.status.value,
        "terminal_stage": report.terminal_stage.value,
        "selected_industry": report.selected_industry,
        "selected_task": (
            report.selected_task.value if report.selected_task is not None else None
        ),
        "selected_supervised_model_key": report.selected_supervised_model_key,
        "selected_anomaly_model_key": report.selected_anomaly_model_key,
        "selected_operating_row_id": report.selected_operating_row_id,
        "anomaly_event_count": report.anomaly_event_count,
        "diagnosis_factor_count": report.diagnosis_factor_count,
        "train_row_count": report.train_row_count,
        "validation_row_count": report.validation_row_count,
        "test_row_count": report.test_row_count,
        "recommendation_status": recommendation_status,
        "recommendation_confidence": recommendation_confidence,
        "changes": changes,
    }


_HAPPY_STATUSES = {AnalysisWorkflowStatus.COMPLETED, AnalysisWorkflowStatus.PARTIAL}
_HAPPY_RECOMMENDATION_STATUSES = {
    RecommendationStatus.GENERATED,
    RecommendationStatus.READY_FOR_OPTIMIZATION,
}


def _assert_finite(value: object, *, label: str) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool), label
    number = float(value)
    assert math.isfinite(number), label
    return number


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("wf") / CSV_NAME
    write_synthetic_csv(path)
    return path


@pytest.fixture(scope="module")
def quality_outcome(synthetic_csv: Path) -> AnalysisWorkflowOutcome:
    return _run(synthetic_csv, RecommendationObjective.IMPROVE_PREDICTED_QUALITY)


@pytest.fixture(scope="module")
def reduce_outcome(synthetic_csv: Path) -> AnalysisWorkflowOutcome:
    return _run(synthetic_csv, RecommendationObjective.REDUCE_ANOMALY_SCORE)


@pytest.fixture(scope="module")
def balance_outcome(synthetic_csv: Path) -> AnalysisWorkflowOutcome:
    return _run(synthetic_csv, RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY)


@pytest.fixture(scope="module")
def explicit_outcome(
    synthetic_csv: Path,
    quality_outcome: AnalysisWorkflowOutcome,
) -> AnalysisWorkflowOutcome:
    row_id = quality_outcome.report.selected_operating_row_id
    assert isinstance(row_id, int)
    return _run(
        synthetic_csv,
        RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        operating_point_selection=OperatingPointSelectionMode.EXPLICIT_ROW_ID,
        explicit_operating_row_id=int(row_id),
    )


@pytest.fixture
def workflow() -> IndustrialProcessAnalysisWorkflow:
    return IndustrialProcessAnalysisWorkflow(
        policy=AnalysisWorkflowPolicy(allow_partial_diagnosis_ensemble=True)
    )


@pytest.fixture(scope="module")
def battery_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("battery") / "battery_anomaly_smoke.csv"
    write_battery_anomaly_csv(path)
    return path


@pytest.fixture(scope="module")
def anomaly_outcome(battery_csv: Path) -> AnalysisWorkflowOutcome:
    return _run_anomaly(battery_csv)


# ---------------------------------------------------------------------------
# 1. Instance creation
# ---------------------------------------------------------------------------


def test_workflow_instance_creation(
    workflow: IndustrialProcessAnalysisWorkflow,
) -> None:
    assert isinstance(workflow, IndustrialProcessAnalysisWorkflow)
    metadata = workflow.get_metadata()
    assert metadata["stage_count"] == len(list(AnalysisWorkflowStage))
    assert metadata["allow_partial_diagnosis_ensemble"] is True
    assert metadata["loads_raw_csv"] is True


# ---------------------------------------------------------------------------
# 2-4. Objective runs
# ---------------------------------------------------------------------------


def test_run_improve_predicted_quality(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    report = quality_outcome.report
    assert report.status in _HAPPY_STATUSES
    assert report.final_recommendation is not None
    assert report.final_recommendation.objective is (
        RecommendationObjective.IMPROVE_PREDICTED_QUALITY
    )
    assert report.model_performance_assessment is not None
    assert (
        report.model_performance_assessment.status
        is ModelPerformanceAcceptanceStatus.ACCEPTABLE
    )
    assert report.metadata["model_performance_assessed"] is True
    assert report.metadata["model_performance_gate_bypassed"] is False


def test_run_reduce_anomaly_score(reduce_outcome: AnalysisWorkflowOutcome) -> None:
    report = reduce_outcome.report
    assert report.status in _HAPPY_STATUSES
    assert report.final_recommendation is not None
    assert report.final_recommendation.objective is (
        RecommendationObjective.REDUCE_ANOMALY_SCORE
    )
    assert report.final_recommendation.status in _HAPPY_RECOMMENDATION_STATUSES


def test_run_balance_quality_and_anomaly(
    balance_outcome: AnalysisWorkflowOutcome,
) -> None:
    report = balance_outcome.report
    assert report.status in _HAPPY_STATUSES
    assert report.final_recommendation is not None
    assert report.final_recommendation.objective is (
        RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY
    )


# ---------------------------------------------------------------------------
# 5-6. Canonical stage order and completeness
# ---------------------------------------------------------------------------


def test_canonical_stage_order(quality_outcome: AnalysisWorkflowOutcome) -> None:
    report = quality_outcome.report
    assert report.status in _HAPPY_STATUSES
    executed_stages = [
        record.stage for record in report.stage_records if record.executed
    ]
    expected = [
        stage
        for stage in AnalysisWorkflowStage
        if stage is not AnalysisWorkflowStage.COHORT_FILTER
    ]
    assert executed_stages == expected
    cohort_record = next(
        record
        for record in report.stage_records
        if record.stage is AnalysisWorkflowStage.COHORT_FILTER
    )
    assert cohort_record.executed is False


def test_all_stage_records_present(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    report = quality_outcome.report
    assert report.status in _HAPPY_STATUSES
    stages = {record.stage for record in report.stage_records}
    assert stages == set(AnalysisWorkflowStage)
    assert len(report.stage_records) == len(list(AnalysisWorkflowStage))
    assert all(
        record.executed or record.stage is AnalysisWorkflowStage.COHORT_FILTER
        for record in report.stage_records
    )


# ---------------------------------------------------------------------------
# 7-8. Final recommendation and terminal status
# ---------------------------------------------------------------------------


def test_final_recommendation_exists(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    assert quality_outcome.report.final_recommendation is not None


def test_status_and_recommendation_status(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    report = quality_outcome.report
    assert report.status in _HAPPY_STATUSES
    recommendation = report.final_recommendation
    assert recommendation is not None
    assert recommendation.status in _HAPPY_RECOMMENDATION_STATUSES
    if report.status is AnalysisWorkflowStatus.COMPLETED:
        assert recommendation.status is RecommendationStatus.GENERATED
        assert report.terminal_stage is AnalysisWorkflowStage.RECOMMENDATION


# ---------------------------------------------------------------------------
# 9-10. Row-count consistency
# ---------------------------------------------------------------------------


def test_row_count_consistency(quality_outcome: AnalysisWorkflowOutcome) -> None:
    report = quality_outcome.report
    assert report.raw_row_count == N_ROWS
    assert report.processed_row_count == N_ROWS
    assert report.cohort_row_count == N_ROWS


def test_split_row_counts_sum_to_processed(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    report = quality_outcome.report
    total = (
        report.train_row_count
        + report.validation_row_count
        + report.test_row_count
    )
    assert total == report.cohort_row_count
    assert report.train_row_count > 0
    assert report.validation_row_count > 0
    assert report.test_row_count > 0


# ---------------------------------------------------------------------------
# 11-12. Routing selections
# ---------------------------------------------------------------------------


def test_selected_industry_is_semiconductor(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    assert quality_outcome.report.selected_industry == "semiconductor"


def test_selected_task_is_regression(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    assert quality_outcome.report.selected_task is AnalysisTask.REGRESSION


# ---------------------------------------------------------------------------
# 13-14. Model keys present
# ---------------------------------------------------------------------------


def test_model_keys_present(quality_outcome: AnalysisWorkflowOutcome) -> None:
    report = quality_outcome.report
    assert isinstance(report.selected_supervised_model_key, str)
    assert report.selected_supervised_model_key.strip() != ""
    assert isinstance(report.selected_anomaly_model_key, str)
    assert report.selected_anomaly_model_key.strip() != ""


# ---------------------------------------------------------------------------
# 15-16. Independent test evaluation and residual calibration
# ---------------------------------------------------------------------------


def test_evaluation_and_calibration_metadata(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    metadata = quality_outcome.report.metadata
    assert metadata["independent_test_evaluation_performed"] is True
    assert metadata["residual_calibration_performed"] is True


# ---------------------------------------------------------------------------
# 17-18. Anomaly-event and diagnosis-factor counts
# ---------------------------------------------------------------------------


def test_anomaly_event_and_diagnosis_factor_counts(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    report = quality_outcome.report
    assert report.anomaly_event_count >= 1
    assert report.diagnosis_factor_count >= 1


# ---------------------------------------------------------------------------
# 19. Recommendation pipeline executed
# ---------------------------------------------------------------------------


def test_recommendation_pipeline_executed(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    assert quality_outcome.report.metadata["recommendation_pipeline_executed"] is True


# ---------------------------------------------------------------------------
# 20-21. Finite anomaly scores (no clipping / abs) on recommendation result
# ---------------------------------------------------------------------------


def test_recommendation_anomaly_scores_finite(
    reduce_outcome: AnalysisWorkflowOutcome,
) -> None:
    recommendation = reduce_outcome.report.final_recommendation
    assert recommendation is not None
    for score in (
        recommendation.baseline_anomaly_score,
        recommendation.proposed_anomaly_score,
    ):
        if score is not None:
            # Sign is preserved by contract; only require finiteness here.
            assert math.isfinite(score)


# ---------------------------------------------------------------------------
# 22-23. Test partition never used for selection / calibration
# ---------------------------------------------------------------------------


def test_test_partition_not_used_for_selection_or_calibration(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    metadata = quality_outcome.report.metadata
    assert metadata["test_used_for_model_selection"] is False
    assert metadata["test_used_for_threshold_calibration"] is False


# ---------------------------------------------------------------------------
# 24. Row identity preserved
# ---------------------------------------------------------------------------


def test_row_identity_preserved(quality_outcome: AnalysisWorkflowOutcome) -> None:
    assert quality_outcome.report.metadata["row_identity_preserved"] is True


# ---------------------------------------------------------------------------
# 25-27. Generated recommendation relationships
# ---------------------------------------------------------------------------


def test_generated_recommendation_relationships(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    report = quality_outcome.report
    recommendation = report.final_recommendation
    assert recommendation is not None
    if recommendation.status is not RecommendationStatus.GENERATED:
        pytest.skip("Recommendation did not reach GENERATED on this run.")

    assert recommendation.changes
    for change in recommendation.changes:
        # current / proposed / delta relationship
        assert change.delta == pytest.approx(
            change.proposed_value - change.current_value
        )
        # proposed value stays within the requested constraint bounds
        low, high = CONSTRAINT_BOUNDS[change.variable]
        assert low <= change.proposed_value <= high
        _assert_finite(change.confidence, label="change confidence")
        # non-causal rationale
        rationale_lower = change.rationale.lower()
        for token in _FORBIDDEN_OUTCOME_PATTERNS:
            assert token not in rationale_lower

    # Result-level disclaimer must be present (association, not causation).
    assert recommendation.disclaimer.strip() != ""
    disclaimer_lower = recommendation.safety_decision.disclaimer.lower()
    assert "association" in disclaimer_lower
    assert "caus" in disclaimer_lower


# ---------------------------------------------------------------------------
# 28. Validation blocker refusal
# ---------------------------------------------------------------------------


def test_validation_blocker_refuses(tmp_path: Path) -> None:
    csv_path = write_invalid_csv(tmp_path / "invalid.csv")
    outcome = _run(csv_path, RecommendationObjective.IMPROVE_PREDICTED_QUALITY)
    report = outcome.report
    assert report.status is AnalysisWorkflowStatus.REFUSED
    assert report.terminal_stage is AnalysisWorkflowStage.VALIDATE
    terminal = report.stage_records[-1]
    assert terminal.stage is AnalysisWorkflowStage.VALIDATE
    assert terminal.structured_refusal is True
    assert terminal.succeeded is False
    assert report.final_recommendation is None
    assert report.model_performance_assessment is None
    assert report.metadata["model_performance_assessed"] is False
    assert report.metadata["model_performance_gate_bypassed"] is False


# ---------------------------------------------------------------------------
# 29. Leakage-check path is exercised on the happy path
# ---------------------------------------------------------------------------


def test_leakage_check_stage_succeeds(
    quality_outcome: AnalysisWorkflowOutcome,
    workflow: IndustrialProcessAnalysisWorkflow,
) -> None:
    # Triggering a leakage BLOCKER through the public request schema is not
    # possible (the request already forbids target/identifier columns as
    # features). Instead, assert the LEAKAGE_CHECK stage ran and passed, and
    # that the leakage-blocker policy flag is surfaced in metadata.
    report = quality_outcome.report
    leakage_records = [
        record
        for record in report.stage_records
        if record.stage is AnalysisWorkflowStage.LEAKAGE_CHECK
    ]
    assert len(leakage_records) == 1
    assert leakage_records[0].executed is True
    assert leakage_records[0].succeeded is True
    assert leakage_records[0].metadata.get("is_safe") is True
    assert workflow.get_metadata()["stop_on_leakage_blocker"] is True


# ---------------------------------------------------------------------------
# 30. Unsupported task refusal
# ---------------------------------------------------------------------------


def test_unsupported_task_refuses(tmp_path: Path) -> None:
    csv_path = write_classification_csv(tmp_path / "classification.csv")
    outcome = _run(csv_path, RecommendationObjective.REDUCE_ANOMALY_SCORE)
    report = outcome.report
    assert report.status is AnalysisWorkflowStatus.REFUSED
    assert report.terminal_stage is AnalysisWorkflowStage.TASK_ROUTING
    assert report.selected_task is not AnalysisTask.REGRESSION
    assert report.inferred_task is AnalysisTask.CLASSIFICATION
    assert report.task_selection_source is TaskSelectionSource.ROUTER
    assert report.task_override_applied is False
    assert report.stage_records[-1].structured_refusal is True


def test_auto_continuous_target_uses_router_regression(
    synthetic_csv: Path,
) -> None:
    outcome = _run(synthetic_csv, RecommendationObjective.REDUCE_ANOMALY_SCORE)
    report = outcome.report
    assert report.inferred_task is AnalysisTask.REGRESSION
    assert report.selected_task is AnalysisTask.REGRESSION
    assert report.task_selection_source is TaskSelectionSource.ROUTER
    assert report.task_override_applied is False
    assert report.metadata["inferred_task"] == "REGRESSION"
    assert report.metadata["task_selection_source"] == "ROUTER"
    assert report.metadata["task_override_applied"] is False


def test_auto_low_cardinality_numeric_records_router_inference(
    tmp_path: Path,
) -> None:
    csv_path = write_soh_like_csv(tmp_path / "soh_like.csv")
    before = csv_path.read_bytes()
    outcome = _run(
        csv_path,
        RecommendationObjective.REDUCE_ANOMALY_SCORE,
        target_column="SOH",
        column_role_overrides={
            **{name: ColumnRole.CONTROLLABLE_PROCESS for name in PROCESS_FEATURES},
            "SOH": ColumnRole.TARGET_QUALITY,
        },
    )
    report = outcome.report
    assert report.inferred_task is AnalysisTask.CLASSIFICATION
    assert report.selected_task is AnalysisTask.CLASSIFICATION
    assert report.task_selection_source is TaskSelectionSource.ROUTER
    assert report.task_override_applied is False
    assert report.status is AnalysisWorkflowStatus.REFUSED
    assert report.terminal_stage is AnalysisWorkflowStage.TASK_ROUTING
    assert csv_path.read_bytes() == before


def test_explicit_regression_overrides_inferred_classification(
    tmp_path: Path,
) -> None:
    csv_path = write_soh_like_csv(tmp_path / "soh_override.csv")
    before = pl.read_csv(csv_path)
    soh_before = before.get_column("SOH").to_list()
    dtype_before = before.schema["SOH"]
    outcome = _run(
        csv_path,
        RecommendationObjective.REDUCE_ANOMALY_SCORE,
        target_column="SOH",
        requested_task=AnalysisTask.REGRESSION,
        column_role_overrides={
            **{name: ColumnRole.CONTROLLABLE_PROCESS for name in PROCESS_FEATURES},
            "SOH": ColumnRole.TARGET_QUALITY,
        },
    )
    report = outcome.report
    assert report.inferred_task is AnalysisTask.CLASSIFICATION
    assert report.selected_task is AnalysisTask.REGRESSION
    assert report.task_selection_source is TaskSelectionSource.USER_OVERRIDE
    assert report.task_override_applied is True
    assert report.metadata["inferred_task"] == "CLASSIFICATION"
    assert report.metadata["selected_task"] == "REGRESSION"
    assert report.metadata["task_selection_source"] == "USER_OVERRIDE"
    assert report.metadata["task_override_applied"] is True
    override_warning = (
        "The task router inferred CLASSIFICATION, while the user explicitly "
        "selected REGRESSION. The explicit selection was used."
    )
    assert override_warning in report.warnings
    task_stage = next(
        record
        for record in report.stage_records
        if record.stage is AnalysisWorkflowStage.TASK_ROUTING
    )
    assert override_warning in task_stage.warnings
    assert report.status is not AnalysisWorkflowStatus.REFUSED or (
        report.terminal_stage is not AnalysisWorkflowStage.TASK_ROUTING
    )
    executed_stages = {
        record.stage for record in report.stage_records if record.executed
    }
    assert AnalysisWorkflowStage.SPLIT in executed_stages
    assert AnalysisWorkflowStage.SUPERVISED_SCREENING in executed_stages
    after = pl.read_csv(csv_path)
    assert after.get_column("SOH").to_list() == soh_before
    assert after.schema["SOH"] == dtype_before


def test_explicit_classification_refuses_without_silent_regression(
    tmp_path: Path,
) -> None:
    csv_path = write_synthetic_csv(tmp_path / "explicit_cls.csv")
    outcome = _run(
        csv_path,
        RecommendationObjective.REDUCE_ANOMALY_SCORE,
        requested_task=AnalysisTask.CLASSIFICATION,
    )
    report = outcome.report
    assert report.status is AnalysisWorkflowStatus.REFUSED
    assert report.terminal_stage is AnalysisWorkflowStage.TASK_ROUTING
    assert report.selected_task is AnalysisTask.CLASSIFICATION
    assert report.task_selection_source is TaskSelectionSource.USER_OVERRIDE
    assert report.task_override_applied is True
    assert report.selected_task is not AnalysisTask.REGRESSION
    assert "not yet supported" in report.stage_records[-1].message.lower()
    executed = [record.stage for record in report.stage_records if record.executed]
    assert AnalysisWorkflowStage.SPLIT not in executed


def test_explicit_regression_rejects_nonnumeric_target(tmp_path: Path) -> None:
    rows = _build_synthetic_rows(n_rows=N_ROWS)
    for row in rows:
        row[TARGET_COLUMN] = "pass" if float(row[TARGET_COLUMN]) >= 80.0 else "fail"
    frame = pl.DataFrame(rows).select(ALL_CSV_COLUMNS)
    csv_path = tmp_path / "string_target.csv"
    frame.write_csv(csv_path)
    before = frame.get_column(TARGET_COLUMN).to_list()
    outcome = _run(
        csv_path,
        RecommendationObjective.REDUCE_ANOMALY_SCORE,
        requested_task=AnalysisTask.REGRESSION,
    )
    report = outcome.report
    assert report.status is AnalysisWorkflowStatus.REFUSED
    assert report.terminal_stage is AnalysisWorkflowStage.TASK_ROUTING
    assert report.selected_task is AnalysisTask.REGRESSION
    assert report.task_override_applied is True
    assert "numeric target" in report.stage_records[-1].message.lower()
    after = pl.read_csv(csv_path)
    assert after.get_column(TARGET_COLUMN).to_list() == before
    assert AnalysisWorkflowStage.SPLIT not in {
        record.stage for record in report.stage_records if record.executed
    }
    assert report.stage_records[-1].metadata.get("target_refusal_code") == (
        "TARGET_NON_NUMERIC_FOR_REGRESSION"
    )


# ---------------------------------------------------------------------------
# 30B. Constant target safety gate
# ---------------------------------------------------------------------------


def _assert_no_supervised_stages(report: AnalysisWorkflowReport) -> None:
    executed = {record.stage for record in report.stage_records if record.executed}
    for stage in (
        AnalysisWorkflowStage.SPLIT,
        AnalysisWorkflowStage.SUPERVISED_SCREENING,
        AnalysisWorkflowStage.SUPERVISED_FINAL_EVALUATION,
        AnalysisWorkflowStage.RESIDUAL_CALIBRATION,
        AnalysisWorkflowStage.DIAGNOSIS,
        AnalysisWorkflowStage.RECOMMENDATION,
    ):
        assert stage not in executed


def test_constant_soh_auto_refuses_before_split(tmp_path: Path) -> None:
    csv_path = write_constant_soh_csv(tmp_path / "constant_soh_auto.csv")
    before = pl.read_csv(csv_path)
    soh_before = before.get_column("SOH").to_list()
    dtype_before = before.schema["SOH"]
    outcome = _run(
        csv_path,
        RecommendationObjective.REDUCE_ANOMALY_SCORE,
        target_column="SOH",
        column_role_overrides={
            **{name: ColumnRole.CONTROLLABLE_PROCESS for name in PROCESS_FEATURES},
            "SOH": ColumnRole.TARGET_QUALITY,
        },
    )
    report = outcome.report
    assert report.status is AnalysisWorkflowStatus.REFUSED
    assert report.terminal_stage is AnalysisWorkflowStage.VALIDATE
    assert report.stage_records[-1].structured_refusal is True
    assert report.stage_records[-1].metadata["target_refusal_code"] == "TARGET_CONSTANT"
    assert report.stage_records[-1].metadata["target_unique_non_null_count"] == 1
    assert report.metadata["target_refusal_code"] == "TARGET_CONSTANT"
    assert report.metadata["target_unique_non_null_count"] == 1
    assert report.model_performance_assessment is None
    _assert_no_supervised_stages(report)
    after = pl.read_csv(csv_path)
    assert after.get_column("SOH").to_list() == soh_before
    assert after.schema["SOH"] == dtype_before
    assert all(value == 0 for value in soh_before)


def test_constant_soh_explicit_regression_cannot_bypass(tmp_path: Path) -> None:
    csv_path = write_constant_soh_csv(tmp_path / "constant_soh_explicit.csv")
    before = pl.read_csv(csv_path)
    soh_before = before.get_column("SOH").to_list()
    outcome = _run(
        csv_path,
        RecommendationObjective.REDUCE_ANOMALY_SCORE,
        target_column="SOH",
        requested_task=AnalysisTask.REGRESSION,
        column_role_overrides={
            **{name: ColumnRole.CONTROLLABLE_PROCESS for name in PROCESS_FEATURES},
            "SOH": ColumnRole.TARGET_QUALITY,
        },
    )
    report = outcome.report
    assert report.status is AnalysisWorkflowStatus.REFUSED
    assert report.terminal_stage is AnalysisWorkflowStage.VALIDATE
    assert report.stage_records[-1].metadata["target_refusal_code"] == "TARGET_CONSTANT"
    assert "constant" in report.stage_records[-1].message.lower()
    assert AnalysisWorkflowStage.TASK_ROUTING not in {
        record.stage for record in report.stage_records if record.executed
    }
    _assert_no_supervised_stages(report)
    after = pl.read_csv(csv_path)
    assert after.get_column("SOH").to_list() == soh_before


def test_varying_soh_explicit_regression_still_progresses(tmp_path: Path) -> None:
    csv_path = write_soh_like_csv(tmp_path / "varying_soh.csv")
    outcome = _run(
        csv_path,
        RecommendationObjective.REDUCE_ANOMALY_SCORE,
        target_column="SOH",
        requested_task=AnalysisTask.REGRESSION,
        column_role_overrides={
            **{name: ColumnRole.CONTROLLABLE_PROCESS for name in PROCESS_FEATURES},
            "SOH": ColumnRole.TARGET_QUALITY,
        },
    )
    report = outcome.report
    assert report.metadata["target_suitable"] is True
    assert report.metadata["target_unique_non_null_count"] == 5
    executed = {record.stage for record in report.stage_records if record.executed}
    assert AnalysisWorkflowStage.SPLIT in executed
    assert AnalysisWorkflowStage.SUPERVISED_SCREENING in executed


# ---------------------------------------------------------------------------
# 31. No constraints / no confirmed variables refusal
# ---------------------------------------------------------------------------


def test_no_constraints_refuses_at_recommendation(synthetic_csv: Path) -> None:
    outcome = _run(
        synthetic_csv,
        RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        request_constraints=[],
        user_confirmed_controllable_variables=[],
        user_verified_variables=[],
    )
    report = outcome.report
    assert report.status is AnalysisWorkflowStatus.REFUSED
    assert report.terminal_stage is AnalysisWorkflowStage.RECOMMENDATION
    assert report.stage_records[-1].structured_refusal is True
    assert report.final_recommendation is None


# ---------------------------------------------------------------------------
# 32-35. Operating-point selection modes
# ---------------------------------------------------------------------------


def test_operating_point_explicit_row_id(
    quality_outcome: AnalysisWorkflowOutcome,
    explicit_outcome: AnalysisWorkflowOutcome,
) -> None:
    expected_row_id = quality_outcome.report.selected_operating_row_id
    report = explicit_outcome.report
    assert report.status in _HAPPY_STATUSES
    assert report.selected_operating_row_id == expected_row_id


def test_operating_point_top_residual_default(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    # The shared happy-path fixture uses TOP_RESIDUAL_ANOMALY by default.
    report = quality_outcome.report
    assert report.status in _HAPPY_STATUSES
    assert isinstance(report.selected_operating_row_id, int)


@pytest.mark.parametrize(
    "mode",
    [
        OperatingPointSelectionMode.TOP_UNSUPERVISED_ANOMALY,
        OperatingPointSelectionMode.LATEST_ROW,
    ],
)
def test_operating_point_alternate_modes(
    synthetic_csv: Path,
    mode: OperatingPointSelectionMode,
) -> None:
    outcome = _run(
        synthetic_csv,
        RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        operating_point_selection=mode,
    )
    report = outcome.report
    assert report.status in _HAPPY_STATUSES
    assert isinstance(report.selected_operating_row_id, int)


# ---------------------------------------------------------------------------
# 36-37. Warning aggregation order and maximum
# ---------------------------------------------------------------------------


def test_warning_aggregation_respects_limit(synthetic_csv: Path) -> None:
    outcome = _run(
        synthetic_csv,
        RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        policy_overrides={
            "include_stage_warnings": True,
            "maximum_aggregated_warnings": 1,
        },
    )
    warnings = outcome.report.warnings
    assert len(warnings) <= 1
    # Aggregated warnings preserve order and contain no duplicates.
    assert list(dict.fromkeys(warnings)) == warnings


# ---------------------------------------------------------------------------
# 38-40. Immutability of request, constraints, and CSV file
# ---------------------------------------------------------------------------


def test_inputs_are_not_mutated(synthetic_csv: Path) -> None:
    request = build_request(
        synthetic_csv, RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY
    )
    request_snapshot = request.model_dump()
    constraints_snapshot = [c.model_dump() for c in request.request_constraints]
    csv_bytes_before = synthetic_csv.read_bytes()

    workflow = _make_workflow()
    outcome = workflow.run(request)

    assert isinstance(outcome, AnalysisWorkflowOutcome)
    assert request.model_dump() == request_snapshot
    assert [c.model_dump() for c in request.request_constraints] == (
        constraints_snapshot
    )
    assert synthetic_csv.read_bytes() == csv_bytes_before


# ---------------------------------------------------------------------------
# 41-44. Determinism, instance isolation, and statelessness
# ---------------------------------------------------------------------------


def test_determinism_and_state_isolation(
    synthetic_csv: Path,
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    # A separate workflow instance produces an identical business result,
    # demonstrating determinism, per-instance state isolation, and the absence
    # of cross-run caching. Models are stable after recommendation.
    other_workflow = _make_workflow()
    other_outcome = other_workflow.run(
        build_request(synthetic_csv, RecommendationObjective.IMPROVE_PREDICTED_QUALITY)
    )
    assert _business_fingerprint(other_outcome.report) == _business_fingerprint(
        quality_outcome.report
    )

    # Running the same instance again yields the same business fingerprint.
    repeat_outcome = other_workflow.run(
        build_request(synthetic_csv, RecommendationObjective.IMPROVE_PREDICTED_QUALITY)
    )
    assert _business_fingerprint(repeat_outcome.report) == _business_fingerprint(
        other_outcome.report
    )


# ---------------------------------------------------------------------------
# 45+. ANOMALY_ONLY analysis mode (battery-like fixture, constant SOH)
# ---------------------------------------------------------------------------


def test_anomaly_only_succeeds_not_refused(
    anomaly_outcome: AnalysisWorkflowOutcome,
) -> None:
    report = anomaly_outcome.report
    assert report.status is not AnalysisWorkflowStatus.REFUSED
    assert report.status is AnalysisWorkflowStatus.PARTIAL
    assert report.analysis_mode is AnalysisExecutionMode.ANOMALY_ONLY


def test_anomaly_only_soh_not_used_as_target(battery_csv: Path) -> None:
    # SOH is constant zero in the fixture; ANOMALY_ONLY analysis must not
    # require it (or any target) and must not be blocked by it.
    before = pl.read_csv(battery_csv)
    soh_values = before.get_column(BATTERY_TARGET_COLUMN).to_list()
    assert all(value == 0 for value in soh_values)


def test_anomaly_only_target_column_none(
    anomaly_outcome: AnalysisWorkflowOutcome,
) -> None:
    report = anomaly_outcome.report
    assert report.metadata["target_column"] is None


def test_anomaly_only_task_routing_not_executed(
    anomaly_outcome: AnalysisWorkflowOutcome,
) -> None:
    executed = {
        record.stage for record in anomaly_outcome.report.stage_records if record.executed
    }
    assert AnalysisWorkflowStage.TASK_ROUTING not in executed
    routing_record = next(
        record
        for record in anomaly_outcome.report.stage_records
        if record.stage is AnalysisWorkflowStage.TASK_ROUTING
    )
    assert routing_record.executed is False


def test_anomaly_only_no_target_suitability_blocking(
    anomaly_outcome: AnalysisWorkflowOutcome,
) -> None:
    report = anomaly_outcome.report
    assert report.metadata["target_suitable"] is None
    assert report.metadata["target_suitability_message"] == "NOT_APPLICABLE"
    validate_record = next(
        record
        for record in report.stage_records
        if record.stage is AnalysisWorkflowStage.VALIDATE
    )
    assert validate_record.executed is True
    assert validate_record.succeeded is True
    assert validate_record.structured_refusal is False
    assert validate_record.metadata["target_suitability"] == "NOT_APPLICABLE"


def test_anomaly_only_split_performed(
    anomaly_outcome: AnalysisWorkflowOutcome,
) -> None:
    report = anomaly_outcome.report
    split_record = next(
        record
        for record in report.stage_records
        if record.stage is AnalysisWorkflowStage.SPLIT
    )
    assert split_record.executed is True
    assert split_record.succeeded is True
    assert report.train_row_count > 0
    assert report.validation_row_count > 0
    assert report.test_row_count > 0


def test_anomaly_only_preprocessing_train_only(
    anomaly_outcome: AnalysisWorkflowOutcome,
) -> None:
    # Preprocessing being fit on the training partition only is enforced by
    # the pipeline's leakage checker; here we confirm the observable
    # contracts: disjoint partition sizes and no test-partition usage for
    # selection or calibration.
    report = anomaly_outcome.report
    total = (
        report.train_row_count + report.validation_row_count + report.test_row_count
    )
    assert total == report.cohort_row_count
    leakage_record = next(
        record
        for record in report.stage_records
        if record.stage is AnalysisWorkflowStage.LEAKAGE_CHECK
    )
    assert leakage_record.succeeded is True
    assert leakage_record.metadata["is_safe"] is True
    assert report.metadata["test_used_for_model_selection"] is False
    assert report.metadata["test_used_for_threshold_calibration"] is False


def test_anomaly_only_supervised_stages_not_executed(
    anomaly_outcome: AnalysisWorkflowOutcome,
) -> None:
    executed = {
        record.stage for record in anomaly_outcome.report.stage_records if record.executed
    }
    assert AnalysisWorkflowStage.SUPERVISED_SCREENING not in executed
    assert AnalysisWorkflowStage.SUPERVISED_FINAL_EVALUATION not in executed
    assert anomaly_outcome.report.selected_supervised_model_key is None
    assert anomaly_outcome.report.model_performance_assessment is None


def test_anomaly_only_residual_stages_not_executed(
    anomaly_outcome: AnalysisWorkflowOutcome,
) -> None:
    executed = {
        record.stage for record in anomaly_outcome.report.stage_records if record.executed
    }
    assert AnalysisWorkflowStage.RESIDUAL_CALIBRATION not in executed
    assert AnalysisWorkflowStage.RESIDUAL_FINAL_EVALUATION not in executed
    assert anomaly_outcome.report.metadata["residual_calibration_performed"] is False


def test_anomaly_only_anomaly_stages_executed(
    anomaly_outcome: AnalysisWorkflowOutcome,
) -> None:
    executed = {
        record.stage for record in anomaly_outcome.report.stage_records if record.executed
    }
    assert AnalysisWorkflowStage.ANOMALY_SCREENING in executed
    assert AnalysisWorkflowStage.ANOMALY_FINAL_EVALUATION in executed
    report = anomaly_outcome.report
    assert isinstance(report.selected_anomaly_model_key, str)
    assert report.selected_anomaly_model_key.strip() != ""


def test_anomaly_only_independent_test_evaluation_performed(
    anomaly_outcome: AnalysisWorkflowOutcome,
) -> None:
    assert (
        anomaly_outcome.report.metadata["independent_test_evaluation_performed"]
        is True
    )


def test_anomaly_only_event_selection_source(
    anomaly_outcome: AnalysisWorkflowOutcome,
) -> None:
    report = anomaly_outcome.report
    selection_record = next(
        record
        for record in report.stage_records
        if record.stage is AnalysisWorkflowStage.ANOMALY_EVENT_SELECTION
    )
    assert selection_record.executed is True
    assert selection_record.succeeded is True
    assert selection_record.metadata["selection_source"] == "UNSUPERVISED_ANOMALY_SCORE"
    assert report.metadata["anomaly_event_selection_source"] == (
        "UNSUPERVISED_ANOMALY_SCORE"
    )
    assert report.anomaly_event_count >= 1


def test_anomaly_only_diagnosis_executed_robust_source(
    anomaly_outcome: AnalysisWorkflowOutcome,
) -> None:
    report = anomaly_outcome.report
    diagnosis_record = next(
        record
        for record in report.stage_records
        if record.stage is AnalysisWorkflowStage.DIAGNOSIS
    )
    assert diagnosis_record.executed is True
    assert diagnosis_record.succeeded is True
    assert diagnosis_record.metadata["diagnosis_source"] == "ROBUST_GROUP_COMPARISON"
    assert diagnosis_record.metadata["target_based_diagnosis"] is False
    assert diagnosis_record.metadata["association_not_causation"] is True
    assert report.metadata["diagnosis_source"] == "ROBUST_GROUP_COMPARISON"
    assert report.diagnosis_factor_count >= 1


def test_anomaly_only_recommendation_pipeline_not_executed(
    anomaly_outcome: AnalysisWorkflowOutcome,
) -> None:
    report = anomaly_outcome.report
    assert report.metadata["recommendation_pipeline_executed"] is False
    assert report.final_recommendation is None
    recommendation_record = next(
        record
        for record in report.stage_records
        if record.stage is AnalysisWorkflowStage.RECOMMENDATION
    )
    assert recommendation_record.executed is False
    assert recommendation_record.metadata["recommendation_applicable"] is False
    assert report.terminal_stage is AnalysisWorkflowStage.DIAGNOSIS


def test_anomaly_only_row_identity_preserved(
    anomaly_outcome: AnalysisWorkflowOutcome,
) -> None:
    assert anomaly_outcome.report.metadata["row_identity_preserved"] is True


def test_anomaly_only_csv_unchanged_after_run(battery_csv: Path) -> None:
    before = pl.read_csv(battery_csv)
    _run_anomaly(battery_csv)
    after = pl.read_csv(battery_csv)
    assert after.equals(before)
    for column, dtype in before.schema.items():
        assert after.schema[column] == dtype


def test_anomaly_only_determinism(battery_csv: Path) -> None:
    first = _run_anomaly(battery_csv).report
    second = _run_anomaly(battery_csv).report
    assert first.selected_anomaly_model_key == second.selected_anomaly_model_key
    assert first.anomaly_event_count == second.anomaly_event_count
    assert first.diagnosis_factor_count == second.diagnosis_factor_count
    assert first.status is second.status


def test_supervised_full_workflow_regression_still_passes(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    # Guards against ANOMALY_ONLY changes silently affecting the default
    # SUPERVISED path exercised throughout the rest of this module.
    report = quality_outcome.report
    assert report.analysis_mode is AnalysisExecutionMode.SUPERVISED
    assert report.status in _HAPPY_STATUSES
    assert report.final_recommendation is not None
    assert report.model_performance_assessment is not None


# ---------------------------------------------------------------------------
# Anomaly context windows (Step 11B.9)
# ---------------------------------------------------------------------------

ORDERED_BATTERY_N_ROWS = 90
ORDERED_BATTERY_FEATURES = [
    "RSOCmin",
    "RSOCmax",
    "RSOCavg",
    "ChgPmax",
    "CellV01",
    "CellV02",
    "current",
]
# TIME split puts the last 20% in test (~rows 72-89). Keep a clustered set
# there with enough surrounding normal rows for robust group comparison.
ORDERED_BATTERY_OUTLIER_IDS = (80, 82, 83, 85, 86)


def write_ordered_battery_context_csv(path: Path) -> Path:
    """Write a 90-row ordered battery fixture with clustered late outliers."""
    rows: list[dict[str, Any]] = []
    for i in range(ORDERED_BATTERY_N_ROWS):
        # Keep the background regime stable so TIME-split test rows are not
        # wholesale distribution-shift outliers relative to training.
        rsoc = float(70.0 + (i % 5) * 0.2)
        voltage = float(3.65 + (i % 7) * 0.01)
        current = -1.5 if (20 <= i < 60) else (1.0 if i >= 60 else 0.0)
        chg = 100.0 if current > 0 else 0.0
        if i in ORDERED_BATTERY_OUTLIER_IDS:
            rsoc = 95.0 - (i - 80) * 2.0
            voltage = 3.20
            current = -8.5
            chg = 0.0
        rows.append(
            {
                BATTERY_ID_COLUMN: f"CELL-{i:03d}",
                BATTERY_DATE_COLUMN: "2026-01-01",
                "timestamp": 1_700_000_000 + i * 60,
                "RSOCmin": rsoc - 0.5,
                "RSOCmax": rsoc + 0.5,
                "RSOCavg": rsoc,
                "ChgPmax": chg,
                "CellV01": voltage,
                "CellV02": voltage - 0.01,
                "current": current,
                BATTERY_TARGET_COLUMN: 0,
            }
        )
    frame = pl.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path)
    return path


@pytest.fixture(scope="module")
def ordered_battery_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("ordered_battery") / "ordered_battery_context.csv"
    return write_ordered_battery_context_csv(path)


@pytest.fixture(scope="module")
def ordered_battery_outcome(ordered_battery_csv: Path) -> AnalysisWorkflowOutcome:
    request = build_anomaly_request(
        ordered_battery_csv,
        feature_columns=list(ORDERED_BATTERY_FEATURES),
        identifier_columns=[BATTERY_ID_COLUMN],
        timestamp_column="timestamp",
        excluded_columns=[BATTERY_DATE_COLUMN, BATTERY_TARGET_COLUMN],
    )
    workflow = IndustrialProcessAnalysisWorkflow(policy=_anomaly_policy())
    return workflow.run(request)


def test_anomaly_context_windows_created_for_events(
    ordered_battery_outcome: AnalysisWorkflowOutcome,
) -> None:
    report = ordered_battery_outcome.report
    assert report.anomaly_event_count >= 1
    assert len(report.anomaly_context_windows) == len(report.anomaly_events)
    for window in report.anomaly_context_windows:
        assert window.radius == ANOMALY_CONTEXT_RADIUS
        assert window.order_basis is AnomalyContextOrderBasis.SORTED_ANALYSIS_ORDER
        assert 1 <= len(window.rows) <= (2 * ANOMALY_CONTEXT_RADIUS + 1)
        centers = [row for row in window.rows if row.is_center_event]
        assert len(centers) == 1
        assert centers[0].relative_offset == 0
        assert centers[0].original_row_id == window.center_original_row_id
        assert len(window.feature_names) <= ANOMALY_CONTEXT_MAX_FEATURES
        assert window.feature_names == [
            factor.variable
            for factor in report.diagnosis_factors
            if factor.variable in set(ORDERED_BATTERY_FEATURES)
        ][:ANOMALY_CONTEXT_MAX_FEATURES]


def test_anomaly_context_loaded_order_without_timestamp(battery_csv: Path) -> None:
    outcome = _run_anomaly(battery_csv)
    report = outcome.report
    assert report.anomaly_context_windows
    assert all(
        window.order_basis is AnomalyContextOrderBasis.LOADED_ROW_ORDER
        for window in report.anomaly_context_windows
    )


def test_anomaly_context_preserves_raw_values_and_boundaries(
    ordered_battery_csv: Path,
    ordered_battery_outcome: AnalysisWorkflowOutcome,
) -> None:
    raw = pl.read_csv(ordered_battery_csv)
    report = ordered_battery_outcome.report
    assert report.anomaly_context_windows
    window = report.anomaly_context_windows[0]
    offsets = [row.relative_offset for row in window.rows]
    assert 0 in offsets
    assert min(offsets) >= -ANOMALY_CONTEXT_RADIUS
    assert max(offsets) <= ANOMALY_CONTEXT_RADIUS
    center = next(row for row in window.rows if row.is_center_event)
    raw_row = raw.row(int(center.original_row_id), named=True)
    for feature in window.feature_names:
        context_value = next(
            item.value for item in center.feature_values if item.feature_name == feature
        )
        expected = raw_row[feature]
        if expected is None:
            assert context_value is None
        else:
            assert context_value == pytest.approx(float(expected))
    after = pl.read_csv(ordered_battery_csv)
    assert after.equals(raw)


def test_anomaly_context_overlapping_windows_allowed(
    ordered_battery_outcome: AnalysisWorkflowOutcome,
) -> None:
    report = ordered_battery_outcome.report
    assert len(report.anomaly_context_windows) == len(report.anomaly_events)
    assert len(report.anomaly_context_windows) >= 2
    first_ids = {row.original_row_id for row in report.anomaly_context_windows[0].rows}
    second_ids = {row.original_row_id for row in report.anomaly_context_windows[1].rows}
    # Overlap is allowed; windows remain independent per event.
    assert first_ids
    assert second_ids
    assert report.anomaly_context_windows[0].event_rank != (
        report.anomaly_context_windows[1].event_rank
    )

def test_anomaly_context_does_not_mutate_anomaly_or_diagnosis(
    ordered_battery_csv: Path,
) -> None:
    first = IndustrialProcessAnalysisWorkflow(policy=_anomaly_policy()).run(
        build_anomaly_request(
            ordered_battery_csv,
            feature_columns=list(ORDERED_BATTERY_FEATURES),
            identifier_columns=[BATTERY_ID_COLUMN],
            timestamp_column="timestamp",
            excluded_columns=[BATTERY_DATE_COLUMN, BATTERY_TARGET_COLUMN],
        )
    ).report
    second = IndustrialProcessAnalysisWorkflow(policy=_anomaly_policy()).run(
        build_anomaly_request(
            ordered_battery_csv,
            feature_columns=list(ORDERED_BATTERY_FEATURES),
            identifier_columns=[BATTERY_ID_COLUMN],
            timestamp_column="timestamp",
            excluded_columns=[BATTERY_DATE_COLUMN, BATTERY_TARGET_COLUMN],
        )
    ).report
    assert [event.sample_id for event in first.anomaly_events] == [
        event.sample_id for event in second.anomaly_events
    ]
    assert [event.anomaly_score for event in first.anomaly_events] == [
        event.anomaly_score for event in second.anomaly_events
    ]
    assert [factor.variable for factor in first.diagnosis_factors] == [
        factor.variable for factor in second.diagnosis_factors
    ]
    assert first.model_dump(mode="json")["anomaly_context_windows"] == (
        second.model_dump(mode="json")["anomaly_context_windows"]
    )


def test_supervised_context_windows_still_generated(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    report = quality_outcome.report
    assert len(report.anomaly_context_windows) == len(report.anomaly_events)
    if report.anomaly_events:
        assert report.anomaly_context_windows[0].radius == ANOMALY_CONTEXT_RADIUS


# ---------------------------------------------------------------------------
# Step 11B.10 Explicit operating cohort filter
# ---------------------------------------------------------------------------


COHORT_FIXTURE_N_ROWS = 150
COHORT_FEATURES = [
    "RSOCavg",
    "RSOCmin",
    "RSOCmax",
    "ChgPmax",
    "M01CV05",
    "TempAvg",
]


def write_multi_regime_battery_csv(path: Path) -> Path:
    """Write a multi-regime battery-like CSV with local outliers per regime."""
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(11)
    for i in range(COHORT_FIXTURE_N_ROWS):
        if i < 50:
            base_soc = 95.0 - (i % 10) * 0.2
            regime = "high"
        elif i < 100:
            base_soc = 55.0 - (i % 10) * 0.2
            regime = "mid"
        else:
            base_soc = 15.0 - (i % 10) * 0.2
            regime = "low"
        rsoc = float(base_soc + rng.normal(0.0, 0.05))
        chg = 14.5 + rng.normal(0.0, 0.05)
        m01 = 4.05 + rng.normal(0.0, 0.01)
        temp = 25.0 + rng.normal(0.0, 0.2)
        # Inject a clear local outlier inside the high-SOC cohort only.
        if i == 20:
            chg = 40.0
            m01 = 4.8
        if i == 70:
            chg = 35.0
        if i == 120:
            chg = 30.0
        rows.append(
            {
                "SerialNumber": f"CELL-{i // 30:03d}",
                "Date": f"2024-01-{(i % 28) + 1:02d}",
                "SOH": 0.0,
                "RSOCavg": rsoc,
                "RSOCmin": rsoc - 0.5,
                "RSOCmax": rsoc + 0.5,
                "ChgPmax": float(chg),
                "M01CV05": float(m01),
                "TempAvg": float(temp),
                "regime": regime,
            }
        )
    frame = pl.DataFrame(rows)
    frame.write_csv(path)
    return path


@pytest.fixture()
def multi_regime_csv(tmp_path: Path) -> Path:
    return write_multi_regime_battery_csv(tmp_path / "multi_regime_battery.csv")


def test_cohort_filter_absent_stage_skipped(
    anomaly_outcome: AnalysisWorkflowOutcome,
) -> None:
    record = next(
        item
        for item in anomaly_outcome.report.stage_records
        if item.stage is AnalysisWorkflowStage.COHORT_FILTER
    )
    assert record.executed is False
    assert "No operating cohort filter was configured." in record.message
    assert anomaly_outcome.report.cohort_filter_summary.configured is False
    assert (
        anomaly_outcome.report.cohort_row_count
        == anomaly_outcome.report.processed_row_count
    )


def test_cohort_filter_applied_and_split_uses_filtered_rows(
    multi_regime_csv: Path,
) -> None:
    before = pl.read_csv(multi_regime_csv)
    csv_bytes = multi_regime_csv.read_bytes()
    request = build_anomaly_request(
        multi_regime_csv,
        feature_columns=list(COHORT_FEATURES),
        identifier_columns=["SerialNumber"],
        excluded_columns=["Date", "SOH", "regime"],
        cohort_filter=NumericCohortFilter(
            column_name="RSOCavg",
            lower_bound=80.0,
            upper_bound=100.0,
            exclude_filter_column_from_features=True,
        ),
    )
    outcome = _make_anomaly_workflow().run(request)
    report = outcome.report
    assert report.status is not AnalysisWorkflowStatus.REFUSED
    cohort_record = next(
        item
        for item in report.stage_records
        if item.stage is AnalysisWorkflowStage.COHORT_FILTER
    )
    assert cohort_record.executed is True
    assert cohort_record.succeeded is True
    assert report.cohort_filter_summary.configured is True
    assert report.cohort_filter_summary.column_name == "RSOCavg"
    assert report.cohort_filter_summary.lower_bound == pytest.approx(80.0)
    assert report.cohort_filter_summary.upper_bound == pytest.approx(100.0)
    assert report.raw_row_count == COHORT_FIXTURE_N_ROWS
    assert report.processed_row_count == COHORT_FIXTURE_N_ROWS
    assert report.cohort_row_count == report.cohort_filter_summary.retained_row_count
    assert report.cohort_row_count < report.processed_row_count
    split_total = (
        report.train_row_count + report.validation_row_count + report.test_row_count
    )
    assert split_total == report.cohort_row_count
    assert report.metadata["feature_count"] == len(COHORT_FEATURES) - 1
    assert report.selected_anomaly_model_key is not None
    assert report.anomaly_event_count >= 1
    assert report.diagnosis_factor_count >= 1
    # Filtered context neighbors must use cohort analysis order only.
    for window in report.anomaly_context_windows:
        for row in window.rows:
            row_id = int(row.original_row_id)
            value = float(before[row_id, "RSOCavg"])
            assert value >= 80.0
            assert value <= 100.0
    assert multi_regime_csv.read_bytes() == csv_bytes
    assert before.to_dicts() == pl.read_csv(multi_regime_csv).to_dicts()


def test_cohort_filter_keeps_filter_column_when_exclusion_disabled(
    multi_regime_csv: Path,
) -> None:
    request = build_anomaly_request(
        multi_regime_csv,
        feature_columns=list(COHORT_FEATURES),
        identifier_columns=["SerialNumber"],
        excluded_columns=["Date", "SOH", "regime"],
        cohort_filter=NumericCohortFilter(
            column_name="RSOCavg",
            lower_bound=80.0,
            upper_bound=100.0,
            exclude_filter_column_from_features=False,
        ),
    )
    report = _make_anomaly_workflow().run(request).report
    assert report.status is not AnalysisWorkflowStatus.REFUSED
    assert report.metadata["feature_count"] == len(COHORT_FEATURES)
    assert report.cohort_filter_summary.exclude_filter_column_from_features is False


def test_cohort_filter_refuses_when_no_features_remain(
    multi_regime_csv: Path,
) -> None:
    request = build_anomaly_request(
        multi_regime_csv,
        feature_columns=["RSOCavg"],
        identifier_columns=["SerialNumber"],
        excluded_columns=["Date", "SOH", "regime"],
        max_simultaneous_changes=1,
        cohort_filter=NumericCohortFilter(
            column_name="RSOCavg",
            lower_bound=80.0,
            upper_bound=100.0,
            exclude_filter_column_from_features=True,
        ),
    )
    report = _make_anomaly_workflow().run(request).report
    assert report.status is AnalysisWorkflowStatus.REFUSED
    assert report.terminal_stage is AnalysisWorkflowStage.COHORT_FILTER
    assert "no modeling features" in report.stage_records[-1].message.lower()


def test_cohort_filter_refuses_too_few_rows(multi_regime_csv: Path) -> None:
    request = build_anomaly_request(
        multi_regime_csv,
        feature_columns=list(COHORT_FEATURES),
        identifier_columns=["SerialNumber"],
        excluded_columns=["Date", "SOH", "regime"],
        cohort_filter=NumericCohortFilter(
            column_name="ChgPmax",
            lower_bound=39.0,
            upper_bound=41.0,
        ),
    )
    report = _make_anomaly_workflow().run(request).report
    assert report.status is AnalysisWorkflowStatus.REFUSED
    assert report.terminal_stage is AnalysisWorkflowStage.COHORT_FILTER
    message = report.stage_records[-1].message.lower()
    assert "too few rows" in message or "zero rows" in message

def test_cohort_filter_preserves_original_row_ids(
    multi_regime_csv: Path,
) -> None:
    before = pl.read_csv(multi_regime_csv)
    report = _make_anomaly_workflow().run(
        build_anomaly_request(
            multi_regime_csv,
            feature_columns=list(COHORT_FEATURES),
            identifier_columns=["SerialNumber"],
            excluded_columns=["Date", "SOH", "regime"],
            cohort_filter=NumericCohortFilter(
                column_name="RSOCavg",
                lower_bound=80.0,
                upper_bound=100.0,
            ),
        )
    ).report
    for event in report.anomaly_events:
        row_id = int(event.sample_id)
        assert 0 <= row_id < before.height
        assert before[row_id, "RSOCavg"] >= 80.0


def test_cohort_filter_determinism(multi_regime_csv: Path) -> None:
    request = build_anomaly_request(
        multi_regime_csv,
        feature_columns=list(COHORT_FEATURES),
        identifier_columns=["SerialNumber"],
        excluded_columns=["Date", "SOH", "regime"],
        cohort_filter=NumericCohortFilter(
            column_name="RSOCavg",
            lower_bound=80.0,
            upper_bound=100.0,
        ),
    )
    first = _make_anomaly_workflow().run(request).report
    second = _make_anomaly_workflow().run(request).report
    assert first.cohort_row_count == second.cohort_row_count
    assert first.cohort_filter_summary.model_dump() == (
        second.cohort_filter_summary.model_dump()
    )
    assert [event.sample_id for event in first.anomaly_events] == [
        event.sample_id for event in second.anomaly_events
    ]


def test_supervised_workflow_still_skips_cohort_filter(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    record = next(
        item
        for item in quality_outcome.report.stage_records
        if item.stage is AnalysisWorkflowStage.COHORT_FILTER
    )
    assert record.executed is False
    assert quality_outcome.report.cohort_filter_summary.configured is False
