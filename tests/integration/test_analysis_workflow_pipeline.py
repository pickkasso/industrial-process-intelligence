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
    AnalysisWorkflowOutcome,
    AnalysisWorkflowPolicy,
    AnalysisWorkflowReport,
    AnalysisWorkflowRequest,
    AnalysisWorkflowStage,
    AnalysisWorkflowStatus,
    IndustrialProcessAnalysisWorkflow,
    OperatingPointSelectionMode,
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
    assert executed_stages == list(AnalysisWorkflowStage)


def test_all_stage_records_present(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    report = quality_outcome.report
    assert report.status in _HAPPY_STATUSES
    stages = {record.stage for record in report.stage_records}
    assert stages == set(AnalysisWorkflowStage)
    assert len(report.stage_records) == len(list(AnalysisWorkflowStage))
    assert all(record.executed for record in report.stage_records)


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


def test_split_row_counts_sum_to_processed(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    report = quality_outcome.report
    total = (
        report.train_row_count
        + report.validation_row_count
        + report.test_row_count
    )
    assert total == report.processed_row_count
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
    assert report.stage_records[-1].structured_refusal is True


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
