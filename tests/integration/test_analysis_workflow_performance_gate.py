"""Integration tests for Step 10D model-performance acceptance gate wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from process_intelligence.core.enums import ColumnRole
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
    RecommendationSafetyPolicy,
    RecommendationStatus,
)
from process_intelligence.workflow import (
    AnalysisWorkflowOutcome,
    AnalysisWorkflowPolicy,
    AnalysisWorkflowRequest,
    AnalysisWorkflowStage,
    AnalysisWorkflowStatus,
    IndustrialProcessAnalysisWorkflow,
    OperatingPointSelectionMode,
)

CSV_NAME = "semiconductor_process_perf_gate.csv"
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
CONSTRAINT_BOUNDS: dict[str, tuple[float, float]] = {
    "chamber_temperature": (150.0, 300.0),
    "chamber_pressure": (10.0, 70.0),
    "gas_flow_rate": (60.0, 200.0),
    "rf_power": (200.0, 500.0),
    "deposition_time": (30.0, 80.0),
}


def _build_synthetic_rows(*, n_rows: int = N_ROWS) -> list[dict[str, Any]]:
    rng = np.random.default_rng(RNG_SEED)
    rows: list[dict[str, Any]] = []
    anomaly_indices = {i for i in range(7, n_rows, 15)}
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
    rows = _build_synthetic_rows(n_rows=n_rows)
    frame = pl.DataFrame(rows).select(ALL_CSV_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path)
    return path


def _lenient_policy() -> ModelPerformanceAcceptancePolicy:
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


def _strict_impossible_policy() -> ModelPerformanceAcceptancePolicy:
    return ModelPerformanceAcceptancePolicy(
        rules=[
            MetricAcceptanceRule(
                metric_name="rmse",
                direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
                threshold=0.0,
            ),
        ],
        minimum_test_rows=1,
    )


def _missing_metric_policy() -> ModelPerformanceAcceptancePolicy:
    return ModelPerformanceAcceptancePolicy(
        rules=[
            MetricAcceptanceRule(
                metric_name="nonexistent_metric",
                direction=MetricAcceptanceDirection.HIGHER_IS_BETTER,
                threshold=0.5,
            ),
        ],
        minimum_test_rows=1,
    )


def build_request(
    csv_path: Path,
    *,
    objective: RecommendationObjective = RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
    performance_policy: ModelPerformanceAcceptancePolicy | None = None,
    **overrides: Any,
) -> AnalysisWorkflowRequest:
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
        "model_performance_policy": performance_policy or _lenient_policy(),
    }
    params.update(overrides)
    return AnalysisWorkflowRequest(**params)


def _happy_workflow() -> IndustrialProcessAnalysisWorkflow:
    return IndustrialProcessAnalysisWorkflow(
        policy=AnalysisWorkflowPolicy(
            allow_partial_diagnosis_ensemble=True,
            require_semiconductor_industry=False,
            require_regression_task=True,
            require_residual_diagnosis=True,
        )
    )


def _run(
    csv_path: Path,
    *,
    performance_policy: ModelPerformanceAcceptancePolicy | None = None,
    objective: RecommendationObjective = RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
    **request_overrides: Any,
) -> AnalysisWorkflowOutcome:
    return _happy_workflow().run(
        build_request(
            csv_path,
            objective=objective,
            performance_policy=performance_policy,
            **request_overrides,
        )
    )


@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("perf_gate") / CSV_NAME
    write_synthetic_csv(path)
    return path


@pytest.fixture(scope="module")
def acceptable_outcome(synthetic_csv: Path) -> AnalysisWorkflowOutcome:
    return _run(synthetic_csv, performance_policy=_lenient_policy())


def test_explicit_performance_policy_happy_path(
    acceptable_outcome: AnalysisWorkflowOutcome,
) -> None:
    report = acceptable_outcome.report
    assert report.model_performance_assessment is not None
    assert (
        report.model_performance_assessment.status
        is ModelPerformanceAcceptanceStatus.ACCEPTABLE
    )
    assert report.status in {
        AnalysisWorkflowStatus.COMPLETED,
        AnalysisWorkflowStatus.PARTIAL,
    }
    assert report.final_recommendation is not None
    assert report.final_recommendation.status in {
        RecommendationStatus.GENERATED,
        RecommendationStatus.READY_FOR_OPTIMIZATION,
    }


def test_acceptable_allows_safety_and_recommendation(
    acceptable_outcome: AnalysisWorkflowOutcome,
) -> None:
    report = acceptable_outcome.report
    assert report.model_performance_assessment is not None
    assert (
        report.model_performance_assessment.status
        is ModelPerformanceAcceptanceStatus.ACCEPTABLE
    )
    assert report.metadata["model_performance_assessed"] is True
    assert report.metadata["model_performance_status"] == "ACCEPTABLE"
    assert report.metadata["model_performance_gate_bypassed"] is False
    assert report.final_recommendation is not None
    assert report.final_recommendation.status is not RecommendationStatus.REFUSED


def test_strict_policy_marks_unacceptable_and_refuses(
    synthetic_csv: Path,
) -> None:
    outcome = _run(synthetic_csv, performance_policy=_strict_impossible_policy())
    report = outcome.report
    assert report.model_performance_assessment is not None
    assert (
        report.model_performance_assessment.status
        is ModelPerformanceAcceptanceStatus.UNACCEPTABLE
    )
    assert report.status is AnalysisWorkflowStatus.REFUSED
    assert report.final_recommendation is not None
    assert report.final_recommendation.status is RecommendationStatus.REFUSED
    assert report.final_recommendation.changes == []
    assert report.final_recommendation.proposed_prediction is None
    assert report.final_recommendation.proposed_anomaly_score is None
    assert report.metadata["model_performance_gate_bypassed"] is False
    assert report.metadata["model_performance_status"] == "UNACCEPTABLE"


def test_missing_required_metric_unavailable_and_refuses(
    synthetic_csv: Path,
) -> None:
    outcome = _run(synthetic_csv, performance_policy=_missing_metric_policy())
    report = outcome.report
    assert report.model_performance_assessment is not None
    assert (
        report.model_performance_assessment.status
        is ModelPerformanceAcceptanceStatus.UNAVAILABLE
    )
    assert report.status is AnalysisWorkflowStatus.REFUSED
    assert report.final_recommendation is not None
    assert report.final_recommendation.status is RecommendationStatus.REFUSED
    assert report.final_recommendation.changes == []
    assert report.final_recommendation.proposed_prediction is None
    assert report.final_recommendation.proposed_anomaly_score is None
    assert report.metadata["model_performance_gate_bypassed"] is False
    assert report.metadata["model_performance_status"] == "UNAVAILABLE"


def test_model_performance_requirement_not_disabled(
    acceptable_outcome: AnalysisWorkflowOutcome,
) -> None:
    default_policy = RecommendationSafetyPolicy()
    assert default_policy.require_acceptable_model_performance is True
    assert default_policy.require_final_evaluation is True
    report = acceptable_outcome.report
    assert report.metadata["model_performance_gate_bypassed"] is False


def test_acceptance_uses_independent_test_metrics_only(
    acceptable_outcome: AnalysisWorkflowOutcome,
) -> None:
    assessment = acceptable_outcome.report.model_performance_assessment
    assert assessment is not None
    assert assessment.independent_test_evaluation is True
    assert assessment.evaluation_available is True
    observed_names = {item.metric_name for item in assessment.metric_results}
    assert observed_names <= {"rmse", "mae", "r2"}
    for item in assessment.metric_results:
        if item.available:
            assert item.observed_value is not None


def test_exact_threshold_boundary_acceptable(synthetic_csv: Path) -> None:
    probe = _run(synthetic_csv, performance_policy=_lenient_policy())
    assessment = probe.report.model_performance_assessment
    assert assessment is not None
    observed = {
        item.metric_name: item.observed_value
        for item in assessment.metric_results
        if item.available and item.observed_value is not None
    }
    assert "rmse" in observed
    boundary_policy = ModelPerformanceAcceptancePolicy(
        rules=[
            MetricAcceptanceRule(
                metric_name="rmse",
                direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
                threshold=float(observed["rmse"]),
            ),
        ],
        minimum_test_rows=1,
    )
    outcome = _run(synthetic_csv, performance_policy=boundary_policy)
    assert outcome.report.model_performance_assessment is not None
    assert (
        outcome.report.model_performance_assessment.status
        is ModelPerformanceAcceptanceStatus.ACCEPTABLE
    )
    assert outcome.report.final_recommendation is not None
    assert outcome.report.final_recommendation.status is not RecommendationStatus.REFUSED


def test_request_policy_immutability(synthetic_csv: Path) -> None:
    policy = _lenient_policy()
    original_threshold = policy.rules[0].threshold
    request = build_request(synthetic_csv, performance_policy=policy)
    policy.rules[0].threshold = 0.0
    assert request.model_performance_policy.rules[0].threshold == pytest.approx(
        original_threshold
    )
    outcome = _happy_workflow().run(request)
    assert outcome.report.model_performance_assessment is not None
    assert (
        outcome.report.model_performance_assessment.status
        is ModelPerformanceAcceptanceStatus.ACCEPTABLE
    )


def test_three_objective_regression_with_performance_gate(
    synthetic_csv: Path,
) -> None:
    for objective in (
        RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        RecommendationObjective.REDUCE_ANOMALY_SCORE,
        RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
    ):
        outcome = _run(
            synthetic_csv,
            objective=objective,
            performance_policy=_lenient_policy(),
        )
        assert outcome.report.model_performance_assessment is not None
        assert (
            outcome.report.model_performance_assessment.status
            is ModelPerformanceAcceptanceStatus.ACCEPTABLE
        )
        assert outcome.report.status in {
            AnalysisWorkflowStatus.COMPLETED,
            AnalysisWorkflowStatus.PARTIAL,
        }


def test_early_termination_has_no_assessment(synthetic_csv: Path) -> None:
    invalid = synthetic_csv.parent / "invalid_perf.csv"
    rows = _build_synthetic_rows()
    frame = pl.DataFrame(rows).select(ALL_CSV_COLUMNS)
    frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("all_null_sensor"))
    frame.write_csv(invalid)
    refused = IndustrialProcessAnalysisWorkflow(
        policy=AnalysisWorkflowPolicy(
            allow_partial_diagnosis_ensemble=True,
            stop_on_validation_blocker=True,
        )
    ).run(build_request(invalid))
    assert refused.report.terminal_stage is AnalysisWorkflowStage.VALIDATE
    assert refused.report.model_performance_assessment is None
    assert refused.report.metadata["model_performance_assessed"] is False
    assert refused.report.metadata["model_performance_gate_bypassed"] is False


def test_row_identity_and_determinism_with_performance_gate(
    synthetic_csv: Path,
) -> None:
    first = _run(synthetic_csv, performance_policy=_lenient_policy())
    second = _run(synthetic_csv, performance_policy=_lenient_policy())
    assert first.report.metadata["row_identity_preserved"] is True
    assert first.report.selected_operating_row_id == second.report.selected_operating_row_id
    assert first.report.model_performance_assessment is not None
    assert second.report.model_performance_assessment is not None
    assert (
        first.report.model_performance_assessment.status
        == second.report.model_performance_assessment.status
    )
    assert (
        first.report.model_performance_assessment.metric_results
        == second.report.model_performance_assessment.metric_results
    )


def test_negative_anomaly_score_regression_with_performance_gate(
    synthetic_csv: Path,
) -> None:
    outcome = _run(
        synthetic_csv,
        objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
        performance_policy=_lenient_policy(),
    )
    report = outcome.report
    assert report.model_performance_assessment is not None
    assert report.final_recommendation is not None
    if report.final_recommendation.baseline_anomaly_score is not None:
        assert isinstance(report.final_recommendation.baseline_anomaly_score, float)
    assert report.metadata["anomaly_score_direction"] == "higher_is_more_anomalous"
