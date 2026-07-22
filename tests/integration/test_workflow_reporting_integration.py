"""Integration tests for workflow presentation reporting (Step 11A).

Runs the real ``IndustrialProcessAnalysisWorkflow`` on synthetic CSV data and
maps the resulting ``AnalysisWorkflowReport`` through
``AnalysisWorkflowReportBuilder``. Does not hard-code exact metric values,
model keys, anomaly scores, or proposed values.
"""

from __future__ import annotations

import json
import math
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
    RecommendationStatus,
)
from process_intelligence.reporting import AnalysisWorkflowReportBuilder
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

CSV_NAME = "semiconductor_reporting.csv"
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
                metric_name="r2",
                direction=MetricAcceptanceDirection.HIGHER_IS_BETTER,
                threshold=1_000_000.0,
            )
        ],
        minimum_test_rows=1,
    )


def _missing_metric_policy() -> ModelPerformanceAcceptancePolicy:
    return ModelPerformanceAcceptancePolicy(
        rules=[
            MetricAcceptanceRule(
                metric_name="definitely_missing_metric_xyz",
                direction=MetricAcceptanceDirection.HIGHER_IS_BETTER,
                threshold=0.5,
            )
        ],
        minimum_test_rows=1,
    )


def build_request(
    csv_path: Path,
    objective: RecommendationObjective,
    *,
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


def _make_workflow() -> IndustrialProcessAnalysisWorkflow:
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
    objective: RecommendationObjective = RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
    *,
    performance_policy: ModelPerformanceAcceptancePolicy | None = None,
    **request_overrides: Any,
) -> AnalysisWorkflowOutcome:
    workflow = _make_workflow()
    return workflow.run(
        build_request(
            csv_path,
            objective,
            performance_policy=performance_policy,
            **request_overrides,
        )
    )


def _assert_no_sensitive_payload(payload: dict[str, Any], csv_path: Path) -> None:
    encoded = json.dumps(payload)
    assert str(csv_path.resolve()) not in encoded
    assert str(csv_path) not in encoded
    assert "DataFrame" not in encoded
    assert "ndarray" not in encoded
    assert "IsolationForest" not in encoded
    assert "traceback" not in encoded.lower()


@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("reporting") / CSV_NAME
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
def unacceptable_outcome(synthetic_csv: Path) -> AnalysisWorkflowOutcome:
    return _run(
        synthetic_csv,
        performance_policy=_strict_impossible_policy(),
    )


@pytest.fixture(scope="module")
def unavailable_outcome(synthetic_csv: Path) -> AnalysisWorkflowOutcome:
    return _run(
        synthetic_csv,
        performance_policy=_missing_metric_policy(),
    )


def test_happy_path_presentation_completed_or_partial(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    source = quality_outcome.report
    presentation = AnalysisWorkflowReportBuilder().build(source).report
    assert source.status in {
        AnalysisWorkflowStatus.COMPLETED,
        AnalysisWorkflowStatus.PARTIAL,
    }
    assert presentation.overview.status is source.status
    assert presentation.recommendation is not None
    assert presentation.recommendation.status in {
        RecommendationStatus.GENERATED,
        RecommendationStatus.READY_FOR_OPTIMIZATION,
    }


def test_acceptable_performance_displayed(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    source = quality_outcome.report
    presentation = AnalysisWorkflowReportBuilder().build(source).report
    assert source.model_performance_assessment is not None
    assert (
        source.model_performance_assessment.status
        is ModelPerformanceAcceptanceStatus.ACCEPTABLE
    )
    assert presentation.model_performance is not None
    assert (
        presentation.model_performance.status
        is ModelPerformanceAcceptanceStatus.ACCEPTABLE
    )
    assert presentation.model_performance.independent_test_evaluation is True
    assert presentation.model_summary.independent_test_evaluation_performed is True


def test_unacceptable_workflow_refused_presentation(
    unacceptable_outcome: AnalysisWorkflowOutcome,
) -> None:
    source = unacceptable_outcome.report
    presentation = AnalysisWorkflowReportBuilder().build(source).report
    assert source.status is AnalysisWorkflowStatus.REFUSED
    assert source.model_performance_assessment is not None
    assert (
        source.model_performance_assessment.status
        is ModelPerformanceAcceptanceStatus.UNACCEPTABLE
    )
    assert presentation.overview.status is AnalysisWorkflowStatus.REFUSED
    assert presentation.model_performance is not None
    assert (
        presentation.model_performance.status
        is ModelPerformanceAcceptanceStatus.UNACCEPTABLE
    )
    assert presentation.recommendation is not None
    assert presentation.recommendation.status is RecommendationStatus.REFUSED


def test_unavailable_workflow_refused_presentation(
    unavailable_outcome: AnalysisWorkflowOutcome,
) -> None:
    source = unavailable_outcome.report
    presentation = AnalysisWorkflowReportBuilder().build(source).report
    assert source.status is AnalysisWorkflowStatus.REFUSED
    assert source.model_performance_assessment is not None
    assert (
        source.model_performance_assessment.status
        is ModelPerformanceAcceptanceStatus.UNAVAILABLE
    )
    assert presentation.overview.status is AnalysisWorkflowStatus.REFUSED
    assert presentation.model_performance is not None
    assert (
        presentation.model_performance.status
        is ModelPerformanceAcceptanceStatus.UNAVAILABLE
    )


@pytest.mark.parametrize(
    "outcome_fixture",
    ["quality_outcome", "reduce_outcome", "balance_outcome"],
)
def test_objective_presentations(
    outcome_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    outcome = request.getfixturevalue(outcome_fixture)
    assert isinstance(outcome, AnalysisWorkflowOutcome)
    source = outcome.report
    presentation = AnalysisWorkflowReportBuilder().build(source).report
    assert presentation.overview.status in {
        AnalysisWorkflowStatus.COMPLETED,
        AnalysisWorkflowStatus.PARTIAL,
        AnalysisWorkflowStatus.REFUSED,
    }
    assert presentation.model_performance is not None
    assert len(presentation.stages) == len(list(AnalysisWorkflowStage))


def test_stage_order_matches_workflow(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    source = quality_outcome.report
    presentation = AnalysisWorkflowReportBuilder().build(source).report
    assert len(presentation.stages) == 21
    assert [stage.stage for stage in presentation.stages] == [
        record.stage for record in source.stage_records
    ]
    assert [stage.sequence for stage in presentation.stages] == list(range(1, 22))


def test_recommendation_mapping_for_generated_or_ready(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    source = quality_outcome.report
    presentation = AnalysisWorkflowReportBuilder().build(source).report
    assert source.final_recommendation is not None
    assert presentation.recommendation is not None
    assert presentation.recommendation.status is source.final_recommendation.status
    if source.final_recommendation.status is RecommendationStatus.GENERATED:
        assert len(presentation.recommendation.changes) >= 1
        for src_change, view_change in zip(
            source.final_recommendation.changes,
            presentation.recommendation.changes,
            strict=True,
        ):
            assert view_change.variable == src_change.variable
            assert view_change.current_value == pytest.approx(src_change.current_value)
            assert view_change.proposed_value == pytest.approx(src_change.proposed_value)
            assert view_change.delta == pytest.approx(src_change.delta)
            assert view_change.confidence == pytest.approx(src_change.confidence)
    else:
        assert presentation.recommendation.changes == []
        assert presentation.recommendation.proposed_prediction is None
        assert presentation.recommendation.proposed_anomaly_score is None


def test_refused_recommendation_mapping(
    unacceptable_outcome: AnalysisWorkflowOutcome,
) -> None:
    source = unacceptable_outcome.report
    presentation = AnalysisWorkflowReportBuilder().build(source).report
    assert presentation.recommendation is not None
    assert presentation.recommendation.status is RecommendationStatus.REFUSED
    assert presentation.recommendation.changes == []
    assert presentation.recommendation.proposed_prediction is None
    assert presentation.recommendation.proposed_anomaly_score is None


def test_negative_anomaly_scores_preserved_without_clipping(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    source = quality_outcome.report
    presentation = AnalysisWorkflowReportBuilder().build(source).report
    assert source.final_recommendation is not None
    assert presentation.recommendation is not None
    for attr in ("baseline_anomaly_score", "proposed_anomaly_score"):
        src_value = getattr(source.final_recommendation, attr)
        view_value = getattr(presentation.recommendation, attr)
        if src_value is None:
            assert view_value is None
            continue
        assert view_value == pytest.approx(src_value)
        assert math.isfinite(view_value)
        # No abs/clip/offset rewrite
        assert view_value == src_value


def test_performance_rules_and_safety_flags(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    source = quality_outcome.report
    presentation = AnalysisWorkflowReportBuilder().build(source).report
    assert presentation.model_performance is not None
    assert source.model_performance_assessment is not None
    assert len(presentation.model_performance.metrics) == len(
        source.model_performance_assessment.metric_results
    )
    assert presentation.metadata["model_performance_gate_bypassed"] is False
    assert presentation.data_summary.row_identity_preserved is True
    assert presentation.model_summary.test_used_for_model_selection is False
    assert presentation.model_summary.test_used_for_threshold_calibration is False
    assert presentation.metadata["test_used_for_model_selection"] is False
    assert presentation.metadata["test_used_for_threshold_calibration"] is False


def test_warnings_and_disclaimers_present(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    presentation = AnalysisWorkflowReportBuilder().build(quality_outcome.report).report
    assert isinstance(presentation.warnings, list)
    assert presentation.disclaimers
    joined = " ".join(presentation.disclaimers).lower()
    assert "model-based" in joined
    assert "causation" in joined
    assert "verification" in joined
    assert "not guaranteed" in joined


def test_no_sensitive_objects_or_paths(
    quality_outcome: AnalysisWorkflowOutcome,
    synthetic_csv: Path,
) -> None:
    presentation = AnalysisWorkflowReportBuilder().build(quality_outcome.report).report
    dumped = presentation.model_dump(mode="json")
    _assert_no_sensitive_payload(dumped, synthetic_csv)
    assert "model" not in dumped
    assert "estimator" not in dumped
    encoded = json.dumps(dumped)
    assert "C:\\" not in encoded
    assert "/Users/" not in encoded


def test_json_serialization_and_determinism(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    source = quality_outcome.report
    builder = AnalysisWorkflowReportBuilder()
    first = builder.build(source).report.model_dump(mode="json")
    second = builder.build(source).report.model_dump(mode="json")
    assert first == second
    json.dumps(first)
    assert isinstance(first["overview"]["started_at"], str)
    assert isinstance(first["overview"]["status"], str)


def test_workflow_report_immutability(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    source = quality_outcome.report
    before = source.model_dump(mode="python")
    warnings_before = list(source.warnings)
    metadata_before = dict(source.metadata)
    AnalysisWorkflowReportBuilder().build(source)
    assert source.model_dump(mode="python") == before
    assert source.warnings == warnings_before
    assert source.metadata == metadata_before


def test_datetime_and_counts_preserved(
    quality_outcome: AnalysisWorkflowOutcome,
) -> None:
    source = quality_outcome.report
    presentation = AnalysisWorkflowReportBuilder().build(source).report
    assert presentation.overview.started_at == source.started_at
    assert presentation.overview.completed_at == source.completed_at
    assert presentation.overview.total_seconds == pytest.approx(source.total_seconds)
    assert presentation.data_summary.raw_row_count == source.raw_row_count
    assert presentation.data_summary.processed_row_count == source.processed_row_count
    assert presentation.data_summary.cohort_row_count == source.cohort_row_count
    assert presentation.data_summary.train_row_count == source.train_row_count
    assert presentation.data_summary.validation_row_count == source.validation_row_count
    assert presentation.data_summary.test_row_count == source.test_row_count
    split_total = (
        presentation.data_summary.train_row_count
        + presentation.data_summary.validation_row_count
        + presentation.data_summary.test_row_count
    )
    assert split_total == presentation.data_summary.cohort_row_count


def test_builder_does_not_rewrite_backend_status(
    quality_outcome: AnalysisWorkflowOutcome,
    unacceptable_outcome: AnalysisWorkflowOutcome,
) -> None:
    for source in (quality_outcome.report, unacceptable_outcome.report):
        assert isinstance(source, AnalysisWorkflowReport)
        presentation = AnalysisWorkflowReportBuilder().build(source).report
        assert presentation.overview.status is source.status
        if source.final_recommendation is None:
            assert presentation.recommendation is None
        else:
            assert presentation.recommendation is not None
            assert (
                presentation.recommendation.status
                is source.final_recommendation.status
            )
