"""Integration acceptance tests for the synthetic manufacturing demo workflow."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

from process_intelligence.core.exceptions import DataLeakageError
from process_intelligence.demo_data import (
    DEMO_CONTROLLABLE_COLUMNS,
    DEMO_IDENTITY_COLUMNS,
    DEMO_MEASURED_COLUMNS,
    DEMO_QUALITY_COLUMNS,
    GROUND_TRUTH_METADATA_COLUMNS,
    DemoDatasetConfiguration,
    generate_demo_dataset,
)
from process_intelligence.demo_validation import (
    MIN_ANOMALY_ENRICHMENT_FACTOR,
    assert_no_demo_feature_leakage,
    default_demo_excluded_columns,
    default_demo_feature_columns,
    validate_demo_anomaly_only,
    validate_demo_supervised,
    validate_demo_workflows,
)
from process_intelligence.workflow import AnalysisExecutionMode

ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = ROOT / "scripts" / "validate_demo_workflow.py"

_FORBIDDEN_MODEL_FEATURES: tuple[str, ...] = (
    *DEMO_QUALITY_COLUMNS,
    *GROUND_TRUTH_METADATA_COLUMNS,
    *DEMO_IDENTITY_COLUMNS,
)


def _default_config() -> DemoDatasetConfiguration:
    return DemoDatasetConfiguration()


def _assert_process_only_features(feature_columns: tuple[str, ...] | list[str]) -> None:
    features = list(feature_columns)
    assert features == [*DEMO_CONTROLLABLE_COLUMNS, *DEMO_MEASURED_COLUMNS]
    for name in DEMO_CONTROLLABLE_COLUMNS:
        assert name in features
    for name in DEMO_MEASURED_COLUMNS:
        assert name in features
    for name in _FORBIDDEN_MODEL_FEATURES:
        assert name not in features


def test_default_demo_supervised_validation_passes() -> None:
    result = validate_demo_supervised(_default_config())
    assert result.passed is True
    assert result.leakage_free is True
    assert result.selected_model is not None
    assert result.selected_model.strip() != ""
    assert result.selected_task == "REGRESSION"
    assert "rmse" in result.regression_metrics
    assert "mae" in result.regression_metrics
    assert math.isfinite(result.regression_metrics["rmse"])
    assert math.isfinite(result.regression_metrics["mae"])
    assert result.beats_mean_baseline is True
    assert result.feature_count == len(result.feature_columns)
    _assert_process_only_features(result.feature_columns)
    assert result.excluded_columns == tuple(
        default_demo_excluded_columns(analysis_mode=AnalysisExecutionMode.SUPERVISED)
    )
    assert "defect_rate" in result.excluded_columns
    if result.residual_available:
        assert result.residual_matched_injected_rows >= 1
        assert result.residual_enrichment_factor is not None
        assert result.residual_enrichment_factor >= MIN_ANOMALY_ENRICHMENT_FACTOR


def test_default_demo_anomaly_only_validation_passes() -> None:
    result = validate_demo_anomaly_only(_default_config())
    assert result.passed is True
    assert result.leakage_free is True
    assert result.selected_model is not None
    assert result.selected_model.strip() != ""
    assert result.detected_row_count >= 1
    assert result.matched_injected_anomaly_rows >= 1
    assert result.enrichment_factor is not None
    assert result.enrichment_factor >= MIN_ANOMALY_ENRICHMENT_FACTOR
    assert result.feature_count == len(result.feature_columns)
    _assert_process_only_features(result.feature_columns)
    assert result.excluded_columns == tuple(
        default_demo_excluded_columns(analysis_mode=AnalysisExecutionMode.ANOMALY_ONLY)
    )
    for name in DEMO_QUALITY_COLUMNS:
        assert name in result.excluded_columns


def test_validation_report_is_deterministic_for_fixed_seed() -> None:
    config = DemoDatasetConfiguration(random_seed=42, row_count=1500)
    first = validate_demo_workflows(config)
    second = validate_demo_workflows(config)
    assert first.passed is True
    assert second.passed is True
    assert first.supervised.selected_model == second.supervised.selected_model
    assert first.anomaly_only.selected_model == second.anomaly_only.selected_model
    assert first.supervised.regression_metrics == second.supervised.regression_metrics
    assert first.anomaly_only.enrichment_factor == second.anomaly_only.enrichment_factor
    assert (
        first.supervised.residual_enrichment_factor
        == second.supervised.residual_enrichment_factor
    )
    assert first.supervised.feature_columns == second.supervised.feature_columns
    assert first.anomaly_only.feature_columns == second.anomaly_only.feature_columns


def test_leakage_columns_absent_from_feature_list() -> None:
    features = default_demo_feature_columns()
    assert features == [*DEMO_CONTROLLABLE_COLUMNS, *DEMO_MEASURED_COLUMNS]
    _assert_process_only_features(features)
    assert_no_demo_feature_leakage(
        features,
        analysis_mode=AnalysisExecutionMode.SUPERVISED,
    )
    assert_no_demo_feature_leakage(
        features,
        analysis_mode=AnalysisExecutionMode.ANOMALY_ONLY,
    )


def test_corrupted_feature_list_with_injected_anomaly_is_rejected() -> None:
    corrupted = [*default_demo_feature_columns(), "injected_anomaly"]
    with pytest.raises(DataLeakageError, match="injected_anomaly"):
        assert_no_demo_feature_leakage(
            corrupted,
            analysis_mode=AnalysisExecutionMode.SUPERVISED,
        )
    with pytest.raises(DataLeakageError, match="injected_anomaly"):
        assert_no_demo_feature_leakage(
            corrupted,
            analysis_mode=AnalysisExecutionMode.ANOMALY_ONLY,
        )


def test_quality_score_forbidden_as_supervised_feature() -> None:
    corrupted = [*default_demo_feature_columns(), "quality_score"]
    with pytest.raises(DataLeakageError, match="quality_score"):
        assert_no_demo_feature_leakage(
            corrupted,
            analysis_mode=AnalysisExecutionMode.SUPERVISED,
        )


def test_defect_rate_forbidden_as_supervised_feature() -> None:
    corrupted = [*default_demo_feature_columns(), "defect_rate"]
    with pytest.raises(DataLeakageError, match="defect_rate"):
        assert_no_demo_feature_leakage(
            corrupted,
            analysis_mode=AnalysisExecutionMode.SUPERVISED,
        )


def test_quality_outputs_forbidden_as_anomaly_only_features() -> None:
    for quality_column in DEMO_QUALITY_COLUMNS:
        corrupted = [*default_demo_feature_columns(), quality_column]
        with pytest.raises(DataLeakageError, match=quality_column):
            assert_no_demo_feature_leakage(
                corrupted,
                analysis_mode=AnalysisExecutionMode.ANOMALY_ONLY,
            )


def test_identity_columns_forbidden_as_model_features() -> None:
    for identity_column in DEMO_IDENTITY_COLUMNS:
        corrupted = [*default_demo_feature_columns(), identity_column]
        with pytest.raises(DataLeakageError, match=identity_column):
            assert_no_demo_feature_leakage(
                corrupted,
                analysis_mode=AnalysisExecutionMode.SUPERVISED,
            )
        with pytest.raises(DataLeakageError, match=identity_column):
            assert_no_demo_feature_leakage(
                corrupted,
                analysis_mode=AnalysisExecutionMode.ANOMALY_ONLY,
            )


def test_metrics_are_finite_and_enrichment_above_threshold() -> None:
    report = validate_demo_workflows(_default_config())
    assert report.passed is True
    assert report.enrichment_threshold == MIN_ANOMALY_ENRICHMENT_FACTOR
    for name in ("rmse", "mae"):
        assert math.isfinite(report.supervised.regression_metrics[name])
    assert report.anomaly_only.enrichment_factor is not None
    assert report.anomaly_only.enrichment_factor >= MIN_ANOMALY_ENRICHMENT_FACTOR


def test_cli_exits_successfully() -> None:
    completed = subprocess.run(
        [sys.executable, str(CLI_PATH)],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "FINAL: PASS" in completed.stdout
    assert "selected_model:" in completed.stdout
    assert "enrichment:" in completed.stdout
    assert "supervised_feature_count:" in completed.stdout
    assert "anomaly_only_feature_count:" in completed.stdout
    assert "quality_outputs_and_ground_truth_excluded: True" in completed.stdout


def test_default_demo_has_injected_anomalies_in_time_split_test_region() -> None:
    frame = generate_demo_dataset(_default_config())
    n_rows = len(frame)
    test_start = int(round(n_rows * 0.80))
    test_anomaly_count = int(frame.loc[test_start:, "injected_anomaly"].sum())
    assert test_anomaly_count >= 1


def test_default_demo_supervised_recommendation_safety_and_plausibility() -> None:
    """Supervised demo recommendation either generates safely or refuses honestly.

    After restoring full-factor diagnosis confidence, the default demo may refuse
    at the 0.20 gate. When generation succeeds, proposed setpoints must remain
    inside approved bounds and declared-domain extrapolation must be flagged.
    """
    import tempfile

    from process_intelligence.demo_data import (
        DEMO_QUALITY_SCORE_DECLARED_MAXIMUM,
        DEMO_QUALITY_SCORE_DECLARED_MINIMUM,
        demo_setpoint_bound_map,
        write_demo_workflow_csv,
    )
    from process_intelligence.demo_validation import (
        _run_supervised_workflow,
        default_demo_feature_columns,
    )
    from process_intelligence.recommendation import RecommendationStatus
    from process_intelligence.recommendation.enums import (
        TargetPredictionPlausibilityStatus,
        WhatIfVerificationStatus,
    )
    from process_intelligence.workflow import AnalysisWorkflowStatus

    frame = generate_demo_dataset(_default_config())
    bounds = demo_setpoint_bound_map()
    with tempfile.TemporaryDirectory(prefix="demo_rec_") as temp_dir:
        csv_path = write_demo_workflow_csv(frame, Path(temp_dir) / "demo.csv")
        outcome = _run_supervised_workflow(
            csv_path=csv_path,
            feature_columns=default_demo_feature_columns(),
        )
    report = outcome.report
    recommendation = report.final_recommendation
    assert recommendation is not None
    assert report.status in {
        AnalysisWorkflowStatus.COMPLETED,
        AnalysisWorkflowStatus.REFUSED,
        AnalysisWorkflowStatus.PARTIAL,
    }

    operating_row_id = report.selected_operating_row_id
    assert operating_row_id is not None
    row = frame.loc[int(operating_row_id)]
    for name, (low, high) in bounds.items():
        value = float(row[name])
        assert low <= value <= high

    if recommendation.status is RecommendationStatus.REFUSED:
        assert report.status is AnalysisWorkflowStatus.REFUSED
        assert recommendation.proposed_prediction is None
        assert not recommendation.changes
        return

    assert recommendation.status is RecommendationStatus.GENERATED
    assert report.status is AnalysisWorkflowStatus.COMPLETED
    assert len(recommendation.changes) >= 1

    approved = set(DEMO_CONTROLLABLE_COLUMNS)
    measured = set(DEMO_MEASURED_COLUMNS)
    for change in recommendation.changes:
        assert change.variable in approved
        assert change.variable not in measured
        low, high = bounds[change.variable]
        assert low <= change.proposed_value <= high
        assert math.isfinite(change.current_value)
        assert math.isfinite(change.proposed_value)
        assert math.isfinite(change.delta)

    plausibility = recommendation.target_prediction_plausibility
    assert plausibility is not None
    assert plausibility.proposed_status is not None
    assert recommendation.baseline_prediction is not None
    assert recommendation.proposed_prediction is not None
    assert math.isfinite(recommendation.baseline_prediction)
    assert math.isfinite(recommendation.proposed_prediction)
    assert recommendation.proposed_prediction >= recommendation.baseline_prediction

    proposed = float(recommendation.proposed_prediction)
    if (
        proposed < DEMO_QUALITY_SCORE_DECLARED_MINIMUM
        or proposed > DEMO_QUALITY_SCORE_DECLARED_MAXIMUM
    ):
        assert (
            plausibility.proposed_status
            is TargetPredictionPlausibilityStatus.OUTSIDE_DECLARED_DOMAIN
        )
        assert any("declared" in message.lower() for message in recommendation.warnings)

    verification = report.recommendation_verification
    assert verification is not None
    assert verification.status is WhatIfVerificationStatus.COMPLETED
    assert verification.baseline_objective_value is not None
    assert verification.proposed_objective_value is not None
    assert math.isfinite(verification.baseline_objective_value)
    assert math.isfinite(verification.proposed_objective_value)
