"""Unit tests for model performance acceptance (Step 10D)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.evaluation import (
    FinalEvaluationOutcome,
    FinalEvaluationReport,
    MetricAcceptanceDirection,
    MetricAcceptanceResult,
    MetricAcceptanceRule,
    ModelPerformanceAcceptanceOutcome,
    ModelPerformanceAcceptancePolicy,
    ModelPerformanceAcceptanceReport,
    ModelPerformanceAcceptanceStatus,
    ModelPerformanceAssessor,
)


def _rule(
    metric_name: str = "r2",
    *,
    direction: MetricAcceptanceDirection = MetricAcceptanceDirection.HIGHER_IS_BETTER,
    threshold: float = 0.5,
    required: bool = True,
) -> MetricAcceptanceRule:
    return MetricAcceptanceRule(
        metric_name=metric_name,
        direction=direction,
        threshold=threshold,
        required=required,
    )


def _policy(**overrides: Any) -> ModelPerformanceAcceptancePolicy:
    payload: dict[str, Any] = {
        "rules": [
            _rule("r2", direction=MetricAcceptanceDirection.HIGHER_IS_BETTER, threshold=0.5),
            _rule(
                "mae",
                direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
                threshold=1.0,
            ),
        ],
    }
    payload.update(overrides)
    return ModelPerformanceAcceptancePolicy(**payload)


def _final_report(**overrides: Any) -> FinalEvaluationReport:
    payload: dict[str, Any] = {
        "task": AnalysisTask.REGRESSION,
        "model_name": "Linear Regression",
        "estimator_key": "linear_regression",
        "target_column": "y",
        "feature_columns": ["f1", "f2"],
        "refit_on_train_validation": True,
        "train_row_count": 6,
        "validation_row_count": 3,
        "test_row_count": 3,
        "final_fit_row_count": 9,
        "validation_metrics": {"rmse": 0.2, "mae": 0.1, "r2": 0.99},
        "test_metrics": {"rmse": 0.8, "mae": 0.6, "r2": 0.7},
        "primary_metric_name": "rmse",
        "higher_is_better": False,
        "validation_primary_metric": 0.2,
        "test_primary_metric": 0.8,
        "generalization_gap": 0.6,
        "performance_degraded": True,
        "fit_seconds": 0.01,
        "test_evaluation_seconds": 0.02,
        "total_seconds": 0.03,
        "evaluated_at": datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
        "warnings": [],
    }
    payload.update(overrides)
    return FinalEvaluationReport(**payload)


# --- 1. enums ---


def test_metric_acceptance_direction_values() -> None:
    assert MetricAcceptanceDirection.HIGHER_IS_BETTER == "HIGHER_IS_BETTER"
    assert MetricAcceptanceDirection.LOWER_IS_BETTER == "LOWER_IS_BETTER"
    assert list(MetricAcceptanceDirection) == [
        MetricAcceptanceDirection.HIGHER_IS_BETTER,
        MetricAcceptanceDirection.LOWER_IS_BETTER,
    ]


def test_model_performance_acceptance_status_values() -> None:
    assert ModelPerformanceAcceptanceStatus.ACCEPTABLE == "ACCEPTABLE"
    assert ModelPerformanceAcceptanceStatus.UNACCEPTABLE == "UNACCEPTABLE"
    assert ModelPerformanceAcceptanceStatus.UNAVAILABLE == "UNAVAILABLE"
    assert {member.value for member in ModelPerformanceAcceptanceStatus} == {
        "ACCEPTABLE",
        "UNACCEPTABLE",
        "UNAVAILABLE",
    }


# --- 2-7. rule / policy validation ---


def test_rule_valid_creation() -> None:
    rule = _rule("rmse", direction=MetricAcceptanceDirection.LOWER_IS_BETTER, threshold=2.5)
    assert rule.metric_name == "rmse"
    assert rule.direction is MetricAcceptanceDirection.LOWER_IS_BETTER
    assert rule.threshold == pytest.approx(2.5)
    assert rule.required is True


def test_rule_rejects_empty_metric_name() -> None:
    with pytest.raises(ValidationError):
        _rule("")
    with pytest.raises(ValidationError):
        _rule("   ")


@pytest.mark.parametrize("bad_threshold", [True, False, float("nan"), float("inf")])
def test_rule_rejects_invalid_threshold(bad_threshold: object) -> None:
    with pytest.raises(ValidationError):
        MetricAcceptanceRule(
            metric_name="r2",
            direction=MetricAcceptanceDirection.HIGHER_IS_BETTER,
            threshold=bad_threshold,  # type: ignore[arg-type]
        )


def test_policy_rejects_empty_rules() -> None:
    with pytest.raises(ValidationError):
        ModelPerformanceAcceptancePolicy(rules=[])


def test_policy_rejects_duplicate_metric() -> None:
    with pytest.raises(ValidationError):
        ModelPerformanceAcceptancePolicy(
            rules=[
                _rule("r2", threshold=0.5),
                _rule("r2", threshold=0.6),
            ]
        )


@pytest.mark.parametrize("bad_rows", [0, True, False, -1])
def test_policy_rejects_invalid_minimum_rows(bad_rows: object) -> None:
    with pytest.raises(ValidationError):
        ModelPerformanceAcceptancePolicy(
            rules=[_rule()],
            minimum_test_rows=bad_rows,  # type: ignore[arg-type]
        )


# --- 8-13. metric comparison ---


def test_higher_is_better_boundary_passes() -> None:
    outcome = ModelPerformanceAssessor(
        policy=ModelPerformanceAcceptancePolicy(
            rules=[_rule("r2", threshold=0.7)],
        )
    ).assess(_final_report(test_metrics={"rmse": 0.8, "mae": 0.6, "r2": 0.7}))
    result = outcome.report.metric_results[0]
    assert result.available is True
    assert result.passed is True
    assert result.observed_value == pytest.approx(0.7)


def test_higher_is_better_below_threshold_fails() -> None:
    outcome = ModelPerformanceAssessor(
        policy=ModelPerformanceAcceptancePolicy(
            rules=[_rule("r2", threshold=0.8)],
        )
    ).assess(_final_report(test_metrics={"rmse": 0.8, "mae": 0.6, "r2": 0.7}))
    result = outcome.report.metric_results[0]
    assert result.passed is False
    assert outcome.report.status is ModelPerformanceAcceptanceStatus.UNACCEPTABLE


def test_lower_is_better_boundary_passes() -> None:
    outcome = ModelPerformanceAssessor(
        policy=ModelPerformanceAcceptancePolicy(
            rules=[
                _rule(
                    "mae",
                    direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
                    threshold=0.6,
                )
            ],
        )
    ).assess(_final_report(test_metrics={"rmse": 0.8, "mae": 0.6, "r2": 0.7}))
    assert outcome.report.metric_results[0].passed is True
    assert outcome.report.status is ModelPerformanceAcceptanceStatus.ACCEPTABLE


def test_lower_is_better_above_threshold_fails() -> None:
    outcome = ModelPerformanceAssessor(
        policy=ModelPerformanceAcceptancePolicy(
            rules=[
                _rule(
                    "mae",
                    direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
                    threshold=0.5,
                )
            ],
        )
    ).assess(_final_report(test_metrics={"rmse": 0.8, "mae": 0.6, "r2": 0.7}))
    assert outcome.report.metric_results[0].passed is False
    assert outcome.report.status is ModelPerformanceAcceptanceStatus.UNACCEPTABLE


def test_optional_metric_unavailable_does_not_block() -> None:
    outcome = ModelPerformanceAssessor(
        policy=ModelPerformanceAcceptancePolicy(
            rules=[
                _rule("r2", threshold=0.5),
                _rule(
                    "explained_variance",
                    direction=MetricAcceptanceDirection.HIGHER_IS_BETTER,
                    threshold=0.5,
                    required=False,
                ),
            ],
        )
    ).assess(_final_report())
    assert outcome.report.status is ModelPerformanceAcceptanceStatus.ACCEPTABLE
    optional = [
        item for item in outcome.report.metric_results if item.metric_name == "explained_variance"
    ][0]
    assert optional.available is False
    assert optional.passed is None
    assert optional.observed_value is None


def test_required_metric_unavailable_is_unavailable() -> None:
    outcome = ModelPerformanceAssessor(
        policy=ModelPerformanceAcceptancePolicy(
            rules=[
                _rule(
                    "missing_metric",
                    direction=MetricAcceptanceDirection.HIGHER_IS_BETTER,
                    threshold=0.5,
                )
            ],
        )
    ).assess(_final_report())
    assert outcome.report.status is ModelPerformanceAcceptanceStatus.UNAVAILABLE
    assert outcome.report.unavailable_required_rule_count == 1
    assert outcome.report.metric_results[0].available is False
    assert outcome.report.metric_results[0].passed is None


def test_missing_r2_from_constant_target_is_not_acceptable() -> None:
    """Constant-target defense: omitted R² must not yield ACCEPTABLE."""
    outcome = ModelPerformanceAssessor(
        policy=ModelPerformanceAcceptancePolicy(
            rules=[
                _rule("r2", threshold=0.0),
                _rule(
                    "mae",
                    direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
                    threshold=1_000.0,
                ),
                _rule(
                    "rmse",
                    direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
                    threshold=1_000.0,
                ),
            ],
        )
    ).assess(
        _final_report(
            test_metrics={"rmse": 0.0, "mae": 0.0},
            validation_metrics={"rmse": 0.0, "mae": 0.0},
        )
    )
    assert outcome.report.status is not ModelPerformanceAcceptanceStatus.ACCEPTABLE
    assert outcome.report.status is ModelPerformanceAcceptanceStatus.UNAVAILABLE
    r2_result = next(
        item for item in outcome.report.metric_results if item.metric_name == "r2"
    )
    assert r2_result.available is False
    assert r2_result.observed_value is None
    assert r2_result.passed is None
    assert outcome.report.independent_test_evaluation is True


# --- 14-18. report statuses ---


def test_acceptable_report() -> None:
    outcome = ModelPerformanceAssessor(policy=_policy()).assess(_final_report())
    report = outcome.report
    assert report.status is ModelPerformanceAcceptanceStatus.ACCEPTABLE
    assert report.evaluation_available is True
    assert report.independent_test_evaluation is True
    assert report.failed_required_rule_count == 0
    assert report.unavailable_required_rule_count == 0
    assert report.test_row_count == 3


def test_unacceptable_report() -> None:
    outcome = ModelPerformanceAssessor(
        policy=ModelPerformanceAcceptancePolicy(
            rules=[_rule("r2", threshold=0.95)],
        )
    ).assess(_final_report())
    assert outcome.report.status is ModelPerformanceAcceptanceStatus.UNACCEPTABLE
    assert outcome.report.failed_required_rule_count == 1


def test_unavailable_report_when_evaluation_missing_type() -> None:
    with pytest.raises(TypeError):
        ModelPerformanceAssessor(policy=_policy()).assess(None)  # type: ignore[arg-type]


def test_non_independent_test_report_unavailable() -> None:
    report = _final_report()
    broken = FinalEvaluationReport.model_construct(
        **{
            **report.model_dump(),
            "test_metrics": {},
            "test_row_count": 0,
        }
    )
    outcome = ModelPerformanceAssessor(policy=_policy()).assess(broken)
    assert outcome.report.status is ModelPerformanceAcceptanceStatus.UNAVAILABLE
    assert outcome.report.independent_test_evaluation is False
    assert any("independent" in warning.lower() for warning in outcome.report.warnings)


def test_insufficient_test_rows_unavailable() -> None:
    outcome = ModelPerformanceAssessor(
        policy=ModelPerformanceAcceptancePolicy(
            rules=[_rule("r2", threshold=0.5)],
            minimum_test_rows=10,
        )
    ).assess(_final_report(test_row_count=3))
    assert outcome.report.status is ModelPerformanceAcceptanceStatus.UNAVAILABLE
    assert any("test_row_count" in warning for warning in outcome.report.warnings)


# --- 19-20. no validation metrics / no model access ---


def test_does_not_use_validation_metrics() -> None:
    report = _final_report(
        validation_metrics={"rmse": 0.01, "mae": 0.01, "r2": 0.999},
        test_metrics={"rmse": 5.0, "mae": 4.0, "r2": 0.1},
    )
    outcome = ModelPerformanceAssessor(
        policy=ModelPerformanceAcceptancePolicy(
            rules=[_rule("r2", threshold=0.5)],
        )
    ).assess(report)
    assert outcome.report.status is ModelPerformanceAcceptanceStatus.UNACCEPTABLE
    assert outcome.report.metric_results[0].observed_value == pytest.approx(0.1)


def test_does_not_access_model_or_predict() -> None:
    fake_model = MagicMock()
    outcome_input = FinalEvaluationOutcome(
        final_model=fake_model,
        report=_final_report(),
    )
    result = ModelPerformanceAssessor(policy=_policy()).assess(outcome_input)
    assert result.report.status is ModelPerformanceAcceptanceStatus.ACCEPTABLE
    fake_model.predict.assert_not_called()
    fake_model.evaluate.assert_not_called()
    fake_model.fit.assert_not_called()


# --- 21-22. immutability / independence ---


def test_policy_immutability() -> None:
    policy = _policy()
    assessor = ModelPerformanceAssessor(policy=policy)
    policy.rules[0].threshold = 0.99
    outcome = assessor.assess(_final_report())
    assert outcome.report.status is ModelPerformanceAcceptanceStatus.ACCEPTABLE
    assert outcome.report.metric_results[0].threshold == pytest.approx(0.5)


def test_repeated_assess_calls_are_independent() -> None:
    assessor = ModelPerformanceAssessor(policy=_policy())
    first = assessor.assess(_final_report())
    second = assessor.assess(
        _final_report(test_metrics={"rmse": 9.0, "mae": 8.0, "r2": 0.0})
    )
    assert first.report.status is ModelPerformanceAcceptanceStatus.ACCEPTABLE
    assert second.report.status is ModelPerformanceAcceptanceStatus.UNACCEPTABLE
    assert first.report.metric_results[0].observed_value == pytest.approx(0.7)
    assert second.report.metric_results[0].observed_value == pytest.approx(0.0)


# --- 23-25. metadata / assessed_at / round-trip ---


def test_metadata_scalar_only() -> None:
    report = ModelPerformanceAssessor(policy=_policy()).assess(_final_report()).report
    for key, value in report.metadata.items():
        assert isinstance(key, str) and key.strip() != ""
        assert value is None or isinstance(value, (str, int, float, bool))
        if isinstance(value, float):
            assert value == value  # finite (not NaN)


def test_assessed_at_timezone_aware_utc() -> None:
    report = ModelPerformanceAssessor(policy=_policy()).assess(_final_report()).report
    assert report.assessed_at.tzinfo is not None
    assert report.assessed_at.utcoffset() is not None
    assert report.assessed_at.utcoffset().total_seconds() == 0


def test_report_round_trip() -> None:
    report = ModelPerformanceAssessor(policy=_policy()).assess(_final_report()).report
    restored = ModelPerformanceAcceptanceReport.model_validate(report.model_dump())
    assert restored.status is report.status
    assert restored.task is report.task
    assert restored.metric_results == report.metric_results
    assert restored.metadata == report.metadata


def test_assessor_rejects_invalid_policy_type() -> None:
    with pytest.raises(TypeError):
        ModelPerformanceAssessor(policy={"rules": []})  # type: ignore[arg-type]


def test_outcome_is_frozen_dataclass() -> None:
    import dataclasses

    outcome = ModelPerformanceAssessor(policy=_policy()).assess(_final_report())
    assert isinstance(outcome, ModelPerformanceAcceptanceOutcome)
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.report = outcome.report  # type: ignore[misc]


def test_metric_acceptance_result_availability_contract() -> None:
    ok = MetricAcceptanceResult(
        metric_name="r2",
        observed_value=0.8,
        threshold=0.5,
        direction=MetricAcceptanceDirection.HIGHER_IS_BETTER,
        required=True,
        available=True,
        passed=True,
        message="Metric r2=0.8 meets HIGHER_IS_BETTER threshold 0.5.",
    )
    assert ok.passed is True
    with pytest.raises(ValidationError):
        MetricAcceptanceResult(
            metric_name="r2",
            observed_value=0.8,
            threshold=0.5,
            direction=MetricAcceptanceDirection.HIGHER_IS_BETTER,
            required=True,
            available=False,
            passed=None,
            message="unavailable",
        )
    with pytest.raises(ValidationError):
        MetricAcceptanceResult(
            metric_name="r2",
            observed_value=None,
            threshold=0.5,
            direction=MetricAcceptanceDirection.HIGHER_IS_BETTER,
            required=True,
            available=True,
            passed=True,
            message="inconsistent",
        )
