"""Tests for supervised recommendation target-domain plausibility."""

from __future__ import annotations

import math

import pytest

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.demo_data import (
    DEMO_QUALITY_SCORE_DECLARED_MAXIMUM,
    DEMO_QUALITY_SCORE_DECLARED_MINIMUM,
)
from process_intelligence.evaluation.performance_acceptance import (
    MetricAcceptanceDirection,
)
from process_intelligence.recommendation import (
    QualityOptimizationDirection,
    RecommendationObjective,
)
from process_intelligence.recommendation.enums import TargetPredictionPlausibilityStatus
from process_intelligence.recommendation.target_domain import (
    RecommendationTargetPlausibility,
    TargetPredictionDomain,
    assess_recommendation_target_plausibility,
    build_target_prediction_domain,
    evaluate_target_prediction_plausibility,
)
from process_intelligence.ui.demo_configuration import (
    DATA_SOURCE_UPLOAD_CSV,
    DemoAnalysisTemplate,
    build_demo_configuration_preset,
)
from process_intelligence.ui.schemas import (
    UiMetricRuleInput,
    UiVariableConstraintInput,
    WorkflowUiSubmission,
)
from process_intelligence.workflow.enums import AnalysisExecutionMode


def test_build_domain_from_training_targets_and_optional_declared_bounds() -> None:
    domain = build_target_prediction_domain(
        [50.0, 80.0, 60.0],
        declared_minimum=40.0,
        declared_maximum=100.0,
    )
    assert domain is not None
    assert domain.observed_minimum == 50.0
    assert domain.observed_maximum == 80.0
    assert domain.declared_minimum == 40.0
    assert domain.declared_maximum == 100.0


def test_prediction_at_observed_and_declared_boundaries() -> None:
    domain = TargetPredictionDomain(
        observed_minimum=40.0,
        observed_maximum=100.0,
        declared_minimum=40.0,
        declared_maximum=100.0,
    )
    lower = evaluate_target_prediction_plausibility(40.0, domain=domain)
    upper = evaluate_target_prediction_plausibility(100.0, domain=domain)
    assert lower.status is TargetPredictionPlausibilityStatus.WITHIN_OBSERVED_RANGE
    assert upper.status is TargetPredictionPlausibilityStatus.WITHIN_OBSERVED_RANGE
    assert lower.warning_message is None
    assert upper.warning_message is None


def test_prediction_inside_observed_and_declared_ranges() -> None:
    domain = TargetPredictionDomain(
        observed_minimum=45.0,
        observed_maximum=95.0,
        declared_minimum=40.0,
        declared_maximum=100.0,
    )
    result = evaluate_target_prediction_plausibility(70.0, domain=domain)
    assert result.status is TargetPredictionPlausibilityStatus.WITHIN_OBSERVED_RANGE


def test_prediction_outside_observed_but_inside_declared() -> None:
    domain = TargetPredictionDomain(
        observed_minimum=50.0,
        observed_maximum=90.0,
        declared_minimum=40.0,
        declared_maximum=100.0,
    )
    result = evaluate_target_prediction_plausibility(95.0, domain=domain)
    assert result.status is TargetPredictionPlausibilityStatus.OUTSIDE_OBSERVED_RANGE
    assert result.warning_message is not None
    assert "extrapolation" in result.warning_message.lower()


def test_prediction_outside_declared_domain() -> None:
    domain = TargetPredictionDomain(
        observed_minimum=50.0,
        observed_maximum=90.0,
        declared_minimum=40.0,
        declared_maximum=100.0,
    )
    result = evaluate_target_prediction_plausibility(115.55, domain=domain)
    assert result.status is TargetPredictionPlausibilityStatus.OUTSIDE_DECLARED_DOMAIN
    assert result.raw_prediction == pytest.approx(115.55)
    assert result.warning_message is not None
    assert "must not be interpreted" in result.warning_message.lower()


def test_absent_declared_domain_uses_observed_only() -> None:
    domain = TargetPredictionDomain(
        observed_minimum=50.0,
        observed_maximum=90.0,
    )
    inside = evaluate_target_prediction_plausibility(70.0, domain=domain)
    outside = evaluate_target_prediction_plausibility(95.0, domain=domain)
    assert inside.status is TargetPredictionPlausibilityStatus.WITHIN_OBSERVED_RANGE
    assert outside.status is TargetPredictionPlausibilityStatus.OUTSIDE_OBSERVED_RANGE


def test_domain_unavailable_when_no_bounds() -> None:
    result = evaluate_target_prediction_plausibility(70.0, domain=None)
    assert result.status is TargetPredictionPlausibilityStatus.DOMAIN_UNAVAILABLE


def test_non_finite_prediction_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        evaluate_target_prediction_plausibility(math.nan, domain=None)
    with pytest.raises(ValueError, match="finite"):
        evaluate_target_prediction_plausibility(math.inf, domain=None)


def test_plausibility_assessment_serialization_round_trip() -> None:
    assessment = assess_recommendation_target_plausibility(
        baseline_prediction=96.5,
        proposed_prediction=115.5,
        domain=TargetPredictionDomain(
            observed_minimum=45.0,
            observed_maximum=98.0,
            declared_minimum=40.0,
            declared_maximum=100.0,
        ),
    )
    payload = assessment.model_dump(mode="json")
    restored = RecommendationTargetPlausibility.model_validate(payload)
    assert restored.baseline_status is TargetPredictionPlausibilityStatus.WITHIN_OBSERVED_RANGE
    assert (
        restored.proposed_status
        is TargetPredictionPlausibilityStatus.OUTSIDE_DECLARED_DOMAIN
    )
    assert restored.proposed_raw_prediction == pytest.approx(115.5)
    assert restored.warning_messages


def test_builtin_demo_supplies_declared_quality_domain() -> None:
    preset = build_demo_configuration_preset(
        DemoAnalysisTemplate.SUPERVISED_QUALITY_PREDICTION
    )
    assert preset.declared_target_minimum == DEMO_QUALITY_SCORE_DECLARED_MINIMUM
    assert preset.declared_target_maximum == DEMO_QUALITY_SCORE_DECLARED_MAXIMUM
    assert preset.declared_target_minimum == 40.0
    assert preset.declared_target_maximum == 100.0


def test_uploaded_csv_configuration_does_not_receive_demo_bounds() -> None:
    assert DATA_SOURCE_UPLOAD_CSV == "Upload CSV"
    submission = WorkflowUiSubmission(
        analysis_mode=AnalysisExecutionMode.SUPERVISED,
        target_column="y",
        feature_columns=["x1", "x2"],
        requested_task=AnalysisTask.REGRESSION,
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        quality_direction=QualityOptimizationDirection.MAXIMIZE,
        performance_rules=[
            UiMetricRuleInput(
                metric_name="rmse",
                direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
                threshold=1.0,
                required=True,
            )
        ],
        constraints=[
            UiVariableConstraintInput(variable="x1", minimum=0.0, maximum=1.0),
        ],
        user_confirmed_controllable_variables=["x1"],
        user_verified_variables=["x1"],
        max_simultaneous_changes=1,
    )
    assert submission.declared_target_minimum is None
    assert submission.declared_target_maximum is None


def test_does_not_silently_clip_outside_declared_prediction() -> None:
    assessment = assess_recommendation_target_plausibility(
        baseline_prediction=90.0,
        proposed_prediction=115.55,
        domain=build_target_prediction_domain(
            [40.0, 100.0],
            declared_minimum=40.0,
            declared_maximum=100.0,
        ),
    )
    assert assessment.proposed_raw_prediction == pytest.approx(115.55)
    assert assessment.proposed_raw_prediction is not None
    assert assessment.proposed_raw_prediction > 100.0
