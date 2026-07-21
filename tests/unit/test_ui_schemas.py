"""Unit tests for Streamlit UI Pydantic schemas (Step 11B)."""

from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import ValidationError

from process_intelligence.core.enums import ColumnRole
from process_intelligence.evaluation import MetricAcceptanceDirection
from process_intelligence.recommendation import (
    QualityOptimizationDirection,
    RecommendationObjective,
)
from process_intelligence.ui import (
    StreamlitUiConfig,
    UiMetricRuleInput,
    UiVariableConstraintInput,
    WorkflowUiSubmission,
)
from process_intelligence.workflow import OperatingPointSelectionMode


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


def _submission(**overrides: Any) -> WorkflowUiSubmission:
    payload: dict[str, Any] = {
        "target_column": "quality",
        "feature_columns": ["pressure", "temperature"],
        "timestamp_column": None,
        "identifier_columns": [],
        "excluded_columns": [],
        "column_role_overrides": {},
        "objective": RecommendationObjective.REDUCE_ANOMALY_SCORE,
        "quality_direction": None,
        "quality_target": None,
        "performance_rules": [_rule()],
        "constraints": [_constraint()],
        "user_confirmed_controllable_variables": ["pressure"],
        "user_verified_variables": ["pressure"],
        "max_simultaneous_changes": 2,
        "operating_point_selection": OperatingPointSelectionMode.TOP_RESIDUAL_ANOMALY,
        "explicit_operating_row_id": None,
        "metadata": {"source": "unit-test"},
    }
    payload.update(overrides)
    return WorkflowUiSubmission(**payload)


def test_valid_metric_rule() -> None:
    rule = _rule()
    assert rule.metric_name == "rmse"
    assert rule.direction is MetricAcceptanceDirection.LOWER_IS_BETTER
    assert rule.threshold == 10.0
    assert rule.required is True


@pytest.mark.parametrize("threshold", [True, float("nan"), float("inf"), float("-inf")])
def test_metric_threshold_rejects_bool_nan_inf(threshold: object) -> None:
    with pytest.raises(ValidationError):
        _rule(threshold=threshold)


def test_valid_constraint() -> None:
    item = _constraint()
    assert item.variable == "pressure"
    assert item.minimum < item.maximum


def test_constraint_rejects_minimum_ge_maximum() -> None:
    with pytest.raises(ValidationError):
        _constraint(minimum=10.0, maximum=10.0)
    with pytest.raises(ValidationError):
        _constraint(minimum=20.0, maximum=10.0)


def test_valid_submission() -> None:
    submission = _submission()
    assert submission.target_column == "quality"
    assert len(submission.performance_rules) == 1


def test_metric_duplicate_rejected() -> None:
    with pytest.raises(ValidationError):
        _submission(performance_rules=[_rule(), _rule(metric_name="rmse")])


def test_constraint_duplicate_rejected() -> None:
    with pytest.raises(ValidationError):
        _submission(
            constraints=[
                _constraint(variable="pressure"),
                _constraint(variable="pressure", minimum=1.0, maximum=2.0),
            ]
        )


def test_feature_constraint_relationship() -> None:
    with pytest.raises(ValidationError):
        _submission(constraints=[_constraint(variable="missing_feature")])


def test_confirmation_verification_relationship() -> None:
    with pytest.raises(ValidationError):
        _submission(user_confirmed_controllable_variables=["missing"])
    with pytest.raises(ValidationError):
        _submission(user_verified_variables=["missing"])


def test_objective_direction_relationship() -> None:
    with pytest.raises(ValidationError):
        _submission(
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
            quality_direction=None,
        )
    submission = _submission(
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        quality_direction=QualityOptimizationDirection.MAXIMIZE,
    )
    assert submission.quality_direction is QualityOptimizationDirection.MAXIMIZE


def test_target_relationship() -> None:
    with pytest.raises(ValidationError):
        _submission(
            objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
            quality_direction=QualityOptimizationDirection.TARGET,
            quality_target=None,
        )
    with pytest.raises(ValidationError):
        _submission(
            objective=RecommendationObjective.REDUCE_ANOMALY_SCORE,
            quality_direction=None,
            quality_target=1.0,
        )
    submission = _submission(
        objective=RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
        quality_direction=QualityOptimizationDirection.TARGET,
        quality_target=90.0,
    )
    assert submission.quality_target == 90.0


def test_operating_row_relationship() -> None:
    with pytest.raises(ValidationError):
        _submission(
            operating_point_selection=OperatingPointSelectionMode.EXPLICIT_ROW_ID,
            explicit_operating_row_id=None,
        )
    with pytest.raises(ValidationError):
        _submission(
            operating_point_selection=OperatingPointSelectionMode.LATEST_ROW,
            explicit_operating_row_id=7,
        )
    submission = _submission(
        operating_point_selection=OperatingPointSelectionMode.EXPLICIT_ROW_ID,
        explicit_operating_row_id=12,
    )
    assert submission.explicit_operating_row_id == 12


@pytest.mark.parametrize("value", [0, True, False, -1])
def test_max_changes_rejects_invalid(value: object) -> None:
    with pytest.raises(ValidationError):
        _submission(max_simultaneous_changes=value)


def test_metadata_scalar_restriction() -> None:
    with pytest.raises(ValidationError):
        _submission(metadata={"bad": {"nested": 1}})
    with pytest.raises(ValidationError):
        _submission(metadata={"bad": float("nan")})


def test_mutable_state_independence() -> None:
    rules = [_rule()]
    constraints = [_constraint()]
    metadata = {"a": 1}
    submission = _submission(
        performance_rules=rules,
        constraints=constraints,
        metadata=metadata,
        column_role_overrides={"pressure": ColumnRole.STATE_SENSOR},
    )
    rules.append(_rule(metric_name="mae"))
    constraints.append(_constraint(variable="temperature", minimum=1.0, maximum=2.0))
    metadata["b"] = 2
    assert len(submission.performance_rules) == 1
    assert len(submission.constraints) == 1
    assert "b" not in submission.metadata


def test_config_strict_validation() -> None:
    config = StreamlitUiConfig()
    assert config.preview_row_count == 20
    with pytest.raises(ValidationError):
        StreamlitUiConfig(preview_row_count=0)
    with pytest.raises(ValidationError):
        StreamlitUiConfig(maximum_upload_bytes=True)
    with pytest.raises(ValidationError):
        StreamlitUiConfig(page_title="")
    with pytest.raises(ValidationError):
        StreamlitUiConfig(extra_field=1)  # type: ignore[call-arg]


def test_round_trip() -> None:
    submission = _submission(
        column_role_overrides={"pressure": ColumnRole.CONTROLLABLE_PROCESS},
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        quality_direction=QualityOptimizationDirection.MINIMIZE,
    )
    dumped = submission.model_dump(mode="json")
    restored = WorkflowUiSubmission.model_validate(dumped)
    assert restored == submission
    assert not math.isnan(restored.performance_rules[0].threshold)
