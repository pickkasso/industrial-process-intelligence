"""Unit tests for core Pydantic schemas."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from process_intelligence.core import (
    AnalysisTask,
    AnomalyEvent,
    AnomalyType,
    ColumnRole,
    ColumnRoleAssignment,
    IndustryScore,
    ModelEvaluation,
    ModelSpec,
    PreprocessingEvent,
    RootCauseFactor,
    SchemaHints,
    VariableConstraint,
)


def test_preprocessing_event_creation() -> None:
    event = PreprocessingEvent(
        step_name="impute",
        affected_columns=["temp"],
        rows_before=10,
        rows_after=10,
        parameters={"strategy": "median"},
        warnings=[],
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert event.step_name == "impute"
    assert event.rows_before == 10
    assert event.rows_after == 10


@pytest.mark.parametrize("field_name", ["rows_before", "rows_after"])
def test_preprocessing_event_rejects_negative_row_counts(field_name: str) -> None:
    payload = {
        "step_name": "drop_na",
        "affected_columns": ["x"],
        "rows_before": 5,
        "rows_after": 4,
        "parameters": {},
        "warnings": [],
        "timestamp": datetime(2026, 1, 1, tzinfo=UTC),
        field_name: -1,
    }
    with pytest.raises(ValidationError):
        PreprocessingEvent.model_validate(payload)


def test_industry_score_confidence_bounds() -> None:
    valid = IndustryScore(
        industry_name="semiconductor",
        score=0.8,
        confidence=0.75,
        evidence=["keyword match"],
        uncertain_factors=[],
        requires_user_confirmation=False,
    )
    assert valid.confidence == 0.75

    with pytest.raises(ValidationError):
        IndustryScore(
            industry_name="semiconductor",
            score=0.8,
            confidence=1.5,
            evidence=[],
            uncertain_factors=[],
            requires_user_confirmation=True,
        )


def test_column_role_assignment_uses_column_role() -> None:
    assignment = ColumnRoleAssignment(
        column_name="setpoint",
        role=ColumnRole.CONTROLLABLE_PROCESS,
        confidence=0.9,
        evidence=["name hint"],
        alternative_roles=[ColumnRole.STATE_SENSOR],
        use_in_model=True,
    )
    assert assignment.role is ColumnRole.CONTROLLABLE_PROCESS
    assert assignment.alternative_roles == [ColumnRole.STATE_SENSOR]


def test_column_role_assignment_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        ColumnRoleAssignment(
            column_name="x",
            role=ColumnRole.UNKNOWN,
            confidence=-0.1,
            evidence=[],
            alternative_roles=[],
            use_in_model=False,
        )


def test_model_spec_rejects_negative_priority() -> None:
    with pytest.raises(ValidationError):
        ModelSpec(
            name="ridge",
            task=AnalysisTask.REGRESSION,
            estimator_key="ridge",
            priority=-1,
        )


def test_model_spec_rejects_non_positive_time_budget() -> None:
    with pytest.raises(ValidationError):
        ModelSpec(
            name="ridge",
            task=AnalysisTask.REGRESSION,
            estimator_key="ridge",
            priority=1,
            time_budget_seconds=0.0,
        )


def test_anomaly_event_creation() -> None:
    event = AnomalyEvent(
        anomaly_id="a-1",
        anomaly_type=AnomalyType.PROCESS_INPUT,
        anomaly_score=2.5,
        severity="high",
        sample_id="S001",
        model_confidence=0.8,
        detector="isolation_forest",
        rationale="elevated temperature",
    )
    assert event.anomaly_type is AnomalyType.PROCESS_INPUT
    assert event.contributing_variables == []


def test_anomaly_event_model_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        AnomalyEvent(
            anomaly_id="a-1",
            anomaly_type=AnomalyType.DRIFT,
            anomaly_score=1.0,
            severity="medium",
            model_confidence=1.1,
            detector="drift",
            rationale="shift detected",
        )


def test_root_cause_factor_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        RootCauseFactor(
            variable="temp",
            direction="increase",
            role=ColumnRole.STATE_SENSOR,
            controllable=False,
            evidence="contribution high",
            confidence=-0.01,
            needs_verification=True,
        )


def test_variable_constraint_creation() -> None:
    constraint = VariableConstraint(
        variable="pressure",
        adjustable=True,
        minimum=1.0,
        maximum=5.0,
        max_change_ratio=0.1,
        unit="bar",
        cost=2.0,
        equipment_operating_range=(0.5, 6.0),
        max_simultaneous_changes=2,
        fixed=False,
    )
    assert constraint.variable == "pressure"
    assert constraint.safety_constraints == []


def test_variable_constraint_rejects_minimum_greater_than_maximum() -> None:
    with pytest.raises(ValidationError):
        VariableConstraint(
            variable="x",
            adjustable=True,
            minimum=10.0,
            maximum=1.0,
            fixed=False,
        )


def test_variable_constraint_rejects_negative_max_change_ratio() -> None:
    with pytest.raises(ValidationError):
        VariableConstraint(
            variable="x",
            adjustable=True,
            max_change_ratio=-0.1,
            fixed=False,
        )


def test_variable_constraint_rejects_negative_cost() -> None:
    with pytest.raises(ValidationError):
        VariableConstraint(
            variable="x",
            adjustable=True,
            cost=-1.0,
            fixed=False,
        )


def test_variable_constraint_rejects_non_positive_max_simultaneous_changes() -> None:
    with pytest.raises(ValidationError):
        VariableConstraint(
            variable="x",
            adjustable=True,
            max_simultaneous_changes=0,
            fixed=False,
        )


def test_variable_constraint_rejects_inverted_equipment_operating_range() -> None:
    with pytest.raises(ValidationError):
        VariableConstraint(
            variable="x",
            adjustable=True,
            equipment_operating_range=(5.0, 1.0),
            fixed=False,
        )


def test_model_evaluation_mutable_defaults_are_independent() -> None:
    first = ModelEvaluation()
    second = ModelEvaluation()
    first.metrics["rmse"] = 1.2
    first.notes.append("note")
    assert second.metrics == {}
    assert second.notes == []


def test_schema_hints_mutable_defaults_are_independent() -> None:
    first = SchemaHints()
    second = SchemaHints()
    first.keywords.append("wafer")
    first.synonyms["temp"] = ["temperature"]
    assert second.keywords == []
    assert second.synonyms == {}


def test_schema_round_trip_via_model_dump_and_validate() -> None:
    original = AnomalyEvent(
        anomaly_id="a-2",
        anomaly_type=AnomalyType.QUALITY_OUTPUT,
        anomaly_score=3.0,
        severity="low",
        contributing_variables=["yield"],
        model_confidence=0.55,
        detector="residual",
        rationale="quality drop",
    )
    restored = AnomalyEvent.model_validate(original.model_dump())
    assert restored == original


def test_invalid_enum_string_fails_validation() -> None:
    with pytest.raises(ValidationError):
        ColumnRoleAssignment(
            column_name="x",
            role="NOT_A_ROLE",  # type: ignore[arg-type]
            confidence=0.5,
            evidence=[],
            alternative_roles=[],
            use_in_model=False,
        )
