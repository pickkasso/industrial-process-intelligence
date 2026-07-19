"""Unit tests for string-based domain enumerations."""

import json

import pytest

from process_intelligence.core import AnalysisTask, AnomalyType, ColumnRole

EXPECTED_COLUMN_ROLES = (
    "IDENTIFIER",
    "TIME",
    "CONTROLLABLE_PROCESS",
    "STATE_SENSOR",
    "CONTEXT",
    "TARGET_QUALITY",
    "DERIVED_FEATURE",
    "UNKNOWN",
)

EXPECTED_ANALYSIS_TASKS = (
    "REGRESSION",
    "CLASSIFICATION",
    "UNSUPERVISED_ANOMALY",
    "TIME_SERIES_ANOMALY",
    "RESIDUAL_ANOMALY",
    "DRIFT_DETECTION",
)

EXPECTED_ANOMALY_TYPES = (
    "DATA_QUALITY",
    "PROCESS_INPUT",
    "STATE_SENSOR",
    "QUALITY_OUTPUT",
    "RELATIONSHIP",
    "DRIFT",
    "MULTIVARIATE_COMBINATION",
)


def test_column_role_member_count() -> None:
    assert len(ColumnRole) == 8


def test_analysis_task_member_count() -> None:
    assert len(AnalysisTask) == 6


def test_anomaly_type_member_count() -> None:
    assert len(AnomalyType) == 7


def test_column_role_member_names_match_document() -> None:
    assert tuple(member.name for member in ColumnRole) == EXPECTED_COLUMN_ROLES


def test_analysis_task_member_names_match_document() -> None:
    assert tuple(member.name for member in AnalysisTask) == EXPECTED_ANALYSIS_TASKS


def test_anomaly_type_member_names_match_document() -> None:
    assert tuple(member.name for member in AnomalyType) == EXPECTED_ANOMALY_TYPES


def test_column_role_values_match_document() -> None:
    assert tuple(member.value for member in ColumnRole) == EXPECTED_COLUMN_ROLES


def test_analysis_task_values_match_document() -> None:
    assert tuple(member.value for member in AnalysisTask) == EXPECTED_ANALYSIS_TASKS


def test_anomaly_type_values_match_document() -> None:
    assert tuple(member.value for member in AnomalyType) == EXPECTED_ANOMALY_TYPES


def test_enums_can_be_created_from_string_values() -> None:
    assert ColumnRole("IDENTIFIER") is ColumnRole.IDENTIFIER
    assert AnalysisTask("REGRESSION") is AnalysisTask.REGRESSION
    assert AnomalyType("DATA_QUALITY") is AnomalyType.DATA_QUALITY


def test_invalid_string_values_raise_value_error() -> None:
    with pytest.raises(ValueError):
        ColumnRole("NOT_A_ROLE")
    with pytest.raises(ValueError):
        AnalysisTask("NOT_A_TASK")
    with pytest.raises(ValueError):
        AnomalyType("NOT_AN_ANOMALY")


def test_enums_are_string_based_for_serialization() -> None:
    assert isinstance(ColumnRole.IDENTIFIER, str)
    assert isinstance(AnalysisTask.REGRESSION, str)
    assert isinstance(AnomalyType.DATA_QUALITY, str)

    assert ColumnRole.IDENTIFIER == "IDENTIFIER"
    assert AnalysisTask.REGRESSION == "REGRESSION"
    assert AnomalyType.DATA_QUALITY == "DATA_QUALITY"

    payload = {
        "column_role": ColumnRole.IDENTIFIER,
        "analysis_task": AnalysisTask.REGRESSION,
        "anomaly_type": AnomalyType.DATA_QUALITY,
    }
    assert json.dumps(payload) == (
        '{"column_role": "IDENTIFIER", '
        '"analysis_task": "REGRESSION", '
        '"anomaly_type": "DATA_QUALITY"}'
    )
