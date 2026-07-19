"""Unit tests for Data Quality Score calculation (Step 2D)."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from process_intelligence.core.exceptions import DataValidationError
from process_intelligence.core.schemas import ValidationIssue
from process_intelligence.data import (
    DEFAULT_ISSUE_TYPE_MULTIPLIERS,
    DEFAULT_SEVERITY_WEIGHTS,
    ISSUE_TYPE_DIMENSIONS,
    ISSUE_TYPE_MODELING_IMPACTS,
    DataQualityScore,
    DataQualityScorer,
    DataQualityWeights,
    QualityPenalty,
)


def _issue(
    *,
    issue_type: str = "DUPLICATE_ROWS",
    column: str | None = None,
    severity: str = "WARNING",
    message: str = "duplicate rows found",
    suggested_action: str | None = "review duplicates",
) -> ValidationIssue:
    return ValidationIssue(
        issue_type=issue_type,
        column=column,
        severity=severity,
        message=message,
        suggested_action=suggested_action,
    )


def test_data_quality_weights_default_construction() -> None:
    weights = DataQualityWeights()
    assert isinstance(weights, DataQualityWeights)


def test_default_severity_weight_values() -> None:
    assert DEFAULT_SEVERITY_WEIGHTS["ERROR"] == pytest.approx(20.0)
    assert DEFAULT_SEVERITY_WEIGHTS["WARNING"] == pytest.approx(8.0)
    assert DEFAULT_SEVERITY_WEIGHTS["INFO"] == pytest.approx(2.0)
    weights = DataQualityWeights()
    assert weights.severity_weights["ERROR"] == pytest.approx(20.0)
    assert weights.severity_weights["WARNING"] == pytest.approx(8.0)
    assert weights.severity_weights["INFO"] == pytest.approx(2.0)


def test_default_issue_multiplier_values() -> None:
    expected = {
        "EMPTY_DATASET": 5.0,
        "DUPLICATE_ROWS": 1.0,
        "ALL_NULL_COLUMN": 1.5,
        "HIGH_MISSING_RATIO": 1.0,
        "CONSTANT_COLUMN": 0.75,
        "NEAR_CONSTANT_COLUMN": 0.5,
        "NAN_VALUES": 1.0,
        "INFINITE_VALUES": 1.5,
        "NUMERIC_STRING_COLUMN": 0.75,
    }
    for key, value in expected.items():
        assert DEFAULT_ISSUE_TYPE_MULTIPLIERS[key] == pytest.approx(value)
    weights = DataQualityWeights()
    for key, value in expected.items():
        assert weights.issue_type_multipliers[key] == pytest.approx(value)


def test_default_mappings_are_not_externally_mutable() -> None:
    with pytest.raises(TypeError):
        DEFAULT_SEVERITY_WEIGHTS["ERROR"] = 99.0  # type: ignore[index]
    with pytest.raises(TypeError):
        DEFAULT_ISSUE_TYPE_MULTIPLIERS["EMPTY_DATASET"] = 99.0  # type: ignore[index]
    with pytest.raises(TypeError):
        ISSUE_TYPE_DIMENSIONS["EMPTY_DATASET"] = "OTHER"  # type: ignore[index]
    with pytest.raises(TypeError):
        ISSUE_TYPE_MODELING_IMPACTS["EMPTY_DATASET"] = "changed"  # type: ignore[index]


def test_default_weight_instances_do_not_share_dicts() -> None:
    first = DataQualityWeights()
    second = DataQualityWeights()
    assert first.severity_weights is not second.severity_weights
    assert first.issue_type_multipliers is not second.issue_type_multipliers
    first.severity_weights["ERROR"] = 1.0
    assert second.severity_weights["ERROR"] == pytest.approx(20.0)


@pytest.mark.parametrize("missing_key", ["ERROR", "WARNING", "INFO"])
def test_missing_required_severity_key_fails_validation(missing_key: str) -> None:
    payload = {
        "ERROR": 20.0,
        "WARNING": 8.0,
        "INFO": 2.0,
    }
    del payload[missing_key]
    with pytest.raises(ValidationError):
        DataQualityWeights(severity_weights=payload)


def test_negative_severity_weight_rejected() -> None:
    with pytest.raises(ValidationError):
        DataQualityWeights(
            severity_weights={"ERROR": -1.0, "WARNING": 8.0, "INFO": 2.0}
        )


def test_negative_issue_multiplier_rejected() -> None:
    with pytest.raises(ValidationError):
        DataQualityWeights(issue_type_multipliers={"DUPLICATE_ROWS": -0.5})


def test_bool_weight_rejected() -> None:
    with pytest.raises(ValidationError):
        DataQualityWeights(
            severity_weights={"ERROR": True, "WARNING": 8.0, "INFO": 2.0}
        )
    with pytest.raises(ValidationError):
        DataQualityWeights(issue_type_multipliers={"DUPLICATE_ROWS": False})


def test_nan_weight_rejected() -> None:
    with pytest.raises(ValidationError):
        DataQualityWeights(
            severity_weights={"ERROR": math.nan, "WARNING": 8.0, "INFO": 2.0}
        )
    with pytest.raises(ValidationError):
        DataQualityWeights(issue_type_multipliers={"DUPLICATE_ROWS": math.nan})


def test_positive_infinity_weight_rejected() -> None:
    with pytest.raises(ValidationError):
        DataQualityWeights(
            severity_weights={
                "ERROR": math.inf,
                "WARNING": 8.0,
                "INFO": 2.0,
            }
        )
    with pytest.raises(ValidationError):
        DataQualityWeights(issue_type_multipliers={"DUPLICATE_ROWS": math.inf})


def test_negative_infinity_weight_rejected() -> None:
    with pytest.raises(ValidationError):
        DataQualityWeights(
            severity_weights={
                "ERROR": -math.inf,
                "WARNING": 8.0,
                "INFO": 2.0,
            }
        )
    with pytest.raises(ValidationError):
        DataQualityWeights(issue_type_multipliers={"DUPLICATE_ROWS": -math.inf})


def test_data_quality_scorer_default_construction() -> None:
    scorer = DataQualityScorer()
    assert isinstance(scorer, DataQualityScorer)


def test_empty_issues_total_score_is_100() -> None:
    result = DataQualityScorer().score([])
    assert result.total_score == pytest.approx(100.0)


def test_empty_issues_total_deduction_is_zero() -> None:
    result = DataQualityScorer().score([])
    assert result.total_deduction == pytest.approx(0.0)


def test_empty_issues_issue_count_is_zero() -> None:
    result = DataQualityScorer().score([])
    assert result.issue_count == 0


def test_empty_issues_severity_counts_include_all_levels() -> None:
    result = DataQualityScorer().score([])
    assert result.severity_counts == {"ERROR": 0, "WARNING": 0, "INFO": 0}


def test_empty_issues_dimension_deductions_include_all_defaults() -> None:
    result = DataQualityScorer().score([])
    assert set(result.dimension_deductions) == {
        "COMPLETENESS",
        "UNIQUENESS",
        "VALIDITY",
        "VARIABILITY",
        "OTHER",
    }
    assert all(
        value == pytest.approx(0.0) for value in result.dimension_deductions.values()
    )


def test_error_issue_deduction_is_correct() -> None:
    result = DataQualityScorer().score(
        [_issue(issue_type="NAN_VALUES", severity="ERROR", column="x")]
    )
    assert result.penalties[0].deducted_points == pytest.approx(20.0)
    assert result.total_score == pytest.approx(80.0)


def test_warning_issue_deduction_is_correct() -> None:
    result = DataQualityScorer().score([_issue(issue_type="DUPLICATE_ROWS")])
    assert result.penalties[0].deducted_points == pytest.approx(8.0)
    assert result.total_score == pytest.approx(92.0)


def test_info_issue_deduction_is_correct() -> None:
    result = DataQualityScorer().score(
        [_issue(issue_type="NEAR_CONSTANT_COLUMN", severity="INFO", column="x")]
    )
    assert result.penalties[0].deducted_points == pytest.approx(1.0)
    assert result.total_score == pytest.approx(99.0)


def test_issue_type_multiplier_is_applied() -> None:
    result = DataQualityScorer().score(
        [_issue(issue_type="EMPTY_DATASET", severity="ERROR")]
    )
    assert result.penalties[0].multiplier == pytest.approx(5.0)
    assert result.penalties[0].deducted_points == pytest.approx(100.0)


def test_unregistered_issue_type_uses_multiplier_one() -> None:
    result = DataQualityScorer().score(
        [_issue(issue_type="CUSTOM_ISSUE", severity="WARNING")]
    )
    assert result.penalties[0].multiplier == pytest.approx(1.0)
    assert result.penalties[0].deducted_points == pytest.approx(8.0)


def test_deduction_rounded_to_two_decimal_places() -> None:
    weights = DataQualityWeights(
        severity_weights={"ERROR": 20.0, "WARNING": 8.0, "INFO": 2.0},
        issue_type_multipliers={"CUSTOM": 0.333},
    )
    result = DataQualityScorer(weights=weights).score(
        [_issue(issue_type="CUSTOM", severity="WARNING")]
    )
    assert result.penalties[0].deducted_points == pytest.approx(2.66)


def test_multiple_issue_deductions_sum_correctly() -> None:
    issues = [
        _issue(issue_type="DUPLICATE_ROWS", severity="WARNING"),
        _issue(issue_type="NAN_VALUES", severity="WARNING", column="a"),
        _issue(issue_type="CONSTANT_COLUMN", severity="WARNING", column="b"),
    ]
    result = DataQualityScorer().score(issues)
    assert result.total_deduction == pytest.approx(22.0)
    assert result.total_score == pytest.approx(78.0)


def test_total_score_floors_at_zero_when_deduction_exceeds_100() -> None:
    issues = [
        _issue(issue_type="EMPTY_DATASET", severity="ERROR"),
        _issue(issue_type="INFINITE_VALUES", severity="ERROR", column="x"),
    ]
    result = DataQualityScorer().score(issues)
    assert result.total_score == pytest.approx(0.0)


def test_total_deduction_may_exceed_100() -> None:
    issues = [
        _issue(issue_type="EMPTY_DATASET", severity="ERROR"),
        _issue(issue_type="INFINITE_VALUES", severity="ERROR", column="x"),
    ]
    result = DataQualityScorer().score(issues)
    assert result.total_deduction == pytest.approx(130.0)
    assert result.total_deduction > 100.0


def test_penalties_preserve_input_order() -> None:
    issues = [
        _issue(issue_type="NAN_VALUES", severity="WARNING", column="a", message="a"),
        _issue(issue_type="EMPTY_DATASET", severity="ERROR", message="b"),
        _issue(issue_type="DUPLICATE_ROWS", severity="WARNING", message="c"),
    ]
    result = DataQualityScorer().score(issues)
    assert [penalty.reason for penalty in result.penalties] == ["a", "b", "c"]
    assert [penalty.issue_type for penalty in result.penalties] == [
        "NAN_VALUES",
        "EMPTY_DATASET",
        "DUPLICATE_ROWS",
    ]


def test_input_issue_list_is_not_mutated() -> None:
    issues = [
        _issue(issue_type="DUPLICATE_ROWS", message="original"),
        _issue(issue_type="NAN_VALUES", column="x", message="nan"),
    ]
    original_snapshot = [
        (
            issue.issue_type,
            issue.column,
            issue.severity,
            issue.message,
            issue.suggested_action,
        )
        for issue in issues
    ]
    DataQualityScorer().score(issues)
    after_snapshot = [
        (
            issue.issue_type,
            issue.column,
            issue.severity,
            issue.message,
            issue.suggested_action,
        )
        for issue in issues
    ]
    assert after_snapshot == original_snapshot
    assert len(issues) == 2


def test_repeated_identical_issues_each_deduct() -> None:
    issues = [
        _issue(issue_type="DUPLICATE_ROWS", message="dup-1"),
        _issue(issue_type="DUPLICATE_ROWS", message="dup-2"),
    ]
    result = DataQualityScorer().score(issues)
    assert result.issue_count == 2
    assert result.total_deduction == pytest.approx(16.0)
    assert len(result.penalties) == 2


def test_severity_counts_are_accurate() -> None:
    issues = [
        _issue(severity="ERROR", issue_type="INFINITE_VALUES", column="x"),
        _issue(severity="WARNING", issue_type="DUPLICATE_ROWS"),
        _issue(severity="WARNING", issue_type="NAN_VALUES", column="y"),
        _issue(severity="INFO", issue_type="NEAR_CONSTANT_COLUMN", column="z"),
    ]
    result = DataQualityScorer().score(issues)
    assert result.severity_counts == {"ERROR": 1, "WARNING": 2, "INFO": 1}


def test_issue_type_counts_are_accurate() -> None:
    issues = [
        _issue(issue_type="DUPLICATE_ROWS"),
        _issue(issue_type="NAN_VALUES", column="a"),
        _issue(issue_type="DUPLICATE_ROWS"),
    ]
    result = DataQualityScorer().score(issues)
    assert result.issue_type_counts == {"DUPLICATE_ROWS": 2, "NAN_VALUES": 1}


def test_issue_type_counts_preserve_first_seen_order() -> None:
    issues = [
        _issue(issue_type="NAN_VALUES", column="a"),
        _issue(issue_type="CONSTANT_COLUMN", column="b"),
        _issue(issue_type="NAN_VALUES", column="c"),
        _issue(issue_type="DUPLICATE_ROWS"),
    ]
    result = DataQualityScorer().score(issues)
    assert list(result.issue_type_counts.keys()) == [
        "NAN_VALUES",
        "CONSTANT_COLUMN",
        "DUPLICATE_ROWS",
    ]


def test_dimension_deductions_are_accurate() -> None:
    issues = [
        _issue(issue_type="HIGH_MISSING_RATIO", severity="WARNING", column="a"),
        _issue(issue_type="DUPLICATE_ROWS", severity="WARNING"),
        _issue(issue_type="NAN_VALUES", severity="WARNING", column="b"),
        _issue(issue_type="CONSTANT_COLUMN", severity="WARNING", column="c"),
    ]
    result = DataQualityScorer().score(issues)
    assert result.dimension_deductions["COMPLETENESS"] == pytest.approx(8.0)
    assert result.dimension_deductions["UNIQUENESS"] == pytest.approx(8.0)
    assert result.dimension_deductions["VALIDITY"] == pytest.approx(8.0)
    assert result.dimension_deductions["VARIABILITY"] == pytest.approx(6.0)
    assert result.dimension_deductions["OTHER"] == pytest.approx(0.0)


def test_unregistered_issue_type_aggregates_to_other_dimension() -> None:
    result = DataQualityScorer().score(
        [_issue(issue_type="UNKNOWN_TYPE", severity="INFO")]
    )
    assert result.penalties[0].dimension == "OTHER"
    assert result.dimension_deductions["OTHER"] == pytest.approx(2.0)


def test_quality_penalty_preserves_original_message() -> None:
    message = "Column 'temp' has missing ratio 0.4000"
    result = DataQualityScorer().score(
        [_issue(issue_type="HIGH_MISSING_RATIO", column="temp", message=message)]
    )
    assert result.penalties[0].reason == message


def test_quality_penalty_preserves_recommended_action() -> None:
    action = "Impute or drop the column after review."
    result = DataQualityScorer().score(
        [
            _issue(
                issue_type="HIGH_MISSING_RATIO",
                column="temp",
                suggested_action=action,
            )
        ]
    )
    assert result.penalties[0].recommended_action == action


def test_known_issue_type_modeling_impact_is_non_empty() -> None:
    for issue_type in DEFAULT_ISSUE_TYPE_MULTIPLIERS:
        impact = ISSUE_TYPE_MODELING_IMPACTS[issue_type]
        assert isinstance(impact, str)
        assert impact.strip() != ""
        result = DataQualityScorer().score(
            [_issue(issue_type=issue_type, severity="WARNING")]
        )
        assert result.penalties[0].modeling_impact.strip() != ""


def test_unregistered_issue_type_uses_default_modeling_impact() -> None:
    result = DataQualityScorer().score(
        [_issue(issue_type="BRAND_NEW_ISSUE", severity="INFO")]
    )
    assert result.penalties[0].modeling_impact == (
        "Unknown impact on downstream modeling; review required."
    )


def test_lowercase_severity_is_normalized() -> None:
    result = DataQualityScorer().score(
        [_issue(issue_type="DUPLICATE_ROWS", severity="warning")]
    )
    assert result.penalties[0].severity == "WARNING"
    assert result.severity_counts["WARNING"] == 1
    assert result.penalties[0].deducted_points == pytest.approx(8.0)


def test_unknown_severity_raises_data_validation_error() -> None:
    with pytest.raises(DataValidationError):
        DataQualityScorer().score(
            [_issue(issue_type="DUPLICATE_ROWS", severity="CRITICAL")]
        )


def test_str_input_raises_type_error() -> None:
    with pytest.raises(TypeError):
        DataQualityScorer().score("not-issues")  # type: ignore[arg-type]


def test_bytes_input_raises_type_error() -> None:
    with pytest.raises(TypeError):
        DataQualityScorer().score(b"not-issues")  # type: ignore[arg-type]


def test_non_validation_issue_element_raises_type_error() -> None:
    with pytest.raises(TypeError):
        DataQualityScorer().score([_issue(), {"issue_type": "x"}])  # type: ignore[list-item]


def test_list_input_succeeds() -> None:
    result = DataQualityScorer().score([_issue()])
    assert result.issue_count == 1


def test_tuple_input_succeeds() -> None:
    result = DataQualityScorer().score((_issue(),))
    assert result.issue_count == 1


def test_passed_weights_object_is_not_mutated_during_score() -> None:
    weights = DataQualityWeights()
    severity_before = dict(weights.severity_weights)
    multipliers_before = dict(weights.issue_type_multipliers)
    DataQualityScorer(weights=weights).score(
        [
            _issue(issue_type="EMPTY_DATASET", severity="ERROR"),
            _issue(issue_type="CUSTOM_ISSUE", severity="WARNING"),
        ]
    )
    assert weights.severity_weights == severity_before
    assert weights.issue_type_multipliers == multipliers_before


def test_scorer_instances_do_not_share_state() -> None:
    first = DataQualityScorer(
        weights=DataQualityWeights(
            severity_weights={"ERROR": 10.0, "WARNING": 5.0, "INFO": 1.0},
            issue_type_multipliers={"DUPLICATE_ROWS": 2.0},
        )
    )
    second = DataQualityScorer()
    first_result = first.score([_issue(issue_type="DUPLICATE_ROWS")])
    second_result = second.score([_issue(issue_type="DUPLICATE_ROWS")])
    assert first_result.total_deduction == pytest.approx(10.0)
    assert second_result.total_deduction == pytest.approx(8.0)


def test_consecutive_scores_do_not_accumulate() -> None:
    scorer = DataQualityScorer()
    first = scorer.score([_issue(issue_type="DUPLICATE_ROWS")])
    second = scorer.score([])
    third = scorer.score(
        [
            _issue(issue_type="EMPTY_DATASET", severity="ERROR"),
        ]
    )
    assert first.total_deduction == pytest.approx(8.0)
    assert second.total_score == pytest.approx(100.0)
    assert second.issue_count == 0
    assert third.total_deduction == pytest.approx(100.0)
    assert third.issue_count == 1


def test_data_quality_score_model_dump_round_trip() -> None:
    original = DataQualityScorer().score(
        [
            _issue(
                issue_type="HIGH_MISSING_RATIO",
                column="temp",
                severity="WARNING",
                message="missing high",
                suggested_action="impute",
            )
        ]
    )
    restored = DataQualityScore.model_validate(original.model_dump())
    assert restored == original
    assert isinstance(restored.penalties[0], QualityPenalty)
