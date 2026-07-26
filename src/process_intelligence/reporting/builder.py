"""Build UI/API-safe presentation DTOs from analysis workflow reports (Step 11A).

Maps public ``AnalysisWorkflowReport`` fields into JSON-serializable view models.
Does not mutate inputs, call models, recompute metrics, or generate recommendations.
"""

from __future__ import annotations

import math
import re
from typing import Any

from process_intelligence.core.schemas import AnomalyEvent, RootCauseFactor
from process_intelligence.evaluation.performance_acceptance import (
    MetricAcceptanceResult,
    ModelPerformanceAcceptanceReport,
)
from process_intelligence.recommendation.schemas import (
    RecommendationChange,
    RecommendationResult,
)
from process_intelligence.recommendation.what_if_verification import (
    RecommendationWhatIfVerificationResult,
    WhatIfVerificationScenario,
)
from process_intelligence.reporting.schemas import (
    AnomalyContextIdentifierValueView,
    AnomalyContextRowView,
    AnomalyContextValueView,
    AnomalyContextWindowView,
    AnomalyEventView,
    DiagnosisFactorView,
    ModelPerformanceView,
    PerformanceMetricView,
    RecommendationChangeView,
    RecommendationView,
    RecommendationWhatIfVerificationView,
    WhatIfVerificationScenarioView,
    WorkflowCohortFilterSummaryView,
    WorkflowDataSummaryView,
    WorkflowModelSummaryView,
    WorkflowOverviewView,
    WorkflowPresentationOutcome,
    WorkflowPresentationReport,
    WorkflowRoutingSummaryView,
    WorkflowStageView,
    stability_classification_message,
    stage_status_label,
)
from process_intelligence.workflow.enums import AnalysisExecutionMode, AnalysisWorkflowStatus
from process_intelligence.workflow.schemas import (
    AnalysisWorkflowReport,
    AnalysisWorkflowStageRecord,
    AnomalyContextWindow,
    ScalarMetadataValue,
)

_OMISSION_WARNING = (
    "Additional presentation warnings were omitted due to the configured limit."
)

_UNSAFE_WORDING_WARNING = (
    "Unsafe wording detected in recommendation rationale text."
)

_OPERATING_ROW_MISSING_WARNING = (
    "Selected operating row ID was not present among selected anomaly events."
)

_ANOMALY_ONLY_OVERVIEW = (
    "Anomaly-only analysis completed. Recommendation generation is not "
    "enabled for this analysis mode."
)

_DISCLAIMER_MODEL_BASED = (
    "These outputs are model-based decision support and not operational commands."
)
_DISCLAIMER_NO_CAUSATION = (
    "Diagnosis and prediction results describe associations and do not establish "
    "causation."
)
_DISCLAIMER_VERIFICATION = (
    "Proposed changes require domain, process safety, operational, and "
    "experimental verification."
)
_DISCLAIMER_NO_GUARANTEE = (
    "Real-process improvement is not guaranteed by these presentation results."
)
_DISCLAIMER_PERFORMANCE = (
    "Performance acceptance only confirms configured independent-test metric "
    "thresholds."
)
_DISCLAIMER_EXTRAPOLATION = (
    "Absence of extrapolation or uncertainty evaluation does not establish "
    "deployment safety."
)
_DISCLAIMER_STATISTICAL_ANOMALY = (
    "A statistical anomaly does not automatically mean a defect."
)
_DISCLAIMER_DOMAIN_INTERPRETATION = (
    "Domain verification is required before operational interpretation."
)

_ANOMALY_SCORE_DIRECTION = "higher_is_more_anomalous"

_EVIDENCE_FLOAT_PATTERN = re.compile(
    r"(?P<key>reference_median|anomaly_median|signed_location_difference|"
    r"robust_scale|robust_z_score|group_deviation_z|"
    r"raw_association_score|normalized_association_score|"
    r"normalized_ensemble_score)"
    r"=(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)

_EVIDENCE_STATUS_PATTERN = re.compile(
    r"robust_scale_status=(?P<value>[A-Za-z_]+)"
)

_PRESENTATION_METADATA_ALLOWLIST: frozenset[str] = frozenset(
    {
        "raw_csv_loaded",
        "raw_row_count",
        "processed_row_count",
        "cohort_row_count",
        "train_row_count",
        "validation_row_count",
        "test_row_count",
        "analysis_mode",
        "selected_industry",
        "selected_task",
        "inferred_task",
        "task_selection_source",
        "task_override_applied",
        "selected_supervised_model_available",
        "selected_anomaly_model_available",
        "residual_calibration_performed",
        "independent_test_evaluation_performed",
        "anomaly_event_count",
        "anomaly_event_selection_source",
        "diagnosis_performed",
        "diagnosis_source",
        "recommendation_pipeline_executed",
        "recommendation_generated",
        "recommendation_applicable",
        "row_identity_preserved",
        "test_used_for_model_selection",
        "test_used_for_threshold_calibration",
        "anomaly_score_direction",
        "model_performance_assessed",
        "model_performance_status",
        "model_performance_rule_count",
        "model_performance_required_rule_count",
        "model_performance_failed_rule_count",
        "model_performance_unavailable_rule_count",
        "model_performance_gate_bypassed",
        "extrapolation_computed",
        "uncertainty_computed",
        "workflow_orchestration_only",
        "target_column",
        "feature_count",
        "target_suitable",
        "target_unique_non_null_count",
        "target_refusal_code",
        "target_suitability_message",
    }
)

_FORBIDDEN_STAGE_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "path",
        "file_path",
        "csv_path",
        "absolute_path",
        "traceback",
        "exception",
        "stack",
        "model",
        "estimator",
        "dataframe",
        "frame",
        "ndarray",
        "predictions",
        "scores array",
        "scores_array",
        "raw_rows",
        "request object",
        "report object",
        "request",
        "report",
    }
)

_FORBIDDEN_STAGE_METADATA_SUBSTRINGS: tuple[str, ...] = (
    "path",
    "traceback",
    "exception",
    "stack",
    "dataframe",
    "ndarray",
    "estimator",
    "raw_rows",
    "predictions",
)

_UNSAFE_RATIONALE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"guaranteed", re.IGNORECASE),
    re.compile(r"proven\s+optimal", re.IGNORECASE),
    re.compile(r"will\s+improve", re.IGNORECASE),
    re.compile(r"will\s+fix", re.IGNORECASE),
    re.compile(r"confirmed\s+root\s+cause", re.IGNORECASE),
    re.compile(r"반드시\s*개선"),
    re.compile(r"최적\s*조건\s*확정"),
)

_HEADLINES: dict[AnalysisWorkflowStatus, str] = {
    AnalysisWorkflowStatus.COMPLETED: (
        "Analysis completed with a generated recommendation"
    ),
    AnalysisWorkflowStatus.PARTIAL: (
        "Analysis completed without an executable recommendation"
    ),
    AnalysisWorkflowStatus.REFUSED: (
        "Analysis was stopped by a safety or validation gate"
    ),
}


def _require_strict_bool(value: object, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(
            f"{field_name} must be a bool (0/1 and strings rejected), "
            f"got {type(value).__name__}"
        )
    return value


def _require_strict_int_ge1(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{field_name} must be an int >= 1 (bool not allowed), "
            f"got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1, got {value}")
    return value


def _metadata_bool(
    metadata: dict[str, ScalarMetadataValue],
    key: str,
    *,
    default: bool,
) -> bool:
    if key not in metadata:
        return default
    value = metadata[key]
    if type(value) is not bool:
        raise ValueError(
            f"workflow metadata[{key!r}] must be bool when present, "
            f"got {type(value).__name__}"
        )
    return value


def _metadata_optional_str(
    metadata: dict[str, ScalarMetadataValue],
    key: str,
) -> str | None:
    if key not in metadata:
        return None
    value = metadata[key]
    if value is None:
        return None
    if not isinstance(value, str) or value == "" or value.strip() == "":
        raise ValueError(
            f"workflow metadata[{key!r}] must be a non-empty str or None when present"
        )
    return value


def _is_allowed_model_key_name(key_lower: str) -> bool:
    return key_lower == "model_key" or key_lower.endswith("_model_key")


def _is_forbidden_stage_metadata_key(key: str) -> bool:
    key_lower = key.casefold()
    if key_lower in _FORBIDDEN_STAGE_METADATA_KEYS:
        return True
    if _is_allowed_model_key_name(key_lower):
        return False
    for token in _FORBIDDEN_STAGE_METADATA_SUBSTRINGS:
        if token in key_lower:
            return True
    if key_lower == "model" or key_lower.endswith(".model"):
        return True
    return False


def _filter_stage_metadata(
    metadata: dict[str, ScalarMetadataValue],
) -> dict[str, ScalarMetadataValue]:
    filtered: dict[str, ScalarMetadataValue] = {}
    for key, value in metadata.items():
        if _is_forbidden_stage_metadata_key(key):
            continue
        if value is None or isinstance(value, (str, bool)):
            filtered[key] = value
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            filtered[key] = value
            continue
        if isinstance(value, float):
            filtered[key] = value
            continue
    return filtered


def _filter_presentation_metadata(
    metadata: dict[str, ScalarMetadataValue],
) -> dict[str, ScalarMetadataValue]:
    filtered: dict[str, ScalarMetadataValue] = {}
    for key, value in metadata.items():
        if key not in _PRESENTATION_METADATA_ALLOWLIST:
            continue
        if value is None or isinstance(value, (str, bool)):
            filtered[key] = value
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            filtered[key] = value
            continue
        if isinstance(value, float):
            filtered[key] = value
            continue
    return filtered


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in values:
        if item == "" or item.strip() == "":
            continue
        if item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
    return cleaned


def _cap_warnings(warnings: list[str], *, maximum_warnings: int) -> list[str]:
    deduped = _dedupe_preserve_order(warnings)
    if len(deduped) <= maximum_warnings:
        return deduped
    if maximum_warnings == 1:
        return [_OMISSION_WARNING]
    kept = deduped[: maximum_warnings - 1]
    kept.append(_OMISSION_WARNING)
    return kept


def _normalize_row_id_for_compare(value: int | str | None) -> str | None:
    """Normalize a row identity to a JSON-safe comparison key without rewriting it."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return None
        return text
    return None


def _json_safe_row_id(value: object) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return None
        return text
    return None


def _extract_evidence_floats(evidence: str) -> dict[str, float]:
    extracted: dict[str, float] = {}
    for match in _EVIDENCE_FLOAT_PATTERN.finditer(evidence):
        key = match.group("key")
        raw = match.group("value")
        try:
            number = float(raw)
        except ValueError:
            continue
        if not math.isfinite(number):
            continue
        extracted[key] = number
    return extracted


def _extract_robust_scale_status(evidence: str) -> str | None:
    match = _EVIDENCE_STATUS_PATTERN.search(evidence)
    if match is None:
        return None
    value = match.group("value").strip()
    if value == "":
        return None
    return value


def _build_overview(report: AnalysisWorkflowReport) -> WorkflowOverviewView:
    recommendation_status = (
        report.final_recommendation.status
        if report.final_recommendation is not None
        else None
    )
    recommendation_text = (
        recommendation_status.value
        if recommendation_status is not None
        else "none"
    )
    refusal_code = report.metadata.get("target_refusal_code")
    if report.analysis_mode is AnalysisExecutionMode.ANOMALY_ONLY and (
        report.status is not AnalysisWorkflowStatus.REFUSED
    ):
        summary = _ANOMALY_ONLY_OVERVIEW
    elif (
        report.status is AnalysisWorkflowStatus.REFUSED
        and refusal_code in {"TARGET_CONSTANT", "TARGET_ALL_NULL"}
    ):
        summary = (
            "Supervised analysis was not run because the selected target has "
            "no variation."
        )
    else:
        summary = (
            f"Workflow terminated at stage {report.terminal_stage.value} with status "
            f"{report.status.value}. Recommendation status: {recommendation_text}."
        )
    return WorkflowOverviewView(
        status=report.status,
        terminal_stage=report.terminal_stage,
        headline=_HEADLINES[report.status],
        summary=summary,
        started_at=report.started_at,
        completed_at=report.completed_at,
        total_seconds=report.total_seconds,
        recommendation_status=recommendation_status,
    )


def _build_data_summary(report: AnalysisWorkflowReport) -> WorkflowDataSummaryView:
    row_identity_preserved = _metadata_bool(
        report.metadata,
        "row_identity_preserved",
        default=False,
    )
    return WorkflowDataSummaryView(
        raw_row_count=report.raw_row_count,
        processed_row_count=report.processed_row_count,
        cohort_row_count=report.cohort_row_count,
        train_row_count=report.train_row_count,
        validation_row_count=report.validation_row_count,
        test_row_count=report.test_row_count,
        anomaly_event_count=report.anomaly_event_count,
        diagnosis_factor_count=report.diagnosis_factor_count,
        selected_operating_row_id=report.selected_operating_row_id,
        row_identity_preserved=row_identity_preserved,
    )


def _format_cohort_range_display(
    *,
    configured: bool,
    lower_bound: float | None,
    upper_bound: float | None,
    include_lower: bool | None,
    include_upper: bool | None,
) -> str:
    if not configured:
        return "Not configured"
    if (
        lower_bound is None
        or upper_bound is None
        or include_lower is None
        or include_upper is None
    ):
        raise ValueError("configured cohort filter requires bounds and include flags")
    left_symbol = "≤" if include_lower else "<"
    right_symbol = "≤" if include_upper else "<"
    return f"{lower_bound} {left_symbol} x {right_symbol} {upper_bound}"


def _build_cohort_filter_summary(
    report: AnalysisWorkflowReport,
) -> WorkflowCohortFilterSummaryView:
    summary = report.cohort_filter_summary
    if summary.configured:
        exclude = summary.exclude_filter_column_from_features
        feature_display = "No" if exclude else "Yes"
        range_display = _format_cohort_range_display(
            configured=True,
            lower_bound=summary.lower_bound,
            upper_bound=summary.upper_bound,
            include_lower=summary.include_lower,
            include_upper=summary.include_upper,
        )
    else:
        feature_display = "Not applicable"
        range_display = "Not configured"
    return WorkflowCohortFilterSummaryView(
        configured=summary.configured,
        column_name=summary.column_name,
        lower_bound=summary.lower_bound,
        upper_bound=summary.upper_bound,
        include_lower=summary.include_lower,
        include_upper=summary.include_upper,
        exclude_filter_column_from_features=(
            summary.exclude_filter_column_from_features
        ),
        source_row_count=summary.source_row_count,
        retained_row_count=summary.retained_row_count,
        excluded_row_count=summary.excluded_row_count,
        null_excluded_count=summary.null_excluded_count,
        range_display=range_display,
        filter_column_used_as_feature_display=feature_display,
    )


def _build_routing_summary(
    report: AnalysisWorkflowReport,
) -> WorkflowRoutingSummaryView:
    target_column: str | None = None
    feature_count: int | None = None
    target_suitable: bool | None = None
    target_unique_non_null_count: int | None = None
    target_refusal_code: str | None = None
    target_suitability_message: str | None = None
    if "target_column" in report.metadata:
        raw_target = report.metadata["target_column"]
        if raw_target is None:
            target_column = None
        elif isinstance(raw_target, str) and raw_target != "" and raw_target.strip() != "":
            target_column = raw_target
        else:
            raise ValueError(
                "workflow metadata['target_column'] must be a non-empty str or None"
            )
    if "feature_count" in report.metadata:
        raw_count = report.metadata["feature_count"]
        if raw_count is None:
            feature_count = None
        elif isinstance(raw_count, int) and not isinstance(raw_count, bool) and raw_count >= 0:
            feature_count = raw_count
        else:
            raise ValueError(
                "workflow metadata['feature_count'] must be an int >= 0 or None"
            )
    if "target_suitable" in report.metadata:
        raw_suitable = report.metadata["target_suitable"]
        if raw_suitable is None:
            target_suitable = None
        elif type(raw_suitable) is bool:
            target_suitable = raw_suitable
        else:
            raise ValueError(
                "workflow metadata['target_suitable'] must be a bool or None"
            )
    if "target_unique_non_null_count" in report.metadata:
        raw_unique = report.metadata["target_unique_non_null_count"]
        if raw_unique is None:
            target_unique_non_null_count = None
        elif (
            isinstance(raw_unique, int)
            and not isinstance(raw_unique, bool)
            and raw_unique >= 0
        ):
            target_unique_non_null_count = raw_unique
        else:
            raise ValueError(
                "workflow metadata['target_unique_non_null_count'] must be "
                "an int >= 0 or None"
            )
    if "target_refusal_code" in report.metadata:
        raw_code = report.metadata["target_refusal_code"]
        if raw_code is None:
            target_refusal_code = None
        elif isinstance(raw_code, str) and raw_code.strip() != "":
            target_refusal_code = raw_code
        else:
            raise ValueError(
                "workflow metadata['target_refusal_code'] must be a non-empty "
                "str or None"
            )
    if "target_suitability_message" in report.metadata:
        raw_message = report.metadata["target_suitability_message"]
        if raw_message is None:
            target_suitability_message = None
        elif isinstance(raw_message, str) and raw_message.strip() != "":
            target_suitability_message = raw_message
        else:
            raise ValueError(
                "workflow metadata['target_suitability_message'] must be a "
                "non-empty str or None"
            )
    return WorkflowRoutingSummaryView(
        selected_industry=report.selected_industry,
        selected_task=report.selected_task,
        inferred_task=report.inferred_task,
        task_selection_source=(
            None
            if report.task_selection_source is None
            else report.task_selection_source.value
        ),
        task_override_applied=report.task_override_applied,
        target_column=target_column,
        feature_count=feature_count,
        target_suitable=target_suitable,
        target_unique_non_null_count=target_unique_non_null_count,
        target_refusal_code=target_refusal_code,
        target_suitability_message=target_suitability_message,
    )


def _build_model_summary(report: AnalysisWorkflowReport) -> WorkflowModelSummaryView:
    metadata = report.metadata
    return WorkflowModelSummaryView(
        supervised_model_key=report.selected_supervised_model_key,
        anomaly_model_key=report.selected_anomaly_model_key,
        independent_test_evaluation_performed=_metadata_bool(
            metadata,
            "independent_test_evaluation_performed",
            default=False,
        ),
        residual_calibration_performed=_metadata_bool(
            metadata,
            "residual_calibration_performed",
            default=False,
        ),
        test_used_for_model_selection=_metadata_bool(
            metadata,
            "test_used_for_model_selection",
            default=False,
        ),
        test_used_for_threshold_calibration=_metadata_bool(
            metadata,
            "test_used_for_threshold_calibration",
            default=False,
        ),
        anomaly_score_direction=_metadata_optional_str(
            metadata,
            "anomaly_score_direction",
        ),
    )


def _build_metric_view(result: MetricAcceptanceResult) -> PerformanceMetricView:
    return PerformanceMetricView(
        metric_name=result.metric_name,
        observed_value=result.observed_value,
        threshold=result.threshold,
        direction=result.direction,
        required=result.required,
        available=result.available,
        passed=result.passed,
        message=result.message,
    )


def _build_model_performance(
    assessment: ModelPerformanceAcceptanceReport,
) -> ModelPerformanceView:
    return ModelPerformanceView(
        status=assessment.status,
        evaluation_available=assessment.evaluation_available,
        independent_test_evaluation=assessment.independent_test_evaluation,
        test_row_count=assessment.test_row_count,
        metrics=[_build_metric_view(item) for item in assessment.metric_results],
        required_rule_count=assessment.required_rule_count,
        passed_required_rule_count=assessment.passed_required_rule_count,
        failed_required_rule_count=assessment.failed_required_rule_count,
        unavailable_required_rule_count=assessment.unavailable_required_rule_count,
        assessed_at=assessment.assessed_at,
        warnings=list(assessment.warnings),
    )


def _build_stage_view(
    record: AnalysisWorkflowStageRecord,
    *,
    sequence: int,
    include_stage_metadata: bool,
    include_stage_warnings: bool,
) -> WorkflowStageView:
    metadata: dict[str, ScalarMetadataValue] = {}
    if include_stage_metadata:
        metadata = _filter_stage_metadata(dict(record.metadata))
    warnings: list[str] = []
    if include_stage_warnings:
        warnings = list(record.warnings)
    return WorkflowStageView(
        sequence=sequence,
        stage=record.stage,
        executed=record.executed,
        succeeded=record.succeeded,
        structured_refusal=record.structured_refusal,
        status_label=stage_status_label(
            executed=record.executed,
            succeeded=record.succeeded,
            structured_refusal=record.structured_refusal,
        ),
        row_count=record.row_count,
        message=record.message,
        warnings=warnings,
        metadata=metadata,
    )


def _build_change_view(change: RecommendationChange) -> RecommendationChangeView:
    return RecommendationChangeView(
        variable=change.variable,
        current_value=change.current_value,
        proposed_value=change.proposed_value,
        delta=change.delta,
        relative_delta=change.relative_delta,
        rationale=change.rationale,
        confidence=change.confidence,
        requires_verification=change.requires_verification,
    )


def _build_recommendation_view(
    recommendation: RecommendationResult,
) -> RecommendationView:
    return RecommendationView(
        status=recommendation.status,
        objective=recommendation.objective,
        changes=[_build_change_view(change) for change in recommendation.changes],
        confidence=recommendation.confidence,
        baseline_prediction=recommendation.baseline_prediction,
        proposed_prediction=recommendation.proposed_prediction,
        baseline_anomaly_score=recommendation.baseline_anomaly_score,
        proposed_anomaly_score=recommendation.proposed_anomaly_score,
        extrapolation_flag=recommendation.extrapolation_flag,
        uncertainty_available=recommendation.uncertainty_available,
        generated_at=recommendation.generated_at,
        disclaimer=recommendation.disclaimer,
        warnings=list(recommendation.warnings),
        safety_status=recommendation.safety_decision.status,
        safety_messages=list(recommendation.safety_decision.messages),
        target_prediction_plausibility=(
            None
            if recommendation.target_prediction_plausibility is None
            else recommendation.target_prediction_plausibility.model_copy(deep=True)
        ),
    )


def _build_what_if_scenario_view(
    scenario: WhatIfVerificationScenario,
) -> WhatIfVerificationScenarioView:
    perturbed_value = None
    if scenario.perturbed_variable is not None:
        perturbed_value = scenario.variable_values.get(scenario.perturbed_variable)
    return WhatIfVerificationScenarioView(
        scenario_id=scenario.scenario_id,
        scenario_type=scenario.scenario_type,
        perturbed_variable=scenario.perturbed_variable,
        perturbation_direction=scenario.perturbation_direction,
        perturbed_value=perturbed_value,
        variable_values=dict(scenario.variable_values),
        predicted_quality=scenario.predicted_quality,
        anomaly_score=scenario.anomaly_score,
        objective_value=scenario.objective_value,
        improves_over_baseline=scenario.improves_over_baseline,
        improves_or_matches_proposed=scenario.improves_or_matches_proposed,
        extrapolated=scenario.extrapolated,
        warnings=list(scenario.warnings),
    )


def _build_recommendation_verification_view(
    verification: RecommendationWhatIfVerificationResult,
) -> RecommendationWhatIfVerificationView:
    return RecommendationWhatIfVerificationView(
        status=verification.status,
        objective=verification.objective,
        baseline_objective_value=verification.baseline_objective_value,
        proposed_objective_value=verification.proposed_objective_value,
        scenario_count=verification.scenario_count,
        neighbor_scenario_count=verification.neighbor_scenario_count,
        improving_neighbor_count=verification.improving_neighbor_count,
        non_improving_neighbor_count=verification.non_improving_neighbor_count,
        extrapolated_scenario_count=verification.extrapolated_scenario_count,
        stability_classification=verification.stability_classification,
        stability_message=stability_classification_message(
            verification.stability_classification
        ),
        scenarios=[
            _build_what_if_scenario_view(scenario)
            for scenario in verification.scenarios
        ],
        warnings=list(verification.warnings),
        rationale=verification.rationale,
    )


def _event_original_row_id(event: AnomalyEvent) -> int | str:
    sample_id = _json_safe_row_id(event.sample_id)
    if sample_id is not None:
        return sample_id
    anomaly_id = event.anomaly_id.strip()
    if anomaly_id == "":
        raise ValueError("anomaly event is missing a JSON-safe original row ID")
    return anomaly_id


def _event_selection_source(
    event: AnomalyEvent,
    *,
    report_selection_source: str | None,
) -> str:
    detector = event.detector.strip()
    if detector != "":
        return detector
    if report_selection_source is not None and report_selection_source.strip() != "":
        return report_selection_source
    return "selected_anomaly_event"


def _event_is_anomaly_flagged(event: AnomalyEvent) -> bool | None:
    detector = event.detector.strip()
    if detector == "top_anomaly_score_candidate":
        return False
    if detector in {
        "unsupervised_anomaly_model",
        "residual_anomaly_detector",
    }:
        return True
    return None


def _build_anomaly_event_views(
    report: AnalysisWorkflowReport,
) -> tuple[list[AnomalyEventView], list[str]]:
    score_direction = _metadata_optional_str(
        report.metadata,
        "anomaly_score_direction",
    )
    if score_direction is None:
        score_direction = _ANOMALY_SCORE_DIRECTION
    report_selection_source = _metadata_optional_str(
        report.metadata,
        "anomaly_event_selection_source",
    )
    operating_compare = _normalize_row_id_for_compare(report.selected_operating_row_id)
    views: list[AnomalyEventView] = []
    matched_operating = False
    for rank, event in enumerate(report.anomaly_events, start=1):
        if not isinstance(event, AnomalyEvent):
            raise ValueError(
                "workflow anomaly_events entries must be AnomalyEvent, "
                f"got {type(event).__name__}"
            )
        original_row_id = _event_original_row_id(event)
        is_operating = (
            operating_compare is not None
            and _normalize_row_id_for_compare(original_row_id) == operating_compare
        )
        if is_operating:
            matched_operating = True
        views.append(
            AnomalyEventView(
                rank=rank,
                original_row_id=original_row_id,
                anomaly_score=float(event.anomaly_score),
                is_operating_row=is_operating,
                selection_source=_event_selection_source(
                    event,
                    report_selection_source=report_selection_source,
                ),
                score_direction=score_direction,
                is_anomaly_flagged=_event_is_anomaly_flagged(event),
            )
        )
    warnings: list[str] = []
    if (
        report.selected_operating_row_id is not None
        and views
        and not matched_operating
    ):
        warnings.append(_OPERATING_ROW_MISSING_WARNING)
    return views, warnings


def _build_diagnosis_factor_views(
    report: AnalysisWorkflowReport,
) -> list[DiagnosisFactorView]:
    source = _metadata_optional_str(report.metadata, "diagnosis_source")
    views: list[DiagnosisFactorView] = []
    for rank, factor in enumerate(report.diagnosis_factors, start=1):
        if not isinstance(factor, RootCauseFactor):
            raise ValueError(
                "workflow diagnosis_factors entries must be RootCauseFactor, "
                f"got {type(factor).__name__}"
            )
        evidence_values = _extract_evidence_floats(factor.evidence)
        # diagnostic_score is the unified bounded ranking score for all rows.
        # Prefer ensemble RRF score when present; otherwise robust
        # raw_association_score (= confidence); finally factor.confidence.
        diagnostic_score = evidence_values.get("normalized_ensemble_score")
        if diagnostic_score is None:
            diagnostic_score = evidence_values.get("raw_association_score")
        if diagnostic_score is None:
            diagnostic_score = evidence_values.get("normalized_association_score")
        if diagnostic_score is None:
            diagnostic_score = float(factor.confidence)
        robust_z_score = evidence_values.get("robust_z_score")
        if robust_z_score is None:
            robust_z_score = evidence_values.get("group_deviation_z")
        # Explicit None marker in evidence wins over a parsed float fallback.
        if "robust_z_score=None" in factor.evidence:
            robust_z_score = None
        raw_group_difference = evidence_values.get("signed_location_difference")
        if raw_group_difference is None and factor.deviation is not None:
            raw_group_difference = float(factor.deviation)
        views.append(
            DiagnosisFactorView(
                rank=rank,
                feature_name=factor.variable,
                diagnostic_score=diagnostic_score,
                direction=factor.direction,
                anomaly_group_value=evidence_values.get("anomaly_median"),
                normal_group_value=evidence_values.get("reference_median"),
                raw_group_difference=raw_group_difference,
                robust_scale=evidence_values.get("robust_scale"),
                robust_z_score=robust_z_score,
                robust_scale_status=_extract_robust_scale_status(factor.evidence),
                effect_size=factor.deviation,
                confidence=float(factor.confidence),
                source=source,
            )
        )
    return views


def _build_anomaly_context_window_views(
    report: AnalysisWorkflowReport,
) -> list[AnomalyContextWindowView]:
    """Map workflow context windows into presentation views without recalculation."""
    views: list[AnomalyContextWindowView] = []
    for window in report.anomaly_context_windows:
        if not isinstance(window, AnomalyContextWindow):
            raise ValueError(
                "workflow anomaly_context_windows entries must be "
                f"AnomalyContextWindow, got {type(window).__name__}"
            )
        rows: list[AnomalyContextRowView] = []
        for row in window.rows:
            rows.append(
                AnomalyContextRowView(
                    analysis_position=row.analysis_position,
                    original_row_id=row.original_row_id,
                    relative_offset=row.relative_offset,
                    is_center_event=row.is_center_event,
                    is_selected_anomaly_event=row.is_selected_anomaly_event,
                    identifier_values=[
                        AnomalyContextIdentifierValueView(
                            column_name=item.column_name,
                            value=item.value,
                        )
                        for item in row.identifier_values
                    ],
                    timestamp_value=row.timestamp_value,
                    feature_values=[
                        AnomalyContextValueView(
                            feature_name=item.feature_name,
                            value=item.value,
                        )
                        for item in row.feature_values
                    ],
                )
            )
        views.append(
            AnomalyContextWindowView(
                event_rank=window.event_rank,
                center_original_row_id=window.center_original_row_id,
                center_anomaly_score=float(window.center_anomaly_score),
                radius=window.radius,
                order_basis=window.order_basis,
                feature_names=list(window.feature_names),
                rows=rows,
            )
        )
    return views


def _rationale_has_unsafe_wording(recommendation: RecommendationResult | None) -> bool:
    if recommendation is None:
        return False
    for change in recommendation.changes:
        for pattern in _UNSAFE_RATIONALE_PATTERNS:
            if pattern.search(change.rationale) is not None:
                return True
    return False


def _aggregate_warnings(
    report: AnalysisWorkflowReport,
    *,
    model_performance: ModelPerformanceView | None,
    stages: list[WorkflowStageView],
    recommendation: RecommendationView | None,
    presentation_warnings: list[str],
    include_stage_warnings: bool,
    maximum_warnings: int,
) -> list[str]:
    aggregated: list[str] = []
    aggregated.extend(report.warnings)
    aggregated.extend(presentation_warnings)
    if model_performance is not None:
        aggregated.extend(model_performance.warnings)
    if include_stage_warnings:
        for stage in stages:
            aggregated.extend(stage.warnings)
    if recommendation is not None:
        aggregated.extend(recommendation.safety_messages)
        aggregated.extend(recommendation.warnings)
    if _rationale_has_unsafe_wording(report.final_recommendation):
        aggregated.append(_UNSAFE_WORDING_WARNING)
    return _cap_warnings(aggregated, maximum_warnings=maximum_warnings)


def _aggregate_disclaimers(
    report: AnalysisWorkflowReport,
) -> list[str]:
    disclaimers: list[str] = []
    if report.final_recommendation is not None:
        disclaimers.append(report.final_recommendation.disclaimer)
    if report.analysis_mode is AnalysisExecutionMode.ANOMALY_ONLY:
        disclaimers.extend(
            [
                _DISCLAIMER_MODEL_BASED,
                _DISCLAIMER_NO_CAUSATION,
                _DISCLAIMER_STATISTICAL_ANOMALY,
                _DISCLAIMER_DOMAIN_INTERPRETATION,
                _DISCLAIMER_NO_GUARANTEE,
            ]
        )
    else:
        disclaimers.extend(
            [
                _DISCLAIMER_MODEL_BASED,
                _DISCLAIMER_NO_CAUSATION,
                _DISCLAIMER_VERIFICATION,
                _DISCLAIMER_NO_GUARANTEE,
                _DISCLAIMER_PERFORMANCE,
                _DISCLAIMER_EXTRAPOLATION,
            ]
        )
    return _dedupe_preserve_order(disclaimers)


class AnalysisWorkflowReportBuilder:
    """Convert ``AnalysisWorkflowReport`` into a UI/API-safe presentation DTO.

    Reads only public report fields and allowlisted scalar metadata. Does not
    mutate inputs, call estimators, recompute metrics, or generate recommendations.
    """

    def __init__(
        self,
        *,
        maximum_warnings: int = 200,
        include_stage_metadata: bool = True,
        include_stage_warnings: bool = True,
    ) -> None:
        """Create a presentation builder with isolated configuration.

        Args:
            maximum_warnings: Maximum aggregated presentation warnings (>= 1).
            include_stage_metadata: Whether filtered stage metadata is included.
            include_stage_warnings: Whether stage warning bodies are aggregated.

        Raises:
            ValueError: If configuration values fail strict validation.
        """
        self.maximum_warnings = _require_strict_int_ge1(
            maximum_warnings,
            field_name="maximum_warnings",
        )
        self.include_stage_metadata = _require_strict_bool(
            include_stage_metadata,
            field_name="include_stage_metadata",
        )
        self.include_stage_warnings = _require_strict_bool(
            include_stage_warnings,
            field_name="include_stage_warnings",
        )

    def build(
        self,
        workflow_report: AnalysisWorkflowReport,
    ) -> WorkflowPresentationOutcome:
        """Build a presentation DTO from one analysis workflow report.

        Args:
            workflow_report: Backend typed workflow report. Must be an
                ``AnalysisWorkflowReport`` instance.

        Returns:
            Immutable presentation outcome containing the validated DTO.

        Raises:
            TypeError: If ``workflow_report`` is not an ``AnalysisWorkflowReport``.
        """
        if not isinstance(workflow_report, AnalysisWorkflowReport):
            raise TypeError(
                "workflow_report must be AnalysisWorkflowReport, "
                f"got {type(workflow_report).__name__}"
            )

        maximum_warnings = self.maximum_warnings
        include_stage_metadata = self.include_stage_metadata
        include_stage_warnings = self.include_stage_warnings

        overview = _build_overview(workflow_report)
        data_summary = _build_data_summary(workflow_report)
        cohort_filter_summary = _build_cohort_filter_summary(workflow_report)
        routing_summary = _build_routing_summary(workflow_report)
        model_summary = _build_model_summary(workflow_report)

        model_performance: ModelPerformanceView | None = None
        if workflow_report.model_performance_assessment is not None:
            model_performance = _build_model_performance(
                workflow_report.model_performance_assessment
            )

        stages = [
            _build_stage_view(
                record,
                sequence=index,
                include_stage_metadata=include_stage_metadata,
                include_stage_warnings=include_stage_warnings,
            )
            for index, record in enumerate(workflow_report.stage_records, start=1)
        ]

        anomaly_events, event_warnings = _build_anomaly_event_views(workflow_report)
        diagnosis_factors = _build_diagnosis_factor_views(workflow_report)
        anomaly_context_windows = _build_anomaly_context_window_views(workflow_report)

        recommendation: RecommendationView | None = None
        if workflow_report.final_recommendation is not None:
            recommendation = _build_recommendation_view(
                workflow_report.final_recommendation
            )

        recommendation_verification: RecommendationWhatIfVerificationView | None = None
        if workflow_report.recommendation_verification is not None:
            recommendation_verification = _build_recommendation_verification_view(
                workflow_report.recommendation_verification
            )

        warnings = _aggregate_warnings(
            workflow_report,
            model_performance=model_performance,
            stages=stages,
            recommendation=recommendation,
            presentation_warnings=event_warnings,
            include_stage_warnings=include_stage_warnings,
            maximum_warnings=maximum_warnings,
        )
        disclaimers = _aggregate_disclaimers(workflow_report)
        metadata = _filter_presentation_metadata(dict(workflow_report.metadata))

        presentation = WorkflowPresentationReport(
            overview=overview,
            data_summary=data_summary,
            cohort_filter_summary=cohort_filter_summary,
            routing_summary=routing_summary,
            model_summary=model_summary,
            model_performance=model_performance,
            stages=stages,
            anomaly_events=anomaly_events,
            diagnosis_factors=diagnosis_factors,
            anomaly_context_windows=anomaly_context_windows,
            recommendation=recommendation,
            recommendation_verification=recommendation_verification,
            warnings=warnings,
            disclaimers=disclaimers,
            metadata=metadata,
            dataset_fingerprint=workflow_report.dataset_fingerprint,
            run_manifest=(
                None
                if workflow_report.run_manifest is None
                else workflow_report.run_manifest.model_copy(deep=True)
            ),
        )
        return WorkflowPresentationOutcome(report=presentation)

    def get_metadata(self) -> dict[str, Any]:
        """Return scalar builder capability metadata for the current instance."""
        return {
            "maximum_warnings": self.maximum_warnings,
            "include_stage_metadata": self.include_stage_metadata,
            "include_stage_warnings": self.include_stage_warnings,
            "produces_json_safe_dto": True,
            "performs_model_scoring": False,
            "performs_model_fit": False,
            "performs_model_refit": False,
            "performs_metric_recalculation": False,
            "performs_diagnosis": False,
            "performs_recommendation": False,
            "includes_raw_dataframe": False,
            "includes_model_object": False,
            "includes_estimator_object": False,
            "preserves_negative_anomaly_scores": True,
            "rewrites_backend_results": False,
        }
