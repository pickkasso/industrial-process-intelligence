"""Build UI/API-safe presentation DTOs from analysis workflow reports (Step 11A).

Maps public ``AnalysisWorkflowReport`` fields into JSON-serializable view models.
Does not mutate inputs, call models, recompute metrics, or generate recommendations.
"""

from __future__ import annotations

import re
from typing import Any

from process_intelligence.evaluation.performance_acceptance import (
    MetricAcceptanceResult,
    ModelPerformanceAcceptanceReport,
)
from process_intelligence.recommendation.schemas import (
    RecommendationChange,
    RecommendationResult,
)
from process_intelligence.reporting.schemas import (
    ModelPerformanceView,
    PerformanceMetricView,
    RecommendationChangeView,
    RecommendationView,
    WorkflowDataSummaryView,
    WorkflowModelSummaryView,
    WorkflowOverviewView,
    WorkflowPresentationOutcome,
    WorkflowPresentationReport,
    WorkflowRoutingSummaryView,
    WorkflowStageView,
    stage_status_label,
)
from process_intelligence.workflow.enums import AnalysisWorkflowStatus
from process_intelligence.workflow.schemas import (
    AnalysisWorkflowReport,
    AnalysisWorkflowStageRecord,
    ScalarMetadataValue,
)

_OMISSION_WARNING = (
    "Additional presentation warnings were omitted due to the configured limit."
)

_UNSAFE_WORDING_WARNING = (
    "Unsafe wording detected in recommendation rationale text."
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

_PRESENTATION_METADATA_ALLOWLIST: frozenset[str] = frozenset(
    {
        "raw_csv_loaded",
        "raw_row_count",
        "processed_row_count",
        "train_row_count",
        "validation_row_count",
        "test_row_count",
        "selected_industry",
        "selected_task",
        "selected_supervised_model_available",
        "selected_anomaly_model_available",
        "residual_calibration_performed",
        "independent_test_evaluation_performed",
        "anomaly_event_count",
        "diagnosis_performed",
        "recommendation_pipeline_executed",
        "recommendation_generated",
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
        train_row_count=report.train_row_count,
        validation_row_count=report.validation_row_count,
        test_row_count=report.test_row_count,
        anomaly_event_count=report.anomaly_event_count,
        diagnosis_factor_count=report.diagnosis_factor_count,
        selected_operating_row_id=report.selected_operating_row_id,
        row_identity_preserved=row_identity_preserved,
    )


def _build_routing_summary(
    report: AnalysisWorkflowReport,
) -> WorkflowRoutingSummaryView:
    target_column: str | None = None
    feature_count: int | None = None
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
    return WorkflowRoutingSummaryView(
        selected_industry=report.selected_industry,
        selected_task=report.selected_task,
        target_column=target_column,
        feature_count=feature_count,
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
    )


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
    include_stage_warnings: bool,
    maximum_warnings: int,
) -> list[str]:
    aggregated: list[str] = []
    aggregated.extend(report.warnings)
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

        recommendation: RecommendationView | None = None
        if workflow_report.final_recommendation is not None:
            recommendation = _build_recommendation_view(
                workflow_report.final_recommendation
            )

        warnings = _aggregate_warnings(
            workflow_report,
            model_performance=model_performance,
            stages=stages,
            recommendation=recommendation,
            include_stage_warnings=include_stage_warnings,
            maximum_warnings=maximum_warnings,
        )
        disclaimers = _aggregate_disclaimers(workflow_report)
        metadata = _filter_presentation_metadata(dict(workflow_report.metadata))

        presentation = WorkflowPresentationReport(
            overview=overview,
            data_summary=data_summary,
            routing_summary=routing_summary,
            model_summary=model_summary,
            model_performance=model_performance,
            stages=stages,
            recommendation=recommendation,
            warnings=warnings,
            disclaimers=disclaimers,
            metadata=metadata,
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
