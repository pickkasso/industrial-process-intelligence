"""Public API for workflow presentation reporting (Step 11A).

Exposes JSON-safe presentation DTOs and ``AnalysisWorkflowReportBuilder``.
Does not expose Streamlit, REST, CLI, persistence, or rendering helpers.
"""

from process_intelligence.reporting.builder import AnalysisWorkflowReportBuilder
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
    WorkflowCohortFilterSummaryView,
    WorkflowDataSummaryView,
    WorkflowModelSummaryView,
    WorkflowOverviewView,
    WorkflowPresentationOutcome,
    WorkflowPresentationReport,
    WorkflowRoutingSummaryView,
    WorkflowStageView,
)

__all__ = [
    "AnalysisWorkflowReportBuilder",
    "AnomalyContextIdentifierValueView",
    "AnomalyContextRowView",
    "AnomalyContextValueView",
    "AnomalyContextWindowView",
    "AnomalyEventView",
    "DiagnosisFactorView",
    "ModelPerformanceView",
    "PerformanceMetricView",
    "RecommendationChangeView",
    "RecommendationView",
    "WorkflowCohortFilterSummaryView",
    "WorkflowDataSummaryView",
    "WorkflowModelSummaryView",
    "WorkflowOverviewView",
    "WorkflowPresentationOutcome",
    "WorkflowPresentationReport",
    "WorkflowRoutingSummaryView",
    "WorkflowStageView",
]
