"""Public API for the industrial analysis workflow orchestrator (Step 10C).

Exposes the workflow enums, Pydantic contracts, and the concrete
``IndustrialProcessAnalysisWorkflow`` orchestrator. The workflow wires existing
public components from data, routing, industries, evaluation, models, diagnosis,
and recommendation packages. It does not reimplement any modeling, diagnosis, or
recommendation algorithm.
"""

from process_intelligence.workflow.cohort_filter import (
    NumericCohortFilterOutcome,
    apply_numeric_cohort_filter,
    list_numeric_cohort_filter_candidates,
    observed_numeric_range,
    preview_numeric_cohort_filter_row_count,
)
from process_intelligence.workflow.dataset_fingerprint import (
    compute_dataset_content_fingerprint,
    compute_dataset_content_fingerprint_from_bytes,
    is_valid_dataset_fingerprint,
    normalize_optional_dataset_fingerprint,
)
from process_intelligence.workflow.enums import (
    AnalysisExecutionMode,
    AnalysisWorkflowStage,
    AnalysisWorkflowStatus,
    AnomalyContextOrderBasis,
    OperatingPointSelectionMode,
    TaskSelectionSource,
)
from process_intelligence.workflow.pipeline import (
    IndustrialProcessAnalysisWorkflow,
)
from process_intelligence.workflow.schemas import (
    ANOMALY_CONTEXT_MAX_FEATURES,
    ANOMALY_CONTEXT_RADIUS,
    AnalysisWorkflowOutcome,
    AnalysisWorkflowPolicy,
    AnalysisWorkflowReport,
    AnalysisWorkflowRequest,
    AnalysisWorkflowStageRecord,
    AnomalyContextIdentifierValue,
    AnomalyContextRow,
    AnomalyContextValue,
    AnomalyContextWindow,
    AnomalyRecommendationConfig,
    CohortFilterSummary,
    NumericCohortFilter,
)

__all__ = [
    "ANOMALY_CONTEXT_MAX_FEATURES",
    "ANOMALY_CONTEXT_RADIUS",
    "AnalysisExecutionMode",
    "AnalysisWorkflowOutcome",
    "AnalysisWorkflowPolicy",
    "AnalysisWorkflowReport",
    "AnalysisWorkflowRequest",
    "AnalysisWorkflowStage",
    "AnalysisWorkflowStageRecord",
    "AnalysisWorkflowStatus",
    "AnomalyContextIdentifierValue",
    "AnomalyContextOrderBasis",
    "AnomalyContextRow",
    "AnomalyContextValue",
    "AnomalyContextWindow",
    "AnomalyRecommendationConfig",
    "CohortFilterSummary",
    "IndustrialProcessAnalysisWorkflow",
    "NumericCohortFilter",
    "NumericCohortFilterOutcome",
    "OperatingPointSelectionMode",
    "TaskSelectionSource",
    "apply_numeric_cohort_filter",
    "compute_dataset_content_fingerprint",
    "compute_dataset_content_fingerprint_from_bytes",
    "is_valid_dataset_fingerprint",
    "list_numeric_cohort_filter_candidates",
    "normalize_optional_dataset_fingerprint",
    "observed_numeric_range",
    "preview_numeric_cohort_filter_row_count",
]
