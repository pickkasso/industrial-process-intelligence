"""Workflow stage, status, and operating-point selection enumerations (Step 10C)."""

from enum import StrEnum


class AnalysisExecutionMode(StrEnum):
    """How the analysis workflow executes modeling and diagnosis stages.

    SUPERVISED runs the target-based regression path with residual diagnosis and
    recommendation. ANOMALY_ONLY runs label-free anomaly detection and robust
    group comparison without a target or recommendation generation.
    """

    SUPERVISED = "SUPERVISED"
    ANOMALY_ONLY = "ANOMALY_ONLY"


class AnalysisWorkflowStage(StrEnum):
    """Canonical orchestration stages for the industrial analysis workflow.

    Stages follow the production raw-CSV analysis path from load through
    recommendation. The workflow invokes existing public components and does
    not reimplement modeling, diagnosis, or recommendation algorithms.
    """

    LOAD = "LOAD"
    PROFILE = "PROFILE"
    VALIDATE = "VALIDATE"
    QUALITY_SCORE = "QUALITY_SCORE"
    SORT = "SORT"
    COHORT_FILTER = "COHORT_FILTER"
    PREPROCESS = "PREPROCESS"
    INDUSTRY_ROUTING = "INDUSTRY_ROUTING"
    TASK_ROUTING = "TASK_ROUTING"
    ROLE_MAPPING = "ROLE_MAPPING"
    SPLIT = "SPLIT"
    LEAKAGE_CHECK = "LEAKAGE_CHECK"
    SUPERVISED_SCREENING = "SUPERVISED_SCREENING"
    SUPERVISED_FINAL_EVALUATION = "SUPERVISED_FINAL_EVALUATION"
    ANOMALY_SCREENING = "ANOMALY_SCREENING"
    ANOMALY_FINAL_EVALUATION = "ANOMALY_FINAL_EVALUATION"
    RESIDUAL_CALIBRATION = "RESIDUAL_CALIBRATION"
    RESIDUAL_FINAL_EVALUATION = "RESIDUAL_FINAL_EVALUATION"
    ANOMALY_EVENT_SELECTION = "ANOMALY_EVENT_SELECTION"
    DIAGNOSIS = "DIAGNOSIS"
    RECOMMENDATION = "RECOMMENDATION"


class AnalysisWorkflowStatus(StrEnum):
    """Terminal business status for a single analysis workflow run.

    COMPLETED means recommendation finished with a generated result.
    PARTIAL means analysis progressed but ended in a structured partial state
    such as READY_FOR_OPTIMIZATION. REFUSED means the run stopped safely due
    to validation, leakage, unsupported task, or safety refusal.
    """

    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    REFUSED = "REFUSED"


class OperatingPointSelectionMode(StrEnum):
    """How the workflow selects the operating point for recommendation.

    EXPLICIT_ROW_ID uses a caller-provided row identity. TOP_RESIDUAL_ANOMALY
    prefers residual anomaly indicators. TOP_UNSUPERVISED_ANOMALY prefers
    unsupervised indicators. LATEST_ROW uses the latest temporal or last test
    row when no anomaly preference applies.
    """

    EXPLICIT_ROW_ID = "EXPLICIT_ROW_ID"
    TOP_RESIDUAL_ANOMALY = "TOP_RESIDUAL_ANOMALY"
    TOP_UNSUPERVISED_ANOMALY = "TOP_UNSUPERVISED_ANOMALY"
    LATEST_ROW = "LATEST_ROW"


class TaskSelectionSource(StrEnum):
    """How the final analysis task was chosen for a workflow run.

    ROUTER means the task router inference was used unchanged (AUTO).
    USER_OVERRIDE means the caller explicitly selected REGRESSION or
    CLASSIFICATION and that selection was applied over router inference.
    """

    ROUTER = "ROUTER"
    USER_OVERRIDE = "USER_OVERRIDE"


class AnomalyContextOrderBasis(StrEnum):
    """Row adjacency basis used for anomaly context windows.

    LOADED_ROW_ORDER means neighboring rows follow the CSV load order when no
    timestamp sort was applied. SORTED_ANALYSIS_ORDER means neighboring rows
    follow the workflow chronological sort used for analysis. Original row IDs
    remain identity only and are not used as arithmetic neighbors.
    """

    LOADED_ROW_ORDER = "LOADED_ROW_ORDER"
    SORTED_ANALYSIS_ORDER = "SORTED_ANALYSIS_ORDER"
