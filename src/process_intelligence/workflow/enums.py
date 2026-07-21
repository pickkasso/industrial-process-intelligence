"""Workflow stage, status, and operating-point selection enumerations (Step 10C)."""

from enum import StrEnum


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
