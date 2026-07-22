"""Build ``AnalysisWorkflowRequest`` from UI submissions (Step 11B).

Converts validated ``WorkflowUiSubmission`` values into the public workflow
request contract. Does not infer constraints, controllability, metric
direction, or quality direction, and does not store uploaded files or
DataFrames.
"""

from __future__ import annotations

from pathlib import Path

from process_intelligence.core.schemas import VariableConstraint
from process_intelligence.evaluation.performance_acceptance import (
    MetricAcceptanceRule,
    ModelPerformanceAcceptancePolicy,
)
from process_intelligence.ui.schemas import (
    UiMetricRuleInput,
    UiVariableConstraintInput,
    WorkflowUiSubmission,
)
from process_intelligence.workflow.enums import AnalysisExecutionMode
from process_intelligence.workflow.schemas import (
    AnalysisWorkflowPolicy,
    AnalysisWorkflowRequest,
    ScalarMetadataValue,
)


class WorkflowUiRequestBuilder:
    """Convert a UI submission and CSV path into ``AnalysisWorkflowRequest``.

    The builder deep-copies its workflow policy and never mutates the caller's
    ``WorkflowUiSubmission``. It does not infer missing constraints or
    controllability confirmations.
    """

    def __init__(
        self,
        *,
        workflow_policy: AnalysisWorkflowPolicy | None = None,
    ) -> None:
        if workflow_policy is None:
            self._workflow_policy = AnalysisWorkflowPolicy()
        elif isinstance(workflow_policy, AnalysisWorkflowPolicy):
            self._workflow_policy = workflow_policy.model_copy(deep=True)
        else:
            raise TypeError(
                "workflow_policy must be AnalysisWorkflowPolicy or None, "
                f"got {type(workflow_policy).__name__}"
            )

    def build(
        self,
        *,
        csv_path: Path,
        submission: WorkflowUiSubmission,
    ) -> AnalysisWorkflowRequest:
        """Build a validated analysis workflow request.

        Args:
            csv_path: Existing regular ``.csv`` file path.
            submission: Validated UI submission DTO.

        Returns:
            A publicly validated ``AnalysisWorkflowRequest``.

        Raises:
            TypeError: If ``csv_path`` or ``submission`` have the wrong type.
            ValueError: If ``csv_path`` is missing, not a regular file, or not
                a ``.csv`` path.
        """
        if not isinstance(csv_path, Path):
            raise TypeError(
                f"csv_path must be Path, got {type(csv_path).__name__}"
            )
        if not isinstance(submission, WorkflowUiSubmission):
            raise TypeError(
                "submission must be WorkflowUiSubmission, "
                f"got {type(submission).__name__}"
            )

        if not csv_path.exists():
            raise FileNotFoundError(f"csv_path does not exist: {csv_path.name}")
        if not csv_path.is_file():
            raise ValueError("csv_path must be a regular file")
        if csv_path.suffix.lower() != ".csv":
            raise ValueError(
                f"csv_path must have a .csv extension, got {csv_path.suffix!r}"
            )

        submission_copy = submission.model_copy(deep=True)
        metadata = dict(submission_copy.metadata)
        metadata["ui_source"] = "streamlit_mvp"
        metadata["builds_analysis_workflow_request"] = True
        metadata["analysis_mode"] = submission_copy.analysis_mode.value

        if submission_copy.analysis_mode is AnalysisExecutionMode.ANOMALY_ONLY:
            return AnalysisWorkflowRequest(
                csv_path=csv_path,
                analysis_mode=AnalysisExecutionMode.ANOMALY_ONLY,
                target_column=None,
                feature_columns=list(submission_copy.feature_columns),
                model_performance_policy=None,
                timestamp_column=submission_copy.timestamp_column,
                identifier_columns=list(submission_copy.identifier_columns),
                excluded_columns=list(submission_copy.excluded_columns),
                column_role_overrides=dict(submission_copy.column_role_overrides),
                requested_task=None,
                objective=None,
                quality_direction=None,
                quality_target=None,
                request_constraints=[],
                industry_constraints=[],
                user_overrides=[],
                user_confirmed_controllable_variables=[],
                user_verified_variables=[],
                max_simultaneous_changes=submission_copy.max_simultaneous_changes,
                operating_point_selection=submission_copy.operating_point_selection,
                explicit_operating_row_id=submission_copy.explicit_operating_row_id,
                cohort_filter=(
                    None
                    if submission_copy.cohort_filter is None
                    else submission_copy.cohort_filter.model_copy(deep=True)
                ),
                metadata=metadata,
            )

        rules = [
            self._to_metric_rule(rule) for rule in submission_copy.performance_rules
        ]
        constraints = [
            self._to_variable_constraint(item) for item in submission_copy.constraints
        ]
        performance_policy = ModelPerformanceAcceptancePolicy(rules=rules)

        return AnalysisWorkflowRequest(
            csv_path=csv_path,
            analysis_mode=AnalysisExecutionMode.SUPERVISED,
            target_column=submission_copy.target_column,
            feature_columns=list(submission_copy.feature_columns),
            model_performance_policy=performance_policy,
            timestamp_column=submission_copy.timestamp_column,
            identifier_columns=list(submission_copy.identifier_columns),
            excluded_columns=list(submission_copy.excluded_columns),
            column_role_overrides=dict(submission_copy.column_role_overrides),
            requested_task=submission_copy.requested_task,
            objective=submission_copy.objective,
            quality_direction=submission_copy.quality_direction,
            quality_target=submission_copy.quality_target,
            request_constraints=constraints,
            industry_constraints=[],
            user_overrides=[],
            user_confirmed_controllable_variables=list(
                submission_copy.user_confirmed_controllable_variables
            ),
            user_verified_variables=list(submission_copy.user_verified_variables),
            max_simultaneous_changes=submission_copy.max_simultaneous_changes,
            operating_point_selection=submission_copy.operating_point_selection,
            explicit_operating_row_id=submission_copy.explicit_operating_row_id,
            cohort_filter=None,
            metadata=metadata,
        )

    def get_metadata(self) -> dict[str, ScalarMetadataValue]:
        """Return scalar capability metadata for this builder instance."""
        return {
            "builds_analysis_workflow_request": True,
            "infers_constraints": False,
            "infers_controllability": False,
            "infers_metric_direction": False,
            "stores_uploaded_file": False,
            "stores_raw_dataframe": False,
            "performs_modeling": False,
            "workflow_policy_stop_on_validation_blocker": (
                self._workflow_policy.stop_on_validation_blocker
            ),
        }

    @staticmethod
    def _to_metric_rule(rule: UiMetricRuleInput) -> MetricAcceptanceRule:
        return MetricAcceptanceRule(
            metric_name=rule.metric_name,
            direction=rule.direction,
            threshold=rule.threshold,
            required=rule.required,
        )

    @staticmethod
    def _to_variable_constraint(
        item: UiVariableConstraintInput,
    ) -> VariableConstraint:
        return VariableConstraint(
            variable=item.variable,
            adjustable=True,
            minimum=item.minimum,
            maximum=item.maximum,
            fixed=False,
        )
