"""Recommendation safety gate for eligibility and refusal decisions (Step 9A).

Evaluates leakage, model trust, diagnosis confidence, controllability,
constraints, verification, and extrapolation signals. Does not compute
proposed recommendation values or run optimization.
"""

from __future__ import annotations

from datetime import UTC, datetime

from process_intelligence.core.enums import ColumnRole
from process_intelligence.core.schemas import RootCauseFactor, VariableConstraint
from process_intelligence.evaluation.leakage import LeakageSeverity
from process_intelligence.recommendation.enums import (
    RecommendationReasonCode,
    RecommendationSafetyStatus,
)
from process_intelligence.recommendation.schemas import (
    DEFAULT_RECOMMENDATION_DISCLAIMER,
    RecommendationRequest,
    RecommendationSafetyContext,
    RecommendationSafetyDecision,
    RecommendationSafetyPolicy,
    ScalarMetadataValue,
    VariableEligibilityAssessment,
)

_HARD_GLOBAL_BLOCKERS = frozenset(
    {
        RecommendationReasonCode.LEAKAGE_BLOCKER,
        RecommendationReasonCode.FINAL_EVALUATION_MISSING,
        RecommendationReasonCode.MODEL_PERFORMANCE_UNACCEPTABLE,
        RecommendationReasonCode.DIAGNOSIS_CONFIDENCE_TOO_LOW,
        RecommendationReasonCode.REFERENCE_SAMPLE_TOO_SMALL,
        RecommendationReasonCode.EXTRAPOLATION_RISK,
        RecommendationReasonCode.UNCERTAINTY_UNAVAILABLE,
        RecommendationReasonCode.UNCERTAINTY_UNACCEPTABLE,
    }
)


class RecommendationSafetyGate:
    """Stateless safety evaluator for recommendation readiness.

    Holds an isolated deep copy of ``RecommendationSafetyPolicy``. Does not
    cache decisions, mutate caller-owned inputs, or generate proposed values.
    """

    def __init__(
        self,
        *,
        policy: RecommendationSafetyPolicy | None = None,
    ) -> None:
        """Create a gate with an isolated policy copy.

        Args:
            policy: Optional safety policy. When ``None``, defaults are used.
                The provided policy is deep-copied so later mutations do not
                affect this gate.

        Raises:
            TypeError: If ``policy`` is not ``None`` or a
                ``RecommendationSafetyPolicy``.
        """
        if policy is None:
            self._policy = RecommendationSafetyPolicy()
        elif not isinstance(policy, RecommendationSafetyPolicy):
            raise TypeError(
                "policy must be RecommendationSafetyPolicy, "
                f"got {type(policy).__name__}"
            )
        else:
            self._policy = policy.model_copy(deep=True)

    def evaluate(
        self,
        request: RecommendationRequest,
        *,
        context: RecommendationSafetyContext,
    ) -> RecommendationSafetyDecision:
        """Evaluate recommendation safety and variable eligibility.

        Expected safety failures return a structured ``REFUSED`` decision.
        Invalid types raise ``TypeError``. Inputs are not mutated.

        Args:
            request: Validated recommendation request.
            context: External safety signals (leakage, performance, etc.).

        Returns:
            A ``RecommendationSafetyDecision`` with eligible/blocked variables.
        """
        if not isinstance(request, RecommendationRequest):
            raise TypeError(
                f"request must be RecommendationRequest, got {type(request).__name__}"
            )
        if not isinstance(context, RecommendationSafetyContext):
            raise TypeError(
                "context must be RecommendationSafetyContext, "
                f"got {type(context).__name__}"
            )

        policy = self._policy
        global_reasons: list[RecommendationReasonCode] = []
        messages: list[str] = []
        has_leakage_warning = False
        has_constraint_warning = False
        user_controllability_override_count = 0
        user_verification_override_count = 0

        # --- global gates (deterministic order) ---
        if policy.require_safe_leakage_report and context.leakage_report.blocker_count > 0:
            global_reasons.append(RecommendationReasonCode.LEAKAGE_BLOCKER)
            messages.append(
                "Recommendation refused because the leakage report contains "
                f"{context.leakage_report.blocker_count} blocker issue(s)."
            )

        for issue in context.leakage_report.issues:
            if issue.severity is LeakageSeverity.WARNING:
                has_leakage_warning = True
                warning_message = (
                    f"Leakage warning ({issue.issue_type}): {issue.message}"
                )
                if warning_message not in messages:
                    messages.append(warning_message)

        if (
            policy.require_final_evaluation
            and not context.final_evaluation_available
        ):
            global_reasons.append(RecommendationReasonCode.FINAL_EVALUATION_MISSING)
            messages.append(
                "Recommendation refused because final evaluation is unavailable."
            )

        if (
            policy.require_acceptable_model_performance
            and not context.model_performance_acceptable
        ):
            global_reasons.append(
                RecommendationReasonCode.MODEL_PERFORMANCE_UNACCEPTABLE
            )
            if context.model_performance_reason is not None:
                messages.append(
                    "Recommendation refused because model performance is "
                    f"unacceptable: {context.model_performance_reason}"
                )
            else:
                messages.append(
                    "Recommendation refused because model performance is unacceptable."
                )

        if request.diagnosis.confidence < policy.minimum_diagnosis_confidence:
            global_reasons.append(
                RecommendationReasonCode.DIAGNOSIS_CONFIDENCE_TOO_LOW
            )
            messages.append(
                "Recommendation refused because diagnosis confidence "
                f"({request.diagnosis.confidence}) is below the minimum "
                f"({policy.minimum_diagnosis_confidence})."
            )

        if request.diagnosis.reference_row_count < policy.minimum_reference_rows:
            global_reasons.append(RecommendationReasonCode.REFERENCE_SAMPLE_TOO_SMALL)
            messages.append(
                "Recommendation refused because reference_row_count "
                f"({request.diagnosis.reference_row_count}) is below the minimum "
                f"({policy.minimum_reference_rows})."
            )

        if policy.block_on_extrapolation and context.extrapolation_detected:
            global_reasons.append(RecommendationReasonCode.EXTRAPOLATION_RISK)
            messages.append(
                "Recommendation refused because extrapolation risk was reported."
            )

        if policy.require_uncertainty and not context.uncertainty_available:
            global_reasons.append(RecommendationReasonCode.UNCERTAINTY_UNAVAILABLE)
            messages.append(
                "Recommendation refused because uncertainty estimates are unavailable."
            )

        if policy.require_acceptable_uncertainty and (
            not context.uncertainty_available
            or context.uncertainty_acceptable is not True
        ):
            if (
                RecommendationReasonCode.UNCERTAINTY_UNACCEPTABLE
                not in global_reasons
            ):
                global_reasons.append(
                    RecommendationReasonCode.UNCERTAINTY_UNACCEPTABLE
                )
            messages.append(
                "Recommendation refused because uncertainty is unavailable "
                "or not acceptable."
            )

        has_global_blocker = any(
            code in _HARD_GLOBAL_BLOCKERS for code in global_reasons
        )

        constraint_by_variable = {
            constraint.variable: constraint for constraint in request.constraints
        }
        confirmed = set(request.user_confirmed_controllable_variables)
        verified = set(request.user_verified_variables)

        assessments: list[VariableEligibilityAssessment] = []
        seen_factor_variables: set[str] = set()

        for rank, factor in enumerate(request.diagnosis.factors, start=1):
            if factor.variable in seen_factor_variables:
                # Defensive: DiagnosisResult already rejects duplicates.
                continue
            seen_factor_variables.add(factor.variable)

            assessment, flags = self._assess_variable(
                factor=factor,
                factor_rank=rank,
                request=request,
                policy=policy,
                constraint_by_variable=constraint_by_variable,
                confirmed=confirmed,
                verified=verified,
            )
            if flags["constraint_warning"]:
                has_constraint_warning = True
            if flags["controllability_override"]:
                user_controllability_override_count += 1
            if flags["verification_override"]:
                user_verification_override_count += 1
            assessments.append(assessment)

        # Apply candidate limit in diagnosis ranking order.
        limited_assessments: list[VariableEligibilityAssessment] = []
        eligible_kept = 0
        for assessment in assessments:
            if not assessment.eligible:
                limited_assessments.append(assessment)
                continue
            if eligible_kept < policy.maximum_candidate_variables:
                limited_assessments.append(assessment)
                eligible_kept += 1
                continue
            codes = list(assessment.reason_codes)
            if RecommendationReasonCode.CANDIDATE_LIMIT_EXCEEDED not in codes:
                codes.append(RecommendationReasonCode.CANDIDATE_LIMIT_EXCEEDED)
            limited_assessments.append(
                assessment.model_copy(
                    update={
                        "eligible": False,
                        "reason_codes": codes,
                    }
                )
            )
        assessments = limited_assessments

        eligible_variables = [
            item.variable for item in assessments if item.eligible
        ]
        blocked_variables = [
            item.variable for item in assessments if not item.eligible
        ]

        if has_constraint_warning:
            warning_message = (
                "One or more variables lack constraints; later optimizers must "
                "not treat unconstrained variables as change candidates."
            )
            if warning_message not in messages:
                messages.append(warning_message)

        for assessment in assessments:
            for warning in assessment.warnings:
                message = f"Variable {assessment.variable}: {warning}"
                if message not in messages:
                    messages.append(message)
            for code in assessment.reason_codes:
                message = (
                    f"Variable {assessment.variable} blocked: {code.value}."
                )
                if message not in messages:
                    messages.append(message)

        partial_eligibility = bool(eligible_variables) and bool(blocked_variables)

        if has_global_blocker:
            # Force all assessments ineligible while preserving reason codes.
            forced: list[VariableEligibilityAssessment] = []
            for assessment in assessments:
                forced.append(
                    assessment.model_copy(update={"eligible": False})
                )
            assessments = forced
            eligible_variables = []
            blocked_variables = [item.variable for item in assessments]
        elif partial_eligibility:
            if RecommendationReasonCode.PARTIAL_ELIGIBILITY not in global_reasons:
                global_reasons.append(RecommendationReasonCode.PARTIAL_ELIGIBILITY)
            messages.append(
                "Partial eligibility: some diagnosis factors are eligible while "
                "others remain blocked."
            )
            if not policy.allow_partial_eligibility:
                forced = [
                    assessment.model_copy(update={"eligible": False})
                    for assessment in assessments
                ]
                assessments = forced
                eligible_variables = []
                blocked_variables = [item.variable for item in assessments]
                if RecommendationReasonCode.NO_ELIGIBLE_VARIABLES not in global_reasons:
                    global_reasons.append(
                        RecommendationReasonCode.NO_ELIGIBLE_VARIABLES
                    )
                messages.append(
                    "Recommendation refused because partial eligibility is disabled."
                )
        elif not eligible_variables:
            if RecommendationReasonCode.NO_ELIGIBLE_VARIABLES not in global_reasons:
                global_reasons.append(RecommendationReasonCode.NO_ELIGIBLE_VARIABLES)
            messages.append(
                "Recommendation refused because no variables are eligible "
                "for recommendation."
            )

        has_hard_blocker_now = any(
            code in _HARD_GLOBAL_BLOCKERS
            or code is RecommendationReasonCode.NO_ELIGIBLE_VARIABLES
            for code in global_reasons
        )
        has_override = (
            user_controllability_override_count > 0
            or user_verification_override_count > 0
        )
        has_caution_signal = (
            bool(blocked_variables)
            or has_override
            or has_constraint_warning
            or has_leakage_warning
            or RecommendationReasonCode.PARTIAL_ELIGIBILITY in global_reasons
        )

        if has_hard_blocker_now or not eligible_variables:
            status = RecommendationSafetyStatus.REFUSED
        elif has_caution_signal:
            status = RecommendationSafetyStatus.CAUTION
        else:
            status = RecommendationSafetyStatus.APPROVED

        # Deduplicate messages while preserving order.
        unique_messages: list[str] = []
        seen_messages: set[str] = set()
        for message in messages:
            if message in seen_messages:
                continue
            seen_messages.add(message)
            unique_messages.append(message)

        metadata: dict[str, ScalarMetadataValue] = {
            "diagnosis_factor_count": len(request.diagnosis.factors),
            "eligible_variable_count": len(eligible_variables),
            "blocked_variable_count": len(blocked_variables),
            "global_blocker_count": sum(
                1
                for code in global_reasons
                if code in _HARD_GLOBAL_BLOCKERS
                or code is RecommendationReasonCode.NO_ELIGIBLE_VARIABLES
            ),
            "user_controllability_override_count": user_controllability_override_count,
            "user_verification_override_count": user_verification_override_count,
            "partial_eligibility": partial_eligibility
            and policy.allow_partial_eligibility
            and status is not RecommendationSafetyStatus.REFUSED,
            "recommendation_generated": False,
            "optimization_performed": False,
            "association_not_causation": True,
        }

        return RecommendationSafetyDecision(
            status=status,
            objective=request.objective,
            eligible_variables=list(eligible_variables),
            blocked_variables=list(blocked_variables),
            variable_assessments=list(assessments),
            global_reason_codes=list(global_reasons),
            messages=unique_messages,
            disclaimer=DEFAULT_RECOMMENDATION_DISCLAIMER,
            evaluated_at=datetime.now(tz=UTC),
            metadata=metadata,
        )

    def get_metadata(self) -> dict[str, ScalarMetadataValue]:
        """Return scalar-only gate metadata for the current policy.

        Returns an independent dict on every call. Does not include estimators,
        models, DataFrames, or ndarray values.
        """
        policy = self._policy
        return {
            "require_safe_leakage_report": policy.require_safe_leakage_report,
            "require_final_evaluation": policy.require_final_evaluation,
            "require_acceptable_model_performance": (
                policy.require_acceptable_model_performance
            ),
            "minimum_diagnosis_confidence": policy.minimum_diagnosis_confidence,
            "minimum_reference_rows": policy.minimum_reference_rows,
            "require_constraints": policy.require_constraints,
            "require_controllable_process_role": (
                policy.require_controllable_process_role
            ),
            "require_user_controllability_confirmation": (
                policy.require_user_controllability_confirmation
            ),
            "block_unverified_factors": policy.block_unverified_factors,
            "block_on_extrapolation": policy.block_on_extrapolation,
            "require_uncertainty": policy.require_uncertainty,
            "require_acceptable_uncertainty": policy.require_acceptable_uncertainty,
            "maximum_candidate_variables": policy.maximum_candidate_variables,
            "allow_partial_eligibility": policy.allow_partial_eligibility,
            "generates_recommendations": False,
            "performs_optimization": False,
            "association_not_causation": True,
        }

    def _assess_variable(
        self,
        *,
        factor: RootCauseFactor,
        factor_rank: int,
        request: RecommendationRequest,
        policy: RecommendationSafetyPolicy,
        constraint_by_variable: dict[str, VariableConstraint],
        confirmed: set[str],
        verified: set[str],
    ) -> tuple[VariableEligibilityAssessment, dict[str, bool]]:
        reason_codes: list[RecommendationReasonCode] = []
        warnings: list[str] = []
        flags = {
            "constraint_warning": False,
            "controllability_override": False,
            "verification_override": False,
        }

        variable = factor.variable
        current_value = request.current_values.get(variable)
        constraint = constraint_by_variable.get(variable)
        constraint_present = constraint is not None
        user_confirmed = variable in confirmed
        user_verified = variable in verified

        # 1. current value
        if current_value is None:
            reason_codes.append(RecommendationReasonCode.CURRENT_VALUE_MISSING)

        # 2. constraint
        if policy.require_constraints and not constraint_present:
            reason_codes.append(RecommendationReasonCode.CONSTRAINT_MISSING)
        elif not constraint_present:
            flags["constraint_warning"] = True
            warnings.append(
                "Constraint missing; variable should not be used as an optimizer "
                "change candidate without an explicit constraint."
            )

        # 3-4. controllability and user confirmation
        base_controllable = (
            factor.role is ColumnRole.CONTROLLABLE_PROCESS and factor.controllable
        )
        if user_confirmed and not base_controllable:
            flags["controllability_override"] = True
            warnings.append(
                "User confirmed controllability overrides uncertain role or "
                "controllable flag; treat as caution."
            )
            controllable_candidate = True
        else:
            controllable_candidate = base_controllable

        if policy.require_controllable_process_role and not controllable_candidate:
            reason_codes.append(RecommendationReasonCode.NON_CONTROLLABLE_VARIABLE)

        if (
            policy.require_user_controllability_confirmation
            and not user_confirmed
        ):
            reason_codes.append(RecommendationReasonCode.USER_CONFIRMATION_REQUIRED)

        # 5. verification
        if factor.needs_verification and policy.block_unverified_factors:
            if not user_verified:
                reason_codes.append(RecommendationReasonCode.VERIFICATION_REQUIRED)
            else:
                flags["verification_override"] = True
                warnings.append(
                    "Factor needs verification; user verification recorded with caution."
                )

        # 6. current value within constraint bounds
        if current_value is not None and constraint is not None:
            if not _current_value_within_constraint(current_value, constraint):
                reason_codes.append(
                    RecommendationReasonCode.CURRENT_VALUE_OUTSIDE_CONSTRAINT
                )

        blocking_present = any(
            code
            in {
                RecommendationReasonCode.CURRENT_VALUE_MISSING,
                RecommendationReasonCode.CONSTRAINT_MISSING,
                RecommendationReasonCode.NON_CONTROLLABLE_VARIABLE,
                RecommendationReasonCode.USER_CONFIRMATION_REQUIRED,
                RecommendationReasonCode.VERIFICATION_REQUIRED,
                RecommendationReasonCode.CURRENT_VALUE_OUTSIDE_CONSTRAINT,
                RecommendationReasonCode.CANDIDATE_LIMIT_EXCEEDED,
            }
            for code in reason_codes
        )
        eligible = not blocking_present

        assessment = VariableEligibilityAssessment(
            variable=variable,
            factor_rank=factor_rank,
            factor_confidence=factor.confidence,
            factor_role=str(factor.role),
            factor_controllable=factor.controllable,
            factor_needs_verification=factor.needs_verification,
            current_value=current_value,
            constraint_present=constraint_present,
            user_confirmed_controllable=user_confirmed,
            user_verified=user_verified,
            eligible=eligible,
            reason_codes=reason_codes,
            warnings=warnings,
        )
        return assessment, flags


def _current_value_within_constraint(
    current_value: float,
    constraint: VariableConstraint,
) -> bool:
    """Check numeric bounds using existing VariableConstraint fields only.

    Uses ``minimum`` / ``maximum`` when present. Does not invent allowed-values
    or step membership checks that are absent from the core contract.
    """
    if constraint.minimum is not None and current_value < constraint.minimum:
        return False
    if constraint.maximum is not None and current_value > constraint.maximum:
        return False
    return True
