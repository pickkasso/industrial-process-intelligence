"""End-to-end acceptance validation for the synthetic manufacturing demo dataset.

Runs the public ``IndustrialProcessAnalysisWorkflow`` against the deterministic
demo generator and checks supervised regression quality, anomaly-event overlap
with injected ground truth, residual-anomaly overlap when present, and explicit
feature-leakage guards. Result DTOs are serializable scalars only.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl

from process_intelligence.core.enums import AnalysisTask, ColumnRole
from process_intelligence.core.exceptions import DataLeakageError, DataValidationError
from process_intelligence.core.schemas import VariableConstraint
from process_intelligence.data import DatasetLoader, DatasetSorter
from process_intelligence.demo_data import (
    DEMO_CONTROLLABLE_COLUMNS,
    DEMO_IDENTITY_COLUMNS,
    DEMO_MEASURED_COLUMNS,
    DEMO_QUALITY_COLUMNS,
    DEMO_QUALITY_SCORE_DECLARED_MAXIMUM,
    DEMO_QUALITY_SCORE_DECLARED_MINIMUM,
    GROUND_TRUTH_METADATA_COLUMNS,
    DemoDatasetConfiguration,
    demo_setpoint_bound_map,
    generate_demo_dataset,
    summarize_demo_dataset,
    write_demo_workflow_csv,
)
from process_intelligence.demo_evaluation import (
    injected_anomaly_row_ids,
    overlap_precision_and_enrichment,
)
from process_intelligence.evaluation import (
    DatasetSplitter,
    MetricAcceptanceDirection,
    MetricAcceptanceRule,
    ModelPerformanceAcceptancePolicy,
    SplitConfig,
    SplitStrategy,
    evaluate_regression,
)
from process_intelligence.recommendation import (
    QualityOptimizationDirection,
    RecommendationObjective,
    RecommendationStatus,
    TargetPredictionPlausibilityStatus,
)
from process_intelligence.workflow import (
    AnalysisExecutionMode,
    AnalysisWorkflowOutcome,
    AnalysisWorkflowPolicy,
    AnalysisWorkflowReport,
    AnalysisWorkflowRequest,
    AnalysisWorkflowStage,
    AnalysisWorkflowStatus,
    IndustrialProcessAnalysisWorkflow,
    OperatingPointSelectionMode,
)

TARGET_COLUMN: Final[str] = "quality_score"
TIMESTAMP_COLUMN: Final[str] = "timestamp"
IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = tuple(
    name for name in DEMO_IDENTITY_COLUMNS if name != TIMESTAMP_COLUMN
)

# Default-demo anomaly-only enrichment is ~6.7 and residual enrichment is ~33
# with seed=42 / 1500 rows / max_anomaly_events=5. Require enrichment clearly
# above random selection (1.0) with a stable margin that tolerates minor
# detector ranking jitter without asserting exact model names or floats.
MIN_ANOMALY_ENRICHMENT_FACTOR: Final[float] = 2.0
_RESIDUAL_DETECTOR: Final[str] = "residual_anomaly_detector"

# Quality outputs other than the supervised target must not predict that target.
_SIBLING_QUALITY_COLUMNS: Final[tuple[str, ...]] = tuple(
    name for name in DEMO_QUALITY_COLUMNS if name != TARGET_COLUMN
)

# Identity/order columns may drive TIME sorting/splitting but are not model features.
_FORBIDDEN_IDENTITY_FEATURES: Final[tuple[str, ...]] = DEMO_IDENTITY_COLUMNS

_FORBIDDEN_SUPERVISED_FEATURES: Final[tuple[str, ...]] = (
    *_FORBIDDEN_IDENTITY_FEATURES,
    *GROUND_TRUTH_METADATA_COLUMNS,
    TARGET_COLUMN,
    *_SIBLING_QUALITY_COLUMNS,
)
_FORBIDDEN_ANOMALY_FEATURES: Final[tuple[str, ...]] = (
    *_FORBIDDEN_IDENTITY_FEATURES,
    *GROUND_TRUTH_METADATA_COLUMNS,
    *DEMO_QUALITY_COLUMNS,
)


@dataclass(frozen=True, slots=True)
class DemoSupervisedValidationResult:
    """Serializable acceptance outcome for the supervised demo workflow."""

    row_count: int
    anomaly_row_count: int
    workflow_status: str
    selected_model: str | None
    selected_task: str | None
    regression_metrics: dict[str, float]
    naive_mean_baseline_metrics: dict[str, float]
    beats_mean_baseline: bool
    feature_count: int
    feature_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]
    anomaly_event_count: int
    residual_anomaly_event_count: int
    residual_matched_injected_rows: int
    residual_overlap_precision: float | None
    residual_baseline_anomaly_rate: float
    residual_enrichment_factor: float | None
    residual_available: bool
    leakage_free: bool
    passed: bool
    messages: tuple[str, ...] = field(default_factory=tuple)
    recommendation_status: str | None = None
    recommendation_change_count: int = 0
    recommendation_variables_within_bounds: bool | None = None
    baseline_raw_prediction: float | None = None
    proposed_raw_prediction: float | None = None
    proposed_prediction_plausibility: str | None = None
    declared_domain_extrapolation_detected: bool = False
    recommendation_safety_handled: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly dictionary representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DemoAnomalyValidationResult:
    """Serializable acceptance outcome for the anomaly-only demo workflow."""

    row_count: int
    anomaly_row_count: int
    workflow_status: str
    selected_model: str | None
    anomaly_event_count: int
    detected_row_count: int
    matched_injected_anomaly_rows: int
    overlap_precision: float | None
    random_baseline_anomaly_rate: float
    enrichment_factor: float | None
    feature_count: int
    feature_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]
    leakage_free: bool
    passed: bool
    messages: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly dictionary representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DemoValidationReport:
    """Combined supervised + anomaly-only demo acceptance report."""

    configuration: DemoDatasetConfiguration
    supervised: DemoSupervisedValidationResult
    anomaly_only: DemoAnomalyValidationResult
    enrichment_threshold: float
    passed: bool
    messages: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly dictionary representation."""
        return {
            "configuration": {
                "row_count": self.configuration.row_count,
                "random_seed": self.configuration.random_seed,
                "anomaly_fraction": self.configuration.anomaly_fraction,
                "sampling_interval_minutes": (
                    self.configuration.sampling_interval_minutes
                ),
            },
            "supervised": self.supervised.to_dict(),
            "anomaly_only": self.anomaly_only.to_dict(),
            "enrichment_threshold": self.enrichment_threshold,
            "passed": self.passed,
            "messages": list(self.messages),
        }


def default_demo_feature_columns() -> list[str]:
    """Return the canonical process feature list used by demo validation.

    Includes controllable setpoints and measured process variables only.
    Quality outputs, ground-truth metadata, and identity/order columns are
    intentionally excluded from model features.
    """
    return [*DEMO_CONTROLLABLE_COLUMNS, *DEMO_MEASURED_COLUMNS]


def default_demo_excluded_columns(
    *,
    analysis_mode: AnalysisExecutionMode,
) -> list[str]:
    """Return columns excluded from modeling for the given demo analysis mode.

    Timestamp remains available for TIME sorting/splitting via
    ``timestamp_column`` and is therefore not listed here. Identifier columns
    are declared separately via ``identifier_columns``.
    """
    if analysis_mode is AnalysisExecutionMode.SUPERVISED:
        return [*GROUND_TRUTH_METADATA_COLUMNS, *_SIBLING_QUALITY_COLUMNS]
    if analysis_mode is AnalysisExecutionMode.ANOMALY_ONLY:
        return [*GROUND_TRUTH_METADATA_COLUMNS, *DEMO_QUALITY_COLUMNS]
    raise DataValidationError(f"unsupported analysis_mode: {analysis_mode!r}")


def assert_no_demo_feature_leakage(
    feature_columns: Sequence[str],
    *,
    analysis_mode: AnalysisExecutionMode,
    target_column: str = TARGET_COLUMN,
) -> None:
    """Raise ``DataLeakageError`` when forbidden demo columns appear as features.

    Supervised mode (target ``quality_score``) forbids:
    - ground-truth metadata
    - the supervised target
    - sibling quality outputs such as ``defect_rate``
    - identity/order columns

    Anomaly-only mode forbids:
    - ground-truth metadata
    - all quality outputs
    - identity/order columns
    """
    if not isinstance(feature_columns, Sequence) or isinstance(
        feature_columns, (str, bytes)
    ):
        raise TypeError(
            "feature_columns must be a sequence of str, "
            f"got {type(feature_columns).__name__}"
        )
    if not isinstance(target_column, str) or not target_column.strip():
        raise DataValidationError(
            "target_column must be a non-empty str when checking supervised leakage"
        )
    cleaned = [str(name) for name in feature_columns]
    if analysis_mode is AnalysisExecutionMode.SUPERVISED:
        forbidden: tuple[str, ...] = (
            *_FORBIDDEN_IDENTITY_FEATURES,
            *GROUND_TRUTH_METADATA_COLUMNS,
            target_column,
            *(
                name
                for name in DEMO_QUALITY_COLUMNS
                if name != target_column
            ),
        )
    elif analysis_mode is AnalysisExecutionMode.ANOMALY_ONLY:
        forbidden = _FORBIDDEN_ANOMALY_FEATURES
    else:  # pragma: no cover - defensive enum guard
        raise DataValidationError(f"unsupported analysis_mode: {analysis_mode!r}")

    leaked = [name for name in cleaned if name in forbidden]
    if leaked:
        raise DataLeakageError(
            "Demo feature list contains forbidden identity, quality-output, "
            f"ground-truth, or target leakage columns: {leaked}"
        )


@dataclass(frozen=True, slots=True)
class _SupervisedRecommendationAssessment:
    handled: bool
    status: str | None
    change_count: int
    variables_within_bounds: bool | None
    baseline_raw: float | None
    proposed_raw: float | None
    proposed_plausibility: str | None
    declared_extrapolation: bool
    messages: tuple[str, ...]


def _assess_supervised_recommendation(
    report: AnalysisWorkflowReport,
) -> _SupervisedRecommendationAssessment:
    """Assess recommendation/plausibility handling for supervised demo acceptance.

    FINAL PASS may mean either a generated recommendation with declared-domain
    extrapolation clearly detected, or a genuine safety refusal (for example
    insufficient diagnosis confidence). Finite predictions alone are not enough.
    """
    messages: list[str] = []
    recommendation = report.final_recommendation
    if recommendation is None:
        messages.append("Supervised recommendation result was missing.")
        return _SupervisedRecommendationAssessment(
            handled=False,
            status=None,
            change_count=0,
            variables_within_bounds=None,
            baseline_raw=None,
            proposed_raw=None,
            proposed_plausibility=None,
            declared_extrapolation=False,
            messages=tuple(messages),
        )

    status = recommendation.status.value
    change_count = len(recommendation.changes)
    baseline_raw = recommendation.baseline_prediction
    proposed_raw = recommendation.proposed_prediction
    plausibility = recommendation.target_prediction_plausibility
    proposed_status = (
        None
        if plausibility is None or plausibility.proposed_status is None
        else plausibility.proposed_status.value
    )
    declared_extrapolation = (
        plausibility is not None
        and plausibility.proposed_status
        is TargetPredictionPlausibilityStatus.OUTSIDE_DECLARED_DOMAIN
    )
    bounds_map = demo_setpoint_bound_map()
    variables_within_bounds: bool | None = None

    if recommendation.status is RecommendationStatus.GENERATED:
        if change_count < 1:
            messages.append("GENERATED recommendation had no changes.")
            handled = False
        else:
            within = True
            for change in recommendation.changes:
                if change.variable not in bounds_map:
                    within = False
                    messages.append(
                        "Recommendation changed a non-approved controllable "
                        f"variable: {change.variable!r}."
                    )
                    continue
                low, high = bounds_map[change.variable]
                if not (
                    low <= float(change.proposed_value) <= high
                    and low <= float(change.current_value) <= high
                ):
                    within = False
                    messages.append(
                        "Recommendation variable "
                        f"{change.variable!r} left approved engineering bounds."
                    )
            variables_within_bounds = within
            if plausibility is None or proposed_status is None:
                messages.append(
                    "GENERATED recommendation missing target-prediction "
                    "plausibility status."
                )
                handled = False
            elif declared_extrapolation:
                messages.append(
                    "Declared-domain extrapolation detected for the raw proposed "
                    "quality prediction; estimated improvement must not be "
                    "interpreted quantitatively as an attainable quality score."
                )
                handled = within
            elif (
                proposed_raw is not None
                and math.isfinite(float(proposed_raw))
                and (
                    float(proposed_raw) < DEMO_QUALITY_SCORE_DECLARED_MINIMUM
                    or float(proposed_raw) > DEMO_QUALITY_SCORE_DECLARED_MAXIMUM
                )
            ):
                messages.append(
                    "Proposed prediction is outside the declared demo quality "
                    "domain but plausibility status did not flag "
                    "OUTSIDE_DECLARED_DOMAIN."
                )
                handled = False
            else:
                messages.append(
                    "Recommendation generated with plausibility status present; "
                    "proposed prediction remained inside the declared domain."
                )
                handled = within
    elif recommendation.status is RecommendationStatus.REFUSED:
        messages.append(
            "Recommendation was refused by the safety gate "
            f"(status={status}). Genuine refusal after successful modeling is "
            "accepted when confidence or other safety gates block generation."
        )
        handled = True
    elif recommendation.status is RecommendationStatus.READY_FOR_OPTIMIZATION:
        messages.append(
            "Recommendation remained READY_FOR_OPTIMIZATION without generated "
            "changes; treated as handled safety outcome for demo acceptance."
        )
        handled = True
    else:
        messages.append(f"Unexpected recommendation status: {status}.")
        handled = False

    return _SupervisedRecommendationAssessment(
        handled=handled,
        status=status,
        change_count=change_count,
        variables_within_bounds=variables_within_bounds,
        baseline_raw=baseline_raw,
        proposed_raw=proposed_raw,
        proposed_plausibility=proposed_status,
        declared_extrapolation=declared_extrapolation,
        messages=tuple(messages),
    )


def _feature_policy_is_clean(
    feature_columns: Sequence[str],
    *,
    analysis_mode: AnalysisExecutionMode,
) -> bool:
    """Return True when the feature list obeys the demo leakage policy."""
    forbidden = (
        _FORBIDDEN_SUPERVISED_FEATURES
        if analysis_mode is AnalysisExecutionMode.SUPERVISED
        else _FORBIDDEN_ANOMALY_FEATURES
    )
    return not any(name in feature_columns for name in forbidden)


def validate_demo_supervised(
    configuration: DemoDatasetConfiguration | None = None,
    *,
    enrichment_threshold: float = MIN_ANOMALY_ENRICHMENT_FACTOR,
) -> DemoSupervisedValidationResult:
    """Run supervised demo acceptance against the public workflow API."""
    config = configuration or DemoDatasetConfiguration()
    threshold = _validate_enrichment_threshold(enrichment_threshold)
    frame = generate_demo_dataset(config)
    summary = summarize_demo_dataset(frame)
    row_count = _summary_int(summary, "row_count")
    anomaly_row_count = _summary_int(summary, "anomaly_row_count")
    feature_columns = default_demo_feature_columns()
    excluded_columns = default_demo_excluded_columns(
        analysis_mode=AnalysisExecutionMode.SUPERVISED,
    )
    messages: list[str] = []

    try:
        assert_no_demo_feature_leakage(
            feature_columns,
            analysis_mode=AnalysisExecutionMode.SUPERVISED,
            target_column=TARGET_COLUMN,
        )
        leakage_free = True
    except DataLeakageError as exc:
        return DemoSupervisedValidationResult(
            row_count=row_count,
            anomaly_row_count=anomaly_row_count,
            workflow_status="NOT_RUN",
            selected_model=None,
            selected_task=None,
            regression_metrics={},
            naive_mean_baseline_metrics={},
            beats_mean_baseline=False,
            feature_count=len(feature_columns),
            feature_columns=tuple(feature_columns),
            excluded_columns=tuple(excluded_columns),
            anomaly_event_count=0,
            residual_anomaly_event_count=0,
            residual_matched_injected_rows=0,
            residual_overlap_precision=None,
            residual_baseline_anomaly_rate=float(
                anomaly_row_count / max(row_count, 1)
            ),
            residual_enrichment_factor=None,
            residual_available=False,
            leakage_free=False,
            passed=False,
            messages=(str(exc),),
        )

    with tempfile.TemporaryDirectory(prefix="demo_supervised_") as temp_dir:
        csv_path = write_demo_workflow_csv(frame, Path(temp_dir) / "demo.csv")
        outcome = _run_supervised_workflow(csv_path=csv_path, feature_columns=feature_columns)
        report = outcome.report
        baseline_metrics = _naive_mean_baseline_metrics(
            csv_path=csv_path,
            target_column=TARGET_COLUMN,
        )
        regression_metrics = _extract_regression_metrics(report)
        beats_baseline = _beats_mean_baseline(
            regression_metrics=regression_metrics,
            baseline_metrics=baseline_metrics,
        )
        ground_truth_ids = _injected_anomaly_row_ids(frame)
        baseline_rate = float(anomaly_row_count / max(row_count, 1))
        residual_events = [
            event
            for event in report.anomaly_events
            if event.detector == _RESIDUAL_DETECTOR
        ]
        residual_ids = _event_row_ids(residual_events)
        residual_matched = sorted(set(residual_ids) & ground_truth_ids)
        residual_available = len(residual_ids) > 0
        residual_precision, residual_enrichment = _overlap_stats(
            detected_ids=residual_ids,
            ground_truth_ids=ground_truth_ids,
            baseline_rate=baseline_rate,
        )

        modeling_ok = _supervised_modeling_succeeded(report)
        metrics_finite = _metrics_are_finite(regression_metrics)
        residual_ok = True
        if residual_available:
            residual_ok = (
                len(residual_matched) >= 1
                and residual_enrichment is not None
                and residual_enrichment >= threshold
            )
        feature_policy_ok = _feature_policy_is_clean(
            feature_columns,
            analysis_mode=AnalysisExecutionMode.SUPERVISED,
        )
        recommendation_assessment = _assess_supervised_recommendation(report)
        recommendation_ok = recommendation_assessment.handled

        passed = (
            leakage_free
            and modeling_ok
            and metrics_finite
            and beats_baseline
            and residual_ok
            and feature_policy_ok
            and recommendation_ok
        )

        if not modeling_ok:
            messages.append(
                "Supervised workflow did not complete modeling/diagnosis stages "
                f"(status={report.status.value}, terminal={report.terminal_stage.value})."
            )
        if not metrics_finite:
            messages.append("Primary regression metrics are missing or non-finite.")
        if not beats_baseline:
            messages.append(
                "Selected model did not beat the naive mean-target baseline on RMSE."
            )
        if residual_available and not residual_ok:
            messages.append(
                "Residual anomaly events did not show meaningful overlap with "
                f"injected anomalies (enrichment={residual_enrichment!r}, "
                f"threshold={threshold})."
            )
        elif not residual_available:
            messages.append(
                "No residual anomaly events were exposed; residual overlap was skipped."
            )
        if not feature_policy_ok:
            messages.append(
                "Supervised feature list retained forbidden identity, quality-output, "
                "or ground-truth columns."
            )
        messages.extend(recommendation_assessment.messages)
        if passed:
            messages.append("Supervised demo acceptance criteria passed.")

        return DemoSupervisedValidationResult(
            row_count=row_count,
            anomaly_row_count=anomaly_row_count,
            workflow_status=report.status.value,
            selected_model=report.selected_supervised_model_key,
            selected_task=(
                report.selected_task.value if report.selected_task is not None else None
            ),
            regression_metrics=regression_metrics,
            naive_mean_baseline_metrics=baseline_metrics,
            beats_mean_baseline=beats_baseline,
            feature_count=len(feature_columns),
            feature_columns=tuple(feature_columns),
            excluded_columns=tuple(excluded_columns),
            anomaly_event_count=int(report.anomaly_event_count),
            residual_anomaly_event_count=len(residual_ids),
            residual_matched_injected_rows=len(residual_matched),
            residual_overlap_precision=residual_precision,
            residual_baseline_anomaly_rate=baseline_rate,
            residual_enrichment_factor=residual_enrichment,
            residual_available=residual_available,
            leakage_free=leakage_free,
            passed=passed,
            messages=tuple(messages),
            recommendation_status=recommendation_assessment.status,
            recommendation_change_count=recommendation_assessment.change_count,
            recommendation_variables_within_bounds=(
                recommendation_assessment.variables_within_bounds
            ),
            baseline_raw_prediction=recommendation_assessment.baseline_raw,
            proposed_raw_prediction=recommendation_assessment.proposed_raw,
            proposed_prediction_plausibility=(
                recommendation_assessment.proposed_plausibility
            ),
            declared_domain_extrapolation_detected=(
                recommendation_assessment.declared_extrapolation
            ),
            recommendation_safety_handled=recommendation_ok,
        )


def validate_demo_anomaly_only(
    configuration: DemoDatasetConfiguration | None = None,
    *,
    enrichment_threshold: float = MIN_ANOMALY_ENRICHMENT_FACTOR,
) -> DemoAnomalyValidationResult:
    """Run anomaly-only demo acceptance against the public workflow API."""
    config = configuration or DemoDatasetConfiguration()
    threshold = _validate_enrichment_threshold(enrichment_threshold)
    frame = generate_demo_dataset(config)
    summary = summarize_demo_dataset(frame)
    row_count = _summary_int(summary, "row_count")
    anomaly_row_count = _summary_int(summary, "anomaly_row_count")
    feature_columns = default_demo_feature_columns()
    excluded_columns = default_demo_excluded_columns(
        analysis_mode=AnalysisExecutionMode.ANOMALY_ONLY,
    )
    messages: list[str] = []
    baseline_rate = float(anomaly_row_count / max(row_count, 1))

    try:
        assert_no_demo_feature_leakage(
            feature_columns,
            analysis_mode=AnalysisExecutionMode.ANOMALY_ONLY,
        )
        leakage_free = True
    except DataLeakageError as exc:
        return DemoAnomalyValidationResult(
            row_count=row_count,
            anomaly_row_count=anomaly_row_count,
            workflow_status="NOT_RUN",
            selected_model=None,
            anomaly_event_count=0,
            detected_row_count=0,
            matched_injected_anomaly_rows=0,
            overlap_precision=None,
            random_baseline_anomaly_rate=baseline_rate,
            enrichment_factor=None,
            feature_count=len(feature_columns),
            feature_columns=tuple(feature_columns),
            excluded_columns=tuple(excluded_columns),
            leakage_free=False,
            passed=False,
            messages=(str(exc),),
        )

    with tempfile.TemporaryDirectory(prefix="demo_anomaly_") as temp_dir:
        csv_path = write_demo_workflow_csv(frame, Path(temp_dir) / "demo.csv")
        outcome = _run_anomaly_only_workflow(
            csv_path=csv_path,
            feature_columns=feature_columns,
        )
        report = outcome.report
        ground_truth_ids = _injected_anomaly_row_ids(frame)
        detected_ids = _event_row_ids(report.anomaly_events)
        matched_ids = sorted(set(detected_ids) & ground_truth_ids)
        precision, enrichment = _overlap_stats(
            detected_ids=detected_ids,
            ground_truth_ids=ground_truth_ids,
            baseline_rate=baseline_rate,
        )
        modeling_ok = _anomaly_modeling_succeeded(report)
        enrichment_ok = (
            len(matched_ids) >= 1
            and enrichment is not None
            and enrichment >= threshold
        )
        feature_policy_ok = _feature_policy_is_clean(
            feature_columns,
            analysis_mode=AnalysisExecutionMode.ANOMALY_ONLY,
        )
        passed = (
            leakage_free
            and modeling_ok
            and enrichment_ok
            and feature_policy_ok
        )

        if not modeling_ok:
            messages.append(
                "Anomaly-only workflow did not complete anomaly diagnosis "
                f"(status={report.status.value}, terminal={report.terminal_stage.value})."
            )
        if not enrichment_ok:
            messages.append(
                "Detected anomaly events were not sufficiently concentrated in "
                f"injected anomaly rows (enrichment={enrichment!r}, "
                f"threshold={threshold}, matched={len(matched_ids)})."
            )
        if not feature_policy_ok:
            messages.append(
                "Anomaly-only feature list retained forbidden identity, quality-output, "
                "or ground-truth columns."
            )
        if passed:
            messages.append("Anomaly-only demo acceptance criteria passed.")

        return DemoAnomalyValidationResult(
            row_count=row_count,
            anomaly_row_count=anomaly_row_count,
            workflow_status=report.status.value,
            selected_model=report.selected_anomaly_model_key,
            anomaly_event_count=int(report.anomaly_event_count),
            detected_row_count=len(detected_ids),
            matched_injected_anomaly_rows=len(matched_ids),
            overlap_precision=precision,
            random_baseline_anomaly_rate=baseline_rate,
            enrichment_factor=enrichment,
            feature_count=len(feature_columns),
            feature_columns=tuple(feature_columns),
            excluded_columns=tuple(excluded_columns),
            leakage_free=leakage_free,
            passed=passed,
            messages=tuple(messages),
        )


def validate_demo_workflows(
    configuration: DemoDatasetConfiguration | None = None,
    *,
    enrichment_threshold: float = MIN_ANOMALY_ENRICHMENT_FACTOR,
) -> DemoValidationReport:
    """Run supervised and anomaly-only demo acceptance checks."""
    config = configuration or DemoDatasetConfiguration()
    threshold = _validate_enrichment_threshold(enrichment_threshold)
    supervised = validate_demo_supervised(
        config,
        enrichment_threshold=threshold,
    )
    anomaly_only = validate_demo_anomaly_only(
        config,
        enrichment_threshold=threshold,
    )
    passed = supervised.passed and anomaly_only.passed
    messages: list[str] = []
    messages.extend(supervised.messages)
    messages.extend(anomaly_only.messages)
    messages.append(
        "Overall demo acceptance: PASS"
        if passed
        else "Overall demo acceptance: FAIL"
    )
    return DemoValidationReport(
        configuration=config,
        supervised=supervised,
        anomaly_only=anomaly_only,
        enrichment_threshold=threshold,
        passed=passed,
        messages=tuple(messages),
    )


def format_demo_validation_report(report: DemoValidationReport) -> str:
    """Return a concise human-readable acceptance report."""
    supervised = report.supervised
    anomaly = report.anomaly_only
    lines = [
        "Demo workflow acceptance report",
        f"row_count: {supervised.row_count}",
        f"anomaly_row_count: {supervised.anomaly_row_count}",
        f"enrichment_threshold: {report.enrichment_threshold}",
        "",
        "FEATURE_POLICY",
        f"  supervised_feature_count: {supervised.feature_count}",
        f"  anomaly_only_feature_count: {anomaly.feature_count}",
        (
            "  quality_outputs_and_ground_truth_excluded: "
            f"{_quality_and_ground_truth_excluded(supervised, anomaly)}"
        ),
        f"  supervised_excluded_columns: {list(supervised.excluded_columns)}",
        f"  anomaly_only_excluded_columns: {list(anomaly.excluded_columns)}",
        "",
        "SUPERVISED",
        f"  status: {supervised.workflow_status}",
        f"  selected_model: {supervised.selected_model}",
        f"  selected_task: {supervised.selected_task}",
        f"  feature_count: {supervised.feature_count}",
        f"  feature_columns: {list(supervised.feature_columns)}",
        f"  regression_metrics: {_format_metrics(supervised.regression_metrics)}",
        f"  naive_mean_baseline: {_format_metrics(supervised.naive_mean_baseline_metrics)}",
        f"  beats_mean_baseline: {supervised.beats_mean_baseline}",
        f"  anomaly_event_count: {supervised.anomaly_event_count}",
        (
            f"  residual_events: {supervised.residual_anomaly_event_count}, "
            f"matched: {supervised.residual_matched_injected_rows}, "
            f"enrichment: {_format_optional_float(supervised.residual_enrichment_factor)}"
        ),
        f"  recommendation_status: {supervised.recommendation_status}",
        f"  recommendation_change_count: {supervised.recommendation_change_count}",
        (
            "  recommendation_variables_within_bounds: "
            f"{supervised.recommendation_variables_within_bounds}"
        ),
        (
            "  baseline_raw_prediction: "
            f"{_format_optional_float(supervised.baseline_raw_prediction)}"
        ),
        (
            "  proposed_raw_prediction: "
            f"{_format_optional_float(supervised.proposed_raw_prediction)}"
        ),
        (
            "  proposed_prediction_plausibility: "
            f"{supervised.proposed_prediction_plausibility}"
        ),
        (
            "  declared_domain_extrapolation_detected: "
            f"{supervised.declared_domain_extrapolation_detected}"
        ),
        (
            "  recommendation_safety_handled: "
            f"{supervised.recommendation_safety_handled}"
        ),
        f"  passed: {supervised.passed}",
        "",
        "ANOMALY_ONLY",
        f"  status: {anomaly.workflow_status}",
        f"  selected_model: {anomaly.selected_model}",
        f"  feature_count: {anomaly.feature_count}",
        f"  feature_columns: {list(anomaly.feature_columns)}",
        f"  detected_rows: {anomaly.detected_row_count}",
        f"  matched_injected_rows: {anomaly.matched_injected_anomaly_rows}",
        f"  overlap_precision: {_format_optional_float(anomaly.overlap_precision)}",
        f"  baseline_anomaly_rate: {anomaly.random_baseline_anomaly_rate:.6f}",
        f"  enrichment: {_format_optional_float(anomaly.enrichment_factor)}",
        f"  passed: {anomaly.passed}",
        "",
        f"FINAL: {'PASS' if report.passed else 'FAIL'}",
    ]
    return "\n".join(lines)


def _summary_int(summary: Mapping[str, object], key: str) -> int:
    value = summary[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(
            f"summary[{key!r}] must be int, got {type(value).__name__}"
        )
    return value


def _validate_enrichment_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataValidationError(
            "enrichment_threshold must be a finite float > 1 "
            f"(bool not allowed), got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number) or number <= 1.0:
        raise DataValidationError(
            f"enrichment_threshold must be a finite float > 1.0, got {number}"
        )
    return number


def _lenient_performance_policy() -> ModelPerformanceAcceptancePolicy:
    return ModelPerformanceAcceptancePolicy(
        rules=[
            MetricAcceptanceRule(
                metric_name="rmse",
                direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
                threshold=1_000_000.0,
            ),
            MetricAcceptanceRule(
                metric_name="mae",
                direction=MetricAcceptanceDirection.LOWER_IS_BETTER,
                threshold=1_000_000.0,
            ),
            MetricAcceptanceRule(
                metric_name="r2",
                direction=MetricAcceptanceDirection.HIGHER_IS_BETTER,
                threshold=-1_000_000.0,
            ),
        ],
        minimum_test_rows=1,
    )


def _demo_workflow_policy() -> AnalysisWorkflowPolicy:
    return AnalysisWorkflowPolicy(
        allow_partial_diagnosis_ensemble=True,
        require_semiconductor_industry=False,
        require_regression_task=True,
        require_residual_diagnosis=True,
    )


def _supervised_role_overrides() -> dict[str, ColumnRole]:
    roles: dict[str, ColumnRole] = {
        name: ColumnRole.CONTROLLABLE_PROCESS for name in DEMO_CONTROLLABLE_COLUMNS
    }
    roles.update({name: ColumnRole.STATE_SENSOR for name in DEMO_MEASURED_COLUMNS})
    roles[TARGET_COLUMN] = ColumnRole.TARGET_QUALITY
    return roles


def _anomaly_role_overrides() -> dict[str, ColumnRole]:
    roles: dict[str, ColumnRole] = {
        name: ColumnRole.CONTROLLABLE_PROCESS for name in DEMO_CONTROLLABLE_COLUMNS
    }
    roles.update({name: ColumnRole.STATE_SENSOR for name in DEMO_MEASURED_COLUMNS})
    return roles


def _controllable_constraints() -> list[VariableConstraint]:
    bounds = demo_setpoint_bound_map()
    return [
        VariableConstraint(
            variable=name,
            adjustable=True,
            minimum=low,
            maximum=high,
            fixed=False,
        )
        for name, (low, high) in bounds.items()
    ]


def _run_supervised_workflow(
    *,
    csv_path: Path,
    feature_columns: list[str],
) -> AnalysisWorkflowOutcome:
    request = AnalysisWorkflowRequest(
        csv_path=csv_path,
        feature_columns=list(feature_columns),
        analysis_mode=AnalysisExecutionMode.SUPERVISED,
        target_column=TARGET_COLUMN,
        requested_task=AnalysisTask.REGRESSION,
        timestamp_column=TIMESTAMP_COLUMN,
        identifier_columns=list(IDENTIFIER_COLUMNS),
        excluded_columns=default_demo_excluded_columns(
            analysis_mode=AnalysisExecutionMode.SUPERVISED,
        ),
        column_role_overrides=_supervised_role_overrides(),
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        quality_direction=QualityOptimizationDirection.MAXIMIZE,
        declared_target_minimum=DEMO_QUALITY_SCORE_DECLARED_MINIMUM,
        declared_target_maximum=DEMO_QUALITY_SCORE_DECLARED_MAXIMUM,
        model_performance_policy=_lenient_performance_policy(),
        operating_point_selection=OperatingPointSelectionMode.TOP_RESIDUAL_ANOMALY,
        user_confirmed_controllable_variables=list(DEMO_CONTROLLABLE_COLUMNS),
        user_verified_variables=list(DEMO_CONTROLLABLE_COLUMNS),
        request_constraints=_controllable_constraints(),
        max_simultaneous_changes=2,
    )
    return IndustrialProcessAnalysisWorkflow(policy=_demo_workflow_policy()).run(request)


def _run_anomaly_only_workflow(
    *,
    csv_path: Path,
    feature_columns: list[str],
) -> AnalysisWorkflowOutcome:
    request = AnalysisWorkflowRequest(
        csv_path=csv_path,
        feature_columns=list(feature_columns),
        analysis_mode=AnalysisExecutionMode.ANOMALY_ONLY,
        timestamp_column=TIMESTAMP_COLUMN,
        identifier_columns=list(IDENTIFIER_COLUMNS),
        excluded_columns=default_demo_excluded_columns(
            analysis_mode=AnalysisExecutionMode.ANOMALY_ONLY,
        ),
        column_role_overrides=_anomaly_role_overrides(),
        operating_point_selection=OperatingPointSelectionMode.TOP_UNSUPERVISED_ANOMALY,
    )
    return IndustrialProcessAnalysisWorkflow(policy=_demo_workflow_policy()).run(request)


def _extract_regression_metrics(report: AnalysisWorkflowReport) -> dict[str, float]:
    assessment = report.model_performance_assessment
    if assessment is None:
        return {}
    metrics: dict[str, float] = {}
    for result in assessment.metric_results:
        if result.available and result.observed_value is not None:
            metrics[result.metric_name] = float(result.observed_value)
    return metrics


def _naive_mean_baseline_metrics(
    *,
    csv_path: Path,
    target_column: str,
) -> dict[str, float]:
    """Compute mean-target baseline metrics on the workflow TIME-split test set."""
    loaded = DatasetLoader().load(csv_path)
    sorted_frame = DatasetSorter().sort(loaded.frame, by=TIMESTAMP_COLUMN).frame
    split = DatasetSplitter().split(
        sorted_frame,
        SplitConfig(
            strategy=SplitStrategy.TIME,
            time_column=TIMESTAMP_COLUMN,
            test_size=0.20,
            validation_size=0.20,
            random_state=42,
        ),
    )
    train_val = pl.concat([split.train, split.validation], how="vertical_relaxed")
    if target_column not in train_val.columns or target_column not in split.test.columns:
        raise DataValidationError(
            f"target column {target_column!r} missing from split partitions"
        )
    mean_raw = train_val.get_column(target_column).mean()
    if isinstance(mean_raw, bool) or not isinstance(mean_raw, (int, float)):
        raise DataValidationError(
            "naive mean baseline target mean must be numeric, "
            f"got {type(mean_raw).__name__}"
        )
    mean_target = float(mean_raw)
    if not math.isfinite(mean_target):
        raise DataValidationError("naive mean baseline target mean must be finite")
    y_true = split.test.get_column(target_column).to_numpy()
    y_pred = np.full(shape=y_true.shape, fill_value=mean_target, dtype=float)
    metrics = evaluate_regression(y_true, y_pred)
    payload: dict[str, float] = {
        "rmse": float(metrics.rmse),
        "mae": float(metrics.mae),
    }
    if metrics.r2 is not None:
        payload["r2"] = float(metrics.r2)
    return payload


def _beats_mean_baseline(
    *,
    regression_metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
) -> bool:
    if "rmse" not in regression_metrics or "rmse" not in baseline_metrics:
        return False
    model_rmse = float(regression_metrics["rmse"])
    baseline_rmse = float(baseline_metrics["rmse"])
    if not math.isfinite(model_rmse) or not math.isfinite(baseline_rmse):
        return False
    return model_rmse < baseline_rmse


def _metrics_are_finite(metrics: Mapping[str, float]) -> bool:
    required = ("rmse", "mae")
    if any(name not in metrics for name in required):
        return False
    return all(math.isfinite(float(metrics[name])) for name in required)


def _supervised_modeling_succeeded(report: AnalysisWorkflowReport) -> bool:
    if report.selected_supervised_model_key is None:
        return False
    if report.selected_task is not AnalysisTask.REGRESSION:
        return False
    if report.model_performance_assessment is None:
        return False
    diagnosis = next(
        (
            record
            for record in report.stage_records
            if record.stage is AnalysisWorkflowStage.DIAGNOSIS
        ),
        None,
    )
    if diagnosis is None or not diagnosis.executed or not diagnosis.succeeded:
        return False
    if report.status is AnalysisWorkflowStatus.REFUSED:
        # Recommendation-stage safety refusals are allowed after successful diagnosis.
        refused_before_diagnosis = any(
            record.structured_refusal
            and record.stage
            not in {
                AnalysisWorkflowStage.RECOMMENDATION,
                AnalysisWorkflowStage.WHAT_IF_VERIFICATION,
            }
            for record in report.stage_records
        )
        return not refused_before_diagnosis
    return report.status in {
        AnalysisWorkflowStatus.COMPLETED,
        AnalysisWorkflowStatus.PARTIAL,
    }


def _anomaly_modeling_succeeded(report: AnalysisWorkflowReport) -> bool:
    if report.selected_anomaly_model_key is None:
        return False
    if report.anomaly_event_count < 1:
        return False
    diagnosis = next(
        (
            record
            for record in report.stage_records
            if record.stage is AnalysisWorkflowStage.DIAGNOSIS
        ),
        None,
    )
    if diagnosis is None or not diagnosis.executed or not diagnosis.succeeded:
        return False
    return report.status in {
        AnalysisWorkflowStatus.COMPLETED,
        AnalysisWorkflowStatus.PARTIAL,
    }


def _injected_anomaly_row_ids(frame: pd.DataFrame) -> set[int]:
    return injected_anomaly_row_ids(frame)


def _event_row_ids(events: Sequence[object]) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    for event in events:
        sample_id = getattr(event, "sample_id", None)
        if isinstance(sample_id, bool) or not isinstance(sample_id, int):
            continue
        if sample_id in seen:
            continue
        seen.add(sample_id)
        ids.append(sample_id)
    return ids


def _overlap_stats(
    *,
    detected_ids: Sequence[int],
    ground_truth_ids: set[int],
    baseline_rate: float,
) -> tuple[float | None, float | None]:
    return overlap_precision_and_enrichment(
        detected_ids=detected_ids,
        ground_truth_ids=ground_truth_ids,
        baseline_rate=baseline_rate,
    )


def _format_metrics(metrics: Mapping[str, float]) -> str:
    if not metrics:
        return "{}"
    parts = [
        f"{name}={value:.6f}"
        for name, value in metrics.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    return "{" + ", ".join(parts) + "}"


def _format_optional_float(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.6f}"


def _quality_and_ground_truth_excluded(
    supervised: DemoSupervisedValidationResult,
    anomaly: DemoAnomalyValidationResult,
) -> bool:
    """Confirm quality outputs and ground-truth metadata are absent from features."""
    forbidden = (*DEMO_QUALITY_COLUMNS, *GROUND_TRUTH_METADATA_COLUMNS)
    supervised_clean = not any(name in supervised.feature_columns for name in forbidden)
    anomaly_clean = not any(name in anomaly.feature_columns for name in forbidden)
    return supervised_clean and anomaly_clean
