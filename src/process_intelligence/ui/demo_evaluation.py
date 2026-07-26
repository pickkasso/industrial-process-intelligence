"""Built-in-demo-only ground-truth evaluation for displayed anomaly events.

Presentation-only helpers. Never feeds ground-truth labels into modeling,
diagnosis, recommendation, or run readiness.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Final

import pandas as pd  # type: ignore[import-untyped]
import streamlit as st

from process_intelligence.demo_data import GROUND_TRUTH_METADATA_COLUMNS
from process_intelligence.demo_evaluation import (
    available_anomaly_types,
    dataset_anomaly_prevalence,
    ground_truth_lookup,
    injected_anomaly_row_ids,
    overlap_precision_and_enrichment,
)
from process_intelligence.reporting.schemas import (
    AnomalyEventView,
    WorkflowPresentationReport,
)
from process_intelligence.ui.demo_configuration import is_builtin_demo_source
from process_intelligence.workflow.enums import AnalysisExecutionMode

_RESIDUAL_DETECTOR: Final[str] = "residual_anomaly_detector"
_NOT_AVAILABLE: Final[str] = "N/A"

_PANEL_NOTICE: Final[str] = (
    "This section is available only because the built-in synthetic dataset "
    "contains known injected anomalies. These labels were not used for model "
    "training or anomaly scoring. Results do not represent production accuracy."
)


class DemoAnomalyEvaluationStatus(StrEnum):
    """Availability status for demo ground-truth evaluation."""

    AVAILABLE = "AVAILABLE"
    NO_DISPLAYED_EVENTS = "NO_DISPLAYED_EVENTS"
    NO_GROUND_TRUTH_ANOMALIES = "NO_GROUND_TRUTH_ANOMALIES"
    INVALID_SAMPLE_IDS = "INVALID_SAMPLE_IDS"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class DemoDetectedEventEvaluation:
    """Evaluation of one displayed anomaly-event representative."""

    display_rank: int
    original_row_id: int | None
    anomaly_score: float | None
    ground_truth_match: bool | None
    anomaly_type: str | None
    timestamp: str | None
    sample_id_valid: bool
    evaluation_note: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly dictionary representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DemoAnomalyEvaluationSummary:
    """Serializable demo-only evaluation of displayed anomaly events."""

    evaluated_event_count: int
    matched_event_count: int
    unmatched_event_count: int
    selected_event_precision: float | None
    dataset_anomaly_row_count: int
    dataset_anomaly_prevalence: float | None
    enrichment_factor: float | None
    matched_anomaly_types: tuple[str, ...]
    available_anomaly_types: tuple[str, ...]
    evaluation_status: DemoAnomalyEvaluationStatus
    explanatory_messages: tuple[str, ...] = field(default_factory=tuple)
    event_details: tuple[DemoDetectedEventEvaluation, ...] = field(
        default_factory=tuple
    )
    analysis_mode: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly dictionary representation."""
        return {
            "evaluated_event_count": self.evaluated_event_count,
            "matched_event_count": self.matched_event_count,
            "unmatched_event_count": self.unmatched_event_count,
            "selected_event_precision": self.selected_event_precision,
            "dataset_anomaly_row_count": self.dataset_anomaly_row_count,
            "dataset_anomaly_prevalence": self.dataset_anomaly_prevalence,
            "enrichment_factor": self.enrichment_factor,
            "matched_anomaly_types": list(self.matched_anomaly_types),
            "available_anomaly_types": list(self.available_anomaly_types),
            "evaluation_status": self.evaluation_status.value,
            "explanatory_messages": list(self.explanatory_messages),
            "event_details": [item.to_dict() for item in self.event_details],
            "analysis_mode": self.analysis_mode,
        }


def is_demo_evaluation_panel_eligible(
    *,
    data_source: object,
    report: WorkflowPresentationReport | None,
    current_dataset_fingerprint: str | None,
) -> bool:
    """Return True when the demo evaluation panel may be shown.

    Requires the active built-in demo source and a non-stale report whose
    dataset fingerprint matches the currently loaded demo bytes. Column-name
    heuristics alone are never used to identify the demo.
    """
    if not is_builtin_demo_source(data_source):
        return False
    if report is None:
        return False
    if report.dataset_fingerprint is None or current_dataset_fingerprint is None:
        return False
    return report.dataset_fingerprint == current_dataset_fingerprint


def select_displayed_events_for_evaluation(
    report: WorkflowPresentationReport,
) -> tuple[list[AnomalyEventView], str | None]:
    """Select presentation events for demo evaluation by analysis mode.

    ANOMALY_ONLY uses all displayed unsupervised anomaly events.
    SUPERVISED uses residual anomaly events only when present.
    """
    if not isinstance(report, WorkflowPresentationReport):
        raise TypeError(
            "report must be WorkflowPresentationReport, "
            f"got {type(report).__name__}"
        )
    mode_value = report.metadata.get("analysis_mode")
    analysis_mode = str(mode_value) if mode_value is not None else None
    events = list(report.anomaly_events)
    if analysis_mode == AnalysisExecutionMode.SUPERVISED.value:
        residual = [
            event
            for event in events
            if event.selection_source == _RESIDUAL_DETECTOR
        ]
        return residual, analysis_mode
    return events, analysis_mode


def evaluate_demo_anomaly_events(
    *,
    events: Sequence[AnomalyEventView],
    demo_frame: pd.DataFrame,
    analysis_mode: str | None = None,
) -> DemoAnomalyEvaluationSummary:
    """Compare displayed anomaly-event representatives to demo ground truth.

    Metrics describe precision among the displayed event representatives only.
    This is not detector recall and not full anomaly-model precision.
    """
    if not isinstance(demo_frame, pd.DataFrame):
        raise TypeError(
            f"demo_frame must be a pandas.DataFrame, got {type(demo_frame).__name__}"
        )
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise TypeError(
            "events must be a sequence of AnomalyEventView, "
            f"got {type(events).__name__}"
        )

    missing_ground_truth = [
        name
        for name in GROUND_TRUTH_METADATA_COLUMNS
        if name not in demo_frame.columns
    ]
    if missing_ground_truth:
        return DemoAnomalyEvaluationSummary(
            evaluated_event_count=0,
            matched_event_count=0,
            unmatched_event_count=0,
            selected_event_precision=None,
            dataset_anomaly_row_count=0,
            dataset_anomaly_prevalence=None,
            enrichment_factor=None,
            matched_anomaly_types=(),
            available_anomaly_types=(),
            evaluation_status=DemoAnomalyEvaluationStatus.UNAVAILABLE,
            explanatory_messages=(
                "Demo ground-truth columns are missing from the loaded dataset; "
                "evaluation is unavailable.",
            ),
            event_details=(),
            analysis_mode=analysis_mode,
        )

    total_rows = int(len(demo_frame))
    ground_truth_ids = injected_anomaly_row_ids(demo_frame)
    anomaly_row_count = len(ground_truth_ids)
    prevalence = dataset_anomaly_prevalence(
        anomaly_row_count=anomaly_row_count,
        total_row_count=total_rows,
    )
    type_catalog = available_anomaly_types(demo_frame)

    if not events:
        status = (
            DemoAnomalyEvaluationStatus.NO_GROUND_TRUTH_ANOMALIES
            if anomaly_row_count == 0
            else DemoAnomalyEvaluationStatus.NO_DISPLAYED_EVENTS
        )
        messages = _interpretation_messages(
            status=status,
            selected_event_precision=None,
            enrichment_factor=None,
            analysis_mode=analysis_mode,
        )
        return DemoAnomalyEvaluationSummary(
            evaluated_event_count=0,
            matched_event_count=0,
            unmatched_event_count=0,
            selected_event_precision=None,
            dataset_anomaly_row_count=anomaly_row_count,
            dataset_anomaly_prevalence=prevalence,
            enrichment_factor=None,
            matched_anomaly_types=(),
            available_anomaly_types=type_catalog,
            evaluation_status=status,
            explanatory_messages=messages,
            event_details=(),
            analysis_mode=analysis_mode,
        )

    if anomaly_row_count == 0:
        zero_gt_details: list[DemoDetectedEventEvaluation] = []
        zero_gt_seen: set[int] = set()
        for event in events:
            detail, row_id = _evaluate_single_event(
                event=event,
                demo_frame=demo_frame,
                ground_truth_ids=ground_truth_ids,
                seen_ids=zero_gt_seen,
            )
            zero_gt_details.append(detail)
            if detail.sample_id_valid and row_id is not None:
                zero_gt_seen.add(row_id)
        messages = _interpretation_messages(
            status=DemoAnomalyEvaluationStatus.NO_GROUND_TRUTH_ANOMALIES,
            selected_event_precision=None,
            enrichment_factor=None,
            analysis_mode=analysis_mode,
        )
        return DemoAnomalyEvaluationSummary(
            evaluated_event_count=0,
            matched_event_count=0,
            unmatched_event_count=0,
            selected_event_precision=None,
            dataset_anomaly_row_count=0,
            dataset_anomaly_prevalence=prevalence,
            enrichment_factor=None,
            matched_anomaly_types=(),
            available_anomaly_types=type_catalog,
            evaluation_status=DemoAnomalyEvaluationStatus.NO_GROUND_TRUTH_ANOMALIES,
            explanatory_messages=messages,
            event_details=tuple(zero_gt_details),
            analysis_mode=analysis_mode,
        )

    details_list: list[DemoDetectedEventEvaluation] = []
    valid_ids: list[int] = []
    seen_ids: set[int] = set()
    matched_types: list[str] = []
    matched_type_seen: set[str] = set()

    for event in events:
        detail, row_id = _evaluate_single_event(
            event=event,
            demo_frame=demo_frame,
            ground_truth_ids=ground_truth_ids,
            seen_ids=seen_ids,
        )
        details_list.append(detail)
        if not detail.sample_id_valid or row_id is None:
            continue
        if row_id in seen_ids:
            # Duplicate displayed sample IDs: keep first occurrence only.
            continue
        seen_ids.add(row_id)
        valid_ids.append(row_id)
        if detail.ground_truth_match and detail.anomaly_type is not None:
            if detail.anomaly_type not in matched_type_seen:
                matched_type_seen.add(detail.anomaly_type)
                matched_types.append(detail.anomaly_type)

    if not valid_ids:
        messages = _interpretation_messages(
            status=DemoAnomalyEvaluationStatus.INVALID_SAMPLE_IDS,
            selected_event_precision=None,
            enrichment_factor=None,
            analysis_mode=analysis_mode,
        )
        return DemoAnomalyEvaluationSummary(
            evaluated_event_count=0,
            matched_event_count=0,
            unmatched_event_count=0,
            selected_event_precision=None,
            dataset_anomaly_row_count=anomaly_row_count,
            dataset_anomaly_prevalence=prevalence,
            enrichment_factor=None,
            matched_anomaly_types=(),
            available_anomaly_types=type_catalog,
            evaluation_status=DemoAnomalyEvaluationStatus.INVALID_SAMPLE_IDS,
            explanatory_messages=messages,
            event_details=tuple(details_list),
            analysis_mode=analysis_mode,
        )

    matched_count = sum(1 for row_id in valid_ids if row_id in ground_truth_ids)
    unmatched_count = len(valid_ids) - matched_count
    precision, enrichment = overlap_precision_and_enrichment(
        detected_ids=valid_ids,
        ground_truth_ids=ground_truth_ids,
        baseline_rate=prevalence if prevalence is not None else 0.0,
    )
    messages = _interpretation_messages(
        status=DemoAnomalyEvaluationStatus.AVAILABLE,
        selected_event_precision=precision,
        enrichment_factor=enrichment,
        analysis_mode=analysis_mode,
    )
    return DemoAnomalyEvaluationSummary(
        evaluated_event_count=len(valid_ids),
        matched_event_count=matched_count,
        unmatched_event_count=unmatched_count,
        selected_event_precision=precision,
        dataset_anomaly_row_count=anomaly_row_count,
        dataset_anomaly_prevalence=prevalence,
        enrichment_factor=enrichment,
        matched_anomaly_types=tuple(sorted(matched_types)),
        available_anomaly_types=type_catalog,
        evaluation_status=DemoAnomalyEvaluationStatus.AVAILABLE,
        explanatory_messages=messages,
        event_details=tuple(details_list),
        analysis_mode=analysis_mode,
    )


def evaluate_demo_anomaly_presentation(
    *,
    report: WorkflowPresentationReport,
    demo_frame: pd.DataFrame,
) -> DemoAnomalyEvaluationSummary:
    """Evaluate the presentation report's displayed anomaly events on a demo frame."""
    events, analysis_mode = select_displayed_events_for_evaluation(report)
    return evaluate_demo_anomaly_events(
        events=events,
        demo_frame=demo_frame,
        analysis_mode=analysis_mode,
    )


def format_percentage(value: float | None) -> str:
    """Format a ratio as a percentage, or ``N/A`` when unavailable."""
    if value is None or not math.isfinite(float(value)):
        return _NOT_AVAILABLE
    return f"{100.0 * float(value):.1f}%"


def format_enrichment(value: float | None) -> str:
    """Format an enrichment factor with a multiplication suffix."""
    if value is None or not math.isfinite(float(value)):
        return _NOT_AVAILABLE
    return f"{float(value):.2f}×"


def format_optional_score(value: float | None) -> str:
    """Format an anomaly score, treating non-finite values as unavailable."""
    if value is None:
        return _NOT_AVAILABLE
    number = float(value)
    if not math.isfinite(number):
        return _NOT_AVAILABLE
    return f"{number:.6g}"


def render_demo_ground_truth_evaluation(
    summary: DemoAnomalyEvaluationSummary,
) -> None:
    """Render the compact demo ground-truth evaluation panel in Streamlit."""
    if not isinstance(summary, DemoAnomalyEvaluationSummary):
        raise TypeError(
            "summary must be DemoAnomalyEvaluationSummary, "
            f"got {type(summary).__name__}"
        )

    st.header("Demo ground-truth evaluation")
    st.info(_PANEL_NOTICE)
    st.caption(
        "Metrics describe precision among the displayed event representatives "
        "only. The workflow may expose only a limited number of top events. "
        "This is not detector recall and not full anomaly-model precision."
    )
    st.caption(f"Evaluation status: {summary.evaluation_status.value}")
    metric_cols = st.columns(3)
    metric_cols[0].metric(
        "Displayed events evaluated",
        summary.evaluated_event_count,
    )
    metric_cols[1].metric(
        "Ground-truth matches",
        summary.matched_event_count,
    )
    metric_cols[2].metric(
        "Precision among displayed event representatives",
        format_percentage(summary.selected_event_precision),
    )
    metric_cols2 = st.columns(2)
    metric_cols2[0].metric(
        "Full-demo anomaly prevalence",
        format_percentage(summary.dataset_anomaly_prevalence),
    )
    metric_cols2[1].metric(
        "Enrichment over demo prevalence",
        format_enrichment(summary.enrichment_factor),
    )
    if summary.matched_anomaly_types:
        st.caption(
            "Matched anomaly types: "
            + ", ".join(summary.matched_anomaly_types)
        )
    else:
        st.caption("Matched anomaly types: none")

    if summary.event_details:
        st.dataframe(
            [
                {
                    "Rank": detail.display_rank,
                    "Original row ID": (
                        _NOT_AVAILABLE
                        if detail.original_row_id is None
                        else detail.original_row_id
                    ),
                    "Score": format_optional_score(detail.anomaly_score),
                    "Ground-truth match": _match_label(detail.ground_truth_match),
                    "Injected anomaly type": (
                        _NOT_AVAILABLE
                        if detail.anomaly_type is None
                        else detail.anomaly_type
                    ),
                    "Timestamp": (
                        _NOT_AVAILABLE
                        if detail.timestamp is None
                        else detail.timestamp
                    ),
                }
                for detail in summary.event_details
            ],
            use_container_width=True,
            hide_index=True,
        )

    for message in summary.explanatory_messages:
        st.write(message)


def _evaluate_single_event(
    *,
    event: AnomalyEventView,
    demo_frame: pd.DataFrame,
    ground_truth_ids: set[int],
    seen_ids: set[int],
) -> tuple[DemoDetectedEventEvaluation, int | None]:
    rank = int(event.rank)
    score = _safe_score(event.anomaly_score)
    raw_id = event.original_row_id
    if isinstance(raw_id, bool) or not isinstance(raw_id, int):
        return (
            DemoDetectedEventEvaluation(
                display_rank=rank,
                original_row_id=None,
                anomaly_score=score,
                ground_truth_match=None,
                anomaly_type=None,
                timestamp=None,
                sample_id_valid=False,
                evaluation_note="Sample ID is not a valid integer row identity.",
            ),
            None,
        )

    lookup = ground_truth_lookup(demo_frame, row_id=raw_id)
    if lookup is None:
        return (
            DemoDetectedEventEvaluation(
                display_rank=rank,
                original_row_id=raw_id,
                anomaly_score=score,
                ground_truth_match=None,
                anomaly_type=None,
                timestamp=None,
                sample_id_valid=False,
                evaluation_note="Sample ID is outside the demo dataset row range.",
            ),
            None,
        )

    is_duplicate = raw_id in seen_ids
    matched = bool(raw_id in ground_truth_ids)
    anomaly_type = lookup.get("anomaly_type")
    anomaly_type_text = (
        None if anomaly_type is None else str(anomaly_type)
    )
    timestamp_text = _format_timestamp(lookup.get("timestamp"))
    if is_duplicate:
        note = (
            "Duplicate displayed sample ID; metrics keep the first occurrence."
        )
    elif matched:
        note = "Displayed event representative matches an injected anomaly row."
    else:
        note = (
            "Displayed event representative does not match an injected anomaly "
            "row. Unmatched events are not automatically proven false positives."
        )
    return (
        DemoDetectedEventEvaluation(
            display_rank=rank,
            original_row_id=raw_id,
            anomaly_score=score,
            ground_truth_match=matched,
            anomaly_type=anomaly_type_text,
            timestamp=timestamp_text,
            sample_id_valid=True,
            evaluation_note=note,
        ),
        raw_id,
    )


def _safe_score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _format_timestamp(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return str(value.isoformat())
    text = str(value).strip()
    return text if text != "" else None


def _match_label(value: bool | None) -> str:
    if value is None:
        return _NOT_AVAILABLE
    return "true" if value else "false"


def _interpretation_messages(
    *,
    status: DemoAnomalyEvaluationStatus,
    selected_event_precision: float | None,
    enrichment_factor: float | None,
    analysis_mode: str | None,
) -> tuple[str, ...]:
    messages: list[str] = [
        (
            "Displayed-event precision is the share of valid displayed event "
            "representatives that land on injected anomaly rows. It is not "
            "detector recall or production accuracy."
        )
    ]
    if analysis_mode == AnalysisExecutionMode.SUPERVISED.value:
        messages.append(
            "Supervised evaluation uses residual anomaly events exposed by the "
            "presentation report when available."
        )
    if status is DemoAnomalyEvaluationStatus.NO_DISPLAYED_EVENTS:
        messages.append(
            "No anomaly-event representatives were available for evaluation."
        )
        return tuple(messages)
    if status is DemoAnomalyEvaluationStatus.NO_GROUND_TRUTH_ANOMALIES:
        messages.append(
            "The loaded demo dataset contains no injected anomaly rows."
        )
        return tuple(messages)
    if status is DemoAnomalyEvaluationStatus.INVALID_SAMPLE_IDS:
        messages.append(
            "Displayed anomaly events did not provide valid original row IDs "
            "for ground-truth comparison."
        )
        return tuple(messages)
    if status is DemoAnomalyEvaluationStatus.UNAVAILABLE:
        messages.append("Demo ground-truth evaluation is unavailable.")
        return tuple(messages)

    if (
        enrichment_factor is not None
        and math.isfinite(enrichment_factor)
        and enrichment_factor > 1.0
    ):
        messages.append(
            "Displayed anomaly events are more concentrated in injected anomaly "
            "rows than the overall demo prevalence."
        )
    elif enrichment_factor is not None and math.isfinite(enrichment_factor):
        messages.append(
            "Displayed events are not more concentrated in injected anomalies "
            "than the demo baseline."
        )
    else:
        messages.append(
            "Enrichment over demo prevalence could not be computed for the "
            "current displayed events."
        )
    if selected_event_precision is not None:
        messages.append(
            "Unmatched displayed events are not automatically proven false "
            "positives, because synthetic labels may not represent every "
            "unusual process state."
        )
    return tuple(messages)


def summary_contains_no_runtime_objects(summary: DemoAnomalyEvaluationSummary) -> bool:
    """Return True when ``summary.to_dict()`` contains only JSON-safe scalars."""
    payload = summary.to_dict()
    return _is_json_safe(payload)


def _is_json_safe(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _is_json_safe(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(_is_json_safe(item) for item in value)
    return False
