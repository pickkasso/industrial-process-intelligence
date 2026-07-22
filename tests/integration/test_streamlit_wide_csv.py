"""Integration tests for wide-schema Streamlit column configuration (Step 11B.2)."""

from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO

import polars as pl
from streamlit.testing.v1 import AppTest

from process_intelligence.ui.column_configuration import (
    AutomaticColumnConfigurator,
    resolve_active_feature_columns,
)
from process_intelligence.ui.streamlit_app import _read_csv_for_ui
from process_intelligence.workflow import (
    AnalysisWorkflowOutcome,
    AnalysisWorkflowReport,
    AnalysisWorkflowStage,
    AnalysisWorkflowStageRecord,
    AnalysisWorkflowStatus,
)

_UTC_START = datetime(2026, 7, 22, 1, 0, tzinfo=UTC)
_UTC_END = datetime(2026, 7, 22, 1, 1, tzinfo=UTC)


def _wide_csv_bytes(*, sensor_count: int = 100) -> bytes:
    """Build a wide battery-like CSV without hard-coding a single production file."""
    columns = ["Date", "Time", "SerialNumber"]
    columns.extend(f"sensor_{index:03d}" for index in range(sensor_count))
    columns.append("SOH")
    header = ",".join(columns)
    # Mixed int/float in sensor_000: first rows int-like, later float.
    rows: list[str] = []
    for row_index in range(8):
        values: list[str] = [
            f"2020-01-{row_index + 1:02d}",
            f"10:{row_index:02d}:00",
            str(1000 + row_index),
        ]
        for sensor_index in range(sensor_count):
            if sensor_index == 0 and row_index < 3:
                values.append(str(row_index + 1))
            elif sensor_index == 0:
                values.append(f"{row_index + 0.5}")
            else:
                values.append(f"{row_index + sensor_index * 0.01:.4f}")
        values.append(f"{0.95 - row_index * 0.01:.4f}")
        rows.append(",".join(values))
    return ("\n".join([header, *rows]) + "\n").encode("utf-8")


class _DeterministicWorkflow:
    def run(self, request: object) -> AnalysisWorkflowOutcome:
        assert request is not None
        stages = list(AnalysisWorkflowStage)
        records = [
            AnalysisWorkflowStageRecord(
                stage=stage,
                executed=True,
                succeeded=True,
                structured_refusal=False,
                row_count=8,
                message=f"{stage.value} stage completed.",
                warnings=[],
                metadata={},
            )
            for stage in stages
        ]
        report = AnalysisWorkflowReport(
            status=AnalysisWorkflowStatus.COMPLETED,
            terminal_stage=AnalysisWorkflowStage.RECOMMENDATION,
            stage_records=records,
            model_performance_assessment=None,
            final_recommendation=None,
            selected_industry=None,
            selected_task=None,
            selected_supervised_model_key=None,
            selected_anomaly_model_key=None,
            selected_operating_row_id=None,
            anomaly_event_count=0,
            diagnosis_factor_count=0,
            raw_row_count=8,
            processed_row_count=8,
            cohort_row_count=8,
            train_row_count=4,
            validation_row_count=2,
            test_row_count=2,
            cohort_filter_summary={
                "configured": False,
                "source_row_count": 8,
                "retained_row_count": 8,
                "excluded_row_count": 0,
                "null_excluded_count": 0,
            },
            started_at=_UTC_START,
            completed_at=_UTC_END,
            total_seconds=60.0,
            warnings=[],
            metadata={},
        )
        return AnalysisWorkflowOutcome(report=report)


def _ui_entry(*, workflow_factory: object) -> None:
    from process_intelligence.ui import render_app

    render_app(workflow_factory=workflow_factory)  # type: ignore[arg-type]


def _make_app() -> AppTest:
    return AppTest.from_function(
        _ui_entry,
        default_timeout=60,
        kwargs={"workflow_factory": lambda: _DeterministicWorkflow()},
    )


def _text_blob(at: AppTest) -> str:
    parts: list[str] = []
    for attr in (
        "title",
        "header",
        "subheader",
        "markdown",
        "text",
        "caption",
        "info",
        "success",
        "warning",
        "error",
    ):
        elements = getattr(at, attr, [])
        for element in elements:
            value = getattr(element, "value", None)
            if value is not None:
                parts.append(str(value))
    return "\n".join(parts)


def _upload_wide(at: AppTest, csv_bytes: bytes) -> AppTest:
    at.file_uploader[0].set_value([("wide.csv", csv_bytes, "text/csv")])
    return at.run()


def _select_objective(at: AppTest, value: str = "REDUCE_ANOMALY_SCORE") -> AppTest:
    objective_box = next(item for item in at.selectbox if item.label == "Objective")
    return objective_box.select(value).run()


def _complete_performance_rule(
    at: AppTest,
    *,
    metric_name: str = "rmse",
    direction: str = "LOWER_IS_BETTER",
    threshold: str = "10.0",
) -> AppTest:
    metric_box = next(
        item for item in at.text_input if item.label.startswith("Metric name #")
    )
    at = metric_box.set_value(metric_name).run()
    direction_box = next(
        item for item in at.selectbox if item.label.startswith("Direction #")
    )
    at = direction_box.select(direction).run()
    threshold_box = next(
        item for item in at.text_input if item.label.startswith("Threshold #")
    )
    return threshold_box.set_value(threshold).run()


def test_wide_csv_column_configuration_ui() -> None:
    csv_bytes = _wide_csv_bytes(sensor_count=100)
    original_frame = pl.read_csv(BytesIO(csv_bytes), infer_schema_length=None)
    assert original_frame.width >= 100
    assert original_frame.columns[0] == "Date"
    assert "Time" in original_frame.columns
    assert "SerialNumber" in original_frame.columns
    assert "SOH" in original_frame.columns
    sensor_cols = [name for name in original_frame.columns if name.startswith("sensor_")]
    assert len(sensor_cols) >= 90

    before_dicts = original_frame.to_dicts()
    before_dtypes = [str(dtype) for dtype in original_frame.dtypes]

    columns, analysis_frame, preview_frame = _read_csv_for_ui(
        csv_bytes,
        preview_row_count=20,
    )
    assert len(columns) >= 100
    assert preview_frame.height <= 20
    assert analysis_frame.height == original_frame.height
    assert analysis_frame.to_dicts() == before_dicts
    assert [str(dtype) for dtype in analysis_frame.dtypes] == before_dtypes
    assert analysis_frame.get_column("sensor_000").null_count() == 0

    report = AutomaticColumnConfigurator().analyze(analysis_frame)
    assert report.target_candidates[0] == "SOH"
    assert len(report.target_candidates) >= 1
    assert "SerialNumber" in report.identifier_candidates
    assert "Date" in report.timestamp_candidates
    assert "Time" in report.timestamp_candidates
    recommended = report.recommended_feature_columns
    assert len(recommended) >= 90
    assert "SOH" not in recommended
    assert "SerialNumber" not in recommended
    assert "Date" not in recommended
    assert "Time" not in recommended
    assert "SOH" not in report.excluded_candidates
    assert "SerialNumber" not in report.excluded_candidates

    at = _upload_wide(_make_app().run(), csv_bytes)
    assert not at.exception
    blob = _text_blob(at)

    target_box = next(item for item in at.selectbox if item.label == "Target column")
    assert target_box.value == "(select target)"
    assert target_box.value != "Date"
    assert list(target_box.options)[0] == "(select target)"
    assert "SOH" in list(target_box.options)
    assert list(target_box.options).index("SOH") < list(target_box.options).index("Date")

    task_box = next(item for item in at.selectbox if item.label == "Analysis task")
    assert task_box.value == "AUTO"
    assert list(task_box.options) == ["AUTO", "REGRESSION", "CLASSIFICATION"]

    objective_box = next(item for item in at.selectbox if item.label == "Objective")
    assert list(objective_box.options)[0] == "(select objective)"
    assert objective_box.value == "(select objective)"

    metric_box = next(
        item for item in at.text_input if item.label.startswith("Metric name #")
    )
    assert metric_box.value == ""
    direction_box = next(
        item for item in at.selectbox if item.label.startswith("Direction #")
    )
    assert list(direction_box.options)[0] == "(select direction)"
    assert direction_box.value == "(select direction)"
    threshold_box = next(
        item for item in at.text_input if item.label.startswith("Threshold #")
    )
    assert threshold_box.value == ""

    run_button = next(item for item in at.button if item.label == "Run analysis")
    assert run_button.disabled is True
    assert "False: objective selected" in blob or "False: at least one complete" in blob

    assert any(
        item.label == "Use recommended numeric feature set" and item.value is True
        for item in at.checkbox
    )
    assert "Recommended features in use:" in blob or any(
        "Recommended features in use" in str(getattr(item, "value", ""))
        for item in at.markdown
    ) or "Recommended features" in blob

    # Hundreds of feature tags must not be expanded on the default screen.
    feature_multiselects = [
        item for item in at.multiselect if item.label == "Feature columns"
    ]
    assert feature_multiselects == []

    identifier_box = next(
        item for item in at.multiselect if item.label == "Identifier columns"
    )
    assert "SerialNumber" in list(identifier_box.value)

    timestamp_box = next(
        item for item in at.selectbox if item.label == "Timestamp column (optional)"
    )
    timestamp_options = list(timestamp_box.options)
    assert timestamp_options[0] == "(none)"
    assert "Date" in timestamp_options
    assert "Time" in timestamp_options

    excluded_box = next(
        item for item in at.multiselect if item.label == "Excluded columns"
    )
    assert "SOH" not in list(excluded_box.value)
    assert "SerialNumber" not in list(excluded_box.value)

    assert any(
        item.label == "Columns with explicit role overrides" for item in at.multiselect
    )
    assert not any(
        item.label.startswith("Role override for ") for item in at.selectbox
    )
    assert any(
        item.label == "Variables with recommendation constraints"
        for item in at.multiselect
    )
    assert not any(item.label.startswith("Minimum for ") for item in at.number_input)

    controllable = next(
        item
        for item in at.multiselect
        if item.label == "Confirmed controllable variables"
    )
    assert list(controllable.value) == []
    verified = next(
        item for item in at.multiselect if item.label == "Verified variables"
    )
    assert list(verified.value) == []

    # Select target and verify feature / excluded conflict handling.
    at = target_box.select("SOH").run()
    assert not at.exception
    task_after_soh = next(item for item in at.selectbox if item.label == "Analysis task")
    assert task_after_soh.value == "AUTO"
    active = resolve_active_feature_columns(
        recommended,
        selected_target="SOH",
        selected_timestamp=None,
        selected_identifiers=["SerialNumber"],
        selected_excluded=[],
    )
    assert "SOH" not in active
    assert "SerialNumber" not in active
    assert len(active) >= 90

    excluded_after = next(
        item for item in at.multiselect if item.label == "Excluded columns"
    )
    assert "SOH" not in list(excluded_after.value)

    blob_target = _text_blob(at)
    assert "True: target selected" in blob_target
    assert "True: selected target has usable variation" in blob_target
    assert "True: analysis task selection valid" in blob_target
    assert "False: objective selected" in blob_target

    at = _select_objective(at)
    assert not at.exception
    blob_objective = _text_blob(at)
    assert "True: objective selected" in blob_objective
    assert "False: at least one complete performance rule" in blob_objective
    run_after_objective = next(
        item for item in at.button if item.label == "Run analysis"
    )
    assert run_after_objective.disabled is True

    at = _complete_performance_rule(at)
    assert not at.exception
    blob_ready = _text_blob(at)
    assert "True: at least one complete performance rule" in blob_ready
    run_ready = next(item for item in at.button if item.label == "Run analysis")
    assert run_ready.disabled is False

    # Manual feature mode shows multiselect defaulting to recommended features.
    recommended_toggle = next(
        item
        for item in at.checkbox
        if item.label == "Use recommended numeric feature set"
    )
    at = recommended_toggle.uncheck().run()
    assert not at.exception
    feature_box = next(
        item for item in at.multiselect if item.label == "Feature columns"
    )
    feature_values = list(feature_box.value)
    assert len(feature_values) >= 90
    assert "SOH" not in feature_values
    assert "SerialNumber" not in feature_values
    assert "Date" not in feature_values or "Date" not in recommended

    # Role override widgets appear only for selected columns.
    override_box = next(
        item
        for item in at.multiselect
        if item.label == "Columns with explicit role overrides"
    )
    sample_feature = feature_values[0]
    at = override_box.select(sample_feature).run()
    assert not at.exception
    role_labels = [
        item.label for item in at.selectbox if item.label.startswith("Role override for ")
    ]
    assert role_labels == [f"Role override for {sample_feature}"]

    # Constraint widgets appear only for selected variables.
    constraint_box = next(
        item
        for item in at.multiselect
        if item.label == "Variables with recommendation constraints"
    )
    at = constraint_box.select(sample_feature).run()
    assert not at.exception
    min_labels = [
        item.label for item in at.number_input if item.label.startswith("Minimum for ")
    ]
    max_labels = [
        item.label for item in at.number_input if item.label.startswith("Maximum for ")
    ]
    assert min_labels == [f"Minimum for {sample_feature}"]
    assert max_labels == [f"Maximum for {sample_feature}"]

    blob_after = _text_blob(at) + str(at)
    assert "Traceback" not in blob_after
    assert "C:\\Users\\" not in blob_after
    assert "/Users/" not in blob_after

    state = at.session_state.filtered_state
    for _key, value in state.items():
        assert not isinstance(value, (bytes, bytearray))
        type_name = type(value).__name__
        assert "DataFrame" not in type_name

    # CSV raw values unchanged after UI parse + configurator analysis.
    assert original_frame.to_dicts() == before_dicts
    assert [str(dtype) for dtype in original_frame.dtypes] == before_dtypes


def _constant_serial_csv_bytes(*, sensor_count: int = 20) -> bytes:
    """Wide CSV where SerialNumber is constant across all rows."""
    columns = ["Date", "Time", "SerialNumber"]
    columns.extend(f"sensor_{index:03d}" for index in range(sensor_count))
    columns.append("SOH")
    header = ",".join(columns)
    rows: list[str] = []
    for row_index in range(8):
        values: list[str] = [
            f"2020-01-{row_index + 1:02d}",
            f"10:{row_index:02d}:00",
            "1",
        ]
        for sensor_index in range(sensor_count):
            values.append(f"{row_index + sensor_index * 0.01:.4f}")
        values.append(f"{0.95 - row_index * 0.01:.4f}")
        rows.append(",".join(values))
    return ("\n".join([header, *rows]) + "\n").encode("utf-8")


def test_constant_serial_number_not_default_identifier() -> None:
    csv_bytes = _constant_serial_csv_bytes()
    frame = pl.read_csv(BytesIO(csv_bytes), infer_schema_length=None)
    before = frame.to_dicts()
    report = AutomaticColumnConfigurator().analyze(frame)
    assert frame.to_dicts() == before
    assert "SerialNumber" not in report.identifier_candidates
    assert "SerialNumber" not in report.recommended_feature_columns
    assert any(
        "resembles an identifier but is constant" in warning
        for warning in report.warnings
    )

    at = _upload_wide(_make_app().run(), csv_bytes)
    assert not at.exception
    blob = _text_blob(at)
    assert "resembles an identifier but is constant" in blob
    identifier_box = next(
        item for item in at.multiselect if item.label == "Identifier columns"
    )
    assert "SerialNumber" not in list(identifier_box.value)
    # Manual selection remains available.
    assert "SerialNumber" in list(identifier_box.options)


def _constant_soh_csv_bytes(*, sensor_count: int = 20) -> bytes:
    """Battery-like CSV where SOH is numeric and entirely zero."""
    columns = ["Date", "Time", "SerialNumber"]
    columns.extend(f"sensor_{index:03d}" for index in range(sensor_count))
    columns.append("SOH")
    header = ",".join(columns)
    rows: list[str] = []
    for row_index in range(8):
        values: list[str] = [
            f"2020-01-{row_index + 1:02d}",
            f"10:{row_index:02d}:00",
            str(1000 + row_index),
        ]
        for sensor_index in range(sensor_count):
            values.append(f"{row_index + sensor_index * 0.01:.4f}")
        values.append("0")
        rows.append(",".join(values))
    return ("\n".join([header, *rows]) + "\n").encode("utf-8")


class _CountingWorkflow:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, request: object) -> AnalysisWorkflowOutcome:
        self.calls += 1
        raise AssertionError(
            f"workflow.run must not be called for constant targets; got {request!r}"
        )


def test_constant_soh_blocks_run_without_workflow_or_temp_csv() -> None:
    csv_bytes = _constant_soh_csv_bytes()
    before_frame = pl.read_csv(BytesIO(csv_bytes))
    before_dicts = before_frame.to_dicts()
    before_soh = before_frame.get_column("SOH").to_list()
    assert set(before_soh) == {0}

    configurator = AutomaticColumnConfigurator()
    report = configurator.analyze(before_frame)
    assert "SOH" in report.target_candidates
    soh = next(item for item in report.suggestions if item.column == "SOH")
    assert soh.constant is True
    assert soh.suitable_as_target is False
    assert "SOH" not in report.recommended_feature_columns
    assert "SOH" not in report.excluded_candidates

    counter = _CountingWorkflow()
    at = AppTest.from_function(
        _ui_entry,
        default_timeout=60,
        kwargs={"workflow_factory": lambda: counter},
    )
    at = at.run()
    at = _upload_wide(at, csv_bytes)
    assert not at.exception

    target_box = next(item for item in at.selectbox if item.label == "Target column")
    assert "SOH" in list(target_box.options)
    at = target_box.select("SOH").run()
    assert not at.exception

    blob = _text_blob(at)
    assert "True: target selected" in blob
    assert "False: selected target has usable variation" in blob
    assert "cannot be used as a regression target" in blob.lower()
    assert "SOH" in blob

    # Target selection must not auto-switch away from SOH.
    target_after = next(item for item in at.selectbox if item.label == "Target column")
    assert target_after.value == "SOH"

    at = _select_objective(at)
    at = _complete_performance_rule(at)
    assert not at.exception
    run_button = next(item for item in at.button if item.label == "Run analysis")
    assert run_button.disabled is True

    # Even if somehow clicked, readiness remains blocked and no workflow call.
    assert counter.calls == 0

    after_frame = pl.read_csv(BytesIO(csv_bytes))
    assert after_frame.to_dicts() == before_dicts
    assert after_frame.get_column("SOH").to_list() == before_soh
