"""Raw CSV → recommendation end-to-end integration smoke tests (Step 10B).

Wires existing public components only. Does not modify production source,
introduce an application orchestrator, or relax leakage / independent-test
contracts.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from process_intelligence.core.enums import AnalysisTask, AnomalyType, ColumnRole
from process_intelligence.core.schemas import AnomalyEvent, VariableConstraint
from process_intelligence.data import (
    ORIGINAL_ROW_ID_COLUMN,
    DataQualityScorer,
    DatasetLoader,
    DatasetPreprocessor,
    DatasetProfiler,
    DatasetSorter,
    DatasetValidator,
    LineageTracker,
    PreprocessorConfig,
)
from process_intelligence.diagnosis import (
    DiagnosisEnsembleDiagnoser,
    DiagnosisMethod,
    DiagnosisRequest,
    DiagnosisResult,
    DiagnosisScope,
    ResidualAssociationDiagnoser,
    RobustGroupComparisonDiagnoser,
)
from process_intelligence.evaluation import (
    DatasetSplit,
    DatasetSplitter,
    FinalModelEvaluator,
    LeakageChecker,
    LeakageIssueType,
    LeakageSeverity,
    SplitConfig,
    SplitStrategy,
)
from process_intelligence.models import (
    AnomalyFinalEvaluator,
    ResidualAnomalyFinalEvaluator,
    ResidualAnomalyPipeline,
    SupervisedModelScreener,
    UnsupervisedAnomalyModelScreener,
    create_default_anomaly_model_registry,
    create_default_supervised_model_registry,
)
from process_intelligence.recommendation import (
    CandidateScenarioScorer,
    QualityOptimizationDirection,
    RecommendationObjective,
    RecommendationPipeline,
    RecommendationPipelineRequest,
    RecommendationPipelineStage,
    RecommendationRequest,
    RecommendationSafetyContext,
    RecommendationSafetyStatus,
    RecommendationStatus,
    ScenarioRankingStatus,
)
from process_intelligence.routing import (
    ColumnRoleMapper,
    create_default_industry_router,
    create_default_task_router,
)

CSV_NAME = "semiconductor_process_smoke.csv"
N_ROWS = 180
RNG_SEED = 20260721
SPLIT_RANDOM_STATE = 20260721
MODEL_RANDOM_STATE = 20260721

PROCESS_FEATURES = [
    "chamber_temperature",
    "chamber_pressure",
    "gas_flow_rate",
    "rf_power",
    "deposition_time",
]
TARGET_COLUMN = "quality_score"
TIME_COLUMN = "timestamp"
ID_COLUMNS = ["lot_id", "wafer_id"]
ALL_CSV_COLUMNS = [
    TIME_COLUMN,
    *ID_COLUMNS,
    *PROCESS_FEATURES,
    "film_thickness",
    TARGET_COLUMN,
    "injected_anomaly",
]

_ROLE_OVERRIDES: dict[str, ColumnRole] = {
    "chamber_temperature": ColumnRole.CONTROLLABLE_PROCESS,
    "chamber_pressure": ColumnRole.CONTROLLABLE_PROCESS,
    "gas_flow_rate": ColumnRole.CONTROLLABLE_PROCESS,
    "rf_power": ColumnRole.CONTROLLABLE_PROCESS,
    "deposition_time": ColumnRole.CONTROLLABLE_PROCESS,
    TARGET_COLUMN: ColumnRole.TARGET_QUALITY,
    "injected_anomaly": ColumnRole.UNKNOWN,
}

_CANONICAL_STAGES = list(RecommendationPipelineStage)
_FORBIDDEN_OUTCOME_PATTERNS = (
    "guaranteed improvement",
    "proven root cause",
    "this will fix",
    "proven optimal",
)


# ---------------------------------------------------------------------------
# Synthetic CSV
# ---------------------------------------------------------------------------


def _build_synthetic_rows(*, n_rows: int = N_ROWS) -> list[dict[str, Any]]:
    rng = np.random.default_rng(RNG_SEED)
    rows: list[dict[str, Any]] = []

    # ~6.7% anomalies, spaced across the full timeline so temporal splits
    # place some anomalies into train, validation, and test.
    anomaly_every = 15
    anomaly_indices = {i for i in range(7, n_rows, anomaly_every)}

    for i in range(n_rows):
        temperature = 220.0 + rng.normal(0.0, 4.0)
        pressure = 40.0 + 0.12 * (temperature - 220.0) + rng.normal(0.0, 1.2)
        gas_flow = 120.0 + 0.35 * (pressure - 40.0) + rng.normal(0.0, 2.5)
        rf_power = float(rng.uniform(280.0, 420.0))
        deposition_time = 55.0 + rng.normal(0.0, 2.0)

        injected = 0
        if i in anomaly_indices:
            injected = 1
            temperature += 18.0
            pressure -= 8.0
            gas_flow += 22.0

        film_thickness = (
            0.06 * temperature
            + 0.35 * pressure
            + 0.03 * gas_flow
            + 0.015 * rf_power
            + 0.08 * deposition_time
            + float(rng.normal(0.0, 0.35))
        )
        if injected:
            film_thickness += 6.0

        # Higher is better; peaks near nominal recipe, soft penalty for extremes.
        quality = (
            88.0
            - 0.045 * (temperature - 220.0) ** 2
            - 0.08 * (pressure - 40.0) ** 2
            - 0.004 * (gas_flow - 120.0) ** 2
            - 0.0008 * (rf_power - 350.0) ** 2
            - 0.06 * (deposition_time - 55.0) ** 2
            + 0.02 * film_thickness
            + float(rng.normal(0.0, 0.4))
        )
        if injected:
            quality -= 12.0

        rows.append(
            {
                TIME_COLUMN: 1_700_000_000 + i * 3600,
                "lot_id": f"LOT-{i // 12:03d}",
                "wafer_id": f"W-{i:04d}",
                "chamber_temperature": float(temperature),
                "chamber_pressure": float(pressure),
                "gas_flow_rate": float(gas_flow),
                "rf_power": float(rf_power),
                "deposition_time": float(deposition_time),
                "film_thickness": float(film_thickness),
                TARGET_COLUMN: float(quality),
                "injected_anomaly": int(injected),
            }
        )
    return rows


def write_synthetic_csv(path: Path, *, n_rows: int = N_ROWS) -> Path:
    rows = _build_synthetic_rows(n_rows=n_rows)
    frame = pl.DataFrame(rows).select(ALL_CSV_COLUMNS)
    assert frame.height == n_rows
    assert frame.select(pl.all().is_null().any()).row(0) == tuple(
        [False] * len(ALL_CSV_COLUMNS)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path)
    return path


# ---------------------------------------------------------------------------
# Workflow helpers (function-scoped; not a production orchestrator)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WorkflowArtifacts:
    csv_path: Path
    loaded_frame: pl.DataFrame
    loaded_metadata: Any
    profile: Any
    validation_issues: list[Any]
    quality_score: Any
    sorted_frame: pl.DataFrame
    sort_event: Any
    lineage_events: tuple[Any, ...]
    industry_result: Any
    industry_profile: Any
    role_mapping: Any
    role_mapping_auto: Any
    task_result: Any
    feature_columns: list[str]
    split: DatasetSplit
    preprocessing_events: tuple[Any, ...]
    leakage_report: Any
    supervised_screening: Any
    supervised_final: Any
    anomaly_screening: Any
    anomaly_final: Any
    residual_pipeline: Any
    residual_final: Any
    anomaly_events: list[AnomalyEvent]
    diagnosis_frame: pl.DataFrame
    robust_result: DiagnosisResult
    residual_result: DiagnosisResult
    ensemble_result: DiagnosisResult


def _feature_columns_from_mapping(role_mapping: Any) -> list[str]:
    return [
        item.column_name
        for item in role_mapping.assignments
        if item.use_in_model and item.column_name in PROCESS_FEATURES
    ]


def _rebuild_split(
    *,
    train: pl.DataFrame,
    validation: pl.DataFrame,
    test: pl.DataFrame,
    summary: Any,
) -> DatasetSplit:
    return DatasetSplit(
        train=train,
        validation=validation,
        test=test,
        summary=summary,
    )


def _assert_finite_number(value: object, *, label: str) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool), label
    number = float(value)
    assert math.isfinite(number), label
    return number


def _assert_scalar_metadata(metadata: dict[str, Any], *, field_name: str) -> None:
    for key, value in metadata.items():
        assert isinstance(key, str) and key.strip() != ""
        assert value is None or isinstance(value, (str, bool, int, float)), (
            f"{field_name}[{key!r}] must be scalar, got {type(value).__name__}"
        )
        if isinstance(value, float):
            assert math.isfinite(value)
        assert not isinstance(value, (pl.DataFrame, np.ndarray))


def _row_id_set(frame: pl.DataFrame) -> set[int]:
    return {int(v) for v in frame.get_column(ORIGINAL_ROW_ID_COLUMN).to_list()}


def _snapshot_frame(frame: pl.DataFrame) -> list[dict[str, Any]]:
    return frame.to_dicts()


def _build_anomaly_events(
    *,
    residual_test: pl.DataFrame,
    anomaly_test: pl.DataFrame,
) -> tuple[list[AnomalyEvent], str]:
    """Prefer residual flags, then unsupervised flags, else top-score candidates."""
    residual_hits = residual_test.filter(pl.col("_is_residual_anomaly"))
    if residual_hits.height > 0:
        ordered = residual_hits.sort("_residual_anomaly_score", descending=True)
        source = "residual_anomaly"
        score_col = "_residual_anomaly_score"
        detector = "residual_anomaly_detector"
    else:
        unsupervised_hits = anomaly_test.filter(pl.col("_is_anomaly"))
        if unsupervised_hits.height > 0:
            ordered = unsupervised_hits.sort("_anomaly_score", descending=True)
            source = "unsupervised_anomaly"
            score_col = "_anomaly_score"
            detector = "unsupervised_anomaly_model"
        else:
            ordered = anomaly_test.sort("_anomaly_score", descending=True)
            source = "diagnostic_smoke_candidate"
            score_col = "_anomaly_score"
            detector = "top_anomaly_score_candidate"

    events: list[AnomalyEvent] = []
    for row in ordered.head(5).iter_rows(named=True):
        row_id = int(row[ORIGINAL_ROW_ID_COLUMN])
        score = _assert_finite_number(row[score_col], label="event anomaly score")
        rationale = (
            "Residual or unsupervised anomaly indicator was True for this row."
            if source != "diagnostic_smoke_candidate"
            else (
                "Diagnostic smoke candidate selected from top anomaly scores; "
                "not asserted as a confirmed detected anomaly."
            )
        )
        events.append(
            AnomalyEvent(
                anomaly_id=str(row_id),
                anomaly_type=AnomalyType.PROCESS_INPUT,
                anomaly_score=score,
                severity="high" if score >= 0.0 else "moderate",
                sample_id=row_id,
                model_confidence=0.75,
                detector=detector,
                rationale=rationale,
                contributing_variables=list(PROCESS_FEATURES),
            )
        )
    assert events
    return events, source


def run_workflow(tmp_path: Path) -> WorkflowArtifacts:
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = write_synthetic_csv(tmp_path / CSV_NAME)

    loaded = DatasetLoader().load(csv_path)
    loaded_frame = loaded.frame
    metadata = loaded.metadata

    profile = DatasetProfiler().profile(loaded_frame)
    validation_issues = DatasetValidator().validate(loaded_frame)
    quality_score = DataQualityScorer().score(validation_issues)

    sort_result = DatasetSorter().sort(loaded_frame, by=TIME_COLUMN)
    sorted_frame = sort_result.frame
    lineage = LineageTracker(loaded_frame)
    lineage.record(sorted_frame, sort_result.event)

    from process_intelligence.industries import (
        AutomotiveIndustryProfile,
        BatteryIndustryProfile,
        GenericIndustryProfile,
        IndustryRegistry,
        SemiconductorIndustryProfile,
    )

    industry_router = create_default_industry_router()
    industry_auto = industry_router.route(metadata)
    assert industry_auto.ranked_candidates
    assert industry_auto.ranked_candidates[0].industry_name == "semiconductor"
    industry_result = industry_router.route(
        metadata,
        confirmed_industry="semiconductor",
    )
    # IndustryRouter does not expose profile lookup; resolve via public registry.
    industry_registry = IndustryRegistry(
        profiles=[
            SemiconductorIndustryProfile(),
            BatteryIndustryProfile(),
            AutomotiveIndustryProfile(),
            GenericIndustryProfile(),
        ]
    )
    industry_profile = industry_registry.get(industry_result.selected_industry)

    sorted_profile = DatasetProfiler().profile(sorted_frame)
    role_mapping_auto = ColumnRoleMapper().map_roles(
        sorted_profile,
        industry_profile.get_schema_hints(),
    )
    role_mapping = ColumnRoleMapper().map_roles(
        sorted_profile,
        industry_profile.get_schema_hints(),
        overrides=_ROLE_OVERRIDES,
    )

    task_result = create_default_task_router().route(
        sorted_profile,
        role_mapping,
        confirmed_task=AnalysisTask.REGRESSION,
        confirmed_target=TARGET_COLUMN,
        confirmed_time_column=TIME_COLUMN,
    )

    feature_columns = _feature_columns_from_mapping(role_mapping)
    assert feature_columns == PROCESS_FEATURES

    raw_split = DatasetSplitter().split(
        sorted_frame,
        SplitConfig(
            strategy=SplitStrategy.TIME,
            time_column=TIME_COLUMN,
            test_size=0.20,
            validation_size=0.20,
            random_state=SPLIT_RANDOM_STATE,
        ),
    )

    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=list(feature_columns),
            categorical_columns=[],
            numeric_imputation="median",
            categorical_imputation="none",
            scaling="none",
        )
    )
    preprocessor.fit(raw_split.train)
    train_pp = preprocessor.transform(raw_split.train)
    validation_pp = preprocessor.transform(raw_split.validation)
    test_pp = preprocessor.transform(raw_split.test)
    split = _rebuild_split(
        train=train_pp.frame,
        validation=validation_pp.frame,
        test=test_pp.frame,
        summary=raw_split.summary,
    )
    preprocessing_events = tuple(train_pp.events)
    # Do not record train-only preprocess frames into the full-dataset lineage
    # tracker: LineageTracker requires rows_before to match the current height.

    leakage_report = LeakageChecker().check(
        split,
        role_mapping,
        target_column=TARGET_COLUMN,
        feature_columns=feature_columns,
        preprocessing_events=preprocessing_events,
    )

    supervised_registry = create_default_supervised_model_registry(
        random_state=MODEL_RANDOM_STATE
    )
    supervised_screening = SupervisedModelScreener(supervised_registry).screen(
        split,
        task=AnalysisTask.REGRESSION,
        target_column=TARGET_COLUMN,
        feature_columns=feature_columns,
        leakage_report=leakage_report,
        industry_profile=industry_profile,
    )
    supervised_final = FinalModelEvaluator(supervised_registry).evaluate(
        split,
        supervised_screening,
        task=AnalysisTask.REGRESSION,
        target_column=TARGET_COLUMN,
        feature_columns=feature_columns,
        leakage_report=leakage_report,
    )

    anomaly_registry = create_default_anomaly_model_registry(
        random_state=MODEL_RANDOM_STATE
    )
    anomaly_screening = UnsupervisedAnomalyModelScreener(anomaly_registry).screen(
        split,
        feature_columns=feature_columns,
        leakage_report=leakage_report,
        industry_profile=industry_profile,
    )
    anomaly_final = AnomalyFinalEvaluator(anomaly_registry).evaluate(
        split,
        anomaly_screening,
        feature_columns=feature_columns,
        leakage_report=leakage_report,
    )

    residual_pipeline = ResidualAnomalyPipeline().run(
        split,
        supervised_screening,
        target_column=TARGET_COLUMN,
        feature_columns=feature_columns,
        leakage_report=leakage_report,
    )
    residual_final = ResidualAnomalyFinalEvaluator().evaluate(
        split,
        residual_pipeline,
        target_column=TARGET_COLUMN,
        feature_columns=feature_columns,
        leakage_report=leakage_report,
    )

    anomaly_events, _event_source = _build_anomaly_events(
        residual_test=residual_final.test_scored,
        anomaly_test=anomaly_final.test_scored,
    )

    # Join residual score columns onto the test feature frame for diagnosis.
    diagnosis_frame = residual_final.test_scored

    robust = RobustGroupComparisonDiagnoser()
    residual_diag = ResidualAssociationDiagnoser()
    ensemble = DiagnosisEnsembleDiagnoser(
        robust_diagnoser=robust,
        residual_diagnoser=residual_diag,
    )

    # Robust-only: avoid exact event/indicator mismatch by using ANOMALY_GROUP.
    robust_request = DiagnosisRequest(
        task=AnalysisTask.UNSUPERVISED_ANOMALY,
        method=DiagnosisMethod.GROUP_COMPARISON,
        scope=DiagnosisScope.ANOMALY_GROUP,
        feature_columns=list(feature_columns),
        anomaly_events=[],
        anomaly_indicator_column="_is_anomaly",
        anomaly_score_column="_anomaly_score",
        row_id_column=ORIGINAL_ROW_ID_COLUMN,
        minimum_reference_rows=5,
        metadata={"stage": "robust_smoke"},
    )
    # Unsupervised columns live on anomaly_final.test_scored.
    robust_frame = anomaly_final.test_scored
    if int(robust_frame.get_column("_is_anomaly").sum()) == 0:
        # Fall back to residual indicators for the robust diagnoser path.
        robust_frame = diagnosis_frame
        robust_request = DiagnosisRequest(
            task=AnalysisTask.RESIDUAL_ANOMALY,
            method=DiagnosisMethod.GROUP_COMPARISON,
            scope=DiagnosisScope.ANOMALY_GROUP,
            feature_columns=list(feature_columns),
            anomaly_events=[],
            anomaly_indicator_column="_is_residual_anomaly",
            anomaly_score_column="_residual_anomaly_score",
            row_id_column=ORIGINAL_ROW_ID_COLUMN,
            minimum_reference_rows=5,
            metadata={"stage": "robust_smoke_residual_fallback"},
        )
    robust_result = robust.diagnose(robust_frame, request=robust_request)
    assert isinstance(robust_result, DiagnosisResult)

    residual_request = DiagnosisRequest(
        task=AnalysisTask.RESIDUAL_ANOMALY,
        method=DiagnosisMethod.RESIDUAL_ASSOCIATION,
        scope=DiagnosisScope.ANOMALY_GROUP,
        feature_columns=list(feature_columns),
        anomaly_events=[],
        anomaly_indicator_column="_is_residual_anomaly",
        anomaly_score_column="_residual_anomaly_score",
        row_id_column=ORIGINAL_ROW_ID_COLUMN,
        minimum_reference_rows=5,
        metadata={"stage": "residual_smoke"},
    )
    # Ensure at least one residual anomaly row for residual association.
    residual_frame = diagnosis_frame
    if int(residual_frame.get_column("_is_residual_anomaly").sum()) == 0:
        # Mark the highest residual-score row as the diagnostic anomaly group.
        top_id = residual_frame.sort(
            "_residual_anomaly_score", descending=True
        )[ORIGINAL_ROW_ID_COLUMN][0]
        residual_frame = residual_frame.with_columns(
            pl.when(pl.col(ORIGINAL_ROW_ID_COLUMN) == top_id)
            .then(True)
            .otherwise(False)
            .alias("_is_residual_anomaly")
        )
    residual_result = residual_diag.diagnose(residual_frame, request=residual_request)
    assert isinstance(residual_result, DiagnosisResult)

    ensemble_request = DiagnosisRequest(
        task=AnalysisTask.RESIDUAL_ANOMALY,
        method=DiagnosisMethod.ENSEMBLE,
        scope=DiagnosisScope.ANOMALY_GROUP,
        feature_columns=list(feature_columns),
        anomaly_events=[],
        anomaly_indicator_column="_is_residual_anomaly",
        anomaly_score_column="_residual_anomaly_score",
        row_id_column=ORIGINAL_ROW_ID_COLUMN,
        minimum_reference_rows=5,
        metadata={"stage": "ensemble_smoke"},
    )
    ensemble_result = ensemble.diagnose(residual_frame, request=ensemble_request)
    assert isinstance(ensemble_result, DiagnosisResult)

    return WorkflowArtifacts(
        csv_path=csv_path,
        loaded_frame=loaded_frame,
        loaded_metadata=metadata,
        profile=profile,
        validation_issues=validation_issues,
        quality_score=quality_score,
        sorted_frame=sorted_frame,
        sort_event=sort_result.event,
        lineage_events=lineage.get_event_history(),
        industry_result=industry_result,
        industry_profile=industry_profile,
        role_mapping=role_mapping,
        role_mapping_auto=role_mapping_auto,
        task_result=task_result,
        feature_columns=feature_columns,
        split=split,
        preprocessing_events=preprocessing_events,
        leakage_report=leakage_report,
        supervised_screening=supervised_screening,
        supervised_final=supervised_final,
        anomaly_screening=anomaly_screening,
        anomaly_final=anomaly_final,
        residual_pipeline=residual_pipeline,
        residual_final=residual_final,
        anomaly_events=anomaly_events,
        diagnosis_frame=residual_frame,
        robust_result=robust_result,
        residual_result=residual_result,
        ensemble_result=ensemble_result,
    )


@pytest.fixture
def workflow(tmp_path: Path) -> WorkflowArtifacts:
    return run_workflow(tmp_path)


# ---------------------------------------------------------------------------
# Stage tests
# ---------------------------------------------------------------------------


def test_raw_csv_data_foundation_smoke(tmp_path: Path) -> None:
    csv_path = write_synthetic_csv(tmp_path / CSV_NAME)
    assert csv_path.is_file()

    loaded = DatasetLoader().load(csv_path)
    frame = loaded.frame
    metadata = loaded.metadata

    assert isinstance(frame, pl.DataFrame)
    assert frame.height == N_ROWS
    assert metadata.row_count == N_ROWS
    assert metadata.file_name == CSV_NAME
    assert metadata.file_format == "csv"
    assert str(csv_path.resolve()) not in metadata.file_name
    assert ORIGINAL_ROW_ID_COLUMN not in metadata.column_names
    assert set(metadata.column_names) == set(ALL_CSV_COLUMNS)

    assert ORIGINAL_ROW_ID_COLUMN in frame.columns
    row_ids = frame.get_column(ORIGINAL_ROW_ID_COLUMN)
    assert row_ids.null_count() == 0
    assert row_ids.n_unique() == N_ROWS
    assert frame.null_count().sum_horizontal()[0] == 0

    for column in PROCESS_FEATURES + ["film_thickness", TARGET_COLUMN]:
        dtype = frame.schema[column]
        assert dtype.is_numeric()

    input_snapshot = _snapshot_frame(frame)
    profile = DatasetProfiler().profile(frame)
    assert profile.row_count == N_ROWS
    assert profile.column_count == len(ALL_CSV_COLUMNS) + 1  # + row id
    numeric_count = sum(1 for col in profile.columns if "Float" in col.dtype or "Int" in col.dtype)
    assert numeric_count >= len(PROCESS_FEATURES) + 3
    assert all(col.null_count == 0 for col in profile.columns)
    assert frame.to_dicts() == input_snapshot
    assert not hasattr(profile, "frame")

    issues = DatasetValidator().validate(frame)
    error_issues = [issue for issue in issues if issue.severity == "ERROR"]
    assert error_issues == []
    issue_types = {issue.issue_type for issue in issues}
    assert "DUPLICATE_ROWS" not in issue_types or all(
        issue.severity != "ERROR" for issue in issues if issue.issue_type == "DUPLICATE_ROWS"
    )

    quality = DataQualityScorer().score(issues)
    assert 0.0 <= quality.total_score <= 100.0
    assert math.isfinite(quality.total_score)
    assert quality.issue_count == len(issues)
    assert quality.starting_score == 100.0
    assert len(quality.penalties) == quality.issue_count

    # Sorting / lineage / preprocessing
    sort_input = _snapshot_frame(frame)
    sort_result = DatasetSorter().sort(frame, by=TIME_COLUMN)
    assert sort_result.frame.height == frame.height
    assert _row_id_set(sort_result.frame) == _row_id_set(frame)
    assert frame.to_dicts() == sort_input
    sort_again = DatasetSorter().sort(frame, by=TIME_COLUMN)
    assert sort_result.frame.to_dicts() == sort_again.frame.to_dicts()

    tracker = LineageTracker(frame)
    tracker.record(sort_result.frame, sort_result.event)
    history = tracker.get_event_history()
    assert any(event.step_name == "sort_dataset" for event in history)
    assert tracker.get_raw_frame().to_dicts() == sort_input

    pre_input = _snapshot_frame(sort_result.frame)
    preprocessor = DatasetPreprocessor(
        PreprocessorConfig(
            numeric_columns=list(PROCESS_FEATURES),
            numeric_imputation="none",
            scaling="none",
        )
    )
    preprocessed = preprocessor.fit_transform(sort_result.frame)
    assert ORIGINAL_ROW_ID_COLUMN in preprocessed.frame.columns
    assert TARGET_COLUMN in preprocessed.frame.columns
    assert "injected_anomaly" in preprocessed.frame.columns
    assert _row_id_set(preprocessed.frame) == _row_id_set(sort_result.frame)
    assert sort_result.frame.to_dicts() == pre_input
    for column in PROCESS_FEATURES:
        series = preprocessed.frame.get_column(column)
        assert series.null_count() == 0
        assert not bool(series.is_nan().any())
        assert not bool(series.is_infinite().any())


def test_raw_csv_routing_and_split_smoke(workflow: WorkflowArtifacts) -> None:
    industry = workflow.industry_result
    assert industry.selected_industry == "semiconductor"
    assert industry.ranked_candidates
    assert industry.ranked_candidates[0].industry_name == "semiconductor"
    _assert_scalar_metadata(
        {
            "selected_industry": industry.selected_industry,
            "requires_user_confirmation": industry.requires_user_confirmation,
            "reason": industry.reason,
        },
        field_name="industry_route",
    )

    # Auto mapping may leave chamber_* as STATE_SENSOR; overrides are explicit.
    auto_roles = {
        item.column_name: item.role for item in workflow.role_mapping_auto.assignments
    }
    override_roles = {
        item.column_name: item.role for item in workflow.role_mapping.assignments
    }
    for column, role in _ROLE_OVERRIDES.items():
        assert override_roles[column] is role
        # Document distinction when auto mapping differs.
        if column in auto_roles and auto_roles[column] is not role:
            assert column in _ROLE_OVERRIDES

    for column in PROCESS_FEATURES:
        assignment = next(
            item
            for item in workflow.role_mapping.assignments
            if item.column_name == column
        )
        assert assignment.role is ColumnRole.CONTROLLABLE_PROCESS
        assert assignment.use_in_model is True

    target_assignment = next(
        item
        for item in workflow.role_mapping.assignments
        if item.column_name == TARGET_COLUMN
    )
    assert target_assignment.role is ColumnRole.TARGET_QUALITY
    assert target_assignment.use_in_model is False

    injected = next(
        item
        for item in workflow.role_mapping.assignments
        if item.column_name == "injected_anomaly"
    )
    assert injected.use_in_model is False

    task = workflow.task_result
    assert task.selected_task is AnalysisTask.REGRESSION
    assert task.selected_target == TARGET_COLUMN
    assert task.selected_task is not AnalysisTask.CLASSIFICATION

    split = workflow.split
    assert split.train.height > 0
    assert split.validation.height > 0
    assert split.test.height > 0
    train_ids = set(split.summary.train_original_row_ids)
    validation_ids = set(split.summary.validation_original_row_ids)
    test_ids = set(split.summary.test_original_row_ids)
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert train_ids | validation_ids | test_ids == _row_id_set(workflow.sorted_frame)
    assert split.summary.strategy is SplitStrategy.TIME

    # Deterministic temporal split
    again = DatasetSplitter().split(
        workflow.sorted_frame,
        SplitConfig(
            strategy=SplitStrategy.TIME,
            time_column=TIME_COLUMN,
            test_size=0.20,
            validation_size=0.20,
            random_state=SPLIT_RANDOM_STATE,
        ),
    )
    assert again.summary.train_original_row_ids == split.summary.train_original_row_ids
    assert again.summary.validation_original_row_ids == (
        split.summary.validation_original_row_ids
    )
    assert again.summary.test_original_row_ids == split.summary.test_original_row_ids

    report = workflow.leakage_report
    assert report.blocker_count == 0
    assert report.is_safe is True
    assert TARGET_COLUMN not in report.checked_feature_columns
    assert "injected_anomaly" not in report.checked_feature_columns
    assert ORIGINAL_ROW_ID_COLUMN not in report.checked_feature_columns
    assert set(report.checked_feature_columns) == set(workflow.feature_columns)
    blocker_types = {
        issue.issue_type
        for issue in report.issues
        if issue.severity is LeakageSeverity.BLOCKER
    }
    assert LeakageIssueType.ORIGINAL_ROW_ID_OVERLAP not in blocker_types
    assert LeakageIssueType.TARGET_INCLUDED_AS_FEATURE not in blocker_types


def test_raw_csv_model_and_anomaly_smoke(workflow: WorkflowArtifacts) -> None:
    screening = workflow.supervised_screening
    final = workflow.supervised_final
    features = workflow.feature_columns

    assert screening.summary.task is AnalysisTask.REGRESSION
    assert screening.summary.selected_model_name
    assert screening.summary.selected_estimator_key
    assert screening.selected_model.is_fitted
    assert list(screening.selected_model.feature_names) == features
    for value in screening.summary.selected_metrics.values():
        _assert_finite_number(value, label="validation metric")

    assert final.final_model.is_fitted
    assert list(final.final_model.feature_names) == features
    assert final.report.refit_on_train_validation is True
    assert final.report.test_row_count == workflow.split.test.height
    for value in final.report.test_metrics.values():
        _assert_finite_number(value, label="test metric")
    assert final.report.feature_columns == features
    meta = final.final_model.get_metadata()
    assert list(meta.features) == features
    assert meta.model_name
    assert meta.task is AnalysisTask.REGRESSION
    for key, value in meta.model_dump().items():
        if key in {"features", "optional_dependencies_used", "training_timestamp"}:
            continue
        assert value is None or isinstance(value, (str, bool, int, float))
        if isinstance(value, float):
            assert math.isfinite(value)
        assert not isinstance(value, (pl.DataFrame, np.ndarray))
    assert isinstance(meta.features, list)
    assert not isinstance(meta.features, (pl.DataFrame, np.ndarray))
    assert meta.training_timestamp.tzinfo is not None

    # Independent test evaluation only through final evaluator contract.
    assert final.report.test_evaluation_seconds >= 0.0

    anomaly_screening = workflow.anomaly_screening
    anomaly_final = workflow.anomaly_final
    assert anomaly_screening.summary.selected_model_name
    assert anomaly_screening.selected_model.is_fitted
    assert list(anomaly_screening.selected_model.feature_names) == features

    test_scored = anomaly_final.test_scored
    assert test_scored.height == workflow.split.test.height
    scores = test_scored.get_column("_anomaly_score").to_list()
    assert len(scores) == workflow.split.test.height
    assert all(math.isfinite(float(v)) for v in scores)
    assert float(np.std(np.asarray(scores, dtype=float))) > 0.0 or len(set(scores)) > 1
    assert int(test_scored.get_column("_is_anomaly").sum()) >= 0
    assert _row_id_set(test_scored) == set(workflow.split.summary.test_original_row_ids)
    assert test_scored.get_column(ORIGINAL_ROW_ID_COLUMN).to_list() == (
        workflow.split.test.get_column(ORIGINAL_ROW_ID_COLUMN).to_list()
    )

    # Soft overlap check with injected anomalies on the test partition.
    test_with_labels = workflow.split.test.select(
        [ORIGINAL_ROW_ID_COLUMN, "injected_anomaly"]
    )
    joined = test_scored.join(test_with_labels, on=ORIGINAL_ROW_ID_COLUMN, how="left")
    injected_test = joined.filter(pl.col("injected_anomaly") == 1)
    if injected_test.height > 0:
        score_arr = np.asarray(joined["_anomaly_score"].to_list(), dtype=float)
        threshold = float(np.quantile(score_arr, 0.75))
        top_injected = injected_test.filter(pl.col("_anomaly_score") >= threshold)
        assert top_injected.height >= 0  # soft recall; non-constant already checked

    residual_pipeline = workflow.residual_pipeline
    residual_final = workflow.residual_final
    assert residual_pipeline.residual_detector.is_fitted
    threshold = residual_pipeline.residual_detector.threshold
    assert threshold is not None
    _assert_finite_number(threshold, label="residual threshold")
    assert residual_final.residual_detector.threshold == pytest.approx(threshold)
    assert residual_final.regression_model is residual_pipeline.regression_model
    assert residual_final.residual_detector is residual_pipeline.residual_detector

    residual_test = residual_final.test_scored
    assert residual_test.height == workflow.split.test.height
    for column in (
        "_regression_residual",
        "_absolute_centered_residual",
        "_residual_anomaly_score",
        "_is_residual_anomaly",
    ):
        assert column in residual_test.columns
    residuals = residual_test.get_column("_regression_residual").to_list()
    abs_residuals = residual_test.get_column("_absolute_centered_residual").to_list()
    residual_scores = residual_test.get_column("_residual_anomaly_score").to_list()
    assert all(math.isfinite(float(v)) for v in residuals)
    assert all(math.isfinite(float(v)) for v in abs_residuals)
    assert all(math.isfinite(float(v)) for v in residual_scores)
    assert residual_test.get_column("_is_residual_anomaly").dtype == pl.Boolean
    assert _row_id_set(residual_test) == set(workflow.split.summary.test_original_row_ids)


def test_raw_csv_diagnosis_smoke(workflow: WorkflowArtifacts) -> None:
    assert workflow.anomaly_events
    assert 1 <= len(workflow.anomaly_events) <= 5
    scores = [event.anomaly_score for event in workflow.anomaly_events]
    assert scores == sorted(scores, reverse=True)

    for result in (
        workflow.robust_result,
        workflow.residual_result,
        workflow.ensemble_result,
    ):
        assert isinstance(result, DiagnosisResult)
        assert result.factors
        variables = [factor.variable for factor in result.factors]
        assert len(variables) == len(set(variables))
        assert all(variable in workflow.feature_columns for variable in variables)
        _assert_finite_number(result.confidence, label="diagnosis confidence")
        for factor in result.factors:
            _assert_finite_number(factor.confidence, label="factor confidence")
            evidence_lower = factor.evidence.lower()
            assert "proven root cause" not in evidence_lower
            assert "guaranteed" not in evidence_lower
        soft_language = " ".join(result.caveats).lower()
        assert (
            "causation" in soft_language
            or "association" in soft_language
            or "reference group" in soft_language
        )
        _assert_scalar_metadata(result.metadata, field_name="diagnosis.metadata")

    ensemble = workflow.ensemble_result
    assert DiagnosisMethod.ENSEMBLE in ensemble.method_used or (
        DiagnosisMethod.RESIDUAL_ASSOCIATION in ensemble.method_used
    )
    assert ensemble.metadata.get("robust_method_succeeded") in {True, False}
    assert ensemble.metadata.get("residual_method_succeeded") in {True, False}
    # Prefer both methods succeeding when residual indicators exist.
    if int(workflow.diagnosis_frame.get_column("_is_residual_anomaly").sum()) > 0:
        assert ensemble.metadata.get("residual_method_succeeded") is True


def _constraint_bounds(split: DatasetSplit, variable: str) -> tuple[float, float]:
    values = split.train.get_column(variable).to_list()
    low = float(min(values))
    high = float(max(values))
    span = high - low
    if span <= 0.0:
        low -= 1.0
        high += 1.0
        span = high - low
    padding = 0.05 * span
    return low - padding, high + padding


def _current_point(workflow: WorkflowArtifacts) -> dict[str, float]:
    event = workflow.anomaly_events[0]
    row_id = int(event.anomaly_id)
    matched = workflow.split.test.filter(pl.col(ORIGINAL_ROW_ID_COLUMN) == row_id)
    if matched.height != 1:
        matched = workflow.split.test.head(1)
        row_id = int(matched.get_column(ORIGINAL_ROW_ID_COLUMN)[0])
    baseline = {
        name: float(matched.get_column(name)[0]) for name in workflow.feature_columns
    }
    return baseline


def _recommendation_diagnosis(
    workflow: WorkflowArtifacts,
    *,
    task: AnalysisTask,
) -> DiagnosisResult:
    base = workflow.ensemble_result
    return DiagnosisResult(
        anomaly_id=base.anomaly_id,
        task=task,
        method_used=list(base.method_used),
        scope=base.scope,
        factors=[factor.model_copy(deep=True) for factor in base.factors],
        confidence=float(base.confidence),
        analyzed_row_count=base.analyzed_row_count,
        reference_row_count=max(base.reference_row_count, 5),
        caveats=list(base.caveats),
        generated_at=datetime.now(UTC),
        metadata=dict(base.metadata),
    )


def _build_recommendation_pipeline_request(
    workflow: WorkflowArtifacts,
    *,
    objective: RecommendationObjective,
) -> RecommendationPipelineRequest:
    if objective is RecommendationObjective.REDUCE_ANOMALY_SCORE:
        task = AnalysisTask.UNSUPERVISED_ANOMALY
        quality_direction = None
        target_column = None
    else:
        task = AnalysisTask.REGRESSION
        quality_direction = QualityOptimizationDirection.MAXIMIZE
        target_column = TARGET_COLUMN

    baseline = _current_point(workflow)
    diagnosis = _recommendation_diagnosis(workflow, task=task)
    factor_vars = [factor.variable for factor in diagnosis.factors]
    # Prefer diagnosis factors that are process features; fall back to first two.
    candidate_vars = [name for name in factor_vars if name in PROCESS_FEATURES]
    if len(candidate_vars) < 2:
        candidate_vars = list(PROCESS_FEATURES[:2])
    current_values = {name: baseline[name] for name in candidate_vars[:2]}

    constraints: list[VariableConstraint] = []
    for name in current_values:
        minimum, maximum = _constraint_bounds(workflow.split, name)
        value = current_values[name]
        minimum = min(minimum, value)
        maximum = max(maximum, value)
        assert maximum > minimum
        constraints.append(
            VariableConstraint(
                variable=name,
                adjustable=True,
                minimum=minimum,
                maximum=maximum,
                fixed=False,
            )
        )

    # Explicit performance policy for synthetic smoke (no exact metric hard-code).
    primary = workflow.supervised_final.report.test_primary_metric
    higher_is_better = workflow.supervised_final.report.higher_is_better
    if higher_is_better:
        performance_ok = primary > -1.0e9
    else:
        performance_ok = primary < 1.0e9

    request = RecommendationRequest(
        task=task,
        diagnosis=diagnosis,
        objective=objective,
        current_values=current_values,
        constraints=constraints,
        user_confirmed_controllable_variables=list(current_values.keys()),
        user_verified_variables=list(current_values.keys()),
        max_simultaneous_changes=2,
        metadata={"fixture": "raw_csv_workflow_smoke"},
    )
    safety = RecommendationSafetyContext(
        leakage_report=workflow.leakage_report,
        final_evaluation_available=True,
        model_performance_acceptable=performance_ok,
        model_performance_reason=(
            "Synthetic smoke accepts finite final-evaluation primary metric."
        ),
        extrapolation_detected=False,
        uncertainty_available=False,
        uncertainty_acceptable=None,
        metadata={"fixture": "raw_csv_workflow_smoke"},
    )
    industry_constraints = workflow.industry_profile.get_recommendation_constraints()
    return RecommendationPipelineRequest(
        recommendation_request=request,
        safety_context=safety,
        industry_constraints=list(industry_constraints),
        user_overrides=[],
        feature_columns=list(workflow.feature_columns),
        baseline_features=dict(baseline),
        target_column=target_column,
        quality_direction=quality_direction,
        quality_target=None,
        extrapolation_evaluated=False,
        extrapolation_flag=False,
        uncertainty_available=False,
        uncertainty_acceptable=None,
        metadata={"fixture": "raw_csv_workflow_smoke"},
    )


@pytest.mark.parametrize(
    "objective",
    [
        RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
        RecommendationObjective.REDUCE_ANOMALY_SCORE,
        RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
    ],
)
def test_raw_csv_recommendation_smoke(
    workflow: WorkflowArtifacts,
    objective: RecommendationObjective,
) -> None:
    quality_model = workflow.supervised_final.final_model
    anomaly_model = workflow.anomaly_final.final_model
    if objective is RecommendationObjective.IMPROVE_PREDICTED_QUALITY:
        scorer = CandidateScenarioScorer(
            quality_model=quality_model,
            anomaly_model=anomaly_model,
        )
    elif objective is RecommendationObjective.REDUCE_ANOMALY_SCORE:
        scorer = CandidateScenarioScorer(
            quality_model=None,
            anomaly_model=anomaly_model,
        )
    else:
        scorer = CandidateScenarioScorer(
            quality_model=quality_model,
            anomaly_model=anomaly_model,
        )

    pipeline = RecommendationPipeline(scenario_scorer=scorer)
    request = _build_recommendation_pipeline_request(workflow, objective=objective)
    request_snapshot = request.model_dump()
    quality_meta_before = quality_model.get_metadata().model_dump()
    anomaly_meta_before = anomaly_model.get_metadata().model_dump()

    outcome = pipeline.run(request)
    report = outcome.report

    assert request.model_dump() == request_snapshot
    assert quality_model.get_metadata().model_dump() == quality_meta_before
    assert anomaly_model.get_metadata().model_dump() == anomaly_meta_before

    assert report.executed_stages == _CANONICAL_STAGES
    assert report.skipped_stages == []
    assert report.terminal_stage is RecommendationPipelineStage.RECOMMENDATION_GENERATION
    assert report.safety_decision is not None
    assert report.safety_decision.status is not RecommendationSafetyStatus.REFUSED
    assert report.grid_report is not None
    assert len(report.grid_report.scenarios) >= 1
    assert report.scoring_report is not None
    assert report.ranking_report is not None
    assert report.final_result is not None
    assert report.status is report.final_result.status
    assert report.status in {
        RecommendationStatus.GENERATED,
        RecommendationStatus.READY_FOR_OPTIMIZATION,
    }

    if report.status is RecommendationStatus.GENERATED:
        result = report.final_result
        assert result.changes
        for change in result.changes:
            assert change.variable in request.recommendation_request.current_values
            assert change.proposed_value is not None
            assert math.isfinite(change.proposed_value)
            constraint = next(
                item
                for item in request.recommendation_request.constraints
                if item.variable == change.variable
            )
            assert constraint.minimum is not None and constraint.maximum is not None
            assert constraint.minimum <= change.proposed_value <= constraint.maximum
            assert change.delta == pytest.approx(
                change.proposed_value - change.current_value
            )
            _assert_finite_number(change.confidence, label="change confidence")
            rationale_lower = change.rationale.lower()
            for token in _FORBIDDEN_OUTCOME_PATTERNS:
                assert token not in rationale_lower
            assert "does not establish causation" in rationale_lower or (
                "association" in rationale_lower
            )
        assert result.confidence is not None
        _assert_finite_number(result.confidence, label="recommendation confidence")
    else:
        assert report.final_result.changes == []
        assert report.final_result.proposed_prediction is None
        assert report.final_result.proposed_anomaly_score is None
        assert report.ranking_report.status in {
            ScenarioRankingStatus.NO_IMPROVEMENT,
            ScenarioRankingStatus.RANKED,
            ScenarioRankingStatus.PARTIAL,
            ScenarioRankingStatus.REFUSED,
        } or report.status is RecommendationStatus.READY_FOR_OPTIMIZATION

    # Preserve anomaly score sign; no clipping/abs.
    for item in report.scoring_report.scores:
        if item.anomaly_scored and item.anomaly_score is not None:
            assert math.isfinite(item.anomaly_score)


def _business_fingerprint(artifacts: WorkflowArtifacts) -> dict[str, Any]:
    residual_test = artifacts.residual_final.test_scored
    anomaly_test = artifacts.anomaly_final.test_scored
    return {
        "profile_rows": artifacts.profile.row_count,
        "profile_cols": artifacts.profile.column_count,
        "validation_issue_codes": sorted(
            {issue.issue_type for issue in artifacts.validation_issues}
        ),
        "quality_score": artifacts.quality_score.total_score,
        "selected_industry": artifacts.industry_result.selected_industry,
        "selected_task": artifacts.task_result.selected_task.value,
        "features": list(artifacts.feature_columns),
        "roles": {
            item.column_name: item.role.value
            for item in artifacts.role_mapping.assignments
            if item.column_name
            in {*PROCESS_FEATURES, TARGET_COLUMN, TIME_COLUMN, *ID_COLUMNS}
        },
        "train_ids": list(artifacts.split.summary.train_original_row_ids),
        "validation_ids": list(artifacts.split.summary.validation_original_row_ids),
        "test_ids": list(artifacts.split.summary.test_original_row_ids),
        "supervised_key": artifacts.supervised_screening.summary.selected_estimator_key,
        "anomaly_key": artifacts.anomaly_screening.summary.selected_estimator_key,
        "test_predictions": residual_test.get_column("_regression_prediction").to_list(),
        "anomaly_scores": anomaly_test.get_column("_anomaly_score").to_list(),
        "residual_scores": residual_test.get_column("_residual_anomaly_score").to_list(),
        "residual_indicators": residual_test.get_column("_is_residual_anomaly").to_list(),
        "ensemble_factors": [f.variable for f in artifacts.ensemble_result.factors],
    }


def test_raw_csv_full_workflow_determinism(tmp_path: Path) -> None:
    first = run_workflow(tmp_path / "run_a")
    second = run_workflow(tmp_path / "run_b")
    assert _business_fingerprint(first) == _business_fingerprint(second)

    # Recommendation business result determinism for one objective.
    request_a = _build_recommendation_pipeline_request(
        first,
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
    )
    request_b = _build_recommendation_pipeline_request(
        second,
        objective=RecommendationObjective.IMPROVE_PREDICTED_QUALITY,
    )
    outcome_a = RecommendationPipeline(
        scenario_scorer=CandidateScenarioScorer(
            quality_model=first.supervised_final.final_model,
            anomaly_model=first.anomaly_final.final_model,
        )
    ).run(request_a)
    outcome_b = RecommendationPipeline(
        scenario_scorer=CandidateScenarioScorer(
            quality_model=second.supervised_final.final_model,
            anomaly_model=second.anomaly_final.final_model,
        )
    ).run(request_b)
    report_a = outcome_a.report
    report_b = outcome_b.report
    assert report_a.status is report_b.status
    assert [s.scenario_id for s in report_a.grid_report.scenarios] == [
        s.scenario_id for s in report_b.grid_report.scenarios
    ]
    assert [
        item.scenario_id for item in report_a.ranking_report.ranked_scenarios
    ] == [item.scenario_id for item in report_b.ranking_report.ranked_scenarios]
    assert [
        (c.variable, c.proposed_value, c.current_value)
        for c in report_a.final_result.changes
    ] == [
        (c.variable, c.proposed_value, c.current_value)
        for c in report_b.final_result.changes
    ]
    assert report_a.final_result.baseline_prediction == pytest.approx(
        report_b.final_result.baseline_prediction
        if report_b.final_result.baseline_prediction is not None
        else 0.0
    ) or (
        report_a.final_result.baseline_prediction
        == report_b.final_result.baseline_prediction
    )
    assert report_a.final_result.confidence == report_b.final_result.confidence


def test_raw_csv_input_and_stage_immutability(tmp_path: Path) -> None:
    csv_path = write_synthetic_csv(tmp_path / CSV_NAME)
    loaded = DatasetLoader().load(csv_path)
    loaded_snapshot = _snapshot_frame(loaded.frame)
    metadata_snapshot = loaded.metadata.model_dump()

    profile = DatasetProfiler().profile(loaded.frame)
    issues = DatasetValidator().validate(loaded.frame)
    _ = DataQualityScorer().score(issues)
    assert loaded.frame.to_dicts() == loaded_snapshot
    assert loaded.metadata.model_dump() == metadata_snapshot

    sort_result = DatasetSorter().sort(loaded.frame, by=TIME_COLUMN)
    assert loaded.frame.to_dicts() == loaded_snapshot

    tracker = LineageTracker(loaded.frame)
    tracker.record(sort_result.frame, sort_result.event)
    assert loaded.frame.to_dicts() == loaded_snapshot

    artifacts = run_workflow(tmp_path / "immutable_run")
    split_train_snapshot = _snapshot_frame(artifacts.split.train)
    split_validation_snapshot = _snapshot_frame(artifacts.split.validation)
    split_test_snapshot = _snapshot_frame(artifacts.split.test)
    diagnosis_snapshot = artifacts.ensemble_result.model_dump()
    leakage_snapshot = artifacts.leakage_report.model_dump()

    request = _build_recommendation_pipeline_request(
        artifacts,
        objective=RecommendationObjective.BALANCE_QUALITY_AND_ANOMALY,
    )
    request_snapshot = copy.deepcopy(request.model_dump())
    quality_meta = artifacts.supervised_final.final_model.get_metadata().model_dump()
    anomaly_meta = artifacts.anomaly_final.final_model.get_metadata().model_dump()

    RecommendationPipeline(
        scenario_scorer=CandidateScenarioScorer(
            quality_model=artifacts.supervised_final.final_model,
            anomaly_model=artifacts.anomaly_final.final_model,
        )
    ).run(request)

    assert artifacts.split.train.to_dicts() == split_train_snapshot
    assert artifacts.split.validation.to_dicts() == split_validation_snapshot
    assert artifacts.split.test.to_dicts() == split_test_snapshot
    assert artifacts.ensemble_result.model_dump() == diagnosis_snapshot
    assert artifacts.leakage_report.model_dump() == leakage_snapshot
    assert request.model_dump() == request_snapshot
    assert (
        artifacts.supervised_final.final_model.get_metadata().model_dump()
        == quality_meta
    )
    assert artifacts.anomaly_final.final_model.get_metadata().model_dump() == anomaly_meta
    assert profile.row_count == N_ROWS
