"""Shared sklearn outlier-detector adapter for unsupervised anomaly models."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Self

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import sklearn  # type: ignore[import-untyped]
from pydantic import Field

from process_intelligence.core.enums import AnalysisTask, AnomalyType
from process_intelligence.core.exceptions import (
    DataValidationError,
    InsufficientDataError,
    ProcessIntelligenceError,
)
from process_intelligence.core.protocols import (
    BaseAnomalyModel,
    DataFrameLike,
    SeriesLike,
)
from process_intelligence.core.schemas import (
    AnomalyEvent,
    ExplanationResult,
    ModelEvaluation,
    ModelMetadata,
    ModelSpec,
)
from process_intelligence.models.anomaly import AnomalyDetectionResult
from process_intelligence.models.sklearn_adapter import (
    _apply_random_state,
    _clone_estimator,
    _column_names,
    _frame_to_numeric_array,
    _row_count,
    _validate_feature_dtypes,
    _validate_feature_names,
    _validate_numeric_finite_frame,
)

_ANOMALY_SCORE_DIRECTION = "higher_is_more_anomalous"
_DEFAULT_THRESHOLD = 0.0


class SklearnOutlierModelMetadata(ModelMetadata):
    """ModelMetadata extended with shared outlier-detector settings."""

    estimator_key: str = ""
    estimator_class: str = ""
    fit_row_count: int = Field(default=0, ge=0)
    fitted: bool = False
    fitted_at: datetime | None = None
    threshold: float | None = None
    anomaly_score_direction: str = _ANOMALY_SCORE_DIRECTION
    parameters: dict[str, Any] = Field(default_factory=dict)


def _validate_optional_random_state(value: object) -> int | None:
    """Validate ``None`` or a non-negative integer random seed (bool excluded)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            "random_state must be None or a non-negative int "
            f"(bool not allowed), got {type(value).__name__}"
        )
    if value < 0:
        raise ValueError(f"random_state must be >= 0, got {value}")
    return value


def _require_outlier_estimator_methods(estimator: object) -> Any:
    """Require the public sklearn outlier-detector method contract."""
    for method_name in ("fit", "predict", "decision_function", "score_samples"):
        method = getattr(estimator, method_name, None)
        if not callable(method):
            raise TypeError(
                f"estimator must have a callable {method_name} method, "
                f"got {type(estimator).__name__}"
            )
    return estimator


def _validate_metadata_parameters(
    value: object,
) -> dict[str, Any]:
    """Validate and deep-copy optional metadata parameter mapping."""
    if value is None:
        return {}
    if isinstance(value, (str, bytes)) or not isinstance(value, Mapping):
        raise TypeError(
            "metadata_parameters must be a Mapping[str, Any] or None "
            f"(str/bytes not allowed), got {type(value).__name__}"
        )
    return copy.deepcopy(dict(value))


def _validate_x_frame(
    frame: object,
    *,
    allow_empty_rows: bool,
) -> tuple[DataFrameLike, tuple[str, ...]]:
    """Validate a feature frame for fit or prediction."""
    if not isinstance(frame, (pl.DataFrame, pd.DataFrame)):
        raise TypeError(
            "X must be a polars.DataFrame or pandas.DataFrame, "
            f"got {type(frame).__name__}"
        )
    row_count = _row_count(frame)
    if row_count == 0 and not allow_empty_rows:
        raise InsufficientDataError("X must contain at least one row")

    feature_names = _validate_feature_names(_column_names(frame))
    _validate_feature_dtypes(frame, feature_names)
    if row_count > 0:
        _validate_numeric_finite_frame(frame, feature_names)
    return frame, feature_names


def _ensure_prediction_schema(
    feature_names: tuple[str, ...],
    expected: tuple[str, ...],
) -> None:
    """Require exact feature name and order match with training schema."""
    if feature_names != expected:
        raise DataValidationError(
            "X feature names and order must exactly match training features: "
            f"expected {list(expected)}, got {list(feature_names)}"
        )


class SklearnOutlierDetectorAdapter(BaseAnomalyModel, ABC):
    """Abstract adapter wrapping a cloneable sklearn outlier detector.

    Concrete models supply estimator-role validation and optional fit guards.
    Isolation Forest is intentionally not migrated onto this adapter.
    """

    def __init__(
        self,
        *,
        spec: ModelSpec,
        estimator: Any,
        random_state: int | None = None,
        metadata_parameters: Mapping[str, Any] | None = None,
    ) -> None:
        """Create an adapter around a cloned outlier-detector template.

        Args:
            spec: Model specification with ``task=UNSUPERVISED_ANOMALY``.
            estimator: Sklearn-style outlier detector exposing ``fit``,
                ``predict``, ``decision_function``, and ``score_samples``.
                The input estimator is cloned and never fitted or mutated.
            random_state: Optional non-negative seed applied to the clone when
                the estimator exposes a ``random_state`` parameter.
            metadata_parameters: Optional model-specific settings stored for
                metadata. Deep-copied; never mutated.

        Raises:
            TypeError: If ``spec``, ``estimator``, ``random_state``, or
                ``metadata_parameters`` is invalid, or if cloning fails.
            DataValidationError: If ``spec.task`` is not unsupervised anomaly.
            ValueError: If ``random_state`` is negative.
        """
        if not isinstance(spec, ModelSpec):
            raise TypeError(f"spec must be ModelSpec, got {type(spec).__name__}")
        if spec.task is not AnalysisTask.UNSUPERVISED_ANOMALY:
            raise DataValidationError(
                f"{type(self).__name__} requires task=UNSUPERVISED_ANOMALY, "
                f"got {spec.task!r}"
            )

        validated = _require_outlier_estimator_methods(estimator)
        validated = self._validate_estimator(validated)
        validated_random_state = _validate_optional_random_state(random_state)
        parameters = _validate_metadata_parameters(metadata_parameters)

        template = _clone_estimator(validated)
        if validated_random_state is not None:
            template = _apply_random_state(template, validated_random_state)

        self._spec: ModelSpec = spec.model_copy(deep=True)
        self._estimator_template: Any = template
        self._random_state: int | None = validated_random_state
        self._metadata_parameters: dict[str, Any] = parameters
        self._fitted_estimator: Any | None = None
        self._is_fitted: bool = False
        self._feature_names: tuple[str, ...] = ()
        self._fit_row_count: int = 0
        self._threshold: float | None = None
        self._fitted_at: datetime | None = None

    @abstractmethod
    def _validate_estimator(self, estimator: Any) -> Any:
        """Validate estimator type or role for the concrete detector.

        Args:
            estimator: Candidate outlier detector after method-contract checks.

        Returns:
            The validated estimator object.

        Raises:
            TypeError: If the estimator is unsuitable for the concrete model.
        """

    def _guard_fit(self, *, row_count: int, feature_count: int) -> None:
        """Optional model-specific fit guard invoked after X validation.

        Args:
            row_count: Number of training rows.
            feature_count: Number of feature columns.

        Raises:
            InsufficientDataError: If data is insufficient for the detector.
            DataValidationError: If model-specific validation fails.
        """
        _ = (row_count, feature_count)

    def _explain_unavailable_note(self) -> str:
        """Return the note describing missing native feature importance."""
        estimator_name = type(self._estimator_template).__name__
        return (
            f"{estimator_name} does not provide native feature importance "
            "(coef_ / feature_importances_)"
        )

    @property
    def is_fitted(self) -> bool:
        """Whether the model has a successfully fitted estimator."""
        return self._is_fitted

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Feature names learned during the last successful fit."""
        return self._feature_names

    @property
    def fit_row_count(self) -> int:
        """Number of training rows from the last successful fit."""
        return self._fit_row_count

    @property
    def threshold(self) -> float | None:
        """Anomaly-score decision boundary, or ``None`` before fit.

        With ``anomaly_score = -decision_function``, the sklearn decision
        boundary of ``0`` maps to ``0.0``.
        """
        return self._threshold

    @property
    def fitted_at(self) -> datetime | None:
        """Timezone-aware UTC timestamp of the last successful fit."""
        return self._fitted_at

    def fit(
        self,
        X: DataFrameLike,
        y: SeriesLike | None = None,
    ) -> Self:
        """Fit a fresh estimator clone on training features only.

        Args:
            X: Polars or Pandas numeric feature frame.
            y: Must be ``None``. Unsupervised fitting does not use labels.

        Returns:
            The fitted model instance.

        Raises:
            DataValidationError: If ``y`` is not ``None`` or feature validation
                fails.
            InsufficientDataError: If ``X`` has no rows or columns, or a
                concrete fit guard rejects the sample size.
            TypeError: If ``X`` has an unsupported type.
        """
        if y is not None:
            raise DataValidationError(
                f"{type(self).__name__}.fit does not accept y; "
                "unsupervised anomaly detection uses features only"
            )

        frame, feature_names = _validate_x_frame(X, allow_empty_rows=False)
        matrix = _frame_to_numeric_array(frame, feature_names)
        self._guard_fit(
            row_count=int(matrix.shape[0]),
            feature_count=int(matrix.shape[1]),
        )

        candidate = _clone_estimator(self._estimator_template)
        candidate.fit(matrix)

        self._fitted_estimator = candidate
        self._is_fitted = True
        self._feature_names = feature_names
        self._fit_row_count = int(matrix.shape[0])
        self._threshold = float(_DEFAULT_THRESHOLD)
        self._fitted_at = datetime.now(tz=UTC)
        return self

    def _require_fitted(self, *, operation: str) -> Any:
        """Return the fitted estimator or raise if the model is not fitted."""
        if not self._is_fitted or self._fitted_estimator is None:
            raise ProcessIntelligenceError(
                f"{operation} requires a successful fit before it can be called"
            )
        return self._fitted_estimator

    def _prepare_prediction_matrix(
        self,
        X: DataFrameLike,
    ) -> tuple[DataFrameLike, np.ndarray]:
        """Validate prediction ``X`` and convert it to a numeric matrix."""
        frame, feature_names = _validate_x_frame(X, allow_empty_rows=True)
        _ensure_prediction_schema(feature_names, self._feature_names)
        if _row_count(frame) == 0:
            return frame, np.empty((0, len(self._feature_names)), dtype=np.float64)
        return frame, _frame_to_numeric_array(frame, feature_names)

    def decision_function(self, X: DataFrameLike) -> np.ndarray:
        """Return sklearn decision scores (higher means more normal).

        Args:
            X: Feature frame matching the training schema.

        Returns:
            One-dimensional float ndarray. Empty when ``X`` has zero rows.
        """
        estimator = self._require_fitted(operation="decision_function")
        _frame, matrix = self._prepare_prediction_matrix(X)
        if matrix.shape[0] == 0:
            return np.empty(0, dtype=np.float64)

        values = np.asarray(estimator.decision_function(matrix), dtype=np.float64)
        values = np.reshape(values, -1)
        if values.ndim != 1:
            raise ProcessIntelligenceError(
                "estimator.decision_function must return a 1D array, "
                f"got shape {values.shape}"
            )
        if values.shape[0] != matrix.shape[0]:
            raise ProcessIntelligenceError(
                "decision_function row count "
                f"({values.shape[0]}) must match X row count ({matrix.shape[0]})"
            )
        if not np.isfinite(values).all():
            raise ProcessIntelligenceError(
                "decision_function results must be finite"
            )
        return values

    def score_samples(self, X: DataFrameLike) -> np.ndarray:
        """Return anomaly scores where higher values are more anomalous.

        Scores are defined as ``-decision_function(X)``. Sklearn's native
        ``score_samples`` values are not exposed.

        Args:
            X: Feature frame matching the training schema.

        Returns:
            One-dimensional float ndarray of anomaly scores.
        """
        decisions = self.decision_function(X)
        return np.asarray(-decisions, dtype=np.float64)

    def predict(self, X: DataFrameLike) -> np.ndarray:
        """Return raw outlier labels (``1`` normal, ``-1`` anomaly).

        Args:
            X: Feature frame matching the training schema.

        Returns:
            One-dimensional integer ndarray of ``-1`` / ``1`` labels.
        """
        estimator = self._require_fitted(operation="predict")
        _frame, matrix = self._prepare_prediction_matrix(X)
        if matrix.shape[0] == 0:
            return np.empty(0, dtype=np.int64)

        values = np.asarray(estimator.predict(matrix))
        values = np.reshape(values, -1)
        if values.ndim != 1:
            raise ProcessIntelligenceError(
                f"estimator.predict must return a 1D array, got shape {values.shape}"
            )
        if values.shape[0] != matrix.shape[0]:
            raise ProcessIntelligenceError(
                "predict row count "
                f"({values.shape[0]}) must match X row count ({matrix.shape[0]})"
            )
        unique = set(np.unique(values).tolist())
        if not unique.issubset({-1, 1}):
            raise ProcessIntelligenceError(
                "estimator.predict must return only -1 or 1 values, "
                f"got {sorted(unique)}"
            )
        return values.astype(np.int64, copy=False)

    def detect(self, X: DataFrameLike) -> AnomalyDetectionResult:
        """Score and classify rows, preserving input order.

        Anomaly flags follow raw predictions (``-1``), not a re-threshold of
        floating-point anomaly scores.

        Args:
            X: Feature frame matching the training schema.

        Returns:
            Structured anomaly detection result for every input row.
        """
        self._require_fitted(operation="detect")
        scores = self.score_samples(X)
        raw = self.predict(X)
        row_count = int(scores.shape[0])
        if row_count == 0:
            return AnomalyDetectionResult(
                scores=[],
                is_anomaly=[],
                raw_predictions=[],
                threshold=float(_DEFAULT_THRESHOLD),
                row_count=0,
                anomaly_count=0,
                anomaly_fraction=0.0,
                score_min=None,
                score_max=None,
                score_mean=None,
                warnings=[],
            )

        score_list = [float(value) for value in scores.tolist()]
        raw_list = [int(value) for value in raw.tolist()]
        is_anomaly = [value == -1 for value in raw_list]
        anomaly_count = sum(1 for flag in is_anomaly if flag)
        return AnomalyDetectionResult(
            scores=score_list,
            is_anomaly=is_anomaly,
            raw_predictions=raw_list,
            threshold=float(_DEFAULT_THRESHOLD),
            row_count=row_count,
            anomaly_count=anomaly_count,
            anomaly_fraction=float(anomaly_count) / float(row_count),
            score_min=float(min(score_list)),
            score_max=float(max(score_list)),
            score_mean=float(sum(score_list) / row_count),
            warnings=[],
        )

    def classify_anomalies(self, X: DataFrameLike) -> list[AnomalyEvent]:
        """Convert anomalous rows into ``AnomalyEvent`` records.

        Args:
            X: Feature frame matching the training schema.

        Returns:
            One ``AnomalyEvent`` per row flagged as anomalous.
        """
        result = self.detect(X)
        events: list[AnomalyEvent] = []
        detector_key = self._spec.estimator_key
        for index, (flag, score) in enumerate(
            zip(result.is_anomaly, result.scores, strict=True)
        ):
            if not flag:
                continue
            events.append(
                AnomalyEvent(
                    anomaly_id=f"{detector_key}-{index}",
                    anomaly_type=AnomalyType.MULTIVARIATE_COMBINATION,
                    anomaly_score=float(score),
                    severity="anomaly",
                    sample_id=index,
                    model_confidence=0.0,
                    detector=detector_key,
                    rationale=(
                        "Flagged by outlier detector raw prediction (-1); "
                        "anomaly score is -decision_function"
                    ),
                )
            )
        return events

    def evaluate(
        self,
        X: DataFrameLike,
        y: SeriesLike | None = None,
    ) -> ModelEvaluation:
        """Reject ground-truth anomaly evaluation (out of Step 7C scope).

        Args:
            X: Feature frame (unused).
            y: Optional ground-truth labels (unused).

        Raises:
            ProcessIntelligenceError: Always, because anomaly ground-truth
                metrics are not implemented in Step 7C.
        """
        _ = (X, y)
        raise ProcessIntelligenceError(
            "anomaly evaluation metrics with ground-truth labels are not "
            "included in Step 7C scope"
        )

    def explain(self, X: DataFrameLike) -> ExplanationResult:
        """Return an empty global explanation fallback.

        SHAP and permutation importance are intentionally not used.

        Args:
            X: Feature frame used only for interface compatibility.

        Returns:
            Empty ``ExplanationResult`` noting that native importance is
            unavailable.
        """
        _ = X
        self._require_fitted(operation="explain")
        return ExplanationResult(
            method="native_importance_unavailable",
            feature_importances={},
            notes=[self._explain_unavailable_note()],
        )

    def get_metadata(self) -> ModelMetadata:
        """Return outlier-detector metadata without estimator or training data.

        Returns:
            A new ``SklearnOutlierModelMetadata`` instance. Mutating the
            returned object does not affect model state.
        """
        estimator_obj = (
            self._fitted_estimator
            if self._fitted_estimator is not None
            else self._estimator_template
        )
        return SklearnOutlierModelMetadata(
            model_name=self._spec.name,
            version=sklearn.__version__,
            task=self._spec.task,
            features=list(self._feature_names),
            training_timestamp=self._fitted_at,
            seed=self._random_state,
            optional_dependencies_used=list(self._spec.optional_dependencies),
            estimator_key=self._spec.estimator_key,
            estimator_class=type(estimator_obj).__name__,
            fit_row_count=self._fit_row_count,
            fitted=self._is_fitted,
            fitted_at=self._fitted_at,
            threshold=self._threshold,
            anomaly_score_direction=_ANOMALY_SCORE_DIRECTION,
            parameters=copy.deepcopy(self._metadata_parameters),
        )
