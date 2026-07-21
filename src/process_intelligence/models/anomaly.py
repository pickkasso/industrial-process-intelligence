"""Isolation Forest unsupervised anomaly detection (Step 7A)."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any, Literal, Self

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import sklearn  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, field_validator, model_validator
from sklearn.base import is_classifier, is_regressor  # type: ignore[import-untyped]
from sklearn.ensemble import IsolationForest  # type: ignore[import-untyped]

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
_FACTORY_TIME_BUDGET_SECONDS = 15.0


def _is_finite_number(value: object) -> bool:
    """Return True when ``value`` is a finite real number (bool excluded)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _require_real_int(value: object, *, field_name: str) -> int:
    """Validate a non-bool integer value."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{field_name} must be an int (bool not allowed), "
            f"got {type(value).__name__}"
        )
    return value


class IsolationForestConfig(BaseModel):
    """Configuration for sklearn Isolation Forest anomaly detection."""

    n_estimators: int = 100
    contamination: float | Literal["auto"] = 0.05
    max_samples: int | float | Literal["auto"] = "auto"
    max_features: int | float = 1.0
    bootstrap: bool = False
    random_state: int = 42
    n_jobs: int = 1

    @field_validator("n_estimators", mode="before")
    @classmethod
    def _validate_n_estimators(cls, value: object) -> int:
        number = _require_real_int(value, field_name="n_estimators")
        if number < 1:
            raise ValueError(f"n_estimators must be >= 1, got {number}")
        return number

    @field_validator("contamination", mode="before")
    @classmethod
    def _validate_contamination(cls, value: object) -> float | Literal["auto"]:
        if isinstance(value, str):
            if value != "auto":
                raise ValueError(
                    "contamination string must be exactly 'auto', "
                    f"got {value!r}"
                )
            return "auto"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "contamination must be 'auto' or a float in (0.0, 0.5], "
                f"got {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(
                f"contamination must be a finite float, got {value!r}"
            )
        if number <= 0.0 or number > 0.5:
            raise ValueError(
                f"contamination float must be in (0.0, 0.5], got {number}"
            )
        return number

    @field_validator("max_samples", mode="before")
    @classmethod
    def _validate_max_samples(cls, value: object) -> int | float | Literal["auto"]:
        if isinstance(value, str):
            if value != "auto":
                raise ValueError(
                    "max_samples string must be exactly 'auto', "
                    f"got {value!r}"
                )
            return "auto"
        if isinstance(value, bool):
            raise ValueError("max_samples must not be a bool")
        if isinstance(value, int) and not isinstance(value, bool):
            if value < 1:
                raise ValueError(f"max_samples int must be >= 1, got {value}")
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(
                    f"max_samples must be a finite float, got {value!r}"
                )
            if value <= 0.0 or value > 1.0:
                raise ValueError(
                    f"max_samples float must be in (0.0, 1.0], got {value}"
                )
            return value
        raise ValueError(
            "max_samples must be 'auto', int, or float, "
            f"got {type(value).__name__}"
        )

    @field_validator("max_features", mode="before")
    @classmethod
    def _validate_max_features(cls, value: object) -> int | float:
        if isinstance(value, bool):
            raise ValueError("max_features must not be a bool")
        if isinstance(value, int) and not isinstance(value, bool):
            if value < 1:
                raise ValueError(f"max_features int must be >= 1, got {value}")
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(
                    f"max_features must be a finite float, got {value!r}"
                )
            if value <= 0.0 or value > 1.0:
                raise ValueError(
                    f"max_features float must be in (0.0, 1.0], got {value}"
                )
            return value
        raise ValueError(
            f"max_features must be int or float, got {type(value).__name__}"
        )

    @field_validator("bootstrap", mode="before")
    @classmethod
    def _validate_bootstrap(cls, value: object) -> bool:
        if not isinstance(value, bool):
            raise ValueError(
                f"bootstrap must be a bool, got {type(value).__name__}"
            )
        return value

    @field_validator("random_state", mode="before")
    @classmethod
    def _validate_random_state(cls, value: object) -> int:
        number = _require_real_int(value, field_name="random_state")
        if number < 0:
            raise ValueError(f"random_state must be >= 0, got {number}")
        return number

    @field_validator("n_jobs", mode="before")
    @classmethod
    def _validate_n_jobs(cls, value: object) -> int:
        number = _require_real_int(value, field_name="n_jobs")
        if number == 0 or number < -1:
            raise ValueError(
                f"n_jobs must be a positive int or -1, got {number}"
            )
        return number


class AnomalyDetectionResult(BaseModel):
    """Structured Isolation Forest anomaly scores and binary decisions."""

    scores: list[float] = Field(default_factory=list)
    is_anomaly: list[bool] = Field(default_factory=list)
    raw_predictions: list[int] = Field(default_factory=list)
    threshold: float
    row_count: int
    anomaly_count: int
    anomaly_fraction: float
    score_min: float | None = None
    score_max: float | None = None
    score_mean: float | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_consistency(self) -> Self:
        if self.row_count < 0:
            raise ValueError(f"row_count must be >= 0, got {self.row_count}")
        if self.anomaly_count < 0:
            raise ValueError(
                f"anomaly_count must be >= 0, got {self.anomaly_count}"
            )
        if self.anomaly_count > self.row_count:
            raise ValueError(
                "anomaly_count cannot exceed row_count: "
                f"{self.anomaly_count} > {self.row_count}"
            )
        if len(self.scores) != self.row_count:
            raise ValueError(
                f"scores length ({len(self.scores)}) must equal "
                f"row_count ({self.row_count})"
            )
        if len(self.is_anomaly) != self.row_count:
            raise ValueError(
                f"is_anomaly length ({len(self.is_anomaly)}) must equal "
                f"row_count ({self.row_count})"
            )
        if len(self.raw_predictions) != self.row_count:
            raise ValueError(
                f"raw_predictions length ({len(self.raw_predictions)}) must "
                f"equal row_count ({self.row_count})"
            )
        if not _is_finite_number(self.threshold):
            raise ValueError(
                f"threshold must be a finite float, got {self.threshold!r}"
            )
        for index, score in enumerate(self.scores):
            if not _is_finite_number(score):
                raise ValueError(
                    f"scores[{index}] must be a finite float, got {score!r}"
                )
        for index, raw in enumerate(self.raw_predictions):
            if raw not in (-1, 1):
                raise ValueError(
                    f"raw_predictions[{index}] must be -1 or 1, got {raw!r}"
                )
            expected_anomaly = raw == -1
            if bool(self.is_anomaly[index]) is not expected_anomaly:
                raise ValueError(
                    "is_anomaly must match raw_predictions == -1 at "
                    f"index {index}"
                )
        true_count = sum(1 for flag in self.is_anomaly if flag)
        if true_count != self.anomaly_count:
            raise ValueError(
                "anomaly_count must equal the number of True values in "
                f"is_anomaly ({true_count}), got {self.anomaly_count}"
            )
        if not _is_finite_number(self.anomaly_fraction):
            raise ValueError(
                "anomaly_fraction must be a finite float in [0.0, 1.0], "
                f"got {self.anomaly_fraction!r}"
            )
        fraction = float(self.anomaly_fraction)
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError(
                f"anomaly_fraction must be in [0.0, 1.0], got {fraction}"
            )
        if self.row_count == 0:
            if fraction != 0.0:
                raise ValueError(
                    "anomaly_fraction must be 0.0 when row_count is 0"
                )
            if (
                self.score_min is not None
                or self.score_max is not None
                or self.score_mean is not None
            ):
                raise ValueError(
                    "score_min, score_max, and score_mean must be None "
                    "when row_count is 0"
                )
        else:
            expected_fraction = self.anomaly_count / self.row_count
            if not math.isclose(
                fraction,
                expected_fraction,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "anomaly_fraction must equal anomaly_count / row_count "
                    f"({expected_fraction}), got {fraction}"
                )
            for name, summary in (
                ("score_min", self.score_min),
                ("score_max", self.score_max),
                ("score_mean", self.score_mean),
            ):
                if summary is None or not _is_finite_number(summary):
                    raise ValueError(
                        f"{name} must be a finite float when row_count >= 1, "
                        f"got {summary!r}"
                    )
            assert self.score_min is not None
            assert self.score_max is not None
            assert self.score_mean is not None
            if not (self.score_min <= self.score_mean <= self.score_max):
                raise ValueError(
                    "score summaries must satisfy "
                    "score_min <= score_mean <= score_max"
                )
        if len(self.warnings) != len(set(self.warnings)):
            raise ValueError("warnings must not contain duplicate values")
        return self


class IsolationForestModelMetadata(ModelMetadata):
    """ModelMetadata extended with Isolation Forest-specific settings."""

    estimator_key: str = ""
    estimator_class: str = ""
    fit_row_count: int = Field(default=0, ge=0)
    fitted: bool = False
    fitted_at: datetime | None = None
    contamination: float | Literal["auto"] = 0.05
    n_estimators: int = 100
    max_samples: int | float | Literal["auto"] = "auto"
    max_features: int | float = 1.0
    bootstrap: bool = False
    threshold: float | None = None
    anomaly_score_direction: str = _ANOMALY_SCORE_DIRECTION


def _build_isolation_forest(config: IsolationForestConfig) -> IsolationForest:
    """Create a fresh IsolationForest from ``config`` values."""
    return IsolationForest(
        n_estimators=config.n_estimators,
        contamination=config.contamination,
        max_samples=config.max_samples,
        max_features=config.max_features,
        bootstrap=config.bootstrap,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )


def _require_anomaly_estimator(estimator: object) -> Any:
    """Validate an unsupervised outlier estimator duck-type and role."""
    for method_name in ("fit", "predict", "decision_function", "score_samples"):
        method = getattr(estimator, method_name, None)
        if not callable(method):
            raise TypeError(
                f"estimator must have a callable {method_name} method, "
                f"got {type(estimator).__name__}"
            )
    try:
        supervised = bool(is_regressor(estimator) or is_classifier(estimator))
    except AttributeError:
        # Duck-typed objects without sklearn tags are allowed when they expose
        # the required methods and remain cloneable.
        supervised = False
    if supervised:
        raise TypeError(
            "estimator must be an outlier detector, not a "
            f"regressor/classifier ({type(estimator).__name__})"
        )
    return estimator


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


class IsolationForestAnomalyModel(BaseAnomalyModel):
    """Isolation Forest concrete model implementing ``BaseAnomalyModel``."""

    def __init__(
        self,
        *,
        spec: ModelSpec,
        config: IsolationForestConfig | None = None,
        estimator: Any | None = None,
    ) -> None:
        """Create an Isolation Forest anomaly model around a cloned estimator.

        Args:
            spec: Model specification with ``task=UNSUPERVISED_ANOMALY``.
            config: Isolation Forest hyperparameters. ``None`` uses defaults.
            estimator: Optional sklearn outlier detector template. When
                provided it is cloned and never fitted or mutated. When
                omitted, an ``IsolationForest`` is built from ``config``.

        Raises:
            TypeError: If ``spec``, ``config``, or ``estimator`` is invalid,
                or if the estimator cannot be cloned.
            DataValidationError: If ``spec.task`` is not unsupervised anomaly.
        """
        if not isinstance(spec, ModelSpec):
            raise TypeError(f"spec must be ModelSpec, got {type(spec).__name__}")
        if spec.task is not AnalysisTask.UNSUPERVISED_ANOMALY:
            raise DataValidationError(
                "IsolationForestAnomalyModel requires "
                "task=UNSUPERVISED_ANOMALY, "
                f"got {spec.task!r}"
            )

        if config is None:
            resolved_config = IsolationForestConfig()
        elif isinstance(config, IsolationForestConfig):
            resolved_config = config.model_copy(deep=True)
        else:
            raise TypeError(
                "config must be IsolationForestConfig or None, "
                f"got {type(config).__name__}"
            )

        if estimator is None:
            template = _build_isolation_forest(resolved_config)
        else:
            validated = _require_anomaly_estimator(estimator)
            template = _clone_estimator(validated)
            template = _apply_random_state(template, resolved_config.random_state)

        self._spec: ModelSpec = spec.model_copy(deep=True)
        self._config: IsolationForestConfig = resolved_config
        self._estimator_template: Any = template
        self._fitted_estimator: Any | None = None
        self._is_fitted: bool = False
        self._feature_names: tuple[str, ...] = ()
        self._fit_row_count: int = 0
        self._threshold: float | None = None
        self._fitted_at: datetime | None = None

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

        With ``anomaly_score = -decision_function``, the sklearn
        Isolation Forest decision boundary of ``0`` maps to ``0.0``.
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
        """Fit Isolation Forest on training features only.

        Args:
            X: Polars or Pandas numeric feature frame.
            y: Must be ``None``. Unsupervised fitting does not use labels.

        Returns:
            The fitted model instance.

        Raises:
            DataValidationError: If ``y`` is not ``None`` or feature validation
                fails.
            InsufficientDataError: If ``X`` has no rows or columns.
            TypeError: If ``X`` has an unsupported type.
        """
        if y is not None:
            raise DataValidationError(
                "IsolationForestAnomalyModel.fit does not accept y; "
                "unsupervised anomaly detection uses features only"
            )

        frame, feature_names = _validate_x_frame(X, allow_empty_rows=False)
        matrix = _frame_to_numeric_array(frame, feature_names)

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

        Scores are defined as ``-decision_function(X)`` so that larger values
        indicate stronger anomaly evidence. Sklearn's native
        ``score_samples`` values are not exposed.

        Args:
            X: Feature frame matching the training schema.

        Returns:
            One-dimensional float ndarray of anomaly scores.
        """
        decisions = self.decision_function(X)
        return np.asarray(-decisions, dtype=np.float64)

    def predict(self, X: DataFrameLike) -> np.ndarray:
        """Return raw Isolation Forest labels (``1`` normal, ``-1`` anomaly).

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

        Anomaly flags follow raw Isolation Forest predictions (``-1``), not a
        re-threshold of floating-point anomaly scores.

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
        """Convert Isolation Forest anomaly rows into ``AnomalyEvent`` records.

        Args:
            X: Feature frame matching the training schema.

        Returns:
            One ``AnomalyEvent`` per row flagged as anomalous.
        """
        result = self.detect(X)
        events: list[AnomalyEvent] = []
        for index, (flag, score) in enumerate(
            zip(result.is_anomaly, result.scores, strict=True)
        ):
            if not flag:
                continue
            events.append(
                AnomalyEvent(
                    anomaly_id=f"isolation_forest-{index}",
                    anomaly_type=AnomalyType.MULTIVARIATE_COMBINATION,
                    anomaly_score=float(score),
                    severity="anomaly",
                    sample_id=index,
                    model_confidence=0.0,
                    detector=self._spec.estimator_key,
                    rationale=(
                        "Flagged by Isolation Forest raw prediction (-1); "
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
        """Reject ground-truth anomaly evaluation (out of Step 7A scope).

        Args:
            X: Feature frame (unused).
            y: Optional ground-truth labels (unused).

        Raises:
            ProcessIntelligenceError: Always, because anomaly ground-truth
                metrics are not implemented in Step 7A.
        """
        _ = (X, y)
        raise ProcessIntelligenceError(
            "anomaly evaluation metrics with ground-truth labels are not "
            "included in Step 7A scope"
        )

    def explain(self, X: DataFrameLike) -> ExplanationResult:
        """Return an empty global explanation fallback.

        Isolation Forest does not expose ``coef_`` or ``feature_importances_``.
        SHAP and permutation importance are intentionally not used in Step 7A.

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
            notes=[
                "Isolation Forest does not provide native feature importance "
                "(coef_ / feature_importances_)"
            ],
        )

    def get_metadata(self) -> ModelMetadata:
        """Return Isolation Forest metadata without estimator or training data.

        Returns:
            A new ``IsolationForestModelMetadata`` instance. Mutating the
            returned object does not affect model state.
        """
        estimator_obj = (
            self._fitted_estimator
            if self._fitted_estimator is not None
            else self._estimator_template
        )
        return IsolationForestModelMetadata(
            model_name=self._spec.name,
            version=sklearn.__version__,
            task=self._spec.task,
            features=list(self._feature_names),
            training_timestamp=self._fitted_at,
            seed=self._config.random_state,
            optional_dependencies_used=list(self._spec.optional_dependencies),
            estimator_key=self._spec.estimator_key,
            estimator_class=type(estimator_obj).__name__,
            fit_row_count=self._fit_row_count,
            fitted=self._is_fitted,
            fitted_at=self._fitted_at,
            contamination=self._config.contamination,
            n_estimators=self._config.n_estimators,
            max_samples=self._config.max_samples,
            max_features=self._config.max_features,
            bootstrap=self._config.bootstrap,
            threshold=self._threshold,
            anomaly_score_direction=_ANOMALY_SCORE_DIRECTION,
        )


def create_isolation_forest_anomaly_model(
    *,
    config: IsolationForestConfig | None = None,
) -> IsolationForestAnomalyModel:
    """Create an independent unfitted Isolation Forest anomaly model.

    Args:
        config: Optional hyperparameters. ``None`` uses defaults. The input
            config is deep-copied and never mutated.

    Returns:
        A new ``IsolationForestAnomalyModel`` with a dedicated estimator,
        config, and ``ModelSpec``.

    Raises:
        TypeError: If ``config`` is not ``None`` or ``IsolationForestConfig``.
    """
    if config is None:
        resolved = IsolationForestConfig()
    elif isinstance(config, IsolationForestConfig):
        resolved = config.model_copy(deep=True)
    else:
        raise TypeError(
            "config must be IsolationForestConfig or None, "
            f"got {type(config).__name__}"
        )

    spec = ModelSpec(
        name="Isolation Forest",
        task=AnalysisTask.UNSUPERVISED_ANOMALY,
        estimator_key="isolation_forest",
        optional_dependencies=[],
        priority=10,
        time_budget_seconds=_FACTORY_TIME_BUDGET_SECONDS,
    )
    return IsolationForestAnomalyModel(spec=spec, config=resolved)
