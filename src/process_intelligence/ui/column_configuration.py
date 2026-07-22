"""Automatic column-configuration suggestions for the Streamlit MVP (Step 11B.2).

Produces structural UI recommendations from dtype, nulls, cardinality,
uniqueness, monotonicity, and column-name heuristics. Does not mutate CSV
values, infer final roles, controllability, or recommendation constraints.

Category precedence (single category per column):

1. all-null / unsupported dtype
2. identifier name heuristic with usable distinct non-null values (>= 2)
3. identifier-like name that is constant / all-null / unusable → review
4. timestamp name heuristic
5. high-confidence target name heuristic
6. constant column
7. numeric feature candidate
8. review required / excluded

Identifier-like names are separated from identifier suitability: a constant or
all-null SerialNumber may look like an identifier but is not an automatic
identifier candidate. Target suitability is tracked separately from category
via ``suitable_as_target``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from enum import StrEnum
from typing import Self

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ScalarMetadataValue = str | int | float | bool | None

_IDENTIFIER_NAME_TOKENS: frozenset[str] = frozenset(
    {
        "id",
        "identifier",
        "serial",
        "serialnumber",
        "lot",
        "lotid",
        "wafer",
        "waferid",
        "batch",
        "batchid",
        "sample",
        "sampleid",
        "device",
        "deviceid",
        "cellid",
        "recordid",
    }
)

_TIMESTAMP_NAME_TOKENS: frozenset[str] = frozenset(
    {
        "date",
        "time",
        "timestamp",
        "datetime",
        "recordedat",
        "createdat",
        "measurementtime",
        "cycletime",
    }
)

# Higher priority values sort earlier in target_candidates.
_TARGET_NAME_PRIORITIES: dict[str, int] = {
    "soh": 100,
    "stateofhealth": 95,
    "qualityscore": 90,
    "defectrate": 90,
    "filmthickness": 85,
    "target": 80,
    "label": 75,
    "response": 70,
    "outcome": 70,
    "quality": 65,
    "yield": 60,
    "capacity": 55,
    "efficiency": 50,
    "performance": 45,
    "y": 20,
}

_NAME_SEPARATOR_RE = re.compile(r"[\s_-]+")


def _require_non_empty_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be str, got {type(value).__name__}")
    if value == "" or value.strip() == "":
        raise ValueError(f"{field_name} must be a non-empty, non-whitespace string")
    return value


def _require_strict_bool(value: object, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(
            f"{field_name} must be a bool (0/1 and strings rejected), "
            f"got {type(value).__name__}"
        )
    return value


def _require_strict_int_ge(
    value: object,
    *,
    field_name: str,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{field_name} must be an int >= {minimum} "
            f"(bool not allowed), got {type(value).__name__}"
        )
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {value}")
    return value


def _require_unit_interval_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a finite float in [0, 1] "
            f"(bool not allowed), got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {number}")
    return number


def _validate_scalar_metadata(
    value: object,
    *,
    field_name: str = "metadata",
) -> dict[str, ScalarMetadataValue]:
    if not isinstance(value, dict):
        raise ValueError(
            f"{field_name} must be a dict[str, scalar], got {type(value).__name__}"
        )
    cleaned: dict[str, ScalarMetadataValue] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or key == "" or key.strip() == "":
            raise ValueError(f"{field_name} keys must be non-empty strings")
        if raw is None or isinstance(raw, (str, bool)):
            cleaned[key] = raw
            continue
        if isinstance(raw, int) and not isinstance(raw, bool):
            cleaned[key] = raw
            continue
        if isinstance(raw, float):
            if not math.isfinite(raw):
                raise ValueError(
                    f"{field_name}[{key!r}] float must be finite, got {raw!r}"
                )
            cleaned[key] = raw
            continue
        raise ValueError(
            f"{field_name}[{key!r}] must be str, int, float, bool, or None "
            f"(no DataFrame, Series, ndarray, or nested objects); "
            f"got {type(raw).__name__}"
        )
    return cleaned


def _validate_unique_non_empty_strings(
    values: list[str],
    *,
    field_name: str,
) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _require_non_empty_str(item, field_name=field_name)
        if text in seen:
            raise ValueError(f"{field_name} must not contain duplicates: {text!r}")
        seen.add(text)
        cleaned.append(text)
    return cleaned


def normalize_column_name_for_heuristic(name: str) -> str:
    """Normalize a column name for heuristic token matching only.

    Does not modify the original column name stored on the frame or report.
    """
    stripped = name.strip().casefold()
    return _NAME_SEPARATOR_RE.sub("", stripped)


class UiColumnSuggestionCategory(StrEnum):
    """Structural suggestion category for one CSV column.

    Values are automatic UI proposals only. They do not finalize column roles,
    controllability, verification, or recommendation constraints.
    """

    TARGET_CANDIDATE = "TARGET_CANDIDATE"
    FEATURE_CANDIDATE = "FEATURE_CANDIDATE"
    TIMESTAMP_CANDIDATE = "TIMESTAMP_CANDIDATE"
    IDENTIFIER_CANDIDATE = "IDENTIFIER_CANDIDATE"
    EXCLUDED_CANDIDATE = "EXCLUDED_CANDIDATE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class UiColumnSuggestion(BaseModel):
    """One column suggestion derived from structural CSV metadata.

    ``confidence`` is a UI recommendation-strength heuristic score, not a
    statistical probability.
    """

    model_config = ConfigDict(extra="forbid")

    column: str
    dtype: str
    category: UiColumnSuggestionCategory
    priority: int
    confidence: float
    reasons: list[str]
    numeric: bool
    null_count: int
    unique_count: int
    unique_ratio: float
    constant: bool
    suitable_as_target: bool
    monotonic_non_decreasing: bool | None

    @field_validator("column", "dtype", mode="before")
    @classmethod
    def _validate_text(cls, value: object) -> str:
        return _require_non_empty_str(value, field_name="column/dtype")

    @field_validator("category", mode="before")
    @classmethod
    def _validate_category(cls, value: object) -> UiColumnSuggestionCategory:
        if isinstance(value, UiColumnSuggestionCategory):
            return value
        if isinstance(value, str):
            try:
                return UiColumnSuggestionCategory(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid UiColumnSuggestionCategory: {value!r}"
                ) from exc
        raise ValueError(
            f"category must be UiColumnSuggestionCategory, got {type(value).__name__}"
        )

    @field_validator("priority", mode="before")
    @classmethod
    def _validate_priority(cls, value: object) -> int:
        return _require_strict_int_ge(value, field_name="priority", minimum=0)

    @field_validator("confidence", "unique_ratio", mode="before")
    @classmethod
    def _validate_unit_floats(cls, value: object) -> float:
        return _require_unit_interval_float(value, field_name="unit interval float")

    @field_validator("null_count", "unique_count", mode="before")
    @classmethod
    def _validate_counts(cls, value: object) -> int:
        return _require_strict_int_ge(value, field_name="count", minimum=0)

    @field_validator("numeric", "constant", "suitable_as_target", mode="before")
    @classmethod
    def _validate_bools(cls, value: object) -> bool:
        return _require_strict_bool(value, field_name="bool field")

    @field_validator("monotonic_non_decreasing", mode="before")
    @classmethod
    def _validate_optional_bool(cls, value: object) -> bool | None:
        if value is None:
            return None
        return _require_strict_bool(value, field_name="monotonic_non_decreasing")

    @field_validator("reasons", mode="before")
    @classmethod
    def _validate_reasons_before(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            raise ValueError(
                f"reasons must be a list[str], got {type(value).__name__}"
            )
        return list(value)

    @field_validator("reasons", mode="after")
    @classmethod
    def _validate_reasons(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(value, field_name="reasons")


class UiColumnConfigurationReport(BaseModel):
    """Aggregate automatic column-configuration suggestions for one frame.

    Suggestions are UI proposals only. Final target, roles, controllability,
    verification, and constraints remain user-confirmed.
    """

    model_config = ConfigDict(extra="forbid")

    column_count: int
    row_count: int
    suggestions: list[UiColumnSuggestion]
    target_candidates: list[str]
    recommended_feature_columns: list[str]
    timestamp_candidates: list[str]
    identifier_candidates: list[str]
    excluded_candidates: list[str]
    review_required_columns: list[str]
    numeric_columns: list[str]
    nonnumeric_columns: list[str]
    warnings: list[str]
    metadata: dict[str, ScalarMetadataValue] = Field(default_factory=dict)

    @field_validator("column_count", "row_count", mode="before")
    @classmethod
    def _validate_counts(cls, value: object) -> int:
        return _require_strict_int_ge(value, field_name="count", minimum=0)

    @field_validator("suggestions", mode="before")
    @classmethod
    def _validate_suggestions_before(cls, value: object) -> list[UiColumnSuggestion]:
        if not isinstance(value, list):
            raise ValueError(
                f"suggestions must be a list[UiColumnSuggestion], "
                f"got {type(value).__name__}"
            )
        return list(value)

    @field_validator("suggestions", mode="after")
    @classmethod
    def _validate_suggestions(
        cls,
        value: list[UiColumnSuggestion],
    ) -> list[UiColumnSuggestion]:
        copied: list[UiColumnSuggestion] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, UiColumnSuggestion):
                raise ValueError(
                    "suggestions entries must be UiColumnSuggestion, "
                    f"got {type(item).__name__}"
                )
            if item.column in seen:
                raise ValueError(
                    f"suggestions must not contain duplicate columns: {item.column!r}"
                )
            seen.add(item.column)
            copied.append(item.model_copy(deep=True))
        return copied

    @field_validator(
        "target_candidates",
        "recommended_feature_columns",
        "timestamp_candidates",
        "identifier_candidates",
        "excluded_candidates",
        "review_required_columns",
        "numeric_columns",
        "nonnumeric_columns",
        mode="before",
    )
    @classmethod
    def _validate_name_lists_before(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            raise ValueError(
                f"column name list must be a list[str], got {type(value).__name__}"
            )
        return list(value)

    @field_validator(
        "target_candidates",
        "recommended_feature_columns",
        "timestamp_candidates",
        "identifier_candidates",
        "excluded_candidates",
        "review_required_columns",
        "numeric_columns",
        "nonnumeric_columns",
        mode="after",
    )
    @classmethod
    def _validate_name_lists(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(value, field_name="column name list")

    @field_validator("warnings", mode="before")
    @classmethod
    def _validate_warnings_before(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            raise ValueError(
                f"warnings must be a list[str], got {type(value).__name__}"
            )
        return list(value)

    @field_validator("warnings", mode="after")
    @classmethod
    def _validate_warnings(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(value, field_name="warnings")

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> dict[str, ScalarMetadataValue]:
        if value is None:
            return {}
        return _validate_scalar_metadata(value)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> Self:
        if self.column_count != len(self.suggestions):
            raise ValueError(
                "column_count must equal len(suggestions) "
                f"(got {self.column_count} != {len(self.suggestions)})"
            )

        suggestion_columns = [item.column for item in self.suggestions]
        suggestion_set = set(suggestion_columns)

        for field_name, names in (
            ("target_candidates", self.target_candidates),
            ("recommended_feature_columns", self.recommended_feature_columns),
            ("timestamp_candidates", self.timestamp_candidates),
            ("identifier_candidates", self.identifier_candidates),
            ("excluded_candidates", self.excluded_candidates),
            ("review_required_columns", self.review_required_columns),
            ("numeric_columns", self.numeric_columns),
            ("nonnumeric_columns", self.nonnumeric_columns),
        ):
            missing = [name for name in names if name not in suggestion_set]
            if missing:
                raise ValueError(
                    f"{field_name} contains columns absent from suggestions: {missing}"
                )

        numeric_set = set(self.numeric_columns)
        for name in self.recommended_feature_columns:
            if name not in numeric_set:
                raise ValueError(
                    "recommended_feature_columns must be a subset of "
                    f"numeric_columns; {name!r} is not numeric"
                )

        recommended_set = set(self.recommended_feature_columns)
        for name in self.excluded_candidates:
            if name in recommended_set:
                raise ValueError(
                    "excluded_candidates must not appear in "
                    f"recommended_feature_columns: {name!r}"
                )
        for name in self.identifier_candidates:
            if name in recommended_set:
                raise ValueError(
                    "identifier_candidates must not appear in "
                    f"recommended_feature_columns: {name!r}"
                )
        for name in self.timestamp_candidates:
            if name in recommended_set:
                raise ValueError(
                    "timestamp_candidates must not appear in "
                    f"recommended_feature_columns: {name!r}"
                )

        return self


def resolve_active_feature_columns(
    recommended_feature_columns: Sequence[str],
    *,
    selected_target: str | None = None,
    selected_timestamp: str | None = None,
    selected_identifiers: Sequence[str] = (),
    selected_excluded: Sequence[str] = (),
) -> list[str]:
    """Remove user-selected non-feature roles from a recommended feature list.

    Preserves the original recommended order. Does not mutate inputs.
    """
    blocked: set[str] = set()
    if selected_target is not None:
        blocked.add(selected_target)
    if selected_timestamp is not None:
        blocked.add(selected_timestamp)
    blocked.update(selected_identifiers)
    blocked.update(selected_excluded)
    return [name for name in recommended_feature_columns if name not in blocked]


class AutomaticColumnConfigurator:
    """Propose column configuration candidates from structural CSV metadata.

    Analyzes dtype, null counts, cardinality, uniqueness, monotonicity, and
    safe name heuristics only. Never mutates the input frame, never parses
    datetime strings, and never finalizes target, controllability, or
    constraints.
    """

    def __init__(
        self,
        *,
        identifier_unique_ratio: float = 0.95,
        near_unique_ratio: float = 0.98,
        minimum_target_name_priority: int = 1,
    ) -> None:
        self._identifier_unique_ratio = _require_unit_interval_float(
            identifier_unique_ratio,
            field_name="identifier_unique_ratio",
        )
        self._near_unique_ratio = _require_unit_interval_float(
            near_unique_ratio,
            field_name="near_unique_ratio",
        )
        if self._near_unique_ratio < self._identifier_unique_ratio:
            raise ValueError(
                "near_unique_ratio must be >= identifier_unique_ratio "
                f"(got {self._near_unique_ratio} < {self._identifier_unique_ratio})"
            )
        self._minimum_target_name_priority = _require_strict_int_ge(
            minimum_target_name_priority,
            field_name="minimum_target_name_priority",
            minimum=0,
        )

    def analyze(self, frame: pl.DataFrame) -> UiColumnConfigurationReport:
        """Analyze a Polars DataFrame and return column-configuration suggestions.

        Args:
            frame: Non-empty Polars DataFrame. Values are never modified.

        Returns:
            A validated ``UiColumnConfigurationReport``.

        Raises:
            TypeError: If ``frame`` is not a Polars DataFrame.
            ValueError: If ``frame`` has fewer than one row or one column.
        """
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(
                f"frame must be a polars DataFrame, got {type(frame).__name__}"
            )
        if frame.height < 1:
            raise ValueError("frame must contain at least one row")
        if frame.width < 1:
            raise ValueError("frame must contain at least one column")

        row_count = int(frame.height)
        suggestions: list[UiColumnSuggestion] = []
        warnings: list[str] = []

        for column in frame.columns:
            series = frame.get_column(column)
            suggestion = self._suggest_column(series=series, row_count=row_count)
            suggestions.append(suggestion)

        target_candidates = self._ordered_target_candidates(suggestions)
        identifier_candidates = [
            item.column
            for item in suggestions
            if item.category is UiColumnSuggestionCategory.IDENTIFIER_CANDIDATE
        ]
        timestamp_candidates = [
            item.column
            for item in suggestions
            if item.category is UiColumnSuggestionCategory.TIMESTAMP_CANDIDATE
        ]
        review_required_columns = [
            item.column
            for item in suggestions
            if item.category is UiColumnSuggestionCategory.REVIEW_REQUIRED
        ]
        numeric_columns = [item.column for item in suggestions if item.numeric]
        nonnumeric_columns = [
            item.column for item in suggestions if not item.numeric
        ]

        identifier_set = set(identifier_candidates)
        timestamp_set = set(timestamp_candidates)
        high_confidence_targets = {
            item.column
            for item in suggestions
            if item.category is UiColumnSuggestionCategory.TARGET_CANDIDATE
            and item.priority >= self._minimum_target_name_priority
        }

        recommended_feature_columns: list[str] = []
        for item in suggestions:
            name = item.column
            if item.category is not UiColumnSuggestionCategory.FEATURE_CANDIDATE:
                continue
            if not item.numeric or item.constant or item.null_count == row_count:
                continue
            if name in identifier_set or name in timestamp_set:
                continue
            if name in high_confidence_targets:
                continue
            if self._is_unsupported_dtype(item.dtype):
                continue
            recommended_feature_columns.append(name)

        # Explicit excluded UI defaults only. Identifier / timestamp / target
        # candidates are omitted from recommended features automatically but are
        # not pre-selected as user-submitted excluded columns.
        recommended_set = set(recommended_feature_columns)
        automatic_role_set = (
            identifier_set | timestamp_set | set(target_candidates)
        )
        excluded_candidates: list[str] = []
        for item in suggestions:
            name = item.column
            if name in recommended_set or name in automatic_role_set:
                continue
            if item.category in {
                UiColumnSuggestionCategory.IDENTIFIER_CANDIDATE,
                UiColumnSuggestionCategory.TIMESTAMP_CANDIDATE,
                UiColumnSuggestionCategory.TARGET_CANDIDATE,
                UiColumnSuggestionCategory.REVIEW_REQUIRED,
            }:
                continue
            should_exclude = (
                item.category is UiColumnSuggestionCategory.EXCLUDED_CANDIDATE
                or item.constant
                or item.null_count == row_count
                or self._is_unsupported_dtype(item.dtype)
                or not item.numeric
            )
            if should_exclude:
                excluded_candidates.append(name)

        if not target_candidates:
            warnings.append(
                "No target name candidates were found; the user must select a target."
            )
        if not recommended_feature_columns:
            warnings.append(
                "No numeric feature candidates were recommended; review columns manually."
            )
        for item in suggestions:
            if not item.constant:
                continue
            normalized = normalize_column_name_for_heuristic(item.column)
            if normalized not in _IDENTIFIER_NAME_TOKENS:
                continue
            warnings.append(
                f"{item.column} resembles an identifier but is constant and was "
                "not selected as an identifier."
            )

        return UiColumnConfigurationReport(
            column_count=len(suggestions),
            row_count=row_count,
            suggestions=suggestions,
            target_candidates=target_candidates,
            recommended_feature_columns=recommended_feature_columns,
            timestamp_candidates=timestamp_candidates,
            identifier_candidates=identifier_candidates,
            excluded_candidates=excluded_candidates,
            review_required_columns=review_required_columns,
            numeric_columns=numeric_columns,
            nonnumeric_columns=nonnumeric_columns,
            warnings=warnings,
            metadata={
                "identifier_unique_ratio": self._identifier_unique_ratio,
                "near_unique_ratio": self._near_unique_ratio,
                "minimum_target_name_priority": self._minimum_target_name_priority,
                "infers_final_target": False,
                "modifies_raw_data": False,
            },
        )

    def get_metadata(self) -> dict[str, ScalarMetadataValue]:
        """Return scalar capability metadata for this configurator instance."""
        return {
            "identifier_unique_ratio": self._identifier_unique_ratio,
            "near_unique_ratio": self._near_unique_ratio,
            "analyzes_dtype": True,
            "analyzes_null_count": True,
            "analyzes_cardinality": True,
            "analyzes_monotonicity": True,
            "infers_final_target": False,
            "infers_controllability": False,
            "infers_constraints": False,
            "modifies_raw_data": False,
            "performs_modeling": False,
        }

    def _suggest_column(
        self,
        *,
        series: pl.Series,
        row_count: int,
    ) -> UiColumnSuggestion:
        column = series.name
        if not isinstance(column, str) or column.strip() == "":
            raise ValueError("frame columns must have non-empty string names")

        dtype = series.dtype
        dtype_str = str(dtype)
        is_bool = dtype == pl.Boolean
        is_numeric = bool(dtype.is_numeric()) and not is_bool
        unsupported = self._is_unsupported_polars_dtype(dtype)

        null_count = int(series.null_count())
        unique_count = int(series.n_unique())
        unique_ratio = float(unique_count) / float(row_count)
        non_null = series.drop_nulls()
        unique_non_null_count = int(non_null.n_unique()) if non_null.len() > 0 else 0
        constant = non_null.len() > 0 and unique_non_null_count == 1
        all_null = null_count == row_count
        suitable_as_identifier = (
            not all_null and not constant and unique_non_null_count >= 2
        )
        suitable_as_target = (
            is_numeric and not all_null and not constant and non_null.len() > 0
            and unique_non_null_count >= 2
        )

        monotonic: bool | None
        if is_numeric and non_null.len() > 0:
            monotonic = bool(non_null.is_sorted())
        else:
            monotonic = None

        normalized = normalize_column_name_for_heuristic(column)
        reasons: list[str] = [f"dtype={dtype_str}"]
        identifier_name = normalized in _IDENTIFIER_NAME_TOKENS
        timestamp_name = normalized in _TIMESTAMP_NAME_TOKENS
        target_priority = _TARGET_NAME_PRIORITIES.get(normalized)

        if identifier_name:
            reasons.append("column name matches identifier heuristic")
        if timestamp_name:
            reasons.append("column name matches timestamp heuristic")
        if target_priority is not None:
            reasons.append("column name matches target heuristic")
        if constant:
            reasons.append("constant column")
        if all_null:
            reasons.append("all values are null")
        if is_bool:
            reasons.append("boolean column requires review")
        if unsupported:
            reasons.append("unsupported nested/object dtype")
        if is_numeric and monotonic is True:
            reasons.append("numeric values are monotonic non-decreasing")
        elif is_numeric and monotonic is False:
            reasons.append("numeric values are not monotonic non-decreasing")
        if target_priority is not None and not suitable_as_target:
            reasons.append("not suitable as supervised regression target")
        if identifier_name and not suitable_as_identifier:
            if all_null:
                reasons.append(
                    "identifier-like name but all-null; not selected as identifier"
                )
            elif constant:
                reasons.append(
                    "identifier-like name but constant; not selected as identifier"
                )
            else:
                reasons.append(
                    "identifier-like name but fewer than two distinct non-null "
                    "values; not selected as identifier"
                )

        category: UiColumnSuggestionCategory
        priority = 0
        confidence = 0.4

        if unsupported or all_null:
            if identifier_name:
                category = UiColumnSuggestionCategory.REVIEW_REQUIRED
                priority = 20
                confidence = 0.7
            else:
                category = UiColumnSuggestionCategory.EXCLUDED_CANDIDATE
                priority = 0
                confidence = 0.95
        elif identifier_name and suitable_as_identifier:
            category = UiColumnSuggestionCategory.IDENTIFIER_CANDIDATE
            priority = 50
            confidence = self._identifier_confidence(unique_ratio)
            if unique_ratio >= self._identifier_unique_ratio:
                reasons.append(
                    "unique ratio meets identifier uniqueness heuristic"
                )
        elif identifier_name:
            category = UiColumnSuggestionCategory.REVIEW_REQUIRED
            priority = 20
            confidence = 0.7
        elif timestamp_name:
            category = UiColumnSuggestionCategory.TIMESTAMP_CANDIDATE
            priority = 50
            confidence = 0.7
            if not is_numeric:
                reasons.append(
                    "timestamp name with non-numeric dtype; parsing not performed"
                )
                confidence = 0.55
            elif monotonic is True:
                confidence = 0.8
        elif target_priority is not None:
            # High-confidence target names win over constant / feature heuristics.
            category = UiColumnSuggestionCategory.TARGET_CANDIDATE
            priority = int(target_priority)
            confidence = min(0.95, 0.55 + (priority / 200.0))
        elif constant:
            category = UiColumnSuggestionCategory.EXCLUDED_CANDIDATE
            priority = 0
            confidence = 0.95
        elif is_bool:
            category = UiColumnSuggestionCategory.REVIEW_REQUIRED
            priority = 10
            confidence = 0.6
        elif not is_numeric:
            category = UiColumnSuggestionCategory.EXCLUDED_CANDIDATE
            priority = 0
            confidence = 0.85
            reasons.append("non-numeric free-text or categorical column")
        elif self._is_low_cardinality_numeric(
            unique_count=unique_count,
            row_count=row_count,
        ):
            category = UiColumnSuggestionCategory.REVIEW_REQUIRED
            priority = 5
            confidence = 0.55
            reasons.append("low-cardinality numeric; possible categorical code")
        elif (
            row_count >= 20
            and unique_ratio >= self._near_unique_ratio
            and not identifier_name
            and is_numeric
        ):
            category = UiColumnSuggestionCategory.REVIEW_REQUIRED
            priority = 5
            confidence = 0.5
            reasons.append("near-unique numeric values without identifier name")
        else:
            category = UiColumnSuggestionCategory.FEATURE_CANDIDATE
            priority = 1
            confidence = 0.75
            reasons.append("numeric non-constant feature candidate")

        return UiColumnSuggestion(
            column=column,
            dtype=dtype_str,
            category=category,
            priority=priority,
            confidence=confidence,
            reasons=reasons,
            numeric=is_numeric,
            null_count=null_count,
            unique_count=unique_count,
            unique_ratio=unique_ratio,
            constant=constant,
            suitable_as_target=suitable_as_target,
            monotonic_non_decreasing=monotonic,
        )

    def _identifier_confidence(self, unique_ratio: float) -> float:
        if unique_ratio >= self._near_unique_ratio:
            return 0.95
        if unique_ratio >= self._identifier_unique_ratio:
            return 0.9
        return 0.7

    @staticmethod
    def _ordered_target_candidates(
        suggestions: Sequence[UiColumnSuggestion],
    ) -> list[str]:
        indexed = [
            (index, item)
            for index, item in enumerate(suggestions)
            if item.category is UiColumnSuggestionCategory.TARGET_CANDIDATE
        ]
        indexed.sort(key=lambda pair: (-pair[1].priority, pair[0]))
        return [item.column for _, item in indexed]

    @staticmethod
    def _is_low_cardinality_numeric(*, unique_count: int, row_count: int) -> bool:
        # Require enough rows so tiny preview frames are not all flagged.
        if row_count < 20:
            return False
        return unique_count <= 10 and (unique_count / float(row_count)) <= 0.1

    @staticmethod
    def _is_unsupported_polars_dtype(dtype: pl.DataType) -> bool:
        if getattr(dtype, "is_nested", lambda: False)():
            return True
        return dtype == pl.Object

    @staticmethod
    def _is_unsupported_dtype(dtype_str: str) -> bool:
        lowered = dtype_str.casefold()
        return (
            lowered.startswith("list(")
            or lowered.startswith("array(")
            or lowered.startswith("struct(")
            or lowered == "object"
        )
