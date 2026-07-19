"""Column-role schema mapping from DatasetProfile and SchemaHints."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Self

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from process_intelligence.core.enums import ColumnRole
from process_intelligence.core.exceptions import DataValidationError
from process_intelligence.core.schemas import ColumnRoleAssignment, SchemaHints
from process_intelligence.data.profiler import ColumnProfile, DatasetProfile

_ORIGINAL_ROW_ID = "_original_row_id"

_NON_ALNUM_HANGUL = re.compile(r"[^0-9a-z\uac00-\ud7a3]+")

_USE_IN_MODEL_ROLES: frozenset[ColumnRole] = frozenset(
    {
        ColumnRole.CONTROLLABLE_PROCESS,
        ColumnRole.STATE_SENSOR,
        ColumnRole.CONTEXT,
        ColumnRole.DERIVED_FEATURE,
    }
)

_EXACT_TIME_NAMES: frozenset[str] = frozenset(
    {
        "timestamp",
        "datetime",
        "date",
        "time",
        "event time",
        "measurement time",
        "recorded at",
    }
)

_EXACT_IDENTIFIER_NAMES: frozenset[str] = frozenset(
    {
        "id",
        "uuid",
        "vin",
        "serial",
        "serial number",
    }
)

_EXACT_TARGET_NAMES: frozenset[str] = frozenset(
    {
        "target",
        "label",
        "response",
        "quality target",
        "prediction target",
    }
)

_NUMERIC_UNKNOWN_ALTERNATIVES: tuple[ColumnRole, ...] = (
    ColumnRole.CONTROLLABLE_PROCESS,
    ColumnRole.STATE_SENSOR,
    ColumnRole.TARGET_QUALITY,
)

_STRING_UNKNOWN_ALTERNATIVES: tuple[ColumnRole, ...] = (
    ColumnRole.CONTEXT,
    ColumnRole.IDENTIFIER,
)

_BOOLEAN_UNKNOWN_ALTERNATIVES: tuple[ColumnRole, ...] = (
    ColumnRole.CONTEXT,
    ColumnRole.TARGET_QUALITY,
)


def _normalize_column_name(value: str) -> str:
    """Normalize a column name for exact whole-name matching."""
    if not value:
        return ""
    folded = value.casefold()
    replaced = _NON_ALNUM_HANGUL.sub(" ", folded)
    return " ".join(replaced.split())


def _validate_unit_interval(value: object, *, field_name: str) -> float:
    """Reject bools, non-numbers, NaN, infinities, and values outside [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a finite number in [0, 1] "
            f"(bool not allowed), got {value!r}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(
            f"{field_name} must be a finite number in [0, 1], got {value!r}"
        )
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {number}")
    return number


def _use_in_model_for_role(role: ColumnRole) -> bool:
    """Return whether a role is included as a model feature input."""
    return role in _USE_IN_MODEL_ROLES


def _is_temporal_dtype(dtype: str) -> bool:
    """Return True when a profiler dtype string is temporal."""
    lowered = dtype.casefold()
    return (
        lowered == "date"
        or lowered.startswith("datetime")
        or lowered == "time"
    )


def _is_boolean_dtype(dtype: str) -> bool:
    """Return True when a profiler dtype string is Boolean/Bool."""
    lowered = dtype.casefold()
    return lowered in {"boolean", "bool"}


def _is_string_like_dtype(dtype: str) -> bool:
    """Return True when a profiler dtype string is string-like."""
    lowered = dtype.casefold()
    return (
        lowered == "string"
        or lowered == "utf8"
        or lowered.startswith("categorical")
        or lowered.startswith("enum")
    )


def _is_numeric_dtype(dtype: str) -> bool:
    """Return True when a profiler dtype string is numeric."""
    lowered = dtype.casefold()
    return (
        lowered.startswith("int")
        or lowered.startswith("uint")
        or lowered.startswith("float")
        or lowered.startswith("decimal")
    )


def _zero_role_counts() -> dict[ColumnRole, int]:
    """Return a fresh zero-count map for all ColumnRole values."""
    return {role: 0 for role in ColumnRole}


def _dedupe_roles(roles: list[ColumnRole]) -> list[ColumnRole]:
    """Return roles without duplicates, preserving first-seen order."""
    seen: set[ColumnRole] = set()
    result: list[ColumnRole] = []
    for role in roles:
        if role not in seen:
            seen.add(role)
            result.append(role)
    return result


class ColumnRoleMappingPolicy(BaseModel):
    """Thresholds controlling automatic column-role mapping confidence."""

    minimum_auto_confidence: float = 0.80
    context_cardinality_ratio: float = 0.20
    identifier_cardinality_ratio: float = 0.90

    @field_validator(
        "minimum_auto_confidence",
        "context_cardinality_ratio",
        "identifier_cardinality_ratio",
        mode="before",
    )
    @classmethod
    def _validate_unit_interval_fields(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> float:
        field_name = info.field_name or "policy threshold"
        return _validate_unit_interval(value, field_name=field_name)

    @model_validator(mode="after")
    def _validate_cardinality_ordering(self) -> Self:
        """Require context cardinality threshold strictly below identifier."""
        if self.context_cardinality_ratio >= self.identifier_cardinality_ratio:
            raise ValueError(
                "context_cardinality_ratio must be less than "
                "identifier_cardinality_ratio"
            )
        return self


class ColumnRoleMappingResult(BaseModel):
    """Outcome of column-role mapping for an entire dataset profile."""

    assignments: list[ColumnRoleAssignment] = Field(default_factory=list)
    unresolved_columns: list[str] = Field(default_factory=list)
    low_confidence_columns: list[str] = Field(default_factory=list)
    requires_user_confirmation: bool = False
    warnings: list[str] = Field(default_factory=list)
    role_counts: dict[ColumnRole, int] = Field(default_factory=_zero_role_counts)

    @model_validator(mode="after")
    def _validate_result_consistency(self) -> Self:
        """Enforce assignment uniqueness and list/count consistency."""
        column_names = [assignment.column_name for assignment in self.assignments]
        if len(column_names) != len(set(column_names)):
            raise ValueError("assignments must not contain duplicate column_name values")

        expected_unresolved = [
            assignment.column_name
            for assignment in self.assignments
            if assignment.role == ColumnRole.UNKNOWN
        ]
        if self.unresolved_columns != expected_unresolved:
            raise ValueError(
                "unresolved_columns must exactly match UNKNOWN assignments "
                "in original column order"
            )

        assignment_name_set = set(column_names)
        for column_name in self.low_confidence_columns:
            if column_name not in assignment_name_set:
                raise ValueError(
                    f"low_confidence_columns contains unknown column "
                    f"{column_name!r}"
                )

        expected_counts = _zero_role_counts()
        for assignment in self.assignments:
            expected_counts[assignment.role] += 1

        if set(self.role_counts.keys()) != set(ColumnRole):
            raise ValueError("role_counts must include every ColumnRole key")
        for role in ColumnRole:
            if self.role_counts.get(role, -1) != expected_counts[role]:
                raise ValueError(
                    "role_counts must aggregate assignments by ColumnRole"
                )

        # Preserve ColumnRole declaration order in role_counts.
        ordered_counts = {
            role: expected_counts[role] for role in ColumnRole
        }
        self.role_counts = ordered_counts
        return self


class ColumnRoleMapper:
    """Map dataset columns to ColumnRole assignments using schema hints."""

    def __init__(
        self,
        policy: ColumnRoleMappingPolicy | None = None,
    ) -> None:
        """Create a mapper with an isolated deep-copied policy.

        Args:
            policy: Mapping thresholds. When ``None``, defaults are used.
                The provided policy is deep-copied so later mutations do not
                affect this mapper.

        Raises:
            TypeError: If ``policy`` is not ``None`` or a
                ``ColumnRoleMappingPolicy``.
        """
        if policy is None:
            self._policy = ColumnRoleMappingPolicy()
        elif not isinstance(policy, ColumnRoleMappingPolicy):
            raise TypeError(
                "policy must be ColumnRoleMappingPolicy, "
                f"got {type(policy).__name__}"
            )
        else:
            self._policy = policy.model_copy(deep=True)

    def map_roles(
        self,
        dataset_profile: DatasetProfile,
        schema_hints: SchemaHints,
        *,
        overrides: Mapping[str, ColumnRole] | None = None,
    ) -> ColumnRoleMappingResult:
        """Assign column roles from a profile, schema hints, and overrides.

        Args:
            dataset_profile: Lightweight dataset profile. Not modified.
            schema_hints: Industry schema hints. Not modified.
            overrides: Optional exact column-name to role overrides.
                Not modified and not retained on the mapper.

        Returns:
            A ``ColumnRoleMappingResult`` with per-column assignments.

        Raises:
            TypeError: If an argument has an invalid type.
            DataValidationError: If profile columns are duplicated or
                overrides are invalid.
        """
        if not isinstance(dataset_profile, DatasetProfile):
            raise TypeError(
                "dataset_profile must be DatasetProfile, "
                f"got {type(dataset_profile).__name__}"
            )
        if not isinstance(schema_hints, SchemaHints):
            raise TypeError(
                "schema_hints must be SchemaHints, "
                f"got {type(schema_hints).__name__}"
            )

        column_names = [column.name for column in dataset_profile.columns]
        if len(column_names) != len(set(column_names)):
            raise DataValidationError(
                "dataset_profile.columns must not contain duplicate column names"
            )

        resolved_overrides = self._validate_overrides(
            overrides,
            column_names=column_names,
        )

        canonical_index = self._build_canonical_index(schema_hints)
        synonym_index = self._build_synonym_index(schema_hints)

        assignments: list[ColumnRoleAssignment] = []
        warnings: list[str] = []
        has_synonym_conflict = False

        for column in dataset_profile.columns:
            assignment, warning = self._map_single_column(
                column,
                dataset_profile=dataset_profile,
                overrides=resolved_overrides,
                canonical_index=canonical_index,
                synonym_index=synonym_index,
            )
            assignments.append(assignment)
            if warning is not None:
                has_synonym_conflict = True
                if warning not in warnings:
                    warnings.append(warning)

        unresolved_columns = [
            assignment.column_name
            for assignment in assignments
            if assignment.role == ColumnRole.UNKNOWN
        ]
        low_confidence_columns = [
            assignment.column_name
            for assignment in assignments
            if assignment.role != ColumnRole.UNKNOWN
            and assignment.confidence < self._policy.minimum_auto_confidence
        ]

        requires_user_confirmation = bool(
            unresolved_columns
            or low_confidence_columns
            or has_synonym_conflict
        )

        role_counts = _zero_role_counts()
        for assignment in assignments:
            role_counts[assignment.role] += 1

        return ColumnRoleMappingResult(
            assignments=assignments,
            unresolved_columns=unresolved_columns,
            low_confidence_columns=low_confidence_columns,
            requires_user_confirmation=requires_user_confirmation,
            warnings=warnings,
            role_counts=role_counts,
        )

    def _validate_overrides(
        self,
        overrides: Mapping[str, ColumnRole] | None,
        *,
        column_names: list[str],
    ) -> dict[str, ColumnRole]:
        """Validate and copy overrides without mutating the input mapping."""
        if overrides is None:
            return {}

        if not isinstance(overrides, Mapping):
            raise TypeError(
                "overrides must be a Mapping[str, ColumnRole] or None, "
                f"got {type(overrides).__name__}"
            )

        column_name_set = set(column_names)
        resolved: dict[str, ColumnRole] = {}
        normalized_to_key: dict[str, str] = {}

        for key, value in overrides.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"override key must be str, got {type(key).__name__}"
                )
            if not key.strip():
                raise DataValidationError(
                    "override key must be a non-empty non-whitespace string"
                )
            if not isinstance(value, ColumnRole):
                raise TypeError(
                    "override value must be ColumnRole, "
                    f"got {type(value).__name__}"
                )
            if key == _ORIGINAL_ROW_ID:
                raise DataValidationError(
                    "overrides must not include reserved column "
                    f"{_ORIGINAL_ROW_ID!r}"
                )
            if key not in column_name_set:
                raise DataValidationError(
                    f"override column {key!r} is not present in DatasetProfile"
                )

            normalized = _normalize_column_name(key)
            if normalized in normalized_to_key:
                raise DataValidationError(
                    "override keys collide after normalization: "
                    f"{normalized_to_key[normalized]!r} and {key!r}"
                )
            normalized_to_key[normalized] = key
            resolved[key] = value

        return resolved

    @staticmethod
    def _build_canonical_index(
        schema_hints: SchemaHints,
    ) -> dict[str, tuple[str, ColumnRole]]:
        """Map normalized canonical names to (canonical key, role)."""
        index: dict[str, tuple[str, ColumnRole]] = {}
        for canonical_name, role in schema_hints.default_roles.items():
            normalized = _normalize_column_name(canonical_name)
            if normalized and normalized not in index:
                index[normalized] = (canonical_name, role)
        return index

    @staticmethod
    def _build_synonym_index(
        schema_hints: SchemaHints,
    ) -> dict[str, list[tuple[str, ColumnRole]]]:
        """Map normalized synonyms to candidate (canonical, role) pairs."""
        index: dict[str, list[tuple[str, ColumnRole]]] = {}
        for canonical_name, synonyms in schema_hints.synonyms.items():
            if canonical_name not in schema_hints.default_roles:
                continue
            role = schema_hints.default_roles[canonical_name]
            for synonym in synonyms:
                normalized = _normalize_column_name(synonym)
                if not normalized:
                    continue
                candidates = index.setdefault(normalized, [])
                pair = (canonical_name, role)
                if pair not in candidates:
                    candidates.append(pair)
        return index

    def _map_single_column(
        self,
        column: ColumnProfile,
        *,
        dataset_profile: DatasetProfile,
        overrides: Mapping[str, ColumnRole],
        canonical_index: Mapping[str, tuple[str, ColumnRole]],
        synonym_index: Mapping[str, list[tuple[str, ColumnRole]]],
    ) -> tuple[ColumnRoleAssignment, str | None]:
        """Map one column and optionally return a synonym-conflict warning."""
        column_name = column.name

        if column_name in overrides:
            role = overrides[column_name]
            return (
                ColumnRoleAssignment(
                    column_name=column_name,
                    role=role,
                    confidence=1.0,
                    evidence=[
                        "User explicitly assigned this column role via override."
                    ],
                    alternative_roles=[],
                    use_in_model=_use_in_model_for_role(role),
                ),
                None,
            )

        if column_name == _ORIGINAL_ROW_ID:
            return (
                ColumnRoleAssignment(
                    column_name=column_name,
                    role=ColumnRole.IDENTIFIER,
                    confidence=1.0,
                    evidence=[
                        "Reserved lineage column _original_row_id "
                        "is treated as IDENTIFIER."
                    ],
                    alternative_roles=[],
                    use_in_model=False,
                ),
                None,
            )

        normalized_name = _normalize_column_name(column_name)

        canonical_match = canonical_index.get(normalized_name)
        if canonical_match is not None:
            canonical_key, role = canonical_match
            return (
                ColumnRoleAssignment(
                    column_name=column_name,
                    role=role,
                    confidence=0.98,
                    evidence=[
                        "Matched canonical schema hint "
                        f"{canonical_key!r} exactly after normalization."
                    ],
                    alternative_roles=[],
                    use_in_model=_use_in_model_for_role(role),
                ),
                None,
            )

        synonym_candidates = synonym_index.get(normalized_name, [])
        if synonym_candidates:
            roles = _dedupe_roles([role for _, role in synonym_candidates])
            if len(roles) > 1:
                conflict_bits = [
                    f"{canonical!r}->{role.value}"
                    for canonical, role in synonym_candidates
                ]
                warning = (
                    f"Synonym mapping for column {column_name!r} is ambiguous "
                    f"across canonical names: {', '.join(conflict_bits)}."
                )
                return (
                    ColumnRoleAssignment(
                        column_name=column_name,
                        role=ColumnRole.UNKNOWN,
                        confidence=0.0,
                        evidence=[
                            "Ambiguous synonym matches with conflicting roles: "
                            + ", ".join(conflict_bits)
                        ],
                        alternative_roles=roles,
                        use_in_model=False,
                    ),
                    warning,
                )

            canonical_name, role = synonym_candidates[0]
            return (
                ColumnRoleAssignment(
                    column_name=column_name,
                    role=role,
                    confidence=0.95,
                    evidence=[
                        "Matched schema synonym for column "
                        f"{column_name!r} to canonical name "
                        f"{canonical_name!r}."
                    ],
                    alternative_roles=[],
                    use_in_model=_use_in_model_for_role(role),
                ),
                None,
            )

        if _is_temporal_dtype(column.dtype):
            return (
                ColumnRoleAssignment(
                    column_name=column_name,
                    role=ColumnRole.TIME,
                    confidence=0.90,
                    evidence=[
                        f"Inferred TIME from temporal dtype {column.dtype!r}."
                    ],
                    alternative_roles=[],
                    use_in_model=False,
                ),
                None,
            )

        if normalized_name in _EXACT_TIME_NAMES:
            return (
                ColumnRoleAssignment(
                    column_name=column_name,
                    role=ColumnRole.TIME,
                    confidence=0.88,
                    evidence=[
                        "Inferred TIME from exact temporal column name pattern."
                    ],
                    alternative_roles=[],
                    use_in_model=False,
                ),
                None,
            )

        tokens = normalized_name.split()
        if (
            normalized_name in _EXACT_IDENTIFIER_NAMES
            or (tokens and tokens[-1] == "id")
        ):
            return (
                ColumnRoleAssignment(
                    column_name=column_name,
                    role=ColumnRole.IDENTIFIER,
                    confidence=0.88,
                    evidence=[
                        "Inferred IDENTIFIER from identifier naming pattern."
                    ],
                    alternative_roles=[],
                    use_in_model=False,
                ),
                None,
            )

        if normalized_name in _EXACT_TARGET_NAMES:
            return (
                ColumnRoleAssignment(
                    column_name=column_name,
                    role=ColumnRole.TARGET_QUALITY,
                    confidence=0.85,
                    evidence=[
                        "Inferred TARGET_QUALITY from explicit target "
                        "naming pattern."
                    ],
                    alternative_roles=[],
                    use_in_model=False,
                ),
                None,
            )

        if _is_boolean_dtype(column.dtype):
            return (
                ColumnRoleAssignment(
                    column_name=column_name,
                    role=ColumnRole.CONTEXT,
                    confidence=0.80,
                    evidence=[
                        "Inferred CONTEXT from Boolean dtype; "
                        "may also be a label."
                    ],
                    alternative_roles=[ColumnRole.TARGET_QUALITY],
                    use_in_model=True,
                ),
                None,
            )

        if (
            _is_string_like_dtype(column.dtype)
            and dataset_profile.row_count > 0
        ):
            ratio = column.cardinality_ratio
            if ratio <= self._policy.context_cardinality_ratio:
                return (
                    ColumnRoleAssignment(
                        column_name=column_name,
                        role=ColumnRole.CONTEXT,
                        confidence=0.75,
                        evidence=[
                            "Inferred CONTEXT from low-cardinality string "
                            "column statistics."
                        ],
                        alternative_roles=[ColumnRole.TARGET_QUALITY],
                        use_in_model=True,
                    ),
                    None,
                )
            if ratio >= self._policy.identifier_cardinality_ratio:
                return (
                    ColumnRoleAssignment(
                        column_name=column_name,
                        role=ColumnRole.IDENTIFIER,
                        confidence=0.70,
                        evidence=[
                            "Inferred IDENTIFIER from high-cardinality string "
                            "column statistics."
                        ],
                        alternative_roles=[ColumnRole.CONTEXT],
                        use_in_model=False,
                    ),
                    None,
                )

        return (
            self._unknown_assignment(column),
            None,
        )

    @staticmethod
    def _unknown_assignment(column: ColumnProfile) -> ColumnRoleAssignment:
        """Build an UNKNOWN assignment with dtype-based alternatives."""
        if _is_numeric_dtype(column.dtype):
            alternatives = list(_NUMERIC_UNKNOWN_ALTERNATIVES)
        elif _is_string_like_dtype(column.dtype):
            alternatives = list(_STRING_UNKNOWN_ALTERNATIVES)
        elif _is_boolean_dtype(column.dtype):
            alternatives = list(_BOOLEAN_UNKNOWN_ALTERNATIVES)
        else:
            alternatives = []

        return ColumnRoleAssignment(
            column_name=column.name,
            role=ColumnRole.UNKNOWN,
            confidence=0.0,
            evidence=[
                "Insufficient evidence to assign a column role automatically."
            ],
            alternative_roles=alternatives,
            use_in_model=False,
        )
