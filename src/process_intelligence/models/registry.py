"""In-memory model registry for ModelSpec and estimator factories (Step 6A)."""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable
from typing import Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.exceptions import DataValidationError, ProcessIntelligenceError
from process_intelligence.core.protocols import BaseAnalysisModel, BaseIndustryProfile
from process_intelligence.core.schemas import ModelSpec

ModelFactory: TypeAlias = Callable[[], BaseAnalysisModel]
"""Zero-argument callable that returns a ``BaseAnalysisModel`` instance."""

_SOURCE_RANK: dict[str, int] = {
    "industry_profile": 0,
    "registry": 1,
}

_REASON_FACTORY = "factory not registered"
_REASON_MISSING_DEPS = "missing optional dependencies"
_REASON_TIME_BUDGET = "time budget exceeded"


def _normalize_key(value: str) -> str:
    """Strip surrounding whitespace and casefold a non-empty string key."""
    return value.strip().casefold()


def _validate_non_empty_str(value: object, *, field_name: str) -> str:
    """Validate a non-empty string used as a registry lookup key."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty or whitespace-only")
    return value


def _validate_task(task: object) -> AnalysisTask:
    """Validate that ``task`` is an ``AnalysisTask`` member."""
    if not isinstance(task, AnalysisTask):
        raise TypeError(f"task must be AnalysisTask, got {type(task).__name__}")
    return task


def _validate_replace(replace: object) -> bool:
    """Validate that ``replace`` is a real ``bool`` (not a truthy stand-in)."""
    if not isinstance(replace, bool):
        raise TypeError(f"replace must be bool, got {type(replace).__name__}")
    return replace


def _validate_time_budget_seconds(value: object) -> float | None:
    """Validate an optional positive finite time budget in seconds."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            "time_budget_seconds must be a finite number > 0 "
            f"(bool not allowed), got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise DataValidationError(
            f"time_budget_seconds must be a finite number > 0, got {value!r}"
        )
    if number <= 0.0:
        raise DataValidationError(
            f"time_budget_seconds must be > 0, got {number}"
        )
    return number


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    """Return unique values preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _missing_optional_dependencies(optional_dependencies: list[str]) -> list[str]:
    """Return uninstalled optional package names without importing them."""
    missing: list[str] = []
    seen: set[str] = set()
    for package_name in optional_dependencies:
        if package_name in seen:
            continue
        seen.add(package_name)
        if importlib.util.find_spec(package_name) is None:
            missing.append(package_name)
    return missing


def _spec_identity(spec: ModelSpec) -> tuple[AnalysisTask, str]:
    """Return the registry identity for a model specification."""
    return (spec.task, _normalize_key(spec.name))


class ModelCandidateStatus(BaseModel):
    """Availability status for a single model candidate after registry inspection."""

    spec: ModelSpec
    source: Literal["registry", "industry_profile"]
    available: bool
    factory_registered: bool
    missing_optional_dependencies: list[str] = Field(default_factory=list)
    unavailable_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_status_consistency(self) -> ModelCandidateStatus:
        """Enforce availability invariants and reject duplicate reason lists."""
        if len(self.missing_optional_dependencies) != len(
            set(self.missing_optional_dependencies)
        ):
            raise ValueError("missing_optional_dependencies must not contain duplicates")
        if len(self.unavailable_reasons) != len(set(self.unavailable_reasons)):
            raise ValueError("unavailable_reasons must not contain duplicates")

        if self.available:
            if not self.factory_registered:
                raise ValueError(
                    "available=True requires factory_registered=True"
                )
            if self.missing_optional_dependencies:
                raise ValueError(
                    "available=True requires missing_optional_dependencies to be empty"
                )
            if self.unavailable_reasons:
                raise ValueError(
                    "available=True requires unavailable_reasons to be empty"
                )
        elif not self.unavailable_reasons:
            raise ValueError(
                "available=False requires at least one unavailable_reason"
            )

        return self


class ModelRegistry:
    """In-memory registry of model specifications and estimator factories."""

    def __init__(self) -> None:
        """Create an empty registry with isolated instance state."""
        self._specs: dict[tuple[AnalysisTask, str], ModelSpec] = {}
        self._factories: dict[str, ModelFactory] = {}

    def register_factory(
        self,
        estimator_key: str,
        factory: ModelFactory,
        *,
        replace: bool = False,
    ) -> None:
        """Register a zero-argument factory for an estimator key.

        Args:
            estimator_key: Estimator key used by ``ModelSpec.estimator_key``.
            factory: Callable that returns a ``BaseAnalysisModel`` instance.
            replace: When ``True``, replace an existing factory for the same
                normalized key.

        Raises:
            TypeError: If ``estimator_key``, ``factory``, or ``replace`` has an
                invalid type.
            ValueError: If ``estimator_key`` is empty or whitespace-only.
            DataValidationError: If the key is already registered and
                ``replace`` is ``False``.
        """
        validated_key = _validate_non_empty_str(estimator_key, field_name="estimator_key")
        if not callable(factory):
            raise TypeError(f"factory must be callable, got {type(factory).__name__}")
        _validate_replace(replace)

        key = _normalize_key(validated_key)
        if key in self._factories and not replace:
            raise DataValidationError(
                f"Duplicate factory registration for estimator_key {validated_key!r}"
            )
        self._factories[key] = factory

    def unregister_factory(self, estimator_key: str) -> ModelFactory:
        """Remove and return the factory registered for ``estimator_key``.

        Args:
            estimator_key: Estimator key to remove. Matching ignores case and
                surrounding whitespace.

        Returns:
            The previously registered factory callable.

        Raises:
            TypeError: If ``estimator_key`` is not a string.
            ValueError: If ``estimator_key`` is empty or whitespace-only.
            KeyError: If no factory is registered for the key.
        """
        validated_key = _validate_non_empty_str(estimator_key, field_name="estimator_key")
        key = _normalize_key(validated_key)
        try:
            return self._factories.pop(key)
        except KeyError as exc:
            raise KeyError(validated_key) from exc

    def register(self, spec: ModelSpec, *, replace: bool = False) -> None:
        """Register a model specification.

        Args:
            spec: Model specification to register. A deep copy is stored.
            replace: When ``True``, replace an existing spec with the same task
                and normalized name while preserving registration order.

        Raises:
            TypeError: If ``spec`` is not a ``ModelSpec`` or ``replace`` is not
                a ``bool``.
            DataValidationError: If a duplicate identity is registered without
                ``replace=True``.
        """
        if not isinstance(spec, ModelSpec):
            raise TypeError(f"spec must be ModelSpec, got {type(spec).__name__}")
        _validate_replace(replace)

        key = _spec_identity(spec)
        if key in self._specs and not replace:
            raise DataValidationError(
                f"Duplicate model registration for task={spec.task!r}, "
                f"name={spec.name!r}"
            )
        self._specs[key] = spec.model_copy(deep=True)

    def unregister(self, task: AnalysisTask, model_name: str) -> ModelSpec:
        """Remove and return a deep copy of a registered model specification.

        Args:
            task: Analysis task of the registered specification.
            model_name: Model name to remove. Matching ignores case and
                surrounding whitespace.

        Returns:
            A deep copy of the removed ``ModelSpec``.

        Raises:
            TypeError: If ``task`` or ``model_name`` has an invalid type.
            ValueError: If ``model_name`` is empty or whitespace-only.
            KeyError: If no matching specification is registered.
        """
        validated_task = _validate_task(task)
        validated_name = _validate_non_empty_str(model_name, field_name="model_name")
        key = (validated_task, _normalize_key(validated_name))
        try:
            removed = self._specs.pop(key)
        except KeyError as exc:
            raise KeyError((validated_task, validated_name)) from exc
        return removed.model_copy(deep=True)

    def get(self, task: AnalysisTask, model_name: str) -> ModelSpec:
        """Return a deep copy of a registered model specification.

        Args:
            task: Analysis task of the registered specification.
            model_name: Model name to look up. Matching ignores case and
                surrounding whitespace.

        Returns:
            A deep copy of the registered ``ModelSpec``.

        Raises:
            TypeError: If ``task`` or ``model_name`` has an invalid type.
            ValueError: If ``model_name`` is empty or whitespace-only.
            KeyError: If no matching specification is registered.
        """
        validated_task = _validate_task(task)
        validated_name = _validate_non_empty_str(model_name, field_name="model_name")
        key = (validated_task, _normalize_key(validated_name))
        try:
            return self._specs[key].model_copy(deep=True)
        except KeyError as exc:
            raise KeyError((validated_task, validated_name)) from exc

    def contains(self, task: AnalysisTask, model_name: str) -> bool:
        """Return whether a model specification is registered.

        Args:
            task: Analysis task of the registered specification.
            model_name: Model name to look up. Matching ignores case and
                surrounding whitespace.

        Returns:
            ``True`` if registered, otherwise ``False``.

        Raises:
            TypeError: If ``task`` or ``model_name`` has an invalid type.
            ValueError: If ``model_name`` is empty or whitespace-only.
        """
        validated_task = _validate_task(task)
        validated_name = _validate_non_empty_str(model_name, field_name="model_name")
        key = (validated_task, _normalize_key(validated_name))
        return key in self._specs

    def list_specs(
        self,
        task: AnalysisTask | None = None,
    ) -> tuple[ModelSpec, ...]:
        """Return registered specifications in registration order.

        Args:
            task: When provided, only specifications for this task are returned.
                When ``None``, all specifications are returned.

        Returns:
            Deep copies of registered specifications as a tuple.

        Raises:
            TypeError: If ``task`` is neither ``None`` nor an ``AnalysisTask``.
        """
        if task is not None and not isinstance(task, AnalysisTask):
            raise TypeError(
                f"task must be AnalysisTask or None, got {type(task).__name__}"
            )

        specs: list[ModelSpec] = []
        for registered in self._specs.values():
            if task is None or registered.task is task:
                specs.append(registered.model_copy(deep=True))
        return tuple(specs)

    def inspect_candidates(
        self,
        task: AnalysisTask,
        *,
        industry_profile: BaseIndustryProfile | None = None,
        time_budget_seconds: float | None = None,
    ) -> tuple[ModelCandidateStatus, ...]:
        """Inspect merged model candidates for a task, including unavailable ones.

        Args:
            task: Analysis task whose candidates are inspected.
            industry_profile: Optional industry profile whose default candidates
                are merged and may override registry identities.
            time_budget_seconds: Optional positive finite budget used to exclude
                candidates whose ``time_budget_seconds`` exceeds this value.

        Returns:
            Candidate status objects sorted by priority, source precedence, and
            original source-internal order.

        Raises:
            TypeError: If ``task``, ``industry_profile``, or
                ``time_budget_seconds`` has an invalid type.
            DataValidationError: If profile candidates are malformed or contain
                duplicate identities within a single source, or if the time
                budget is non-finite or non-positive.
        """
        validated_task = _validate_task(task)
        if industry_profile is not None and not isinstance(
            industry_profile, BaseIndustryProfile
        ):
            raise TypeError(
                "industry_profile must be BaseIndustryProfile or None, "
                f"got {type(industry_profile).__name__}"
            )
        budget = _validate_time_budget_seconds(time_budget_seconds)

        registry_specs = self._registry_specs_for_task(validated_task)
        profile_specs = self._profile_specs_for_task(validated_task, industry_profile)
        merged = self._merge_candidates(registry_specs, profile_specs)

        statuses: list[ModelCandidateStatus] = []
        for spec, source, _order in merged:
            statuses.append(
                self._build_candidate_status(
                    spec=spec,
                    source=source,
                    time_budget_seconds=budget,
                )
            )
        return tuple(statuses)

    def get_candidates(
        self,
        task: AnalysisTask,
        *,
        industry_profile: BaseIndustryProfile | None = None,
        time_budget_seconds: float | None = None,
    ) -> tuple[ModelSpec, ...]:
        """Return available model candidates for a task.

        Args:
            task: Analysis task whose candidates are requested.
            industry_profile: Optional industry profile merged into candidates.
            time_budget_seconds: Optional positive finite budget filter.

        Returns:
            Deep copies of available ``ModelSpec`` values in inspection order.
            Returns an empty tuple when no candidate is available.
        """
        statuses = self.inspect_candidates(
            task,
            industry_profile=industry_profile,
            time_budget_seconds=time_budget_seconds,
        )
        return tuple(
            status.spec.model_copy(deep=True)
            for status in statuses
            if status.available
        )

    def instantiate(self, spec: ModelSpec) -> BaseAnalysisModel:
        """Create a model instance from a registered factory.

        Args:
            spec: Model specification whose ``estimator_key`` selects the
                factory. The input specification is not modified.

        Returns:
            A new ``BaseAnalysisModel`` instance produced by the factory.

        Raises:
            TypeError: If ``spec`` is not a ``ModelSpec``.
            ProcessIntelligenceError: If the factory is missing or required
                optional dependencies are not installed.
            DataValidationError: If the factory return value is not a
                ``BaseAnalysisModel`` instance.
        """
        if not isinstance(spec, ModelSpec):
            raise TypeError(f"spec must be ModelSpec, got {type(spec).__name__}")

        factory_key = _normalize_key(spec.estimator_key)
        factory = self._factories.get(factory_key)
        if factory is None:
            raise ProcessIntelligenceError(
                f"No factory registered for estimator_key {spec.estimator_key!r}"
            )

        missing = _missing_optional_dependencies(list(spec.optional_dependencies))
        if missing:
            raise ProcessIntelligenceError(
                "Missing optional dependencies for "
                f"{spec.name!r}: {', '.join(missing)}"
            )

        model = factory()
        if not isinstance(model, BaseAnalysisModel):
            raise DataValidationError(
                "factory must return a BaseAnalysisModel instance, "
                f"got {type(model).__name__}"
            )
        return model

    def __len__(self) -> int:
        """Return the number of registered model specifications."""
        return len(self._specs)

    def _registry_specs_for_task(self, task: AnalysisTask) -> list[ModelSpec]:
        """Return deep copies of registry specs for ``task`` in order."""
        specs: list[ModelSpec] = []
        seen: set[tuple[AnalysisTask, str]] = set()
        for registered in self._specs.values():
            if registered.task is not task:
                continue
            identity = _spec_identity(registered)
            if identity in seen:
                raise DataValidationError(
                    f"Duplicate registry candidate identity for "
                    f"task={task!r}, name={registered.name!r}"
                )
            seen.add(identity)
            specs.append(registered.model_copy(deep=True))
        return specs

    def _profile_specs_for_task(
        self,
        task: AnalysisTask,
        industry_profile: BaseIndustryProfile | None,
    ) -> list[ModelSpec]:
        """Validate and return deep copies of industry-profile candidates."""
        if industry_profile is None:
            return []

        candidates = industry_profile.get_default_model_candidates(task)
        if not isinstance(candidates, list):
            raise DataValidationError(
                "industry_profile.get_default_model_candidates must return a list, "
                f"got {type(candidates).__name__}"
            )

        specs: list[ModelSpec] = []
        seen: set[tuple[AnalysisTask, str]] = set()
        for index, item in enumerate(candidates):
            if not isinstance(item, ModelSpec):
                raise DataValidationError(
                    "industry_profile candidates must be ModelSpec instances, "
                    f"got {type(item).__name__} at index {index}"
                )
            if item.task is not task:
                raise DataValidationError(
                    "industry_profile candidate task mismatch: "
                    f"expected {task!r}, got {item.task!r} for name={item.name!r}"
                )
            identity = _spec_identity(item)
            if identity in seen:
                raise DataValidationError(
                    f"Duplicate industry_profile candidate identity for "
                    f"task={task!r}, name={item.name!r}"
                )
            seen.add(identity)
            specs.append(item.model_copy(deep=True))
        return specs

    def _merge_candidates(
        self,
        registry_specs: list[ModelSpec],
        profile_specs: list[ModelSpec],
    ) -> list[tuple[ModelSpec, Literal["registry", "industry_profile"], int]]:
        """Merge registry and profile candidates with profile identity override."""
        profile_identities = {_spec_identity(spec) for spec in profile_specs}
        merged: list[tuple[ModelSpec, Literal["registry", "industry_profile"], int]] = []

        for order, spec in enumerate(profile_specs):
            merged.append((spec, "industry_profile", order))

        for order, spec in enumerate(registry_specs):
            if _spec_identity(spec) in profile_identities:
                continue
            merged.append((spec, "registry", order))

        merged.sort(
            key=lambda item: (
                item[0].priority,
                _SOURCE_RANK[item[1]],
                item[2],
            )
        )
        return merged

    def _build_candidate_status(
        self,
        *,
        spec: ModelSpec,
        source: Literal["registry", "industry_profile"],
        time_budget_seconds: float | None,
    ) -> ModelCandidateStatus:
        """Build an availability status for one merged candidate."""
        factory_registered = _normalize_key(spec.estimator_key) in self._factories
        missing = _missing_optional_dependencies(list(spec.optional_dependencies))
        time_budget_exceeded = (
            time_budget_seconds is not None
            and spec.time_budget_seconds is not None
            and spec.time_budget_seconds > time_budget_seconds
        )

        reasons: list[str] = []
        if not factory_registered:
            reasons.append(_REASON_FACTORY)
        if missing:
            reasons.append(_REASON_MISSING_DEPS)
        if time_budget_exceeded:
            reasons.append(_REASON_TIME_BUDGET)

        available = not reasons
        return ModelCandidateStatus(
            spec=spec.model_copy(deep=True),
            source=source,
            available=available,
            factory_registered=factory_registered,
            missing_optional_dependencies=list(missing),
            unavailable_reasons=list(reasons),
        )
