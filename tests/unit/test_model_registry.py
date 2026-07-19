"""Unit tests for ModelRegistry and ModelCandidateStatus (Step 6A)."""

from __future__ import annotations

from typing import Self

import numpy as np
import pytest
from pydantic import ValidationError

from process_intelligence.core.enums import AnalysisTask
from process_intelligence.core.exceptions import DataValidationError, ProcessIntelligenceError
from process_intelligence.core.protocols import (
    BaseAnalysisModel,
    BaseIndustryProfile,
    DataFrameLike,
    SeriesLike,
)
from process_intelligence.core.schemas import (
    DatasetMetadata,
    ExplanationResult,
    IndustryScore,
    ModelEvaluation,
    ModelMetadata,
    ModelSpec,
    PreprocessingRules,
    SchemaHints,
    ValidationIssue,
    VariableConstraint,
)
from process_intelligence.models import (
    ModelCandidateStatus,
    ModelRegistry,
)
from process_intelligence.models.registry import ModelFactory


class StubAnalysisModel(BaseAnalysisModel):
    """Minimal BaseAnalysisModel stub for registry factory tests."""

    def __init__(self, label: str = "stub") -> None:
        self.label = label

    def fit(self, X: DataFrameLike, y: SeriesLike | None = None) -> Self:
        _ = (X, y)
        return self

    def predict(self, X: DataFrameLike) -> np.ndarray:
        _ = X
        return np.asarray([0.0])

    def evaluate(
        self,
        X: DataFrameLike,
        y: SeriesLike | None = None,
    ) -> ModelEvaluation:
        _ = (X, y)
        return ModelEvaluation()

    def explain(self, X: DataFrameLike) -> ExplanationResult:
        _ = X
        return ExplanationResult(method="stub", feature_importances={})

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_name="stub",
            version="0",
            task=AnalysisTask.REGRESSION,
        )


class StubIndustryProfile(BaseIndustryProfile):
    """Industry profile stub that returns configurable model candidates."""

    def __init__(
        self,
        *,
        candidates: list[ModelSpec] | object | None = None,
        raise_on_candidates: Exception | None = None,
    ) -> None:
        self._candidates = [] if candidates is None else candidates
        self._raise_on_candidates = raise_on_candidates

    @property
    def industry_name(self) -> str:
        return "stub"

    def score_industry(self, metadata: DatasetMetadata) -> IndustryScore:
        _ = metadata
        return IndustryScore(
            industry_name="stub",
            score=0.0,
            confidence=0.0,
            evidence=[],
            uncertain_factors=[],
            requires_user_confirmation=True,
        )

    def get_schema_hints(self) -> SchemaHints:
        return SchemaHints()

    def get_preprocessing_rules(self) -> PreprocessingRules:
        return PreprocessingRules()

    def get_default_model_candidates(self, task: AnalysisTask) -> list[ModelSpec]:
        if self._raise_on_candidates is not None:
            raise self._raise_on_candidates
        _ = task
        return self._candidates  # type: ignore[return-value]

    def validate_physical_ranges(self, frame: DataFrameLike) -> list[ValidationIssue]:
        _ = frame
        return []

    def get_recommendation_constraints(self) -> list[VariableConstraint]:
        return []


def _spec(
    name: str = "ridge",
    *,
    task: AnalysisTask = AnalysisTask.REGRESSION,
    estimator_key: str = "ridge",
    optional_dependencies: list[str] | None = None,
    priority: int = 1,
    time_budget_seconds: float | None = None,
) -> ModelSpec:
    return ModelSpec(
        name=name,
        task=task,
        estimator_key=estimator_key,
        optional_dependencies=[] if optional_dependencies is None else optional_dependencies,
        priority=priority,
        time_budget_seconds=time_budget_seconds,
    )


# --- ModelCandidateStatus -------------------------------------------------


def test_available_status_creation() -> None:
    status = ModelCandidateStatus(
        spec=_spec(),
        source="registry",
        available=True,
        factory_registered=True,
    )
    assert status.available is True
    assert status.missing_optional_dependencies == []
    assert status.unavailable_reasons == []


def test_unavailable_status_creation() -> None:
    status = ModelCandidateStatus(
        spec=_spec(),
        source="industry_profile",
        available=False,
        factory_registered=False,
        unavailable_reasons=["factory not registered"],
    )
    assert status.available is False
    assert status.unavailable_reasons == ["factory not registered"]


def test_available_rejects_unregistered_factory() -> None:
    with pytest.raises(ValidationError):
        ModelCandidateStatus(
            spec=_spec(),
            source="registry",
            available=True,
            factory_registered=False,
        )


def test_available_rejects_missing_dependencies() -> None:
    with pytest.raises(ValidationError):
        ModelCandidateStatus(
            spec=_spec(),
            source="registry",
            available=True,
            factory_registered=True,
            missing_optional_dependencies=["xgboost"],
        )


def test_available_rejects_unavailable_reasons() -> None:
    with pytest.raises(ValidationError):
        ModelCandidateStatus(
            spec=_spec(),
            source="registry",
            available=True,
            factory_registered=True,
            unavailable_reasons=["factory not registered"],
        )


def test_unavailable_rejects_empty_reasons() -> None:
    with pytest.raises(ValidationError):
        ModelCandidateStatus(
            spec=_spec(),
            source="registry",
            available=False,
            factory_registered=True,
        )


def test_missing_dependency_duplicates_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelCandidateStatus(
            spec=_spec(),
            source="registry",
            available=False,
            factory_registered=True,
            missing_optional_dependencies=["xgboost", "xgboost"],
            unavailable_reasons=["missing optional dependencies"],
        )


def test_unavailable_reason_duplicates_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelCandidateStatus(
            spec=_spec(),
            source="registry",
            available=False,
            factory_registered=False,
            unavailable_reasons=["factory not registered", "factory not registered"],
        )


def test_mutable_default_lists_not_shared() -> None:
    first = ModelCandidateStatus(
        spec=_spec(),
        source="registry",
        available=True,
        factory_registered=True,
    )
    second = ModelCandidateStatus(
        spec=_spec(name="linear"),
        source="registry",
        available=True,
        factory_registered=True,
    )
    first.missing_optional_dependencies.append("xgboost")
    assert second.missing_optional_dependencies == []


def test_model_candidate_status_round_trip() -> None:
    status = ModelCandidateStatus(
        spec=_spec(),
        source="registry",
        available=False,
        factory_registered=False,
        missing_optional_dependencies=["lightgbm"],
        unavailable_reasons=["factory not registered", "missing optional dependencies"],
    )
    restored = ModelCandidateStatus.model_validate(status.model_dump())
    assert restored == status


# --- Registry basics ------------------------------------------------------


def test_empty_registry() -> None:
    registry = ModelRegistry()
    assert len(registry) == 0
    assert registry.list_specs() == ()


def test_initial_length_zero() -> None:
    assert len(ModelRegistry()) == 0


def test_registry_instance_state_isolation() -> None:
    left = ModelRegistry()
    right = ModelRegistry()
    left.register(_spec())
    assert len(left) == 1
    assert len(right) == 0


def test_register_rejects_non_model_spec() -> None:
    with pytest.raises(TypeError):
        ModelRegistry().register("not-a-spec")  # type: ignore[arg-type]


def test_register_rejects_non_bool_replace() -> None:
    with pytest.raises(TypeError):
        ModelRegistry().register(_spec(), replace=1)  # type: ignore[arg-type]


def test_register_success_and_length() -> None:
    registry = ModelRegistry()
    registry.register(_spec())
    assert len(registry) == 1


def test_register_does_not_mutate_input_spec() -> None:
    registry = ModelRegistry()
    spec = _spec(optional_dependencies=["numpy"])
    original = list(spec.optional_dependencies)
    registry.register(spec)
    spec.optional_dependencies.append("mutated")
    assert registry.get(AnalysisTask.REGRESSION, "ridge").optional_dependencies == original


def test_duplicate_task_name_rejected() -> None:
    registry = ModelRegistry()
    registry.register(_spec())
    with pytest.raises(DataValidationError):
        registry.register(_spec())


def test_duplicate_name_case_insensitive() -> None:
    registry = ModelRegistry()
    registry.register(_spec(name="Ridge"))
    with pytest.raises(DataValidationError):
        registry.register(_spec(name="ridge"))


def test_duplicate_name_whitespace_insensitive() -> None:
    registry = ModelRegistry()
    registry.register(_spec(name="ridge"))
    with pytest.raises(DataValidationError):
        registry.register(_spec(name="  ridge  "))


def test_same_name_different_task_allowed() -> None:
    registry = ModelRegistry()
    registry.register(_spec(name="model", task=AnalysisTask.REGRESSION))
    registry.register(_spec(name="model", task=AnalysisTask.CLASSIFICATION))
    assert len(registry) == 2


def test_replace_preserves_registration_order() -> None:
    registry = ModelRegistry()
    registry.register(_spec(name="a", priority=1))
    registry.register(_spec(name="b", priority=2))
    registry.register(_spec(name="c", priority=3))
    registry.register(_spec(name="b", priority=9, estimator_key="replaced"), replace=True)
    names = [spec.name for spec in registry.list_specs()]
    assert names == ["a", "b", "c"]
    assert registry.get(AnalysisTask.REGRESSION, "b").priority == 9
    assert registry.get(AnalysisTask.REGRESSION, "b").estimator_key == "replaced"


def test_new_spec_appended_last() -> None:
    registry = ModelRegistry()
    registry.register(_spec(name="a"))
    registry.register(_spec(name="b"))
    registry.register(_spec(name="c"))
    assert [spec.name for spec in registry.list_specs()] == ["a", "b", "c"]


def test_get_success_case_and_whitespace() -> None:
    registry = ModelRegistry()
    registry.register(_spec(name="Ridge"))
    assert registry.get(AnalysisTask.REGRESSION, "ridge").name == "Ridge"
    assert registry.get(AnalysisTask.REGRESSION, "  RIDGE  ").name == "Ridge"


def test_get_rejects_invalid_task() -> None:
    with pytest.raises(TypeError):
        ModelRegistry().get("REGRESSION", "ridge")  # type: ignore[arg-type]


def test_get_rejects_non_string_name() -> None:
    with pytest.raises(TypeError):
        ModelRegistry().get(AnalysisTask.REGRESSION, 1)  # type: ignore[arg-type]


def test_get_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        ModelRegistry().get(AnalysisTask.REGRESSION, "   ")


def test_get_missing_raises_key_error() -> None:
    with pytest.raises(KeyError):
        ModelRegistry().get(AnalysisTask.REGRESSION, "missing")


def test_get_return_copy_isolation() -> None:
    registry = ModelRegistry()
    registry.register(_spec(optional_dependencies=["numpy"]))
    fetched = registry.get(AnalysisTask.REGRESSION, "ridge")
    fetched.optional_dependencies.append("mutated")
    assert registry.get(AnalysisTask.REGRESSION, "ridge").optional_dependencies == [
        "numpy"
    ]


def test_contains_true_and_false() -> None:
    registry = ModelRegistry()
    registry.register(_spec())
    assert registry.contains(AnalysisTask.REGRESSION, "ridge") is True
    assert registry.contains(AnalysisTask.REGRESSION, "missing") is False


def test_unregister_success_and_order() -> None:
    registry = ModelRegistry()
    registry.register(_spec(name="a"))
    registry.register(_spec(name="b"))
    registry.register(_spec(name="c"))
    removed = registry.unregister(AnalysisTask.REGRESSION, "b")
    assert removed.name == "b"
    assert len(registry) == 2
    assert [spec.name for spec in registry.list_specs()] == ["a", "c"]


def test_unregister_missing_raises_key_error() -> None:
    with pytest.raises(KeyError):
        ModelRegistry().unregister(AnalysisTask.REGRESSION, "missing")


def test_list_specs_order_filter_and_copies() -> None:
    registry = ModelRegistry()
    registry.register(_spec(name="a", task=AnalysisTask.REGRESSION))
    registry.register(_spec(name="b", task=AnalysisTask.CLASSIFICATION))
    registry.register(_spec(name="c", task=AnalysisTask.REGRESSION))
    all_specs = registry.list_specs()
    assert isinstance(all_specs, tuple)
    assert [spec.name for spec in all_specs] == ["a", "b", "c"]
    filtered = registry.list_specs(AnalysisTask.REGRESSION)
    assert [spec.name for spec in filtered] == ["a", "c"]
    filtered[0].optional_dependencies.append("mutated")
    assert registry.get(AnalysisTask.REGRESSION, "a").optional_dependencies == []


def test_empty_list_specs_is_empty_tuple() -> None:
    assert ModelRegistry().list_specs() == ()


# --- Factory --------------------------------------------------------------


def test_register_factory_success() -> None:
    registry = ModelRegistry()
    registry.register_factory("ridge", StubAnalysisModel)
    assert callable(registry.unregister_factory("ridge"))


def test_register_factory_rejects_non_string_key() -> None:
    with pytest.raises(TypeError):
        ModelRegistry().register_factory(1, StubAnalysisModel)  # type: ignore[arg-type]


def test_register_factory_rejects_empty_key() -> None:
    with pytest.raises(ValueError):
        ModelRegistry().register_factory("  ", StubAnalysisModel)


def test_register_factory_rejects_non_callable() -> None:
    with pytest.raises(TypeError):
        ModelRegistry().register_factory("ridge", "not-callable")  # type: ignore[arg-type]


def test_register_factory_rejects_non_bool_replace() -> None:
    with pytest.raises(TypeError):
        ModelRegistry().register_factory(
            "ridge",
            StubAnalysisModel,
            replace=1,  # type: ignore[arg-type]
        )


def test_register_factory_duplicate_and_casefold() -> None:
    registry = ModelRegistry()
    registry.register_factory("Ridge", StubAnalysisModel)
    with pytest.raises(DataValidationError):
        registry.register_factory("ridge", StubAnalysisModel)


def test_register_factory_replace() -> None:
    registry = ModelRegistry()
    registry.register_factory("ridge", lambda: StubAnalysisModel("old"))
    registry.register_factory(
        "ridge",
        lambda: StubAnalysisModel("new"),
        replace=True,
    )
    model = registry.instantiate(_spec())
    assert isinstance(model, StubAnalysisModel)
    assert model.label == "new"


def test_unregister_factory_success_and_missing() -> None:
    registry = ModelRegistry()
    factory: ModelFactory = StubAnalysisModel
    registry.register_factory("ridge", factory)
    assert registry.unregister_factory("ridge") is factory
    with pytest.raises(KeyError):
        registry.unregister_factory("ridge")


def test_unregister_factory_keeps_spec() -> None:
    registry = ModelRegistry()
    registry.register(_spec())
    registry.register_factory("ridge", StubAnalysisModel)
    registry.unregister_factory("ridge")
    assert registry.contains(AnalysisTask.REGRESSION, "ridge") is True
    statuses = registry.inspect_candidates(AnalysisTask.REGRESSION)
    assert statuses[0].factory_registered is False
    assert statuses[0].available is False


def test_register_factory_does_not_call_factory() -> None:
    calls = {"count": 0}

    def factory() -> StubAnalysisModel:
        calls["count"] += 1
        return StubAnalysisModel()

    ModelRegistry().register_factory("ridge", factory)
    assert calls["count"] == 0


# --- Candidate inspection -------------------------------------------------


def test_inspect_rejects_invalid_task() -> None:
    with pytest.raises(TypeError):
        ModelRegistry().inspect_candidates("REGRESSION")  # type: ignore[arg-type]


def test_inspect_rejects_invalid_industry_profile() -> None:
    with pytest.raises(TypeError):
        ModelRegistry().inspect_candidates(
            AnalysisTask.REGRESSION,
            industry_profile="profile",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("budget", [0, -1.0])
def test_inspect_rejects_non_positive_time_budget(budget: float) -> None:
    with pytest.raises(DataValidationError):
        ModelRegistry().inspect_candidates(
            AnalysisTask.REGRESSION,
            time_budget_seconds=budget,
        )


def test_inspect_rejects_bool_time_budget() -> None:
    with pytest.raises(TypeError):
        ModelRegistry().inspect_candidates(
            AnalysisTask.REGRESSION,
            time_budget_seconds=True,  # type: ignore[arg-type]
        )


def test_inspect_rejects_nan_time_budget() -> None:
    with pytest.raises(DataValidationError):
        ModelRegistry().inspect_candidates(
            AnalysisTask.REGRESSION,
            time_budget_seconds=float("nan"),
        )


def test_inspect_rejects_infinity_time_budget() -> None:
    with pytest.raises(DataValidationError):
        ModelRegistry().inspect_candidates(
            AnalysisTask.REGRESSION,
            time_budget_seconds=float("inf"),
        )


def test_inspect_registry_candidates_and_task_filter() -> None:
    registry = ModelRegistry()
    registry.register(_spec(name="reg", task=AnalysisTask.REGRESSION))
    registry.register(_spec(name="clf", task=AnalysisTask.CLASSIFICATION))
    registry.register_factory("ridge", StubAnalysisModel)
    statuses = registry.inspect_candidates(AnalysisTask.REGRESSION)
    assert isinstance(statuses, tuple)
    assert len(statuses) == 1
    assert statuses[0].spec.name == "reg"
    assert statuses[0].source == "registry"


def test_inspect_priority_and_source_ordering() -> None:
    registry = ModelRegistry()
    registry.register(_spec(name="reg-high", priority=5))
    registry.register(_spec(name="reg-low", priority=1))
    registry.register_factory("ridge", StubAnalysisModel)
    profile = StubIndustryProfile(
        candidates=[
            _spec(name="profile-mid", priority=1, estimator_key="ridge"),
            _spec(name="profile-late", priority=1, estimator_key="ridge"),
        ]
    )
    statuses = registry.inspect_candidates(
        AnalysisTask.REGRESSION,
        industry_profile=profile,
    )
    names = [status.spec.name for status in statuses]
    # priority 1: profile-mid, profile-late (source industry first among ties with
    # registry reg-low also priority 1), then reg-high priority 5.
    assert names == ["profile-mid", "profile-late", "reg-low", "reg-high"]
    assert [status.source for status in statuses] == [
        "industry_profile",
        "industry_profile",
        "registry",
        "registry",
    ]


def test_inspect_priority_source_tie_preserves_order() -> None:
    registry = ModelRegistry()
    registry.register(_spec(name="a", priority=1))
    registry.register(_spec(name="b", priority=1))
    registry.register_factory("ridge", StubAnalysisModel)
    names = [
        status.spec.name
        for status in registry.inspect_candidates(AnalysisTask.REGRESSION)
    ]
    assert names == ["a", "b"]


def test_inspect_profile_override_same_identity() -> None:
    registry = ModelRegistry()
    registry.register(_spec(name="shared", priority=9, estimator_key="old"))
    registry.register_factory("ridge", StubAnalysisModel)
    registry.register_factory("old", StubAnalysisModel)
    profile = StubIndustryProfile(
        candidates=[_spec(name="shared", priority=1, estimator_key="ridge")]
    )
    statuses = registry.inspect_candidates(
        AnalysisTask.REGRESSION,
        industry_profile=profile,
    )
    assert len(statuses) == 1
    assert statuses[0].source == "industry_profile"
    assert statuses[0].spec.priority == 1
    assert statuses[0].spec.estimator_key == "ridge"


def test_inspect_rejects_duplicate_identity_within_profile() -> None:
    registry = ModelRegistry()
    profile = StubIndustryProfile(
        candidates=[_spec(name="dup"), _spec(name="DUP")]
    )
    with pytest.raises(DataValidationError):
        registry.inspect_candidates(
            AnalysisTask.REGRESSION,
            industry_profile=profile,
        )


def test_inspect_rejects_non_list_profile_candidates() -> None:
    registry = ModelRegistry()
    profile = StubIndustryProfile(candidates=("not", "a", "list"))  # type: ignore[arg-type]
    with pytest.raises(DataValidationError):
        registry.inspect_candidates(
            AnalysisTask.REGRESSION,
            industry_profile=profile,
        )


def test_inspect_rejects_non_model_spec_profile_candidate() -> None:
    registry = ModelRegistry()
    profile = StubIndustryProfile(candidates=["bad"])  # type: ignore[list-item]
    with pytest.raises(DataValidationError):
        registry.inspect_candidates(
            AnalysisTask.REGRESSION,
            industry_profile=profile,
        )


def test_inspect_rejects_profile_task_mismatch() -> None:
    registry = ModelRegistry()
    profile = StubIndustryProfile(
        candidates=[_spec(task=AnalysisTask.CLASSIFICATION)]
    )
    with pytest.raises(DataValidationError):
        registry.inspect_candidates(
            AnalysisTask.REGRESSION,
            industry_profile=profile,
        )


def test_inspect_propagates_profile_domain_exception() -> None:
    registry = ModelRegistry()
    profile = StubIndustryProfile(
        raise_on_candidates=DataValidationError("profile failed")
    )
    with pytest.raises(DataValidationError, match="profile failed"):
        registry.inspect_candidates(
            AnalysisTask.REGRESSION,
            industry_profile=profile,
        )


def test_inspect_factory_and_dependency_and_budget_reasons() -> None:
    registry = ModelRegistry()
    registry.register(
        _spec(
            name="heavy",
            optional_dependencies=["definitely_missing_pkg_xyz", "numpy"],
            time_budget_seconds=100.0,
            priority=1,
        )
    )
    statuses = registry.inspect_candidates(
        AnalysisTask.REGRESSION,
        time_budget_seconds=10.0,
    )
    status = statuses[0]
    assert status.available is False
    assert status.factory_registered is False
    assert status.missing_optional_dependencies == ["definitely_missing_pkg_xyz"]
    assert status.unavailable_reasons == [
        "factory not registered",
        "missing optional dependencies",
        "time budget exceeded",
    ]


def test_inspect_installed_dependency_not_missing() -> None:
    registry = ModelRegistry()
    registry.register(_spec(optional_dependencies=["numpy"]))
    registry.register_factory("ridge", StubAnalysisModel)
    status = registry.inspect_candidates(AnalysisTask.REGRESSION)[0]
    assert status.available is True
    assert status.missing_optional_dependencies == []
    assert status.unavailable_reasons == []


def test_inspect_preserves_dependency_order_and_no_crash() -> None:
    registry = ModelRegistry()
    registry.register(
        _spec(
            optional_dependencies=[
                "definitely_missing_a",
                "numpy",
                "definitely_missing_b",
                "definitely_missing_a",
            ]
        )
    )
    status = registry.inspect_candidates(AnalysisTask.REGRESSION)[0]
    assert status.missing_optional_dependencies == [
        "definitely_missing_a",
        "definitely_missing_b",
    ]


def test_inspect_includes_unavailable_and_does_not_mutate_registry() -> None:
    registry = ModelRegistry()
    registry.register(_spec(name="a"))
    before = len(registry)
    statuses = registry.inspect_candidates(AnalysisTask.REGRESSION)
    assert len(statuses) == 1
    assert statuses[0].available is False
    statuses[0].spec.optional_dependencies.append("mutated")
    assert len(registry) == before
    assert registry.get(AnalysisTask.REGRESSION, "a").optional_dependencies == []


# --- get_candidates -------------------------------------------------------


def test_get_candidates_filters_unavailable() -> None:
    registry = ModelRegistry()
    registry.register(_spec(name="ready", estimator_key="ready"))
    registry.register(
        _spec(
            name="missing-factory",
            estimator_key="absent",
            priority=0,
        )
    )
    registry.register(
        _spec(
            name="missing-dep",
            estimator_key="ready",
            optional_dependencies=["definitely_missing_pkg_xyz"],
            priority=2,
        )
    )
    registry.register(
        _spec(
            name="too-slow",
            estimator_key="ready",
            time_budget_seconds=50.0,
            priority=3,
        )
    )
    registry.register_factory("ready", StubAnalysisModel)
    candidates = registry.get_candidates(
        AnalysisTask.REGRESSION,
        time_budget_seconds=10.0,
    )
    assert isinstance(candidates, tuple)
    assert [spec.name for spec in candidates] == ["ready"]


def test_get_candidates_empty_tuple_when_none_available() -> None:
    registry = ModelRegistry()
    registry.register(_spec())
    assert registry.get_candidates(AnalysisTask.REGRESSION) == ()


def test_get_candidates_priority_and_profile_override_and_copy() -> None:
    registry = ModelRegistry()
    registry.register(_spec(name="shared", priority=5, estimator_key="ridge"))
    registry.register(_spec(name="other", priority=2, estimator_key="ridge"))
    registry.register_factory("ridge", StubAnalysisModel)
    profile = StubIndustryProfile(
        candidates=[_spec(name="shared", priority=1, estimator_key="ridge")]
    )
    candidates = registry.get_candidates(
        AnalysisTask.REGRESSION,
        industry_profile=profile,
    )
    assert [spec.name for spec in candidates] == ["shared", "other"]
    candidates[0].optional_dependencies.append("mutated")
    assert registry.get(AnalysisTask.REGRESSION, "shared").optional_dependencies == []


# --- instantiate ----------------------------------------------------------


def test_instantiate_success_and_independent_instances() -> None:
    registry = ModelRegistry()
    registry.register_factory("ridge", StubAnalysisModel)
    first = registry.instantiate(_spec())
    second = registry.instantiate(_spec())
    assert isinstance(first, StubAnalysisModel)
    assert isinstance(second, StubAnalysisModel)
    assert first is not second


def test_instantiate_rejects_non_spec() -> None:
    with pytest.raises(TypeError):
        ModelRegistry().instantiate("bad")  # type: ignore[arg-type]


def test_instantiate_missing_factory_raises_domain_error() -> None:
    with pytest.raises(ProcessIntelligenceError):
        ModelRegistry().instantiate(_spec())


def test_instantiate_missing_optional_dependency_raises_domain_error() -> None:
    registry = ModelRegistry()
    registry.register_factory("ridge", StubAnalysisModel)
    with pytest.raises(ProcessIntelligenceError):
        registry.instantiate(
            _spec(optional_dependencies=["definitely_missing_pkg_xyz"])
        )


def test_instantiate_rejects_non_model_return() -> None:
    registry = ModelRegistry()
    registry.register_factory("ridge", lambda: "not-a-model")  # type: ignore[arg-type, return-value]
    with pytest.raises(DataValidationError):
        registry.instantiate(_spec())


def test_instantiate_propagates_factory_domain_exception() -> None:
    def factory() -> StubAnalysisModel:
        raise DataValidationError("factory failed")

    registry = ModelRegistry()
    registry.register_factory("ridge", factory)
    with pytest.raises(DataValidationError, match="factory failed"):
        registry.instantiate(_spec())


def test_instantiate_does_not_mutate_registry_or_input() -> None:
    registry = ModelRegistry()
    registry.register(_spec(optional_dependencies=["numpy"]))
    registry.register_factory("ridge", StubAnalysisModel)
    spec = _spec(optional_dependencies=["numpy"])
    original = list(spec.optional_dependencies)
    before = len(registry)
    registry.instantiate(spec)
    assert len(registry) == before
    assert spec.optional_dependencies == original
