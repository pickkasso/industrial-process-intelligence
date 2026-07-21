"""Unit tests for create_default_analysis_workflow (Step 11B)."""

from __future__ import annotations

import inspect

from process_intelligence.ui import create_default_analysis_workflow
from process_intelligence.ui import workflow_factory as workflow_factory_module
from process_intelligence.workflow import IndustrialProcessAnalysisWorkflow


def test_returns_real_workflow() -> None:
    workflow = create_default_analysis_workflow()
    assert isinstance(workflow, IndustrialProcessAnalysisWorkflow)


def test_new_instance_each_call() -> None:
    first = create_default_analysis_workflow()
    second = create_default_analysis_workflow()
    assert first is not second


def test_no_raw_estimator_in_factory_source() -> None:
    source = inspect.getsource(workflow_factory_module)
    assert "sklearn" not in source
    assert "Estimator" not in source
    assert "RandomForest" not in source
    assert "IsolationForest" not in source


def test_no_private_registry_access() -> None:
    source = inspect.getsource(workflow_factory_module)
    assert "_registry" not in source
    assert "._" not in source.replace("create_default_analysis_workflow", "")


def test_no_fitted_model_cache() -> None:
    source = inspect.getsource(workflow_factory_module.create_default_analysis_workflow)
    assert "lru_cache" not in source
    assert "cache_resource" not in source
    assert "@cache" not in source
    assert "_CACHE" not in source
    assert "_fitted" not in source


def test_default_workflow_get_metadata_usable() -> None:
    workflow = create_default_analysis_workflow()
    metadata = workflow.get_metadata()
    assert metadata["loads_raw_csv"] is True
    assert metadata["performs_recommendation"] is True
    for value in metadata.values():
        assert value is None or isinstance(value, (str, int, float, bool))


def test_factory_has_no_module_state() -> None:
    names = [
        name
        for name, value in vars(workflow_factory_module).items()
        if not name.startswith("_")
        and name not in {
            "IndustrialProcessAnalysisWorkflow",
            "create_default_analysis_workflow",
            "annotations",
        }
        and not inspect.ismodule(value)
        and not inspect.isfunction(value)
        and not inspect.isclass(value)
    ]
    assert names == []
