"""Factory helpers for default production analysis workflows (Step 11B).

Creates a fresh ``IndustrialProcessAnalysisWorkflow`` with public default
dependencies. Does not retain fitted models, estimators, or workflow instances
between calls.
"""

from __future__ import annotations

from process_intelligence.workflow import IndustrialProcessAnalysisWorkflow


def create_default_analysis_workflow() -> IndustrialProcessAnalysisWorkflow:
    """Create a new default production analysis workflow instance.

    Returns:
        A newly constructed ``IndustrialProcessAnalysisWorkflow`` using the
        public constructor defaults (loader, routers, registries, and residual
        components). No global singleton or fitted-model store is retained.
    """
    return IndustrialProcessAnalysisWorkflow()
