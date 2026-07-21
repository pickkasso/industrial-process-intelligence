"""Public exports for the diagnosis package (Step 8A–8D)."""

from process_intelligence.diagnosis.ensemble import (
    DiagnosisEnsembleConfig,
    DiagnosisEnsembleDiagnoser,
    EnsembleFactorStatistic,
)
from process_intelligence.diagnosis.enums import DiagnosisMethod, DiagnosisScope
from process_intelligence.diagnosis.protocols import BaseRootCauseDiagnoser
from process_intelligence.diagnosis.residual_association import (
    ResidualAssociationConfig,
    ResidualAssociationDiagnoser,
    ResidualFeatureStatistic,
)
from process_intelligence.diagnosis.robust_group_comparison import (
    RobustFeatureStatistic,
    RobustGroupComparisonConfig,
    RobustGroupComparisonDiagnoser,
)
from process_intelligence.diagnosis.schemas import (
    DiagnosisBatchResult,
    DiagnosisOutcome,
    DiagnosisRequest,
    DiagnosisResult,
)

__all__ = [
    "BaseRootCauseDiagnoser",
    "DiagnosisBatchResult",
    "DiagnosisEnsembleConfig",
    "DiagnosisEnsembleDiagnoser",
    "DiagnosisMethod",
    "DiagnosisOutcome",
    "DiagnosisRequest",
    "DiagnosisResult",
    "DiagnosisScope",
    "EnsembleFactorStatistic",
    "ResidualAssociationConfig",
    "ResidualAssociationDiagnoser",
    "ResidualFeatureStatistic",
    "RobustFeatureStatistic",
    "RobustGroupComparisonConfig",
    "RobustGroupComparisonDiagnoser",
]
