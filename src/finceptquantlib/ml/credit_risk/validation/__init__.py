"""
ML Model Validation Subpackage

Comprehensive tools for validating machine learning models in
production and regulatory contexts. Includes:

- **Backtesting**: Cross-validated out-of-sample evaluation.
- **Stability**: PSI/CSI for drift detection.
- **Interpretability**: SHAP, permutation importance, PDP.
- **Discrimination**: ROC, KS, Gini, lift, gains.
- **Calibration**: Brier score, Hosmer-Lemeshow, Spiegelhalter.

All implementations are pure Python with no external dependencies
beyond the standard library.
"""

from finceptquantlib.ml.credit_risk.validation.backtesting import (
    BacktestResult,
    MLBacktester,
)
from finceptquantlib.ml.credit_risk.validation.calibration_metrics import (
    CalibrationReport,
    brier_score,
    brier_skill_score,
    calibration_curve,
    full_calibration_report,
    hosmer_lemeshow,
    spiegelhalter_test,
)
from finceptquantlib.ml.credit_risk.validation.discrimination_metrics import (
    DiscriminationResult,
    accuracy_ratio,
    auc_roc,
    cap_curve,
    full_discrimination_report,
    gains_chart,
    gini_coefficient,
    ks_statistic,
    lift_curve,
    roc_curve,
)
from finceptquantlib.ml.credit_risk.validation.interpretability import (
    PDPResult,
    PermutationImportanceResult,
    SHAPResult,
    kernel_shap,
    partial_dependence,
    permutation_importance,
    tree_shap,
)
from finceptquantlib.ml.credit_risk.validation.stability import (
    StabilityResult,
    characteristic_stability_index,
    population_stability_index,
)

__all__ = [
    # Backtesting
    "BacktestResult",
    "MLBacktester",
    # Stability
    "StabilityResult",
    "population_stability_index",
    "characteristic_stability_index",
    # Interpretability
    "SHAPResult",
    "PermutationImportanceResult",
    "PDPResult",
    "kernel_shap",
    "tree_shap",
    "permutation_importance",
    "partial_dependence",
    # Discrimination
    "DiscriminationResult",
    "ks_statistic",
    "gini_coefficient",
    "auc_roc",
    "roc_curve",
    "cap_curve",
    "accuracy_ratio",
    "full_discrimination_report",
    "lift_curve",
    "gains_chart",
    # Calibration
    "CalibrationReport",
    "brier_score",
    "brier_skill_score",
    "hosmer_lemeshow",
    "calibration_curve",
    "spiegelhalter_test",
    "full_calibration_report",
]
