"""
Credit Scoring Subpackage

Provides tools for credit risk model development, validation,
and monitoring:

- **Logistic Regression**: IRLS and gradient descent with full
  statistical inference (standard errors, p-values, AIC, BIC).
- **WoE Binning**: Weight of Evidence transformation with
  monotonic enforcement and Information Value computation.
- **Scorecard**: Points-based credit scorecard construction
  using the PDO (Points to Double the Odds) methodology.
- **Calibration**: Platt Scaling and Isotonic Regression for
  probability calibration.
- **Migration**: Rating transition matrix estimation, generator
  matrix computation, and cumulative default probabilities.
- **Discrimination**: KS statistic, Gini coefficient, AUC-ROC,
  CAP curve, and accuracy ratio.
- **Performance**: Comprehensive model validation report with
  discrimination, calibration, and Hosmer-Lemeshow tests.
"""

from finceptquantlib.ml.credit_risk.credit.calibration import (
    CalibrationResult,
    IsotonicRegression,
    PlattScaling,
)
from finceptquantlib.ml.credit_risk.credit.discrimination import (
    DiscriminationResult,
    accuracy_ratio,
    auc_roc,
    cap_curve,
    full_discrimination_report,
    gini_coefficient,
    ks_statistic,
    roc_curve,
)
from finceptquantlib.ml.credit_risk.credit.logistic import (
    LogisticRegression,
    LogisticRegressionResult,
)
from finceptquantlib.ml.credit_risk.credit.migration import (
    MigrationMatrixResult,
    RatingMigration,
)
from finceptquantlib.ml.credit_risk.credit.performance import (
    ModelPerformanceReport,
    evaluate_credit_model,
)
from finceptquantlib.ml.credit_risk.credit.scorecard import (
    Scorecard,
    ScorecardResult,
)
from finceptquantlib.ml.credit_risk.credit.woe import (
    BinResult,
    WoEBinner,
)

__all__ = [
    # logistic
    "LogisticRegression",
    "LogisticRegressionResult",
    # woe
    "WoEBinner",
    "BinResult",
    # scorecard
    "Scorecard",
    "ScorecardResult",
    # calibration
    "PlattScaling",
    "IsotonicRegression",
    "CalibrationResult",
    # migration
    "RatingMigration",
    "MigrationMatrixResult",
    # discrimination
    "DiscriminationResult",
    "ks_statistic",
    "gini_coefficient",
    "auc_roc",
    "roc_curve",
    "cap_curve",
    "accuracy_ratio",
    "full_discrimination_report",
    # performance
    "ModelPerformanceReport",
    "evaluate_credit_model",
]
