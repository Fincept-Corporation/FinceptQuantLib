"""
Machine Learning Subpackage
===========================

Reorganized into domain-based modules:
- modeling: General ML (preprocessing, clustering, regression, feature engineering)
- credit_risk: Credit scoring and validation
- markets: Financial markets ML (curves, portfolio optimization)

Provides a comprehensive suite of machine learning tools for
quantitative finance, credit risk modelling, portfolio optimization,
and model validation. All implementations are pure Python with no
external dependencies beyond the standard library.
"""

# Base classes and infrastructure
from finceptquantlib.ml._base import (
    Classifier,
    Clusterer,
    Estimator,
    FitResult,
    ModelType,
    PredictResult,
    Transformer,
)

# Evaluation metrics
from finceptquantlib.ml._metrics import (
    accuracy,
    adjusted_r_squared,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r_squared,
    recall_score,
    root_mean_squared_error,
)

# Subpackage re-exports
from finceptquantlib.ml.modeling import *  # noqa: F401,F403
from finceptquantlib.ml.credit_risk import *  # noqa: F401,F403
from finceptquantlib.ml.markets import *  # noqa: F401,F403

__all__ = [
    # Base
    "Estimator",
    "Classifier",
    "Transformer",
    "Clusterer",
    "FitResult",
    "PredictResult",
    "ModelType",
    # Metrics
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
    "r_squared",
    "adjusted_r_squared",
    "accuracy",
    "log_loss",
    "confusion_matrix",
    "precision_score",
    "recall_score",
    "f1_score",
]

# Extend __all__ with subpackage exports
from finceptquantlib.ml.modeling import __all__ as _modeling_all
from finceptquantlib.ml.credit_risk import __all__ as _credit_risk_all
from finceptquantlib.ml.markets import __all__ as _markets_all

for _submodule_all in [_modeling_all, _credit_risk_all, _markets_all]:
    __all__.extend([item for item in _submodule_all if item not in __all__])
