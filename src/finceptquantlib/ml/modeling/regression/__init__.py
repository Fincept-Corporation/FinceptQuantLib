"""
Regression Subpackage

Provides linear, tree-based, ensemble, and credit risk regression
models. All implementations are pure Python with full type
annotations.

Modules
-------
linear
    OLS, Ridge, Lasso, and Elastic Net regression.
trees
    Decision tree and gradient boosted regressors.
ensemble
    Random forest regressor.
lgd_models
    Beta regression and two-stage LGD models for IFRS 9 ECL.
ead_models
    Credit Conversion Factor model for EAD estimation.
"""

from finceptquantlib.ml.modeling.regression.linear import (
    ElasticNet,
    LassoRegression,
    LinearRegressionResult,
    OLSRegression,
    RidgeRegression,
)
from finceptquantlib.ml.modeling.regression.trees import (
    DecisionTreeRegressor,
    GradientBoostedRegressor,
    TreeNode,
    TreeResult,
)
from finceptquantlib.ml.modeling.regression.ensemble import (
    RandomForestRegressor,
)
from finceptquantlib.ml.modeling.regression.lgd_models import (
    BetaRegressionLGD,
    LGDModelResult,
    TwoStageLGD,
)
from finceptquantlib.ml.modeling.regression.ead_models import (
    CCFModel,
    EADModelResult,
)

__all__ = [
    # linear.py
    "LinearRegressionResult",
    "OLSRegression",
    "RidgeRegression",
    "LassoRegression",
    "ElasticNet",
    # trees.py
    "TreeNode",
    "TreeResult",
    "DecisionTreeRegressor",
    "GradientBoostedRegressor",
    # ensemble.py
    "RandomForestRegressor",
    # lgd_models.py
    "LGDModelResult",
    "BetaRegressionLGD",
    "TwoStageLGD",
    # ead_models.py
    "EADModelResult",
    "CCFModel",
]
