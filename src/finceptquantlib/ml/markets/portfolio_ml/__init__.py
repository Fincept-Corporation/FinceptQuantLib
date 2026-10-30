"""
Portfolio ML Subpackage

Machine learning tools for portfolio construction and optimization:

- **covariance**: Robust covariance estimation (Ledoit-Wolf shrinkage,
  Minimum Covariance Determinant)
- **factor_models**: Statistical (PCA) and cross-sectional
  (Fama-MacBeth) factor models
- **deep_portfolio**: Neural network-based portfolio optimization
- **transaction_cost**: Transaction cost prediction and Almgren-Chriss
  optimal execution
"""

from finceptquantlib.ml.markets.portfolio_ml.covariance import (
    LedoitWolfShrinkage,
    MinimumCovarianceDeterminant,
    ShrinkageCovarianceResult,
)

from finceptquantlib.ml.markets.portfolio_ml.factor_models import (
    CrossSectionalFactorModel,
    FactorModelResult,
    StatisticalFactorModel,
)

from finceptquantlib.ml.markets.portfolio_ml.deep_portfolio import (
    DeepPortfolioResult,
    NeuralPortfolioOptimizer,
)

from finceptquantlib.ml.markets.portfolio_ml.transaction_cost import (
    TransactionCostModel,
    TransactionCostResult,
)

__all__ = [
    # Covariance estimation
    "LedoitWolfShrinkage",
    "MinimumCovarianceDeterminant",
    "ShrinkageCovarianceResult",
    # Factor models
    "StatisticalFactorModel",
    "CrossSectionalFactorModel",
    "FactorModelResult",
    # Deep portfolio
    "NeuralPortfolioOptimizer",
    "DeepPortfolioResult",
    # Transaction costs
    "TransactionCostModel",
    "TransactionCostResult",
]
