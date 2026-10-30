"""
Portfolio Analytics Module

Provides comprehensive portfolio optimization and analytics:
- Mean-Variance Optimization (Markowitz)
- Black-Litterman Model (views + equilibrium)
- Risk Parity / Equal Risk Contribution
- Factor Models and Risk Decomposition
- Performance Attribution

Each module includes:
- Analytical solutions (where available)
- Numerical optimization
- Risk metrics and constraints
"""

from finceptquantlib.portfolio.optimization import (
    MeanVarianceOptimizer,
    PortfolioConstraints,
    OptimizationResult,
    efficient_frontier,
    min_variance_portfolio,
    max_sharpe_portfolio,
    target_return_portfolio,
)

from finceptquantlib.portfolio.black_litterman import (
    BlackLitterman,
    View,
    ViewType,
    equilibrium_returns,
)

from finceptquantlib.portfolio.risk_parity import (
    RiskParity,
    RiskParityResult,
    EqualRiskContribution,
    HierarchicalRiskParity,
    risk_parity_weights,
    hrp_weights,
    inverse_volatility_weights,
)

from finceptquantlib.portfolio.risk_metrics import (
    PortfolioRisk,
    calculate_var,
    parametric_cvar,
    marginal_risk_contribution,
    component_risk_contribution,
    incremental_var,
    tracking_error,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    diversification_ratio,
    effective_number_of_bets,
    calculate_portfolio_risk,
)

__all__ = [
    # Mean-Variance
    "MeanVarianceOptimizer",
    "PortfolioConstraints",
    "OptimizationResult",
    "efficient_frontier",
    "min_variance_portfolio",
    "max_sharpe_portfolio",
    "target_return_portfolio",
    # Black-Litterman
    "BlackLitterman",
    "View",
    "ViewType",
    "equilibrium_returns",
    # Risk Parity
    "RiskParity",
    "RiskParityResult",
    "EqualRiskContribution",
    "HierarchicalRiskParity",
    "risk_parity_weights",
    "hrp_weights",
    "inverse_volatility_weights",
    # Risk Metrics
    "PortfolioRisk",
    "calculate_var",
    "parametric_cvar",
    "marginal_risk_contribution",
    "component_risk_contribution",
    "incremental_var",
    "tracking_error",
    "information_ratio",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "diversification_ratio",
    "effective_number_of_bets",
    "calculate_portfolio_risk",
]
