"""
Portfolio Risk Metrics

Implements comprehensive risk analytics:
- Value at Risk (VaR): Parametric, Historical, Monte Carlo
- Conditional VaR (CVaR/Expected Shortfall)
- Risk contributions: Marginal, Component, Incremental
- Other metrics: Tracking error, Diversification ratio

References:
- Jorion, P. (2007). "Value at Risk."
- Rockafellar, R.T. & Uryasev, S. (2000). "Optimization of Conditional Value-at-Risk."
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import math

from finceptquantlib.core.utilities.distributions import norm_ppf


@dataclass
class PortfolioRisk:
    """
    Comprehensive portfolio risk metrics.

    Attributes:
        volatility: Portfolio standard deviation (annualized)
        var_95: 95% Value at Risk
        var_99: 99% Value at Risk
        cvar_95: 95% Conditional VaR (Expected Shortfall)
        cvar_99: 99% Conditional VaR
        max_drawdown: Maximum drawdown (if returns provided)
        sharpe_ratio: Sharpe ratio (if returns provided)
        sortino_ratio: Sortino ratio (if returns provided)
        tracking_error: Tracking error (if benchmark provided)
        information_ratio: Information ratio (if benchmark provided)
        diversification_ratio: Diversification ratio
        effective_n: Effective number of independent bets
    """
    volatility: float
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    diversification_ratio: float = 1.0
    effective_n: float = 0.0


def calculate_var(
    weights: List[float],
    covariance_matrix: List[List[float]],
    confidence: float = 0.95,
    horizon: float = 1.0,
    portfolio_value: float = 1.0,
    method: str = "parametric"
) -> float:
    """
    Calculate Value at Risk.

    VaR = Portfolio Value × z_α × σ_p × √horizon

    where z_α is the quantile of standard normal.

    Args:
        weights: Portfolio weights
        covariance_matrix: Asset covariance matrix
        confidence: Confidence level (0.95 or 0.99 typical)
        horizon: Time horizon in years
        portfolio_value: Total portfolio value
        method: "parametric" (assumes normal), "historical", or "monte_carlo"

    Returns:
        VaR amount
    """
    n = len(weights)

    # Portfolio variance
    var = 0.0
    for i in range(n):
        for j in range(n):
            var += weights[i] * weights[j] * covariance_matrix[i][j]

    vol = math.sqrt(max(0.0, var))

    # Z-score for confidence level
    z = norm_ppf(confidence)

    # VaR
    return portfolio_value * z * vol * math.sqrt(horizon)


def parametric_cvar(
    weights: List[float],
    covariance_matrix: List[List[float]],
    confidence: float = 0.95,
    horizon: float = 1.0,
    portfolio_value: float = 1.0
) -> float:
    """
    Calculate Conditional VaR (Expected Shortfall).

    For normal distribution:
    CVaR = Portfolio Value × σ_p × φ(z_α) / (1-α) × √horizon

    where φ is the standard normal PDF.

    CVaR represents the expected loss given that loss exceeds VaR.

    Args:
        weights: Portfolio weights
        covariance_matrix: Asset covariance matrix
        confidence: Confidence level
        horizon: Time horizon
        portfolio_value: Total portfolio value

    Returns:
        CVaR amount
    """
    n = len(weights)

    var = sum(
        weights[i] * weights[j] * covariance_matrix[i][j]
        for i in range(n) for j in range(n)
    )
    vol = math.sqrt(max(0.0, var))

    z = norm_ppf(confidence)
    phi_z = math.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)

    # CVaR for normal
    cvar = portfolio_value * vol * phi_z / (1 - confidence) * math.sqrt(horizon)

    return cvar


def marginal_risk_contribution(
    weights: List[float],
    covariance_matrix: List[List[float]]
) -> List[float]:
    """
    Calculate Marginal Risk Contribution (MRC) for each asset.

    MRC_i = ∂σ_p / ∂w_i = (Σw)_i / σ_p

    This measures the sensitivity of portfolio risk to a change in weight.

    Args:
        weights: Portfolio weights
        covariance_matrix: Asset covariance matrix

    Returns:
        List of marginal risk contributions
    """
    n = len(weights)

    # Portfolio variance
    sigma_w = [sum(covariance_matrix[i][j] * weights[j] for j in range(n)) for i in range(n)]
    var = sum(weights[i] * sigma_w[i] for i in range(n))
    vol = math.sqrt(var) if var > 0 else 1e-10

    # MRC
    return [sigma_w[i] / vol for i in range(n)]


def component_risk_contribution(
    weights: List[float],
    covariance_matrix: List[List[float]]
) -> Tuple[List[float], List[float]]:
    """
    Calculate Component Risk Contribution (CRC).

    CRC_i = w_i × MRC_i = w_i × (Σw)_i / σ_p

    The sum of CRCs equals portfolio volatility.

    Args:
        weights: Portfolio weights
        covariance_matrix: Asset covariance matrix

    Returns:
        Tuple of (absolute CRCs, percentage CRCs)
    """
    mrc = marginal_risk_contribution(weights, covariance_matrix)

    # Absolute contribution
    crc_abs = [weights[i] * mrc[i] for i in range(len(weights))]

    # Percentage of total risk
    total = sum(crc_abs)
    if total > 0:
        crc_pct = [c / total for c in crc_abs]
    else:
        crc_pct = [1.0 / len(weights)] * len(weights)

    return crc_abs, crc_pct


def incremental_var(
    weights: List[float],
    covariance_matrix: List[List[float]],
    asset_index: int,
    confidence: float = 0.95,
    weight_change: float = 0.01
) -> float:
    """
    Calculate Incremental VaR for adding weight to an asset.

    Measures how VaR changes when weight of asset_index increases.

    Args:
        weights: Current portfolio weights
        covariance_matrix: Asset covariance matrix
        asset_index: Index of asset to analyze
        confidence: Confidence level
        weight_change: Size of weight increment

    Returns:
        Change in VaR per unit weight change
    """
    # Current VaR
    var_current = calculate_var(weights, covariance_matrix, confidence)

    # New weights with increment
    new_weights = weights.copy()
    new_weights[asset_index] += weight_change

    # Renormalize
    total = sum(new_weights)
    new_weights = [w / total for w in new_weights]

    var_new = calculate_var(new_weights, covariance_matrix, confidence)

    return (var_new - var_current) / weight_change


def tracking_error(
    portfolio_returns: List[float],
    benchmark_returns: List[float]
) -> float:
    """
    Calculate Tracking Error.

    TE = std(r_p - r_b)

    Args:
        portfolio_returns: List of portfolio returns
        benchmark_returns: List of benchmark returns

    Returns:
        Annualized tracking error
    """
    if len(portfolio_returns) != len(benchmark_returns):
        raise ValueError("Return series must have same length")
    if len(portfolio_returns) < 2:
        raise ValueError("Need at least 2 return observations for tracking error")

    # Active returns
    active = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]

    # Standard deviation (sample variance)
    mean_active = sum(active) / len(active)
    variance = sum((a - mean_active) ** 2 for a in active) / (len(active) - 1)

    return math.sqrt(variance * 252)  # Annualize assuming daily


def information_ratio(
    portfolio_returns: List[float],
    benchmark_returns: List[float]
) -> float:
    """
    Calculate Information Ratio.

    IR = mean(r_p - r_b) / TE

    Args:
        portfolio_returns: List of portfolio returns
        benchmark_returns: List of benchmark returns

    Returns:
        Information ratio
    """
    active = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
    mean_active = sum(active) / len(active) * 252  # Annualized

    te = tracking_error(portfolio_returns, benchmark_returns)

    return mean_active / te if te > 0 else 0


def max_drawdown(returns: List[float]) -> float:
    """
    Calculate Maximum Drawdown.

    MDD = max(peak - trough) / peak

    Args:
        returns: List of returns

    Returns:
        Maximum drawdown as positive number
    """
    # Build cumulative returns
    cum = [1.0]
    for r in returns:
        cum.append(cum[-1] * (1 + r))

    # Find max drawdown
    peak = cum[0]
    max_dd = 0

    for value in cum:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        max_dd = max(max_dd, dd)

    return max_dd


def sharpe_ratio(
    returns: List[float],
    rf_rate: float = 0.0
) -> float:
    """
    Calculate Sharpe Ratio.

    SR = (mean(r) - rf) / std(r)

    Args:
        returns: List of returns
        rf_rate: Risk-free rate (annualized)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    mean_daily = sum(returns) / len(returns)
    mean_ret = mean_daily * 252  # Annualize
    variance = sum((r - mean_daily) ** 2 for r in returns) / (len(returns) - 1)
    vol = math.sqrt(variance * 252)

    return (mean_ret - rf_rate) / vol if vol > 0 else 0


def sortino_ratio(
    returns: List[float],
    target_return: float = 0.0
) -> float:
    """
    Calculate Sortino Ratio.

    Uses downside deviation instead of standard deviation.

    SR = (mean(r) - target) / downside_dev

    Args:
        returns: List of returns
        target_return: Target return (typically 0 or rf)

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    daily_target = target_return / 252

    mean_ret = sum(returns) / len(returns) * 252

    # Downside deviation (sample variance with N-1)
    downside = [min(r - daily_target, 0) ** 2 for r in returns]
    downside_var = sum(downside) / (len(downside) - 1)
    downside_dev = math.sqrt(downside_var * 252)

    return (mean_ret - target_return) / downside_dev if downside_dev > 0 else 0


def diversification_ratio(
    weights: List[float],
    covariance_matrix: List[List[float]]
) -> float:
    """
    Calculate Diversification Ratio.

    DR = sum(w_i × σ_i) / σ_p

    Higher ratio = more diversified (1 = fully concentrated).

    Args:
        weights: Portfolio weights
        covariance_matrix: Asset covariance matrix

    Returns:
        Diversification ratio (>= 1)
    """
    n = len(weights)

    # Individual volatilities
    vols = [math.sqrt(covariance_matrix[i][i]) for i in range(n)]
    weighted_vols = sum(weights[i] * vols[i] for i in range(n))

    # Portfolio volatility
    var = sum(
        weights[i] * weights[j] * covariance_matrix[i][j]
        for i in range(n) for j in range(n)
    )
    port_vol = math.sqrt(max(0.0, var))

    return weighted_vols / port_vol if port_vol > 0 else 1.0


def effective_number_of_bets(risk_contributions: List[float]) -> float:
    """
    Calculate Effective Number of Bets (ENB).

    ENB = 1 / sum(rc_i^2)

    where rc_i is the risk contribution percentage of asset i.

    Args:
        risk_contributions: List of percentage risk contributions (should sum to 1)

    Returns:
        Effective N (between 1 and N)
    """
    sum_sq = sum(rc ** 2 for rc in risk_contributions)
    return 1.0 / sum_sq if sum_sq > 0 else len(risk_contributions)


def calculate_portfolio_risk(
    weights: List[float],
    covariance_matrix: List[List[float]],
    returns: Optional[List[List[float]]] = None,
    benchmark_returns: Optional[List[float]] = None,
    rf_rate: float = 0.0
) -> PortfolioRisk:
    """
    Calculate comprehensive portfolio risk metrics.

    Args:
        weights: Portfolio weights
        covariance_matrix: Asset covariance matrix
        returns: Optional historical returns (assets × time)
        benchmark_returns: Optional benchmark returns
        rf_rate: Risk-free rate

    Returns:
        PortfolioRisk dataclass with all metrics
    """
    n = len(weights)

    # Portfolio volatility
    var = sum(
        weights[i] * weights[j] * covariance_matrix[i][j]
        for i in range(n) for j in range(n)
    )
    vol = math.sqrt(max(0.0, var))

    # VaR and CVaR
    var_95 = calculate_var(weights, covariance_matrix, 0.95)
    var_99 = calculate_var(weights, covariance_matrix, 0.99)
    cvar_95 = parametric_cvar(weights, covariance_matrix, 0.95)
    cvar_99 = parametric_cvar(weights, covariance_matrix, 0.99)

    # Diversification
    div_ratio = diversification_ratio(weights, covariance_matrix)
    _, rc_pct = component_risk_contribution(weights, covariance_matrix)
    eff_n = effective_number_of_bets(rc_pct)

    # Additional metrics if returns provided
    sr = 0.0
    sortino = 0.0
    mdd = 0.0
    te = 0.0
    ir = 0.0

    if returns is not None and len(returns) > 0:
        # Compute portfolio returns
        n_periods = len(returns[0]) if returns else 0
        port_returns = [
            sum(weights[i] * returns[i][t] for i in range(n))
            for t in range(n_periods)
        ]

        sr = sharpe_ratio(port_returns, rf_rate)
        sortino = sortino_ratio(port_returns, rf_rate)
        mdd = max_drawdown(port_returns)

        if benchmark_returns is not None:
            te = tracking_error(port_returns, benchmark_returns)
            ir = information_ratio(port_returns, benchmark_returns)

    return PortfolioRisk(
        volatility=vol,
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        max_drawdown=mdd,
        sharpe_ratio=sr,
        sortino_ratio=sortino,
        tracking_error=te,
        information_ratio=ir,
        diversification_ratio=div_ratio,
        effective_n=eff_n
    )
