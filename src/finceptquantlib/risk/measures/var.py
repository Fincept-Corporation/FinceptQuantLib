"""
Value at Risk (VaR) Implementation

Provides multiple VaR methodologies:
- Historical Simulation
- Parametric (Delta-Normal, Delta-Gamma)
- Monte Carlo Simulation

Also implements:
- Expected Shortfall (Conditional VaR)
- Marginal and Component VaR
- Backtesting
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Optional, List, Callable, Dict
from enum import Enum
import math
import random
from scipy import stats


def _cholesky(A: list[list[float]]) -> list[list[float]]:
    """
    Cholesky decomposition A = L @ L.T

    Args:
        A: Symmetric positive definite matrix

    Returns:
        Lower triangular matrix L
    """
    n = len(A)
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))

            if i == j:
                L[i][j] = math.sqrt(A[i][i] - s)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]

    return L


class VaRMethod(Enum):
    """VaR calculation method."""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"


@dataclass
class VaRResult:
    """Result of VaR calculation."""
    var: float                      # Value at Risk
    expected_shortfall: float       # Expected Shortfall (CVaR)
    confidence_level: float         # Confidence level
    holding_period: int             # Holding period in days
    method: str                     # Calculation method
    distribution: Optional[list[float]] = None  # P&L distribution
    percentile_pnl: Optional[float] = None  # P&L at VaR percentile


class HistoricalVaR:
    """
    Historical Simulation VaR.

    Uses historical returns to construct P&L distribution.
    No assumption about distribution shape.

    Advantages:
    - Captures fat tails
    - Non-parametric
    - Easy to understand

    Disadvantages:
    - Requires long history
    - Slow to adapt to regime changes
    - May not capture extreme events
    """

    def __init__(
        self,
        returns: Union[List[float], list[float]],
        portfolio_value: float,
        confidence_level: float = 0.99,
        holding_period: int = 1,
    ):
        """
        Initialize Historical VaR.

        Args:
            returns: Historical returns (daily)
            portfolio_value: Current portfolio value
            confidence_level: VaR confidence level (e.g., 0.99 for 99%)
            holding_period: Holding period in days
        """
        self.returns = [float(r) for r in returns]
        self.portfolio_value = portfolio_value
        self.confidence_level = confidence_level
        self.holding_period = holding_period

    def _scale_returns(self) -> list[float]:
        """Scale returns for holding period using square root of time."""
        scale = math.sqrt(self.holding_period)
        return [r * scale for r in self.returns]

    def _percentile(self, data: list[float], pct: float) -> float:
        """Calculate percentile of a list."""
        sorted_data = sorted(data)
        n = len(sorted_data)
        idx = pct / 100 * (n - 1)
        lower = int(idx)
        upper = min(lower + 1, n - 1)
        frac = idx - lower
        return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac

    def calculate(self) -> VaRResult:
        """
        Calculate VaR and Expected Shortfall.

        Returns:
            VaRResult with VaR and ES
        """
        scaled_returns = self._scale_returns()
        pnl = [r * self.portfolio_value for r in scaled_returns]

        # VaR at confidence level
        alpha = 1 - self.confidence_level
        var_percentile = self._percentile(pnl, alpha * 100)
        var = -var_percentile  # VaR is positive

        # Expected Shortfall (average of losses beyond VaR)
        tail_losses = [p for p in pnl if p <= var_percentile]
        if len(tail_losses) > 0:
            es = -sum(tail_losses) / len(tail_losses)
        else:
            es = var

        return VaRResult(
            var=var,
            expected_shortfall=es,
            confidence_level=self.confidence_level,
            holding_period=self.holding_period,
            method="historical",
            distribution=pnl,
            percentile_pnl=var_percentile,
        )

    def rolling_var(
        self,
        window: int = 250,
    ) -> list[float]:
        """
        Calculate rolling VaR.

        Args:
            window: Rolling window size

        Returns:
            List of rolling VaR values
        """
        n = len(self.returns)
        rolling_vars = [float('nan')] * n
        alpha = 1 - self.confidence_level
        scale = math.sqrt(self.holding_period)

        for i in range(window, n):
            window_returns = self.returns[i-window:i]
            pnl = [r * self.portfolio_value * scale for r in window_returns]
            rolling_vars[i] = -self._percentile(pnl, alpha * 100)

        return rolling_vars

    def backtest(
        self,
        actual_pnl: list[float],
    ) -> Dict[str, float]:
        """
        Backtest VaR model.

        Args:
            actual_pnl: Actual P&L over period

        Returns:
            Dict with backtesting statistics
        """
        rolling_vars = self.rolling_var()

        # Find valid indices (non-nan)
        valid_indices = [i for i, v in enumerate(rolling_vars) if not math.isnan(v)]

        if len(valid_indices) == 0:
            return {"exceptions": 0, "exception_rate": 0.0}

        # Count exceptions (actual loss > VaR)
        exceptions = sum(1 for i in valid_indices if actual_pnl[i] < -rolling_vars[i])
        exception_rate = exceptions / len(valid_indices)
        expected_rate = 1 - self.confidence_level

        # Kupiec POF test
        n = len(valid_indices)
        x = exceptions
        p = 1 - self.confidence_level

        if x == 0 or x == n:
            lr_pof = float('inf')
        else:
            lr_pof = -2 * (math.log(p**x * (1-p)**(n-x)) -
                         math.log((x/n)**x * (1-x/n)**(n-x)))

        return {
            "exceptions": int(exceptions),
            "exception_rate": exception_rate,
            "expected_rate": expected_rate,
            "kupiec_lr": lr_pof,
            "kupiec_pvalue": 1 - stats.chi2.cdf(lr_pof, 1),
        }


class ParametricVaR:
    """
    Parametric (Variance-Covariance) VaR.

    Assumes returns are normally distributed.

    VaR = μ - z_α * σ

    where z_α is the critical value at confidence level α.

    Can be extended to Delta-Gamma for non-linear positions.
    """

    def __init__(
        self,
        portfolio_value: float,
        expected_return: float = 0.0,
        volatility: float = 0.0,
        returns: Optional[list[float]] = None,
        confidence_level: float = 0.99,
        holding_period: int = 1,
    ):
        """
        Initialize Parametric VaR.

        Args:
            portfolio_value: Current portfolio value
            expected_return: Expected daily return
            volatility: Daily volatility (std dev)
            returns: Historical returns (to estimate params if not provided)
            confidence_level: VaR confidence level
            holding_period: Holding period in days
        """
        self.portfolio_value = portfolio_value
        self.confidence_level = confidence_level
        self.holding_period = holding_period

        if returns is not None:
            n = len(returns)
            self.expected_return = sum(returns) / n if n > 0 else 0.0
            mean_r = self.expected_return
            self.volatility = math.sqrt(sum((r - mean_r)**2 for r in returns) / (n - 1)) if n > 1 else 0.0
        else:
            self.expected_return = expected_return
            self.volatility = volatility

    def calculate(self) -> VaRResult:
        """
        Calculate parametric VaR.

        Returns:
            VaRResult
        """
        # Scale to holding period
        mu = self.expected_return * self.holding_period
        sigma = self.volatility * math.sqrt(self.holding_period)

        # Critical value
        alpha = 1 - self.confidence_level
        z = stats.norm.ppf(alpha)

        # VaR
        var_return = -(mu + z * sigma)
        var = var_return * self.portfolio_value

        # Expected Shortfall (for normal distribution)
        es_multiplier = stats.norm.pdf(z) / alpha
        es = self.portfolio_value * (sigma * es_multiplier - mu)

        return VaRResult(
            var=var,
            expected_shortfall=es,
            confidence_level=self.confidence_level,
            holding_period=self.holding_period,
            method="parametric",
            percentile_pnl=-var,
        )

    def delta_gamma_var(
        self,
        delta: float,
        gamma: float,
        underlying_vol: float,
    ) -> VaRResult:
        """
        Delta-Gamma VaR for non-linear positions.

        Uses Cornish-Fisher expansion to adjust for gamma.

        Args:
            delta: Portfolio delta
            gamma: Portfolio gamma
            underlying_vol: Underlying asset volatility

        Returns:
            VaRResult
        """
        sigma = underlying_vol * math.sqrt(self.holding_period)
        alpha = 1 - self.confidence_level
        z = stats.norm.ppf(alpha)

        # Delta component
        var_delta = -delta * self.portfolio_value * z * sigma

        # Gamma adjustment
        var_gamma = 0.5 * gamma * self.portfolio_value * (sigma ** 2 - sigma * z)

        var = var_delta + var_gamma

        # Approximate ES
        es = var * 1.15  # Rough approximation

        return VaRResult(
            var=var,
            expected_shortfall=es,
            confidence_level=self.confidence_level,
            holding_period=self.holding_period,
            method="delta_gamma",
        )


class MonteCarloVaR:
    """
    Monte Carlo Simulation VaR.

    Generates many scenarios using simulation,
    then calculates VaR from simulated distribution.

    Advantages:
    - Flexible (any distribution, path-dependency)
    - Handles non-linear payoffs
    - Can incorporate complex dynamics

    Disadvantages:
    - Computationally intensive
    - Model dependent
    - May have simulation error
    """

    def __init__(
        self,
        portfolio_value: float,
        confidence_level: float = 0.99,
        holding_period: int = 1,
        num_simulations: int = 10000,
        seed: Optional[int] = None,
    ):
        """
        Initialize Monte Carlo VaR.

        Args:
            portfolio_value: Current portfolio value
            confidence_level: VaR confidence level
            holding_period: Holding period in days
            num_simulations: Number of simulations
            seed: Random seed for reproducibility
        """
        self.portfolio_value = portfolio_value
        self.confidence_level = confidence_level
        self.holding_period = holding_period
        self.num_simulations = num_simulations
        self.seed = seed

    def _percentile(self, data: list[float], pct: float) -> float:
        """Calculate percentile of a list."""
        sorted_data = sorted(data)
        n = len(sorted_data)
        idx = pct / 100 * (n - 1)
        lower = int(idx)
        upper = min(lower + 1, n - 1)
        frac = idx - lower
        return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac

    def calculate_gbm(
        self,
        drift: float,
        volatility: float,
    ) -> VaRResult:
        """
        Calculate VaR using Geometric Brownian Motion.

        dS = muS dt + sigmaS dW

        Args:
            drift: Annual drift (mu)
            volatility: Annual volatility (sigma)

        Returns:
            VaRResult
        """
        rng = random.Random(self.seed)

        # Daily parameters
        dt = self.holding_period / 252
        mu_daily = drift * dt
        sigma_daily = volatility * math.sqrt(dt)

        # Generate random returns
        pnl = []
        for _ in range(self.num_simulations):
            z = rng.gauss(0, 1)
            log_return = (mu_daily - 0.5 * sigma_daily**2) + sigma_daily * z
            final_value = self.portfolio_value * math.exp(log_return)
            pnl.append(final_value - self.portfolio_value)

        # Calculate VaR and ES
        alpha = 1 - self.confidence_level
        var_percentile = self._percentile(pnl, alpha * 100)
        var = -var_percentile

        tail_losses = [p for p in pnl if p <= var_percentile]
        es = -sum(tail_losses) / len(tail_losses) if len(tail_losses) > 0 else var

        return VaRResult(
            var=var,
            expected_shortfall=es,
            confidence_level=self.confidence_level,
            holding_period=self.holding_period,
            method="monte_carlo_gbm",
            distribution=pnl,
            percentile_pnl=var_percentile,
        )

    def calculate_multivariate(
        self,
        weights: list[float],
        expected_returns: list[float],
        covariance_matrix: list[list[float]],
    ) -> VaRResult:
        """
        Calculate VaR for portfolio using multivariate normal.

        Args:
            weights: Portfolio weights
            expected_returns: Expected returns for each asset
            covariance_matrix: Return covariance matrix

        Returns:
            VaRResult
        """
        rng = random.Random(self.seed)

        n_assets = len(weights)

        # Scale for holding period
        mu = [r * self.holding_period for r in expected_returns]
        cov = [[c * self.holding_period for c in row] for row in covariance_matrix]

        # Cholesky decomposition (pure Python)
        L = _cholesky(cov)

        pnl = []
        for _ in range(self.num_simulations):
            # Generate standard normal vector
            z = [rng.gauss(0, 1) for _ in range(n_assets)]

            # Correlated returns = mu + L @ z
            returns = []
            for i in range(n_assets):
                r = mu[i]
                for j in range(i + 1):
                    r += L[i][j] * z[j]
                returns.append(r)

            # Portfolio return
            port_return = sum(r * w for r, w in zip(returns, weights))
            pnl.append(port_return * self.portfolio_value)

        # Calculate VaR and ES
        alpha = 1 - self.confidence_level
        var_percentile = self._percentile(pnl, alpha * 100)
        var = -var_percentile

        tail_losses = [p for p in pnl if p <= var_percentile]
        es = -sum(tail_losses) / len(tail_losses) if len(tail_losses) > 0 else var

        return VaRResult(
            var=var,
            expected_shortfall=es,
            confidence_level=self.confidence_level,
            holding_period=self.holding_period,
            method="monte_carlo_multivariate",
            distribution=pnl,
            percentile_pnl=var_percentile,
        )

    def calculate_with_pricing(
        self,
        pricing_func: Callable[[dict], float],
        risk_factors: List[str],
        factor_scenarios: Dict[str, list[float]],
    ) -> VaRResult:
        """
        Calculate VaR with full revaluation.

        Args:
            pricing_func: Function that takes scenario dict and returns portfolio value
            risk_factors: List of risk factor names
            factor_scenarios: Dict of {factor_name: list of simulated values}

        Returns:
            VaRResult
        """
        pnl = []

        for i in range(self.num_simulations):
            scenario = {f: factor_scenarios[f][i] for f in risk_factors}
            scenario_value = pricing_func(scenario)
            pnl.append(scenario_value - self.portfolio_value)

        alpha = 1 - self.confidence_level
        var_percentile = self._percentile(pnl, alpha * 100)
        var = -var_percentile

        tail_losses = [p for p in pnl if p <= var_percentile]
        es = -sum(tail_losses) / len(tail_losses) if len(tail_losses) > 0 else var

        return VaRResult(
            var=var,
            expected_shortfall=es,
            confidence_level=self.confidence_level,
            holding_period=self.holding_period,
            method="monte_carlo_full_reval",
            distribution=pnl,
            percentile_pnl=var_percentile,
        )



# marginal_var, component_var, incremental_var are defined in
# finceptquantlib.risk.measures.advanced_var (canonical location)
