"""
Portfolio Analytics

Provides tools for portfolio-level risk management:
- Portfolio construction and management
- Aggregated risk metrics
- Diversification analysis
- Risk attribution
- Hedging optimization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import math

from ..measures.var import (
    HistoricalVaR,
    ParametricVaR,
    VaRResult,
)


class AssetClassRisk(Enum):
    """Asset class classification."""
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    FX = "fx"
    COMMODITIES = "commodities"
    CREDIT = "credit"
    DERIVATIVES = "derivatives"
    CASH = "cash"


@dataclass
class Position:
    """
    Individual portfolio position.
    """
    instrument_id: str
    quantity: float
    market_value: float
    asset_class: AssetClassRisk
    currency: str = "USD"
    delta: float = 1.0  # For linear instruments
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    duration: Optional[float] = None
    credit_spread_sens: float = 0.0
    fx_sens: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def notional(self) -> float:
        """Absolute notional exposure."""
        return abs(self.market_value)


@dataclass
class PortfolioRiskResult:
    """Comprehensive portfolio risk metrics."""
    total_market_value: float
    total_var: float
    total_expected_shortfall: float
    total_delta: float
    total_gamma: float
    total_vega: float
    total_duration: float
    diversification_ratio: float
    concentration_metrics: Dict[str, float]
    risk_by_asset_class: Dict[AssetClassRisk, float]
    risk_attribution: Dict[str, float]
    stress_results: Dict[str, float] = field(default_factory=dict)


class Portfolio:
    """
    Portfolio Manager.

    Manages a collection of positions and provides
    portfolio-level analytics.
    """

    def __init__(
        self,
        name: str = "Portfolio",
        base_currency: str = "USD",
    ):
        """
        Initialize portfolio.

        Args:
            name: Portfolio name
            base_currency: Base reporting currency
        """
        self.name = name
        self.base_currency = base_currency
        self._positions: Dict[str, Position] = {}

    def add_position(self, position: Position) -> None:
        """Add or update a position."""
        self._positions[position.instrument_id] = position

    def remove_position(self, instrument_id: str) -> None:
        """Remove a position."""
        if instrument_id in self._positions:
            del self._positions[instrument_id]

    def get_position(self, instrument_id: str) -> Optional[Position]:
        """Get a position by ID."""
        return self._positions.get(instrument_id)

    @property
    def positions(self) -> List[Position]:
        """All positions."""
        return list(self._positions.values())

    @property
    def num_positions(self) -> int:
        """Number of positions."""
        return len(self._positions)

    def total_market_value(self) -> float:
        """Total portfolio market value."""
        return sum(p.market_value for p in self.positions)

    def total_notional(self) -> float:
        """Total absolute notional."""
        return sum(p.notional for p in self.positions)

    def weights(self) -> Dict[str, float]:
        """Position weights by market value."""
        total = self.total_market_value()
        if total == 0:
            return {p.instrument_id: 0 for p in self.positions}
        return {p.instrument_id: p.market_value / total for p in self.positions}

    def by_asset_class(self) -> Dict[AssetClassRisk, List[Position]]:
        """Group positions by asset class."""
        groups: Dict[AssetClassRisk, List[Position]] = {}
        for p in self.positions:
            if p.asset_class not in groups:
                groups[p.asset_class] = []
            groups[p.asset_class].append(p)
        return groups

    def by_currency(self) -> Dict[str, List[Position]]:
        """Group positions by currency."""
        groups: Dict[str, List[Position]] = {}
        for p in self.positions:
            if p.currency not in groups:
                groups[p.currency] = []
            groups[p.currency].append(p)
        return groups

    def total_delta(self) -> float:
        """Total portfolio delta."""
        return sum(p.delta * p.market_value for p in self.positions)

    def total_gamma(self) -> float:
        """Total portfolio gamma."""
        return sum(p.gamma * p.market_value for p in self.positions)

    def total_vega(self) -> float:
        """Total portfolio vega."""
        return sum(p.vega for p in self.positions)

    def total_theta(self) -> float:
        """Total portfolio theta."""
        return sum(p.theta for p in self.positions)

    def weighted_duration(self) -> float:
        """Market-value weighted duration."""
        total_mv = 0
        duration_mv = 0
        for p in self.positions:
            if p.duration is not None:
                total_mv += p.market_value
                duration_mv += p.duration * p.market_value
        return duration_mv / total_mv if total_mv != 0 else 0

    def concentration_by_position(self) -> Dict[str, float]:
        """Position concentrations."""
        return self.weights()

    def concentration_by_asset_class(self) -> Dict[AssetClassRisk, float]:
        """Asset class concentrations."""
        total = self.total_market_value()
        if total == 0:
            return {}
        result = {}
        for ac, positions in self.by_asset_class().items():
            result[ac] = sum(p.market_value for p in positions) / total
        return result

    def herfindahl_index(self) -> float:
        """
        Herfindahl-Hirschman Index for concentration.

        HHI = Σ wᵢ²

        Range: 1/N (fully diversified) to 1 (single position)
        """
        weights = list(self.weights().values())
        return sum(w**2 for w in weights)

    def effective_number_of_positions(self) -> float:
        """
        Effective number of positions (1 / HHI).

        Higher is more diversified.
        """
        hhi = self.herfindahl_index()
        return 1 / hhi if hhi > 0 else 0


class PortfolioAnalytics:
    """
    Portfolio Analytics Engine.

    Calculates comprehensive risk metrics for a portfolio.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        returns_history: Optional[Dict[str, list[float]]] = None,
        correlation_matrix: Optional[list[list[float]]] = None,
    ):
        """
        Initialize analytics.

        Args:
            portfolio: Portfolio to analyze
            returns_history: Historical returns by instrument
            correlation_matrix: Correlation matrix between instruments
        """
        self.portfolio = portfolio
        self.returns_history = returns_history or {}
        self.correlation_matrix = correlation_matrix

    @staticmethod
    def _std(data: list[float], ddof: int = 1) -> float:
        """Calculate standard deviation with degrees of freedom adjustment."""
        n = len(data)
        if n <= ddof:
            return 0.0
        mean = sum(data) / n
        variance = sum((x - mean) ** 2 for x in data) / (n - ddof)
        return math.sqrt(variance)

    @staticmethod
    def _mean(data: list[float]) -> float:
        """Calculate mean."""
        if not data:
            return 0.0
        return sum(data) / len(data)

    @staticmethod
    def _mat_vec_mult(matrix: list[list[float]], vec: list[float]) -> list[float]:
        """Matrix-vector multiplication."""
        return [sum(row[j] * vec[j] for j in range(len(vec))) for row in matrix]

    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        """Dot product of two vectors."""
        return sum(ai * bi for ai, bi in zip(a, b))

    def portfolio_var(
        self,
        confidence_level: float = 0.99,
        holding_period: int = 1,
        method: str = "parametric",
    ) -> VaRResult:
        """
        Calculate portfolio VaR.

        Args:
            confidence_level: VaR confidence level
            holding_period: Holding period in days
            method: "historical", "parametric", or "monte_carlo"

        Returns:
            VaRResult
        """
        portfolio_value = self.portfolio.total_market_value()

        if method == "historical" and self.returns_history:
            # Calculate portfolio returns
            weights = self.portfolio.weights()
            portfolio_returns: Optional[list[float]] = None

            for inst_id, w in weights.items():
                if inst_id in self.returns_history:
                    inst_returns = [r * w for r in self.returns_history[inst_id]]
                    if portfolio_returns is None:
                        portfolio_returns = inst_returns
                    else:
                        portfolio_returns = [a + b for a, b in zip(portfolio_returns, inst_returns)]

            if portfolio_returns is not None:
                var_calc = HistoricalVaR(
                    returns=portfolio_returns,
                    portfolio_value=portfolio_value,
                    confidence_level=confidence_level,
                    holding_period=holding_period,
                )
                return var_calc.calculate()

        # Parametric fallback
        # Estimate portfolio volatility
        positions = self.portfolio.positions
        n = len(positions)

        if n == 0:
            return VaRResult(
                var=0, expected_shortfall=0,
                confidence_level=confidence_level,
                holding_period=holding_period,
                method=method,
            )

        # Estimate volatilities from returns if available
        volatilities: list[float] = []
        for p in positions:
            if p.instrument_id in self.returns_history:
                vol = self._std(self.returns_history[p.instrument_id], ddof=1) * math.sqrt(252)
            else:
                vol = 0.2  # Default assumption
            volatilities.append(vol)

        # Use correlation matrix or assume correlation
        if self.correlation_matrix is not None and len(self.correlation_matrix) == n:
            corr = self.correlation_matrix
        else:
            # Average correlation assumption: 0.7 off-diagonal, 1.0 diagonal
            corr = [[0.7 if i != j else 1.0 for j in range(n)] for i in range(n)]

        # Calculate portfolio variance
        weights_arr = [p.market_value / portfolio_value for p in positions]

        # Build covariance matrix: cov[i][j] = vol[i] * vol[j] * corr[i][j]
        cov = [[volatilities[i] * volatilities[j] * corr[i][j] for j in range(n)] for i in range(n)]

        # Portfolio variance = w' * Cov * w
        cov_w = self._mat_vec_mult(cov, weights_arr)
        port_variance = self._dot(weights_arr, cov_w)
        portfolio_vol = math.sqrt(port_variance)

        var_calc = ParametricVaR(
            portfolio_value=portfolio_value,
            volatility=portfolio_vol / math.sqrt(252),  # Daily vol
            confidence_level=confidence_level,
            holding_period=holding_period,
        )

        return var_calc.calculate()

    def risk_contribution(
        self,
        var_result: Optional[VaRResult] = None,
    ) -> Dict[str, float]:
        """
        Calculate risk contribution by position.

        Uses component VaR approach.

        Args:
            var_result: Pre-calculated VaR (will calculate if None)

        Returns:
            Dict of {instrument_id: risk_contribution}
        """
        if var_result is None:
            var_result = self.portfolio_var()

        positions = self.portfolio.positions
        n = len(positions)
        if n == 0:
            return {}

        portfolio_value = self.portfolio.total_market_value()
        weights = [p.market_value / portfolio_value for p in positions]

        # Estimate volatilities
        volatilities: list[float] = []
        for p in positions:
            if p.instrument_id in self.returns_history:
                vol = self._std(self.returns_history[p.instrument_id], ddof=1)
            else:
                vol = 0.01
            volatilities.append(vol)

        # Correlation
        if self.correlation_matrix is not None and len(self.correlation_matrix) == n:
            corr = self.correlation_matrix
        else:
            # Average correlation assumption: 0.7 off-diagonal, 1.0 diagonal
            corr = [[0.7 if i != j else 1.0 for j in range(n)] for i in range(n)]

        # Build covariance matrix
        cov = [[volatilities[i] * volatilities[j] * corr[i][j] for j in range(n)] for i in range(n)]

        # Calculate portfolio volatility
        cov_w = self._mat_vec_mult(cov, weights)
        port_var = self._dot(weights, cov_w)
        port_vol = math.sqrt(port_var)

        # Marginal contribution: (Cov @ w) / port_vol
        if port_vol > 0:
            marginal = [cov_w[i] / port_vol for i in range(n)]
        else:
            marginal = [0.0] * n

        # Component contribution (weight * marginal)
        component = [weights[i] * marginal[i] for i in range(n)]

        # Scale to VaR
        if port_vol > 0:
            scale = var_result.var / (port_vol * portfolio_value)
        else:
            scale = 0

        result = {}
        for i, p in enumerate(positions):
            result[p.instrument_id] = component[i] * portfolio_value * scale

        return result

    def diversification_ratio(self) -> float:
        """
        Calculate diversification ratio.

        DR = Σ wᵢσᵢ / σₚ

        DR > 1 indicates diversification benefit.
        """
        positions = self.portfolio.positions
        n = len(positions)
        if n == 0:
            return 1.0

        portfolio_value = self.portfolio.total_market_value()
        weights = [p.market_value / portfolio_value for p in positions]

        volatilities: list[float] = []
        for p in positions:
            if p.instrument_id in self.returns_history:
                vol = self._std(self.returns_history[p.instrument_id], ddof=1)
            else:
                vol = 0.01
            volatilities.append(vol)

        # Sum of weighted volatilities
        weighted_vol_sum = sum(abs(weights[i]) * volatilities[i] for i in range(n))

        # Portfolio volatility
        if self.correlation_matrix is not None and len(self.correlation_matrix) == n:
            corr = self.correlation_matrix
        else:
            # Average correlation assumption: 0.7 off-diagonal, 1.0 diagonal
            corr = [[0.7 if i != j else 1.0 for j in range(n)] for i in range(n)]

        # Build covariance matrix
        cov = [[volatilities[i] * volatilities[j] * corr[i][j] for j in range(n)] for i in range(n)]

        # Portfolio volatility = sqrt(w' * Cov * w)
        cov_w = self._mat_vec_mult(cov, weights)
        port_var = self._dot(weights, cov_w)
        port_vol = math.sqrt(port_var)

        if port_vol > 0:
            return weighted_vol_sum / port_vol
        return 1.0

    def risk_by_asset_class(
        self,
        var_result: Optional[VaRResult] = None,
    ) -> Dict[AssetClassRisk, float]:
        """
        Calculate risk by asset class.

        Args:
            var_result: Pre-calculated VaR

        Returns:
            Dict of {AssetClassRisk: risk_contribution}
        """
        contributions = self.risk_contribution(var_result)

        result: Dict[AssetClassRisk, float] = {}
        for p in self.portfolio.positions:
            ac = p.asset_class
            if ac not in result:
                result[ac] = 0
            result[ac] += contributions.get(p.instrument_id, 0)

        return result

    def analyze(
        self,
        confidence_level: float = 0.99,
    ) -> PortfolioRiskResult:
        """
        Full portfolio risk analysis.

        Returns:
            PortfolioRiskResult with all metrics
        """
        var_result = self.portfolio_var(confidence_level=confidence_level)
        risk_contrib = self.risk_contribution(var_result)
        risk_by_ac = self.risk_by_asset_class(var_result)
        div_ratio = self.diversification_ratio()

        concentration = {
            "hhi": self.portfolio.herfindahl_index(),
            "effective_n": self.portfolio.effective_number_of_positions(),
            "top_5": sum(sorted(
                [abs(w) for w in self.portfolio.weights().values()],
                reverse=True
            )[:5]),
        }

        return PortfolioRiskResult(
            total_market_value=self.portfolio.total_market_value(),
            total_var=var_result.var,
            total_expected_shortfall=var_result.expected_shortfall,
            total_delta=self.portfolio.total_delta(),
            total_gamma=self.portfolio.total_gamma(),
            total_vega=self.portfolio.total_vega(),
            total_duration=self.portfolio.weighted_duration(),
            diversification_ratio=div_ratio,
            concentration_metrics=concentration,
            risk_by_asset_class=risk_by_ac,
            risk_attribution=risk_contrib,
        )


def optimal_hedge_ratio(
    portfolio_delta: float,
    hedge_delta: float,
    portfolio_gamma: float = 0,
    hedge_gamma: float = 0,
    target_delta: float = 0,
    target_gamma: float = 0,
) -> Tuple[float, float]:
    """
    Calculate optimal hedge ratios for delta and gamma.

    Args:
        portfolio_delta: Current portfolio delta
        hedge_delta: Hedge instrument delta
        portfolio_gamma: Current portfolio gamma
        hedge_gamma: Hedge instrument gamma
        target_delta: Target delta (0 for neutral)
        target_gamma: Target gamma (0 for neutral)

    Returns:
        (hedge_units_for_delta, hedge_units_for_gamma)
    """
    # Delta hedge
    if abs(hedge_delta) > 1e-10:
        delta_hedge = (target_delta - portfolio_delta) / hedge_delta
    else:
        delta_hedge = 0

    # Gamma hedge (if provided)
    if abs(hedge_gamma) > 1e-10:
        gamma_hedge = (target_gamma - portfolio_gamma) / hedge_gamma
    else:
        gamma_hedge = 0

    return (delta_hedge, gamma_hedge)


def _std(data: list[float], ddof: int = 1) -> float:
    """Calculate standard deviation with degrees of freedom adjustment."""
    n = len(data)
    if n <= ddof:
        return 0.0
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / (n - ddof)
    return math.sqrt(variance)


def _mean(data: list[float]) -> float:
    """Calculate mean."""
    if not data:
        return 0.0
    return sum(data) / len(data)


def tracking_error(
    portfolio_returns: list[float],
    benchmark_returns: list[float],
    annualize: bool = True,
) -> float:
    """
    Calculate tracking error vs benchmark.

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns
        annualize: Annualize the result

    Returns:
        Tracking error
    """
    active_returns = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
    te = _std(active_returns, ddof=1)

    if annualize:
        te *= math.sqrt(252)

    return te


def information_ratio(
    portfolio_returns: list[float],
    benchmark_returns: list[float],
) -> float:
    """
    Calculate Information Ratio.

    IR = Mean(Active Return) / Tracking Error

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Information ratio
    """
    active_returns = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
    te = tracking_error(portfolio_returns, benchmark_returns, annualize=False)

    if te == 0:
        return 0

    return _mean(active_returns) * math.sqrt(252) / te


def sharpe_ratio(
    returns: list[float],
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate Sharpe Ratio.

    SR = (Mean Return - Rf) / StdDev

    Args:
        returns: Portfolio returns
        risk_free_rate: Risk-free rate (annual)

    Returns:
        Sharpe ratio (annualized)
    """
    daily_rf = risk_free_rate / 252
    excess_returns = [r - daily_rf for r in returns]
    vol = _std(returns, ddof=1)

    if vol == 0:
        return 0

    return _mean(excess_returns) * math.sqrt(252) / vol
