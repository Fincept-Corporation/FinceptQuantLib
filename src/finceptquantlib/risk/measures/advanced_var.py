"""
Advanced VaR Analytics

Enhanced VaR methods:
- Incremental VaR (marginal contribution)
- Component VaR by risk factor
- VaR decomposition and attribution
- Liquidity-adjusted VaR
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
import math


@dataclass
class IncrementalVaRResult:
    """
    Result of incremental VaR calculation.

    Attributes:
        position_name: Position identifier
        standalone_var: VaR of position in isolation
        incremental_var: Change in portfolio VaR from adding position
        marginal_var: Derivative of portfolio VaR w.r.t. position size
        contribution_pct: % contribution to total VaR
    """
    position_name: str
    standalone_var: float
    incremental_var: float
    marginal_var: float
    contribution_pct: float


@dataclass
class ComponentVaRResult:
    """
    Component VaR decomposition by risk factor.

    Attributes:
        risk_factor: Risk factor name (equity, rates, credit, FX, etc.)
        component_var: VaR contribution from this factor
        standalone_var: VaR if only this factor varied
        diversification_benefit: Benefit from correlation with other factors
    """
    risk_factor: str
    component_var: float
    standalone_var: float
    diversification_benefit: float


class IncrementalVaRCalculator:
    """
    Calculate Incremental and Marginal VaR.

    Incremental VaR = VaR(portfolio + position) - VaR(portfolio)
    Marginal VaR = ∂VaR/∂w_i (sensitivity to position weight)

    Examples:
        >>> calculator = IncrementalVaRCalculator(  # doctest: +SKIP
        ...     portfolio_returns=portfolio_returns_matrix,
        ...     confidence_level=0.99
        ... )
        >>> incremental = calculator.calculate_incremental_var(  # doctest: +SKIP
        ...     new_position_returns=new_asset_returns
        ... )
        >>> print(f"Incremental VaR: ${incremental.incremental_var:,.2f}")  # doctest: +SKIP
    """

    def __init__(
        self,
        portfolio_returns: List[float],
        confidence_level: float = 0.99,
        portfolio_value: float = 1_000_000.0
    ) -> None:
        """
        Initialize incremental VaR calculator.

        Args:
            portfolio_returns: Historical portfolio returns
            confidence_level: VaR confidence level
            portfolio_value: Portfolio value
        """
        self.portfolio_returns = portfolio_returns
        self.confidence_level = confidence_level
        self.portfolio_value = portfolio_value

        # Calculate baseline portfolio VaR
        self.portfolio_var = self._calculate_var(portfolio_returns)

    def calculate_incremental_var(
        self,
        new_position_returns: List[float],
        position_weight: float = 0.1,
        position_name: str = "New Position"
    ) -> IncrementalVaRResult:
        """
        Calculate incremental VaR for adding a new position.

        Args:
            new_position_returns: Returns of new position
            position_weight: Weight of new position in combined portfolio
            position_name: Position identifier

        Returns:
            IncrementalVaRResult
        """
        if len(new_position_returns) != len(self.portfolio_returns):
            raise ValueError("Return histories must have same length")

        # Combined portfolio returns
        combined_returns = [
            (1 - position_weight) * pr + position_weight * nr
            for pr, nr in zip(self.portfolio_returns, new_position_returns)
        ]

        # Combined VaR
        combined_var = self._calculate_var(combined_returns)

        # Incremental VaR
        incremental_var = combined_var - self.portfolio_var

        # Standalone VaR of new position
        standalone_var = self._calculate_var(new_position_returns)

        # Marginal VaR (using finite difference)
        epsilon = 0.001
        combined_plus = [
            (1 - position_weight - epsilon) * pr + (position_weight + epsilon) * nr
            for pr, nr in zip(self.portfolio_returns, new_position_returns)
        ]
        var_plus = self._calculate_var(combined_plus)
        marginal_var = (var_plus - combined_var) / epsilon

        # Contribution percentage
        contribution_pct = (incremental_var / combined_var * 100) if combined_var > 0 else 0

        return IncrementalVaRResult(
            position_name=position_name,
            standalone_var=standalone_var * self.portfolio_value * position_weight,
            incremental_var=incremental_var * self.portfolio_value,
            marginal_var=marginal_var * self.portfolio_value,
            contribution_pct=contribution_pct
        )

    def _calculate_var(self, returns: List[float]) -> float:
        """Calculate VaR from return series."""
        if not returns:
            return 0.0

        sorted_returns = sorted(returns)
        index = int(len(sorted_returns) * (1 - self.confidence_level))
        var_return = -sorted_returns[index]  # Negative of percentile

        return var_return


class ComponentVaRCalculator:
    """
    Decompose portfolio VaR by risk factors.

    Uses Euler allocation principle: VaR_i = (∂VaR/∂w_i) * w_i

    Examples:
        >>> calculator = ComponentVaRCalculator(  # doctest: +SKIP
        ...     returns_by_factor={
        ...         "Equity": equity_returns,
        ...         "Rates": rates_returns,
        ...         "Credit": credit_returns
        ...     },
        ...     weights={"Equity": 0.5, "Rates": 0.3, "Credit": 0.2}
        ... )
        >>> components = calculator.decompose_var()  # doctest: +SKIP
        >>> for factor, result in components.items():  # doctest: +SKIP
        ...     print(f"{factor}: ${result.component_var:,.2f}")
    """

    def __init__(
        self,
        returns_by_factor: Dict[str, List[float]],
        weights: Dict[str, float],
        confidence_level: float = 0.99,
        portfolio_value: float = 1_000_000.0
    ) -> None:
        """
        Initialize component VaR calculator.

        Args:
            returns_by_factor: Returns for each risk factor
            weights: Portfolio weights by risk factor
            confidence_level: VaR confidence
            portfolio_value: Portfolio value
        """
        self.returns_by_factor = returns_by_factor
        self.weights = weights
        self.confidence_level = confidence_level
        self.portfolio_value = portfolio_value

        # Validate
        if abs(sum(weights.values()) - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")

    def decompose_var(self) -> Dict[str, ComponentVaRResult]:
        """
        Decompose VaR into components by risk factor.

        Returns:
            Dict of ComponentVaRResult by factor name
        """
        # Calculate portfolio returns
        n_periods = len(next(iter(self.returns_by_factor.values())))
        portfolio_returns = [0.0] * n_periods

        for factor, returns in self.returns_by_factor.items():
            weight = self.weights.get(factor, 0.0)
            for i in range(n_periods):
                portfolio_returns[i] += weight * returns[i]

        # Component VaR for each factor using Euler allocation
        components = {}

        for factor in self.returns_by_factor.keys():
            # Marginal VaR
            marginal_var = self._marginal_var(factor, portfolio_returns)

            # Component VaR = Marginal VaR * Weight
            weight = self.weights[factor]
            component_var = marginal_var * weight * self.portfolio_value

            # Standalone VaR (100% in this factor)
            standalone_var = self._calculate_var(self.returns_by_factor[factor]) * self.portfolio_value

            # Diversification benefit
            diversification = standalone_var - component_var

            components[factor] = ComponentVaRResult(
                risk_factor=factor,
                component_var=component_var,
                standalone_var=standalone_var,
                diversification_benefit=diversification
            )

        return components

    def _marginal_var(self, factor: str, portfolio_returns: List[float]) -> float:
        """Calculate marginal VaR for a factor using finite difference."""
        epsilon = 0.01
        factor_returns = self.returns_by_factor[factor]

        # Increase factor weight slightly
        perturbed_returns = portfolio_returns[:]
        for i in range(len(perturbed_returns)):
            perturbed_returns[i] += epsilon * factor_returns[i]

        var_plus = self._calculate_var(perturbed_returns)
        var_base = self._calculate_var(portfolio_returns)

        marginal = (var_plus - var_base) / epsilon
        return marginal

    def _calculate_var(self, returns: List[float]) -> float:
        """Calculate VaR."""
        sorted_returns = sorted(returns)
        index = int(len(sorted_returns) * (1 - self.confidence_level))
        return -sorted_returns[index]


@dataclass
class LiquidityAdjustment:
    """
    Liquidity adjustment parameters.

    Attributes:
        bid_ask_spread: Bid-ask spread (%)
        market_depth: Market depth (units tradeable at mid)
        liquidation_horizon: Days to liquidate position
        daily_volume: Average daily trading volume
    """
    bid_ask_spread: float
    market_depth: float
    liquidation_horizon: int = 10
    daily_volume: float = 0.0

    def liquidity_cost(self, position_size: float) -> float:
        """
        Estimate cost of liquidating position.

        Args:
            position_size: Position size to liquidate

        Returns:
            Liquidity cost as % of position value
        """
        # Cost from bid-ask spread
        spread_cost = self.bid_ask_spread / 2

        # Market impact cost (square-root model)
        if self.daily_volume > 0:
            participation_rate = position_size / (self.daily_volume * self.liquidation_horizon)
            impact_cost = 0.01 * math.sqrt(participation_rate)  # Simplified
        else:
            impact_cost = 0.0

        return spread_cost + impact_cost


class LiquidityAdjustedVaR:
    """
    Liquidity-Adjusted VaR.

    Incorporates cost of unwinding positions under stress.

    LVaR = VaR + Liquidity Cost

    Examples:
        >>> lvar = LiquidityAdjustedVaR(  # doctest: +SKIP
        ...     returns=returns,
        ...     position_values=[1_000_000, 500_000, 300_000],
        ...     liquidity_params=[
        ...         LiquidityAdjustment(0.001, 10_000_000, 1, 5_000_000),  # Liquid
        ...         LiquidityAdjustment(0.005, 1_000_000, 10, 100_000),   # Less liquid
        ...         LiquidityAdjustment(0.02, 100_000, 30, 10_000)        # Illiquid
        ...     ]
        ... )
        >>> result = lvar.calculate(confidence=0.99)  # doctest: +SKIP
    """

    def __init__(
        self,
        returns: List[List[float]],  # Returns for each position
        position_values: List[float],
        liquidity_params: List[LiquidityAdjustment],
        confidence_level: float = 0.99
    ) -> None:
        """
        Initialize liquidity-adjusted VaR.

        Args:
            returns: Returns matrix (positions x time)
            position_values: Current position values
            liquidity_params: Liquidity parameters per position
            confidence_level: VaR confidence
        """
        self.returns = returns
        self.position_values = position_values
        self.liquidity_params = liquidity_params
        self.confidence_level = confidence_level

    def calculate(self) -> Dict[str, float]:
        """
        Calculate liquidity-adjusted VaR.

        Returns:
            Dict with standard_var, liquidity_cost, total_lvar
        """
        # Standard VaR
        n_periods = len(self.returns[0]) if self.returns else 0
        portfolio_returns = [0.0] * n_periods

        total_value = sum(self.position_values)
        weights = [v / total_value for v in self.position_values] if total_value > 0 else []

        for i, position_returns in enumerate(self.returns):
            weight = weights[i] if i < len(weights) else 0
            for t in range(n_periods):
                portfolio_returns[t] += weight * position_returns[t]

        # VaR
        sorted_returns = sorted(portfolio_returns)
        var_index = int(len(sorted_returns) * (1 - self.confidence_level))
        var_return = -sorted_returns[var_index]
        standard_var = var_return * total_value

        # Liquidity cost
        total_liquidity_cost = 0.0
        for i, (value, liq_params) in enumerate(zip(self.position_values, self.liquidity_params)):
            cost_pct = liq_params.liquidity_cost(value)
            total_liquidity_cost += value * cost_pct

        # Total LVaR
        total_lvar = standard_var + total_liquidity_cost

        return {
            "standard_var": standard_var,
            "liquidity_cost": total_liquidity_cost,
            "liquidity_adjusted_var": total_lvar,
            "liquidity_adjustment_pct": (total_liquidity_cost / standard_var * 100) if standard_var > 0 else 0
        }


def incremental_var(
    portfolio_returns: List[float],
    new_position_returns: List[float],
    position_weight: float,
    confidence_level: float = 0.99,
    portfolio_value: float = 1_000_000.0
) -> float:
    """
    Calculate incremental VaR for adding a position.

    Args:
        portfolio_returns: Existing portfolio returns
        new_position_returns: New position returns
        position_weight: Weight of new position
        confidence_level: VaR confidence
        portfolio_value: Portfolio value

    Returns:
        Incremental VaR amount
    """
    calculator = IncrementalVaRCalculator(portfolio_returns, confidence_level, portfolio_value)
    result = calculator.calculate_incremental_var(new_position_returns, position_weight)
    return result.incremental_var


def component_var(
    returns_by_factor: Dict[str, List[float]],
    weights: Dict[str, float],
    confidence_level: float = 0.99,
    portfolio_value: float = 1_000_000.0
) -> Dict[str, float]:
    """
    Calculate component VaR by risk factor.

    Args:
        returns_by_factor: Returns for each factor
        weights: Portfolio weights by factor
        confidence_level: VaR confidence
        portfolio_value: Portfolio value

    Returns:
        Dict of component VaR by factor
    """
    calculator = ComponentVaRCalculator(returns_by_factor, weights, confidence_level, portfolio_value)
    results = calculator.decompose_var()
    return {factor: result.component_var for factor, result in results.items()}


def marginal_var(
    portfolio_returns: List[float],
    position_returns: List[float],
    current_weight: float,
    confidence_level: float = 0.99
) -> float:
    """
    Calculate marginal VaR (sensitivity of VaR to position weight).

    dVaR/dw = Marginal VaR

    Args:
        portfolio_returns: Portfolio returns
        position_returns: Position returns
        current_weight: Current position weight
        confidence_level: VaR confidence

    Returns:
        Marginal VaR (per unit weight change)
    """
    epsilon = 0.001

    # VaR at current weight
    current_portfolio = [
        (1 - current_weight) * pr + current_weight * pos_r
        for pr, pos_r in zip(portfolio_returns, position_returns)
    ]
    var_current = -sorted(current_portfolio)[int(len(current_portfolio) * (1 - confidence_level))]

    # VaR at slightly higher weight
    perturbed_portfolio = [
        (1 - current_weight - epsilon) * pr + (current_weight + epsilon) * pos_r
        for pr, pos_r in zip(portfolio_returns, position_returns)
    ]
    var_perturbed = -sorted(perturbed_portfolio)[int(len(perturbed_portfolio) * (1 - confidence_level))]

    # Marginal VaR
    return (var_perturbed - var_current) / epsilon
