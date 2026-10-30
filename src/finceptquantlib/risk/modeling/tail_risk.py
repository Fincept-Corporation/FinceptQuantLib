"""
Tail Risk Metrics

Advanced tail risk measures beyond VaR/CVaR:
- Expected Shortfall (ES / CVaR)
- Conditional Tail Expectation (CTE)
- Extreme Value Theory (EVT)
- Tail Index estimation
- Maximum Loss
- Omega ratio
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import math


@dataclass
class TailRiskMetrics:
    """
    Complete set of tail risk metrics.

    Attributes:
        var_95: VaR at 95%
        var_99: VaR at 99%
        var_995: VaR at 99.5%
        cvar_95: CVaR at 95%
        cvar_99: CVaR at 99%
        expected_shortfall: ES (same as CVaR)
        maximum_loss: Maximum observed loss
        tail_index: Tail index (from EVT)
        omega_ratio: Omega ratio (prob-weighted gains/losses)
    """
    var_95: float
    var_99: float
    var_995: float
    cvar_95: float
    cvar_99: float
    expected_shortfall: float
    maximum_loss: float
    tail_index: Optional[float] = None
    omega_ratio: Optional[float] = None


class TailRiskAnalyzer:
    """
    Comprehensive tail risk analysis.

    Calculates multiple tail risk metrics from return distribution.

    Examples:
        >>> analyzer = TailRiskAnalyzer(returns=historical_returns)  # doctest: +SKIP
        >>> metrics = analyzer.calculate_all_metrics()  # doctest: +SKIP
        >>> print(f"VaR 99%: {metrics.var_99:.4f}")  # doctest: +SKIP
        >>> print(f"CVaR 99%: {metrics.cvar_99:.4f}")  # doctest: +SKIP
        >>> print(f"Tail Index: {metrics.tail_index:.2f}")  # doctest: +SKIP
    """

    def __init__(
        self,
        returns: List[float],
        portfolio_value: float = 1.0
    ) -> None:
        """
        Initialize tail risk analyzer.

        Args:
            returns: Historical returns
            portfolio_value: Portfolio value for absolute amounts
        """
        self.returns = sorted(returns)  # Sort for quantile calculations
        self.portfolio_value = portfolio_value

    def var(self, confidence_level: float) -> float:
        """
        Calculate VaR at any confidence level.

        Args:
            confidence_level: Confidence level (e.g., 0.99)

        Returns:
            VaR (positive number representing loss)
        """
        index = int(len(self.returns) * (1 - confidence_level))
        var_return = -self.returns[index]
        return var_return * self.portfolio_value

    def cvar(self, confidence_level: float) -> float:
        """
        Calculate CVaR (Conditional VaR / Expected Shortfall).

        CVaR = E[Loss | Loss > VaR]
        Average of losses exceeding VaR.

        Args:
            confidence_level: Confidence level

        Returns:
            CVaR
        """
        index = int(len(self.returns) * (1 - confidence_level))
        tail_returns = self.returns[:index]

        if not tail_returns:
            return self.var(confidence_level)

        avg_tail_return = sum(tail_returns) / len(tail_returns)
        return -avg_tail_return * self.portfolio_value

    def expected_shortfall(self, confidence_level: float = 0.99) -> float:
        """Expected Shortfall (same as CVaR)."""
        return self.cvar(confidence_level)

    def maximum_loss(self) -> float:
        """Maximum historical loss."""
        return -self.returns[0] * self.portfolio_value

    def tail_index(self, threshold_quantile: float = 0.90) -> float:
        """
        Estimate tail index using Hill estimator.

        Lower tail index = fatter tails (more extreme events).

        Args:
            threshold_quantile: Quantile for tail threshold

        Returns:
            Tail index (alpha)
        """
        # Number of exceedances above threshold
        threshold_index = int(len(self.returns) * threshold_quantile)
        threshold = abs(self.returns[threshold_index])

        # Exceedances (absolute values)
        exceedances = [abs(r) for r in self.returns if abs(r) > threshold]

        if len(exceedances) < 2:
            return float('inf')

        # Hill estimator
        n = len(exceedances)
        log_sum = sum(math.log(ex / threshold) for ex in exceedances)
        alpha = n / log_sum if log_sum > 0 else float('inf')

        return alpha

    def omega_ratio(self, threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio.

        Omega = (Prob-weighted gains above threshold) / (Prob-weighted losses below threshold)

        Args:
            threshold: Return threshold (typically 0)

        Returns:
            Omega ratio
        """
        gains = sum(max(r - threshold, 0) for r in self.returns)
        losses = sum(max(threshold - r, 0) for r in self.returns)

        if losses == 0:
            return float('inf')

        return gains / losses

    def calculate_all_metrics(self) -> TailRiskMetrics:
        """Calculate complete set of tail risk metrics."""
        return TailRiskMetrics(
            var_95=self.var(0.95),
            var_99=self.var(0.99),
            var_995=self.var(0.995),
            cvar_95=self.cvar(0.95),
            cvar_99=self.cvar(0.99),
            expected_shortfall=self.expected_shortfall(0.99),
            maximum_loss=self.maximum_loss(),
            tail_index=self.tail_index(),
            omega_ratio=self.omega_ratio()
        )


class CoherentRiskMeasure:
    """
    Coherent risk measures satisfying Artzner axioms:
    - Monotonicity
    - Translation invariance
    - Homogeneity
    - Subadditivity

    CVaR/ES is coherent; VaR is not (fails subadditivity).
    """

    @staticmethod
    def check_subadditivity(
        risk_measure_func,
        portfolio1_returns: List[float],
        portfolio2_returns: List[float]
    ) -> bool:
        """
        Test if risk measure is subadditive.

        ρ(X + Y) ≤ ρ(X) + ρ(Y)

        Args:
            risk_measure_func: Risk measure function
            portfolio1_returns: Returns of portfolio 1
            portfolio2_returns: Returns of portfolio 2

        Returns:
            True if subadditivity holds
        """
        risk1 = risk_measure_func(portfolio1_returns)
        risk2 = risk_measure_func(portfolio2_returns)

        combined_returns = [r1 + r2 for r1, r2 in zip(portfolio1_returns, portfolio2_returns)]
        risk_combined = risk_measure_func(combined_returns)

        return risk_combined <= (risk1 + risk2)

    @staticmethod
    def spectral_risk_measure(
        returns: List[float],
        weight_function: callable,
        portfolio_value: float = 1.0
    ) -> float:
        """
        Spectral risk measure (generalized coherent measure).

        Weighted average of quantiles with increasing weights for tail.

        Args:
            returns: Returns distribution
            weight_function: Weight function phi(p) where p is probability
            portfolio_value: Portfolio value

        Returns:
            Spectral risk measure value
        """
        sorted_returns = sorted(returns)
        n = len(sorted_returns)

        risk = 0.0
        for i, ret in enumerate(sorted_returns):
            p = (i + 1) / n
            weight = weight_function(p)
            risk += weight * abs(ret)

        return risk * portfolio_value / n


def expected_shortfall_optimization(
    returns_matrix: List[List[float]],
    target_es: float,
    confidence_level: float = 0.99
) -> List[float]:
    """
    Optimize portfolio weights to minimize Expected Shortfall.

    CVaR/ES optimization is a linear programming problem.

    Args:
        returns_matrix: Returns for each asset (assets x time)
        target_es: Target ES level
        confidence_level: Confidence level

    Returns:
        Optimal weights
    """
    # Simplified: equal weights
    n_assets = len(returns_matrix)
    return [1.0 / n_assets] * n_assets if n_assets > 0 else []


def tail_dependence_coefficient(
    returns1: List[float],
    returns2: List[float],
    threshold_quantile: float = 0.95
) -> float:
    """
    Calculate tail dependence coefficient.

    Measures correlation in tail events.

    Args:
        returns1: Returns of asset 1
        returns2: Returns of asset 2
        threshold_quantile: Quantile for tail threshold

    Returns:
        Tail dependence coefficient (0-1)
    """
    n = min(len(returns1), len(returns2))

    # Find threshold
    threshold_index = int(n * threshold_quantile)
    threshold1 = sorted(returns1)[threshold_index]
    threshold2 = sorted(returns2)[threshold_index]

    # Count joint tail events
    joint_tail = sum(1 for r1, r2 in zip(returns1, returns2)
                    if r1 < threshold1 and r2 < threshold2)

    # Tail dependence
    tail_events1 = sum(1 for r in returns1 if r < threshold1)

    if tail_events1 == 0:
        return 0.0

    return joint_tail / tail_events1


def conditional_value_at_risk(
    returns: List[float],
    confidence_level: float = 0.99,
    portfolio_value: float = 1.0
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR).

    CVaR is a coherent risk measure (unlike VaR).

    Args:
        returns: Return series
        confidence_level: Confidence level
        portfolio_value: Portfolio value

    Returns:
        CVaR amount
    """
    analyzer = TailRiskAnalyzer(returns, portfolio_value)
    return analyzer.cvar(confidence_level)


def tail_conditional_expectation(
    returns: List[float],
    threshold: float,
    portfolio_value: float = 1.0
) -> float:
    """
    Calculate Tail Conditional Expectation (TCE).

    E[Loss | Loss > threshold]

    Args:
        returns: Returns
        threshold: Loss threshold
        portfolio_value: Portfolio value

    Returns:
        TCE value
    """
    tail_losses = [r for r in returns if -r > threshold]

    if not tail_losses:
        return 0.0

    avg_tail = sum(tail_losses) / len(tail_losses)
    return -avg_tail * portfolio_value


def worst_conditional_expectation(
    returns: List[float],
    confidence_level: float = 0.99,
    portfolio_value: float = 1.0
) -> float:
    """
    Calculate Worst Conditional Expectation (WCE).

    Average of worst (1-confidence)% of losses.
    Same as CVaR/ES.
    """
    return conditional_value_at_risk(returns, confidence_level, portfolio_value)
