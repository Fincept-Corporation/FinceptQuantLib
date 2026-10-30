"""
Correlation Stress Testing

Stress scenarios for correlation structures:
- Correlation breakdown (goes to 1 in crisis)
- Decorrelation scenarios
- Regime shifts
- Tail correlation stress
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import math


@dataclass
class CorrelationScenario:
    """
    Correlation stress scenario.

    Attributes:
        name: Scenario name
        correlation_matrix: Stressed correlation matrix
        description: Scenario description
        probability: Scenario probability (optional)
    """
    name: str
    correlation_matrix: List[List[float]]
    description: str = ""
    probability: Optional[float] = None


class CorrelationStressTester:
    """
    Test portfolio under correlation stress scenarios.

    Key scenarios:
    - Crisis correlation (all correlations → 1)
    - Decorrelation (all correlations → 0)
    - Regime shift (correlations flip sign)
    - Tail correlation spike

    Examples:
        >>> stress = CorrelationStressTester(
        ...     base_correlations=[[1.0, 0.5], [0.5, 1.0]],
        ...     volatilities=[0.20, 0.15],
        ...     weights=[0.6, 0.4]
        ... )
        >>> crisis_var = stress.stress_crisis_correlation(confidence=0.99)
        >>> print(f"Crisis VaR: ${crisis_var:,.2f}")
        Crisis VaR: $409,331.47
    """

    def __init__(
        self,
        base_correlations: List[List[float]],
        volatilities: List[float],
        weights: List[float],
        portfolio_value: float = 1_000_000.0
    ) -> None:
        """
        Initialize correlation stress tester.

        Args:
            base_correlations: Base correlation matrix
            volatilities: Asset volatilities
            weights: Portfolio weights
            portfolio_value: Portfolio value
        """
        self.base_corr = base_correlations
        self.vols = volatilities
        self.weights = weights
        self.portfolio_value = portfolio_value

        # Validate
        n = len(volatilities)
        if len(base_correlations) != n or any(len(row) != n for row in base_correlations):
            raise ValueError("Correlation matrix dimensions must match number of assets")

    def stress_crisis_correlation(
        self,
        crisis_correlation: float = 0.90,
        confidence: float = 0.99
    ) -> float:
        """
        Stress test with all correlations increasing to crisis level.

        In market crises, correlations tend toward 1 (all assets move together).

        Args:
            crisis_correlation: Correlation level in crisis (typically 0.8-0.95)
            confidence: VaR confidence level

        Returns:
            VaR under crisis correlation scenario
        """
        n = len(self.vols)

        # Build stressed correlation matrix
        stressed_corr = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    stressed_corr[i][j] = 1.0
                else:
                    stressed_corr[i][j] = crisis_correlation

        # Calculate covariance under stress
        stressed_cov = self._correlation_to_covariance(stressed_corr, self.vols)

        # Portfolio variance
        portfolio_var = self._portfolio_variance(stressed_cov, self.weights)
        portfolio_std = math.sqrt(portfolio_var)

        # VaR (parametric)
        from scipy.stats import norm
        z_score = norm.ppf(confidence)
        var = z_score * portfolio_std * self.portfolio_value

        return var

    def stress_decorrelation(self, confidence: float = 0.99) -> float:
        """
        Stress with all correlations going to zero.

        Diversification benefits disappear.
        """
        n = len(self.vols)

        # Zero correlation matrix (identity)
        zero_corr = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        zero_cov = self._correlation_to_covariance(zero_corr, self.vols)

        portfolio_var = self._portfolio_variance(zero_cov, self.weights)
        portfolio_std = math.sqrt(portfolio_var)

        from scipy.stats import norm
        z_score = norm.ppf(confidence)
        return z_score * portfolio_std * self.portfolio_value

    def stress_correlation_sign_flip(self, confidence: float = 0.99) -> float:
        """
        Regime shift: correlations flip sign.

        Hedges become positively correlated with risk.
        """
        n = len(self.vols)

        # Flip signs of off-diagonal elements
        flipped_corr = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    flipped_corr[i][j] = 1.0
                else:
                    flipped_corr[i][j] = -self.base_corr[i][j]

        flipped_cov = self._correlation_to_covariance(flipped_corr, self.vols)
        portfolio_var = self._portfolio_variance(flipped_cov, self.weights)
        portfolio_std = math.sqrt(portfolio_var)

        from scipy.stats import norm
        z_score = norm.ppf(confidence)
        return z_score * portfolio_std * self.portfolio_value

    def generate_correlation_scenarios(self) -> List[CorrelationScenario]:
        """
        Generate standard correlation stress scenarios.

        Returns:
            List of CorrelationScenario objects
        """
        scenarios = []

        n = len(self.vols)

        # Scenario 1: Crisis (high correlation)
        crisis_corr = [[1.0 if i == j else 0.90 for j in range(n)] for i in range(n)]
        scenarios.append(CorrelationScenario(
            name="Crisis Correlation",
            correlation_matrix=crisis_corr,
            description="All correlations increase to 0.90",
            probability=0.05
        ))

        # Scenario 2: Decorrelation
        zero_corr = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        scenarios.append(CorrelationScenario(
            name="Decorrelation",
            correlation_matrix=zero_corr,
            description="All correlations drop to zero",
            probability=0.10
        ))

        # Scenario 3: Regime shift
        flipped = [[1.0 if i == j else -self.base_corr[i][j] for j in range(n)] for i in range(n)]
        scenarios.append(CorrelationScenario(
            name="Regime Shift",
            correlation_matrix=flipped,
            description="Correlations flip sign",
            probability=0.05
        ))

        return scenarios

    def _correlation_to_covariance(
        self,
        corr: List[List[float]],
        vols: List[float]
    ) -> List[List[float]]:
        """Convert correlation matrix to covariance matrix."""
        n = len(vols)
        cov = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                cov[i][j] = corr[i][j] * vols[i] * vols[j]

        return cov

    def _portfolio_variance(
        self,
        cov: List[List[float]],
        weights: List[float]
    ) -> float:
        """Calculate portfolio variance."""
        var = 0.0
        n = len(weights)

        for i in range(n):
            for j in range(n):
                var += weights[i] * weights[j] * cov[i][j]

        return var



# conditional_value_at_risk, tail_conditional_expectation, and
# worst_conditional_expectation are defined in
# finceptquantlib.risk.modeling.tail_risk (canonical location)
