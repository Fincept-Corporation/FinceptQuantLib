"""
VaR Model Backtesting

Framework for validating VaR models:
- Exception counting
- Coverage tests (Kupiec, Christoffersen)
- Magnitude tests
- Traffic light zones (Basel)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
import math


class TrafficLightZone(Enum):
    """Basel traffic light zones for VaR backtesting."""
    GREEN = "green"  # 0-4 exceptions
    YELLOW = "yellow"  # 5-9 exceptions
    RED = "red"  # 10+ exceptions


@dataclass
class BacktestResult:
    """
    VaR backtest result.

    Attributes:
        num_observations: Total observations
        num_exceptions: Number of VaR exceedances
        exception_rate: Actual exception rate
        expected_exceptions: Expected exceptions
        kupiec_statistic: Kupiec POF test statistic
        kupiec_pvalue: P-value for Kupiec test
        passes_kupiec: Whether model passes Kupiec test
        christoffersen_statistic: Independence test statistic
        christoffersen_pvalue: P-value
        traffic_light_zone: Basel traffic light zone
        model_adequate: Overall adequacy assessment
    """
    num_observations: int
    num_exceptions: int
    exception_rate: float
    expected_exceptions: float
    kupiec_statistic: float
    kupiec_pvalue: float
    passes_kupiec: bool
    christoffersen_statistic: float
    christoffersen_pvalue: float
    traffic_light_zone: TrafficLightZone
    model_adequate: bool


class VaRBacktester:
    """
    VaR Model Backtesting Framework.

    Validates VaR models using statistical tests:
    - Kupiec POF Test (unconditional coverage)
    - Christoffersen Test (conditional coverage/independence)
    - Basel Traffic Light Test

    Examples:
        >>> backtester = VaRBacktester(confidence_level=0.99)
        >>> backtester.add_observation(actual_return=-0.025, var_estimate=0.020)
        >>> backtester.add_observation(actual_return=-0.015, var_estimate=0.020)
        >>> # ... add more observations
        >>> result = backtester.run_tests()
        >>> print(f"Exception rate: {result.exception_rate:.2%}")
        Exception rate: 50.00%
        >>> print(f"Kupiec test: {'PASS' if result.passes_kupiec else 'FAIL'}")
        Kupiec test: FAIL
    """

    def __init__(self, confidence_level: float = 0.99) -> None:
        """
        Initialize backtester.

        Args:
            confidence_level: VaR confidence level (e.g., 0.99 for 99%)
        """
        self.confidence_level = confidence_level
        self.expected_exception_rate = 1 - confidence_level
        self.actual_returns: List[float] = []
        self.var_estimates: List[float] = []

    def add_observation(self, actual_return: float, var_estimate: float) -> None:
        """
        Add observation for backtesting.

        Args:
            actual_return: Actual portfolio return (negative for loss)
            var_estimate: VaR estimate (positive number)
        """
        self.actual_returns.append(actual_return)
        self.var_estimates.append(var_estimate)

    def run_tests(self, significance_level: float = 0.05) -> BacktestResult:
        """
        Run all backtesting statistical tests.

        Args:
            significance_level: Test significance level (typically 5%)

        Returns:
            BacktestResult with comprehensive statistics
        """
        n = len(self.actual_returns)
        if n == 0:
            raise ValueError("No observations added")

        # Count exceptions (losses exceeding VaR)
        exceptions = [1 if -self.actual_returns[i] > self.var_estimates[i] else 0
                     for i in range(n)]
        num_exceptions = sum(exceptions)

        # Exception rate
        exception_rate = num_exceptions / n
        expected_exceptions = n * self.expected_exception_rate

        # Kupiec POF Test (Proportion of Failures)
        kupiec_stat, kupiec_pval = self._kupiec_test(n, num_exceptions, self.expected_exception_rate)
        passes_kupiec = kupiec_pval > significance_level

        # Christoffersen Independence Test
        christ_stat, christ_pval = self._christoffersen_test(exceptions)

        # Basel Traffic Light
        traffic_light = self._traffic_light_zone(num_exceptions)

        # Overall adequacy
        model_adequate = passes_kupiec and christ_pval > significance_level and traffic_light != TrafficLightZone.RED

        return BacktestResult(
            num_observations=n,
            num_exceptions=num_exceptions,
            exception_rate=exception_rate,
            expected_exceptions=expected_exceptions,
            kupiec_statistic=kupiec_stat,
            kupiec_pvalue=kupiec_pval,
            passes_kupiec=passes_kupiec,
            christoffersen_statistic=christ_stat,
            christoffersen_pvalue=christ_pval,
            traffic_light_zone=traffic_light,
            model_adequate=model_adequate
        )

    def _kupiec_test(self, n: int, x: int, p: float) -> Tuple[float, float]:
        """
        Kupiec Proportion of Failures test.

        LR = -2 * ln((p^x * (1-p)^(n-x)) / (p_hat^x * (1-p_hat)^(n-x)))
        ~ Chi-squared(1) under null hypothesis

        Returns:
            (test_statistic, p_value)
        """
        if x == 0 or x == n:
            return 0.0, 1.0

        p_hat = x / n

        # Log-likelihood ratio
        lr = -2 * (x * math.log(p / p_hat) + (n - x) * math.log((1 - p) / (1 - p_hat)))

        # P-value from chi-squared(1)
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(lr, df=1)

        return lr, p_value

    def _christoffersen_test(self, exceptions: List[int]) -> Tuple[float, float]:
        """
        Christoffersen conditional coverage test.

        Tests independence of exceptions (no clustering).

        Returns:
            (test_statistic, p_value)
        """
        # Count transitions
        n00, n01, n10, n11 = 0, 0, 0, 0

        for i in range(len(exceptions) - 1):
            if exceptions[i] == 0 and exceptions[i+1] == 0:
                n00 += 1
            elif exceptions[i] == 0 and exceptions[i+1] == 1:
                n01 += 1
            elif exceptions[i] == 1 and exceptions[i+1] == 0:
                n10 += 1
            else:
                n11 += 1

        # Transition probabilities
        n0 = n00 + n01
        n1 = n10 + n11

        if n0 == 0 or n1 == 0:
            return 0.0, 1.0

        pi0 = n01 / n0 if n0 > 0 else 0
        pi1 = n11 / n1 if n1 > 0 else 0
        pi = (n01 + n11) / (n0 + n1) if (n0 + n1) > 0 else 0

        # LR test statistic
        if pi0 == 0 or pi1 == 0 or pi == 0 or pi == 1:
            return 0.0, 1.0

        lr = -2 * (
            n00 * math.log(1 - pi) + n01 * math.log(pi) +
            n10 * math.log(1 - pi) + n11 * math.log(pi) -
            n00 * math.log(1 - pi0) - n01 * math.log(pi0) -
            n10 * math.log(1 - pi1) - n11 * math.log(pi1)
        )

        # Chi-squared(1)
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(lr, df=1)

        return lr, p_value

    def _traffic_light_zone(self, num_exceptions: int) -> TrafficLightZone:
        """
        Determine Basel traffic light zone.

        For 250 trading days at 99% confidence:
        - Green: 0-4 exceptions
        - Yellow: 5-9 exceptions
        - Red: 10+ exceptions
        """
        if num_exceptions <= 4:
            return TrafficLightZone.GREEN
        elif num_exceptions <= 9:
            return TrafficLightZone.YELLOW
        else:
            return TrafficLightZone.RED


def backtest_var_model(
    actual_returns: List[float],
    var_estimates: List[float],
    confidence_level: float = 0.99
) -> BacktestResult:
    """
    Quick backtest of VaR model.

    Args:
        actual_returns: Actual portfolio returns
        var_estimates: VaR estimates for each period
        confidence_level: VaR confidence level

    Returns:
        BacktestResult
    """
    backtester = VaRBacktester(confidence_level)
    for actual, var_est in zip(actual_returns, var_estimates):
        backtester.add_observation(actual, var_est)

    return backtester.run_tests()
