"""
Generalized Pareto Distribution (GPD)

Used in Peaks Over Threshold (POT) approach for modeling excesses
over a high threshold.

The GPD is the limiting distribution for excesses over a threshold
as the threshold approaches the endpoint of the distribution.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class GPD:
    """
    Generalized Pareto Distribution.

    F(x) = 1 - (1 + ξx/σ)^(-1/ξ)  for ξ ≠ 0
    F(x) = 1 - exp(-x/σ)          for ξ = 0 (Exponential)

    Domain:
    - x ≥ 0 when ξ ≥ 0
    - 0 ≤ x ≤ -σ/ξ when ξ < 0

    Attributes:
        sigma: Scale parameter (σ > 0)
        xi: Shape parameter (ξ)
    """
    sigma: float = 1.0
    xi: float = 0.0

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("Scale parameter sigma must be positive")

    def _support_check(self, x: float) -> bool:
        """Check if x is in support of distribution."""
        if x < 0:
            return False
        if self.xi < 0:
            return x <= -self.sigma / self.xi
        return True

    def pdf(self, x: float) -> float:
        """
        Probability density function.

        f(x) = (1/σ)(1 + ξx/σ)^(-1-1/ξ)  for ξ ≠ 0
        f(x) = (1/σ)exp(-x/σ)             for ξ = 0
        """
        if not self._support_check(x):
            return 0.0

        if abs(self.xi) < 1e-10:
            return math.exp(-x / self.sigma) / self.sigma
        else:
            t = 1 + self.xi * x / self.sigma
            if t <= 0:
                return 0.0
            return (t ** (-1 - 1 / self.xi)) / self.sigma

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function.

        F(x) = 1 - (1 + ξx/σ)^(-1/ξ)
        """
        if x < 0:
            return 0.0
        if self.xi < 0 and x >= -self.sigma / self.xi:
            return 1.0

        if abs(self.xi) < 1e-10:
            return 1 - math.exp(-x / self.sigma)
        else:
            t = 1 + self.xi * x / self.sigma
            if t <= 0:
                return 1.0 if self.xi < 0 else 0.0
            return 1 - t ** (-1 / self.xi)

    def ppf(self, p: float) -> float:
        """
        Percent point function (quantile function).

        x = σ((1-p)^(-ξ) - 1)/ξ  for ξ ≠ 0
        x = -σ ln(1-p)           for ξ = 0
        """
        if p < 0 or p >= 1:
            raise ValueError("p must be in [0, 1)")

        if p == 0:
            return 0.0

        if abs(self.xi) < 1e-10:
            return -self.sigma * math.log(1 - p)
        else:
            return self.sigma * ((1 - p) ** (-self.xi) - 1) / self.xi

    def sample(self, n: int, seed: Optional[int] = None) -> List[float]:
        """Generate random samples using inverse transform."""
        rng = random.Random(seed)

        return [self.ppf(rng.random()) for _ in range(n)]

    def mean(self) -> float:
        """
        Expected value.

        E[X] = σ/(1-ξ)  for ξ < 1
        """
        if self.xi >= 1:
            return float('inf')
        return self.sigma / (1 - self.xi)

    def variance(self) -> float:
        """
        Variance.

        Var[X] = σ²/((1-ξ)²(1-2ξ))  for ξ < 0.5
        """
        if self.xi >= 0.5:
            return float('inf')
        return self.sigma ** 2 / ((1 - self.xi) ** 2 * (1 - 2 * self.xi))

    def excess_mean(self, x: float) -> float:
        """
        Mean excess function e(x) = E[X - x | X > x].

        For GPD: e(x) = (σ + ξx)/(1 - ξ)

        This is linear in x, a key property of GPD.
        """
        if self.xi >= 1:
            return float('inf')
        return (self.sigma + self.xi * x) / (1 - self.xi)

    def var_excess(self, threshold: float, prob: float) -> float:
        """
        VaR for excesses over threshold.

        P(X > x | X > threshold) = prob

        Returns x such that prob of excesses exceed x.
        """
        # Conditional tail probability
        return threshold + self.ppf(1 - prob)

    @classmethod
    def fit_mle(cls, excesses: List[float], max_iter: int = 100, tol: float = 1e-6) -> "GPD":
        """
        Fit GPD to excesses using Maximum Likelihood Estimation.

        Maximises the log-likelihood over (sigma, xi) jointly, starting from
        the probability-weighted-moment estimates.

        Args:
            excesses: Data (excesses over threshold)
            max_iter: Maximum iterations (scaled for Nelder-Mead)
            tol: Convergence tolerance

        Returns:
            Fitted GPD
        """
        from scipy.optimize import minimize

        n = len(excesses)
        if n < 2:
            raise ValueError("Need at least 2 data points")

        data = [max(1e-12, x) for x in excesses]  # Ensure positive
        start = cls.fit_pwm(data)

        def negative_log_likelihood(params: List[float]) -> float:
            log_sigma, xi = params
            sigma = math.exp(log_sigma)  # keeps sigma > 0
            nll = n * log_sigma
            if abs(xi) < 1e-10:
                return nll + sum(data) / sigma
            for x in data:
                w = 1 + xi * x / sigma
                if w <= 0:
                    return float("inf")  # observation outside the support
                nll += (1 + 1 / xi) * math.log(w)
            return nll

        x0 = [math.log(start.sigma), start.xi]
        if not math.isfinite(negative_log_likelihood(x0)):
            # PWM start can sit outside the support when xi < 0
            x0 = [math.log(max(data)), 0.0]

        result = minimize(
            negative_log_likelihood,
            x0,
            method="Nelder-Mead",
            options={"maxiter": max_iter * 20, "xatol": tol, "fatol": tol},
        )
        if not math.isfinite(result.fun):
            return start
        log_sigma, xi = result.x
        return cls(sigma=math.exp(log_sigma), xi=float(xi))

    @classmethod
    def fit_pwm(cls, excesses: List[float]) -> "GPD":
        """
        Fit GPD using Probability Weighted Moments.

        More robust than MLE for small samples.
        """
        n = len(excesses)
        if n < 2:
            raise ValueError("Need at least 2 data points")

        x_sorted = sorted(excesses)

        # Compute PWMs
        a0 = sum(x_sorted) / n
        a1 = sum((n - i - 1) / (n - 1) * x_sorted[i] for i in range(n)) / n

        # Estimate parameters
        xi = 2 - a0 / (a0 - 2 * a1)
        sigma = 2 * a0 * a1 / (a0 - 2 * a1)

        return cls(sigma=max(sigma, 1e-6), xi=xi)


@dataclass
class POTResult:
    """Result of Peaks Over Threshold analysis."""
    threshold: float
    n_exceedances: int
    exceedance_rate: float
    gpd: GPD
    var_estimates: dict
    es_estimates: dict


def peaks_over_threshold(
    data: List[float],
    threshold: Optional[float] = None,
    quantile: float = 0.95,
    fit_method: str = "mle"
) -> POTResult:
    """
    Perform Peaks Over Threshold (POT) analysis.

    Args:
        data: Data series (typically losses)
        threshold: Explicit threshold (overrides quantile)
        quantile: Quantile for threshold selection (default 95%)
        fit_method: "mle" or "pwm"

    Returns:
        POTResult with fitted GPD and risk estimates
    """
    n = len(data)
    if n < 20:
        raise ValueError("Need at least 20 data points")

    # Determine threshold
    if threshold is None:
        sorted_data = sorted(data)
        idx = int(quantile * n)
        threshold = sorted_data[idx]

    # Extract excesses
    excesses = [x - threshold for x in data if x > threshold]
    n_exceed = len(excesses)

    if n_exceed < 10:
        raise ValueError(f"Only {n_exceed} exceedances; need at least 10")

    exceedance_rate = n_exceed / n

    # Fit GPD
    if fit_method == "mle":
        gpd = GPD.fit_mle(excesses)
    else:
        gpd = GPD.fit_pwm(excesses)

    # Compute VaR and ES at various levels
    var_estimates = {}
    es_estimates = {}

    for p in [0.95, 0.99, 0.999]:
        # VaR_p = u + (σ/ξ)*((n/N_u * (1-p))^(-ξ) - 1), with N_u/n = exceedance_rate
        tail_ratio = (1 - p) / exceedance_rate
        if abs(gpd.xi) < 1e-10:
            var_p = threshold - gpd.sigma * math.log(tail_ratio)
        else:
            var_p = threshold + (gpd.sigma / gpd.xi) * (tail_ratio ** (-gpd.xi) - 1)

        var_estimates[p] = var_p

        # ES_p = VaR_p / (1 - ξ) + (σ - ξu) / (1 - ξ)
        if gpd.xi < 1:
            es_p = var_p / (1 - gpd.xi) + (gpd.sigma - gpd.xi * threshold) / (1 - gpd.xi)
            es_estimates[p] = es_p

    return POTResult(
        threshold=threshold,
        n_exceedances=n_exceed,
        exceedance_rate=exceedance_rate,
        gpd=gpd,
        var_estimates=var_estimates,
        es_estimates=es_estimates
    )


def mean_excess_plot(data: List[float], n_thresholds: int = 50) -> List[Tuple[float, float]]:
    """
    Compute mean excess function for threshold selection.

    The mean excess function e(u) = E[X - u | X > u] should be
    approximately linear in u for GPD-distributed tails.

    Args:
        data: Data series
        n_thresholds: Number of threshold values

    Returns:
        List of (threshold, mean_excess) pairs
    """
    sorted_data = sorted(data)
    n = len(data)

    # Use quantiles as thresholds
    results = []
    for i in range(1, n_thresholds + 1):
        q = i / (n_thresholds + 1)
        idx = int(q * n)
        u = sorted_data[idx]

        excesses = [x - u for x in sorted_data if x > u]
        if len(excesses) > 5:
            mean_excess = sum(excesses) / len(excesses)
            results.append((u, mean_excess))

    return results


def hill_estimator(data: List[float], k: Optional[int] = None) -> Tuple[float, float]:
    """
    Hill estimator for tail index ξ.

    The Hill estimator estimates ξ assuming Pareto-like tails.

    H_{n,k} = (1/k) * Σᵢ₌₁ᵏ ln(X_{(n-i+1)} / X_{(n-k)})

    Args:
        data: Data series
        k: Number of upper order statistics to use (default: n^0.6)

    Returns:
        (xi_estimate, standard_error)
    """
    n = len(data)
    if k is None:
        k = int(n ** 0.6)

    k = min(k, n - 1)
    if k < 2:
        raise ValueError("Need more data points")

    sorted_data = sorted(data, reverse=True)

    # Hill estimator
    x_k = sorted_data[k]
    if x_k <= 0:
        raise ValueError("Data must be positive for Hill estimator")

    hill_sum = sum(math.log(sorted_data[i] / x_k) for i in range(k))
    xi = hill_sum / k

    # Asymptotic standard error
    se = xi / math.sqrt(k)

    return xi, se
