"""
Generalized Extreme Value (GEV) Distribution

The GEV distribution unifies the three extreme value distributions:
- Gumbel (Type I): ξ = 0
- Fréchet (Type II): ξ > 0
- Weibull (Type III): ξ < 0

Used in Block Maxima approach for modeling extremes.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GEV:
    """
    Generalized Extreme Value distribution.

    F(x) = exp(-(1 + ξ(x-μ)/σ)^(-1/ξ))  for ξ ≠ 0
    F(x) = exp(-exp(-(x-μ)/σ))           for ξ = 0 (Gumbel)

    Attributes:
        mu: Location parameter
        sigma: Scale parameter (σ > 0)
        xi: Shape parameter (ξ)
    """
    mu: float = 0.0
    sigma: float = 1.0
    xi: float = 0.0

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("Scale parameter sigma must be positive")

    def _standardize(self, x: float) -> float:
        """Standardize x to (x - μ) / σ."""
        return (x - self.mu) / self.sigma

    def _support_check(self, x: float) -> bool:
        """Check if x is in support of distribution."""
        z = self._standardize(x)
        if abs(self.xi) < 1e-10:
            return True
        if self.xi > 0:
            return z > -1 / self.xi
        return z < -1 / self.xi

    def pdf(self, x: float) -> float:
        """
        Probability density function.

        f(x) = (1/σ) * (1 + ξz)^(-1-1/ξ) * exp(-(1 + ξz)^(-1/ξ))
        """
        if not self._support_check(x):
            return 0.0

        z = self._standardize(x)

        if abs(self.xi) < 1e-10:
            # Gumbel case
            t = math.exp(-z)
            return t * math.exp(-t) / self.sigma
        else:
            t = 1 + self.xi * z
            if t <= 0:
                return 0.0
            return (t ** (-1 - 1 / self.xi)) * math.exp(-(t ** (-1 / self.xi))) / self.sigma

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function.

        F(x) = exp(-(1 + ξz)^(-1/ξ))
        """
        if not self._support_check(x):
            z = self._standardize(x)
            if abs(self.xi) < 1e-10:
                return 0.0
            if self.xi > 0:
                return 0.0 if z <= -1 / self.xi else 1.0
            return 1.0 if z >= -1 / self.xi else 0.0

        z = self._standardize(x)

        if abs(self.xi) < 1e-10:
            return math.exp(-math.exp(-z))
        else:
            t = 1 + self.xi * z
            if t <= 0:
                return 0.0 if self.xi > 0 else 1.0
            return math.exp(-(t ** (-1 / self.xi)))

    def ppf(self, p: float) -> float:
        """
        Percent point function (quantile function).

        x = μ + σ * ((-ln p)^(-ξ) - 1) / ξ  for ξ ≠ 0
        x = μ - σ * ln(-ln p)               for ξ = 0
        """
        if p <= 0 or p >= 1:
            raise ValueError("p must be in (0, 1)")

        if abs(self.xi) < 1e-10:
            return self.mu - self.sigma * math.log(-math.log(p))
        else:
            return self.mu + self.sigma * ((-math.log(p)) ** (-self.xi) - 1) / self.xi

    def sample(self, n: int, seed: Optional[int] = None) -> List[float]:
        """Generate random samples using inverse transform."""
        rng = random.Random(seed)

        return [self.ppf(rng.random()) for _ in range(n)]

    def mean(self) -> float:
        """
        Expected value.

        E[X] = μ + σ(Γ(1-ξ) - 1)/ξ  for ξ < 1, ξ ≠ 0
        E[X] = μ + σγ                for ξ = 0 (γ = Euler's constant)
        """
        if self.xi >= 1:
            return float('inf')

        if abs(self.xi) < 1e-10:
            # Euler-Mascheroni constant
            euler_gamma = 0.5772156649015329
            return self.mu + self.sigma * euler_gamma
        else:
            g1 = math.gamma(1 - self.xi)
            return self.mu + self.sigma * (g1 - 1) / self.xi

    def variance(self) -> float:
        """
        Variance.

        Var[X] = σ²(Γ(1-2ξ) - Γ(1-ξ)²)/ξ²  for ξ < 0.5
        Var[X] = σ²π²/6                      for ξ = 0
        """
        if self.xi >= 0.5:
            return float('inf')

        if abs(self.xi) < 1e-10:
            return self.sigma ** 2 * math.pi ** 2 / 6
        else:
            g1 = math.gamma(1 - self.xi)
            g2 = math.gamma(1 - 2 * self.xi)
            return self.sigma ** 2 * (g2 - g1 ** 2) / self.xi ** 2

    def return_level(self, return_period: float) -> float:
        """
        Compute return level for given return period.

        The return level x_T is exceeded on average once every T periods.

        x_T = F^(-1)(1 - 1/T)

        Args:
            return_period: T, the return period

        Returns:
            Return level x_T
        """
        if return_period <= 1:
            raise ValueError("Return period must be > 1")

        p = 1 - 1 / return_period
        return self.ppf(p)

    @classmethod
    def fit_pwm(cls, data: List[float]) -> "GEV":
        """
        Fit GEV using Probability Weighted Moments (PWM).

        This is more robust than MLE for small samples.
        """
        n = len(data)
        if n < 3:
            raise ValueError("Need at least 3 data points")

        # Sort data
        x_sorted = sorted(data)

        # Compute PWMs
        b0 = sum(x_sorted) / n
        b1 = sum((i / (n - 1)) * x_sorted[i] for i in range(n)) / n
        b2 = sum((i * (i - 1) / ((n - 1) * (n - 2))) * x_sorted[i] for i in range(n)) / n

        # Compute parameters from PWMs
        c = (2 * b1 - b0) / (3 * b2 - b0) - math.log(2) / math.log(3)

        # Hosking (1985) approximation for kappa = -xi, refined by Newton below
        xi = -(7.8590 * c + 2.9554 * c ** 2)

        for _ in range(20):
            g1 = math.gamma(1 - xi) if xi < 1 else float('inf')
            g2 = math.gamma(1 - 2 * xi) if xi < 0.5 else float('inf')

            if math.isinf(g1) or math.isinf(g2):
                break

            f = (3 ** xi - 1) / (2 ** xi - 1) - (3 * b2 - b0) / (2 * b1 - b0)
            df = (3 ** xi * math.log(3) * (2 ** xi - 1) - (3 ** xi - 1) * 2 ** xi * math.log(2)) / (2 ** xi - 1) ** 2

            if abs(df) < 1e-12:
                break

            xi_new = xi - f / df
            if abs(xi_new - xi) < 1e-8:
                break
            xi = xi_new

        # Compute sigma and mu
        if abs(xi) < 1e-10:
            sigma = (2 * b1 - b0) / math.log(2)
            mu = b0 - 0.5772156649015329 * sigma
        else:
            g1 = math.gamma(1 - xi)
            sigma = xi * (2 * b1 - b0) / (g1 * (2 ** xi - 1))
            mu = b0 - sigma * (g1 - 1) / xi

        return cls(mu=mu, sigma=sigma, xi=xi)

    @classmethod
    def fit_mle(cls, data: List[float], max_iter: int = 100, tol: float = 1e-6) -> "GEV":
        """
        Fit GEV using Maximum Likelihood Estimation.

        Maximises the log-likelihood over (mu, sigma, xi) jointly, starting from
        the probability-weighted-moment estimates.
        """
        from scipy.optimize import minimize

        gev = cls.fit_pwm(data)

        def negative_log_likelihood(params: List[float]) -> float:
            mu, log_sigma, xi = params
            sigma = math.exp(log_sigma)  # keeps sigma > 0
            nll = len(data) * log_sigma
            for x in data:
                z = (x - mu) / sigma
                if abs(xi) < 1e-10:
                    nll += z + math.exp(-z)
                else:
                    w = 1 + xi * z
                    if w <= 0:
                        return float("inf")  # observation outside the support
                    nll += (1 + 1 / xi) * math.log(w) + w ** (-1 / xi)
            return nll

        result = minimize(
            negative_log_likelihood,
            [gev.mu, math.log(gev.sigma), gev.xi],
            method="Nelder-Mead",
            options={"maxiter": max_iter * 20, "xatol": tol, "fatol": tol},
        )
        if not math.isfinite(result.fun):
            return gev
        mu, log_sigma, xi = result.x
        return cls(mu=float(mu), sigma=math.exp(log_sigma), xi=float(xi))


@dataclass
class Gumbel:
    """
    Gumbel (Type I) extreme value distribution.

    Special case of GEV with ξ = 0.

    F(x) = exp(-exp(-(x-μ)/σ))

    Attributes:
        mu: Location parameter
        sigma: Scale parameter (σ > 0)
    """
    mu: float = 0.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("Scale parameter sigma must be positive")
        self._gev = GEV(mu=self.mu, sigma=self.sigma, xi=0.0)

    def pdf(self, x: float) -> float:
        return self._gev.pdf(x)

    def cdf(self, x: float) -> float:
        return self._gev.cdf(x)

    def ppf(self, p: float) -> float:
        return self._gev.ppf(p)

    def sample(self, n: int, seed: Optional[int] = None) -> List[float]:
        return self._gev.sample(n, seed)

    def mean(self) -> float:
        euler_gamma = 0.5772156649015329
        return self.mu + self.sigma * euler_gamma

    def variance(self) -> float:
        return self.sigma ** 2 * math.pi ** 2 / 6


@dataclass
class Frechet:
    """
    Fréchet (Type II) extreme value distribution.

    Special case of GEV with ξ > 0.

    Attributes:
        mu: Location parameter
        sigma: Scale parameter (σ > 0)
        alpha: Shape parameter (α > 0), where ξ = 1/α
    """
    mu: float = 0.0
    sigma: float = 1.0
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("Scale parameter sigma must be positive")
        if self.alpha <= 0:
            raise ValueError("Shape parameter alpha must be positive")
        self._gev = GEV(mu=self.mu, sigma=self.sigma, xi=1 / self.alpha)

    def pdf(self, x: float) -> float:
        return self._gev.pdf(x)

    def cdf(self, x: float) -> float:
        return self._gev.cdf(x)

    def ppf(self, p: float) -> float:
        return self._gev.ppf(p)

    def sample(self, n: int, seed: Optional[int] = None) -> List[float]:
        return self._gev.sample(n, seed)


@dataclass
class Weibull:
    """
    Weibull (Type III) extreme value distribution.

    Special case of GEV with ξ < 0.

    Attributes:
        mu: Location parameter (upper endpoint)
        sigma: Scale parameter (σ > 0)
        alpha: Shape parameter (α > 0), where ξ = -1/α
    """
    mu: float = 0.0
    sigma: float = 1.0
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("Scale parameter sigma must be positive")
        if self.alpha <= 0:
            raise ValueError("Shape parameter alpha must be positive")
        self._gev = GEV(mu=self.mu, sigma=self.sigma, xi=-1 / self.alpha)

    def pdf(self, x: float) -> float:
        return self._gev.pdf(x)

    def cdf(self, x: float) -> float:
        return self._gev.cdf(x)

    def ppf(self, p: float) -> float:
        return self._gev.ppf(p)

    def sample(self, n: int, seed: Optional[int] = None) -> List[float]:
        return self._gev.sample(n, seed)
