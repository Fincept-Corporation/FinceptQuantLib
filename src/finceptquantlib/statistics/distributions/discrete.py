"""
Discrete Probability Distributions

Standard discrete distributions:
- Binomial
- Poisson
- Negative Binomial
- Geometric
- Hypergeometric
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# =============================================================================
# Binomial Distribution
# =============================================================================

@dataclass
class Binomial:
    """
    Binomial distribution.

    P(X = k) = C(n,k) * p^k * (1-p)^(n-k)

    Attributes:
        n: Number of trials
        p: Probability of success
    """
    n: int
    p: float

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ValueError("n must be non-negative")
        if not 0 <= self.p <= 1:
            raise ValueError("p must be in [0, 1]")

    def pmf(self, k: int) -> float:
        """Probability mass function P(X = k)."""
        if k < 0 or k > self.n:
            return 0.0
        return _binomial_coef(self.n, k) * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k: int) -> float:
        """Cumulative distribution function P(X ≤ k)."""
        if k < 0:
            return 0.0
        if k >= self.n:
            return 1.0
        return sum(self.pmf(i) for i in range(k + 1))

    def mean(self) -> float:
        return self.n * self.p

    def variance(self) -> float:
        return self.n * self.p * (1 - self.p)

    def std(self) -> float:
        return math.sqrt(self.variance())

    def skewness(self) -> float:
        q = 1 - self.p
        denom = self.n * self.p * q
        if denom <= 0:  # degenerate point mass (p in {0, 1}) — moments undefined
            return None  # undefined moment (degenerate) -> JSON null, not invalid NaN
        return (q - self.p) / math.sqrt(denom)

    def kurtosis(self) -> float:
        """Excess kurtosis."""
        q = 1 - self.p
        denom = self.n * self.p * q
        if denom <= 0:  # degenerate point mass (p in {0, 1}) — moments undefined
            return None  # undefined moment (degenerate) -> JSON null, not invalid NaN
        return (1 - 6 * self.p * q) / denom

    def mode(self) -> int:
        """Mode of the distribution."""
        return int(math.floor((self.n + 1) * self.p))


# =============================================================================
# Poisson Distribution
# =============================================================================

@dataclass
class Poisson:
    """
    Poisson distribution.

    P(X = k) = λ^k * e^(-λ) / k!

    Attributes:
        lam: Rate parameter (λ)
    """
    lam: float

    def __post_init__(self) -> None:
        if self.lam < 0:
            raise ValueError("Lambda must be non-negative")

    def pmf(self, k: int) -> float:
        """Probability mass function."""
        if k < 0:
            return 0.0
        if self.lam == 0:
            return 1.0 if k == 0 else 0.0
        return math.exp(-self.lam + k * math.log(self.lam) - _log_factorial(k))

    def cdf(self, k: int) -> float:
        """Cumulative distribution function."""
        if k < 0:
            return 0.0
        return sum(self.pmf(i) for i in range(k + 1))

    def mean(self) -> float:
        return self.lam

    def variance(self) -> float:
        return self.lam

    def std(self) -> float:
        return math.sqrt(self.lam)

    def skewness(self) -> float:
        return 1 / math.sqrt(self.lam) if self.lam > 0 else float('inf')

    def kurtosis(self) -> float:
        return 1 / self.lam if self.lam > 0 else float('inf')

    def mode(self) -> int:
        return int(math.floor(self.lam))


# =============================================================================
# Negative Binomial Distribution
# =============================================================================

@dataclass
class NegativeBinomial:
    """
    Negative Binomial distribution.

    Number of failures before r successes.

    P(X = k) = C(k+r-1, k) * p^r * (1-p)^k

    Attributes:
        r: Number of successes
        p: Probability of success
    """
    r: float  # Can be non-integer (generalized)
    p: float

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ValueError("r must be positive")
        if not 0 < self.p <= 1:
            raise ValueError("p must be in (0, 1]")

    def pmf(self, k: int) -> float:
        """Probability mass function."""
        if k < 0:
            return 0.0
        # Using gamma function for generalized form
        coef = (math.gamma(k + self.r) /
                (math.gamma(self.r) * math.factorial(k)))
        return coef * (self.p ** self.r) * ((1 - self.p) ** k)

    def cdf(self, k: int) -> float:
        """Cumulative distribution function."""
        if k < 0:
            return 0.0
        return sum(self.pmf(i) for i in range(k + 1))

    def mean(self) -> float:
        return self.r * (1 - self.p) / self.p

    def variance(self) -> float:
        return self.r * (1 - self.p) / (self.p ** 2)

    def skewness(self) -> float:
        denom = self.r * (1 - self.p)
        if denom <= 0:  # degenerate (p == 1) — moments undefined
            return None  # undefined moment (degenerate) -> JSON null, not invalid NaN
        return (2 - self.p) / math.sqrt(denom)

    def kurtosis(self) -> float:
        denom = self.r * (1 - self.p)
        if denom <= 0:  # degenerate (p == 1) — moments undefined
            return None  # undefined moment (degenerate) -> JSON null, not invalid NaN
        return 6 / self.r + (self.p ** 2) / denom


# =============================================================================
# Geometric Distribution
# =============================================================================

@dataclass
class Geometric:
    """
    Geometric distribution.

    Number of trials until first success.

    P(X = k) = (1-p)^(k-1) * p  for k = 1, 2, ...

    Attributes:
        p: Probability of success
    """
    p: float

    def __post_init__(self) -> None:
        if not 0 < self.p <= 1:
            raise ValueError("p must be in (0, 1]")

    def pmf(self, k: int) -> float:
        """Probability mass function."""
        if k < 1:
            return 0.0
        return ((1 - self.p) ** (k - 1)) * self.p

    def cdf(self, k: int) -> float:
        """Cumulative distribution function."""
        if k < 1:
            return 0.0
        return 1 - (1 - self.p) ** k

    def ppf(self, q: float) -> int:
        """Quantile function."""
        if q <= 0:
            return 1
        if q >= 1:
            return float('inf')
        return int(math.ceil(math.log(1 - q) / math.log(1 - self.p)))

    def mean(self) -> float:
        return 1 / self.p

    def variance(self) -> float:
        return (1 - self.p) / (self.p ** 2)

    def median(self) -> int:
        return int(math.ceil(-1 / math.log2(1 - self.p)))


# =============================================================================
# Hypergeometric Distribution
# =============================================================================

@dataclass
class Hypergeometric:
    """
    Hypergeometric distribution.

    Sampling without replacement.

    P(X = k) = C(K,k) * C(N-K, n-k) / C(N, n)

    Attributes:
        N: Population size
        K: Number of success states in population
        n: Number of draws
    """
    N: int  # Population size
    K: int  # Success states
    n: int  # Draws

    def __post_init__(self) -> None:
        if self.N < 0:
            raise ValueError("N must be non-negative")
        if self.K < 0 or self.K > self.N:
            raise ValueError("K must be in [0, N]")
        if self.n < 0 or self.n > self.N:
            raise ValueError("n must be in [0, N]")

    def pmf(self, k: int) -> float:
        """Probability mass function."""
        if k < max(0, self.n - (self.N - self.K)):
            return 0.0
        if k > min(self.n, self.K):
            return 0.0

        return (_binomial_coef(self.K, k) *
                _binomial_coef(self.N - self.K, self.n - k) /
                _binomial_coef(self.N, self.n))

    def cdf(self, k: int) -> float:
        """Cumulative distribution function."""
        k_min = max(0, self.n - (self.N - self.K))
        if k < k_min:
            return 0.0
        k_max = min(self.n, self.K)
        if k >= k_max:
            return 1.0
        return sum(self.pmf(i) for i in range(k_min, k + 1))

    def mean(self) -> float:
        return self.n * self.K / self.N

    def variance(self) -> float:
        return (self.n * self.K * (self.N - self.K) * (self.N - self.n) /
                (self.N ** 2 * (self.N - 1)))

    def mode(self) -> int:
        return int(math.floor((self.n + 1) * (self.K + 1) / (self.N + 2)))


# =============================================================================
# Helper Functions
# =============================================================================

def _binomial_coef(n: int, k: int) -> float:
    """Binomial coefficient C(n, k) = n! / (k! * (n-k)!)."""
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    k = min(k, n - k)
    result = 1.0
    for i in range(k):
        result = result * (n - i) / (i + 1)
    return result


def _log_factorial(n: int) -> float:
    """Log factorial using Stirling approximation for large n."""
    if n < 0:
        return float('inf')
    if n <= 1:
        return 0.0
    if n < 20:
        return sum(math.log(i) for i in range(2, n + 1))
    # Stirling approximation
    return n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n)


# =============================================================================
# Probability Generating Functions
# =============================================================================

def binomial_pgf(n: int, p: float, z: float) -> float:
    """
    Probability generating function for Binomial(n, p).

    G(z) = (1 - p + pz)^n
    """
    return (1 - p + p * z) ** n


def poisson_pgf(lam: float, z: float) -> float:
    """
    Probability generating function for Poisson(λ).

    G(z) = exp(λ(z - 1))
    """
    return math.exp(lam * (z - 1))


def geometric_pgf(p: float, z: float) -> float:
    """
    Probability generating function for Geometric(p).

    G(z) = pz / (1 - (1-p)z)
    """
    return p * z / (1 - (1 - p) * z)
