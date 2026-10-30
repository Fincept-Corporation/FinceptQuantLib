"""
Continuous Probability Distributions

Standard continuous distributions with PDF, CDF, quantile functions:
- Normal (Gaussian)
- Student's t
- Chi-squared
- F-distribution
- Gamma
- Beta
- Exponential
- Lognormal
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from finceptquantlib.core.utilities.distributions import _regularized_incomplete_beta, norm_ppf


# =============================================================================
# Normal Distribution
# =============================================================================

@dataclass
class Normal:
    """
    Normal (Gaussian) distribution.

    f(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))

    Attributes:
        mu: Mean
        sigma: Standard deviation
    """
    mu: float = 0.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("Sigma must be positive")

    def pdf(self, x: float) -> float:
        """Probability density function."""
        z = (x - self.mu) / self.sigma
        return math.exp(-0.5 * z * z) / (self.sigma * math.sqrt(2 * math.pi))

    def cdf(self, x: float) -> float:
        """Cumulative distribution function."""
        z = (x - self.mu) / self.sigma
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    def ppf(self, p: float) -> float:
        """Percent point function (quantile)."""
        if p <= 0 or p >= 1:
            raise ValueError("p must be in (0, 1)")
        return self.mu + self.sigma * norm_ppf(p)

    def mean(self) -> float:
        return self.mu

    def variance(self) -> float:
        return self.sigma ** 2

    def std(self) -> float:
        return self.sigma

    def skewness(self) -> float:
        return 0.0

    def kurtosis(self) -> float:
        return 0.0  # Excess kurtosis

    def entropy(self) -> float:
        """Differential entropy."""
        return 0.5 * math.log(2 * math.pi * math.e * self.sigma ** 2)


# =============================================================================
# Student's t Distribution
# =============================================================================

@dataclass
class StudentT:
    """
    Student's t distribution.

    f(x) = Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) * (1 + x²/ν)^(-(ν+1)/2)

    Attributes:
        df: Degrees of freedom (ν)
        loc: Location parameter
        scale: Scale parameter
    """
    df: float
    loc: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.df <= 0:
            raise ValueError("Degrees of freedom must be positive")
        if self.scale <= 0:
            raise ValueError("Scale must be positive")

    def pdf(self, x: float) -> float:
        """Probability density function."""
        z = (x - self.loc) / self.scale
        coef = math.gamma((self.df + 1) / 2) / (
            math.sqrt(self.df * math.pi) * math.gamma(self.df / 2)
        )
        return coef * (1 + z ** 2 / self.df) ** (-(self.df + 1) / 2) / self.scale

    def cdf(self, x: float) -> float:
        """CDF using regularized incomplete beta function."""
        z = (x - self.loc) / self.scale
        # Use approximation for CDF
        return self._t_cdf(z, self.df)

    def _t_cdf(self, t: float, df: float) -> float:
        """Student t CDF approximation."""
        # Use incomplete beta function relation
        x = df / (df + t ** 2)
        if t >= 0:
            return 1 - 0.5 * _regularized_incomplete_beta(df / 2, 0.5, x)
        else:
            return 0.5 * _regularized_incomplete_beta(df / 2, 0.5, x)

    def mean(self) -> float:
        if self.df > 1:
            return self.loc
        return float('nan')

    def variance(self) -> float:
        if self.df > 2:
            return self.scale ** 2 * self.df / (self.df - 2)
        elif self.df > 1:
            return float('inf')
        return float('nan')

    def skewness(self) -> float:
        if self.df > 3:
            return 0.0
        return float('nan')

    def kurtosis(self) -> float:
        """Excess kurtosis."""
        if self.df > 4:
            return 6 / (self.df - 4)
        elif self.df > 2:
            return float('inf')
        return float('nan')


# =============================================================================
# Chi-Squared Distribution
# =============================================================================

@dataclass
class ChiSquared:
    """
    Chi-squared distribution.

    f(x) = x^(k/2-1) * e^(-x/2) / (2^(k/2) * Γ(k/2))

    Special case of Gamma(k/2, 2).

    Attributes:
        df: Degrees of freedom (k)
    """
    df: float

    def __post_init__(self) -> None:
        if self.df <= 0:
            raise ValueError("Degrees of freedom must be positive")

    def pdf(self, x: float) -> float:
        if x <= 0:
            return 0.0
        k = self.df
        return (x ** (k/2 - 1) * math.exp(-x/2) /
                (2 ** (k/2) * math.gamma(k/2)))

    def cdf(self, x: float) -> float:
        if x <= 0:
            return 0.0
        return _lower_incomplete_gamma(self.df / 2, x / 2) / math.gamma(self.df / 2)

    def mean(self) -> float:
        return self.df

    def variance(self) -> float:
        return 2 * self.df

    def skewness(self) -> float:
        return math.sqrt(8 / self.df)

    def kurtosis(self) -> float:
        return 12 / self.df


# =============================================================================
# F Distribution
# =============================================================================

@dataclass
class FDistribution:
    """
    F distribution.

    Ratio of two chi-squared distributions.

    Attributes:
        df1: Numerator degrees of freedom
        df2: Denominator degrees of freedom
    """
    df1: float
    df2: float

    def __post_init__(self) -> None:
        if self.df1 <= 0 or self.df2 <= 0:
            raise ValueError("Degrees of freedom must be positive")

    def pdf(self, x: float) -> float:
        if x <= 0:
            return 0.0
        d1, d2 = self.df1, self.df2
        num = ((d1/d2) ** (d1/2) * x ** (d1/2 - 1) *
               (1 + d1*x/d2) ** (-(d1+d2)/2))
        return num / _beta_function(d1/2, d2/2)

    def mean(self) -> float:
        if self.df2 > 2:
            return self.df2 / (self.df2 - 2)
        return float('nan')

    def variance(self) -> float:
        if self.df2 > 4:
            d1, d2 = self.df1, self.df2
            return (2 * d2 ** 2 * (d1 + d2 - 2) /
                    (d1 * (d2 - 2) ** 2 * (d2 - 4)))
        return float('nan')


# =============================================================================
# Gamma Distribution
# =============================================================================

@dataclass
class Gamma:
    """
    Gamma distribution.

    f(x) = x^(α-1) * e^(-x/β) / (β^α * Γ(α))

    Attributes:
        shape: Shape parameter (α, k)
        scale: Scale parameter (β, θ)
    """
    shape: float  # α or k
    scale: float = 1.0  # β or θ

    def __post_init__(self) -> None:
        if self.shape <= 0:
            raise ValueError("Shape must be positive")
        if self.scale <= 0:
            raise ValueError("Scale must be positive")

    def pdf(self, x: float) -> float:
        if x <= 0:
            return 0.0
        a, b = self.shape, self.scale
        return (x ** (a - 1) * math.exp(-x / b) /
                (b ** a * math.gamma(a)))

    def cdf(self, x: float) -> float:
        if x <= 0:
            return 0.0
        return (_lower_incomplete_gamma(self.shape, x / self.scale) /
                math.gamma(self.shape))

    def mean(self) -> float:
        return self.shape * self.scale

    def variance(self) -> float:
        return self.shape * self.scale ** 2

    def skewness(self) -> float:
        return 2 / math.sqrt(self.shape)

    def kurtosis(self) -> float:
        return 6 / self.shape


# =============================================================================
# Beta Distribution
# =============================================================================

@dataclass
class Beta:
    """
    Beta distribution on [0, 1].

    f(x) = x^(α-1) * (1-x)^(β-1) / B(α, β)

    Attributes:
        alpha: First shape parameter
        beta: Second shape parameter
    """
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("Parameters must be positive")

    def pdf(self, x: float) -> float:
        if x <= 0 or x >= 1:
            return 0.0
        a, b = self.alpha, self.beta
        return (x ** (a - 1) * (1 - x) ** (b - 1) /
                _beta_function(a, b))

    def cdf(self, x: float) -> float:
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        return _regularized_incomplete_beta(self.alpha, self.beta, x)

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return a * b / ((a + b) ** 2 * (a + b + 1))

    def mode(self) -> Optional[float]:
        a, b = self.alpha, self.beta
        if a > 1 and b > 1:
            return (a - 1) / (a + b - 2)
        return None


# =============================================================================
# Exponential Distribution
# =============================================================================

@dataclass
class Exponential:
    """
    Exponential distribution.

    f(x) = λ * e^(-λx)

    Attributes:
        rate: Rate parameter (λ = 1/scale)
    """
    rate: float = 1.0

    def __post_init__(self) -> None:
        if self.rate <= 0:
            raise ValueError("Rate must be positive")

    @property
    def scale(self) -> float:
        return 1 / self.rate

    def pdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        return self.rate * math.exp(-self.rate * x)

    def cdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        return 1 - math.exp(-self.rate * x)

    def ppf(self, p: float) -> float:
        if p <= 0 or p >= 1:
            raise ValueError("p must be in (0, 1)")
        return -math.log(1 - p) / self.rate

    def mean(self) -> float:
        return 1 / self.rate

    def variance(self) -> float:
        return 1 / self.rate ** 2

    def memoryless_property(self, s: float, t: float) -> float:
        """P(X > s + t | X > s) = P(X > t)."""
        return math.exp(-self.rate * t)


# =============================================================================
# Lognormal Distribution
# =============================================================================

@dataclass
class Lognormal:
    """
    Lognormal distribution.

    If X ~ Lognormal(μ, σ), then log(X) ~ Normal(μ, σ)

    Attributes:
        mu: Mean of log(X)
        sigma: Std dev of log(X)
    """
    mu: float = 0.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("Sigma must be positive")

    def pdf(self, x: float) -> float:
        if x <= 0:
            return 0.0
        return (math.exp(-(math.log(x) - self.mu) ** 2 / (2 * self.sigma ** 2)) /
                (x * self.sigma * math.sqrt(2 * math.pi)))

    def cdf(self, x: float) -> float:
        if x <= 0:
            return 0.0
        return 0.5 * (1 + math.erf((math.log(x) - self.mu) / (self.sigma * math.sqrt(2))))

    def ppf(self, p: float) -> float:
        if p <= 0 or p >= 1:
            raise ValueError("p must be in (0, 1)")
        return math.exp(self.mu + self.sigma * norm_ppf(p))

    def mean(self) -> float:
        return math.exp(self.mu + self.sigma ** 2 / 2)

    def variance(self) -> float:
        return (math.exp(self.sigma ** 2) - 1) * math.exp(2 * self.mu + self.sigma ** 2)

    def median(self) -> float:
        return math.exp(self.mu)

    def mode(self) -> float:
        return math.exp(self.mu - self.sigma ** 2)


# =============================================================================
# Helper Functions
# =============================================================================

def _beta_function(a: float, b: float) -> float:
    """Beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b)."""
    return math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))


def _lower_incomplete_gamma(a: float, x: float, max_iter: int = 100) -> float:
    """Lower incomplete gamma function γ(a, x)."""
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0

    # Series expansion for small x
    if x < a + 1:
        term = 1.0 / a
        sum_val = term
        for n in range(1, max_iter):
            term *= x / (a + n)
            sum_val += term
            if abs(term) < 1e-10:
                break
        return sum_val * math.exp(-x + a * math.log(x))

    # Continued fraction for large x
    return math.gamma(a) - _upper_incomplete_gamma(a, x)


def _upper_incomplete_gamma(a: float, x: float, max_iter: int = 100) -> float:
    """Upper incomplete gamma function Γ(a, x)."""
    # Continued fraction
    b = x + 1 - a
    c = 1e30
    d = 1 / b
    h = d

    for n in range(1, max_iter):
        an = -n * (n - a)
        b += 2
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-10:
            break

    return math.exp(-x + a * math.log(x)) * h

