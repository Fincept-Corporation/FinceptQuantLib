"""
Statistical Methods Module

Comprehensive statistical toolkit:

- distributions/: Probability distributions (continuous, discrete)
- timeseries/: Time series models (ARIMA, GARCH)
- regression/: Regression methods
- hypothesis/: Statistical tests
- bayesian/: Bayesian methods

Example:
    >>> from finceptquantlib.statistics import Normal, GARCH, AR
    >>>
    >>> # Normal distribution
    >>> n = Normal(mu=0, sigma=1)
    >>> n.cdf(1.96)  # ≈ 0.975
    0.975002
    >>>
    >>> # Fit GARCH model
    >>> garch = GARCH(p=1, q=1)
    >>> result = garch.fit(returns)  # doctest: +SKIP
"""

from finceptquantlib.statistics.distributions import (
    # Continuous
    Normal,
    StudentT,
    ChiSquared,
    FDistribution,
    Gamma,
    Beta,
    Exponential,
    Lognormal,
    # Discrete
    Binomial,
    Poisson,
    NegativeBinomial,
    Geometric,
    Hypergeometric,
)

from finceptquantlib.statistics.timeseries import (
    AR,
    MA,
    ARIMA,
    ARIMAResult,
    GARCH,
    EGARCH,
    GJRGARCH,
    GARCHResult,
)

__all__ = [
    # Continuous distributions
    "Normal",
    "StudentT",
    "ChiSquared",
    "FDistribution",
    "Gamma",
    "Beta",
    "Exponential",
    "Lognormal",
    # Discrete distributions
    "Binomial",
    "Poisson",
    "NegativeBinomial",
    "Geometric",
    "Hypergeometric",
    # Time series
    "AR",
    "MA",
    "ARIMA",
    "ARIMAResult",
    "GARCH",
    "EGARCH",
    "GJRGARCH",
    "GARCHResult",
]
