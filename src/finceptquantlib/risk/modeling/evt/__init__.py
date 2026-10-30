"""
Extreme Value Theory (EVT)

Statistical methods for modeling extreme events:

Block Maxima (GEV):
- GEV: Generalized Extreme Value distribution
- Gumbel: Type I extreme value (ξ = 0)
- Frechet: Type II extreme value (ξ > 0)
- Weibull: Type III extreme value (ξ < 0)

Peaks Over Threshold (POT):
- GPD: Generalized Pareto Distribution
- peaks_over_threshold: Complete POT analysis
- mean_excess_plot: Threshold selection tool
- hill_estimator: Tail index estimation

Example:
    >>> from finceptquantlib.risk.modeling.evt import GEV, GPD, peaks_over_threshold
    >>> losses = GPD(sigma=0.01, xi=0.2).sample(2520, seed=7)  # ~10y of daily losses
    >>>
    >>> # Block maxima approach (monthly maxima)
    >>> block_maxima = [max(losses[i:i + 21]) for i in range(0, len(losses), 21)]
    >>> gev = GEV.fit_pwm(block_maxima)
    >>> level_100 = gev.return_level(100)  # exceeded once every 100 months
    >>>
    >>> # Peaks over threshold approach
    >>> result = peaks_over_threshold(losses, quantile=0.95)
    >>> var_99 = result.var_estimates[0.99]
    >>> es_99 = result.es_estimates[0.99]
    >>> 0 < var_99 < es_99
    True
"""

from .gev import (
    GEV,
    Gumbel,
    Frechet,
    Weibull,
)

from .gpd import (
    GPD,
    POTResult,
    peaks_over_threshold,
    mean_excess_plot,
    hill_estimator,
)

__all__ = [
    # Block Maxima
    "GEV",
    "Gumbel",
    "Frechet",
    "Weibull",
    # Peaks Over Threshold
    "GPD",
    "POTResult",
    "peaks_over_threshold",
    "mean_excess_plot",
    "hill_estimator",
]
