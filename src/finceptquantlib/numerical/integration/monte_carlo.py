"""
Monte Carlo Integration

Stochastic integration methods:
- Basic Monte Carlo
- Importance sampling
- Stratified sampling
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple
import random
import math


def monte_carlo_integrate(
    f: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    n_samples: int = 10000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Monte Carlo integration for multi-dimensional functions.

    Args:
        f: Function to integrate
        bounds: List of (lower, upper) bounds for each dimension
        n_samples: Number of random samples
        seed: Random seed for reproducibility

    Returns:
        (estimate, std_error): Integral estimate and standard error
    """
    rng = random.Random(seed)

    # Volume of integration region
    volume = 1.0
    for lo, hi in bounds:
        volume *= (hi - lo)

    # Sample and evaluate
    sum_f = 0.0
    sum_f2 = 0.0

    for _ in range(n_samples):
        # Generate uniform random point
        x = [rng.uniform(lo, hi) for lo, hi in bounds]

        fx = f(x)
        sum_f += fx
        sum_f2 += fx * fx

    # Estimate
    mean_f = sum_f / n_samples
    estimate = volume * mean_f

    # Standard error
    if n_samples > 1:
        var_f = (sum_f2 / n_samples - mean_f ** 2) * n_samples / (n_samples - 1)
        std_error = volume * math.sqrt(var_f / n_samples)
    else:
        std_error = float('inf')

    return estimate, std_error


def importance_sampling(
    f: Callable[[float], float],
    g: Callable[[float], float],
    sample_g: Callable[[], float],
    n_samples: int = 10000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Importance sampling for 1D integration.

    Computes integral of f(x) using samples from proposal distribution g(x).
    The integral is estimated as E[f(x)/g(x)] where x ~ g.

    Args:
        f: Integrand (unnormalized)
        g: Proposal density (must be normalized)
        sample_g: Function that samples from g
        n_samples: Number of samples
        seed: Random seed

    Returns:
        (estimate, std_error): Integral estimate and standard error

    Example:
        >>> # Integrate exp(-x^2) from 0 to inf using exponential proposal
        >>> import math
        >>> f = lambda x: math.exp(-x**2) if x >= 0 else 0
        >>> g = lambda x: math.exp(-x) if x >= 0 else 0  # Exponential(1)
        >>> sample_g = lambda: -math.log(random.random())
        >>> result, err = importance_sampling(f, g, sample_g)
    """
    # The sampler is user-supplied; seeding the global generator makes
    # samplers that draw from `random` reproducible.
    if seed is not None:
        random.seed(seed)

    sum_w = 0.0
    sum_w2 = 0.0

    for _ in range(n_samples):
        x = sample_g()
        gx = g(x)

        if gx > 1e-15:
            w = f(x) / gx
        else:
            w = 0.0

        sum_w += w
        sum_w2 += w * w

    # Estimate
    estimate = sum_w / n_samples

    # Standard error
    if n_samples > 1:
        var_w = (sum_w2 / n_samples - estimate ** 2) * n_samples / (n_samples - 1)
        std_error = math.sqrt(var_w / n_samples)
    else:
        std_error = float('inf')

    return estimate, std_error


def stratified_sampling(
    f: Callable[[float], float],
    a: float,
    b: float,
    n_strata: int = 10,
    samples_per_stratum: int = 100,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Stratified sampling for 1D integration.

    Divides domain into strata and samples uniformly within each.
    Reduces variance compared to pure Monte Carlo.

    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        n_strata: Number of strata
        samples_per_stratum: Samples per stratum
        seed: Random seed

    Returns:
        (estimate, std_error): Integral estimate and standard error
    """
    rng = random.Random(seed)

    stratum_width = (b - a) / n_strata
    total_samples = n_strata * samples_per_stratum

    sum_f = 0.0
    sum_f2 = 0.0

    for i in range(n_strata):
        stratum_a = a + i * stratum_width
        stratum_b = stratum_a + stratum_width

        for _ in range(samples_per_stratum):
            x = rng.uniform(stratum_a, stratum_b)
            fx = f(x)
            sum_f += fx
            sum_f2 += fx * fx

    # Estimate
    mean_f = sum_f / total_samples
    estimate = (b - a) * mean_f

    # Standard error (approximate)
    if total_samples > 1:
        var_f = (sum_f2 / total_samples - mean_f ** 2) * total_samples / (total_samples - 1)
        std_error = (b - a) * math.sqrt(var_f / total_samples)
    else:
        std_error = float('inf')

    return estimate, std_error
