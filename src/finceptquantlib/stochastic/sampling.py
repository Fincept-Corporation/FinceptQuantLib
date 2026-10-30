"""
Random Sampling Utilities

Core random number generation and sampling functions used throughout
stochastic simulation modules.

This module provides the single source of truth for sampling algorithms.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple


# =============================================================================
# Poisson Sampling
# =============================================================================

def poisson_sample(lambda_: float, rng: Optional[random.Random] = None) -> int:
    """
    Sample from Poisson distribution using Knuth algorithm.

    P(X = k) = λ^k * e^(-λ) / k!

    For small λ (< 30), uses direct method.
    For large λ, uses normal approximation.

    Args:
        lambda_: Rate parameter (λ > 0)
        rng: Random number generator (uses global if None)

    Returns:
        Poisson random variate
    """
    if lambda_ <= 0:
        return 0

    if rng is None:
        rng = random

    # For large lambda, use normal approximation
    if lambda_ > 30:
        return max(0, int(round(rng.gauss(lambda_, math.sqrt(lambda_)))))

    # Knuth algorithm for small lambda
    L = math.exp(-lambda_)
    k = 0
    p = 1.0

    while p > L:
        k += 1
        p *= rng.random()

    return k - 1


def poisson_samples(lambda_: float, n: int, seed: Optional[int] = None) -> List[int]:
    """
    Generate multiple Poisson samples.

    Args:
        lambda_: Rate parameter
        n: Number of samples
        seed: Random seed

    Returns:
        List of n Poisson random variates
    """
    rng = random.Random(seed) if seed is not None else random
    return [poisson_sample(lambda_, rng) for _ in range(n)]


# =============================================================================
# Exponential Sampling
# =============================================================================

def exponential_sample(rate: float, rng: Optional[random.Random] = None) -> float:
    """
    Sample from exponential distribution using inverse transform.

    f(x) = λ * e^(-λx)

    Args:
        rate: Rate parameter (λ > 0)
        rng: Random number generator

    Returns:
        Exponential random variate
    """
    if rate <= 0:
        raise ValueError("Rate must be positive")

    if rng is None:
        rng = random

    u = rng.random()
    # Avoid log(0) — random() returns [0, 1), so use 1-u to get (0, 1]
    return -math.log(1.0 - u) / rate


def exponential_samples(rate: float, n: int, seed: Optional[int] = None) -> List[float]:
    """Generate multiple exponential samples."""
    rng = random.Random(seed) if seed is not None else random
    return [exponential_sample(rate, rng) for _ in range(n)]


# =============================================================================
# Gamma Sampling
# =============================================================================

def gamma_sample(alpha: float, beta: float = 1.0, rng: Optional[random.Random] = None) -> float:
    """
    Sample from gamma distribution using Marsaglia and Tsang's method.

    f(x) = β^α / Γ(α) * x^(α-1) * e^(-βx)

    Args:
        alpha: Shape parameter (α > 0)
        beta: Rate parameter (β > 0)
        rng: Random number generator

    Returns:
        Gamma random variate
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("Shape and rate must be positive")

    if rng is None:
        rng = random

    # For alpha < 1, use Ahrens-Dieter method
    if alpha < 1:
        return gamma_sample(1 + alpha, beta, rng) * (rng.random() ** (1 / alpha))

    # Marsaglia and Tsang's method for alpha >= 1
    d = alpha - 1 / 3
    c = 1 / math.sqrt(9 * d)

    while True:
        x = rng.gauss(0, 1)
        v = (1 + c * x) ** 3

        if v > 0:
            u = rng.random()
            if u < 1 - 0.0331 * (x ** 2) ** 2:
                return d * v / beta
            if math.log(u) < 0.5 * x ** 2 + d * (1 - v + math.log(v)):
                return d * v / beta


def gamma_samples(alpha: float, beta: float, n: int, seed: Optional[int] = None) -> List[float]:
    """Generate multiple gamma samples."""
    rng = random.Random(seed) if seed is not None else random
    return [gamma_sample(alpha, beta, rng) for _ in range(n)]


# =============================================================================
# Chi-Squared Sampling
# =============================================================================

def chi_squared_sample(df: float, rng: Optional[random.Random] = None) -> float:
    """
    Sample from chi-squared distribution.

    χ²(k) = Gamma(k/2, 1/2)

    Args:
        df: Degrees of freedom (k > 0)
        rng: Random number generator

    Returns:
        Chi-squared random variate
    """
    return gamma_sample(df / 2, 0.5, rng)


# =============================================================================
# Beta Sampling
# =============================================================================

def beta_sample(alpha: float, beta: float, rng: Optional[random.Random] = None) -> float:
    """
    Sample from beta distribution.

    If X ~ Gamma(α, 1) and Y ~ Gamma(β, 1), then X/(X+Y) ~ Beta(α, β)

    Args:
        alpha: First shape parameter (α > 0)
        beta: Second shape parameter (β > 0)
        rng: Random number generator

    Returns:
        Beta random variate in [0, 1]
    """
    if rng is None:
        rng = random

    x = gamma_sample(alpha, 1.0, rng)
    y = gamma_sample(beta, 1.0, rng)
    return x / (x + y)


# =============================================================================
# Normal Jump Samplers (for Jump-Diffusion)
# =============================================================================

def normal_jump_sampler(mu: float, sigma: float, rng: Optional[random.Random] = None) -> float:
    """
    Sample jump size from normal distribution.

    Used in Merton jump-diffusion model.

    Args:
        mu: Mean jump size
        sigma: Jump size volatility
        rng: Random number generator

    Returns:
        Jump size
    """
    if rng is None:
        rng = random
    return rng.gauss(mu, sigma)


def double_exponential_sampler(
    p: float,
    eta1: float,
    eta2: float,
    rng: Optional[random.Random] = None
) -> float:
    """
    Sample from double exponential (asymmetric Laplace) distribution.

    Used in Kou jump-diffusion model.

    f(x) = p * η₁ * e^(-η₁x) for x > 0
           (1-p) * η₂ * e^(η₂x) for x < 0

    Args:
        p: Probability of positive jump (0 < p < 1)
        eta1: Rate of positive jumps (η₁ > 1 for finite mean)
        eta2: Rate of negative jumps (η₂ > 0)
        rng: Random number generator

    Returns:
        Jump size
    """
    if rng is None:
        rng = random

    if rng.random() < p:
        # Positive jump: exponential with rate eta1
        return exponential_sample(eta1, rng)
    else:
        # Negative jump: -exponential with rate eta2
        return -exponential_sample(eta2, rng)


def uniform_jump_sampler(a: float, b: float, rng: Optional[random.Random] = None) -> float:
    """
    Sample jump size from uniform distribution.

    Args:
        a: Lower bound
        b: Upper bound
        rng: Random number generator

    Returns:
        Jump size in [a, b]
    """
    if rng is None:
        rng = random
    return rng.uniform(a, b)


# =============================================================================
# Inverse Gaussian (Wald) Sampling
# =============================================================================

def inverse_gaussian_sample(
    mu: float,
    lambda_: float,
    rng: Optional[random.Random] = None
) -> float:
    """
    Sample from inverse Gaussian (Wald) distribution.

    Used in NIG and VG processes.

    Michael-Schucany-Haas algorithm.

    Args:
        mu: Mean (μ > 0)
        lambda_: Shape parameter (λ > 0)
        rng: Random number generator

    Returns:
        Inverse Gaussian random variate
    """
    if mu <= 0 or lambda_ <= 0:
        raise ValueError("Parameters must be positive")

    if rng is None:
        rng = random

    v = rng.gauss(0, 1)
    y = v * v
    x = mu + (mu * mu * y) / (2 * lambda_) - \
        (mu / (2 * lambda_)) * math.sqrt(4 * mu * lambda_ * y + mu * mu * y * y)

    if rng.random() <= mu / (mu + x):
        return x
    else:
        return mu * mu / x


# =============================================================================
# Multivariate Normal
# =============================================================================

def multivariate_normal_sample(
    mean: List[float],
    cov: List[List[float]],
    rng: Optional[random.Random] = None
) -> List[float]:
    """
    Sample from multivariate normal distribution.

    Uses Cholesky decomposition: X = μ + L @ Z where L @ L' = Σ

    Args:
        mean: Mean vector
        cov: Covariance matrix
        rng: Random number generator

    Returns:
        Sample vector
    """
    if rng is None:
        rng = random

    n = len(mean)

    # Cholesky decomposition
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(max(0, cov[i][i] - s))
            else:
                L[i][j] = (cov[i][j] - s) / L[j][j] if L[j][j] > 0 else 0

    # Generate standard normal vector
    z = [rng.gauss(0, 1) for _ in range(n)]

    # Transform: X = μ + L @ Z
    x = [mean[i] + sum(L[i][j] * z[j] for j in range(i + 1)) for i in range(n)]

    return x


def correlated_normals(
    rho: float,
    n: int,
    seed: Optional[int] = None
) -> Tuple[List[float], List[float]]:
    """
    Generate correlated standard normal pairs.

    Args:
        rho: Correlation coefficient
        n: Number of pairs
        seed: Random seed

    Returns:
        (z1, z2) where corr(z1, z2) = rho
    """
    rng = random.Random(seed) if seed is not None else random

    z1 = [rng.gauss(0, 1) for _ in range(n)]
    z_indep = [rng.gauss(0, 1) for _ in range(n)]

    # z2 = rho * z1 + sqrt(1 - rho^2) * z_indep
    sqrt_1_minus_rho2 = math.sqrt(1 - rho * rho)
    z2 = [rho * z1[i] + sqrt_1_minus_rho2 * z_indep[i] for i in range(n)]

    return z1, z2


# =============================================================================
# Antithetic and Variance Reduction
# =============================================================================

def antithetic_normal_samples(
    n: int,
    seed: Optional[int] = None
) -> Tuple[List[float], List[float]]:
    """
    Generate antithetic normal pairs for variance reduction.

    Returns (Z, -Z) which are negatively correlated.

    Args:
        n: Number of samples
        seed: Random seed

    Returns:
        (z, -z) standard normal samples
    """
    rng = random.Random(seed) if seed is not None else random

    z = [rng.gauss(0, 1) for _ in range(n)]
    z_anti = [-zi for zi in z]

    return z, z_anti


def sobol_sequence(dim: int, n: int, skip: int = 0) -> List[List[float]]:
    """
    Generate Sobol quasi-random sequence (simplified version).

    For production use, consider a full Sobol implementation.

    Args:
        dim: Dimension
        n: Number of points
        skip: Points to skip

    Returns:
        List of n points, each of dimension dim
    """
    # Simplified: use Halton sequence as placeholder
    # Full Sobol requires direction numbers
    points = []

    def halton(index: int, base: int) -> float:
        result = 0.0
        f = 1.0 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i //= base
            f /= base
        return result

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    for i in range(skip, skip + n):
        point = [halton(i + 1, primes[d % len(primes)]) for d in range(dim)]
        points.append(point)

    return points
