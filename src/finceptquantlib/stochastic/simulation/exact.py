"""
Exact Simulation Methods

Analytical transition density sampling for special SDEs:
- Geometric Brownian Motion
- Ornstein-Uhlenbeck
- Cox-Ingersoll-Ross
- Heston stochastic volatility
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from finceptquantlib.stochastic.sampling import poisson_sample


@dataclass
class ExactSimulationResult:
    """Result of exact simulation."""
    times: List[float]
    paths: List[List[float]]
    n_paths: int
    n_steps: int


def simulate_gbm_exact(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    seed: Optional[int] = None
) -> ExactSimulationResult:
    """
    Exact simulation of Geometric Brownian Motion.

    dS = μS dt + σS dW

    Solution: S(t) = S(0) exp((μ - σ²/2)t + σW(t))

    Transition: S(t+Δt) = S(t) exp((μ - σ²/2)Δt + σ√Δt Z)
    where Z ~ N(0,1)

    Args:
        S0: Initial price
        mu: Drift rate
        sigma: Volatility
        T: Terminal time
        n_steps: Number of time steps
        n_paths: Number of paths
        seed: Random seed
    """
    rng = random.Random(seed) if seed is not None else random

    dt = T / n_steps
    drift = (mu - 0.5 * sigma ** 2) * dt
    vol = sigma * math.sqrt(dt)
    times = [i * dt for i in range(n_steps + 1)]

    paths = []
    for _ in range(n_paths):
        S = [S0]
        for _ in range(n_steps):
            Z = rng.gauss(0, 1)
            S_new = S[-1] * math.exp(drift + vol * Z)
            S.append(S_new)
        paths.append(S)

    return ExactSimulationResult(times, paths, n_paths, n_steps)


def simulate_ou_exact(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    seed: Optional[int] = None
) -> ExactSimulationResult:
    """
    Exact simulation of Ornstein-Uhlenbeck process.

    dX = κ(θ - X)dt + σdW

    Transition density:
    X(t+Δt)|X(t) ~ N(μ(Δt), σ²(Δt))

    where:
    μ(Δt) = θ + (X(t) - θ)e^(-κΔt)
    σ²(Δt) = (σ²/2κ)(1 - e^(-2κΔt))
    """
    rng = random.Random(seed) if seed is not None else random

    dt = T / n_steps
    exp_kdt = math.exp(-kappa * dt)
    var_dt = (sigma ** 2 / (2 * kappa)) * (1 - exp_kdt ** 2)
    std_dt = math.sqrt(var_dt)
    times = [i * dt for i in range(n_steps + 1)]

    paths = []
    for _ in range(n_paths):
        X = [X0]
        for _ in range(n_steps):
            mean = theta + (X[-1] - theta) * exp_kdt
            X_new = mean + std_dt * rng.gauss(0, 1)
            X.append(X_new)
        paths.append(X)

    return ExactSimulationResult(times, paths, n_paths, n_steps)


def simulate_cir_exact(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    seed: Optional[int] = None
) -> ExactSimulationResult:
    """
    Exact simulation of Cox-Ingersoll-Ross process.

    dr = κ(θ - r)dt + σ√r dW

    Transition density is non-central chi-squared:
    r(t+Δt) = c × χ²_d(λ)

    where:
    c = σ²(1 - e^(-κΔt))/(4κ)
    d = 4κθ/σ²
    λ = 4κe^(-κΔt)r(t)/(σ²(1 - e^(-κΔt)))

    Uses exact algorithm from Glasserman (2003).
    """
    rng = random.Random(seed) if seed is not None else random

    dt = T / n_steps
    exp_kdt = math.exp(-kappa * dt)
    c = sigma ** 2 * (1 - exp_kdt) / (4 * kappa)
    d = 4 * kappa * theta / sigma ** 2
    times = [i * dt for i in range(n_steps + 1)]

    paths = []
    for _ in range(n_paths):
        r = [r0]
        for _ in range(n_steps):
            # Non-centrality parameter
            lam = 4 * kappa * exp_kdt * r[-1] / (sigma ** 2 * (1 - exp_kdt))

            # Sample non-central chi-squared
            r_new = c * _ncx2_sample(d, lam, rng)
            r.append(max(r_new, 0))
        paths.append(r)

    return ExactSimulationResult(times, paths, n_paths, n_steps)


def simulate_heston_exact(
    S0: float,
    v0: float,
    r: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    seed: Optional[int] = None
) -> Tuple[List[float], List[List[float]], List[List[float]]]:
    """
    Quasi-exact simulation of Heston stochastic volatility.

    dS = rS dt + √v S dW_S
    dv = κ(θ - v)dt + σ_v √v dW_v
    Corr(dW_S, dW_v) = ρ

    Uses exact CIR for variance and the Broadie-Kaya conditional step for the
    log-price, with the integrated variance approximated by the trapezoid rule.

    Args:
        S0: Initial price
        v0: Initial variance
        r: Risk-free rate
        kappa: Variance mean reversion
        theta: Long-term variance
        sigma_v: Vol of vol
        rho: Correlation
        T: Terminal time
        n_steps: Number of steps
        n_paths: Number of paths
        seed: Random seed

    Returns:
        (times, S_paths, v_paths)
    """
    rng = random.Random(seed) if seed is not None else random

    dt = T / n_steps
    times = [i * dt for i in range(n_steps + 1)]

    # CIR parameters
    exp_kdt = math.exp(-kappa * dt)
    c = sigma_v ** 2 * (1 - exp_kdt) / (4 * kappa)
    d = 4 * kappa * theta / sigma_v ** 2

    S_paths = []
    v_paths = []

    for _ in range(n_paths):
        S = [S0]
        v = [v0]
        log_S = math.log(S0)

        for _ in range(n_steps):
            v_curr = v[-1]

            # Sample variance using exact CIR
            lam = 4 * kappa * exp_kdt * v_curr / (sigma_v ** 2 * (1 - exp_kdt))
            v_new = c * _ncx2_sample(d, lam, rng)
            v_new = max(v_new, 0)
            v.append(v_new)

            # Integrated variance approximation (trapezoidal)
            v_int = 0.5 * (v_curr + v_new) * dt

            # Log-price conditional on the variance path (Broadie-Kaya): the part of
            # dW_S correlated with dW_v is recovered from the variance increment,
            # the rest is independent Gaussian noise.
            Z = rng.gauss(0, 1)
            log_S += (
                r * dt - 0.5 * v_int
                + (rho / sigma_v) * (v_new - v_curr - kappa * theta * dt + kappa * v_int)
                + math.sqrt(max((1 - rho ** 2) * v_int, 0.0)) * Z
            )

            S.append(math.exp(log_S))

        S_paths.append(S)
        v_paths.append(v)

    return times, S_paths, v_paths


def _ncx2_sample(df: float, nc: float, rng=None) -> float:
    """
    Sample from non-central chi-squared distribution.

    χ²_d(λ) where d is degrees of freedom and λ is non-centrality.

    Uses multiple methods depending on parameters.
    """
    if rng is None:
        rng = random

    if nc <= 0:
        # Central chi-squared
        return _chi2_sample(df, rng)

    if df > 1:
        # Decomposition: χ²_d(λ) = χ²_{d-1} + (Z + √λ)²
        # where Z ~ N(0,1) independent of χ²_{d-1}
        Z = rng.gauss(0, 1)
        return _chi2_sample(df - 1, rng) + (Z + math.sqrt(nc)) ** 2
    else:
        # For d ≤ 1, use Poisson mixture
        # χ²_d(λ) = χ²_{d+2N} where N ~ Poisson(λ/2)
        N = poisson_sample(nc / 2, rng)
        return _chi2_sample(df + 2 * N, rng)


def _chi2_sample(df: float, rng=None) -> float:
    """Sample from central chi-squared with df degrees of freedom."""
    if df <= 0:
        return 0.0

    # Chi-squared = 2 * Gamma(df/2, 1)
    return 2 * _gamma_sample(df / 2, 1.0, rng)


def _gamma_sample(shape: float, scale: float, rng=None) -> float:
    """Sample from Gamma(shape, scale) using Marsaglia-Tsang."""
    if rng is None:
        rng = random

    if shape < 1:
        return _gamma_sample(shape + 1, scale, rng) * rng.random() ** (1 / shape)

    d = shape - 1 / 3
    c = 1 / math.sqrt(9 * d)

    while True:
        x = rng.gauss(0, 1)
        v = (1 + c * x) ** 3

        if v > 0:
            u = rng.random()
            if u < 1 - 0.0331 * x ** 4:
                return d * v * scale
            if math.log(u) < 0.5 * x ** 2 + d * (1 - v + math.log(v)):
                return d * v * scale


