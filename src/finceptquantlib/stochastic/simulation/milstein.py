"""
Milstein Scheme

Higher-order discretization for SDEs including diffusion derivative:
dX = μ(X,t)dt + σ(X,t)dW

Milstein scheme:
X_{n+1} = X_n + μΔt + σΔW + ½σσ'((ΔW)² - Δt)

where σ' = ∂σ/∂x

Strong convergence: O(Δt)
Weak convergence: O(Δt)

Better accuracy than Euler-Maruyama when σ depends on X.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


@dataclass
class MilsteinResult:
    """Result of Milstein simulation."""
    times: List[float]
    paths: List[List[float]]
    n_paths: int
    n_steps: int
    dt: float


def milstein(
    drift: Callable[[float, float], float],
    diffusion: Callable[[float, float], float],
    diffusion_deriv: Callable[[float, float], float],
    X0: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    seed: Optional[int] = None
) -> MilsteinResult:
    """
    Milstein discretization for 1D SDE.

    Solves: dX = μ(X,t)dt + σ(X,t)dW

    Args:
        drift: μ(X, t) drift function
        diffusion: σ(X, t) diffusion function
        diffusion_deriv: ∂σ/∂x (X, t) derivative of diffusion
        X0: Initial value
        T: Terminal time
        n_steps: Number of time steps
        n_paths: Number of paths
        seed: Random seed

    Returns:
        MilsteinResult with times and paths

    Example:
        >>> # CIR: dX = κ(θ-X)dt + σ√X dW
        >>> drift = lambda x, t: 0.5 * (0.04 - x)
        >>> diffusion = lambda x, t: 0.1 * math.sqrt(max(x, 0))
        >>> diff_deriv = lambda x, t: 0.05 / math.sqrt(max(x, 1e-10))
        >>> result = milstein(drift, diffusion, diff_deriv, 0.04, 1.0, 252)
    """
    rng = random.Random(seed) if seed is not None else random

    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    times = [i * dt for i in range(n_steps + 1)]

    paths = []
    for _ in range(n_paths):
        X = [X0]
        for i in range(n_steps):
            t = times[i]
            x = X[-1]

            dW = rng.gauss(0, sqrt_dt)
            mu = drift(x, t)
            sigma = diffusion(x, t)
            sigma_prime = diffusion_deriv(x, t)

            # Milstein correction term
            correction = 0.5 * sigma * sigma_prime * (dW ** 2 - dt)

            X_new = x + mu * dt + sigma * dW + correction
            X.append(X_new)
        paths.append(X)

    return MilsteinResult(
        times=times,
        paths=paths,
        n_paths=n_paths,
        n_steps=n_steps,
        dt=dt
    )


def milstein_implicit(
    drift: Callable[[float, float], float],
    diffusion: Callable[[float, float], float],
    diffusion_deriv: Callable[[float, float], float],
    drift_deriv: Callable[[float, float], float],
    X0: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    theta: float = 0.5,
    seed: Optional[int] = None
) -> MilsteinResult:
    """
    Implicit Milstein scheme for better stability.

    Uses theta-method for drift:
    X_{n+1} = X_n + (1-θ)μ(X_n)Δt + θμ(X_{n+1})Δt + σ(X_n)ΔW + correction

    Solved via Newton iteration.

    Args:
        drift: μ(X, t)
        diffusion: σ(X, t)
        diffusion_deriv: ∂σ/∂x
        drift_deriv: ∂μ/∂x (for Newton)
        X0: Initial value
        T: Terminal time
        n_steps: Number of steps
        n_paths: Number of paths
        theta: Implicitness (0=explicit, 0.5=Crank-Nicolson, 1=fully implicit)
        seed: Random seed
    """
    rng = random.Random(seed) if seed is not None else random

    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    times = [i * dt for i in range(n_steps + 1)]

    paths = []
    for _ in range(n_paths):
        X = [X0]
        for i in range(n_steps):
            t = times[i]
            t_next = times[i + 1]
            x = X[-1]

            dW = rng.gauss(0, sqrt_dt)
            sigma = diffusion(x, t)
            sigma_prime = diffusion_deriv(x, t)

            # Explicit part
            explicit = (x + (1 - theta) * drift(x, t) * dt +
                       sigma * dW +
                       0.5 * sigma * sigma_prime * (dW ** 2 - dt))

            # Newton iteration for implicit part
            x_new = explicit  # Initial guess
            for _ in range(5):  # Usually converges quickly
                F = x_new - explicit - theta * drift(x_new, t_next) * dt
                F_prime = 1 - theta * drift_deriv(x_new, t_next) * dt
                x_new = x_new - F / F_prime

            X.append(x_new)
        paths.append(X)

    return MilsteinResult(
        times=times,
        paths=paths,
        n_paths=n_paths,
        n_steps=n_steps,
        dt=dt
    )


def milstein_nd(
    drift: Callable[[List[float], float], List[float]],
    diffusion: Callable[[List[float], float], List[List[float]]],
    diffusion_deriv: Callable[[List[float], float], List[List[List[float]]]],
    X0: List[float],
    T: float,
    n_steps: int,
    n_paths: int = 1,
    seed: Optional[int] = None
) -> Tuple[List[float], List[List[List[float]]]]:
    """
    Milstein scheme for multi-dimensional SDEs.

    Requires cross-derivatives and Lévy area simulation for full
    strong order 1.0 convergence.

    This simplified version uses commutative noise approximation.

    Args:
        drift: μ(X, t) → d-vector
        diffusion: σ(X, t) → d×m matrix
        diffusion_deriv: ∂σ_ij/∂x_k → d×m×d tensor
        X0: Initial values
        T: Terminal time
        n_steps: Number of steps
        n_paths: Number of paths
        seed: Random seed

    Returns:
        (times, paths): paths[path][dim][time]
    """
    rng = random.Random(seed) if seed is not None else random

    d = len(X0)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    times = [i * dt for i in range(n_steps + 1)]

    all_paths = []
    for _ in range(n_paths):
        X = [[x0] for x0 in X0]

        for i in range(n_steps):
            t = times[i]
            x_curr = [X[j][-1] for j in range(d)]

            mu = drift(x_curr, t)
            sigma = diffusion(x_curr, t)
            m = len(sigma[0]) if sigma else d

            # Diffusion derivatives: dsigma[i][j][k] = ∂σ_ij/∂x_k
            dsigma = diffusion_deriv(x_curr, t)

            # Generate Brownian increments
            dW = [rng.gauss(0, sqrt_dt) for _ in range(m)]

            for j in range(d):
                # Euler term
                diff_term = sum(sigma[j][k] * dW[k] for k in range(m))
                X_new = x_curr[j] + mu[j] * dt + diff_term

                # Milstein correction (commutative noise approximation)
                for k1 in range(m):
                    for l in range(d):
                        X_new += 0.5 * sigma[l][k1] * dsigma[j][k1][l] * (
                            dW[k1] ** 2 - dt
                        )

                X[j].append(X_new)

        all_paths.append(X)

    return times, all_paths
