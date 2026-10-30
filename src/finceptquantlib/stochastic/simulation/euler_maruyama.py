"""
Euler-Maruyama Scheme

First-order discretization for SDEs:
dX = μ(X,t)dt + σ(X,t)dW

Discretization:
X_{n+1} = X_n + μ(X_n, t_n)Δt + σ(X_n, t_n)ΔW_n

Strong convergence: O(√Δt)
Weak convergence: O(Δt)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from finceptquantlib.stochastic.sampling import poisson_sample


@dataclass
class EulerMaruyamaResult:
    """Result of Euler-Maruyama simulation."""
    times: List[float]
    paths: List[List[float]]  # paths[path_idx][time_idx]
    n_paths: int
    n_steps: int
    dt: float


def euler_maruyama(
    drift: Callable[[float, float], float],
    diffusion: Callable[[float, float], float],
    X0: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    seed: Optional[int] = None
) -> EulerMaruyamaResult:
    """
    Euler-Maruyama discretization for 1D SDE.

    Solves: dX = μ(X,t)dt + σ(X,t)dW

    Args:
        drift: μ(X, t) drift function
        diffusion: σ(X, t) diffusion function
        X0: Initial value
        T: Terminal time
        n_steps: Number of time steps
        n_paths: Number of paths to simulate
        seed: Random seed

    Returns:
        EulerMaruyamaResult with times and paths

    Example:
        >>> # GBM: dS = μS dt + σS dW
        >>> drift = lambda x, t: 0.05 * x
        >>> diffusion = lambda x, t: 0.2 * x
        >>> result = euler_maruyama(drift, diffusion, 100, 1.0, 252)
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
            X_new = x + drift(x, t) * dt + diffusion(x, t) * dW
            X.append(X_new)
        paths.append(X)

    return EulerMaruyamaResult(
        times=times,
        paths=paths,
        n_paths=n_paths,
        n_steps=n_steps,
        dt=dt
    )


def euler_maruyama_nd(
    drift: Callable[[List[float], float], List[float]],
    diffusion: Callable[[List[float], float], List[List[float]]],
    X0: List[float],
    T: float,
    n_steps: int,
    n_paths: int = 1,
    seed: Optional[int] = None
) -> Tuple[List[float], List[List[List[float]]]]:
    """
    Euler-Maruyama for multi-dimensional SDEs.

    Solves: dX = μ(X,t)dt + σ(X,t)dW

    where X ∈ ℝ^d, μ: ℝ^d → ℝ^d, σ: ℝ^d → ℝ^(d×m)

    Args:
        drift: μ(X, t) → d-vector
        diffusion: σ(X, t) → d×m matrix
        X0: Initial values (d-vector)
        T: Terminal time
        n_steps: Number of time steps
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
        # Initialize: X[dim][time]
        X = [[x0] for x0 in X0]

        for i in range(n_steps):
            t = times[i]
            x_curr = [X[j][-1] for j in range(d)]

            mu = drift(x_curr, t)
            sigma = diffusion(x_curr, t)
            m = len(sigma[0]) if sigma else d

            # Generate Brownian increments
            dW = [rng.gauss(0, sqrt_dt) for _ in range(m)]

            # Update each dimension
            for j in range(d):
                diffusion_term = sum(sigma[j][k] * dW[k] for k in range(m))
                X_new = x_curr[j] + mu[j] * dt + diffusion_term
                X[j].append(X_new)

        all_paths.append(X)

    return times, all_paths


def euler_maruyama_with_jumps(
    drift: Callable[[float, float], float],
    diffusion: Callable[[float, float], float],
    jump_intensity: float,
    jump_size_sampler: Callable[[], float],
    X0: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    seed: Optional[int] = None
) -> EulerMaruyamaResult:
    """
    Euler-Maruyama for jump-diffusion SDEs.

    Solves: dX = μ(X,t)dt + σ(X,t)dW + J dN

    where N is Poisson process with intensity λ.

    Args:
        drift: μ(X, t) drift function
        diffusion: σ(X, t) diffusion function
        jump_intensity: λ, Poisson intensity
        jump_size_sampler: Function to sample jump sizes
        X0: Initial value
        T: Terminal time
        n_steps: Number of time steps
        n_paths: Number of paths
        seed: Random seed

    Returns:
        EulerMaruyamaResult with times and paths
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

            # Diffusion part
            dW = rng.gauss(0, sqrt_dt)
            X_new = x + drift(x, t) * dt + diffusion(x, t) * dW

            # Jump part
            n_jumps = poisson_sample(jump_intensity * dt)
            for _ in range(n_jumps):
                X_new += jump_size_sampler()

            X.append(X_new)
        paths.append(X)

    return EulerMaruyamaResult(
        times=times,
        paths=paths,
        n_paths=n_paths,
        n_steps=n_steps,
        dt=dt
    )


