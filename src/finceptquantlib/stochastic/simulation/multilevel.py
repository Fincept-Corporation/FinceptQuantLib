"""
Multilevel Monte Carlo (MLMC)

Variance reduction technique using hierarchical discretization:
- Uses multiple levels of resolution
- Exploits strong convergence for variance reduction
- Achieves O(ε⁻²) complexity vs O(ε⁻³) for standard MC

Reference: Giles (2008, 2015)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


@dataclass
class MLMCResult:
    """Result of MLMC estimation."""
    estimate: float
    variance: float
    std_error: float
    level_estimates: List[float]
    level_variances: List[float]
    level_samples: List[int]
    total_cost: float


def mlmc_estimate(
    simulator: Callable[[float, int, int], Tuple[List[float], List[List[float]]]],
    payoff: Callable[[List[float]], float],
    T: float,
    n_levels: int,
    base_steps: int = 4,
    n_pilot: int = 1000,
    target_variance: float = 1e-4,
    seed: Optional[int] = None
) -> MLMCResult:
    """
    Multilevel Monte Carlo estimator.

    Uses a hierarchy of discretizations with step sizes:
    h_l = T / (base_steps * 2^l)

    Computes E[P(X_T)] where P is a payoff function.

    MLMC identity:
    E[P_L] = E[P_0] + Σ_{l=1}^L E[P_l - P_{l-1}]

    Args:
        simulator: Function(T, n_steps, n_paths) → (times, paths)
        payoff: Function(path) → payoff value
        T: Terminal time
        n_levels: Number of levels L
        base_steps: Steps at level 0
        n_pilot: Pilot samples per level for variance estimation
        target_variance: Target variance for optimal allocation
        seed: Random seed

    Returns:
        MLMCResult with estimates and diagnostics
    """
    # The simulator is user-supplied; seeding the global generator makes
    # simulators that draw from `random` reproducible.
    if seed is not None:
        random.seed(seed)

    level_means = []
    level_vars = []

    # Step 1: Pilot run to estimate variances at each level
    for level in range(n_levels):
        n_steps_fine = base_steps * (2 ** level)
        n_steps_coarse = n_steps_fine // 2 if level > 0 else 0

        if level == 0:
            # Level 0: Just E[P_0]
            _, paths = simulator(T, n_steps_fine, n_pilot)
            payoffs = [payoff(path) for path in paths]
            level_means.append(sum(payoffs) / n_pilot)
            level_vars.append(_variance(payoffs))
        else:
            # Level l > 0: E[P_l - P_{l-1}]
            # NOTE: For proper MLMC variance reduction, fine and coarse
            # paths must share the same Brownian motion (coupled paths).
            # Use mlmc_coupled_paths() for drift/diffusion SDEs.
            # This generic simulator interface cannot enforce coupling,
            # so variance reduction at higher levels may be suboptimal.
            diffs = []
            for _ in range(n_pilot):
                _, paths_fine = simulator(T, n_steps_fine, 1)
                _, paths_coarse = simulator(T, n_steps_coarse, 1)
                P_fine = payoff(paths_fine[0])
                P_coarse = payoff(paths_coarse[0])
                diffs.append(P_fine - P_coarse)

            level_means.append(sum(diffs) / n_pilot)
            level_vars.append(_variance(diffs))

    # Step 2: Optimal sample allocation
    # Minimize cost subject to variance constraint
    # N_l ∝ √(V_l / C_l) where C_l ∝ 2^l
    costs = [base_steps * (2 ** l) for l in range(n_levels)]

    # Total samples needed for target variance
    sum_sqrt_vc = sum(math.sqrt(level_vars[l] * costs[l])
                      for l in range(n_levels) if level_vars[l] > 0)

    n_samples = []
    for l in range(n_levels):
        if level_vars[l] > 0:
            N_l = int(math.ceil(
                (1 / target_variance) * sum_sqrt_vc *
                math.sqrt(level_vars[l] / costs[l])
            ))
            N_l = max(N_l, n_pilot)  # At least pilot samples
        else:
            N_l = n_pilot
        n_samples.append(N_l)

    # Step 3: Generate additional samples
    final_means = []
    final_vars = []

    for level in range(n_levels):
        n_steps_fine = base_steps * (2 ** level)
        n_steps_coarse = n_steps_fine // 2 if level > 0 else 0
        n_extra = n_samples[level] - n_pilot

        if level == 0:
            if n_extra > 0:
                _, paths = simulator(T, n_steps_fine, n_extra)
                extra_payoffs = [payoff(path) for path in paths]
                # Combine pilot mean with extra samples
                combined_sum = level_means[0] * n_pilot + sum(extra_payoffs)
                final_means.append(combined_sum / n_samples[level])
            else:
                final_means.append(level_means[level])
            final_vars.append(level_vars[level] / n_samples[level])
        else:
            if n_extra > 0:
                extra_diffs = []
                for _ in range(n_extra):
                    _, paths_fine = simulator(T, n_steps_fine, 1)
                    _, paths_coarse = simulator(T, n_steps_coarse, 1)
                    P_fine = payoff(paths_fine[0])
                    P_coarse = payoff(paths_coarse[0])
                    extra_diffs.append(P_fine - P_coarse)
                # Combine with pilot estimate
                combined_mean = (level_means[level] * n_pilot +
                               sum(extra_diffs)) / n_samples[level]
                final_means.append(combined_mean)
            else:
                final_means.append(level_means[level])
            final_vars.append(level_vars[level] / n_samples[level])

    # Step 4: Combine estimates
    estimate = sum(final_means)
    total_variance = sum(final_vars)
    total_cost = sum(n_samples[l] * costs[l] for l in range(n_levels))

    return MLMCResult(
        estimate=estimate,
        variance=total_variance,
        std_error=math.sqrt(total_variance),
        level_estimates=final_means,
        level_variances=level_vars,
        level_samples=n_samples,
        total_cost=total_cost
    )


def mlmc_coupled_paths(
    drift: Callable[[float, float], float],
    diffusion: Callable[[float, float], float],
    X0: float,
    T: float,
    n_steps_fine: int,
    seed: Optional[int] = None
) -> Tuple[List[float], List[float], List[float]]:
    """
    Generate coupled fine and coarse paths for MLMC.

    Uses same Brownian increments for both paths.
    Coarse path uses n_steps_fine/2 steps.

    Args:
        drift: μ(X, t)
        diffusion: σ(X, t)
        X0: Initial value
        T: Terminal time
        n_steps_fine: Number of fine steps (must be even)
        seed: Random seed

    Returns:
        (times_fine, path_fine, path_coarse)
    """
    rng = random.Random(seed) if seed is not None else random

    if n_steps_fine % 2 != 0:
        raise ValueError("n_steps_fine must be even")

    n_steps_coarse = n_steps_fine // 2
    dt_fine = T / n_steps_fine
    dt_coarse = T / n_steps_coarse
    sqrt_dt_fine = math.sqrt(dt_fine)

    times_fine = [i * dt_fine for i in range(n_steps_fine + 1)]

    X_fine = [X0]
    X_coarse = [X0]

    for i in range(n_steps_coarse):
        t_coarse = i * dt_coarse
        x_coarse = X_coarse[-1]

        # Two fine steps
        t_fine_1 = 2 * i * dt_fine
        t_fine_2 = (2 * i + 1) * dt_fine
        x_fine = X_fine[-1]

        # First fine step
        dW1 = rng.gauss(0, sqrt_dt_fine)
        x_fine_new = (x_fine +
                      drift(x_fine, t_fine_1) * dt_fine +
                      diffusion(x_fine, t_fine_1) * dW1)
        X_fine.append(x_fine_new)

        # Second fine step
        dW2 = rng.gauss(0, sqrt_dt_fine)
        x_fine = x_fine_new
        x_fine_new = (x_fine +
                      drift(x_fine, t_fine_2) * dt_fine +
                      diffusion(x_fine, t_fine_2) * dW2)
        X_fine.append(x_fine_new)

        # Coarse step uses combined Brownian increment
        dW_coarse = dW1 + dW2
        x_coarse_new = (x_coarse +
                       drift(x_coarse, t_coarse) * dt_coarse +
                       diffusion(x_coarse, t_coarse) * dW_coarse)
        X_coarse.append(x_coarse_new)

    return times_fine, X_fine, X_coarse


def _variance(values: List[float]) -> float:
    """Calculate sample variance."""
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    return sum((x - mean) ** 2 for x in values) / (n - 1)
