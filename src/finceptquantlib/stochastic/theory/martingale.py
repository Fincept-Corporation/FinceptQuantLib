"""
Martingale Theory Tools

Theoretical tools for martingale analysis:
- Martingale tests
- Optional stopping
- Doob's inequalities
"""

from __future__ import annotations

import math
from typing import Callable, List, Tuple


def is_martingale_test(
    paths: List[List[float]],
    times: List[float],
    confidence: float = 0.95
) -> Tuple[bool, float, float]:
    """
    Test if a process is a martingale using regression.

    For a martingale: E[X_{t+s} | F_t] = X_t

    Tests whether mean increment E[ΔX] = 0 at each time step.

    Args:
        paths: List of sample paths
        times: Time grid
        confidence: Confidence level for test

    Returns:
        (is_martingale, mean_increment, t_statistic)
    """
    from scipy import stats  # type: ignore

    n_times = len(times) - 1

    # Collect all increments
    increments = []
    for path in paths:
        for i in range(n_times):
            increments.append(path[i + 1] - path[i])

    mean_inc = sum(increments) / len(increments)
    std_inc = math.sqrt(sum((x - mean_inc) ** 2 for x in increments) / (len(increments) - 1))
    se = std_inc / math.sqrt(len(increments))

    t_stat = mean_inc / se if se > 0 else float('inf')

    # Two-sided t-test
    critical = stats.t.ppf((1 + confidence) / 2, len(increments) - 1)
    is_martingale = abs(t_stat) < critical

    return is_martingale, mean_inc, t_stat


def optional_stopping_value(
    paths: List[List[float]],
    times: List[float],
    stopping_condition: Callable[[float, float], bool]
) -> Tuple[List[float], List[float]]:
    """
    Compute stopped process values.

    For a martingale M and stopping time τ:
    E[M_τ] = E[M_0] (under regularity conditions)

    Args:
        paths: Sample paths
        times: Time grid
        stopping_condition: Function(x, t) → bool (True means stop)

    Returns:
        (stopping_times, stopped_values)
    """
    stopping_times = []
    stopped_values = []

    for path in paths:
        stopped = False
        for i, (t, x) in enumerate(zip(times, path)):
            if stopping_condition(x, t):
                stopping_times.append(t)
                stopped_values.append(x)
                stopped = True
                break

        if not stopped:
            # Stopped at terminal time
            stopping_times.append(times[-1])
            stopped_values.append(path[-1])

    return stopping_times, stopped_values


def doob_maximal_inequality(
    paths: List[List[float]],
    p: float = 2
) -> Tuple[float, float]:
    """
    Verify Doob's maximal inequality.

    For a non-negative submartingale X:
    λ^p * P(max X_t ≥ λ) ≤ E[X_T^p]

    Returns empirical and theoretical bounds.

    Args:
        paths: Sample paths
        p: Power (default 2)

    Returns:
        (empirical_ratio, theoretical_bound)
    """
    maxima = [max(path) for path in paths]
    terminals = [path[-1] for path in paths]

    E_max_p = sum(m ** p for m in maxima) / len(maxima)
    E_terminal_p = sum(t ** p for t in terminals) / len(terminals)

    # For p > 1: E[max^p] ≤ (p/(p-1))^p * E[X_T^p]
    if p > 1:
        doob_const = (p / (p - 1)) ** p
        theoretical_bound = doob_const * E_terminal_p
    else:
        theoretical_bound = float('inf')

    ratio = E_max_p / E_terminal_p if E_terminal_p > 0 else float('inf')

    return ratio, theoretical_bound


def quadratic_variation_martingale(
    path: List[float],
    times: List[float]
) -> List[float]:
    """
    Compute the quadratic variation process.

    For a continuous martingale M, the process:
    M_t² - [M]_t

    is also a martingale.

    Args:
        path: Martingale path values
        times: Time grid

    Returns:
        Quadratic variation process [M]_t
    """
    qv = [0.0]
    for i in range(1, len(path)):
        increment = (path[i] - path[i - 1]) ** 2
        qv.append(qv[-1] + increment)
    return qv


def predictable_representation(
    payoff_paths: List[float],
    underlying_paths: List[List[float]],
    times: List[float]
) -> List[float]:
    """
    Estimate predictable representation (hedging strategy).

    For martingale representation theorem:
    H_T = E[H_T] + ∫_0^T θ_s dM_s

    Estimates θ using regression.

    Args:
        payoff_paths: Terminal payoff values H_T
        underlying_paths: Martingale paths M
        times: Time grid

    Returns:
        Estimated hedging ratios θ at each time
    """
    n_paths = len(payoff_paths)
    n_times = len(times)

    # Simple regression at each time step
    thetas = []

    for i in range(n_times - 1):
        # Regress payoff changes on underlying changes
        dH = []
        dM = []

        for j in range(n_paths):
            if i + 1 < len(underlying_paths[j]):
                dM_val = underlying_paths[j][i + 1] - underlying_paths[j][i]
                # Approximate payoff increment
                # This is simplified - actual implementation needs careful treatment
                dM.append(dM_val)
                dH.append(payoff_paths[j] / n_times)  # Simplified

        if len(dM) > 1:
            cov = sum(dm * dh for dm, dh in zip(dM, dH)) / len(dM)
            var = sum(dm ** 2 for dm in dM) / len(dM)
            theta = cov / var if var > 0 else 0.0
        else:
            theta = 0.0

        thetas.append(theta)

    return thetas


def local_martingale_test(
    paths: List[List[float]],
    times: List[float],
    localizing_seq: List[float]
) -> Tuple[bool, List[float]]:
    """
    Test if a process is a local martingale.

    A local martingale is a process M such that M^{τ_n} is a
    true martingale for some localizing sequence τ_n ↑ ∞.

    Tests martingale property for stopped processes.

    Args:
        paths: Sample paths
        times: Time grid
        localizing_seq: Stopping levels [τ_1, τ_2, ...]

    Returns:
        (is_local_martingale, stopped_means)
    """
    stopped_means = []

    for level in localizing_seq:
        # Stop paths when they exceed level
        stopped_values = []
        for path in paths:
            for i, x in enumerate(path):
                if abs(x) >= level:
                    stopped_values.append(x)
                    break
            else:
                stopped_values.append(path[-1])

        stopped_means.append(sum(stopped_values) / len(stopped_values))

    # Check if stopped means are approximately constant
    mean_of_means = sum(stopped_means) / len(stopped_means)
    variance = sum((m - mean_of_means) ** 2 for m in stopped_means) / len(stopped_means)

    # Heuristic: small variance indicates local martingale
    is_local_mart = variance < 0.01 * mean_of_means ** 2 if mean_of_means != 0 else variance < 1e-6

    return is_local_mart, stopped_means
