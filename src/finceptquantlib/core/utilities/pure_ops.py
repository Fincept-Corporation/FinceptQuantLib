"""
Pure Python numerical operations.

Provides pure Python functions for performance-critical computations.
"""

from __future__ import annotations

import math
from typing import Sequence

# Legacy constant for backward compatibility
NUMBA_AVAILABLE = False


# =============================================================================
# Basic Mathematical Operations
# =============================================================================

def _exp_array(x: list[float]) -> list[float]:
    """Vectorized exponential."""
    return [math.exp(v) for v in x]


def _log_array(x: list[float]) -> list[float]:
    """Vectorized natural log."""
    return [math.log(v) for v in x]


def _sqrt_array(x: list[float]) -> list[float]:
    """Vectorized square root."""
    return [math.sqrt(v) for v in x]


def _power_array(x: list[float], n: float) -> list[float]:
    """Vectorized power."""
    return [v ** n for v in x]


# =============================================================================
# Statistical Functions
# =============================================================================

def _mean(x: Sequence[float]) -> float:
    """Compute mean."""
    n = len(x)
    if n == 0:
        return 0.0
    return sum(x) / n


def _std(x: Sequence[float], ddof: int = 0) -> float:
    """Compute standard deviation."""
    n = len(x)
    if n <= ddof:
        return 0.0
    mean = _mean(x)
    variance = sum((v - mean) ** 2 for v in x) / (n - ddof)
    return math.sqrt(variance)


def _cov_matrix(returns: list[list[float]]) -> list[list[float]]:
    """
    Compute covariance matrix.

    Args:
        returns: List of observations, each observation is a list of asset returns
                 Shape: (n_observations, n_assets)

    Returns:
        Covariance matrix (n_assets, n_assets)
    """
    if not returns:
        return []

    n_obs = len(returns)
    n_assets = len(returns[0])

    # Compute means
    means = [0.0] * n_assets
    for j in range(n_assets):
        for i in range(n_obs):
            means[j] += returns[i][j]
        means[j] /= n_obs

    # Compute covariance
    cov = [[0.0] * n_assets for _ in range(n_assets)]
    for i in range(n_assets):
        for j in range(i, n_assets):
            total = 0.0
            for k in range(n_obs):
                total += (returns[k][i] - means[i]) * (returns[k][j] - means[j])
            cov[i][j] = total / (n_obs - 1) if n_obs > 1 else 0.0
            cov[j][i] = cov[i][j]

    return cov


def _percentile(x: Sequence[float], p: float) -> float:
    """
    Compute percentile (0-100 scale).

    Uses linear interpolation method.
    """
    sorted_x = sorted(x)
    n = len(sorted_x)

    if n == 0:
        return 0.0
    if n == 1:
        return sorted_x[0]

    # Index calculation
    idx = (p / 100.0) * (n - 1)
    lower = int(math.floor(idx))
    upper = int(math.ceil(idx))

    if lower == upper:
        return sorted_x[lower]

    # Linear interpolation
    weight = idx - lower
    return sorted_x[lower] * (1 - weight) + sorted_x[upper] * weight


def _quantile(x: Sequence[float], q: float) -> float:
    """Compute quantile (0-1 scale)."""
    return _percentile(x, q * 100.0)


# =============================================================================
# Interpolation Functions
# =============================================================================

def _linear_interp(x: float, x_data: Sequence[float], y_data: Sequence[float]) -> float:
    """
    Linear interpolation.

    Args:
        x: Point to interpolate
        x_data: Known x values (sorted)
        y_data: Known y values

    Returns:
        Interpolated y value
    """
    n = len(x_data)

    if n == 0:
        return 0.0

    # Handle extrapolation
    if x <= x_data[0]:
        return y_data[0]
    if x >= x_data[n - 1]:
        return y_data[n - 1]

    # Binary search for interval
    low, high = 0, n - 1
    while low < high - 1:
        mid = (low + high) // 2
        if x_data[mid] <= x:
            low = mid
        else:
            high = mid

    # Linear interpolation
    t = (x - x_data[low]) / (x_data[high] - x_data[low])
    return y_data[low] * (1 - t) + y_data[high] * t


def _log_linear_interp(x: float, x_data: Sequence[float], y_data: Sequence[float]) -> float:
    """
    Log-linear interpolation (for discount factors).

    Interpolates in log space: log(y) is linearly interpolated.
    """
    n = len(x_data)

    if n == 0:
        return 1.0

    if x <= x_data[0]:
        return y_data[0]
    if x >= x_data[n - 1]:
        # Log-linear extrapolation
        if n < 2:
            return y_data[n - 1]
        slope = math.log(y_data[n - 1] / y_data[n - 2]) / (x_data[n - 1] - x_data[n - 2])
        return y_data[n - 1] * math.exp(slope * (x - x_data[n - 1]))

    # Binary search
    low, high = 0, n - 1
    while low < high - 1:
        mid = (low + high) // 2
        if x_data[mid] <= x:
            low = mid
        else:
            high = mid

    # Log-linear interpolation
    t = (x - x_data[low]) / (x_data[high] - x_data[low])
    log_y = math.log(y_data[low]) * (1 - t) + math.log(y_data[high]) * t
    return math.exp(log_y)


def _cubic_spline_interp(
    x: float,
    x_data: Sequence[float],
    y_data: Sequence[float],
    coeffs: Sequence[float],
) -> float:
    """
    Cubic spline interpolation.

    Args:
        x: Point to interpolate
        x_data: Knot x values
        y_data: Knot y values
        coeffs: Second derivatives at knots (from spline fitting)

    Returns:
        Interpolated value
    """
    n = len(x_data)

    if n == 0:
        return 0.0

    # Handle extrapolation
    if x <= x_data[0]:
        return y_data[0]
    if x >= x_data[n - 1]:
        return y_data[n - 1]

    # Find interval
    low, high = 0, n - 1
    while low < high - 1:
        mid = (low + high) // 2
        if x_data[mid] <= x:
            low = mid
        else:
            high = mid

    # Cubic spline formula
    h = x_data[high] - x_data[low]
    a = (x_data[high] - x) / h
    b = (x - x_data[low]) / h

    return (
        a * y_data[low] + b * y_data[high] +
        ((a**3 - a) * coeffs[low] + (b**3 - b) * coeffs[high]) * h**2 / 6.0
    )


# =============================================================================
# Discount Factor and Rate Calculations
# =============================================================================

def _df_to_zero_rate(df: float, t: float) -> float:
    """Convert discount factor to continuously compounded zero rate."""
    if t <= 0:
        return 0.0
    return -math.log(df) / t


def _zero_rate_to_df(rate: float, t: float) -> float:
    """Convert zero rate to discount factor."""
    return math.exp(-rate * t)


def _forward_rate(df1: float, df2: float, t1: float, t2: float) -> float:
    """Calculate forward rate between two times."""
    if t2 <= t1:
        return 0.0
    return (df1 / df2 - 1) / (t2 - t1)


def _discount_cashflows(
    cashflows: Sequence[float],
    times: Sequence[float],
    discount_factors: Sequence[float],
) -> float:
    """
    Discount a series of cashflows.

    Args:
        cashflows: Cash flow amounts
        times: Times of cash flows
        discount_factors: Discount factors at each time

    Returns:
        Present value
    """
    pv = 0.0
    for cf, df in zip(cashflows, discount_factors):
        pv += cf * df
    return pv


# =============================================================================
# Option Pricing Helpers
# =============================================================================

def _norm_cdf(x: float) -> float:
    """Standard normal CDF using error function (machine precision)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _black_scholes_call(
    spot: float,
    strike: float,
    rate: float,
    time: float,
    volatility: float,
) -> float:
    """Black-Scholes call option price."""
    if time <= 0:
        return max(spot - strike, 0.0)

    sqrt_t = math.sqrt(time)
    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * time) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    return spot * _norm_cdf(d1) - strike * math.exp(-rate * time) * _norm_cdf(d2)


def _black_scholes_put(
    spot: float,
    strike: float,
    rate: float,
    time: float,
    volatility: float,
) -> float:
    """Black-Scholes put option price."""
    if time <= 0:
        return max(strike - spot, 0.0)

    sqrt_t = math.sqrt(time)
    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * time) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    return strike * math.exp(-rate * time) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def _black76_call(
    forward: float,
    strike: float,
    volatility: float,
    time: float,
    discount_factor: float,
) -> float:
    """Black-76 call option price."""
    if time <= 0:
        return discount_factor * max(forward - strike, 0.0)

    sqrt_t = math.sqrt(time)
    d1 = (math.log(forward / strike) + 0.5 * volatility**2 * time) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    return discount_factor * (forward * _norm_cdf(d1) - strike * _norm_cdf(d2))


def _black76_put(
    forward: float,
    strike: float,
    volatility: float,
    time: float,
    discount_factor: float,
) -> float:
    """Black-76 put option price."""
    if time <= 0:
        return discount_factor * max(strike - forward, 0.0)

    sqrt_t = math.sqrt(time)
    d1 = (math.log(forward / strike) + 0.5 * volatility**2 * time) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    return discount_factor * (strike * _norm_cdf(-d2) - forward * _norm_cdf(-d1))


# =============================================================================
# Monte Carlo Helpers
# =============================================================================

def _gbm_paths(
    spot: float,
    drift: float,
    volatility: float,
    time: float,
    n_paths: int,
    n_steps: int,
    seed: int = 42,
) -> list[list[float]]:
    """
    Generate Geometric Brownian Motion paths.

    Args:
        spot: Initial spot price
        drift: Drift rate (mu)
        volatility: Volatility (sigma)
        time: Total time horizon
        n_paths: Number of simulation paths
        n_steps: Number of time steps
        seed: Random seed

    Returns:
        List of paths, each path is a list of prices
    """
    import random
    rng = random.Random(seed)

    dt = time / n_steps
    sqrt_dt = math.sqrt(dt)

    paths = []
    for _ in range(n_paths):
        path = [spot]
        current = spot
        for _ in range(n_steps):
            z = rng.gauss(0, 1)
            current = current * math.exp(
                (drift - 0.5 * volatility**2) * dt + volatility * sqrt_dt * z
            )
            path.append(current)
        paths.append(path)

    return paths


def _antithetic_paths(
    spot: float,
    drift: float,
    volatility: float,
    time: float,
    n_paths: int,
    n_steps: int,
    seed: int = 42,
) -> list[list[float]]:
    """
    Generate GBM paths with antithetic variates for variance reduction.

    Returns 2*n_paths paths (original + antithetic).
    """
    import random
    rng = random.Random(seed)

    dt = time / n_steps
    sqrt_dt = math.sqrt(dt)

    paths = []

    for _ in range(n_paths):
        path1 = [spot]
        path2 = [spot]
        current1 = spot
        current2 = spot

        for _ in range(n_steps):
            z = rng.gauss(0, 1)

            # Original path
            current1 = current1 * math.exp(
                (drift - 0.5 * volatility**2) * dt + volatility * sqrt_dt * z
            )
            path1.append(current1)

            # Antithetic path (uses -z)
            current2 = current2 * math.exp(
                (drift - 0.5 * volatility**2) * dt + volatility * sqrt_dt * (-z)
            )
            path2.append(current2)

        paths.append(path1)
        paths.append(path2)

    return paths


# =============================================================================
# VaR Calculations
# =============================================================================

def _historical_var(returns: Sequence[float], confidence: float) -> float:
    """
    Calculate historical VaR.

    Args:
        returns: Historical returns
        confidence: Confidence level (e.g., 0.99)

    Returns:
        VaR (as positive number)
    """
    alpha = 1.0 - confidence
    return -_quantile(returns, alpha)


def _expected_shortfall(returns: Sequence[float], confidence: float) -> float:
    """
    Calculate Expected Shortfall (CVaR).

    ES = average of returns beyond VaR threshold.
    """
    var = _historical_var(returns, confidence)

    # Find returns worse than VaR
    tail_returns = [r for r in returns if r < -var]

    if not tail_returns:
        return var

    return -sum(tail_returns) / len(tail_returns)


def _parametric_var(
    portfolio_value: float,
    portfolio_std: float,
    confidence: float,
    holding_period: int,
) -> float:
    """
    Calculate parametric (delta-normal) VaR.

    Args:
        portfolio_value: Current portfolio value
        portfolio_std: Portfolio return standard deviation (daily)
        confidence: Confidence level
        holding_period: Holding period in days

    Returns:
        VaR (as positive number)
    """
    from finceptquantlib.core.utilities.distributions import norm_ppf
    z = norm_ppf(confidence)
    return portfolio_value * portfolio_std * z * math.sqrt(holding_period)


# =============================================================================
# Matrix Operations
# =============================================================================

def _cholesky(A: list[list[float]]) -> list[list[float]]:
    """
    Cholesky decomposition A = L @ L.T

    Args:
        A: Symmetric positive definite matrix

    Returns:
        Lower triangular matrix L
    """
    n = len(A)
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))

            if i == j:
                L[i][j] = math.sqrt(A[i][i] - s)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]

    return L


def _matrix_multiply(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Matrix multiplication."""
    m = len(A)
    k = len(A[0]) if A else 0
    n = len(B[0]) if B else 0

    C = [[0.0] * n for _ in range(m)]

    for i in range(m):
        for j in range(n):
            C[i][j] = sum(A[i][l] * B[l][j] for l in range(k))

    return C
