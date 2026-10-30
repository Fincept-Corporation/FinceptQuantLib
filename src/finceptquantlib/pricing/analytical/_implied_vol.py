"""
Safeguarded implied volatility root finder shared by the analytical models.

Option prices are monotonically increasing in volatility, so a bracket
[lo, hi] always contains the root. Newton steps are taken when they stay
inside the bracket; otherwise the solver bisects. This converges for deep
in/out-of-the-money options where vega is tiny and plain Newton diverges.
"""

from __future__ import annotations

from typing import Callable


def solve_implied_vol(
    price_fn: Callable[[float], float],
    vega_fn: Callable[[float], float],
    target: float,
    guess: float,
    lo: float,
    hi: float,
    tol: float,
    max_iter: int,
    model: str,
    hi_limit: float = 100.0,
) -> float:
    """
    Solve price_fn(sigma) == target for sigma.

    Args:
        price_fn: Option price as a function of volatility (increasing)
        vega_fn: Derivative of price_fn with respect to volatility
        target: Observed option price
        guess: Initial volatility guess
        lo: Lower volatility bound
        hi: Initial upper volatility bound (expanded up to hi_limit)
        tol: Absolute price tolerance
        max_iter: Maximum iterations
        model: Model name used in error messages
        hi_limit: Largest volatility considered

    Returns:
        Implied volatility

    Raises:
        ValueError: If the price is outside the model's attainable range
            or the solver does not converge
    """
    f_lo = price_fn(lo) - target
    if f_lo > tol:
        raise ValueError(f"{model} implied vol: price is below the zero-volatility value")
    if f_lo >= -tol:
        return lo

    f_hi = price_fn(hi) - target
    while f_hi < 0 and hi < hi_limit:
        lo = hi
        hi = min(hi * 2, hi_limit)
        f_hi = price_fn(hi) - target
    if f_hi < -tol:
        raise ValueError(f"{model} implied vol: price exceeds the maximum attainable value")

    sigma = min(max(guess, lo), hi)
    for _ in range(max_iter):
        diff = price_fn(sigma) - target
        if abs(diff) < tol:
            return sigma
        if diff > 0:
            hi = sigma
        else:
            lo = sigma

        vega = vega_fn(sigma)
        candidate = sigma - diff / vega if vega > 0 else lo - 1.0
        sigma = candidate if lo < candidate < hi else 0.5 * (lo + hi)

        if hi - lo < 1e-14:
            return sigma

    raise ValueError(f"{model} implied vol did not converge")
