"""
Monotone Interpolation

Shape-preserving interpolation methods that maintain monotonicity:
- Monotone convex (Hagan-West)
- PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
"""

from __future__ import annotations

from typing import List
import bisect


def pchip(
    x: List[float],
    y: List[float],
    xi: float,
) -> float:
    """
    PCHIP (Piecewise Cubic Hermite Interpolating Polynomial).

    Preserves monotonicity and shape of the data.
    Uses Fritsch-Carlson algorithm for slope determination.

    Args:
        x: X coordinates (sorted ascending)
        y: Y values
        xi: Point to interpolate at

    Returns:
        Interpolated value at xi
    """
    n = len(x)
    if n < 2:
        return y[0] if y else 0.0

    # Compute slopes at each point
    d = _pchip_slopes(x, y)

    # Handle boundaries
    if xi <= x[0]:
        return y[0]
    if xi >= x[-1]:
        return y[-1]

    # Find interval
    i = bisect.bisect_right(x, xi) - 1
    i = min(i, n - 2)

    # Hermite interpolation on interval [x[i], x[i+1]]
    h = x[i + 1] - x[i]
    t = (xi - x[i]) / h

    # Hermite basis functions
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2

    return h00 * y[i] + h10 * h * d[i] + h01 * y[i + 1] + h11 * h * d[i + 1]


def _pchip_slopes(x: List[float], y: List[float]) -> List[float]:
    """
    Compute PCHIP slopes using Fritsch-Carlson method.

    Ensures monotonicity preservation.
    """
    n = len(x)
    if n < 2:
        return [0.0] * n

    # Compute secant slopes
    h = [x[i + 1] - x[i] for i in range(n - 1)]
    delta = [(y[i + 1] - y[i]) / h[i] for i in range(n - 1)]

    # Initialize slopes
    d = [0.0] * n

    # Interior points: use weighted harmonic mean
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] > 0:
            # Same sign - compute weighted harmonic mean
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
        else:
            # Different signs or zero - set slope to zero
            d[i] = 0.0

    # Endpoints: one-sided differences
    d[0] = _pchip_endpoint_slope(h[0], h[1] if n > 2 else h[0],
                                  delta[0], delta[1] if n > 2 else delta[0])
    d[-1] = _pchip_endpoint_slope(h[-1], h[-2] if n > 2 else h[-1],
                                   delta[-1], delta[-2] if n > 2 else delta[-1])

    return d


def _pchip_endpoint_slope(h1: float, h2: float, del1: float, del2: float) -> float:
    """Compute endpoint slope for PCHIP."""
    # Non-centered shape-preserving three-point formula
    d = ((2 * h1 + h2) * del1 - h1 * del2) / (h1 + h2)

    # Ensure monotonicity
    if d * del1 < 0:
        d = 0.0
    elif del1 * del2 < 0 and abs(d) > abs(3 * del1):
        d = 3 * del1

    return d


def monotone_convex(
    x: List[float],
    y: List[float],
    xi: float,
) -> float:
    """
    Monotone convex interpolation (Hagan-West method).

    Designed for yield curves. Preserves monotonicity and convexity.

    Args:
        x: X coordinates (sorted ascending)
        y: Y values (typically forward rates or similar)
        xi: Point to interpolate at

    Returns:
        Interpolated value at xi

    Reference:
        Hagan & West (2006): "Interpolation Methods for Curve Construction"
    """
    n = len(x)
    if n < 2:
        return y[0] if y else 0.0

    # Handle boundaries
    if xi <= x[0]:
        return y[0]
    if xi >= x[-1]:
        return y[-1]

    # Find interval
    i = bisect.bisect_right(x, xi) - 1
    i = min(i, n - 2)

    # Compute discrete forward rates
    f = [0.0] * (n - 1)
    for j in range(n - 1):
        f[j] = (y[j + 1] - y[j]) / (x[j + 1] - x[j])

    # Compute instantaneous forward rates at nodes
    f_inst = [0.0] * n

    # Interior nodes
    for j in range(1, n - 1):
        if f[j - 1] * f[j] > 0:
            # Same sign
            f_inst[j] = (f[j - 1] * (x[j + 1] - x[j]) + f[j] * (x[j] - x[j - 1])) / (x[j + 1] - x[j - 1])
        else:
            f_inst[j] = 0.0

    # Endpoints
    f_inst[0] = f[0] - 0.5 * (f[1] - f[0]) if n > 2 else f[0]
    f_inst[-1] = f[-1] + 0.5 * (f[-1] - f[-2]) if n > 2 else f[-1]

    # Ensure monotonicity
    for j in range(n - 1):
        f_inst[j] = max(0, min(2 * f[j], f_inst[j])) if f[j] >= 0 else min(0, max(2 * f[j], f_inst[j]))
        f_inst[j + 1] = max(0, min(2 * f[j], f_inst[j + 1])) if f[j] >= 0 else min(0, max(2 * f[j], f_inst[j + 1]))

    # Interpolate using Hermite polynomial
    h = x[i + 1] - x[i]
    t = (xi - x[i]) / h

    # Hermite basis
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2

    return h00 * y[i] + h10 * h * f_inst[i] + h01 * y[i + 1] + h11 * h * f_inst[i + 1]
