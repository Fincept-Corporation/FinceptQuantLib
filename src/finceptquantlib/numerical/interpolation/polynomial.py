"""
Polynomial Interpolation

Basic polynomial interpolation methods:
- Linear interpolation
- Cubic polynomial interpolation
- Lagrange interpolation
"""

from __future__ import annotations

from typing import List, Callable
import bisect


def linear_interp(
    x: List[float],
    y: List[float],
    xi: float,
    extrapolate: bool = False,
) -> float:
    """
    Linear interpolation.

    Args:
        x: X coordinates (sorted ascending)
        y: Y values
        xi: Point to interpolate at
        extrapolate: Allow extrapolation outside range

    Returns:
        Interpolated value at xi
    """
    n = len(x)
    if n == 0:
        raise ValueError("Empty arrays")
    if n == 1:
        return y[0]

    # Handle boundaries
    if xi <= x[0]:
        if extrapolate:
            # Linear extrapolation
            slope = (y[1] - y[0]) / (x[1] - x[0])
            return y[0] + slope * (xi - x[0])
        return y[0]

    if xi >= x[-1]:
        if extrapolate:
            # Linear extrapolation
            slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
            return y[-1] + slope * (xi - x[-1])
        return y[-1]

    # Find interval
    i = bisect.bisect_right(x, xi) - 1

    # Interpolate
    t = (xi - x[i]) / (x[i + 1] - x[i])
    return y[i] * (1 - t) + y[i + 1] * t


def cubic_interp(
    x: List[float],
    y: List[float],
    xi: float,
) -> float:
    """
    Local cubic interpolation using 4 neighboring points.

    Args:
        x: X coordinates (sorted ascending)
        y: Y values
        xi: Point to interpolate at

    Returns:
        Interpolated value at xi
    """
    n = len(x)
    if n < 4:
        return linear_interp(x, y, xi)

    # Find interval
    i = bisect.bisect_right(x, xi) - 1
    i = max(1, min(i, n - 3))

    # Use 4 points: i-1, i, i+1, i+2
    x0, x1, x2, x3 = x[i - 1], x[i], x[i + 1], x[i + 2]
    y0, y1, y2, y3 = y[i - 1], y[i], y[i + 1], y[i + 2]

    # Lagrange interpolation with 4 points
    result = 0.0
    for j, (xj, yj) in enumerate([(x0, y0), (x1, y1), (x2, y2), (x3, y3)]):
        basis = yj
        for k, xk in enumerate([x0, x1, x2, x3]):
            if k != j:
                basis *= (xi - xk) / (xj - xk)
        result += basis

    return result


def lagrange_interp(
    x: List[float],
    y: List[float],
    xi: float,
) -> float:
    """
    Lagrange polynomial interpolation.

    Uses all points to construct the interpolating polynomial.
    O(n^2) complexity - use for small n only.

    Args:
        x: X coordinates
        y: Y values
        xi: Point to interpolate at

    Returns:
        Interpolated value at xi
    """
    n = len(x)
    if n == 0:
        raise ValueError("Empty arrays")
    if n == 1:
        return y[0]

    result = 0.0
    for i in range(n):
        basis = y[i]
        for j in range(n):
            if i != j:
                basis *= (xi - x[j]) / (x[i] - x[j])
        result += basis

    return result


def linear_interp_1d(
    x: List[float],
    y: List[float],
) -> Callable[[float], float]:
    """
    Create a linear interpolation function.

    Args:
        x: X coordinates (sorted ascending)
        y: Y values

    Returns:
        Interpolation function f(xi) -> yi
    """
    def interpolate(xi: float) -> float:
        return linear_interp(x, y, xi, extrapolate=True)
    return interpolate
