"""
Interpolation Methods

Various interpolation schemes:
- Polynomial: Linear, cubic, Lagrange
- Spline: Cubic spline, B-spline
- Monotone: Monotone convex, PCHIP
"""

from finceptquantlib.numerical.interpolation.polynomial import (
    linear_interp,
    cubic_interp,
    lagrange_interp,
    linear_interp_1d,
)

from finceptquantlib.numerical.interpolation.spline import (
    cubic_spline,
    CubicSpline,
)

from finceptquantlib.numerical.interpolation.monotone import (
    monotone_convex,
    pchip,
)

__all__ = [
    # Polynomial
    "linear_interp",
    "cubic_interp",
    "lagrange_interp",
    "linear_interp_1d",
    # Spline
    "cubic_spline",
    "CubicSpline",
    # Monotone
    "monotone_convex",
    "pchip",
]
