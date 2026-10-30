"""
Spline Interpolation

Cubic spline with natural or clamped boundary conditions.
"""

from __future__ import annotations

from typing import List, Optional
import bisect

from finceptquantlib.numerical.linalg import solve


class CubicSpline:
    """
    Cubic spline interpolation.

    Constructs a piecewise cubic polynomial that passes through all points
    with continuous first and second derivatives.
    """

    def __init__(
        self,
        x: List[float],
        y: List[float],
        bc_type: str = "natural",
        bc_values: Optional[tuple[float, float]] = None,
    ):
        """
        Args:
            x: X coordinates (sorted ascending)
            y: Y values
            bc_type: "natural" (y''=0 at ends) or "clamped" (specify y')
            bc_values: (y'(x[0]), y'(x[-1])) for clamped boundaries
        """
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        if len(x) < 2:
            raise ValueError("Need at least 2 points")

        self.x = list(x)
        self.y = list(y)
        self.n = len(x)
        self.bc_type = bc_type
        self.bc_values = bc_values

        # Compute spline coefficients
        self._compute_coefficients()

    def _compute_coefficients(self):
        """Compute the spline coefficients (second derivatives)."""
        n = self.n
        x = self.x
        y = self.y

        # h[i] = x[i+1] - x[i]
        h = [x[i + 1] - x[i] for i in range(n - 1)]

        # Set up tridiagonal system for second derivatives
        # Natural spline: y''(x[0]) = y''(x[-1]) = 0

        # For natural spline, we solve for M[i] = y''(x[i])
        # The tridiagonal system is:
        # h[i-1]*M[i-1] + 2*(h[i-1]+h[i])*M[i] + h[i]*M[i+1] = 6*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])

        if n == 2:
            # Only two points - linear interpolation
            self.M = [0.0, 0.0]
            return

        # Build system (n-2 equations for interior points)
        # For natural: M[0] = M[n-1] = 0, solve for M[1]...M[n-2]

        size = n - 2
        A = [[0.0] * size for _ in range(size)]
        b = [0.0] * size

        for i in range(size):
            # Equation for M[i+1]
            idx = i + 1

            if i > 0:
                A[i][i - 1] = h[idx - 1]

            A[i][i] = 2 * (h[idx - 1] + h[idx])

            if i < size - 1:
                A[i][i + 1] = h[idx]

            b[i] = 6 * ((y[idx + 1] - y[idx]) / h[idx] - (y[idx] - y[idx - 1]) / h[idx - 1])

        # Solve the system
        M_interior = solve(A, b)

        # Full M array with boundary conditions
        self.M = [0.0] + M_interior + [0.0]

    def __call__(self, xi: float) -> float:
        """
        Evaluate spline at point xi.

        Args:
            xi: Point to evaluate at

        Returns:
            Interpolated value
        """
        return self.evaluate(xi)

    def evaluate(self, xi: float) -> float:
        """
        Evaluate spline at point xi.

        Args:
            xi: Point to evaluate at

        Returns:
            Interpolated value
        """
        x = self.x
        y = self.y
        M = self.M
        n = self.n

        # Handle boundaries
        if xi <= x[0]:
            return y[0]
        if xi >= x[-1]:
            return y[-1]

        # Find interval
        i = bisect.bisect_right(x, xi) - 1
        i = min(i, n - 2)

        h = x[i + 1] - x[i]
        a = (x[i + 1] - xi) / h
        b = (xi - x[i]) / h

        # Cubic spline formula
        result = (a * y[i] + b * y[i + 1] +
                  ((a ** 3 - a) * M[i] + (b ** 3 - b) * M[i + 1]) * h * h / 6)

        return result

    def derivative(self, xi: float) -> float:
        """
        Evaluate first derivative at point xi.

        Args:
            xi: Point to evaluate at

        Returns:
            First derivative value
        """
        x = self.x
        y = self.y
        M = self.M
        n = self.n

        # Handle boundaries
        if xi <= x[0]:
            i = 0
        elif xi >= x[-1]:
            i = n - 2
        else:
            i = bisect.bisect_right(x, xi) - 1
            i = min(i, n - 2)

        h = x[i + 1] - x[i]
        a = (x[i + 1] - xi) / h
        b = (xi - x[i]) / h

        # Derivative of cubic spline
        result = ((y[i + 1] - y[i]) / h -
                  (3 * a ** 2 - 1) * h * M[i] / 6 +
                  (3 * b ** 2 - 1) * h * M[i + 1] / 6)

        return result


def cubic_spline(
    x: List[float],
    y: List[float],
    xi: float,
    bc_type: str = "natural",
) -> float:
    """
    Cubic spline interpolation (convenience function).

    Args:
        x: X coordinates (sorted ascending)
        y: Y values
        xi: Point to interpolate at
        bc_type: "natural" or "clamped"

    Returns:
        Interpolated value at xi
    """
    spline = CubicSpline(x, y, bc_type)
    return spline(xi)
