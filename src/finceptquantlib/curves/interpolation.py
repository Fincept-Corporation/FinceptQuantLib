"""
Interpolation Methods for Yield Curves

Implements various interpolation methods used in curve construction:
- Linear: Simple linear interpolation
- Log-Linear: Linear in log space (for discount factors)
- Cubic Spline: Natural cubic spline
- Monotone Convex: Hagan-West method
- Flat Forward/Backward: Piecewise constant forwards

All interpolators support automatic differentiation through the AD engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union
import math
import bisect

from finceptquantlib.core.foundations.autodiff import Dual


class Interpolator(ABC):
    """
    Abstract base class for interpolation methods.

    Interpolators take a set of (x, y) points and provide
    methods to interpolate values at arbitrary x positions.
    """

    def __init__(
        self,
        x: list[float],
        y: list[float],
    ):
        """
        Initialize interpolator with data points.

        Args:
            x: X coordinates (must be sorted ascending)
            y: Y values at each x
        """
        self._x = [float(v) for v in x]
        self._y = [float(v) for v in y]

        if len(self._x) != len(self._y):
            raise ValueError("x and y must have same length")
        if len(self._x) < 2:
            raise ValueError("At least 2 points required")
        # Check ascending order
        for i in range(len(self._x) - 1):
            if self._x[i+1] <= self._x[i]:
                raise ValueError("x must be sorted ascending with no duplicates")

    @property
    def x(self) -> list[float]:
        """X coordinates."""
        return self._x

    @property
    def y(self) -> list[float]:
        """Y values."""
        return self._y

    @abstractmethod
    def __call__(self, x: Union[float, list[float]]) -> Union[float, list[float]]:
        """
        Interpolate at x.

        Args:
            x: Point(s) to interpolate at

        Returns:
            Interpolated value(s)
        """

    def _find_segment(self, x: float) -> int:
        """
        Find the segment index for x.

        Returns i such that x[i] <= x < x[i+1]
        """
        idx = bisect.bisect_right(self._x, x) - 1
        return max(0, min(idx, len(self._x) - 2))

    def derivative(self, x: Union[float, list[float]]) -> Union[float, list[float]]:
        """
        Compute the derivative at x using automatic differentiation.

        Args:
            x: Point(s) to evaluate derivative

        Returns:
            Derivative value(s)
        """
        if isinstance(x, (int, float)):
            x_dual = Dual(x, 1.0)
            result = self._interpolate_dual(x_dual)
            return result.deriv if isinstance(result, Dual) else 0.0
        else:
            return [self.derivative(xi) for xi in x]

    def _interpolate_dual(self, x: Dual) -> Dual:
        """Interpolate with Dual number support for AD."""
        # Default implementation - override for efficiency
        return Dual(self(x.value), self.derivative(x.value))


class LinearInterpolator(Interpolator):
    """
    Linear interpolation.

    Connects points with straight lines. Simple and fast but
    produces discontinuous first derivatives.

    y(x) = y_i + (y_{i+1} - y_i) * (x - x_i) / (x_{i+1} - x_i)
    """

    def __call__(self, x: Union[float, list[float]]) -> Union[float, list[float]]:
        if isinstance(x, (int, float)):
            return self._interpolate_scalar(x)
        return [self._interpolate_scalar(xi) for xi in x]

    def _interpolate_scalar(self, x: float) -> float:
        # Handle extrapolation
        if x <= self._x[0]:
            return float(self._y[0])
        if x >= self._x[-1]:
            return float(self._y[-1])

        i = self._find_segment(x)
        t = (x - self._x[i]) / (self._x[i + 1] - self._x[i])
        return float(self._y[i] + t * (self._y[i + 1] - self._y[i]))

    def derivative(self, x: Union[float, list[float]]) -> Union[float, list[float]]:
        if isinstance(x, (int, float)):
            return self._derivative_scalar(x)
        return [self._derivative_scalar(xi) for xi in x]

    def _derivative_scalar(self, x: float) -> float:
        if x <= self._x[0]:
            # Flat extrapolation => zero derivative
            return 0.0
        if x >= self._x[-1]:
            return 0.0

        i = self._find_segment(x)
        return float((self._y[i + 1] - self._y[i]) / (self._x[i + 1] - self._x[i]))


class LogLinearInterpolator(Interpolator):
    """
    Log-linear interpolation.

    Linear interpolation in log space. Commonly used for discount
    factors to ensure positivity and smooth continuous compounding.

    log(y(x)) = log(y_i) + (log(y_{i+1}) - log(y_i)) * (x - x_i) / (x_{i+1} - x_i)
    """

    def __init__(self, x: list[float], y: list[float]):
        super().__init__(x, y)
        if any(yi <= 0 for yi in self._y):
            raise ValueError("All y values must be positive for log-linear interpolation")
        self._log_y = [math.log(yi) for yi in self._y]

    def __call__(self, x: Union[float, list[float]]) -> Union[float, list[float]]:
        if isinstance(x, (int, float)):
            return self._interpolate_scalar(x)
        return [self._interpolate_scalar(xi) for xi in x]

    def _interpolate_scalar(self, x: float) -> float:
        if x <= self._x[0]:
            return float(self._y[0])
        if x >= self._x[-1]:
            return float(self._y[-1])

        i = self._find_segment(x)
        t = (x - self._x[i]) / (self._x[i + 1] - self._x[i])
        log_y = self._log_y[i] + t * (self._log_y[i + 1] - self._log_y[i])
        return float(math.exp(log_y))

    def derivative(self, x: Union[float, list[float]]) -> Union[float, list[float]]:
        if isinstance(x, (int, float)):
            return self._derivative_scalar(x)
        return [self._derivative_scalar(xi) for xi in x]

    def _derivative_scalar(self, x: float) -> float:
        if x <= self._x[0] or x >= self._x[-1]:
            return 0.0

        i = self._find_segment(x)
        y = self._interpolate_scalar(x)
        slope = (self._log_y[i + 1] - self._log_y[i]) / (self._x[i + 1] - self._x[i])
        return float(y * slope)


class CubicSplineInterpolator(Interpolator):
    """
    Natural cubic spline interpolation.

    Produces smooth curves with continuous first and second derivatives.
    Uses natural boundary conditions (second derivative = 0 at ends).

    S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3
    """

    def __init__(self, x: list[float], y: list[float]):
        super().__init__(x, y)
        self._compute_coefficients()

    def _compute_coefficients(self) -> None:
        """Compute cubic spline coefficients using tridiagonal solver."""
        n = len(self._x)
        h = [self._x[i + 1] - self._x[i] for i in range(n - 1)]

        # Build tridiagonal system for second derivatives
        # Natural spline: M[0] = M[n-1] = 0

        # Diagonal
        diag = [0.0] * n
        diag[0] = 1.0
        diag[-1] = 1.0
        for i in range(1, n - 1):
            diag[i] = 2 * (h[i - 1] + h[i])

        # Off-diagonals
        upper = [0.0] * (n - 1)
        lower = [0.0] * (n - 1)
        for i in range(1, n - 1):
            upper[i] = h[i]
            lower[i - 1] = h[i - 1]

        # Right-hand side
        rhs = [0.0] * n
        for i in range(1, n - 1):
            rhs[i] = 6 * ((self._y[i + 1] - self._y[i]) / h[i] -
                          (self._y[i] - self._y[i - 1]) / h[i - 1])

        # Solve tridiagonal system (Thomas algorithm)
        M = self._solve_tridiagonal(lower, diag, upper, rhs)

        # Compute polynomial coefficients
        self._a = self._y[:-1].copy()
        self._b = [0.0] * (n - 1)
        self._c = [0.0] * (n - 1)
        self._d = [0.0] * (n - 1)

        for i in range(n - 1):
            self._b[i] = (self._y[i + 1] - self._y[i]) / h[i] - h[i] * (2 * M[i] + M[i + 1]) / 6
            self._c[i] = M[i] / 2
            self._d[i] = (M[i + 1] - M[i]) / (6 * h[i])

    def _solve_tridiagonal(
        self,
        lower: list[float],
        diag: list[float],
        upper: list[float],
        rhs: list[float],
    ) -> list[float]:
        """Solve tridiagonal system using Thomas algorithm."""
        n = len(diag)
        c = [0.0] * (n - 1)
        d = [0.0] * n

        # Forward sweep
        c[0] = upper[0] / diag[0]
        d[0] = rhs[0] / diag[0]

        for i in range(1, n - 1):
            denom = diag[i] - lower[i - 1] * c[i - 1]
            c[i] = upper[i] / denom
            d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / denom

        d[n - 1] = (rhs[n - 1] - lower[n - 2] * d[n - 2]) / (diag[n - 1] - lower[n - 2] * c[n - 2])

        # Back substitution
        x_sol = [0.0] * n
        x_sol[n - 1] = d[n - 1]
        for i in range(n - 2, -1, -1):
            x_sol[i] = d[i] - c[i] * x_sol[i + 1]

        return x_sol

    def __call__(self, x: Union[float, list[float]]) -> Union[float, list[float]]:
        if isinstance(x, (int, float)):
            return self._interpolate_scalar(x)
        return [self._interpolate_scalar(xi) for xi in x]

    def _interpolate_scalar(self, x: float) -> float:
        if x <= self._x[0]:
            return float(self._y[0])
        if x >= self._x[-1]:
            return float(self._y[-1])

        i = self._find_segment(x)
        dx = x - self._x[i]
        return float(self._a[i] + self._b[i] * dx + self._c[i] * dx**2 + self._d[i] * dx**3)

    def derivative(self, x: Union[float, list[float]]) -> Union[float, list[float]]:
        if isinstance(x, (int, float)):
            return self._derivative_scalar(x)
        return [self._derivative_scalar(xi) for xi in x]

    def _derivative_scalar(self, x: float) -> float:
        if x <= self._x[0]:
            # Return slope at left boundary (flat extrapolation => 0)
            return 0.0
        if x >= self._x[-1]:
            return 0.0

        i = self._find_segment(x)
        dx = x - self._x[i]
        return float(self._b[i] + 2 * self._c[i] * dx + 3 * self._d[i] * dx**2)


class MonotoneConvexInterpolator(Interpolator):
    """
    Monotone Convex interpolation (Hagan-West method).

    Preserves monotonicity and produces smooth forward rates.
    Particularly good for yield curve construction.

    Reference: Hagan & West, "Methods for Constructing a Yield Curve"
    """

    def __init__(self, x: list[float], y: list[float]):
        super().__init__(x, y)
        self._compute_derivatives()

    def _compute_derivatives(self) -> None:
        """Compute monotone-preserving derivatives."""
        n = len(self._x)
        h = [self._x[i + 1] - self._x[i] for i in range(n - 1)]
        delta = [(self._y[i + 1] - self._y[i]) / h[i] for i in range(n - 1)]

        # Initialize derivatives
        self._m = [0.0] * n

        # Interior points: weighted average
        for i in range(1, n - 1):
            if delta[i - 1] * delta[i] > 0:
                w = h[i] / (h[i - 1] + h[i])
                self._m[i] = (1 - w) * delta[i - 1] + w * delta[i]
            else:
                self._m[i] = 0.0

        # Endpoints
        self._m[0] = delta[0]
        self._m[-1] = delta[-1]

        # Apply monotonicity constraint
        for i in range(n - 1):
            if delta[i] == 0:
                self._m[i] = 0
                self._m[i + 1] = 0
            else:
                alpha = self._m[i] / delta[i]
                beta = self._m[i + 1] / delta[i]

                if alpha < 0 or beta < 0:
                    self._m[i] = 0
                    self._m[i + 1] = 0
                elif alpha**2 + beta**2 > 9:
                    tau = 3 / math.sqrt(alpha**2 + beta**2)
                    self._m[i] = tau * alpha * delta[i]
                    self._m[i + 1] = tau * beta * delta[i]

    def __call__(self, x: Union[float, list[float]]) -> Union[float, list[float]]:
        if isinstance(x, (int, float)):
            return self._interpolate_scalar(x)
        return [self._interpolate_scalar(xi) for xi in x]

    def _interpolate_scalar(self, x: float) -> float:
        if x <= self._x[0]:
            return float(self._y[0])
        if x >= self._x[-1]:
            return float(self._y[-1])

        i = self._find_segment(x)
        h = self._x[i + 1] - self._x[i]
        t = (x - self._x[i]) / h

        # Hermite basis functions
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2

        return float(
            h00 * self._y[i] +
            h10 * h * self._m[i] +
            h01 * self._y[i + 1] +
            h11 * h * self._m[i + 1]
        )

    def derivative(self, x: Union[float, list[float]]) -> Union[float, list[float]]:
        if isinstance(x, (int, float)):
            return self._derivative_scalar(x)
        return [self._derivative_scalar(xi) for xi in x]

    def _derivative_scalar(self, x: float) -> float:
        if x <= self._x[0] or x >= self._x[-1]:
            return 0.0

        i = self._find_segment(x)
        h = self._x[i + 1] - self._x[i]
        t = (x - self._x[i]) / h

        # Derivatives of Hermite basis functions
        dh00 = 6 * t**2 - 6 * t
        dh10 = 3 * t**2 - 4 * t + 1
        dh01 = -6 * t**2 + 6 * t
        dh11 = 3 * t**2 - 2 * t

        return float(
            (dh00 * self._y[i] +
             dh10 * h * self._m[i] +
             dh01 * self._y[i + 1] +
             dh11 * h * self._m[i + 1]) / h
        )


class FlatForwardInterpolator(Interpolator):
    """
    Flat forward interpolation.

    Forward rates are constant between nodes.
    Produces a step function in forward rates.
    """

    def __call__(self, x: Union[float, list[float]]) -> Union[float, list[float]]:
        if isinstance(x, (int, float)):
            return self._interpolate_scalar(x)
        return [self._interpolate_scalar(xi) for xi in x]

    def _interpolate_scalar(self, x: float) -> float:
        if x <= self._x[0]:
            return float(self._y[0])
        if x >= self._x[-1]:
            return float(self._y[-1])

        i = self._find_segment(x)
        return float(self._y[i + 1])  # Use right value (forward)

    def derivative(self, x: Union[float, list[float]]) -> Union[float, list[float]]:
        return 0.0 if isinstance(x, (int, float)) else [0.0] * len(x)


class FlatBackwardInterpolator(Interpolator):
    """
    Flat backward interpolation.

    Uses the left node value. Forward rates step at each node.
    """

    def __call__(self, x: Union[float, list[float]]) -> Union[float, list[float]]:
        if isinstance(x, (int, float)):
            return self._interpolate_scalar(x)
        return [self._interpolate_scalar(xi) for xi in x]

    def _interpolate_scalar(self, x: float) -> float:
        if x <= self._x[0]:
            return float(self._y[0])
        if x >= self._x[-1]:
            return float(self._y[-1])

        i = self._find_segment(x)
        return float(self._y[i])  # Use left value (backward)

    def derivative(self, x: Union[float, list[float]]) -> Union[float, list[float]]:
        return 0.0 if isinstance(x, (int, float)) else [0.0] * len(x)
