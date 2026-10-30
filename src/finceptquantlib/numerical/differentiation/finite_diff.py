"""
Finite Difference Methods

Numerical differentiation using finite differences:
- Jacobian computation (forward, central)
- Gradient computation
- Hessian computation
- Parallel implementations for expensive functions
"""

from __future__ import annotations

from typing import Callable, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


def finite_difference_jacobian(
    f: Callable[[List[float]], List[float]],
    x: List[float],
    h: float = 1e-7,
    method: str = "forward",
) -> List[List[float]]:
    """
    Compute Jacobian using finite differences.

    Args:
        f: Vector function f(x) -> y
        x: Point at which to compute Jacobian
        h: Step size
        method: "forward" or "central"

    Returns:
        Jacobian matrix J[i][j] = df_i/dx_j
    """
    n = len(x)
    fx = f(x)
    m = len(fx)

    J = [[0.0] * n for _ in range(m)]

    if method == "central":
        for j in range(n):
            x_up = x.copy()
            x_down = x.copy()
            x_up[j] += h
            x_down[j] -= h
            f_up = f(x_up)
            f_down = f(x_down)
            for i in range(m):
                J[i][j] = (f_up[i] - f_down[i]) / (2 * h)
    else:  # forward
        for j in range(n):
            x_up = x.copy()
            x_up[j] += h
            f_up = f(x_up)
            for i in range(m):
                J[i][j] = (f_up[i] - fx[i]) / h

    return J


def finite_difference_gradient(
    f: Callable[[List[float]], float],
    x: List[float],
    h: float = 1e-7,
    method: str = "forward",
) -> List[float]:
    """
    Compute gradient of scalar function using finite differences.

    Args:
        f: Scalar function f(x) -> float
        x: Point at which to compute gradient
        h: Step size
        method: "forward" or "central"

    Returns:
        Gradient vector
    """
    n = len(x)
    grad = [0.0] * n

    if method == "central":
        for j in range(n):
            x_up = x.copy()
            x_down = x.copy()
            x_up[j] += h
            x_down[j] -= h
            grad[j] = (f(x_up) - f(x_down)) / (2 * h)
    else:  # forward
        fx = f(x)
        for j in range(n):
            x_up = x.copy()
            x_up[j] += h
            grad[j] = (f(x_up) - fx) / h

    return grad


def finite_difference_derivative(
    f: Callable[[float], float],
    x: float,
    h: float = 1e-7,
    method: str = "central",
) -> float:
    """
    Compute derivative of 1D function using finite differences.

    Args:
        f: Scalar function f(x) -> float
        x: Point at which to compute derivative
        h: Step size
        method: "forward" or "central"

    Returns:
        Derivative f'(x)
    """
    if method == "central":
        return (f(x + h) - f(x - h)) / (2 * h)
    else:
        return (f(x + h) - f(x)) / h


def finite_difference_hessian(
    f: Callable[[List[float]], float],
    x: List[float],
    h: float = 1e-5,
) -> List[List[float]]:
    """
    Compute Hessian matrix using finite differences.

    Args:
        f: Scalar function f(x) -> float
        x: Point at which to compute Hessian
        h: Step size

    Returns:
        Hessian matrix H[i][j] = d^2f/dx_i dx_j
    """
    n = len(x)
    H = [[0.0] * n for _ in range(n)]

    fx = f(x)

    for i in range(n):
        # Diagonal: (f(x+h_i) - 2*f(x) + f(x-h_i)) / h^2
        x_up = x.copy()
        x_down = x.copy()
        x_up[i] += h
        x_down[i] -= h
        H[i][i] = (f(x_up) - 2 * fx + f(x_down)) / (h * h)

        # Off-diagonal: symmetric
        for j in range(i + 1, n):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()

            x_pp[i] += h
            x_pp[j] += h
            x_pm[i] += h
            x_pm[j] -= h
            x_mp[i] -= h
            x_mp[j] += h
            x_mm[i] -= h
            x_mm[j] -= h

            H[i][j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h * h)
            H[j][i] = H[i][j]

    return H


class JacobianComputer:
    """
    Configurable Jacobian computation with caching.
    """

    def __init__(
        self,
        method: str = "forward",
        h: float = 1e-7,
        use_ad: bool = False,
    ):
        """
        Args:
            method: "forward" or "central" for finite differences
            h: Step size for finite differences
            use_ad: Use automatic differentiation if available
        """
        self.method = method
        self.h = h
        self.use_ad = use_ad
        self._cache: Optional[tuple] = None

    def __call__(
        self,
        f: Callable[[List[float]], List[float]],
        x: List[float],
    ) -> List[List[float]]:
        """Compute Jacobian at point x."""
        if self.use_ad:
            return self._ad_jacobian(f, x)
        return finite_difference_jacobian(f, x, self.h, self.method)

    def _ad_jacobian(
        self,
        f: Callable[[List[float]], List[float]],
        x: List[float],
    ) -> List[List[float]]:
        """
        Compute Jacobian using automatic differentiation.
        Falls back to finite differences if AD not available.
        """
        try:
            from finceptquantlib.core.foundations.autodiff import Dual

            n = len(x)
            fx = f(x)
            m = len(fx)
            J = [[0.0] * n for _ in range(m)]

            for j in range(n):
                # Create dual numbers with derivative in j-th direction
                x_dual = [Dual(xi, 1.0 if i == j else 0.0) for i, xi in enumerate(x)]
                f_dual = f(x_dual)

                for i in range(m):
                    if hasattr(f_dual[i], 'dual'):
                        J[i][j] = f_dual[i].dual
                    else:
                        J[i][j] = 0.0

            return J
        except (ImportError, TypeError):
            # Fall back to finite differences
            return finite_difference_jacobian(f, x, self.h, self.method)


# =============================================================================
# Parallel Jacobian Computation
# =============================================================================

def parallel_jacobian(
    f: Callable[[List[float]], List[float]],
    x: List[float],
    h: float = 1e-7,
    method: str = "forward",
    max_workers: Optional[int] = None,
) -> List[List[float]]:
    """
    Compute Jacobian using parallel finite differences.

    Uses thread pool to compute column derivatives in parallel.
    Beneficial for expensive function evaluations.

    Args:
        f: Vector function f(x) -> y
        x: Point at which to compute Jacobian
        h: Step size
        method: "forward" or "central"
        max_workers: Number of threads (defaults to min(n, CPU count))

    Returns:
        Jacobian matrix J[i][j] = df_i/dx_j
    """
    n = len(x)
    fx = f(x)
    m = len(fx)

    if max_workers is None:
        max_workers = min(n, os.cpu_count() or 4)

    # Initialize Jacobian
    J = [[0.0] * n for _ in range(m)]

    def compute_column(j: int) -> tuple[int, List[float]]:
        """Compute j-th column of Jacobian."""
        if method == "central":
            x_up = x.copy()
            x_down = x.copy()
            x_up[j] += h
            x_down[j] -= h
            f_up = f(x_up)
            f_down = f(x_down)
            col = [(f_up[i] - f_down[i]) / (2 * h) for i in range(m)]
        else:  # forward
            x_up = x.copy()
            x_up[j] += h
            f_up = f(x_up)
            col = [(f_up[i] - fx[i]) / h for i in range(m)]
        return j, col

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_column, j) for j in range(n)]
        for future in as_completed(futures):
            j, col = future.result()
            for i in range(m):
                J[i][j] = col[i]

    return J


def parallel_gradient(
    f: Callable[[List[float]], float],
    x: List[float],
    h: float = 1e-7,
    method: str = "forward",
    max_workers: Optional[int] = None,
) -> List[float]:
    """
    Compute gradient using parallel finite differences.

    Args:
        f: Scalar function f(x) -> float
        x: Point at which to compute gradient
        h: Step size
        method: "forward" or "central"
        max_workers: Number of threads

    Returns:
        Gradient vector
    """
    n = len(x)

    if max_workers is None:
        max_workers = min(n, os.cpu_count() or 4)

    grad = [0.0] * n

    if method == "central":
        def compute_element(j: int) -> tuple[int, float]:
            x_up = x.copy()
            x_down = x.copy()
            x_up[j] += h
            x_down[j] -= h
            return j, (f(x_up) - f(x_down)) / (2 * h)
    else:
        fx = f(x)

        def compute_element(j: int) -> tuple[int, float]:
            x_up = x.copy()
            x_up[j] += h
            return j, (f(x_up) - fx) / h

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_element, j) for j in range(n)]
        for future in as_completed(futures):
            j, val = future.result()
            grad[j] = val

    return grad


class ParallelJacobianComputer:
    """
    Parallel Jacobian computation with configurable thread pool.
    """

    def __init__(
        self,
        method: str = "forward",
        h: float = 1e-7,
        max_workers: Optional[int] = None,
    ):
        """
        Args:
            method: "forward" or "central" for finite differences
            h: Step size for finite differences
            max_workers: Number of threads (None = auto)
        """
        self.method = method
        self.h = h
        self.max_workers = max_workers

    def __call__(
        self,
        f: Callable[[List[float]], List[float]],
        x: List[float],
    ) -> List[List[float]]:
        """Compute Jacobian at point x using parallel threads."""
        return parallel_jacobian(f, x, self.h, self.method, self.max_workers)
