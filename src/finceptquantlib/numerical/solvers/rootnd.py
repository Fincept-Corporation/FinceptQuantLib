"""
Multi-dimensional Root Finding

Solvers for F(x) = 0 where F: R^n -> R^n:
- NewtonRaphson: Standard Newton's method
- Broyden: Quasi-Newton with Jacobian update
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from finceptquantlib.numerical.linalg import solve, lstsq, norm
from finceptquantlib.numerical.differentiation import finite_difference_jacobian, JacobianComputer


@dataclass
class RootResultND:
    """Result of multi-dimensional root finding."""
    x: List[float]
    converged: bool
    iterations: int
    residual: float
    message: str = ""


class NewtonRaphson:
    """
    Newton-Raphson method for systems of equations.

    Solves F(x) = 0 using x_{k+1} = x_k - J^{-1} F(x_k)
    """

    def __init__(
        self,
        tol: float = 1e-10,
        maxiter: int = 100,
        damping: float = 1.0,
        jacobian_method: str = "forward",
        jacobian_h: float = 1e-7,
    ):
        """
        Args:
            tol: Convergence tolerance on residual norm
            maxiter: Maximum iterations
            damping: Damping factor (0 < damping <= 1)
            jacobian_method: "forward" or "central" for finite differences
            jacobian_h: Step size for finite difference Jacobian
        """
        self.tol = tol
        self.maxiter = maxiter
        self.damping = damping
        self._jacobian = JacobianComputer(method=jacobian_method, h=jacobian_h)

    def solve(
        self,
        F: Callable[[List[float]], List[float]],
        x0: List[float],
        jacobian: Optional[Callable[[List[float]], List[List[float]]]] = None,
    ) -> RootResultND:
        """
        Solve F(x) = 0.

        Args:
            F: Vector function to find root of
            x0: Initial guess
            jacobian: Optional analytical Jacobian function

        Returns:
            RootResultND with solution and convergence info
        """
        x = [float(v) for v in x0]

        for iteration in range(self.maxiter):
            Fx = F(x)
            residual = norm(Fx)

            if residual < self.tol:
                return RootResultND(
                    x=x,
                    converged=True,
                    iterations=iteration + 1,
                    residual=residual,
                    message="Converged",
                )

            # Compute Jacobian
            if jacobian is not None:
                J = jacobian(x)
            else:
                J = self._jacobian(F, x)

            # Solve J * dx = -F(x)
            neg_Fx = [-v for v in Fx]
            try:
                dx = solve(J, neg_Fx)
            except ValueError:
                # Singular Jacobian - try least squares
                dx = lstsq(J, neg_Fx)

            # Update with damping
            x = [xi + self.damping * dxi for xi, dxi in zip(x, dx)]

        Fx = F(x)
        return RootResultND(
            x=x,
            converged=False,
            iterations=self.maxiter,
            residual=norm(Fx),
            message="Max iterations reached",
        )


class Broyden:
    """
    Broyden's method for systems of equations.

    Quasi-Newton method that updates Jacobian approximation.
    Faster than Newton when Jacobian is expensive.
    """

    def __init__(
        self,
        tol: float = 1e-10,
        maxiter: int = 100,
        jacobian_h: float = 1e-7,
    ):
        self.tol = tol
        self.maxiter = maxiter
        self.jacobian_h = jacobian_h

    def solve(
        self,
        F: Callable[[List[float]], List[float]],
        x0: List[float],
    ) -> RootResultND:
        """
        Solve F(x) = 0 using Broyden's method.

        Args:
            F: Vector function to find root of
            x0: Initial guess

        Returns:
            RootResultND with solution and convergence info
        """
        n = len(x0)
        x = [float(v) for v in x0]

        # Initial Jacobian approximation
        J = finite_difference_jacobian(F, x, self.jacobian_h)
        Fx = F(x)

        for iteration in range(self.maxiter):
            residual = norm(Fx)

            if residual < self.tol:
                return RootResultND(
                    x=x,
                    converged=True,
                    iterations=iteration + 1,
                    residual=residual,
                    message="Converged",
                )

            # Solve J * dx = -F
            neg_Fx = [-v for v in Fx]
            try:
                dx = solve(J, neg_Fx)
            except ValueError:
                dx = lstsq(J, neg_Fx)

            # Update x
            x_new = [xi + dxi for xi, dxi in zip(x, dx)]
            Fx_new = F(x_new)

            # Broyden update: J_new = J + (dF - J*dx) * dx^T / (dx^T * dx)
            dF = [fn - f for fn, f in zip(Fx_new, Fx)]
            Jdx = [sum(J[i][j] * dx[j] for j in range(n)) for i in range(n)]
            y = [df - jd for df, jd in zip(dF, Jdx)]

            dx_norm_sq = sum(d * d for d in dx)
            if dx_norm_sq > 1e-15:
                for i in range(n):
                    for j in range(n):
                        J[i][j] += y[i] * dx[j] / dx_norm_sq

            x = x_new
            Fx = Fx_new

        return RootResultND(
            x=x,
            converged=False,
            iterations=self.maxiter,
            residual=norm(Fx),
            message="Max iterations reached",
        )


# =============================================================================
# Convenience function with simple API
# =============================================================================

def find_root_nd(
    F: Callable[[List[float]], List[float]],
    x0: List[float],
    tol: float = 1e-10,
    maxiter: int = 100,
    method: str = "newton",
    return_info: bool = True,
) -> RootResultND | List[float]:
    """
    Find root of F(x) = 0 for vector function.

    Args:
        F: Vector function to find root of
        x0: Initial guess
        tol: Tolerance for convergence
        maxiter: Maximum iterations
        method: "newton" or "broyden"
        return_info: If False, return just the solution vector

    Returns:
        RootResultND if return_info=True, else List[float]
    """
    if method == "newton":
        solver = NewtonRaphson(tol=tol, maxiter=maxiter)
        result = solver.solve(F, x0)
    elif method == "broyden":
        solver = Broyden(tol=tol, maxiter=maxiter)
        result = solver.solve(F, x0)
    else:
        raise ValueError(f"Unknown method: {method}")

    if return_info:
        return result
    return result.x
