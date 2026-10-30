"""
Nonlinear Least Squares

Solvers for min ||F(x)||^2:
- LevenbergMarquardt: Trust-region method (recommended)
- BoundedLevenbergMarquardt: With variable bounds
- GaussNewton: Basic least squares
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import math

from finceptquantlib.numerical.linalg import solve, lstsq
from finceptquantlib.numerical.differentiation import JacobianComputer


@dataclass
class LeastSquaresResult:
    """Result of nonlinear least squares optimization."""
    x: List[float]
    converged: bool
    iterations: int
    cost: float  # ||F(x)||^2
    residual_norm: float  # ||F(x)||
    jacobian: Optional[List[List[float]]] = None
    message: str = ""


class LevenbergMarquardt:
    """
    Levenberg-Marquardt algorithm for nonlinear least squares.

    Minimizes ||F(x)||^2 using adaptive trust region.
    Interpolates between Gauss-Newton and gradient descent.
    """

    def __init__(
        self,
        tol: float = 1e-10,
        maxiter: int = 200,
        lambda_init: float = 1e-3,
        lambda_up: float = 10.0,
        lambda_down: float = 10.0,
        jacobian_method: str = "forward",
        jacobian_h: float = 1e-7,
    ):
        self.tol = tol
        self.maxiter = maxiter
        self.lambda_init = lambda_init
        self.lambda_up = lambda_up
        self.lambda_down = lambda_down
        self._jacobian = JacobianComputer(method=jacobian_method, h=jacobian_h)

    def solve(
        self,
        F: Callable[[List[float]], List[float]],
        x0: List[float],
        jacobian: Optional[Callable[[List[float]], List[List[float]]]] = None,
    ) -> LeastSquaresResult:
        """
        Minimize ||F(x)||^2.

        Args:
            F: Vector function (residuals)
            x0: Initial guess
            jacobian: Optional analytical Jacobian function

        Returns:
            LeastSquaresResult with solution and convergence info
        """
        x = [float(v) for v in x0]
        lam = self.lambda_init

        Fx = F(x)
        cost = sum(v * v for v in Fx)

        J = None
        for iteration in range(self.maxiter):
            residual_norm = math.sqrt(cost)

            if residual_norm < self.tol:
                return LeastSquaresResult(
                    x=x,
                    converged=True,
                    iterations=iteration + 1,
                    cost=cost,
                    residual_norm=residual_norm,
                    jacobian=J,
                    message="Converged",
                )

            # Compute Jacobian
            if jacobian is not None:
                J = jacobian(x)
            else:
                J = self._jacobian(F, x)

            n = len(x)
            m = len(Fx)

            # Normal equations: (J^T J + lambda*I) dx = -J^T F
            JtJ = [[sum(J[k][i] * J[k][j] for k in range(m)) for j in range(n)] for i in range(n)]
            JtF = [sum(J[k][i] * Fx[k] for k in range(m)) for i in range(n)]

            # Add Marquardt diagonal scaling for better conditioning
            for i in range(n):
                JtJ[i][i] += lam * max(JtJ[i][i], 1e-6)

            neg_JtF = [-v for v in JtF]

            # Solve for step
            try:
                dx = solve(JtJ, neg_JtF)
            except ValueError:
                dx = lstsq(JtJ, neg_JtF)

            # Try step
            x_new = [xi + dxi for xi, dxi in zip(x, dx)]
            Fx_new = F(x_new)
            cost_new = sum(v * v for v in Fx_new)

            # Accept or reject
            if cost_new < cost:
                x = x_new
                Fx = Fx_new
                cost = cost_new
                lam /= self.lambda_down
            else:
                lam *= self.lambda_up

            # Check for stagnation
            if lam > 1e15:
                return LeastSquaresResult(
                    x=x,
                    converged=False,
                    iterations=iteration + 1,
                    cost=cost,
                    residual_norm=math.sqrt(cost),
                    jacobian=J,
                    message="Lambda overflow - problem may be ill-conditioned",
                )

        return LeastSquaresResult(
            x=x,
            converged=False,
            iterations=self.maxiter,
            cost=cost,
            residual_norm=math.sqrt(cost),
            jacobian=J,
            message="Max iterations reached",
        )


class GaussNewton:
    """
    Gauss-Newton algorithm for nonlinear least squares.

    Special case of Levenberg-Marquardt with lambda=0.
    Faster convergence near solution but less robust.
    """

    def __init__(
        self,
        tol: float = 1e-10,
        maxiter: int = 100,
        jacobian_method: str = "forward",
        jacobian_h: float = 1e-7,
    ):
        self.tol = tol
        self.maxiter = maxiter
        self._jacobian = JacobianComputer(method=jacobian_method, h=jacobian_h)

    def solve(
        self,
        F: Callable[[List[float]], List[float]],
        x0: List[float],
        jacobian: Optional[Callable[[List[float]], List[List[float]]]] = None,
    ) -> LeastSquaresResult:
        """Minimize ||F(x)||^2."""
        x = [float(v) for v in x0]

        J = None
        for iteration in range(self.maxiter):
            Fx = F(x)
            cost = sum(v * v for v in Fx)
            residual_norm = math.sqrt(cost)

            if residual_norm < self.tol:
                return LeastSquaresResult(
                    x=x,
                    converged=True,
                    iterations=iteration + 1,
                    cost=cost,
                    residual_norm=residual_norm,
                    jacobian=J,
                    message="Converged",
                )

            # Compute Jacobian
            if jacobian is not None:
                J = jacobian(x)
            else:
                J = self._jacobian(F, x)

            # Solve J * dx = -F (least squares)
            neg_Fx = [-v for v in Fx]
            dx = lstsq(J, neg_Fx)

            x = [xi + dxi for xi, dxi in zip(x, dx)]

        Fx = F(x)
        cost = sum(v * v for v in Fx)
        return LeastSquaresResult(
            x=x,
            converged=False,
            iterations=self.maxiter,
            cost=cost,
            residual_norm=math.sqrt(cost),
            jacobian=J,
            message="Max iterations reached",
        )


class BoundedLevenbergMarquardt:
    """
    Levenberg-Marquardt with box constraints (variable bounds).

    Minimizes ||F(x)||^2 subject to lower <= x <= upper.
    Uses projected gradient approach.
    """

    def __init__(
        self,
        tol: float = 1e-10,
        maxiter: int = 200,
        lambda_init: float = 1e-3,
        lambda_up: float = 10.0,
        lambda_down: float = 10.0,
        jacobian_method: str = "forward",
        jacobian_h: float = 1e-7,
    ):
        self.tol = tol
        self.maxiter = maxiter
        self.lambda_init = lambda_init
        self.lambda_up = lambda_up
        self.lambda_down = lambda_down
        self._jacobian = JacobianComputer(method=jacobian_method, h=jacobian_h)

    def _project(
        self,
        x: List[float],
        lower: List[float],
        upper: List[float],
    ) -> List[float]:
        """Project x onto box constraints."""
        return [max(lo, min(hi, xi)) for xi, lo, hi in zip(x, lower, upper)]

    def solve(
        self,
        F: Callable[[List[float]], List[float]],
        x0: List[float],
        bounds: Optional[Tuple[List[float], List[float]]] = None,
        jacobian: Optional[Callable[[List[float]], List[List[float]]]] = None,
    ) -> LeastSquaresResult:
        """
        Minimize ||F(x)||^2 with optional bounds.

        Args:
            F: Vector function (residuals)
            x0: Initial guess
            bounds: Optional (lower, upper) bounds for each variable
            jacobian: Optional analytical Jacobian function

        Returns:
            LeastSquaresResult with solution and convergence info
        """
        n = len(x0)

        # Default bounds: unbounded
        if bounds is None:
            lower = [-1e15] * n
            upper = [1e15] * n
        else:
            lower, upper = bounds
            if len(lower) != n or len(upper) != n:
                raise ValueError("Bounds must have same length as x0")

        # Project initial guess
        x = self._project([float(v) for v in x0], lower, upper)
        lam = self.lambda_init

        Fx = F(x)
        cost = sum(v * v for v in Fx)

        J = None
        for iteration in range(self.maxiter):
            residual_norm = math.sqrt(cost)

            if residual_norm < self.tol:
                return LeastSquaresResult(
                    x=x,
                    converged=True,
                    iterations=iteration + 1,
                    cost=cost,
                    residual_norm=residual_norm,
                    jacobian=J,
                    message="Converged",
                )

            # Compute Jacobian
            if jacobian is not None:
                J = jacobian(x)
            else:
                J = self._jacobian(F, x)

            m = len(Fx)

            # Normal equations: (J^T J + lambda*I) dx = -J^T F
            JtJ = [[sum(J[k][i] * J[k][j] for k in range(m)) for j in range(n)] for i in range(n)]
            JtF = [sum(J[k][i] * Fx[k] for k in range(m)) for i in range(n)]

            # Add Marquardt diagonal scaling for better conditioning
            for i in range(n):
                JtJ[i][i] += lam * max(JtJ[i][i], 1e-6)

            neg_JtF = [-v for v in JtF]

            # Solve for step
            try:
                dx = solve(JtJ, neg_JtF)
            except ValueError:
                dx = lstsq(JtJ, neg_JtF)

            # Project onto bounds
            x_new = self._project([xi + dxi for xi, dxi in zip(x, dx)], lower, upper)

            Fx_new = F(x_new)
            cost_new = sum(v * v for v in Fx_new)

            # Accept or reject
            if cost_new < cost:
                x = x_new
                Fx = Fx_new
                cost = cost_new
                lam /= self.lambda_down
            else:
                lam *= self.lambda_up

            # Check for stagnation
            if lam > 1e15:
                return LeastSquaresResult(
                    x=x,
                    converged=False,
                    iterations=iteration + 1,
                    cost=cost,
                    residual_norm=math.sqrt(cost),
                    jacobian=J,
                    message="Lambda overflow - problem may be ill-conditioned",
                )

        return LeastSquaresResult(
            x=x,
            converged=False,
            iterations=self.maxiter,
            cost=cost,
            residual_norm=math.sqrt(cost),
            jacobian=J,
            message="Max iterations reached",
        )


# =============================================================================
# Convenience function with simple API
# =============================================================================

def least_squares(
    F: Callable[[List[float]], List[float]],
    x0: List[float],
    bounds: Optional[Tuple[List[float], List[float]]] = None,
    tol: float = 1e-10,
    maxiter: int = 200,
    return_info: bool = True,
) -> LeastSquaresResult | List[float]:
    """
    Solve nonlinear least squares problem.

    Args:
        F: Vector function (residuals)
        x0: Initial guess
        bounds: Optional (lower, upper) bounds
        tol: Convergence tolerance
        maxiter: Maximum iterations
        return_info: If False, return just solution vector

    Returns:
        LeastSquaresResult if return_info=True, else List[float]
    """
    if bounds is not None:
        solver = BoundedLevenbergMarquardt(tol=tol, maxiter=maxiter)
        result = solver.solve(F, x0, bounds=bounds)
    else:
        solver = LevenbergMarquardt(tol=tol, maxiter=maxiter)
        result = solver.solve(F, x0)

    if return_info:
        return result
    return result.x
