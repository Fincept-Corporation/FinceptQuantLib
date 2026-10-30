"""
General Optimization

Unconstrained and constrained optimization:
- nelder_mead: Derivative-free simplex method
- bfgs: Quasi-Newton with line search
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional
import math

from finceptquantlib.numerical.differentiation import finite_difference_gradient


@dataclass
class OptimizeResult:
    """Result of optimization."""
    x: List[float]
    fun: float  # Function value at solution
    converged: bool
    iterations: int
    message: str = ""


def nelder_mead(
    f: Callable[[List[float]], float],
    x0: List[float],
    tol: float = 1e-8,
    maxiter: int = 1000,
    alpha: float = 1.0,
    gamma: float = 2.0,
    rho: float = 0.5,
    sigma: float = 0.5,
) -> OptimizeResult:
    """
    Nelder-Mead simplex algorithm for unconstrained optimization.

    Derivative-free method suitable for noisy or non-smooth functions.

    Args:
        f: Objective function to minimize
        x0: Initial guess
        tol: Convergence tolerance
        maxiter: Maximum iterations
        alpha: Reflection coefficient
        gamma: Expansion coefficient
        rho: Contraction coefficient
        sigma: Shrink coefficient

    Returns:
        OptimizeResult with solution and convergence info
    """
    n = len(x0)

    # Initialize simplex: n+1 vertices
    simplex = [[float(v) for v in x0]]
    for i in range(n):
        vertex = [float(v) for v in x0]
        vertex[i] += 0.00025 if vertex[i] == 0.0 else 0.05 * vertex[i]
        simplex.append(vertex)

    # Evaluate at all vertices
    fvals = [f(v) for v in simplex]

    for iteration in range(maxiter):
        # Sort by function value
        order = sorted(range(n + 1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        # Check convergence
        frange = fvals[-1] - fvals[0]
        if frange < tol:
            return OptimizeResult(
                x=simplex[0],
                fun=fvals[0],
                converged=True,
                iterations=iteration + 1,
                message="Converged",
            )

        # Centroid of all but worst
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]

        # Reflection
        xr = [centroid[j] + alpha * (centroid[j] - simplex[-1][j]) for j in range(n)]
        fr = f(xr)

        if fvals[0] <= fr < fvals[-2]:
            # Accept reflection
            simplex[-1] = xr
            fvals[-1] = fr
            continue

        if fr < fvals[0]:
            # Try expansion
            xe = [centroid[j] + gamma * (xr[j] - centroid[j]) for j in range(n)]
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
            continue

        # Contraction
        if fr < fvals[-1]:
            # Outside contraction
            xc = [centroid[j] + rho * (xr[j] - centroid[j]) for j in range(n)]
        else:
            # Inside contraction
            xc = [centroid[j] + rho * (simplex[-1][j] - centroid[j]) for j in range(n)]

        fc = f(xc)
        if fc < min(fr, fvals[-1]):
            simplex[-1] = xc
            fvals[-1] = fc
            continue

        # Shrink
        for i in range(1, n + 1):
            simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j])
                          for j in range(n)]
            fvals[i] = f(simplex[i])

    return OptimizeResult(
        x=simplex[0],
        fun=fvals[0],
        converged=False,
        iterations=maxiter,
        message="Max iterations reached",
    )


def bfgs(
    f: Callable[[List[float]], float],
    x0: List[float],
    grad: Optional[Callable[[List[float]], List[float]]] = None,
    tol: float = 1e-8,
    maxiter: int = 200,
    h: float = 1e-7,
) -> OptimizeResult:
    """
    BFGS quasi-Newton method for unconstrained optimization.

    Uses approximate Hessian inverse updated via BFGS formula.

    Args:
        f: Objective function to minimize
        x0: Initial guess
        grad: Gradient function (uses finite differences if None)
        tol: Convergence tolerance on gradient norm
        maxiter: Maximum iterations
        h: Step size for finite difference gradient

    Returns:
        OptimizeResult with solution and convergence info
    """
    n = len(x0)
    x = [float(v) for v in x0]

    # Initial Hessian inverse approximation (identity)
    H = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    # Compute initial gradient
    if grad is not None:
        g = grad(x)
    else:
        g = finite_difference_gradient(f, x, h)

    for iteration in range(maxiter):
        # Check convergence
        gnorm = math.sqrt(sum(gi * gi for gi in g))
        if gnorm < tol:
            return OptimizeResult(
                x=x,
                fun=f(x),
                converged=True,
                iterations=iteration + 1,
                message="Converged",
            )

        # Search direction: p = -H @ g
        p = [-sum(H[i][j] * g[j] for j in range(n)) for i in range(n)]

        # Line search (backtracking)
        alpha = 1.0
        fx = f(x)
        c1 = 1e-4
        rho_ls = 0.5

        for _ in range(20):
            x_new = [x[i] + alpha * p[i] for i in range(n)]
            fx_new = f(x_new)
            if fx_new <= fx + c1 * alpha * sum(g[i] * p[i] for i in range(n)):
                break
            alpha *= rho_ls
        else:
            # Line search failed
            return OptimizeResult(
                x=x,
                fun=fx,
                converged=False,
                iterations=iteration + 1,
                message="Line search failed",
            )

        # Update x
        s = [alpha * p[i] for i in range(n)]
        x = [x[i] + s[i] for i in range(n)]

        # New gradient
        if grad is not None:
            g_new = grad(x)
        else:
            g_new = finite_difference_gradient(f, x, h)

        y = [g_new[i] - g[i] for i in range(n)]
        g = g_new

        # BFGS update
        sy = sum(s[i] * y[i] for i in range(n))
        if sy > 1e-15:
            # Hy = H @ y
            Hy = [sum(H[i][j] * y[j] for j in range(n)) for i in range(n)]
            yHy = sum(y[i] * Hy[i] for i in range(n))

            # H = H + (sy + yHy)/sy^2 * s@s^T - (Hy@s^T + s@Hy^T)/sy
            coef1 = (sy + yHy) / (sy * sy)
            coef2 = 1.0 / sy

            for i in range(n):
                for j in range(n):
                    H[i][j] += coef1 * s[i] * s[j] - coef2 * (Hy[i] * s[j] + s[i] * Hy[j])

    return OptimizeResult(
        x=x,
        fun=f(x),
        converged=False,
        iterations=maxiter,
        message="Max iterations reached",
    )
