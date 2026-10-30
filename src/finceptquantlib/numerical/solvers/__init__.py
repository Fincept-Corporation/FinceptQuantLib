"""
Numerical Solvers

Root finding, optimization, and least squares:
- root1d: 1D root finding (brent, newton, bisect, secant, ridder)
- rootnd: Multi-dimensional root finding (Newton-Raphson, Broyden)
- leastsq: Nonlinear least squares (Levenberg-Marquardt, Gauss-Newton)
- optimize: General optimization (Nelder-Mead, BFGS)
"""

from finceptquantlib.numerical.solvers.root1d import (
    brent,
    newton,
    bisect,
    secant,
    ridder,
    RootResult,
    find_root,
    find_root_newton,
)

from finceptquantlib.numerical.solvers.rootnd import (
    NewtonRaphson,
    Broyden,
    RootResultND,
    find_root_nd,
)

from finceptquantlib.numerical.solvers.leastsq import (
    LevenbergMarquardt,
    BoundedLevenbergMarquardt,
    GaussNewton,
    LeastSquaresResult,
    least_squares,
)

from finceptquantlib.numerical.solvers.optimize import (
    nelder_mead,
    bfgs,
    OptimizeResult,
)

__all__ = [
    # 1D Root Finding
    "brent",
    "newton",
    "bisect",
    "secant",
    "ridder",
    "RootResult",
    "find_root",
    "find_root_newton",
    # Multi-dim Root Finding
    "NewtonRaphson",
    "Broyden",
    "RootResultND",
    "find_root_nd",
    # Least Squares
    "LevenbergMarquardt",
    "BoundedLevenbergMarquardt",
    "GaussNewton",
    "LeastSquaresResult",
    "least_squares",
    # Optimization
    "nelder_mead",
    "bfgs",
    "OptimizeResult",
]
