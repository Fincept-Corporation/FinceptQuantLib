"""
Nonlinear Least Squares

Delegates to finceptquantlib.numerical.solvers.leastsq for implementations.

Provides:
- LevenbergMarquardt: Trust-region method (recommended)
- BoundedLevenbergMarquardt: With variable bounds
- GaussNewton: Basic least squares
"""

from finceptquantlib.numerical.solvers.leastsq import (
    LeastSquaresResult,
    LevenbergMarquardt,
    BoundedLevenbergMarquardt,
    GaussNewton,
    least_squares,
)

__all__ = [
    "LeastSquaresResult",
    "LevenbergMarquardt",
    "BoundedLevenbergMarquardt",
    "GaussNewton",
    "least_squares",
]
