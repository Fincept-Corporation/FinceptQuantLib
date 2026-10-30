"""
Jacobian Computation

Delegates to finceptquantlib.numerical.differentiation for implementations.

Provides:
- JacobianComputer: Finite difference Jacobian computation
- ParallelJacobianComputer: Threaded parallel Jacobian computation
"""

from finceptquantlib.numerical.differentiation.finite_diff import (
    finite_difference_jacobian,
    finite_difference_gradient,
    finite_difference_derivative,
    finite_difference_hessian,
    JacobianComputer,
    parallel_jacobian,
    parallel_gradient,
    ParallelJacobianComputer,
)

__all__ = [
    "finite_difference_jacobian",
    "finite_difference_gradient",
    "finite_difference_derivative",
    "finite_difference_hessian",
    "JacobianComputer",
    "parallel_jacobian",
    "parallel_gradient",
    "ParallelJacobianComputer",
]
