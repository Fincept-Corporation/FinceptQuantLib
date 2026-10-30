"""
Numerical Differentiation

Finite difference methods for computing derivatives:
- Jacobian matrix computation
- Gradient computation
- Hessian computation
- Parallel implementations
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
