"""
Multi-dimensional Root Finding

Delegates to finceptquantlib.numerical.solvers.rootnd for implementations.

Provides:
- NewtonRaphson: Standard Newton's method
- Broyden: Broyden's quasi-Newton method
- find_root_nd: Convenience function
"""

from finceptquantlib.numerical.solvers.rootnd import (
    RootResultND,
    NewtonRaphson,
    Broyden,
    find_root_nd,
)

__all__ = [
    "RootResultND",
    "NewtonRaphson",
    "Broyden",
    "find_root_nd",
]
