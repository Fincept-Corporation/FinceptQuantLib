"""
1D Root Finding

Delegates to finceptquantlib.numerical.solvers.root1d for implementations.

Provides:
- brent: Brent's method (recommended default)
- newton: Newton-Raphson with optional derivative
- bisect: Simple bisection
- secant: Secant method
- ridder: Ridder's method
"""

from finceptquantlib.numerical.solvers.root1d import (
    RootResult,
    brent,
    newton,
    bisect,
    secant,
    ridder,
    find_root,
    find_root_newton,
)

__all__ = [
    "RootResult",
    "brent",
    "newton",
    "bisect",
    "secant",
    "ridder",
    "find_root",
    "find_root_newton",
]
