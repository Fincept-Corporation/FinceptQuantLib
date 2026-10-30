"""
ODE/PDE Solvers

Numerical methods for differential equations:
- IVP: Initial value problems (Euler, RK4, RK45)
- BVP: Boundary value problems
- PDE: Finite difference methods
"""

from finceptquantlib.numerical.ode.ivp import (
    euler,
    rk4,
    rk45,
    solve_ivp,
    IVPResult,
)

__all__ = [
    "euler",
    "rk4",
    "rk45",
    "solve_ivp",
    "IVPResult",
]
