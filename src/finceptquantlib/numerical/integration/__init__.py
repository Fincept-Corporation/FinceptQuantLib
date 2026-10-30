"""
Numerical Integration

Quadrature methods for computing definite integrals:
- Simpson's rule
- Trapezoidal rule
- Gauss-Legendre quadrature
- Adaptive quadrature
- Monte Carlo integration
"""

from finceptquantlib.numerical.integration.quadrature import (
    simpson,
    trapezoid,
    gauss_legendre,
    adaptive_quad,
    romberg,
)

from finceptquantlib.numerical.integration.monte_carlo import (
    monte_carlo_integrate,
    importance_sampling,
    stratified_sampling,
)

__all__ = [
    # Quadrature
    "simpson",
    "trapezoid",
    "gauss_legendre",
    "adaptive_quad",
    "romberg",
    # Monte Carlo
    "monte_carlo_integrate",
    "importance_sampling",
    "stratified_sampling",
]
