"""
Copula Models

Dependence structures for multivariate modeling:

Elliptical Copulas:
- GaussianCopula: Normal copula, zero tail dependence
- StudentTCopula: Heavy tails, positive tail dependence

Archimedean Copulas:
- ClaytonCopula: Lower tail dependence
- GumbelCopula: Upper tail dependence
- FrankCopula: No tail dependence, allows negative dependence
- JoeCopula: Upper tail dependence

Example:
    >>> from . import GaussianCopula, ClaytonCopula
    >>>
    >>> # Gaussian copula with correlation 0.5
    >>> gc = GaussianCopula([[1.0, 0.5], [0.5, 1.0]])
    >>> samples = gc.sample(1000)
    >>> gc.tail_dependence()  # (0, 0)
    (0.0, 0.0)
    >>>
    >>> # Clayton copula with theta=2
    >>> cc = ClaytonCopula(theta=2.0)
    >>> cc.tail_dependence()  # (0.707, 0) - lower tail dependence
    (0.707107, 0.0)
"""

from .elliptical import (
    GaussianCopula,
    StudentTCopula,
)

from .archimedean import (
    ClaytonCopula,
    GumbelCopula,
    FrankCopula,
    JoeCopula,
)

__all__ = [
    # Elliptical
    "GaussianCopula",
    "StudentTCopula",
    # Archimedean
    "ClaytonCopula",
    "GumbelCopula",
    "FrankCopula",
    "JoeCopula",
]
