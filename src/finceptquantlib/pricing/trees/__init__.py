"""
Tree-Based Pricing Methods

Lattice methods for derivative pricing:
- Binomial trees (CRR, JR, Tian)
- Trinomial trees
- American option pricing
"""

from finceptquantlib.pricing.trees.binomial import (
    TreeMethod,
    BinomialTreeResult,
    binomial_params,
    binomial_european,
    binomial_american,
    binomial_bermudan,
    binomial_barrier,
)

__all__ = [
    "TreeMethod",
    "BinomialTreeResult",
    "binomial_params",
    "binomial_european",
    "binomial_american",
    "binomial_bermudan",
    "binomial_barrier",
]
