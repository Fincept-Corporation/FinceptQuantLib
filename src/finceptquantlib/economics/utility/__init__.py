"""
Utility Theory

Economic utility functions and related concepts:

Functions:
- CRRA: Constant Relative Risk Aversion
- CARA: Constant Absolute Risk Aversion
- QuadraticUtility: Mean-variance preferences
- LogUtility: Special case of CRRA (γ=1)
- ProspectTheoryValue: Kahneman-Tversky value function
- HyperbolicAbsoluteRiskAversion: General HARA class

Tools:
- expected_utility: E[u(X)]
- certainty_equivalent_approximation
- stochastic_dominance_first: FSD test
- stochastic_dominance_second: SSD test

Example:
    >>> from finceptquantlib.economics.utility import CRRA, expected_utility
    >>>
    >>> u = CRRA(gamma=2)
    >>> u(100)  # Utility of 100
    0.99
    >>> u.certainty_equivalent([90, 110], [0.5, 0.5])  # CE of 50-50 lottery
    99.0
    >>> u.risk_premium([90, 110], [0.5, 0.5])  # Risk premium
    1.0
"""

from finceptquantlib.economics.utility.functions import (
    UtilityFunction,
    CRRA,
    CARA,
    QuadraticUtility,
    LogUtility,
    ProspectTheoryValue,
    HyperbolicAbsoluteRiskAversion,
    expected_utility,
    certainty_equivalent_approximation,
    stochastic_dominance_first,
    stochastic_dominance_second,
)

__all__ = [
    "UtilityFunction",
    "CRRA",
    "CARA",
    "QuadraticUtility",
    "LogUtility",
    "ProspectTheoryValue",
    "HyperbolicAbsoluteRiskAversion",
    "expected_utility",
    "certainty_equivalent_approximation",
    "stochastic_dominance_first",
    "stochastic_dominance_second",
]
