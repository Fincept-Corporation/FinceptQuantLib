"""
Stochastic Calculus Theory

Theoretical tools for stochastic calculus:
- Ito's lemma and calculus rules
- Girsanov theorem and measure change
- Martingale theory
"""

from finceptquantlib.stochastic.theory.ito import (
    ItoProcess,
    ito_lemma_1d,
    ito_product_rule,
    quadratic_variation,
    covariation,
    stochastic_integral_approx,
    ito_to_stratonovich,
    stratonovich_to_ito,
)

from finceptquantlib.stochastic.theory.girsanov import (
    MeasureChange,
    risk_neutral_drift,
    esscher_transform,
    simulate_under_Q,
    change_of_numeraire,
)

from finceptquantlib.stochastic.theory.martingale import (
    is_martingale_test,
    optional_stopping_value,
    doob_maximal_inequality,
    quadratic_variation_martingale,
    predictable_representation,
    local_martingale_test,
)

__all__ = [
    # Ito
    "ItoProcess",
    "ito_lemma_1d",
    "ito_product_rule",
    "quadratic_variation",
    "covariation",
    "stochastic_integral_approx",
    "ito_to_stratonovich",
    "stratonovich_to_ito",
    # Girsanov
    "MeasureChange",
    "risk_neutral_drift",
    "esscher_transform",
    "simulate_under_Q",
    "change_of_numeraire",
    # Martingale
    "is_martingale_test",
    "optional_stopping_value",
    "doob_maximal_inequality",
    "quadratic_variation_martingale",
    "predictable_representation",
    "local_martingale_test",
]
