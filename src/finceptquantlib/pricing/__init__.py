"""
Derivative Pricing Module

Comprehensive pricing methods:

- analytical/: Closed-form solutions (Black-Scholes, Black-76, Bachelier, Kirk)
- trees/: Lattice methods (binomial, trinomial)
- monte_carlo/: Simulation-based pricing
- pde/: Finite difference methods
- fourier/: Transform-based pricing

Example:
    >>> from finceptquantlib.pricing import bs_call, bs_greeks, binomial_american
    >>> from finceptquantlib.core.foundations.types import OptionType
    >>>
    >>> # Black-Scholes call
    >>> price = bs_call(S=100, K=100, r=0.05, q=0, sigma=0.2, T=1.0)
    >>>
    >>> # Greeks
    >>> greeks = bs_greeks(100, 100, 0.05, 0, 0.2, 1.0, OptionType.CALL)
    >>>
    >>> # American put via binomial tree
    >>> result = binomial_american(100, 100, 0.05, 0, 0.2, 1.0, 100, OptionType.PUT)
"""

# Analytical formulas
from finceptquantlib.pricing.analytical import (
    # Black-Scholes
    d1, d2,
    bs_call, bs_put, bs_price,
    bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho,
    bs_vanna, bs_volga,
    bs_greeks, BSGreeks,
    bs_implied_vol,
    bs_digital_call, bs_digital_put,
    bs_asset_or_nothing_call, bs_asset_or_nothing_put,
    norm_cdf, norm_pdf, norm_ppf,
    # Black-76
    black76_d1, black76_d2,
    black76_call, black76_put, black76_price,
    black76_delta, black76_gamma, black76_vega, black76_theta,
    black76_greeks, Black76Greeks,
    black76_implied_vol,
    caplet_price, floorlet_price, swaption_price,
    # Bachelier
    bachelier_call, bachelier_put, bachelier_price,
    bachelier_delta, bachelier_gamma, bachelier_vega,
    bachelier_greeks, BachelierGreeks,
    bachelier_implied_vol,
    lognormal_to_normal_vol, normal_to_lognormal_vol,
    shifted_lognormal_call, shifted_lognormal_put,
    # Kirk/Spread
    kirk_spread_call, kirk_spread_put, kirk_spread_price,
    kirk_delta1, kirk_delta2,
    kirk_greeks, KirkGreeks,
    margrabe_call, margrabe_put,
    basket_call_levy,
)

# Tree methods
from finceptquantlib.pricing.trees import (
    TreeMethod,
    BinomialTreeResult,
    binomial_params,
    binomial_european,
    binomial_american,
    binomial_bermudan,
    binomial_barrier,
)

__all__ = [
    # Black-Scholes
    "d1", "d2",
    "bs_call", "bs_put", "bs_price",
    "bs_delta", "bs_gamma", "bs_vega", "bs_theta", "bs_rho",
    "bs_vanna", "bs_volga",
    "bs_greeks", "BSGreeks",
    "bs_implied_vol",
    "bs_digital_call", "bs_digital_put",
    "bs_asset_or_nothing_call", "bs_asset_or_nothing_put",
    "norm_cdf", "norm_pdf", "norm_ppf",
    # Black-76
    "black76_d1", "black76_d2",
    "black76_call", "black76_put", "black76_price",
    "black76_delta", "black76_gamma", "black76_vega", "black76_theta",
    "black76_greeks", "Black76Greeks",
    "black76_implied_vol",
    "caplet_price", "floorlet_price", "swaption_price",
    # Bachelier
    "bachelier_call", "bachelier_put", "bachelier_price",
    "bachelier_delta", "bachelier_gamma", "bachelier_vega",
    "bachelier_greeks", "BachelierGreeks",
    "bachelier_implied_vol",
    "lognormal_to_normal_vol", "normal_to_lognormal_vol",
    "shifted_lognormal_call", "shifted_lognormal_put",
    # Kirk/Spread
    "kirk_spread_call", "kirk_spread_put", "kirk_spread_price",
    "kirk_delta1", "kirk_delta2",
    "kirk_greeks", "KirkGreeks",
    "margrabe_call", "margrabe_put",
    "basket_call_levy",
    # Trees
    "TreeMethod",
    "BinomialTreeResult",
    "binomial_params",
    "binomial_european",
    "binomial_american",
    "binomial_bermudan",
    "binomial_barrier",
]
