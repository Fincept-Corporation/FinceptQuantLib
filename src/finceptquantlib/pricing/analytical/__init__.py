"""
Analytical Pricing Formulas

Closed-form option pricing:
- Black-Scholes-Merton (equity options)
- Black-76 (futures, caps/floors, swaptions)
- Bachelier (normal model for negative rates)
- Kirk (spread options)
"""

from finceptquantlib.core.utilities.distributions import norm_cdf, norm_pdf, norm_ppf
from finceptquantlib.pricing.analytical.black_scholes import (
    # Core functions
    d1, d2,
    bs_call, bs_put, bs_price,
    # Greeks
    bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho,
    bs_vanna, bs_volga,
    bs_greeks, BSGreeks,
    # Implied vol
    bs_implied_vol,
    # Digital options
    bs_digital_call, bs_digital_put,
    bs_asset_or_nothing_call, bs_asset_or_nothing_put,
    BlackScholesResult,
    black_scholes_price,
    black_scholes_greeks,
    black_scholes_implied_volatility,
)

from finceptquantlib.pricing.analytical.black76 import (
    # Core
    black76_d1, black76_d2,
    black76_call, black76_put, black76_price,
    # Greeks
    black76_delta, black76_gamma, black76_vega, black76_theta,
    black76_greeks, Black76Greeks,
    # Implied vol
    black76_implied_vol,
    # Interest rate
    caplet_price, floorlet_price, swaption_price,
    Black76Result,
    black76_option_price,
    black76_option_greeks,
    black76_option_implied_volatility,
)

from finceptquantlib.pricing.analytical.bachelier import (
    # Core
    bachelier_call, bachelier_put, bachelier_price,
    # Greeks
    bachelier_delta, bachelier_gamma, bachelier_vega,
    bachelier_greeks, BachelierGreeks,
    # Implied vol
    bachelier_implied_vol,
    # Conversions
    lognormal_to_normal_vol, normal_to_lognormal_vol,
    # Shifted
    shifted_lognormal_call, shifted_lognormal_put,
    BachelierResult,
    bachelier_option_price,
    bachelier_option_greeks,
    bachelier_option_implied_volatility,
)

from finceptquantlib.pricing.analytical.kirk import (
    # Spread options
    kirk_spread_call, kirk_spread_put, kirk_spread_price,
    kirk_delta1, kirk_delta2,
    kirk_greeks, KirkGreeks,
    # Exchange options
    margrabe_call, margrabe_put,
    # Basket
    basket_call_levy,
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
    # Additional
    "BlackScholesResult",
    "black_scholes_price",
    "black_scholes_greeks",
    "black_scholes_implied_volatility",
    "Black76Result",
    "black76_option_price",
    "black76_option_greeks",
    "black76_option_implied_volatility",
    "BachelierResult",
    "bachelier_option_price",
    "bachelier_option_greeks",
    "bachelier_option_implied_volatility",
]
