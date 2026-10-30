"""
Financial Instruments Module

Reorganized into logical groups:
- fixed_income: Bonds, swaps, money market, securitization
- rates_derivatives: Advanced swaps, options, futures, inflation swaps
- equity_alternatives: Equity derivatives, commodities, convertibles, exotics
- fx_credit: FX instruments and credit derivatives
"""

from .fixed_income import *
from .rates_derivatives import *
from .equity_alternatives import *
from .fx_credit import *

# Combine __all__ from all submodules
from .fixed_income import __all__ as _fixed_income_all
from .rates_derivatives import __all__ as _rates_deriv_all
from .equity_alternatives import __all__ as _equity_alt_all
from .fx_credit import __all__ as _fx_credit_all

# Equity barrier options share a class name with FX barrier options
# (fx_credit wins the plain name); expose the equity variant explicitly.
from .equity_alternatives.exotics.barriers import BarrierOption as ExoticBarrier

__all__ = list(dict.fromkeys(
    list(_fixed_income_all) + list(_rates_deriv_all) + list(_equity_alt_all)
    + list(_fx_credit_all) + ["ExoticBarrier"]
))
