"""
FX and Credit Instruments
"""

from .fx import *
from .credit import *

__all__ = [
    # FX Rate Matrix
    "FXQuote",
    "FXRateMatrix",
    "QuoteConvention",
    "parse_pair",
    "invert_pair",
    "get_pip_size",
    "rate_to_pips",
    "pips_to_rate",
    # FX Forward Engine
    "FXForwardQuote",
    "FXForwardCalculator",
    "NDFQuote",
    "calculate_forward_points",
    "calculate_outright_forward",
    "implied_deposit_rate",
    # FX Forwards
    "FXSpot",
    "FXForward",
    "FXSwap",
    "FXForwardResult",
    "forward_point_curve",
    # FX Options
    "FXOption",
    "FXOptionResult",
    "BarrierOption",
    "DigitalOption",
    "BarrierType",
    "DigitalType",
    "garman_kohlhagen_call",
    "garman_kohlhagen_put",
    "garman_kohlhagen_greeks",
    # FX Cross-currency
    "CrossCurrencySwap",
    "XCCYResult",
    # Credit CDS
    "CreditDefaultSwap",
    "CDSResult",
    "CDSBootstrap",
    "CDSType",
    "CDSIndex",
    # Credit Hazard curves
    "HazardRateCurve",
    "SurvivalCurve",
    "ConstantHazardRate",
    "PiecewiseHazardRate",
]
