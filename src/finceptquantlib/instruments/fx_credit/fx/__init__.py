"""
FX Instruments Module

Provides pricing and analytics for foreign exchange instruments:
- FX Rate Matrix with automatic triangulation
- FX Spot, Forwards, and Outrights
- FX Swaps
- FX Options (vanilla, barrier, digital)
- Cross-Currency Swaps
- NDF (Non-Deliverable Forwards)
"""

from .forwards import (
    FXSpot,
    FXForward,
    FXSwap,
    FXForwardResult,
    forward_point_curve,
)
from .options import (
    FXOption,
    FXOptionResult,
    BarrierOption,
    DigitalOption,
    BarrierType,
    DigitalType,
    garman_kohlhagen_call,
    garman_kohlhagen_put,
    garman_kohlhagen_greeks,
)
from .xccy import (
    CrossCurrencySwap,
    XCCYResult,
)
from .rates import (
    FXQuote,
    FXRateMatrix,
    QuoteConvention,
    parse_pair,
    invert_pair,
    get_pip_size,
    rate_to_pips,
    pips_to_rate,
)
from .forward_engine import (
    FXForwardQuote,
    FXForwardCalculator,
    NDFQuote,
    calculate_forward_points,
    calculate_outright_forward,
    implied_deposit_rate,
)

__all__ = [
    # Rate Matrix
    "FXQuote",
    "FXRateMatrix",
    "QuoteConvention",
    "parse_pair",
    "invert_pair",
    "get_pip_size",
    "rate_to_pips",
    "pips_to_rate",
    # Forward Engine
    "FXForwardQuote",
    "FXForwardCalculator",
    "NDFQuote",
    "calculate_forward_points",
    "calculate_outright_forward",
    "implied_deposit_rate",
    # Forwards (instrument)
    "FXSpot",
    "FXForward",
    "FXSwap",
    "FXForwardResult",
    # Options
    "FXOption",
    "FXOptionResult",
    "BarrierOption",
    "DigitalOption",
    "BarrierType",
    # Cross-currency
    "CrossCurrencySwap",
    "XCCYResult",
    # Additional
    "forward_point_curve",
    "DigitalType",
    "garman_kohlhagen_call",
    "garman_kohlhagen_put",
    "garman_kohlhagen_greeks",
]
