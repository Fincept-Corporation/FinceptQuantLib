"""
Exotic Options Module

Provides pricing and analytics for exotic options:
- Barrier Options: Single/double barriers, discrete monitoring
- Asian Options: Arithmetic/geometric average, price/strike
- Lookback Options: Fixed/floating strike
- Other Exotics: Chooser, compound, exchange options
"""

from .barriers import (
    BarrierOption,
    BarrierType,
    DoubleBarrierOption,
    DiscreteBarrierOption,
    down_and_out_call,
    up_and_out_call,
    down_and_in_put,
)

from .asians import (
    AsianOption,
    AveragingType,
    GeometricAsianOption,
    ArithmeticAsianOption,
    AverageStrike,
    geometric_asian_call,
    arithmetic_asian_call,
    asian_option_mc,
)

from .lookbacks import (
    LookbackOption,
    FloatingStrikeLookback,
    FixedStrikeLookback,
    floating_lookback_call,
    floating_lookback_put,
    fixed_lookback_call,
    fixed_lookback_put,
)

from .other import (
    ChooserOption,
    CompoundOption,
    ExchangeOption,
    ForwardStartOption,
    chooser_option,
    exchange_option,
    forward_start_call,
    compound_call_on_call,
)

__all__ = [
    # Barriers
    "BarrierOption",
    "BarrierType",
    "DoubleBarrierOption",
    "DiscreteBarrierOption",
    # Asians
    "AsianOption",
    "AveragingType",
    "GeometricAsianOption",
    "ArithmeticAsianOption",
    # Lookbacks
    "LookbackOption",
    "FloatingStrikeLookback",
    "FixedStrikeLookback",
    # Other
    "ChooserOption",
    "CompoundOption",
    "ExchangeOption",
    "ForwardStartOption",
    # Additional Barriers
    "down_and_out_call",
    "up_and_out_call",
    "down_and_in_put",
    # Additional Asians
    "AverageStrike",
    "geometric_asian_call",
    "arithmetic_asian_call",
    "asian_option_mc",
    # Additional Lookbacks
    "floating_lookback_call",
    "floating_lookback_put",
    "fixed_lookback_call",
    "fixed_lookback_put",
    # Additional Other
    "chooser_option",
    "exchange_option",
    "forward_start_call",
    "compound_call_on_call",
]
