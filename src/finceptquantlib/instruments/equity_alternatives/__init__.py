"""
Equity Alternatives Instruments
"""

from .alternative import (
    CatastropheBond,
    WeatherDerivative,
    WeatherIndex,
    WeatherMeasurement,
    degree_days,
)

from .commodity import (
    CommodityFuture,
    CommodityIndex,
    CommodityOption,
    CommoditySwap,
    CommodityType,
    SpreadOption,
)

from .convertible import (
    CallProvision,
    ConversionTerms,
    ConversionTrigger,
    ConvertibleBond,
    PutProvision,
)

from .equity import (
    EquitySwap,
    TotalReturnSwap,
    VarianceSwap,
    VolatilitySwap,
    calculate_realized_variance,
    calculate_realized_volatility,
)

from .securitization import (
    AssetBackedSecurity,
    CDO,
    CLO,
    MBSType,
    MortgageBackedSecurity,
    MortgagePool,
    PrepaymentModel,
    Tranche,
)

from .exotics import *

__all__ = [
    "AssetBackedSecurity",
    "CDO",
    "CLO",
    "CallProvision",
    "CatastropheBond",
    "CommodityFuture",
    "CommodityIndex",
    "CommodityOption",
    "CommoditySwap",
    "CommodityType",
    "ConversionTerms",
    "ConversionTrigger",
    "ConvertibleBond",
    "EquitySwap",
    "MBSType",
    "MortgageBackedSecurity",
    "MortgagePool",
    "PrepaymentModel",
    "PutProvision",
    "SpreadOption",
    "TotalReturnSwap",
    "Tranche",
    "VarianceSwap",
    "VolatilitySwap",
    "WeatherDerivative",
    "WeatherIndex",
    "WeatherMeasurement",
    "calculate_realized_variance",
    "calculate_realized_volatility",
    "degree_days",
    # Exotics - Barriers
    "BarrierOption",
    "BarrierType",
    "DoubleBarrierOption",
    "DiscreteBarrierOption",
    "down_and_out_call",
    "up_and_out_call",
    "down_and_in_put",
    # Exotics - Asians
    "AsianOption",
    "AveragingType",
    "GeometricAsianOption",
    "ArithmeticAsianOption",
    "AverageStrike",
    "geometric_asian_call",
    "arithmetic_asian_call",
    "asian_option_mc",
    # Exotics - Lookbacks
    "LookbackOption",
    "FloatingStrikeLookback",
    "FixedStrikeLookback",
    "floating_lookback_call",
    "floating_lookback_put",
    "fixed_lookback_call",
    "fixed_lookback_put",
    # Exotics - Other
    "ChooserOption",
    "CompoundOption",
    "ExchangeOption",
    "ForwardStartOption",
    "chooser_option",
    "exchange_option",
    "forward_start_call",
    "compound_call_on_call",
]
