"""
Fixed Income Instruments
"""

from .basis import (
    FFvSOFRBasis,
    RFRBasisSwap,
    SingleCurrencyBasisSwap,
    TenorBasisSwap,
)

from .bonds import (
    Bond,
    BondAnalytics,
    BondPriceResult,
    FixedRateBond,
    FloatingRateBond,
    InflationLinkedBond,
    ZeroCouponBond,
)

from .embedded_options import (
    CallSchedule,
    CallableBond,
    CallablePuttableBond,
    PutSchedule,
    PuttableBond,
)

from .fra import (
    FRAResult,
    FRAStrip,
    ForwardRateAgreement,
)

from .money_market import (
    CertificateOfDeposit,
    CommercialPaper,
    Deposit,
    DepositResult,
    DepositType,
    Repo,
    TreasuryBill,
)

from .ois import (
    OISIndex,
    OISResult,
    _tenor_to_years,
    build_ois_curve,
)

from .swaps import (
    BasisSwap,
    CashFlow,
    FixedLeg,
    FloatingLeg,
    InterestRateSwap,
    OvernightIndexSwap,
    SwapLeg,
    SwapLegResult,
    SwapResult,
    SwapType,
)

__all__ = [
    "BasisSwap",
    "Bond",
    "BondAnalytics",
    "BondPriceResult",
    "CallSchedule",
    "CallableBond",
    "CallablePuttableBond",
    "CashFlow",
    "CertificateOfDeposit",
    "CommercialPaper",
    "Deposit",
    "DepositResult",
    "DepositType",
    "FFvSOFRBasis",
    "FRAResult",
    "FRAStrip",
    "FixedLeg",
    "FixedRateBond",
    "FloatingLeg",
    "FloatingRateBond",
    "ForwardRateAgreement",
    "InflationLinkedBond",
    "InterestRateSwap",
    "OISIndex",
    "OISResult",
    "OvernightIndexSwap",
    "PutSchedule",
    "PuttableBond",
    "RFRBasisSwap",
    "Repo",
    "SingleCurrencyBasisSwap",
    "SwapLeg",
    "SwapLegResult",
    "SwapResult",
    "SwapType",
    "TenorBasisSwap",
    "TreasuryBill",
    "ZeroCouponBond",
    "build_ois_curve",
]
