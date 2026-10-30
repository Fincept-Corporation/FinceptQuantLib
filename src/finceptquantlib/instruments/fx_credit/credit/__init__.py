"""
Credit Derivatives Module

Provides pricing and analytics for credit instruments:
- Credit Default Swaps (CDS): Single-name and index
- CDS Options (CDS Swaptions)
- Credit-linked instruments
- Hazard rate models
"""

from .cds import (
    CreditDefaultSwap,
    CDSResult,
    CDSBootstrap,
    CDSType,
    CDSIndex,
)
from .hazard import (
    HazardRateCurve,
    SurvivalCurve,
    ConstantHazardRate,
    PiecewiseHazardRate,
)

__all__ = [
    # CDS
    "CreditDefaultSwap",
    "CDSResult",
    "CDSBootstrap",
    "CDSType",
    "CDSIndex",
    # Hazard curves
    "HazardRateCurve",
    "SurvivalCurve",
    "ConstantHazardRate",
    "PiecewiseHazardRate",
]
