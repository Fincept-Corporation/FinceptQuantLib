"""
Volatility Module

Provides volatility modeling and surfaces:
- Interest rate volatility surfaces
- SABR model calibration
- Volatility cube construction
- Local volatility (Dupire, SVI, SSVI)
- Implied volatility calculations
"""

from finceptquantlib.volatility.surface import (
    VolatilitySurface,
    FlatVolatility,
    TermStructureVolatility,
    VolatilitySurfaceGrid,
    VolatilityCube,
    VolatilityPoint,
    build_surface_from_points,
)
from finceptquantlib.volatility.sabr import (
    SABRParameters,
    SABRModel,
    SABRSmileDynamics,
    calibrate_sabr,
    sabr_implied_vol,
    sabr_normal_vol,
    sabr_density,
)
# Local vol base classes from volatility module
from finceptquantlib.volatility.local_vol import (
    LocalVolatility,
    ConstantLocalVol,
    ParametricLocalVol,
    implied_to_local_vol,
)
# Comprehensive local vol from models module (preferred)
from finceptquantlib.models.local_vol import (
    DupireLocalVol,
    LocalVolSurface,
    SVI,
    SSVI,
    create_flat_local_vol,
    create_svi_local_vol,
)

__all__ = [
    # Volatility surfaces
    "VolatilitySurface",
    "FlatVolatility",
    "TermStructureVolatility",
    "VolatilitySurfaceGrid",
    "VolatilityCube",
    "VolatilityPoint",
    "build_surface_from_points",
    # SABR
    "SABRParameters",
    "SABRModel",
    "SABRSmileDynamics",
    "calibrate_sabr",
    "sabr_implied_vol",
    "sabr_normal_vol",
    "sabr_density",
    # Local vol base classes
    "LocalVolatility",
    "ConstantLocalVol",
    "ParametricLocalVol",
    "implied_to_local_vol",
    # Local vol models (from models module)
    "DupireLocalVol",
    "LocalVolSurface",
    "SVI",
    "SSVI",
    "create_flat_local_vol",
    "create_svi_local_vol",
]
