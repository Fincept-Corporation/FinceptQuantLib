"""
Pricing Models Module

Provides advanced pricing models for derivatives:
- Stochastic Volatility: Heston, SABR, Stochastic Local Vol
- Jump Diffusion: Merton, Kou Double Exponential, Variance Gamma
- Interest Rate: Vasicek, CIR, Hull-White, Black-Karasinski
- Local Volatility: Dupire

Each model includes:
- Analytical pricing (where available)
- Monte Carlo simulation
- Calibration routines
- Greeks via automatic differentiation
"""

from finceptquantlib.models.stochastic_vol import (
    HestonModel,
    HestonParameters,
    HestonPricer,
    SABRModelEnhanced,
    SABRParameters,
)

from finceptquantlib.models.jump_diffusion import (
    MertonJumpDiffusion,
    MertonParameters,
    KouJumpDiffusion,
    KouParameters,
    VarianceGamma,
    VGParameters,
    merton_call_price,
    kou_call_price,
    vg_call_price,
)

from finceptquantlib.models.short_rate import (
    VasicekModel,
    CIRModel,
    HullWhiteModel,
    BlackKarasinskiModel,
    ShortRateParameters,
    create_vasicek,
    create_cir,
    create_hull_white,
)

from finceptquantlib.models.local_vol import (
    DupireLocalVol,
    LocalVolSurface,
    SVI,
    SSVI,
    create_flat_local_vol,
    create_svi_local_vol,
)

__all__ = [
    # Stochastic Volatility
    "HestonModel",
    "HestonParameters",
    "HestonPricer",
    "SABRModelEnhanced",
    "SABRParameters",
    # Jump Diffusion
    "MertonJumpDiffusion",
    "MertonParameters",
    "KouJumpDiffusion",
    "KouParameters",
    "VarianceGamma",
    "VGParameters",
    "merton_call_price",
    "kou_call_price",
    "vg_call_price",
    # Short Rate
    "VasicekModel",
    "CIRModel",
    "HullWhiteModel",
    "BlackKarasinskiModel",
    "ShortRateParameters",
    "create_vasicek",
    "create_cir",
    "create_hull_white",
    # Local Volatility
    "DupireLocalVol",
    "LocalVolSurface",
    "SVI",
    "SSVI",
    "create_flat_local_vol",
    "create_svi_local_vol",
]
