"""
Stochastic Processes

Core stochastic processes for financial modeling:
- Brownian motion (Wiener, GBM, bridges)
- Mean-reverting (OU, CIR, Vasicek, Hull-White)
- Poisson processes (homogeneous, compound)
- Jump-diffusion (Merton, Kou, Bates)
- Lévy processes (VG, NIG, CGMY)
"""

from finceptquantlib.stochastic.processes.brownian import (
    WienerProcess,
    GeometricBrownianMotion,
    BrownianBridge,
    correlated_brownian,
    reflected_brownian,
)

from finceptquantlib.stochastic.processes.mean_reverting import (
    OrnsteinUhlenbeck,
    CoxIngersollRoss,
    Vasicek,
    HullWhite,
)

from finceptquantlib.stochastic.processes.poisson import (
    PoissonProcess,
    InhomogeneousPoissonProcess,
    CompoundPoissonProcess,
    normal_jump_sampler,
    double_exponential_sampler,
    uniform_jump_sampler,
)

from finceptquantlib.stochastic.processes.jump_diffusion import (
    # New names (preferred)
    MertonProcess,
    KouProcess,
    BatesProcess,
    BatesModel,
)

from finceptquantlib.stochastic.processes.levy import (
    VarianceGammaProcess,
    NormalInverseGaussian,
    CGMYProcess,
)

__all__ = [
    # Brownian
    "WienerProcess",
    "GeometricBrownianMotion",
    "BrownianBridge",
    "correlated_brownian",
    "reflected_brownian",
    # Mean-reverting
    "OrnsteinUhlenbeck",
    "CoxIngersollRoss",
    "Vasicek",
    "HullWhite",
    # Poisson
    "PoissonProcess",
    "InhomogeneousPoissonProcess",
    "CompoundPoissonProcess",
    "normal_jump_sampler",
    "double_exponential_sampler",
    "uniform_jump_sampler",
    # Jump-diffusion (new names)
    "MertonProcess",
    "KouProcess",
    "BatesProcess",
    # Jump-diffusion (backward compatibility)
    "BatesModel",
    # Lévy
    "VarianceGammaProcess",
    "NormalInverseGaussian",
    "CGMYProcess",
]
