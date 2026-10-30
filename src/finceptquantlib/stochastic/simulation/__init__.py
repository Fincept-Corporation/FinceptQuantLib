"""
SDE Simulation Methods

Numerical schemes for stochastic differential equations:
- Euler-Maruyama (first-order)
- Milstein (higher-order)
- Exact simulation (for special processes)
- Multilevel Monte Carlo
"""

from finceptquantlib.stochastic.simulation.euler_maruyama import (
    euler_maruyama,
    euler_maruyama_nd,
    euler_maruyama_with_jumps,
    EulerMaruyamaResult,
)

from finceptquantlib.stochastic.simulation.milstein import (
    milstein,
    milstein_implicit,
    milstein_nd,
    MilsteinResult,
)

from finceptquantlib.stochastic.simulation.exact import (
    simulate_gbm_exact,
    simulate_ou_exact,
    simulate_cir_exact,
    simulate_heston_exact,
    ExactSimulationResult,
)

from finceptquantlib.stochastic.simulation.multilevel import (
    mlmc_estimate,
    mlmc_coupled_paths,
    MLMCResult,
)

__all__ = [
    # Euler-Maruyama
    "euler_maruyama",
    "euler_maruyama_nd",
    "euler_maruyama_with_jumps",
    "EulerMaruyamaResult",
    # Milstein
    "milstein",
    "milstein_implicit",
    "milstein_nd",
    "MilsteinResult",
    # Exact
    "simulate_gbm_exact",
    "simulate_ou_exact",
    "simulate_cir_exact",
    "simulate_heston_exact",
    "ExactSimulationResult",
    # MLMC
    "mlmc_estimate",
    "mlmc_coupled_paths",
    "MLMCResult",
]
