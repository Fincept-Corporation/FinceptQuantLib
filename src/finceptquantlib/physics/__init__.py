"""
Physics Module

Physics-inspired computational methods:

Information Theory (physics.information_theory):
- Shannon entropy and generalizations
- KL/JS divergence
- Mutual information
- Transfer entropy for causality
- Rényi and Tsallis entropy

Statistical Mechanics (physics.statistical_mechanics):
- Boltzmann distribution
- Partition functions and free energy
- Ising and Potts models
- Maximum entropy methods
- MCMC sampling

Thermodynamics (physics.thermodynamics):
- Thermodynamic potentials (F, G, H)
- Carnot cycle and efficiency
- Van der Waals gas
- Phase transitions
- Legendre transforms

Applications to Finance:
- Market entropy as disorder measure
- Temperature as volatility analog
- Phase transitions in market regimes
- Information flow between assets
- Maximum entropy portfolio construction

Example:
    >>> from finceptquantlib.physics import (
    ...     shannon_entropy, BoltzmannDistribution, CarnotCycle
    ... )
    >>>
    >>> # Information content
    >>> shannon_entropy([0.5, 0.5], base=2)  # 1 bit
    1.0
    >>>
    >>> # Statistical mechanics
    >>> boltzmann = BoltzmannDistribution([0, 1, 2], temperature=1.0)
    >>>
    >>> # Thermodynamic efficiency
    >>> engine = CarnotCycle(T_hot=500, T_cold=300)
"""

from finceptquantlib.physics.information_theory import (
    shannon_entropy,
    joint_entropy,
    conditional_entropy,
    mutual_information,
    kl_divergence,
    js_divergence,
    cross_entropy,
    renyi_entropy,
    tsallis_entropy,
    differential_entropy_gaussian,
    differential_entropy_exponential,
    entropy_rate_markov,
    transfer_entropy,
    fisher_information,
)

from finceptquantlib.physics.statistical_mechanics import (
    BoltzmannDistribution,
    maximum_entropy_distribution,
    IsingModel,
    PottsModel,
    critical_temperature_ising_2d,
)

from finceptquantlib.physics.thermodynamics import (
    ThermodynamicState,
    helmholtz_free_energy,
    gibbs_free_energy,
    enthalpy,
    ideal_gas_internal_energy,
    ideal_gas_entropy,
    CarnotCycle,
    VanDerWaalsGas,
    clausius_clapeyron,
    joule_thomson_coefficient,
    maxwell_relations,
    LegendreTransform,
)

__all__ = [
    # Information Theory
    "shannon_entropy",
    "joint_entropy",
    "conditional_entropy",
    "mutual_information",
    "kl_divergence",
    "js_divergence",
    "cross_entropy",
    "renyi_entropy",
    "tsallis_entropy",
    "differential_entropy_gaussian",
    "differential_entropy_exponential",
    "entropy_rate_markov",
    "transfer_entropy",
    "fisher_information",
    # Statistical Mechanics
    "BoltzmannDistribution",
    "maximum_entropy_distribution",
    "IsingModel",
    "PottsModel",
    "critical_temperature_ising_2d",
    # Thermodynamics
    "ThermodynamicState",
    "helmholtz_free_energy",
    "gibbs_free_energy",
    "enthalpy",
    "ideal_gas_internal_energy",
    "ideal_gas_entropy",
    "CarnotCycle",
    "VanDerWaalsGas",
    "clausius_clapeyron",
    "joule_thomson_coefficient",
    "maxwell_relations",
    "LegendreTransform",
]
