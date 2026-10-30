"""
Statistical Mechanics

Physics-inspired models for complex systems:

Boltzmann Distribution:
- BoltzmannDistribution: Gibbs measure
- Partition function, free energy
- Entropy, heat capacity

Spin Models:
- IsingModel: Binary spin interactions
- PottsModel: Multi-state generalization

Maximum Entropy:
- maximum_entropy_distribution: MaxEnt inference

Example:
    >>> from finceptquantlib.physics.statistical_mechanics import (
    ...     BoltzmannDistribution, IsingModel
    ... )
    >>>
    >>> # Energy levels and temperature
    >>> boltzmann = BoltzmannDistribution(energies=[0, 1, 2], temperature=1.0)
    >>> boltzmann.probabilities
    [0.665241, 0.244728, 0.090031]
    >>> boltzmann.free_energy()
    -0.407606
    >>>
    >>> # Ising model simulation
    >>> ising = IsingModel(n_spins=100, J=1.0, temperature=2.0)
    >>> samples = ising.metropolis_sample(1000)
"""

from finceptquantlib.physics.statistical_mechanics.ensembles import (
    BoltzmannDistribution,
    maximum_entropy_distribution,
    IsingModel,
    PottsModel,
    critical_temperature_ising_2d,
)

__all__ = [
    "BoltzmannDistribution",
    "maximum_entropy_distribution",
    "IsingModel",
    "PottsModel",
    "critical_temperature_ising_2d",
]
