"""
Thermodynamics

Thermodynamic laws and potentials:

Potentials:
- helmholtz_free_energy: F = U - TS
- gibbs_free_energy: G = U + PV - TS
- enthalpy: H = U + PV
- ideal_gas_internal_energy: U = nCvT
- ideal_gas_entropy: S(T, V)

Cycles:
- CarnotCycle: Maximum efficiency heat engine

Gas Models:
- VanDerWaalsGas: Non-ideal gas with interactions

Relations:
- clausius_clapeyron: Phase equilibrium
- joule_thomson_coefficient: Throttling
- maxwell_relations: Thermodynamic identities
- LegendreTransform: Potential transformations

Example:
    >>> from finceptquantlib.physics.thermodynamics import (
    ...     CarnotCycle, VanDerWaalsGas
    ... )
    >>>
    >>> # Carnot efficiency
    >>> engine = CarnotCycle(T_hot=500, T_cold=300)
    >>> engine.efficiency()  # 0.4
    0.4
    >>>
    >>> # Van der Waals critical point
    >>> gas = VanDerWaalsGas(a=1.0, b=0.1)
    >>> Tc, Pc, Vc = gas.critical_point()
"""

from finceptquantlib.physics.thermodynamics.laws import (
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
