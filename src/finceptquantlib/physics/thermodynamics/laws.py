"""
Thermodynamics

Thermodynamic potentials and relations with financial interpretations:
- Internal energy, enthalpy, Helmholtz/Gibbs free energy
- Legendre transforms
- Maxwell relations
- Thermodynamic cycles
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


@dataclass
class ThermodynamicState:
    """
    Thermodynamic state variables.

    In financial analogy:
    - Temperature T → Market volatility/noise
    - Pressure P → Market stress/demand pressure
    - Volume V → Market capacity/liquidity
    - Energy U → Total market value
    - Entropy S → Market disorder/information
    """
    temperature: float  # T
    pressure: float     # P
    volume: float       # V
    entropy: Optional[float] = None  # S
    internal_energy: Optional[float] = None  # U


def helmholtz_free_energy(U: float, T: float, S: float) -> float:
    """
    Helmholtz free energy.

    F = U - T*S

    Useful at constant T, V.
    """
    return U - T * S


def gibbs_free_energy(U: float, P: float, V: float, T: float, S: float) -> float:
    """
    Gibbs free energy.

    G = U + PV - TS = H - TS = F + PV

    Useful at constant T, P.
    """
    return U + P * V - T * S


def enthalpy(U: float, P: float, V: float) -> float:
    """
    Enthalpy.

    H = U + PV

    Useful at constant P.
    """
    return U + P * V


def ideal_gas_internal_energy(n: float, T: float, cv: float = 1.5) -> float:
    """
    Internal energy of ideal gas.

    U = n * Cv * T

    Args:
        n: Number of moles
        T: Temperature
        cv: Heat capacity at constant volume (default: monatomic 3/2)
    """
    R = 1.0  # Using natural units
    return n * cv * R * T


def ideal_gas_entropy(
    n: float,
    T: float,
    V: float,
    cv: float = 1.5,
    T0: float = 1.0,
    V0: float = 1.0
) -> float:
    """
    Entropy of ideal gas (Sackur-Tetrode like).

    S = n*Cv*ln(T/T0) + n*R*ln(V/V0) + S0

    Args:
        n: Number of moles
        T: Temperature
        V: Volume
        cv: Heat capacity ratio
        T0, V0: Reference state
    """
    if T <= 0 or T0 <= 0 or V <= 0 or V0 <= 0:
        raise ValueError("T, T0, V, V0 must all be positive")
    R = 1.0
    return n * cv * R * math.log(T / T0) + n * R * math.log(V / V0)


@dataclass
class CarnotCycle:
    """
    Carnot heat engine cycle.

    The most efficient cycle between two temperatures.

    Attributes:
        T_hot: Hot reservoir temperature
        T_cold: Cold reservoir temperature
    """
    T_hot: float
    T_cold: float

    def __post_init__(self) -> None:
        if self.T_cold >= self.T_hot:
            raise ValueError("T_cold must be less than T_hot")
        if self.T_cold <= 0:
            raise ValueError("Temperatures must be positive")

    def efficiency(self) -> float:
        """
        Carnot efficiency.

        η = 1 - T_cold/T_hot

        Maximum possible efficiency for heat engine.
        """
        return 1 - self.T_cold / self.T_hot

    def work_per_cycle(self, Q_hot: float) -> float:
        """
        Work output per cycle.

        W = η * Q_hot
        """
        return self.efficiency() * Q_hot

    def heat_rejected(self, Q_hot: float) -> float:
        """
        Heat rejected to cold reservoir.

        Q_cold = Q_hot - W = Q_hot * (T_cold/T_hot)
        """
        return Q_hot * self.T_cold / self.T_hot

    def coefficient_of_performance_refrigerator(self) -> float:
        """
        COP for Carnot refrigerator.

        COP_R = T_cold / (T_hot - T_cold)
        """
        return self.T_cold / (self.T_hot - self.T_cold)

    def coefficient_of_performance_heat_pump(self) -> float:
        """
        COP for Carnot heat pump.

        COP_HP = T_hot / (T_hot - T_cold)
        """
        return self.T_hot / (self.T_hot - self.T_cold)


@dataclass
class VanDerWaalsGas:
    """
    Van der Waals gas model.

    (P + a/V²)(V - b) = RT

    Includes molecular interactions (a) and volume exclusion (b).
    """
    a: float = 1.0  # Attraction parameter
    b: float = 0.1  # Volume exclusion
    R: float = 1.0  # Gas constant

    def pressure(self, V: float, T: float) -> float:
        """
        Pressure from van der Waals equation.

        P = RT/(V-b) - a/V²
        """
        if V <= self.b:
            return float('inf')
        return self.R * T / (V - self.b) - self.a / (V ** 2)

    def critical_point(self) -> Tuple[float, float, float]:
        """
        Critical point (Tc, Pc, Vc).

        Vc = 3b
        Tc = 8a/(27Rb)
        Pc = a/(27b²)
        """
        Vc = 3 * self.b
        Tc = 8 * self.a / (27 * self.R * self.b)
        Pc = self.a / (27 * self.b ** 2)
        return Tc, Pc, Vc

    def reduced_equation(self, Vr: float, Tr: float) -> float:
        """
        Pressure in reduced units.

        Pr = 8Tr/(3Vr - 1) - 3/Vr²

        (Law of corresponding states)
        """
        if Vr <= 1 / 3:
            return float('inf')
        return 8 * Tr / (3 * Vr - 1) - 3 / (Vr ** 2)

    def compressibility_factor(self, V: float, T: float) -> float:
        """
        Compressibility factor Z = PV/(RT).

        Z = 1 for ideal gas.
        """
        if T <= 0:
            raise ValueError("Temperature must be positive")
        P = self.pressure(V, T)
        return P * V / (self.R * T)


def clausius_clapeyron(
    T: float,
    delta_H: float,
    P0: float,
    T0: float,
    R: float = 1.0
) -> float:
    """
    Clausius-Clapeyron equation for vapor pressure.

    ln(P/P0) = -(ΔH/R)(1/T - 1/T0)

    Describes phase equilibrium curve.

    Args:
        T: Temperature
        delta_H: Enthalpy of vaporization
        P0: Reference pressure at T0
        T0: Reference temperature
        R: Gas constant
    """
    return P0 * math.exp(-delta_H / R * (1 / T - 1 / T0))


def joule_thomson_coefficient(
    cp: float,
    V: float,
    T: float,
    dV_dT_P: float
) -> float:
    """
    Joule-Thomson coefficient.

    μ_JT = (1/Cp)[T(∂V/∂T)_P - V]

    Describes temperature change during throttling.

    Args:
        cp: Heat capacity at constant pressure
        V: Volume
        T: Temperature
        dV_dT_P: (∂V/∂T) at constant P
    """
    return (T * dV_dT_P - V) / cp


def maxwell_relations() -> dict:
    """
    Maxwell relations from thermodynamic potentials.

    Returns dictionary of relations as symbolic strings.
    """
    return {
        "dT/dV at S": "-dP/dS at V",  # From Helmholtz
        "dT/dP at S": "dV/dS at P",   # From Gibbs
        "dS/dV at T": "dP/dT at V",   # From Helmholtz
        "dS/dP at T": "-dV/dT at P",  # From Gibbs
    }


@dataclass
class LegendreTransform:
    """
    Legendre transform for thermodynamic potentials.

    Given f(x), the Legendre transform is:
    g(p) = sup_x [px - f(x)]

    where p = df/dx.
    """

    @staticmethod
    def transform(
        f: Callable[[float], float],
        df: Callable[[float], float],
        x_range: Tuple[float, float],
        n_points: int = 1000
    ) -> Callable[[float], float]:
        """
        Numerically compute Legendre transform.

        Args:
            f: Original function
            df: Derivative of f
            x_range: Range of x values
            n_points: Grid points

        Returns:
            Legendre transform g(p)
        """
        x_min, x_max = x_range
        xs = [x_min + (x_max - x_min) * i / n_points for i in range(n_points + 1)]

        # Precompute f values
        fs = [f(x) for x in xs]

        def g(p: float) -> float:
            # sup_x [p*x - f(x)] over all grid points
            best_val = float('-inf')
            for x, f_x in zip(xs, fs):
                val = p * x - f_x
                if val > best_val:
                    best_val = val
            return best_val

        return g

    @staticmethod
    def helmholtz_from_energy(
        U: Callable[[float], float],
        T: float
    ) -> Callable[[float], float]:
        """
        Compute Helmholtz free energy from internal energy.

        F(T, V) = U - TS = Legendre transform of U(S, V) w.r.t. S

        (Simplified for T as parameter)
        """
        def F(V: float) -> float:
            # F = U(V) - T*S where we need to eliminate S
            # This requires knowing U(S,V) relationship
            return U(V)  # Placeholder

        return F
