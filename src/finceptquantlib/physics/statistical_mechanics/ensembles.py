"""
Statistical Mechanics Ensembles

Statistical mechanical frameworks applicable to financial systems:
- Boltzmann distribution
- Canonical ensemble
- Grand canonical ensemble
- Maximum entropy methods
"""

from __future__ import annotations

import math
import random
from bisect import bisect_left
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


@dataclass
class BoltzmannDistribution:
    """
    Boltzmann distribution.

    P(state) ∝ exp(-E(state) / (k_B * T))

    In finance, temperature T is often interpreted as market volatility
    or noise level.

    Attributes:
        energies: List of energy levels
        temperature: Temperature parameter (k_B * T)
    """
    energies: List[float]
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        self._compute_partition()

    def _compute_partition(self) -> None:
        """Compute partition function and probabilities."""
        beta = 1 / self.temperature
        # Shift energies by minimum to prevent overflow in exp()
        self._E_min = min(self.energies)
        self._boltzmann_factors = [math.exp(-beta * (E - self._E_min)) for E in self.energies]
        self._Z = sum(self._boltzmann_factors)
        self._probabilities = [bf / self._Z for bf in self._boltzmann_factors]

    @property
    def partition_function(self) -> float:
        """Partition function Z = Σ exp(-βE)."""
        return self._Z

    @property
    def probabilities(self) -> List[float]:
        """State probabilities."""
        return self._probabilities

    def free_energy(self) -> float:
        """
        Helmholtz free energy.

        F = -T * ln(Z)

        Uses log-space: ln(Z) = ln(Z_shifted) + beta * E_min
        so F = -T * ln(Z_shifted) + E_min
        """
        return -self.temperature * math.log(self._Z) + self._E_min

    def average_energy(self) -> float:
        """
        Mean energy.

        <E> = Σ P(i) * E(i) = -∂ln(Z)/∂β
        """
        return sum(p * E for p, E in zip(self._probabilities, self.energies))

    def energy_variance(self) -> float:
        """
        Variance of energy.

        σ²_E = <E²> - <E>² = T² * ∂²ln(Z)/∂T²
        """
        mean_E = self.average_energy()
        mean_E2 = sum(p * E ** 2 for p, E in zip(self._probabilities, self.energies))
        return mean_E2 - mean_E ** 2

    def entropy(self) -> float:
        """
        Entropy.

        S = -Σ P(i) ln P(i) = (U - F) / T
        """
        entropy = 0.0
        for p in self._probabilities:
            if p > 0:
                entropy -= p * math.log(p)
        return entropy

    def heat_capacity(self) -> float:
        """
        Heat capacity.

        C = dU/dT = σ²_E / T²
        """
        return self.energy_variance() / (self.temperature ** 2)

    def sample(self, n: int, seed: Optional[int] = None) -> List[int]:
        """Sample states according to Boltzmann distribution."""
        rng = random.Random(seed) if seed is not None else random

        # Precompute cumulative distribution
        cumulative = []
        acc = 0.0
        for p in self._probabilities:
            acc += p
            cumulative.append(acc)

        samples = []
        for _ in range(n):
            r = rng.random()
            idx = bisect_left(cumulative, r)
            samples.append(min(idx, len(self._probabilities) - 1))

        return samples


def maximum_entropy_distribution(
    constraints: List[Tuple[Callable[[int], float], float]],
    n_states: int,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> List[float]:
    """
    Find maximum entropy distribution subject to constraints.

    max H(P) = -Σ pᵢ ln pᵢ
    s.t. Σ pᵢ f_k(i) = μ_k for each constraint
         Σ pᵢ = 1

    Uses iterative scaling algorithm.

    Args:
        constraints: List of (function, target_mean) tuples
        n_states: Number of states
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Maximum entropy probabilities
    """
    # Initialize with uniform distribution
    p = [1 / n_states] * n_states
    lambdas = [0.0] * len(constraints)

    for iteration in range(max_iter):
        # Update probabilities
        weights = [0.0] * n_states
        for i in range(n_states):
            w = 0.0
            for k, (f, _) in enumerate(constraints):
                w += lambdas[k] * f(i)
            weights[i] = math.exp(-w)

        total_weight = sum(weights)
        p_new = [w / total_weight for w in weights]

        # Update Lagrange multipliers
        max_change = 0.0
        for k, (f, mu_k) in enumerate(constraints):
            current_mean = sum(p_new[i] * f(i) for i in range(n_states))
            error = current_mean - mu_k

            # Update lambda
            if abs(error) > tol:
                # Simple gradient step
                lambdas[k] += 0.1 * error
                max_change = max(max_change, abs(error))

        p = p_new

        if max_change < tol:
            break

    return p


@dataclass
class IsingModel:
    """
    1D Ising model.

    H = -J Σᵢ sᵢsᵢ₊₁ - h Σᵢ sᵢ

    where sᵢ ∈ {-1, +1}

    Used to model binary interactions in financial networks.

    Attributes:
        n_spins: Number of spins
        J: Coupling constant
        h: External field
        temperature: Temperature (k_B * T)
    """
    n_spins: int
    J: float = 1.0
    h: float = 0.0
    temperature: float = 1.0

    def energy(self, spins: List[int]) -> float:
        """
        Calculate energy of spin configuration.

        H = -J Σ sᵢsᵢ₊₁ - h Σ sᵢ
        """
        E = 0.0
        n = len(spins)

        # Coupling energy (periodic boundary)
        for i in range(n):
            E -= self.J * spins[i] * spins[(i + 1) % n]

        # Field energy
        E -= self.h * sum(spins)

        return E

    def magnetization(self, spins: List[int]) -> float:
        """Average magnetization m = (1/N) Σ sᵢ."""
        return sum(spins) / len(spins)

    def partition_function_exact(self) -> float:
        """
        Exact partition function for small systems.

        Z = Σ exp(-βH)
        """
        if self.n_spins > 20:
            raise ValueError("Too many spins for exact computation")

        beta = 1 / self.temperature
        Z = 0.0

        # Enumerate all 2^n configurations
        for config in range(2 ** self.n_spins):
            spins = [1 if (config >> i) & 1 else -1 for i in range(self.n_spins)]
            Z += math.exp(-beta * self.energy(spins))

        return Z

    def transfer_matrix_free_energy(self) -> float:
        """
        Free energy using transfer matrix method.

        F = -T * ln(Z) = -T * N * ln(λ_max)

        where λ_max is largest eigenvalue of transfer matrix.
        """
        beta = 1 / self.temperature
        K = beta * self.J
        H = beta * self.h

        # Transfer matrix elements
        T_pp = math.exp(K + H)  # s_i = s_{i+1} = +1
        T_pm = math.exp(-K)      # s_i = +1, s_{i+1} = -1
        T_mp = math.exp(-K)      # s_i = -1, s_{i+1} = +1
        T_mm = math.exp(K - H)  # s_i = s_{i+1} = -1

        # Eigenvalues of 2x2 transfer matrix
        trace = T_pp + T_mm
        det = T_pp * T_mm - T_pm * T_mp
        discriminant = trace ** 2 - 4 * det

        if discriminant < 0:
            discriminant = 0

        lambda_max = (trace + math.sqrt(discriminant)) / 2

        return -self.temperature * self.n_spins * math.log(lambda_max)

    def metropolis_sample(
        self,
        n_samples: int,
        n_warmup: int = 1000,
        seed: Optional[int] = None
    ) -> List[List[int]]:
        """
        Generate samples using Metropolis algorithm.

        MCMC sampling from Boltzmann distribution.
        """
        rng = random.Random(seed) if seed is not None else random

        # Initialize random configuration
        spins = [rng.choice([-1, 1]) for _ in range(self.n_spins)]
        beta = 1 / self.temperature

        samples = []

        for step in range(n_warmup + n_samples):
            # Pick random spin to flip
            i = rng.randint(0, self.n_spins - 1)

            # Calculate energy change
            left = (i - 1) % self.n_spins
            right = (i + 1) % self.n_spins
            delta_E = 2 * spins[i] * (self.J * (spins[left] + spins[right]) + self.h)

            # Metropolis acceptance
            if delta_E <= 0 or rng.random() < math.exp(-beta * delta_E):
                spins[i] *= -1

            if step >= n_warmup:
                samples.append(spins[:])

        return samples


@dataclass
class PottsModel:
    """
    q-state Potts model.

    H = -J Σᵢⱼ δ(sᵢ, sⱼ)

    Generalization of Ising (q=2 is Ising).

    Attributes:
        n_sites: Number of sites
        q: Number of states per site
        J: Coupling constant
        temperature: Temperature
    """
    n_sites: int
    q: int = 3
    J: float = 1.0
    temperature: float = 1.0

    def energy(self, states: List[int]) -> float:
        """Energy with periodic boundary."""
        E = 0.0
        n = len(states)

        for i in range(n):
            if states[i] == states[(i + 1) % n]:
                E -= self.J

        return E

    def metropolis_sample(
        self,
        n_samples: int,
        n_warmup: int = 1000,
        seed: Optional[int] = None
    ) -> List[List[int]]:
        """MCMC sampling using Metropolis algorithm."""
        rng = random.Random(seed) if seed is not None else random

        # Initialize
        states = [rng.randint(0, self.q - 1) for _ in range(self.n_sites)]
        beta = 1 / self.temperature

        samples = []

        for step in range(n_warmup + n_samples):
            i = rng.randint(0, self.n_sites - 1)
            old_state = states[i]
            new_state = rng.randint(0, self.q - 2)
            if new_state >= old_state:
                new_state += 1

            # Energy change
            left = (i - 1) % self.n_sites
            right = (i + 1) % self.n_sites

            delta_E = 0
            # Lost bonds
            if old_state == states[left]:
                delta_E += self.J
            if old_state == states[right]:
                delta_E += self.J
            # Gained bonds
            if new_state == states[left]:
                delta_E -= self.J
            if new_state == states[right]:
                delta_E -= self.J

            # Accept/reject
            if delta_E <= 0 or rng.random() < math.exp(-beta * delta_E):
                states[i] = new_state

            if step >= n_warmup:
                samples.append(states[:])

        return samples


def critical_temperature_ising_2d() -> float:
    """
    Critical temperature of 2D Ising model.

    T_c = 2J / ln(1 + √2) ≈ 2.269J

    (Onsager solution)
    """
    return 2 / math.log(1 + math.sqrt(2))
