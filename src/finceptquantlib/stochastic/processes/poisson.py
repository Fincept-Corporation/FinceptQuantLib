"""
Poisson Processes

Point processes for modeling discrete events:
- Homogeneous Poisson process
- Non-homogeneous (inhomogeneous) Poisson process
- Compound Poisson process
- Doubly stochastic (Cox) process
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from finceptquantlib.stochastic.sampling import poisson_sample


@dataclass
class PoissonProcess:
    """
    Homogeneous Poisson process.

    Properties:
    - N(0) = 0
    - Independent increments
    - N(t) - N(s) ~ Poisson(λ(t-s))
    - P(N(t+h) - N(t) = 1) = λh + o(h)
    - Inter-arrival times ~ Exp(λ)

    Attributes:
        intensity: λ, average rate of events per unit time
    """
    intensity: float  # λ
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.intensity < 0:
            raise ValueError("Intensity must be non-negative")
        self._rng = random.Random(self.seed) if self.seed is not None else random

    def simulate(
        self,
        T: float,
        n_paths: int = 1
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """
        Simulate Poisson process paths.

        Returns jump times and cumulative counts.

        Args:
            T: Terminal time
            n_paths: Number of paths

        Returns:
            (jump_times, counts): Jump times and N(t) values for each path
        """
        all_times = []
        all_counts = []

        for _ in range(n_paths):
            times = [0.0]
            counts = [0]
            t = 0.0

            while t < T:
                # Inter-arrival time is exponential
                u = self._rng.random()
                tau = -math.log(1.0 - u) / self.intensity
                t += tau

                if t <= T:
                    times.append(t)
                    counts.append(counts[-1] + 1)

            # Add terminal time if not already there
            if times[-1] < T:
                times.append(T)
                counts.append(counts[-1])

            all_times.append(times)
            all_counts.append(counts)

        return all_times, all_counts

    def simulate_fixed_grid(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> Tuple[List[float], List[List[int]]]:
        """
        Simulate on fixed time grid.

        Args:
            T: Terminal time
            n_steps: Number of time steps
            n_paths: Number of paths

        Returns:
            (times, paths): Time grid and N(t) values
        """
        dt = T / n_steps
        times = [i * dt for i in range(n_steps + 1)]

        paths = []
        for _ in range(n_paths):
            N = [0]
            for _ in range(n_steps):
                # Number of jumps in interval ~ Poisson(λdt)
                n_jumps = poisson_sample(self.intensity * dt)
                N.append(N[-1] + n_jumps)
            paths.append(N)

        return times, paths

    def expected_value(self, t: float) -> float:
        """E[N(t)] = λt."""
        return self.intensity * t

    def variance(self, t: float) -> float:
        """Var[N(t)] = λt."""
        return self.intensity * t

    def prob_k_events(self, k: int, t: float) -> float:
        """P(N(t) = k) = (λt)^k * e^(-λt) / k!"""
        lam_t = self.intensity * t
        if k < 0:
            return 0.0
        log_prob = k * math.log(lam_t) - lam_t - sum(
            math.log(i) for i in range(1, k + 1)
        )
        return math.exp(log_prob)


@dataclass
class InhomogeneousPoissonProcess:
    """
    Inhomogeneous (non-homogeneous) Poisson process.

    Time-varying intensity λ(t).

    Properties:
    - N(t) - N(s) ~ Poisson(∫_s^t λ(u)du)
    - Compensator: Λ(t) = ∫_0^t λ(s)ds

    Attributes:
        intensity_func: λ(t) function
        intensity_bound: Upper bound on λ(t) for simulation
    """
    intensity_func: Callable[[float], float]
    intensity_bound: float  # λ* upper bound
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.intensity_bound < 0:
            raise ValueError("Intensity bound must be non-negative")
        self._rng = random.Random(self.seed) if self.seed is not None else random

    def simulate(
        self,
        T: float,
        n_paths: int = 1
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """
        Simulate using thinning algorithm.

        1. Generate homogeneous Poisson(λ*) arrivals
        2. Accept each arrival with probability λ(t)/λ*
        """
        all_times = []
        all_counts = []

        for _ in range(n_paths):
            times = [0.0]
            counts = [0]
            t = 0.0

            while t < T:
                # Generate candidate from Poisson(λ*)
                u = self._rng.random()
                tau = -math.log(1.0 - u) / self.intensity_bound
                t += tau

                if t <= T:
                    # Accept/reject
                    accept_prob = self.intensity_func(t) / self.intensity_bound
                    if self._rng.random() < accept_prob:
                        times.append(t)
                        counts.append(counts[-1] + 1)

            # Add terminal time
            if times[-1] < T:
                times.append(T)
                counts.append(counts[-1])

            all_times.append(times)
            all_counts.append(counts)

        return all_times, all_counts

    def compensator(self, t: float, n_steps: int = 100) -> float:
        """
        Numerical computation of compensator Λ(t) = ∫_0^t λ(s)ds.
        """
        dt = t / n_steps
        integral = 0.0
        for i in range(n_steps):
            s = (i + 0.5) * dt
            integral += self.intensity_func(s) * dt
        return integral


@dataclass
class CompoundPoissonProcess:
    """
    Compound Poisson process.

    X(t) = Σ_{i=1}^{N(t)} Y_i

    where N(t) is Poisson process and Y_i are iid jump sizes.

    Attributes:
        intensity: λ, jump arrival rate
        jump_size_sampler: Function that samples jump sizes
        jump_mean: Mean of jump sizes (for moments)
        jump_variance: Variance of jump sizes (for moments)
    """
    intensity: float
    jump_size_sampler: Callable[[], float]
    jump_mean: float = 0.0
    jump_variance: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.intensity < 0:
            raise ValueError("Intensity must be non-negative")
        self._rng = random.Random(self.seed) if self.seed is not None else random

    def simulate(
        self,
        T: float,
        n_paths: int = 1
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Simulate compound Poisson paths.

        Returns:
            (times, values): Jump times and X(t) values
        """
        all_times = []
        all_values = []

        for _ in range(n_paths):
            times = [0.0]
            values = [0.0]
            t = 0.0

            while t < T:
                u = self._rng.random()
                tau = -math.log(1.0 - u) / self.intensity
                t += tau

                if t <= T:
                    jump_size = self.jump_size_sampler()
                    times.append(t)
                    values.append(values[-1] + jump_size)

            # Terminal time
            if times[-1] < T:
                times.append(T)
                values.append(values[-1])

            all_times.append(times)
            all_values.append(values)

        return all_times, all_values

    def simulate_fixed_grid(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> Tuple[List[float], List[List[float]]]:
        """Simulate on fixed time grid."""
        dt = T / n_steps
        times = [i * dt for i in range(n_steps + 1)]

        paths = []
        for _ in range(n_paths):
            X = [0.0]
            for _ in range(n_steps):
                n_jumps = poisson_sample(self.intensity * dt)
                total_jump = sum(self.jump_size_sampler() for _ in range(n_jumps))
                X.append(X[-1] + total_jump)
            paths.append(X)

        return times, paths

    def expected_value(self, t: float) -> float:
        """E[X(t)] = λt * E[Y]."""
        return self.intensity * t * self.jump_mean

    def variance(self, t: float) -> float:
        """Var[X(t)] = λt * E[Y²] = λt * (Var[Y] + E[Y]²)."""
        E_Y2 = self.jump_variance + self.jump_mean ** 2
        return self.intensity * t * E_Y2


def normal_jump_sampler(mean: float = 0.0, std: float = 1.0) -> Callable[[], float]:
    """Create sampler for normal jump sizes."""
    def sampler() -> float:
        return random.gauss(mean, std)
    return sampler


def double_exponential_sampler(
    p: float = 0.5,
    eta1: float = 1.0,
    eta2: float = 1.0
) -> Callable[[], float]:
    """
    Create sampler for double exponential (Kou) jump sizes.

    Y = X with probability p, -X' with probability 1-p
    where X ~ Exp(η₁), X' ~ Exp(η₂)

    Used in Kou jump-diffusion model.
    """
    def sampler() -> float:
        if random.random() < p:
            return -math.log(1 - random.random()) / eta1
        else:
            return math.log(1 - random.random()) / eta2  # Negative exponential
    return sampler


def uniform_jump_sampler(a: float = -1.0, b: float = 1.0) -> Callable[[], float]:
    """Create sampler for uniform jump sizes."""
    def sampler() -> float:
        return random.uniform(a, b)
    return sampler
