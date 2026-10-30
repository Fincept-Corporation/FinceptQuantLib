"""
Brownian Motion and Geometric Brownian Motion

Standard and generalized Brownian motion processes:
- Wiener process (standard Brownian motion)
- Geometric Brownian motion (GBM)
- Brownian bridge
- Correlated Brownian motions
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class WienerProcess:
    """
    Standard Wiener process (Brownian motion).

    Properties:
    - W(0) = 0
    - Independent increments
    - W(t) - W(s) ~ N(0, t-s)
    - Continuous paths

    Example:
        >>> w = WienerProcess()
        >>> paths = w.simulate(T=1.0, n_steps=252, n_paths=1000)
    """
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed) if self.seed is not None else random

    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1,
        W0: float = 0.0
    ) -> Tuple[List[float], List[List[float]]]:
        """
        Simulate Wiener process paths.

        Args:
            T: Terminal time
            n_steps: Number of time steps
            n_paths: Number of paths to simulate
            W0: Initial value (usually 0)

        Returns:
            (times, paths): Time grid and simulated paths
        """
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        times = [i * dt for i in range(n_steps + 1)]

        paths = []
        for _ in range(n_paths):
            W = [W0]
            for _ in range(n_steps):
                dW = self._rng.gauss(0, sqrt_dt)
                W.append(W[-1] + dW)
            paths.append(W)

        return times, paths

    def increment(self, dt: float) -> float:
        """Generate single Wiener increment."""
        return self._rng.gauss(0, math.sqrt(dt))

    def quadratic_variation(self, T: float) -> float:
        """Expected quadratic variation [W,W]_T = T."""
        return T

    @staticmethod
    def covariance(s: float, t: float) -> float:
        """Cov(W_s, W_t) = min(s, t)."""
        return min(s, t)


@dataclass
class GeometricBrownianMotion:
    """
    Geometric Brownian Motion (GBM).

    dS = μ*S*dt + σ*S*dW

    Solution: S(t) = S(0) * exp((μ - σ²/2)*t + σ*W(t))

    Used for modeling stock prices under risk-neutral measure:
    - μ = r (risk-free rate)
    - σ = volatility

    Attributes:
        mu: Drift coefficient
        sigma: Volatility coefficient
        S0: Initial value
    """
    mu: float
    sigma: float
    S0: float = 100.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError("Volatility must be non-negative")
        if self.S0 <= 0:
            raise ValueError("Initial value must be positive")
        self._rng = random.Random(self.seed) if self.seed is not None else random

    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> Tuple[List[float], List[List[float]]]:
        """
        Simulate GBM paths using exact solution.

        Args:
            T: Terminal time
            n_steps: Number of time steps
            n_paths: Number of paths

        Returns:
            (times, paths): Time grid and simulated paths
        """
        dt = T / n_steps
        drift = (self.mu - 0.5 * self.sigma ** 2) * dt
        vol = self.sigma * math.sqrt(dt)

        times = [i * dt for i in range(n_steps + 1)]

        paths = []
        for _ in range(n_paths):
            S = [self.S0]
            for _ in range(n_steps):
                Z = self._rng.gauss(0, 1)
                S_new = S[-1] * math.exp(drift + vol * Z)
                S.append(S_new)
            paths.append(S)

        return times, paths

    def exact_value(self, t: float, W_t: float) -> float:
        """
        Exact solution at time t given Wiener path value.

        S(t) = S(0) * exp((μ - σ²/2)*t + σ*W(t))
        """
        return self.S0 * math.exp(
            (self.mu - 0.5 * self.sigma ** 2) * t + self.sigma * W_t
        )

    def expected_value(self, t: float) -> float:
        """E[S(t)] = S(0) * exp(μ*t)."""
        return self.S0 * math.exp(self.mu * t)

    def variance(self, t: float) -> float:
        """Var[S(t)] = S(0)² * exp(2μt) * (exp(σ²t) - 1)."""
        return (self.S0 ** 2 * math.exp(2 * self.mu * t) *
                (math.exp(self.sigma ** 2 * t) - 1))

    def pdf(self, S: float, t: float) -> float:
        """
        Probability density function (lognormal).

        f(S,t) = 1/(S*σ*√(2πt)) * exp(-(ln(S/S0) - (μ-σ²/2)t)²/(2σ²t))
        """
        if S <= 0 or t <= 0:
            return 0.0
        m = math.log(self.S0) + (self.mu - 0.5 * self.sigma ** 2) * t
        v = self.sigma * math.sqrt(t)
        x = math.log(S)
        return math.exp(-0.5 * ((x - m) / v) ** 2) / (S * v * math.sqrt(2 * math.pi))

    def quantile(self, p: float, t: float) -> float:
        """
        Quantile function (inverse CDF).

        Q(p) = S(0) * exp((μ-σ²/2)t + σ*√t * Φ⁻¹(p))
        """
        from finceptquantlib.core.utilities.distributions import norm_ppf
        m = math.log(self.S0) + (self.mu - 0.5 * self.sigma ** 2) * t
        v = self.sigma * math.sqrt(t)
        return math.exp(m + v * norm_ppf(p))


@dataclass
class BrownianBridge:
    """
    Brownian Bridge.

    Wiener process conditioned to hit specific value at terminal time:
    B(t) = W(t) - (t/T)*W(T), or equivalently
    B(t) | B(0)=a, B(T)=b

    Properties:
    - B(0) = a, B(T) = b (endpoints fixed)
    - E[B(t)] = a + (t/T)(b-a)
    - Var[B(t)] = t(T-t)/T

    Used in:
    - Simulation algorithms (stratified sampling)
    - Barrier option pricing
    - Statistical bridges
    """
    a: float = 0.0  # Starting value
    b: float = 0.0  # Ending value
    T: float = 1.0  # Terminal time
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.T <= 0:
            raise ValueError("Terminal time must be positive")
        self._rng = random.Random(self.seed) if self.seed is not None else random

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1
    ) -> Tuple[List[float], List[List[float]]]:
        """
        Simulate Brownian bridge paths.

        Uses the construction:
        B(t) = a + (t/T)(b-a) + W(t) - (t/T)*W(T)

        Implemented via sequential conditioning.
        """
        dt = self.T / n_steps
        times = [i * dt for i in range(n_steps + 1)]

        paths = []
        for _ in range(n_paths):
            # Generate Wiener increments
            W = [0.0]
            for _ in range(n_steps):
                dW = self._rng.gauss(0, math.sqrt(dt))
                W.append(W[-1] + dW)
            W_T = W[-1]

            # Transform to bridge
            path = [self.a]
            for i in range(1, n_steps + 1):
                t = times[i]
                ratio = t / self.T
                B_t = self.a + ratio * (self.b - self.a) + W[i] - ratio * W_T
                path.append(B_t)
            paths.append(path)

        return times, paths

    def mean(self, t: float) -> float:
        """E[B(t)] = a + (t/T)(b-a)."""
        return self.a + (t / self.T) * (self.b - self.a)

    def variance(self, t: float) -> float:
        """Var[B(t)] = t(T-t)/T."""
        return t * (self.T - t) / self.T

    def covariance(self, s: float, t: float) -> float:
        """
        Cov[B(s), B(t)] = min(s,t) - st/T = s(T-t)/T for s <= t.
        """
        s, t = min(s, t), max(s, t)
        return s * (self.T - t) / self.T


def correlated_brownian(
    n_assets: int,
    correlation_matrix: List[List[float]],
    T: float,
    n_steps: int,
    n_paths: int = 1,
    seed: Optional[int] = None
) -> Tuple[List[float], List[List[List[float]]]]:
    """
    Generate correlated Brownian motion paths.

    Uses Cholesky decomposition: W = L * Z where Z ~ N(0,I).

    Args:
        n_assets: Number of correlated assets
        correlation_matrix: n x n correlation matrix
        T: Terminal time
        n_steps: Number of time steps
        n_paths: Number of path sets
        seed: Random seed

    Returns:
        (times, paths): Time grid and paths[path_idx][asset_idx][time_idx]
    """
    from finceptquantlib.numerical.linalg import cholesky

    rng = random.Random(seed) if seed is not None else random

    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    times = [i * dt for i in range(n_steps + 1)]

    # Cholesky decomposition
    L = cholesky(correlation_matrix)

    all_paths = []
    for _ in range(n_paths):
        # Initialize paths
        paths = [[0.0] for _ in range(n_assets)]

        for _ in range(n_steps):
            # Generate independent normals
            Z = [rng.gauss(0, sqrt_dt) for _ in range(n_assets)]

            # Apply Cholesky transformation
            dW = [sum(L[i][j] * Z[j] for j in range(i + 1))
                  for i in range(n_assets)]

            # Update paths
            for i in range(n_assets):
                paths[i].append(paths[i][-1] + dW[i])

        all_paths.append(paths)

    return times, all_paths


def reflected_brownian(
    T: float,
    n_steps: int,
    lower: float = 0.0,
    upper: float = float('inf'),
    n_paths: int = 1,
    seed: Optional[int] = None
) -> Tuple[List[float], List[List[float]]]:
    """
    Reflected Brownian motion.

    Brownian motion reflected at boundaries.
    |W(t)| is reflected BM at 0.

    Args:
        T: Terminal time
        n_steps: Number of steps
        lower: Lower reflection barrier
        upper: Upper reflection barrier
        n_paths: Number of paths
        seed: Random seed

    Returns:
        (times, paths): Reflected paths
    """
    rng = random.Random(seed) if seed is not None else random

    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    times = [i * dt for i in range(n_steps + 1)]

    paths = []
    for _ in range(n_paths):
        X = [0.0]
        for _ in range(n_steps):
            dW = rng.gauss(0, sqrt_dt)
            X_new = X[-1] + dW

            # Reflect at boundaries (with iteration limit to prevent hang)
            for _ in range(100):
                if lower <= X_new <= upper:
                    break
                if X_new < lower:
                    X_new = 2 * lower - X_new
                if X_new > upper:
                    X_new = 2 * upper - X_new
            else:
                # Clamp if reflection doesn't converge
                X_new = max(lower, min(X_new, upper))

            X.append(X_new)
        paths.append(X)

    return times, paths
