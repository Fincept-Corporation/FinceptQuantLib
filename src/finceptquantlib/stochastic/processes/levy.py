"""
Lévy Processes

Pure jump processes with stationary independent increments:
- Variance Gamma (VG)
- Normal Inverse Gaussian (NIG)
- CGMY process
- Generalized Hyperbolic
"""

from __future__ import annotations

import cmath
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class VarianceGammaProcess:
    """
    Variance Gamma (VG) Process.

    X(t) = θG(t) + σW(G(t))

    where G(t) is Gamma process with parameters (t/ν, 1/ν).

    Equivalently: VG = BM subordinated by Gamma time change.

    Parameters:
    - σ (sigma): Volatility of BM
    - θ (theta): Drift of BM (controls skewness)
    - ν (nu): Variance rate of Gamma subordinator (controls kurtosis)

    Properties:
    - Pure jump process (infinite activity, finite variation)
    - Fat tails
    - Skewness controlled by θ
    - Semi-heavy tails

    For asset pricing:
    S(t) = S(0) * exp((r + ω)t + X(t))
    where ω = (1/ν) * ln(1 - θν - σ²ν/2) is the convexity correction
    """
    sigma: float  # BM volatility
    theta: float  # Drift (skewness)
    nu: float  # Gamma variance rate (kurtosis)
    S0: float = 100.0  # Initial price
    r: float = 0.0  # Risk-free rate
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("Sigma must be positive")
        if self.nu <= 0:
            raise ValueError("Nu must be positive")
        # Check that drift correction is finite
        if 1 - self.theta * self.nu - 0.5 * self.sigma ** 2 * self.nu <= 0:
            raise ValueError("Parameters violate finiteness condition")
        if self.S0 <= 0:
            raise ValueError("Initial price must be positive")
        self._rng = random.Random(self.seed) if self.seed is not None else random

    @property
    def omega(self) -> float:
        """Convexity correction for martingale condition."""
        return math.log(1 - self.theta * self.nu - 0.5 * self.sigma ** 2 * self.nu) / self.nu

    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> Tuple[List[float], List[List[float]]]:
        """
        Simulate VG paths using difference of Gamma processes.

        VG can be written as difference of two Gamma processes:
        X(t) = G_p(t) - G_m(t)

        where G_p and G_m are independent Gamma processes.
        """
        dt = T / n_steps
        times = [i * dt for i in range(n_steps + 1)]

        # Parameters for Gamma processes
        # VG = θG + σW(G) where G ~ Gamma(dt/ν, 1/ν)
        shape = dt / self.nu
        scale = self.nu

        paths = []
        for _ in range(n_paths):
            S = [self.S0]
            log_S = math.log(self.S0)

            for _ in range(n_steps):
                # Sample subordinated time change
                G = self._gamma_sample(shape, scale)

                # VG increment
                dX = self.theta * G + self.sigma * math.sqrt(G) * self._rng.gauss(0, 1)

                # Update log price
                log_S += (self.r + self.omega) * dt + dX
                S.append(math.exp(log_S))

            paths.append(S)

        return times, paths

    def _gamma_sample(self, shape: float, scale: float) -> float:
        """Sample from Gamma distribution using Marsaglia-Tsang method."""
        if shape < 1:
            # Use Ahrens-Dieter method for shape < 1
            return self._gamma_sample(shape + 1, scale) * self._rng.random() ** (1 / shape)

        d = shape - 1 / 3
        c = 1 / math.sqrt(9 * d)

        while True:
            x = self._rng.gauss(0, 1)
            v = (1 + c * x) ** 3

            if v > 0:
                u = self._rng.random()
                if u < 1 - 0.0331 * x ** 4:
                    return d * v * scale
                if math.log(u) < 0.5 * x ** 2 + d * (1 - v + math.log(v)):
                    return d * v * scale

    def characteristic_function(self, u: complex, t: float) -> complex:
        """
        Characteristic function.

        φ(u) = (1 - iuθν + σ²ν u²/2)^(-t/ν)
        """
        iu = 1j * u
        base = 1 - iu * self.theta * self.nu + 0.5 * self.sigma ** 2 * self.nu * u ** 2
        return base ** (-t / self.nu) * cmath.exp(iu * (self.r + self.omega) * t)

    def mean(self, t: float) -> float:
        """E[X(t)] = θt."""
        return self.theta * t

    def variance(self, t: float) -> float:
        """Var[X(t)] = (σ² + νθ²)t."""
        return (self.sigma ** 2 + self.nu * self.theta ** 2) * t

    def skewness(self) -> float:
        """Skewness (time-independent ratio)."""
        var = self.sigma ** 2 + self.nu * self.theta ** 2
        third_moment = self.theta * (3 * self.sigma ** 2 + 2 * self.nu * self.theta ** 2)
        return third_moment / var ** 1.5

    def excess_kurtosis(self) -> float:
        """Excess kurtosis (time-independent)."""
        var = self.sigma ** 2 + self.nu * self.theta ** 2
        # 4th cumulant contribution from VG
        c4 = 3 * (self.sigma ** 4 * self.nu + 2 * self.theta ** 4 * self.nu ** 2 +
                  4 * self.sigma ** 2 * self.theta ** 2 * self.nu)
        return c4 / var ** 2


@dataclass
class NormalInverseGaussian:
    """
    Normal Inverse Gaussian (NIG) Process.

    X(t) = βδ²I(t) + δW(I(t))

    where I(t) is Inverse Gaussian subordinator with parameters (t, δ√(α²-β²)).

    Parameters:
    - α (alpha): Tail heaviness (α > 0)
    - β (beta): Asymmetry (-α < β < α)
    - δ (delta): Scale (δ > 0)
    - μ (mu): Location

    Properties:
    - Pure jump process (infinite activity, infinite variation)
    - Heavier tails than VG
    - Good fit for financial returns

    Parametric constraints: α > 0, |β| < α, δ > 0
    """
    alpha: float  # Tail parameter
    beta: float  # Asymmetry
    delta: float  # Scale
    mu: float = 0.0  # Location
    S0: float = 100.0
    r: float = 0.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("Alpha must be positive")
        if abs(self.beta) >= self.alpha:
            raise ValueError("|Beta| must be less than alpha")
        if self.delta <= 0:
            raise ValueError("Delta must be positive")
        if self.S0 <= 0:
            raise ValueError("Initial price must be positive")
        self._rng = random.Random(self.seed) if self.seed is not None else random

    @property
    def gamma(self) -> float:
        """γ = √(α² - β²)."""
        return math.sqrt(self.alpha ** 2 - self.beta ** 2)

    @property
    def omega(self) -> float:
        """Convexity correction: ω = δ(√(α² - (β+1)²) - γ)."""
        return self.delta * (
            math.sqrt(self.alpha ** 2 - (self.beta + 1) ** 2) - self.gamma
        )

    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> Tuple[List[float], List[List[float]]]:
        """
        Simulate NIG paths using Inverse Gaussian subordination.
        """
        dt = T / n_steps
        times = [i * dt for i in range(n_steps + 1)]

        # IG subordinator parameters
        ig_mu = dt
        ig_lambda = self.delta ** 2 * self.gamma ** 2 * dt ** 2

        paths = []
        for _ in range(n_paths):
            S = [self.S0]
            log_S = math.log(self.S0)

            for _ in range(n_steps):
                # Sample IG time change
                I = self._inverse_gaussian_sample(ig_mu, ig_lambda)

                # NIG increment
                dX = (self.mu * dt + self.beta * self.delta ** 2 * I +
                      self.delta * math.sqrt(I) * self._rng.gauss(0, 1))

                log_S += (self.r + self.omega) * dt + dX
                S.append(math.exp(log_S))

            paths.append(S)

        return times, paths

    def _inverse_gaussian_sample(self, mu: float, lam: float) -> float:
        """Sample from Inverse Gaussian distribution."""
        nu = self._rng.gauss(0, 1)
        y = nu ** 2
        x = mu + (mu ** 2 * y) / (2 * lam) - (mu / (2 * lam)) * math.sqrt(
            4 * mu * lam * y + mu ** 2 * y ** 2
        )

        if self._rng.random() <= mu / (mu + x):
            return x
        else:
            return mu ** 2 / x

    def characteristic_function(self, u: complex, t: float) -> complex:
        """
        Characteristic function.

        φ(u) = exp(iuμt + δt(γ - √(α² - (β+iu)²)))
        """
        iu = 1j * u
        sqrt_arg = self.alpha ** 2 - (self.beta + iu) ** 2
        inner = self.gamma - sqrt_arg ** 0.5

        return cmath.exp(iu * (self.mu + self.r + self.omega) * t +
                        self.delta * t * inner)

    def mean(self, t: float) -> float:
        """E[X(t)] = μt + δβt/γ."""
        return (self.mu + self.delta * self.beta / self.gamma) * t

    def variance(self, t: float) -> float:
        """Var[X(t)] = δα²t/γ³."""
        return self.delta * self.alpha ** 2 * t / self.gamma ** 3


@dataclass
class CGMYProcess:
    """
    CGMY (Carr-Geman-Madan-Yor) Process.

    A tempered stable process with Lévy density:
    ν(x) = C * exp(-G|x|)/|x|^(1+Y) for x < 0
    ν(x) = C * exp(-M*x)/x^(1+Y) for x > 0

    Parameters:
    - C: Overall level of activity (C > 0)
    - G: Rate of decay for negative jumps (G > 0)
    - M: Rate of decay for positive jumps (M > 0)
    - Y: Fine structure (-∞ < Y < 2)

    Special cases:
    - Y < 0: Finite activity
    - 0 ≤ Y < 1: Infinite activity, finite variation
    - 1 ≤ Y < 2: Infinite activity, infinite variation

    Y = 0 gives Variance Gamma (approximately)
    """
    C: float  # Activity
    G: float  # Negative decay
    M: float  # Positive decay
    Y: float  # Fine structure
    S0: float = 100.0
    r: float = 0.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.C <= 0:
            raise ValueError("C must be positive")
        if self.G <= 0:
            raise ValueError("G must be positive")
        if self.M <= 0:
            raise ValueError("M must be positive")
        if self.Y >= 2:
            raise ValueError("Y must be < 2")
        if self.S0 <= 0:
            raise ValueError("Initial price must be positive")
        self._rng = random.Random(self.seed) if self.seed is not None else random

    @property
    def omega(self) -> float:
        """
        Convexity correction.

        ω = -C*Γ(-Y)*((M-1)^Y - M^Y + (G+1)^Y - G^Y)
        """
        if self.Y == 0:
            # Limit as Y → 0
            return -self.C * (math.log(self.M - 1) - math.log(self.M) +
                             math.log(self.G + 1) - math.log(self.G))
        else:
            return -self.C * math.gamma(-self.Y) * (
                (self.M - 1) ** self.Y - self.M ** self.Y +
                (self.G + 1) ** self.Y - self.G ** self.Y
            )

    def characteristic_function(self, u: complex, t: float) -> complex:
        """
        Characteristic function.

        φ(u) = exp(t*C*Γ(-Y)*((M-iu)^Y - M^Y + (G+iu)^Y - G^Y))
        """
        iu = 1j * u

        if abs(self.Y) < 1e-10:
            # Y ≈ 0: Use logarithmic form
            psi = self.C * (
                math.log(self.M - iu) - math.log(self.M) +
                math.log(self.G + iu) - math.log(self.G)
            )
        else:
            psi = self.C * math.gamma(-self.Y) * (
                (self.M - iu) ** self.Y - self.M ** self.Y +
                (self.G + iu) ** self.Y - self.G ** self.Y
            )

        return cmath.exp(t * psi + iu * (self.r + self.omega) * t)

    def mean(self, t: float) -> float:
        """E[X(t)] = C*Γ(1-Y)*(1/M^(1-Y) - 1/G^(1-Y))*t for Y < 1."""
        if self.Y >= 1:
            return float('nan')  # Mean doesn't exist
        return self.C * math.gamma(1 - self.Y) * (
            self.M ** (self.Y - 1) - self.G ** (self.Y - 1)
        ) * t

    def variance(self, t: float) -> float:
        """Var[X(t)] = C*Γ(2-Y)*(1/M^(2-Y) + 1/G^(2-Y))*t."""
        return self.C * math.gamma(2 - self.Y) * (
            self.M ** (self.Y - 2) + self.G ** (self.Y - 2)
        ) * t

    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> Tuple[List[float], List[List[float]]]:
        """
        Simulate CGMY using series representation.

        Note: This is a simplified approximation.
        Exact simulation is computationally intensive.
        """
        dt = T / n_steps
        times = [i * dt for i in range(n_steps + 1)]

        paths = []
        for _ in range(n_paths):
            S = [self.S0]
            log_S = math.log(self.S0)

            for _ in range(n_steps):
                # Simplified: use characteristic function approximation
                # In practice, use shot noise or series methods
                mean = self.mean(dt) if self.Y < 1 else 0
                var = self.variance(dt)

                dX = self._rng.gauss(mean, math.sqrt(var))
                log_S += (self.r + self.omega) * dt + dX - mean
                S.append(math.exp(log_S))

            paths.append(S)

        return times, paths
