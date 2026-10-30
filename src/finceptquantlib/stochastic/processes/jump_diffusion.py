"""
Jump-Diffusion Processes

Models combining continuous diffusion with discrete jumps:
- Merton jump-diffusion (MertonProcess)
- Kou double-exponential jump-diffusion (KouProcess)
- Bates stochastic volatility with jumps (BatesProcess)

Note: These are simulation-focused classes for path generation.
For pricing models, see finceptquantlib.models.jump_diffusion.
"""

from __future__ import annotations

import math
import cmath
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from finceptquantlib.stochastic.sampling import poisson_sample, double_exponential_sampler


@dataclass
class MertonProcess:
    """
    Merton Jump-Diffusion Process (1976).

    Simulation-focused implementation for path generation.

    dS/S = (μ - λk)dt + σdW + dJ

    where:
    - dJ is compound Poisson with intensity λ
    - Jump sizes Y ~ N(μ_J, σ_J²)
    - k = E[e^Y - 1] = exp(μ_J + σ_J²/2) - 1

    Under risk-neutral measure with r:
    dS/S = (r - λk)dt + σdW + dJ

    Note: For pricing, see finceptquantlib.models.jump_diffusion.MertonJumpDiffusion

    Attributes:
        mu: Drift rate
        sigma: Diffusion volatility
        lambda_: Jump intensity (average jumps per year)
        mu_J: Mean of log jump size
        sigma_J: Std dev of log jump size
        S0: Initial asset price
    """
    mu: float
    sigma: float
    lambda_: float  # Jump intensity
    mu_J: float  # Mean log jump
    sigma_J: float  # Jump vol
    S0: float = 100.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError("Volatility must be non-negative")
        if self.lambda_ < 0:
            raise ValueError("Jump intensity must be non-negative")
        if self.sigma_J < 0:
            raise ValueError("Jump volatility must be non-negative")
        if self.S0 <= 0:
            raise ValueError("Initial price must be positive")

    @property
    def k(self) -> float:
        """Expected percentage jump: k = E[e^Y - 1]."""
        return math.exp(self.mu_J + 0.5 * self.sigma_J ** 2) - 1

    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> Tuple[List[float], List[List[float]]]:
        """
        Simulate Merton jump-diffusion paths.

        Uses exact simulation of GBM between jumps.

        Args:
            T: Terminal time
            n_steps: Number of time steps
            n_paths: Number of paths

        Returns:
            (times, paths): Time grid and simulated paths
        """
        rng = random.Random(self.seed) if self.seed is not None else random

        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        times = [i * dt for i in range(n_steps + 1)]

        # Adjusted drift for jump compensation
        drift_adj = self.mu - self.lambda_ * self.k - 0.5 * self.sigma ** 2

        paths = []
        for _ in range(n_paths):
            S = [self.S0]
            log_S = math.log(self.S0)

            for _ in range(n_steps):
                # Diffusion part
                dW = rng.gauss(0, sqrt_dt)
                log_S += drift_adj * dt + self.sigma * dW

                # Jump part - using centralized sampling
                n_jumps = poisson_sample(self.lambda_ * dt, rng)
                for _ in range(n_jumps):
                    J = rng.gauss(self.mu_J, self.sigma_J)
                    log_S += J

                S.append(math.exp(log_S))

            paths.append(S)

        return times, paths

    def characteristic_function(self, u: complex, t: float) -> complex:
        """
        Characteristic function φ(u) = E[exp(iu*log(S(t)/S(0)))].

        Used for FFT pricing methods.
        """
        iu = 1j * u

        # Diffusion part
        phi_diff = cmath.exp(
            iu * (self.mu - self.lambda_ * self.k - 0.5 * self.sigma ** 2) * t
            - 0.5 * self.sigma ** 2 * u ** 2 * t
        )

        # Jump part: E[exp(iu*Y)] = exp(iu*μ_J - σ_J²*u²/2)
        phi_jump_single = cmath.exp(
            iu * self.mu_J - 0.5 * self.sigma_J ** 2 * u ** 2
        )
        phi_jump = cmath.exp(self.lambda_ * t * (phi_jump_single - 1))

        return phi_diff * phi_jump

    def expected_value(self, t: float) -> float:
        """E[S(t)] = S(0) * exp(μt)."""
        return self.S0 * math.exp(self.mu * t)


@dataclass
class KouProcess:
    """
    Kou Double-Exponential Jump-Diffusion Process (2002).

    Simulation-focused implementation for path generation.

    dS/S = (μ - λk)dt + σdW + dJ

    Jump sizes have double exponential (asymmetric Laplace) distribution:
    f(y) = p * η₁ * exp(-η₁y) * 1_{y>0} + (1-p) * η₂ * exp(η₂y) * 1_{y<0}

    Parameters:
    - η₁ > 1: Parameter for positive jumps (decay rate)
    - η₂ > 0: Parameter for negative jumps (decay rate)
    - p: Probability of positive jump

    Note: For pricing, see finceptquantlib.models.jump_diffusion.KouJumpDiffusion

    Attributes:
        mu: Drift rate
        sigma: Diffusion volatility
        lambda_: Jump intensity
        p: Probability of upward jump
        eta1: Positive jump decay (must be > 1)
        eta2: Negative jump decay (must be > 0)
        S0: Initial price
    """
    mu: float
    sigma: float
    lambda_: float
    p: float  # Prob of up jump
    eta1: float  # Up jump parameter (> 1)
    eta2: float  # Down jump parameter (> 0)
    S0: float = 100.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError("Volatility must be non-negative")
        if self.lambda_ < 0:
            raise ValueError("Jump intensity must be non-negative")
        if not 0 <= self.p <= 1:
            raise ValueError("Jump probability must be in [0, 1]")
        if self.eta1 <= 1:
            raise ValueError("eta1 must be > 1 for finite expectations")
        if self.eta2 <= 0:
            raise ValueError("eta2 must be > 0")
        if self.S0 <= 0:
            raise ValueError("Initial price must be positive")

    @property
    def k(self) -> float:
        """Expected percentage jump: k = E[e^Y - 1]."""
        # E[e^Y] = p * η₁/(η₁-1) + (1-p) * η₂/(η₂+1)
        E_exp_Y = (self.p * self.eta1 / (self.eta1 - 1) +
                   (1 - self.p) * self.eta2 / (self.eta2 + 1))
        return E_exp_Y - 1

    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> Tuple[List[float], List[List[float]]]:
        """Simulate Kou model paths."""
        rng = random.Random(self.seed) if self.seed is not None else random

        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        times = [i * dt for i in range(n_steps + 1)]

        drift_adj = self.mu - self.lambda_ * self.k - 0.5 * self.sigma ** 2

        paths = []
        for _ in range(n_paths):
            S = [self.S0]
            log_S = math.log(self.S0)

            for _ in range(n_steps):
                # Diffusion
                dW = rng.gauss(0, sqrt_dt)
                log_S += drift_adj * dt + self.sigma * dW

                # Jumps - using centralized sampling
                n_jumps = poisson_sample(self.lambda_ * dt, rng)
                for _ in range(n_jumps):
                    J = double_exponential_sampler(self.p, self.eta1, self.eta2, rng)
                    log_S += J

                S.append(math.exp(log_S))

            paths.append(S)

        return times, paths

    def characteristic_function(self, u: complex, t: float) -> complex:
        """Characteristic function for FFT pricing."""
        iu = 1j * u

        # Diffusion part
        phi_diff = cmath.exp(
            iu * (self.mu - self.lambda_ * self.k - 0.5 * self.sigma ** 2) * t
            - 0.5 * self.sigma ** 2 * u ** 2 * t
        )

        # Jump part
        # E[e^{iuY}] = p*η₁/(η₁-iu) + (1-p)*η₂/(η₂+iu)
        phi_jump_single = (self.p * self.eta1 / (self.eta1 - iu) +
                          (1 - self.p) * self.eta2 / (self.eta2 + iu))
        phi_jump = cmath.exp(self.lambda_ * t * (phi_jump_single - 1))

        return phi_diff * phi_jump


@dataclass
class BatesProcess:
    """
    Bates Stochastic Volatility with Jumps Process (1996).

    Simulation-focused implementation for path generation.

    Combines Heston stochastic volatility with Merton jumps:

    dS/S = (r - λk)dt + √v dW_S + dJ
    dv = κ(θ - v)dt + σ_v √v dW_v

    Corr(dW_S, dW_v) = ρ

    Jump sizes Y ~ N(μ_J, σ_J²)

    Note: For pricing, see finceptquantlib.models.stochastic_vol for related models.

    Attributes:
        r: Risk-free rate
        kappa: Vol mean reversion speed
        theta: Long-term variance
        sigma_v: Vol of vol
        rho: Correlation between S and v
        v0: Initial variance
        lambda_: Jump intensity
        mu_J: Mean log jump
        sigma_J: Jump volatility
        S0: Initial price
    """
    r: float  # Risk-free rate
    kappa: float  # Vol mean reversion
    theta: float  # Long-term variance
    sigma_v: float  # Vol of vol
    rho: float  # Correlation
    v0: float  # Initial variance
    lambda_: float  # Jump intensity
    mu_J: float  # Mean log jump
    sigma_J: float  # Jump volatility
    S0: float = 100.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.theta < 0:
            raise ValueError("Long-term variance must be non-negative")
        if self.sigma_v < 0:
            raise ValueError("Vol of vol must be non-negative")
        if not -1 <= self.rho <= 1:
            raise ValueError("Correlation must be in [-1, 1]")
        if self.v0 < 0:
            raise ValueError("Initial variance must be non-negative")
        if self.lambda_ < 0:
            raise ValueError("Jump intensity must be non-negative")
        if self.sigma_J < 0:
            raise ValueError("Jump volatility must be non-negative")
        if self.S0 <= 0:
            raise ValueError("Initial price must be positive")

    @property
    def feller_satisfied(self) -> bool:
        """Check Feller condition for variance process."""
        return 2 * self.kappa * self.theta >= self.sigma_v ** 2

    @property
    def k(self) -> float:
        """Expected percentage jump."""
        return math.exp(self.mu_J + 0.5 * self.sigma_J ** 2) - 1

    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> Tuple[List[float], List[List[float]], List[List[float]]]:
        """
        Simulate Bates model paths.

        Returns:
            (times, S_paths, v_paths): Time grid, price paths, variance paths
        """
        rng = random.Random(self.seed) if self.seed is not None else random

        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        times = [i * dt for i in range(n_steps + 1)]

        drift_adj = self.r - self.lambda_ * self.k

        S_paths = []
        v_paths = []

        for _ in range(n_paths):
            S = [self.S0]
            v = [self.v0]
            log_S = math.log(self.S0)

            for _ in range(n_steps):
                v_curr = max(v[-1], 0)
                sqrt_v = math.sqrt(v_curr)

                # Correlated Brownian motions
                Z1 = rng.gauss(0, 1)
                Z2 = rng.gauss(0, 1)
                dW_S = sqrt_dt * Z1
                dW_v = sqrt_dt * (self.rho * Z1 + math.sqrt(1 - self.rho ** 2) * Z2)

                # Price process
                log_S += (drift_adj - 0.5 * v_curr) * dt + sqrt_v * dW_S

                # Jumps - using centralized sampling
                n_jumps = poisson_sample(self.lambda_ * dt, rng)
                for _ in range(n_jumps):
                    J = rng.gauss(self.mu_J, self.sigma_J)
                    log_S += J

                S.append(math.exp(log_S))

                # Variance process (Euler with truncation)
                v_new = (v_curr + self.kappa * (self.theta - v_curr) * dt +
                        self.sigma_v * sqrt_v * dW_v)
                v.append(max(v_new, 0))

            S_paths.append(S)
            v_paths.append(v)

        return times, S_paths, v_paths

    def characteristic_function(self, u: complex, t: float) -> complex:
        """
        Characteristic function for FFT pricing.

        Combines Heston CF with jump CF.
        """
        iu = 1j * u

        # Heston part (see Heston 1993)
        sigma2 = self.sigma_v ** 2
        alpha = -0.5 * u ** 2 - 0.5 * iu
        beta = self.kappa - self.rho * self.sigma_v * iu
        gamma = 0.5 * sigma2

        d = (beta ** 2 - 4 * alpha * gamma) ** 0.5
        r_plus = (beta + d) / sigma2
        r_minus = (beta - d) / sigma2

        g = r_minus / r_plus

        exp_dt = cmath.exp(-d * t)
        D = r_minus * (1 - exp_dt) / (1 - g * exp_dt)
        C = self.kappa * (r_minus * t - 2 / sigma2 *
                          cmath.log((1 - g * exp_dt) / (1 - g)))

        phi_heston = cmath.exp(C * self.theta + D * self.v0 +
                              iu * (self.r - self.lambda_ * self.k) * t)

        # Jump part
        phi_jump_single = cmath.exp(
            iu * self.mu_J - 0.5 * self.sigma_J ** 2 * u ** 2
        )
        phi_jump = cmath.exp(self.lambda_ * t * (phi_jump_single - 1))

        return phi_heston * phi_jump


# Alias for backward compatibility
BatesModel = BatesProcess
