"""
Mean-Reverting Processes

Stochastic processes that revert to a long-term mean:
- Ornstein-Uhlenbeck (OU) process
- Cox-Ingersoll-Ross (CIR) process
- Vasicek model
- Hull-White model
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class OrnsteinUhlenbeck:
    """
    Ornstein-Uhlenbeck (OU) process.

    dX = κ(θ - X)dt + σdW

    Parameters:
    - κ (kappa): Mean reversion speed
    - θ (theta): Long-term mean
    - σ (sigma): Volatility

    Properties:
    - Mean reverting to θ
    - Stationary distribution: N(θ, σ²/(2κ))
    - Half-life: ln(2)/κ

    Used for: interest rates, volatility, commodity prices.
    """
    kappa: float  # Mean reversion speed
    theta: float  # Long-term mean
    sigma: float  # Volatility
    X0: float = 0.0  # Initial value
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError("Mean reversion speed must be positive")
        if self.sigma < 0:
            raise ValueError("Volatility must be non-negative")
        self._rng = random.Random(self.seed) if self.seed is not None else random

    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1,
        method: str = "exact"
    ) -> Tuple[List[float], List[List[float]]]:
        """
        Simulate OU process paths.

        Args:
            T: Terminal time
            n_steps: Number of time steps
            n_paths: Number of paths
            method: "exact" (analytical) or "euler" (Euler-Maruyama)

        Returns:
            (times, paths): Time grid and simulated paths
        """
        dt = T / n_steps
        times = [i * dt for i in range(n_steps + 1)]

        if method == "exact":
            return self._simulate_exact(dt, n_steps, n_paths, times)
        else:
            return self._simulate_euler(dt, n_steps, n_paths, times)

    def _simulate_exact(
        self,
        dt: float,
        n_steps: int,
        n_paths: int,
        times: List[float]
    ) -> Tuple[List[float], List[List[float]]]:
        """Exact simulation using analytical transition density."""
        exp_kdt = math.exp(-self.kappa * dt)
        var_dt = (self.sigma ** 2 / (2 * self.kappa)) * (1 - exp_kdt ** 2)
        std_dt = math.sqrt(var_dt)

        paths = []
        for _ in range(n_paths):
            X = [self.X0]
            for _ in range(n_steps):
                mean = self.theta + (X[-1] - self.theta) * exp_kdt
                X_new = mean + std_dt * self._rng.gauss(0, 1)
                X.append(X_new)
            paths.append(X)

        return times, paths

    def _simulate_euler(
        self,
        dt: float,
        n_steps: int,
        n_paths: int,
        times: List[float]
    ) -> Tuple[List[float], List[List[float]]]:
        """Euler-Maruyama discretization."""
        sqrt_dt = math.sqrt(dt)

        paths = []
        for _ in range(n_paths):
            X = [self.X0]
            for _ in range(n_steps):
                dW = self._rng.gauss(0, sqrt_dt)
                drift = self.kappa * (self.theta - X[-1]) * dt
                diffusion = self.sigma * dW
                X.append(X[-1] + drift + diffusion)
            paths.append(X)

        return times, paths

    def mean(self, t: float) -> float:
        """E[X(t)] = θ + (X₀ - θ)e^(-κt)."""
        return self.theta + (self.X0 - self.theta) * math.exp(-self.kappa * t)

    def variance(self, t: float) -> float:
        """Var[X(t)] = (σ²/2κ)(1 - e^(-2κt))."""
        return (self.sigma ** 2 / (2 * self.kappa)) * (1 - math.exp(-2 * self.kappa * t))

    def stationary_mean(self) -> float:
        """Long-run mean = θ."""
        return self.theta

    def stationary_variance(self) -> float:
        """Long-run variance = σ²/(2κ)."""
        return self.sigma ** 2 / (2 * self.kappa)

    def half_life(self) -> float:
        """Time for mean to decay by half: ln(2)/κ."""
        return math.log(2) / self.kappa

    def autocorrelation(self, lag: float) -> float:
        """Autocorrelation at lag τ: e^(-κτ)."""
        return math.exp(-self.kappa * lag)

    def transition_density(self, x: float, t: float) -> float:
        """
        Transition density p(x,t|X0,0).

        X(t)|X(0) ~ N(μ(t), σ²(t))
        """
        mu = self.mean(t)
        var = self.variance(t)
        if var <= 0:
            return float('inf') if abs(x - mu) < 1e-10 else 0.0
        return math.exp(-0.5 * (x - mu) ** 2 / var) / math.sqrt(2 * math.pi * var)


@dataclass
class CoxIngersollRoss:
    """
    Cox-Ingersoll-Ross (CIR) process.

    dr = κ(θ - r)dt + σ√r dW

    Parameters:
    - κ (kappa): Mean reversion speed
    - θ (theta): Long-term mean
    - σ (sigma): Volatility

    Feller condition: 2κθ ≥ σ² ensures r > 0.

    Properties:
    - Non-negative (if Feller condition holds)
    - Mean reverting
    - Non-central chi-squared distribution

    Used for: interest rates, stochastic volatility.
    """
    kappa: float  # Mean reversion speed
    theta: float  # Long-term mean
    sigma: float  # Volatility
    r0: float = 0.05  # Initial value
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError("Mean reversion speed must be positive")
        if self.theta < 0:
            raise ValueError("Long-term mean must be non-negative")
        if self.sigma < 0:
            raise ValueError("Volatility must be non-negative")
        if self.r0 < 0:
            raise ValueError("Initial value must be non-negative")
        self._rng = random.Random(self.seed) if self.seed is not None else random

    @property
    def feller_satisfied(self) -> bool:
        """Check if Feller condition is satisfied."""
        return 2 * self.kappa * self.theta >= self.sigma ** 2

    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1,
        method: str = "euler_full_truncation"
    ) -> Tuple[List[float], List[List[float]]]:
        """
        Simulate CIR process paths.

        Args:
            T: Terminal time
            n_steps: Number of time steps
            n_paths: Number of paths
            method: "euler_full_truncation", "euler_reflection", or "exact"

        Returns:
            (times, paths): Time grid and simulated paths
        """
        dt = T / n_steps
        times = [i * dt for i in range(n_steps + 1)]

        if method == "exact":
            return self._simulate_exact(dt, n_steps, n_paths, times)
        elif method == "euler_reflection":
            return self._simulate_euler_reflection(dt, n_steps, n_paths, times)
        else:  # euler_full_truncation
            return self._simulate_euler_truncation(dt, n_steps, n_paths, times)

    def _simulate_euler_truncation(
        self,
        dt: float,
        n_steps: int,
        n_paths: int,
        times: List[float]
    ) -> Tuple[List[float], List[List[float]]]:
        """Euler with full truncation (set negative to 0)."""
        sqrt_dt = math.sqrt(dt)

        paths = []
        for _ in range(n_paths):
            r = [self.r0]
            for _ in range(n_steps):
                r_curr = max(r[-1], 0)  # Truncate for diffusion
                dW = self._rng.gauss(0, sqrt_dt)
                drift = self.kappa * (self.theta - r_curr) * dt
                diffusion = self.sigma * math.sqrt(r_curr) * dW
                r_new = max(r_curr + drift + diffusion, 0)  # Truncate result
                r.append(r_new)
            paths.append(r)

        return times, paths

    def _simulate_euler_reflection(
        self,
        dt: float,
        n_steps: int,
        n_paths: int,
        times: List[float]
    ) -> Tuple[List[float], List[List[float]]]:
        """Euler with reflection at zero."""
        sqrt_dt = math.sqrt(dt)

        paths = []
        for _ in range(n_paths):
            r = [self.r0]
            for _ in range(n_steps):
                r_curr = r[-1]
                dW = self._rng.gauss(0, sqrt_dt)
                drift = self.kappa * (self.theta - r_curr) * dt
                diffusion = self.sigma * math.sqrt(max(r_curr, 0)) * dW
                r_new = abs(r_curr + drift + diffusion)  # Reflect at zero
                r.append(r_new)
            paths.append(r)

        return times, paths

    def _simulate_exact(
        self,
        dt: float,
        n_steps: int,
        n_paths: int,
        times: List[float]
    ) -> Tuple[List[float], List[List[float]]]:
        """
        Exact simulation using non-central chi-squared.

        X(t)|X(s) = σ²(1-e^(-κ(t-s)))/(4κ) * χ²_d(λ)

        where d = 4κθ/σ² and λ = 4κe^(-κ(t-s))X(s)/(σ²(1-e^(-κ(t-s))))
        """
        exp_kdt = math.exp(-self.kappa * dt)
        c = self.sigma ** 2 * (1 - exp_kdt) / (4 * self.kappa)
        d = 4 * self.kappa * self.theta / (self.sigma ** 2)

        paths = []
        for _ in range(n_paths):
            r = [self.r0]
            for _ in range(n_steps):
                # Non-centrality parameter
                lam = 4 * self.kappa * exp_kdt * r[-1] / (
                    self.sigma ** 2 * (1 - exp_kdt)
                )

                # Generate non-central chi-squared
                r_new = c * self._ncx2_sample(d, lam)
                r.append(max(r_new, 0))
            paths.append(r)

        return times, paths

    def _ncx2_sample(self, df: float, nc: float) -> float:
        """
        Sample from non-central chi-squared.

        Simple approximation using normal.
        For production, use exact algorithms.
        """
        # Johnson approximation for large df+nc
        if df + nc > 30:
            mean = df + nc
            var = 2 * df + 4 * nc
            return max(0, self._rng.gauss(mean, math.sqrt(var)))

        # Direct simulation for small parameters
        # χ²_d(λ) = χ²_(d+2N) where N ~ Poisson(λ/2)
        N = 0
        if nc > 0:
            L = math.exp(-nc / 2)
            p = 1.0
            while p > L:
                N += 1
                p *= self._rng.random()
            N -= 1

        # Sum of (d + 2N) standard normals squared
        result = sum(self._rng.gauss(0, 1) ** 2 for _ in range(int(df) + 2 * N))

        # Handle fractional degrees of freedom
        frac = df - int(df)
        if frac > 0:
            # Approximate fractional part
            result += frac * self._rng.gauss(0, 1) ** 2

        return result

    def mean(self, t: float) -> float:
        """E[r(t)] = θ + (r₀ - θ)e^(-κt)."""
        return self.theta + (self.r0 - self.theta) * math.exp(-self.kappa * t)

    def variance(self, t: float) -> float:
        """
        Var[r(t)] = r₀(σ²/κ)(e^(-κt) - e^(-2κt)) +
                    θ(σ²/2κ)(1 - e^(-κt))²
        """
        exp_kt = math.exp(-self.kappa * t)
        exp_2kt = exp_kt ** 2
        s2_k = self.sigma ** 2 / self.kappa
        s2_2k = self.sigma ** 2 / (2 * self.kappa)

        return (self.r0 * s2_k * (exp_kt - exp_2kt) +
                self.theta * s2_2k * (1 - exp_kt) ** 2)

    def stationary_mean(self) -> float:
        """Long-run mean = θ."""
        return self.theta

    def stationary_variance(self) -> float:
        """Long-run variance = θσ²/(2κ)."""
        return self.theta * self.sigma ** 2 / (2 * self.kappa)

    def bond_price(self, t: float, T: float) -> float:
        """
        Zero-coupon bond price P(t,T) under CIR model.

        P(t,T) = A(t,T) * exp(-B(t,T) * r(t))
        """
        tau = T - t
        if tau <= 0:
            return 1.0

        h = math.sqrt(self.kappa ** 2 + 2 * self.sigma ** 2)

        # A(t,T) and B(t,T)
        exp_h = math.exp(h * tau)
        denom = (h + self.kappa) * (exp_h - 1) + 2 * h

        A = ((2 * h * math.exp((self.kappa + h) * tau / 2)) / denom) ** (
            2 * self.kappa * self.theta / self.sigma ** 2
        )

        B = 2 * (exp_h - 1) / denom

        return A * math.exp(-B * self.r0)


@dataclass
class Vasicek:
    """
    Vasicek interest rate model.

    dr = κ(θ - r)dt + σdW

    Same as OU process, specialized for interest rates.
    Can produce negative rates.

    Note: This is essentially OrnsteinUhlenbeck with r0 instead of X0.
    Provided for convenience and naming convention in rates context.
    """
    kappa: float  # Mean reversion speed
    theta: float  # Long-term mean rate
    sigma: float  # Volatility
    r0: float = 0.05  # Initial rate
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._ou = OrnsteinUhlenbeck(
            kappa=self.kappa,
            theta=self.theta,
            sigma=self.sigma,
            X0=self.r0,
            seed=self.seed
        )

    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> Tuple[List[float], List[List[float]]]:
        """Simulate Vasicek rate paths."""
        return self._ou.simulate(T, n_steps, n_paths, method="exact")

    def mean(self, t: float) -> float:
        """Expected rate at time t."""
        return self._ou.mean(t)

    def variance(self, t: float) -> float:
        """Variance of rate at time t."""
        return self._ou.variance(t)

    def bond_price(self, t: float, T: float) -> float:
        """
        Zero-coupon bond price P(t,T) under Vasicek.

        P(t,T) = A(t,T) * exp(-B(t,T) * r(t))
        """
        tau = T - t
        if tau <= 0:
            return 1.0

        B = (1 - math.exp(-self.kappa * tau)) / self.kappa

        # A(t,T) = exp((B-tau)(κ²θ - σ²/2)/κ² - σ²B²/(4κ))
        A = math.exp(
            (B - tau) * (self.kappa ** 2 * self.theta - 0.5 * self.sigma ** 2)
            / self.kappa ** 2
            - self.sigma ** 2 * B ** 2 / (4 * self.kappa)
        )

        return A * math.exp(-B * self.r0)

    def yield_rate(self, t: float, T: float) -> float:
        """Yield to maturity R(t,T) = -ln(P(t,T))/(T-t)."""
        tau = T - t
        if tau <= 0:
            return self.r0
        return -math.log(self.bond_price(t, T)) / tau


@dataclass
class HullWhite:
    """
    Hull-White (extended Vasicek) model.

    dr = (θ(t) - κr)dt + σdW

    θ(t) is calibrated to fit initial term structure.

    For constant θ: dr = κ(θ/κ - r)dt + σdW = κ(θ̄ - r)dt + σdW
    """
    kappa: float  # Mean reversion speed
    sigma: float  # Volatility
    r0: float = 0.05  # Initial rate
    theta_func: Optional[callable] = None  # θ(t) function
    theta_const: float = 0.05  # Constant θ if no function provided
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError("Mean reversion speed must be positive")
        if self.sigma < 0:
            raise ValueError("Volatility must be non-negative")
        self._rng = random.Random(self.seed) if self.seed is not None else random

    def theta(self, t: float) -> float:
        """Time-dependent drift parameter."""
        if self.theta_func is not None:
            return self.theta_func(t)
        return self.theta_const

    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> Tuple[List[float], List[List[float]]]:
        """
        Simulate Hull-White paths using Euler discretization.
        """
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        times = [i * dt for i in range(n_steps + 1)]

        paths = []
        for _ in range(n_paths):
            r = [self.r0]
            for i in range(n_steps):
                t = times[i]
                dW = self._rng.gauss(0, sqrt_dt)
                drift = (self.theta(t) - self.kappa * r[-1]) * dt
                diffusion = self.sigma * dW
                r.append(r[-1] + drift + diffusion)
            paths.append(r)

        return times, paths
