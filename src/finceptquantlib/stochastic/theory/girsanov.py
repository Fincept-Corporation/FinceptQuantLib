"""
Girsanov Theorem and Measure Change

Tools for changing probability measures in stochastic calculus:
- Girsanov theorem
- Radon-Nikodym derivative
- Risk-neutral transformation
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


@dataclass
class MeasureChange:
    """
    Represents a change of measure via Girsanov theorem.

    Under P: dX = μ dt + σ dW^P
    Under Q: dX = (μ + σθ) dt + σ dW^Q

    where θ is the market price of risk and
    dW^Q = dW^P - θdt

    The Radon-Nikodym derivative is:
    dQ/dP = exp(-θW^P_T - ½θ²T)
    """
    theta: Callable[[float, float], float]  # Market price of risk θ(x, t)

    def drift_under_Q(
        self,
        mu_P: Callable[[float, float], float],
        sigma: Callable[[float, float], float]
    ) -> Callable[[float, float], float]:
        """
        Compute drift under measure Q.

        μ_Q = μ_P + σθ
        """
        def mu_Q(x: float, t: float) -> float:
            return mu_P(x, t) + sigma(x, t) * self.theta(x, t)

        return mu_Q

    def radon_nikodym(
        self,
        W_path: List[float],
        times: List[float]
    ) -> float:
        """
        Compute Radon-Nikodym derivative dQ/dP along a path.

        For constant θ:
        dQ/dP = exp(-θW_T - ½θ²T)

        For state-dependent θ(X,t):
        dQ/dP = exp(-∫θdW - ½∫θ²dt)
        """
        # Approximate using discretization
        log_RN = 0.0

        for i in range(len(times) - 1):
            dt = times[i + 1] - times[i]
            dW = W_path[i + 1] - W_path[i]
            theta_val = self.theta(W_path[i], times[i])

            log_RN -= theta_val * dW + 0.5 * theta_val ** 2 * dt

        return math.exp(log_RN)


def risk_neutral_drift(
    mu: float,
    r: float,
    sigma: float
) -> float:
    """
    Compute market price of risk for risk-neutral measure.

    Under physical measure: dS/S = μdt + σdW^P
    Under risk-neutral Q: dS/S = rdt + σdW^Q

    θ = (μ - r) / σ

    Args:
        mu: Physical drift
        r: Risk-free rate
        sigma: Volatility

    Returns:
        Market price of risk θ
    """
    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    return (mu - r) / sigma


def esscher_transform(
    characteristic_fn: Callable[[complex], complex],
    h: float
) -> Callable[[complex], complex]:
    """
    Esscher transform of characteristic function.

    Used for pricing under incomplete markets.
    The Esscher measure change is parameterized by h:

    φ^h(u) = φ(u - ih) / φ(-ih)

    Args:
        characteristic_fn: Original characteristic function φ(u)
        h: Esscher parameter

    Returns:
        Transformed characteristic function φ^h
    """
    def phi_h(u: complex) -> complex:
        return characteristic_fn(u - 1j * h) / characteristic_fn(-1j * h)

    return phi_h


def simulate_under_Q(
    mu_P: Callable[[float, float], float],
    sigma: Callable[[float, float], float],
    theta: Callable[[float, float], float],
    X0: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    seed: Optional[int] = None
) -> Tuple[List[float], List[List[float]], List[float]]:
    """
    Simulate paths under measure Q with importance weights.

    Generates paths under P-measure but returns Q-measure weights.

    This is useful for computing expectations under Q:
    E^Q[f(X)] = E^P[f(X) * (dQ/dP)]

    Args:
        mu_P: Drift under P
        sigma: Diffusion (same under P and Q)
        theta: Market price of risk
        X0: Initial value
        T: Terminal time
        n_steps: Number of steps
        n_paths: Number of paths
        seed: Random seed

    Returns:
        (times, paths, weights): Paths simulated under P with Q-weights
    """
    rng = random.Random(seed) if seed is not None else random

    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    times = [i * dt for i in range(n_steps + 1)]

    paths = []
    weights = []

    for _ in range(n_paths):
        X = [X0]
        log_weight = 0.0

        for i in range(n_steps):
            t = times[i]
            x = X[-1]

            # Generate under P
            dW = rng.gauss(0, sqrt_dt)
            X_new = x + mu_P(x, t) * dt + sigma(x, t) * dW
            X.append(X_new)

            # Accumulate weight
            theta_val = theta(x, t)
            log_weight -= theta_val * dW + 0.5 * theta_val ** 2 * dt

        paths.append(X)
        weights.append(math.exp(log_weight))

    return times, paths, weights


def change_of_numeraire(
    discount_bond_vol: Callable[[float, float], float],
    mu_under_Q: Callable[[float, float], float],
    sigma: Callable[[float, float], float],
    rho: float = 0.0
) -> Callable[[float, float], float]:
    """
    Compute drift adjustment for change of numeraire.

    When changing from money market numeraire to T-bond numeraire:
    μ_T = μ_Q + σ ρ σ_P(t,T)

    where σ_P(t,T) is the volatility of the T-bond.

    Used in forward measure pricing.

    Args:
        discount_bond_vol: σ_P(t, T) bond volatility
        mu_under_Q: Drift under risk-neutral Q
        sigma: Asset volatility
        rho: Correlation with bond

    Returns:
        Drift under forward measure
    """
    def mu_forward(x: float, t: float) -> float:
        return (mu_under_Q(x, t) +
                sigma(x, t) * rho * discount_bond_vol(x, t))

    return mu_forward
