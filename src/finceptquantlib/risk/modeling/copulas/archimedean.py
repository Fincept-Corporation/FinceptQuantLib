"""
Archimedean Copulas

One-parameter copulas with specific dependence structures:
- Clayton copula (lower tail dependence)
- Gumbel copula (upper tail dependence)
- Frank copula (no tail dependence)
- Joe copula (upper tail dependence)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


# =============================================================================
# Clayton Copula
# =============================================================================

@dataclass
class ClaytonCopula:
    """
    Clayton copula.

    C(u, v) = (u^(-θ) + v^(-θ) - 1)^(-1/θ)

    Properties:
    - Lower tail dependence: λ_L = 2^(-1/θ)
    - Upper tail dependence: λ_U = 0
    - θ > 0 for positive dependence
    - θ → 0 gives independence
    - θ → ∞ gives comonotonicity

    Attributes:
        theta: Dependence parameter (θ > 0)
    """
    theta: float

    def __post_init__(self) -> None:
        if self.theta <= 0:
            raise ValueError("Clayton parameter theta must be positive")

    def cdf(self, u: float, v: float) -> float:
        """
        Copula CDF.

        C(u, v) = (u^(-θ) + v^(-θ) - 1)^(-1/θ)
        """
        u = max(1e-10, min(1 - 1e-10, u))
        v = max(1e-10, min(1 - 1e-10, v))

        val = u ** (-self.theta) + v ** (-self.theta) - 1
        if val <= 0:
            return 0.0
        return val ** (-1 / self.theta)

    def pdf(self, u: float, v: float) -> float:
        """
        Copula density.

        c(u, v) = (1 + θ) * (uv)^(-1-θ) * (u^(-θ) + v^(-θ) - 1)^(-2-1/θ)
        """
        u = max(1e-10, min(1 - 1e-10, u))
        v = max(1e-10, min(1 - 1e-10, v))

        term1 = (1 + self.theta) * (u * v) ** (-1 - self.theta)
        val = u ** (-self.theta) + v ** (-self.theta) - 1
        if val <= 0:
            return 0.0
        term2 = val ** (-2 - 1 / self.theta)

        return term1 * term2

    def sample(self, n: int, seed: Optional[int] = None) -> List[List[float]]:
        """
        Generate samples using conditional inverse method.

        Algorithm:
        1. Generate u ~ U(0,1)
        2. Generate t ~ U(0,1)
        3. v = ((t^(-θ/(1+θ)) - 1) * u^(-θ) + 1)^(-1/θ)
        """
        rng = random.Random(seed)

        samples = []
        for _ in range(n):
            u = rng.random()
            t = rng.random()

            # Conditional inverse
            a = t ** (-self.theta / (1 + self.theta)) - 1
            b = u ** (-self.theta)
            v = (a * b + 1) ** (-1 / self.theta)

            samples.append([u, v])

        return samples

    def tail_dependence(self) -> Tuple[float, float]:
        """
        Tail dependence coefficients.

        Lower: λ_L = 2^(-1/θ)
        Upper: λ_U = 0
        """
        lower = 2 ** (-1 / self.theta)
        return lower, 0.0

    def generator(self, t: float) -> float:
        """
        Generator function φ(t) = (t^(-θ) - 1) / θ
        """
        return (t ** (-self.theta) - 1) / self.theta

    def generator_inverse(self, s: float) -> float:
        """
        Inverse generator φ^(-1)(s) = (1 + θs)^(-1/θ)
        """
        return (1 + self.theta * s) ** (-1 / self.theta)

    @classmethod
    def from_kendall_tau(cls, tau: float) -> "ClaytonCopula":
        """
        Construct from Kendall's tau.

        τ = θ / (θ + 2)
        θ = 2τ / (1 - τ)
        """
        if tau <= 0 or tau >= 1:
            raise ValueError("Kendall's tau must be in (0, 1) for Clayton")
        theta = 2 * tau / (1 - tau)
        return cls(theta)


# =============================================================================
# Gumbel Copula
# =============================================================================

@dataclass
class GumbelCopula:
    """
    Gumbel copula.

    C(u, v) = exp(-((−ln u)^θ + (−ln v)^θ)^(1/θ))

    Properties:
    - Lower tail dependence: λ_L = 0
    - Upper tail dependence: λ_U = 2 - 2^(1/θ)
    - θ ≥ 1 required
    - θ = 1 gives independence
    - θ → ∞ gives comonotonicity

    Attributes:
        theta: Dependence parameter (θ ≥ 1)
    """
    theta: float

    def __post_init__(self) -> None:
        if self.theta < 1:
            raise ValueError("Gumbel parameter theta must be >= 1")

    def cdf(self, u: float, v: float) -> float:
        """
        Copula CDF.

        C(u, v) = exp(-((−ln u)^θ + (−ln v)^θ)^(1/θ))
        """
        u = max(1e-10, min(1 - 1e-10, u))
        v = max(1e-10, min(1 - 1e-10, v))

        lu = -math.log(u)
        lv = -math.log(v)

        return math.exp(-((lu ** self.theta + lv ** self.theta) ** (1 / self.theta)))

    def pdf(self, u: float, v: float) -> float:
        """
        Copula density.
        """
        u = max(1e-10, min(1 - 1e-10, u))
        v = max(1e-10, min(1 - 1e-10, v))

        lu = -math.log(u)
        lv = -math.log(v)

        A = (lu ** self.theta + lv ** self.theta) ** (1 / self.theta)
        C = math.exp(-A)

        # Derivative terms
        term1 = C / (u * v)
        term2 = (lu * lv) ** (self.theta - 1)
        term3 = A ** (2 - 2 * self.theta)
        term4 = A + self.theta - 1

        return term1 * term2 * term3 * term4

    def sample(self, n: int, seed: Optional[int] = None) -> List[List[float]]:
        """
        Generate samples using Marshall-Olkin algorithm.

        Uses stable distribution for mixing variable.
        """
        rng = random.Random(seed)

        samples = []
        alpha = 1 / self.theta

        for _ in range(n):
            # Positive alpha-stable mixing variable with Laplace transform
            # exp(-s^alpha) (Kanter's representation, U ~ Uniform(0, pi)).
            # The symmetric Chambers-Mallows-Stuck form can go negative,
            # which made (E / V) ** alpha complex.
            W = -math.log(rng.random())
            U = math.pi * rng.random()

            if abs(alpha - 1) < 1e-10:
                V = 1.0  # theta = 1: independence copula
            else:
                t1 = math.sin(alpha * U) / (math.sin(U) ** (1 / alpha))
                t2 = (math.sin((1 - alpha) * U) / W) ** ((1 - alpha) / alpha)
                V = t1 * t2

            # Generate exponentials
            E1 = -math.log(rng.random())
            E2 = -math.log(rng.random())

            # Transform
            u = math.exp(-(E1 / V) ** alpha)
            v = math.exp(-(E2 / V) ** alpha)

            samples.append([u, v])

        return samples

    def tail_dependence(self) -> Tuple[float, float]:
        """
        Tail dependence coefficients.

        Lower: λ_L = 0
        Upper: λ_U = 2 - 2^(1/θ)
        """
        upper = 2 - 2 ** (1 / self.theta)
        return 0.0, upper

    def generator(self, t: float) -> float:
        """
        Generator function φ(t) = (-ln t)^θ
        """
        if t <= 0:
            return float('inf')
        return (-math.log(t)) ** self.theta

    def generator_inverse(self, s: float) -> float:
        """
        Inverse generator φ^(-1)(s) = exp(-s^(1/θ))
        """
        if s < 0:
            return 1.0
        return math.exp(-(s ** (1 / self.theta)))

    @classmethod
    def from_kendall_tau(cls, tau: float) -> "GumbelCopula":
        """
        Construct from Kendall's tau.

        τ = 1 - 1/θ
        θ = 1/(1-τ)
        """
        if tau < 0 or tau >= 1:
            raise ValueError("Kendall's tau must be in [0, 1) for Gumbel")
        theta = 1 / (1 - tau) if tau > 0 else 1.0
        return cls(theta)


# =============================================================================
# Frank Copula
# =============================================================================

@dataclass
class FrankCopula:
    """
    Frank copula.

    C(u, v) = -1/θ * ln(1 + (e^(-θu) - 1)(e^(-θv) - 1)/(e^(-θ) - 1))

    Properties:
    - Lower tail dependence: λ_L = 0
    - Upper tail dependence: λ_U = 0
    - θ ∈ ℝ\\{0}
    - θ > 0 for positive dependence
    - θ < 0 for negative dependence
    - θ → 0 gives independence

    Attributes:
        theta: Dependence parameter (θ ≠ 0)
    """
    theta: float

    def __post_init__(self) -> None:
        if abs(self.theta) < 1e-10:
            raise ValueError("Frank parameter theta cannot be zero")

    def cdf(self, u: float, v: float) -> float:
        """
        Copula CDF.
        """
        u = max(1e-10, min(1 - 1e-10, u))
        v = max(1e-10, min(1 - 1e-10, v))

        if abs(self.theta) < 1e-8:
            return u * v  # Independence limit

        num = (math.exp(-self.theta * u) - 1) * (math.exp(-self.theta * v) - 1)
        den = math.exp(-self.theta) - 1

        return -math.log(1 + num / den) / self.theta

    def pdf(self, u: float, v: float) -> float:
        """
        Copula density.
        """
        u = max(1e-10, min(1 - 1e-10, u))
        v = max(1e-10, min(1 - 1e-10, v))

        if abs(self.theta) < 1e-8:
            return 1.0  # Independence

        e_u = math.exp(-self.theta * u)
        e_v = math.exp(-self.theta * v)
        e_theta = math.exp(-self.theta)

        num = -self.theta * (e_theta - 1) * e_u * e_v
        term = (e_theta - 1) + (e_u - 1) * (e_v - 1)
        den = term ** 2

        return num / den

    def sample(self, n: int, seed: Optional[int] = None) -> List[List[float]]:
        """
        Generate samples using conditional inverse method.
        """
        rng = random.Random(seed)

        samples = []
        for _ in range(n):
            u = rng.random()
            t = rng.random()

            # Conditional inverse: solve for v given u and t
            # Use Newton's method
            v = 0.5
            for _ in range(20):
                e_u = math.exp(-self.theta * u)
                e_v = math.exp(-self.theta * v)
                e_theta = math.exp(-self.theta)

                # Conditional CDF
                num = (e_u - 1) * (e_v - 1)
                den = (e_theta - 1) + (e_u - 1) * (e_v - 1)
                C_uv = 1 + num / den

                # Derivative w.r.t. v
                dCdv = -self.theta * (e_u - 1) * e_v * (e_theta - 1) / (den ** 2)

                # Newton step
                if abs(dCdv) < 1e-12:
                    break
                v_new = v - (C_uv - t) / dCdv
                v_new = max(1e-10, min(1 - 1e-10, v_new))

                if abs(v_new - v) < 1e-10:
                    break
                v = v_new

            samples.append([u, v])

        return samples

    def tail_dependence(self) -> Tuple[float, float]:
        """
        Tail dependence coefficients.

        Frank copula has no tail dependence.
        """
        return 0.0, 0.0

    def generator(self, t: float) -> float:
        """
        Generator function φ(t) = -ln((e^(-θt) - 1)/(e^(-θ) - 1))
        """
        if t <= 0:
            return float('inf')
        if t >= 1:
            return 0.0

        num = math.exp(-self.theta * t) - 1
        den = math.exp(-self.theta) - 1
        return -math.log(num / den)

    def generator_inverse(self, s: float) -> float:
        """
        Inverse generator φ^(-1)(s) = -1/θ * ln(1 + e^(-s)(e^(-θ) - 1))
        """
        if s < 0:
            return 1.0

        val = 1 + math.exp(-s) * (math.exp(-self.theta) - 1)
        if val <= 0:
            return 0.0
        return -math.log(val) / self.theta

    @classmethod
    def from_kendall_tau(cls, tau: float) -> "FrankCopula":
        """
        Construct from Kendall's tau.

        τ = 1 - 4/θ * (1 - D₁(θ))

        where D₁ is first Debye function. Solved numerically.
        """
        if abs(tau) >= 1:
            raise ValueError("Kendall's tau must be in (-1, 1)")

        # Solve numerically
        def debye1(x: float) -> float:
            """First Debye function D₁(x) = 1/x * ∫₀ˣ t/(eᵗ-1) dt"""
            if abs(x) < 1e-10:
                return 1.0
            # Numerical integration
            n_points = 100
            result = 0.0
            for i in range(1, n_points + 1):
                t = x * i / n_points
                if t < 700:  # Avoid overflow
                    result += t / (math.exp(t) - 1) * (x / n_points)
            return result / x

        def kendall_from_theta(theta: float) -> float:
            if abs(theta) < 1e-10:
                return 0.0
            return 1 - 4 / theta * (1 - debye1(theta))

        # Bisection search
        low, high = -30.0, 30.0
        if tau > 0:
            low = 0.01
        else:
            high = -0.01

        for _ in range(50):
            mid = (low + high) / 2
            tau_mid = kendall_from_theta(mid)

            if tau_mid < tau:
                if tau > 0:
                    low = mid
                else:
                    high = mid
            else:
                if tau > 0:
                    high = mid
                else:
                    low = mid

            if abs(tau_mid - tau) < 1e-6:
                break

        return cls((low + high) / 2)


# =============================================================================
# Joe Copula
# =============================================================================

@dataclass
class JoeCopula:
    """
    Joe copula.

    C(u, v) = 1 - ((1-u)^θ + (1-v)^θ - (1-u)^θ(1-v)^θ)^(1/θ)

    Properties:
    - Lower tail dependence: λ_L = 0
    - Upper tail dependence: λ_U = 2 - 2^(1/θ)
    - θ ≥ 1 required
    - θ = 1 gives independence
    - θ → ∞ gives comonotonicity

    Attributes:
        theta: Dependence parameter (θ ≥ 1)
    """
    theta: float

    def __post_init__(self) -> None:
        if self.theta < 1:
            raise ValueError("Joe parameter theta must be >= 1")

    def cdf(self, u: float, v: float) -> float:
        """
        Copula CDF.
        """
        u = max(1e-10, min(1 - 1e-10, u))
        v = max(1e-10, min(1 - 1e-10, v))

        ub = (1 - u) ** self.theta
        vb = (1 - v) ** self.theta

        return 1 - (ub + vb - ub * vb) ** (1 / self.theta)

    def pdf(self, u: float, v: float) -> float:
        """
        Copula density.
        """
        u = max(1e-10, min(1 - 1e-10, u))
        v = max(1e-10, min(1 - 1e-10, v))

        ub = (1 - u) ** self.theta
        vb = (1 - v) ** self.theta
        A = ub + vb - ub * vb

        if A <= 0:
            return 0.0

        term1 = A ** (1 / self.theta - 2)
        term2 = ((1 - u) ** (self.theta - 1)) * ((1 - v) ** (self.theta - 1))
        term3 = self.theta - 1 + A

        return term1 * term2 * term3

    def sample(self, n: int, seed: Optional[int] = None) -> List[List[float]]:
        """
        Generate samples using conditional inverse method.
        """
        rng = random.Random(seed)

        samples = []
        for _ in range(n):
            u = rng.random()
            t = rng.random()

            # Solve for v using Newton's method
            v = 0.5
            ub = (1 - u) ** self.theta

            for _ in range(20):
                vb = (1 - v) ** self.theta
                A = ub + vb - ub * vb

                if A <= 0:
                    break

                # Conditional CDF C(v|u)
                C_uv = 1 - (A ** (1 / self.theta - 1)) * (vb - ub * vb)

                # Derivative
                dCdv = (A ** (1 / self.theta - 2)) * ((1 - v) ** (self.theta - 1)) * (
                    self.theta * (1 - ub) * A ** (-1) * (vb - ub * vb) +
                    (1 - ub)
                )

                if abs(dCdv) < 1e-12:
                    break

                v_new = v - (C_uv - t) / dCdv
                v_new = max(1e-10, min(1 - 1e-10, v_new))

                if abs(v_new - v) < 1e-10:
                    break
                v = v_new

            samples.append([u, v])

        return samples

    def tail_dependence(self) -> Tuple[float, float]:
        """
        Tail dependence coefficients.

        Lower: λ_L = 0
        Upper: λ_U = 2 - 2^(1/θ)
        """
        upper = 2 - 2 ** (1 / self.theta)
        return 0.0, upper

    def generator(self, t: float) -> float:
        """
        Generator function φ(t) = -ln(1 - (1-t)^θ)
        """
        if t <= 0:
            return float('inf')
        if t >= 1:
            return 0.0

        val = 1 - (1 - t) ** self.theta
        if val <= 0:
            return float('inf')
        return -math.log(val)

    def generator_inverse(self, s: float) -> float:
        """
        Inverse generator φ^(-1)(s) = 1 - (1 - e^(-s))^(1/θ)
        """
        if s < 0:
            return 1.0

        val = 1 - math.exp(-s)
        if val <= 0:
            return 1.0
        return 1 - val ** (1 / self.theta)

    @classmethod
    def from_kendall_tau(cls, tau: float) -> "JoeCopula":
        """
        Construct from Kendall's tau (numerical approximation).

        No closed-form solution; use bisection.
        """
        if tau < 0 or tau >= 1:
            raise ValueError("Kendall's tau must be in [0, 1) for Joe")

        def kendall_from_theta(theta: float) -> float:
            """Numerical approximation of Kendall's tau."""
            # τ = 1 + 4/θ² * ∫₀¹ t*ln(t)*(1-t)^(2(1-θ)/θ) dt
            # Approximation: use Simpson's rule
            n_points = 100
            result = 0.0
            h = 1.0 / n_points

            for i in range(1, n_points):
                t = i * h
                if t > 0 and t < 1:
                    val = t * math.log(t) * ((1 - t) ** (2 * (1 - theta) / theta))
                    result += val

            result *= h
            return 1 + 4 / (theta ** 2) * result

        # Bisection search
        low, high = 1.0, 50.0

        for _ in range(50):
            mid = (low + high) / 2
            tau_mid = kendall_from_theta(mid)

            if tau_mid < tau:
                low = mid
            else:
                high = mid

            if abs(tau_mid - tau) < 1e-6:
                break

        return cls((low + high) / 2)
