"""
Elliptical Copulas

Copulas derived from elliptical distributions:
- Gaussian copula
- Student-t copula
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from finceptquantlib.statistics.distributions import Normal


# =============================================================================
# Gaussian Copula
# =============================================================================

@dataclass
class GaussianCopula:
    """
    Gaussian (Normal) copula.

    C(u₁, ..., uₙ) = Φₙ(Φ⁻¹(u₁), ..., Φ⁻¹(uₙ); Σ)

    where Φₙ is the n-dimensional normal CDF and Σ is correlation matrix.

    Properties:
    - Symmetric tail dependence
    - Tail dependence coefficient = 0 (asymptotically independent)
    - Easy to calibrate from correlations

    Attributes:
        correlation: Correlation matrix (as flat list, row-major)
        dim: Dimension
    """
    correlation: List[List[float]]

    def __post_init__(self) -> None:
        self.dim = len(self.correlation)
        self._validate_correlation()

    def _validate_correlation(self) -> None:
        """Validate correlation matrix is positive definite."""
        n = self.dim
        for i in range(n):
            if len(self.correlation[i]) != n:
                raise ValueError("Correlation matrix must be square")
            if abs(self.correlation[i][i] - 1.0) > 1e-10:
                raise ValueError("Diagonal must be 1")
            for j in range(i):
                if abs(self.correlation[i][j] - self.correlation[j][i]) > 1e-10:
                    raise ValueError("Matrix must be symmetric")

    def cdf(self, u: List[float]) -> float:
        """
        Copula CDF.

        C(u) = Φₙ(Φ⁻¹(u₁), ..., Φ⁻¹(uₙ))

        Note: Multivariate normal CDF is computed via approximation.
        """
        if len(u) != self.dim:
            raise ValueError(f"Expected {self.dim} values")

        # Transform to normal
        norm = Normal()
        x = [norm.ppf(max(1e-10, min(1 - 1e-10, ui))) for ui in u]

        # Multivariate normal CDF (approximation for 2D)
        if self.dim == 2:
            return self._bivariate_normal_cdf(x[0], x[1], self.correlation[0][1])

        # For higher dimensions, use Monte Carlo
        return self._monte_carlo_cdf(x, n_samples=10000)

    def _bivariate_normal_cdf(self, x: float, y: float, rho: float) -> float:
        """Bivariate normal CDF approximation."""
        norm = Normal()

        if abs(rho) < 1e-10:
            return norm.cdf(x) * norm.cdf(y)

        # Drezner-Wesolowsky approximation
        if rho < 0:
            return max(0, norm.cdf(x) - self._bivariate_normal_cdf(x, -y, -rho))

        # Use series approximation for positive rho
        a = [0.3253030, 0.4211071, 0.1334425, 0.006374323]
        b = [0.1337764, 0.6243247, 1.3425378, 2.2626645]

        if abs(rho) < 0.925:
            h = x
            k = y
            hk = h * k
            bvn = 0.0

            for i in range(4):
                r = rho * b[i]
                rr = 1 - r * r
                bvn += a[i] * math.exp((r * hk * 2 - h * h - k * k) / (2 * rr)) / math.sqrt(rr)

            return norm.cdf(x) * norm.cdf(y) + bvn * rho / (2 * math.pi)
        else:
            # High correlation case
            if rho > 0:
                return min(norm.cdf(x), norm.cdf(y))
            return max(0, norm.cdf(x) + norm.cdf(y) - 1)

    def _monte_carlo_cdf(self, x: List[float], n_samples: int = 10000) -> float:
        """Monte Carlo estimate of multivariate normal CDF."""
        from finceptquantlib.numerical.linalg import cholesky

        L = cholesky(self.correlation)
        count = 0

        for _ in range(n_samples):
            z = [random.gauss(0, 1) for _ in range(self.dim)]
            y = [sum(L[i][j] * z[j] for j in range(i + 1)) for i in range(self.dim)]

            if all(y[i] <= x[i] for i in range(self.dim)):
                count += 1

        return count / n_samples

    def pdf(self, u: List[float]) -> float:
        """
        Copula density.

        c(u) = |Σ|^(-1/2) * exp(-½ x'(Σ⁻¹ - I)x) / ∏φ(Φ⁻¹(uᵢ))
        """
        norm = Normal()
        x = [norm.ppf(max(1e-10, min(1 - 1e-10, ui))) for ui in u]

        # For 2D
        if self.dim == 2:
            rho = self.correlation[0][1]
            det = 1 - rho ** 2
            if det <= 0:
                return float('inf')

            quad = (x[0] ** 2 + x[1] ** 2 - 2 * rho * x[0] * x[1]) / det

            return math.exp(-0.5 * (quad - x[0] ** 2 - x[1] ** 2)) / math.sqrt(det)

        return 0.0  # Higher dim needs matrix operations

    def sample(self, n: int, seed: Optional[int] = None) -> List[List[float]]:
        """
        Generate samples from copula.

        Args:
            n: Number of samples
            seed: Random seed

        Returns:
            List of n samples, each of dimension dim
        """
        from finceptquantlib.numerical.linalg import cholesky

        rng = random.Random(seed)

        L = cholesky(self.correlation)
        norm = Normal()
        samples = []

        for _ in range(n):
            z = [rng.gauss(0, 1) for _ in range(self.dim)]
            x = [sum(L[i][j] * z[j] for j in range(i + 1)) for i in range(self.dim)]
            u = [norm.cdf(xi) for xi in x]
            samples.append(u)

        return samples

    def tail_dependence(self) -> Tuple[float, float]:
        """
        Tail dependence coefficients.

        Gaussian copula has zero tail dependence.

        Returns:
            (lower, upper) tail dependence
        """
        return 0.0, 0.0

    @classmethod
    def from_kendall_tau(cls, tau: float) -> "GaussianCopula":
        """
        Construct 2D Gaussian copula from Kendall's tau.

        τ = (2/π) * arcsin(ρ)
        ρ = sin(πτ/2)
        """
        rho = math.sin(math.pi * tau / 2)
        return cls([[1.0, rho], [rho, 1.0]])


# =============================================================================
# Student-t Copula
# =============================================================================

@dataclass
class StudentTCopula:
    """
    Student-t copula.

    C(u₁, ..., uₙ) = tₙ,ν(t⁻¹_ν(u₁), ..., t⁻¹_ν(uₙ); Σ)

    Properties:
    - Positive tail dependence
    - Heavier tails than Gaussian
    - Better for modeling extreme co-movements

    Attributes:
        correlation: Correlation matrix
        df: Degrees of freedom
    """
    correlation: List[List[float]]
    df: float

    def __post_init__(self) -> None:
        self.dim = len(self.correlation)
        if self.df <= 0:
            raise ValueError("Degrees of freedom must be positive")

    def sample(self, n: int, seed: Optional[int] = None) -> List[List[float]]:
        """
        Generate samples from Student-t copula.

        Algorithm:
        1. Generate multivariate t from correlated normals / √(χ²/ν)
        2. Transform marginals via t CDF
        """
        from finceptquantlib.numerical.linalg import cholesky
        from finceptquantlib.statistics.distributions import StudentT

        rng = random.Random(seed)

        L = cholesky(self.correlation)
        t_dist = StudentT(df=self.df)
        samples = []

        for _ in range(n):
            # Generate normal vector
            z = [rng.gauss(0, 1) for _ in range(self.dim)]
            x = [sum(L[i][j] * z[j] for j in range(i + 1)) for i in range(self.dim)]

            # Generate chi-squared for denominator
            s = sum(rng.gauss(0, 1) ** 2 for _ in range(int(self.df))) / self.df

            # Scale to get t-distributed
            t = [xi / math.sqrt(s) for xi in x]

            # Transform to uniform via t CDF
            u = [t_dist.cdf(ti) for ti in t]
            samples.append(u)

        return samples

    def tail_dependence(self) -> Tuple[float, float]:
        """
        Tail dependence coefficient for 2D case.

        λ = 2 * t_{ν+1}(-√((ν+1)(1-ρ)/(1+ρ)))

        Upper and lower are equal (symmetric).
        """
        if self.dim != 2:
            raise ValueError("Tail dependence only defined for 2D")

        rho = self.correlation[0][1]
        from finceptquantlib.statistics.distributions import StudentT

        t_dist = StudentT(df=self.df + 1)
        arg = -math.sqrt((self.df + 1) * (1 - rho) / (1 + rho))
        lam = 2 * t_dist.cdf(arg)

        return lam, lam

    @classmethod
    def from_kendall_tau(cls, tau: float, df: float) -> "StudentTCopula":
        """
        Construct 2D Student-t copula from Kendall's tau.

        Same relation as Gaussian: ρ = sin(πτ/2)
        """
        rho = math.sin(math.pi * tau / 2)
        return cls([[1.0, rho], [rho, 1.0]], df=df)
