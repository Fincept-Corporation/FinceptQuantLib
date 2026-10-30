"""
Utility Functions

Standard utility functions used in economics and finance:
- CRRA (Constant Relative Risk Aversion)
- CARA (Constant Absolute Risk Aversion)
- Quadratic
- Logarithmic
- Prospect Theory value function

Each utility function provides:
- u(x): Utility value
- u'(x): Marginal utility
- u''(x): Second derivative (for risk aversion)
- Arrow-Pratt measures of risk aversion
- Certainty equivalent computation
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple


class UtilityFunction(ABC):
    """Abstract base class for utility functions."""

    @abstractmethod
    def __call__(self, x: float) -> float:
        """Evaluate utility at x."""
        pass

    @abstractmethod
    def marginal(self, x: float) -> float:
        """First derivative u'(x)."""
        pass

    @abstractmethod
    def second_derivative(self, x: float) -> float:
        """Second derivative u''(x)."""
        pass

    def absolute_risk_aversion(self, x: float) -> float:
        """
        Arrow-Pratt absolute risk aversion.

        A(x) = -u''(x) / u'(x)
        """
        u_prime = self.marginal(x)
        if abs(u_prime) < 1e-15:
            return float('inf')
        return -self.second_derivative(x) / u_prime

    def relative_risk_aversion(self, x: float) -> float:
        """
        Arrow-Pratt relative risk aversion.

        R(x) = -x * u''(x) / u'(x)
        """
        return x * self.absolute_risk_aversion(x)

    def certainty_equivalent(
        self,
        outcomes: List[float],
        probabilities: Optional[List[float]] = None
    ) -> float:
        """
        Compute certainty equivalent of a lottery.

        CE is x such that u(CE) = E[u(X)]
        """
        n = len(outcomes)
        if probabilities is None:
            probabilities = [1 / n] * n

        expected_utility = sum(p * self(x) for x, p in zip(outcomes, probabilities))
        return self.inverse(expected_utility)

    def risk_premium(
        self,
        outcomes: List[float],
        probabilities: Optional[List[float]] = None
    ) -> float:
        """
        Risk premium: E[X] - CE

        The amount a risk-averse agent would pay to avoid risk.
        """
        n = len(outcomes)
        if probabilities is None:
            probabilities = [1 / n] * n

        expected_value = sum(p * x for x, p in zip(outcomes, probabilities))
        ce = self.certainty_equivalent(outcomes, probabilities)

        return expected_value - ce

    @abstractmethod
    def inverse(self, u: float) -> float:
        """Inverse utility function."""
        pass


@dataclass
class CRRA(UtilityFunction):
    """
    Constant Relative Risk Aversion utility.

    u(x) = x^(1-γ) / (1-γ)  for γ ≠ 1
    u(x) = ln(x)             for γ = 1

    Properties:
    - R(x) = γ (constant)
    - A(x) = γ/x (decreasing in wealth)

    Attributes:
        gamma: Risk aversion coefficient (γ > 0)
    """
    gamma: float = 2.0

    def __post_init__(self) -> None:
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")

    def __call__(self, x: float) -> float:
        if x <= 0:
            return float('-inf')

        if abs(self.gamma - 1) < 1e-10:
            return math.log(x)
        return (x ** (1 - self.gamma) - 1) / (1 - self.gamma)

    def marginal(self, x: float) -> float:
        if x <= 0:
            return float('inf')
        return x ** (-self.gamma)

    def second_derivative(self, x: float) -> float:
        if x <= 0:
            return float('-inf')
        return -self.gamma * x ** (-self.gamma - 1)

    def relative_risk_aversion(self, x: float) -> float:
        return self.gamma

    def absolute_risk_aversion(self, x: float) -> float:
        if x <= 0:
            return float('inf')
        return self.gamma / x

    def inverse(self, u: float) -> float:
        if abs(self.gamma - 1) < 1e-10:
            return math.exp(u)
        val = u * (1 - self.gamma) + 1
        if val <= 0:
            return 1e-10
        return val ** (1 / (1 - self.gamma))


@dataclass
class CARA(UtilityFunction):
    """
    Constant Absolute Risk Aversion utility.

    u(x) = -exp(-αx) / α  for α ≠ 0
    u(x) = x              for α = 0

    Properties:
    - A(x) = α (constant)
    - R(x) = αx (increasing in wealth)

    Attributes:
        alpha: Risk aversion coefficient (α > 0)
    """
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")

    def __call__(self, x: float) -> float:
        if abs(self.alpha) < 1e-10:
            return x
        return -math.exp(-self.alpha * x) / self.alpha

    def marginal(self, x: float) -> float:
        if abs(self.alpha) < 1e-10:
            return 1.0
        return math.exp(-self.alpha * x)

    def second_derivative(self, x: float) -> float:
        if abs(self.alpha) < 1e-10:
            return 0.0
        return -self.alpha * math.exp(-self.alpha * x)

    def absolute_risk_aversion(self, x: float) -> float:
        return self.alpha

    def relative_risk_aversion(self, x: float) -> float:
        return self.alpha * x

    def inverse(self, u: float) -> float:
        if abs(self.alpha) < 1e-10:
            return u
        val = -self.alpha * u
        if val <= 0:
            return float('inf')
        return -math.log(val) / self.alpha


@dataclass
class QuadraticUtility(UtilityFunction):
    """
    Quadratic utility.

    u(x) = x - (b/2)x²

    Properties:
    - Mean-variance preferences
    - Increasing risk aversion (IARA)
    - Satiation point at x = 1/b

    Attributes:
        b: Risk aversion coefficient (b > 0)
    """
    b: float = 0.01

    def __post_init__(self) -> None:
        if self.b <= 0:
            raise ValueError("b must be positive")

    def __call__(self, x: float) -> float:
        return x - 0.5 * self.b * x ** 2

    def marginal(self, x: float) -> float:
        return 1 - self.b * x

    def second_derivative(self, x: float) -> float:
        return -self.b

    def absolute_risk_aversion(self, x: float) -> float:
        denom = 1 - self.b * x
        if abs(denom) < 1e-15:
            return float('inf')
        return self.b / denom

    def satiation_point(self) -> float:
        """Point where marginal utility is zero."""
        return 1 / self.b

    def inverse(self, u: float) -> float:
        # Solve x - (b/2)x² = u
        # bx²/2 - x + u = 0
        # x = (1 ± √(1 - 2bu)) / b
        discriminant = 1 - 2 * self.b * u
        if discriminant < 0:
            return float('nan')
        return (1 - math.sqrt(discriminant)) / self.b


@dataclass
class LogUtility(UtilityFunction):
    """
    Logarithmic utility (special case of CRRA with γ=1).

    u(x) = ln(x)

    Properties:
    - R(x) = 1 (constant)
    - Myopic optimality in portfolio choice
    """

    def __call__(self, x: float) -> float:
        if x <= 0:
            return float('-inf')
        return math.log(x)

    def marginal(self, x: float) -> float:
        if x <= 0:
            return float('inf')
        return 1 / x

    def second_derivative(self, x: float) -> float:
        if x <= 0:
            return float('-inf')
        return -1 / (x ** 2)

    def relative_risk_aversion(self, x: float) -> float:
        return 1.0

    def absolute_risk_aversion(self, x: float) -> float:
        if x <= 0:
            return float('inf')
        return 1 / x

    def inverse(self, u: float) -> float:
        return math.exp(u)


@dataclass
class ProspectTheoryValue(UtilityFunction):
    """
    Prospect Theory value function (Kahneman & Tversky).

    v(x) = x^α           for x ≥ 0 (gains)
    v(x) = -λ(-x)^β      for x < 0 (losses)

    Properties:
    - Reference-dependent
    - Loss aversion (λ > 1)
    - Diminishing sensitivity (α, β < 1)

    Attributes:
        alpha: Gain curvature (0 < α ≤ 1)
        beta: Loss curvature (0 < β ≤ 1)
        lambda_: Loss aversion coefficient (λ > 1)
    """
    alpha: float = 0.88
    beta: float = 0.88
    lambda_: float = 2.25

    def __post_init__(self) -> None:
        if not (0 < self.alpha <= 1):
            raise ValueError("alpha must be in (0, 1]")
        if not (0 < self.beta <= 1):
            raise ValueError("beta must be in (0, 1]")
        if self.lambda_ <= 0:
            raise ValueError("lambda must be positive")

    def __call__(self, x: float) -> float:
        if x >= 0:
            return x ** self.alpha
        return -self.lambda_ * ((-x) ** self.beta)

    def marginal(self, x: float) -> float:
        if x > 0:
            return self.alpha * x ** (self.alpha - 1)
        elif x < 0:
            return self.lambda_ * self.beta * ((-x) ** (self.beta - 1))
        else:
            # At reference point, use right derivative
            return self.alpha if self.alpha < 1 else 1.0

    def second_derivative(self, x: float) -> float:
        if x > 0:
            return self.alpha * (self.alpha - 1) * x ** (self.alpha - 2)
        elif x < 0:
            return -self.lambda_ * self.beta * (self.beta - 1) * ((-x) ** (self.beta - 2))
        return 0.0

    def inverse(self, v: float) -> float:
        if v >= 0:
            return v ** (1 / self.alpha)
        return -((-v / self.lambda_) ** (1 / self.beta))

    def loss_aversion_ratio(self) -> float:
        """Ratio of marginal utility of losses to gains at reference point."""
        return self.lambda_


@dataclass
class HyperbolicAbsoluteRiskAversion(UtilityFunction):
    """
    HARA (Hyperbolic Absolute Risk Aversion) utility.

    u(x) = (1-γ)/γ * (ax/(1-γ) + b)^γ

    This is the most general form, nesting:
    - CRRA: a=1, b=0
    - CARA: γ→-∞
    - Quadratic: γ=-1

    Attributes:
        gamma: Curvature parameter
        a: Scale parameter (a > 0)
        b: Subsistence level
    """
    gamma: float = 0.5
    a: float = 1.0
    b: float = 0.0

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise ValueError("a must be positive")

    def _core(self, x: float) -> float:
        return self.a * x / (1 - self.gamma) + self.b

    def __call__(self, x: float) -> float:
        core = self._core(x)
        if core <= 0:
            return float('-inf')

        if abs(self.gamma) < 1e-10:
            return math.log(core)
        return (1 - self.gamma) / self.gamma * (core ** self.gamma)

    def marginal(self, x: float) -> float:
        core = self._core(x)
        if core <= 0:
            return float('inf')
        return self.a * core ** (self.gamma - 1)

    def second_derivative(self, x: float) -> float:
        core = self._core(x)
        if core <= 0:
            return float('-inf')
        return self.a ** 2 * (self.gamma - 1) / (1 - self.gamma) * core ** (self.gamma - 2)

    def absolute_risk_aversion(self, x: float) -> float:
        core = self._core(x)
        if core <= 0:
            return float('inf')
        return (1 - self.gamma) / core

    def relative_risk_aversion(self, x: float) -> float:
        return x * self.absolute_risk_aversion(x)

    def inverse(self, u: float) -> float:
        if abs(self.gamma) < 1e-10:
            core = math.exp(u)
        else:
            val = u * self.gamma / (1 - self.gamma)
            if val <= 0:
                return float('-inf')
            core = val ** (1 / self.gamma)
        return (core - self.b) * (1 - self.gamma) / self.a


def expected_utility(
    utility: UtilityFunction,
    outcomes: List[float],
    probabilities: Optional[List[float]] = None
) -> float:
    """
    Compute expected utility of a lottery.

    E[u(X)] = Σ p_i * u(x_i)
    """
    n = len(outcomes)
    if probabilities is None:
        probabilities = [1 / n] * n

    return sum(p * utility(x) for x, p in zip(outcomes, probabilities))


def certainty_equivalent_approximation(
    mean: float,
    variance: float,
    risk_aversion: float
) -> float:
    """
    Second-order approximation of certainty equivalent.

    CE ≈ μ - (1/2) * A * σ²

    where A is absolute risk aversion.
    """
    return mean - 0.5 * risk_aversion * variance


def stochastic_dominance_first(
    cdf_a: List[Tuple[float, float]],
    cdf_b: List[Tuple[float, float]]
) -> bool:
    """
    Test if distribution A first-order stochastically dominates B.

    A FSD B iff F_A(x) ≤ F_B(x) for all x.

    Args:
        cdf_a: CDF of A as list of (x, F(x)) pairs
        cdf_b: CDF of B as list of (x, F(x)) pairs
    """
    # Combine and sort all x values
    all_x = sorted({x for x, _ in cdf_a + cdf_b})

    for x in all_x:
        # Find CDF values at x
        fa = max((f for xi, f in cdf_a if xi <= x), default=0)
        fb = max((f for xi, f in cdf_b if xi <= x), default=0)

        if fa > fb + 1e-10:
            return False

    return True


def stochastic_dominance_second(
    cdf_a: List[Tuple[float, float]],
    cdf_b: List[Tuple[float, float]]
) -> bool:
    """
    Test if distribution A second-order stochastically dominates B.

    A SSD B iff ∫_{-∞}^x F_A(t)dt ≤ ∫_{-∞}^x F_B(t)dt for all x.

    All risk-averse agents prefer A to B.
    """
    # Combine and sort all x values
    all_x = sorted({x for x, _ in cdf_a + cdf_b})

    integral_a = 0.0
    integral_b = 0.0
    prev_x = all_x[0] if all_x else 0

    for x in all_x:
        fa = max((f for xi, f in cdf_a if xi <= x), default=0)
        fb = max((f for xi, f in cdf_b if xi <= x), default=0)

        dx = x - prev_x
        integral_a += fa * dx
        integral_b += fb * dx

        if integral_a > integral_b + 1e-10:
            return False

        prev_x = x

    return True
