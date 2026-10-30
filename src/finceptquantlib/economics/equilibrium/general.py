"""
General Equilibrium Theory

Arrow-Debreu style general equilibrium models:
- Exchange economies
- Walrasian equilibrium
- Pareto efficiency
- Welfare theorems
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


@dataclass
class Consumer:
    """
    Consumer in an exchange economy.

    Attributes:
        id: Consumer identifier
        endowment: Initial endowment of goods
        utility: Utility function u(x₁, ..., xₙ)
        demand: Marshallian demand function d(p, w) -> x
    """
    id: int
    endowment: List[float]
    utility: Callable[[List[float]], float]
    demand: Optional[Callable[[List[float], float], List[float]]] = None


@dataclass
class ExchangeEconomy:
    """
    Pure exchange economy.

    An economy with:
    - L commodities
    - I consumers with endowments and preferences

    Attributes:
        consumers: List of consumers
        n_goods: Number of goods
    """
    consumers: List[Consumer]
    n_goods: int

    def aggregate_endowment(self) -> List[float]:
        """Total endowment in economy."""
        total = [0.0] * self.n_goods
        for c in self.consumers:
            for l in range(self.n_goods):
                total[l] += c.endowment[l]
        return total

    def excess_demand(self, prices: List[float]) -> List[float]:
        """
        Compute aggregate excess demand.

        z(p) = Σᵢ xᵢ(p, p·eᵢ) - Σᵢ eᵢ
        """
        if len(prices) != self.n_goods:
            raise ValueError(f"Expected {self.n_goods} prices")

        aggregate_demand = [0.0] * self.n_goods
        aggregate_supply = [0.0] * self.n_goods

        for consumer in self.consumers:
            # Wealth = p · e
            wealth = sum(p * e for p, e in zip(prices, consumer.endowment))

            # Demand
            if consumer.demand is not None:
                x = consumer.demand(prices, wealth)
            else:
                # Use Cobb-Douglas if no demand specified
                x = self._cobb_douglas_demand(prices, wealth)

            for l in range(self.n_goods):
                aggregate_demand[l] += x[l]
                aggregate_supply[l] += consumer.endowment[l]

        return [aggregate_demand[l] - aggregate_supply[l] for l in range(self.n_goods)]

    def _cobb_douglas_demand(self, prices: List[float], wealth: float) -> List[float]:
        """Default Cobb-Douglas demand with equal shares."""
        # x_l = α_l * w / p_l with α_l = 1/L
        L = self.n_goods
        alpha = 1 / L
        return [alpha * wealth / max(p, 1e-10) for p in prices]

    def is_pareto_efficient(self, allocation: List[List[float]], tol: float = 1e-6) -> bool:
        """
        Check if allocation is Pareto efficient.

        Allocation is (weakly) Pareto efficient if no reallocation
        can make everyone strictly better off.
        """
        # Check feasibility first
        total_alloc = [0.0] * self.n_goods
        for x in allocation:
            for l in range(self.n_goods):
                total_alloc[l] += x[l]

        total_endow = self.aggregate_endowment()
        for l in range(self.n_goods):
            if total_alloc[l] > total_endow[l] + tol:
                return False
            # Leftover resources could be shared out to make everyone better off
            if total_alloc[l] < total_endow[l] - tol:
                return False

        if any(v <= 0 for x in allocation for v in x):
            raise ValueError(
                "Pareto check needs an interior allocation "
                "(every consumer holds a positive amount of every good)"
            )

        # For an interior allocation with differentiable, monotone (quasi-concave)
        # utilities, no Pareto improvement exists iff every consumer has the same
        # marginal rates of substitution (first-order conditions of the planner problem).
        mrs = []
        for consumer, x in zip(self.consumers, allocation):
            grad = []
            for l in range(self.n_goods):
                h = 1e-6 * max(1.0, abs(x[l]))
                up = list(x)
                down = list(x)
                up[l] += h
                down[l] -= h
                grad.append((consumer.utility(up) - consumer.utility(down)) / (2 * h))
            mrs.append([g / grad[0] for g in grad[1:]])

        return all(
            abs(a - b) <= 1e-4 * max(1.0, abs(b))
            for other in mrs[1:]
            for a, b in zip(other, mrs[0])
        )

    def walras_law_check(self, prices: List[float]) -> float:
        """
        Verify Walras' Law: p · z(p) = 0.

        Returns the value of p · z(p), should be ≈ 0.
        """
        z = self.excess_demand(prices)
        return sum(p * z_l for p, z_l in zip(prices, z))


def find_walrasian_equilibrium(
    economy: ExchangeEconomy,
    initial_prices: Optional[List[float]] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    learning_rate: float = 0.1
) -> Tuple[List[float], List[float]]:
    """
    Find Walrasian equilibrium prices using tatonnement.

    The tatonnement process:
    p_{t+1} = p_t + λ * z(p_t)

    Args:
        economy: Exchange economy
        initial_prices: Starting prices (default: uniform)
        max_iter: Maximum iterations
        tol: Convergence tolerance
        learning_rate: Step size λ

    Returns:
        (equilibrium_prices, excess_demand)
    """
    L = economy.n_goods

    if initial_prices is None:
        prices = [1.0] * L
    else:
        prices = list(initial_prices)

    # Normalize prices (numeraire: last good)
    def normalize(p: List[float]) -> List[float]:
        p_L = p[-1] if p[-1] > 1e-10 else 1.0
        return [pi / p_L for pi in p]

    prices = normalize(prices)

    for iteration in range(max_iter):
        z = economy.excess_demand(prices)

        # Check convergence
        max_excess = max(abs(z_l) for z_l in z)
        if max_excess < tol:
            return prices, z

        # Tatonnement update
        for l in range(L - 1):  # Don't update numeraire
            prices[l] = max(1e-10, prices[l] + learning_rate * z[l])

        prices = normalize(prices)

    return prices, economy.excess_demand(prices)


def contract_curve(
    utility_1: Callable[[float, float], float],
    utility_2: Callable[[float, float], float],
    total_x: float,
    total_y: float,
    n_points: int = 100
) -> List[Tuple[float, float]]:
    """
    Compute contract curve for 2-consumer, 2-good exchange economy.

    The contract curve is the set of Pareto efficient allocations.

    Args:
        utility_1: Consumer 1's utility u₁(x₁, y₁)
        utility_2: Consumer 2's utility u₂(x₂, y₂)
        total_x: Total amount of good x
        total_y: Total amount of good y
        n_points: Number of points on curve

    Returns:
        List of (x₁, y₁) allocations on contract curve
    """
    curve = []

    for i in range(n_points + 1):
        x1 = (i / n_points) * total_x
        x2 = total_x - x1

        # For each x1, find y1 that maximizes u2 subject to u1 constant
        # Simplified: use golden section search

        best_y1 = 0.0
        best_u2 = float('-inf')

        for j in range(n_points + 1):
            y1 = (j / n_points) * total_y
            y2 = total_y - y1

            if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
                u2 = utility_2(x2, y2)
                if u2 > best_u2:
                    best_u2 = u2
                    best_y1 = y1

        curve.append((x1, best_y1))

    return curve


@dataclass
class CobbDouglasConsumer:
    """
    Consumer with Cobb-Douglas preferences.

    u(x) = ∏ᵢ xᵢ^αᵢ

    Demand: xᵢ = αᵢ * w / pᵢ
    """
    alphas: List[float]
    endowment: List[float]

    def __post_init__(self) -> None:
        # Normalize alphas
        total = sum(self.alphas)
        self.alphas = [a / total for a in self.alphas]

    def utility(self, x: List[float]) -> float:
        result = 1.0
        for xi, ai in zip(x, self.alphas):
            if xi <= 0:
                return float('-inf')
            result *= xi ** ai
        return result

    def demand(self, prices: List[float], wealth: float) -> List[float]:
        """Marshallian demand."""
        return [a * wealth / max(p, 1e-10) for a, p in zip(self.alphas, prices)]

    def indirect_utility(self, prices: List[float], wealth: float) -> float:
        """Indirect utility function v(p, w)."""
        x = self.demand(prices, wealth)
        return self.utility(x)

    def expenditure(self, prices: List[float], utility: float) -> float:
        """Expenditure function e(p, u)."""
        # e(p,u) = u * ∏ᵢ (pᵢ/αᵢ)^αᵢ
        result = utility
        for p, a in zip(prices, self.alphas):
            result *= (p / a) ** a
        return result


@dataclass
class CESConsumer:
    """
    Consumer with CES (Constant Elasticity of Substitution) preferences.

    u(x) = (Σᵢ αᵢ xᵢ^ρ)^(1/ρ)

    Elasticity of substitution σ = 1/(1-ρ)
    """
    alphas: List[float]
    rho: float  # ρ < 1; ρ = 0 is the Cobb-Douglas limit
    endowment: List[float]

    def __post_init__(self) -> None:
        if self.rho >= 1:
            raise ValueError("rho must be < 1")
        # Normalize alphas
        total = sum(self.alphas)
        self.alphas = [a / total for a in self.alphas]

    def elasticity_of_substitution(self) -> float:
        """Elasticity of substitution σ = 1/(1-ρ)."""
        return 1 / (1 - self.rho)

    def utility(self, x: List[float]) -> float:
        if abs(self.rho) < 1e-10:
            # Cobb-Douglas limit
            result = 1.0
            for xi, ai in zip(x, self.alphas):
                if xi <= 0:
                    return float('-inf')
                result *= xi ** ai
            return result

        inner = sum(a * (max(xi, 1e-10) ** self.rho) for xi, a in zip(x, self.alphas))
        return inner ** (1 / self.rho)

    def demand(self, prices: List[float], wealth: float) -> List[float]:
        """Marshallian demand for CES.

        xᵢ = αᵢ^σ pᵢ^(-σ) w / Σⱼ αⱼ^σ pⱼ^(1-σ), which maximises u(x) on the budget line.
        At ρ = 0 (σ = 1) this is the Cobb-Douglas demand xᵢ = αᵢ w / pᵢ.
        """
        if abs(self.rho) < 1e-10:
            return [a * wealth / p for a, p in zip(self.alphas, prices)]

        sigma = self.elasticity_of_substitution()
        denominator = sum(a ** sigma * p ** (1 - sigma) for a, p in zip(self.alphas, prices))
        return [
            a ** sigma * p ** (-sigma) * wealth / denominator
            for a, p in zip(self.alphas, prices)
        ]
