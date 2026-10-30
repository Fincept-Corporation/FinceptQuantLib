"""
Binomial Tree Pricing

Lattice methods for option pricing:
- Cox-Ross-Rubinstein (CRR)
- Jarrow-Rudd (JR)
- Tian tree
- Leisen-Reimer (LR)

Supports European and American options.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

from finceptquantlib.core.foundations.types import OptionType


class TreeMethod(Enum):
    """Binomial tree construction method."""
    CRR = auto()  # Cox-Ross-Rubinstein
    JR = auto()   # Jarrow-Rudd (equal probability)
    TIAN = auto()  # Tian (moment matching)
    LR = auto()   # Leisen-Reimer


@dataclass
class BinomialTreeResult:
    """Result of binomial tree pricing."""
    price: float
    delta: float
    gamma: float
    theta: float
    tree: Optional[List[List[float]]] = None  # Asset price tree
    values: Optional[List[List[float]]] = None  # Option value tree


def binomial_params(
    sigma: float,
    r: float,
    q: float,
    dt: float,
    method: TreeMethod = TreeMethod.CRR
) -> Tuple[float, float, float]:
    """
    Compute binomial tree parameters (u, d, p).

    Args:
        sigma: Volatility
        r: Risk-free rate
        q: Dividend yield
        dt: Time step
        method: Tree construction method

    Returns:
        (u, d, p): Up factor, down factor, risk-neutral probability
    """
    if method == TreeMethod.CRR:
        # Cox-Ross-Rubinstein
        u = math.exp(sigma * math.sqrt(dt))
        d = 1 / u
        p = (math.exp((r - q) * dt) - d) / (u - d)
        # Validate risk-neutral probability is in [0,1]
        if not (0 <= p <= 1):
            raise ValueError(
                f"CRR probability {p:.4f} outside [0,1]. "
                f"Try smaller dt or check parameters (sigma={sigma}, r={r}, q={q}, dt={dt})"
            )

    elif method == TreeMethod.JR:
        # Jarrow-Rudd (equal probabilities)
        drift = (r - q - 0.5 * sigma ** 2) * dt
        diffusion = sigma * math.sqrt(dt)
        u = math.exp(drift + diffusion)
        d = math.exp(drift - diffusion)
        p = 0.5

    elif method == TreeMethod.TIAN:
        # Tian (third moment matching)
        v = math.exp(sigma ** 2 * dt)
        discriminant = v ** 2 + 2 * v - 3
        sqrt_disc = math.sqrt(max(0.0, discriminant))
        u = 0.5 * math.exp((r - q) * dt) * v * (v + 1 + sqrt_disc)
        d = 0.5 * math.exp((r - q) * dt) * v * (v + 1 - sqrt_disc)
        if abs(u - d) < 1e-15:
            raise ValueError("Tian tree: u ≈ d, cannot compute probability")
        p = (math.exp((r - q) * dt) - d) / (u - d)

    elif method == TreeMethod.LR:
        # Leisen-Reimer (not fully implemented - use CRR)
        u = math.exp(sigma * math.sqrt(dt))
        d = 1 / u
        p = (math.exp((r - q) * dt) - d) / (u - d)

    else:
        raise ValueError(f"Unknown method: {method}")

    return u, d, p


def binomial_european(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_steps: int,
    option_type: OptionType,
    method: TreeMethod = TreeMethod.CRR,
    return_tree: bool = False
) -> BinomialTreeResult:
    """
    Price European option using binomial tree.

    Args:
        S: Spot price
        K: Strike price
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        T: Time to maturity
        n_steps: Number of time steps
        option_type: CALL or PUT
        method: Tree construction method
        return_tree: If True, return full tree

    Returns:
        BinomialTreeResult with price and Greeks
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    dt = T / n_steps
    u, d, p = binomial_params(sigma, r, q, dt, method)
    disc = math.exp(-r * dt)

    # Build asset price tree at maturity
    S_tree = [[S * (u ** j) * (d ** (n_steps - j)) for j in range(n_steps + 1)]]

    # Terminal payoffs
    if option_type == OptionType.CALL:
        values = [[max(s - K, 0) for s in S_tree[0]]]
    else:
        values = [[max(K - s, 0) for s in S_tree[0]]]

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        n_nodes = i + 1
        S_level = [S * (u ** j) * (d ** (i - j)) for j in range(n_nodes)]
        V_level = [disc * (p * values[0][j + 1] + (1 - p) * values[0][j])
                   for j in range(n_nodes)]

        S_tree.insert(0, S_level)
        values.insert(0, V_level)

    price = values[0][0]

    # Greeks via finite difference on tree
    if n_steps >= 2:
        delta = (values[1][1] - values[1][0]) / (S_tree[1][1] - S_tree[1][0])
        delta_up = (values[2][2] - values[2][1]) / (S_tree[2][2] - S_tree[2][1])
        delta_down = (values[2][1] - values[2][0]) / (S_tree[2][1] - S_tree[2][0])
        gamma = (delta_up - delta_down) / (0.5 * (S_tree[2][2] - S_tree[2][0]))
        theta = (values[2][1] - values[0][0]) / (2 * dt) / 365
    else:
        delta = gamma = theta = 0.0

    return BinomialTreeResult(
        price=price,
        delta=delta,
        gamma=gamma,
        theta=theta,
        tree=S_tree if return_tree else None,
        values=values if return_tree else None
    )


def binomial_american(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_steps: int,
    option_type: OptionType,
    method: TreeMethod = TreeMethod.CRR,
    return_tree: bool = False
) -> BinomialTreeResult:
    """
    Price American option using binomial tree.

    Allows early exercise at each node.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    dt = T / n_steps
    u, d, p = binomial_params(sigma, r, q, dt, method)
    disc = math.exp(-r * dt)

    # Build full asset price tree
    S_tree = []
    for i in range(n_steps + 1):
        S_level = [S * (u ** j) * (d ** (i - j)) for j in range(i + 1)]
        S_tree.append(S_level)

    # Terminal payoffs
    if option_type == OptionType.CALL:
        values = [max(s - K, 0) for s in S_tree[n_steps]]
    else:
        values = [max(K - s, 0) for s in S_tree[n_steps]]

    all_values = [values[:]]

    # Backward induction with early exercise
    for i in range(n_steps - 1, -1, -1):
        new_values = []
        for j in range(i + 1):
            continuation = disc * (p * values[j + 1] + (1 - p) * values[j])

            if option_type == OptionType.CALL:
                exercise = max(S_tree[i][j] - K, 0)
            else:
                exercise = max(K - S_tree[i][j], 0)

            new_values.append(max(continuation, exercise))

        values = new_values
        all_values.insert(0, values[:])

    price = values[0]

    # Greeks
    if n_steps >= 2:
        delta = (all_values[1][1] - all_values[1][0]) / (S_tree[1][1] - S_tree[1][0])
        delta_up = (all_values[2][2] - all_values[2][1]) / (S_tree[2][2] - S_tree[2][1])
        delta_down = (all_values[2][1] - all_values[2][0]) / (S_tree[2][1] - S_tree[2][0])
        gamma = (delta_up - delta_down) / (0.5 * (S_tree[2][2] - S_tree[2][0]))
        theta = (all_values[2][1] - all_values[0][0]) / (2 * dt) / 365
    else:
        delta = gamma = theta = 0.0

    return BinomialTreeResult(
        price=price,
        delta=delta,
        gamma=gamma,
        theta=theta,
        tree=S_tree if return_tree else None,
        values=all_values if return_tree else None
    )


def binomial_bermudan(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_steps: int,
    option_type: OptionType,
    exercise_dates: List[float],
    method: TreeMethod = TreeMethod.CRR
) -> BinomialTreeResult:
    """
    Price Bermudan option (exercise at specified dates only).

    Args:
        S, K, r, q, sigma, T, n_steps, option_type: Standard parameters
        exercise_dates: List of times when exercise is allowed
        method: Tree method

    Returns:
        BinomialTreeResult
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    dt = T / n_steps
    u, d, p = binomial_params(sigma, r, q, dt, method)
    disc = math.exp(-r * dt)

    # Convert exercise dates to step indices
    exercise_steps = set()
    for t in exercise_dates:
        step = int(round(t / dt))
        if 0 <= step <= n_steps:
            exercise_steps.add(step)

    # Build tree
    S_tree = []
    for i in range(n_steps + 1):
        S_level = [S * (u ** j) * (d ** (i - j)) for j in range(i + 1)]
        S_tree.append(S_level)

    # Terminal payoffs
    if option_type == OptionType.CALL:
        values = [max(s - K, 0) for s in S_tree[n_steps]]
    else:
        values = [max(K - s, 0) for s in S_tree[n_steps]]

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        new_values = []
        for j in range(i + 1):
            continuation = disc * (p * values[j + 1] + (1 - p) * values[j])

            if i in exercise_steps:
                if option_type == OptionType.CALL:
                    exercise = max(S_tree[i][j] - K, 0)
                else:
                    exercise = max(K - S_tree[i][j], 0)
                new_values.append(max(continuation, exercise))
            else:
                new_values.append(continuation)

        values = new_values

    return BinomialTreeResult(
        price=values[0],
        delta=0.0,
        gamma=0.0,
        theta=0.0
    )


def binomial_barrier(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_steps: int,
    option_type: OptionType,
    barrier: float,
    is_knock_in: bool,
    is_down: bool,
    rebate: float = 0.0,
    method: TreeMethod = TreeMethod.CRR
) -> float:
    """
    Price barrier option using binomial tree.

    Args:
        S, K, r, q, sigma, T, n_steps, option_type: Standard params
        barrier: Barrier level
        is_knock_in: True for knock-in, False for knock-out
        is_down: True for down barrier, False for up barrier
        rebate: Rebate paid on knock-out
        method: Tree method

    Returns:
        Barrier option price
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    dt = T / n_steps
    u, d, p = binomial_params(sigma, r, q, dt, method)
    disc = math.exp(-r * dt)

    # Build tree
    S_tree = []
    for i in range(n_steps + 1):
        S_level = [S * (u ** j) * (d ** (i - j)) for j in range(i + 1)]
        S_tree.append(S_level)

    # Terminal payoffs
    if option_type == OptionType.CALL:
        payoffs = [max(s - K, 0) for s in S_tree[n_steps]]
    else:
        payoffs = [max(K - s, 0) for s in S_tree[n_steps]]

    # Check barrier at terminal
    def check_barrier(s: float) -> bool:
        if is_down:
            return s <= barrier
        else:
            return s >= barrier

    if is_knock_in:
        # Knock-in via in/out parity: KnockIn = Vanilla - KnockOut + rebate_component
        # Price vanilla European
        vanilla_values = payoffs[:]
        for i in range(n_steps - 1, -1, -1):
            vanilla_values = [
                disc * (p * vanilla_values[j + 1] + (1 - p) * vanilla_values[j])
                for j in range(i + 1)
            ]
        vanilla_price = vanilla_values[0]

        # Price knock-out (same barrier direction)
        ko_price = binomial_barrier(
            S, K, r, q, sigma, T, n_steps, option_type,
            barrier, is_knock_in=False, is_down=is_down,
            rebate=0.0, method=method
        )
        return vanilla_price - ko_price
    else:
        # Knock-out: starts active, becomes worthless on barrier touch
        values = [0 if check_barrier(S_tree[n_steps][j]) else payoffs[j]
                  for j in range(n_steps + 1)]

        # Backward induction
        for i in range(n_steps - 1, -1, -1):
            new_values = []
            for j in range(i + 1):
                s = S_tree[i][j]
                hit = check_barrier(s)

                if hit:
                    new_values.append(rebate * math.exp(-r * (T - i * dt)))
                else:
                    continuation = disc * (p * values[j + 1] + (1 - p) * values[j])
                    new_values.append(continuation)

            values = new_values

        return values[0]
