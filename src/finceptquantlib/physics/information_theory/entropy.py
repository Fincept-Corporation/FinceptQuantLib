"""
Information Theory and Entropy

Entropy measures and information-theoretic quantities:
- Shannon entropy
- Relative entropy (KL divergence)
- Mutual information
- Cross entropy
- Rényi entropy
- Tsallis entropy
"""

from __future__ import annotations

import math
import random
from typing import Callable, List, Optional


def shannon_entropy(
    probabilities: List[float],
    base: float = math.e
) -> float:
    """
    Shannon entropy.

    H(X) = -Σ p(x) log p(x)

    Args:
        probabilities: Probability distribution
        base: Logarithm base (e for nats, 2 for bits)

    Returns:
        Entropy in specified units
    """
    log_base = math.log(base)
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log(p) / log_base
    return entropy


def joint_entropy(
    joint_prob: List[List[float]],
    base: float = math.e
) -> float:
    """
    Joint entropy.

    H(X, Y) = -Σᵢⱼ p(xᵢ, yⱼ) log p(xᵢ, yⱼ)
    """
    log_base = math.log(base)
    entropy = 0.0
    for row in joint_prob:
        for p in row:
            if p > 0:
                entropy -= p * math.log(p) / log_base
    return entropy


def conditional_entropy(
    joint_prob: List[List[float]],
    base: float = math.e
) -> float:
    """
    Conditional entropy H(Y|X).

    H(Y|X) = H(X, Y) - H(X)

    Args:
        joint_prob: Joint probability p(x, y)

    Returns:
        H(Y|X)
    """
    # Marginal p(x)
    marginal_x = [sum(row) for row in joint_prob]

    h_joint = joint_entropy(joint_prob, base)
    h_x = shannon_entropy(marginal_x, base)

    return h_joint - h_x


def mutual_information(
    joint_prob: List[List[float]],
    base: float = math.e
) -> float:
    """
    Mutual information.

    I(X; Y) = H(X) + H(Y) - H(X, Y)
            = Σᵢⱼ p(xᵢ, yⱼ) log(p(xᵢ, yⱼ) / (p(xᵢ)p(yⱼ)))

    Measures dependence between X and Y.
    """
    n_x = len(joint_prob)
    n_y = len(joint_prob[0]) if joint_prob else 0

    # Marginals
    marginal_x = [sum(joint_prob[i]) for i in range(n_x)]
    marginal_y = [sum(joint_prob[i][j] for i in range(n_x)) for j in range(n_y)]

    log_base = math.log(base)
    mi = 0.0
    for i in range(n_x):
        for j in range(n_y):
            pxy = joint_prob[i][j]
            px = marginal_x[i]
            py = marginal_y[j]

            if pxy > 0 and px > 0 and py > 0:
                mi += pxy * math.log(pxy / (px * py)) / log_base

    return mi


def kl_divergence(
    p: List[float],
    q: List[float],
    base: float = math.e
) -> float:
    """
    Kullback-Leibler divergence (relative entropy).

    D_KL(P || Q) = Σ p(x) log(p(x) / q(x))

    Measures how P differs from Q.
    Not symmetric: D_KL(P||Q) ≠ D_KL(Q||P)

    Args:
        p: True distribution
        q: Approximate distribution
    """
    if len(p) != len(q):
        raise ValueError("Distributions must have same length")

    log_base = math.log(base)
    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            kl += pi * math.log(pi / qi) / log_base

    return kl


def js_divergence(
    p: List[float],
    q: List[float],
    base: float = math.e
) -> float:
    """
    Jensen-Shannon divergence.

    JS(P || Q) = (D_KL(P || M) + D_KL(Q || M)) / 2

    where M = (P + Q) / 2.

    Symmetric and bounded: 0 ≤ JS ≤ log(2)
    """
    if len(p) != len(q):
        raise ValueError("Distributions must have same length")

    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]

    return (kl_divergence(p, m, base) + kl_divergence(q, m, base)) / 2


def cross_entropy(
    p: List[float],
    q: List[float],
    base: float = math.e
) -> float:
    """
    Cross entropy.

    H(P, Q) = -Σ p(x) log q(x) = H(P) + D_KL(P || Q)

    Common loss function in machine learning.
    """
    log_base = math.log(base)
    ce = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            ce -= pi * math.log(qi) / log_base
    return ce


def renyi_entropy(
    probabilities: List[float],
    alpha: float,
    base: float = math.e
) -> float:
    """
    Rényi entropy of order α.

    H_α(X) = (1/(1-α)) * log(Σ p(x)^α)

    Special cases:
    - α → 0: Hartley entropy (log of support size)
    - α → 1: Shannon entropy
    - α → ∞: Min-entropy (-log max p(x))

    Args:
        alpha: Order parameter (α ≥ 0, α ≠ 1)
    """
    if abs(alpha - 1) < 1e-10:
        return shannon_entropy(probabilities, base)

    log_base = math.log(base)

    if alpha == 0:
        # Hartley entropy
        support_size = sum(1 for p in probabilities if p > 0)
        return math.log(support_size) / log_base

    if alpha == float('inf'):
        # Min-entropy
        return -math.log(max(probabilities)) / log_base

    sum_p_alpha = sum(p ** alpha for p in probabilities if p > 0)

    return math.log(sum_p_alpha) / ((1 - alpha) * log_base)


def tsallis_entropy(
    probabilities: List[float],
    q: float
) -> float:
    """
    Tsallis entropy.

    S_q(X) = (1/(q-1)) * (1 - Σ p(x)^q)

    Non-extensive entropy used in statistical mechanics.

    Special case: q → 1 gives Shannon entropy.

    Args:
        q: Entropic index (q > 0)
    """
    if abs(q - 1) < 1e-10:
        # Shannon limit
        return shannon_entropy(probabilities)

    sum_p_q = sum(p ** q for p in probabilities if p > 0)

    return (1 - sum_p_q) / (q - 1)


def differential_entropy_gaussian(
    variance: float,
    base: float = math.e
) -> float:
    """
    Differential entropy of Gaussian.

    h(X) = (1/2) * log(2πeσ²)

    This is the continuous analog of Shannon entropy.
    """
    return 0.5 * math.log(2 * math.pi * math.e * variance) / math.log(base)


def differential_entropy_exponential(
    rate: float,
    base: float = math.e
) -> float:
    """
    Differential entropy of exponential distribution.

    h(X) = 1 - log(λ)

    Args:
        rate: Rate parameter λ
    """
    return 1 - math.log(rate) / math.log(base)


def entropy_rate_markov(
    transition_matrix: List[List[float]],
    stationary_dist: Optional[List[float]] = None,
    base: float = math.e
) -> float:
    """
    Entropy rate of stationary Markov chain.

    H(X) = -Σᵢ πᵢ Σⱼ Pᵢⱼ log Pᵢⱼ

    where π is stationary distribution and P is transition matrix.
    """
    n = len(transition_matrix)

    # Compute stationary distribution if not provided
    if stationary_dist is None:
        # Power iteration
        pi = [1 / n] * n
        for _ in range(1000):
            pi_new = [0.0] * n
            for j in range(n):
                for i in range(n):
                    pi_new[j] += pi[i] * transition_matrix[i][j]
            pi = pi_new
        stationary_dist = pi

    # Compute entropy rate
    log_base = math.log(base)
    h_rate = 0.0
    for i in range(n):
        for j in range(n):
            p_ij = transition_matrix[i][j]
            if p_ij > 0 and stationary_dist[i] > 0:
                h_rate -= stationary_dist[i] * p_ij * math.log(p_ij) / log_base

    return h_rate


def transfer_entropy(
    x_history: List[float],
    y_history: List[float],
    lag: int = 1,
    n_bins: int = 10
) -> float:
    """
    Transfer entropy from Y to X.

    TE_{Y→X} = I(X_t; Y_{t-lag} | X_{t-lag})

    Measures information transfer from Y to X.

    Args:
        x_history: Time series X
        y_history: Time series Y
        lag: Time lag
        n_bins: Discretization bins
    """
    n = len(x_history) - lag
    if n < 10:
        raise ValueError("Not enough data points")

    # Discretize
    def discretize(series: List[float], n_bins: int) -> List[int]:
        min_val = min(series)
        max_val = max(series)
        if max_val <= min_val:
            return [0] * len(series)
        rng_val = max_val - min_val
        return [min(int((x - min_val) / rng_val * n_bins), n_bins - 1) for x in series]

    x_disc = discretize(x_history, n_bins)
    y_disc = discretize(y_history, n_bins)

    # Count joint occurrences
    counts_xyz = {}  # (x_t, x_{t-lag}, y_{t-lag})
    counts_xz = {}   # (x_t, x_{t-lag})
    counts_yz = {}   # (x_{t-lag}, y_{t-lag})
    counts_z = {}    # x_{t-lag}

    for t in range(lag, len(x_disc)):
        x_t = x_disc[t]
        x_lag = x_disc[t - lag]
        y_lag = y_disc[t - lag]

        key_xyz = (x_t, x_lag, y_lag)
        key_xz = (x_t, x_lag)
        key_yz = (x_lag, y_lag)
        key_z = x_lag

        counts_xyz[key_xyz] = counts_xyz.get(key_xyz, 0) + 1
        counts_xz[key_xz] = counts_xz.get(key_xz, 0) + 1
        counts_yz[key_yz] = counts_yz.get(key_yz, 0) + 1
        counts_z[key_z] = counts_z.get(key_z, 0) + 1

    # Compute transfer entropy
    te = 0.0
    total = sum(counts_xyz.values())

    for key_xyz, count_xyz in counts_xyz.items():
        x_t, x_lag, y_lag = key_xyz

        p_xyz = count_xyz / total
        p_xz = counts_xz.get((x_t, x_lag), 0) / total
        p_yz = counts_yz.get((x_lag, y_lag), 0) / total
        p_z = counts_z.get(x_lag, 0) / total

        if p_xyz > 0 and p_xz > 0 and p_yz > 0 and p_z > 0:
            te += p_xyz * math.log(p_xyz * p_z / (p_xz * p_yz))

    return te


def fisher_information(
    pdf: Callable[[float, float], float],
    theta: float,
    h: float = 1e-5,
    n_samples: int = 10000,
    seed: Optional[int] = None
) -> float:
    """
    Fisher information (numerical approximation).

    I(θ) = E[(∂/∂θ log f(X;θ))²]
         = -E[∂²/∂θ² log f(X;θ)]

    Measures information about θ contained in X.

    Args:
        pdf: PDF as function of (x, theta)
        theta: Parameter value
        h: Finite difference step
        n_samples: Number of Monte Carlo samples
        seed: Random seed for reproducibility
    """
    rng = random.Random(seed) if seed is not None else random
    fisher = 0.0

    for _ in range(n_samples):
        x = rng.gauss(theta, 1)  # Sample from distribution

        f_theta = pdf(x, theta)
        f_plus = pdf(x, theta + h)
        f_minus = pdf(x, theta - h)

        if f_theta > 1e-15:
            score = (f_plus - f_minus) / (2 * h * f_theta)
            fisher += score ** 2

    return fisher / n_samples
