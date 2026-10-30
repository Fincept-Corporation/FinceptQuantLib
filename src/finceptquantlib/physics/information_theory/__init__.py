"""
Information Theory

Entropy and information measures:

Discrete Entropy:
- shannon_entropy: H(X) = -Σ p log p
- joint_entropy: H(X, Y)
- conditional_entropy: H(Y|X)
- mutual_information: I(X; Y)

Divergences:
- kl_divergence: Kullback-Leibler D_KL(P || Q)
- js_divergence: Jensen-Shannon (symmetric)
- cross_entropy: H(P, Q)

Generalized Entropies:
- renyi_entropy: Rényi entropy H_α
- tsallis_entropy: Non-extensive S_q

Continuous:
- differential_entropy_gaussian
- differential_entropy_exponential

Time Series:
- entropy_rate_markov: Entropy rate of Markov chain
- transfer_entropy: Information flow TE_{Y→X}

Example:
    >>> from finceptquantlib.physics.information_theory import (
    ...     shannon_entropy, kl_divergence, mutual_information
    ... )
    >>>
    >>> # Fair coin entropy
    >>> shannon_entropy([0.5, 0.5], base=2)  # 1.0 bit
    1.0
    >>>
    >>> # KL divergence between distributions
    >>> kl_divergence([0.4, 0.6], [0.5, 0.5])
    0.020136
"""

from finceptquantlib.physics.information_theory.entropy import (
    shannon_entropy,
    joint_entropy,
    conditional_entropy,
    mutual_information,
    kl_divergence,
    js_divergence,
    cross_entropy,
    renyi_entropy,
    tsallis_entropy,
    differential_entropy_gaussian,
    differential_entropy_exponential,
    entropy_rate_markov,
    transfer_entropy,
    fisher_information,
)

__all__ = [
    "shannon_entropy",
    "joint_entropy",
    "conditional_entropy",
    "mutual_information",
    "kl_divergence",
    "js_divergence",
    "cross_entropy",
    "renyi_entropy",
    "tsallis_entropy",
    "differential_entropy_gaussian",
    "differential_entropy_exponential",
    "entropy_rate_markov",
    "transfer_entropy",
    "fisher_information",
]
