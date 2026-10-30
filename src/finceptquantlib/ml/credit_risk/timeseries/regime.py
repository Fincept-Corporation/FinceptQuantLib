"""
Regime Detection via Hidden Markov Models

Implements Gaussian Hidden Markov Models for financial regime detection
using the Baum-Welch (EM) algorithm for parameter estimation and the
Viterbi algorithm for decoding the most likely state sequence.

All computations are pure Python with log-space arithmetic for
numerical stability in the forward-backward algorithm.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple

from finceptquantlib.ml._base import FitResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LOG_2PI: float = math.log(2.0 * math.pi)
_MIN_VAR: float = 1e-6
_LOG_ZERO: float = -1e300


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class HMMResult(FitResult):
    """Result of Gaussian HMM fitting.

    Attributes
    ----------
    n_states : int
        Number of hidden states.
    transition_matrix : List[List[float]]
        State transition probability matrix (n_states x n_states).
    means : List[float]
        Emission means for each state.
    variances : List[float]
        Emission variances for each state.
    stationary_distribution : List[float]
        Long-run stationary distribution of states.
    state_sequence : List[int]
        Most likely state sequence (Viterbi decoding).
    state_probabilities : List[List[float]]
        Posterior state probabilities (gamma) at each time step.
    log_likelihood : float
        Final log-likelihood of the model.
    """

    n_states: int = 0
    transition_matrix: List[List[float]] = field(default_factory=list)
    means: List[float] = field(default_factory=list)
    variances: List[float] = field(default_factory=list)
    stationary_distribution: List[float] = field(default_factory=list)
    state_sequence: List[int] = field(default_factory=list)
    state_probabilities: List[List[float]] = field(default_factory=list)
    log_likelihood: float = 0.0


# ---------------------------------------------------------------------------
# Helper functions (log-space arithmetic)
# ---------------------------------------------------------------------------

def _log_sum_exp(values: List[float]) -> float:
    """Numerically stable log-sum-exp.

    Parameters
    ----------
    values : List[float]
        Log-space values to sum.

    Returns
    -------
    float
        log(sum(exp(values))).
    """
    if not values:
        return _LOG_ZERO
    max_val = max(values)
    if max_val <= _LOG_ZERO:
        return _LOG_ZERO
    total = 0.0
    for v in values:
        if v > _LOG_ZERO:
            total += math.exp(v - max_val)
    return max_val + math.log(total) if total > 0.0 else _LOG_ZERO


def _log_gaussian_pdf(x: float, mu: float, var: float) -> float:
    """Log of Gaussian probability density.

    Parameters
    ----------
    x : float
        Observation value.
    mu : float
        Mean of the Gaussian.
    var : float
        Variance of the Gaussian (must be > 0).

    Returns
    -------
    float
        Log N(x; mu, var).
    """
    var = max(var, _MIN_VAR)
    return -0.5 * (_LOG_2PI + math.log(var) + (x - mu) ** 2 / var)


# ---------------------------------------------------------------------------
# GaussianHMM
# ---------------------------------------------------------------------------

class GaussianHMM:
    """Gaussian Hidden Markov Model for regime detection.

    Uses the Baum-Welch (EM) algorithm for parameter estimation
    and the Viterbi algorithm for state sequence decoding. All
    forward-backward computations use log-space for numerical
    stability.

    Parameters
    ----------
    n_states : int
        Number of hidden states (regimes).
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on log-likelihood change.
    random_state : int
        Random seed for reproducible initialisation.

    Examples
    --------
    >>> model = GaussianHMM(n_states=2)
    >>> result = model.fit([0.01, -0.02, 0.015, -0.03, 0.005])
    >>> result.converged
    True
    """

    def __init__(
        self,
        n_states: int = 2,
        max_iter: int = 100,
        tol: float = 1e-6,
        random_state: int = 42,
    ) -> None:
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # Model parameters (set after fit)
        self._means: List[float] = []
        self._variances: List[float] = []
        self._trans: List[List[float]] = []
        self._initial: List[float] = []
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_params(self, obs: List[float]) -> None:
        """Initialise parameters via k-means-like data splitting.

        Parameters
        ----------
        obs : List[float]
            Observation sequence.
        """
        n = len(obs)
        k = self.n_states

        # Sort observations and split into k equal segments
        sorted_obs = sorted(obs)
        seg_size = max(1, n // k)
        self._means = []
        self._variances = []
        overall_var = _variance(obs)

        for s in range(k):
            start = s * seg_size
            end = min(start + seg_size, n) if s < k - 1 else n
            segment = sorted_obs[start:end]
            self._means.append(sum(segment) / len(segment))
            self._variances.append(
                max(_variance(segment), overall_var * 0.1, _MIN_VAR)
            )

        # Equal transition probabilities
        self._trans = [
            [1.0 / k for _ in range(k)] for _ in range(k)
        ]
        # Equal initial probabilities
        self._initial = [1.0 / k for _ in range(k)]

    # ------------------------------------------------------------------
    # Forward-backward (log-space)
    # ------------------------------------------------------------------

    def _forward(
        self, obs: List[float]
    ) -> Tuple[List[List[float]], float]:
        """Forward algorithm in log-space.

        Parameters
        ----------
        obs : List[float]
            Observation sequence of length T.

        Returns
        -------
        Tuple[List[List[float]], float]
            (log_alpha, log_likelihood) where log_alpha is
            T x n_states.
        """
        T = len(obs)
        k = self.n_states
        log_alpha: List[List[float]] = [
            [0.0] * k for _ in range(T)
        ]

        # t = 0
        for j in range(k):
            log_alpha[0][j] = (
                math.log(max(self._initial[j], 1e-300))
                + _log_gaussian_pdf(
                    obs[0], self._means[j], self._variances[j]
                )
            )

        # Log transition matrix
        log_trans = [
            [
                math.log(max(self._trans[i][j], 1e-300))
                for j in range(k)
            ]
            for i in range(k)
        ]

        # t = 1 .. T-1
        for t in range(1, T):
            for j in range(k):
                terms = [
                    log_alpha[t - 1][i] + log_trans[i][j]
                    for i in range(k)
                ]
                log_alpha[t][j] = (
                    _log_sum_exp(terms)
                    + _log_gaussian_pdf(
                        obs[t], self._means[j], self._variances[j]
                    )
                )

        log_ll = _log_sum_exp(log_alpha[T - 1])
        return log_alpha, log_ll

    def _backward(self, obs: List[float]) -> List[List[float]]:
        """Backward algorithm in log-space.

        Parameters
        ----------
        obs : List[float]
            Observation sequence of length T.

        Returns
        -------
        List[List[float]]
            log_beta of shape T x n_states.
        """
        T = len(obs)
        k = self.n_states
        log_beta: List[List[float]] = [
            [0.0] * k for _ in range(T)
        ]
        # beta_{T-1}(i) = 1 => log = 0 (already set)

        log_trans = [
            [
                math.log(max(self._trans[i][j], 1e-300))
                for j in range(k)
            ]
            for i in range(k)
        ]

        for t in range(T - 2, -1, -1):
            for i in range(k):
                terms = [
                    log_trans[i][j]
                    + _log_gaussian_pdf(
                        obs[t + 1],
                        self._means[j],
                        self._variances[j],
                    )
                    + log_beta[t + 1][j]
                    for j in range(k)
                ]
                log_beta[t][i] = _log_sum_exp(terms)

        return log_beta

    def _compute_gamma_xi(
        self,
        obs: List[float],
        log_alpha: List[List[float]],
        log_beta: List[List[float]],
    ) -> Tuple[List[List[float]], List[List[List[float]]]]:
        """Compute gamma and xi from forward-backward results.

        Parameters
        ----------
        obs : List[float]
            Observations of length T.
        log_alpha : List[List[float]]
            Forward log probabilities (T x k).
        log_beta : List[List[float]]
            Backward log probabilities (T x k).

        Returns
        -------
        Tuple[List[List[float]], List[List[List[float]]]]
            (gamma, xi) where gamma is T x k and xi is
            (T-1) x k x k.
        """
        T = len(obs)
        k = self.n_states

        # Gamma: gamma_t(i) = alpha_t(i) * beta_t(i) / P(O)
        gamma: List[List[float]] = []
        for t in range(T):
            log_vals = [
                log_alpha[t][i] + log_beta[t][i]
                for i in range(k)
            ]
            log_norm = _log_sum_exp(log_vals)
            row = []
            for i in range(k):
                if log_norm > _LOG_ZERO:
                    row.append(math.exp(log_vals[i] - log_norm))
                else:
                    row.append(1.0 / k)
            gamma.append(row)

        # Xi: xi_t(i,j) = alpha_t(i)*a_{ij}*b_j(o_{t+1})*beta_{t+1}(j)/P(O)
        log_trans = [
            [
                math.log(max(self._trans[i][j], 1e-300))
                for j in range(k)
            ]
            for i in range(k)
        ]

        xi: List[List[List[float]]] = []
        for t in range(T - 1):
            xi_t: List[List[float]] = [
                [0.0] * k for _ in range(k)
            ]
            log_vals: List[float] = []
            log_mat: List[List[float]] = [
                [0.0] * k for _ in range(k)
            ]
            for i in range(k):
                for j in range(k):
                    val = (
                        log_alpha[t][i]
                        + log_trans[i][j]
                        + _log_gaussian_pdf(
                            obs[t + 1],
                            self._means[j],
                            self._variances[j],
                        )
                        + log_beta[t + 1][j]
                    )
                    log_mat[i][j] = val
                    log_vals.append(val)
            log_norm = _log_sum_exp(log_vals)
            for i in range(k):
                for j in range(k):
                    if log_norm > _LOG_ZERO:
                        xi_t[i][j] = math.exp(
                            log_mat[i][j] - log_norm
                        )
                    else:
                        xi_t[i][j] = 1.0 / (k * k)
            xi.append(xi_t)

        return gamma, xi

    # ------------------------------------------------------------------
    # Stationary distribution
    # ------------------------------------------------------------------

    def _stationary_distribution(self) -> List[float]:
        """Compute stationary distribution by power iteration.

        Solves pi * A = pi with sum(pi) = 1.

        Returns
        -------
        List[float]
            Stationary distribution.
        """
        k = self.n_states
        pi = [1.0 / k] * k
        for _ in range(1000):
            new_pi = [0.0] * k
            for j in range(k):
                for i in range(k):
                    new_pi[j] += pi[i] * self._trans[i][j]
            # Normalise
            s = sum(new_pi)
            if s > 0:
                new_pi = [p / s for p in new_pi]
            # Check convergence
            diff = sum(abs(new_pi[i] - pi[i]) for i in range(k))
            pi = new_pi
            if diff < 1e-12:
                break
        return pi

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, observations: List[float]) -> HMMResult:
        """Fit the HMM using the Baum-Welch (EM) algorithm.

        Parameters
        ----------
        observations : List[float]
            Univariate observation sequence of length T.

        Returns
        -------
        HMMResult
            Fitting result with estimated parameters and decoded
            states.

        Raises
        ------
        ValueError
            If observations has fewer elements than n_states.
        """
        T = len(observations)
        k = self.n_states
        if T < k:
            raise ValueError(
                f"Need at least {k} observations, got {T}"
            )

        self._init_params(observations)

        prev_ll = _LOG_ZERO
        converged = False
        iteration = 0

        for iteration in range(1, self.max_iter + 1):
            # E-step
            log_alpha, log_ll = self._forward(observations)
            log_beta = self._backward(observations)
            gamma, xi = self._compute_gamma_xi(
                observations, log_alpha, log_beta
            )

            # Check convergence
            if abs(log_ll - prev_ll) < self.tol and iteration > 1:
                converged = True
                break
            prev_ll = log_ll

            # M-step
            # Initial state probabilities
            self._initial = [gamma[0][i] for i in range(k)]

            # Transition matrix
            for i in range(k):
                denom = sum(gamma[t][i] for t in range(T - 1))
                for j in range(k):
                    numer = sum(
                        xi[t][i][j] for t in range(T - 1)
                    )
                    if denom > 1e-300:
                        self._trans[i][j] = max(numer / denom, 1e-300)
                    else:
                        self._trans[i][j] = 1.0 / k
                # Normalise row
                row_sum = sum(self._trans[i])
                if row_sum > 0:
                    self._trans[i] = [
                        v / row_sum for v in self._trans[i]
                    ]

            # Emission means and variances
            for j in range(k):
                denom = sum(gamma[t][j] for t in range(T))
                if denom > 1e-300:
                    self._means[j] = (
                        sum(
                            gamma[t][j] * observations[t]
                            for t in range(T)
                        )
                        / denom
                    )
                    self._variances[j] = max(
                        sum(
                            gamma[t][j]
                            * (observations[t] - self._means[j]) ** 2
                            for t in range(T)
                        )
                        / denom,
                        _MIN_VAR,
                    )

        # Final forward pass for log-likelihood
        log_alpha, log_ll = self._forward(observations)
        log_beta = self._backward(observations)
        gamma, _ = self._compute_gamma_xi(
            observations, log_alpha, log_beta
        )

        state_seq = self.decode(observations)
        stat_dist = self._stationary_distribution()
        self._fitted = True

        return HMMResult(
            converged=converged,
            iterations=iteration,
            loss=-log_ll,
            message=(
                "Converged" if converged
                else f"Max iterations ({self.max_iter}) reached"
            ),
            n_states=k,
            transition_matrix=[row[:] for row in self._trans],
            means=self._means[:],
            variances=self._variances[:],
            stationary_distribution=stat_dist,
            state_sequence=state_seq,
            state_probabilities=gamma,
            log_likelihood=log_ll,
        )

    def decode(self, observations: List[float]) -> List[int]:
        """Decode the most likely state sequence (Viterbi algorithm).

        Parameters
        ----------
        observations : List[float]
            Observation sequence of length T.

        Returns
        -------
        List[int]
            Most likely state at each time step.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted and not self._means:
            raise RuntimeError("Model must be fitted before decoding")

        T = len(observations)
        k = self.n_states

        # Log transition matrix
        log_trans = [
            [
                math.log(max(self._trans[i][j], 1e-300))
                for j in range(k)
            ]
            for i in range(k)
        ]

        # delta[t][j] = best log-prob ending in state j at time t
        delta: List[List[float]] = [
            [0.0] * k for _ in range(T)
        ]
        psi: List[List[int]] = [
            [0] * k for _ in range(T)
        ]

        # t = 0
        for j in range(k):
            delta[0][j] = (
                math.log(max(self._initial[j], 1e-300))
                + _log_gaussian_pdf(
                    observations[0],
                    self._means[j],
                    self._variances[j],
                )
            )

        # t = 1 .. T-1
        for t in range(1, T):
            for j in range(k):
                best_val = _LOG_ZERO
                best_i = 0
                for i in range(k):
                    val = delta[t - 1][i] + log_trans[i][j]
                    if val > best_val:
                        best_val = val
                        best_i = i
                delta[t][j] = (
                    best_val
                    + _log_gaussian_pdf(
                        observations[t],
                        self._means[j],
                        self._variances[j],
                    )
                )
                psi[t][j] = best_i

        # Backtrack
        states: List[int] = [0] * T
        best_val = _LOG_ZERO
        for j in range(k):
            if delta[T - 1][j] > best_val:
                best_val = delta[T - 1][j]
                states[T - 1] = j

        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1][states[t + 1]]

        return states

    def predict_regime(
        self, observations: List[float]
    ) -> List[List[float]]:
        """Return posterior state probabilities.

        Parameters
        ----------
        observations : List[float]
            Observation sequence of length T.

        Returns
        -------
        List[List[float]]
            Posterior probabilities (gamma), shape T x n_states.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                "Model must be fitted before prediction"
            )

        log_alpha, _ = self._forward(observations)
        log_beta = self._backward(observations)
        gamma, _ = self._compute_gamma_xi(
            observations, log_alpha, log_beta
        )
        return gamma

    def sample(
        self, n_steps: int
    ) -> Tuple[List[float], List[int]]:
        """Generate observations and states from the fitted model.

        Parameters
        ----------
        n_steps : int
            Number of time steps to generate.

        Returns
        -------
        Tuple[List[float], List[int]]
            (observations, states) sampled from the model.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                "Model must be fitted before sampling"
            )

        rng = random.Random(self.random_state)
        observations: List[float] = []
        states: List[int] = []

        # Sample initial state
        state = _sample_discrete(self._initial, rng)
        for _ in range(n_steps):
            states.append(state)
            # Sample observation from emission distribution
            mu = self._means[state]
            sigma = math.sqrt(max(self._variances[state], _MIN_VAR))
            obs = rng.gauss(mu, sigma)
            observations.append(obs)
            # Transition to next state
            state = _sample_discrete(self._trans[state], rng)

        return observations, states


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _variance(data: List[float]) -> float:
    """Compute population variance of a list.

    Parameters
    ----------
    data : List[float]
        Data values.

    Returns
    -------
    float
        Population variance.
    """
    n = len(data)
    if n < 2:
        return _MIN_VAR
    mu = sum(data) / n
    return sum((x - mu) ** 2 for x in data) / n


def _sample_discrete(
    probs: List[float], rng: random.Random
) -> int:
    """Sample from a discrete distribution.

    Parameters
    ----------
    probs : List[float]
        Probability vector (should sum to ~1).
    rng : random.Random
        Random number generator instance.

    Returns
    -------
    int
        Sampled index.
    """
    u = rng.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if u <= cumulative:
            return i
    return len(probs) - 1
