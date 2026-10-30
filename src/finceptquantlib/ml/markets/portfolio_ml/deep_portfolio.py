"""
Deep Learning Portfolio Optimization

Neural network-based portfolio construction that learns optimal
asset allocation weights from historical returns and optional
conditioning features.

- **NeuralPortfolioOptimizer**: Multi-layer feedforward network
  with softmax output (long-only constraint) trained to maximize
  Sharpe ratio, minimize variance, or optimize mean-variance utility.

All computations are pure Python with no external numerical
dependencies. Backpropagation is implemented from scratch.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from finceptquantlib.ml._base import FitResult
from finceptquantlib.ml._matrix_utils import (
    column_mean,
    dot_product,
    mat_vec_mul,
    transpose,
)


# -------------------------------------------------------------------
# Result dataclass
# -------------------------------------------------------------------


@dataclass
class DeepPortfolioResult:
    """Result of neural portfolio optimization.

    Attributes
    ----------
    weights : List[float]
        Optimal portfolio weights (n_assets,). Sum to 1, all >= 0.
    expected_return : float
        Expected portfolio return (annualized or per-period).
    risk : float
        Portfolio risk (standard deviation of returns).
    sharpe_ratio : float
        Sharpe ratio (excess return / risk).
    training_loss_history : List[float]
        Loss value at each training epoch.
    """

    weights: List[float]
    expected_return: float
    risk: float
    sharpe_ratio: float
    training_loss_history: List[float]


# -------------------------------------------------------------------
# Activation functions and helpers
# -------------------------------------------------------------------


def _tanh(x: float) -> float:
    """Numerically stable hyperbolic tangent."""
    x = max(-20.0, min(20.0, x))
    return math.tanh(x)


def _tanh_deriv(tanh_x: float) -> float:
    """Derivative of tanh given tanh(x)."""
    return 1.0 - tanh_x * tanh_x


def _softmax(z: List[float]) -> List[float]:
    """Softmax function with numerical stability.

    w_i = exp(z_i - max(z)) / sum(exp(z_j - max(z)))

    Parameters
    ----------
    z : List[float]
        Input logits.

    Returns
    -------
    List[float]
        Probability vector summing to 1.
    """
    z_max = max(z)
    exps = [math.exp(zi - z_max) for zi in z]
    total = sum(exps)
    if total == 0:
        n = len(z)
        return [1.0 / n] * n
    return [e / total for e in exps]


def _softmax_jacobian(w: List[float]) -> List[List[float]]:
    """Jacobian of softmax output w.r.t. input logits.

    J[i][j] = w[i] * (delta_{ij} - w[j])

    Parameters
    ----------
    w : List[float]
        Softmax output.

    Returns
    -------
    List[List[float]]
        Jacobian matrix (n x n).
    """
    n = len(w)
    J = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                J[i][j] = w[i] * (1.0 - w[j])
            else:
                J[i][j] = -w[i] * w[j]
    return J


# -------------------------------------------------------------------
# Simple feedforward network
# -------------------------------------------------------------------


class _Layer:
    """Single fully connected layer with tanh activation.

    Attributes
    ----------
    weights : List[List[float]]
        Weight matrix (input_dim x output_dim).
    biases : List[float]
        Bias vector (output_dim,).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rng: random.Random,
    ) -> None:
        # Xavier initialization
        scale = math.sqrt(2.0 / (input_dim + output_dim))
        self.weights: List[List[float]] = [
            [rng.gauss(0, scale) for _ in range(output_dim)]
            for _ in range(input_dim)
        ]
        self.biases: List[float] = [0.0] * output_dim

        # Gradient accumulators
        self.grad_weights: List[List[float]] = [
            [0.0] * output_dim for _ in range(input_dim)
        ]
        self.grad_biases: List[float] = [0.0] * output_dim

        # Cached activations for backprop
        self._input: List[float] = []
        self._output: List[float] = []

    def forward(
        self, x: List[float], apply_activation: bool = True
    ) -> List[float]:
        """Forward pass through the layer.

        Parameters
        ----------
        x : List[float]
            Input vector.
        apply_activation : bool
            Whether to apply tanh activation.

        Returns
        -------
        List[float]
            Output vector.
        """
        self._input = x
        out_dim = len(self.biases)
        in_dim = len(x)
        z = [0.0] * out_dim
        for j in range(out_dim):
            s = self.biases[j]
            for i in range(in_dim):
                s += x[i] * self.weights[i][j]
            if apply_activation:
                z[j] = _tanh(s)
            else:
                z[j] = s
        self._output = z
        return z

    def zero_grad(self) -> None:
        """Reset gradient accumulators to zero."""
        in_dim = len(self.weights)
        out_dim = len(self.biases)
        self.grad_weights = [
            [0.0] * out_dim for _ in range(in_dim)
        ]
        self.grad_biases = [0.0] * out_dim


class _Network:
    """Feedforward neural network with softmax output.

    Architecture: input -> [hidden + tanh] x L -> output + softmax
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        output_dim: int,
        rng: random.Random,
    ) -> None:
        self.layers: List[_Layer] = []
        dims = [input_dim] + hidden_layers + [output_dim]
        for i in range(len(dims) - 1):
            self.layers.append(
                _Layer(dims[i], dims[i + 1], rng)
            )
        self.output_dim = output_dim

    def forward(self, x: List[float]) -> List[float]:
        """Forward pass returning softmax weights.

        Parameters
        ----------
        x : List[float]
            Input features.

        Returns
        -------
        List[float]
            Portfolio weights (sum to 1, all >= 0).
        """
        h = x
        for i, layer in enumerate(self.layers):
            is_last = i == len(self.layers) - 1
            h = layer.forward(h, apply_activation=not is_last)
        # Apply softmax to last layer output
        return _softmax(h)

    def zero_grad(self) -> None:
        """Reset all gradient accumulators."""
        for layer in self.layers:
            layer.zero_grad()

    def backward(
        self, dL_dw: List[float], weights: List[float]
    ) -> None:
        """Backpropagate gradient through the network.

        Parameters
        ----------
        dL_dw : List[float]
            Gradient of loss w.r.t. portfolio weights (n_assets,).
        weights : List[float]
            Current softmax output (portfolio weights).
        """
        n_layers = len(self.layers)

        # Gradient through softmax: dL/dz = J_softmax^T @ dL/dw
        J = _softmax_jacobian(weights)
        Jt = transpose(J)
        delta = mat_vec_mul(Jt, dL_dw)

        # Backprop through layers in reverse
        for layer_idx in range(n_layers - 1, -1, -1):
            layer = self.layers[layer_idx]
            is_last = layer_idx == n_layers - 1
            in_dim = len(layer.weights)
            out_dim = len(layer.biases)

            if not is_last:
                # Apply tanh derivative
                for j in range(out_dim):
                    delta[j] *= _tanh_deriv(layer._output[j])

            # Accumulate gradients
            for i in range(in_dim):
                for j in range(out_dim):
                    layer.grad_weights[i][j] += (
                        layer._input[i] * delta[j]
                    )
            for j in range(out_dim):
                layer.grad_biases[j] += delta[j]

            # Propagate to previous layer
            if layer_idx > 0:
                new_delta = [0.0] * in_dim
                for i in range(in_dim):
                    s = 0.0
                    for j in range(out_dim):
                        s += layer.weights[i][j] * delta[j]
                    new_delta[i] = s
                delta = new_delta

    def update(self, lr: float, batch_size: int) -> None:
        """Update weights with averaged gradients.

        Parameters
        ----------
        lr : float
            Learning rate.
        batch_size : int
            Number of samples in the batch (for averaging).
        """
        for layer in self.layers:
            in_dim = len(layer.weights)
            out_dim = len(layer.biases)
            for i in range(in_dim):
                for j in range(out_dim):
                    layer.weights[i][j] -= (
                        lr * layer.grad_weights[i][j] / batch_size
                    )
            for j in range(out_dim):
                layer.biases[j] -= (
                    lr * layer.grad_biases[j] / batch_size
                )


# -------------------------------------------------------------------
# NeuralPortfolioOptimizer
# -------------------------------------------------------------------


class NeuralPortfolioOptimizer:
    """Neural network-based portfolio optimizer.

    Learns optimal portfolio weights by training a feedforward
    neural network to optimize a financial objective (Sharpe ratio,
    minimum variance, etc.) over historical return data.

    Parameters
    ----------
    n_assets : int
        Number of assets in the portfolio.
    hidden_layers : List[int]
        Sizes of hidden layers. Default is [64, 32].
    learning_rate : float
        SGD learning rate. Default is 0.001.
    max_iter : int
        Maximum training epochs. Default is 500.
    objective : str
        Optimization objective. One of:
        - ``"sharpe"``: maximize Sharpe ratio
        - ``"min_variance"``: minimize portfolio variance
        - ``"max_return"``: maximize expected return
        - ``"mean_variance"``: mean-variance utility
    risk_free_rate : float
        Risk-free rate for Sharpe computation. Default is 0.0.
    random_state : int
        Random seed. Default is 42.
    risk_aversion : float
        Risk aversion parameter for mean_variance objective.
        Default is 1.0.

    Examples
    --------
    >>> from finceptquantlib.ml.markets.portfolio_ml.deep_portfolio import (
    ...     NeuralPortfolioOptimizer,
    ... )
    >>> returns = [
    ...     [0.01, 0.02, -0.01],
    ...     [0.03, -0.01, 0.02],
    ...     [-0.02, 0.01, 0.005],
    ... ]
    >>> opt = NeuralPortfolioOptimizer(
    ...     n_assets=3, hidden_layers=[16, 8], max_iter=100
    ... )
    >>> fit_result = opt.fit(returns)
    >>> result = opt.predict_weights()
    >>> abs(sum(result.weights) - 1.0) < 1e-6
    True
    """

    _VALID_OBJECTIVES = (
        "sharpe",
        "min_variance",
        "max_return",
        "mean_variance",
    )

    def __init__(
        self,
        n_assets: int,
        hidden_layers: Optional[List[int]] = None,
        learning_rate: float = 0.001,
        max_iter: int = 500,
        objective: str = "sharpe",
        risk_free_rate: float = 0.0,
        random_state: int = 42,
        risk_aversion: float = 1.0,
    ) -> None:
        if objective not in self._VALID_OBJECTIVES:
            raise ValueError(
                f"objective must be one of "
                f"{self._VALID_OBJECTIVES}, got '{objective}'"
            )
        self.n_assets = n_assets
        self.hidden_layers = (
            hidden_layers if hidden_layers is not None
            else [64, 32]
        )
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.objective = objective
        self.risk_free_rate = risk_free_rate
        self.random_state = random_state
        self.risk_aversion = risk_aversion

        self._network: Optional[_Network] = None
        self._returns: Optional[List[List[float]]] = None
        self._features: Optional[List[List[float]]] = None
        self._fitted: bool = False

    def _compute_portfolio_stats(
        self,
        returns: List[List[float]],
        weights: List[float],
    ) -> Tuple[float, float]:
        """Compute portfolio mean return and std.

        Parameters
        ----------
        returns : List[List[float]]
            Asset returns (T x N).
        weights : List[float]
            Portfolio weights (N,).

        Returns
        -------
        Tuple[float, float]
            (mean_return, std_return).
        """
        T = len(returns)
        port_returns = [
            dot_product(returns[t], weights) for t in range(T)
        ]
        mean_r = sum(port_returns) / T
        var_r = sum(
            (r - mean_r) ** 2 for r in port_returns
        ) / max(T - 1, 1)
        std_r = math.sqrt(max(var_r, 0.0))
        return mean_r, std_r

    def _compute_loss_and_gradient(
        self,
        returns: List[List[float]],
        weights: List[float],
    ) -> Tuple[float, List[float]]:
        """Compute loss and its gradient w.r.t. weights.

        Parameters
        ----------
        returns : List[List[float]]
            Asset returns (T x N).
        weights : List[float]
            Current portfolio weights (N,).

        Returns
        -------
        Tuple[float, List[float]]
            (loss, dL/dw) where dL/dw is (N,).
        """
        T = len(returns)
        N = len(weights)

        # Portfolio returns
        port_ret = [
            dot_product(returns[t], weights) for t in range(T)
        ]
        mean_r = sum(port_ret) / T
        excess_mean = mean_r - self.risk_free_rate
        var_r = sum(
            (r - mean_r) ** 2 for r in port_ret
        ) / max(T - 1, 1)
        std_r = math.sqrt(max(var_r, 1e-12))

        # d(mean_r)/dw_j = (1/T) sum_t r_{t,j}
        d_mean = [0.0] * N
        for j in range(N):
            s = 0.0
            for t in range(T):
                s += returns[t][j]
            d_mean[j] = s / T

        # d(var_r)/dw_j = (2/(T-1)) sum_t (port_ret_t - mean_r)
        #                 * (r_{t,j} - d_mean_j)
        # But d_mean_j is the mean of column j, and the chain rule
        # is more nuanced. Let's derive carefully.
        # var_r = (1/(T-1)) sum_t (p_t - mu_p)^2
        # d(var_r)/dw_j = (2/(T-1)) sum_t (p_t - mu_p)
        #                 * d(p_t - mu_p)/dw_j
        # d(p_t)/dw_j = r_{t,j}
        # d(mu_p)/dw_j = d_mean_j = mean of column j
        # So: d(var_r)/dw_j = (2/(T-1)) sum_t (p_t - mu_p)
        #                     * (r_{t,j} - d_mean_j)
        d_var = [0.0] * N
        denom = max(T - 1, 1)
        for j in range(N):
            s = 0.0
            for t in range(T):
                s += (port_ret[t] - mean_r) * (
                    returns[t][j] - d_mean[j]
                )
            d_var[j] = 2.0 * s / denom

        # d(std_r)/dw_j = d_var_j / (2 * std_r)
        d_std = [dv / (2.0 * std_r) for dv in d_var]

        if self.objective == "sharpe":
            # Loss = -excess_mean / std_r
            loss = -excess_mean / std_r if std_r > 1e-12 else 0.0
            # dL/dw_j = -(d_mean_j * std_r - excess_mean * d_std_j)
            #           / std_r^2
            grad = [0.0] * N
            for j in range(N):
                grad[j] = -(
                    d_mean[j] * std_r
                    - excess_mean * d_std[j]
                ) / (std_r * std_r)

        elif self.objective == "min_variance":
            loss = var_r
            grad = d_var

        elif self.objective == "max_return":
            loss = -mean_r
            grad = [-d for d in d_mean]

        else:  # mean_variance
            lam = self.risk_aversion
            loss = -mean_r + 0.5 * lam * var_r
            grad = [
                -d_mean[j] + 0.5 * lam * d_var[j]
                for j in range(N)
            ]

        return loss, grad

    def fit(
        self,
        returns: List[List[float]],
        features: Optional[List[List[float]]] = None,
    ) -> FitResult:
        """Train the neural portfolio optimizer.

        Parameters
        ----------
        returns : List[List[float]]
            Historical asset returns (T x N). T observations,
            N assets.
        features : List[List[float]] or None
            Optional conditioning features (T x K). If None,
            uses the mean of recent returns as static input.

        Returns
        -------
        FitResult
            Training result with convergence info and final loss.
        """
        T = len(returns)
        N = len(returns[0]) if returns else 0

        if N != self.n_assets:
            raise ValueError(
                f"Expected {self.n_assets} assets, got {N}"
            )

        self._returns = returns

        # Prepare features
        if features is not None:
            self._features = features
            input_dim = len(features[0])
            # Use mean feature vector as conditioning input
            feat_mean = [0.0] * input_dim
            for t in range(T):
                for k in range(input_dim):
                    feat_mean[k] += features[t][k]
            feat_mean = [v / T for v in feat_mean]
            input_vec = feat_mean
        else:
            # Use column means of returns as static input
            input_dim = N
            input_vec = column_mean(returns)
            self._features = None

        # Build network
        rng = random.Random(self.random_state)
        self._network = _Network(
            input_dim=input_dim,
            hidden_layers=self.hidden_layers,
            output_dim=N,
            rng=rng,
        )

        loss_history: List[float] = []
        best_loss = float("inf")
        patience_counter = 0
        patience = 50

        for epoch in range(self.max_iter):
            self._network.zero_grad()

            # Forward pass
            weights = self._network.forward(input_vec)

            # Compute loss and gradient w.r.t. weights
            loss, dL_dw = self._compute_loss_and_gradient(
                returns, weights
            )
            loss_history.append(loss)

            # Backpropagate
            self._network.backward(dL_dw, weights)

            # Update parameters
            self._network.update(self.learning_rate, 1)

            # Early stopping check
            if loss < best_loss - 1e-8:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        self._fitted = True
        self._input_vec = input_vec

        converged = patience_counter >= patience or (
            len(loss_history) > 1
            and abs(loss_history[-1] - loss_history[-2]) < 1e-8
        )

        return FitResult(
            converged=converged,
            iterations=len(loss_history),
            loss=loss_history[-1] if loss_history else 0.0,
            message=(
                "Converged" if converged
                else f"Stopped at max_iter={self.max_iter}"
            ),
        )

    def predict_weights(
        self, features: Optional[List[float]] = None
    ) -> DeepPortfolioResult:
        """Predict optimal portfolio weights.

        Parameters
        ----------
        features : List[float] or None
            Conditioning features for this period. If None,
            uses the training mean features.

        Returns
        -------
        DeepPortfolioResult
            Portfolio weights and performance statistics.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if not self._fitted or self._network is None:
            raise RuntimeError(
                "Model must be fitted before predict_weights"
            )

        if features is not None:
            input_vec = features
        else:
            input_vec = self._input_vec

        weights = self._network.forward(input_vec)

        # Compute stats from training data
        if self._returns is not None:
            mean_r, std_r = self._compute_portfolio_stats(
                self._returns, weights
            )
        else:
            mean_r, std_r = 0.0, 0.0

        sharpe = (
            (mean_r - self.risk_free_rate) / std_r
            if std_r > 1e-12
            else 0.0
        )

        return DeepPortfolioResult(
            weights=weights,
            expected_return=mean_r,
            risk=std_r,
            sharpe_ratio=sharpe,
            training_loss_history=[],
        )
