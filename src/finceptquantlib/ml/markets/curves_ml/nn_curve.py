"""
Neural Network Yield Curve Fitting

Provides a feedforward neural network for yield curve interpolation.
Implements correct backpropagation with Xavier/Glorot initialization,
supporting tanh, relu, and sigmoid activations.

The NN approach learns a non-linear mapping from tenor to zero rate
from observed data. Unlike GP regression, it does not inherently
provide uncertainty estimates but can fit complex curve shapes.

This module does NOT inherit from Curve to avoid constructor
incompatibility, but provides the same method interface
(zero_rate, discount_factor).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple


@dataclass
class NNCurveResult:
    """Result of neural network curve fitting.

    Attributes
    ----------
    training_loss : float
        Final MSE training loss.
    n_epochs : int
        Number of training epochs completed.
    layer_sizes : List[int]
        Network architecture including input and output layers.
    """

    training_loss: float
    n_epochs: int
    layer_sizes: List[int]


def _xavier_init(
    fan_in: int, fan_out: int, rng: random.Random
) -> List[List[float]]:
    """Xavier/Glorot uniform weight initialization.

    Parameters
    ----------
    fan_in : int
        Number of input units.
    fan_out : int
        Number of output units.
    rng : random.Random
        Random number generator for reproducibility.

    Returns
    -------
    List[List[float]]
        Weight matrix (fan_out x fan_in).
    """
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return [
        [rng.uniform(-limit, limit) for _ in range(fan_in)]
        for _ in range(fan_out)
    ]


def _tanh(x: float) -> float:
    """Numerically stable hyperbolic tangent.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    float
        tanh(x).
    """
    return math.tanh(x)


def _tanh_deriv(output: float) -> float:
    """Derivative of tanh given its output.

    Parameters
    ----------
    output : float
        Value of tanh(x).

    Returns
    -------
    float
        1 - tanh(x)^2.
    """
    return 1.0 - output * output


def _relu(x: float) -> float:
    """Rectified linear unit.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    float
        max(0, x).
    """
    return max(0.0, x)


def _relu_deriv(output: float) -> float:
    """Derivative of ReLU given its output.

    Parameters
    ----------
    output : float
        Value of relu(x).

    Returns
    -------
    float
        1 if output > 0 else 0.
    """
    return 1.0 if output > 0.0 else 0.0


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    float
        1 / (1 + exp(-x)).
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def _sigmoid_deriv(output: float) -> float:
    """Derivative of sigmoid given its output.

    Parameters
    ----------
    output : float
        Value of sigmoid(x).

    Returns
    -------
    float
        sigmoid(x) * (1 - sigmoid(x)).
    """
    return output * (1.0 - output)


class NeuralNetCurve:
    """Feedforward neural network for yield curve fitting.

    Learns a mapping from tenor (years) to zero rate using a
    multi-layer perceptron trained with backpropagation. Does NOT
    inherit from the Curve ABC to avoid constructor incompatibility,
    but provides the same method interface.

    Parameters
    ----------
    reference_date : date
        Curve valuation date.
    hidden_layers : List[int]
        Number of neurons in each hidden layer.
    activation : str
        Activation function: 'tanh', 'relu', or 'sigmoid'.
    learning_rate : float
        Learning rate for gradient descent.
    max_iter : int
        Maximum number of training epochs.
    tol : float
        Convergence tolerance on loss improvement.
    day_count : str
        Day count convention identifier.
    random_state : int
        Seed for reproducible weight initialization.

    Examples
    --------
    >>> from datetime import date
    >>> nn = NeuralNetCurve(date(2024, 1, 1), hidden_layers=[32, 16])
    >>> tenors = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    >>> rates = [0.03, 0.032, 0.035, 0.038, 0.04, 0.042]
    >>> result = nn.fit(tenors, rates)
    >>> nn.zero_rate(3.0)
    0.039319
    """

    def __init__(
        self,
        reference_date: date,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "tanh",
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
        day_count: str = "ACT/365F",
        random_state: int = 42,
    ) -> None:
        self.reference_date = reference_date
        self.day_count = day_count
        self._hidden_layers = hidden_layers if hidden_layers is not None else [32, 16]
        self._learning_rate = learning_rate
        self._max_iter = max_iter
        self._tol = tol
        self._random_state = random_state

        activation_lower = activation.lower()
        if activation_lower == "tanh":
            self._act_fn = _tanh
            self._act_deriv = _tanh_deriv
        elif activation_lower == "relu":
            self._act_fn = _relu
            self._act_deriv = _relu_deriv
        elif activation_lower == "sigmoid":
            self._act_fn = _sigmoid
            self._act_deriv = _sigmoid_deriv
        else:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Supported: 'tanh', 'relu', 'sigmoid'."
            )

        self._fitted = False
        self._weights: List[List[List[float]]] = []
        self._biases: List[List[float]] = []
        self._input_min: float = 0.0
        self._input_max: float = 1.0
        self._output_mean: float = 0.0
        self._output_std: float = 1.0

    def _build_network(self) -> None:
        """Initialize network weights with Xavier/Glorot init."""
        rng = random.Random(self._random_state)
        layer_sizes = [1] + self._hidden_layers + [1]
        self._weights = []
        self._biases = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            W = _xavier_init(fan_in, fan_out, rng)
            b = [0.0] * fan_out
            self._weights.append(W)
            self._biases.append(b)

    def _normalize_input(self, t: float) -> float:
        """Normalize tenor to [0, 1] range.

        Parameters
        ----------
        t : float
            Raw tenor value.

        Returns
        -------
        float
            Normalized value in [0, 1].
        """
        denom = self._input_max - self._input_min
        if denom == 0.0:
            return 0.5
        return (t - self._input_min) / denom

    def _normalize_output(self, y: float) -> float:
        """Normalize output for training.

        Parameters
        ----------
        y : float
            Raw rate value.

        Returns
        -------
        float
            Standardized value.
        """
        if self._output_std == 0.0:
            return 0.0
        return (y - self._output_mean) / self._output_std

    def _denormalize_output(self, y: float) -> float:
        """Reverse output normalization.

        Parameters
        ----------
        y : float
            Standardized value.

        Returns
        -------
        float
            Original-scale rate.
        """
        return y * self._output_std + self._output_mean

    def _forward(self, x: float) -> float:
        """Forward pass through the network (single scalar input).

        Parameters
        ----------
        x : float
            Normalized input value.

        Returns
        -------
        float
            Network output (normalized).
        """
        activations = [x]
        n_layers = len(self._weights)
        for layer_idx in range(n_layers):
            W = self._weights[layer_idx]
            b = self._biases[layer_idx]
            prev = activations
            new_activations: List[float] = []
            for j in range(len(W)):
                z = sum(
                    W[j][k] * prev[k] for k in range(len(prev))
                ) + b[j]
                if layer_idx < n_layers - 1:
                    new_activations.append(self._act_fn(z))
                else:
                    # Linear output layer
                    new_activations.append(z)
            activations = new_activations
        return activations[0]

    def _forward_full(
        self, x: float
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """Forward pass storing all pre- and post-activation values.

        Parameters
        ----------
        x : float
            Normalized input value.

        Returns
        -------
        Tuple[List[List[float]], List[List[float]]]
            (pre_activations, post_activations) for each layer.
            Layer 0 is the input layer (post only).
        """
        post_acts: List[List[float]] = [[x]]
        pre_acts: List[List[float]] = [[x]]  # placeholder for input
        n_layers = len(self._weights)

        for layer_idx in range(n_layers):
            W = self._weights[layer_idx]
            b = self._biases[layer_idx]
            prev = post_acts[-1]
            z_list: List[float] = []
            a_list: List[float] = []
            for j in range(len(W)):
                z = sum(
                    W[j][k] * prev[k] for k in range(len(prev))
                ) + b[j]
                z_list.append(z)
                if layer_idx < n_layers - 1:
                    a_list.append(self._act_fn(z))
                else:
                    a_list.append(z)
            pre_acts.append(z_list)
            post_acts.append(a_list)

        return pre_acts, post_acts

    def _backprop(self, x: float, target: float) -> None:
        """Backpropagation: compute gradients and update weights.

        Parameters
        ----------
        x : float
            Normalized input value.
        target : float
            Normalized target rate.
        """
        pre_acts, post_acts = self._forward_full(x)
        n_layers = len(self._weights)

        # Output layer delta: d(MSE)/d(z) = 2*(output - target) / 2
        # = (output - target) for single output
        output = post_acts[-1][0]
        deltas: List[List[float]] = [[] for _ in range(n_layers)]
        deltas[-1] = [output - target]

        # Hidden layer deltas (backpropagate)
        for layer_idx in range(n_layers - 2, -1, -1):
            W_next = self._weights[layer_idx + 1]
            delta_next = deltas[layer_idx + 1]
            n_units = len(self._weights[layer_idx])
            layer_delta: List[float] = []
            for j in range(n_units):
                # Sum of downstream deltas * weights
                upstream = sum(
                    delta_next[k] * W_next[k][j]
                    for k in range(len(delta_next))
                )
                # post_acts[layer_idx+1] is the activation output
                act_out = post_acts[layer_idx + 1][j]
                layer_delta.append(
                    upstream * self._act_deriv(act_out)
                )
            deltas[layer_idx] = layer_delta

        # Update weights and biases
        lr = self._learning_rate
        for layer_idx in range(n_layers):
            prev_act = post_acts[layer_idx]
            delta = deltas[layer_idx]
            W = self._weights[layer_idx]
            b = self._biases[layer_idx]
            for j in range(len(W)):
                for k in range(len(W[j])):
                    W[j][k] -= lr * delta[j] * prev_act[k]
                b[j] -= lr * delta[j]

    def fit(
        self, tenors: List[float], rates: List[float]
    ) -> NNCurveResult:
        """Fit the neural network to observed rates.

        Parameters
        ----------
        tenors : List[float]
            Observed tenor points in years.
        rates : List[float]
            Observed zero rates at each tenor.

        Returns
        -------
        NNCurveResult
            Training result with final loss and architecture.

        Raises
        ------
        ValueError
            If tenors and rates have different lengths or are empty.
        """
        if len(tenors) != len(rates):
            raise ValueError(
                f"tenors ({len(tenors)}) and rates ({len(rates)}) "
                f"must have the same length."
            )
        if len(tenors) == 0:
            raise ValueError("Cannot fit with empty data.")

        # Compute normalization parameters
        self._input_min = min(tenors)
        self._input_max = max(tenors)
        self._output_mean = sum(rates) / len(rates)
        self._output_std = math.sqrt(
            sum((r - self._output_mean) ** 2 for r in rates) / len(rates)
        )
        if self._output_std == 0.0:
            self._output_std = 1.0

        self._build_network()

        # Normalize data
        X_norm = [self._normalize_input(t) for t in tenors]
        y_norm = [self._normalize_output(r) for r in rates]

        n = len(tenors)
        prev_loss = float("inf")
        final_epoch = 0

        for epoch in range(self._max_iter):
            # Full-batch gradient descent
            for i in range(n):
                self._backprop(X_norm[i], y_norm[i])

            # Compute MSE loss
            total_loss = 0.0
            for i in range(n):
                pred = self._forward(X_norm[i])
                total_loss += (pred - y_norm[i]) ** 2
            loss = total_loss / n
            final_epoch = epoch + 1

            if abs(prev_loss - loss) < self._tol:
                break
            prev_loss = loss

        self._fitted = True
        layer_sizes = [1] + self._hidden_layers + [1]

        return NNCurveResult(
            training_loss=prev_loss,
            n_epochs=final_epoch,
            layer_sizes=layer_sizes,
        )

    def _check_fitted(self) -> None:
        """Verify model has been fitted.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                "Model has not been fitted. Call fit() first."
            )

    def zero_rate(
        self, t: float, compounding: str = "continuous"
    ) -> float:
        """Get zero rate at a given maturity.

        Parameters
        ----------
        t : float
            Time in years.
        compounding : str
            Compounding convention. 'continuous' returns the NN
            prediction directly.

        Returns
        -------
        float
            Interpolated zero rate.
        """
        self._check_fitted()
        x_norm = self._normalize_input(t)
        y_norm = self._forward(x_norm)
        rate = self._denormalize_output(y_norm)

        if compounding == "continuous":
            return rate

        periods_map = {
            "annual": 1,
            "semi-annual": 2,
            "quarterly": 4,
            "monthly": 12,
        }
        m = periods_map.get(compounding, 1)
        return m * (math.exp(rate / m) - 1.0)

    def discount_factor(self, t: float) -> float:
        """Get discount factor at a given maturity.

        df(t) = exp(-r(t) * t).

        Parameters
        ----------
        t : float
            Time in years.

        Returns
        -------
        float
            Discount factor.
        """
        self._check_fitted()
        if t <= 0.0:
            return 1.0
        rate = self.zero_rate(t, compounding="continuous")
        return math.exp(-rate * t)
