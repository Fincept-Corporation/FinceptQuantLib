"""
Neural Network Volatility Surface

Provides a feedforward neural network for fitting volatility
surfaces over (strike, expiry) space. Inherits from
VolatilitySurface ABC to provide the standard interface.

The NN takes a 2D input (strike, expiry) and produces a scalar
volatility output. Trained with backpropagation to minimize MSE
on observed market volatilities.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from finceptquantlib.volatility.surface import VolatilitySurface


@dataclass
class NNVolSurfaceResult:
    """Result of NN volatility surface fitting.

    Attributes
    ----------
    training_loss : float
        Final MSE training loss.
    n_epochs : int
        Number of training epochs completed.
    """

    training_loss: float
    n_epochs: int


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
        Random number generator.

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
    """Numerically stable tanh.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    float
        tanh(x).
    """
    x = max(-20.0, min(20.0, x))
    return math.tanh(x)


def _tanh_deriv(output: float) -> float:
    """Derivative of tanh given output.

    Parameters
    ----------
    output : float
        tanh(x).

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
    """Derivative of ReLU given output.

    Parameters
    ----------
    output : float
        relu(x).

    Returns
    -------
    float
        1 if output > 0 else 0.
    """
    return 1.0 if output > 0.0 else 0.0


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    float
        sigmoid(x).
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def _sigmoid_deriv(output: float) -> float:
    """Derivative of sigmoid given output.

    Parameters
    ----------
    output : float
        sigmoid(x).

    Returns
    -------
    float
        sigmoid(x) * (1 - sigmoid(x)).
    """
    return output * (1.0 - output)


class NeuralNetVolSurface(VolatilitySurface):
    """Feedforward neural network volatility surface.

    Learns a mapping from (strike, expiry) to implied volatility
    using a multi-layer perceptron. Inherits from VolatilitySurface
    to provide the standard interface.

    Parameters
    ----------
    hidden_layers : List[int]
        Number of neurons per hidden layer.
    activation : str
        Activation function: 'tanh', 'relu', or 'sigmoid'.
    learning_rate : float
        Learning rate for gradient descent.
    max_iter : int
        Maximum number of training epochs.
    random_state : int
        Seed for reproducible initialization.

    Examples
    --------
    >>> nn = NeuralNetVolSurface(hidden_layers=[32, 16])
    >>> strikes = [0.8, 0.9, 1.0, 1.1, 1.2]
    >>> expiries = [0.25, 0.5, 1.0, 2.0, 5.0]
    >>> vols = [0.25, 0.22, 0.20, 0.22, 0.25]
    >>> result = nn.fit(strikes, expiries, vols)
    >>> nn.volatility(1.0, 0.5)
    0.187814
    """

    def __init__(
        self,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "tanh",
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self._hidden_layers = (
            hidden_layers if hidden_layers is not None else [32, 16]
        )
        self._learning_rate = learning_rate
        self._max_iter = max_iter
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

        # Input normalization parameters
        self._strike_min: float = 0.0
        self._strike_max: float = 1.0
        self._expiry_min: float = 0.0
        self._expiry_max: float = 1.0

        # Output normalization
        self._vol_mean: float = 0.0
        self._vol_std: float = 1.0

    def _build_network(self) -> None:
        """Initialize network weights."""
        rng = random.Random(self._random_state)
        layer_sizes = [2] + self._hidden_layers + [1]
        self._weights = []
        self._biases = []
        for i in range(len(layer_sizes) - 1):
            fin = layer_sizes[i]
            fout = layer_sizes[i + 1]
            self._weights.append(_xavier_init(fin, fout, rng))
            self._biases.append([0.0] * fout)

    def _normalize_strike(self, s: float) -> float:
        """Normalize strike to [0, 1].

        Parameters
        ----------
        s : float
            Raw strike.

        Returns
        -------
        float
            Normalized strike.
        """
        denom = self._strike_max - self._strike_min
        if denom == 0.0:
            return 0.5
        return (s - self._strike_min) / denom

    def _normalize_expiry(self, e: float) -> float:
        """Normalize expiry to [0, 1].

        Parameters
        ----------
        e : float
            Raw expiry.

        Returns
        -------
        float
            Normalized expiry.
        """
        denom = self._expiry_max - self._expiry_min
        if denom == 0.0:
            return 0.5
        return (e - self._expiry_min) / denom

    def _normalize_vol(self, v: float) -> float:
        """Normalize volatility.

        Parameters
        ----------
        v : float
            Raw volatility.

        Returns
        -------
        float
            Standardized volatility.
        """
        if self._vol_std == 0.0:
            return 0.0
        return (v - self._vol_mean) / self._vol_std

    def _denormalize_vol(self, v: float) -> float:
        """Reverse volatility normalization.

        Parameters
        ----------
        v : float
            Standardized volatility.

        Returns
        -------
        float
            Original-scale volatility.
        """
        return v * self._vol_std + self._vol_mean

    def _forward(self, inputs: List[float]) -> float:
        """Forward pass through the network.

        Parameters
        ----------
        inputs : List[float]
            Normalized input vector [strike, expiry].

        Returns
        -------
        float
            Network output (normalized volatility).
        """
        activations = list(inputs)
        n_layers = len(self._weights)
        for layer_idx in range(n_layers):
            W = self._weights[layer_idx]
            b = self._biases[layer_idx]
            new_act: List[float] = []
            for j in range(len(W)):
                z = sum(
                    W[j][k] * activations[k]
                    for k in range(len(activations))
                ) + b[j]
                if layer_idx < n_layers - 1:
                    new_act.append(self._act_fn(z))
                else:
                    new_act.append(z)
            activations = new_act
        return activations[0]

    def _forward_full(
        self, inputs: List[float]
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """Forward pass storing all activations.

        Parameters
        ----------
        inputs : List[float]
            Normalized input vector.

        Returns
        -------
        Tuple[List[List[float]], List[List[float]]]
            (pre_activations, post_activations) per layer.
        """
        post_acts: List[List[float]] = [list(inputs)]
        pre_acts: List[List[float]] = [list(inputs)]
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

    def _backprop(
        self, inputs: List[float], target: float
    ) -> None:
        """Backpropagation for a single 2D sample.

        Parameters
        ----------
        inputs : List[float]
            Normalized [strike, expiry].
        target : float
            Normalized target volatility.
        """
        pre_acts, post_acts = self._forward_full(inputs)
        n_layers = len(self._weights)
        lr = self._learning_rate

        # Output delta
        output = post_acts[-1][0]
        deltas: List[List[float]] = [[] for _ in range(n_layers)]
        deltas[-1] = [output - target]

        # Hidden deltas
        for layer_idx in range(n_layers - 2, -1, -1):
            W_next = self._weights[layer_idx + 1]
            delta_next = deltas[layer_idx + 1]
            n_units = len(self._weights[layer_idx])
            layer_delta: List[float] = []
            for j in range(n_units):
                upstream = sum(
                    delta_next[k] * W_next[k][j]
                    for k in range(len(delta_next))
                )
                act_out = post_acts[layer_idx + 1][j]
                layer_delta.append(
                    upstream * self._act_deriv(act_out)
                )
            deltas[layer_idx] = layer_delta

        # Weight updates
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
        self,
        strikes: List[float],
        expiries: List[float],
        vols: List[float],
    ) -> NNVolSurfaceResult:
        """Fit the NN to observed volatilities.

        Parameters
        ----------
        strikes : List[float]
            Strike values for each observation.
        expiries : List[float]
            Expiry values for each observation.
        vols : List[float]
            Observed volatilities.

        Returns
        -------
        NNVolSurfaceResult
            Training result with final loss.

        Raises
        ------
        ValueError
            If inputs have inconsistent lengths or are empty.
        """
        if not (len(strikes) == len(expiries) == len(vols)):
            raise ValueError(
                "strikes, expiries, and vols must have the same "
                "length."
            )
        if len(strikes) == 0:
            raise ValueError("Cannot fit with empty data.")

        # Compute normalization params
        self._strike_min = min(strikes)
        self._strike_max = max(strikes)
        self._expiry_min = min(expiries)
        self._expiry_max = max(expiries)
        self._vol_mean = sum(vols) / len(vols)
        self._vol_std = math.sqrt(
            sum((v - self._vol_mean) ** 2 for v in vols) / len(vols)
        )
        if self._vol_std == 0.0:
            self._vol_std = 1.0

        self._build_network()

        n = len(strikes)
        S_norm = [self._normalize_strike(s) for s in strikes]
        E_norm = [self._normalize_expiry(e) for e in expiries]
        V_norm = [self._normalize_vol(v) for v in vols]

        prev_loss = float("inf")
        final_epoch = 0

        for epoch in range(self._max_iter):
            for i in range(n):
                self._backprop(
                    [S_norm[i], E_norm[i]], V_norm[i]
                )

            total_loss = 0.0
            for i in range(n):
                pred = self._forward([S_norm[i], E_norm[i]])
                total_loss += (pred - V_norm[i]) ** 2
            loss = total_loss / n
            final_epoch = epoch + 1

            if abs(prev_loss - loss) < 1e-8:
                break
            prev_loss = loss

        self._fitted = True

        return NNVolSurfaceResult(
            training_loss=prev_loss,
            n_epochs=final_epoch,
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

    def volatility(
        self,
        strike: float,
        expiry: float,
        forward: Optional[float] = None,
    ) -> float:
        """Get volatility for given strike and expiry.

        Parameters
        ----------
        strike : float
            Option strike.
        expiry : float
            Option expiry in years.
        forward : float, optional
            Forward price (unused).

        Returns
        -------
        float
            Predicted volatility.
        """
        self._check_fitted()
        s_norm = self._normalize_strike(strike)
        e_norm = self._normalize_expiry(expiry)
        v_norm = self._forward([s_norm, e_norm])
        return self._denormalize_vol(v_norm)

    def total_variance(
        self,
        strike: float,
        expiry: float,
        forward: Optional[float] = None,
    ) -> float:
        """Get total variance (vol^2 * T).

        Parameters
        ----------
        strike : float
            Option strike.
        expiry : float
            Option expiry in years.
        forward : float, optional
            Forward price (unused).

        Returns
        -------
        float
            Total variance.
        """
        vol = self.volatility(strike, expiry, forward)
        return vol * vol * expiry
