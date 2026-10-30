"""
Curve Autoencoder for Dimensionality Reduction

Provides an autoencoder neural network for learning low-dimensional
latent representations of yield curves. Useful for:
- Yield curve scenario generation
- Anomaly detection in curve shapes
- Feature extraction for downstream models
- Understanding principal modes of curve variation

The autoencoder compresses high-dimensional curve vectors (one value
per tenor) into a small latent space, then reconstructs them. The
latent space captures the most important modes of curve variation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class AutoencoderResult:
    """Result of autoencoder training.

    Attributes
    ----------
    reconstruction_error : float
        Mean squared reconstruction error on training data.
    latent_dim : int
        Dimensionality of the latent space.
    n_epochs : int
        Number of training epochs completed.
    explained_variance : float
        Fraction of input variance explained by reconstruction,
        computed as 1 - MSE / Var(input).
    """

    reconstruction_error: float
    latent_dim: int
    n_epochs: int
    explained_variance: float


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
    """Numerically stable hyperbolic tangent.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    float
        tanh(x), clamped for numerical stability.
    """
    x = max(-20.0, min(20.0, x))
    return math.tanh(x)


def _tanh_deriv(output: float) -> float:
    """Derivative of tanh given output.

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


class CurveAutoencoder:
    """Autoencoder for yield curve dimensionality reduction.

    Learns a compressed latent representation of yield curves via
    an encoder-decoder architecture trained to minimize
    reconstruction error.

    Architecture:
        Encoder: input_dim -> hidden_dim -> latent_dim
        Decoder: latent_dim -> hidden_dim -> input_dim

    Parameters
    ----------
    input_dim : int
        Number of tenor points per curve.
    latent_dim : int
        Dimensionality of latent space.
    hidden_dim : int
        Number of neurons in the hidden layers.
    learning_rate : float
        Learning rate for gradient descent.
    max_iter : int
        Maximum number of training epochs.
    random_state : int
        Seed for reproducible initialization.

    Examples
    --------
    >>> ae = CurveAutoencoder(input_dim=6, latent_dim=2)
    >>> curves = [
    ...     [0.03, 0.032, 0.035, 0.038, 0.04, 0.042],
    ...     [0.025, 0.028, 0.031, 0.034, 0.037, 0.039],
    ... ]
    >>> result = ae.fit(curves)
    >>> latent = ae.encode(curves)
    >>> reconstructed = ae.reconstruct(curves)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        hidden_dim: int = 16,
        learning_rate: float = 0.01,
        max_iter: int = 500,
        random_state: int = 42,
    ) -> None:
        self._input_dim = input_dim
        self._latent_dim = latent_dim
        self._hidden_dim = hidden_dim
        self._learning_rate = learning_rate
        self._max_iter = max_iter
        self._random_state = random_state

        self._fitted = False

        # Encoder weights: input -> hidden -> latent
        self._enc_W1: List[List[float]] = []
        self._enc_b1: List[float] = []
        self._enc_W2: List[List[float]] = []
        self._enc_b2: List[float] = []

        # Decoder weights: latent -> hidden -> output
        self._dec_W1: List[List[float]] = []
        self._dec_b1: List[float] = []
        self._dec_W2: List[List[float]] = []
        self._dec_b2: List[float] = []

        # Input normalization
        self._input_mean: List[float] = []
        self._input_std: List[float] = []

    def _build_network(self) -> None:
        """Initialize encoder and decoder weights."""
        rng = random.Random(self._random_state)

        # Encoder
        self._enc_W1 = _xavier_init(
            self._input_dim, self._hidden_dim, rng
        )
        self._enc_b1 = [0.0] * self._hidden_dim
        self._enc_W2 = _xavier_init(
            self._hidden_dim, self._latent_dim, rng
        )
        self._enc_b2 = [0.0] * self._latent_dim

        # Decoder
        self._dec_W1 = _xavier_init(
            self._latent_dim, self._hidden_dim, rng
        )
        self._dec_b1 = [0.0] * self._hidden_dim
        self._dec_W2 = _xavier_init(
            self._hidden_dim, self._input_dim, rng
        )
        self._dec_b2 = [0.0] * self._input_dim

    def _compute_normalization(
        self, curves: List[List[float]]
    ) -> None:
        """Compute per-feature mean and std for normalization.

        Parameters
        ----------
        curves : List[List[float]]
            Training curves (n_curves x input_dim).
        """
        n = len(curves)
        d = self._input_dim
        self._input_mean = [0.0] * d
        self._input_std = [1.0] * d

        for row in curves:
            for j in range(d):
                self._input_mean[j] += row[j]
        for j in range(d):
            self._input_mean[j] /= n

        for row in curves:
            for j in range(d):
                diff = row[j] - self._input_mean[j]
                self._input_std[j] += diff * diff
        for j in range(d):
            self._input_std[j] = math.sqrt(
                self._input_std[j] / n
            )
            if self._input_std[j] == 0.0:
                self._input_std[j] = 1.0

    def _normalize(self, x: List[float]) -> List[float]:
        """Normalize a single curve vector.

        Parameters
        ----------
        x : List[float]
            Raw curve values.

        Returns
        -------
        List[float]
            Standardized values.
        """
        return [
            (x[j] - self._input_mean[j]) / self._input_std[j]
            for j in range(self._input_dim)
        ]

    def _denormalize(self, x: List[float]) -> List[float]:
        """Reverse normalization.

        Parameters
        ----------
        x : List[float]
            Standardized values.

        Returns
        -------
        List[float]
            Original-scale values.
        """
        return [
            x[j] * self._input_std[j] + self._input_mean[j]
            for j in range(self._input_dim)
        ]

    def _layer_forward(
        self,
        inp: List[float],
        W: List[List[float]],
        b: List[float],
        apply_activation: bool,
    ) -> List[float]:
        """Forward pass through a single layer.

        Parameters
        ----------
        inp : List[float]
            Input activations.
        W : List[List[float]]
            Weight matrix (out_dim x in_dim).
        b : List[float]
            Bias vector (out_dim,).
        apply_activation : bool
            If True, apply tanh activation.

        Returns
        -------
        List[float]
            Layer output.
        """
        out_dim = len(W)
        in_dim = len(inp)
        result: List[float] = []
        for j in range(out_dim):
            z = sum(W[j][k] * inp[k] for k in range(in_dim)) + b[j]
            if apply_activation:
                result.append(_tanh(z))
            else:
                result.append(z)
        return result

    def _encode_single(
        self, x: List[float]
    ) -> Tuple[List[float], List[float], List[float]]:
        """Encode a single normalized curve, returning intermediates.

        Parameters
        ----------
        x : List[float]
            Normalized input.

        Returns
        -------
        Tuple[List[float], List[float], List[float]]
            (hidden_activation, latent_output, input_copy).
        """
        h = self._layer_forward(x, self._enc_W1, self._enc_b1, True)
        z = self._layer_forward(h, self._enc_W2, self._enc_b2, False)
        return h, z, x

    def _decode_single(
        self, z: List[float]
    ) -> Tuple[List[float], List[float]]:
        """Decode a single latent vector, returning intermediates.

        Parameters
        ----------
        z : List[float]
            Latent vector.

        Returns
        -------
        Tuple[List[float], List[float]]
            (hidden_activation, reconstruction).
        """
        h = self._layer_forward(z, self._dec_W1, self._dec_b1, True)
        out = self._layer_forward(
            h, self._dec_W2, self._dec_b2, False
        )
        return h, out

    def _backprop_single(
        self, x: List[float], target: List[float]
    ) -> float:
        """Backpropagation for a single sample.

        Computes gradients through the full encoder-decoder
        and updates weights in place.

        Parameters
        ----------
        x : List[float]
            Normalized input curve.
        target : List[float]
            Target (same as input for autoencoder).

        Returns
        -------
        float
            Sample MSE loss.
        """
        lr = self._learning_rate
        d = self._input_dim

        # Forward pass through encoder
        enc_h, latent, _ = self._encode_single(x)
        # Forward pass through decoder
        dec_h, output = self._decode_single(latent)

        # MSE loss per sample
        loss = sum(
            (output[j] - target[j]) ** 2 for j in range(d)
        ) / d

        # --- Decoder backward pass ---
        # Output layer delta: d(MSE)/d(z_out) = 2/d * (output - target)
        # Simplified as (output - target) scaled
        d_out = [
            2.0 * (output[j] - target[j]) / d for j in range(d)
        ]

        # Gradients for dec_W2, dec_b2
        # dec_W2 shape: (input_dim x hidden_dim)
        for j in range(d):
            for k in range(self._hidden_dim):
                self._dec_W2[j][k] -= lr * d_out[j] * dec_h[k]
            self._dec_b2[j] -= lr * d_out[j]

        # Decoder hidden layer delta
        d_dec_h = [0.0] * self._hidden_dim
        for k in range(self._hidden_dim):
            s = sum(
                d_out[j] * self._dec_W2[j][k] for j in range(d)
            )
            d_dec_h[k] = s * _tanh_deriv(dec_h[k])

        # Gradients for dec_W1, dec_b1
        # dec_W1 shape: (hidden_dim x latent_dim)
        for j in range(self._hidden_dim):
            for k in range(self._latent_dim):
                self._dec_W1[j][k] -= lr * d_dec_h[j] * latent[k]
            self._dec_b1[j] -= lr * d_dec_h[j]

        # --- Propagate to encoder ---
        # Delta at latent output from decoder
        d_latent = [0.0] * self._latent_dim
        for k in range(self._latent_dim):
            d_latent[k] = sum(
                d_dec_h[j] * self._dec_W1[j][k]
                for j in range(self._hidden_dim)
            )
            # Latent layer is linear, so no activation derivative

        # Gradients for enc_W2, enc_b2
        # enc_W2 shape: (latent_dim x hidden_dim)
        for j in range(self._latent_dim):
            for k in range(self._hidden_dim):
                self._enc_W2[j][k] -= lr * d_latent[j] * enc_h[k]
            self._enc_b2[j] -= lr * d_latent[j]

        # Encoder hidden layer delta
        d_enc_h = [0.0] * self._hidden_dim
        for k in range(self._hidden_dim):
            s = sum(
                d_latent[j] * self._enc_W2[j][k]
                for j in range(self._latent_dim)
            )
            d_enc_h[k] = s * _tanh_deriv(enc_h[k])

        # Gradients for enc_W1, enc_b1
        # enc_W1 shape: (hidden_dim x input_dim)
        for j in range(self._hidden_dim):
            for k in range(self._input_dim):
                self._enc_W1[j][k] -= lr * d_enc_h[j] * x[k]
            self._enc_b1[j] -= lr * d_enc_h[j]

        return loss

    def fit(
        self, curves: List[List[float]]
    ) -> AutoencoderResult:
        """Fit the autoencoder to a collection of curves.

        Parameters
        ----------
        curves : List[List[float]]
            Training curves, shape (n_curves x n_tenors).
            Each row is one curve observation.

        Returns
        -------
        AutoencoderResult
            Training result with reconstruction error and
            explained variance.

        Raises
        ------
        ValueError
            If curves are empty or have inconsistent dimensions.
        """
        if len(curves) == 0:
            raise ValueError("Cannot fit with empty data.")
        if any(len(c) != self._input_dim for c in curves):
            raise ValueError(
                f"All curves must have length {self._input_dim}."
            )

        self._compute_normalization(curves)
        self._build_network()

        X_norm = [self._normalize(c) for c in curves]
        n = len(X_norm)

        final_loss = float("inf")
        final_epoch = 0

        for epoch in range(self._max_iter):
            epoch_loss = 0.0
            for i in range(n):
                sample_loss = self._backprop_single(
                    X_norm[i], X_norm[i]
                )
                epoch_loss += sample_loss
            avg_loss = epoch_loss / n
            final_loss = avg_loss
            final_epoch = epoch + 1

        # Compute explained variance
        total_var = 0.0
        recon_mse = 0.0
        d = self._input_dim
        for i in range(n):
            enc_h, latent, _ = self._encode_single(X_norm[i])
            _, recon = self._decode_single(latent)
            for j in range(d):
                recon_mse += (recon[j] - X_norm[i][j]) ** 2
                total_var += X_norm[i][j] ** 2
        recon_mse /= n * d
        total_var /= n * d

        explained = 1.0 - (recon_mse / total_var) if total_var > 0.0 else 0.0
        explained = max(0.0, min(1.0, explained))

        self._fitted = True

        return AutoencoderResult(
            reconstruction_error=final_loss,
            latent_dim=self._latent_dim,
            n_epochs=final_epoch,
            explained_variance=explained,
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

    def encode(
        self, curves: List[List[float]]
    ) -> List[List[float]]:
        """Map curves to latent space.

        Parameters
        ----------
        curves : List[List[float]]
            Curves to encode (n_curves x input_dim).

        Returns
        -------
        List[List[float]]
            Latent representations (n_curves x latent_dim).
        """
        self._check_fitted()
        result: List[List[float]] = []
        for c in curves:
            x_norm = self._normalize(c)
            _, latent, _ = self._encode_single(x_norm)
            result.append(latent)
        return result

    def decode(
        self, latent: List[List[float]]
    ) -> List[List[float]]:
        """Reconstruct curves from latent representations.

        Parameters
        ----------
        latent : List[List[float]]
            Latent vectors (n_curves x latent_dim).

        Returns
        -------
        List[List[float]]
            Reconstructed curves (n_curves x input_dim).
        """
        self._check_fitted()
        result: List[List[float]] = []
        for z in latent:
            _, recon_norm = self._decode_single(z)
            recon = self._denormalize(recon_norm)
            result.append(recon)
        return result

    def reconstruct(
        self, curves: List[List[float]]
    ) -> List[List[float]]:
        """Encode then decode curves (full reconstruction).

        Parameters
        ----------
        curves : List[List[float]]
            Input curves (n_curves x input_dim).

        Returns
        -------
        List[List[float]]
            Reconstructed curves (n_curves x input_dim).
        """
        latent = self.encode(curves)
        return self.decode(latent)
