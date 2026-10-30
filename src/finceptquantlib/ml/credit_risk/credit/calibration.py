"""
Probability Calibration for Credit Scoring Models

Provides Platt Scaling and Isotonic Regression for calibrating
predicted probabilities from credit scoring models. Proper
calibration ensures that predicted probabilities match observed
default rates, which is essential for:

- Regulatory capital calculations (PD calibration)
- Expected loss estimation
- Stress testing

All computations are pure Python with no external dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CalibrationResult:
    """Result of probability calibration.

    Attributes
    ----------
    calibrated_probabilities : List[float]
        Calibrated probability estimates.
    method : str
        Calibration method used (``"platt"`` or ``"isotonic"``).
    parameters : Dict[str, float]
        Fitted calibration parameters.
    brier_score_before : float
        Brier score before calibration.
    brier_score_after : float
        Brier score after calibration.
    """

    calibrated_probabilities: List[float]
    method: str
    parameters: Dict[str, float]
    brier_score_before: float
    brier_score_after: float


def _brier_score(y: List[float], p: List[float]) -> float:
    """Compute Brier score.

    BS = (1/n) * sum((y_i - p_i)^2)

    Parameters
    ----------
    y : List[float]
        True binary labels.
    p : List[float]
        Predicted probabilities.

    Returns
    -------
    float
        Brier score (lower is better).
    """
    n = len(y)
    if n == 0:
        return 0.0
    return sum((yi - pi) ** 2 for yi, pi in zip(y, p)) / n


def _sigmoid(z: float) -> float:
    """Numerically stable sigmoid.

    Parameters
    ----------
    z : float
        Input value.

    Returns
    -------
    float
        Sigmoid output in (0, 1).
    """
    z = max(-500.0, min(500.0, z))
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


class PlattScaling:
    """Platt Scaling for probability calibration.

    Fits a sigmoid function P(y=1|s) = 1 / (1 + exp(A*s + B))
    to transform raw model scores into calibrated probabilities.
    Uses Platt's target encoding to avoid overfitting.

    Parameters
    ----------
    max_iter : int
        Maximum iterations for gradient descent. Default is 200.
    lr : float
        Learning rate. Default is 0.01.
    tol : float
        Convergence tolerance. Default is 1e-7.

    References
    ----------
    Platt, J. (1999). "Probabilistic Outputs for Support Vector
    Machines and Comparisons to Regularized Likelihood Methods."
    """

    def __init__(
        self,
        max_iter: int = 200,
        lr: float = 0.01,
        tol: float = 1e-7,
    ) -> None:
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self._A: float = 0.0
        self._B: float = 0.0
        self._fitted: bool = False

    def fit(
        self, scores: List[float], y: List[int]
    ) -> CalibrationResult:
        """Fit Platt scaling parameters.

        Optimizes A and B by minimizing the negative log-likelihood
        of the sigmoid model using gradient descent. Uses Platt's
        target encoding:
            t_i = (y_i * N_+ + 1) / (N_+ + 2)

        where N_+ is the number of positive samples.

        Parameters
        ----------
        scores : List[float]
            Raw model scores or probabilities.
        y : List[int]
            True binary labels (0 or 1).

        Returns
        -------
        CalibrationResult
            Calibration result with fitted parameters and
            Brier scores before and after calibration.

        Raises
        ------
        ValueError
            If inputs are invalid.
        """
        if len(scores) != len(y):
            raise ValueError(
                "scores and y must have same length"
            )
        if len(scores) == 0:
            raise ValueError("Inputs must not be empty")

        n = len(scores)
        n_pos = sum(y)
        n_neg = n - n_pos

        # Platt's target encoding to avoid overfitting
        target_pos = (n_pos + 1.0) / (n_pos + 2.0)
        target_neg = 1.0 / (n_neg + 2.0)
        targets = [
            target_pos if yi == 1 else target_neg for yi in y
        ]

        # Initialize A and B
        A = 0.0
        B = math.log((n_neg + 1.0) / (n_pos + 1.0))

        # Gradient descent to minimize NLL
        prev_nll = 1e30

        for iteration in range(self.max_iter):
            # Compute probabilities: p = 1 / (1 + exp(A*s + B))
            # NLL = -sum(t*log(p) + (1-t)*log(1-p))
            nll = 0.0
            grad_A = 0.0
            grad_B = 0.0
            eps = 1e-15

            for i in range(n):
                f = A * scores[i] + B
                p = _sigmoid(-f)
                p = max(eps, min(1.0 - eps, p))

                nll -= (
                    targets[i] * math.log(p)
                    + (1.0 - targets[i]) * math.log(1.0 - p)
                )

                # Gradient: d(NLL)/dA and d(NLL)/dB
                # p = sigmoid(-f), dp/dA = -p*(1-p)*s
                # d(NLL)/df = -(t*(1-p) - (1-t)*p) = p - t
                diff = p - targets[i]
                grad_A -= diff * scores[i]
                grad_B -= diff

            # Check convergence
            if abs(nll - prev_nll) < self.tol:
                break
            prev_nll = nll

            # Update (gradient ascent on likelihood = descent
            # on NLL, but our gradients are negated)
            A += self.lr * grad_A / n
            B += self.lr * grad_B / n

        self._A = A
        self._B = B
        self._fitted = True

        # Compute Brier scores
        y_float = [float(yi) for yi in y]
        brier_before = _brier_score(y_float, scores)
        calibrated = self.calibrate(scores)
        brier_after = _brier_score(y_float, calibrated)

        return CalibrationResult(
            calibrated_probabilities=calibrated,
            method="platt",
            parameters={"A": self._A, "B": self._B},
            brier_score_before=brier_before,
            brier_score_after=brier_after,
        )

    def calibrate(self, scores: List[float]) -> List[float]:
        """Apply Platt scaling to transform scores.

        P(y=1|s) = 1 / (1 + exp(A*s + B))

        Parameters
        ----------
        scores : List[float]
            Raw model scores or probabilities.

        Returns
        -------
        List[float]
            Calibrated probabilities.

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                "PlattScaling has not been fitted"
            )
        return [
            _sigmoid(-(self._A * s + self._B)) for s in scores
        ]


class IsotonicRegression:
    """Isotonic (monotonic) regression for probability calibration.

    Implements the Pool Adjacent Violators Algorithm (PAVA) to
    fit a non-decreasing (or non-increasing) step function to
    the data. Uses linear interpolation for out-of-sample
    calibration.

    Parameters
    ----------
    increasing : bool
        If True (default), fit a non-decreasing function.
        If False, fit a non-increasing function.

    References
    ----------
    Barlow, R. E., et al. (1972). "Statistical Inference Under
    Order Restrictions."
    """

    def __init__(self, increasing: bool = True) -> None:
        self.increasing = increasing
        self._x_vals: List[float] = []
        self._y_vals: List[float] = []
        self._fitted: bool = False

    def fit(
        self, scores: List[float], y: List[float]
    ) -> CalibrationResult:
        """Fit isotonic regression using PAVA.

        The Pool Adjacent Violators Algorithm merges adjacent
        blocks that violate the monotonicity constraint, replacing
        them with their weighted average.

        Parameters
        ----------
        scores : List[float]
            Raw model scores.
        y : List[float]
            Target values (typically binary 0/1).

        Returns
        -------
        CalibrationResult
            Calibration result with Brier scores.

        Raises
        ------
        ValueError
            If inputs are invalid.
        """
        if len(scores) != len(y):
            raise ValueError(
                "scores and y must have same length"
            )
        if len(scores) == 0:
            raise ValueError("Inputs must not be empty")

        # Sort by score
        paired = sorted(zip(scores, y), key=lambda x: x[0])
        sorted_x = [p[0] for p in paired]
        sorted_y = [p[1] for p in paired]

        # PAVA - Pool Adjacent Violators Algorithm
        n = len(sorted_y)

        # Each block: (sum_y, count, sum_y/count)
        blocks: List[List[float]] = [
            [sorted_y[i], 1.0] for i in range(n)
        ]

        if self.increasing:
            # Merge adjacent blocks where left > right
            i = 0
            while i < len(blocks) - 1:
                left_mean = blocks[i][0] / blocks[i][1]
                right_mean = blocks[i + 1][0] / blocks[i + 1][1]
                if left_mean > right_mean:
                    # Pool: merge blocks i and i+1
                    blocks[i][0] += blocks[i + 1][0]
                    blocks[i][1] += blocks[i + 1][1]
                    blocks.pop(i + 1)
                    # Check backwards
                    if i > 0:
                        i -= 1
                else:
                    i += 1
        else:
            # Merge adjacent blocks where left < right
            i = 0
            while i < len(blocks) - 1:
                left_mean = blocks[i][0] / blocks[i][1]
                right_mean = blocks[i + 1][0] / blocks[i + 1][1]
                if left_mean < right_mean:
                    blocks[i][0] += blocks[i + 1][0]
                    blocks[i][1] += blocks[i + 1][1]
                    blocks.pop(i + 1)
                    if i > 0:
                        i -= 1
                else:
                    i += 1

        # Expand blocks back to individual values
        fitted_y: List[float] = []
        for block in blocks:
            mean_val = block[0] / block[1]
            count = int(block[1])
            fitted_y.extend([mean_val] * count)

        # Store unique (x, y) pairs for interpolation
        # Average y values at duplicate x values
        x_vals: List[float] = []
        y_vals: List[float] = []

        i = 0
        while i < n:
            x_val = sorted_x[i]
            y_sum = fitted_y[i]
            count = 1
            while i + count < n and sorted_x[i + count] == x_val:
                y_sum += fitted_y[i + count]
                count += 1
            x_vals.append(x_val)
            y_vals.append(y_sum / count)
            i += count

        self._x_vals = x_vals
        self._y_vals = y_vals
        self._fitted = True

        # Compute Brier scores
        brier_before = _brier_score(y, scores)
        calibrated = self.calibrate(scores)
        brier_after = _brier_score(y, calibrated)

        return CalibrationResult(
            calibrated_probabilities=calibrated,
            method="isotonic",
            parameters={
                "n_blocks": float(len(blocks)),
                "n_unique_points": float(len(x_vals)),
            },
            brier_score_before=brier_before,
            brier_score_after=brier_after,
        )

    def calibrate(self, scores: List[float]) -> List[float]:
        """Apply isotonic calibration via linear interpolation.

        For scores outside the fitted range, uses the boundary
        values (clamped interpolation).

        Parameters
        ----------
        scores : List[float]
            Raw model scores.

        Returns
        -------
        List[float]
            Calibrated values.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                "IsotonicRegression has not been fitted"
            )

        result: List[float] = []
        x_vals = self._x_vals
        y_vals = self._y_vals
        n = len(x_vals)

        for s in scores:
            if n == 1:
                result.append(y_vals[0])
                continue

            if s <= x_vals[0]:
                result.append(y_vals[0])
            elif s >= x_vals[-1]:
                result.append(y_vals[-1])
            else:
                # Binary search for interpolation interval
                lo, hi = 0, n - 1
                while lo < hi - 1:
                    mid = (lo + hi) // 2
                    if x_vals[mid] <= s:
                        lo = mid
                    else:
                        hi = mid

                # Linear interpolation
                dx = x_vals[hi] - x_vals[lo]
                if dx == 0.0:
                    result.append(y_vals[lo])
                else:
                    t = (s - x_vals[lo]) / dx
                    val = y_vals[lo] + t * (
                        y_vals[hi] - y_vals[lo]
                    )
                    result.append(val)

        return result
