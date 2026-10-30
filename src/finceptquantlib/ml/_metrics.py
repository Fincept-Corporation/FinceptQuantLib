"""
Shared Evaluation Metrics

Common loss functions and evaluation metrics used across
all ML submodules. All operate on plain Python lists.
"""

from __future__ import annotations

import math
from typing import List


def mean_squared_error(
    y_true: List[float], y_pred: List[float]
) -> float:
    """Mean Squared Error.

    MSE = (1/n) * sum((y_true_i - y_pred_i)^2)

    Parameters
    ----------
    y_true : List[float]
        Actual values.
    y_pred : List[float]
        Predicted values.

    Returns
    -------
    float
        MSE value (non-negative).

    Raises
    ------
    ValueError
        If inputs have different lengths or are empty.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty")
    n = len(y_true)
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n


def root_mean_squared_error(
    y_true: List[float], y_pred: List[float]
) -> float:
    """Root Mean Squared Error.

    RMSE = sqrt(MSE)

    Parameters
    ----------
    y_true : List[float]
        Actual values.
    y_pred : List[float]
        Predicted values.

    Returns
    -------
    float
        RMSE value (non-negative).
    """
    return math.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(
    y_true: List[float], y_pred: List[float]
) -> float:
    """Mean Absolute Error.

    MAE = (1/n) * sum(|y_true_i - y_pred_i|)

    Parameters
    ----------
    y_true : List[float]
        Actual values.
    y_pred : List[float]
        Predicted values.

    Returns
    -------
    float
        MAE value (non-negative).
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty")
    n = len(y_true)
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n


def r_squared(
    y_true: List[float], y_pred: List[float]
) -> float:
    """Coefficient of Determination (R-squared).

    R^2 = 1 - SS_res / SS_tot

    where SS_res = sum((y_true - y_pred)^2)
          SS_tot = sum((y_true - mean(y_true))^2)

    Parameters
    ----------
    y_true : List[float]
        Actual values.
    y_pred : List[float]
        Predicted values.

    Returns
    -------
    float
        R^2 value. 1.0 for perfect fit, can be negative for
        predictions worse than the mean.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty")
    n = len(y_true)
    y_mean = sum(y_true) / n
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    ss_tot = sum((t - y_mean) ** 2 for t in y_true)
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def adjusted_r_squared(
    y_true: List[float],
    y_pred: List[float],
    n_features: int,
) -> float:
    """Adjusted R-squared.

    R^2_adj = 1 - (1 - R^2) * (n - 1) / (n - p - 1)

    Parameters
    ----------
    y_true : List[float]
        Actual values.
    y_pred : List[float]
        Predicted values.
    n_features : int
        Number of predictor variables.

    Returns
    -------
    float
        Adjusted R^2 value.
    """
    n = len(y_true)
    if n <= n_features + 1:
        return 0.0
    r2 = r_squared(y_true, y_pred)
    return 1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1)


def log_loss(
    y_true: List[float],
    y_pred: List[float],
    eps: float = 1e-15,
) -> float:
    """Binary Cross-Entropy (Log Loss).

    L = -(1/n) * sum(y*log(p) + (1-y)*log(1-p))

    Parameters
    ----------
    y_true : List[float]
        True binary labels (0 or 1).
    y_pred : List[float]
        Predicted probabilities in [0, 1].
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    float
        Log loss value (non-negative, lower is better).
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty")
    n = len(y_true)
    total = 0.0
    for t, p in zip(y_true, y_pred):
        p_clipped = max(eps, min(1.0 - eps, p))
        total += t * math.log(p_clipped) + (1.0 - t) * math.log(
            1.0 - p_clipped
        )
    return -total / n


def accuracy(
    y_true: List[float], y_pred: List[float]
) -> float:
    """Classification accuracy.

    Accuracy = (number correct) / n

    Parameters
    ----------
    y_true : List[float]
        True labels.
    y_pred : List[float]
        Predicted labels.

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty")
    n = len(y_true)
    correct = sum(
        1 for t, p in zip(y_true, y_pred) if abs(t - p) < 0.5
    )
    return correct / n


def confusion_matrix(
    y_true: List[float], y_pred: List[float]
) -> List[List[int]]:
    """Binary confusion matrix.

    Returns [[TN, FP], [FN, TP]].

    Parameters
    ----------
    y_true : List[float]
        True binary labels (0 or 1).
    y_pred : List[float]
        Predicted binary labels (0 or 1).

    Returns
    -------
    List[List[int]]
        2x2 confusion matrix [[TN, FP], [FN, TP]].
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    tp = fn = fp = tn = 0
    for t, p in zip(y_true, y_pred):
        t_bin = int(round(t))
        p_bin = int(round(p))
        if t_bin == 1 and p_bin == 1:
            tp += 1
        elif t_bin == 1 and p_bin == 0:
            fn += 1
        elif t_bin == 0 and p_bin == 1:
            fp += 1
        else:
            tn += 1
    return [[tn, fp], [fn, tp]]


def precision_score(
    y_true: List[float], y_pred: List[float]
) -> float:
    """Precision = TP / (TP + FP).

    Parameters
    ----------
    y_true : List[float]
        True binary labels.
    y_pred : List[float]
        Predicted binary labels.

    Returns
    -------
    float
        Precision in [0, 1].
    """
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1][1]
    fp = cm[0][1]
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall_score(
    y_true: List[float], y_pred: List[float]
) -> float:
    """Recall = TP / (TP + FN).

    Parameters
    ----------
    y_true : List[float]
        True binary labels.
    y_pred : List[float]
        Predicted binary labels.

    Returns
    -------
    float
        Recall in [0, 1].
    """
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1][1]
    fn = cm[1][0]
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def f1_score(
    y_true: List[float], y_pred: List[float]
) -> float:
    """F1 Score = 2 * precision * recall / (precision + recall).

    Parameters
    ----------
    y_true : List[float]
        True binary labels.
    y_pred : List[float]
        Predicted binary labels.

    Returns
    -------
    float
        F1 score in [0, 1].
    """
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    if p + r == 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)
