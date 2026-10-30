"""
General-Purpose Discrimination Metrics

Re-exports credit discrimination metrics for general ML model
evaluation and adds additional discrimination tools (lift curve,
gains chart) commonly used in model validation.

All operations use pure Python lists without external dependencies.
"""

from __future__ import annotations

from typing import List, Tuple

from finceptquantlib.ml.credit_risk.credit.discrimination import (
    DiscriminationResult,
    accuracy_ratio,
    auc_roc,
    cap_curve,
    full_discrimination_report,
    gini_coefficient,
    ks_statistic,
    roc_curve,
)

__all__ = [
    "DiscriminationResult",
    "accuracy_ratio",
    "auc_roc",
    "cap_curve",
    "full_discrimination_report",
    "gini_coefficient",
    "ks_statistic",
    "roc_curve",
    "lift_curve",
    "gains_chart",
]


def lift_curve(
    y_true: List[int],
    y_score: List[float],
    n_bins: int = 10,
) -> Tuple[List[float], List[float]]:
    """Compute cumulative lift curve.

    Lift measures how much better the model is at identifying
    positive events compared to random selection. Cumulative
    lift at the top x% of the population is:

        lift(x) = (cumulative event rate in top x%) / (overall event rate)

    Parameters
    ----------
    y_true : List[int]
        True binary labels (0 or 1).
    y_score : List[float]
        Predicted scores (higher = more likely positive).
    n_bins : int, optional
        Number of bins (deciles). Default is 10.

    Returns
    -------
    Tuple[List[float], List[float]]
        (cumulative_population_pct, cumulative_lift) where:
        - cumulative_population_pct: fraction of population
          examined, from 1/n_bins to 1.0.
        - cumulative_lift: lift value at each cumulative fraction.

    Raises
    ------
    ValueError
        If inputs are invalid or contain no positive samples.

    Examples
    --------
    >>> y_true = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
    >>> y_score = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    >>> pct, lift = lift_curve(y_true, y_score, n_bins=5)
    """
    if len(y_true) != len(y_score):
        raise ValueError(
            "y_true and y_score must have the same length"
        )
    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty")

    n = len(y_true)
    n_pos = sum(y_true)
    if n_pos == 0:
        raise ValueError("y_true must contain at least one positive")

    overall_rate = n_pos / n

    # Sort by score descending
    paired = sorted(
        zip(y_score, y_true), key=lambda x: -x[0]
    )

    bin_size = n / n_bins
    cum_pop_pct: List[float] = []
    cum_lift: List[float] = []

    cum_events = 0
    for i in range(n):
        cum_events += paired[i][1]
        bin_idx = i + 1
        # Check if we've reached the end of a bin
        expected_end = round(bin_size * (len(cum_pop_pct) + 1))
        if bin_idx >= expected_end or i == n - 1:
            pct = bin_idx / n
            cum_rate = cum_events / bin_idx
            lift_val = cum_rate / overall_rate
            cum_pop_pct.append(pct)
            cum_lift.append(lift_val)

    return cum_pop_pct, cum_lift


def gains_chart(
    y_true: List[int],
    y_score: List[float],
    n_bins: int = 10,
) -> Tuple[List[float], List[float]]:
    """Compute cumulative gains chart.

    Gains measure the cumulative proportion of all positive events
    captured when examining the top x% of the population sorted
    by model score:

        gains(x) = (cumulative events in top x%) / (total events)

    A perfect model captures all events in the first n_pos/n of
    the population. A random model follows the diagonal.

    Parameters
    ----------
    y_true : List[int]
        True binary labels (0 or 1).
    y_score : List[float]
        Predicted scores (higher = more likely positive).
    n_bins : int, optional
        Number of bins (deciles). Default is 10.

    Returns
    -------
    Tuple[List[float], List[float]]
        (cumulative_population_pct, cumulative_gains) where:
        - cumulative_population_pct: fraction of population
          examined, starting at 0.0.
        - cumulative_gains: fraction of total events captured
          at each cumulative fraction, starting at 0.0.

    Raises
    ------
    ValueError
        If inputs are invalid or contain no positive samples.

    Examples
    --------
    >>> y_true = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
    >>> y_score = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    >>> pct, gains = gains_chart(y_true, y_score, n_bins=5)
    """
    if len(y_true) != len(y_score):
        raise ValueError(
            "y_true and y_score must have the same length"
        )
    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty")

    n = len(y_true)
    n_pos = sum(y_true)
    if n_pos == 0:
        raise ValueError("y_true must contain at least one positive")

    # Sort by score descending
    paired = sorted(
        zip(y_score, y_true), key=lambda x: -x[0]
    )

    bin_size = n / n_bins
    cum_pop_pct: List[float] = [0.0]
    cum_gains: List[float] = [0.0]

    cum_events = 0
    for i in range(n):
        cum_events += paired[i][1]
        bin_idx = i + 1
        expected_end = round(bin_size * len(cum_pop_pct))
        if bin_idx >= expected_end or i == n - 1:
            pct = bin_idx / n
            gain = cum_events / n_pos
            cum_pop_pct.append(pct)
            cum_gains.append(gain)

    return cum_pop_pct, cum_gains
