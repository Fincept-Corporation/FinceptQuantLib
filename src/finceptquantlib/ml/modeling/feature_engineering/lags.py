"""
Lag and Lead Feature Generators

Generate lagged, lead, and return features from time series data.
Useful for creating autoregressive features and multi-horizon
return targets for financial ML models.

All functions return List[List[float]] where each inner list
represents a column (one per lag/lead/period). NaN is used for
positions where data is unavailable.

Functions
---------
- add_lags: Backward-looking shifted series
- add_leads: Forward-looking shifted series
- add_returns: Simple returns at multiple horizons
- add_log_returns: Log returns at multiple horizons
"""

from __future__ import annotations

import math
from typing import List

NAN = float("nan")


def add_lags(
    data: List[float], lags: List[int]
) -> List[List[float]]:
    """Create lagged versions of a time series.

    For a lag of ``k``, the value at position ``i`` is ``data[i - k]``.
    Positions where ``i - k < 0`` are filled with NaN.

    Parameters
    ----------
    data : List[float]
        Input time series of length ``n``.
    lags : List[int]
        List of positive lag values. E.g., ``[1, 2, 5]`` creates
        1-step, 2-step, and 5-step lagged features.

    Returns
    -------
    List[List[float]]
        Matrix of shape ``(n, len(lags))``. Each inner list has
        length equal to ``len(lags)``, with one value per lag.

    Examples
    --------
    >>> add_lags([10.0, 20.0, 30.0], [1])
    [[nan], [10.0], [20.0]]
    """
    n = len(data)
    result: List[List[float]] = []
    for i in range(n):
        row: List[float] = []
        for lag in lags:
            idx = i - lag
            if idx < 0:
                row.append(NAN)
            else:
                row.append(data[idx])
        result.append(row)
    return result


def add_leads(
    data: List[float], leads: List[int]
) -> List[List[float]]:
    """Create forward-looking (lead) versions of a time series.

    For a lead of ``k``, the value at position ``i`` is ``data[i + k]``.
    Positions where ``i + k >= n`` are filled with NaN.

    Parameters
    ----------
    data : List[float]
        Input time series of length ``n``.
    leads : List[int]
        List of positive lead values. E.g., ``[1, 5]`` creates
        1-step and 5-step forward features.

    Returns
    -------
    List[List[float]]
        Matrix of shape ``(n, len(leads))``.

    Examples
    --------
    >>> add_leads([10.0, 20.0, 30.0], [1])
    [[20.0], [30.0], [nan]]
    """
    n = len(data)
    result: List[List[float]] = []
    for i in range(n):
        row: List[float] = []
        for lead in leads:
            idx = i + lead
            if idx >= n:
                row.append(NAN)
            else:
                row.append(data[idx])
        result.append(row)
    return result


def add_returns(
    prices: List[float], periods: List[int]
) -> List[List[float]]:
    """Compute simple returns at multiple horizons.

    Simple return for period ``k`` at index ``i``:
    ``(prices[i] - prices[i - k]) / prices[i - k]``

    Parameters
    ----------
    prices : List[float]
        Price series of length ``n``.
    periods : List[int]
        List of return horizons. E.g., ``[1, 5, 21]`` for daily,
        weekly, and monthly returns.

    Returns
    -------
    List[List[float]]
        Matrix of shape ``(n, len(periods))``. NaN where
        insufficient history or zero denominator.

    Examples
    --------
    >>> add_returns([100.0, 110.0, 121.0], [1])
    [[nan], [0.1], [0.1]]
    """
    n = len(prices)
    result: List[List[float]] = []
    for i in range(n):
        row: List[float] = []
        for period in periods:
            idx = i - period
            if idx < 0 or prices[idx] == 0.0:
                row.append(NAN)
            else:
                row.append(
                    (prices[i] - prices[idx]) / prices[idx]
                )
        result.append(row)
    return result


def add_log_returns(
    prices: List[float], periods: List[int]
) -> List[List[float]]:
    """Compute log returns at multiple horizons.

    Log return for period ``k`` at index ``i``:
    ``ln(prices[i] / prices[i - k])``

    Parameters
    ----------
    prices : List[float]
        Price series of length ``n``. All values must be positive
        for valid log return computation.
    periods : List[int]
        List of return horizons.

    Returns
    -------
    List[List[float]]
        Matrix of shape ``(n, len(periods))``. NaN where
        insufficient history or non-positive prices.

    Examples
    --------
    >>> result = add_log_returns([100.0, 110.0], [1])
    >>> round(result[1][0], 6)
    0.095310
    """
    n = len(prices)
    result: List[List[float]] = []
    for i in range(n):
        row: List[float] = []
        for period in periods:
            idx = i - period
            if idx < 0 or prices[idx] <= 0.0 or prices[i] <= 0.0:
                row.append(NAN)
            else:
                row.append(math.log(prices[i] / prices[idx]))
        result.append(row)
    return result
