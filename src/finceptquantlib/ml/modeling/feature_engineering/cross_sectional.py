"""
Cross-Sectional Transformations

Transformations applied across assets at a single point in time.
Used for normalizing features across a cross-section of securities
to remove level effects and enable comparable scoring.

Functions
---------
- cross_sectional_rank: Rank normalized to [0, 1]
- cross_sectional_percentile: Percentile rank
- cross_sectional_zscore: Z-score across section
- cross_sectional_demean: Subtract cross-sectional mean
- winsorized_zscore: Clip outliers then zscore
"""

from __future__ import annotations

import math
from typing import List

NAN = float("nan")


def cross_sectional_rank(values: List[float]) -> List[float]:
    """Rank values normalized to [0, 1].

    Parameters
    ----------
    values : List[float]
        Cross-sectional values (one per asset). NaN values are
        preserved in the output.

    Returns
    -------
    List[float]
        Rank-normalized values in [0, 1]. Ties receive the average rank.
        NaN inputs produce NaN outputs.

    Examples
    --------
    >>> cross_sectional_rank([10.0, 30.0, 20.0])
    [0.0, 1.0, 0.5]
    """
    n = len(values)
    if n == 0:
        return []

    # Identify valid (non-NaN) entries
    indexed: List[tuple] = []
    for i, v in enumerate(values):
        if not math.isnan(v):
            indexed.append((v, i))

    num_valid = len(indexed)
    if num_valid == 0:
        return [NAN] * n

    result: List[float] = [NAN] * n

    if num_valid == 1:
        result[indexed[0][1]] = 0.5
        return result

    # Sort by value
    indexed.sort(key=lambda t: t[0])

    # Assign average ranks for ties
    i = 0
    while i < num_valid:
        j = i
        while j < num_valid and indexed[j][0] == indexed[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        # Normalize to [0, 1]
        normalized = avg_rank / (num_valid - 1)
        for k in range(i, j):
            result[indexed[k][1]] = normalized
        i = j

    return result


def cross_sectional_percentile(values: List[float]) -> List[float]:
    """Percentile rank across the cross-section.

    Parameters
    ----------
    values : List[float]
        Cross-sectional values. NaN values are preserved.

    Returns
    -------
    List[float]
        Percentile rank in [0, 1], where the value represents the
        fraction of values that are less than or equal to each element.

    Examples
    --------
    >>> cross_sectional_percentile([10.0, 30.0, 20.0])
    [0.3333333333333333, 1.0, 0.6666666666666666]
    """
    n = len(values)
    if n == 0:
        return []

    indexed: List[tuple] = []
    for i, v in enumerate(values):
        if not math.isnan(v):
            indexed.append((v, i))

    num_valid = len(indexed)
    if num_valid == 0:
        return [NAN] * n

    result: List[float] = [NAN] * n
    indexed.sort(key=lambda t: t[0])

    # For each value, percentile = count(values <= x) / n
    i = 0
    while i < num_valid:
        j = i
        while j < num_valid and indexed[j][0] == indexed[i][0]:
            j += 1
        # All values at indices i..j-1 are equal; j values are <= this
        pct = j / num_valid
        for k in range(i, j):
            result[indexed[k][1]] = pct
        i = j

    return result


def cross_sectional_zscore(values: List[float]) -> List[float]:
    """Z-score across the cross-section.

    Parameters
    ----------
    values : List[float]
        Cross-sectional values. NaN values are preserved.

    Returns
    -------
    List[float]
        Z-scores: (x - mean) / std. Returns 0.0 if std is zero.
    """
    n = len(values)
    if n == 0:
        return []

    valid = [v for v in values if not math.isnan(v)]
    num_valid = len(valid)
    if num_valid == 0:
        return [NAN] * n

    mean = sum(valid) / num_valid
    if num_valid < 2:
        return [0.0 if not math.isnan(v) else NAN for v in values]

    var = sum((v - mean) ** 2 for v in valid) / (num_valid - 1)
    std = math.sqrt(var)

    if std == 0.0:
        return [0.0 if not math.isnan(v) else NAN for v in values]

    result: List[float] = [NAN] * n
    for i, v in enumerate(values):
        if not math.isnan(v):
            result[i] = (v - mean) / std
    return result


def cross_sectional_demean(values: List[float]) -> List[float]:
    """Subtract cross-sectional mean from each value.

    Parameters
    ----------
    values : List[float]
        Cross-sectional values. NaN values are preserved.

    Returns
    -------
    List[float]
        Demeaned values: x - mean(x).
    """
    n = len(values)
    if n == 0:
        return []

    valid = [v for v in values if not math.isnan(v)]
    if len(valid) == 0:
        return [NAN] * n

    mean = sum(valid) / len(valid)
    result: List[float] = [NAN] * n
    for i, v in enumerate(values):
        if not math.isnan(v):
            result[i] = v - mean
    return result


def winsorized_zscore(
    values: List[float],
    lower: float = 0.01,
    upper: float = 0.99,
) -> List[float]:
    """Winsorized z-score: clip to percentile bounds then z-score.

    Parameters
    ----------
    values : List[float]
        Cross-sectional values. NaN values are preserved.
    lower : float, optional
        Lower percentile for clipping, by default 0.01.
    upper : float, optional
        Upper percentile for clipping, by default 0.99.

    Returns
    -------
    List[float]
        Z-scores of winsorized values.

    Notes
    -----
    First clips values to the [lower, upper] percentile range,
    then applies z-score normalization on the clipped values.
    """
    n = len(values)
    if n == 0:
        return []

    valid = sorted(v for v in values if not math.isnan(v))
    num_valid = len(valid)
    if num_valid == 0:
        return [NAN] * n

    # Compute percentile thresholds using linear interpolation
    def _percentile(sorted_vals: List[float], p: float) -> float:
        if len(sorted_vals) == 1:
            return sorted_vals[0]
        idx = p * (len(sorted_vals) - 1)
        lo = int(math.floor(idx))
        hi = min(lo + 1, len(sorted_vals) - 1)
        frac = idx - lo
        return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac

    lo_val = _percentile(valid, lower)
    hi_val = _percentile(valid, upper)

    # Clip values
    clipped: List[float] = [NAN] * n
    for i, v in enumerate(values):
        if math.isnan(v):
            continue
        clipped[i] = max(lo_val, min(hi_val, v))

    return cross_sectional_zscore(clipped)
