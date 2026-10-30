"""
Rolling Window Statistics

Pure Python rolling and expanding window statistics for time series
feature engineering. All functions return List[float] with NaN for
positions with insufficient data.

Functions
---------
- rolling_mean, rolling_std, rolling_skewness, rolling_kurtosis
- rolling_zscore, rolling_correlation, rolling_beta
- expanding_mean, expanding_std
"""

from __future__ import annotations

import math
from typing import List

NAN = float("nan")


def rolling_mean(data: List[float], window: int) -> List[float]:
    """Rolling window arithmetic mean.

    Parameters
    ----------
    data : List[float]
        Input time series.
    window : int
        Rolling window size.

    Returns
    -------
    List[float]
        Rolling mean values. First ``window - 1`` values are NaN.
    """
    n = len(data)
    if window <= 0:
        raise ValueError("window must be positive")
    result: List[float] = [NAN] * n
    if window > n:
        return result

    s = sum(data[:window])
    result[window - 1] = s / window
    for i in range(window, n):
        s += data[i] - data[i - window]
        result[i] = s / window
    return result


def rolling_std(
    data: List[float], window: int, ddof: int = 1
) -> List[float]:
    """Rolling window standard deviation.

    Parameters
    ----------
    data : List[float]
        Input time series.
    window : int
        Rolling window size.
    ddof : int, optional
        Delta degrees of freedom, by default 1 (sample std).

    Returns
    -------
    List[float]
        Rolling standard deviation. First ``window - 1`` values are NaN.
    """
    n = len(data)
    if window <= 0:
        raise ValueError("window must be positive")
    if window <= ddof:
        raise ValueError("window must be greater than ddof")
    result: List[float] = [NAN] * n
    if window > n:
        return result

    for i in range(window - 1, n):
        w = data[i - window + 1: i + 1]
        mean = sum(w) / window
        var = sum((x - mean) ** 2 for x in w) / (window - ddof)
        result[i] = math.sqrt(var)
    return result


def rolling_skewness(data: List[float], window: int) -> List[float]:
    """Rolling window Fisher's skewness.

    Parameters
    ----------
    data : List[float]
        Input time series.
    window : int
        Rolling window size. Must be >= 3.

    Returns
    -------
    List[float]
        Rolling skewness. First ``window - 1`` values are NaN.

    Notes
    -----
    Uses the adjusted Fisher-Pearson standardized moment coefficient:
    G1 = [n/((n-1)(n-2))] * sum[((x - mean)/std)^3]
    """
    n = len(data)
    if window < 3:
        raise ValueError("window must be >= 3 for skewness")
    result: List[float] = [NAN] * n
    if window > n:
        return result

    for i in range(window - 1, n):
        w = data[i - window + 1: i + 1]
        m = sum(w) / window
        m2 = sum((x - m) ** 2 for x in w) / window
        if m2 == 0.0:
            result[i] = 0.0
            continue
        std_pop = math.sqrt(m2)
        m3 = sum(((x - m) / std_pop) ** 3 for x in w) / window
        # Adjusted Fisher-Pearson
        adj = (window * (window - 1)) ** 0.5 / (window - 2)
        result[i] = adj * m3

    return result


def rolling_kurtosis(data: List[float], window: int) -> List[float]:
    """Rolling window excess kurtosis.

    Parameters
    ----------
    data : List[float]
        Input time series.
    window : int
        Rolling window size. Must be >= 4.

    Returns
    -------
    List[float]
        Rolling excess kurtosis. First ``window - 1`` values are NaN.

    Notes
    -----
    Uses the sample excess kurtosis formula with bias correction:
    G2 = [(n+1)*n / ((n-1)(n-2)(n-3))] * sum[((x-mean)/s)^4]
         - 3*(n-1)^2 / ((n-2)(n-3))
    """
    n = len(data)
    if window < 4:
        raise ValueError("window must be >= 4 for kurtosis")
    result: List[float] = [NAN] * n
    if window > n:
        return result

    for i in range(window - 1, n):
        w = data[i - window + 1: i + 1]
        nw = window
        m = sum(w) / nw
        m2 = sum((x - m) ** 2 for x in w) / nw
        if m2 == 0.0:
            result[i] = 0.0
            continue
        std_pop = math.sqrt(m2)
        m4 = sum(((x - m) / std_pop) ** 4 for x in w) / nw
        # Bias-corrected excess kurtosis
        num = nw * (nw + 1)
        d1 = (nw - 1) * (nw - 2) * (nw - 3)
        correction = 3.0 * (nw - 1) ** 2 / ((nw - 2) * (nw - 3))
        result[i] = (num / d1) * nw * m4 / 1.0 - correction
        # Simplified: apply standard formula
        # Recompute with sample std
        s2 = m2 * nw / (nw - 1)
        s = math.sqrt(s2)
        sum4 = sum(((x - m) / s) ** 4 for x in w)
        result[i] = (
            (nw * (nw + 1)) / ((nw - 1) * (nw - 2) * (nw - 3)) * sum4
            - 3.0 * (nw - 1) ** 2 / ((nw - 2) * (nw - 3))
        )

    return result


def rolling_zscore(data: List[float], window: int) -> List[float]:
    """Rolling window z-score.

    Parameters
    ----------
    data : List[float]
        Input time series.
    window : int
        Rolling window size.

    Returns
    -------
    List[float]
        Z-scores: (x - rolling_mean) / rolling_std.
        First ``window - 1`` values are NaN.
    """
    n = len(data)
    means = rolling_mean(data, window)
    stds = rolling_std(data, window, ddof=1)
    result: List[float] = [NAN] * n
    for i in range(n):
        if math.isnan(means[i]) or math.isnan(stds[i]):
            continue
        if stds[i] == 0.0:
            result[i] = 0.0
        else:
            result[i] = (data[i] - means[i]) / stds[i]
    return result


def rolling_correlation(
    x: List[float], y: List[float], window: int
) -> List[float]:
    """Rolling window Pearson correlation coefficient.

    Parameters
    ----------
    x : List[float]
        First time series.
    y : List[float]
        Second time series.
    window : int
        Rolling window size.

    Returns
    -------
    List[float]
        Rolling Pearson correlation. First ``window - 1`` values are NaN.
    """
    n = len(x)
    if n != len(y):
        raise ValueError("x and y must have the same length")
    if window <= 1:
        raise ValueError("window must be > 1")

    result: List[float] = [NAN] * n
    if window > n:
        return result

    for i in range(window - 1, n):
        wx = x[i - window + 1: i + 1]
        wy = y[i - window + 1: i + 1]
        mx = sum(wx) / window
        my = sum(wy) / window

        cov = sum((wx[j] - mx) * (wy[j] - my) for j in range(window))
        var_x = sum((wx[j] - mx) ** 2 for j in range(window))
        var_y = sum((wy[j] - my) ** 2 for j in range(window))

        denom = math.sqrt(var_x * var_y)
        if denom == 0.0:
            result[i] = NAN
        else:
            result[i] = cov / denom

    return result


def rolling_beta(
    returns: List[float],
    benchmark: List[float],
    window: int,
) -> List[float]:
    """Rolling window beta: cov(returns, benchmark) / var(benchmark).

    Parameters
    ----------
    returns : List[float]
        Asset return series.
    benchmark : List[float]
        Benchmark return series.
    window : int
        Rolling window size.

    Returns
    -------
    List[float]
        Rolling beta. First ``window - 1`` values are NaN.
    """
    n = len(returns)
    if n != len(benchmark):
        raise ValueError("returns and benchmark must have the same length")
    if window <= 1:
        raise ValueError("window must be > 1")

    result: List[float] = [NAN] * n
    if window > n:
        return result

    for i in range(window - 1, n):
        wr = returns[i - window + 1: i + 1]
        wb = benchmark[i - window + 1: i + 1]
        mr = sum(wr) / window
        mb = sum(wb) / window

        cov = sum(
            (wr[j] - mr) * (wb[j] - mb) for j in range(window)
        )
        var_b = sum((wb[j] - mb) ** 2 for j in range(window))

        if var_b == 0.0:
            result[i] = NAN
        else:
            result[i] = cov / var_b

    return result


def expanding_mean(data: List[float]) -> List[float]:
    """Expanding (cumulative) mean.

    Parameters
    ----------
    data : List[float]
        Input time series.

    Returns
    -------
    List[float]
        Cumulative mean at each position.
    """
    n = len(data)
    if n == 0:
        return []
    result: List[float] = [0.0] * n
    s = 0.0
    for i in range(n):
        s += data[i]
        result[i] = s / (i + 1)
    return result


def expanding_std(
    data: List[float], ddof: int = 1
) -> List[float]:
    """Expanding (cumulative) standard deviation.

    Parameters
    ----------
    data : List[float]
        Input time series.
    ddof : int, optional
        Delta degrees of freedom, by default 1.

    Returns
    -------
    List[float]
        Cumulative std. First ``ddof`` values are NaN (insufficient data).
    """
    n = len(data)
    result: List[float] = [NAN] * n
    if n <= ddof:
        return result

    s = 0.0
    s2 = 0.0
    for i in range(n):
        s += data[i]
        s2 += data[i] ** 2
        count = i + 1
        if count <= ddof:
            continue
        mean = s / count
        var = (s2 / count - mean ** 2) * count / (count - ddof)
        result[i] = math.sqrt(max(var, 0.0))
    return result
