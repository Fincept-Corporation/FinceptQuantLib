"""
Technical Analysis Indicators

Pure Python implementations of common technical analysis indicators
for financial time series. All functions operate on List[float] inputs
and return List[float] outputs with NaN for insufficient data.

Indicators
----------
- SMA, EMA: Moving averages
- RSI: Relative Strength Index (Wilder's smoothing)
- MACD: Moving Average Convergence Divergence
- Bollinger Bands: Volatility bands around SMA
- ATR: Average True Range (Wilder's smoothing)
- OBV: On-Balance Volume
- VWAP: Volume Weighted Average Price
- Stochastic Oscillator: %K and %D
- ADX: Average Directional Index
- Williams %R: Momentum oscillator
- CCI: Commodity Channel Index
"""

from __future__ import annotations

import math
from typing import List, Tuple

NAN = float("nan")


def _is_nan(x: float) -> bool:
    """Check if a value is NaN."""
    return math.isnan(x)


def sma(prices: List[float], window: int) -> List[float]:
    """Simple Moving Average.

    Parameters
    ----------
    prices : List[float]
        Price series.
    window : int
        Number of periods to average.

    Returns
    -------
    List[float]
        SMA values. First ``window - 1`` values are NaN.

    Examples
    --------
    >>> sma([1.0, 2.0, 3.0, 4.0, 5.0], 3)
    [nan, nan, 2.0, 3.0, 4.0]
    """
    n = len(prices)
    if window <= 0:
        raise ValueError("window must be positive")
    result: List[float] = [NAN] * n
    if window > n:
        return result

    running_sum = sum(prices[:window])
    result[window - 1] = running_sum / window
    for i in range(window, n):
        running_sum += prices[i] - prices[i - window]
        result[i] = running_sum / window
    return result


def ema(prices: List[float], span: int) -> List[float]:
    """Exponential Moving Average.

    Uses alpha = 2 / (span + 1). The first value is seeded with the
    SMA of the first ``span`` data points.

    Parameters
    ----------
    prices : List[float]
        Price series.
    span : int
        EMA span (period).

    Returns
    -------
    List[float]
        EMA values. First ``span - 1`` values are NaN.

    Examples
    --------
    >>> result = ema([1.0, 2.0, 3.0, 4.0, 5.0], 3)
    >>> round(result[2], 4)
    2.0
    """
    n = len(prices)
    if span <= 0:
        raise ValueError("span must be positive")
    result: List[float] = [NAN] * n
    if span > n:
        return result

    alpha = 2.0 / (span + 1)
    # Seed with SMA of first span points
    seed = sum(prices[:span]) / span
    result[span - 1] = seed
    for i in range(span, n):
        result[i] = alpha * prices[i] + (1.0 - alpha) * result[i - 1]
    return result


def rsi(prices: List[float], period: int = 14) -> List[float]:
    """Relative Strength Index using Wilder's smoothing method.

    Parameters
    ----------
    prices : List[float]
        Price series.
    period : int, optional
        RSI look-back period, by default 14.

    Returns
    -------
    List[float]
        RSI values in [0, 100]. First ``period`` values are NaN.

    Notes
    -----
    Uses Wilder's smoothing: avg_gain = (prev_avg_gain * (period-1) + gain) / period.
    The initial average gain/loss is the simple average of the first ``period`` changes.
    """
    n = len(prices)
    if period <= 0:
        raise ValueError("period must be positive")
    result: List[float] = [NAN] * n
    if n < period + 1:
        return result

    # Calculate price changes
    changes: List[float] = [prices[i] - prices[i - 1] for i in range(1, n)]

    # Initial average gain and loss (simple average)
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(period):
        if changes[i] > 0:
            avg_gain += changes[i]
        else:
            avg_loss += abs(changes[i])
    avg_gain /= period
    avg_loss /= period

    if avg_loss == 0.0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - 100.0 / (1.0 + rs)

    # Wilder's smoothing
    for i in range(period, len(changes)):
        gain = changes[i] if changes[i] > 0 else 0.0
        loss = abs(changes[i]) if changes[i] < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0.0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return result


def macd(
    prices: List[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[List[float], List[float], List[float]]:
    """Moving Average Convergence Divergence.

    Parameters
    ----------
    prices : List[float]
        Price series.
    fast : int, optional
        Fast EMA period, by default 12.
    slow : int, optional
        Slow EMA period, by default 26.
    signal : int, optional
        Signal line EMA period, by default 9.

    Returns
    -------
    Tuple[List[float], List[float], List[float]]
        (MACD line, Signal line, Histogram).
        MACD line = fast EMA - slow EMA.
        Signal line = EMA of MACD line.
        Histogram = MACD - Signal.
    """
    n = len(prices)
    fast_ema = ema(prices, fast)
    slow_ema = ema(prices, slow)

    macd_line: List[float] = [NAN] * n
    signal_line: List[float] = [NAN] * n
    histogram: List[float] = [NAN] * n

    # MACD line starts where slow EMA is available
    for i in range(n):
        if not _is_nan(fast_ema[i]) and not _is_nan(slow_ema[i]):
            macd_line[i] = fast_ema[i] - slow_ema[i]

    # Signal line: EMA of MACD line values (skip NaN prefix)
    macd_valid: List[float] = [v for v in macd_line if not _is_nan(v)]
    if len(macd_valid) >= signal:
        signal_ema = ema(macd_valid, signal)
        start_idx = slow - 1  # first valid MACD index
        for j, val in enumerate(signal_ema):
            idx = start_idx + j
            if idx < n:
                signal_line[idx] = val

    # Histogram
    for i in range(n):
        if not _is_nan(macd_line[i]) and not _is_nan(signal_line[i]):
            histogram[i] = macd_line[i] - signal_line[i]

    return macd_line, signal_line, histogram


def bollinger_bands(
    prices: List[float],
    window: int = 20,
    num_std: float = 2.0,
) -> Tuple[List[float], List[float], List[float]]:
    """Bollinger Bands.

    Parameters
    ----------
    prices : List[float]
        Price series.
    window : int, optional
        SMA window for middle band, by default 20.
    num_std : float, optional
        Number of standard deviations for bands, by default 2.0.

    Returns
    -------
    Tuple[List[float], List[float], List[float]]
        (Upper band, Middle band, Lower band).
    """
    n = len(prices)
    middle = sma(prices, window)
    upper: List[float] = [NAN] * n
    lower: List[float] = [NAN] * n

    for i in range(window - 1, n):
        window_data = prices[i - window + 1: i + 1]
        mean = middle[i]
        variance = sum((x - mean) ** 2 for x in window_data) / window
        std = math.sqrt(variance)
        upper[i] = mean + num_std * std
        lower[i] = mean - num_std * std

    return upper, middle, lower


def atr(
    high: List[float],
    low: List[float],
    close: List[float],
    period: int = 14,
) -> List[float]:
    """Average True Range using Wilder's smoothing.

    Parameters
    ----------
    high : List[float]
        High prices.
    low : List[float]
        Low prices.
    close : List[float]
        Close prices.
    period : int, optional
        ATR period, by default 14.

    Returns
    -------
    List[float]
        ATR values. First ``period`` values are NaN.

    Notes
    -----
    True Range = max(H-L, |H-C_prev|, |L-C_prev|).
    ATR uses Wilder's smoothing after initial simple average.
    """
    n = len(high)
    if n != len(low) or n != len(close):
        raise ValueError("high, low, close must have the same length")
    if period <= 0:
        raise ValueError("period must be positive")

    result: List[float] = [NAN] * n
    if n < period + 1:
        return result

    # True Range series (starts at index 1)
    tr: List[float] = [NAN] * n
    tr[0] = high[0] - low[0]  # first TR is just H-L
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # Initial ATR is simple average of first period TRs
    initial_sum = sum(tr[i] for i in range(1, period + 1))
    result[period] = initial_sum / period

    # Wilder's smoothing
    for i in range(period + 1, n):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period

    return result


def obv(close: List[float], volume: List[float]) -> List[float]:
    """On-Balance Volume.

    Parameters
    ----------
    close : List[float]
        Close prices.
    volume : List[float]
        Volume data.

    Returns
    -------
    List[float]
        OBV values. First value equals first volume.
    """
    n = len(close)
    if n != len(volume):
        raise ValueError("close and volume must have the same length")
    if n == 0:
        return []

    result: List[float] = [0.0] * n
    result[0] = volume[0]
    for i in range(1, n):
        if close[i] > close[i - 1]:
            result[i] = result[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            result[i] = result[i - 1] - volume[i]
        else:
            result[i] = result[i - 1]
    return result


def vwap(
    high: List[float],
    low: List[float],
    close: List[float],
    volume: List[float],
) -> List[float]:
    """Volume Weighted Average Price (cumulative).

    Parameters
    ----------
    high : List[float]
        High prices.
    low : List[float]
        Low prices.
    close : List[float]
        Close prices.
    volume : List[float]
        Volume data.

    Returns
    -------
    List[float]
        Cumulative VWAP. Typical price = (H + L + C) / 3.
    """
    n = len(high)
    if n != len(low) or n != len(close) or n != len(volume):
        raise ValueError("All inputs must have the same length")
    if n == 0:
        return []

    result: List[float] = [0.0] * n
    cum_tp_vol = 0.0
    cum_vol = 0.0
    for i in range(n):
        tp = (high[i] + low[i] + close[i]) / 3.0
        cum_tp_vol += tp * volume[i]
        cum_vol += volume[i]
        result[i] = cum_tp_vol / cum_vol if cum_vol != 0.0 else NAN
    return result


def stochastic_oscillator(
    high: List[float],
    low: List[float],
    close: List[float],
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[List[float], List[float]]:
    """Stochastic Oscillator (%K and %D).

    Parameters
    ----------
    high : List[float]
        High prices.
    low : List[float]
        Low prices.
    close : List[float]
        Close prices.
    k_period : int, optional
        Look-back period for %K, by default 14.
    d_period : int, optional
        SMA period for %D, by default 3.

    Returns
    -------
    Tuple[List[float], List[float]]
        (%K line, %D line). Values in [0, 100].
    """
    n = len(high)
    if n != len(low) or n != len(close):
        raise ValueError("high, low, close must have the same length")

    k_line: List[float] = [NAN] * n

    for i in range(k_period - 1, n):
        window_high = max(high[i - k_period + 1: i + 1])
        window_low = min(low[i - k_period + 1: i + 1])
        denom = window_high - window_low
        if denom == 0.0:
            k_line[i] = 50.0  # midpoint when range is zero
        else:
            k_line[i] = 100.0 * (close[i] - window_low) / denom

    # %D is the SMA of %K; NaN where %K is NaN or there is insufficient data
    d_line_out: List[float] = [NAN] * n
    start = k_period - 1 + d_period - 1
    for i in range(start, n):
        # Compute SMA of valid %K values
        vals = k_line[i - d_period + 1: i + 1]
        if all(not _is_nan(v) for v in vals):
            d_line_out[i] = sum(vals) / d_period

    return k_line, d_line_out


def adx(
    high: List[float],
    low: List[float],
    close: List[float],
    period: int = 14,
) -> List[float]:
    """Average Directional Index.

    Parameters
    ----------
    high : List[float]
        High prices.
    low : List[float]
        Low prices.
    close : List[float]
        Close prices.
    period : int, optional
        ADX period, by default 14.

    Returns
    -------
    List[float]
        ADX values. Requires ``2 * period`` bars to produce first value.

    Notes
    -----
    Uses +DI and -DI with Wilder's smoothing for directional movement
    and true range. ADX is the smoothed average of DX.
    """
    n = len(high)
    if n != len(low) or n != len(close):
        raise ValueError("high, low, close must have the same length")

    result: List[float] = [NAN] * n
    if n < 2 * period + 1:
        return result

    # +DM and -DM
    plus_dm: List[float] = [0.0] * n
    minus_dm: List[float] = [0.0] * n
    tr_vals: List[float] = [0.0] * n

    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr_vals[i] = max(hl, hc, lc)

    # Wilder's smoothed +DM, -DM, TR (first value is sum of first period)
    smooth_plus_dm = sum(plus_dm[1: period + 1])
    smooth_minus_dm = sum(minus_dm[1: period + 1])
    smooth_tr = sum(tr_vals[1: period + 1])

    # DX series
    dx_vals: List[float] = [NAN] * n

    def _calc_dx(s_pdm: float, s_mdm: float, s_tr: float) -> float:
        if s_tr == 0.0:
            return 0.0
        plus_di = 100.0 * s_pdm / s_tr
        minus_di = 100.0 * s_mdm / s_tr
        di_sum = plus_di + minus_di
        if di_sum == 0.0:
            return 0.0
        return 100.0 * abs(plus_di - minus_di) / di_sum

    dx_vals[period] = _calc_dx(smooth_plus_dm, smooth_minus_dm, smooth_tr)

    for i in range(period + 1, n):
        smooth_plus_dm = (
            smooth_plus_dm - smooth_plus_dm / period + plus_dm[i]
        )
        smooth_minus_dm = (
            smooth_minus_dm - smooth_minus_dm / period + minus_dm[i]
        )
        smooth_tr = smooth_tr - smooth_tr / period + tr_vals[i]
        dx_vals[i] = _calc_dx(
            smooth_plus_dm, smooth_minus_dm, smooth_tr
        )

    # ADX = Wilder's smoothed DX, first value at 2*period
    first_adx_idx = 2 * period
    if first_adx_idx >= n:
        return result

    adx_sum = 0.0
    for i in range(period, first_adx_idx + 1):
        if not _is_nan(dx_vals[i]):
            adx_sum += dx_vals[i]
    result[first_adx_idx] = adx_sum / (period + 1)

    for i in range(first_adx_idx + 1, n):
        result[i] = (
            (result[i - 1] * (period - 1) + dx_vals[i]) / period
        )

    return result


def williams_r(
    high: List[float],
    low: List[float],
    close: List[float],
    period: int = 14,
) -> List[float]:
    """Williams %R.

    Parameters
    ----------
    high : List[float]
        High prices.
    low : List[float]
        Low prices.
    close : List[float]
        Close prices.
    period : int, optional
        Look-back period, by default 14.

    Returns
    -------
    List[float]
        Williams %R values in [-100, 0].
    """
    n = len(high)
    if n != len(low) or n != len(close):
        raise ValueError("high, low, close must have the same length")

    result: List[float] = [NAN] * n
    for i in range(period - 1, n):
        highest = max(high[i - period + 1: i + 1])
        lowest = min(low[i - period + 1: i + 1])
        denom = highest - lowest
        if denom == 0.0:
            result[i] = -50.0
        else:
            result[i] = -100.0 * (highest - close[i]) / denom
    return result


def cci(
    high: List[float],
    low: List[float],
    close: List[float],
    period: int = 20,
) -> List[float]:
    """Commodity Channel Index.

    Parameters
    ----------
    high : List[float]
        High prices.
    low : List[float]
        Low prices.
    close : List[float]
        Close prices.
    period : int, optional
        CCI period, by default 20.

    Returns
    -------
    List[float]
        CCI values. CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation).

    Notes
    -----
    Typical Price (TP) = (High + Low + Close) / 3.
    Mean Deviation is the average absolute deviation from the SMA of TP.
    """
    n = len(high)
    if n != len(low) or n != len(close):
        raise ValueError("high, low, close must have the same length")

    tp: List[float] = [
        (high[i] + low[i] + close[i]) / 3.0 for i in range(n)
    ]
    tp_sma = sma(tp, period)
    result: List[float] = [NAN] * n

    for i in range(period - 1, n):
        mean_tp = tp_sma[i]
        mean_dev = sum(
            abs(tp[j] - mean_tp)
            for j in range(i - period + 1, i + 1)
        ) / period
        if mean_dev == 0.0:
            result[i] = 0.0
        else:
            result[i] = (tp[i] - mean_tp) / (0.015 * mean_dev)

    return result
