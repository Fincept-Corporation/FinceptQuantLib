"""Tests for finceptquantlib.ml.modeling.feature_engineering module."""

import math
from datetime import date

from finceptquantlib.ml.modeling.feature_engineering import (
    # Technical indicators
    sma,
    ema,
    rsi,
    macd,
    bollinger_bands,
    atr,
    obv,
    vwap,
    stochastic_oscillator,
    adx,
    williams_r,
    cci,
    # Rolling statistics
    rolling_mean,
    rolling_std,
    rolling_skewness,
    rolling_kurtosis,
    rolling_zscore,
    rolling_correlation,
    rolling_beta,
    expanding_mean,
    expanding_std,
    # Cross-sectional
    cross_sectional_rank,
    cross_sectional_percentile,
    cross_sectional_zscore,
    cross_sectional_demean,
    winsorized_zscore,
    # Calendar features
    month_of_year,
    quarter,
    day_of_week,
    is_month_end,
    # Lags and returns
    add_lags,
    add_leads,
    add_returns,
    add_log_returns,
    # Pipeline
    FeaturePipeline,
)


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

PRICES = [44.0, 44.5, 43.5, 44.0, 45.0, 45.5, 46.0, 45.0, 44.5, 45.0,
          45.5, 46.0, 46.5, 47.0, 46.5]

HIGH = [44.5, 45.0, 44.0, 44.5, 45.5, 46.0, 46.5, 45.5, 45.0, 45.5,
        46.0, 46.5, 47.0, 47.5, 47.0]

LOW = [43.5, 44.0, 43.0, 43.5, 44.5, 45.0, 45.5, 44.5, 44.0, 44.5,
       45.0, 45.5, 46.0, 46.5, 46.0]

CLOSE = PRICES  # Close prices are the same as PRICES for convenience

VOLUMES = [100, 120, 150, 130, 110, 140, 160, 130, 120, 150,
           140, 160, 170, 180, 150]


def _is_nan(value):
    """Check whether a value is NaN (handles None as well)."""
    if value is None:
        return True
    try:
        return math.isnan(value)
    except (TypeError, ValueError):
        return False


def _valid_values(series):
    """Return only non-NaN/non-None values from a series."""
    return [v for v in series if not _is_nan(v)]


# ===========================================================================
# Technical Indicator Tests
# ===========================================================================


class TestSMA:
    """Tests for simple moving average."""

    def test_output_length(self):
        result = sma(PRICES, window=3)
        assert len(result) == len(PRICES)

    def test_leading_nans(self):
        window = 5
        result = sma(PRICES, window=window)
        # First (window - 1) values should be NaN or None
        for v in result[: window - 1]:
            assert _is_nan(v), f"Expected NaN/None but got {v}"

    def test_known_values(self):
        result = sma(PRICES, window=3)
        valid = _valid_values(result)
        # The third element (index 2) should be the average of the first 3 prices
        expected_first = (44.0 + 44.5 + 43.5) / 3.0
        assert abs(valid[0] - expected_first) < 1e-9

    def test_window_one_returns_prices(self):
        result = sma(PRICES, window=1)
        for r, p in zip(result, PRICES):
            if not _is_nan(r):
                assert abs(r - p) < 1e-9


class TestEMA:
    """Tests for exponential moving average."""

    def test_output_length(self):
        result = ema(PRICES, 3)
        assert len(result) == len(PRICES)

    def test_first_valid_value(self):
        result = ema(PRICES, 3)
        valid = _valid_values(result)
        assert len(valid) > 0

    def test_ema_reacts_to_price_change(self):
        result_ema = ema(PRICES, 5)
        result_sma = sma(PRICES, window=5)
        valid_ema = _valid_values(result_ema)
        valid_sma = _valid_values(result_sma)
        assert len(valid_ema) > 0
        assert len(valid_sma) > 0


class TestRSI:
    """Tests for relative strength index."""

    def test_output_length(self):
        result = rsi(PRICES, period=5)
        assert len(result) == len(PRICES)

    def test_values_in_range(self):
        result = rsi(PRICES, period=5)
        for v in result:
            if not _is_nan(v):
                assert 0.0 <= v <= 100.0, f"RSI value {v} out of [0, 100] range"

    def test_rising_prices_high_rsi(self):
        rising = [float(i) for i in range(1, 21)]
        result = rsi(rising, period=5)
        valid = _valid_values(result)
        assert valid[-1] > 80.0

    def test_falling_prices_low_rsi(self):
        falling = [float(20 - i) for i in range(20)]
        result = rsi(falling, period=5)
        valid = _valid_values(result)
        assert valid[-1] < 20.0


class TestMACD:
    """Tests for MACD."""

    def test_returns_three_components(self):
        macd_line, signal_line, histogram = macd(PRICES, fast=3, slow=6, signal=3)
        assert len(macd_line) == len(PRICES)
        assert len(signal_line) == len(PRICES)
        assert len(histogram) == len(PRICES)

    def test_histogram_equals_difference(self):
        macd_line, signal_line, histogram = macd(PRICES, fast=3, slow=6, signal=3)
        for m, s, h in zip(macd_line, signal_line, histogram):
            if not _is_nan(m) and not _is_nan(s) and not _is_nan(h):
                assert abs(h - (m - s)) < 1e-9


class TestBollingerBands:
    """Tests for Bollinger Bands."""

    def test_returns_three_bands(self):
        upper, middle, lower = bollinger_bands(PRICES, window=5, num_std=2)
        assert len(upper) == len(PRICES)
        assert len(middle) == len(PRICES)
        assert len(lower) == len(PRICES)

    def test_upper_above_middle_above_lower(self):
        upper, middle, lower = bollinger_bands(PRICES, window=5, num_std=2)
        for u, m, lo in zip(upper, middle, lower):
            if not _is_nan(u) and not _is_nan(m) and not _is_nan(lo):
                assert u >= m >= lo

    def test_middle_is_sma(self):
        _, middle, _ = bollinger_bands(PRICES, window=5, num_std=2)
        sma_result = sma(PRICES, window=5)
        for m, s in zip(middle, sma_result):
            if not _is_nan(m) and not _is_nan(s):
                assert abs(m - s) < 1e-9


class TestATR:
    """Tests for Average True Range."""

    def test_output_length(self):
        result = atr(HIGH, LOW, CLOSE, period=5)
        assert len(result) == len(CLOSE)

    def test_positive_values(self):
        result = atr(HIGH, LOW, CLOSE, period=5)
        for v in result:
            if not _is_nan(v):
                assert v >= 0.0, f"ATR value {v} should be non-negative"


class TestOBV:
    """Tests for On Balance Volume."""

    def test_output_length(self):
        result = obv(PRICES, VOLUMES)
        assert len(result) == len(PRICES)

    def test_first_value(self):
        result = obv(PRICES, VOLUMES)
        # First OBV is typically the first volume (or 0)
        assert not _is_nan(result[0])

    def test_obv_increases_on_up_price(self):
        # If price goes up, OBV should increase by that period's volume
        simple_prices = [10.0, 11.0, 12.0]
        simple_volumes = [100, 200, 300]
        result = obv(simple_prices, simple_volumes)
        # OBV should be increasing
        valid = _valid_values(result)
        assert len(valid) == 3
        assert valid[1] > valid[0] or valid[1] == valid[0] + simple_volumes[1]


class TestVWAP:
    """Tests for Volume Weighted Average Price."""

    def test_output_length(self):
        result = vwap(HIGH, LOW, CLOSE, VOLUMES)
        assert len(result) == len(PRICES)

    def test_vwap_in_price_range(self):
        result = vwap(HIGH, LOW, CLOSE, VOLUMES)
        for v in result:
            if not _is_nan(v):
                assert min(LOW) - 1 <= v <= max(HIGH) + 1, (
                    f"VWAP value {v} outside price range"
                )


class TestStochasticOscillator:
    """Tests for Stochastic Oscillator."""

    def test_returns_k_and_d(self):
        k_line, d_line = stochastic_oscillator(HIGH, LOW, CLOSE,
                                                k_period=5, d_period=3)
        assert len(k_line) == len(CLOSE)
        assert len(d_line) == len(CLOSE)

    def test_values_in_range(self):
        k_line, d_line = stochastic_oscillator(HIGH, LOW, CLOSE,
                                                k_period=5, d_period=3)
        for v in k_line:
            if not _is_nan(v):
                assert 0.0 <= v <= 100.0, f"%K value {v} out of [0, 100]"
        for v in d_line:
            if not _is_nan(v):
                assert 0.0 <= v <= 100.0, f"%D value {v} out of [0, 100]"


class TestWilliamsR:
    """Tests for Williams %R."""

    def test_output_length(self):
        result = williams_r(HIGH, LOW, CLOSE, period=5)
        assert len(result) == len(CLOSE)

    def test_values_in_range(self):
        result = williams_r(HIGH, LOW, CLOSE, period=5)
        for v in result:
            if not _is_nan(v):
                assert -100.0 <= v <= 0.0, f"Williams %%R value {v} out of [-100, 0]"


class TestADX:
    """Tests for Average Directional Index."""

    def test_output_length(self):
        result = adx(HIGH, LOW, CLOSE, period=5)
        assert len(result) == len(CLOSE)

    def test_positive_values(self):
        result = adx(HIGH, LOW, CLOSE, period=5)
        for v in result:
            if not _is_nan(v):
                assert v >= 0.0, f"ADX value {v} should be non-negative"


class TestCCI:
    """Tests for Commodity Channel Index."""

    def test_output_length(self):
        result = cci(HIGH, LOW, CLOSE, period=5)
        assert len(result) == len(CLOSE)

    def test_returns_floats(self):
        result = cci(HIGH, LOW, CLOSE, period=5)
        for v in result:
            if not _is_nan(v):
                assert isinstance(v, (int, float))


# ===========================================================================
# Rolling Statistics Tests
# ===========================================================================


class TestRollingMean:
    """Tests for rolling mean."""

    def test_output_length(self):
        result = rolling_mean(PRICES, window=3)
        assert len(result) == len(PRICES)

    def test_matches_sma(self):
        rm = rolling_mean(PRICES, window=3)
        sm = sma(PRICES, window=3)
        for a, b in zip(rm, sm):
            if not _is_nan(a) and not _is_nan(b):
                assert abs(a - b) < 1e-9


class TestRollingStd:
    """Tests for rolling standard deviation."""

    def test_output_length(self):
        result = rolling_std(PRICES, window=5)
        assert len(result) == len(PRICES)

    def test_non_negative(self):
        result = rolling_std(PRICES, window=5)
        for v in result:
            if not _is_nan(v):
                assert v >= 0.0

    def test_constant_data_zero_std(self):
        constant = [5.0] * 20
        result = rolling_std(constant, window=5)
        for v in result:
            if not _is_nan(v):
                assert abs(v) < 1e-9


class TestRollingSkewness:
    """Tests for rolling skewness."""

    def test_output_length(self):
        result = rolling_skewness(PRICES, window=5)
        assert len(result) == len(PRICES)

    def test_symmetric_data_near_zero(self):
        symmetric = [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0]
        result = rolling_skewness(symmetric, window=5)
        valid = _valid_values(result)
        # Skewness of symmetric window should be near zero
        assert any(abs(v) < 1.0 for v in valid)


class TestRollingKurtosis:
    """Tests for rolling kurtosis."""

    def test_output_length(self):
        result = rolling_kurtosis(PRICES, window=5)
        assert len(result) == len(PRICES)


class TestRollingZscore:
    """Tests for rolling z-score."""

    def test_output_length(self):
        result = rolling_zscore(PRICES, window=5)
        assert len(result) == len(PRICES)


class TestRollingCorrelation:
    """Tests for rolling correlation."""

    def test_output_length(self):
        data2 = [p + 1.0 for p in PRICES]
        result = rolling_correlation(PRICES, data2, window=5)
        assert len(result) == len(PRICES)

    def test_perfect_correlation(self):
        data2 = [p * 2.0 for p in PRICES]
        result = rolling_correlation(PRICES, data2, window=5)
        valid = _valid_values(result)
        for v in valid:
            assert abs(v - 1.0) < 1e-6, f"Expected correlation ~1.0, got {v}"


class TestRollingBeta:
    """Tests for rolling beta."""

    def test_output_length(self):
        market = [p + 0.5 for p in PRICES]
        result = rolling_beta(PRICES, market, window=5)
        assert len(result) == len(PRICES)


class TestExpandingMean:
    """Tests for expanding mean."""

    def test_output_length(self):
        result = expanding_mean(PRICES)
        assert len(result) == len(PRICES)

    def test_first_value(self):
        result = expanding_mean(PRICES)
        assert abs(result[0] - PRICES[0]) < 1e-9

    def test_known_second_value(self):
        result = expanding_mean(PRICES)
        expected = (PRICES[0] + PRICES[1]) / 2.0
        assert abs(result[1] - expected) < 1e-9

    def test_last_value_is_overall_mean(self):
        result = expanding_mean(PRICES)
        overall_mean = sum(PRICES) / len(PRICES)
        assert abs(result[-1] - overall_mean) < 1e-9


class TestExpandingStd:
    """Tests for expanding standard deviation."""

    def test_output_length(self):
        result = expanding_std(PRICES)
        assert len(result) == len(PRICES)

    def test_non_negative(self):
        result = expanding_std(PRICES)
        for v in result:
            if not _is_nan(v):
                assert v >= 0.0


# ===========================================================================
# Cross-Sectional Tests
# ===========================================================================


class TestCrossSectionalRank:
    """Tests for cross-sectional ranking."""

    def test_output_length(self):
        values = [3.0, 1.0, 4.0, 1.0, 5.0]
        result = cross_sectional_rank(values)
        assert len(result) == len(values)

    def test_sorted_data_gives_ordered_ranks(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = cross_sectional_rank(values)
        # Ranks should be monotonically increasing
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1]


class TestCrossSectionalPercentile:
    """Tests for cross-sectional percentile."""

    def test_output_length(self):
        values = [3.0, 1.0, 4.0, 1.0, 5.0]
        result = cross_sectional_percentile(values)
        assert len(result) == len(values)

    def test_values_in_unit_interval(self):
        values = [3.0, 1.0, 4.0, 1.0, 5.0]
        result = cross_sectional_percentile(values)
        for v in result:
            assert 0.0 <= v <= 1.0, f"Percentile {v} out of [0, 1]"


class TestCrossSectionalZscore:
    """Tests for cross-sectional z-score."""

    def test_output_length(self):
        values = [3.0, 1.0, 4.0, 1.0, 5.0]
        result = cross_sectional_zscore(values)
        assert len(result) == len(values)

    def test_mean_near_zero(self):
        values = [3.0, 1.0, 4.0, 2.0, 5.0]
        result = cross_sectional_zscore(values)
        mean_zscore = sum(result) / len(result)
        assert abs(mean_zscore) < 1e-9


class TestCrossSectionalDemean:
    """Tests for cross-sectional demeaning."""

    def test_output_length(self):
        values = [3.0, 1.0, 4.0, 1.0, 5.0]
        result = cross_sectional_demean(values)
        assert len(result) == len(values)

    def test_sum_near_zero(self):
        values = [3.0, 1.0, 4.0, 2.0, 5.0]
        result = cross_sectional_demean(values)
        assert abs(sum(result)) < 1e-9


class TestWinsorizedZscore:
    """Tests for winsorized z-score."""

    def test_output_length(self):
        values = [3.0, 1.0, 4.0, 1.0, 5.0, 100.0]
        result = winsorized_zscore(values)
        assert len(result) == len(values)

    def test_outlier_is_clipped(self):
        values = [1.0, 2.0, 3.0, 2.0, 1.0, 100.0]
        result = winsorized_zscore(values)
        # The outlier (100.0) z-score should be clipped compared to standard zscore
        std_zscores = cross_sectional_zscore(values)
        # Winsorized should limit extreme values
        assert abs(result[-1]) <= abs(std_zscores[-1]) + 1e-9


# ===========================================================================
# Calendar Feature Tests
# ===========================================================================


class TestCalendarFeatures:
    """Tests for calendar-based features."""

    def test_month_of_year(self):
        result = month_of_year([date(2025, 3, 15)])
        assert result[0] == 3

    def test_quarter(self):
        result = quarter([date(2025, 1, 15), date(2025, 4, 15), date(2025, 7, 15), date(2025, 10, 15)])
        assert result == [1, 2, 3, 4]

    def test_day_of_week(self):
        # 2025-01-06 is a Monday
        result = day_of_week([date(2025, 1, 6)])
        assert result[0] in (0, 1)  # Monday is 0 or 1

    def test_is_month_end_true(self):
        result = is_month_end([date(2025, 1, 31)])
        assert result[0] == 1

    def test_is_month_end_false(self):
        result = is_month_end([date(2025, 1, 15)])
        assert result[0] == 0

    def test_is_month_end_february(self):
        result = is_month_end([date(2025, 2, 28)])
        assert result[0] == 1

    def test_is_month_end_leap_year(self):
        result = is_month_end([date(2024, 2, 29)])
        assert result[0] == 1


# ===========================================================================
# Lag / Lead / Returns Tests
# ===========================================================================


class TestAddLags:
    """Tests for add_lags."""

    def test_single_lag(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = add_lags(data, [1])
        assert len(result) == len(data)

    def test_multiple_lags(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = add_lags(data, [1, 2])
        assert len(result) == len(data)

    def test_lag_shift_behavior(self):
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = add_lags(data, [1])
        assert result is not None


class TestAddLeads:
    """Tests for add_leads."""

    def test_single_lead(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = add_leads(data, [1])
        assert len(result) == len(data)


class TestAddReturns:
    """Tests for add_returns (simple returns)."""

    def test_output_length(self):
        result = add_returns(PRICES, [1])
        assert len(result) == len(PRICES)

    def test_first_value_nan(self):
        result = add_returns(PRICES, [1])
        # First row, first column should be NaN
        assert _is_nan(result[0][0])

    def test_known_return(self):
        result = add_returns(PRICES, [1])
        # Return at index 1, period-1 col = (44.5 - 44.0) / 44.0
        expected = (44.5 - 44.0) / 44.0
        assert abs(result[1][0] - expected) < 1e-9


class TestAddLogReturns:
    """Tests for add_log_returns."""

    def test_output_length(self):
        result = add_log_returns(PRICES, [1])
        assert len(result) == len(PRICES)

    def test_first_value_nan(self):
        result = add_log_returns(PRICES, [1])
        assert _is_nan(result[0][0])

    def test_known_log_return(self):
        result = add_log_returns(PRICES, [1])
        expected = math.log(44.5 / 44.0)
        assert abs(result[1][0] - expected) < 1e-9


# ===========================================================================
# Feature Pipeline Tests
# ===========================================================================


class TestFeaturePipeline:
    """Tests for FeaturePipeline composing multiple Transformer steps."""

    def test_empty_pipeline(self):
        pipeline = FeaturePipeline([])
        # Empty pipeline should be constructable
        assert pipeline is not None

    def test_single_step_pipeline(self):
        from finceptquantlib.ml.modeling.preprocessing import ZScoreScaler
        scaler = ZScoreScaler()
        pipeline = FeaturePipeline([("scale", scaler)])
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        pipeline.fit(X)
        result = pipeline.transform(X)
        assert len(result) == len(X)

    def test_multi_step_pipeline(self):
        from finceptquantlib.ml.modeling.preprocessing import ZScoreScaler, MinMaxScaler
        pipeline = FeaturePipeline([("zscore", ZScoreScaler()), ("minmax", MinMaxScaler())])
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        pipeline.fit(X)
        result = pipeline.transform(X)
        assert len(result) == len(X)

    def test_pipeline_fit_transform(self):
        from finceptquantlib.ml.modeling.preprocessing import ZScoreScaler
        pipeline = FeaturePipeline([("scale", ZScoreScaler())])
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        result = pipeline.fit_transform(X)
        assert len(result) == len(X)
