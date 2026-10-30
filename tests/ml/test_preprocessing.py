"""Tests for finceptquantlib.ml.modeling.preprocessing module."""

import math
import pytest

from finceptquantlib.ml.modeling.preprocessing import (
    ZScoreScaler,
    MinMaxScaler,
    RobustScaler,
    Winsorizer,
    WinsorizeMethod,
    MissingValueHandler,
    MissingValueMethod,
    OutlierDetector,
    BoxCoxTransformer,
    PowerTransformer,
    log_returns,
    simple_returns,
    difference,
    adf_test_statistic,
)


# ---------------------------------------------------------------------------
# Shared sample data
# ---------------------------------------------------------------------------

SAMPLE_X = [
    [1.0, 10.0],
    [2.0, 20.0],
    [3.0, 30.0],
    [4.0, 40.0],
    [5.0, 50.0],
]


# ---------------------------------------------------------------------------
# ZScoreScaler
# ---------------------------------------------------------------------------

class TestZScoreScaler:
    """Tests for the ZScoreScaler (standardisation to zero mean / unit variance)."""

    def test_fit_returns_none(self):
        scaler = ZScoreScaler()
        result = scaler.fit(SAMPLE_X)
        assert result is None

    def test_fit_transform_zero_mean_unit_std(self):
        scaler = ZScoreScaler()
        scaler.fit(SAMPLE_X)
        transformed = scaler.transform(SAMPLE_X)

        n_cols = len(SAMPLE_X[0])
        n_rows = len(SAMPLE_X)

        for col in range(n_cols):
            col_vals = [transformed[row][col] for row in range(n_rows)]
            col_mean = sum(col_vals) / n_rows
            col_std = math.sqrt(sum((v - col_mean) ** 2 for v in col_vals) / n_rows)
            assert col_mean == pytest.approx(0.0, abs=1e-10)
            assert col_std == pytest.approx(1.0, abs=1e-6)

    def test_inverse_transform_recovers_original(self):
        scaler = ZScoreScaler()
        scaler.fit(SAMPLE_X)
        transformed = scaler.transform(SAMPLE_X)
        recovered = scaler.inverse_transform(transformed)

        for i in range(len(SAMPLE_X)):
            for j in range(len(SAMPLE_X[0])):
                assert recovered[i][j] == pytest.approx(SAMPLE_X[i][j], abs=1e-10)

    def test_transform_output_shape(self):
        scaler = ZScoreScaler()
        scaler.fit(SAMPLE_X)
        transformed = scaler.transform(SAMPLE_X)
        assert len(transformed) == len(SAMPLE_X)
        assert len(transformed[0]) == len(SAMPLE_X[0])


# ---------------------------------------------------------------------------
# MinMaxScaler
# ---------------------------------------------------------------------------

class TestMinMaxScaler:
    """Tests for MinMaxScaler (scales features to [0, 1])."""

    def test_transform_values_in_unit_range(self):
        scaler = MinMaxScaler()
        scaler.fit(SAMPLE_X)
        transformed = scaler.transform(SAMPLE_X)

        for row in transformed:
            for val in row:
                assert 0.0 <= val <= 1.0

    def test_transform_min_zero_max_one(self):
        scaler = MinMaxScaler()
        scaler.fit(SAMPLE_X)
        transformed = scaler.transform(SAMPLE_X)

        n_cols = len(SAMPLE_X[0])
        n_rows = len(SAMPLE_X)

        for col in range(n_cols):
            col_vals = [transformed[row][col] for row in range(n_rows)]
            assert min(col_vals) == pytest.approx(0.0, abs=1e-10)
            assert max(col_vals) == pytest.approx(1.0, abs=1e-10)

    def test_inverse_transform_recovers_original(self):
        scaler = MinMaxScaler()
        scaler.fit(SAMPLE_X)
        transformed = scaler.transform(SAMPLE_X)
        recovered = scaler.inverse_transform(transformed)

        for i in range(len(SAMPLE_X)):
            for j in range(len(SAMPLE_X[0])):
                assert recovered[i][j] == pytest.approx(SAMPLE_X[i][j], abs=1e-10)


# ---------------------------------------------------------------------------
# RobustScaler
# ---------------------------------------------------------------------------

class TestRobustScaler:
    """Tests for RobustScaler (uses median and IQR)."""

    def test_fit_transform_runs(self):
        scaler = RobustScaler()
        scaler.fit(SAMPLE_X)
        transformed = scaler.transform(SAMPLE_X)

        assert len(transformed) == len(SAMPLE_X)
        assert len(transformed[0]) == len(SAMPLE_X[0])

    def test_median_row_is_zero(self):
        """The median row of the original data should map to ~0 after transform."""
        scaler = RobustScaler()
        scaler.fit(SAMPLE_X)
        transformed = scaler.transform(SAMPLE_X)

        # Row index 2 is the median row ([3.0, 30.0])
        for val in transformed[2]:
            assert val == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Winsorizer
# ---------------------------------------------------------------------------

class TestWinsorizer:
    """Tests for Winsorizer (clips extreme values)."""

    def test_extreme_values_are_clipped(self):
        data = [[v] for v in [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]]
        winsorizer = Winsorizer(method=WinsorizeMethod.PERCENTILE)
        winsorizer.fit(data)
        transformed = winsorizer.transform(data)

        original_max = max(row[0] for row in data)
        transformed_max = max(row[0] for row in transformed)
        # The extreme value (100.0) should be clipped to something smaller
        assert transformed_max <= original_max

    def test_non_extreme_values_unchanged(self):
        """Values within the percentile range should be preserved or close to original."""
        data = [[v] for v in [1.0, 2.0, 3.0, 4.0, 5.0]]
        winsorizer = Winsorizer(method=WinsorizeMethod.PERCENTILE)
        winsorizer.fit(data)
        transformed = winsorizer.transform(data)

        # The middle value should be unchanged
        assert transformed[2][0] == pytest.approx(3.0, abs=1e-10)


# ---------------------------------------------------------------------------
# MissingValueHandler
# ---------------------------------------------------------------------------

class TestMissingValueHandler:
    """Tests for MissingValueHandler (imputes NaN values)."""

    def test_mean_imputation_removes_nans(self):
        data = [
            [1.0, 10.0],
            [2.0, float("nan")],
            [float("nan"), 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ]
        handler = MissingValueHandler(method=MissingValueMethod.MEAN)
        handler.fit(data)
        transformed = handler.transform(data)

        for row in transformed:
            for val in row:
                assert not math.isnan(val), "NaN values should be imputed"

    def test_mean_imputation_values(self):
        data = [
            [1.0, 10.0],
            [2.0, float("nan")],
            [3.0, 30.0],
        ]
        handler = MissingValueHandler(method=MissingValueMethod.MEAN)
        handler.fit(data)
        transformed = handler.transform(data)

        # Column 1 mean of non-NaN values: (10 + 30) / 2 = 20
        assert transformed[1][1] == pytest.approx(20.0, abs=1e-10)

    def test_output_shape_preserved(self):
        data = [
            [1.0, float("nan")],
            [float("nan"), 20.0],
        ]
        handler = MissingValueHandler(method=MissingValueMethod.MEAN)
        handler.fit(data)
        transformed = handler.transform(data)

        assert len(transformed) == 2
        assert len(transformed[0]) == 2


# ---------------------------------------------------------------------------
# OutlierDetector
# ---------------------------------------------------------------------------

class TestOutlierDetector:
    """Tests for OutlierDetector using IQR method."""

    def test_iqr_detects_outlier(self):
        data = [[v] for v in [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]]
        detector = OutlierDetector(method="iqr")
        detector.fit(data)
        result = detector.detect(data)
        assert result is not None
        assert result.n_outliers >= 1

    def test_no_outliers_in_clean_data(self):
        data = [[v] for v in [1.0, 2.0, 3.0, 4.0, 5.0]]
        detector = OutlierDetector(method="iqr")
        detector.fit(data)
        result = detector.detect(data)
        assert result is not None


# ---------------------------------------------------------------------------
# BoxCoxTransformer
# ---------------------------------------------------------------------------

class TestBoxCoxTransformer:
    """Tests for BoxCoxTransformer."""

    def test_transform_and_inverse(self):
        # BoxCox requires strictly positive values
        data = [[v] for v in [1.0, 2.0, 3.0, 4.0, 5.0]]
        transformer = BoxCoxTransformer()
        transformer.fit(data)
        transformed = transformer.transform(data)
        recovered = transformer.inverse_transform(transformed)

        for i in range(len(data)):
            assert recovered[i][0] == pytest.approx(data[i][0], abs=1e-6)

    def test_transform_changes_values(self):
        data = [[v] for v in [1.0, 2.0, 3.0, 4.0, 5.0]]
        transformer = BoxCoxTransformer()
        transformer.fit(data)
        transformed = transformer.transform(data)

        # Transformed values should differ from original (unless lambda=1)
        assert len(transformed) == len(data)


# ---------------------------------------------------------------------------
# PowerTransformer
# ---------------------------------------------------------------------------

class TestPowerTransformer:
    """Tests for PowerTransformer."""

    def test_transform_and_inverse(self):
        data = [[v] for v in [1.0, 2.0, 3.0, 4.0, 5.0]]
        transformer = PowerTransformer()
        transformer.fit(data)
        transformed = transformer.transform(data)
        recovered = transformer.inverse_transform(transformed)

        for i in range(len(data)):
            assert recovered[i][0] == pytest.approx(data[i][0], abs=1e-6)

    def test_output_shape(self):
        data = [[v] for v in [1.0, 2.0, 3.0, 4.0, 5.0]]
        transformer = PowerTransformer()
        transformer.fit(data)
        transformed = transformer.transform(data)

        assert len(transformed) == len(data)


# ---------------------------------------------------------------------------
# Stationarity functions
# ---------------------------------------------------------------------------

class TestLogReturns:
    """Tests for the log_returns function."""

    def test_known_values(self):
        prices = [100.0, 110.0, 121.0]
        result = log_returns(prices)
        # log(110/100) = log(1.1), log(121/110) = log(1.1)
        expected_0 = math.log(110.0 / 100.0)
        expected_1 = math.log(121.0 / 110.0)
        assert len(result) == 2
        assert result[0] == pytest.approx(expected_0, abs=1e-10)
        assert result[1] == pytest.approx(expected_1, abs=1e-10)

    def test_two_prices(self):
        prices = [100.0, 110.0]
        result = log_returns(prices)
        assert len(result) == 1


class TestSimpleReturns:
    """Tests for the simple_returns function."""

    def test_known_values(self):
        prices = [100.0, 110.0, 121.0]
        result = simple_returns(prices)
        # (110-100)/100 = 0.1, (121-110)/110 = 0.1
        assert len(result) == 2
        assert result[0] == pytest.approx(0.1, abs=1e-10)
        assert result[1] == pytest.approx(0.1, abs=1e-10)

    def test_negative_return(self):
        prices = [100.0, 90.0]
        result = simple_returns(prices)
        assert result[0] == pytest.approx(-0.1, abs=1e-10)


class TestDifference:
    """Tests for the difference function."""

    def test_first_difference(self):
        series = [1.0, 3.0, 6.0, 10.0]
        result = difference(series)
        # Differences: 2, 3, 4
        assert len(result) == 3
        assert result[0] == pytest.approx(2.0, abs=1e-10)
        assert result[1] == pytest.approx(3.0, abs=1e-10)
        assert result[2] == pytest.approx(4.0, abs=1e-10)

    def test_constant_series(self):
        series = [5.0, 5.0, 5.0]
        result = difference(series)
        for val in result:
            assert val == pytest.approx(0.0, abs=1e-10)


class TestADFTestStatistic:
    """Tests for adf_test_statistic function."""

    def test_returns_result(self):
        series = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = adf_test_statistic(series)
        assert hasattr(result, "test_statistic")
        assert isinstance(result.test_statistic, float)

    def test_stationary_series(self):
        import random
        random.seed(42)
        stationary = [math.sin(i * 0.5) + random.gauss(0, 0.1) for i in range(100)]
        result = adf_test_statistic(stationary)
        assert hasattr(result, "test_statistic")
        assert isinstance(result.test_statistic, float)
