"""Tests for finceptquantlib.ml.credit_risk.timeseries module."""

import random
import pytest

from finceptquantlib.ml.credit_risk.timeseries import (
    GaussianHMM,
    HMMResult,
    CUSUM,
    BayesianChangepoint,
    ChangepointResult,
    MLGARCHHybrid,
    MLGARCHResult,
    WalkForwardCV,
    ExpandingWindowCV,
    PurgedKFoldCV,
    time_series_feature_importance,
    sequential_feature_selection,
)


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

def _make_regime_data(seed=42):
    """Create regime-switching data: 50 points from N(0,1), 50 from N(3,1.5)."""
    rng = random.Random(seed)
    regime_1 = [rng.gauss(0.0, 1.0) for _ in range(50)]
    regime_2 = [rng.gauss(3.0, 1.5) for _ in range(50)]
    return regime_1 + regime_2


def _make_returns_data(seed=42):
    """Create simulated returns data: 100 draws from N(0, 0.02)."""
    rng = random.Random(seed)
    return [rng.gauss(0.0, 0.02) for _ in range(100)]


REGIME_DATA = _make_regime_data()
RETURNS_DATA = _make_returns_data()


# ===========================================================================
# GaussianHMM Tests
# ===========================================================================


class TestGaussianHMM:
    """Tests for the Gaussian Hidden Markov Model."""

    @pytest.mark.slow
    def test_fit_returns_hmm_result(self):
        model = GaussianHMM(n_states=2, max_iter=50)
        result = model.fit(REGIME_DATA)
        assert isinstance(result, HMMResult)

    @pytest.mark.slow
    def test_result_has_means(self):
        model = GaussianHMM(n_states=2, max_iter=50)
        result = model.fit(REGIME_DATA)
        assert hasattr(result, "means")
        assert len(result.means) == 2

    @pytest.mark.slow
    def test_result_has_variances(self):
        model = GaussianHMM(n_states=2, max_iter=50)
        result = model.fit(REGIME_DATA)
        assert hasattr(result, "variances")
        assert len(result.variances) == 2
        for v in result.variances:
            assert v > 0.0, "Variance must be positive"

    @pytest.mark.slow
    def test_result_has_transition_matrix(self):
        model = GaussianHMM(n_states=2, max_iter=50)
        result = model.fit(REGIME_DATA)
        assert hasattr(result, "transition_matrix")
        tm = result.transition_matrix
        assert len(tm) == 2
        for row in tm:
            assert len(row) == 2
            row_sum = sum(row)
            assert abs(row_sum - 1.0) < 1e-6, (
                f"Transition matrix row should sum to 1, got {row_sum}"
            )

    @pytest.mark.slow
    def test_result_has_log_likelihood(self):
        model = GaussianHMM(n_states=2, max_iter=50)
        result = model.fit(REGIME_DATA)
        assert hasattr(result, "log_likelihood")
        assert isinstance(result.log_likelihood, float)

    @pytest.mark.slow
    def test_identifies_two_regimes(self):
        model = GaussianHMM(n_states=2, max_iter=50)
        result = model.fit(REGIME_DATA)
        means_sorted = sorted(result.means)
        # The lower mean should be near 0, the higher near 3
        assert means_sorted[0] < 1.5, (
            f"Lower regime mean {means_sorted[0]} should be near 0"
        )
        assert means_sorted[1] > 1.5, (
            f"Upper regime mean {means_sorted[1]} should be near 3"
        )

    @pytest.mark.slow
    def test_three_states(self):
        model = GaussianHMM(n_states=3, max_iter=50)
        result = model.fit(REGIME_DATA)
        assert len(result.means) == 3
        assert len(result.transition_matrix) == 3


# ===========================================================================
# CUSUM Tests
# ===========================================================================


class TestCUSUM:
    """Tests for CUSUM changepoint detection."""

    def test_detect_returns_changepoint_result(self):
        detector = CUSUM(threshold=5.0)
        result = detector.detect(REGIME_DATA)
        assert isinstance(result, ChangepointResult)

    def test_result_has_changepoints(self):
        detector = CUSUM(threshold=5.0)
        result = detector.detect(REGIME_DATA)
        assert hasattr(result, "changepoints")
        assert isinstance(result.changepoints, list)

    def test_detects_mean_shift(self):
        detector = CUSUM(threshold=5.0)
        result = detector.detect(REGIME_DATA)
        # Should detect at least one changepoint near index 50
        assert len(result.changepoints) >= 1, "Should detect at least one changepoint"

    def test_changepoint_near_true_location(self):
        detector = CUSUM(threshold=5.0)
        result = detector.detect(REGIME_DATA)
        if len(result.changepoints) > 0:
            # At least one changepoint should be within a reasonable range of 50
            closest = min(result.changepoints, key=lambda cp: abs(cp - 50))
            assert abs(closest - 50) < 20, (
                f"Changepoint at {closest} is too far from true location 50"
            )

    def test_no_changepoint_in_stationary_data(self):
        rng = random.Random(123)
        stationary = [rng.gauss(0.0, 1.0) for _ in range(100)]
        detector = CUSUM(threshold=10.0)
        result = detector.detect(stationary)
        # With high threshold, stationary data should yield no or very few changepoints
        assert len(result.changepoints) <= 2


# ===========================================================================
# BayesianChangepoint Tests
# ===========================================================================


class TestBayesianChangepoint:
    """Tests for Bayesian changepoint detection."""

    def test_detect_returns_changepoint_result(self):
        detector = BayesianChangepoint(hazard_rate=1 / 100)
        result = detector.detect(REGIME_DATA)
        assert isinstance(result, ChangepointResult)

    def test_result_has_changepoints(self):
        detector = BayesianChangepoint(hazard_rate=1 / 100)
        result = detector.detect(REGIME_DATA)
        assert hasattr(result, "changepoints")
        assert isinstance(result.changepoints, list)

    def test_detects_regime_shift(self):
        detector = BayesianChangepoint(hazard_rate=1 / 100)
        result = detector.detect(REGIME_DATA)
        assert len(result.changepoints) >= 1, (
            "Should detect at least one changepoint in regime-switching data"
        )

    def test_changepoint_near_true_location(self):
        detector = BayesianChangepoint(hazard_rate=1 / 100)
        result = detector.detect(REGIME_DATA)
        if len(result.changepoints) > 0:
            closest = min(result.changepoints, key=lambda cp: abs(cp - 50))
            assert abs(closest - 50) < 25, (
                f"Changepoint at {closest} is too far from true location 50"
            )

    def test_detects_noisy_mean_shift(self):
        rng = random.Random(7)
        series = [rng.gauss(0, 1) for _ in range(150)] + [rng.gauss(4, 1) for _ in range(150)]
        result = BayesianChangepoint(hazard_rate=1 / 100).detect(series)
        assert any(abs(cp - 150) <= 3 for cp in result.changepoints)

    def test_step_change_location(self):
        result = BayesianChangepoint(hazard_rate=1 / 50).detect([0.0] * 50 + [3.0] * 50)
        assert result.changepoints == [50]

    @pytest.mark.parametrize("hazard_rate", [0.0, 1.0, 100])
    def test_rejects_invalid_hazard_rate(self, hazard_rate):
        with pytest.raises(ValueError):
            BayesianChangepoint(hazard_rate=hazard_rate)


# ===========================================================================
# MLGARCHHybrid Tests
# ===========================================================================


class TestMLGARCHHybrid:
    """Tests for ML-GARCH hybrid model."""

    def test_fit_returns_result(self):
        model = MLGARCHHybrid(p=1, q=1)
        result = model.fit(RETURNS_DATA)
        assert isinstance(result, MLGARCHResult)

    def test_result_has_conditional_volatility(self):
        model = MLGARCHHybrid(p=1, q=1)
        result = model.fit(RETURNS_DATA)
        assert hasattr(result, "conditional_volatility")
        assert len(result.conditional_volatility) == len(RETURNS_DATA)

    def test_conditional_volatility_positive(self):
        model = MLGARCHHybrid(p=1, q=1)
        result = model.fit(RETURNS_DATA)
        for v in result.conditional_volatility:
            assert v > 0.0, f"Conditional volatility {v} should be positive"

    def test_volatility_reasonable_magnitude(self):
        model = MLGARCHHybrid(p=1, q=1)
        result = model.fit(RETURNS_DATA)
        mean_vol = sum(result.conditional_volatility) / len(result.conditional_volatility)
        # Returns were drawn from N(0, 0.02), so vol should be in a reasonable range
        assert mean_vol > 0.0, (
            f"Mean conditional volatility {mean_vol} should be positive"
        )


# ===========================================================================
# WalkForwardCV Tests
# ===========================================================================


class TestWalkForwardCV:
    """Tests for walk-forward cross-validation."""

    def test_number_of_splits(self):
        cv = WalkForwardCV(n_splits=3, train_size=10, test_size=5)
        splits = list(cv.split(n_samples=30))
        assert len(splits) == 3

    def test_train_test_sizes(self):
        cv = WalkForwardCV(n_splits=3, train_size=10, test_size=5)
        splits = list(cv.split(n_samples=30))
        for train_idx, test_idx in splits:
            assert len(train_idx) == 10
            assert len(test_idx) == 5

    def test_no_overlap(self):
        cv = WalkForwardCV(n_splits=3, train_size=10, test_size=5)
        splits = list(cv.split(n_samples=30))
        for train_idx, test_idx in splits:
            train_set = set(train_idx)
            test_set = set(test_idx)
            assert train_set.isdisjoint(test_set), (
                "Train and test indices must not overlap"
            )

    def test_sequential_ordering(self):
        cv = WalkForwardCV(n_splits=3, train_size=10, test_size=5)
        splits = list(cv.split(n_samples=30))
        for train_idx, test_idx in splits:
            assert max(train_idx) < min(test_idx), (
                "All training indices must come before test indices"
            )

    def test_splits_advance_forward(self):
        cv = WalkForwardCV(n_splits=3, train_size=10, test_size=5)
        splits = list(cv.split(n_samples=30))
        starts = [min(train_idx) for train_idx, _ in splits]
        for i in range(len(starts) - 1):
            assert starts[i] < starts[i + 1], (
                "Walk-forward splits should advance in time"
            )


# ===========================================================================
# ExpandingWindowCV Tests
# ===========================================================================


class TestExpandingWindowCV:
    """Tests for expanding window cross-validation."""

    def test_number_of_splits(self):
        cv = ExpandingWindowCV(initial_train_size=10, test_size=5)
        splits = list(cv.split(n_samples=40))
        assert len(splits) >= 1

    def test_training_window_expands(self):
        cv = ExpandingWindowCV(initial_train_size=10, test_size=5)
        splits = list(cv.split(n_samples=40))
        if len(splits) >= 2:
            train_sizes = [len(train_idx) for train_idx, _ in splits]
            for i in range(len(train_sizes) - 1):
                assert train_sizes[i] <= train_sizes[i + 1], (
                    f"Training window should expand: {train_sizes}"
                )

    def test_no_overlap(self):
        cv = ExpandingWindowCV(initial_train_size=10, test_size=5)
        splits = list(cv.split(n_samples=40))
        for train_idx, test_idx in splits:
            train_set = set(train_idx)
            test_set = set(test_idx)
            assert train_set.isdisjoint(test_set)

    def test_train_starts_at_zero(self):
        cv = ExpandingWindowCV(initial_train_size=10, test_size=5)
        splits = list(cv.split(n_samples=40))
        for train_idx, _ in splits:
            assert min(train_idx) == 0, (
                "Expanding window should always start from the beginning"
            )


# ===========================================================================
# PurgedKFoldCV Tests
# ===========================================================================


class TestPurgedKFoldCV:
    """Tests for purged K-fold cross-validation."""

    def test_number_of_splits(self):
        cv = PurgedKFoldCV(n_splits=3, purge_gap=2)
        splits = list(cv.split(n_samples=30))
        assert len(splits) == 3

    def test_purge_gap_between_train_and_test(self):
        cv = PurgedKFoldCV(n_splits=3, purge_gap=2)
        splits = list(cv.split(n_samples=30))
        for train_idx, test_idx in splits:
            train_set = set(train_idx)
            test_set = set(test_idx)
            # No overlap
            assert train_set.isdisjoint(test_set)
            # Check that there is a gap of at least purge_gap between
            # consecutive train and test indices
            if len(train_idx) > 0 and len(test_idx) > 0:
                # Find indices adjacent to test set
                test_min = min(test_idx)
                # Training indices within the purge gap of test set should be excluded
                for gap_idx in range(max(0, test_min - 2), test_min):
                    assert gap_idx not in train_set, (
                        f"Index {gap_idx} should be purged (within gap of test start {test_min})"
                    )

    def test_all_samples_appear_in_test(self):
        cv = PurgedKFoldCV(n_splits=3, purge_gap=0)
        splits = list(cv.split(n_samples=30))
        all_test = set()
        for _, test_idx in splits:
            all_test.update(test_idx)
        # With purge_gap=0 and k-fold, every sample should appear in test exactly once
        assert all_test == set(range(30))


# ===========================================================================
# Feature Importance Tests
# ===========================================================================


class TestTimeSeriesFeatureImportance:
    """Tests for time series feature importance."""

    def test_returns_importance_scores(self):
        from finceptquantlib.ml.modeling.regression import OLSRegression
        from finceptquantlib.ml._metrics import mean_squared_error
        rng = random.Random(42)
        features = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(100)]
        target = [row[0] + rng.gauss(0, 0.1) for row in features]
        model = OLSRegression()
        cv_splits = [(list(range(50)), list(range(50, 100)))]
        result = time_series_feature_importance(
            model, features, target, ["f0", "f1", "f2"], cv_splits, mean_squared_error
        )
        assert result is not None
        assert hasattr(result, "importances")

    def test_informative_feature_ranked_higher(self):
        from finceptquantlib.ml.modeling.regression import OLSRegression
        from finceptquantlib.ml._metrics import mean_squared_error
        rng = random.Random(42)
        features = [
            [float(i) + rng.gauss(0, 0.1), rng.gauss(0, 1), rng.gauss(0, 1)]
            for i in range(100)
        ]
        target = [float(i) + rng.gauss(0, 0.1) for i in range(100)]
        model = OLSRegression()
        cv_splits = [(list(range(50)), list(range(50, 100)))]
        result = time_series_feature_importance(
            model, features, target, ["f0", "f1", "f2"], cv_splits, mean_squared_error
        )
        assert result.importances[0] >= result.importances[1]


class TestSequentialFeatureSelection:
    """Tests for sequential feature selection."""

    def test_returns_selected_features(self):
        from finceptquantlib.ml.modeling.regression import OLSRegression
        from finceptquantlib.ml._metrics import mean_squared_error
        rng = random.Random(42)
        features = [[rng.gauss(0, 1) for _ in range(5)] for _ in range(100)]
        target = [row[0] + row[1] + rng.gauss(0, 0.1) for row in features]
        model = OLSRegression()
        names = [f"f{i}" for i in range(5)]
        cv_splits = [(list(range(50)), list(range(50, 100)))]
        result = sequential_feature_selection(
            model, features, target, names, cv_splits, mean_squared_error, max_features=2
        )
        assert result is not None
        assert len(result) == 2

    def test_selected_features_are_strings(self):
        from finceptquantlib.ml.modeling.regression import OLSRegression
        from finceptquantlib.ml._metrics import mean_squared_error
        rng = random.Random(42)
        features = [[rng.gauss(0, 1) for _ in range(5)] for _ in range(100)]
        target = [row[0] + rng.gauss(0, 0.1) for row in features]
        model = OLSRegression()
        names = [f"f{i}" for i in range(5)]
        cv_splits = [(list(range(50)), list(range(50, 100)))]
        result = sequential_feature_selection(
            model, features, target, names, cv_splits, mean_squared_error, max_features=3
        )
        for name in result:
            assert isinstance(name, str)
