"""Tests for ML model validation, stability, discrimination, and calibration metrics."""

import pytest

from finceptquantlib.ml.credit_risk.validation import (
    MLBacktester,
    BacktestResult,
    # Stability
    population_stability_index,
    characteristic_stability_index,
    StabilityResult,
    # Discrimination
    lift_curve,
    gains_chart,
    # Calibration
    brier_score,
    brier_skill_score,
    hosmer_lemeshow,
    calibration_curve,
    spiegelhalter_test,
    # Interpretability
    permutation_importance,
    PermutationImportanceResult,
    partial_dependence,
    PDPResult,
)

# These are from the credit subpackage
from finceptquantlib.ml.credit_risk.credit import (
    ks_statistic,
    gini_coefficient,
    auc_roc,
    roc_curve,
)

from finceptquantlib.ml.modeling.regression import OLSRegression
from finceptquantlib.ml._metrics import mean_squared_error
from finceptquantlib.ml.credit_risk.timeseries import WalkForwardCV


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

Y_TRUE = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
Y_SCORES = [0.1, 0.2, 0.15, 0.3, 0.25, 0.7, 0.8, 0.9, 0.85, 0.6]

EXPECTED_DIST = [0.1] * 5 + [0.2] * 5
ACTUAL_DIST = [0.15] * 5 + [0.25] * 5


# ---------------------------------------------------------------------------
# MLBacktester
# ---------------------------------------------------------------------------


class TestMLBacktester:
    """Tests for walk-forward backtesting of ML models."""

    def test_constructor(self):
        model = OLSRegression()
        cv = WalkForwardCV(n_splits=3, train_size=10, test_size=5)
        bt = MLBacktester(model=model, cv=cv, metric=mean_squared_error)
        assert bt is not None

    def test_run_returns_backtest_result(self):
        model = OLSRegression()
        X = [[float(i)] for i in range(30)]
        y = [2.0 * i + 0.1 for i in range(30)]
        cv = WalkForwardCV(n_splits=3, train_size=10, test_size=5)
        bt = MLBacktester(model=model, cv=cv, metric=mean_squared_error)
        result = bt.run(X, y)
        assert isinstance(result, BacktestResult)

    def test_run_produces_metric_values(self):
        model = OLSRegression()
        X = [[float(i)] for i in range(30)]
        y = [2.0 * i + 0.1 for i in range(30)]
        cv = WalkForwardCV(n_splits=3, train_size=10, test_size=5)
        bt = MLBacktester(model=model, cv=cv, metric=mean_squared_error)
        result = bt.run(X, y)
        assert hasattr(result, "metric_values")
        assert len(result.metric_values) >= 1


# ---------------------------------------------------------------------------
# Stability metrics
# ---------------------------------------------------------------------------


class TestStabilityMetrics:
    """Tests for Population Stability Index and Characteristic Stability Index."""

    def test_psi_identical_distributions(self):
        same = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = population_stability_index(same, same)

        assert isinstance(result, StabilityResult)
        assert abs(result.index) < 1e-10, "PSI should be 0 for identical distributions"

    def test_psi_different_distributions(self):
        result = population_stability_index(EXPECTED_DIST, ACTUAL_DIST)

        assert isinstance(result, StabilityResult)
        assert result.index > 0.0, "PSI should be positive for different distributions"

    def test_csi_identical_distributions(self):
        same = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = characteristic_stability_index(same, same)

        assert isinstance(result, StabilityResult)

    def test_csi_different_distributions(self):
        result = characteristic_stability_index(EXPECTED_DIST, ACTUAL_DIST)
        assert result is not None


# ---------------------------------------------------------------------------
# Discrimination metrics
# ---------------------------------------------------------------------------


class TestDiscriminationMetrics:
    """Tests for KS, Gini, AUC-ROC, ROC curve, lift, and gains."""

    def test_ks_statistic_range(self):
        ks_val, _ = ks_statistic(Y_TRUE, Y_SCORES)
        assert 0.0 <= ks_val <= 1.0, f"KS statistic {ks_val} out of [0, 1]"

    def test_ks_statistic_good_separation(self):
        ks_val, _ = ks_statistic(Y_TRUE, Y_SCORES)
        assert ks_val > 0.3, "KS should be reasonably high for well-separated scores"

    def test_gini_coefficient(self):
        gini = gini_coefficient(Y_TRUE, Y_SCORES)
        assert isinstance(gini, float)

    def test_auc_roc_range(self):
        auc = auc_roc(Y_TRUE, Y_SCORES)
        assert 0.0 <= auc <= 1.0, f"AUC-ROC {auc} out of [0, 1]"

    def test_auc_roc_good_model(self):
        auc = auc_roc(Y_TRUE, Y_SCORES)
        assert auc > 0.8, "AUC should be high for well-separated data"

    def test_auc_roc_perfect_separation(self):
        perfect_true = [0, 0, 0, 1, 1, 1]
        perfect_scores = [0.0, 0.1, 0.2, 0.8, 0.9, 1.0]
        auc = auc_roc(perfect_true, perfect_scores)
        assert abs(auc - 1.0) < 1e-10, "Perfect separation should yield AUC = 1.0"

    def test_roc_curve_returns_two_lists(self):
        fpr, tpr = roc_curve(Y_TRUE, Y_SCORES)
        assert len(fpr) == len(tpr)
        assert fpr[0] <= fpr[-1]
        assert tpr[0] <= tpr[-1]

    def test_lift_curve_shape(self):
        n_bins = 5
        percentages, lifts = lift_curve(Y_TRUE, Y_SCORES, n_bins)
        assert len(percentages) == n_bins
        assert len(lifts) == n_bins

    def test_gains_chart_shape(self):
        n_bins = 5
        percentages, cumulative_gains = gains_chart(Y_TRUE, Y_SCORES, n_bins)
        assert len(percentages) >= n_bins
        assert len(cumulative_gains) >= n_bins


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


class TestCalibrationMetrics:
    """Tests for Brier score, Hosmer-Lemeshow, and calibration curves."""

    def test_brier_score_range(self):
        bs = brier_score(Y_TRUE, Y_SCORES)
        assert 0.0 <= bs <= 1.0, f"Brier score {bs} out of [0, 1]"

    def test_brier_score_perfect(self):
        perfect_prob = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        bs = brier_score(Y_TRUE, perfect_prob)
        assert abs(bs) < 1e-10, "Perfect predictions should have Brier score = 0"

    def test_brier_skill_score(self):
        bss = brier_skill_score(Y_TRUE, Y_SCORES)
        assert isinstance(bss, float)

    def test_hosmer_lemeshow_returns_statistic_and_pvalue(self):
        statistic, p_value = hosmer_lemeshow(Y_TRUE, Y_SCORES, n_groups=5)

        assert isinstance(statistic, float)
        assert isinstance(p_value, float)
        assert statistic >= 0.0
        assert 0.0 <= p_value <= 1.0

    def test_calibration_curve_shape(self):
        n_bins = 5
        bin_means, bin_fractions = calibration_curve(Y_TRUE, Y_SCORES, n_bins)

        assert len(bin_means) <= n_bins
        assert len(bin_fractions) <= n_bins
        assert len(bin_means) == len(bin_fractions)

    def test_spiegelhalter_test(self):
        result = spiegelhalter_test(Y_TRUE, Y_SCORES)
        assert result is not None


# ---------------------------------------------------------------------------
# Interpretability
# ---------------------------------------------------------------------------


class TestInterpretability:
    """Tests for permutation importance and partial dependence."""

    @pytest.fixture
    def fitted_model_and_data(self):
        """Prepare a simple OLS model with data for interpretability tests."""
        import random as _rng
        _rng.seed(42)
        X = [[float(i), _rng.gauss(0, 1.0)] for i in range(30)]
        y = [2.0 * x[0] + 1.0 * x[1] + 0.1 for x in X]
        model = OLSRegression()
        model.fit(X, y)
        return model, X, y

    def test_permutation_importance_returns_result(self, fitted_model_and_data):
        model, X, y = fitted_model_and_data
        result = permutation_importance(model, X, y, metric=mean_squared_error, n_repeats=5)

        assert isinstance(result, PermutationImportanceResult)
        assert result.importances_mean is not None

    def test_permutation_importance_shape(self, fitted_model_and_data):
        model, X, y = fitted_model_and_data
        n_features = len(X[0])
        result = permutation_importance(model, X, y, metric=mean_squared_error, n_repeats=5)

        assert len(result.importances_mean) == n_features

    def test_partial_dependence_returns_result(self, fitted_model_and_data):
        model, X, _ = fitted_model_and_data
        result = partial_dependence(model, X, feature_index=0, grid_resolution=10)

        assert isinstance(result, PDPResult)

    def test_partial_dependence_grid_size(self, fitted_model_and_data):
        model, X, _ = fitted_model_and_data
        grid_resolution = 10
        result = partial_dependence(model, X, feature_index=0, grid_resolution=grid_resolution)

        assert len(result.feature_values) == grid_resolution
        assert len(result.pdp_values) == grid_resolution
