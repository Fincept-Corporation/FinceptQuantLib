"""Tests for ML-based portfolio optimization and covariance estimation."""

import random

import pytest

from finceptquantlib.ml.markets.portfolio_ml import (
    CrossSectionalFactorModel,
    LedoitWolfShrinkage,
    MinimumCovarianceDeterminant,
    NeuralPortfolioOptimizer,
    DeepPortfolioResult,
    ShrinkageCovarianceResult,
    StatisticalFactorModel,
    FactorModelResult,
    TransactionCostModel,
    TransactionCostResult,
)


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------


def _generate_returns(n_periods: int = 30, n_assets: int = 3, seed: int = 42):
    """Generate a returns matrix ~ N(0, 0.02) with the given seed."""
    rng = random.Random(seed)
    return [
        [rng.gauss(0.0, 0.02) for _ in range(n_assets)]
        for _ in range(n_periods)
    ]


RETURNS = _generate_returns()


# ---------------------------------------------------------------------------
# LedoitWolfShrinkage
# ---------------------------------------------------------------------------


class TestLedoitWolfShrinkage:
    """Tests for Ledoit-Wolf shrinkage covariance estimator."""

    def test_constructor(self):
        lw = LedoitWolfShrinkage()
        assert lw is not None

    def test_estimate_returns_result(self):
        lw = LedoitWolfShrinkage()
        result = lw.estimate(RETURNS)

        assert isinstance(result, ShrinkageCovarianceResult)
        assert result.covariance is not None
        assert result.shrinkage_intensity is not None

    def test_covariance_shape(self):
        lw = LedoitWolfShrinkage()
        result = lw.estimate(RETURNS)

        n_assets = 3
        cov = result.covariance
        assert len(cov) == n_assets
        for row in cov:
            assert len(row) == n_assets

    def test_covariance_symmetric(self):
        lw = LedoitWolfShrinkage()
        result = lw.estimate(RETURNS)
        cov = result.covariance

        for i in range(len(cov)):
            for j in range(len(cov)):
                assert abs(cov[i][j] - cov[j][i]) < 1e-10, (
                    f"Covariance not symmetric at ({i},{j})"
                )

    def test_covariance_positive_diagonal(self):
        lw = LedoitWolfShrinkage()
        result = lw.estimate(RETURNS)
        cov = result.covariance

        for i in range(len(cov)):
            assert cov[i][i] >= 0.0, f"Diagonal element {i} is negative"

    def test_shrinkage_intensity_in_range(self):
        lw = LedoitWolfShrinkage()
        result = lw.estimate(RETURNS)

        assert 0.0 <= result.shrinkage_intensity <= 1.0


# ---------------------------------------------------------------------------
# MinimumCovarianceDeterminant
# ---------------------------------------------------------------------------


class TestMinimumCovarianceDeterminant:
    """Tests for robust covariance estimation via MCD."""

    @pytest.mark.slow
    def test_constructor(self):
        mcd = MinimumCovarianceDeterminant(support_fraction=0.75)
        assert mcd is not None

    @pytest.mark.slow
    def test_estimate_returns_covariance(self):
        mcd = MinimumCovarianceDeterminant(support_fraction=0.75)
        cov = mcd.estimate(RETURNS)

        assert cov is not None
        n_assets = 3
        assert len(cov) == n_assets
        for row in cov:
            assert len(row) == n_assets

    @pytest.mark.slow
    def test_robust_to_outliers(self):
        """MCD should tolerate a few outlier observations."""
        returns_with_outliers = [row[:] for row in RETURNS]
        # Inject extreme outliers
        returns_with_outliers[0] = [0.5, -0.5, 0.5]
        returns_with_outliers[1] = [-0.5, 0.5, -0.5]

        mcd = MinimumCovarianceDeterminant(support_fraction=0.75)
        cov = mcd.estimate(returns_with_outliers)

        for i in range(len(cov)):
            assert cov[i][i] >= 0.0, "Diagonal should remain non-negative"


# ---------------------------------------------------------------------------
# StatisticalFactorModel
# ---------------------------------------------------------------------------


class TestStatisticalFactorModel:
    """Tests for PCA-style statistical factor model."""

    def test_constructor(self):
        fm = StatisticalFactorModel(n_factors=2)
        assert fm is not None

    def test_fit_returns_result(self):
        fm = StatisticalFactorModel(n_factors=2)
        result = fm.fit(RETURNS)

        assert isinstance(result, FactorModelResult)
        assert result.factor_loadings is not None
        assert result.factor_returns is not None
        assert result.idiosyncratic_risk is not None

    def test_factor_loadings_shape(self):
        n_factors = 2
        n_assets = 3
        fm = StatisticalFactorModel(n_factors=n_factors)
        result = fm.fit(RETURNS)

        loadings = result.factor_loadings
        assert len(loadings) == n_assets
        for row in loadings:
            assert len(row) == n_factors

    def test_idiosyncratic_risk_non_negative(self):
        fm = StatisticalFactorModel(n_factors=2)
        result = fm.fit(RETURNS)

        for v in result.idiosyncratic_risk:
            assert v >= 0.0, "Idiosyncratic risk must be non-negative"


# ---------------------------------------------------------------------------
# CrossSectionalFactorModel
# ---------------------------------------------------------------------------


class TestCrossSectionalFactorModel:
    """Tests for cross-sectional factor model."""

    def test_constructor(self):
        fm = CrossSectionalFactorModel()
        assert fm is not None

    def test_fit_with_factors(self):
        n_periods = 30
        n_assets = 3
        n_factors = 2

        rng = random.Random(42)
        # characteristics: n_periods x n_assets x n_factors
        characteristics = [
            [[rng.gauss(0.0, 1.0) for _ in range(n_factors)] for _ in range(n_assets)]
            for _ in range(n_periods)
        ]

        fm = CrossSectionalFactorModel()
        result = fm.fit(RETURNS, characteristics)

        assert result is not None


# ---------------------------------------------------------------------------
# NeuralPortfolioOptimizer
# ---------------------------------------------------------------------------


class TestNeuralPortfolioOptimizer:
    """Tests for deep-learning portfolio optimizer."""

    @pytest.mark.slow
    def test_constructor(self):
        opt = NeuralPortfolioOptimizer(n_assets=3, hidden_layers=[16])
        assert opt is not None

    @pytest.mark.slow
    def test_fit_and_predict_weights(self):
        opt = NeuralPortfolioOptimizer(n_assets=3, hidden_layers=[16])
        opt.fit(RETURNS)
        result = opt.predict_weights()

        assert isinstance(result, DeepPortfolioResult)
        assert result.weights is not None
        assert len(result.weights) == 3

    @pytest.mark.slow
    def test_weights_sum_to_one(self):
        opt = NeuralPortfolioOptimizer(n_assets=3, hidden_layers=[16])
        opt.fit(RETURNS)
        result = opt.predict_weights()

        weight_sum = sum(result.weights)
        assert abs(weight_sum - 1.0) < 0.1, (
            f"Weights sum to {weight_sum}, expected ~1.0"
        )


# ---------------------------------------------------------------------------
# TransactionCostModel
# ---------------------------------------------------------------------------


class TestTransactionCostModel:
    """Tests for transaction cost estimation."""

    def test_constructor(self):
        tc = TransactionCostModel()
        assert tc is not None

    def test_fit_and_predict(self):
        tc = TransactionCostModel()
        # Fit with training data: X = [[trade_size, volume, spread], ...], y = [cost, ...]
        X_train = [[1000.0, 100000.0, 0.01], [2000.0, 200000.0, 0.02], [500.0, 50000.0, 0.015],
                    [1500.0, 150000.0, 0.012], [3000.0, 300000.0, 0.025]]
        y_train = [5.0, 20.0, 3.75, 9.0, 37.5]
        tc.fit(X_train, y_train)
        pred = tc.predict([[1000.0, 100000.0, 0.01]])
        assert pred is not None

    def test_predict_returns_values(self):
        tc = TransactionCostModel()
        X_train = [[1000.0, 100000.0, 0.01], [2000.0, 200000.0, 0.02], [500.0, 50000.0, 0.015],
                    [1500.0, 150000.0, 0.012], [3000.0, 300000.0, 0.025]]
        y_train = [5.0, 20.0, 3.75, 9.0, 37.5]
        tc.fit(X_train, y_train)
        result = tc.predict(X_train)
        assert isinstance(result, TransactionCostResult)
        assert result.predicted_costs is not None
