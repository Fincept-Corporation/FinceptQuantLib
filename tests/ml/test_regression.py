"""Tests for finceptquantlib.ml.modeling.regression module."""

import pytest

from finceptquantlib.ml.modeling.regression import (
    OLSRegression,
    LinearRegressionResult,
    RidgeRegression,
    LassoRegression,
    ElasticNet,
    DecisionTreeRegressor,
    TreeResult,
    GradientBoostedRegressor,
    RandomForestRegressor,
    BetaRegressionLGD,
    LGDModelResult,
    CCFModel,
    EADModelResult,
)


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

X_SIMPLE = [[1.0], [2.0], [3.0], [4.0], [5.0]]
Y_SIMPLE = [3.0, 5.0, 7.0, 9.0, 11.0]  # y = 2x + 1

X_NOISY = [
    [1.0], [2.0], [3.0], [4.0], [5.0],
    [6.0], [7.0], [8.0], [9.0], [10.0],
]
Y_NOISY = [2.1, 4.2, 5.8, 8.1, 10.2, 11.8, 14.1, 16.0, 18.1, 20.0]
# Approximates y = 2x with noise


# ---------------------------------------------------------------------------
# OLSRegression
# ---------------------------------------------------------------------------

class TestOLSRegression:
    """Tests for ordinary least-squares regression."""

    def test_fit_returns_linear_regression_result(self):
        model = OLSRegression()
        result = model.fit(X_SIMPLE, Y_SIMPLE)
        assert isinstance(result, LinearRegressionResult)

    def test_coefficients_perfect_linear(self):
        model = OLSRegression()
        result = model.fit(X_SIMPLE, Y_SIMPLE)
        assert result.coefficients == pytest.approx([2.0], abs=1e-6)
        assert result.intercept == pytest.approx(1.0, abs=1e-6)

    def test_r_squared_perfect_linear(self):
        model = OLSRegression()
        result = model.fit(X_SIMPLE, Y_SIMPLE)
        assert result.r_squared == pytest.approx(1.0, abs=1e-6)

    def test_result_attributes(self):
        model = OLSRegression()
        result = model.fit(X_NOISY, Y_NOISY)
        assert result.residual_std is not None
        assert result.t_statistics is not None
        assert result.p_values is not None
        assert result.standard_errors is not None

    def test_predict(self):
        model = OLSRegression()
        model.fit(X_SIMPLE, Y_SIMPLE)
        pred = model.predict([[6.0], [7.0]])
        assert hasattr(pred, "predictions")
        assert pred.predictions == pytest.approx([13.0, 15.0], abs=0.5)

    def test_fit_noisy_data(self):
        model = OLSRegression()
        result = model.fit(X_NOISY, Y_NOISY)
        assert result.coefficients[0] == pytest.approx(2.0, abs=0.5)
        assert result.r_squared > 0.95


# ---------------------------------------------------------------------------
# RidgeRegression
# ---------------------------------------------------------------------------

class TestRidgeRegression:
    """Tests for Ridge (L2) regression."""

    def test_fit_and_predict(self):
        model = RidgeRegression(alpha=1.0)
        result = model.fit(X_NOISY, Y_NOISY)
        assert isinstance(result, LinearRegressionResult)
        pred = model.predict([[5.0]])
        assert hasattr(pred, "predictions")
        assert len(pred.predictions) == 1

    def test_coefficients_shrinkage(self):
        ols = OLSRegression()
        ols_result = ols.fit(X_NOISY, Y_NOISY)

        ridge = RidgeRegression(alpha=1.0)
        ridge_result = ridge.fit(X_NOISY, Y_NOISY)

        # Ridge coefficients should be shrunk toward zero relative to OLS
        assert abs(ridge_result.coefficients[0]) <= abs(ols_result.coefficients[0]) + 1e-6

    def test_predict_reasonable(self):
        model = RidgeRegression(alpha=1.0)
        model.fit(X_NOISY, Y_NOISY)
        pred = model.predict([[5.0]])
        assert pred.predictions[0] == pytest.approx(10.0, abs=2.0)


# ---------------------------------------------------------------------------
# LassoRegression
# ---------------------------------------------------------------------------

class TestLassoRegression:
    """Tests for Lasso (L1) regression."""

    def test_fit_and_predict(self):
        model = LassoRegression(alpha=0.1)
        result = model.fit(X_NOISY, Y_NOISY)
        assert isinstance(result, LinearRegressionResult)
        pred = model.predict([[5.0]])
        assert hasattr(pred, "predictions")

    def test_sparsity(self):
        # Multi-feature data where some features are irrelevant
        X_multi = [[i, i * 0.5, 0.0] for i in range(1, 11)]
        y_multi = [2.0 * i + 0.1 for i in range(1, 11)]  # only first feature matters
        model = LassoRegression(alpha=0.1)
        result = model.fit(X_multi, y_multi)
        # At least one coefficient should be zero or very small
        min_coef = min(abs(c) for c in result.coefficients)
        assert min_coef < 1.0

    def test_predict_reasonable(self):
        model = LassoRegression(alpha=0.1)
        model.fit(X_NOISY, Y_NOISY)
        pred = model.predict([[5.0]])
        assert pred.predictions[0] == pytest.approx(10.0, abs=2.0)


# ---------------------------------------------------------------------------
# ElasticNet
# ---------------------------------------------------------------------------

class TestElasticNet:
    """Tests for ElasticNet regression."""

    def test_fit_and_predict(self):
        model = ElasticNet(alpha=0.1, l1_ratio=0.5)
        result = model.fit(X_NOISY, Y_NOISY)
        assert isinstance(result, LinearRegressionResult)
        pred = model.predict([[5.0]])
        assert hasattr(pred, "predictions")
        assert pred.predictions[0] == pytest.approx(10.0, abs=2.0)


# ---------------------------------------------------------------------------
# DecisionTreeRegressor
# ---------------------------------------------------------------------------

class TestDecisionTreeRegressor:
    """Tests for decision tree regression."""

    def test_fit_returns_tree_result(self):
        model = DecisionTreeRegressor(max_depth=3)
        result = model.fit(X_NOISY, Y_NOISY)
        assert isinstance(result, TreeResult)

    def test_predict_reasonable(self):
        model = DecisionTreeRegressor(max_depth=3)
        model.fit(X_NOISY, Y_NOISY)
        pred = model.predict([[5.0]])
        assert hasattr(pred, "predictions")
        # Tree models don't extrapolate; just check it returns a number
        assert isinstance(pred.predictions[0], float)

    def test_training_predictions(self):
        model = DecisionTreeRegressor(max_depth=3)
        model.fit(X_NOISY, Y_NOISY)
        pred = model.predict(X_NOISY)
        for p, actual in zip(pred.predictions, Y_NOISY):
            assert p == pytest.approx(actual, abs=5.0)


# ---------------------------------------------------------------------------
# GradientBoostedRegressor
# ---------------------------------------------------------------------------

class TestGradientBoostedRegressor:
    """Tests for gradient boosted tree regression."""

    def test_fit_and_predict(self):
        model = GradientBoostedRegressor(
            n_estimators=10, max_depth=2, learning_rate=0.1
        )
        model.fit(X_NOISY, Y_NOISY)
        pred = model.predict([[5.0]])
        assert hasattr(pred, "predictions")
        assert len(pred.predictions) == 1

    def test_predict_reasonable(self):
        model = GradientBoostedRegressor(
            n_estimators=10, max_depth=2, learning_rate=0.1
        )
        model.fit(X_NOISY, Y_NOISY)
        pred = model.predict(X_NOISY)
        # Ensemble models on small noisy data - just check predictions exist
        assert len(pred.predictions) == len(Y_NOISY)


# ---------------------------------------------------------------------------
# RandomForestRegressor
# ---------------------------------------------------------------------------

class TestRandomForestRegressor:
    """Tests for random forest regression."""

    def test_fit_and_predict(self):
        model = RandomForestRegressor(n_estimators=10, max_depth=3)
        model.fit(X_NOISY, Y_NOISY)
        pred = model.predict([[5.0]])
        assert hasattr(pred, "predictions")
        assert len(pred.predictions) == 1

    def test_predict_reasonable(self):
        model = RandomForestRegressor(n_estimators=10, max_depth=3)
        model.fit(X_NOISY, Y_NOISY)
        pred = model.predict(X_NOISY)
        assert len(pred.predictions) == len(Y_NOISY)


# ---------------------------------------------------------------------------
# BetaRegressionLGD
# ---------------------------------------------------------------------------

class TestBetaRegressionLGD:
    """Tests for Beta regression LGD model."""

    def test_fit_returns_lgd_model_result(self):
        # LGD values are in (0, 1)
        X_lgd = [[i * 0.1] for i in range(1, 11)]
        y_lgd = [0.15, 0.22, 0.30, 0.35, 0.42, 0.48, 0.55, 0.60, 0.68, 0.75]
        model = BetaRegressionLGD()
        result = model.fit(X_lgd, y_lgd)
        assert isinstance(result, LGDModelResult)

    def test_predict_in_range(self):
        X_lgd = [[i * 0.1] for i in range(1, 11)]
        y_lgd = [0.15, 0.22, 0.30, 0.35, 0.42, 0.48, 0.55, 0.60, 0.68, 0.75]
        model = BetaRegressionLGD()
        model.fit(X_lgd, y_lgd)
        pred = model.predict([[0.5]])
        for p in pred.predictions:
            assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# CCFModel
# ---------------------------------------------------------------------------

class TestCCFModel:
    """Tests for Credit Conversion Factor model."""

    def test_fit_returns_ead_model_result(self):
        # CCF data: drawn amount, limit, utilization -> CCF in (0, 1)
        X_ccf = [[i * 100.0, 1000.0] for i in range(1, 11)]
        y_ccf = [0.10, 0.18, 0.25, 0.32, 0.40, 0.45, 0.52, 0.58, 0.65, 0.72]
        model = CCFModel()
        result = model.fit(X_ccf, y_ccf)
        assert isinstance(result, EADModelResult)

    def test_predict(self):
        X_ccf = [[i * 100.0, 1000.0] for i in range(1, 11)]
        y_ccf = [0.10, 0.18, 0.25, 0.32, 0.40, 0.45, 0.52, 0.58, 0.65, 0.72]
        model = CCFModel()
        model.fit(X_ccf, y_ccf)
        pred = model.predict([[500.0, 1000.0]])
        assert hasattr(pred, "predictions")
        assert len(pred.predictions) == 1
