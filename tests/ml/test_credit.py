"""Tests for finceptquantlib.ml.credit_risk.credit module."""

import pytest

from finceptquantlib.ml.credit_risk.credit import (
    LogisticRegression,
    LogisticRegressionResult,
    WoEBinner,
    BinResult,
    Scorecard,
    ScorecardResult,
    PlattScaling,
    CalibrationResult,
    IsotonicRegression,
    RatingMigration,
    MigrationMatrixResult,
    ks_statistic,
    gini_coefficient,
    auc_roc,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# Linearly separable: first 10 are class 0, last 10 are class 1
X_SEPARABLE = [[i, i * 0.5] for i in range(20)]
Y_SEPARABLE = [0] * 10 + [1] * 10

# Discrimination metric test data
Y_TRUE = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
Y_SCORES = [0.1, 0.2, 0.15, 0.3, 0.25, 0.7, 0.8, 0.9, 0.85, 0.6]


# ---------------------------------------------------------------------------
# LogisticRegression
# ---------------------------------------------------------------------------

class TestLogisticRegression:
    """Tests for logistic regression classifier."""

    def test_fit_returns_result(self):
        model = LogisticRegression()
        result = model.fit(X_SEPARABLE, Y_SEPARABLE)
        assert isinstance(result, LogisticRegressionResult)

    def test_result_attributes(self):
        model = LogisticRegression()
        result = model.fit(X_SEPARABLE, Y_SEPARABLE)
        assert result.coefficients is not None
        assert result.intercept is not None
        assert result.log_likelihood is not None
        assert result.aic is not None
        assert result.bic is not None

    def test_predict_binary(self):
        model = LogisticRegression()
        model.fit(X_SEPARABLE, Y_SEPARABLE)
        pred = model.predict(X_SEPARABLE)
        assert hasattr(pred, "predictions")
        assert len(pred.predictions) == len(Y_SEPARABLE)
        # All predictions should be 0 or 1
        for p in pred.predictions:
            assert p in (0, 1)

    def test_predict_correct_labels(self):
        model = LogisticRegression()
        model.fit(X_SEPARABLE, Y_SEPARABLE)
        pred = model.predict(X_SEPARABLE)
        # Most predictions should match the true labels for separable data
        correct = sum(1 for p, t in zip(pred.predictions, Y_SEPARABLE) if p == t)
        accuracy = correct / len(Y_SEPARABLE)
        assert accuracy > 0.8

    def test_predict_proba(self):
        model = LogisticRegression()
        model.fit(X_SEPARABLE, Y_SEPARABLE)
        probs = model.predict_proba(X_SEPARABLE)
        assert len(probs) == len(Y_SEPARABLE)
        for row in probs:
            assert len(row) == 2
            assert 0.0 <= row[0] <= 1.0
            assert 0.0 <= row[1] <= 1.0

    def test_predict_proba_ordering(self):
        model = LogisticRegression()
        model.fit(X_SEPARABLE, Y_SEPARABLE)
        probs = model.predict_proba(X_SEPARABLE)
        # Class-0 examples should generally have lower P(class=1) than class-1 examples
        avg_prob_class0 = sum(probs[i][1] for i in range(10)) / 10
        avg_prob_class1 = sum(probs[i][1] for i in range(10, 20)) / 10
        assert avg_prob_class0 < avg_prob_class1


# ---------------------------------------------------------------------------
# WoEBinner
# ---------------------------------------------------------------------------

class TestWoEBinner:
    """Tests for Weight of Evidence binning."""

    def test_fit_and_transform(self):
        feature_values = list(range(20))
        target = Y_SEPARABLE
        binner = WoEBinner(n_bins=5)
        result = binner.fit(feature_values, target)
        assert isinstance(result, BinResult)
        woe_values = binner.transform(feature_values)
        assert len(woe_values) == len(feature_values)

    def test_information_value(self):
        feature_values = list(range(20))
        target = Y_SEPARABLE
        binner = WoEBinner(n_bins=5)
        result = binner.fit(feature_values, target)
        assert result.total_iv is not None
        assert result.total_iv >= 0.0

    def test_woe_values_are_numeric(self):
        feature_values = list(range(20))
        target = Y_SEPARABLE
        binner = WoEBinner(n_bins=5)
        binner.fit(feature_values, target)
        woe_values = binner.transform(feature_values)
        for v in woe_values:
            assert isinstance(v, (int, float))


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

class TestScorecard:
    """Tests for points-based scorecard."""

    def test_construction(self):
        sc = Scorecard(pdo=20, base_score=600, base_odds=50)
        assert sc is not None

    def test_build_from_logistic(self):
        model = LogisticRegression()
        model.fit(X_SEPARABLE, Y_SEPARABLE)
        feature_names = ["feat0", "feat1"]
        # Fit WoE binners per feature (1D feature lists)
        woe_binners = {}
        for idx, name in enumerate(feature_names):
            binner = WoEBinner(n_bins=5)
            feat_col = [row[idx] for row in X_SEPARABLE]
            binner.fit(feat_col, Y_SEPARABLE)
            woe_binners[name] = binner
        sc = Scorecard(pdo=20, base_score=600, base_odds=50)
        result = sc.build(model, woe_binners, feature_names)
        assert isinstance(result, ScorecardResult)

    def test_score_produces_values(self):
        model = LogisticRegression()
        model.fit(X_SEPARABLE, Y_SEPARABLE)
        feature_names = ["feat0", "feat1"]
        woe_binners = {}
        for idx, name in enumerate(feature_names):
            binner = WoEBinner(n_bins=5)
            feat_col = [row[idx] for row in X_SEPARABLE]
            binner.fit(feat_col, Y_SEPARABLE)
            woe_binners[name] = binner
        sc = Scorecard(pdo=20, base_score=600, base_odds=50)
        sc.build(model, woe_binners, feature_names)
        # score() takes a dict per applicant
        applicant = {"feat0": 5.0, "feat1": 2.5}
        s = sc.score(applicant)
        assert isinstance(s, (int, float))


# ---------------------------------------------------------------------------
# PlattScaling
# ---------------------------------------------------------------------------

class TestPlattScaling:
    """Tests for Platt probability calibration."""

    def test_fit_returns_calibration_result(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        calibrator = PlattScaling()
        result = calibrator.fit(scores, labels)
        assert isinstance(result, CalibrationResult)

    def test_predict_in_range(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        calibrator = PlattScaling()
        calibrator.fit(scores, labels)
        calibrated = calibrator.calibrate(scores)
        for p in calibrated:
            assert 0.0 <= p <= 1.0

    def test_calibrated_produces_values(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        calibrator = PlattScaling()
        calibrator.fit(scores, labels)
        calibrated = calibrator.calibrate(scores)
        assert len(calibrated) == len(scores)
        for p in calibrated:
            assert isinstance(p, float)


# ---------------------------------------------------------------------------
# IsotonicRegression
# ---------------------------------------------------------------------------

class TestIsotonicRegression:
    """Tests for isotonic regression calibration."""

    def test_fit_and_calibrate(self):
        x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        model = IsotonicRegression()
        model.fit(x, y)
        preds = model.calibrate(x)
        assert len(preds) == len(x)

    def test_monotonically_increasing(self):
        x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        model = IsotonicRegression()
        model.fit(x, y)
        preds = model.calibrate(x)
        # Predictions should be monotonically non-decreasing
        for i in range(1, len(preds)):
            assert preds[i] >= preds[i - 1] - 1e-10


# ---------------------------------------------------------------------------
# RatingMigration
# ---------------------------------------------------------------------------

class TestRatingMigration:
    """Tests for rating migration matrix estimation."""

    def test_estimate_returns_migration_matrix_result(self):
        ratings = ["AAA", "AA", "A", "BBB", "BB"]
        start = ["AAA", "AAA", "AA", "AA", "A", "A", "BBB", "BBB", "AAA", "AA", "A", "BBB"]
        end =   ["AAA", "AA",  "AA", "A",  "A", "BBB","BBB","BB",  "AAA", "AA", "A", "BBB"]
        model = RatingMigration(ratings=ratings)
        result = model.estimate(start, end)
        assert isinstance(result, MigrationMatrixResult)

    def test_transition_matrix_exists(self):
        ratings = ["AAA", "AA", "A", "BBB", "BB"]
        start = ["AAA", "AAA", "AA", "AA", "A", "A", "BBB", "BBB"]
        end =   ["AAA", "AA",  "AA", "A",  "A", "BBB","BBB","BB"]
        model = RatingMigration(ratings=ratings)
        result = model.estimate(start, end)
        assert result.matrix is not None

    def test_transition_matrix_rows_sum_to_one(self):
        ratings = ["AAA", "AA", "A", "BBB", "BB"]
        start = ["AAA", "AAA", "AA", "AA", "A", "A", "BBB", "BBB", "AAA", "AA", "A", "BBB"]
        end =   ["AAA", "AA",  "AA", "A",  "A", "BBB","BBB","BB",  "AAA", "AA", "A", "BBB"]
        model = RatingMigration(ratings=ratings)
        result = model.estimate(start, end)
        for row in result.matrix:
            row_sum = sum(row)
            assert row_sum == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Discrimination metrics
# ---------------------------------------------------------------------------

class TestKSStatistic:
    """Tests for Kolmogorov-Smirnov statistic."""

    def test_returns_tuple(self):
        result = ks_statistic(Y_TRUE, Y_SCORES)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)

    def test_in_range(self):
        ks_val, _ = ks_statistic(Y_TRUE, Y_SCORES)
        assert 0.0 <= ks_val <= 1.0

    def test_good_separation(self):
        ks_val, _ = ks_statistic(Y_TRUE, Y_SCORES)
        assert ks_val > 0.5


class TestGiniCoefficient:
    """Tests for Gini coefficient."""

    def test_returns_float(self):
        result = gini_coefficient(Y_TRUE, Y_SCORES)
        assert isinstance(result, float)

    def test_good_model(self):
        result = gini_coefficient(Y_TRUE, Y_SCORES)
        assert result > 0.5


class TestAUCROC:
    """Tests for area under ROC curve."""

    def test_returns_float(self):
        result = auc_roc(Y_TRUE, Y_SCORES)
        assert isinstance(result, float)

    def test_in_range(self):
        result = auc_roc(Y_TRUE, Y_SCORES)
        assert 0.0 <= result <= 1.0

    def test_good_model_high_auc(self):
        result = auc_roc(Y_TRUE, Y_SCORES)
        assert result > 0.8

    def test_random_model_near_half(self):
        # Random scores should give AUC close to 0.5
        import random
        random.seed(42)
        y_true_rand = [0] * 50 + [1] * 50
        y_scores_rand = [random.random() for _ in range(100)]
        result = auc_roc(y_true_rand, y_scores_rand)
        assert result == pytest.approx(0.5, abs=0.15)

    def test_perfect_separation(self):
        y_true_perf = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        y_scores_perf = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
        result = auc_roc(y_true_perf, y_scores_perf)
        assert result == pytest.approx(1.0, abs=0.01)


class TestROCCurve:
    """Tests for ROC curve computation."""

    def test_returns_tuple(self):
        result = roc_curve(Y_TRUE, Y_SCORES)
        assert len(result) >= 2
        fpr, tpr = result[0], result[1]
        assert len(fpr) == len(tpr)

    def test_starts_at_origin(self):
        fpr, tpr = roc_curve(Y_TRUE, Y_SCORES)[:2]
        assert fpr[0] == pytest.approx(0.0, abs=1e-6)
        assert tpr[0] == pytest.approx(0.0, abs=1e-6)

    def test_ends_at_one(self):
        fpr, tpr = roc_curve(Y_TRUE, Y_SCORES)[:2]
        assert fpr[-1] == pytest.approx(1.0, abs=1e-6)
        assert tpr[-1] == pytest.approx(1.0, abs=1e-6)
