"""Tests for finceptquantlib.ml._metrics module."""

import math

import pytest

from finceptquantlib.ml._metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r_squared,
    adjusted_r_squared,
    log_loss,
    accuracy,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


# ---------------------------------------------------------------------------
# mean_squared_error
# ---------------------------------------------------------------------------

class TestMeanSquaredError:
    def test_perfect_predictions(self):
        y = [1.0, 2.0, 3.0]
        assert mean_squared_error(y, y) == pytest.approx(0.0, abs=1e-10)

    def test_known_values(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.5, 2.5, 3.5]
        # ((0.5)^2 + (0.5)^2 + (0.5)^2) / 3 = 0.75 / 3 = 0.25
        assert mean_squared_error(y_true, y_pred) == pytest.approx(0.25, abs=1e-10)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            mean_squared_error([1.0, 2.0], [1.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mean_squared_error([], [])


# ---------------------------------------------------------------------------
# root_mean_squared_error
# ---------------------------------------------------------------------------

class TestRootMeanSquaredError:
    def test_sqrt_of_mse(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.5, 2.5, 3.5]
        mse = mean_squared_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        assert rmse == pytest.approx(math.sqrt(mse), abs=1e-10)

    def test_perfect_predictions(self):
        y = [5.0, 10.0]
        assert root_mean_squared_error(y, y) == pytest.approx(0.0, abs=1e-10)

    def test_known_values(self):
        y_true = [0.0, 0.0]
        y_pred = [3.0, 4.0]
        # MSE = (9 + 16) / 2 = 12.5, RMSE = sqrt(12.5)
        assert root_mean_squared_error(y_true, y_pred) == pytest.approx(
            math.sqrt(12.5), abs=1e-10
        )


# ---------------------------------------------------------------------------
# mean_absolute_error
# ---------------------------------------------------------------------------

class TestMeanAbsoluteError:
    def test_perfect_predictions(self):
        y = [1.0, 2.0, 3.0]
        assert mean_absolute_error(y, y) == pytest.approx(0.0, abs=1e-10)

    def test_known_values(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.5, 2.5, 3.5]
        # (0.5 + 0.5 + 0.5) / 3 = 0.5
        assert mean_absolute_error(y_true, y_pred) == pytest.approx(0.5, abs=1e-10)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            mean_absolute_error([1.0], [1.0, 2.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mean_absolute_error([], [])


# ---------------------------------------------------------------------------
# r_squared
# ---------------------------------------------------------------------------

class TestRSquared:
    def test_perfect_fit(self):
        y_true = [1.0, 2.0, 3.0, 4.0]
        assert r_squared(y_true, y_true) == pytest.approx(1.0, abs=1e-10)

    def test_mean_prediction(self):
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_mean = [2.5, 2.5, 2.5, 2.5]
        assert r_squared(y_true, y_mean) == pytest.approx(0.0, abs=1e-10)

    def test_constant_y_returns_zero(self):
        y_true = [5.0, 5.0, 5.0]
        y_pred = [5.0, 5.0, 5.0]
        # ss_tot = 0 -> returns 0.0
        assert r_squared(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)

    def test_known_value(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.1, 2.1, 2.9]
        # ss_res = 0.01 + 0.01 + 0.01 = 0.03
        # mean = 2, ss_tot = 1 + 0 + 1 = 2
        # R^2 = 1 - 0.03/2 = 0.985
        assert r_squared(y_true, y_pred) == pytest.approx(0.985, abs=1e-10)


# ---------------------------------------------------------------------------
# adjusted_r_squared
# ---------------------------------------------------------------------------

class TestAdjustedRSquared:
    def test_with_n_features(self):
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.1, 2.9, 4.1, 4.9]
        n = 5
        p = 2
        r2 = r_squared(y_true, y_pred)
        expected = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
        assert adjusted_r_squared(y_true, y_pred, n_features=p) == pytest.approx(
            expected, abs=1e-10
        )

    def test_n_leq_p_plus_1_returns_zero(self):
        # n=2, n_features=2 -> n <= n_features + 1 -> 0.0
        y_true = [1.0, 2.0]
        y_pred = [1.0, 2.0]
        assert adjusted_r_squared(y_true, y_pred, n_features=2) == pytest.approx(
            0.0, abs=1e-10
        )

    def test_perfect_fit(self):
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        # R^2 = 1 -> adj R^2 = 1 - 0 = 1
        assert adjusted_r_squared(y_true, y_true, n_features=1) == pytest.approx(
            1.0, abs=1e-10
        )


# ---------------------------------------------------------------------------
# log_loss
# ---------------------------------------------------------------------------

class TestLogLoss:
    def test_perfect_predictions_near_zero(self):
        y_true = [1.0, 0.0, 1.0]
        y_pred = [0.999, 0.001, 0.999]
        result = log_loss(y_true, y_pred)
        assert result >= 0.0
        assert result < 0.01

    def test_known_binary_case(self):
        y_true = [1.0, 0.0]
        y_pred = [0.9, 0.1]
        expected = -(math.log(0.9) + math.log(0.9)) / 2.0
        assert log_loss(y_true, y_pred) == pytest.approx(expected, abs=1e-10)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            log_loss([1.0], [0.5, 0.5])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            log_loss([], [])

    def test_clipping_prevents_log_zero(self):
        # Exact 0 and 1 should not cause errors
        result = log_loss([1.0, 0.0], [1.0, 0.0])
        assert math.isfinite(result)


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------

class TestAccuracy:
    def test_all_correct(self):
        y = [0.0, 1.0, 0.0, 1.0]
        assert accuracy(y, y) == pytest.approx(1.0, abs=1e-10)

    def test_half_correct(self):
        y_true = [0.0, 0.0, 1.0, 1.0]
        y_pred = [0.0, 1.0, 0.0, 1.0]
        assert accuracy(y_true, y_pred) == pytest.approx(0.5, abs=1e-10)

    def test_none_correct(self):
        y_true = [0.0, 0.0]
        y_pred = [1.0, 1.0]
        assert accuracy(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# confusion_matrix
# ---------------------------------------------------------------------------

class TestConfusionMatrix:
    def test_returns_correct_shape(self):
        cm = confusion_matrix([0.0, 1.0], [0.0, 1.0])
        assert len(cm) == 2
        assert len(cm[0]) == 2
        assert len(cm[1]) == 2

    def test_known_values(self):
        # 2 TN, 1 FP, 1 FN, 2 TP
        y_true = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        y_pred = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]
        cm = confusion_matrix(y_true, y_pred)
        # [[TN, FP], [FN, TP]]
        assert cm == [[2, 1], [1, 2]]

    def test_all_true_negatives(self):
        y = [0.0, 0.0, 0.0]
        cm = confusion_matrix(y, y)
        assert cm == [[3, 0], [0, 0]]

    def test_all_true_positives(self):
        y = [1.0, 1.0, 1.0]
        cm = confusion_matrix(y, y)
        assert cm == [[0, 0], [0, 3]]


# ---------------------------------------------------------------------------
# precision_score
# ---------------------------------------------------------------------------

class TestPrecisionScore:
    def test_known_values(self):
        # TP=2, FP=1 -> precision = 2/3
        y_true = [0.0, 1.0, 1.0, 0.0]
        y_pred = [0.0, 1.0, 1.0, 1.0]
        assert precision_score(y_true, y_pred) == pytest.approx(2.0 / 3.0, abs=1e-10)

    def test_perfect_precision(self):
        y_true = [1.0, 1.0, 0.0]
        y_pred = [1.0, 1.0, 0.0]
        assert precision_score(y_true, y_pred) == pytest.approx(1.0, abs=1e-10)

    def test_zero_denominator(self):
        # No positive predictions -> TP=0, FP=0
        y_true = [1.0, 1.0]
        y_pred = [0.0, 0.0]
        assert precision_score(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# recall_score
# ---------------------------------------------------------------------------

class TestRecallScore:
    def test_known_values(self):
        # TP=1, FN=1 -> recall = 1/2
        y_true = [1.0, 1.0, 0.0]
        y_pred = [1.0, 0.0, 0.0]
        assert recall_score(y_true, y_pred) == pytest.approx(0.5, abs=1e-10)

    def test_perfect_recall(self):
        y_true = [1.0, 1.0, 0.0]
        y_pred = [1.0, 1.0, 1.0]
        assert recall_score(y_true, y_pred) == pytest.approx(1.0, abs=1e-10)

    def test_zero_denominator(self):
        # No actual positives -> TP=0, FN=0
        y_true = [0.0, 0.0]
        y_pred = [1.0, 0.0]
        assert recall_score(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# f1_score
# ---------------------------------------------------------------------------

class TestF1Score:
    def test_harmonic_mean(self):
        y_true = [0.0, 1.0, 1.0, 0.0]
        y_pred = [0.0, 1.0, 1.0, 1.0]
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        expected = 2.0 * p * r / (p + r)
        assert f1_score(y_true, y_pred) == pytest.approx(expected, abs=1e-10)

    def test_perfect_f1(self):
        y = [0.0, 1.0, 1.0, 0.0]
        assert f1_score(y, y) == pytest.approx(1.0, abs=1e-10)

    def test_zero_f1(self):
        # No true positives, precision and recall both 0
        y_true = [0.0, 0.0]
        y_pred = [0.0, 0.0]
        assert f1_score(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)

    def test_known_value(self):
        # TP=2, FP=1, FN=1 -> P=2/3, R=2/3, F1=2/3
        y_true = [1.0, 1.0, 1.0, 0.0]
        y_pred = [1.0, 1.0, 0.0, 1.0]
        assert f1_score(y_true, y_pred) == pytest.approx(2.0 / 3.0, abs=1e-10)
