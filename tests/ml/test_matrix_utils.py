"""Tests for finceptquantlib.ml._matrix_utils module."""

import math

import pytest

from finceptquantlib.ml._matrix_utils import (
    column_mean,
    column_std,
    column_min,
    column_max,
    column_median,
    column_percentile,
    center_matrix,
    standardize_matrix,
    add_intercept,
    covariance_matrix,
    correlation_matrix,
    dot_product,
    mat_vec_mul,
    mat_mul,
    transpose,
    euclidean_distance,
    distance_matrix,
)


# ---------------------------------------------------------------------------
# column_mean
# ---------------------------------------------------------------------------

class TestColumnMean:
    def test_known_values(self):
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        result = column_mean(X)
        assert result[0] == pytest.approx(3.0, abs=1e-10)
        assert result[1] == pytest.approx(4.0, abs=1e-10)

    def test_single_row(self):
        X = [[7.0, 8.0, 9.0]]
        result = column_mean(X)
        assert result == [pytest.approx(7.0, abs=1e-10),
                          pytest.approx(8.0, abs=1e-10),
                          pytest.approx(9.0, abs=1e-10)]

    def test_empty_input(self):
        assert column_mean([]) == []


# ---------------------------------------------------------------------------
# column_std
# ---------------------------------------------------------------------------

class TestColumnStd:
    def test_population_std_ddof0(self):
        X = [[1.0], [3.0]]
        # mean=2, var = ((1)^2 + (1)^2)/2 = 1, std = 1
        result = column_std(X, ddof=0)
        assert result[0] == pytest.approx(1.0, abs=1e-10)

    def test_sample_std_ddof1(self):
        X = [[1.0], [3.0]]
        # mean=2, var = ((1)^2 + (1)^2)/1 = 2, std = sqrt(2)
        result = column_std(X, ddof=1)
        assert result[0] == pytest.approx(math.sqrt(2.0), abs=1e-10)

    def test_constant_column(self):
        X = [[5.0], [5.0], [5.0]]
        result = column_std(X, ddof=0)
        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_multiple_columns(self):
        X = [[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]]
        result = column_std(X, ddof=0)
        # col 0: mean=3, var = (4+0+4)/3, std = sqrt(8/3)
        assert result[0] == pytest.approx(math.sqrt(8.0 / 3.0), abs=1e-10)


# ---------------------------------------------------------------------------
# column_min / column_max
# ---------------------------------------------------------------------------

class TestColumnMin:
    def test_known_values(self):
        X = [[3.0, 1.0], [1.0, 5.0], [2.0, 3.0]]
        result = column_min(X)
        assert result == [pytest.approx(1.0, abs=1e-10),
                          pytest.approx(1.0, abs=1e-10)]

    def test_empty(self):
        assert column_min([]) == []


class TestColumnMax:
    def test_known_values(self):
        X = [[3.0, 1.0], [1.0, 5.0], [2.0, 3.0]]
        result = column_max(X)
        assert result == [pytest.approx(3.0, abs=1e-10),
                          pytest.approx(5.0, abs=1e-10)]

    def test_empty(self):
        assert column_max([]) == []


# ---------------------------------------------------------------------------
# column_median
# ---------------------------------------------------------------------------

class TestColumnMedian:
    def test_odd_rows(self):
        X = [[1.0], [3.0], [2.0]]
        result = column_median(X)
        assert result[0] == pytest.approx(2.0, abs=1e-10)

    def test_even_rows(self):
        X = [[1.0], [4.0], [2.0], [3.0]]
        result = column_median(X)
        assert result[0] == pytest.approx(2.5, abs=1e-10)

    def test_empty(self):
        assert column_median([]) == []


# ---------------------------------------------------------------------------
# column_percentile
# ---------------------------------------------------------------------------

class TestColumnPercentile:
    def test_q0_gives_min(self):
        X = [[3.0], [1.0], [2.0]]
        result = column_percentile(X, q=0.0)
        assert result[0] == pytest.approx(1.0, abs=1e-10)

    def test_q1_gives_max(self):
        X = [[3.0], [1.0], [2.0]]
        result = column_percentile(X, q=1.0)
        assert result[0] == pytest.approx(3.0, abs=1e-10)

    def test_q05_gives_median(self):
        X = [[1.0], [2.0], [3.0]]
        result = column_percentile(X, q=0.5)
        assert result[0] == pytest.approx(2.0, abs=1e-10)

    def test_empty(self):
        assert column_percentile([], q=0.5) == []

    def test_interpolation(self):
        X = [[10.0], [20.0], [30.0], [40.0]]
        result = column_percentile(X, q=0.25)
        # pos = 0.25 * 3 = 0.75, lo=0, hi=1, frac=0.75
        # 10*(0.25) + 20*(0.75) = 2.5 + 15 = 17.5
        assert result[0] == pytest.approx(17.5, abs=1e-10)


# ---------------------------------------------------------------------------
# center_matrix
# ---------------------------------------------------------------------------

class TestCenterMatrix:
    def test_zero_column_means(self):
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        centered = center_matrix(X)
        means = column_mean(centered)
        for m in means:
            assert m == pytest.approx(0.0, abs=1e-10)

    def test_known_values(self):
        X = [[1.0], [3.0], [5.0]]
        # mean = 3
        centered = center_matrix(X)
        assert centered[0][0] == pytest.approx(-2.0, abs=1e-10)
        assert centered[1][0] == pytest.approx(0.0, abs=1e-10)
        assert centered[2][0] == pytest.approx(2.0, abs=1e-10)


# ---------------------------------------------------------------------------
# standardize_matrix
# ---------------------------------------------------------------------------

class TestStandardizeMatrix:
    def test_zero_mean_unit_std(self):
        X = [[1.0, 10.0], [3.0, 20.0], [5.0, 30.0], [7.0, 40.0]]
        std_X, means, stds = standardize_matrix(X, ddof=0)
        col_means = column_mean(std_X)
        col_stds = column_std(std_X, ddof=0)
        for m in col_means:
            assert m == pytest.approx(0.0, abs=1e-10)
        for s in col_stds:
            assert s == pytest.approx(1.0, abs=1e-10)

    def test_returns_means_and_stds(self):
        X = [[1.0], [3.0]]
        _, means, stds = standardize_matrix(X, ddof=0)
        assert means[0] == pytest.approx(2.0, abs=1e-10)
        assert stds[0] == pytest.approx(1.0, abs=1e-10)

    def test_constant_column_gives_zeros(self):
        X = [[5.0], [5.0], [5.0]]
        std_X, _, _ = standardize_matrix(X, ddof=0)
        for row in std_X:
            assert row[0] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# add_intercept
# ---------------------------------------------------------------------------

class TestAddIntercept:
    def test_prepends_ones(self):
        X = [[2.0, 3.0], [4.0, 5.0]]
        result = add_intercept(X)
        assert len(result) == 2
        assert len(result[0]) == 3
        assert result[0][0] == pytest.approx(1.0, abs=1e-10)
        assert result[0][1] == pytest.approx(2.0, abs=1e-10)
        assert result[0][2] == pytest.approx(3.0, abs=1e-10)
        assert result[1][0] == pytest.approx(1.0, abs=1e-10)

    def test_empty_rows(self):
        result = add_intercept([])
        assert result == []


# ---------------------------------------------------------------------------
# covariance_matrix
# ---------------------------------------------------------------------------

class TestCovarianceMatrix:
    def test_identity_like(self):
        # Two independent columns with unit variance
        X = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        cov = covariance_matrix(X, ddof=1)
        # diagonal should be positive, off-diagonal near zero
        assert cov[0][0] > 0
        assert cov[1][1] > 0
        assert abs(cov[0][1]) < abs(cov[0][0])

    def test_known_values(self):
        X = [[1.0, 2.0], [3.0, 4.0]]
        cov = covariance_matrix(X, ddof=1)
        # mean = [2, 3]
        # cov[0][0] = (1-2)^2 + (3-2)^2 = 2 / (2-1) = 2
        # cov[1][1] = (2-3)^2 + (4-3)^2 = 2 / 1 = 2
        # cov[0][1] = (1-2)(2-3) + (3-2)(4-3) = 1 + 1 = 2 / 1 = 2
        assert cov[0][0] == pytest.approx(2.0, abs=1e-10)
        assert cov[1][1] == pytest.approx(2.0, abs=1e-10)
        assert cov[0][1] == pytest.approx(2.0, abs=1e-10)
        assert cov[1][0] == pytest.approx(2.0, abs=1e-10)

    def test_symmetric(self):
        X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]
        cov = covariance_matrix(X)
        m = len(cov)
        for i in range(m):
            for j in range(m):
                assert cov[i][j] == pytest.approx(cov[j][i], abs=1e-10)


# ---------------------------------------------------------------------------
# correlation_matrix
# ---------------------------------------------------------------------------

class TestCorrelationMatrix:
    def test_self_correlation_is_one(self):
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        corr = correlation_matrix(X)
        assert corr[0][0] == pytest.approx(1.0, abs=1e-10)
        assert corr[1][1] == pytest.approx(1.0, abs=1e-10)

    def test_range_minus_one_to_one(self):
        X = [[1.0, 10.0], [2.0, 5.0], [3.0, 1.0], [4.0, 8.0]]
        corr = correlation_matrix(X)
        m = len(corr)
        for i in range(m):
            for j in range(m):
                assert -1.0 - 1e-10 <= corr[i][j] <= 1.0 + 1e-10

    def test_perfect_positive_correlation(self):
        X = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]
        corr = correlation_matrix(X)
        assert corr[0][1] == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# dot_product
# ---------------------------------------------------------------------------

class TestDotProduct:
    def test_known_values(self):
        assert dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(
            32.0, abs=1e-10
        )

    def test_orthogonal_vectors(self):
        assert dot_product([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-10)

    def test_zero_vector(self):
        assert dot_product([0.0, 0.0], [5.0, 10.0]) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# mat_vec_mul
# ---------------------------------------------------------------------------

class TestMatVecMul:
    def test_identity_matrix(self):
        I = [[1.0, 0.0], [0.0, 1.0]]
        v = [3.0, 7.0]
        result = mat_vec_mul(I, v)
        assert result[0] == pytest.approx(3.0, abs=1e-10)
        assert result[1] == pytest.approx(7.0, abs=1e-10)

    def test_known_values(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        x = [5.0, 6.0]
        result = mat_vec_mul(A, x)
        assert result[0] == pytest.approx(17.0, abs=1e-10)
        assert result[1] == pytest.approx(39.0, abs=1e-10)


# ---------------------------------------------------------------------------
# mat_mul
# ---------------------------------------------------------------------------

class TestMatMul:
    def test_identity_multiplication(self):
        I = [[1.0, 0.0], [0.0, 1.0]]
        A = [[2.0, 3.0], [4.0, 5.0]]
        result = mat_mul(I, A)
        for i in range(2):
            for j in range(2):
                assert result[i][j] == pytest.approx(A[i][j], abs=1e-10)

    def test_known_values(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[5.0, 6.0], [7.0, 8.0]]
        result = mat_mul(A, B)
        # [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        assert result[0][0] == pytest.approx(19.0, abs=1e-10)
        assert result[0][1] == pytest.approx(22.0, abs=1e-10)
        assert result[1][0] == pytest.approx(43.0, abs=1e-10)
        assert result[1][1] == pytest.approx(50.0, abs=1e-10)

    def test_non_square(self):
        A = [[1.0, 2.0, 3.0]]  # 1x3
        B = [[4.0], [5.0], [6.0]]  # 3x1
        result = mat_mul(A, B)
        assert len(result) == 1
        assert len(result[0]) == 1
        assert result[0][0] == pytest.approx(32.0, abs=1e-10)


# ---------------------------------------------------------------------------
# transpose
# ---------------------------------------------------------------------------

class TestTranspose:
    def test_known_values(self):
        A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        T = transpose(A)
        assert len(T) == 3
        assert len(T[0]) == 2
        assert T[0][0] == pytest.approx(1.0, abs=1e-10)
        assert T[0][1] == pytest.approx(4.0, abs=1e-10)
        assert T[2][1] == pytest.approx(6.0, abs=1e-10)

    def test_double_transpose_is_original(self):
        A = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        TT = transpose(transpose(A))
        for i in range(len(A)):
            for j in range(len(A[0])):
                assert TT[i][j] == pytest.approx(A[i][j], abs=1e-10)

    def test_empty(self):
        assert transpose([]) == []

    def test_square(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        T = transpose(A)
        assert T[0][1] == pytest.approx(3.0, abs=1e-10)
        assert T[1][0] == pytest.approx(2.0, abs=1e-10)


# ---------------------------------------------------------------------------
# euclidean_distance
# ---------------------------------------------------------------------------

class TestEuclideanDistance:
    def test_same_point(self):
        assert euclidean_distance([1.0, 2.0], [1.0, 2.0]) == pytest.approx(
            0.0, abs=1e-10
        )

    def test_known_distance(self):
        # 3-4-5 triangle
        assert euclidean_distance([0.0, 0.0], [3.0, 4.0]) == pytest.approx(
            5.0, abs=1e-10
        )

    def test_one_dimension(self):
        assert euclidean_distance([0.0], [7.0]) == pytest.approx(7.0, abs=1e-10)


# ---------------------------------------------------------------------------
# distance_matrix
# ---------------------------------------------------------------------------

class TestDistanceMatrix:
    def test_symmetric(self):
        X = [[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]]
        dist = distance_matrix(X)
        n = len(X)
        for i in range(n):
            for j in range(n):
                assert dist[i][j] == pytest.approx(dist[j][i], abs=1e-10)

    def test_zero_diagonal(self):
        X = [[1.0, 2.0], [3.0, 4.0]]
        dist = distance_matrix(X)
        for i in range(len(X)):
            assert dist[i][i] == pytest.approx(0.0, abs=1e-10)

    def test_known_distances(self):
        X = [[0.0, 0.0], [3.0, 4.0]]
        dist = distance_matrix(X)
        assert dist[0][1] == pytest.approx(5.0, abs=1e-10)
        assert dist[1][0] == pytest.approx(5.0, abs=1e-10)

    def test_single_point(self):
        dist = distance_matrix([[5.0, 5.0]])
        assert dist == [[pytest.approx(0.0, abs=1e-10)]]
