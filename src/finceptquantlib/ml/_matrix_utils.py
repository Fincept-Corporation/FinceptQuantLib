"""
Internal Matrix Utilities for ML Module

Convenience wrappers around numerical.linalg for common
ML operations: covariance, correlation, centering, etc.
All operate on List[List[float]] matrices.
"""

from __future__ import annotations

import math
from typing import List, Tuple


def column_mean(X: List[List[float]]) -> List[float]:
    """Compute mean of each column.

    Parameters
    ----------
    X : List[List[float]]
        Data matrix (n_samples x n_features).

    Returns
    -------
    List[float]
        Mean for each feature (n_features,).
    """
    n = len(X)
    if n == 0:
        return []
    m = len(X[0])
    means = [0.0] * m
    for row in X:
        for j in range(m):
            means[j] += row[j]
    for j in range(m):
        means[j] /= n
    return means


def column_std(
    X: List[List[float]], ddof: int = 0
) -> List[float]:
    """Compute standard deviation of each column.

    Parameters
    ----------
    X : List[List[float]]
        Data matrix (n_samples x n_features).
    ddof : int
        Delta degrees of freedom. Use 0 for population,
        1 for sample standard deviation.

    Returns
    -------
    List[float]
        Standard deviation for each feature.
    """
    n = len(X)
    if n <= ddof:
        return [0.0] * (len(X[0]) if X else 0)
    m = len(X[0])
    means = column_mean(X)
    variances = [0.0] * m
    for row in X:
        for j in range(m):
            diff = row[j] - means[j]
            variances[j] += diff * diff
    return [
        math.sqrt(v / (n - ddof)) if v > 0 else 0.0
        for v in variances
    ]


def column_min(X: List[List[float]]) -> List[float]:
    """Compute minimum of each column.

    Parameters
    ----------
    X : List[List[float]]
        Data matrix (n_samples x n_features).

    Returns
    -------
    List[float]
        Minimum for each feature.
    """
    if not X:
        return []
    m = len(X[0])
    mins = [X[0][j] for j in range(m)]
    for row in X[1:]:
        for j in range(m):
            if row[j] < mins[j]:
                mins[j] = row[j]
    return mins


def column_max(X: List[List[float]]) -> List[float]:
    """Compute maximum of each column.

    Parameters
    ----------
    X : List[List[float]]
        Data matrix (n_samples x n_features).

    Returns
    -------
    List[float]
        Maximum for each feature.
    """
    if not X:
        return []
    m = len(X[0])
    maxs = [X[0][j] for j in range(m)]
    for row in X[1:]:
        for j in range(m):
            if row[j] > maxs[j]:
                maxs[j] = row[j]
    return maxs


def column_median(X: List[List[float]]) -> List[float]:
    """Compute median of each column.

    Parameters
    ----------
    X : List[List[float]]
        Data matrix (n_samples x n_features).

    Returns
    -------
    List[float]
        Median for each feature.
    """
    if not X:
        return []
    n = len(X)
    m = len(X[0])
    medians = []
    for j in range(m):
        col = sorted(row[j] for row in X)
        if n % 2 == 1:
            medians.append(col[n // 2])
        else:
            medians.append((col[n // 2 - 1] + col[n // 2]) / 2.0)
    return medians


def column_percentile(
    X: List[List[float]], q: float
) -> List[float]:
    """Compute percentile of each column.

    Parameters
    ----------
    X : List[List[float]]
        Data matrix (n_samples x n_features).
    q : float
        Percentile in [0, 1].

    Returns
    -------
    List[float]
        Percentile value for each feature.
    """
    if not X:
        return []
    n = len(X)
    m = len(X[0])
    result = []
    for j in range(m):
        col = sorted(row[j] for row in X)
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        result.append(col[lo] * (1 - frac) + col[hi] * frac)
    return result


def center_matrix(X: List[List[float]]) -> List[List[float]]:
    """Subtract column means from data matrix.

    Parameters
    ----------
    X : List[List[float]]
        Data matrix (n_samples x n_features).

    Returns
    -------
    List[List[float]]
        Centered matrix with zero column means.
    """
    means = column_mean(X)
    m = len(means)
    return [
        [row[j] - means[j] for j in range(m)]
        for row in X
    ]


def standardize_matrix(
    X: List[List[float]], ddof: int = 0
) -> Tuple[List[List[float]], List[float], List[float]]:
    """Standardize to zero mean and unit variance.

    Parameters
    ----------
    X : List[List[float]]
        Data matrix (n_samples x n_features).
    ddof : int
        Degrees of freedom for std computation.

    Returns
    -------
    Tuple[List[List[float]], List[float], List[float]]
        (standardized_X, means, stds)
    """
    means = column_mean(X)
    stds = column_std(X, ddof=ddof)
    m = len(means)
    result = []
    for row in X:
        new_row = []
        for j in range(m):
            if stds[j] > 0:
                new_row.append((row[j] - means[j]) / stds[j])
            else:
                new_row.append(0.0)
        result.append(new_row)
    return result, means, stds


def add_intercept(X: List[List[float]]) -> List[List[float]]:
    """Prepend a column of ones for intercept term.

    Parameters
    ----------
    X : List[List[float]]
        Data matrix (n_samples x n_features).

    Returns
    -------
    List[List[float]]
        Matrix with intercept column (n_samples x (n_features+1)).
    """
    return [[1.0] + row for row in X]


def covariance_matrix(
    X: List[List[float]], ddof: int = 1
) -> List[List[float]]:
    """Compute sample covariance matrix.

    Cov(i, j) = (1/(n-ddof)) * sum_k (X_ki - mean_i)(X_kj - mean_j)

    Parameters
    ----------
    X : List[List[float]]
        Data matrix (n_samples x n_features).
    ddof : int
        Delta degrees of freedom (1 for sample covariance).

    Returns
    -------
    List[List[float]]
        Covariance matrix (n_features x n_features).
    """
    n = len(X)
    if n <= ddof:
        m = len(X[0]) if X else 0
        return [[0.0] * m for _ in range(m)]
    centered = center_matrix(X)
    m = len(centered[0])
    cov = [[0.0] * m for _ in range(m)]
    for row in centered:
        for i in range(m):
            for j in range(i, m):
                cov[i][j] += row[i] * row[j]
    denom = n - ddof
    for i in range(m):
        for j in range(i, m):
            cov[i][j] /= denom
            cov[j][i] = cov[i][j]
    return cov


def correlation_matrix(
    X: List[List[float]]
) -> List[List[float]]:
    """Compute Pearson correlation matrix.

    Corr(i, j) = Cov(i, j) / (std_i * std_j)

    Parameters
    ----------
    X : List[List[float]]
        Data matrix (n_samples x n_features).

    Returns
    -------
    List[List[float]]
        Correlation matrix (n_features x n_features).
    """
    cov = covariance_matrix(X, ddof=1)
    m = len(cov)
    stds = [math.sqrt(cov[i][i]) if cov[i][i] > 0 else 0.0
            for i in range(m)]
    corr = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if stds[i] > 0 and stds[j] > 0:
                corr[i][j] = cov[i][j] / (stds[i] * stds[j])
                corr[i][j] = max(-1.0, min(1.0, corr[i][j]))
            elif i == j:
                corr[i][j] = 1.0
    return corr


def dot_product(a: List[float], b: List[float]) -> float:
    """Compute dot product of two vectors.

    Parameters
    ----------
    a : List[float]
        First vector.
    b : List[float]
        Second vector.

    Returns
    -------
    float
        Dot product a . b.
    """
    return sum(ai * bi for ai, bi in zip(a, b))


def mat_vec_mul(
    A: List[List[float]], x: List[float]
) -> List[float]:
    """Matrix-vector multiplication.

    Parameters
    ----------
    A : List[List[float]]
        Matrix (m x n).
    x : List[float]
        Vector (n,).

    Returns
    -------
    List[float]
        Result vector (m,).
    """
    return [dot_product(row, x) for row in A]


def mat_mul(
    A: List[List[float]], B: List[List[float]]
) -> List[List[float]]:
    """Matrix-matrix multiplication.

    Parameters
    ----------
    A : List[List[float]]
        Matrix (m x k).
    B : List[List[float]]
        Matrix (k x n).

    Returns
    -------
    List[List[float]]
        Result matrix (m x n).
    """
    m = len(A)
    k = len(B)
    n = len(B[0]) if B else 0
    result = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for p in range(k):
                s += A[i][p] * B[p][j]
            result[i][j] = s
    return result


def transpose(A: List[List[float]]) -> List[List[float]]:
    """Matrix transpose.

    Parameters
    ----------
    A : List[List[float]]
        Matrix (m x n).

    Returns
    -------
    List[List[float]]
        Transposed matrix (n x m).
    """
    if not A:
        return []
    m = len(A)
    n = len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two vectors.

    Parameters
    ----------
    a : List[float]
        First vector.
    b : List[float]
        Second vector.

    Returns
    -------
    float
        Euclidean distance.
    """
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def distance_matrix(X: List[List[float]]) -> List[List[float]]:
    """Compute pairwise Euclidean distance matrix.

    Parameters
    ----------
    X : List[List[float]]
        Data matrix (n_samples x n_features).

    Returns
    -------
    List[List[float]]
        Distance matrix (n_samples x n_samples).
    """
    n = len(X)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_distance(X[i], X[j])
            dist[i][j] = d
            dist[j][i] = d
    return dist
