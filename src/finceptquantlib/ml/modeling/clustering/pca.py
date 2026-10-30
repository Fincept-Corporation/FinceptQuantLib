"""
Principal Component Analysis (PCA)

Dimensionality reduction via eigendecomposition of the covariance
matrix using the power iteration method. All operations use pure
Python with no NumPy dependency.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional

from finceptquantlib.ml._matrix_utils import (
    column_mean,
    covariance_matrix,
)


@dataclass
class PCAResult:
    """Result of principal component analysis.

    Attributes
    ----------
    components : List[List[float]]
        Principal component vectors (n_components x n_features).
        Each row is a unit eigenvector.
    explained_variance : List[float]
        Eigenvalue (variance) for each component.
    explained_variance_ratio : List[float]
        Fraction of total variance explained by each component.
    cumulative_variance_ratio : List[float]
        Cumulative sum of explained variance ratios.
    mean : List[float]
        Per-feature mean used for centering.
    n_components : int
        Number of components retained.
    total_variance : float
        Sum of all eigenvalues (total data variance).
    """

    components: List[List[float]]
    explained_variance: List[float]
    explained_variance_ratio: List[float]
    cumulative_variance_ratio: List[float]
    mean: List[float]
    n_components: int
    total_variance: float


class PCA:
    """Principal Component Analysis via power iteration.

    Parameters
    ----------
    n_components : int or None
        Number of components to keep. If ``None``, all
        components are kept unless ``variance_threshold``
        is specified.
    variance_threshold : float or None
        If set, keep enough components to explain at least
        this fraction of total variance (value in [0, 1]).
        Ignored when ``n_components`` is set.

    Examples
    --------
    >>> from finceptquantlib.ml.modeling.clustering import PCA
    >>> X = [[2.5, 2.4], [0.5, 0.7], [2.2, 2.9],
    ...      [1.9, 2.2], [3.1, 3.0]]
    >>> pca = PCA(n_components=1)
    >>> result = pca.fit(X)
    >>> len(result.components)
    1
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        variance_threshold: Optional[float] = None,
    ) -> None:
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self._components: List[List[float]] = []
        self._mean: List[float] = []
        self._explained_variance: List[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: List[List[float]]) -> PCAResult:
        """Fit PCA on the data.

        Centers the data, computes the covariance matrix, and
        extracts principal components via power iteration with
        deflation.

        Parameters
        ----------
        X : List[List[float]]
            Data matrix (n_samples x n_features).

        Returns
        -------
        PCAResult
            Principal components, explained variance, and
            related statistics.
        """
        m = len(X[0]) if X else 0
        self._mean = column_mean(X)

        cov = covariance_matrix(X, ddof=1)

        # Extract all eigenvalues / eigenvectors via power iteration
        eigenvalues, eigenvectors = self._eigen_decomposition(cov, m)

        total_var = sum(eigenvalues)
        if total_var <= 0.0:
            total_var = 1.0  # prevent division by zero

        # Determine how many components to keep
        k = m
        if self.n_components is not None:
            k = min(self.n_components, m)
        elif self.variance_threshold is not None:
            cumsum = 0.0
            k = m
            for idx, ev in enumerate(eigenvalues):
                cumsum += ev / total_var
                if cumsum >= self.variance_threshold:
                    k = idx + 1
                    break

        components = eigenvectors[:k]
        explained = eigenvalues[:k]
        ratios = [ev / total_var for ev in explained]
        cum_ratios: List[float] = []
        cs = 0.0
        for r in ratios:
            cs += r
            cum_ratios.append(cs)

        self._components = components
        self._explained_variance = explained

        return PCAResult(
            components=components,
            explained_variance=explained,
            explained_variance_ratio=ratios,
            cumulative_variance_ratio=cum_ratios,
            mean=self._mean,
            n_components=k,
            total_variance=total_var,
        )

    def transform(
        self, X: List[List[float]]
    ) -> List[List[float]]:
        """Project data onto principal components.

        Parameters
        ----------
        X : List[List[float]]
            Data matrix (n_samples x n_features).

        Returns
        -------
        List[List[float]]
            Projected data (n_samples x n_components).

        Raises
        ------
        RuntimeError
            If the model has not been fit.
        """
        if not self._components:
            raise RuntimeError("PCA has not been fit yet.")
        m = len(self._mean)
        # Center and project: Z = (X - mean) @ components^T
        result: List[List[float]] = []
        for row in X:
            centered = [row[j] - self._mean[j] for j in range(m)]
            projected = [
                sum(
                    centered[j] * comp[j] for j in range(m)
                )
                for comp in self._components
            ]
            result.append(projected)
        return result

    def inverse_transform(
        self, Z: List[List[float]]
    ) -> List[List[float]]:
        """Reconstruct data from principal component space.

        Parameters
        ----------
        Z : List[List[float]]
            Projected data (n_samples x n_components).

        Returns
        -------
        List[List[float]]
            Reconstructed data (n_samples x n_features).

        Raises
        ------
        RuntimeError
            If the model has not been fit.
        """
        if not self._components:
            raise RuntimeError("PCA has not been fit yet.")
        m = len(self._mean)
        k = len(self._components)
        result: List[List[float]] = []
        for z_row in Z:
            reconstructed = [self._mean[j] for j in range(m)]
            for c_idx in range(k):
                for j in range(m):
                    reconstructed[j] += (
                        z_row[c_idx] * self._components[c_idx][j]
                    )
            result.append(reconstructed)
        return result

    def fit_transform(
        self, X: List[List[float]]
    ) -> List[List[float]]:
        """Fit and transform in one step.

        Parameters
        ----------
        X : List[List[float]]
            Data matrix (n_samples x n_features).

        Returns
        -------
        List[List[float]]
            Projected data (n_samples x n_components).
        """
        self.fit(X)
        return self.transform(X)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _eigen_decomposition(
        cov: List[List[float]],
        m: int,
        max_iter: int = 1000,
        tol: float = 1e-10,
    ) -> tuple[List[float], List[List[float]]]:
        """Eigendecomposition via power iteration with deflation.

        Parameters
        ----------
        cov : List[List[float]]
            Symmetric covariance matrix (m x m).
        m : int
            Matrix dimension.
        max_iter : int
            Maximum power iterations per eigenvector.
        tol : float
            Convergence tolerance.

        Returns
        -------
        tuple[List[float], List[List[float]]]
            (eigenvalues sorted descending,
             eigenvectors as row vectors).
        """
        # Work on a mutable copy for deflation
        C = [list(row) for row in cov]
        rng = random.Random(42)

        eigenvalues: List[float] = []
        eigenvectors: List[List[float]] = []

        for _ in range(m):
            # Random initial vector
            v = [rng.gauss(0.0, 1.0) for _ in range(m)]
            norm = math.sqrt(sum(vi * vi for vi in v))
            if norm == 0.0:
                norm = 1.0
            v = [vi / norm for vi in v]

            eigenvalue = 0.0
            for iteration in range(max_iter):
                # Matrix-vector product: w = C @ v
                w = [
                    sum(C[i][j] * v[j] for j in range(m))
                    for i in range(m)
                ]
                # New eigenvalue estimate: v^T @ w
                eigenvalue_new = sum(v[i] * w[i] for i in range(m))

                # Normalize w
                norm = math.sqrt(sum(wi * wi for wi in w))
                if norm < 1e-15:
                    break
                v_new = [wi / norm for wi in w]

                # Check convergence
                diff = math.sqrt(
                    sum(
                        (v_new[i] - v[i]) ** 2 for i in range(m)
                    )
                )
                v = v_new
                eigenvalue = eigenvalue_new

                if diff < tol:
                    break

            # Ensure non-negative eigenvalue (numerical noise)
            eigenvalue = max(0.0, eigenvalue)
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)

            # Deflate: C = C - eigenvalue * v * v^T
            for i in range(m):
                for j in range(m):
                    C[i][j] -= eigenvalue * v[i] * v[j]

        return eigenvalues, eigenvectors
