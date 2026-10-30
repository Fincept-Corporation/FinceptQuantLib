"""
Factor Models for Portfolio Construction

Provides statistical and cross-sectional factor models for
decomposing asset returns into systematic factor exposures:

- **StatisticalFactorModel**: PCA-based factor extraction
  via eigendecomposition of the return covariance matrix.
- **CrossSectionalFactorModel**: Fama-MacBeth (1973) style
  cross-sectional regression of returns on characteristics.

All computations are pure Python with no external numerical
dependencies.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from finceptquantlib.ml._matrix_utils import (
    center_matrix,
    column_mean,
    covariance_matrix,
    dot_product,
    mat_mul,
    mat_vec_mul,
    transpose,
)


# -------------------------------------------------------------------
# Result dataclass
# -------------------------------------------------------------------


@dataclass
class FactorModelResult:
    """Result of factor model estimation.

    Attributes
    ----------
    factor_loadings : List[List[float]]
        Factor loading matrix (n_assets x n_factors).
    factor_returns : List[List[float]]
        Estimated factor returns (n_observations x n_factors).
    idiosyncratic_risk : List[float]
        Idiosyncratic (residual) variance per asset.
    r_squared : List[float]
        R-squared per asset (fraction of variance explained).
    n_factors : int
        Number of factors extracted.
    explained_variance_ratio : List[float]
        Fraction of total variance explained by each factor.
    """

    factor_loadings: List[List[float]]
    factor_returns: List[List[float]]
    idiosyncratic_risk: List[float]
    r_squared: List[float]
    n_factors: int
    explained_variance_ratio: List[float]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _vec_norm(v: List[float]) -> float:
    """Euclidean norm of a vector."""
    return math.sqrt(sum(x * x for x in v))


def _vec_scale(v: List[float], s: float) -> List[float]:
    """Scale a vector by a scalar."""
    return [x * s for x in v]


def _vec_sub(a: List[float], b: List[float]) -> List[float]:
    """Element-wise subtraction a - b."""
    return [ai - bi for ai, bi in zip(a, b)]


def _outer_product(
    a: List[float], b: List[float]
) -> List[List[float]]:
    """Outer product of two vectors."""
    return [[ai * bj for bj in b] for ai in a]


def _mat_sub(
    A: List[List[float]], B: List[List[float]]
) -> List[List[float]]:
    """Element-wise subtraction A - B."""
    return [
        [A[i][j] - B[i][j] for j in range(len(A[0]))]
        for i in range(len(A))
    ]


def _power_iteration(
    C: List[List[float]],
    n_iter: int = 300,
    tol: float = 1e-10,
    seed: int = 0,
) -> Tuple[float, List[float]]:
    """Find dominant eigenvalue/eigenvector via power iteration.

    Parameters
    ----------
    C : List[List[float]]
        Symmetric matrix (p x p).
    n_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance for eigenvector change.
    seed : int
        Random seed for initial vector.

    Returns
    -------
    Tuple[float, List[float]]
        (eigenvalue, eigenvector) pair.
    """
    p = len(C)
    rng = random.Random(seed)
    v = [rng.gauss(0, 1) for _ in range(p)]
    norm = _vec_norm(v)
    if norm > 0:
        v = _vec_scale(v, 1.0 / norm)

    eigenvalue = 0.0
    for _ in range(n_iter):
        v_new = mat_vec_mul(C, v)
        norm = _vec_norm(v_new)
        if norm < 1e-14:
            break
        v_new = _vec_scale(v_new, 1.0 / norm)

        # Check convergence
        diff = _vec_norm(_vec_sub(v_new, v))
        # Also check -v (eigenvector sign ambiguity)
        diff_neg = _vec_norm(
            [v_new[i] + v[i] for i in range(p)]
        )
        v = v_new
        if min(diff, diff_neg) < tol:
            break

    # Rayleigh quotient for eigenvalue
    Cv = mat_vec_mul(C, v)
    eigenvalue = dot_product(v, Cv)
    return eigenvalue, v


def _deflate_matrix(
    C: List[List[float]],
    eigenvalue: float,
    eigenvector: List[float],
) -> List[List[float]]:
    """Deflate matrix: C = C - lambda * v v'.

    Parameters
    ----------
    C : List[List[float]]
        Matrix to deflate.
    eigenvalue : float
        Eigenvalue to remove.
    eigenvector : List[float]
        Corresponding eigenvector (unit norm).

    Returns
    -------
    List[List[float]]
        Deflated matrix.
    """
    p = len(C)
    vvt = _outer_product(eigenvector, eigenvector)
    return [
        [
            C[i][j] - eigenvalue * vvt[i][j]
            for j in range(p)
        ]
        for i in range(p)
    ]


def _ols_fit(
    X: List[List[float]], y: List[float]
) -> List[float]:
    """Ordinary least squares: beta = (X'X)^{-1} X'y.

    Solves via normal equations with Cholesky decomposition.

    Parameters
    ----------
    X : List[List[float]]
        Design matrix (n x k).
    y : List[float]
        Response vector (n,).

    Returns
    -------
    List[float]
        Coefficient vector (k,).
    """
    Xt = transpose(X)
    XtX = mat_mul(Xt, X)
    Xty = mat_vec_mul(Xt, y)

    # Solve via Cholesky
    k = len(XtX)
    # Regularize slightly for numerical stability
    for i in range(k):
        XtX[i][i] += 1e-10

    # Cholesky decomposition
    L = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(i + 1):
            s = sum(L[i][m] * L[j][m] for m in range(j))
            if i == j:
                val = XtX[i][i] - s
                L[i][j] = math.sqrt(max(val, 1e-14))
            else:
                if L[j][j] > 0:
                    L[i][j] = (XtX[i][j] - s) / L[j][j]

    # Forward substitution L z = Xty
    z = [0.0] * k
    for i in range(k):
        s = sum(L[i][m] * z[m] for m in range(i))
        if L[i][i] > 0:
            z[i] = (Xty[i] - s) / L[i][i]

    # Back substitution L' beta = z
    Lt = transpose(L)
    beta = [0.0] * k
    for i in range(k - 1, -1, -1):
        s = sum(Lt[i][m] * beta[m] for m in range(i + 1, k))
        if Lt[i][i] > 0:
            beta[i] = (z[i] - s) / Lt[i][i]

    return beta


# -------------------------------------------------------------------
# StatisticalFactorModel
# -------------------------------------------------------------------


class StatisticalFactorModel:
    """PCA-based statistical factor model.

    Extracts latent factors from the return covariance matrix
    using eigendecomposition via power iteration with deflation.

    Parameters
    ----------
    n_factors : int
        Number of principal component factors to extract.
        Default is 5.

    Examples
    --------
    >>> from finceptquantlib.ml.markets.portfolio_ml.factor_models import (
    ...     StatisticalFactorModel,
    ... )
    >>> returns = [
    ...     [0.01, 0.02, 0.015],
    ...     [0.03, -0.01, 0.005],
    ...     [-0.02, 0.01, -0.005],
    ...     [0.005, 0.015, 0.01],
    ... ]
    >>> model = StatisticalFactorModel(n_factors=2)
    >>> result = model.fit(returns)
    >>> result.n_factors
    2
    """

    def __init__(self, n_factors: int = 5) -> None:
        if n_factors < 1:
            raise ValueError(
                f"n_factors must be >= 1, got {n_factors}"
            )
        self.n_factors = n_factors
        self._loadings: Optional[List[List[float]]] = None
        self._factor_returns: Optional[List[List[float]]] = None
        self._eigenvalues: Optional[List[float]] = None

    def fit(
        self, returns: List[List[float]]
    ) -> FactorModelResult:
        """Fit PCA factor model to asset returns.

        Parameters
        ----------
        returns : List[List[float]]
            Asset return matrix (T x N) where T is the number
            of observations and N is the number of assets.

        Returns
        -------
        FactorModelResult
            Factor model estimation results including loadings,
            factor returns, idiosyncratic risk, and R-squared.

        Raises
        ------
        ValueError
            If n_factors exceeds number of assets.
        """
        T = len(returns)
        N = len(returns[0]) if returns else 0

        k = min(self.n_factors, N, T)
        if k < 1:
            raise ValueError(
                f"Cannot extract {self.n_factors} factors "
                f"from {N} assets with {T} observations"
            )

        # Center returns (demean each column)
        X_centered = center_matrix(returns)

        # Covariance matrix (N x N)
        C = covariance_matrix(returns, ddof=1)
        total_var = sum(C[i][i] for i in range(N))

        # Eigendecomposition via power iteration + deflation
        eigenvalues: List[float] = []
        eigenvectors: List[List[float]] = []
        C_deflated = [row[:] for row in C]

        for factor_idx in range(k):
            lam, v = _power_iteration(
                C_deflated,
                n_iter=500,
                tol=1e-12,
                seed=42 + factor_idx,
            )
            # Ensure positive eigenvalue
            lam = max(lam, 0.0)
            eigenvalues.append(lam)
            eigenvectors.append(v)
            C_deflated = _deflate_matrix(C_deflated, lam, v)

        # Factor loadings: N x K (eigenvectors as columns)
        # Each eigenvector is length N
        loadings = [
            [eigenvectors[f][i] for f in range(k)]
            for i in range(N)
        ]

        # Factor returns: T x K
        # f_t = X_centered[t] @ loadings (N-vector @ N x K)
        factor_returns = [
            [
                dot_product(X_centered[t], eigenvectors[f])
                for f in range(k)
            ]
            for t in range(T)
        ]

        # Reconstructed returns: T x N
        # r_hat_t = factor_returns[t] @ loadings' (K-vector @ K x N)
        loadings_T = transpose(loadings)  # K x N
        reconstructed = mat_mul(factor_returns, loadings_T)

        # Idiosyncratic variance per asset
        idio_var = [0.0] * N
        total_asset_var = [0.0] * N
        for t in range(T):
            for j in range(N):
                resid = X_centered[t][j] - reconstructed[t][j]
                idio_var[j] += resid * resid
                total_asset_var[j] += (
                    X_centered[t][j] * X_centered[t][j]
                )

        denom = max(T - 1, 1)
        idio_risk = [v / denom for v in idio_var]
        total_asset_var = [v / denom for v in total_asset_var]

        # R-squared per asset
        r_squared = []
        for j in range(N):
            if total_asset_var[j] > 0:
                r2 = 1.0 - idio_risk[j] / total_asset_var[j]
                r_squared.append(max(0.0, min(1.0, r2)))
            else:
                r_squared.append(0.0)

        # Explained variance ratio per factor
        evr = []
        for lam in eigenvalues:
            if total_var > 0:
                evr.append(lam / total_var)
            else:
                evr.append(0.0)

        self._loadings = loadings
        self._factor_returns = factor_returns
        self._eigenvalues = eigenvalues

        return FactorModelResult(
            factor_loadings=loadings,
            factor_returns=factor_returns,
            idiosyncratic_risk=idio_risk,
            r_squared=r_squared,
            n_factors=k,
            explained_variance_ratio=evr,
        )

    def predict(
        self, factor_values: List[float]
    ) -> List[float]:
        """Predict expected returns from factor values.

        Parameters
        ----------
        factor_values : List[float]
            Factor realization values (n_factors,).

        Returns
        -------
        List[float]
            Expected returns per asset (n_assets,).

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self._loadings is None:
            raise RuntimeError(
                "Model must be fitted before calling predict"
            )
        # expected_return = loadings @ factor_values
        return mat_vec_mul(self._loadings, factor_values)

    def factor_covariance(self) -> List[List[float]]:
        """Compute covariance matrix of factor returns.

        Returns
        -------
        List[List[float]]
            Factor covariance matrix (n_factors x n_factors).

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self._factor_returns is None:
            raise RuntimeError(
                "Model must be fitted before calling "
                "factor_covariance"
            )
        return covariance_matrix(self._factor_returns, ddof=1)


# -------------------------------------------------------------------
# CrossSectionalFactorModel
# -------------------------------------------------------------------


class CrossSectionalFactorModel:
    """Fama-MacBeth cross-sectional factor model.

    Performs period-by-period cross-sectional regressions of
    asset returns on observable characteristics (factor
    exposures), following Fama & MacBeth (1973).

    Parameters
    ----------
    n_factors : int or None
        Number of factors (characteristics). If None, inferred
        from input data. Default is None.

    Notes
    -----
    Fama, E.F. & MacBeth, J.D. (1973). "Risk, Return, and
    Equilibrium: Empirical Tests." Journal of Political Economy
    81(3):607-636.

    Examples
    --------
    >>> from finceptquantlib.ml.markets.portfolio_ml.factor_models import (
    ...     CrossSectionalFactorModel,
    ... )
    >>> # 3 time periods, 4 assets, 2 characteristics
    >>> returns = [
    ...     [0.01, 0.02, -0.01, 0.03],
    ...     [0.02, -0.01, 0.01, 0.00],
    ...     [-0.01, 0.03, 0.02, -0.02],
    ... ]
    >>> chars = [
    ...     [[1.0, 0.5], [0.8, 0.3], [1.2, 0.7], [0.6, 0.9]],
    ...     [[1.1, 0.4], [0.9, 0.2], [1.1, 0.8], [0.7, 0.8]],
    ...     [[1.0, 0.6], [0.7, 0.4], [1.3, 0.6], [0.5, 1.0]],
    ... ]
    >>> model = CrossSectionalFactorModel()
    >>> result = model.fit(returns, chars)
    """

    def __init__(
        self, n_factors: Optional[int] = None
    ) -> None:
        self.n_factors = n_factors
        self._mean_factor_returns: Optional[List[float]] = None
        self._premium_std_errors: Optional[List[float]] = None
        self._last_loadings: Optional[List[List[float]]] = None

    def fit(
        self,
        returns: List[List[float]],
        characteristics: List[List[List[float]]],
    ) -> FactorModelResult:
        """Fit Fama-MacBeth cross-sectional factor model.

        Parameters
        ----------
        returns : List[List[float]]
            Asset return matrix (T x N). T time periods, N assets.
        characteristics : List[List[List[float]]]
            Factor exposure matrix (T x N x K). For each time
            period t, characteristics[t] is an N x K matrix of
            asset characteristics.

        Returns
        -------
        FactorModelResult
            Factor model results with loadings, factor returns,
            idiosyncratic risk, and fit statistics.

        Raises
        ------
        ValueError
            If dimensions are inconsistent.
        """
        T = len(returns)
        N = len(returns[0]) if returns else 0

        if len(characteristics) != T:
            raise ValueError(
                f"characteristics has {len(characteristics)} "
                f"periods but returns has {T}"
            )

        K = len(characteristics[0][0]) if (
            characteristics and characteristics[0]
            and characteristics[0][0]
        ) else 0

        if self.n_factors is not None and K != self.n_factors:
            raise ValueError(
                f"Expected {self.n_factors} factors but "
                f"characteristics have {K} columns"
            )

        # Fama-MacBeth: cross-sectional regression each period
        # r_{i,t} = Z_{i,t-1} @ gamma_t + epsilon_{i,t}
        all_factor_returns: List[List[float]] = []

        for t in range(T):
            # Design matrix: N x K
            Z_t = characteristics[t]
            # Response: returns at time t (N,)
            r_t = returns[t]
            # OLS: gamma_t = (Z'Z)^{-1} Z'r
            gamma_t = _ols_fit(Z_t, r_t)
            all_factor_returns.append(gamma_t)

        # Average factor returns across time (Fama-MacBeth)
        mean_fr = [0.0] * K
        for t in range(T):
            for k in range(K):
                mean_fr[k] += all_factor_returns[t][k]
        mean_fr = [v / T for v in mean_fr]

        # Standard errors (Newey-West lag 0 = OLS standard errors)
        fr_var = [0.0] * K
        for t in range(T):
            for k in range(K):
                diff = all_factor_returns[t][k] - mean_fr[k]
                fr_var[k] += diff * diff
        fr_var = [v / max(T - 1, 1) for v in fr_var]
        fr_se = [math.sqrt(v / T) for v in fr_var]

        # Factor loadings = last period characteristics (N x K)
        last_loadings = characteristics[-1]

        # Reconstruct returns and compute idiosyncratic risk
        idio_var = [0.0] * N
        total_var = [0.0] * N
        r_means = column_mean(returns)

        for t in range(T):
            Z_t = characteristics[t]
            for i in range(N):
                fitted = dot_product(
                    Z_t[i], all_factor_returns[t]
                )
                resid = returns[t][i] - fitted
                idio_var[i] += resid * resid
                dev = returns[t][i] - r_means[i]
                total_var[i] += dev * dev

        denom = max(T - 1, 1)
        idio_risk = [v / denom for v in idio_var]
        total_var = [v / denom for v in total_var]

        # R-squared per asset
        r_squared = []
        for i in range(N):
            if total_var[i] > 0:
                r2 = 1.0 - idio_risk[i] / total_var[i]
                r_squared.append(max(0.0, min(1.0, r2)))
            else:
                r_squared.append(0.0)

        # Explained variance ratio per factor
        # Use fraction of cross-sectional variance explained
        total_cs_var = sum(fr_var)
        evr = []
        for k in range(K):
            if total_cs_var > 0:
                evr.append(fr_var[k] / total_cs_var)
            else:
                evr.append(1.0 / K if K > 0 else 0.0)

        self._mean_factor_returns = mean_fr
        self._premium_std_errors = fr_se
        self._last_loadings = last_loadings

        return FactorModelResult(
            factor_loadings=last_loadings,
            factor_returns=all_factor_returns,
            idiosyncratic_risk=idio_risk,
            r_squared=r_squared,
            n_factors=K,
            explained_variance_ratio=evr,
        )

    def predict(
        self, characteristics: List[List[float]]
    ) -> List[float]:
        """Predict expected returns from characteristics.

        Parameters
        ----------
        characteristics : List[List[float]]
            Asset characteristics matrix (N x K).

        Returns
        -------
        List[float]
            Expected returns per asset (N,).

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self._mean_factor_returns is None:
            raise RuntimeError(
                "Model must be fitted before calling predict"
            )
        return mat_vec_mul(
            characteristics, self._mean_factor_returns
        )

    def factor_premiums(self) -> List[float]:
        """Fama-MacBeth factor risk premiums (time-series mean of
        the per-period cross-sectional slopes).

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self._mean_factor_returns is None:
            raise RuntimeError("Model must be fitted first")
        return list(self._mean_factor_returns)

    def premium_t_statistics(self) -> List[float]:
        """Fama-MacBeth t-statistics of the factor premiums
        (premium / standard error of the per-period slopes).

        A factor whose slope never varies has no standard error;
        its t-statistic is NaN.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self._mean_factor_returns is None or self._premium_std_errors is None:
            raise RuntimeError("Model must be fitted first")
        return [
            m / se if se > 0 else float("nan")
            for m, se in zip(self._mean_factor_returns, self._premium_std_errors)
        ]
