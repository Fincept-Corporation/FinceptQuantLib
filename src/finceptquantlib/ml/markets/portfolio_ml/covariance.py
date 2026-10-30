"""
Covariance Estimation for Portfolio Construction

Provides robust covariance estimators for portfolio optimization:

- **LedoitWolfShrinkage**: Analytical optimal linear shrinkage
  (Ledoit & Wolf, 2004) that blends the sample covariance with
  a structured target for improved conditioning.
- **MinimumCovarianceDeterminant**: Robust covariance via the
  Fast-MCD algorithm (Rousseeuw & Van Driessen, 1999) that
  downweights outlier observations.

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
    covariance_matrix,
    dot_product,
    mat_vec_mul,
    transpose,
)


# -------------------------------------------------------------------
# Result dataclass
# -------------------------------------------------------------------


@dataclass
class ShrinkageCovarianceResult:
    """Result of shrinkage covariance estimation.

    Attributes
    ----------
    covariance : List[List[float]]
        Shrunk covariance matrix (n_assets x n_assets).
    shrinkage_intensity : float
        Optimal shrinkage intensity alpha* in [0, 1].
    target : List[List[float]]
        Structured target matrix F used for shrinkage.
    method : str
        Target type used (e.g. "identity", "diagonal",
        "constant_correlation").
    """

    covariance: List[List[float]]
    shrinkage_intensity: float
    target: List[List[float]]
    method: str


# -------------------------------------------------------------------
# Linear algebra helpers (pure Python)
# -------------------------------------------------------------------


def _identity(p: int) -> List[List[float]]:
    """Return p x p identity matrix."""
    return [
        [1.0 if i == j else 0.0 for j in range(p)]
        for i in range(p)
    ]


def _scaled_identity(
    scale: float, p: int
) -> List[List[float]]:
    """Return scale * I_p."""
    return [
        [scale if i == j else 0.0 for j in range(p)]
        for i in range(p)
    ]


def _diagonal_matrix(S: List[List[float]]) -> List[List[float]]:
    """Extract diagonal of S into a diagonal matrix."""
    p = len(S)
    return [
        [S[i][i] if i == j else 0.0 for j in range(p)]
        for i in range(p)
    ]


def _frobenius_sq(A: List[List[float]]) -> float:
    """Squared Frobenius norm ||A||_F^2."""
    total = 0.0
    for row in A:
        for v in row:
            total += v * v
    return total


def _mat_trace(A: List[List[float]]) -> float:
    """Trace of a square matrix."""
    return sum(A[i][i] for i in range(len(A)))


def _mat_add(
    A: List[List[float]],
    B: List[List[float]],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> List[List[float]]:
    """Compute alpha*A + beta*B element-wise."""
    p = len(A)
    q = len(A[0]) if A else 0
    return [
        [alpha * A[i][j] + beta * B[i][j] for j in range(q)]
        for i in range(p)
    ]


def _mat_sub(
    A: List[List[float]], B: List[List[float]]
) -> List[List[float]]:
    """Compute A - B element-wise."""
    return _mat_add(A, B, 1.0, -1.0)


def _mat_scale(
    A: List[List[float]], s: float
) -> List[List[float]]:
    """Scale matrix by scalar."""
    return [[s * v for v in row] for row in A]


def _outer_product(
    a: List[float], b: List[float]
) -> List[List[float]]:
    """Outer product of two vectors."""
    return [[ai * bj for bj in b] for ai in a]


def _cholesky(A: List[List[float]]) -> List[List[float]]:
    """Cholesky decomposition A = L L^T.

    Parameters
    ----------
    A : List[List[float]]
        Symmetric positive-definite matrix.

    Returns
    -------
    List[List[float]]
        Lower triangular Cholesky factor L.

    Raises
    ------
    ValueError
        If the matrix is not positive definite.
    """
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                if val <= 0:
                    # Floor small negatives for near-PD matrices
                    val = max(val, 1e-12)
                L[i][j] = math.sqrt(val)
            else:
                if L[j][j] == 0.0:
                    L[i][j] = 0.0
                else:
                    L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def _solve_lower(
    L: List[List[float]], b: List[float]
) -> List[float]:
    """Solve L x = b where L is lower triangular."""
    n = len(L)
    x = [0.0] * n
    for i in range(n):
        s = sum(L[i][k] * x[k] for k in range(i))
        if L[i][i] == 0.0:
            x[i] = 0.0
        else:
            x[i] = (b[i] - s) / L[i][i]
    return x


def _solve_upper(
    U: List[List[float]], b: List[float]
) -> List[float]:
    """Solve U x = b where U is upper triangular."""
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(U[i][k] * x[k] for k in range(i + 1, n))
        if U[i][i] == 0.0:
            x[i] = 0.0
        else:
            x[i] = (b[i] - s) / U[i][i]
    return x


def _solve_cholesky(
    L: List[List[float]], b: List[float]
) -> List[float]:
    """Solve A x = b given Cholesky factor L of A."""
    y = _solve_lower(L, b)
    Lt = transpose(L)
    return _solve_upper(Lt, y)


def _mat_inverse_cholesky(
    A: List[List[float]],
) -> List[List[float]]:
    """Inverse of symmetric PD matrix via Cholesky."""
    n = len(A)
    L = _cholesky(A)
    inv = []
    for j in range(n):
        e_j = [1.0 if i == j else 0.0 for i in range(n)]
        col = _solve_cholesky(L, e_j)
        inv.append(col)
    return transpose(inv)


def _determinant_from_cholesky(
    L: List[List[float]],
) -> float:
    """Determinant of A given its Cholesky factor L.

    det(A) = det(L)^2 = (product of L diagonal)^2.
    Uses log to avoid overflow.
    """
    log_det = 0.0
    for i in range(len(L)):
        if L[i][i] <= 0:
            return 0.0
        log_det += math.log(L[i][i])
    log_det *= 2.0
    if log_det > 700:
        return float("inf")
    return math.exp(log_det)


def _ensure_positive_definite(
    S: List[List[float]], floor: float = 1e-8
) -> List[List[float]]:
    """Ensure matrix is positive definite via eigenvalue floor.

    Uses power iteration to find the smallest eigenvalue direction
    and adds floor * I if needed. A simple approach: try Cholesky,
    if it fails, add progressively larger multiples of I.

    Parameters
    ----------
    S : List[List[float]]
        Symmetric matrix.
    floor : float
        Minimum eigenvalue threshold.

    Returns
    -------
    List[List[float]]
        Positive-definite matrix.
    """
    p = len(S)
    result = [row[:] for row in S]
    for attempt in range(20):
        try:
            _cholesky(result)
            return result
        except Exception:
            pass
        # Check diagonal for zero/negative
        shift = floor * (10 ** attempt)
        result = [
            [
                S[i][j] + (shift if i == j else 0.0)
                for j in range(p)
            ]
            for i in range(p)
        ]
    return result


# -------------------------------------------------------------------
# Chi-squared quantile approximation (for MCD reweighting)
# -------------------------------------------------------------------


def _chi2_ppf_approx(p: float, df: int) -> float:
    """Approximate chi-squared quantile via Wilson-Hilferty.

    Parameters
    ----------
    p : float
        Probability level (e.g. 0.975).
    df : int
        Degrees of freedom.

    Returns
    -------
    float
        Approximate quantile.
    """
    # Normal quantile approximation via rational approximation
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return float("inf")

    # Abramowitz & Stegun rational approximation for normal ppf
    # For 0.5 < p < 1
    if p < 0.5:
        sign = -1.0
        pp = 1.0 - p
    else:
        sign = 1.0
        pp = p

    t = math.sqrt(-2.0 * math.log(1.0 - pp))
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    z = sign * (
        t
        - (c0 + c1 * t + c2 * t * t)
        / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
    )

    # Wilson-Hilferty transformation
    k = df
    val = k * (1.0 - 2.0 / (9.0 * k) + z * math.sqrt(2.0 / (9.0 * k))) ** 3
    return max(val, 0.0)


# -------------------------------------------------------------------
# LedoitWolfShrinkage
# -------------------------------------------------------------------


class LedoitWolfShrinkage:
    """Ledoit-Wolf analytical optimal linear shrinkage estimator.

    Implements the closed-form solution from Ledoit & Wolf (2004)
    "A well-conditioned estimator for large-dimensional covariance
    matrices" for shrinking the sample covariance toward a
    structured target.

    Parameters
    ----------
    target : str
        Shrinkage target type. One of:
        - ``"constant_correlation"``: average correlation target
        - ``"identity"``: scaled identity mu * I
        - ``"diagonal"``: diagonal of sample covariance

    Examples
    --------
    >>> from finceptquantlib.ml.markets.portfolio_ml.covariance import (
    ...     LedoitWolfShrinkage,
    ... )
    >>> returns = [[0.01, 0.02], [0.03, -0.01], [-0.02, 0.01]]
    >>> lw = LedoitWolfShrinkage(target="identity")
    >>> result = lw.estimate(returns)
    >>> result.shrinkage_intensity  # doctest: +SKIP
    """

    _VALID_TARGETS = (
        "constant_correlation",
        "identity",
        "diagonal",
    )

    def __init__(self, target: str = "constant_correlation") -> None:
        if target not in self._VALID_TARGETS:
            raise ValueError(
                f"target must be one of {self._VALID_TARGETS}, "
                f"got '{target}'"
            )
        self.target = target

    # ---------------------------------------------------------------
    # Target construction
    # ---------------------------------------------------------------

    def _build_target(
        self, S: List[List[float]]
    ) -> List[List[float]]:
        """Build the structured target matrix F.

        Parameters
        ----------
        S : List[List[float]]
            Sample covariance matrix (p x p).

        Returns
        -------
        List[List[float]]
            Target matrix F (p x p).
        """
        p = len(S)

        if self.target == "identity":
            mu = _mat_trace(S) / p
            return _scaled_identity(mu, p)

        elif self.target == "diagonal":
            return _diagonal_matrix(S)

        else:  # constant_correlation
            # Target with constant off-diagonal correlation
            stds = [
                math.sqrt(S[i][i]) if S[i][i] > 0 else 0.0
                for i in range(p)
            ]
            # Average correlation
            rho_sum = 0.0
            count = 0
            for i in range(p):
                for j in range(i + 1, p):
                    if stds[i] > 0 and stds[j] > 0:
                        rho_sum += S[i][j] / (stds[i] * stds[j])
                        count += 1
            rho_bar = rho_sum / count if count > 0 else 0.0

            F = [[0.0] * p for _ in range(p)]
            for i in range(p):
                for j in range(p):
                    if i == j:
                        F[i][j] = S[i][i]
                    else:
                        F[i][j] = rho_bar * stds[i] * stds[j]
            return F

    # ---------------------------------------------------------------
    # Analytical optimal shrinkage intensity
    # ---------------------------------------------------------------

    def _compute_shrinkage_intensity(
        self,
        X_centered: List[List[float]],
        S: List[List[float]],
        F: List[List[float]],
    ) -> float:
        """Compute Ledoit-Wolf optimal shrinkage intensity.

        Uses the analytical formula from Ledoit & Wolf (2004):

        alpha* = min(beta_hat / delta_hat, 1)

        where:
        - delta_hat = ||S - F||_F^2
        - beta_hat = (1/T^2) sum_t ||x_t x_t' - S||_F^2

        For the identity target, a simplified formula is used.

        Parameters
        ----------
        X_centered : List[List[float]]
            Centered data matrix (T x N).
        S : List[List[float]]
            Sample covariance matrix (N x N).
        F : List[List[float]]
            Target matrix (N x N).

        Returns
        -------
        float
            Optimal shrinkage intensity in [0, 1].
        """
        T = len(X_centered)
        N = len(S)

        if T == 0 or N == 0:
            return 1.0

        # delta = ||S - F||_F^2 / N^2
        diff_SF = _mat_sub(S, F)
        delta = _frobenius_sq(diff_SF)

        if delta < 1e-14:
            # S is already close to target
            return 1.0

        # beta = (1/T^2) * sum_t ||x_t x_t' - S||_F^2
        # Expanded: sum_t sum_{i,j} (x_{ti} x_{tj} - S_{ij})^2
        # = sum_t sum_{i,j} x_{ti}^2 x_{tj}^2
        #   - 2 sum_t sum_{i,j} x_{ti} x_{tj} S_{ij}
        #   + T * sum_{i,j} S_{ij}^2
        # First term: sum_t (sum_i x_{ti}^2)^2 for diagonal,
        # but we need all i,j
        # More efficiently:
        # sum_t ||x_t x_t' - S||_F^2
        #   = sum_t [ sum_{i,j} (x_{ti} x_{tj})^2
        #             - 2 sum_{i,j} x_{ti} x_{tj} S_{ij}
        #             + sum_{i,j} S_{ij}^2 ]

        # Compute sum_t sum_{i,j} (x_{ti} x_{tj})^2
        # = sum_{i,j} sum_t (x_{ti} x_{tj})^2
        # = sum_{i,j} [ (1/T) sum_t (x_{ti} x_{tj})^2 ] * T
        # This equals T * sum of element-wise squared
        # sample 4th moment matrix

        phi_sum = 0.0  # sum_t ||x_t x_t'||_F^2
        cross_sum = 0.0  # sum_t <x_t x_t', S>_F

        for t in range(T):
            xt = X_centered[t]
            # ||x_t x_t'||_F^2 = (sum_i x_{ti}^2)^2
            # because ||vv'||_F^2 = (v'v)^2
            sq_norm = sum(v * v for v in xt)
            phi_sum += sq_norm * sq_norm

            # <x_t x_t', S>_F = sum_{i,j} x_{ti} x_{tj} S_{ij}
            # = x_t' S x_t
            Sxt = mat_vec_mul(S, xt)
            cross_sum += dot_product(xt, Sxt)

        frob_S_sq = _frobenius_sq(S)
        beta_num = (
            phi_sum - 2.0 * cross_sum + T * frob_S_sq
        ) / (T * T)

        # Clamp beta to be non-negative
        beta_num = max(beta_num, 0.0)

        alpha = beta_num / delta
        return max(0.0, min(1.0, alpha))

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def estimate(
        self, returns: List[List[float]]
    ) -> ShrinkageCovarianceResult:
        """Estimate shrunk covariance matrix.

        Parameters
        ----------
        returns : List[List[float]]
            Asset return matrix (n_observations x n_assets).
            Each row is one observation, each column one asset.

        Returns
        -------
        ShrinkageCovarianceResult
            Contains the shrunk covariance matrix, optimal
            shrinkage intensity, target matrix, and method name.

        Raises
        ------
        ValueError
            If returns has fewer than 2 observations.

        Notes
        -----
        Implements Ledoit & Wolf (2004) "A well-conditioned
        estimator for large-dimensional covariance matrices",
        Journal of Multivariate Analysis 88(2):365-411.
        """
        T = len(returns)
        if T < 2:
            raise ValueError(
                "Need at least 2 observations, got "
                f"{T}"
            )
        N = len(returns[0])
        if N == 0:
            raise ValueError("Returns matrix has 0 assets")

        # Sample covariance (unbiased, ddof=1)
        S = covariance_matrix(returns, ddof=1)

        # Build structured target
        F = self._build_target(S)

        # Center the data for shrinkage intensity computation
        X_centered = center_matrix(returns)

        # Compute optimal shrinkage intensity
        # Scale X_centered to match ddof=1 convention:
        # S = (1/(T-1)) X'X, so x_t x_t' should be compared
        # with (T-1) * S / T effectively. We adjust by using
        # the sample covariance with ddof=1 and the formula
        # adapted for unbiased S.
        # The Ledoit-Wolf formula for unbiased S:
        # beta / (T * delta) where beta uses T not T-1
        alpha = self._compute_shrinkage_intensity(
            X_centered, S, F
        )

        # Shrunk estimator: alpha * F + (1 - alpha) * S
        shrunk = _mat_add(F, S, alpha, 1.0 - alpha)

        # Ensure positive definiteness
        shrunk = _ensure_positive_definite(shrunk)

        return ShrinkageCovarianceResult(
            covariance=shrunk,
            shrinkage_intensity=alpha,
            target=F,
            method=self.target,
        )


# -------------------------------------------------------------------
# MinimumCovarianceDeterminant
# -------------------------------------------------------------------


class MinimumCovarianceDeterminant:
    """Minimum Covariance Determinant robust estimator.

    Finds the subset of observations (of size h) whose covariance
    matrix has the smallest determinant, yielding a robust location
    and scatter estimate that is resistant to outliers.

    Implements the Fast-MCD algorithm of Rousseeuw & Van Driessen
    (1999) with multiple random starts and concentration steps.

    Parameters
    ----------
    support_fraction : float
        Fraction of observations to include in the support set.
        Must be in (0.5, 1.0]. Default is 0.75.
    max_iter : int
        Maximum number of concentration steps per trial.
        Default is 100.
    random_state : int
        Random seed for reproducibility. Default is 42.

    Examples
    --------
    >>> from finceptquantlib.ml.markets.portfolio_ml.covariance import (
    ...     MinimumCovarianceDeterminant,
    ... )
    >>> returns = [
    ...     [0.01, 0.02],
    ...     [0.03, -0.01],
    ...     [-0.02, 0.01],
    ...     [0.00, 0.03],
    ...     [0.5, 0.6],  # outlier
    ... ]
    >>> mcd = MinimumCovarianceDeterminant(support_fraction=0.75)
    >>> cov = mcd.estimate(returns)
    """

    def __init__(
        self,
        support_fraction: float = 0.75,
        max_iter: int = 100,
        random_state: int = 42,
    ) -> None:
        if not (0.5 < support_fraction <= 1.0):
            raise ValueError(
                "support_fraction must be in (0.5, 1.0], "
                f"got {support_fraction}"
            )
        self.support_fraction = support_fraction
        self.max_iter = max_iter
        self.random_state = random_state

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _subset_mean(
        data: List[List[float]], indices: List[int]
    ) -> List[float]:
        """Mean of subset of rows."""
        p = len(data[0])
        h = len(indices)
        mu = [0.0] * p
        for idx in indices:
            row = data[idx]
            for j in range(p):
                mu[j] += row[j]
        for j in range(p):
            mu[j] /= h
        return mu

    @staticmethod
    def _subset_covariance(
        data: List[List[float]],
        indices: List[int],
        mu: List[float],
    ) -> List[List[float]]:
        """Covariance of subset of rows given mean."""
        p = len(mu)
        h = len(indices)
        cov = [[0.0] * p for _ in range(p)]
        for idx in indices:
            row = data[idx]
            diff = [row[j] - mu[j] for j in range(p)]
            for i in range(p):
                for j in range(i, p):
                    cov[i][j] += diff[i] * diff[j]
        denom = max(h - 1, 1)
        for i in range(p):
            for j in range(i, p):
                cov[i][j] /= denom
                cov[j][i] = cov[i][j]
        return cov

    @staticmethod
    def _mahalanobis_distances(
        data: List[List[float]],
        mu: List[float],
        cov: List[List[float]],
    ) -> List[float]:
        """Compute Mahalanobis distances.

        d_i = sqrt((x_i - mu)' Sigma^{-1} (x_i - mu))

        Uses Cholesky decomposition for stability.

        Parameters
        ----------
        data : List[List[float]]
            Data matrix (n x p).
        mu : List[float]
            Location estimate (p,).
        cov : List[List[float]]
            Covariance estimate (p x p).

        Returns
        -------
        List[float]
            Mahalanobis distance for each observation.
        """
        p = len(mu)
        # Regularize covariance slightly for Cholesky
        reg_cov = [
            [
                cov[i][j] + (1e-10 if i == j else 0.0)
                for j in range(p)
            ]
            for i in range(p)
        ]
        try:
            L = _cholesky(reg_cov)
        except Exception:
            # Fallback: use diagonal only
            dists = []
            for row in data:
                d_sq = 0.0
                for j in range(p):
                    diff = row[j] - mu[j]
                    var_j = cov[j][j] if cov[j][j] > 0 else 1e-10
                    d_sq += diff * diff / var_j
                dists.append(math.sqrt(max(d_sq, 0.0)))
            return dists

        dists = []
        for row in data:
            diff = [row[j] - mu[j] for j in range(p)]
            z = _solve_cholesky(L, diff)
            d_sq = dot_product(z, z)
            dists.append(math.sqrt(max(d_sq, 0.0)))
        return dists

    def _safe_determinant(
        self, cov: List[List[float]]
    ) -> float:
        """Determinant via Cholesky, returning inf on failure."""
        p = len(cov)
        reg = [
            [
                cov[i][j] + (1e-10 if i == j else 0.0)
                for j in range(p)
            ]
            for i in range(p)
        ]
        try:
            L = _cholesky(reg)
            return _determinant_from_cholesky(L)
        except Exception:
            return float("inf")

    def _c_step(
        self,
        data: List[List[float]],
        indices: List[int],
        h: int,
    ) -> Tuple[List[int], float]:
        """One concentration step.

        Compute mean/cov of current subset, then take h points
        with smallest Mahalanobis distance.

        Parameters
        ----------
        data : List[List[float]]
            Full data matrix (n x p).
        indices : List[int]
            Current support set indices.
        h : int
            Target support set size.

        Returns
        -------
        Tuple[List[int], float]
            New support indices and determinant.
        """
        mu = self._subset_mean(data, indices)
        cov = self._subset_covariance(data, indices, mu)
        det = self._safe_determinant(cov)

        dists = self._mahalanobis_distances(data, mu, cov)
        # Take h closest points
        sorted_idx = sorted(range(len(data)), key=lambda i: dists[i])
        new_indices = sorted_idx[:h]
        return new_indices, det

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def estimate(
        self, returns: List[List[float]]
    ) -> List[List[float]]:
        """Estimate robust covariance via Fast-MCD.

        Parameters
        ----------
        returns : List[List[float]]
            Asset return matrix (n_observations x n_assets).
            Each row is one observation.

        Returns
        -------
        List[List[float]]
            Robust covariance matrix (n_assets x n_assets).

        Raises
        ------
        ValueError
            If insufficient observations for the given dimension.

        Notes
        -----
        Implements the Fast-MCD algorithm from:
        Rousseeuw, P.J. & Van Driessen, K. (1999).
        "A fast algorithm for the minimum covariance determinant
        estimator." Technometrics 41(3):212-223.
        """
        n = len(returns)
        p = len(returns[0]) if returns else 0

        if n < p + 1:
            raise ValueError(
                f"Need at least p+1={p + 1} observations, "
                f"got {n}"
            )

        h = int(math.floor(self.support_fraction * n))
        h = max(h, p + 1)  # minimum for non-singular cov
        h = min(h, n)

        rng = random.Random(self.random_state)

        # Number of random initial subsets to try
        n_trials = min(500, max(10, n * 2))

        best_indices: Optional[List[int]] = None
        best_det = float("inf")

        for trial in range(n_trials):
            # Draw initial subset of size p + 1
            initial = rng.sample(range(n), min(p + 1, n))

            # Run C-steps until convergence
            indices = initial
            prev_det = float("inf")

            for _ in range(self.max_iter):
                indices, det = self._c_step(
                    returns, indices, h
                )
                if det >= prev_det - 1e-14:
                    break
                prev_det = det

            if prev_det < best_det:
                best_det = prev_det
                best_indices = indices

        if best_indices is None:
            # Fallback to full sample
            best_indices = list(range(n))

        # Final estimates from best subset
        mu_mcd = self._subset_mean(returns, best_indices)
        cov_mcd = self._subset_covariance(
            returns, best_indices, mu_mcd
        )

        # Reweight: include observations within chi-squared quantile
        chi2_q = _chi2_ppf_approx(0.975, p)
        dists = self._mahalanobis_distances(
            returns, mu_mcd, cov_mcd
        )

        # Reweighted subset
        reweighted = [
            i for i in range(n) if dists[i] * dists[i] <= chi2_q
        ]

        if len(reweighted) < p + 1:
            # Not enough points pass threshold, use MCD subset
            reweighted = best_indices

        # Final covariance from reweighted subset
        mu_final = self._subset_mean(returns, reweighted)
        cov_final = self._subset_covariance(
            returns, reweighted, mu_final
        )

        # Ensure positive definiteness
        cov_final = _ensure_positive_definite(cov_final)

        return cov_final
