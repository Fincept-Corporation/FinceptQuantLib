"""
Gaussian Process Yield Curve Fitting

Provides non-parametric Bayesian curve fitting using Gaussian process
regression. Supports RBF, Matern 3/2, and Matern 5/2 kernels with
optional hyperparameter optimization via marginal likelihood
maximization.

The GP approach provides:
- Smooth interpolation with uncertainty quantification
- Principled extrapolation with growing uncertainty
- Automatic regularization through the prior

This module does NOT inherit from Curve to avoid constructor
incompatibility, but provides the same method interface
(zero_rate, discount_factor, forward_rate).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Dict, List


@dataclass
class GPCurveResult:
    """Result of Gaussian process curve fitting.

    Attributes
    ----------
    mean : List[float]
        Posterior mean at fitted tenor points.
    std : List[float]
        Posterior standard deviation at fitted tenor points.
    log_marginal_likelihood : float
        Log marginal likelihood of the fitted model.
    kernel_params : Dict[str, float]
        Optimized kernel hyperparameters.
    tenors_fitted : List[float]
        Tenor points used for fitting.
    """

    mean: List[float]
    std: List[float]
    log_marginal_likelihood: float
    kernel_params: Dict[str, float]
    tenors_fitted: List[float]


class KernelType(Enum):
    """Supported GP kernel functions.

    Attributes
    ----------
    RBF : str
        Radial basis function (squared exponential) kernel.
    MATERN_32 : str
        Matern kernel with nu=3/2.
    MATERN_52 : str
        Matern kernel with nu=5/2.
    """

    RBF = "rbf"
    MATERN_32 = "matern_32"
    MATERN_52 = "matern_52"


def _cholesky(A: List[List[float]]) -> List[List[float]]:
    """Compute lower-triangular Cholesky factor L such that A = L L^T.

    Parameters
    ----------
    A : List[List[float]]
        Symmetric positive-definite matrix (n x n).

    Returns
    -------
    List[List[float]]
        Lower-triangular Cholesky factor (n x n).

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
                if val <= 0.0:
                    raise ValueError(
                        f"Matrix is not positive definite at index {i}. "
                        f"Diagonal element {A[i][i]}, sum {s}."
                    )
                L[i][j] = math.sqrt(val)
            else:
                if L[j][j] == 0.0:
                    raise ValueError(
                        "Zero diagonal in Cholesky factor."
                    )
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def _forward_solve(L: List[List[float]], b: List[float]) -> List[float]:
    """Solve L x = b where L is lower triangular.

    Parameters
    ----------
    L : List[List[float]]
        Lower-triangular matrix (n x n).
    b : List[float]
        Right-hand side vector (n,).

    Returns
    -------
    List[float]
        Solution vector x (n,).
    """
    n = len(b)
    x = [0.0] * n
    for i in range(n):
        s = sum(L[i][k] * x[k] for k in range(i))
        x[i] = (b[i] - s) / L[i][i]
    return x


def _backward_solve(L: List[List[float]], b: List[float]) -> List[float]:
    """Solve L^T x = b where L is lower triangular.

    Parameters
    ----------
    L : List[List[float]]
        Lower-triangular matrix (n x n). Solves with L^T.
    b : List[float]
        Right-hand side vector (n,).

    Returns
    -------
    List[float]
        Solution vector x (n,).
    """
    n = len(b)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(L[k][i] * x[k] for k in range(i + 1, n))
        x[i] = (b[i] - s) / L[i][i]
    return x


def _cholesky_solve(
    L: List[List[float]], b: List[float]
) -> List[float]:
    """Solve A x = b given Cholesky factor L of A.

    Solves L y = b (forward), then L^T x = y (backward).

    Parameters
    ----------
    L : List[List[float]]
        Lower-triangular Cholesky factor.
    b : List[float]
        Right-hand side vector.

    Returns
    -------
    List[float]
        Solution vector x = A^{-1} b.
    """
    y = _forward_solve(L, b)
    return _backward_solve(L, y)


def _log_det_from_cholesky(L: List[List[float]]) -> float:
    """Compute log determinant from Cholesky factor.

    log|A| = 2 * sum(log(L_ii))

    Parameters
    ----------
    L : List[List[float]]
        Lower-triangular Cholesky factor.

    Returns
    -------
    float
        Log determinant of the original matrix.
    """
    return 2.0 * sum(math.log(L[i][i]) for i in range(len(L)))


class GaussianProcessCurve:
    """Gaussian process regression for yield curve fitting.

    Provides a non-parametric Bayesian approach to yield curve
    interpolation with uncertainty quantification. Does NOT inherit
    from the Curve ABC to avoid constructor incompatibility, but
    provides the same method interface (zero_rate, discount_factor).

    Parameters
    ----------
    reference_date : date
        Curve valuation date.
    kernel : str
        Kernel function: 'rbf', 'matern_32', or 'matern_52'.
    length_scale : float
        Kernel length scale parameter.
    signal_variance : float
        Kernel signal variance (output scale).
    noise_variance : float
        Observation noise variance.
    optimize_hyperparams : bool
        If True, optimize hyperparameters by maximizing
        log marginal likelihood.
    day_count : str
        Day count convention identifier.

    Examples
    --------
    >>> from datetime import date
    >>> gp = GaussianProcessCurve(date(2024, 1, 1))
    >>> tenors = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    >>> rates = [0.03, 0.032, 0.035, 0.038, 0.04, 0.042]
    >>> result = gp.fit(tenors, rates)
    >>> gp.zero_rate(3.0)  # interpolated rate
    0.036291
    """

    def __init__(
        self,
        reference_date: date,
        kernel: str = "rbf",
        length_scale: float = 1.0,
        signal_variance: float = 1.0,
        noise_variance: float = 1e-6,
        optimize_hyperparams: bool = True,
        day_count: str = "ACT/365F",
    ) -> None:
        self.reference_date = reference_date
        self.day_count = day_count
        self.optimize_hyperparams = optimize_hyperparams

        kernel_lower = kernel.lower()
        if kernel_lower == "rbf":
            self._kernel_type = KernelType.RBF
        elif kernel_lower in ("matern_32", "matern32"):
            self._kernel_type = KernelType.MATERN_32
        elif kernel_lower in ("matern_52", "matern52"):
            self._kernel_type = KernelType.MATERN_52
        else:
            raise ValueError(
                f"Unknown kernel '{kernel}'. "
                f"Supported: 'rbf', 'matern_32', 'matern_52'."
            )

        self._length_scale = length_scale
        self._signal_variance = signal_variance
        self._noise_variance = noise_variance

        # Fitted state
        self._fitted = False
        self._train_tenors: List[float] = []
        self._train_rates: List[float] = []
        self._alpha: List[float] = []
        self._L: List[List[float]] = []

    def _kernel_func(
        self,
        x1: float,
        x2: float,
        length_scale: float,
        signal_variance: float,
    ) -> float:
        """Evaluate kernel function k(x1, x2).

        Parameters
        ----------
        x1 : float
            First input.
        x2 : float
            Second input.
        length_scale : float
            Length scale parameter.
        signal_variance : float
            Signal variance parameter.

        Returns
        -------
        float
            Kernel value.
        """
        r = abs(x1 - x2)

        if self._kernel_type == KernelType.RBF:
            return signal_variance * math.exp(
                -0.5 * (r / length_scale) ** 2
            )
        elif self._kernel_type == KernelType.MATERN_32:
            scaled = math.sqrt(3.0) * r / length_scale
            return signal_variance * (1.0 + scaled) * math.exp(-scaled)
        else:  # MATERN_52
            scaled = math.sqrt(5.0) * r / length_scale
            return signal_variance * (
                1.0 + scaled + scaled ** 2 / 3.0
            ) * math.exp(-scaled)

    def _build_kernel_matrix(
        self,
        X1: List[float],
        X2: List[float],
        length_scale: float,
        signal_variance: float,
    ) -> List[List[float]]:
        """Build kernel matrix K(X1, X2).

        Parameters
        ----------
        X1 : List[float]
            First set of inputs (n1,).
        X2 : List[float]
            Second set of inputs (n2,).
        length_scale : float
            Kernel length scale.
        signal_variance : float
            Kernel signal variance.

        Returns
        -------
        List[List[float]]
            Kernel matrix (n1 x n2).
        """
        n1 = len(X1)
        n2 = len(X2)
        K = [[0.0] * n2 for _ in range(n1)]
        for i in range(n1):
            for j in range(n2):
                K[i][j] = self._kernel_func(
                    X1[i], X2[j], length_scale, signal_variance
                )
        return K

    def _compute_log_marginal_likelihood(
        self,
        tenors: List[float],
        rates: List[float],
        length_scale: float,
        signal_variance: float,
        noise_variance: float,
    ) -> float:
        """Compute log marginal likelihood.

        log p(y|X, theta) = -0.5 * y^T K_y^{-1} y
                            - 0.5 * log|K_y|
                            - n/2 * log(2 pi)

        Parameters
        ----------
        tenors : List[float]
            Training tenor points.
        rates : List[float]
            Training rate values.
        length_scale : float
            Kernel length scale.
        signal_variance : float
            Kernel signal variance.
        noise_variance : float
            Observation noise variance.

        Returns
        -------
        float
            Log marginal likelihood. Returns -inf on failure.
        """
        n = len(tenors)
        K = self._build_kernel_matrix(
            tenors, tenors, length_scale, signal_variance
        )
        for i in range(n):
            K[i][i] += noise_variance

        try:
            L = _cholesky(K)
        except ValueError:
            return float("-inf")

        alpha = _cholesky_solve(L, rates)
        data_fit = -0.5 * sum(
            rates[i] * alpha[i] for i in range(n)
        )
        complexity = -0.5 * _log_det_from_cholesky(L)
        constant = -0.5 * n * math.log(2.0 * math.pi)

        return data_fit + complexity + constant

    def _optimize_hyperparams(
        self,
        tenors: List[float],
        rates: List[float],
    ) -> None:
        """Optimize kernel hyperparameters via grid search.

        Searches over log-spaced grids for length_scale,
        signal_variance, and noise_variance to maximize
        log marginal likelihood.

        Parameters
        ----------
        tenors : List[float]
            Training tenor points.
        rates : List[float]
            Training rate values.
        """
        tenor_range = max(tenors) - min(tenors) if len(tenors) > 1 else 1.0
        rate_var = (
            sum((r - sum(rates) / len(rates)) ** 2 for r in rates)
            / len(rates)
        ) if rates else 1.0

        ls_candidates = [
            tenor_range * f
            for f in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        ]
        sv_candidates = [
            rate_var * f for f in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        ]
        nv_candidates = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

        best_lml = float("-inf")
        best_ls = self._length_scale
        best_sv = self._signal_variance
        best_nv = self._noise_variance

        for ls in ls_candidates:
            for sv in sv_candidates:
                for nv in nv_candidates:
                    lml = self._compute_log_marginal_likelihood(
                        tenors, rates, ls, sv, nv
                    )
                    if lml > best_lml:
                        best_lml = lml
                        best_ls = ls
                        best_sv = sv
                        best_nv = nv

        self._length_scale = best_ls
        self._signal_variance = best_sv
        self._noise_variance = best_nv

    def fit(
        self, tenors: List[float], rates: List[float]
    ) -> GPCurveResult:
        """Fit the GP model to observed rates.

        Parameters
        ----------
        tenors : List[float]
            Observed tenor points in years.
        rates : List[float]
            Observed zero rates at each tenor.

        Returns
        -------
        GPCurveResult
            Fitting result with posterior mean, std, and
            hyperparameters.

        Raises
        ------
        ValueError
            If tenors and rates have different lengths or are empty.
        """
        if len(tenors) != len(rates):
            raise ValueError(
                f"tenors ({len(tenors)}) and rates ({len(rates)}) "
                f"must have the same length."
            )
        if len(tenors) == 0:
            raise ValueError("Cannot fit with empty data.")

        self._train_tenors = list(tenors)
        self._train_rates = list(rates)
        n = len(tenors)

        if self.optimize_hyperparams:
            self._optimize_hyperparams(tenors, rates)

        K = self._build_kernel_matrix(
            tenors, tenors,
            self._length_scale, self._signal_variance,
        )
        for i in range(n):
            K[i][i] += self._noise_variance

        self._L = _cholesky(K)
        self._alpha = _cholesky_solve(self._L, rates)

        lml = self._compute_log_marginal_likelihood(
            tenors, rates,
            self._length_scale,
            self._signal_variance,
            self._noise_variance,
        )

        # Compute posterior mean and std at training points
        post_mean: List[float] = []
        post_std: List[float] = []
        for t in tenors:
            mu = self._predict_mean(t)
            sigma = self.uncertainty(t)
            post_mean.append(mu)
            post_std.append(sigma)

        self._fitted = True

        return GPCurveResult(
            mean=post_mean,
            std=post_std,
            log_marginal_likelihood=lml,
            kernel_params={
                "length_scale": self._length_scale,
                "signal_variance": self._signal_variance,
                "noise_variance": self._noise_variance,
                "kernel": self._kernel_type.value,
            },
            tenors_fitted=list(tenors),
        )

    def _predict_mean(self, t: float) -> float:
        """Compute GP posterior mean at a single point.

        mu(t*) = k(t*, X)^T alpha

        Parameters
        ----------
        t : float
            Query point (tenor in years).

        Returns
        -------
        float
            Posterior mean rate.
        """
        k_star = [
            self._kernel_func(
                t, xi, self._length_scale, self._signal_variance
            )
            for xi in self._train_tenors
        ]
        return sum(
            k_star[i] * self._alpha[i]
            for i in range(len(self._alpha))
        )

    def _check_fitted(self) -> None:
        """Verify model has been fitted.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                "Model has not been fitted. Call fit() first."
            )

    def zero_rate(
        self, t: float, compounding: str = "continuous"
    ) -> float:
        """Get zero rate at a given maturity.

        Parameters
        ----------
        t : float
            Time in years.
        compounding : str
            Compounding convention. Currently only 'continuous'
            returns the GP prediction directly.

        Returns
        -------
        float
            Interpolated zero rate.
        """
        self._check_fitted()
        rate = self._predict_mean(t)

        if compounding == "continuous":
            return rate

        periods_map = {
            "annual": 1,
            "semi-annual": 2,
            "quarterly": 4,
            "monthly": 12,
        }
        m = periods_map.get(compounding, 1)
        return m * (math.exp(rate / m) - 1.0)

    def discount_factor(self, t: float) -> float:
        """Get discount factor at a given maturity.

        df(t) = exp(-r(t) * t) where r(t) is the zero rate.

        Parameters
        ----------
        t : float
            Time in years.

        Returns
        -------
        float
            Discount factor.
        """
        self._check_fitted()
        if t <= 0.0:
            return 1.0
        rate = self.zero_rate(t, compounding="continuous")
        return math.exp(-rate * t)

    def uncertainty(self, t: float) -> float:
        """Get posterior standard deviation at a given maturity.

        var(t*) = k(t*, t*) - k(t*, X)^T K_y^{-1} k(X, t*)

        Parameters
        ----------
        t : float
            Time in years.

        Returns
        -------
        float
            Posterior standard deviation (sqrt of variance).
        """
        if not self._alpha:
            raise RuntimeError(
                "Model has not been fitted. Call fit() first."
            )

        k_star = [
            self._kernel_func(
                t, xi, self._length_scale, self._signal_variance
            )
            for xi in self._train_tenors
        ]
        k_ss = self._kernel_func(
            t, t, self._length_scale, self._signal_variance
        )

        # v = L^{-1} k_star via forward substitution
        v = _forward_solve(self._L, k_star)
        var = k_ss - sum(vi * vi for vi in v)
        return math.sqrt(max(var, 0.0))

    def forward_rate(
        self, t: float, dt: float = 0.01
    ) -> float:
        """Compute instantaneous forward rate via finite difference.

        f(t) = -d/dt[ln(df(t))]

        Approximated using central differences.

        Parameters
        ----------
        t : float
            Time in years.
        dt : float
            Finite difference step size.

        Returns
        -------
        float
            Instantaneous forward rate.
        """
        self._check_fitted()
        if t <= dt:
            df0 = self.discount_factor(0.0)
            df1 = self.discount_factor(t + dt)
            return -math.log(df1 / df0) / (t + dt) if df0 > 0 else 0.0

        df_plus = self.discount_factor(t + dt)
        df_minus = self.discount_factor(t - dt)
        if df_minus <= 0.0:
            return 0.0
        return -math.log(df_plus / df_minus) / (2.0 * dt)
