"""
Linear Regression Models

Ordinary Least Squares, Ridge, Lasso, and Elastic Net regression
with full statistical diagnostics. All implementations are pure
Python using the shared matrix utilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

from finceptquantlib.core.utilities.distributions import _regularized_incomplete_beta
from finceptquantlib.ml._base import Estimator, FitResult, PredictResult
from finceptquantlib.ml._matrix_utils import (
    add_intercept,
    dot_product,
    mat_mul,
    mat_vec_mul,
    transpose,
)


# ---------------------------------------------------------------------------
# Helper: Normal CDF approximation (Abramowitz & Stegun 26.2.17)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Approximate standard normal CDF using rational approximation.

    Parameters
    ----------
    x : float
        Quantile value.

    Returns
    -------
    float
        P(Z <= x) for Z ~ N(0,1).
    """
    if x < -8.0:
        return 0.0
    if x > 8.0:
        return 1.0
    sign = 1.0
    if x < 0:
        sign = -1.0
        x = -x
    t = 1.0 / (1.0 + 0.2316419 * x)
    b1, b2, b3, b4, b5 = (
        0.319381530,
        -0.356563782,
        1.781477937,
        -1.821255978,
        1.330274429,
    )
    poly = ((((b5 * t + b4) * t + b3) * t + b2) * t + b1) * t
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    cdf_pos = 1.0 - pdf * poly
    return 0.5 + sign * (cdf_pos - 0.5)


def _two_tailed_p_value(t_stat: float, df: int) -> float:
    """Approximate two-tailed p-value for a t-statistic.

    For df >= 30 uses normal approximation; for smaller df uses
    the Cornish-Fisher expansion for improved accuracy.

    Parameters
    ----------
    t_stat : float
        The t-statistic.
    df : int
        Degrees of freedom.

    Returns
    -------
    float
        Two-tailed p-value.
    """
    if df <= 0:
        return 1.0
    if df >= 30:
        # Normal approximation is adequate for large df
        return 2.0 * (1.0 - _norm_cdf(abs(t_stat)))
    # For small df, use a better approximation via the
    # regularized incomplete beta function relationship:
    # p = I_{df/(df+t^2)}(df/2, 1/2)
    # We approximate using continued fraction for the beta CDF.
    x = df / (df + t_stat * t_stat)
    p = _regularized_beta(x, df / 2.0, 0.5)
    return max(0.0, min(1.0, p))


def _regularized_beta(
    x: float, a: float, b: float, max_iter: int = 200
) -> float:
    """Regularized incomplete beta function I_x(a, b).

    Delegates to the shared continued-fraction implementation in
    :mod:`finceptquantlib.core.utilities.distributions`.

    Parameters
    ----------
    x : float
        Upper limit of integration in [0, 1].
    a : float
        First shape parameter (positive).
    b : float
        Second shape parameter (positive).
    max_iter : int
        Maximum iterations for continued fraction.

    Returns
    -------
    float
        I_x(a, b) value in [0, 1].
    """
    del max_iter  # retained for API compatibility
    return max(0.0, min(1.0, _regularized_incomplete_beta(a, b, x)))


def _f_cdf(f_val: float, d1: int, d2: int) -> float:
    """CDF of the F-distribution using the regularized beta function.

    Parameters
    ----------
    f_val : float
        F-statistic value.
    d1 : int
        Numerator degrees of freedom.
    d2 : int
        Denominator degrees of freedom.

    Returns
    -------
    float
        P(F <= f_val).
    """
    if f_val <= 0.0:
        return 0.0
    x = d1 * f_val / (d1 * f_val + d2)
    return _regularized_beta(x, d1 / 2.0, d2 / 2.0)


# ---------------------------------------------------------------------------
# Linear algebra helpers (Gaussian elimination with partial pivoting)
# ---------------------------------------------------------------------------

def _solve_linear_system(
    A: List[List[float]], b: List[float]
) -> List[float]:
    """Solve A * x = b via Gaussian elimination with partial pivoting.

    Parameters
    ----------
    A : List[List[float]]
        Square matrix (n x n).
    b : List[float]
        Right-hand side vector (n,).

    Returns
    -------
    List[float]
        Solution vector x (n,).

    Raises
    ------
    ValueError
        If the matrix is singular.
    """
    n = len(A)
    # Build augmented matrix
    aug = [A[i][:] + [b[i]] for i in range(n)]

    # Forward elimination with partial pivoting
    for col in range(n):
        # Find pivot
        max_val = abs(aug[col][col])
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row
        if max_val < 1e-14:
            raise ValueError(
                "Singular or near-singular matrix encountered "
                "during Gaussian elimination"
            )
        if max_row != col:
            aug[col], aug[max_row] = aug[max_row], aug[col]

        # Eliminate below
        pivot = aug[col][col]
        for row in range(col + 1, n):
            factor = aug[row][col] / pivot
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = aug[i][n]
        for j in range(i + 1, n):
            s -= aug[i][j] * x[j]
        x[i] = s / aug[i][i]
    return x


def _invert_matrix(A: List[List[float]]) -> List[List[float]]:
    """Invert a square matrix using Gauss-Jordan elimination.

    Parameters
    ----------
    A : List[List[float]]
        Square matrix (n x n).

    Returns
    -------
    List[List[float]]
        Inverse matrix (n x n).

    Raises
    ------
    ValueError
        If the matrix is singular.
    """
    n = len(A)
    # Build [A | I]
    aug = [
        A[i][:] + [1.0 if j == i else 0.0 for j in range(n)]
        for i in range(n)
    ]

    for col in range(n):
        # Partial pivoting
        max_val = abs(aug[col][col])
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row
        if max_val < 1e-14:
            raise ValueError(
                "Singular matrix cannot be inverted"
            )
        if max_row != col:
            aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(2 * n):
                aug[row][j] -= factor * aug[col][j]

    return [aug[i][n:] for i in range(n)]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LinearRegressionResult(FitResult):
    """Result of fitting a linear regression model.

    Attributes
    ----------
    coefficients : List[float]
        Estimated regression coefficients (excluding intercept).
    intercept : float
        Estimated intercept term.
    r_squared : float
        Coefficient of determination.
    adjusted_r_squared : float
        Adjusted R-squared for the number of predictors.
    residual_std : float
        Standard deviation of residuals.
    standard_errors : List[float]
        Standard errors of the coefficients (including intercept
        as the first element).
    t_statistics : List[float]
        t-statistics for each coefficient (including intercept).
    p_values : List[float]
        Two-tailed p-values for each coefficient (including intercept).
    f_statistic : float
        Overall F-statistic for model significance.
    f_p_value : float
        P-value of the F-statistic.
    """

    coefficients: List[float] = field(default_factory=list)
    intercept: float = 0.0
    r_squared: float = 0.0
    adjusted_r_squared: float = 0.0
    residual_std: float = 0.0
    standard_errors: List[float] = field(default_factory=list)
    t_statistics: List[float] = field(default_factory=list)
    p_values: List[float] = field(default_factory=list)
    f_statistic: float = 0.0
    f_p_value: float = 0.0


# ---------------------------------------------------------------------------
# OLS Regression
# ---------------------------------------------------------------------------

class OLSRegression(Estimator):
    """Ordinary Least Squares regression with full diagnostics.

    Solves the normal equations (X'X)b = X'y using Gaussian
    elimination with partial pivoting. Computes R-squared,
    adjusted R-squared, standard errors, t-statistics, p-values,
    and the F-statistic.

    Examples
    --------
    >>> model = OLSRegression()
    >>> result = model.fit([[1.0], [2.0], [3.0]], [2.0, 4.0, 6.0])
    >>> result.r_squared  # doctest: +SKIP
    1.0
    """

    def __init__(self) -> None:
        self._coefficients: List[float] = []
        self._intercept: float = 0.0
        self._fitted: bool = False

    def fit(
        self, X: List[List[float]], y: List[float]
    ) -> LinearRegressionResult:
        """Fit OLS regression model.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix (n_samples x n_features).
        y : List[float]
            Target values (n_samples,).

        Returns
        -------
        LinearRegressionResult
            Detailed regression diagnostics.

        Raises
        ------
        ValueError
            If X and y have incompatible dimensions or if the
            design matrix is singular.
        """
        n = len(X)
        if n == 0:
            raise ValueError("X must not be empty")
        p = len(X[0])  # number of features (excluding intercept)
        if len(y) != n:
            raise ValueError(
                "X and y must have the same number of samples"
            )

        # Add intercept column (first column = 1)
        X_aug = add_intercept(X)
        p_aug = p + 1  # includes intercept

        # Normal equations: (X'X) b = X'y
        Xt = transpose(X_aug)
        XtX = mat_mul(Xt, X_aug)
        Xty = mat_vec_mul(Xt, y)

        beta = _solve_linear_system(XtX, Xty)

        self._intercept = beta[0]
        self._coefficients = beta[1:]
        self._fitted = True

        # Predictions and residuals
        y_pred = mat_vec_mul(X_aug, beta)
        residuals = [y[i] - y_pred[i] for i in range(n)]
        ss_res = sum(r * r for r in residuals)
        y_mean = sum(y) / n
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)

        # R-squared and adjusted R-squared
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if n > p + 1:
            adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
        else:
            adj_r2 = 0.0

        # Residual standard deviation
        df_resid = n - p_aug
        if df_resid > 0:
            s2 = ss_res / df_resid
            resid_std = math.sqrt(s2)
        else:
            s2 = 0.0
            resid_std = 0.0

        # Standard errors: sqrt(diag(s^2 * (X'X)^{-1}))
        standard_errors: List[float] = []
        t_statistics: List[float] = []
        p_values: List[float] = []

        try:
            XtX_inv = _invert_matrix(XtX)
            for k in range(p_aug):
                se = math.sqrt(abs(s2 * XtX_inv[k][k]))
                standard_errors.append(se)
                if se > 1e-14:
                    t_stat = beta[k] / se
                else:
                    t_stat = 0.0
                t_statistics.append(t_stat)
                p_values.append(
                    _two_tailed_p_value(t_stat, df_resid)
                )
        except ValueError:
            standard_errors = [0.0] * p_aug
            t_statistics = [0.0] * p_aug
            p_values = [1.0] * p_aug

        # F-statistic: (SS_reg / p) / (SS_res / (n - p - 1))
        ss_reg = ss_tot - ss_res
        if p > 0 and df_resid > 0 and ss_res > 0:
            f_stat = (ss_reg / p) / (ss_res / df_resid)
            f_p = 1.0 - _f_cdf(f_stat, p, df_resid)
        else:
            f_stat = 0.0
            f_p = 1.0

        mse = ss_res / n if n > 0 else 0.0
        return LinearRegressionResult(
            converged=True,
            iterations=1,
            loss=mse,
            message="OLS fit completed",
            coefficients=self._coefficients,
            intercept=self._intercept,
            r_squared=r2,
            adjusted_r_squared=adj_r2,
            residual_std=resid_std,
            standard_errors=standard_errors,
            t_statistics=t_statistics,
            p_values=p_values,
            f_statistic=f_stat,
            f_p_value=f_p,
        )

    def predict(
        self, X: List[List[float]]
    ) -> PredictResult:
        """Predict using the fitted OLS model.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix (n_samples x n_features).

        Returns
        -------
        PredictResult
            Predicted values.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted")
        preds = [
            self._intercept + dot_product(row, self._coefficients)
            for row in X
        ]
        return PredictResult(predictions=preds)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary.

        Returns
        -------
        Dict[str, Any]
            JSON-serializable model parameters.
        """
        return {
            "type": "OLSRegression",
            "coefficients": self._coefficients,
            "intercept": self._intercept,
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OLSRegression":
        """Reconstruct model from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Serialized model parameters.

        Returns
        -------
        OLSRegression
            Reconstructed model.
        """
        model = cls()
        model._coefficients = data.get("coefficients", [])
        model._intercept = data.get("intercept", 0.0)
        model._fitted = data.get("fitted", False)
        return model


# ---------------------------------------------------------------------------
# Ridge Regression
# ---------------------------------------------------------------------------

class RidgeRegression(Estimator):
    """Ridge (L2-penalised) regression.

    Solves (X'X + alpha * I) b = X'y using Gaussian elimination.
    The intercept column is **not** penalised.

    Parameters
    ----------
    alpha : float
        Regularisation strength (non-negative). Default is 1.0.

    Examples
    --------
    >>> model = RidgeRegression(alpha=0.1)
    >>> result = model.fit([[1.0], [2.0], [3.0]], [2.0, 4.0, 6.0])
    """

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha
        self._coefficients: List[float] = []
        self._intercept: float = 0.0
        self._fitted: bool = False

    def fit(
        self, X: List[List[float]], y: List[float]
    ) -> LinearRegressionResult:
        """Fit Ridge regression model.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix (n_samples x n_features).
        y : List[float]
            Target values (n_samples,).

        Returns
        -------
        LinearRegressionResult
            Regression diagnostics.
        """
        n = len(X)
        if n == 0:
            raise ValueError("X must not be empty")
        p = len(X[0])
        if len(y) != n:
            raise ValueError(
                "X and y must have the same number of samples"
            )

        X_aug = add_intercept(X)
        p_aug = p + 1

        Xt = transpose(X_aug)
        XtX = mat_mul(Xt, X_aug)
        Xty = mat_vec_mul(Xt, y)

        # Add penalty: alpha * I but do not penalise the intercept
        for k in range(1, p_aug):
            XtX[k][k] += self.alpha

        beta = _solve_linear_system(XtX, Xty)

        self._intercept = beta[0]
        self._coefficients = beta[1:]
        self._fitted = True

        # Compute diagnostics
        y_pred = mat_vec_mul(X_aug, beta)
        residuals = [y[i] - y_pred[i] for i in range(n)]
        ss_res = sum(r * r for r in residuals)
        y_mean = sum(y) / n
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)

        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if n > p + 1:
            adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
        else:
            adj_r2 = 0.0

        df_resid = n - p_aug
        s2 = ss_res / df_resid if df_resid > 0 else 0.0
        resid_std = math.sqrt(s2) if s2 > 0 else 0.0

        # Standard errors from (X'X + alpha*I)^{-1}
        standard_errors: List[float] = []
        t_statistics: List[float] = []
        p_values_list: List[float] = []
        try:
            XtX_inv = _invert_matrix(XtX)
            for k in range(p_aug):
                se = math.sqrt(abs(s2 * XtX_inv[k][k]))
                standard_errors.append(se)
                if se > 1e-14:
                    t_stat = beta[k] / se
                else:
                    t_stat = 0.0
                t_statistics.append(t_stat)
                p_values_list.append(
                    _two_tailed_p_value(t_stat, df_resid)
                )
        except ValueError:
            standard_errors = [0.0] * p_aug
            t_statistics = [0.0] * p_aug
            p_values_list = [1.0] * p_aug

        ss_reg = ss_tot - ss_res
        if p > 0 and df_resid > 0 and ss_res > 0:
            f_stat = (ss_reg / p) / (ss_res / df_resid)
            f_p = 1.0 - _f_cdf(f_stat, p, df_resid)
        else:
            f_stat = 0.0
            f_p = 1.0

        mse = ss_res / n if n > 0 else 0.0
        return LinearRegressionResult(
            converged=True,
            iterations=1,
            loss=mse,
            message=f"Ridge fit completed (alpha={self.alpha})",
            coefficients=self._coefficients,
            intercept=self._intercept,
            r_squared=r2,
            adjusted_r_squared=adj_r2,
            residual_std=resid_std,
            standard_errors=standard_errors,
            t_statistics=t_statistics,
            p_values=p_values_list,
            f_statistic=f_stat,
            f_p_value=f_p,
        )

    def predict(
        self, X: List[List[float]]
    ) -> PredictResult:
        """Predict using the fitted Ridge model.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix (n_samples x n_features).

        Returns
        -------
        PredictResult
            Predicted values.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted")
        preds = [
            self._intercept + dot_product(row, self._coefficients)
            for row in X
        ]
        return PredictResult(predictions=preds)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary.

        Returns
        -------
        Dict[str, Any]
            JSON-serializable model parameters.
        """
        return {
            "type": "RidgeRegression",
            "alpha": self.alpha,
            "coefficients": self._coefficients,
            "intercept": self._intercept,
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RidgeRegression":
        """Reconstruct model from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Serialized model parameters.

        Returns
        -------
        RidgeRegression
            Reconstructed model.
        """
        model = cls(alpha=data.get("alpha", 1.0))
        model._coefficients = data.get("coefficients", [])
        model._intercept = data.get("intercept", 0.0)
        model._fitted = data.get("fitted", False)
        return model


# ---------------------------------------------------------------------------
# Lasso Regression (Coordinate Descent)
# ---------------------------------------------------------------------------

class LassoRegression(Estimator):
    """Lasso (L1-penalised) regression via coordinate descent.

    Minimises (1/2n)||y - Xb||^2 + alpha * ||b||_1.
    The intercept is **not** penalised and is updated
    separately.

    Parameters
    ----------
    alpha : float
        Regularisation strength. Default is 1.0.
    max_iter : int
        Maximum coordinate descent iterations. Default is 1000.
    tol : float
        Convergence tolerance on max absolute coefficient change.
        Default is 1e-4.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self._coefficients: List[float] = []
        self._intercept: float = 0.0
        self._fitted: bool = False

    @staticmethod
    def _soft_threshold(z: float, gamma: float) -> float:
        """Soft thresholding operator S(z, gamma).

        Parameters
        ----------
        z : float
            Input value.
        gamma : float
            Threshold (non-negative).

        Returns
        -------
        float
            sign(z) * max(|z| - gamma, 0).
        """
        if z > gamma:
            return z - gamma
        elif z < -gamma:
            return z + gamma
        return 0.0

    def fit(
        self, X: List[List[float]], y: List[float]
    ) -> LinearRegressionResult:
        """Fit Lasso regression via coordinate descent.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix (n_samples x n_features).
        y : List[float]
            Target values (n_samples,).

        Returns
        -------
        LinearRegressionResult
            Regression diagnostics.
        """
        n = len(X)
        if n == 0:
            raise ValueError("X must not be empty")
        p = len(X[0])
        if len(y) != n:
            raise ValueError(
                "X and y must have the same number of samples"
            )

        # Pre-compute X_j' * X_j for each feature
        col_norms_sq: List[float] = []
        for j in range(p):
            norm_sq = sum(X[i][j] * X[i][j] for i in range(n))
            col_norms_sq.append(norm_sq)

        # Initialise coefficients
        beta = [0.0] * p
        intercept = sum(y) / n
        residual = [y[i] - intercept for i in range(n)]

        converged = False
        iteration = 0
        for iteration in range(1, self.max_iter + 1):
            max_change = 0.0
            for j in range(p):
                # Add back current beta_j contribution
                for i in range(n):
                    residual[i] += X[i][j] * beta[j]

                # Compute rho_j = X_j' * residual
                rho_j = sum(
                    X[i][j] * residual[i] for i in range(n)
                )

                # Update beta_j with soft thresholding
                if col_norms_sq[j] > 1e-14:
                    new_beta_j = self._soft_threshold(
                        rho_j, n * self.alpha
                    ) / col_norms_sq[j]
                else:
                    new_beta_j = 0.0

                change = abs(new_beta_j - beta[j])
                if change > max_change:
                    max_change = change
                beta[j] = new_beta_j

                # Remove updated beta_j contribution from residual
                for i in range(n):
                    residual[i] -= X[i][j] * beta[j]

            # Update intercept
            mean_residual = sum(residual) / n
            intercept += mean_residual
            for i in range(n):
                residual[i] -= mean_residual

            if max_change < self.tol:
                converged = True
                break

        self._coefficients = beta
        self._intercept = intercept
        self._fitted = True

        # Compute diagnostics
        y_pred = [
            intercept + dot_product(X[i], beta)
            for i in range(n)
        ]
        ss_res = sum(
            (y[i] - y_pred[i]) ** 2 for i in range(n)
        )
        y_mean = sum(y) / n
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        nnz = sum(1 for b in beta if abs(b) > 1e-14)
        if n > nnz + 1:
            adj_r2 = (
                1.0 - (1.0 - r2) * (n - 1) / (n - nnz - 1)
            )
        else:
            adj_r2 = 0.0
        df_resid = max(n - nnz - 1, 1)
        s2 = ss_res / df_resid
        resid_std = math.sqrt(s2) if s2 > 0 else 0.0

        mse = ss_res / n
        return LinearRegressionResult(
            converged=converged,
            iterations=iteration,
            loss=mse,
            message=(
                "Lasso converged"
                if converged
                else f"Lasso did not converge in {self.max_iter} iters"
            ),
            coefficients=self._coefficients,
            intercept=self._intercept,
            r_squared=r2,
            adjusted_r_squared=adj_r2,
            residual_std=resid_std,
            standard_errors=[],
            t_statistics=[],
            p_values=[],
            f_statistic=0.0,
            f_p_value=1.0,
        )

    def predict(
        self, X: List[List[float]]
    ) -> PredictResult:
        """Predict using the fitted Lasso model.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix (n_samples x n_features).

        Returns
        -------
        PredictResult
            Predicted values.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted")
        preds = [
            self._intercept + dot_product(row, self._coefficients)
            for row in X
        ]
        return PredictResult(predictions=preds)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary.

        Returns
        -------
        Dict[str, Any]
            JSON-serializable model parameters.
        """
        return {
            "type": "LassoRegression",
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "coefficients": self._coefficients,
            "intercept": self._intercept,
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LassoRegression":
        """Reconstruct model from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Serialized model parameters.

        Returns
        -------
        LassoRegression
            Reconstructed model.
        """
        model = cls(
            alpha=data.get("alpha", 1.0),
            max_iter=data.get("max_iter", 1000),
            tol=data.get("tol", 1e-4),
        )
        model._coefficients = data.get("coefficients", [])
        model._intercept = data.get("intercept", 0.0)
        model._fitted = data.get("fitted", False)
        return model


# ---------------------------------------------------------------------------
# Elastic Net (Coordinate Descent)
# ---------------------------------------------------------------------------

class ElasticNet(Estimator):
    """Elastic Net regression via coordinate descent.

    Combines L1 and L2 penalties:
    (1/2n)||y - Xb||^2 + alpha*l1_ratio*||b||_1
        + 0.5*alpha*(1-l1_ratio)*||b||^2

    Parameters
    ----------
    alpha : float
        Overall regularisation strength. Default is 1.0.
    l1_ratio : float
        Mix between L1 (1.0) and L2 (0.0). Default is 0.5.
    max_iter : int
        Maximum coordinate descent iterations. Default is 1000.
    tol : float
        Convergence tolerance. Default is 1e-4.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError("l1_ratio must be in [0, 1]")
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self._coefficients: List[float] = []
        self._intercept: float = 0.0
        self._fitted: bool = False

    @staticmethod
    def _soft_threshold(z: float, gamma: float) -> float:
        """Soft thresholding operator S(z, gamma).

        Parameters
        ----------
        z : float
            Input value.
        gamma : float
            Threshold.

        Returns
        -------
        float
            sign(z) * max(|z| - gamma, 0).
        """
        if z > gamma:
            return z - gamma
        elif z < -gamma:
            return z + gamma
        return 0.0

    def fit(
        self, X: List[List[float]], y: List[float]
    ) -> LinearRegressionResult:
        """Fit Elastic Net via coordinate descent.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix (n_samples x n_features).
        y : List[float]
            Target values (n_samples,).

        Returns
        -------
        LinearRegressionResult
            Regression diagnostics.
        """
        n = len(X)
        if n == 0:
            raise ValueError("X must not be empty")
        p = len(X[0])
        if len(y) != n:
            raise ValueError(
                "X and y must have the same number of samples"
            )

        l1_penalty = n * self.alpha * self.l1_ratio
        l2_penalty = n * self.alpha * (1.0 - self.l1_ratio)

        col_norms_sq: List[float] = []
        for j in range(p):
            norm_sq = sum(X[i][j] * X[i][j] for i in range(n))
            col_norms_sq.append(norm_sq)

        beta = [0.0] * p
        intercept = sum(y) / n
        residual = [y[i] - intercept for i in range(n)]

        converged = False
        iteration = 0
        for iteration in range(1, self.max_iter + 1):
            max_change = 0.0
            for j in range(p):
                for i in range(n):
                    residual[i] += X[i][j] * beta[j]

                rho_j = sum(
                    X[i][j] * residual[i] for i in range(n)
                )

                denom = col_norms_sq[j] + l2_penalty
                if denom > 1e-14:
                    new_beta_j = self._soft_threshold(
                        rho_j, l1_penalty
                    ) / denom
                else:
                    new_beta_j = 0.0

                change = abs(new_beta_j - beta[j])
                if change > max_change:
                    max_change = change
                beta[j] = new_beta_j

                for i in range(n):
                    residual[i] -= X[i][j] * beta[j]

            # Update intercept
            mean_residual = sum(residual) / n
            intercept += mean_residual
            for i in range(n):
                residual[i] -= mean_residual

            if max_change < self.tol:
                converged = True
                break

        self._coefficients = beta
        self._intercept = intercept
        self._fitted = True

        y_pred = [
            intercept + dot_product(X[i], beta)
            for i in range(n)
        ]
        ss_res = sum(
            (y[i] - y_pred[i]) ** 2 for i in range(n)
        )
        y_mean = sum(y) / n
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        nnz = sum(1 for b in beta if abs(b) > 1e-14)
        if n > nnz + 1:
            adj_r2 = (
                1.0 - (1.0 - r2) * (n - 1) / (n - nnz - 1)
            )
        else:
            adj_r2 = 0.0
        df_resid = max(n - nnz - 1, 1)
        s2 = ss_res / df_resid
        resid_std = math.sqrt(s2) if s2 > 0 else 0.0
        mse = ss_res / n

        return LinearRegressionResult(
            converged=converged,
            iterations=iteration,
            loss=mse,
            message=(
                "ElasticNet converged"
                if converged
                else (
                    f"ElasticNet did not converge in "
                    f"{self.max_iter} iterations"
                )
            ),
            coefficients=self._coefficients,
            intercept=self._intercept,
            r_squared=r2,
            adjusted_r_squared=adj_r2,
            residual_std=resid_std,
            standard_errors=[],
            t_statistics=[],
            p_values=[],
            f_statistic=0.0,
            f_p_value=1.0,
        )

    def predict(
        self, X: List[List[float]]
    ) -> PredictResult:
        """Predict using the fitted Elastic Net model.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix (n_samples x n_features).

        Returns
        -------
        PredictResult
            Predicted values.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted")
        preds = [
            self._intercept + dot_product(row, self._coefficients)
            for row in X
        ]
        return PredictResult(predictions=preds)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary.

        Returns
        -------
        Dict[str, Any]
            JSON-serializable model parameters.
        """
        return {
            "type": "ElasticNet",
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "coefficients": self._coefficients,
            "intercept": self._intercept,
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ElasticNet":
        """Reconstruct model from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Serialized model parameters.

        Returns
        -------
        ElasticNet
            Reconstructed model.
        """
        model = cls(
            alpha=data.get("alpha", 1.0),
            l1_ratio=data.get("l1_ratio", 0.5),
            max_iter=data.get("max_iter", 1000),
            tol=data.get("tol", 1e-4),
        )
        model._coefficients = data.get("coefficients", [])
        model._intercept = data.get("intercept", 0.0)
        model._fitted = data.get("fitted", False)
        return model
