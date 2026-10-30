"""
Logistic Regression for Credit Scoring

Production-quality logistic regression implementation with IRLS
(Iteratively Reweighted Least Squares) and gradient descent solvers.
Provides statistical inference (standard errors, p-values, AIC/BIC)
required for regulatory credit model documentation.

All computations are pure Python with no external numerical
dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from finceptquantlib.ml._base import (
    Classifier,
    FitResult,
    PredictResult,
)
from finceptquantlib.ml._matrix_utils import (
    add_intercept,
    mat_vec_mul,
    transpose,
)
from finceptquantlib.ml._metrics import log_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sigmoid(z: float) -> float:
    """Numerically stable sigmoid function.

    sigma(z) = 1 / (1 + exp(-z))

    Clamps input to [-500, 500] to prevent overflow.

    Parameters
    ----------
    z : float
        Input value.

    Returns
    -------
    float
        Sigmoid output in (0, 1).
    """
    z = max(-500.0, min(500.0, z))
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def _norm_cdf(x: float) -> float:
    """Approximation of the standard normal CDF.

    Uses the Abramowitz and Stegun approximation (formula 26.2.17)
    with maximum error < 7.5e-8.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    float
        P(Z <= x) for standard normal Z.
    """
    if x < -8.0:
        return 0.0
    if x > 8.0:
        return 1.0

    sign = 1
    if x < 0:
        sign = -1
        x = -x

    # Constants for Abramowitz & Stegun 26.2.17
    p = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429

    t = 1.0 / (1.0 + p * x)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t

    pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    cdf = 1.0 - pdf * (
        b1 * t + b2 * t2 + b3 * t3 + b4 * t4 + b5 * t5
    )

    if sign == -1:
        cdf = 1.0 - cdf

    return cdf


def _solve_linear_system(
    A: List[List[float]], b: List[float]
) -> List[float]:
    """Solve Ax = b using Gaussian elimination with partial pivoting.

    Parameters
    ----------
    A : List[List[float]]
        Square coefficient matrix (n x n).
    b : List[float]
        Right-hand side vector (n,).

    Returns
    -------
    List[float]
        Solution vector x (n,).

    Raises
    ------
    ValueError
        If the system is singular or near-singular.
    """
    n = len(A)
    # Augmented matrix
    aug = [A[i][:] + [b[i]] for i in range(n)]

    for col in range(n):
        # Partial pivoting: find max absolute value in column
        max_val = abs(aug[col][col])
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row

        if max_val < 1e-14:
            raise ValueError(
                f"Singular or near-singular matrix at column {col}"
            )

        # Swap rows
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
    """Invert a square matrix using Gaussian elimination.

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
    # Augment with identity
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

        # Scale pivot row
        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot

        # Eliminate all other rows
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(2 * n):
                aug[row][j] -= factor * aug[col][j]

    # Extract inverse
    return [aug[i][n:] for i in range(n)]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class LogisticRegressionResult(FitResult):
    """Result of logistic regression fitting.

    Extends FitResult with statistical inference quantities
    needed for credit model documentation and regulatory review.

    Attributes
    ----------
    coefficients : List[float]
        Fitted coefficients (excluding intercept).
    intercept : float
        Fitted intercept term.
    log_likelihood : float
        Maximized log-likelihood value.
    aic : float
        Akaike Information Criterion = -2*LL + 2*k.
    bic : float
        Bayesian Information Criterion = -2*LL + k*ln(n).
    p_values : List[float]
        Two-sided p-values for each coefficient (Wald test).
    standard_errors : List[float]
        Standard errors of coefficients from inverse Hessian.
    """

    coefficients: List[float] = field(default_factory=list)
    intercept: float = 0.0
    log_likelihood: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    p_values: List[float] = field(default_factory=list)
    standard_errors: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------


class LogisticRegression(Classifier):
    """Logistic Regression classifier for credit scoring.

    Supports IRLS (Newton-Raphson with Fisher scoring) and gradient
    descent optimization. Provides full statistical inference
    including standard errors, p-values, AIC, and BIC.

    Parameters
    ----------
    penalty : str
        Regularization type: ``"none"`` or ``"l2"``.
    C : float
        Inverse regularization strength. Smaller values specify
        stronger regularization. Only used when ``penalty="l2"``.
    max_iter : int
        Maximum number of optimization iterations.
    tol : float
        Convergence tolerance on the log-likelihood change.
    solver : str
        Optimization algorithm: ``"irls"`` (default) for
        Iteratively Reweighted Least Squares (Newton-Raphson),
        or ``"gradient_descent"`` as fallback.

    Examples
    --------
    >>> model = LogisticRegression(solver="irls")
    >>> result = model.fit(X_train, y_train)  # doctest: +SKIP
    >>> proba = model.predict_proba(X_test)  # doctest: +SKIP
    """

    def __init__(
        self,
        penalty: str = "none",
        C: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-6,
        solver: str = "irls",
    ) -> None:
        if penalty not in ("none", "l2"):
            raise ValueError(
                f"penalty must be 'none' or 'l2', got '{penalty}'"
            )
        if solver not in ("irls", "gradient_descent"):
            raise ValueError(
                f"solver must be 'irls' or 'gradient_descent', "
                f"got '{solver}'"
            )
        if C <= 0:
            raise ValueError("C must be positive")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")

        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver

        # Fitted parameters (set after fit)
        self._weights: Optional[List[float]] = None
        self._n_features: int = 0
        self._fit_result: Optional[LogisticRegressionResult] = None

    # -----------------------------------------------------------------------
    # Internal computation
    # -----------------------------------------------------------------------

    def _compute_probabilities(
        self, X_aug: List[List[float]], w: List[float]
    ) -> List[float]:
        """Compute predicted probabilities P(y=1|X).

        Parameters
        ----------
        X_aug : List[List[float]]
            Design matrix with intercept column.
        w : List[float]
            Weight vector (intercept + coefficients).

        Returns
        -------
        List[float]
            Predicted probabilities.
        """
        z = mat_vec_mul(X_aug, w)
        return [_sigmoid(zi) for zi in z]

    def _log_likelihood(
        self,
        y: List[float],
        p: List[float],
        w: List[float],
    ) -> float:
        """Compute log-likelihood with optional L2 penalty.

        Parameters
        ----------
        y : List[float]
            Binary targets.
        p : List[float]
            Predicted probabilities.
        w : List[float]
            Weight vector.

        Returns
        -------
        float
            Log-likelihood value.
        """
        eps = 1e-15
        ll = 0.0
        for yi, pi in zip(y, p):
            pi_c = max(eps, min(1.0 - eps, pi))
            ll += yi * math.log(pi_c) + (1.0 - yi) * math.log(
                1.0 - pi_c
            )

        if self.penalty == "l2":
            # Penalize all weights except intercept (index 0)
            l2_term = sum(wi * wi for wi in w[1:])
            ll -= l2_term / (2.0 * self.C)

        return ll

    def _fit_irls(
        self, X_aug: List[List[float]], y: List[float]
    ) -> Tuple[List[float], int, bool, float]:
        """Fit using IRLS (Newton-Raphson with Fisher scoring).

        w_{k+1} = w_k + (X'WX + lambda*I)^{-1} X'(y - p)

        where W = diag(p_i * (1 - p_i)) is the weight matrix.

        Parameters
        ----------
        X_aug : List[List[float]]
            Design matrix with intercept (n x (p+1)).
        y : List[float]
            Binary targets.

        Returns
        -------
        Tuple[List[float], int, bool, float]
            (weights, iterations, converged, final_ll).
        """
        n = len(X_aug)
        p_dim = len(X_aug[0])

        # Initialize weights to zero
        w = [0.0] * p_dim
        prev_ll = -1e30
        converged = False
        n_iter = 0

        for iteration in range(self.max_iter):
            n_iter = iteration + 1

            # Compute probabilities
            probs = self._compute_probabilities(X_aug, w)

            # Current log-likelihood
            ll = self._log_likelihood(y, probs, w)

            # Check convergence
            if abs(ll - prev_ll) < self.tol:
                converged = True
                break
            prev_ll = ll

            # Compute gradient: X'(y - p) - penalty term
            residuals = [yi - pi for yi, pi in zip(y, probs)]
            Xt = transpose(X_aug)
            gradient = mat_vec_mul(Xt, residuals)

            if self.penalty == "l2":
                lam = 1.0 / self.C
                for j in range(1, p_dim):
                    gradient[j] -= lam * w[j]

            # Compute Hessian: X'WX + penalty
            # W_diag = p_i * (1 - p_i), clamped for stability
            w_diag = [
                max(1e-10, pi * (1.0 - pi)) for pi in probs
            ]

            # X'WX: for each pair (j, k), sum_i X_ij * w_i * X_ik
            hessian = [[0.0] * p_dim for _ in range(p_dim)]
            for i in range(n):
                for j in range(p_dim):
                    xw = X_aug[i][j] * w_diag[i]
                    for k in range(j, p_dim):
                        val = xw * X_aug[i][k]
                        hessian[j][k] += val
                        if k != j:
                            hessian[k][j] += val

            # Add L2 penalty to diagonal (skip intercept)
            if self.penalty == "l2":
                lam = 1.0 / self.C
                for j in range(1, p_dim):
                    hessian[j][j] += lam

            # A tiny relative ridge keeps the system solvable when features
            # are collinear or the classes are separable
            jitter = 1e-10 * max(1.0, sum(hessian[j][j] for j in range(p_dim)) / p_dim)
            for j in range(p_dim):
                hessian[j][j] += jitter

            # Solve Hessian * delta = gradient
            try:
                delta = _solve_linear_system(hessian, gradient)
            except ValueError:
                # Hessian singular - fall back to small step
                lr = 0.01
                delta = [lr * g for g in gradient]

            # Step halving keeps the log-likelihood non-decreasing
            step = 1.0
            for _ in range(30):
                candidate = [wj + step * dj for wj, dj in zip(w, delta)]
                candidate_ll = self._log_likelihood(
                    y, self._compute_probabilities(X_aug, candidate), candidate
                )
                if candidate_ll >= ll:
                    break
                step *= 0.5
            w = candidate

        probs = self._compute_probabilities(X_aug, w)
        final_ll = self._log_likelihood(y, probs, w)

        return w, n_iter, converged, final_ll

    def _fit_gradient_descent(
        self, X_aug: List[List[float]], y: List[float]
    ) -> Tuple[List[float], int, bool, float]:
        """Fit using gradient descent with optional L2 penalty.

        Parameters
        ----------
        X_aug : List[List[float]]
            Design matrix with intercept (n x (p+1)).
        y : List[float]
            Binary targets.

        Returns
        -------
        Tuple[List[float], int, bool, float]
            (weights, iterations, converged, final_ll).
        """
        n = len(X_aug)
        p_dim = len(X_aug[0])
        lr = 1.0 / n  # initial learning rate

        w = [0.0] * p_dim
        prev_ll = -1e30
        converged = False
        n_iter = 0

        Xt = transpose(X_aug)

        for iteration in range(self.max_iter):
            n_iter = iteration + 1

            probs = self._compute_probabilities(X_aug, w)
            ll = self._log_likelihood(y, probs, w)

            if abs(ll - prev_ll) < self.tol:
                converged = True
                break
            prev_ll = ll

            # Gradient of log-likelihood
            residuals = [yi - pi for yi, pi in zip(y, probs)]
            gradient = mat_vec_mul(Xt, residuals)

            # L2 penalty gradient
            if self.penalty == "l2":
                lam = 1.0 / self.C
                for j in range(1, p_dim):
                    gradient[j] -= lam * w[j]

            # Update
            for j in range(p_dim):
                w[j] += lr * gradient[j]

        probs = self._compute_probabilities(X_aug, w)
        final_ll = self._log_likelihood(y, probs, w)

        return w, n_iter, converged, final_ll

    def _compute_standard_errors(
        self,
        X_aug: List[List[float]],
        w: List[float],
    ) -> List[float]:
        """Compute standard errors from inverse Fisher information.

        SE_j = sqrt(diag(H^{-1})_j) where H = X'WX.

        Parameters
        ----------
        X_aug : List[List[float]]
            Design matrix with intercept.
        w : List[float]
            Fitted weights.

        Returns
        -------
        List[float]
            Standard errors for all parameters
            (intercept + coefficients).
        """
        n = len(X_aug)
        p_dim = len(X_aug[0])
        probs = self._compute_probabilities(X_aug, w)
        w_diag = [max(1e-10, pi * (1.0 - pi)) for pi in probs]

        # Build X'WX (Fisher information matrix)
        fisher = [[0.0] * p_dim for _ in range(p_dim)]
        for i in range(n):
            for j in range(p_dim):
                xw = X_aug[i][j] * w_diag[i]
                for k in range(j, p_dim):
                    val = xw * X_aug[i][k]
                    fisher[j][k] += val
                    if k != j:
                        fisher[k][j] += val

        # Add L2 penalty
        if self.penalty == "l2":
            lam = 1.0 / self.C
            for j in range(1, p_dim):
                fisher[j][j] += lam

        # Invert to get variance-covariance matrix
        try:
            cov = _invert_matrix(fisher)
            se = [
                math.sqrt(max(0.0, cov[j][j]))
                for j in range(p_dim)
            ]
        except ValueError:
            se = [0.0] * p_dim

        return se

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def fit(
        self, X: List[List[float]], y: List[float]
    ) -> LogisticRegressionResult:
        """Fit logistic regression model.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix (n_samples x n_features).
        y : List[float]
            Binary target values (0 or 1).

        Returns
        -------
        LogisticRegressionResult
            Fitting result with coefficients, standard errors,
            p-values, AIC, and BIC.

        Raises
        ------
        ValueError
            If inputs are invalid.
        """
        n = len(X)
        if n == 0:
            raise ValueError("X must not be empty")
        if len(y) != n:
            raise ValueError("X and y must have same length")

        self._n_features = len(X[0])

        # Add intercept column
        X_aug = add_intercept(X)

        # Fit
        if self.solver == "irls":
            w, n_iter, converged, final_ll = self._fit_irls(
                X_aug, y
            )
        else:
            w, n_iter, converged, final_ll = (
                self._fit_gradient_descent(X_aug, y)
            )

        self._weights = w

        # Number of parameters
        k = len(w)

        # AIC and BIC
        aic = -2.0 * final_ll + 2.0 * k
        bic = -2.0 * final_ll + k * math.log(n)

        # Standard errors and p-values
        se_all = self._compute_standard_errors(X_aug, w)

        # z-statistics and two-sided p-values (Wald test)
        p_values_all: List[float] = []
        for j in range(len(w)):
            if se_all[j] > 1e-15:
                z_stat = w[j] / se_all[j]
                # Two-sided p-value
                p_val = 2.0 * (1.0 - _norm_cdf(abs(z_stat)))
            else:
                p_val = 1.0
            p_values_all.append(p_val)

        # Compute final loss (log loss metric)
        probs = self._compute_probabilities(X_aug, w)
        loss_val = log_loss(y, probs)

        # Message
        msg = "Converged" if converged else "Did not converge"

        result = LogisticRegressionResult(
            converged=converged,
            iterations=n_iter,
            loss=loss_val,
            message=msg,
            coefficients=w[1:],
            intercept=w[0],
            log_likelihood=final_ll,
            aic=aic,
            bic=bic,
            p_values=p_values_all[1:],
            standard_errors=se_all[1:],
        )

        self._fit_result = result
        return result

    def predict(
        self, X: List[List[float]]
    ) -> PredictResult:
        """Predict binary class labels.

        Uses threshold 0.5 on predicted probabilities.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix (n_samples x n_features).

        Returns
        -------
        PredictResult
            Predicted class labels (0.0 or 1.0).

        Raises
        ------
        RuntimeError
            If model has not been fitted.
        """
        if self._weights is None:
            raise RuntimeError("Model has not been fitted")

        proba = self.predict_proba(X)
        predictions = [
            1.0 if row[1] >= 0.5 else 0.0 for row in proba
        ]
        return PredictResult(predictions=predictions)

    def predict_proba(
        self, X: List[List[float]]
    ) -> List[List[float]]:
        """Predict class probabilities.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix (n_samples x n_features).

        Returns
        -------
        List[List[float]]
            Probability matrix (n_samples x 2) where each row
            is [P(y=0), P(y=1)].

        Raises
        ------
        RuntimeError
            If model has not been fitted.
        """
        if self._weights is None:
            raise RuntimeError("Model has not been fitted")

        X_aug = add_intercept(X)
        probs = self._compute_probabilities(X_aug, self._weights)
        return [[1.0 - p, p] for p in probs]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary.

        Returns
        -------
        Dict[str, Any]
            JSON-serializable model parameters including
            hyperparameters and fitted weights.
        """
        data: Dict[str, Any] = {
            "penalty": self.penalty,
            "C": self.C,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "solver": self.solver,
            "n_features": self._n_features,
        }
        if self._weights is not None:
            data["weights"] = self._weights
        if self._fit_result is not None:
            data["fit_result"] = {
                "converged": self._fit_result.converged,
                "iterations": self._fit_result.iterations,
                "loss": self._fit_result.loss,
                "coefficients": self._fit_result.coefficients,
                "intercept": self._fit_result.intercept,
                "log_likelihood": (
                    self._fit_result.log_likelihood
                ),
                "aic": self._fit_result.aic,
                "bic": self._fit_result.bic,
                "p_values": self._fit_result.p_values,
                "standard_errors": (
                    self._fit_result.standard_errors
                ),
            }
        return data

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any]
    ) -> "LogisticRegression":
        """Reconstruct model from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Serialized model parameters from ``to_dict()``.

        Returns
        -------
        LogisticRegression
            Reconstructed model instance.
        """
        model = cls(
            penalty=data.get("penalty", "none"),
            C=data.get("C", 1.0),
            max_iter=data.get("max_iter", 100),
            tol=data.get("tol", 1e-6),
            solver=data.get("solver", "irls"),
        )
        model._n_features = data.get("n_features", 0)
        if "weights" in data:
            model._weights = data["weights"]
        if "fit_result" in data:
            fr = data["fit_result"]
            model._fit_result = LogisticRegressionResult(
                converged=fr["converged"],
                iterations=fr["iterations"],
                loss=fr["loss"],
                coefficients=fr["coefficients"],
                intercept=fr["intercept"],
                log_likelihood=fr["log_likelihood"],
                aic=fr["aic"],
                bic=fr["bic"],
                p_values=fr["p_values"],
                standard_errors=fr["standard_errors"],
            )
        return model
