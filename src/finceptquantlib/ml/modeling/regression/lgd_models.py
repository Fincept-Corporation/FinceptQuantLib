"""
Loss Given Default (LGD) Models

Beta regression and two-stage LGD models for IFRS 9 Expected
Credit Loss calculations. All implementations are pure Python.

The beta regression uses a logit link for the conditional mean
and estimates a precision parameter phi via maximum likelihood.
Downturn LGD is derived per regulatory guidance (EBA/GL/2017/16).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from finceptquantlib.ml._base import Estimator, FitResult, PredictResult
from finceptquantlib.ml._metrics import mean_squared_error, r_squared
from finceptquantlib.ml._matrix_utils import (
    add_intercept,
    dot_product,
    mat_vec_mul,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-10  # Clamp for lgd values near 0 or 1
_DOWNTURN_K = 1.5  # Regulatory guidance multiplier for std


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _logit(p: float) -> float:
    """Logit function: log(p / (1-p)).

    Parameters
    ----------
    p : float
        Probability in (0, 1).

    Returns
    -------
    float
        Logit value.
    """
    p = max(_EPS, min(1.0 - _EPS, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    """Sigmoid (inverse logit) function: 1 / (1 + exp(-x)).

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    float
        Value in (0, 1).
    """
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _log_beta_function(a: float, b: float) -> float:
    """Log of the Beta function: ln(B(a,b)).

    Parameters
    ----------
    a : float
        First shape parameter (positive).
    b : float
        Second shape parameter (positive).

    Returns
    -------
    float
        ln(B(a,b)).
    """
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _digamma(x: float) -> float:
    """Digamma function psi(x) = d/dx ln(Gamma(x)).

    Uses the asymptotic series expansion for x >= 8 and the
    recurrence relation psi(x) = psi(x+1) - 1/x for small x.

    Parameters
    ----------
    x : float
        Positive input value.

    Returns
    -------
    float
        Digamma value.
    """
    result = 0.0
    # Use recurrence to shift x >= 8
    while x < 8.0:
        result -= 1.0 / x
        x += 1.0
    # Asymptotic series for large x
    result += math.log(x) - 0.5 / x
    xx = 1.0 / (x * x)
    result -= xx * (
        1.0 / 12.0
        - xx * (
            1.0 / 120.0
            - xx * (1.0 / 252.0)
        )
    )
    return result


def _trigamma(x: float) -> float:
    """Trigamma function psi'(x) = d^2/dx^2 ln(Gamma(x)).

    Parameters
    ----------
    x : float
        Positive input value.

    Returns
    -------
    float
        Trigamma value.
    """
    result = 0.0
    while x < 8.0:
        result += 1.0 / (x * x)
        x += 1.0
    # Asymptotic series
    result += 1.0 / x + 0.5 / (x * x)
    xx = 1.0 / (x * x)
    result += xx * (
        1.0 / 6.0
        - xx * (
            1.0 / 30.0
            - xx * (1.0 / 42.0)
        )
    )
    return result


def _beta_log_likelihood(
    y: List[float],
    mu: List[float],
    phi: float,
) -> float:
    """Log-likelihood of the beta regression model.

    L = sum(log(f_beta(y_i; mu_i*phi, (1-mu_i)*phi)))

    where f_beta is the beta PDF with shape parameters
    a = mu*phi, b = (1-mu)*phi.

    Parameters
    ----------
    y : List[float]
        Observed LGD values in (0, 1).
    mu : List[float]
        Predicted mean values in (0, 1).
    phi : float
        Precision parameter (positive).

    Returns
    -------
    float
        Total log-likelihood.
    """
    ll = 0.0
    for yi, mui in zip(y, mu):
        a = mui * phi
        b = (1.0 - mui) * phi
        if a <= 0 or b <= 0:
            continue
        ll += (
            math.lgamma(a + b)
            - math.lgamma(a)
            - math.lgamma(b)
            + (a - 1.0) * math.log(max(yi, _EPS))
            + (b - 1.0) * math.log(max(1.0 - yi, _EPS))
        )
    return ll


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LGDModelResult(FitResult):
    """Result of fitting an LGD model.

    Attributes
    ----------
    coefficients : List[float]
        Estimated regression coefficients.
    intercept : float
        Intercept term.
    mean_lgd : float
        Mean predicted LGD across the training sample.
    downturn_lgd : float
        Downturn LGD estimate (mean + k * std).
    r_squared : float
        Pseudo R-squared of the model.
    """

    coefficients: List[float] = field(default_factory=list)
    intercept: float = 0.0
    mean_lgd: float = 0.0
    downturn_lgd: float = 0.0
    r_squared: float = 0.0


# ---------------------------------------------------------------------------
# Beta Regression LGD
# ---------------------------------------------------------------------------

class BetaRegressionLGD(Estimator):
    """Beta regression model for Loss Given Default.

    Uses a logit link function: logit(mu_i) = X_i * beta, with a
    precision parameter phi estimated via MLE. The beta distribution
    naturally models outcomes bounded in (0, 1).

    Downturn LGD is computed as mean_lgd + k * std(lgd), where k
    follows EBA regulatory guidance (default 1.5).

    Parameters
    ----------
    max_iter : int
        Maximum gradient ascent iterations. Default is 100.
    tol : float
        Convergence tolerance on log-likelihood change.
        Default is 1e-6.
    learning_rate : float
        Step size for gradient ascent. Default is 0.01.
    downturn_k : float
        Multiplier for standard deviation in downturn LGD.
        Default is 1.5 per EBA/GL/2017/16.

    Notes
    -----
    LGD values of exactly 0 or 1 are clamped to (eps, 1-eps)
    since the beta distribution has support on the open interval.
    """

    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-6,
        learning_rate: float = 0.01,
        downturn_k: float = _DOWNTURN_K,
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.downturn_k = downturn_k
        self._coefficients: List[float] = []
        self._intercept: float = 0.0
        self._phi: float = 1.0
        self._fitted: bool = False

    def _clamp_lgd(self, lgd: List[float]) -> List[float]:
        """Clamp LGD values to the open interval (eps, 1-eps).

        Parameters
        ----------
        lgd : List[float]
            Raw LGD values.

        Returns
        -------
        List[float]
            Clamped values strictly in (0, 1).
        """
        return [
            max(_EPS, min(1.0 - _EPS, v)) for v in lgd
        ]

    def fit(
        self, X: List[List[float]], y: List[float]
    ) -> LGDModelResult:
        """Fit beta regression LGD model via gradient ascent on MLE.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix (n_samples x n_features).
        y : List[float]
            LGD values in (0, 1).

        Returns
        -------
        LGDModelResult
            Model diagnostics including downturn LGD.

        Raises
        ------
        ValueError
            If inputs are invalid.
        """
        n = len(X)
        if n == 0:
            raise ValueError("X must not be empty")
        p = len(X[0])
        if len(y) != n:
            raise ValueError(
                "X and y must have the same number of samples"
            )

        # Clamp target values
        y_clamped = self._clamp_lgd(y)

        # Augmented design matrix with intercept column
        X_aug = add_intercept(X)
        p_aug = p + 1

        # Initialize parameters: beta from logit of mean(y)
        y_mean = sum(y_clamped) / n
        beta = [_logit(y_mean)] + [0.0] * p
        phi = 10.0  # Initial precision

        prev_ll = -math.inf
        converged = False
        iteration = 0

        for iteration in range(1, self.max_iter + 1):
            # Forward pass: compute mu = sigmoid(X_aug * beta)
            eta = mat_vec_mul(X_aug, beta)
            mu = [_sigmoid(e) for e in eta]
            mu = [max(_EPS, min(1.0 - _EPS, m)) for m in mu]

            ll = _beta_log_likelihood(y_clamped, mu, phi)

            if abs(ll - prev_ll) < self.tol and iteration > 1:
                converged = True
                break
            prev_ll = ll

            # Gradient w.r.t. beta:
            # dL/d_beta_j = sum_i [
            #   phi * (log(y_i/(1-y_i))
            #          - (digamma(mu_i*phi) - digamma((1-mu_i)*phi)))
            #   * mu_i * (1 - mu_i) * X_aug[i][j]
            # ]
            grad_beta = [0.0] * p_aug
            for i in range(n):
                a_i = mu[i] * phi
                b_i = (1.0 - mu[i]) * phi
                if a_i <= 0 or b_i <= 0:
                    continue
                score_mu = phi * (
                    math.log(max(y_clamped[i], _EPS))
                    - math.log(max(1.0 - y_clamped[i], _EPS))
                    - _digamma(a_i)
                    + _digamma(b_i)
                )
                d_mu_d_eta = mu[i] * (1.0 - mu[i])
                for j in range(p_aug):
                    grad_beta[j] += (
                        score_mu * d_mu_d_eta * X_aug[i][j]
                    )

            # Gradient w.r.t. phi:
            # dL/d_phi = sum_i [
            #   mu_i * (log(y_i) - digamma(mu_i*phi))
            #   + (1-mu_i) * (log(1-y_i) - digamma((1-mu_i)*phi))
            #   + digamma(phi)
            # ]
            grad_phi = 0.0
            for i in range(n):
                a_i = mu[i] * phi
                b_i = (1.0 - mu[i]) * phi
                if a_i <= 0 or b_i <= 0:
                    continue
                grad_phi += (
                    mu[i] * (
                        math.log(max(y_clamped[i], _EPS))
                        - _digamma(a_i)
                    )
                    + (1.0 - mu[i]) * (
                        math.log(
                            max(1.0 - y_clamped[i], _EPS)
                        )
                        - _digamma(b_i)
                    )
                    + _digamma(phi)
                )

            # Update parameters
            for j in range(p_aug):
                beta[j] += self.learning_rate * grad_beta[j]
            phi += self.learning_rate * grad_phi
            phi = max(0.1, phi)  # Ensure phi stays positive

        self._intercept = beta[0]
        self._coefficients = beta[1:]
        self._phi = phi
        self._fitted = True

        # Compute predictions
        eta_final = mat_vec_mul(X_aug, beta)
        mu_final = [
            max(_EPS, min(1.0 - _EPS, _sigmoid(e)))
            for e in eta_final
        ]

        mean_lgd = sum(mu_final) / n
        lgd_std = math.sqrt(
            sum((m - mean_lgd) ** 2 for m in mu_final) / n
        )
        downturn_lgd = min(
            1.0, mean_lgd + self.downturn_k * lgd_std
        )

        # Pseudo R-squared (McFadden)
        null_mu = [y_mean] * n
        ll_null = _beta_log_likelihood(
            y_clamped, null_mu, phi
        )
        ll_full = _beta_log_likelihood(
            y_clamped, mu_final, phi
        )
        if abs(ll_null) > _EPS:
            pseudo_r2 = 1.0 - ll_full / ll_null
        else:
            pseudo_r2 = 0.0

        mse = mean_squared_error(y_clamped, mu_final)

        return LGDModelResult(
            converged=converged,
            iterations=iteration,
            loss=mse,
            message=(
                "Beta regression converged"
                if converged
                else (
                    f"Beta regression did not converge in "
                    f"{self.max_iter} iterations"
                )
            ),
            coefficients=self._coefficients,
            intercept=self._intercept,
            mean_lgd=mean_lgd,
            downturn_lgd=downturn_lgd,
            r_squared=pseudo_r2,
        )

    def predict(
        self, X: List[List[float]]
    ) -> PredictResult:
        """Predict LGD values for new samples.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix (n_samples x n_features).

        Returns
        -------
        PredictResult
            Predicted LGD values in [0, 1].

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted")
        preds: List[float] = []
        for row in X:
            eta = (
                self._intercept
                + dot_product(row, self._coefficients)
            )
            mu = _sigmoid(eta)
            preds.append(max(0.0, min(1.0, mu)))
        return PredictResult(
            predictions=preds,
            metadata={"phi": self._phi},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary.

        Returns
        -------
        Dict[str, Any]
            JSON-serializable model parameters.
        """
        return {
            "type": "BetaRegressionLGD",
            "max_iter": self.max_iter,
            "tol": self.tol,
            "learning_rate": self.learning_rate,
            "downturn_k": self.downturn_k,
            "coefficients": self._coefficients,
            "intercept": self._intercept,
            "phi": self._phi,
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any]
    ) -> "BetaRegressionLGD":
        """Reconstruct model from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Serialized model parameters.

        Returns
        -------
        BetaRegressionLGD
            Reconstructed model.
        """
        model = cls(
            max_iter=data.get("max_iter", 100),
            tol=data.get("tol", 1e-6),
            learning_rate=data.get("learning_rate", 0.01),
            downturn_k=data.get("downturn_k", _DOWNTURN_K),
        )
        model._coefficients = data.get("coefficients", [])
        model._intercept = data.get("intercept", 0.0)
        model._phi = data.get("phi", 1.0)
        model._fitted = data.get("fitted", False)
        return model


# ---------------------------------------------------------------------------
# Two-Stage LGD Model
# ---------------------------------------------------------------------------

class TwoStageLGD(Estimator):
    """Two-stage LGD model combining recovery classification and
    beta regression.

    Stage 1: Logistic classifier predicts P(full recovery) where
    full recovery is defined as LGD == 0 (or < eps).

    Stage 2: Beta regression predicts LGD conditional on partial
    recovery (LGD > 0).

    Combined prediction:
        E[LGD] = P(partial) * E[LGD | partial]
               = (1 - P(full_recovery)) * beta_predicted_lgd

    This structure is standard in IFRS 9 frameworks and handles
    the point mass at zero that pure beta regression cannot
    accommodate.

    Notes
    -----
    The logistic classifier in Stage 1 uses gradient descent on
    the binary cross-entropy loss. Convergence criteria and
    iteration limits are shared with the beta regression stage.
    """

    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-6,
        learning_rate: float = 0.01,
        downturn_k: float = _DOWNTURN_K,
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.downturn_k = downturn_k
        # Stage 1: logistic coefficients for P(full recovery)
        self._stage1_coef: List[float] = []
        self._stage1_intercept: float = 0.0
        # Stage 2: beta regression
        self._stage2: Optional[BetaRegressionLGD] = None
        self._fitted: bool = False

    def _fit_logistic(
        self,
        X_aug: List[List[float]],
        y_binary: List[float],
    ) -> int:
        """Fit a logistic regression for Stage 1.

        Parameters
        ----------
        X_aug : List[List[float]]
            Augmented feature matrix with intercept column.
        y_binary : List[float]
            Binary labels: 1 = full recovery, 0 = partial.

        Returns
        -------
        int
            Number of iterations performed.
        """
        n = len(X_aug)
        p = len(X_aug[0])
        beta = [0.0] * p

        for iteration in range(1, self.max_iter + 1):
            # Predictions
            preds = [
                _sigmoid(dot_product(X_aug[i], beta))
                for i in range(n)
            ]

            # Gradient of cross-entropy: X' * (y - p) / n
            grad = [0.0] * p
            for i in range(n):
                err = y_binary[i] - preds[i]
                for j in range(p):
                    grad[j] += err * X_aug[i][j]

            max_grad = max(abs(g) for g in grad)
            for j in range(p):
                beta[j] += self.learning_rate * grad[j] / n

            if max_grad / n < self.tol:
                break

        self._stage1_intercept = beta[0]
        self._stage1_coef = beta[1:]
        return iteration

    def fit(
        self, X: List[List[float]], y: List[float]
    ) -> LGDModelResult:
        """Fit two-stage LGD model.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix (n_samples x n_features).
        y : List[float]
            LGD values in [0, 1].

        Returns
        -------
        LGDModelResult
            Model diagnostics.
        """
        n = len(X)
        if n == 0:
            raise ValueError("X must not be empty")
        if len(y) != n:
            raise ValueError(
                "X and y must have the same number of samples"
            )

        # Stage 1: classify full recovery (lgd ~= 0)
        recovery_threshold = _EPS * 10
        y_binary = [
            1.0 if lgd < recovery_threshold else 0.0
            for lgd in y
        ]

        X_aug = add_intercept(X)
        s1_iters = self._fit_logistic(X_aug, y_binary)

        # Stage 2: beta regression on partial recovery cases
        partial_idx = [
            i for i in range(n) if y[i] >= recovery_threshold
        ]

        if len(partial_idx) > 1:
            X_partial = [X[i] for i in partial_idx]
            y_partial = [y[i] for i in partial_idx]
            self._stage2 = BetaRegressionLGD(
                max_iter=self.max_iter,
                tol=self.tol,
                learning_rate=self.learning_rate,
                downturn_k=self.downturn_k,
            )
            self._stage2.fit(X_partial, y_partial)
        else:
            self._stage2 = None

        self._fitted = True

        # Combined predictions
        preds = self._predict_raw(X)
        mean_lgd = sum(preds) / n
        lgd_std = math.sqrt(
            sum((p - mean_lgd) ** 2 for p in preds) / n
        )
        downturn_lgd = min(
            1.0, mean_lgd + self.downturn_k * lgd_std
        )
        mse = mean_squared_error(y, preds)
        r2 = r_squared(y, preds)

        all_coefs = self._stage1_coef[:]
        if self._stage2 is not None:
            all_coefs.extend(self._stage2._coefficients)

        return LGDModelResult(
            converged=True,
            iterations=s1_iters,
            loss=mse,
            message="Two-stage LGD model fit completed",
            coefficients=all_coefs,
            intercept=self._stage1_intercept,
            mean_lgd=mean_lgd,
            downturn_lgd=downturn_lgd,
            r_squared=r2,
        )

    def _predict_raw(
        self, X: List[List[float]]
    ) -> List[float]:
        """Generate raw predictions combining both stages.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix.

        Returns
        -------
        List[float]
            Combined LGD predictions.
        """
        preds: List[float] = []
        for row in X:
            # Stage 1: P(full recovery)
            eta = (
                self._stage1_intercept
                + dot_product(row, self._stage1_coef)
            )
            p_full = _sigmoid(eta)
            p_partial = 1.0 - p_full

            # Stage 2: E[LGD | partial]
            if self._stage2 is not None:
                s2_pred = self._stage2.predict([row])
                lgd_partial = s2_pred.predictions[0]
            else:
                lgd_partial = 0.5  # Fallback

            combined = p_partial * lgd_partial
            preds.append(max(0.0, min(1.0, combined)))
        return preds

    def predict(
        self, X: List[List[float]]
    ) -> PredictResult:
        """Predict LGD values for new samples.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix (n_samples x n_features).

        Returns
        -------
        PredictResult
            Combined LGD predictions in [0, 1].
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted")
        return PredictResult(predictions=self._predict_raw(X))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary.

        Returns
        -------
        Dict[str, Any]
            JSON-serializable model parameters.
        """
        return {
            "type": "TwoStageLGD",
            "max_iter": self.max_iter,
            "tol": self.tol,
            "learning_rate": self.learning_rate,
            "downturn_k": self.downturn_k,
            "stage1_coef": self._stage1_coef,
            "stage1_intercept": self._stage1_intercept,
            "stage2": (
                self._stage2.to_dict()
                if self._stage2 is not None
                else None
            ),
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TwoStageLGD":
        """Reconstruct model from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Serialized model parameters.

        Returns
        -------
        TwoStageLGD
            Reconstructed model.
        """
        model = cls(
            max_iter=data.get("max_iter", 100),
            tol=data.get("tol", 1e-6),
            learning_rate=data.get("learning_rate", 0.01),
            downturn_k=data.get("downturn_k", _DOWNTURN_K),
        )
        model._stage1_coef = data.get("stage1_coef", [])
        model._stage1_intercept = data.get(
            "stage1_intercept", 0.0
        )
        s2_data = data.get("stage2")
        if s2_data is not None:
            model._stage2 = BetaRegressionLGD.from_dict(s2_data)
        else:
            model._stage2 = None
        model._fitted = data.get("fitted", False)
        return model
