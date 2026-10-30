"""
ML-Augmented GARCH Model

Implements a GARCH(p,q) model with optional external feature
augmentation for conditional volatility modelling. Parameter
estimation uses quasi-maximum likelihood with stationarity and
positivity constraints. All computations are pure Python.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from finceptquantlib.ml._base import FitResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MIN_VARIANCE: float = 1e-10
_LOG_2PI: float = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MLGARCHResult(FitResult):
    """Result of ML-augmented GARCH fitting.

    Attributes
    ----------
    garch_params : Dict[str, float]
        Estimated GARCH parameters including ``omega``,
        ``alpha_1`` ... ``alpha_q``, ``beta_1`` ... ``beta_p``.
    feature_coefficients : List[float]
        Coefficients for external features (gamma_k). Empty if no
        features provided.
    conditional_volatility : List[float]
        Estimated conditional standard deviation at each time step.
    standardized_residuals : List[float]
        Returns divided by conditional standard deviation.
    """

    garch_params: Dict[str, float] = field(default_factory=dict)
    feature_coefficients: List[float] = field(default_factory=list)
    conditional_volatility: List[float] = field(default_factory=list)
    standardized_residuals: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MLGARCHHybrid
# ---------------------------------------------------------------------------

class MLGARCHHybrid:
    """ML-augmented GARCH(p,q) model.

    Standard GARCH variance equation:

        sigma^2_t = omega
                    + sum_{i=1}^{q} alpha_i * eps^2_{t-i}
                    + sum_{j=1}^{p} beta_j  * sigma^2_{t-j}

    With optional external feature augmentation:

        sigma^2_t += sum_{k} gamma_k * f_{t-1, k}

    Features can include VIX levels, trading volume, cross-asset
    volatility, or any external signal available at time t-1.

    Parameters
    ----------
    p : int
        GARCH order (number of lagged variance terms).
    q : int
        ARCH order (number of lagged squared residual terms).
    max_iter : int
        Maximum iterations for parameter optimisation.
    tol : float
        Convergence tolerance on log-likelihood change.

    Examples
    --------
    >>> model = MLGARCHHybrid(p=1, q=1)
    >>> result = model.fit([0.01, -0.02, 0.015, -0.03, 0.005])
    >>> len(result.conditional_volatility) == 5
    True
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> None:
        if p < 1:
            raise ValueError("p must be >= 1")
        if q < 1:
            raise ValueError("q must be >= 1")
        self.p = p
        self.q = q
        self.max_iter = max_iter
        self.tol = tol

        # Fitted parameters
        self._omega: float = 0.0
        self._alphas: List[float] = []
        self._betas: List[float] = []
        self._gammas: List[float] = []
        self._fitted: bool = False
        self._unconditional_var: float = 1e-4

    # ------------------------------------------------------------------
    # Log-likelihood computation
    # ------------------------------------------------------------------

    def _compute_variance_series(
        self,
        returns: List[float],
        omega: float,
        alphas: List[float],
        betas: List[float],
        gammas: List[float],
        features: Optional[List[List[float]]],
    ) -> List[float]:
        """Compute conditional variance series.

        Parameters
        ----------
        returns : List[float]
            Return series of length T.
        omega : float
            Constant term.
        alphas : List[float]
            ARCH coefficients (length q).
        betas : List[float]
            GARCH coefficients (length p).
        gammas : List[float]
            Feature coefficients (length n_features).
        features : Optional[List[List[float]]]
            External features, shape T x n_features. Features at
            index t are lagged (represent info at t-1).

        Returns
        -------
        List[float]
            Conditional variance at each time step.
        """
        T = len(returns)
        sigma2 = [0.0] * T

        # Initialise with unconditional variance estimate
        sample_var = sum(r * r for r in returns) / T
        sample_var = max(sample_var, _MIN_VARIANCE)

        for t in range(T):
            var_t = omega

            # ARCH terms: alpha_i * eps^2_{t-i}
            for i, a in enumerate(alphas):
                lag = t - (i + 1)
                if lag >= 0:
                    var_t += a * returns[lag] ** 2
                else:
                    var_t += a * sample_var

            # GARCH terms: beta_j * sigma^2_{t-j}
            for j, b in enumerate(betas):
                lag = t - (j + 1)
                if lag >= 0:
                    var_t += b * sigma2[lag]
                else:
                    var_t += b * sample_var

            # Feature augmentation: gamma_k * f_{t-1, k}
            if features is not None and gammas:
                feat_idx = t - 1
                if feat_idx >= 0:
                    for k, g in enumerate(gammas):
                        if k < len(features[feat_idx]):
                            var_t += g * abs(features[feat_idx][k])

            sigma2[t] = max(var_t, _MIN_VARIANCE)

        return sigma2

    def _log_likelihood(
        self,
        returns: List[float],
        sigma2: List[float],
    ) -> float:
        """Gaussian quasi-log-likelihood.

        Parameters
        ----------
        returns : List[float]
            Return series.
        sigma2 : List[float]
            Conditional variance series.

        Returns
        -------
        float
            Sum of log-likelihood contributions.
        """
        ll = 0.0
        for t in range(len(returns)):
            s2 = max(sigma2[t], _MIN_VARIANCE)
            ll += -0.5 * (_LOG_2PI + math.log(s2) + returns[t] ** 2 / s2)
        return ll

    # ------------------------------------------------------------------
    # Parameter initialisation
    # ------------------------------------------------------------------

    def _init_params(
        self,
        returns: List[float],
        n_features: int,
    ) -> Tuple[float, List[float], List[float], List[float]]:
        """Initialise GARCH parameters.

        Parameters
        ----------
        returns : List[float]
            Return series.
        n_features : int
            Number of external features.

        Returns
        -------
        Tuple[float, List[float], List[float], List[float]]
            (omega, alphas, betas, gammas) initial values.
        """
        sample_var = sum(r * r for r in returns) / len(returns)
        sample_var = max(sample_var, _MIN_VARIANCE)

        # Standard initialisation: omega captures most variance,
        # alpha and beta split the rest
        total_persistence = 0.85
        alpha_share = 0.10
        beta_share = total_persistence - alpha_share

        omega = sample_var * (1.0 - total_persistence)
        alphas = [alpha_share / self.q] * self.q
        betas = [beta_share / self.p] * self.p
        gammas = [0.0] * n_features

        return omega, alphas, betas, gammas

    # ------------------------------------------------------------------
    # Optimisation (coordinate descent with constraints)
    # ------------------------------------------------------------------

    def _enforce_constraints(
        self,
        omega: float,
        alphas: List[float],
        betas: List[float],
        gammas: List[float],
    ) -> Tuple[float, List[float], List[float], List[float]]:
        """Enforce positivity and stationarity constraints.

        Parameters
        ----------
        omega : float
            Constant term.
        alphas : List[float]
            ARCH coefficients.
        betas : List[float]
            GARCH coefficients.
        gammas : List[float]
            Feature coefficients.

        Returns
        -------
        Tuple[float, List[float], List[float], List[float]]
            Constrained parameter values.
        """
        omega = max(omega, _MIN_VARIANCE)
        alphas = [max(a, 0.0) for a in alphas]
        betas = [max(b, 0.0) for b in betas]

        # Stationarity: sum(alpha) + sum(beta) < 1
        persistence = sum(alphas) + sum(betas)
        if persistence >= 0.9999:
            scale = 0.999 / persistence
            alphas = [a * scale for a in alphas]
            betas = [b * scale for b in betas]

        return omega, alphas, betas, gammas

    def _numerical_gradient(
        self,
        returns: List[float],
        features: Optional[List[List[float]]],
        omega: float,
        alphas: List[float],
        betas: List[float],
        gammas: List[float],
        eps: float = 1e-6,
    ) -> Tuple[float, List[float], List[float], List[float]]:
        """Compute numerical gradient of log-likelihood.

        Parameters
        ----------
        returns : List[float]
            Return series.
        features : Optional[List[List[float]]]
            External features.
        omega : float
            Current omega.
        alphas : List[float]
            Current ARCH coefficients.
        betas : List[float]
            Current GARCH coefficients.
        gammas : List[float]
            Current feature coefficients.
        eps : float
            Finite difference step size.

        Returns
        -------
        Tuple[float, List[float], List[float], List[float]]
            Gradient for each parameter group.
        """
        def _ll(o: float, a: List[float], b: List[float],
                g: List[float]) -> float:
            s2 = self._compute_variance_series(
                returns, o, a, b, g, features
            )
            return self._log_likelihood(returns, s2)

        base_ll = _ll(omega, alphas, betas, gammas)

        # Gradient for omega
        d_omega = (
            _ll(omega + eps, alphas, betas, gammas) - base_ll
        ) / eps

        # Gradient for alphas
        d_alphas: List[float] = []
        for i in range(len(alphas)):
            a_plus = alphas[:]
            a_plus[i] += eps
            d_alphas.append(
                (_ll(omega, a_plus, betas, gammas) - base_ll) / eps
            )

        # Gradient for betas
        d_betas: List[float] = []
        for j in range(len(betas)):
            b_plus = betas[:]
            b_plus[j] += eps
            d_betas.append(
                (_ll(omega, alphas, b_plus, gammas) - base_ll) / eps
            )

        # Gradient for gammas
        d_gammas: List[float] = []
        for k in range(len(gammas)):
            g_plus = gammas[:]
            g_plus[k] += eps
            d_gammas.append(
                (_ll(omega, alphas, betas, g_plus) - base_ll) / eps
            )

        return d_omega, d_alphas, d_betas, d_gammas

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        returns: List[float],
        features: Optional[List[List[float]]] = None,
    ) -> MLGARCHResult:
        """Fit the GARCH model via quasi-maximum likelihood.

        Parameters
        ----------
        returns : List[float]
            Return series of length T.
        features : Optional[List[List[float]]]
            External features, shape T x n_features. Each row
            contains feature values available at time t-1
            (i.e., already lagged). Pass None if no features.

        Returns
        -------
        MLGARCHResult
            Fitting result with estimated parameters, conditional
            volatility, and standardized residuals.

        Raises
        ------
        ValueError
            If returns has fewer elements than max(p, q) + 1.
        """
        T = len(returns)
        min_obs = max(self.p, self.q) + 1
        if T < min_obs:
            raise ValueError(
                f"Need at least {min_obs} observations, got {T}"
            )

        n_features = 0
        if features is not None:
            if len(features) != T:
                raise ValueError(
                    "features length must match returns length"
                )
            n_features = len(features[0]) if features else 0

        omega, alphas, betas, gammas = self._init_params(
            returns, n_features
        )

        prev_ll = -math.inf
        converged = False
        iteration = 0
        learning_rate = 1e-5

        for iteration in range(1, self.max_iter + 1):
            sigma2 = self._compute_variance_series(
                returns, omega, alphas, betas, gammas, features
            )
            ll = self._log_likelihood(returns, sigma2)

            if abs(ll - prev_ll) < self.tol and iteration > 1:
                converged = True
                break
            prev_ll = ll

            # Gradient ascent step
            d_omega, d_alphas, d_betas, d_gammas = (
                self._numerical_gradient(
                    returns, features,
                    omega, alphas, betas, gammas,
                )
            )

            # Adaptive learning rate
            grad_norm = math.sqrt(
                d_omega ** 2
                + sum(d ** 2 for d in d_alphas)
                + sum(d ** 2 for d in d_betas)
                + sum(d ** 2 for d in d_gammas)
            )
            if grad_norm > 0:
                lr = min(learning_rate, 1.0 / grad_norm)
            else:
                lr = learning_rate

            omega += lr * d_omega
            alphas = [a + lr * d for a, d in zip(alphas, d_alphas)]
            betas = [b + lr * d for b, d in zip(betas, d_betas)]
            gammas = [g + lr * d for g, d in zip(gammas, d_gammas)]

            omega, alphas, betas, gammas = self._enforce_constraints(
                omega, alphas, betas, gammas
            )

        # Final computation
        sigma2 = self._compute_variance_series(
            returns, omega, alphas, betas, gammas, features
        )
        ll = self._log_likelihood(returns, sigma2)

        cond_vol = [math.sqrt(max(s, _MIN_VARIANCE)) for s in sigma2]
        std_resid = [
            returns[t] / cond_vol[t] if cond_vol[t] > 0 else 0.0
            for t in range(T)
        ]

        # Store fitted parameters
        self._omega = omega
        self._alphas = alphas[:]
        self._betas = betas[:]
        self._gammas = gammas[:]
        self._unconditional_var = (
            omega / max(1.0 - sum(alphas) - sum(betas), 1e-10)
        )
        self._fitted = True

        # Build parameter dictionary
        garch_params: Dict[str, float] = {"omega": omega}
        for i, a in enumerate(alphas):
            garch_params[f"alpha_{i + 1}"] = a
        for j, b in enumerate(betas):
            garch_params[f"beta_{j + 1}"] = b

        return MLGARCHResult(
            converged=converged,
            iterations=iteration,
            loss=-ll,
            message=(
                "Converged" if converged
                else f"Max iterations ({self.max_iter}) reached"
            ),
            garch_params=garch_params,
            feature_coefficients=gammas[:],
            conditional_volatility=cond_vol,
            standardized_residuals=std_resid,
        )

    def predict(
        self,
        returns: List[float],
        features: Optional[List[List[float]]] = None,
        n_ahead: int = 1,
    ) -> List[float]:
        """Forecast conditional volatility.

        Parameters
        ----------
        returns : List[float]
            Historical return series.
        features : Optional[List[List[float]]]
            External features aligned with returns.
        n_ahead : int
            Number of steps ahead to forecast.

        Returns
        -------
        List[float]
            Forecasted conditional standard deviations for
            ``n_ahead`` future steps.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                "Model must be fitted before prediction"
            )

        sigma2 = self._compute_variance_series(
            returns, self._omega, self._alphas, self._betas,
            self._gammas, features,
        )

        forecasts: List[float] = []
        T = len(returns)

        # Extend variance and return sequences for forecasting
        ext_sigma2 = sigma2[:]
        ext_returns = returns[:]

        for h in range(n_ahead):
            var_h = self._omega

            # ARCH terms
            for i, a in enumerate(self._alphas):
                idx = T + h - (i + 1)
                if idx >= 0 and idx < len(ext_returns):
                    var_h += a * ext_returns[idx] ** 2
                else:
                    # For multi-step, E[eps^2_{t+h}] = sigma^2_{t+h}
                    sigma_idx = T + h - (i + 1)
                    if 0 <= sigma_idx < len(ext_sigma2):
                        var_h += a * ext_sigma2[sigma_idx]
                    else:
                        var_h += a * self._unconditional_var

            # GARCH terms
            for j, b in enumerate(self._betas):
                idx = T + h - (j + 1)
                if 0 <= idx < len(ext_sigma2):
                    var_h += b * ext_sigma2[idx]
                else:
                    var_h += b * self._unconditional_var

            var_h = max(var_h, _MIN_VARIANCE)
            ext_sigma2.append(var_h)
            # Assume future returns = 0 for multi-step forecast
            ext_returns.append(0.0)

            forecasts.append(math.sqrt(var_h))

        return forecasts

    def forecast_volatility(
        self,
        returns: List[float],
        features: Optional[List[List[float]]] = None,
        horizon: int = 10,
    ) -> List[float]:
        """Forecast the term structure of volatility.

        Parameters
        ----------
        returns : List[float]
            Historical return series.
        features : Optional[List[List[float]]]
            External features aligned with returns.
        horizon : int
            Number of steps for the volatility term structure.

        Returns
        -------
        List[float]
            Forecasted conditional standard deviations for each
            horizon step (1, 2, ..., horizon).

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        return self.predict(returns, features, n_ahead=horizon)
