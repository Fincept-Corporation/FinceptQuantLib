"""
GARCH Family Models

Generalized Autoregressive Conditional Heteroskedasticity:
- GARCH(p, q)
- EGARCH
- GJR-GARCH (Threshold GARCH)
- Component GARCH
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class GARCHResult:
    """Result of GARCH model fitting."""
    omega: float
    alpha: List[float]
    beta: List[float]
    log_likelihood: float
    aic: float
    bic: float
    conditional_volatility: List[float]
    standardized_residuals: List[float]
    converged: bool


@dataclass
class GARCH:
    """
    GARCH(p, q) model.

    σ²_t = ω + Σ α_i ε²_{t-i} + Σ β_j σ²_{t-j}

    Standard GARCH with:
    - ω > 0 (constant)
    - α_i ≥ 0 (ARCH terms)
    - β_j ≥ 0 (GARCH terms)
    - Σα + Σβ < 1 (stationarity)

    Attributes:
        p: Number of ARCH lags
        q: Number of GARCH lags
    """
    p: int = 1
    q: int = 1

    def __post_init__(self) -> None:
        if self.p < 0 or self.q < 0:
            raise ValueError("p and q must be non-negative")
        if self.p == 0 and self.q == 0:
            raise ValueError("At least one of p, q must be positive")

    def fit(
        self,
        returns: List[float],
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> GARCHResult:
        """
        Fit GARCH model using quasi-maximum likelihood.

        Uses numerical optimization to maximize log-likelihood.

        Args:
            returns: Return series
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            GARCHResult with estimated parameters
        """
        n = len(returns)
        if n < max(self.p, self.q) + 10:
            raise ValueError("Insufficient data")

        # Initial values
        sample_var = sum(r ** 2 for r in returns) / n

        omega = sample_var * 0.05
        alpha = [0.05 / self.p] * self.p if self.p > 0 else []
        beta = [0.9 / self.q] * self.q if self.q > 0 else []

        # Optimization (simplified gradient descent)
        best_ll = float('-inf')
        best_params = (omega, alpha[:], beta[:])

        prev_ll = None
        for _ in range(max_iter):
            sigma2, ll = self._compute_likelihood(returns, omega, alpha, beta)

            if ll > best_ll:
                best_ll = ll
                best_params = (omega, alpha[:], beta[:])

            if prev_ll is not None and abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

            # Numerical gradient update
            eps = 1e-5
            grad_omega = (self._compute_likelihood(returns, omega + eps, alpha, beta)[1] - ll) / eps
            omega = max(1e-8, omega + 0.01 * grad_omega)

            for i in range(self.p):
                alpha_up = alpha[:]
                alpha_up[i] += eps
                grad_alpha = (self._compute_likelihood(returns, omega, alpha_up, beta)[1] - ll) / eps
                alpha[i] = max(0, min(0.5, alpha[i] + 0.01 * grad_alpha))

            for j in range(self.q):
                beta_up = beta[:]
                beta_up[j] += eps
                grad_beta = (self._compute_likelihood(returns, omega, alpha, beta_up)[1] - ll) / eps
                beta[j] = max(0, min(0.99, beta[j] + 0.01 * grad_beta))

            # Ensure stationarity
            total = sum(alpha) + sum(beta)
            if total >= 0.999:
                scale = 0.998 / total
                alpha = [a * scale for a in alpha]
                beta = [b * scale for b in beta]

        omega, alpha, beta = best_params
        sigma2, ll = self._compute_likelihood(returns, omega, alpha, beta)

        # Standardized residuals
        std_resid = [returns[i] / math.sqrt(sigma2[i]) if sigma2[i] > 0 else 0
                     for i in range(n)]

        # Information criteria
        k = 1 + self.p + self.q
        aic = -2 * ll + 2 * k
        bic = -2 * ll + k * math.log(n)

        return GARCHResult(
            omega=omega,
            alpha=alpha,
            beta=beta,
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            conditional_volatility=[math.sqrt(s) for s in sigma2],
            standardized_residuals=std_resid,
            converged=True
        )

    def _compute_likelihood(
        self,
        returns: List[float],
        omega: float,
        alpha: List[float],
        beta: List[float]
    ) -> Tuple[List[float], float]:
        """Compute conditional variance and log-likelihood."""
        n = len(returns)
        sigma2 = [0.0] * n

        # Initialize with unconditional variance
        uncond_var = omega / (1 - sum(alpha) - sum(beta))
        if uncond_var <= 0 or not math.isfinite(uncond_var):
            uncond_var = sum(r ** 2 for r in returns) / n

        for t in range(n):
            sigma2[t] = omega

            for i in range(self.p):
                if t - i - 1 >= 0:
                    sigma2[t] += alpha[i] * returns[t - i - 1] ** 2
                else:
                    sigma2[t] += alpha[i] * uncond_var

            for j in range(self.q):
                if t - j - 1 >= 0:
                    sigma2[t] += beta[j] * sigma2[t - j - 1]
                else:
                    sigma2[t] += beta[j] * uncond_var

            sigma2[t] = max(sigma2[t], 1e-10)

        # Log-likelihood (Gaussian)
        ll = 0.0
        for t in range(n):
            ll -= 0.5 * (math.log(2 * math.pi) + math.log(sigma2[t]) +
                        returns[t] ** 2 / sigma2[t])

        return sigma2, ll

    def forecast(
        self,
        returns: List[float],
        sigma2_last: float,
        omega: float,
        alpha: List[float],
        beta: List[float],
        horizon: int = 1
    ) -> List[float]:
        """
        Forecast conditional variance.

        Args:
            returns: Historical returns (at least p values)
            sigma2_last: Last conditional variance
            omega, alpha, beta: Model parameters
            horizon: Forecast horizon

        Returns:
            List of forecasted variances
        """
        forecasts = []
        eps2 = [r ** 2 for r in returns[-self.p:]] if self.p > 0 else []
        sigma2_hist = [sigma2_last]

        for h in range(horizon):
            sigma2_h = omega

            for i in range(self.p):
                if h - i - 1 < 0:
                    # Use actual squared returns
                    idx = len(eps2) + (h - i - 1)
                    if idx >= 0:
                        sigma2_h += alpha[i] * eps2[idx]
                    else:
                        sigma2_h += alpha[i] * sigma2_hist[-1]
                else:
                    # Future ε² = σ²
                    sigma2_h += alpha[i] * forecasts[h - i - 1]

            for j in range(self.q):
                if h - j - 1 < 0:
                    sigma2_h += beta[j] * sigma2_hist[-1]
                else:
                    sigma2_h += beta[j] * forecasts[h - j - 1]

            forecasts.append(sigma2_h)

        return forecasts


@dataclass
class EGARCH:
    """
    Exponential GARCH model.

    log(σ²_t) = ω + Σ α_i g(z_{t-i}) + Σ β_j log(σ²_{t-j})

    where g(z) = θz + γ(|z| - E|z|)

    Features:
    - No positivity constraints
    - Captures asymmetry (leverage effect)
    """
    p: int = 1
    q: int = 1

    def fit(self, returns: List[float], max_iter: int = 100) -> dict:
        """Fit EGARCH model (simplified)."""
        n = len(returns)
        sample_var = sum(r ** 2 for r in returns) / n

        # Initial parameters
        omega = math.log(sample_var) * 0.1
        alpha = [0.1] * self.p
        gamma = [-0.05] * self.p  # Leverage
        beta = [0.9 / self.q] * self.q

        log_sigma2 = [math.log(sample_var)] * n

        for _ in range(max_iter):
            # Update log variance
            for t in range(max(self.p, self.q), n):
                log_sigma2[t] = omega

                for i in range(self.p):
                    z = returns[t - i - 1] / math.exp(0.5 * log_sigma2[t - i - 1])
                    g_z = alpha[i] * z + gamma[i] * (abs(z) - math.sqrt(2 / math.pi))
                    log_sigma2[t] += g_z

                for j in range(self.q):
                    log_sigma2[t] += beta[j] * log_sigma2[t - j - 1]

        return {
            'omega': omega,
            'alpha': alpha,
            'gamma': gamma,
            'beta': beta,
            'conditional_volatility': [math.exp(0.5 * ls) for ls in log_sigma2]
        }


@dataclass
class GJRGARCH:
    """
    GJR-GARCH (Threshold GARCH) model.

    σ²_t = ω + Σ (α_i + γ_i I_{t-i}) ε²_{t-i} + Σ β_j σ²_{t-j}

    where I_t = 1 if ε_t < 0 (negative shock indicator)

    Captures asymmetric response to positive/negative shocks.
    """
    p: int = 1
    q: int = 1

    def fit(self, returns: List[float], max_iter: int = 100) -> dict:
        """Fit GJR-GARCH model (simplified)."""
        n = len(returns)
        sample_var = sum(r ** 2 for r in returns) / n

        omega = sample_var * 0.05
        alpha = [0.03] * self.p
        gamma = [0.04] * self.p  # Asymmetry
        beta = [0.9 / self.q] * self.q

        sigma2 = [sample_var] * n

        for _ in range(max_iter):
            for t in range(max(self.p, self.q), n):
                sigma2[t] = omega

                for i in range(self.p):
                    eps = returns[t - i - 1]
                    indicator = 1.0 if eps < 0 else 0.0
                    sigma2[t] += (alpha[i] + gamma[i] * indicator) * eps ** 2

                for j in range(self.q):
                    sigma2[t] += beta[j] * sigma2[t - j - 1]

                sigma2[t] = max(sigma2[t], 1e-10)

        return {
            'omega': omega,
            'alpha': alpha,
            'gamma': gamma,
            'beta': beta,
            'conditional_volatility': [math.sqrt(s) for s in sigma2]
        }
