"""
ARIMA Models

Autoregressive Integrated Moving Average:
- AR(p): Autoregressive
- MA(q): Moving Average
- ARMA(p, q): Combined
- ARIMA(p, d, q): With differencing
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


@dataclass
class ARIMAResult:
    """Result of ARIMA model fitting."""
    ar_coefs: List[float]
    ma_coefs: List[float]
    intercept: float
    sigma2: float
    log_likelihood: float
    aic: float
    bic: float
    residuals: List[float]
    fitted_values: List[float]


@dataclass
class AR:
    """
    Autoregressive AR(p) model.

    X_t = c + φ_1 X_{t-1} + ... + φ_p X_{t-p} + ε_t

    Attributes:
        p: Order of autoregression
    """
    p: int

    def __post_init__(self) -> None:
        if self.p < 1:
            raise ValueError("Order p must be at least 1")

    def fit(self, y: List[float]) -> ARIMAResult:
        """
        Fit AR(p) model using Yule-Walker equations.

        Args:
            y: Time series data

        Returns:
            ARIMAResult with estimated parameters
        """
        n = len(y)
        if n <= self.p:
            raise ValueError("Insufficient data for AR order")

        # Demean
        y_mean = sum(y) / n
        y_centered = [yi - y_mean for yi in y]

        # Compute autocorrelations
        gamma = self._autocovariance(y_centered, self.p)

        # Yule-Walker (Γφ = γ, Γ the Toeplitz matrix of autocovariances),
        # solved with Levinson-Durbin
        phi = self._levinson_durbin(gamma[:self.p + 1])

        # Intercept: c = μ(1 - Σφ_i)
        intercept = y_mean * (1 - sum(phi))

        # Residuals and fitted values
        fitted = [0.0] * n
        residuals = [0.0] * n

        for t in range(self.p, n):
            fitted[t] = intercept
            for i in range(self.p):
                fitted[t] += phi[i] * y[t - i - 1]
            residuals[t] = y[t] - fitted[t]

        # Variance estimate
        sigma2 = sum(r ** 2 for r in residuals[self.p:]) / (n - self.p - self.p)
        sigma2 = max(sigma2, 1e-10)

        # Log-likelihood
        ll = -0.5 * (n - self.p) * (1 + math.log(2 * math.pi) + math.log(sigma2))

        # Information criteria
        k = self.p + 1  # AR coefs + intercept
        aic = -2 * ll + 2 * k
        bic = -2 * ll + k * math.log(n - self.p)

        return ARIMAResult(
            ar_coefs=phi,
            ma_coefs=[],
            intercept=intercept,
            sigma2=sigma2,
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            residuals=residuals,
            fitted_values=fitted
        )

    def _autocovariance(self, y: List[float], max_lag: int) -> List[float]:
        """Compute sample autocovariances."""
        n = len(y)
        gamma = []
        for k in range(max_lag + 1):
            cov = sum(y[t] * y[t - k] for t in range(k, n)) / n
            gamma.append(cov)
        return gamma

    def _levinson_durbin(self, gamma: List[float]) -> List[float]:
        """Levinson-Durbin algorithm for AR coefficients."""
        p = len(gamma) - 1
        if p == 0:
            return []

        phi = [[0.0] * p for _ in range(p)]
        phi[0][0] = gamma[1] / gamma[0]

        for k in range(1, p):
            num = gamma[k + 1] - sum(phi[k - 1][j] * gamma[k - j] for j in range(k))
            denom = gamma[0] - sum(phi[k - 1][j] * gamma[j + 1] for j in range(k))

            if abs(denom) < 1e-10:
                break

            phi[k][k] = num / denom

            for j in range(k):
                phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - 1 - j]

        return phi[p - 1]

    def forecast(
        self,
        y: List[float],
        ar_coefs: List[float],
        intercept: float,
        horizon: int
    ) -> List[float]:
        """
        Forecast future values.

        Args:
            y: Historical data
            ar_coefs: AR coefficients
            intercept: Model intercept
            horizon: Forecast horizon

        Returns:
            Forecasted values
        """
        forecasts = []
        history = y[-self.p:]

        for h in range(horizon):
            f = intercept
            for i in range(self.p):
                if i < len(forecasts):
                    f += ar_coefs[i] * forecasts[-(i + 1)]
                else:
                    idx = len(history) - (i + 1 - len(forecasts))
                    if idx >= 0:
                        f += ar_coefs[i] * history[idx]
            forecasts.append(f)

        return forecasts


@dataclass
class MA:
    """
    Moving Average MA(q) model.

    X_t = μ + ε_t + θ_1 ε_{t-1} + ... + θ_q ε_{t-q}
    """
    q: int

    def __post_init__(self) -> None:
        if self.q < 1:
            raise ValueError("Order q must be at least 1")

    def fit(self, y: List[float], max_iter: int = 100) -> ARIMAResult:
        """
        Fit MA(q) model using innovations algorithm.

        Args:
            y: Time series data
            max_iter: Maximum iterations

        Returns:
            ARIMAResult
        """
        n = len(y)
        y_mean = sum(y) / n
        y_centered = [yi - y_mean for yi in y]

        # Initialize
        theta = [0.0] * self.q
        sigma2 = sum(yc ** 2 for yc in y_centered) / n

        for _ in range(max_iter):
            # Compute residuals
            residuals = [0.0] * n
            for t in range(n):
                residuals[t] = y_centered[t]
                for j in range(min(t, self.q)):
                    residuals[t] -= theta[j] * residuals[t - j - 1]

            # Update theta using autocorrelations of residuals
            gamma = [sum(residuals[t] * residuals[t - k] for t in range(k, n)) / n
                     for k in range(self.q + 1)]

            for j in range(self.q):
                if abs(gamma[0]) > 1e-10:
                    theta[j] = gamma[j + 1] / gamma[0] * 0.5 + theta[j] * 0.5

            sigma2 = gamma[0]

        # Final residuals
        residuals = [0.0] * n
        fitted = [0.0] * n
        for t in range(n):
            fitted[t] = y_mean
            for j in range(min(t, self.q)):
                fitted[t] += theta[j] * residuals[t - j - 1]
            residuals[t] = y[t] - fitted[t]

        ll = -0.5 * n * (1 + math.log(2 * math.pi) + math.log(max(sigma2, 1e-10)))
        k = self.q + 2
        aic = -2 * ll + 2 * k
        bic = -2 * ll + k * math.log(n)

        return ARIMAResult(
            ar_coefs=[],
            ma_coefs=theta,
            intercept=y_mean,
            sigma2=sigma2,
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            residuals=residuals,
            fitted_values=fitted
        )


@dataclass
class ARIMA:
    """
    ARIMA(p, d, q) model.

    Combines AR(p), differencing of order d, and MA(q).
    """
    p: int
    d: int
    q: int

    def __post_init__(self) -> None:
        if self.p < 0 or self.d < 0 or self.q < 0:
            raise ValueError("Orders must be non-negative")
        if self.p == 0 and self.q == 0:
            raise ValueError("At least one of p, q must be positive")

    def difference(self, y: List[float], d: int = 1) -> List[float]:
        """Apply differencing d times."""
        result = y[:]
        for _ in range(d):
            result = [result[i] - result[i - 1] for i in range(1, len(result))]
        return result

    def integrate(self, dy: List[float], y_init: List[float]) -> List[float]:
        """Reverse differencing (cumsum)."""
        result = list(y_init)
        for diff in dy:
            result.append(result[-1] + diff)
        return result[len(y_init):]

    def fit(self, y: List[float]) -> ARIMAResult:
        """
        Fit ARIMA(p, d, q) model.

        1. Difference the series d times
        2. Fit ARMA(p, q) to differenced series
        """
        # Difference
        y_diff = self.difference(y, self.d)

        if self.q == 0:
            # Pure AR
            ar_model = AR(self.p)
            return ar_model.fit(y_diff)
        elif self.p == 0:
            # Pure MA
            ma_model = MA(self.q)
            return ma_model.fit(y_diff)
        else:
            # ARMA - simplified estimation
            return self._fit_arma(y_diff)

    def _fit_arma(self, y: List[float]) -> ARIMAResult:
        """Fit ARMA using iterative approach."""
        n = len(y)
        y_mean = sum(y) / n

        # Initialize with AR fit
        ar = AR(self.p)
        ar_result = ar.fit(y)
        ar_coefs = ar_result.ar_coefs
        residuals = ar_result.residuals

        # Estimate MA from residuals
        ma_coefs = [0.0] * self.q
        for iteration in range(50):
            # Update residuals
            fitted = [0.0] * n
            for t in range(max(self.p, self.q), n):
                fitted[t] = y_mean * (1 - sum(ar_coefs))
                for i in range(self.p):
                    fitted[t] += ar_coefs[i] * y[t - i - 1]
                for j in range(self.q):
                    fitted[t] += ma_coefs[j] * residuals[t - j - 1]
                residuals[t] = y[t] - fitted[t]

            # Update MA coefficients
            gamma_r = [sum(residuals[t] * residuals[t - k] for t in range(k, n)) / n
                       for k in range(self.q + 1)]
            for j in range(self.q):
                if abs(gamma_r[0]) > 1e-10:
                    ma_coefs[j] = 0.5 * gamma_r[j + 1] / gamma_r[0] + 0.5 * ma_coefs[j]

        sigma2 = sum(r ** 2 for r in residuals[max(self.p, self.q):]) / (n - self.p - self.q)
        ll = -0.5 * (n - max(self.p, self.q)) * (1 + math.log(2 * math.pi) + math.log(max(sigma2, 1e-10)))
        k = self.p + self.q + 2
        aic = -2 * ll + 2 * k
        bic = -2 * ll + k * math.log(n)

        return ARIMAResult(
            ar_coefs=ar_coefs,
            ma_coefs=ma_coefs,
            intercept=y_mean * (1 - sum(ar_coefs)),
            sigma2=sigma2,
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            residuals=residuals,
            fitted_values=fitted
        )

    def forecast(
        self,
        y: List[float],
        result: ARIMAResult,
        horizon: int
    ) -> List[float]:
        """Forecast future values."""
        y_diff = self.difference(y, self.d)

        # Forecast on differenced series
        diff_forecasts = []
        history = y_diff[-(self.p + self.q):]
        residuals = result.residuals[-(self.q):]

        for h in range(horizon):
            f = result.intercept
            for i in range(self.p):
                if i < len(diff_forecasts):
                    f += result.ar_coefs[i] * diff_forecasts[-(i + 1)]
                else:
                    idx = len(history) - (i + 1 - len(diff_forecasts))
                    if idx >= 0:
                        f += result.ar_coefs[i] * history[idx]

            for j in range(self.q):
                if j < len(residuals) and h == 0:
                    f += result.ma_coefs[j] * residuals[-(j + 1)]
                # Future residuals are zero in expectation

            diff_forecasts.append(f)

        # Integrate back
        if self.d > 0:
            y_init = y[-self.d:]
            return self.integrate(diff_forecasts, y_init)
        return diff_forecasts
