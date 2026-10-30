"""
Parametric Yield Curve Models

Implements popular parametric models for yield curves:
- Nelson-Siegel (1987): 3-factor model
- Nelson-Siegel-Svensson (1994): 4-factor extension
- Diebold-Li: Time-varying Nelson-Siegel

These models represent the yield curve with a small number of parameters,
making them suitable for smoothing, extrapolation, and forecasting.
"""

from __future__ import annotations

from datetime import date
from typing import Union, Tuple
import math
from scipy.optimize import minimize

from finceptquantlib.curves.curve import Curve

# Squared rate errors are ~1e-7, far below L-BFGS-B's gradient tolerance, which would stop
# the fit at the initial tau; measuring the error in basis points lets tau be optimised.
_BP2 = 1e8


class NelsonSiegelCurve(Curve):
    """
    Nelson-Siegel parametric yield curve model.

    The model specifies the instantaneous forward rate as:

    f(t) = β₀ + β₁ * exp(-t/τ) + β₂ * (t/τ) * exp(-t/τ)

    The zero rate is:

    r(t) = β₀ + β₁ * (1 - exp(-t/τ)) / (t/τ)
              + β₂ * ((1 - exp(-t/τ)) / (t/τ) - exp(-t/τ))

    Parameters:
    - β₀ (beta0): Long-term rate level (asymptote)
    - β₁ (beta1): Short-term component (slope)
    - β₂ (beta2): Medium-term component (curvature)
    - τ (tau): Decay parameter (controls where curvature peaks)

    Reference: Nelson & Siegel (1987), "Parsimonious Modeling of Yield Curves"
    """

    def __init__(
        self,
        reference_date: date,
        beta0: float,
        beta1: float,
        beta2: float,
        tau: float,
        day_count: str = "ACT/365F",
    ):
        """
        Initialize Nelson-Siegel curve.

        Args:
            reference_date: Curve valuation date
            beta0: Long-term rate level
            beta1: Short-term component
            beta2: Medium-term component (curvature)
            tau: Decay parameter
            day_count: Day count convention
        """
        self.reference_date = reference_date
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau

        if isinstance(day_count, str):
            from finceptquantlib.scheduling.daycount import _get_convention
            day_count = _get_convention(day_count)
        self._day_count = day_count

        # For compatibility with base class
        self._tenors = []
        self._values = []
        self._interpolation_method = "nelson_siegel"

    @property
    def parameters(self) -> Tuple[float, float, float, float]:
        """Return model parameters (beta0, beta1, beta2, tau)."""
        return (self.beta0, self.beta1, self.beta2, self.tau)

    def zero_rate(
        self,
        t: Union[float, date, list[float]],
        compounding: str = "continuous"
    ) -> Union[float, list[float]]:
        """
        Calculate zero rate using Nelson-Siegel formula.

        r(t) = β₀ + β₁ * (1 - exp(-t/τ)) / (t/τ)
              + β₂ * ((1 - exp(-t/τ)) / (t/τ) - exp(-t/τ))
        """
        if isinstance(t, date):
            t = self._day_count.year_fraction(self.reference_date, t)

        scalar_input = isinstance(t, (int, float))
        if scalar_input:
            t_list = [max(float(t), 1e-10)]
        else:
            t_list = [max(float(ti), 1e-10) for ti in t]

        rates = []
        for ti in t_list:
            x = ti / self.tau
            exp_x = math.exp(-x)
            factor1 = (1 - exp_x) / x
            factor2 = factor1 - exp_x

            rate = self.beta0 + self.beta1 * factor1 + self.beta2 * factor2

            if compounding != "continuous":
                periods_per_year = {
                    "annual": 1, "semi-annual": 2, "quarterly": 4, "monthly": 12
                }.get(compounding, 1)
                rate = periods_per_year * (math.exp(rate / periods_per_year) - 1)

            rates.append(rate)

        return float(rates[0]) if scalar_input else rates

    def discount_factor(
        self,
        t: Union[float, date, list[float]]
    ) -> Union[float, list[float]]:
        """Calculate discount factor from zero rate."""
        if isinstance(t, date):
            t = self._day_count.year_fraction(self.reference_date, t)

        scalar_input = isinstance(t, (int, float))
        if scalar_input:
            t_list = [float(t)]
        else:
            t_list = [float(ti) for ti in t]

        rate = self.zero_rate(t)
        if scalar_input:
            rates = [rate]
        else:
            rates = rate

        dfs = []
        for ti, ri in zip(t_list, rates):
            if ti <= 0:
                dfs.append(1.0)
            else:
                dfs.append(math.exp(-ri * ti))

        return float(dfs[0]) if scalar_input else dfs

    def instantaneous_forward(self, t: Union[float, date]) -> float:
        """
        Calculate instantaneous forward rate.

        f(t) = β₀ + β₁ * exp(-t/τ) + β₂ * (t/τ) * exp(-t/τ)
        """
        if isinstance(t, date):
            t = self._day_count.year_fraction(self.reference_date, t)

        t = max(t, 1e-10)
        x = t / self.tau
        exp_x = math.exp(-x)

        return self.beta0 + self.beta1 * exp_x + self.beta2 * x * exp_x

    @classmethod
    def fit(
        cls,
        reference_date: date,
        tenors: list[float],
        rates: list[float],
        tau_initial: float = 1.5,
        fix_tau: bool = False,
    ) -> "NelsonSiegelCurve":
        """
        Fit Nelson-Siegel model to market data.

        Args:
            reference_date: Curve valuation date
            tenors: Maturities in years
            rates: Market zero rates
            tau_initial: Initial guess for tau
            fix_tau: If True, don't optimize tau

        Returns:
            Fitted NelsonSiegelCurve
        """
        tenors_list = [float(t) for t in tenors]
        rates_list = [float(r) for r in rates]
        n = len(tenors_list)

        def _design_matrix(t_list: list[float], tau: float) -> list[list[float]]:
            """Build design matrix for linear regression."""
            matrix = []
            for t in t_list:
                t = max(t, 1e-10)
                x = t / tau
                exp_x = math.exp(-x)
                col1 = 1.0
                col2 = (1 - exp_x) / x
                col3 = col2 - exp_x
                matrix.append([col1, col2, col3])
            return matrix

        def _solve_ols(X: list[list[float]], y: list[float]) -> list[float]:
            """Simple OLS solver (X'X)^-1 X'y"""
            n_obs = len(X)
            m = len(X[0])
            XtX = [[sum(X[k][i] * X[k][j] for k in range(n_obs)) for j in range(m)] for i in range(m)]
            Xty = [sum(X[k][i] * y[k] for k in range(n_obs)) for i in range(m)]
            return _solve_3x3(XtX, Xty)

        def _solve_3x3(A: list[list[float]], b: list[float]) -> list[float]:
            """Solve 3x3 linear system using Gaussian elimination with pivoting."""
            sz = 3
            M = [[A[i][j] for j in range(sz)] + [b[i]] for i in range(sz)]

            for col in range(sz):
                max_row = col
                for row in range(col + 1, sz):
                    if abs(M[row][col]) > abs(M[max_row][col]):
                        max_row = row
                M[col], M[max_row] = M[max_row], M[col]

                if abs(M[col][col]) < 1e-12:
                    return [0.0, 0.0, 0.0]

                for row in range(col + 1, sz):
                    factor = M[row][col] / M[col][col]
                    for j in range(col, sz + 1):
                        M[row][j] -= factor * M[col][j]

            x = [0.0] * sz
            for i in range(sz - 1, -1, -1):
                x[i] = M[i][sz]
                for j in range(i + 1, sz):
                    x[i] -= M[i][j] * x[j]
                x[i] /= M[i][i]

            return x

        def _objective(params: list[float]) -> float:
            """Objective function: sum of squared errors."""
            if fix_tau:
                beta = params
                tau = tau_initial
            else:
                beta = params[:3]
                tau = max(params[3], 0.1)

            X = _design_matrix(tenors_list, tau)
            predicted = [sum(X[i][j] * beta[j] for j in range(3)) for i in range(n)]
            return _BP2 * sum((predicted[i] - rates_list[i]) ** 2 for i in range(n))

        # Initial guess for betas via OLS with initial tau
        X = _design_matrix(tenors_list, tau_initial)
        beta_init = _solve_ols(X, rates_list)

        if fix_tau:
            x0 = beta_init
            bounds = [(None, None)] * 3
        else:
            x0 = beta_init + [tau_initial]
            bounds = [(None, None)] * 3 + [(0.1, 30.0)]

        result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds)

        if fix_tau:
            beta0, beta1, beta2 = result.x
            tau = tau_initial
        else:
            beta0, beta1, beta2, tau = result.x

        return cls(
            reference_date=reference_date,
            beta0=beta0,
            beta1=beta1,
            beta2=beta2,
            tau=tau,
        )

    def __repr__(self) -> str:
        return (f"NelsonSiegelCurve(β₀={self.beta0:.4f}, β₁={self.beta1:.4f}, "
                f"β₂={self.beta2:.4f}, τ={self.tau:.4f})")


class NelsonSiegelSvenssonCurve(Curve):
    """
    Nelson-Siegel-Svensson (extended) parametric yield curve model.

    Adds a second curvature term to Nelson-Siegel:

    f(t) = β₀ + β₁ * exp(-t/τ₁) + β₂ * (t/τ₁) * exp(-t/τ₁)
              + β₃ * (t/τ₂) * exp(-t/τ₂)

    Parameters:
    - β₀: Long-term rate level
    - β₁: Short-term component
    - β₂: First medium-term component
    - β₃: Second medium-term component
    - τ₁: First decay parameter
    - τ₂: Second decay parameter

    Reference: Svensson (1994), "Estimating and Interpreting Forward Interest Rates"
    """

    def __init__(
        self,
        reference_date: date,
        beta0: float,
        beta1: float,
        beta2: float,
        beta3: float,
        tau1: float,
        tau2: float,
        day_count: str = "ACT/365F",
    ):
        """
        Initialize Svensson curve.

        Args:
            reference_date: Curve valuation date
            beta0: Long-term rate level
            beta1: Short-term component
            beta2: First medium-term component
            beta3: Second medium-term component
            tau1: First decay parameter
            tau2: Second decay parameter
            day_count: Day count convention
        """
        self.reference_date = reference_date
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.tau1 = tau1
        self.tau2 = tau2

        if isinstance(day_count, str):
            from finceptquantlib.scheduling.daycount import _get_convention
            day_count = _get_convention(day_count)
        self._day_count = day_count

        self._tenors = []
        self._values = []
        self._interpolation_method = "svensson"

    @property
    def parameters(self) -> Tuple[float, float, float, float, float, float]:
        """Return model parameters."""
        return (self.beta0, self.beta1, self.beta2, self.beta3, self.tau1, self.tau2)

    def zero_rate(
        self,
        t: Union[float, date, list[float]],
        compounding: str = "continuous"
    ) -> Union[float, list[float]]:
        """
        Calculate zero rate using Svensson formula.

        r(t) = β₀ + β₁ * (1 - exp(-t/τ₁)) / (t/τ₁)
              + β₂ * ((1 - exp(-t/τ₁)) / (t/τ₁) - exp(-t/τ₁))
              + β₃ * ((1 - exp(-t/τ₂)) / (t/τ₂) - exp(-t/τ₂))
        """
        if isinstance(t, date):
            t = self._day_count.year_fraction(self.reference_date, t)

        scalar_input = isinstance(t, (int, float))
        if scalar_input:
            t_list = [max(float(t), 1e-10)]
        else:
            t_list = [max(float(ti), 1e-10) for ti in t]

        rates = []
        for ti in t_list:
            x1 = ti / self.tau1
            x2 = ti / self.tau2
            exp_x1 = math.exp(-x1)
            exp_x2 = math.exp(-x2)

            factor1 = (1 - exp_x1) / x1
            factor2 = factor1 - exp_x1
            factor3 = (1 - exp_x2) / x2 - exp_x2

            rate = (self.beta0 + self.beta1 * factor1 +
                    self.beta2 * factor2 + self.beta3 * factor3)

            if compounding != "continuous":
                periods_per_year = {
                    "annual": 1, "semi-annual": 2, "quarterly": 4, "monthly": 12
                }.get(compounding, 1)
                rate = periods_per_year * (math.exp(rate / periods_per_year) - 1)

            rates.append(rate)

        return float(rates[0]) if scalar_input else rates

    def discount_factor(
        self,
        t: Union[float, date, list[float]]
    ) -> Union[float, list[float]]:
        """Calculate discount factor from zero rate."""
        if isinstance(t, date):
            t = self._day_count.year_fraction(self.reference_date, t)

        scalar_input = isinstance(t, (int, float))
        if scalar_input:
            t_list = [float(t)]
        else:
            t_list = [float(ti) for ti in t]

        rate = self.zero_rate(t)
        if scalar_input:
            rates = [rate]
        else:
            rates = rate

        dfs = []
        for ti, ri in zip(t_list, rates):
            if ti <= 0:
                dfs.append(1.0)
            else:
                dfs.append(math.exp(-ri * ti))

        return float(dfs[0]) if scalar_input else dfs

    def instantaneous_forward(self, t: Union[float, date]) -> float:
        """
        Calculate instantaneous forward rate.

        f(t) = β₀ + β₁ * exp(-t/τ₁) + β₂ * (t/τ₁) * exp(-t/τ₁)
              + β₃ * (t/τ₂) * exp(-t/τ₂)
        """
        if isinstance(t, date):
            t = self._day_count.year_fraction(self.reference_date, t)

        t = max(t, 1e-10)
        x1 = t / self.tau1
        x2 = t / self.tau2
        exp_x1 = math.exp(-x1)
        exp_x2 = math.exp(-x2)

        return (self.beta0 + self.beta1 * exp_x1 +
                self.beta2 * x1 * exp_x1 + self.beta3 * x2 * exp_x2)

    @classmethod
    def fit(
        cls,
        reference_date: date,
        tenors: list[float],
        rates: list[float],
        tau1_initial: float = 1.5,
        tau2_initial: float = 5.0,
    ) -> "NelsonSiegelSvenssonCurve":
        """
        Fit Svensson model to market data.

        Args:
            reference_date: Curve valuation date
            tenors: Maturities in years
            rates: Market zero rates
            tau1_initial: Initial guess for tau1
            tau2_initial: Initial guess for tau2

        Returns:
            Fitted NelsonSiegelSvenssonCurve
        """
        tenors_list = [float(t) for t in tenors]
        rates_list = [float(r) for r in rates]
        n = len(tenors_list)

        def _objective(params: list[float]) -> float:
            """Objective function: sum of squared errors."""
            beta0, beta1, beta2, beta3, tau1, tau2 = params
            tau1 = max(tau1, 0.1)
            tau2 = max(tau2, 0.1)

            curve = cls(
                reference_date=reference_date,
                beta0=beta0, beta1=beta1, beta2=beta2, beta3=beta3,
                tau1=tau1, tau2=tau2
            )
            predicted = curve.zero_rate(tenors_list)
            return _BP2 * sum((predicted[i] - rates_list[i]) ** 2 for i in range(n))

        # Initial guess
        ns_fit = NelsonSiegelCurve.fit(reference_date, tenors_list, rates_list, tau_initial=tau1_initial)
        x0 = [ns_fit.beta0, ns_fit.beta1, ns_fit.beta2, 0.0,
              ns_fit.tau, tau2_initial]

        bounds = [(None, None)] * 4 + [(0.1, 30.0)] * 2

        result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds)
        beta0, beta1, beta2, beta3, tau1, tau2 = result.x

        return cls(
            reference_date=reference_date,
            beta0=beta0, beta1=beta1, beta2=beta2, beta3=beta3,
            tau1=tau1, tau2=tau2
        )

    def __repr__(self) -> str:
        return (f"NelsonSiegelSvenssonCurve(β₀={self.beta0:.4f}, β₁={self.beta1:.4f}, "
                f"β₂={self.beta2:.4f}, β₃={self.beta3:.4f}, "
                f"τ₁={self.tau1:.4f}, τ₂={self.tau2:.4f})")
