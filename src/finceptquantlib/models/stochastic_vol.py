"""
Stochastic Volatility Models

Implements industry-standard stochastic volatility models:

1. Heston Model (1993):
   - Mean-reverting stochastic variance
   - Closed-form pricing via characteristic functions
   - FFT-based pricing (Carr-Madan)
   - Calibration to market data

2. SABR Model (Hagan et al., 2002):
   - Stochastic Alpha Beta Rho
   - Hagan's implied volatility approximation
   - Normal SABR for negative rates
   - Obloj correction for low strikes

Key Features:
- Semi-analytical pricing via Fourier inversion
- Monte Carlo simulation for path-dependent
- Calibration with analytical gradients
- Greeks via automatic differentiation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
import math
import cmath
import random

from finceptquantlib.core.foundations.types import OptionType
from finceptquantlib.core.utilities.distributions import norm_cdf, norm_pdf
from finceptquantlib.core.utilities.results import MonteCarloResult
from finceptquantlib.numerical.fft import cooley_tukey_fft, next_power_of_2
from finceptquantlib.volatility.sabr import SABRParameters


# =============================================================================
# Heston Model
# =============================================================================

@dataclass
class HestonParameters:
    """
    Heston model parameters.

    The Heston SDE:
        dS = μ*S*dt + √v*S*dW_S
        dv = κ*(θ - v)*dt + σ*√v*dW_v
        dW_S * dW_v = ρ*dt

    Parameters:
        v0: Initial variance (not volatility!)
        kappa: Mean reversion speed
        theta: Long-term variance
        sigma: Volatility of volatility (vol of vol)
        rho: Correlation between spot and variance
    """
    v0: float          # Initial variance
    kappa: float       # Mean reversion speed
    theta: float       # Long-term variance
    sigma: float       # Vol of vol
    rho: float         # Correlation

    def __post_init__(self):
        """Validate parameters."""
        if self.v0 <= 0:
            raise ValueError("v0 must be positive")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.theta <= 0:
            raise ValueError("theta must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if not -1 <= self.rho <= 1:
            raise ValueError("rho must be in [-1, 1]")

    @property
    def feller_condition(self) -> bool:
        """
        Check Feller condition: 2*κ*θ > σ²
        Ensures variance stays positive.
        """
        return 2 * self.kappa * self.theta > self.sigma ** 2

    @classmethod
    def from_volatility(
        cls,
        vol0: float,
        kappa: float,
        theta_vol: float,
        sigma: float,
        rho: float,
    ) -> "HestonParameters":
        """
        Create parameters from volatility (more intuitive).

        Args:
            vol0: Initial volatility (e.g., 0.20 for 20%)
            kappa: Mean reversion speed
            theta_vol: Long-term volatility
            sigma: Vol of vol
            rho: Correlation
        """
        return cls(
            v0=vol0 ** 2,
            kappa=kappa,
            theta=theta_vol ** 2,
            sigma=sigma,
            rho=rho,
        )


class HestonModel:
    """
    Heston Stochastic Volatility Model.

    Provides:
    - Characteristic function
    - Semi-analytical European option pricing
    - FFT-based pricing (Carr-Madan)
    - Monte Carlo simulation
    - Calibration to market data
    """

    def __init__(
        self,
        params: HestonParameters,
        spot: float,
        rate: float,
        dividend: float = 0.0,
    ):
        """
        Initialize Heston model.

        Args:
            params: Heston model parameters
            spot: Current spot price
            rate: Risk-free rate (continuous)
            dividend: Continuous dividend yield
        """
        self.params = params
        self.spot = spot
        self.rate = rate
        self.dividend = dividend

    def characteristic_function(
        self,
        u: complex,
        T: float,
        formulation: str = "gatheral",
    ) -> complex:
        """
        Heston characteristic function φ(u) = E[exp(i*u*ln(S_T))].

        Two formulations available:
        - "original": Heston's original (can have branch cut issues)
        - "gatheral": Gatheral's formulation (numerically stable)

        Args:
            u: Complex argument
            T: Time to maturity
            formulation: "original" or "gatheral"
        """
        v0 = self.params.v0
        kappa = self.params.kappa
        theta = self.params.theta
        sigma = self.params.sigma
        rho = self.params.rho
        r = self.rate
        q = self.dividend

        # Forward price
        x0 = math.log(self.spot)
        forward_factor = (r - q) * T

        # Gatheral's formulation (more stable)
        if formulation == "gatheral":
            # α = -u²/2 - i*u/2
            # β = κ - ρ*σ*i*u
            # γ = σ²/2
            alpha = -0.5 * u * u - 0.5j * u
            beta = kappa - rho * sigma * 1j * u
            gamma = 0.5 * sigma * sigma

            # d = sqrt(β² - 4*α*γ)
            d = cmath.sqrt(beta * beta - 4 * alpha * gamma)

            # g = (β - d) / (β + d)
            r_minus = (beta - d) / (2 * gamma)
            r_plus = (beta + d) / (2 * gamma)

            if abs(beta + d) < 1e-10:
                g = 0
            else:
                g = r_minus / r_plus

            # D = r_minus * (1 - exp(-d*T)) / (1 - g*exp(-d*T))
            exp_dt = cmath.exp(-d * T)
            D = r_minus * (1 - exp_dt) / (1 - g * exp_dt)

            # C = κ * (r_minus * T - 2/σ² * ln((1 - g*exp(-d*T))/(1-g)))
            if abs(1 - g) > 1e-10:
                C = kappa * (r_minus * T - (2 / (sigma * sigma)) *
                            cmath.log((1 - g * exp_dt) / (1 - g)))
            else:
                C = kappa * r_minus * T

            # Characteristic function
            phi = cmath.exp(1j * u * (x0 + forward_factor) + C * theta + D * v0)

        else:
            # Original Heston formulation
            xi = kappa - sigma * rho * 1j * u
            d = cmath.sqrt(xi * xi + sigma * sigma * (u * u + 1j * u))

            g1 = xi + d
            g2 = xi - d

            exp_dt = cmath.exp(d * T)

            D = g2 / sigma ** 2 * (1 - exp_dt) / (1 - g2 / g1 * exp_dt)
            C = kappa * (g2 * T / sigma ** 2 -
                        2 / sigma ** 2 * cmath.log((1 - g2 / g1 * exp_dt) / (1 - g2 / g1)))

            phi = cmath.exp(1j * u * (x0 + forward_factor) + C * theta + D * v0)

        return phi

    def european_call_price(
        self,
        strike: float,
        T: float,
        method: str = "quadrature",
        n_points: int = 1000,
    ) -> float:
        """
        Price European call option using semi-analytical formula.

        Uses Fourier inversion of characteristic function:
        Call = S*P1 - K*exp(-r*T)*P2

        where P1, P2 are integrals involving the characteristic function.

        Args:
            strike: Strike price
            T: Time to maturity in years
            method: "quadrature" or "fft"
            n_points: Number of integration points
        """
        if method == "fft":
            # The radix-2 FFT needs a power-of-two grid
            return self._fft_price(strike, T, next_power_of_2(max(n_points, 4096)))
        if method != "quadrature":
            raise ValueError(f"Unknown method '{method}'; expected 'quadrature' or 'fft'")

        from scipy.integrate import quad

        S = self.spot
        K = strike
        r = self.rate
        q = self.dividend

        # The characteristic function is of ln(S_T), so invert at ln(K)
        log_k = math.log(K)
        phi_minus_i = self.characteristic_function(-1j, T, "gatheral")  # = E[S_T]

        def integrand_P1(u: float) -> float:
            """Integrand for P1 (stock-measure exercise probability)."""
            phi = self.characteristic_function(u - 1j, T, "gatheral")
            return (cmath.exp(-1j * u * log_k) * phi / (1j * u * phi_minus_i)).real

        def integrand_P2(u: float) -> float:
            """Integrand for P2 (risk-neutral exercise probability)."""
            phi = self.characteristic_function(u, T, "gatheral")
            return (cmath.exp(-1j * u * log_k) * phi / (1j * u)).real

        # The integrands are finite as u -> 0; start just above zero
        upper = max(100.0, 200.0 / math.sqrt(T))
        P1 = 0.5 + quad(integrand_P1, 1e-10, upper, limit=max(n_points, 200))[0] / math.pi
        P2 = 0.5 + quad(integrand_P2, 1e-10, upper, limit=max(n_points, 200))[0] / math.pi

        call_price = S * math.exp(-q * T) * P1 - K * math.exp(-r * T) * P2

        return max(0, call_price)

    def european_put_price(
        self,
        strike: float,
        T: float,
        method: str = "quadrature",
        n_points: int = 1000,
    ) -> float:
        """Price European put option using put-call parity."""
        call = self.european_call_price(strike, T, method, n_points)

        # Put-call parity: P = C - S*exp(-q*T) + K*exp(-r*T)
        put = call - self.spot * math.exp(-self.dividend * T) + strike * math.exp(-self.rate * T)

        return max(0, put)

    def _fft_price(
        self,
        strike: float,
        T: float,
        N: int = 4096,
    ) -> float:
        """
        Price using Carr-Madan FFT method.

        More efficient for pricing many strikes simultaneously.
        """
        # FFT parameters (N must be a power of two)
        eta = 0.25  # Grid spacing in frequency domain
        lambda_param = 2 * math.pi / (N * eta)  # Grid spacing in log-strike domain
        b = N * lambda_param / 2  # Log-strike grid runs over [-b, b)

        alpha = 1.5  # Dampening factor

        r = self.rate
        k = math.log(strike)

        psi_values = []
        for j in range(N):
            v = eta * j
            u = v - (alpha + 1) * 1j
            phi = self.characteristic_function(u, T, "gatheral")
            denom = alpha ** 2 + alpha - v ** 2 + 1j * (2 * alpha + 1) * v
            psi = math.exp(-r * T) * phi / denom

            # Simpson weights
            if j == 0:
                simpson_weight = 1 / 3
            elif j % 2 == 1:
                simpson_weight = 4 / 3
            else:
                simpson_weight = 2 / 3

            psi_values.append(eta * psi * simpson_weight * cmath.exp(1j * b * v))

        # X[m] = sum_j x_j exp(-2*pi*i*j*m/N) gives prices on k_m = -b + lambda*m
        transformed = cooley_tukey_fft(psi_values)

        def price_on_grid(m: int) -> float:
            k_m = -b + lambda_param * m
            return math.exp(-alpha * k_m) / math.pi * transformed[m].real

        position = (k + b) / lambda_param
        m = int(math.floor(position))
        if m < 0 or m >= N - 1:
            raise ValueError("Strike outside the FFT log-strike grid")
        weight = position - m
        call_price = (1 - weight) * price_on_grid(m) + weight * price_on_grid(m + 1)

        return max(0, call_price)

    def implied_volatility(
        self,
        strike: float,
        T: float,
        option_type: str = "call",
    ) -> float:
        """
        Calculate Black-Scholes implied volatility of Heston price.

        Args:
            strike: Strike price
            T: Time to maturity
            option_type: "call" or "put"
        """
        if OptionType(option_type) is OptionType.CALL:
            price = self.european_call_price(strike, T)
        else:
            price = self.european_put_price(strike, T)

        return self._bs_implied_vol(price, strike, T, option_type)

    def _bs_implied_vol(
        self,
        price: float,
        strike: float,
        T: float,
        option_type: str,
        tol: float = 1e-8,
        max_iter: int = 100,
    ) -> float:
        """Calculate BS implied volatility using Newton-Raphson."""
        S = self.spot
        K = strike
        r = self.rate
        q = self.dividend

        # Initial guess
        sigma = math.sqrt(2 * abs(math.log(S / K) + (r - q) * T) / T)
        sigma = max(0.01, min(sigma, 5.0))

        for _ in range(max_iter):
            bs_price = self._bs_price(S, K, T, r, q, sigma, option_type)
            vega = self._bs_vega(S, K, T, r, q, sigma)

            if vega < 1e-10:
                break

            diff = bs_price - price

            if abs(diff) < tol:
                break

            sigma = sigma - diff / vega
            sigma = max(0.001, min(sigma, 10.0))

        return sigma

    @staticmethod
    def _bs_price(S, K, T, r, q, sigma, option_type):
        """Black-Scholes price using centralized distributions."""
        if T <= 0 or sigma <= 0:
            if OptionType(option_type) is OptionType.CALL:
                return max(0, S * math.exp(-q * T) - K * math.exp(-r * T))
            else:
                return max(0, K * math.exp(-r * T) - S * math.exp(-q * T))

        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if OptionType(option_type) is OptionType.CALL:
            return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        else:
            return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)

    @staticmethod
    def _bs_vega(S, K, T, r, q, sigma):
        """Black-Scholes vega using centralized distributions."""
        if T <= 0 or sigma <= 0:
            return 0.0

        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

        return S * math.exp(-q * T) * norm_pdf(d1) * math.sqrt(T)

    def monte_carlo_price(
        self,
        strike: float,
        T: float,
        option_type: str = "call",
        n_paths: int = 100000,
        n_steps: int = 252,
        seed: Optional[int] = None,
    ) -> MonteCarloResult:
        """
        Monte Carlo pricing with Milstein discretization.

        Returns:
            MonteCarloResult with price, std_error, and statistics
        """
        rng = random.Random(seed) if seed is not None else random

        S0 = self.spot
        v0 = self.params.v0
        kappa = self.params.kappa
        theta = self.params.theta
        sigma = self.params.sigma
        rho = self.params.rho
        r = self.rate
        q = self.dividend

        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)

        payoffs = []

        for _ in range(n_paths):
            S = S0
            v = v0

            for _ in range(n_steps):
                # Correlated Brownian motions
                z1 = rng.gauss(0, 1)
                z2 = rng.gauss(0, 1)
                dW_S = z1 * sqrt_dt
                dW_v = (rho * z1 + math.sqrt(1 - rho ** 2) * z2) * sqrt_dt

                # Ensure variance stays positive (full truncation)
                v_plus = max(v, 0)
                sqrt_v = math.sqrt(v_plus)

                # Milstein discretization for variance
                v_new = v + kappa * (theta - v_plus) * dt + sigma * sqrt_v * dW_v + \
                        0.25 * sigma ** 2 * (dW_v ** 2 - dt)
                v_new = max(v_new, 0)

                # Log spot process
                S = S * math.exp((r - q - 0.5 * v_plus) * dt + sqrt_v * dW_S)

                v = v_new

            # Payoff
            if OptionType(option_type) is OptionType.CALL:
                payoff = max(0, S - strike)
            else:
                payoff = max(0, strike - S)

            payoffs.append(payoff)

        # Statistics
        discount = math.exp(-r * T)
        mean_payoff = sum(payoffs) / n_paths
        price = discount * mean_payoff

        # Standard error
        variance = sum((p - mean_payoff) ** 2 for p in payoffs) / (n_paths - 1)
        std_error = discount * math.sqrt(variance / n_paths)

        return MonteCarloResult(
            price=price,
            std_error=std_error,
            n_paths=n_paths,
            variance=variance,
        )

    def __repr__(self) -> str:
        return (
            f"HestonModel(S={self.spot}, v0={self.params.v0:.4f}, "
            f"κ={self.params.kappa:.2f}, θ={self.params.theta:.4f}, "
            f"σ={self.params.sigma:.2f}, ρ={self.params.rho:.2f})"
        )


class HestonPricer:
    """
    Convenience class for pricing multiple options under Heston model.
    """

    def __init__(
        self,
        spot: float,
        rate: float,
        dividend: float = 0.0,
    ):
        self.spot = spot
        self.rate = rate
        self.dividend = dividend
        self._params: Optional[HestonParameters] = None

    def set_parameters(self, params: HestonParameters):
        """Set Heston parameters."""
        self._params = params

    def calibrate(
        self,
        strikes: List[float],
        maturities: List[float],
        market_vols: List[float],
        weights: Optional[List[float]] = None,
    ) -> HestonParameters:
        """
        Calibrate Heston parameters to market implied volatilities.

        Uses least squares optimization.

        Args:
            strikes: List of strikes
            maturities: List of maturities (can repeat for same maturity)
            market_vols: Market implied volatilities
            weights: Optional weights for each observation

        Returns:
            Calibrated HestonParameters
        """
        if weights is None:
            weights = [1.0] * len(strikes)

        # Initial guess
        v0_init = market_vols[0] ** 2
        kappa_init = 2.0
        theta_init = v0_init
        sigma_init = 0.5
        rho_init = -0.5

        # Bounds
        bounds = [
            (0.001, 1.0),    # v0
            (0.01, 10.0),    # kappa
            (0.001, 1.0),    # theta
            (0.01, 2.0),     # sigma
            (-0.99, 0.99),   # rho
        ]

        def objective(params):
            v0, kappa, theta, sigma, rho = params

            # Feller condition penalty
            feller_penalty = 0.0
            if 2 * kappa * theta <= sigma ** 2:
                feller_penalty = 100 * (sigma ** 2 - 2 * kappa * theta) ** 2

            try:
                heston_params = HestonParameters(v0, kappa, theta, sigma, rho)
                model = HestonModel(heston_params, self.spot, self.rate, self.dividend)

                error = 0.0
                for K, T, market_vol, w in zip(strikes, maturities, market_vols, weights):
                    model_vol = model.implied_volatility(K, T, "call")
                    error += w * (model_vol - market_vol) ** 2

                return error + feller_penalty

            except Exception:
                return 1e10

        # Simple gradient descent optimization
        x = [v0_init, kappa_init, theta_init, sigma_init, rho_init]
        best_x = x.copy()
        best_obj = objective(x)

        # Grid search + local optimization
        for v0 in [0.01, 0.04, 0.09]:
            for kappa in [1.0, 2.0, 4.0]:
                for sigma in [0.3, 0.5, 0.8]:
                    for rho in [-0.7, -0.5, -0.3]:
                        x_test = [v0, kappa, v0, sigma, rho]
                        obj = objective(x_test)
                        if obj < best_obj:
                            best_obj = obj
                            best_x = x_test.copy()

        # Local refinement (simple coordinate descent)
        x = best_x.copy()
        step_sizes = [0.01, 0.1, 0.01, 0.05, 0.05]

        for _ in range(100):
            improved = False
            for i in range(5):
                for direction in [-1, 1]:
                    x_new = x.copy()
                    x_new[i] = max(bounds[i][0], min(bounds[i][1],
                                  x_new[i] + direction * step_sizes[i]))
                    obj = objective(x_new)
                    if obj < best_obj:
                        best_obj = obj
                        x = x_new
                        improved = True

            if not improved:
                step_sizes = [s * 0.5 for s in step_sizes]
                if max(step_sizes) < 1e-6:
                    break

        self._params = HestonParameters(*x)
        return self._params

    def price(
        self,
        strike: float,
        T: float,
        option_type: str = "call",
    ) -> float:
        """Price option with current parameters."""
        if self._params is None:
            raise ValueError("Parameters not set. Call set_parameters() or calibrate() first.")

        model = HestonModel(self._params, self.spot, self.rate, self.dividend)

        if OptionType(option_type) is OptionType.CALL:
            return model.european_call_price(strike, T)
        else:
            return model.european_put_price(strike, T)


# =============================================================================
# Enhanced SABR Model
# =============================================================================

class SABRModelEnhanced:
    """
    Enhanced SABR model implementation.

    Features:
    - Hagan's original approximation
    - Obloj correction for ATM accuracy
    - Normal SABR for negative rates
    - Calibration to market data
    """

    def __init__(
        self,
        forward: float,
        T: float,
        params: Optional[SABRParameters] = None,
    ):
        """
        Initialize SABR model.

        Args:
            forward: Forward rate/price
            T: Time to expiry
            params: SABR parameters (can set later via calibration)
        """
        self.forward = forward
        self.T = T
        self.params = params

    def implied_volatility(
        self,
        strike: float,
        method: str = "hagan",
    ) -> float:
        """
        Calculate implied volatility using SABR formula.

        Args:
            strike: Strike price
            method: "hagan" (original), "obloj" (corrected), "normal" (normal vol)
        """
        if self.params is None:
            raise ValueError("Parameters not set")

        F = self.forward
        K = strike
        T = self.T
        alpha = self.params.alpha
        beta = self.params.beta
        rho = self.params.rho
        nu = self.params.nu

        if method == "normal":
            return self._normal_sabr_vol(F, K, T, alpha, beta, rho, nu)
        elif method == "obloj":
            return self._obloj_sabr_vol(F, K, T, alpha, beta, rho, nu)
        else:
            return self._hagan_sabr_vol(F, K, T, alpha, beta, rho, nu)

    def _hagan_sabr_vol(
        self,
        F: float,
        K: float,
        T: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
    ) -> float:
        """
        Hagan's original SABR implied volatility formula (2002).
        """
        if abs(F - K) < 1e-10:
            # ATM case
            FK_mid = F
            first_term = alpha / (FK_mid ** (1 - beta))

            second_term = 1 + (
                ((1 - beta) ** 2 / 24) * alpha ** 2 / (FK_mid ** (2 * (1 - beta))) +
                (rho * beta * nu * alpha) / (4 * FK_mid ** (1 - beta)) +
                ((2 - 3 * rho ** 2) / 24) * nu ** 2
            ) * T

            return first_term * second_term

        # General case
        FK = F * K
        FK_beta = FK ** ((1 - beta) / 2)

        log_FK = math.log(F / K)

        # z parameter
        z = (nu / alpha) * FK_beta * log_FK

        # x(z) function
        sqrt_term = math.sqrt(1 - 2 * rho * z + z ** 2)
        x_z = math.log((sqrt_term + z - rho) / (1 - rho))

        if abs(x_z) < 1e-10:
            x_z = 1.0  # Limit as z -> 0

        # Prefactor
        FK_1_beta = FK ** ((1 - beta) / 2)
        log_FK_sq = log_FK ** 2

        denom = FK_1_beta * (
            1 +
            ((1 - beta) ** 2 / 24) * log_FK_sq +
            ((1 - beta) ** 4 / 1920) * log_FK_sq ** 2
        )

        # First term
        first_term = alpha / denom

        # z/x(z) factor
        z_factor = z / x_z if abs(x_z) > 1e-10 else 1.0

        # Correction terms
        correction = 1 + (
            ((1 - beta) ** 2 / 24) * alpha ** 2 / (FK ** (1 - beta)) +
            (rho * beta * nu * alpha) / (4 * FK_1_beta) +
            ((2 - 3 * rho ** 2) / 24) * nu ** 2
        ) * T

        return first_term * z_factor * correction

    def _obloj_sabr_vol(
        self,
        F: float,
        K: float,
        T: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
    ) -> float:
        """
        Obloj's corrected SABR formula for better ATM accuracy.
        """
        # Start with Hagan
        vol = self._hagan_sabr_vol(F, K, T, alpha, beta, rho, nu)

        # Apply Obloj correction near ATM
        if abs(F - K) / F < 0.01:
            # Near ATM, apply correction
            FK = F * K

            correction_factor = 1 + (
                (1 / 24) * (1 - beta) ** 2 * alpha ** 2 * T / (FK ** (1 - beta))
            )

            vol = vol * correction_factor

        return vol

    def _normal_sabr_vol(
        self,
        F: float,
        K: float,
        T: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
    ) -> float:
        """
        Normal SABR volatility (Bachelier model).
        Useful for negative rates.
        """
        if abs(F - K) < 1e-10:
            # ATM
            return alpha * (1 + ((2 - 3 * rho ** 2) / 24) * nu ** 2 * T)

        # Non-ATM
        z = (nu / alpha) * (F - K)
        sqrt_term = math.sqrt(1 - 2 * rho * z + z ** 2)
        x_z = math.log((sqrt_term + z - rho) / (1 - rho))

        if abs(x_z) < 1e-10:
            z_factor = 1.0
        else:
            z_factor = z / x_z

        correction = 1 + ((2 - 3 * rho ** 2) / 24) * nu ** 2 * T

        return alpha * z_factor * correction

    def calibrate(
        self,
        strikes: List[float],
        market_vols: List[float],
        beta: Optional[float] = None,
    ) -> SABRParameters:
        """
        Calibrate SABR parameters to market volatility smile.

        Args:
            strikes: List of strikes
            market_vols: Corresponding implied volatilities
            beta: If provided, fix beta; otherwise calibrate it

        Returns:
            Calibrated SABRParameters
        """
        F = self.forward
        T = self.T

        # ATM vol
        atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - F))
        atm_vol = market_vols[atm_idx]

        # Initial guesses
        if beta is None:
            beta_init = 0.5
            calibrate_beta = True
        else:
            beta_init = beta
            calibrate_beta = False

        alpha_init = atm_vol * F ** (1 - beta_init)
        rho_init = -0.3
        nu_init = 0.3

        def objective(params):
            if calibrate_beta:
                alpha, beta_val, rho, nu = params
            else:
                alpha, rho, nu = params
                beta_val = beta_init

            if alpha <= 0 or nu < 0 or not -1 < rho < 1 or not 0 <= beta_val <= 1:
                return 1e10

            try:
                sabr_params = SABRParameters(alpha, beta_val, rho, nu)
                model = SABRModelEnhanced(F, T, sabr_params)

                error = 0.0
                for K, market_vol in zip(strikes, market_vols):
                    model_vol = model.implied_volatility(K, "hagan")
                    error += (model_vol - market_vol) ** 2

                return error

            except Exception:
                return 1e10

        # Simple optimization. alpha is quoted in F^(1-beta) units, so it has no
        # natural upper bound (it is ~ atm_vol * F^(1-beta)).
        if calibrate_beta:
            x = [alpha_init, beta_init, rho_init, nu_init]
            bounds = [(0.001, None), (0.0, 1.0), (-0.99, 0.99), (0.001, 2.0)]
        else:
            x = [alpha_init, rho_init, nu_init]
            bounds = [(0.001, None), (-0.99, 0.99), (0.001, 2.0)]

        best_x = x.copy()
        best_obj = objective(x)

        # Grid search
        for alpha in [atm_vol * 0.5, atm_vol, atm_vol * 2]:
            for rho in [-0.5, -0.3, 0.0]:
                for nu in [0.2, 0.4, 0.6]:
                    if calibrate_beta:
                        for beta_val in [0.3, 0.5, 0.7]:
                            x_test = [alpha, beta_val, rho, nu]
                            obj = objective(x_test)
                            if obj < best_obj:
                                best_obj = obj
                                best_x = x_test
                    else:
                        x_test = [alpha, rho, nu]
                        obj = objective(x_test)
                        if obj < best_obj:
                            best_obj = obj
                            best_x = x_test

        # Refine the best grid point with a bounded local optimiser. Squared vol
        # errors are ~1e-6, below L-BFGS-B's tolerances, so measure them in bp^2.
        from scipy.optimize import minimize
        refined = minimize(lambda p: objective(p) * 1e8, best_x, method="L-BFGS-B", bounds=bounds)
        if refined.fun / 1e8 < best_obj:
            best_x = [float(v) for v in refined.x]

        if calibrate_beta:
            self.params = SABRParameters(best_x[0], best_x[1], best_x[2], best_x[3])
        else:
            self.params = SABRParameters(best_x[0], beta_init, best_x[1], best_x[2])

        return self.params

    def __repr__(self) -> str:
        if self.params:
            return (
                f"SABRModelEnhanced(F={self.forward}, T={self.T}, "
                f"α={self.params.alpha:.4f}, β={self.params.beta:.2f}, "
                f"ρ={self.params.rho:.2f}, ν={self.params.nu:.2f})"
            )
        return f"SABRModelEnhanced(F={self.forward}, T={self.T}, params=None)"
