"""
Local Volatility Models

This module implements local volatility models:
- Dupire Local Volatility: Extract local vol from market prices
- Local Volatility Surface: Interpolation and calibration

The local volatility model assumes:
dS/S = (r - q)dt + σ(S,t)dW

where σ(S,t) is a deterministic function of spot and time.

Key concepts:
- Dupire formula: Extracts local vol from European option prices
- Arbitrage-free surface construction
- Monte Carlo with local vol

References:
- Dupire, B. (1994). "Pricing with a Smile." Risk, 7, 18-20.
- Gatheral, J. (2006). "The Volatility Surface."
"""

from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple, Dict
from bisect import bisect_right
import math
import random

from finceptquantlib.core.foundations.types import OptionType
from finceptquantlib.core.utilities.distributions import norm_cdf, norm_pdf
from finceptquantlib.core.utilities.results import MonteCarloResult, SimulationResult


@dataclass
class LocalVolSurface:
    """
    Local volatility surface representation.

    Stores local volatility values on a grid of (strike, time) points
    and provides interpolation methods.

    Attributes:
        strikes: List of strike prices
        maturities: List of maturities (in years)
        local_vols: 2D list of local volatilities [maturity_idx][strike_idx]
        spot: Reference spot price
    """
    strikes: List[float]
    maturities: List[float]
    local_vols: List[List[float]]
    spot: float

    def __post_init__(self):
        # Validate dimensions
        n_mats = len(self.maturities)
        n_strikes = len(self.strikes)

        if len(self.local_vols) != n_mats:
            raise ValueError(f"local_vols has {len(self.local_vols)} rows, expected {n_mats}")

        for i, row in enumerate(self.local_vols):
            if len(row) != n_strikes:
                raise ValueError(f"Row {i} has {len(row)} values, expected {n_strikes}")

    def get_local_vol(self, S: float, t: float) -> float:
        """
        Get interpolated local volatility at (S, t).

        Uses bilinear interpolation on the grid.

        Args:
            S: Spot price
            t: Time

        Returns:
            Local volatility
        """
        if t <= 0:
            # Return ATM vol at first maturity
            return self._interpolate_strike(S, 0)

        # Find time bracket using bisect
        t_idx = bisect_right(self.maturities, t)
        if t_idx >= len(self.maturities):
            t_idx = len(self.maturities) - 1

        if t_idx == 0:
            return self._interpolate_strike(S, 0)

        # Time interpolation weights
        t_lower = self.maturities[t_idx - 1]
        t_upper = self.maturities[t_idx]
        t_weight = (t - t_lower) / (t_upper - t_lower) if t_upper > t_lower else 0

        # Interpolate at both time levels
        vol_lower = self._interpolate_strike(S, t_idx - 1)
        vol_upper = self._interpolate_strike(S, t_idx)

        return vol_lower * (1 - t_weight) + vol_upper * t_weight

    def _interpolate_strike(self, S: float, t_idx: int) -> float:
        """Interpolate local vol at given time index across strikes."""
        vols = self.local_vols[t_idx]

        # Find strike bracket using bisect
        k_idx = bisect_right(self.strikes, S)
        if k_idx >= len(self.strikes):
            k_idx = len(self.strikes) - 1

        if k_idx == 0:
            return vols[0]

        # Linear interpolation
        K_lower = self.strikes[k_idx - 1]
        K_upper = self.strikes[k_idx]
        k_weight = (S - K_lower) / (K_upper - K_lower) if K_upper > K_lower else 0

        return vols[k_idx - 1] * (1 - k_weight) + vols[k_idx] * k_weight

    def to_implied_vol_surface(
        self,
        r: float = 0.0,
        q: float = 0.0,
        n_paths: int = 50000,
        n_steps_per_year: int = 252
    ) -> 'LocalVolSurface':
        """
        Convert local vol surface to implied vol surface via Monte Carlo.

        Returns a new surface with the same grid but implied vols.
        """
        implied_vols = []

        for t_idx, T in enumerate(self.maturities):
            row = []
            for K in self.strikes:
                # Price option via MC
                mc_result = self._mc_price(K, T, r, q, n_paths, int(T * n_steps_per_year))

                # Convert to implied vol
                iv = self._implied_vol_from_price(self.spot, K, T, r, q, mc_result.price)
                row.append(iv)

            implied_vols.append(row)

        return LocalVolSurface(
            strikes=self.strikes.copy(),
            maturities=self.maturities.copy(),
            local_vols=implied_vols,  # Actually implied vols
            spot=self.spot
        )

    def _mc_price(
        self,
        K: float,
        T: float,
        r: float,
        q: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None,
        option_type: str = "call"
    ) -> MonteCarloResult:
        """Monte Carlo option price using local vol."""
        rng = random.Random(seed) if seed is not None else random

        dt = T / max(n_steps, 1)
        payoffs = []

        for _ in range(n_paths):
            S = self.spot

            for step in range(n_steps):
                t = step * dt
                sigma = self.get_local_vol(S, t)

                dW = math.sqrt(dt) * rng.gauss(0, 1)
                S *= math.exp((r - q - 0.5 * sigma ** 2) * dt + sigma * dW)

            if option_type == "put":
                payoff = max(K - S, 0)
            else:
                payoff = max(S - K, 0)
            payoffs.append(payoff)

        mean_payoff = sum(payoffs) / n_paths
        price = math.exp(-r * T) * mean_payoff
        variance = sum((p - mean_payoff) ** 2 for p in payoffs) / (n_paths - 1)
        std_error = math.exp(-r * T) * math.sqrt(variance / n_paths)

        return MonteCarloResult(
            price=price,
            std_error=std_error,
            n_paths=n_paths,
            variance=variance,
        )

    def _implied_vol_from_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        price: float
    ) -> float:
        """Newton-Raphson implied vol calculation."""
        sigma = 0.2

        for _ in range(50):
            bs_price = self._black_scholes(S, K, T, r, q, sigma)
            vega = self._black_scholes_vega(S, K, T, r, q, sigma)

            if abs(vega) < 1e-10:
                break

            diff = bs_price - price
            if abs(diff) < 1e-10:
                break

            sigma -= diff / vega
            sigma = max(0.001, min(sigma, 5.0))

        return sigma

    def _black_scholes(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float
    ) -> float:
        """Black-Scholes call price using centralized distributions."""
        if sigma <= 0 or T <= 0:
            return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0)

        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

    def _black_scholes_vega(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float
    ) -> float:
        """Black-Scholes vega using centralized distributions."""
        if sigma <= 0 or T <= 0:
            return 0.0

        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return S * math.exp(-q * T) * math.sqrt(T) * norm_pdf(d1)


class DupireLocalVol:
    """
    Dupire Local Volatility Model.

    Extracts local volatility from European option prices using:

    σ²_loc(K,T) = [∂C/∂T + (r-q)K·∂C/∂K + qC] / [0.5K²·∂²C/∂K²]

    Or in terms of implied volatility:

    σ²_loc = [σ²_imp + 2σ_imp·T·(∂σ/∂T + (r-q)K·∂σ/∂K)] /
             [(1 + K·d1·√T·∂σ/∂K)² + K²·T·σ_imp·(∂²σ/∂K² - d1·√T·(∂σ/∂K)²)]

    Key applications:
    - Calibrate to market smile
    - Price exotic options consistently with vanillas
    - Generate arbitrage-free volatility surfaces
    """

    def __init__(
        self,
        spot: float,
        r: float = 0.0,
        q: float = 0.0,
        implied_vol_func: Optional[Callable[[float, float], float]] = None
    ):
        """
        Initialize Dupire model.

        Args:
            spot: Current spot price
            r: Risk-free rate
            q: Dividend yield
            implied_vol_func: Function(K, T) returning implied volatility
        """
        self.spot = spot
        self.r = r
        self.q = q
        self._implied_vol_func = implied_vol_func

    def set_implied_vol_surface(
        self,
        implied_vol_func: Callable[[float, float], float]
    ):
        """Set the implied volatility function."""
        self._implied_vol_func = implied_vol_func

    def implied_vol(self, K: float, T: float) -> float:
        """Get implied volatility at (K, T)."""
        if self._implied_vol_func is None:
            raise ValueError("Implied volatility function not set")
        return self._implied_vol_func(K, T)

    def local_vol_from_implied(
        self,
        K: float,
        T: float,
        eps_K: float = 0.01,
        eps_T: float = 0.01
    ) -> float:
        """
        Compute local volatility from implied volatility surface.

        Uses the Dupire formula in terms of implied vol:

        σ²_loc(K,T) = σ²_imp × [∂σ/∂T term] / [strike convexity term]

        Args:
            K: Strike price
            T: Time to maturity
            eps_K: Strike perturbation for derivatives
            eps_T: Time perturbation for derivatives

        Returns:
            Local volatility at (K, T)
        """
        if T <= eps_T:
            # At T=0, local vol = implied vol
            return self.implied_vol(K, max(T, 0.001))

        sigma = self.implied_vol(K, T)

        # Partial derivatives (numerical)
        # ∂σ/∂T
        sigma_T_up = self.implied_vol(K, T + eps_T)
        sigma_T_down = self.implied_vol(K, max(T - eps_T, 0.001))
        dσ_dT = (sigma_T_up - sigma_T_down) / (2 * eps_T)

        # ∂σ/∂K
        sigma_K_up = self.implied_vol(K + eps_K, T)
        sigma_K_down = self.implied_vol(max(K - eps_K, 0.01), T)
        dσ_dK = (sigma_K_up - sigma_K_down) / (2 * eps_K)

        # ∂²σ/∂K²
        d2σ_dK2 = (sigma_K_up - 2 * sigma + sigma_K_down) / (eps_K ** 2)

        # Forward and d1
        F = self.spot * math.exp((self.r - self.q) * T)
        sqrt_T = math.sqrt(T)

        if sigma * sqrt_T < 1e-10:
            return sigma

        d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)

        # Numerator: σ² + 2σT(∂σ/∂T + (r-q)K·∂σ/∂K)
        numerator = sigma ** 2 + 2 * sigma * T * (
            dσ_dT + (self.r - self.q) * K * dσ_dK
        )

        # Denominator terms
        term1 = (1 + K * d1 * sqrt_T * dσ_dK) ** 2
        term2 = K ** 2 * T * sigma * (d2σ_dK2 - d1 * sqrt_T * dσ_dK ** 2)
        denominator = term1 + term2

        if denominator <= 0:
            # Arbitrage - return implied vol
            return sigma

        local_var = numerator / denominator

        if local_var <= 0:
            return sigma

        return math.sqrt(local_var)

    def local_vol_from_prices(
        self,
        call_prices: Callable[[float, float], float],
        K: float,
        T: float,
        eps_K: Optional[float] = None,
        eps_T: float = 0.01
    ) -> float:
        """
        Compute local volatility directly from call prices.

        Uses Dupire's original formula:
        σ²_loc = [∂C/∂T + (r-q)K·∂C/∂K + qC] / [0.5K²·∂²C/∂K²]

        Args:
            call_prices: Function(K, T) returning undiscounted call prices
            K: Strike price
            T: Time to maturity
            eps_K: Strike perturbation (defaults to 1% of K)
            eps_T: Time perturbation

        Returns:
            Local volatility at (K, T)
        """
        if eps_K is None:
            eps_K = 0.01 * K

        if T <= eps_T:
            return 0.2  # Default

        C = call_prices(K, T)

        # ∂C/∂T
        C_T_up = call_prices(K, T + eps_T)
        C_T_down = call_prices(K, max(T - eps_T, 0.001))
        dC_dT = (C_T_up - C_T_down) / (2 * eps_T)

        # ∂C/∂K
        C_K_up = call_prices(K + eps_K, T)
        C_K_down = call_prices(max(K - eps_K, 0.01), T)
        dC_dK = (C_K_up - C_K_down) / (2 * eps_K)

        # ∂²C/∂K²
        d2C_dK2 = (C_K_up - 2 * C + C_K_down) / (eps_K ** 2)

        # Dupire formula
        numerator = dC_dT + (self.r - self.q) * K * dC_dK + self.q * C
        denominator = 0.5 * K ** 2 * d2C_dK2

        if denominator <= 0:
            return 0.2  # Arbitrage or numerical issue

        local_var = numerator / denominator

        if local_var <= 0:
            return 0.2

        return math.sqrt(local_var)

    def build_local_vol_surface(
        self,
        strikes: List[float],
        maturities: List[float]
    ) -> LocalVolSurface:
        """
        Build a local volatility surface on a grid.

        Args:
            strikes: List of strikes
            maturities: List of maturities

        Returns:
            LocalVolSurface object
        """
        local_vols = []

        for T in maturities:
            row = []
            for K in strikes:
                try:
                    lv = self.local_vol_from_implied(K, T)
                except (ValueError, ZeroDivisionError, OverflowError):
                    lv = self.implied_vol(K, T)
                row.append(lv)
            local_vols.append(row)

        return LocalVolSurface(
            strikes=strikes,
            maturities=maturities,
            local_vols=local_vols,
            spot=self.spot
        )

    def european_option_price(
        self,
        K: float,
        T: float,
        option_type: OptionType = OptionType.CALL,
        n_paths: int = 100000,
        n_steps: int = 252,
        seed: Optional[int] = None
    ) -> MonteCarloResult:
        """
        Price European option using Monte Carlo with local vol.

        Args:
            K: Strike price
            T: Time to maturity
            option_type: CALL or PUT
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            seed: Random seed for reproducibility

        Returns:
            MonteCarloResult with price, std_error, and statistics
        """
        rng = random.Random(seed) if seed is not None else random

        dt = T / n_steps
        payoffs = []

        for _ in range(n_paths):
            S = self.spot

            for step in range(n_steps):
                t = step * dt
                sigma = self.local_vol_from_implied(S, t)

                dW = math.sqrt(dt) * rng.gauss(0, 1)
                S *= math.exp((self.r - self.q - 0.5 * sigma ** 2) * dt + sigma * dW)

            if option_type == OptionType.CALL:
                payoff = max(S - K, 0)
            else:
                payoff = max(K - S, 0)

            payoffs.append(payoff)

        mean_payoff = sum(payoffs) / n_paths
        price = math.exp(-self.r * T) * mean_payoff

        variance = sum((p - mean_payoff) ** 2 for p in payoffs) / (n_paths - 1)
        std_error = math.exp(-self.r * T) * math.sqrt(variance / n_paths)

        return MonteCarloResult(
            price=price,
            std_error=std_error,
            n_paths=n_paths,
            variance=variance,
        )

    def path_simulation(
        self,
        T: float,
        n_paths: int = 10000,
        n_steps: int = 252,
        seed: Optional[int] = None
    ) -> SimulationResult:
        """
        Generate Monte Carlo paths under local vol dynamics.

        Args:
            T: Final time
            n_paths: Number of paths
            n_steps: Number of time steps
            seed: Random seed

        Returns:
            SimulationResult with times, paths, and statistics
        """
        rng = random.Random(seed) if seed is not None else random

        dt = T / n_steps
        times = [i * dt for i in range(n_steps + 1)]
        paths = []

        for _ in range(n_paths):
            path = [self.spot]
            S = self.spot

            for step in range(n_steps):
                t = step * dt
                sigma = self.local_vol_from_implied(S, t)

                dW = math.sqrt(dt) * rng.gauss(0, 1)
                S *= math.exp((self.r - self.q - 0.5 * sigma ** 2) * dt + sigma * dW)

                path.append(S)

            paths.append(path)

        result = SimulationResult(
            times=times,
            paths=paths,
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
        )
        result.compute_statistics()
        return result


class SVI:
    """
    Stochastic Volatility Inspired (SVI) Parameterization.

    The SVI model parameterizes total implied variance as:

    w(k) = a + b × [ρ(k-m) + √((k-m)² + σ²)]

    where:
    - k = log(K/F) is log-moneyness
    - w = σ²_imp × T is total variance
    - a, b, ρ, m, σ are parameters

    Key properties:
    - Simple 5-parameter model
    - Matches observed volatility smiles well
    - Can be made arbitrage-free (SSVI)
    """

    def __init__(
        self,
        a: float,
        b: float,
        rho: float,
        m: float,
        sigma: float
    ):
        """
        Initialize SVI model.

        Args:
            a: Vertical shift (ATM variance level)
            b: Slope multiplier (controls wings)
            rho: Correlation (-1 < ρ < 1)
            m: Horizontal shift
            sigma: Curvature parameter (σ > 0)
        """
        self.a = a
        self.b = b
        self.rho = rho
        self.m = m
        self.sigma = sigma

        self._validate()

    def _validate(self):
        """Check parameter constraints."""
        if self.b < 0:
            raise ValueError("b must be non-negative")
        if abs(self.rho) >= 1:
            raise ValueError("rho must be in (-1, 1)")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.a + self.b * self.sigma * math.sqrt(1 - self.rho ** 2) < 0:
            raise ValueError("Parameters violate no-arbitrage: minimum variance negative")

    def total_variance(self, k: float) -> float:
        """
        Total implied variance w(k) = σ²_imp × T.

        Args:
            k: Log-moneyness ln(K/F)

        Returns:
            Total variance
        """
        return self.a + self.b * (
            self.rho * (k - self.m) +
            math.sqrt((k - self.m) ** 2 + self.sigma ** 2)
        )

    def implied_vol(self, K: float, F: float, T: float) -> float:
        """
        Implied volatility at strike K.

        Args:
            K: Strike price
            F: Forward price
            T: Time to maturity

        Returns:
            Implied volatility
        """
        k = math.log(K / F)
        w = self.total_variance(k)

        if w <= 0 or T <= 0:
            return 0.2  # Default

        return math.sqrt(w / T)

    @classmethod
    def calibrate(
        cls,
        strikes: List[float],
        forward: float,
        T: float,
        implied_vols: List[float],
        initial_guess: Optional[Tuple[float, float, float, float, float]] = None
    ) -> 'SVI':
        """
        Calibrate SVI to market implied volatilities.

        Args:
            strikes: List of strikes
            forward: Forward price
            T: Time to maturity
            implied_vols: List of implied volatilities
            initial_guess: Starting parameters (a, b, rho, m, sigma)

        Returns:
            Calibrated SVI model
        """
        # Convert to log-moneyness and total variance
        k_values = [math.log(K / forward) for K in strikes]
        w_values = [iv ** 2 * T for iv in implied_vols]

        # Initial guess
        if initial_guess is None:
            atm_var = w_values[len(w_values) // 2]
            a = atm_var
            b = 0.1
            rho = -0.3
            m = 0.0
            sigma = 0.3
        else:
            a, b, rho, m, sigma = initial_guess

        # Objective function
        def objective(params):
            a, b, rho, m, sigma = params

            # Constraints
            if b < 0 or abs(rho) >= 1 or sigma <= 0:
                return 1e10
            if a + b * sigma * math.sqrt(1 - rho ** 2) < 0:
                return 1e10

            error = 0
            for k, w_market in zip(k_values, w_values):
                w_model = a + b * (rho * (k - m) + math.sqrt((k - m) ** 2 + sigma ** 2))
                error += (w_model - w_market) ** 2

            return error

        # Simple gradient descent
        params = [a, b, rho, m, sigma]
        learning_rate = 0.01

        for _ in range(200):
            grad = []
            for i in range(5):
                eps = 0.001
                params_up = params.copy()
                params_up[i] += eps
                params_down = params.copy()
                params_down[i] -= eps
                grad.append((objective(params_up) - objective(params_down)) / (2 * eps))

            for i in range(5):
                params[i] -= learning_rate * grad[i]

            # Constraints
            params[1] = max(0.001, params[1])  # b > 0
            params[2] = max(-0.99, min(0.99, params[2]))  # -1 < rho < 1
            params[4] = max(0.001, params[4])  # sigma > 0

        return cls(params[0], params[1], params[2], params[3], params[4])


class SSVI:
    """
    Surface SVI (SSVI) - Arbitrage-free volatility surface.

    SSVI extends SVI to a full surface with guaranteed no butterfly
    and calendar spread arbitrage. The parameterization is:

    w(k,θ) = θ/2 × [1 + ρφ(θ)k + √((φ(θ)k + ρ)² + (1-ρ²))]

    where:
    - θ = ATM total variance at time T
    - φ(θ) = η / (θ^γ × (1 + θ)^(1-γ)) controls the wings
    - η and γ are surface parameters

    Key advantage: Automatically arbitrage-free by construction.

    Reference:
        Gatheral, J. and Jacquier, A. (2014). "Arbitrage-free SVI volatility surfaces."
    """

    def __init__(
        self,
        theta_func: Callable[[float], float],
        eta: float,
        gamma: float,
        rho: float
    ):
        """
        Initialize SSVI model.

        Args:
            theta_func: Function returning ATM total variance θ(T)
            eta: Wing parameter (η > 0)
            gamma: Power law parameter (0 < γ < 1)
            rho: Correlation (-1 < ρ < 1)
        """
        self.theta_func = theta_func
        self.eta = eta
        self.gamma = gamma
        self.rho = rho

        self._validate()

    def _validate(self):
        """Check parameter constraints."""
        if self.eta <= 0:
            raise ValueError("eta must be positive")
        if not (0 < self.gamma < 1):
            raise ValueError("gamma must be in (0, 1)")
        if abs(self.rho) >= 1:
            raise ValueError("rho must be in (-1, 1)")

    def phi(self, theta: float) -> float:
        """Wing function φ(θ)."""
        if theta <= 0:
            return self.eta
        return self.eta / (theta ** self.gamma * (1 + theta) ** (1 - self.gamma))

    def total_variance(self, k: float, T: float) -> float:
        """
        Total implied variance w(k, T).

        Args:
            k: Log-moneyness
            T: Time to maturity

        Returns:
            Total variance
        """
        theta = self.theta_func(T)
        if theta <= 0:
            return 0.04 * T  # Default 20% vol

        phi_val = self.phi(theta)

        term1 = phi_val * k + self.rho
        term2 = math.sqrt(term1 ** 2 + (1 - self.rho ** 2))

        return 0.5 * theta * (1 + self.rho * phi_val * k + term2)

    def implied_vol(self, K: float, F: float, T: float) -> float:
        """
        Implied volatility at (K, T).

        Args:
            K: Strike
            F: Forward
            T: Maturity

        Returns:
            Implied volatility
        """
        if T <= 0:
            return 0.2

        k = math.log(K / F)
        w = self.total_variance(k, T)

        if w <= 0:
            return 0.2

        return math.sqrt(w / T)

    def build_implied_vol_surface(
        self,
        spot: float,
        r: float,
        q: float,
        strikes: List[float],
        maturities: List[float]
    ) -> LocalVolSurface:
        """
        Build implied volatility surface on a grid.

        Returns LocalVolSurface with implied vols (not local vols).
        """
        impl_vols = []

        for T in maturities:
            F = spot * math.exp((r - q) * T)
            row = []
            for K in strikes:
                iv = self.implied_vol(K, F, T)
                row.append(iv)
            impl_vols.append(row)

        return LocalVolSurface(
            strikes=strikes,
            maturities=maturities,
            local_vols=impl_vols,
            spot=spot
        )


# Factory functions

def create_flat_local_vol(
    spot: float,
    sigma: float,
    r: float = 0.0,
    q: float = 0.0
) -> DupireLocalVol:
    """Create a constant local vol model."""
    model = DupireLocalVol(spot, r, q)
    model.set_implied_vol_surface(lambda K, T: sigma)
    return model


def create_svi_local_vol(
    spot: float,
    forward_func: Callable[[float], float],
    svi_params_by_maturity: Dict[float, Tuple[float, float, float, float, float]],
    r: float = 0.0,
    q: float = 0.0
) -> DupireLocalVol:
    """
    Create Dupire model with SVI implied vol surface.

    Args:
        spot: Spot price
        forward_func: Function returning forward price for maturity
        svi_params_by_maturity: Dict mapping maturity to (a, b, rho, m, sigma)
        r: Risk-free rate
        q: Dividend yield

    Returns:
        DupireLocalVol model
    """
    maturities = sorted(svi_params_by_maturity.keys())

    def implied_vol_func(K: float, T: float) -> float:
        # Find bracketing maturities
        if T <= maturities[0]:
            params = svi_params_by_maturity[maturities[0]]
            svi = SVI(*params)
            return svi.implied_vol(K, forward_func(maturities[0]), maturities[0])

        if T >= maturities[-1]:
            params = svi_params_by_maturity[maturities[-1]]
            svi = SVI(*params)
            return svi.implied_vol(K, forward_func(maturities[-1]), maturities[-1])

        # Interpolate
        for i, mat in enumerate(maturities[1:], 1):
            if mat >= T:
                T_lower = maturities[i - 1]
                T_upper = mat
                break

        weight = (T - T_lower) / (T_upper - T_lower)

        params_lower = svi_params_by_maturity[T_lower]
        params_upper = svi_params_by_maturity[T_upper]

        svi_lower = SVI(*params_lower)
        svi_upper = SVI(*params_upper)

        iv_lower = svi_lower.implied_vol(K, forward_func(T_lower), T_lower)
        iv_upper = svi_upper.implied_vol(K, forward_func(T_upper), T_upper)

        # Interpolate in variance space
        var_lower = iv_lower ** 2 * T_lower
        var_upper = iv_upper ** 2 * T_upper

        var_interp = var_lower * (1 - weight) + var_upper * weight

        return math.sqrt(var_interp / T) if T > 0 else 0.2

    model = DupireLocalVol(spot, r, q)
    model.set_implied_vol_surface(implied_vol_func)

    return model
