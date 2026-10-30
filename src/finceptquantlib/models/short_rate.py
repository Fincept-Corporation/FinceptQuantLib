"""
Short Rate Models for Interest Rate Derivatives

This module implements classic short rate models:
- Vasicek: Mean-reverting normal model
- Cox-Ingersoll-Ross (CIR): Mean-reverting square-root model
- Hull-White (Extended Vasicek): Time-dependent calibratable model
- Black-Karasinski: Log-normal short rate model

Each model includes:
- Analytical bond pricing (where available)
- Zero-coupon bond option pricing
- Monte Carlo simulation
- Calibration to market data

References:
- Vasicek (1977): "An Equilibrium Characterization of the Term Structure"
- Cox, Ingersoll, Ross (1985): "A Theory of the Term Structure of Interest Rates"
- Hull, White (1990): "Pricing Interest-Rate-Derivative Securities"
- Black, Karasinski (1991): "Bond and Option Pricing when Short Rates are Lognormal"
"""

from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple
import math
import random

from finceptquantlib.core.foundations.types import OptionType
from finceptquantlib.core.utilities.distributions import norm_cdf


@dataclass
class ShortRateParameters:
    """
    Parameters for short rate models.

    The general form is:
    dr = κ(θ - r)dt + σ·f(r)·dW

    where f(r) depends on the model:
    - Vasicek: f(r) = 1 (normal)
    - CIR: f(r) = √r (square-root)
    - Hull-White: f(r) = 1 (time-dependent θ)
    - Black-Karasinski: f(r) = r (log-normal)

    Attributes:
        kappa: Mean reversion speed
        theta: Long-term mean level
        sigma: Volatility
        r0: Initial short rate
    """
    kappa: float
    theta: float
    sigma: float
    r0: float

    def __post_init__(self):
        if self.kappa <= 0:
            raise ValueError("Mean reversion speed κ must be positive")
        if self.sigma <= 0:
            raise ValueError("Volatility σ must be positive")

    @property
    def feller_condition(self) -> bool:
        """
        Check Feller condition for CIR model: 2κθ > σ²

        Ensures the rate stays positive in CIR model.
        """
        return 2 * self.kappa * self.theta > self.sigma ** 2


class VasicekModel:
    """
    Vasicek (1977) Mean-Reverting Short Rate Model.

    Dynamics: dr = κ(θ - r)dt + σdW

    Key properties:
    - Analytical formulas for bonds and bond options
    - Rates can go negative (allows for negative rate environments)
    - Mean-reverting to θ with speed κ
    - Stationary distribution: N(θ, σ²/(2κ))

    Bond price formula:
    P(t,T) = A(t,T) × exp(-B(t,T) × r(t))

    where:
    B(τ) = [1 - exp(-κτ)] / κ
    A(τ) = exp[(B(τ) - τ)(κ²θ - σ²/2)/κ² - σ²B(τ)²/(4κ)]
    """

    def __init__(self, params: ShortRateParameters):
        self.params = params

    def B(self, tau: float) -> float:
        """B(τ) function for bond pricing."""
        kappa = self.params.kappa
        if abs(kappa * tau) < 1e-10:
            return tau
        return (1 - math.exp(-kappa * tau)) / kappa

    def A(self, tau: float) -> float:
        """A(τ) function for bond pricing."""
        kappa = self.params.kappa
        theta = self.params.theta
        sigma = self.params.sigma

        B_tau = self.B(tau)

        term1 = (B_tau - tau) * (kappa ** 2 * theta - 0.5 * sigma ** 2) / kappa ** 2
        term2 = sigma ** 2 * B_tau ** 2 / (4 * kappa)

        return math.exp(term1 - term2)

    def zero_coupon_bond_price(
        self,
        r: float,
        tau: float
    ) -> float:
        """
        Price of a zero-coupon bond paying 1 at time τ.

        P(r, τ) = A(τ) × exp(-B(τ) × r)

        Args:
            r: Current short rate
            tau: Time to maturity

        Returns:
            Bond price
        """
        if tau <= 0:
            return 1.0
        return self.A(tau) * math.exp(-self.B(tau) * r)

    def yield_curve(
        self,
        r: float,
        maturities: List[float]
    ) -> List[float]:
        """
        Generate yield curve at given maturities.

        R(τ) = -ln(P(τ)) / τ
        """
        yields = []
        for tau in maturities:
            if tau <= 0:
                yields.append(r)
            else:
                P = self.zero_coupon_bond_price(r, tau)
                yields.append(-math.log(P) / tau)
        return yields

    def zero_coupon_bond_option(
        self,
        r: float,
        T: float,
        S: float,
        K: float,
        option_type: OptionType = OptionType.CALL
    ) -> float:
        """
        Price of a European option on a zero-coupon bond.

        The option expires at T and the bond matures at S > T.

        Call price:
        C = P(t,S) × N(d₁) - K × P(t,T) × N(d₂)

        where:
        d₁ = [ln(P(t,S)/(K×P(t,T))) + σ_P²/2] / σ_P
        d₂ = d₁ - σ_P
        σ_P = σ × B(S-T) × √[(1-exp(-2κT))/(2κ)]

        Args:
            r: Current short rate
            T: Option expiry
            S: Bond maturity (S > T)
            K: Strike price
            option_type: CALL or PUT

        Returns:
            Option price
        """
        if S <= T:
            raise ValueError("Bond maturity S must be greater than option expiry T")

        kappa = self.params.kappa
        sigma = self.params.sigma

        P_T = self.zero_coupon_bond_price(r, T)
        P_S = self.zero_coupon_bond_price(r, S)

        # Bond price volatility
        B_S_T = self.B(S - T)
        if abs(kappa) < 1e-10:
            sigma_P = sigma * B_S_T * math.sqrt(T)
        else:
            sigma_P = sigma * B_S_T * math.sqrt((1 - math.exp(-2 * kappa * T)) / (2 * kappa))

        if sigma_P < 1e-10:
            # Zero vol case
            if option_type == OptionType.CALL:
                return max(P_S - K * P_T, 0)
            else:
                return max(K * P_T - P_S, 0)

        d1 = (math.log(P_S / (K * P_T)) + 0.5 * sigma_P ** 2) / sigma_P
        d2 = d1 - sigma_P

        if option_type == OptionType.CALL:
            return P_S * norm_cdf(d1) - K * P_T * norm_cdf(d2)
        else:
            return K * P_T * norm_cdf(-d2) - P_S * norm_cdf(-d1)

    def caplet_price(
        self,
        r: float,
        T_start: float,
        T_end: float,
        K: float,
        notional: float = 1.0
    ) -> float:
        """
        Price of an interest rate caplet.

        A caplet pays max(L(T_start, T_end) - K, 0) at T_end,
        where L is the LIBOR/forward rate.

        Using the relationship between caplets and bond puts:
        Caplet = (1 + K×τ) × Put on ZCB

        Args:
            r: Current short rate
            T_start: Caplet start (fixing date)
            T_end: Caplet end (payment date)
            K: Strike rate
            notional: Notional amount

        Returns:
            Caplet price
        """
        tau = T_end - T_start
        bond_strike = 1 / (1 + K * tau)

        # Caplet = (1 + K×τ) × Put(T_start, T_end, 1/(1+K×τ))
        put_price = self.zero_coupon_bond_option(r, T_start, T_end, bond_strike, OptionType.PUT)

        return notional * (1 + K * tau) * put_price

    def floorlet_price(
        self,
        r: float,
        T_start: float,
        T_end: float,
        K: float,
        notional: float = 1.0
    ) -> float:
        """
        Price of an interest rate floorlet.

        Floorlet = (1 + K×τ) × Call on ZCB
        """
        tau = T_end - T_start
        bond_strike = 1 / (1 + K * tau)

        call_price = self.zero_coupon_bond_option(r, T_start, T_end, bond_strike, OptionType.CALL)

        return notional * (1 + K * tau) * call_price

    def monte_carlo_paths(
        self,
        T: float,
        n_paths: int = 10000,
        n_steps: int = 252,
        seed: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generate Monte Carlo paths for the short rate.

        Uses exact simulation:
        r(t+dt) | r(t) ~ N(μ, σ²)
        μ = r(t)exp(-κdt) + θ(1 - exp(-κdt))
        σ² = σ² × (1 - exp(-2κdt)) / (2κ)

        Args:
            T: Final time
            n_paths: Number of paths
            n_steps: Number of time steps
            seed: Random seed

        Returns:
            List of rate paths
        """
        rng = random.Random(seed) if seed is not None else random

        kappa = self.params.kappa
        theta = self.params.theta
        sigma = self.params.sigma
        r0 = self.params.r0

        dt = T / n_steps

        # Exact simulation parameters
        mean_coef = math.exp(-kappa * dt)
        mean_add = theta * (1 - mean_coef)

        if abs(kappa) < 1e-10:
            vol = sigma * math.sqrt(dt)
        else:
            vol = sigma * math.sqrt((1 - math.exp(-2 * kappa * dt)) / (2 * kappa))

        paths = []
        for _ in range(n_paths):
            path = [r0]
            r = r0

            for _ in range(n_steps):
                r = r * mean_coef + mean_add + vol * rng.gauss(0, 1)
                path.append(r)

            paths.append(path)

        return paths


class CIRModel:
    """
    Cox-Ingersoll-Ross (1985) Square-Root Model.

    Dynamics: dr = κ(θ - r)dt + σ√r dW

    Key properties:
    - Rates are always non-negative (if Feller condition holds)
    - Feller condition: 2κθ > σ² ensures positivity
    - Non-central chi-squared distribution for rates
    - Analytical bond pricing formulas

    Bond price formula:
    P(t,T) = A(τ) × exp(-B(τ) × r(t))

    where the functions A and B differ from Vasicek.
    """

    def __init__(self, params: ShortRateParameters):
        self.params = params
        self._gamma_cache: Optional[float] = None
        if not params.feller_condition:
            import warnings
            warnings.warn("Feller condition not satisfied: rates may hit zero", stacklevel=2)

    def _gamma(self) -> float:
        """Helper: γ = √(κ² + 2σ²)"""
        if self._gamma_cache is None:
            kappa = self.params.kappa
            sigma = self.params.sigma
            self._gamma_cache = math.sqrt(kappa ** 2 + 2 * sigma ** 2)
        return self._gamma_cache

    def B(self, tau: float) -> float:
        """B(τ) function for CIR bond pricing."""
        kappa = self.params.kappa
        gamma = self._gamma()

        numerator = 2 * (math.exp(gamma * tau) - 1)
        denominator = (gamma + kappa) * (math.exp(gamma * tau) - 1) + 2 * gamma

        return numerator / denominator

    def A(self, tau: float) -> float:
        """A(τ) function for CIR bond pricing."""
        kappa = self.params.kappa
        theta = self.params.theta
        sigma = self.params.sigma
        gamma = self._gamma()

        inner = 2 * gamma * math.exp((kappa + gamma) * tau / 2)
        denominator = (gamma + kappa) * (math.exp(gamma * tau) - 1) + 2 * gamma

        exponent = 2 * kappa * theta / sigma ** 2

        return (inner / denominator) ** exponent

    def zero_coupon_bond_price(
        self,
        r: float,
        tau: float
    ) -> float:
        """
        Price of a zero-coupon bond in CIR model.

        P(r, τ) = A(τ) × exp(-B(τ) × r)
        """
        if tau <= 0:
            return 1.0

        return self.A(tau) * math.exp(-self.B(tau) * r)

    def yield_curve(
        self,
        r: float,
        maturities: List[float]
    ) -> List[float]:
        """Generate yield curve at given maturities."""
        yields = []
        for tau in maturities:
            if tau <= 0:
                yields.append(r)
            else:
                P = self.zero_coupon_bond_price(r, tau)
                yields.append(-math.log(P) / tau)
        return yields

    def zero_coupon_bond_option(
        self,
        r: float,
        T: float,
        S: float,
        K: float,
        option_type: OptionType = OptionType.CALL
    ) -> float:
        """
        Price of a European option on a zero-coupon bond in CIR.

        Uses the non-central chi-squared distribution.
        Approximation using Jamshidian's formula.

        Args:
            r: Current short rate
            T: Option expiry
            S: Bond maturity
            K: Strike price
            option_type: CALL or PUT

        Returns:
            Option price
        """
        if S <= T:
            raise ValueError("Bond maturity S must be greater than option expiry T")

        kappa = self.params.kappa
        theta = self.params.theta
        sigma = self.params.sigma

        P_T = self.zero_coupon_bond_price(r, T)
        P_S = self.zero_coupon_bond_price(r, S)

        # Find r* such that P(r*, S-T) = K
        # Use Newton-Raphson
        r_star = theta
        for _ in range(20):
            P_val = self.A(S - T) * math.exp(-self.B(S - T) * r_star)
            dP = -self.B(S - T) * P_val

            diff = P_val - K
            if abs(diff) < 1e-10:
                break
            if abs(dP) < 1e-15:
                break
            r_star -= diff / dP
            r_star = max(0.0001, r_star)

        # Chi-squared parameters
        gamma = self._gamma()

        psi = (kappa + gamma) / sigma ** 2
        phi = 2 * gamma / (sigma ** 2 * (math.exp(gamma * T) - 1))

        # Degrees of freedom
        df = 4 * kappa * theta / sigma ** 2

        # Non-centrality parameters
        lambda_S = 2 * phi * r * math.exp(gamma * T) * self.B(S - T) / (self.B(S - T) + psi)
        lambda_T = 2 * phi * r * math.exp(gamma * T)

        # Chi-squared CDFs (approximate using normal for large df)
        x_S = 2 * r_star * (self.B(S - T) + psi)
        x_T = 2 * r_star * psi

        # Use normal approximation for chi-squared
        chi_S = self._chi2_cdf_approx(x_S, df, lambda_S)
        chi_T = self._chi2_cdf_approx(x_T, df, lambda_T)

        if option_type == OptionType.CALL:
            return P_S * (1 - chi_S) - K * P_T * (1 - chi_T)
        else:
            return K * P_T * chi_T - P_S * chi_S

    def monte_carlo_paths(
        self,
        T: float,
        n_paths: int = 10000,
        n_steps: int = 252,
        seed: Optional[int] = None,
        scheme: str = "qe"
    ) -> List[List[float]]:
        """
        Generate Monte Carlo paths for CIR model.

        Schemes:
        - "euler": Euler-Maruyama (with reflection)
        - "milstein": Milstein scheme
        - "qe": Quadratic Exponential (Andersen 2008)

        Args:
            T: Final time
            n_paths: Number of paths
            n_steps: Time steps
            seed: Random seed
            scheme: Discretization scheme

        Returns:
            List of rate paths
        """
        rng = random.Random(seed) if seed is not None else random

        kappa = self.params.kappa
        theta = self.params.theta
        sigma = self.params.sigma
        r0 = self.params.r0

        dt = T / n_steps

        paths = []

        for _ in range(n_paths):
            path = [r0]
            r = r0

            for _ in range(n_steps):
                if scheme == "euler":
                    # Euler with reflection
                    dW = math.sqrt(dt) * rng.gauss(0, 1)
                    r_new = r + kappa * (theta - r) * dt + sigma * math.sqrt(max(r, 0)) * dW
                    r = max(r_new, 0)

                elif scheme == "milstein":
                    # Milstein scheme
                    dW = math.sqrt(dt) * rng.gauss(0, 1)
                    r_new = r + kappa * (theta - r) * dt + sigma * math.sqrt(max(r, 0)) * dW
                    r_new += 0.25 * sigma ** 2 * (dW ** 2 - dt)
                    r = max(r_new, 0)

                elif scheme == "qe":
                    # Quadratic Exponential (simplified version)
                    exp_kdt = math.exp(-kappa * dt)
                    m = theta + (r - theta) * exp_kdt
                    s2 = r * sigma ** 2 * exp_kdt * (1 - exp_kdt) / kappa
                    s2 += theta * sigma ** 2 * (1 - exp_kdt) ** 2 / (2 * kappa)

                    m = max(m, 1e-8)
                    psi = s2 / (m * m)

                    if psi <= 1.5:
                        # Use moment-matched distribution
                        inv_psi = 2.0 / psi
                        b2 = inv_psi - 1 + math.sqrt(inv_psi * (inv_psi - 1))
                        a = m / (1 + b2)
                        Z = rng.gauss(0, 1)
                        r = a * (math.sqrt(b2) + Z) ** 2
                    else:
                        # Use exponential distribution
                        p = (psi - 1) / (psi + 1)
                        beta_qe = (1 - p) / max(m, 1e-15)
                        U = rng.random()
                        if U <= p:
                            r = 0
                        else:
                            r = math.log((1 - p) / max(1 - U, 1e-15)) / beta_qe

                path.append(r)

            paths.append(path)

        return paths

    def _chi2_cdf_approx(self, x: float, df: float, ncp: float) -> float:
        """
        Approximate non-central chi-squared CDF.

        Uses normal approximation for large df.
        """
        # Wilson-Hilferty approximation
        mean = df + ncp
        var = 2 * (df + 2 * ncp)
        std = math.sqrt(var)

        z = (x - mean) / std
        return norm_cdf(z)


class HullWhiteModel:
    """
    Hull-White (1990) Extended Vasicek Model.

    Dynamics: dr = [θ(t) - κr]dt + σdW

    The time-dependent θ(t) allows exact calibration to the initial
    term structure. This is the key advantage over Vasicek.

    θ(t) = (∂f/∂T)(0,t) + κf(0,t) + σ²(1-e^{-2κt})/(2κ)

    where f(0,t) is the instantaneous forward rate at time 0 for date t.

    Key features:
    - Exact fit to initial yield curve
    - Analytical pricing formulas
    - Can price caps, floors, swaptions
    - Rates can go negative
    """

    def __init__(
        self,
        params: ShortRateParameters,
        initial_curve: Optional[Callable[[float], float]] = None
    ):
        """
        Initialize Hull-White model.

        Args:
            params: Model parameters (kappa, sigma, r0)
            initial_curve: Function returning zero rate for given maturity
                          If None, assumes flat curve at theta
        """
        self.params = params
        if initial_curve is None:
            # Flat curve at theta
            self.initial_curve = lambda t: params.theta
        else:
            self.initial_curve = initial_curve

    def market_discount_factor(self, T: float) -> float:
        """
        Market discount factor P^M(0,T) from initial curve.
        """
        if T <= 0:
            return 1.0
        rate = self.initial_curve(T)
        return math.exp(-rate * T)

    def instantaneous_forward(self, t: float, eps: float = 0.0001) -> float:
        """
        Instantaneous forward rate f(0,t).

        f(0,t) = -∂ln(P^M(0,t))/∂t
        """
        P_t = self.market_discount_factor(t)
        P_t_eps = self.market_discount_factor(t + eps)

        if P_t_eps <= 0 or P_t <= 0:
            return self.params.theta

        return -math.log(P_t_eps / P_t) / eps

    def theta_t(self, t: float, eps: float = 0.0001) -> float:
        """
        Time-dependent drift θ(t).

        θ(t) = ∂f/∂t + κf(0,t) + σ²(1-e^{-2κt})/(2κ)
        """
        kappa = self.params.kappa
        sigma = self.params.sigma

        f_t = self.instantaneous_forward(t)
        f_t_eps = self.instantaneous_forward(t + eps)

        df_dt = (f_t_eps - f_t) / eps

        if abs(kappa) < 1e-10:
            term3 = sigma ** 2 * t
        else:
            term3 = sigma ** 2 * (1 - math.exp(-2 * kappa * t)) / (2 * kappa)

        return df_dt + kappa * f_t + term3

    def B(self, tau: float) -> float:
        """B(τ) function: [1 - exp(-κτ)] / κ"""
        kappa = self.params.kappa
        if abs(kappa * tau) < 1e-10:
            return tau
        return (1 - math.exp(-kappa * tau)) / kappa

    def A(self, t: float, T: float) -> float:
        """
        A(t,T) function for Hull-White bond pricing.

        A(t,T) = P^M(0,T)/P^M(0,t) × exp(B(T-t)f(0,t) - σ²/4κ × B(T-t)² × (1-e^{-2κt}))
        """
        kappa = self.params.kappa
        sigma = self.params.sigma

        P_T = self.market_discount_factor(T)
        P_t = self.market_discount_factor(t)

        if P_t <= 0:
            return 0

        B_tau = self.B(T - t)
        f_t = self.instantaneous_forward(t)

        if abs(kappa) < 1e-10:
            term = sigma ** 2 * t * B_tau ** 2 / 2
        else:
            term = sigma ** 2 / (4 * kappa) * B_tau ** 2 * (1 - math.exp(-2 * kappa * t))

        return P_T / P_t * math.exp(B_tau * f_t - term)

    def zero_coupon_bond_price(
        self,
        r: float,
        t: float,
        T: float
    ) -> float:
        """
        Price of ZCB at time t maturing at T.

        P(t,T) = A(t,T) × exp(-B(T-t) × r)
        """
        if T <= t:
            return 1.0

        return self.A(t, T) * math.exp(-self.B(T - t) * r)

    def zero_coupon_bond_option(
        self,
        r: float,
        t: float,
        T: float,
        S: float,
        K: float,
        option_type: OptionType = OptionType.CALL
    ) -> float:
        """
        European option on zero-coupon bond.

        Args:
            r: Current short rate
            t: Current time
            T: Option expiry
            S: Bond maturity
            K: Strike price
            option_type: CALL or PUT
        """
        if S <= T:
            raise ValueError("Bond maturity S must be > option expiry T")

        kappa = self.params.kappa
        sigma = self.params.sigma

        P_T = self.zero_coupon_bond_price(r, t, T)
        P_S = self.zero_coupon_bond_price(r, t, S)

        # Bond price volatility
        B_S_T = self.B(S - T)

        if abs(kappa) < 1e-10:
            sigma_P = sigma * B_S_T * math.sqrt(T - t)
        else:
            sigma_P = sigma * B_S_T * math.sqrt((1 - math.exp(-2 * kappa * (T - t))) / (2 * kappa))

        if sigma_P < 1e-10:
            if option_type == OptionType.CALL:
                return max(P_S - K * P_T, 0)
            else:
                return max(K * P_T - P_S, 0)

        d1 = (math.log(P_S / (K * P_T)) + 0.5 * sigma_P ** 2) / sigma_P
        d2 = d1 - sigma_P

        if option_type == OptionType.CALL:
            return P_S * norm_cdf(d1) - K * P_T * norm_cdf(d2)
        else:
            return K * P_T * norm_cdf(-d2) - P_S * norm_cdf(-d1)

    def caplet_price(
        self,
        r: float,
        t: float,
        T_start: float,
        T_end: float,
        K: float,
        notional: float = 1.0
    ) -> float:
        """
        Price a caplet using bond option relationship.

        Args:
            r: Current short rate
            t: Current time
            T_start: Caplet fixing date
            T_end: Caplet payment date
            K: Strike rate
            notional: Notional amount

        Returns:
            Caplet price
        """
        tau = T_end - T_start
        bond_strike = 1 / (1 + K * tau)

        put_price = self.zero_coupon_bond_option(
            r, t, T_start, T_end, bond_strike, OptionType.PUT
        )

        return notional * (1 + K * tau) * put_price

    def swaption_price(
        self,
        r: float,
        t: float,
        T_option: float,
        swap_tenors: List[float],
        K: float,
        notional: float = 1.0,
        payer: bool = True
    ) -> float:
        """
        Price a European swaption using Jamshidian decomposition.

        A payer swaption is the option to enter a swap paying fixed.

        Args:
            r: Current short rate
            t: Current time
            T_option: Option expiry (= swap start)
            swap_tenors: List of swap payment dates
            K: Strike rate (fixed rate)
            notional: Notional
            payer: True for payer swaption

        Returns:
            Swaption price
        """
        # Jamshidian decomposition: swaption = portfolio of bond options
        # Find r* such that swap value = 0
        def swap_value(r_star: float) -> float:
            annuity = 0
            for i, T_i in enumerate(swap_tenors):
                if i == 0:
                    tau_i = T_i - T_option
                else:
                    tau_i = T_i - swap_tenors[i - 1]
                P_i = self.A(T_option, T_i) * math.exp(-self.B(T_i - T_option) * r_star)
                annuity += tau_i * P_i

            P_n = self.A(T_option, swap_tenors[-1]) * math.exp(-self.B(swap_tenors[-1] - T_option) * r_star)

            return 1 - P_n - K * annuity

        # Newton-Raphson to find r*
        r_star = self.params.theta
        for _ in range(20):
            sv = swap_value(r_star)
            if abs(sv) < 1e-10:
                break
            # Numerical derivative
            eps = 0.0001
            dsv = (swap_value(r_star + eps) - sv) / eps
            if abs(dsv) < 1e-15:
                break
            r_star -= sv / dsv

        # Bond strikes
        strikes = []
        for T_i in swap_tenors:
            K_i = self.A(T_option, T_i) * math.exp(-self.B(T_i - T_option) * r_star)
            strikes.append(K_i)

        # Sum of bond options: puts for a payer swaption, calls for a receiver
        total = 0

        for i, (T_i, K_i) in enumerate(zip(swap_tenors, strikes)):
            if i == 0:
                tau_i = T_i - T_option
            else:
                tau_i = T_i - swap_tenors[i - 1]

            if payer:
                opt = self.zero_coupon_bond_option(r, t, T_option, T_i, K_i, OptionType.PUT)
                # Coupon contribution
                total += notional * K * tau_i * opt
                # Principal repayment on last period
                if i == len(swap_tenors) - 1:
                    total += notional * opt
            else:
                opt = self.zero_coupon_bond_option(r, t, T_option, T_i, K_i, OptionType.CALL)
                total += notional * K * tau_i * opt
                if i == len(swap_tenors) - 1:
                    total += notional * opt

        return total

    def calibrate(
        self,
        swaption_prices: List[Tuple[float, List[float], float, float]],
        initial_kappa: float = 0.1
    ) -> 'HullWhiteModel':
        """
        Calibrate kappa and sigma to swaption prices.

        Args:
            swaption_prices: List of (expiry, tenors, strike, market_price)
            initial_kappa: Starting value for kappa

        Returns:
            Calibrated model
        """
        kappa = initial_kappa
        sigma = self.params.sigma

        for iteration in range(50):
            # Compute model prices and gradients
            model_prices = []
            for T_opt, tenors, K, _ in swaption_prices:
                model = HullWhiteModel(
                    ShortRateParameters(kappa, self.params.theta, sigma, self.params.r0),
                    self.initial_curve
                )
                price = model.swaption_price(self.params.r0, 0, T_opt, tenors, K)
                model_prices.append(price)

            # Error
            error = sum(
                (mp - sp[3]) ** 2
                for mp, sp in zip(model_prices, swaption_prices)
            )

            if error < 1e-10:
                break

            # Gradient (numerical)
            eps = 0.001
            grad_kappa = 0
            grad_sigma = 0

            for i, (T_opt, tenors, K, market_price) in enumerate(swaption_prices):
                # Kappa gradient
                model_up = HullWhiteModel(
                    ShortRateParameters(kappa + eps, self.params.theta, sigma, self.params.r0),
                    self.initial_curve
                )
                price_up = model_up.swaption_price(self.params.r0, 0, T_opt, tenors, K)
                grad_kappa += 2 * (model_prices[i] - market_price) * (price_up - model_prices[i]) / eps

                # Sigma gradient
                model_up = HullWhiteModel(
                    ShortRateParameters(kappa, self.params.theta, sigma + eps, self.params.r0),
                    self.initial_curve
                )
                price_up = model_up.swaption_price(self.params.r0, 0, T_opt, tenors, K)
                grad_sigma += 2 * (model_prices[i] - market_price) * (price_up - model_prices[i]) / eps

            # Update
            lr = 0.01
            kappa -= lr * grad_kappa
            sigma -= lr * grad_sigma

            kappa = max(0.001, kappa)
            sigma = max(0.001, sigma)

        return HullWhiteModel(
            ShortRateParameters(kappa, self.params.theta, sigma, self.params.r0),
            self.initial_curve
        )

    def monte_carlo_paths(
        self,
        T: float,
        n_paths: int = 10000,
        n_steps: int = 252,
        seed: Optional[int] = None
    ) -> List[List[float]]:
        """Generate Monte Carlo paths with time-dependent drift."""
        rng = random.Random(seed) if seed is not None else random

        kappa = self.params.kappa
        sigma = self.params.sigma
        r0 = self.params.r0

        dt = T / n_steps

        paths = []

        for _ in range(n_paths):
            path = [r0]
            r = r0

            for step in range(n_steps):
                t = step * dt
                theta_current = self.theta_t(t)

                # Euler scheme with time-dependent drift
                dW = math.sqrt(dt) * rng.gauss(0, 1)
                r = r + (theta_current - kappa * r) * dt + sigma * dW

                path.append(r)

            paths.append(path)

        return paths


class BlackKarasinskiModel:
    """
    Black-Karasinski (1991) Log-Normal Short Rate Model.

    Dynamics: d(ln r) = [θ(t) - κ ln r]dt + σdW

    Equivalently: dr = r[θ(t) - κ ln r]dt + σr dW

    Key properties:
    - Rates are always positive (log-normal)
    - No analytical bond pricing (requires numerical methods)
    - Better fit for interest rate distributions
    - Common for modeling rates in practice
    """

    def __init__(
        self,
        params: ShortRateParameters,
        initial_curve: Optional[Callable[[float], float]] = None
    ):
        self.params = params
        if initial_curve is None:
            self.initial_curve = lambda t: max(params.theta, 0.001)
        else:
            self.initial_curve = initial_curve

    def trinomial_tree(
        self,
        T: float,
        n_steps: int = 100
    ) -> Tuple[List[List[float]], List[List[float]], float]:
        """
        Build a calibrated trinomial tree for the log rate.

        Returns node values, Arrow-Debreu prices, and time step.
        """
        kappa = self.params.kappa
        sigma = self.params.sigma

        dt = T / n_steps

        # Volatility of log(r) over dt
        vol_log_r = sigma * math.sqrt(dt)

        # Spacing
        dx = vol_log_r * math.sqrt(3)

        # Build tree
        log_r_tree = []  # log(r) values
        ad_prices = []  # Arrow-Debreu prices

        # Initialize
        log_r_tree.append([math.log(self.params.r0)])
        ad_prices.append([1.0])

        for i in range(1, n_steps + 1):
            t = i * dt
            n_nodes = 2 * i + 1

            # Log-rate nodes centered at mean
            mean_log_r = math.log(self.initial_curve(t))
            nodes = [mean_log_r + (j - i) * dx for j in range(n_nodes)]
            log_r_tree.append(nodes)

            # Arrow-Debreu prices
            new_ad = [0.0] * n_nodes

            for j, log_r in enumerate(log_r_tree[i - 1]):
                ad = ad_prices[i - 1][j]

                # Mean reversion of log(r): E[log_r_next] = log_r + kappa*(theta_log - log_r)*dt
                theta_log = math.log(max(self.initial_curve(i * dt), 1e-10))
                mean_next = log_r + kappa * (theta_log - log_r) * dt

                # Discount by current rate
                r_current = math.exp(log_r)
                discount = math.exp(-r_current * dt)
                ad_disc = ad * discount

                # Find closest node index in next layer
                k_center = round((mean_next - log_r_tree[i][0]) / dx) if dx > 0 else 0
                k_center = max(1, min(k_center, n_nodes - 2))

                # Only 3 neighbors get nonzero probability
                new_ad[k_center] += ad_disc * (2.0 / 3.0)
                new_ad[k_center - 1] += ad_disc * (1.0 / 6.0)
                new_ad[k_center + 1] += ad_disc * (1.0 / 6.0)

            ad_prices.append(new_ad)

        return log_r_tree, ad_prices, dt

    def zero_coupon_bond_price(
        self,
        T: float,
        n_steps: int = 100
    ) -> float:
        """
        Price ZCB using trinomial tree.
        """
        log_r_tree, ad_prices, dt = self.trinomial_tree(T, n_steps)

        # Sum Arrow-Debreu prices at maturity
        return sum(ad_prices[-1])

    def monte_carlo_paths(
        self,
        T: float,
        n_paths: int = 10000,
        n_steps: int = 252,
        seed: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generate Monte Carlo paths for Black-Karasinski.

        Simulates log(r) process using Euler scheme.
        """
        rng = random.Random(seed) if seed is not None else random

        kappa = self.params.kappa
        sigma = self.params.sigma
        r0 = self.params.r0

        dt = T / n_steps

        paths = []

        for _ in range(n_paths):
            path = [r0]
            log_r = math.log(r0)

            for step in range(n_steps):
                t = step * dt
                theta_t = math.log(self.initial_curve(t + dt))  # Simplified

                # Euler on log(r)
                dW = math.sqrt(dt) * rng.gauss(0, 1)
                log_r = log_r + (theta_t - kappa * log_r) * dt + sigma * dW

                r = math.exp(log_r)
                path.append(r)

            paths.append(path)

        return paths


# Factory functions

def create_vasicek(
    kappa: float = 0.1,
    theta: float = 0.05,
    sigma: float = 0.01,
    r0: float = 0.03
) -> VasicekModel:
    """Create a Vasicek model with given parameters."""
    return VasicekModel(ShortRateParameters(kappa, theta, sigma, r0))


def create_cir(
    kappa: float = 0.1,
    theta: float = 0.05,
    sigma: float = 0.02,
    r0: float = 0.03
) -> CIRModel:
    """Create a CIR model with given parameters."""
    return CIRModel(ShortRateParameters(kappa, theta, sigma, r0))


def create_hull_white(
    kappa: float = 0.1,
    sigma: float = 0.01,
    r0: float = 0.03,
    initial_curve: Optional[Callable[[float], float]] = None
) -> HullWhiteModel:
    """Create a Hull-White model."""
    return HullWhiteModel(
        ShortRateParameters(kappa, 0.05, sigma, r0),
        initial_curve
    )
