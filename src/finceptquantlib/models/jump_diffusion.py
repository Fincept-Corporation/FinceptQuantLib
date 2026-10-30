"""
Jump Diffusion Models for Option Pricing

This module implements various jump diffusion models:
- Merton Jump Diffusion: Log-normal jump sizes
- Kou Double Exponential: Asymmetric jumps with exponential distribution
- Variance Gamma: Infinite activity Lévy process

Each model includes:
- Closed-form pricing (where available)
- Characteristic function for FFT pricing
- Monte Carlo simulation
- Calibration routines
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import math
import cmath
import random

from finceptquantlib.core.foundations.types import OptionType
from finceptquantlib.core.utilities.distributions import norm_cdf, norm_pdf
from finceptquantlib.core.utilities.results import MonteCarloResult
from finceptquantlib.stochastic.sampling import poisson_sample
from finceptquantlib.numerical.fft import cooley_tukey_fft


@dataclass
class MertonParameters:
    """
    Parameters for Merton Jump Diffusion Model.

    The model assumes asset price follows:
    dS/S = (μ - λκ)dt + σdW + (J-1)dN

    where:
    - σ: diffusion volatility
    - λ: jump intensity (expected number of jumps per year)
    - μ_J: mean of log-jump size
    - σ_J: std dev of log-jump size
    - J ~ LogNormal(μ_J, σ_J²)
    - κ = E[J-1] = exp(μ_J + σ_J²/2) - 1

    Attributes:
        sigma: Diffusion volatility (annualized)
        lambda_jump: Jump intensity (jumps per year)
        mu_jump: Mean of log-jump size
        sigma_jump: Std dev of log-jump size
    """
    sigma: float
    lambda_jump: float
    mu_jump: float
    sigma_jump: float

    def __post_init__(self):
        if self.sigma < 0:
            raise ValueError("Diffusion volatility must be non-negative")
        if self.lambda_jump < 0:
            raise ValueError("Jump intensity must be non-negative")
        if self.sigma_jump < 0:
            raise ValueError("Jump size std dev must be non-negative")

    @property
    def kappa(self) -> float:
        """Expected relative jump size: E[J-1] = exp(μ_J + σ_J²/2) - 1"""
        return math.exp(self.mu_jump + 0.5 * self.sigma_jump ** 2) - 1.0

    @property
    def total_variance(self) -> float:
        """Total instantaneous variance including jumps."""
        jump_variance = self.lambda_jump * (
            self.sigma_jump ** 2 + self.mu_jump ** 2
        )
        return self.sigma ** 2 + jump_variance


class MertonJumpDiffusion:
    """
    Merton (1976) Jump Diffusion Model.

    The stock price follows:
    dS/S = (r - q - λκ)dt + σdW + (J-1)dN(t)

    where N(t) is a Poisson process with intensity λ, and J is
    log-normally distributed: ln(J) ~ N(μ_J, σ_J²)

    Key features:
    - Can generate volatility smile/skew
    - Captures sudden large moves (jumps)
    - Semi-analytical pricing via infinite series

    References:
        Merton, R. C. (1976). "Option pricing when underlying stock returns
        are discontinuous." Journal of Financial Economics, 3, 125-144.
    """

    def __init__(self, params: MertonParameters):
        self.params = params

    def characteristic_function(self, u: complex, T: float, r: float, q: float) -> complex:
        """
        Characteristic function of log-price under risk-neutral measure.

        φ(u) = E[exp(i·u·ln(S_T/S_0))]

        For Merton model:
        φ(u) = exp(iuωT - 0.5σ²u²T + λT(exp(iuμ_J - 0.5σ_J²u²) - 1))

        where ω = r - q - 0.5σ² - λκ (drift adjustment)

        Args:
            u: Complex frequency
            T: Time to maturity
            r: Risk-free rate
            q: Dividend yield

        Returns:
            Complex characteristic function value
        """
        sigma = self.params.sigma
        lambda_j = self.params.lambda_jump
        mu_j = self.params.mu_jump
        sigma_j = self.params.sigma_jump
        kappa = self.params.kappa

        # Drift under risk-neutral measure
        omega = r - q - 0.5 * sigma ** 2 - lambda_j * kappa

        # Diffusion component
        diffusion = 1j * u * omega * T - 0.5 * sigma ** 2 * u ** 2 * T

        # Jump component
        jump_cf = cmath.exp(1j * u * mu_j - 0.5 * sigma_j ** 2 * u ** 2) - 1
        jump = lambda_j * T * jump_cf

        return cmath.exp(diffusion + jump)

    def european_price_series(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        option_type: OptionType = OptionType.CALL,
        n_terms: int = 50
    ) -> float:
        """
        European option price using Merton's infinite series formula.

        The call price is:
        C = Σ_{n=0}^∞ [exp(-λ'T)(λ'T)^n / n!] × BS(S, K, T, r_n, σ_n)

        where:
        - λ' = λ(1 + κ)
        - r_n = r - λκ + nγ/T, where γ = ln(1+κ)
        - σ_n² = σ² + nσ_J²/T

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            q: Dividend yield
            option_type: CALL or PUT
            n_terms: Number of series terms (typically 30-50 sufficient)

        Returns:
            Option price
        """
        if S <= 0 or K <= 0 or T <= 0:
            raise ValueError("S, K, and T must be positive")

        sigma = self.params.sigma
        lambda_j = self.params.lambda_jump
        sigma_j = self.params.sigma_jump
        kappa = self.params.kappa

        # Modified intensity
        lambda_prime = lambda_j * (1 + kappa)
        gamma = math.log(1 + kappa) if kappa > -1 else 0

        price = 0.0
        poisson_factor = math.exp(-lambda_prime * T)

        for n in range(n_terms):
            # Poisson weight
            if n > 0:
                poisson_factor *= lambda_prime * T / n

            # Adjusted parameters for n jumps
            sigma_n_sq = sigma ** 2 + n * sigma_j ** 2 / T
            sigma_n = math.sqrt(sigma_n_sq)
            r_n = r - lambda_j * kappa + n * gamma / T

            # Black-Scholes price with adjusted parameters
            bs_price = self._black_scholes(S, K, T, r_n, q, sigma_n, option_type)

            price += poisson_factor * bs_price

            # Convergence check
            if n > 10 and poisson_factor < 1e-15:
                break

        return price

    def european_price_fft(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        option_type: OptionType = OptionType.CALL,
        N: int = 4096,
        alpha: float = 1.5
    ) -> float:
        """
        European option price using FFT (Carr-Madan method).

        Uses the characteristic function for pricing via:
        C(k) = exp(-αk)/π × ∫₀^∞ exp(-iuk)ψ(u)du

        where k = ln(K/S) and ψ(u) is the modified characteristic function.

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            q: Dividend yield
            option_type: CALL or PUT
            N: Number of FFT points (power of 2)
            alpha: Damping factor (typically 1.0-2.0)

        Returns:
            Option price
        """
        # FFT parameters
        eta = 0.25  # Grid spacing in u
        lambda_fft = 2 * math.pi / (N * eta)  # Grid spacing in k
        b = N * lambda_fft / 2  # Upper bound for k

        # Build characteristic function values
        u_values = [eta * j for j in range(N)]
        k_target = math.log(K / S)

        # Modified characteristic function for call
        psi_values = []
        for j, u in enumerate(u_values):
            v = u - (alpha + 1) * 1j
            cf = self.characteristic_function(v, T, r, q)

            # Denominator
            denom = alpha ** 2 + alpha - u ** 2 + 1j * (2 * alpha + 1) * u
            if abs(denom) < 1e-10:
                psi = 0
            else:
                psi = math.exp(-r * T) * cf / denom

            # Simpson's weights
            if j == 0:
                weight = eta / 3
            elif j % 2 == 0:
                weight = 2 * eta / 3
            else:
                weight = 4 * eta / 3

            psi_values.append(psi * weight * cmath.exp(-1j * b * u))

        # FFT - using centralized implementation
        fft_result = cooley_tukey_fft(psi_values)

        # Interpolate to get price at target strike
        k_values = [-b + lambda_fft * j for j in range(N)]

        # Find bracketing indices
        idx = int((k_target + b) / lambda_fft)
        idx = max(0, min(idx, N - 2))

        # Linear interpolation
        w = (k_target - k_values[idx]) / lambda_fft
        call_price_complex = (1 - w) * fft_result[idx] + w * fft_result[idx + 1]
        call_price = math.exp(-alpha * k_target) / math.pi * call_price_complex.real

        call_price = max(0, call_price) * S

        if option_type == OptionType.PUT:
            # Put-call parity
            return call_price - S * math.exp(-q * T) + K * math.exp(-r * T)

        return call_price

    def monte_carlo_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        option_type: OptionType = OptionType.CALL,
        n_paths: int = 100000,
        n_steps: int = 252,
        seed: Optional[int] = None
    ) -> MonteCarloResult:
        """
        Monte Carlo simulation for option pricing.

        Simulates paths using:
        S_{t+dt} = S_t × exp[(r - q - 0.5σ² - λκ)dt + σ√dt × Z + Σ_{i=1}^{N_t} ln(J_i)]

        where N_t ~ Poisson(λdt) and ln(J_i) ~ N(μ_J, σ_J²)

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            q: Dividend yield
            option_type: CALL or PUT
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            seed: Random seed for reproducibility

        Returns:
            MonteCarloResult with price, std_error, and statistics
        """
        rng = random.Random(seed) if seed is not None else random

        sigma = self.params.sigma
        lambda_j = self.params.lambda_jump
        mu_j = self.params.mu_jump
        sigma_j = self.params.sigma_jump
        kappa = self.params.kappa

        dt = T / n_steps
        drift = (r - q - 0.5 * sigma ** 2 - lambda_j * kappa) * dt
        vol_sqrt_dt = sigma * math.sqrt(dt)

        payoffs = []

        for _ in range(n_paths):
            S_t = S

            for _ in range(n_steps):
                # Diffusion
                Z = rng.gauss(0, 1)
                S_t *= math.exp(drift + vol_sqrt_dt * Z)

                # Jumps (Poisson process) - using centralized sampling
                n_jumps = poisson_sample(lambda_j * dt, rng)
                for _ in range(n_jumps):
                    jump = math.exp(rng.gauss(mu_j, sigma_j))
                    S_t *= jump

            # Payoff
            if option_type == OptionType.CALL:
                payoff = max(S_t - K, 0)
            else:
                payoff = max(K - S_t, 0)

            payoffs.append(payoff)

        # Discount and compute statistics
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

    def implied_volatility(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        option_type: OptionType = OptionType.CALL
    ) -> float:
        """
        Compute Black-Scholes implied volatility from model price.

        Uses Newton-Raphson iteration.
        """
        target_price = self.european_price_series(S, K, T, r, q, option_type)

        # Initial guess: total volatility
        sigma = math.sqrt(self.params.total_variance)

        for _ in range(50):
            bs_price = self._black_scholes(S, K, T, r, q, sigma, option_type)
            vega = self._black_scholes_vega(S, K, T, r, q, sigma)

            if abs(vega) < 1e-10:
                break

            diff = bs_price - target_price
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
        sigma: float,
        option_type: OptionType
    ) -> float:
        """Black-Scholes formula using centralized distribution functions."""
        if sigma <= 0 or T <= 0:
            if option_type == OptionType.CALL:
                return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0)
            else:
                return max(K * math.exp(-r * T) - S * math.exp(-q * T), 0)

        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option_type == OptionType.CALL:
            return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        else:
            return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)

    def _black_scholes_vega(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float
    ) -> float:
        """Black-Scholes vega using centralized distribution functions."""
        if sigma <= 0 or T <= 0:
            return 0.0

        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return S * math.exp(-q * T) * math.sqrt(T) * norm_pdf(d1)


@dataclass
class KouParameters:
    """
    Parameters for Kou Double Exponential Jump Diffusion Model.

    Jump sizes follow double exponential distribution:
    - Probability p of upward jump with rate η₁
    - Probability (1-p) of downward jump with rate η₂

    Attributes:
        sigma: Diffusion volatility
        lambda_jump: Jump intensity
        p: Probability of upward jump
        eta1: Rate parameter for upward jumps (η₁ > 1 required)
        eta2: Rate parameter for downward jumps (η₂ > 0)
    """
    sigma: float
    lambda_jump: float
    p: float
    eta1: float
    eta2: float

    def __post_init__(self):
        if self.sigma < 0:
            raise ValueError("Diffusion volatility must be non-negative")
        if self.lambda_jump < 0:
            raise ValueError("Jump intensity must be non-negative")
        if not (0 <= self.p <= 1):
            raise ValueError("Jump probability p must be in [0, 1]")
        if self.eta1 <= 1:
            raise ValueError("Upward jump rate η₁ must be > 1 for finite expectation")
        if self.eta2 <= 0:
            raise ValueError("Downward jump rate η₂ must be positive")

    @property
    def kappa(self) -> float:
        """Expected relative jump size: E[J-1]"""
        return self.p * self.eta1 / (self.eta1 - 1) + (1 - self.p) * self.eta2 / (self.eta2 + 1) - 1


class KouJumpDiffusion:
    """
    Kou (2002) Double Exponential Jump Diffusion Model.

    The model uses asymmetric double exponential jumps:
    - Upward jumps: Y ~ Exp(η₁) with probability p
    - Downward jumps: Y ~ Exp(η₂) with probability 1-p

    Key advantages over Merton:
    - Better fit to equity return distributions
    - Analytical formulas via Laplace transform
    - Captures asymmetric jump behavior

    References:
        Kou, S. G. (2002). "A Jump-Diffusion Model for Option Pricing."
        Management Science, 48(8), 1086-1101.
    """

    def __init__(self, params: KouParameters):
        self.params = params

    def characteristic_function(self, u: complex, T: float, r: float, q: float) -> complex:
        """
        Characteristic function of log-price.

        For Kou model:
        φ(u) = exp(iuωT - 0.5σ²u²T + λT[p·η₁/(η₁-iu) + (1-p)·η₂/(η₂+iu) - 1])
        """
        sigma = self.params.sigma
        lambda_j = self.params.lambda_jump
        p = self.params.p
        eta1 = self.params.eta1
        eta2 = self.params.eta2
        kappa = self.params.kappa

        omega = r - q - 0.5 * sigma ** 2 - lambda_j * kappa

        # Diffusion
        diffusion = 1j * u * omega * T - 0.5 * sigma ** 2 * u ** 2 * T

        # Jump component
        jump_up = p * eta1 / (eta1 - 1j * u)
        jump_down = (1 - p) * eta2 / (eta2 + 1j * u)
        jump = lambda_j * T * (jump_up + jump_down - 1)

        return cmath.exp(diffusion + jump)

    def european_price_fft(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        option_type: OptionType = OptionType.CALL,
        N: int = 4096,
        alpha: float = 1.5
    ) -> float:
        """
        European option price using FFT (Carr-Madan method).

        Similar implementation to Merton FFT pricing.
        """
        eta = 0.25
        lambda_fft = 2 * math.pi / (N * eta)
        b = N * lambda_fft / 2

        u_values = [eta * j for j in range(N)]
        k_target = math.log(K / S)

        psi_values = []
        for j, u in enumerate(u_values):
            v = u - (alpha + 1) * 1j
            cf = self.characteristic_function(v, T, r, q)

            denom = alpha ** 2 + alpha - u ** 2 + 1j * (2 * alpha + 1) * u
            if abs(denom) < 1e-10:
                psi = 0
            else:
                psi = math.exp(-r * T) * cf / denom

            if j == 0:
                weight = eta / 3
            elif j % 2 == 0:
                weight = 2 * eta / 3
            else:
                weight = 4 * eta / 3

            psi_values.append(psi * weight * cmath.exp(-1j * b * u))

        # FFT - using centralized implementation
        fft_result = cooley_tukey_fft(psi_values)

        k_values = [-b + lambda_fft * j for j in range(N)]
        idx = int((k_target + b) / lambda_fft)
        idx = max(0, min(idx, N - 2))

        w = (k_target - k_values[idx]) / lambda_fft
        call_price_complex = (1 - w) * fft_result[idx] + w * fft_result[idx + 1]
        call_price = math.exp(-alpha * k_target) / math.pi * call_price_complex.real

        call_price = max(0, call_price) * S

        if option_type == OptionType.PUT:
            return call_price - S * math.exp(-q * T) + K * math.exp(-r * T)

        return call_price

    def monte_carlo_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        option_type: OptionType = OptionType.CALL,
        n_paths: int = 100000,
        n_steps: int = 252,
        seed: Optional[int] = None
    ) -> MonteCarloResult:
        """
        Monte Carlo simulation with double exponential jumps.
        """
        rng = random.Random(seed) if seed is not None else random

        sigma = self.params.sigma
        lambda_j = self.params.lambda_jump
        p = self.params.p
        eta1 = self.params.eta1
        eta2 = self.params.eta2
        kappa = self.params.kappa

        dt = T / n_steps
        drift = (r - q - 0.5 * sigma ** 2 - lambda_j * kappa) * dt
        vol_sqrt_dt = sigma * math.sqrt(dt)

        payoffs = []

        for _ in range(n_paths):
            S_t = S

            for _ in range(n_steps):
                Z = rng.gauss(0, 1)
                S_t *= math.exp(drift + vol_sqrt_dt * Z)

                # Jumps - using centralized Poisson sampling
                n_jumps = poisson_sample(lambda_j * dt, rng)
                for _ in range(n_jumps):
                    if rng.random() < p:
                        # Upward jump: ln(J) ~ Exp(η₁), so J = exp(Y), Y ~ Exp(η₁)
                        jump = math.exp(rng.expovariate(eta1))
                    else:
                        # Downward jump: ln(J) ~ -Exp(η₂), so J = exp(-Y), Y ~ Exp(η₂)
                        jump = math.exp(-rng.expovariate(eta2))
                    S_t *= jump

            if option_type == OptionType.CALL:
                payoff = max(S_t - K, 0)
            else:
                payoff = max(K - S_t, 0)

            payoffs.append(payoff)

        mean_payoff = sum(payoffs) / n_paths
        price = math.exp(-r * T) * mean_payoff
        variance = sum((pf - mean_payoff) ** 2 for pf in payoffs) / (n_paths - 1)
        std_error = math.exp(-r * T) * math.sqrt(variance / n_paths)

        return MonteCarloResult(
            price=price,
            std_error=std_error,
            n_paths=n_paths,
            variance=variance,
        )


@dataclass
class VGParameters:
    """
    Parameters for Variance Gamma Process.

    The VG process is a pure jump process with infinite activity
    constructed as a Brownian motion with drift subordinated by
    a Gamma process.

    Attributes:
        sigma: Volatility of the Brownian motion
        theta: Drift of the Brownian motion (skewness)
        nu: Variance rate of the Gamma subordinator (controls kurtosis)
    """
    sigma: float
    theta: float
    nu: float

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError("Sigma must be positive")
        if self.nu <= 0:
            raise ValueError("Nu (variance rate) must be positive")

    @property
    def omega(self) -> float:
        """Convexity correction: ω = (1/ν)ln(1 - θν - σ²ν/2)"""
        val = 1 - self.theta * self.nu - 0.5 * self.sigma ** 2 * self.nu
        if val <= 0:
            raise ValueError("Parameters lead to non-finite convexity correction")
        return math.log(val) / self.nu


class VarianceGamma:
    """
    Variance Gamma (VG) Process for Option Pricing.

    The VG process X(t) is defined as:
    X(t) = θG(t) + σW(G(t))

    where G(t) is a Gamma process with unit mean rate and variance rate ν,
    and W is a standard Brownian motion.

    Key properties:
    - Infinite activity (infinite number of jumps in any interval)
    - Finite variation
    - Can generate skewness (θ) and excess kurtosis (controlled by ν)
    - Semi-analytical pricing via FFT

    Special cases:
    - θ = 0: Symmetric VG
    - ν → 0: Converges to geometric Brownian motion

    References:
        Madan, D. B., Carr, P., & Chang, E. C. (1998). "The Variance Gamma
        Process and Option Pricing." European Finance Review, 2, 79-105.
    """

    def __init__(self, params: VGParameters):
        self.params = params

    def characteristic_function(self, u: complex, T: float, r: float, q: float) -> complex:
        """
        Characteristic function of log-price under VG.

        φ(u) = exp(iuωT) × (1 - iuθν + 0.5σ²νu²)^(-T/ν)

        where ω is the convexity correction ensuring the martingale property.
        """
        sigma = self.params.sigma
        theta = self.params.theta
        nu = self.params.nu
        omega = self.params.omega

        # VG characteristic function
        vg_cf = (1 - 1j * u * theta * nu + 0.5 * sigma ** 2 * nu * u ** 2) ** (-T / nu)

        # Risk-neutral drift adjustment
        drift = 1j * u * (r - q + omega) * T

        return cmath.exp(drift) * vg_cf

    def european_price_fft(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        option_type: OptionType = OptionType.CALL,
        N: int = 4096,
        alpha: float = 1.5
    ) -> float:
        """
        European option price using FFT (Carr-Madan method).
        """
        eta = 0.25
        lambda_fft = 2 * math.pi / (N * eta)
        b = N * lambda_fft / 2

        u_values = [eta * j for j in range(N)]
        k_target = math.log(K / S)

        psi_values = []
        for j, u in enumerate(u_values):
            v = u - (alpha + 1) * 1j

            try:
                cf = self.characteristic_function(v, T, r, q)
            except (ValueError, OverflowError):
                cf = 0

            denom = alpha ** 2 + alpha - u ** 2 + 1j * (2 * alpha + 1) * u
            if abs(denom) < 1e-10:
                psi = 0
            else:
                psi = math.exp(-r * T) * cf / denom

            if j == 0:
                weight = eta / 3
            elif j % 2 == 0:
                weight = 2 * eta / 3
            else:
                weight = 4 * eta / 3

            psi_values.append(psi * weight * cmath.exp(-1j * b * u))

        # FFT - using centralized implementation
        fft_result = cooley_tukey_fft(psi_values)

        k_values = [-b + lambda_fft * j for j in range(N)]
        idx = int((k_target + b) / lambda_fft)
        idx = max(0, min(idx, N - 2))

        w = (k_target - k_values[idx]) / lambda_fft
        call_price_complex = (1 - w) * fft_result[idx] + w * fft_result[idx + 1]
        call_price = math.exp(-alpha * k_target) / math.pi * call_price_complex.real

        call_price = max(0, call_price) * S

        if option_type == OptionType.PUT:
            return call_price - S * math.exp(-q * T) + K * math.exp(-r * T)

        return call_price

    def monte_carlo_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        option_type: OptionType = OptionType.CALL,
        n_paths: int = 100000,
        n_steps: int = 252,
        seed: Optional[int] = None
    ) -> MonteCarloResult:
        """
        Monte Carlo simulation using difference of Gamma processes.

        The VG process can be represented as:
        X(t) = G⁺(t) - G⁻(t)

        where G⁺ and G⁻ are independent Gamma processes.
        """
        rng = random.Random(seed) if seed is not None else random

        sigma = self.params.sigma
        theta = self.params.theta
        nu = self.params.nu
        omega = self.params.omega

        dt = T / n_steps

        # Gamma process parameters
        # G ~ Gamma(dt/nu, 1/nu)
        shape = dt / nu
        scale = nu

        payoffs = []

        for _ in range(n_paths):
            log_S = math.log(S)

            for _ in range(n_steps):
                # Simulate Gamma subordinator
                G = rng.gammavariate(shape, scale)

                # VG increment: θG + σ√G × Z
                Z = rng.gauss(0, 1)
                dX = theta * G + sigma * math.sqrt(G) * Z

                # Risk-neutral drift
                log_S += (r - q + omega) * dt + dX

            S_T = math.exp(log_S)

            if option_type == OptionType.CALL:
                payoff = max(S_T - K, 0)
            else:
                payoff = max(K - S_T, 0)

            payoffs.append(payoff)

        mean_payoff = sum(payoffs) / n_paths
        price = math.exp(-r * T) * mean_payoff
        variance = sum((pf - mean_payoff) ** 2 for pf in payoffs) / (n_paths - 1)
        std_error = math.exp(-r * T) * math.sqrt(variance / n_paths)

        return MonteCarloResult(
            price=price,
            std_error=std_error,
            n_paths=n_paths,
            variance=variance,
        )

    def calibrate(
        self,
        market_prices: List[Tuple[float, float, float]],
        S: float,
        r: float,
        q: float = 0.0,
        option_type: OptionType = OptionType.CALL
    ) -> 'VarianceGamma':
        """
        Calibrate VG parameters to market prices.

        Args:
            market_prices: List of (strike, maturity, price) tuples
            S: Spot price
            r: Risk-free rate
            q: Dividend yield
            option_type: Option type

        Returns:
            Calibrated VarianceGamma model
        """
        def objective(params):
            try:
                vg_params = VGParameters(
                    sigma=params[0],
                    theta=params[1],
                    nu=params[2]
                )
                model = VarianceGamma(vg_params)
            except ValueError:
                return 1e10

            total_error = 0
            for K, T, market_price in market_prices:
                try:
                    model_price = model.european_price_fft(S, K, T, r, q, option_type)
                    total_error += (model_price - market_price) ** 2
                except (ValueError, OverflowError, ZeroDivisionError):
                    total_error += 1e6

            return total_error

        # Simple gradient descent
        params = [self.params.sigma, self.params.theta, self.params.nu]
        learning_rate = 0.01

        for _ in range(100):
            grad = []
            for i in range(3):
                eps = 0.001
                params_up = params.copy()
                params_up[i] += eps
                params_down = params.copy()
                params_down[i] -= eps
                grad.append((objective(params_up) - objective(params_down)) / (2 * eps))

            for i in range(3):
                params[i] -= learning_rate * grad[i]

            # Constraints
            params[0] = max(0.01, params[0])  # sigma > 0
            params[2] = max(0.001, params[2])  # nu > 0

        return VarianceGamma(VGParameters(
            sigma=params[0],
            theta=params[1],
            nu=params[2]
        ))


# Convenience functions for quick pricing

def merton_call_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lambda_jump: float,
    mu_jump: float,
    sigma_jump: float,
    q: float = 0.0
) -> float:
    """
    Quick Merton jump diffusion call pricing.

    Args:
        S: Spot price
        K: Strike
        T: Time to maturity
        r: Risk-free rate
        sigma: Diffusion volatility
        lambda_jump: Jump intensity
        mu_jump: Mean log-jump
        sigma_jump: Std dev log-jump
        q: Dividend yield

    Returns:
        Call option price
    """
    params = MertonParameters(sigma, lambda_jump, mu_jump, sigma_jump)
    model = MertonJumpDiffusion(params)
    return model.european_price_series(S, K, T, r, q, OptionType.CALL)


def kou_call_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lambda_jump: float,
    p: float,
    eta1: float,
    eta2: float,
    q: float = 0.0
) -> float:
    """
    Quick Kou double exponential call pricing.
    """
    params = KouParameters(sigma, lambda_jump, p, eta1, eta2)
    model = KouJumpDiffusion(params)
    return model.european_price_fft(S, K, T, r, q, OptionType.CALL)


def vg_call_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    theta: float,
    nu: float,
    q: float = 0.0
) -> float:
    """
    Quick Variance Gamma call pricing.
    """
    params = VGParameters(sigma, theta, nu)
    model = VarianceGamma(params)
    return model.european_price_fft(S, K, T, r, q, OptionType.CALL)
