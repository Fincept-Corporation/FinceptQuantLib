"""
Bachelier (Normal) Model

Arithmetic Brownian motion pricing:
- Used for negative rate environments
- Used for spread options
- Linear payoff approximations

Parameter Conventions:
---------------------
- F: forward price
- K: strike price
- r: risk-free rate (decimal)
- sigma_n / normal_vol: normal volatility (in price units, not percentage)
- T: time to maturity (years)

Note: Normal volatility is in absolute price units, e.g., if forward is 100
and normal vol is 20, this means the forward has std dev of 20 per sqrt(year).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from finceptquantlib.core.foundations.types import OptionType
from finceptquantlib.core.utilities.distributions import norm_cdf, norm_pdf
from finceptquantlib.pricing.analytical._implied_vol import solve_implied_vol


# =============================================================================
# Bachelier Formulas
# =============================================================================

def bachelier_call(F: float, K: float, r: float, sigma_n: float, T: float) -> float:
    """
    Bachelier (Normal) model call price.

    Under Bachelier: dF = σ_n dW (arithmetic BM)

    C = e^(-rT) * [(F-K)N(d) + σ_n√T * n(d)]

    where d = (F-K) / (σ_n√T)

    Args:
        F: Forward price
        K: Strike price
        r: Risk-free rate
        sigma_n: Normal volatility (in price units, not %)
        T: Time to maturity

    Returns:
        Call price
    """
    if T <= 0:
        return max(F - K, 0)
    if sigma_n <= 0:
        return math.exp(-r * T) * max(F - K, 0)

    sqrt_T = math.sqrt(T)
    sigma_sqrt_T = sigma_n * sqrt_T

    d = (F - K) / sigma_sqrt_T

    return math.exp(-r * T) * ((F - K) * norm_cdf(d) + sigma_sqrt_T * norm_pdf(d))


def bachelier_put(F: float, K: float, r: float, sigma_n: float, T: float) -> float:
    """
    Bachelier put price.

    P = e^(-rT) * [(K-F)N(-d) + σ_n√T * n(d)]
    """
    if T <= 0:
        return max(K - F, 0)
    if sigma_n <= 0:
        return math.exp(-r * T) * max(K - F, 0)

    sqrt_T = math.sqrt(T)
    sigma_sqrt_T = sigma_n * sqrt_T

    d = (F - K) / sigma_sqrt_T

    return math.exp(-r * T) * ((K - F) * norm_cdf(-d) + sigma_sqrt_T * norm_pdf(d))


def bachelier_price(
    F: float,
    K: float,
    r: float,
    sigma_n: float,
    T: float,
    option_type: OptionType
) -> float:
    """Bachelier option price."""
    if option_type == OptionType.CALL:
        return bachelier_call(F, K, r, sigma_n, T)
    else:
        return bachelier_put(F, K, r, sigma_n, T)


# =============================================================================
# Greeks
# =============================================================================

def bachelier_delta(
    F: float, K: float, r: float, sigma_n: float, T: float,
    option_type: OptionType
) -> float:
    """
    Bachelier delta: ∂V/∂F

    Call: e^(-rT) * N(d)
    Put: e^(-rT) * (N(d) - 1)
    """
    if T <= 0 or sigma_n <= 0:
        if option_type == OptionType.CALL:
            return math.exp(-r * T) if F > K else 0.0
        else:
            return -math.exp(-r * T) if F < K else 0.0

    d = (F - K) / (sigma_n * math.sqrt(T))
    discount = math.exp(-r * T)

    if option_type == OptionType.CALL:
        return discount * norm_cdf(d)
    else:
        return discount * (norm_cdf(d) - 1)


def bachelier_gamma(F: float, K: float, r: float, sigma_n: float, T: float) -> float:
    """
    Bachelier gamma: ∂²V/∂F²

    Γ = e^(-rT) * n(d) / (σ_n√T)
    """
    if T <= 0 or sigma_n <= 0:
        return 0.0

    d = (F - K) / (sigma_n * math.sqrt(T))
    return math.exp(-r * T) * norm_pdf(d) / (sigma_n * math.sqrt(T))


def bachelier_vega(F: float, K: float, r: float, sigma_n: float, T: float) -> float:
    """
    Bachelier vega: ∂V/∂σ_n

    ν = e^(-rT) * √T * n(d)
    """
    if T <= 0:
        return 0.0

    d = (F - K) / (sigma_n * math.sqrt(T)) if sigma_n > 0 else 0

    return math.exp(-r * T) * math.sqrt(T) * norm_pdf(d)


@dataclass
class BachelierGreeks:
    """Bachelier Greeks."""
    delta: float
    gamma: float
    vega: float


def bachelier_greeks(
    F: float, K: float, r: float, sigma_n: float, T: float,
    option_type: OptionType
) -> BachelierGreeks:
    """Compute Bachelier Greeks."""
    return BachelierGreeks(
        delta=bachelier_delta(F, K, r, sigma_n, T, option_type),
        gamma=bachelier_gamma(F, K, r, sigma_n, T),
        vega=bachelier_vega(F, K, r, sigma_n, T)
    )


# =============================================================================
# Implied Volatility
# =============================================================================

def bachelier_implied_vol(
    price: float,
    F: float,
    K: float,
    r: float,
    T: float,
    option_type: OptionType,
    tol: float = 1e-8,
    max_iter: int = 100
) -> float:
    """
    Bachelier implied (normal) volatility.
    """
    if T <= 0:
        raise ValueError("Cannot compute implied vol with T <= 0")

    # Initial guess based on ATM approximation
    discount = math.exp(-r * T)
    intrinsic = max(F - K, 0) if option_type == OptionType.CALL else max(K - F, 0)
    time_value = price / discount - intrinsic
    guess = max(1e-6, time_value * math.sqrt(2 * math.pi / T))
    scale = max(abs(F), abs(K), abs(F - K), 1e-8)

    return solve_implied_vol(
        price_fn=lambda sigma_n: bachelier_price(F, K, r, sigma_n, T, option_type),
        vega_fn=lambda sigma_n: bachelier_vega(F, K, r, sigma_n, T),
        target=price,
        guess=guess,
        lo=1e-12,
        hi=max(guess, scale),
        tol=tol,
        max_iter=max_iter,
        model="Bachelier",
        hi_limit=scale * 1e4,
    )


# =============================================================================
# Volatility Conversion
# =============================================================================

def lognormal_to_normal_vol(
    sigma_ln: float,
    F: float,
    K: float,
    T: float
) -> float:
    """
    Convert lognormal (Black) vol to normal (Bachelier) vol.

    Approximation: σ_n ≈ σ_ln * F  (at-the-money)

    More accurate: σ_n ≈ σ_ln * (F + K) / 2

    Exact (Hagan): Uses expansion matching option prices.
    """
    avg = (F + K) / 2
    if avg <= 0:
        raise ValueError("Cannot convert from lognormal vol when (F+K)/2 <= 0")
    return sigma_ln * avg


def normal_to_lognormal_vol(
    sigma_n: float,
    F: float,
    K: float,
    T: float
) -> float:
    """
    Convert normal (Bachelier) vol to lognormal (Black) vol.

    σ_ln ≈ 2 * σ_n / (F + K)
    """
    avg = F + K
    if abs(avg) < 1e-15:
        raise ValueError("Cannot convert to lognormal vol when F + K ≈ 0")
    return 2 * sigma_n / avg


# =============================================================================
# Negative Rate Applications
# =============================================================================

def shifted_lognormal_call(
    F: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    shift: float = 0.0
) -> float:
    """
    Shifted lognormal (displaced diffusion) call.

    Applies shift to allow negative rates:
    Model: d(F+shift) = σ(F+shift) dW

    Price is Black76 call on (F+shift) with strike (K+shift).
    """
    from finceptquantlib.pricing.analytical.black76 import black76_call

    F_shifted = F + shift
    K_shifted = K + shift

    if F_shifted <= 0 or K_shifted <= 0:
        # Fall back to Bachelier for negative shifted values
        sigma_n = sigma * abs(F_shifted) if abs(F_shifted) > 0.01 else sigma * 0.01
        return bachelier_call(F, K, r, sigma_n, T)

    return black76_call(F_shifted, K_shifted, r, sigma, T)


def shifted_lognormal_put(
    F: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    shift: float = 0.0
) -> float:
    """Shifted lognormal put."""
    from finceptquantlib.pricing.analytical.black76 import black76_put

    F_shifted = F + shift
    K_shifted = K + shift

    if F_shifted <= 0 or K_shifted <= 0:
        sigma_n = sigma * abs(F_shifted) if F_shifted != 0 else sigma * 0.01
        return bachelier_put(F, K, r, sigma_n, T)

    return black76_put(F_shifted, K_shifted, r, sigma, T)


# =============================================================================
# Standardized API
# =============================================================================

@dataclass
class BachelierResult:
    """Comprehensive Bachelier pricing result with standardized field names."""
    price: float
    delta: float
    gamma: float
    vega: float

    # Input parameters
    forward: float = 0.0
    strike: float = 0.0
    risk_free_rate: float = 0.0
    normal_volatility: float = 0.0
    time_to_maturity: float = 0.0
    option_type: Optional[OptionType] = None


def bachelier_option_price(
    forward: float,
    strike: float,
    risk_free_rate: float,
    normal_volatility: float,
    time_to_maturity: float,
    option_type: OptionType,
) -> float:
    """
    Bachelier option price with standardized parameters.

    Args:
        forward: Forward price
        strike: Strike price
        risk_free_rate: Risk-free rate (decimal)
        normal_volatility: Normal volatility (in price units)
        time_to_maturity: Time to maturity in years
        option_type: CALL or PUT

    Returns:
        Option price
    """
    return bachelier_price(forward, strike, risk_free_rate, normal_volatility, time_to_maturity, option_type)


def bachelier_option_greeks(
    forward: float,
    strike: float,
    risk_free_rate: float,
    normal_volatility: float,
    time_to_maturity: float,
    option_type: OptionType,
) -> BachelierResult:
    """
    Compute Bachelier price and Greeks with standardized parameters.

    Args:
        forward: Forward price
        strike: Strike price
        risk_free_rate: Risk-free rate (decimal)
        normal_volatility: Normal volatility (in price units)
        time_to_maturity: Time to maturity in years
        option_type: CALL or PUT

    Returns:
        BachelierResult with price and all Greeks
    """
    price = bachelier_price(forward, strike, risk_free_rate, normal_volatility, time_to_maturity, option_type)
    greeks = bachelier_greeks(forward, strike, risk_free_rate, normal_volatility, time_to_maturity, option_type)

    return BachelierResult(
        price=price,
        delta=greeks.delta,
        gamma=greeks.gamma,
        vega=greeks.vega,
        forward=forward,
        strike=strike,
        risk_free_rate=risk_free_rate,
        normal_volatility=normal_volatility,
        time_to_maturity=time_to_maturity,
        option_type=option_type,
    )


def bachelier_option_implied_volatility(
    market_price: float,
    forward: float,
    strike: float,
    risk_free_rate: float,
    time_to_maturity: float,
    option_type: OptionType,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """
    Bachelier implied normal volatility with standardized parameters.

    Args:
        market_price: Observed market price
        forward: Forward price
        strike: Strike price
        risk_free_rate: Risk-free rate (decimal)
        time_to_maturity: Time to maturity in years
        option_type: CALL or PUT
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations

    Returns:
        Implied normal volatility (in price units)
    """
    return bachelier_implied_vol(
        market_price, forward, strike, risk_free_rate,
        time_to_maturity, option_type, tolerance, max_iterations
    )
